from pytorch_grad_cam import ScoreCAM
import json
import os
import re
import sys
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPTokenizerFast

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finegrained.params import build_parser as build_base_parser, resolve_dataset_paths
from finegrained.load_dataset import get_finegrained_dataset_cls
from finegrained.clip_finegrained_baseline import ClipFinegrainedBaseline, _HFCLIPWrapper

DEFAULT_MASK_DIR = "finegrained/mask"


class _ForwardCapture:
    """Capture one module output tensor from the latest forward."""

    def __init__(self, module: torch.nn.Module):
        self.tensor: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.tensor = output

    def clear(self) -> None:
        self.tensor = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class ClipUnlearnFinegrained(ClipFinegrainedBaseline):
    def __init__(self, args):
        super().__init__(args)

    def _normalize_name(self, name: str) -> str:
        return name.strip().replace(" ", "_")

    def _read_txt_image_list(self, txt_path: str) -> List[str]:
        items: List[str] = []
        if not os.path.exists(txt_path):
            return items
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(os.path.basename(line))
        return items

    def _unique_keep_order(self, items: Sequence[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _load_images_from_df_lists(
        self,
        df_root: str,
        split: str,
        item_folder: str,
        class_names: Sequence[str],
    ) -> List[str]:
        images: List[str] = []
        for class_name in class_names:
            txt_path = os.path.join(df_root, split, "Df", item_folder, f"{self._normalize_name(class_name)}.txt")
            images.extend(self._read_txt_image_list(txt_path))
        return self._unique_keep_order(images)

    def _build_single_train_dataset(self, transform):
        dataset_cls = get_finegrained_dataset_cls(self.args.dataset)
        forget_class_names = [self._normalize_name(x) for x in self.args.forget_classes]
        train_files = self._load_images_from_df_lists(
            df_root=self.args.df_root,
            split=self.args.train_split,
            item_folder=self.args.train_item_folder,
            class_names=forget_class_names,
        )
        return dataset_cls(
            label_names=self.label_names,
            annotation_file=self.args.train_annotation_file,
            image_root=self.args.train_image_root,
            split=self.args.train_split,
            transform=transform,
            return_meta=self.args.return_meta,
            selected_files=train_files,
            forget_class_names=forget_class_names,
        )

    def _get_model_device(self, model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _resolve_block_stop(self, n_blocks: int, layer: int) -> int:
        """
        Convert layer hyperparameter to "run first k blocks" count.
        - layer > 0: 1-based index (1..N)
        - layer < 0: python-style from tail (-1..-N)
        Returns:
            k in [1, n_blocks]
        """
        if n_blocks <= 0:
            raise ValueError("No transformer blocks found in visual encoder.")
        if layer == 0:
            raise ValueError("`layer` must not be 0. Use 1..N or -1..-N.")
        if layer > 0:
            k = layer
        else:
            k = n_blocks + layer + 1
        if k < 1 or k > n_blocks:
            raise ValueError(f"`layer` out of range: got {layer}, valid is 1..{n_blocks} or -1..{-n_blocks}.")
        return k

    def _encode_image_patches(self, model, images: torch.Tensor, layer: int) -> Tuple[torch.Tensor, int]:
        clip_model = getattr(model, "clip_model", None)
        if clip_model is None:
            raise AttributeError("Cannot locate visual encoder on model.")
        vision_model = getattr(clip_model, "vision_model", None)
        if vision_model is None:
            raise AttributeError("HF clip_model has no vision_model.")

        x = vision_model.embeddings(images)
        if hasattr(vision_model, "pre_layrnorm"):
            x = vision_model.pre_layrnorm(x)

        layers = getattr(getattr(vision_model, "encoder", None), "layers", None)
        if layers is None or len(layers) == 0:
            raise AttributeError("Cannot find vision_model.encoder.layers for HF CLIP.")
        stop_k = self._resolve_block_stop(len(layers), layer)
        for blk in layers[:stop_k]:
            x = blk(
                x,
                attention_mask=None,
                causal_attention_mask=None,
            )[0]

        patch_feats = x[:, 1:, :]
        n_patches = int(patch_feats.shape[1])
        grid = int(round(n_patches ** 0.5))
        if grid * grid != n_patches:
            raise ValueError(f"Patch token count {n_patches} is not a square number.")
        return patch_feats, grid

    def _project_patch_feats(self, model, patch_feats: torch.Tensor) -> torch.Tensor:
        clip_model = getattr(model, "clip_model", None)
        if clip_model is not None and hasattr(clip_model, "visual_projection"):
            return clip_model.visual_projection(patch_feats)
        raise AttributeError("HF CLIP model has no visual_projection.")

    def _autocast_context(self, device: torch.device):
        if device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", enabled=getattr(self.args, "amp", True))
        return nullcontext()

    def _mask_to_patch_attention(self, mask_tensor: torch.Tensor, grid_size: int) -> torch.Tensor:
        mask_tensor = mask_tensor.unsqueeze(1)
        mask_resized = F.interpolate(mask_tensor, size=(grid_size, grid_size), mode="nearest")
        patch_mask = (mask_resized.squeeze(1) > 0.5).float()
        return patch_mask.flatten(1)

    def _masked_max_mean_pool(
        self,
        patch_feats: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = []
        for i in range(patch_feats.size(0)):
            m = patch_mask[i] > 0.5
            feats = patch_feats[i][m]
            if feats.numel() == 0:
                feats = patch_feats[i]
            max_pool = feats.max(dim=0).values
            mean_pool = feats.mean(dim=0)
            pooled.append(0.5 * (max_pool + mean_pool))
        return torch.stack(pooled, dim=0)

    def _build_class_name_to_idx(self, class_names: Sequence[str]) -> Dict[str, int]:
        return {self._normalize_name(name): idx for idx, name in enumerate(class_names)}

    def _extract_image_stem(self, file_name: str) -> str:
        no_ext = os.path.splitext(file_name)[0]
        m = re.search(r"(\d{12})", no_ext)
        if m:
            return m.group(1)
        if "__" in no_ext:
            return no_ext.split("__", 1)[0]
        if "_" in no_ext:
            return no_ext.split("_", 1)[0]
        return no_ext

    def _extract_score(self, file_name: str) -> float:
        no_ext = os.path.splitext(file_name)[0]
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", no_ext)
        score = 0.0
        for tok in nums:
            try:
                val = float(tok)
            except ValueError:
                continue
            if 0.0 <= val <= 1.0:
                score = val
        return score

    def _extract_class_idx(
        self,
        rel_path: str,
        class_name_to_idx: Dict[str, int],
    ) -> Optional[int]:
        norm_rel = rel_path.lower().replace("-", "_")
        for name, idx in sorted(class_name_to_idx.items(), key=lambda x: len(x[0]), reverse=True):
            pat = rf"(^|[^a-z0-9]){re.escape(name)}([^a-z0-9]|$)"
            if re.search(pat, norm_rel):
                return idx
        return None

    def _build_mask_index(
        self,
        mask_root: str,  # 预处理好的遗忘 mask 根目录
        mask_suffix: str,  # 遗忘 mask 文件后缀名，例如 ".png"
        class_name_to_idx: Dict[str, int],  # class name 到 class idx 的映射，用于从 mask 文件路径中推断 mask 对应的类别索引
    ) -> Dict[str, List[Dict[str, object]]]:
        idx: Dict[str, List[Dict[str, object]]] = {}
        for root, _, files in os.walk(mask_root):
            for file_name in files:
                if not file_name.endswith(mask_suffix):
                    continue
                path = os.path.join(root, file_name)
                rel_path = os.path.relpath(path, mask_root)
                stem = self._extract_image_stem(file_name)
                class_idx = self._extract_class_idx(rel_path, class_name_to_idx)
                score = self._extract_score(file_name)
                idx.setdefault(stem, []).append(
                    {"path": path, "class_idx": class_idx, "score": score}
                )
        return idx  # mask 图片的 image path、class idx 和 score 信息，构建一个以 image stem 为键，包含 mask 信息列表为值的索引字典。

    def _pick_mask_entry(
        self,
        image_path: str,
        mask_index: Dict[str, List[Dict[str, object]]],
        target_indices: Sequence[int],
        allow_any_fallback: bool,
    ) -> Optional[Dict[str, object]]:
        """
        为单张图像挑选一个最合适的 SAM mask 条目。
        Args:
            image_path: 原图路径，用其文件名 stem 在 `mask_index` 中查候选。
            mask_index: `_build_mask_index` 生成的索引，键为 image stem，值为 entry 列表。
            target_indices: 当前分支允许的类别索引集合（遗忘分支或保留分支候选类）。
            allow_any_fallback: 是否允许“无类别匹配时退化到任意类别最高分”。

        Returns:
            选中的 entry（包含 `path/class_idx/score`）为score 最高的那个类，或 None。
        """
        stem = os.path.splitext(os.path.basename(image_path))[0]
        entries = mask_index.get(stem, [])
        if not entries:
            return None

        target_set = set(int(x) for x in target_indices)
        candidates = [e for e in entries if e.get("class_idx") in target_set]
        if candidates:
            return max(candidates, key=lambda x: float(x.get("score", 0.0)))

        if allow_any_fallback:
            return max(entries, key=lambda x: float(x.get("score", 0.0)))
        return None

    def _load_mask_from_entry(self, entry: Dict[str, object], target_size: int) -> torch.Tensor:
        mask_path = str(entry["path"])
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((target_size, target_size), resample=Image.NEAREST)
        return torch.from_numpy(np.array(mask)).float() / 255.0

    def _load_forget_masks(
        self,
        image_paths: Sequence[str],
        mask_index: Dict[str, List[Dict[str, object]]],
        target_size: int,
        forget_indices: Sequence[int],
    ) -> torch.Tensor:
        masks = []
        for path in image_paths:
            # Forget branch: each image must have a forget-class mask.
            entry = self._pick_mask_entry(
                image_path=path,
                mask_index=mask_index,
                target_indices=forget_indices,
                allow_any_fallback=False,
            )
            if entry is None:
                stem = os.path.splitext(os.path.basename(path))[0]
                raise FileNotFoundError(f"Forget-class SAM3 mask not found for image stem: {stem}")
            masks.append(self._load_mask_from_entry(entry, target_size))
        return torch.stack(masks, dim=0)

    def _select_retain_mask_targets(
        self,
        image_paths: Sequence[str],  # 当前 batch 每张图的路径
        retain_labels: torch.Tensor,  # 当前 batch 的保留标签，shape (B, num_classes)，值为0/1表示每个类是否为保留候选
        target_size: int,  # 训练图像尺寸（如 224），用于把缓存 mask resize 到 patch 对齐尺寸
        class_names: Sequence[str],  # 类别名列表（索引到文本 prompt，生成 retain text）
        # 预处理缓存 entries，键是 image_stem，值里至少有 class_idx/mask_bits/shape
        retain_cache_entries: Dict[str, Dict[str, object]],
        retain_mask_tensor_cache: Dict[Tuple[str, int], torch.Tensor],  # 运行时解码缓存，避免每次重复 unpack 同一张图 mask
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        def _decode_cached_mask(stem: str, entry: Dict[str, object], out_size: int) -> torch.Tensor:
            cache_key = (stem, out_size)
            cached = retain_mask_tensor_cache.get(cache_key)
            if cached is not None:
                return cached

            shape = entry.get("shape")
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ValueError(f"Invalid retain cache shape for stem={stem}: {shape}")
            h, w = int(shape[0]), int(shape[1])
            bits = entry.get("mask_bits")
            if bits is None:
                raise ValueError(f"Missing mask_bits in retain cache for stem={stem}")
            if isinstance(bits, torch.Tensor):
                bit_arr = bits.detach().cpu().numpy().astype(np.uint8)
            else:
                bit_arr = np.asarray(bits, dtype=np.uint8)

            flat = np.unpackbits(bit_arr, bitorder="little")
            flat = flat[: h * w]
            mask = torch.from_numpy(flat.reshape(h, w).astype(np.float32))
            if h != out_size or w != out_size:
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask, size=(out_size, out_size), mode="nearest")
                mask = mask.squeeze(0).squeeze(0)
            mask = (mask > 0.5).float().cpu()
            retain_mask_tensor_cache[cache_key] = mask
            return mask

        valid_indices: List[int] = []
        masks: List[torch.Tensor] = []
        texts: List[str] = []
        for i, path in enumerate(image_paths):
            # Retain candidates are exactly non-forget labels that co-occur in this same image.
            candidate_indices = torch.nonzero(retain_labels[i] > 0.5).flatten().tolist()
            if not candidate_indices:
                continue

            stem = os.path.splitext(os.path.basename(path))[0]
            entry = retain_cache_entries.get(stem)
            if entry is None:
                continue
            cls_idx = int(entry.get("class_idx", -1))
            candidate_set = set(int(x) for x in candidate_indices)
            if cls_idx not in candidate_set:
                continue
            best_mask = _decode_cached_mask(stem=stem, entry=entry, out_size=target_size)

            valid_indices.append(i)
            masks.append(best_mask.float())
            texts.append(f"a photo of {class_names[cls_idx].replace('_', ' ')}")

        if not valid_indices:
            empty_idx = torch.empty((0,), dtype=torch.long)
            empty_mask = torch.empty((0, target_size, target_size), dtype=torch.float32)
            return empty_idx, empty_mask, []

        return (
            torch.tensor(valid_indices, dtype=torch.long),  # 表示在当前 batch 中哪些样本有效（有可用 retain cache 且类匹配）
            torch.stack(masks, dim=0),  # 形状 [N, target_size, target_size]，float32，对应每个有效样本的 retain mask tensor
            texts,  # 长度 N，每个元素是该样本选中的保留类文本, 如 "a photo of dog"
        )

    def _build_text_pool(self, class_names: Sequence[str], retain_indices: Sequence[int]) -> List[str]:
        return [f"a photo of {class_names[idx].replace('_', ' ')}" for idx in retain_indices]

    def _get_gradcam_layer(self, model, layer: int) -> torch.nn.Module:
        clip_model = getattr(model, "clip_model", None)
        if clip_model is None or not hasattr(clip_model, "vision_model"):
            raise AttributeError("Cannot find visual tower for Grad-CAM.")
        layers = getattr(getattr(clip_model.vision_model, "encoder", None), "layers", None)
        if layers is None or len(layers) == 0:
            raise AttributeError("Cannot find vision_model.encoder.layers for Grad-CAM.")
        block_idx = self._resolve_block_stop(len(layers), layer) - 1
        target_block = layers[block_idx]
        if not hasattr(target_block, "layer_norm1"):
            raise AttributeError("HF CLIP Grad-CAM target block has no layer_norm1.")
        return target_block.layer_norm1

    def _tokens_without_cls(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Unexpected token tensor rank: {x.ndim}, expected 3.")
        if x.shape[1] < 2:
            raise ValueError(f"Unexpected token length: {x.shape[1]}, expected >= 2.")
        return x[:, 1:, :]

    def _gradcam_positive_patch_selection(
        self,
        token_acts: torch.Tensor,
        token_grads: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        acts = self._tokens_without_cls(token_acts)
        grads = self._tokens_without_cls(token_grads)
        # Grad-CAM token importance: mean gradient over tokens as channel weights.
        weights = grads.mean(dim=1, keepdim=True)
        cam_scores = (acts * weights).sum(dim=-1)
        pos_mask = (cam_scores > 0).float()

        # If no positive token exists, keep the single most contributing token.
        empty_mask = pos_mask.sum(dim=1) < 0.5
        if empty_mask.any():
            top_idx = cam_scores.argmax(dim=1)
            pos_mask[empty_mask] = 0.0
            pos_mask[empty_mask, top_idx[empty_mask]] = 1.0
        return acts, pos_mask

    def _select_retain_targets_from_cache(
        self,
        image_paths: Sequence[str],
        retain_labels: torch.Tensor,
        class_names: Sequence[str],
        retain_cache_entries: Dict[str, Dict[str, object]],
    ) -> Tuple[torch.Tensor, List[str]]:
        valid_indices: List[int] = []
        texts: List[str] = []
        for i, path in enumerate(image_paths):
            candidate_indices = torch.nonzero(retain_labels[i] > 0.5).flatten().tolist()
            if not candidate_indices:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            entry = retain_cache_entries.get(stem)
            if entry is None:
                continue
            cls_idx = int(entry.get("class_idx", -1))
            if cls_idx not in set(int(x) for x in candidate_indices):
                continue
            valid_indices.append(i)
            texts.append(f"a photo of {class_names[cls_idx].replace('_', ' ')}")
        if not valid_indices:
            return torch.empty((0,), dtype=torch.long), []
        return torch.tensor(valid_indices, dtype=torch.long), texts

    def _build_optimizer(self, model, backend: str):
        if backend != "hf_transformers" or not hasattr(model, "clip_model"):
            return torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        clip_model = model.clip_model
        vision_lr = self.args.vision_lr if self.args.vision_lr > 0 else self.args.lr
        text_lr = self.args.text_lr if self.args.text_lr > 0 else self.args.lr
        proj_lr = self.args.proj_lr if self.args.proj_lr > 0 else self.args.lr
        logit_scale_lr = float(self.args.logit_scale_lr)

        groups = []
        assigned = set()

        def _group_from_params(params, lr_value: float):
            picked = []
            for p in params:
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in assigned:
                    continue
                assigned.add(pid)
                picked.append(p)
            if picked:
                groups.append({"params": picked, "lr": lr_value, "weight_decay": self.args.weight_decay})

        _group_from_params(clip_model.vision_model.parameters(), vision_lr)
        _group_from_params(clip_model.text_model.parameters(), text_lr)
        _group_from_params(clip_model.visual_projection.parameters(), proj_lr)
        if hasattr(clip_model, "text_projection") and isinstance(clip_model.text_projection, torch.nn.Module):
            _group_from_params(clip_model.text_projection.parameters(), proj_lr)

        other_params = []
        for name, p in clip_model.named_parameters():
            if name == "logit_scale":
                continue
            pid = id(p)
            if not p.requires_grad or pid in assigned:
                continue
            assigned.add(pid)
            other_params.append(p)
        if other_params:
            groups.append({"params": other_params, "lr": self.args.lr, "weight_decay": self.args.weight_decay})

        if hasattr(clip_model, "logit_scale"):
            if logit_scale_lr > 0.0:
                clip_model.logit_scale.requires_grad_(True)
                groups.append({"params": [clip_model.logit_scale], "lr": logit_scale_lr, "weight_decay": 0.0})
            else:
                clip_model.logit_scale.requires_grad_(False)

        if not groups:
            raise RuntimeError("No trainable parameters found to build optimizer.")
        return torch.optim.AdamW(groups)

    def train_one_epoch(
        self,
        model,
        tokenizer_fn,
        train_loader,
        mask_index: Dict[str, List[Dict[str, object]]],
        retain_cache_entries: Dict[str, Dict[str, object]],
        retain_mask_tensor_cache: Dict[Tuple[str, int], torch.Tensor],
        class_names: Sequence[str],
        forget_indices: Sequence[int],
        retain_indices: Sequence[int],
        optimizer,
        scaler,
        lambda_rtf: float,
        # lambda_syn: float,
        lambda_keep: float,
        lambda_ce: float,
        sample_k: int,
        layer: int,
        epoch_idx: int,
        max_epoch: int,
        log_interval: int,
    ) -> Dict[str, float]:
        device = self._get_model_device(model)
        iters_per_epoch = len(train_loader)
        model.train()
        train_iter = iter(train_loader)
        with torch.no_grad():
            text_pool_features = self._encode_text_features(
                model,
                self._build_text_pool(class_names, retain_indices),
                tokenizer_fn,
                device,
            )

        running = {"rtf": 0.0, "keep": 0.0, "ce": 0.0, "tot": 0.0}
        for it in range(iters_per_epoch):
            train_s = next(train_iter)

            img_df = train_s["image"].to(device, non_blocking=True)
            df_paths = train_s["image_path"]

            txt_df = self._labels_to_texts(train_s["forget_label"], class_names, forget_indices)
            df_tokens = tokenizer_fn(txt_df).to(device)
            # Build retain supervision from the same image batch:
            # one retain class + its SAM mask per image (if available).
            retain_idx_cpu, retain_masks_cpu, retain_texts = self._select_retain_mask_targets(
                image_paths=df_paths,
                retain_labels=train_s["retain_label"],
                target_size=img_df.shape[-1],
                class_names=class_names,
                retain_cache_entries=retain_cache_entries,
                retain_mask_tensor_cache=retain_mask_tensor_cache,
            )

            optimizer.zero_grad(set_to_none=True)
            with self._autocast_context(device):
                sam3_masks = self._load_forget_masks(
                    image_paths=df_paths,
                    mask_index=mask_index,
                    target_size=img_df.shape[-1],
                    forget_indices=forget_indices,
                ).to(device, non_blocking=True)
                patch_feats, grid = self._encode_image_patches(model, img_df, layer=layer)
                df_text_feats = model.encode_text(df_tokens)
                df_text_feats = df_text_feats / df_text_feats.norm(dim=-1, keepdim=True)
                # Forget loss uses patch features pooled by forget-class mask on the same image.
                patch_attn = self._mask_to_patch_attention(sam3_masks, grid)
                pooled_patch_feats = self._masked_max_mean_pool(patch_feats, patch_attn)

                pooled_patch_feats = self._project_patch_feats(model, pooled_patch_feats)
                pooled_patch_feats = pooled_patch_feats / pooled_patch_feats.norm(dim=-1, keepdim=True)

                contrast_text_feats = torch.cat([df_text_feats, text_pool_features], dim=0)
                logit_scale = model.logit_scale.exp()
                labels = torch.arange(img_df.size(0), device=device)

                logits_i2t = logit_scale * (pooled_patch_feats @ contrast_text_feats.t())
                loss_rtf_i2t = F.cross_entropy(logits_i2t, labels)

                logits_t2i = logit_scale * (df_text_feats @ pooled_patch_feats.t())
                loss_rtf_t2i = F.cross_entropy(logits_t2i, labels)

                # Negative CE for forget objective.
                loss_rtf = -0.5 * (loss_rtf_i2t + loss_rtf_t2i)

                if retain_idx_cpu.numel() > 0:
                    retain_idx = retain_idx_cpu.to(device, non_blocking=True)
                    retain_masks = retain_masks_cpu.to(device, non_blocking=True)
                    retain_patch_feats = patch_feats.index_select(0, retain_idx)
                    # Retain loss uses retain-class mask pooled patches from the same images.
                    retain_patch_attn = self._mask_to_patch_attention(retain_masks, grid)
                    retain_pooled_patch_feats = self._masked_max_mean_pool(retain_patch_feats, retain_patch_attn)
                    retain_pooled_patch_feats = self._project_patch_feats(model, retain_pooled_patch_feats)
                    retain_pooled_patch_feats = retain_pooled_patch_feats / retain_pooled_patch_feats.norm(
                        dim=-1, keepdim=True
                    )

                    dr_tokens = tokenizer_fn(retain_texts).to(device)
                    dr_text_feats = model.encode_text(dr_tokens)
                    dr_text_feats = dr_text_feats / dr_text_feats.norm(dim=-1, keepdim=True)
                    retain_targets = torch.arange(retain_pooled_patch_feats.size(0), device=device)

                    retain_logits_i2t = logit_scale * (retain_pooled_patch_feats @ dr_text_feats.t())
                    retain_logits_t2i = logit_scale * (dr_text_feats @ retain_pooled_patch_feats.t())
                    # Positive CE for retain objective (same form as forget branch, opposite sign).
                    loss_ce = 0.5 * (
                        F.cross_entropy(retain_logits_i2t, retain_targets)
                        + F.cross_entropy(retain_logits_t2i, retain_targets)
                    )
                else:
                    loss_ce = torch.zeros((), device=device, dtype=loss_rtf.dtype)

                loss = (
                    lambda_rtf * loss_rtf
                    + lambda_ce * loss_ce
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["rtf"] += float(loss_rtf.detach().item())
            running["ce"] += float(loss_ce.detach().item())
            running["tot"] += float(loss.detach().item())

            if (it + 1) % log_interval == 0:
                t = it + 1
                print(
                    f"[Unlearn EP {epoch_idx + 1}/{max_epoch}] it={t}/{iters_per_epoch} "
                    f"rtf={running['rtf']/t:.4f} "
                    # f"keep={running['keep']/t:.4f} ce={running['ce']/t:.4f} "
                    f"total={running['tot']/t:.4f}"
                )
                with open(f'{self.args.output_dir}/train_log.jsonl', 'a', encoding='utf-8') as fw:
                    fw.write(json.dumps({
                        'Epoch': epoch_idx + 1,
                        'Iteration': it + 1,
                        # 'Keep': f'{running["keep"]/t:.4f}',
                        'CE': f'{running["ce"]/t:.4f}',
                        'Total': f'{running["tot"]/t:.4f}'
                    }) + '\n')

        denom = float(max(iters_per_epoch, 1))
        return {
            "loss_rtf": running["rtf"] / denom,
            "loss_ce": running["ce"] / denom,
            "loss_total": running["tot"] / denom,
        }

    def train_one_epoch_gradcam(
        self,
        model,
        tokenizer_fn,
        train_loader,
        mask_index: Dict[str, List[Dict[str, object]]],
        retain_cache_entries: Dict[str, Dict[str, object]],
        retain_mask_tensor_cache: Dict[Tuple[str, int], torch.Tensor],
        class_names: Sequence[str],
        forget_indices: Sequence[int],
        retain_indices: Sequence[int],
        optimizer,
        scaler,
        lambda_rtf: float,
        # lambda_syn: float,
        lambda_keep: float,
        lambda_ce: float,
        sample_k: int,
        layer: int,
        epoch_idx: int,
        max_epoch: int,
        log_interval: int,
    ) -> Dict[str, float]:
        del mask_index, retain_mask_tensor_cache, sample_k
        device = self._get_model_device(model)
        iters_per_epoch = len(train_loader)
        model.train()
        train_iter = iter(train_loader)
        with torch.no_grad():
            text_pool_features = self._encode_text_features(
                model,
                self._build_text_pool(class_names, retain_indices),
                tokenizer_fn,
                device,
            )
        gradcam_layer = self._get_gradcam_layer(model, layer=layer)
        layer_capture = _ForwardCapture(gradcam_layer)

        running = {"rtf": 0.0, "keep": 0.0, "ce": 0.0, "tot": 0.0}
        try:
            for it in range(iters_per_epoch):
                train_s = next(train_iter)
                img_df = train_s["image"].to(device, non_blocking=True)
                df_paths = train_s["image_path"]
                txt_df = self._labels_to_texts(train_s["forget_label"], class_names, forget_indices)

                retain_idx_cpu, retain_texts = self._select_retain_targets_from_cache(
                    image_paths=df_paths,
                    retain_labels=train_s["retain_label"],
                    class_names=class_names,
                    retain_cache_entries=retain_cache_entries,
                )

                optimizer.zero_grad(set_to_none=True)
                with self._autocast_context(device):
                    df_tokens = tokenizer_fn(txt_df).to(device)
                    df_text_feats = model.encode_text(df_tokens)
                    df_text_feats = df_text_feats / df_text_feats.norm(dim=-1, keepdim=True)

                    # Layer-selectable Grad-CAM (clip_example.py style): use target logit gradients
                    # to keep only positively contributing patch tokens for forget targets.
                    layer_capture.clear()
                    df_img_feats = model.encode_image(img_df)
                    df_img_feats = df_img_feats / df_img_feats.norm(dim=-1, keepdim=True)
                    if layer_capture.tensor is None:
                        raise RuntimeError("Grad-CAM forward capture failed on forget branch.")
                    df_score = (df_img_feats * df_text_feats).sum(dim=-1).sum()
                    # We only need first-order grads for CAM mask generation.
                    # Keep graph retention off to reduce memory pressure.
                    df_grads = torch.autograd.grad(df_score, layer_capture.tensor, retain_graph=False)[0]
                    df_patch_feats, df_pos_mask = self._gradcam_positive_patch_selection(layer_capture.tensor, df_grads)

                    pooled_patch_feats = self._masked_max_mean_pool(df_patch_feats, df_pos_mask)
                    pooled_patch_feats = self._project_patch_feats(model, pooled_patch_feats)
                    pooled_patch_feats = pooled_patch_feats / pooled_patch_feats.norm(dim=-1, keepdim=True)

                    contrast_text_feats = torch.cat([df_text_feats, text_pool_features], dim=0)
                    logit_scale = model.logit_scale.exp()
                    labels = torch.arange(img_df.size(0), device=device)
                    logits_i2t = logit_scale * (pooled_patch_feats @ contrast_text_feats.t())
                    logits_t2i = logit_scale * (df_text_feats @ pooled_patch_feats.t())
                    loss_rtf = -0.5 * (
                        F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels)
                    )

                    if lambda_ce > 0.0 and retain_idx_cpu.numel() > 0:
                        retain_idx = retain_idx_cpu.to(device, non_blocking=True)
                        retain_imgs = img_df.index_select(0, retain_idx)
                        dr_tokens = tokenizer_fn(retain_texts).to(device)
                        dr_text_feats = model.encode_text(dr_tokens)
                        dr_text_feats = dr_text_feats / dr_text_feats.norm(dim=-1, keepdim=True)

                        # Retain branch does the opposite: use retain-cache top class as target
                        # and keep patches with positive contribution to that class.
                        layer_capture.clear()
                        dr_img_feats = model.encode_image(retain_imgs)
                        dr_img_feats = dr_img_feats / dr_img_feats.norm(dim=-1, keepdim=True)
                        if layer_capture.tensor is None:
                            raise RuntimeError("Grad-CAM forward capture failed on retain branch.")
                        dr_score = (dr_img_feats * dr_text_feats).sum(dim=-1).sum()
                        dr_grads = torch.autograd.grad(dr_score, layer_capture.tensor, retain_graph=False)[0]
                        dr_patch_feats, dr_pos_mask = self._gradcam_positive_patch_selection(
                            layer_capture.tensor, dr_grads)

                        retain_pooled_patch_feats = self._masked_max_mean_pool(dr_patch_feats, dr_pos_mask)
                        retain_pooled_patch_feats = self._project_patch_feats(model, retain_pooled_patch_feats)
                        retain_pooled_patch_feats = retain_pooled_patch_feats / retain_pooled_patch_feats.norm(
                            dim=-1, keepdim=True
                        )
                        retain_targets = torch.arange(retain_pooled_patch_feats.size(0), device=device)
                        retain_logits_i2t = logit_scale * (retain_pooled_patch_feats @ dr_text_feats.t())
                        retain_logits_t2i = logit_scale * (dr_text_feats @ retain_pooled_patch_feats.t())
                        loss_ce = 0.5 * (
                            F.cross_entropy(retain_logits_i2t, retain_targets)
                            + F.cross_entropy(retain_logits_t2i, retain_targets)
                        )
                    else:
                        loss_ce = torch.zeros((), device=device, dtype=loss_rtf.dtype)

                    loss = (
                        lambda_rtf * loss_rtf
                        + lambda_ce * loss_ce
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running["rtf"] += float(loss_rtf.detach().item())
                running["ce"] += float(loss_ce.detach().item())
                running["tot"] += float(loss.detach().item())

                if (it + 1) % log_interval == 0:
                    t = it + 1
                    print(
                        f"[Unlearn EP {epoch_idx + 1}/{max_epoch}] it={t}/{iters_per_epoch} "
                        f"rtf={running['rtf']/t:.4f} "
                        # f"keep={running['keep']/t:.4f} ce={running['ce']/t:.4f} "
                        f"total={running['tot']/t:.4f}"
                    )
                    with open(f'{self.args.output_dir}/train_log.jsonl', 'a', encoding='utf-8') as fw:
                        fw.write(json.dumps({
                            'Epoch': epoch_idx + 1,
                            'Iteration': it + 1,
                            # 'Keep': f'{running["keep"]/t:.4f}',
                            'CE': f'{running["ce"]/t:.4f}',
                            'Total': f'{running["tot"]/t:.4f}'
                        }) + '\n')
        finally:
            layer_capture.close()

        denom = float(max(iters_per_epoch, 1))
        return {
            "loss_rtf": running["rtf"] / denom,
            "loss_ce": running["ce"] / denom,
            "loss_total": running["tot"] / denom,
        }

    def _evaluate_for_selection(
        self,
        model: torch.nn.Module,
        tokenize_fn,
        image_size: int,
        class_names: Sequence[str],
        retain_topk_indices: Sequence[int],
    ) -> Dict[str, float]:
        device = self._get_model_device(model)
        eval_transform = self._build_eval_transform(image_size)
        forget_loader, retain_loader = self._build_test_dataloaders_from_folders(transform=eval_transform)

        text_features = self._encode_text_features(model, class_names, tokenize_fn, device)
        forget_cache = self._collect_eval_cache(model, forget_loader, text_features, device)
        retain_cache = self._collect_eval_cache(model, retain_loader, text_features, device)
        forget_acc = self._single_accuracy_from_cache(forget_cache)
        retain_acc = self._single_accuracy_from_cache(retain_cache)
        retain_topk_class_acc = self._topk_retain_accuracy_from_cache(
            cache=retain_cache,
            class_names=class_names,
            retain_indices=retain_topk_indices,
        )
        retain_topk_acc = float(np.mean(list(retain_topk_class_acc.values()))) if retain_topk_class_acc else 0.0
        score = (1.0 - forget_acc) + retain_topk_acc + retain_acc
        return {
            "forget_success": 1.0 - float(forget_acc),
            "retain_accuracy": float(retain_acc),
            "retain_topk_accuracy": float(retain_topk_acc),
            "retain_topk_class_accuracy": retain_topk_class_acc,
            "selection_score": float(score)
        }

    def _prepare_training_context(self):
        args = self.args
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, tokenize_fn, image_size, backend = self._load_clip_backend(device)
        model = model.float().to(device)
        if getattr(args, "freeze_text_tower", False):
            clip_model = getattr(model, "clip_model", None)
            if clip_model is not None and hasattr(clip_model, "text_model"):
                for p in clip_model.text_model.parameters():
                    p.requires_grad = False
                if hasattr(clip_model, "text_projection"):
                    text_proj = clip_model.text_projection
                    if isinstance(text_proj, torch.nn.Parameter):
                        text_proj.requires_grad = False
                    elif isinstance(text_proj, torch.nn.Module):
                        for p in text_proj.parameters():
                            p.requires_grad = False

        args.return_meta = True
        train_transform = self._build_eval_transform(image_size)
        df_dataset = self._build_single_train_dataset(transform=train_transform)
        train_loader = DataLoader(
            df_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
        )

        class_names = self._build_class_name_list()
        forget_indices, retain_indices = self._build_forget_retain_indices(args.forget_classes)
        retain_topk_indices = self._compute_topk_retain_classes(df_dataset, forget_indices, args.retain_topk)

        class_name_to_idx = self._build_class_name_to_idx(class_names)
        mask_index = self._build_mask_index(args.sam3_mask_dir, args.sam3_mask_suffix, class_name_to_idx)
        if not mask_index:
            raise FileNotFoundError(f"No mask files found under: {args.sam3_mask_dir}")
        if not args.retain_cache_path:
            raise ValueError("`--retain_cache_path` is required.")
        retain_cache_blob = torch.load(args.retain_cache_path, map_location="cpu")
        retain_cache_entries = retain_cache_blob.get("entries", {})
        if not retain_cache_entries:
            raise ValueError(f"No retain cache entries found in: {args.retain_cache_path}")
        print(
            f"Loaded retain cache: {len(retain_cache_entries)} entries "
            f"from {args.retain_cache_path}"
        )
        retain_mask_tensor_cache: Dict[Tuple[str, int], torch.Tensor] = {}

        optimizer = self._build_optimizer(model, backend=backend)
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"), init_scale=1024)

        return {
            "model": model,
            "tokenize_fn": tokenize_fn,
            "image_size": image_size,
            "backend": backend,
            "train_loader": train_loader,
            "class_names": class_names,
            "forget_indices": forget_indices,
            "retain_indices": retain_indices,
            "retain_topk_indices": retain_topk_indices,
            "mask_index": mask_index,
            "retain_cache_entries": retain_cache_entries,
            "retain_mask_tensor_cache": retain_mask_tensor_cache,
            "optimizer": optimizer,
            "scaler": scaler,
        }

    def _load_model_from_pretrained_dir(self, model_dir: str):
        device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        clip_model = CLIPModel.from_pretrained(model_dir)

        tokenizer_source = getattr(self.args, "clip_path", None) or model_dir
        tokenizer = CLIPTokenizerFast.from_pretrained(tokenizer_source)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        model = _HFCLIPWrapper(clip_model=clip_model, pad_token_id=pad_token_id).float().to(device)

        def tokenize_fn(texts: Sequence[str]) -> torch.Tensor:
            encoded = tokenizer(
                list(texts),
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            return encoded["input_ids"]

        image_size = int(getattr(clip_model.config.vision_config, "image_size", 224))
        model.eval()
        return model, tokenize_fn, image_size

    def _run_train_loop(self, train_epoch_fn) -> None:
        args = self.args
        ctx = self._prepare_training_context()
        model = ctx["model"]
        tokenize_fn = ctx["tokenize_fn"]
        image_size = ctx["image_size"]
        class_names = ctx["class_names"]
        forget_indices = ctx["forget_indices"]
        retain_indices = ctx["retain_indices"]
        retain_topk_indices = ctx["retain_topk_indices"]
        train_loader = ctx["train_loader"]
        mask_index = ctx["mask_index"]
        retain_cache_entries = ctx["retain_cache_entries"]
        retain_mask_tensor_cache = ctx["retain_mask_tensor_cache"]
        optimizer = ctx["optimizer"]
        scaler = ctx["scaler"]

        os.makedirs(args.output_dir, exist_ok=True)

        best_score = -1
        best_epoch_info = None
        for ep in range(args.max_epoch):
            train_epoch_fn(
                model=model,
                tokenizer_fn=tokenize_fn,
                train_loader=train_loader,
                mask_index=mask_index,
                retain_cache_entries=retain_cache_entries,
                retain_mask_tensor_cache=retain_mask_tensor_cache,
                class_names=class_names,
                forget_indices=forget_indices,
                retain_indices=retain_indices,
                optimizer=optimizer,
                scaler=scaler,
                lambda_rtf=args.lambda_rtf,
                lambda_keep=args.lambda_keep,
                lambda_ce=args.lambda_ce,
                sample_k=args.sample_k,
                layer=args.layer,
                epoch_idx=ep,
                max_epoch=args.max_epoch,
                log_interval=args.log_interval,
            )
            if self.args.do_eval:
                eval_metrics = self._evaluate_for_selection(
                    model=model,
                    tokenize_fn=tokenize_fn,
                    image_size=image_size,
                    class_names=class_names,
                    retain_topk_indices=retain_topk_indices,
                )
                print(eval_metrics)
                with open(f'{self.args.output_dir}/eval_log.jsonl', 'a', encoding='utf-8') as fw:
                    fw.write(json.dumps(
                        {
                            'epoch': ep + 1,
                            'metric': eval_metrics
                        }
                    ) + '\n')
                score = eval_metrics['forget_success'] + eval_metrics['retain_accuracy']
                if score > best_score:
                    print(f'New Best Epoch: {ep + 1}')
                    best_score = score
                    model.clip_model.save_pretrained(args.output_dir)
                    best_epoch_info = {
                        'best_epoch': ep + 1,
                        'forget_success': eval_metrics['forget_success'],
                        'coexisting_accuracy': eval_metrics['retain_topk_accuracy'],
                        'retain_accuracy': eval_metrics['retain_accuracy']
                    }
                    with open(f'{self.args.output_dir}/best_epoch.json', 'w', encoding='utf-8') as fw:
                        json.dump(best_epoch_info, fw, ensure_ascii=False, indent=2)

        if not self.args.do_eval:
            # Save last checkpoint
            model.clip_model.save_pretrained(args.output_dir)
            best_epoch_info = {'best_epoch': args.max_epoch}

        best_model, best_tokenize_fn, best_image_size = self._load_model_from_pretrained_dir(args.output_dir)
        final_eval_metrics = self.run_original_eval(
            model=best_model,
            tokenize_fn=best_tokenize_fn,
            image_size=best_image_size,
            retain_topk_indices=retain_topk_indices,
        )

        best_epoch_path = os.path.join(self.args.output_dir, 'best_epoch.json')
        if best_epoch_info is None and os.path.exists(best_epoch_path):
            with open(best_epoch_path, 'r', encoding='utf-8') as fr:
                best_epoch_info = json.load(fr)
        if best_epoch_info is None:
            best_epoch_info = {}
        best_epoch_info['map_c'] = final_eval_metrics['map_c']
        best_epoch_info['map_r'] = final_eval_metrics['map_r']
        with open(best_epoch_path, 'w', encoding='utf-8') as fw:
            json.dump(best_epoch_info, fw, ensure_ascii=False, indent=2)


    def train_ours(self) -> None:
        self._run_train_loop(self.train_one_epoch)

    def train_gradcam(self) -> None:
        self._run_train_loop(self.train_one_epoch_gradcam)

def build_parser():
    parser = build_base_parser()
    parser.description = "Finegrained CLIP unlearning (SAM3-mask guided), COCO split logic."
    parser.set_defaults(method="ours")
    parser.add_argument("--lambda_rtf", type=float, default=3.0)
    parser.add_argument("--lambda_keep", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=3.0)
    parser.add_argument("--sample_k", type=int, default=5)
    parser.add_argument(
        "--layer",
        type=int,
        default=1,
        help="Target layer for `ours`/`gradcam`: 1..N (1-based) or -1..-N (from last block).",
    )
    parser.add_argument("--sam3_mask_dir", type=str, default=DEFAULT_MASK_DIR)
    parser.add_argument("--sam3_mask_suffix", type=str, default=".png")
    parser.add_argument("--retain_cache_path", type=str, default="")
    parser.add_argument("--vision_lr", type=float, default=-1.0)
    parser.add_argument("--text_lr", type=float, default=-1.0)
    parser.add_argument("--proj_lr", type=float, default=-1.0)
    parser.add_argument("--logit_scale_lr", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)
    parser.add_argument("--freeze_text_tower", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.set_defaults(freeze_text_tower=False)
    parser.set_defaults(do_eval=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = resolve_dataset_paths(args)

    clip_unlearn_finegrained = ClipUnlearnFinegrained(args)
    if args.method == "ours":
        clip_unlearn_finegrained.train_ours()
    elif args.method == "gradcam":
        clip_unlearn_finegrained.train_gradcam()
    else:
        raise NotImplementedError(f"Unknown method for unlearn training: {args.method}")


if __name__ == "__main__":
    main()

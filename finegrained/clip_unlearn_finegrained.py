import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finegrained.clip_finegrained_baseline import (
    _build_class_name_list,
    _build_test_dataloaders_from_folders,
    _build_eval_transform,
    _build_forget_retain_indices,
    _compute_topk_retain_classes,
    _encode_text_features,
    _evaluate_single_accuracy,
    _labels_to_texts,
    _load_clip_backend,
    run_original_eval,
)
from finegrained.load_dataset import COCODataSet
from finegrained.params import build_parser as build_base_parser


DEFAULT_MASK_DIR = "finegrained/mask"


def _normalize_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def _read_txt_image_list(txt_path: str) -> List[str]:
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


def _unique_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _load_images_from_df_lists(
    df_root: str,
    split: str,
    item_folder: str,
    class_names: Sequence[str],
) -> List[str]:
    images: List[str] = []
    for class_name in class_names:
        txt_path = os.path.join(df_root, split, "Df", item_folder, f"{_normalize_name(class_name)}.txt")
        images.extend(_read_txt_image_list(txt_path))
    return _unique_keep_order(images)


def _build_single_train_dataset(args, transform):
    forget_class_names = [_normalize_name(x) for x in args.forget_classes]
    train_files = _load_images_from_df_lists(
        df_root=args.df_root,
        split=args.train_split,
        item_folder=args.train_item_folder,
        class_names=forget_class_names,
    )
    return COCODataSet(
        annotation_file=args.train_annotation_file,
        image_root=args.train_image_root,
        split=args.train_split,
        transform=transform,
        return_meta=args.return_meta,
        selected_files=train_files,
        forget_class_names=forget_class_names,
    )


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _encode_image_patches(model, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
    visual = model.visual
    if not hasattr(visual, "conv1"):
        raise AttributeError("Visual encoder does not expose conv1, cannot extract patch features.")
    x = visual.conv1(images)
    grid = x.shape[-1]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat(
        [
            visual.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)

    # Use penultimate transformer-layer tokens: input to the last block.
    blocks = getattr(visual.transformer, "resblocks", None)
    if blocks is not None and len(blocks) > 0:
        if len(blocks) >= 2:
            for blk in blocks[:1]:
                x = blk(x)
        else:
            x = blocks[0](x)
    else:
        # Fallback when block list is not exposed by backend implementation.
        print("Warning: visual transformer blocks not found, using output tokens of ln_pre as patch features.")
        x = visual.transformer(x)

    x = x.permute(1, 0, 2)
    # Keep penultimate-layer patch tokens in visual hidden space.
    patch_feats = x[:, 1:, :]
    return patch_feats, grid 


def _mask_to_patch_attention(mask_tensor: torch.Tensor, grid_size: int) -> torch.Tensor:
    mask_tensor = mask_tensor.unsqueeze(1)
    mask_resized = F.interpolate(mask_tensor, size=(grid_size, grid_size), mode="nearest")
    patch_mask = (mask_resized.squeeze(1) > 0.5).float()
    return patch_mask.flatten(1)


def _masked_max_mean_pool(
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


def _build_class_name_to_idx(class_names: Sequence[str]) -> Dict[str, int]:
    return {_normalize_name(name): idx for idx, name in enumerate(class_names)}


def _extract_image_stem(file_name: str) -> str:
    no_ext = os.path.splitext(file_name)[0]
    m = re.search(r"(\d{12})", no_ext)
    if m:
        return m.group(1)
    if "__" in no_ext:
        return no_ext.split("__", 1)[0]
    if "_" in no_ext:
        return no_ext.split("_", 1)[0]
    return no_ext


def _extract_score(file_name: str) -> float:
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
    mask_root: str, # 预处理好的遗忘 mask 根目录
    mask_suffix: str, # 遗忘 mask 文件后缀名，例如 ".png"
    class_name_to_idx: Dict[str, int], # class name 到 class idx 的映射，用于从 mask 文件路径中推断 mask 对应的类别索引
) -> Dict[str, List[Dict[str, object]]]:
    idx: Dict[str, List[Dict[str, object]]] = {}
    for root, _, files in os.walk(mask_root):
        for file_name in files:
            if not file_name.endswith(mask_suffix):
                continue
            path = os.path.join(root, file_name)
            rel_path = os.path.relpath(path, mask_root)
            stem = _extract_image_stem(file_name)
            class_idx = _extract_class_idx(rel_path, class_name_to_idx)
            score = _extract_score(file_name)
            idx.setdefault(stem, []).append(
                {"path": path, "class_idx": class_idx, "score": score}
            )
    return idx # mask 图片的 image path、class idx 和 score 信息，构建一个以 image stem 为键，包含 mask 信息列表为值的索引字典。


def _pick_mask_entry(
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


def _load_mask_from_entry(entry: Dict[str, object], target_size: int) -> torch.Tensor:
    mask_path = str(entry["path"])
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((target_size, target_size), resample=Image.NEAREST)
    return torch.from_numpy(np.array(mask)).float() / 255.0


def _load_forget_masks(
    image_paths: Sequence[str],
    mask_index: Dict[str, List[Dict[str, object]]],
    target_size: int,
    forget_indices: Sequence[int],
) -> torch.Tensor:
    masks = []
    for path in image_paths:
        # Forget branch: each image must have a forget-class mask.
        entry = _pick_mask_entry(
            image_path=path,
            mask_index=mask_index,
            target_indices=forget_indices,
            allow_any_fallback=False,
        )
        if entry is None:
            stem = os.path.splitext(os.path.basename(path))[0]
            raise FileNotFoundError(f"Forget-class SAM3 mask not found for image stem: {stem}")
        masks.append(_load_mask_from_entry(entry, target_size))
    return torch.stack(masks, dim=0)


def _select_retain_mask_targets(
    image_paths: Sequence[str], #当前 batch 每张图的路径
    retain_labels: torch.Tensor, #当前 batch 的保留标签，shape (B, num_classes)，值为0/1表示每个类是否为保留候选
    target_size: int, #训练图像尺寸（如 224），用于把缓存 mask resize 到 patch 对齐尺寸
    class_names: Sequence[str], #类别名列表（索引到文本 prompt，生成 retain text）
    retain_cache_entries: Dict[str, Dict[str, object]], #预处理缓存 entries，键是 image_stem，值里至少有 class_idx/mask_bits/shape
    retain_mask_tensor_cache: Dict[Tuple[str, int], torch.Tensor], #运行时解码缓存，避免每次重复 unpack 同一张图 mask
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
        torch.tensor(valid_indices, dtype=torch.long), #表示在当前 batch 中哪些样本有效（有可用 retain cache 且类匹配）
        torch.stack(masks, dim=0), #形状 [N, target_size, target_size]，float32，对应每个有效样本的 retain mask tensor
        texts, #长度 N，每个元素是该样本选中的保留类文本, 如 "a photo of dog"
    )


def _build_text_pool(class_names: Sequence[str], retain_indices: Sequence[int]) -> List[str]:
    return [f"a photo of {class_names[idx].replace('_', ' ')}" for idx in retain_indices]


def train_one_epoch(
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
    epoch_idx: int,
    max_epoch: int,
    log_interval: int,
) -> Dict[str, float]:
    device = _get_model_device(model)
    iters_per_epoch = len(train_loader)
    model.train()
    train_iter = iter(train_loader)
    with torch.no_grad():
        text_pool_features = _encode_text_features(
            model,
            _build_text_pool(class_names, retain_indices),
            tokenizer_fn,
            device,
        )

    running = {"rtf": 0.0, "keep": 0.0, "ce": 0.0, "tot": 0.0}
    for it in range(iters_per_epoch):
        train_s = next(train_iter)

        img_df = train_s["image"].to(device, non_blocking=True)
        df_paths = train_s["image_path"]

        txt_df = _labels_to_texts(train_s["forget_label"], class_names, forget_indices)
        df_tokens = tokenizer_fn(txt_df).to(device)
        # Build retain supervision from the same image batch:
        # one retain class + its SAM mask per image (if available).
        retain_idx_cpu, retain_masks_cpu, retain_texts = _select_retain_mask_targets(
            image_paths=df_paths,
            retain_labels=train_s["retain_label"],
            target_size=img_df.shape[-1],
            class_names=class_names,
            retain_cache_entries=retain_cache_entries,
            retain_mask_tensor_cache=retain_mask_tensor_cache,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            sam3_masks = _load_forget_masks(
                image_paths=df_paths,
                mask_index=mask_index,
                target_size=img_df.shape[-1],
                forget_indices=forget_indices,
            ).to(device, non_blocking=True)
            patch_feats, grid = _encode_image_patches(model, img_df)
            df_text_feats = model.encode_text(df_tokens)
            df_text_feats = df_text_feats / df_text_feats.norm(dim=-1, keepdim=True)
            # Forget loss uses patch features pooled by forget-class mask on the same image.
            patch_attn = _mask_to_patch_attention(sam3_masks, grid)
            pooled_patch_feats = _masked_max_mean_pool(patch_feats, patch_attn)

            if getattr(model.visual, "proj", None) is not None:
                pooled_patch_feats = pooled_patch_feats @ model.visual.proj
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
                retain_patch_attn = _mask_to_patch_attention(retain_masks, grid)
                retain_pooled_patch_feats = _masked_max_mean_pool(retain_patch_feats, retain_patch_attn)
                if getattr(model.visual, "proj", None) is not None:
                    retain_pooled_patch_feats = retain_pooled_patch_feats @ model.visual.proj
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

            loss_keep = torch.zeros((), device=device, dtype=loss_rtf.dtype)
            loss = (
                lambda_rtf * loss_rtf
                + lambda_keep * loss_keep
                + lambda_ce * loss_ce
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["rtf"] += float(loss_rtf.detach().item())
        running["keep"] += float(loss_keep.detach().item())
        running["ce"] += float(loss_ce.detach().item())
        running["tot"] += float(loss.detach().item())

        if (it + 1) % log_interval == 0:
            t = it + 1
            print(
                f"[Unlearn EP {epoch_idx + 1}/{max_epoch}] it={t}/{iters_per_epoch} "
                f"rtf={running['rtf']/t:.4f} "
                f"keep={running['keep']/t:.4f} ce={running['ce']/t:.4f} "
                f"total={running['tot']/t:.4f}"
            )

    denom = float(max(iters_per_epoch, 1))
    return {
        "loss_rtf": running["rtf"] / denom,
        "loss_keep": running["keep"] / denom,
        "loss_ce": running["ce"] / denom,
        "loss_total": running["tot"] / denom,
    }


def _evaluate_for_selection(
    model: torch.nn.Module,
    args,
    tokenize_fn,
    image_size: int,
    class_names: Sequence[str],
) -> Dict[str, float]:
    device = _get_model_device(model)
    eval_transform = _build_eval_transform(image_size)
    forget_loader, retain_loader = _build_test_dataloaders_from_folders(args, transform=eval_transform)

    text_features = _encode_text_features(model, class_names, tokenize_fn, device)
    forget_acc = _evaluate_single_accuracy(model, forget_loader, text_features, device)
    retain_acc = _evaluate_single_accuracy(model, retain_loader, text_features, device)
    score = (1.0 - forget_acc) + retain_acc
    return {
        "forget_success": 1.0 - float(forget_acc),
        "retain_accuracy": float(retain_acc),
        "selection_score": float(score),
        "forget_test_size": len(forget_loader.dataset),
        "retain_test_size": len(retain_loader.dataset),
    }


def build_parser():
    parser = build_base_parser()
    parser.description = "Finegrained CLIP unlearning (SAM3-mask guided), COCO split logic."
    parser.add_argument("--lambda_rtf", type=float, default=3.0)
    parser.add_argument("--lambda_keep", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=3.0)
    parser.add_argument("--sample_k", type=int, default=5)
    parser.add_argument("--sam3_mask_dir", type=str, default=DEFAULT_MASK_DIR)
    parser.add_argument("--sam3_mask_suffix", type=str, default=".png")
    parser.add_argument("--retain_cache_path", type=str, default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.train_annotation_file = os.path.join(
        args.coco_root, "annotations", f"instances_{args.train_split}2017.json"
    )
    args.val_annotation_file = os.path.join(
        args.coco_root, "annotations", f"instances_{args.val_split}2017.json"
    )
    args.train_image_root = os.path.join(args.coco_root, f"{args.train_split}2017")
    args.val_image_root = os.path.join(args.coco_root, f"{args.val_split}2017")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenize_fn, image_size, backend = _load_clip_backend(args, device)
    model = model.float().to(device)

    args.return_meta = True
    train_transform = _build_eval_transform(image_size)
    df_dataset = _build_single_train_dataset(args, transform=train_transform)
    train_loader = DataLoader(
        df_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    class_names = _build_class_name_list()
    forget_indices, retain_indices = _build_forget_retain_indices(args.forget_classes)

    class_name_to_idx = _build_class_name_to_idx(class_names)
    mask_index = _build_mask_index(args.sam3_mask_dir, args.sam3_mask_suffix, class_name_to_idx)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "clip_unlearn_finegrained.pt")
    config_path = os.path.join(args.output_dir, "config.json")
    best_score = float("-inf")
    best_epoch = -1
    best_metrics = None

    for ep in range(args.max_epoch):
        train_stats = train_one_epoch(
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
            epoch_idx=ep,
            max_epoch=args.max_epoch,
            log_interval=args.log_interval,
        )

        eval_metrics = _evaluate_for_selection(
            model=model,
            args=args,
            tokenize_fn=tokenize_fn,
            image_size=image_size,
            class_names=class_names,
        )
        cur_score = eval_metrics["selection_score"]
        print(
            f"[Eval EP {ep + 1}/{args.max_epoch}] "
            f"forget_success={eval_metrics['forget_success']:.4f} "
            f"retain_acc={eval_metrics['retain_accuracy']:.4f} "
            f"score={cur_score:.4f}"
        )

        if cur_score > best_score:
            best_score = cur_score
            best_epoch = ep + 1
            best_metrics = eval_metrics
            torch.save(
                {
                    "model": model.state_dict(),
                    "clip_arch": args.clip_arch,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "best_forget_success": eval_metrics["forget_success"],
                    "best_retain_accuracy": eval_metrics["retain_accuracy"],
                },
                ckpt_path,
            )
            save_cfg = dict(vars(args))
            save_cfg.update(
                {
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "best_forget_success": eval_metrics["forget_success"],
                    "best_retain_accuracy": eval_metrics["retain_accuracy"],
                    "selection_metric": "forget_success + retain_accuracy",
                    "train_stats_at_best_epoch": train_stats,
                }
            )
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(save_cfg, f, ensure_ascii=False, indent=2)
            print(f"[Best Updated] epoch={best_epoch} score={best_score:.4f} -> overwrite checkpoint/config")

    if best_epoch < 0:
        raise RuntimeError("No epoch completed successfully, no best checkpoint saved.")

    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model"], strict=True)
    print(
        f"Final best epoch: {best_epoch}, "
        f"forget_success={best_metrics['forget_success']:.4f}, "
        f"retain_acc={best_metrics['retain_accuracy']:.4f}, "
        f"score={best_score:.4f}"
    )

    retain_topk_indices = _compute_topk_retain_classes(df_dataset, forget_indices, args.retain_topk)
    final_eval_metrics = run_original_eval(
        args,
        model=model,
        tokenize_fn=tokenize_fn,
        image_size=image_size,
        backend=backend,
        retain_topk_indices=retain_topk_indices,
    )
    final_cfg = dict(vars(args))
    final_cfg.update(
        {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_forget_success": best_metrics["forget_success"],
            "best_retain_accuracy": best_metrics["retain_accuracy"],
            "selection_metric": "forget_success + retain_accuracy",
            "final_original_eval": final_eval_metrics,
        }
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_cfg, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

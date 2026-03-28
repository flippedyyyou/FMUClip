import copy
import json
import os
import sys
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPModel, CLIPTokenizerFast

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finegrained.params import parse_args
from finegrained.load_dataset import get_finegrained_dataset_cls
from finegrained.flickr30k_labels.flickr30k import LABEL_NAMES as FLICKR30K_LABEL_NAMES
from finegrained.coco_labels.coco import LABEL_NAMES as COCO_LABEL_NAMES


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


class _HFCLIPWrapper(torch.nn.Module):
    def __init__(self, clip_model: CLIPModel, pad_token_id: int):
        super().__init__()
        self.clip_model = clip_model
        self.pad_token_id = int(pad_token_id)

    @property
    def logit_scale(self):
        return self.clip_model.logit_scale

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.clip_model.get_image_features(pixel_values=pixel_values)

    def encode_text(self, input_ids: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(input_ids, dict):
            token_ids = input_ids["input_ids"]
            attention_mask = input_ids.get("attention_mask")
            if attention_mask is None:
                attention_mask = (token_ids != self.pad_token_id).long()
        else:
            token_ids = input_ids
            attention_mask = (token_ids != self.pad_token_id).long()
        return self.clip_model.get_text_features(
            input_ids=token_ids,
            attention_mask=attention_mask,
        )


class _FolderClassEvalDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], num_classes: int, transform):
        self.samples = list(samples)
        self.num_classes = int(num_classes)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, class_idx = self.samples[index]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        image = self.transform(img)

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[class_idx] = 1.0
        return {
            "image": image,
            "label": label,
            "eval_class_idx": torch.tensor(class_idx, dtype=torch.long),
            "image_path": image_path,
        }


class ClipFinegrainedBaseline:
    def __init__(self, args):
        self.args = args
        if args.dataset == 'coco2017_instances':
            self.label_names = COCO_LABEL_NAMES
        elif args.dataset == 'flickr30k_entities':
            self.label_names = FLICKR30K_LABEL_NAMES
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    def _load_clip_backend(self, device: torch.device):
        model_name = getattr(self.args, "clip_path", None)
        clip_model = CLIPModel.from_pretrained(model_name)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        model = _HFCLIPWrapper(clip_model=clip_model, pad_token_id=pad_token_id).to(device)

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
        return model, tokenize_fn, image_size, "hf_transformers"

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

    def _best_epoch_json_path(self) -> str:
        return os.path.join(self.args.output_dir, "best_epoch.json")

    def _write_best_epoch_info(self, payload: Dict[str, object]) -> None:
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(self._best_epoch_json_path(), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _merge_final_eval_into_best_epoch(
        self,
        final_eval_metrics: Dict[str, object],
        base_info: Dict[str, object] = None,
    ) -> Dict[str, object]:
        best_epoch_info = dict(base_info or {})
        best_epoch_path = self._best_epoch_json_path()
        if not best_epoch_info and os.path.exists(best_epoch_path):
            with open(best_epoch_path, "r", encoding="utf-8") as handle:
                best_epoch_info = json.load(handle)
        best_epoch_info["map_c"] = final_eval_metrics["map_c"]
        best_epoch_info["map_r"] = final_eval_metrics["map_r"]
        self._write_best_epoch_info(best_epoch_info)
        return best_epoch_info

    def _build_eval_transform(self, image_size: int):
        return transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )

    def _build_class_name_list(self) -> List[str]:
        return [name for _, name in sorted(self.label_names.items())]

    def _resolve_test_images_root(self) -> str:
        explicit = getattr(self.args, "test_images_root", None)
        if explicit:
            return explicit
        return os.path.join(self.args.df_root, self.args.val_split, "test_images")

    def _build_test_dataloaders_from_folders(
        self,
        transform,
        forget_classes: Sequence[str] = None,
        retain_exclude_classes: Sequence[str] = None,
    ):
        class_names = self._build_class_name_list()
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        test_images_root = self._resolve_test_images_root()

        if not os.path.isdir(test_images_root):
            raise FileNotFoundError(f"test_images root not found: {test_images_root}")

        forget_source = forget_classes if forget_classes is not None else self.args.forget_classes
        retain_exclude_source = (
            retain_exclude_classes if retain_exclude_classes is not None else forget_source
        )
        forget_names = {self._normalize_name(x) for x in forget_source}
        retain_exclude_names = {self._normalize_name(x) for x in retain_exclude_source}
        unknown = [n for n in sorted(forget_names) if n not in name_to_idx]
        if unknown:
            raise ValueError(f"Unknown forget classes: {unknown}")
        forget_idx_set = {name_to_idx[n] for n in forget_names}
        unknown_retain_exclude = [n for n in sorted(retain_exclude_names) if n not in name_to_idx]
        if unknown_retain_exclude:
            raise ValueError(f"Unknown retain exclude classes: {unknown_retain_exclude}")
        retain_exclude_idx_set = {name_to_idx[n] for n in retain_exclude_names}

        forget_samples: List[Tuple[str, int]] = []
        retain_samples: List[Tuple[str, int]] = []

        for class_name, class_idx in name_to_idx.items():
            class_dir = os.path.join(test_images_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            image_paths = []
            for fname in sorted(os.listdir(class_dir)):
                fpath = os.path.join(class_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                image_paths.append(fpath)

            if class_idx in forget_idx_set:
                dst = forget_samples
            elif class_idx not in retain_exclude_idx_set:
                dst = retain_samples
            else:
                continue
            dst.extend((p, class_idx) for p in image_paths)

        forget_dataset = _FolderClassEvalDataset(
            samples=forget_samples,
            num_classes=len(class_names),
            transform=transform,
        )
        retain_dataset = _FolderClassEvalDataset(
            samples=retain_samples,
            num_classes=len(class_names),
            transform=transform,
        )
        forget_loader = DataLoader(
            forget_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=False,
        )
        retain_loader = DataLoader(
            retain_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=False,
        )
        return forget_loader, retain_loader

    def _build_forget_retain_indices(self, forget_classes: Sequence[str]) -> Tuple[List[int], List[int]]:
        norm_forget = {name.strip().replace(" ", "_") for name in forget_classes}
        all_classes = self._build_class_name_list()
        name_to_idx = {name: idx for idx, name in enumerate(all_classes)}

        forget_indices = []
        for name in norm_forget:
            if name not in name_to_idx:
                raise ValueError(f"Unknown forget class: {name}")
            forget_indices.append(name_to_idx[name])
        forget_indices = sorted(set(forget_indices))
        retain_indices = [i for i in range(len(all_classes)) if i not in set(forget_indices)]
        return forget_indices, retain_indices

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

    def _build_train_datasets(self, transform):
        dataset_cls = get_finegrained_dataset_cls(self.args.dataset)
        forget_class_names = [self._normalize_name(x) for x in self.args.forget_classes]
        all_class_names = [self._normalize_name(v) for _, v in sorted(self.label_names.items())]
        retain_class_names = [c for c in all_class_names if c not in set(forget_class_names)]

        # Forget set: item3 of forget classes
        forget_files = self._load_images_from_df_lists(
            df_root=self.args.df_root,
            split=self.args.train_split,
            item_folder=self.args.train_item_folder,
            class_names=forget_class_names,
        )
        # Retain set: item1 of non-forget classes
        retain_files = self._load_images_from_df_lists(
            df_root=self.args.df_root,
            split=self.args.train_split,
            item_folder=self.args.retain_item_folder,
            class_names=retain_class_names,
        )

        df_dataset = dataset_cls(
            label_names=self.label_names,
            annotation_file=self.args.train_annotation_file,
            image_root=self.args.train_image_root,
            split=self.args.train_split,
            transform=transform,
            return_meta=self.args.return_meta,
            selected_files=forget_files,
            forget_class_names=forget_class_names,
        )
        dr_dataset = dataset_cls(
            label_names=self.label_names,
            annotation_file=self.args.train_annotation_file,
            image_root=self.args.train_image_root,
            split=self.args.train_split,
            transform=transform,
            return_meta=self.args.return_meta,
            selected_files=retain_files,
            forget_class_names=forget_class_names,
        )
        return df_dataset, dr_dataset

    def _encode_text_features(
        self,
        model,
        class_names: Sequence[str],
        tokenize_fn: Callable[[Sequence[str]], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        text_tokens = tokenize_fn(list(class_names)).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _get_logits_and_feats(
        self,
        model,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        sim_i2t = logit_scale * (image_features @ text_features.t())
        sim_t2i = logit_scale * (text_features @ image_features.t())
        return sim_i2t, sim_t2i, image_features, text_features

    def _get_model_device(self, model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _labels_to_texts(
        self,
        labels: torch.Tensor,
        class_names: Sequence[str],
        prefer_indices: Sequence[int],
    ) -> List[str]:
        texts: List[str] = []
        prefer_set = set(prefer_indices)
        for row in labels:
            idxs = torch.nonzero(row > 0.5).flatten().tolist()
            choice = None
            for idx in idxs:
                if idx in prefer_set:
                    choice = idx
                    break
            if choice is None and idxs:
                choice = idxs[0]
            if choice is None:
                choice = 0
            name = class_names[choice].replace("_", " ")
            texts.append(f"a photo of {name}")
        return texts

    def _normalize_logits_for_export_torch(self, logits: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities for human-readable JSON export.
        return torch.softmax(logits, dim=1)

    def _normalize_logits_for_export_numpy(self, logits: np.ndarray) -> np.ndarray:
        # Stable softmax for export readability (values in [0, 1], sum to 1 per row).
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        denom = np.sum(exp_shifted, axis=1, keepdims=True)
        return exp_shifted / np.clip(denom, a_min=1e-12, a_max=None)

    def _round_sig(self, value: float, sig: int = 4) -> float:
        x = float(value)
        if x == 0.0 or not np.isfinite(x):
            return x
        return float(f"{x:.{sig}g}")

    def _evaluate_single_accuracy(
        self,
        model,
        data_loader,
        text_features: torch.Tensor,
        device: torch.device,
    ) -> float:
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                eval_class_idx = batch.get("eval_class_idx")
                if eval_class_idx is not None:
                    eval_class_idx = eval_class_idx.to(device, non_blocking=True)

                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * (image_features @ text_features.t())
                pred_global = logits.argmax(dim=1)
                if eval_class_idx is not None and (eval_class_idx >= 0).any():
                    valid_mask = eval_class_idx >= 0
                    gt_global = eval_class_idx
                else:
                    valid_mask = labels.sum(dim=1) == 1
                    gt_global = labels.argmax(dim=1)
                if valid_mask.any():
                    batch_correct = (pred_global[valid_mask] == gt_global[valid_mask]).sum().item()
                    correct += int(batch_correct)
                    total += int(valid_mask.sum().item())

        return float(correct / total) if total > 0 else 0.0

    def _dump_topk_results(
        self,
        model,
        data_loader,
        text_features: torch.Tensor,
        class_names: Sequence[str],
        device: torch.device,
        out_path: str,
        topk: int = 5,
    ) -> None:
        model.eval()

        rows = []
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                image_paths = batch.get("image_path")
                eval_class_idx = batch.get("eval_class_idx")
                if eval_class_idx is not None:
                    eval_class_idx = eval_class_idx.to(device, non_blocking=True)

                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * (image_features @ text_features.t())
                export_scores = self._normalize_logits_for_export_torch(logits)

                if eval_class_idx is not None and (eval_class_idx >= 0).any():
                    valid_mask = eval_class_idx >= 0
                    gt_global = eval_class_idx
                else:
                    valid_mask = labels.sum(dim=1) == 1
                    gt_global = labels.argmax(dim=1)
                if not valid_mask.any():
                    continue

                k = min(topk, logits.size(1))
                topk_scores, topk_indices = export_scores.topk(k, dim=1)
                pred_global = logits.argmax(dim=1)

                for i in range(images.size(0)):
                    if not valid_mask[i]:
                        continue
                    image_path = None
                    if image_paths is not None:
                        image_path = image_paths[i]
                    pred_idx = int(pred_global[i].item())
                    label_idx = int(gt_global[i].item())
                    rows.append(
                        {
                            "image_path": image_path,
                            "label": class_names[label_idx],
                            "pred_name": class_names[pred_idx],
                            "top5": [
                                {
                                    "name": class_names[int(topk_indices[i, j].item())],
                                    "score": float(topk_scores[i, j].item()),
                                }
                                for j in range(k)
                            ],
                        }
                    )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _compute_topk_retain_classes(
        self,
        df_dataset,
        forget_indices: Sequence[int],
        k: int,
    ) -> List[int]:
        if k <= 0:
            return []
        forget_set = set(forget_indices)
        counts = np.zeros(len(self.label_names), dtype=np.int64)
        for target in df_dataset.targets:
            for idx in np.where(target > 0.5)[0]:
                if idx in forget_set:
                    continue
                counts[idx] += 1
        topk = np.argsort(-counts)  # 返回数组值从大到小的索引值
        topk = [int(i) for i in topk if counts[int(i)] > 0 and i not in forget_set]
        for idx in topk[:k]:
            print(f"Retain class: {self.label_names[idx]} with count: {counts[idx]}")
        print()
        return topk[:k]

    def _evaluate_single_class_accuracy(
        model,
        data_loader,
        text_features: torch.Tensor,
        class_index: int,
        device: torch.device,
    ) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                eval_class_idx = batch.get("eval_class_idx")
                if eval_class_idx is not None:
                    eval_class_idx = eval_class_idx.to(device, non_blocking=True)

                if eval_class_idx is not None and (eval_class_idx >= 0).any():
                    mask = eval_class_idx == class_index
                else:
                    mask = (labels.sum(dim=1) == 1) & (labels[:, class_index] > 0.5)
                if not mask.any():
                    continue

                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * (image_features @ text_features.t())
                preds = logits.argmax(dim=1)

                correct += int((preds[mask] == class_index).sum().item())
                total += int(mask.sum().item())
        return float(correct / total) if total > 0 else 0.0

    def _evaluate_topk_retain_accuracy(
        self,
        model,
        data_loader,
        text_features: torch.Tensor,
        class_names: Sequence[str],
        device: torch.device,
        retain_indices: Sequence[int],
    ) -> Dict[str, float]:
        if not retain_indices:
            return {}
        results: Dict[str, float] = {}
        for idx in retain_indices:
            acc = self._evaluate_single_class_accuracy(
                model=model,
                data_loader=data_loader,
                text_features=text_features,
                class_index=idx,
                device=device,
            )
            results[class_names[idx]] = acc
        return results

    def _average_precision_binary(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        positives = int(y_true.sum())
        if positives == 0:
            return float("nan")
        return float(average_precision_score(y_true, y_score))

    def _compute_map_for_indices(
        self,
        gt: np.ndarray,
        scores: np.ndarray,
        indices: Sequence[int],
    ) -> float:
        ap_values: List[float] = []
        for class_idx in indices:
            ap = self._average_precision_binary(gt[:, class_idx].astype(np.int32), scores[:, class_idx])
            if not np.isnan(ap):
                ap_values.append(ap)
        if not ap_values:
            return 0.0
        return float(np.mean(ap_values))

    def _build_val_joint_multilabel_loader(
        self,
        transform,
        forget_indices: Sequence[int],
        retain_topk_indices: Sequence[int],
    ):
        dataset_cls = get_finegrained_dataset_cls(self.args.dataset)
        if not forget_indices or not retain_topk_indices:
            return DataLoader(
                dataset_cls(
                    label_names=self.label_names,
                    annotation_file=self.args.val_annotation_file,
                    image_root=self.args.val_image_root,
                    split=self.args.val_split,
                    transform=transform,
                    return_meta=self.args.return_meta,
                    selected_files=[],
                    forget_class_names=self.args.forget_classes,
                ),
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                drop_last=False,
            )

        full_dataset = dataset_cls(
            label_names=self.label_names,
            annotation_file=self.args.val_annotation_file,
            image_root=self.args.val_image_root,
            split=self.args.val_split,
            transform=transform,
            return_meta=self.args.return_meta,
            forget_class_names=self.args.forget_classes,
        )

        forget_idx_arr = np.array(list(forget_indices), dtype=np.int64)
        topk_idx_arr = np.array(list(retain_topk_indices), dtype=np.int64)
        selected_files: List[str] = []
        max_per_class = int(getattr(self.args, "joint_multilabel_max_per_class", 0))
        cap_indices = sorted(set(int(i) for i in forget_indices) | set(int(i) for i in retain_topk_indices))
        cap_counts = {idx: 0 for idx in cap_indices}

        for file_name, target in zip(full_dataset.file_names, full_dataset.targets):
            has_forget = bool(np.any(target[forget_idx_arr] > 0.5))
            has_topk_retain = bool(np.any(target[topk_idx_arr] > 0.5))
            if not (has_forget and has_topk_retain):
                continue

            if max_per_class > 0 and cap_indices:
                present_cap_indices = [idx for idx in cap_indices if target[idx] > 0.5]
                if not present_cap_indices:
                    continue
                if not any(cap_counts[idx] < max_per_class for idx in present_cap_indices):
                    continue
                selected_files.append(file_name)
                for idx in present_cap_indices:
                    if cap_counts[idx] < max_per_class:
                        cap_counts[idx] += 1
                continue

            selected_files.append(file_name)

        subset_dataset = dataset_cls(
            label_names=self.label_names,
            annotation_file=self.args.val_annotation_file,
            image_root=self.args.val_image_root,
            split=self.args.val_split,
            transform=transform,
            return_meta=self.args.return_meta,
            selected_files=selected_files,
            forget_class_names=self.args.forget_classes,
        )

        return DataLoader(
            subset_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=False,
        )

    def _evaluate_joint_multilabel_ap(
        self,
        model,
        data_loader,
        text_features: torch.Tensor,
        class_names: Sequence[str],
        forget_indices: Sequence[int],
        retain_topk_indices: Sequence[int],
        device: torch.device,
    ) -> Dict[str, object]:
        model.eval()
        all_scores: List[np.ndarray] = []  # 每个元素 [B, C]，最后拼接成 [N, C]
        all_labels: List[np.ndarray] = []  # 每个元素 [B, C]，最后拼接成 [N, C]
        per_class_ap: Dict[int, float] = {}

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * (image_features @ text_features.t())

                all_scores.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        if not all_scores:
            return {
                "eval_size": 0,
                "forget_map": 0.0,
                "forget_class_ap": {},
                "retain_topk_ap": {},
                "other_map": 0.0,
                "other_classes": [],
            }

        scores = np.concatenate(all_scores, axis=0)  # 所有 batch 拼接得到的 all_scores，shape [N, C]，C为数据集的全部候选集类别数
        gt = np.concatenate(all_labels, axis=0)  # 所有 batch 拼接得到的 all_scores，shape [N, C], 0/1 二值标签矩阵
        present_classes = np.where(gt.sum(axis=0) > 0)[0].tolist()  # 只保留“出现过”的类计算 AP，避免某些完全没有正样本的类导致 AP 计算出 NaN

        for class_idx in present_classes:
            ap = self._average_precision_binary(gt[:, class_idx].astype(np.int32), scores[:, class_idx])
            if not np.isnan(ap):
                per_class_ap[int(class_idx)] = float(ap)

        forget_aps = [per_class_ap[idx] for idx in forget_indices if idx in per_class_ap]
        forget_class_ap = {class_names[idx]: per_class_ap[idx]
                           for idx in forget_indices if idx in per_class_ap}  # 最多 F 项
        retain_topk_ap = {  # 最多 K 项
            class_names[idx]: per_class_ap[idx] for idx in retain_topk_indices if idx in per_class_ap
        }

        excluded = set(forget_indices) | set(retain_topk_indices)
        other_indices = [idx for idx in present_classes if idx not in excluded]
        other_map = self._compute_map_for_indices(gt, scores, other_indices)

        return {
            "eval_size": int(gt.shape[0]),
            "forget_map": float(np.mean(forget_aps)) if forget_aps else 0.0,
            "forget_class_ap": forget_class_ap,
            "retain_topk_ap": retain_topk_ap,
            "other_map": other_map,
            "other_classes": [class_names[idx] for idx in other_indices],
        }

    def _dump_joint_multilabel_results(
        self,
        model,
        data_loader,
        text_features: torch.Tensor,
        class_names: Sequence[str],
        device: torch.device,
        out_path: str,
        topk: int = 5,
    ) -> None:
        model.eval()
        rows = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                image_paths = batch.get("image_path")

                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * (image_features @ text_features.t())
                export_scores = self._normalize_logits_for_export_torch(logits)

                k = min(topk, logits.size(1))
                topk_scores, topk_indices = export_scores.topk(k, dim=1)
                pred_global = logits.argmax(dim=1)

                for i in range(images.size(0)):
                    gt_indices = torch.nonzero(labels[i] > 0.5).flatten().tolist()
                    if not gt_indices:
                        continue
                    image_path = image_paths[i] if image_paths is not None else None
                    pred_idx = int(pred_global[i].item())
                    rows.append(
                        {
                            "image_path": image_path,
                            "gt": [class_names[int(idx)] for idx in gt_indices],
                            "pred_name": class_names[pred_idx],
                            "top5": [
                                {
                                    "name": class_names[int(topk_indices[i, j].item())],
                                    "score": float(topk_scores[i, j].item()),
                                }
                                for j in range(k)
                            ],
                        }
                    )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _collect_eval_cache(
        self,
        model,
        data_loader,
        text_features: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, object]:
        model.eval()
        all_logits: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_eval_idx: List[np.ndarray] = []
        all_paths: List[str] = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                eval_class_idx = batch.get("eval_class_idx")
                image_paths = batch.get("image_path")

                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * (image_features @ text_features.t())

                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                if eval_class_idx is None:
                    all_eval_idx.append(np.full((labels.size(0),), -1, dtype=np.int64))
                else:
                    all_eval_idx.append(eval_class_idx.detach().cpu().numpy().astype(np.int64))

                if image_paths is None:
                    all_paths.extend([None] * labels.size(0))
                else:
                    all_paths.extend(list(image_paths))

        if not all_logits:
            num_classes = int(text_features.size(0))
            return {
                "logits": np.zeros((0, num_classes), dtype=np.float32),
                "labels": np.zeros((0, num_classes), dtype=np.float32),
                "eval_idx": np.zeros((0,), dtype=np.int64),
                "image_paths": [],
            }

        return {
            "logits": np.concatenate(all_logits, axis=0),
            "labels": np.concatenate(all_labels, axis=0),
            "eval_idx": np.concatenate(all_eval_idx, axis=0),
            "image_paths": all_paths,
        }

    def _single_accuracy_from_cache(self, cache: Dict[str, object]) -> float:
        logits = cache["logits"]
        labels = cache["labels"]
        eval_idx = cache["eval_idx"]
        if logits.shape[0] == 0:
            return 0.0

        if np.any(eval_idx >= 0):
            valid = eval_idx >= 0
            gt = eval_idx
        else:
            valid = labels.sum(axis=1) == 1
            gt = labels.argmax(axis=1)
        if valid.sum() == 0:
            return 0.0
        pred = logits.argmax(axis=1)
        return float((pred[valid] == gt[valid]).mean())

    def _single_class_accuracy_from_cache(self, cache: Dict[str, object], class_index: int) -> float:
        logits = cache["logits"]
        labels = cache["labels"]
        eval_idx = cache["eval_idx"]
        if logits.shape[0] == 0:
            return 0.0

        if np.any(eval_idx >= 0):
            mask = eval_idx == class_index
        else:
            mask = (labels.sum(axis=1) == 1) & (labels[:, class_index] > 0.5)
        if mask.sum() == 0:
            return 0.0
        pred = logits.argmax(axis=1)
        return float((pred[mask] == class_index).mean())

    def _topk_retain_accuracy_from_cache(
        self,
        cache: Dict[str, object],
        class_names: Sequence[str],
        retain_indices: Sequence[int],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for idx in retain_indices:
            out[class_names[idx]] = self._single_class_accuracy_from_cache(cache, int(idx))
        return out

    def _joint_multilabel_ap_from_cache(
        self,
        cache: Dict[str, object],
        class_names: Sequence[str],
        forget_indices: Sequence[int],
        retain_topk_indices: Sequence[int],
    ) -> Dict[str, object]:
        scores = cache["logits"]
        gt = cache["labels"]
        if scores.shape[0] == 0:
            return {
                "eval_size": 0,
                "forget_map": 0.0,
                "forget_class_ap": {},
                "retain_topk_ap": {},
                "other_map": 0.0,
                "other_classes": [],
            }

        present_classes = np.where(gt.sum(axis=0) > 0)[0].tolist()
        per_class_ap: Dict[int, float] = {}
        for class_idx in present_classes:
            ap = self._average_precision_binary(gt[:, class_idx].astype(np.int32), scores[:, class_idx])
            if not np.isnan(ap):
                per_class_ap[int(class_idx)] = float(ap)

        forget_aps = [per_class_ap[idx] for idx in forget_indices if idx in per_class_ap]
        forget_class_ap = {class_names[idx]: per_class_ap[idx] for idx in forget_indices if idx in per_class_ap}
        retain_topk_ap = {
            class_names[idx]: per_class_ap[idx] for idx in retain_topk_indices if idx in per_class_ap
        }
        excluded = set(forget_indices) | set(retain_topk_indices)
        other_indices = [idx for idx in present_classes if idx not in excluded]
        other_map = self._compute_map_for_indices(gt, scores, other_indices)
        return {
            "eval_size": int(gt.shape[0]),
            "forget_map": float(np.mean(forget_aps)) if forget_aps else 0.0,
            "forget_class_ap": forget_class_ap,
            "retain_topk_ap": retain_topk_ap,
            "other_map": other_map,
            "other_classes": [class_names[idx] for idx in other_indices],
        }

    def _dump_topk_results_from_cache(
        self,
        cache: Dict[str, object],
        class_names: Sequence[str],
        out_path: str,
        topk: int = 5,
    ) -> None:
        logits = cache["logits"]
        labels = cache["labels"]
        eval_idx = cache["eval_idx"]
        image_paths = cache["image_paths"]
        rows: List[Dict[str, object]] = []
        if logits.shape[0] > 0:
            export_scores = self._normalize_logits_for_export_numpy(logits)
            if np.any(eval_idx >= 0):
                valid_mask = eval_idx >= 0
                gt_global = eval_idx
            else:
                valid_mask = labels.sum(axis=1) == 1
                gt_global = labels.argmax(axis=1)

            pred_global = logits.argmax(axis=1)
            k = min(topk, logits.shape[1])
            topk_indices = np.argsort(-export_scores, axis=1)[:, :k]
            topk_scores = np.take_along_axis(export_scores, topk_indices, axis=1)

            for i in range(logits.shape[0]):
                if not bool(valid_mask[i]):
                    continue
                label_idx = int(gt_global[i])
                pred_idx = int(pred_global[i])
                rows.append(
                    {
                        "image_path": image_paths[i] if i < len(image_paths) else None,
                        "label": class_names[label_idx],
                        "pred_name": class_names[pred_idx],
                        "top5": [
                            {"name": class_names[int(topk_indices[i, j])], "score": float(topk_scores[i, j])}
                            for j in range(k)
                        ],
                    }
                )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _dump_joint_multilabel_results_from_cache(
        self,
        cache: Dict[str, object],
        class_names: Sequence[str],
        out_path: str,
        topk: int = 5,
    ) -> None:
        logits = cache["logits"]
        labels = cache["labels"]
        image_paths = cache["image_paths"]
        rows: List[Dict[str, object]] = []
        if logits.shape[0] > 0:
            export_scores = self._normalize_logits_for_export_numpy(logits)
            pred_global = logits.argmax(axis=1)
            k = min(topk, logits.shape[1])
            topk_indices = np.argsort(-export_scores, axis=1)[:, :k]
            topk_scores = np.take_along_axis(export_scores, topk_indices, axis=1)

            for i in range(logits.shape[0]):
                gt_indices = np.where(labels[i] > 0.5)[0].tolist()
                if not gt_indices:
                    continue
                pred_idx = int(pred_global[i])
                rows.append(
                    {
                        "image_path": image_paths[i] if i < len(image_paths) else None,
                        "gt": [class_names[int(idx)] for idx in gt_indices],
                        "pred_name": class_names[pred_idx],
                        "top5": [
                            {"name": class_names[int(topk_indices[i, j])], "score": float(topk_scores[i, j])}
                            for j in range(k)
                        ],
                    }
                )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run_original_eval(
        self,
        model: torch.nn.Module = None,
        tokenize_fn: Callable[[Sequence[str]], torch.Tensor] = None,
        image_size: int = None,
        retain_topk_indices: Sequence[int] = None,
    ) -> Dict[str, object]:
        # assert retain_topk_indices is not None
        device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        if model is None or tokenize_fn is None or image_size is None:
            model, tokenize_fn, image_size, backend = self._load_clip_backend(device)
        eval_transform = self._build_eval_transform(image_size)

        forget_loader, retain_loader = self._build_test_dataloaders_from_folders(transform=eval_transform)
        class_names = self._build_class_name_list()
        forget_indices, retain_indices = self._build_forget_retain_indices(self.args.forget_classes)
        if retain_topk_indices is None and self.args.retain_topk > 0:
            df_dataset, _ = self._build_train_datasets(transform=eval_transform)
            retain_topk_indices = self._compute_topk_retain_classes(
                df_dataset=df_dataset,
                forget_indices=forget_indices,
                k=self.args.retain_topk,
            )
        text_features = self._encode_text_features(model, class_names, tokenize_fn, device)
        joint_multilabel_loader = self._build_val_joint_multilabel_loader(
            transform=eval_transform,
            forget_indices=forget_indices,
            retain_topk_indices=retain_topk_indices,
        )

        print('\n********* Evaluating *********')
        forget_cache = self._collect_eval_cache(
            model=model, data_loader=forget_loader, text_features=text_features, device=device)
        retain_cache = self._collect_eval_cache(
            model=model, data_loader=retain_loader, text_features=text_features, device=device)
        
        
        forget_acc = self._single_accuracy_from_cache(forget_cache)
        retain_acc = self._single_accuracy_from_cache(retain_cache)
        joint_ap_metrics = self._evaluate_joint_multilabel_ap(
            model=model,
            data_loader=joint_multilabel_loader,
            text_features=text_features,
            class_names=class_names,
            forget_indices=forget_indices,
            retain_topk_indices=retain_topk_indices or [],
            device=device,
        )

        os.makedirs(self.args.output_dir, exist_ok=True)
        data_info = {
            "forget_classes": list(self.args.forget_classes),
            "retain_classes": joint_ap_metrics["other_classes"],
            'coexsiting_classes': [class_names[i] for i in retain_topk_indices],
            "forget_test_size": len(forget_loader.dataset),
            "retain_test_size": len(retain_loader.dataset),
            "joint_multilabel_val_size": joint_ap_metrics["eval_size"],
            "joint_multilabel_max_per_class": int(getattr(self.args, "joint_multilabel_max_per_class", 0)),

        }

        topk_acc = self._topk_retain_accuracy_from_cache(
            cache=retain_cache,
            class_names=class_names,
            retain_indices=retain_topk_indices,
        )
        topk_acc_mean = float(np.mean(list(topk_acc.values())))
   
        metrics = {
            "acc_f": self._round_sig(forget_acc, sig=4),
            'acc_c': self._round_sig(topk_acc_mean, sig=4),
            "acc_r": self._round_sig(retain_acc, sig=4),
            'map_c': self._round_sig(float(np.mean(list(joint_ap_metrics["retain_topk_ap"].values()))), sig=4),
            "map_r": self._round_sig(joint_ap_metrics["other_map"], sig=4),
            'acc_topk': topk_acc,
            'ap_topk': joint_ap_metrics["retain_topk_ap"]
        }

        os.makedirs(os.path.join(self.args.output_dir, 'eval_results'), exist_ok=True)

        data_info_path = os.path.join(self.args.output_dir, 'eval_results', "data_info.json")
        with open(data_info_path, 'w', encoding='utf-8') as fw:
            json.dump(data_info, fw, ensure_ascii=False, indent=2)
        metrics_path = os.path.join(self.args.output_dir, 'eval_results', "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        forget_topk_path = os.path.join(self.args.output_dir, 'eval_results', "forget_test_topk.jsonl")
        retain_topk_path = os.path.join(self.args.output_dir, 'eval_results', "retain_test_topk.jsonl")
        joint_map_topk_path = os.path.join(self.args.output_dir, 'eval_results', "joint_multilabel_val_topk.jsonl")

        self._dump_topk_results(
            model=model,
            data_loader=forget_loader,
            text_features=text_features,
            class_names=class_names,
            device=device,
            out_path=forget_topk_path,
        )
        # self._dump_topk_results(
        #     model=model,
        #     data_loader=retain_loader,
        #     text_features=text_features,
        #     class_names=class_names,
        #     device=device,
        #     out_path=retain_topk_path,
        # )
        self._dump_joint_multilabel_results(
            model=model,
            data_loader=joint_multilabel_loader,
            text_features=text_features,
            class_names=class_names,
            device=device,
            out_path=joint_map_topk_path,
        )
        
        print(f'Acc_f: {forget_acc:.4f}\tAcc_c: {topk_acc_mean:.4f}\tAcc_r: {retain_acc:.4f}')
        # print(f"Forget test accuracy: {forget_acc:.4f}")
        # print(f"Retain test accuracy: {retain_acc:.4f}")
        # if retain_topk_indices:
        #     print(f"Retain top-{self.args.retain_topk} accuracy mean: {metrics['retain_topk_accuracy_mean']:.4f}")
        # print(f"Joint multi-label val size: {joint_ap_metrics['eval_size']}")
        # print(f"Forget AP (mean over forget classes): {joint_ap_metrics['forget_map']:.4f}")
        # print(f"Top-{self.args.retain_topk} retain AP: {joint_ap_metrics['retain_topk_ap']}")
        # print(f"Other classes mAP: {joint_ap_metrics['other_map']:.4f}")
        # print(f"Saved metrics to: {metrics_path}")
        return metrics

    def _evaluate_for_selection(
        self,
        model: torch.nn.Module,
        tokenize_fn: Callable[[Sequence[str]], torch.Tensor],
        image_size: int,
        class_names: Sequence[str],
    ) -> Dict[str, float]:
        device = self._get_model_device(model)
        eval_transform = self._build_eval_transform(image_size)
        forget_loader, retain_loader = self._build_test_dataloaders_from_folders(transform=eval_transform)

        text_features = self._encode_text_features(model, class_names, tokenize_fn, device)
        forget_cache = self._collect_eval_cache(
            model=model,
            data_loader=forget_loader,
            text_features=text_features,
            device=device,
        )
        retain_cache = self._collect_eval_cache(
            model=model,
            data_loader=retain_loader,
            text_features=text_features,
            device=device,
        )
        forget_acc = self._single_accuracy_from_cache(forget_cache)
        retain_acc = self._single_accuracy_from_cache(retain_cache)
        return {
            "forget_success": 1.0 - float(forget_acc),
            "retain_accuracy": float(retain_acc),
            "selection_score": float((1.0 - float(forget_acc)) + float(retain_acc)),
            "forget_test_size": len(forget_loader.dataset),
            "retain_test_size": len(retain_loader.dataset),
        }


def main() -> None:
    args = parse_args()
    clip_finegrained_baseline_runner = ClipFinegrainedBaseline(args)
    if args.method == 'original':
        clip_finegrained_baseline_runner.run_original_eval()
        return
    raise NotImplementedError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()

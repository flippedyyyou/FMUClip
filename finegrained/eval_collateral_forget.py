import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finegrained.clip_finegrained_baseline import (
    _average_precision_binary,
    _build_test_dataloaders_from_folders,
    _build_val_joint_multilabel_loader,
    _build_class_name_list,
    _build_eval_transform,
    _build_forget_retain_indices,
    _build_train_datasets,
    _compute_topk_retain_classes,
    _compute_map_for_indices,
    _encode_text_features,
    _load_clip_backend,
)
from finegrained.load_dataset import build_test_dataloaders
from finegrained.params import build_parser as build_base_parser


def build_parser():
    parser = build_base_parser()
    parser.description = "Evaluate collateral forgetting before vs after (co-occurrence top/bottom + original_eval metrics)."
    parser.add_argument(
        "--before_ckpt",
        type=str,
        default="",
        help="Optional. Deprecated for collateral eval; before model is initialized from --clip_arch.",
    )
    parser.add_argument("--after_ckpt", type=str, required=True, help="Path to checkpoint after unlearning.")
    parser.add_argument(
        "--cooccur_eval_k",
        type=int,
        default=5,
        help="Evaluate top-K and bottom-K non-target classes ranked by co-occurrence frequency.",
    )
    parser.add_argument(
        "--group_source",
        type=str,
        default="train_df",
        choices=["train_df", "val_forget"],
        help="Source used to compute co-occurrence classes.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="collateral_eval_metrics.json",
        help="Output json file name under output_dir.",
    )
    return parser


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _extract_state_dict(raw_ckpt: Dict) -> Dict[str, torch.Tensor]:
    if "model" in raw_ckpt and isinstance(raw_ckpt["model"], dict):
        return raw_ckpt["model"]
    if "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], dict):
        return raw_ckpt["state_dict"]
    if all(isinstance(v, torch.Tensor) for v in raw_ckpt.values()):
        return raw_ckpt
    raise ValueError("Checkpoint format not recognized.")


def _safe_torch_load(ckpt_path: str, device: torch.device):
    # PyTorch>=2.6 defaults to weights_only=True; this breaks TorchScript archives.
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only argument.
        return torch.load(ckpt_path, map_location=device)
    except RuntimeError as e:
        if "TorchScript archives" in str(e):
            return torch.jit.load(ckpt_path, map_location=device)
        raise


def _maybe_strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _load_before_model_from_arch(args, device: torch.device):
    model, tokenize_fn, image_size, backend = _load_clip_backend(args, device)
    model = model.float().to(device).eval()
    return model, tokenize_fn, image_size, backend


def _load_after_model_from_ckpt(args, ckpt_path: str, device: torch.device):
    model, tokenize_fn, image_size, backend = _load_clip_backend(args, device)
    model = model.float().to(device)
    if not ckpt_path:
        raise ValueError("`--after_ckpt` is required to load the after-unlearning model.")

    raw = _safe_torch_load(ckpt_path, device)
    if isinstance(raw, torch.jit.ScriptModule):
        print(f"[Info] Loaded TorchScript checkpoint via torch.jit.load: {ckpt_path}")
        raw = raw.to(device).eval()
        return raw, tokenize_fn, image_size, backend
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint format not supported for {ckpt_path}: {type(raw)}")

    state_dict = _extract_state_dict(raw)
    state_dict = _maybe_strip_prefix(state_dict, "module.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys when loading {ckpt_path}: {len(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected keys when loading {ckpt_path}: {len(unexpected)}")
    model.eval()
    return model, tokenize_fn, image_size, backend


class _TaggedDataset(Dataset):
    def __init__(self, dataset: Dataset, tag: str):
        self.dataset = dataset
        self.tag = tag

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = dict(self.dataset[index])
        item["_source_tag"] = self.tag
        return item


def _build_tagged_eval_loader(
    args,
    eval_transform,
    forget_indices: Sequence[int],
    retain_topk_indices: Sequence[int],
):
    forget_loader, retain_loader = _build_test_dataloaders_from_folders(args, transform=eval_transform)
    joint_loader = _build_val_joint_multilabel_loader(
        args=args,
        transform=eval_transform,
        forget_indices=forget_indices,
        retain_topk_indices=retain_topk_indices,
    )

    tagged = ConcatDataset(
        [
            _TaggedDataset(forget_loader.dataset, "forget"),
            _TaggedDataset(retain_loader.dataset, "retain"),
            _TaggedDataset(joint_loader.dataset, "joint"),
        ]
    )
    merged_loader = DataLoader(
        tagged,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    return merged_loader, forget_loader, retain_loader, joint_loader


def _collect_tagged_logits_cache(
    model: torch.nn.Module,
    tagged_loader,
    text_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Dict[str, object]]:
    slots = {
        "forget": {"logits": [], "labels": [], "eval_idx": [], "paths": []},
        "retain": {"logits": [], "labels": [], "eval_idx": [], "paths": []},
        "joint": {"logits": [], "labels": [], "eval_idx": [], "paths": []},
    }

    model.eval()
    with torch.no_grad():
        for batch in tagged_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            eval_class_idx = batch.get("eval_class_idx")
            image_paths = batch.get("image_path")
            source_tags = batch.get("_source_tag")

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * (image_features @ text_features.t())

            logits_np = logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            if eval_class_idx is None:
                eval_idx_np = np.full((labels.size(0),), -1, dtype=np.int64)
            else:
                eval_idx_np = eval_class_idx.detach().cpu().numpy().astype(np.int64)

            for i in range(labels.size(0)):
                tag = str(source_tags[i])
                if tag not in slots:
                    continue
                slots[tag]["logits"].append(logits_np[i])
                slots[tag]["labels"].append(labels_np[i])
                slots[tag]["eval_idx"].append(eval_idx_np[i])
                if image_paths is None:
                    slots[tag]["paths"].append(None)
                else:
                    slots[tag]["paths"].append(image_paths[i])

    out: Dict[str, Dict[str, object]] = {}
    num_classes = int(text_features.size(0))
    for tag, rec in slots.items():
        if rec["logits"]:
            out[tag] = {
                "logits": np.stack(rec["logits"], axis=0),
                "labels": np.stack(rec["labels"], axis=0),
                "eval_idx": np.asarray(rec["eval_idx"], dtype=np.int64),
                "image_paths": list(rec["paths"]),
            }
        else:
            out[tag] = {
                "logits": np.zeros((0, num_classes), dtype=np.float32),
                "labels": np.zeros((0, num_classes), dtype=np.float32),
                "eval_idx": np.zeros((0,), dtype=np.int64),
                "image_paths": [],
            }
    return out


def _single_accuracy_from_cache(cache: Dict[str, object]) -> float:
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


def _single_class_accuracy_from_cache(cache: Dict[str, object], class_idx: int) -> float:
    logits = cache["logits"]
    labels = cache["labels"]
    eval_idx = cache["eval_idx"]
    if logits.shape[0] == 0:
        return 0.0
    if np.any(eval_idx >= 0):
        mask = eval_idx == class_idx
    else:
        mask = (labels.sum(axis=1) == 1) & (labels[:, class_idx] > 0.5)
    if mask.sum() == 0:
        return 0.0
    pred = logits.argmax(axis=1)
    return float((pred[mask] == class_idx).mean())


def _topk_retain_accuracy_from_cache(
    retain_cache: Dict[str, object],
    class_names: Sequence[str],
    retain_topk_indices: Sequence[int],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for idx in retain_topk_indices:
        out[class_names[idx]] = _single_class_accuracy_from_cache(retain_cache, int(idx))
    return out


def _joint_multilabel_ap_from_cache(
    joint_cache: Dict[str, object],
    class_names: Sequence[str],
    forget_indices: Sequence[int],
    retain_topk_indices: Sequence[int],
) -> Dict[str, object]:
    scores = joint_cache["logits"]
    gt = joint_cache["labels"]
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
        ap = _average_precision_binary(gt[:, class_idx].astype(np.int32), scores[:, class_idx])
        if not np.isnan(ap):
            per_class_ap[int(class_idx)] = float(ap)

    forget_aps = [per_class_ap[idx] for idx in forget_indices if idx in per_class_ap]
    forget_class_ap = {class_names[idx]: per_class_ap[idx] for idx in forget_indices if idx in per_class_ap}
    retain_topk_ap = {
        class_names[idx]: per_class_ap[idx] for idx in retain_topk_indices if idx in per_class_ap
    }
    excluded = set(forget_indices) | set(retain_topk_indices)
    other_indices = [idx for idx in present_classes if idx not in excluded]
    other_map = _compute_map_for_indices(gt, scores, other_indices)
    return {
        "eval_size": int(gt.shape[0]),
        "forget_map": float(np.mean(forget_aps)) if forget_aps else 0.0,
        "forget_class_ap": forget_class_ap,
        "retain_topk_ap": retain_topk_ap,
        "other_map": other_map,
        "other_classes": [class_names[idx] for idx in other_indices],
    }


def _collect_logits_and_labels(
    model: torch.nn.Module,
    data_loader,
    text_features: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * (image_features @ text_features.t())

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    if not all_logits:
        return np.zeros((0, text_features.size(0)), dtype=np.float32), np.zeros((0, text_features.size(0)), dtype=np.float32)
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def _single_class_accuracy_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    class_idx: int,
) -> float:
    if logits.size == 0:
        return float("nan")
    valid = (labels.sum(axis=1) == 1) & (labels[:, class_idx] > 0.5)
    if valid.sum() == 0:
        return float("nan")
    preds = logits.argmax(axis=1)
    return float((preds[valid] == class_idx).mean())


def _group_accuracy_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    class_indices: Sequence[int],
) -> float:
    accs: List[float] = []
    for idx in class_indices:
        a = _single_class_accuracy_from_logits(logits, labels, idx)
        if not np.isnan(a):
            accs.append(a)
    if not accs:
        return 0.0
    return float(np.mean(accs))


def _group_map_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    class_indices: Sequence[int],
) -> float:
    if len(class_indices) == 0 or logits.size == 0:
        return 0.0
    return _compute_map_for_indices(labels, logits, class_indices)


def _single_class_ap_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    class_idx: int,
) -> float:
    if logits.size == 0:
        return float("nan")
    return _average_precision_binary(labels[:, class_idx].astype(np.int32), logits[:, class_idx])


def _per_class_metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    class_indices: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for idx in class_indices:
        out[int(idx)] = {
            "accuracy": _single_class_accuracy_from_logits(logits, labels, int(idx)),
            "AP": _single_class_ap_from_logits(logits, labels, int(idx)),
        }
    return out


def _rank_cooccur_classes(
    args,
    forget_indices: Sequence[int],
    num_classes: int,
    image_size: int,
) -> List[Tuple[int, int]]:
    if args.group_source == "train_df":
        df_dataset, _ = _build_train_datasets(args, transform=None)
        targets = np.stack(df_dataset.targets, axis=0) if df_dataset.targets else np.zeros((0, num_classes), dtype=np.float32)
    else:
        prev = args.return_meta
        args.return_meta = True
        forget_loader, _ = build_test_dataloaders(args, transform=_build_eval_transform(image_size))
        args.return_meta = prev
        rows: List[np.ndarray] = []
        for b in forget_loader:
            rows.append(b["label"].cpu().numpy())
        targets = np.concatenate(rows, axis=0) if rows else np.zeros((0, num_classes), dtype=np.float32)

    if targets.size == 0:
        return []

    forget_mask = targets[:, forget_indices].sum(axis=1) > 0.5
    if forget_mask.sum() == 0:
        return []

    counts = targets[forget_mask].sum(axis=0).astype(np.int64)
    ranked = []
    forget_set = set(forget_indices)
    for idx in np.argsort(-counts):
        idx_i = int(idx)
        if idx_i in forget_set:
            continue
        c = int(counts[idx_i])
        ranked.append((idx_i, c))
    return ranked


def _pick_bottomk_indices(ranked: Sequence[Tuple[int, float]], k: int) -> List[int]:
    if k <= 0:
        return []
    picked = list(reversed(ranked[-k:]))
    return [int(idx) for idx, _ in picked]


def _evaluate_original_style_metrics(
    args,
    backend: str,
    class_names: Sequence[str],
    forget_indices: Sequence[int],
    retain_topk_indices: Sequence[int],
    cache_by_tag: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    forget_cache = cache_by_tag["forget"]
    retain_cache = cache_by_tag["retain"]
    joint_cache = cache_by_tag["joint"]

    forget_acc = _single_accuracy_from_cache(forget_cache)
    retain_acc = _single_accuracy_from_cache(retain_cache)
    joint_ap_metrics = _joint_multilabel_ap_from_cache(
        joint_cache=joint_cache,
        class_names=class_names,
        forget_indices=forget_indices,
        retain_topk_indices=retain_topk_indices,
    )

    metrics: Dict[str, object] = {
        "backend": backend,
        "clip_model_path": args.clip_model_path,
        "clip_arch": args.clip_arch,
        "forget_classes": list(args.forget_classes),
        "forget_test_size": int(forget_cache["labels"].shape[0]),
        "retain_test_size": int(retain_cache["labels"].shape[0]),
        "joint_multilabel_val_size": joint_ap_metrics["eval_size"],
        "joint_multilabel_max_per_class": int(getattr(args, "joint_multilabel_max_per_class", 0)),
        "forget_success": 1.0 - float(forget_acc),
        "retain_accuracy": float(retain_acc),
        "forget_map": float(joint_ap_metrics["forget_map"]),
        "forget_class_ap": dict(joint_ap_metrics["forget_class_ap"]),
        "retain_topk_ap": dict(joint_ap_metrics["retain_topk_ap"]),
        "other_map": float(joint_ap_metrics["other_map"]),
        "other_classes": list(joint_ap_metrics["other_classes"]),
    }
    if retain_topk_indices:
        topk_acc = _topk_retain_accuracy_from_cache(
            retain_cache=retain_cache,
            class_names=class_names,
            retain_topk_indices=retain_topk_indices,
        )
        metrics["retain_topk_classes"] = [class_names[i] for i in retain_topk_indices]
        metrics["retain_topk_accuracy"] = topk_acc
    return metrics


def _rank_semsim_classes(
    model: torch.nn.Module,
    tokenize_fn,
    class_names: Sequence[str],
    forget_indices: Sequence[int],
    device: torch.device,
) -> List[Tuple[int, float]]:
    text_feats = _encode_text_features(model, class_names, tokenize_fn, device)
    forget_vec = text_feats[forget_indices].mean(dim=0, keepdim=True)
    forget_vec = forget_vec / forget_vec.norm(dim=-1, keepdim=True)
    sims = (text_feats @ forget_vec.t()).squeeze(1).detach().cpu().numpy()

    ranked = []
    forget_set = set(forget_indices)
    for idx in np.argsort(-sims):
        idx_i = int(idx)
        if idx_i in forget_set:
            continue
        ranked.append((idx_i, float(sims[idx_i])))
    return ranked


def _pick_topk_indices(ranked: Sequence[Tuple[int, float]], k: int) -> List[int]:
    if k <= 0:
        return []
    return [idx for idx, _ in ranked[:k]]


def _to_score_map(ranked: Sequence[Tuple[int, float]]) -> Dict[int, float]:
    return {int(idx): float(score) for idx, score in ranked}


def _minmax_normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    if not score_map:
        return {}
    vals = np.array(list(score_map.values()), dtype=np.float64)
    v_min = float(vals.min())
    v_max = float(vals.max())
    if abs(v_max - v_min) < 1e-12:
        return {k: 0.0 for k in score_map.keys()}
    return {k: float((v - v_min) / (v_max - v_min)) for k, v in score_map.items()}


def _sort_score_map(score_map: Dict[int, float]) -> List[Tuple[int, float]]:
    return sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)


def _build_groups_and_rankings(
    cooccur_ranked: Sequence[Tuple[int, int]],
    semsim_ranked: Sequence[Tuple[int, float]],
    cooccur_topk: int,
    semsim_topk: int,
    norm_mean_topk: int,
) -> Tuple[Dict[str, List[int]], Dict[str, List[Tuple[int, float]]], Dict[str, Dict[int, float]]]:
    cooccur_score = _to_score_map(cooccur_ranked)
    semsim_score = _to_score_map(semsim_ranked)

    # Align key-space: non-target classes should exist in both maps.
    common_ids = sorted(set(cooccur_score.keys()) & set(semsim_score.keys()))
    cooccur_score = {k: cooccur_score[k] for k in common_ids}
    semsim_score = {k: semsim_score[k] for k in common_ids}

    cooccur_norm = _minmax_normalize(cooccur_score)
    semsim_norm = _minmax_normalize(semsim_score)
    norm_mean_score = {
        k: float(0.5 * (cooccur_norm.get(k, 0.0) + semsim_norm.get(k, 0.0))) for k in common_ids
    }

    ranked = {
        "cooccur": _sort_score_map(cooccur_score),
        "semsim": _sort_score_map(semsim_score),
        "norm_mean": _sort_score_map(norm_mean_score),
    }
    groups = {
        "cooccur": _pick_topk_indices(ranked["cooccur"], cooccur_topk),
        "semsim": _pick_topk_indices(ranked["semsim"], semsim_topk),
        "norm_mean": _pick_topk_indices(ranked["norm_mean"], norm_mean_topk),
    }
    score_maps = {
        "cooccur": cooccur_score,
        "semsim": semsim_score,
        "cooccur_norm": cooccur_norm,
        "semsim_norm": semsim_norm,
        "norm_mean": norm_mean_score,
    }
    return groups, ranked, score_maps


def _evaluate_groups(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for gname, idxs in groups.items():
        out[gname] = {
            "accuracy": _group_accuracy_from_logits(logits, labels, idxs),
            "mAP": _group_map_from_logits(logits, labels, idxs),
            "num_classes": len(idxs),
        }
    return out


def _add_drop(
    before_metrics: Dict[str, Dict[str, float]],
    after_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for g in before_metrics.keys():
        out[g] = {
            "accuracy_drop": float(before_metrics[g]["accuracy"] - after_metrics[g]["accuracy"]),
            "mAP_drop": float(before_metrics[g]["mAP"] - after_metrics[g]["mAP"]),
        }
    return out


def _add_per_class_drop(
    before_metrics: Dict[int, Dict[str, float]],
    after_metrics: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    keys = sorted(set(before_metrics.keys()) | set(after_metrics.keys()))
    for idx in keys:
        b = before_metrics.get(idx, {})
        a = after_metrics.get(idx, {})
        b_acc = float(b.get("accuracy", float("nan")))
        a_acc = float(a.get("accuracy", float("nan")))
        b_ap = float(b.get("AP", float("nan")))
        a_ap = float(a.get("AP", float("nan")))
        out[idx] = {
            "accuracy_drop": float(b_acc - a_acc) if not (np.isnan(b_acc) or np.isnan(a_acc)) else float("nan"),
            "AP_drop": float(b_ap - a_ap) if not (np.isnan(b_ap) or np.isnan(a_ap)) else float("nan"),
        }
    return out


def _flatten_unique(indices_groups: Sequence[Sequence[int]]) -> List[int]:
    seen = set()
    out: List[int] = []
    for group in indices_groups:
        for idx in group:
            i = int(idx)
            if i in seen:
                continue
            seen.add(i)
            out.append(i)
    return out


def _write_text_report(
    out_path: str,
    groups_named: Dict[str, List[str]],
    before_metrics: Dict[str, Dict[str, float]],
    after_metrics: Dict[str, Dict[str, float]],
    drop_metrics: Dict[str, Dict[str, float]],
    per_class_rows: List[Dict[str, object]],
    groupwise_rows: Dict[str, List[Dict[str, object]]],
) -> None:
    lines: List[str] = []
    lines.append("Collateral Forgetting Evaluation")
    lines.append("")
    for g in before_metrics.keys():
        classes = groups_named.get(g, [])
        b = before_metrics[g]
        a = after_metrics[g]
        d = drop_metrics[g]
        lines.append(f"[{g}]")
        lines.append(f"num_classes: {len(classes)}")
        lines.append("classes: " + (", ".join(classes) if classes else "(empty)"))
        lines.append(
            "accuracy: "
            f"{b['accuracy']:.6f} -> {a['accuracy']:.6f} (drop {d['accuracy_drop']:.6f})"
        )
        lines.append(
            "mAP: "
            f"{b['mAP']:.6f} -> {a['mAP']:.6f} (drop {d['mAP_drop']:.6f})"
        )
        lines.append("group_score_sorted_class_metrics:")
        for row in groupwise_rows.get(g, []):
            lines.append(
                f"  {row['class']} (score={row['group_score']:.6f}): "
                f"acc {row['acc_before']:.6f}->{row['acc_after']:.6f} (drop {row['acc_drop']:.6f}), "
                f"AP {row['ap_before']:.6f}->{row['ap_after']:.6f} (drop {row['ap_drop']:.6f})"
            )
        lines.append("")
    lines.append("Per-class Metrics")
    lines.append("class,groups,acc_before,acc_after,acc_drop,ap_before,ap_after,ap_drop")
    for row in per_class_rows:
        lines.append(
            f"{row['class']},{row['groups']},{row['acc_before']:.6f},{row['acc_after']:.6f},"
            f"{row['acc_drop']:.6f},{row['ap_before']:.6f},{row['ap_after']:.6f},{row['ap_drop']:.6f}"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.train_annotation_file = os.path.join(args.coco_root, "annotations", f"instances_{args.train_split}2017.json")
    args.val_annotation_file = os.path.join(args.coco_root, "annotations", f"instances_{args.val_split}2017.json")
    args.train_image_root = os.path.join(args.coco_root, f"{args.train_split}2017")
    args.val_image_root = os.path.join(args.coco_root, f"{args.val_split}2017")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    class_names = _build_class_name_list()
    forget_indices, _ = _build_forget_retain_indices(args.forget_classes)

    before_model, before_tokenize, image_size, backend = _load_before_model_from_arch(args, device)
    after_model, after_tokenize, _, _ = _load_after_model_from_ckpt(args, args.after_ckpt, device)

    cooccur_ranked = _rank_cooccur_classes(
        args,
        forget_indices=forget_indices,
        num_classes=len(class_names),
        image_size=image_size,
    )
    cooccur_eval_k = int(args.cooccur_eval_k)
    groups = {
        "cooccur_top": _pick_topk_indices(cooccur_ranked, cooccur_eval_k),
        "cooccur_bottom": _pick_bottomk_indices(cooccur_ranked, cooccur_eval_k),
    }
    rankings = {"cooccur": cooccur_ranked}

    eval_transform = _build_eval_transform(image_size)
    df_dataset, _ = _build_train_datasets(args, transform=eval_transform)
    retain_topk_indices = _compute_topk_retain_classes(
        df_dataset=df_dataset,
        forget_indices=forget_indices,
        k=args.retain_topk,
    )
    tagged_loader, _, _, _ = _build_tagged_eval_loader(
        args=args,
        eval_transform=eval_transform,
        forget_indices=forget_indices,
        retain_topk_indices=retain_topk_indices,
    )

    before_text = _encode_text_features(before_model, class_names, before_tokenize, device)
    before_cache = _collect_tagged_logits_cache(before_model, tagged_loader, before_text, device)
    after_text = _encode_text_features(after_model, class_names, after_tokenize, device)
    after_cache = _collect_tagged_logits_cache(after_model, tagged_loader, after_text, device)

    # 1) Metrics aligned with run_original_eval for before/after checkpoints.
    before_original_eval = _evaluate_original_style_metrics(
        args=args,
        backend=backend,
        class_names=class_names,
        forget_indices=forget_indices,
        retain_topk_indices=retain_topk_indices,
        cache_by_tag=before_cache,
    )
    after_original_eval = _evaluate_original_style_metrics(
        args=args,
        backend=backend,
        class_names=class_names,
        forget_indices=forget_indices,
        retain_topk_indices=retain_topk_indices,
        cache_by_tag=after_cache,
    )

    # Reuse retain split logits cached in the single forward pass.
    before_logits = before_cache["retain"]["logits"]
    labels = before_cache["retain"]["labels"]
    after_logits = after_cache["retain"]["logits"]

    before_metrics = _evaluate_groups(before_logits, labels, groups)
    after_metrics = _evaluate_groups(after_logits, labels, groups)
    drop_metrics = _add_drop(before_metrics, after_metrics)

    idx_to_name = {i: n for i, n in enumerate(class_names)}
    groups_named = {k: [idx_to_name[i] for i in v] for k, v in groups.items()}
    name_to_groups: Dict[int, List[str]] = {}
    for gname, idxs in groups.items():
        for i in idxs:
            name_to_groups.setdefault(int(i), []).append(gname)

    class_indices_union = _flatten_unique([groups["cooccur_top"], groups["cooccur_bottom"]])
    per_class_before = _per_class_metrics_from_logits(before_logits, labels, class_indices_union)
    per_class_after = _per_class_metrics_from_logits(after_logits, labels, class_indices_union)
    per_class_drop = _add_per_class_drop(per_class_before, per_class_after)

    per_class_rows: List[Dict[str, object]] = []
    for idx in class_indices_union:
        b = per_class_before[idx]
        a = per_class_after[idx]
        d = per_class_drop[idx]
        per_class_rows.append(
            {
                "class": idx_to_name[idx],
                "class_index": int(idx),
                "groups": "|".join(name_to_groups.get(int(idx), [])),
                "acc_before": float(b["accuracy"]),
                "acc_after": float(a["accuracy"]),
                "acc_drop": float(d["accuracy_drop"]),
                "ap_before": float(b["AP"]),
                "ap_after": float(a["AP"]),
                "ap_drop": float(d["AP_drop"]),
            }
        )
    per_class_rows.sort(
        key=lambda x: x["acc_drop"] if not np.isnan(float(x["acc_drop"])) else -1e9,
        reverse=True,
    )
    row_by_class_idx = {int(r["class_index"]): r for r in per_class_rows}
    groupwise_rows: Dict[str, List[Dict[str, object]]] = {}
    for g in ["cooccur_top", "cooccur_bottom"]:
        rows: List[Dict[str, object]] = []
        for idx, score in rankings["cooccur"]:
            if idx not in set(groups[g]):
                continue
            base = row_by_class_idx.get(int(idx))
            if base is None:
                continue
            merged = dict(base)
            merged["group_score"] = float(score)
            rows.append(merged)
        groupwise_rows[g] = rows

    semsim_ap = _average_precision_binary(
        (labels[:, forget_indices].sum(axis=1) > 0).astype(np.int32),
        before_logits[:, forget_indices].max(axis=1),
    )

    original_eval_drop = {
        "forget_success_drop": float(before_original_eval["forget_success"]) - float(after_original_eval["forget_success"]),
        "retain_accuracy_drop": float(before_original_eval["retain_accuracy"]) - float(after_original_eval["retain_accuracy"]),
        "forget_map_drop": float(before_original_eval["forget_map"]) - float(after_original_eval["forget_map"]),
        "other_map_drop": float(before_original_eval["other_map"]) - float(after_original_eval["other_map"]),
    }

    metrics = {
        "backend": backend,
        "clip_arch": args.clip_arch,
        "clip_pretrained": args.clip_pretrained,
        "forget_classes": list(args.forget_classes),
        "before_model_source": "clip_arch",
        "before_clip_arch": args.clip_arch,
        "before_ckpt": args.before_ckpt,
        "after_ckpt": args.after_ckpt,
        "eval_retain_size": int(labels.shape[0]),
        "group_source": args.group_source,
        "cooccur_eval_k": cooccur_eval_k,
        "groups": groups_named,
        "group_rankings": {
            "cooccur": [{"class": idx_to_name[idx], "score": float(score)} for idx, score in rankings["cooccur"]]
        },
        "original_eval_before": before_original_eval,
        "original_eval_after": after_original_eval,
        "original_eval_drop": original_eval_drop,
        "before": before_metrics,
        "after": after_metrics,
        "drop": drop_metrics,
        "per_class": per_class_rows,
        "per_class_by_group_sorted": groupwise_rows,
        "debug": {
            "cooccur_ranked_top20": [
                {"class": idx_to_name[idx], "count": score} for idx, score in cooccur_ranked[:20]
            ],
            "forget_presence_ap_on_retain_eval": float(semsim_ap) if not np.isnan(semsim_ap) else None,
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    txt_out_path = os.path.splitext(out_path)[0] + ".txt"
    _write_text_report(
        out_path=txt_out_path,
        groups_named=groups_named,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        drop_metrics=drop_metrics,
        per_class_rows=per_class_rows,
        groupwise_rows=groupwise_rows,
    )

    per_class_path = os.path.join(args.output_dir, "collateral_per_class.csv")
    with open(per_class_path, "w", encoding="utf-8") as f:
        f.write("class,class_index,groups,acc_before,acc_after,acc_drop,ap_before,ap_after,ap_drop\n")
        for row in per_class_rows:
            f.write(
                f"{row['class']},{row['class_index']},{row['groups']},{row['acc_before']:.6f},{row['acc_after']:.6f},"
                f"{row['acc_drop']:.6f},{row['ap_before']:.6f},{row['ap_after']:.6f},{row['ap_drop']:.6f}\n"
            )

    print(f"Saved to: {out_path}")
    print(f"Saved to: {txt_out_path}")
    print(f"Saved to: {per_class_path}")
    print(
        "Original-eval metrics (before -> after): "
        f"forget_success {before_original_eval['forget_success']:.4f}->{after_original_eval['forget_success']:.4f}, "
        f"retain_accuracy {before_original_eval['retain_accuracy']:.4f}->{after_original_eval['retain_accuracy']:.4f}, "
        f"forget_map {before_original_eval['forget_map']:.4f}->{after_original_eval['forget_map']:.4f}, "
        f"other_map {before_original_eval['other_map']:.4f}->{after_original_eval['other_map']:.4f}"
    )
    for g in ["cooccur_top", "cooccur_bottom"]:
        b = before_metrics[g]
        a = after_metrics[g]
        d = drop_metrics[g]
        group_cls = groups_named.get(g, [])
        print(f"[{g}] classes: {', '.join(group_cls) if group_cls else '(empty)'}")
        print(
            f"[{g}] n={b['num_classes']} "
            f"acc: {b['accuracy']:.4f} -> {a['accuracy']:.4f} (drop {d['accuracy_drop']:.4f}), "
            f"mAP: {b['mAP']:.4f} -> {a['mAP']:.4f} (drop {d['mAP_drop']:.4f})"
        )
        for row in groupwise_rows.get(g, []):
            print(
                f"  - {row['class']} (score={row['group_score']:.4f}): "
                f"acc {row['acc_before']:.4f}->{row['acc_after']:.4f} (drop {row['acc_drop']:.4f}), "
                f"AP {row['ap_before']:.4f}->{row['ap_after']:.4f} (drop {row['ap_drop']:.4f})"
            )
    print("Top class drops by accuracy:")
    for row in per_class_rows[:10]:
        print(
            f"{row['class']} ({row['groups']}): "
            f"acc {row['acc_before']:.4f}->{row['acc_after']:.4f} (drop {row['acc_drop']:.4f}), "
            f"AP {row['ap_before']:.4f}->{row['ap_after']:.4f} (drop {row['ap_drop']:.4f})"
        )


if __name__ == "__main__":
    main()

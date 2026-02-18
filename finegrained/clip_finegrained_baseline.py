import json
import os
import sys
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torchvision import transforms
import open_clip

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finegrained.coco_labels.coco import LABEL_NAMES
from finegrained.load_dataset import build_test_dataloaders, build_train_dataloader
from finegrained.params import parse_args


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_clip_backend(args, device: torch.device):
    # args.clip_model_path 这里建议传 "ViT-B-32" 这种 model_name
    # args.clip_pretrained 建议传 "openai" 或 "laion2b_s34b_b79k" 这种 pretrained tag
    model_name = getattr(args, "clip_arch", "ViT-B-32")
    pretrained = getattr(args, "clip_pretrained", "openai")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    tokenize_fn = open_clip.get_tokenizer(model_name)

    # open_clip 的 input size：从 preprocess 里取
    image_size = preprocess.transforms[0].size if hasattr(preprocess, "transforms") else 224

    model.eval()
    return model, tokenize_fn, image_size, "open_clip"

def _build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def _build_class_name_list() -> List[str]:
    return [name for _, name in sorted(LABEL_NAMES.items())]


def _build_forget_retain_indices(forget_classes: Sequence[str]) -> Tuple[List[int], List[int]]:
    norm_forget = {name.strip().replace(" ", "_") for name in forget_classes}
    all_classes = _build_class_name_list()
    name_to_idx = {name: idx for idx, name in enumerate(all_classes)}

    forget_indices = []
    for name in norm_forget:
        if name not in name_to_idx:
            raise ValueError(f"Unknown forget class: {name}")
        forget_indices.append(name_to_idx[name])
    forget_indices = sorted(set(forget_indices))
    retain_indices = [i for i in range(len(all_classes)) if i not in set(forget_indices)]
    return forget_indices, retain_indices


def _encode_text_features(
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


def _evaluate_single_accuracy(
    model,
    data_loader,
    text_features: torch.Tensor,
    eval_indices: Sequence[int],
    device: torch.device,
) -> float:
    if len(eval_indices) == 0:
        return 0.0

    model.eval()
    eval_indices_tensor = torch.tensor(eval_indices, dtype=torch.long, device=device)
    subset_text_features = text_features[eval_indices_tensor]

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * (image_features @ subset_text_features.t())
            pred_local = logits.argmax(dim=1)

            target_subset = labels.index_select(1, eval_indices_tensor)
            valid_mask = labels.sum(dim=1) == 1
            if valid_mask.any():
                gt_local = target_subset.argmax(dim=1)
                batch_correct = (pred_local[valid_mask] == gt_local[valid_mask]).sum().item()
                correct += int(batch_correct)
                total += int(valid_mask.sum().item())

    return float(correct / total) if total > 0 else 0.0


def _average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _compute_map_for_indices(
    gt: np.ndarray,
    scores: np.ndarray,
    indices: Sequence[int],
) -> float:
    ap_values: List[float] = []
    for class_idx in indices:
        ap = _average_precision_binary(gt[:, class_idx].astype(np.int32), scores[:, class_idx])
        if not np.isnan(ap):
            ap_values.append(ap)
    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))


def _evaluate_multi_label_map(
    model,
    data_loader,
    text_features: torch.Tensor,
    forget_indices: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

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
        return {"forget_map": 0.0, "retain_map": 0.0}

    scores = np.concatenate(all_scores, axis=0)
    gt = np.concatenate(all_labels, axis=0)

    forget_set = set(forget_indices)
    present_classes = np.where(gt.sum(axis=0) > 0)[0].tolist()
    retain_indices = [idx for idx in present_classes if idx not in forget_set]

    forget_map = _compute_map_for_indices(gt, scores, forget_indices)
    retain_map = _compute_map_for_indices(gt, scores, retain_indices)
    return {"forget_map": forget_map, "retain_map": retain_map}


def run_original_eval(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenize_fn, image_size, backend = _load_clip_backend(args, device)
    eval_transform = _build_eval_transform(image_size)

    forget_loader, retain_loader = build_test_dataloaders(args, transform=eval_transform)
    # multi_label_loader = build_train_dataloader(args, transform=eval_transform)
    class_names = _build_class_name_list()
    forget_indices, retain_indices = _build_forget_retain_indices(args.forget_classes)
    text_features = _encode_text_features(model, class_names, tokenize_fn, device)

    forget_acc = _evaluate_single_accuracy(
        model=model,
        data_loader=forget_loader,
        text_features=text_features,
        eval_indices=forget_indices,
        device=device,
    )
    retain_acc = _evaluate_single_accuracy(
        model=model,
        data_loader=retain_loader,
        text_features=text_features,
        eval_indices=retain_indices,
        device=device,
    )
    # multi_map = _evaluate_multi_label_map(
    #     model=model,
    #     data_loader=multi_label_loader,
    #     text_features=text_features,
    #     forget_indices=forget_indices,
    #     device=device,
    # )

    os.makedirs(args.output_dir, exist_ok=True)
    metrics = {
        "backend": backend,
        "clip_model_path": args.clip_model_path,
        "clip_arch": args.clip_arch,
        "forget_classes": list(args.forget_classes),
        "forget_test_size": len(forget_loader.dataset),
        "retain_test_size": len(retain_loader.dataset),
        "forget_accuracy": forget_acc,
        "retain_accuracy": retain_acc,
        # "multi_label_eval_size": len(multi_label_loader.dataset),
        # "forget_map": multi_map["forget_map"],
        # "retain_map": multi_map["retain_map"],
    }
    metrics_path = os.path.join(args.output_dir, "original_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Forget test accuracy: {forget_acc:.4f}")
    print(f"Retain test accuracy: {retain_acc:.4f}")
    # print(f"Forget mAP (multi-label): {multi_map['forget_map']:.4f}")
    # print(f"Retain mAP (multi-label): {multi_map['retain_map']:.4f}")
    print(f"Saved metrics to: {metrics_path}")


def main() -> None:
    args = parse_args()
    if args.original_eval:
        run_original_eval(args)
        return
    raise NotImplementedError("Only `--original_eval` mode is implemented in this script.")


if __name__ == "__main__":
    main()

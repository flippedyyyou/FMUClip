import copy
import json
import os
import sys
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torchvision import transforms
import open_clip
import torch.nn.functional as F
from torch.utils.data import DataLoader

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finegrained.coco_labels.coco import LABEL_NAMES
from finegrained.load_dataset import COCODataSet, build_test_dataloaders
from finegrained.params import parse_args


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_clip_backend(args, device: torch.device):
    # args.clip_model_path 这里建议传 "ViT-B-32" 这种 model_name
    model_name = getattr(args, "clip_arch", "ViT-B-32")

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


def _build_train_datasets(args, transform):
    forget_class_names = [_normalize_name(x) for x in args.forget_classes]
    all_class_names = [_normalize_name(v) for _, v in sorted(LABEL_NAMES.items())]
    retain_class_names = [c for c in all_class_names if c not in set(forget_class_names)]

    # Forget set: item3 of forget classes
    forget_files = _load_images_from_df_lists(
        df_root=args.df_root,
        split=args.train_split,
        item_folder=args.train_item_folder,
        class_names=forget_class_names,
    )
    # Retain set: item1 of non-forget classes
    retain_files = _load_images_from_df_lists(
        df_root=args.df_root,
        split=args.train_split,
        item_folder=args.retain_item_folder,
        class_names=retain_class_names,
    )

    df_dataset = COCODataSet(
        annotation_file=args.train_annotation_file,
        image_root=args.train_image_root,
        split=args.train_split,
        transform=transform,
        return_meta=args.return_meta,
        selected_files=forget_files,
        forget_class_names=forget_class_names,
    )
    dr_dataset = COCODataSet(
        annotation_file=args.train_annotation_file,
        image_root=args.train_image_root,
        split=args.train_split,
        transform=transform,
        return_meta=args.return_meta,
        selected_files=retain_files,
        forget_class_names=forget_class_names,
    )
    return df_dataset, dr_dataset


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


def _get_logits_and_feats(
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


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _labels_to_texts(
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


def supervised_unlearn_train_cliperase(
    model,
    teacher,
    df_train_loader,
    dr_train_loader,
    tokenizer_fn: Callable[[Sequence[str]], torch.Tensor],
    class_names: Sequence[str],
    forget_indices: Sequence[int],
    retain_indices: Sequence[int],
    optimizer,
    scaler,
    lambda_df: float = 1.0,
    lambda_dr: float = 1.0,
    lambda_uni: float = 1.0,
    epoch_idx: int = 0,
    max_epoch: int = 1,
    log_interval: int = 50,
) -> Dict[str, float]:
    device = _get_model_device(model)
    iters_per_epoch = min(len(df_train_loader), len(dr_train_loader))

    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return (x * mask).sum() / denom

    model.train()
    df_iter = iter(df_train_loader)
    dr_iter = iter(dr_train_loader)
    running = {"forget": 0.0, "retain": 0.0, "kl": 0.0, "tot": 0.0}

    for it in range(iters_per_epoch):
        df_s = next(df_iter)
        dr_s = next(dr_iter)

        img_df = df_s["image"].to(device, non_blocking=True)
        img_dr = dr_s["image"].to(device, non_blocking=True)

        txt_df = _labels_to_texts(df_s["label"], class_names, forget_indices)
        txt_dr = _labels_to_texts(dr_s["label"], class_names, retain_indices)
        txt_df = tokenizer_fn(txt_df).to(device)
        txt_dr = tokenizer_fn(txt_dr).to(device)

        img_all = torch.cat([img_df, img_dr], dim=0)
        txt_all = torch.cat([txt_df, txt_dr], dim=0)

        batch_forget = img_df.size(0)
        batch_retain = img_dr.size(0)
        batch_total = batch_forget + batch_retain

        flags = torch.cat(
            [
                torch.ones(batch_forget, dtype=torch.long, device=device),
                torch.zeros(batch_retain, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        forget_mask = flags.float()
        retain_mask = (1 - flags).float()
        targets = torch.arange(batch_total, device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            sim_i2t_u, sim_t2i_u, _, _ = _get_logits_and_feats(model, img_all, txt_all)
            with torch.no_grad():
                sim_i2t_t, sim_t2i_t, _, _ = _get_logits_and_feats(teacher, img_all, txt_all)

            ce_img = F.cross_entropy(sim_i2t_u, targets, reduction="none")
            ce_txt = F.cross_entropy(sim_t2i_u, targets, reduction="none")

            forget_img_loss = masked_mean(ce_img, forget_mask)
            forget_txt_loss = masked_mean(ce_txt, forget_mask)
            loss_forget = -(forget_img_loss + forget_txt_loss)

            retain_img_loss = masked_mean(ce_img, retain_mask)
            retain_txt_loss = masked_mean(ce_txt, retain_mask)
            loss_retain = retain_img_loss + retain_txt_loss

            log_p_img = F.log_softmax(sim_i2t_u, dim=-1)
            p_img_t = F.softmax(sim_i2t_t, dim=-1)
            log_p_txt = F.log_softmax(sim_t2i_u, dim=-1)
            p_txt_t = F.softmax(sim_t2i_t, dim=-1)

            kl_img_all = F.kl_div(log_p_img, p_img_t, reduction="none").sum(dim=-1)
            kl_txt_all = F.kl_div(log_p_txt, p_txt_t, reduction="none").sum(dim=-1)
            kl_img = masked_mean(kl_img_all, retain_mask)
            kl_txt = masked_mean(kl_txt_all, retain_mask)
            loss_kl = kl_img + kl_txt

            loss = lambda_df * loss_forget + lambda_dr * loss_retain + lambda_uni * loss_kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["forget"] += float(loss_forget.detach().item())
        running["retain"] += float(loss_retain.detach().item())
        running["kl"] += float(loss_kl.detach().item())
        running["tot"] += float(loss.detach().item())

        if (it + 1) % log_interval == 0:
            t = it + 1
            print(
                f"[ClipErase EP {epoch_idx+1}/{max_epoch}] it={t}/{iters_per_epoch} "
                f"forget={running['forget']/t:.4f} "
                f"retain={running['retain']/t:.4f} "
                f"kl={running['kl']/t:.4f} "
                f"total={running['tot']/t:.4f}"
            )

    denom = float(max(iters_per_epoch, 1))
    return {
        "loss_forget": running["forget"] / denom,
        "loss_retain": running["retain"] / denom,
        "loss_kl": running["kl"] / denom,
        "loss_total": running["tot"] / denom,
    }


def _evaluate_single_accuracy(
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

            if eval_class_idx is not None and (eval_class_idx >= 0).any():
                valid_mask = eval_class_idx >= 0
                gt_global = eval_class_idx
            else:
                valid_mask = labels.sum(dim=1) == 1
                gt_global = labels.argmax(dim=1)
            if not valid_mask.any():
                continue

            k = min(topk, logits.size(1))
            topk_scores, topk_indices = logits.topk(k, dim=1)
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
    df_dataset: COCODataSet,
    forget_indices: Sequence[int],
    k: int,
) -> List[int]:
    if k <= 0:
        return []
    forget_set = set(forget_indices)
    counts = np.zeros(len(LABEL_NAMES), dtype=np.int64)
    for target in df_dataset.targets:
        for idx in np.where(target > 0.5)[0]:
            if idx in forget_set:
                continue
            counts[idx] += 1
    topk = np.argsort(-counts) # 返回数组值从大到小的索引值
    topk = [int(i) for i in topk if counts[int(i)] > 0 and i not in forget_set]
    for idx in topk[:k]:
        print(f"Retain class: {LABEL_NAMES[idx]} with count: {counts[idx]}")
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
        acc = _evaluate_single_class_accuracy(
            model=model,
            data_loader=data_loader,
            text_features=text_features,
            class_index=idx,
            device=device,
        )
        results[class_names[idx]] = acc
    return results


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


def run_original_eval(
    args,
    model: torch.nn.Module = None,
    tokenize_fn: Callable[[Sequence[str]], torch.Tensor] = None,
    image_size: int = None,
    backend: str = None,
    retain_topk_indices: Sequence[int] = None,
) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if model is None or tokenize_fn is None or image_size is None or backend is None:
        model, tokenize_fn, image_size, backend = _load_clip_backend(args, device)
    eval_transform = _build_eval_transform(image_size)

    prev_return_meta = args.return_meta
    args.return_meta = True
    forget_loader, retain_loader = build_test_dataloaders(args, transform=eval_transform)
    args.return_meta = prev_return_meta
    # multi_label_loader = build_train_dataloader(args, transform=eval_transform)
    class_names = _build_class_name_list()
    forget_indices, retain_indices = _build_forget_retain_indices(args.forget_classes)
    if retain_topk_indices is None and args.retain_topk > 0:
        df_dataset, _ = _build_train_datasets(args, transform=eval_transform)
        retain_topk_indices = _compute_topk_retain_classes(
            df_dataset=df_dataset,
            forget_indices=forget_indices,
            k=args.retain_topk,
        )
    text_features = _encode_text_features(model, class_names, tokenize_fn, device)

    forget_acc = _evaluate_single_accuracy(
        model=model,
        data_loader=forget_loader,
        text_features=text_features,
        device=device,
    )
    retain_acc = _evaluate_single_accuracy(
        model=model,
        data_loader=retain_loader,
        text_features=text_features,
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
        "forget_success": 1.0 - forget_acc,
        "retain_accuracy": retain_acc,
        # "multi_label_eval_size": len(multi_label_loader.dataset),
        # "forget_map": multi_map["forget_map"],
        # "retain_map": multi_map["retain_map"],
    }
    if retain_topk_indices:
        topk_acc = _evaluate_topk_retain_accuracy(
            model=model,
            data_loader=retain_loader,
            text_features=text_features,
            class_names=class_names,
            device=device,
            retain_indices=retain_topk_indices,
        )
        metrics["retain_topk_classes"] = [class_names[i] for i in retain_topk_indices]
        metrics["retain_topk_accuracy"] = topk_acc
    metrics_path = os.path.join(args.output_dir, "original_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    forget_topk_path = os.path.join(args.output_dir, "forget_test_topk.jsonl")
    retain_topk_path = os.path.join(args.output_dir, "retain_test_topk.jsonl")
    _dump_topk_results(
        model=model,
        data_loader=forget_loader,
        text_features=text_features,
        class_names=class_names,
        device=device,
        out_path=forget_topk_path,
    )
    _dump_topk_results(
        model=model,
        data_loader=retain_loader,
        text_features=text_features,
        class_names=class_names,
        device=device,
        out_path=retain_topk_path,
    )

    print(f"Forget test accuracy: {forget_acc:.4f}")
    print(f"Retain test accuracy: {retain_acc:.4f}")
    # print(f"Forget mAP (multi-label): {multi_map['forget_map']:.4f}")
    # print(f"Retain mAP (multi-label): {multi_map['retain_map']:.4f}")
    print(f"Saved metrics to: {metrics_path}")


def _evaluate_for_selection(
    model: torch.nn.Module,
    args,
    tokenize_fn: Callable[[Sequence[str]], torch.Tensor],
    image_size: int,
    class_names: Sequence[str],
) -> Dict[str, float]:
    device = _get_model_device(model)
    eval_transform = _build_eval_transform(image_size)
    prev_return_meta = args.return_meta
    args.return_meta = True
    forget_loader, retain_loader = build_test_dataloaders(args, transform=eval_transform)
    args.return_meta = prev_return_meta

    text_features = _encode_text_features(model, class_names, tokenize_fn, device)
    forget_acc = _evaluate_single_accuracy(
        model=model,
        data_loader=forget_loader,
        text_features=text_features,
        device=device,
    )
    retain_acc = _evaluate_single_accuracy(
        model=model,
        data_loader=retain_loader,
        text_features=text_features,
        device=device,
    )
    return {
        "forget_success": 1.0 - float(forget_acc),
        "retain_accuracy": float(retain_acc),
        "selection_score": float((1.0 - float(forget_acc)) + float(retain_acc)),
        "forget_test_size": len(forget_loader.dataset),
        "retain_test_size": len(retain_loader.dataset),
    }


def run_cliperase(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenize_fn, image_size, backend = _load_clip_backend(args, device)
    model = model.float().to(device)
    teacher = copy.deepcopy(model).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    train_transform = _build_eval_transform(image_size)
    df_dataset, dr_dataset = _build_train_datasets(args, transform=train_transform)
    df_loader = DataLoader(
        df_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    dr_loader = DataLoader(
        dr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    class_names = _build_class_name_list()
    forget_indices, retain_indices = _build_forget_retain_indices(args.forget_classes)
    retain_topk_indices = _compute_topk_retain_classes(
        df_dataset=df_dataset,
        forget_indices=forget_indices,
        k=args.retain_topk,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    ckpt_path = os.path.join(args.output_dir, "clip_finegrained_cliperase.pth")
    eval_interval = 2
    best_score = float("-inf")
    best_epoch = -1
    best_metrics = None

    for ep in range(args.max_epoch):
        train_stats = supervised_unlearn_train_cliperase(
            model=model,
            teacher=teacher,
            df_train_loader=df_loader,
            dr_train_loader=dr_loader,
            tokenizer_fn=tokenize_fn,
            class_names=class_names,
            forget_indices=forget_indices,
            retain_indices=retain_indices,
            optimizer=optimizer,
            scaler=scaler,
            lambda_df=args.lambda_df,
            lambda_dr=args.lambda_dr,
            lambda_uni=args.lambda_uni,
            epoch_idx=ep,
            max_epoch=args.max_epoch,
            log_interval=args.log_interval,
        )

        should_eval = ((ep + 1) % eval_interval == 0) or ((ep + 1) == args.max_epoch)
        if not should_eval:
            continue

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
                    "eval_interval_epoch": eval_interval,
                    "train_stats_at_best_epoch": train_stats,
                }
            )
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(save_cfg, f, ensure_ascii=False, indent=2)
            print(f"[Best Updated] epoch={best_epoch} score={best_score:.4f} -> overwrite checkpoint/config")

    if best_epoch < 0:
        raise RuntimeError("No evaluation executed, no best checkpoint saved.")

    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model"], strict=True)
    print(
        f"Final best epoch: {best_epoch}, "
        f"forget_success={best_metrics['forget_success']:.4f}, "
        f"retain_acc={best_metrics['retain_accuracy']:.4f}, "
        f"score={best_score:.4f}"
    )

    run_original_eval(
        args,
        model=model,
        tokenize_fn=tokenize_fn,
        image_size=image_size,
        backend=backend,
        retain_topk_indices=retain_topk_indices,
    )


def main() -> None:
    args = parse_args()
    if args.original_eval:
        run_original_eval(args)
        return
    if args.method == "cliperase":
        run_cliperase(args)
        return
    raise NotImplementedError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()

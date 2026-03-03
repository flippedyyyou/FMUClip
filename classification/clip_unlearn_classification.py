# coding=utf-8
from classification_utils import (
    _load_dict_from_path,
    _setup_logging,
    _encode_text_features,
    _encode_image_features,
    _get_logits_with_text_features,
    _get_logits_and_feats,
    ClassificationDataset,
    _load_forget_jsonl,
    _build_indices_from_list,
    _iter_labels,
    _compute_text_features,
    _evaluate_and_dump,
    _tokenize_texts
)
import argparse
import copy
import json
import logging
import os
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from lavis.models.clip_models.model import load_openai_model
from lavis.models.clip_models.tokenizer import tokenize
import importlib.util
import sys


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)


def _resolve_clip_model(model):
    return model.clip_model if hasattr(model, "clip_model") else model


def _encode_image_patches(model, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
    clip_model = _resolve_clip_model(model)
    if not hasattr(clip_model, "visual"):
        raise AttributeError("CLIP model does not expose a visual encoder for patch extraction.")
    visual = clip_model.visual
    if not hasattr(visual, "conv1"):
        raise AttributeError("Visual encoder does not expose patch embeddings.")
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
    blocks = getattr(visual.transformer, "resblocks", None)
    if blocks is not None and len(blocks) > 0:
        if len(blocks) >= 2:
            for blk in blocks[:-1]:
                x = blk(x)
        else:
            x = blocks[0](x)
    else:
        x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    patch_feats = x[:, 1:, :]
    return patch_feats, grid


def _load_sam3_masks(image_paths: Sequence[str], mask_dir: str, mask_suffix: str, target_size: int) -> torch.Tensor:
    masks = []
    for path in image_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        mask_path = os.path.join(mask_dir, f"{base}{mask_suffix}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"SAM3 mask not found: {mask_path}")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((target_size, target_size), resample=Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        masks.append(mask_tensor)
    return torch.stack(masks, dim=0)


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


def _build_text_pool(class_names: Sequence[str], retain_labels: Iterable[int]) -> List[str]:
    unique_labels = sorted(set(retain_labels))
    return [f"a photo of {class_names[label].replace('_', ' ')}" for label in unique_labels]


def _compute_class_stats(
    model,
    dataset: Dataset,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    topk: int = 5,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    text_features = _compute_text_features(model, class_names, device)
    model.eval()

    per_class = {idx: {"total": 0, "correct": 0, "cos_sum": 0.0} for idx in range(len(class_names))}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits, image_features = _get_logits_with_text_features(model, images, text_features)
            preds = logits.argmax(dim=1)
            cos_vals = F.cosine_similarity(image_features, text_features[labels], dim=1)

            for i in range(labels.numel()):
                label_idx = int(labels[i].item())
                per_class[label_idx]["total"] += 1
                per_class[label_idx]["correct"] += int(preds[i].item() == label_idx)
                per_class[label_idx]["cos_sum"] += float(cos_vals[i].item())

    results = {}
    for idx, stats in per_class.items():
        total = stats["total"]
        if total == 0:
            continue
        results[idx] = {
            "class_name": class_names[idx],
            "cosine_similarity_avg": stats["cos_sum"] / total,
            "accuracy": stats["correct"] / total,
            "count": total,
        }
    return results


def build_datasets(
    dataset_name: str,
    data_root: str,
    image_size: int,
    forget_indices: List[int],
    forget_classes: Set[int],
) -> Tuple[Dataset, Dataset, List[str]]:
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if dataset_name == "cifar100":
        base = datasets.CIFAR100(root=data_root, train=True, transform=tfm, download=False)
        class_names = base.classes
        use_index_path = True
    elif dataset_name == "imagenet":
        base = datasets.ImageNet(root=data_root, split="train", transform=tfm)
        class_names = base.classes
        use_index_path = False

    # 遗忘集：直接使用传入的 JSONL 索引
    df_dataset = ClassificationDataset(base, class_names, sorted(forget_indices), use_index_path=use_index_path)

    # 保留集：遍历原数据集，只要 label 不在 forget_classes 中，就加入保留集
    retain_indices = []
    for idx, (_, label) in enumerate(base):
        if label not in forget_classes:
            retain_indices.append(idx)

    dr_dataset = ClassificationDataset(base, class_names, retain_indices, use_index_path=use_index_path)

    return df_dataset, dr_dataset, class_names


def build_cifar100_test_datasets(
    data_root: str,
    image_size: int,
    class_names: Sequence[str],
    forget_classes: Set[int],
) -> Tuple[Dataset, Dataset]:
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_base = datasets.CIFAR100(root=data_root, train=False, transform=tfm, download=False)
    test_labels: Sequence[int] = getattr(test_base, "targets")

    df_test_indices = [idx for idx, label in enumerate(test_labels) if int(label) in forget_classes]
    dr_test_indices = [idx for idx, label in enumerate(test_labels) if int(label) not in forget_classes]

    df_test_dataset = ClassificationDataset(test_base, class_names, df_test_indices, use_index_path=True)
    dr_test_dataset = ClassificationDataset(test_base, class_names, dr_test_indices, use_index_path=True)
    return df_test_dataset, dr_test_dataset


def supervised_unlearn_train(
    model,
    teacher,
    df_loader,
    dr_loader,
    text_pool: List[str],
    optimizer,
    scaler,
    sam3_mask_dir: str,
    sam3_mask_suffix: str,
    label_mapping: dict,  # 修改点：传入加载好的字典映射
    lambda_rtf: float,
    lambda_keep: float,
    lambda_ce: float,
    max_epoch: int,
    log_interval: int,
) -> None:
    device = model.device
    mse = nn.MSELoss()

    iters_per_epoch = min(len(df_loader), len(dr_loader))

    for ep in range(max_epoch):
        model.train()
        df_iter = iter(df_loader)
        dr_iter = iter(dr_loader)
        with torch.no_grad():
            text_pool_tokens = _tokenize_texts(text_pool, device)
            text_pool_features = _encode_text_features(model, text_pool_tokens)

        running = {"rtf": 0.0, "keep": 0.0, "ce": 0.0, "tot": 0.0}
        for it in range(iters_per_epoch):
            df_s = next(df_iter)
            dr_s = next(dr_iter)
            img_df = df_s["image"].to(device, non_blocking=True)
            img_dr = dr_s["image"].to(device, non_blocking=True)
            df_labels = df_s["label"]  # 获取遗忘样本的类别索引
            df_image_paths = df_s["image_path"]
            dr_texts = dr_s["text"]
            dr_text_tokens = tokenize(dr_texts).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                # 1. 动态获取 Concept Tokens：根据 label_mapping 匹配类名
                concept_texts = [label_mapping[int(lbl)] for lbl in df_labels]
                concept_tokens = tokenize(concept_texts).to(device)

                sam3_masks = _load_sam3_masks(df_image_paths, sam3_mask_dir, sam3_mask_suffix, img_df.shape[-1])
                sam3_masks = sam3_masks.to(device, non_blocking=True)

                # 2. 计算 RTF Loss：使用对应类别的文本特征与 retain 文本池做对比
                patch_feats, grid = _encode_image_patches(model, img_df)
                df_text_feats = _encode_text_features(model, concept_tokens)
                patch_attn = _mask_to_patch_attention(sam3_masks, grid)
                pooled_patch_feats = _masked_max_mean_pool(patch_feats, patch_attn)

                clip_model = _resolve_clip_model(model)
                if getattr(clip_model.visual, "proj", None) is not None:
                    pooled_patch_feats = pooled_patch_feats @ clip_model.visual.proj
                pooled_patch_feats = pooled_patch_feats / pooled_patch_feats.norm(dim=-1, keepdim=True)

                contrast_text_feats = torch.cat([df_text_feats, text_pool_features], dim=0)
                logit_scale = clip_model.logit_scale.exp()
                forget_targets = torch.arange(img_df.size(0), device=device)

                logits_i2t = logit_scale * (pooled_patch_feats @ contrast_text_feats.t())
                loss_rtf_i2t = F.cross_entropy(logits_i2t, forget_targets)

                logits_t2i = logit_scale * (df_text_feats @ pooled_patch_feats.t())
                loss_rtf_t2i = F.cross_entropy(logits_t2i, forget_targets)
                loss_rtf = -0.5 * (loss_rtf_i2t + loss_rtf_t2i)

                sim_i2t_dr_u, sim_t2i_dr_u, img_dr_u, txt_dr_u = _get_logits_and_feats(
                    model, img_dr, dr_text_tokens
                )
                with torch.no_grad():
                    sim_i2t_dr_t, sim_t2i_dr_t, img_dr_t, txt_dr_t = _get_logits_and_feats(
                        teacher, img_dr, dr_text_tokens
                    )
                loss_keep = mse(sim_i2t_dr_u, sim_i2t_dr_t) + mse(sim_t2i_dr_u, sim_t2i_dr_t)
                retain_targets = torch.arange(img_dr.size(0), device=device)
                loss_ce = F.cross_entropy(sim_i2t_dr_u, retain_targets) + F.cross_entropy(sim_t2i_dr_u, retain_targets)

                loss = lambda_rtf * loss_rtf + lambda_keep * loss_keep + lambda_ce * loss_ce

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["rtf"] += float(loss_rtf.detach().item())
            running["keep"] += float(loss_keep.detach().item())
            running["ce"] += float(loss_ce.detach().item())
            running["tot"] += float(loss.detach().item())

            if (it + 1) % log_interval == 0:
                t = it + 1
                logging.info(
                    "EP %d/%d it=%d/%d rtf=%.4f keep=%.4f ce=%.4f total=%.4f",
                    ep + 1,
                    max_epoch,
                    t,
                    iters_per_epoch,
                    running["rtf"] / t,
                    running["keep"] / t,
                    running["ce"] / t,
                    running["tot"] / t,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIP classification unlearning for CIFAR100/ImageNet")
    parser.add_argument("--dataset", choices=["cifar100", "imagenet"], required=True)
    parser.add_argument("--dict-path", required=True, help="Path to Python file containing LABEL_NAMES")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--forget_list", required=True, help="Path to forget indices or filenames list.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch", default="ViT-L-14-336px")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--sample_k", type=int, default=5)
    parser.add_argument("--lambda_attn", type=float, default=1.0)
    parser.add_argument("--lambda_syn", type=float, default=1.0)
    parser.add_argument("--lambda_keep", type=float, default=1.0)
    parser.add_argument("--lambda_uni", type=float, default=1.0)
    parser.add_argument("--concept_token", default="object")
    parser.add_argument("--sam3_mask_dir", required=True)
    parser.add_argument("--sam3_mask_suffix", default=".png")
    parser.add_argument("--reward_arch", default="ViT-L-14")
    parser.add_argument("--reward_process", type=int, default=1)
    parser.add_argument("--process_batch", type=int, default=0)
    parser.add_argument("--reward_amplify", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.output)

    # 加载映射字典
    label_mapping = _load_dict_from_path(args.dict_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_openai_model(os.path.join(os.path.expanduser(
        "/datanfs4/shenruoyan/checkpoints/clip"), f"{args.arch}.pt"), device, jit=False)
    model.float()
    model = model.to(device)
    teacher = copy.deepcopy(model).eval()

    # forget_list = _load_forget_list(args.forget_list)
    # # forget_indices = _build_indices_from_list(
    #     datasets.CIFAR100(root=args.data_root, train=True, download=False)
    #     if args.dataset == "cifar100"
    #     else datasets.ImageNet(root=args.data_root, split="train"),
    #     forget_list,
    # )
    # 1. 加载 JSONL (获取索引和类别集)
    forget_indices_list, forget_classes_set = _load_forget_jsonl(args.forget_list)
    logging.info(f"Loaded {len(forget_indices_list)} forget indices from {args.forget_list}.")
    logging.info("Loaded %d forget classes from JSONL: %s", len(forget_classes_set), sorted(forget_classes_set))

    # 3. 构建数据集 (传入类别集用于过滤保留集)
    df_dataset, dr_dataset, class_names = build_datasets(
        args.dataset,
        args.data_root,
        image_size=model.visual.image_size if isinstance(model.visual.image_size, int) else model.visual.image_size[0],
        forget_indices=forget_indices_list,
        forget_classes=forget_classes_set,
    )

    df_train_dataset = df_dataset
    dr_train_dataset = dr_dataset

    if args.dataset == "cifar100":
        if not forget_classes_set:
            raise ValueError("No forget class loaded from JSONL, cannot build CIFAR100 test split.")
        df_test_dataset, dr_test_dataset = build_cifar100_test_datasets(
            args.data_root,
            image_size=model.visual.image_size if isinstance(model.visual.image_size, int) else model.visual.image_size[0],
            class_names=class_names,
            forget_classes=forget_classes_set,
        )
        logging.info(
            "CIFAR100 test split ready: forget=%d, retain=%d",
            len(df_test_dataset),
            len(dr_test_dataset),
        )
    else:
        raise ValueError("This script currently supports CIFAR100 test-split evaluation only.")

    df_loader = DataLoader(
        df_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dr_loader = DataLoader(
        dr_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    retain_labels = list(_iter_labels(dr_train_dataset.dataset, dr_train_dataset.indices))
    text_pool = _build_text_pool(class_names, retain_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    logging.info("Unlearn configuration: %s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    supervised_unlearn_train(
        model,
        teacher,
        df_loader,
        dr_loader,
        text_pool,
        optimizer,
        scaler,
        sam3_mask_dir=args.sam3_mask_dir,
        sam3_mask_suffix=args.sam3_mask_suffix,
        label_mapping=label_mapping,  # 传入加载的字典
        lambda_rtf=args.lambda_attn,
        lambda_keep=args.lambda_keep,
        lambda_ce=args.lambda_uni,
        max_epoch=args.max_epoch,
        log_interval=50,
    )

    model_path = os.path.join(args.output, "clip_unlearn.pth")
    torch.save(
        {
            "model": model.state_dict(),
            "arch": args.arch,
            "dataset": args.dataset,
        },
        model_path,
    )
    logging.info("已保存模型: %s", model_path)

    df_jsonl = os.path.join(args.output, "topk_cifar100_test_forget.jsonl")
    dr_jsonl = os.path.join(args.output, "topk_cifar100_test_retain.jsonl")
    df_acc = _evaluate_and_dump(
        model,
        df_test_dataset,
        class_names,
        device,
        df_jsonl,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        topk=5,
    )
    dr_acc = _evaluate_and_dump(
        model,
        dr_test_dataset,
        class_names,
        device,
        dr_jsonl,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        topk=5,
    )
    logging.info("遗忘集准确率: %.4f", df_acc)
    logging.info("保留集准确率: %.4f", dr_acc)
    logging.info("已保存最匹配结果: %s, %s", df_jsonl, dr_jsonl)

    forget_labels = sorted(forget_classes_set)
    retain_labels = sorted(set(range(len(class_names))) - set(forget_labels))
    retain_stats = _compute_class_stats(
        model,
        dr_test_dataset,
        class_names,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        topk=5,
    )
    forget_stats = _compute_class_stats(
        model,
        df_test_dataset,
        class_names,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        topk=5,
    )
    metrics_path = os.path.join(args.output, "class_metrics_cifar100_test.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": "cifar100_test",
                "forget_count": len(df_test_dataset),
                "retain_count": len(dr_test_dataset),
                "forget_accuracy": df_acc,
                "retain_accuracy": dr_acc,
                "retain_classes": [retain_stats[idx] for idx in retain_labels if idx in retain_stats],
                "forget_classes": [forget_stats[idx] for idx in forget_labels if idx in forget_stats],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logging.info("已保存分类统计结果: %s", metrics_path)


if __name__ == "__main__":
    main()

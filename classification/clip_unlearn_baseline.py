# coding=utf-8
import argparse
import copy
import importlib.util
import json
import logging
import os
import sys
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from lavis.models.clip_models.model import load_openai_model


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _select_neg_texts_by_minsim(
    teacher,
    images: torch.Tensor,
    text_tokens: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        sim_i2t_t, _, _, _ = _get_logits_and_feats(teacher, images, text_tokens)
        batch_size = sim_i2t_t.size(0)
        if batch_size <= 1:
            return text_tokens
        mask = torch.eye(batch_size, device=sim_i2t_t.device, dtype=sim_i2t_t.dtype) * 1e9
        sim_masked = sim_i2t_t + mask
        neg_idx = torch.argmin(sim_masked, dim=1)
    return text_tokens[neg_idx]


def _select_neg_texts_by_similarity_range(
    teacher,
    images: torch.Tensor,
    text_tokens: torch.Tensor,
    lower_percent: int = 0,
    upper_percent: int = 20,
) -> torch.Tensor:
    with torch.no_grad():
        sim_i2t_t, _, _, _ = _get_logits_and_feats(teacher, images, text_tokens)
        batch_size = sim_i2t_t.size(0)
        if batch_size <= 1:
            return text_tokens
        sim = sim_i2t_t.clone()
        diag = torch.eye(batch_size, device=sim.device, dtype=torch.bool)
        sim[diag] = -float("inf")
        sorted_idx = torch.argsort(sim, dim=1, descending=True)
        span = max(1, batch_size - 1)
        lower_idx = max(0, int(span * lower_percent / 100))
        upper_idx = max(lower_idx + 1, int(span * upper_percent / 100))
        upper_idx = min(upper_idx, span)
        candidates = sorted_idx[:, lower_idx:upper_idx]
        if candidates.numel() == 0:
            candidates = sorted_idx[:, :1]
        rand = torch.randint(0, candidates.size(1), (batch_size,), device=sim.device)
        neg_idx = candidates[torch.arange(batch_size, device=sim.device), rand]
    return text_tokens[neg_idx]


def _load_forget_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_datasets(
    dataset_name: str,
    data_root: str,
    image_size: int,
    forget_indices: Optional[Set[int]] = None,
    forget_classes: Optional[Set[int]] = None,
) -> Tuple[Dataset, Dataset, List[str]]:
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if dataset_name == "cifar100":
        base = datasets.CIFAR100(root=data_root, train=True, transform=tfm, download=False)
        class_names = base.classes
        use_index_path = True
    elif dataset_name == "imagenet":
        base = datasets.ImageNet(root=data_root, split="train", transform=tfm)
        class_names = base.classes
        use_index_path = False
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    forget_indices = set(forget_indices or set())
    forget_classes = forget_classes or set()

    total_indices = list(range(len(base)))
    labels: Optional[Sequence[int]] = getattr(base, "targets", None)
    if labels is None and hasattr(base, "samples"):
        labels = [int(item[1]) for item in base.samples]
    if labels is None:
        labels = [int(base[idx][1]) for idx in total_indices]

    retain_indices = [
        idx for idx in total_indices
        if idx not in forget_indices and int(labels[idx]) not in forget_classes
    ]

    df_dataset = ClassificationDataset(base, class_names, sorted(forget_indices), use_index_path=use_index_path)
    dr_dataset = ClassificationDataset(base, class_names, retain_indices, use_index_path=use_index_path)

    return df_dataset, dr_dataset, class_names


def build_cifar100_test_datasets(
    data_root: str,
    image_size: int,
    class_names: Sequence[str],
    forget_classes: Set[int],
) -> Tuple[Dataset, Dataset]:
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_base = datasets.CIFAR100(root=data_root, train=False, transform=tfm, download=False)
    test_labels: Sequence[int] = getattr(test_base, "targets")

    df_test_indices = [idx for idx, label in enumerate(test_labels) if int(label) in forget_classes]
    dr_test_indices = [idx for idx, label in enumerate(test_labels) if int(label) not in forget_classes]

    df_test_dataset = ClassificationDataset(
        test_base,
        class_names,
        df_test_indices,
        use_index_path=True,
    )
    dr_test_dataset = ClassificationDataset(
        test_base,
        class_names,
        dr_test_indices,
        use_index_path=True,
    )
    return df_test_dataset, dr_test_dataset

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

def supervised_unlearn_train(
    model,
    teacher,
    df_train_loader,
    dr_train_loader,
    optimizer,
    scaler,
    lambda_md: float = 1.0,
    lambda_keep: float = 2.0,
    lambda_uni: float = 0.1,
    max_epoch: int = 1,
    log_interval: int = 50,
    neg_mode: str = "shuffle",
) -> None:
    device = model.device
    mse = nn.MSELoss()
    iters_per_epoch = min(len(df_train_loader), len(dr_train_loader))

    for ep in range(max_epoch):
        model.train()
        df_iter = iter(df_train_loader)
        dr_iter = iter(dr_train_loader)
        running = {"md": 0.0, "keep": 0.0, "uni": 0.0, "tot": 0.0}

        for it in range(iters_per_epoch):
            df_s = next(df_iter)
            dr_s = next(dr_iter)
            img_df = df_s["image"].to(device, non_blocking=True)
            img_dr = dr_s["image"].to(device, non_blocking=True)

            txt_df = df_s.get("text_input", df_s.get("text", None))
            txt_dr = dr_s.get("text_input", dr_s.get("text", None))
            if not isinstance(txt_df, torch.Tensor):
                txt_df = _tokenize_texts(txt_df, device)
            if not isinstance(txt_dr, torch.Tensor):
                txt_dr = _tokenize_texts(txt_dr, device)
            txt_df = txt_df.to(device, non_blocking=True)
            txt_dr = txt_dr.to(device, non_blocking=True)

            batch_size = img_df.size(0)
            if neg_mode == "shuffle":
                perm = torch.randperm(batch_size, device=img_df.device)
                df_text_neg = txt_df[perm]
            elif neg_mode == "simrange":
                df_text_neg = _select_neg_texts_by_similarity_range(
                    teacher, img_df, txt_df, lower_percent=0, upper_percent=20
                )
            else:
                df_text_neg = _select_neg_texts_by_minsim(teacher, img_df, txt_df)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                sim_i2t_df_u, sim_t2i_df_u, _, _ = _get_logits_and_feats(model, img_df, txt_df)
                sim_i2t_df_t_r, sim_t2i_df_t_r, _, _ = _get_logits_and_feats(teacher, img_df, df_text_neg)
                loss_md = mse(sim_i2t_df_u, sim_i2t_df_t_r) + mse(sim_t2i_df_u, sim_t2i_df_t_r)

                sim_i2t_dr_u, sim_t2i_dr_u, img_dr_u, txt_dr_u = _get_logits_and_feats(model, img_dr, txt_dr)
                with torch.no_grad():
                    sim_i2t_dr_t, sim_t2i_dr_t, img_dr_t, txt_dr_t = _get_logits_and_feats(teacher, img_dr, txt_dr)
                loss_keep = mse(sim_i2t_dr_u, sim_i2t_dr_t) + mse(sim_t2i_dr_u, sim_t2i_dr_t)
                loss_uni = mse(img_dr_u, img_dr_t) + mse(txt_dr_u, txt_dr_t)

                loss = lambda_md * loss_md + lambda_keep * loss_keep + lambda_uni * loss_uni

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["md"] += float(loss_md.detach().item())
            running["keep"] += float(loss_keep.detach().item())
            running["uni"] += float(loss_uni.detach().item())
            running["tot"] += float(loss.detach().item())

            if (it + 1) % log_interval == 0:
                t = it + 1
                logging.info(
                    f"[EP {ep+1}/{max_epoch}] it={t}/{iters_per_epoch} "
                    f"md={running['md']/t:.4f} keep={running['keep']/t:.4f} "
                    f"uni={running['uni']/t:.4f} total={running['tot']/t:.4f}"
                )


def supervised_unlearn_train_cliperase(
    cfg,
    model,
    teacher,
    df_train_loader,
    dr_train_loader,
    optimizer,
    scaler,
    lambda_df: float = 1.0,
    lambda_dr: float = 1.0,
    lambda_uni: float = 1.0,
    max_epoch: int = 1,
    log_interval: int = 50,
) -> None:
    device = model.device
    iters_per_epoch = min(len(df_train_loader), len(dr_train_loader))

    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return (x * mask).sum() / denom

    for ep in range(max_epoch):
        model.train()
        df_iter = iter(df_train_loader)
        dr_iter = iter(dr_train_loader)
        running = {"forget": 0.0, "retain": 0.0, "kl": 0.0, "tot": 0.0}

        for it in range(iters_per_epoch):
            df_s = next(df_iter)
            dr_s = next(dr_iter)

            img_df = df_s["image"].to(device, non_blocking=True)
            img_dr = dr_s["image"].to(device, non_blocking=True)

            txt_df = df_s.get("text_input", df_s.get("text", None))
            txt_dr = dr_s.get("text_input", dr_s.get("text", None))
            if not isinstance(txt_df, torch.Tensor):
                txt_df = _tokenize_texts(txt_df, device)
            if not isinstance(txt_dr, torch.Tensor):
                txt_dr = _tokenize_texts(txt_dr, device)
            txt_df = txt_df.to(device, non_blocking=True)
            txt_dr = txt_dr.to(device, non_blocking=True)

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
                logging.info(
                    f"[ClipErase EP {ep+1}/{max_epoch}] it={t}/{iters_per_epoch} "
                    f"forget={running['forget']/t:.4f} "
                    f"retain={running['retain']/t:.4f} "
                    f"kl={running['kl']/t:.4f} "
                    f"total={running['tot']/t:.4f}"
                )

        logging.info(f"[ClipErase EP {ep+1}] done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIP classification unlearning for CIFAR100/ImageNet")
    parser.add_argument("--dataset", choices=["cifar100", "imagenet"], required=True)
    parser.add_argument("--dict-path", required=True, help="Path to Python file containing LABEL_NAMES")
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--method",
        choices=["cliperase", "shuffle", "graddiff", "bbf", "minsim", "simrange"],
        required=True,
    )
    parser.add_argument("--forget_list", required=True, help="Path to forget indices or filenames list.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch", default="ViT-L-14-336px")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--neg_mode", choices=["shuffle", "minsim", "simrange"], default=None)
    parser.add_argument("--lambda_df", type=float, default=1.0)
    parser.add_argument("--lambda_dr", type=float, default=2.0)
    parser.add_argument("--lambda_uni", type=float, default=0.1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.output)
    label_mapping = _load_dict_from_path(args.dict_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.expanduser("/datanfs4/shenruoyan/checkpoints/clip")
    ckpt_path = os.path.join(ckpt_dir, f"{args.arch}.pt")
    model = load_openai_model(ckpt_path, device, jit=False)
    model.float().to(device)

    teacher = copy.deepcopy(model).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # 1. 加载 JSONL (获取索引和类别集)
    forget_indices_list, forget_classes_set = _load_forget_jsonl(args.forget_list)

    logging.info(f"Loaded {len(forget_indices_list)} forget indices from {args.forget_list}.")
    logging.info("Loaded %d forget classes from JSONL: %s", len(forget_classes_set), sorted(forget_classes_set))

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
        raise ValueError("This baseline currently supports CIFAR100 test-split evaluation only.")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    logging.info("Unlearn configuration: %s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    if args.method == "cliperase":
        supervised_unlearn_train_cliperase(
            cfg=args,
            model=model,
            teacher=teacher,
            df_train_loader=df_loader,
            dr_train_loader=dr_loader,
            optimizer=optimizer,
            scaler=scaler,
            lambda_df=args.lambda_df,
            lambda_dr=args.lambda_dr,
            lambda_uni=args.lambda_uni,
            max_epoch=args.max_epoch,
            log_interval=args.log_interval,
        )
    else:
        if args.neg_mode:
            neg_mode = args.neg_mode
        elif args.method == "graddiff":
            neg_mode = "minsim"
        elif args.method == "bbf":
            neg_mode = "simrange"
        elif args.method == "minsim":
            neg_mode = "minsim"
        elif args.method == "simrange":
            neg_mode = "simrange"
        else:
            neg_mode = "shuffle"

        supervised_unlearn_train(
            model=model,
            teacher=teacher,
            df_train_loader=df_loader,
            dr_train_loader=dr_loader,
            optimizer=optimizer,
            scaler=scaler,
            lambda_md=args.lambda_df,
            lambda_keep=args.lambda_dr,
            lambda_uni=args.lambda_uni,
            max_epoch=args.max_epoch,
            log_interval=args.log_interval,
            neg_mode=neg_mode,
        )

    model_path = os.path.join(args.output, "model_final.pth")
    torch.save(
        {
            "model": model.state_dict(),
            "arch": args.arch,
            "dataset": args.dataset,
        },
        model_path,
    )
    logging.info("Saved model: %s", model_path)

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
    logging.info("Saved top-k results: %s, %s", df_jsonl, dr_jsonl)
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

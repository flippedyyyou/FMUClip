import copy
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finegrained.clip_finegrained_baseline import (
    _build_class_name_list,
    _build_eval_transform,
    _build_forget_retain_indices,
    _build_train_datasets,
    _compute_topk_retain_classes,
    _encode_text_features,
    _evaluate_single_accuracy,
    _get_logits_and_feats,
    _labels_to_texts,
    _load_clip_backend,
    run_original_eval,
)
from finegrained.load_dataset import build_test_dataloaders
from finegrained.params import build_parser as build_base_parser


DEFAULT_MASK_DIR = "/datanfs4/shenruoyan/FMUClip/finegrained/mask"


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _encode_image_patches(model, images: torch.Tensor) -> Tuple[torch.Tensor, int]:
    # in:
    #   images: [B, 3, H, W]
    # out:
    #   patch_feats: [B, P, Dv], grid: int (=G), P=G*G
    visual = model.visual
    if not hasattr(visual, "conv1"):
        raise AttributeError("Visual encoder does not expose conv1, cannot extract patch features.")
    x = visual.conv1(images) #x: [B, Dv, G, G]
    grid = x.shape[-1] # grid: G
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat(
        [
            visual.class_embedding.to(x.dtype) # cls token: [B, 1, Dv]
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2) # x: [P+1, B, Dv]   (transformer expected layout)

    # Use penultimate transformer-layer tokens: input to the last block.
    blocks = getattr(visual.transformer, "resblocks", None)
    if blocks is not None and len(blocks) > 0:
        if len(blocks) >= 2:
            for blk in blocks[:1]:
                x = blk(x)
                # each blk keeps shape: [P+1, B, Dv]
        else:
            x = blocks[0](x)
    else:
        # Fallback when block list is not exposed by backend implementation.
        print("Warning: visual transformer blocks not found, using output tokens of ln_pre as patch features.")
        x = visual.transformer(x)

    x = x.permute(1, 0, 2) # x: [B, P+1, Dv]
    # Keep penultimate-layer patch tokens in visual hidden space.
    patch_feats = x[:, 1:, :] # patch_feats: [B, P, Dv] (drop cls token)
    return patch_feats, grid 


def _downsample_mask_to_patch(mask_tensor: torch.Tensor, grid_size: int) -> torch.Tensor:
    # in:  sam3_masks: [B, H, W], grid=G
    # out: patch_attn: [B, G^2]
    mask_tensor = mask_tensor.unsqueeze(1) # [B, 1, H, W]
    # 空间下采样。使用最近邻插值将掩码缩放到 G × G 的网格大小。
    mask_resized = F.interpolate(mask_tensor, size=(grid_size, grid_size), mode="nearest") # [B, 1, G, G]
    # 下采样后的像素点是否属于目标（True 转为 1.0，False 转为 0.0）。
    patch_mask = (mask_resized.squeeze(1) > 0.5).float() # [B, G, G]
    return patch_mask.flatten(1) # [B, G^2]


def _compute_patch_importance(
    patch_feats: torch.Tensor,
    text_feats: torch.Tensor,
    logit_scale: float = 10.0,
) -> torch.Tensor:
    # in:  patch_feats: [B, G^2, Dv], text_feats: [B, D]
    # out: patch_weights: [B, G^2]
    patch_norm = F.normalize(patch_feats, dim=-1) # [B, G^2, Dv]
    text_norm = F.normalize(text_feats, dim=-1).unsqueeze(1) # [B, 1, D]
    patch_scores = text_norm @ patch_norm.transpose(-2, -1) # [B, 1, G^2]
    patch_scores = patch_scores.squeeze(1)  # [B, G^2]
    # squashes the range from 0 to 1 for every element
    return torch.sigmoid(patch_scores * logit_scale)


def _weighted_pool_with_threshold(
    patch_feats: torch.Tensor,
    patch_weights: torch.Tensor,
    target_patch_mask: torch.Tensor,
    select_threshold: torch.Tensor,
) -> torch.Tensor:
    # in:  patch_feats: [B, G^2, Dv], patch_weights: [B, G^2], target_patch_mask: [B, G^2], select_threshold: [B, 1]
    # out: pooled_feats: [B, Dv]
    # Only target patches are allowed to contribute; non-target patches are forced to 0.
    target_mask = (target_patch_mask > 0.5).float()
    keep_mask = (patch_weights > select_threshold).float()
    eff_weights = patch_weights * keep_mask * target_mask # [B, G^2]
    valid = eff_weights.sum(dim=1, keepdim=True)
    fallback = valid <= 1e-6
    if fallback.any():
        # If no target patch passes threshold, fallback to weighted target patches only.
        fallback_weights = patch_weights * target_mask
        eff_weights = torch.where(fallback, fallback_weights, eff_weights)
        valid = eff_weights.sum(dim=1, keepdim=True)
    pooled = (patch_feats * eff_weights.unsqueeze(-1)).sum(dim=1) / valid.clamp_min(1e-6)
    return pooled


def _compute_sam3_min_threshold(
    patch_weights: torch.Tensor,
    target_patch: torch.Tensor,
    fallback_threshold: float,
) -> torch.Tensor:
    # Per-sample threshold = minimum weight among SAM3-covered target patches.
    target_mask = target_patch > 0.5
    inf_fill = torch.full_like(patch_weights, float("inf"))
    target_weights = torch.where(target_mask, patch_weights, inf_fill) #目标区域保留原权重，非目标区域填充负无穷
    min_target = target_weights.min(dim=1, keepdim=True).values #选目标区域内的最小值为下限
    has_target = target_mask.any(dim=1, keepdim=True) #[B, 1] (bool), 标记是否有目标patch
    #如果有目标则用 min_target，否则用默认sam3_min_weight_threshold
    fallback = torch.full_like(min_target, fallback_threshold) 
    return torch.where(has_target, min_target, fallback)


def _build_mask_index(mask_root: str, mask_suffix: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for root, _, files in os.walk(mask_root):
        for file_name in files:
            if not file_name.endswith(mask_suffix):
                continue
            stem = os.path.splitext(file_name)[0]
            if stem not in idx:
                idx[stem] = os.path.join(root, file_name)
    return idx


def _load_sam3_masks(
    image_paths: Sequence[str],
    mask_index: Dict[str, str],
    target_size: int,
) -> torch.Tensor:
    masks = []
    for path in image_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        mask_path = mask_index.get(stem)
        if not mask_path:
            raise FileNotFoundError(f"SAM3 mask not found for image stem: {stem}")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((target_size, target_size), resample=Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        masks.append(mask_tensor)
    return torch.stack(masks, dim=0)


def _build_text_pool(class_names: Sequence[str], retain_indices: Sequence[int]) -> List[str]:
    return [f"a photo of {class_names[idx].replace('_', ' ')}" for idx in retain_indices]


def train_one_epoch(
    model,
    teacher,
    tokenizer_fn,
    df_loader,
    dr_loader,
    mask_index: Dict[str, str],
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
    patch_importance_scale: float,
    sam3_min_weight_threshold: float,
    epoch_idx: int,
    max_epoch: int,
    log_interval: int,
) -> Dict[str, float]:
    device = _get_model_device(model)
    mse = nn.MSELoss()
    iters_per_epoch = min(len(df_loader), len(dr_loader))
    model.train()
    df_iter = iter(df_loader)
    dr_iter = iter(dr_loader)
    with torch.no_grad():
        text_pool_features = _encode_text_features(
            model,
            _build_text_pool(class_names, retain_indices),
            tokenizer_fn,
            device,
        )

    running = {"rtf": 0.0, "keep": 0.0, "ce": 0.0, "tot": 0.0}
    for it in range(iters_per_epoch):
        df_s = next(df_iter)
        dr_s = next(dr_iter)

        img_df = df_s["image"].to(device, non_blocking=True) #img_df: [B, 3, H, W]
        img_dr = dr_s["image"].to(device, non_blocking=True)
        df_paths = df_s["image_path"]

        txt_df = _labels_to_texts(df_s["label"], class_names, forget_indices)
        txt_dr = _labels_to_texts(dr_s["label"], class_names, retain_indices)
        df_tokens = tokenizer_fn(txt_df).to(device) # df_tokens: [B, L] (L是token长度)
        dr_tokens = tokenizer_fn(txt_dr).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            sam3_masks = _load_sam3_masks(df_paths, mask_index, img_df.shape[-1]).to(device, non_blocking=True) #sam3_masks: [B, H, W]
            patch_feats, grid = _encode_image_patches(model, img_df) # patch_feats: [B, G^2, Dv], grid: G
            df_text_feats = model.encode_text(df_tokens) # df_text_feats: [B, D]
            df_text_feats = df_text_feats / df_text_feats.norm(dim=-1, keepdim=True) # [B, D]
            target_patch = _downsample_mask_to_patch(sam3_masks, grid) # [B, G^2]
            patch_feats_for_weight = patch_feats
            if getattr(model.visual, "proj", None) is not None:
                patch_feats_for_weight = patch_feats_for_weight @ model.visual.proj  # [B, G^2, D]
            patch_weights = _compute_patch_importance( # [B, G^2]
                patch_feats=patch_feats_for_weight,
                text_feats=df_text_feats,
                logit_scale=patch_importance_scale,
            )
            # Auto threshold per sample from SAM3 target patches:
            # [B, 1]. min weight among target patches. If no target patch exists, fallback to sam3_min_weight_threshold.
            sam3_dynamic_threshold = _compute_sam3_min_threshold(
                patch_weights=patch_weights,
                target_patch=target_patch,
                fallback_threshold=sam3_min_weight_threshold,
            ) 
            pooled_patch_feats = _weighted_pool_with_threshold(
                patch_feats=patch_feats,
                patch_weights=patch_weights,
                target_patch_mask=target_patch,
                select_threshold=sam3_dynamic_threshold,
            )# [B, Dv]

            if getattr(model.visual, "proj", None) is not None:
                pooled_patch_feats = pooled_patch_feats @ model.visual.proj
            pooled_patch_feats = F.normalize(pooled_patch_feats, dim=-1)

            contrast_text_feats = torch.cat([df_text_feats, text_pool_features], dim=0)
            logit_scale = model.logit_scale.exp()
            labels = torch.arange(img_df.size(0), device=device)

            logits_i2t = logit_scale * (pooled_patch_feats @ contrast_text_feats.t())
            loss_rtf_i2t = F.cross_entropy(logits_i2t, labels)

            logits_t2i = logit_scale * (df_text_feats @ pooled_patch_feats.t())
            loss_rtf_t2i = F.cross_entropy(logits_t2i, labels)

            loss_rtf = -0.5 * (loss_rtf_i2t + loss_rtf_t2i)

            sim_i2t_dr_u, sim_t2i_dr_u, _, _ = _get_logits_and_feats(model, img_dr, dr_tokens)
            with torch.no_grad():
                sim_i2t_dr_t, sim_t2i_dr_t, _, _ = _get_logits_and_feats(teacher, img_dr, dr_tokens)
            loss_keep = mse(sim_i2t_dr_u, sim_i2t_dr_t) + mse(sim_t2i_dr_u, sim_t2i_dr_t)

            retain_targets = torch.arange(img_dr.size(0), device=device)
            loss_ce = F.cross_entropy(sim_i2t_dr_u, retain_targets) + F.cross_entropy(sim_t2i_dr_u, retain_targets)

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
    prev_return_meta = args.return_meta
    args.return_meta = True
    forget_loader, retain_loader = build_test_dataloaders(args, transform=eval_transform)
    args.return_meta = prev_return_meta

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
    parser.add_argument("--patch_importance_scale", type=float, default=10.0)
    parser.add_argument("--sam3_min_weight_threshold", type=float, default=0.5)
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
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    args.return_meta = True
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

    mask_index = _build_mask_index(args.sam3_mask_dir, args.sam3_mask_suffix)
    if not mask_index:
        raise FileNotFoundError(f"No mask files found under: {args.sam3_mask_dir}")

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
            teacher=teacher,
            tokenizer_fn=tokenize_fn,
            df_loader=df_loader,
            dr_loader=dr_loader,
            mask_index=mask_index,
            class_names=class_names,
            forget_indices=forget_indices,
            retain_indices=retain_indices,
            optimizer=optimizer,
            scaler=scaler,
            lambda_rtf=args.lambda_rtf,
            lambda_keep=args.lambda_keep,
            lambda_ce=args.lambda_ce,
            sample_k=args.sample_k,
            patch_importance_scale=args.patch_importance_scale,
            sam3_min_weight_threshold=args.sam3_min_weight_threshold,
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
    run_original_eval(
        args,
        model=model,
        tokenize_fn=tokenize_fn,
        image_size=image_size,
        backend=backend,
        retain_topk_indices=retain_topk_indices,
    )


if __name__ == "__main__":
    main()

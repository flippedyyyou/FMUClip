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
    _get_logits_and_feats,
    _labels_to_texts,
    _load_clip_backend,
    run_original_eval,
)
from finegrained.params import build_parser as build_base_parser


DEFAULT_MASK_DIR = "/home/shenruoyan/FMUClip/finegrained/mask"


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
            for blk in blocks[:-1]:
                x = blk(x)
        else:
            x = blocks[0](x)
    else:
        # Fallback when block list is not exposed by backend implementation.
        print("Warning: visual transformer blocks not found, using output tokens of ln_pre as patch features.")
        x = visual.transformer(x)

    x = x.permute(1, 0, 2)
    # Keep semantics closer to CLIP grad-cam penultimate activations:
    # do not pass through final ln_post head; only project if needed for text-space matching.
    if visual.proj is not None:
        x = x @ visual.proj
    patch_feats = x[:, 1:, :]
    patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)
    return patch_feats, grid


def _mask_to_patch_attention(mask_tensor: torch.Tensor, grid_size: int) -> torch.Tensor:
    mask_tensor = mask_tensor.unsqueeze(1)
    mask_resized = F.interpolate(mask_tensor, size=(grid_size, grid_size), mode="nearest")
    patch_mask = (mask_resized.squeeze(1) > 0.5).float()
    return patch_mask.flatten(1)


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


def supervised_unlearn_train(
    model,
    teacher,
    tokenizer_fn,
    df_loader,
    dr_loader,
    text_pool_tokens: torch.Tensor,
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
    max_epoch: int,
    log_interval: int,
) -> None:
    device = _get_model_device(model)
    mse = nn.MSELoss()
    iters_per_epoch = min(len(df_loader), len(dr_loader))

    for ep in range(max_epoch):
        model.train()
        df_iter = iter(df_loader)
        dr_iter = iter(dr_loader)
        with torch.no_grad():
            text_pool_features = _encode_text_features(model, _build_text_pool(class_names, retain_indices), tokenizer_fn, device)

        running = {"rtf": 0.0, "keep": 0.0, "ce": 0.0, "tot": 0.0}
        for it in range(iters_per_epoch):
            df_s = next(df_iter)
            dr_s = next(dr_iter)

            img_df = df_s["image"].to(device, non_blocking=True)
            img_dr = dr_s["image"].to(device, non_blocking=True)
            df_paths = df_s["image_path"]

            txt_df = _labels_to_texts(df_s["label"], class_names, forget_indices)
            txt_dr = _labels_to_texts(dr_s["label"], class_names, retain_indices)
            df_tokens = tokenizer_fn(txt_df).to(device)
            dr_tokens = tokenizer_fn(txt_dr).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                sam3_masks = _load_sam3_masks(df_paths, mask_index, img_df.shape[-1]).to(device, non_blocking=True)
                patch_feats, grid = _encode_image_patches(model, img_df)
                df_text_feats = model.encode_text(df_tokens)
                df_text_feats = df_text_feats / df_text_feats.norm(dim=-1, keepdim=True)

                patch_sim = F.cosine_similarity(patch_feats, df_text_feats.unsqueeze(1), dim=-1)
                patch_attn = _mask_to_patch_attention(sam3_masks, grid)
                attn_sum = patch_attn.sum(dim=1)
                masked_similarity = patch_sim * patch_attn
                loss_rtf = (masked_similarity.sum(dim=1) / attn_sum.clamp(min=1.0)).mean()

                # syn_mask = 1.0 - (sam3_masks > 0.5).float()
                # syn_img = img_df * syn_mask.unsqueeze(1)

                # syn_img_feats = model.encode_image(syn_img)
                # syn_img_feats = syn_img_feats / syn_img_feats.norm(dim=-1, keepdim=True)
                # clip_scores = model.logit_scale.exp() * (syn_img_feats @ text_pool_features.t())
                # clip_probs = torch.softmax(clip_scores, dim=-1)
                # cur_k = min(sample_k, clip_scores.size(1))
                # topk_probs, indices = torch.topk(clip_probs, cur_k, dim=-1, largest=True)
                # adv = topk_probs.detach()

                # sim_i2t_syn_u = model.logit_scale.exp() * (syn_img_feats @ text_pool_features.t())
                # rep_output = torch.repeat_interleave(sim_i2t_syn_u, cur_k, dim=0)
                # text_index = indices.flatten()
                # ce = F.cross_entropy(rep_output, text_index, reduction="none").view(img_df.size(0), cur_k)
                # loss_syn = (adv * ce).mean()

                sim_i2t_dr_u, sim_t2i_dr_u, img_dr_u, txt_dr_u = _get_logits_and_feats(model, img_dr, dr_tokens)
                with torch.no_grad():
                    sim_i2t_dr_t, sim_t2i_dr_t, img_dr_t, txt_dr_t = _get_logits_and_feats(teacher, img_dr, dr_tokens)
                loss_keep = mse(sim_i2t_dr_u, sim_i2t_dr_t) + mse(sim_t2i_dr_u, sim_t2i_dr_t)

                retain_targets = torch.arange(img_dr.size(0), device=device)
                loss_ce = F.cross_entropy(sim_i2t_dr_u, retain_targets) + F.cross_entropy(sim_t2i_dr_u, retain_targets)

                loss = (
                    lambda_rtf * loss_rtf
                    # + lambda_syn * loss_syn
                    + lambda_keep * loss_keep
                    + lambda_ce * loss_ce
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["rtf"] += float(loss_rtf.detach().item())
            # running["syn"] += float(loss_syn.detach().item())
            running["keep"] += float(loss_keep.detach().item())
            running["ce"] += float(loss_ce.detach().item())
            running["tot"] += float(loss.detach().item())

            if (it + 1) % log_interval == 0:
                t = it + 1
                print(
                    f"[Unlearn EP {ep+1}/{max_epoch}] it={t}/{iters_per_epoch} "
                    f"rtf={running['rtf']/t:.4f}"
                    f"keep={running['keep']/t:.4f} ce={running['ce']/t:.4f} "
                    f"total={running['tot']/t:.4f}"
                )


def build_parser():
    parser = build_base_parser()
    parser.description = "Finegrained CLIP unlearning (SAM3-mask guided), COCO split logic."
    parser.add_argument("--lambda_rtf", type=float, default=3.0)
    parser.add_argument("--lambda_keep", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=3.0)
    parser.add_argument("--sample_k", type=int, default=5)
    parser.add_argument("--sam3_mask_dir", type=str, default=DEFAULT_MASK_DIR)
    parser.add_argument("--sam3_mask_suffix", type=str, default=".png")
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
    text_pool = _build_text_pool(class_names, retain_indices)
    text_pool_tokens = tokenize_fn(text_pool).to(device)

    mask_index = _build_mask_index(args.sam3_mask_dir, args.sam3_mask_suffix)
    if not mask_index:
        raise FileNotFoundError(f"No mask files found under: {args.sam3_mask_dir}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    supervised_unlearn_train(
        model=model,
        teacher=teacher,
        tokenizer_fn=tokenize_fn,
        df_loader=df_loader,
        dr_loader=dr_loader,
        text_pool_tokens=text_pool_tokens,
        mask_index=mask_index,
        class_names=class_names,
        forget_indices=forget_indices,
        retain_indices=retain_indices,
        optimizer=optimizer,
        scaler=scaler,
        lambda_rtf=args.lambda_rtf,
        # lambda_syn=args.lambda_syn,
        lambda_keep=args.lambda_keep,
        lambda_ce=args.lambda_ce,
        sample_k=args.sample_k,
        max_epoch=args.max_epoch,
        log_interval=args.log_interval,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "clip_unlearn_finegrained.pth")
    torch.save(
        {
            "model": model.state_dict(),
            "clip_arch": args.clip_arch,
            "clip_pretrained": args.clip_pretrained,
        },
        ckpt_path,
    )
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

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

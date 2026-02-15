# coding=utf-8
from retrieval_utils import (
    _tokenize_texts,
    _select_neg_texts_by_minsim,
    _resolve_clip_model,
    _save_unlearned_checkpoint,
    prepare_dr_data,
    prepare_df_data,
    prepare_df_data_for_test,
    prepare_dr_data_for_test,
    _get_logits_and_feats,
    _load_forget_train_ids,
    _load_forget_test_ids,
    eval_split_no_tta,
    _select_neg_texts_by_proximal,
    _dump_detailed_topk_results
)
import contextlib
import math
import os
import json
import time
import random
import logging
import datetime
from tqdm import tqdm
import copy
import sys

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms


import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now
from lavis.common.logger import MetricLogger

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from torch.utils.data import DataLoader
from torch import nn
from lavis.tasks import *

from params import parse_args
from clip_unlearn_reward import get_reward_model
from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_TTA
from torch import nn

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def _evaluate_cifar100_forget_retain(
    model,
    data_root: str,
    forget_class: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    output_dir: str,
):
    device = model.device
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = datasets.CIFAR100(
        root=data_root, train=False, transform=tfm, download=False
    )
    class_names = [c.replace("_", " ") for c in dataset.classes]

    forget_class_norm = _normalize_class_name(forget_class)
    class_map = { _normalize_class_name(n): i for i, n in enumerate(class_names) }
    if forget_class_norm not in class_map:
        raise ValueError(
            f"Forget class '{forget_class}' not found in CIFAR-100 classes."
        )
    forget_idx = class_map[forget_class_norm]

    tokens = _tokenize_texts(class_names).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logit_scale = model.logit_scale.exp()
    correct_forget = 0
    total_forget = 0
    correct_retain = 0
    total_retain = 0
    per_sample_results = []
    topk = min(10, len(class_names))
    sample_index = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            img_features = model.encode_image(images)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (img_features @ text_features.t())
            preds = logits.argmax(dim=1)
            topk_scores, topk_indices = torch.topk(logits, k=topk, dim=1)

            labels_cpu = labels.detach().cpu().tolist()
            preds_cpu = preds.detach().cpu().tolist()
            topk_scores_cpu = topk_scores.detach().cpu().tolist()
            topk_indices_cpu = topk_indices.detach().cpu().tolist()

            for i, (label_idx, pred_idx, one_topk_idx, one_topk_score) in enumerate(
                zip(labels_cpu, preds_cpu, topk_indices_cpu, topk_scores_cpu)
            ):
                top10_predictions = []
                for cls_idx, cls_score in zip(one_topk_idx, one_topk_score):
                    top10_predictions.append(
                        {
                            "class_index": int(cls_idx),
                            "class_name": class_names[int(cls_idx)],
                            "score": float(cls_score),
                        }
                    )

                per_sample_results.append(
                    {
                        "sample_index": int(sample_index + i),
                        "label_index": int(label_idx),
                        "label_name": class_names[int(label_idx)],
                        "pred_index": int(pred_idx),
                        "pred_name": class_names[int(pred_idx)],
                        "pred_score": float(one_topk_score[0]),
                        "top10_predictions": top10_predictions,
                    }
                )
            sample_index += len(labels_cpu)

            forget_mask = labels == forget_idx
            retain_mask = ~forget_mask

            if forget_mask.any():
                correct_forget += (preds[forget_mask] == labels[forget_mask]).sum().item()
                total_forget += int(forget_mask.sum().item())
            if retain_mask.any():
                correct_retain += (preds[retain_mask] == labels[retain_mask]).sum().item()
                total_retain += int(retain_mask.sum().item())

    acc_forget = correct_forget / total_forget if total_forget else 0.0
    acc_retain = correct_retain / total_retain if total_retain else 0.0
    forget_rate = 1.0 - acc_forget

    results = {
        "dataset": "cifar100",
        "forget_class": class_names[forget_idx],
        "forget_class_index": forget_idx,
        "forget_accuracy": round(acc_forget, 6),
        "forget_rate": round(forget_rate, 6),
        "retain_accuracy": round(acc_retain, 6),
        "forget_count": int(total_forget),
        "retain_count": int(total_retain),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results_cifar100_forget_retain.json")
    detail_out_path = os.path.join(output_dir, "results_cifar100_test_predictions_top10.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4)
    with open(detail_out_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "dataset": "cifar100",
                "num_samples": len(per_sample_results),
                "predictions": per_sample_results,
            },
            fp,
            indent=4,
            ensure_ascii=False,
        )
    logging.info(f"[CIFAR100] {results}")
    logging.info(f"[CIFAR100] Saved results to: {out_path}")
    logging.info(f"[CIFAR100] Saved detailed predictions to: {detail_out_path}")


# 读取图片路径

def load_image_paths(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# 准备自定义数据集


def prepare_custom_dataset(image_paths, caption_data):
    dataset = []
    for img_path in image_paths:
        captions = caption_data.get(img_path, [])  # 获取该图片路径的所有 caption
        dataset.append({"image": img_path, "caption": captions})
    return dataset


def _is_hf_clip(model):
    # HuggingFace Transformers: CLIPModel
    return hasattr(model, "get_image_features") and hasattr(model, "get_text_features")


def _is_openai_clip(model):
    # openai/clip: has encode_image/encode_text and a .logit_scale Parameter
    return hasattr(model, "encode_image") and hasattr(model, "encode_text") and hasattr(model, "logit_scale")


def _is_open_clip(model):
    # open_clip-pytorch: encode_image/encode_text 且通常也有 logit_scale
    return hasattr(model, "encode_image") and hasattr(model, "encode_text")


def _encode_image(model, images: torch.Tensor):
    feats = model.encode_image(images)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _encode_text(model, text_tokens: torch.Tensor):
    # 新版：显式要求传入 LongTensor 的 token ids（batch, seq_len）
    # 兼容类型与设备
    if not isinstance(text_tokens, torch.Tensor):
        raise TypeError("text_tokens 必须是 Tensor（'text_input'），而不是 list[str]")
    feats = model.encode_text(text_tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _get_logit_scale(model):
    # HF: model.logit_scale；openai/clip & open_clip: 通常也是 model.logit_scale
    ls = getattr(model, "logit_scale", None)
    if ls is None:
        # 极少数封装：挂在子模块里
        for child in model.modules():
            if hasattr(child, "logit_scale"):
                ls = child.logit_scale
                break
    if ls is None:
        raise AttributeError("Could not find logit_scale on model.")
    # 有的是 nn.Parameter，有的是 Tensor，统一 exp()
    return ls.exp() if torch.is_tensor(ls) or isinstance(ls, torch.nn.Parameter) else torch.tensor(ls).exp()


def _encode_image_patches(model, images: torch.Tensor):
    clip_model = _resolve_clip_model(model)
    if not hasattr(clip_model, "visual"):
        raise AttributeError(
            "CLIP model does not expose a visual encoder for patch extraction.")
    visual = clip_model.visual
    if not hasattr(visual, "conv1"):
        raise AttributeError(
            "Visual encoder does not expose patch embeddings.")

    x = visual.conv1(images)
    grid = x.shape[-1]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat(
        [
            visual.class_embedding.to(x.dtype) #CLS
            + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x,
        ],
        dim=1,
    )
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj

    patch_feats = x[:, 1:, :]
    patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)
    return patch_feats, grid


def _mask_to_patch_attention(mask_tensor: torch.Tensor, grid_size: int):
    mask_tensor = mask_tensor.unsqueeze(1)
    mask_resized = F.interpolate(mask_tensor, size=(
        grid_size, grid_size), mode="nearest")
    patch_mask = (mask_resized.squeeze(1) > 0.5).float()
    return patch_mask.flatten(1)


@torch.no_grad()
def _frozen_init_image_feats(model, images):
    """
    取当前 initial_state（未进行本轮TTA更新前）的图像特征，作为“原模型特征”做保持正则。
    """
    model.eval()
    feats = model.get_image_features(images)
    return feats.detach()


@torch.no_grad()
def _frozen_init_text_feats(model, text=None, tokenized_prompts=None):
    """
    取当前 initial_state 的文本特征，用于 Dr 漂移约束（只在文本侧 TTA 用）。
    """
    model.eval()
    feats = model.get_text_features(
        text=text, tokenized_prompts=tokenized_prompts)
    return feats.detach()


def _fmt_topk_rows(indices: torch.Tensor, scores: torch.Tensor, mapper):
    """
    indices: [K] LongTensor
    scores:  [K] Tensor
    mapper:  callable(int) -> str  把 id 映射成文本或图片路径
    return:  多行字符串，每行: rank. id  score  content
    """
    lines = []
    for r, (idx, sc) in enumerate(zip(indices.tolist(), scores.tolist()), start=1):
        content = mapper(idx)
        lines.append(f"{r:>2}. id={idx:<6} score={sc:.4f}  {content}")
    return "\n".join(lines)


def _maybe_to_device(batch, device):
    if batch is None:
        return None
    if hasattr(batch, "to"):
        batch = batch.to(device)
    return batch


def _normalize_text_input(text_value):
    if text_value is None:
        return None
    if isinstance(text_value, (list, tuple)):
        if len(text_value) == 1:
            return text_value[0]
        return list(text_value)
    return text_value


def _resolve_text_inputs(dataset, index, samples, device):
    raw_text = None
    tokenized = None

    if isinstance(samples, dict):
        for key in ("text", "text_input", "caption", "captions"):
            if key in samples and samples[key] is not None:
                raw_text = samples[key]
                break

        for key in ("tokenized_prompts", "tokenized_text", "tokenized_caption", "tokenized_captions"):
            if key in samples and samples[key] is not None:
                tokenized = samples[key]
                break

        if tokenized is None:
            candidate = samples.get("input_ids") or samples.get("tokenized")
            if candidate is not None:
                tokenized = candidate

    if raw_text is None and hasattr(dataset, "text") and len(dataset.text) > index:
        raw_text = dataset.text[index]

    if tokenized is None and hasattr(dataset, "tokenized_prompts"):
        tokenized = dataset.tokenized_prompts[index]

    if isinstance(tokenized, dict):
        tokenized = tokenized.get("input_ids")

    tokenized = _maybe_to_device(tokenized, device)
    if tokenized is not None and hasattr(tokenized, "dim") and tokenized.dim() == 1:
        tokenized = tokenized.unsqueeze(0)

    raw_text = _normalize_text_input(raw_text)

    if raw_text is None and tokenized is None:
        raise ValueError(f"No textual inputs found for sample index {index}.")

    return raw_text, tokenized


# def _dump_topk_results(scores_np, loader, task, out_file, k=10):
#     """
#     将评测阶段的 score 矩阵导出为 JSONL：
#     - I2T: 每张图像的 Top-K 文本（含 id、score、内容）
#     - T2I: 每条文本的 Top-K 图像（含 id、score、路径）
#     """
#     os.makedirs(os.path.dirname(out_file), exist_ok=True)
#     import json
#     K = max(1, k)
#     scores = torch.from_numpy(scores_np)  # [Nq, Nc]
#     # id→内容映射
#     if task == "i2t":
#         id2cand = lambda j: loader.dataset.text[j]
#         query_list = getattr(loader.dataset, "image", [f"#{i}" for i in range(scores.size(0))])
#     else:
#         id2cand = lambda j: loader.dataset.image[j]
#         query_list = getattr(loader.dataset, "text", [f"#{i}" for i in range(scores.size(0))])

#     with open(out_file, "w", encoding="utf-8") as f:
#         for i in range(scores.size(0)):
#             row = scores[i]                               # [Nc]
#             v, idx = torch.topk(row, min(K, row.numel())) # (K), (K)
#             items = []
#             for r, (jj, sc) in enumerate(zip(idx.tolist(), v.tolist()), start=1):
#                 items.append({"rank": r, "id": jj, "score": float(sc), "content": id2cand(jj)})
#             obj = {"query_id": i, "query": query_list[i], "topk": items}
#             f.write(json.dumps(obj, ensure_ascii=False) + "\n")
#     logging.info(f"[TopK] Saved Top-{K} results to {out_file}")

def supervised_unlearn_train(
    cfg, model, teacher,
    df_train_loader, dr_train_loader,
    optimizer, scaler,
    lambda_md=1.0,          # Df 的“多模态解耦”损失（unlearn vs teacher@shuffled）
    lambda_keep=2.0,        # Dr 的“多模态保持”损失（unlearn vs teacher@matched）
    lambda_uni=0.1,         # 单模态保持（img/text embedding vs teacher）
    max_epoch=1,
    log_interval=50,
    neg_mode="shuffle",     # 新增：'shuffle' 或 'minsim'
    concept_token="dog",
    sam3_mask_dir=None,
    sam3_mask_suffix=".png",
):
    device = model.device
    mse = nn.MSELoss()

    if not sam3_mask_dir:
        raise ValueError(
            "sam3_mask_dir must be provided to compute attention-guided loss.")
    concept_tokens = _tokenize_texts([concept_token]).to(device)
    clip_model = _resolve_clip_model(model)
    visual = clip_model.visual if hasattr(clip_model, "visual") else None
    if visual is None or not hasattr(visual, "image_size"):
        raise AttributeError(
            "CLIP visual encoder does not expose image_size for SAM3 mask mapping.")
    image_size = visual.image_size[0] if isinstance(
        visual.image_size, (tuple, list)) else visual.image_size

    # 以较短的一方为 epoch 基础步数
    iters_per_epoch = min(len(df_train_loader), len(dr_train_loader))
    for ep in range(max_epoch):
        model.train()
        df_iter = iter(df_train_loader)
        dr_iter = iter(dr_train_loader)
        running = {"md": 0.0, "keep": 0.0, "uni": 0.0, "tot": 0.0}
        for it in range(iters_per_epoch):
            # ===== 1) 取 batch =====
            df_s = next(df_iter)
            dr_s = next(dr_iter)
            img_df = df_s["image"].to(device, non_blocking=True)
            img_dr = dr_s["image"].to(device, non_blocking=True)
            # ✅ 直接使用 dataloader 提供的 token ids（LongTensor）：
            # 兼容 list[str] 或 LongTensor；若是 list[str] 则现场 tokenize 成 [B, L] LongTensor
            txt_df = df_s.get("text_input", df_s.get("text", None))
            txt_dr = dr_s.get("text_input", dr_s.get("text", None))
            if not isinstance(txt_df, torch.Tensor):
                txt_df = _tokenize_texts(txt_df)          # -> LongTensor[B, L]
            if not isinstance(txt_dr, torch.Tensor):
                txt_dr = _tokenize_texts(txt_dr)          # -> LongTensor[B, L]
            txt_df = txt_df.to(device, non_blocking=True)
            txt_dr = txt_dr.to(device, non_blocking=True)

            B = img_df.size(0)

            if neg_mode == "shuffle":
                # ✅ baseline：随机打乱
                perm = torch.randperm(B).cpu()
                df_text_neg = [txt_df[i] for i in perm.tolist()]
            elif neg_mode == "simrange":
                # ✅ ours：对每张图像选 teacher 相似度在某范围内的文本
                df_text_neg = _select_neg_texts_by_proximal(
                    teacher, img_df, txt_df)
            else:
                # ✅ ours：对每张图像选 teacher 最不相似的文本
                df_text_neg = _select_neg_texts_by_minsim(
                    teacher, img_df, txt_df)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                # --- 遗忘模型：Df（正确配对）的多模态相似度 ---
                sim_i2t_df_u, sim_t2i_df_u, img_df_u, txt_df_u = _get_logits_and_feats(
                    model, img_df, txt_df
                )
                # --- 原模型：Df（错配文本）的多模态相似度 ---
                sim_i2t_df_t_r, sim_t2i_df_t_r, img_df_t, txt_df_t_r = _get_logits_and_feats(
                    teacher, img_df, df_text_neg
                )
                # 多模态“解耦”损失（把 unlearn 的正确配对，拉向 teacher 的错配分布）
                loss_md = mse(sim_i2t_df_u, sim_i2t_df_t_r) + \
                    mse(sim_t2i_df_u, sim_t2i_df_t_r)

                # ===== 3) Dr：保持一致（与原模型 matched 分布接近）=====
                sim_i2t_dr_u, sim_t2i_dr_u, img_dr_u, txt_dr_u = _get_logits_and_feats(
                    model, img_dr, txt_dr
                )
                with torch.no_grad():
                    sim_i2t_dr_t, sim_t2i_dr_t, img_dr_t, txt_dr_t = _get_logits_and_feats(
                        teacher, img_dr, txt_dr
                    )
                loss_keep = mse(sim_i2t_dr_u, sim_i2t_dr_t) + \
                    mse(sim_t2i_dr_u, sim_t2i_dr_t)

                # ===== 4) 单模态保持（避免 encoder 漂移过大）=====
                loss_uni = mse(img_dr_u, img_dr_t) + mse(txt_dr_u, txt_dr_t)

                loss = lambda_md * loss_md + lambda_keep * loss_keep + lambda_uni * loss_uni

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # log
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
        logging.info(f"[EP {ep+1}] done.")


def supervised_unlearn_train_cliperase(
    cfg, model, teacher,
    df_train_loader, dr_train_loader,
    optimizer, scaler,
    lambda_df=1.0,          # Df 部分（forget）权重
    lambda_dr=1.0,          # Dr 部分（retain）权重
    lambda_uni=1.0,         # KL loss 权重
    max_epoch=1,
    log_interval=50,
):
    """
    ClipErase 风格 supervised baseline：
    - Df: CE loss 取负号（让模型在遗忘集上“学坏”）
    - Dr: CE loss 正常，加上 KL(unlearn || teacher) 约束
    - 每个 step = Df/Dr/ KL 三者加权求和的 total loss 更新一次
    """
    device = model.device
    iters_per_epoch = min(len(df_train_loader), len(dr_train_loader))

    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x, mask: [B]
        denom = mask.sum().clamp_min(1.0)
        return (x * mask).sum() / denom

    for ep in range(max_epoch):
        model.train()
        df_iter = iter(df_train_loader)
        dr_iter = iter(dr_train_loader)

        running = {"forget": 0.0, "retain": 0.0, "kl": 0.0, "tot": 0.0}

        for it in range(iters_per_epoch):
            # ===== 1) 取 Df / Dr batch =====
            df_s = next(df_iter)
            dr_s = next(dr_iter)

            img_df = df_s["image"].to(device, non_blocking=True)
            img_dr = dr_s["image"].to(device, non_blocking=True)

            # 文本：优先用已经 tokenized 的 text_input / tokenized_prompts，没有就 raw text + tokenizer
            txt_df = df_s.get("text_input", df_s.get("text", None))
            txt_dr = dr_s.get("text_input", dr_s.get("text", None))

            if not isinstance(txt_df, torch.Tensor):
                txt_df = _tokenize_texts(txt_df)   # -> LongTensor [Bf, L]
            if not isinstance(txt_dr, torch.Tensor):
                txt_dr = _tokenize_texts(txt_dr)   # -> LongTensor [Br, L]

            txt_df = txt_df.to(device, non_blocking=True)
            txt_dr = txt_dr.to(device, non_blocking=True)

            # ===== 2) 拼成一个 batch: [Df, Dr] =====
            img_all = torch.cat([img_df, img_dr], dim=0)   # [Bf+Br, ...]
            txt_all = torch.cat([txt_df, txt_dr], dim=0)   # [Bf+Br, L]

            Bf = img_df.size(0)
            Br = img_dr.size(0)
            B = Bf + Br

            # flags: 1 = forget(Df), 0 = retain(Dr)
            flags = torch.cat(
                [
                    torch.ones(Bf, dtype=torch.long, device=device),
                    torch.zeros(Br, dtype=torch.long, device=device),
                ],
                dim=0,
            )
            forget_mask = flags.float()             # [B]
            retain_mask = (1 - flags).float()       # [B]

            targets = torch.arange(B, device=device)  # diag target

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                # ----- 当前待遗忘模型的 logits -----
                sim_i2t_u, sim_t2i_u, _, _ = _get_logits_and_feats(
                    model, img_all, txt_all
                )  # [B,B], [B,B]

                # ----- teacher logits (用于 KL / 对照) -----
                with torch.no_grad():
                    sim_i2t_t, sim_t2i_t, _, _ = _get_logits_and_feats(
                        teacher, img_all, txt_all
                    )

                # ===== CE 部分：每个样本一条 CE loss =====
                ce_img = F.cross_entropy(
                    sim_i2t_u, targets, reduction="none")  # [B]
                ce_txt = F.cross_entropy(
                    sim_t2i_u, targets, reduction="none")  # [B]

                # Df: 只在 forget_mask == 1 上取平均，再取负号
                forget_img_loss = masked_mean(ce_img, forget_mask)
                forget_txt_loss = masked_mean(ce_txt, forget_mask)
                loss_forget = -(forget_img_loss + forget_txt_loss)

                # Dr: 只在 retain_mask == 1 上取平均，正常加
                retain_img_loss = masked_mean(ce_img, retain_mask)
                retain_txt_loss = masked_mean(ce_txt, retain_mask)
                loss_retain = (retain_img_loss + retain_txt_loss)

                # ===== KL 部分：只在 Dr 样本上做 KL(unlearn || teacher) =====
                # 先对整批算 KL，再用 retain_mask 做 masked_mean
                log_p_img = F.log_softmax(sim_i2t_u, dim=-1)    # [B, B]
                p_img_t = F.softmax(sim_i2t_t, dim=-1)        # [B, B]

                log_p_txt = F.log_softmax(sim_t2i_u, dim=-1)    # [B, B]
                p_txt_t = F.softmax(sim_t2i_t, dim=-1)        # [B, B]

                kl_img_all = F.kl_div(log_p_img, p_img_t,
                                      reduction="none").sum(dim=-1)  # [B]
                kl_txt_all = F.kl_div(log_p_txt, p_txt_t,
                                      reduction="none").sum(dim=-1)  # [B]

                kl_img = masked_mean(kl_img_all, retain_mask)
                kl_txt = masked_mean(kl_txt_all, retain_mask)
                loss_kl = kl_img + kl_txt

                # ===== 总 loss：Df 负号 + Dr 正常 + KL on Dr =====
                loss = (
                    lambda_df * loss_forget
                    + lambda_dr * loss_retain
                    + lambda_uni * loss_kl
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 日志记录
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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    print('\n job_ID {}: \n'.format(job_id))

    cfg = Config(args)
    cfg.forget_train_file = args.forget_train_file
    cfg.forget_test_file = args.forget_test_file

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # Prepare for Dr and Df
    data_name = list(cfg.datasets_cfg.keys())[0]
    if 'flickr30k' in data_name:
        data_type = 'flickr30k'
    elif 'coco' in data_name:
        data_type = 'coco'
    elif data_name == 'nlvr':
        data_type = 'nlvr'
    elif 'snli_ve' in data_name:
        data_type = 've'

    dtrain = datasets[data_name]['train']
    dtest = datasets[data_name]['test']

    forget_train_ids, forget_train_id_set, _ = _load_forget_train_ids(
        dtrain, cfg, data_type)
    forget_test_ids, forget_test_id_set, _ = _load_forget_test_ids(
        dtest, cfg, data_type)
    print(f"Loaded {len(forget_test_ids)} forget test images.")
    dr = prepare_dr_data(dtrain, cfg, data_type,
                         df_ids_set=forget_train_id_set)
    df = prepare_df_data(dtrain, cfg, data_type,
                         df_ids=forget_train_ids, df_ids_set=forget_train_id_set)
    df_for_test = prepare_df_data_for_test(
        dtrain, dtest, cfg, data_type, df_ids=forget_test_ids, df_ids_set=forget_test_id_set
    )

    retain_sample_size = len(df_for_test.annotation*5)  # 建议 *5，和后续 AUC 切片一致

    dr_for_test = prepare_dr_data_for_test(
        dtrain,
        dtest,
        cfg,
        data_type,
        retain_sample_size,
        df_ids_set=forget_test_id_set,
    )

    datasets[data_name]['df'] = df_for_test
    datasets[data_name]['dr'] = dr_for_test

    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task,
                        model=model, datasets=datasets)
    df_loader = runner.dataloaders['df']
    dr_loader = runner.dataloaders['dr']
    device = runner.model.device

    # —— 将日志同时写入文件 —— #
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, f"run_{now()}.log")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"[Log] Saving logs to: {log_path}")

    # ========= 新增：判断是否只评测原始模型 =========
    use_original_only = bool(getattr(args, "original_eval", False)) or (
        getattr(args, "unlearn_method", "") in [
            "original", "none", "clip-original"]
    )

    if not use_original_only:
        # 只有在需要做遗忘训练时才初始化 scaler / optimizer / teacher / train_loader
        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        # create policy model（遗忘模型 / 被训练的模型）
        print("unlearn policy arch:", args.arch)

        # # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, eps=1e-06, weight_decay=args.weight_decay)
        optim_state = copy.deepcopy(optimizer.state_dict())

        # 冻结“原模型”作为 teacher
        teacher = copy.deepcopy(model).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # 训练用 dataloader（直接走 LAVIS dataset 的 __getitem__）
        bs_train = cfg.run_cfg.batch_size_train
        num_workers = cfg.run_cfg.num_workers
        df_train_loader = DataLoader(
            df, batch_size=bs_train, shuffle=True, num_workers=num_workers, drop_last=True)
        dr_train_loader = DataLoader(
            dr, batch_size=bs_train, shuffle=True, num_workers=num_workers, drop_last=True)

        # 当传入 --cliperase 或 --unlearn_method cliperase 时，走 ClipErase baseline
        use_cliperase = bool(getattr(args, "cliperase", False)) or (
            getattr(args, "unlearn_method", "") == "cliperase"
        )

        if use_cliperase:
            logging.info("[Train] Using ClipErase-style supervised baseline.")
            supervised_unlearn_train_cliperase(
                cfg, model, teacher, df_train_loader, dr_train_loader,
                optimizer, scaler,
                lambda_df=getattr(args, "lambda_df", 1.0),   # Df (forget) 权重
                lambda_dr=getattr(args, "lambda_dr", 1.0),   # Dr (retain) 权重
                lambda_uni=getattr(args, "lambda_uni", 1.0),  # KL 权重
                max_epoch=getattr(args, "max_epoch", 1),
                log_interval=getattr(args, "log_interval", 50),
            )
        else:
            logging.info(
                "[Train] Using Multidelete-style supervised baseline.")
            logging.info(
                "[Train] Using MSE-based supervised baseline (shuffle/minsim).")
            # ========= 监督式 unlearning（替代原 TTA 训练），评测仍用 df/dr/test =========
            supervised_unlearn_train(
                cfg, model, teacher, df_train_loader, dr_train_loader,
                optimizer, scaler,
                # Df 多模态解耦损失权重
                lambda_md=getattr(args, "lambda_df", 1.0),
                lambda_keep=getattr(args, "lambda_dr",
                                    2.0),        # Dr 多模态保持损失权重
                lambda_uni=getattr(args, "lambda_uni",
                                   0.1),        # 单模态 MSE 权重
                max_epoch=getattr(args, "max_epoch", 1),
                log_interval=getattr(args, "log_interval", 50),
                neg_mode=args.neg_mode,
                concept_token=getattr(args, "concept_token", "dog"),
                sam3_mask_dir=getattr(args, "sam3_mask_dir", None),
                sam3_mask_suffix=getattr(args, "sam3_mask_suffix", ".png"),
            )
    else:
        logging.info(
            "[Eval-only] Skip unlearning. Directly evaluate ORIGINAL CLIP on df/dr/test.")

    if not getattr(args, "external_eval_only", False):
        # 监督训练完后，再在 df/dr/test 上按原规则评测（不做 TTA）
        task_name = "i2t" if args.retrieval_task == "image2text" else "t2i"
        # 确保 df_loader 只加载遗忘集的图像
        print(
            f"df_loader contains {len(df_loader.dataset.image)} images from the forget test set.")
        score_df = eval_split_no_tta(df_loader, model, task=task_name, text_bs=128)
        score_dr = eval_split_no_tta(dr_loader, model, task=task_name, text_bs=128)

        # 调用评估函数
        eval_df = task._report_metrics(
            score_df, score_df.T, df_loader.dataset.txt2img, df_loader.dataset.img2txt)
        eval_dr = task._report_metrics(
            score_dr, score_dr.T, dr_loader.dataset.txt2img, dr_loader.dataset.img2txt)

        # 输出
        for name, result in [("df", eval_df), ("dr", eval_dr)]:
            output_filename = os.path.join(
                args.output, f"results_{args.retrieval_task}_{name}.json")
            logging.info(output_filename)
            result = {k: round(v, 3) for k, v in result.items()}
            logging.info(result)
            with open(output_filename, "w") as fp:
                json.dump(result, fp, indent=4)

        # 训练后再根据评测得到的分数矩阵，导出 Top-K 结果
        task_suffix = "i2t" if args.retrieval_task == "image2text" else "t2i"
        df_res_path = os.path.join(
            args.output, f"detailed_top10_{task_suffix}_df.jsonl")
        _dump_detailed_topk_results(
            score_df, df_loader, task_name, df_res_path, k=10)
        dr_res_path = os.path.join(
            args.output, f"detailed_top10_{task_suffix}_dr.jsonl")
        _dump_detailed_topk_results(
            score_dr, dr_loader, task_name, dr_res_path, k=10)

        # 用训完的模型评测官方 Flickr30k 测试集（全量）
        # 用 runner 自带的 'test' dataloader
        test_loader = runner.dataloaders.get('test', None)
        if test_loader is not None:
            task_name = "i2t" if args.retrieval_task == "image2text" else "t2i"
            full_test_scores = eval_split_no_tta(
                test_loader, model, task=task_name, text_bs=128)
            # 汇报标准 Recall@K
            eval_test = task._report_metrics(
                full_test_scores,
                full_test_scores.T,
                test_loader.dataset.txt2img,
                test_loader.dataset.img2txt
            )
            test_out = os.path.join(
                args.output, f"results_{args.retrieval_task}_official_test.json")
            logging.info(test_out)
            with open(test_out, "w") as fp:
                json.dump({k: round(v, 3)
                          for k, v in eval_test.items()}, fp, indent=4)
            logging.info(
                f"[OFFICIAL TEST] { {k: round(v, 3) for k, v in eval_test.items()} }")
        else:
            logging.warning(
                "[OFFICIAL TEST] runner.dataloaders['test'] 不存在，已跳过官方测试集评测。")
    else:
        logging.info("[Eval-only] Skipping internal df/dr/official test evaluation.")

    if getattr(args, "external_test_dataset", "") == "cifar100":
        if not args.external_test_root:
            raise ValueError("external_test_root is required for CIFAR-100 evaluation.")
        if not args.external_test_forget_class:
            raise ValueError("external_test_forget_class is required for CIFAR-100 evaluation.")
        _evaluate_cifar100_forget_retain(
            model,
            data_root=args.external_test_root,
            forget_class=args.external_test_forget_class,
            image_size=args.external_test_image_size,
            batch_size=args.external_test_batch_size,
            num_workers=args.external_test_num_workers,
            output_dir=args.output,
        )
        
    if not use_original_only and args.save_unlearned_model and get_rank() == 0:
        save_dir = os.path.join(args.output, args.unlearned_subdir)
        _save_unlearned_checkpoint(
            model,
            save_dir,
            args.unlearned_model_name,
            args.unlearned_meta_name,
            args.cfg_path,
            args_dict=vars(args),
            job_id=job_id,
        )


if __name__ == "__main__":
    main()

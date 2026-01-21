# coding=utf-8
import os
import json
import time
import random
import logging
import datetime
from tqdm import tqdm
import copy

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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
from lavis.tasks import *

from params import parse_args
from clip_unlearn_reward import get_reward_model
from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_TTA
from lavis.models.clip_models.tokenizer import tokenize
from torch import nn

# -------- Reward utilities (advantage & regularizer) ----------
def _centered_adv(scores: torch.Tensor, mode: str) -> torch.Tensor:
    """
    scores: [B, K] 或 [BK] 的 clipscore（已 >=0 处理）
    mode: 'df' 或 'dr'
    返回: 与每个样本对应的 advantage，形状与 scores 对齐
    """
    # 统一到 [B, K]
    if scores.dim() == 1:
        scores = scores.view(-1, 1)
    mean = scores.mean(dim=-1, keepdim=True)
    std  = scores.std(dim=-1, keepdim=True) + 1e-6
    if mode == "df":   # 遗忘：高于基线应被惩罚
        # adv = (mean - scores) / std
        adv = - mean
    else:              # 保留：高于基线应被奖励
        # adv = (scores - mean) / std
        adv = mean
    return adv

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
    feats = model.get_text_features(text=text, tokenized_prompts=tokenized_prompts)
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


def _normalize_annotation_id(ann, cfg):
    if cfg.run_cfg.task == 'retrieval':
        return os.path.basename(ann['image'])
    if cfg.run_cfg.task == 'vqa':
        return ann['image']
    if cfg.model_cfg.model_type == 'nlvr':
        return str(tuple(ann['images']))
    if cfg.model_cfg.model_type in ['base', 've']:
        return ann['image']
    raise NotImplementedError(
        f"Unsupported task/model combination: task={cfg.run_cfg.task}, model={cfg.model_cfg.model_type}"
    )


def _load_forget_ids(dataset_train_ori, cfg, data_type):
    with open(f'/datanfs4/shenruoyan/FMUClip/Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = [i.strip() for i in f.readlines() if i.strip()]

    df_ids = df_ids[: cfg.run_cfg.df_size]

    train_ids = {_normalize_annotation_id(ann, cfg) for ann in dataset_train_ori.annotation}

    filtered_ids = [img_id for img_id in df_ids if img_id in train_ids]
    ignored = sorted(set(df_ids) - set(filtered_ids))
    if ignored:
        sample_ignored = ', '.join(ignored[:5])
        logging.warning(
            "%d forget images are not part of the official training split and will be ignored: %s%s",
            len(ignored),
            sample_ignored,
            "..." if len(ignored) > 5 else "",
        )

    if not filtered_ids:
        raise ValueError("No forget images remaining after filtering against the training split.")

    logging.info(
        "Loaded %d forget images from list (requested %d).",
        len(filtered_ids),
        len(df_ids),
    )

    return filtered_ids, set(filtered_ids), train_ids


def prepare_dr_data(dataset_train_ori, cfg, data_type, df_ids_set=None):
    if df_ids_set is None:
        _, df_ids_set, _ = _load_forget_ids(dataset_train_ori, cfg, data_type)

    dataset = copy.deepcopy(dataset_train_ori)

    if cfg.run_cfg.task == 'retrieval':
        dataset.annotation = [
            ann for ann in dataset.annotation if os.path.basename(ann['image']) not in df_ids_set
        ]

    elif cfg.run_cfg.task == 'vqa':
        dataset.annotation = [ann for ann in dataset.annotation if ann['image'] not in df_ids_set]
        dataset._add_instance_ids()

    elif cfg.model_cfg.model_type == 'nlvr':
        dataset.annotation = [
            ann for ann in dataset.annotation if str(tuple(ann['images'])) not in df_ids_set
        ]
        dataset._add_instance_ids()

    elif cfg.model_cfg.model_type == 've':
        dataset.annotation = [ann for ann in dataset.annotation if ann['image'] not in df_ids_set]
        dataset._add_instance_ids()

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

    return dataset

def prepare_df_data(dataset_train_ori, cfg, data_type, df_ids=None, df_ids_set=None):
    if df_ids is None or df_ids_set is None:
        df_ids, df_ids_set, _ = _load_forget_ids(dataset_train_ori, cfg, data_type)

    dataset = copy.deepcopy(dataset_train_ori)
    
    if cfg.run_cfg.task == 'retrieval':
        dataset.annotation = [
            ann for ann in dataset.annotation if os.path.basename(ann['image']) in df_ids_set
        ]

    elif cfg.run_cfg.task == 'vqa':
        dataset.annotation = [ann for ann in dataset.annotation if ann['image'] in df_ids_set]
        dataset._add_instance_ids()

    elif cfg.model_cfg.model_type == 'nlvr':
        dataset.annotation = [
            ann for ann in dataset.annotation if str(tuple(ann['images'])) in df_ids_set
        ]
        dataset._add_instance_ids()

    elif cfg.model_cfg.model_type == 've':
        dataset.annotation = [ann for ann in dataset.annotation if ann['image'] in df_ids_set]
        dataset._add_instance_ids()

    # assert num_image_after_removal == cfg.run_cfg.df_size, f"{num_image_after_removal}, {cfg.run_cfg.df_size}"

    return dataset


def prepare_df_data_for_test(
    dataset_train_ori, dataset_test_ori, cfg, data_type, df_ids=None, df_ids_set=None
):
    if df_ids is None or df_ids_set is None:
        df_ids, df_ids_set, _ = _load_forget_ids(dataset_train_ori, cfg, data_type)


    if cfg.run_cfg.task == 'retrieval':
        df_for_test = copy.deepcopy(dataset_test_ori)
        annotation = [
            ann for ann in dataset_train_ori.annotation if os.path.basename(ann['image']) in df_ids_set
        ]

        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')
        df_for_test.annotation = test_anno

        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = df_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for _, caption in enumerate(ann["caption"]):
                text.append(text_processor(caption))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

        df_for_test.text = text
        df_for_test.image = image
        df_for_test.txt2img = txt2img
        df_for_test.img2txt = img2txt

    elif cfg.run_cfg.task == 'vqa':
        # breakpoint()
        # Retrieval train and test data are same. To use VQA test data for Df, copy the ori train data
        df_for_test = copy.deepcopy(dataset_train_ori)

        df_for_test.annotation = [ann for ann in df_for_test.annotation if ann['image'] in df_ids_set]
        df_for_test._add_instance_ids()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [
            ann for ann in df_for_test.annotation if str(tuple(ann['images'])) in df_ids_set
        ]
        df_for_test._add_instance_ids()

    elif cfg.model_cfg.model_type in ['base', 've']:
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [ann for ann in df_for_test.annotation if ann['image'] in df_ids_set]
        df_for_test._add_instance_ids()

    return df_for_test

def prepare_dr_data_for_test(
    dataset_train_ori,
    dataset_test_ori,
    cfg,
    data_type,
    sample_size=None,
    df_ids_set=None,
):
    if df_ids_set is None:
        _, df_ids_set, _ = _load_forget_ids(dataset_train_ori, cfg, data_type)


    if cfg.run_cfg.task == 'retrieval':
        dr_for_test = copy.deepcopy(dataset_test_ori)
        annotation = [
            ann for ann in dataset_train_ori.annotation if os.path.basename(ann['image']) not in df_ids_set
        ]

        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')

        # >>> 新增：检索分支支持 sample_size 下采样，避免 OOM <<<
        if sample_size is not None and sample_size < len(test_anno):
            anno_id = np.arange(len(test_anno))
            indices = np.random.choice(anno_id, sample_size, replace=False)
            test_anno = [test_anno[i] for i in indices]
        # <<< 新增结束 >>>

        dr_for_test.annotation = test_anno


        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = dr_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for _, caption in enumerate(ann["caption"]):
                text.append(text_processor(caption))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

        dr_for_test.text = text
        dr_for_test.image = image
        dr_for_test.txt2img = txt2img
        dr_for_test.img2txt = img2txt

    elif cfg.run_cfg.task == 'vqa':
        # breakpoint()
        # Retrieval train and test data are same. To use VQA test data for Df, copy the ori train data
        dr_for_test = copy.deepcopy(dataset_train_ori)

        dr_for_test.annotation = [ann for ann in dr_for_test.annotation if ann['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        dr_for_test.annotation = [
            ann for ann in dr_for_test.annotation if str(tuple(ann['images'])) not in df_ids_set
        ]

        if sample_size is not None:
            anno_id = np.arange(len(dr_for_test.annotation))
            indices = np.random.choice(anno_id, sample_size, replace=False)
            dr_for_test.annotation = [dr_for_test.annotation[i] for i in indices]

        dr_for_test._add_instance_ids()

    elif cfg.model_cfg.model_type in ['base', 've']:
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        dr_for_test.annotation = [ann for ann in dr_for_test.annotation if ann['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()

    return dr_for_test

@torch.no_grad()
def eval_split_no_tta(loader, model, task="i2t", text_bs=128):
    """
    用“当前权重”一次性评测一个 split（不做任何更新）：
    - 评分矩阵放 CPU float16 + pinned（省显存，几乎不降速）
    返回：score 矩阵（numpy）
    """
    model.eval()
    device = model.device
    exported_state = None
    logit_scale = model.clip_model.logit_scale.exp()

    if task == "i2t":
        # 1) 缓存全部文本特征 → CPU half
        with torch.amp.autocast('cuda'):
            text_ids = tokenize_all_text(loader.dataset.text, model, text_bs)
            text_embeds = get_all_text_embeds(text_ids, model, text_bs)  # [N_t, D]
        text_embeds = text_embeds.half().cpu().pin_memory()

        # 2) 评分矩阵（CPU half + pinned）
        num_img, num_txt = len(loader.dataset.image), len(loader.dataset.text)
        scores = torch.full((num_img, num_txt), -100.0, dtype=torch.float16, device='cpu').pin_memory()
        # 3) 逐图像前向并写入该行
        for i, samples in enumerate(tqdm(loader, total=len(loader), ncols=150, desc="EVAL I2T")):
            image = samples["image"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                img_feat = model.get_image_features(image)                         # [1,D]
                logits = logit_scale * (img_feat @ text_embeds.to(device).T)      # [1,N_t]
            scores[i] = logits.squeeze(0).to('cpu', dtype=torch.float16, non_blocking=True)
        return scores.numpy()

    else:  # task == "t2i"
        # 1) 缓存全部图像特征 → CPU half
        with torch.amp.autocast('cuda'):
            image_embeds = get_all_image_embeds(loader, model)   # [N_i, D]
        image_embeds = image_embeds.half().cpu().pin_memory()

        # 2) 评分矩阵（CPU half + pinned）
        num_txt, num_img = len(loader.dataset.text), len(loader.dataset.image)
        scores = torch.full((num_txt, num_img), -100.0, dtype=torch.float16, device='cpu').pin_memory()

        # 3) 逐文本前向并写入该行
        for i, samples in enumerate(tqdm(loader, total=len(loader), ncols=150, desc="EVAL T2I")):
            raw_text = samples.get("text", None)
            tokenized_prompts = samples.get("tokenized_prompts", None)
            if tokenized_prompts is not None:
                tokenized_prompts = tokenized_prompts.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                txt_feat = model.get_text_features(text=raw_text, tokenized_prompts=tokenized_prompts)  # [1,D]
                logits = logit_scale * (txt_feat @ image_embeds.to(device).T)                            # [1,N_i]
            scores[i] = logits.squeeze(0).to('cpu', dtype=torch.float16, non_blocking=True)
        return scores.numpy()

def _dump_topk_results(scores_np, loader, task, out_file, k=10):
    """
    将评测阶段的 score 矩阵导出为 JSONL：
    - I2T: 每张图像的 Top-K 文本（含 id、score、内容）
    - T2I: 每条文本的 Top-K 图像（含 id、score、路径）
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    import json
    K = max(1, k)
    scores = torch.from_numpy(scores_np)  # [Nq, Nc]
    # id→内容映射
    if task == "i2t":
        id2cand = lambda j: loader.dataset.text[j]
        query_list = getattr(loader.dataset, "image", [f"#{i}" for i in range(scores.size(0))])
    else:
        id2cand = lambda j: loader.dataset.image[j]
        query_list = getattr(loader.dataset, "text", [f"#{i}" for i in range(scores.size(0))])

    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(scores.size(0)):
            row = scores[i]                               # [Nc]
            v, idx = torch.topk(row, min(K, row.numel())) # (K), (K)
            items = []
            for r, (jj, sc) in enumerate(zip(idx.tolist(), v.tolist()), start=1):
                items.append({"rank": r, "id": jj, "score": float(sc), "content": id2cand(jj)})
            obj = {"query_id": i, "query": query_list[i], "topk": items}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    logging.info(f"[TopK] Saved Top-{K} results to {out_file}")

def tokenize_all_text(texts, model, text_bs=128):
    """tokenize all text and return: (text_ids)"""
    num_text = len(texts)
    text_ids = []
    i = 0
    while i < num_text:
        text = texts[i : min(num_text, i + text_bs)]
        input_ids = tokenize(text).to(model.device)
        text_ids.append(input_ids)
        i += text_bs
    text_ids = torch.cat(text_ids, dim=0)

    return text_ids


def get_all_text_embeds(text_inputs, model, text_bs=128):
    logging.info("Extracting ALL Text features...")
    text_embeds = []
    i = 0
    while i < text_inputs.shape[0]:
        batch = text_inputs[i : min(text_inputs.shape[0], i + text_bs)]
        text_features = model.get_text_features(text=None, tokenized_prompts=batch)
        text_embeds.append(text_features)
        i += text_bs

    return torch.cat(text_embeds, dim=0)


def get_all_image_embeds(data_loader, model):
    """extract all image embeddings"""
    logging.info("Extracting ALL image features...")
    image_embeds = []
    for samples in data_loader:
        image = samples["image"].to(model.device)
        image_features = model.get_image_features(image)
        image_embeds.append(image_features)

    return torch.cat(image_embeds, dim=0)


def tune_image(image, model, reward_model, optimizer, scaler,
               mode="df", lambda_df=1.0, lambda_dr=1.0,
               drift_coef=0.0, init_feats=None, args=None,
               *,# ← 从这里开始强制关键字
               sample_info=None,         # dict: {"image_path": "...", "gt_txt_ids": [...]}
               id2text=None,             # callable: int -> str (文本字符串)
               topk_print=10):
    """
    针对 image encoder 的 TTA：
    - mode: 'df'（遗忘集）或 'dr'（保留集）
    - lambda_df / lambda_dr: 奖励强度权重（论文里 λ1 / λ2）
    - drift_coef: 保留集特征保持正则系数（近似 KL）
    - init_feats: 与 image 对应的“原模型图像特征”，仅用于 Dr 漂移约束
    """
    sample_k = reward_model.sample_k
    bs = image.shape[0]
    model.train()

    # policy gradient for single sample
    reward_model.set_image_features(images=image)
    loss_df_sum = loss_dr_sum = drift_sum = total_sum = 0.0
    for step in range(args.tta_steps):    
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text = model(image)
            # sample results
            # 遗忘集(df) 采样“相似度最低”的 K 个候选；保留集(dr) 采样“最高”的 K
            value, index = torch.topk(
                logits_per_image, sample_k, dim=-1,
                largest = (mode != "df")
            )
            text_index = index.flatten()

            # ——【新增】仅用于可读性展示：top-10 —— 
            with torch.no_grad():
                k = min(topk_print, logits_per_image.size(-1))
                v10, i10 = torch.topk(logits_per_image[0], k, dim=-1)
                if sample_info is not None:
                    img_path = sample_info.get("image_path")
                    gt_txt_ids = sample_info.get("gt_txt_ids")
                    logging.info(
                        f"[{mode.upper()}][step {step+1}/{args.tta_steps}] "
                        f"image={img_path}  GT_txt_ids={gt_txt_ids}"
                    )
                if id2text is not None:
                    block = _fmt_topk_rows(i10, v10, id2text)
                    logging.info("Top-10 (image→text):\n" + block)

            # raw CLIPScore
            clip_score = reward_model.CLIPScore(text_index=text_index, pairwise=False)
            clip_score = clip_score.clamp_min(0.0)
            # [B,K]
            scores_2d = clip_score.view(bs, sample_k)
            adv = _centered_adv(scores_2d, mode=mode)     # GRPO/优势基线
            # 奖励后处理（与原reward流程兼容）
            adv = reward_model.rewards_post_process(adv) if reward_model.process_batch else adv

            # 交叉熵作为 -log pθ(y|x) 的代理（REINFORCE: E[R * ∇log p]）
            rep_output = torch.repeat_interleave(logits_per_image, sample_k, dim=0)
            ce = F.cross_entropy(rep_output, text_index, reduction='none').view(bs, sample_k)

            # Df / Dr 的加权
            loss_df = (adv * ce).mean() if mode == "df" else torch.tensor(0.0, device=ce.device)
            loss_dr = (adv * ce).mean() if mode == "dr" else torch.tensor(0.0, device=ce.device)
            loss = lambda_df * loss_df + lambda_dr * loss_dr

            # 保留集：加入“原模型特征保持”正则（KL近似为 MSE）
            drift_loss = torch.tensor(0.0, device=image.device)
            if (mode == "dr") and (drift_coef > 0.0) and (init_feats is not None):
                # 文本库来自预缓存的 model.text_features（[N_txt, D]，已归一化）
                # old logits: 用 initial_state 特征（init_feats）与文本库打分
                # cur logits: 用当前特征与同一文本库打分
                text_bank = model.text_features.to(image.device)                       # [N_txt, D]
                cur_feats = model.get_image_features(image)                            # [B,D]
                logit_scale = model.clip_model.logit_scale.exp()
                logits_old = logit_scale * (init_feats @ text_bank.t())                # [B, N_txt]
                logits_cur = logit_scale * (cur_feats @ text_bank.t())                 # [B, N_txt]
                log_p_old = F.log_softmax(logits_old, dim=-1)                          # target
                log_p_cur = F.log_softmax(logits_cur, dim=-1)                          # input
                # KL(old || cur): use F.kl_div with log_target=True
                drift_loss = F.kl_div(log_p_cur, log_p_old, reduction='batchmean', log_target=True)
                loss = loss + drift_coef * drift_loss

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # —— 累加到样本级统计 —— 
        loss_df_sum += float(loss_df.detach().item())
        loss_dr_sum += float(loss_dr.detach().item())
        drift_sum   += float(drift_loss.detach().item()) if isinstance(drift_loss, torch.Tensor) else 0.0
        total_sum   += float(loss.detach().item())

    # —— 仅打印一次（8步平均）——
    steps = max(1, args.tta_steps)
    logging.info(
        f"[{mode.upper()}] steps={args.tta_steps} "
        f"loss_df={loss_df_sum/steps:.6f} loss_dr={loss_dr_sum/steps:.6f} "
        f"drift={drift_sum/steps:.6f} total={total_sum/steps:.6f}"
    )


def tune_text(text=None, tokenized_prompts=None, model=None, reward_model=None,
              optimizer=None, scaler=None,
              mode="df", lambda_df=1.0, lambda_dr=1.0,
              drift_coef=0.0, init_feats=None, args=None,
              *,             # 强制关键字
              sample_info=None,         # dict: {"text": "..."} 或 {"text_ids": [...]}
              id2image=None,            # callable: int -> str (图片路径)
              topk_print=10):
    """
    针对 text encoder 的 TTA（only_visual == False 时使用）：
    - mode: 'df'（遗忘集）或 'dr'（保留集）
    - lambda_df / lambda_dr: 奖励强度权重（对应 λ1 / λ2）
    - drift_coef: 文本特征保持正则系数（Dr 可选，MSE ~ KL）
    - init_feats: 与 text 对应的“原模型文本特征”，仅用于 Dr 漂移约束
    """
    assert model is not None and reward_model is not None
    sample_k = reward_model.sample_k
    bs = 1  # 当前实现按单条文本进行 TTA（与现有循环一致）
    model.train()

    # 让 reward_model 知道当前这条文本（用于计算 pairwise CLIPScore）
    # 如果 reward_model 里需要缓存文本/图像特征，请保持与 image 侧一致的接口约定
    #reward_model.set_text_features(captions=text, tokenized_cap=tokenized_prompts)
    reward_model.set_text_features(captions=text, tokenized_cap=tokenized_prompts)
    loss_df_sum = loss_dr_sum = drift_sum = total_sum = 0.0
    for step in range(args.tta_steps):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # 前向：对称于图像侧，文本侧走可微路径；图像特征从缓存中取（_prepare_image_cache）
            _, logits_per_text = model(images=None, text=text, tokenized_prompts=tokenized_prompts)

            # 采样 top-k 目标图像（和 tune_image 里对文本的做法完全对称）
            # 遗忘集(df) 采样“相似度最低”的 K 个候选；保留集(dr) 采样“最高”的 K
            value, index = torch.topk(
                logits_per_text, sample_k, dim=-1,
                largest = (mode != "df")
            )
            images_index = index.flatten()  # [B*K] = [K]

            # ——【新增】仅用于可读性展示：top-10 —— 
            # with torch.no_grad():
            #     k = min(topk_print, logits_per_text.size(-1))
            #     v10, i10 = torch.topk(logits_per_text[0], k, dim=-1)
            #     if sample_info is not None:
            #         raw_t = sample_info.get("text")
            #         logging.info(
            #             f"[{mode.upper()}][step {step+1}/{args.tta_steps}] text={raw_t}"
            #         )
            #     if id2image is not None:
            #         block = _fmt_topk_rows(i10, v10, id2image)
            #         logging.info("Top-10 (text→image):\n" + block)

            # 原始 CLIPScore（与 image 侧保持一致，使用 reward_model 的统一接口）
            clip_score = reward_model.CLIPScore(text_index=None, images_index=images_index, pairwise=False)
            clip_score = clip_score.clamp_min(0.0)

            # 组成 [B, K]，做居中优势
            scores_2d = clip_score.view(bs, sample_k)
            adv = _centered_adv(scores_2d, mode=mode)    # df: mean-score; dr: score-mean
            adv = reward_model.rewards_post_process(adv) if reward_model.process_batch else adv

            # 交叉熵作为 -log pθ(y|x) 代理（与 image 侧一致）
            rep_output = torch.repeat_interleave(logits_per_text, sample_k, dim=0)  # [B*K, I]
            ce = F.cross_entropy(rep_output, images_index, reduction='none').view(bs, sample_k)

            # Df / Dr 的加权求和
            loss_df = (adv * ce).mean() if mode == "df" else torch.tensor(0.0, device=ce.device)
            loss_dr = (adv * ce).mean() if mode == "dr" else torch.tensor(0.0, device=ce.device)
            loss = lambda_df * loss_df + lambda_dr * loss_dr

            # 保留集文本漂移约束（可选），与图像侧对称
            drift_loss = torch.tensor(0.0, device=rep_output.device)
            if (mode == "dr") and (drift_coef > 0.0) and (init_feats is not None):
                image_bank = model.image_features.to(rep_output.device)               # [N_img, D]
                cur_feats = model.get_text_features(text=text, tokenized_prompts=tokenized_prompts)  # [B,D]
                logit_scale = model.clip_model.logit_scale.exp()
                logits_old = logit_scale * (init_feats @ image_bank.t())              # [B, N_img]
                logits_cur = logit_scale * (cur_feats @ image_bank.t())               # [B, N_img]
                log_p_old = F.log_softmax(logits_old, dim=-1)                         # target
                log_p_cur = F.log_softmax(logits_cur, dim=-1)                         # input
                drift_loss = F.kl_div(log_p_cur, log_p_old, reduction='batchmean', log_target=True)
                loss = loss + drift_coef * drift_loss

        # 反传与一步优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # —— 累加到样本级统计 —— 
        loss_df_sum += float(loss_df.detach().item())
        loss_dr_sum += float(loss_dr.detach().item())
        drift_sum   += float(drift_loss.detach().item()) if isinstance(drift_loss, torch.Tensor) else 0.0
        total_sum   += float(loss.detach().item())

    # —— 仅打印一次（8步平均）——
    steps = max(1, args.tta_steps)
    logging.info(
        f"[{mode.upper()}] steps={args.tta_steps} "
        f"loss_df={loss_df_sum/steps:.6f} loss_dr={loss_dr_sum/steps:.6f} "
        f"drift={drift_sum/steps:.6f} total={total_sum/steps:.6f}"
    )

def export_unlearned_checkpoint(model_state, args, cfg):
    """Persist the adapted CLIP weights for downstream loading (e.g., LLaVA)."""

    export_dir = args.export_dir or args.output
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, args.export_name)

    run_cfg = None
    if hasattr(cfg, "run_cfg"):
        try:
            run_cfg = cfg.run_cfg.to_dict()
        except Exception:
            run_cfg = getattr(cfg.run_cfg, "__dict__", None)

    payload = {
        "model_state_dict": model_state,
        "arch": args.arch,
        "retrieval_task": args.retrieval_task,
        "cfg_path": getattr(args, "cfg_path", None),
        "run_cfg": run_cfg,
    }

    torch.save(payload, export_path)
    logging.info(f"[Export] Saved unlearned weights to: {export_path}")
    return export_path


def test_time_tune_split(
    df_loader,
    dr_loader,
    device,
    model,
    reward_model=None,
    scaler=None,
    optimizer=None,
    optim_state=None,
    text_bs=128,
    args=None,
    lambda_df=1.0,
    lambda_dr=2.0,
    drift_coef=0.01, 
    save_state: bool = False,
    extra_eval_loaders=None,
):
    """
    在 Df / Dr 两个 split 上分别进行 TTA + 评测。
    - lambda_df / lambda_dr：对应你图里的 λ1 / λ2
    - drift_coef：保留集的特征保持正则系数(可选)
    """
    model.eval()
    device = model.device
    extra_eval_loaders = extra_eval_loaders or {}
    # extra_scores = {}


    def _prepare_text_cache(loader):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                text_ids = tokenize_all_text(loader.dataset.text, model, text_bs)
                text_embeds = get_all_text_embeds(text_ids, model, text_bs)
                model.set_text_features(text_features=text_embeds)
                reward_model.set_many_text_features(loader.dataset.text, text_bs=text_bs)
        return text_embeds
    
    def _prepare_image_cache(loader):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_embeds = get_all_image_embeds(loader, model)
                model.set_image_features(image_features=image_embeds)
                reward_model.set_image_features_with_dataloder(loader)
        return image_embeds

    def _run_one_split(loader, mode, lam_df, lam_dr, drift, write_scores: bool = False):
        # score_i2t = torch.full((len(loader.dataset.image), len(loader.dataset.text)), -100.0, device=device)
        # score_t2i = torch.full((len(loader.dataset.text), len(loader.dataset.image)), -100.0, device=device)
        num_images = len(getattr(loader.dataset, "image", []))
        num_texts = len(getattr(loader.dataset, "text", []))
        pin_memory = model.device.type == "cuda"

        
        # 仅在需要在线评测时才分配评分矩阵（默认训练阶段不写）
        if write_scores:
            if model.only_visual:  # i2t
                score_i2t = torch.full((len(loader.dataset.image), len(loader.dataset.text)),
                                   -100.0, dtype=torch.float16, device='cpu', pin_memory=True)
            else:  # t2i
                score_t2i = torch.full((len(loader.dataset.text), len(loader.dataset.image)),
                                   -100.0, dtype=torch.float16, device='cpu', pin_memory=True)
        if model.only_visual: # retrieval_task==image2text
            _prepare_text_cache(loader)    
            pbar = tqdm(enumerate(loader), total=len(loader), ncols=150, desc=f"{mode.upper()} TTA")
            for i, samples in pbar:
                image = samples["image"].to(model.device)
                # 取“初始特征”用于 Dr 漂移约束
                init_feats = _frozen_init_image_feats(model, image) if mode == "dr" else None
                # mapper: 文本 id -> 文本字符串
                id2text = lambda j: loader.dataset.text[j]

                # 样本“原信息”
                img_path = loader.dataset.image[i] if hasattr(loader.dataset, "image") else f"#{i}"
                gt_txt_ids = loader.dataset.img2txt.get(i) if hasattr(loader.dataset, "img2txt") else None
                sample_info = {"image_path": img_path, "gt_txt_ids": gt_txt_ids}

                tune_image(image, model, reward_model, optimizer, scaler,
                        mode=mode, lambda_df=lam_df, lambda_dr=lam_dr,
                        drift_coef=drift, init_feats=init_feats, args=args, 
                        sample_info=sample_info, id2text=id2text, topk_print=10)   # ← 新增)
                # 评测
                # model.eval()
                # with torch.no_grad():
                #     with torch.cuda.amp.autocast():
                #         logits_per_image, _ = model(image)
                #     # score_i2t[i] = logits_per_image[0]
                #     score_i2t[i].copy_(
                #         logits_per_image[0].detach().to(dtype=torch.float32),
                #         non_blocking=pin_memory,
                #     )
                if write_scores:
                    model.eval()
                    with torch.no_grad():
                        with torch.amp.autocast('cuda'):
                            if model.only_visual:
                                logits_per_image, _ = model(image)
                                score_i2t[i] = logits_per_image[0].to('cpu', dtype=torch.float16, non_blocking=True)
                            else:
                                raw_text = samples.get("text", None)
                                tokenized_prompts = samples.get("tokenized_prompts", None)
                                if tokenized_prompts is not None:
                                    tokenized_prompts = tokenized_prompts.to(model.device, non_blocking=True)
                                _, logits_per_text = model(images=None, text=raw_text, tokenized_prompts=tokenized_prompts)
                                score_t2i[i] = logits_per_text[0].to('cpu', dtype=torch.float16, non_blocking=True)
                # 让进度条显示最近一次总 loss（从日志里拿不到时也有参考）
                pbar.set_postfix_str(f"img={i+1}/{len(loader)}")
                model.momentum_update_model()
                # 权重复位
                # model.reset_initial()
                optimizer.load_state_dict(optim_state)
            # return score_i2t.detach().cpu().numpy()
            if write_scores:
                return (score_i2t if model.only_visual else score_t2i).numpy()
            else:
                return None
        else: # retrieval_task==text2image
            score_t2i = torch.full(
                (num_texts, num_images),
                -100.0,
                dtype=torch.float32,
                device="cpu",
                pin_memory=pin_memory,
            )
            _prepare_image_cache(loader)
            # pbar = tqdm(enumerate(loader), total=len(loader), ncols=150, desc=f"{mode.upper()} TTA")
            # # for i, text in pbar:

            # for i, samples in pbar:
            # # 取原始文本或已tokenize的prompts（按你的DataLoader字段来）
            # # 假设 samples 中有 "text" 或 "tokenized_prompts"
            #     raw_text = samples.get("text", None)
            #     tokenized_prompts = samples.get("tokenized_prompts", None)
            #     if tokenized_prompts is not None:
            #         tokenized_prompts = tokenized_prompts.to(model.device)
            #     # Dr：用 initial_state 的文本特征做漂移约束
            #     init_feats = _frozen_init_text_feats(model, text=raw_text, tokenized_prompts=tokenized_prompts) if mode == "dr" else None
            #     # init_feats = _frozen_init_text_feats(model, text) if mode == "dr" else None
            #     tune_text(text=raw_text, tokenized_prompts=tokenized_prompts,
            #       model=model, reward_model=reward_model,
            #       optimizer=optimizer, scaler=scaler,
            #       mode=mode, lambda_df=lam_df, lambda_dr=lam_dr,
            #       drift_coef=drift, init_feats=init_feats, args=args)

            total_text = len(getattr(loader.dataset, "text", []))
            if total_text == 0:
                raise ValueError("Dataset must provide text entries for text-to-image retrieval.")

            pbar = tqdm(range(total_text), total=total_text, ncols=150, desc=f"{mode.upper()} TTA")

            for i in pbar:
                # dataset.text 保存了逐条文本（df/dr 与原测试集相同），按 index 取出处理
                raw_text, tokenized_prompts = _resolve_text_inputs(loader.dataset, i, None, model.device)
                print(f"Processing text {i+1}/{total_text}: {raw_text}, {tokenized_prompts}")
                if mode == "dr":
                    init_feats = _frozen_init_text_feats(
                        model, text=raw_text, tokenized_prompts=tokenized_prompts
                    )
                else:
                    init_feats = None

                id2image = lambda j: loader.dataset.image[j]
                sample_info = {"text": raw_text}

                tune_text(
                    text=raw_text, tokenized_prompts=tokenized_prompts,
                    model=model, reward_model=reward_model,
                    optimizer=optimizer, scaler=scaler,
                    mode=mode, lambda_df=lam_df, lambda_dr=lam_dr,
                    drift_coef=drift, init_feats=init_feats, args=args,
                    sample_info=sample_info, id2image=id2image, topk_print=10,  # ← 新增
                )

                # 评测
                model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        _, logits_per_text = model(images=None, text=raw_text, tokenized_prompts=tokenized_prompts)
                    # score_t2i[i] = logits_per_text[0]
                    score_t2i[i].copy_(
                        logits_per_text[0].detach().to(dtype=torch.float32),
                        non_blocking=pin_memory,
                    )
                # 让进度条显示最近一次总 loss（从日志里拿不到时也有参考）
                # pbar.set_postfix_str(f"txt={i+1}/{len(loader)}")
                pbar.set_postfix_str(f"txt={i+1}/{total_text}")

                model.momentum_update_model()
                # model.reset_initial()
                optimizer.load_state_dict(optim_state)
            # return score_t2i.detach().cpu().numpy()
            return score_t2i.numpy()

    # 原逻辑：每个样本重置权重
    # score_dr = _run_one_split(dr_loader, mode="dr", lam_df=0.0, lam_dr=lambda_dr, drift=drift_coef)
    # score_df = _run_one_split(df_loader, mode="df", lam_df=lambda_df, lam_dr=0.0, drift=0.0)

    # 现逻辑：保存原始权重与优化器状态，确保保留和遗忘集都跑完之后再统一重置模型权重
    saved_clip_state = copy.deepcopy(model.clip_model.state_dict())
    saved_initial_state = copy.deepcopy(model.initial_state_dict)
    saved_momentum_state = copy.deepcopy(getattr(model, "momentum_state_dict", None))
    saved_update_counter = model.update_counter

    try:
        # —— 顺序适配阶段：不写评分矩阵（跨样本继承权重）——
        _run_one_split(df_loader, mode="df", lam_df=lambda_df, lam_dr=0.0, drift=0.0, write_scores=False)
        _run_one_split(dr_loader, mode="dr", lam_df=0.0, lam_dr=lambda_dr, drift=drift_coef, write_scores=False)

        # —— 统一权重评测（一次性）——
        task = "i2t" if model.only_visual else "t2i"
        score_df = eval_split_no_tta(df_loader, model, task=task, text_bs=text_bs)
        score_dr = eval_split_no_tta(dr_loader, model, task=task, text_bs=text_bs)
    
        # for name, loader in extra_eval_loaders.items():
        #     extra_scores[name] = eval_split_no_tta(loader, model, task=task, text_bs=text_bs)

        # —— 导出每个样本的 Top-10 —— #
        topk_df_path = os.path.join(args.output, f"top10_{task}_df.jsonl")
        topk_dr_path = os.path.join(args.output, f"top10_{task}_dr.jsonl")
        _dump_topk_results(score_df, df_loader, task, topk_df_path, k=10)
        _dump_topk_results(score_dr, dr_loader, task, topk_dr_path, k=10)

        if save_state:
            exported_state = copy.deepcopy(model.clip_model.state_dict())
    finally:
        # 所有样本迭代完成后再统一恢复为原始权重
        model.clip_model.load_state_dict(saved_clip_state)
        model.initial_state_dict = copy.deepcopy(saved_initial_state)
        if saved_momentum_state is not None:
            model.momentum_state_dict = copy.deepcopy(saved_momentum_state)
        model.update_counter = saved_update_counter
        if optimizer is not None and optim_state is not None:
            optimizer.load_state_dict(optim_state)

    return score_df, score_dr, exported_state

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

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)


    ## Prepare for Dr and Df
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
    
    forget_ids, forget_id_set, _ = _load_forget_ids(dtrain, cfg, data_type)
    dr = prepare_dr_data(dtrain, cfg, data_type, df_ids_set=forget_id_set)
    df = prepare_df_data(dtrain, cfg, data_type, df_ids=forget_ids, df_ids_set=forget_id_set)
    df_for_test = prepare_df_data_for_test(
        dtrain, dtest, cfg, data_type, df_ids=forget_ids, df_ids_set=forget_id_set
    )

    # retain_sample_size = None
    # if hasattr(df_for_test, "text") and df_for_test.text is not None:
    #     retain_sample_size = len(df_for_test.text)
    # elif hasattr(df_for_test, "annotation") and df_for_test.annotation is not None:
    #     retain_sample_size = len(df_for_test.annotation)
    retain_sample_size = len(df_for_test.annotation) * 5  # 建议 *5，和后续 AUC 切片一致

    dr_for_test = prepare_dr_data_for_test(
        dtrain,
        dtest,
        cfg,
        data_type,
        retain_sample_size,
        df_ids_set=forget_id_set,
    )    
    
    datasets[data_name]['df'] = df_for_test
    datasets[data_name]['dr'] = dr_for_test


    # if args.unlearn_method in ['retrain', 'ft']:
    #     datasets[data_name]['train'] = dr
    #     runner.train()

    # elif args.unlearn_method in ['neggrad']:
    #     if args.task == 'retrieval':
    #         task = NegativeGradientRetrievalTask.setup_task(cfg=cfg)
    #     elif args.task == 'vqa':
    #         task = NegativeGradientVQATask.setup_task(cfg=cfg)

    #     datasets[data_name]['train'] = df
    #     runner.train()

    # elif args.unlearn_method == 'dtd':
    #     # if args.unlearn_method == 'dtd':
    #     runner_class = DescentToDelete
    #     # elif args.unlearn_method == 'fisher':
    #     #     runner_class = Fisher
    #     # elif args.unlearn_method == 'ours':
    #     #     runner_class = MultimodalUnlearn

    #     runner = runner_class(
    #         cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets,
    #     )
    #     runner.unlearn()

    # if 'vlul' in args.unlearn_method:
    #     cfg.run_cfg.batch_size_train = cfg.run_cfg.batch_size_train // 2
    #     datasets[data_name]['train'] = df
    #     datasets[data_name]['dr_train'] = dr

    #     if args.task == 'retrieval':
    #         task = VLUnlearnClassificationTask.setup_task(cfg=cfg)
    #     elif args.task == 'vqa':
    #         task = VLUnlearnVQATask.setup_task(cfg=cfg)
    #     elif args.task in ['nlvr', 've']:
    #         task = VLUnlearnClassificationTask.setup_task(cfg=cfg)

    #     runner_class = MultimodalUnlearn
    #     model_ori = task.build_model(cfg)
    #     model_ori.eval()
    #     runner = runner_class(
    #         cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets, 
    #     )
    #     runner.unlearn(args, cfg, model_ori)

    # else:
    #     raise NotImplementedError


    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    df_loader = runner.dataloaders['df']
    dr_loader = runner.dataloaders['dr']
    device = runner.model.device

    # —— 将日志同时写入文件 —— #
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, f"run_{now()}.log")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"[Log] Saving logs to: {log_path}")

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # create model
    print("unlearn policy arch:", args.arch)
    model = CLIPRet_TTA(device, arch=args.arch, only_visual=(args.retrieval_task == "image2text"),
                            momentum_update=args.momentum_update, update_freq=args.update_freq,
                            update_w=args.update_w, momentum=args.tta_momentum)
    model = model.to(device)

    # define the CLIPRewards
    reward_model = get_reward_model(device, args)

    # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-06, weight_decay=args.weight_decay)
    optim_state = copy.deepcopy(optimizer.state_dict())

    # Df / Dr 分别做策略更新与评测
    score_df, score_dr, exported_state = test_time_tune_split(
        df_loader, dr_loader, device, model, reward_model=reward_model,
        scaler=scaler, optimizer=optimizer, optim_state=optim_state,
        text_bs=128, args=args, lambda_df=args.lambda_df, lambda_dr=args.lambda_dr, drift_coef=args.drift_coef,save_state=args.save_unlearned,
    )

    # 分别汇报 Recall@K
    eval_df = task._report_metrics(score_df, score_df.T,
                                   df_loader.dataset.txt2img, df_loader.dataset.img2txt)
    eval_dr = task._report_metrics(score_dr, score_dr.T,
                                   dr_loader.dataset.txt2img, dr_loader.dataset.img2txt)

    # 输出
    for name, result in [("df", eval_df), ("dr", eval_dr)]:
        output_filename = os.path.join(args.output, f"results_{args.retrieval_task}_{name}.json")
        logging.info(output_filename)
        result = {k: round(v, 3) for k, v in result.items()}
        logging.info(result)
        with open(output_filename, "w") as fp:
            json.dump(result, fp, indent=4)
        
    if args.save_unlearned and exported_state is not None:
        export_unlearned_checkpoint(exported_state, args, cfg)


if __name__ == "__main__":
    main()

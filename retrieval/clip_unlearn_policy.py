# coding=utf-8
import math
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
        adv = (mean - scores) / std
        # adv = - mean
    else:              # 保留：高于基线应被奖励
        adv = (scores - mean) / std
        # adv = mean
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


def _load_forget_train_ids(dataset_train_ori, cfg, data_type):
    with open(f'/datanfs4/shenruoyan/FMUClip/Df/coco/forget_orange_train.txt', 'r') as f:
        df_ids = [i.strip() for i in f.readlines() if i.strip()]

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

def _load_forget_test_ids(dataset_test_ori, cfg, data_type):
    """
    修改：现在直接使用测试集来加载遗忘集测试集的图片路径。
    """
    with open(f'/datanfs4/shenruoyan/FMUClip/Df/coco/forget_orange_test.txt', 'r') as f:
        df_ids = [i.strip() for i in f.readlines() if i.strip()]

    test_ids = {_normalize_annotation_id(ann, cfg) for ann in dataset_test_ori.annotation}

    # 过滤遗忘集图片，只保留那些在测试集中的图片
    filtered_ids = [img_id for img_id in df_ids if img_id in test_ids]
    ignored = sorted(set(df_ids) - set(filtered_ids))
    if ignored:
        sample_ignored = ', '.join(ignored[:5])
        logging.warning(
            "%d forget images are not part of the official test split and will be ignored: %s%s",
            len(ignored),
            sample_ignored,
            "..." if len(ignored) > 5 else "",
        )

    if not filtered_ids:
        raise ValueError("No forget images remaining after filtering against the test split.")

    logging.info(
        "Loaded %d forget images from list (requested %d).",
        len(filtered_ids),
        len(df_ids),
    )

    return filtered_ids, set(filtered_ids), test_ids


def prepare_dr_data(dataset_train_ori, cfg, data_type, sample_size=None, df_ids_set=None):
    if df_ids_set is None:
        _, df_ids_set, _ = _load_forget_train_ids(dataset_train_ori, cfg, data_type)

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

    # 确保固定随机种子，每次划分相同的批次
    np.random.seed(cfg.run_cfg.seed)
    np.random.shuffle(dataset.annotation)

    # 返回处理后的数据集
    return dataset

def prepare_df_data(dataset_train_ori, cfg, data_type, df_ids=None, df_ids_set=None, max_df_size=1000):
    if df_ids is None or df_ids_set is None:
        df_ids, df_ids_set, _ = _load_forget_train_ids(dataset_train_ori, cfg, data_type)

    dataset = copy.deepcopy(dataset_train_ori)

    # 限制遗忘集的大小（最多取前 max_df_size 个样本）
    if len(df_ids_set) > max_df_size:
        logging.info(f"Limiting the size of df to {max_df_size}.")
        df_ids_set = set(list(df_ids_set)[:max_df_size])  # 限制为最多1000个图像

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

    # 使用固定种子进行数据划分（打乱）
    np.random.seed(cfg.run_cfg.seed)  # 设置固定的种子
    np.random.shuffle(dataset.annotation)

    return dataset


def prepare_df_data_for_test(dataset_train_ori, dataset_test_ori, cfg, data_type, df_ids=None, df_ids_set=None, max_df_size=100):
    """
    准备遗忘集的测试数据，设置df大小上限，确保每次实验使用固定种子。
    """
    if df_ids is None or df_ids_set is None:
        df_ids, df_ids_set, _ = _load_forget_test_ids(dataset_test_ori, cfg, data_type)

    # 限制遗忘集的大小（最多取前 max_df_size 个样本）
    if len(df_ids_set) > max_df_size:
        logging.info(f"Limiting the size of df to {max_df_size}.")
        df_ids_set = set(list(df_ids_set)[:max_df_size])  # 限制为最多1000个图像

    dataset = copy.deepcopy(dataset_test_ori)

    # 获取遗忘集样本
    dataset.annotation = [
        ann for ann in dataset.annotation if os.path.basename(ann['image']) in df_ids_set
    ]
    
    # 确保固定随机种子，每次划分相同的批次
    np.random.seed(cfg.run_cfg.seed)
    np.random.shuffle(dataset.annotation)

    # 返回处理后的数据集
    return dataset


def prepare_dr_data_for_test(
    dataset_train_ori,
    dataset_test_ori,
    cfg,
    data_type,
    sample_size=None,
    df_ids_set=None,
):
    if df_ids_set is None:
        _, df_ids_set, _ = _load_forget_test_ids(dataset_test_ori, cfg, data_type)

    # 使用固定的随机种子，确保每次划分的数据一致
    np.random.seed(cfg.run_cfg.seed) 

    if cfg.run_cfg.task == 'retrieval':
        dr_for_test = copy.deepcopy(dataset_test_ori)
        annotation = [
            ann for ann in dataset_test_ori.annotation if os.path.basename(ann['image']) not in df_ids_set
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
            for caption in ann["caption"]:
                # 确保 caption 是字符串，若是列表则转换为字符串
                if isinstance(caption, list):
                    caption = " ".join(caption)  # 将列表连接成一个字符串
                text.append(text_processor(caption))  # 传递字符串给 text_processor
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
        # 文本数：优先用 dataset.text；没有就用 len(loader)
        if hasattr(loader.dataset, "text"):
            num_txt = len(loader.dataset.text)
        else:
            num_txt = len(loader)

        # 图像数：优先用 dataset.image；没有就用 base_dataset.image 或 image_embeds.size(0)
        if hasattr(loader.dataset, "image"):
            num_img = len(loader.dataset.image)
        else:
            base = getattr(loader.dataset, "base_dataset", loader.dataset)
            if hasattr(base, "image"):
                num_img = len(base.image)
            else:
                num_img = image_embeds.size(0)

        scores = torch.full((num_txt, num_img), -100.0,
                            dtype=torch.float16, device='cpu').pin_memory()

        # 3) 逐文本前向并写入该行
        for i, samples in enumerate(tqdm(loader, total=len(loader), ncols=150, desc="EVAL T2I")):
            raw_text = samples.get("text", None)
            tokenized_prompts = samples.get("tokenized_prompts", None)

            # -------- 关键：df/dr loader 只给 image + index，这里用 index 还原文本 --------
            if raw_text is None:
                idx = samples.get("index", None)
                if idx is not None:
                    idx = int(idx.item()) if hasattr(idx, "item") else int(idx)
                    # 先尝试 dataset.text（df/dr wrapper 一般有 text 列表）
                    if hasattr(loader.dataset, "text") and len(loader.dataset.text) > 0:
                        # 通常 df/dr 的 text 顺序和 loader 一致，直接用 i；不放心也可以用 idx
                        if i < len(loader.dataset.text):
                            raw_text = loader.dataset.text[i]
                        else:
                            raw_text = loader.dataset.text[idx]
                    else:
                        # 兜底：从 base_dataset.annotation 里拿 caption
                        base = getattr(loader.dataset, "base_dataset", loader.dataset)
                        ann = base.annotation[idx]
                        if "caption" in ann:
                            raw_text = ann["caption"]
                        elif "sentences" in ann and ann["sentences"]:
                            sent0 = ann["sentences"][0]
                            if isinstance(sent0, dict) and "raw" in sent0:
                                raw_text = sent0["raw"]

            if raw_text is None:
                raise ValueError("Cannot resolve text for T2I eval; got None")

            if tokenized_prompts is not None:
                tokenized_prompts = tokenized_prompts.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                txt_feat = model.get_text_features(
                    text=raw_text,
                    tokenized_prompts=tokenized_prompts
                )  # [1,D]
                logits = logit_scale * (txt_feat @ image_embeds.to(device).T)  # [1,N_i]
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
            with torch.no_grad():
                # 取出已缓存特征（均应已 L2 归一化）
                img_feats  = reward_model.image_features.to(model.device)     # [B, D]
                text_bank  = reward_model.text_features.to(model.device)      # [N_txt, D]
                logit_w    = getattr(reward_model, "clipscore_weight", 1.0)   # 与 CLIPScore 一致的缩放（如有）
                bs         = img_feats.size(0)
                num_txt    = text_bank.size(0)

                scores_ref_rows = []
                CHUNK = getattr(args, "select_chunk", 4096)  # 可通过 args 配；默认 4096
                for st in range(0, num_txt, CHUNK):
                    ed = min(num_txt, st + CHUNK)
                    # text 子块 [m, D]
                    t_blk = text_bank[st:ed]                                # [m, D]
                    # 相似度矩阵 [B, m] = [B, D] @ [D, m]
                    sc_blk = logit_w * (img_feats @ t_blk.t())              # [B, m]
                    scores_ref_rows.append(sc_blk)
                # 拼回整行 [B, N_txt]
                scores_ref = torch.cat(scores_ref_rows, dim=1)

            # 根据参考分（固定评判者）选候选
            value, index = torch.topk(
                scores_ref, sample_k, dim=-1,
                largest=(mode != "df")  # df 取最小K；dr 取最大K
            )
            text_index = index.flatten()
            # ——【新增】仅用于可读性展示：top-10 —— 
            with torch.no_grad():
                k = min(topk_print, logits_per_image.size(-1))
                v10, i10 = torch.topk(scores_ref[0], k, dim=-1)
                if sample_info is not None:
                    img_path = sample_info.get("image_path")
                    gt_txt_ids = sample_info.get("gt_txt_ids")
                    logging.info(
                        f"[{mode.upper()}][step {step+1}/{args.tta_steps}] "
                        f"image={img_path}  GT_txt_ids={gt_txt_ids}"
                    )

                    if gt_txt_ids is not None:
                        ids_to_eval = gt_txt_ids
                        if isinstance(ids_to_eval, torch.Tensor):
                            ids_to_eval = ids_to_eval.tolist()
                        elif isinstance(ids_to_eval, np.ndarray):
                            ids_to_eval = ids_to_eval.tolist()
                        elif not isinstance(ids_to_eval, (list, tuple, set)):
                            ids_to_eval = [ids_to_eval]

                        all_scores = logits_per_image[0].detach()
                        sorted_indices = torch.argsort(all_scores, descending=True)
                        ranks = torch.empty_like(sorted_indices, dtype=torch.long)
                        ranks[sorted_indices] = torch.arange(
                            1,
                            sorted_indices.numel() + 1,
                            device=sorted_indices.device,
                            dtype=torch.long,
                        )
                        rank_msgs = []
                        for gt_id in ids_to_eval:
                            if not isinstance(gt_id, (int, np.integer)):
                                try:
                                    gt_id = int(gt_id)
                                except Exception:
                                    continue
                            if 0 <= gt_id < all_scores.numel():
                                gt_rank = int(ranks[gt_id].item())
                                gt_score = float(all_scores[gt_id].item())
                                gt_text = None
                                if id2text is not None:
                                    try:
                                        gt_text = str(id2text(gt_id))
                                    except Exception:
                                        gt_text = None
                                if gt_text is not None:
                                    gt_text = gt_text.replace("\n", " ")
                                msg = f"id={gt_id} rank={gt_rank} score={gt_score:.4f}"
                                if gt_text is not None:
                                    msg += f" text={gt_text}"
                                rank_msgs.append(msg)
                        if rank_msgs:
                            logging.info(
                                f"[{mode.upper()}][step {step+1}/{args.tta_steps}] "
                                + " | ".join(rank_msgs)
                            )

                if id2text is not None:
                    block = _fmt_topk_rows(i10, v10, id2text)
                    logging.info("Top-10 (image→text):\n" + block)

            # ===== 权重/优势：df 用“bottom-k 越低权重越高、全非负”的奖励；dr 维持原逻辑 =====
            if mode == "df":
                # value: 由上面的 topk 选出来的“参考相似度”，这里是 bottom-k → 数值越低越“应该忘记”
                # 设计：把相似度低的候选给更大正权重；不产生负值（只奖励不惩罚）
                # 做法：对 -value 做 softmax（或对中心化后的 -value / tau 做 softmax 更稳）
                tau = float(getattr(args, "df_weight_tau", 0.05))  # 温度，越小越偏向最低者
                vals = value                                     # [B,K]，越低越“该忘”
                vals_c = vals - vals.mean(dim=-1, keepdim=True)  # 轻度中心化增强分辨率
                adv  = torch.softmax(-(vals_c) / tau, dim=-1).detach()  # 全正且和为1
                # （可选）再按需要做批后处理
                if reward_model.process_batch:
                    adv = reward_model.rewards_post_process(adv)
            else:
                # dr 分支：沿用你原先的 advantage 设计
                clip_score = reward_model.CLIPScore(text_index=text_index, pairwise=False)
                clip_score = clip_score.clamp_min(0.0)
                scores_2d = clip_score.view(bs, sample_k)
                adv = _centered_adv(scores_2d, mode=mode)
                if reward_model.process_batch:
                    adv = reward_model.rewards_post_process(adv)

            # 交叉熵作为 -log pθ(y|x) 的代理（REINFORCE: E[R * ∇log p]）
            rep_output = torch.repeat_interleave(logits_per_image, sample_k, dim=0)
            ce = F.cross_entropy(rep_output, text_index, reduction='none').view(bs, sample_k)

            # Df / Dr 的加权
            loss_df = (adv * ce).mean() if mode == "df" else torch.tensor(0.0, device=ce.device)
            loss_dr = (adv * ce).mean() if mode == "dr" else torch.tensor(0.0, device=ce.device)
            loss = lambda_df * loss_df + lambda_dr * loss_dr

            # === 改1：先把 loss_main 设成 loss（无论 df/dr 都有定义） ===
            loss_main = loss  # // 改

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
                loss_main = loss + drift_coef * drift_loss

        # ========== 两种模式 ==========
        # 1) “只计算不更新”（联合一步更新时使用）：直接返回当前 step 的分解损失
        compute_only = getattr(args, "joint_update", False)
        if compute_only:
            # 不做 backward/step，只把“主损失 +（如有）dr_kl”回传给外层
            ret = {
                "loss_main": loss_main,  # 标量 Tensor（未 detach，便于外层组合后统一 backward）
                "dr_kl": (drift_loss if (mode == "dr" and drift_coef > 0.0) else None),
            }
            return ret

        # 2) 常规单分支更新（你之前的行为）：反传与一步优化
        scaler.scale(loss_main).backward()
        scaler.unscale_(optimizer)

        total_sq = 0.0
        group_logs = []
        for g_idx, group in enumerate(optimizer.param_groups):
            group_sq = 0.0
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad_norm = param.grad.data.norm(2)
                if torch.isfinite(grad_norm):
                    group_sq += float(grad_norm.item()) ** 2
            group_norm = math.sqrt(group_sq) if group_sq > 0.0 else 0.0
            group_logs.append(f"g{g_idx}:{group_norm:.6f}")
            total_sq += group_sq
        total_grad_norm = math.sqrt(total_sq) if total_sq > 0.0 else 0.0
        scaler.step(optimizer)
        scaler.update()

        loss_df_val = float(loss_df.detach().item())
        loss_dr_val = float(loss_dr.detach().item())
        drift_val = float(drift_loss.detach().item()) if torch.is_tensor(drift_loss) else float(drift_loss)
        total_val = float(loss.detach().item())

        logging.info(
            f"[{mode.upper()}][step {step+1}/{args.tta_steps}] "
            f"loss={total_val:.6f} loss_df={loss_df_val:.6f} "
            f"loss_dr={loss_dr_val:.6f} drift={drift_val:.6f} "
            f"grad_norm={total_grad_norm:.6f} ({', '.join(group_logs)})"
        )

        # —— 累加到样本级统计 ——
        loss_df_sum += loss_df_val
        loss_dr_sum += loss_dr_val
        drift_sum   += drift_val
        total_sum   += total_val

    # —— 仅打印一次（8步平均）——（仅在常规单分支更新模式下会走到这里）
    if not getattr(args, "joint_update", False):
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
    reward_model.set_text_features(captions=text, tokenized_cap=tokenized_prompts)

    loss_df_sum = loss_dr_sum = drift_sum = total_sum = 0.0
    for step in range(args.tta_steps):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            # 前向：文本侧走可微路径；图像特征从缓存中取（_prepare_image_cache 预先填好）
            _, logits_per_text = model(images=None, text=text, tokenized_prompts=tokenized_prompts)

            # 采样 top-k 目标图像（和 tune_image 里对文本的做法完全对称）
            # 遗忘集(df) 采样“相似度最低”的 K 个候选；保留集(dr) 采样“最高”的 K
            value, index = torch.topk(
                logits_per_text, sample_k, dim=-1,
                largest=(mode != "df")
            )
            images_index = index.flatten()  # [B*K] = [K]

            # 原始 CLIPScore（与 image 侧保持一致，使用 reward_model 的统一接口）
            clip_score = reward_model.CLIPScore(
                text_index=None,
                images_index=images_index,
                pairwise=False,
            )
            clip_score = clip_score.clamp_min(0.0)

            # ---- 处理多 caption 情况 ----
            total = clip_score.numel()
            assert total % sample_k == 0, f"clip_score len={total}, sample_k={sample_k}"
            n_caps = total // sample_k            # 例如 60 / 12 = 5

            # [n_caps, K]
            scores_caps = clip_score.view(n_caps, sample_k)

            # 对多条 caption 的得分做平均 → [1, K]
            scores_2d = scores_caps.mean(dim=0, keepdim=True)

            # 居中优势（df: mean - score; dr: score - mean）
            adv = _centered_adv(scores_2d, mode=mode)
            adv = reward_model.rewards_post_process(adv) if reward_model.process_batch else adv

            # 交叉熵作为 -log pθ(y|x) 代理（与 image 侧一致）
            # logits_per_text 的 shape 此时大概率是 [n_caps, N_img]
            rep_output = torch.repeat_interleave(logits_per_text, sample_k, dim=0)  # [n_caps*K, I]

            # 先算出所有 caption×image 的交叉熵，再 reshape 回 [n_caps, K]
            ce_raw = F.cross_entropy(rep_output, images_index, reduction='none')    # [n_caps*K]
            ce_caps = ce_raw.view(n_caps, sample_k)                                 # [n_caps, K]

            # 多 caption 平均 → [1, K]
            ce = ce_caps.mean(dim=0, keepdim=True)

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
                drift_loss = F.kl_div(
                    log_p_cur, log_p_old,
                    reduction='batchmean',
                    log_target=True
                )
                loss_main = loss + drift_coef * drift_loss
            else:
                loss_main = loss

        # ====== 联合更新模式：只返回 loss，不在这里 step ======
        compute_only = getattr(args, "joint_update", False)
        if compute_only:
            return {
                "loss_main": loss_main,
                "dr_kl": (drift_loss if (mode == "dr" and drift_coef > 0.0) else None),
            }

        # ====== 普通模式：本函数内部直接更新 ======
        scaler.scale(loss_main).backward()
        scaler.step(optimizer)
        scaler.update()

        # —— 累加到样本级统计 —— 
        loss_df_sum += float(loss_df.detach().item())
        loss_dr_sum += float(loss_dr.detach().item())
        drift_sum   += float(drift_loss.detach().item()) if isinstance(drift_loss, torch.Tensor) else 0.0
        total_sum   += float(loss_main.detach().item())

    # —— 仅打印一次（steps 平均）——
    steps = max(1, args.tta_steps)
    logging.info(
        f"[{mode.upper()}] steps={args.tta_steps} "
        f"loss_df={loss_df_sum/steps:.6f} loss_dr={loss_dr_sum/steps:.6f} "
        f"drift={drift_sum/steps:.6f} total={total_sum/steps:.6f}"
    )


def training_split(
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
    extra_eval_loaders=None,
):
    """
    在 Df / Dr 上进行多轮（epoch）TTA + 评测。
    轮数由 args.max_epoch 控制（缺省为 1）
    每轮都会先训练（_train_joint），再评测、导出 topk 和日志
    所有轮数完成后再统一恢复权重
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
    
    def _get_text_from_dataset(dataset, index: int):
        """
        根据 df/dr 的 wrapper dataset + index，取回原始 Flickr 文本。
        优先走 base_dataset.annotation[index]['caption'] 这种典型 Flickr30k 格式。
        """
        # 有些 wrapper dataset 会有 base_dataset
        base = getattr(dataset, "base_dataset", dataset)

        # Flickr30k 在 lavis 里的典型结构是 annotation list
        ann = base.annotation[index]

        # 优先 caption
        if "caption" in ann:
            return ann["caption"]
        # 有些版本用 'sentences': [{'raw': ...}, ...]
        if "sentences" in ann and ann["sentences"]:
            sent0 = ann["sentences"][0]
            if isinstance(sent0, dict) and "raw" in sent0:
                return sent0["raw"]

        raise ValueError(f"Cannot find text for index={index}, ann keys={list(ann.keys())}")

    def _train_joint(df_loader, dr_loader, lam_df, lam_dr, drift):
        """
        联合一步更新：每个“大 step”用 1 个 Df batch 和 R 个 Dr batch（R=dr_df_ratio，默认 4）
        total_loss = (loss_df + sum_j loss_dr[j] + sum_j dr_kl[j]) / (R + 1)
        然后只做一次 backward/step。
        """
        # 标识联合模式：让 tune_image 只返回 loss，不自行优化
        args.joint_update = True
        model.train()

        # 比例（R），默认 4，可在命令行/配置中覆盖
        R = int(getattr(args, "dr_df_ratio", 1))
        assert R >= 1, "dr_df_ratio must be >= 1"

        # i2t：缓存文本库；t2i：缓存图像库
        if model.only_visual:
            _prepare_text_cache(dr_loader)
        else:
            _prepare_image_cache(dr_loader)

        df_it, dr_it = iter(df_loader), iter(dr_loader)
        # 可走的“大 step”数量：Df 的步数，且 Dr 足够每步提供 R 个 batch
        num_steps = min(len(df_loader), len(dr_loader) // R)

        # 若用户未显式设置 log_interval，则默认每步都打印一次
        if not hasattr(args, "log_interval") or args.log_interval is None:
            args.log_interval = 1
        pbar = tqdm(range(num_steps), ncols=150, desc=f"JOINT TTA R={R}:1")
        for step in pbar:
            try:
                # 1) 取一个 Df batch
                df_s = next(df_it)
            except StopIteration:
                break

            # 2) 取 R 个 Dr batch（若不够则停止）
            dr_batches = []
            for _ in range(R):
                try:
                    dr_s = next(dr_it)
                    dr_batches.append(dr_s)
                except StopIteration:
                    dr_batches = []
                    break
            if not dr_batches:
                break

            # ========= 分支 1：image2text（only_visual=True），和现在完全一致 =========
            if model.only_visual:
                # === 预处理：搬到设备，按最小 batch 对齐 ===
                df_img = df_s["image"].to(model.device, non_blocking=True)
                b = df_img.size(0)
                dr_imgs = []
                for s in dr_batches:
                    x = s["image"].to(model.device, non_blocking=True)
                    b = min(b, x.size(0))
                    dr_imgs.append(x)
                if df_img.size(0) != b:
                    df_img = df_img[:b]
                dr_imgs = [x[:b] if x.size(0) != b else x for x in dr_imgs]

                # 3) 只计算不更新 —— Df
                id2text = (lambda j: dr_loader.dataset.text[j]) if hasattr(dr_loader.dataset, "text") else None
                df_out = tune_image(
                    df_img, model, reward_model, optimizer, scaler,
                    mode="df", lambda_df=lam_df, lambda_dr=0.0,
                    drift_coef=0.0, init_feats=None, args=args,
                    sample_info=None, id2text=id2text, topk_print=10
                )
                # 4) 只计算不更新 —— Dr（R 个），收集到列表
                dr_out_list = []
                for x in dr_imgs:
                    init_feats = _frozen_init_image_feats(model, x) if drift > 0 else None
                    dr_out = tune_image(
                        x, model, reward_model, optimizer, scaler,
                        mode="dr", lambda_df=0.0, lambda_dr=lam_dr,
                        drift_coef=drift, init_feats=init_feats, args=args,
                        sample_info=None, id2text=id2text, topk_print=10
                    )
                    dr_out_list.append(dr_out)

            # ========= 分支 2：text2image（only_visual=False），用 tune_text 训练文本 encoder =========
            else:
                # 1) 从 Df batch 中拿 index → 还原文本
                df_idx_tensor = df_s["index"]
                # 既兼容 tensor([0]) 又兼容纯 int
                df_index = int(df_idx_tensor.item() if hasattr(df_idx_tensor, "item") else df_idx_tensor)
                df_text = _get_text_from_dataset(df_loader.dataset, df_index)

                # 2) 从 Dr batch 里拿 index → 还原文本（R 个）
                dr_texts = []
                for s in dr_batches:
                    idx_t = s["index"]
                    idx = int(idx_t.item() if hasattr(idx_t, "item") else idx_t)
                    dr_texts.append(_get_text_from_dataset(dr_loader.dataset, idx))

                # 3) id2image 用于日志可视化（从 base_dataset 里拿 image 列表）
                base_ds = getattr(dr_loader.dataset, "base_dataset", dr_loader.dataset)
                if hasattr(base_ds, "image"):
                    id2image = lambda j: base_ds.image[j]
                else:
                    # 兜底：从 annotation 里拿 image 字段
                    id2image = lambda j: base_ds.annotation[j]["image"]

                # 4) 只计算不更新 —— Df 文本 batch
                df_out = tune_text(
                    text=df_text,
                    tokenized_prompts=None,    # 让 reward_model 内部自己 tokenize
                    model=model,
                    reward_model=reward_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    mode="df",
                    lambda_df=lam_df,
                    lambda_dr=0.0,
                    drift_coef=0.0,
                    init_feats=None,
                    args=args,
                    sample_info={"text": df_text},
                    id2image=id2image,
                    topk_print=10,
                )

                # 5) 只计算不更新 —— Dr 文本 batch（R 个）
                dr_out_list = []
                for raw in dr_texts:
                    init_feats = _frozen_init_text_feats(
                        model,
                        text=raw,
                        tokenized_prompts=None
                    ) if drift > 0 else None

                    dr_out = tune_text(
                        text=raw,
                        tokenized_prompts=None,
                        model=model,
                        reward_model=reward_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        mode="dr",
                        lambda_df=0.0,
                        lambda_dr=lam_dr,
                        drift_coef=drift,
                        init_feats=init_feats,
                        args=args,
                        sample_info={"text": raw},
                        id2image=id2image,
                        topk_print=10,
                    )
                    dr_out_list.append(dr_out)

                # text2image 这边 b 用 1 就行（每个大 step 只处理 1 条文本）
                b = 1

            # ====== 后面这段组合 total_loss + 反传 + 日志，i2t/t2i 共用 ======
            total_loss = df_out["loss_main"]
            kl_acc = 0.0
            for dr_out in dr_out_list:
                total_loss = total_loss + dr_out["loss_main"]
                if dr_out.get("dr_kl") is not None:
                    total_loss = total_loss + dr_out["dr_kl"]
                    kl_acc = kl_acc + float(dr_out["dr_kl"].detach().item())
            denom = float(R + 1)
            total_loss = total_loss / denom

            # 6) 一次性反向更新 + 梯度范数统计
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            import math
            _group_logs, _total_sq = [], 0.0
            for g_idx, group in enumerate(optimizer.param_groups):
                _g_sq = 0.0
                for p in group.get("params", []):
                    if p.grad is None:
                        continue
                    gn = p.grad.data.norm(2)
                    if torch.isfinite(gn):
                        _g_sq += float(gn.item()) ** 2
                _group_logs.append(f"g{g_idx}:{math.sqrt(_g_sq) if _g_sq>0 else 0.0:.6f}")
                _total_sq += _g_sq
            _total_grad_norm = math.sqrt(_total_sq) if _total_sq > 0.0 else 0.0
            max_norm = float(getattr(args, "max_grad_norm", 1.0))
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # —— 新增：打印这一“大 step”的日志 —— 
            try:
                dfv = float(df_out["loss_main"].detach().item())
            except Exception:
                dfv = float(df_out["loss_main"].item())
            drv = float(torch.stack([dr["loss_main"] for dr in dr_out_list]).mean().detach().item())
            pbar.set_postfix_str(
                f"b={b}, df={dfv:.4f}, dr~={drv:.4f}, kl~={kl_acc/R if R>0 else 0:.4f}, "
                f"grad={_total_grad_norm:.6f}"
            )
            logging.info(
                f"[JOINT][step {step+1}/{num_steps}] "
                f"df={dfv:.6f} dr~={drv:.6f} kl~={kl_acc/R if R>0 else 0:.6f} "
                f"grad_norm={_total_grad_norm:.6f} ({', '.join(_group_logs)})"
            )

        # 退出联合模式
        args.joint_update = False


    # 原逻辑：每个样本重置权重
    # score_dr = _run_one_split(dr_loader, mode="dr", lam_df=0.0, lam_dr=lambda_dr, drift=drift_coef)
    # score_df = _run_one_split(df_loader, mode="df", lam_df=lambda_df, lam_dr=0.0, drift=0.0)

    # 保存原始权重与优化器状态，所有 epoch 完成后再统一恢复
    # saved_clip_state = copy.deepcopy(model.clip_model.state_dict())
    # saved_initial_state = copy.deepcopy(model.initial_state_dict)
    # saved_momentum_state = copy.deepcopy(getattr(model, "momentum_state_dict", None))
    # saved_update_counter = model.update_counter

    # === 多轮训练与评测 ===
    max_epoch = int(getattr(args, "max_epoch", 1))
    last_score_df, last_score_dr = None, None
    try:
        for epoch in range(1, max_epoch + 1):
            # —— 训练（联合一步更新）——
            _train_joint(df_loader, dr_loader, lambda_df, lambda_dr, drift_coef)

            # —— 评测 —— 
            task = "i2t" if model.only_visual else "t2i"
            score_df = eval_split_no_tta(df_loader, model, task=task, text_bs=text_bs)
            score_dr = eval_split_no_tta(dr_loader, model, task=task, text_bs=text_bs)
            last_score_df, last_score_dr = score_df, score_dr

            # —— 每轮导出 Top-10（文件名带 epoch 后缀）——
            topk_df_path = os.path.join(args.output, f"top10_{task}_df_e{epoch}.jsonl")
            topk_dr_path = os.path.join(args.output, f"top10_{task}_dr_e{epoch}.jsonl")
            _dump_topk_results(score_df, df_loader, task, topk_df_path, k=10)
            _dump_topk_results(score_dr, dr_loader, task, topk_dr_path, k=10)

            logging.info(f"[EPOCH {epoch}] finished.")
    finally:
        # —— 所有 epoch 完成后统一恢复初始权重（如需保留，可注释掉这部分）——
        # model.clip_model.load_state_dict(saved_clip_state)
        # model.initial_state_dict = copy.deepcopy(saved_initial_state)
        # if saved_momentum_state is not None:
        #     model.momentum_state_dict = copy.deepcopy(saved_momentum_state)
        # model.update_counter = saved_update_counter
        if optimizer is not None and optim_state is not None:
            optimizer.load_state_dict(optim_state)

    return last_score_df, last_score_dr

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
    
    forget_train_ids, forget_train_id_set, _ = _load_forget_train_ids(dtrain, cfg, data_type)
    forget_test_ids, forget_test_id_set, _ = _load_forget_test_ids(dtest, cfg, data_type)
    dr = prepare_dr_data(dtrain, cfg, data_type, df_ids_set=forget_train_id_set)
    df = prepare_df_data(dtrain, cfg, data_type, df_ids=forget_train_ids, df_ids_set=forget_train_id_set)
    df_for_test = prepare_df_data_for_test(
        dtrain, dtest, cfg, data_type, df_ids=forget_test_ids, df_ids_set=forget_test_id_set
    )

    retain_sample_size = len(df_for_test.annotation)  # 建议 *5，和后续 AUC 切片一致

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
    score_df, score_dr = training_split(
        df_loader, dr_loader, device, model, reward_model=reward_model,
        scaler=scaler, optimizer=optimizer, optim_state=optim_state,
        text_bs=128, args=args, lambda_df=args.lambda_df, lambda_dr=args.lambda_dr, drift_coef=args.drift_coef
    )

    # （最后一轮）分别汇报 Recall@K
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

    # 用训完的模型评测官方 Flickr30k 测试集（全量）
    # 用 runner 自带的 'test' dataloader
    test_loader = runner.dataloaders.get('test', None)
    if test_loader is not None:
        task_name = "i2t" if model.only_visual else "t2i"
        full_test_scores = eval_split_no_tta(test_loader, model, task=task_name, text_bs=128)
        # 汇报标准 Recall@K
        eval_test = task._report_metrics(
            full_test_scores,
            full_test_scores.T,
            test_loader.dataset.txt2img,
            test_loader.dataset.img2txt
        )
        test_out = os.path.join(args.output, f"results_{args.retrieval_task}_official_test.json")
        logging.info(test_out)
        with open(test_out, "w") as fp:
            json.dump({k: round(v, 3) for k, v in eval_test.items()}, fp, indent=4)
        logging.info(f"[OFFICIAL TEST] { {k: round(v, 3) for k, v in eval_test.items()} }")
    else:
        logging.warning("[OFFICIAL TEST] runner.dataloaders['test'] 不存在，已跳过官方测试集评测。")

if __name__ == "__main__":
    main()

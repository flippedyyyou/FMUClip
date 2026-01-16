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
from custom_models import CLIPRet
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
    else:              # 保留：高于基线应被奖励
        adv = (scores - mean) / std
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


def prepare_dr_data(dataset_train_ori, cfg, data_type):
    with open(f'/datanfs4/shenruoyan/FMUClip/Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)

    dataset = copy.deepcopy(dataset_train_ori)

    if cfg.run_cfg.task == 'retrieval':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if os.path.basename(i['image']) not in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.run_cfg.task == 'vqa':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 'nlvr':
        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if str(tuple(i['images'])) not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 've':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

    return dataset

def prepare_df_data(dataset_train_ori, cfg, data_type):
    with open(f'/datanfs4/shenruoyan/FMUClip/Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)

    dataset = copy.deepcopy(dataset_train_ori)
    
    if cfg.run_cfg.task == 'retrieval':
        dataset.annotation = [i for i in dataset.annotation if os.path.basename(i['image']) in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.run_cfg.task == 'vqa':
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 'nlvr':
        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if str(tuple(i['images'])) in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 've':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    # assert num_image_after_removal == cfg.run_cfg.df_size, f"{num_image_after_removal}, {cfg.run_cfg.df_size}"

    return dataset


def prepare_df_data_for_test(dataset_train_ori, dataset_test_ori, cfg, data_type):
    with open(f'/datanfs4/shenruoyan/FMUClip/Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)


    if cfg.run_cfg.task == 'retrieval':
        # Retrieval train and test data are different. We want to use retrieval test data for Df. So copy the ori test data
        df_for_test = copy.deepcopy(dataset_test_ori)

        annotation = [i for i in dataset_test_ori.annotation if os.path.basename(i['image']) in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in annotation]))

        # Convert to grouped format for init of RetrievalEvalDataset
        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')
        df_for_test.annotation = test_anno      # For __len__ method

        # init of RetrievalEvalDataset
        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = df_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                text.append(text_processor(caption))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

        df_for_test.text = text
        df_for_test.image = image
        df_for_test.txt2img = txt2img
        df_for_test.img2txt = img2txt

    # elif cfg.run_cfg.task == 'vqa':
    #     # breakpoint()
    #     # Retrieval train and test data are same. To use VQA test data for Df, copy the ori train data
    #     df_for_test = copy.deepcopy(dataset_train_ori)

    #     df_for_test.annotation = [i for i in df_for_test.annotation if i['image'] in df_ids_set]
    #     df_for_test._add_instance_ids()
    #     num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))
        # breakpoint()

    # elif cfg.run_cfg.task == 'multimodal_classification':
    #     breakpoint()
    #     df_for_test = copy.deepcopy(dataset_test_ori)

    #     df_for_test.annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
    #     df_for_test._add_instance_ids()
    #     num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))
    #     breakpoint()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [i for i in df_for_test.annotation if str(tuple(i['images'])) in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in df_for_test.annotation]))

    elif cfg.model_cfg.model_type in ['base', 've']:
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [i for i in df_for_test.annotation if i['image'] in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))

    # assert num_image_after_removal == cfg.run_cfg.df_size, f"{num_image_after_removal}, {cfg.run_cfg.df_size}"

    return df_for_test

def prepare_dr_data_for_test(dataset_train_ori, dataset_test_ori, cfg, data_type, sample_size=None):
    with open(f'/datanfs4/shenruoyan/FMUClip/Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)


    if cfg.run_cfg.task == 'retrieval':
        num_image_before_removal = len(set([i['image'] for i in dataset_train_ori.annotation]))

        # Retrieval train and test data are different. We want to use retrieval test data for Df. So copy the ori test data
        dr_for_test = copy.deepcopy(dataset_test_ori)

        annotation = [i for i in dataset_test_ori.annotation if os.path.basename(i['image']) not in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in annotation]))

        # Convert to grouped format for init of RetrievalEvalDataset
        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')
        dr_for_test.annotation = test_anno      # For __len__ method

        # init of RetrievalEvalDataset
        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = dr_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
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

        dr_for_test.annotation = [i for i in dr_for_test.annotation if i['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))
        # breakpoint()

    # elif cfg.run_cfg.task == 'multimodal_classification':
    #     breakpoint()
    #     dr_for_test = copy.deepcopy(dataset_test_ori)

    #     dr_for_test.annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
    #     dr_for_test._add_instance_ids()
    #     num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))
    #     breakpoint()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dr_for_test.annotation]))
        dr_for_test.annotation = [i for i in dr_for_test.annotation if str(tuple(i['images'])) not in df_ids_set]

        if sample_size is not None:
            anno_id = np.arange(len(dr_for_test.annotation))
            indices = np.random.choice(anno_id, sample_size, replace=False)
            dr_for_test.annotation = [dr_for_test.annotation[i] for i in indices]

        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dr_for_test.annotation]))

    elif cfg.model_cfg.model_type in ['base', 've']:
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        num_image_before_removal = len(set([i['image'] for i in dr_for_test.annotation]))
        dr_for_test.annotation = [i for i in dr_for_test.annotation if i['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

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
        # num_img, num_txt = len(loader.dataset.image), len(loader.dataset.text)
        # scores = torch.full((num_img, num_txt), -100.0, dtype=torch.float16, device='cpu').pin_memory()

        # # 3) 逐图像前向并写入该行
        # for i, samples in enumerate(tqdm(loader, total=len(loader), ncols=150, desc="EVAL I2T")):
        #     image = samples["image"].to(device, non_blocking=True)
        #     with torch.amp.autocast('cuda'):
        #         img_feat = model.get_image_features(image)                         # [1,D]
        #         logits = logit_scale * (img_feat @ text_embeds.to(device).T)      # [1,N_t]
        #     scores[i] = logits.squeeze(0).to('cpu', dtype=torch.float16, non_blocking=True)
        num_img, num_txt = len(loader.dataset.image), len(loader.dataset.text)
        scores = torch.full((num_img, num_txt), -100.0, dtype=torch.float16, device='cpu').pin_memory()

        # 3) 逐 batch 前向并写入对应的多行
        row = 0
        for samples in tqdm(loader, total=len(loader), ncols=150, desc="EVAL I2T"):
            image = samples["image"].to(device, non_blocking=True)                # [B, ...]
            with torch.amp.autocast('cuda'):
                img_feat = model.get_image_features(image)                        # [B, D]
                logits = logit_scale * (img_feat @ text_embeds.to(device).T)      # [B, N_t]
            B = logits.size(0)
            scores[row:row+B] = logits.to('cpu', dtype=torch.float16, non_blocking=True)
            row += B
        return scores.numpy()

    else:  # task == "t2i"
        # 1) 缓存全部图像特征 → CPU half
        with torch.amp.autocast('cuda'):
            image_embeds = get_all_image_embeds(loader, model)   # [N_i, D]
        image_embeds = image_embeds.half().cpu().pin_memory()

        # 2) 评分矩阵（CPU half + pinned）
        # num_txt, num_img = len(loader.dataset.text), len(loader.dataset.image)
        # scores = torch.full((num_txt, num_img), -100.0, dtype=torch.float16, device='cpu').pin_memory()

        # # 3) 逐文本前向并写入该行
        # for i, samples in enumerate(tqdm(loader, total=len(loader), ncols=150, desc="EVAL T2I")):
        #     raw_text = samples.get("text", None)
        #     tokenized_prompts = samples.get("tokenized_prompts", None)
        #     if tokenized_prompts is not None:
        #         tokenized_prompts = tokenized_prompts.to(device, non_blocking=True)
        #     with torch.amp.autocast('cuda'):
        #         txt_feat = model.get_text_features(text=raw_text, tokenized_prompts=tokenized_prompts)  # [1,D]
        #         logits = logit_scale * (txt_feat @ image_embeds.to(device).T)                            # [1,N_i]
        #     scores[i] = logits.squeeze(0).to('cpu', dtype=torch.float16, non_blocking=True)
        num_txt, num_img = len(loader.dataset.text), len(loader.dataset.image)
        scores = torch.full((num_txt, num_img), -100.0, dtype=torch.float16, device='cpu').pin_memory()

        # 3) 逐 batch 前向并写入对应的多行
        row = 0
        for samples in tqdm(loader, total=len(loader), ncols=150, desc="EVAL T2I"):
            raw_text = samples.get("text", None)                                   # list[str] 或 None
            tokenized_prompts = samples.get("tokenized_prompts", None)
            if tokenized_prompts is not None:
                tokenized_prompts = tokenized_prompts.to(device, non_blocking=True) # [B, L]
            with torch.amp.autocast('cuda'):
                txt_feat = model.get_text_features(text=raw_text, tokenized_prompts=tokenized_prompts)  # [B, D]
                logits = logit_scale * (txt_feat @ image_embeds.to(device).T)                           # [B, N_i]
            B = logits.size(0)
            scores[row:row+B] = logits.to('cpu', dtype=torch.float16, non_blocking=True)
            row += B
        return scores.numpy()


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


def zero_shot_split(df_loader, dr_loader, device, model, reward_model=None,
                         scaler=None, optimizer=None, optim_state=None,
                         text_bs=128, args=None, lambda_df=1.0, lambda_dr=2.0,
                         drift_coef=0.01):
    """
    在 Df / Dr 两个 split 上分别进行评测。
    - lambda_df / lambda_dr：对应你图里的 λ1 / λ2
    - drift_coef：保留集的特征保持正则系数(可选)
    """
    model.eval(); 
    device = model.device

    # 现逻辑：保存原始权重与优化器状态，确保保留和遗忘集都跑完之后再统一重置模型权重
    saved_clip_state = copy.deepcopy(model.clip_model.state_dict())
    saved_initial_state = copy.deepcopy(model.initial_state_dict)
    saved_momentum_state = copy.deepcopy(getattr(model, "momentum_state_dict", None))
    saved_update_counter = model.update_counter

    try:
        # —— 统一权重评测（一次性）——
        task = "i2t" if model.only_visual else "t2i"
        score_dr = eval_split_no_tta(dr_loader, model, task=task, text_bs=text_bs)
        score_df = eval_split_no_tta(df_loader, model, task=task, text_bs=text_bs)
    finally:
        # 所有样本迭代完成后再统一恢复为原始权重
        model.clip_model.load_state_dict(saved_clip_state)
        model.initial_state_dict = copy.deepcopy(saved_initial_state)
        if saved_momentum_state is not None:
            model.momentum_state_dict = copy.deepcopy(saved_momentum_state)
        model.update_counter = saved_update_counter
        if optimizer is not None and optim_state is not None:
            optimizer.load_state_dict(optim_state)

    return score_df, score_dr


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
    dr = prepare_dr_data(dtrain, cfg, data_type)
    df = prepare_df_data(dtrain, cfg, data_type)
    df_for_test = prepare_df_data_for_test(dtrain, dtest, cfg, data_type)
    dr_for_test = prepare_dr_data_for_test(dtrain, dtest, cfg, data_type, len(df_for_test.annotation)*5)
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

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # create model
    print("unlearn policy arch:", args.arch)
    model = CLIPRet(device, arch=args.arch, only_visual=(args.retrieval_task == "image2text"),
                            momentum_update=args.momentum_update, update_freq=args.update_freq,
                            update_w=args.update_w, momentum=args.tta_momentum)
    model = model.to(device)

    # define the CLIPRewards
    reward_model = get_reward_model(device, args)

    # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-06, weight_decay=args.weight_decay)
    optim_state = copy.deepcopy(optimizer.state_dict())

    # Df / Dr 分别做策略更新与评测
    score_df, score_dr = zero_shot_split(
        df_loader, dr_loader, device, model, reward_model=reward_model,
        scaler=scaler, optimizer=optimizer, optim_state=optim_state,
        text_bs=128, args=args, lambda_df=args.lambda_df, lambda_dr=args.lambda_dr, drift_coef=args.drift_coef
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


if __name__ == "__main__":
    main()

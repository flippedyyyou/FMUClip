# coding=utf-8
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

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image


from lavis.models.clip_models.tokenizer import tokenize as clip_tokenize
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
from lavis.models.clip_models.tokenizer import tokenize
from torch import nn

try:
    import clip as openai_clip  # OpenAI CLIP 的 tokenizer
except Exception:
    openai_clip = None
try:
    # LAVIS 自带的 CLIP tokenizer（有些环境可用）
    from lavis.models.clip_models.tokenizer import tokenize as lavis_tokenize
except Exception:
    lavis_tokenize = None


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

def _tokenize_texts(texts, context_length=77):
    """把 list[str] -> LongTensor token ids；若已是 Tensor 则原样返回。"""
    if isinstance(texts, torch.Tensor):
        return texts
    if isinstance(texts, (list, tuple)) and len(texts) > 0 and isinstance(texts[0], str):
        if openai_clip is not None:
            return openai_clip.tokenize(texts, context_length=context_length)
        if lavis_tokenize is not None:
            return lavis_tokenize(texts, context_length=context_length)
        raise RuntimeError("No tokenizer available: install `clip` or ensure LAVIS tokenizer is importable.")
    raise TypeError(f"Unsupported texts type: {type(texts)}")


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

def _to_token_ids(texts, device):
    """
    Normalize `texts` to a LongTensor token-id matrix [B, context_len]
    accepted inputs:
      - list[str]
      - single str
      - torch.LongTensor (already tokenized)
    """
    if isinstance(texts, torch.Tensor):
        return texts.to(device)
    # 2) 列表 / 元组
    if isinstance(texts, (list, tuple)):
        if len(texts) == 0:
            # 空列表也给个合理提示
            raise ValueError("texts is an empty list/tuple")

        first = texts[0]

        # 2a) 列表里是 Tensor：视为已经 tokenized，stack 再搬到 device
        if isinstance(first, torch.Tensor):
            return torch.stack(list(texts), dim=0).to(device)

        # 2b) 列表里是字符串：正常 tokenize
        if isinstance(first, str):
            return clip_tokenize(list(texts)).to(device)

        # 其它类型（比如混合类型）直接报错
        raise TypeError(
            f"Unsupported list element type: {type(first)}. "
            "Expect list[str] or list[LongTensor]."
        )

    # 3) 单个字符串
    if isinstance(texts, str):
        return clip_tokenize([texts]).to(device)

    # 4) 其它类型直接报错
    raise TypeError(f"Unsupported text type: {type(texts)}. Expect list[str] or LongTensor.")

def _resolve_clip_model(model):
    return model.clip_model if hasattr(model, "clip_model") else model


def _encode_image_patches(model, images: torch.Tensor):
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


def _load_sam3_masks(image_paths, mask_dir, mask_suffix, target_size):
    masks = []
    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{base}{mask_suffix}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"SAM3 mask not found: {mask_path}")
        mask = Image.open(mask_path).convert("L")
        if target_size is not None:
            mask = mask.resize((target_size, target_size), resample=Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        masks.append(mask_tensor)
    return torch.stack(masks, dim=0)


def _mask_to_patch_attention(mask_tensor: torch.Tensor, grid_size: int):
    mask_tensor = mask_tensor.unsqueeze(1)
    mask_resized = F.interpolate(mask_tensor, size=(grid_size, grid_size), mode="nearest")
    patch_mask = (mask_resized.squeeze(1) > 0.5).float()
    return patch_mask.flatten(1)

def _save_unlearned_checkpoint(model, output_dir, model_name, meta_name, cfg_path, args_dict, job_id):
    """Persist the unlearned CLIP checkpoint and a small metadata file.

    The function strips any DistributedDataParallel wrapper and stores the state dict so
    the weights can later be loaded in downstream projects such as LLaVA 1.5.
    """

    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model

    ckpt_path = os.path.join(output_dir, model_name)
    torch.save(model_to_save.state_dict(), ckpt_path)

    metadata = {
        "saved_at": now(),
        "job_id": job_id,
        "cfg_path": cfg_path,
        "model_class": model_to_save.__class__.__name__,
        "save_dir": os.path.abspath(output_dir),
        "args": args_dict,
    }

    meta_path = os.path.join(output_dir, meta_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logging.info("[Checkpoint] Unlearned CLIP saved to %s", ckpt_path)
    logging.info("[Checkpoint] Metadata saved to %s", meta_path)
    return ckpt_path, meta_path


def _get_logits_and_feats(model, images, texts, *, return_feats=True):
    """
    images: FloatTensor [B, 3, H, W] on device
    texts : list[str] or LongTensor [B, ctx]
    returns: sim_i2t [B,B], sim_t2i [B,B], img_feat [B,D], txt_feat [B,D]
    """
    device = images.device
    # --- image features ---
    img_feat = model.encode_image(images)                  # [B, D]
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    # --- text features (always LongTensor token ids) ---
    text_tokens = _to_token_ids(texts, device)             # [B, ctx] LongTensor
    txt_feat = model.encode_text(text_tokens)              # [B, D]
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    # logit scale (LAVIS CLIP exposes .logit_scale; fallback to .clip_model.logit_scale if present)
    ls_param = getattr(model, "logit_scale", None)
    if ls_param is None and hasattr(model, "clip_model"):
        ls_param = model.clip_model.logit_scale
    if ls_param is None:
        raise RuntimeError("Cannot find `logit_scale` on CLIP model.")
    logit_scale = ls_param.exp()

    sim_i2t = logit_scale * (img_feat @ txt_feat.t())      # [B, B]
    sim_t2i = logit_scale * (txt_feat @ img_feat.t())      # [B, B]

    if return_feats:
        return sim_i2t, sim_t2i, img_feat, txt_feat
    else:
        return sim_i2t, sim_t2i

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
    with open(f'/datanfs4/shenruoyan/FMUClip/Df/flickr30k/forget_dog_train.txt', 'r') as f:
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
    with open(f'/datanfs4/shenruoyan/FMUClip/Df/flickr30k/forget_dog_test.txt', 'r') as f:
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

    if cfg.run_cfg.task == 'retrieval':
        df_for_test = copy.deepcopy(dataset_test_ori)
        annotation = [
            ann for ann in dataset_test_ori.annotation if os.path.basename(ann['image']) in df_ids_set
        ]

        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')

        # >>> 新增：检索分支支持 sample_size 下采样，避免 OOM <<<
        if max_df_size is not None and max_df_size < len(test_anno):
            anno_id = np.arange(len(test_anno))
            indices = np.random.choice(anno_id, max_df_size, replace=False)
            test_anno = [test_anno[i] for i in indices]
        # <<< 新增结束 >>>

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
            for caption in ann["caption"]:
                # 确保 caption 是字符串，若是列表则转换为字符串
                if isinstance(caption, list):
                    caption = " ".join(caption)  # 将列表连接成一个字符串
                text.append(text_processor(caption))  # 传递字符串给 text_processor
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

        if max_df_size is not None:
            anno_id = np.arange(len(df_for_test.annotation))
            indices = np.random.choice(anno_id, max_df_size, replace=False)
            df_for_test.annotation = [df_for_test.annotation[i] for i in indices]

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
    print(f"Evaluating with {len(loader.dataset.image), len(loader.dataset.text)} samples.")
    """
    用“当前权重”一次性评测一个 split（不做任何更新）：
    - 评分矩阵放 CPU float16 + pinned（省显存，几乎不降速）
    返回：score 矩阵（numpy）
    """
    model.eval()
    device = model.device
    logit_scale = model.logit_scale.exp()

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
                img_feat = model.encode_image(image)             # [B, D]
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
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
            # 假设这里是在遍历 text loader / 或者在 image loader 外面 encode 全部文本
                for batch in loader:  # 具体变量名按你原来的来
                    # 1) 拿到 raw text（和你 train 里一样的写法）
                    raw_text = batch.get("text", batch.get("text_input", None))

                    # 统一成 list[str]
                    if isinstance(raw_text, (list, tuple)):
                        texts = list(raw_text)
                    else:
                        texts = [raw_text]

                    # 2) 用你已经写好的 _to_token_ids，绝不再出现 None
                    tokenized_prompts = _to_token_ids(texts, device)   # -> LongTensor [B, ctx]

                    # 3) encode_text
                    with torch.amp.autocast("cuda", enabled=True):
                        txt_feat = model.encode_text(tokenized_prompts)    # [B, D]

                    # 4) 正常收集 text 特征
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                # 归一化，和 image/text 特征的余弦相似度匹配
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
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
        with torch.no_grad():
            # LAVIS-CLIP: 直接传 token ids 给 encode_text
            text_features = model.encode_text(batch)           # [B, D]
            # 归一化，和 image/text 特征的余弦相似度匹配
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embeds.append(text_features)
        i += text_bs

    return torch.cat(text_embeds, dim=0)


def get_all_image_embeds(data_loader, model):
    """extract all image embeddings"""
    logging.info("Extracting ALL image features...")
    image_embeds = []
    for samples in data_loader:
        image = samples["image"].to(model.device)
        img_feats = model.encode_image(image)             # [B, D]
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        image_embeds.append(img_feats)

    return torch.cat(image_embeds, dim=0)

def _select_neg_texts_by_minsim(teacher, images, text_list):
    """
    给一批图像与其对应文本，使用 teacher 的 i2t 相似度矩阵，
    对每张图像选择“相似度最低”的文本作为负样本（并屏蔽对角，避免选到自身正样本）。
    返回：按每张图像对应的“最难负文本”重排后的文本列表（长度 B）
    """
    with torch.no_grad():
        # 先用 teacher 计算这批 (img, text_list) 的全量相似度矩阵
        sim_i2t_t, _, _, _ = _get_logits_and_feats(teacher, images, text_list)  # 期望形状 [B, B]
        # 屏蔽对角：防止选到本来的正例
        B = sim_i2t_t.size(0)
        mask = torch.eye(B, device=sim_i2t_t.device, dtype=sim_i2t_t.dtype) * 1e9
        sim_masked = sim_i2t_t + mask
        # 每行 argmin，得到每张图像对应的“最不相似”的文本索引
        neg_idx = torch.argmin(sim_masked, dim=1)   # [B]
    neg_idx_list = neg_idx.detach().cpu().tolist()
    neg_texts = [text_list[j] for j in neg_idx_list]
    return neg_texts

def _select_neg_texts_by_similarity_range(teacher, images, text_list, lower_percent=20, upper_percent=50):
    """
    给一批图像与其对应文本，使用 teacher 的 i2t 相似度矩阵，
    对每张图像选择相似度在指定百分比范围内的文本作为负样本（并屏蔽对角，避免选到自身正样本）。
    
    参数:
    - teacher: 训练好的教师模型
    - images: 当前批次的图像（Tensor）
    - text_list: 文本列表（所有文本）
    - lower_percent: 选择相似度最低的前百分之多少（例如20表示前20%）
    - upper_percent: 选择相似度最高的前百分之多少（例如50表示前50%）
    
    返回:
    - neg_texts: 选择的负样本文本列表
    """
    with torch.no_grad():
        # 计算图像与文本之间的相似度矩阵 [B, B]
        sim_i2t_t, _, _, _ = _get_logits_and_feats(teacher, images, text_list)
        
        # 屏蔽对角：防止选到本来的正例
        B = sim_i2t_t.size(0)
        mask = torch.eye(B, device=sim_i2t_t.device, dtype=sim_i2t_t.dtype) * 1e9
        sim_masked = sim_i2t_t + mask
        
        # 将相似度矩阵展平并进行排序
        sim_scores = sim_masked.view(-1)  # 展平后的相似度分数
        _, sorted_indices = torch.sort(sim_scores, descending=True)  # 从高到低排序
        
        # 调试输出，查看相似度的排序
        if len(sorted_indices) == 0:
            logging.warning("Sorted indices are empty. Check the similarity matrix.")
        
        # 计算需要选择的样本范围
        num_samples = sim_masked.size(0)
        lower_idx = int(num_samples * lower_percent / 100)
        upper_idx = int(num_samples * upper_percent / 100)
        
        # 调试输出，查看选定的范围
        # logging.info(f"Selecting samples from index {lower_idx} to {upper_idx} of {num_samples} samples.")

        # 选择在指定百分比范围内的负样本
        selected_indices = sorted_indices[lower_idx:upper_idx]

        # 如果选择的范围为空，记录警告
        if len(selected_indices) == 0:
            logging.warning(f"No samples selected in the range {lower_percent}% to {upper_percent}%.")
        
        # 将选择出的索引转换为图像-文本索引
        selected_image_indices = [idx // num_samples for idx in selected_indices]
        selected_text_indices = [idx % num_samples for idx in selected_indices]
        
        # 获取相应的文本列表
        neg_texts = [text_list[j] for j in selected_text_indices]
        
        # 如果返回的负样本列表为空，记录警告
        if len(neg_texts) == 0:
            logging.warning("No negative texts selected. The returned list is empty.")
        
    return neg_texts



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
        raise ValueError("sam3_mask_dir must be provided to compute attention-guided loss.")
    concept_tokens = _tokenize_texts([concept_token]).to(device)
    clip_model = _resolve_clip_model(model)
    visual = clip_model.visual if hasattr(clip_model, "visual") else None
    if visual is None or not hasattr(visual, "image_size"):
        raise AttributeError("CLIP visual encoder does not expose image_size for SAM3 mask mapping.")
    image_size = visual.image_size[0] if isinstance(visual.image_size, (tuple, list)) else visual.image_size

    # 以较短的一方为 epoch 基础步数
    iters_per_epoch = min(len(df_train_loader), len(dr_train_loader))
    for ep in range(max_epoch):
        model.train()
        df_iter = iter(df_train_loader)
        dr_iter = iter(dr_train_loader)
        running = {"attn": 0.0, "keep": 0.0, "uni": 0.0, "tot": 0.0}
        for it in range(iters_per_epoch):
            # ===== 1) 取 batch =====
            df_s = next(df_iter); dr_s = next(dr_iter)
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
            df_image_paths = df_s.get("image_path")

            B = img_df.size(0)

            if neg_mode == "shuffle":
                # ✅ baseline：随机打乱
                perm = torch.randperm(B).cpu()
                df_text_neg = [txt_df[i] for i in perm.tolist()]
            elif neg_mode == "simrange":
                # ✅ ours：对每张图像选 teacher 相似度在某范围内的文本
                df_text_neg = _select_neg_texts_by_similarity_range(teacher, img_df, txt_df, lower_percent=0, upper_percent=20)
            else:
                # ✅ ours：对每张图像选 teacher 最不相似的文本
                df_text_neg = _select_neg_texts_by_minsim(teacher, img_df, txt_df)
                
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                if df_image_paths is None:
                    raise KeyError("Df batch missing image_path; update dataset to return image_path for SAM3 masks.")
                sam3_masks = _load_sam3_masks(df_image_paths, sam3_mask_dir, sam3_mask_suffix, image_size)
                sam3_masks = sam3_masks.to(device, non_blocking=True)

                patch_feats, grid = _encode_image_patches(model, img_df)
                text_feat = _encode_text(model, concept_tokens).squeeze(0)
                patch_sim = torch.einsum("bpd,d->bp", patch_feats, text_feat)
                patch_attn = _mask_to_patch_attention(sam3_masks, grid)
                attn_sum = patch_attn.sum(dim=1)
                masked_similarity = patch_sim * patch_attn
                loss_attn = (masked_similarity.sum(dim=1) / attn_sum.clamp(min=1.0)).mean()

                # ===== 3) Dr：保持一致（与原模型 matched 分布接近）=====
                sim_i2t_dr_u, sim_t2i_dr_u, img_dr_u, txt_dr_u = _get_logits_and_feats(
                    model, img_dr, txt_dr
                )
                with torch.no_grad():
                    sim_i2t_dr_t, sim_t2i_dr_t, img_dr_t, txt_dr_t = _get_logits_and_feats(
                        teacher, img_dr, txt_dr
                    )
                loss_keep = mse(sim_i2t_dr_u, sim_i2t_dr_t) + mse(sim_t2i_dr_u, sim_t2i_dr_t)

                # ===== 4) 单模态保持（避免 encoder 漂移过大）=====
                loss_uni = mse(img_dr_u, img_dr_t) + mse(txt_dr_u, txt_dr_t)

                loss = lambda_md * loss_attn + lambda_keep * loss_keep + lambda_uni * loss_uni

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            # log
            running["attn"] += float(loss_attn.detach().item())
            running["keep"] += float(loss_keep.detach().item())
            running["uni"] += float(loss_uni.detach().item())
            running["tot"] += float(loss.detach().item())
            if (it + 1) % log_interval == 0:
                t = it + 1
                logging.info(
                    f"[EP {ep+1}/{max_epoch}] it={t}/{iters_per_epoch} "
                    f"attn={running['attn']/t:.4f} keep={running['keep']/t:.4f} "
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
                ce_img = F.cross_entropy(sim_i2t_u, targets, reduction="none")  # [B]
                ce_txt = F.cross_entropy(sim_t2i_u, targets, reduction="none")  # [B]

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
                p_img_t   = F.softmax(sim_i2t_t, dim=-1)        # [B, B]

                log_p_txt = F.log_softmax(sim_t2i_u, dim=-1)    # [B, B]
                p_txt_t   = F.softmax(sim_t2i_t, dim=-1)        # [B, B]

                kl_img_all = F.kl_div(log_p_img, p_img_t, reduction="none").sum(dim=-1)  # [B]
                kl_txt_all = F.kl_div(log_p_txt, p_txt_t, reduction="none").sum(dim=-1)  # [B]

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
    print(f"Loaded {len(forget_test_ids)} forget test images.")
    dr = prepare_dr_data(dtrain, cfg, data_type, df_ids_set=forget_train_id_set)
    df = prepare_df_data(dtrain, cfg, data_type, df_ids=forget_train_ids, df_ids_set=forget_train_id_set)
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

     # ========= 新增：判断是否只评测原始模型 =========
    use_original_only = bool(getattr(args, "original_eval", False)) or (
        getattr(args, "unlearn_method", "") in ["original", "none", "clip-original"]
    )

    if not use_original_only:
        # 只有在需要做遗忘训练时才初始化 scaler / optimizer / teacher / train_loader
        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        # create policy model（遗忘模型 / 被训练的模型）
        print("unlearn policy arch:", args.arch)

        # # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-06, weight_decay=args.weight_decay)
        optim_state = copy.deepcopy(optimizer.state_dict())

        # 冻结“原模型”作为 teacher
        teacher = copy.deepcopy(model).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # 训练用 dataloader（直接走 LAVIS dataset 的 __getitem__）
        bs_train = cfg.run_cfg.batch_size_train
        num_workers = cfg.run_cfg.num_workers
        df_train_loader = DataLoader(df, batch_size=bs_train, shuffle=True, num_workers=num_workers, drop_last=True)
        dr_train_loader = DataLoader(dr, batch_size=bs_train, shuffle=True, num_workers=num_workers, drop_last=True)


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
                lambda_uni=getattr(args, "lambda_uni", 1.0), # KL 权重
                max_epoch=getattr(args, "max_epoch", 1),
                log_interval=getattr(args, "log_interval", 50),
            )
        else:
            logging.info("[Train] Using Multidelete-style supervised baseline.")
            logging.info("[Train] Using MSE-based supervised baseline (shuffle/minsim).")
            # ========= 监督式 unlearning（替代原 TTA 训练），评测仍用 df/dr/test =========
            supervised_unlearn_train(
                cfg, model, teacher, df_train_loader, dr_train_loader,
                optimizer, scaler,
                lambda_md=getattr(args, "lambda_df", 1.0),          # Df 多模态解耦损失权重
                lambda_keep=getattr(args, "lambda_dr", 2.0),        # Dr 多模态保持损失权重
                lambda_uni=getattr(args, "lambda_uni", 0.1),        # 单模态 MSE 权重
                max_epoch=getattr(args, "max_epoch", 1),
                log_interval=getattr(args, "log_interval", 50),
                neg_mode=args.neg_mode,
                concept_token=getattr(args, "concept_token", "dog"),
                sam3_mask_dir=getattr(args, "sam3_mask_dir", None),
                sam3_mask_suffix=getattr(args, "sam3_mask_suffix", ".png"),
            )
    else:
        logging.info("[Eval-only] Skip unlearning. Directly evaluate ORIGINAL CLIP on df/dr/test.")


    # 监督训练完后，再在 df/dr/test 上按原规则评测（不做 TTA）
    task_name = "i2t" if args.retrieval_task == "image2text" else "t2i"
    # 确保 df_loader 只加载遗忘集的图像
    print(f"df_loader contains {len(df_loader.dataset.image)} images from the forget test set.")
    score_df = eval_split_no_tta(df_loader, model, task=task_name, text_bs=128)
    score_dr = eval_split_no_tta(dr_loader, model, task=task_name, text_bs=128)
    # 设置要查询的关键词
    keyword = "cat"  # 这里可以是任意关键词

    # 调用评估函数
    eval_df = task._report_metrics(score_df, score_df.T, df_loader.dataset.txt2img, df_loader.dataset.img2txt)
    eval_dr = task._report_metrics(score_dr, score_dr.T, dr_loader.dataset.txt2img, dr_loader.dataset.img2txt)

    # 输出
    for name, result in [("df", eval_df), ("dr", eval_dr)]:
        output_filename = os.path.join(args.output, f"results_{args.retrieval_task}_{name}.json")
        logging.info(output_filename)
        result = {k: round(v, 3) for k, v in result.items()}
        logging.info(result)
        with open(output_filename, "w") as fp:
            json.dump(result, fp, indent=4)

    # 训练后再根据评测得到的分数矩阵，导出 Top-K 结果
    _dump_topk_results(score_df, df_loader, task_name, os.path.join(args.output, "top10_i2t_df.jsonl"), k=10)
    _dump_topk_results(score_dr, dr_loader, task_name, os.path.join(args.output, "top10_i2t_dr.jsonl"), k=10)

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

    # 用训完的模型评测官方 Flickr30k 测试集（全量）
    # 用 runner 自带的 'test' dataloader
    test_loader = runner.dataloaders.get('test', None)
    if test_loader is not None:
        task_name = "i2t" if args.retrieval_task == "image2text" else "t2i"
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

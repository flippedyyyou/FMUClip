import os
import logging
import json
import copy
import pandas as pd

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from lavis.models.clip_models.tokenizer import tokenize as clip_tokenize
from lavis.common.utils import now
from lavis.models.clip_models.tokenizer import tokenize


try:
    import clip as openai_clip  # OpenAI CLIP 的 tokenizer
except Exception:
    openai_clip = None
try:
    # LAVIS 自带的 CLIP tokenizer（有些环境可用）
    from lavis.models.clip_models.tokenizer import tokenize as lavis_tokenize
except Exception:
    lavis_tokenize = None


def _tokenize_texts(texts, context_length=77):
    """把 list[str] -> LongTensor token ids；若已是 Tensor 则原样返回。"""
    if isinstance(texts, torch.Tensor):
        return texts
    if isinstance(texts, (list, tuple)) and len(texts) > 0 and isinstance(texts[0], str):
        if openai_clip is not None:
            return openai_clip.tokenize(texts, context_length=context_length)
        if lavis_tokenize is not None:
            return lavis_tokenize(texts, context_length=context_length)
        raise RuntimeError(
            "No tokenizer available: install `clip` or ensure LAVIS tokenizer is importable.")
    raise TypeError(f"Unsupported texts type: {type(texts)}")


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
    raise TypeError(
        f"Unsupported text type: {type(texts)}. Expect list[str] or LongTensor.")


def _resolve_clip_model(model):
    return model.clip_model if hasattr(model, "clip_model") else model


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
    with open(cfg.forget_train_file, 'r') as f:
        df_ids = [i.strip() for i in f.readlines() if i.strip()]

    train_ids = {_normalize_annotation_id(
        ann, cfg) for ann in dataset_train_ori.annotation}

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
        raise ValueError(
            "No forget images remaining after filtering against the training split.")

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
    with open(cfg.forget_test_file, 'r') as f:
        df_ids = [i.strip() for i in f.readlines() if i.strip()]

    test_ids = {_normalize_annotation_id(ann, cfg)
                for ann in dataset_test_ori.annotation}

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
        raise ValueError(
            "No forget images remaining after filtering against the test split.")

    logging.info(
        "Loaded %d forget images from list (requested %d).",
        len(filtered_ids),
        len(df_ids),
    )

    return filtered_ids, set(filtered_ids), test_ids


def prepare_dr_data(dataset_train_ori, cfg, data_type, sample_size=None, df_ids_set=None):
    if df_ids_set is None:
        _, df_ids_set, _ = _load_forget_train_ids(
            dataset_train_ori, cfg, data_type)

    dataset = copy.deepcopy(dataset_train_ori)

    if cfg.run_cfg.task == 'retrieval':
        dataset.annotation = [
            ann for ann in dataset.annotation if os.path.basename(ann['image']) not in df_ids_set
        ]

    elif cfg.run_cfg.task == 'vqa':
        dataset.annotation = [
            ann for ann in dataset.annotation if ann['image'] not in df_ids_set]
        dataset._add_instance_ids()

    elif cfg.model_cfg.model_type == 'nlvr':
        dataset.annotation = [
            ann for ann in dataset.annotation if str(tuple(ann['images'])) not in df_ids_set
        ]
        dataset._add_instance_ids()

    elif cfg.model_cfg.model_type == 've':
        dataset.annotation = [
            ann for ann in dataset.annotation if ann['image'] not in df_ids_set]
        dataset._add_instance_ids()

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

    # 确保固定随机种子，每次划分相同的批次
    np.random.seed(cfg.run_cfg.seed)
    np.random.shuffle(dataset.annotation)

    # 返回处理后的数据集
    return dataset


def prepare_df_data(dataset_train_ori, cfg, data_type, df_ids=None, df_ids_set=None, max_df_size=1000):
    if df_ids is None or df_ids_set is None:
        df_ids, df_ids_set, _ = _load_forget_train_ids(
            dataset_train_ori, cfg, data_type)

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
        dataset.annotation = [
            ann for ann in dataset.annotation if ann['image'] in df_ids_set]
        dataset._add_instance_ids()

    elif cfg.model_cfg.model_type == 'nlvr':
        dataset.annotation = [
            ann for ann in dataset.annotation if str(tuple(ann['images'])) in df_ids_set
        ]
        dataset._add_instance_ids()

    elif cfg.model_cfg.model_type == 've':
        dataset.annotation = [
            ann for ann in dataset.annotation if ann['image'] in df_ids_set]
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
        df_ids, df_ids_set, _ = _load_forget_test_ids(
            dataset_test_ori, cfg, data_type)

    if cfg.run_cfg.task == 'retrieval':
        df_for_test = copy.deepcopy(dataset_test_ori)
        annotation = [
            ann for ann in dataset_test_ori.annotation if os.path.basename(ann['image']) in df_ids_set
        ]

        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(
            ['image'])['caption'].apply(list).reset_index()
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

        df_for_test.annotation = [
            ann for ann in df_for_test.annotation if ann['image'] in df_ids_set]
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
            df_for_test.annotation = [
                df_for_test.annotation[i] for i in indices]

        df_for_test._add_instance_ids()

    elif cfg.model_cfg.model_type in ['base', 've']:
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [
            ann for ann in df_for_test.annotation if ann['image'] in df_ids_set]
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
        _, df_ids_set, _ = _load_forget_test_ids(
            dataset_test_ori, cfg, data_type)

    # 使用固定的随机种子，确保每次划分的数据一致
    np.random.seed(cfg.run_cfg.seed)

    if cfg.run_cfg.task == 'retrieval':
        dr_for_test = copy.deepcopy(dataset_test_ori)
        annotation = [
            ann for ann in dataset_test_ori.annotation if os.path.basename(ann['image']) not in df_ids_set
        ]

        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(
            ['image'])['caption'].apply(list).reset_index()
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

        dr_for_test.annotation = [
            ann for ann in dr_for_test.annotation if ann['image'] not in df_ids_set]
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
            dr_for_test.annotation = [
                dr_for_test.annotation[i] for i in indices]

        dr_for_test._add_instance_ids()

    elif cfg.model_cfg.model_type in ['base', 've']:
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        dr_for_test.annotation = [
            ann for ann in dr_for_test.annotation if ann['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()

    return dr_for_test


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
    # [B, ctx] LongTensor
    text_tokens = _to_token_ids(texts, device)
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


def _mask_to_patch_attention(mask_tensor: torch.Tensor, grid_size: int):
    mask_tensor = mask_tensor.unsqueeze(1)
    mask_resized = F.interpolate(mask_tensor, size=(
        grid_size, grid_size), mode="nearest")
    # 掩码值大于 0.3 的区域设为 1（前景），小于 0.3 的区域设为 0（背景）
    patch_mask = (mask_resized.squeeze(1) > 0.3).float()
    return patch_mask.flatten(1)  # [B, N]


def _centered_adv(scores: torch.Tensor, mode: str) -> torch.Tensor:
    """
    scores: [B, K] or [BK]
    mode: 'df' or 'dr'
    return: advantage scores aligned with input shape
    """
    if scores.dim() == 1:
        scores = scores.view(-1, 1)
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True) + 1e-6
    if mode == "df":
        adv = (mean - scores) / std
    else:
        adv = (scores - mean) / std
    return adv


def _select_neg_texts_by_minsim(teacher, images, text_list):
    """
    给一批图像与其对应文本，使用 teacher 的 i2t 相似度矩阵，
    对每张图像选择“相似度最低”的文本作为负样本（并屏蔽对角，避免选到自身正样本）。
    返回：按每张图像对应的“最难负文本”重排后的文本列表（长度 B）
    """
    with torch.no_grad():
        # 先用 teacher 计算这批 (img, text_list) 的全量相似度矩阵
        sim_i2t_t, _, _, _ = _get_logits_and_feats(
            teacher, images, text_list)  # 期望形状 [B, B]
        # 屏蔽对角：防止选到本来的正例
        B = sim_i2t_t.size(0)
        mask = torch.eye(B, device=sim_i2t_t.device,
                         dtype=sim_i2t_t.dtype) * 1e9
        sim_masked = sim_i2t_t + mask
        # 每行 argmin，得到每张图像对应的“最不相似”的文本索引
        neg_idx = torch.argmin(sim_masked, dim=1)   # [B]
    neg_idx_list = neg_idx.detach().cpu().tolist()
    neg_texts = [text_list[j] for j in neg_idx_list]
    return neg_texts


def _load_sam3_masks(image_paths, mask_dir, mask_suffix, target_size):
    masks = []
    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{base}{mask_suffix}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"SAM3 mask not found: {mask_path}")
        mask = Image.open(mask_path).convert("L")
        if target_size is not None:
            mask = mask.resize((target_size, target_size),
                               resample=Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        masks.append(mask_tensor)
    return torch.stack(masks, dim=0)


def tokenize_all_text(texts, model, text_bs=128):
    """tokenize all text and return: (text_ids)"""
    num_text = len(texts)
    text_ids = []
    i = 0
    while i < num_text:
        text = texts[i: min(num_text, i + text_bs)]
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
        batch = text_inputs[i: min(text_inputs.shape[0], i + text_bs)]
        with torch.no_grad():
            # LAVIS-CLIP: 直接传 token ids 给 encode_text
            text_features = model.encode_text(batch)           # [B, D]
            # 归一化，和 image/text 特征的余弦相似度匹配
            text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)
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


@torch.no_grad()
def eval_split_no_tta(loader, model, task="i2t", text_bs=128):
    print(
        f"Evaluating with {len(loader.dataset.image), len(loader.dataset.text)} samples.")
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
            text_embeds = get_all_text_embeds(
                text_ids, model, text_bs)  # [N_t, D]
        text_embeds = text_embeds.half().cpu().pin_memory()

        # 2) 评分矩阵（CPU half + pinned）
        num_img, num_txt = len(loader.dataset.image), len(loader.dataset.text)
        scores = torch.full((num_img, num_txt), -100.0,
                            dtype=torch.float16, device='cpu').pin_memory()
        # 3) 逐图像前向并写入该行
        for i, samples in enumerate(tqdm(loader, total=len(loader), ncols=150, desc="EVAL I2T")):
            image = samples["image"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                img_feat = model.encode_image(image)             # [B, D]
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = logit_scale * \
                    (img_feat @ text_embeds.to(device).T)      # [1,N_t]
            scores[i] = logits.squeeze(0).to(
                'cpu', dtype=torch.float16, non_blocking=True)
        return scores.numpy()

    else:  # task == "t2i"
        # 1) 缓存全部图像特征 → CPU half
        with torch.amp.autocast('cuda'):
            image_embeds = get_all_image_embeds(loader, model)   # [N_i, D]
        image_embeds = image_embeds.half().cpu().pin_memory()

        # 2) 评分矩阵（CPU half + pinned）
        num_txt, num_img = len(loader.dataset.text), len(loader.dataset.image)
        scores = torch.full((num_txt, num_img), -100.0,
                            dtype=torch.float16, device='cpu').pin_memory()

        # 3) 逐文本前向并写入该行
        for i, samples in enumerate(tqdm(loader, total=len(loader), ncols=150, desc="EVAL T2I")):
            raw_text = samples.get("text", None)
            tokenized_prompts = samples.get("tokenized_prompts", None)
            if tokenized_prompts is not None:
                tokenized_prompts = tokenized_prompts.to(
                    device, non_blocking=True)
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
                    tokenized_prompts = _to_token_ids(
                        texts, device)   # -> LongTensor [B, ctx]

                    # 3) encode_text
                    with torch.amp.autocast("cuda", enabled=True):
                        txt_feat = model.encode_text(
                            tokenized_prompts)    # [B, D]

                    # 4) 正常收集 text 特征
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                # 归一化，和 image/text 特征的余弦相似度匹配
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                # [1,N_i]
                logits = logit_scale * (txt_feat @ image_embeds.to(device).T)
            scores[i] = logits.squeeze(0).to(
                'cpu', dtype=torch.float16, non_blocking=True)
        return scores.numpy()


def _select_neg_texts_by_proximal(teacher, images, text_list):
    """
    给一批图像与其对应文本，使用 teacher 的 i2t 相似度矩阵，
    对每张图像选择每一行中除去对角线后最相似的索引（避免选到自身正样本）。
    返回:
    - neg_texts: 选择的负样本文本列表
    """
    with torch.no_grad():
        # 计算图像与文本之间的相似度矩阵 [B, B]
        sim_i2t_t, _, _, _ = _get_logits_and_feats(teacher, images, text_list)

        # 1. 克隆一份矩阵，避免原地修改原始数据
        matrix = sim_i2t_t.clone()
        
        # 2. 将对角线元素填充为负无穷大
        # 这样在取 argmax 时，程序就会自动忽略掉“自己”
        matrix.fill_diagonal_(float('-inf'))
        
        # 3. 在每一行（dim=1）寻找最大值的索引
        most_similar_indices = torch.argmax(matrix, dim=1)
        
        return most_similar_indices


def _dump_detailed_topk_results(scores_np, loader, task, out_file, k=10):
    """
    导出详细的评测结果到 JSONL：
    - I2T: 每张图检索到的 Top-10 文本 + 该图对应的 5 个真实 Caption
    - T2I: 每个文本检索到的 Top-10 图像文件名 + 该文本对应的真实图像文件名
    """
    # 确保在函数开头或全局已经 import os
    import os
    import json
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    import json
    import os

    K = max(1, k)
    scores = torch.from_numpy(scores_np)  # [N_query, N_candidate]
    dataset = loader.dataset

    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(scores.size(0)):
            row = scores[i]
            v, idx = torch.topk(row, min(K, row.numel()))

            result_entry = {}

            if task == "i2t":
                # 查询是图像，候选是文本
                img_path = dataset.image[i]
                query_basename = os.path.basename(img_path)

                # 1. 提取 Top-K 文本内容
                topk_items = []
                for jj, sc in zip(idx.tolist(), v.tolist()):
                    topk_items.append({
                        "text_id": jj,
                        "score": float(sc),
                        "content": dataset.text[jj]
                    })

                # 2. 提取真值 (通常是 5 个 captions)
                gt_indices = dataset.img2txt.get(i, [])
                gt_contents = [dataset.text[gi] for gi in gt_indices]

                result_entry = {
                    "image_basename": query_basename,
                    "top10_retrieved_texts": topk_items,
                    "ground_truth_captions": gt_contents
                }

            else:  # task == "t2i"
                # 查询是文本，候选是图像
                query_text = dataset.text[i]

                # 1. 提取 Top-K 图像文件名
                topk_items = []
                for jj, sc in zip(idx.tolist(), v.tolist()):
                    topk_items.append({
                        "image_id": jj,
                        "score": float(sc),
                        "image_basename": os.path.basename(dataset.image[jj])
                    })

                # 2. 提取真值图像
                gt_img_idx = dataset.txt2img.get(i, None)
                gt_basename = os.path.basename(
                    dataset.image[gt_img_idx]) if gt_img_idx is not None else "None"

                result_entry = {
                    "query_text": query_text,
                    "top10_retrieved_images": topk_items,
                    "ground_truth_image": gt_basename
                }

            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    logging.info(f"[Detailed TopK] Saved results to {out_file}")

# coding=utf-8
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

from clip_unlearn_reward import get_reward_model
from lavis.models.clip_models.model import load_openai_model
from lavis.models.clip_models.tokenizer import tokenize
import importlib.util
import sys


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_FRACTION = 0.1
TRAIN_FRACTION = 0.7

def _load_dict_from_path(file_path: str):
    """从指定路径动态加载包含 LABEL_NAMES 的字典"""
    spec = importlib.util.spec_from_file_location("mapping_module", file_path)
    mapping_module = importlib.util.module_from_spec(spec)
    sys.modules["mapping_module"] = mapping_module
    spec.loader.exec_module(mapping_module)
    if not hasattr(mapping_module, "LABEL_NAMES"):
        raise AttributeError(f"Mapping file {file_path} must contain 'LABEL_NAMES' dict.")
    return mapping_module.LABEL_NAMES

def _setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "unlearn_classification.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def _tokenize_texts(texts: Sequence[str], device: torch.device) -> torch.Tensor:
    tokens = tokenize(texts).to(device)
    return tokens


def _encode_text_features(model, text_tokens: torch.Tensor) -> torch.Tensor:
    feats = model.encode_text(text_tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _encode_image_features(model, images: torch.Tensor) -> torch.Tensor:
    feats = model.encode_image(images)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _get_logits_with_text_features(
    model,
    images: torch.Tensor,
    text_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_features = _encode_image_features(model, images)
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * (image_features @ text_features.t())
    return logits, image_features


def _get_logits_and_feats(
    model,
    images: torch.Tensor,
    text_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_features = _encode_image_features(model, images)
    text_features = _encode_text_features(model, text_tokens)
    logit_scale = model.logit_scale.exp()
    sim_i2t = logit_scale * (image_features @ text_features.t())
    sim_t2i = logit_scale * (text_features @ image_features.t())
    return sim_i2t, sim_t2i, image_features, text_features

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
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    patch_feats = x[:, 1:, :]
    patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)
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


class ClassificationDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        class_names: Sequence[str],
        indices: Sequence[int],
        use_index_path: bool = False,
    ) -> None:
        self.dataset = dataset
        self.class_names = list(class_names)
        self.indices = list(indices)
        self.use_index_path = use_index_path

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        data_idx = self.indices[idx]
        image, label = self.dataset[data_idx]
        if self.use_index_path:
            image_path = f"{data_idx}"
        else:
            image_path = getattr(self.dataset, "samples", None)
            if image_path is None:
                image_path = f"{data_idx}"
            else:
                image_path = image_path[data_idx][0]
        return {
            "image": image,
            "label": label,
            "text": self.class_names[label],
            "image_path": image_path,
        }


# def _load_forget_list(path: str) -> List[str]:
#     with open(path, "r", encoding="utf-8") as handle:
#         items = [line.strip() for line in handle if line.strip()]
#     return items
def _load_forget_jsonl(path: str) -> Tuple[List[int], Set[int]]:
    """返回索引列表和对应的类别 ID 集合"""
    indices = []
    forget_classes = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                indices.append(int(data["image_index"]))
                forget_classes.add(int(data["class_index"]))
    return indices, forget_classes


def _build_indices_from_list(
    dataset: Dataset,
    forget_list: Sequence[str],
) -> Set[int]:
    indices: Set[int] = set()
    sample_paths = None
    if hasattr(dataset, "samples"):
        sample_paths = [os.path.basename(p[0]) for p in dataset.samples]
    for entry in forget_list:
        try:
            indices.add(int(entry))
            continue
        except ValueError:
            pass
        if sample_paths is None:
            continue
        basename = os.path.basename(entry)
        if basename in sample_paths:
            idx = sample_paths.index(basename)
            indices.add(idx)
    return indices


def _build_text_pool(class_names: Sequence[str], retain_labels: Iterable[int]) -> List[str]:
    unique_labels = sorted(set(retain_labels))
    return [class_names[label] for label in unique_labels]


def _iter_labels(dataset: Dataset, indices: Sequence[int]) -> Iterable[int]:
    for idx in indices:
        _, label = dataset[idx]
        yield int(label)

def _split_eval_indices(
    dataset: ClassificationDataset,
    train_fraction: float = 0.7,
    test_sampling_ratio: float = 0.1, # 30% 之后再取 10%
) -> Tuple[List[int], List[int]]:
    per_class: dict[int, List[int]] = {}
    for idx in dataset.indices:
        _, label = dataset.dataset[idx]
        per_class.setdefault(int(label), []).append(idx)
    
    train_indices: List[int] = []
    test_indices: List[int] = []
    
    for label, indices in per_class.items():
        # 1. 划分为 70% 训练集
        split_point = int(len(indices) * train_fraction)
        train_part = indices[:split_point]
        remaining_part = indices[split_point:]
        
        # 2. 从剩下的 30% 中再取 10% 作为最终测试集
        test_count = max(1, int(len(remaining_part) * test_sampling_ratio)) if remaining_part else 0
        test_part = remaining_part[:test_count]
        
        train_indices.extend(train_part)
        test_indices.extend(test_part)
        
    return train_indices, test_indices

def _compute_text_features(
    model,
    class_names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    tokens = _tokenize_texts(class_names, device)
    with torch.no_grad():
        text_features = _encode_text_features(model, tokens)
    return text_features


def _evaluate_and_dump(
    model,
    dataset: Dataset,
    class_names: Sequence[str],
    device: torch.device,
    output_path: str,
    batch_size: int,
    num_workers: int,
    topk: int = 5,
) -> float:
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

    correct = 0
    total = 0
    k = min(topk, len(class_names))
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            image_paths = batch["image_path"]

            logits, _ = _get_logits_with_text_features(model, images, text_features)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            topk_scores, topk_indices = logits.topk(k, dim=1)
            for i in range(len(image_paths)):
                label_idx = int(labels[i].item())
                pred_idx = int(preds[i].item())
                matches = [
                    {
                        "index": int(topk_indices[i, j].item()),
                        "name": class_names[int(topk_indices[i, j].item())],
                        "score": float(topk_scores[i, j].item()),
                    }
                    for j in range(k)
                ]
                rows.append(
                    {
                        "image_path": image_paths[i],
                        "label_index": label_idx,
                        "label_name": class_names[label_idx],
                        "pred_index": pred_idx,
                        "pred_name": class_names[pred_idx],
                        "topk": matches,
                    }
                )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return correct / total if total else 0.0


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


def supervised_unlearn_train(
    model,
    teacher,
    reward_model,
    df_loader,
    dr_loader,
    text_pool: List[str],
    text_pool_tokens: torch.Tensor,
    optimizer,
    scaler,
    sam3_mask_dir: str,
    sam3_mask_suffix: str,
    label_mapping: dict,  # 修改点：传入加载好的字典映射
    lambda_attn: float,
    lambda_syn: float,
    lambda_keep: float,
    lambda_uni: float,
    max_epoch: int,
    log_interval: int,
) -> None:
    device = model.device
    mse = nn.MSELoss()

    reward_model.set_text_features(captions=text_pool)
    iters_per_epoch = min(len(df_loader), len(dr_loader))

    for ep in range(max_epoch):
        model.train()
        df_iter = iter(df_loader)
        dr_iter = iter(dr_loader)
        with torch.no_grad():
            text_pool_features = _encode_text_features(model, text_pool_tokens)

        running = {"attn": 0.0, "syn": 0.0, "keep": 0.0, "uni": 0.0, "tot": 0.0}
        for it in range(iters_per_epoch):
            df_s = next(df_iter)
            dr_s = next(dr_iter)
            img_df = df_s["image"].to(device, non_blocking=True)
            img_dr = dr_s["image"].to(device, non_blocking=True)
            df_labels = df_s["label"] # 获取遗忘样本的类别索引
            df_image_paths = df_s["image_path"]
            dr_texts = dr_s["text"]
            dr_text_tokens = tokenize(dr_texts).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                # 1. 动态获取 Concept Tokens：根据 label_mapping 匹配类名
                concept_texts = [label_mapping[int(lbl)] for lbl in df_labels]
                print("concept names:", concept_texts)
                concept_tokens = tokenize(concept_texts).to(device)

                sam3_masks = _load_sam3_masks(df_image_paths, sam3_mask_dir, sam3_mask_suffix, img_df.shape[-1])
                sam3_masks = sam3_masks.to(device, non_blocking=True)

                # 2. 计算 Attention Loss：使用对应类别的文本特征
                patch_feats, grid = _encode_image_patches(model, img_df)
                # 编码 batch 中每个样本对应的类别文本特征
                df_text_feats = _encode_text_features(model, concept_tokens) 
                
                # 计算每个 patch 与其对应类别文本特征的相似度
                patch_sim = F.cosine_similarity(patch_feats, df_text_feats.unsqueeze(1), dim=-1)
                patch_attn = _mask_to_patch_attention(sam3_masks, grid)
                attn_sum = patch_attn.sum(dim=1)
                masked_similarity = patch_sim * patch_attn
                # exp_similarity = torch.exp(masked_similarity)
                loss_attn = (masked_similarity.sum(dim=1) / attn_sum.clamp(min=1.0)).mean()

                sam3_binary = (sam3_masks > 0.5).float()
                syn_mask = 1.0 - sam3_binary
                syn_img = img_df * syn_mask.unsqueeze(1)

                reward_target = reward_model.clip_model.visual.image_size
                if isinstance(reward_target, (tuple, list)):
                    reward_target = reward_target[0]
                reward_img = syn_img
                if reward_img.shape[-1] != reward_target:
                    reward_img = F.interpolate(
                        reward_img,
                        size=reward_target,
                        mode="bicubic",
                        align_corners=True,
                    )
                reward_model.set_image_features(images=reward_img)
                clip_scores = reward_model.clipscore_weight * (
                    reward_model.image_features @ reward_model.text_features.t()
                )
                clip_probs = torch.softmax(clip_scores, dim=-1)
                sample_k = min(reward_model.sample_k, clip_scores.size(1))
                topk_probs, indices = torch.topk(clip_probs, sample_k, dim=-1, largest=True)
                adv = topk_probs.detach()

                sim_i2t_syn_u, img_syn_u = _get_logits_with_text_features(
                    model, syn_img, text_pool_features
                )
                rep_output = torch.repeat_interleave(sim_i2t_syn_u, sample_k, dim=0)
                text_index = indices.flatten()
                ce = F.cross_entropy(rep_output, text_index, reduction="none").view(img_df.size(0), sample_k)
                loss_syn = (adv * ce).mean()

                sim_i2t_dr_u, sim_t2i_dr_u, img_dr_u, txt_dr_u = _get_logits_and_feats(
                    model, img_dr, dr_text_tokens
                )
                with torch.no_grad():
                    sim_i2t_dr_t, sim_t2i_dr_t, img_dr_t, txt_dr_t = _get_logits_and_feats(
                        teacher, img_dr, dr_text_tokens
                    )
                loss_keep = mse(sim_i2t_dr_u, sim_i2t_dr_t) + mse(sim_t2i_dr_u, sim_t2i_dr_t)
                loss_uni = mse(img_dr_u, img_dr_t) + mse(txt_dr_u, txt_dr_t)

                loss = lambda_attn * loss_attn + lambda_keep * loss_keep + lambda_syn * loss_syn + lambda_uni * loss_uni

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["attn"] += float(loss_attn.detach().item())
            running["syn"] += float(loss_syn.detach().item())
            running["keep"] += float(loss_syn.detach().item())
            running["uni"] += float(loss_uni.detach().item())
            running["tot"] += float(loss.detach().item())

            if (it + 1) % log_interval == 0:
                t = it + 1
                logging.info(
                    "EP %d/%d it=%d/%d attn=%.4f syn=%.4f uni=%.4f total=%.4f",
                    ep + 1,
                    max_epoch,
                    t,
                    iters_per_epoch,
                    running["attn"] / t,
                    running["syn"] / t,
                    running["uni"] / t,
                    running["tot"] / t,
                )

def _compute_text_features(
    model,
    class_names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    tokens = _tokenize_texts(class_names, device)
    with torch.no_grad():
        text_features = _encode_text_features(model, tokens)
    return text_features


def _evaluate_and_dump(
    model,
    dataset: Dataset,
    class_names: Sequence[str],
    device: torch.device,
    output_path: str,
    batch_size: int,
    num_workers: int,
    topk: int = 5,
) -> float:
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

    correct = 0
    total = 0
    k = min(topk, len(class_names))
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            image_paths = batch["image_path"]

            logits, _ = _get_logits_with_text_features(model, images, text_features)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            topk_scores, topk_indices = logits.topk(k, dim=1)
            for i in range(len(image_paths)):
                label_idx = int(labels[i].item())
                pred_idx = int(preds[i].item())
                matches = [
                    {
                        "index": int(topk_indices[i, j].item()),
                        "name": class_names[int(topk_indices[i, j].item())],
                        "score": float(topk_scores[i, j].item()),
                    }
                    for j in range(k)
                ]
                rows.append(
                    {
                        "image_path": image_paths[i],
                        "label_index": label_idx,
                        "label_name": class_names[label_idx],
                        "pred_index": pred_idx,
                        "pred_name": class_names[pred_idx],
                        "topk": matches,
                    }
                )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return correct / total if total else 0.0

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
    model = load_openai_model(os.path.join(os.path.expanduser("/datanfs4/shenruoyan/checkpoints/clip"), f"{args.arch}.pt"), device, jit=False)
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
    forget_indices = set(forget_indices_list)

    # 2. 打印一下信息确认读取正确
    logging.info(f"从 JSONL 加载了 {len(forget_indices)} 个遗忘样本索引")

    # 3. 构建数据集 (传入类别集用于过滤保留集)
    df_dataset, dr_dataset, class_names = build_datasets(
        args.dataset,
        args.data_root,
        image_size=model.visual.image_size if isinstance(model.visual.image_size, int) else model.visual.image_size[0],
        forget_indices=forget_indices_list,
        forget_classes=forget_classes_set,
    )

    # 执行划分：70% 训练，测试集为剩余 30% 中的 10%
    df_train_indices, df_test_indices = _split_eval_indices(df_dataset)
    dr_train_indices, dr_test_indices = _split_eval_indices(dr_dataset)

    df_train_dataset = ClassificationDataset(
        df_dataset.dataset,
        df_dataset.class_names,
        df_train_indices,
        use_index_path=df_dataset.use_index_path,
    )
    df_test_dataset = ClassificationDataset(
        df_dataset.dataset,
        df_dataset.class_names,
        df_test_indices,
        use_index_path=df_dataset.use_index_path,
    )
    dr_train_dataset = ClassificationDataset(
        dr_dataset.dataset,
        dr_dataset.class_names,
        dr_train_indices,
        use_index_path=dr_dataset.use_index_path,
    )
    dr_test_dataset = ClassificationDataset(
        dr_dataset.dataset,
        dr_dataset.class_names,
        dr_test_indices,
        use_index_path=dr_dataset.use_index_path,
    )


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
    text_pool_tokens = _tokenize_texts(text_pool, device)

    args.multiple_reward_models = 0
    args.sample_k = args.sample_k
    reward_model = get_reward_model(device, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    logging.info("Unlearn configuration: %s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    supervised_unlearn_train(
        model,
        teacher,
        reward_model,
        df_loader,
        dr_loader,
        text_pool,
        text_pool_tokens,
        optimizer,
        scaler,
        sam3_mask_dir=args.sam3_mask_dir,
        sam3_mask_suffix=args.sam3_mask_suffix,
        label_mapping=label_mapping, # 传入加载的字典
        lambda_attn=args.lambda_attn,
        lambda_syn=args.lambda_syn,
        lambda_keep=args.lambda_keep,
        lambda_uni=args.lambda_uni,
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

    df_jsonl = os.path.join(args.output, "topk_df.jsonl")
    dr_jsonl = os.path.join(args.output, "topk_dr.jsonl")
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

    forget_labels = sorted(set(_iter_labels(df_test_dataset.dataset, df_test_dataset.indices)))
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
    metrics_path = os.path.join(args.output, "class_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
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
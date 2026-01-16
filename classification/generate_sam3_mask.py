import argparse
import os
import json
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image

# 从外部字典文件导入映射表
from datasets.cifar100 import LABEL_NAMES

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from torchvision import datasets, transforms

def _load_jsonl_entries(path: str) -> List[dict]:
    """读取遗忘集 JSONL 文件"""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries

def _save_jsonl_entries(path: str, entries: List[dict]):
    """将有效（生成了 Mask）的条目写回 JSONL"""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

def _maybe_load_cifar100(dataset_name, data_root, train, download):
    if dataset_name != "cifar100":
        return None
    return datasets.CIFAR100(root=data_root, train=train, download=download)

def _resolve_entry(image_index: int, cifar100: datasets.CIFAR100) -> Tuple[Image.Image, str]:
    image, _ = cifar100[image_index]
    if not isinstance(image, Image.Image):
        image = transforms.ToPILImage()(image)
    return image.convert("RGB"), str(image_index)

def main():
    parser = argparse.ArgumentParser(description="根据 JSONL 文件自动匹配类别生成 SAM3 Mask")
    parser.add_argument("--jsonl-path", required=True, help="输入的遗忘集 JSONL 文件路径")
    parser.add_argument("--output-dir", required=True, help="Mask 保存目录")
    parser.add_argument("--bpe-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 SAM3 模型
    model = build_sam3_image_model(bpe_path=args.bpe_path, checkpoint_path=args.checkpoint)
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)

    # 加载数据集和 JSONL
    jsonl_entries = _load_jsonl_entries(args.jsonl_path)
    cifar100 = _maybe_load_cifar100("cifar100", args.data_root, args.train, False)
    
    valid_entries = []

    for entry in jsonl_entries:
        img_idx = entry["image_index"]
        cls_idx = entry["class_index"]
        
        # 核心逻辑：从导入的 LABEL_NAMES 自动匹配 Prompt
        if cls_idx not in LABEL_NAMES:
            print(f"警告：类别索引 {cls_idx} 在映射表中不存在，跳过索引 {img_idx}")
            continue
        
        current_prompt = LABEL_NAMES[cls_idx]
        image, base = _resolve_entry(img_idx, cifar100)

        # 设置图像和动态 Prompt
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=current_prompt)

        masks = state["masks"]
        scores = state["scores"]
        
        if masks.numel() == 0:
            print(f"跳过：索引 {img_idx} ({current_prompt}) 未找到有效分割区域")
            continue 

        best_idx = torch.argmax(scores).item()
        if scores[best_idx].item() < args.confidence_threshold:
            print(f"跳过：索引 {img_idx} 分数过低")
            continue 

        # 保存结果
        best_mask = masks[best_idx, 0].cpu().numpy().astype(np.uint8) * 255
        out_path = os.path.join(args.output_dir, f"{base}.png")
        Image.fromarray(best_mask, mode="L").save(out_path)
        
        print(f"成功处理：索引 {img_idx} | 类别: {current_prompt}")
        valid_entries.append(entry)

    # 更新 JSONL（仅保留有 Mask 的样本，确保后续训练代码读取时不报错）
    _save_jsonl_entries(args.jsonl_path, valid_entries)

if __name__ == "__main__":
    main()
import importlib.util
import json
import logging
import os
import sys
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from lavis.models.clip_models.tokenizer import tokenize



def _load_dict_from_path(file_path: str):
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


def _iter_labels(dataset: Dataset, indices: Sequence[int]) -> Iterable[int]:
    for idx in indices:
        _, label = dataset[idx]
        yield int(label)


def _split_eval_indices(
    dataset: ClassificationDataset,
    train_fraction: float = 0.7,
    test_fraction: float = 0.3,
    max_test_per_class: int = 50,
    max_train_count: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    per_class: dict[int, List[int]] = {}
    for idx in dataset.indices:
        _, label = dataset.dataset[idx]
        per_class.setdefault(int(label), []).append(idx)

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []

    for _, indices in per_class.items():
        split_point = int(len(indices) * train_fraction)
        test_count = int(len(indices) * test_fraction)
        test_count = min(test_count, max_test_per_class)
        test_count = min(test_count, len(indices) - split_point)

        train_part = indices[:split_point]
        test_part = indices[split_point: split_point + test_count]

        train_indices.extend(train_part)
        test_indices.extend(test_part)
    rng.shuffle(train_indices)
    if max_train_count is not None and len(train_indices) >= max_train_count:
        train_indices = train_indices[:max_train_count]

    return train_indices, test_indices


def _tokenize_texts(texts: Sequence[str], device: torch.device) -> torch.Tensor:
    if isinstance(texts, torch.Tensor):
        return texts.to(device)
    return tokenize(texts).to(device)

def _compute_text_features(
    model,
    class_names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    tokens = _tokenize_texts(class_names, device)
    with torch.no_grad():
        text_features = _encode_text_features(model, tokens)
    return text_features


def _encode_text_features(model, text_tokens: torch.Tensor) -> torch.Tensor:
    feats = model.encode_text(text_tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

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

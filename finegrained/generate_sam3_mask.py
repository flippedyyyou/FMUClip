import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finegrained.coco_labels.coco import LABEL_NAMES as COCO_LABEL_NAMES
from finegrained.flickr30k_labels.flickr30k import LABEL_NAMES as FLICKR30K_LABEL_NAMES
from finegrained.load_dataset import get_finegrained_dataset_cls


def build_sam3_processor(
    bpe_path: str,
    checkpoint_path: str,
    confidence_threshold: float = 0.5,
) -> Sam3Processor:
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path)
    return Sam3Processor(model, confidence_threshold=confidence_threshold)


def get_best_mask_for_prompt_from_state(
    processor: Sam3Processor,
    image_state,
    prompt: str,
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Run SAM3 text prompting on an existing image state and return the highest-score mask.

    Returns:
        (mask, score), where mask shape is [H, W] bool tensor on CPU.
        If SAM3 returns no candidate masks, returns (None, -inf).
    """
    state = processor.set_text_prompt(state=image_state, prompt=prompt)
    masks = state["masks"]
    scores = state["scores"]
    if masks.numel() == 0:
        return None, float("-inf")
    best_idx = torch.argmax(scores).item()
    best_score = float(scores[best_idx].item())
    best_mask = (masks[best_idx, 0].detach().cpu() > 0)
    return best_mask, best_score


def get_best_mask_for_prompt(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
) -> Tuple[Optional[torch.Tensor], float]:
    image_state = processor.set_image(image)
    return get_best_mask_for_prompt_from_state(processor, image_state, prompt)


def _normalize_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def _parse_forget_classes(raw: str) -> List[str]:
    items = [_normalize_name(x) for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("`--forget-classes` cannot be empty.")
    return items


def _read_txt_image_list(txt_path: str) -> List[str]:
    items: List[str] = []
    if not os.path.exists(txt_path):
        return items
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(os.path.basename(line))
    return items


def _load_images_from_df_lists(
    df_root: str,
    split: str,
    item_folder: str,
    class_names: Sequence[str],
) -> List[str]:
    out: List[str] = []
    seen = set()
    for class_name in class_names:
        txt_path = os.path.join(df_root, split, "Df", item_folder, f"{_normalize_name(class_name)}.txt")
        for x in _read_txt_image_list(txt_path):
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
    return out


def _pack_bool_mask(mask_bool: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    arr = mask_bool.detach().cpu().numpy().astype(np.uint8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    bits = np.packbits(arr.reshape(-1), bitorder="little")
    return torch.from_numpy(bits), (h, w)


def _resolve_dataset_assets(
    dataset: str,
    coco_root: str,
    split: str,
    flickr_instances_file: str,
    flickr_image_root: str,
) -> Tuple[Dict[int, str], str, str]:
    normalized = str(dataset).strip().lower()
    if normalized in {"coco", "coco2017", "coco2017_instances"}:
        annotation_file = os.path.join(coco_root, "annotations", f"instances_{split}2017.json")
        image_root = os.path.join(coco_root, f"{split}2017")
        return COCO_LABEL_NAMES, annotation_file, image_root

    if normalized in {"flickr", "flickr30k", "flickr30k_entities"}:
        if not flickr_instances_file:
            raise ValueError("`--flickr-instances-file` is required when --dataset=flickr30k_entities.")
        if not flickr_image_root:
            raise ValueError("`--flickr-image-root` is required when --dataset=flickr30k_entities.")
        return FLICKR30K_LABEL_NAMES, flickr_instances_file, flickr_image_root

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_retain_mask_cache(
    dataset: str,
    df_root: str,
    annotation_file: str,
    image_root: str,
    label_names: Dict[int, str],
    split: str,
    item_folder: str,
    forget_classes: Sequence[str],
    bpe_path: str,
    checkpoint_path: str,
    confidence_threshold: float,
    out_path: str,
) -> None:
    class_names = [name for _, name in sorted(label_names.items())]
    forget_names = {_normalize_name(x) for x in forget_classes}
    selected_files = _load_images_from_df_lists(
        df_root=df_root,
        split=split,
        item_folder=item_folder,
        class_names=list(forget_names),
    )
    if not selected_files:
        raise ValueError("No training images found from Df lists.")

    dataset_cls = get_finegrained_dataset_cls(dataset)
    dataset_obj = dataset_cls(
        label_names=label_names,
        annotation_file=annotation_file,
        image_root=image_root,
        split=split,
        selected_files=selected_files,
        forget_class_names=list(forget_names),
        return_meta=False,
    )
    processor = build_sam3_processor(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        confidence_threshold=confidence_threshold,
    )

    entries: Dict[str, Dict[str, object]] = {}
    miss_retain, miss_sam = 0, 0
    for i in range(len(dataset_obj)):
        file_name = dataset_obj.file_names[i]
        stem = os.path.splitext(file_name)[0]
        image_path = os.path.join(dataset_obj.image_root, file_name)
        retain_vec = torch.from_numpy(dataset_obj.targets[i].copy()) * (1.0 - dataset_obj.forget_mask)
        candidate_indices = torch.nonzero(retain_vec > 0.5).flatten().tolist()
        if not candidate_indices:
            miss_retain += 1
            continue

        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image_state = processor.set_image(image)
        best_mask = None
        best_score = float("-inf")
        best_cls_idx = -1
        for cidx in candidate_indices:
            prompt = class_names[int(cidx)].replace("_", " ")
            mask_tensor, score = get_best_mask_for_prompt_from_state(
                processor=processor,
                image_state=image_state,
                prompt=prompt,
            )
            if mask_tensor is None:
                continue
            if float(score) > best_score:
                best_mask = mask_tensor
                best_score = float(score)
                best_cls_idx = int(cidx)
        if best_mask is None:
            miss_sam += 1
            continue
        packed, shape = _pack_bool_mask(best_mask > 0)
        entries[stem] = {
            "class_idx": int(best_cls_idx),
            "mask_bits": packed,
            "shape": [int(shape[0]), int(shape[1])],
            "score": float(best_score),
        }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "version": 1,
            "split": split,
            "item_folder": item_folder,
            "forget_classes": list(forget_names),
            "num_images": len(dataset_obj),
            "num_cached": len(entries),
            "num_missing_retain_label": int(miss_retain),
            "num_missing_sam_mask": int(miss_sam),
            "entries": entries,
        },
        out_path,
    )
    print(
        f"Saved retain cache to {out_path}. "
        f"cached={len(entries)}/{len(dataset_obj)}, miss_retain={miss_retain}, miss_sam={miss_sam}"
    )


def _load_image_paths(list_path, image_root=None):
    with open(list_path, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    if image_root:
        paths = [
            p if os.path.isabs(p) else os.path.join(image_root, p)
            for p in paths
        ]
    return paths


def _save_image_paths(list_path, image_paths):
    with open(list_path, "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(f"{path}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch-generate SAM3 binary masks for forget images.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco2017_instances",
        choices=["coco2017_instances", "flickr30k_entities"],
        help="Finegrained dataset backend.",
    )
    parser.add_argument("--build-retain-cache", action="store_true")
    parser.add_argument("--image-list", default="", help="txt file with image paths (one per line)")
    parser.add_argument("--image-root", default="", help="optional root to prepend to relative paths")
    parser.add_argument("--output-dir", default="", help="directory to save binary mask pngs")
    parser.add_argument("--prompt", default="", help="text prompt for SAM3 segmentation")
    parser.add_argument("--bpe-path", required=True, help="path to SAM3 BPE vocab")
    parser.add_argument("--checkpoint", required=True, help="path to SAM3 checkpoint")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--df-root", default="")
    parser.add_argument("--coco-root", default="")
    parser.add_argument("--flickr-instances-file", default="")
    parser.add_argument("--flickr-image-root", default="")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-item-folder", default="item3")
    parser.add_argument("--forget-classes", default="")
    parser.add_argument("--retain-cache-out", default="")
    args = parser.parse_args()

    if args.build_retain_cache:
        if not args.df_root or not args.retain_cache_out or not args.forget_classes:
            raise ValueError(
                "For --build-retain-cache, require --df-root --retain-cache-out --forget-classes."
            )
        label_names, train_annotation_file, train_image_root = _resolve_dataset_assets(
            dataset=args.dataset,
            coco_root=args.coco_root,
            split=args.train_split,
            flickr_instances_file=args.flickr_instances_file,
            flickr_image_root=args.flickr_image_root,
        )
        build_retain_mask_cache(
            dataset=args.dataset,
            df_root=args.df_root,
            annotation_file=train_annotation_file,
            image_root=train_image_root,
            label_names=label_names,
            split=args.train_split,
            item_folder=args.train_item_folder,
            forget_classes=_parse_forget_classes(args.forget_classes),
            bpe_path=args.bpe_path,
            checkpoint_path=args.checkpoint,
            confidence_threshold=args.confidence_threshold,
            out_path=args.retain_cache_out,
        )
        return

    if not args.image_list or not args.output_dir or not args.prompt:
        raise ValueError("Require --image-list --output-dir --prompt for forget-mask generation mode.")

    os.makedirs(args.output_dir, exist_ok=True)

    processor = build_sam3_processor(
        bpe_path=args.bpe_path,
        checkpoint_path=args.checkpoint,
        confidence_threshold=args.confidence_threshold,
    )

    image_paths = _load_image_paths(args.image_list, args.image_root or None)
    if not image_paths:
        raise ValueError("No image paths found.")

    valid_image_paths = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        best_mask_tensor, confidence = get_best_mask_for_prompt(
            processor=processor,
            image=image,
            prompt=args.prompt,
        )
        if best_mask_tensor is None:
            print(f"Warning: No masks for {image_path}. Consider lowering confidence_threshold.")
            continue  # Skip this image if no masks are found

        if confidence < args.confidence_threshold:
            print(f"Warning: Confidence for {image_path} is below threshold. Removing from list.")
            continue  # Skip this image and do not save the mask

        best_mask = best_mask_tensor.numpy().astype(np.uint8) * 255
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(args.output_dir, f"{base}.png")
        Image.fromarray(best_mask, mode="L").save(out_path)
        print(f"Saved {out_path}")

        valid_image_paths.append(image_path)  # Keep valid image paths

    # Save the remaining valid image paths back to the text file
    _save_image_paths(args.image_list, valid_image_paths)


if __name__ == "__main__":
    main()

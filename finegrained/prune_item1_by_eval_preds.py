#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class PredEntry:
    label: str
    pred_name: str
    image_basename: str
    coco_file_name: str
    pred_score: float
    label_score: float


def extract_coco_file_name(image_basename: str) -> str:
    if "_" in image_basename:
        return image_basename.split("_", 1)[1]
    return image_basename


def read_predictions(eval_dir: str) -> Dict[str, List[PredEntry]]:
    grouped: Dict[str, List[PredEntry]] = defaultdict(list)
    jsonl_files = sorted(glob.glob(os.path.join(eval_dir, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files found under: {eval_dir}")

    for jsonl_path in jsonl_files:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                label = obj["label"]
                pred_name = obj["pred_name"]
                image_basename = os.path.basename(obj["image_path"])
                coco_file_name = extract_coco_file_name(image_basename)

                top5 = obj.get("top5", [])
                pred_score = None
                label_score = None
                for item in top5:
                    name = item.get("name")
                    score = float(item.get("score", 0.0))
                    if name == pred_name and pred_score is None:
                        pred_score = score
                    if name == label and label_score is None:
                        label_score = score
                if pred_score is None:
                    pred_score = float("-inf")
                if label_score is None:
                    label_score = float("-inf")

                grouped[label].append(
                    PredEntry(
                        label=label,
                        pred_name=pred_name,
                        image_basename=image_basename,
                        coco_file_name=coco_file_name,
                        pred_score=pred_score,
                        label_score=label_score,
                    )
                )
    return grouped


def select_keep_entries(entries: List[PredEntry], target: int) -> List[PredEntry]:
    n = len(entries)
    if n < target:
        raise ValueError(f"Class {entries[0].label if entries else 'UNKNOWN'} has only {n} predictions, less than target={target}.")
    if n == target:
        return entries

    need_remove = n - target
    wrong = [e for e in entries if e.pred_name != e.label]
    correct = [e for e in entries if e.pred_name == e.label]

    # Remove misclassified samples first. If there are more than needed,
    # remove the most confidently wrong ones first.
    wrong_sorted = sorted(wrong, key=lambda x: x.pred_score, reverse=True)
    remove: Set[str] = set()
    for e in wrong_sorted[:need_remove]:
        remove.add(e.image_basename)
    need_remove -= min(len(wrong_sorted), len(remove))

    # If still need to remove, remove low-confidence correctly predicted samples.
    if need_remove > 0:
        correct_sorted = sorted(correct, key=lambda x: (x.label_score, x.pred_score))
        for e in correct_sorted[:need_remove]:
            remove.add(e.image_basename)

    keep = [e for e in entries if e.image_basename not in remove]
    if len(keep) != target:
        raise RuntimeError(f"Selection failed for class {entries[0].label}: expected {target}, got {len(keep)}")
    return keep


def update_item1(item1_dir: str, keep_map: Dict[str, List[PredEntry]]) -> None:
    for cls, kept_entries in keep_map.items():
        json_path = os.path.join(item1_dir, f"{cls}.json")
        txt_path = os.path.join(item1_dir, f"{cls}.txt")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing class json: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        images = obj.get("images", [])
        target_count = len(kept_entries)
        # Keep exact multiplicity requested by kept_entries to avoid over-keeping
        # when item1 json contains duplicated file_name rows.
        need_counts = Counter(e.coco_file_name for e in kept_entries)
        filtered_images: List[dict] = []
        for im in images:
            file_name = im.get("file_name")
            if not file_name:
                continue
            if need_counts[file_name] <= 0:
                continue
            filtered_images.append(im)
            need_counts[file_name] -= 1
            if len(filtered_images) == target_count:
                break

        if len(filtered_images) != target_count:
            missing = sum(v for v in need_counts.values() if v > 0)
            raise RuntimeError(
                f"{cls}: filtered item1 json has {len(filtered_images)} entries, "
                f"expected {target_count}, missing={missing}."
            )

        obj["num"] = len(filtered_images)
        obj["images"] = filtered_images
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")

        # Keep txt synchronized with item1 json file_name.
        with open(txt_path, "w", encoding="utf-8") as f:
            for im in filtered_images:
                f.write(f"{im['file_name']}\n")


def update_single_train_images(single_dir: str, keep_map: Dict[str, List[PredEntry]]) -> None:
    for cls, kept_entries in keep_map.items():
        cls_dir = os.path.join(single_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Missing class image dir: {cls_dir}")

        keep_names = {e.image_basename for e in kept_entries}
        for name in os.listdir(cls_dir):
            path = os.path.join(cls_dir, name)
            if not os.path.isfile(path):
                continue
            if name not in keep_names:
                os.remove(path)

        final_files = [n for n in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, n))]
        if len(final_files) != len(kept_entries):
            raise RuntimeError(
                f"{cls}: single_train_images has {len(final_files)} files, expected {len(kept_entries)}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune item1/single_train_images to target size per class using eval predictions.")
    parser.add_argument(
        "--eval-dir",
        default="/datanfs4/shenruoyan/FMUClip/finegrained/output/original_eval/flickr30k_entities/original_table7_DF3_DR1_UNI3_03100034",
    )
    parser.add_argument(
        "--item1-dir",
        default="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/train/Df/item1",
    )
    parser.add_argument(
        "--single-dir",
        default="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/train/Df/single_train_images",
    )
    parser.add_argument("--target-per-class", type=int, default=50)
    args = parser.parse_args()

    pred_groups = read_predictions(args.eval_dir)
    keep_map: Dict[str, List[PredEntry]] = {}
    for cls, entries in pred_groups.items():
        keep_map[cls] = select_keep_entries(entries, args.target_per_class)

    update_item1(args.item1_dir, keep_map)
    update_single_train_images(args.single_dir, keep_map)

    print(f"Done. Updated {len(keep_map)} classes to {args.target_per_class} samples/class.")


if __name__ == "__main__":
    main()

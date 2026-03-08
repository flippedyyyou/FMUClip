from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

DEFAULT_EVAL_DIR = Path(
    "/datanfs4/shenruoyan/FMUClip/finegrained/output/original_eval/original_banana3_100images_DF3_DR1_UNI3_03022152"
)
DEFAULT_TEST_IMAGES_DIR = Path(
    "/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances/val/test_images"
)
DEFAULT_IMAGENET_ROOT = Path("/datanfs4/shenruoyan/datasets/imagenet-1000/imagenet1k")
DEFAULT_ITEM1_DIR = Path(
    "/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances/val/Df/item1"
)
DEFAULT_DATASET_NAME = "imagenet1k"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# You can extend this mapping on your server if some labels cannot be matched.
COCO_TO_IMAGENET_HINTS: Dict[str, List[str]] = {
    "airplane": ["airliner", "warplane", "aircraft"],
    "cell_phone": ["cellular telephone", "mobile phone"],
    "couch": ["sofa"],
    "dining_table": ["dining table", "table"],
    "fire_hydrant": ["fire hydrant"],
    "hair_drier": ["hair dryer", "hair drier", "blow dryer"],
    "handbag": ["purse", "handbag"],
    "hot_dog": ["hotdog", "hot dog"],
    "motorcycle": ["motor scooter", "moped", "motorcycle"],
    "potted_plant": ["pot", "flowerpot", "plant"],
    "remote": ["remote control", "remote"],
    "sports_ball": ["ball"],
    "stop_sign": ["stop sign"],
    "teddy_bear": ["teddy", "toy bear", "bear"],
    "toothbrush": ["toothbrush"],
    "traffic_light": ["traffic light", "stoplight"],
    "tv": ["television", "tv", "monitor"],
    "wine_glass": ["wine glass", "goblet"],
}


@dataclass(frozen=True)
class MisclassifiedItem:
    image_id: str
    label: str
    pred_name: str


def normalize_name(name: str) -> str:
    s = name.strip().lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_image_id_from_path(image_path: str) -> str:
    # Example: /.../000000575205.jpg -> 000000575205
    return Path(image_path).stem


def parse_eval_jsonl_files(eval_dir: Path, eval_files: Sequence[str]) -> List[MisclassifiedItem]:
    items: List[MisclassifiedItem] = []

    for file_name in eval_files:
        p = eval_dir / file_name
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                label = str(row.get("label", "")).strip()
                pred_name = str(row.get("pred_name", "")).strip()
                image_path = str(row.get("image_path", "")).strip()
                if not label or not pred_name or not image_path:
                    continue
                if label == pred_name:
                    continue
                image_id = extract_image_id_from_path(image_path)
                items.append(
                    MisclassifiedItem(
                        image_id=image_id,
                        label=label,
                        pred_name=pred_name,
                    )
                )

    return items


def deduplicate_items(items: Iterable[MisclassifiedItem]) -> List[MisclassifiedItem]:
    # Deduplicate by (image_id, label), because an image id can appear under multiple labels.
    seen = set()
    out: List[MisclassifiedItem] = []
    for it in items:
        key = (it.image_id, it.label)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def find_target_files(test_images_dir: Path, image_id: str, label: str) -> List[Path]:
    label_dir = test_images_dir / label
    targets: List[Path] = []

    if label_dir.exists() and label_dir.is_dir():
        for p in label_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.endswith(f"_{image_id}"):
                targets.append(p)

    if targets:
        return sorted(targets)

    # Fallback: global search if class folder matching failed.
    pattern = f"*_{image_id}.*"
    for p in test_images_dir.rglob(pattern):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            targets.append(p)

    return sorted(set(targets))


def parse_imagenet_dir_aliases(dir_name: str) -> List[str]:
    # Example: "000_tench, Tinca tinca"
    # Name part after the first underscore: "tench, Tinca tinca"
    if "_" in dir_name:
        name_part = dir_name.split("_", 1)[1]
    else:
        name_part = dir_name
    raw_aliases = [x.strip() for x in name_part.split(",") if x.strip()]
    aliases = [normalize_name(x) for x in raw_aliases]
    return [a for a in aliases if a]


def build_imagenet_index(imagenet_root: Path) -> Tuple[Dict[str, List[Path]], Dict[Path, List[Path]]]:
    alias_to_dirs: Dict[str, List[Path]] = defaultdict(list)
    dir_to_images: Dict[Path, List[Path]] = {}

    for class_dir in sorted(imagenet_root.iterdir()):
        if not class_dir.is_dir():
            continue

        aliases = parse_imagenet_dir_aliases(class_dir.name)
        for a in aliases:
            alias_to_dirs[a].append(class_dir)

        imgs = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if imgs:
            dir_to_images[class_dir] = sorted(imgs)

    return alias_to_dirs, dir_to_images


def match_imagenet_dirs_for_label(label: str, alias_to_dirs: Dict[str, List[Path]]) -> List[Path]:
    norm_label = normalize_name(label)
    candidates: List[str] = [norm_label]
    candidates.extend(normalize_name(x) for x in COCO_TO_IMAGENET_HINTS.get(label, []))

    matched: List[Path] = []
    seen = set()

    # Pass 1: exact alias match.
    for c in candidates:
        for d in alias_to_dirs.get(c, []):
            if d not in seen:
                matched.append(d)
                seen.add(d)

    if matched:
        return matched

    # Pass 2: fuzzy contains match.
    all_aliases = list(alias_to_dirs.keys())
    for c in candidates:
        for a in all_aliases:
            if not c:
                continue
            if c in a or a in c:
                for d in alias_to_dirs[a]:
                    if d not in seen:
                        matched.append(d)
                        seen.add(d)

    return matched


def choose_source_images(
    label: str,
    needed: int,
    alias_to_dirs: Dict[str, List[Path]],
    dir_to_images: Dict[Path, List[Path]],
    rng: random.Random,
) -> List[Path]:
    class_dirs = match_imagenet_dirs_for_label(label, alias_to_dirs)
    pool: List[Path] = []
    for d in class_dirs:
        pool.extend(dir_to_images.get(d, []))

    if not pool:
        return []

    if needed <= len(pool):
        rng.shuffle(pool)
        return pool[:needed]

    # If needed exceeds available unique images, sample with replacement.
    return [rng.choice(pool) for _ in range(needed)]


def replace_images_for_misclassified_items(
    test_images_dir: Path,
    items: Sequence[MisclassifiedItem],
    alias_to_dirs: Dict[str, List[Path]],
    dir_to_images: Dict[Path, List[Path]],
    rng: random.Random,
    dry_run: bool = False,
) -> Dict[str, object]:
    by_label: Dict[str, List[MisclassifiedItem]] = defaultdict(list)
    for it in items:
        by_label[it.label].append(it)

    replaced = 0
    missing_target = 0
    missing_source = 0
    unresolved_labels: Dict[str, int] = defaultdict(int)
    details: List[Dict[str, str]] = []

    for label, label_items in sorted(by_label.items()):
        # Build all target paths for this label first.
        target_records: List[Tuple[MisclassifiedItem, Path]] = []
        for it in label_items:
            targets = find_target_files(test_images_dir, it.image_id, it.label)
            if not targets:
                missing_target += 1
                continue
            # Usually one match. If multiple, replace all matches for consistency.
            for t in targets:
                target_records.append((it, t))

        if not target_records:
            continue

        sources = choose_source_images(
            label=label,
            needed=len(target_records),
            alias_to_dirs=alias_to_dirs,
            dir_to_images=dir_to_images,
            rng=rng,
        )
        if not sources:
            unresolved_labels[label] += len(target_records)
            missing_source += len(target_records)
            continue

        for (item, target_path), src_path in zip(target_records, sources):
            if not dry_run:
                shutil.copy2(src_path, target_path)
            replaced += 1
            print(f"Replaced: {target_path} <= {src_path} (label={label}, pred={item.pred_name})")
            details.append(
                {
                    "label": label,
                    "image_id": item.image_id,
                    "pred_name": item.pred_name,
                    "target": str(target_path),
                    "source": str(src_path),
                }
            )

    replaced_pairs = sorted({(d["label"], d["image_id"]) for d in details})

    return {
        "total_misclassified_items": len(items),
        "replaced": replaced,
        "missing_target": missing_target,
        "missing_source": missing_source,
        "unresolved_labels": dict(sorted(unresolved_labels.items())),
        "replaced_label_image_ids": [[label, image_id] for label, image_id in replaced_pairs],
        "details": details,
    }


def update_item1_dataset_fields(
    item1_dir: Path,
    dataset_name: str,
    replaced_label_image_ids: Sequence[Sequence[str]],
    dry_run: bool = False,
) -> Dict[str, int]:
    updated_files = 0
    skipped = 0
    touched_items = 0

    replaced_by_label: Dict[str, Set[str]] = defaultdict(set)
    for pair in replaced_label_image_ids:
        if len(pair) != 2:
            continue
        label, image_id = str(pair[0]), str(pair[1])
        replaced_by_label[label].add(image_id)

    for json_file in sorted(item1_dir.glob("*.json")):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue

        if not isinstance(data, dict):
            skipped += 1
            continue

        images = data.get("images")
        if not isinstance(images, list):
            skipped += 1
            continue

        # Match by file name first (e.g., "airplane.json" -> label "airplane").
        candidate_ids = replaced_by_label.get(json_file.stem, set())
        if not candidate_ids:
            # Fallback to concept field if present.
            concept = str(data.get("concept", "")).strip()
            candidate_ids = replaced_by_label.get(concept, set())
        if not candidate_ids:
            continue

        changed = False
        for idx, item in enumerate(images):
            if not isinstance(item, dict):
                continue
            image_id = str(item.get("img_id", "")).strip()
            if image_id not in candidate_ids:
                continue
            images[idx] = {"dataset": dataset_name, **{k: v for k, v in item.items() if k != "dataset"}}
            touched_items += 1
            changed = True

        if changed and not dry_run:
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")

        if changed:
            updated_files += 1

    return {"updated_files": updated_files, "touched_items": touched_items, "skipped": skipped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "1) Replace misclassified test images with semantically matched ImageNet images; "
            "2) Update dataset fields in item1 JSON files."
        )
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--eval-files",
        nargs="+",
        default=["retain_test_topk.jsonl", "forget_test_topk.jsonl"],
        help="JSONL files under --eval-dir to scan for label!=pred_name.",
    )
    parser.add_argument("--test-images-dir", type=Path, default=DEFAULT_TEST_IMAGES_DIR)
    parser.add_argument("--imagenet-root", type=Path, default=DEFAULT_IMAGENET_ROOT)
    parser.add_argument("--item1-dir", type=Path, default=DEFAULT_ITEM1_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional output path for JSON report.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.eval_dir.exists():
        raise FileNotFoundError(f"eval dir not found: {args.eval_dir}")
    if not args.test_images_dir.exists():
        raise FileNotFoundError(f"test_images dir not found: {args.test_images_dir}")
    if not args.imagenet_root.exists():
        raise FileNotFoundError(f"imagenet root not found: {args.imagenet_root}")
    if not args.item1_dir.exists():
        raise FileNotFoundError(f"item1 dir not found: {args.item1_dir}")

    rng = random.Random(args.seed)

    raw_items = parse_eval_jsonl_files(args.eval_dir, args.eval_files)
    items = deduplicate_items(raw_items)

    alias_to_dirs, dir_to_images = build_imagenet_index(args.imagenet_root)

    replace_report = replace_images_for_misclassified_items(
        test_images_dir=args.test_images_dir,
        items=items,
        alias_to_dirs=alias_to_dirs,
        dir_to_images=dir_to_images,
        rng=rng,
        dry_run=args.dry_run,
    )

    json_report = update_item1_dataset_fields(
        item1_dir=args.item1_dir,
        dataset_name=args.dataset_name,
        replaced_label_image_ids=replace_report.get("replaced_label_image_ids", []),
        dry_run=args.dry_run,
    )

    report = {
        "dry_run": args.dry_run,
        "eval_dir": str(args.eval_dir),
        "eval_files": list(args.eval_files),
        "test_images_dir": str(args.test_images_dir),
        "imagenet_root": str(args.imagenet_root),
        "item1_dir": str(args.item1_dir),
        "dataset_name": args.dataset_name,
        "replace_report": replace_report,
        "json_report": json_report,
    }

    if args.report_path is not None:
        if not args.dry_run:
            args.report_path.parent.mkdir(parents=True, exist_ok=True)
            args.report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            print(f"[dry-run] report not written: {args.report_path}")


if __name__ == "__main__":
    main()

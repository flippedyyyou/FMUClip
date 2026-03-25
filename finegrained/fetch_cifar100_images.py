from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from PIL import Image
from torchvision import datasets, transforms

DEFAULT_EVAL_DIR = Path(
    "/datanfs4/shenruoyan/FMUClip/finegrained/ckpt/original/flickr30k_entities/filter_cifar_imagenet_girl_5/eval_results"
)
DEFAULT_TEST_IMAGES_DIR = Path(
    "/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/val/test_images"
)
DEFAULT_CIFAR100_ROOT = Path("/datanfs4/shenruoyan/datasets/cifar-100-python/cifar-100-python")
DEFAULT_ITEM1_DIR = Path(
    "/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/val/Df/item1"
)
DEFAULT_DATASET_NAME = "cifar100"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# You can extend these mappings on your server if some labels cannot be matched.
COCO_TO_CIFAR100_HINTS: Dict[str, List[str]] = {
    "airplane": ["rocket"],
    "apple": ["apple"],
    "backpack": [],
    "banana": [],
    "baseball_bat": [],
    "baseball_glove": [],
    "bear": ["bear"],
    "bed": ["bed"],
    "bench": [],
    "bicycle": ["bicycle"],
    "bird": [],
    "boat": [],
    "book": [],
    "bottle": ["bottle"],
    "bowl": ["bowl"],
    "broccoli": [],
    "bus": ["bus"],
    "cake": [],
    "car": [],
    "carrot": [],
    "cat": [],
    "cell_phone": ["telephone"],
    "chair": ["chair"],
    "clock": ["clock"],
    "couch": ["couch"],
    "cow": ["cattle"],
    "cup": ["cup"],
    "dining_table": ["table"],
    "dog": [],
    "donut": [],
    "elephant": ["elephant"],
    "fire_hydrant": [],
    "fork": [],
    "frisbee": [],
    "giraffe": [],
    "hair_drier": [],
    "handbag": [],
    "horse": [],
    "hot_dog": [],
    "keyboard": ["keyboard"],
    "kite": [],
    "knife": [],
    "laptop": [],
    "microwave": [],
    "motorcycle": ["motorcycle"],
    "mouse": ["mouse"],
    "orange": ["orange"],
    "oven": [],
    "parking_meter": [],
    "person": ["man", "woman"],
    "pizza": [],
    "potted_plant": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "refrigerator": [],
    "remote": [],
    "sandwich": [],
    "scissors": [],
    "sheep": [],
    "sink": [],
    "skateboard": [],
    "skis": [],
    "snowboard": [],
    "spoon": [],
    "sports_ball": [],
    "stop_sign": [],
    "suitcase": [],
    "surfboard": [],
    "teddy_bear": ["bear"],
    "tennis_racket": [],
    "tie": [],
    "toaster": [],
    "toilet": [],
    "toothbrush": [],
    "traffic_light": [],
    "train": ["train"],
    "truck": ["pickup_truck"],
    "tv": ["television"],
    "umbrella": [],
    "vase": [],
    "wine_glass": [],
    "zebra": [],
}

FLICKR30K_TO_CIFAR100_HINTS: Dict[str, List[str]] = {
    "man": ["man"],
    "woman": ["woman"],
    "girl": ["girl"],
    "boy": ["boy"],
    "dog": [],
    "table": ["table"],
    "orange": ["orange"],
    "person": ["man", "woman", "boy", "girl", "baby"],
    "road": ["road"],
    "car": ["streetcar"],
    "chair": ["chair"],
    "bench": [],
    "bicycle": ["bicycle"],
    "mountain": ["mountain"],
    "boat": [],
    "baby": ["baby"],
    "backpack": [],
    "house": ["house"],
    "horse": [],
    "train": ["train"],
    "tank": ["tank"],
    "skateboard": [],
    "book": [],
    "truck": ["pickup_truck"],
    "umbrella": [],
    "forest": ["forest"],
    "motorcycle": ["motorcycle"],
    "bus": ["bus"],
    "bridge": ["bridge"],
    "cup": ["cup"],
}


@dataclass(frozen=True)
class MisclassifiedItem:
    image_id: str
    label: str
    pred_name: str


@dataclass(frozen=True)
class CifarImageRef:
    split: str
    index: int
    class_name: str


def normalize_name(name: str) -> str:
    s = name.strip().lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_image_id_from_path(image_path: str) -> str:
    stem = Path(image_path).stem
    m = re.search(r"(\d+)$", stem)
    if m:
        return m.group(1)
    return stem


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
                items.append(
                    MisclassifiedItem(
                        image_id=extract_image_id_from_path(image_path),
                        label=label,
                        pred_name=pred_name,
                    )
                )

    return items


def deduplicate_items(items: Iterable[MisclassifiedItem]) -> List[MisclassifiedItem]:
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

    pattern = f"*_{image_id}.*"
    for p in test_images_dir.rglob(pattern):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            targets.append(p)

    return sorted(set(targets))


def infer_label_space(test_images_dir: Path) -> str:
    path_str = str(test_images_dir)
    if "flickr30k_entities" in path_str:
        return "flickr30k"
    if "coco2017_instances" in path_str:
        return "coco"
    return "coco"


def get_cifar100_hints(test_images_dir: Path) -> Dict[str, List[str]]:
    label_space = infer_label_space(test_images_dir)
    if label_space == "flickr30k":
        return FLICKR30K_TO_CIFAR100_HINTS
    return COCO_TO_CIFAR100_HINTS


def resolve_cifar100_root(cifar100_root: Path) -> Path:
    root = cifar100_root.expanduser()

    # Support both:
    # 1) parent directory that contains "cifar-100-python/"
    # 2) extracted dataset directory itself ".../cifar-100-python"
    if (root / "train").exists() and (root / "test").exists() and (root / "meta").exists():
        return root.parent

    extracted_dir = root / "cifar-100-python"
    if (extracted_dir / "train").exists() and (extracted_dir / "test").exists() and (extracted_dir / "meta").exists():
        return root

    return root


def build_cifar100_index(
    cifar100_root: Path,
    include_train: bool,
    include_test: bool,
) -> Dict[str, List[CifarImageRef]]:
    class_to_refs: Dict[str, List[CifarImageRef]] = defaultdict(list)
    dataset_root = resolve_cifar100_root(cifar100_root)

    if include_train:
        train_ds = datasets.CIFAR100(root=str(dataset_root), train=True, download=False)
        for idx, class_idx in enumerate(train_ds.targets):
            class_name = normalize_name(train_ds.classes[class_idx])
            class_to_refs[class_name].append(CifarImageRef(split="train", index=idx, class_name=class_name))

    if include_test:
        test_ds = datasets.CIFAR100(root=str(dataset_root), train=False, download=False)
        for idx, class_idx in enumerate(test_ds.targets):
            class_name = normalize_name(test_ds.classes[class_idx])
            class_to_refs[class_name].append(CifarImageRef(split="test", index=idx, class_name=class_name))

    return {k: v[:] for k, v in sorted(class_to_refs.items())}


def match_cifar100_classes_for_label(
    label: str,
    class_to_refs: Dict[str, List[CifarImageRef]],
    hints: Dict[str, List[str]],
) -> List[str]:
    norm_label = normalize_name(label)
    candidates: List[str] = [norm_label]
    candidates.extend(normalize_name(x) for x in hints.get(label, []))

    matched: List[str] = []
    seen = set()

    for c in candidates:
        if c in class_to_refs and c not in seen:
            matched.append(c)
            seen.add(c)

    if matched:
        return matched

    all_classes = list(class_to_refs.keys())
    for c in candidates:
        for name in all_classes:
            if not c:
                continue
            if c in name or name in c:
                if name not in seen:
                    matched.append(name)
                    seen.add(name)

    return matched


def choose_source_refs(
    label: str,
    needed: int,
    test_images_dir: Path,
    class_to_refs: Dict[str, List[CifarImageRef]],
    rng: random.Random,
) -> List[CifarImageRef]:
    class_names = match_cifar100_classes_for_label(
        label=label,
        class_to_refs=class_to_refs,
        hints=get_cifar100_hints(test_images_dir),
    )
    pool: List[CifarImageRef] = []
    for class_name in class_names:
        pool.extend(class_to_refs.get(class_name, []))

    if not pool:
        return []

    if needed <= len(pool):
        rng.shuffle(pool)
        return pool[:needed]

    return [rng.choice(pool) for _ in range(needed)]


def load_cifar100_datasets(
    cifar100_root: Path,
    include_train: bool,
    include_test: bool,
) -> Dict[str, datasets.CIFAR100]:
    loaded: Dict[str, datasets.CIFAR100] = {}
    dataset_root = resolve_cifar100_root(cifar100_root)
    if include_train:
        loaded["train"] = datasets.CIFAR100(root=str(dataset_root), train=True, download=False)
    if include_test:
        loaded["test"] = datasets.CIFAR100(root=str(dataset_root), train=False, download=False)
    return loaded


def render_cifar_ref_to_target(
    ref: CifarImageRef,
    target_path: Path,
    loaded_datasets: Dict[str, datasets.CIFAR100],
) -> None:
    ds = loaded_datasets[ref.split]
    image, _ = ds[ref.index]
    if not isinstance(image, Image.Image):
        image = transforms.ToPILImage()(image)
    image = image.convert("RGB")
    image.save(target_path)


def replace_images_for_misclassified_items(
    test_images_dir: Path,
    items: Sequence[MisclassifiedItem],
    class_to_refs: Dict[str, List[CifarImageRef]],
    loaded_datasets: Dict[str, datasets.CIFAR100],
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
        target_records: List[Tuple[MisclassifiedItem, Path]] = []
        for it in label_items:
            targets = find_target_files(test_images_dir, it.image_id, it.label)
            if not targets:
                missing_target += 1
                continue
            for target in targets:
                target_records.append((it, target))

        if not target_records:
            continue

        source_refs = choose_source_refs(
            label=label,
            needed=len(target_records),
            test_images_dir=test_images_dir,
            class_to_refs=class_to_refs,
            rng=rng,
        )
        if not source_refs:
            unresolved_labels[label] += len(target_records)
            missing_source += len(target_records)
            continue

        for (item, target_path), src_ref in zip(target_records, source_refs):
            if not dry_run:
                render_cifar_ref_to_target(src_ref, target_path, loaded_datasets)
            replaced += 1
            source_desc = f"cifar100:{src_ref.split}:{src_ref.class_name}:{src_ref.index}"
            print(f"Replaced: {target_path} <= {source_desc} (label={label}, pred={item.pred_name})")
            details.append(
                {
                    "label": label,
                    "image_id": item.image_id,
                    "pred_name": item.pred_name,
                    "target": str(target_path),
                    "source": source_desc,
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

        candidate_ids = replaced_by_label.get(json_file.stem, set())
        if not candidate_ids:
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
            "1) Replace misclassified test images with semantically matched CIFAR-100 images; "
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
    parser.add_argument("--cifar100-root", type=Path, default=DEFAULT_CIFAR100_ROOT)
    parser.add_argument("--item1-dir", type=Path, default=DEFAULT_ITEM1_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cifar-split",
        choices=["train", "test", "both"],
        default="train",
        help="Which CIFAR-100 split to sample replacement images from.",
    )
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
    if not args.cifar100_root.exists():
        raise FileNotFoundError(f"cifar100 root not found: {args.cifar100_root}")
    if not args.item1_dir.exists():
        raise FileNotFoundError(f"item1 dir not found: {args.item1_dir}")

    include_train = args.cifar_split in {"train", "both"}
    include_test = args.cifar_split in {"test", "both"}
    rng = random.Random(args.seed)

    raw_items = parse_eval_jsonl_files(args.eval_dir, args.eval_files)
    items = deduplicate_items(raw_items)

    class_to_refs = build_cifar100_index(
        cifar100_root=args.cifar100_root,
        include_train=include_train,
        include_test=include_test,
    )
    loaded_datasets = load_cifar100_datasets(
        cifar100_root=args.cifar100_root,
        include_train=include_train,
        include_test=include_test,
    )

    replace_report = replace_images_for_misclassified_items(
        test_images_dir=args.test_images_dir,
        items=items,
        class_to_refs=class_to_refs,
        loaded_datasets=loaded_datasets,
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
        "cifar100_root": str(args.cifar100_root),
        "cifar_split": args.cifar_split,
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

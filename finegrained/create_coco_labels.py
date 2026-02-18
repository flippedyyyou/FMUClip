import argparse
from pathlib import Path

from pycocotools.coco import COCO

DEFAULT_ANN_PATH = "/datanfs4/shenruoyan/datasets/coco2017/annotations/instances_train2017.json"
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("coco.py")


def normalize_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def load_coco_label_names(annotation_path: Path) -> list[str]:
    coco = COCO(str(annotation_path))
    categories = coco.loadCats(coco.getCatIds())
    categories = sorted(categories, key=lambda x: x["id"])
    return [normalize_name(cat["name"]) for cat in categories]


def format_label_dict(label_names: list[str]) -> str:
    lines = ["LABEL_NAMES = {"]

    row_items = []
    for idx, name in enumerate(label_names):
        row_items.append(f'{idx}: "{name}"')
        if len(row_items) == 5 or idx == len(label_names) - 1:
            suffix = "," if idx != len(label_names) - 1 else ""
            lines.append("    " + ", ".join(row_items) + suffix)
            row_items = []

    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate coco.py LABEL_NAMES from COCO instances_train2017.json."
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=Path(DEFAULT_ANN_PATH),
        help="Path to instances_train2017.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output python file path, e.g. multi-label/coco.py",
    )
    args = parser.parse_args()

    if not args.annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {args.annotation_path}")

    label_names = load_coco_label_names(args.annotation_path)
    content = format_label_dict(label_names)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(content, encoding="utf-8")
    print(f"Saved {len(label_names)} labels to {args.output_path}")


if __name__ == "__main__":
    main()

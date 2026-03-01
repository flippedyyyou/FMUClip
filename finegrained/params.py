import argparse
import os
from typing import List


def _parse_forget_classes(raw: str) -> List[str]:
    items = [x.strip() for x in raw.split(",")]
    items = [x for x in items if x]
    if not items:
        raise argparse.ArgumentTypeError("`--forget_classes` cannot be empty.")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyperparameters for COCO multi-label data loading.")

    parser.add_argument(
        "--df_root",
        type=str,
        default="/home/shenruoyan/FMUClip/data/classification/coco2017_instances",
        help="Root folder containing train/val Df lists (txt/json).",
    )
    parser.add_argument(
        "--coco_root",
        type=str,
        default="/datanfs4/shenruoyan/datasets/coco2017",
        help="COCO root folder containing annotations/, train2017/, val2017/.",
    )

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--train_item_folder", type=str, default="item3")
    parser.add_argument("--retain_item_folder", type=str, default="item1")
    parser.add_argument("--test_item_folder", type=str, default="item1")
    parser.add_argument(
        "--test_item_format",
        type=str,
        default="txt",
        choices=["txt", "json"],
        help="Test list format under Df/item*: txt image list or json with bbox.",
    )
    parser.add_argument(
        "--test_max_per_class",
        type=int,
        default=100,
        help="Maximum number of test samples per concept.",
    )

    parser.add_argument(
        "--forget_classes",
        type=_parse_forget_classes,
        required=True,
        help="Comma-separated class names, e.g. apple,dog,horse",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--no_shuffle_train", action="store_false", dest="shuffle_train")
    parser.set_defaults(shuffle_train=True)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--return_meta", action="store_true")
    parser.add_argument(
        "--method",
        choices=["cliperase"],
        default="cliperase",
        help="Training method when not using --original_eval.",
    )
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lambda_df", type=float, default=3.0)
    parser.add_argument("--lambda_dr", type=float, default=1.0)
    parser.add_argument("--lambda_uni", type=float, default=3.0)
    parser.add_argument("--retain_topk", type=int, default=5)
    parser.add_argument("--original_eval", action="store_true", help="Only run original CLIP evaluation on test sets.")
    parser.add_argument(
        "--clip_model_path",
        type=str,
        default="",
        help="Path to CLIP checkpoint .pt file used for evaluation.",
    )
    parser.add_argument(
        "--clip_arch",
        type=str,
        default="ViT-L-14-336",
        help="CLIP architecture name (used by open_clip fallback).",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="openai",
        help="Pretrained tag for open_clip (e.g. openai, laion2b_s34b_b79k).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="finegrained/output")

    return parser


def parse_args():
    args = build_parser().parse_args()

    args.train_annotation_file = os.path.join(
        args.coco_root, "annotations", f"instances_{args.train_split}2017.json"
    )
    args.val_annotation_file = os.path.join(
        args.coco_root, "annotations", f"instances_{args.val_split}2017.json"
    )
    args.train_image_root = os.path.join(args.coco_root, f"{args.train_split}2017")
    args.val_image_root = os.path.join(args.coco_root, f"{args.val_split}2017")
    if args.original_eval and not args.clip_model_path:
        raise ValueError("`--clip_model_path` is required when `--original_eval` is enabled.")
    return args

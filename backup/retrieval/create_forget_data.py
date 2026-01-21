#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import sys
from typing import List, Tuple

import pandas as pd


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to find the image and comment columns with some common fallbacks.
    Returns (image_col, comment_col).
    """
    cols = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in cols}

    # image column candidates
    img_candidates = ["image_name", "image", "image_id", "image_path", "filename", "file", "img"]
    # comment/text column candidates
    txt_candidates = ["comment", "caption", "text", "sentence", "description"]

    image_col = None
    for cand in img_candidates:
        if cand in lower:
            image_col = lower[cand]
            break

    comment_col = None
    for cand in txt_candidates:
        if cand in lower:
            comment_col = lower[cand]
            break

    if image_col is None or comment_col is None:
        raise ValueError(
            f"Missing required columns. Tried image in {img_candidates} and text in {txt_candidates}. "
            f"Found columns: {list(df.columns)}"
        )
    return image_col, comment_col


def build_pattern(keywords: List[str], whole_word: bool, case_sensitive: bool) -> re.Pattern:
    """
    Build a compiled regex pattern for OR of keywords.
    - whole_word=True ensures letters are not attached to left/right (e.g., avoid 'grapple' for 'apple').
      We implement via negative lookaround for ASCII letters: (?<![A-Za-z])kw(?![A-Za-z])
      This is safer than \b for cases like "apple's" or hyphenated tokens.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    parts = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        escaped = re.escape(kw)
        if whole_word:
            parts.append(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])")
        else:
            parts.append(escaped)
    if not parts:
        raise ValueError("No valid keywords provided.")
    pattern = "|".join(parts)
    return re.compile(pattern, flags)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_or_link(src: str, dst_dir: str, mode: str):
    """
    mode: 'copy' (default) or 'symlink'
    """
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not os.path.exists(src):
        print(f"[WARN] Source image not found: {src}")
        return
    try:
        if os.path.exists(dst):
            return
        if mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy(src, dst)
    except Exception as e:
        print(f"[WARN] Failed to place {src} -> {dst}: {e}")


def split_dataset_by_keyword(
    csv_path: str,
    image_folder: str,
    out_forget: str,
    out_retain: str,
    keywords: List[str],
    whole_word: bool = True,
    case_sensitive: bool = False,
    link_mode: str = "copy",
    dry_run: bool = False,
    sep: str = "|",
    data_type: str | None = None,
    seed: int | None = None,
    df_size: int | None = None,
    out_df_root: str | None = None,
    id_mode: str = "filename",
):
    """
    Split dataset into Forget and Retain sets based on presence of keyword(s) in the comment column.
    """
    # Read CSV
    try:
        df = pd.read_csv(csv_path, sep=sep, engine="python")
    except Exception as e:
        print(f"[ERROR] Reading CSV failed: {e}")
        sys.exit(1)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Detect columns
    try:
        image_col, comment_col = detect_columns(df)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Build regex pattern
    pattern = build_pattern(keywords, whole_word=whole_word, case_sensitive=case_sensitive)

    # Mark rows that match any keyword in comment
    mask = df[comment_col].astype(str).str.contains(pattern, na=False)

    # Identify images containing target concepts
    matched_series = df.loc[mask, image_col].dropna().astype(str)
    if id_mode == "filename":
        keyword_images = matched_series.apply(os.path.basename).unique()
        all_images = df[image_col].dropna().astype(str).apply(os.path.basename)
    elif id_mode == "path":
        keyword_images = matched_series.unique()
        all_images = df[image_col].dropna().astype(str)
    else:
        raise ValueError("id_mode must be 'filename' or 'path'")

    # Optional: limit Df size deterministically (sort then take head)
    keyword_images = sorted(keyword_images)
    if df_size is not None:
        keyword_images = keyword_images[: int(df_size)]
    keyword_set = set(keyword_images)

    # Normalize the CSV’s image column to the same ID space for filtering
    img_ids = all_images.tolist()
    df_norm = df.copy()
    df_norm[image_col] = img_ids

    # Split
    forget_set = df_norm[df_norm[image_col].astype(str).isin(keyword_set)].copy()
    retain_set = df_norm[~df_norm[image_col].astype(str).isin(keyword_set)].copy()

    # Report
    num_forget_imgs = forget_set[image_col].astype(str).nunique()
    num_retain_imgs = retain_set[image_col].astype(str).nunique()
    print(f"✅ Matched keywords: {keywords}")
    print(f"✅ Forget Set images: {num_forget_imgs} | rows: {len(forget_set)}")
    print(f"✅ Retain Set images: {num_retain_imgs} | rows: {len(retain_set)}")

    # Create output dirs
    ensure_dir(out_forget)
    ensure_dir(out_retain)

    # Save captions
    forget_csv = os.path.join(out_forget, "forget_set_captions.csv")
    retain_csv = os.path.join(out_retain, "retain_set_captions.csv")
    forget_set.to_csv(forgot_csv := forget_csv, index=False)
    retain_set.to_csv(retain_csv, index=False)
    print(f"📝 Saved: {forgot_csv}")
    print(f"📝 Saved: {retain_csv}")

    # Write Df/{data_type}/image-{seed}.txt for your unlearning pipeline
    if data_type and seed is not None:
        root = out_df_root or "Df"
        dst_dir = os.path.join(root, data_type)
        os.makedirs(dst_dir, exist_ok=True)
        df_txt = os.path.join(dst_dir, f"image-{seed}.txt")
        with open(df_txt, "w", encoding="utf-8") as f:
            for name in keyword_images:
                f.write(str(name).strip() + "\n")
        print(f"🧾 Wrote Df list: {df_txt} (n={len(keyword_images)})")

        # (Optional) also write Retain list for对照/调试
        retain_names = sorted(set(all_images.unique()) - keyword_set)
        retain_txt = os.path.join(dst_dir, f"image-retain-{seed}.txt")
        with open(retain_txt, "w", encoding="utf-8") as f:
            for name in retain_names:
                f.write(str(name).strip() + "\n")
        print(f"🧾 Wrote Dr list (optional): {retain_txt} (n={len(retain_names)})")

    # Copy/link images
    if dry_run:
        print("🧪 Dry-run enabled: skipping image copy/link.")
        return

    print(f"📁 Placing Forget Set images into: {out_forget} (mode={link_mode})")
    for img in forget_set[image_col].astype(str).dropna().unique():
        src = os.path.join(image_folder, img)
        copy_or_link(src, out_forget, mode=link_mode)

    print(f"📁 Placing Retain Set images into: {out_retain} (mode={link_mode})")
    for img in retain_set[image_col].astype(str).dropna().unique():
        src = os.path.join(image_folder, img)
        copy_or_link(src, out_retain, mode=link_mode)

    print("🎉 Dataset split complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into Forget/Retain sets based on (whole-word) keyword match in captions."
    )
    parser.add_argument("--csv", required=True, help="Path to the CSV file (default sep='|').")
    parser.add_argument("--image-folder", required=True, help="Directory containing image files.")
    parser.add_argument(
        "--keywords",
        required=True,
        help="Comma-separated keywords, e.g., 'apple,banana'. Defaults to whole-word match.",
    )
    parser.add_argument(
        "--forget-dir",
        default=None,
        help="Output directory for Forget Set. If omitted, will be '<image-folder>/forget_set_<first_keyword>'.",
    )
    parser.add_argument(
        "--retain-dir",
        default=None,
        help="Output directory for Retain Set. If omitted, will be '<image-folder>/retain_set_<first_keyword>'.",
    )
    parser.add_argument(
        "--no-whole-word",
        action="store_true",
        help="Disable whole-word matching (use substring match). Not recommended.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Enable case-sensitive matching. Default is case-insensitive.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Place images by copying (default) or creating symlinks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not place images; only create CSVs and print stats.",
    )
    parser.add_argument("--sep", default="|", help="CSV delimiter. Default: |")
    parser.add_argument("--data-type", default=None, help="e.g., flickr30k/coco/nlvr/ve")
    parser.add_argument("--seed", type=int, default=None, help="Seed to name Df file, writes image-<seed>.txt")
    parser.add_argument("--df-size", type=int, default=None, help="Cap the number of Df images (take first N after sort)，即按文件名排序删掉符合keywords匹配的前多少张图。")
    parser.add_argument("--out-df-root", default=None, help="Root dir for Df tree (default: ./Df)")
    parser.add_argument("--id-mode", choices=["filename", "path"], default="filename",
                        help="How to write IDs into txt. filename表示把 ID 写成纯文件名（如 12345.jpg）")
    return parser.parse_args()


def main():
    args = parse_args()
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if not keywords:
        print("[ERROR] No keywords provided.")
        sys.exit(1)

    first_kw = re.sub(r"\W+", "_", keywords[0])  # for folder suffix
    forget_dir = args.forget_dir or os.path.join(args.image_folder, f"forget_set_{first_kw}")
    retain_dir = args.retain_dir or os.path.join(args.image_folder, f"retain_set_{first_kw}")

    split_dataset_by_keyword(
        csv_path=args.csv,
        image_folder=args.image_folder,
        out_forget=forget_dir,
        out_retain=retain_dir,
        keywords=keywords,
        whole_word=not args.no_whole_word,
        case_sensitive=args.case_sensitive,
        link_mode=args.link_mode,
        dry_run=args.dry_run,
        sep=args.sep,
        data_type=args.data_type,
        seed=args.seed,
        df_size=args.df_size,
        out_df_root=args.out_df_root,
        id_mode=args.id_mode,
    )


if __name__ == "__main__":
    main()


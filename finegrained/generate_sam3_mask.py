import argparse
import os

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


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
    parser.add_argument("--image-list", required=True, help="txt file with image paths (one per line)")
    parser.add_argument("--image-root", default="", help="optional root to prepend to relative paths")
    parser.add_argument("--output-dir", required=True, help="directory to save binary mask pngs")
    parser.add_argument("--prompt", required=True, help="text prompt for SAM3 segmentation")
    parser.add_argument("--bpe-path", required=True, help="path to SAM3 BPE vocab")
    parser.add_argument("--checkpoint", required=True, help="path to SAM3 checkpoint")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = build_sam3_image_model(bpe_path=args.bpe_path, checkpoint_path=args.checkpoint)
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)

    image_paths = _load_image_paths(args.image_list, args.image_root or None)
    if not image_paths:
        raise ValueError("No image paths found.")

    valid_image_paths = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=args.prompt)

        masks = state["masks"]
        scores = state["scores"]
        if masks.numel() == 0:
            print(f"Warning: No masks for {image_path}. Consider lowering confidence_threshold.")
            continue  # Skip this image if no masks are found

        best_idx = torch.argmax(scores).item()
        confidence = scores[best_idx].item()

        if confidence < args.confidence_threshold:
            print(f"Warning: Confidence for {image_path} is below threshold. Removing from list.")
            continue  # Skip this image and do not save the mask

        best_mask = masks[best_idx, 0].cpu().numpy().astype(np.uint8) * 255
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(args.output_dir, f"{base}.png")
        Image.fromarray(best_mask, mode="L").save(out_path)
        print(f"Saved {out_path}")

        valid_image_paths.append(image_path)  # Keep valid image paths

    # Save the remaining valid image paths back to the text file
    _save_image_paths(args.image_list, valid_image_paths)


if __name__ == "__main__":
    main()

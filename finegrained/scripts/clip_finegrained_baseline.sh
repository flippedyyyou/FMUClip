#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

FORGET_CLASSES=banana
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
CLIP_MODEL_PATH="/datanfs4/shenruoyan/checkpoints/clip/ViT-L-14-336px.pt"
CLIP_ARCH="ViT-L-14-336"
OUTPUT_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/output/original_eval"
BATCH_SIZE=16
NUM_WORKERS=4

python /datanfs4/shenruoyan/FMUClip/finegrained/clip_finegrained_baseline.py \
  --original_eval \
  --forget_classes "${FORGET_CLASSES}" \
  --coco_root "${COCO_ROOT}" \
  --df_root "${DF_ROOT}" \
  --train_item_folder item3 \
  --test_item_folder item1 \
  --clip_model_path "${CLIP_MODEL_PATH}" \
  --clip_arch "${CLIP_ARCH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

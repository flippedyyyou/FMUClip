#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

DEVICE="cuda"
FORGET_CLASSES=banana
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
CLIP_ARCH="ViT-L-14-336"
CLIP_PRETRAINED="openai"
CLIP_MODEL_PATH="/datanfs4/shenruoyan/checkpoints/clip/ViT-L-14-336px.pt"
BATCH_SIZE=16
NUM_WORKERS=4
MAX_EPOCH=20
LR=1e-6
WEIGHT_DECAY=5e-4
LAMBDA_DF=3
LAMBDA_DR=1
LAMBDA_UNI=3
METHOD="cliperase"
OUTPUT_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/output/baselines/${METHOD}_${FORGET_CLASSES}4_DF${LAMBDA_DF}_DR${LAMBDA_DR}_UNI${LAMBDA_UNI}_$(date +%m%d%H%M)"

python /datanfs4/shenruoyan/FMUClip/finegrained/clip_finegrained_baseline.py \
  --method "${METHOD}" \
  --forget_classes "${FORGET_CLASSES}" \
  --coco_root "${COCO_ROOT}" \
  --df_root "${DF_ROOT}" \
  --train_item_folder item4 \
  --retain_item_folder item1 \
  --test_item_folder item1 \
  --clip_arch "${CLIP_ARCH}" \
  --clip_pretrained "${CLIP_PRETRAINED}" \
  --clip_model_path "${CLIP_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_epoch "${MAX_EPOCH}" \
  --lr "${LR}" \
  --log_interval 10 \
  --weight_decay "${WEIGHT_DECAY}" \
  --lambda_df "${LAMBDA_DF}" \
  --lambda_dr "${LAMBDA_DR}" \
  --lambda_uni "${LAMBDA_UNI}" \
  --device "${DEVICE}"

#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3

DEVICE="cuda"
FORGET_CLASSES=bird
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
SAM3_MASK_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/mask"
CLIP_ARCH="ViT-L-14-336"
CLIP_PRETRAINED="openai"
CLIP_MODEL_PATH="/datanfs4/shenruoyan/checkpoints/clip/ViT-L-14-336px.pt"
BATCH_SIZE=4
NUM_WORKERS=4
MAX_EPOCH=50
LR=5e-6
WEIGHT_DECAY=5e-4
LAMBDA_RTF=5
LAMBDA_KEEP=0
LAMBDA_CE=1
SAMPLE_K=5
RETAIN_TOPK=5
METHOD="ours"
LAYER=1

OUTPUT_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/output/unlearn/${METHOD}_${FORGET_CLASSES}2_100images_A${LAMBDA_RTF}_K${LAMBDA_KEEP}_C${LAMBDA_CE}_LAYER${LAYER}_$(date +%m%d%H%M)"

python /datanfs4/shenruoyan/FMUClip/finegrained/clip_unlearn_finegrained.py \
  --forget_classes "${FORGET_CLASSES}" \
  --coco_root "${COCO_ROOT}" \
  --df_root "${DF_ROOT}" \
  --sam3_mask_dir "${SAM3_MASK_DIR}" \
  --train_item_folder item2 \
  --retain_item_folder item1 \
  --test_item_folder item1 \
  --test_item_format json \
  --test_max_per_class 50 \
  --clip_arch "${CLIP_ARCH}" \
  --clip_pretrained "${CLIP_PRETRAINED}" \
  --clip_model_path "${CLIP_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_epoch "${MAX_EPOCH}" \
  --lr "${LR}" \
  --log_interval 1 \
  --weight_decay "${WEIGHT_DECAY}" \
  --lambda_rtf "${LAMBDA_RTF}" \
  --lambda_keep "${LAMBDA_KEEP}" \
  --lambda_ce "${LAMBDA_CE}" \
  --sample_k "${SAMPLE_K}" \
  --retain_topk "${RETAIN_TOPK}" \
  --device "${DEVICE}"

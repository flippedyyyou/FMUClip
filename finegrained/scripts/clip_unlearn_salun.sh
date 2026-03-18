#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a

# Adjust this to the appropriate CUDA device
export CUDA_VISIBLE_DEVICES=1

DEVICE="cuda"
TRAIN_SPLIT="train"
VAL_SPLIT="val"
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
COCO_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
FLICKR_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities"
FLICKR_INSTANCES_FILE="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/train/meta/instances.json"
FLICKR_IMAGE_ROOT="/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"
RETAIN_ITEM_FOLDER="item1"
CLIP_ARCH="local-dir:${OPENCLIP_PATH}"
BATCH_SIZE=16
NUM_WORKERS=4
MAX_EPOCH=10
LR=2e-5
WEIGHT_DECAY=5e-4
LAMBDA_DF=3
LAMBDA_DR=1
LAMBDA_UNI=3
METHOD="salun"
RETAIN_TOPK=5

SALUN_THRESHOLD=0.5
SALUN_MASK_STEPS=100

for DATASET in "coco2017_instances"; do  # "flickr30k_entities" or "coco2017_instances"
  if [ "${DATASET}" = "coco2017_instances" ]; then
    DF_ROOT="${COCO_DF_ROOT}"
    FORGET_CLASSES="airplane"
    TRAIN_ITEM_FOLDER="item5"
    
  elif [ "${DATASET}" = "flickr30k_entities" ]; then
    DF_ROOT="${FLICKR_DF_ROOT}"
    FORGET_CLASSES="table"
    TRAIN_ITEM_FOLDER="item7"
  else
    echo "Unknown dataset: ${DATASET}"
    exit 1
  fi

  OUTPUT_DIR="finegrained/output/salun_eval/${METHOD}_${DATASET}_${FORGET_CLASSES}7_DF${LAMBDA_DF}_DR${LAMBDA_DR}_UNI${LAMBDA_UNI}_$(date +%m%d%H%M)"

  echo "METHOD: ${METHOD}"
  echo "DATASET: ${DATASET}"
  echo "FORGET_CLASSES: ${FORGET_CLASSES}"
  echo "LEARNING_RATE: ${LR}"
  echo "TRAIN_ITEM_FOLDER: ${TRAIN_ITEM_FOLDER}"
  echo "SALUN_THRESHOLD: ${SALUN_THRESHOLD}"
  echo "SALUN_MASK_STEPS: ${SALUN_MASK_STEPS}"
  echo "OUTPUT_DIR: ${OUTPUT_DIR}"

  python finegrained/baselines/clip_unlearn_salun.py \
    --dataset "${DATASET}" \
    --forget_classes "${FORGET_CLASSES}" \
    --df_root "${DF_ROOT}" \
    --coco_root "${COCO_ROOT}" \
    --flickr_instances_file "${FLICKR_INSTANCES_FILE}" \
    --flickr_image_root "${FLICKR_IMAGE_ROOT}" \
    --train_item_folder "${TRAIN_ITEM_FOLDER}" \
    --retain_item_folder "${RETAIN_ITEM_FOLDER}" \
    --train_split "${TRAIN_SPLIT}" \
    --val_split "${VAL_SPLIT}" \
    --test_item_folder item1 \
    --test_item_format json \
    --test_max_per_class 50 \
    --joint_multilabel_max_per_class 50 \
    --clip_arch "${CLIP_ARCH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --max_epoch "${MAX_EPOCH}" \
    --lr "${LR}" \
    --log_interval 1 \
    --weight_decay "${WEIGHT_DECAY}" \
    --lambda_df "${LAMBDA_DF}" \
    --lambda_dr "${LAMBDA_DR}" \
    --lambda_uni "${LAMBDA_UNI}" \
    --retain_topk "${RETAIN_TOPK}" \
    --salun_threshold "${SALUN_THRESHOLD}" \
    --salun_mask_steps "${SALUN_MASK_STEPS}" \
    --device "${DEVICE}"
done

#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=3

DEVICE="cuda"
TRAIN_SPLIT="train"
VAL_SPLIT="val"
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
COCO_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
FLICKR_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities"
FLICKR_INSTANCES_FILE="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/train/meta/instances.json"
FLICKR_IMAGE_ROOT="/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"
RETAIN_ITEM_FOLDER="item1"
BATCH_SIZE=16
NUM_WORKERS=0
RETAIN_TOPK=5
SLUG_RATIO_DIVISOR=6
SLUG_SEARCH_MULTIPLIERS="0.5,1.0,2.0,4.0"
SLUG_MAX_CANDIDATES=3

for DATASET in "flickr30k_entities"; do
  if [ "${DATASET}" = "coco2017_instances" ]; then
    DF_ROOT="${COCO_DF_ROOT}"
    ITEM_NUM=5
    FORGET_CLASSES="airplane"
    TRAIN_ITEM_FOLDER="item${ITEM_NUM}"
  elif [ "${DATASET}" = "flickr30k_entities" ]; then
    DF_ROOT="${FLICKR_DF_ROOT}"
    ITEM_NUM=7
    FORGET_CLASSES="table"
    TRAIN_ITEM_FOLDER="item${ITEM_NUM}"
  else
    echo "Unknown dataset: ${DATASET}"
    exit 1
  fi

  OUTPUT_DIR="finegrained/ckpt/slug/${DATASET}/${FORGET_CLASSES}_${ITEM_NUM}"

  python finegrained/baselines/clip_unlearn_slug.py \
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
    --clip_path "${CLIP_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --retain_topk "${RETAIN_TOPK}" \
    --slug_parts all \
    --slug_ratio_divisor "${SLUG_RATIO_DIVISOR}" \
    --slug_search_multipliers "${SLUG_SEARCH_MULTIPLIERS}" \
    --slug_max_candidates_per_part "${SLUG_MAX_CANDIDATES}" \
    --device "${DEVICE}"
done

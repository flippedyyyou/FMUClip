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
MAX_EPOCH=20
LR=1e-6
WEIGHT_DECAY=5e-4
RETAIN_TOPK=5
GA_EVAL_INTERVAL=1

for DATASET in "flickr30k_entities"; do
  if [ "${DATASET}" = "coco2017_instances" ]; then
    DF_ROOT="${COCO_DF_ROOT}"
    FORGET_SPECS=(
      "airplane:5"
    )
  elif [ "${DATASET}" = "flickr30k_entities" ]; then
    DF_ROOT="${FLICKR_DF_ROOT}"
    FORGET_SPECS=(
      "girl:5"
      "table:5"
      "dog:7"
    )
  else
    echo "Unknown dataset: ${DATASET}"
    exit 1
  fi

  for FORGET_SPEC in "${FORGET_SPECS[@]}"; do
    IFS=":" read -r FORGET_CLASSES ITEM_NUM <<< "${FORGET_SPEC}"
    TRAIN_ITEM_FOLDER="item${ITEM_NUM}"
    OUTPUT_DIR="finegrained/ckpt/ga/${DATASET}/${FORGET_CLASSES}_${ITEM_NUM}"

    echo "DATASET: ${DATASET}"
    echo "FORGET_CLASSES: ${FORGET_CLASSES}"
    echo "TRAIN_ITEM_FOLDER: ${TRAIN_ITEM_FOLDER}"
    echo "OUTPUT_DIR: ${OUTPUT_DIR}"

    python finegrained/baselines/clip_unlearn_ga.py \
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
      --max_epoch "${MAX_EPOCH}" \
      --lr "${LR}" \
      --log_interval 1 \
      --weight_decay "${WEIGHT_DECAY}" \
      --retain_topk "${RETAIN_TOPK}" \
      --ga_eval_interval "${GA_EVAL_INTERVAL}" \
      --device "${DEVICE}"
  done
done

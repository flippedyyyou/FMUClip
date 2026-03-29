#!/usr/bin/env bash
set -euo pipefail

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0

DEVICE="cuda"
DATASET="coco2017_instances"
FORGET_CLASSES="airplane"
TRAIN_SPLIT="train"
VAL_SPLIT="val"
TRAIN_MAX_PER_CLASS=80
ITEM_NUM_LIST=(2 3 4 5 6)
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
COCO_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
FLICKR_INSTANCES_FILE="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/train/meta/instances.json"
FLICKR_IMAGE_ROOT="/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"
RETAIN_ITEM_FOLDER="item1"
BATCH_SIZE=16
NUM_WORKERS=0
MAX_EPOCH=20
LR=1e-5
WEIGHT_DECAY=5e-4
LAMBDA_DF=3
LAMBDA_DR=1
LAMBDA_UNI=3
RETAIN_TOPK=5
METHOD="cliperase"
DF_ROOT="${COCO_DF_ROOT}"
BASE_OUTPUT_DIR="finegrained/ckpt/sweep_${METHOD}_scale/${DATASET}/${FORGET_CLASSES}_item_ablation"
SUMMARY_CSV="${BASE_OUTPUT_DIR}/summary.csv"

mkdir -p "${BASE_OUTPUT_DIR}"
echo "train_item_folder,acc_f,acc_c,acc_r" > "${SUMMARY_CSV}"

for ITEM_NUM in "${ITEM_NUM_LIST[@]}"; do
  TRAIN_ITEM_FOLDER="item${ITEM_NUM}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TRAIN_ITEM_FOLDER}/trainmax_${TRAIN_MAX_PER_CLASS}/DF${LAMBDA_DF}_DR${LAMBDA_DR}_UNI${LAMBDA_UNI}"

  echo "METHOD: ${METHOD}"
  echo "DATASET: ${DATASET}"
  echo "FORGET_CLASSES: ${FORGET_CLASSES}"
  echo "TRAIN_ITEM_FOLDER: ${TRAIN_ITEM_FOLDER}"
  echo "TRAIN_MAX_PER_CLASS: ${TRAIN_MAX_PER_CLASS}"
  echo "OUTPUT_DIR: ${OUTPUT_DIR}"

  python finegrained/baselines/clip_unlearn_cliperase.py \
    --method "${METHOD}" \
    --dataset "${DATASET}" \
    --forget_classes "${FORGET_CLASSES}" \
    --df_root "${DF_ROOT}" \
    --coco_root "${COCO_ROOT}" \
    --flickr_instances_file "${FLICKR_INSTANCES_FILE}" \
    --flickr_image_root "${FLICKR_IMAGE_ROOT}" \
    --train_item_folder "${TRAIN_ITEM_FOLDER}" \
    --train_max_per_class "${TRAIN_MAX_PER_CLASS}" \
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
    --lambda_df "${LAMBDA_DF}" \
    --lambda_dr "${LAMBDA_DR}" \
    --lambda_uni "${LAMBDA_UNI}" \
    --retain_topk "${RETAIN_TOPK}" \
    --device "${DEVICE}"

  CFG_PATH="${OUTPUT_DIR}/config.json"
  row="$(python finegrained/summarize_item_result.py \
    --config "${CFG_PATH}" \
    --train_item_folder "${TRAIN_ITEM_FOLDER}")"
  echo "${row}" >> "${SUMMARY_CSV}"
done

echo "[DONE] summary csv: ${SUMMARY_CSV}"

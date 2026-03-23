#!/usr/bin/env bash
set -euo pipefail
set -a
source .env
set +a
export CUDA_VISIBLE_DEVICES=2

DEVICE="cuda"
TRAIN_SPLIT="train"
VAL_SPLIT="val"  # use "test" for flickr30k_entities if your df_root is train/test
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
COCO_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
FLICKR_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities"
FLICKR_INSTANCES_FILE="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/train/meta/instances.json"
FLICKR_IMAGE_ROOT="/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"
SAM3_MASK_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/mask"
BATCH_SIZE=4
NUM_WORKERS=0
MAX_EPOCH=20
LR=5e-6
WEIGHT_DECAY=3e-4
LAMBDA_RTF=2
LAMBDA_CE=1
SAMPLE_K=5
RETAIN_TOPK=5
METHOD="ours"
LAYER=1



for DATASET in "flickr30k_entities"; do  # "flickr30k_entities" or "coco2017_instances"
  if [ "${DATASET}" = "coco2017_instances" ]; then
    DF_ROOT="${COCO_DF_ROOT}"
    FORGET_CLASSES="airplane"
    ITEM_NUM=5
    TRAIN_ITEM_FOLDER="item${ITEM_NUM}"
    RETAIN_CACHE_PATH="/datanfs4/shenruoyan/FMUClip/finegrained/mask/coco/${TRAIN_ITEM_FOLDER}/retain_cache_${FORGET_CLASSES}.pt"
    
  elif [ "${DATASET}" = "flickr30k_entities" ]; then
    DF_ROOT="${FLICKR_DF_ROOT}"
    FORGET_CLASSES="table"
    ITEM_NUM=7
    TRAIN_ITEM_FOLDER="item${ITEM_NUM}"
    RETAIN_CACHE_PATH="/datanfs4/shenruoyan/FMUClip/finegrained/mask/flickr/${TRAIN_ITEM_FOLDER}/retain_cache_${FORGET_CLASSES}.pt"
  else
    echo "Unknown dataset: ${DATASET}"
    exit 1
  fi
 
  OUTPUT_DIR="finegrained/ckpt/${METHOD}/${DATASET}/${FORGET_CLASSES}_${ITEM_NUM}/A${LAMBDA_RTF}_C${LAMBDA_CE}_LAYER${LAYER}"

  echo "METHOD: ${METHOD}"
  echo "DATASET: ${DATASET}"
  echo "FORGET_CLASSES: ${FORGET_CLASSES}"
  echo "TRAIN_ITEM_FOLDER: ${TRAIN_ITEM_FOLDER}"
  echo "OUTPUT_DIR: ${OUTPUT_DIR}"

  python finegrained/clip_unlearn_finegrained.py \
    --dataset "${DATASET}" \
    --forget_classes "${FORGET_CLASSES}" \
    --df_root "${DF_ROOT}" \
    --flickr_instances_file "${FLICKR_INSTANCES_FILE}" \
    --flickr_image_root "${FLICKR_IMAGE_ROOT}" \
    --sam3_mask_dir "${SAM3_MASK_DIR}" \
    --retain_cache_path "${RETAIN_CACHE_PATH}" \
    --train_item_folder "${TRAIN_ITEM_FOLDER}" \
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
    --method "${METHOD}" \
    --lambda_rtf "${LAMBDA_RTF}" \
    --lambda_ce "${LAMBDA_CE}" \
    --sample_k "${SAMPLE_K}" \
    --layer "${LAYER}" \
    --retain_topk "${RETAIN_TOPK}" \
    --device "${DEVICE}"
done

#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=2

FORGET_CLASSES="airplane"
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
COCO_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
FLICKR_IMAGE_ROOT="/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"
FLICKR_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities"
FLICKR_INSTANCES_FILE="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/train/meta/instances.json"
TRAIN_SPLIT="train"
TRAIN_ITEM_FOLDER="item5"
BPE_PATH="/datanfs4/shenruoyan/FMUClip/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT="/datanfs4/shenruoyan/checkpoints/sam3/sam3.pt"
CONF_THRESH="0.1"

for DATASET in "flickr30k_entities"; do  # "flickr30k_entities" or "coco2017_instances"
  if [ "${DATASET}" = "coco2017_instances" ]; then
    DF_ROOT="${COCO_DF_ROOT}"
    FORGET_CLASSES="airplane"
    TRAIN_ITEM_FOLDER="item5"
    IMAGE_ROOT="${COCO_ROOT}/${TRAIN_SPLIT}2017"
    
  elif [ "${DATASET}" = "flickr30k_entities" ]; then
    DF_ROOT="${FLICKR_DF_ROOT}"
    FORGET_CLASSES="table"
    TRAIN_ITEM_FOLDER="item7"
    IMAGE_ROOT="${FLICKR_IMAGE_ROOT}"
  else
    echo "Unknown dataset: ${DATASET}"
    exit 1
  fi

MASK_ROOT="/datanfs4/shenruoyan/FMUClip/finegrained/mask/${DATASET}/${TRAIN_ITEM_FOLDER}"

# Stage 1: forget-class mask pngs (original behavior).
python /datanfs4/shenruoyan/FMUClip/finegrained/generate_sam3_mask.py \
  --dataset "${DATASET}" \
  --image-list "${DF_ROOT}/${TRAIN_SPLIT}/Df/${TRAIN_ITEM_FOLDER}/${FORGET_CLASSES}.txt" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${MASK_ROOT}/${FORGET_CLASSES}/" \
  --prompt "${FORGET_CLASSES}" \
  --bpe-path "${BPE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --confidence-threshold "${CONF_THRESH}"

# Stage 2: retain cache (best non-forget class idx + packed mask tensor per image).
python /datanfs4/shenruoyan/FMUClip/finegrained/generate_sam3_mask.py \
  --dataset "${DATASET}" \
  --build-retain-cache \
  --df-root "${DF_ROOT}" \
  --coco-root "${COCO_ROOT}" \
  --flickr-instances-file "${FLICKR_INSTANCES_FILE}" \
  --flickr-image-root "${FLICKR_IMAGE_ROOT}" \
  --train-split "${TRAIN_SPLIT}" \
  --train-item-folder "${TRAIN_ITEM_FOLDER}" \
  --forget-classes "${FORGET_CLASSES}" \
  --retain-cache-out "${MASK_ROOT}/retain_cache_${FORGET_CLASSES}.pt" \
  --bpe-path "${BPE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --confidence-threshold "${CONF_THRESH}"
done

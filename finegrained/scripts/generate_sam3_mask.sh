#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

FORGET_CLASSES="banana"
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
TRAIN_SPLIT="train"
TRAIN_ITEM_FOLDER="item3"
BPE_PATH="/datanfs4/shenruoyan/FMUClip/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
CHECKPOINT="/datanfs4/shenruoyan/checkpoints/sam3/sam3.pt"
CONF_THRESH="0.1"

# Stage 1: forget-class mask pngs (original behavior).
python /datanfs4/shenruoyan/FMUClip/finegrained/generate_sam3_mask.py \
  --image-list "${DF_ROOT}/${TRAIN_SPLIT}/Df/${TRAIN_ITEM_FOLDER}/${FORGET_CLASSES}.txt" \
  --image-root "${COCO_ROOT}/${TRAIN_SPLIT}2017" \
  --output-dir "/datanfs4/shenruoyan/FMUClip/finegrained/mask/coco/${TRAIN_ITEM_FOLDER}/${FORGET_CLASSES}/" \
  --prompt "${FORGET_CLASSES}" \
  --bpe-path "${BPE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --confidence-threshold "${CONF_THRESH}"

# Stage 2: retain cache (best non-forget class idx + packed mask tensor per image).
python /datanfs4/shenruoyan/FMUClip/finegrained/generate_sam3_mask.py \
  --build-retain-cache \
  --df-root "${DF_ROOT}" \
  --coco-root "${COCO_ROOT}" \
  --train-split "${TRAIN_SPLIT}" \
  --train-item-folder "${TRAIN_ITEM_FOLDER}" \
  --forget-classes "${FORGET_CLASSES}" \
  --retain-cache-out "/datanfs4/shenruoyan/FMUClip/finegrained/mask/coco/${TRAIN_ITEM_FOLDER}/retain_cache_${FORGET_CLASSES}.pt" \
  --bpe-path "${BPE_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --confidence-threshold "${CONF_THRESH}"

#!/usr/bin/env bash
set -euo pipefail
set -a
source .env
set +a
export CUDA_VISIBLE_DEVICES=0

DEVICE="cuda"
TRAIN_SPLIT="train"
VAL_SPLIT="val"
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
COCO_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
FLICKR_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities"
FLICKR_INSTANCES_FILE="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/train/meta/instances.json"
FLICKR_IMAGE_ROOT="/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"
SAM3_MASK_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/mask"
BATCH_SIZE=16
NUM_WORKERS=0
MAX_EPOCH=20
LR=2e-5
WEIGHT_DECAY=3e-4
LAMBDA_RTF=2
LAMBDA_CE=1
SAMPLE_K=5
RETAIN_TOPK=5
METHOD="gradcam"
LAYER=1
FORGET_LAYERS=1
RETAIN_LAYERS=11
RUN_TAG="$(date +%m%d%H%M%S)"

join_by_comma() {
  local IFS=","
  echo "$*"
}

for DATASET in "coco2017_instances"; do
  if [ "${DATASET}" = "coco2017_instances" ]; then
    DF_ROOT="${COCO_DF_ROOT}"
    CONCEPT_SPECS=(
      "airplane:5"
      "cow:5"
    )
  elif [ "${DATASET}" = "flickr30k_entities" ]; then
    DF_ROOT="${FLICKR_DF_ROOT}"
    CONCEPT_SPECS=(
      "girl:4"
      "boy:4"
    )
  else
    echo "Unknown dataset: ${DATASET}"
    exit 1
  fi

  MULTI_FORGET_CLASSES=()
  MULTI_TRAIN_ITEM_FOLDERS=()
  MULTI_RETAIN_CACHE_PATHS=()
  OUTPUT_NAME_PARTS=()

  for CONCEPT_SPEC in "${CONCEPT_SPECS[@]}"; do
    IFS=":" read -r FORGET_CLASS ITEM_NUM <<< "${CONCEPT_SPEC}"
    TRAIN_ITEM_FOLDER="item${ITEM_NUM}"
    RETAIN_CACHE_PATH="${SAM3_MASK_DIR}/${DATASET}/${TRAIN_ITEM_FOLDER}/retain_cache_${FORGET_CLASS}.pt"
    MULTI_FORGET_CLASSES+=("${FORGET_CLASS}")
    MULTI_TRAIN_ITEM_FOLDERS+=("${TRAIN_ITEM_FOLDER}")
    MULTI_RETAIN_CACHE_PATHS+=("${RETAIN_CACHE_PATH}")
    OUTPUT_NAME_PARTS+=("${FORGET_CLASS}_${ITEM_NUM}")
  done

  FORGET_CLASSES="$(join_by_comma "${MULTI_FORGET_CLASSES[@]}")"
  MULTI_FORGET_CLASSES_ARG="$(join_by_comma "${MULTI_FORGET_CLASSES[@]}")"
  MULTI_TRAIN_ITEM_FOLDERS_ARG="$(join_by_comma "${MULTI_TRAIN_ITEM_FOLDERS[@]}")"
  MULTI_RETAIN_CACHE_PATHS_ARG="$(join_by_comma "${MULTI_RETAIN_CACHE_PATHS[@]}")"
  RUN_NAME="$(IFS=_; echo "${OUTPUT_NAME_PARTS[*]}")"

  BASE_OUTPUT_DIR="finegrained/ckpt/vitb-32/sweep_${METHOD}_multi/${DATASET}/${RUN_NAME}_${RUN_TAG}"
  mkdir -p "${BASE_OUTPUT_DIR}"

  echo "DATASET: ${DATASET}"
  echo "FORGET_CLASSES: ${FORGET_CLASSES}"
  echo "MULTI_TRAIN_ITEM_FOLDERS: ${MULTI_TRAIN_ITEM_FOLDERS_ARG}"
  echo "BASE_OUTPUT_DIR: ${BASE_OUTPUT_DIR}"

  for FORGET_LAYER in ${FORGET_LAYERS}; do
    for RETAIN_LAYER in ${RETAIN_LAYERS}; do
      for LR in "1e-5" "1e-6" "5e-6"; do
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/LR_${LR}/L${LAYER}_FL${FORGET_LAYER}_RL${RETAIN_LAYER}"
        echo "[RUN] layer=${LAYER}, forget_layer=${FORGET_LAYER}, retain_layer=${RETAIN_LAYER}, output=${OUTPUT_DIR}"
        python finegrained/clip_unlearn_finegrained.py \
          --do_eval \
          --dataset "${DATASET}" \
          --forget_classes "${FORGET_CLASSES}" \
          --multi_forget_classes "${MULTI_FORGET_CLASSES_ARG}" \
          --multi_train_item_folders "${MULTI_TRAIN_ITEM_FOLDERS_ARG}" \
          --multi_retain_cache_paths "${MULTI_RETAIN_CACHE_PATHS_ARG}" \
          --coco_root "${COCO_ROOT}" \
          --df_root "${DF_ROOT}" \
          --flickr_instances_file "${FLICKR_INSTANCES_FILE}" \
          --flickr_image_root "${FLICKR_IMAGE_ROOT}" \
          --sam3_mask_dir "${SAM3_MASK_DIR}" \
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
          --forget_layer "${FORGET_LAYER}" \
          --retain_layer "${RETAIN_LAYER}" \
          --retain_topk "${RETAIN_TOPK}" \
          --device "${DEVICE}"
      done
    done
  done
done

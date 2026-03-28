#!/usr/bin/env bash
set -euo pipefail
set -a
source .env
set +a
export CUDA_VISIBLE_DEVICES=0

DEVICE="cuda"
TRAIN_SPLIT="train"
VAL_SPLIT="val"  # use "test" for flickr30k_entities if your df_root is train/test
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
COCO_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
FLICKR_DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities"
FLICKR_INSTANCES_FILE="/datanfs4/shenruoyan/FMUClip/data/classification/flickr30k_entities/train/meta/instances.json"
FLICKR_IMAGE_ROOT="/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"
SAM3_MASK_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/mask"
BATCH_SIZE=16
NUM_WORKERS=0
MAX_EPOCH=20
LR="2e-5"
WEIGHT_DECAY=3e-4
LAMBDA_RTF=2
LAMBDA_CE=1
SAMPLE_K=5
RETAIN_TOPK=5
METHOD="gradcam"
LAYERS="1 2 3 6 9 12 18 23 24"
FORGET_LAYER=1
RETAIN_LAYER=11

for DATASET in "coco2017_instances"; do  # "flickr30k_entities" or "coco2017_instances"
  if [ "${DATASET}" = "coco2017_instances" ]; then
    DF_ROOT="${COCO_DF_ROOT}"
    FORGET_SPECS=(
      "airplane:5"
      "cow:5"
    )
  elif [ "${DATASET}" = "flickr30k_entities" ]; then
    DF_ROOT="${FLICKR_DF_ROOT}"
    FORGET_SPECS=(
      "girl:4"
      "boy:4"
    )
  else
    echo "Unknown dataset: ${DATASET}"
    exit 1
  fi

  for FORGET_SPEC in "${FORGET_SPECS[@]}"; do
    IFS=":" read -r FORGET_CLASSES ITEM_NUM <<< "${FORGET_SPEC}"
    TRAIN_ITEM_FOLDER="item${ITEM_NUM}"
    RETAIN_CACHE_PATH="/datanfs4/shenruoyan/FMUClip/finegrained/mask/${DATASET}/${TRAIN_ITEM_FOLDER}/retain_cache_${FORGET_CLASSES}.pt"
    BASE_OUTPUT_DIR="finegrained/ckpt/vitb-32/sweep_${METHOD}_layer_ablation/${DATASET}/${FORGET_CLASSES}_${ITEM_NUM}"
    mkdir -p "${BASE_OUTPUT_DIR}"
    SUMMARY_CSV="${BASE_OUTPUT_DIR}/summary.csv"

    echo "DATASET: ${DATASET}"
    echo "FORGET_CLASSES: ${FORGET_CLASSES}"
    echo "TRAIN_ITEM_FOLDER: ${TRAIN_ITEM_FOLDER}"
    echo "OUTPUT_DIR: ${BASE_OUTPUT_DIR}"
    echo "layer,forget_layer,retain_layer,best_acc_mean,best_map_mean,output_dir" > "${SUMMARY_CSV}"

    for LAYER in ${LAYERS}; do
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/LR_${LR}/L${LAYER}_FL${FORGET_LAYER}_RL${RETAIN_LAYER}"
      echo "[RUN] layer=${LAYER}, forget_layer=${FORGET_LAYER}, retain_layer=${RETAIN_LAYER}, output=${OUTPUT_DIR}"
      python finegrained/clip_unlearn_finegrained.py \
        --do_eval \
        --dataset "${DATASET}" \
        --forget_classes "${FORGET_CLASSES}" \
        --coco_root "${COCO_ROOT}" \
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
        --forget_layer "${FORGET_LAYER}" \
        --retain_layer "${RETAIN_LAYER}" \
        --retain_topk "${RETAIN_TOPK}" \
        --device "${DEVICE}"

      CFG_PATH="${OUTPUT_DIR}/config.json"
      row="$(python /datanfs4/shenruoyan/FMUClip/finegrained/summarize_layer_result.py \
        --config "${CFG_PATH}" \
        --layer "${LAYER}" \
        --forget_layer "${FORGET_LAYER}" \
        --retain_layer "${RETAIN_LAYER}" \
        --output_dir "${OUTPUT_DIR}")"
      echo "${row}" >> "${SUMMARY_CSV}"
    done

    echo "[DONE] summary csv: ${SUMMARY_CSV}"
  done
done

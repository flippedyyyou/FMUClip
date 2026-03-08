#!/usr/bin/env bash
set -euo pipefail
set -a
source .env
set +a
export CUDA_VISIBLE_DEVICES=2

DEVICE="cuda"
FORGET_CLASSES=airplane
TRAIN_ITEM_FOLDER="item5"
COCO_ROOT="/datanfs4/shenruoyan/datasets/coco2017"
DF_ROOT="/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances"
SAM3_MASK_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/mask"
RETAIN_CACHE_PATH="/datanfs4/shenruoyan/FMUClip/finegrained/mask/coco/${TRAIN_ITEM_FOLDER}/retain_cache_${FORGET_CLASSES}.pt"
CLIP_ARCH="local-dir:${OPENCLIP_PATH}"
BATCH_SIZE=4
NUM_WORKERS=4
MAX_EPOCH=20
LR=5e-6
WEIGHT_DECAY=3e-4
LAMBDA_RTF=2
LAMBDA_CE=1
SAMPLE_K=5
RETAIN_TOPK=5
METHOD="ours"

# 可传入自定义层列表，例如：
# bash finegrained/scripts/sweep_ours_layers.sh "1 3 6 9 12"
LAYERS="${1:-1 3 6 9 12 18 24}"
RUN_TAG="$(date +%m%d%H%M%S)"
BASE_OUTPUT_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/output/unlearn/sweep_${METHOD}_${FORGET_CLASSES}_${RUN_TAG}"
mkdir -p "${BASE_OUTPUT_DIR}"
SUMMARY_CSV="${BASE_OUTPUT_DIR}/summary.csv"
# 第一列: best checkpoint 的 (forget_success + retain_topk_accuracy + retain_accuracy)/3
# 第二列: best checkpoint 的 (retain_topk_map + other_map)/2
echo "layer,best_acc_mean,best_map_mean,output_dir" > "${SUMMARY_CSV}"

for LAYER in ${LAYERS}; do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/L${LAYER}"
  echo "[RUN] method=${METHOD}, layer=${LAYER}, output=${OUTPUT_DIR}"
  python /datanfs4/shenruoyan/FMUClip/finegrained/clip_unlearn_finegrained.py \
    --forget_classes "${FORGET_CLASSES}" \
    --coco_root "${COCO_ROOT}" \
    --df_root "${DF_ROOT}" \
    --sam3_mask_dir "${SAM3_MASK_DIR}" \
    --retain_cache_path "${RETAIN_CACHE_PATH}" \
    --train_item_folder "${TRAIN_ITEM_FOLDER}" \
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
    --method "${METHOD}" \
    --lambda_rtf "${LAMBDA_RTF}" \
    --lambda_ce "${LAMBDA_CE}" \
    --sample_k "${SAMPLE_K}" \
    --layer "${LAYER}" \
    --retain_topk "${RETAIN_TOPK}" \
    --device "${DEVICE}"

  CFG_PATH="${OUTPUT_DIR}/config.json"
  row="$(python /datanfs4/shenruoyan/FMUClip/finegrained/summarize_layer_result.py \
    --config "${CFG_PATH}" \
    --layer "${LAYER}" \
    --output_dir "${OUTPUT_DIR}")"
  echo "${row}" >> "${SUMMARY_CSV}"
done

echo "[DONE] summary csv: ${SUMMARY_CSV}"

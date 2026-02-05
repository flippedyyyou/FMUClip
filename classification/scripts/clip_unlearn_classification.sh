#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
code_path=$(cd "$(dirname "$0")/.." && pwd)

# ====== CIFAR100 Unlearning config ======
DATA_ROOT="/datanfs4/shenruoyan/datasets/cifar-100-python"
FORGET_LIST="/datanfs4/shenruoyan/FMUClip/classification/data_split/cifar100_forget0_complete.jsonl"
SAM3_MASK_DIR="/datanfs4/shenruoyan/FMUClip/classification/mask/cifar100/train/cifar100_forget0_10percent"

# ====== Hyperparameters ======
BATCH_SIZE=4
MAX_EPOCH=10
LR=1e-6
WEIGHT_DECAY=5e-4
SAMPLE_K=5
LAMBDA_ATTN=0.5
LAMBDA_SYN=0
LAMBDA_KEEP=0
LAMBDA_UNI=100
CONCEPT_TOKEN=apple

OUTPUT_DIR="${code_path}/output/clip_cifar100_unlearn_rtf${LAMBDA_ATTN}_rdr${LAMBDA_SYN}_uni${LAMBDA_UNI}_${CONCEPT_TOKEN}_$(date +%m%d%H%M)"

python "${code_path}/clip_unlearn_classification.py" \
  --dict-path "/datanfs4/shenruoyan/FMUClip/classification/datasets/cifar100.py" \
  --dataset cifar100 \
  --data_root "${DATA_ROOT}" \
  --forget_list "${FORGET_LIST}" \
  --sam3_mask_dir "${SAM3_MASK_DIR}" \
  --output "${OUTPUT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --max_epoch ${MAX_EPOCH} \
  --lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --sample_k ${SAMPLE_K} \
  --lambda_attn ${LAMBDA_ATTN} \
  --lambda_syn ${LAMBDA_SYN} \
  --lambda_keep ${LAMBDA_KEEP} \
  --lambda_uni ${LAMBDA_UNI}
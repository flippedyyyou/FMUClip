#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
code_path=$(cd "$(dirname "$0")/.." && pwd)

# ====== CIFAR100 Unlearning config ======
DATA_ROOT="/datanfs4/shenruoyan/datasets/cifar-100-python"
FORGET_LIST="/datanfs4/shenruoyan/FMUClip/classification/data_split/cifar100_apple_forget_list.txt"
SAM3_MASK_DIR="/datanfs4/shenruoyan/FMUClip/classification/mask/cifar100/train/apple"

# ====== Hyperparameters ======
BATCH_SIZE=4
MAX_EPOCH=20
LR=1e-6
WEIGHT_DECAY=5e-4
SAMPLE_K=5
METHOD="cliperase"
LAMBDA_ATTN=1
LAMBDA_SYN=0
LAMBDA_KEEP=0
LAMBDA_UNI=10
CONCEPT_TOKEN=apple

OUTPUT_DIR="${code_path}/output/clip_cifar100_unlearn_${CONCEPT_TOKEN}_$(date +%m%d%H%M)"

python "${code_path}/clip_unlearn_classification.py" \
  --dataset cifar100 \
  --data_root "${DATA_ROOT}" \
  --method "${METHOD}" \
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
  --lambda_uni ${LAMBDA_UNI} \
  --concept_token "${CONCEPT_TOKEN}"

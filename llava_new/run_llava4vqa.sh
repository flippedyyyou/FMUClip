set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=3

QUESTION_FILE="data/VQA/coco2017_instances/carrot/df_llava_vqa.jsonl"
OUTPUT_FILE="llava_new/test_slug.jsonl"
BATCH_SIZE=8
MAX_NEW_TOKENS=30

CLIP_PATH="finegrained/ckpt/sweep_ours/coco2017_instances/carrot_7/L1"


python3 llava_new/llava4vqa.py \
  --question-file "$QUESTION_FILE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --llava-path "${LLAVA_PATH}" \
  --output-file "$OUTPUT_FILE" \
  --clip-path "${CLIP_PATH}"


set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=0

QUESTION_FILE="data/VQA/coco2017_instances/carrot/df_llava_vqa.jsonl"
OUTPUT_FILE="llava_new/test_slug.jsonl"
BATCH_SIZE=8
MAX_NEW_TOKENS=30
# CLIP_PATH="/datanfs4/shenruoyan/FMUClip/finegrained/output/unlearn/coco2017_instances/sweep_ours_carrot7_0316220832/L1_sota/HF"
CLIP_PATH="/datanfs4/shenruoyan/FMUClip/finegrained/output/baselines/slug_coco2017_instances_carrot7_03160110/HF"


python3 llava_new/llava4vqa.py \
  --question-file "$QUESTION_FILE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --llava-path "${LLAVA_PATH}" \
  --output-file "$OUTPUT_FILE" \
  --clip-path "${CLIP_PATH}"


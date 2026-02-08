WEIGHT_DIR="/datanfs4/shenruoyan/FMUClip/retrieval/output/clip_flickr_unlearn1k_minsim_horse_df1_dr1_uni1_export0107/unlearned_clip"

QUESTION_FILE="/datanfs4/shenruoyan/FMUClip/Df/flickr30k/non-horse_llava_vqa.jsonl"
BASE_MODEL="/datanfs2/shenruoyan/checkpoints/llava-v1.5-7b"
IMAGE_FOLDER="/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"
OUTPUT_DIR="/datanfs4/shenruoyan/FMUClip/retrieval/output/clip_flickr_unlearn1k_minsim_horse_df1_dr1_uni1_export0107/llava"
# 存放所有推理结果
# mkdir -p "$OUTPUT_DIR"
result_file="${OUTPUT_DIR}/llava_non-horse_answer1.jsonl"

CUDA_VISIBLE_DEVICES=1 python /datanfs4/shenruoyan/FMUClip/llava/eval/model_vqa_loader_unlearned.py \
    --model-base ${BASE_MODEL} \
    --model-path ${BASE_MODEL} \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$result_file" \
    --temperature 0 \
    --max_new_tokens 50 \
    --our_vision_encoder \
    --ve_name ${WEIGHT_DIR} \
    --conv-mode vicuna_v1 &

WEIGHT_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/output/unlearn/coco2017_instances/sweep_ours_carrot7_0316220832/L1_sota/"

QUESTION_FILE="/datanfs4/shenruoyan/FMUClip/data/VQA/coco2017_instances/carrot/df_llava_vqa.jsonl"
BASE_MODEL="/datanfs2/shenruoyan/checkpoints/llava-v1.5-7b"
# IMAGE_FOLDER="/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"
IMAGE_FOLDER=/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances/val/test_images
OUTPUT_DIR="/datanfs4/shenruoyan/FMUClip/finegrained/output/unlearn/coco2017_instances/sweep_ours_carrot7_0316220832/L1_sota/llava"
# 存放所有推理结果
# mkdir -p "$OUTPUT_DIR"
result_file="${OUTPUT_DIR}/carrot_vqa_answer.jsonl"

CUDA_VISIBLE_DEVICES=1 python /datanfs4/shenruoyan/FMUClip/llava/eval/model_vqa_loader_unlearned.py \
    --model-base ${BASE_MODEL} \
    --model-path ${BASE_MODEL} \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$result_file" \
    --temperature 0 \
    --max_new_tokens 100 \
    --our_vision_encoder \
    --ve_name ${WEIGHT_DIR} \
    --conv-mode vicuna_v1 &

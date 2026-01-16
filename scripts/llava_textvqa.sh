########################################
# 定义CLIP权重路径变量
########################################
WEIGHT_DIR="/datanfs4/shenruoyan/FMUClip/retrieval/output/clip_flickr_unlearn1k_graddiff_horse_df1_dr1_uni0_export0108/unlearned_clip"
ORIGINAL_WEIGHT="/datanfs2/shenruoyan/checkpoints/clip-vit-large-patch14-336"
# 在权重文件夹内部建立软链接（如果已经存在则先删除）
# transformers 默认寻找 pytorch_model.bin 而非 .pt
# cp -r ${ORIGINAL_WEIGHT}/* ${WEIGHT_DIR}/
# rm -f ${WEIGHT_DIR}/pytorch_model.bin
# ln -s ${WEIGHT_DIR}/clip_unlearned.pt ${WEIGHT_DIR}/pytorch_model.bin

########################################
# benchmark数据集推理
########################################
QUESTION_FILE="/datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_FOLDER="/datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/train_val_images/train_images"
OUTPUT_DIR="/datanfs4/shenruoyan/FMUClip/retrieval/output/clip_flickr_unlearn1k_graddiff_horse_df1_dr1_uni0_export0108/llava"
ANNOTATION_FILE="/datanfs2/ljq/doing/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json"
# 存放所有推理结果
# mkdir -p "$OUTPUT_DIR"
result_file="${OUTPUT_DIR}/llava_textvqa_answer.jsonl"

CUDA_VISIBLE_DEVICES=5 python /datanfs4/shenruoyan/FMUClip/llava/eval/model_vqa_loader_unlearned.py \
    --model-base /datanfs2/shenruoyan/checkpoints/llava-v1.5-7b \
    --model-path /datanfs2/shenruoyan/checkpoints/llava-v1.5-7b \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$result_file" \
    --temperature 0 \
    --max_new_tokens 128 \
    --our_vision_encoder \
    --ve_name ${WEIGHT_DIR} \
    --conv-mode vicuna_v1 &

########################################
# 评测准确率
########################################
eval_log="${OUTPUT_DIR}/llava_textvqa_eval.txt"
echo "[RUN ] 统一评测 OUTPUT_DIR=$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=5 python /datanfs4/shenruoyan/FMUClip/llava/eval/eval_textvqa_unlearned.py \
    --annotation-file "$ANNOTATION_FILE" \
    --result-file "$result_file" \
    2>&1 | tee "$eval_log"
    

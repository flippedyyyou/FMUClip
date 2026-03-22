set -a
source .env
set +a

# CKPT_PATH="/datanfs4/shenruoyan/FMUClip/finegrained/output/unlearn/coco2017_instances/sweep_ours_carrot7_0316220832/L1_sota/"
CKPT_PATH="/datanfs4/shenruoyan/FMUClip/finegrained/output/baselines/slug_coco2017_instances_carrot7_03160110"

python llava_new/convert_open_clip_to_hf.py \
    --model "ViT-L-14-336-quickgelu" \
    --pretrained "${CKPT_PATH}/clip_finegrained_slug.pth" \
    --pytorch_dump_folder_path "${CKPT_PATH}/HF" \
    --config_path "llava_new/config.json"

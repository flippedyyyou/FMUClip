# 运行 Python 脚本
CUDA_VISIBLE_DEVICES=2 python /datanfs4/shenruoyan/FMUClip/finegrained/generate_sam3_mask.py \
  --image-list "/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances/train/Df/item2/bird.txt" \
  --image-root "/datanfs4/shenruoyan/datasets/coco2017/train2017" \
  --output-dir "/datanfs4/shenruoyan/FMUClip/finegrained/mask/coco/item2/bird/train" \
  --prompt "bird" \
  --bpe-path "/datanfs4/shenruoyan/FMUClip/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --checkpoint "/datanfs4/shenruoyan/checkpoints/sam3/sam3.pt" \
  --confidence-threshold "0.1"
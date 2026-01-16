# 运行 Python 脚本
CUDA_VISIBLE_DEVICES=2 python /datanfs4/shenruoyan/FMUClip/retrieval/generate_sam3_mask.py \
  --image-list "/datanfs4/shenruoyan/FMUClip/Df/coco/forget_horse_train.txt" \
  --image-root "/datanfs4/shenruoyan/datasets/coco2014" \
  --output-dir "/datanfs4/shenruoyan/FMUClip/retrieval/mask/coco/horse/train" \
  --prompt "horse" \
  --bpe-path "/datanfs4/shenruoyan/FMUClip/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --checkpoint "/datanfs4/shenruoyan/checkpoints/sam3/sam3.pt" \
  --confidence-threshold "0.3"
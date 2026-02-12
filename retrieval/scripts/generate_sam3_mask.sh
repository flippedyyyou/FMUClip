# 运行 Python 脚本
CUDA_VISIBLE_DEVICES=2 python /datanfs4/shenruoyan/FMUClip/retrieval/generate_sam3_mask.py \
  --image-list "/datanfs4/shenruoyan/FMUClip/retrieval/Df/item5+/apple.txt" \
  --image-root "/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images" \
  --output-dir "/datanfs4/shenruoyan/FMUClip/retrieval/mask/flickr30k/item5+/apple/train" \
  --prompt "apple" \
  --bpe-path "/datanfs4/shenruoyan/FMUClip/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --checkpoint "/datanfs4/shenruoyan/checkpoints/sam3/sam3.pt" \
  --confidence-threshold "0.3"
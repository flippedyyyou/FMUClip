# 运行 Python 脚本
source .env

CUDA_VISIBLE_DEVICES=2 python /datanfs4/shenruoyan/FMUClip/retrieval/generate_sam3_mask.py \
  --image-list "data/classification/flickr30k_entities/Df/item5+/apple.txt" \
  --image-root "${FLICKR30K_PATH}" \
  --output-dir "/datanfs4/shenruoyan/FMUClip/retrieval/mask/flickr30k/apple/train" \
  --prompt "apple" \
  --bpe-path "${SAM3_PATH}/assets/bpe_simple_vocab_16e6.txt.gz" \
  --checkpoint "${SAM3_PATH}/sam3.pt" \
  --confidence-threshold "0.3"
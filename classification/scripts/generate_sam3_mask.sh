# 运行 Python 脚本
CUDA_VISIBLE_DEVICES=2 python /datanfs4/shenruoyan/FMUClip/classification/generate_sam3_mask.py \
  --jsonl-path "/datanfs4/shenruoyan/FMUClip/classification/data_split/cifar100_forget0_10percent.jsonl" \
  --output-dir "/datanfs4/shenruoyan/FMUClip/classification/mask/cifar100/train/cifar100_forget0_10percent" \
  --bpe-path "/datanfs4/shenruoyan/FMUClip/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --checkpoint "/datanfs4/shenruoyan/checkpoints/sam3/sam3.pt" \
  --confidence-threshold "0.5" \
  --data-root "/datanfs4/shenruoyan/datasets/cifar-100-python" \
  --train
set -euo pipefail

set -a
source .env
set +a

CUDA_VISIBLE_DEVICES= python /datanfs4/shenruoyan/FMUClip/finegrained/eval_collateral_forget.py \
  --forget_classes banana \
  --coco_root /datanfs4/shenruoyan/datasets/coco2017 \
  --df_root /datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances \
  --train_item_folder item3 \
  --retain_item_folder item1 \
  --test_item_folder item1 \
  --clip_arch "local-dir:${OPENCLIP_PATH}" \
  --after_ckpt /datanfs4/shenruoyan/FMUClip/finegrained/output/baselines/cliperase_banana3_100images_DF3_DR1_UNI3_03040200/clip_finegrained_cliperase.pth \
  --cooccur_eval_k 5 \
  --group_source train_df \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda \
  --output_dir /datanfs4/shenruoyan/FMUClip/finegrained/output/collateral_eval

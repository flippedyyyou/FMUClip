#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

root=/datanfs4/shenruoyan/FMUClip
code_path=${root}/retrieval

# ========= 通用参数 =========
tta_steps=8
lr=1e-6
weight_decay=5e-4
reward_arch=ViT-L-14
reward_amplify=0
reward_process=1
process_batch=0
sample_k_t2i=12
sample_k_i2t=16
multiple_reward_models=0

# ========= 动量参数 =========
momentum_update=0 
update_freq=32
tta_momentum=0.9  #0.9998
update_w=1.0

# ========= Unlearn 参数 =========
lambda_df=3
lambda_dr=1
lambda_uni=3
max_epoch=20
neg_mode="cliperase"
concept_token="apple"
cfg_yaml=${code_path}/lavis/projects/clip/exp_flickr_unlearn_tta_llava.yaml
runfile=${code_path}/clip_unlearn_baseline.py
external_test_root="/datanfs4/shenruoyan/datasets/cifar-100-python"

# ========= 运行 =========
output=${code_path}/output/clip_flickr30k_unlearn1k_${neg_mode}_${concept_token}_df${lambda_df}_dr${lambda_dr}_uni${lambda_uni}_${date}
export_dir=${code_path}/output/clip_flickr30k_unlearn1k_${neg_mode}_${concept_token}_df${lambda_df}_dr${lambda_dr}_uni${lambda_uni}_${date}

echo "🚀 Running Unlearning baseline cliperase"
echo "Output -> ${output}"

# ---- image→text ----
python ${runfile} \
    --arch ViT-L-14-336px \
    --cfg-path ${cfg_yaml} \
    --tta_steps ${tta_steps} \
    --lr ${lr} \
    --weight_decay ${weight_decay} \
    --momentum_update ${momentum_update} \
    --update_freq ${update_freq} \
    --tta_momentum ${tta_momentum} \
    --update_w ${update_w} \
    --reward_arch ${reward_arch} \
    --reward_amplify ${reward_amplify} \
    --reward_process ${reward_process} \
    --process_batch ${process_batch} \
    --sample_k ${sample_k_i2t} \
    --output ${output} \
    --multiple_reward_models ${multiple_reward_models} \
    --retrieval_task "image2text" \
    --concept_token ${concept_token} \
    --lambda_df ${lambda_df} \
    --lambda_dr ${lambda_dr} \
    --lambda_uni ${lambda_uni} \
    --max_epoch ${max_epoch} \
    --cliperase \
    --sam3_mask_dir ${code_path}/mask/flickr30k/apple/train \
    --sam3_mask_suffix .png \
    --save_unlearned_model \
    --unlearned_model_name clip_unlearned.pt \
    --unlearned_meta_name clip_unlearned_meta.json \
    --unlearned_subdir unlearned_clip \
    --forget_train_file "data/classification/flickr30k_entities/Df/item5+" \
    --forget_test_file "/datanfs4/shenruoyan/FMUClip/retrieval/Df/flickr30k/forget_apple_test.txt" \
    --external_test_dataset "cifar100" \
    --external_test_root "${external_test_root}" \
    --external_test_forget_class "${concept_token}" \
    --external_test_image_size 336 \
    --external_test_batch_size 64 \
    --external_test_num_workers 4 \
    --external_eval_only

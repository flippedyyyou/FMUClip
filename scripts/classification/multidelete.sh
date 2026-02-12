python baselines/multidelete.py \
  --model_path path/to/model.pt \
  --teacher_model_path path/to/teacher.pt \
  --df_dataset_path path/to/df_dataset.pt \
  --dr_dataset_path path/to/dr_dataset.pt \
  --unlearn_method vlul-md-multi-image \
  --task nlvr \
  --output_dir output/multidelete_run \
  --per_device_train_batch_size 8 \
  --num_train_epochs 5 \
  --save_model
  
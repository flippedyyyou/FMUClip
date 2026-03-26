import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Support direct script execution:
# python finegrained/baselines/clip_unlearn_salun.py
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from finegrained.params import build_parser as build_base_parser, resolve_dataset_paths
from finegrained.clip_finegrained_baseline import ClipFinegrainedBaseline


def build_parser():
    parser = build_base_parser()
    parser.description = "Finegrained CLIP unlearning using SalUn (Saliency Unlearning)."
    parser.add_argument("--salun_threshold", type=float, default=0.5,
                        help="Threshold for Saliency Unlearning mask (e.g. 0.5 means top 50% weights).")
    parser.add_argument("--salun_mask_steps", type=int, default=100,
                        help="Number of steps (batches) to accumulate gradients for masking.")
    return parser


class SalUn(ClipFinegrainedBaseline):
    def __init__(self, args):
        super().__init__(args)

    def generate_mask(self, model, tokenize_fn, forget_loader, class_names, forget_indices, device) -> Dict[str, torch.Tensor]:
        # Usually gradients can be computed in eval mode, but train is also fine. We just want gradient accumulation.
        model.eval()
        model.zero_grad()

        gradients = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                gradients[name] = torch.zeros_like(param, device=device)

        steps = 0
        max_steps = self.args.salun_mask_steps

        print(f"Generating SalUn mask using up to {max_steps} batches...")
        for batch in forget_loader:
            if steps >= max_steps:
                break

            images = batch["image"].to(device, non_blocking=True)
            txt_labels = batch.get("label", batch.get("forget_label"))
            if txt_labels is None:
                # Fallback if the dataset outputs slightly different keys
                txt_labels = batch["label"]

            # Convert labels to text descriptions
            texts = self._labels_to_texts(txt_labels, class_names, forget_indices)
            text_tokens = tokenize_fn(texts).to(device)

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                # Compute Gradient Ascent Loss (Negative Contrastive Loss)
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features.t())
                logits_per_text = logit_scale * (text_features @ image_features.t())

                labels = torch.arange(images.size(0), device=device)

                # Standard contrastive loss
                loss_i2t = F.cross_entropy(logits_per_image, labels)
                loss_t2i = F.cross_entropy(logits_per_text, labels)
                loss = (loss_i2t + loss_t2i) / 2.0

                # We want the gradient of the Unlearning Objective (which is -loss)
                # Equivalently, we want the gradient of Ascent, so we minimize -loss.
                unlearn_loss = -loss

            model.zero_grad()
            unlearn_loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        gradients[name] += torch.abs(param.grad.detach())

            steps += 1

        print("Gradient accumulation complete. Computing mask threshold...")
        # Flatten all gradients to find the threshold
        all_elements = torch.cat([tensor.flatten() for tensor in gradients.values()])
        all_elements = -all_elements  # Because we want top N%, sort ascending and pick from end, or sort descending.

        # Calculate the threshold index
        threshold_index = int(len(all_elements) * self.args.salun_threshold)
        if threshold_index >= len(all_elements):
            threshold_index = len(all_elements) - 1

        print(f"Total parameters: {len(all_elements)}, thresholding top {self.args.salun_threshold*100}%")

        # Using topk to find the threshold value directly, which is faster than full sort
        # We need the values that are in the top `salun_threshold` fraction.
        # Since all_elements is negative gradients, we want the smallest (most negative) elements.
        threshold_val, _ = torch.kthvalue(all_elements, threshold_index + 1)

        hard_dict = {}
        with torch.no_grad():
            for key, tensor in gradients.items():
                # -tensor <= threshold_val means tensor >= -threshold_val
                # which means the gradient magnitude is in the top threshold%
                mask_tensor = torch.zeros_like(tensor)
                mask_tensor[-tensor <= threshold_val] = 1.0
                hard_dict[key] = mask_tensor

        return hard_dict

    def _apply_mask_to_grads(self, model, mask):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and name in mask:
                param.grad *= mask[name]

    def _restore_masked_params(self, model, mask, theta0):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask and name in theta0:
                    mask_tensor = mask[name]
                    inv_mask_tensor = 1.0 - mask_tensor
                    # For weights where mask is 0, restore the original value
                    param.data.mul_(mask_tensor).add_(theta0[name] * inv_mask_tensor)

    def run(self):
        device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        model, tokenize_fn, image_size, backend = self._load_clip_backend(device)
        model = model.float().to(device)

        train_transform = self._build_eval_transform(image_size)
        df_dataset, dr_dataset = self._build_train_datasets(transform=train_transform)

        # Note: We only need the forget dataset to do gradient ascent.
        df_loader = DataLoader(
            df_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )
        dr_loader = DataLoader(
            dr_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )

        class_names = self._build_class_name_list()
        forget_indices, retain_indices = self._build_forget_retain_indices(self.args.forget_classes)
        retain_topk_indices = self._compute_topk_retain_classes(
            df_dataset=df_dataset,
            forget_indices=forget_indices,
            k=self.args.retain_topk,
        )

        # 1. Generate the SalUn mask
        salun_mask = self.generate_mask(model, tokenize_fn, df_loader, class_names, forget_indices, device)

        # Save original weights to restore the masked out parameters later
        theta0 = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    theta0[name] = param.clone().detach()

        # 2. Setup Unlearning Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scaler = torch.amp.GradScaler(init_scale=1024)

        os.makedirs(self.args.output_dir, exist_ok=True)

        print("Starting SalUn with Random Labeling...")
        for ep in range(self.args.max_epoch):
            model.train()
            iters_per_epoch = min(len(df_loader), len(dr_loader))

            df_iter = iter(df_loader)
            dr_iter = iter(dr_loader)

            running_df_loss = 0.0
            running_dr_loss = 0.0

            for it in range(iters_per_epoch):
                df_batch = next(df_iter)
                dr_batch = next(dr_iter)

                # --- FORGET (Random Labeling) ---
                images_df = df_batch["image"].to(device, non_blocking=True)

                # Create RANDOM text descriptions for the forget set
                random_indices = torch.randint(0, len(class_names), (images_df.size(0),))
                random_texts = [f"a photo of {class_names[idx].replace('_', ' ')}" for idx in random_indices]
                text_tokens_df_random = tokenize_fn(random_texts).to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    # Forward pass for Forget (Random)
                    image_features_df = model.encode_image(images_df)
                    image_features_df = image_features_df / image_features_df.norm(dim=-1, keepdim=True)

                    text_features_df = model.encode_text(text_tokens_df_random)
                    text_features_df = text_features_df / text_features_df.norm(dim=-1, keepdim=True)

                    logit_scale = model.logit_scale.exp()
                    logits_per_image_df = logit_scale * (image_features_df @ text_features_df.t())
                    logits_per_text_df = logit_scale * (text_features_df @ image_features_df.t())

                    labels_df = torch.arange(images_df.size(0), device=device)

                    loss_i2t_df = F.cross_entropy(logits_per_image_df, labels_df)
                    loss_t2i_df = F.cross_entropy(logits_per_text_df, labels_df)

                    # df loss uses standard CE but towards the random labels
                    loss_df = (loss_i2t_df + loss_t2i_df) / 2.0

                    # --- RETAIN (True Labeling) ---
                    images_dr = dr_batch["image"].to(device, non_blocking=True)
                    txt_labels_dr = dr_batch.get("label", dr_batch.get("retain_label"))
                    if txt_labels_dr is None:
                        txt_labels_dr = dr_batch["label"]

                    texts_dr = self._labels_to_texts(txt_labels_dr, class_names, retain_indices)
                    text_tokens_dr = tokenize_fn(texts_dr).to(device)

                    # Forward pass for Retain (True)
                    image_features_dr = model.encode_image(images_dr)
                    image_features_dr = image_features_dr / image_features_dr.norm(dim=-1, keepdim=True)

                    text_features_dr = model.encode_text(text_tokens_dr)
                    text_features_dr = text_features_dr / text_features_dr.norm(dim=-1, keepdim=True)

                    logits_per_image_dr = logit_scale * (image_features_dr @ text_features_dr.t())
                    logits_per_text_dr = logit_scale * (text_features_dr @ image_features_dr.t())

                    labels_dr = torch.arange(images_dr.size(0), device=device)

                    loss_i2t_dr = F.cross_entropy(logits_per_image_dr, labels_dr)
                    loss_t2i_dr = F.cross_entropy(logits_per_text_dr, labels_dr)

                    loss_dr = (loss_i2t_dr + loss_t2i_dr) / 2.0

                    # Total Loss (weighted by hyperparams if needed, here applying standard lambda_df and lambda_dr)
                    total_loss = self.args.lambda_df * loss_df + self.args.lambda_dr * loss_dr

                scaler.scale(total_loss).backward()

                # Apply mask to gradients
                self._apply_mask_to_grads(model, salun_mask)

                scaler.step(optimizer)
                scaler.update()

                # Restore weights that should not be updated (where mask == 0)
                # This prevents weight decay or momentum from altering them
                self._restore_masked_params(model, salun_mask, theta0)

                running_df_loss += loss_df.detach().item()
                running_dr_loss += loss_dr.detach().item()

                if (it + 1) % self.args.log_interval == 0:
                    print(f"[SalUn EP {ep + 1}/{self.args.max_epoch}] it={it+1}/{iters_per_epoch} "
                          f"df_rl_loss={running_df_loss/(it+1):.4f} dr_loss={running_dr_loss/(it+1):.4f}")

        # Evaluation (only once after training)
        print("Evaluating...")
        eval_transform = self._build_eval_transform(image_size)
        forget_loader_eval, retain_loader_eval = self._build_test_dataloaders_from_folders(transform=eval_transform)

        text_features = self._encode_text_features(model, class_names, tokenize_fn, device)

        forget_cache = self._collect_eval_cache(model, forget_loader_eval, text_features, device)
        retain_cache = self._collect_eval_cache(model, retain_loader_eval, text_features, device)

        forget_acc = self._single_accuracy_from_cache(forget_cache)
        retain_acc = self._single_accuracy_from_cache(retain_cache)

        score = (1.0 - forget_acc) + retain_acc
        current_epoch = self.args.max_epoch
        print(f"[Eval EP {current_epoch}/{self.args.max_epoch}] "
              f"forget_success={1.0 - forget_acc:.4f} "
              f"retain_acc={retain_acc:.4f} "
              f"score={score:.4f}")

        metrics = {
            "forget_success": 1.0 - float(forget_acc),
            "retain_accuracy": float(retain_acc),
            "selection_score": float(score)
        }
        model.clip_model.save_pretrained(self.args.output_dir)
        best_epoch_info = {
            "best_epoch": current_epoch,
            "forget_success": metrics["forget_success"],
            "retain_accuracy": metrics["retain_accuracy"],
        }
        self._write_best_epoch_info(best_epoch_info)
        print(f"[Saved] epoch={current_epoch} score={score:.4f} -> overwrite checkpoint")

        print(f"\nFinal epoch: {current_epoch}, "
              f"forget_success={metrics['forget_success']:.4f}, "
              f"retain_acc={metrics['retain_accuracy']:.4f}, "
              f"score={score:.4f}")

        best_model, best_tokenize_fn, best_image_size = self._load_model_from_pretrained_dir(self.args.output_dir)
        final_eval_metrics = self.run_original_eval(
            model=best_model,
            tokenize_fn=best_tokenize_fn,
            image_size=best_image_size,
            retain_topk_indices=retain_topk_indices,
        )
        self._merge_final_eval_into_best_epoch(final_eval_metrics, base_info=best_epoch_info)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = resolve_dataset_paths(args)

    # Required parameters for dataset building in baseline:
    # ensure annotation file paths are populated
    # (resolve_dataset_paths usually takes care of this)

    salun = SalUn(args)
    salun.run()


if __name__ == "__main__":
    main()

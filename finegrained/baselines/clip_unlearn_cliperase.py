import copy
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

class Clipeasre(ClipFinegrainedBaseline):
    def __init__(self, args):
        super().__init__(args)
        def run_cliperase(self) -> None:
            device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
            model, tokenize_fn, image_size, backend = self._load_clip_backend(device)
            model = model.float().to(device)
            teacher = copy.deepcopy(model).eval()
            for param in teacher.parameters():
                param.requires_grad = False

            train_transform = self._build_eval_transform(image_size)
            df_dataset, dr_dataset = self._build_train_datasets(transform=train_transform)
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

            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            scaler = torch.amp.GradScaler(init_scale=1024)

            os.makedirs(self.args.output_dir, exist_ok=True)
            config_path = os.path.join(self.args.output_dir, "config.json")
            eval_interval = 2
            best_score = float("-inf")
            best_epoch = -1
            best_metrics = None

            for ep in range(self.args.max_epoch):
                train_stats = self.supervised_unlearn_train_cliperase(
                    model=model,
                    teacher=teacher,
                    df_train_loader=df_loader,
                    dr_train_loader=dr_loader,
                    tokenizer_fn=tokenize_fn,
                    class_names=class_names,
                    forget_indices=forget_indices,
                    retain_indices=retain_indices,
                    optimizer=optimizer,
                    scaler=scaler,
                    lambda_df=self.args.lambda_df,
                    lambda_dr=self.args.lambda_dr,
                    lambda_uni=self.args.lambda_uni,
                    epoch_idx=ep,
                    max_epoch=self.args.max_epoch,
                    log_interval=self.args.log_interval,
                )

                should_eval = ((ep + 1) % eval_interval == 0) or ((ep + 1) == self.args.max_epoch)
                if not should_eval:
                    continue

                eval_metrics = self._evaluate_for_selection(
                    model=model,
                    tokenize_fn=tokenize_fn,
                    image_size=image_size,
                    class_names=class_names,
                )
                cur_score = eval_metrics["selection_score"]
                print(
                    f"[Eval EP {ep + 1}/{self.args.max_epoch}] "
                    f"forget_success={eval_metrics['forget_success']:.4f} "
                    f"retain_acc={eval_metrics['retain_accuracy']:.4f} "
                    f"score={cur_score:.4f}"
                )

                if cur_score > best_score:
                    best_score = cur_score
                    best_epoch = ep + 1
                    best_metrics = eval_metrics

                    model.clip_model.save_pretrained(self.args.output_dir)
                    save_cfg = dict(vars(self.args))
                    save_cfg.update(
                        {
                            "best_epoch": best_epoch,
                            "best_score": best_score,
                            "best_forget_success": eval_metrics["forget_success"],
                            "best_retain_accuracy": eval_metrics["retain_accuracy"],
                            "selection_metric": "forget_success + retain_accuracy",
                            "eval_interval_epoch": eval_interval,
                            "train_stats_at_best_epoch": train_stats,
                        }
                    )
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(save_cfg, f, ensure_ascii=False, indent=2)
                    print(f"[Best Updated] epoch={best_epoch} score={best_score:.4f} -> overwrite checkpoint/config")

            if best_epoch < 0:
                raise RuntimeError("No evaluation executed, no best checkpoint saved.")

            model.clip_model = CLIPModel.from_pretrained(self.args.output_dir)
            print(
                f"Final best epoch: {best_epoch}, "
                f"forget_success={best_metrics['forget_success']:.4f}, "
                f"retain_acc={best_metrics['retain_accuracy']:.4f}, "
                f"score={best_score:.4f}"
            )

            final_eval_metrics = self.run_original_eval(
                model=model,
                tokenize_fn=tokenize_fn,
                image_size=image_size,
                backend=backend,
                retain_topk_indices=retain_topk_indices,
            )
            final_cfg = dict(vars(self.args))
            final_cfg.update(
                {
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "best_forget_success": best_metrics["forget_success"],
                    "best_retain_accuracy": best_metrics["retain_accuracy"],
                    "selection_metric": "forget_success + retain_accuracy",
                    "eval_interval_epoch": eval_interval,
                    "final_original_eval": final_eval_metrics,
                }
            )
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(final_cfg, f, ensure_ascii=False, indent=2)

    def supervised_unlearn_train_cliperase(
        self,
        model,
        teacher,
        df_train_loader,
        dr_train_loader,
        tokenizer_fn: Callable[[Sequence[str]], torch.Tensor],
        class_names: Sequence[str],
        forget_indices: Sequence[int],
        retain_indices: Sequence[int],
        optimizer,
        scaler,
        lambda_df: float = 1.0,
        lambda_dr: float = 1.0,
        lambda_uni: float = 1.0,
        epoch_idx: int = 0,
        max_epoch: int = 1,
        log_interval: int = 50,
    ) -> Dict[str, float]:
        device = self._get_model_device(model)
        iters_per_epoch = min(len(df_train_loader), len(dr_train_loader))

        def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            denom = mask.sum().clamp_min(1.0)
            return (x * mask).sum() / denom

        model.train()
        df_iter = iter(df_train_loader)
        dr_iter = iter(dr_train_loader)
        running = {"forget": 0.0, "retain": 0.0, "kl": 0.0, "tot": 0.0}

        for it in range(iters_per_epoch):
            df_s = next(df_iter)
            dr_s = next(dr_iter)

            img_df = df_s["image"].to(device, non_blocking=True)
            img_dr = dr_s["image"].to(device, non_blocking=True)

            txt_df = self._labels_to_texts(df_s["label"], class_names, forget_indices)
            txt_dr = self._labels_to_texts(dr_s["label"], class_names, retain_indices)
            txt_df = tokenizer_fn(txt_df).to(device)
            txt_dr = tokenizer_fn(txt_dr).to(device)

            img_all = torch.cat([img_df, img_dr], dim=0)
            txt_all = torch.cat([txt_df, txt_dr], dim=0)

            batch_forget = img_df.size(0)
            batch_retain = img_dr.size(0)
            batch_total = batch_forget + batch_retain

            flags = torch.cat(
                [
                    torch.ones(batch_forget, dtype=torch.long, device=device),
                    torch.zeros(batch_retain, dtype=torch.long, device=device),
                ],
                dim=0,
            )
            forget_mask = flags.float()
            retain_mask = (1 - flags).float()
            targets = torch.arange(batch_total, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                sim_i2t_u, sim_t2i_u, _, _ = self._get_logits_and_feats(model, img_all, txt_all)
                with torch.no_grad():
                    sim_i2t_t, sim_t2i_t, _, _ = self._get_logits_and_feats(teacher, img_all, txt_all)

                ce_img = F.cross_entropy(sim_i2t_u, targets, reduction="none")
                ce_txt = F.cross_entropy(sim_t2i_u, targets, reduction="none")

                forget_img_loss = masked_mean(ce_img, forget_mask)
                forget_txt_loss = masked_mean(ce_txt, forget_mask)
                loss_forget = -(forget_img_loss + forget_txt_loss)

                retain_img_loss = masked_mean(ce_img, retain_mask)
                retain_txt_loss = masked_mean(ce_txt, retain_mask)
                loss_retain = retain_img_loss + retain_txt_loss

                log_p_img = F.log_softmax(sim_i2t_u, dim=-1)
                p_img_t = F.softmax(sim_i2t_t, dim=-1)
                log_p_txt = F.log_softmax(sim_t2i_u, dim=-1)
                p_txt_t = F.softmax(sim_t2i_t, dim=-1)

                kl_img_all = F.kl_div(log_p_img, p_img_t, reduction="none").sum(dim=-1)
                kl_txt_all = F.kl_div(log_p_txt, p_txt_t, reduction="none").sum(dim=-1)
                kl_img = masked_mean(kl_img_all, retain_mask)
                kl_txt = masked_mean(kl_txt_all, retain_mask)
                loss_kl = kl_img + kl_txt

                loss = lambda_df * loss_forget + lambda_dr * loss_retain + lambda_uni * loss_kl

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running["forget"] += float(loss_forget.detach().item())
            running["retain"] += float(loss_retain.detach().item())
            running["kl"] += float(loss_kl.detach().item())
            running["tot"] += float(loss.detach().item())

            if (it + 1) % log_interval == 0:
                t = it + 1
                print(
                    f"[ClipErase EP {epoch_idx+1}/{max_epoch}] it={t}/{iters_per_epoch} "
                    f"forget={running['forget']/t:.4f} "
                    f"retain={running['retain']/t:.4f} "
                    f"kl={running['kl']/t:.4f} "
                    f"total={running['tot']/t:.4f}"
                )

        denom = float(max(iters_per_epoch, 1))
        return {
            "loss_forget": running["forget"] / denom,
            "loss_retain": running["retain"] / denom,
            "loss_kl": running["kl"] / denom,
            "loss_total": running["tot"] / denom,
        }

    
import copy
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Support direct script execution:
# python finegrained/baselines/clip_unlearn_cliperase.py
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from finegrained.clip_finegrained_baseline import ClipFinegrainedBaseline
from finegrained.params import build_parser as build_base_parser, resolve_dataset_paths


def build_parser():
    parser = build_base_parser()
    parser.description = "Finegrained CLIP unlearning using the ClipErase objective."
    parser.set_defaults(method="cliperase")
    parser.add_argument(
        "--cliperase_eval_interval",
        type=int,
        default=2,
        help="Evaluate selection metrics every N epochs.",
    )
    return parser


class ClipEraseRunner(ClipFinegrainedBaseline):
    def __init__(self, args):
        super().__init__(args)

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
        amp_enabled = device.type == "cuda"
        iters_per_epoch = min(len(df_train_loader), len(dr_train_loader))

        model.train()
        teacher.eval()
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
            txt_df = tokenizer_fn(txt_df).to(device, non_blocking=True)
            txt_dr = tokenizer_fn(txt_dr).to(device, non_blocking=True)

            targets_df = torch.arange(img_df.size(0), device=device)
            targets_dr = torch.arange(img_dr.size(0), device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                sim_i2t_df, sim_t2i_df, _, _ = self._get_logits_and_feats(model, img_df, txt_df)
                loss_df_i2t = F.cross_entropy(sim_i2t_df, targets_df)
                loss_df_t2i = F.cross_entropy(sim_t2i_df, targets_df)
                loss_forget = -(loss_df_i2t + loss_df_t2i)

                sim_i2t_dr, sim_t2i_dr, _, _ = self._get_logits_and_feats(model, img_dr, txt_dr)
                loss_dr_i2t = F.cross_entropy(sim_i2t_dr, targets_dr)
                loss_dr_t2i = F.cross_entropy(sim_t2i_dr, targets_dr)
                loss_retain = loss_dr_i2t + loss_dr_t2i

                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        sim_i2t_teacher, sim_t2i_teacher, _, _ = self._get_logits_and_feats(teacher, img_dr, txt_dr)

                log_p_img = F.log_softmax(sim_i2t_dr, dim=-1)
                p_img_t = F.softmax(sim_i2t_teacher, dim=-1)
                log_p_txt = F.log_softmax(sim_t2i_dr, dim=-1)
                p_txt_t = F.softmax(sim_t2i_teacher, dim=-1)

                kl_img = F.kl_div(log_p_img, p_img_t, reduction="batchmean")
                kl_txt = F.kl_div(log_p_txt, p_txt_t, reduction="batchmean")
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
                step = it + 1
                print(
                    f"[ClipErase EP {epoch_idx + 1}/{max_epoch}] it={step}/{iters_per_epoch} "
                    f"forget={running['forget'] / step:.4f} "
                    f"retain={running['retain'] / step:.4f} "
                    f"kl={running['kl'] / step:.4f} "
                    f"total={running['tot'] / step:.4f}"
                )

        denom = float(max(iters_per_epoch, 1))
        return {
            "loss_forget": running["forget"] / denom,
            "loss_retain": running["retain"] / denom,
            "loss_kl": running["kl"] / denom,
            "loss_total": running["tot"] / denom,
        }

    def run(self) -> None:
        device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        model, tokenize_fn, image_size, backend = self._load_clip_backend(device)
        model = model.float().to(device)

        teacher = copy.deepcopy(model).eval()
        if device.type == "cuda":
            teacher = teacher.half()
        teacher = teacher.to(device)
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
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda", init_scale=1024)

        os.makedirs(self.args.output_dir, exist_ok=True)
        eval_interval = max(1, int(self.args.cliperase_eval_interval))
        best_score = float("-inf")
        best_epoch = -1
        best_metrics = None
        best_epoch_info = None

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
            cur_score = float(eval_metrics["selection_score"])
            print(
                f"[Eval EP {ep + 1}/{self.args.max_epoch}] "
                f"forget_success={eval_metrics['forget_success']:.4f} "
                f"retain_acc={eval_metrics['retain_accuracy']:.4f} "
                f"score={cur_score:.4f}"
            )

            if cur_score > best_score:
                best_score = cur_score
                best_epoch = ep + 1
                best_metrics = dict(eval_metrics)

                model.clip_model.save_pretrained(self.args.output_dir)
                best_epoch_info = {
                    "best_epoch": best_epoch,
                    "forget_success": eval_metrics["forget_success"],
                    "retain_accuracy": eval_metrics["retain_accuracy"],
                }
                self._write_best_epoch_info(best_epoch_info)
                print(f"[Best Updated] epoch={best_epoch} score={best_score:.4f}")

        if best_epoch < 0:
            raise RuntimeError("No evaluation executed, no best checkpoint saved.")

        best_model, best_tokenize_fn, best_image_size = self._load_model_from_pretrained_dir(self.args.output_dir)
        print(
            f"Final best epoch: {best_epoch}, "
            f"forget_success={best_metrics['forget_success']:.4f}, "
            f"retain_acc={best_metrics['retain_accuracy']:.4f}, "
            f"score={best_score:.4f}"
        )

        final_eval_metrics = self.run_original_eval(
            model=best_model,
            tokenize_fn=best_tokenize_fn,
            image_size=best_image_size,
            retain_topk_indices=retain_topk_indices,
        )
        self._merge_final_eval_into_best_epoch(final_eval_metrics, base_info=best_epoch_info)

    def run_cliperase(self) -> None:
        self.run()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = resolve_dataset_paths(args)
    runner = ClipEraseRunner(args)
    runner.run()


if __name__ == "__main__":
    main()

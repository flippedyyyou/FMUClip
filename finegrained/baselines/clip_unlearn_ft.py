import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Support direct script execution:
# python finegrained/baselines/clip_unlearn_ft.py
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from finegrained.clip_finegrained_baseline import ClipFinegrainedBaseline
from finegrained.params import build_parser as build_base_parser, resolve_dataset_paths


def build_parser():
    parser = build_base_parser()
    parser.description = "Finegrained CLIP unlearning using SLUG-style FT training."
    parser.set_defaults(method="ft")
    parser.add_argument(
        "--ft_eval_interval",
        type=int,
        default=1,
        help="Evaluate selection metrics every N epochs.",
    )
    parser.add_argument(
        "--ft_grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm; <=0 disables clipping.",
    )
    return parser


class ClipFinegrainedFTRunner(ClipFinegrainedBaseline):
    def __init__(self, args):
        super().__init__(args)

    def _build_ft_train_loader(self, transform):
        df_dataset, dr_dataset = self._build_train_datasets(transform=transform)
        forget_indices, retain_indices = self._build_forget_retain_indices(self.args.forget_classes)
        train_loader = DataLoader(
            dr_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )
        return train_loader, df_dataset, dr_dataset, forget_indices, retain_indices

    def train_one_epoch_ft(
        self,
        model,
        train_loader: DataLoader,
        tokenizer_fn: Callable[[Sequence[str]], torch.Tensor],
        class_names: Sequence[str],
        retain_indices: Sequence[int],
        optimizer,
        scaler,
        epoch_idx: int,
        max_epoch: int,
        log_interval: int,
    ) -> Dict[str, float]:
        device = self._get_model_device(model)
        amp_enabled = device.type == "cuda"
        grad_clip_norm = float(getattr(self.args, "ft_grad_clip_norm", 0.0))

        model.train()
        running = {"clip": 0.0}

        for it, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            text_list = self._labels_to_texts(labels, class_names, retain_indices)
            text_tokens = tokenizer_fn(text_list).to(device, non_blocking=True)
            targets = torch.arange(images.size(0), device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                sim_i2t, sim_t2i, _, _ = self._get_logits_and_feats(model, images, text_tokens)
                loss_i2t = F.cross_entropy(sim_i2t, targets)
                loss_t2i = F.cross_entropy(sim_t2i, targets)
                loss = 0.5 * (loss_i2t + loss_t2i)

            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))

            running["clip"] += float(loss.detach().item())

            if (it + 1) % log_interval == 0:
                step = it + 1
                print(
                    f"[FT EP {epoch_idx + 1}/{max_epoch}] it={step}/{len(train_loader)} "
                    f"clip_loss={running['clip'] / step:.4f}"
                )

        denom = float(max(len(train_loader), 1))
        return {"loss_clip": running["clip"] / denom}

    def run(self) -> None:
        device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        model, tokenize_fn, image_size, backend = self._load_clip_backend(device)
        model = model.float().to(device)

        train_transform = self._build_eval_transform(image_size)
        train_loader, df_dataset, dr_dataset, forget_indices, retain_indices = self._build_ft_train_loader(
            transform=train_transform
        )
        class_names = self._build_class_name_list()
        retain_topk_indices = self._compute_topk_retain_classes(
            df_dataset=df_dataset,
            forget_indices=forget_indices,
            k=self.args.retain_topk,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda", init_scale=1024)

        os.makedirs(self.args.output_dir, exist_ok=True)
        config_path = os.path.join(self.args.output_dir, "config.json")
        ckpt_path = os.path.join(self.args.output_dir, "clip_unlearn_ft.pth")
        eval_interval = max(1, int(self.args.ft_eval_interval))
        best_score = float("-inf")
        best_epoch = -1
        best_metrics = None

        for ep in range(self.args.max_epoch):
            train_stats = self.train_one_epoch_ft(
                model=model,
                train_loader=train_loader,
                tokenizer_fn=tokenize_fn,
                class_names=class_names,
                retain_indices=retain_indices,
                optimizer=optimizer,
                scaler=scaler,
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
                torch.save(
                    {
                        "model": copy.deepcopy(model).state_dict(),
                        "clip_arch": self.args.clip_arch,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "best_forget_success": eval_metrics["forget_success"],
                        "best_retain_accuracy": eval_metrics["retain_accuracy"],
                    },
                    ckpt_path,
                )
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
                        "train_set_construction": {
                            "forget_source": {
                                "split": self.args.train_split,
                                "item_folder": self.args.train_item_folder,
                                "classes": list(self.args.forget_classes),
                                "used_for_training": False,
                            },
                            "retain_source": {
                                "split": self.args.train_split,
                                "item_folder": self.args.retain_item_folder,
                                "classes": [
                                    name for idx, name in enumerate(class_names) if idx not in set(forget_indices)
                                ],
                                "used_for_training": True,
                            },
                            "train_dataset_size": len(dr_dataset),
                            "forget_dataset_size": len(df_dataset),
                            "training_strategy": "retain_only",
                        },
                    }
                )
                with open(config_path, "w", encoding="utf-8") as handle:
                    json.dump(save_cfg, handle, ensure_ascii=False, indent=2)
                print(f"[Best Updated] epoch={best_epoch} score={best_score:.4f}")

        if best_epoch < 0:
            raise RuntimeError("No epoch completed successfully, no best checkpoint saved.")

        best_ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model"], strict=True)
        print(
            f"Final best epoch: {best_epoch}, "
            f"forget_success={best_metrics['forget_success']:.4f}, "
            f"retain_acc={best_metrics['retain_accuracy']:.4f}, "
            f"score={best_score:.4f}"
        )

        self.run_original_eval(
            model=model,
            tokenize_fn=tokenize_fn,
            image_size=image_size,
            backend=backend,
            retain_topk_indices=retain_topk_indices,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = resolve_dataset_paths(args)
    runner = ClipFinegrainedFTRunner(args)
    runner.run()


if __name__ == "__main__":
    main()

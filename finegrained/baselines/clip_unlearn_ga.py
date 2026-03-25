import copy
import json
import os
import sys
from typing import Callable, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINEGRAINED_DIR = os.path.dirname(_CURRENT_DIR)
_PROJECT_ROOT = os.path.dirname(_FINEGRAINED_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from finegrained.clip_finegrained_baseline import ClipFinegrainedBaseline
from finegrained.params import build_parser, resolve_dataset_paths


def build_ga_parser():
    parser = build_parser()
    parser.set_defaults(method="ga")
    parser.add_argument(
        "--ga_eval_interval",
        type=int,
        default=1,
        help="Evaluate selection metrics every N epochs.",
    )
    parser.add_argument(
        "--ga_use_ce",
        action="store_true",
        help="Use contrastive CE loss on forget pairs instead of cosine loss.",
    )
    return parser


class ClipFinegrainedGARunner(ClipFinegrainedBaseline):
    def __init__(self, args):
        super().__init__(args)

    def _build_ga_train_loader(self, transform) -> DataLoader:
        df_dataset, _ = self._build_train_datasets(transform=transform)
        return DataLoader(
            df_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )

    def train_one_epoch_ga(
        self,
        model,
        train_loader: DataLoader,
        tokenizer_fn: Callable[[Sequence[str]], torch.Tensor],
        class_names: Sequence[str],
        forget_indices: Sequence[int],
        optimizer,
        scaler,
        epoch_idx: int = 0,
        max_epoch: int = 1,
        log_interval: int = 50,
    ) -> Dict[str, float]:
        device = self._get_model_device(model)
        amp_enabled = device.type == "cuda"

        model.train()
        running = {"forget": 0.0, "objective": 0.0}

        for it, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            texts = self._labels_to_texts(batch["label"], class_names, forget_indices)
            text_tokens = tokenizer_fn(texts).to(device, non_blocking=True)
            targets = torch.arange(images.size(0), device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                sim_i2t, sim_t2i, image_features, text_features = self._get_logits_and_feats(
                    model,
                    images,
                    text_tokens,
                )
                if self.args.ga_use_ce:
                    forget_loss = 0.5 * (
                        F.cross_entropy(sim_i2t, targets) + F.cross_entropy(sim_t2i, targets)
                    )
                else:
                    forget_loss = 1.0 - F.cosine_similarity(
                        image_features,
                        text_features,
                        dim=1,
                    ).mean()
                objective = -forget_loss

            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()

            running["forget"] += float(forget_loss.detach().item())
            running["objective"] += float(objective.detach().item())

            if (it + 1) % log_interval == 0:
                step = it + 1
                print(
                    f"[GA EP {epoch_idx + 1}/{max_epoch}] it={step}/{len(train_loader)} "
                    f"forget_loss={running['forget'] / step:.4f} "
                    f"objective={running['objective'] / step:.4f}"
                )

        denom = float(max(len(train_loader), 1))
        return {
            "loss_forget": running["forget"] / denom,
            "loss_objective": running["objective"] / denom,
        }

    def _evaluate_for_selection(
        self,
        model: torch.nn.Module,
        tokenize_fn,
        image_size: int,
        class_names: Sequence[str],
        retain_topk_indices: Sequence[int],
    ) -> Dict[str, float]:
        device = self._get_model_device(model)
        eval_transform = self._build_eval_transform(image_size)
        forget_loader, retain_loader = self._build_test_dataloaders_from_folders(transform=eval_transform)

        text_features = self._encode_text_features(model, class_names, tokenize_fn, device)
        forget_cache = self._collect_eval_cache(model, forget_loader, text_features, device)
        retain_cache = self._collect_eval_cache(model, retain_loader, text_features, device)
        forget_acc = self._single_accuracy_from_cache(forget_cache)
        retain_acc = self._single_accuracy_from_cache(retain_cache)
        retain_topk_class_acc = self._topk_retain_accuracy_from_cache(
            cache=retain_cache,
            class_names=class_names,
            retain_indices=retain_topk_indices,
        )
        retain_topk_acc = float(np.mean(list(retain_topk_class_acc.values()))) if retain_topk_class_acc else 0.0
        score = (1.0 - forget_acc) + retain_topk_acc + retain_acc
        return {
            "forget_success": 1.0 - float(forget_acc),
            "retain_accuracy": float(retain_acc),
            "retain_topk_accuracy": float(retain_topk_acc),
            "retain_topk_class_accuracy": retain_topk_class_acc,
            "selection_score": float(score),
            "forget_test_size": len(forget_loader.dataset),
            "retain_test_size": len(retain_loader.dataset),
        }

    def run_ga(self) -> None:
        device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        model, tokenize_fn, image_size, backend = self._load_clip_backend(device)
        model = model.float().to(device)

        train_transform = self._build_eval_transform(image_size)
        df_dataset, _ = self._build_train_datasets(transform=train_transform)
        train_loader = DataLoader(
            df_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )

        class_names = self._build_class_name_list()
        forget_indices, _ = self._build_forget_retain_indices(self.args.forget_classes)
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
        ckpt_path = os.path.join(self.args.output_dir, "clip_finegrained_ga.pth")
        eval_interval = max(1, int(self.args.ga_eval_interval))
        best_score = float("-inf")
        best_epoch = -1
        best_metrics = None

        for ep in range(self.args.max_epoch):
            train_stats = self.train_one_epoch_ga(
                model=model,
                train_loader=train_loader,
                tokenizer_fn=tokenize_fn,
                class_names=class_names,
                forget_indices=forget_indices,
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
                retain_topk_indices=retain_topk_indices,
            )
            print(eval_metrics)
            with open(f'{self.args.output_dir}/eval_log.jsonl', 'a', encoding='utf-8') as fw:
                    fw.write(json.dumps(
                        {
                            'epoch': ep + 1,
                            'metric': eval_metrics
                        }
                    ) + '\n')
            score = eval_metrics['forget_success'] + eval_metrics['retain_accuracy']
            if score > best_score:
                print(f'New Best Epoch: {ep + 1}')
                best_score = score
                model.clip_model.save_pretrained(self.args.output_dir)
                best_epoch = ep + 1
                best_metrics = dict(eval_metrics)
                json.dump({
                        'best_epoch': ep + 1,
                        'forget_success': eval_metrics['forget_success'],
                        'retain_accuracy': eval_metrics['retain_accuracy']
                    },
                        open(f'{self.args.output_dir}/best_epoch.json', 'w', encoding='utf-8'),
                        ensure_ascii=False,
                        indent=2)

        best_ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model"], strict=True)
        print(
            f"Final best epoch: {best_epoch}, "
            f"forget_success={best_metrics['forget_success']:.4f}, "
            f"retain_acc={best_metrics['retain_accuracy']:.4f}, "
            f"retain_topk_acc={best_metrics['retain_topk_accuracy']:.4f}, "
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
                "best_retain_topk_accuracy": best_metrics["retain_topk_accuracy"],
                "selection_metric": "forget_success + retain_topk_accuracy + retain_accuracy",
                "eval_interval_epoch": eval_interval,
                "final_original_eval": final_eval_metrics,
            }
        )
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(final_cfg, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = build_ga_parser().parse_args()
    args = resolve_dataset_paths(args)
    runner = ClipFinegrainedGARunner(args)
    runner.run_ga()


if __name__ == "__main__":
    main()

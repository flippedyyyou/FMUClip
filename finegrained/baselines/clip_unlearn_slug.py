import copy
import json
import math
import os
import sys
from typing import Callable, Dict, List, Sequence, Tuple

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


def _parse_float_list(raw: str) -> List[float]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [float(item) for item in values if item]
    if not parsed:
        raise ValueError("Expected at least one float value.")
    return parsed


def build_slug_parser():
    parser = build_parser()
    parser.set_defaults(method="slug")
    parser.add_argument(
        "--slug_parts",
        choices=["vision", "language", "all"],
        default="all",
        help="Which parameter groups are eligible for SLUG layer selection.",
    )
    parser.add_argument(
        "--slug_ratio_divisor",
        type=float,
        default=10.0,
        help="Base divisor used to turn ||w|| / ||g_f|| into the initial update scale.",
    )
    parser.add_argument(
        "--slug_search_multipliers",
        type=str,
        default="0.5,1.0,2.0,4.0",
        help="Comma separated multipliers applied to the base SLUG step size.",
    )
    parser.add_argument(
        "--slug_max_candidates_per_part",
        type=int,
        default=3,
        help="Maximum Pareto-front layers to evaluate per part.",
    )
    parser.add_argument(
        "--slug_eval_topk",
        type=int,
        default=5,
        help="Top-k saved in the final evaluation jsonl outputs.",
    )
    return parser


class ClipFinegrainedSlugRunner(ClipFinegrainedBaseline):
    def __init__(self, args):
        super().__init__(args)
        self.search_multipliers = _parse_float_list(args.slug_search_multipliers)

    def _build_slug_train_dataloaders(
        self,
        transform,
    ) -> Tuple[DataLoader, DataLoader, object]:
        df_dataset, dr_dataset = self._build_train_datasets(transform=transform)
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
        return df_loader, dr_loader, df_dataset

    def _compute_average_gradients(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        tokenizer_fn: Callable[[Sequence[str]], torch.Tensor],
        class_names: Sequence[str],
        prefer_indices: Sequence[int],
        split: str,
    ) -> Dict[str, torch.Tensor]:
        device = self._get_model_device(model)
        grad_sums = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        num_batches = 0
        amp_enabled = device.type == "cuda"

        model.train()
        for batch in data_loader:
            images = batch["image"].to(device, non_blocking=True)
            texts = self._labels_to_texts(batch["label"], class_names, prefer_indices)
            text_tokens = tokenizer_fn(texts).to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                sim_i2t, sim_t2i, image_features, text_features = self._get_logits_and_feats(
                    model, images, text_tokens
                )
                if split == "forget":
                    loss = 1.0 - F.cosine_similarity(image_features, text_features, dim=1).mean()
                else:
                    targets = torch.arange(images.size(0), device=device)
                    loss = 0.5 * (
                        F.cross_entropy(sim_i2t, targets) + F.cross_entropy(sim_t2i, targets)
                    )

            loss.backward()
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                grad_sums[name] += param.grad.detach()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError(f"No batches were available for `{split}` gradient computation.")

        scale = 1.0 / float(num_batches)
        return {name: grad * scale for name, grad in grad_sums.items()}

    def _identify_pareto_indices(self, points: Sequence[Tuple[float, float]]) -> List[int]:
        pareto_indices: List[int] = []
        for idx, (align_i, ratio_i) in enumerate(points):
            dominated = False
            for jdx, (align_j, ratio_j) in enumerate(points):
                if jdx == idx:
                    continue
                if align_j < align_i and ratio_j > ratio_i:
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(idx)
        return pareto_indices

    def _iter_slug_parts(self) -> List[str]:
        if self.args.slug_parts == "all":
            return ["vision", "language"]
        return [self.args.slug_parts]

    def _is_candidate_layer(self, name: str, part: str) -> bool:
        if any(token in name for token in ("bias", "logit_scale", "position", "embedding")):
            return False
        if part == "vision":
            return "visual" in name
        if part == "language":
            return "visual" not in name
        return True

    def _rank_slug_layers(
        self,
        model: torch.nn.Module,
        forget_grads: Dict[str, torch.Tensor],
        retain_grads: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
        eps = 1e-12
        layer_stats: Dict[str, Dict[str, float]] = {}
        selected: Dict[str, List[str]] = {}

        for part in self._iter_slug_parts():
            part_names: List[str] = []
            part_points: List[Tuple[float, float]] = []
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in forget_grads or name not in retain_grads:
                    continue
                if forget_grads[name].ndim == 0 or not self._is_candidate_layer(name, part):
                    continue

                param_norm = torch.norm(param.detach()).item()
                forget_norm = torch.norm(forget_grads[name]).item()
                retain_norm = torch.norm(retain_grads[name]).item()
                if param_norm <= eps or forget_norm <= eps or retain_norm <= eps:
                    continue

                alignment = F.cosine_similarity(
                    forget_grads[name].flatten().unsqueeze(0),
                    retain_grads[name].flatten().unsqueeze(0),
                    dim=1,
                ).abs().item()
                importance = forget_norm / max(param_norm, eps)
                layer_stats[name] = {
                    "part": part,
                    "alignment": float(alignment),
                    "importance": float(importance),
                    "param_norm": float(param_norm),
                    "forget_grad_norm": float(forget_norm),
                    "retain_grad_norm": float(retain_norm),
                }
                part_names.append(name)
                part_points.append((float(alignment), float(importance)))

            pareto_idx = set(self._identify_pareto_indices(part_points))
            part_layers = [part_names[idx] for idx in range(len(part_names)) if idx in pareto_idx]
            part_layers = sorted(
                part_layers,
                key=lambda key: (
                    layer_stats[key]["importance"],
                    -layer_stats[key]["alignment"],
                ),
                reverse=True,
            )
            selected[part] = part_layers[: max(1, int(self.args.slug_max_candidates_per_part))]
            for name in part_names:
                layer_stats[name]["pareto"] = bool(name in part_layers)
        return selected, layer_stats

    def _apply_single_layer_update(
        self,
        model: torch.nn.Module,
        layer_name: str,
        update_scale: float,
        forget_grads: Dict[str, torch.Tensor],
    ) -> torch.nn.Module:
        candidate = copy.deepcopy(model)
        with torch.no_grad():
            layer = candidate.get_parameter(layer_name)
            update = forget_grads[layer_name].to(device=layer.device, dtype=layer.dtype)
            layer.add_(update_scale * update)
        return candidate

    def run_slug(self) -> None:
        device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        model, tokenize_fn, image_size, backend = self._load_clip_backend(device)
        model = model.float().to(device)

        train_transform = self._build_eval_transform(image_size)
        df_loader, dr_loader, df_dataset = self._build_slug_train_dataloaders(transform=train_transform)

        class_names = self._build_class_name_list()
        forget_indices, retain_indices = self._build_forget_retain_indices(self.args.forget_classes)
        retain_topk_indices = self._compute_topk_retain_classes(
            df_dataset=df_dataset,
            forget_indices=forget_indices,
            k=self.args.retain_topk,
        )

        os.makedirs(self.args.output_dir, exist_ok=True)

        forget_grads = self._compute_average_gradients(
            model=model,
            data_loader=df_loader,
            tokenizer_fn=tokenize_fn,
            class_names=class_names,
            prefer_indices=forget_indices,
            split="forget",
        )
        retain_grads = self._compute_average_gradients(
            model=model,
            data_loader=dr_loader,
            tokenizer_fn=tokenize_fn,
            class_names=class_names,
            prefer_indices=retain_indices,
            split="retain",
        )

        selected_layers, layer_stats = self._rank_slug_layers(
            model=model,
            forget_grads=forget_grads,
            retain_grads=retain_grads,
        )

        base_selection = self._evaluate_for_selection(
            model=model,
            tokenize_fn=tokenize_fn,
            image_size=image_size,
            class_names=class_names,
        )
        search_records: List[Dict[str, object]] = [
            {
                "layer_name": None,
                "part": "baseline",
                "update_scale": 0.0,
                **base_selection,
            }
        ]

        best_model = copy.deepcopy(model)
        best_metrics = dict(base_selection)
        best_score = float(base_selection["selection_score"])
        best_layer = None
        best_scale = 0.0

        for part, layer_names in selected_layers.items():
            for layer_name in layer_names:
                param_norm = layer_stats[layer_name]["param_norm"]
                forget_grad_norm = layer_stats[layer_name]["forget_grad_norm"]
                if forget_grad_norm <= 0.0:
                    continue
                base_scale = -(param_norm / forget_grad_norm) / float(self.args.slug_ratio_divisor)

                for multiplier in self.search_multipliers:
                    update_scale = base_scale * multiplier
                    candidate = self._apply_single_layer_update(
                        model=model,
                        layer_name=layer_name,
                        update_scale=update_scale,
                        forget_grads=forget_grads,
                    )
                    metrics = self._evaluate_for_selection(
                        model=candidate,
                        tokenize_fn=tokenize_fn,
                        image_size=image_size,
                        class_names=class_names,
                    )
                    record = {
                        "layer_name": layer_name,
                        "part": part,
                        "update_scale": float(update_scale),
                        "multiplier": float(multiplier),
                        "alignment": layer_stats[layer_name]["alignment"],
                        "importance": layer_stats[layer_name]["importance"],
                        **metrics,
                    }
                    search_records.append(record)

                    if metrics["selection_score"] > best_score:
                        best_score = float(metrics["selection_score"])
                        best_metrics = dict(metrics)
                        best_model = candidate
                        best_layer = layer_name
                        best_scale = float(update_scale)

        ckpt_path = os.path.join(self.args.output_dir, "clip_finegrained_slug.pth")
        torch.save(
            {
                "model": best_model.state_dict(),
                "clip_arch": self.args.clip_arch,
                "best_layer": best_layer,
                "best_scale": best_scale,
                "best_selection_score": best_score,
                "best_selection_metrics": best_metrics,
            },
            ckpt_path,
        )

        layer_stats_path = os.path.join(self.args.output_dir, "slug_layer_stats.json")
        with open(layer_stats_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "selected_layers": selected_layers,
                    "layer_stats": layer_stats,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        search_path = os.path.join(self.args.output_dir, "slug_search_results.json")
        with open(search_path, "w", encoding="utf-8") as handle:
            json.dump(search_records, handle, ensure_ascii=False, indent=2)

        final_eval_metrics = self.run_original_eval(
            model=best_model,
            tokenize_fn=tokenize_fn,
            image_size=image_size,
            backend=backend,
            retain_topk_indices=retain_topk_indices,
        )

        config_path = os.path.join(self.args.output_dir, "config.json")
        save_cfg = dict(vars(self.args))
        save_cfg.update(
            {
                "slug_selected_layers": selected_layers,
                "slug_best_layer": best_layer,
                "slug_best_scale": best_scale,
                "slug_best_selection_score": best_score,
                "slug_best_selection_metrics": best_metrics,
                "slug_search_multipliers": self.search_multipliers,
                "final_original_eval": final_eval_metrics,
            }
        )
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(save_cfg, handle, ensure_ascii=False, indent=2)

        print(
            f"SLUG best layer={best_layer}, "
            f"scale={best_scale:.6f}, "
            f"forget_success={best_metrics['forget_success']:.4f}, "
            f"retain_accuracy={best_metrics['retain_accuracy']:.4f}, "
            f"score={best_score:.4f}"
        )


def main() -> None:
    args = build_slug_parser().parse_args()
    args = resolve_dataset_paths(args)
    runner = ClipFinegrainedSlugRunner(args)
    runner.run_slug()


if __name__ == "__main__":
    main()

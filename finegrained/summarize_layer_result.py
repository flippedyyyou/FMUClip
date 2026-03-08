import argparse
import json
import os


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one CSV row for layer sweep summary.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--layer", type=int, required=True, help="Layer index")
    parser.add_argument("--output_dir", type=str, required=True, help="Run output directory")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"{args.layer},,,{args.output_dir}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    final_eval = cfg.get("final_original_eval", {})
    forget_success = _safe_float(cfg.get("best_forget_success", 0.0))
    retain_accuracy = _safe_float(cfg.get("best_retain_accuracy", 0.0))
    retain_topk_accuracy = _safe_float(final_eval.get("retain_topk_accuracy", 0.0))
    # `retain_topk_map` in experiment doc corresponds to `retain_topk_ap` in code.
    retain_topk_map = _safe_float(final_eval.get("retain_topk_ap", 0.0))
    other_map = _safe_float(final_eval.get("other_map", 0.0))

    best_acc_mean = (forget_success + retain_topk_accuracy + retain_accuracy) / 3.0
    best_map_mean = (retain_topk_map + other_map) / 2.0
    print(f"{args.layer},{best_acc_mean:.6f},{best_map_mean:.6f},{args.output_dir}")


if __name__ == "__main__":
    main()

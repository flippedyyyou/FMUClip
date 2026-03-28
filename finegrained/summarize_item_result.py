import argparse
import json
import os


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one CSV row for item sweep summary.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--train_item_folder", type=str, required=True, help="Training item folder")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"{args.train_item_folder},,,")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    final_eval = cfg.get("final_original_eval", {})
    acc_f = _safe_float(cfg.get("best_forget_success", 0.0))
    acc_c = _safe_float(final_eval.get("retain_topk_accuracy", 0.0))
    acc_r = _safe_float(cfg.get("best_retain_accuracy", 0.0))
    print(f"{args.train_item_folder},{acc_f:.6f},{acc_c:.6f},{acc_r:.6f}")


if __name__ == "__main__":
    main()

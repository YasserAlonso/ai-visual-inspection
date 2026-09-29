"""Evaluate a saved detector on the test split (or explicitly selected val split)."""

import argparse
import json
from pathlib import Path

from visual_inspection.training.dataset import load_dataset_yaml
from visual_inspection.training.evaluator import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations"))
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset_yaml(args.dataset)
        if args.split not in dataset.splits:
            raise ValueError(f"Dataset YAML has no {args.split} split")
        args.output.mkdir(parents=True, exist_ok=True)
        metrics, per_class, result = evaluate(
            args.model, args.dataset, split=args.split, device=args.device, output=args.output
        )
        target = Path(result.save_dir) / "metrics.json"
        target.write_text(
            json.dumps(
                {"split": args.split, "metrics": metrics, "per_class_map50_95": per_class}, indent=2
            ),
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")
    print(f"Evaluation split: {args.split}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Per-class mAP50-95: {per_class}")
    print(f"Saved: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

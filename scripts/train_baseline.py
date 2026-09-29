"""Run one reproducible, audited YOLO11n fine-tuning experiment."""

import argparse
from pathlib import Path

from visual_inspection.training.config import load_training_config
from visual_inspection.training.trainer import run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/baseline.yaml"))
    parser.add_argument(
        "--allow-audit-fail",
        action="store_true",
        help="Explicitly override a failing image/annotation audit",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        help="Override configured device; numeric CUDA indexes may be set in YAML",
    )
    args = parser.parse_args(argv)
    try:
        config = load_training_config(args.config)
        if args.device:
            from dataclasses import replace

            config = replace(config, device=args.device)
        experiment = run_training(config, allow_failed_audit=args.allow_audit_fail)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Training error: {exc}\n")
    print(f"Experiment complete: {experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

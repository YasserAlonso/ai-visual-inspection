"""Command-line entry point. Exit codes: 0 PASS/WARNING, 1 FAIL, 2 execution error."""

import argparse
import logging
from pathlib import Path

import yaml

from visual_inspection.audit import audit_dataset
from visual_inspection.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Visual Inspection dataset audit")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit an image dataset")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, default=Path("data/reports"))
    audit.add_argument("--config", type=Path)
    audit.add_argument("--phash-threshold", type=int)
    audit.add_argument("--corrupted-image-failure-threshold", type=float)
    audit.add_argument(
        "--data-yaml", type=Path, help="Also validate YOLO labels using dataset YAML"
    )
    for name in ("train", "validation", "test"):
        audit.add_argument(f"--{name}-name")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(
            args.config,
            phash_distance_threshold=args.phash_threshold,
            corrupted_image_failure_threshold=args.corrupted_image_failure_threshold,
            train_name=args.train_name,
            validation_name=args.validation_name,
            test_name=args.test_name,
        )
        result = audit_dataset(args.dataset, args.output, config, yolo_dataset_yaml=args.data_yaml)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.exit(2, f"Audit error: {exc}\n")
    summary = result.summary
    print("\nDataset Audit Complete")
    print(f"Images analyzed: {summary['total_images']:,} valid / {summary['total_files']:,} files")
    for split, count in summary["images_per_split"].items():
        print(f"{split}: {count:,}")
    for label, key in (
        ("Exact duplicate pairs", "exact_duplicate_pairs"),
        ("Near duplicate pairs", "near_duplicate_pairs"),
        ("Cross-split leakage pairs", "cross_split_leakage_pairs"),
        ("Corrupted images", "corrupted_files"),
    ):
        print(f"{label}: {summary[key]:,}")
    print(f"Dataset Health: {summary['health']}")
    print(f"Report: {result.report_directory / 'report.md'}")
    return 1 if summary["health"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

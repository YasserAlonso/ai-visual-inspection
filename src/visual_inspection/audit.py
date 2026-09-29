"""Pipeline orchestration; library clients receive DataFrames and a summary."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from visual_inspection.config import AuditConfig
from visual_inspection.data.duplicates import exact_duplicates, near_duplicates
from visual_inspection.data.leakage import detect_leakage
from visual_inspection.data.report import build_summary, write_reports
from visual_inspection.data.scanner import scan_dataset
from visual_inspection.data.validation import validate_structure


@dataclass
class AuditResult:
    images: pd.DataFrame
    exact: pd.DataFrame
    near: pd.DataFrame
    leakage: pd.DataFrame
    summary: dict
    report_directory: Path


def audit_dataset(
    dataset: Path,
    output: Path,
    config: AuditConfig | None = None,
    *,
    yolo_dataset_yaml: Path | None = None,
    minimum_box_area: float = 0.0001,
    imbalance_warning_ratio: float = 20.0,
) -> AuditResult:
    config = config or AuditConfig()
    root = dataset.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    if output == root or root in output.parents:
        raise ValueError("Report output must be outside the dataset directory")
    yolo_dataset = None
    if yolo_dataset_yaml is not None:
        from visual_inspection.training.dataset import load_dataset_yaml

        yolo_dataset = load_dataset_yaml(yolo_dataset_yaml)
        if yolo_dataset.root != root:
            raise ValueError("YOLO dataset YAML root must match the audited dataset directory")
    issues = validate_structure(root, config)
    images = scan_dataset(root, config)
    logging.getLogger(__name__).info("Matching exact and perceptual hashes")
    exact = exact_duplicates(images)
    near = near_duplicates(images, config.phash_distance_threshold)
    leakage = detect_leakage(exact, near)
    summary = build_summary(images, exact, near, leakage, issues, config)
    destination = write_reports(output, root, images, exact, near, leakage, summary)
    if yolo_dataset_yaml is not None:
        from visual_inspection.training.dataset import validate_annotations

        annotation_issues, annotation_stats = validate_annotations(
            yolo_dataset, minimum_box_area, imbalance_warning_ratio
        )
        annotation_issues.to_csv(destination / "annotation_issues.csv", index=False)
        blockers = (
            int(
                (
                    ~annotation_issues.issue.astype(str).str.startswith(
                        ("empty_annotation", "extremely_small_box")
                    )
                ).sum()
            )
            if len(annotation_issues)
            else 0
        )
        summary.update(
            annotations=annotation_stats,
            annotation_blocking_issues=blockers,
            annotation_warnings=len(annotation_issues) - blockers,
        )
        if blockers:
            summary["failures"].append(f"{blockers} blocking annotation issues")
        if len(annotation_issues) > blockers:
            summary["warnings"].append(f"{len(annotation_issues) - blockers} annotation warnings")
        if annotation_stats["severe_class_imbalance"]:
            summary["warnings"].append("Severe class imbalance detected")
        if annotation_stats["total_labeled_objects"] == 0:
            summary["failures"].append("No valid labeled objects found")
        summary["health"] = (
            "FAIL" if summary["failures"] else ("WARNING" if summary["warnings"] else "PASS")
        )
        (destination / "summary.json").write_text(
            json.dumps(summary | {"dataset": str(root)}, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        report = destination / "report.md"
        markdown = report.read_text(encoding="utf-8")
        prior_health = build_summary(images, exact, near, leakage, issues, config)["health"]
        markdown = markdown.replace(
            f"**Dataset Health: {prior_health}**", f"**Dataset Health: {summary['health']}**", 1
        )
        report.write_text(
            markdown + f"\n## YOLO annotations\n\n"
            f"- Labeled objects: {annotation_stats['total_labeled_objects']}\n"
            f"- Classes: {annotation_stats['objects_per_class']}\n"
            f"- Annotation issues: {len(annotation_issues)}\n",
            encoding="utf-8",
        )
    return AuditResult(images, exact, near, leakage, summary, destination)

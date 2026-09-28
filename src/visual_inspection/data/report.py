"""Health policy and portable CSV, JSON, and Markdown reports."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from visual_inspection.config import AuditConfig
from visual_inspection.data.statistics import image_statistics


def build_summary(
    images: pd.DataFrame,
    exact: pd.DataFrame,
    near: pd.DataFrame,
    leakage: pd.DataFrame,
    issues: list[str],
    config: AuditConfig,
) -> dict:
    stats = image_statistics(images, config.splits)
    warnings = list(issues)
    failures = []
    corrupt_count = int(images.status.isin(["corrupt", "unreadable", "zero_byte"]).sum())
    corrupt_rate = corrupt_count / len(images) if len(images) else 0.0
    invalid_count = int((images.status != "valid").sum())
    if not stats["total_images"]:
        failures.append("No valid images found")
    if (leakage.leakage_type == "exact_duplicate").any():
        failures.append("Exact cross-split leakage detected")
    if corrupt_rate > config.corrupted_image_failure_threshold:
        failures.append("Corrupted/unreadable/empty file rate exceeds configured threshold")
    if invalid_count:
        warnings.append(f"{invalid_count} invalid files require review")
    if len(near):
        warnings.append("Perceptual matches require manual review (possible false positives)")
    if len(exact):
        warnings.append("Byte-identical images found")
    for split, count in stats["images_per_split"].items():
        if count == 0:
            warnings.append(f"No valid images in split: {split}")
    valid = images[images.status == "valid"]
    if (
        len(valid)
        and (
            (valid.width < config.minimum_dimension_warning)
            | (valid.height < config.minimum_dimension_warning)
            | (valid.aspect_ratio > config.aspect_ratio_warning)
            | (valid.aspect_ratio < 1 / config.aspect_ratio_warning)
        ).any()
    ):
        warnings.append("Unusually small dimensions or extreme aspect ratios detected")
    return dict(
        schema_version="1.0",
        **stats,
        config=asdict(config),
        exact_duplicate_pairs=len(exact),
        exact_duplicate_groups=exact.group_id.nunique(),
        near_duplicate_pairs=len(near),
        near_duplicate_groups=near.group_id.nunique(),
        cross_split_leakage_pairs=len(leakage),
        invalid_files=invalid_count,
        corrupted_files=corrupt_count,
        corrupted_file_rate=corrupt_rate,
        structure_issues=issues,
        warnings=warnings,
        failures=failures,
        health="FAIL" if failures else "WARNING" if warnings else "PASS",
    )


def write_reports(
    output: Path,
    root: Path,
    images: pd.DataFrame,
    exact: pd.DataFrame,
    near: pd.DataFrame,
    leakage: pd.DataFrame,
    summary: dict,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%f")
    destination = output / f"{timestamp}_{uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    summary = dict(summary, dataset=str(root), created_at_utc=timestamp)
    for name, frame in (
        ("images", images),
        ("exact_duplicates", exact),
        ("near_duplicates", near),
        ("cross_split_leakage", leakage),
        ("invalid_images", images[images.status != "valid"]),
    ):
        frame.to_csv(destination / f"{name}.csv", index=False)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Dataset audit",
        "",
        f"**Dataset Health: {summary['health']}**",
        "",
        f"Dataset: `{root}`",
        "",
        f"Files inspected: {summary['total_files']}",
        f"Valid images: {summary['total_images']}",
        "",
        "## Split sizes",
        "",
    ]
    lines.extend(f"- {split}: {count}" for split, count in summary["images_per_split"].items())
    lines += [
        "",
        "## Findings",
        "",
        f"- Corrupted/unreadable/empty files: {summary['corrupted_files']}",
        f"- Invalid files (all types): {summary['invalid_files']}",
        f"- Exact duplicates: {summary['exact_duplicate_pairs']} pairs / "
        f"{summary['exact_duplicate_groups']} groups",
        f"- Near duplicates: {summary['near_duplicate_pairs']} pairs / "
        f"{summary['near_duplicate_groups']} groups",
        f"- Cross-split leakage: {summary['cross_split_leakage_pairs']} pairs",
        "",
        "## Warnings and failures",
        "",
    ]
    lines.extend(f"- {message}" for message in summary["failures"] + summary["warnings"])
    if not summary["failures"] and not summary["warnings"]:
        lines.append("No major issues detected.")
    lines += [
        "",
        "See summary.json for distributions and CSV files for individual findings.",
        "Near groups are connected components; not every pair in a group is similar.",
        "Annotations and class balance are not evaluated in Phase 0.",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return destination

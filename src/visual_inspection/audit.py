"""Pipeline orchestration; library clients receive DataFrames and a summary."""

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


def audit_dataset(dataset: Path, output: Path, config: AuditConfig | None = None) -> AuditResult:
    config = config or AuditConfig()
    root = dataset.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    if output == root or root in output.parents:
        raise ValueError("Report output must be outside the dataset directory")
    issues = validate_structure(root, config)
    images = scan_dataset(root, config)
    logging.getLogger(__name__).info("Matching exact and perceptual hashes")
    exact = exact_duplicates(images)
    near = near_duplicates(images, config.phash_distance_threshold)
    leakage = detect_leakage(exact, near)
    summary = build_summary(images, exact, near, leakage, issues, config)
    destination = write_reports(output, root, images, exact, near, leakage, summary)
    return AuditResult(images, exact, near, leakage, summary, destination)

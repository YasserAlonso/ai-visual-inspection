"""Release checks exercise real CLI processes and inspect every exported artifact."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from visual_inspection.config import AuditConfig
from visual_inspection.audit import audit_dataset
from visual_inspection.data.duplicates import exact_duplicates, near_duplicates
from visual_inspection.data.leakage import detect_leakage
from visual_inspection.data.validation import inspect_image

ROOT = Path(__file__).resolve().parents[1]


def test_near_groups_do_not_invent_leakage_pairs():
    images = pd.DataFrame(
        [
            {"path": "a", "split": "train", "status": "valid", "sha256": "a", "phash": "0"},
            {"path": "b", "split": "val", "status": "valid", "sha256": "b", "phash": "1"},
            {"path": "c", "split": "test", "status": "valid", "sha256": "c", "phash": "3"},
        ]
    )
    near = near_duplicates(images, 1)
    assert near.group_id.nunique() == 1
    leakage = detect_leakage(exact_duplicates(images), near)
    assert set(zip(leakage.source, leakage.target)) == {("a", "b"), ("b", "c")}


def test_dimension_settings_are_applied(dataset, tmp_path, image):
    image.save(dataset / "train/images/a.png")
    normal = audit_dataset(
        dataset,
        tmp_path / "reports",
        AuditConfig(minimum_dimension_warning=64, aspect_ratio_warning=1.5),
    )
    unusual = audit_dataset(
        dataset,
        tmp_path / "reports",
        AuditConfig(minimum_dimension_warning=65, aspect_ratio_warning=1.5),
    )
    assert not any("dimensions" in message for message in normal.summary["warnings"])
    assert any("dimensions" in message for message in unusual.summary["warnings"])


def test_demo_cli_reports(tmp_path):
    dataset = tmp_path / "dataset"
    output = tmp_path / "reports"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/create_demo_dataset.py"), "--output", str(dataset)],
        check=True,
        capture_output=True,
        text=True,
    )
    command = [
        sys.executable,
        "-m",
        "visual_inspection.cli",
        "audit",
        "--dataset",
        str(dataset),
        "--output",
        str(output),
    ]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 1, run.stderr
    assert "Dataset Health: FAIL" in run.stdout
    report = next(output.iterdir())
    assert {p.name for p in report.iterdir()} == {
        "summary.json",
        "images.csv",
        "exact_duplicates.csv",
        "near_duplicates.csv",
        "cross_split_leakage.csv",
        "invalid_images.csv",
        "report.md",
    }
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_files"] == 15
    assert summary["total_images"] == 12
    assert summary["images_per_split"] == {"train": 4, "val": 4, "test": 4}
    assert summary["health"] == "FAIL"
    assert summary["corrupted_file_rate"] == 2 / 15
    images = pd.read_csv(report / "images.csv")
    assert len(images) == 15
    assert (images[images.status == "valid"].width == 96).all()
    exact = pd.read_csv(report / "exact_duplicates.csv")
    assert len(exact) == summary["exact_duplicate_pairs"] == 3
    assert (exact.source_split == exact.target_split).sum() == 1
    expected = {"train/images/0.png", "train/images/exact_copy.png", "val/images/exact_copy.png"}
    assert set(exact.source) | set(exact.target) == expected
    near = pd.read_csv(report / "near_duplicates.csv")
    assert len(near) == summary["near_duplicate_pairs"] == 3
    assert set(near.target) == {"test/images/near_copy.png"}
    assert set(near.source) == expected
    assert near.distance.between(0, 5).all()
    leakage = pd.read_csv(report / "cross_split_leakage.csv")
    assert len(leakage) == summary["cross_split_leakage_pairs"] == 5
    assert leakage.leakage_type.value_counts().to_dict() == {
        "near_duplicate": 3,
        "exact_duplicate": 2,
    }
    assert (leakage.source_split != leakage.target_split).all()
    invalid = pd.read_csv(report / "invalid_images.csv")
    assert dict(zip(invalid.path, invalid.status)) == {
        "train/images/corrupt.jpg": "corrupt",
        "val/images/empty.png": "zero_byte",
        "test/images/notes.txt": "unsupported",
    }
    markdown = (report / "report.md").read_text(encoding="utf-8")
    for text in (
        "Dataset Health: FAIL",
        "Valid images: 12",
        "3 pairs / 1 groups",
        "Cross-split leakage: 5 pairs",
        "Corrupted/unreadable/empty files: 2",
    ):
        assert text in markdown
    # A second actual CLI run must reproduce findings and use a new report directory.
    assert subprocess.run(command, capture_output=True).returncode == 1
    second = next(path for path in output.iterdir() if path != report)
    other = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    summary.pop("created_at_utc")
    other.pop("created_at_utc")
    assert summary == other
    for path in report.glob("*.csv"):
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_split_alias_rejected():
    with pytest.raises(ValueError, match="ignoring case"):
        AuditConfig(train_name="train", validation_name="TRAIN")


def test_permission_error_is_recorded(dataset, image, monkeypatch):
    path = dataset / "train/images/a.png"
    image.save(path)

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("visual_inspection.data.validation.Image.open", denied)
    assert inspect_image(path, dataset, "train")["status"] == "unreadable"


def test_programming_errors_are_not_hidden(dataset, image, monkeypatch):
    path = dataset / "train/images/a.png"
    image.save(path)

    def broken(*args, **kwargs):
        raise RuntimeError("implementation defect")

    monkeypatch.setattr("visual_inspection.data.validation.phash_image", broken)
    with pytest.raises(RuntimeError, match="implementation defect"):
        inspect_image(path, dataset, "train")


def test_cli_yaml_override(dataset, tmp_path, image):
    image.save(dataset / "train/images/a.png")
    (dataset / "train/images/empty.png").touch()
    config = tmp_path / "settings.yaml"
    config.write_text("audit:\n  corrupted_image_failure_threshold: 0.0\n", encoding="utf-8")
    output = tmp_path / "reports"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_dataset.py"),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--config",
            str(config),
            "--corrupted-image-failure-threshold",
            "0.5",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert "Dataset Health: WARNING" in run.stdout
    summary = json.loads(next(output.glob("*/summary.json")).read_text(encoding="utf-8"))
    assert summary["config"]["corrupted_image_failure_threshold"] == 0.5

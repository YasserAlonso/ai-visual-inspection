import json
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageEnhance

from visual_inspection.audit import audit_dataset
from visual_inspection.cli import main
from visual_inspection.config import AuditConfig, load_config


def test_empty_dataset_reports_failure(dataset, tmp_path):
    result = audit_dataset(dataset, tmp_path / "reports")
    assert result.summary["health"] == "FAIL"
    assert result.summary["distributions"]["width"] == {}
    assert len(list(result.report_directory.iterdir())) == 7
    for path in result.report_directory.glob("*.csv"):
        assert len(pd.read_csv(path)) == 0
    assert json.loads((result.report_directory / "summary.json").read_text())["total_images"] == 0


def test_pass_and_statistics(dataset, tmp_path):
    rng = np.random.default_rng(99)
    for split in ("train", "val", "test"):
        Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)).save(
            dataset / split / "images/a.png"
        )
    result = audit_dataset(dataset, tmp_path / "reports")
    assert result.summary["health"] == "PASS"
    assert result.summary["distributions"]["width"]["mean"] == 96
    assert result.summary["distributions"]["height"]["50%"] == 64
    assert result.summary["images_per_split"] == {"train": 1, "val": 1, "test": 1}


def test_cli_failure_and_report(dataset, tmp_path, image, capsys):
    path = dataset / "train/images/a.png"
    image.save(path)
    shutil.copyfile(path, dataset / "val/images/a.png")
    assert main(["audit", "--dataset", str(dataset), "--output", str(tmp_path / "reports")]) == 1
    assert "Dataset Health: FAIL" in capsys.readouterr().out


def test_near_leakage_is_warning(dataset, tmp_path, image):
    image.save(dataset / "train/images/a.png")
    ImageEnhance.Brightness(image).enhance(0.95).save(dataset / "val/images/b.png")
    result = audit_dataset(dataset, tmp_path / "reports")
    assert result.summary["health"] == "WARNING"
    assert list(result.leakage.leakage_type) == ["near_duplicate"]


def test_corruption_threshold_boundary(dataset, tmp_path, image):
    image.save(dataset / "train/images/a.png")
    (dataset / "train/images/b.png").touch()
    result = audit_dataset(
        dataset, tmp_path / "reports", AuditConfig(corrupted_image_failure_threshold=0.5)
    )
    assert result.summary["health"] == "WARNING"
    assert audit_dataset(dataset, tmp_path / "reports").summary["health"] == "FAIL"


def test_config_overrides_and_validation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset:\n  validation_name: validation\naudit:\n  phash_distance_threshold: 2\n"
    )
    config = load_config(path, phash_distance_threshold=7)
    assert config.validation_name == "validation"
    assert config.phash_distance_threshold == 7
    for value in (-1, 65, True, "5"):
        with pytest.raises(ValueError):
            AuditConfig(phash_distance_threshold=value)
    path.write_text("audit:\n  typo: 1\n")
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(ValueError):
        AuditConfig(train_name="../outside")


def test_output_cannot_contaminate_input(dataset):
    with pytest.raises(ValueError, match="outside"):
        audit_dataset(dataset, dataset / "reports")


def test_missing_structure(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    result = audit_dataset(root, tmp_path / "reports")
    assert len(result.summary["structure_issues"]) == 6
    assert result.summary["health"] == "FAIL"


def test_single_image_statistics_are_json_safe(dataset, tmp_path, image):
    image.save(dataset / "train/images/a.png")
    result = audit_dataset(dataset, tmp_path / "reports")
    summary = json.loads((result.report_directory / "summary.json").read_text())
    assert summary["distributions"]["width"]["std"] is None


def test_annotation_audit_extends_phase_zero_report(dataset, tmp_path, image):
    import yaml

    for index, split in enumerate(("train", "val", "test")):
        Image.new("RGB", (96, 64), color=(30 + index * 50, 50, 80)).save(
            dataset / split / "images/a.png"
        )
        (dataset / split / "labels/a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    data_yaml = dataset / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "crack", 1: "pothole"},
            }
        )
    )
    result = audit_dataset(dataset, tmp_path / "reports", yolo_dataset_yaml=data_yaml)
    assert result.summary["health"] == "WARNING"
    assert result.summary["annotations"]["total_labeled_objects"] == 3
    assert result.summary["annotations"]["severe_class_imbalance"]
    assert (result.report_directory / "annotation_issues.csv").is_file()
    report = (result.report_directory / "report.md").read_text()
    assert "Dataset Health: WARNING" in report


def test_annotation_yaml_root_must_match_audit_root(dataset, tmp_path, image):
    import yaml

    other = tmp_path / "other"
    other.mkdir()
    for split in ("x", "y"):
        (other / split).mkdir()
    data_yaml = other / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(other), "train": "x", "val": "y", "names": ["crack"]})
    )
    with pytest.raises(ValueError, match="root must match"):
        audit_dataset(dataset, tmp_path / "reports", yolo_dataset_yaml=data_yaml)

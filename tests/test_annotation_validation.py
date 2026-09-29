from pathlib import Path

from PIL import Image

from visual_inspection.training.dataset import load_dataset_yaml, validate_annotations


def _dataset(root: Path, names="{0: crack, 1: pothole}") -> Path:
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        Image.new("RGB", (32, 32)).save(root / split / "images/a.png")
        (root / split / "labels/a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    path = root / "data.yaml"
    path.write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: val/images\n"
        f"test: test/images\nnames: {names}\n",
        encoding="utf-8",
    )
    return path


def test_valid_labels_and_statistics(tmp_path):
    dataset = load_dataset_yaml(_dataset(tmp_path))
    issues, stats = validate_annotations(dataset)
    assert issues.empty
    assert stats["total_labeled_objects"] == 3
    assert stats["objects_per_class"] == {"crack": 3, "pothole": 0}
    assert stats["severe_class_imbalance"]
    assert stats["box_distributions"]["area"]["mean"] == 0.25


def test_invalid_rows_missing_and_orphan_labels(tmp_path):
    yaml_path = _dataset(tmp_path)
    (tmp_path / "train/labels/a.txt").write_text(
        "0 0.5 0.5 0 0.5\n1 2 0.5 0.2 0.2\n4 0.5 0.5 0.2 0.2\n"
        "0 nope 0.5 0.2 0.2\n0 0.5 0.5 0.2\n0 0.5 0.5 -0.1 0.2\n"
        "0.0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    (tmp_path / "train/labels/orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "val/labels/a.txt").unlink()
    issues, stats = validate_annotations(load_dataset_yaml(yaml_path))
    assert {
        "coordinate_out_of_range_or_zero_box",
        "invalid_class_id",
        "non_numeric_value",
        "malformed_row: expected 5 values",
        "negative_coordinate",
        "orphan_label",
        "missing_label",
    } <= set(issues.issue)
    assert stats["total_labeled_objects"] == 1


def test_tiny_and_empty_label_detection(tmp_path):
    yaml_path = _dataset(tmp_path)
    (tmp_path / "train/labels/a.txt").write_text("0 0.5 0.5 0.001 0.001\n", encoding="utf-8")
    (tmp_path / "val/labels/a.txt").write_text("\n", encoding="utf-8")
    issues, _ = validate_annotations(load_dataset_yaml(yaml_path), minimum_box_area=0.0001)
    assert "extremely_small_box" in set(issues.issue)
    assert "empty_annotation" in set(issues.issue)


def test_single_object_box_statistics_are_json_safe(tmp_path):
    yaml_path = _dataset(tmp_path)
    for split in ("val", "test"):
        (tmp_path / split / "labels/a.txt").write_text("", encoding="utf-8")
    _, stats = validate_annotations(load_dataset_yaml(yaml_path))
    assert stats["box_distributions"]["width"]["std"] is None


def test_invalid_names_rejected(tmp_path):
    yaml_path = _dataset(tmp_path, names="{1: crack}")
    import pytest

    with pytest.raises(ValueError, match="contiguous"):
        load_dataset_yaml(yaml_path)

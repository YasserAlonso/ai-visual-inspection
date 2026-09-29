from pathlib import Path

import yaml

from visual_inspection.training.dataset import load_dataset_yaml, write_absolute_dataset_yaml


def test_dataset_yaml_mapping_and_absolute_generation(tmp_path):
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir()
    source = tmp_path / "dataset.yaml"
    source.write_text(
        "path: .\ntrain: train/images\nval: val/images\nnames:\n  0: crack\n", encoding="utf-8"
    )
    parsed = load_dataset_yaml(source)
    output = write_absolute_dataset_yaml(parsed, tmp_path / "absolute.yaml")
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert Path(data["path"]).is_absolute()
    assert Path(data["train"]).is_absolute()
    assert data["names"] == {0: "crack"}


def test_missing_required_dataset_fields_rejected(tmp_path):
    source = tmp_path / "dataset.yaml"
    source.write_text("train: x\n", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="requires"):
        load_dataset_yaml(source)

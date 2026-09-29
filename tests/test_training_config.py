from dataclasses import replace

import pytest

from visual_inspection.training.config import TrainingConfig, load_training_config


def test_load_baseline_yaml():
    from pathlib import Path

    config = load_training_config(Path("configs/training/baseline.yaml"))
    assert config.model == "yolo11n.pt"
    assert config.epochs == 30
    assert config.dataset.is_absolute()
    assert config.device == "auto"
    assert config.augmentation["flipud"] == 0
    from visual_inspection import __version__

    assert __version__ == "0.2.0.dev0"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"image_size": -1},
        {"batch_size": 0},
        {"workers": -1},
        {"seed": -1},
        {"patience": -1},
        {"model": "yolo11n.yaml"},
        {"experiment_name": "../escape"},
        {"run_label": ""},
        {"imbalance_warning_ratio": 0},
        {"augmentation": {"flipud": 4}},
        {"device": True},
        {"device": -1},
        {"device": "gpu"},
    ],
)
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        replace(TrainingConfig(), **kwargs)


def test_unknown_yaml_setting_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("epoch: 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_training_config(path)


def test_rotation_setting_allows_degrees():
    assert replace(TrainingConfig(), augmentation={"degrees": 5.0}).augmentation["degrees"] == 5

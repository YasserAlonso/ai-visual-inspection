"""Validated baseline configuration with config-relative paths."""

from dataclasses import dataclass, field
from pathlib import Path
import math

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "yolo11n.pt"
    dataset: Path = Path("dataset.yaml")
    epochs: int = 30
    image_size: int = 640
    batch_size: int = 8
    device: str = "auto"
    workers: int = 4
    seed: int = 42
    patience: int = 10
    experiment_name: str = "aeroinspect-detection-baseline"
    run_label: str = "BASELINE TRAINING"
    project_directory: Path = Path("artifacts/experiments")
    audit_output_directory: Path = Path("data/reports/training")
    phash_distance_threshold: int = 5
    corrupted_image_failure_threshold: float = 0.01
    allow_failed_audit: bool = False
    minimum_box_area: float = 0.0001
    imbalance_warning_ratio: float = 20.0
    augmentation: dict[str, float] = field(
        default_factory=lambda: {
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.1,
            "fliplr": 0.5,
            "flipud": 0.0,
            "mosaic": 0.5,
            "mixup": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.5,
            "hsv_v": 0.3,
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty weight name or path")
        if Path(self.model).suffix.lower() != ".pt":
            raise ValueError("model must reference pretrained PyTorch .pt weights")
        for name in ("epochs", "image_size", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("workers", "seed", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        valid_device = (
            self.device in {"auto", "cpu"}
            if isinstance(self.device, str)
            else type(self.device) is int and self.device >= 0
        ) or (isinstance(self.device, str) and self.device.isdigit())
        if not valid_device:
            raise ValueError("device must be auto, cpu, or a non-negative CUDA index")
        if (
            not isinstance(self.experiment_name, str)
            or not self.experiment_name.strip()
            or Path(self.experiment_name).name != self.experiment_name
        ):
            raise ValueError("experiment_name must be a nonempty folder name")
        if not isinstance(self.run_label, str) or not self.run_label.strip():
            raise ValueError("run_label must be a nonempty string")
        for name in ("dataset", "project_directory", "audit_output_directory"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a path")
        if type(self.allow_failed_audit) is not bool:
            raise ValueError("allow_failed_audit must be boolean")
        if (
            type(self.phash_distance_threshold) is not int
            or not 0 <= self.phash_distance_threshold <= 64
        ):
            raise ValueError("phash_distance_threshold must be from 0 to 64")
        for name, value, low, high in (
            ("corrupted_image_failure_threshold", self.corrupted_image_failure_threshold, 0, 1),
            ("minimum_box_area", self.minimum_box_area, 0, 1),
        ):
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be finite and between {low} and {high}")
        if (
            type(self.imbalance_warning_ratio) not in (float, int)
            or not math.isfinite(self.imbalance_warning_ratio)
            or self.imbalance_warning_ratio < 1
        ):
            raise ValueError("imbalance_warning_ratio must be finite and at least 1")
        allowed = {
            "degrees",
            "translate",
            "scale",
            "fliplr",
            "flipud",
            "mosaic",
            "mixup",
            "hsv_h",
            "hsv_s",
            "hsv_v",
        }
        if not isinstance(self.augmentation, dict) or set(self.augmentation) - allowed:
            raise ValueError("augmentation contains unsupported Ultralytics settings")
        if any(
            type(v) not in (float, int)
            or not math.isfinite(v)
            or v < 0
            or v > (180 if k == "degrees" else 1)
            for k, v in self.augmentation.items()
        ):
            raise ValueError("augmentation values must be non-negative; degrees <=180, others <=1")


def load_training_config(path: Path) -> TrainingConfig:
    path = path.resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Training config must be a YAML mapping")
    allowed = set(TrainingConfig.__dataclass_fields__)
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
    for key in ("dataset", "project_directory", "audit_output_directory"):
        if key in document:
            if not isinstance(document[key], str) or not document[key].strip():
                raise ValueError(f"{key} must be a nonempty path")
            document[key] = (path.parent / document[key]).resolve()
    if isinstance(document.get("model"), str):
        model_path = path.parent / document["model"]
        if model_path.is_file():
            document["model"] = str(model_path.resolve())
    return TrainingConfig(**document)

"""Strict, lightweight YAML configuration with explicit CLI overrides."""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AuditConfig:
    train_name: str = "train"
    validation_name: str = "val"
    test_name: str = "test"
    perceptual_hash: str = "phash"
    phash_distance_threshold: int = 5
    corrupted_image_failure_threshold: float = 0.01
    minimum_dimension_warning: int = 32
    aspect_ratio_warning: float = 10.0

    def __post_init__(self) -> None:
        for name in self.splits:
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or ":" in name
            ):
                raise ValueError("Split names must be single directory names")
        if len({name.casefold() for name in self.splits}) != 3:
            raise ValueError("Split names must be distinct, ignoring case")
        if self.perceptual_hash != "phash":
            raise ValueError("Only phash is supported in Phase 0")
        if type(self.phash_distance_threshold) is not int or not (
            0 <= self.phash_distance_threshold <= 64
        ):
            raise ValueError("phash_distance_threshold must be an integer from 0 to 64")
        rate = self.corrupted_image_failure_threshold
        if type(rate) not in (int, float) or not 0 <= rate <= 1:
            raise ValueError("corrupted_image_failure_threshold must be between 0 and 1")
        if type(self.minimum_dimension_warning) is not int or self.minimum_dimension_warning < 1:
            raise ValueError("minimum_dimension_warning must be a positive integer")
        ratio = self.aspect_ratio_warning
        if type(ratio) not in (int, float) or not 1 <= ratio < float("inf"):
            raise ValueError("aspect_ratio_warning must be finite and at least 1")

    @property
    def splits(self) -> tuple[str, str, str]:
        return self.train_name, self.validation_name, self.test_name


def load_config(path: Path | None = None, **overrides: object) -> AuditConfig:
    values = {}
    if path is not None:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) - {"dataset", "audit"}:
            raise ValueError("Configuration must contain dataset and/or audit mappings")
        dataset_keys = {"train_name", "validation_name", "test_name"}
        all_keys = {field.name for field in fields(AuditConfig)}
        for section, allowed in (("dataset", dataset_keys), ("audit", all_keys - dataset_keys)):
            content = document.get(section, {})
            if not isinstance(content, dict) or set(content) - allowed:
                raise ValueError(f"Unknown keys or invalid mapping in {section}")
            values.update(content)
    values.update({key: value for key, value in overrides.items() if value is not None})
    return AuditConfig(**values)

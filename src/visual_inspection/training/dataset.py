"""YOLO detection dataset YAML and label validation/statistics."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math

import pandas as pd
import yaml

from visual_inspection.data.validation import EXTENSIONS

ISSUE_COLUMNS = ["split", "path", "line", "issue"]


@dataclass
class YoloDataset:
    yaml_path: Path
    root: Path
    names: dict[int, str]
    splits: dict[str, Path]


def load_dataset_yaml(path: Path) -> YoloDataset:
    path = path.resolve()
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not {"path", "train", "val", "names"} <= set(doc):
        raise ValueError("Dataset YAML requires path, train, val, and names")
    root = Path(doc["path"])
    root = (root if root.is_absolute() else path.parent / root).resolve()
    raw_names = doc["names"]
    if isinstance(raw_names, list):
        names = dict(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        try:
            names = {int(key): value for key, value in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Class names must use integer IDs") from exc
    else:
        raise ValueError("names must be a list or integer-keyed mapping")
    if (
        not names
        or sorted(names) != list(range(len(names)))
        or any(not isinstance(name, str) or not name.strip() for name in names.values())
    ):
        raise ValueError("Class IDs must be contiguous from zero with nonempty names")
    splits = {}
    for key in ("train", "val", "test"):
        if key not in doc:
            continue
        entries = doc[key] if isinstance(doc[key], list) else [doc[key]]
        if len(entries) != 1 or not isinstance(entries[0], str):
            raise ValueError(f"{key} must point to one image directory for Phase 1")
        candidate = Path(entries[0])
        candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if not candidate.is_dir():
            raise ValueError(f"Dataset split directory does not exist: {candidate}")
        splits[key] = candidate
    if not {"train", "val"} <= set(splits):
        raise ValueError("Dataset must include train and val image directories")
    return YoloDataset(path, root, names, splits)


def _image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def validate_annotations(
    dataset: YoloDataset, minimum_box_area: float = 0.0001, imbalance_warning_ratio: float = 20.0
) -> tuple[pd.DataFrame, dict]:
    issues: list[dict] = []
    object_rows: list[dict] = []
    per_image: list[dict] = []
    for split, images_dir in dataset.splits.items():
        labels_dir = images_dir.parent / "labels"
        images = _image_files(images_dir)
        image_stems = {p.relative_to(images_dir).with_suffix("").as_posix() for p in images}
        label_files = sorted(labels_dir.rglob("*.txt")) if labels_dir.is_dir() else []
        label_stems = {p.relative_to(labels_dir).with_suffix("").as_posix() for p in label_files}
        for missing in sorted(image_stems - label_stems):
            issues.append(
                dict(split=split, path=f"{missing}.txt", line=None, issue="missing_label")
            )
            per_image.append(dict(split=split, image=missing, object_count=0))
        for orphan in sorted(label_stems - image_stems):
            issues.append(dict(split=split, path=f"{orphan}.txt", line=None, issue="orphan_label"))
        label_by_stem = {
            p.relative_to(labels_dir).with_suffix("").as_posix(): p for p in label_files
        }
        for stem in sorted(image_stems & label_stems):
            label = label_by_stem[stem]
            try:
                rows = label.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as exc:
                issues.append(
                    dict(
                        split=split,
                        path=label.as_posix(),
                        line=None,
                        issue=f"unreadable_label: {exc}",
                    )
                )
                continue
            if not rows or not any(line.strip() for line in rows):
                issues.append(
                    dict(split=split, path=label.as_posix(), line=None, issue="empty_annotation")
                )
                per_image.append(dict(split=split, image=stem, object_count=0))
                continue
            object_count = 0
            for line_number, raw in enumerate(rows, 1):
                parts = raw.split()
                if len(parts) != 5:
                    issues.append(
                        dict(
                            split=split,
                            path=label.as_posix(),
                            line=line_number,
                            issue="malformed_row: expected 5 values",
                        )
                    )
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    issues.append(
                        dict(
                            split=split,
                            path=label.as_posix(),
                            line=line_number,
                            issue="invalid_class_id",
                        )
                    )
                    continue
                try:
                    coordinates = [float(value) for value in parts[1:]]
                except ValueError:
                    issues.append(
                        dict(
                            split=split,
                            path=label.as_posix(),
                            line=line_number,
                            issue="non_numeric_value",
                        )
                    )
                    continue
                if class_id not in dataset.names:
                    issues.append(
                        dict(
                            split=split,
                            path=label.as_posix(),
                            line=line_number,
                            issue="invalid_class_id",
                        )
                    )
                    continue
                x, y, width, height = coordinates
                if not all(math.isfinite(value) for value in coordinates):
                    issue = "non_finite_coordinate"
                elif any(value < 0 for value in coordinates):
                    issue = "negative_coordinate"
                elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    issue = "coordinate_out_of_range_or_zero_box"
                elif (
                    x - width / 2 < 0
                    or x + width / 2 > 1
                    or y - height / 2 < 0
                    or y + height / 2 > 1
                ):
                    issue = "box_exceeds_image_bounds"
                elif width * height < minimum_box_area:
                    issue = "extremely_small_box"
                else:
                    issue = None
                if issue:
                    issues.append(
                        dict(split=split, path=label.as_posix(), line=line_number, issue=issue)
                    )
                    continue
                object_count += 1
                object_rows.append(
                    dict(
                        split=split,
                        image=stem,
                        class_id=class_id,
                        class_name=dataset.names[class_id],
                        width=width,
                        height=height,
                        area=width * height,
                    )
                )
            per_image.append(dict(split=split, image=stem, object_count=object_count))
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    objects = pd.DataFrame(
        object_rows, columns=["split", "image", "class_id", "class_name", "width", "height", "area"]
    )
    per_image_frame = pd.DataFrame(per_image, columns=["split", "image", "object_count"])
    counts = Counter(objects.class_name) if len(objects) else Counter()
    nonzero = [count for count in counts.values() if count]
    has_missing_class = bool(
        len(objects) and any(counts.get(name, 0) == 0 for name in dataset.names.values())
    )
    ratio = None if has_missing_class else max(nonzero) / min(nonzero) if len(nonzero) > 1 else 1.0
    box_distributions = {}
    for column in ("width", "height", "area"):
        box_distributions[column] = (
            {
                key: float(value) if pd.notna(value) else None
                for key, value in objects[column].describe().items()
            }
            if len(objects)
            else {}
        )
    stats = dict(
        total_labeled_objects=len(objects),
        objects_per_class={name: counts.get(name, 0) for name in dataset.names.values()},
        class_distribution_ratio=ratio,
        severe_class_imbalance=has_missing_class or ratio > imbalance_warning_ratio,
        objects_per_image=per_image_frame.to_dict("records"),
        box_distributions=box_distributions,
        issue_count=len(issue_frame),
        split_directories={k: str(v) for k, v in dataset.splits.items()},
        names=dataset.names,
    )
    return issue_frame, stats


def write_absolute_dataset_yaml(dataset: YoloDataset, destination: Path) -> Path:
    document = {
        "path": str(dataset.root),
        **{key: str(value) for key, value in dataset.splits.items()},
        "names": dataset.names,
    }
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return destination

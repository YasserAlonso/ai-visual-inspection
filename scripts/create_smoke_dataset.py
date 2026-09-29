"""Create tiny random images/labels strictly for a software training smoke test."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

from visual_inspection.training.config import TrainingConfig


def create_smoke_dataset(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing path: {destination}")
    rng = np.random.default_rng(2718)
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        images, labels = destination / split / "images", destination / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        for index in range(count):
            array = rng.integers(70, 115, size=(128, 128, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            draw = ImageDraw.Draw(image)
            x, y = 36 + int(rng.integers(0, 15)), 40 + int(rng.integers(0, 15))
            draw.rectangle((x, y, x + 23, y + 9), fill=(220, 55, 40))
            image.save(images / f"sample_{index:02}.png")
            cx, cy = (x + 11.5) / 128, (y + 4.5) / 128
            (labels / f"sample_{index:02}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {24 / 128:.6f} {10 / 128:.6f}\n", encoding="utf-8"
            )
    yaml_path = destination / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(destination),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "synthetic_defect"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    smoke_config = asdict(
        TrainingConfig(
            dataset=yaml_path,
            epochs=1,
            image_size=128,
            batch_size=2,
            workers=0,
            patience=1,
            experiment_name="synthetic-smoke-test",
            run_label="SOFTWARE SMOKE TEST ONLY",
            project_directory=destination.parent / "experiments",
            audit_output_directory=destination.parent / "reports",
        )
    )
    for key in ("dataset", "project_directory", "audit_output_directory"):
        smoke_config[key] = str(smoke_config[key])
    config_path = destination.parent / "training-smoke.yaml"
    config_path.write_text(yaml.safe_dump(smoke_config, sort_keys=False), encoding="utf-8")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke/dataset"))
    dataset_yaml = create_smoke_dataset(parser.parse_args().output)
    print(f"SOFTWARE SMOKE TEST ONLY; synthetic dataset config: {dataset_yaml}")
    print(
        f"Run the one-epoch software smoke test: python scripts/train_baseline.py --config "
        f"{dataset_yaml.parent.parent / 'training-smoke.yaml'}"
    )

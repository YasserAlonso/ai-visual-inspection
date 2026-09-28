"""Create a small reproducible dataset containing deliberate audit findings."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


def create_demo(destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"Refusing to overwrite existing path: {destination}")
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        (destination / split / "images").mkdir(parents=True)
        (destination / split / "labels").mkdir()
        for number in range(3):
            Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)).save(
                destination / split / "images" / f"{number}.png"
            )
    source = destination / "train/images/0.png"
    shutil.copyfile(source, destination / "train/images/exact_copy.png")
    shutil.copyfile(source, destination / "val/images/exact_copy.png")
    with Image.open(source) as image:
        ImageEnhance.Brightness(image).enhance(0.95).save(destination / "test/images/near_copy.png")
    (destination / "train/images/corrupt.jpg").write_bytes(b"broken image")
    (destination / "val/images/empty.png").touch()
    (destination / "test/images/notes.txt").write_text("unsupported file", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/demo"))
    create_demo(parser.parse_args().output)

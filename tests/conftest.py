from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        for folder in ("images", "labels"):
            (root / split / folder).mkdir(parents=True)
    return root


@pytest.fixture
def image() -> Image.Image:
    return Image.fromarray(np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8))

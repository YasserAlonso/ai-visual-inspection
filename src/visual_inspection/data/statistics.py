"""JSON-safe distributions of valid-image metadata."""

import pandas as pd


def image_statistics(images: pd.DataFrame, splits: tuple[str, ...]) -> dict:
    valid = images[images.status == "valid"]
    distributions = {}
    for column in ("width", "height", "aspect_ratio", "size_bytes"):
        series = valid[column].astype(float)
        distributions[column] = (
            {
                key: float(value) if pd.notna(value) else None
                for key, value in series.describe().items()
            }
            if len(series)
            else {}
        )
    return dict(
        total_files=len(images),
        total_images=len(valid),
        images_per_split={split: int((valid.split == split).sum()) for split in splits},
        distributions=distributions,
    )

"""Cross-split leakage includes only actual matching pairs, never inferred group edges."""

import pandas as pd

from visual_inspection.data.duplicates import PAIR_COLUMNS


def detect_leakage(exact: pd.DataFrame, near: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for kind, pairs in (("exact_duplicate", exact), ("near_duplicate", near)):
        subset = pairs[pairs.source_split != pairs.target_split].copy()
        subset["leakage_type"] = kind
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)[PAIR_COLUMNS + ["leakage_type"]]

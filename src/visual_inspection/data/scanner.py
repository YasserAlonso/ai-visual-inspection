"""Deterministic recursive scanning without following directory symlinks."""

import logging
import os
from pathlib import Path

import pandas as pd

from visual_inspection.config import AuditConfig
from visual_inspection.data.validation import inspect_image

LOGGER = logging.getLogger(__name__)
COLUMNS = [
    "path",
    "split",
    "size_bytes",
    "width",
    "height",
    "aspect_ratio",
    "format",
    "mode",
    "sha256",
    "phash",
    "status",
    "error",
]


def scan_dataset(root: Path, config: AuditConfig) -> pd.DataFrame:
    records = []
    for split in config.splits:
        directory = root / split / "images"
        if not directory.is_dir():
            continue
        if directory.is_symlink() or directory.parent.is_symlink():
            raise ValueError(f"Split/image directories cannot be symlinks: {directory}")
        LOGGER.info("Scanning %s", split)

        def on_error(error: OSError) -> None:
            raise error  # Incomplete enumeration must not silently produce a PASS.

        for current, dirs, files in os.walk(directory, followlinks=False, onerror=on_error):
            dirs.sort()
            for name in dirs:
                if (Path(current) / name).is_symlink():
                    raise ValueError(f"Symlink directory cannot be audited: {Path(current) / name}")
            for name in sorted(files):
                records.append(inspect_image(Path(current) / name, root, split))
                if len(records) % 500 == 0:
                    LOGGER.info("Inspected %d files", len(records))
    return pd.DataFrame(records, columns=COLUMNS)

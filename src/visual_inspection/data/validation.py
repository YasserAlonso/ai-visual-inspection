"""Image validation and dataset layout checks; annotation checks can be added here."""

import warnings
from pathlib import Path

from PIL import Image

from visual_inspection.config import AuditConfig
from visual_inspection.data.hashing import phash_image, sha256_file

EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF"})


def validate_structure(root: Path, config: AuditConfig) -> list[str]:
    issues = []
    for split in config.splits:
        for folder in ("images", "labels"):
            if not (root / split / folder).is_dir():
                issues.append(f"Missing directory: {split}/{folder}")
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name not in config.splits:
            issues.append(f"Unrecognized root directory (not scanned): {child.name}")
    return issues


def inspect_image(path: Path, root: Path, split: str) -> dict:
    record = dict(
        path=path.relative_to(root).as_posix(),
        split=split,
        size_bytes=None,
        width=None,
        height=None,
        aspect_ratio=None,
        format=None,
        mode=None,
        sha256=None,
        phash=None,
        status="valid",
        error="",
    )
    try:
        if path.is_symlink():
            record.update(status="unreadable", error="Symbolic links are not followed")
            return record
        record["size_bytes"] = path.stat().st_size
        if record["size_bytes"] == 0:
            record.update(status="zero_byte", error="Empty file")
        elif path.suffix.lower() not in EXTENSIONS:
            record.update(status="unsupported", error="Unsupported extension")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    if image.format not in FORMATS:
                        record.update(status="unsupported", error="Unsupported decoded format")
                        return record
                    width, height = image.size
                    record.update(
                        width=width,
                        height=height,
                        aspect_ratio=width / height,
                        format=image.format,
                        mode=image.mode,
                        phash=phash_image(image),
                    )
            record["sha256"] = sha256_file(path)
    except (PermissionError, FileNotFoundError) as exc:
        record.update(status="unreadable", error=str(exc))
    except (
        OSError,
        SyntaxError,
        ValueError,
        EOFError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        # Isolate expected decoding failures without hiding programming/resource errors.
        record.update(status="corrupt", error=f"{type(exc).__name__}: {exc}")
    return record

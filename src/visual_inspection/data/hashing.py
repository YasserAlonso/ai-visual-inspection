"""Streaming byte identity and orientation-normalized visual fingerprints."""

import hashlib
from pathlib import Path

import imagehash
from PIL import Image, ImageOps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash_image(image: Image.Image) -> str:
    return str(imagehash.phash(ImageOps.exif_transpose(image).convert("RGB"), hash_size=8))

"""Convert inspected Pascal VOC XML annotations to YOLO without editing source files.

This converter is intentionally fail-closed: the class map must be populated from
the acquired dataset's authoritative metadata before it will run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Source image directory")
    parser.add_argument("--annotations", type=Path, required=True, help="Pascal VOC XML directory")
    parser.add_argument("--output", type=Path, required=True, help="New processed output directory")
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/datasets/aircraft_fuselage_det2023.yaml"),
    )
    args = parser.parse_args()
    images_dir, annotations_dir = args.images.resolve(), args.annotations.resolve()
    output_dir, config_path = args.output.resolve(), args.dataset_config.resolve()
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        parser.error("--images and --annotations must be existing directories")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_classes = config.get("source_classes") if isinstance(config, dict) else None
    if not source_classes:
        parser.error(
            f"{config_path} has no verified source_classes mapping; inspect the original package first"
        )
    if not isinstance(source_classes, dict) or any(
        not isinstance(source, str) or not isinstance(name, str) or not name.strip()
        for source, name in source_classes.items()
    ):
        parser.error("source_classes must map exact source label strings to canonical names")
    names = list(dict.fromkeys(source_classes.values()))
    class_ids = {name: index for index, name in enumerate(names)}
    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    images_by_name: dict[str, list[Path]] = {}
    for path in image_paths:
        images_by_name.setdefault(path.name, []).append(path)
    records = []
    problems = []
    seen = set()
    annotated_images = set()
    xml_files = sorted(annotations_dir.rglob("*.xml"))
    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
            filename = (root.findtext("filename") or f"{xml_path.stem}.jpg").strip()
            candidates = images_by_name.get(Path(filename).name, [])
            if len(candidates) != 1:
                raise ValueError(f"image lookup for {filename!r} found {len(candidates)} matches")
            image_path = candidates[0]
            rel = image_path.relative_to(images_dir)
            output_key = rel.with_suffix("").as_posix()
            if output_key in seen:
                raise ValueError(f"duplicate output image path: {output_key}")
            seen.add(output_key)
            annotated_images.add(image_path.resolve())
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            if width <= 0 or height <= 0:
                raise ValueError("image has invalid dimensions")
            rows = []
            for obj in root.findall("object"):
                source_name = (obj.findtext("name") or "").strip()
                if source_name not in source_classes:
                    raise ValueError(f"unmapped source class {source_name!r}")
                box = obj.find("bndbox")
                if box is None:
                    raise ValueError(f"object {source_name!r} is missing bndbox")
                xmin, ymin, xmax, ymax = (
                    float(box.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")
                )
                # VOC coordinates are 1-based inclusive. Convert to pixel-edge coordinates.
                x0, y0, x1, y1 = xmin - 1.0, ymin - 1.0, xmax, ymax
                if not all(map(math.isfinite, (x0, y0, x1, y1))) or not (
                    0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
                ):
                    raise ValueError(
                        f"invalid/out-of-bounds VOC box {(xmin, ymin, xmax, ymax)} for {width}x{height}"
                    )
                xc, yc = (x0 + x1) / (2 * width), (y0 + y1) / (2 * height)
                bw, bh = (x1 - x0) / width, (y1 - y0) / height
                rows.append(
                    f"{class_ids[source_classes[source_name]]} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"
                )
            records.append((image_path, rel, output_key, rows))
        except (ET.ParseError, OSError, ValueError, TypeError) as exc:
            problems.append(
                {"annotation": xml_path.relative_to(annotations_dir).as_posix(), "error": str(exc)}
            )
    missing_annotations = sorted(
        path.relative_to(images_dir).as_posix()
        for path in image_paths
        if path.resolve() not in annotated_images
    )
    orphan_annotations = sorted(
        item["annotation"] for item in problems if "found 0 matches" in item["error"]
    )
    summary = {
        "dataset_id": "aircraft_fuselage_det2023",
        "native_format": "Pascal VOC XML (must be verified against source package)",
        "output_format": "YOLO detection labels in images/all and labels/all",
        "class_names": names,
        "images_processed": len(records),
        "objects_per_class": {
            name: sum(
                row.startswith(f"{class_ids[name]} ") for _, _, _, rows in records for row in rows
            )
            for name in names
        },
        "invalid_annotations": problems,
        "skipped_files": [item["annotation"] for item in problems],
        "missing_images": orphan_annotations,
        "missing_labels": missing_annotations,
        "source_sha256": {
            p.relative_to(images_dir).as_posix(): digest_file(p) for p in image_paths
        },
    }
    if output_dir.exists():
        summary_path = output_dir / "conversion_summary.json"
        files_match = True
        for image_path, rel, key, rows in records:
            image_target = output_dir / "images" / "all" / rel
            label_target = output_dir / "labels" / "all" / Path(key).with_suffix(".txt")
            expected_label = "\n".join(rows) + ("\n" if rows else "")
            if (
                not image_target.is_file()
                or digest_file(image_target) != summary["source_sha256"][rel.as_posix()]
                or not label_target.is_file()
                or label_target.read_text(encoding="utf-8") != expected_label
            ):
                files_match = False
                break
        if (
            summary_path.is_file()
            and files_match
            and json.loads(summary_path.read_text(encoding="utf-8")) == summary
        ):
            print(
                f"Existing output matches this source and mapping; leaving it unchanged: {output_dir}"
            )
            return 0
        parser.error(
            f"output exists and does not match this conversion; choose a new path: {output_dir}"
        )
    if problems or missing_annotations or orphan_annotations:
        print(json.dumps(summary, indent=2))
        print(
            "Conversion stopped: resolve malformed entries, missing images, or missing annotations first."
        )
        return 2
    try:
        for image_path, rel, key, rows in records:
            image_target = output_dir / "images" / "all" / rel
            label_target = output_dir / "labels" / "all" / Path(key).with_suffix(".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_target)
            label_target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "conversion_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

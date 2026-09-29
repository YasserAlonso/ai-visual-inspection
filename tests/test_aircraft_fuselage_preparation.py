import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_aircraft_fuselage_det2023.py"


def test_voc_conversion_and_safe_rerun(tmp_path):
    images = tmp_path / "source" / "JPEGImages"
    annotations = tmp_path / "source" / "Annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (20, 10), "white").save(images / "sample.png")
    (annotations / "sample.xml").write_text(
        """<annotation><filename>sample.png</filename><size><width>20</width><height>10</height></size>
        <object><name>source mark</name><bndbox><xmin>1</xmin><ymin>1</ymin>
        <xmax>20</xmax><ymax>10</ymax></bndbox></object></annotation>""",
        encoding="utf-8",
    )
    config = tmp_path / "dataset.yaml"
    config.write_text(yaml.safe_dump({"source_classes": {"source mark": "surface_mark"}}))
    output = tmp_path / "processed"
    command = [
        sys.executable,
        str(SCRIPT),
        "--images",
        str(images),
        "--annotations",
        str(annotations),
        "--output",
        str(output),
        "--dataset-config",
        str(config),
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert (output / "labels/all/sample.txt").read_text(encoding="utf-8") == (
        "0 0.50000000 0.50000000 1.00000000 1.00000000\n"
    )
    assert yaml.safe_load((output / "conversion_summary.json").read_text())[
        "objects_per_class"
    ] == {"surface_mark": 1}
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    assert "leaving it unchanged" in second.stdout
    assert (images / "sample.png").is_file()


def test_converter_refuses_unverified_empty_class_map(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    config = tmp_path / "dataset.yaml"
    config.write_text("source_classes: {}\n")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--images",
            str(images),
            "--annotations",
            str(annotations),
            "--output",
            str(tmp_path / "processed"),
            "--dataset-config",
            str(config),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "no verified source_classes mapping" in result.stderr
    assert not (tmp_path / "processed").exists()

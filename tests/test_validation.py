from visual_inspection.config import AuditConfig
from visual_inspection.data.scanner import scan_dataset
from visual_inspection.data.validation import validate_structure


def test_bad_files_do_not_stop_scan(dataset, image):
    folder = dataset / "train/images"
    image.save(folder / "valid.PNG")
    (folder / "broken.jpg").write_bytes(b"not an image")
    (folder / "empty.png").touch()
    (folder / "notes.txt").write_text("unsupported")
    frame = scan_dataset(dataset, AuditConfig()).set_index("path")
    assert len(frame) == 4
    assert frame.loc["train/images/broken.jpg", "status"] == "corrupt"
    assert frame.loc["train/images/empty.png", "status"] == "zero_byte"
    assert frame.loc["train/images/notes.txt", "status"] == "unsupported"
    valid = frame.loc["train/images/valid.PNG"]
    assert (valid.width, valid.height, valid.aspect_ratio) == (96, 64, 1.5)
    assert valid.size_bytes > 0
    assert validate_structure(dataset, AuditConfig()) == []


def test_truncated_image(dataset, image):
    path = dataset / "train/images/truncated.png"
    image.save(path)
    path.write_bytes(path.read_bytes()[:100])
    assert scan_dataset(dataset, AuditConfig()).iloc[0].status == "corrupt"

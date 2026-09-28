import shutil

from visual_inspection.config import AuditConfig
from visual_inspection.data.duplicates import exact_duplicates, near_duplicates
from visual_inspection.data.leakage import detect_leakage
from visual_inspection.data.scanner import scan_dataset


def test_cross_split_exact(dataset, image):
    path = dataset / "train/images/a.png"
    image.save(path)
    shutil.copyfile(path, dataset / "val/images/a.png")
    shutil.copyfile(path, dataset / "train/images/copy.png")
    frame = scan_dataset(dataset, AuditConfig())
    leakage = detect_leakage(exact_duplicates(frame), near_duplicates(frame, 5))
    assert len(leakage) == 2
    assert set(leakage.leakage_type) == {"exact_duplicate"}
    assert (leakage.source_split != leakage.target_split).all()

import shutil

from PIL import ImageEnhance

from visual_inspection.config import AuditConfig
from visual_inspection.data.duplicates import HammingIndex, exact_duplicates, near_duplicates
from visual_inspection.data.scanner import scan_dataset


def test_exact_and_near(dataset, image):
    first = dataset / "train/images/a.png"
    image.save(first)
    shutil.copyfile(first, dataset / "train/images/copy.png")
    ImageEnhance.Brightness(image).enhance(0.95).save(dataset / "val/images/modified.png")
    frame = scan_dataset(dataset, AuditConfig())
    assert len(exact_duplicates(frame)) == 1
    near = near_duplicates(frame, 5)
    assert len(near) == 2
    assert near.group_id.nunique() == 1
    assert near.distance.max() <= 5


def test_index_matches_brute_force():
    import random

    rng = random.Random(10)
    values = [rng.getrandbits(64) for _ in range(100)] + [0, 0, 1, 3]
    index = HammingIndex()
    for i, value in enumerate(values):
        index.add(value, i)
    for radius in (0, 5, 32, 64):
        for query in values[::10]:
            expected = {
                (i, (query ^ value).bit_count())
                for i, value in enumerate(values)
                if (query ^ value).bit_count() <= radius
            }
            assert set(index.query(query, radius)) == expected

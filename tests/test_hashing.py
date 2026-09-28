import shutil

from visual_inspection.data.hashing import phash_image, sha256_file


def test_identical_bytes(tmp_path, image):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    image.save(first)
    shutil.copyfile(first, second)
    assert sha256_file(first) == sha256_file(second)
    assert len(sha256_file(first)) == 64
    assert len(phash_image(image)) == 16

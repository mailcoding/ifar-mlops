import numpy as np
import pytest

from mlops.datasets.image_qa import image_stats, is_probable_mask, mask_score

Image = pytest.importorskip("PIL.Image")  # PIL = extra [train] ; skip proprement sans lui


def _save_mask(path):
    """Masque binaire : bloc blanc (255) sur fond noir (0)."""
    arr = np.zeros((48, 48), dtype=np.uint8)
    arr[12:36, 12:36] = 255
    Image.fromarray(arr, "L").save(path)


def _save_roi(path, seed=0):
    """Pseudo-ROI en niveaux de gris variés (aucun pixel aux extrêmes)."""
    arr = np.random.default_rng(seed).integers(40, 210, size=(48, 48), dtype=np.uint8)
    Image.fromarray(arr, "L").save(path)


def test_is_probable_mask_true_for_binary_mask(tmp_path):
    p = tmp_path / "mask.png"
    _save_mask(p)
    assert is_probable_mask(p) is True


def test_is_probable_mask_false_for_grayscale_roi(tmp_path):
    p = tmp_path / "roi.png"
    _save_roi(p)
    assert is_probable_mask(p) is False


def test_mask_score_higher_for_mask_than_roi(tmp_path):
    m, r = tmp_path / "m.png", tmp_path / "r.png"
    _save_mask(m)
    _save_roi(r)
    assert mask_score(m) > mask_score(r)


def test_image_stats_fields_and_ranges(tmp_path):
    p = tmp_path / "m.png"
    _save_mask(p)
    st = image_stats(p)
    assert set(st) == {"extreme_fraction", "entropy_bits", "n_levels"}
    assert 0.0 <= st["extreme_fraction"] <= 1.0
    assert st["entropy_bits"] >= 0.0

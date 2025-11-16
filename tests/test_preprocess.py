import numpy as np
import image_to_csv.preprocess as preprocess_module
from image_to_csv.preprocess import preprocess, deskew


def test_preprocess_returns_same_image_when_disabled():
    img = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    out = preprocess(img, do_clean=False)
    assert out is img


def test_preprocess_clean_produces_bgr_image(monkeypatch):
    img = np.full((6, 6, 3), 200, dtype=np.uint8)
    called = {}

    def fake_deskew(gray):
        called["gray"] = gray
        return gray

    monkeypatch.setattr(preprocess_module, "deskew", fake_deskew)
    out = preprocess(img, do_clean=True)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out is not img
    assert "gray" in called


def test_deskew_no_lines_returns_input():
    gray = np.full((10, 10), 255, dtype=np.uint8)
    out = deskew(gray)
    assert out.shape == gray.shape
    assert np.array_equal(out, gray)

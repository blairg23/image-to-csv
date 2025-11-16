import builtins
import sys
import types

import numpy as np
import pytest

from image_to_csv import ocr


def test_ocr_lines_tesseract_converts_text(monkeypatch):
    fake_module = types.SimpleNamespace(image_to_string=lambda img: "foo\nbar\n")
    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        types.SimpleNamespace(Image=types.SimpleNamespace(fromarray=lambda img: img)),
    )
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    lines = ocr.ocr_lines_tesseract(img)
    assert lines == ["foo", "bar"]


def test_ocr_lines_tesseract_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pytesseract":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="pytesseract not installed"):
        ocr.ocr_lines_tesseract(img)

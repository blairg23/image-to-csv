import sys
import types
import numpy as np

from image_to_csv import ocr


def test_ocr_table_paddle_caches_engine(monkeypatch):
    """Verify Paddle engine initializes once and is reused."""
    init_calls = []

    class FakeEngine:
        def __call__(self, img):
            return [{"type": "table", "res": {"html": "<table></table>"}}]

    def fake_engine_factory(*args, **kwargs):
        init_calls.append((args, kwargs))
        return FakeEngine()

    fake_module = types.SimpleNamespace(PPStructure=fake_engine_factory)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    ocr.reset_paddle_engine()
    img = np.zeros((2, 2, 3), dtype=np.uint8)

    first = ocr.ocr_table_paddle(img)
    second = ocr.ocr_table_paddle(img)

    assert first == "<table></table>"
    assert second == first
    assert len(init_calls) == 1
    ocr.reset_paddle_engine()


def test_ocr_table_paddle_debug_callback(monkeypatch):
    """Debug callbacks receive the raw Paddle response."""
    class FakeEngine:
        def __call__(self, img):
            return [{"type": "table", "res": {"html": "<table></table>"}}]

    def fake_factory(*args, **kwargs):
        return FakeEngine()

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PPStructure=fake_factory))
    ocr.reset_paddle_engine()
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    captured = []

    def debug_cb(result):
        captured.append(result)

    ocr.ocr_table_paddle(img, debug_callback=debug_cb)
    assert len(captured) == 1
    assert captured[0][0]["res"]["html"] == "<table></table>"

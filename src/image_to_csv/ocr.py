import inspect
from importlib import import_module
from typing import Callable, List, Optional
import numpy as np
import cv2

_paddle_engine = None


def _sanitize_img(img_bgr):
    if img_bgr is None:
        raise RuntimeError("Image is None (cv2.imread failed?)")
    if not isinstance(img_bgr, np.ndarray):
        raise RuntimeError(f"Expected numpy array, got {type(img_bgr)}")
    if img_bgr.size == 0 or img_bgr.ndim != 3:
        raise RuntimeError(f"Bad image shape: {img_bgr.shape}")
    if img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8, copy=False)
    return np.ascontiguousarray(img_bgr)


def _create_paddle_engine():
    """Create a fresh PaddleOCR engine, supporting PPStructure v2+."""
    try:
        paddleocr = import_module("paddleocr")
    except ImportError as e:
        raise RuntimeError(
            "PaddleOCR not installed. Run: poetry install --with paddle"
        ) from e
    except Exception as e:
        raise RuntimeError(f"PaddleOCR failed to initialize: {e}") from e

    engine_cls = None
    for attr in ("PPStructure", "PPStructureV3", "TableStructureRecognition"):
        engine_cls = getattr(paddleocr, attr, None)
        if engine_cls is not None:
            break
    if engine_cls is None:
        raise RuntimeError(
            "Installed PaddleOCR does not expose a table engine (PPStructure/PPStructureV3). Upgrade paddleocr."
        )

    kwargs = {}
    try:
        sig = inspect.signature(engine_cls)
        if "show_log" in sig.parameters:
            kwargs["show_log"] = False
        if "lang" in sig.parameters:
            kwargs["lang"] = "en"
    except (TypeError, ValueError):
        kwargs = {}

    try:
        return engine_cls(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create PaddleOCR engine: {e}") from e


def get_paddle_engine():
    """Return the cached PaddleOCR engine, instantiating it on first use."""
    global _paddle_engine
    if _paddle_engine is None:
        _paddle_engine = _create_paddle_engine()
    return _paddle_engine


def reset_paddle_engine():
    """Reset the cached PaddleOCR engine (used by tests or manual resets)."""
    global _paddle_engine
    _paddle_engine = None


def ocr_table_paddle(
    img_bgr,
    engine=None,
    debug_callback: Optional[Callable[[Optional[list]], None]] = None,
):
    """Run PaddleOCR table recognition and return extracted HTML if available.

    The Paddle engine is cached per process, so repeated calls avoid expensive
    re-initialization. Pass an explicit engine for tests or custom workflows.
    """
    img_bgr = _sanitize_img(img_bgr)
    ocr_engine = engine or get_paddle_engine()
    try:
        if callable(ocr_engine):
            result = ocr_engine(img_bgr)
        elif hasattr(ocr_engine, "predict"):
            result = ocr_engine.predict(img_bgr)
        elif hasattr(ocr_engine, "predict_iter"):
            result = list(ocr_engine.predict_iter(img_bgr))
        else:
            raise RuntimeError(
                "Unsupported PaddleOCR engine type; no callable/predict interface"
            )
    except Exception as e:
        raise RuntimeError(f"PaddleOCR inference failed: {e}") from e
    if debug_callback:
        try:
            debug_callback(result)
        except Exception:
            pass
    for item in result or []:
        res = item.get("res", {})
        if item.get("type") == "table" and isinstance(res, dict) and "html" in res:
            return res["html"]
    return None


def ocr_lines_tesseract(img_bgr) -> List[str]:
    """Fallback OCR line extraction using Tesseract."""
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        raise RuntimeError(
            "pytesseract not installed. Install with: poetry install -E tesseract"
        )
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(Image.fromarray(rgb))
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

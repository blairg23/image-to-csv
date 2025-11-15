from typing import List, Optional
import numpy as np
import cv2

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

def ocr_table_paddle(img_bgr):
    """Run PaddleOCR table recognition and return extracted HTML if available."""

    from typing import Optional
    import cv2

    # 1. Sanitize input image
    img_bgr = _sanitize_img(img_bgr)

    # 2. Lazy import and error handling
    try:
        from paddleocr import PPStructure
    except ImportError as e:
        raise RuntimeError("PaddleOCR not installed. Run: poetry install --with paddle") from e
    except Exception as e:
        raise RuntimeError(f"PaddleOCR failed to initialize: {e}") from e

    # 3. Initialize engine
    try:
        engine = PPStructure(show_log=False, lang="en")
    except Exception as e:
        raise RuntimeError(f"Failed to create PaddleOCR engine: {e}") from e

    # 4. Run OCR
    try:
        result = engine(img_bgr)
    except Exception as e:
        raise RuntimeError(f"PaddleOCR inference failed: {e}") from e

    # 5. Extract table HTML
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
        raise RuntimeError("pytesseract not installed. Install with: poetry install -E tesseract")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(Image.fromarray(rgb))
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

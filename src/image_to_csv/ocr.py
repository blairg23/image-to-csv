from typing import List, Optional
import cv2

def ocr_table_paddle(img_bgr) -> Optional[str]:
    """Use PaddleOCR structure model to extract HTML tables."""
    try:
        from paddleocr import PPStructure
    except Exception:
        raise RuntimeError("PaddleOCR not installed. Install with: poetry install -E paddle")
    engine = PPStructure(show_log=False, lang="en")
    result = engine(img_bgr)
    for item in result:
        if item.get("type") == "table" and "res" in item and "html" in item["res"]:
            return item["res"]["html"]
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

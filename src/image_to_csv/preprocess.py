import cv2
import numpy as np


def preprocess(img_bgr, do_clean=True):
    """Convert to grayscale, denoise, binarize, and deskew."""
    if not do_clean:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=20)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = deskew(th)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def deskew(gray_or_bin):
    """Estimate skew angle and rotate to fix it."""
    gray = (
        gray_or_bin
        if len(gray_or_bin.shape) == 2
        else cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
    )
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0.0
    if lines is not None:
        angles = []
        for rho_theta in lines[:50]:
            rho, theta = rho_theta[0]
            deg = theta * 180 / np.pi
            if 20 < deg < 160:
                angles.append(deg - 90)
        if angles:
            angle = float(np.median(angles))
    if abs(angle) < 0.2:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

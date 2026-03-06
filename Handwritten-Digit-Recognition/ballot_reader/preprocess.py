import cv2
import numpy as np
from typing import Tuple, Optional

def normalize_illumination(img_bgr: np.ndarray) -> np.ndarray:
    """Correct for uneven lighting/shadows using background estimation."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Estimate background via large-kernel median blur or morphological opening
    # Median blur is robust to noise and small text
    bg = cv2.medianBlur(gray, 51)
    
    # Divide original by background to normalize
    norm = cv2.divide(gray, bg, scale=255)
    
    # Optional: CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)
    
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

def deskew_image(img_bgr: np.ndarray, max_skew: float = 10.0, 
                 hough_thresh: int = 200) -> Tuple[np.ndarray, float]:
    """Deskew an image using Hough line detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)
    if lines is None:
        return img_bgr, 0.0
    
    angles = []
    for l in lines:
        rho, theta = l[0]
        angle = (theta - np.pi / 2) * 180.0 / np.pi
        if abs(angle) < max_skew:
            angles.append(angle)
            
    if not angles:
        return img_bgr, 0.0
        
    skew_angle = float(np.median(angles))
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew_angle, 1.0)
    rotated = cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, skew_angle

def denoise_image(img_bgr: np.ndarray) -> np.ndarray:
    """Apply denoising to the image."""
    # Fast non-local means denoising
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)

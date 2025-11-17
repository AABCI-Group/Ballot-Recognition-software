
import cv2, numpy as np
from typing import Tuple

def estimate_skew_angle(img_gray: np.ndarray) -> float:
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None: return 0.0
    angles = []
    for rho, theta in lines[:,0,:]:
        deg = (theta * 180/np.pi)
        if deg < 10 or deg > 170:
            a = deg if deg < 90 else deg-180
            angles.append(a)
    if not angles: return 0.0
    return float(np.median(angles))

def deskew_bgr(img_bgr: np.ndarray, max_angle: float=10.0) -> Tuple[np.ndarray, float]:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = np.clip(estimate_skew_angle(g), -max_angle, max_angle)
    h,w = g.shape
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    desk = cv2.warpAffine(img_bgr, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return desk, float(angle)

def normalize(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)
    return out

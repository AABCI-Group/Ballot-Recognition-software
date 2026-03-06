import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .config import BallotConfig

def order_quad_points(pts4: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def rectify_vote_box(vote_box_bgr: np.ndarray, config: BallotConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect printed vote-box border and warp to a canonical square."""
    h, w = vote_box_bgr.shape[:2]
    meta = {
        "rectify_ok": False,
        "method": None,
        "quad": None,
        "canonical_size": config.rectify_canonical_size,
    }
    
    if min(h, w) < 20:
        return vote_box_bgr, meta

    # Pre-process for edge detection
    gray = cv2.cvtColor(vote_box_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Detect edges
    edges = cv2.Canny(gray, config.rectify_canny1, config.rectify_canny2)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        meta["method"] = "no_contours"
        return vote_box_bgr, meta

    area_img = float(h * w)
    best_contour = None
    best_approx = None
    best_area = 0.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < config.rectify_min_area_frac * area_img:
            continue
            
        peri = cv2.arcLength(c, True)
        eps = config.rectify_poly_eps_frac * peri
        approx = cv2.approxPolyDP(c, eps, True)
        
        # Look for quads
        if len(approx) != 4:
            continue
            
        # Check aspect ratio of bounding rect
        pts = approx.reshape(-1, 2).astype(np.float32)
        x, y, ww, hh = cv2.boundingRect(pts.astype(np.int32))
        ar = ww / float(hh + 1e-6)
        if abs(ar - 1.0) > config.rectify_max_aspect_dev:
            continue
            
        if area > best_area:
            best_area = area
            best_contour = c
            best_approx = approx

    if best_approx is None:
        meta["method"] = "no_quad"
        return vote_box_bgr, meta

    # Order points and warp
    quad = order_quad_points(best_approx.reshape(-1, 2))
    meta["quad"] = quad.tolist()
    
    dst = np.array([
        [0, 0],
        [config.rectify_canonical_size - 1, 0],
        [config.rectify_canonical_size - 1, config.rectify_canonical_size - 1],
        [0, config.rectify_canonical_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(
        vote_box_bgr, M, (config.rectify_canonical_size, config.rectify_canonical_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    meta["rectify_ok"] = True
    meta["method"] = "warp_perspective"
    return warped, meta

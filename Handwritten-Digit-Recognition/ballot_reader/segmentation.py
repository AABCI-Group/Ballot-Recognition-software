import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .config import BallotConfig

def get_components(mask_bool: np.ndarray) -> List[Dict[str, Any]]:
    """Get connected components from a boolean mask."""
    fg = mask_bool.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area == 0:
            continue
        comp_mask = (labels == i)
        comps.append({
            "id": i,
            "x": x, "y": y, "w": w, "h": h,
            "area": area,
            "mask": comp_mask,
        })
    return comps

def skeletonize_bool(mask_bool: np.ndarray) -> np.ndarray:
    """Return skeleton (0/1) using morphological skeletonization."""
    img = (mask_bool.astype(np.uint8) * 255)
    if img.sum() == 0:
        return np.zeros_like(img, np.uint8)

    # Prefer OpenCV thinning if available
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        sk = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return (sk > 0).astype(np.uint8)

    # Fallback: morphological skeletonization
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)

def remove_border_components(enhanced_255: np.ndarray, config: BallotConfig) -> np.ndarray:
    """Remove components that touch the border and look like printed lines."""
    fg = (enhanced_255 < 128).astype(np.uint8)
    if fg.sum() == 0:
        return fg.astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    H, W = fg.shape
    keep = np.zeros_like(fg)

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_border = (x == 0 or y == 0 or x + w == W or y + h == H)

        # Heuristic for printed borders: long and thin
        is_long_horizontal = touches_border and (w >= 0.9 * W and h <= 0.15 * H)
        is_long_vertical   = touches_border and (h >= 0.9 * H and w <= 0.15 * W)

        if is_long_horizontal or is_long_vertical:
            continue

        keep[labels == i] = 1

    # If we removed everything, fallback to original
    if keep.sum() < 0.2 * fg.sum():
        keep = fg

    return keep.astype(bool)

def strip_thin_border_pieces(mask_bool: np.ndarray, ratio_thresh: float = 0.25) -> np.ndarray:
    """Strip thin pieces stuck to the border."""
    m = mask_bool.astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = m.shape
    cleaned = np.zeros_like(m)

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_left   = int(x == 0)
        touches_right  = int(x + w == W)
        touches_top    = int(y == 0)
        touches_bottom = int(y + h == H)
        touch_sides = touches_left + touches_right + touches_top + touches_bottom

        area_ratio = area / float(max(1, w * h))
        drop = (touch_sides >= 2) and (area_ratio < ratio_thresh)

        if not drop:
            cleaned[labels == i] = 1

    return cleaned.astype(bool)

def kill_outer_frame(mask_bool: np.ndarray, frame: int = 2) -> np.ndarray:
    """Zero out the outer frame of the mask."""
    m = mask_bool.copy()
    h, w = m.shape
    f = min(frame, h // 2, w // 2)
    if f <= 0:
        return m
    m[:f, :] = 0
    m[-f:, :] = 0
    m[:, :f] = 0
    m[:, -f:] = 0
    return m

def clean_components(mask_bool: np.ndarray, min_area: int = 8) -> np.ndarray:
    """Remove tiny components from the mask."""
    m = mask_bool.astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    
    H, W = m.shape
    scaled_min_area = max(min_area, int(0.0009 * H * W))
    
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cleaned = np.zeros_like(m)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < scaled_min_area:
            continue
        if max(w, h) <= 3 and area < (2 * scaled_min_area):
            continue
        cleaned[labels == i] = 1
    return cleaned.astype(bool)

def group_components_into_digits(comps: List[Dict[str, Any]], 
                                 overlap_thresh: float = 0.4) -> List[List[Dict[str, Any]]]:
    """Group components that overlap horizontally into digit candidates."""
    groups = []
    used = set()
    for i, c in enumerate(comps):
        if i in used:
            continue
        group = [c]
        used.add(i)
        for j, d in enumerate(comps):
            if j in used:
                continue
            # Horizontal overlap ratio
            x1a, x1b = c["x"], c["x"] + c["w"]
            x2a, x2b = d["x"], d["x"] + d["w"]
            inter = max(0, min(x1b, x2b) - max(x1a, x2a))
            overlap = inter / max(1.0, float(min(c["w"], d["w"])))
            
            if overlap >= overlap_thresh:
                group.append(d)
                used.add(j)
        groups.append(group)
    return groups

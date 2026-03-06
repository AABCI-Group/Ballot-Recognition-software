import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
from .config import BallotConfig
from .segmentation import get_components, skeletonize_bool

def score_digit_mask_basic(mask_bool: np.ndarray, config: BallotConfig) -> Tuple[float, Dict[str, Any]]:
    """Score a digit mask for digit-likeness."""
    if mask_bool is None or mask_bool.sum() == 0:
        return -999.0, {"reason": "empty"}

    H, W = mask_bool.shape
    ink = float(mask_bool.mean())
    
    if ink < config.enh_score_min_ink_frac:
        return -5.0, {"ink_frac": ink, "reason": "too_little_ink"}
    if ink > config.enh_score_max_ink_frac:
        return -2.0, {"ink_frac": ink, "reason": "too_much_ink"}

    comps = get_components(mask_bool)
    ncomp = len(comps)

    sk = skeletonize_bool(mask_bool)
    sk_len = int(sk.sum())

    # Junction count
    junc = 0
    if sk_len > 0:
        ys, xs = np.where(sk > 0)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y-1), min(H, y+2)
            x0, x1 = max(0, x-1), min(W, x+2)
            neigh = int(sk[y0:y1, x0:x1].sum()) - 1
            if neigh >= 3:
                junc += 1

    # Heuristic scoring
    score = 0.0
    score += 2.0 * min(1.0, ink / 0.06)
    score += 2.0 * min(1.0, sk_len / 80.0)
    score -= 0.4 * max(0, ncomp - 2)
    score -= 0.2 * min(10, junc)

    info = {
        "ink_frac": ink,
        "num_components": ncomp,
        "skeleton_len": sk_len,
        "junction_count": junc,
        "score": float(score)
    }
    return float(score), info

def enhance_vote_box(box_bgr: np.ndarray, config: BallotConfig, 
                     prefer_thin: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Multi-path enhancement (A/B/C) for vote boxes."""
    gray = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2GRAY)
    
    # Pre-denoise
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    # --- Path A: Adaptive threshold ---
    block = 21 if min(g.shape[:2]) < 90 else 31
    C = 11
    thA = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, block, C)
    maskA = _make_ink_mask(thA)

    # --- Path B: Otsu after normalization ---
    bg = cv2.medianBlur(g, 51)
    norm = cv2.divide(g, bg, scale=255)
    thB = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    maskB = _make_ink_mask(thB)
    
    # --- Path C: Thin-preserving (Rescue) ---
    thC = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, block, max(3, C - 6))
    maskC = _make_ink_mask(thC, minimal=True)
    
    # Score and choose
    sA, infoA = score_digit_mask_basic(maskA, config)
    sB, infoB = score_digit_mask_basic(maskB, config)
    sC, infoC = score_digit_mask_basic(maskC, config)
    
    # Selection logic
    candidates = [("A", sA, maskA, infoA), ("B", sB, maskB, infoB), ("C", sC, maskC, infoC)]
    
    if prefer_thin:
        # Boost C if we're looking for thin strokes
        best = max(candidates, key=lambda x: x[1] + (0.5 if x[0] == "C" else 0))
    else:
        best = max(candidates, key=lambda x: x[1])
        
    chosen_name, chosen_score, chosen_mask, chosen_info = best
    chosen_enhanced = np.where(chosen_mask, 0, 255).astype(np.uint8)

    meta = {
        "chosen": chosen_name,
        "score_A": sA, "score_B": sB, "score_C": sC,
        "info_A": infoA, "info_B": infoB, "info_C": infoC,
    }
    
    return chosen_enhanced, chosen_mask, meta

def _make_ink_mask(bin_inv_255: np.ndarray, minimal: bool = False) -> np.ndarray:
    """Convert binary inverse image to a clean boolean ink mask."""
    m = (bin_inv_255 > 0).astype(np.uint8)
    
    # Small OPEN to remove specks
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, 
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), 1)
    
    if not minimal:
        # Mild CLOSE to reconnect broken strokes
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, 
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
        
        # Diagonal-friendly close
        diag1 = np.eye(3, dtype=np.uint8)
        diag2 = np.fliplr(diag1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, diag1, 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, diag2, 1)
        
    return m.astype(bool)

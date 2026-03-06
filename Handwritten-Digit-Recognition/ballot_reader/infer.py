import os
import json
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List
from .config import BallotConfig
from .mnist import mask_to_mnist28
from .segmentation import skeletonize_bool, get_components

def load_mnist28_model(model_path: str):
    """Load the Keras MNIST model."""
    from tensorflow.keras import models
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return models.load_model(model_path)

def apply_calibration(probs: np.ndarray, calib: Optional[Dict[str, Any]]) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
    if calib is None:
        return probs
    T = float(calib.get("temperature", 1.0))
    if T <= 0:
        return probs
    p = np.clip(probs.astype(np.float32), 1e-8, 1.0)
    logits = np.log(p) / T
    expv = np.exp(logits - np.max(logits))
    return expv / float(np.sum(expv) + 1e-12)

def entropy(probs: np.ndarray) -> float:
    """Compute entropy of probability distribution."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))



def _save_mnist_debug(x: np.ndarray, config: BallotConfig, tag: str):
    if not getattr(config, "debug_dir", None):
        return
    os.makedirs(config.debug_dir, exist_ok=True)
    img = (np.clip(x[0, :, :, 0], 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(config.debug_dir, f"mnist28_{tag}.png"), img)

def classify_with_variants(model, mask_bool: np.ndarray, config: BallotConfig,
                           calib: Optional[Dict[str, Any]] = None) -> Tuple[Optional[int], np.ndarray, Dict[str, Any]]:
    """Classify a digit mask using multiple preprocessing variants with sane gating."""
    ink_frac = float(mask_bool.mean())

    variants = [
        ("v0_nodil_bin", mask_to_mnist28(mask_bool, margin=4, dilate_px=0, resize_mode="binary_nearest_blur")),
        ("v1_d1_bin",    mask_to_mnist28(mask_bool, margin=4, dilate_px=1, resize_mode="binary_nearest_blur")),
        ("v2_oldish",    mask_to_mnist28(mask_bool, margin=4, dilate_px=1, resize_mode="gray_area_blur")),
        # aggressive: keep, but gate/penalize
        ("v3_d2_bin",    mask_to_mnist28(mask_bool, margin=3, dilate_px=2, resize_mode="binary_nearest_blur")),
        ("v4_d2_gray",   mask_to_mnist28(mask_bool, margin=3, dilate_px=2, resize_mode="gray_area_blur")),
    ]

    all_meta: List[Dict[str, Any]] = []

    # First pass: run model on all variants, collect meta
    for name, x in variants:
        if x is None:
            continue

        _save_mnist_debug(x, config, name)

        probs = model.predict(x, verbose=0)[0].astype(np.float32)
        probs = probs / float(np.sum(probs) + 1e-12)
        probs = apply_calibration(probs, calib)

        pred = int(np.argmax(probs))

        # Proper top1/top2 margin
        order = np.argsort(-probs)
        top1 = float(probs[order[0]])
        top2 = float(probs[order[1]])
        margin = float(top1 - top2)
        conf = top1
        ent = entropy(probs)

        top = order[:3]
        print(name, [(int(i), float(probs[i])) for i in top])

        meta = {
            "variant": name, "pred": pred, "conf": conf, "margin": margin,
            "entropy": ent, "probs": probs,
        }
        all_meta.append(meta)

    if not all_meta:
        return None, np.zeros(10), {"variants": []}

    # Identify "safe" variants
    SAFE = {"v0_nodil_bin", "v1_d1_bin", "v2_oldish"}
    safe_metas = [m for m in all_meta if m["variant"] in SAFE]
    agg_metas  = [m for m in all_meta if m["variant"] not in SAFE]

    def base_score(m):
        return (2.2 * m["margin"]) + (1.0 * m["conf"]) - (0.35 * max(0.0, m["entropy"] - 1.6))

    best_safe = max(safe_metas, key=base_score) if safe_metas else None
    if best_safe is None:
        best = max(all_meta, key=base_score)
        return int(best["pred"]), best["probs"], {"best": best, "variants": all_meta}

    # Start with best safe as the answer
    best = best_safe
    best_score = base_score(best_safe)

    # Aggressive variants can ONLY refine/confirm the SAME prediction
    for m in agg_metas:
        s = base_score(m)
        m["score"] = float(s)

        if m["pred"] == best_safe["pred"] and s > best_score:
            best = m
            best_score = s

    return int(best["pred"]), best["probs"], {"best": best, "variants": all_meta}

def score_seven_vs_four(mask_bool: np.ndarray) -> Tuple[bool, bool, Dict[str, Any], Dict[str, Any]]:
    """Heuristic scoring for 4 vs 7 ambiguity."""
    H, W = mask_bool.shape
    ink_frac = float(mask_bool.mean())
    
    # Skeleton and junctions
    sk = skeletonize_bool(mask_bool)
    sk_len = int(sk.sum())
    
    # Top bar presence
    top_h = max(1, int(0.20 * H))
    top_band = mask_bool[:top_h, :]
    top_bar = float(top_band.sum(axis=1).max() / float(max(1, W)))
    
    # Left vertical stroke presence
    left_w = max(1, int(0.25 * W))
    left_band = mask_bool[:, :left_w]
    left_score = float(left_band.mean() / (ink_frac + 1e-6))
    
    # 7 score: strong top bar, weak left stroke
    seven_score = 0.5 * top_bar - 0.3 * left_score
    # 4 score: weak top bar, strong left stroke
    four_score = 0.5 * left_score - 0.3 * top_bar
    
    features = {
        "top_bar": top_bar,
        "left_score": left_score,
        "seven_score": seven_score,
        "four_score": four_score,
        "sk_len": sk_len,
    }
    
    return seven_score > 0.4, four_score > 0.4, features, {}

def is_blank_or_noise(mask_bool: np.ndarray, config: BallotConfig) -> Tuple[bool, Dict[str, Any]]:
    """Determine if a mask is blank or just noise."""
    H, W = mask_bool.shape
    ink_px = int(mask_bool.sum())
    ink_frac = float(ink_px / float(max(1, H * W)))
    
    reasons = {"ink_px": ink_px, "ink_frac": ink_frac}
    
    if ink_px < config.blank_min_ink_px_abs or ink_frac < config.blank_min_ink_frac:
        reasons["min_ink"] = True
        return True, reasons
        
    # Bounding box metrics
    ys, xs = np.where(mask_bool)
    if len(xs) == 0: return True, {"empty": True}
    bb_w, bb_h = (xs.max() - xs.min() + 1), (ys.max() - ys.min() + 1)
    bb_w_frac, bb_h_frac = bb_w / W, bb_h / H
    
    reasons.update({"bb_w_frac": bb_w_frac, "bb_h_frac": bb_h_frac})
    
    if bb_w_frac < config.blank_min_bbox_w_frac or bb_h_frac < config.blank_min_bbox_h_frac:
        # Check thin digit exception (e.g., "1")
        if not (bb_h_frac > 0.55 and bb_w / bb_h < 0.3):
            reasons["small_bbox"] = True
            return True, reasons
            
    return False, reasons

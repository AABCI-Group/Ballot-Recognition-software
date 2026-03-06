import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

def mask_to_mnist28(mask_bool: np.ndarray, margin: int = 4, dilate_px: int = 0, 
                    resize_mode: str = "binary_nearest_blur") -> Optional[np.ndarray]:
    """Convert a boolean mask to a 28x28 MNIST-style input tensor."""
    if mask_bool is None or mask_bool.sum() == 0:
        return None

    m = (mask_bool.astype(np.uint8) * 255)
    ys, xs = np.where(mask_bool)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    crop = m[y1:y2+1, x1:x2+1]

    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        crop = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)

    target_inner = 28 - 2 * margin
    h, w = crop.shape
    scale = min(target_inner / float(max(h, w)), 1.0) if max(h, w) > 0 else 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    canvas = np.zeros((28, 28), np.uint8)

    if resize_mode == "binary_nearest_blur":
        bin01 = (crop > 0).astype(np.uint8) * 255
        resized = cv2.resize(bin01, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        resized = cv2.GaussianBlur(resized, (3, 3), 0)
    else:
        interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
        resized = cv2.GaussianBlur(resized, (3, 3), 0)

    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    # Gentle centering
    bin01 = (canvas > 0).astype(np.uint8)
    M = cv2.moments(bin01, binaryImage=True)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dx = int(np.clip(round(14 - cx), -2, 2))
        dy = int(np.clip(round(14 - cy), -2, 2))
        if dx != 0 or dy != 0:
            Mshift = np.float32([[1, 0, dx], [0, 1, dy]])
            canvas = cv2.warpAffine(canvas, Mshift, (28, 28), 
                                    flags=cv2.INTER_CUBIC, borderValue=0)

    x = (canvas.astype(np.float32) / 255.0)
    x = np.clip(x, 0.0, 1.0)

    return x[None, ..., None]

def stable_crop(mask_bool: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Try two pads and keep the crop that preserves topology better."""
    def _adaptive_tight_crop(m_bool, pad_frac=0.08, pad_min=2, pad_max=10):
        ys, xs = np.where(m_bool)
        if len(xs) == 0: return None, None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bb_w, bb_h = (x2 - x1 + 1), (y2 - y1 + 1)
        pad = max(pad_min, int(round(pad_frac * max(bb_w, bb_h))))
        pad = min(pad, pad_max)
        H, W = m_bool.shape
        xa, xb = max(0, x1 - pad), min(W - 1, x2 + pad)
        ya, yb = max(0, y1 - pad), min(H - 1, y2 + pad)
        crop = m_bool[ya:yb+1, xa:xb+1].copy()
        return crop, {"pad": int(pad), "bbox": [x1, y1, x2, y2], "crop_box": [xa, ya, xb, yb]}

    c1, m1 = _adaptive_tight_crop(mask_bool, pad_frac=0.06)
    c2, m2 = _adaptive_tight_crop(mask_bool, pad_frac=0.12)
    
    if c1 is None: return c2, {"chosen": "pad12", "meta": m2}
    if c2 is None: return c1, {"chosen": "pad06", "meta": m1}
    
    # Topology check (skeleton length)
    from .segmentation import skeletonize_bool, get_components
    sk1 = skeletonize_bool(c1).sum()
    sk2 = skeletonize_bool(c2).sum()
    
    if sk2 > sk1 * 1.08:
        return c2, {"chosen": "pad12", "meta": m2}
    return c1, {"chosen": "pad06", "meta": m1}

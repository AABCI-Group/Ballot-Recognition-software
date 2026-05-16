
import cv2, numpy as np
from dataclasses import dataclass

@dataclass
class GeoConfig:
    min_area_frac: float
    max_area_frac: float
    min_circularity: float
    min_ellipse_edge_pct: float
    min_template_ncc: float
    min_blob_count: int = 0
    min_blob_spread: float = 0.0

def _binary(mask_or_gray):
    if len(mask_or_gray.shape)==2:
        g = mask_or_gray
    else:
        g = cv2.cvtColor(mask_or_gray, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)[1]
    return thr

def circularity(mask: np.ndarray) -> float:
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    c = max(cnts, key=cv2.contourArea)
    A = cv2.contourArea(c); P = cv2.arcLength(c, True)
    if P==0: return 0.0
    return float(4*np.pi*A/(P*P))

def ellipse_edge_pct(mask: np.ndarray) -> float:
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    c = max(cnts, key=cv2.contourArea)
    if len(c)<5: return 0.0
    (x,y),(MA,ma),angle = cv2.fitEllipse(c)
    ell_mask = np.zeros_like(mask)
    cv2.ellipse(ell_mask, ((x,y),(MA,ma),angle), 255, thickness=2)
    overlap = (ell_mask>0) & (mask>0)
    return float(overlap.sum()/max(1,(ell_mask>0).sum()))

def template_ncc(crop: np.ndarray, template: np.ndarray) -> float:
    if template is None: return 1.0
    h,w = crop.shape[:2]
    th,tw = template.shape[:2]
    scale = max(8, min(h,w)) / max(th,tw)
    tmp = cv2.resize(template, (max(1,int(tw*scale)), max(1,int(th*scale))))
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_tmp  = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    if gray_crop.shape[0]<gray_tmp.shape[0] or gray_crop.shape[1]<gray_tmp.shape[1]:
        return 0.0
    res = cv2.matchTemplate(gray_crop, gray_tmp, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) if res.size else 0.0

def stamp_blob_features(crop_bgr: np.ndarray) -> dict:
    """
    Official stamp impressions are a cluster of small round grey/black blobs.
    This rejects common ballot false positives such as logos, faces, text, and
    handwritten digits that may have a YOLO-like box but no dot cluster.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return {"blob_count": 0, "blob_spread": 0.0}

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel = np.ones((2, 2), dtype=np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = float(h * w)
    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < max(4.0, crop_area * 0.002) or area > crop_area * 0.22:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circ = float(4 * np.pi * area / (perimeter * perimeter))
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(1.0, float(bh))
        if circ < 0.28 or aspect < 0.35 or aspect > 2.8:
            continue
        centers.append((x + bw / 2.0, y + bh / 2.0))

    if len(centers) < 2:
        return {"blob_count": len(centers), "blob_spread": 0.0}

    pts = np.array(centers, dtype=np.float32)
    spread_x = (pts[:, 0].max() - pts[:, 0].min()) / max(1.0, float(w))
    spread_y = (pts[:, 1].max() - pts[:, 1].min()) / max(1.0, float(h))
    return {
        "blob_count": len(centers),
        "blob_spread": float(max(spread_x, spread_y)),
    }

def passes_geometry(crop_bgr: np.ndarray, img_shape, bbox, cfg: GeoConfig, template_bgr=None):
    H,W = img_shape[:2]
    x0,y0,x1,y1 = bbox
    area_frac = ((x1-x0)*(y1-y0))/(W*H + 1e-9)
    if area_frac<cfg.min_area_frac or area_frac>cfg.max_area_frac:
        return False, {"area_frac": area_frac}
    mask = _binary(crop_bgr)
    circ = circularity(mask)
    ellp = ellipse_edge_pct(mask)
    sim  = template_ncc(crop_bgr, template_bgr) if template_bgr is not None else 1.0
    blob_feats = stamp_blob_features(crop_bgr)
    feats = {"area_frac": area_frac, "circularity": circ, "ellipse_edge_pct": ellp, "template_ncc": sim, **blob_feats}
    ok = (
        (circ >= cfg.min_circularity)
        and (ellp >= cfg.min_ellipse_edge_pct)
        and (sim >= cfg.min_template_ncc)
        and (blob_feats["blob_count"] >= cfg.min_blob_count)
        and (blob_feats["blob_spread"] >= cfg.min_blob_spread)
    )
    return ok, feats

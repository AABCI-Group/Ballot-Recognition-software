
import cv2, numpy as np
from dataclasses import dataclass

@dataclass
class GeoConfig:
    min_area_frac: float
    max_area_frac: float
    min_circularity: float
    min_ellipse_edge_pct: float
    min_template_ncc: float

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
    feats = {"area_frac": area_frac, "circularity": circ, "ellipse_edge_pct": ellp, "template_ncc": sim}
    ok = (circ>=cfg.min_circularity) and (ellp>=cfg.min_ellipse_edge_pct) and (sim>=cfg.min_template_ncc)
    return ok, feats

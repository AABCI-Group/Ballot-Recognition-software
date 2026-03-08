import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class CropResult:
    input_path: str
    output_path: str
    debug_dir: Optional[str]
    bbox: Tuple[int, int, int, int]
    used_fallback: bool


def _normalize_lighting_gray(gray: np.ndarray, blur_sigma: float = 25.0) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    normalized = cv2.divide(gray, bg, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _angle(p1: np.ndarray, p2: np.ndarray, p0: np.ndarray) -> float:
    dx1 = float(p1[0] - p0[0])
    dy1 = float(p1[1] - p0[1])
    dx2 = float(p2[0] - p0[0])
    dy2 = float(p2[1] - p0[1])
    denom = np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)
    return (dx1 * dx2 + dy1 * dy2) / denom


def _is_rect_quad(quad_points: np.ndarray, max_cosine_threshold: float = 0.35) -> bool:
    max_cosine = 0.0
    pts = quad_points.reshape(4, 2)
    for i in range(2, 5):
        cosine = abs(_angle(pts[i % 4], pts[i - 2], pts[i - 1]))
        max_cosine = max(max_cosine, cosine)
    return max_cosine < max_cosine_threshold


def _score_contour(contour: np.ndarray, image_shape: Tuple[int, int, int], min_area_ratio: float) -> float:
    h, w = image_shape[:2]
    img_area = float(h * w)
    area = float(cv2.contourArea(contour))
    if area < img_area * min_area_ratio:
        return -1.0

    x, y, cw, ch = cv2.boundingRect(contour)
    rect_area = float(max(1, cw * ch))
    fill_ratio = area / rect_area
    if fill_ratio < 0.28:
        return -1.0

    if area > 0.95 * img_area and fill_ratio > 0.90:
        return -1.0

    margin_x = max(8, int(0.01 * w))
    margin_y = max(8, int(0.01 * h))
    touches = 0
    if x <= margin_x:
        touches += 1
    if y <= margin_y:
        touches += 1
    if (x + cw) >= (w - margin_x):
        touches += 1
    if (y + ch) >= (h - margin_y):
        touches += 1

    border_score = 1.0
    if touches >= 2:
        border_score = 0.06
    elif touches == 1:
        border_score = 0.42

    aspect = max(cw, ch) / float(max(1, min(cw, ch)))
    aspect_score = max(0.35, 1.0 - abs(aspect - 1.65) / 2.3)

    m = cv2.moments(contour)
    if abs(m.get("m00", 0.0)) < 1e-6:
        center_score = 0.2
    else:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dist = np.hypot(cx - (w / 2.0), cy - (h / 2.0))
        diag = max(1.0, np.hypot(w, h))
        center_score = max(0.2, 1.0 - (dist / diag) * 1.7)

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    minrect_area = float(max(1.0, rw * rh))
    rectangularity = max(0.2, min(1.0, area / minrect_area))

    return area * fill_ratio * aspect_score * border_score * center_score * rectangularity


def _pick_best_contour(mask: np.ndarray, image_shape: Tuple[int, int, int], min_area_ratio: float) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for contour in contours:
        score = _score_contour(contour, image_shape, min_area_ratio)
        if score > best_score:
            best_score = score
            best = contour
    return best


def _find_ballot_contour_omrchecker(image: np.ndarray, min_area_ratio: float = 0.18) -> Optional[np.ndarray]:
    # OMRChecker CropPage style detector.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm = _normalize_lighting_gray(gray, blur_sigma=18.0)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    _, trunc = cv2.threshold(blur, 200, 255, cv2.THRESH_TRUNC)
    trunc = cv2.normalize(trunc, None, 0, 255, cv2.NORM_MINMAX)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed = cv2.morphologyEx(trunc, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Canny(closed, 55, 185)

    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(image.shape[0] * image.shape[1])
    min_area = max(min_area_ratio * img_area, 80000.0)
    hulls = [cv2.convexHull(c) for c in contours]
    hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:12]

    best = None
    best_score = -1.0
    for contour in hulls:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        if not _is_rect_quad(approx.reshape(4, 2)):
            continue
        score = _score_contour(contour, image.shape, min_area_ratio)
        if score > best_score:
            best_score = score
            best = contour
    return best


def _find_ballot_contour_edges(image: np.ndarray, min_area_ratio: float = 0.18) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = _normalize_lighting_gray(gray)
    blur = cv2.GaussianBlur(normalized, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _pick_best_contour(edges, image.shape, min_area_ratio)


def _find_ballot_contour_brightness(image: np.ndarray, min_area_ratio: float = 0.18) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = _normalize_lighting_gray(gray)
    blur = cv2.GaussianBlur(normalized, (7, 7), 0)

    otsu_t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(blur, min(245, int(otsu_t + 4)), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _pick_best_contour(mask, image.shape, min_area_ratio)


def _find_ballot_contour_white_chroma(image: np.ndarray, min_area_ratio: float = 0.18) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    norm_val = _normalize_lighting_gray(val, blur_sigma=21.0)

    otsu_t, _ = cv2.threshold(norm_val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    val_th = max(122, int(otsu_t - 8))
    sat_th = int(min(100, np.percentile(sat, 72) + 18))
    white_like = ((norm_val >= val_th) & (sat <= sat_th)).astype(np.uint8) * 255

    # Preserve highlights and glare as paper.
    highlight = ((val >= 245) & (sat <= 70)).astype(np.uint8) * 255
    white_like = cv2.bitwise_or(white_like, highlight)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    white_like = cv2.morphologyEx(white_like, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_like = cv2.morphologyEx(white_like, cv2.MORPH_OPEN, kernel, iterations=1)
    return _pick_best_contour(white_like, image.shape, min_area_ratio)


def _contour_to_quad(contour: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return None

    for eps in (0.02, 0.025, 0.03):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        if _is_rect_quad(quad):
            return _order_points(quad)
    return None


def _find_ballot_quad_edges(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Edge/shape based ballot detector that is robust across lighting.

    Strategy:
    - Work on a resized copy for stability.
    - Use adaptive Canny edges and morphology.
    - Look for large convex quadrilaterals with a plausible aspect ratio.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    # Downscale for contour detection to make things more stable and faster.
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > 1400:
        scale = 1400.0 / float(max_dim)
    elif max_dim < 600:
        scale = 600.0 / float(max_dim)

    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Canny thresholds based on image statistics.
    v = float(np.median(gray))
    lower = int(max(10.0, 0.66 * v))
    upper = int(min(255.0, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(rh * rw)
    best_quad = None
    best_score = -1.0

    icx, icy = rw / 2.0, rh / 2.0
    diag = max(1.0, np.hypot(rw, rh))

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < 0.08 * img_area or area > 0.90 * img_area:
            continue

        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        x, y, cw, ch = cv2.boundingRect(approx)
        aspect = max(cw, ch) / float(max(1, min(cw, ch)))
        # Ballot is a tall rectangle; allow a range but exclude near‑squares and extreme slivers.
        if aspect < 1.5 or aspect > 7.0:
            continue

        # Penalize quads that hug the image border (likely the whole frame).
        margin = int(0.04 * max(rh, rw))
        touches = 0
        if x <= margin:
            touches += 1
        if y <= margin:
            touches += 1
        if (x + cw) >= (rw - margin):
            touches += 1
        if (y + ch) >= (rh - margin):
            touches += 1
        if touches >= 3:
            border_penalty = 0.2
        elif touches == 2:
            border_penalty = 0.45
        else:
            border_penalty = 1.0

        # Prefer quads whose center is near the image center.
        cx = x + cw / 2.0
        cy = y + ch / 2.0
        dist = np.hypot(cx - icx, cy - icy)
        center_score = max(0.3, 1.0 - (dist / (0.7 * diag)))

        # Prefer tall portrait orientation.
        tall = ch >= cw
        orient_score = 1.0 if tall else 0.6

        score = area * border_penalty * center_score * orient_score
        if score > best_score:
            best_score = score
            best_quad = approx.reshape(4, 2).astype(np.float32)

    if best_quad is None:
        return None

    # Map back to original image coordinates.
    inv_scale = 1.0 / scale
    quad_fullres = best_quad * inv_scale
    return _order_points(quad_fullres)


def _find_ballot_quad_paper_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Ballot quad detector driven by the LAB-based paper mask.

    This is especially helpful when the ballot is small in the frame or
    lighting makes edge-based detection unstable.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    mask = _build_paper_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    img_area = float(h * w)
    area_frac = area / max(1.0, img_area)
    # Ignore tiny regions; the ballot should occupy a reasonable fraction.
    if area_frac < 0.08:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    ordered = _order_points(box)

    # Basic aspect-ratio sanity check.
    (tl, tr, br, bl) = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    width = max(width_a, width_b)
    height = max(height_a, height_b)
    aspect = max(width, height) / float(max(1.0, min(width, height)))
    if aspect < 1.4 or aspect > 8.5:
        return None

    return ordered


def _warp_from_quad(image: np.ndarray, ordered_quad: np.ndarray) -> np.ndarray:
    ordered = _order_points(ordered_quad.astype(np.float32))

    tl, tr, br, bl = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = max(2, int(max(width_a, width_b)))
    max_height = max(2, int(max(height_a, height_b)))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(
        image,
        M,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


def _warp_from_contour(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    ordered = _contour_to_quad(contour)
    if ordered is None:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        ordered = _order_points(box)
    return _warp_from_quad(image, ordered)


def _tight_bbox_from_mask(mask: np.ndarray, pad: int = 0) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + pad)
    return x0, y0, x1, y1


def _trim_uniform_border(warped: np.ndarray, white_threshold: int = 250) -> np.ndarray:
    """
    Remove uniform white border around the warped ballot.

    The perspective warp fills everything outside the quadrilateral with
    pure white, so we can simply crop to the tightest box that contains
    non‑white pixels.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    non_white = gray < white_threshold
    ys, xs = np.where(non_white)
    if ys.size == 0 or xs.size == 0:
        return warped

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # Small padding to avoid cutting right at content.
    pad = 4
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(warped.shape[0], y1 + pad)
    x1 = min(warped.shape[1], x1 + pad)
    return warped[y0:y1, x0:x1]


def _refine_with_paper_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    On the already warped ballot, build a lighting‑robust paper mask and
    crop tightly to the main page region, whitening any residual background.

    This runs after geometric rectification so the ballot is roughly
    axis‑aligned and fills most of the frame, which makes mask‑based
    refinement more stable across different lighting and backgrounds.
    """
    h, w = image.shape[:2]
    paper_mask = _build_paper_mask(image)

    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, np.full((h, w), 255, dtype=np.uint8)

    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # If the detected page area is too small, or the bbox collapses to a
    # thin strip, trust the geometric warp instead of over‑cropping.
    area_frac = cv2.contourArea(contour) / float(max(1, h * w))
    x0, y0, x1, y1 = _tight_bbox_from_mask(mask, pad=int(0.01 * max(h, w)))
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    if area_frac < 0.45 or bw < int(0.55 * w) or bh < int(0.55 * h):
        return image, np.full((h, w), 255, dtype=np.uint8)

    # Whiten non‑paper pixels and crop tightly around the page.
    isolated = image.copy()
    isolated[mask == 0] = (255, 255, 255)
    return isolated[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _build_paper_mask(image: np.ndarray) -> np.ndarray:
    """
    Build a binary mask of the ballot paper using LAB "whiteness".

    The ballot is nearly neutral (low chroma) and brighter than the
    dark, textured background (granite table, desk, etc). We work in
    LAB space to separate lightness from chroma and then keep the
    largest bright, low‑chroma region.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    A = lab[:, :, 1].astype(np.float32) - 128.0
    B = lab[:, :, 2].astype(np.float32) - 128.0

    # Normalize lightness to compensate for shading.
    norm_L = _normalize_lighting_gray(L, blur_sigma=17.0)
    norm_L = cv2.GaussianBlur(norm_L, (5, 5), 0)

    chroma = np.sqrt(A * A + B * B)

    # Dynamic thresholds: ballot is among the brighter, low‑chroma pixels.
    L_thr = max(160.0, float(np.percentile(norm_L, 65)))
    C_thr = min(25.0, float(np.percentile(chroma, 45)))

    paper = ((norm_L >= L_thr) & (chroma <= C_thr)).astype(np.uint8) * 255

    # Morphology: close gaps along edges, then open to drop small speckles.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, kernel_open, iterations=1)
    return paper


def _trim_warped_to_page_axes(warped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Use row/column white-like occupancy to estimate the page extent
    # and then explicitly remove any non-paper pixels using the mask.
    mask = _build_paper_mask(warped)
    h, w = mask.shape[:2]
    row_ratio = mask.mean(axis=1) / 255.0
    col_ratio = mask.mean(axis=0) / 255.0

    row_thr = float(np.clip(np.percentile(row_ratio, 55), 0.45, 0.92))
    col_thr = float(np.clip(np.percentile(col_ratio, 55), 0.45, 0.92))

    rows = np.where(row_ratio >= row_thr)[0]
    cols = np.where(col_ratio >= col_thr)[0]

    # If we cannot find a reliable axis trim, still remove background
    # using the paper mask over the full warped image.
    if rows.size == 0 or cols.size == 0:
        isolated = warped.copy()
        isolated[mask == 0] = (255, 255, 255)
        return isolated, mask

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    # Guardrails: never return an over-tight crop. If the proposed crop
    # is too small, keep the full warped image but still strip background.
    if (y1 - y0) < int(0.60 * h) or (x1 - x0) < int(0.45 * w):
        isolated = warped.copy()
        isolated[mask == 0] = (255, 255, 255)
        return isolated, mask

    trim_mask = mask[y0:y1, x0:x1]
    trimmed = warped[y0:y1, x0:x1].copy()
    trimmed[trim_mask == 0] = (255, 255, 255)
    return trimmed, trim_mask


def _rectify_and_crop_paper(warped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Final cleanup step on the warped ballot:
    - Trim uniform white border introduced by the geometric warp using
      a simple non‑white occupancy test.

    We intentionally keep masking light‑touch here so that warping /
    quad selection does the heavy lifting, and we avoid "chewing" into
    the ballot under difficult lighting.
    """
    trimmed = _trim_uniform_border(warped)
    mask = np.full(trimmed.shape[:2], 255, dtype=np.uint8)
    return trimmed, mask


def _bbox_from_contour(contour: np.ndarray, shape: Tuple[int, int, int], pad_frac: float = 0.004) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    pad_x = int(cw * pad_frac)
    pad_y = int(ch * pad_frac)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + cw + pad_x)
    y1 = min(h, y + ch + pad_y)
    return x0, y0, x1, y1


def crop_ballot_paper(input_path: str, output_path: str, debug_dir: Optional[str] = None) -> CropResult:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input image not found: {src}")

    image = cv2.imread(str(src))
    if image is None:
        raise ValueError(f"Failed to read image: {src}")

    # We track a contour only in the legacy fallback path so that debug
    # drawing logic can check it safely.
    contour: Optional[np.ndarray] = None

    # 1) Prefer purely geometric, edge-based quad detection.
    quad = _find_ballot_quad_edges(image)
    detector = "edge_quad"
    used_fallback = quad is None

    # 2) If edge-based quad fails, try LAB paper-mask based quad.
    if quad is None:
        quad = _find_ballot_quad_paper_mask(image)
        if quad is not None:
            detector = "lab_paper_quad"
            used_fallback = False

    if quad is not None:
        warped = _warp_from_quad(image, quad)
        x0, y0, x1, y1 = _tight_bbox_from_mask(
            cv2.fillPoly(np.zeros(image.shape[:2], dtype=np.uint8), [quad.astype(np.int32)], 255)
        )
        bbox = (x0, y0, x1, y1)
    else:
        # 3) Fallback to legacy contour-based detectors (multi-cue).
        c_omr = _find_ballot_contour_omrchecker(image)
        c_edge = _find_ballot_contour_edges(image)
        c_bright = _find_ballot_contour_brightness(image)
        c_white = _find_ballot_contour_white_chroma(image)
        candidates = [c for c in (c_omr, c_edge, c_bright, c_white) if c is not None]

        detector = "fallback_full_frame"
        if candidates:
            contour = max(candidates, key=lambda c: _score_contour(c, image.shape, min_area_ratio=0.18))
            if c_omr is not None and contour is c_omr:
                detector = "omr_crop_page"
            elif c_edge is not None and contour is c_edge:
                detector = "edge_mask"
            elif c_bright is not None and contour is c_bright:
                detector = "brightness_mask"
            else:
                detector = "white_chroma_mask"

        if contour is None:
            h, w = image.shape[:2]
            bbox = (0, 0, w, h)
            warped = image.copy()
        else:
            bbox = _bbox_from_contour(contour, image.shape)
            warped = _warp_from_contour(image, contour)

    isolated, paper_mask = _rectify_and_crop_paper(warped)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), isolated):
        raise RuntimeError(f"Failed to write cropped image: {out_path}")

    debug_path: Optional[Path] = None
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(debug_path / "original_input.png"), image)

        detected = image.copy()
        if contour is not None:
            cv2.drawContours(detected, [contour], -1, (0, 255, 0), 3)
            rot_box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
            cv2.polylines(detected, [rot_box], True, (255, 0, 255), 3)
        x0, y0, x1, y1 = bbox
        cv2.rectangle(detected, (x0, y0), (x1, y1), (255, 0, 0), 2)

        cv2.imwrite(str(debug_path / "detected_region.png"), detected)
        cv2.imwrite(str(debug_path / "warped_before_mask.png"), warped)
        cv2.imwrite(str(debug_path / "paper_mask.png"), paper_mask)
        cv2.imwrite(str(debug_path / "final_cropped_output.png"), isolated)

        meta = {
            "input_path": str(src),
            "output_path": str(out_path),
            "bbox_xyxy": [int(v) for v in bbox],
            "used_fallback_full_image": used_fallback,
            "cropped_shape_hw": [int(isolated.shape[0]), int(isolated.shape[1])],
            "detector": detector,
            "strategy": "omr_plus_multicue_warp_and_axis_trim",
        }
        (debug_path / "crop_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return CropResult(
        input_path=str(src),
        output_path=str(out_path),
        debug_dir=str(debug_path) if debug_path else None,
        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
        used_fallback=used_fallback,
    )

# -*- coding: utf-8 -*-
import os, csv, sys, re, json
import cv2
import numpy as np
import urllib.request

# ---------------- CONFIG ----------------
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MODEL_PATH_28 = r"Handwritten-Digit-Recognition/tf-cnn-model.keras"  # MNIST-style 28x28 model (0-9)

OUT_DIR = "debug_ballot"  # root output dir for bulk runs
os.makedirs(OUT_DIR, exist_ok=True)

# Candidate-box placement heuristics
MIN_Y_FRAC = 0.05
MIN_X_FRAC = 0.60
MIN_AREA_FRAC = 0.005
PAD_FRAC = 0.00  # padding inside detected vote-box bounding rect

# NULL thresholds
MIN_TOP_CONF = 0.60
MIN_MARGIN   = 0.10

# If you know expected candidates per ballot, you can set this,
# but leaving detection-driven is usually better unless your layout is fixed.
EXPECTED_BOXES = None  # e.g. 15 or 9, or None


# ---------------- Utilities ----------------
def read_image_from_url(url):
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    img_array = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def deskew_with_hough(img, max_skew=10, hough_thresh=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)
    if lines is None:
        return img, 0.0

    angles = []
    for l in lines:
        rho, theta = l[0]
        angle = (theta - np.pi / 2) * 180.0 / np.pi
        if abs(angle) < max_skew:
            angles.append(angle)

    if not angles:
        return img, 0.0

    skew_angle = float(np.median(angles))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), skew_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, skew_angle


def kill_outer_frame(mask, frame=2):
    if mask.dtype != np.uint8:
        m = mask.astype(np.uint8)
    else:
        m = mask.copy()

    if m.max() == 255:
        m = (m > 0).astype(np.uint8)

    h, w = m.shape
    f = min(frame, h // 2, w // 2)
    if f <= 0:
        return m

    m[:f, :] = 0
    m[-f:, :] = 0
    m[:, :f] = 0
    m[:, -f:] = 0
    return m


def clean_components(mask, min_area=8):
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cleaned = np.zeros_like(m)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1
    return cleaned


def get_components(mask_bool):
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


def horizontal_overlap_ratio(c1, c2):
    x1a, x1b = c1["x"], c1["x"] + c1["w"]
    x2a, x2b = c2["x"], c2["x"] + c2["w"]
    inter = max(0, min(x1b, x2b) - max(x1a, x2a))
    return inter / max(1.0, float(min(c1["w"], c2["w"])))


def group_components_into_digits(comps, overlap_thresh=0.4):
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
            if horizontal_overlap_ratio(c, d) >= overlap_thresh:
                group.append(d)
                used.add(j)
        groups.append(group)
    return groups


def build_group_masks(mask_shape, groups):
    group_masks = []
    for group in groups:
        gmask = np.zeros(mask_shape, dtype=bool)
        for comp in group:
            gmask |= comp["mask"]
        group_masks.append(gmask)
    return group_masks


# ---------------- OCR name line (optional) ----------------
def ocr_name_line(row_bgr):
    try:
        import pytesseract
        from PIL import Image
        if 'TESSERACT_EXE' in globals() and os.path.exists(TESSERACT_EXE):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
    except Exception:
        return None, None

    H, W, _ = row_bgr.shape
    roi = row_bgr[int(0.10*H):int(0.55*H), int(0.03*W):int(0.55*W)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    proc = clahe.apply(gray)

    _, th = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) < 128:
        th = 255 - th

    try:
        cfg = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
        text = pytesseract.image_to_string(Image.fromarray(th), config=cfg)
    except Exception:
        return None, None

    text = re.sub(
        r"[^\w\s'’\u2013\u2014\-ÁÉÍÓÚÄËÏÖÜÂÊÎÔÛÀÈÌÒÙÇÑ]",
        " ",
        text or ""
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None, None

    m = re.search(
        r"^([A-Z'’\-ÁÉÍÓÚÄËÏÖÜÂÊÎÔÛÀÈÌÒÙÇÑ]+)\s*[\u2014\u2013\-]",
        text
    )
    surname = m.group(1) if m else None
    return text, surname


def pick_candidate_name(full_line, surname, row_idx):
    if surname:
        return surname

    if full_line:
        ban = {"TOGHCHÁN", "TOGHCÁN", "TOGHCAN", "DON", "UACHTARÁN", "PRESIDENTIAL",
               "ELECTION", "TREORACHA", "INSTRUCTIONS", "WRITE", "FILL", "FOLD"}
        tokens = re.findall(r"[A-Z'’\-ÁÉÍÓÚÄËÏÖÜÂÊÎÔÛÀÈÌÒÙÇÑ]+", full_line.upper())
        tokens = [t for t in tokens if t not in ban]
        if tokens:
            return tokens[0]

    return f"ROW_{row_idx}"


# ---------------- Detection ----------------
def detect_vote_boxes(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 51)
    norm = cv2.divide(gray, bg, scale=255)

    try:
        th = cv2.ximgproc.niBlackThreshold(norm, 255, cv2.THRESH_BINARY_INV, 41, k=0.2)
    except Exception:
        th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 7)

    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), 2)

    H, W = th.shape
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-6)
        area = w * h
        if 0.65 <= ar <= 1.5 and area > 0.0015 * W * H and w > 0.04*W and h > 0.04*H:
            cands.append((x, y, w, h))

    if not cands:
        return []

    # cluster by x; keep right-most cluster
    try:
        from sklearn.cluster import DBSCAN
        xs = np.array([x for x,_,_,_ in cands], dtype=np.float32).reshape(-1,1)
        labels = DBSCAN(eps=0.08*W, min_samples=2).fit(xs).labels_
        clusters = {}
        for lab, box in zip(labels, cands):
            clusters.setdefault(lab, []).append(box)
        keys = [k for k in clusters if k != -1] or [-1]
        best_k = max(keys, key=lambda k: np.median([b[0] for b in clusters[k]]))
        boxes = clusters[best_k]
    except Exception:
        boxes = sorted(cands, key=lambda b: b[0], reverse=True)[:max(10, len(cands))]

    boxes.sort(key=lambda b: b[1])
    return boxes


def filter_boxes_layout(boxes, img_shape):
    H, W = img_shape[:2]
    boxes = [b for b in boxes if b[0] > MIN_X_FRAC*W and b[1] > MIN_Y_FRAC*H]
    if not boxes:
        return []

    boxes = [b for b in boxes if b[2]*b[3] > MIN_AREA_FRAC*W*H]
    if not boxes:
        return []

    areas = np.array([w*h for (_,_,w,h) in boxes], dtype=float)
    med = float(np.median(areas))
    lo, hi = 0.55*med, 1.7*med
    boxes = [b for b in boxes if lo <= b[2]*b[3] <= hi]

    boxes.sort(key=lambda b: b[1])
    return boxes


def boxes_to_rows(img_bgr, boxes):
    H, W, _ = img_bgr.shape
    if not boxes:
        return []
    centers = [y + h//2 for (_,y,_,h) in boxes]
    rows = []
    top = max(0, centers[0] - int(0.9 * boxes[0][3]))
    for i in range(len(centers)-1):
        mid = (centers[i] + centers[i+1]) // 2
        rows.append((top, mid))
        top = mid
    bottom = min(H-1, centers[-1] + int(0.9 * boxes[-1][3]))
    rows.append((top, bottom))
    merged = []
    for y1, y2 in rows:
        if y2 - y1 < 10:
            continue
        if not merged or y1 > merged[-1][1] - 5:
            merged.append([y1, y2])
        else:
            merged[-1][1] = max(merged[-1][1], y2)
    return [(a,b) for a,b in merged]


def find_row_bands(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    yPositions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > W // 3:
            yPositions.append(y)
    yPositions = sorted(yPositions)
    rows = []
    for i in range(len(yPositions) - 1):
        y1, y2 = yPositions[i], yPositions[i + 1]
        if y2 - y1 > 0.04 * H:
            rows.append((y1, y2))
    return rows


# ---------------- Vote box cropping ----------------
def crop_rightmost_square(row_bgr):
    """
    Fallback method when we don't have detected boxes:
    find the right-most square-ish blob or take right strip.
    """
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
    bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    H, W = bin_inv.shape
    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-6)
        area = w * h
        if x > 0.60 * W and 0.5 < ar < 1.5 and area > 0.005 * W * H:
            cands.append((x, y, w, h))

    if not cands:
        return row_bgr[:, int(0.75 * W):]

    x, y, w, h = max(cands, key=lambda b: b[0])

    pad_x = int(0.05 * w)
    pad_y = int(0.05 * h)
    xi = max(x + pad_x, 0)
    xe = min(x + w - pad_x, W)
    yi = max(y + pad_y, 0)
    ye = min(y + h - pad_y, H)
    return row_bgr[yi:ye, xi:xe]


def pad_crop_from_box(img, box):
    x, y, w, h = box
    H, W = img.shape[:2]
    pad = int(PAD_FRAC * min(w, h))
    xi = max(x + pad, 0); yi = max(y + pad, 0)
    xe = min(x + w - pad, W); ye = min(y + h - pad, H)
    return img[yi:ye, xi:xe]


def crop_vote_box_interior(vote_box_bgr, inner_margin=3):
    h, w = vote_box_bgr.shape[:2]
    y1 = inner_margin
    y2 = h - inner_margin
    x1 = inner_margin
    x2 = w - inner_margin
    if y2 <= y1 or x2 <= x1:
        return vote_box_bgr
    return vote_box_bgr[y1:y2, x1:x2]


# ---------------- Enhancement / border removal ----------------
def remove_horizontal_rules(bin_img):
    k = max(5, int(0.015 * bin_img.shape[1]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    rules = cv2.morphologyEx(255 - bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.max(bin_img, 255 - rules)
    return cleaned


def enhance_vote_box(box_bgr):
    gray = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)

    # This helps when ballot rows have printed ruling lines.
    #th = remove_horizontal_rules(th)

    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)
    return 255 - th  # white bg (255), black ink (0)


def remove_border_components_v1(enhanced):
    fg = (enhanced < 128).astype(np.uint8)  # 1 = ink
    if fg.sum() == 0:
        return fg.astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    H, W = fg.shape
    keep = np.zeros_like(fg)

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_border = (x == 0 or y == 0 or x + w == W or y + h == H)

        is_long_horizontal = touches_border and (w >= 0.9 * W and h <= 0.15 * H)
        is_long_vertical   = touches_border and (h >= 0.9 * H and w <= 0.15 * W)

        if is_long_horizontal or is_long_vertical:
            continue

        keep[labels == i] = 1

    if keep.sum() < 0.2 * fg.sum():
        keep = fg

    return keep.astype(bool)


def remove_border_components_v2(enhanced):
    fg = (enhanced < 128).astype(np.uint8)
    if fg.sum() == 0:
        return fg.astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    H, W = fg.shape
    keep = np.zeros_like(fg)

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_border = (x == 0 or y == 0 or x + w == W or y + h == H)

        long_horizontal = touches_border and (w >= 0.95 * W and h <= 0.12 * H)
        long_vertical   = touches_border and (h >= 0.95 * H and w <= 0.12 * W)

        if long_horizontal or long_vertical:
            continue

        keep[labels == i] = 1

    if keep.sum() == 0:
        keep = fg
    return keep.astype(bool)


def remove_border_components_v3(enhanced):
    fg = (enhanced < 128).astype(np.uint8)
    if fg.sum() == 0:
        return fg.astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    H, W = fg.shape
    keep = np.zeros_like(fg)

    edge_margin = max(1, int(0.02 * min(H, W)))

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_border = (x == 0 or y == 0 or x + w == W or y + h == H)

        if touches_border:
            continue

        if (w >= 0.98 * W and h <= 0.12 * H) or (h >= 0.98 * H and w <= 0.12 * W):
            continue

        # also drop near-border sparse stuff
        near_border = (x <= edge_margin or y <= edge_margin or
                       x + w >= W - edge_margin or y + h >= H - edge_margin)
        if near_border and area < 0.01 * (H * W):
            continue

        keep[labels == i] = 1

    if keep.sum() == 0:
        keep = fg

    return keep.astype(bool)


def strip_thin_border_pieces(mask, ratio_thresh=0.25):
    m = (mask > 0).astype(np.uint8)
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

    if cleaned.sum() == 0:
        cleaned = m

    return cleaned.astype(bool)


def is_frame_only_mask(mask_bool,
                       band_frac=0.16,
                       min_border_ratio=0.8,
                       max_area_ratio=0.30):
    m = (mask_bool > 0).astype(np.uint8)
    total = m.sum()
    if total == 0:
        return False

    H, W = m.shape
    ys, xs = np.where(m)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    bb_area = (y2 - y1 + 1) * (x2 - x1 + 1)
    area_ratio = total / float(max(1, bb_area))

    band = max(1, int(band_frac * min(H, W)))
    border = np.zeros_like(m, np.uint8)
    border[:band, :] = 1
    border[-band:, :] = 1
    border[:, :band] = 1
    border[:, -band:] = 1

    border_ink = (m & border).sum()
    border_ratio = border_ink / float(total)

    return (border_ratio >= min_border_ratio) and (area_ratio <= max_area_ratio)


def tight_center_crop(mask_bool, pad=2):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    H, W = mask_bool.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad)
    y2 = min(H - 1, y2 + pad)

    return mask_bool[y1:y2+1, x1:x2+1].copy()


def mask_to_mnist28(mask_bool, margin=4, min_stroke_px=2):
    if mask_bool is None or mask_bool.sum() == 0:
        return None

    m = mask_bool.astype(np.uint8) * 255
    ys, xs = np.where(mask_bool)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    crop = m[y1:y2+1, x1:x2+1]

    Hc, Wc = crop.shape
    stroke_est = max(1, int(round(0.008 * max(Hc, Wc))))
    if stroke_est < min_stroke_px:
        k = max(1, min_stroke_px - stroke_est)
        crop = cv2.dilate(
            crop,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1)),
            1
        )

    target_inner = 28 - 2*margin
    h, w = crop.shape
    scale = min(target_inner / float(max(h, w)), 1.0) if max(h, w) > 0 else 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((28, 28), np.uint8)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    bin01 = (canvas > 0).astype(np.uint8)
    M = cv2.moments(bin01, binaryImage=True)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dx = int(round(14 - cx))
        dy = int(round(14 - cy))
        Mshift = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas = cv2.warpAffine(canvas, Mshift, (28, 28), flags=cv2.INTER_CUBIC, borderValue=0)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    x = (canvas.astype(np.float32) / 255.0)
    x = np.clip(x, 0.0, 1.0)
    return x[None, ..., None]


# ---------------- Model & prediction ----------------
def load_mnist28_model(model_path=MODEL_PATH_28):
    from tensorflow.keras import models
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return models.load_model(model_path)


def override_three_if_skinny_one(pred, probs, mask_bool):
    if pred != 3:
        return pred
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return pred
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    aspect = h / float(w + 1e-6)
    if aspect > 2.2 and probs[1] >= probs[3] * 0.7:
        return 1
    return pred
def assign_boxes_to_rows(boxes, rows):
    """
    For each row band (y1,y2), pick the box whose vertical center lies inside the band.
    Returns a list same length as rows: either a box (x,y,w,h) or None.
    If multiple boxes land in the same row, pick the right-most (largest x).
    """
    assigned = [None] * len(rows)
    if not boxes or not rows:
        return assigned

    for (x, y, w, h) in boxes:
        cy = y + h / 2.0
        for i, (y1, y2) in enumerate(rows):
            if y1 <= cy <= y2:
                if assigned[i] is None or x > assigned[i][0]:
                    assigned[i] = (x, y, w, h)
                break
    return assigned

def classify_single_digit_mask(model28, mask_bool):
    x28 = mask_to_mnist28(mask_bool, margin=4)
    if x28 is None:
        return None, None, 0.0, 0.0

    probs = model28.predict(x28, verbose=0)[0]
    pred = int(np.argmax(probs))
    top2 = np.partition(probs, -2)[-2:]
    margin = float(top2.max() - top2.min())
    conf = float(probs[pred])

    if conf < MIN_TOP_CONF or margin < MIN_MARGIN:
        return None, probs, conf, margin

    return pred, probs, conf, margin


def predict_digit_from_box(model28, vote_box_bgr, row_idx, debug_prefix, border_fn, debug_dir=None):
    # Important: trim interior to reduce box-frame leakage
    vote_box_bgr = crop_vote_box_interior(vote_box_bgr, inner_margin=3)

    enhanced = enhance_vote_box(vote_box_bgr)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_enhanced.png"), enhanced)

    mask = border_fn(enhanced)
    mask = strip_thin_border_pieces(mask)
    mask = kill_outer_frame(mask, frame=2)
    mask = clean_components(mask.astype(np.uint8), min_area=8)

    if is_frame_only_mask(mask):
        return None, {"ink_ratio": float(mask.mean()), "frame_only": True}

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_mask.png"),
                    np.where(mask > 0, 0, 255).astype(np.uint8))

    digit_mask = tight_center_crop(mask.astype(bool), pad=2)
    if digit_mask is None or digit_mask.sum() == 0:
        return None, {"ink_ratio": 0.0}

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_tight.png"),
                    np.where(digit_mask, 0, 255).astype(np.uint8))

        x28_debug = mask_to_mnist28(digit_mask, margin=4)
        if x28_debug is not None:
            vis = (x28_debug[0, :, :, 0] * 255).astype("uint8")
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_mnist28.png"), vis)

    ink_ratio = float(digit_mask.mean())
    comps = get_components(digit_mask)
    if not comps:
        return None, {"ink_ratio": ink_ratio}

    groups = group_components_into_digits(comps, overlap_thresh=0.4)
    group_masks = build_group_masks(digit_mask.shape, groups)

    def group_x_center(gmask):
        ys, xs = np.where(gmask)
        return xs.mean() if len(xs) else np.inf

    group_masks.sort(key=group_x_center)

    def classify_group(mask_bool):
        d, probs, conf, margin = classify_single_digit_mask(model28, mask_bool)
        if d is None:
            return None, probs, conf, margin
        d = override_three_if_skinny_one(d, probs, mask_bool)
        if conf < MIN_TOP_CONF or margin < MIN_MARGIN:
            return None, probs, conf, margin
        return d, probs, conf, margin

    # Single digit (multi-stroke ok)
    if len(group_masks) == 1:
        d, probs, conf, margin = classify_group(group_masks[0])
        return d, {"ink_ratio": ink_ratio, "probs": probs, "confidence": conf, "margin": margin, "multi_digit": False}

    # Two groups => maybe "10"
    if len(group_masks) == 2:
        d1, p1, c1, m1 = classify_group(group_masks[0])
        d2, p2, c2, m2 = classify_group(group_masks[1])

        if d1 == 1 and d2 == 0:
            return 10, {"ink_ratio": ink_ratio, "multi_digit": True, "digits": [d1, d2]}

        d_full, p_full, c_full, m_full = classify_group(digit_mask)
        return d_full, {"ink_ratio": ink_ratio, "probs": p_full, "confidence": c_full, "margin": m_full, "multi_digit": False}

    # 3+ groups: fallback to full-mask
    d_full, p_full, c_full, m_full = classify_group(digit_mask)
    return d_full, {"ink_ratio": ink_ratio, "probs": p_full, "confidence": c_full, "margin": m_full, "multi_digit": False, "groups": len(group_masks)}


# ---------------- Box scoring / enforcing ----------------
def box_score(b, W, H):
    x, y, w, h = b
    ar = w / float(h + 1e-6)
    ar_pen = 1.0 - min(abs(ar - 1.0), 1.0)
    right_bias = x / float(W)
    area = (w*h) / float(W*H)
    return 0.55*right_bias + 0.35*ar_pen + 0.10*area


def enforce_k_boxes(boxes, img_shape, k):
    if not boxes:
        return []

    H, W = img_shape[:2]
    centers_y = np.array([y + h/2.0 for (_,y,_,h) in boxes], dtype=np.float32).reshape(-1, 1)

    def pick_representatives(labels):
        chosen = []
        for lab in sorted(set(labels)):
            idxs = [i for i, L in enumerate(labels) if L == lab]
            if not idxs:
                continue
            scored = sorted(
                idxs,
                key=lambda i: (box_score(boxes[i], W, H), boxes[i][0]),
                reverse=True
            )
            chosen.append(boxes[scored[0]])
        chosen.sort(key=lambda b: b[1])
        return chosen

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = km.fit_predict(centers_y)
        chosen = pick_representatives(labels)
        if len(chosen) < k:
            remaining = [b for b in boxes if b not in chosen]
            remaining.sort(key=lambda b: b[1])
            chosen += remaining[:max(0, k - len(chosen))]
        return chosen[:k]
    except Exception:
        bands = np.linspace(0, H, k+1).astype(int)
        chosen = []
        for i in range(k):
            y1, y2 = bands[i], bands[i+1]
            in_band = [b for b in boxes if (b[1] + b[3]/2.0) >= y1 and (b[1] + b[3]/2.0) < y2]
            if not in_band:
                continue
            best = max(in_band, key=lambda b: (box_score(b, W, H), b[0]))
            chosen.append(best)
        chosen.sort(key=lambda b: b[1])
        if len(chosen) < k:
            remaining = [b for b in boxes if b not in chosen]
            remaining.sort(key=lambda b: (-b[0], b[1]))
            chosen += remaining[:max(0, k - len(chosen))]
        return chosen[:k]


def digits_sequence_ok(results):
    digits = [r["digit"] for r in results if isinstance(r["digit"], int)]
    if not digits:
        return False
    n = len(digits)
    return sorted(digits) == list(range(1, n+1))


# ---------------- Pipeline ----------------
def run_full_pipeline(img, model28, border_fn, debug_dir=None):
    raw_boxes = detect_vote_boxes(img)
    boxes = filter_boxes_layout(raw_boxes, img.shape)

    # --- IMPORTANT CHANGE ---
    # Do NOT force k based on len(boxes). If you want to trim extras, only trim when there are TOO MANY.
    if EXPECTED_BOXES is not None and len(boxes) > EXPECTED_BOXES:
        boxes = enforce_k_boxes(boxes, img.shape, k=int(EXPECTED_BOXES))

    # Row bands should come from structure, not from "however many boxes we happened to detect"
    rows = find_row_bands(img)
    if EXPECTED_BOXES is not None:
        # Optional sanity: if row detector under-finds, fall back to boxes-derived rows
        if len(rows) < int(EXPECTED_BOXES) and len(boxes) >= 3:
            rows = boxes_to_rows(img, boxes)
    elif len(rows) < 3 and len(boxes) >= 3:
        rows = boxes_to_rows(img, boxes)

    # Debug: draw row bands
    if debug_dir:
        H, W = img.shape[:2]
        dbg = img.copy()
        for idx, (y1, y2) in enumerate(rows, start=1):
            cv2.rectangle(dbg, (0, y1), (W-1, y2), (0, 0, 255), 3)
            cv2.putText(dbg, str(idx), (10, (y1 + y2)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "rows_highlighted.png"), dbg)

    # --- NEW: assign boxes to rows by y ---
    boxes_per_row = assign_boxes_to_rows(boxes, rows)

    results = []
    for i, (y1, y2) in enumerate(rows, start=1):
        row = img[y1:y2, :]

        box = boxes_per_row[i-1]
        if box is not None:
            vote_box = pad_crop_from_box(img, box)
        else:
            vote_box = crop_rightmost_square(row)

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"row_{i:02d}_box_raw.png"), vote_box)

        pred_digit, meta = predict_digit_from_box(
            model28, vote_box, i,
            debug_prefix=f"row_{i:02d}",
            border_fn=border_fn,
            debug_dir=debug_dir
        )

        full_line, surname = ocr_name_line(row)
        candidate = pick_candidate_name(full_line, surname, i)

        results.append({
            "row": i,
            "candidate": candidate,
            "digit": pred_digit if pred_digit is not None else "NULL"
        })

    return results


def process_one_image(img_path, model28, debug=True):
    img = cv2.imread(img_path)
    if img is None:
        return None, {"error": f"Could not read image: {img_path}"}

    img, angle = deskew_with_hough(img)
    dbg_dir = None
    if debug:
        base = os.path.splitext(os.path.basename(img_path))[0]
        dbg_dir = os.path.join(OUT_DIR, base)
        os.makedirs(dbg_dir, exist_ok=True)

    border_used = "v1"
    results = run_full_pipeline(img, model28, border_fn=remove_border_components_v1, debug_dir=dbg_dir)

    if digits_sequence_ok(results):
        pass
    else:
        border_used = "v2"
        results = run_full_pipeline(img, model28, border_fn=remove_border_components_v2, debug_dir=dbg_dir)
        if not digits_sequence_ok(results):
            border_used = "v3"
            results = run_full_pipeline(img, model28, border_fn=remove_border_components_v3, debug_dir=dbg_dir)

    # Write per-image CSV (like your single-image script)
    if dbg_dir:
        csv_path = os.path.join(dbg_dir, "ballot_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row", "candidate", "digit"])
            for r in results:
                w.writerow([r["row"], r["candidate"], r["digit"]])

    meta = {
        "image_path": img_path,
        "deskew_angle_deg": float(angle),
        "border_fn_used": border_used,
        "sequence_ok": bool(digits_sequence_ok(results)),
        "numbers_found": int(sum(1 for r in results if isinstance(r["digit"], int))),
    }
    return results, meta


def main(images_dir, debug=True):
    try:
        model28 = load_mnist28_model(MODEL_PATH_28)
        print("[INFO] Loaded MNIST 28x28 model.")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not os.path.isdir(images_dir):
        print(f"[ERROR] Folder not found: {images_dir}")
        sys.exit(1)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    audit_log = []
    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith(valid_ext):
            continue

        img_path = os.path.join(images_dir, filename)
        print(f"\n===== Processing: {img_path} =====")

        results, meta = process_one_image(img_path, model28, debug=debug)
        if results is None:
            print("[ERROR]", meta.get("error"))
            audit_log.append({"image": filename, **meta})
            continue

        print(f"[INFO] Deskewed by {meta['deskew_angle_deg']:.2f} deg, border={meta['border_fn_used']}")
        print("==== Results (Candidate → Number) ====")
        for r in results:
            print(f"{r['candidate']:<20} -> {r['digit']}")
        print(f"[INFO] Numbers in sequence: {meta['sequence_ok']}")
        print(f"[INFO] Total numbers found: {meta['numbers_found']}")

        audit_log.append({
            "image": filename,
            **meta,
            "results": results,
        })

    audit_path = os.path.join(OUT_DIR, "audit_log.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved JSON audit log: {audit_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bulk digit recognizer for ballots")
    parser.add_argument("--images", required=True, help="Directory of ballot images to scan")
    parser.add_argument("--no-debug", action="store_true", help="Disable writing debug images per ballot")
    args = parser.parse_args()

    main(args.images, debug=(not args.no_debug))
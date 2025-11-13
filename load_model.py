# -*- coding: utf-8 -*-
import os, csv, sys, re
import cv2
import numpy as np
import urllib.request

# ---------------- CONFIG ----------------
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
IMAGE_PATH = "assets/model/SampleBallots/7-Seater-Ballot-Papers-Mock-Election-images-0.jpg"
MODEL_PATH_28 = r"tf-cnn-model.keras"           # your MNIST-style 28x28 model (0-9)
OUT_DIR = "debug_ballot"
os.makedirs(OUT_DIR, exist_ok=True)
for f in os.listdir(OUT_DIR):
    try:
        os.remove(os.path.join(OUT_DIR, f))
    except Exception:
        pass

# Candidate-box placement heuristics
MIN_Y_FRAC = 0.25       # ignore boxes found above top 25% of page
MIN_X_FRAC = 0.60       # keep only boxes far on the right
MIN_AREA_FRAC = 0.005   # minimal area vs full page
PAD_FRAC = 0.08         # crop padding to remove borders (6–12% works well)

# NULL thresholds
MIN_TOP_CONF = 0.60
MIN_MARGIN   = 0.10

EXPECTED_BOXES = 9
def read_image_from_url(url):
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    img_array = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img



# --------------- OCR name line (optional) ---------------
def ocr_name_line(row_bgr):
    """
    Try to read the bold candidate name line on the left of a row.
    Returns (full_line_text, SURNAME) or (None, None) if OCR not available.
    """
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

# --------------- Detection ----------------
def detect_vote_boxes(img_bgr):
    """
    Return vote boxes (x,y,w,h), sorted by y.
    Improved: illumination normalization, adaptive threshold, coarse geometry,
    then cluster by x and keep the right-most column.
    """
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

    # cluster by x; keep right-most cluster (helps with perspective)
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
        boxes = sorted(cands, key=lambda b: b[0], reverse=True)[:10]

    boxes.sort(key=lambda b: b[1])
    return boxes

def filter_boxes_layout(boxes, img_shape):
    """Reject header/instruction artefacts; keep only real candidate boxes."""
    H, W = img_shape[:2]
    # 1) keep far right and sufficiently low on the page
    boxes = [b for b in boxes if b[0] > MIN_X_FRAC*W and b[1] > MIN_Y_FRAC*H]
    if not boxes:
        return []

    # 2) basic minimum area
    boxes = [b for b in boxes if b[2]*b[3] > MIN_AREA_FRAC*W*H]
    if not boxes:
        return []

    # 3) size-consistency around median area (reject tiny outliers like printed "1")
    areas = np.array([w*h for (_,_,w,h) in boxes], dtype=float)
    med = float(np.median(areas))
    lo, hi = 0.55*med, 1.7*med
    boxes = [b for b in boxes if lo <= b[2]*b[3] <= hi]

    # 4) sort by y
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
    # merge small gaps
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



def boxes_to_tight_rows(img_bgr, boxes, up_frac=0.55, down_frac=0.65):
    """
    Make one tight row per detected vote box.
    Each row extends only a fraction of the box height above/below its center,
    clamped by the midpoints to adjacent boxes to avoid overlap.
    """
    H, W, _ = img_bgr.shape
    if not boxes:
        return []

    # sort by y, compute centers
    boxes = sorted(boxes, key=lambda b: b[1])
    centers = [y + h/2.0 for (_, y, _, h) in boxes]

    # midpoints between neighboring centers
    mids = [(centers[i] + centers[i+1]) / 2.0 for i in range(len(centers)-1)]

    rows = []
    for i, (x, y, w, h) in enumerate(boxes):
        c = centers[i]
        # preferred tight band around the box
        y1 = int(c - up_frac*h)
        y2 = int(c + down_frac*h)

        # clamp so rows don't overlap
        if i > 0:
            y1 = max(y1, int(mids[i-1]))
        if i < len(boxes) - 1:
            y2 = min(y2, int(mids[i]))

        # final image bounds + minimal height guard
        y1 = max(0, y1)
        y2 = min(H-1, max(y1+12, y2))
        rows.append((y1, y2))

    return rows
# --------------- Vote-box crop & enhancement ---------------
def crop_rightmost_square(row_bgr):
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
    bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    H, W = bin_inv.shape
    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        area = w * h
        if x > 0.60 * W and 0.8 < ar < 1.25 and area > 0.008 * W * H:
            cands.append((x, y, w, h))
    if not cands:
        return row_bgr[:, int(0.65 * W):]

    x, y, w, h = max(cands, key=lambda b: b[0])
    pad = int(PAD_FRAC * min(w, h))
    xi = max(x + pad, 0); yi = max(y + pad, 0)
    xe = min(x + w - pad, W); ye = min(y + h - pad, H)
    return row_bgr[yi:ye, xi:xe]

def pad_crop_from_box(img, box):
    """Crop box from full image with padding to remove borders."""
    x, y, w, h = box
    H, W = img.shape[:2]
    pad = int(PAD_FRAC * min(w, h))
    xi = max(x + pad, 0); yi = max(y + pad, 0)
    xe = min(x + w - pad, W); ye = min(y + h - pad, H)
    return img[yi:ye, xi:xe]

def enhance_vote_box(box_bgr):
    gray = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)
    return 255 - th   # return white bg (255), black ink (0)

def remove_border_components(enhanced):
    fg = (enhanced < 128).astype(np.uint8)  # 1 = ink
    if fg.sum() == 0:
        return fg.astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    H, W = fg.shape
    keep = np.zeros_like(fg)

    # Be conservative near edges
    edge_margin = max(1, int(0.02 * min(H, W)))   # was 0.05

    for i in range(1, n):  # skip background
        x, y, w, h, area = stats[i]

        touches_border = (x == 0 or y == 0 or x + w == W or y + h == H)
        near_border = (x <= edge_margin or y <= edge_margin or
                       x + w >= W - edge_margin or y + h >= H - edge_margin)

        # Only drop if it actually touches a border (not just "near")
        if touches_border:
            continue

        # Drop only *extreme* ruling lines that span almost the entire side
        if (w >= 0.98 * W and h <= 0.12 * H) or (h >= 0.98 * H and w <= 0.12 * W):
            continue

        keep[labels == i] = 1

    # Fallback: if nothing kept, keep the largest non-border component
    if keep.sum() == 0 and n > 1:
        best = -1
        best_area = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if x == 0 or y == 0 or x + w == W or y + h == H:
                continue
            if area > best_area:
                best_area = area
                best = i
        if best > 0:
            keep[labels == best] = 1

    return keep.astype(bool)

def tight_center_crop(mask_bool, pad=2):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = mask_bool.shape
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w-1, x2 + pad); y2 = min(h-1, y2 + pad)
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
        crop = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1)), 1)

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

    # Slight blur to better match MNIST anti-aliasing
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    x = (canvas.astype(np.float32) / 255.0)
    x = np.clip(x, 0.0, 1.0)
    return x[None, ..., None]

# --------------- Model & prediction ---------------
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

def predict_digit_from_box(model28, vote_box_bgr, row_idx, debug_prefix):
    enhanced = enhance_vote_box(vote_box_bgr)
    cv2.imwrite(os.path.join(OUT_DIR, f"{debug_prefix}_enhanced.png"), enhanced)

    mask = remove_border_components(enhanced)
    cv2.imwrite(os.path.join(OUT_DIR, f"{debug_prefix}_mask.png"),
                np.where(mask, 0, 255).astype(np.uint8))

    digit_mask = tight_center_crop(mask, pad=2)
    if digit_mask is None or digit_mask.sum() == 0:
        return None, {"ink_ratio": 0.0}

    cv2.imwrite(os.path.join(OUT_DIR, f"{debug_prefix}_tight.png"),
                np.where(digit_mask, 0, 255).astype(np.uint8))

    x28 = mask_to_mnist28(digit_mask, margin=4)
    if x28 is None:
        return None, {"ink_ratio": 0.0}

    vis = (x28[0,:,:,0] * 255).astype("uint8")
    cv2.imwrite(os.path.join(OUT_DIR, f"{debug_prefix}_mnist28.png"), vis)

    probs = model28.predict(x28, verbose=0)[0]
    pred = int(np.argmax(probs))
    top2 = np.partition(probs, -2)[-2:]
    margin = float(top2.max() - top2.min())
    conf = float(probs[pred])
    ink_ratio = float(digit_mask.mean())

    # Optional skinny-stroke override to avoid 1→3
    pred = override_three_if_skinny_one(pred, probs, digit_mask)

    # NULL if low confidence or low margin
    if conf < MIN_TOP_CONF or margin < MIN_MARGIN:
        pred = None

    return pred, {"ink_ratio": ink_ratio, "probs": probs, "confidence": conf}

# --------------- Name selection (ONLY candidate + number) ---------------
def pick_candidate_name(full_line, surname, row_idx):
    # Prefer the SURNAME if OCR found it in all-caps before a dash.
    if surname:
        return surname

    # Else, pull the first ALL-CAPS token from the OCR line (skip headers).
    if full_line:
        ban = {"TOGHCHÁN", "TOGHCÁN", "TOGHCAN", "DON", "UACHTARÁN", "PRESIDENTIAL",
               "ELECTION", "TREORACHA", "INSTRUCTIONS", "WRITE", "FILL", "FOLD"}
        tokens = re.findall(r"[A-Z'’\-ÁÉÍÓÚÄËÏÖÜÂÊÎÔÛÀÈÌÒÙÇÑ]+", full_line.upper())
        tokens = [t for t in tokens if t not in ban]
        if tokens:
            return tokens[0]

    return f"ROW_{row_idx}"

# --------------- NEW: Box scoring & enforcement to exactly k boxes ---------------
def box_score(b, W, H):
    x, y, w, h = b
    ar = w / float(h + 1e-6)
    ar_pen = 1.0 - min(abs(ar - 1.0), 1.0)         # closer to square → higher
    right_bias = x / float(W)                       # farther right → higher
    area = (w*h) / float(W*H)
    return 0.55*right_bias + 0.35*ar_pen + 0.10*area

def enforce_k_boxes(boxes, img_shape, k=8):
    """
    Take a list of (x,y,w,h) and return exactly k boxes, one per row, sorted by y.
    Uses KMeans on vertical centers. Falls back to fixed bands if sklearn not present.
    """
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
            # pick the best scoring, tie-break by rightmost
            scored = sorted(
                idxs,
                key=lambda i: (box_score(boxes[i], W, H), boxes[i][0]),
                reverse=True
            )
            chosen.append(boxes[scored[0]])
        chosen.sort(key=lambda b: b[1])  # order by y
        return chosen

    # Try sklearn KMeans (more robust if rows are uneven)
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = km.fit_predict(centers_y)
        chosen = pick_representatives(labels)
        # If duplicates dropped for any reason, top up from remaining by y
        if len(chosen) < k:
            remaining = [b for b in boxes if b not in chosen]
            remaining.sort(key=lambda b: b[1])
            chosen += remaining[:max(0, k - len(chosen))]
        return chosen[:k]
    except Exception:
        # Fallback: split the page into k horizontal bands and take best from each band
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
        # If we still have < k, top up by rightmost-then-y
        if len(chosen) < k:
            remaining = [b for b in boxes if b not in chosen]
            remaining.sort(key=lambda b: (-b[0], b[1]))
            chosen += remaining[:max(0, k - len(chosen))]
        return chosen[:k]

# --------------- Main ---------------
def main():
    #img = read_image_from_url(IMAGE_PATH)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] Could not read image: {IMAGE_PATH}")
        sys.exit(1)

    try:
        model28 = load_mnist28_model(MODEL_PATH_28)
        print("[INFO] Loaded MNIST 28x28 model.")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # detect and FILTER boxes
    raw_boxes = detect_vote_boxes(img)
    boxes = filter_boxes_layout(raw_boxes, img.shape)

    # enforce exactly EXPECTED_BOXES where feasible
    if len(boxes) >= 2:
        k = max(3, min(20, len(boxes)))   # or use len(find_row_bands(img))
        boxes = enforce_k_boxes(boxes, img.shape, k=k)

    # derive rows from filtered boxes, else fallback
    if len(boxes) >= 3:
        rows = boxes_to_tight_rows(img, boxes)
    else:
        rows = find_row_bands(img)

    print(f"[INFO] Vote boxes (raw): {len(raw_boxes)}; kept: {len(boxes)}; rows: {len(rows)}")

    # debug overlay
    dbg = img.copy()
    for i, (x, y, w, h) in enumerate(raw_boxes, 1):
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 165, 255), 2)  # orange = raw
        cv2.putText(dbg, f"BR{i}", (x - 50, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
    for i, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)     # green = kept/enforced
        cv2.putText(dbg, f"B{i}", (x - 40, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    for i, (y1, y2) in enumerate(rows, 1):
        cv2.rectangle(dbg, (0, y1), (img.shape[1] - 1, y2), (255, 0, 0), 2)  # blue = rows
        cv2.putText(dbg, f"R{i}", (10, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "rows_from_boxes_debug.png"), dbg)

    # iterate rows
    results = []
    for i, (y1, y2) in enumerate(rows, start=1):
        row = img[y1:y2, :]

        # prefer filtered detected boxes; else fallback crop
        if i <= len(boxes):
            vote_box = pad_crop_from_box(img, boxes[i - 1])
        else:
            vote_box = crop_rightmost_square(row)

        cv2.imwrite(os.path.join(OUT_DIR, f"row_{i:02d}_box_raw.png"), vote_box)

        pred_digit, meta = predict_digit_from_box(
            model28, vote_box, i, debug_prefix=f"row_{i:02d}"
        )

        full_line, surname = ocr_name_line(row)
        candidate = pick_candidate_name(full_line, surname, i)

        results.append({
            "row": i,
            "candidate": candidate,
            "digit": pred_digit if pred_digit is not None else "NULL"
        })

    # compact report (candidate -> number)
    print("\n==== Results (Candidate → Number) ====")
    for r in results:
        print(f"{r['candidate']:<20} -> {r['digit']}")

    # compact CSV
    csv_path = os.path.join(OUT_DIR, "ballot_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "candidate", "digit"])
        for r in results:
            w.writerow([r["row"], r["candidate"], r["digit"]])
    print(f"\n[INFO] Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()

import os, csv, sys, re
import cv2
import numpy as np

#Add this path if using windows
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


IMAGE_PATH = "assets/model/Presidantial_Election_V1.jpg"
MODEL_PATH_28 = r"tf-cnn-model.h5"           # your MNIST-style 28x28 model (0-9)
OUT_DIR = "debug_ballot"
os.makedirs(OUT_DIR, exist_ok=True)


# OCR: Candidate name line
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

    pil_img = Image.fromarray(th)

    try:
        cfg = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
        text = pytesseract.image_to_string(pil_img, config=cfg)
    except Exception:
        return None, None

    text = re.sub(r"[^\w\s'’–—-]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None, None

    m = re.search(r"^([A-Z'’\-]+)\s*[—\-–]", text)
    surname = m.group(1) if m else None
    return text, surname

def detect_vote_boxes(img_bgr):
    """
    Return vote boxes (x,y,w,h), sorted by y. Much more tolerant to bottom shadows
    and perspective. It normalizes lighting, uses a generous area filter, and
    clusters by x to keep the right-most column.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- illumination normalization (shadow resistant) ---
    bg = cv2.medianBlur(gray, 51)
    norm = cv2.divide(gray, bg, scale=255)

    # --- threshold (try Sauvola if available, else adaptive) ---
    try:
        th = cv2.ximgproc.niBlackThreshold(norm, 255, cv2.THRESH_BINARY_INV,
                                           41, k=0.2)  # Sauvola-like
    except Exception:
        th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 7)

    # close gaps so each box is a single blob
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), 2)

    H, W = th.shape
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- generous geometric filters (old ones were too strict) ---
    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-6)
        area = w * h
        if (0.65 <= ar <= 1.5 and                # roughly square
            area > 0.0015 * W * H and            # was 0.005 → missed small/bottom boxes
            w > 0.04 * W and h > 0.04 * H and    # avoid tiny noise
            x > 0.45 * W):                        # keep right half (not only 0.60)
            cands.append((x, y, w, h))

    if not cands:
        return []

    # --- cluster by x and keep the rightmost cluster (robust to perspective) ---
    xs = np.array([x for x,_,_,_ in cands], dtype=np.float32).reshape(-1,1)
    # try DBSCAN; if unavailable, fall back to a simple rightmost filter
    try:
        from sklearn.cluster import DBSCAN
        eps = 0.08 * W  # how far boxes can drift horizontally
        labels = DBSCAN(eps=eps, min_samples=2).fit(xs).labels_
        clusters = {}
        for lab, box in zip(labels, cands):
            clusters.setdefault(lab, []).append(box)
        # drop noise cluster (-1) unless it's the only one
        keys = [k for k in clusters if k != -1] or [-1]
        # pick cluster with the largest median x (the right column)
        best_k = max(keys, key=lambda k: np.median([b[0] for b in clusters[k]]))
        boxes = clusters[best_k]
    except Exception:
        # fallback: take the N boxes with largest x (N<=10), then sort by y
        boxes = sorted(cands, key=lambda b: b[0], reverse=True)[:10]

    boxes.sort(key=lambda b: b[1])
    return boxes



def boxes_to_rows(img_bgr, boxes):
    """
    Convert the list of sorted boxes [(x,y,w,h), ...] to row (y1,y2) bands by
    splitting halfway between successive box centers. This reliably covers bottom rows.
    """
    H, W, _ = img_bgr.shape
    if not boxes:
        return []

    centers = [y + h//2 for (_,y,_,h) in boxes]
    rows = []

    # Top boundary: a little above the first box
    top = max(0, centers[0] - int(0.9 * boxes[0][3]))
    for i in range(len(centers)-1):
        mid = (centers[i] + centers[i+1]) // 2
        rows.append((top, mid))
        top = mid
    # Bottom boundary: a little below the last box
    bottom = min(H-1, centers[-1] + int(0.9 * boxes[-1][3]))
    rows.append((top, bottom))

    # Small clamp/merge just in case
    merged = []
    for y1, y2 in rows:
        if y2 - y1 < 10:
            continue
        if not merged or y1 > merged[-1][1] - 5:
            merged.append([y1, y2])
        else:
            merged[-1][1] = max(merged[-1][1], y2)
    return [(a,b) for a,b in merged]

# Row detection
def find_row_bands(img_bgr):
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)#Converts whole image to gray
    #converts gray to black and white (removes shadows)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 
                                   15,#15x15 pixel block size affected
                                    8# The mininum level of darkness 
                                    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))#Detects line 80pixels wide 1 tall
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)#Attempts to removes noise of image 2 times
    contours, _ = cv2.findContours(morph, 
                                   cv2.RETR_EXTERNAL,#Only searches for the outer boder of row 
                                   cv2.CHAIN_APPROX_SIMPLE #gets contours of 4 edges instead of everypixel
                                   )

    H, W = gray.shape#gets total height and width of shape
    yPositions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)#gets the y coordinate and width of each expected row
        #if the width is at least a third of the image then it is a valid row
        if w > W // 3:
            yPositions.append(y)
    yPositions = sorted(yPositions)

    rows = []
    #finds the positions closest to eachother and adds them to get row coordinates
    for i in range(len(yPositions) - 1):
        y1, y2 = yPositions[i], yPositions[i + 1]
        if y2 - y1 > 0.04 * H:
            rows.append((y1, y2))
    return rows


# Crop rightmost vote box  
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
        return row_bgr[:, int(0.75 * W):]

    x, y, w, h = max(cands, key=lambda b: b[0])
    pad = int(0.06 * min(w, h))
    xi = max(x + pad, 0); yi = max(y + pad, 0)
    xe = min(x + w - pad, W); ye = min(y + h - pad, H)
    return row_bgr[yi:ye, xi:xe]


# Enhance vote box
def enhance_vote_box(box_bgr):
    gray = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)

    # Adaptive threshold → ink=255, bg=0
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)

    # Light closing to reconnect broken curves
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)

    # Return white bg (255), black ink (0) as your pipeline expects
    return 255 - th



#  Remove box borders / ruling lines
def remove_border_components(enhanced):
    fg = (enhanced < 128).astype(np.uint8)  # 1=ink
    if fg.sum() == 0:
        return fg.astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    H, W = fg.shape
    keep = np.zeros_like(fg)

    for i in range(1, n):  # skip background
        x, y, w, h, area = stats[i]

        if x == 0 or y == 0 or x + w == W or y + h == H:
            continue

        long_horizontal = (w >= 0.90 * W and h <= 0.15 * H)
        long_vertical   = (h >= 0.90 * H and w <= 0.15 * W)
        if long_horizontal or long_vertical:
            continue

        keep[labels == i] = 1

    return keep.astype(bool)


# Tight crop around ink
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


# Convert mask to MNIST 28x28
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

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    x = (canvas.astype(np.float32) / 255.0)
    x = np.clip(x, 0.0, 1.0)
    return x[None, ..., None]


# Load CNN model
def load_mnist28_model(model_path=MODEL_PATH_28):
    from tensorflow.keras import models
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return models.load_model(model_path)


# Predict digit from vote box
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
    conf = float(probs[pred])         # confidence of top class
    ink_ratio = float(digit_mask.mean())

    # If confidence < 0.5, treat as NULL (no valid digit)
    if conf < 0.4:
        pred = None

    return pred, {"ink_ratio": ink_ratio, "probs": probs, "confidence": conf}


# Main
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] Could not read image: {IMAGE_PATH}")
        sys.exit(1)

    # # --- (optional) trim tiny bottom shadow band that often confuses row detectors ---
    # H, W, _ = img.shape
    # img = img[: int(0.98 * H), :]
    # H = img.shape[0]

    # --- load MNIST-style model ---
    try:
        model28 = load_mnist28_model(MODEL_PATH_28)
        print("[INFO] Loaded MNIST 28x28 model.")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # --- detect vote boxes and derive rows from their vertical centers ---
    boxes = detect_vote_boxes(img)                 # -> list of (x,y,w,h), sorted by y
    if len(boxes) >= 3:
        rows = boxes_to_rows(img, boxes)           # -> list of (y1,y2) bands, same order
    else:
        # fallback to your original method if boxes were not found reliably
        rows = find_row_bands(img)

    print(f"[INFO] Vote boxes found: {len(boxes)}; rows derived: {len(rows)}")

    # --- visual debug for detection ---
    dbg = img.copy()
    for i, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(dbg, f"B{i}", (x - 40, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for i, (y1, y2) in enumerate(rows, 1):
        cv2.rectangle(dbg, (0, y1), (img.shape[1] - 1, y2), (255, 0, 0), 2)
        cv2.putText(dbg, f"R{i}", (10, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "rows_from_boxes_debug.png"), dbg)

    # --- iterate rows (and use detected boxes when available) ---
    results = []
    for i, (y1, y2) in enumerate(rows, start=1):
        row = img[y1:y2, :]

        # if we have a matching box for this row, crop it directly; else fallback
        if i <= len(boxes):
            x, y, w, h = boxes[i - 1]
            vote_box = img[y:y + h, x:x + w]
        else:
            vote_box = crop_rightmost_square(row)

        cv2.imwrite(os.path.join(OUT_DIR, f"row_{i:02d}_box_raw.png"), vote_box)

        # classify; your predict_digit_from_box should set pred=None if conf<0.5
        pred_digit, meta = predict_digit_from_box(
            model28, vote_box, i, debug_prefix=f"row_{i:02d}"
        )

        # OCR candidate name (optional if tesseract available)
        full_line, surname = ocr_name_line(row)

        results.append({
            "row": i,
            "name_line": full_line,
            "surname": surname,
            "digit": pred_digit if pred_digit is not None else "NULL",
            "ink_ratio": round(meta.get("ink_ratio", 0.0), 4),
            "confidence": round(meta.get("confidence", 0.0), 3),
        })

    # --- report ---
    print("\n==== Results (Name ↔ Digit) ====")
    for r in results:
        name = r["name_line"] or r["surname"] or f"Row {r['row']}"
        conf = r.get("confidence", 0.0)
        print(f"{name:<40} -> {r['digit']} (ink={r['ink_ratio']}, conf={conf:.2f})")

    # --- save CSV ---
    csv_path = os.path.join(OUT_DIR, "ballot_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "surname", "name_line", "digit", "ink_ratio", "confidence"])
        for r in results:
            w.writerow([r["row"], r["surname"] or "", r["name_line"] or "",
                        r["digit"], r["ink_ratio"], r["confidence"]])
    print(f"\n[INFO] Saved CSV: {csv_path}")



if __name__ == "__main__":
    main()

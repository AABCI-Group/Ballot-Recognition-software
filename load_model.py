import os, csv, sys, re
import cv2
import numpy as np

#Add this path if using windows
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


IMAGE_PATH = "assets/model/Ballot_Paper_V1.jpg"
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
    gray = cv2.fastNlMeansDenoising(gray, None, 25, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0,0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 128:
        binary = 255 - binary
    binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    return binary  # white bg (255), black ink (0)


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
    ink_ratio = float(digit_mask.mean())
    return pred, {"ink_ratio": ink_ratio, "probs": probs}


# Main
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] Could not read image: {IMAGE_PATH}")
        sys.exit(1)

    try:
        #loads tf-cnn-model.h5
        model28 = load_mnist28_model(MODEL_PATH_28)
        print("[INFO] Loaded MNIST 28x28 model.")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    rows = find_row_bands(img)
    rows = sorted(rows, key=lambda r: r[0])#sorts row coordinates to assending order
    print(f"[INFO] Detected {len(rows)} rows.")

    results = []
    for i, (y1, y2) in enumerate(rows, start=1):
        row = img[y1:y2, :]
        vote_box = crop_rightmost_square(row)
        cv2.imwrite(os.path.join(OUT_DIR, f"row_{i:02d}_box_raw.png"), vote_box)

        pred_digit, meta = predict_digit_from_box(
            model28, vote_box, i, debug_prefix=f"row_{i:02d}"
        )

        full_line, surname = ocr_name_line(row)

        results.append({
            "row": i,
            "name_line": full_line,
            "surname": surname,
            "digit": pred_digit,
            "ink_ratio": round(meta.get("ink_ratio", 0.0), 4),
        })

    print("\n==== Results (Name ↔ Digit) ====")
    for r in results:
        name = r["name_line"] or r["surname"] or f"Row {r['row']}"
        print(f"{name:<40} -> {r['digit']} (ink={r['ink_ratio']})")

    csv_path = os.path.join(OUT_DIR, "ballot_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row","surname","name_line","digit","ink_ratio"])
        for r in results:
            w.writerow([r["row"], r["surname"] or "", r["name_line"] or "", r["digit"], r["ink_ratio"]])
    print(f"\n[INFO] Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()

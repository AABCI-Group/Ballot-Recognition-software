
#!/usr/bin/env python3
"""
Create synthetic ballot images by compositing a transparent PNG stamp onto blank ballot scans.
Outputs:
 - images/ (composited images)
 - masks/  (binary masks of stamp placement)
 - labels_det/ (YOLO detection .txt files)
 - labels_seg/ (YOLO-Seg polygon .txt files)
 - manifest.csv

Usage example:
python src/tools/generate_synthetic.py \\
    --blanks_dir data/blanks \\
    --stamp_png assets/stamp.png \\
    --out_dir data/synth \\
    --count 500
"""
import os, argparse, random, csv
from pathlib import Path
import numpy as np
import cv2

random.seed(42)
np.random.seed(42)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def load_stamp(stamp_path):
    stamp = cv2.imread(stamp_path, cv2.IMREAD_UNCHANGED)
    if stamp is None:
        raise FileNotFoundError(f"Stamp not found: {stamp_path}")
    if stamp.shape[2] < 4:
        raise ValueError("Stamp PNG must have alpha channel.")
    bgr = stamp[..., :3]
    alpha = stamp[..., 3]
    return bgr, alpha
def random_border_position(img_w, img_h, stamp_w, stamp_h, border_frac=0.2, max_tries=100):
    """
    Pick a top-left (x, y) for the stamp so that its CENTER lies within the outer
    `border_frac` of the image (top/bottom/left/right), not near the center.
    Ensures the whole stamp stays inside the image.
    """
    border_x = int(border_frac * img_w)
    border_y = int(border_frac * img_h)

    # clamp so we don't get negative max for randint if stamp is big
    max_x = max(0, img_w - stamp_w)
    max_y = max(0, img_h - stamp_h)

    # rejection sampling: keep sampling until center is in the border region
    for _ in range(max_tries):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        cx = x + stamp_w / 2.0
        cy = y + stamp_h / 2.0

        in_left_border   = cx <= border_x
        in_right_border  = cx >= (img_w - border_x)
        in_top_border    = cy <= border_y
        in_bottom_border = cy >= (img_h - border_y)

        if in_left_border or in_right_border or in_top_border or in_bottom_border:
            return x, y

    # Fallback: if we somehow never hit the border zone, just return the last sample
    return x, y

def random_position(img_w, img_h, stamp_w, stamp_h, placement="mixed"):
    max_x = max(0, img_w - stamp_w)
    max_y = max(0, img_h - stamp_h)
    mode = placement
    if placement == "mixed":
        mode = random.choices(
            ["border", "anywhere", "top", "right_column"],
            weights=[0.35, 0.25, 0.25, 0.15],
            k=1,
        )[0]

    if mode == "border":
        return random_border_position(img_w, img_h, stamp_w, stamp_h, border_frac=0.24)
    if mode == "top":
        return random.randint(0, max_x), random.randint(0, max(0, int(img_h * 0.22) - stamp_h))
    if mode == "right_column":
        x0 = min(max_x, max(0, int(img_w * 0.70)))
        return random.randint(x0, max_x), random.randint(0, max_y)
    return random.randint(0, max_x), random.randint(0, max_y)

def augment_blank(blank):
    out = blank.copy()
    if random.random() < 0.65:
        out = cv2.convertScaleAbs(out, alpha=random.uniform(0.82, 1.18), beta=random.randint(-18, 18))
    if random.random() < 0.35:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), random.uniform(0.1, 0.8))
    if random.random() < 0.25:
        noise = np.random.normal(0, random.uniform(2, 8), out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out

def add_pen_decoys(blank, count=None):
    out = blank.copy()
    h, w = out.shape[:2]
    n = count if count is not None else random.randint(0, 5)
    color_choices = [(20, 20, 20), (35, 25, 65), (55, 55, 55)]
    for _ in range(n):
        color = random.choice(color_choices)
        thickness = random.randint(2, max(2, int(min(w, h) * 0.004)))
        cx = random.randint(int(w * 0.55), max(int(w * 0.56), int(w * 0.95)))
        cy = random.randint(int(h * 0.05), max(int(h * 0.06), int(h * 0.95)))
        kind = random.choice(["digit", "stroke", "dot"])
        if kind == "digit":
            txt = random.choice(list("123456789"))
            font_scale = random.uniform(0.7, 1.8) * (h / 1600.0)
            cv2.putText(out, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        elif kind == "stroke":
            pts = []
            for _ in range(random.randint(2, 4)):
                pts.append([cx + random.randint(-40, 40), cy + random.randint(-40, 40)])
            cv2.polylines(out, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
        else:
            radius = random.randint(3, max(4, int(min(w, h) * 0.012)))
            cv2.circle(out, (cx, cy), radius, color, -1, cv2.LINE_AA)
    return out

def random_transform_stamp(bgr, alpha, scale, angle):
    h0, w0 = alpha.shape
    new_w = max(1, int(w0 * scale))
    new_h = max(1, int(h0 * scale))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    bgr_r = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    alpha_r = cv2.resize(alpha, (new_w, new_h), interpolation=interp)

    M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1.0)
    bgr_rot = cv2.warpAffine(bgr_r, M, (new_w, new_h), borderValue=(0, 0, 0))
    alpha_rot = cv2.warpAffine(alpha_r, M, (new_w, new_h), borderValue=0)

    return bgr_rot, alpha_rot

def apply_stamp(blank, stamp_bgr, stamp_alpha, x, y, opacity):
    h, w = blank.shape[:2]
    hs, ws = stamp_alpha.shape

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + ws)
    y2 = min(h, y + hs)

    sx = max(0, -x)
    sy = max(0, -y)
    ex = sx + (x2 - x1)
    ey = sy + (y2 - y1)

    if ex <= sx or ey <= sy:
        return blank, None

    roi = blank[y1:y2, x1:x2]
    stamp_roi = stamp_bgr[sy:ey, sx:ex]
    alpha_roi = (stamp_alpha[sy:ey, sx:ex] / 255.0) * opacity
    alpha_roi = np.expand_dims(alpha_roi, axis=2)

    blended = (alpha_roi * stamp_roi + (1 - alpha_roi) * roi).astype("uint8")
    blank[y1:y2, x1:x2] = blended

    mask = np.zeros((h, w), dtype="uint8")
    mask[y1:y2, x1:x2] = (stamp_alpha[sy:ey, sx:ex] > 10).astype("uint8") * 255
    return blank, mask

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def save_yolo_bbox(filepath, bbox, w, h):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / w
    cy = (y0 + y1) / 2 / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    with open(filepath, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def generate(args):
    blanks = [p for p in Path(args.blanks_dir).glob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff"}]
    if not blanks:
        raise FileNotFoundError("No blank ballot images found.")

    stamp_bgr, stamp_alpha = load_stamp(args.stamp_png)

    out_images = Path(args.out_dir) / "images"
    out_masks  = Path(args.out_dir) / "masks"
    out_det    = Path(args.out_dir) / "labels_det"
    out_seg    = Path(args.out_dir) / "labels_seg"

    for d in [out_images, out_masks, out_det, out_seg]:
        ensure_dir(d)

    manifest = []

    for i in range(args.count):
        blank_path = random.choice(blanks)
        blank = cv2.imread(str(blank_path))
        if blank is None:
            continue

        blank = augment_blank(blank)
        if random.random() < args.decoy_ratio:
            blank = add_pen_decoys(blank)

        h, w = blank.shape[:2]
        is_negative = random.random() < args.negative_ratio
        target_frac = random.uniform(args.min_stamp_frac, args.max_stamp_frac)
        orig_h = stamp_alpha.shape[0]
        scale = (h * target_frac) / orig_h
        angle   = random.uniform(args.min_angle, args.max_angle)
        opacity = random.uniform(args.min_opacity, args.max_opacity)


        if is_negative:
            stamped = blank.copy()
            mask = None
        else:
            sb, sa = random_transform_stamp(stamp_bgr, stamp_alpha, scale, angle)
            hs, ws = sa.shape[:2]
            x, y = random_position(w, h, ws, hs, args.placement)
            stamped, mask = apply_stamp(blank.copy(), sb, sa, x, y, opacity)

        img_file = out_images / f"synth_{i:06d}.png"
        mask_file = out_masks  / f"synth_{i:06d}.png"
        det_file = out_det     / f"synth_{i:06d}.txt"
        seg_file = out_seg     / f"synth_{i:06d}.txt"

        cv2.imwrite(str(img_file), stamped)

        if mask is None:
            cv2.imwrite(str(mask_file), np.zeros((h, w), dtype=np.uint8))
            open(det_file, "w").close()
            open(seg_file, "w").close()
            manifest.append([img_file.as_posix(), mask_file.as_posix(), 0])
            continue

        cv2.imwrite(str(mask_file), mask)

        bbox = bbox_from_mask(mask)
        if bbox:
            save_yolo_bbox(det_file, bbox, w, h)
            x0,y0,x1,y1 = bbox
            with open(seg_file, "w") as f:
                f.write(f"0 {x0/w} {y0/h} {x1/w} {y0/h} {x1/w} {y1/h} {x0/w} {y1/h}\n")

        manifest.append([img_file.as_posix(), mask_file.as_posix(), 1])

    with open(Path(args.out_dir) / "manifest.csv", "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["image", "mask", "has_stamp"])
        writer.writerows(manifest)

    print(f"✅ Done! Generated {args.count} synthetic images in {args.out_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--blanks_dir", required=True)
    p.add_argument("--stamp_png", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--negative_ratio", type=float, default=0.30, help="Fraction of images with no stamp and empty YOLO labels.")
    p.add_argument("--decoy_ratio", type=float, default=0.70, help="Fraction of images receiving non-stamp pen/noise marks.")
    p.add_argument("--placement", choices=["mixed", "border", "top", "right_column", "anywhere"], default="mixed")
    p.add_argument("--min_stamp_frac", type=float, default=0.030, help="Minimum stamp height as a fraction of page height.")
    p.add_argument("--max_stamp_frac", type=float, default=0.070, help="Maximum stamp height as a fraction of page height.")
    p.add_argument("--min_angle", type=float, default=-35.0)
    p.add_argument("--max_angle", type=float, default=35.0)
    p.add_argument("--min_opacity", type=float, default=0.22)
    p.add_argument("--max_opacity", type=float, default=0.95)
    return p.parse_args()

if __name__ == "__main__":
    generate(parse_args())

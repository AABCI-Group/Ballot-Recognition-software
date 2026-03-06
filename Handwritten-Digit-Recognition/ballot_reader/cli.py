import os
import argparse
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .config import BallotConfig
from .io import read_image, write_json, write_csv
from .preprocess import normalize_illumination, deskew_image
from .layout import (
    detect_vote_boxes,
    detect_vote_boxes_legacy,
    find_row_bands,
    assign_boxes_to_rows,
    repair_missing_boxes,
    rows_from_boxes,
    refine_rows_with_separators,
    normalize_rows_box_aware,
    select_best_contiguous_rows,
    trim_rows_to_box_span,
    drop_short_boxless_rows,
    recover_missing_rows,
    uniformize_row_boundaries,
    expand_row_tops,
)
from .rectify import rectify_vote_box
from .enhance import enhance_vote_box
from .segmentation import (
    remove_border_components,
    strip_thin_border_pieces,
    kill_outer_frame,
    clean_components,
)
from .mnist import stable_crop
from .infer import load_mnist28_model, classify_with_variants, is_blank_or_noise
from .debug import DebugWriter
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../ballot_reader
REPO_DIR = PKG_DIR.parent                          # .../Handwritten-Digit-Recognition
MODEL_PATH_28 = str(REPO_DIR / "tf-cnn-model.keras")


def _align_boxes_to_rows(boxes: List[Optional[tuple]], rows: List[tuple]) -> List[Optional[tuple]]:
    """
    Force boxes to be exactly one entry per row (same ordering), padding with None
    or trimming if needed. This prevents IndexError and makes output deterministic.
    """
    if boxes is None:
        boxes = []
    target = len(rows)
    if len(boxes) == target:
        return boxes
    if len(boxes) > target:
        return boxes[:target]
    # pad
    return boxes + [None] * (target - len(boxes))


def process_ballot(
    img_path: str,
    model,
    config: BallotConfig,
    debug_writer: Optional[DebugWriter] = None
) -> List[Dict[str, Any]]:
    """Run the full ballot processing pipeline on a single image."""
    img = read_image(img_path)
    if img is None:
        return []
    
    
    def row_box_overlap_frac(
        row: Tuple[int, int],
        box: Tuple[int, int, int, int],
    ) -> float:
        """Fraction of box height that lies inside row band."""
        ry1, ry2 = row
        x, y, w, h = box
        by1, by2 = y, y + h
        inter = max(0, min(ry2, by2) - max(ry1, by1))
        return inter / float(h + 1e-6)
    
    
    # 1. Preprocessing
    img = normalize_illumination(img)
    img, angle = deskew_image(img)

    # 2. Layout Detection
    #    Detect vote boxes (right-side ROI), then derive rows from the detected
    #    boxes. This avoids fragile row-first detection being confused by photos,
    #    logos and text.
    raw_boxes = detect_vote_boxes(img, config)

    expected = getattr(config, "expected_boxes", None)
    # --- Add near the top of process_ballot (after expected is known) ---
    def _body_header_cut(img_h: int) -> int:
        # config.min_y_frac is already used across layout functions as a header cut
        return int(getattr(config, "min_y_frac", 0.0) * img_h)

    def _filter_body_boxes(boxes, header_cut: int):
        if not boxes:
            return []
        out = []
        for (x, y, w, h) in boxes:
            cy = y + 0.5 * h
            if cy >= header_cut:
                out.append((x, y, w, h))
        return out

    def _filter_body_rows(rows, header_cut: int):
        if not rows:
            return []
        # keep any row that extends into body (so header slivers don’t poison coverage)
        return [(y1, y2) for (y1, y2) in rows if y2 > header_cut]

    def _longest_none_run(xs):
        best = cur = 0
        for v in xs:
            if v is None:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best
    def _box_interior_ink_frac(img_bgr, box, border_frac=0.20) -> float:
        x, y, w, h = box
        H, W = img_bgr.shape[:2]
        x0 = max(0, min(W - 1, x))
        y0 = max(0, min(H - 1, y))
        x1 = max(0, min(W, x + w))
        y1 = max(0, min(H, y + h))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return 1.0

        roi = img_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ignore border (printed box line)
        bx = int(border_frac * (x1 - x0))
        by = int(border_frac * (y1 - y0))
        inner = gray[by: (y1 - y0) - by, bx: (x1 - x0) - bx]
        if inner.size == 0:
            return 1.0

        # ink fraction: how many pixels are "dark"
        thr = int(np.percentile(inner, 60))  # adaptive-ish
        ink = (inner < thr).mean()
        return float(ink)
    def _row_coverage_suspicious(rows_body, expected_rows, img_h, header_cut) -> bool:
        if not rows_body:
            return True
        denom = max(1.0, float(img_h - header_cut))
        k = min(len(rows_body), expected_rows) if expected_rows else len(rows_body)
        top = float(max(header_cut, rows_body[0][0]))
        bottom = float(rows_body[k - 1][1])
        cov = max(0.0, bottom - top) / denom
        # Tuneable: 0.70 tends to catch “rows are cramped in the wrong region”
        return cov < 0.70

    def _should_fallback_to_legacy(img_bgr, raw_boxes, rows, img_shape, expected_rows) -> bool:
        H, W = img_shape[:2]
        header_cut = _body_header_cut(H)

        boxes_body = _filter_body_boxes(raw_boxes, header_cut)
        if boxes_body:
            rights = np.array([(x + w) / float(W) for (x, y, w, h) in boxes_body], dtype=np.float32)
            gaps   = np.array([(W - (x + w)) / float(W) for (x, y, w, h) in boxes_body], dtype=np.float32)

            # vote boxes should be very near the right edge; tune these per your forms
            if float(np.median(rights)) < 0.92:
                return True
            if float(np.median(gaps)) > 0.08:
                return True

        # --- Condition E: boxes too large to be vote boxes (likely photo frames)
        if boxes_body:
            ws = np.array([w / float(W) for (x, y, w, h) in boxes_body], dtype=np.float32)
            hs = np.array([h / float(H) for (x, y, w, h) in boxes_body], dtype=np.float32)

            # these numbers are intentionally conservative; adjust once you inspect a few ballots
            if float(np.median(ws)) > 0.20:
                return True
            if float(np.median(hs)) > 0.10:
                return True
        
        
        rows_body = _filter_body_rows(rows, header_cut)

        # --- Condition A: significantly fewer boxes than expected (ignore header)
        if expected_rows:
            if len(boxes_body) < max(3, int(0.70 * expected_rows)):
                return True

        # --- Condition B: many rows have no assigned box (use ASSIGNED, not repaired)
        # Evaluate only the first expected rows in the body, to avoid header artifacts.
        if rows_body and boxes_body and expected_rows:
            eval_rows = rows_body[:expected_rows] if len(rows_body) >= expected_rows else rows_body
            assigned = assign_boxes_to_rows(boxes_body, eval_rows, overlap_thresh=0.50)  # existing fn :contentReference[oaicite:3]{index=3}
            missing = sum(1 for a in assigned if a is None)
            missing_frac = missing / float(max(1, len(eval_rows)))

            # Your example looks like ~4/15 missing => 0.27; you said “too many”
            # so set threshold at ~0.30 (tune as needed).
            if missing_frac >= 0.30:
                return True

            # Also catch “big gap” patterns (e.g. 3+ consecutive NULL rows)
            if _longest_none_run(assigned) >= 3:
                return True

        # --- Condition C: row coverage suspiciously low (below header cut)
        if expected_rows and len(rows_body) >= max(2, int(0.6 * expected_rows)):
            if _row_coverage_suspicious(rows_body, expected_rows, H, header_cut):
                return True
        # --- Condition F: assigned boxes overlap rows poorly
        if rows_body and boxes_body and expected_rows:
            eval_rows = rows_body[:expected_rows] if len(rows_body) >= expected_rows else rows_body
            assigned = assign_boxes_to_rows(boxes_body, eval_rows, overlap_thresh=0.10)  # looser threshold for scoring
            overlaps = []
            for r, b in zip(eval_rows, assigned):
                if b is None:
                    overlaps.append(0.0)
                else:
                    overlaps.append(row_box_overlap_frac(r, b))
            if float(np.median(overlaps)) < 0.35:
                return True
        
        
        # --- Condition G: detected "boxes" have too much ink inside (likely portraits/logos)
        if boxes_body:
            # sample up to 8 boxes for speed
            sample = boxes_body[:: max(1, len(boxes_body)//8)]
            inks = [_box_interior_ink_frac(img_bgr, b) for b in sample]
            # Vote boxes should be mostly empty; portraits are not.
            # Start with 0.18 and tune after you print inks once.
            if float(np.median(inks)) > 0.18:
                return True
        return False

    def _layout_score(raw_boxes, rows, img_shape, expected_rows) -> float:
        """
        Score used only to decide if legacy is *better* than new.
        Higher is better.
        """
        H, W = img_shape[:2]
        header_cut = _body_header_cut(H)
        boxes_body = _filter_body_boxes(raw_boxes, header_cut)
        rows_body = _filter_body_rows(rows, header_cut)

        n = len(boxes_body)
        score = float(n)

        if expected_rows:
            score -= 2.0 * float(abs(n - expected_rows))  # closeness to expected count

        # reward sane coverage (below header)
        if rows_body:
            denom = max(1.0, float(H - header_cut))
            k = min(len(rows_body), expected_rows) if expected_rows else len(rows_body)
            top = float(max(header_cut, rows_body[0][0]))
            bottom = float(rows_body[k - 1][1])
            cov = max(0.0, bottom - top) / denom
            score += 5.0 * cov

        # penalize “x scatter” (boxes should form a tight column)
        if len(boxes_body) >= 4:
            xs = np.array([b[0] for b in boxes_body], dtype=np.float32)
            score -= 2.0 * float(np.std(xs) / max(1.0, W))

        return score
    def _rows_failed(rows: List[tuple], expected_rows: Optional[int], img_h: int) -> bool:
        if not rows:
            return True
        if expected_rows is None or int(expected_rows) <= 0:
            return False
        exp = int(expected_rows)
        if len(rows) < max(1, int(0.6 * exp)):
            return True
        if len(rows) >= exp:
            top = float(rows[0][0])
            bottom = float(rows[exp - 1][1])
            usable = max(1.0, float(img_h) - top)
            span = max(0.0, bottom - top)
            coverage = span / usable
            if coverage < 0.95:
                return True
        return False
    rows = rows_from_boxes(raw_boxes, img.shape, config)

    if rows:
        cys = np.array([(y1 + y2) / 2.0 for (y1, y2) in rows], dtype=np.float32)
        med_gap = float(np.median(np.diff(np.sort(cys)))) if len(cys) >= 2 else None
        rows = refine_rows_with_separators(img, rows, config, med_gap=med_gap)

    if not rows or (expected is not None and len(rows) < max(2, int(expected) // 2)):
        rows = find_row_bands(img, raw_boxes, config)

    expected = getattr(config, "expected_boxes", None)

    # Fallback: if separator-based gave too few rows but we have enough boxes, use box-anchored rows
    if expected is not None and len(raw_boxes) >= expected:
        if not rows or len(rows) < expected:
            box_rows = rows_from_boxes(raw_boxes, img.shape, config)
            if len(box_rows) >= expected:
                rows = box_rows
                if rows:
                    cys = np.array([(y1 + y2) / 2.0 for (y1, y2) in rows], dtype=np.float32)
                    med_gap = float(np.median(np.diff(np.sort(cys)))) if len(cys) >= 2 else None
                    rows = refine_rows_with_separators(img, rows, config, med_gap=med_gap)
    elif not rows:
        rows = rows_from_boxes(raw_boxes, img.shape, config)
        if rows:
            cys = np.array([(y1 + y2) / 2.0 for (y1, y2) in rows], dtype=np.float32)
            med_gap = float(np.median(np.diff(np.sort(cys)))) if len(cys) >= 2 else None
            rows = refine_rows_with_separators(img, rows, config, med_gap=med_gap)
    expected = getattr(config, "expected_boxes", None)
    exp = int(expected) if expected is not None and int(expected) > 0 else None

    if _should_fallback_to_legacy(img, raw_boxes, rows, img.shape, exp):
        legacy_boxes = detect_vote_boxes_legacy(img, config)  # existing fn :contentReference[oaicite:4]{index=4}
        if legacy_boxes:
            # Build rows from legacy using ONLY the permitted row-building steps
            legacy_rows = rows_from_boxes(legacy_boxes, img.shape, config)
            if legacy_rows:
                cys = np.array([(y1 + y2) / 2.0 for (y1, y2) in legacy_rows], dtype=np.float32)
                med_gap = float(np.median(np.diff(np.sort(cys)))) if len(cys) >= 2 else None
                legacy_rows = refine_rows_with_separators(img, legacy_rows, config, med_gap=med_gap)
            if not legacy_rows or (exp is not None and len(legacy_rows) < max(2, exp // 2)):
                legacy_rows = find_row_bands(img, legacy_boxes, config)

            # Only switch if legacy is clearly better
            new_score = _layout_score(raw_boxes, rows, img.shape, exp)
            leg_score = _layout_score(legacy_boxes, legacy_rows, img.shape, exp)

            if leg_score > new_score:
                raw_boxes = legacy_boxes
                rows = legacy_rows
                
    # ---- Normalize: ONLY split a band if it contains 2+ vote-box centers
    rows = normalize_rows_box_aware(rows, raw_boxes, img.shape, config)

    # ---- Trim header/footer rows, but KEEP blank interior candidate rows.
    rows = trim_rows_to_box_span(rows, raw_boxes, min_overlap=0.30)

    # ---- Drop obvious spurious slivers that have no box and are much shorter than typical candidate rows
    rows = drop_short_boxless_rows(
        rows,
        raw_boxes,
        min_overlap=0.30,
        height_frac=0.6,
        expected_rows=config.expected_boxes,
    )

    # ---- If we have fewer rows than expected, split largest band(s) to recover missing candidate row.
    if config.expected_boxes is not None and len(rows) < config.expected_boxes:
        rows = recover_missing_rows(
            rows,
            expected=int(config.expected_boxes),
            img_height=img.shape[0],
        )

    # ---- If expected count is known, select best contiguous block + force uniform grid
    if config.expected_boxes is not None:
        rows = select_best_contiguous_rows(
            rows,
            raw_boxes,
            expected=int(config.expected_boxes),
            min_overlap=0.30,
        )
        rows = uniformize_row_boundaries(
            rows,
            expected=int(config.expected_boxes),
            img_height=img.shape[0],
            max_height_cv=None,
            boxes=raw_boxes,
        )

    # ---- Expand row tops slightly so content isn't cut off.
    rows = expand_row_tops(rows, img.shape[0], expand_frac=0.08)

    # ---- Assign detected vote boxes to these rows, repair missing ones
    assigned = assign_boxes_to_rows(raw_boxes, rows, overlap_thresh=0.50)
    boxes = repair_missing_boxes(rows, assigned, raw_boxes, img.shape)
    
    # CRITICAL: enforce one box per row to prevent IndexError and keep output stable
    boxes = _align_boxes_to_rows(boxes, rows)
    
    if debug_writer:
        debug_writer.draw_overlay(img, boxes, rows, "layout_overlay", assigned=assigned)
        # If your DebugWriter doesn't have save_text, this will safely no-op.
        try:
            debug_writer.save_text(
                f"img={os.path.basename(img_path)}\n"
                f"rows={len(rows)} raw_boxes={len(raw_boxes)} boxes={len(boxes)}\n",
                ["_meta"],
                "counts.txt",
            )
        except Exception:
            pass

    results: List[Dict[str, Any]] = []

    for idx, (y1, y2) in enumerate(rows):
        row_num = idx + 1
        box = boxes[idx] if idx < len(boxes) else None

        # If box is missing, record NULL deterministically and continue
        if box is None:
            results.append({
                "row": row_num,
                "digit": "NULL",
                "meta": {
                    "row": row_num,
                    "box": None,
                    "is_blank": True,
                    "reason": "missing_box_for_row",
                }
            })
            continue

        x, y, w, h = box
        # Clamp ROI to image bounds (extra safety)
        H, W = img.shape[:2]
        x = int(max(0, min(x, W - 1)))
        y = int(max(0, min(y, H - 1)))
        w = int(max(1, min(w, W - x)))
        h = int(max(1, min(h, H - y)))

        vote_box_roi = img[y:y + h, x:x + w]

        # 3. Rectification
        rectified, rmeta = rectify_vote_box(vote_box_roi, config)

        # 4. Enhancement
        enhanced_255, mask_bool, emeta = enhance_vote_box(rectified, config)

        # 5. Segmentation & Cleaning
        mask = remove_border_components(enhanced_255, config)
        mask = strip_thin_border_pieces(mask)
        mask = kill_outer_frame(mask)
        mask = clean_components(mask)

        # 6. Blank/Noise Gating
        is_blank, bmeta = is_blank_or_noise(mask, config)

        digit = "NULL"
        decision_meta: Dict[str, Any] = {
            "row": row_num,
            "box": (x, y, w, h),
            "rectify": rmeta,
            "enhance": emeta,
            "blank_gate": bmeta,
            "is_blank": is_blank,
        }

        if not is_blank:
            # 7. MNIST Prep & Inference
            digit_mask, cmeta = stable_crop(mask)
            if digit_mask is not None:
                pred, probs, imeta = classify_with_variants(model, digit_mask, config)
                digit = pred if pred is not None else "NULL"
                decision_meta.update({
                    "digit": digit,
                    "crop": cmeta,
                    "inference": imeta,
                })

                if debug_writer:
                    debug_writer.save_image(rectified, [f"row_{row_num:02d}", "box_00"], "rectified")
                    debug_writer.save_image(enhanced_255, [f"row_{row_num:02d}", "box_00"], "enhanced")
                    debug_writer.save_image(
                        np.where(mask, 0, 255).astype(np.uint8),
                        [f"row_{row_num:02d}", "box_00"],
                        "mask",
                    )
                    debug_writer.save_image(
                        np.where(digit_mask, 0, 255).astype(np.uint8),
                        [f"row_{row_num:02d}", "box_00"],
                        "mnist_input",
                    )
                    debug_writer.save_decision(decision_meta, [f"row_{row_num:02d}", "box_00"])

        results.append({
            "row": row_num,
            "digit": digit,
            "meta": decision_meta
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Ballot Reader Pro CLI")
    parser.add_argument("--input", required=True, help="Path to ballot image or directory")
    parser.add_argument("--out", default="debug_output", help="Output directory for debug visuals")
    parser.add_argument("--model", help="Path to Keras model file")
    parser.add_argument("--expected", type=int, default=None, help="Expected number of candidate rows (e.g. 15)")
    args = parser.parse_args()

    config = BallotConfig()
    if args.model:
        config.model_path = args.model
    else:
        config.model_path = MODEL_PATH_28
    if args.expected is not None:
        config.expected_boxes = args.expected

    try:
        model = load_mnist28_model(config.model_path)
        print(f"[INFO] Loaded model from {config.model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    input_path = args.input
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    else:
        print(f"[ERROR] Invalid input path: {input_path}")
        return

    for f in files:
        ballot_id = os.path.splitext(os.path.basename(f))[0]
        print(f"[INFO] Processing ballot: {ballot_id}")

        debug_writer = DebugWriter(args.out, ballot_id)
        results = process_ballot(f, model, config, debug_writer)

        # Save summary results
        summary = [{"row": r["row"], "digit": r["digit"]} for r in results]
        write_json(os.path.join(debug_writer.out_dir, "results.json"), summary)
        write_csv(os.path.join(debug_writer.out_dir, "results.csv"), summary, ["row", "digit"])

        print(f"[INFO] Results for {ballot_id}:")
        for r in summary:
            print(f"  Row {r['row']:02d}: {r['digit']}")


if __name__ == "__main__":
    main()
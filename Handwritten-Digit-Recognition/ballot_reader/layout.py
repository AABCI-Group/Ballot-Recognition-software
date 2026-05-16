import cv2
import numpy as np
from typing import List, Tuple, Optional
from .config import BallotConfig

import numpy as np
from typing import List, Tuple, Optional

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

def row_has_any_box(
    row: Tuple[int, int],
    boxes: List[Tuple[int, int, int, int]],
    min_overlap: float = 0.30,
) -> bool:
    """True if any box overlaps this row by >= min_overlap fraction of box height."""
    for b in boxes or []:
        if row_box_overlap_frac(row, b) >= min_overlap:
            return True
    return False

def filter_rows_by_boxes(
    rows: List[Tuple[int, int]],
    boxes: List[Tuple[int, int, int, int]],
    min_overlap: float = 0.30,
) -> List[Tuple[int, int]]:
    """Keep only rows that contain at least one vote box overlap."""
    if not rows:
        return rows
    if not boxes:
        return rows
    kept = [r for r in rows if row_has_any_box(r, boxes, min_overlap=min_overlap)]
    # If filtering removes everything (bad box detection), fall back to original rows
    return kept if kept else rows

def select_best_contiguous_rows(
    rows: List[Tuple[int, int]],
    boxes: List[Tuple[int, int, int, int]],
    expected: Optional[int],
    min_overlap: float = 0.30,
) -> List[Tuple[int, int]]:
    """
    If expected is set, choose the best contiguous block of that many rows.
    Score each candidate block by how many unique boxes are well-explained by the block.
    This avoids 'drop from top' shifting errors.
    """
    if not rows or not expected or expected <= 0:
        return rows
    if len(rows) <= expected:
        return rows

    # Precompute box->best row overlap within any given row list by using overlap function.
    # For block scoring, count boxes that overlap any row in block above threshold.
    def block_score(block: List[Tuple[int, int]]) -> int:
        score = 0
        for b in boxes or []:
            best = 0.0
            for r in block:
                best = max(best, row_box_overlap_frac(r, b))
            if best >= min_overlap:
                score += 1
        return score

    best_i = 0
    best_score = -1
    best_tiebreak = None  # prefer blocks with tighter row heights / less spread

    for i in range(0, len(rows) - expected + 1):
        block = rows[i : i + expected]
        s = block_score(block)

        # Tiebreak: prefer blocks with more uniform heights
        hs = np.array([y2 - y1 for (y1, y2) in block], dtype=np.float32)
        spread = float(np.std(hs)) if len(hs) else 1e9
        tiebreak = (spread, block[0][0])  # lower spread, then higher up

        if s > best_score or (s == best_score and (best_tiebreak is None or tiebreak < best_tiebreak)):
            best_score = s
            best_i = i
            best_tiebreak = tiebreak

    return rows[best_i : best_i + expected]


def trim_rows_to_box_span(
    rows: List[Tuple[int, int]],
    boxes: List[Tuple[int, int, int, int]],
    min_overlap: float = 0.30,
) -> List[Tuple[int, int]]:
    """
    Drop only the leading and trailing rows that have no vote-box overlap,
    but keep all rows between the first and last box, even if some of those
    interior rows are blank.

    This prevents header/footer rows from being counted as candidates while
    still preserving empty candidate rows.
    """
    if not rows or not boxes:
        return rows

    # Precompute which rows see any box and basic height stats.
    touches: List[bool] = []
    heights: List[int] = []
    for (y1, y2) in rows:
        touches.append(row_has_any_box((y1, y2), boxes, min_overlap=min_overlap))
        heights.append(max(1, y2 - y1))

    touching_indices: List[int] = [i for i, t in enumerate(touches) if t]
    if not touching_indices:
        # If box detection was very poor, do not drop anything; downstream
        # logic (like repair_missing_boxes) can still recover boxes.
        return rows

    med_h = float(np.median(heights)) if heights else 0.0

    def looks_like_candidate(idx: int) -> bool:
        """Heuristic: row height close to median or already has a box."""
        if idx < 0 or idx >= len(rows):
            return False
        if touches[idx]:
            return True
        if med_h <= 0:
            return False
        h = float(heights[idx])
        # Treat header/footer slivers (very short) as non-candidates.
        return 0.7 * med_h <= h <= 1.35 * med_h

    first = touching_indices[0]
    last = touching_indices[-1]

    # Allow any contiguous run of plausible candidate rows just above/below the
    # box span to stay, even if their own boxes were missed by detection.
    # This is more robust on ballots where multiple end rows are unmarked.
    start = first
    end = last

    # Walk upwards from the first touching row while rows still look
    # candidate-like (by height or existing box overlap).
    i = first - 1
    while i >= 0 and looks_like_candidate(i):
        start = i
        i -= 1

    # Walk downwards from the last touching row with the same heuristic.
    i = last + 1
    while i < len(rows) and looks_like_candidate(i):
        end = i
        i += 1

    return rows[start : end + 1]


def drop_short_boxless_rows(
    rows: List[Tuple[int, int]],
    boxes: List[Tuple[int, int, int, int]],
    min_overlap: float = 0.30,
    height_frac: float = 0.6,
    expected_rows: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Remove obvious spurious rows that are both:
      - significantly shorter than the median row height, and
      - have no overlapping vote box.

    This is aimed at thin split bands caused by scanner seams or stray rules,
    while preserving normal-height candidate rows (even if their box was missed).

    If expected_rows is set and dropping would leave expected_rows - 1 rows,
    we keep all rows to avoid dropping a real candidate row whose box was missed.
    """
    if not rows:
        return rows

    heights = [max(1, y2 - y1) for (y1, y2) in rows]
    med_h = float(np.median(heights)) if heights else 0.0
    if med_h <= 0:
        return rows

    kept: List[Tuple[int, int]] = []
    for r, h in zip(rows, heights):
        has_box = row_has_any_box(r, boxes, min_overlap=min_overlap)
        if (not has_box) and (h < height_frac * med_h):
            # Likely a spurious sliver; drop it.
            continue
        kept.append(r)

    # If we would end up one short of expected, keep all rows (preserve missed box row)
    if expected_rows is not None and len(kept) == expected_rows - 1:
        return rows
    return kept


def recover_missing_rows(
    rows: List[Tuple[int, int]],
    expected: int,
    img_height: int,
    min_height: int = 10,
) -> List[Tuple[int, int]]:
    """
    When we have fewer rows than expected (e.g. a separator was missed), split
    oversized row band(s) into equal-height strips using median row height,
    so boundaries align with a uniform grid instead of arbitrary midpoints.
    """
    if not rows or expected <= 0 or len(rows) >= expected:
        return rows

    heights = np.array([y2 - y1 for (y1, y2) in rows], dtype=np.float32)
    # Use median row height as typical single-candidate height (avoid huge bands inflating median)
    med_h = float(np.median(heights))
    if med_h <= 0:
        med_h = max(min_height, img_height / max(expected, 1))

    out = list(rows)
    # Repeatedly split the largest band into equal-height strips until we have expected rows
    while len(out) < expected:
        indexed = [(i, y1, y2, y2 - y1) for i, (y1, y2) in enumerate(out)]
        if not indexed:
            break
        indexed.sort(key=lambda t: -t[3])
        i, y1, y2, h = indexed[0]
        if h < 2 * min_height:
            break
        need = expected - len(out)
        # How many rows should this band become? (by height relative to median)
        k = max(2, min(need + 1, int(round(h / med_h))))
        if k <= 1:
            break
        # Split into k equal-height strips
        step = (y2 - y1) / k
        new_rows = []
        for j in range(k):
            a = int(y1 + j * step)
            b = int(y1 + (j + 1) * step) if j < k - 1 else y2
            if b - a >= min_height:
                new_rows.append((a, b))
        if len(new_rows) < 2:
            break
        out = out[:i] + new_rows + out[i + 1:]
        out.sort(key=lambda r: r[0])

    return out[:expected] if len(out) > expected else out


def uniformize_row_boundaries(
    rows: List[Tuple[int, int]],
    expected: int,
    img_height: int,
    max_height_cv: Optional[float] = None,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    When we have exactly `expected` rows, replace boundaries with a uniform grid.
    Span is chosen so it covers all candidates: from first row top to at least
    the bottom of the bottommost vote box (and image), so the last candidate is
    never cut off when pre-uniformize rows were merged too high.
    If max_height_cv is set, only apply when height coefficient of variation
    exceeds it; otherwise always apply when len(rows)==expected.
    """
    if not rows or len(rows) != expected or expected <= 0:
        return rows
    if max_height_cv is not None:
        heights = np.array([y2 - y1 for (y1, y2) in rows], dtype=np.float32)
        if len(heights) < 2:
            return rows
        mean_h = float(np.mean(heights))
        std_h = float(np.std(heights))
        cv = std_h / (mean_h + 1e-6)
        if cv <= max_height_cv:
            return rows

    y0 = rows[0][0]
    y_end = rows[-1][1]
    # Extend span downward so we never cut off the last candidate (e.g. O'Brien).
    # Use boxes' vertical extent and image bottom when row boundaries are wrong.
    if boxes:
        box_bottom = max(b[1] + b[3] for b in boxes)
        med_h = int(np.median([b[3] for b in boxes])) if boxes else 0
        margin = max(med_h // 2, 20)
        y_end = max(y_end, box_bottom + margin)
    y_end = min(img_height - 1, max(y_end, int(0.92 * img_height)))
    span = max(1, y_end - y0)
    step = span / expected
    return [
        (int(y0 + i * step), int(y0 + (i + 1) * step))
        for i in range(expected)
    ]


def expand_row_tops(
    rows: List[Tuple[int, int]],
    img_height: int,
    expand_frac: float = 0.08,
) -> List[Tuple[int, int]]:
    """
    Expand each row's top (y1) upward so content is not cut off; cap to avoid overlap.
    """
    if not rows or len(rows) <= 1:
        return rows

    heights = [y2 - y1 for (y1, y2) in rows]
    med_h = float(np.median(heights)) if heights else 50.0
    expand_max = int(expand_frac * med_h)
    expand_max = max(2, min(expand_max, int(0.15 * med_h)))

    out: List[Tuple[int, int]] = []
    prev_y2 = 0
    for (y1, y2) in rows:
        # Expand upward (decrease y1) without overlapping previous row
        expand = min(expand_max, y1 - prev_y2, int(0.12 * (y2 - y1)))
        if expand > 0:
            new_y1 = max(0, max(prev_y2, y1 - expand))
            out.append((int(new_y1), int(y2)))
        else:
            out.append((int(y1), int(y2)))
        prev_y2 = out[-1][1]

    return out


def _cluster_1d(values: List[float], tol: float) -> List[List[float]]:
    """Cluster sorted 1D values by proximity tolerance."""
    if not values:
        return []
    v = sorted(values)
    clusters = [[v[0]]]
    for x in v[1:]:
        if abs(x - clusters[-1][-1]) <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return clusters


def _dedupe_boxes_by_cy(
    boxes: List[Tuple[int, int, int, int]],
    median_h: float,
    tol_frac: float = 0.35,
) -> List[Tuple[int, int, int, int]]:
    """Merge/dedupe boxes that are essentially the same row (by cy)."""
    if not boxes:
        return []
    tol = float(tol_frac * max(1.0, median_h))
    items = []
    for (x, y, w, h) in boxes:
        cy = y + h / 2.0
        items.append((cy, x, y, w, h))
    items.sort(key=lambda t: t[0])

    cys = [t[0] for t in items]
    cy_clusters = _cluster_1d(cys, tol=tol)

    # For each cluster, pick representative by median geometry (robust)
    out = []
    idx = 0
    for cl in cy_clusters:
        k = len(cl)
        group = items[idx:idx + k]
        idx += k
        xs = np.array([g[1] for g in group], dtype=np.float32)
        ys = np.array([g[2] for g in group], dtype=np.float32)
        ws = np.array([g[3] for g in group], dtype=np.float32)
        hs = np.array([g[4] for g in group], dtype=np.float32)

        # Keep the right-most within the cluster if you want stability:
        # but use median y/h to be robust.
        rep_x = int(np.median(xs))
        rep_y = int(np.median(ys))
        rep_w = int(np.median(ws))
        rep_h = int(np.median(hs))
        out.append((rep_x, rep_y, rep_w, rep_h))

    out.sort(key=lambda b: b[1])
    return out


def rows_from_boxes(
    boxes: List[Tuple[int, int, int, int]],
    img_shape: Tuple[int, int, int],
    config: BallotConfig,
) -> List[Tuple[int, int]]:
    """
    Build candidate row bands anchored to vote-box geometry.

    Returns: list of (y1, y2) row bands, one per candidate.
    """
    H, W = img_shape[:2]
    if not boxes:
        return []

    # ---- Step 1: robust stats
    hs = np.array([b[3] for b in boxes], dtype=np.float32)
    med_h = float(np.median(hs)) if len(hs) else max(1.0, 0.05 * H)

    # Deduplicate boxes by cy proximity (removes double-detections)
    boxes_d = _dedupe_boxes_by_cy(boxes, median_h=med_h, tol_frac=0.35)

    # Compute centers
    centers = []
    for (x, y, w, h) in boxes_d:
        cy = y + h / 2.0
        centers.append((cy, y, h))
    centers.sort(key=lambda t: t[0])

    cys = np.array([c[0] for c in centers], dtype=np.float32)
    if len(cys) < 2:
        # Single row fallback
        header_cut = int(config.min_y_frac * H)
        cy = float(cys[0])
        y1 = max(header_cut, int(cy - 1.2 * med_h))
        y2 = min(H - 1, int(cy + 1.2 * med_h))
        if y2 - y1 < int(0.6 * med_h):
            y2 = min(H - 1, y1 + int(0.6 * med_h))
        return [(y1, y2)]

    # Median gap between consecutive centers (ignore huge outliers)
    diffs = np.diff(np.sort(cys))
    diffs = diffs[diffs > 1.0]  # remove zeros
    if len(diffs) == 0:
        med_gap = float(2.2 * med_h)
    else:
        d_med = float(np.median(diffs))
        # ignore absurdly large gaps for median gap estimate
        trimmed = diffs[diffs < 3.0 * d_med] if np.any(diffs < 3.0 * d_med) else diffs
        med_gap = float(np.median(trimmed)) if len(trimmed) else d_med

    header_cut = int(config.min_y_frac * H)

    # ---- Step 2: drop header boxes (B1–B2)
    # (A) hard cutoff
    keep = [i for i, cy in enumerate(cys) if cy >= header_cut]

    if not keep:
        return []

    cys2 = cys[keep]

    # (B) spacing heuristic: header region often has a big jump to first candidate
    # drop boxes that are too high if they sit above a large gap to the next one
    # Rule: if (next_cy - cy) > 1.8*med_gap and cy is near top/header, treat as header.
    filtered = []
    for j in range(len(cys2)):
        cy = float(cys2[j])
        nxt = float(cys2[j + 1]) if j + 1 < len(cys2) else None
        if nxt is not None:
            if (nxt - cy) > 1.8 * med_gap and cy < (header_cut + 1.2 * med_gap):
                # likely instruction/header box
                continue
        filtered.append(cy)

    cys_final = np.array(filtered, dtype=np.float32) if filtered else cys2

    # (C) enforce expected candidate count if set (e.g. 15)
    # Drop from the top only (don’t mess with candidate rows).
    expected = getattr(config, "expected_boxes", None)
    if expected is not None and len(cys_final) > expected:
        # drop the highest boxes until count matches
        excess = len(cys_final) - int(expected)
        cys_final = cys_final[excess:]

    # ---- Step 3: row boundaries from candidate centers
    cys_final = np.sort(cys_final)
    n = len(cys_final)
    if n == 0:
        return []

    # Boundaries are midpoints between centers, with padded ends
    pad = 0.55 * med_gap
    bnds = np.zeros(n + 1, dtype=np.float32)
    bnds[0] = max(float(header_cut), float(cys_final[0] - pad))
    for i in range(n - 1):
        bnds[i + 1] = 0.5 * (cys_final[i] + cys_final[i + 1])
    bnds[n] = min(float(H - 1), float(cys_final[-1] + pad))

    # Convert to integer rows and clamp
    rows: List[Tuple[int, int]] = []
    min_h = int(max(10, 0.6 * med_h))
    for i in range(n):
        y1 = int(max(0, np.floor(bnds[i])))
        y2 = int(min(H - 1, np.ceil(bnds[i + 1])))
        if y2 - y1 < min_h:
            # expand around center if too small
            cy = float(cys_final[i])
            y1 = int(max(0, cy - 0.5 * min_h))
            y2 = int(min(H - 1, y1 + min_h))
        rows.append((y1, y2))

    # Make monotone + non-overlapping
    rows_fixed = []
    prev_y2 = 0
    for (y1, y2) in rows:
        y1 = max(y1, prev_y2)
        y2 = max(y2, y1 + min_h)
        y2 = min(y2, H - 1)
        rows_fixed.append((int(y1), int(y2)))
        prev_y2 = y2
    return rows_fixed


def refine_rows_with_separators(
    img_bgr: np.ndarray,
    rows: List[Tuple[int, int]],
    config: BallotConfig,
    med_gap: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """
    Snap row boundaries to nearby separator y's (if found),
    but never add/remove rows and never break min-height constraints.
    """
    if not rows:
        return rows

    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    # Light separator detector: long-ish horizontal strokes in left 82% (avoid box column)
    roi = th[:, : int(0.82 * W)]
    proj = roi.mean(axis=1) / 255.0
    k = max(9, int(0.012 * H) | 1)
    proj_s = cv2.blur(proj.astype(np.float32).reshape(-1, 1), (1, k)).ravel()
    thr = float(np.percentile(proj_s, 92))
    thr = max(thr, 0.12)

    sep_ys = []
    for y in range(2, H - 2):
        v = proj_s[y]
        if v > thr and v >= proj_s[y - 1] and v >= proj_s[y + 1]:
            sep_ys.append(y)

    if not sep_ys:
        return rows

    # cluster separators
    tol = max(4, int(0.008 * H))
    sep_ys = sorted(sep_ys)
    clustered = []
    cur = [sep_ys[0]]
    for y in sep_ys[1:]:
        if abs(y - cur[-1]) <= tol:
            cur.append(y)
        else:
            clustered.append(int(np.median(cur)))
            cur = [y]
    clustered.append(int(np.median(cur)))
    sep_ys = clustered

    # estimate med_gap from rows if not provided
    if med_gap is None:
        cys = np.array([(y1 + y2) / 2.0 for (y1, y2) in rows], dtype=np.float32)
        if len(cys) >= 2:
            med_gap = float(np.median(np.diff(np.sort(cys))))
        else:
            med_gap = float(max(20, 0.06 * H))

    snap_tol = float(0.12 * med_gap)
    row_heights = [y2 - y1 for (y1, y2) in rows]
    med_h = float(np.median(row_heights)) if row_heights else 20.0
    min_h = int(max(10, 0.6 * med_h))

    # Convert rows into boundaries list, snap internal boundaries only
    bnds = [rows[0][0]] + [r[1] for r in rows]  # y0, y1, ..., yn
    bnds2 = bnds[:]

    for i in range(1, len(bnds2) - 1):
        y = bnds2[i]
        # nearest sep
        nearest = min(sep_ys, key=lambda s: abs(s - y))
        if abs(nearest - y) <= snap_tol:
            # propose snap, check local constraints
            y_prev = bnds2[i - 1]
            y_next = bnds2[i + 1]
            if (nearest - y_prev) >= min_h and (y_next - nearest) >= min_h:
                bnds2[i] = int(nearest)

    # rebuild rows
    out = []
    for y1, y2 in zip(bnds2[:-1], bnds2[1:]):
        if y2 - y1 >= min_h:
            out.append((int(y1), int(y2)))
        else:
            out.append((int(y1), int(y1 + min_h)))
    return out


def _filter_vote_box_column(
    boxes: List[Tuple[int, int, int, int]],
    img_width: int,
    config: BallotConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    Keep rectangles that align with the far-right printed vote-box column.

    Portraits and party logos can produce square-ish contours inside the broad
    right-side ROI, but their right edge ends well before the vote-box column.
    """
    if not boxes:
        return boxes

    min_right = float(getattr(config, "vote_box_min_right_edge_frac", 0.84)) * img_width
    edge_filtered = [b for b in boxes if (b[0] + b[2]) >= min_right]
    if edge_filtered:
        boxes = edge_filtered

    if len(boxes) >= 4:
        widths = np.array([b[2] for b in boxes], dtype=np.float32)
        med_w = float(np.median(widths)) if len(widths) else 0.0
        if med_w > 0:
            right_edges = np.array([b[0] + b[2] for b in boxes], dtype=np.float32)
            right_anchor = float(np.median(np.sort(right_edges)[len(right_edges) // 2:]))
            tolerance = max(0.035 * img_width, 0.65 * med_w)
            filtered = [b for b in boxes if (b[0] + b[2]) >= right_anchor - tolerance]
            if len(filtered) >= max(3, int(0.55 * len(boxes))):
                boxes = filtered

    boxes.sort(key=lambda b: b[1])
    return boxes


def detect_vote_boxes(img_bgr: np.ndarray, config: BallotConfig) -> List[Tuple[int, int, int, int]]:
    """Detect vote boxes (box-first).

    This detector intentionally ignores the left ~70–80% of the ballot (where
    photos/logos/text live) and focuses on a right-side ROI where vote boxes are
    printed. Boxes are detected via contour-based rectangle finding (approxPolyDP)
    rather than strict line detection, making it more tolerant of:
      - slight skew
      - broken/partial box borders
      - imperfect printing / crumpled paper

    Returns: list of (x, y, w, h) in **full-image coordinates**, sorted by y.
    """

    H, W = img_bgr.shape[:2]

    def crop_right_roi(img: np.ndarray) -> Tuple[np.ndarray, int]:
        """Crop right-side ROI and return (roi_bgr, x_offset)."""
        roi_frac = float(getattr(config, "vote_box_roi_frac", 0.27))
        pad_frac = float(getattr(config, "vote_box_roi_pad_frac", 0.02))
        roi_w = int(max(1, roi_frac * W))
        pad = int(max(0, pad_frac * W))

        x0 = max(0, W - roi_w - pad)
        x1 = min(W, W + pad)
        return img[:, x0:x1], x0

    def detect_boxes_in_roi(roi_bgr: np.ndarray, x_off: int) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Normalize illumination: robust to scan shadows.
        bg = cv2.medianBlur(gray, 51)
        norm = cv2.divide(gray, bg, scale=255)

        # Binarize: ink=255
        th = cv2.adaptiveThreshold(
            norm,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            7,
        )

        # Repair broken lines / strengthen edges.
        th = cv2.morphologyEx(
            th,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=2,
        )
        th = cv2.dilate(
            th,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        aspect_min = float(getattr(config, "vote_box_aspect_min", 0.65))
        aspect_max = float(getattr(config, "vote_box_aspect_max", 1.55))
        poly_eps = float(getattr(config, "vote_box_poly_eps_frac", 0.020))

        # Area thresholds are based on FULL IMAGE size, so they remain stable
        # even if ROI width changes.
        min_area = float(getattr(config, "vote_box_min_area_frac", config.min_area_frac)) * (W * H)
        max_area = float(getattr(config, "vote_box_max_area_frac", 0.045)) * (W * H)

        boxes: List[Tuple[int, int, int, int]] = []
        for c in contours:
            area_c = float(cv2.contourArea(c))
            if area_c <= 0:
                continue

            peri = float(cv2.arcLength(c, True))
            if peri <= 0:
                continue

            # Approximate contour to polygon, keep rectangles only
            approx = cv2.approxPolyDP(c, poly_eps * peri, True)
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            x, y, w, h = cv2.boundingRect(approx)
            box_area = float(w * h)
            if box_area < min_area or box_area > max_area:
                continue

            ar = w / float(h + 1e-6)
            if not (aspect_min <= ar <= aspect_max):
                continue

            # Extent rejects many non-box shapes that happen to be 4-sided
            extent = area_c / float(box_area + 1e-6)
            if extent < 0.25:
                continue

            boxes.append((int(x + x_off), int(y), int(w), int(h)))

        boxes.sort(key=lambda b: b[1])
        return boxes

    # --- Primary: right-side ROI detector
    roi, x0 = crop_right_roi(img_bgr)
    boxes = detect_boxes_in_roi(roi, x_off=x0)

    # Global sanity constraints.
    if boxes:
        boxes = [b for b in boxes if b[1] >= int(config.min_y_frac * H)]
        boxes = [b for b in boxes if b[0] >= int(config.min_x_frac * W)]
        boxes.sort(key=lambda b: b[1])

    if boxes:
        boxes = _filter_vote_box_column(boxes, W, config)

    # --- Conservative fallback: legacy whole-image contour detector.
    # Kept for backwards compatibility on scans where ROI detection fails.
    if not boxes:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bg = cv2.medianBlur(gray, 51)
        norm = cv2.divide(gray, bg, scale=255)
        th = cv2.adaptiveThreshold(
            norm,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            7,
        )
        th = cv2.morphologyEx(
            th,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand: List[Tuple[int, int, int, int]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / float(h + 1e-6)
            area = float(w * h)
            if (
                float(getattr(config, "vote_box_aspect_min", 0.65)) <= ar <= float(getattr(config, "vote_box_aspect_max", 1.55))
                and area > float(getattr(config, "vote_box_min_area_frac", config.min_area_frac)) * (W * H)
                and x > config.min_x_frac * W
                and y > config.min_y_frac * H
            ):
                cand.append((int(x), int(y), int(w), int(h)))
        cand.sort(key=lambda b: b[1])
        boxes = _filter_vote_box_column(cand, W, config)

    # If expected count is known, keep the most plausible K boxes.
    if boxes and config.expected_boxes is not None and len(boxes) > int(config.expected_boxes):
        k = int(config.expected_boxes)
        areas = np.array([b[2] * b[3] for b in boxes], dtype=np.float32)
        med_area = float(np.median(areas)) if len(areas) else 0.0

        def score(b: Tuple[int, int, int, int]) -> float:
            x, y, w, h = b
            ar = w / float(h + 1e-6)
            sq = 1.0 - abs(1.0 - ar)
            a = float(w * h)
            a_score = 1.0 - (abs(a - med_area) / (med_area + 1e-6)) if med_area > 0 else 0.0
            return 1.5 * (x / float(W)) + 1.0 * sq + 0.5 * a_score

        boxes = sorted(boxes, key=score, reverse=True)[:k]
        boxes.sort(key=lambda b: b[1])

    return boxes

def find_row_bands(
    img_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    config: BallotConfig,
) -> List[Tuple[int, int]]:
    """
    Robust row segmentation:
      - Detect row separators using BOTH:
          (A) morphology-based long horizontal rule detection
          (B) horizontal projection peak detection (works when rules are broken by icons/photos)
      - Cluster separator y positions (double/thick rules -> one)
      - Add synthetic top/bottom boundaries so missing top/bottom box doesn’t drop rows
      - Build rows between consecutive boundaries
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def _cluster_ys(ys: List[int], tol: int) -> List[int]:
        if not ys:
            return []
        ys = sorted(int(y) for y in ys)
        clusters = [[ys[0]]]
        for y in ys[1:]:
            if abs(y - clusters[-1][-1]) <= tol:
                clusters[-1].append(y)
            else:
                clusters.append([y])
        return [int(np.median(c)) for c in clusters]

    # --- Common binarization (ink = 255)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    # --- (A) Morphology-based horizontal rule detection (good when lines are continuous)
    def _sep_lines_morph(th_img: np.ndarray) -> List[int]:
        ys = []
        # Try two kernel widths to handle partial/broken lines better
        for frac in (0.35, 0.55):
            k_w = max(25, int(frac * W))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
            horiz = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, kernel, iterations=1)
            horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)), iterations=1)

            contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                # relaxed width check because icons/portraits break the rule
                if w >= 0.35 * W and h <= 0.05 * H:
                    ys.append(y + h // 2)
        return ys

    # --- (B) Projection-based detection (good when rules are broken)
    def _sep_lines_projection(th_img: np.ndarray) -> List[int]:
        # Ignore the far-right vote-box column (reduces false peaks from boxes)
        x0 = 0
        x1 = int(0.82 * W)

        roi = th_img[:, x0:x1]

        # Horizontal ink density per row in ROI
        proj = roi.mean(axis=1) / 255.0  # 0..1

        # Smooth
        k = max(9, int(0.012 * H) | 1)  # odd-ish window
        proj_s = cv2.blur(proj.astype(np.float32).reshape(-1, 1), (1, k)).ravel()

        # Dynamic threshold: pick “line-like” rows
        # Lines are very dense compared to text rows; using percentile helps across scans.
        thr = float(np.percentile(proj_s, 92))
        thr = max(thr, 0.12)  # keep sane floor

        ys = []
        for y in range(2, H - 2):
            v = proj_s[y]
            if v > thr and v >= proj_s[y - 1] and v >= proj_s[y + 1]:
                ys.append(y)
        return ys

    # Collect candidates from both detectors
    ys = []
    ys += _sep_lines_morph(th)
    ys += _sep_lines_projection(th)

    # Cluster to unique separator y’s
    tol = max(4, int(0.008 * H))
    sep_ys = _cluster_ys(ys, tol)
    sep_ys = sorted(sep_ys)

    # Estimate typical row height from separators (fallback to box height)
    gaps = [b - a for a, b in zip(sep_ys[:-1], sep_ys[1:]) if b > a]
    if gaps:
        med_gap = int(np.median(gaps))
    elif boxes:
        med_gap = int(2.2 * np.median([b[3] for b in boxes]))
    else:
        med_gap = int(0.06 * H)

    # Conservative header cut based only on configuration.
    # We intentionally do NOT tie this to the first detected box so that
    # ballots with few marks still get a full set of candidate rows.
    header_cut = int(config.min_y_frac * H)
    sep_ys = [y for y in sep_ys if y >= header_cut]

    if not sep_ys:
        return []

    # ---- Add synthetic top/bottom boundaries so we don’t lose first/last rows
    bounds = list(sep_ys)

    # If there might be a row ABOVE first separator (common when top box is missed),
    # add a boundary one row-height above.
    top_syn = int(max(header_cut, bounds[0] - med_gap))
    if top_syn < bounds[0] - int(0.35 * med_gap):
        bounds = [top_syn] + bounds

    # Always add a bottom boundary
    bot_syn = int(min(H - 1, bounds[-1] + med_gap))
    if bot_syn > bounds[-1] + int(0.35 * med_gap):
        bounds = bounds + [bot_syn]

    # Final clustering/dedup after synthetic additions
    bounds = _cluster_ys(bounds, tol)
    bounds = sorted(bounds)

    # Build rows between consecutive boundaries
    rows: List[Tuple[int, int]] = []
    min_h = max(10, int(0.45 * med_gap))
    for y1, y2 in zip(bounds[:-1], bounds[1:]):
        if y2 - y1 >= min_h:
            rows.append((int(y1), int(y2)))

    return rows

def normalize_rows(rows, boxes, img_shape, config):
    H, W = img_shape[:2]
    if not rows:
        return rows

    # Always use config-based cutoff as the baseline (stable across images)
    header_cut = int(config.min_y_frac * H)

    # Optional: only *tighten* cutoff if boxes are plentiful and consistent
    if boxes and len(boxes) >= 6:
        y_first = min(b[1] for b in boxes)
        h_med = int(np.median([b[3] for b in boxes]))
        box_cut = max(0, int(y_first - 1.2 * h_med))
        header_cut = max(header_cut, box_cut)

    rows = [(y1, y2) for (y1, y2) in rows if y2 > header_cut]
    if not rows:
        return rows

    heights = [y2 - y1 for (y1, y2) in rows]
    med_h = int(np.median(heights))

    normalized = []
    for (y1, y2) in rows:
        h = y2 - y1
        if h > 1.75 * med_h:
            n = max(2, int(round(h / med_h)))
            step = h / n
            for k in range(n):
                a = int(y1 + k * step)
                b = int(y1 + (k + 1) * step)
                normalized.append((a, b))
        else:
            normalized.append((y1, y2))

    return sorted(normalized, key=lambda r: r[0])



def normalize_rows_box_aware(
    rows: List[Tuple[int, int]],
    boxes: List[Tuple[int, int, int, int]],
    img_shape: Tuple[int, int, int],
    config: BallotConfig,
) -> List[Tuple[int, int]]:
    """
    Do NOT split based on height alone.
    Only split a band if it contains 2+ vote-box centers.
    """
    H, W = img_shape[:2]
    if not rows:
        return rows

    header_cut = int(config.min_y_frac * H)
    rows = [(y1, y2) for (y1, y2) in rows if y2 > header_cut]
    if not rows:
        return rows

    # Precompute box centers
    centers = []
    for (x, y, w, h) in boxes or []:
        centers.append(y + h / 2.0)

    if not centers:
        return rows

    centers = sorted(centers)
    out: List[Tuple[int, int]] = []
    for (y1, y2) in rows:
        inside = [c for c in centers if y1 <= c < y2]
        if len(inside) <= 1:
            out.append((y1, y2))
            continue

        # Split by midpoints of the centers inside this band
        inside = sorted(inside)
        bnds = [y1]
        for a, b in zip(inside[:-1], inside[1:]):
            bnds.append(int(0.5 * (a + b)))
        bnds.append(y2)

        # enforce minimum height
        min_h = int(max(10, 0.6 * np.median([b[3] for b in boxes]) if boxes else 20))
        for a, b in zip(bnds[:-1], bnds[1:]):
            if b - a >= min_h:
                out.append((int(a), int(b)))
            else:
                out.append((int(a), int(a + min_h)))

    out.sort(key=lambda r: r[0])
    return out



def assign_boxes_to_rows(
    boxes: List[Tuple[int, int, int, int]],
    rows: List[Tuple[int, int]],
    overlap_thresh: float = 0.50,
):
    assigned = [None] * len(rows)
    if not boxes or not rows:
        return assigned

    for box in boxes:
        x, y, w, h = box
        box_y1, box_y2 = y, y + h
        best_i = None
        best_overlap = 0.0
        for i, (ry1, ry2) in enumerate(rows):
            inter = max(0, min(ry2, box_y2) - max(ry1, box_y1))
            overlap = inter / float(h + 1e-6)  # fraction of box height inside row
            if overlap > best_overlap:
                best_overlap = overlap
                best_i = i

        if best_i is not None and best_overlap >= overlap_thresh:
            # Keep right-most if collision
            if assigned[best_i] is None or x > assigned[best_i][0]:
                assigned[best_i] = box

    return assigned


def repair_missing_boxes(rows: List[Tuple[int, int]], 
                         assigned: List[Optional[Tuple[int, int, int, int]]], 
                         boxes: List[Tuple[int, int, int, int]], 
                         img_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    """Fill in missing boxes using median geometry."""
    H, W = img_shape[:2]
    if not boxes:
        return []
        
    med_x = int(np.median([b[0] for b in boxes]))
    med_w = int(np.median([b[2] for b in boxes]))
    med_h = int(np.median([b[3] for b in boxes]))
    
    repaired = []
    for i, (y1, y2) in enumerate(rows):
        box = assigned[i]
        if box is None:
            cy = (y1 + y2) // 2
            y = int(cy - med_h // 2)
            # Clamp to image boundaries
            x = max(0, min(W - med_w - 1, med_x))
            y = max(0, min(H - med_h - 1, y))
            box = (x, y, med_w, med_h)
        repaired.append(box)
        
    return repaired
from typing import Dict, Any

def _rotate_90n(img: np.ndarray, n: int) -> np.ndarray:
    """Rotate image by 90 degrees * n (n can be 0..3)."""
    n = n % 4
    if n == 0:
        return img
    if n == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if n == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def find_row_bands_best_orientation(
    img_bgr: np.ndarray,
    config: BallotConfig,
    expected: Optional[int] = None,
    return_best_image: bool = False,
):
    """
    Try orientations {0,90,180,270} and choose the best row layout.

    Returns
    -------
    rows : List[(y1,y2)]
    meta : dict containing:
        rotation
        boxes
        rows
        image (optional if return_best_image=True)
    """

    H0, W0 = img_bgr.shape[:2]
    exp = expected if expected is not None else getattr(config, "expected_boxes", None)

    best_score = -1e18
    best_rows = []
    best_boxes = []
    best_img = img_bgr
    best_rot = 0

    def score_layout(rows, boxes):
        r = len(rows) if rows else 0
        b = len(boxes) if boxes else 0

        if r == 0:
            return -1e9

        score = 0.0

        if exp:
            score += 200 - 40 * abs(r - exp)
        else:
            score += 10 * r

        score += 2 * min(b, 2 * (exp if exp else 40))

        if boxes:
            xs = np.array([x for (x, y, w, h) in boxes])
            score += 5 * float(np.median(xs) / max(1.0, W0))

        return score

    for n in range(4):

        rot = 90 * n
        img_r = _rotate_90n(img_bgr, n)

        boxes = detect_vote_boxes(img_r, config)

        rows = rows_from_boxes(boxes, img_r.shape, config)

        if rows:
            cys = np.array([(y1 + y2) / 2 for (y1, y2) in rows])
            med_gap = float(np.median(np.diff(np.sort(cys)))) if len(cys) >= 2 else None
            rows = refine_rows_with_separators(img_r, rows, config, med_gap)

        if not rows:
            rows = find_row_bands(img_r, boxes, config)

        score = score_layout(rows, boxes)

        if score > best_score:
            best_score = score
            best_rows = rows
            best_boxes = boxes
            best_img = img_r
            best_rot = rot

    meta = {
        "rotation": best_rot,
        "rows": best_rows,
        "boxes": best_boxes,
    }

    if return_best_image:
        meta["image"] = best_img

    return best_rows, meta

def detect_vote_boxes_legacy(img_bgr: np.ndarray, config: BallotConfig) -> List[Tuple[int, int, int, int]]:
    """
    Legacy whole-image contour-based vote box detector (for 'perfect' ballots).

    IMPORTANT: This function intentionally preserves the legacy logic exactly
    (thresholding/morphology/contour filtering), with only minimal adaptation
    to the new repo's imports/types.

    Returns: list of (x, y, w, h) in full-image coordinates, sorted by y.
    """
    H, W = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 51)
    norm = cv2.divide(gray, bg, scale=255)
    th = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        7,
    )
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h + 1e-6)
        area = float(w * h)
        if (
            float(getattr(config, "vote_box_aspect_min", 0.65))
            <= ar
            <= float(getattr(config, "vote_box_aspect_max", 1.55))
            and area > float(getattr(config, "vote_box_min_area_frac", config.min_area_frac)) * (W * H)
            and x > config.min_x_frac * W
            and y > config.min_y_frac * H
        ):
            cand.append((int(x), int(y), int(w), int(h)))

    cand.sort(key=lambda b: b[1])
    return _filter_vote_box_column(cand, W, config)

"""Debug runner for orientation/deskew row detection.

Usage:
  python -m tools.debug_orientation_rows --image /path/to/ballot.png --out debug_orient

It will:
  - Try 0/90/180/270 rotations (and optional deskew angles) using the same
    scoring logic as the pipeline.
  - Save overlay images for each trial with the score in the filename.
  - Save the chosen best overlay as "best_overlay.png".
"""

import argparse
import os
from pathlib import Path

import cv2

from ballot_reader.config import BallotConfig
from ballot_reader.io import read_image
from ballot_reader.layout import find_row_bands_best_orientation
from ballot_reader.debug import DebugWriter


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug row detection across orientations")
    ap.add_argument("--image", required=True, help="Input ballot image path")
    ap.add_argument("--out", default="debug_orientation", help="Output directory")
    ap.add_argument("--expected", type=int, default=None, help="Expected number of candidate rows")
    ap.add_argument("--no-deskew", action="store_true", help="Disable small deskew search")
    ap.add_argument("--no-rotate", action="store_true", help="Disable multi-orientation search")
    args = ap.parse_args()

    img = read_image(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    cfg = BallotConfig()
    if args.expected is not None:
        cfg.expected_boxes = int(args.expected)
    if args.no_deskew:
        cfg.try_deskew = False
    if args.no_rotate:
        cfg.try_multiple_orientations = False

    ballot_id = Path(args.image).stem
    dw = DebugWriter(args.out, ballot_id + "_orient")

    rows, meta = find_row_bands_best_orientation(img, cfg, return_best_image=True)
    trials = meta.get("trials", [])

    # Save overlays for all trials
    for i, t in enumerate(trials):
        rot = int(t.get("rotation", 0))
        dsk = float(t.get("deskew", 0.0))
        score = float(t.get("score", 0.0))
        # We don't store the trial image to avoid memory bloat; re-run for overlay.
        # This is a debug tool only.
        # Recompute best image for this trial for visualization.
        # NOTE: this is consistent with the production rotation/deskew logic.
        from ballot_reader.layout import _rotate_90n, _rotate_bound  # type: ignore

        vis_img = _rotate_90n(img, rot)
        if abs(dsk) > 1e-6:
            vis_img, _ = _rotate_bound(vis_img, dsk)
        boxes = t.get("boxes", [])
        rows_t = t.get("rows", [])
        name = f"trial_{i:02d}_r{rot}_d{dsk:+.1f}_s{score:+.3f}"
        try:
            dw.draw_overlay(vis_img, boxes, rows_t, name)
        except Exception:
            # If overlay fails for some reason, at least dump the image.
            dw.save_image(vis_img, [], name)

    # Save chosen best overlay
    best_img = meta.get("image", img)
    best_boxes = meta.get("boxes", [])
    best_name = f"best_r{meta.get('rotation', 0)}_d{meta.get('deskew', 0.0):+.1f}_s{meta.get('score', 0.0):+.3f}"
    dw.draw_overlay(best_img, best_boxes, rows, "best_overlay")

    # Save a short text summary
    try:
        txt = [
            f"image={os.path.basename(args.image)}",
            f"best_rotation={meta.get('rotation', 0)}",
            f"best_deskew={meta.get('deskew', 0.0)}",
            f"best_score={meta.get('score', 0.0)}",
            f"rows={len(rows)} boxes={len(best_boxes)}",
            "",
            "All trials:",
        ]
        for t in trials:
            txt.append(
                f"  rot={t.get('rotation', 0)} deskew={t.get('deskew', 0.0):+} score={t.get('score', 0.0):+.4f} rows={len(t.get('rows', []) or [])} boxes={len(t.get('boxes', []) or [])}"
            )
        # DebugWriter may not have save_text in some forks; write directly.
        out_path = os.path.join(dw.out_dir, "summary.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()

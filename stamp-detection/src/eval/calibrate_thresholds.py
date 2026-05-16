
import argparse
import json
from pathlib import Path

import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def has_gt(label_file: Path) -> bool:
    return label_file.exists() and label_file.read_text().strip() != ""


def image_stems(images_dir: Path) -> set[str]:
    return {
        p.stem
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def label_stems(labels_dir: Path) -> set[str]:
    return {p.stem for p in labels_dir.glob("*.txt")}


def metrics(rows, score_valid: float, score_review: float) -> dict:
    tp = fp = tn = fn = review = 0
    for row in rows:
        pred_valid = row["geo_ok"] and row["score"] >= score_valid
        pred_review = row["geo_ok"] and score_review <= row["score"] < score_valid
        if pred_review:
            review += 1

        if row["gt"] and pred_valid:
            tp += 1
        elif row["gt"] and not pred_valid:
            fn += 1
        elif (not row["gt"]) and pred_valid:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    return {
        "score_valid": float(score_valid),
        "score_review": float(score_review),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "review": review,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_manifest", default="outputs/inference_manifest.json")
    ap.add_argument("--labels_dir", default="data/yolo_split/val/labels")
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--max_fpr", type=float, default=0.002)
    ap.add_argument("--valid_min", type=float, default=0.05)
    ap.add_argument("--valid_max", type=float, default=0.95)
    ap.add_argument("--valid_steps", type=int, default=181)
    ap.add_argument("--review_min", type=float, default=0.05)
    ap.add_argument("--review_max", type=float, default=0.5)
    ap.add_argument("--review_steps", type=int, default=10)
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    target_stems = image_stems(Path(args.images_dir)) if args.images_dir else label_stems(labels_dir)

    manifest = json.load(open(args.pred_manifest))
    rows = []
    for rec in manifest:
        stem = Path(rec["image"]).stem
        if stem not in target_stems:
            continue
        rows.append(
            {
                "score": float(rec.get("score") or 0.0),
                "geo_ok": bool(rec["geo_ok"]) if "geo_ok" in rec else rec.get("decision") in {"VALID STAMP", "REVIEW REQUIRED"},
                "gt": has_gt(labels_dir / f"{stem}.txt"),
            }
        )

    valid_thresholds = np.linspace(args.valid_min, args.valid_max, args.valid_steps)
    review_thresholds = np.linspace(args.review_min, args.review_max, args.review_steps)

    best = None
    for score_valid in valid_thresholds:
        for score_review in review_thresholds:
            if score_review > score_valid:
                continue
            result = metrics(rows, score_valid, score_review)
            if result["fpr"] > args.max_fpr:
                continue
            if best is None:
                best = result
                continue
            key = (result["recall"], result["precision"], -result["score_valid"])
            best_key = (best["recall"], best["precision"], -best["score_valid"])
            if key > best_key:
                best = result

    if best is None:
        print(f"No threshold met max_fpr={args.max_fpr}")
        return

    gt_pos = sum(1 for row in rows if row["gt"])
    gt_neg = len(rows) - gt_pos
    print("Evaluated images:", len(rows))
    print("Ground-truth positives:", gt_pos)
    print("Ground-truth negatives:", gt_neg)
    print("Recommended score_valid:", best["score_valid"])
    print("Recommended score_review:", best["score_review"])
    print("TP FP TN FN:", best["tp"], best["fp"], best["tn"], best["fn"])
    print("Precision:", best["precision"])
    print("Recall:", best["recall"])
    print("FPR:", best["fpr"])
    print("Review count:", best["review"])


if __name__ == "__main__":
    main()

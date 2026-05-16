import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

VALID = "VALID STAMP"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def has_gt(label_file: Path) -> bool:
    return label_file.exists() and label_file.read_text().strip() != ""


def image_paths(images_dir: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def label_stems(labels_dir: Path) -> set[str]:
    return {p.stem for p in labels_dir.glob("*.txt")}


def load_thresholds(path: Path | None) -> tuple[float, float]:
    if path is None:
        return 0.6, 0.35
    values = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = float(value.strip())
    return float(values["score_valid"]), float(values["score_review"])


def likely_cause(rec: dict, score_valid: float) -> str:
    bbox = rec.get("bbox")
    decision = rec.get("decision")
    score = float(rec.get("score") or 0.0)
    geo_ok = bool(rec.get("geo_ok", False))

    if not bbox:
        return "no detection / no bbox"
    if not geo_ok:
        return "failed geometry gate"
    if decision == "REVIEW REQUIRED":
        return "review decision instead of valid"
    if score < score_valid:
        return "score below threshold"
    return "decision not valid"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_manifest", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--val_rules", default="configs/val_rules.yaml")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    images_by_stem = image_paths(Path(args.images_dir)) if args.images_dir else {}
    target_stems = set(images_by_stem) if images_by_stem else label_stems(labels_dir)
    score_valid, _ = load_thresholds(Path(args.val_rules) if args.val_rules else None)

    manifest = json.load(open(args.pred_manifest))
    rows = []
    for rec in manifest:
        stem = Path(rec["image"]).stem
        if stem not in target_stems:
            continue
        label_path = labels_dir / f"{stem}.txt"
        if not has_gt(label_path) or rec.get("decision") == VALID:
            continue
        features = rec.get("features") or {}
        row = {
            "cause": likely_cause(rec, score_valid),
            "image_path": str(images_by_stem.get(stem, Path(rec["image"]))),
            "label_path": str(label_path),
            "score": float(rec.get("score") or 0.0),
            "decision": rec.get("decision"),
            "bbox_present": bool(rec.get("bbox")),
            "bbox": rec.get("bbox"),
            "geo_ok": bool(rec.get("geo_ok", False)),
            "area_frac": features.get("area_frac"),
            "circularity": features.get("circularity"),
            "ellipse_edge_pct": features.get("ellipse_edge_pct"),
            "template_ncc": features.get("template_ncc"),
            "blob_count": features.get("blob_count"),
            "blob_spread": features.get("blob_spread"),
        }
        rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cause"]].append(row)

    print("False negatives:", len(rows))
    for cause, cause_rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        print(f"\n{cause}: {len(cause_rows)}")
        display_rows = cause_rows if args.limit <= 0 else cause_rows[: args.limit]
        for row in display_rows:
            print(
                f"  {row['image_path']} score={row['score']:.4f} "
                f"decision={row['decision']} bbox={row['bbox_present']} "
                f"geo_ok={row['geo_ok']} label={row['label_path']} "
                f"features={{area_frac:{row['area_frac']}, circularity:{row['circularity']}, "
                f"ellipse_edge_pct:{row['ellipse_edge_pct']}, template_ncc:{row['template_ncc']}, "
                f"blob_count:{row['blob_count']}, blob_spread:{row['blob_spread']}}}"
            )

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["cause"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()

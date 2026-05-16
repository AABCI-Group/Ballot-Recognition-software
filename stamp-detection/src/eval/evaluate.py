
import argparse
import json
from pathlib import Path


VALID = "VALID STAMP"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_manifest", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument(
        "--images_dir",
        default=None,
        help="Optional image split directory. When supplied, only these image stems are evaluated.",
    )
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    target_stems = image_stems(Path(args.images_dir)) if args.images_dir else label_stems(labels_dir)

    data = json.load(open(args.pred_manifest))
    records = [rec for rec in data if Path(rec["image"]).stem in target_stems]

    tp = fp = tn = fn = 0
    gt_pos = gt_neg = 0
    for rec in records:
        stem = Path(rec["image"]).stem
        gt = has_gt(labels_dir / f"{stem}.txt")
        pred = rec["decision"]
        if gt:
            gt_pos += 1
        else:
            gt_neg += 1

        if gt and pred == VALID:
            tp += 1
        elif gt and pred != VALID:
            fn += 1
        elif (not gt) and pred == VALID:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)

    print("Evaluated images:", len(records))
    print("Ground-truth positives:", gt_pos)
    print("Ground-truth negatives:", gt_neg)
    print("TP FP TN FN:", tp, fp, tn, fn)
    print("Precision:", precision)
    print("Recall:", recall)
    print("FPR:", fpr)


if __name__ == "__main__":
    main()

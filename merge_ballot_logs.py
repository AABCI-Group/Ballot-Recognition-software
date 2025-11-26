import json
import os

# Adjust these if your paths are different
YOLO_LOG = "outputs/inference_manifest.json"
DIGIT_LOG = "debug_ballot/audit_log.json"

OUTPUT_JSON = "logs/ballots_merged.json"
OUTPUT_CSV = "logs/ballots_merged.csv"


def normalize_name(path_or_name: str) -> str:
    """
    Normalize image identity:
    - handle both / and \\
    - drop folders, keep only the filename
    """
    return os.path.basename(path_or_name.replace("\\", "/"))


def simplify_yolo(yolo_rec):
    """
    Strip YOLO down to: is the stamp valid or not?
    """
    label = yolo_rec.get("decision", "NO STAMP")
    score = yolo_rec.get("score", 0.0)


    return {
        "stamp_label": label,
    }


def simplify_digit(digit_rec):
    """
    Keep useful ballot info from the digit pipeline.
    """
    return {
        "sequence_ok": digit_rec.get("sequence_ok"),
        "numbers_found": digit_rec.get("numbers_found"),
        "results": digit_rec.get("results", []),
        "border_fn_used": digit_rec.get("border_fn_used"),
    }


def main():
    # --- Load YOLO log ---
    if not os.path.exists(YOLO_LOG):
        raise FileNotFoundError(f"YOLO log not found: {YOLO_LOG}")
    with open(YOLO_LOG, "r", encoding="utf-8") as f:
        yolo_data = json.load(f)

    # --- Load DIGIT log ---
    if not os.path.exists(DIGIT_LOG):
        raise FileNotFoundError(f"Digit log not found: {DIGIT_LOG}")
    with open(DIGIT_LOG, "r", encoding="utf-8") as f:
        digit_data = json.load(f)

    # Index YOLO records by normalized filename
    yolo_by_name = {}
    for rec in yolo_data:
        name = normalize_name(rec["image"])
        yolo_by_name[name] = rec

    # Index DIGIT records by normalized filename
    digit_by_name = {}
    for rec in digit_data:
        # image_path has the full path, image is just the filename
        if "image_path" in rec:
            name = normalize_name(rec["image_path"])
        else:
            name = normalize_name(rec["image"])
        digit_by_name[name] = rec

    # Union of all filenames seen in either log
    all_names = sorted(set(yolo_by_name.keys()) | set(digit_by_name.keys()))

    merged = []
    for name in all_names:
        y_raw = yolo_by_name.get(name)
        d_raw = digit_by_name.get(name)

        ballot = {
            "image": name,
        }

        # Stamp info (YOLO)
        if y_raw is not None:
            ballot["stamp"] = simplify_yolo(y_raw)
        else:
            ballot["stamp"] = {
                "stamp_label": "NO YOLO DATA",
            }

        # Digit info
        if d_raw is not None:
            ballot["digits"] = simplify_digit(d_raw)
        else:
            ballot["digits"] = None

        merged.append(ballot)

    # Ensure logs/ directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # Write JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Also write a simple CSV summary
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write("image,stamp_label,sequence_ok,numbers_found\n")
        for b in merged:
            stamp = b["stamp"]
            digits = b["digits"] or {}
            f.write(
                f"{b['image']},"
                f"\"{stamp['stamp_label']}\","
                f"{digits.get('sequence_ok')},"
                f"{digits.get('numbers_found')}\n"
            )

    print(f"Wrote {len(merged)} merged ballots to {OUTPUT_JSON}")
    print(f"CSV summary: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

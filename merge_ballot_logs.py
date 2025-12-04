import json
import os
import csv
import random
from datetime import datetime

import requests  # pip install requests

# ---------- Supabase config ----------
SUPABASE_URL = "https://wcuzjrawfvhocbaibfbi.supabase.co"  # your URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndjdXpqcmF3ZnZob2NiYWliZmJpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjMxNzIwMSwiZXhwIjoyMDc3ODkzMjAxfQ.DKofOwqLe3VLDA0EER36YX_f04Xtqj7jygn3BYiFFg8"  # put your key in an env var

if not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_KEY environment variable with your Supabase anon/service key")

# Adjust these if your paths are different
YOLO_LOG = "outputs/inference_manifest.json"
DIGIT_LOG = "debug_ballot/audit_log.json"
OUTPUT_JSON = "logs/ballots_merged.json"
OUTPUT_CSV = "logs/ballots_merged.csv"

# Row → candidate name mapping for this ballot paper
ROW_TO_CANDIDATE = {
    1: "Adams",
    2: "Crawford",
    3: "Daniels",
    4: "Dewey",
    5: "Grolish",
    6: "Guinness",
    7: "Hennessey",
    8: "Middleton",
    9: "Power",
    10: "Tia",
}


def normalize_name(path_or_name: str) -> str:
    """Normalize image identity: drop folders, keep only the filename."""
    return os.path.basename(path_or_name.replace("\\", "/"))


def simplify_yolo(yolo_rec):
    """Strip YOLO down to: is the stamp valid or not?"""
    label = yolo_rec.get("decision", "NO STAMP")
    score = yolo_rec.get("score", 0.0)
    return {
        "stamp_label": label,
        "score": score,
    }


def simplify_digit(digit_rec):
    """Keep useful ballot info from the digit pipeline."""
    return {
        "sequence_ok": digit_rec.get("sequence_ok"),
        "numbers_found": digit_rec.get("numbers_found"),
        "results": digit_rec.get("results", []),
        "border_fn_used": digit_rec.get("border_fn_used"),
    }


# ---------- Supabase helpers ----------

def get_existing_ballot_ids() -> set[int]:
    """
    Fetch existing ballot_id values from Supabase and return as a set.
    """
    url = f"{SUPABASE_URL}/rest/v1/Ballots"
    params = {"select": "ballot_id"}
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    existing = {row["ballot_id"] for row in data if "ballot_id" in row}
    print(f"Fetched {len(existing)} existing ballot_ids from Supabase")
    return existing


def generate_unique_ballot_id(existing_ids: set[int]) -> int:
    """
    Generate a random ballot_id between 1 and 200,000 that is not in existing_ids.
    Also inserts the new id into existing_ids so it won't be reused.
    """
    if len(existing_ids) >= 200_000:
        raise RuntimeError("Ballot ID space exhausted (1–200000).")

    while True:
        new_id = random.randint(1, 200_000)
        if new_id not in existing_ids:
            existing_ids.add(new_id)
            return new_id


def insert_rows_into_supabase(rows: list[dict]):
    """
    Insert a list of row dicts into the ballots table in Supabase.
    Uses REST API directly.
    """
    if not rows:
        print("No rows to insert into Supabase.")
        return

    url = f"{SUPABASE_URL}/rest/v1/Ballots"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        # Uncomment the next line to turn it into an UPSERT on ballot_id:
        # "Prefer": "resolution=merge-duplicates"
    }

    resp = requests.post(url, json=rows, headers=headers)
    try:
        resp.raise_for_status()
    except Exception as e:
        print("Error inserting into Supabase:", resp.text)
        raise

    print(f"Inserted {len(rows)} rows into Supabase.")


# ---------- Main merging / CSV + DB logic ----------

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

    # --- Fetch existing ballot IDs from Supabase ---
    existing_ids = get_existing_ballot_ids()

    # Index YOLO records by normalized filename
    yolo_by_name = {normalize_name(rec["image"]): rec for rec in yolo_data}

    # Index DIGIT records by normalized filename
    digit_by_name = {}
    for rec in digit_data:
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

        ballot = {"image": name}

        # Stamp info (YOLO)
        if y_raw is not None:
            ballot["stamp"] = simplify_yolo(y_raw)
        else:
            ballot["stamp"] = {"stamp_label": "NO YOLO DATA", "score": 0.0}

        # Digit info
        ballot["digits"] = simplify_digit(d_raw) if d_raw is not None else None

        merged.append(ballot)

    # Ensure logs/ directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # Write merged JSON (optional)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # --- Build rows for CSV + Supabase insert ---
    rows_for_db: list[dict] = []

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "created_at",
                "vote_preference",
                "verification_type",
                "candidate_name",
                "ballot_id",
                "box_location",
                "image_url",
            ]
        )

        for b in merged:
            image_name = b["image"]
            stamp_label = b["stamp"]["stamp_label"]
            digits = b["digits"] or {}

            sequence_ok = digits.get("sequence_ok", False)

            # Decide verification type
            if stamp_label == "VALID STAMP" and sequence_ok:
                verification_type = "Valid"
            else:
                verification_type = "Doubtful"

            # Generate a unique random ballot_id for THIS ballot
            ballot_id = generate_unique_ballot_id(existing_ids)

            created_at = datetime.utcnow().isoformat() + "+00:00"
            box_location = ""       # fill in if/when you know it
            image_url = image_name  # or a full URL if you have one

            # One output row per NON-NULL digit
            for res in digits.get("results", []):
                digit = res.get("digit")
                if digit in (None, "", "NULL"):
                    continue  # skip NULL digits

                row_num = res.get("row")
                candidate_name = ROW_TO_CANDIDATE.get(row_num, f"ROW_{row_num}")

                # CSV
                writer.writerow(
                    [
                        created_at,
                        digit,             # vote_preference
                        verification_type,
                        candidate_name,
                        ballot_id,
                        box_location,
                        image_url,
                    ]
                )

                # For DB insert
                rows_for_db.append(
                    {
                        "created_at": created_at,
                        "vote_preference": int(digit),
                        "verification_type": verification_type,
                        "candidate_name": candidate_name,
                        "ballot_id": ballot_id,
                        "box_location": box_location,
                        "image_url": image_url,
                    }
                )

    print(f"Wrote {len(merged)} merged ballots to {OUTPUT_JSON}")
    print(f"CSV summary: {OUTPUT_CSV}")

    # --- Insert into Supabase ---
    insert_rows_into_supabase(rows_for_db)


if __name__ == "__main__":
    main()

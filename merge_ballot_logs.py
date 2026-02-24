import json
import os
import csv
from datetime import datetime

import requests  # pip install requests

# ---------- Supabase config ----------
SUPABASE_URL = "https://wcuzjrawfvhocbaibfbi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndjdXpqcmF3ZnZob2NiYWliZmJpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjMxNzIwMSwiZXhwIjoyMDc3ODkzMjAxfQ.DKofOwqLe3VLDA0EER36YX_f04Xtqj7jygn3BYiFFg8"  # <-- move to env var ideally

if not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_KEY environment variable with your Supabase anon/service key")

# ---------- Constants ----------
ELECTION_ID = 1

# Normalised table names (as per your screenshots)
TBL_CANDIDATE = "CandidateTBL"
TBL_BALLOT_PAPER = "BallotPaperTBL"
TBL_BALLOT_PREF = "BallotPreferenceTBL"

# Adjust these if your paths are different
YOLO_LOG = "outputs/inference_manifest.json"
DIGIT_LOG = "debug_ballot/audit_log.json"
OUTPUT_JSON = "logs/ballots_merged.json"
OUTPUT_CSV = "logs/ballots_merged.csv"


def normalize_name(path_or_name: str) -> str:
    """Normalize image identity: drop folders, keep only the filename."""
    return os.path.basename(path_or_name.replace("\\", "/"))


def simplify_yolo(yolo_rec):
    """Strip YOLO down to: is the stamp valid or not?"""
    label = yolo_rec.get("decision", "NO STAMP")
    score = yolo_rec.get("score", 0.0)
    return {"stamp_label": label, "score": score}



def simplify_digit(digit_rec):
    """Keep useful ballot info from the digit pipeline."""
    return {
        "sequence_ok": digit_rec.get("sequence_ok"),
        "numbers_found": digit_rec.get("numbers_found"),
        "results": digit_rec.get("results", []),
        "border_fn_used": digit_rec.get("border_fn_used"),
    }


# ---------- Supabase helpers ----------

def _headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def fetch_candidates_for_election(election_id: int) -> list[dict]:
    """
    Fetch candidates for an election, sorted by candidate id asc.
    Row mapping is built from this sorted list.
    """
    url = f"{SUPABASE_URL}/rest/v1/{TBL_CANDIDATE}"
    params = {
        "select": "*",
        "election_id": f"eq.{election_id}",
        "order": "id.asc",
    }
    resp = requests.get(url, params=params, headers=_headers())
    resp.raise_for_status()
    candidates = resp.json()
    if not candidates:
        raise RuntimeError(f"No candidates found for election_id={election_id} in {TBL_CANDIDATE}")

    return candidates


def _candidate_display_name(candidate_row: dict) -> str:
    """
    Tries common name fields. Adjust if your CandidateTBL uses a specific column.
    """
    for k in ("name", "candidate_name", "full_name", "display_name"):
        if k in candidate_row and candidate_row[k]:
            return str(candidate_row[k])
    # fallback:
    return f"Candidate_{candidate_row.get('id')}"


def build_row_to_candidate_map(candidates_sorted: list[dict]) -> dict[int, dict]:
    """
    Row 1 -> candidates_sorted[0]
    Row 2 -> candidates_sorted[1]
    ...
    """
    return {i + 1: c for i, c in enumerate(candidates_sorted)}


def get_next_random_ballot_id(election_id: int) -> int:
    """
    In your BallotPaperTBL screenshot, random_ballot_id looks like 1000001, 1000002, ...
    This safely generates the next id by taking max(random_ballot_id) + 1 for the election.

    If there are no rows yet, it starts at 1000001.
    """
    url = f"{SUPABASE_URL}/rest/v1/{TBL_BALLOT_PAPER}"
    params = {
        "select": "random_ballot_id",
        "election_id": f"eq.{election_id}",
        "order": "random_ballot_id.desc",
        "limit": "1",
    }
    resp = requests.get(url, params=params, headers=_headers())
    resp.raise_for_status()
    rows = resp.json()

    if not rows:
        return 1_000_001

    current_max = rows[0].get("random_ballot_id")
    if current_max is None:
        return 1_000_001
    return int(current_max) + 1


def insert_ballot_papers(rows: list[dict]):
    """Insert rows into BallotPaperTBL."""
    if not rows:
        return

    url = f"{SUPABASE_URL}/rest/v1/{TBL_BALLOT_PAPER}"
    resp = requests.post(url, json=rows, headers=_headers())
    try:
        resp.raise_for_status()
    except Exception:
        print("Error inserting into BallotPaperTBL:", resp.text)
        raise


def insert_ballot_preferences(rows: list[dict]):
    """Insert rows into BallotPreferenceTBL."""
    if not rows:
        return

    url = f"{SUPABASE_URL}/rest/v1/{TBL_BALLOT_PREF}"
    resp = requests.post(url, json=rows, headers=_headers())
    try:
        resp.raise_for_status()
    except Exception:
        print("Error inserting into BallotPreferenceTBL:", resp.text)
        raise


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

    # --- Fetch candidates for election (NEW LOGIC) ---
    candidates_sorted = fetch_candidates_for_election(ELECTION_ID)
    row_to_candidate = build_row_to_candidate_map(candidates_sorted)

    # Index YOLO records by normalized filename
    yolo_by_name = {normalize_name(rec["image"]): rec for rec in yolo_data}

    # Index DIGIT records by normalized filename
    digit_by_name = {}
    for rec in digit_data:
        name = normalize_name(rec.get("image_path") or rec.get("image"))
        digit_by_name[name] = rec

    # Union of all filenames seen in either log
    all_names = sorted(set(yolo_by_name.keys()) | set(digit_by_name.keys()))
    merged = []

    for name in all_names:
        y_raw = yolo_by_name.get(name)
        d_raw = digit_by_name.get(name)

        ballot = {"image": name}
        ballot["stamp"] = simplify_yolo(y_raw) if y_raw is not None else {"stamp_label": "NO YOLO DATA", "score": 0.0}
        ballot["digits"] = simplify_digit(d_raw) if d_raw is not None else None
        merged.append(ballot)

    # Ensure logs/ directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # Write merged JSON (optional)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # --- Build rows for CSV + Supabase insert (NEW NORMALISED INSERTS) ---
    ballot_paper_rows: list[dict] = []
    ballot_pref_rows: list[dict] = []

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "created_at",
                "election_id",
                "random_ballot_id",
                "ballot_state",
                "box_location",
                "image_url",
                "row_num",
                "candidate_id",
                "candidate_name",
                "preference",
            ]
        )

        next_ballot_id = get_next_random_ballot_id(ELECTION_ID)

        for b in merged:
            image_name = b["image"]
            stamp_label = b["stamp"]["stamp_label"]
            digits = b["digits"] or {}

            sequence_ok = bool(digits.get("sequence_ok", False))

            # Map into BallotPaperTBL.ballot_state (enum)
            if stamp_label == "VALID STAMP" and sequence_ok:
                ballot_state = "Valid"
            else:
                ballot_state = "Doubtful"

            created_at = datetime.utcnow().isoformat() + "+00:00"
            box_location = ""        # fill when known
            image_url = image_name   # or full URL if you have one
            random_ballot_id = next_ballot_id
            next_ballot_id += 1

            # Insert ONE BallotPaper row per image/ballot
            ballot_paper_rows.append(
                {
                    "box_location": box_location,
                    "image_url": image_url,
                    "random_ballot_id": random_ballot_id,
                    "ballot_state": ballot_state,
                    "election_id": ELECTION_ID,
                }
            )

            # Insert MANY BallotPreference rows per detected preference
            for res in digits.get("results", []):
                pref = res.get("digit")
                if pref in (None, "", "NULL"):
                    continue

                row_num = res.get("row")
                if not isinstance(row_num, int):
                    # skip malformed row info
                    continue

                cand = row_to_candidate.get(row_num)
                if cand is None:
                    # Digit pipeline returned a row outside candidate list (skip)
                    continue

                candidate_id = int(cand["id"])
                candidate_name = _candidate_display_name(cand)

                ballot_pref_rows.append(
                    {
                        "preference": int(pref),
                        "random_ballot_id": int(random_ballot_id),
                        "candidate_id": candidate_id,
                    }
                )

                # CSV output
                writer.writerow(
                    [
                        created_at,
                        ELECTION_ID,
                        random_ballot_id,
                        ballot_state,
                        box_location,
                        image_url,
                        row_num,
                        candidate_id,
                        candidate_name,
                        int(pref),
                    ]
                )

    print(f"Wrote {len(merged)} merged ballots to {OUTPUT_JSON}")
    print(f"CSV summary: {OUTPUT_CSV}")
    print(f"Prepared {len(ballot_paper_rows)} BallotPaper rows and {len(ballot_pref_rows)} BallotPreference rows.")

    # --- Insert into Supabase (Normalised) ---
    insert_ballot_papers(ballot_paper_rows)
    insert_ballot_preferences(ballot_pref_rows)

    print("Inserted into BallotPaperTBL and BallotPreferenceTBL successfully.")


if __name__ == "__main__":
    main()
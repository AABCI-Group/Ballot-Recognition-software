# merge_ballot_logs.py
import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional
import glob
import requests  # pip install requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random
from math import inf



def fetch_all_random_ballot_ids(election_id: int, page_size: int = 1000) -> set[int]:
    """
    Pulls ALL random_ballot_id values for an election into memory.
    Uses limit/offset pagination.
    """
    url = f"{SUPABASE_URL}/rest/v1/{TBL_BALLOT_PAPER}"
    existing: set[int] = set()
    offset = 0

    while True:
        params = {
            "select": "random_ballot_id",
            "election_id": f"eq.{election_id}",
            "random_ballot_id": "not.is.null",
            "limit": str(page_size),
            "offset": str(offset),
            "order": "random_ballot_id.asc",
        }
        resp = _session_get(url, params=params, headers=_headers())
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        for r in rows:
            v = r.get("random_ballot_id")
            if v is None:
                continue
            existing.add(int(v))

        if len(rows) < page_size:
            break

        offset += page_size

    return existing

def random_ballot_id_exists(election_id: int, ballot_id: int) -> bool:
    """
    Cheap existence check used by Lambda runtime to avoid full-table scans.
    """
    url = f"{SUPABASE_URL}/rest/v1/{TBL_BALLOT_PAPER}"
    params = {
        "select": "random_ballot_id",
        "election_id": f"eq.{election_id}",
        "random_ballot_id": f"eq.{ballot_id}",
        "limit": "1",
    }
    resp = _session_get(url, params=params, headers=_headers())
    resp.raise_for_status()
    return bool(resp.json())


def allocate_random_ballot_id(
    election_id: int,
    used_this_run: set[int],
    min_id: int = 1_000_001,
    max_id: int = 9_999_999,
    max_attempts: int = 40,
) -> int:
    """
    Allocate an ID without preloading the entire table.
    """
    if max_id < min_id:
        raise ValueError("max_id must be >= min_id")

    if not supabase_enabled():
        while True:
            candidate = random.randint(min_id, max_id)
            if candidate not in used_this_run:
                used_this_run.add(candidate)
                return candidate

    for _ in range(max_attempts):
        candidate = random.randint(min_id, max_id)
        if candidate in used_this_run:
            continue
        if not random_ballot_id_exists(election_id, candidate):
            used_this_run.add(candidate)
            return candidate

    # Fallback to monotonic allocation from current max.
    candidate = get_next_random_ballot_id(election_id)
    while candidate in used_this_run:
        candidate += 1
    used_this_run.add(candidate)
    return candidate

def get_random_unused_ballot_id(
    election_id: int,
    min_id: int = 1_000_001,
    max_id: int = 9_999_999,
    max_tries: int = 200_000,
) -> int:
    """
    Loads all existing IDs, then samples randomly until it finds an unused one.
    """
    if max_id < min_id:
        raise ValueError("max_id must be >= min_id")

    existing = fetch_all_random_ballot_ids(election_id)
    space = (max_id - min_id) + 1
    available = space - len(existing)

    if available <= 0:
        raise RuntimeError(
            f"No available random_ballot_id values in range [{min_id}, {max_id}] "
            f"for election_id={election_id}. Existing={len(existing)}"
        )

    for _ in range(max_tries):
        candidate = random.randint(min_id, max_id)
        if candidate not in existing:
            return candidate

    raise RuntimeError(
        f"Failed to find an unused random_ballot_id after {max_tries} tries. "
        f"Range may be too dense; widen the range."
    )
# ---------- Supabase config ----------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://wcuzjrawfvhocbaibfbi.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


# ---------- Constants ----------
ELECTION_ID = int(os.getenv("ELECTION_ID", "1"))

# Normalised table names
TBL_CANDIDATE = "CandidateTBL"
TBL_BALLOT_PAPER = "BallotPaperTBL"
TBL_BALLOT_PREF = "BallotPreferenceTBL"

# Paths
YOLO_LOG = os.getenv("YOLO_LOG", "stamp-detection/outputs/inference_manifest.json")

OUTPUT_JSON = os.getenv("OUTPUT_JSON", "logs/ballots_merged.json")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "logs/ballots_merged.csv")
DIGIT_OUT_DIR = os.getenv("DIGIT_OUT_DIR", "debug_ballot")
MERGE_IMAGE_URL = os.getenv("MERGE_IMAGE_URL", "").strip()
REQUEST_TIMEOUT = float(os.getenv("SUPABASE_HTTP_TIMEOUT_SEC", "30"))

_retry = Retry(total=5, connect=5, read=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "POST", "PATCH"), raise_on_status=False)
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
SESSION.mount("http://", HTTPAdapter(max_retries=_retry))

def _session_get(url: str, **kwargs):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    return SESSION.get(url, **kwargs)

def _session_post(url: str, **kwargs):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    return SESSION.post(url, **kwargs)



def load_digit_runs(out_dir: str) -> List[dict]:
    """
    Loads digit results from per-ballot folders:
      debug_ballot/ballot_<ballot_id>/results.json

    Produces records shaped like:
      {"image": "<ballot_id>.png", "results": [...], "sequence_ok": bool, "numbers_found": int}
    """
    records: List[dict] = []

    pattern = os.path.join(out_dir, "ballot_*", "results.json")
    paths = glob.glob(pattern)

    if not paths:
        print(f"[WARN] No digit results found under: {pattern}")
        return records

    for p in paths:
        ballot_dir = os.path.basename(os.path.dirname(p))   # ballot_<id>
        ballot_id = ballot_dir.replace("ballot_", "")       # <id>
        with open(p, "r", encoding="utf-8") as f:
            summary = json.load(f)  # [{"row":1,"digit":"NULL"},...]

        # Convert to your merge shape
        results = []
        numbers_found = 0
        prefs = []

        for r in summary:
            row = r.get("row")
            digit = r.get("digit")
            results.append({"row": row, "digit": digit})
            if isinstance(digit, int):
                numbers_found += 1
                prefs.append(digit)

        # sequence_ok: preferences should be 1..k with no duplicates
        # (simple check; you can make stricter if you want)
        uniq = sorted(set([d for d in prefs if isinstance(d, int)]))
        sequence_ok = (uniq == list(range(1, len(uniq) + 1))) if uniq else False

        records.append({
            "image": ballot_id,            # note: no extension; we stem-match later
            "sequence_ok": sequence_ok,
            "numbers_found": numbers_found,
            "results": results,
            "border_fn_used": None
        })

    return records
# ---------- Helpers ----------
def normalize_name(path_or_name: str) -> str:
    """Normalize image identity: drop folders, keep only the filename."""
    return os.path.basename(str(path_or_name).replace("\\", "/"))


def stem_name(path_or_name: str) -> str:
    """Filename without extension."""
    return os.path.splitext(normalize_name(path_or_name))[0]


def simplify_yolo(yolo_rec: dict) -> dict:
    """Strip YOLO down to: is the stamp valid or not?"""
    label = (yolo_rec or {}).get("decision", "NO STAMP")
    score = (yolo_rec or {}).get("score", 0.0)
    return {"stamp_label": label, "score": score}


def simplify_digit(digit_rec: dict) -> dict:
    """Keep useful ballot info from the digit pipeline."""
    return {
        "sequence_ok": digit_rec.get("sequence_ok"),
        "numbers_found": digit_rec.get("numbers_found"),
        "results": digit_rec.get("results", []),
        "border_fn_used": digit_rec.get("border_fn_used"),
    }


# ---------- Supabase helpers ----------
def _require_supabase_key() -> None:
    if not SUPABASE_KEY:
        raise RuntimeError("Set SUPABASE_KEY environment variable with your Supabase service key")

def _headers(return_representation: bool = False) -> dict:
    _require_supabase_key()
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if return_representation:
        # tells PostgREST to return inserted rows (useful for debugging)
        h["Prefer"] = "return=representation"
    return h


def supabase_enabled() -> bool:
    return bool(SUPABASE_KEY)



def fetch_candidates_for_election(election_id: int) -> List[dict]:
    """Fetch candidates for an election, sorted by candidate id asc."""
    url = f"{SUPABASE_URL}/rest/v1/{TBL_CANDIDATE}"
    params = {
        "select": "*",
        "election_id": f"eq.{election_id}",
        "order": "id.asc",
    }
    resp = _session_get(url, params=params, headers=_headers())
    resp.raise_for_status()
    candidates = resp.json()
    if not candidates:
        raise RuntimeError(f"No candidates found for election_id={election_id} in {TBL_CANDIDATE}")
    return candidates


def _candidate_display_name(candidate_row: dict) -> str:
    """Tries common name fields."""
    for k in ("name", "candidate_name", "full_name", "display_name"):
        if k in candidate_row and candidate_row[k]:
            return str(candidate_row[k])
    return f"Candidate_{candidate_row.get('id')}"


def build_row_to_candidate_map(candidates_sorted: List[dict]) -> Dict[int, dict]:
    """Row 1 -> candidates_sorted[0], etc."""
    return {i + 1: c for i, c in enumerate(candidates_sorted)}


def get_next_random_ballot_id(election_id: int) -> int:
    """Generate next random_ballot_id by max + 1, fallback to 1000001."""
    url = f"{SUPABASE_URL}/rest/v1/{TBL_BALLOT_PAPER}"
    params = {
        "select": "random_ballot_id",
        "election_id": f"eq.{election_id}",
        "order": "random_ballot_id.desc",
        "limit": "1",
    }
    resp = _session_get(url, params=params, headers=_headers())
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return 1_000_001
    current_max = rows[0].get("random_ballot_id")
    if current_max is None:
        return 1_000_001
    return int(current_max) + 1


def insert_ballot_papers(rows: List[dict]) -> None:
    """Insert rows into BallotPaperTBL."""
    if not rows:
        print("No BallotPaper rows to insert.")
        return

    url = f"{SUPABASE_URL}/rest/v1/{TBL_BALLOT_PAPER}"
    resp = _session_post(url, json=rows, headers=_headers(return_representation=True))

    if resp.status_code >= 400:
        print("Error inserting into BallotPaperTBL:", resp.status_code, resp.text)
        resp.raise_for_status()

    try:
        inserted = resp.json()
        print(f"Inserted BallotPaper rows: {len(inserted)}")
    except Exception:
        print("Inserted BallotPaper rows (no JSON body). Status:", resp.status_code)


def insert_ballot_preferences(rows: List[dict]) -> None:
    """Insert rows into BallotPreferenceTBL."""
    if not rows:
        print("No BallotPreference rows to insert.")
        return

    url = f"{SUPABASE_URL}/rest/v1/{TBL_BALLOT_PREF}"
    resp = _session_post(url, json=rows, headers=_headers(return_representation=True))

    if resp.status_code >= 400:
        print("Error inserting into BallotPreferenceTBL:", resp.status_code, resp.text)
        resp.raise_for_status()

    try:
        inserted = resp.json()
        print(f"Inserted BallotPreference rows: {len(inserted)}")
    except Exception:
        print("Inserted BallotPreference rows (no JSON body). Status:", resp.status_code)


def read_current_box() -> str:
    env_box = os.getenv("BOX_LOCATION", "").strip()
    if env_box:
        return env_box
    try:
        with open("current_box.json", "r", encoding="utf-8") as f:
            return (json.load(f) or {}).get("box_location", "")
    except Exception:
        return ""

# ---------- Main ----------
def main():
    # --- Load YOLO log ---
    if not os.path.exists(YOLO_LOG):
        raise FileNotFoundError(f"YOLO log not found: {YOLO_LOG}")
    with open(YOLO_LOG, "r", encoding="utf-8") as f:
        yolo_data = json.load(f)

    # --- Load DIGIT log ---
    digit_data = load_digit_runs(DIGIT_OUT_DIR)

    # --- Fetch candidates for election (optional in offline/local mode) ---
    row_to_candidate: Dict[int, dict] = {}
    if supabase_enabled():
        candidates_sorted = fetch_candidates_for_election(ELECTION_ID)
        row_to_candidate = build_row_to_candidate_map(candidates_sorted)
    else:
        print("[WARN] SUPABASE_KEY missing; running in offline merge mode (no candidate lookup, no inserts).")

    # --- Index YOLO records by normalized filename ---
    yolo_by_name: Dict[str, dict] = {}
    for rec in yolo_data:
        img = rec.get("image")
        if not img:
            continue
        yolo_by_name[normalize_name(img)] = rec

    # --- Index DIGIT records by filename AND stem fallback ---
    digit_by_name: Dict[str, dict] = {}
    digit_by_stem: Dict[str, dict] = {}

    for rec in digit_data:
        raw = rec.get("image_path") or rec.get("image") or rec.get("ballot_id")
        if not raw:
            continue
        fname = normalize_name(raw)       # e.g. "Valid.png" OR "Valid"
        st = os.path.splitext(fname)[0]   # e.g. "Valid"
        digit_by_name[fname] = rec
        digit_by_stem[st] = rec

    # ---- Debug key samples (safe now) ----
    print("YOLO keys sample:", list(yolo_by_name.keys())[:10])
    print("DIGIT keys sample:", list(digit_by_name.keys())[:10])
    print("DIGIT stems sample:", list(digit_by_stem.keys())[:10])

    # Union of all filenames seen in either log
    all_names = sorted(yolo_by_name.keys())
    merged: List[dict] = []

    # --- Merge ---
    for name in all_names:
        y_raw = yolo_by_name.get(name)

        d_raw = digit_by_name.get(name)
        if d_raw is None:
            d_raw = digit_by_stem.get(os.path.splitext(name)[0])

        ballot = {"image": name}
        ballot["stamp"] = simplify_yolo(y_raw) if y_raw is not None else {"stamp_label": "NO YOLO DATA", "score": 0.0}
        ballot["digits"] = simplify_digit(d_raw) if d_raw is not None else None
        merged.append(ballot)

    # Ensure logs/ directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # Write merged JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # --- Build rows for CSV + Supabase insert ---
    ballot_paper_rows: List[dict] = []
    ballot_pref_rows: List[dict] = []

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

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
        used_this_run: set[int] = set()

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
            box_location = read_current_box()
            # Persist only the image filename, even if MERGE_IMAGE_URL is a full S3 URI.
            image_url = normalize_name(MERGE_IMAGE_URL) if MERGE_IMAGE_URL else normalize_name(image_name)
            random_ballot_id = allocate_random_ballot_id(ELECTION_ID, used_this_run)

            # Insert ONE BallotPaper row per ballot
            ballot_paper_rows.append(
                {
                    "box_location": box_location,
                    "image_url": image_url,
                    "random_ballot_id": random_ballot_id,
                    "ballot_state": ballot_state,
                    "election_id": ELECTION_ID,
                }
            )

            # Diagnostics: show if digits missing
            if b["digits"] is None:
                print(f"[WARN] No DIGIT match for {image_name} -> preferences will be 0")
            else:
                print(
                    f"[OK] DIGITS found for {image_name}: "
                    f"sequence_ok={digits.get('sequence_ok')} results={len(digits.get('results', []))}"
                )

            # Insert MANY BallotPreference rows per detected preference
            for res in digits.get("results", []):
                pref = res.get("digit")
                if pref in (None, "", "NULL"):
                    continue

                row_num = res.get("row")
                if not isinstance(row_num, int):
                    continue

                cand = row_to_candidate.get(row_num)
                if cand is None:
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
    if ballot_pref_rows:
        print("First 5 BallotPreference rows:", ballot_pref_rows[:5])

    # --- Insert into Supabase (optional) ---
    if supabase_enabled():
        insert_ballot_papers(ballot_paper_rows)
        insert_ballot_preferences(ballot_pref_rows)
    else:
        print("[INFO] Skipped Supabase inserts (SUPABASE_KEY not set).")

    print("Done.")


if __name__ == "__main__":
    main()

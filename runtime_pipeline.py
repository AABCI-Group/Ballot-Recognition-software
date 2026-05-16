import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional

from image_parity import local_file_diagnostics, sha256_file


REPO_ROOT = Path(__file__).resolve().parent
STAMP_ROOT = REPO_ROOT / "stamp-detection"
HWR_ROOT = REPO_ROOT / "Handwritten-Digit-Recognition"
DEFAULT_STAMP_WEIGHTS = STAMP_ROOT / "runs" / "train" / "yolo_stamp" / "weights" / "best.pt"
DEFAULT_DIGIT_MODEL = HWR_ROOT / "tf-cnn-model.keras"


def _run_checked(cmd: list[str], cwd: Path, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(cmd, cwd=str(cwd), env=merged_env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def fetch_expected_rows() -> Optional[int]:
    """
    Pulls expected row count from Supabase via merge_ballot_logs helpers.
    Returns None when unavailable so ballot reader can run without --expected.
    """
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import merge_ballot_logs as mbl

        candidates = mbl.fetch_candidates_for_election(mbl.ELECTION_ID)
        return len(candidates)
    except Exception as exc:
        print(f"[WARN] Could not fetch candidate count from Supabase: {exc}")
        return None


def _load_json(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _runtime_diagnostics(image: Path, weights: Path, model: Path) -> dict:
    return {
        "input_image": local_file_diagnostics(image),
        "stamp_weights": {
            "path": str(weights),
            "bytes": weights.stat().st_size,
            "sha256": sha256_file(weights),
        },
        "digit_model": {
            "path": str(model),
            "bytes": model.stat().st_size,
            "sha256": sha256_file(model),
        },
        "runtime_environment": {
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "package_versions": {
            "python": sys.version.split()[0],
            "opencv_python_headless": _safe_version("opencv-python-headless"),
            "opencv_python": _safe_version("opencv-python"),
            "ultralytics": _safe_version("ultralytics"),
            "torch": _safe_version("torch"),
            "torchvision": _safe_version("torchvision"),
            "tensorflow_cpu": _safe_version("tensorflow-cpu"),
            "tensorflow": _safe_version("tensorflow"),
            "numpy": _safe_version("numpy"),
        },
    }


def _build_debug_summary(yolo_log: Path, merged_json: Path, image_url: Optional[str]) -> dict:
    yolo_data = _load_json(yolo_log) or []
    merged_data = _load_json(merged_json) or []

    yolo_rec = yolo_data[0] if yolo_data else {}
    merged_rec = merged_data[0] if merged_data else {}
    stamp = merged_rec.get("stamp") or {}
    digits = merged_rec.get("digits") or {}

    preferences = []
    for result in digits.get("results", []):
        digit = result.get("digit")
        row = result.get("row")
        if isinstance(digit, int) and isinstance(row, int):
            preferences.append({"row": row, "digit": digit})

    stamp_label = stamp.get("stamp_label") or yolo_rec.get("decision")
    sequence_ok = bool(digits.get("sequence_ok", False))
    ballot_state = "Valid" if stamp_label == "VALID STAMP" and sequence_ok else "Doubtful"

    return {
        "image_name": Path(str(merged_rec.get("image") or yolo_rec.get("image") or "")).name,
        "image_url": Path(image_url).name if image_url else None,
        "stamp_decision": {
            "label": stamp_label,
            "score": stamp.get("score", yolo_rec.get("score")),
            "bbox": yolo_rec.get("bbox"),
            "features": yolo_rec.get("features"),
        },
        "digit_summary": {
            "sequence_ok": digits.get("sequence_ok"),
            "numbers_found": digits.get("numbers_found"),
            "preferences": preferences,
        },
        "derived_ballot_state": ballot_state,
    }


def process_single_ballot(
    image_path: str,
    *,
    stamp_weights: Optional[str] = None,
    digit_model: Optional[str] = None,
    work_root: Optional[str] = None,
    yolo_device: Optional[str] = None,
    image_url: Optional[str] = None,
) -> dict:
    """
    Run stamp detection + handwritten extraction + merge on one image.
    Outputs are written under work_root so Lambda can safely use /tmp.
    """
    image = Path(image_path).resolve()
    if not image.exists():
        raise FileNotFoundError(f"Input image not found: {image}")

    weights = Path(stamp_weights or DEFAULT_STAMP_WEIGHTS).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Stamp weights not found: {weights}")

    model = Path(digit_model or DEFAULT_DIGIT_MODEL).resolve()
    if not model.exists():
        raise FileNotFoundError(f"Digit model not found: {model}")

    work_dir = Path(work_root or os.getenv("PIPELINE_WORK_ROOT", str(REPO_ROOT))).resolve()
    stamp_out = (work_dir / "stamp_outputs").resolve()
    digit_out = (work_dir / "debug_ballot").resolve()
    logs_dir = (work_dir / "logs").resolve()

    shutil.rmtree(stamp_out, ignore_errors=True)
    shutil.rmtree(digit_out, ignore_errors=True)
    stamp_out.mkdir(parents=True, exist_ok=True)
    digit_out.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pipeline_input = image

    # 1) Stamp detection
    predict_cmd = [
        sys.executable,
        "-m",
        "src.infer.predict",
        "--weights",
        str(weights),
        "--images",
        str(pipeline_input),
        "--out_dir",
        str(stamp_out),
    ]
    if yolo_device:
        predict_cmd.extend(["--device", yolo_device])
    predict_proc = _run_checked(predict_cmd, cwd=STAMP_ROOT)

    # 2) Handwritten extraction
    expected_rows = fetch_expected_rows()
    hwr_cmd = [
        sys.executable,
        "-m",
        "ballot_reader.cli",
        "--input",
        str(pipeline_input),
        "--out",
        str(digit_out),
        "--model",
        str(model),
    ]
    if expected_rows is not None:
        hwr_cmd.extend(["--expected", str(expected_rows)])
    hwr_proc = _run_checked(hwr_cmd, cwd=HWR_ROOT)

    # 3) Merge + Supabase insert
    yolo_log = stamp_out / "inference_manifest.json"
    merged_json = logs_dir / "ballots_merged.json"
    merged_csv = logs_dir / "ballots_merged.csv"
    merge_env = {
        "YOLO_LOG": str(yolo_log),
        "DIGIT_OUT_DIR": str(digit_out),
        "OUTPUT_JSON": str(merged_json),
        "OUTPUT_CSV": str(merged_csv),
    }
    if image_url:
        merge_env["MERGE_IMAGE_URL"] = image_url
    merge_proc = _run_checked([sys.executable, "merge_ballot_logs.py"], cwd=REPO_ROOT, env=merge_env)

    return {
        "original_image": str(image),
        "pipeline_input": str(pipeline_input),
        "work_dir": str(work_dir),
        "yolo_log": str(yolo_log),
        "digit_out": str(digit_out),
        "merged_json": str(merged_json),
        "merged_csv": str(merged_csv),
        "diagnostics": _runtime_diagnostics(image, weights, model),
        "debug_summary": _build_debug_summary(yolo_log, merged_json, image_url),
        "predict_stdout": predict_proc.stdout,
        "predict_stderr": predict_proc.stderr,
        "hwr_stdout": hwr_proc.stdout,
        "hwr_stderr": hwr_proc.stderr,
        "merge_stdout": merge_proc.stdout,
        "merge_stderr": merge_proc.stderr,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Runtime single-ballot pipeline")
    parser.add_argument("--image", required=True, help="Path to one ballot image")
    parser.add_argument("--stamp_weights", default=None, help="Path to YOLO weights")
    parser.add_argument("--digit_model", default=None, help="Path to Keras model")
    parser.add_argument("--work_root", default=None, help="Working/output directory root")
    parser.add_argument("--yolo_device", default=os.getenv("YOLO_DEVICE"), help="YOLO device (cpu, 0, etc.)")
    parser.add_argument("--image_url", default=None, help="Optional canonical image URL stored by merge")
    args = parser.parse_args()

    result = process_single_ballot(
        image_path=args.image,
        stamp_weights=args.stamp_weights,
        digit_model=args.digit_model,
        work_root=args.work_root,
        yolo_device=args.yolo_device,
        image_url=args.image_url,
    )
    print(result)

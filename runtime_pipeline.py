import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent
STAMP_ROOT = REPO_ROOT / "stamp-detection"
HWR_ROOT = REPO_ROOT / "Handwritten-Digit-Recognition"
REMOVE_BG_ROOT = REPO_ROOT / "remove-background"
DEFAULT_STAMP_WEIGHTS = STAMP_ROOT / "runs" / "train" / "yolo_stamp" / "weights" / "best.pt"
DEFAULT_DIGIT_MODEL = HWR_ROOT / "tf-cnn-model.keras"

if str(REMOVE_BG_ROOT) not in sys.path:
    sys.path.insert(0, str(REMOVE_BG_ROOT))

from remove_background import crop_ballot_paper


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
    Run background removal + stamp detection + handwritten extraction + merge on one image.
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
    remove_bg_out = (work_dir / "remove_background").resolve()
    remove_bg_debug = (work_dir / "debug_remove_background").resolve()

    shutil.rmtree(stamp_out, ignore_errors=True)
    shutil.rmtree(digit_out, ignore_errors=True)
    shutil.rmtree(remove_bg_out, ignore_errors=True)
    shutil.rmtree(remove_bg_debug, ignore_errors=True)
    stamp_out.mkdir(parents=True, exist_ok=True)
    digit_out.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    remove_bg_out.mkdir(parents=True, exist_ok=True)
    remove_bg_debug.mkdir(parents=True, exist_ok=True)

    # 0) Background removal / ballot crop (must run first)
    cropped_image = remove_bg_out / f"{image.stem}_ballot_crop.png"
    crop_result = crop_ballot_paper(
        input_path=str(image),
        output_path=str(cropped_image),
        debug_dir=str(remove_bg_debug / image.stem),
    )

    # 1) Stamp detection
    predict_cmd = [
        sys.executable,
        "-m",
        "src.infer.predict",
        "--weights",
        str(weights),
        "--images",
        str(cropped_image),
        "--out_dir",
        str(stamp_out),
    ]
    if yolo_device:
        predict_cmd.extend(["--device", yolo_device])
    _run_checked(predict_cmd, cwd=STAMP_ROOT)

    # 2) Handwritten extraction
    expected_rows = fetch_expected_rows()
    hwr_cmd = [
        sys.executable,
        "-m",
        "ballot_reader.cli",
        "--input",
        str(cropped_image),
        "--out",
        str(digit_out),
        "--model",
        str(model),
    ]
    if expected_rows is not None:
        hwr_cmd.extend(["--expected", str(expected_rows)])
    _run_checked(hwr_cmd, cwd=HWR_ROOT)

    # 3) Merge + Supabase insert
    merge_env = {
        "YOLO_LOG": str(stamp_out / "inference_manifest.json"),
        "DIGIT_OUT_DIR": str(digit_out),
        "OUTPUT_JSON": str(logs_dir / "ballots_merged.json"),
        "OUTPUT_CSV": str(logs_dir / "ballots_merged.csv"),
    }
    if image_url:
        merge_env["MERGE_IMAGE_URL"] = image_url
    merge_proc = _run_checked([sys.executable, "merge_ballot_logs.py"], cwd=REPO_ROOT, env=merge_env)

    return {
        "original_image": str(image),
        "cropped_image": str(cropped_image),
        "remove_background": {
            "debug_dir": crop_result.debug_dir,
            "bbox": list(crop_result.bbox),
            "used_fallback": crop_result.used_fallback,
        },
        "work_dir": str(work_dir),
        "yolo_log": str(stamp_out / "inference_manifest.json"),
        "digit_out": str(digit_out),
        "merged_json": str(logs_dir / "ballots_merged.json"),
        "merged_csv": str(logs_dir / "ballots_merged.csv"),
        "merge_stdout": merge_proc.stdout,
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

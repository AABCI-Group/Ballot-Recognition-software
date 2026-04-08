# Ballot Stamp Verifier: Complete Setup and End-to-End Process Guide

This guide explains the project from zero: what it does, how to install it, and what each pipeline operation does internally.

## 1) What this project does

The repository processes ballot images and produces:
- Stamp validation results (`VALID STAMP`, `REVIEW REQUIRED`, `NO STAMP`)
- Handwritten preference digit extraction per candidate row
- Merged audit outputs (`JSON`, `CSV`)
- Optional inserts into Supabase tables

It is split into three main engines:
- `remove-background/`: crops/rectifies ballot paper from a raw photo
- `stamp-detection/`: detects and validates official stamp using YOLO + geometry rules
- `Handwritten-Digit-Recognition/`: detects vote boxes and reads handwritten digits with a TensorFlow model

Main orchestrators at repo root:
- `runtime_pipeline.py`: single-ballot runtime flow (crop -> stamp -> digits -> merge)
- `lambda_handler.py`: AWS Lambda S3 event entrypoint with idempotency
- `merge_ballot_logs.py`: merges outputs and optionally writes to Supabase

## 2) High-level architecture

For one ballot image, runtime flow is:
1. Input image is loaded.
2. Ballot paper is detected, perspective-corrected, and cropped.
3. Cropped image is sent to stamp inference.
4. Same cropped image is sent to handwritten digit extraction.
5. Results are merged into final records.
6. Optional database writes happen (if Supabase key exists).

Key output locations (default local run):
- `stamp_outputs/inference_manifest.json` (or `stamp-detection/outputs/inference_manifest.json` in legacy flow)
- `debug_ballot/ballot_<id>/results.json`
- `logs/ballots_merged.json`
- `logs/ballots_merged.csv`

## 3) Prerequisites (fresh machine)

Minimum:
- Python 3.11 recommended
- `pip`
- Git

Optional but recommended for training/inference speed:
- NVIDIA GPU + updated driver

Cloud/runtime optional:
- AWS account (for Lambda/S3/DynamoDB path)
- Supabase project/service key (for DB inserts)

## 4) Project setup from zero (local)

### 4.1 Clone and enter the repo

```powershell
git clone https://github.com/AABCI-Group/Ballot-Recognition-software.git
cd ballot-stamp-verifier
```

### 4.2 Create and activate virtual environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If `py` is unavailable, use installed Python executable path.

### 4.3 Install runtime dependencies

If your goal is running the integrated runtime pipeline:

```powershell
pip install -r requirements-runtime.txt
```

This installs CPU runtime libraries (Torch CPU wheel, OpenCV headless, TensorFlow CPU, boto3, etc.).

### 4.4 Install stamp training dependencies (optional, if training stamp model)

```powershell
cd stamp-detection
pip install -r requirements.txt
```

For GPU Torch, install the matching PyTorch wheels as documented in `stamp-detection/README.md`.

### 4.5 Install handwritten module dependencies (optional, if not already covered)

```powershell
cd ..\Handwritten-Digit-Recognition
pip install -r requirements.txt
cd ..
```

### 4.6 Configure environment

Copy and edit env file:

```powershell
Copy-Item .env.example .env
```

Important variables:
- `SUPABASE_URL`
- `SUPABASE_KEY` (required for DB writes; if omitted, merge runs in offline mode)
- `ELECTION_ID`

For Lambda path, also configure:
- `IDEMPOTENCY_TABLE`
- `YOLO_DEVICE` (usually `cpu` in Lambda)

## 5) First run (single ballot, simplest end-to-end)

Use:

```powershell
python runtime_pipeline.py --image uploads/your_ballot.jpg --yolo_device cpu
```

What this command does internally:
1. Validates input image/model paths.
2. Creates working folders for stamp, digit, logs, and debug artifacts.
3. Runs ballot crop (`remove-background/remove_background.py`).
4. Runs stamp predictor (`python -m src.infer.predict`) in `stamp-detection/`.
5. Runs handwritten digit reader (`python -m ballot_reader.cli`) in `Handwritten-Digit-Recognition/`.
6. Runs `merge_ballot_logs.py` with env variables pointing to produced outputs.
7. Returns structured summary object with generated file paths.

## 6) Detailed operation breakdown by stage

## 6.1 Stage A: Ballot paper crop and rectification (`remove-background`)

Entry function: `crop_ballot_paper(...)`

Core operations:
1. Reads raw image.
2. Tries geometric quad detection first:
   - Edge-based quad detection (`_find_ballot_quad_edges`)
   - LAB paper-mask quad detection (`_find_ballot_quad_paper_mask`)
3. If quad found:
   - Warps perspective (`_warp_from_quad`)
4. If quad not found:
   - Falls back to contour detectors:
     - OMR-style contour
     - edge mask contour
     - brightness mask contour
     - white/chroma contour
5. Crops/rectifies and trims white border.
6. Writes cropped ballot image and debug files.

Why this stage exists:
- Raw phone images include desk/background/perspective skew.
- Downstream stamp and digit detectors are more stable on normalized ballot-only images.

Important debug outputs (when debug path provided):
- `original_input.png`
- `detected_region.png`
- `warped_before_mask.png`
- `paper_mask.png`
- `final_cropped_output.png`
- `crop_meta.json`

## 6.2 Stage B: Stamp detection and decisioning (`stamp-detection`)

Main runtime script: `stamp-detection/src/infer/predict.py`

Core operations:
1. Loads decision thresholds from `configs/val_rules.yaml`.
2. Builds `Thresholds` and `GeoConfig`.
3. Loads YOLO model from `--weights`.
4. Resolves device:
   - `auto` -> CUDA if available, else CPU
5. For each image:
   - Deskews + normalizes image (`deskew_bgr`, `normalize`)
   - Runs YOLO detection (`model.predict`)
   - Chooses best box confidence
   - Runs geometry validation on crop:
     - area fraction check
     - circularity
     - ellipse edge overlap
     - optional template NCC
6. Combines model score + geometry checks into decision:
   - `VALID STAMP`
   - `REVIEW REQUIRED`
   - `NO STAMP`
7. Writes:
   - annotated image
   - crop (if bbox exists)
   - `inference_manifest.json`

Decision logic is in `stamp-detection/src/infer/postprocess.py`:
- score >= `score_valid` and geometry pass -> valid
- score < `score_review` -> no stamp
- otherwise -> review required

## 6.3 Stage C: Handwritten digit extraction (`Handwritten-Digit-Recognition`)

Main runtime script: `ballot_reader/cli.py`

Core operations per ballot:
1. Preprocess full ballot:
   - illumination normalization
   - deskew by Hough line median angle
2. Layout detection:
   - Detect vote boxes (right-side ROI first)
   - Build row bands from box geometry
   - Refine rows via separator lines
   - Optional legacy fallback if quality checks fail
3. Align one box per row:
   - assign boxes to rows
   - repair missing box entries with median geometry
4. For each row:
   - Rectify vote box (perspective to canonical square)
   - Enhance ink via multiple thresholding paths
   - Segment and clean components (remove border noise/frame)
   - Blank/noise gate
   - If non-blank: crop digit + run model inference with preprocessing variants
5. Emit row result:
   - recognized digit or `NULL`
   - rich debug metadata
6. Write `results.json` and `results.csv` per ballot directory.

Model load:
- `load_mnist28_model` loads Keras model path (default `tf-cnn-model.keras`)

## 6.4 Stage D: Merge and optional database persistence (`merge_ballot_logs.py`)

Core operations:
1. Loads stamp manifest (`YOLO_LOG` env).
2. Loads digit outputs from `DIGIT_OUT_DIR/ballot_*/results.json`.
3. Matches stamp and digit records by normalized filename/stem.
4. Builds merged ballot records:
   - stamp label + score
   - sequence quality and extracted row digits
5. Writes merged artifacts:
   - `OUTPUT_JSON`
   - `OUTPUT_CSV`
6. If `SUPABASE_KEY` exists:
   - fetches candidate rows for election
   - inserts one row into `BallotPaperTBL` per ballot
   - inserts many rows into `BallotPreferenceTBL` for row preferences
7. If no key:
   - runs in offline merge mode and skips inserts.

Ballot state mapping:
- `VALID STAMP` + digit `sequence_ok=true` -> `Valid`
- otherwise -> `Doubtful`

## 7) Runtime entrypoints and when to use each

## 7.1 `runtime_pipeline.py` (recommended local runtime)

Use when processing one image end-to-end with strict checked subprocess calls.

Example:

```powershell
python runtime_pipeline.py --image uploads/1000013245.jpg --yolo_device cpu
```

## 7.2 `run_full_extraction.py` (legacy multi-image + S3 upload orchestrator)

Use when you want local directory processing plus S3 upload in one script.

Example:

```powershell
python run_full_extraction.py --images uploads --bucket ballot-imgs --s3_prefix raw-images/
```

## 7.3 `upload_server.py` (manual capture/upload front-end)

What it provides:
- drag/drop web uploader
- persisted current box location (`/set_box?box=...`)
- uploaded image serving
- health endpoint (`/healthz`)

Run:

```powershell
python upload_server.py
```

Open:
- `http://localhost:8000/`

## 7.4 `lambda_handler.py` (AWS Lambda S3 event processing)

What it does:
1. Receives S3 `ObjectCreated` event.
2. Picks newest record in event batch.
3. Computes idempotency key (`bucket + key + version + etag`).
4. Uses DynamoDB conditional write to prevent duplicate processing.
5. Downloads object to `/tmp`.
6. Runs `process_single_ballot(...)`.
7. Marks DynamoDB status `SUCCEEDED` or `FAILED`.

## 8) Stamp model training process (if you need to retrain)

From `stamp-detection/`:

1. Prepare blank ballots in `data/blanks/` and stamp PNG in `assets/stamp.png`.
2. Generate synthetic data:

```powershell
python src/tools/generate_synthetic.py --blanks_dir data/blanks --stamp_png assets/stamp.png --out_dir data/synth --count 100
```

3. Split YOLO dataset:

```powershell
python src/tools/split_yolo.py --images data/synth/images --labels_det data/synth/labels_det --out_dir data/yolo_split
```

4. Train:

```powershell
python -m src.models.train_yolo --epochs 50 --imgsz 640 --batch 16
```

5. Export (if needed explicitly):

```powershell
python -m src.models.export_yolo --weights runs/train/yolo_stamp/weights/best.pt
```

6. Evaluate/calibrate:

```powershell
python -m src.eval.evaluate --pred_manifest outputs/inference_manifest.json --labels_dir data/yolo_split/val/labels
python -m src.eval.calibrate_thresholds
```

## 9) Expected files after successful end-to-end run

Common outputs:
- Cropped ballot image under runtime working directory
- Stamp:
  - `*_annot.png`
  - `crops/*_crop.png` (if bbox found)
  - `inference_manifest.json`
- Digits:
  - `debug_ballot/ballot_<id>/results.json`
  - `debug_ballot/ballot_<id>/results.csv`
- Merge:
  - `logs/ballots_merged.json`
  - `logs/ballots_merged.csv`

## 10) Troubleshooting checklist

If stamp inference fails:
- Check weights path exists.
- Run with `--yolo_device cpu` first.
- Verify `ultralytics`, `torch`, `opencv` installed in same environment.

If digit extraction fails:
- Confirm `Handwritten-Digit-Recognition/tf-cnn-model.keras` exists.
- Ensure TensorFlow is installed in active environment.

If merge fails:
- Ensure `YOLO_LOG` and `DIGIT_OUT_DIR` point to real output files.
- For Supabase writes, verify `SUPABASE_KEY` is set.

If Lambda duplicates occur:
- Verify DynamoDB table key is `idempotency_key` and TTL field is enabled.

## 11) Recommended first-time learning order

1. Run `runtime_pipeline.py` on one sample image.
2. Inspect crop debug output.
3. Inspect stamp manifest + annotated output.
4. Inspect `debug_ballot/ballot_*/results.json`.
5. Inspect `logs/ballots_merged.json`.
6. Then move to `run_full_extraction.py` or Lambda deployment.

## 12) Quick command reference

Local one-image runtime:

```powershell
python runtime_pipeline.py --image uploads/your_ballot.jpg --yolo_device cpu
```

Local upload server:

```powershell
python upload_server.py
```

Legacy batch runtime:

```powershell
python run_full_extraction.py --images uploads --bucket <bucket> --s3_prefix raw-images/
```

Stamp-only prediction:

```powershell
cd stamp-detection
python -m src.infer.predict --weights runs/train/yolo_stamp/weights/best.pt --images data/SampleBallots --out_dir outputs
```

Digit-only prediction:

```powershell
cd Handwritten-Digit-Recognition
python -m ballot_reader.cli --input ..\uploads --out ..\debug_ballot --model tf-cnn-model.keras
```

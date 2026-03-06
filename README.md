# Ballot Verifier Workspace

This repo is now split into two engines:

- `stamp-detection/`: stamp detection training/inference pipeline (YOLO + geometry checks)
- `Handwritten-Digit-Recognition/`: handwritten digit extraction pipeline

The base folder orchestrates both pipelines and merges results.

## Base-folder entry points

- Upload server: `python upload_server.py`
- Folder watcher: `powershell -ExecutionPolicy Bypass -File watch_downloads.ps1`
- Full pipeline (stamp + handwritten + merge):
  - `python run_full_extraction.py --images uploads/ --bucket ballot-imgs --s3_prefix raw-images/`
- Merge only:
  - `python merge_ballot_logs.py`

## Stamp pipeline

Run stamp-only commands from `stamp-detection/`:

- `python -m src.runtime.service_cli --prepare_synth`
- `python -m src.runtime.service_cli --train`
- `python -m src.runtime.service_cli --predict --images data/SampleBallots`

For full details, see `stamp-detection/README.md`.

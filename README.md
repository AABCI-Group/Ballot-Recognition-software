# Ballot Verifier Workspace

This repo is now split into two engines:

- `stamp-detection/`: stamp detection training/inference pipeline (YOLO + geometry checks)
- `Handwritten-Digit-Recognition/`: handwritten digit extraction pipeline

The base folder orchestrates both pipelines and merges results.

## Base-folder entry points

- Upload server (local/manual): `python upload_server.py`
- Full runtime pipeline for one image (raw ballot + stamp + handwritten + merge):
  - `python runtime_pipeline.py --image uploads/1000013245.jpg --yolo_device cpu`
- Byte-parity probe for local file vs S3 object:
  - `python compare_image_parity.py --local-file uploads/IMG20260312213954.jpg --bucket your-bucket --key raw-images/IMG20260312213954.jpg`
- Standalone background-removal test:
  - python remove-background/cli.py --input uploads --out_dir runtime-test/remove-background/crops --debug_dir runtime-test/remove-background/debug
- Legacy orchestrator (still available):
  - `python run_full_extraction.py --images uploads/ --bucket ballot-imgs --s3_prefix raw-images/`
- Merge only:
  - `python merge_ballot_logs.py`

`watch_downloads.ps1` is retired. Use Lambda S3 ObjectCreated processing via `lambda_handler.py`.

## Container and Lambda deployment

See `DEPLOYMENT.md` for runtime-only Dockerfiles, Lambda S3 trigger flow, idempotency strategy, and local event simulation.

Lambda now emits `input_diagnostics` for each S3 event, including S3 bucket/key, S3 `head_object` metadata, downloaded-file byte size, SHA-256, and decoded image dimensions. That makes it straightforward to prove whether bytes changed before or after S3 storage.

## Stamp pipeline

Run stamp-only commands from `stamp-detection/`:

- `python -m src.runtime.service_cli --prepare_synth`
- `python -m src.runtime.service_cli --train`
- `python -m src.runtime.service_cli --predict --images data/SampleBallots`

For full details, see `stamp-detection/README.md`.

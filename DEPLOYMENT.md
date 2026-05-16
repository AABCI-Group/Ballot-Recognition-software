# Containerization and Deployment Design (Runtime-Only)

This design uses existing model artifacts only:
- Stamp model: `stamp-detection/runs/train/yolo_stamp/weights/best.pt`
- Digit model: `Handwritten-Digit-Recognition/tf-cnn-model.keras`

No training pipeline is required in images.

## Recommended Topology

Use **two images**:
1. `Dockerfile.app` for local/manual server use (`upload_server.py`, smoke tests, optional batch invocation).
2. `Dockerfile.lambda` for AWS Lambda S3 event processing.

Reason: the Lambda image should be minimal and tuned for `/tmp` ephemeral execution and event handler startup path. The app image keeps local/dev ergonomics.

## Lambda Event Flow (watch_downloads replacement)

`lambda_handler.lambda_handler`:
1. Receives `ObjectCreated` event from S3.
2. Chooses the newest record when a batch contains multiple records.
3. Uses DynamoDB conditional write (`idempotency_key`) to dedupe retries/duplicates.
4. Downloads object to `/tmp/ballot-runtime/.../input/<filename>`.
   - Lambda now records `input_diagnostics` with:
     - S3 bucket/key/version id
     - `head_object` content length, content type, content disposition, content encoding, etag, and custom metadata
     - downloaded file byte size
     - downloaded file SHA-256
     - decoded image width/height/channels
5. Runs `runtime_pipeline.process_single_ballot(...)`:
   - raw ballot image from S3 is used directly as pipeline input
   - stamp inference (`src.infer.predict`)
   - digit extraction (`ballot_reader.cli`)
   - merge + Supabase insert (`merge_ballot_logs.py`)
6. Updates idempotency row to `SUCCEEDED` or `FAILED`.

## Idempotency Strategy

Table key: `idempotency_key` (string) composed from `bucket + key + versionId + eTag`.

Processing guard:
- `PutItem` with `ConditionExpression attribute_not_exists(idempotency_key)`.
- If condition fails, event is treated as duplicate and skipped.
- Row status transitions: `IN_PROGRESS` -> `SUCCEEDED` / `FAILED`.
- TTL cleanup via `ttl` epoch field (`IDEMPOTENCY_TTL_DAYS`).

## CPU/GPU Notes

- Lambda path is CPU-only (`YOLO_DEVICE=cpu` default).
- Stamp predictor now supports `--device` and auto fallback (GPU if available, else CPU).
- Previous hardcoded `device=0` behavior has been removed.

## Paths, State, Persistence

Local Docker mounts (compose):
- `./uploads -> /app/uploads`
- `./logs -> /app/runtime/logs`
- `./debug_ballot -> /app/runtime/debug_ballot`
- `./stamp-detection/outputs -> /app/runtime/stamp_outputs`

Lambda:
- All runtime writes go under `/tmp/ballot-runtime`.
- No assumption of persistent local filesystem between invocations.

## Security and Ops

- No secrets in code/image layers; provide via env/Secrets Manager.
- `merge_ballot_logs.py` now requires `SUPABASE_KEY` from environment (no embedded default key).
- App container runs as non-root `appuser`.
- Health check endpoint: `GET /healthz` in `upload_server.py`.
- Supabase requests use retry + backoff + timeout controls.

## Build

```bash
docker build -f Dockerfile.app -t ballot-stamp-verifier:app .
docker build -f Dockerfile.lambda -t ballot-stamp-verifier:lambda .
```

## ECR/Lambda Manifest Error (PowerShell)

If Lambda returns:
`The image manifest, config or layer media type ... is not supported`,
build and push a single-platform image without provenance/sbom attestations.

```powershell
$AWS_REGION = "us-east-1"
$ACCOUNT_ID = "246121021858"
$REPO_NAME  = "ballot-stamp-verifier-lambda"
$IMAGE_TAG  = "lambda-amd64"

$ECR_URI = "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME"

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker buildx build `
  --platform linux/amd64 `
  --provenance=false `
  --sbom=false `
  -f Dockerfile.lambda `
  -t "${ECR_URI}:$IMAGE_TAG" `
  --push `
  .

aws lambda update-function-configuration `
  --function-name "ballot-stamp-verifier-lambda" `
  --architectures x86_64 `
  --region $AWS_REGION

aws lambda update-function-code `
  --function-name "ballot-stamp-verifier-lambda" `
  --image-uri "${ECR_URI}:$IMAGE_TAG" `
  --region $AWS_REGION
```

If your function architecture is `arm64`, use `--platform linux/arm64` and
set `--architectures arm64`.
## Local Smoke Test (single image)

```bash
python runtime_pipeline.py \
  --image uploads/1000013245.jpg \
  --stamp_weights stamp-detection/runs/train/yolo_stamp/weights/best.pt \
  --digit_model Handwritten-Digit-Recognition/tf-cnn-model.keras \
  --work_root ./runtime-test \
  --yolo_device cpu \
  --image_url s3://example-bucket/raw-images/1000013245.jpg

This runtime path now skips `remove_background` and feeds the original ballot image
directly into stamp detection and handwritten extraction.
```

## Local Lambda Invocation (S3 event simulation)

Start lambda container:
```bash
docker compose up lambda-local
```

Invoke with sample event:
```bash
curl -s -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d @events/s3_object_created.json
```

The invocation response now includes `input_diagnostics`, and the same JSON is printed to container logs under a `[lambda-input-diagnostics]` marker.

## Local Byte-Parity Checks

Use the same pinned Lambda runtime image to compare a local file and an S3 object:

```powershell
docker build -f Dockerfile.lambda -t ballot-stamp-verifier:lambda .

docker run --rm `
  --env-file .env `
  -e AWS_REGION=us-east-1 `
  -e AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID `
  -e AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY `
  -e AWS_SESSION_TOKEN=$env:AWS_SESSION_TOKEN `
  -v ${PWD}:/workspace `
  ballot-stamp-verifier:lambda `
  python /var/task/compare_image_parity.py `
    --local-file /workspace/uploads/IMG20260312213954.jpg `
    --bucket your-bucket `
    --key raw-images/IMG20260312213954.jpg
```

If you only want to hash the S3 object exactly as Lambda downloads it:

```powershell
docker run --rm `
  --env-file .env `
  -e AWS_REGION=us-east-1 `
  -e AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID `
  -e AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY `
  -e AWS_SESSION_TOKEN=$env:AWS_SESSION_TOKEN `
  ballot-stamp-verifier:lambda `
  python /var/task/compare_image_parity.py `
    --bucket your-bucket `
    --key raw-images/IMG20260312213954.jpg
```

This uses the same `requirements-runtime.txt` dependency set and base image as Lambda, which helps eliminate host-environment drift when reproducing byte and decode behavior locally.

## Failure and Retry Behavior

- Lambda raises on failures so AWS retry/DLQ behavior applies.
- Duplicate event retries are short-circuited by idempotency table.
- Supabase transient failures are retried by HTTP adapter policy.

## Python Compatibility Guidance

Use Python **3.11** for mixed TensorFlow + PyTorch runtime stability.
Python 3.13 is not recommended for this dependency set.

To minimize dependency drift, run local parity probes through `Dockerfile.lambda` instead of a host Python environment. The runtime pipeline already reports package versions, Python executable, and platform in its diagnostics payload.

## AWS Lambda Configuration Recommendations

- Runtime image: `Dockerfile.lambda` pushed to ECR.
- Architecture: `x86_64` (most compatible with TensorFlow/PyTorch wheel mix).
- Memory: start at `3008 MB` and tune with CloudWatch duration.
- Timeout: start at `120s`.
- Ephemeral storage: at least `2048 MB` (increase if larger ballot images/batches).
- Reserved concurrency: set based on Supabase throughput and expected S3 ingest rate.
- Retry destination: configure DLQ (SQS) or on-failure destination.

## IAM Permissions (Lambda Role)

- `s3:GetObject` on input bucket/prefix.
- `dynamodb:PutItem`, `dynamodb:UpdateItem`, `dynamodb:GetItem` on idempotency table.
- Network egress to Supabase endpoint (via NAT/public subnet as needed).

## DynamoDB Table Shape

- Partition key: `idempotency_key` (String).
- TTL attribute: `ttl` (Number, epoch seconds).
- Optional GSI for operations dashboard by `status`.

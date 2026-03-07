import json
import os
import tempfile
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from runtime_pipeline import process_single_ballot


s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

IDEMPOTENCY_TABLE = os.getenv("IDEMPOTENCY_TABLE", "")
IDEMPOTENCY_TTL_DAYS = int(os.getenv("IDEMPOTENCY_TTL_DAYS", "7"))
PIPELINE_TMP_ROOT = os.getenv("PIPELINE_TMP_ROOT", "/tmp/ballot-runtime")
STAMP_WEIGHTS = os.getenv(
    "STAMP_WEIGHTS",
    "/var/task/stamp-detection/runs/train/yolo_stamp/weights/best.pt",
)
DIGIT_MODEL = os.getenv(
    "DIGIT_MODEL",
    "/var/task/Handwritten-Digit-Recognition/tf-cnn-model.keras",
)
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")


def _require_env() -> None:
    if not IDEMPOTENCY_TABLE:
        raise RuntimeError("IDEMPOTENCY_TABLE is required for duplicate event handling")


def _record_sort_key(record: Dict[str, Any]) -> str:
    # Sort by eventTime + sequencer for deterministic newest-first processing.
    event_time = record.get("eventTime") or ""
    sequencer = (((record.get("s3") or {}).get("object") or {}).get("sequencer")) or ""
    return f"{event_time}|{sequencer}"


def _pick_newest_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("No S3 records in event")
    return sorted(records, key=_record_sort_key, reverse=True)[0]


def _idempotency_key(bucket: str, key: str, version_id: Optional[str], etag: Optional[str]) -> str:
    version = version_id or "no-version"
    etag_norm = (etag or "").replace('"', "")
    return f"{bucket}::{key}::{version}::{etag_norm}"


def _put_if_absent(token: str, payload: Dict[str, Any]) -> bool:
    expires = int((datetime.now(timezone.utc) + timedelta(days=IDEMPOTENCY_TTL_DAYS)).timestamp())
    item = {
        "idempotency_key": {"S": token},
        "status": {"S": "IN_PROGRESS"},
        "created_at": {"S": datetime.now(timezone.utc).isoformat()},
        "ttl": {"N": str(expires)},
        "payload": {"S": json.dumps(payload, separators=(",", ":"))},
    }
    try:
        dynamodb.put_item(
            TableName=IDEMPOTENCY_TABLE,
            Item=item,
            ConditionExpression="attribute_not_exists(idempotency_key)",
        )
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def _set_status(token: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    expr_values = {
        ":status": {"S": status},
        ":updated_at": {"S": datetime.now(timezone.utc).isoformat()},
    }
    update = "SET #s = :status, updated_at = :updated_at"
    expr_names = {"#s": "status"}

    if details is not None:
        expr_values[":details"] = {"S": json.dumps(details, separators=(",", ":"))}
        update += ", details = :details"

    dynamodb.update_item(
        TableName=IDEMPOTENCY_TABLE,
        Key={"idempotency_key": {"S": token}},
        UpdateExpression=update,
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
    )


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    _require_env()
    records = event.get("Records") or []
    if not records:
        return {"status": "ignored", "reason": "no_records"}

    record = _pick_newest_record(records)
    s3_info = record.get("s3") or {}
    bucket = ((s3_info.get("bucket") or {}).get("name"))
    obj = s3_info.get("object") or {}
    raw_key = obj.get("key")
    version_id = obj.get("versionId")
    etag = obj.get("eTag")

    if not bucket or not raw_key:
        raise ValueError("S3 record is missing bucket or key")

    key = urllib.parse.unquote_plus(raw_key)
    token = _idempotency_key(bucket, key, version_id, etag)
    payload = {"bucket": bucket, "key": key, "version_id": version_id, "etag": etag}

    if not _put_if_absent(token, payload):
        return {
            "status": "duplicate",
            "bucket": bucket,
            "key": key,
            "idempotency_key": token,
        }

    try:
        Path(PIPELINE_TMP_ROOT).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ballot_", dir=PIPELINE_TMP_ROOT) as tmp_dir:
            input_dir = Path(tmp_dir) / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            local_image = input_dir / Path(key).name
            s3.download_file(bucket, key, str(local_image))

            image_url = f"s3://{bucket}/{key}"
            pipeline_result = process_single_ballot(
                image_path=str(local_image),
                stamp_weights=STAMP_WEIGHTS,
                digit_model=DIGIT_MODEL,
                work_root=tmp_dir,
                yolo_device=YOLO_DEVICE,
                image_url=image_url,
            )

            _set_status(token, "SUCCEEDED", {"pipeline": pipeline_result})
            return {
                "status": "ok",
                "bucket": bucket,
                "key": key,
                "idempotency_key": token,
                "result": pipeline_result,
            }
    except Exception as exc:
        _set_status(token, "FAILED", {"error": str(exc)})
        raise

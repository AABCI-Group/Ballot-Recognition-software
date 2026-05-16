import json
import os
import tempfile
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from image_parity import local_file_diagnostics, summarize_s3_head, to_pretty_json
from runtime_pipeline import process_single_ballot


s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

IDEMPOTENCY_TABLE = os.getenv("IDEMPOTENCY_TABLE", "")
IDEMPOTENCY_TTL_DAYS = int(os.getenv("IDEMPOTENCY_TTL_DAYS", "7"))
IDEMPOTENCY_IN_PROGRESS_TIMEOUT_SEC = int(os.getenv("IDEMPOTENCY_IN_PROGRESS_TIMEOUT_SEC", "130"))
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
    now = datetime.now(timezone.utc)
    expires = int((now + timedelta(days=IDEMPOTENCY_TTL_DAYS)).timestamp())
    item = {
        "idempotency_key": {"S": token},
        "status": {"S": "IN_PROGRESS"},
        "created_at": {"S": now.isoformat()},
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


def _get_item(token: str) -> Optional[Dict[str, Any]]:
    resp = dynamodb.get_item(
        TableName=IDEMPOTENCY_TABLE,
        Key={"idempotency_key": {"S": token}},
        ConsistentRead=True,
    )
    return resp.get("Item")


def _parse_created_at(item: Dict[str, Any]) -> Optional[datetime]:
    value = ((item.get("created_at") or {}).get("S")) if item else None
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _claim_existing(token: str, payload: Dict[str, Any]) -> bool:
    item = _get_item(token)
    if not item:
        # A race may have deleted/expired the item between writes.
        return _put_if_absent(token, payload)

    status = ((item.get("status") or {}).get("S")) or ""
    now = datetime.now(timezone.utc)
    expires = int((now + timedelta(days=IDEMPOTENCY_TTL_DAYS)).timestamp())
    created_at = _parse_created_at(item)
    stale_cutoff = now - timedelta(seconds=IDEMPOTENCY_IN_PROGRESS_TIMEOUT_SEC)

    # Reclaim stale locks created by hard timeout (Lambda timeout kills process).
    reclaim_stale = status == "IN_PROGRESS" and created_at is not None and created_at <= stale_cutoff
    retry_failed = status == "FAILED"
    if not (reclaim_stale or retry_failed):
        return False

    expr_values = {
        ":in_progress": {"S": "IN_PROGRESS"},
        ":now": {"S": now.isoformat()},
        ":ttl": {"N": str(expires)},
        ":payload": {"S": json.dumps(payload, separators=(",", ":"))},
    }
    condition = "#s = :expected_status"
    expr_values[":expected_status"] = {"S": status}

    # Keep condition tight for stale IN_PROGRESS reclaiming.
    if reclaim_stale and created_at is not None:
        condition += " AND created_at = :expected_created_at"
        expr_values[":expected_created_at"] = {"S": created_at.isoformat()}

    try:
        dynamodb.update_item(
            TableName=IDEMPOTENCY_TABLE,
            Key={"idempotency_key": {"S": token}},
            ConditionExpression=condition,
            UpdateExpression="SET #s = :in_progress, created_at = :now, updated_at = :now, ttl = :ttl, payload = :payload",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues=expr_values,
        )
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def _acquire_processing_slot(token: str, payload: Dict[str, Any]) -> bool:
    if _put_if_absent(token, payload):
        return True
    return _claim_existing(token, payload)


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


def _head_s3_object(bucket: str, key: str, version_id: Optional[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"Bucket": bucket, "Key": key}
    if version_id:
        params["VersionId"] = version_id
    response = s3.head_object(**params)
    return summarize_s3_head(response, bucket=bucket, key=key, version_id=version_id)


def _download_s3_object(bucket: str, key: str, version_id: Optional[str], destination: Path) -> None:
    if version_id:
        s3.download_file(bucket, key, str(destination), ExtraArgs={"VersionId": version_id})
        return
    s3.download_file(bucket, key, str(destination))


def _build_input_diagnostics(bucket: str, key: str, version_id: Optional[str], local_image: Path) -> Dict[str, Any]:
    s3_object = _head_s3_object(bucket, key, version_id)
    downloaded_file = local_file_diagnostics(local_image)
    size_matches = (
        s3_object.get("content_length") == downloaded_file.get("bytes")
        if s3_object.get("content_length") is not None
        else None
    )
    diagnostics = {
        "s3_object": s3_object,
        "downloaded_file": downloaded_file,
        "size_matches_head": size_matches,
    }
    print("[lambda-input-diagnostics]")
    print(to_pretty_json(diagnostics))
    return diagnostics


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

    if not _acquire_processing_slot(token, payload):
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
            _download_s3_object(bucket, key, version_id, local_image)
            input_diagnostics = _build_input_diagnostics(bucket, key, version_id, local_image)

            image_url = f"s3://{bucket}/{key}"
            pipeline_result = process_single_ballot(
                image_path=str(local_image),
                stamp_weights=STAMP_WEIGHTS,
                digit_model=DIGIT_MODEL,
                work_root=tmp_dir,
                yolo_device=YOLO_DEVICE,
                image_url=image_url,
            )

            _set_status(token, "SUCCEEDED", {"input_diagnostics": input_diagnostics, "pipeline": pipeline_result})
            return {
                "status": "ok",
                "bucket": bucket,
                "key": key,
                "idempotency_key": token,
                "input_diagnostics": input_diagnostics,
                "result": pipeline_result,
            }
    except Exception as exc:
        failure_details: Dict[str, Any] = {"error": str(exc), "bucket": bucket, "key": key}
        print("[lambda-input-failure]")
        print(to_pretty_json(failure_details))
        _set_status(token, "FAILED", failure_details)
        raise

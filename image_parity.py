import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_image_dimensions(path: Path) -> Dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"decoded": False, "decode_error": "cv2.imread returned None"}

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    return {
        "decoded": True,
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "dtype": str(image.dtype),
    }


def local_file_diagnostics(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    stats = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stats.st_size,
        "sha256": sha256_file(resolved),
        **decode_image_dimensions(resolved),
    }


def summarize_s3_head(head: Dict[str, Any], *, bucket: str, key: str, version_id: Optional[str]) -> Dict[str, Any]:
    metadata = head.get("Metadata") or {}
    summary = {
        "bucket": bucket,
        "key": key,
        "version_id": head.get("VersionId") or version_id,
        "content_length": head.get("ContentLength"),
        "content_type": head.get("ContentType"),
        "content_encoding": head.get("ContentEncoding"),
        "content_disposition": head.get("ContentDisposition"),
        "cache_control": head.get("CacheControl"),
        "etag": (head.get("ETag") or "").replace('"', ""),
        "last_modified": head.get("LastModified").isoformat() if head.get("LastModified") else None,
        "metadata": metadata,
    }
    checksum_fields = (
        "ChecksumSHA256",
        "ChecksumSHA1",
        "ChecksumCRC32",
        "ChecksumCRC32C",
        "ChecksumCRC64NVME",
    )
    checksums = {name: head.get(name) for name in checksum_fields if head.get(name)}
    if checksums:
        summary["checksums"] = checksums
    return summary


def to_pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str)

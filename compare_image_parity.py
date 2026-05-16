import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import boto3

from image_parity import local_file_diagnostics, summarize_s3_head


def fetch_s3_diagnostics(
    *,
    bucket: str,
    key: str,
    version_id: Optional[str],
    download_path: Optional[Path],
) -> Dict[str, Any]:
    s3 = boto3.client("s3")
    head_params: Dict[str, Any] = {"Bucket": bucket, "Key": key}
    if version_id:
        head_params["VersionId"] = version_id
    head = s3.head_object(**head_params)
    summary = summarize_s3_head(head, bucket=bucket, key=key, version_id=version_id)

    local_copy = download_path
    if local_copy is None:
        suffix = Path(key).suffix or ".bin"
        with tempfile.NamedTemporaryFile(prefix="s3_object_", suffix=suffix, delete=False) as handle:
            local_copy = Path(handle.name)

    extra_args = {"VersionId": version_id} if version_id else None
    if extra_args:
        s3.download_file(bucket, key, str(local_copy), ExtraArgs=extra_args)
    else:
        s3.download_file(bucket, key, str(local_copy))

    downloaded = local_file_diagnostics(local_copy)
    return {
        "s3_object": summary,
        "downloaded_file": downloaded,
        "size_matches_head": summary.get("content_length") == downloaded.get("bytes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a local image with an S3 object using the same hash/image diagnostics as Lambda."
    )
    parser.add_argument("--local-file", default=None, help="Optional local file to hash/decode")
    parser.add_argument("--bucket", default=None, help="S3 bucket to inspect")
    parser.add_argument("--key", default=None, help="S3 key to inspect")
    parser.add_argument("--version-id", default=None, help="Optional S3 version id")
    parser.add_argument(
        "--download-path",
        default=None,
        help="Optional path to save the S3 object locally before hashing",
    )
    args = parser.parse_args()

    result: Dict[str, Any] = {}

    if args.local_file:
        result["local_file"] = local_file_diagnostics(Path(args.local_file))

    if args.bucket or args.key:
        if not args.bucket or not args.key:
            raise SystemExit("--bucket and --key must be provided together")
        result["s3_probe"] = fetch_s3_diagnostics(
            bucket=args.bucket,
            key=args.key,
            version_id=args.version_id,
            download_path=Path(args.download_path).resolve() if args.download_path else None,
        )

    if not result:
        raise SystemExit("Provide --local-file and/or --bucket with --key")

    if "local_file" in result and "s3_probe" in result:
        result["comparison"] = {
            "sha256_match": result["local_file"]["sha256"] == result["s3_probe"]["downloaded_file"]["sha256"],
            "byte_size_match": result["local_file"]["bytes"] == result["s3_probe"]["downloaded_file"]["bytes"],
            "width_match": result["local_file"].get("width") == result["s3_probe"]["downloaded_file"].get("width"),
            "height_match": result["local_file"].get("height") == result["s3_probe"]["downloaded_file"].get("height"),
        }

    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

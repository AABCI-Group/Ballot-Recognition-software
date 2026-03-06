import argparse
import os
import shutil
import subprocess
import sys

import boto3


parser = argparse.ArgumentParser(description="Run stamp detection + handwritten detection, then merge logs")
parser.add_argument("--weights", default="runs/train/yolo_stamp/weights/best.pt")
parser.add_argument("--images", required=True, help="Path to image file or directory of images")
parser.add_argument("--bucket", required=True, help="S3 bucket name to upload images to")
parser.add_argument("--s3_prefix", default="inputs/", help="S3 key prefix for uploaded images")
args = parser.parse_args()

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
STAMP_ROOT = os.path.join(REPO_ROOT, "stamp-detection")
HWR_ROOT = os.path.join(REPO_ROOT, "Handwritten-Digit-Recognition")

if os.path.exists("debug_ballot"):
    shutil.rmtree("debug_ballot")
os.makedirs("debug_ballot", exist_ok=True)

s3 = boto3.client("s3")


def guess_content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".bmp":
        return "image/bmp"
    if ext in [".tif", ".tiff"]:
        return "image/tiff"
    return "application/octet-stream"


def upload_one_file(local_path: str, bucket: str, key: str) -> None:
    content_type = guess_content_type(local_path)
    print(f"Uploading {local_path} -> s3://{bucket}/{key} (Content-Type={content_type})")

    with open(local_path, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=f,
            ContentType=content_type,
            ContentDisposition="inline",
        )


def upload_images_to_s3(path: str, bucket: str, prefix: str) -> None:
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.lower().endswith(valid_exts):
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, path)
                    key = os.path.join(prefix, rel_path).replace("\\", "/")
                    upload_one_file(full_path, bucket, key)
    else:
        if not path.lower().endswith(valid_exts):
            print(f"Warning: {path} does not look like an image, uploading anyway.")
        key = os.path.join(prefix, os.path.basename(path)).replace("\\", "/")
        upload_one_file(path, bucket, key)


print("Uploading images to S3...")
upload_images_to_s3(args.images, args.bucket, args.s3_prefix)
print("Upload complete.")

weights_arg = args.weights
if not os.path.isabs(weights_arg):
    weights_arg = os.path.join(STAMP_ROOT, weights_arg)

p1 = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "src.infer.predict",
        "--weights",
        weights_arg,
        "--images",
        os.path.abspath(args.images),
        "--out_dir",
        os.path.join(STAMP_ROOT, "outputs"),
    ],
    cwd=STAMP_ROOT,
)

input_abs = os.path.abspath(args.images)
out_abs = os.path.abspath("debug_ballot")

try:
    sys.path.insert(0, REPO_ROOT)
    import merge_ballot_logs as mbl

    candidates = mbl.fetch_candidates_for_election(mbl.ELECTION_ID)
    expected_rows = len(candidates)
except Exception as e:
    print(f"Could not fetch candidate count from Supabase: {e}; running ballot_reader without --expected")
    expected_rows = None

cli_cmd = [sys.executable, "-m", "ballot_reader.cli", "--input", input_abs, "--out", out_abs]
if expected_rows is not None:
    cli_cmd.extend(["--expected", str(expected_rows)])

p2 = subprocess.Popen(cli_cmd, cwd=HWR_ROOT)

print("Started stamp and handwritten processes, waiting for completion...")
p1.wait()
p2.wait()
print("Both processes finished.")

print("Running merge script...")
merge_proc = subprocess.run([sys.executable, "merge_ballot_logs.py"], capture_output=True, text=True, cwd=REPO_ROOT)
print(merge_proc.stdout)
print(merge_proc.stderr)

print("All done!")

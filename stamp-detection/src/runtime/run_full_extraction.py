import argparse
import subprocess
import os
import boto3
import shutil
parser = argparse.ArgumentParser(description="Run YOLO predict, digit model, then merge logs")
parser.add_argument("--weights", default="runs/train/yolo_stamp/weights/best.pt")
parser.add_argument("--images", required=True, help="Path to image file or directory of images")
parser.add_argument("--bucket", required=True, help="S3 bucket name to upload images to")
parser.add_argument("--s3_prefix", default="inputs/", help="S3 key prefix for uploaded images")
args = parser.parse_args()

if os.path.exists("debug_ballot"):
    shutil.rmtree("debug_ballot")
os.makedirs("debug_ballot", exist_ok=True)
# ---- S3 setup ----
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
            ContentDisposition="inline",  # <--- force browser to treat as inline
        )

    # Debug: read back what S3 actually stored
    meta = s3.head_object(Bucket=bucket, Key=key)
    print("S3 stored ContentType:", meta.get("ContentType"))
    print("S3 stored ContentDisposition:", meta.get("ContentDisposition"))


def upload_images_to_s3(path, bucket, prefix):
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


# --- Upload images that will be processed ---
print("Uploading images to S3...")
upload_images_to_s3(args.images, args.bucket, args.s3_prefix)
print("Upload complete.")

# --- Run YOLO + DIGIT in parallel ---
p1 = subprocess.Popen([
    "python", "-m", "src.infer.predict",
    "--weights", args.weights,
    "--images", args.images,
    "--out_dir", "outputs"
])

# p2 = subprocess.Popen([
#     # "python", "Handwritten-Digit-Recognition/load_model-bulk.py",
#     # "--images", args.images

#     # "python", "Handwritten-Digit-Recognition/ballot_reader/cli.py",
#     #  "--input", args.images,
#     #  "--out", "debug_ballot"
# ])

repo_root = os.getcwd()
hwr_root = os.path.join(repo_root, "Handwritten-Digit-Recognition")

input_abs = os.path.abspath(args.images)              # absolute path to uploads/
out_abs   = os.path.abspath("debug_ballot")           # absolute path in repo root

# Candidate row count from Supabase (same config as merge_ballot_logs.py)
try:
    import sys
    sys.path.insert(0, repo_root)
    import merge_ballot_logs as mbl
    candidates = mbl.fetch_candidates_for_election(mbl.ELECTION_ID)
    expected_rows = len(candidates)
except Exception as e:
    print(f"Could not fetch candidate count from Supabase: {e}; running ballot_reader without --expected")
    expected_rows = None

cli_cmd = ["python", "-m", "ballot_reader.cli", "--input", input_abs, "--out", out_abs]
if expected_rows is not None:
    cli_cmd.extend(["--expected", str(expected_rows)])

p2 = subprocess.Popen(cli_cmd, cwd=hwr_root)

print("Started YOLO and Digit processes… waiting for them to finish...")

p1.wait()
p2.wait()

print("Both processes finished.")

print("Running merge script...")

merge_proc = subprocess.run(
    ["python", "merge_ballot_logs.py"],
    capture_output=True,
    text=True
)

print(merge_proc.stdout)
print(merge_proc.stderr)

print("All done!")

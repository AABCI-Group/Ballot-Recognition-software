import subprocess, argparse, sys

ap = argparse.ArgumentParser(description="Ballot Stamp Verifier (offline CLI)")
ap.add_argument("--prepare_synth", action="store_true")
ap.add_argument("--train", action="store_true")
ap.add_argument("--predict", action="store_true")
ap.add_argument("--weights", default="runs/train/yolo_stamp/weights/best.pt")
ap.add_argument("--images", default="data/synth/images")
args = ap.parse_args()

if args.prepare_synth:
    subprocess.run([
        sys.executable, "src/tools/generate_synthetic.py",
        "--blanks_dir", "data/blanks",
        "--stamp_png", "assets/stamp.png",
        "--out_dir", "data/synth",
        "--count", "2000"
    ], check=True)
    subprocess.run([
        sys.executable, "src/tools/split_yolo.py",
        "--images", "data/synth/images",
        "--labels_det", "data/synth/labels_det",
        "--out_dir", "data/yolo_split"
    ], check=True)

if args.train:
    subprocess.run([
        sys.executable, "-m", "src.models.train_yolo",
        "--data", "configs/yolo_dataset.yaml",
        "--epochs", "150",
        "--imgsz", "1280",
        "--batch", "16"
    ], check=True)

if args.predict:
    subprocess.run([
        sys.executable, "-m", "src.infer.predict",
        "--weights", args.weights,
        "--images", args.images,
        "--out_dir", "outputs"
    ], check=True)

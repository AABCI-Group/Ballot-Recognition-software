from ultralytics import YOLO
import argparse, yaml, os

DEFAULT_MODEL = "yolov8n.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/yolo_dataset.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hyp", default="configs/yolo_hyps.yaml")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # Load hyperparameters YAML and merge into kwargs
    hyp = {}
    if args.hyp and os.path.exists(args.hyp):
        with open(args.hyp, "r") as f:
            hyp = yaml.safe_load(f) or {}

    overrides = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": 0,
        "workers": args.workers,
        "project": "runs/train",
        "name": "yolo_stamp",
        "patience": 20,
        "close_mosaic": 10,
        "seed": 42,
        "optimizer": "SGD",
        "amp": True,
        "pretrained": True,
        "exist_ok": True,
        "val": True,
        "single_cls": True,
        "cache": False,
        "plots": True,
        "cos_lr": False,
        "overlap_mask": False,
        "hsv_h": 0.0,
        # Merge hyps (e.g., lr0/lrf/mosaic/etc.)
        **hyp,
    }

    model = YOLO(args.model)
    # ⬇ The important fix
    model.train(**overrides)

    # Export best weights to ONNX and TorchScript (if present)
    best = "runs/train/yolo_stamp/weights/best.pt"
    if os.path.exists(best):
        YOLO(best).export(format="onnx", opset=13, simplify=True)
        YOLO(best).export(format="torchscript")
        print("Training complete and exports written")
    else:
        print("raining finished but best.pt not found yet. Check runs/train/yolo_stamp for results.")

if __name__ == "__main__":
    main()


from ultralytics import YOLO
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True)
args=ap.parse_args()
YOLO(args.weights).export(format="onnx", opset=13, simplify=True)
YOLO(args.weights).export(format="torchscript")

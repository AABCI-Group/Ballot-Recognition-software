
import argparse
from pathlib import Path
import subprocess

ap=argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--out", default="outputs")
args=ap.parse_args()

imgs = sorted([p for p in Path(args.images).glob("**/*") if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff"}])
for p in imgs:
    subprocess.run(["python","-m","src.infer.predict","--weights",args.weights,"--images",str(p),"--out_dir",args.out], check=True)
print("Batch complete.")

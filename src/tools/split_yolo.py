
import argparse, shutil, random
from pathlib import Path
random.seed(123)

def collect(img_dir):
    return sorted([p for p in Path(img_dir).glob("*.png")])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels_det", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    imgs = collect(args.images)
    random.shuffle(imgs)
    n = len(imgs)
    n_test = int(n*args.test)
    n_val  = int(n*args.val)
    test = imgs[:n_test]; val = imgs[n_test:n_test+n_val]; train = imgs[n_test+n_val:]

    for split, items in [("train",train), ("val",val), ("test",test)]:
        (Path(args.out_dir)/split/"images").mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir)/split/"labels").mkdir(parents=True, exist_ok=True)
        for im in items:
            lb = Path(args.labels_det)/ (im.stem + ".txt")
            (Path(args.out_dir)/split/"images"/im.name).write_bytes(im.read_bytes())
            dst = Path(args.out_dir)/split/"labels"/(im.stem+".txt")
            if lb.exists():
                dst.write_bytes(lb.read_bytes())
            else:
                dst.write_text("")
if __name__ == "__main__":
    main()

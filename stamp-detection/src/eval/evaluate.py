
import argparse, json
from pathlib import Path
#
ap=argparse.ArgumentParser()
ap.add_argument("--pred_manifest", required=True)
ap.add_argument("--labels_dir", required=True)
args=ap.parse_args()

def has_gt(label_file: Path) -> bool:
    return label_file.exists() and label_file.read_text().strip()!=""

data = json.load(open(args.pred_manifest))
TP=FP=TN=FN=0
for rec in data:
    p = Path(rec["image"])
    gt = has_gt(Path(args.labels_dir)/(p.stem+".txt"))
    pred = rec["decision"]
    if gt and pred=="VALID STAMP": TP+=1
    elif gt and pred!="VALID STAMP": FN+=1
    elif (not gt) and pred=="VALID STAMP": FP+=1
    else: TN+=1

print("TP FP TN FN:", TP, FP, TN, FN)
precision = TP/(TP+FP+1e-9)
recall = TP/(TP+FN+1e-9)
print("Precision:", precision)
print("Recall:", recall)
print("FPR:", FP/(FP+TN+1e-9))

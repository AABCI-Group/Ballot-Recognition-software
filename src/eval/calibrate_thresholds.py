
import json, numpy as np

TARGET_FPR = 0.002  # 0.2%
#
manifest = json.load(open("outputs/inference_manifest.json"))
scores=[]; ys=[]
for r in manifest:
    scores.append(r["score"])
    ys.append(1 if r.get("bbox") else 0)

scores=np.array(scores); ys=np.array(ys)

thrs = np.linspace(0.2, 0.9, 50)
neg = scores[ys==0]
pos = scores[ys==1]

best=0.6
for t in thrs:
    fpr = (neg>=t).mean() if neg.size else 0.0
    if fpr<=TARGET_FPR:
        best=t; break
print("Recommended score_valid:", best)

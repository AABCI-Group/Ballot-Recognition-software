
import argparse, yaml, cv2, json
from ultralytics import YOLO
from pathlib import Path
from src.preprocess.deskew import deskew_bgr, normalize
from src.infer.geometry_checks import GeoConfig, passes_geometry
from src.infer.postprocess import decide, Thresholds

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--images", required=True, help="file or directory")
ap.add_argument("--val_rules", default="configs/val_rules.yaml")
ap.add_argument("--template", default=None, help="optional template PNG of stamp (color)")
ap.add_argument("--out_dir", default="outputs")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.val_rules))
th = Thresholds(cfg["score_valid"], cfg["score_review"])
geo = GeoConfig(cfg["min_area_frac"], cfg["max_area_frac"], cfg["min_circularity"], cfg["min_ellipse_edge_pct"], cfg["min_template_ncc"])

template = cv2.imread(args.template) if args.template else None

model = YOLO(args.weights)

paths = [Path(args.images)]
if Path(args.images).is_dir():
    paths = sorted([p for p in Path(args.images).glob("**/*") if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff"}])

out_dir = Path(args.out_dir); (out_dir/"crops").mkdir(parents=True, exist_ok=True)

manifest = []
for p in paths:
    im = cv2.imread(str(p))
    if im is None: continue
    im, skew = deskew_bgr(im)
    im = normalize(im)
    #res = model.predict(im, imgsz=1024, conf=0.01, iou=0.5, verbose=False, device=0)[0]

    res = model.predict(im, imgsz=1024, conf=0.1, iou=0.5, verbose=False, device=0)[0]
    H,W = im.shape[:2]
    best_score = 0.0; best_bbox=None; feats={}; geo_ok=False
    for b in res.boxes:
        score = float(b.conf.cpu().item())
        x0,y0,x1,y1 = map(int, b.xyxy.cpu().numpy().ravel())
        crop = im[max(0,y0):min(H,y1), max(0,x0):min(W,x1)]
        ok, f = passes_geometry(crop, im.shape, (x0,y0,x1,y1), geo, template)
        #ok, f = True, {}  # BYPASS geometry checks for debugging

        if score>best_score:
            best_score=score; best_bbox=(x0,y0,x1,y1); feats=f; geo_ok=ok
    print(p.name, "best_score:", best_score, "geo_ok:", geo_ok, "feats:", feats)
    decision = decide(best_score, geo_ok, feats, best_bbox, th)

    rec = {
        "image": p.as_posix(),
        "decision": decision.label,
        "score": decision.conf,
        "bbox": decision.bbox,
        "features": decision.features,
        "skew_deg": skew
    }
    manifest.append(rec)

    vis = im.copy()
    if decision.bbox:
        x0,y0,x1,y1=decision.bbox
        color = (0,255,0) if decision.label=="VALID STAMP" else (0,255,255) if decision.label=="REVIEW REQUIRED" else (0,0,255)
        import cv2 as _cv2
        _cv2.rectangle(vis,(x0,y0),(x1,y1),color,2)
    cv2.putText(vis, f"{decision.label} ({decision.conf:.2f})", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    cv2.putText(vis, f"{decision.label} ({decision.conf:.2f})", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir/(p.stem+"_annot.png")), vis)
    if decision.bbox:
        x0,y0,x1,y1=decision.bbox
        crop = im[y0:y1, x0:x1]
        cv2.imwrite(str(out_dir/"crops"/(p.stem+"_crop.png")), crop)

with open(out_dir/"inference_manifest.json","w") as f:
    json.dump(manifest, f, indent=2)
print(f"Done. Wrote {len(manifest)} records to {out_dir}/inference_manifest.json")

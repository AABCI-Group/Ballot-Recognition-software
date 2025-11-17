
import json, hashlib, time, platform, subprocess
from pathlib import Path

def git_rev():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
    except Exception:
        return "nogit"

def summarize(inf_manifest, model_path, out_file="audit_log.json"):
    data = json.load(open(inf_manifest))
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.platform(),
        "python": platform.python_version(),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "git_rev": git_rev(),
        "inference": data
    }
    with open(out_file,"w") as f:
        json.dump(record, f, indent=2)
    print("Audit written:", out_file)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--weights", required=True)
    a=ap.parse_args(); summarize(a.manifest, a.weights)

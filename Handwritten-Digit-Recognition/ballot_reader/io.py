import os
import cv2
import json
import csv
import numpy as np
import urllib.request
from typing import Any, Dict, List, Optional, Union

def read_image(path_or_url: str) -> Optional[np.ndarray]:
    """Read an image from a local path or a URL."""
    if path_or_url.startswith(('http://', 'https://')):
        try:
            with urllib.request.urlopen(path_or_url) as resp:
                data = resp.read()
            img_array = np.asarray(bytearray(data), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[ERROR] Failed to read image from URL: {e}")
            return None
    else:
        if not os.path.exists(path_or_url):
            print(f"[ERROR] Image path does not exist: {path_or_url}")
            return None
        return cv2.imread(path_or_url)

def write_json(path: str, obj: Any) -> None:
    """Write an object to a JSON file, handling numpy types."""
    def _to_jsonable(x):
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, set):
            return [_to_jsonable(v) for v in sorted(x)]
        return x

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False)

def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write a list of dicts to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def safe_mkdir(path: str) -> None:
    """Safely create a directory if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)

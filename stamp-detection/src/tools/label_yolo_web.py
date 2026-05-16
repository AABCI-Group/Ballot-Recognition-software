#!/usr/bin/env python3
"""Small local browser labeler for one-class YOLO stamp boxes."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stamp YOLO Labeler</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d0d5dd;
      --accent: #0f766e;
      --danger: #b42318;
      --warn: #b54708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .controls, .actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 14px;
      cursor: pointer;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    button.danger { background: var(--danger); border-color: var(--danger); color: white; }
    button.warn { background: var(--warn); border-color: var(--warn); color: white; }
    button:disabled { opacity: 0.45; cursor: default; }
    select {
      min-width: 260px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
    }
    .meta {
      min-width: 0;
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    main {
      min-height: 0;
      display: grid;
      grid-template-columns: 280px 1fr;
    }
    aside {
      min-height: 0;
      border-right: 1px solid var(--line);
      background: #fff;
      overflow: auto;
    }
    .row {
      width: 100%;
      border: 0;
      border-bottom: 1px solid #eef0f3;
      border-radius: 0;
      padding: 9px 10px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      text-align: left;
      background: #fff;
    }
    .row.active { background: #e6f4f1; }
    .row span:first-child {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .badge { color: var(--muted); font-size: 12px; }
    .badge.box { color: var(--accent); }
    .badge.empty { color: var(--warn); }
    .stage {
      min-width: 0;
      min-height: 0;
      padding: 12px;
      display: grid;
      place-items: center;
      overflow: auto;
    }
    canvas {
      background: #fff;
      border: 1px solid var(--line);
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
      cursor: crosshair;
      max-width: 100%;
      max-height: calc(100vh - 86px);
    }
    .status { font-weight: 700; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { display: none; }
      header { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="controls">
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <select id="imageSelect"></select>
    </div>
    <div class="meta">
      <span id="counter"></span>
      <span id="filename"></span>
      <span id="status" class="status"></span>
    </div>
    <div class="actions">
      <button id="clearBtn">Clear</button>
      <button id="emptyBtn" class="warn">No Stamp</button>
      <button id="saveBtn" class="primary">Save Box</button>
    </div>
  </header>
  <main>
    <aside id="list"></aside>
    <section class="stage">
      <canvas id="canvas"></canvas>
    </section>
  </main>
  <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const imageSelect = document.getElementById('imageSelect');
    const listEl = document.getElementById('list');
    const filenameEl = document.getElementById('filename');
    const counterEl = document.getElementById('counter');
    const statusEl = document.getElementById('status');
    const img = new Image();

    let images = [];
    let index = 0;
    let currentBox = null;
    let dragStart = null;
    let draftBox = null;
    let scale = 1;

    function normalizedToCanvas(box) {
      return {
        x: (box.cx - box.w / 2) * canvas.width,
        y: (box.cy - box.h / 2) * canvas.height,
        w: box.w * canvas.width,
        h: box.h * canvas.height
      };
    }

    function canvasToNormalized(box) {
      const x = Math.max(0, Math.min(canvas.width, box.x));
      const y = Math.max(0, Math.min(canvas.height, box.y));
      const w = Math.max(0, Math.min(canvas.width - x, box.w));
      const h = Math.max(0, Math.min(canvas.height - y, box.h));
      return {
        cx: (x + w / 2) / canvas.width,
        cy: (y + h / 2) / canvas.height,
        w: w / canvas.width,
        h: h / canvas.height
      };
    }

    function displayBox() {
      if (draftBox) return draftBox;
      if (currentBox) return normalizedToCanvas(currentBox);
      return null;
    }

    function draw() {
      if (!img.complete || !img.naturalWidth) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const box = displayBox();
      if (box && box.w > 2 && box.h > 2) {
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#e11d48';
        ctx.fillStyle = 'rgba(225, 29, 72, 0.12)';
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeRect(box.x, box.y, box.w, box.h);
      }
    }

    function canvasPoint(ev) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (ev.clientX - rect.left) * (canvas.width / rect.width),
        y: (ev.clientY - rect.top) * (canvas.height / rect.height)
      };
    }

    function fitCanvas() {
      const maxW = Math.max(320, window.innerWidth - 330);
      const maxH = Math.max(320, window.innerHeight - 92);
      const s = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
      scale = s > 0 ? s : 1;
      canvas.width = Math.round(img.naturalWidth * scale);
      canvas.height = Math.round(img.naturalHeight * scale);
      draw();
    }

    async function api(path, options) {
      const res = await fetch(path, options);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function loadList() {
      images = await api('/api/images');
      imageSelect.innerHTML = '';
      images.forEach((item, i) => {
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = item.name;
        imageSelect.appendChild(opt);
      });
      renderList();
      if (images.length) loadImage(0);
    }

    function labelText(item) {
      if (!item.labeled) return 'new';
      return item.empty ? 'empty' : 'box';
    }

    function renderList() {
      listEl.innerHTML = '';
      images.forEach((item, i) => {
        const btn = document.createElement('button');
        btn.className = 'row' + (i === index ? ' active' : '');
        btn.innerHTML = `<span>${item.name}</span><span class="badge ${labelText(item)}">${labelText(item)}</span>`;
        btn.onclick = () => loadImage(i);
        listEl.appendChild(btn);
      });
    }

    async function loadImage(i) {
      if (!images.length) return;
      index = Math.max(0, Math.min(images.length - 1, i));
      imageSelect.value = String(index);
      const item = images[index];
      filenameEl.textContent = item.name;
      counterEl.textContent = `${index + 1}/${images.length}`;
      statusEl.textContent = labelText(item);
      currentBox = null;
      draftBox = null;
      const label = await api(`/api/label?name=${encodeURIComponent(item.name)}`);
      if (label.box) currentBox = label.box;
      img.onload = fitCanvas;
      img.src = `/api/image?name=${encodeURIComponent(item.name)}&v=${Date.now()}`;
      renderList();
    }

    async function save(box) {
      const item = images[index];
      await api('/api/label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: item.name, box })
      });
      images = await api('/api/images');
      await loadImage(index);
    }

    canvas.addEventListener('mousedown', (ev) => {
      dragStart = canvasPoint(ev);
      draftBox = { x: dragStart.x, y: dragStart.y, w: 0, h: 0 };
      draw();
    });

    canvas.addEventListener('mousemove', (ev) => {
      if (!dragStart) return;
      const p = canvasPoint(ev);
      draftBox = {
        x: Math.min(dragStart.x, p.x),
        y: Math.min(dragStart.y, p.y),
        w: Math.abs(p.x - dragStart.x),
        h: Math.abs(p.y - dragStart.y)
      };
      draw();
    });

    window.addEventListener('mouseup', () => {
      if (!dragStart) return;
      dragStart = null;
      if (draftBox && draftBox.w > 2 && draftBox.h > 2) {
        currentBox = canvasToNormalized(draftBox);
      }
      draftBox = null;
      draw();
    });

    document.getElementById('prevBtn').onclick = () => loadImage(index - 1);
    document.getElementById('nextBtn').onclick = () => loadImage(index + 1);
    document.getElementById('clearBtn').onclick = () => { currentBox = null; draftBox = null; draw(); };
    document.getElementById('emptyBtn').onclick = () => save(null);
    document.getElementById('saveBtn').onclick = () => {
      if (!currentBox) {
        alert('Draw a box first, or use No Stamp.');
        return;
      }
      save(currentBox);
    };
    imageSelect.onchange = () => loadImage(Number(imageSelect.value));

    window.addEventListener('resize', fitCanvas);
    window.addEventListener('keydown', (ev) => {
      const key = ev.key.toLowerCase();
      if (key === 'a') loadImage(index - 1);
      if (key === 'd') loadImage(index + 1);
      if (key === 's') document.getElementById('saveBtn').click();
      if (key === 'n') document.getElementById('emptyBtn').click();
      if (key === 'delete' || key === 'backspace') document.getElementById('clearBtn').click();
    });

    loadList().catch(err => alert(err.message));
  </script>
</body>
</html>
"""


class LabelServer(BaseHTTPRequestHandler):
    images_dir: Path
    labels_dir: Path
    files: list[Path]

    def _json(self, status: int, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, status: int, text: str):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _image_by_name(self, name: str) -> Path | None:
        for path in self.files:
            if path.name == name:
                return path
        return None

    def _label_path(self, name: str) -> Path:
        return self.labels_dir / (Path(name).stem + ".txt")

    def _read_label(self, name: str):
        path = self._label_path(name)
        if not path.exists():
            return {"labeled": False, "empty": False, "box": None}
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {"labeled": True, "empty": True, "box": None}
        parts = text.split()
        if len(parts) < 5:
            return {"labeled": True, "empty": False, "box": None}
        return {
            "labeled": True,
            "empty": False,
            "box": {
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            },
        }

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/images":
            rows = []
            for path in self.files:
                label = self._read_label(path.name)
                rows.append({"name": path.name, **label})
            self._json(200, rows)
            return

        if parsed.path == "/api/label":
            name = parse_qs(parsed.query).get("name", [""])[0]
            if not self._image_by_name(name):
                self._text(404, "Unknown image")
                return
            self._json(200, self._read_label(name))
            return

        if parsed.path == "/api/image":
            name = parse_qs(parsed.query).get("name", [""])[0]
            path = self._image_by_name(name)
            if not path:
                self._text(404, "Unknown image")
                return
            data = path.read_bytes()
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self._text(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/label":
            self._text(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            name = payload["name"]
        except Exception:
            self._text(400, "Invalid JSON")
            return
        if not self._image_by_name(name):
            self._text(404, "Unknown image")
            return

        self.labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = self._label_path(name)
        box = payload.get("box")
        if box is None:
            label_path.write_text("", encoding="utf-8")
            self._json(200, {"ok": True, "empty": True})
            return

        try:
            cx = max(0.0, min(1.0, float(box["cx"])))
            cy = max(0.0, min(1.0, float(box["cy"])))
            bw = max(0.0, min(1.0, float(box["w"])))
            bh = max(0.0, min(1.0, float(box["h"])))
        except Exception:
            self._text(400, "Invalid box")
            return
        if bw <= 0 or bh <= 0:
            self._text(400, "Box width and height must be positive")
            return

        label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
        self._json(200, {"ok": True, "empty": False})

    def log_message(self, fmt, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder containing ballot images")
    parser.add_argument("--labels", required=True, help="Folder where YOLO .txt labels are written")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images).resolve()
    labels_dir = Path(args.labels).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not files:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    LabelServer.images_dir = images_dir
    LabelServer.labels_dir = labels_dir
    LabelServer.files = files

    server = ThreadingHTTPServer((args.host, args.port), LabelServer)
    print(f"Labeling {len(files)} images")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Open: http://{args.host}:{args.port}")
    print("Stop with Ctrl+C")
    server.serve_forever()


if __name__ == "__main__":
    main()

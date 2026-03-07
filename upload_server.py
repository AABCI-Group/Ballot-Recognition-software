#!/usr/bin/env python3
"""
Minimal image upload server with:
- Nice inlined CSS + JS (drag/drop + progress + XHR upload)
- Persistent "current box location" (set via /set_box?box=BOX_A)
- /box (human view) and /box.json (machine)
- Serves uploaded images from /uploads/<filename>

Run:
  python3 server.py
Then open:
  http://localhost:8000/
"""

import cgi
import html
import json
import mimetypes
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOST = "0.0.0.0"
PORT = 8000

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATE_FILE = BASE_DIR / "current_box.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Persistent "current box" state
# ----------------------------
def get_current_box() -> str:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return (data.get("box_location") or "").strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""
    return ""


def set_current_box(box: str) -> None:
    box = (box or "").strip()
    payload = {
        "box_location": box,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------
# File helpers
# ----------------------------
def safe_filename(name: str) -> str:
    """
    Strip path traversal, NUL, etc. Keeps the basename only.
    """
    name = os.path.basename(name or "").strip().replace("\x00", "")
    return name or "upload"


def unique_path(directory: Path, filename: str) -> Path:
    """
    If filename exists, create filename_1.ext, filename_2.ext, ...
    """
    base, ext = os.path.splitext(filename)
    candidate = directory / filename
    i = 1
    while candidate.exists():
        candidate = directory / f"{base}_{i}{ext}"
        i += 1
    return candidate


# ----------------------------
# HTML pages
# ----------------------------
def upload_page(max_mb: int) -> str:
    # NOTE: This HTML uses NORMAL { } braces for CSS/JS.
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Image Upload</title>
  <style>
    :root {{
      --bg: #0b1220;
      --card: #101a31;
      --text: #e6eefc;
      --muted: #a6b3d1;
      --border: rgba(230, 238, 252, 0.15);
      --accent: #7aa2ff;
      --accent2: #67e8f9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: radial-gradient(1200px 600px at 20% 0%, rgba(122,162,255,.25), transparent 60%),
                  radial-gradient(900px 500px at 100% 10%, rgba(103,232,249,.18), transparent 55%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    .card {{
      width: min(560px, 100%);
      background: linear-gradient(180deg, rgba(16,26,49,0.95), rgba(16,26,49,0.85));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 18px 60px rgba(0,0,0,0.35);
    }}
    h1 {{
      font-size: 20px;
      margin: 0 0 10px;
      letter-spacing: .2px;
    }}
    p {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }}
    .meta {{
      margin: 0 0 14px;
      color: rgba(166,179,209,.95);
      font-size: 13px;
    }}
    .drop {{
      border: 2px dashed rgba(230,238,252,0.25);
      border-radius: 16px;
      padding: 22px;
      text-align: center;
      transition: 160ms ease;
      background: rgba(11,18,32,0.35);
      position: relative;
      overflow: hidden;
      user-select: none;
    }}
    .drop.dragover {{
      border-color: rgba(122,162,255,0.9);
      background: rgba(122,162,255,0.10);
      transform: translateY(-1px);
    }}
    .drop .hint {{
      font-size: 14px;
      color: var(--text);
      margin-bottom: 6px;
    }}
    .drop .subhint {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 14px;
    }}
    .row {{
      display: flex;
      gap: 10px;
      justify-content: center;
      flex-wrap: wrap;
    }}
    button {{
      appearance: none;
      border: 1px solid rgba(230,238,252,0.18);
      border-radius: 12px;
      padding: 10px 14px;
      background: rgba(230,238,252,0.06);
      color: var(--text);
      cursor: pointer;
      font-weight: 600;
      transition: 160ms ease;
    }}
    button.primary {{
      border-color: rgba(122,162,255,0.6);
      background: linear-gradient(180deg, rgba(122,162,255,0.35), rgba(122,162,255,0.18));
    }}
    button:hover {{
      transform: translateY(-1px);
      border-color: rgba(230,238,252,0.35);
    }}
    button:disabled {{
      opacity: 0.55;
      cursor: not-allowed;
      transform: none;
    }}
    input[type=file] {{
      display: none;
    }}
    .status {{
      margin-top: 14px;
      font-size: 13px;
      color: var(--muted);
      min-height: 18px;
      word-break: break-word;
    }}
    .progress {{
      margin-top: 10px;
      height: 10px;
      border-radius: 999px;
      background: rgba(230,238,252,0.10);
      overflow: hidden;
      display: none;
    }}
    .bar {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, rgba(122,162,255,0.9), rgba(103,232,249,0.85));
    }}
    .footer {{
      margin-top: 14px;
      font-size: 12px;
      color: rgba(166,179,209,0.85);
    }}
    a {{
      color: var(--accent2);
      text-decoration: none;
      font-weight: 700;
    }}
    code {{
      background: rgba(230,238,252,0.08);
      padding: 2px 6px;
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Upload an image</h1>
    <p>Drag & drop an image here, or choose a file. (Max size: {max_mb} MB)</p>

    <div class="meta">
      Current box location: <b id="currentBox">(loading...)</b>
      &nbsp;·&nbsp;
      <a href="/box">view</a>
    </div>

    <div id="drop" class="drop" role="button" tabindex="0" aria-label="Upload dropzone">
      <div class="hint">Drop your image here</div>
      <div class="subhint">PNG, JPG, GIF, WEBP, etc.</div>

      <div class="row">
        <button class="primary" id="chooseBtn" type="button">Choose file</button>
        <button id="uploadBtn" type="button" disabled>Upload</button>
      </div>

      <input id="fileInput" type="file" accept="image/*" />
      <div class="status" id="status"></div>

      <div class="progress" id="progress">
        <div class="bar" id="bar"></div>
      </div>
    </div>

    <div class="footer">
      Tip: Scan a QR code like <code>http://YOUR_LAPTOP_IP:8000/set_box?box=BOX_A</code> to set the box location.
    </div>
  </div>

<script>
(async () => {{
  const currentBoxEl = document.getElementById('currentBox');
  try {{
    const r = await fetch('/box.json', {{ cache: 'no-store' }});
    const j = await r.json();
    currentBoxEl.textContent = (j.box_location || '(not set)');
  }} catch {{
    currentBoxEl.textContent = '(unknown)';
  }}
}})();
</script>

<script>
(() => {{
  const drop = document.getElementById('drop');
  const fileInput = document.getElementById('fileInput');
  const chooseBtn = document.getElementById('chooseBtn');
  const uploadBtn = document.getElementById('uploadBtn');
  const statusEl = document.getElementById('status');
  const progressEl = document.getElementById('progress');
  const barEl = document.getElementById('bar');

  let selectedFile = null;

  function setStatus(msg) {{
    statusEl.textContent = msg || '';
  }}

  function setSelectedFile(file) {{
    selectedFile = file;
    if (file) {{
      uploadBtn.disabled = false;
      setStatus(`Selected: ${{
        file.name
      }} (${{
        Math.round(file.size / 1024)
      }} KB)`);
    }} else {{
      uploadBtn.disabled = true;
      setStatus('');
    }}
  }}

  function prevent(e) {{
    e.preventDefault();
    e.stopPropagation();
  }}

  ['dragenter','dragover'].forEach((evt) => {{
    drop.addEventListener(evt, (e) => {{
      prevent(e);
      drop.classList.add('dragover');
    }});
  }});

  ['dragleave','drop'].forEach((evt) => {{
    drop.addEventListener(evt, (e) => {{
      prevent(e);
      drop.classList.remove('dragover');
    }});
  }});

  drop.addEventListener('drop', (e) => {{
    const files = e.dataTransfer && e.dataTransfer.files;
    if (!files || !files.length) return;
    const file = files[0];
    if (!file.type || !file.type.startsWith('image/')) {{
      setStatus('Please drop an image file.');
      return;
    }}
    setSelectedFile(file);
  }});

  drop.addEventListener('click', () => fileInput.click());
  drop.addEventListener('keydown', (e) => {{
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  }});

  chooseBtn.addEventListener('click', (e) => {{
    e.stopPropagation();
    fileInput.click();
  }});

  fileInput.addEventListener('change', () => {{
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    if (!file.type || !file.type.startsWith('image/')) {{
      setStatus('Please choose an image file.');
      setSelectedFile(null);
      return;
    }}
    setSelectedFile(file);
  }});

  uploadBtn.addEventListener('click', async (e) => {{
    e.stopPropagation();
    if (!selectedFile) return;

    const form = new FormData();
    form.append('file', selectedFile);

    setStatus('Uploading...');
    progressEl.style.display = 'block';
    barEl.style.width = '0%';

    await new Promise((resolve, reject) => {{
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/upload', true);

      xhr.upload.onprogress = (evt) => {{
        if (evt.lengthComputable) {{
          const pct = Math.round((evt.loaded / evt.total) * 100);
          barEl.style.width = pct + '%';
        }}
      }};

      xhr.onload = () => {{
        if (xhr.status >= 200 && xhr.status < 300) {{
          // Replace the whole document with the server response (success page)
          document.open();
          document.write(xhr.responseText);
          document.close();
          resolve();
        }} else {{
          reject(new Error(xhr.responseText || 'Upload failed'));
        }}
      }};

      xhr.onerror = () => reject(new Error('Network error'));
      xhr.send(form);
    }}).catch((err) => {{
      setStatus(err.message || 'Upload failed');
      progressEl.style.display = 'none';
    }});
  }});
}})();
</script>
</body>
</html>
"""


def success_page(saved_as: str) -> str:
    escaped_name = html.escape(saved_as)
    img_url = f"/uploads/{escaped_name}"
    current_box = html.escape(get_current_box() or "(not set)")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Uploaded</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background: #0b1220;
      color: #e6eefc;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    .card {{
      width: min(560px, 100%);
      background: #101a31;
      border: 1px solid rgba(230, 238, 252, 0.15);
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 18px 60px rgba(0,0,0,0.35);
    }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    p {{ margin: 0 0 12px; color: rgba(166,179,209,0.95); }}
    .row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    a.button {{
      display: inline-block;
      padding: 10px 14px;
      border-radius: 12px;
      border: 1px solid rgba(230,238,252,0.18);
      background: rgba(230,238,252,0.06);
      color: #e6eefc;
      text-decoration: none;
      font-weight: 700;
    }}
    a.button.primary {{
      border-color: rgba(122,162,255,0.6);
      background: linear-gradient(180deg, rgba(122,162,255,0.35), rgba(122,162,255,0.18));
    }}
    a.link {{
      color: #67e8f9;
      text-decoration: none;
      font-weight: 700;
    }}
    code {{
      background: rgba(230,238,252,0.08);
      padding: 2px 6px;
      border-radius: 8px;
    }}
    .preview {{
      margin: 14px 0 16px;
      border: 1px solid rgba(230, 238, 252, 0.15);
      border-radius: 14px;
      overflow: hidden;
      background: rgba(11,18,32,0.35);
    }}
    .preview img {{
      display: block;
      width: 100%;
      height: auto;
      max-height: 360px;
      object-fit: contain;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Uploaded ✅</h1>
    <p>Saved as: <code>{escaped_name}</code></p>
    <p>Current box location: <b>{current_box}</b> · <a class="link" href="/box">view</a></p>

    <div class="preview">
      <img src="{img_url}" alt="Uploaded image: {escaped_name}" />
    </div>

    <div class="row">
      <a class="button primary" href="/">Upload another</a>
      <a class="button" href="{img_url}">Open image</a>
    </div>
  </div>
</body>
</html>"""


def error_page(message: str) -> str:
    msg = html.escape(message)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Error</title>
  <style>
    body {{
      margin:0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background:#0b1220;
      color:#e6eefc;
      min-height:100vh;
      display:grid;
      place-items:center;
      padding:24px;
    }}
    .card {{
      width:min(560px,100%);
      background:#101a31;
      border:1px solid rgba(230,238,252,0.15);
      border-radius:18px;
      padding:22px;
    }}
    p {{ color: rgba(166,179,209,0.95); }}
    a {{ color:#67e8f9; text-decoration:none; font-weight:700; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Upload failed</h1>
    <p>{msg}</p>
    <p><a href="/">← Back</a></p>
  </div>
</body>
</html>"""


def set_box_page(box: str) -> str:
    safe = html.escape(box)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Box set</title>
  <style>
    body {{
      margin:0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background:#0b1220;
      color:#e6eefc;
      min-height:100vh;
      display:grid;
      place-items:center;
      padding:24px;
    }}
    .card {{
      width:min(560px,100%);
      background:#101a31;
      border:1px solid rgba(230,238,252,0.15);
      border-radius:18px;
      padding:22px;
    }}
    p {{ color: rgba(166,179,209,0.95); }}
    a {{ color:#67e8f9; text-decoration:none; font-weight:700; }}
    code {{ background: rgba(230,238,252,0.08); padding:2px 6px; border-radius:8px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>✅ Box location set</h1>
    <p>Current box is now: <code>{safe}</code></p>
    <p>This stays active until you scan a different QR code.</p>
    <p><a href="/">← Back</a> · <a href="/box">View current box</a></p>
  </div>
</body>
</html>"""


def box_page() -> str:
    current = html.escape(get_current_box() or "(not set)")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Current box</title>
  <style>
    body {{
      margin:0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background:#0b1220;
      color:#e6eefc;
      min-height:100vh;
      display:grid;
      place-items:center;
      padding:24px;
    }}
    .card {{
      width:min(560px,100%);
      background:#101a31;
      border:1px solid rgba(230,238,252,0.15);
      border-radius:18px;
      padding:22px;
    }}
    p {{ color: rgba(166,179,209,0.95); }}
    a {{ color:#67e8f9; text-decoration:none; font-weight:700; }}
    code {{ background: rgba(230,238,252,0.08); padding:2px 6px; border-radius:8px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Current box</h1>
    <p><code>{current}</code></p>
    <p><a href="/">← Back</a></p>
  </div>
</body>
</html>"""


# ----------------------------
# HTTP handler
# ----------------------------
class UploadHandler(BaseHTTPRequestHandler):
    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, text: str, status: int = 200) -> None:
        self._send_bytes(text.encode("utf-8"), "text/html; charset=utf-8", status=status)

    def _send_json(self, obj, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._send_html(upload_page(MAX_UPLOAD_BYTES // (1024 * 1024)))

        if path == "/healthz":
            return self._send_json({"ok": True, "service": "upload_server"}, status=200)

        if path == "/set_box":
            box = (qs.get("box", [""])[0] or "").strip()
            if not box:
                return self._send_html(error_page("Missing ?box=..."), status=400)
            set_current_box(box)
            return self._send_html(set_box_page(box), status=200)

        if path == "/box":
            return self._send_html(box_page(), status=200)

        if path == "/box.json":
            return self._send_json(
                {"box_location": get_current_box(), "state_file": str(STATE_FILE.name)},
                status=200,
            )

        if path.startswith("/uploads/"):
            rel = os.path.basename(path[len("/uploads/"):])  # prevent traversal
            file_path = UPLOAD_DIR / rel
            if not file_path.is_file():
                return self.send_error(404, "Not found")

            ctype, _ = mimetypes.guess_type(str(file_path))
            ctype = ctype or "application/octet-stream"
            try:
                data = file_path.read_bytes()
            except OSError:
                return self.send_error(500, "Failed to read file")
            return self._send_bytes(data, ctype, status=200)

        return self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            return self.send_error(404, "Not found")

        # Size guard
        length_header = self.headers.get("Content-Length")
        if not length_header:
            return self._send_html(error_page("Missing Content-Length"), status=411)

        try:
            length = int(length_header)
        except ValueError:
            return self._send_html(error_page("Invalid Content-Length"), status=400)

        if length > MAX_UPLOAD_BYTES:
            return self._send_html(
                error_page(f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)."),
                status=413,
            )

        if self.headers.get_content_type() != "multipart/form-data":
            return self._send_html(error_page("Expected multipart/form-data"), status=400)

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )

            if "file" not in form:
                return self._send_html(error_page("No file field found."), status=400)

            field = form["file"]
            if not getattr(field, "file", None) or not field.filename:
                return self._send_html(error_page("No file selected."), status=400)

            upload_type = field.type or ""
            if not upload_type.startswith("image/"):
                return self._send_html(error_page("Only image uploads are allowed."), status=400)

            filename = safe_filename(field.filename)
            save_path = unique_path(UPLOAD_DIR, filename)

            with open(save_path, "wb") as out:
                while True:
                    chunk = field.file.read(64 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

            return self._send_html(success_page(save_path.name), status=200)

        except Exception as e:
            return self._send_html(error_page(f"Server error: {e}"), status=500)


class QuietUploadHandler(UploadHandler):
    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            return


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        # Suppress traceback spam for expected disconnects
        pass


def main() -> None:
    server = QuietThreadingHTTPServer((HOST, PORT), QuietUploadHandler)
    print(f"Server running on http://{HOST}:{PORT} ...")
    print("QR set box endpoint: /set_box?box=YOUR_BOX_ID")
    server.serve_forever()


if __name__ == "__main__":
    main()

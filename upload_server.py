from http.server import HTTPServer, BaseHTTPRequestHandler
import os

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SimpleUploadServer(BaseHTTPRequestHandler):
    def do_GET(self):
        html = """
        <html>
        <body>
            <h2>Upload image</h2>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" />
                <input type="submit" value="Upload" />
            </form>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        content_type = self.headers.get('Content-Type')
        boundary = content_type.split("=")[1].encode()
        length = int(self.headers.get('Content-Length'))
        body = self.rfile.read(length)

        # Split multipart form-data
        parts = body.split(boundary)
        for part in parts:
            if b"Content-Disposition" in part:
                header, file_data = part.split(b"\r\n\r\n", 1)
                filename_marker = b'filename="'
                start = header.find(filename_marker) + len(filename_marker)
                end = header.find(b'"', start)
                filename = header[start:end].decode()

                # Remove ending boundary dashes
                file_data = file_data.rstrip(b"\r\n--")

                # Save file
                with open(os.path.join(UPLOAD_DIR, filename), "wb") as f:
                    f.write(file_data)

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Uploaded!")
                return

server = HTTPServer(("0.0.0.0", 8000), SimpleUploadServer)
print("Server running on port 8000...")
server.serve_forever()

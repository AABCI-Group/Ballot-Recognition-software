import qrcode

LAPTOP_IP = "10.103.178.74"
PORT = 8000

boxes = ["Behy 83", "St Oliver Plunkett 82", "Ardagh National School 81"]

for b in boxes:
    url = f"http://{LAPTOP_IP}:{PORT}/set_box?box={b}"
    img = qrcode.make(url)
    img.save(f"qr_{b}.png")
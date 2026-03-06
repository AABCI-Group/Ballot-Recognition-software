from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import os
    
roman_map = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    # extend up to N as needed
}


def make_roman_image(text, img_size=28):
    from PIL import Image, ImageDraw, ImageFont
    import random
    img = Image.new("L", (img_size, img_size), color=255)  # white background
    draw = ImageDraw.Draw(img)

    # Use a basic font (change path to any .ttf font available)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    # --- FIXED: use textbbox instead of textsize ---
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # for older Pillow versions
        w, h = draw.textsize(text, font=font)
    # ---------------------------------------------

    # Center text
    x = (img_size - w) // 2
    y = (img_size - h) // 2
    draw.text((x, y), text, fill=0, font=font)

    # Slight random rotation for variation
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, expand=False, fillcolor=255)

    return np.array(img)


roman_imgs = []
roman_labels = []

SAMPLES_PER_CLASS = 100   # tune this

os.makedirs("roman_samples", exist_ok=True)

for value, text in roman_map.items():
    if value > 10:
        continue
    for i in range(5):
        arr = make_roman_image(text)
        img = Image.fromarray(arr)
        img.save(f"roman_samples/{value}_{i}.png")

roman_imgs = np.stack(roman_imgs, axis=0)
roman_labels = np.array(roman_labels, dtype=np.int64)

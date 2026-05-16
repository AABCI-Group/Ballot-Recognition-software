#!/usr/bin/env python3
"""Generate print-master ballots for real-world stamp detector data collection."""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.lib.pagesizes import A3
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BLANK_DIR = ROOT / "data" / "SampleBallots" / "Original-Ballot"
DEFAULT_STAMP_PATHS = [ROOT / "assets" / "Stamp.png", ROOT / "assets" / "stamp_2.png"]

OUTPUT_IMAGES = THIS_DIR / "images"
OUTPUT_PDF = THIS_DIR / "ballots_a3_print.pdf"
OUTPUT_MANIFEST = THIS_DIR / "manifest.csv"

BALLOT_HEIGHT_PX = 4800
PDF_MARGIN_PT = 36

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Placement:
    name: str
    count: int


PLACEMENTS = [
    Placement("top_instruction_text", 12),
    Placement("candidate_name_party_text", 8),
    Placement("party_logo_or_candidate_photo", 6),
    Placement("blank_white_area", 8),
    Placement("near_page_edge_or_margin", 4),
    Placement("partly_off_ballot_area", 2),
]

APPEARANCES = [
    "clear_dark",
    "normal_grey",
    "faint_light",
    "slightly_rotated",
    "overlapping_dark_text",
]


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def resize_ballot(im: Image.Image, target_height: int = BALLOT_HEIGHT_PX) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGBA")
    scale = target_height / im.height
    target_width = round(im.width * scale)
    return im.resize((target_width, target_height), Image.Resampling.LANCZOS)


def make_paper_variant(base: Image.Image, rng: random.Random) -> Image.Image:
    im = base.convert("RGBA")
    rgb = im.convert("RGB")

    # Slight print/scan unevenness while keeping the ballot clean enough to print.
    brightness = rng.uniform(0.965, 1.025)
    contrast = rng.uniform(0.965, 1.035)
    lut = []
    for value in range(256):
        adjusted = (value - 128) * contrast + 128
        adjusted *= brightness
        lut.append(max(0, min(255, int(adjusted))))
    rgb = rgb.point(lut * 3)

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = im.size

    for _ in range(rng.randint(120, 240)):
        x = rng.randrange(w)
        y = rng.randrange(h)
        radius = rng.choice([1, 1, 1, 2, 2, 3])
        alpha = rng.randint(10, 42)
        shade = rng.randint(60, 170)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(shade, shade, shade, alpha))

    for _ in range(rng.randint(8, 18)):
        x = rng.randrange(w)
        y = rng.randrange(h)
        length = rng.randint(8, 34)
        color = rng.randint(70, 180)
        alpha = rng.randint(8, 28)
        draw.line((x, y, x + rng.randint(-8, 8), y + length), fill=(color, color, color, alpha), width=1)

    return Image.alpha_composite(rgb.convert("RGBA"), overlay)


def add_hard_negative_marks(im: Image.Image, rng: random.Random, heavier: bool = False) -> None:
    draw = ImageDraw.Draw(im)
    w, h = im.size
    font = load_font(rng.randint(42, 72))
    pen_colors = [
        (25, 43, 120, rng.randint(135, 210)),
        (24, 24, 24, rng.randint(115, 190)),
        (70, 70, 70, rng.randint(70, 130)),
    ]

    digit_count = rng.randint(4, 8) if heavier else rng.randint(1, 4)
    for i in range(digit_count):
        x = rng.randint(int(w * 0.045), int(w * 0.18))
        y = rng.randint(int(h * 0.22), int(h * 0.86))
        text = str((i % 5) + 1)
        scratch = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(scratch)
        sd.text((x, y), text, font=font, fill=rng.choice(pen_colors))
        scratch = scratch.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BICUBIC, center=(x, y), fillcolor=(0, 0, 0, 0))
        im.alpha_composite(scratch)

    mark_count = rng.randint(20, 48) if heavier else rng.randint(10, 30)
    for _ in range(mark_count):
        x = rng.randint(int(w * 0.03), int(w * 0.96))
        y = rng.randint(int(h * 0.08), int(h * 0.94))
        if rng.random() < 0.55:
            length = rng.randint(4, 28)
            draw.line(
                (x, y, x + rng.randint(-length, length), y + rng.randint(-length, length)),
                fill=rng.choice(pen_colors),
                width=rng.choice([1, 1, 2, 2, 3]),
            )
        else:
            r = rng.randint(2, 8)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=rng.choice(pen_colors))


def stamp_parameters_like_synth(
    stamp: Image.Image,
    ballot_height: int,
    rng: random.Random,
    appearance: str,
) -> tuple[float, float, float]:
    """Mirror src/tools/generate_synthetic.py stamp sizing and blending ranges."""
    target_frac = rng.uniform(0.030, 0.070)
    scale = (ballot_height * target_frac) / stamp.height
    angle = rng.uniform(-35.0, 35.0)
    opacity = rng.uniform(0.22, 0.95)

    if appearance == "clear_dark":
        opacity = rng.uniform(0.72, 0.95)
    elif appearance == "normal_grey":
        opacity = rng.uniform(0.45, 0.70)
    elif appearance == "faint_light":
        opacity = rng.uniform(0.22, 0.38)
    elif appearance == "slightly_rotated":
        angle = rng.choice([-1, 1]) * rng.uniform(8.0, 18.0)
    elif appearance == "overlapping_dark_text":
        opacity = rng.uniform(0.30, 0.55)

    return scale, angle, opacity


def transform_stamp_like_synth(stamp: Image.Image, scale: float, angle: float) -> Image.Image:
    stamp = stamp.convert("RGBA")
    new_w = max(1, int(stamp.width * scale))
    new_h = max(1, int(stamp.height * scale))
    resample = Image.Resampling.BOX if scale < 1 else Image.Resampling.BICUBIC
    resized = stamp.resize((new_w, new_h), resample)
    return resized.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))


def apply_stamp_like_synth(ballot: Image.Image, stamp: Image.Image, x: int, y: int, opacity: float) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """Alpha blend using the same math as apply_stamp in the synthetic generator."""
    out = ballot.convert("RGBA")
    stamp = stamp.convert("RGBA")
    bw, bh = out.size
    sw, sh = stamp.size

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + sw)
    y2 = min(bh, y + sh)
    if x2 <= x1 or y2 <= y1:
        return out, None

    sx = max(0, -x)
    sy = max(0, -y)
    ex = sx + (x2 - x1)
    ey = sy + (y2 - y1)

    roi = out.crop((x1, y1, x2, y2)).convert("RGBA")
    stamp_roi = stamp.crop((sx, sy, ex, ey)).convert("RGBA")
    alpha = stamp_roi.getchannel("A").point(lambda p: max(0, min(255, int(p * opacity))))
    stamp_roi.putalpha(alpha)

    blended = Image.alpha_composite(roi, stamp_roi)
    out.paste(blended, (x1, y1))

    bbox = alpha.point(lambda p: 255 if p > 10 else 0).getbbox()
    if bbox is None:
        return out, None
    bx0, by0, bx1, by1 = bbox
    return out, (x1 + bx0, y1 + by0, bx1 - bx0, by1 - by0)


def pick_position(im_size: tuple[int, int], stamp_size: tuple[int, int], placement: str, rng: random.Random) -> tuple[int, int]:
    w, h = im_size
    sw, sh = stamp_size

    def between(a: int, b: int) -> int:
        low = min(a, b)
        high = max(a, b)
        return rng.randint(low, high)

    regions = {
        "top_instruction_text": (0.16, 0.055, 0.82, 0.19),
        "candidate_name_party_text": (0.22, 0.25, 0.88, 0.82),
        "party_logo_or_candidate_photo": (0.62, 0.23, 0.96, 0.83),
        "blank_white_area": (0.18, 0.86, 0.82, 0.965),
        "near_page_edge_or_margin": (0.0, 0.06, 1.0, 0.94),
        "partly_off_ballot_area": (-0.12, 0.08, 0.92, 0.90),
    }

    if placement == "near_page_edge_or_margin":
        side = rng.choice(["left", "right", "top", "bottom"])
        if side == "left":
            return between(-sw // 8, int(w * 0.08)), between(int(h * 0.10), int(h * 0.86))
        if side == "right":
            return between(int(w * 0.88), w - sw // 2), between(int(h * 0.10), int(h * 0.86))
        if side == "top":
            return between(int(w * 0.12), int(w * 0.76)), between(-sh // 10, int(h * 0.06))
        return between(int(w * 0.12), int(w * 0.76)), between(int(h * 0.90), h - sh // 2)

    if placement == "partly_off_ballot_area":
        side = rng.choice(["left", "right", "bottom"])
        if side == "left":
            return between(-sw // 2, -sw // 5), between(int(h * 0.20), int(h * 0.70))
        if side == "right":
            return between(w - (sw * 3) // 5, w - sw // 4), between(int(h * 0.20), int(h * 0.70))
        return between(int(w * 0.18), int(w * 0.70)), between(h - (sh * 3) // 5, h - sh // 4)

    x1, y1, x2, y2 = regions[placement]
    left = int(w * x1)
    top = int(h * y1)
    right = int(w * x2) - sw
    bottom = int(h * y2) - sh
    if right < left:
        right = left
    if bottom < top:
        bottom = top
    return rng.randint(left, right), rng.randint(top, bottom)


def visible_bbox(position: tuple[int, int], stamp: Image.Image, ballot_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    x, y = position
    sw, sh = stamp.size
    bw, bh = ballot_size
    left = max(0, x)
    top = max(0, y)
    right = min(bw, x + sw)
    bottom = min(bh, y + sh)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def create_pdf(image_paths: Iterable[Path], pdf_path: Path) -> None:
    page_w, page_h = A3
    max_w = page_w - 2 * PDF_MARGIN_PT
    max_h = page_h - 2 * PDF_MARGIN_PT
    c = canvas.Canvas(str(pdf_path), pagesize=A3)

    for path in image_paths:
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min(max_w / iw, max_h / ih)
        draw_w = iw * scale
        draw_h = ih * scale
        x = (page_w - draw_w) / 2
        y = (page_h - draw_h) / 2
        c.drawImage(ImageReader(str(path)), x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
        c.showPage()

    c.save()


def verify_pdf(pdf_path: Path) -> tuple[int, float, float]:
    data = pdf_path.read_bytes()
    page_count = len(re.findall(rb"/Type\s*/Page\b", data))
    match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", data)
    if not match:
        raise ValueError(f"Could not find a PDF MediaBox in {pdf_path}")
    width = float(match.group(1))
    height = float(match.group(2))
    return page_count, width, height


def build_plan(seed: int) -> list[str | None]:
    plan: list[str | None] = []
    for placement in PLACEMENTS:
        plan.extend([placement.name] * placement.count)
    plan.extend([None] * 10)
    rng = random.Random(seed)
    rng.shuffle(plan)
    return plan


def clean_previous_outputs() -> None:
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    for pattern in ("ballot_*.png", "ballot_*.jpg", "ballot_*.jpeg"):
        for path in OUTPUT_IMAGES.glob(pattern):
            path.unlink()


def generate(seed: int = 20260503, count: int = 50) -> None:
    rng = random.Random(seed)
    blank_paths = image_files(DEFAULT_BLANK_DIR)
    stamp_paths = [p for p in DEFAULT_STAMP_PATHS if p.exists()]
    if not blank_paths:
        raise FileNotFoundError(f"No blank ballot images found in {DEFAULT_BLANK_DIR}")
    if not stamp_paths:
        raise FileNotFoundError(f"No stamp assets found in {ROOT / 'assets'}")

    plan = build_plan(seed)
    if count != 50:
        raise ValueError("This generator is configured for exactly 50 ballots to preserve the requested mix.")

    clean_previous_outputs()
    rows = []
    generated_paths = []

    for index, placement in enumerate(plan, start=1):
        blank_path = rng.choice(blank_paths)
        base = resize_ballot(Image.open(blank_path))
        ballot = make_paper_variant(base, rng)
        stamp_present = placement is not None
        appearance = ""
        bbox = None
        stamp_asset = ""
        stamp_x = stamp_y = stamp_w = stamp_h = ""
        stamp_scale = stamp_angle = stamp_opacity = ""

        add_hard_negative_marks(ballot, rng, heavier=not stamp_present)

        if stamp_present:
            appearance = APPEARANCES[(index - 1) % len(APPEARANCES)]
            if placement == "candidate_name_party_text" and rng.random() < 0.55:
                appearance = "overlapping_dark_text"
            stamp_asset_path = rng.choice(stamp_paths)
            stamp_asset = stamp_asset_path.name
            stamp_source = Image.open(stamp_asset_path).convert("RGBA")
            scale, angle, opacity = stamp_parameters_like_synth(stamp_source, ballot.height, rng, appearance)
            stamp = transform_stamp_like_synth(stamp_source, scale, angle)
            x, y = pick_position(ballot.size, stamp.size, placement, rng)
            ballot, bbox = apply_stamp_like_synth(ballot, stamp, x, y, opacity)
            stamp_x, stamp_y = x, y
            stamp_w, stamp_h = stamp.size
            stamp_scale = f"{scale:.6f}"
            stamp_angle = f"{angle:.3f}"
            stamp_opacity = f"{opacity:.3f}"

        output_path = OUTPUT_IMAGES / f"ballot_{index:03d}.jpg"
        ballot.convert("RGB").save(output_path, dpi=(300, 300), quality=95, subsampling=0)
        generated_paths.append(output_path)

        rows.append(
            {
                "filename": output_path.name,
                "blank_source": str(blank_path.relative_to(ROOT)),
                "stamp_present": "yes" if stamp_present else "no",
                "placement_category": placement or "none",
                "appearance": appearance,
                "stamp_asset": stamp_asset,
                "stamp_x": stamp_x,
                "stamp_y": stamp_y,
                "stamp_width": stamp_w,
                "stamp_height": stamp_h,
                "stamp_scale": stamp_scale,
                "stamp_angle": stamp_angle,
                "stamp_opacity": stamp_opacity,
                "visible_bbox_x": "" if bbox is None else bbox[0],
                "visible_bbox_y": "" if bbox is None else bbox[1],
                "visible_bbox_width": "" if bbox is None else bbox[2],
                "visible_bbox_height": "" if bbox is None else bbox[3],
                "image_width": ballot.width,
                "image_height": ballot.height,
                "seed": seed,
            }
        )

    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    create_pdf(generated_paths, OUTPUT_PDF)
    pages, width, height = verify_pdf(OUTPUT_PDF)

    print(f"Generated {len(generated_paths)} images in {OUTPUT_IMAGES}")
    print(f"Wrote manifest: {OUTPUT_MANIFEST}")
    print(f"Wrote PDF: {OUTPUT_PDF}")
    print(f"PDF pages: {pages}")
    print(f"PDF first page size: {width:.2f} x {height:.2f} pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 50 A3 print-master ballots with varied stamp examples.")
    parser.add_argument("--seed", type=int, default=20260503, help="Random seed for repeatable output.")
    args = parser.parse_args()
    generate(seed=args.seed)


if __name__ == "__main__":
    main()

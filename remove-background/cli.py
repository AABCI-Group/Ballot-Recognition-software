import argparse
import json
from pathlib import Path

from remove_background import VALID_EXTENSIONS, crop_ballot_paper


def _iter_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() in VALID_EXTENSIONS:
            yield input_path
        return

    if input_path.is_dir():
        for path in sorted(input_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop ballot paper and remove outer background")
    parser.add_argument("--input", required=True, help="Input image file or directory")
    parser.add_argument("--out_dir", required=True, help="Directory to write cropped image(s)")
    parser.add_argument("--debug_dir", required=True, help="Directory to write debug visuals")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    out_dir = Path(args.out_dir).resolve()
    debug_dir = Path(args.debug_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for image in _iter_images(input_path):
        output_path = out_dir / f"{image.stem}_ballot_crop.png"
        debug_path = debug_dir / image.stem
        result = crop_ballot_paper(str(image), str(output_path), str(debug_path))
        results.append(
            {
                "input_path": result.input_path,
                "output_path": result.output_path,
                "debug_dir": result.debug_dir,
                "bbox_xyxy": list(result.bbox),
                "used_fallback_full_image": result.used_fallback,
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

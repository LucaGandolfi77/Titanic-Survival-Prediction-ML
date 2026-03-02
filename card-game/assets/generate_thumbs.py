#!/usr/bin/env python3
"""
Generate thumbnails for images placed in assets/cards/
Creates two subfolders: thumbs/ (300x420) and mini/ (120x168)

Usage:
    python3 generate_thumbs.py

Requires: Pillow
"""
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("Error: Pillow is not installed. Install it with: pip install Pillow")
    exit(1)

BASE = Path(__file__).parent / "cards"
OUT_THUMBS = BASE / "thumbs"
OUT_MINI = BASE / "mini"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

THUMB_SIZE = (300, 420)  # 5:7
MINI_SIZE = (120, 168)

SKIP = {"missing_card.svg"}


def ensure_outdirs():
    OUT_THUMBS.mkdir(parents=True, exist_ok=True)
    OUT_MINI.mkdir(parents=True, exist_ok=True)


def process_image(path: Path, size, outdir: Path):
    try:
        with Image.open(path) as im:
            # convert transparencies to white background
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")

            thumb = ImageOps.fit(im, size, Image.LANCZOS)
            out_path = outdir / f"{path.stem}.jpg"
            thumb.save(out_path, format="JPEG", quality=85)
            return True
    except Exception as e:
        print(f"Failed to process {path}: {e}")
        return False


def main():
    if not BASE.exists():
        print(f"Directory {BASE} not found. Create assets/cards/ and add images.")
        return

    ensure_outdirs()
    imgs = [p for p in sorted(BASE.iterdir()) if p.is_file() and p.name not in SKIP and p.suffix.lower() in EXTS]
    if not imgs:
        print("No images found in assets/cards/")
        return

    print(f"Found {len(imgs)} images. Generating thumbnails...")
    ok = 0
    for p in imgs:
        r1 = process_image(p, THUMB_SIZE, OUT_THUMBS)
        r2 = process_image(p, MINI_SIZE, OUT_MINI)
        if r1 and r2:
            ok += 1
    print(f"Done. Processed {ok}/{len(imgs)} images.")
    print(f"Thumbs: {OUT_THUMBS}\nMini: {OUT_MINI}")


if __name__ == "__main__":
    main()

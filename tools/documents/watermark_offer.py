#!/usr/bin/env python3
"""Overlay a watermark without resampling or rewriting the evidence screenshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", default="闲鱼 草原小可汗")
    parser.add_argument("--font", type=Path, default=Path.home() / ".fonts/msyh.ttc")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a different output path; the source must be preserved.")
    if not args.font.is_file():
        parser.error("Chinese font not found; provide --font with a CJK font path.")

    with Image.open(args.input) as opened:
        source = opened.convert("RGB")
    width, height = source.size
    font = ImageFont.truetype(str(args.font), max(24, round(width * 0.037)))
    label_box = font.getbbox(args.label)
    tile = Image.new("RGBA", (label_box[2] + 20, label_box[3] - label_box[1] + 20))
    ImageDraw.Draw(tile).text((10, 10 - label_box[1]), args.label, font=font, fill=(76, 101, 146, 48))
    tile = tile.rotate(27, expand=True, resample=Image.Resampling.BICUBIC)
    overlay = Image.new("RGBA", source.size)
    for row, y in enumerate(range(-100, height, max(180, round(height / 7)))):
        offset = -(tile.width // 2) if row % 2 else 24
        for x in range(offset, width, tile.width + 85):
            overlay.alpha_composite(tile, (x, y))
    small_font = ImageFont.truetype(str(args.font), max(22, round(width * 0.028)))
    draw = ImageDraw.Draw(overlay)
    draw.text((width - 34, height - 36), args.label, font=small_font, anchor="rs", fill=(76, 101, 146, 96))

    # Keep every dark source pixel unchanged, including the existing black redactions.
    red, green, blue = source.split()
    brightest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    allowed = brightest.point(lambda value: 0 if value <= 80 else 255)
    overlay.putalpha(ImageChops.multiply(overlay.getchannel("A"), allowed))
    output = Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")
    delta = ImageChops.difference(source, output)
    if ImageChops.multiply(delta, Image.merge("RGB", (ImageChops.invert(allowed),) * 3)).getbbox():
        raise RuntimeError("Existing dark pixels changed.")
    outside = overlay.getchannel("A").point(lambda value: 255 if value == 0 else 0)
    if ImageChops.multiply(delta, Image.merge("RGB", (outside,) * 3)).getbbox():
        raise RuntimeError("Pixels outside the watermark changed.")
    if output.size != source.size or not delta.getbbox():
        raise RuntimeError("Unexpected output size or empty watermark.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output, format="PNG", optimize=True)
    print(f"Saved {args.output} ({width} x {height}); source, dark redactions, and pixels outside the watermark verified.")


if __name__ == "__main__":
    main()

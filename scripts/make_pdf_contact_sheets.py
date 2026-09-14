#!/usr/bin/env python3
"""Create diagnostic contact sheets from pdftoppm PNG page renders."""

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RENDER_ROOT = ROOT / "tmp" / "pdfs" / "topic_renders_v2"


def make_sheet(directory: Path) -> Path:
    pages = sorted(directory.glob("page-*.png"))
    if not pages:
        raise RuntimeError(f"No pages in {directory}")
    opened = [Image.open(path).convert("RGB") for path in pages]
    thumb_w = 560
    thumbs = []
    for index, item in enumerate(opened, start=1):
        height = round(item.height * thumb_w / item.width)
        resized = item.resize((thumb_w, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, height + 34), "white")
        canvas.paste(resized, (0, 34))
        ImageDraw.Draw(canvas).text((10, 9), f"Page {index}", fill="#223344")
        thumbs.append(canvas)
    gap = 18
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    row_heights = [max(thumbs[i].height for i in range(r * cols, min((r + 1) * cols, len(thumbs)))) for r in range(rows)]
    width = cols * thumb_w + (cols + 1) * gap
    height = sum(row_heights) + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), "#dce3e8")
    y = gap
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if index >= len(thumbs):
                continue
            x = gap + col * (thumb_w + gap)
            sheet.paste(thumbs[index], (x, y))
        y += row_heights[row] + gap
    output = directory / "contact_sheet.jpg"
    sheet.save(output, quality=90, optimize=True)
    return output


def main() -> None:
    for directory in sorted(path for path in RENDER_ROOT.iterdir() if path.is_dir()):
        print(make_sheet(directory))


if __name__ == "__main__":
    main()

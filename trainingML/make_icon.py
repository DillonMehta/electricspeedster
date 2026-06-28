#!/usr/bin/env python3
"""
make_icon.py — turn the CueBench logo into app/menu-bar icons.

Outputs (in this folder):
  cuebench_icon.png  — a macOS *template* image (black mark on transparent, trimmed + padded)
                       for the menu bar status item. Template = adapts to light/dark menu bars.
  CueBench.icns      — multi-resolution Finder/Dock icon for CueBench.app (mark on a white
                       rounded-square, the usual macOS app-icon look).

Source: the CueBench "C + underline" mark. Re-run after changing the logo:
  python make_icon.py [path-to-logo.png]
"""
from __future__ import annotations
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCES = [
    "/Users/dillonmehta/cuebench/logo.png",
    "/Users/dillonmehta/cuebench/cuebench/app/logo-mark.png",
]


def load_mark(src: str) -> Image.Image:
    """Return the logo as RGBA, with a real alpha mask of the mark (drops a white background)."""
    im = Image.open(src).convert("RGBA")
    arr = np.array(im)
    alpha = arr[..., 3]
    if (alpha > 10).mean() < 0.98:           # already has transparency -> trust it
        return im
    # White background baked in: derive a mask from darkness (the mark is near-black).
    lum = arr[..., :3].mean(axis=2)
    mask = np.clip((255 - lum - 30) * 4, 0, 255).astype("uint8")
    out = np.zeros_like(arr)
    out[..., 3] = mask                       # black mark, alpha = darkness
    return Image.fromarray(out, "RGBA")


def trimmed(im: Image.Image, pad_frac: float = 0.06) -> Image.Image:
    bbox = im.getbbox()
    if bbox:
        im = im.crop(bbox)
    w, h = im.size
    side = max(w, h)
    pad = int(side * pad_frac)
    canvas = Image.new("RGBA", (side + 2 * pad, side + 2 * pad), (0, 0, 0, 0))
    canvas.paste(im, ((side - w) // 2 + pad, (side - h) // 2 + pad), im)
    return canvas


def make_template_icon(mark: Image.Image, out_path: str, size: int = 256):
    """Menu-bar template: force the shape to solid black, keep alpha, trim+pad, square."""
    arr = np.array(mark)
    arr[..., 0] = arr[..., 1] = arr[..., 2] = 0          # solid black; alpha carries the shape
    tem = trimmed(Image.fromarray(arr, "RGBA"))
    tem = tem.resize((size, size), Image.LANCZOS)
    tem.save(out_path)
    print(f"wrote {out_path}  ({size}x{size} template)")


def make_app_icns(mark: Image.Image, out_icns: str):
    """App icon: the colored/dark mark centered on a white rounded square, at all icon sizes."""
    base = 1024
    icon = Image.new("RGBA", (base, base), (0, 0, 0, 0))
    # white rounded-square background (macOS-style)
    bg = Image.new("RGBA", (base, base), (0, 0, 0, 0))
    d = ImageDraw.Draw(bg)
    d.rounded_rectangle([0, 0, base - 1, base - 1], radius=int(base * 0.225),
                        fill=(255, 255, 255, 255))
    icon = Image.alpha_composite(icon, bg)
    # paste mark at ~60% size, centered
    m = trimmed(mark, pad_frac=0.0)
    target = int(base * 0.60)
    m = m.resize((target, target), Image.LANCZOS)
    off = (base - target) // 2
    icon.alpha_composite(m, (off, off))

    iconset = os.path.join(HERE, "CueBench.iconset")
    os.makedirs(iconset, exist_ok=True)
    specs = [(16, 1), (16, 2), (32, 1), (32, 2), (128, 1), (128, 2),
             (256, 1), (256, 2), (512, 1), (512, 2)]
    for pt, scale in specs:
        px = pt * scale
        name = f"icon_{pt}x{pt}{'@2x' if scale == 2 else ''}.png"
        icon.resize((px, px), Image.LANCZOS).save(os.path.join(iconset, name))
    try:
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out_icns], check=True)
        print(f"wrote {out_icns}")
    except Exception as e:
        # Fallback: single-image icns via sips
        png1024 = os.path.join(HERE, "_appicon_1024.png")
        icon.save(png1024)
        subprocess.run(["sips", "-s", "format", "icns", png1024, "--out", out_icns],
                       check=False)
        print(f"wrote {out_icns} (via sips fallback; iconutil said {e!r})")
    finally:
        # tidy the intermediate iconset
        import shutil
        shutil.rmtree(iconset, ignore_errors=True)
        try:
            os.remove(os.path.join(HERE, "_appicon_1024.png"))
        except OSError:
            pass


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else next(
        (p for p in DEFAULT_SOURCES if os.path.exists(p)), None)
    if not src or not os.path.exists(src):
        print("No logo source found. Pass one: python make_icon.py /path/to/logo.png")
        return 1
    print(f"source: {src}")
    mark = load_mark(src)
    make_template_icon(mark, os.path.join(HERE, "cuebench_icon.png"))
    make_app_icns(mark, os.path.join(HERE, "CueBench.icns"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

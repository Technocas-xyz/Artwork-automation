"""Tests for the OpenCV similarity maths in src/compare.py.

The same design must score ~the same whatever the background is and however it
was resized / softened / re-encoded. Run:  python tests/test_compare.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.compare import similarity

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        FAILS.append(name)


def make_art(size: int = 1024) -> Image.Image:
    """A coloured, detailed design on a TRANSPARENT canvas (drawn 2x, then
    downsampled so the edges are anti-aliased like real artwork)."""
    s = size * 2
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = s / 1024.0

    def p(*v):
        return [int(x * k) for x in v]

    d.ellipse(p(150, 150, 874, 874), fill=(200, 30, 40, 255))
    d.ellipse(p(250, 250, 774, 774), fill=(250, 220, 40, 255))
    d.rectangle(p(330, 330, 694, 694), fill=(30, 90, 200, 255))
    d.polygon(p(512, 360, 560, 480, 690, 480, 585, 555, 625, 680,
                512, 605, 399, 680, 439, 555, 334, 480, 464, 480), fill=(255, 255, 255, 255))
    for i in range(6):
        d.line(p(200 + i * 100, 880, 260 + i * 100, 960), fill=(140, 60, 170, 255), width=int(14 * k))
    d.arc(p(100, 100, 924, 924), 200, 340, fill=(30, 160, 80, 255), width=int(18 * k))
    return im.resize((size, size), Image.LANCZOS)


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def jpeg(im: Image.Image, q: int = 70, bg=(255, 255, 255)) -> bytes:
    flat = Image.new("RGB", im.size, bg)
    flat.paste(im, mask=im.getchannel("A") if im.mode == "RGBA" else None)
    b = io.BytesIO()
    flat.save(b, "JPEG", quality=q)
    return b.getvalue()


def on_bg(im: Image.Image, bg) -> Image.Image:
    flat = Image.new("RGBA", im.size, bg + (255,))
    return Image.alpha_composite(flat, im)


def run(a: bytes, b: bytes) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return similarity(a, b)


def show(r: dict) -> str:
    return (f"overall={r['overall_pct']} detail={r['detail_pct']} shape={r['shape_pct']} "
            f"edges={r['edges_pct']} colour={r['colour_pct']}")


art = make_art(1024)
orig = png(art)

# identical -> 100
r = run(orig, orig)
check("identical = 100", r["overall_pct"] == 100 and r["edges_pct"] == 100
      and r["detail_pct"] == 100, show(r))

# resized 0.8x
small = png(art.resize((819, 819), Image.LANCZOS))
r = run(orig, small)
check("resize 0.8x overall>=90 edges>=80", r["overall_pct"] >= 90 and r["edges_pct"] >= 80, show(r))

# gaussian blur r=2
blur = png(art.filter(ImageFilter.GaussianBlur(2)))
r = run(orig, blur)
check("blur r=2 overall>=90 edges>=80", r["overall_pct"] >= 90 and r["edges_pct"] >= 80, show(r))

# JPEG round trip (flattened on white, as a JPEG must be)
r = run(orig, jpeg(art, 70))
check("JPEG overall>=90 edges>=80", r["overall_pct"] >= 90 and r["edges_pct"] >= 80, show(r))

# same art: black background vs transparent
black = png(on_bg(art, (0, 0, 0)))
r = run(black, orig)
check("black bg vs transparent overall>=90", r["overall_pct"] >= 90, show(r))
r = run(orig, black)
check("transparent vs black bg overall>=90", r["overall_pct"] >= 90, show(r))

# black vs white background
white = png(on_bg(art, (255, 255, 255)))
r = run(black, white)
check("black vs white bg overall>=90", r["overall_pct"] >= 90, show(r))

# canvas padding must not change the shape score (content bbox aspect, not canvas)
wide = Image.new("RGBA", (1600, 1024), (0, 0, 0, 0))
wide.paste(art, (288, 0))
r = run(orig, png(wide))
check("wider canvas, same art: shape>=90 overall>=90",
      r["shape_pct"] >= 90 and r["overall_pct"] >= 90, show(r))

# a genuinely different design must still score clearly lower
other = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
od = ImageDraw.Draw(other)
for i in range(8):
    od.rectangle((60 + i * 110, 100 + (i % 3) * 200, 120 + i * 110, 900), fill=(20, 140, 60, 255))
r = run(orig, png(other))
check("different design overall<70", r["overall_pct"] < 70, show(r))

print("RESULT: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED"))
sys.exit(1 if FAILS else 0)

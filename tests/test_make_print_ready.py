"""make_print_ready: never stretch — fit inside the target keeping aspect, then
pad transparent to the exact target. No network, no browser.

Run from the project root:  python tests/test_make_print_ready.py
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.postprocess import make_print_ready  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, extra="") -> None:
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{extra}]"))
    if not cond:
        FAILS.append(name)


def _png(w: int, h: int, alpha: int = 255) -> bytes:
    a = np.zeros((h, w, 4), np.uint8)
    a[:, :, 0] = 180
    a[:, :, 3] = alpha
    b = io.BytesIO()
    Image.fromarray(a, "RGBA").save(b, "PNG")
    return b.getvalue()


# The agent computes the target for a 1122x1402 input, width 10.75in @ 300 DPI,
# height auto: width_px=3225, height_px=round(3225*1402/1122)=4030.
width_px = round(10.75 * 300)
height_px = round(width_px * 1402 / 1122)
check("target math 1122x1402 @10.75/300 auto -> 3225x4030",
      (width_px, height_px) == (3225, 4030), (width_px, height_px))

# Case A: PhotoCraft returned a SQUARE 3225x3225. Must NOT stretch: fit inside
# 3225x4030 keeping square aspect -> 3225x3225, then pad transparent to target.
out, info = make_print_ready(_png(3225, 3225), 3225, 4030, 300, 120)
check("square result fits without stretch", info["fitted_size"] == (3225, 3225), info)
check("square result padded to exact target", info["to_size"] == (3225, 4030) and info["padded"], info)
arr = np.array(Image.open(io.BytesIO(out)))
check("padding is transparent (top & bottom rows)",
      bool((arr[0, :, 3] == 0).all()) and bool((arr[-1, :, 3] == 0).all()))

# Case B: correct portrait aspect, off size -> upscaled to fill, no pad.
out2, info2 = make_print_ready(_png(1122, 1402), 3225, 4030, 300, 120)
check("portrait result upscaled to fill", info2["resized"] and info2["to_size"] == (3225, 4030), info2)
check("portrait result needs no pad", not info2["padded"], info2)

# Case C: already exact -> no-op.
_, info3 = make_print_ready(_png(3225, 4030), 3225, 4030, 300, 120)
check("exact target is a no-op", not info3["resized"] and not info3["padded"], info3)

# Case D: alpha threshold still applies to semi-transparent pixels.
out4, info4 = make_print_ready(_png(10, 10, alpha=100), 10, 10, 300, 120)
check("semi-transparent thresholded", info4["thresholded"] and info4["changed_pixels"] == 100, info4)


def _png_alpha_rows(w, h, alphas):
    """A w x h image where each ROW y gets alpha alphas[y] (len(alphas)==h)."""
    a = np.zeros((h, w, 4), np.uint8)
    a[:, :, 0] = 180
    for y in range(h):
        a[y, :, 3] = alphas[y]
    b = io.BytesIO()
    Image.fromarray(a, "RGBA").save(b, "PNG")
    return b.getvalue()


# Case F: count ONLY pixels strictly between 0 and 255, split by destination.
# A 10-wide x 4-tall image with NO fully-transparent border (so trimming is a
# no-op and the target equals the size -> no resize/pad to shift rows):
#   row0 alpha=10  (partial, <120  -> 0)
#   row1 alpha=255 (already opaque, untouched -> NOT counted)
#   row2 alpha=50  (partial, <120  -> 0)
#   row3 alpha=200 (partial, >=120 -> 255)
# Partial pixels = rows 0,2,3 = 30; transparent (rows 0,2)=20; opaque (row3)=10.
# Row1 (already 255) is NOT counted.
outF, infoF = make_print_ready(_png_alpha_rows(10, 4, [10, 255, 50, 200]), 10, 4, 300, 120)
check("no trim/resize (target == size, no transparent border)",
      not infoF["trimmed"] and not infoF["resized"] and not infoF["padded"], infoF)
check("only strictly-partial pixels counted", infoF["changed_pixels"] == 30, infoF)
check("became_transparent counted (alpha<threshold)", infoF["became_transparent"] == 20, infoF)
check("became_opaque counted (alpha>=threshold)", infoF["became_opaque"] == 10, infoF)
arrF = np.array(Image.open(io.BytesIO(outF)))
check("partial<thr row0 forced to 0", bool((arrF[0, :, 3] == 0).all()), None)
check("fully opaque row1 stays 255", bool((arrF[1, :, 3] == 255).all()), None)
check("partial<thr row2 forced to 0", bool((arrF[2, :, 3] == 0).all()), None)
check("partial>=thr row3 forced to 255", bool((arrF[3, :, 3] == 255).all()), None)

# Summary text.
_, info5 = make_print_ready(_png(3225, 3225), 3225, 4030, 300, 120)
check("summary text",
      info5["summary"] == "Print ready: 3225 x 4030 px @ 300 DPI = 10.75 x 13.43 in",
      info5["summary"])


def _png_with_design(canvas_w, canvas_h, design_w, design_h):
    """A fully transparent canvas with an opaque design centered in it."""
    a = np.zeros((canvas_h, canvas_w, 4), np.uint8)
    x0 = (canvas_w - design_w) // 2
    y0 = (canvas_h - design_h) // 2
    a[y0:y0 + design_h, x0:x0 + design_w, 0] = 180
    a[y0:y0 + design_h, x0:x0 + design_w, 3] = 255
    b = io.BytesIO()
    Image.fromarray(a, "RGBA").save(b, "PNG")
    return b.getvalue()


# Case E (the DTF reason): PhotoCraft returns a 3225x3225 canvas holding a
# 2580x3225 design with transparent side margins. The print width is the DESIGN
# width, so make_print_ready must TRIM the transparent sides first, then fit the
# 2580x3225 design to a 3225-wide target. Expected output:
#   3225 x round(3225*3225/2580) px, and the design spans the FULL width.
design_w, design_h = 2580, 3225
tw = 3225
th = round(tw * design_h / design_w)   # round(3225*3225/2580) = 4031
canvas = _png_with_design(3225, 3225, design_w, design_h)
outE, infoE = make_print_ready(canvas, tw, th, 300, 120)
check("trimmed the transparent border", infoE["trimmed"] and infoE["from_size"] == (2580, 3225), infoE)
check("output sized to the trimmed design aspect", infoE["to_size"] == (tw, th), infoE)
arrE = np.array(Image.open(io.BytesIO(outE)))
# The design now spans the full width: the left and right edge columns have
# opaque pixels somewhere (no transparent side margin left).
left_opaque = bool((arrE[:, 0, 3] > 0).any())
right_opaque = bool((arrE[:, -1, 3] > 0).any())
check("design spans the full target width (no side margin)", left_opaque and right_opaque,
      (left_opaque, right_opaque))

print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)

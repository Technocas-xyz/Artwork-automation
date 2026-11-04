"""Tests for the Print Ready QA maths in src/printready.py.

A clean transparent artwork must score "ready" at 300 DPI both at 1024px and at
its 3225px upscale (no resolution-dependent false alarms). Run:
    python tests/test_printready_qa.py
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from PIL import Image, ImageDraw

from src.printready import measure_print_ready, _status_low_bad, _measure_min_stroke

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        FAILS.append(name)


def make_art(size: int = 1024, with_black: bool = True) -> Image.Image:
    """Clean artwork on a transparent canvas, anti-aliased (drawn 4x then reduced).
    Pale/white RGB under fully transparent pixels, as many exports leave it."""
    s = size * 4
    k = s / 1024.0
    im = Image.new("RGBA", (s, s), (255, 255, 255, 0))
    d = ImageDraw.Draw(im)

    def p(*v):
        return [int(x * k) for x in v]

    d.ellipse(p(150, 150, 874, 874), fill=(200, 30, 40, 255))
    d.ellipse(p(250, 250, 774, 774), fill=(250, 220, 40, 255))
    d.rectangle(p(330, 330, 694, 694), fill=(30, 90, 200, 255))
    d.polygon(p(512, 360, 560, 480, 690, 480, 585, 555, 625, 680,
                512, 605, 399, 680, 439, 555, 334, 480, 464, 480), fill=(255, 255, 255, 255))
    if with_black:
        for i in range(6):
            d.line(p(200 + i * 100, 880, 260 + i * 100, 960), fill=(0, 0, 0, 255), width=int(22 * k))
    # Tiny decorative specks (under 1mm even at the 3225px size) must be ignored.
    for i in range(5):
        d.ellipse(p(60 + i * 20, 60, 62 + i * 20, 62), fill=(30, 160, 80, 255))
    # Pillow premultiplies RGBA internally when resizing, so AA edges keep the art
    # colour. Then put white RGB back under fully transparent pixels, as many
    # exports do (it must NOT count as a halo).
    out = np.asarray(im.resize((size, size), Image.BOX)).copy()
    out[out[:, :, 3] == 0, :3] = 255
    return Image.fromarray(out, "RGBA")


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def by_id(rep: dict) -> dict:
    return {c["id"]: c for c in rep["checks"]}


# --- boundary: exactly the threshold passes --------------------------------
check("status_low_bad: exactly warn passes", _status_low_bad(300, 300, 150) == "pass")
check("status_low_bad: just below warn warns", _status_low_bad(299.9, 300, 150) == "warn")
check("status_low_bad: exactly fail is warn, not fail", _status_low_bad(150, 300, 150) == "warn")
check("status_low_bad: below fail fails", _status_low_bad(149, 300, 150) == "fail")

art = make_art(1024)
big = art.resize((3225, 3225), Image.LANCZOS)

for label, im in (("1024px", art), ("3225px upscale", big)):
    rep = measure_print_ready(png(im), dpi_override=300)
    c = by_id(rep)
    detail = " ".join(f"{k}={v['status']}:{v['value_str']}" for k, v in c.items()
                      if v["status"] in ("warn", "fail"))
    check(f"{label}: score>=90", rep["score"] >= 90, f"score={rep['score']} {detail}")
    check(f"{label}: no halo fail", c["halo"]["status"] != "fail", c["halo"]["value_str"])
    check(f"{label}: no halo warn", c["halo"]["status"] == "pass", c["halo"]["value_str"])
    check(f"{label}: no DPI warn", c["dpi"]["status"] == "pass", c["dpi"]["value_str"])
    check(f"{label}: soft edges pass", c["soft_edges"]["status"] == "pass", c["soft_edges"]["value_str"])
    check(f"{label}: gradients pass", c["gradients"]["status"] == "pass", c["gradients"]["value_str"])
    check(f"{label}: min stroke pass", c["min_stroke"]["status"] == "pass", c["min_stroke"]["value_str"])
    check(f"{label}: min feature pass", c["min_feature"]["status"] == "pass", c["min_feature"]["value_str"])

# --- invisible white fringe (alpha 0 with white RGB) must not count --------
c = by_id(measure_print_ready(png(art), dpi_override=300))
check("halo ignores alpha==0 white RGB", c["halo"]["value"] is not None and c["halo"]["value"] < 1.5,
      c["halo"]["value_str"])

# --- a real pale fringe IS caught ------------------------------------------
fr = np.asarray(make_art(1024, with_black=False)).copy()
alpha = fr[:, :, 3]
from scipy import ndimage  # noqa: E402  (only needed by this test)
core = alpha >= 250
ring = ndimage.binary_dilation(core, iterations=3) & ~core
fr[ring] = (235, 235, 235, 120)
# a dark artwork so the pale fringe is clearly lighter than its neighbour
fr[core & (fr[:, :, 0] > 180) & (fr[:, :, 1] < 60)] = (10, 10, 10, 255)
rep = measure_print_ready(png(Image.fromarray(fr, "RGBA")), dpi_override=300)
check("pale semi-transparent fringe is flagged", by_id(rep)["halo"]["status"] in ("warn", "fail"),
      by_id(rep)["halo"]["value_str"])

# --- pure black: no dark area -> n/a, not scored ---------------------------
rep = measure_print_ready(png(make_art(1024, with_black=False)), dpi_override=300)
check("no dark area -> black check is n/a", by_id(rep)["black"]["status"] == "na",
      by_id(rep)["black"]["value_str"])
check("black check present (scored) when there are dark lines",
      by_id(measure_print_ready(png(art), dpi_override=300))["black"]["status"] == "pass")

# --- min stroke: thick blob is not reported as ~1-2px -----------------------
blob = np.zeros((400, 400), dtype=bool)
blob[50:350, 50:350] = True
check("min stroke of a solid 300px block is large", _measure_min_stroke(blob, 300) > 100,
      f"{_measure_min_stroke(blob, 300):.1f}px")
line = np.zeros((400, 400), dtype=bool)
line[100:112, 20:380] = True
sw = _measure_min_stroke(line, 300)
check("min stroke of a 12px line ~12px", 10 <= sw <= 14, f"{sw:.1f}px")
both = blob.copy()
both[:, :] |= False
thin = np.zeros((400, 400), dtype=bool)
thin[10:300, 10:300] = True
thin[320:326, 20:380] = True      # 6px line
check("min stroke finds the thin line next to a block", 4 <= _measure_min_stroke(thin, 300) <= 8,
      f"{_measure_min_stroke(thin, 300):.1f}px")
speck = line.copy()
speck[300:303, 300:303] = True    # 3px speck (<0.5mm) must be ignored
check("sub-0.5mm speck ignored for stroke", 10 <= _measure_min_stroke(speck, 300) <= 14,
      f"{_measure_min_stroke(speck, 300):.1f}px")

# --- DPI: a PNG saved at 300 DPI reads back as 299.9994 and must pass -------
buf = io.BytesIO()
art.save(buf, "PNG", dpi=(300, 300))
rep_dpi = measure_print_ready(buf.getvalue())          # no override: read from the file
cd = by_id(rep_dpi)["dpi"]
check("PNG saved at 300 DPI -> DPI pass", cd["status"] == "pass", cd["value_str"])
check("DPI value is the rounded 300", cd["value"] == 300.0 and cd["value_str"].startswith("300 DPI"),
      cd["value_str"])
buf = io.BytesIO()
art.save(buf, "PNG", dpi=(299, 299))
check("299 DPI still warns", by_id(measure_print_ready(buf.getvalue()))["dpi"]["status"] == "warn")


# --- Edge sharpness: silhouette alpha transition, not interior texture ------
def lettering(size: int = 1024) -> Image.Image:
    """Brush-lettering-like strokes (curves, thick/thin) with heavy interior
    shading, anti-aliased, on a transparent canvas."""
    s = size * 4
    k = s / 1024.0
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, x in enumerate(range(120, 900, 170)):
        d.arc([int(x * k), int(220 * k), int((x + 150) * k), int(520 * k)], 20, 330,
              fill=(40 + i * 15, 20, 90, 255), width=int(46 * k))
        d.line([int((x + 20) * k), int(560 * k), int((x + 120) * k), int(860 * k)],
               fill=(20, 60 + i * 20, 120, 255), width=int(38 * k))
    # Busy interior texture / shading INSIDE the strokes must not count as soft.
    rng = np.random.default_rng(3)
    tex = (rng.random((s, s)) * 90).astype(np.uint8)
    a = np.asarray(im).copy()
    solid = a[:, :, 3] == 255
    for c in range(3):
        ch = a[:, :, c].astype(np.int16)
        ch[solid] = np.clip(ch[solid] + tex[solid] - 45, 0, 255)
        a[:, :, c] = ch.astype(np.uint8)
    return Image.fromarray(a, "RGBA").resize((size, size), Image.BOX)


def feathered(im: Image.Image, px: float) -> Image.Image:
    """Same art with an inward linear alpha ramp of `px` pixels at its edge."""
    import cv2
    a = np.asarray(im).copy()
    solid = (a[:, :, 3] > 127).astype(np.uint8)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    a[:, :, 3] = (np.clip(dist / px, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


text = lettering(1024)
c = by_id(measure_print_ready(png(text), dpi_override=300))["soft_edges"]
check("clean anti-aliased lettering (busy interior) -> edge sharpness pass",
      c["status"] == "pass" and c["value"] < 20, c["value_str"])
c = by_id(measure_print_ready(png(lettering(3225)), dpi_override=300))["soft_edges"]
check("same lettering at 3225px -> pass", c["status"] == "pass", c["value_str"])
c = by_id(measure_print_ready(png(feathered(text, 6)), dpi_override=300))["soft_edges"]
check("6px feathered edge -> edge sharpness fail", c["status"] == "fail", c["value_str"])
c = by_id(measure_print_ready(png(feathered(text, 1.5)), dpi_override=300))["soft_edges"]
check("1.5px ramp (normal anti-alias) -> pass", c["status"] == "pass", c["value_str"])
c = by_id(measure_print_ready(png(art), dpi_override=300))["soft_edges"]
check("clean art -> pass", c["status"] == "pass", c["value_str"])

print("RESULT: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED"))
sys.exit(1 if FAILS else 0)

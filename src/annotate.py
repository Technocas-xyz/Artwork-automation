"""Local image annotation for the Artwork Identification workflow.

ALL annotation is drawn LOCALLY here — we never ask ChatGPT to draw on the
image, because it redraws the artwork instead of overlaying. Two jobs:

  Object mode (agent-side, Pillow only):
    - grid_overlay()  : a labelled 10x10 grid (cols A-J, rows 1-10) drawn over a
                        COPY of the artwork. This copy is what we send to
                        ChatGPT so it can name each object with the grid cells it
                        occupies (text JSON), without touching the artwork.
    - cells_to_bbox() : turn ChatGPT's cell references into a normalised bbox —
                        treating them as a HINT, not truth. If the cells are
                        scattered or cover more than half the grid, we reject the
                        hint (return None) and draw no pin: a badly placed pin is
                        worse than none.
    - draw_badges()   : numbered circular badges drawn on the ORIGINAL image at
                        each object's bbox centre.

  Colour mode (server-side, needs cv2):
    - colour_groups() : OpenCV k-means over the opaque pixels, returning numbered
                        groups with a swatch RGB + pixel %, plus an overlay image.
                        NO ChatGPT turn. cv2 is imported LAZILY inside this
                        function so the packaged agent (which excludes cv2) can
                        still import this module for the Pillow-only helpers.

Pillow/numpy only at module import time. cv2 is touched only when colour_groups
is actually called (server), mirroring the guarded import in src/compare.py.
"""
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# 10x10 grid: columns A..J (left->right), rows 1..10 (top->bottom).
GRID_N = 10
_COLS = "ABCDEFGHIJ"


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A TrueType font at `size` if one is available, else Pillow's default.

    We never fail on a missing font: labels degrade to the bitmap default
    rather than breaking annotation."""
    for name in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> tuple[int, int]:
    """Width/height of `text` in `font`, across Pillow versions."""
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        try:
            return font.getsize(text)  # very old Pillow
        except Exception:
            return (len(text) * 6, 11)


# ---------------------------------------------------------------------------
# Grid overlay (agent-side, Pillow only)
# ---------------------------------------------------------------------------

def grid_overlay(image_bytes: bytes) -> bytes:
    """Return a PNG copy of the artwork with a labelled 10x10 grid drawn over it.

    Thin, semi-transparent lines so the artwork stays readable underneath;
    column letters A-J along the top and row numbers 1-10 down the left, each
    with a small solid backing chip so they're legible on any artwork. The
    result is sent to ChatGPT ONLY to let it reference cells — the original is
    never modified."""
    base = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    line_rgba = (255, 0, 0, 90)          # thin semi-transparent red
    line_w = max(1, round(min(w, h) / 900))
    label_px = max(11, round(min(w, h) / 45))
    font = _load_font(label_px)

    # Grid lines.
    for i in range(1, GRID_N):
        x = round(w * i / GRID_N)
        d.line([(x, 0), (x, h)], fill=line_rgba, width=line_w)
        y = round(h * i / GRID_N)
        d.line([(0, y), (w, y)], fill=line_rgba, width=line_w)
    # Outer border.
    d.rectangle([(0, 0), (w - 1, h - 1)], outline=(255, 0, 0, 130), width=line_w)

    def _chip(cx: int, cy: int, text: str) -> None:
        tw, th = _text_size(d, text, font)
        pad = max(2, label_px // 5)
        x0, y0 = cx - tw / 2 - pad, cy - th / 2 - pad
        x1, y1 = cx + tw / 2 + pad, cy + th / 2 + pad
        d.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255, 210))
        d.text((cx - tw / 2, cy - th / 2), text, fill=(200, 0, 0, 255), font=font)

    # Column letters centred in each column, near the top; row numbers near left.
    for c in range(GRID_N):
        cx = round(w * (c + 0.5) / GRID_N)
        _chip(cx, round(label_px * 0.9), _COLS[c])
    for r in range(GRID_N):
        cy = round(h * (r + 0.5) / GRID_N)
        _chip(round(label_px * 0.9), cy, str(r + 1))

    out = Image.alpha_composite(base, overlay)
    buf = io.BytesIO()
    out.convert("RGBA").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Cell references -> bbox (agent-side). The hint is treated with suspicion.
# ---------------------------------------------------------------------------

def _parse_cell(cell: str) -> tuple[int, int] | None:
    """Parse a cell like 'D3' / 'd3' into (col_index 0-9, row_index 0-9).

    Accepts letter-then-number only. Returns None for anything off-grid or
    malformed — ChatGPT invents cells like 'K11' or 'D' and we must not crash."""
    if not isinstance(cell, str):
        return None
    s = cell.strip().upper()
    if len(s) < 2 or s[0] not in _COLS or not s[1:].isdigit():
        return None
    col = _COLS.index(s[0])
    row = int(s[1:]) - 1
    if not (0 <= col < GRID_N and 0 <= row < GRID_N):
        return None
    return col, row


def cells_to_bbox(cells: list[str], max_cells_fraction: float = 0.5,
                  max_span_cells: int = 7) -> dict | None:
    """Convert ChatGPT's grid cells into a NORMALISED bbox, or None to drop the pin.

    The cells are a hint from a model that is unreliable at exact positions, so
    we reject a hint that is clearly untrustworthy rather than draw a misleading
    pin (a badly placed pin is worse than no pin):

      - No valid cells                       -> None
      - Cells cover > `max_cells_fraction`
        of the whole grid (default half)     -> None  (basically "everywhere")
      - The cells span more than
        `max_span_cells` columns OR rows      -> None  (scattered across the image)

    Otherwise returns {x, y, w, h} in 0..1 image coordinates: the tight box
    around the referenced cells. Callers place a numbered badge at its centre."""
    parsed = [p for p in (_parse_cell(c) for c in (cells or [])) if p is not None]
    if not parsed:
        return None

    total_cells = GRID_N * GRID_N
    if len(set(parsed)) > max_cells_fraction * total_cells:
        return None   # references half the grid or more — not a locatable object

    cols = [c for c, _ in parsed]
    rows = [r for _, r in parsed]
    col_span = max(cols) - min(cols) + 1
    row_span = max(rows) - min(rows) + 1
    if col_span > max_span_cells or row_span > max_span_cells:
        return None   # scattered across the image — the hint is not reliable

    x = min(cols) / GRID_N
    y = min(rows) / GRID_N
    w = col_span / GRID_N
    h = row_span / GRID_N
    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}


# ---------------------------------------------------------------------------
# Numbered badges on the ORIGINAL image (agent-side, Pillow only)
# ---------------------------------------------------------------------------

def draw_badges(original_bytes: bytes, objects: list[dict]) -> bytes:
    """Draw numbered circular badges on the ORIGINAL artwork (never the grid copy).

    `objects` is [{id, name, bbox?}, ...]; a badge is drawn only for objects with
    a bbox (a surviving hint), centred on that bbox. Objects without a bbox are
    still in the list on the UI, just un-pinned. Returns a PNG."""
    base = Image.open(io.BytesIO(original_bytes)).convert("RGBA")
    w, h = base.size
    d = ImageDraw.Draw(base)

    radius = max(12, round(min(w, h) / 22))
    font = _load_font(max(12, round(radius * 1.1)))

    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        cx = round((bbox["x"] + bbox["w"] / 2) * w)
        cy = round((bbox["y"] + bbox["h"] / 2) * h)
        num = str(obj.get("id", "?"))
        # White ring + red disc so it reads on any artwork.
        d.ellipse([(cx - radius - 2, cy - radius - 2), (cx + radius + 2, cy + radius + 2)],
                  fill=(255, 255, 255, 255))
        d.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)],
                  fill=(220, 30, 30, 255))
        tw, th = _text_size(d, num, font)
        d.text((cx - tw / 2, cy - th / 2), num, fill=(255, 255, 255, 255), font=font)

    buf = io.BytesIO()
    base.save(buf, format="PNG")
    return buf.getvalue()


def has_transparency(image_bytes: bytes) -> bool:
    """True if the image has any meaningfully transparent pixels.

    Drives the regenerate prompt's background clause: transparent art asks for a
    transparent background, opaque art asks to keep the original background."""
    im = Image.open(io.BytesIO(image_bytes))
    if im.mode not in ("RGBA", "LA") and "transparency" not in im.info:
        return False
    im = im.convert("RGBA")
    alpha = np.asarray(im)[:, :, 3]
    # "Meaningful" = at least 1% of pixels below near-opaque, so a stray edge
    # pixel doesn't flip an otherwise-opaque design.
    return float((alpha < 250).mean()) > 0.01


# ---------------------------------------------------------------------------
# Colour groups (SERVER-side, cv2). Lazy cv2 import — see module docstring.
# ---------------------------------------------------------------------------

def colour_groups(image_bytes: bytes, k_min: int = 5, k_max: int = 8) -> dict:
    """Cluster the opaque pixels into colour groups with OpenCV k-means.

    Fully local, no ChatGPT. Transparent pixels are ignored. Returns:
        {
          "groups": [{"id", "name", "rgb":[r,g,b], "hex", "pct"}...],  # by pct desc
          "overlay": <PNG bytes>,   # each group's pixels tinted + numbered
        }
    `k` is clamped into [k_min, k_max] and never exceeds the number of distinct
    colours present. cv2 is imported here so importing this module elsewhere
    (the agent) does not require cv2."""
    import cv2  # lazy: server-only dependency

    im = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = im.size
    arr = np.asarray(im)                     # H x W x 4, uint8
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    opaque = alpha > 20                      # ignore transparent pixels

    total = int(w * h)
    n_opaque = int(opaque.sum())
    n_transparent = total - n_opaque
    logger.info("colour_groups: image %dx%d (%d px), opaque=%d, transparent(ignored)=%d",
                w, h, total, n_opaque, n_transparent)
    if n_opaque == 0:
        # Nothing to cluster (fully transparent). Return an empty result rather
        # than crashing; the UI shows "no colour groups found".
        logger.warning("colour_groups: image is fully transparent (alpha<=20 everywhere) — "
                       "no opaque pixels to cluster, returning 0 groups.")
        empty = io.BytesIO()
        im.save(empty, format="PNG")
        return {"groups": [], "overlay": empty.getvalue()}

    # cv2.kmeans requires a C-contiguous float32 N x 3 array. Boolean-mask
    # indexing already yields a contiguous copy, but make it explicit so a
    # future refactor cannot silently hand cv2 a non-contiguous view (which
    # raises and would zero the group list).
    samples = np.ascontiguousarray(rgb[opaque], dtype=np.float32)   # N x 3
    distinct = int(np.unique(samples.astype(np.uint8), axis=0).shape[0])
    # k must not exceed the number of distinct colours (cv2 errors if k > N of
    # unique samples in effect), and is clamped into [k_min, k_max]. A normal
    # opaque photo/artwork has thousands of distinct colours, so k lands at
    # k_max; a flat 2-3 colour logo clamps down to `distinct`.
    k = max(1, min(k_max, distinct))
    if distinct >= k_min:
        k = max(k_min, k)
    logger.info("colour_groups: distinct colours=%d, k=%d (k_min=%d, k_max=%d)",
                distinct, k, k_min, k_max)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    attempts = 3
    _compactness, labels, centres = cv2.kmeans(
        samples, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    centres_u8 = np.clip(centres, 0, 255).astype(np.uint8)

    counts = np.bincount(labels, minlength=k)
    order = np.argsort(counts)[::-1]          # largest group first

    # Drop negligible clusters. k-means on antialiased art often splits one
    # colour into a big cluster plus a few near-empty ones (edge pixels); showing
    # "0.0%" groups is noise. Keep only clusters that are at least min_pct of the
    # opaque area, so the list is the handful of colours a person would name.
    min_pct = 0.5

    # Overlay: place a numbered badge at each surviving cluster's centroid.
    overlay_im = im.copy()
    od = ImageDraw.Draw(overlay_im)
    ys, xs = np.where(opaque)
    label_full = np.full((h, w), -1, dtype=np.int32)
    label_full[ys, xs] = labels

    groups: list[dict] = []
    radius = max(12, round(min(w, h) / 22))
    font = _load_font(max(12, round(radius * 1.1)))
    new_id = 0
    for cluster in order:
        pct = round(100.0 * float(counts[cluster]) / n_opaque, 1)
        if pct < min_pct:
            continue
        new_id += 1
        r, g, b = (int(centres_u8[cluster][0]),
                   int(centres_u8[cluster][1]),
                   int(centres_u8[cluster][2]))
        groups.append({
            "id": new_id,
            "name": _colour_name(r, g, b),
            "rgb": [r, g, b],
            "hex": f"#{r:02X}{g:02X}{b:02X}",
            "pct": pct,
        })
        mask = label_full == cluster
        m_ys, m_xs = np.where(mask)
        if m_xs.size:
            cx, cy = int(m_xs.mean()), int(m_ys.mean())
            od.ellipse([(cx - radius - 2, cy - radius - 2), (cx + radius + 2, cy + radius + 2)],
                       fill=(255, 255, 255, 255))
            od.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)],
                       fill=(r, g, b, 255))
            # Number in black or white depending on the swatch luminance.
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            txt_fill = (0, 0, 0, 255) if lum > 140 else (255, 255, 255, 255)
            num = str(new_id)
            tw, th = _text_size(od, num, font)
            od.text((cx - tw / 2, cy - th / 2), num, fill=txt_fill, font=font)

    logger.info("colour_groups: %d group(s) kept (min_pct=%.1f%% of opaque area): %s",
                len(groups), min_pct,
                ", ".join(f"{g['name']} {g['pct']}%" for g in groups) or "(none)")

    buf = io.BytesIO()
    overlay_im.save(buf, format="PNG")
    return {"groups": groups, "overlay": buf.getvalue()}


def _colour_name(r: int, g: int, b: int) -> str:
    """A short human colour name for a swatch — enough for the operator to
    recognise which group is which alongside the swatch itself."""
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 25:
        if mx > 220:
            return "White / light grey"
        if mx < 45:
            return "Black"
        if mx < 110:
            return "Dark grey"
        return "Grey"
    # Hue from the dominant channel(s).
    if r >= g and r >= b:
        if g > 150 and b < 120:
            name = "Yellow" if g > 200 else "Orange"
        elif g > 60 and b < g:
            name = "Brown"
        else:
            name = "Red"
    elif g >= r and g >= b:
        name = "Yellow-green" if r > 150 else "Green"
    else:
        name = "Blue" if b > g else "Purple"
    tone = "dark " if mx < 110 else ("light " if mn > 150 else "")
    return (tone + name).strip().capitalize()

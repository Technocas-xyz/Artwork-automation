"""Local aspect-ratio helpers for the Custom Operation "Aspect Ratio Enhancement".

Resizing the canvas shape is done locally with Pillow — never by ChatGPT —
because redrawing to change canvas shape would discard the original pixels for
no benefit. The only operation performed here is TRANSPARENT PADDING to reach a
target ratio. We never crop and never stretch/distort the design.
"""
from __future__ import annotations

import io
from math import gcd
from pathlib import Path

from PIL import Image

from config.workflows import normalise_ratio

DEFAULT_DPI = 300


def _simplify_ratio(w: int, h: int) -> str:
    """Return the simplified aspect ratio as a 'W:H' string (e.g. 3:2)."""
    if w <= 0 or h <= 0:
        return "0:0"
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def _dpi_from_image(img: Image.Image) -> tuple[int, bool]:
    """Return (dpi, had_dpi). Falls back to DEFAULT_DPI when the file carries none."""
    dpi = img.info.get("dpi")
    if dpi:
        # dpi may be a tuple (x, y); take the horizontal value, round sensibly.
        val = dpi[0] if isinstance(dpi, (tuple, list)) else dpi
        try:
            val = round(float(val))
        except (TypeError, ValueError):
            val = 0
        if val and val > 0:
            return val, True
    return DEFAULT_DPI, False


def image_info(path: str | Path, dpi_override: int | None = None) -> dict:
    """Inspect an image file and return its dimensions, ratio and print size.

    Returns keys:
        width, height        - pixel dimensions
        dpi                  - the DPI used for the inch calculation
        dpi_from_file        - True if the DPI came from the file, False if assumed
        ratio                - simplified aspect ratio string, e.g. "3:2"
        inches_w, inches_h   - print size in inches at `dpi`
    """
    path = Path(path)
    with Image.open(path) as img:
        width, height = img.size
        file_dpi, had_dpi = _dpi_from_image(img)

    dpi = dpi_override if (dpi_override and dpi_override > 0) else file_dpi
    inches_w = round(width / dpi, 2) if dpi else 0
    inches_h = round(height / dpi, 2) if dpi else 0

    info = {
        "width": width,
        "height": height,
        "dpi": dpi,
        "dpi_from_file": had_dpi and not dpi_override,
        "ratio": _simplify_ratio(width, height),
        "ratio_normalised": normalise_ratio(width, height),
        "inches_w": inches_w,
        "inches_h": inches_h,
    }

    # Merge in the rich feature set so the left panel can show the extra
    # measurements the moment a file is uploaded. Imported lazily to avoid a
    # heavy dependency at module import time.
    try:
        from src.postprocess import extract_features
        feats = extract_features(path.read_bytes())
        info.update({
            "total_pixels": feats["total_pixels"],
            "channels": feats["channels"],
            "has_alpha": feats["has_alpha"],
            "transparent_pixel_pct": feats["transparent_pixel_pct"],
            "mean_colour": feats["mean_colour"],
            "dominant_colours": feats["dominant_colours"],
            "edge_density": feats["edge_density"],
            "n_contours": feats["n_contours"],
        })
    except Exception as exc:  # never let feature extraction break basic info
        print(f"[aspect] feature extraction failed: {exc}")

    return info


def fit_to_ratio(image_bytes: bytes, target_w: float, target_h: float) -> bytes:
    """Pad an image with transparency to reach target_w:target_h aspect ratio.

    NEVER crops. NEVER stretches/distorts. The original pixels are preserved
    exactly and centred; only transparent padding is added on the short axis.
    Returns PNG bytes (RGBA).
    """
    if target_w <= 0 or target_h <= 0:
        raise ValueError("Target ratio dimensions must be positive.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size
    target_ratio = target_w / target_h
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-6:
        # Already at the target ratio — return unchanged as PNG.
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    if current_ratio > target_ratio:
        # Too wide: pad the height (top/bottom).
        new_w = w
        new_h = round(w / target_ratio)
    else:
        # Too tall: pad the width (left/right).
        new_h = h
        new_w = round(h * target_ratio)

    # Guard against rounding that would crop.
    new_w = max(new_w, w)
    new_h = max(new_h, h)

    canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    offset = ((new_w - w) // 2, (new_h - h) // 2)
    canvas.paste(img, offset, img)

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()

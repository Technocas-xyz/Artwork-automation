"""Artwork extraction utilities.

Primary extraction is done by ChatGPT (via get_boxes in generator.py).
This module provides:
- crop_boxes: crop an image given bounding box dicts (Pillow only)
- grid_split: manual grid fallback for when ChatGPT detection fails
- validate_crops: flag suspect crops with warnings
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


class ExtractionError(Exception):
    """Raised when extraction fails to produce any results."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def crop_boxes(
    image_bytes: bytes,
    boxes: list[dict],
    padding: int = 8,
) -> list[bytes]:
    """Crop an image at the given bounding boxes.

    Parameters
    ----------
    image_bytes : bytes
        The full mockup sheet as PNG/JPEG bytes.
    boxes : list[dict]
        Each dict must have keys: x, y, w, h (ints, pixel coords).
    padding : int
        Extra pixels to add around each box (default 8).

    Returns list of PNG bytes in box order.
    Pixels are exactly as they appear in the sheet — no trimming, no
    background removal, no connected-component processing.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img_w, img_h = img.size

    # Convert to RGBA for consistent output
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    results: list[bytes] = []
    for box in boxes:
        x = int(box["x"])
        y = int(box["y"])
        w = int(box["w"])
        h = int(box["h"])

        # Apply padding
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(img_w, x + w + padding)
        y1 = min(img_h, y + h + padding)

        if x1 <= x0 or y1 <= y0:
            continue

        crop = img.crop((x0, y0, x1, y1))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        results.append(buf.getvalue())

    if not results:
        raise ExtractionError("crop_boxes produced no crops from the given boxes.")
    return results


def grid_split(
    image_bytes: bytes,
    cols: int,
    rows: int,
    padding: int = 8,
    inset_pct: float = 4.0,
) -> list[bytes]:
    """Manual grid split — fallback when ChatGPT detection fails.

    Divides the image into cols x rows equal cells, applies an inset to
    avoid neighbour bleed, and trims each cell to its content.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size
    cell_w = w // cols
    cell_h = h // rows

    results: list[bytes] = []
    for row in range(rows):
        for col in range(cols):
            x0 = col * cell_w
            y0 = row * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            cell = img.crop((x0, y0, x1, y1))

            # Apply inset
            if inset_pct > 0:
                cw, ch = cell.size
                ix = int(cw * inset_pct / 100)
                iy = int(ch * inset_pct / 100)
                if cw - 2 * ix > 10 and ch - 2 * iy > 10:
                    cell = cell.crop((ix, iy, cw - ix, ch - iy))

            buf = io.BytesIO()
            cell.save(buf, format="PNG")
            results.append(buf.getvalue())

    if not results:
        raise ExtractionError("Grid split produced no cells.")
    return results


def validate_crops(crops: list[bytes]) -> list[str | None]:
    """Flag suspect crops with warning strings.

    Checks:
    - Content touches crop edge (possible cut-off) — only for images with transparency
    - Area is a strong outlier vs median
    - Aspect ratio is far from the median
    """
    if not crops:
        return []

    infos: list[dict] = []
    for idx, data in enumerate(crops):
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        arr = np.array(img)
        h, w = arr.shape[:2]
        alpha = arr[:, :, 3]
        has_transparency = bool((alpha < 250).any())

        if has_transparency:
            # Use alpha to detect content
            content = alpha > 128
            touches = _content_touches_edge(content)
        else:
            # Fully opaque image (e.g. JPEG crop) — can't use alpha for edge detection.
            content = np.ones((h, w), dtype=bool)
            touches = False

        # Debug for crop 1
        if idx == 0:
            top_f = float(content[0, :].sum()) / w if w > 0 else 0
            bot_f = float(content[h-1, :].sum()) / w if w > 0 else 0
            left_f = float(content[:, 0].sum()) / h if h > 0 else 0
            right_f = float(content[:, w-1].sum()) / h if h > 0 else 0
            print(f"[validate] Crop 1 debug: has_transparency={has_transparency} touches={touches} "
                  f"top={top_f:.2f} bot={bot_f:.2f} left={left_f:.2f} right={right_f:.2f} "
                  f"alpha_min={int(alpha.min())} alpha_max={int(alpha.max())}")

        infos.append({
            "w": w, "h": h,
            "area": int(content.sum()),
            "aspect": w / max(h, 1),
            "touches_edge": touches,
        })

    areas = [i["area"] for i in infos]
    aspects = [i["aspect"] for i in infos]
    median_area = float(np.median(areas)) if areas else 1
    median_aspect = float(np.median(aspects)) if aspects else 1

    warnings: list[str | None] = []
    for info in infos:
        issues: list[str] = []
        if info["touches_edge"]:
            issues.append("Content touches edge (may be cut off)")
        if median_area > 0 and abs(info["area"] - median_area) > median_area * 0.6:
            issues.append("Unusual size")
        if median_aspect > 0 and abs(info["aspect"] - median_aspect) > median_aspect * 0.5:
            issues.append("Unusual shape")
        warnings.append("; ".join(issues) if issues else None)

    return warnings


# Legacy compatibility
def auto_detect(image_bytes: bytes, **kwargs) -> list[bytes]:
    """Legacy — raises error directing caller to use ChatGPT-based detection."""
    raise ExtractionError(
        "auto_detect is no longer available. Use get_boxes() from src.generator for ChatGPT-based detection."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _content_touches_edge(content: np.ndarray) -> bool:
    """Check if content is likely cut off at the edge.

    Only flags if content touches MORE THAN TWO sides, or if the touching
    run covers more than 60% of any single edge. Tight crops naturally touch
    edges — only genuinely cut-off designs run along most of one side.
    """
    h, w = content.shape
    if h == 0 or w == 0:
        return False

    threshold = 0.6  # 60% of edge length

    top_frac = float(content[0, :].sum()) / w if w > 0 else 0
    bot_frac = float(content[h - 1, :].sum()) / w if w > 0 else 0
    left_frac = float(content[:, 0].sum()) / h if h > 0 else 0
    right_frac = float(content[:, w - 1].sum()) / h if h > 0 else 0

    # Count sides with ANY touching
    sides_touching = sum(1 for f in [top_frac, bot_frac, left_frac, right_frac] if f > 0)

    flagged = False
    # Flag if more than 2 sides touched
    if sides_touching > 2:
        flagged = True

    # Flag if any single edge has >60% coverage
    if top_frac > threshold or bot_frac > threshold or left_frac > threshold or right_frac > threshold:
        flagged = True

    if flagged:
        print(f"[validate] Edge flag: top={top_frac:.2f} bot={bot_frac:.2f} left={left_frac:.2f} right={right_frac:.2f} sides={sides_touching}")

    return flagged

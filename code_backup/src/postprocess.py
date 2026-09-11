"""Post-processing utilities for generated artwork images.

Agent-side operations only — NO OpenCV (cv2) import here, so the packaged agent
.exe stays free of the large cv2 .pyd that breaks PyInstaller one-file
extraction. The cv2-dependent comparison / geometry functions live in
src/compare.py, which the agent never imports.

Responsibilities:
  1. White-background removal for DTF print production (is_opaque_white_bg,
     remove_white_background) — used across all workflows.
  2. Deterministic local pixel operations for the Custom Operation workflow
     (black_out, half_tone). Pillow/numpy only; they never open a chat.
"""
from __future__ import annotations

import io
import math
from collections import deque

import numpy as np
from PIL import Image, ImageFilter


def is_opaque_white_bg(png_bytes: bytes, tolerance: int = 12) -> bool:
    """Check whether the image has an opaque white background.

    Samples the four corner pixels. Returns True if all four are
    near-white (R, G, B >= 255 - tolerance) with full alpha.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size

    corners = [
        img.getpixel((0, 0)),
        img.getpixel((w - 1, 0)),
        img.getpixel((0, h - 1)),
        img.getpixel((w - 1, h - 1)),
    ]

    threshold = 255 - tolerance
    for r, g, b, a in corners:
        if a < 255:
            return False
        if r < threshold or g < threshold or b < threshold:
            return False
    return True


def remove_white_background(png_bytes: bytes, tolerance: int = 12) -> bytes:
    """Remove white background connected to image edges, preserving interior white.

    Uses a flood-fill from all border pixels to identify background regions.
    Only pixels reachable from the edge that are near-white have their alpha set
    to 0. Interior white (outlines, highlights) is untouched.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    pixels = img.load()
    w, h = img.size
    threshold = 255 - tolerance

    visited = [[False] * h for _ in range(w)]

    def _is_white(x: int, y: int) -> bool:
        r, g, b, _a = pixels[x, y]
        return r >= threshold and g >= threshold and b >= threshold

    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            if _is_white(x, y) and not visited[x][y]:
                visited[x][y] = True
                queue.append((x, y))

    for y in range(h):
        for x in (0, w - 1):
            if _is_white(x, y) and not visited[x][y]:
                visited[x][y] = True
                queue.append((x, y))

    while queue:
        cx, cy = queue.popleft()
        r, g, b, _a = pixels[cx, cy]
        pixels[cx, cy] = (r, g, b, 0)
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                if _is_white(nx, ny):
                    visited[nx][ny] = True
                    queue.append((nx, ny))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Deterministic local operations for the Custom Operation workflow
# ---------------------------------------------------------------------------


def _load_rgba(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def _to_png_bytes(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def black_out(png_bytes: bytes, threshold: int = 40, feather: int = 1) -> bytes:
    """Make the BLACK areas of the artwork transparent (knockout black).

    On a dark garment the fabric shows through the transparent areas as the
    black, saving ink and giving a cleaner result. This is standard DTF / screen
    print practice for dark garments — it does NOT turn everything black.

    - A pixel counts as black when all of R, G and B are below `threshold`.
    - Those pixels get alpha 0; every other pixel is left untouched.
    - `threshold` is configurable so near-black (not pure #000000) also drops out.
    - `feather` softens the cut boundary by roughly one pixel so the edge does
      not look jagged on press.
    """
    img = _load_rgba(png_bytes)
    arr = np.array(img).astype(np.uint8)  # (H, W, 4)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    # Black mask: all channels below threshold (and the pixel is actually visible).
    black_mask = (r < threshold) & (g < threshold) & (b < threshold) & (a > 0)

    new_alpha = a.astype(np.float32)

    if feather and feather > 0:
        # Soften the boundary: blur the binary mask and scale alpha down by how
        # "black" the neighbourhood is, so the cut edge fades over ~1px.
        mask_img = Image.fromarray((black_mask * 255).astype(np.uint8), mode="L")
        blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather)))
        soft = np.array(blurred).astype(np.float32) / 255.0  # 0..1, 1 = fully black
        new_alpha = new_alpha * (1.0 - soft)
    else:
        new_alpha[black_mask] = 0.0

    out = arr.copy()
    out[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)
    # RGB left untouched for all non-dropped pixels.
    return _to_png_bytes(Image.fromarray(out, mode="RGBA"))


def half_tone(
    png_bytes: bytes,
    lpi: float = 40.0,
    angle: float = 22.5,
    dpi: float = 300.0,
    dot: str = "round",
    min_radius_px: float = 0.6,
) -> bytes:
    """Apply an AM halftone dot treatment as a deterministic local operation.

    - Grayscale tone drives dot SIZE only; every drawn dot is fully opaque.
    - Cell size is derived from lpi and dpi.
    - The dot grid is rotated by `angle` degrees.
    - Pixels outside the original alpha stay transparent.
    - Dots below `min_radius_px` are dropped rather than drawn as specks.
    """
    if lpi <= 0 or dpi <= 0:
        raise ValueError("lpi and dpi must be positive.")

    img = _load_rgba(png_bytes)
    w, h = img.size
    arr = np.array(img)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0

    # Grayscale tone (0 = light, 1 = dark ink coverage). Use luminance.
    rgb = arr[:, :, :3].astype(np.float32)
    lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
    ink = 1.0 - lum  # darker source -> more ink -> bigger dot

    # Cell size in pixels: one halftone cell per line at the given LPI.
    cell = max(2.0, dpi / lpi)
    max_r = cell / 2.0

    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Output canvas: transparent, dots will be solid black where drawn.
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out_arr = np.zeros((h, w, 4), dtype=np.uint8)

    # Diagonal span so the rotated grid fully covers the image.
    diag = int(math.ceil(math.hypot(w, h))) + int(cell) * 2
    cx, cy = w / 2.0, h / 2.0

    # Iterate grid cell centres in the rotated coordinate frame.
    n_steps = int(math.ceil(diag / cell)) + 1
    start = -n_steps
    end = n_steps

    yy, xx = np.mgrid[0:h, 0:w]

    for gj in range(start, end):
        for gi in range(start, end):
            # Cell centre in rotated frame -> image frame.
            u = gi * cell
            v = gj * cell
            px = cx + u * cos_t - v * sin_t
            py = cy + u * sin_t + v * cos_t
            ipx, ipy = int(round(px)), int(round(py))
            if ipx < 0 or ipx >= w or ipy < 0 or ipy >= h:
                continue

            # Sample tone and coverage in a window around the cell centre.
            half = int(math.ceil(max_r))
            x0, x1 = max(0, ipx - half), min(w, ipx + half + 1)
            y0, y1 = max(0, ipy - half), min(h, ipy + half + 1)
            if x0 >= x1 or y0 >= y1:
                continue

            cell_alpha = alpha[y0:y1, x0:x1]
            if cell_alpha.mean() < 0.5:
                # Cell mostly outside the artwork — keep transparent.
                continue
            cell_ink = ink[y0:y1, x0:x1]
            # Weight tone by coverage so edges do not over/under-ink.
            tone = float((cell_ink * cell_alpha).sum() / max(cell_alpha.sum(), 1e-6))
            tone = min(1.0, max(0.0, tone))

            # Dot radius encodes tone (area-proportional feels closer to press).
            radius = max_r * math.sqrt(tone)
            if radius < min_radius_px:
                continue

            # Rasterise the dot into the output array.
            r_int = int(math.ceil(radius))
            dx0, dx1 = max(0, ipx - r_int), min(w, ipx + r_int + 1)
            dy0, dy1 = max(0, ipy - r_int), min(h, ipy + r_int + 1)
            if dx0 >= dx1 or dy0 >= dy1:
                continue
            sub_x = xx[dy0:dy1, dx0:dx1]
            sub_y = yy[dy0:dy1, dx0:dx1]
            if dot == "square":
                mask = (np.abs(sub_x - ipx) <= radius) & (np.abs(sub_y - ipy) <= radius)
            else:  # round (default)
                mask = ((sub_x - ipx) ** 2 + (sub_y - ipy) ** 2) <= radius ** 2
            # Only draw where the original artwork had coverage.
            mask = mask & (alpha[dy0:dy1, dx0:dx1] >= 0.5)
            out_arr[dy0:dy1, dx0:dx1, 3][mask] = 255  # fully opaque
            # RGB already 0 (black); nothing else to set.

    out = Image.fromarray(out_arr, mode="RGBA")
    return _to_png_bytes(out)

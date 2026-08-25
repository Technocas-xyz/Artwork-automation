"""Post-processing utilities for generated artwork images.

Handles white background removal for DTF print production.
"""

from __future__ import annotations

import io
from collections import deque

from PIL import Image


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
    Only pixels reachable from the edge that are near-white (R, G, B >= 255 - tolerance)
    have their alpha set to 0. Interior white (outlines, highlights) is untouched.

    Parameters
    ----------
    png_bytes:
        Raw PNG bytes of an RGBA image.
    tolerance:
        How far from pure white (255) a channel can be and still count as background.

    Returns
    -------
    bytes
        Processed PNG with transparent background.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    pixels = img.load()
    w, h = img.size
    threshold = 255 - tolerance

    # Track which pixels have been visited
    visited = [[False] * h for _ in range(w)]

    def _is_white(x: int, y: int) -> bool:
        r, g, b, _a = pixels[x, y]
        return r >= threshold and g >= threshold and b >= threshold

    # Collect all border pixels that are white as flood-fill seeds
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

    # BFS flood fill
    while queue:
        cx, cy = queue.popleft()
        # Set this background pixel to transparent
        r, g, b, _a = pixels[cx, cy]
        pixels[cx, cy] = (r, g, b, 0)

        # Expand to 4-connected neighbours
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                if _is_white(nx, ny):
                    visited[nx][ny] = True
                    queue.append((nx, ny))

    # Encode back to PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

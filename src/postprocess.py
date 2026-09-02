"""Post-processing utilities for generated artwork images.

Two responsibilities live here:
  1. White-background removal for DTF print production (is_opaque_white_bg,
     remove_white_background) — used across all workflows.
  2. Deterministic local pixel operations for the Custom Operation workflow
     (black_out, half_tone). These are pixel transforms, not creative redraws:
     sending them through ChatGPT makes the model REDRAW the artwork, losing the
     exact silhouette and knockouts. They run instantly with Pillow/numpy and
     never open a chat or consume a ChatGPT turn.
"""
from __future__ import annotations

import io
import math
from collections import deque

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# White-background removal (shared across workflows)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Similarity comparison (Aspect Ratio Enhancement > Regenerate)
# ---------------------------------------------------------------------------

def _dominant_colours(img: Image.Image, n: int = 5) -> list[tuple[int, int, int]]:
    """Return up to n dominant RGB colours from the visible (opaque) pixels."""
    im = img.convert("RGBA")
    arr = np.array(im)
    # Keep only sufficiently opaque pixels so transparent padding does not skew it.
    mask = arr[:, :, 3] > 128
    rgb = arr[:, :, :3][mask]
    if rgb.size == 0:
        return []
    # Quantise to reduce noise, then count the most common buckets.
    q = (rgb // 24) * 24
    colours, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:n]
    return [tuple(int(c) for c in colours[i]) for i in order]


def _colour_match_count(a: list, b: list, tolerance: int = 32) -> int:
    """Count how many of a's dominant colours have a near match in b."""
    matched = 0
    for ca in a:
        for cb in b:
            if all(abs(x - y) <= tolerance for x, y in zip(ca, cb)):
                matched += 1
                break
    return matched


def _has_transparency(img: Image.Image) -> bool:
    """True if the image contains any pixel that is not fully opaque."""
    alpha = np.array(img)[:, :, 3]
    return bool((alpha < 255).any())


# ---------------------------------------------------------------------------
# Rich geometry / feature extraction (Aspect Ratio Enhancement)
#
# NOTE: We deliberately do NOT compute MAE or any pixel-by-pixel diff anywhere.
# The canvas aspect ratio changes between the original and the regenerated
# result, so the two images are not spatially aligned — an MAE/absdiff would
# report huge differences for visually identical artwork. All comparisons here
# are alignment-tolerant (contour geometry, edge IoU on a resized map,
# dominant-colour overlap, perceptual hash). Do not add MAE later.
# ---------------------------------------------------------------------------

def _content_channels(png_bytes: bytes) -> int:
    """Number of channels in the source image (before RGBA normalisation)."""
    img = Image.open(io.BytesIO(png_bytes))
    return len(img.getbands())


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Canny edge map (uint8 0/255) from a grayscale array."""
    return cv2.Canny(gray, 80, 200)


def extract_features(png_bytes: bytes) -> dict:
    """Extract a rich feature set from a single image, ignoring the background.

    Returns basic dimensions, colour stats over VISIBLE pixels only, an edge
    density, the content bounding box with per-side margins, and contour stats.
    """
    channels = _content_channels(png_bytes)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    total_pixels = int(w * h)
    has_alpha = _has_transparency(img)

    mask = _content_mask(img)  # uint8 0/255, background-independent
    content_bool = mask > 0
    content_area = int(content_bool.sum())

    alpha = arr[:, :, 3]
    transparent_pixel_pct = round(float((alpha < 128).sum()) / total_pixels * 100, 1)

    # Colour stats over VISIBLE (content) pixels only.
    rgb = arr[:, :, :3]
    if content_area > 0:
        visible = rgb[content_bool]
        mean_colour = [int(round(c)) for c in visible.mean(axis=0)]
    else:
        mean_colour = [0, 0, 0]
    dominant = _dominant_colours_with_share(img, mask, n=5)

    # Edges (Canny) and density over the content area.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = _edge_map(gray)
    edges_in_content = int(((edges > 0) & content_bool).sum())
    edge_density = round(edges_in_content / content_area, 4) if content_area else 0.0

    # Bounding box + per-side margins as a share of the canvas.
    if content_area > 0:
        ys, xs = np.where(content_bool)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bx, by = x0, y0
        bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
        margins = {
            "left": round(x0 / w, 3),
            "right": round((w - 1 - x1) / w, 3),
            "top": round(y0 / h, 3),
            "bottom": round((h - 1 - y1) / h, 3),
        }
    else:
        bx = by = 0
        bw, bh = w, h
        margins = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}

    feat = _shape_features(mask)  # aspect, fill, cx, cy, n_contours
    box_area = max(1, bw * bh)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = max((cv2.contourArea(c) for c in contours), default=0.0)
    largest_contour_share = round(largest_area / box_area, 3)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 4) if h else 0.0,
        "total_pixels": total_pixels,
        "channels": channels,
        "has_alpha": has_alpha,
        "transparent_pixel_pct": transparent_pixel_pct,
        "mean_colour": mean_colour,
        "dominant_colours": dominant,  # [{"rgb":[r,g,b],"share":0.xx}, ...]
        "edge_density": edge_density,
        "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
        "margins": margins,
        "n_contours": feat["n_contours"],
        "largest_contour_share": largest_contour_share,
        # Internal fields used by compare_features (not for display).
        "_fill": feat["fill"],
        "_cx": feat["cx"],
        "_cy": feat["cy"],
    }


def _dominant_colours_with_share(img: Image.Image, mask: np.ndarray, n: int = 5) -> list[dict]:
    """Top-n dominant colours over content pixels, each with its share (0..1)."""
    arr = np.array(img.convert("RGBA"))
    content = mask > 0
    rgb = arr[:, :, :3][content]
    if rgb.size == 0:
        return []
    q = (rgb // 24) * 24
    colours, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    total = counts.sum()
    order = np.argsort(counts)[::-1][:n]
    return [
        {"rgb": [int(c) for c in colours[i]], "share": round(float(counts[i]) / float(total), 3)}
        for i in order
    ]


def _content_mask(img: Image.Image) -> np.ndarray:
    """Build a binary content mask (uint8 0/255), ignoring the background.

    - If the image has alpha, content is alpha > 128.
    - Otherwise threshold against the background colour sampled from the four
      corners (pixels close to the corner colour are treated as background).
    """
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]

    if (alpha < 255).any():
        mask = (alpha > 128).astype(np.uint8) * 255
        return mask

    # No transparency — infer background from the corners.
    rgb = arr[:, :, :3].astype(np.int16)
    corners = np.array([
        rgb[0, 0], rgb[0, w - 1], rgb[h - 1, 0], rgb[h - 1, w - 1],
    ], dtype=np.int16)
    bg = np.median(corners, axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    # Pixels far from the background colour are content.
    mask = (dist > 40).astype(np.uint8) * 255
    return mask


def _shape_features(mask: np.ndarray) -> dict:
    """Extract geometric features from a content mask using OpenCV."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"aspect": 0.0, "fill": 0.0, "cx": 0.5, "cy": 0.5, "n_contours": 0}

    # Bounding box over ALL content.
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    aspect = bw / bh

    box_area = bw * bh
    content_area = float((mask > 0).sum())
    fill = content_area / box_area if box_area else 0.0

    # Centroid as a fraction of the bounding box.
    cx = (float(xs.mean()) - x0) / bw
    cy = (float(ys.mean()) - y0) / bh

    # Significant contours: those covering >0.5% of the total content area.
    total = max(content_area, 1.0)
    significant = [c for c in contours if cv2.contourArea(c) > 0.005 * total]

    return {
        "aspect": aspect,
        "fill": min(1.0, fill),
        "cx": cx,
        "cy": cy,
        "n_contours": len(significant),
    }


def _shape_similarity(a: dict, b: dict) -> int:
    """Compare two shape-feature dicts and return a 0-100 similarity percentage."""
    # Aspect ratio: relative difference, capped.
    aa, ba = a["aspect"], b["aspect"]
    if aa <= 0 or ba <= 0:
        aspect_score = 0.0
    else:
        aspect_score = 1.0 - min(1.0, abs(aa - ba) / max(aa, ba))

    # Fill fraction: absolute difference (both 0..1).
    fill_score = 1.0 - min(1.0, abs(a["fill"] - b["fill"]))

    # Centroid: Euclidean distance in box-fraction space (max ~1.414).
    cdist = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
    centroid_score = 1.0 - min(1.0, cdist / 0.5)  # 0.5 box-fraction = fully off

    # Contour count: relative difference.
    na, nb = a["n_contours"], b["n_contours"]
    if na == 0 and nb == 0:
        contour_score = 1.0
    else:
        contour_score = 1.0 - min(1.0, abs(na - nb) / max(na, nb, 1))

    # Weighted blend — aspect and fill (proportions) matter most for "layout".
    score = (0.35 * aspect_score + 0.30 * fill_score
             + 0.20 * centroid_score + 0.15 * contour_score)
    return int(round(max(0.0, min(1.0, score)) * 100))


def _bg_word(has_alpha: bool) -> str:
    return "transparent" if has_alpha else "opaque"


def _edge_iou(orig_bytes: bytes, gen_bytes: bytes, size: int = 256) -> float:
    """Compare two edge maps by IoU on a shared, aspect-preserving canvas.

    Both edge maps are cropped to their content bounding box, then resized
    PRESERVING ASPECT RATIO and centred (letterboxed) onto a common square
    canvas. Squashing differently-shaped content directly to a square distorts
    each differently and makes the edges stop lining up. IoU (not absdiff) is
    used so a one-pixel offset does not destroy the score, and a 1px dilation is
    applied so near-miss edges still overlap.
    """
    def _edges_cropped(b: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(b)).convert("RGBA")
        mask = _content_mask(img)
        arr = np.array(img)
        gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(gray, 80, 200)
        e = e & ((mask > 0).astype(np.uint8) * 255)  # edges within content only
        # Crop to the content bounding box (based on the mask).
        ys, xs = np.where(mask > 0)
        if ys.size and xs.size:
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            e = e[y0:y1 + 1, x0:x1 + 1]
        return e

    def _fit_letterbox(e: np.ndarray, size: int) -> np.ndarray:
        """Resize preserving aspect ratio onto a size x size canvas, centred."""
        h, w = e.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((size, size), dtype=np.uint8)
        scale = min(size / w, size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(e, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((size, size), dtype=np.uint8)
        ox, oy = (size - nw) // 2, (size - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        # Dilate ~1px so a small offset still counts as overlap.
        return cv2.dilate(canvas, np.ones((3, 3), np.uint8), iterations=1)

    ea = _fit_letterbox(_edges_cropped(orig_bytes), size) > 0
    eb = _fit_letterbox(_edges_cropped(gen_bytes), size) > 0
    inter = int((ea & eb).sum())
    union = int((ea | eb).sum())
    print(f"[edge_iou] baseline_edges={int(ea.sum())} result_edges={int(eb.sum())} "
          f"intersection={inter} union={union} iou={inter/union if union else 1.0:.3f}")
    if union == 0:
        return 1.0
    return inter / union


def _colour_axis(orig_feat: dict, gen_feat: dict) -> int:
    """Colour similarity: dominant-colour overlap + mean-colour distance."""
    oc = [d["rgb"] for d in orig_feat.get("dominant_colours", [])]
    gc = [d["rgb"] for d in gen_feat.get("dominant_colours", [])]
    if oc and gc:
        matches = _colour_match_count(oc, gc, tolerance=32)
        overlap = matches / max(len(oc), 1)
    else:
        overlap = 0.0

    om, gm = orig_feat.get("mean_colour", [0, 0, 0]), gen_feat.get("mean_colour", [0, 0, 0])
    mean_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(om, gm)))
    mean_score = 1.0 - min(1.0, mean_dist / 441.67)  # 441.67 = max RGB distance

    score = 0.6 * overlap + 0.4 * mean_score
    return int(round(max(0.0, min(1.0, score)) * 100))


def _detail_axis(orig_bytes: bytes, gen_bytes: bytes) -> int:
    """Detail similarity via phash on content cropped and composited on grey."""
    MID_GREY = (128, 128, 128)
    orig = Image.open(io.BytesIO(orig_bytes)).convert("RGBA")
    gen = Image.open(io.BytesIO(gen_bytes)).convert("RGBA")
    om, gm = _content_mask(orig), _content_mask(gen)

    def _prep(img: Image.Image, mask: np.ndarray) -> Image.Image:
        ys, xs = np.where(mask > 0)
        if ys.size and xs.size:
            img = img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        bg = Image.new("RGBA", img.size, MID_GREY + (255,))
        return Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")

    hash_size = 16
    h1 = imagehash.phash(_prep(orig, om), hash_size=hash_size)
    h2 = imagehash.phash(_prep(gen, gm), hash_size=hash_size)
    bits = hash_size * hash_size
    return int(round((1 - (h1 - h2) / bits) * 100))


def compare_features(orig_feat: dict, gen_feat: dict, orig_bytes: bytes, gen_bytes: bytes) -> dict:
    """Compare original vs generated across FOUR axes, each 0-100.

    Shape  - bbox aspect, fill ratio, centroid, contour count (contour geometry)
    Edges  - IoU of Canny edge maps resized to a common size (NOT absdiff/MAE)
    Colour - dominant-colour overlap within tolerance + mean-colour distance
    Detail - phash on content cropped and composited on neutral grey

    NOTE: no MAE / pixel-by-pixel diff — the canvas ratio differs between the
    two images so they are not aligned; MAE would be meaningless here.
    """
    # Shape reuses the geometry already in the feature dicts.
    shape_pct = _shape_similarity(
        {"aspect": orig_feat["aspect_ratio"], "fill": orig_feat["_fill"],
         "cx": orig_feat["_cx"], "cy": orig_feat["_cy"], "n_contours": orig_feat["n_contours"]},
        {"aspect": gen_feat["aspect_ratio"], "fill": gen_feat["_fill"],
         "cx": gen_feat["_cx"], "cy": gen_feat["_cy"], "n_contours": gen_feat["n_contours"]},
    )
    edges_pct = int(round(_edge_iou(orig_bytes, gen_bytes) * 100))
    colour_pct = _colour_axis(orig_feat, gen_feat)
    detail_pct = _detail_axis(orig_bytes, gen_bytes)

    # Background note (information, not a warning).
    oa, ga = orig_feat["has_alpha"], gen_feat["has_alpha"]
    if oa == ga:
        bg_note = f"Background: {_bg_word(oa)} in both"
    else:
        bg_note = f"Background: {_bg_word(oa)} in original, {_bg_word(ga)} in result"

    return {
        "shape_pct": shape_pct,
        "edges_pct": edges_pct,
        "colour_pct": colour_pct,
        "detail_pct": detail_pct,
        # Backward-compatible headline for any older callers.
        "similarity_pct": detail_pct,
        "colour_matches": _colour_match_count(
            [d["rgb"] for d in orig_feat.get("dominant_colours", [])],
            [d["rgb"] for d in gen_feat.get("dominant_colours", [])], tolerance=32),
        "colour_total": len(orig_feat.get("dominant_colours", [])),
        "background_note": bg_note,
        "background_changed": oa != ga,
    }


def similarity(original_bytes: bytes, generated_bytes: bytes) -> dict:
    """Convenience entry point: extract features for both, then compare.

    Returns the four axis scores plus the per-image feature dicts so callers
    can store and display everything from a single call.
    """
    orig_feat = extract_features(original_bytes)
    gen_feat = extract_features(generated_bytes)
    result = compare_features(orig_feat, gen_feat, original_bytes, generated_bytes)
    result["features_original"] = orig_feat
    result["features_generated"] = gen_feat
    return result

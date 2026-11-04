"""Image comparison and geometric feature analysis (server-side).

Split out of postprocess.py so the client-side agent does not have to import
OpenCV (cv2). These functions run on the SERVER for the Aspect Ratio
Enhancement comparison and the left-panel image info — the agent never imports
this module, which keeps its packaged .exe free of the large cv2 .pyd that
breaks PyInstaller one-file extraction.

Every function here was MOVED UNCHANGED from postprocess.py.
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Weights for the single blended "overall_pct" headline. Detail carries the
# most weight (it is what the client sees — the artwork itself), with shape,
# edges and colour contributing equally. They sum to 1.0. The UI falls back to
# these exact weights if it ever has to compute overall_pct from an older
# stored payload, so keep the two in sync.
# ---------------------------------------------------------------------------
OVERALL_WEIGHT_DETAIL = 0.40
OVERALL_WEIGHT_SHAPE = 0.20
OVERALL_WEIGHT_EDGES = 0.20
OVERALL_WEIGHT_COLOUR = 0.20


# ---------------------------------------------------------------------------
# Perceptual hash (pHash), numpy-only. Replicates imagehash.phash(hash_size=16,
# highfreq_factor=4) exactly using a numpy DCT (no scipy).
# ---------------------------------------------------------------------------

def _dct_1d(a: np.ndarray) -> np.ndarray:
    """Type-II DCT along the last axis (matches scipy.fftpack.dct default)."""
    n = a.shape[-1]
    # Even/odd extension via FFT — standard DCT-II implementation.
    v = np.empty_like(a, dtype=np.float64)
    v[..., : (n + 1) // 2] = a[..., ::2]
    v[..., (n + 1) // 2 :] = a[..., 1::2][..., ::-1]
    V = np.fft.fft(v, axis=-1)
    k = np.arange(n)
    factor = 2.0 * np.exp(-1j * np.pi * k / (2 * n))
    return np.real(V * factor)


def _dct_2d(a: np.ndarray) -> np.ndarray:
    return _dct_1d(_dct_1d(a.astype(np.float64), ).T).T


def _phash_bits(img: Image.Image, hash_size: int = 16, highfreq_factor: int = 4) -> np.ndarray:
    img_size = hash_size * highfreq_factor
    small = img.convert("L").resize((img_size, img_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    dct = _dct_2d(pixels)
    lowfreq = dct[:hash_size, :hash_size]
    med = np.median(lowfreq)
    return lowfreq > med


def _phash_distance(bits_a: np.ndarray, bits_b: np.ndarray) -> int:
    return int(np.count_nonzero(bits_a != bits_b))



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
    # Background = pixels close to the corner colour that are CONNECTED to the
    # image border. A same-coloured region fully enclosed by the artwork (e.g. a
    # white star on a white canvas) is part of the design, not background.
    near_bg = (dist <= 40).astype(np.uint8)
    n, labels = cv2.connectedComponents(near_bg, connectivity=4)
    if n > 1:
        border_labels = np.unique(np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        border_labels = border_labels[border_labels > 0]
        is_bg = np.isin(labels, border_labels)
    else:
        is_bg = np.zeros((h, w), dtype=bool)
    mask = (~is_bg).astype(np.uint8) * 255
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


_NEUTRAL_GREY = 128      # background every image is flattened onto before comparing
_COMPARE_LONG_SIDE = 512  # common working size (long side) for edges / detail
_EDGE_TOL_PX = 3          # an edge counts as matched if within this many px
_PAD = 6                  # grey border so a silhouette on the bbox edge still yields an edge


def _neutral_gray(img_bytes: bytes, long_side: int = _COMPARE_LONG_SIDE):
    """Background-independent grayscale view of an image.

    1. Find the CONTENT mask (alpha for transparent images, corner-colour
       distance for opaque ones).
    2. Flatten onto one neutral grey: transparent pixels are composited on it
       and, for opaque images, every non-content pixel is REPLACED by it. So art
       on black, on white, or on transparent all become the same picture.
    3. Crop to the content bbox, resize the IMAGE (area when shrinking, Lanczos
       when enlarging; aspect kept) to `long_side`, then add a small grey pad.

    Returns (gray uint8, mask bool) of identical shape. Resizing the image first
    and detecting edges AFTERWARDS keeps thin lines alive (resizing an edge map
    with NEAREST aliases them away)."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    mask = _content_mask(img) > 0
    arr = np.array(img).astype(np.float32)
    a = arr[:, :, 3:4] / 255.0
    rgb = arr[:, :, :3] * a + _NEUTRAL_GREY * (1.0 - a)
    rgb[~mask] = _NEUTRAL_GREY
    gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    ys, xs = np.where(mask)
    if ys.size:
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        gray, mask = gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    h, w = gray.shape[:2]
    scale = long_side / float(max(h, w))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    gray = cv2.resize(gray, (nw, nh), interpolation=interp)
    mask = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0

    gray = cv2.copyMakeBorder(gray, _PAD, _PAD, _PAD, _PAD, cv2.BORDER_CONSTANT,
                              value=_NEUTRAL_GREY)
    mask = cv2.copyMakeBorder(mask.astype(np.uint8), _PAD, _PAD, _PAD, _PAD,
                              cv2.BORDER_CONSTANT, value=0) > 0
    return gray, mask


def _fit_to(gray: np.ndarray, mask: np.ndarray, shape: tuple):
    """Resize (aspect kept) and centre `gray`/`mask` onto a canvas of `shape`
    (h, w), padding with the neutral grey, so two images share one frame."""
    th, tw = shape
    h, w = gray.shape[:2]
    if (h, w) == (th, tw):
        return gray, mask
    s = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LANCZOS4
    g = cv2.resize(gray, (nw, nh), interpolation=interp)
    m = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    cg = np.full((th, tw), _NEUTRAL_GREY, dtype=np.uint8)
    cm = np.zeros((th, tw), dtype=np.uint8)
    oy, ox = (th - nh) // 2, (tw - nw) // 2
    cg[oy:oy + nh, ox:ox + nw] = g
    cm[oy:oy + nh, ox:ox + nw] = m
    return cg, cm > 0


def _prep_pair(orig_bytes: bytes, gen_bytes: bytes):
    """(gray_a, mask_a, gray_b, mask_b) on one common frame (A's size)."""
    ga, ma = _neutral_gray(orig_bytes)
    gb, mb = _neutral_gray(gen_bytes)
    gb, mb = _fit_to(gb, mb, ga.shape[:2])
    return ga, ma, gb, mb


def _canny(gray: np.ndarray) -> np.ndarray:
    """Canny on a lightly smoothed image (suppresses JPEG ringing / noise)."""
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 110) > 0


def _edge_f1_arrays(ga: np.ndarray, gb: np.ndarray) -> float:
    """Edge agreement as an F1 (precision/recall) score with a pixel tolerance.

    precision = share of A's edge pixels within _EDGE_TOL_PX of a B edge,
    recall    = share of B's edge pixels within _EDGE_TOL_PX of an A edge.
    A chamfer / distance-transform match, so a 1-3px shift, softness or a small
    resize does not destroy the score the way strict pixel IoU does."""
    ea, eb = _canny(ga), _canny(gb)
    na, nb = int(ea.sum()), int(eb.sum())
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    # distanceTransform measures distance to the nearest ZERO pixel, so feed it
    # the inverse of the edge map.
    dist_to_b = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_a = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((dist_to_b[ea] <= _EDGE_TOL_PX).sum()) / na
    recall = float((dist_to_a[eb] <= _EDGE_TOL_PX).sum()) / nb
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    print(f"[edge_f1] edges_a={na} edges_b={nb} precision={precision:.3f} "
          f"recall={recall:.3f} f1={f1:.3f}")
    return f1


def _edge_iou(orig_bytes: bytes, gen_bytes: bytes, size: int = 256) -> float:
    """Edge similarity 0..1 (name kept for callers). Now an F1 over Canny edges
    detected AFTER both images are background-neutralised, content-cropped and
    resized (as images) to a common ~512px size; see _edge_f1_arrays."""
    ga, _ma, gb, _mb = _prep_pair(orig_bytes, gen_bytes)
    return _edge_f1_arrays(ga, gb)


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


def _ssim(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    """Mean SSIM of two same-size uint8 grayscale images (Gaussian 11x11, sigma
    1.5), averaged over `region` when given (so shared flat background does not
    inflate it). Clamped to 0..1."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def g(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = g(a), g(b)
    saa = g(a * a) - mu_a * mu_a
    sbb = g(b * b) - mu_b * mu_b
    sab = g(a * b) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    if region is not None and region.any():
        val = float(ssim_map[region].mean())
    else:
        val = float(ssim_map.mean())
    return max(0.0, min(1.0, val))


def _detail_axis(orig_bytes: bytes, gen_bytes: bytes) -> int:
    """Detail similarity: the better of pHash(16x16) and SSIM, both computed on
    the background-neutralised, content-cropped, same-size grayscale images.
    pHash catches structure; SSIM tolerates the normal softness of a GPT
    regeneration. Taking the max means softness is not punished."""
    ga, ma, gb, mb = _prep_pair(orig_bytes, gen_bytes)
    return _detail_from_arrays(ga, ma, gb, mb)


def _detail_from_arrays(ga, ma, gb, mb) -> int:
    hash_size = 16
    b1 = _phash_bits(Image.fromarray(ga), hash_size=hash_size)
    b2 = _phash_bits(Image.fromarray(gb), hash_size=hash_size)
    phash_pct = (1 - _phash_distance(b1, b2) / (hash_size * hash_size)) * 100
    # Region = union of both contents, grown a little so the silhouette counts.
    region = cv2.dilate((ma | mb).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ssim_pct = _ssim(ga, gb, region) * 100
    print(f"[detail] phash={phash_pct:.1f} ssim={ssim_pct:.1f}")
    return int(round(max(phash_pct, ssim_pct)))


def _bbox_aspect(feat: dict) -> float:
    """Aspect (w/h) of the CONTENT bounding box, not the canvas: the canvas
    changes between original and regeneration while the artwork does not."""
    bb = feat.get("bbox") or {}
    bw, bh = bb.get("w", 0), bb.get("h", 0)
    if bw > 0 and bh > 0:
        return bw / bh
    return feat.get("aspect_ratio", 0.0)


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
        {"aspect": _bbox_aspect(orig_feat), "fill": orig_feat["_fill"],
         "cx": orig_feat["_cx"], "cy": orig_feat["_cy"], "n_contours": orig_feat["n_contours"]},
        {"aspect": _bbox_aspect(gen_feat), "fill": gen_feat["_fill"],
         "cx": gen_feat["_cx"], "cy": gen_feat["_cy"], "n_contours": gen_feat["n_contours"]},
    )
    # Edges and Detail share one background-neutral, content-cropped, common-size
    # view of the two images.
    ga, ma, gb, mb = _prep_pair(orig_bytes, gen_bytes)
    edges_pct = int(round(_edge_f1_arrays(ga, gb) * 100))
    colour_pct = _colour_axis(orig_feat, gen_feat)
    detail_pct = _detail_from_arrays(ga, ma, gb, mb)

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
        # Backward-compatible headline for any older callers (unchanged).
        "similarity_pct": detail_pct,
        # Genuine blended headline across all four axes, so a weak axis (e.g.
        # low Edges) is visible in the single number operators read.
        "overall_pct": int(round(
            OVERALL_WEIGHT_DETAIL * detail_pct
            + OVERALL_WEIGHT_SHAPE * shape_pct
            + OVERALL_WEIGHT_EDGES * edges_pct
            + OVERALL_WEIGHT_COLOUR * colour_pct
        )),
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


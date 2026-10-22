"""Pixel Geometry (Before -> Generate -> After) — LOCAL deterministic measurement.

Everything measurable here is measured with OpenCV / Pillow / numpy ONLY, exactly
like src/printready.py and src/colorway.py: it runs SERVER-SIDE (cv2 lives on the
server; the packaged agent has none). Nothing is estimated — any field we cannot
compute from the pixels (or that the operator did not supply) is returned as
``None`` so the caller renders "N/A" and stores ``null`` in the master JSON.

Two public entry points:

  * ``measure_geometry(image_bytes, *, inputs, role, source_before=None)`` returns
    the BEFORE block (role="before") or the AFTER block (role="after"). The AFTER
    block also carries the before-vs-after comparison + QA statuses; it needs the
    already-computed ``source_before`` so similarity is measured against the real
    source, never against what the AI was asked for.

  * ``build_master(artwork_id, before, after)`` assembles the single master object
    {artwork_id, before, after, autofix, approval} the UI offers as "Download JSON".

ALL thresholds and profile values live in ONE constant block below, commented.
cv2 is imported lazily inside the functions that need it, so importing this module
never requires OpenCV.
"""

from __future__ import annotations

import hashlib
import io
import math
from typing import Any

import numpy as np
from PIL import Image


# ===========================================================================
# CONSTANT BLOCK — DTF profile + every threshold. One place, on purpose.
# Change a number here and both the measurement and the PASS/WARN/FAIL verdicts
# move with it. Nothing below reads a magic number that is not defined here.
# ===========================================================================

DTF_PROFILE_ID = "DECOINKS_DTF_STANDARD_V1"      # dtf_profile_id stamped on every BEFORE block
GENERATION_PROMPT_VERSION = "GEOMETRY_GENERATE_V1"  # generation_prompt_version stamped on BEFORE
PRINT_METHOD = "DTF"                              # print_method — fixed for this workflow

# --- Operator-input DEFAULTS (used when the form leaves a value blank). The
#     server passes the operator's choices through; these are the profile
#     fallbacks so a value is never silently empty when it has a sensible DTF
#     default. Optional free-text inputs (garment style/colour) have NO default
#     and show N/A when blank. ---
DEFAULT_RECONSTRUCTION_MODE = "EXACT_RECONSTRUCTION"   # EXACT_RECONSTRUCTION | DTF_OPTIMIZED | REDESIGN
DEFAULT_PRINT_LOCATION = "Full Back"
DEFAULT_TARGET_PRINT_WIDTH_IN = 12.5
DEFAULT_TARGET_DPI = 300
DEFAULT_ASPECT_RATIO_POLICY = "LOCKED"                 # LOCKED | FLEXIBLE
DEFAULT_BACKGROUND_REQUIREMENT = "Transparent"
DEFAULT_DISTRESS_POLICY = "Preserve"                   # Preserve | Remove

# --- Profile numeric limits (feed the QA thresholds + the target geometry) ---
MAX_RATIO_VARIANCE_PCT = 0.5        # max_ratio_variance_pct — allowed aspect drift before FAIL
MAX_TRANSPARENT_MARGIN_IN = 0.10    # max_transparent_margin_in — padding budget per edge

# --- BACKGROUND detection ---
# A pixel counts as "transparent" below this alpha; used for alpha background
# detection and the content mask on images that have an alpha channel.
ALPHA_TRANSPARENT_MAX = 0            # alpha == 0 is transparent (content mask uses alpha > 0)
# Border sampling for opaque images: how thick a frame (fraction of the smaller
# side) to sample when guessing the background colour + its variance.
BORDER_FRAC = 0.03
# Opaque background classification from the sampled border:
BG_SOLID_STDDEV_MAX = 10.0           # per-channel stddev <= this -> a flat colour
BG_WHITE_MIN = 245                   # mean channel >= this (and flat) -> White
BG_COMPLEX_STDDEV = 10.0             # above BG_SOLID_STDDEV_MAX -> Complex (varied) background
# Opaque content mask: a pixel is "content" when it differs from the detected
# border background colour by more than this L2 distance (0..441 for 8-bit RGB).
CONTENT_COLOUR_DISTANCE = 40.0

# --- CONNECTED COMPONENTS ---
COMPONENT_MIN_AREA_PX = 50           # object_count: components at/above this area are "objects"
MICRO_COMPONENT_MAX_AREA_PX = 10     # micro_component_count: components strictly below this area
ISOLATED_COMPONENT_MAX_AREA_PX = 50  # isolated_component_count: small detached specks (<= this)

# --- ALPHA histogram buckets (edges inclusive-low). Used for the AFTER alpha
#     analysis bars and the stray/semi-transparency counts. ---
# alpha_zero: ==0 ; 1_10: 1..10 ; 11_50: 11..50 ; 51_200: 51..200 ;
# 201_254: 201..254 ; 255: ==255
LOW_ALPHA_STRAY_MAX = 50             # low_alpha_outside_intent: alpha 1..50 outside the content
SOLID_CONTENT_ALPHA_MIN = 200        # "solid content" = alpha >= this (dilated to form intent mask)
INTENT_DILATE_PX = 3                 # dilate the solid-content mask by this many px for "intent"

# --- EDGE contamination / halo (only meaningful with transparency; else N/A,
#     the same rule Print Ready uses for halo). ---
EDGE_BAND_PX = 2                     # width of the semi-transparent edge band to inspect
HALO_LIGHT_MIN = 200                 # halo_white_px: edge-band pixel channels all >= this
HALO_DARK_MAX = 55                   # halo_dark_px: edge-band pixel channels all <= this
EDGE_CONTAM_WARN_PCT = 2.0           # edge_contamination_pct WARN threshold
EDGE_CONTAM_FAIL_PCT = 5.0           # edge_contamination_pct FAIL threshold

# --- FINE DETAIL ---
MIN_LINE_WIDTH_PERCENTILE = 5        # minimum_line_width_px: low percentile of the stroke-width map
MIN_LINE_WIDTH_WARN_PX = 3.0         # thinner than this at target res -> WARN
MIN_LINE_WIDTH_FAIL_PX = 1.5         # thinner than this -> FAIL

# --- CLIPPING: content touching a canvas edge (distance 0). ---
CLIP_EDGE_TOUCH_PX = 0

# --- QA thresholds for the similarity-based statuses (percent, 0..100) ---
GEOMETRY_SIM_WARN_PCT = 90.0         # geometry_status: below WARN -> WARN, below FAIL -> FAIL
GEOMETRY_SIM_FAIL_PCT = 75.0
OVERALL_SIM_WARN_PCT = 90.0
OVERALL_SIM_FAIL_PCT = 75.0

# --- RESOLUTION: effective dpi vs target dpi (fraction). ---
DPI_WARN_FRAC = 0.90                 # effective dpi below 90% of target -> WARN
DPI_FAIL_FRAC = 0.75                 # below 75% -> FAIL

# --- Weights for overall_geometry_similarity_pct (weighted mean of available
#     component similarities; missing ones drop out and the rest renormalise). ---
GEOMETRY_SIM_WEIGHTS = {
    "bbox_similarity_pct": 0.40,
    "object_position_similarity_pct": 0.25,
    "object_scale_similarity_pct": 0.25,
    "source_ratio_match_pct": 0.10,   # 100 - source_ratio_difference_pct
}

# Status vocabulary.
PASS, WARN, FAIL, NA = "PASS", "WARN", "FAIL", "N/A"
_STATUS_RANK = {PASS: 0, WARN: 1, FAIL: 2}   # for "worst of" (N/A excluded)


# ---------------------------------------------------------------------------
# The exact ordered field lists. The UI renders every one of these, in order,
# with N/A for a null. Keeping the lists here means the server, the master JSON,
# and the tests all agree on the schema in one place.
# ---------------------------------------------------------------------------

BEFORE_FIELDS = [
    "artwork_id", "source_filename",
    "source_width_px", "source_height_px", "source_aspect_ratio", "source_orientation",
    "source_has_alpha", "source_background_type",
    "source_bbox_x", "source_bbox_y", "source_bbox_width", "source_bbox_height",
    "source_padding_left", "source_padding_right", "source_padding_top", "source_padding_bottom",
    "source_center_x_pct", "source_center_y_pct",
    "reconstruction_mode", "print_method", "print_location", "garment_style", "garment_color",
    "target_print_width_in", "target_print_height_in", "target_dpi",
    "target_width_px", "target_height_px", "target_aspect_ratio",
    "aspect_ratio_policy", "max_ratio_variance_pct",
    "crop_allowed", "stretch_allowed", "reposition_allowed",
    "background_requirement", "max_transparent_margin_in",
    "white_elements_policy", "black_elements_policy", "semi_transparency_allowed",
    "semi_transparency_types", "distress_present", "distress_policy",
    "text_present", "text_must_match_exactly",
    "object_count", "text_region_count", "important_object_geometry",
    "generation_prompt_version", "dtf_profile_id",
]

AFTER_FIELDS = [
    "output_width_px", "output_height_px", "output_aspect_ratio", "output_orientation", "output_has_alpha",
    "effective_dpi_x", "effective_dpi_y",
    "output_bbox_x", "output_bbox_y", "output_bbox_width", "output_bbox_height",
    "padding_left", "padding_right", "padding_top", "padding_bottom",
    "padding_left_in", "padding_right_in", "padding_top_in", "padding_bottom_in",
    "output_center_x_pct", "output_center_y_pct",
    "alpha_zero_count", "alpha_1_10_count", "alpha_11_50_count",
    "alpha_51_200_count", "alpha_201_254_count", "alpha_255_count",
    "low_alpha_outside_intent_count", "semi_transparent_inside_effects_count",
    "edge_pixel_count", "edge_contamination_count", "edge_contamination_pct",
    "halo_white_px", "halo_dark_px", "fringe_pixel_count",
    "connected_component_count", "isolated_component_count", "micro_component_count",
    "smallest_component_area_px",
    "minimum_line_width_px", "minimum_line_width_in", "minimum_text_stroke_px", "minimum_hole_size_px",
    "distance_left_edge_px", "distance_right_edge_px", "distance_top_edge_px", "distance_bottom_edge_px",
    "clipping_detected",
    "source_ratio_difference_pct", "bbox_similarity_pct",
    "object_position_similarity_pct", "object_scale_similarity_pct",
    "text_geometry_similarity_pct", "overall_geometry_similarity_pct",
    "text_ocr_match_pct", "semantic_similarity_pct",
    "missing_objects_count", "extra_objects_count",
    "background_transparency_status", "resolution_status", "geometry_status",
    "alpha_status", "halo_status", "stray_pixel_status", "fine_detail_status",
    "clipping_status", "text_status", "semantic_status", "overall_qa_status",
]

# V1 "not implemented yet" blocks — every field listed, every value None, so the
# UI shows the full shape with N/A and the master JSON records null.
AUTOFIX_FIELDS = [
    "autofix_required", "autofix_count", "autofix_trim_padding", "autofix_remove_strays",
    "autofix_clean_halo", "autofix_alpha_cleanup", "autofix_resize", "autofix_upscale",
    "autofix_strengthen_details", "qa_rerun_required", "qa_rerun_count",
    "final_after_autofix_status",
]
APPROVAL_FIELDS = [
    "approved_version_id", "customer_approved", "approval_timestamp",
    "approved_print_width_in", "approved_print_height_in", "approved_file_hash",
    "production_locked", "final_machine_qa_status", "production_master_version",
    "artwork_vault_status", "gangsheet_ready",
]


# ===========================================================================
# Small helpers
# ===========================================================================

def _round(v: Any, nd: int = 2):
    if v is None:
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _orientation(w: int, h: int) -> str:
    if w == h:
        return "Square"
    return "Landscape" if w > h else "Portrait"


def _load_rgba(image_bytes: bytes) -> "Image.Image":
    return Image.open(io.BytesIO(image_bytes)).convert("RGBA")


def _has_real_alpha(pil_rgba: "Image.Image", original_mode: str) -> bool:
    """True only if the ORIGINAL mode carried alpha AND some pixel is < 255.
    A JPEG (no alpha) is False even though we converted to RGBA for measuring."""
    if "A" not in original_mode and original_mode not in ("P", "PA"):
        # P can carry transparency via a palette; everything else without A is opaque.
        if original_mode != "P":
            return False
    a = np.asarray(pil_rgba)[:, :, 3]
    return bool((a < 255).any())


def _content_mask(arr: np.ndarray, has_alpha: bool) -> "np.ndarray | None":
    """Boolean HxW mask of 'content' pixels.

    Transparent-capable images: content = alpha > ALPHA_TRANSPARENT_MAX.
    Opaque images: sample the border to estimate the flat background colour and
    mark pixels that differ from it by more than CONTENT_COLOUR_DISTANCE.
    Returns None if a mask cannot be derived (e.g. a fully flat image)."""
    h, w = arr.shape[:2]
    if has_alpha:
        return arr[:, :, 3] > ALPHA_TRANSPARENT_MAX
    rgb = arr[:, :, :3].astype(np.float32)
    bg = _border_bg_colour(arr)
    if bg is None:
        return None
    dist = np.sqrt(((rgb - np.array(bg, dtype=np.float32)) ** 2).sum(axis=2))
    mask = dist > CONTENT_COLOUR_DISTANCE
    return mask if mask.any() else None


def _border_pixels(arr: np.ndarray) -> np.ndarray:
    """The RGB pixels in a thin frame around the image, stacked Nx3."""
    h, w = arr.shape[:2]
    bt = max(1, int(round(min(h, w) * BORDER_FRAC)))
    rgb = arr[:, :, :3]
    top = rgb[:bt, :, :].reshape(-1, 3)
    bottom = rgb[-bt:, :, :].reshape(-1, 3)
    left = rgb[:, :bt, :].reshape(-1, 3)
    right = rgb[:, -bt:, :].reshape(-1, 3)
    return np.concatenate([top, bottom, left, right], axis=0)


def _border_bg_colour(arr: np.ndarray):
    px = _border_pixels(arr).astype(np.float32)
    if px.size == 0:
        return None
    return tuple(px.mean(axis=0).tolist())


def _bbox_from_mask(mask: np.ndarray):
    """(x, y, width, height) of the tight content box, or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)


# ===========================================================================
# BEFORE — measure the SOURCE
# ===========================================================================

def _resolve_inputs(inputs: dict | None) -> dict:
    """Fill operator inputs with the DTF profile defaults; blank optional
    free-text stays None so it renders N/A."""
    inp = dict(inputs or {})

    def pick(key, default):
        v = inp.get(key)
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return v

    return {
        "reconstruction_mode": pick("reconstruction_mode", DEFAULT_RECONSTRUCTION_MODE),
        "print_location": pick("print_location", DEFAULT_PRINT_LOCATION),
        "garment_style": (inp.get("garment_style") or "").strip() or None,   # optional -> N/A
        "garment_color": (inp.get("garment_color") or "").strip() or None,   # optional -> N/A
        "target_print_width_in": float(pick("target_print_width_in", DEFAULT_TARGET_PRINT_WIDTH_IN)),
        "target_dpi": int(pick("target_dpi", DEFAULT_TARGET_DPI)),
        "aspect_ratio_policy": pick("aspect_ratio_policy", DEFAULT_ASPECT_RATIO_POLICY),
        "background_requirement": pick("background_requirement", DEFAULT_BACKGROUND_REQUIREMENT),
        "white_elements_policy": (inp.get("white_elements_policy") or "").strip() or None,
        "black_elements_policy": (inp.get("black_elements_policy") or "").strip() or None,
        "semi_transparency_allowed": inp.get("semi_transparency_allowed")
        if inp.get("semi_transparency_allowed") not in (None, "") else None,
        "distress_policy": pick("distress_policy", DEFAULT_DISTRESS_POLICY),
    }


def _source_background_type(arr: np.ndarray, has_alpha: bool):
    """Transparent / White / Solid colour / Complex, or None if undecidable."""
    if has_alpha:
        return "Transparent"
    px = _border_pixels(arr).astype(np.float32)
    if px.size == 0:
        return None
    std = float(px.std(axis=0).mean())
    mean = float(px.mean())
    if std <= BG_SOLID_STDDEV_MAX:
        return "White" if mean >= BG_WHITE_MIN else "Solid colour"
    return "Complex"


def measure_source(image_bytes: bytes, *, artwork_id: str, source_filename: str,
                   inputs: dict | None) -> dict:
    """Build the full BEFORE block for the source artwork."""
    out = {k: None for k in BEFORE_FIELDS}
    out["artwork_id"] = artwork_id
    out["source_filename"] = source_filename
    out["generation_prompt_version"] = GENERATION_PROMPT_VERSION
    out["dtf_profile_id"] = DTF_PROFILE_ID
    out["print_method"] = PRINT_METHOD
    out["max_ratio_variance_pct"] = MAX_RATIO_VARIANCE_PCT
    out["max_transparent_margin_in"] = MAX_TRANSPARENT_MARGIN_IN

    inp = _resolve_inputs(inputs)
    out["reconstruction_mode"] = inp["reconstruction_mode"]
    out["print_location"] = inp["print_location"]
    out["garment_style"] = inp["garment_style"]
    out["garment_color"] = inp["garment_color"]
    out["target_print_width_in"] = _round(inp["target_print_width_in"], 2)
    out["target_dpi"] = inp["target_dpi"]
    out["aspect_ratio_policy"] = inp["aspect_ratio_policy"]
    out["background_requirement"] = inp["background_requirement"]
    out["white_elements_policy"] = inp["white_elements_policy"]
    out["black_elements_policy"] = inp["black_elements_policy"]
    out["semi_transparency_allowed"] = inp["semi_transparency_allowed"]
    out["distress_policy"] = inp["distress_policy"]

    # Mode drives the geometry permissions. EXACT => nothing may move.
    mode = (inp["reconstruction_mode"] or "").upper()
    exact = mode == "EXACT_RECONSTRUCTION"
    out["crop_allowed"] = (not exact)
    out["stretch_allowed"] = (not exact)
    out["reposition_allowed"] = (not exact)
    # text_must_match: Yes if EXACT; text_present needs vision -> N/A in V1.
    out["text_must_match_exactly"] = True if exact else False
    out["text_present"] = None
    # V1 vision-dependent fields: explicitly N/A.
    out["semi_transparency_types"] = None
    out["distress_present"] = None
    out["text_region_count"] = None
    out["important_object_geometry"] = None

    pil = _load_rgba(image_bytes)
    original_mode = Image.open(io.BytesIO(image_bytes)).mode
    arr = np.asarray(pil)
    h, w = arr.shape[:2]
    has_alpha = _has_real_alpha(pil, original_mode)

    out["source_width_px"] = int(w)
    out["source_height_px"] = int(h)
    out["source_aspect_ratio"] = _round(w / h, 4) if h else None
    out["source_orientation"] = _orientation(w, h)
    out["source_has_alpha"] = bool(has_alpha)
    out["source_background_type"] = _source_background_type(arr, has_alpha)

    mask = _content_mask(arr, has_alpha)
    bbox = _bbox_from_mask(mask) if mask is not None else None
    if bbox:
        bx, by, bw, bh = bbox
        out["source_bbox_x"] = bx
        out["source_bbox_y"] = by
        out["source_bbox_width"] = bw
        out["source_bbox_height"] = bh
        out["source_padding_left"] = bx
        out["source_padding_right"] = int(w - (bx + bw))
        out["source_padding_top"] = by
        out["source_padding_bottom"] = int(h - (by + bh))
        out["source_center_x_pct"] = _round(100.0 * (bx + bw / 2.0) / w, 2) if w else None
        out["source_center_y_pct"] = _round(100.0 * (by + bh / 2.0) / h, 2) if h else None
        # target print geometry derives from the bbox proportions.
        tw_in = inp["target_print_width_in"]
        th_in = tw_in * (bh / bw) if bw else None
        out["target_print_height_in"] = _round(th_in, 2) if th_in is not None else None
        if th_in is not None:
            out["target_width_px"] = int(round(tw_in * inp["target_dpi"]))
            out["target_height_px"] = int(round(th_in * inp["target_dpi"]))
            tar_ratio = (out["target_width_px"] / out["target_height_px"]
                         if out["target_height_px"] else None)
            out["target_aspect_ratio"] = _round(tar_ratio, 4)
    # object_count: connected components above a size threshold (COMPONENT ESTIMATE).
    if mask is not None:
        comps = _components(mask)
        out["object_count"] = int(sum(1 for a in comps if a >= COMPONENT_MIN_AREA_PX))

    return out


# ===========================================================================
# Connected components / fine detail (cv2 lazily)
# ===========================================================================

def _components(mask: np.ndarray) -> list[int]:
    """Areas (px) of connected components of a boolean mask, 8-connectivity."""
    import cv2
    m = mask.astype(np.uint8)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    # label 0 is background.
    return [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]


def _min_stroke_width_px(mask: np.ndarray):
    """Minimum line/stroke width via the distance transform: 2 x a low percentile
    of the interior distance values (the thinnest strokes). None if no content."""
    import cv2
    m = mask.astype(np.uint8)
    if not m.any():
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    interior = dist[dist > 0]
    if interior.size == 0:
        return None
    val = float(np.percentile(interior, MIN_LINE_WIDTH_PERCENTILE)) * 2.0
    return val


def _hole_sizes(mask: np.ndarray):
    """Smallest transparent HOLE fully inside the content (px area), or None.
    Holes = background components that do not touch the image border."""
    import cv2
    bg = (~mask).astype(np.uint8)
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(bg, connectivity=8)
    h, w = mask.shape
    smallest = None
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]; y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]; ch = stats[i, cv2.CC_STAT_HEIGHT]
        touches = (x == 0 or y == 0 or (x + cw) >= w or (y + ch) >= h)
        if touches:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        smallest = area if smallest is None else min(smallest, area)
    return smallest


# ===========================================================================
# AFTER — measure the OUTPUT + compare to the source
# ===========================================================================

def _mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _norm_bbox(bbox, w, h):
    """bbox (x,y,bw,bh) -> normalised (x0,y0,x1,y1) in 0..1."""
    x, y, bw, bh = bbox
    return (x / w, y / h, (x + bw) / w, (y + bh) / h)


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def _mask_iou_resized(src_mask: np.ndarray, out_mask: np.ndarray, size: int = 256) -> float:
    """IoU of the two content masks resized to a common square — a shape-overlap
    measure independent of absolute resolution."""
    import cv2
    a = cv2.resize(src_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    b = cv2.resize(out_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return (inter / union) if union else 0.0


def _phash(pil_rgba: "Image.Image") -> int:
    """A 64-bit perceptual hash (DCT-based) — clearly labelled 'pHash' upstream.
    Grey, 32x32, DCT, low-freq 8x8, median threshold."""
    import cv2
    g = cv2.cvtColor(np.asarray(pil_rgba.convert("RGB")), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)
    low = d[:8, :8]
    med = np.median(low[1:])     # skip the DC term
    bits = (low > med).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def _phash_similarity(a: int, b: int) -> float:
    ham = bin(a ^ b).count("1")
    return 100.0 * (1.0 - ham / 64.0)


def _status_from_pct(pct, warn, fail):
    if pct is None:
        return NA
    if pct < fail:
        return FAIL
    if pct < warn:
        return WARN
    return PASS


def _worst(statuses) -> str:
    real = [s for s in statuses if s in _STATUS_RANK]
    if not real:
        return NA
    return max(real, key=lambda s: _STATUS_RANK[s])


def measure_output(image_bytes: bytes, *, before: dict) -> dict:
    """Build the full AFTER block: measure the OUTPUT file, then compare it to
    the already-measured SOURCE (`before`). Never estimates from intent."""
    out = {k: None for k in AFTER_FIELDS}

    pil = _load_rgba(image_bytes)
    original_mode = Image.open(io.BytesIO(image_bytes)).mode
    arr = np.asarray(pil)
    h, w = arr.shape[:2]
    has_alpha = _has_real_alpha(pil, original_mode)

    out["output_width_px"] = int(w)
    out["output_height_px"] = int(h)
    out["output_aspect_ratio"] = _round(w / h, 4) if h else None
    out["output_orientation"] = _orientation(w, h)
    out["output_has_alpha"] = bool(has_alpha)

    tw_in = before.get("target_print_width_in")
    th_in = before.get("target_print_height_in")
    if tw_in:
        out["effective_dpi_x"] = _round(w / tw_in, 1)
    if th_in:
        out["effective_dpi_y"] = _round(h / th_in, 1)

    mask = _content_mask(arr, has_alpha)
    bbox = _bbox_from_mask(mask) if mask is not None else None
    if bbox:
        bx, by, bw, bh = bbox
        out["output_bbox_x"] = bx
        out["output_bbox_y"] = by
        out["output_bbox_width"] = bw
        out["output_bbox_height"] = bh
        out["padding_left"] = bx
        out["padding_right"] = int(w - (bx + bw))
        out["padding_top"] = by
        out["padding_bottom"] = int(h - (by + bh))
        dpi = before.get("target_dpi") or DEFAULT_TARGET_DPI
        out["padding_left_in"] = _round(bx / dpi, 3)
        out["padding_right_in"] = _round((w - (bx + bw)) / dpi, 3)
        out["padding_top_in"] = _round(by / dpi, 3)
        out["padding_bottom_in"] = _round((h - (by + bh)) / dpi, 3)
        out["output_center_x_pct"] = _round(100.0 * (bx + bw / 2.0) / w, 2) if w else None
        out["output_center_y_pct"] = _round(100.0 * (by + bh / 2.0) / h, 2) if h else None
        out["distance_left_edge_px"] = bx
        out["distance_right_edge_px"] = int(w - (bx + bw))
        out["distance_top_edge_px"] = by
        out["distance_bottom_edge_px"] = int(h - (by + bh))
        out["clipping_detected"] = bool(min(bx, by, w - (bx + bw), h - (by + bh)) <= CLIP_EDGE_TOUCH_PX)

    # --- Alpha histogram + stray/semi counts (only if a real alpha channel) ---
    if has_alpha:
        a = arr[:, :, 3]
        out["alpha_zero_count"] = int((a == 0).sum())
        out["alpha_1_10_count"] = int(((a >= 1) & (a <= 10)).sum())
        out["alpha_11_50_count"] = int(((a >= 11) & (a <= 50)).sum())
        out["alpha_51_200_count"] = int(((a >= 51) & (a <= 200)).sum())
        out["alpha_201_254_count"] = int(((a >= 201) & (a <= 254)).sum())
        out["alpha_255_count"] = int((a == 255).sum())
        intent = _intent_mask(a)
        low_alpha = (a >= 1) & (a <= LOW_ALPHA_STRAY_MAX)
        out["low_alpha_outside_intent_count"] = int((low_alpha & ~intent).sum())
        semi = (a >= 1) & (a <= 254)
        out["semi_transparent_inside_effects_count"] = int((semi & intent).sum())
        # Edge band + halo (transparency only — same rule as Print Ready halo).
        edge_counts = _edge_halo(arr, a)
        out.update(edge_counts)
    # else: all alpha + halo fields stay None -> N/A

    # --- Connected components ---
    if mask is not None:
        comps = _components(mask)
        out["connected_component_count"] = int(len(comps))
        out["isolated_component_count"] = int(sum(1 for c in comps if c <= ISOLATED_COMPONENT_MAX_AREA_PX))
        out["micro_component_count"] = int(sum(1 for c in comps if c < MICRO_COMPONENT_MAX_AREA_PX))
        out["smallest_component_area_px"] = int(min(comps)) if comps else None
        # --- Fine detail ---
        mlw = _min_stroke_width_px(mask)
        out["minimum_line_width_px"] = _round(mlw, 2) if mlw is not None else None
        dpi = before.get("target_dpi") or DEFAULT_TARGET_DPI
        out["minimum_line_width_in"] = _round(mlw / dpi, 4) if mlw is not None else None
        out["minimum_hole_size_px"] = _hole_sizes(mask)
    out["minimum_text_stroke_px"] = None   # N/A in V1 (needs text detection)

    # --- Comparison to the source (never to intent) ---
    _compare_to_source(out, before, arr, mask, pil, has_alpha)

    # --- V1 N/A comparison fields ---
    out["text_geometry_similarity_pct"] = None
    out["text_ocr_match_pct"] = None
    out["missing_objects_count"] = None
    out["extra_objects_count"] = None

    # --- QA statuses from thresholds ---
    _qa_statuses(out, before, has_alpha)

    return out


def _intent_mask(alpha: np.ndarray) -> np.ndarray:
    """Dilated solid-content mask — where real artwork is, so strays can be told
    from intended soft edges/effects."""
    import cv2
    solid = (alpha >= SOLID_CONTENT_ALPHA_MIN).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INTENT_DILATE_PX + 1,) * 2)
    return cv2.dilate(solid, k).astype(bool)


def _edge_halo(arr: np.ndarray, alpha: np.ndarray) -> dict:
    """Edge band pixel counts + light/dark halo + fringe, for a transparent image."""
    import cv2
    solid = (alpha >= SOLID_CONTENT_ALPHA_MIN).astype(np.uint8)
    if not solid.any():
        return {"edge_pixel_count": 0, "edge_contamination_count": 0,
                "edge_contamination_pct": 0.0, "halo_white_px": 0,
                "halo_dark_px": 0, "fringe_pixel_count": 0}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_BAND_PX + 1,) * 2)
    dil = cv2.dilate(solid, k)
    ero = cv2.erode(solid, k)
    band = (dil.astype(bool) & ~ero.astype(bool))
    edge_px = int(band.sum())
    rgb = arr[:, :, :3]
    band_rgb = rgb[band]
    band_a = alpha[band]
    # Contamination: semi-transparent pixels sitting in the edge band.
    contam = int(((band_a >= 1) & (band_a <= 254)).sum())
    light = int((band_a > 0)[(band_rgb >= HALO_LIGHT_MIN).all(axis=1)].sum()) if band_rgb.size else 0
    dark = int((band_a > 0)[(band_rgb <= HALO_DARK_MAX).all(axis=1)].sum()) if band_rgb.size else 0
    return {
        "edge_pixel_count": edge_px,
        "edge_contamination_count": contam,
        "edge_contamination_pct": _round(100.0 * contam / edge_px, 2) if edge_px else 0.0,
        "halo_white_px": light,
        "halo_dark_px": dark,
        "fringe_pixel_count": contam,
    }


def _compare_to_source(out: dict, before: dict, out_arr: np.ndarray,
                       out_mask, out_pil, out_has_alpha) -> None:
    """Fill the before-vs-after similarity fields. Each is None (N/A) when its
    inputs are missing, so nothing is ever guessed."""
    s_ratio = before.get("source_aspect_ratio")
    o_ratio = out.get("output_aspect_ratio")
    if s_ratio and o_ratio:
        diff = abs(o_ratio - s_ratio) / s_ratio * 100.0
        out["source_ratio_difference_pct"] = _round(diff, 2)

    # bbox IoU on normalised boxes.
    sb = _source_bbox_norm(before)
    ob = None
    if all(out.get(k) is not None for k in ("output_bbox_x", "output_bbox_y",
                                            "output_bbox_width", "output_bbox_height")):
        ob = _norm_bbox((out["output_bbox_x"], out["output_bbox_y"],
                         out["output_bbox_width"], out["output_bbox_height"]),
                        out["output_width_px"], out["output_height_px"])
    if sb and ob:
        out["bbox_similarity_pct"] = _round(100.0 * _iou(sb, ob), 2)

    # Position + scale similarity from the resized content masks.
    s_mask = before.get("_content_mask_norm")   # not stored; recomputed below if possible
    # position similarity: 1 - normalised centroid distance; scale: area ratio.
    if sb and ob:
        s_cx = (sb[0] + sb[2]) / 2.0; s_cy = (sb[1] + sb[3]) / 2.0
        o_cx = (ob[0] + ob[2]) / 2.0; o_cy = (ob[1] + ob[3]) / 2.0
        dist = math.hypot(o_cx - s_cx, o_cy - s_cy) / math.sqrt(2)   # 0..1
        out["object_position_similarity_pct"] = _round(100.0 * (1.0 - dist), 2)
        s_area = (sb[2] - sb[0]) * (sb[3] - sb[1])
        o_area = (ob[2] - ob[0]) * (ob[3] - ob[1])
        if s_area > 0 and o_area > 0:
            ratio = min(s_area, o_area) / max(s_area, o_area)
            out["object_scale_similarity_pct"] = _round(100.0 * ratio, 2)

    # semantic similarity: perceptual-hash similarity, clearly labelled "pHash".
    src_bytes = before.get("_source_image_bytes")
    if src_bytes:
        try:
            sp = _phash(_load_rgba(src_bytes))
            op = _phash(out_pil)
            out["semantic_similarity_pct"] = _round(_phash_similarity(sp, op), 2)
        except Exception:
            out["semantic_similarity_pct"] = None

    # overall geometry similarity: weighted mean of the available components.
    out["overall_geometry_similarity_pct"] = _weighted_geo_similarity(out)


def _source_bbox_norm(before: dict):
    if all(before.get(k) is not None for k in ("source_bbox_x", "source_bbox_y",
                                               "source_bbox_width", "source_bbox_height",
                                               "source_width_px", "source_height_px")):
        return _norm_bbox((before["source_bbox_x"], before["source_bbox_y"],
                           before["source_bbox_width"], before["source_bbox_height"]),
                          before["source_width_px"], before["source_height_px"])
    return None


def _weighted_geo_similarity(out: dict):
    parts = {
        "bbox_similarity_pct": out.get("bbox_similarity_pct"),
        "object_position_similarity_pct": out.get("object_position_similarity_pct"),
        "object_scale_similarity_pct": out.get("object_scale_similarity_pct"),
        "source_ratio_match_pct": (100.0 - out["source_ratio_difference_pct"])
        if out.get("source_ratio_difference_pct") is not None else None,
    }
    num = 0.0
    den = 0.0
    for key, weight in GEOMETRY_SIM_WEIGHTS.items():
        v = parts.get(key)
        if v is None:
            continue
        num += weight * max(0.0, min(100.0, v))
        den += weight
    return _round(num / den, 2) if den > 0 else None


def _qa_statuses(out: dict, before: dict, has_alpha: bool) -> None:
    # background transparency vs requirement.
    req = (before.get("background_requirement") or "").strip().lower()
    if req == "transparent":
        out["background_transparency_status"] = PASS if has_alpha else FAIL
    elif req:
        out["background_transparency_status"] = PASS  # a non-transparent requirement: measured bg is informational
    else:
        out["background_transparency_status"] = NA

    # resolution: effective dpi vs target dpi.
    target_dpi = before.get("target_dpi")
    eff = out.get("effective_dpi_x")
    if target_dpi and eff is not None:
        frac = eff / target_dpi
        if frac < DPI_FAIL_FRAC:
            out["resolution_status"] = FAIL
        elif frac < DPI_WARN_FRAC:
            out["resolution_status"] = WARN
        else:
            out["resolution_status"] = PASS
    else:
        out["resolution_status"] = NA

    out["geometry_status"] = _status_from_pct(out.get("overall_geometry_similarity_pct"),
                                              GEOMETRY_SIM_WARN_PCT, GEOMETRY_SIM_FAIL_PCT)

    # alpha status: strays outside intent -> WARN/FAIL.
    stray = out.get("low_alpha_outside_intent_count")
    if not has_alpha:
        out["alpha_status"] = NA
        out["stray_pixel_status"] = NA
    else:
        out["alpha_status"] = PASS if (stray is not None and stray == 0) else (
            WARN if (stray is not None and stray < 500) else FAIL)
        out["stray_pixel_status"] = PASS if (stray == 0) else (WARN if (stray or 0) < 500 else FAIL)

    # halo status (transparency only).
    if not has_alpha:
        out["halo_status"] = NA
    else:
        contam_pct = out.get("edge_contamination_pct") or 0.0
        if contam_pct >= EDGE_CONTAM_FAIL_PCT:
            out["halo_status"] = FAIL
        elif contam_pct >= EDGE_CONTAM_WARN_PCT:
            out["halo_status"] = WARN
        else:
            out["halo_status"] = PASS

    # fine detail from minimum line width (at target res).
    mlw = out.get("minimum_line_width_px")
    if mlw is None:
        out["fine_detail_status"] = NA
    elif mlw < MIN_LINE_WIDTH_FAIL_PX:
        out["fine_detail_status"] = FAIL
    elif mlw < MIN_LINE_WIDTH_WARN_PX:
        out["fine_detail_status"] = WARN
    else:
        out["fine_detail_status"] = PASS

    # clipping.
    clip = out.get("clipping_detected")
    out["clipping_status"] = NA if clip is None else (FAIL if clip else PASS)

    # text + semantic: N/A in V1 (semantic is pHash-only, informational).
    out["text_status"] = NA
    out["semantic_status"] = NA

    # overall = worst of the real statuses.
    out["overall_qa_status"] = _worst([
        out["background_transparency_status"], out["resolution_status"], out["geometry_status"],
        out["alpha_status"], out["halo_status"], out["stray_pixel_status"],
        out["fine_detail_status"], out["clipping_status"],
    ])


# ===========================================================================
# Public API
# ===========================================================================

def measure_geometry(image_bytes: bytes, *, inputs: dict | None = None,
                     role: str = "before", artwork_id: str = "",
                     source_filename: str = "", before: dict | None = None) -> dict:
    """Measure one image.

    role="before": returns the BEFORE block for the source (needs artwork_id +
    source_filename + the operator inputs).
    role="after":  returns the AFTER block for the output (needs the already
    computed `before` block to compare against)."""
    if role == "before":
        blk = measure_source(image_bytes, artwork_id=artwork_id,
                             source_filename=source_filename, inputs=inputs)
        return blk
    if role == "after":
        if not before:
            raise ValueError("measure_geometry(role='after') needs the BEFORE block to compare against.")
        return measure_output(image_bytes, before=before)
    raise ValueError(f"Unknown role {role!r} (expected 'before' or 'after').")


def blank_block(fields: list[str]) -> dict:
    """An all-None block of the given schema — used when a stage has no image yet
    so the UI still renders every row as N/A."""
    return {k: None for k in fields}


def autofix_block() -> dict:
    """V1: every auto-fix field present, all None (UI shows 'N/A — not implemented yet')."""
    return {k: None for k in AUTOFIX_FIELDS}


def approval_block(output_bytes: "bytes | None" = None,
                   before: dict | None = None) -> dict:
    """V1 approval block — all None except approved_file_hash, which we CAN fill
    (SHA256 of the output) when the output exists."""
    blk = {k: None for k in APPROVAL_FIELDS}
    if output_bytes:
        blk["approved_file_hash"] = hashlib.sha256(output_bytes).hexdigest()
    if before:
        blk["approved_print_width_in"] = before.get("target_print_width_in")
        blk["approved_print_height_in"] = before.get("target_print_height_in")
    return blk


def _json_safe_block(block: dict | None, fields: list[str]) -> dict:
    """A block limited to its declared fields, so no transient/underscore helper
    key (e.g. a cached image-bytes value a caller stashed on the dict) can leak
    into the Download JSON and make it non-serialisable. Missing fields -> None."""
    if not block:
        return blank_block(fields)
    return {k: block.get(k) for k in fields}


def build_master(artwork_id: str, before: dict | None, after: dict | None,
                 autofix: dict | None = None, approval: dict | None = None) -> dict:
    """The single master object offered as Download JSON. Nulls for every N/A.
    Each block is projected onto its declared schema so only known, JSON-safe
    fields are included (defends the download against a stray non-serialisable
    key stashed on a block)."""
    return {
        "artwork_id": artwork_id,
        "before": _json_safe_block(before, BEFORE_FIELDS),
        "after": _json_safe_block(after, AFTER_FIELDS),
        "autofix": _json_safe_block(autofix, AUTOFIX_FIELDS) if autofix is not None else autofix_block(),
        "approval": _json_safe_block(approval, APPROVAL_FIELDS) if approval is not None else approval_block(),
    }

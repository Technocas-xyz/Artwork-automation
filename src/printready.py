"""UC-8 Print Ready QA — LOCAL measurement half (no ChatGPT).

Everything numeric about an artwork's print-readiness for DTF garment printing is
measured HERE, exactly and repeatably, with Pillow/numpy and (where it helps)
OpenCV. ChatGPT is given these numbers plus the artwork and asked only for the
JUDGEMENT a printer would make — never to re-measure.

Public entry point:
    measure_print_ready(image_bytes, dpi_override=None, file_size_bytes=None)
        -> {
             "checks":  [ {id, label, value, value_str, unit, threshold_str,
                           status ('pass'|'warn'|'fail'), weight, detail}, ... ],
             "score":   int 0..100  (weighted by how badly each issue hurts a print),
             "band":    'ready' | 'fixable' | 'not_ready',
             "summary": str  (a plain-text report the operator can send),
             "measurements": {machine-readable raw numbers for the prompt},
           }

cv2 is imported LAZILY inside the one function that needs it, mirroring
src/annotate.py, so importing this module never requires OpenCV (the packaged
agent has none). Every check is defensive: a single measurement that cannot be
taken is reported as a 'warn' with a note, never an exception into the caller.
"""

from __future__ import annotations

import io
import logging
import math
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ===========================================================================
# THRESHOLDS — the print team tunes everything here, in one place.
#
# Each entry drives one check's pass/warn/fail bands and its WEIGHT (how much a
# failure pulls the overall score down). Weights are relative; the score is a
# weighted average of per-check credit (pass=1.0, warn=0.5, fail=0.0), so raising
# a weight makes that issue matter more. Comments explain WHY each number.
# ===========================================================================
THRESHOLDS: dict[str, dict[str, Any]] = {
    # 1. Print DPI. Below 150 DPI a raster transfer looks soft; 300 is ideal.
    "dpi":            {"warn": 300, "fail": 150, "weight": 8},
    # 3. Semi-transparent pixels print as grey "dirt" on a DTF film. A little is
    #    unavoidable at edges; a lot means a soft/feathered cut that will look dirty.
    "semi_alpha_pct": {"warn": 4.0, "fail": 12.0, "weight": 10},
    # 4. Halo / fringe: pale pixels just outside the content edge (leftover
    #    background). Visible as a light outline on a coloured garment.
    "halo_pct":       {"warn": 1.5, "fail": 5.0, "weight": 9},
    # 5. Soft edges: fraction of edge pixels that are gradual rather than crisp.
    #    Soft edges feather and lose detail through the transfer.
    "soft_edge_pct":  {"warn": 35.0, "fail": 60.0, "weight": 6},
    # 6. Distinct colours after quantisation. Not a hard limit for DTF (full
    #    colour), but a very high count can indicate noise/gradients to review.
    "colour_count":   {"warn": 64, "fail": 200, "weight": 3},
    # 7. Pure black: the darkest sizeable area should be near #000 for a solid
    #    black print. If the darkest area is washed out, blacks look grey.
    "black_level":    {"warn": 40, "fail": 70, "weight": 5},  # 0=perfect black; value = darkest-area luminance
    # 8. Minimum stroke width in MM at the print DPI. Below ~0.8 mm a stroke can
    #    break up on a garment; below ~0.4 mm it may not transfer at all.
    "min_stroke_mm":  {"warn": 0.8, "fail": 0.4, "weight": 10},
    # 9. Smallest text-like feature HEIGHT in MM — a legibility proxy. Below
    #    ~3 mm small text closes up; below ~2 mm it is usually illegible.
    "min_feature_mm": {"warn": 3.0, "fail": 2.0, "weight": 7},
    # 10. Gradient coverage: fraction of the image that is smooth tonal
    #     transition. Large smooth gradients can band on a DTF transfer.
    "gradient_pct":   {"warn": 25.0, "fail": 50.0, "weight": 4},
    # 11. Margin to canvas edge (% of the shorter side). Content touching the
    #     edge risks being clipped when the film is cut/placed.
    "margin_pct":     {"warn": 2.0, "fail": 0.5, "weight": 4},  # SMALLER is worse
    # 13. DTF transfer physical size limit (inches) — the widest film this shop
    #     runs. Larger than this cannot be printed in one piece.
    "max_print_in":   {"warn": 22.0, "fail": 24.0, "weight": 6},  # LARGER is worse
    # Supporting constants (not their own weighted checks, used by the ones above):
    "quantise_colours": 32,      # k for the colour-count quantisation
    "semi_alpha_low": 10,        # alpha in [low, high] counts as semi-transparent
    "semi_alpha_high": 245,
    "edge_soft_band": (30, 120), # gradient-magnitude band counted as a "soft" edge
    "default_dpi": 300,          # assumed when the file carries no DPI metadata
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _status_high_bad(value: float, warn: float, fail: float) -> str:
    """pass/warn/fail where a HIGHER value is worse (e.g. semi_alpha_pct)."""
    if value >= fail:
        return "fail"
    if value >= warn:
        return "warn"
    return "pass"


def _status_low_bad(value: float, warn: float, fail: float) -> str:
    """pass/warn/fail where a LOWER value is worse (e.g. dpi, margin)."""
    if value <= fail:
        return "fail"
    if value <= warn:
        return "warn"
    return "pass"


def _px_to_mm(px: float, dpi: float) -> float:
    return (px / dpi) * 25.4 if dpi > 0 else 0.0


def _content_mask(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Boolean mask of 'content' pixels and whether the image had an alpha
    channel. arr is HxWx4 uint8 (RGBA). Content = opaque-enough pixels; if there
    is no meaningful transparency, content is 'not (near-background)'."""
    alpha = arr[:, :, 3]
    has_alpha = bool((alpha < 250).any())
    if has_alpha:
        return alpha > 20, True
    # Opaque image: treat the dominant border colour as background.
    rgb = arr[:, :, :3]
    h, w = rgb.shape[:2]
    border = np.concatenate([
        rgb[0, :, :].reshape(-1, 3), rgb[-1, :, :].reshape(-1, 3),
        rgb[:, 0, :].reshape(-1, 3), rgb[:, -1, :].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    return dist > 32, False


# ---------------------------------------------------------------------------
# The 13 measurements. Each appends a check dict to `checks` and stashes raw
# numbers in `raw` (for the prompt). Kept small and independent so one failing
# measurement never blocks the rest.
# ---------------------------------------------------------------------------

def measure_print_ready(image_bytes: bytes, dpi_override: float | None = None,
                        file_size_bytes: int | None = None) -> dict:
    """Measure all 13 print-readiness checks. Returns the report dict described
    in the module docstring. Never raises — a broken sub-measurement becomes a
    'warn' check with a note."""
    T = THRESHOLDS
    checks: list[dict] = []
    raw: dict[str, Any] = {}

    im = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = im.size
    arr = np.asarray(im)
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    content, has_alpha = _content_mask(arr)
    n_content = int(content.sum())

    def add(cid, label, value, value_str, unit, threshold_str, status, weight, detail=""):
        checks.append({"id": cid, "label": label, "value": value, "value_str": value_str,
                       "unit": unit, "threshold_str": threshold_str, "status": status,
                       "weight": weight, "detail": detail})

    # --- DPI resolution (needed by several checks) ---
    dpi = float(dpi_override) if dpi_override else 0.0
    if not dpi:
        try:
            xdpi = im.info.get("dpi", (0, 0))
            dpi = float(xdpi[0]) if isinstance(xdpi, (tuple, list)) else float(xdpi or 0)
        except Exception:
            dpi = 0.0
    dpi_known = dpi > 1
    if not dpi_known:
        dpi = float(T["default_dpi"])

    # 1. DIMENSIONS, DPI, PRINT SIZE ----------------------------------------
    print_w_in = w / dpi
    print_h_in = h / dpi
    raw.update(width_px=w, height_px=h, dpi=round(dpi, 1), dpi_known=dpi_known,
               print_w_in=round(print_w_in, 2), print_h_in=round(print_h_in, 2))
    st = _status_low_bad(dpi, T["dpi"]["warn"], T["dpi"]["fail"])
    # Be explicit about where the DPI came from: a file's metadata is often the
    # OS default (Windows writes 96) and may not reflect the intended print size.
    if dpi_override:
        dpi_src = "operator-set"
    elif dpi_known:
        dpi_src = "from file metadata, may not reflect intended print size"
    else:
        dpi_src = "assumed — no DPI in file"
    add("dpi", "Resolution (DPI)", round(dpi, 1),
        f"{dpi:.0f} DPI ({dpi_src})", "DPI",
        f"warn <{T['dpi']['warn']}, fail <{T['dpi']['fail']}", st, T["dpi"]["weight"],
        f"{w}x{h}px = {print_w_in:.2f}x{print_h_in:.2f} in")

    # 2. TRANSPARENCY present + % fully transparent -------------------------
    fully_transparent = float((alpha == 0).mean() * 100.0)
    raw.update(has_transparency=has_alpha, fully_transparent_pct=round(fully_transparent, 1))
    add("transparency", "Transparency", round(fully_transparent, 1),
        f"{'present' if has_alpha else 'none'}, {fully_transparent:.1f}% fully clear", "%",
        "informational", "pass" if has_alpha else "warn", 2,
        "DTF wants a transparent background" if not has_alpha else "")

    # 3. SEMI-TRANSPARENT pixels (print as dirt) ----------------------------
    semi = (alpha >= T["semi_alpha_low"]) & (alpha <= T["semi_alpha_high"])
    semi_pct = float(semi.mean() * 100.0)
    raw["semi_transparent_pct"] = round(semi_pct, 2)
    st = _status_high_bad(semi_pct, T["semi_alpha_pct"]["warn"], T["semi_alpha_pct"]["fail"])
    add("semi_alpha", "Semi-transparent pixels", round(semi_pct, 2),
        f"{semi_pct:.2f}%", "%",
        f"warn >{T['semi_alpha_pct']['warn']}%, fail >{T['semi_alpha_pct']['fail']}%",
        st, T["semi_alpha_pct"]["weight"], "alpha 10-245 prints as grey dirt")

    # 4. HALO / FRINGE: pale pixels just outside the content edge -----------
    # Halo only means something when a background has been REMOVED (i.e. the
    # image has transparency). On a fully opaque image the "content edge" is
    # arbitrary and any measurement is meaningless, so mark it n/a (informational,
    # weight 0, NOT scored) instead of failing a clean opaque design.
    if not has_alpha:
        raw["halo_pct"] = None
        add("halo", "Halo / edge fringe", 0.0, "n/a", "%", "n/a", "na", 0,
            "no transparency - halo not measurable")
    else:
        halo_pct = -1.0
        try:
            halo_pct = _measure_halo(arr, content, rgb)
        except Exception as exc:
            logger.info("printready: halo measure failed (%s)", exc)
            halo_pct = -1.0
        if halo_pct < 0:
            add("halo", "Halo / edge fringe", 0.0, "could not measure", "%",
                f"warn >{T['halo_pct']['warn']}%, fail >{T['halo_pct']['fail']}%", "warn",
                T["halo_pct"]["weight"], "measurement unavailable")
        else:
            raw["halo_pct"] = round(halo_pct, 2)
            st = _status_high_bad(halo_pct, T["halo_pct"]["warn"], T["halo_pct"]["fail"])
            add("halo", "Halo / edge fringe", round(halo_pct, 2), f"{halo_pct:.2f}%", "%",
                f"warn >{T['halo_pct']['warn']}%, fail >{T['halo_pct']['fail']}%", st,
                T["halo_pct"]["weight"], "pale pixels ringing the content edge")

    # 5. EDGE SHARPNESS: fraction of soft edges -----------------------------
    try:
        soft_pct = _measure_soft_edges(lum, T["edge_soft_band"])
        raw["soft_edge_pct"] = round(soft_pct, 1)
        st = _status_high_bad(soft_pct, T["soft_edge_pct"]["warn"], T["soft_edge_pct"]["fail"])
        add("soft_edges", "Edge sharpness", round(soft_pct, 1), f"{soft_pct:.1f}% soft", "%",
            f"warn >{T['soft_edge_pct']['warn']}%, fail >{T['soft_edge_pct']['fail']}%", st,
            T["soft_edge_pct"]["weight"], "soft edges feather in the transfer")
    except Exception as exc:
        logger.info("printready: soft-edge measure failed (%s)", exc)
        add("soft_edges", "Edge sharpness", 0.0, "could not measure", "%", "-", "warn",
            T["soft_edge_pct"]["weight"], "measurement unavailable")

    # 6. DISTINCT COLOUR COUNT (after quantisation) + top colours -----------
    try:
        colour_count, top_colours = _measure_colours(im, content, T["quantise_colours"])
        raw.update(colour_count=colour_count, top_colours=top_colours)
        st = _status_high_bad(colour_count, T["colour_count"]["warn"], T["colour_count"]["fail"])
        top_str = ", ".join(f"{c['hex']} {c['pct']}%" for c in top_colours[:4])
        add("colours", "Distinct colours", colour_count, f"{colour_count} colours", "count",
            f"warn >{T['colour_count']['warn']}, fail >{T['colour_count']['fail']}", st,
            T["colour_count"]["weight"], f"top: {top_str}")
    except Exception as exc:
        logger.info("printready: colour measure failed (%s)", exc)
        add("colours", "Distinct colours", 0, "could not measure", "count", "-", "warn",
            T["colour_count"]["weight"], "measurement unavailable")

    # 7. PURE BLACK check ----------------------------------------------------
    try:
        # Luminance of the darkest sizeable region: the 1st percentile of
        # content luminance (ignores stray single dark pixels).
        cl = lum[content] if n_content else lum.reshape(-1)
        black_level = float(np.percentile(cl, 1)) if cl.size else 255.0
        raw["darkest_luminance"] = round(black_level, 1)
        # Higher darkest-area luminance = worse (washed-out blacks).
        st = _status_high_bad(black_level, T["black_level"]["warn"], T["black_level"]["fail"])
        add("black", "Pure black", round(black_level, 1),
            f"darkest ~{black_level:.0f}/255", "luminance",
            f"warn >{T['black_level']['warn']}, fail >{T['black_level']['fail']}", st,
            T["black_level"]["weight"],
            "near 0 = solid black; high = washed-out blacks")
    except Exception as exc:
        logger.info("printready: black measure failed (%s)", exc)
        add("black", "Pure black", 0.0, "could not measure", "luminance", "-", "warn",
            T["black_level"]["weight"], "measurement unavailable")

    # 8. MINIMUM STROKE WIDTH (px + mm) -------------------------------------
    try:
        stroke_px = _measure_min_stroke(content)
        stroke_mm = _px_to_mm(stroke_px, dpi)
        raw.update(min_stroke_px=round(stroke_px, 1), min_stroke_mm=round(stroke_mm, 2))
        st = _status_low_bad(stroke_mm, T["min_stroke_mm"]["warn"], T["min_stroke_mm"]["fail"])
        add("min_stroke", "Minimum stroke width", round(stroke_mm, 2),
            f"{stroke_mm:.2f} mm ({stroke_px:.0f}px)", "mm",
            f"warn <{T['min_stroke_mm']['warn']}mm, fail <{T['min_stroke_mm']['fail']}mm", st,
            T["min_stroke_mm"]["weight"], "thinnest continuous content run")
    except Exception as exc:
        logger.info("printready: stroke measure failed (%s)", exc)
        add("min_stroke", "Minimum stroke width", 0.0, "could not measure", "mm", "-", "warn",
            T["min_stroke_mm"]["weight"], "measurement unavailable")

    # 9. SMALLEST TEXT-LIKE FEATURE HEIGHT (mm) -----------------------------
    try:
        feat_px = _measure_min_feature(content)
        feat_mm = _px_to_mm(feat_px, dpi)
        raw.update(min_feature_px=round(feat_px, 1), min_feature_mm=round(feat_mm, 2))
        st = _status_low_bad(feat_mm, T["min_feature_mm"]["warn"], T["min_feature_mm"]["fail"])
        add("min_feature", "Smallest feature height", round(feat_mm, 2),
            f"{feat_mm:.2f} mm ({feat_px:.0f}px)", "mm",
            f"warn <{T['min_feature_mm']['warn']}mm, fail <{T['min_feature_mm']['fail']}mm", st,
            T["min_feature_mm"]["weight"], "legibility proxy for small text/detail")
    except Exception as exc:
        logger.info("printready: feature measure failed (%s)", exc)
        add("min_feature", "Smallest feature height", 0.0, "could not measure", "mm", "-", "warn",
            T["min_feature_mm"]["weight"], "measurement unavailable")

    # 10. GRADIENT PRESENCE --------------------------------------------------
    try:
        grad_pct = _measure_gradients(lum, content)
        raw["gradient_pct"] = round(grad_pct, 1)
        st = _status_high_bad(grad_pct, T["gradient_pct"]["warn"], T["gradient_pct"]["fail"])
        add("gradients", "Gradient coverage", round(grad_pct, 1), f"{grad_pct:.1f}%", "%",
            f"warn >{T['gradient_pct']['warn']}%, fail >{T['gradient_pct']['fail']}%", st,
            T["gradient_pct"]["weight"], "large smooth gradients can band")
    except Exception as exc:
        logger.info("printready: gradient measure failed (%s)", exc)
        add("gradients", "Gradient coverage", 0.0, "could not measure", "%", "-", "warn",
            T["gradient_pct"]["weight"], "measurement unavailable")

    # 11. CONTENT BOUNDING BOX + MARGINS ------------------------------------
    try:
        margin_pct, bbox = _measure_margins(content, w, h)
        raw.update(content_bbox=bbox, margin_pct=round(margin_pct, 2))
        st = _status_low_bad(margin_pct, T["margin_pct"]["warn"], T["margin_pct"]["fail"])
        add("margins", "Edge margin", round(margin_pct, 2), f"{margin_pct:.2f}% of short side", "%",
            f"warn <{T['margin_pct']['warn']}%, fail <{T['margin_pct']['fail']}%", st,
            T["margin_pct"]["weight"], f"content bbox {bbox}")
    except Exception as exc:
        logger.info("printready: margin measure failed (%s)", exc)
        add("margins", "Edge margin", 0.0, "could not measure", "%", "-", "warn",
            T["margin_pct"]["weight"], "measurement unavailable")

    # 12. ASPECT RATIO (normalised) + nearest standard placement ------------
    try:
        ratio_str, nearest = _measure_aspect(w, h)
        raw.update(aspect=ratio_str, nearest_placement=nearest)
        add("aspect", "Aspect ratio", 0.0, ratio_str, "ratio", "informational", "pass", 1,
            f"nearest standard: {nearest}")
    except Exception as exc:
        logger.info("printready: aspect measure failed (%s)", exc)
        add("aspect", "Aspect ratio", 0.0, "could not measure", "ratio", "-", "warn", 1, "")

    # 13. FILE SIZE + DTF transfer size limit -------------------------------
    max_dim_in = max(print_w_in, print_h_in)
    raw["max_print_in"] = round(max_dim_in, 2)
    if file_size_bytes:
        raw["file_size_kb"] = round(file_size_bytes / 1024.0, 1)
    st = _status_high_bad(max_dim_in, T["max_print_in"]["warn"], T["max_print_in"]["fail"])
    size_note = f", file {raw.get('file_size_kb', '?')} KB" if file_size_bytes else ""
    add("print_size", "DTF transfer size", round(max_dim_in, 2),
        f"{max_dim_in:.2f} in widest{size_note}", "in",
        f"warn >{T['max_print_in']['warn']}in, fail >{T['max_print_in']['fail']}in", st,
        T["max_print_in"]["weight"], "must fit the transfer film width")

    # --- WEIGHTED SCORE -----------------------------------------------------
    # 'na' checks (informational / not applicable, e.g. halo on an opaque image)
    # do NOT count: they are excluded from both the numerator and the weight
    # total, so a clean opaque design is not penalised for a check that cannot
    # apply to it.
    credit = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    scored = [c for c in checks if c["status"] in credit and c["weight"] > 0]
    tot_w = sum(c["weight"] for c in scored) or 1
    score = round(100.0 * sum(credit[c["status"]] * c["weight"] for c in scored) / tot_w)
    band = "ready" if score >= 90 else ("fixable" if score >= 70 else "not_ready")

    summary = _build_summary(checks, score, band, raw)

    return {"checks": checks, "score": int(score), "band": band,
            "summary": summary, "measurements": raw}


# ---------------------------------------------------------------------------
# Individual measurement kernels
# ---------------------------------------------------------------------------

def _measure_halo(arr: np.ndarray, content: np.ndarray, rgb: np.ndarray) -> float:
    """% of pixels in a thin ring just OUTSIDE the content that are pale (i.e.
    leftover near-background fringe). Uses cv2 dilation for the ring; the halo
    is pale non-content pixels adjacent to content."""
    import cv2  # server-only
    cm = content.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(cm, kernel, iterations=2)
    ring = (dilated > 0) & (~content)
    if ring.sum() == 0:
        return 0.0
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    # "pale" = light-ish but not pure background; leftover feathering.
    pale = ring & (lum > 140) & (lum < 250)
    return float(pale.sum()) / float(ring.sum()) * 100.0


def _measure_soft_edges(lum: np.ndarray, band: tuple) -> float:
    """% of EDGE pixels whose gradient magnitude falls in the 'soft' band —
    gradual transitions rather than hard cuts. Uses cv2 Sobel."""
    import cv2
    l8 = lum.astype(np.uint8)
    gx = cv2.Sobel(l8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    lo, hi = band
    edges = mag > lo               # any real edge
    if edges.sum() == 0:
        return 0.0
    soft = (mag > lo) & (mag < hi)  # present but not crisp
    return float(soft.sum()) / float(edges.sum()) * 100.0


def _measure_colours(im: Image.Image, content: np.ndarray, k: int):
    """Distinct colour count after quantisation, plus the top colours by area.
    Quantises to at most `k` colours (Pillow adaptive palette) and counts those
    that actually appear over the content region."""
    q = im.convert("RGB").quantize(colors=k, method=Image.MEDIANCUT)
    pal = q.getpalette() or []
    idx = np.asarray(q)
    mask = content if content.shape == idx.shape else np.ones_like(idx, dtype=bool)
    vals, counts = np.unique(idx[mask], return_counts=True)
    total = counts.sum() or 1
    order = np.argsort(counts)[::-1]
    top = []
    for v in order[:8]:
        pi = int(vals[v]) * 3
        r, g, b = pal[pi:pi + 3] if pi + 3 <= len(pal) else (0, 0, 0)
        top.append({"hex": f"#{r:02X}{g:02X}{b:02X}", "rgb": [r, g, b],
                    "pct": round(100.0 * counts[v] / total, 1)})
    return int(len(vals)), top


def _measure_min_stroke(content: np.ndarray) -> float:
    """Thinnest continuous run of content, in pixels. Approximated by the
    distance transform: 2x the maximum inscribed radius of the THINNEST limb is
    hard, so instead we take the median of local run-lengths across rows and
    columns of the content and report the small percentile (a robust 'thin')."""
    import cv2
    cm = content.astype(np.uint8)
    if cm.sum() == 0:
        return 0.0
    # Distance transform gives, per content pixel, distance to nearest edge.
    # Stroke width ~= 2 * distance along the skeleton. Use the 20th percentile
    # of (2*dist) over content as a robust "thin stroke" figure.
    dist = cv2.distanceTransform(cm, cv2.DIST_L2, 3)
    d = dist[content]
    if d.size == 0:
        return 0.0
    thin = float(np.percentile(2.0 * d, 5))
    return max(thin, 1.0)


def _measure_min_feature(content: np.ndarray) -> float:
    """Height (px) of the smallest text-like / detail feature — a legibility
    proxy. We take the connected components of content, DROP sub-few-pixel noise
    (a component must be at least 3px in BOTH dimensions and a handful of pixels
    in area), then return a LOW PERCENTILE (10th) of the component heights. The
    percentile — not the raw minimum — is robust to a stray speck, and, when a
    large shape sits alongside small letters, it reflects the small letters
    rather than the whole shape. When there is only one real component (e.g. a
    single logo blob) it returns that component's height."""
    import cv2
    cm = content.astype(np.uint8)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
    if n <= 1:
        return 0.0
    MIN_DIM = 3     # a real feature is at least a few px in each direction
    MIN_AREA = 12   # and more than a handful of pixels (drops antialiasing specks)
    heights = []
    for i in range(1, n):
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if h < MIN_DIM or w < MIN_DIM or area < MIN_AREA:
            continue   # noise / antialiasing speck
        heights.append(h)
    if not heights:
        return 0.0
    # 10th percentile of the surviving component heights = "one of the smallest
    # real features", without letting a single 1px artefact dominate.
    return float(np.percentile(heights, 10))


def _measure_gradients(lum: np.ndarray, content: np.ndarray) -> float:
    """% of content that is smooth tonal transition: low-but-nonzero local
    gradient magnitude (not flat, not a hard edge)."""
    import cv2
    l8 = lum.astype(np.uint8)
    gx = cv2.Sobel(l8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if content.sum() == 0:
        return 0.0
    smooth = content & (mag > 3) & (mag < 25)   # gentle tonal change
    return float(smooth.sum()) / float(content.sum()) * 100.0


def _measure_margins(content: np.ndarray, w: int, h: int):
    """Smallest margin from the content bounding box to the canvas edge, as a %
    of the shorter side, plus the bbox (x, y, w, h)."""
    ys, xs = np.where(content)
    if xs.size == 0:
        return 100.0, [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    left, right, top, bottom = x0, w - 1 - x1, y0, h - 1 - y1
    short = min(w, h) or 1
    margin_pct = 100.0 * min(left, right, top, bottom) / short
    return margin_pct, [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _measure_aspect(w: int, h: int):
    """Normalised ratio string + nearest standard placement label."""
    g = math.gcd(w, h) or 1
    rw, rh = w // g, h // g
    ratio = w / h if h else 1.0
    STANDARD = [
        (1.0, "square (pocket / centre)"),
        (0.77, "A4 portrait (full front)"),
        (1.29, "A4 landscape"),
        (0.5, "tall left-chest / sleeve"),
        (2.0, "wide banner / back strip"),
    ]
    nearest = min(STANDARD, key=lambda s: abs(math.log(ratio / s[0])) if s[0] else 9)
    norm = f"{rw}:{rh}" if max(rw, rh) <= 50 else f"{ratio:.2f}:1"
    return norm, nearest[1]


def _build_summary(checks: list[dict], score: int, band: str, raw: dict) -> str:
    """A plain-text report the operator can copy/send. Failures first, then
    warnings, then the passing checks, then the key numbers."""
    band_label = {"ready": "PRINT READY", "fixable": "FIXABLE",
                  "not_ready": "NOT READY"}[band]
    lines = [f"PRINT READY QA — score {score}/100 ({band_label})", ""]
    order = {"fail": 0, "warn": 1, "pass": 2, "na": 3}
    for c in sorted(checks, key=lambda c: (order.get(c["status"], 4), -c["weight"])):
        tag = c["status"].upper().ljust(4)
        line = f"[{tag}] {c['label']}: {c['value_str']}"
        if c["threshold_str"] and c["threshold_str"] not in ("informational", "-", "n/a"):
            line += f"  (threshold: {c['threshold_str']})"
        if c["detail"]:
            line += f"  — {c['detail']}"
        lines.append(line)
    return "\n".join(lines)


# For the ChatGPT prompt: a compact measurements block (no pass/fail verbiage —
# the printer judges those; we just give the numbers).
def measurements_for_prompt(report: dict) -> str:
    """Format the report's raw measurements as a readable block for
    PRINTREADY_REVIEW's {measurements} placeholder."""
    m = report.get("measurements", {})
    checks = report.get("checks", [])
    lines = [f"Overall local score: {report.get('score')}/100 ({report.get('band')})", ""]
    for c in checks:
        lines.append(f"- {c['label']}: {c['value_str']}  [{c['status']}]")
    return "\n".join(lines)

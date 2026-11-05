"""UC-8 Print Ready QA â€” LOCAL measurement half (no ChatGPT).

Everything numeric about an artwork's print-readiness for DTF garment printing is
measured HERE, exactly and repeatably, with Pillow/numpy and (where it helps)
OpenCV. ChatGPT is given these numbers plus the artwork and asked only for the
JUDGEMENT a printer would make â€” never to re-measure.

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
# THRESHOLDS â€” the print team tunes everything here, in one place.
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
    # 9. Smallest text-like feature HEIGHT in MM â€” a legibility proxy. Below
    #    ~3 mm small text closes up; below ~2 mm it is usually illegible.
    "min_feature_mm": {"warn": 3.0, "fail": 2.0, "weight": 7},
    # 10. Gradient coverage: fraction of the image that is smooth tonal
    #     transition. Large smooth gradients can band on a DTF transfer.
    "gradient_pct":   {"warn": 25.0, "fail": 50.0, "weight": 4},
    # 11. Margin to canvas edge (% of the shorter side). Content touching the
    #     edge risks being clipped when the film is cut/placed.
    "margin_pct":     {"warn": 2.0, "fail": 0.5, "weight": 4},  # SMALLER is worse
    # 13. DTF transfer physical size limit (inches) â€” the widest film this shop
    #     runs. Larger than this cannot be printed in one piece.
    "max_print_in":   {"warn": 22.0, "fail": 24.0, "weight": 6},  # LARGER is worse
    # Supporting constants (not their own weighted checks, used by the ones above):
    "quantise_colours": 32,      # k for the colour-count quantisation
    "semi_alpha_low": 10,        # alpha in [low, high] counts as semi-transparent
    "semi_alpha_high": 245,
    "edge_soft_band": (30, 120), # legacy (unused): old gradient band for "soft" edges
    "edge_alpha_lo": 10,         # alpha <= this = fully clear (edge transition start)
    "edge_alpha_hi": 245,        # alpha >= this = fully opaque (edge transition end)
    "edge_soft_px": 3.0,         # silhouette transition wider than this = soft
    "default_dpi": 300,          # assumed when the file carries no DPI metadata
    "black_min_lum": 80,         # luminance below this counts as a "dark" pixel
    "black_min_share_pct": 2.0,  # <2% dark content -> pure-black check is n/a
    "analysis_long_side": 1024,  # soft-edge / gradient measured on this size
    "content_alpha": 20,         # alpha above this = content
    "stroke_min_comp_mm": 0.5,   # stroke: ignore components smaller than this (mm square)
    "feature_speck_mm": 1.0,     # feature: ignore components under this in BOTH dims
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
    # Strict '<': a value exactly AT the threshold (e.g. 300 DPI against a
    # 300 DPI warn line) meets the requirement and must pass.
    if value < fail:
        return "fail"
    if value < warn:
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
    in the module docstring. Never raises â€” a broken sub-measurement becomes a
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
        # A measurement over an empty / fully transparent crop can yield NaN or
        # inf (mean/percentile/ratio over zero visible pixels). A non-finite
        # value is not JSON-compliant and crashes the whole /api/jobs response,
        # so NEVER store one: fold it to value=None, status "na" (not scored),
        # value_str "n/a", and note why. This is the source-level guard; the
        # recursive sanitiser on the API responses is the belt-and-braces net.
        try:
            if value is not None and isinstance(value, (int, float)) and not math.isfinite(float(value)):
                value = None
                value_str = "n/a"
                status = "na"
                weight = 0
                detail = (detail + " " if detail else "") + "(not measurable on this crop)"
        except Exception:
            value = None
            value_str = "n/a"
            status = "na"
            weight = 0
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
    if dpi_known:
        # PNG stores DPI as pixels-per-metre (300 DPI -> 11811 ppm), which reads
        # back as 299.9994. Round before ANY comparison or display so a file saved
        # at exactly 300 DPI is not warned against a 300 DPI threshold.
        dpi = float(round(dpi))
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
        dpi_src = "assumed â€” no DPI in file"
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
        soft_pct = _measure_soft_edges(arr, has_alpha, content)
        raw["soft_edge_pct"] = round(soft_pct, 1)
        st = _status_high_bad(soft_pct, T["soft_edge_pct"]["warn"], T["soft_edge_pct"]["fail"])
        add("soft_edges", "Edge sharpness", round(soft_pct, 1), f"{soft_pct:.1f}% soft", "%",
            f"warn >{T['soft_edge_pct']['warn']}%, fail >{T['soft_edge_pct']['fail']}%", st,
            T["soft_edge_pct"]["weight"],
            "% of the silhouette edge whose alpha transition is wider than ~3px")
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
        dark_share = float((cl < T["black_min_lum"]).mean() * 100.0) if cl.size else 0.0
        raw["dark_area_pct"] = round(dark_share, 2)
        if dark_share < T["black_min_share_pct"]:
            # No sizeable dark area: the design has no black to judge, so a
            # "washed-out black" warning would be meaningless. Not scored.
            add("black", "Pure black", None, "n/a (no dark area)", "luminance", "n/a",
                "na", 0, f"only {dark_share:.1f}% of the design is dark - not scored")
        else:
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
        stroke_px = _measure_min_stroke(content, dpi, lum=lum)
        stroke_mm = _px_to_mm(stroke_px, dpi)
        raw.update(min_stroke_px=round(stroke_px, 1), min_stroke_mm=round(stroke_mm, 2))
        st = _status_low_bad(stroke_mm, T["min_stroke_mm"]["warn"], T["min_stroke_mm"]["fail"])
        add("min_stroke", "Minimum stroke width", round(stroke_mm, 2),
            f"{stroke_mm:.2f} mm ({stroke_px:.0f}px)", "mm",
            f"warn <{T['min_stroke_mm']['warn']}mm, fail <{T['min_stroke_mm']['fail']}mm", st,
            T["min_stroke_mm"]["weight"], "thinnest stroke (silhouette or inner detail)")
    except Exception as exc:
        logger.info("printready: stroke measure failed (%s)", exc)
        add("min_stroke", "Minimum stroke width", 0.0, "could not measure", "mm", "-", "warn",
            T["min_stroke_mm"]["weight"], "measurement unavailable")

    # 9. SMALLEST TEXT-LIKE FEATURE HEIGHT (mm) -----------------------------
    try:
        feat_px = _measure_min_feature(content, dpi, lum=lum)
        feat_mm = _px_to_mm(feat_px, dpi)
        raw.update(min_feature_px=round(feat_px, 1), min_feature_mm=round(feat_mm, 2))
        st = _status_low_bad(feat_mm, T["min_feature_mm"]["warn"], T["min_feature_mm"]["fail"])
        add("min_feature", "Smallest feature height", round(feat_mm, 2),
            f"{feat_mm:.2f} mm ({feat_px:.0f}px)", "mm",
            f"warn <{T['min_feature_mm']['warn']}mm, fail <{T['min_feature_mm']['fail']}mm", st,
            T["min_feature_mm"]["weight"], "smallest text/detail (silhouette or inner detail)")
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
    """Halo / fringe: % of a thin ring just OUTSIDE the opaque core that is
    leftover pale background.

    Only ring pixels that are actually VISIBLE count: 0 < alpha < 255. A pixel
    with alpha == 0 prints nothing, so its (often white) RGB is ignored. A
    visible, pale pixel is a halo when it is also clearly lighter than the
    artwork next to it, so light-coloured art does not flag its own edge.
    The ring width scales with the image size (2px at 1024, ~6px at 3225) so an
    upscaled file measures the same as the original. Denominator = the ring."""
    import cv2  # server-only
    alpha = arr[:, :, 3]
    core = alpha >= 250
    if not core.any():
        return 0.0
    h, w = alpha.shape
    r = max(2, int(round(max(h, w) / 512.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(core.astype(np.uint8), kernel, iterations=1)
    ring = (dilated > 0) & (~core)
    n_ring = int(ring.sum())
    if n_ring == 0:
        return 0.0
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    # Brightest neighbouring core luminance (what the fringe sits next to).
    core_lum = np.where(core, lum, 0).astype(np.float32)
    near_core = cv2.dilate(core_lum, kernel)
    # alpha 0 prints nothing, and at very low alpha the 8-bit un-premultiplied
    # RGB is rounding noise, so only pixels with real coverage can be a halo.
    visible = (alpha >= 32) & (alpha < 255)
    pale = ring & visible & (lum > 140) & (lum < 250) & (lum > near_core + 25)
    return float(pale.sum()) / float(n_ring) * 100.0


def _analysis_copy(lum: np.ndarray, content: np.ndarray, long_side: int):
    """Luminance + content mask resized so the LONG side is `long_side`, so
    resolution-dependent measures (edge softness, gradients) do not call an
    upscaled print file 'soft' just because it has more pixels."""
    import cv2
    h, w = lum.shape[:2]
    s = long_side / float(max(h, w))
    if abs(s - 1.0) < 1e-3:
        return lum.astype(np.float32), content
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    l2 = cv2.resize(lum.astype(np.float32), (nw, nh), interpolation=interp)
    c2 = cv2.resize(content.astype(np.uint8) * 255, (nw, nh), interpolation=cv2.INTER_AREA) > 127
    return l2, c2


def _grad_mag(lum: np.ndarray) -> np.ndarray:
    import cv2
    l8 = np.clip(lum, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(l8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l8, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _pseudo_alpha(arr: np.ndarray, content: np.ndarray) -> np.ndarray:
    """Alpha-like 0..255 map for an OPAQUE image: colour distance from the
    border background colour, normalised by the typical distance inside the
    artwork. Lets the same transition-width maths run on images without alpha."""
    rgb = arr[:, :, :3].astype(np.float32)
    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb - bg, axis=2)
    ref = float(np.median(dist[content])) if content.any() else 0.0
    return np.clip(dist / max(ref, 1.0), 0.0, 1.0) * 255.0


def _measure_soft_edges(arr: np.ndarray, has_alpha: bool,
                        content: np.ndarray | None = None) -> float:
    """% of the SILHOUETTE edge that is soft (feathered).

    Interior shading, texture and brush strokes are NOT edges and are ignored.
    Only the content boundary counts: the alpha = 50% contour (for opaque images,
    a pseudo-alpha from the distance to the background colour). At each boundary
    pixel the transition width is the distance to the nearest fully-clear pixel
    plus the distance to the nearest fully-opaque pixel (the two meet at the
    edge, so this is the width measured perpendicular to it), minus 1. Done at a
    fixed ~1024 long side so resolution does not matter.

    crisp = width <= 2px (a normal anti-aliased edge is ~1px); soft = width > 3px.
    Returns the soft share of the boundary, 0..100 (0 if there is no edge)."""
    import cv2
    T = THRESHOLDS
    if has_alpha:
        a = arr[:, :, 3].astype(np.float32)
    else:
        if content is None:
            content, _ = _content_mask(arr)
        a = _pseudo_alpha(arr, content)
    h, w = a.shape
    s = T["analysis_long_side"] / float(max(h, w))
    if abs(s - 1.0) > 1e-3:
        a = cv2.resize(a, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                       interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
    lo, hi = T["edge_alpha_lo"], T["edge_alpha_hi"]

    inside = a >= 128
    if not inside.any():
        return 0.0
    eroded = cv2.erode(inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    boundary = inside & (eroded == 0)
    if not boundary.any():
        return 0.0
    # Distance to the nearest fully-clear / fully-opaque pixel.
    d_clear = cv2.distanceTransform((a > lo).astype(np.uint8), cv2.DIST_L2, 5)
    d_opaque = cv2.distanceTransform((a < hi).astype(np.uint8), cv2.DIST_L2, 5)
    width = d_clear + d_opaque - 1.0
    # Boundary pixels with no opaque core anywhere near (hairlines thinner than a
    # pixel) say nothing about feathering; the stroke check covers those.
    usable = boundary & (d_opaque <= 16.0)
    n = int(usable.sum())
    if n == 0:
        return 0.0
    return float((width[usable] > T["edge_soft_px"]).sum()) / n * 100.0


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


def _internal_detail_mask(lum: np.ndarray, content: np.ndarray,
                          filled: bool = True) -> np.ndarray:
    """Binary mask (uint8 0/1) of the INTERNAL structure inside a solid content
    region: luminance strokes/text the alpha mask alone cannot see.

    A design is often ONE big opaque shape (e.g. a filled badge) with thin text
    and lines drawn ON it. The alpha content mask is then a single blob, so the
    stroke/feature checks only measure the outer silhouette and miss the thin
    inner detail. We recover that detail from the luminance channel, restricted
    to the content (eroded 1px so the outer silhouette edge is not re-counted):

      * `filled=True` (stroke width): the actual MARKS — an adaptive-threshold
        split of both polarities, kept only where near a luminance edge, so the
        returned region is the FILLED stroke (its skeleton thickness is the real
        stroke width). Canny alone would return 1px edge rails and understate the
        width, so it is only used to localise, not as the mask itself.
      * `filled=False` (feature size): the above PLUS the dilated Canny edges, so
        a hollow/outline mark still forms a measurable component.

    Returns all-zeros when there is no meaningful internal structure. Never
    raises."""
    import cv2
    if content.sum() == 0:
        return np.zeros(content.shape, dtype=np.uint8)
    l8 = np.clip(lum, 0, 255).astype(np.uint8)
    # Erode the content enough to clear the anti-aliased silhouette ring, which
    # otherwise leaks in as a 1px "stroke". Scales a little with resolution.
    erode_r = max(3, int(round(max(lum.shape) / 600.0)))
    inner = cv2.erode(content.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_r + 1, 2 * erode_r + 1)))
    if inner.sum() == 0:
        inner = content.astype(np.uint8)
    inner_b = inner > 0
    n_inner = int(inner_b.sum())
    if n_inner == 0:
        return np.zeros(content.shape, dtype=np.uint8)

    # Segment the interior into tonal regions by quantising luminance into bands.
    # The artwork's own fill is the dominant band; every OTHER band is a "mark"
    # (text / line / detail drawn on the fill), whatever its colour. This handles
    # light-on-dark and dark-on-light alike, and a thick mark stays thick while a
    # thin line stays thin — unlike a near-edge ring, which would understate a
    # thick block. Only the inner (silhouette-cleared) region is considered.
    band = (l8.astype(np.int32) // 32)        # 8 tonal bands
    vals = band[inner_b]
    # Dominant band = the artwork fill; everything else inside is a mark.
    counts = np.bincount(vals.ravel(), minlength=8)
    dominant = int(np.argmax(counts))
    marks = inner_b & (band != dominant)
    # A second, adjacent band is usually just the fill's anti-aliasing/shading,
    # not a separate mark; drop a band that is merely one step from dominant AND
    # large, so soft tonal variation in the fill is not read as strokes.
    for adj in (dominant - 1, dominant + 1):
        if 0 <= adj < 8 and counts[adj] > 0.15 * n_inner:
            marks &= (band != adj)

    marks = marks.astype(np.uint8)
    # Opening removes 1px slivers (residual AA) so a genuine >=2px mark survives
    # but a hairline artefact does not.
    marks = cv2.morphologyEx(marks, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    detail = marks.astype(bool)
    if not filled:
        # For feature SIZE, also take the luminance edges so an outline-only mark
        # (no tonal fill difference) still forms a measurable component.
        canny = cv2.Canny(cv2.GaussianBlur(l8, (3, 3), 0), 60, 160)
        detail = detail | ((cv2.dilate(canny, np.ones((2, 2), np.uint8)) > 0) & inner_b)
    return detail.astype(np.uint8)


def _skeleton(cm: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Boolean skeleton of a 0/1 mask. Uses cv2.ximgproc.thinning when the
    contrib build is installed, else the ridge of the distance transform
    (pixels that are a local maximum of the distance map)."""
    import cv2
    try:
        if hasattr(cv2, "ximgproc"):
            return cv2.ximgproc.thinning(cm * 255) > 0
    except Exception:
        pass
    ridge = dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8))
    return ridge & (cm > 0)


def _local_thickness(dist: np.ndarray) -> np.ndarray:
    """Per-pixel local thickness (as an inscribed RADIUS): the largest disc that
    covers the pixel and fits inside the shape. Computed in ~30 geometric radius
    levels. Needed because skeleton spurs run into corners, where the raw
    distance falls toward 1 even though the stroke there is as thick as its
    neighbours."""
    import cv2
    thick = np.zeros_like(dist)
    dmax = float(dist.max())
    if dmax <= 0:
        return thick
    r = 1.0
    levels = []
    while r < dmax:
        levels.append(r)
        r = max(r + 0.5, r * 1.3)
    levels.append(dmax)
    for r in levels:
        seeds = (dist >= r - 1e-3)
        reach = cv2.distanceTransform((~seeds).astype(np.uint8), cv2.DIST_L2, 3) <= r + 0.5
        thick = np.where(reach, np.maximum(thick, r), thick)
    return thick


def _stroke_of_mask(mask: np.ndarray, dpi: float) -> float:
    """5th-percentile stroke width (px, original resolution) of ONE binary mask,
    or 0 when the mask has no qualifying content. Shared by the content-mask and
    internal-detail stroke measurements so both use identical maths."""
    import cv2
    cm = mask.astype(np.uint8)
    if cm.sum() == 0:
        return 0.0
    h, w = cm.shape
    s = min(1.0, 2048.0 / max(h, w))
    if s < 1.0:
        cm = (cv2.resize(cm * 255, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                         interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
    # Drop specks smaller than ~0.5mm x 0.5mm at the print DPI.
    min_area = max(4.0, (THRESHOLDS["stroke_min_comp_mm"] / 25.4 * dpi * s) ** 2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    if keep.any():
        cm = keep[labels].astype(np.uint8)
    if cm.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(cm, cv2.DIST_L2, 5)
    skel = _skeleton(cm, dist)
    if not skel.any():
        skel = cm > 0
    thick = _local_thickness(dist)
    widths = 2.0 * thick[skel] - 1.0
    if widths.size == 0:
        return 0.0
    thin = float(np.percentile(widths, 5)) / s
    return max(thin, 1.0)


def _measure_min_stroke(content: np.ndarray, dpi: float = 300.0,
                        lum: np.ndarray | None = None) -> float:
    """Thinnest stroke width in PIXELS (original resolution).

    Measured on TWO masks and reported as the thinner of the two:
      1. the alpha content mask (the outer silhouette), and
      2. the INTERNAL detail inside that content (luminance edges/strokes), when
         `lum` is given — so thin text/lines drawn on one big solid shape are
         caught even though the alpha mask is a single blob.
    Width is the 5th percentile of the local stroke thickness along each mask's
    skeleton; sub-0.5mm specks are dropped. See _stroke_of_mask."""
    outer = _stroke_of_mask(content, dpi)
    best = outer
    if lum is not None:
        detail = _internal_detail_mask(lum, content, filled=True)
        inner = _stroke_of_mask(detail, dpi)
        if inner > 0 and (best == 0 or inner < best):
            best = inner
    return best


def _feature_of_mask(mask: np.ndarray, dpi: float) -> float:
    """10th-percentile component height (px) of ONE binary mask after dropping
    noise and sub-1mm specks, or 0 when nothing qualifies. Shared by the
    content-mask and internal-detail feature measurements."""
    import cv2
    cm = mask.astype(np.uint8)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
    if n <= 1:
        return 0.0
    MIN_DIM = 3     # a real feature is at least a few px in each direction
    MIN_AREA = 12   # and more than a handful of pixels (drops antialiasing specks)
    speck_px = THRESHOLDS["feature_speck_mm"] / 25.4 * dpi
    heights = []
    for i in range(1, n):
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if h < MIN_DIM or w < MIN_DIM or area < MIN_AREA:
            continue   # noise / antialiasing speck
        if h < speck_px and w < speck_px:
            continue   # decorative speck: under 1mm in both dimensions
        heights.append(h)
    if not heights:
        return 0.0
    return float(np.percentile(heights, 10))


def _measure_min_feature(content: np.ndarray, dpi: float = 300.0,
                         lum: np.ndarray | None = None) -> float:
    """Height (px) of the smallest text-like / detail feature - a legibility
    proxy, reported as the smaller of:
      1. the content mask's components, and
      2. the INTERNAL detail components (luminance edges inside the content),
         when `lum` is given — so small text ON a large solid shape is measured
         even though the whole shape is one alpha component.
    Each uses the same speck/noise filters; see _feature_of_mask."""
    outer = _feature_of_mask(content, dpi)
    best = outer
    if lum is not None:
        detail = _internal_detail_mask(lum, content, filled=False)
        # Close 1px gaps so a dashed/aliased stroke forms one component, then
        # measure its components' heights.
        import cv2
        detail = cv2.morphologyEx(detail, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        inner = _feature_of_mask(detail, dpi)
        if inner > 0 and (best == 0 or inner < best):
            best = inner
    return best


def _measure_gradients(lum: np.ndarray, content: np.ndarray) -> float:
    """% of content that is smooth tonal transition: low-but-nonzero local
    gradient magnitude (not flat, not a hard edge). Measured over content pixels
    on a copy resized to a fixed ~1024 long side."""
    if content.sum() == 0:
        return 0.0
    l2, c2 = _analysis_copy(lum, content, THRESHOLDS["analysis_long_side"])
    n = int(c2.sum())
    if n == 0:
        return 0.0
    mag = _grad_mag(l2)
    smooth = c2 & (mag > 3) & (mag < 25)   # gentle tonal change
    return float(smooth.sum()) / float(n) * 100.0


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
    lines = [f"PRINT READY QA â€” score {score}/100 ({band_label})", ""]
    order = {"fail": 0, "warn": 1, "pass": 2, "na": 3}
    for c in sorted(checks, key=lambda c: (order.get(c["status"], 4), -c["weight"])):
        tag = c["status"].upper().ljust(4)
        line = f"[{tag}] {c['label']}: {c['value_str']}"
        if c["threshold_str"] and c["threshold_str"] not in ("informational", "-", "n/a"):
            line += f"  (threshold: {c['threshold_str']})"
        if c["detail"]:
            line += f"  â€” {c['detail']}"
        lines.append(line)
    return "\n".join(lines)


# For the ChatGPT prompt: a compact measurements block (no pass/fail verbiage â€”
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

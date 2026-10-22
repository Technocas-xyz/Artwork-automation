"""Unit tests for src/pixel_geometry.py — the LOCAL deterministic measurement
engine behind the Pixel Geometry workflow.

Everything measured here must be exact and repeatable (OpenCV/Pillow/numpy only),
and every BEFORE/AFTER field key must always be present (value or None) so the UI
can render an N/A row for anything unmeasurable. These tests lock:
  * a synthetic image with a KNOWN bounding box returns that bbox exactly;
  * bbox + padding add up to the canvas on both axes;
  * every BEFORE and AFTER field key is present;
  * an opaque JPG (no alpha) returns None for the alpha/halo fields (N/A), and a
    transparent PNG returns real alpha counts;
  * the master object assembles and is JSON-serialisable with the file hash set.

Run from the project root:  python tests/test_pixel_geometry.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                      # noqa: E402
from PIL import Image                   # noqa: E402

from src.pixel_geometry import (        # noqa: E402
    measure_geometry, build_master, autofix_block, approval_block,
    BEFORE_FIELDS, AFTER_FIELDS, AUTOFIX_FIELDS, APPROVAL_FIELDS,
)

FAILS: list[str] = []


def check(label: str, cond: bool, extra: object = "") -> None:
    print(("PASS " if cond else "FAIL ") + label + ("" if cond else f"  [{extra}]"))
    if not cond:
        FAILS.append(label)


def transparent_png(W, H, bx, by, bw, bh) -> bytes:
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    a = np.asarray(im).copy()
    a[by:by + bh, bx:bx + bw, :] = (220, 30, 40, 255)   # opaque content block
    buf = io.BytesIO(); Image.fromarray(a, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def opaque_jpg(W, H, bx, by, bw, bh) -> bytes:
    im = Image.new("RGB", (W, H), (255, 255, 255))      # white background
    a = np.asarray(im).copy()
    a[by:by + bh, bx:bx + bw, :] = (15, 30, 190)        # solid content block
    buf = io.BytesIO(); Image.fromarray(a, "RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1) Transparent PNG, synthetic KNOWN bbox -> exact bbox + schema + alpha.
# ---------------------------------------------------------------------------
W, H, BX, BY, BW, BH = 400, 300, 50, 40, 120, 90
png = transparent_png(W, H, BX, BY, BW, BH)
before = measure_geometry(png, inputs={"target_print_width_in": 12.5, "target_dpi": 300},
                          role="before", artwork_id="ART-TEST-1", source_filename="box.png")

check("BEFORE has every field key", all(k in before for k in BEFORE_FIELDS),
      [k for k in BEFORE_FIELDS if k not in before])
check("source size exact", before["source_width_px"] == W and before["source_height_px"] == H,
      (before["source_width_px"], before["source_height_px"]))
check("bbox exact on synthetic PNG",
      (before["source_bbox_x"], before["source_bbox_y"],
       before["source_bbox_width"], before["source_bbox_height"]) == (BX, BY, BW, BH),
      (before["source_bbox_x"], before["source_bbox_y"],
       before["source_bbox_width"], before["source_bbox_height"]))
check("padding + bbox == canvas width",
      before["source_padding_left"] + before["source_bbox_width"] + before["source_padding_right"] == W,
      (before["source_padding_left"], before["source_bbox_width"], before["source_padding_right"]))
check("padding + bbox == canvas height",
      before["source_padding_top"] + before["source_bbox_height"] + before["source_padding_bottom"] == H,
      (before["source_padding_top"], before["source_bbox_height"], before["source_padding_bottom"]))
check("transparent PNG has_alpha True", before["source_has_alpha"] is True)
check("transparent PNG background type Transparent", before["source_background_type"] == "Transparent")
check("orientation Landscape (400x300)", before["source_orientation"] == "Landscape")
# EXACT_RECONSTRUCTION default -> nothing may move + text must match.
check("EXACT mode locks crop/stretch/reposition",
      before["crop_allowed"] is False and before["stretch_allowed"] is False
      and before["reposition_allowed"] is False)
check("EXACT mode sets text_must_match_exactly True", before["text_must_match_exactly"] is True)
# target geometry derived from bbox proportion.
check("target_width_px = 12.5in x 300dpi", before["target_width_px"] == 3750, before["target_width_px"])
check("object_count is a component estimate (>=1)", (before["object_count"] or 0) >= 1,
      before["object_count"])
# V1 vision-only fields are explicitly N/A (None).
for na_key in ("semi_transparency_types", "distress_present", "text_present",
               "text_region_count", "important_object_geometry"):
    check(f"V1 N/A: {na_key} is None", before[na_key] is None, before[na_key])


# ---------------------------------------------------------------------------
# 2) AFTER on a transparent PNG output -> alpha fields are REAL (not N/A).
# ---------------------------------------------------------------------------
out_png = transparent_png(3750, 6000, 500, 700, 2600, 4400)
after_png = measure_geometry(out_png, role="after", before=before)
check("AFTER has every field key", all(k in after_png for k in AFTER_FIELDS),
      [k for k in AFTER_FIELDS if k not in after_png])
check("AFTER output size exact", after_png["output_width_px"] == 3750 and after_png["output_height_px"] == 6000)
check("transparent output: alpha_255_count is a real number",
      isinstance(after_png["alpha_255_count"], int))
check("transparent output: halo_status is a verdict (not N/A)",
      after_png["halo_status"] in ("PASS", "WARN", "FAIL"), after_png["halo_status"])
check("AFTER padding + bbox == output width",
      (after_png["padding_left"] + after_png["output_bbox_width"] + after_png["padding_right"])
      == after_png["output_width_px"])
check("overall_qa_status is a verdict",
      after_png["overall_qa_status"] in ("PASS", "WARN", "FAIL", "N/A"), after_png["overall_qa_status"])
# text + OCR are N/A in V1.
check("V1 N/A: text_status", after_png["text_status"] == "N/A")
check("V1 N/A: text_ocr_match_pct is None", after_png["text_ocr_match_pct"] is None)


# ---------------------------------------------------------------------------
# 3) Opaque JPG -> no alpha, so alpha/halo fields are N/A (None).
# ---------------------------------------------------------------------------
jpg = opaque_jpg(1200, 900, 100, 80, 700, 500)
before_j = measure_geometry(jpg, inputs={}, role="before", artwork_id="ART-TEST-2",
                            source_filename="box.jpg")
check("opaque JPG has_alpha False", before_j["source_has_alpha"] is False)
check("opaque JPG background not Transparent", before_j["source_background_type"] != "Transparent",
      before_j["source_background_type"])
out_jpg = opaque_jpg(3750, 2812, 300, 220, 2200, 1700)
after_j = measure_geometry(out_jpg, role="after", before=before_j)
check("opaque output: alpha_zero_count is None (N/A)", after_j["alpha_zero_count"] is None)
check("opaque output: halo_status is N/A", after_j["halo_status"] == "N/A")
check("opaque output: alpha_status is N/A", after_j["alpha_status"] == "N/A")
check("opaque output still has bbox + padding sum == width",
      (after_j["padding_left"] + after_j["output_bbox_width"] + after_j["padding_right"])
      == after_j["output_width_px"])


# ---------------------------------------------------------------------------
# 4) Master object assembles, is JSON-serialisable, hash + N/A nulls present.
# ---------------------------------------------------------------------------
master = build_master("ART-TEST-1", before, after_png,
                      autofix_block(), approval_block(out_png, before))
check("master has the 5 top keys",
      set(master.keys()) == {"artwork_id", "before", "after", "autofix", "approval"},
      list(master.keys()))
check("autofix block has every field", all(k in master["autofix"] for k in AUTOFIX_FIELDS))
check("approval block has every field", all(k in master["approval"] for k in APPROVAL_FIELDS))
check("approved_file_hash is a 64-char sha256",
      isinstance(master["approval"]["approved_file_hash"], str)
      and len(master["approval"]["approved_file_hash"]) == 64)
check("autofix fields are all None (V1 not implemented)",
      all(master["autofix"][k] is None for k in AUTOFIX_FIELDS))
try:
    json.dumps(master)
    check("master is JSON-serialisable", True)
except Exception as exc:
    check("master is JSON-serialisable", False, exc)


# ---------------------------------------------------------------------------
# Summary / exit code.
# ---------------------------------------------------------------------------
print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): " + ", ".join(FAILS))
    sys.exit(1)
print("ALL PIXEL GEOMETRY TESTS PASSED")

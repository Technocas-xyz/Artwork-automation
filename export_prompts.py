"""Export every prompt in the project to JSON for the prompt versioning DB.

Run from the project folder:   python export_prompts.py

Writes prompts_export.json next to itself. Reads the live config, so the export
is always current - nothing is transcribed by hand.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import config.workflows as W

OUT = Path("prompts_export.json")

# key in config -> (workflow, step order, what it does)
# Step order is the sequence within its workflow, so the DB can show them in
# the order an operator actually hits them.
CATALOGUE: dict[str, tuple[str, int, str]] = {
    # --- Text workflow ---
    "TEXT_TURN_0": ("text", 0, "Read the wording out of an uploaded image"),
    "TEXT_TURN_1": ("text", 1, "Collage of 8 numbered style variations"),
    "TEXT_TURN_2": ("text", 2, "Collage of 8 numbered colour variations"),
    "TEXT_TURN_3": ("text", 3, "Final artwork, transparent PNG"),
    "TEXT_REPLACE_COLLAGE": ("text", 1, "Replace wording in a supplied design, 8 variations"),
    "TEXT_REPLACE_FINAL": ("text", 2, "Final artwork from the chosen replacement variation"),
    "TEXT_IMAGE_ELEMENT_COLLAGE": ("text", 1, "Wording + reference image as an element, 8 variations"),
    "TEXT_IMAGE_STYLE_COLLAGE": ("text", 1, "Wording styled after a reference image, 8 variations"),

    # --- Artwork Extraction ---
    "EXTRACT_CONTACT_SHEET": ("mockup", 1, "One numbered contact sheet of every design on a client sheet"),
    "EXTRACT_SINGLE": ("mockup", 2, "Generate one chosen design from the contact sheet"),
    "EXTRACT_BOXES": ("mockup", 0, "Legacy: ask for bounding boxes as JSON"),
    "EXTRACT_ARTWORKS": ("mockup", 0, "Legacy: return each design as a separate image"),
    "MOCKUP_REGENERATE": ("mockup", 0, "Legacy: regenerate one extracted crop"),

    # --- Artwork Generation ---
    "ARTWORK_REGENERATE": ("artwork", 1, "Clean up a supplied artwork for DTF printing"),

    # --- Artwork Identification ---
    # IMPORTANT: IDENTIFY_OBJECTS returns ONLY JSON [{id,name,cells}]; the agent
    # parses that (falling back to a plain list). Re-run this export and re-import
    # into Decoinks whenever these change, or the DB will keep serving an older
    # version (this is exactly what caused "No items were identified" — a stale
    # plain-list AIS.IDENTIFY.OBJECTS was served instead of the JSON prompt).
    "IDENTIFY_OBJECTS": ("identify", 1, "List objects as JSON with grid cells (object mode)"),
    "IDENTIFY_COLOR_GROUPS": ("identify", 1, "List distinct colour groups (colour mode; usually computed locally)"),
    "IDENTIFY_REGENERATE": ("identify", 2, "Reproduce the artwork with only the operator's per-object changes"),

    # --- Print Ready QA (UC-8) ---
    # Measurements are taken locally (src/printready.py); this is the single
    # ChatGPT turn that gives the printer's written judgement of the artwork.
    "PRINTREADY_REVIEW": ("printready", 1, "Printer's assessment of a DTF artwork, given local measurements"),

    # --- Colorways (UC-6) ---
    # Dominant colours are extracted locally; ChatGPT recommends garment colours,
    # then mocks up the artwork per chosen colour, then adapts it per colour.
    "COLORWAY_SUGGEST": ("colorway", 1, "Recommend garment colours for the artwork, given its dominant colours"),
    "COLORWAY_MOCKUP": ("colorway", 2, "Mock up the artwork on a chosen garment colour"),
    "COLORWAY_ADAPT": ("colorway", 3, "Adapt the artwork to read well on a chosen garment colour"),

    # --- Custom Operations ---
    "CUSTOM_RECONSTRUCT": ("custom", 1, "Redraw the design cleanly at high resolution"),
    "CUSTOM_REMOVE_BACKGROUND": ("custom", 2, "Isolate the design on transparency"),
    "CUSTOM_HALO_REMOVAL": ("custom", 3, "Remove the pale fringe left after background removal"),
    "CUSTOM_BLACK_OUT": ("custom", 4, "Superseded by a local Pillow operation"),
    "CUSTOM_HALF_TONE": ("custom", 5, "Superseded by a local Pillow operation"),
    "CUSTOM_DETECT_OBJECTS": ("custom", 6, "List the recolourable objects in the design"),
    "CUSTOM_CHANGE_COLOR": ("custom", 6, "Recolour the chosen objects"),
    "CUSTOM_ASPECT_ADVICE": ("custom", 7, "Recommend target aspect ratios for this design"),
    "CUSTOM_ASPECT_BASELINE": ("custom", 7, "Clean baseline at the current ratio, for comparison"),
    "CUSTOM_ASPECT_REGENERATE": ("custom", 7, "Regenerate the design at the chosen ratio"),
}

# These now run locally with Pillow/numpy. Their prompt text may still sit in
# config but nothing sends it. Exported with active=false so the DB records
# the history without putting them back into production.
RETIRED = {"CUSTOM_BLACK_OUT", "CUSTOM_HALF_TONE",
           "EXTRACT_BOXES", "EXTRACT_ARTWORKS", "MOCKUP_REGENERATE"}

PLACEHOLDER = re.compile(r"\{(\w+)\}")


def collect() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()

    for key, (workflow, step, purpose) in CATALOGUE.items():
        text = getattr(W, key, None)
        if not isinstance(text, str) or not text.strip():
            continue
        seen.add(key)
        rows.append({
            "key": key.lower(),
            "constant": key,
            "workflow": workflow,
            "step": step,
            "purpose": purpose,
            "template": text,
            "placeholders": sorted(set(PLACEHOLDER.findall(text))),
            "version": 1,
            "active": key not in RETIRED,
            "source": "config/workflows.py",
        })

    # CUSTOM_OPERATIONS carries its templates inline, so pull those too and let
    # them win over any same-named constant - the UI reads this structure.
    for op in getattr(W, "CUSTOM_OPERATIONS", []) or []:
        tpl = (op.get("template") or "").strip()
        if not tpl:
            continue
        key = f"custom_{op['key']}"
        rows = [r for r in rows if r["key"] != key]
        rows.append({
            "key": key,
            "constant": f"CUSTOM_OPERATIONS[{op['key']!r}]",
            "workflow": "custom",
            "step": 0,
            "purpose": op.get("label") or op["key"],
            "template": tpl,
            "placeholders": sorted(set(PLACEHOLDER.findall(tpl))),
            "version": 1,
            "active": True,
            "source": "config/workflows.py :: CUSTOM_OPERATIONS",
        })

    # Anything in config we did not catalogue - so a new prompt is never missed.
    for name in dir(W):
        if name.startswith("_") or name in seen:
            continue
        val = getattr(W, name)
        if isinstance(val, str) and len(val) > 80 and name.isupper():
            rows.append({
                "key": name.lower(),
                "constant": name,
                "workflow": "uncatalogued",
                "step": 0,
                "purpose": "NOT CATALOGUED - assign a workflow before import",
                "template": val,
                "placeholders": sorted(set(PLACEHOLDER.findall(val))),
                "version": 1,
                "active": False,
                "source": "config/workflows.py",
            })

    rows.sort(key=lambda r: (r["workflow"], r["step"], r["key"]))
    return rows


def main() -> None:
    rows = collect()
    doc = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project": "artwork-automation",
        "note": (
            "Each row is one prompt at version 1. Only the active version of a "
            "prompt should be fetched at runtime. Rows with active=false are "
            "retired - kept for history, not for production."
        ),
        "count": len(rows),
        "prompts": rows,
    }
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {OUT} - {len(rows)} prompts\n")
    by_wf: dict[str, int] = {}
    for r in rows:
        by_wf[r["workflow"]] = by_wf.get(r["workflow"], 0) + 1
    for wf, n in sorted(by_wf.items()):
        print(f"  {wf:<14} {n}")

    retired = [r["key"] for r in rows if not r["active"]]
    if retired:
        print("\nRetired (active=false):")
        for k in retired:
            print(f"  {k}")

    unc = [r["key"] for r in rows if r["workflow"] == "uncatalogued"]
    if unc:
        print("\nNot catalogued - assign a workflow before importing:")
        for k in unc:
            print(f"  {k}")


if __name__ == "__main__":
    main()

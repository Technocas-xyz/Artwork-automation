"""UC-6 Colorways — LOCAL colour helpers (no ChatGPT).

Two local jobs, mirroring how Print Ready QA / colour-group identification keep
the exact/measurable work off ChatGPT:

  * extract_colours(image_bytes) — the artwork's dominant colours + percentages
    and whether it has transparency. Reuses src.annotate.colour_groups (OpenCV
    k-means), so this needs cv2 and runs SERVER-SIDE at job creation, exactly
    like _precompute_identify_colour.

  * composite_on_swatch(artwork_bytes, hex, ...) — a pixel-accurate local
    PREVIEW of the ACTUAL artwork centred on a flat swatch of a garment colour.
    Pillow only, so the agent can call it too. This sits beside ChatGPT's mockup
    (which REDRAWS the design) so the operator sees the real file on that colour.

  * garment_hex(name) — map a garment colour NAME (black, navy, heather grey,
    sand, forest green, ...) to a hex value for the swatch, returning None when
    the name is unknown so the caller can ask the operator for a hex.

cv2 is only touched inside extract_colours (via annotate.colour_groups), so
importing this module never requires OpenCV.
"""

from __future__ import annotations

import io
import re
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------------
# Garment colour NAME -> hex. Common DTF garment colours. Names are matched
# case-insensitively and loosely (see garment_hex): "Heather Grey", "heather
# gray", "athletic heather" all resolve. Unknown names return None so the
# operator is asked for a hex.
# ---------------------------------------------------------------------------
GARMENT_HEX: dict[str, str] = {
    "white": "#FFFFFF",
    "black": "#111111",           # true garment black is not pure #000
    "navy": "#1F2A44",
    "navy blue": "#1F2A44",
    "royal": "#1E3FA0",
    "royal blue": "#1E3FA0",
    "blue": "#2A4B9B",
    "light blue": "#8FB3D9",
    "sky blue": "#8FC7E8",
    "red": "#B02A2A",
    "cardinal red": "#8E2436",
    "maroon": "#5A2333",
    "burgundy": "#5A2333",
    "orange": "#D2622A",
    "gold": "#C79A3B",
    "yellow": "#E6C233",
    "green": "#2F7A44",
    "kelly green": "#2F8F4E",
    "forest green": "#22432C",
    "olive": "#5A5A32",
    "military green": "#4A503A",
    "purple": "#5A3B8C",
    "pink": "#E39BB5",
    "light pink": "#F2C6D6",
    "hot pink": "#E24E8A",
    "grey": "#8C8C8C",
    "gray": "#8C8C8C",
    "sport grey": "#B6B6B6",
    "sport gray": "#B6B6B6",
    "heather grey": "#B9BCC0",
    "heather gray": "#B9BCC0",
    "athletic heather": "#B9BCC0",
    "dark grey": "#4A4A4A",
    "dark gray": "#4A4A4A",
    "charcoal": "#36393B",
    "graphite": "#3A3D42",
    "sand": "#D8C6A8",
    "natural": "#E7DEC8",
    "cream": "#F1E9D2",
    "tan": "#C8AD82",
    "khaki": "#B7A579",
    "brown": "#5C4433",
    "chocolate": "#4A362A",
    "teal": "#2C6E6E",
    "turquoise": "#3FB0A5",
    "mint": "#A8D5BA",
}


def garment_hex(name: str) -> str | None:
    """Best-effort NAME -> hex for a garment colour.

    Match order: exact (lowercased/trimmed), then a word-overlap heuristic so
    "Heather Grey Marl" or "dark navy" still resolve to the closest known key.
    Returns None when nothing matches so the caller can prompt for a hex."""
    if not name:
        return None
    key = re.sub(r"\s+", " ", name.strip().lower())
    if key in GARMENT_HEX:
        return GARMENT_HEX[key]
    # Strip trailing marketing words that don't change the colour.
    key2 = re.sub(r"\b(t-?shirt|tee|garment|fabric|marl|melange|solid)\b", "", key).strip()
    key2 = re.sub(r"\s+", " ", key2)
    if key2 in GARMENT_HEX:
        return GARMENT_HEX[key2]
    # Word-overlap: prefer the known key sharing the most words with the input,
    # so "dark navy" -> "navy", "athletic heather grey" -> "heather grey". The
    # LAST word of the input is usually the colour noun ("dark NAVY"), so a key
    # that matches it is preferred over one matching only a modifier ("DARK").
    in_word_list = key2.split()
    in_words = set(in_word_list)
    last_word = in_word_list[-1] if in_word_list else ""
    best, best_score = None, (-1, -1, -1)
    for k, hexv in GARMENT_HEX.items():
        kw = set(k.split())
        overlap = len(kw & in_words)
        if not overlap:
            continue
        matches_last = 1 if last_word in kw else 0
        # Rank: (matches the colour noun, word overlap, key length).
        score = (matches_last, overlap, len(kw))
        if score > best_score:
            best, best_score = hexv, score
    return best


def _hex_to_rgb(hexv: str) -> tuple[int, int, int]:
    h = (hexv or "").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (128, 128, 128)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return (128, 128, 128)


def composite_on_swatch(artwork_bytes: bytes, hexv: str, size: int = 600,
                        margin_frac: float = 0.12) -> bytes:
    """Composite the ACTUAL artwork, centred, onto a flat square swatch of the
    garment colour. Pillow only — pixel-accurate, no redraw.

    The artwork is scaled to fit within (1 - 2*margin_frac) of the swatch,
    preserving aspect, and alpha-composited so a transparent PNG shows the
    garment colour through it (exactly what a real print does). Returns PNG
    bytes. Never raises for a normal image; on any failure returns a plain
    swatch so the preview slot is never empty."""
    r, g, b = _hex_to_rgb(hexv)
    swatch = Image.new("RGBA", (size, size), (r, g, b, 255))
    try:
        art = Image.open(io.BytesIO(artwork_bytes)).convert("RGBA")
    except Exception:
        out = io.BytesIO(); swatch.convert("RGB").save(out, format="PNG"); return out.getvalue()

    avail = int(size * (1.0 - 2.0 * margin_frac))
    aw, ah = art.size
    if aw <= 0 or ah <= 0:
        out = io.BytesIO(); swatch.convert("RGB").save(out, format="PNG"); return out.getvalue()
    scale = min(avail / aw, avail / ah)
    new_w = max(1, int(round(aw * scale)))
    new_h = max(1, int(round(ah * scale)))
    art_resized = art.resize((new_w, new_h), Image.LANCZOS)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    # Alpha-composite so transparency reveals the garment colour underneath.
    swatch.alpha_composite(art_resized, (ox, oy))
    out = io.BytesIO()
    swatch.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def extract_colours(image_bytes: bytes) -> dict:
    """The artwork's dominant colours + percentages and whether it has
    transparency. Reuses annotate.colour_groups (OpenCV k-means over opaque
    pixels) so a single implementation feeds both Colorways and colour-group
    Identification. Returns:
        {"colours": [{"name","hex","rgb":[r,g,b],"pct"}...],
         "has_transparency": bool}
    Never raises — on failure returns empty colours (the UI shows a plain note
    and ChatGPT still gets a visual-only recommendation)."""
    try:
        from src.annotate import colour_groups, has_transparency
        result = colour_groups(image_bytes)
        groups = result.get("groups", []) or []
        colours = [{"name": g.get("name", ""), "hex": g.get("hex", ""),
                    "rgb": g.get("rgb", []), "pct": g.get("pct", 0)} for g in groups]
        try:
            transparent = bool(has_transparency(image_bytes))
        except Exception:
            transparent = False
        return {"colours": colours, "has_transparency": transparent}
    except Exception:
        return {"colours": [], "has_transparency": False}


def colours_for_prompt(colours: list[dict]) -> str:
    """Format the extracted colours as the {colours} block for COLORWAY_SUGGEST.
    A plain, readable list of name + hex + percentage — data for ChatGPT to judge
    against, never a re-measurement request."""
    if not colours:
        return "(no dominant colours could be extracted; judge from the image)"
    lines = []
    for c in colours:
        name = c.get("name") or "colour"
        hexv = c.get("hex") or ""
        pct = c.get("pct")
        pct_str = f" — {pct}%" if pct is not None else ""
        lines.append(f"- {name} {hexv}{pct_str}".rstrip())
    return "\n".join(lines)


def slug(name: str) -> str:
    """A filesystem/id-safe slug for a garment colour name (for stage-image keys
    and vault filenames): 'Heather Grey' -> 'heather_grey'."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return s or "colour"

"""Workflow prompt templates for multi-turn generation.

Placeholders:
    {text} — operator-entered design text
    {n}    — style variation number chosen after turn 1
    {m}    — colour variation number chosen after turn 2
"""

from __future__ import annotations

TEXT_TURN_0 = """Look at this image and extract the exact text content from it.

Reply with ONLY the text, nothing else. No explanation, no quotes, no formatting. If there are multiple lines, preserve the line breaks."""

TEXT_TURN_1 = """Create a single collage image containing 8 different text design variations of the text "{text}".

Requirements:
- Arrange them in a 4 wide x 2 tall grid
- Each variation must have a clearly visible number from 1 to 8 placed directly below it, outside the design itself
- Each variation should use a distinctly different typography style: vary between bold script, graffiti, block letters, retro, modern sans, outlined, distressed, and 3D
- Plain white background for the whole collage
- Each design in black only, no colour at this stage
- Designs must be suitable for DTF garment printing: clean edges, no thin fragile strokes, no gradients"""

TEXT_TURN_2 = """Take variation number {n} only.

Create a single new collage showing 8 colour variations of that exact same design. Do not change the typography, letterforms, or layout in any way -- only the colours change.

Requirements:
- Same 4 wide x 2 tall grid
- Number each one 1 to 8, placed below the design, outside it
- Use 8 distinctly different colour treatments suitable for garment printing on both light and dark fabric
- Plain white background for the collage
- Keep colours flat and solid, no gradients or shadows"""

TEXT_TURN_3 = """Generate colour variation number {m}, which is in row {row}, position {col} from the left, counting the 4x2 grid left to right then top to bottom.

Confirm which variation you are generating before producing the image.

Requirements:
- The design only, nothing else in the frame
- Transparent background, PNG
- No numbering, no border, no background colour
- Flat solid colours exactly as shown in that variation
- Centred with even margins"""

MOCKUP_REGENERATE = """Regenerate this artwork as a clean final version suitable for DTF garment printing.

Requirements:
- Keep the design identical - do not change the layout, letterforms, colours, or illustration style
- Preserve all text exactly as it appears, including any accented characters
- Clean sharp edges, no gradients, no thin fragile strokes
- Transparent background, PNG
- The design only, centred with even margins
- Remove any sequence number badge or watermark that came from the sheet layout, such as a small numbered box in a corner. It is not part of the design.
- If the source shows the design printed on fabric, remove all fabric colour and texture. Output the design alone on a transparent background."""

EXTRACT_BOXES = """This sheet contains multiple separate artwork designs.

Give me the exact pixel bounding box of every design.

IMPORTANT: If the designs are shown printed on garments, give me the bounding box of the PRINTED GRAPHIC ONLY - not the garment, and not any product label, size chart, colour swatch or caption text near it. The box must contain the complete graphic including its outermost letters and edges, and nothing else.

Ignore any header, footer, branding, or watermark that belongs to the sheet itself rather than to a design.

Reply with ONLY a JSON array, no explanation, no markdown fences:
[{{"n":1,"x":0,"y":0,"w":0,"h":0}}]

Coordinates must be in the original image's pixel dimensions, which are {width} x {height}. Order them left to right, then top to bottom.

Each box must contain one design only, with no part of any neighbouring design included."""

EXTRACT_ARTWORKS = """Analyze the uploaded client reference image and extract every separate printable artwork/design visible in it.

Rules:
- Extract ALL artworks, not just one. If there are ten garments, return ten.
- If artwork is printed on T-shirts, hoodies, shorts, hats etc., extract only the printed design, not the garment. Exclude collar, sleeves, hem and all blank fabric.
- Remove mockup background, product numbers, captions, colour swatches, labels, branding and any other non-artwork elements.
- Keep all text, illustrations, logos and decorative elements that belong to each design.
- Do not combine separate designs or split one complete design.
- Return each artwork as a separate image with a transparent background.
- Scan the entire reference image carefully so no artwork is missed.

Generate each artwork as its OWN separate image file. Do not combine several designs into one picture, do not arrange them in a grid or collage, and do not return the original sheet back to me. If there are ten designs, generate ten separate images, one at a time."""

EXTRACT_CONTACT_SHEET = """Analyze the uploaded client reference image and extract every separate printable artwork/design visible in it.

Rules:
- Find ALL artworks. If there are ten garments, show ten.
- If artwork is printed on T-shirts, hoodies, shorts, hats etc., extract only the printed design, not the garment. Exclude collar, sleeves, hem and all blank fabric.
- Remove mockup background, product numbers, captions, colour swatches, labels, branding and any other non-artwork elements.
- Keep all text, illustrations, logos and decorative elements that belong to each design.
- Preserve each artwork as accurately as possible.

Arrange all the extracted designs into ONE single collage image on a plain white background, laid out in a neat grid.
Number each design clearly from 1 upward, with the number placed directly below its design, outside the artwork itself.
Return only that one collage image. Do not return the original sheet back to me, and do not return the designs as separate files."""

EXTRACT_SINGLE = """Generate design number {n} from the collage as a single final artwork.

Requirements:
- That design only, nothing else in the frame
- Reproduce it as accurately as possible, preserving all text, letterforms, colours and illustration detail
- Transparent background, PNG
- No numbering, no border, no background colour
- Centred with even margins"""

ARTWORK_REGENERATE = """Regenerate this artwork as a clean final version suitable for DTF garment printing.

Requirements:
- Keep the design identical - do not change the layout, letterforms, colours, or illustration style
- Preserve all text exactly as it appears, including accented characters and any numbers such as verse or scripture references
- Clean sharp edges, no gradients, no thin fragile strokes
- Transparent background, PNG
- The design only, centred with even margins"""

# ---------------------------------------------------------------------------
# Custom Operation workflow prompts (TODO: business to supply final wording)
# ---------------------------------------------------------------------------

CUSTOM_RECONSTRUCT = """Reconstruct the uploaded artwork as a clean, high-resolution, print-ready version.

- Redraw the existing design accurately; do not redesign, reinterpret, simplify, or add anything.
- Preserve the exact layout, proportions, letterforms, colours, shapes, illustration style, and relative positioning.
- Preserve all text exactly as shown, including accented characters, punctuation, spelling, and numbers.
- Correct only defects caused by low resolution, compression, pixelation, blur, rough edges, or poor source quality.
- Use clean sharp edges and solid printable shapes. Do not introduce gradients, shadows, textures, or thin fragile strokes that are not already part of the artwork.
- Transparent background.
- Design only, centred with even margins.
- Output the finished reconstructed artwork as a high-resolution transparent PNG suitable for DTF printing."""

CUSTOM_REMOVE_BACKGROUND = """Remove the background from the uploaded artwork and isolate the printable design only.

- Make all background areas fully transparent.
- Keep every design element exactly unchanged: layout, proportions, letterforms, colours, illustration style, and details.
- Preserve all text exactly, including accented characters, punctuation, and numbers.
- Do not remove white or light-coloured elements that are part of the artwork.
- Create clean, sharp edges with no leftover background pixels, colour contamination, or semi-transparent fringe.
- Prepare for DTF printing: no gradients introduced, no thin fragile strokes, transparent background, design centred with even margins.
- Output only the finished artwork as a transparent PNG."""

CUSTOM_HALO_REMOVAL = """Remove the pale/white colour halo, matte fringe, and contaminated edge pixels around the uploaded artwork.

- Remove only unwanted edge fringe created by previous background removal.
- Decontaminate the edges so no white, pale, grey, or previous-background colour remains around the design.
- Make pixels outside the true artwork boundary fully transparent.
- Keep intentional white, light-coloured, anti-aliased, or internal design elements intact.
- Do not shrink, erode, reshape, redraw, or otherwise alter the artwork.
- Preserve the exact layout, letterforms, colours, illustration style, text, accented characters, and numbers.
- Produce clean sharp DTF-ready edges with no visible halo on either light or dark garments.
- Transparent background, design only, centred with even margins.
- Output only the cleaned transparent PNG."""

CUSTOM_BLACK_OUT = """Convert the uploaded artwork into a single-colour solid black spot-colour separation.

- Preserve the exact silhouette, layout, proportions, letterforms, shapes, spacing, and all design details.
- Convert every printable design element to 100% solid black (#000000) at full opacity.
- Treat the result as a single spot-colour separation: no grayscale, tints, screens, gradients, colour variation, or semi-transparent pixels.
- Preserve all text exactly, including accented characters, punctuation, and numbers.
- Do not merge separate elements if doing so changes the artwork's shapes or negative spaces.
- Maintain clean knockout/negative areas wherever they exist in the original design.
- Use clean sharp edges and production-safe printable strokes.
- Transparent background.
- Design only, centred with even margins.
- Output only the finished single-colour black artwork as a transparent PNG suitable for DTF or screen-print production."""

CUSTOM_HALF_TONE = """Convert the uploaded artwork into a production-ready spot-colour halftone treatment while preserving the original design.

- Preserve the exact layout, silhouettes, letterforms, text, proportions, colours, illustration style, and composition.
- Preserve all text exactly, including accented characters, punctuation, and numbers.
- Convert tonal, shaded, faded, or variable-opacity areas into AM halftone dots, using dot size to reproduce tonal value.
- Use clean round halftone dots with a consistent screen ruling appropriate for textile printing, approximately 35-45 LPI at final print size, with a 22.5 degree screen angle where a single screen is used.
- Do not use continuous-tone grayscale, stochastic noise, diffusion dithering, simulated grain, or semi-transparent pixels to create shading.
- Printed dots must be solid and fully opaque; tonal appearance must come from dot size and spacing.
- Keep highlight knockouts and open areas fully transparent.
- Avoid dots or bridges so small that they are fragile or unreliable in garment production.
- Do not halftone areas that should remain solid unless required by the existing artwork.
- Transparent background, clean sharp edges, design only, centred with even margins.
- Output only the finished high-resolution transparent PNG suitable for DTF/textile print production."""

CUSTOM_DETECT_OBJECTS = """Inspect the uploaded artwork and list the distinct editable objects or colour regions that can be individually recoloured.

Use simple descriptive names that clearly identify each object, such as "Main text", "Basketball", "Outer outline", "Left flower", or "Shirt illustration".

Do not include the transparent background as an object. Do not combine visually separate elements if they could reasonably be recoloured independently.

Reply with ONLY a plain numbered list in this format:
1. Object name
2. Object name
3. Object name

No explanation, headings, colours, descriptions, or additional text."""

CUSTOM_CHANGE_COLOR = """Change only the following objects in the uploaded artwork to the specified colours:

{changes}

- Modify only the specified objects.
- Keep every unspecified object exactly unchanged.
- Change colour only; do not alter shapes, outlines, proportions, positioning, typography, texture, illustration style, or layout.
- Preserve all text exactly, including accented characters, punctuation, and numbers.
- Preserve intentional outlines, highlights, shadows, knockouts, and internal details unless the colour instruction explicitly includes them.
- Apply the requested colours cleanly and consistently without contaminating neighbouring objects.
- Use solid print-safe colour areas with clean sharp edges; do not introduce gradients or semi-transparent pixels.
- Transparent background.
- Design only, centred with even margins.
- Output only the finished recoloured artwork as a transparent PNG."""

# Aspect Ratio Enhancement: ChatGPT only ADVISES; the resize is done locally
# with Pillow (transparent padding, never crop, never stretch).
CUSTOM_ASPECT_ADVICE = """This artwork is currently {width} x {height} pixels, an aspect ratio of {ratio}, which prints at {inches_w} x {inches_h} inches at {dpi} DPI.

Recommend the best aspect ratio for printing this design on garments. Consider standard DTF transfer sizes and common placements: full front, left chest, back, sleeve. Take into account the shape of the design itself - a wide design suits a different placement from a tall one.

Reply with 2 to 4 recommendations. For each one give the ratio, the print size in inches, the placement it suits, and one short sentence on why.

Plain text, no markdown, no preamble."""

CUSTOM_ASPECT_BASELINE = """Regenerate this artwork as a clean, print-ready version at its current proportions.

Requirements:
- Keep the design identical - same layout, letterforms, colours and illustration style
- Preserve all text exactly, including accented characters and numbers
- Remove the background entirely - transparent PNG
- Do not change the aspect ratio or recompose the design
- Clean sharp edges, no gradients, no thin fragile strokes
- The design only, centred with even margins"""

CUSTOM_ASPECT_REGENERATE = """Working from the cleaned, background-free version you just produced, regenerate that same artwork at an aspect ratio of {ratio}, which is {inches_w} x {inches_h} inches at {dpi} DPI.

Requirements:
- Keep the design identical to the cleaned version - same layout, letterforms, colours and illustration style
- Preserve all text exactly, including accented characters and numbers
- Recompose to fit the new proportions without stretching or distorting anything
- Transparent background, PNG
- The design only, centred with even margins
- Clean sharp edges suitable for DTF garment printing"""


def normalise_ratio(w: float, h: float) -> str:
    """Format a ratio as 'W:H (1 : X.XX)' by dividing both sides by the smaller.

    e.g. 177:248 -> '177:248 (1 : 1.40)'. The '1' is always the smaller side.
    Applied everywhere a ratio is shown.
    """
    if w <= 0 or h <= 0:
        return f"{w}:{h}"
    rw = int(w) if float(w).is_integer() else round(w, 2)
    rh = int(h) if float(h).is_integer() else round(h, 2)
    smaller = min(w, h)
    factor = max(w, h) / smaller
    if w <= h:
        return f"{rw}:{rh} (1 : {factor:.2f})"
    return f"{rw}:{rh} ({factor:.2f} : 1)"


# Execution order for custom operations.
# aspect_ratio runs LAST so any redraw/recolour/halftone steps happen on the
# original canvas first, then the final canvas shape is padded to target.
CUSTOM_OPERATIONS_ORDER = [
    "reconstruct",
    "remove_background",
    "halo_removal",
    "black_out",
    "half_tone",
    "change_object_color",
    "aspect_ratio",
]

CUSTOM_OPERATIONS_LABELS = {
    "reconstruct": "Reconstruct",
    "remove_background": "Remove Background",
    "halo_removal": "Halo Removal",
    "black_out": "Black Out",
    "half_tone": "Half Tone",
    "change_object_color": "Change Object Colour",
    "aspect_ratio": "Aspect Ratio Enhancement",
}

# Single-sourced operation definitions: key, label, description, template.
# The UI renders directly from this list. No key construction needed.
# TODO: Migrate the other workflows (Text, Extraction, Artwork) to this same pattern.
CUSTOM_OPERATIONS = [
    {"key": "reconstruct", "label": "Reconstruct", "desc": "Redraw the design cleanly", "template": CUSTOM_RECONSTRUCT},
    {"key": "remove_background", "label": "Remove Background", "desc": "Isolate on transparent", "template": CUSTOM_REMOVE_BACKGROUND},
    {"key": "halo_removal", "label": "Halo Removal", "desc": "Remove pale fringe from edges", "template": CUSTOM_HALO_REMOVAL},
    # black_out and half_tone are DETERMINISTIC LOCAL pixel operations (Pillow/numpy).
    # They do NOT use ChatGPT and carry NO prompt template, so the UI hides the
    # prompt disclosure. half_tone exposes advanced settings (lpi/angle/dot).
    {"key": "black_out", "label": "Black Out", "desc": "Makes black areas transparent so the garment shows through. For printing on black garments.", "local": True,
     "settings": {"threshold": 40}},
    {"key": "half_tone", "label": "Half Tone", "desc": "Halftone dot treatment (local, instant)", "local": True,
     "settings": {"lpi": 40, "angle": 22.5, "dot": "round"}},
    # change_object_color is a TWO-TURN operation and therefore carries TWO distinct templates:
    #   template_detect -> Step A (list objects, text reply)
    #   template_apply  -> Step B (apply chosen colours, {changes} placeholder, image reply)
    # The single-turn ops keep a plain "template" field.
    {"key": "change_object_color", "label": "Change Object Colour", "desc": "Recolour specific elements",
     "two_turn": True,
     "template_detect": CUSTOM_DETECT_OBJECTS,
     "template_apply": CUSTOM_CHANGE_COLOR},
    # aspect_ratio is a LOCAL/advisory operation: ChatGPT only recommends a ratio,
    # the padding is done with Pillow. It shows file info on tick and pauses for a
    # ratio decision. The "advice" template carries the info placeholders.
    {"key": "aspect_ratio", "label": "Aspect Ratio Enhancement", "desc": "Pad canvas to a print-friendly ratio (local, never crops)",
     "advisory": True,
     "template": CUSTOM_ASPECT_ADVICE},
]

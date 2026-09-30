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

# --- Text workflow, "Replace text in a design" mode ------------------------
# Two turns only (collage -> final). There is NO colour stage: the design's
# colours already exist and must be preserved. {target_clause} is built in code
# (see agent.py) from the optional "text to replace" field.
TEXT_REPLACE_COLLAGE = """This image is a finished design. Recreate it with the wording changed to "{new_text}".

Requirements:
- Keep the design identical in every other respect: layout, illustration style, colours, decorative elements, background treatment and composition
- Replace only the wording. {target_clause}
- Match the original typography as closely as possible - same style, weight and character of lettering
- Produce 8 variations showing different ways the new wording can sit in the design: spacing, scale, slight positioning and lettering treatment
- Arrange all 8 into ONE collage on a plain white background in a 4 wide x 2 tall grid
- Number each variation 1 to 8, placed below it, outside the artwork
- Return only that one collage image"""

TEXT_REPLACE_FINAL = """Generate variation number {n} as a single final artwork.

Requirements:
- That variation only, nothing else in the frame
- Keep the design and the new wording exactly as shown in that variation
- Transparent background, PNG
- No numbering, no border
- Centred with even margins
- Clean sharp edges suitable for DTF garment printing"""


# --- Text workflow, "wording with a client-supplied image" modes (UC-3) -----
# Two ways the reference image is used at stage 1. Turns 2 (colour) and 3
# (final) reuse TEXT_TURN_2 / TEXT_TURN_3 unchanged — the colour stage and the
# final are the same regardless of how stage 1 was produced. The reference image
# is attached on turn 1 only.
TEXT_IMAGE_ELEMENT_COLLAGE = """Create a single collage image containing 8 different design variations that combine the uploaded image with the wording "{text}".

Requirements:
- The uploaded image must appear as part of each design. Reproduce it as closely as you can - same subject, same character, same detail
- Vary how the image and wording are arranged: above, below, behind, wrapped, integrated into the lettering
- Vary the typography across the 8 so the client has a real choice
- Arrange them in a 4 wide x 2 tall grid
- Number each variation 1 to 8, placed directly below it, outside the design
- Plain white background for the whole collage
- Designs must be suitable for DTF garment printing: clean edges, no thin fragile strokes, no gradients"""

TEXT_IMAGE_STYLE_COLLAGE = """Create a single collage image containing 8 different text design variations of the wording "{text}", styled after the uploaded image.

Requirements:
- Take the visual style from the uploaded image: its colour character, texture, era, mood and lettering treatment
- Do NOT include the uploaded image itself, or any part of its subject matter, in the designs. Only the wording appears
- Vary the typography across the 8 within that style
- Arrange them in a 4 wide x 2 tall grid
- Number each variation 1 to 8, placed directly below it, outside the design
- Plain white background for the whole collage
- Designs must be suitable for DTF garment printing: clean edges, no thin fragile strokes, no gradients"""


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

# Artwork Identification: turn 1 lists every distinct object; the operator gives
# an instruction per object; turn 2 regenerates with those changes applied.
IDENTIFY_OBJECTS = """A labelled 10x10 grid has been drawn over this artwork to help you describe where things are. Columns are lettered A to J from left to right; rows are numbered 1 to 10 from top to bottom. So the top-left cell is A1 and the bottom-right cell is J10. The grid is only a reference overlay — ignore it as part of the artwork.

List the main objects and elements in the artwork — the things a person would actually want to change. List whole, meaningful elements: each subject, object, piece of text, logo, and distinct background element. Do NOT break a single subject into its parts (a lion is one item called "Lion", never its mane, eyes, paws or tail separately). Aim for roughly 5 to 15 items.

For each object, give the grid cells it mainly sits in. Use only the cells that the object actually covers; if you are unsure, give your best estimate of the few cells at its centre rather than listing many.

Reply with ONLY a JSON array and nothing else — no prose, no explanation, no markdown code fences. Each element is an object with: "id" (1-based integer), "name" (short label), and "cells" (array of cell strings). Exactly this shape:

[
  {{"id": 1, "name": "Lion", "cells": ["D4", "E5"]}},
  {{"id": 2, "name": "Team name text", "cells": ["C8", "D8", "E8"]}}
]

Output the JSON array only."""

IDENTIFY_COLOR_GROUPS = """Look at this artwork and list every distinct colour group in it.

A colour group is all the areas sharing one colour, even when they belong to different elements. Include outlines, fills, text colours, background colours and shadow tones.

Use short names that identify each group clearly, for example "Red lettering", "Dark blue background", "Gold outlines", "Skin tones".

Reply with ONLY a plain numbered list in this format:
1. Colour group name
2. Colour group name
3. Colour group name

No explanation, no headings, no hex codes, no extra text."""

IDENTIFY_REGENERATE = """Reproduce this artwork exactly as it is, with ONLY the following specific changes applied.

This is a reproduction task, not a new artwork. Keep everything that is not named in the change list below exactly as it appears in the original: the background, scenery, all other subjects, the composition, framing, colour palette, lighting, artistic style, and every element not named in a change. Preserve all text that is not being changed, including accented characters.

Do not simplify the scene. Do not remove anything that was not asked to be removed. The result must be recognisably the same artwork, changed only in the ways listed.

Each change below names the object by its number and name, so you know exactly which element it refers to. Apply the changes to those objects only:

{changes}

If a change is impossible without altering other parts, make the smallest possible alteration and keep everything else.

Output requirements (do not let these change the content above):
- {background}
- Clean sharp edges suitable for DTF garment printing"""


# Print Ready QA (UC-8): the artwork is uploaded with a full set of LOCAL
# measurements (src/printready.py). ChatGPT is given the numbers and the image
# and asked ONLY for the printer's judgement — never to re-measure. {measurements}
# is filled with the formatted measurement block.
PRINTREADY_REVIEW = """You are reviewing artwork for DTF garment printing.

Here are the measurements taken from the file:

{measurements}

Look at the artwork alongside these numbers and give a printer's assessment.

Cover:

- Whether this will print cleanly on a garment, and on which garment colours

- Any element that is too fine, too thin or too small to survive the transfer

- Whether the text will remain legible at the stated print size

- Anything the measurements cannot see: awkward composition, elements that

  will disappear against fabric, detail that will fill in

- What specifically to fix, in order of importance

Be concrete. Name the element you mean. Do not repeat the numbers back - I

have them. If the artwork is print ready, say so plainly.

Plain text, no markdown, no preamble."""


# Colorways (UC-6): the artwork's dominant colours are extracted LOCALLY and
# handed to ChatGPT as data ({colours}); ChatGPT judges which garment colours
# suit it. Then one mockup turn per chosen garment colour, and (optionally) an
# adaptation turn per colour. {garment_colour} is the plain colour name.
COLORWAY_SUGGEST = """You are advising on garment colours for a DTF print.

The artwork's dominant colours are:

{colours}

Recommend the garment colours this artwork will look best on.

For each recommendation give the garment colour, and one sentence on why it

works - contrast against the artwork, how the design will read on that fabric,

and anything that will disappear or clash.

Also name any garment colour this artwork should NOT go on, and why.

Cover both light and dark garments. Give 4 to 6 recommendations.

Plain text, no markdown, no preamble."""

COLORWAY_MOCKUP = """Show this artwork printed on a {garment_colour} t-shirt.

Requirements:

- A plain {garment_colour} t-shirt, front view, flat or on a plain background

- The artwork printed on the chest at a realistic size and position

- Reproduce the artwork as closely as you can - same colours, same layout,

  same detail

- No model, no branding, no extra text

- Even lighting, no heavy shadows over the print"""

COLORWAY_ADAPT = """Adapt this artwork so it prints well on a {garment_colour}

garment.

Requirements:

- Keep the design recognisably the same - same layout, letterforms,

  illustration and composition

- Adjust only what is needed to read on {garment_colour} fabric: outlines,

  contrast, and any colour that would disappear against it

- Do not add new elements or remove existing ones

- Transparent background, PNG

- The design only, centred with even margins

- Clean sharp edges suitable for DTF garment printing"""


# ---------------------------------------------------------------------------
# Single source of truth #1: which template each workflow exposes to the UI.
#
# THE RECURRING BUG this prevents: the server's /api/templates endpoint and the
# browser's prompt editors each kept their own hardcoded list of "which prompt
# keys does this workflow use". Every new workflow re-introduced the drift — the
# endpoint would serve a key the UI never read, or the UI would read a key the
# endpoint never served, and the prompt box rendered empty.
#
# Now there is ONE map. `get_templates()` builds its JSON from it, and the UI
# reads the identical keys (it fetches WORKFLOW_TEMPLATE_KEYS via /api/templates
# so it literally cannot name a key the server does not send). Adding a workflow
# = adding one entry here; both sides pick it up automatically.
#
# Shape: {workflow_value: {json_key_the_ui_reads: registry_constant_name}}.
# `json_key` is the field name in the /api/templates response (e.g. "turn1").
# `registry_constant_name` is the CONSTANT in this module / prompt_registry
# (e.g. "TEXT_TURN_1"), resolved live from Prompt Management with the built-in
# text here as the fallback.
WORKFLOW_TEMPLATE_KEYS: dict[str, dict[str, str]] = {
    "text": {
        "turn1": "TEXT_TURN_1",
        "turn2": "TEXT_TURN_2",
        "turn3": "TEXT_TURN_3",
        "replace_collage": "TEXT_REPLACE_COLLAGE",
        "replace_final": "TEXT_REPLACE_FINAL",
        "image_element_collage": "TEXT_IMAGE_ELEMENT_COLLAGE",
        "image_style_collage": "TEXT_IMAGE_STYLE_COLLAGE",
    },
    "mockup": {
        "extract": "EXTRACT_CONTACT_SHEET",
        "regen": "EXTRACT_SINGLE",
    },
    "artwork": {
        # Historically served under both keys; the UI reads DT.artwork with a
        # DT.artwork_regen fallback, so both are kept.
        "artwork": "ARTWORK_REGENERATE",
        "artwork_regen": "ARTWORK_REGENERATE",
    },
    "identify": {
        "identify": "IDENTIFY_OBJECTS",
        "identify_colour": "IDENTIFY_COLOR_GROUPS",
        "identify_regen": "IDENTIFY_REGENERATE",
    },
    "printready": {
        "printready_review": "PRINTREADY_REVIEW",
    },
    "colorway": {
        "colorway_suggest": "COLORWAY_SUGGEST",
        "colorway_mockup": "COLORWAY_MOCKUP",
        "colorway_adapt": "COLORWAY_ADAPT",
    },
}


def template_keys_json() -> dict[str, str]:
    """Flat {json_key: registry_name} across every workflow — what the
    /api/templates endpoint serves. First declaration wins for a shared key
    (e.g. ARTWORK_REGENERATE under both "artwork" and "artwork_regen")."""
    flat: dict[str, str] = {}
    for keys in WORKFLOW_TEMPLATE_KEYS.values():
        for json_key, registry_name in keys.items():
            flat.setdefault(json_key, registry_name)
    return flat


# ---------------------------------------------------------------------------
# Single source of truth #2: what each workflow requires to be submitted.
#
# THE RECURRING BUG this prevents: create_job's per-workflow validation was a
# chain of `if workflow == X` blocks. A new workflow's block could read another
# workflow's field (e.g. checking `files`/`options` from the legacy path, or the
# button gating the wrong variable), so a valid submission was rejected or an
# invalid one slipped through.
#
# Now each workflow DECLARES its own required fields here, and one helper
# (validate_workflow_request) checks ONLY the fields named for that workflow. A
# workflow can never end up checking another's fields, because it never names
# them. Cross-field / placeholder rules that don't reduce to "field is present"
# stay as named extra-rule hooks, dispatched by the same helper.
#
# Each entry: {workflow_value: [(field_name, error_message), ...]}. `field_name`
# is an attribute on GenerateRequest. A string field passes when its .strip() is
# truthy; a list field passes when non-empty. `extra_rules` (below) handles
# anything more than presence.
WORKFLOW_REQUIRED_FIELDS: dict[str, list[tuple[str, str]]] = {
    "mockup": [
        ("mockup_image", "Upload a mockup image first."),
    ],
    "artwork": [
        ("artwork_files", "Upload at least one artwork file."),
    ],
    "identify": [
        ("artwork_files", "Upload an artwork file."),
    ],
    "printready": [
        ("artwork_files", "Upload an artwork file."),
    ],
    "colorway": [
        ("artwork_files", "Upload an artwork file."),
    ],
    "custom": [
        ("artwork_files", "Upload an artwork file."),
        ("custom_operations", "Select at least one operation."),
    ],
    # The legacy edit-options API job (no workflow value set).
    "": [
        ("files", "Select at least one image."),
        ("options", "Select at least one option."),
    ],
}


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

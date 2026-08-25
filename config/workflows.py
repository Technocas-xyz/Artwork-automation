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

TEXT_TURN_3 = """Generate colour variation number {m} as a single final artwork.

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

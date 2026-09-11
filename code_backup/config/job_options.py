"""Operation definitions for artwork edit jobs.

Each key maps to a prompt text block that will be appended when that
operation is selected.  Options containing {value} require a matching
entry in the params dict at build time.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

JOB_OPTIONS: dict[str, str] = {
    "text_only": (
        # TODO: Business to supply real wording.
        "[TEXT_ONLY operation prompt — placeholder]"
    ),
    "remove_background": (
        # TODO: Business to supply real wording.
        "[REMOVE_BACKGROUND operation prompt — placeholder]"
    ),
    "change_background": (
        # TODO: Business to supply real wording.  {value} = target background colour/description.
        "[CHANGE_BACKGROUND operation prompt — apply background: {value} — placeholder]"
    ),
    "blur_background": (
        # TODO: Business to supply real wording.
        "[BLUR_BACKGROUND operation prompt — placeholder]"
    ),
    "recolour": (
        # TODO: Business to supply real wording.  {value} = target colour.
        "[RECOLOUR operation prompt — recolour to: {value} — placeholder]"
    ),
    "upscale_cleanup": (
        # TODO: Business to supply real wording.
        "[UPSCALE_CLEANUP operation prompt — placeholder]"
    ),
}

# Options that require a {value} parameter at build time.
PARAMETERISED_OPTIONS: set[str] = {"change_background", "recolour"}

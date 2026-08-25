"""Prompt construction for artwork edit operations."""

from __future__ import annotations

from config.job_options import JOB_OPTIONS, PARAMETERISED_OPTIONS


# ---------------------------------------------------------------------------
# Base instruction block
# ---------------------------------------------------------------------------

# TODO: Business to supply final wording.
BASE_INSTRUCTION: str = (
    "[BASE INSTRUCTION — placeholder]\n"
    "Context: DTF print production.\n"
    "Output requirements: transparent PNG, high resolution, print-ready.\n"
    "[END BASE INSTRUCTION — placeholder]"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_prompt(
    options: list[str],
    params: dict[str, str] | None = None,
    custom_note: str | None = None,
) -> str:
    """Assemble a multi-line prompt from selected operations.

    Parameters
    ----------
    options:
        Ordered list of operation keys to include (must exist in JOB_OPTIONS).
    params:
        Values for parameterised options.  Keys should match option names
        that contain a {value} placeholder.
    custom_note:
        Freeform text appended at the very end of the prompt (optional).

    Returns
    -------
    str
        The fully assembled prompt string, ready for typing.

    Raises
    ------
    ValueError
        If an unknown option key is provided, or a parameterised option
        is missing its required value in `params`.
    """
    if params is None:
        params = {}

    _validate(options, params)

    sections: list[str] = [BASE_INSTRUCTION]

    for key in options:
        block: str = JOB_OPTIONS[key]
        if key in PARAMETERISED_OPTIONS:
            block = block.format(value=params[key])
        sections.append(block)

    if custom_note and custom_note.strip():
        sections.append(custom_note.strip())

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(options: list[str], params: dict[str, str]) -> None:
    """Check that all option keys are known and required params are present."""
    for key in options:
        if key not in JOB_OPTIONS:
            raise ValueError(
                f"Unknown operation key: {key!r}. "
                f"Valid keys: {sorted(JOB_OPTIONS.keys())}"
            )
        if key in PARAMETERISED_OPTIONS:
            if key not in params or not params[key].strip():
                raise ValueError(
                    f"Operation {key!r} requires a 'params' entry with a non-empty value."
                )

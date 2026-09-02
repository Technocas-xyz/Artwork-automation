"""ChatGPT image-generation orchestration via Playwright sync API.

Supports multi-turn conversations:
    open_chat(page)          — navigate and wait for readiness
    send_turn(page, ...)     — send a single turn, return only NEW images
    generate(page, ...)      — legacy single-turn wrapper (open_chat + send_turn)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout

from config.selectors import (
    COMPOSER_THUMBNAIL,
    CONVERSATION_TURN,
    FILE_INPUT,
    GENERATED_IMG,
    IMAGE_LOADER,
    PROJECT_URL,
    PROMPT_BOX,
    SEND_BUTTON,
    STOP_BUTTON,
)
from src.browser import GenerationTimeoutError, RateLimitError
from src.postprocess import is_opaque_white_bg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def open_chat(page: Page) -> None:
    """Navigate to PROJECT_URL and wait for the prompt box to be ready.

    Retries up to 3 times with a 3-second pause between attempts.
    Timeout raised to 90s per attempt. Only fails after all 3 exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            page.goto(PROJECT_URL, wait_until="domcontentloaded", timeout=90_000)
            page.wait_for_selector(PROMPT_BOX, state="visible", timeout=30_000)

            # Verify we actually landed on the project URL
            if "/g/g-p-" not in page.url:
                page.goto(PROJECT_URL, wait_until="domcontentloaded", timeout=90_000)
                page.wait_for_selector(PROMPT_BOX, state="visible", timeout=30_000)

            return  # Success
        except Exception as exc:
            last_error = exc
            logger.info("open_chat attempt %d/3 failed: %s", attempt, exc)
            if attempt < 3:
                page.wait_for_timeout(3000)

    raise GenerationTimeoutError(
        f"open_chat failed after 3 attempts. Last error: {last_error}"
    )


def send_text_turn(
    page: Page,
    prompt: str,
    image_paths: list[str] | None = None,
    run_id: str = "text_turn",
) -> str:
    """Send a turn expecting a TEXT reply (not images). Returns the reply text.

    Uses a short quiet period (1.5s) since text replies complete quickly.
    Does NOT wait on IMAGE_LOADER.
    """
    try:
        # Upload files if given
        if image_paths:
            page.set_input_files(FILE_INPUT, image_paths)
            _wait_for_upload_thumbnails(page, expected_count=len(image_paths))

        # Type prompt and submit
        page.click(PROMPT_BOX)
        _enter_prompt(page, prompt)
        page.keyboard.press("Enter")

        # Wait for STOP_BUTTON to appear (turn started)
        try:
            page.wait_for_selector(STOP_BUTTON, state="attached", timeout=30_000)
        except PlaywrightTimeout as exc:
            raise GenerationTimeoutError(
                "Stop button never appeared — text turn may not have started."
            ) from exc

        # Wait for turn to complete with SHORT quiet period (1.5s)
        _wait_for_turn_complete(page, quiet_ms=1500, timeout=120_000)

        # Return the text reply
        return get_last_text_reply(page)

    except (GenerationTimeoutError, RateLimitError):
        _save_error_screenshot(page, run_id)
        raise
    except Exception:
        _save_error_screenshot(page, run_id)
        raise


def send_turn(
    page: Page,
    prompt: str,
    image_paths: list[str] | None = None,
    run_id: str = "turn",
) -> list[bytes]:
    """Send a single conversation turn and return ONLY new images from this turn.

    Parameters
    ----------
    page:
        Active page with an open ChatGPT conversation.
    prompt:
        Text to type into the composer.
    image_paths:
        Optional list of file paths to attach. None for text-only follow-ups.
    run_id:
        Identifier for diagnostic screenshots on failure.

    Returns
    -------
    list[bytes]
        Downloaded image bytes for images generated in THIS turn only.
        Empty list if the turn produced no images.
    """
    try:
        # --- Upload images if provided ---
        if image_paths:
            page.set_input_files(FILE_INPUT, image_paths)
            _wait_for_upload_thumbnails(page, expected_count=len(image_paths))

        # --- Snapshot AFTER upload so uploaded file ids are already in the DOM ---
        pre_existing_ids = _collect_image_ids(page)
        logger.info("send_turn before-ids: %d ids captured", len(pre_existing_ids))

        # --- Type prompt ---
        page.click(PROMPT_BOX)
        _enter_prompt(page, prompt)

        # --- Submit ---
        page.keyboard.press("Enter")

        # --- Wait for STOP_BUTTON to APPEAR (confirms new turn started) ---
        try:
            page.wait_for_selector(STOP_BUTTON, state="attached", timeout=60_000)
        except PlaywrightTimeout as exc:
            raise GenerationTimeoutError(
                "Stop button never appeared — turn may not have started."
            ) from exc

        # --- Wait for the entire turn to finish ---
        _wait_for_turn_complete(page, quiet_ms=5000, timeout=900_000)

        # --- Collect new images from the LAST ASSISTANT TURN only ---
        new_srcs = _collect_last_turn_images(page, pre_existing_ids)
        logger.info("send_turn after-collection: %d new images", len(new_srcs))

        if not new_srcs:
            # Turn completed but produced no images (text-only response)
            return []

        # --- Download each image in-browser via fetch ---
        images: list[bytes] = []
        for src in new_srcs:
            byte_array: list[int] = page.evaluate(
                """async (url) => {
                    const resp = await fetch(url, { credentials: 'include' });
                    if (!resp.ok) throw new Error(`Fetch failed: ${resp.status}`);
                    const buf = await resp.arrayBuffer();
                    return Array.from(new Uint8Array(buf));
                }""",
                src,
            )
            images.append(bytes(byte_array))

        return _prefer_transparent(images)

    except (GenerationTimeoutError, RateLimitError):
        _save_error_screenshot(page, run_id)
        raise
    except Exception:
        _save_error_screenshot(page, run_id)
        raise


def generate(
    page: Page,
    image_paths: list[str],
    prompt: str,
    run_id: str,
) -> list[bytes]:
    """Single-turn generation — legacy wrapper around open_chat + send_turn.

    Signature unchanged from original implementation.
    """
    open_chat(page)
    return send_turn(page, prompt=prompt, image_paths=image_paths, run_id=run_id)


def get_last_text_reply(page: Page) -> str:
    """Read the text content of the last assistant conversation turn.

    Returns the inner text of the last conversation turn element,
    stripped of whitespace and ChatGPT UI chrome (timing labels etc.).
    """
    turns = page.query_selector_all(CONVERSATION_TURN)
    if not turns:
        return ""
    last_turn = turns[-1]
    text = last_turn.inner_text()
    if not text:
        return ""
    text = text.strip()
    # Strip ChatGPT UI metadata lines (e.g. "Worked for 23s", "Searching...", timing)
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip known UI chrome patterns
        if re.match(r"^Worked for \d+[smh]", stripped):
            continue
        if re.match(r"^Searching", stripped):
            continue
        if re.match(r"^Thinking", stripped):
            continue
        if re.match(r"^\d+:\d+$", stripped):  # timestamp like "2:34"
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def get_boxes(page: Page, image_path: str, width: int, height: int, run_id: str, prompt_template: str | None = None) -> list[dict]:
    """Ask ChatGPT to identify bounding boxes of designs on a mockup sheet.

    Opens a new chat, uploads the image, sends EXTRACT_BOXES prompt with the
    real dimensions substituted. Parses the JSON reply.

    Parameters
    ----------
    prompt_template : str | None
        Custom prompt template override. Must contain {width} and {height}.
        If None, uses EXTRACT_BOXES from config/workflows.

    Returns list of dicts with keys: n, x, y, w, h (all ints).
    Raises GenerationTimeoutError or ValueError on failure.
    """
    import json as _json
    from config.workflows import EXTRACT_BOXES

    template = prompt_template if prompt_template else EXTRACT_BOXES
    prompt = template.format(width=width, height=height)
    send_turn(page, prompt=prompt, image_paths=[image_path], run_id=run_id)

    reply = get_last_text_reply(page)
    if not reply:
        raise ValueError("ChatGPT returned an empty reply when asked for bounding boxes.")

    # Strip markdown fences if present
    text = reply.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Extract JSON substring: find first "[" and last "]"
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"Could not find a JSON array in ChatGPT's reply. Reply was:\n{reply[:500]}"
        )
    json_str = text[start:end + 1]

    # Parse JSON
    try:
        boxes = _json.loads(json_str)
    except _json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse JSON from ChatGPT's reply. Extracted:\n{json_str[:500]}\n\nFull reply:\n{reply[:500]}"
        ) from exc

    if not isinstance(boxes, list):
        raise ValueError(f"Expected a JSON array, got: {type(boxes).__name__}")

    # Validate and clamp each box
    validated: list[dict] = []
    for i, box in enumerate(boxes):
        try:
            n = int(box.get("n", i + 1))
            x = int(box["x"])
            y = int(box["y"])
            w = int(box["w"])
            h = int(box["h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Box {i+1} has invalid fields: {box}") from exc

        if w <= 0 or h <= 0:
            continue  # skip zero-size boxes

        # Clamp to image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y

        if w > 0 and h > 0:
            validated.append({"n": n, "x": x, "y": y, "w": w, "h": h})

    if not validated:
        raise ValueError("ChatGPT returned no valid bounding boxes.")

    return validated


def extract_artwork_images(
    page: Page,
    image_path: str,
    run_id: str,
    prompt_template: str | None = None,
) -> list[bytes]:
    """Extract all artwork designs from a mockup sheet as separate images.

    ChatGPT generates all designs sequentially in a SINGLE turn. This takes
    several minutes for many designs. Uses extended timeouts and logs progress.

    Parameters
    ----------
    page : Page
        Active page (chat should already be open via open_chat).
    image_path : str
        Path to the mockup sheet image file.
    prompt_template : str | None
        Custom prompt override. If None, uses EXTRACT_ARTWORKS from config.

    Returns list of PNG bytes, one per extracted design.
    """
    from config.workflows import EXTRACT_ARTWORKS

    prompt = prompt_template if prompt_template else EXTRACT_ARTWORKS

    try:
        # Upload first, THEN snapshot so uploaded file ids are excluded
        page.set_input_files(FILE_INPUT, [image_path])
        _wait_for_upload_thumbnails(page, expected_count=1)

        pre_existing_ids = _collect_image_ids(page)
        logger.info("extract_artwork_images before-ids: %d", len(pre_existing_ids))

        page.click(PROMPT_BOX)
        _enter_prompt(page, prompt)
        page.keyboard.press("Enter")

        # Wait for STOP_BUTTON to appear (confirms turn started)
        try:
            page.wait_for_selector(STOP_BUTTON, state="attached", timeout=60_000)
        except PlaywrightTimeout as exc:
            raise GenerationTimeoutError(
                "Stop button never appeared — extraction turn may not have started."
            ) from exc

        # Wait for turn to complete with extended quiet period (15s)
        _wait_for_turn_complete(page, quiet_ms=15000, timeout=1_800_000)

        # Collect from last assistant turn only
        new_srcs = _collect_last_turn_images(page, pre_existing_ids)
        logger.info("extract_artwork_images after-collection: %d new images", len(new_srcs))
        logger.info("extract_artwork_images: %d new images found", len(new_srcs))

        if not new_srcs:
            raise GenerationTimeoutError("Extraction completed but no images were generated.")

        # Download each image
        images: list[bytes] = []
        for i, src in enumerate(new_srcs, 1):
            logger.info("Downloading extracted image %d/%d", i, len(new_srcs))
            byte_array: list[int] = page.evaluate(
                """async (url) => {
                    const resp = await fetch(url, { credentials: 'include' });
                    if (!resp.ok) throw new Error(`Fetch failed: ${resp.status}`);
                    const buf = await resp.arrayBuffer();
                    return Array.from(new Uint8Array(buf));
                }""",
                src,
            )
            images.append(bytes(byte_array))

        return _prefer_transparent(images)

    except (GenerationTimeoutError, RateLimitError):
        _save_error_screenshot(page, run_id)
        raise
    except Exception:
        _save_error_screenshot(page, run_id)
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prefer_transparent(images: list[bytes]) -> list[bytes]:
    """Filter out opaque-white-background duplicates when transparent versions exist.

    If a turn returns both transparent and opaque versions of the same design,
    keep only the transparent ones. If all are the same type, return unchanged.
    Never returns an empty list.
    """
    if len(images) <= 1:
        return images

    opaque_flags = [is_opaque_white_bg(img) for img in images]
    has_opaque = any(opaque_flags)
    has_transparent = any(not f for f in opaque_flags)

    if has_opaque and has_transparent:
        # Mixed — keep only transparent
        kept = [img for img, is_opaque in zip(images, opaque_flags) if not is_opaque]
        dropped = len(images) - len(kept)
        logger.info(
            "Dropped %d opaque-background duplicate(s), keeping %d transparent image(s).",
            dropped,
            len(kept),
        )
        return kept if kept else images  # safety: never return empty

    return images


def _collect_image_ids(page: Page) -> set[str]:
    """Return the set of file ids for all GENERATED_IMG currently in the DOM."""
    ids: set[str] = set()
    elements = page.query_selector_all(GENERATED_IMG)
    for img in elements:
        src = img.get_attribute("src")
        if not src:
            continue
        match = re.search(r"id=(file_[A-Za-z0-9]+)", src)
        if match:
            ids.add(match.group(1))
    return ids


def _collect_new_image_srcs(page: Page, exclude_ids: set[str]) -> list[str]:
    """Collect unique image srcs from the page, excluding pre-existing ids."""
    elements = page.query_selector_all(GENERATED_IMG)
    seen_ids: set[str] = set()
    new_srcs: list[str] = []
    for img in elements:
        src = img.get_attribute("src")
        if not src:
            continue
        match = re.search(r"id=(file_[A-Za-z0-9]+)", src)
        if not match:
            continue
        file_id = match.group(1)
        if file_id in exclude_ids:
            continue
        if file_id not in seen_ids:
            seen_ids.add(file_id)
            new_srcs.append(src)
    return new_srcs


def _collect_last_turn_images(page: Page, exclude_ids: set[str]) -> list[str]:
    """Collect images from the LAST conversation turn only, excluding pre-existing ids.

    Scopes to the last CONVERSATION_TURN element so uploaded images from the
    user's turn are never picked up.
    """
    turns = page.query_selector_all(CONVERSATION_TURN)
    if not turns:
        # Fallback to page-wide search
        return _collect_new_image_srcs(page, exclude_ids)

    last_turn = turns[-1]
    elements = last_turn.query_selector_all(GENERATED_IMG)
    seen_ids: set[str] = set()
    new_srcs: list[str] = []
    for img in elements:
        src = img.get_attribute("src")
        if not src:
            continue
        match = re.search(r"id=(file_[A-Za-z0-9]+)", src)
        if not match:
            continue
        file_id = match.group(1)
        if file_id in exclude_ids:
            continue
        if file_id not in seen_ids:
            seen_ids.add(file_id)
            new_srcs.append(src)
    return new_srcs


def _wait_for_turn_complete(page: Page, quiet_ms: int = 5000, timeout: int = 900_000) -> None:
    """Wait until STOP_BUTTON is absent continuously for `quiet_ms` ms.

    Uses a Python polling loop because Playwright's raf-based wait_for_function
    stops firing when the page goes idle, making it unreliable for time-based
    stability checks.
    """
    deadline = time.monotonic() + timeout / 1000
    absent_since: float | None = None
    while time.monotonic() < deadline:
        present = page.query_selector(STOP_BUTTON) is not None
        if present:
            absent_since = None
        elif absent_since is None:
            absent_since = time.monotonic()
        elif time.monotonic() - absent_since >= quiet_ms / 1000:
            return
        page.wait_for_timeout(500)
    raise GenerationTimeoutError("Turn never completed.")


def _enter_prompt(page: Page, prompt: str) -> None:
    """Enter a prompt into the focused ProseMirror editor via clipboard paste.

    Falls back to character-by-character typing if paste fails verification.
    """
    # Try paste first (much faster for long prompts)
    try:
        page.evaluate("text => navigator.clipboard.writeText(text)", prompt)
        page.click(PROMPT_BOX)
        page.keyboard.press("Control+V")
        # Brief wait for paste to settle
        page.wait_for_timeout(300)
        # Verify paste succeeded by checking editor content length
        content = page.evaluate(
            "() => document.querySelector('#prompt-textarea')?.textContent || ''"
        )
        if len(content.strip()) >= len(prompt.strip()) * 0.8:
            logger.info("_enter_prompt: paste succeeded (%d chars)", len(content))
            return
        else:
            logger.info("_enter_prompt: paste produced %d chars, expected ~%d. Falling back to typing.", len(content), len(prompt))
    except Exception as exc:
        logger.info("_enter_prompt: paste failed (%s), falling back to typing.", exc)

    # Fallback: clear whatever partial paste left and type character by character
    page.click(PROMPT_BOX)
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    _type_multiline(page, prompt)


def _type_multiline(page: Page, text: str) -> None:
    """Type multi-line text into the focused ProseMirror editor (fallback for paste).

    Splits on newlines and inserts Shift+Enter between lines so that
    page.keyboard.type() does not accidentally submit the form.
    """
    lines: list[str] = text.split("\n")
    for i, line in enumerate(lines):
        page.keyboard.type(line, delay=12)
        if i < len(lines) - 1:
            page.keyboard.press("Shift+Enter")


def _wait_for_upload_thumbnails(page: Page, expected_count: int) -> None:
    """Poll until the composer shows exactly `expected_count` upload thumbnails.

    Uses wait_for_function so we never sleep on a fixed timer.  Times out after 30 s.
    Also waits for SEND_BUTTON to appear — it only becomes visible once the
    composer has content, confirming the upload is fully attached.
    """
    page.wait_for_selector(COMPOSER_THUMBNAIL, state="visible", timeout=30_000)

    page.wait_for_function(
        """([selector, count]) => document.querySelectorAll(selector).length >= count""",
        arg=[COMPOSER_THUMBNAIL, expected_count],
        timeout=30_000,
    )

    page.wait_for_selector(SEND_BUTTON, state="visible", timeout=10_000)


def _save_error_screenshot(page: Page, run_id: str) -> None:
    """Persist a debug screenshot under logs/."""
    try:
        path = Path("logs") / f"{run_id}_error.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path))
    except Exception:
        pass

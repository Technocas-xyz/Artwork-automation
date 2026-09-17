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
    CHAT_TITLE_INPUT,
    COMPOSER_THUMBNAIL,
    CONVERSATION_TURN,
    FILE_INPUT,
    GENERATED_IMG,
    IMAGE_LOADER,
    PROJECT_URLS,
    PROJECT_URL_DEFAULT,
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


def project_url_for(workflow: str | None) -> str:
    """Resolve the ChatGPT project URL for a workflow.

    Falls back to PROJECT_URL_DEFAULT (a plain chat) when the workflow key is
    unknown or its configured URL is empty."""
    url = (PROJECT_URLS.get(workflow or "") or "").strip()
    return url or PROJECT_URL_DEFAULT


def _project_id(url: str) -> str | None:
    """Extract the project id (the `g-p-<id>` portion) from a project URL.

    Verification matches on this id only, NOT the whole URL, because ChatGPT may
    later append a name slug (e.g. .../g-p-<id>-dtf-artwork/project). The id is
    the alphanumeric run immediately after `g-p-`, stopping before any `-slug`,
    `/`, or `?`, so the check keeps working when a slug appears."""
    m = re.search(r"g-p-([0-9a-zA-Z]+)", url)
    return m.group(1) if m else None


def _try_open_url(page: Page, url: str) -> None:
    """Navigate to `url` and wait for the prompt box, retrying up to 3 times.

    When `url` is a project URL, verifies we actually landed on THAT project by
    matching the project id (tolerant of a name slug appearing later). A plain
    chatgpt.com target has no id to check. Raises GenerationTimeoutError after
    all attempts fail."""
    project_id = _project_id(url)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=90_000)
            page.wait_for_selector(PROMPT_BOX, state="visible", timeout=30_000)

            # Verify we landed on the CHOSEN project by its id. Match on the id
            # only, so a slug ChatGPT adds later (…/g-p-<id>-dtf-…/project) still
            # counts as the right project.
            if project_id and project_id not in page.url:
                page.goto(url, wait_until="domcontentloaded", timeout=90_000)
                page.wait_for_selector(PROMPT_BOX, state="visible", timeout=30_000)

            return  # Success
        except Exception as exc:
            last_error = exc
            logger.info("open_chat attempt %d/3 for %s failed: %s", attempt, url, exc)
            if attempt < 3:
                page.wait_for_timeout(3000)

    raise GenerationTimeoutError(
        f"open_chat failed after 3 attempts for {url}. Last error: {last_error}"
    )


def open_chat(page: Page, workflow: str | None = None) -> dict:
    """Navigate to the workflow's ChatGPT project and wait for readiness.

    The URL is chosen per workflow from PROJECT_URLS, falling back to
    PROJECT_URL_DEFAULT when the key is missing or its URL is empty. Keeps the
    existing retry + URL verification, verifying against the CHOSEN url.

    If a project URL is configured but fails to load after the retries, this
    falls back to plain chatgpt.com rather than failing the whole run — a broken
    project link must not stop production. The return dict tells the caller what
    happened so it can record a warning:

        {"url": <url actually opened>, "requested_url": <workflow url>,
         "fell_back": bool, "warning": <str|None>}
    """
    requested = project_url_for(workflow)
    logger.info("open_chat: workflow=%r -> %s", workflow, requested)
    try:
        _try_open_url(page, requested)
        return {"url": requested, "requested_url": requested,
                "fell_back": False, "warning": None}
    except Exception as exc:
        # Only the DEFAULT is left to try. If we were already on the default,
        # there is nothing to fall back to — re-raise so the job fails clearly.
        if requested == PROJECT_URL_DEFAULT:
            raise
        warning = (f"Project for workflow '{workflow}' ({requested}) failed to "
                   f"load ({exc}); fell back to {PROJECT_URL_DEFAULT}.")
        logger.info(warning)
        _try_open_url(page, PROJECT_URL_DEFAULT)
        return {"url": PROJECT_URL_DEFAULT, "requested_url": requested,
                "fell_back": True, "warning": warning}


def rename_chat(page: Page, title: str) -> bool:
    """Best-effort: rename the current chat to `title` so it is findable later.

    Cosmetic only — if ChatGPT's rename control is not reachable (markup changed,
    element missing), this returns False silently and never raises, so a job is
    never failed for a naming step."""
    if not title:
        return False
    try:
        el = page.query_selector(CHAT_TITLE_INPUT)
        if not el:
            logger.info("rename_chat: title control not found; skipping rename to %r", title)
            return False
        el.click()
        el.fill(title)
        page.keyboard.press("Enter")
        logger.info("rename_chat: renamed chat to %r", title)
        return True
    except Exception as exc:
        logger.info("rename_chat: skipped (%s)", exc)
        return False


def send_text_turn(
    page: Page,
    prompt: str,
    image_paths: list[str] | None = None,
    run_id: str = "text_turn",
) -> str:
    """Send a turn expecting a TEXT reply (not images). Returns the reply text.

    Uses a short quiet period since text replies complete quickly, and NEVER
    waits on any image-generation signal (IMAGE_LOADER). Every wait is
    timestamped in the log so a slow text turn can be diagnosed at a glance:

        [text-turn <run_id>] submit sent            t=+0.0s
        [text-turn <run_id>] stop button appeared   t=+1.2s   (or 'never appeared')
        [text-turn <run_id>] stop button detached   t=+3.4s
        [text-turn <run_id>] quiet period elapsed   t=+4.2s
        [text-turn <run_id>] reply read (<n> chars) t=+4.3s

    A short reply should finish well under 15s; the timestamps show which wait,
    if any, is spending the time.
    """
    t0 = time.monotonic()

    def _ts(msg: str) -> None:
        logger.info("[text-turn %s] %s  t=+%.1fs", run_id, msg, time.monotonic() - t0)

    try:
        # Upload files if given
        if image_paths:
            page.set_input_files(FILE_INPUT, image_paths)
            _wait_for_upload_thumbnails(page, image_paths, run_id=run_id)
            _ts("upload attached")

        # Type prompt and submit
        page.click(PROMPT_BOX)
        _enter_prompt(page, prompt)
        _ts("prompt entered")

        page.keyboard.press("Enter")
        _ts("submit sent")

        # Wait for STOP_BUTTON to appear (turn started). A fast text reply can
        # come and go before we look, so a miss here is NOT fatal — we log it
        # and fall through to the completion wait, which will read the reply.
        try:
            page.wait_for_selector(STOP_BUTTON, state="attached", timeout=15_000)
            _ts("stop button appeared")
        except PlaywrightTimeout:
            _ts("stop button never appeared (reply may already be finishing)")

        # Wait for the turn to settle. Short quiet period: a text answer streams
        # in a couple of seconds and then the stop button detaches. 1.5s of
        # continuous absence is enough to be sure the stream ended.
        _wait_for_turn_complete(page, quiet_ms=1500, timeout=120_000, label=f"text-turn {run_id}", on_phase=_ts, text_only=True)

        # Return the text reply
        reply = get_last_text_reply(page)
        _ts(f"reply read ({len(reply)} chars)")
        # Raw reply logged so an empty/garbled result is visible at the agent
        # (truncated to keep the log readable).
        preview = reply.replace("\n", " ")[:300]
        logger.info("[text-turn %s] RAW REPLY (%d chars): %r%s",
                    run_id, len(reply), preview, " …" if len(reply) > 300 else "")
        return reply

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
            _wait_for_upload_thumbnails(page, image_paths, run_id=run_id)

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
    workflow: str | None = None,
) -> list[bytes]:
    """Single-turn generation — legacy wrapper around open_chat + send_turn.

    `workflow` is optional so existing callers keep working; when given it
    routes to that workflow's ChatGPT project.
    """
    open_chat(page, workflow)
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
        _wait_for_upload_thumbnails(page, [image_path], run_id=run_id)

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


def _wait_for_turn_complete(page: Page, quiet_ms: int = 5000, timeout: int = 900_000,
                            label: str = "", on_phase=None, text_only: bool = False) -> None:
    """Wait until the turn has settled — the STOP button is gone and stays gone.

    Uses a Python polling loop because Playwright's raf-based wait_for_function
    stops firing when the page goes idle, making it unreliable for time-based
    stability checks.

    FLICKER GUARD (the text-turn 120s hang): after a text answer finishes, the
    STOP button detaches, but ChatGPT briefly re-renders a button while it swaps
    in the "Worked for Ns" chrome / send button. The old logic reset the quiet
    window to zero on ANY single poll that saw a button, so the 1.5s of
    continuous absence never accumulated and the loop spun until the 120s
    timeout. We now require the button to be present for TWO consecutive polls
    (~1s) to count as "still streaming"; a lone flicker poll no longer resets an
    established quiet window.

    `text_only=True` (text replies) never inspects any image-generation signal —
    completion is purely "STOP button gone". Image turns keep their existing
    behaviour: this function only ever waited on the STOP button, so image turns
    are unchanged aside from the same harmless flicker tolerance.

    `on_phase(msg)` (optional) is called once when the stop button first goes
    absent ("stop button detached") and once when the quiet period elapses
    ("quiet period elapsed"), so callers can timestamp the turn precisely.
    """
    def _phase(msg: str) -> None:
        if on_phase:
            try:
                on_phase(msg)
            except Exception:
                pass

    deadline = time.monotonic() + timeout / 1000
    quiet_s = quiet_ms / 1000
    absent_since: float | None = None
    detached_logged = False
    present_streak = 0     # consecutive polls the button was seen present
    poll = 0
    while time.monotonic() < deadline:
        raw_present = page.query_selector(STOP_BUTTON) is not None
        poll += 1
        now = time.monotonic()

        if raw_present:
            present_streak += 1
        else:
            present_streak = 0

        # A single flicker (one poll) does NOT count as streaming once the turn
        # has already detached — only a sustained presence (>=2 polls) does.
        streaming = raw_present and (absent_since is None or present_streak >= 2)

        if streaming:
            absent_since = None
        else:
            if absent_since is None:
                absent_since = now
                if not detached_logged:
                    _phase("stop button detached")
                    detached_logged = True

        absent_for = (now - absent_since) if absent_since is not None else 0.0
        logger.debug("[turn-wait %s] poll=%d stop=%s streak=%d absent_for=%.2fs "
                     "need>=%.2fs waiting_for=%s%s",
                     label, poll, "present" if raw_present else "absent",
                     present_streak, absent_for, quiet_s,
                     "streaming" if streaming else "quiet-period",
                     " (flicker-ignored)" if (raw_present and not streaming) else "")

        if absent_since is not None and absent_for >= quiet_s:
            _phase("quiet period elapsed")
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


def _wait_for_upload_thumbnails(page: Page, image_paths, run_id: str = "upload") -> None:
    """Confirm attached files reached the composer, tolerating a missing blob
    thumbnail so a rendering quirk never kills a job.

    ChatGPT usually shows an ``img[src^='blob:']`` thumbnail per attached file,
    but sometimes attaches the file without rendering that blob (or renders it
    differently). So we treat the upload as attached if EITHER:
      * at least `expected_count` blob thumbnails are present, OR
      * the send button has become enabled (it only enables once the composer
        has content).

    If neither appears within the timeout, we re-attach the files once and wait
    again. Only if that also fails do we raise — with a screenshot and a plain
    message naming the file(s) — so an operator can see whether the file did
    attach and the thumbnail merely looked different.

    `image_paths` is needed for the single retry; `run_id` names the screenshot.
    """
    expected_count = len(image_paths)

    def _attached() -> bool:
        try:
            n_blobs = len(page.query_selector_all(COMPOSER_THUMBNAIL))
        except Exception:
            n_blobs = 0
        if n_blobs >= expected_count:
            return True
        # Send button enabled is independent evidence the composer has content.
        return _send_button_ready(page)

    # Attempt 1: give a busy machine / large file up to 60s.
    if _wait_until(_attached, timeout_ms=60_000):
        _log_upload_state(page, expected_count, "attached")
        return

    # Retry the attach once — ChatGPT occasionally drops the first set_input_files.
    logger.info("upload not confirmed after 60s; re-attaching files once")
    try:
        page.set_input_files(FILE_INPUT, list(image_paths))
    except Exception as exc:
        logger.info("re-attach set_input_files failed: %s", exc)

    if _wait_until(_attached, timeout_ms=60_000):
        _log_upload_state(page, expected_count, "attached after retry")
        return

    # Give up — screenshot + a plain message naming the file(s) that failed.
    _log_upload_state(page, expected_count, "FAILED")
    _save_error_screenshot(page, run_id)
    names = ", ".join(Path(p).name for p in image_paths) or "(no files)"
    raise GenerationTimeoutError(
        f"Upload did not attach in the composer: {names}. "
        f"No blob thumbnail appeared and the send button never enabled."
    )


def _send_button_ready(page: Page) -> bool:
    """True if the send button exists and is enabled (composer has content)."""
    try:
        btn = page.query_selector(SEND_BUTTON)
        if not btn:
            return False
        # Visible + not disabled. ChatGPT disables it while the composer is empty.
        if not btn.is_visible():
            return False
        disabled = btn.get_attribute("disabled")
        aria = btn.get_attribute("aria-disabled")
        return disabled is None and aria not in ("true", "")
    except Exception:
        return False


def _wait_until(predicate, timeout_ms: int, poll_ms: int = 500) -> bool:
    """Poll `predicate` until true or the timeout elapses. Never raises."""
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(poll_ms / 1000.0)
    try:
        return bool(predicate())
    except Exception:
        return False


def _log_upload_state(page: Page, expected_count: int, phase: str) -> None:
    """Log what the upload check saw: blob count, send-button presence/enabled."""
    try:
        n_blobs = len(page.query_selector_all(COMPOSER_THUMBNAIL))
    except Exception:
        n_blobs = -1
    present = enabled = False
    try:
        btn = page.query_selector(SEND_BUTTON)
        present = btn is not None
        enabled = _send_button_ready(page)
    except Exception:
        pass
    logger.info(
        "upload check [%s]: blob_thumbnails=%d (expected %d) send_button_present=%s send_button_enabled=%s",
        phase, n_blobs, expected_count, present, enabled,
    )


def _save_error_screenshot(page: Page, run_id: str) -> None:
    """Persist a debug screenshot under logs/."""
    try:
        path = Path("logs") / f"{run_id}_error.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path))
    except Exception:
        pass

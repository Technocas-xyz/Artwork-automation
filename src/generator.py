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
    COMPOSER_REMOVE_ATTACHMENT,
    COMPOSER_THUMBNAIL,
    ASSISTANT_MESSAGE,
    CONVERSATION_TURN,
    COPY_BUTTON,
    DIL_RENDERER,
    FILE_INPUT,
    GENERATED_IMG_HOSTS,
    GENERATED_IMG,
    IMAGE_LOADER,
    PROJECT_URLS,
    PROJECT_URL_DEFAULT,
    PROMPT_BOX,
    SEND_BUTTON,
    SEND_BUTTON_LABELS,
    STOP_BUTTON,
    STOP_BUTTON_LABELS,
    TURN_ACTION_CONTROLS,
)
from src.browser import GenerationTimeoutError, RateLimitError
from src.postprocess import is_opaque_white_bg

logger = logging.getLogger(__name__)

# Set once we have dumped the page controls on a control-not-found failure, so
# the diagnostic fires the FIRST time a submit/selector fails in a process (when
# it is most informative) without spamming the log on every later turn.
_CONTROLS_DUMPED = False


def _dump_controls_once(page, reason: str) -> None:
    """Dump every button's aria-label + the last turn structure to the log, but
    only the first time per process. Wraps dump_page_controls (defined below)."""
    global _CONTROLS_DUMPED
    if _CONTROLS_DUMPED:
        return
    _CONTROLS_DUMPED = True
    try:
        dump_page_controls(page, reason=reason)
    except Exception:
        pass

# Diagnostics: record retries, fallbacks, timeouts and unconfirmed uploads so a
# stuck job is traceable. Import defensively — a diagnostics problem must never
# stop generation, and generator.py must stay importable if the module is absent.
try:
    from src import agent_diagnostics as _diag
except Exception:  # pragma: no cover
    _diag = None


def _diag_event(fn: str, *args, **kw) -> None:
    """Call a diag.<fn> convenience wrapper if diagnostics is available."""
    if _diag is None:
        return
    try:
        getattr(_diag, fn)(*args, **kw)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Workflow keys that open a ChatGPT chat and therefore NEED a project URL. The
# UI's "textimage" tab submits as "text", so only the underlying workflow values
# appear here. The legacy edit-options job (workflow "") also runs through the
# agent. Keep this in step with the workflows create_job can produce.
CHAT_WORKFLOWS = ("text", "mockup", "artwork", "custom", "identify")


def _has_configured_project(workflow: str | None) -> bool:
    """True only when PROJECT_URLS has a NON-EMPTY entry for this exact key.

    Distinct from project_url_for(): a missing/empty entry is NOT configured,
    even though project_url_for() would still return the default so a bare chat
    can be opened."""
    return bool((PROJECT_URLS.get(workflow or "") or "").strip())


def check_project_urls() -> list[str]:
    """Startup check: every chat-opening workflow should have a project URL.

    Logs a clear WARNING naming any workflow that will fall back to plain
    chatgpt.com (where a project's composer/behaviour is absent). Returns the
    list of unconfigured workflow keys so a caller can surface them too. Called
    once at agent startup — the identify workflow silently fell back for a while
    because nothing flagged the missing entry."""
    missing = [wf for wf in CHAT_WORKFLOWS if not _has_configured_project(wf)]
    if missing:
        logger.warning(
            "PROJECT_URLS has no entry for workflow(s): %s. "
            "Jobs for these will FAIL with 'No ChatGPT project configured' "
            "(plain chatgpt.com has no usable project composer). "
            "Add each to config/selectors.py PROJECT_URLS.",
            ", ".join(missing),
        )
    else:
        logger.info("PROJECT_URLS: all chat workflows configured (%s).",
                    ", ".join(CHAT_WORKFLOWS))
    return missing


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

    # The composer never became visible after 3 tries — the single most likely
    # cause is that ChatGPT moved the composer markup (as with the #prompt-textarea
    # id removal). Dump the live controls so the new attributes are in the log.
    dump_page_controls(page, reason=f"composer-not-visible open_chat {url}")
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
    configured = _has_configured_project(workflow)
    requested = project_url_for(workflow)
    logger.info("open_chat: workflow=%r -> %s (configured=%s)", workflow, requested, configured)

    # No project configured for this workflow — it will land on plain
    # chatgpt.com, where the project composer never appears and the run would
    # otherwise die on repeated #prompt-textarea timeouts. Fail LOUDLY and
    # early with a message that names the cause, instead of three timeouts.
    if not configured:
        raise GenerationTimeoutError(
            f"No ChatGPT project configured for workflow '{workflow}'. "
            f"Add its URL to config/selectors.py PROJECT_URLS."
        )

    try:
        _try_open_url(page, requested)
        return {"url": requested, "requested_url": requested,
                "fell_back": False, "warning": None}
    except Exception as exc:
        # The configured project failed to LOAD (a broken/expired link). That is
        # different from having no project at all: a transient load failure must
        # not stop production, so fall back to plain chatgpt.com and record a
        # warning. (An unconfigured workflow already failed loudly above.)
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
            _clear_composer_attachments(page, run_id=run_id)
            page.set_input_files(FILE_INPUT, image_paths)
            _wait_for_upload_thumbnails(page, image_paths, run_id=run_id)
            _ts("upload attached")

        # Type prompt and submit
        page.click(PROMPT_BOX)
        _enter_prompt(page, prompt)
        _ts("prompt entered")

        # Count reply containers BEFORE submit so the reader can tell a NEW reply
        # from the previous one (accept only when the count grows).
        prior_dil = _count_dil_renderers(page)

        submitted = _submit_prompt(page, run_id=run_id)
        _ts("submit confirmed" if submitted else "submit NOT confirmed")

        # Wait for STOP_BUTTON to appear (turn started). A fast text reply can
        # come and go before we look, so a miss here is NOT fatal — we log it
        # and fall through to the completion wait, which will read the reply.
        # (If _submit_prompt already confirmed via the composer clearing, the
        # stop button may have come and gone; that is fine.)
        try:
            page.wait_for_selector(STOP_BUTTON, state="attached", timeout=15_000)
            _ts("stop button appeared")
        except PlaywrightTimeout:
            _ts("stop button never appeared (reply may already be finishing)")
            _diag_event("timeout", "Text turn: stop button never appeared within 15s",
                        step=run_id, detail="continuing to completion wait")
            # A missing stop button often means the markup moved — dump controls
            # so the new attribute values are readable from the log.
            _dump_controls_once(page, reason=f"stop-button-missing text-turn {run_id}")

        # Wait for the turn to settle. Short quiet period: a text answer streams
        # in a couple of seconds and then the stop button detaches. 1.5s of
        # continuous absence is enough to be sure the stream ended.
        _wait_for_turn_complete(page, quiet_ms=1500, timeout=120_000, label=f"text-turn {run_id}", on_phase=_ts, text_only=True)

        # Return the text reply. Pass the sent prompt so the reader can reject a
        # read that came back equal to it (i.e. the user turn, not the reply), and
        # the prior DilRenderer count so it only accepts a NEW reply container.
        reply = get_last_text_reply(page, sent_prompt=prompt, prior_dil_count=prior_dil)
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
    require_images: bool = False,
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
    require_images:
        When True, a turn that finishes with 0 new images raises
        GenerationTimeoutError("ChatGPT returned no images") instead of
        returning []. Use it for image-generation steps where an empty result
        means failure — so the workflow fails with a clear error rather than
        pausing the operator on an empty selection. Defaults to False so callers
        that legitimately tolerate an empty turn keep their behaviour.

    Returns
    -------
    list[bytes]
        Downloaded image bytes for images generated in THIS turn only.
        Empty list if the turn produced no images (unless require_images).
    """
    try:
        # --- Upload images if provided ---
        if image_paths:
            # Clear any leftover attachment from a prior turn first, so the new
            # upload does not end up with a stale extra file in the composer.
            _clear_composer_attachments(page, run_id=run_id)
            page.set_input_files(FILE_INPUT, image_paths)
            _wait_for_upload_thumbnails(page, image_paths, run_id=run_id)

        # --- Snapshot AFTER upload so uploaded file ids are already in the DOM ---
        pre_existing_ids = _collect_image_ids(page)
        logger.info("send_turn before-ids: %d ids captured", len(pre_existing_ids))

        # --- Type prompt ---
        page.click(PROMPT_BOX)
        _enter_prompt(page, prompt)

        # --- Submit (Enter first, verify, Send-button fallback) ---
        t0 = time.monotonic()
        submitted = _submit_prompt(page, run_id=run_id)

        def _ts(msg: str) -> None:
            logger.info("[image-turn %s] %s  t=+%.1fs", run_id, msg, time.monotonic() - t0)

        # --- Confirm the new turn started (stop control appears). ---
        try:
            page.wait_for_selector(STOP_BUTTON, state="attached", timeout=60_000)
            _ts("stop button appeared")
        except PlaywrightTimeout as exc:
            if not submitted:
                raise GenerationTimeoutError(
                    "Prompt was never submitted — neither Enter nor the Send button "
                    "confirmed. See the control dump in the log."
                ) from exc
            # Submitted but no stop control — log and continue; the image wait
            # below will still poll for the result (a fast text part can detach
            # the stop button before we look).
            _ts("stop button never appeared (continuing to image wait)")
            _dump_controls_once(page, reason=f"stop-button-missing turn {run_id}")

        # --- Wait for the streaming part to settle (stop button detaches). This
        #     only marks the END OF TEXT; the image renders LATER. ---
        _wait_for_turn_complete(page, quiet_ms=5000, timeout=900_000,
                                label=f"image-turn {run_id}", on_phase=_ts)
        _ts("stop button detached / quiet period elapsed")

        # --- Now WAIT for the actual generated image. It appears 30–120s after
        #     the stop button detaches, so this poll is what really gates
        #     collection (the old code collected here and always found 0). ---
        new_srcs = _wait_for_new_image(page, pre_existing_ids, run_id, timeout_ms=180_000)
        logger.info("send_turn after-collection: %d new images", len(new_srcs))

        if not new_srcs:
            # No image after the full wait. Dump the image diagnostic AGAIN, now
            # including the last DilRenderer's outerHTML, since the generated
            # image likely lives inside the new DilRenderer structure.
            _diagnose_images(page, reason="0 new images after image-wait")
            _dump_last_dil_html(page, reason=f"0-images turn {run_id}")
            if require_images:
                raise GenerationTimeoutError("ChatGPT returned no images")
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


# UI chrome that ChatGPT wraps around a reply in the innerText of the WHOLE
# conversation area (said-heading / dom-parent paths). This must never be applied
# to the DilRenderer reply body — that element contains only the reply.
#
# CASE-SENSITIVE and EXACT: these are the literal labels ChatGPT renders. A real
# design whose text is "PRO", "AUTO", "SHARE", "COPY" (uppercase) must survive,
# so we match only the exact-case UI strings, and only strip them from the
# START or END of the text — never from the middle, where a matching line would
# be part of the actual content.
_UI_CHROME_EDGE_EXACT = (
    "Latest response",
    "Instant", "Thinking", "Auto", "Pro",
    "Copy code", "Copy", "Share", "Regenerate", "Read aloud",
    "You said:", "ChatGPT said:", "Assistant said:",
    "ChatGPT can make mistakes. Check important info.",
    "ChatGPT can make mistakes. Check important info. See Cookie Preferences.",
)
# Edge lines that match a case-sensitive pattern (timing/status chrome). Also
# only trimmed from the edges.
_UI_CHROME_EDGE_PATTERNS = (
    r"^Worked for \d+[smh]",
    r"^Searching\b",
    r"^Thinking\b",
    r"^Reasoned\b",
    r"^\d+:\d+$",                                     # timestamp like "2:34"
    r"^ChatGPT can make mistakes\.",                 # footer disclaimer variants
)


def _is_edge_chrome(line: str) -> bool:
    """True if a trimmed line is a known UI-chrome label/pattern (case-sensitive)."""
    s = line.strip()
    if s in _UI_CHROME_EDGE_EXACT:
        return True
    return any(re.match(p, s) for p in _UI_CHROME_EDGE_PATTERNS)


def _strip_ui_chrome(text: str) -> str:
    """Strip ChatGPT UI chrome from the WRAPPING of a reply (used for the
    said-heading and dom-parent read paths, NOT for dil-renderer).

    Case-sensitive, exact, and EDGE-ONLY: chrome lines are removed only from the
    start and end of the text, never from the middle. So a real design reading
    "PRO", "AUTO", "SHARE" or "COPY" (uppercase) is preserved, and a chrome label
    that only appears as a header/footer is stripped. Blank edge lines are also
    trimmed."""
    if not text:
        return text
    lines = text.split("\n")
    # Trim chrome/blank lines from the FRONT.
    while lines and (not lines[0].strip() or _is_edge_chrome(lines[0])):
        lines.pop(0)
    # Trim chrome/blank lines from the BACK.
    while lines and (not lines[-1].strip() or _is_edge_chrome(lines[-1])):
        lines.pop()
    return "\n".join(lines).strip()


# Matches the MDX/directive form  ::name{...}  that ChatGPT now emits as
# citation/reference markers in inner_text().  Pattern is deliberately narrow:
#   ::       — two literal colons (not a URL scheme, not "https://")
#   \w[\w-]* — an identifier (word chars + hyphens; no spaces, slashes, etc.)
#   \s*      — optional whitespace before the brace block
#   \{[^}]*\} — a single brace-enclosed block (attributes); no nested braces.
# This matches  ::chatgpt-content-reference{index="0" source_message_id="..."}
# and any other  ::directive-name{...}  variant.  It will NOT match "::X" without
# a brace block, double-colons in ordinary text, or URL schemes.
_RE_DIRECTIVE = re.compile(r"::\w[\w-]*\s*\{[^}]*\}")


def _strip_chat_directives(text: str) -> str:
    """Remove ChatGPT internal directive markers from reply text.

    These appear in inner_text() / clipboard reads as  ::name{attrs}  blocks —
    e.g. ::chatgpt-content-reference{index="0" source_message_id="..."}.
    They are NOT part of the actual reply; they are serialisation artefacts of
    the internal citation/footnote markup.

    Strategy (conservative):
      1. Remove every ::name{...} occurrence inline.
      2. Drop any line that, after stripping those, is blank or whitespace-only.
      3. Collapse sequences of blank lines to a single blank so the text reads
         cleanly without double-spacing.

    Returns the text unchanged if no directives are found."""
    if "::" not in text:
        return text  # fast path — avoids regex work on the typical clean reply
    cleaned = _RE_DIRECTIVE.sub("", text)
    if cleaned == text:
        return text  # nothing matched
    # Reconstruct line by line. A line that was non-blank in the original but
    # became blank after removing directives (i.e. it contained ONLY a directive)
    # is dropped entirely.  Then collapse any run of consecutive blank lines
    # (whether originally present or produced by drops) to at most one blank,
    # so the result reads cleanly without double-spacing.
    orig_lines = text.split("\n")
    new_lines  = cleaned.split("\n")
    # Pass 1: drop lines that were non-blank but became blank (directive-only lines).
    kept: list[str] = []
    for orig, new in zip(orig_lines, new_lines):
        was_blank = not orig.strip()
        is_blank  = not new.strip()
        if is_blank and not was_blank:
            continue  # this line was purely a directive — remove it
        kept.append(new)
    # Pass 2: collapse any run of consecutive blank lines down to one.
    out: list[str] = []
    prev_blank = False
    for line in kept:
        is_blank = not line.strip()
        if is_blank:
            if not prev_blank:
                out.append(line)
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False
    return "\n".join(out).strip()


def _unwrap_reply(text: str) -> str:
    """Strip wrapping quotes / markdown fences that ChatGPT sometimes adds around
    a short answer, so an image reading "ACT" is accepted whether it comes back
    as ACT, "ACT", 'ACT', `ACT`, or ```ACT```.

    CONSERVATIVE: only strips a matching pair that wraps the WHOLE text (after
    trimming). Never touches quotes/backticks that are part of the content
    (e.g. a design that legitimately contains a quote), and never removes inner
    characters. Returns the text unchanged if nothing clearly wraps it."""
    t = text.strip()
    if not t:
        return t
    # 1. A fenced code block wrapping the whole reply: ```lang\n...\n``` or `...`.
    if t.startswith("```") and t.endswith("```") and len(t) >= 6:
        inner = t[3:-3]
        # Drop an optional leading language tag on the first line.
        inner = re.sub(r"^[A-Za-z0-9_-]*\n", "", inner, count=1)
        t = inner.strip()
    elif t.startswith("`") and t.endswith("`") and len(t) >= 2 and "`" not in t[1:-1]:
        t = t[1:-1].strip()
    # 2. Matching surrounding quotes around the WHOLE (now single-segment) text.
    #    Only when the quote char does not also appear inside — so we never
    #    swallow content like  He said "hi"  or an apostrophe in the middle.
    for q in ('"', "'", "“", "”", "‘", "’"):
        if len(t) >= 2 and t[0] == q and t[-1] == q and q not in t[1:-1]:
            t = t[1:-1].strip()
            break
    # Curly-quote pair "…" / '…' (open+close differ).
    if len(t) >= 2 and t[0] in "“‘" and t[-1] in "”’":
        inner = t[1:-1]
        if not any(c in inner for c in "“”‘’"):
            t = inner.strip()
    return t


def _diag_print(msg: str, *args) -> None:
    """Print a diagnostic line to stdout (teed to web-UI buffer) AND to the
    logger (teed to reply_diag.log file handler). Both channels are needed:
    - print() goes through _Tee → terminal + web-UI log buffer.
    - logger.info() → file handler (reply_diag.log) for post-mortem reading.
    Never raises."""
    try:
        formatted = msg % args if args else msg
    except Exception:
        formatted = msg
    try:
        print(formatted)
    except Exception:
        pass
    try:
        logger.info("%s", formatted)
    except Exception:
        pass


def _diagnose_reply_read(page: Page) -> None:
    """Log what the page actually contains when a reply read is failing, so a
    markup change is READABLE from the agent log rather than requiring a live
    console session. Runs on persistent empty reads AND after a read that
    returned only directive markers (no real text). Never raises.

    Uses _diag_print() so output goes to BOTH the teed stdout (terminal + web-UI
    job log) AND the reply_diag.log file handler (post-mortem reading)."""
    _diag_print("[reply-diag] === REPLY-READ DIAGNOSTIC STARTING ===")

    # ------------------------------------------------------------------ #
    # 1. Per-selector probe: match count + last-element innerText length. #
    # ------------------------------------------------------------------ #
    probes = {
        "DIL_RENDERER": DIL_RENDERER,
        "TURN_ACTION_CONTROLS": TURN_ACTION_CONTROLS,
        "COPY_BUTTON": COPY_BUTTON,
        "ASSISTANT_MESSAGE": ASSISTANT_MESSAGE,
        "CONVERSATION_TURN": CONVERSATION_TURN,
        "[role=any]": "[data-message-author-role]",
        "article": "article",
        "testid conv-turn": "[data-testid^='conversation-turn']",
    }
    for name, sel in probes.items():
        try:
            els = page.query_selector_all(sel)
            last_len = 0
            if els:
                try:
                    last_len = len((els[-1].inner_text() or "").strip())
                except Exception:
                    last_len = -1
            _diag_print("[reply-diag] %-20s matched %d, last innerText=%d chars (%s)",
                        name, len(els), last_len, sel)
        except Exception as exc:
            _diag_print("[reply-diag] %-20s query FAILED (%s)", name, exc)

    # ------------------------------------------------------------------ #
    # 2. Copy-button audit: for every Copy button found, log its context  #
    #    so we can tell whether it belongs to the reply or a sub-element. #
    # ------------------------------------------------------------------ #
    try:
        info = page.evaluate(
            r"""() => {
                const copyBtns = Array.from(document.querySelectorAll(
                    "button[aria-label='Copy'], button[data-testid='copy-turn-action-button']"
                ));
                return copyBtns.map((b, i) => {
                    let ancestor = b.parentElement;
                    let depth = 0;
                    while (ancestor && depth < 8) {
                        const cn = (ancestor.className || '').toString();
                        if (cn.includes('turn') || cn.includes('message') ||
                            cn.includes('markdown') || cn.includes('prose')) break;
                        ancestor = ancestor.parentElement;
                        depth++;
                    }
                    return {
                        index: i,
                        label: b.getAttribute('aria-label'),
                        btnHTML: b.outerHTML.slice(0, 200),
                        ancestorTag: ancestor ? ancestor.tagName : null,
                        ancestorClass: ancestor ? (ancestor.className||'').toString().slice(0,120) : null,
                        ancestorText: ancestor ? (ancestor.innerText||'').slice(0,200) : null
                    };
                });
            }"""
        ) or []
        _diag_print("[reply-diag] %d Copy button(s) found:", len(info))
        for b in info:
            _diag_print("[reply-diag]   Copy[%d] label=%r btnHTML=%r",
                        b.get("index"), b.get("label"), b.get("btnHTML"))
            _diag_print("[reply-diag]   Copy[%d] nearest turn ancestor: <%s class=%r> text=%r",
                        b.get("index"), b.get("ancestorTag"), b.get("ancestorClass"),
                        b.get("ancestorText"))
    except Exception as exc:
        _diag_print("[reply-diag] Copy-button audit FAILED (%s)", exc)

    # ------------------------------------------------------------------ #
    # 3. Message-structure sweep: every substantive text node in <main>.  #
    # ------------------------------------------------------------------ #
    try:
        nodes = page.evaluate(
            r"""() => {
                const main = document.querySelector('main');
                if (!main) return [];
                const results = [];
                function walk(el, depth) {
                    if (depth > 12 || results.length > 40) return;
                    const t = (el.innerText || '').trim();
                    if (t.length < 20) return;
                    const cn = (el.className || '').toString();
                    if (cn.includes('turn') || cn.includes('message') ||
                        cn.includes('markdown') || cn.includes('prose') ||
                        cn.includes('content') || cn.includes('response') ||
                        cn.includes('assistant') || cn.includes('text')) {
                        results.push({ tag: el.tagName, cls: cn.slice(0, 150), text: t.slice(0, 500) });
                    }
                    for (const child of el.children) walk(child, depth + 1);
                }
                walk(main, 0);
                return results;
            }"""
        ) or []
        _diag_print("[reply-diag] message-structure sweep (%d nodes in <main>):", len(nodes))
        for n in nodes:
            _diag_print("[reply-diag]   <%s class=%r> text[:200]=%r",
                        n.get("tag"), n.get("cls"), n.get("text", "")[:200])
    except Exception as exc:
        _diag_print("[reply-diag] message-structure sweep FAILED (%s)", exc)

    # ------------------------------------------------------------------ #
    # 4. Last-turn outerHTML (truncated).                                 #
    # ------------------------------------------------------------------ #
    try:
        turns = page.query_selector_all(CONVERSATION_TURN)
        if turns:
            html = turns[-1].evaluate("el => el.outerHTML") or ""
            _diag_print("[reply-diag] last-turn outerHTML (%d chars, trunc 1500):\n%s",
                        len(html), html[:1500])
        else:
            _diag_print("[reply-diag] no CONVERSATION_TURN elements to dump")
    except Exception as exc:
        _diag_print("[reply-diag] outerHTML dump FAILED (%s)", exc)

    _diag_print("[reply-diag] === DIAGNOSTIC COMPLETE ===")


def _same_as_prompt(text: str, sent_prompt: str | None) -> bool:
    """True if `text` is (essentially) the prompt we just sent — meaning we read
    the USER turn, not the assistant reply. Compared on whitespace-collapsed
    text so trivial rendering differences do not defeat the guard."""
    if not sent_prompt:
        return False
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip()
    return norm(text) == norm(sent_prompt)


def _count_dil_renderers(page: Page) -> int:
    """How many DilRenderer reply containers currently exist. Captured BEFORE a
    turn so the reader can tell a NEW reply from the previous one. Never raises."""
    try:
        return len(page.query_selector_all(DIL_RENDERER))
    except Exception:
        return 0


def _read_reply_via_dil(page: Page, sent_prompt: str | None,
                        prior_count: int | None) -> str:
    """PRIMARY read: the rendered reply container [class*='DilRenderer'].

    Accepts text only when a NEW renderer has appeared since the turn was sent
    (count grew past ``prior_count``); otherwise returns "" so an older reply is
    never re-read. Reads the LAST renderer, but if that one's text equals the
    sent prompt (defensive — user turns should not be DilRenderer), falls back to
    the last renderer whose text differs from the prompt. Strips directives + UI
    chrome; rejects a result equal to the prompt. Never raises."""
    try:
        els = page.query_selector_all(DIL_RENDERER)
    except Exception as exc:
        logger.info("reply-read: dil-renderer query errored (%s).", exc)
        return ""
    n = len(els)
    if n == 0:
        return ""
    if prior_count is not None and n <= prior_count:
        # No new renderer yet — the reply has not rendered; do not read a stale one.
        logger.info("reply-read: dil-renderer count %d not greater than prior %d — "
                    "reply not rendered yet.", n, prior_count)
        return ""
    # Prefer the LAST renderer; if it looks like the prompt, walk back to the
    # last one whose text differs from the prompt.
    candidates = list(reversed(els))
    for el in candidates:
        try:
            raw = el.inner_text() or ""
        except Exception:
            try:
                raw = el.text_content() or ""
            except Exception:
                raw = ""
        # DilRenderer contains ONLY the reply body — never the UI chrome. So we
        # strip directive markers only, NOT UI chrome. Applying chrome stripping
        # here would wipe a real design whose text is e.g. "PRO" or "SHARE".
        cleaned = _strip_chat_directives(raw.strip()).strip() if raw else ""
        if not cleaned:
            continue
        if _same_as_prompt(cleaned, sent_prompt):
            # This renderer holds the prompt (unexpected) — try an earlier one.
            continue
        logger.info("reply-read: dil-renderer returning %d chars (of %d renderers, prior=%s), "
                    "preview=%r", len(cleaned), n, prior_count, cleaned[:120])
        return cleaned
    return ""


def _read_reply_via_copy(page: Page, sent_prompt: str | None) -> str:
    """PRIMARY read: click the LAST reply's Copy button and read the clipboard.

    This yields the reply as MARKDOWN (fenced ```json blocks intact), which is
    what the JSON parsers want. Steps: find the last TURN_ACTION_CONTROLS bar,
    the Copy button within it, clear the clipboard, click, then poll
    clipboard.readText() for up to ~2s. Rejected (returns "") if the clipboard
    stays empty or comes back equal to the prompt we just sent (that would mean
    we copied the user turn). Never raises."""
    try:
        bars = page.query_selector_all(TURN_ACTION_CONTROLS)
        copy_btn = None
        if bars:
            copy_btn = bars[-1].query_selector(COPY_BUTTON)
        if copy_btn is None:
            # No per-turn bar matched; fall back to the last Copy button anywhere.
            btns = page.query_selector_all(COPY_BUTTON)
            copy_btn = btns[-1] if btns else None
        if copy_btn is None:
            logger.info("reply-read: no Copy button found (bars=%d) — copy path unavailable.",
                        len(bars))
            return ""
        # Log which button we are about to click so a wrong-element click is
        # visible in the log without a live console session.
        try:
            all_btns = page.query_selector_all(COPY_BUTTON)
            btn_html = copy_btn.evaluate("el => el.outerHTML").strip()[:200]
            # Is this button inside the last bar, or a page-wide fallback?
            source = "last-bar" if (bars and bars[-1].query_selector(COPY_BUTTON) is not None) else "page-fallback"
            logger.info("reply-read: clicking Copy button (%s of %d found) source=%s html=%r",
                        1, len(all_btns), source, btn_html)
        except Exception:
            pass
        # Clear the clipboard so a stale value cannot masquerade as this reply.
        try:
            page.evaluate("() => navigator.clipboard.writeText('')")
        except Exception as exc:
            logger.warning("reply-read: clipboard clear FAILED (%s) — clipboard permission "
                           "problem; copy path cannot be trusted this turn.", exc)
            return ""
        copy_btn.click()
        deadline = time.time() + 2.0
        got = ""
        while time.time() < deadline:
            try:
                got = page.evaluate("() => navigator.clipboard.readText()") or ""
            except Exception as exc:
                logger.warning("reply-read: clipboard.readText FAILED (%s) — clipboard "
                               "permission problem; abandoning copy path.", exc)
                return ""
            if got.strip():
                break
            page.wait_for_timeout(150)
        got = got.strip()
        if not got:
            logger.info("reply-read: Copy clicked but clipboard stayed empty within 2s.")
            return ""
        if _same_as_prompt(got, sent_prompt):
            logger.info("reply-read: clipboard equals the sent prompt — copied the USER turn, "
                        "rejecting copy result.")
            return ""
        return got
    except Exception as exc:
        logger.info("reply-read: copy path errored (%s).", exc)
        return ""


def _hover_last_turn(page: Page) -> None:
    """Hover the bottom of the thread so ChatGPT renders the per-turn action
    buttons (Copy etc. only appear on hover / focus). Best-effort: move the mouse
    over the last action-controls bar if present, else over the last turn, else
    the bottom-centre of <main>. Never raises."""
    try:
        target = None
        bars = page.query_selector_all(TURN_ACTION_CONTROLS)
        if bars:
            target = bars[-1]
        if target is None:
            turns = page.query_selector_all(CONVERSATION_TURN)
            if turns:
                target = turns[-1]
        if target is not None:
            try:
                target.scroll_into_view_if_needed(timeout=1000)
            except Exception:
                pass
            try:
                target.hover(timeout=1000)
                return
            except Exception:
                pass
        # Last resort: hover the bottom-centre of the main thread region.
        main = page.query_selector("main")
        if main is not None:
            box = main.bounding_box()
            if box:
                page.mouse.move(box["x"] + box["width"] / 2,
                                box["y"] + box["height"] - 10)
    except Exception:
        pass


def _read_reply_via_said_heading(page: Page, sent_prompt: str | None) -> str:
    """DOM read via the screen-reader headings ChatGPT renders in main.innerText.

    Each message is preceded by a heading: the user's turn by "You said:" and the
    assistant's by "ChatGPT said:". So the assistant reply is the text AFTER the
    LAST "ChatGPT said:", cut off at the next "You said:" if one follows.

    This is markup-independent (it reads main.innerText, not any class/testid), so
    it survives ChatGPT's DOM churn — and it works even before the Copy button
    renders, which is why it runs on every attempt. Rejected (returns "") if the
    slice equals the sent prompt. Never raises."""
    try:
        main = page.query_selector("main")
        if main is None:
            return ""
        full = main.inner_text() or ""
    except Exception as exc:
        logger.info("reply-read: said-heading read errored (%s).", exc)
        return ""
    if not full:
        return ""
    # Find the LAST assistant heading. Match a few likely spellings/casings.
    lower = full.lower()
    marker = None
    idx = -1
    for m in ("chatgpt said:", "assistant said:"):
        pos = lower.rfind(m)
        if pos > idx:
            idx, marker = pos, m
    if idx < 0:
        return ""
    after = full[idx + len(marker):]
    # Cut at the next "You said:" (a following user turn), if present.
    nxt = after.lower().find("you said:")
    if nxt >= 0:
        after = after[:nxt]
    after = _strip_ui_chrome(after.strip())
    after = _strip_chat_directives(after).strip() if after else ""
    if not after:
        return ""
    if _same_as_prompt(after, sent_prompt):
        logger.info("reply-read: said-heading slice equals the sent prompt — rejecting.")
        return ""
    logger.info("reply-read: said-heading returning %d chars, preview=%r",
                len(after), after[:120])
    return after


def _read_reply_via_dom(page: Page, sent_prompt: str | None) -> str:
    """FALLBACK read: from the last action-controls bar, walk UP to the turn
    container and read its innerText, EXCLUDING the controls bar itself (its
    Copy/like button labels are not part of the reply). If no controls bar is
    present, fall back to the last ASSISTANT_MESSAGE / CONVERSATION_TURN element.
    Rejected (returns "") if the text equals the sent prompt. Never raises."""
    text = ""
    try:
        bars = page.query_selector_all(TURN_ACTION_CONTROLS)
        if bars:
            # Walk to the enclosing turn container and read its text minus the
            # controls bar, done in one evaluate so we operate on the same node.
            text = bars[-1].evaluate(
                """bar => {
                    let turn = bar.closest("[data-testid^='conversation-turn-'], "
                        + "[class*='conversation-turn'], article[class*='turn']") || bar.parentElement;
                    if (!turn) return "";
                    // Prefer the rendered-markdown body if present; else whole turn.
                    let body = turn.querySelector("[class*='markdown'], [class*='prose']") || turn;
                    let t = body.innerText || "";
                    // If we read the whole turn, subtract the controls bar text.
                    if (body === turn) { t = t.replace(bar.innerText || "", ""); }
                    return t;
                }"""
            ) or ""
        if not text.strip():
            msgs = page.query_selector_all(ASSISTANT_MESSAGE)
            if msgs:
                text = msgs[-1].inner_text() or ""
        if not text.strip():
            turns = page.query_selector_all(CONVERSATION_TURN)
            if turns:
                text = turns[-1].inner_text() or ""
    except Exception as exc:
        logger.info("reply-read: dom path errored (%s).", exc)
        text = ""
    text = _strip_ui_chrome(text.strip()) if text else ""
    if text and _same_as_prompt(text, sent_prompt):
        logger.info("reply-read: DOM text equals the sent prompt — read the USER turn, rejecting.")
        return ""
    return text


def get_last_text_reply(page: Page, sent_prompt: str | None = None,
                        retry_ms: int = 20_000, poll_ms: int = 500,
                        prior_dil_count: int | None = None) -> str:
    """Read the last assistant reply, retrying if it comes back empty.

    Order per attempt:
      1. dil-renderer  — the rendered reply container [class*='DilRenderer'];
         accepted only when a NEW one appeared since the turn (count grew past
         ``prior_dil_count``, captured by the caller before submitting).
      2. said-heading  — main.innerText after the last "ChatGPT said:".
      3. copy-button   — clipboard markdown (keeps ```json fences intact).
      4. dom-parent    — turn-container innerText.
    Every method strips directives + UI chrome and rejects a result equal to
    ``sent_prompt`` (that would mean the USER turn was read). Retries for up to
    ``retry_ms`` (default 20s) because the reply / its action bar can lag the
    stop-button detach. On a persistent empty read it dumps page diagnostics so
    selector drift is visible in the log. Logs which method won and the length.

    ``prior_dil_count`` is the DilRenderer count captured BEFORE the turn was
    submitted (see _count_dil_renderers). When None, the dil-renderer method
    still runs but cannot apply the count-grew guard, so it just reads the last
    renderer whose text differs from the prompt.
    """
    deadline = time.time() + retry_ms / 1000.0
    attempt = 0
    text = ""
    method = "none"
    saw_directive_only = False  # copy returned raw text but it was all directives

    def _clean(s: str) -> str:
        """Strip UI chrome + directive markers; return what actually remains."""
        if not s:
            return ""
        return _strip_chat_directives(_strip_ui_chrome(s.strip())).strip()

    def _read_once() -> tuple[str, str]:
        """One cascade over all four methods; returns (text, method). dil-renderer
        text is directive-stripped only (no chrome). Sets saw_directive_only."""
        nonlocal saw_directive_only
        # 1) dil-renderer — the actual rendered reply container. Most direct.
        #    Directive-stripped only (no UI-chrome stripping — the reply body has
        #    no chrome, and it could wipe a real design reading "PRO"/"SHARE").
        dil = _read_reply_via_dil(page, sent_prompt, prior_dil_count)
        if dil:
            return dil, "dil-renderer"
        # 2) said-heading (main.innerText after the last "ChatGPT said:").
        said = _clean(_read_reply_via_said_heading(page, sent_prompt))
        if said:
            return said, "said-heading"
        # 3) Copy button → clipboard markdown (keeps ```json fences intact).
        raw_copy = _read_reply_via_copy(page, sent_prompt)
        cleaned_copy = _clean(raw_copy)
        if cleaned_copy:
            return cleaned_copy, "copy-button"
        if raw_copy:
            saw_directive_only = True
            logger.info("reply-read: copy returned directive-only (%d raw chars).", len(raw_copy))
        # 4) dom-parent (turn container innerText).
        dom = _clean(_read_reply_via_dom(page, sent_prompt))
        if dom:
            return dom, "dom-parent"
        return "", "none"

    def _streaming() -> bool:
        """True if a stop / generation indicator is currently visible — ChatGPT
        is still rendering, so any text read now may be partial. Never raises."""
        for sel in (STOP_BUTTON, IMAGE_LOADER):
            try:
                el = page.query_selector(sel)
                if el is not None and el.is_visible():
                    return True
            except Exception:
                pass
        return False

    # Phase 1: get non-empty text (retry until the reply renders at all).
    while True:
        attempt += 1
        _hover_last_turn(page)  # Copy only appears on hover/focus.
        text, method = _read_once()
        logger.info("reply-read attempt %d: %d chars via %s",
                    attempt, len(text), method if text else "none")
        if text or time.time() >= deadline:
            break
        time.sleep(poll_ms / 1000.0)

    # Phase 2: STABILITY. ChatGPT keeps rendering progressively after the stop
    # button detaches, so a long reply (e.g. the identify JSON) can be truncated
    # if read too early. Re-read every 500ms and only accept once the text is
    # unchanged for 3 consecutive reads (~1.5s) AND no streaming indicator is
    # showing. Capped at the same 20s deadline. Short replies stabilise instantly.
    if text:
        STABLE_NEEDED = 3
        stable_reads = 1          # the read above counts as the first
        last = text
        while stable_reads < STABLE_NEEDED and time.time() < deadline:
            time.sleep(0.5)
            # If ChatGPT is visibly streaming again, reset the stability window.
            if _streaming():
                logger.info("reply-read: streaming indicator visible — waiting (len=%d).", len(last))
                stable_reads = 0
                cur, m = _read_once()
                if cur:
                    last, method = cur, m
                continue
            cur, m = _read_once()
            if cur and cur == last:
                stable_reads += 1
            else:
                # Grew or changed — reset the window and take the newer text.
                if cur and cur != last:
                    logger.info("reply-read: text still growing (%d -> %d chars) — resetting stability.",
                                len(last), len(cur))
                    last, method = cur, m
                stable_reads = 1 if cur else stable_reads
        text = last
        logger.info("reply-read: reply stable after %d read(s) (%d chars) via %s",
                    stable_reads, len(text), method)

    if text:
        # Strip wrapping quotes/markdown so a short answer ("ACT", `ACT`,
        # ```ACT```) is accepted as-is. Never rejects short text. (Directive
        # stripping already ran inside the loop's _clean(); this only unwraps.)
        unwrapped = _unwrap_reply(text)
        if unwrapped != text:
            logger.info("reply-read: unwrapped quotes/markdown — %r -> %r",
                        text[:100], unwrapped[:100])
            text = unwrapped
        logger.info("reply-read: SUCCEEDED via %s (%d chars)", method, len(text))
    else:
        detail = f"{attempt} attempts over {retry_ms/1000:.0f}s"
        if saw_directive_only:
            detail += "; copy returned directive-only markers on ≥1 attempt"
        logger.warning("reply-read: empty after %s — dumping page diagnostics.", detail)
        _diag_event("fallback", "Assistant reply read back empty", detail=detail)
        _diagnose_reply_read(page)
    return text


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
    # Count reply containers before the turn so the reader only accepts a NEW one.
    prior_dil = _count_dil_renderers(page)
    send_turn(page, prompt=prompt, image_paths=[image_path], run_id=run_id)

    reply = get_last_text_reply(page, sent_prompt=prompt, prior_dil_count=prior_dil)
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
            dump_page_controls(page, reason=f"stop-button-missing extract {run_id}")
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


def _image_identity(src: str, alt: str | None = None) -> str | None:
    """A STABLE identity for a generated-image src used for dedup and
    before/after comparison. Strategy depends on the URL scheme:

    * blob: URLs (current ChatGPT scheme) — the blob UUID is already a unique
      opaque key for this page session; use the raw src. These have no file_id
      or meaningful path component to extract.
    * Historical URL schemes — try, in order:
        1. `id=file_...` / `id=file-...` query param (original estuary scheme),
        2. any `file_...` / `file-...` token anywhere in the URL,
        3. URL path without query string (drops volatile signed-URL args).

    `alt` is included as a tie-breaker so that if two different generated images
    somehow share the same src (unlikely but possible with signed-URL collapse),
    the alt text ("Generated image 1" vs "Generated image 2") keeps them separate.

    Returns None only for a truly empty src."""
    if not src:
        return None
    # blob: URL — use as-is (already unique, no file token to extract).
    if src.startswith("blob:"):
        # Append alt so repeated renders of the same blob don't collapse
        # (shouldn't happen in practice, but makes the key maximally specific).
        suffix = f"#{alt}" if alt else ""
        return src + suffix
    # Historical URL schemes: try file-token extraction.
    m = re.search(r"id=(file[-_][A-Za-z0-9]+)", src)
    if m:
        return m.group(1)
    m = re.search(r"(file[-_][A-Za-z0-9]{6,})", src)
    if m:
        return m.group(1)
    # Drop the query string so a re-signed URL for the same image collapses.
    return src.split("?", 1)[0] or src


def _diagnose_images(page: Page, reason: str = "") -> None:
    """Log what generated-image candidates the page actually contains, so an
    image-URL scheme change is READABLE from the log (mirrors the reply-reader
    diagnostic). Per selector: match count. Plus the src[:120]/alt/naturalWidth
    of the last 5 <img> on the page, and the count inside the last turn
    container. Never raises."""
    banner = "[image-diag]" + (f" reason={reason}" if reason else "")
    probes = {
        "GENERATED_IMG": GENERATED_IMG,
        "alt^=Generated image": "img[alt^='Generated image']",
        "alt*=Generated": "img[alt*='Generated']",
        "main img": "main img",
        "blob: imgs": "img[src^='blob:']",
        "oaiusercontent": "img[src*='oaiusercontent']",
        "estuary": "img[src*='estuary']",
    }
    for name, sel in probes.items():
        try:
            n = len(page.query_selector_all(sel))
        except Exception as exc:
            n = -1
            logger.info("%s %-16s query FAILED (%s)", banner, name, exc)
            continue
        logger.info("%s %-16s matched %d (%s)", banner, name, n, sel)
    # Images inside the last turn container (found the same way the reply reader
    # locates the turn: from the last turn-action-controls bar).
    try:
        bars = page.query_selector_all(TURN_ACTION_CONTROLS)
        if bars:
            n_turn = bars[-1].evaluate(
                """bar => {
                    let turn = bar.closest("[data-testid^='conversation-turn-'], "
                        + "[class*='conversation-turn'], article[class*='turn']") || bar.parentElement;
                    return turn ? turn.querySelectorAll('img').length : -1;
                }"""
            )
            logger.info("%s imgs in last turn container: %s", banner, n_turn)
        else:
            logger.info("%s no turn-action-controls bar to locate the last turn", banner)
    except Exception as exc:
        logger.info("%s last-turn img count FAILED (%s)", banner, exc)
    # Last 5 <img> on the page: src[:120], alt, naturalWidth.
    try:
        info = page.evaluate(
            """() => Array.from(document.querySelectorAll('img')).slice(-5).map(im => ({
                src: (im.getAttribute('src') || '').slice(0, 120),
                alt: im.getAttribute('alt'),
                nw: im.naturalWidth
            }))"""
        ) or []
        logger.info("%s last %d <img> on page:", banner, len(info))
        for im in info:
            logger.info("%s   img src=%r alt=%r naturalWidth=%s",
                        banner, im.get("src"), im.get("alt"), im.get("nw"))
    except Exception as exc:
        logger.info("%s last-imgs dump FAILED (%s)", banner, exc)


def _dump_last_dil_html(page: Page, reason: str = "") -> None:
    """Log the outerHTML (first 1500 chars) of the LAST DilRenderer, so we can
    see where the generated image sits when collection found none. Never raises."""
    banner = "[image-diag]" + (f" reason={reason}" if reason else "")
    try:
        els = page.query_selector_all(DIL_RENDERER)
        if not els:
            logger.info("%s no DilRenderer element to dump", banner)
            return
        html = els[-1].evaluate("el => el.outerHTML") or ""
        logger.info("%s last DilRenderer outerHTML (%d chars, trunc 1500):\n%s",
                    banner, len(html), html[:1500])
    except Exception as exc:
        logger.info("%s DilRenderer dump FAILED (%s)", banner, exc)


def _srcs_from(elements, exclude_ids: set[str]) -> list[str]:
    """Turn a list of <img> handles into a de-duplicated list of NEW srcs,
    excluding any whose identity is in `exclude_ids`. Uses alt + src for
    identity so blob: URLs (which have no file_id) deduplicate correctly.

    Explicitly rejects blob: imgs whose alt does NOT start with "Generated"
    — those are COMPOSER_THUMBNAIL uploads, not generated results. The
    alt-attribute GENERATED_IMG selector should already exclude them, but this
    guard defends against any future selector broadening."""
    seen: set[str] = set()
    new_srcs: list[str] = []
    for img in elements:
        try:
            src = img.get_attribute("src") or ""
            alt = img.get_attribute("alt") or ""
        except Exception:
            continue
        if not src:
            continue
        # Hard guard: reject composer blobs (blob: src without a "Generated" alt).
        if src.startswith("blob:") and not alt.lower().startswith("generated"):
            continue
        ident = _image_identity(src, alt)
        if not ident or ident in exclude_ids or ident in seen:
            continue
        seen.add(ident)
        new_srcs.append(src)
    return new_srcs


def _collect_image_ids(page: Page) -> set[str]:
    """Return the identities of all GENERATED_IMG currently in the DOM (the
    'before' snapshot). Uses the SAME selector + identity function as the 'after'
    collection so a src that was present before is reliably excluded after."""
    ids: set[str] = set()
    for img in page.query_selector_all(GENERATED_IMG):
        try:
            src = img.get_attribute("src") or ""
            alt = img.get_attribute("alt") or ""
        except Exception:
            continue
        if not src:
            continue
        if src.startswith("blob:") and not alt.lower().startswith("generated"):
            continue
        ident = _image_identity(src, alt)
        if ident:
            ids.add(ident)
    return ids


def _collect_new_image_srcs(page: Page, exclude_ids: set[str]) -> list[str]:
    """Collect unique NEW image srcs from the whole page (page-wide fallback)."""
    return _srcs_from(page.query_selector_all(GENERATED_IMG), exclude_ids)


def _collect_last_turn_images(page: Page, exclude_ids: set[str]) -> list[str]:
    """Collect NEW images from the LAST conversation turn only, excluding
    pre-existing ids. Scopes to the last CONVERSATION_TURN so the user's just
    -uploaded images are never picked up. Falls back to a page-wide search if no
    turn element matches. On a 0-image result, dumps the image diagnostic so an
    image-URL scheme change is visible in the log."""
    turns = page.query_selector_all(CONVERSATION_TURN)
    if not turns:
        new_srcs = _collect_new_image_srcs(page, exclude_ids)
    else:
        new_srcs = _srcs_from(turns[-1].query_selector_all(GENERATED_IMG), exclude_ids)
        if not new_srcs:
            # The turn container may not enclose the <img> (markup drift), so try
            # the whole page before giving up — still excluding pre-existing ids.
            page_wide = _collect_new_image_srcs(page, exclude_ids)
            if page_wide:
                logger.info("_collect_last_turn_images: last-turn scope found 0, "
                            "page-wide found %d — using page-wide.", len(page_wide))
                new_srcs = page_wide
    if not new_srcs:
        _diagnose_images(page, reason="0 new images collected")
    return new_srcs


# JS that returns, for the current page, the generated-image candidates that are
# (a) NOT user attachments/avatars, (b) fully loaded, and (c) large enough to be
# a real result. Returns [{src, alt, nw, complete}] for each candidate <img>
# matching GENERATED_IMG. Kept in one round-trip so the poll is cheap.
_NEW_IMAGE_PROBE_JS = r"""
(sel) => {
  const imgs = Array.from(document.querySelectorAll(sel));
  return imgs.map(im => {
    const src = im.getAttribute('src') || '';
    const alt = im.getAttribute('alt') || '';
    return { src, alt, nw: im.naturalWidth || 0, nh: im.naturalHeight || 0,
             complete: !!im.complete };
  });
}
"""


def _is_user_attachment(src: str, alt: str) -> bool:
    """True if an <img> is a user attachment / avatar / data-URL, not a generated
    result. Generated images have alt starting 'Generated'; everything else that
    is a blob:/data: URL or an 'attachment'/'avatar' alt is excluded."""
    a = (alt or "").strip().lower()
    s = (src or "").strip().lower()
    if a.startswith("generated"):
        return False   # definitely a result
    if "user attachment" in a or "avatar" in a or "profile" in a:
        return True
    if s.startswith("data:"):
        return True
    # A bare blob: with no "Generated" alt is a composer thumbnail / attachment.
    if s.startswith("blob:") and not a.startswith("generated"):
        return True
    return False


def _wait_for_new_image(page: Page, exclude_ids: set[str], run_id: str,
                        timeout_ms: int = 180_000, poll_ms: int = 2000) -> list[str]:
    """Poll until a NEW generated image appears that is not in `exclude_ids` and
    is not a user attachment/avatar, is fully loaded (complete && naturalWidth>0,
    >= 256px), and whose src has stopped changing for 2 consecutive polls. Then
    collect and return the new srcs (scoped to the last turn, page-wide fallback).

    Image generation can take 30–120s AFTER the stop button detaches, so this is
    what actually gates collection. Returns [] on timeout. Never raises."""
    t0 = time.monotonic()
    deadline = time.monotonic() + timeout_ms / 1000.0
    MIN_DIM = 256
    stable_src = None
    stable_polls = 0
    while time.monotonic() < deadline:
        try:
            cands = page.evaluate(_NEW_IMAGE_PROBE_JS, GENERATED_IMG) or []
        except Exception:
            cands = []
        # Keep only real, loaded, large-enough, NEW candidates.
        valid = []
        for c in cands:
            src, alt = c.get("src", ""), c.get("alt", "")
            if not src or _is_user_attachment(src, alt):
                continue
            if not c.get("complete") or c.get("nw", 0) < MIN_DIM or c.get("nh", 0) < MIN_DIM:
                continue
            ident = _image_identity(src, alt)
            if not ident or ident in exclude_ids:
                continue
            valid.append(src)
        if valid:
            # Require the newest src to stay put for 2 polls (image finished
            # streaming and its blob URL is not being swapped).
            newest = valid[-1]
            if newest == stable_src:
                stable_polls += 1
            else:
                stable_src, stable_polls = newest, 1
            if stable_polls >= 2:
                elapsed = time.monotonic() - t0
                # Collect via the normal path so dedup/scoping stay identical.
                srcs = _collect_last_turn_images(page, exclude_ids)
                if not srcs:
                    srcs = valid  # last-turn scope missed it; use the validated set
                logger.info("[%s] image turn finished after %.0fs, %d new image(s)",
                            run_id, elapsed, len(srcs))
                return srcs
        time.sleep(poll_ms / 1000.0)
    logger.warning("[%s] no new generated image after %.0fs.",
                   run_id, time.monotonic() - t0)
    return []


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
    streamed = False       # has the button been sustained-present (turn ran)?
    turn_over = False       # text_only: latched once it detaches after streaming
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
            streamed = True

        # SUSTAINED-BLIP GUARD (the identify text-turn 120s hang): the plain
        # flicker guard above only absorbs a ONE-poll blip. The identify listing
        # turn runs on the image-capable page, whose post-answer chrome re-renders
        # a same-selector button for 2-3 consecutive polls every couple of
        # seconds. Each sustained blip satisfied `present_streak >= 2`, reset the
        # quiet window, and the 1.5s of continuous absence never accumulated —
        # so the loop spun to the 120s timeout.
        #
        # For a TEXT turn there is only ever one generation: once the button has
        # detached after the turn demonstrably streamed, the turn is done and any
        # later re-attachment is page chrome, not new output. We latch that and
        # stop treating presence as streaming, letting the quiet window run from
        # the first detach. IMAGE turns are unchanged: a real second image can
        # legitimately restart generation there, so they must still react to a
        # sustained re-attachment.
        if text_only and streamed and absent_since is not None:
            turn_over = True
        if turn_over:
            streaming = False

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
                     "over" if turn_over else ("streaming" if streaming else "quiet-period"),
                     " (post-turn chrome, ignored)" if (raw_present and turn_over)
                     else (" (flicker-ignored)" if (raw_present and not streaming) else ""))

        if absent_since is not None and absent_for >= quiet_s:
            _phase("quiet period elapsed")
            return
        page.wait_for_timeout(500)
    raise GenerationTimeoutError("Turn never completed.")


def _enter_prompt(page: Page, prompt: str) -> None:
    """Enter a prompt into the focused ProseMirror editor via clipboard paste.

    Falls back to character-by-character typing if paste fails verification.
    Typing a 500+ char prompt costs several seconds per turn, so the paste path
    is the fast path we want — verification reads back from the SAME composer
    element PROMPT_BOX matched (a contenteditable, so its text is in
    textContent/innerText, NOT .value), and every outcome is logged.
    """
    expected = len(prompt.strip())

    # 1) Write to the clipboard. A permission failure here is a DISTINCT problem
    #    (the agent grants clipboard-read/write on the context, so this should
    #    not fail) — log it loudly rather than silently degrading to typing.
    try:
        page.evaluate("text => navigator.clipboard.writeText(text)", prompt)
    except Exception as exc:
        logger.warning("_enter_prompt: clipboard.writeText FAILED (%s). This is a "
                       "clipboard-permission problem, not a paste-verify miss — the "
                       "context should grant clipboard access. Typing instead this turn.", exc)
        _diag_event("fallback", "Clipboard write failed (permission?); typed instead",
                    detail=str(exc))
        _click_composer_and_type(page, prompt)
        return

    # 2) Paste into the composer and read the text back from the SAME element the
    #    PROMPT_BOX selector matched — never a hardcoded id (the composer lost its
    #    #prompt-textarea id when it became a bare ProseMirror contenteditable).
    try:
        page.click(PROMPT_BOX)
        page.keyboard.press("Control+V")
        page.wait_for_timeout(300)   # let the paste settle
        composer = page.query_selector(PROMPT_BOX)
        content = ""
        if composer is not None:
            try:
                content = composer.inner_text() or ""
            except Exception:
                content = composer.text_content() or ""
        else:
            logger.warning("_enter_prompt: PROMPT_BOX (%s) matched no element when "
                           "verifying the paste — the composer selector may need "
                           "updating.", PROMPT_BOX)
        actual = len(content.strip())
        if actual >= expected * 0.8:
            logger.info("_enter_prompt: PASTED ok (%d of ~%d chars via %s)",
                        actual, expected, PROMPT_BOX)
            return
        logger.info("_enter_prompt: paste verify short — read %d chars, expected ~%d "
                    "(via %s). Falling back to TYPING.", actual, expected, PROMPT_BOX)
        _diag_event("fallback", "Prompt paste incomplete; fell back to typing",
                    detail=f"pasted {actual} of ~{expected} chars")
    except Exception as exc:
        logger.info("_enter_prompt: paste step failed (%s); falling back to TYPING.", exc)
        _diag_event("fallback", "Prompt paste failed; fell back to typing", detail=str(exc))

    # Fallback: clear whatever partial paste left and type character by character.
    _click_composer_and_type(page, prompt)


def _click_composer_and_type(page: Page, prompt: str) -> None:
    """Clear the composer and type the prompt character by character (fallback)."""
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


def _clear_composer_attachments(page: Page, run_id: str = "") -> None:
    """Remove any files left attached to the composer from a previous turn, so a
    new upload does not end up with a stale extra attachment (the regenerate turn
    showed blob_thumbnails=2 when 1 was expected). Clicks each remove-attachment
    button until none remain (capped). Never raises."""
    try:
        for _ in range(6):  # cap: don't loop forever if a button won't clear
            btns = page.query_selector_all(COMPOSER_REMOVE_ATTACHMENT)
            if not btns:
                break
            n = len(btns)
            try:
                btns[0].click()
            except Exception:
                break
            page.wait_for_timeout(200)
        remaining = len(page.query_selector_all(COMPOSER_THUMBNAIL))
        if remaining:
            logger.info("[%s] composer still shows %d thumbnail(s) after cleanup.",
                        run_id or "cleanup", remaining)
    except Exception as exc:
        logger.info("[%s] composer attachment cleanup errored (%s).", run_id or "cleanup", exc)


def _composer_text(page: Page) -> str:
    """Current text in the composer (the element PROMPT_BOX matched). Empty
    string if it can't be read. Used to detect that a submit cleared the box."""
    try:
        el = page.query_selector(PROMPT_BOX)
        if el is None:
            return ""
        try:
            return (el.inner_text() or "").strip()
        except Exception:
            return (el.text_content() or "").strip()
    except Exception:
        return ""


def _submission_confirmed(page: Page, before_len: int) -> bool:
    """True if a submit appears to have landed: EITHER the stop control is now
    present (a turn started) OR the composer has cleared (its text dropped to
    near-empty from `before_len`). Never raises."""
    # Stop control present → the turn is running.
    try:
        if page.query_selector(STOP_BUTTON) is not None:
            return True
    except Exception:
        pass
    # Composer cleared → ProseMirror consumed the Enter and sent the message.
    now = len(_composer_text(page))
    if before_len > 0 and now <= max(2, before_len // 10):
        return True
    return False


def _submit_prompt(page: Page, run_id: str = "submit") -> bool:
    """Submit the prompt already typed into the composer, resiliently.

    Submission does NOT depend on finding a fixed send-button selector (every
    data-testid has been removed at least once). Order:

      1. Press Enter — ProseMirror submits natively on Enter. Then VERIFY by
         watching (up to ~3s) for the stop control to appear OR the composer to
         clear. Verification is what proves submission, not the presence of a
         button.
      2. If unconfirmed, resolve a Send button by aria-label (SEND_BUTTON_LABELS,
         CSS fallback) and click it, then verify again.
      3. If still unconfirmed, dump the live controls (once per process) so the
         new markup is readable from the log, and return False.

    Logs which path was taken and whether submission was confirmed. Returns True
    iff submission was confirmed.
    """
    before_len = len(_composer_text(page))

    # --- Path 1: Enter (ProseMirror native submit) ---
    try:
        page.keyboard.press("Enter")
    except Exception as exc:
        logger.info("[submit %s] Enter press raised (%s)", run_id, exc)
    if _wait_until(lambda: _submission_confirmed(page, before_len), timeout_ms=3000, poll_ms=200):
        logger.info("[submit %s] CONFIRMED via Enter (composer was %d chars)", run_id, before_len)
        return True
    logger.info("[submit %s] Enter did not confirm within 3s — trying Send button.", run_id)

    # --- Path 2: click a Send button resolved by aria-label ---
    btn = find_button_by_labels(page, SEND_BUTTON_LABELS, SEND_BUTTON, name="SEND_BUTTON")
    if btn is not None:
        try:
            btn.click()
            if _wait_until(lambda: _submission_confirmed(page, before_len), timeout_ms=3000, poll_ms=200):
                logger.info("[submit %s] CONFIRMED via Send button click.", run_id)
                return True
            logger.info("[submit %s] Send button clicked but submission not confirmed within 3s.", run_id)
        except Exception as exc:
            logger.info("[submit %s] Send button click raised (%s)", run_id, exc)
    else:
        logger.info("[submit %s] no Send button found by aria-label/CSS.", run_id)

    # --- Both paths failed: dump controls once so the markup is diagnosable. ---
    logger.warning("[submit %s] submission NOT CONFIRMED by either Enter or Send button — "
                   "dumping page controls.", run_id)
    _diag_event("fallback", "Prompt submission not confirmed",
                step=run_id, detail="Enter and Send-button paths both failed")
    _dump_controls_once(page, reason=f"submit-unconfirmed {run_id}")
    return False


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
    _names = ", ".join(Path(p).name for p in image_paths) or "(no files)"
    logger.info("upload not confirmed after 60s; re-attaching files once")
    _diag_event("upload_issue", "Upload not confirmed after 60s; re-attaching once",
                step=run_id, detail=_names)
    try:
        page.set_input_files(FILE_INPUT, list(image_paths))
    except Exception as exc:
        logger.info("re-attach set_input_files failed: %s", exc)
        _diag_event("retry", "Re-attach set_input_files failed",
                    step=run_id, detail=str(exc))

    if _wait_until(_attached, timeout_ms=60_000):
        _log_upload_state(page, expected_count, "attached after retry")
        return

    # Give up — screenshot + a plain message naming the file(s) that failed.
    _log_upload_state(page, expected_count, "FAILED")
    _save_error_screenshot(page, run_id)
    names = ", ".join(Path(p).name for p in image_paths) or "(no files)"
    _diag_event("upload_issue", "Upload did not attach in the composer after retry",
                step=run_id, detail=names)
    raise GenerationTimeoutError(
        f"Upload did not attach in the composer: {names}. "
        f"No blob thumbnail appeared and the send button never enabled."
    )


# ---------------------------------------------------------------------------
# Resilient button lookup by aria-label.
#
# ChatGPT removed EVERY data-testid at least once (confirmed live). So a control
# must be found by something that survives that: its aria-label. This resolver
# tries each candidate aria-label in turn, then a CSS fallback (which still
# carries the old data-testids so a future restoration also works), and LOGS
# which one matched. On a miss it logs exactly which labels it looked for — so
# the next time the markup changes, the log names the gap instead of the job
# dying on an opaque timeout. `name` is only for the log line.
# ---------------------------------------------------------------------------

def find_button_by_labels(page: Page, labels, css_fallback: str | None = None,
                          name: str = "button"):
    """Return the first button matching any of `labels` (aria-label), else the
    `css_fallback` selector, else None. Never raises; logs what matched/missed."""
    for label in labels or ():
        sel = f"button[aria-label='{label}']"
        try:
            el = page.query_selector(sel)
        except Exception:
            el = None
        if el is not None:
            logger.info("[selector] %s matched via aria-label %r", name, label)
            return el
    if css_fallback:
        try:
            el = page.query_selector(css_fallback)
        except Exception:
            el = None
        if el is not None:
            logger.info("[selector] %s matched via CSS fallback %r", name, css_fallback)
            return el
    logger.info("[selector] %s NOT FOUND — looked for aria-label(s) %s and fallback %r. "
                "ChatGPT markup may have changed; run the control diagnostic.",
                name, list(labels or ()), css_fallback)
    return None


# JS that reads the current controls in one round-trip: every button's
# aria-label + trimmed text, plus the tag/class/trimmed-outerHTML of the last
# element that looks like a conversation turn. Pure DOM, returns plain data.
_CONTROLS_DUMP_JS = r"""
() => {
  const btns = Array.from(document.querySelectorAll('button')).map(b => ({
    label: b.getAttribute('aria-label'),
    text: (b.innerText || '').trim().slice(0, 40),
    disabled: b.disabled || b.getAttribute('aria-disabled') === 'true'
  }));
  // Best-effort "last turn": prefer explicit turn markers, else a class
  // fragment inside <main>, so we log whatever ChatGPT now uses.
  const turnSel = "[data-testid^='conversation-turn-'], main [class*='conversation-turn'], main article[class*='turn'], main [class*='turn']";
  let turns = [];
  try { turns = Array.from(document.querySelectorAll(turnSel)); } catch (e) {}
  let lastTurn = null;
  if (turns.length) {
    const t = turns[turns.length - 1];
    lastTurn = {
      count: turns.length,
      tag: t.tagName,
      className: (t.className || '').toString().slice(0, 200),
      outerStart: (t.outerHTML || '').slice(0, 600)
    };
  }
  return { buttonCount: btns.length, buttons: btns, lastTurn };
}
"""


def dump_page_controls(page: Page, reason: str = "") -> None:
    """Log the page's current controls so a markup change is READABLE from logs.

    ChatGPT has changed its DOM out from under us more than once (ids gone, then
    every data-testid gone). When a control we need can't be found, this dumps —
    to the agent log the operator already reads — every button's aria-label and
    the structure of the last conversation turn, so the NEW attribute values can
    be lifted straight from the log instead of hand-running console scripts.

    Never raises: a diagnostic must not turn one failure into two."""
    try:
        data = page.evaluate(_CONTROLS_DUMP_JS)
    except Exception as exc:
        logger.info("[control-dump] could not read controls (%s)%s", exc,
                    f" — reason: {reason}" if reason else "")
        return

    banner = "[control-dump]" + (f" reason={reason}" if reason else "")
    buttons = data.get("buttons") or []
    logger.info("%s %d button(s) on the page:", banner, data.get("buttonCount", 0))
    for b in buttons:
        logger.info("%s   button aria-label=%r text=%r disabled=%s",
                    banner, b.get("label"), b.get("text"), b.get("disabled"))
    lt = data.get("lastTurn")
    if lt:
        logger.info("%s last turn (of %d): <%s class=%r>",
                    banner, lt.get("count"), (lt.get("tag") or "").lower(), lt.get("className"))
        logger.info("%s last turn outerHTML[:600]: %s", banner, lt.get("outerStart"))
    else:
        logger.info("%s no conversation-turn element matched — the turn selector "
                    "needs updating (config/selectors.py CONVERSATION_TURN).", banner)


def _send_button_ready(page: Page) -> bool:
    """True if the send button exists and is enabled (composer has content).

    Resolves the button by aria-label (SEND_BUTTON_LABELS) with the SEND_BUTTON
    CSS as fallback, so it survives a data-testid churn and logs how it found
    (or failed to find) the control."""
    try:
        btn = find_button_by_labels(page, SEND_BUTTON_LABELS, SEND_BUTTON, name="SEND_BUTTON")
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
        btn = find_button_by_labels(page, SEND_BUTTON_LABELS, SEND_BUTTON, name="SEND_BUTTON")
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

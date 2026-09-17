"""Browser lifecycle management for ChatGPT automation."""

from __future__ import annotations

import asyncio
import threading

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright

from config.selectors import PROMPT_BOX, LOGIN_SCREEN
from src.self_update import profile_dir


def _thread_diag() -> str:
    """Describe the calling thread and whether it has an asyncio event loop.

    Playwright's SYNC API must never run on a thread that already has a running
    asyncio loop — that is what triggers 'Sync API inside the async loop'. We
    log this at every launch so a wrong-thread call is obvious immediately."""
    name = threading.current_thread().name
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        has_loop = loop.is_running()
    except Exception:
        has_loop = False
    # Also detect a *running* loop bound to this thread (the actual danger).
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    return f"thread={name!r} event_loop_running={running} loop_present={has_loop}"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SessionExpiredError(Exception):
    """Raised when the browser session is not authenticated."""


class GenerationTimeoutError(Exception):
    """Raised when image generation does not complete within the expected time."""


class RateLimitError(Exception):
    """Raised when ChatGPT signals a rate-limit or cooldown."""


class ProfileLockedError(Exception):
    """Raised when the browser profile is already in use by another Chromium instance."""


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------


def launch_context(account: str) -> tuple[Playwright, BrowserContext]:
    """Launch a persistent Chromium context for the given account.

    Uses a per-account profile directory so cookies and session state are
    preserved across runs.

    Returns BOTH the Playwright driver and the context: the caller MUST keep the
    Playwright object and call ``pw.stop()`` when tearing down. Not stopping it
    leaves the sync driver's event loop attached to this thread, which makes the
    next ``sync_playwright().start()`` on the same thread raise
    'Sync API inside the async loop'. If the launch itself fails, we stop the
    driver here before raising so a retry starts clean.

    Raises
    ------
    ProfileLockedError
        If the profile directory is already in use by another Chromium instance.
    """
    # Absolute, fixed profile path derived from the app directory (never
    # relative to the CWD, never inside the self-update-swapped code/ folder),
    # so the saved ChatGPT session persists across restarts AND updates.
    profile_path = profile_dir(account)
    print(f"[agent] using browser profile {profile_path}")
    print(f"[agent] launching browser context ({_thread_diag()})")

    pw: Playwright | None = None
    try:
        pw = sync_playwright().start()
        context: BrowserContext = pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_path),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context.grant_permissions(
            ["clipboard-read", "clipboard-write"],
            origin="https://chatgpt.com",
        )
        return pw, context
    except Exception as exc:
        # Fully tear down the half-started driver so the thread is left clean for
        # a retry. Without this, the leaked pw is the source of the async-loop
        # error on the second attempt.
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                pass
        msg = str(exc).lower()
        if "already in use" in msg or "existing browser session" in msg or "lock" in msg:
            raise ProfileLockedError(
                f"Profile '{profile_path}' is already in use by another Chromium instance. "
                "Close any open automation browser windows and try again."
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Session validation
# ---------------------------------------------------------------------------


def is_logged_in(page: Page) -> bool:
    """Check whether the page has an active ChatGPT session.

    A logout is only declared when the LOGIN SCREEN is positively detected —
    never merely because the prompt box has not appeared yet. A missing prompt
    box is ambiguous: it is also missing while the editor is slowly rendering,
    on a busy machine, or during a transient re-render. Treating that ambiguity
    as "logged out" is what produced the false "session expired" that a restart
    cleared. So:

      1. Prompt box visible          -> logged in (fast path).
      2. Prompt box never appeared   -> look for positive login-screen markers.
           - login screen present    -> genuinely logged out.
           - login screen absent     -> assume still logged in (benefit of the
                                         doubt; do not fail a working session).
    """
    # 1) Fast, positive proof of an authenticated, ready editor.
    try:
        prompt = page.wait_for_selector(PROMPT_BOX, state="visible", timeout=10_000)
        if prompt is not None:
            return True
    except Exception:
        pass

    # 2) Prompt box not seen. Only call it logged out if the login screen is
    #    actually showing. Give it a moment in case the auth UI is rendering.
    try:
        login = page.query_selector(LOGIN_SCREEN)
        if login is not None and login.is_visible():
            print("[auth] login screen detected -> logged out")
            return False
    except Exception:
        pass

    # 3) Neither the editor nor the login screen resolved. Do NOT report a false
    #    logout — a restart used to 'fix' this precisely because the session was
    #    fine all along.
    print("[auth] prompt box not visible but no login screen either -> "
          "treating session as still valid (avoiding false logout)")
    return True


def require_login(page: Page, run_id: str) -> None:
    """Assert that the session is active; raise SessionExpiredError if not.

    Takes a screenshot before raising so the caller can inspect the
    login state visually.

    Parameters
    ----------
    page:
        The active page to check.
    run_id:
        Used to name the diagnostic screenshot.

    Raises
    ------
    SessionExpiredError
        Always raised when is_logged_in returns False.
    """
    if not is_logged_in(page):
        from pathlib import Path

        path = Path("logs") / f"{run_id}_session_expired.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            page.screenshot(path=str(path))
        except Exception:
            pass
        raise SessionExpiredError(
            f"Session not authenticated for this profile. Screenshot saved to {path}"
        )

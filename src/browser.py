"""Browser lifecycle management for ChatGPT automation."""

from __future__ import annotations

import asyncio
import atexit
import signal
import sys
import threading
import time

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright

from config.selectors import PROMPT_BOX, LOGIN_SCREEN
from src.self_update import (
    profile_dir, profiles_root, cookies_file_stat, app_dir,
    running_from_temp_location, install_writable,
)


# ---------------------------------------------------------------------------
# Graceful shutdown registry
# ---------------------------------------------------------------------------
# Chromium only flushes some session state (cookies, auth tokens) on a CLEAN
# shutdown. An abrupt process kill — a self-update os.execv, a PC shutdown, or
# Task Manager — can lose it, which shows as "logged out again" next start. We
# keep a reference to the live context + driver and close them on any normal
# exit (atexit) and on SIGTERM/SIGINT, so the common exit paths flush cleanly.
_live_context: BrowserContext | None = None
_live_pw: Playwright | None = None
_owner_thread: threading.Thread | None = None
_shutdown_registered = False


def _close_live(reason: str = "atexit") -> None:
    """Close the registered context + driver IF we are on the owning thread.

    Playwright's sync context is thread-bound: closing it from any other thread
    raises greenlet's "cannot switch to a different thread". atexit and signal
    handlers run on the MAIN thread, which is NOT the browser thread in the GUI.
    So this ONLY closes when called on the owning thread (true for CLI/headless,
    and for the GUI's own browser-thread teardown). Off-thread, it refuses and
    logs — it must NEVER attempt a cross-thread close. The GUI flushes on its
    browser thread via the restart flag / tray-Quit path instead.

    Idempotent and never raises."""
    global _live_context, _live_pw
    if _live_context is None and _live_pw is None:
        return
    if _owner_thread is not None and threading.current_thread() is not _owner_thread:
        print(f"[agent] skip browser close ({reason}): not on the owning thread "
              f"(current={threading.current_thread().name!r}, "
              f"owner={_owner_thread.name!r}); the owning thread handles the flush")
        return
    ctx, pw = _live_context, _live_pw
    _live_context, _live_pw = None, None
    if ctx is not None:
        try:
            print(f"[agent] closing browser context gracefully ({reason})")
            ctx.close()
        except Exception as exc:
            print(f"[agent] direct browser close raised ({reason}): {exc}")
    if pw is not None:
        try:
            pw.stop()
        except Exception:
            pass


def _clear_live_refs() -> None:
    """Let the owning thread's teardown reset the module refs after it closed."""
    global _live_context, _live_pw
    _live_context, _live_pw = None, None


def register_graceful_shutdown() -> None:
    """Install atexit + signal handlers once so the browser context is closed
    cleanly however the process ends. Safe to call multiple times."""
    global _shutdown_registered
    if _shutdown_registered:
        return
    _shutdown_registered = True
    atexit.register(lambda: _close_live("atexit"))

    def _sig(signum, _frame):
        _close_live(f"signal {signum}")
        # Re-raise default behaviour so the process still exits.
        try:
            signal.signal(signum, signal.SIG_DFL)
        except Exception:
            pass
        raise SystemExit(0)

    for _name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        s = getattr(signal, _name, None)
        if s is not None:
            try:
                signal.signal(s, _sig)
            except Exception:
                # Signals can only be set on the main thread; ignore otherwise.
                pass


def close_browser_gracefully() -> None:
    """Public hook for CLI/headless: close directly (the caller owns the
    context there). The GUI uses its own owning-thread flush instead."""
    _close_live("pre-restart")


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
    # Absolute, per-user profile path (LOCALAPPDATA) that does NOT depend on
    # where the exe runs from — so a zip-extraction/temp launch or a fresh
    # re-download no longer yields an empty profile every start.
    profile_path = profile_dir(account)

    # --- Startup diagnostics: prove WHERE we run and WHETHER the session is on
    #     disk. These lines let a stuck designer PC be diagnosed from the log. ---
    print(f"[agent] exe/sys.executable : {sys.executable}")
    print(f"[agent] app_dir            : {app_dir()}")
    print(f"[agent] account key        : {account!r}")
    print(f"[agent] profiles_root      : {profiles_root()}")
    print(f"[agent] using browser profile {profile_path}")
    print(f"[agent] cookies (pre-launch): {cookies_file_stat(account)}")
    if running_from_temp_location():
        print("[agent] WARNING: running from a TEMP/zip-extraction folder — the "
              "profile cannot persist. Extract the agent to a normal folder.")
    if not install_writable():
        print(f"[agent] WARNING: install dir not writable ({app_dir()}); "
              "self-update will not persist (profile is unaffected).")
    print(f"[agent] launching browser context ({_thread_diag()})")

    # Make sure the browser is flushed cleanly on any process exit.
    register_graceful_shutdown()

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
        # Register the live handles + the OWNING thread so atexit / signals can
        # flush them on the right thread even if the owner forgets to close.
        global _live_context, _live_pw, _owner_thread
        _live_context, _live_pw = context, pw
        _owner_thread = threading.current_thread()
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
                f"The ChatGPT browser profile is already open by another process.\n"
                f"Profile: {profile_path}\n"
                "A previous agent's Chrome is still running and holding this profile. "
                "Open Task Manager, end any leftover 'chrome.exe' / 'ArtworkAgent' "
                "processes (or sign out and back in to Windows), then start the agent "
                "again. The agent will NOT open a blank profile — your saved ChatGPT "
                "session is safe."
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
    t0 = time.monotonic()
    try:
        print(f"[auth] checking session: url={page.url!r}")
    except Exception:
        pass
    # Settle the page first so the check isn't racing an in-flight navigation.
    try:
        page.wait_for_load_state("domcontentloaded", timeout=15_000)
    except Exception:
        pass

    # 1) Fast, positive proof of an authenticated, ready editor.
    try:
        prompt = page.wait_for_selector(PROMPT_BOX, state="visible", timeout=10_000)
        if prompt is not None:
            print(f"[auth] prompt box visible after {time.monotonic()-t0:.1f}s "
                  "-> logged in")
            return True
    except Exception:
        pass

    # 2) Prompt box not seen. Only call it logged out if the login screen is
    #    actually showing. Give it a moment in case the auth UI is rendering.
    try:
        login = page.query_selector(LOGIN_SCREEN)
        if login is not None and login.is_visible():
            print(f"[auth] login screen detected after {time.monotonic()-t0:.1f}s "
                  "-> logged out")
            return False
    except Exception:
        pass

    # 3) Neither the editor nor the login screen resolved. Do NOT report a false
    #    logout — a restart used to 'fix' this precisely because the session was
    #    fine all along.
    print(f"[auth] after {time.monotonic()-t0:.1f}s: prompt box not visible but no "
          "login screen either -> treating session as still valid (avoiding false "
          "logout)")
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

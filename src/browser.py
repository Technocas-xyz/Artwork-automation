"""Browser lifecycle management for ChatGPT automation."""

from __future__ import annotations

from playwright.sync_api import BrowserContext, Page, sync_playwright

from config.selectors import PROMPT_BOX


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


def launch_context(account: str) -> BrowserContext:
    """Launch a persistent Chromium context for the given account.

    Uses a per-account profile directory so cookies and session state
    are preserved across runs.

    Parameters
    ----------
    account:
        Identifier used to namespace the browser profile directory.

    Returns
    -------
    BrowserContext
        A Playwright BrowserContext ready for page creation.

    Raises
    ------
    ProfileLockedError
        If the profile directory is already in use by another Chromium instance.
    """
    try:
        pw = sync_playwright().start()
        context: BrowserContext = pw.chromium.launch_persistent_context(
            user_data_dir=f"./profiles/{account}",
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        return context
    except Exception as exc:
        msg = str(exc).lower()
        if "already in use" in msg or "existing browser session" in msg or "lock" in msg:
            raise ProfileLockedError(
                f"Profile './profiles/{account}' is already in use by another Chromium instance. "
                "Close any open automation browser windows and try again."
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Session validation
# ---------------------------------------------------------------------------


def is_logged_in(page: Page) -> bool:
    """Check whether the page has an active ChatGPT session.

    Returns True if the prompt box is present and visible, indicating
    the user is authenticated and the editor is ready.
    """
    try:
        prompt = page.wait_for_selector(PROMPT_BOX, state="visible", timeout=10_000)
        return prompt is not None
    except Exception:
        return False


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

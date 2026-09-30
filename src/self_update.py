"""Agent code self-update.

Only the agent's ~1 MB of Python travels on an update — the 250 MB Chromium
runtime shipped in the first-install ZIP never changes. This module:

  * knows the LOCAL code version (a file next to the executable),
  * asks the server for its version,
  * downloads the code-only bundle, extracts it to a staging folder,
  * swaps staging over the live code directory (keeping a backup),
  * and re-execs the agent so the new code loads.

Design rules honoured here:
  * Never raises into the caller — every entry point returns a status object so
    a failed update can never leave a designer with a broken agent. On failure
    the live code directory is left exactly as it was.
  * The caller decides WHEN to apply/restart (never mid-job). This module only
    performs the mechanics and reports state.

Layout it targets:
  Frozen one-dir build:
    <app>/ArtworkAgent.exe
    <app>/code/            <- updatable .py (agent.py, agent_gui.py, src/, config/)
    <app>/code/code_version.txt
  Running from source (dev):
    the project root is the code dir; code_version.txt lives there too.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import requests

VERSION_FILENAME = "code_version.txt"
_STAGING_NAME = "code_staging"
_BACKUP_NAME = "code_backup"
# The browser profile (saved ChatGPT session) lives here, next to the exe —
# OUTSIDE code/, code_staging/, code_backup/ and _internal/. It must survive a
# self-update, so the swap must never see it.
PROFILES_NAME = "profiles"
# What a bundle is allowed to contain / what we swap. Anything else next to the
# code dir (a local .env, and crucially the browser profile) is left untouched.
_MANAGED_TOPLEVEL = ("agent.py", "agent_gui.py", "src", "config")

# Hard guard: the profile folder must never be swappable. If someone ever adds
# "profiles" to the managed list, fail loudly at import rather than silently
# wiping designers' saved sessions on the next update.
assert PROFILES_NAME not in _MANAGED_TOPLEVEL, "profiles must never be self-update-managed"


class UpdateStatus:
    """Plain result object the GUI/agent can read without catching exceptions."""
    def __init__(self):
        self.local_version: str | None = None
        self.server_version: str | None = None
        self.update_available: bool = False
        self.staged: bool = False          # a newer bundle is downloaded + ready to apply
        self.applied: bool = False         # swap done this call
        self.error: str | None = None      # human-readable; set on any failure
        # Classifies the check outcome so the UI can present it correctly
        # WITHOUT string-matching the error text:
        #   "ok"            — check/stage/apply succeeded
        #   "up_to_date"    — reached the server, nothing newer
        #   "not_supported" — server has no self-update endpoints (404); say nothing
        #   "unreachable"   — could not reach the server (connection/timeout); muted note
        #   "error"         — a real failure (bad download, apply failed); worth showing
        self.reason: str = "ok"
        self.checked_at: float = time.time()

    def as_dict(self) -> dict:
        return {
            "local_version": self.local_version,
            "server_version": self.server_version,
            "update_available": self.update_available,
            "staged": self.staged,
            "applied": self.applied,
            "error": self.error,
            "reason": self.reason,
            "checked_at": self.checked_at,
        }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Directory that holds the executable (frozen) or the project root (source)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    # src/self_update.py -> project root is two parents up.
    return Path(__file__).resolve().parent.parent


def code_dir() -> Path:
    """The live, updatable code directory.

    Frozen: a writable <app>/code folder (created if missing). Source: the
    project root itself."""
    if is_frozen():
        d = app_dir() / "code"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return app_dir()


def updates_allowed() -> tuple[bool, str]:
    """Whether self-update may touch the live code, with a reason when it may not.

    Self-update swaps whole top-level entries (agent.py, agent_gui.py, src/,
    config/) under code_dir(). In a PACKAGED install code_dir() is a dedicated
    <app>/code folder that holds nothing but the updatable bundle, so swapping it
    is safe. In a SOURCE checkout code_dir() IS the project root — the developer's
    working tree, git repo and all — and swapping src/ and config/ there silently
    destroys uncommitted work (this is exactly what wiped generator.py and
    workflows.py: an older staged bundle replaced the whole src/ and config/
    directories in the checkout).

    So updates are allowed ONLY when frozen. When running from source we refuse
    and say why; the caller logs it and leaves the working tree untouched."""
    if not is_frozen():
        return False, ("running from source (code_dir resolves to the project "
                       "checkout); self-update only applies to a packaged install")
    return True, ""


def profiles_root() -> Path:
    """Fixed, writable root for browser profiles, as an ABSOLUTE path derived
    from the app directory — never relative to the current working directory.

    It sits next to the executable (frozen) or the project root (source),
    OUTSIDE code/ and _internal/, so a self-update swap can never take the saved
    ChatGPT session with it."""
    return (app_dir() / PROFILES_NAME).resolve()


def profile_dir(account: str) -> Path:
    """Absolute per-account browser profile directory. Created if missing."""
    safe = "".join(c for c in (account or "acct1") if c.isalnum() or c in ("-", "_")) or "acct1"
    d = profiles_root() / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Logging — a self-update that fails on a packaged agent must leave a trace a
# designer (who has no terminal) can read. Every step prints AND appends to
# <app>/update.log so a stuck update can be diagnosed after the fact.
# ---------------------------------------------------------------------------

def _log_file() -> Path:
    return app_dir() / "update.log"


def log(msg: str) -> None:
    line = f"[update] {msg}"
    try:
        print(line)
    except Exception:
        pass
    try:
        with open(_log_file(), "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    except Exception:
        pass


def _version_file() -> Path:
    return code_dir() / VERSION_FILENAME


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def read_local_version(fallback: str | None = None) -> str:
    """Version recorded in the code dir. Falls back to the bundled
    config.agent_version (or `fallback`) when the file is absent — e.g. a fresh
    install that has never updated."""
    vf = _version_file()
    try:
        if vf.is_file():
            v = vf.read_text(encoding="utf-8").strip()
            if v:
                return v
    except Exception:
        pass
    try:
        from config.agent_version import AGENT_CODE_VERSION
        return AGENT_CODE_VERSION
    except Exception:
        return fallback or "0.0.0"


def write_local_version(version: str) -> None:
    try:
        _version_file().write_text(str(version).strip(), encoding="utf-8")
    except Exception:
        pass


def _version_tuple(v: str) -> tuple:
    parts = []
    for chunk in str(v).strip().split("."):
        num = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def is_newer(server_version: str, local_version: str) -> bool:
    """True if server_version is strictly newer than local_version.

    Prefers packaging's parser when available; falls back to a numeric tuple
    compare so a missing dependency never blocks an update decision."""
    sv, lv = (server_version or "").strip(), (local_version or "").strip()
    if not sv:
        return False
    if sv == lv:
        return False
    try:
        from packaging.version import Version
        return Version(sv) > Version(lv)
    except Exception:
        return _version_tuple(sv) > _version_tuple(lv)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def fetch_server_version(server_url: str, token: str, timeout: float = 15.0):
    """Return (version, reason, detail).

    reason is one of:
      "ok"            — got a version string
      "not_supported" — server returned 404 (no self-update endpoint yet)
      "unreachable"   — could not connect / timed out
      "error"         — reached the server but got another error status/body
    Never raises."""
    try:
        r = requests.get(server_url.rstrip("/") + "/api/agent/code-version",
                         headers=_headers(token), timeout=timeout)
    except requests.exceptions.RequestException as exc:
        # Connection refused, DNS failure, timeout — the server is unreachable.
        return None, "unreachable", str(exc)
    except Exception as exc:
        return None, "unreachable", str(exc)

    if r.status_code == 404:
        # An older server that predates self-update. Not an error.
        return None, "not_supported", None
    if not r.ok:
        return None, "error", f"HTTP {r.status_code}"
    try:
        return (r.json() or {}).get("version"), "ok", None
    except Exception as exc:
        return None, "error", f"bad version response: {exc}"


def _download_bundle(server_url: str, token: str, dest_zip: Path, timeout: float = 120.0) -> None:
    r = requests.get(server_url.rstrip("/") + "/api/agent/code-bundle",
                     headers=_headers(token), timeout=timeout, stream=True)
    r.raise_for_status()
    with open(dest_zip, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                fh.write(chunk)


# ---------------------------------------------------------------------------
# Staging + swap
# ---------------------------------------------------------------------------

def _validate_bundle(staging: Path) -> None:
    """Refuse to apply a bundle that is missing the essentials — a truncated or
    wrong download must never overwrite good code."""
    missing = []
    for essential in ("agent.py", "agent_gui.py"):
        if not (staging / essential).is_file():
            missing.append(essential)
    for d in ("src", "config"):
        if not (staging / d).is_dir():
            missing.append(d + "/")
    if missing:
        # List what IS there, to make a layout mismatch obvious in the log.
        present = sorted(p.name for p in staging.iterdir()) if staging.is_dir() else []
        raise RuntimeError(
            f"bundle missing {missing}; refusing to apply. staging contains: {present}")


def stage_update(server_url: str, token: str, version: str) -> Path:
    """Download + extract the bundle into <app>/code_staging. Returns the path.
    Raises on any failure; leaves nothing partial in the live code dir."""
    staging = app_dir() / _STAGING_NAME
    # Clean any prior staging first.
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "agent_code.zip"
        _download_bundle(server_url, token, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            # Guard against path traversal in a malicious/corrupt zip.
            for name in zf.namelist():
                target = (staging / name).resolve()
                if not str(target).startswith(str(staging.resolve())):
                    raise RuntimeError(f"unsafe path in bundle: {name}")
            zf.extractall(staging)

    _validate_bundle(staging)
    write_version_into(staging, version)
    return staging


def write_version_into(folder: Path, version: str) -> None:
    try:
        (folder / VERSION_FILENAME).write_text(str(version).strip(), encoding="utf-8")
    except Exception:
        pass


def _force_rmtree(path: Path) -> None:
    """rmtree that copes with Windows read-only bits on git/pyc files."""
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, 0o777)
            func(p)
        except Exception:
            pass
    if path.exists():
        shutil.rmtree(path, onerror=_onerror)


def apply_staged(staging: Path | None = None) -> None:
    """Swap the staged code over the live code dir, keeping a backup.

    Windows-safe: the live process imports agent/agent_gui/src/config FROM the
    code dir, so those files can be locked/in-use. We therefore swap by RENAME
    (move the old entry aside, move the new one in) instead of deleting the live
    tree in place — a rename of a directory succeeds even when its files are
    open for reading, whereas rmtree-in-place fails with PermissionError. That
    silent PermissionError is what left agents stuck on baseline code.

    Every step is logged (and written to <app>/update.log). On any failure the
    backup is restored and the reason is raised so the GUI can surface it.
    Only the managed top-level entries are touched; profiles/, .env, etc. are
    left alone. The version file is written LAST, only after a clean swap, so a
    present code/code_version.txt reliably means "fully applied"."""
    # HARD BACKSTOP: never swap code over a source checkout. code_dir() there is
    # the project root, so applying would rename src/ and config/ out of the
    # working tree (destroying uncommitted edits). Refuse before touching a
    # single file, whatever left a staging dir behind.
    ok, why = updates_allowed()
    if not ok:
        log(f"apply: REFUSED — {why}; leaving code and staging untouched")
        raise RuntimeError(f"self-update refused: {why}")

    staging = staging or (app_dir() / _STAGING_NAME)
    if not staging.is_dir():
        raise RuntimeError(f"no staged update to apply at {staging}")
    log(f"apply: staging found at {staging}")
    _validate_bundle(staging)
    log("apply: bundle validated (agent.py, agent_gui.py, src/, config/ present)")

    live = code_dir()
    live.mkdir(parents=True, exist_ok=True)
    backup = app_dir() / _BACKUP_NAME
    _force_rmtree(backup)
    backup.mkdir(parents=True, exist_ok=True)
    log(f"apply: backup dir ready at {backup}")

    staged_ver = ""
    vf = staging / VERSION_FILENAME
    if vf.is_file():
        try:
            staged_ver = vf.read_text(encoding="utf-8").strip()
        except Exception:
            staged_ver = ""

    moved: list[str] = []   # names whose live copy we renamed into backup
    placed: list[str] = []  # names whose staged copy we moved into live
    try:
        for name in _MANAGED_TOPLEVEL:
            src = staging / name
            if not src.exists():
                log(f"apply: staging has no '{name}', skipping")
                continue
            dst = live / name
            # 1) Move the current live entry aside into backup (rename = atomic,
            #    survives files being open for reading on Windows).
            if dst.exists():
                os.replace(str(dst), str(backup / name)) if dst.is_file() else _move_dir(dst, backup / name)
                moved.append(name)
                log(f"apply: moved existing code/{name} -> backup")
            # 2) Move the staged entry into place.
            _move_dir(src, dst) if src.is_dir() else os.replace(str(src), str(dst))
            placed.append(name)
            log(f"apply: placed new code/{name}")

        log(f"apply: swap finished ({len(placed)} entr{'y' if len(placed)==1 else 'ies'}: {placed})")

        # 3) Write the version LAST — this is what the next check compares.
        if staged_ver:
            (live / VERSION_FILENAME).write_text(staged_ver, encoding="utf-8")
            log(f"apply: wrote code/{VERSION_FILENAME} = {staged_ver}")
        else:
            log("apply: WARNING staging had no version file; code version not updated")
    except Exception as exc:
        log(f"apply: FAILED at swap: {exc}\n{traceback.format_exc()}")
        # Roll back: remove anything we placed, restore anything we moved aside.
        for name in placed:
            _force_rmtree(live / name) if (live / name).is_dir() else _silent_unlink(live / name)
        for name in moved:
            b = backup / name
            if b.exists():
                try:
                    _move_dir(b, live / name) if b.is_dir() else os.replace(str(b), str(live / name))
                except Exception as rexc:
                    log(f"apply: rollback of {name} failed: {rexc}")
        log("apply: rolled back to previous code")
        raise RuntimeError(f"apply failed, rolled back to previous code: {exc}")
    finally:
        _force_rmtree(staging)
        log("apply: cleared staging")


def _move_dir(src: Path, dst: Path) -> None:
    """Move directory `src` onto `dst`, replacing dst if present.

    Prefers an atomic same-volume rename; on any failure (e.g. cross-volume, or
    a locked entry) falls back to copytree + best-effort rmtree. Always ensures
    dst does not exist before creating it, so a failed rename can't leave a
    half-state that makes the fallback raise FileExistsError."""
    if dst.exists():
        _force_rmtree(dst)
    try:
        os.replace(str(src), str(dst))   # atomic on same volume
        return
    except OSError:
        pass
    # Fallback: copy the tree in, then remove the source. dst was cleared above;
    # copytree with dirs_exist_ok tolerates any residue from a partial rename.
    shutil.copytree(src, dst, dirs_exist_ok=True)
    _force_rmtree(src)


def _silent_unlink(p: Path) -> None:
    try:
        p.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def check(server_url: str, token: str) -> UpdateStatus:
    """Compare local vs server version. Never raises.

    Classifies the outcome via st.reason so the UI can stay quiet for a server
    that simply doesn't support self-update (404), show a muted note when the
    server is merely unreachable, and reserve errors for real problems."""
    st = UpdateStatus()
    st.local_version = read_local_version()
    version, reason, detail = fetch_server_version(server_url, token)
    st.server_version = version
    st.reason = reason
    if reason == "not_supported":
        # Older server without the endpoints — nothing to say, not an error.
        return st
    if reason == "unreachable":
        # The agent still works; only the check failed. Muted note, not red.
        st.error = "could not reach the server to check for updates"
        return st
    if reason == "error":
        st.error = f"update check failed ({detail})"
        return st
    # reason == "ok"
    st.update_available = is_newer(version or "", st.local_version)
    if not st.update_available:
        st.reason = "up_to_date"
    return st


def check_and_stage(server_url: str, token: str) -> UpdateStatus:
    """Check and, if newer, download+stage the bundle ready to apply. Never
    raises; a failure leaves the current code untouched and records the error."""
    # In a source checkout there is nothing to safely update — refuse to even
    # download a bundle, so a dev machine never accumulates a code_staging/ that
    # a later apply could try to swap over the working tree.
    ok, why = updates_allowed()
    if not ok:
        st = check(server_url, token)
        st.staged = False
        st.reason = "not_supported"
        log(f"check_and_stage: skipped — {why}")
        return st
    st = check(server_url, token)
    if st.error or not st.update_available:
        return st
    try:
        log(f"check_and_stage: server {st.server_version} > local {st.local_version}; downloading bundle")
        stage_update(server_url, token, st.server_version)
        st.staged = True
        st.reason = "ok"
        log(f"check_and_stage: staged {st.server_version} into {app_dir() / _STAGING_NAME}")
    except Exception as exc:
        st.staged = False
        # A download/stage failure IS a real problem worth reporting.
        st.reason = "error"
        st.error = f"download/stage failed: {exc}"
        log(f"check_and_stage: {st.error}")
    return st


def apply_if_staged() -> UpdateStatus:
    """Apply a previously staged update. Never raises."""
    st = UpdateStatus()
    st.local_version = read_local_version()
    # Never apply over a source checkout (would clobber the working tree). Log
    # once if a stale staging dir is present so it is visible, then no-op.
    ok, why = updates_allowed()
    if not ok:
        if (app_dir() / _STAGING_NAME).is_dir():
            log(f"apply_if_staged: staged update present but REFUSED — {why}; "
                f"leaving the checkout untouched (delete {_STAGING_NAME}/ to silence)")
        return st
    staging = app_dir() / _STAGING_NAME
    if not staging.is_dir():
        return st  # nothing staged; no-op
    log(f"apply_if_staged: staged update present (local version {st.local_version}); applying")
    try:
        apply_staged(staging)
        st.applied = True
        st.local_version = read_local_version()
        st.reason = "ok"
        log(f"apply_if_staged: applied OK; code version now {st.local_version}")
    except Exception as exc:
        st.applied = False
        st.reason = "error"
        st.error = str(exc)
        log(f"apply_if_staged: apply failed: {exc}")
    return st


def has_staged_update() -> bool:
    return (app_dir() / _STAGING_NAME).is_dir()


def restart() -> None:
    """Re-exec the agent so freshly-swapped code loads. Best-effort.

    Frozen: relaunch the same executable with the same args. Source: re-exec the
    Python interpreter on the same argv. The caller should have already ensured
    no job is running."""
    try:
        if is_frozen():
            exe = sys.executable
            os.execv(exe, [exe] + sys.argv[1:])
        else:
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        # If exec fails we simply keep running the (already-swapped) code until
        # the next manual restart — nothing is broken, just not yet reloaded.
        pass

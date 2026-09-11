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
import zipfile
from pathlib import Path

import requests

VERSION_FILENAME = "code_version.txt"
_STAGING_NAME = "code_staging"
_BACKUP_NAME = "code_backup"
# What a bundle is allowed to contain / what we swap. Anything else in the live
# code dir (e.g. a local .env) is left untouched by the swap.
_MANAGED_TOPLEVEL = ("agent.py", "agent_gui.py", "src", "config")


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
    for essential in ("agent.py", "agent_gui.py"):
        if not (staging / essential).is_file():
            raise RuntimeError(f"bundle missing {essential}; refusing to apply")
    if not (staging / "src").is_dir() or not (staging / "config").is_dir():
        raise RuntimeError("bundle missing src/ or config/; refusing to apply")


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


def apply_staged(staging: Path | None = None) -> None:
    """Swap the staged code over the live code dir, keeping a backup.

    Only the managed top-level entries are replaced; unrelated files in the code
    dir (local config, the version file we then rewrite) are preserved. On any
    error the backup is restored so the agent keeps its old, working code."""
    staging = staging or (app_dir() / _STAGING_NAME)
    if not staging.is_dir():
        raise RuntimeError("no staged update to apply")
    _validate_bundle(staging)

    live = code_dir()
    backup = app_dir() / _BACKUP_NAME
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir(parents=True, exist_ok=True)

    replaced: list[str] = []
    try:
        for name in _MANAGED_TOPLEVEL:
            src = staging / name
            if not src.exists():
                continue
            dst = live / name
            # Back up the current version of this entry.
            if dst.exists():
                bdst = backup / name
                if dst.is_dir():
                    shutil.copytree(dst, bdst)
                else:
                    shutil.copy2(dst, bdst)
                # Remove the live copy before writing the new one.
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            # Move the staged entry into place.
            shutil.move(str(src), str(dst))
            replaced.append(name)
        # Record the new version alongside the code.
        staged_ver = (staging / VERSION_FILENAME)
        if staged_ver.is_file():
            (live / VERSION_FILENAME).write_text(
                staged_ver.read_text(encoding="utf-8").strip(), encoding="utf-8")
    except Exception as exc:
        # Roll back everything we touched from the backup, best-effort.
        for name in replaced:
            try:
                dst = live / name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst, ignore_errors=True)
                    else:
                        dst.unlink()
                bsrc = backup / name
                if bsrc.exists():
                    if bsrc.is_dir():
                        shutil.copytree(bsrc, dst)
                    else:
                        shutil.copy2(bsrc, dst)
            except Exception:
                pass
        raise RuntimeError(f"apply failed, rolled back to previous code: {exc}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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
    st = check(server_url, token)
    if st.error or not st.update_available:
        return st
    try:
        stage_update(server_url, token, st.server_version)
        st.staged = True
        st.reason = "ok"
    except Exception as exc:
        st.staged = False
        # A download/stage failure IS a real problem worth reporting.
        st.reason = "error"
        st.error = f"download/stage failed: {exc}"
    return st


def apply_if_staged() -> UpdateStatus:
    """Apply a previously staged update. Never raises."""
    st = UpdateStatus()
    st.local_version = read_local_version()
    staging = app_dir() / _STAGING_NAME
    if not staging.is_dir():
        return st  # nothing staged; no-op
    try:
        apply_staged(staging)
        st.applied = True
        st.local_version = read_local_version()
    except Exception as exc:
        st.applied = False
        st.error = str(exc)
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

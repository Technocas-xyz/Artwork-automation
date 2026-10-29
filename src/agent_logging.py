"""Agent-side logging: a persistent, rotating log beside the agent plus a small
errors-only log a designer can send, an in-memory errors ring the GUI shows, and
unhandled-exception hooks for every thread.

Design goals (kept deliberately small and crash-proof — logging must never break
the agent):

  * Log dir resolution (resolve_log_dir):
      - frozen: <app>/logs beside ArtworkAgent.exe,
      - source: <repo>/logs,
      - if that is NOT writable OR the exe runs from a temp/zip folder:
        %LOCALAPPDATA%\\ArtworkAgent\\logs, and we SAY SO at the top of the log.
  * Handlers on the ROOT logger:
      - agent.log   RotatingFileHandler 5 MB x5, everything at INFO+,
      - errors.log  RotatingFileHandler 5 MB x5, WARNING+ only.
    Format: time, level, thread, module, message.
  * SecretRedactingFilter on every handler (token / password / cookie /
    Authorization), reusing agent_diagnostics._redact so there is one rule set.
  * route_print(): wrap stdout/stderr so plain print() output is ALSO emitted
    through logging (nothing is lost), while still writing to the real console.
  * An in-memory ring of the last few errors (time, job id, one-line message,
    full traceback) for the agent window.
  * install_excepthooks(): sys.excepthook + threading.excepthook log the full
    traceback of anything that escapes.

Nothing here raises into the caller.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Reuse the ONE secret-redaction rule set from agent_diagnostics so agent.log,
# errors.log and the diagnostics log all scrub identically. Fall back to a
# no-op only if that module cannot be imported (never let logging setup fail).
try:
    from src.agent_diagnostics import _redact as _diag_redact
except Exception:  # pragma: no cover
    def _diag_redact(text):  # type: ignore
        return text

_MAX_BYTES = 5 * 1024 * 1024    # 5 MB per file
_BACKUP_COUNT = 5               # keep 5 rotated files
_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(module)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_setup_done = False
_log_dir: Path | None = None
_fallback_note = ""             # non-empty when we had to use the LOCALAPPDATA fallback

# Ring of the last errors for the GUI. Each entry: {time, job_id, message, traceback}.
_ERRORS_MAX = 20                # keep a few more than the 5 shown, cheaply
_errors_lock = threading.Lock()
_errors: "deque[dict]" = deque(maxlen=_ERRORS_MAX)


# ---------------------------------------------------------------------------
# Log-directory resolution
# ---------------------------------------------------------------------------

def _localappdata_logs() -> Path:
    base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home())
    return Path(base) / "ArtworkAgent" / "logs"


def _dir_writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def resolve_log_dir() -> tuple[Path, str]:
    """Return (log_dir, note). The note is non-empty only when we fell back to
    the per-user LOCALAPPDATA location, explaining why (so it can be written at
    the top of the log)."""
    # Preferred: <app>/logs (beside the exe when frozen, repo root from source).
    try:
        from src.self_update import app_dir, is_frozen, running_from_temp_location
        preferred = app_dir() / "logs"
        temp_run = False
        try:
            temp_run = bool(is_frozen() and running_from_temp_location())
        except Exception:
            temp_run = False
    except Exception:
        # self_update not importable — derive the project root ourselves.
        if getattr(sys, "frozen", False):
            preferred = Path(sys.executable).resolve().parent / "logs"
        else:
            preferred = Path(__file__).resolve().parent.parent / "logs"
        temp_run = False

    if temp_run:
        fb = _localappdata_logs()
        if _dir_writable(fb):
            return fb, (f"Running from a temporary/zip folder ({preferred.parent}); "
                        f"logging to the per-user location instead: {fb}")
    if _dir_writable(preferred):
        return preferred, ""
    # Preferred not writable (read-only install) — fall back to LOCALAPPDATA.
    fb = _localappdata_logs()
    if _dir_writable(fb):
        return fb, (f"Log folder beside the agent was not writable ({preferred}); "
                    f"logging to the per-user location instead: {fb}")
    # Last resort: give back the preferred path even if un-writable; handlers
    # will simply fail to open and we degrade to console only.
    return preferred, ""


def log_dir() -> Path:
    """The resolved log directory (resolving once on first call)."""
    global _log_dir
    if _log_dir is None:
        _log_dir, _ = resolve_log_dir()
    return _log_dir


# ---------------------------------------------------------------------------
# Secret-redacting filter
# ---------------------------------------------------------------------------

class SecretRedactingFilter(logging.Filter):
    """Scrub secrets (agent token, passwords, cookies, Authorization headers)
    from every record before it is written. Covers BOTH the final message (so
    interpolated args are included) AND any attached traceback (exc_info), which
    is where a secret in an exception message would otherwise leak into the log
    file unredacted."""
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            red = _redact(msg)
            if red != msg:
                record.msg = red
                record.args = ()
        except Exception:
            pass
        # Redact the traceback: pre-format exc_info into a redacted exc_text and
        # drop exc_info so the formatter uses our scrubbed text, not the raw one.
        try:
            if record.exc_info:
                tb = "".join(traceback.format_exception(*record.exc_info))
                record.exc_text = _redact(tb)
                record.exc_info = None
            elif record.exc_text:
                record.exc_text = _redact(record.exc_text)
        except Exception:
            pass
        return True


def _redact(text: str) -> str:
    """Redaction = the diagnostics rule set PLUS a cookie rule (diag already
    covers token/password/authorization/secret/api-key/bearer)."""
    try:
        s = _diag_redact(text)
    except Exception:
        s = text
    try:
        import re
        # Cookie: <...> and Set-Cookie: <...> header lines.
        s = re.sub(r"(?i)(set-)?(cookie\s*[:=]\s*).+", r"\1\2***REDACTED***", s)
    except Exception:
        pass
    return s


# ---------------------------------------------------------------------------
# Errors ring (for the GUI) — fed by a WARNING+ handler
# ---------------------------------------------------------------------------

# Set by the agent so ring entries can be tagged with the job currently running.
_current_job_getter = None   # callable() -> str (job id) or None


def set_current_job_getter(fn) -> None:
    """Register a callable returning the id of the job currently running (or '')
    so error-ring entries can be tagged with it. Optional."""
    global _current_job_getter
    _current_job_getter = fn


def _current_job_id() -> str:
    try:
        return str(_current_job_getter() or "") if _current_job_getter else ""
    except Exception:
        return ""


class _ErrorRingHandler(logging.Handler):
    """Captures WARNING+ records into the in-memory ring for the agent window.
    Stores a one-line message and the full traceback (when present)."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = _redact(record.getMessage())
            one_line = msg.splitlines()[0] if msg else ""
            # Prefer live exc_info; fall back to exc_text (the SecretRedacting
            # filter on an earlier handler may have converted exc_info -> exc_text).
            tb = ""
            if record.exc_info:
                tb = _redact("".join(traceback.format_exception(*record.exc_info)))
            elif getattr(record, "exc_text", None):
                tb = _redact(record.exc_text)
            full = tb or msg
            jid = _current_job_id()
            entry = {
                "time": time.strftime("%H:%M:%S"),
                "job_id": jid,
                "level": record.levelname,
                "message": one_line[:200],
                "traceback": full,
            }
            with _errors_lock:
                _errors.append(entry)
        except Exception:
            pass


def last_errors(n: int = 5) -> list[dict]:
    """The last `n` error/warning entries (newest last), for the GUI."""
    with _errors_lock:
        items = list(_errors)
    return items[-n:]


def last_error_full() -> str:
    """The full traceback/text of the most recent error, for 'Copy last error'."""
    with _errors_lock:
        if not _errors:
            return ""
        return _errors[-1].get("traceback") or _errors[-1].get("message") or ""


def record_error(message: str, job_id: str = "", tb: str = "") -> None:
    """Add an entry to the ring directly (for callers that already have a
    message/traceback and want it shown without going through logging)."""
    try:
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "job_id": job_id or _current_job_id(),
            "level": "ERROR",
            "message": _redact(str(message)).splitlines()[0][:200] if message else "",
            "traceback": _redact(tb or str(message)),
        }
        with _errors_lock:
            _errors.append(entry)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# stdout/stderr tee -> logging
# ---------------------------------------------------------------------------

class _LogTee:
    """Wrap a stream so each printed line is written to the real stream AND
    emitted through logging (so plain print() output is never lost from the log
    files). Buffers partial writes until a newline. Never raises."""
    def __init__(self, stream, level: int):
        self._stream = stream
        self._level = level
        self._buf = ""
        self._logger = logging.getLogger("agent.print")

    def write(self, s):
        try:
            if self._stream is not None:
                self._stream.write(s)
        except Exception:
            pass
        try:
            self._buf += str(s)
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.rstrip("\r")
                if line.strip():
                    # stderr -> ERROR so it also reaches errors.log.
                    self._logger.log(self._level, "%s", line)
        except Exception:
            pass

    def flush(self):
        try:
            if self._stream is not None:
                self._stream.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return bool(self._stream and self._stream.isatty())
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(route_print: bool = True) -> Path:
    """Install the rotating file handlers + errors ring + (optionally) the
    print tee. Idempotent. Returns the resolved log directory. Never raises."""
    global _setup_done, _log_dir, _fallback_note
    if _setup_done:
        return log_dir()
    _setup_done = True

    try:
        _log_dir, _fallback_note = resolve_log_dir()
    except Exception:
        _log_dir, _fallback_note = (_localappdata_logs(), "")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Quiet the noisiest third parties so the files stay readable.
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "asyncio", "playwright"):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass

    redactor = SecretRedactingFilter()
    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    def _rfh(name: str, level: int):
        try:
            h = RotatingFileHandler(str(_log_dir / name), maxBytes=_MAX_BYTES,
                                    backupCount=_BACKUP_COUNT, encoding="utf-8")
            h.setLevel(level)
            h.setFormatter(fmt)
            h.addFilter(redactor)
            # Avoid duplicate handlers across a re-exec within one process.
            for existing in list(root.handlers):
                if isinstance(existing, RotatingFileHandler) and \
                   name in getattr(existing, "baseFilename", ""):
                    return
            root.addHandler(h)
        except Exception:
            pass

    _rfh("agent.log", logging.INFO)
    _rfh("errors.log", logging.WARNING)

    # Errors ring for the GUI (WARNING+).
    ring = _ErrorRingHandler()
    ring.setLevel(logging.WARNING)
    ring.addFilter(redactor)
    if not any(isinstance(h, _ErrorRingHandler) for h in root.handlers):
        root.addHandler(ring)

    # A console handler so logging.* still shows on the terminal when running
    # from source (print() already shows via the real stream).
    try:
        if not any(isinstance(h, logging.StreamHandler)
                   and not isinstance(h, RotatingFileHandler) for h in root.handlers):
            ch = logging.StreamHandler(stream=sys.__stdout__)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter("%(message)s"))
            ch.addFilter(redactor)
            root.addHandler(ch)
    except Exception:
        pass

    # Header line(s) at the top of this session's log, including the fallback
    # note when the beside-the-exe location could not be used.
    lg = logging.getLogger("agent.logging")
    try:
        lg.info("==== agent logging started; log dir = %s ====", _log_dir)
        if _fallback_note:
            lg.warning("log location note: %s", _fallback_note)
    except Exception:
        pass

    # Point agent_diagnostics' errors log at the SAME folder so everything is in
    # one place (best-effort; diagnostics keeps working regardless).
    try:
        from src import agent_diagnostics as _diag
        if hasattr(_diag, "set_log_dir"):
            _diag.set_log_dir(_log_dir)
    except Exception:
        pass

    if route_print:
        route_print_to_logging()

    return _log_dir


def route_print_to_logging() -> None:
    """Wrap stdout/stderr so print() output is also logged. Safe to call once."""
    try:
        if not isinstance(sys.stdout, _LogTee):
            sys.stdout = _LogTee(sys.stdout, logging.INFO)
        if not isinstance(sys.stderr, _LogTee):
            sys.stderr = _LogTee(sys.stderr, logging.ERROR)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Unhandled-exception hooks (main thread + worker threads)
# ---------------------------------------------------------------------------

def install_excepthooks() -> None:
    """Log the full traceback of any exception that escapes a thread. Never
    swallows the default behaviour entirely — chains to the previous hook."""
    lg = logging.getLogger("agent.unhandled")

    prev_excepthook = sys.excepthook

    def _hook(exc_type, exc, tb):
        try:
            if issubclass(exc_type, KeyboardInterrupt):
                return prev_excepthook(exc_type, exc, tb)
            lg.error("Unhandled exception on main thread:", exc_info=(exc_type, exc, tb))
        except Exception:
            pass
        try:
            prev_excepthook(exc_type, exc, tb)
        except Exception:
            pass

    sys.excepthook = _hook

    # threading.excepthook (Python 3.8+): fires for exceptions in any thread.
    try:
        prev_thook = threading.excepthook

        def _thook(args):
            try:
                lg.error("Unhandled exception on thread %r:", getattr(args, "thread", None),
                         exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
            except Exception:
                pass
            try:
                prev_thook(args)
            except Exception:
                pass

        threading.excepthook = _thook
    except Exception:
        pass

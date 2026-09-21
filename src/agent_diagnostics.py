"""Agent diagnostics — complete error, timeout and stall logging.

Every error, timeout, stall, retry, fallback, failed session check and
unconfirmed upload is recorded here so we can see exactly where a job gets stuck
on any designer's PC. Each entry is:

  * appended to a ROTATING local file (agent_errors.log, 5 MB x 3) that lives
    NEXT TO THE EXECUTABLE — outside code/, so a self-update never wipes it, and
  * uploaded to the server as it happens, tagged with the user + agent. If the
    server is unreachable the entry is kept in a local offline queue and sent
    once the server comes back, so a dropped network never loses an entry.

Secret safety: entries are scrubbed for anything that looks like a password,
token, bearer header or a line from a .env file before they are written or sent.
Nothing here ever reads or logs the .env contents.

This module is deliberately import-light and never raises into its caller —
diagnostics must not themselves break a job.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import traceback as _tb
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Where the log lives — next to the exe / project root, OUTSIDE code/.
# ---------------------------------------------------------------------------

def _app_dir() -> Path:
    """Directory holding the executable (frozen) or the project root (source).

    Mirrors self_update.app_dir so the log sits beside update.log and the
    profiles/ folder — all of which survive a self-update swap."""
    try:
        from src.self_update import app_dir
        return app_dir()
    except Exception:
        import sys
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent.parent


LOG_NAME = "agent_errors.log"
_QUEUE_NAME = "agent_errors_queue.jsonl"   # offline upload queue (beside the log)
_MAX_BYTES = 5 * 1024 * 1024               # 5 MB per file
_BACKUP_COUNT = 3                          # keep the last 3 rotated files

_lock = threading.Lock()
_logger: logging.Logger | None = None

# Connection details + poster, injected by the agent once it knows them. The
# poster is a callable(path, json=...) -> response-like with .status_code; we
# keep it injected rather than importing agent.py to avoid a circular import.
_agent_ctx: dict = {"username": "", "agent_name": "", "agent_id": "", "code_version": "", "post": None}


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

# Redact anything that looks like a secret. Broad on purpose: better to over-mask
# than to leak a token into a log a human will read.
_SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?\S+"),
    re.compile(r"(?i)(password[\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"),
    re.compile(r"(?i)(passwd[\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"),
    re.compile(r"(?i)(token[\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"),
    re.compile(r"(?i)(secret[\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"),
    re.compile(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"),
    re.compile(r"\bagt_[A-Za-z0-9_\-]{8,}"),          # our agent token shape
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}"),
]
# Env keys whose values must never appear. If a line looks like KEY=value for
# one of these, the value is masked.
_SECRET_ENV_KEYS = ("APP_PASSWORD_HASH", "APP_SECRET_KEY", "AGENT_TOKEN",
                     "PASSWORD", "PASSWORD_OBF", "DECOINKS", "API_KEY", "SECRET")


def _redact(text: str) -> str:
    if not text:
        return text
    s = str(text)
    for pat in _SECRET_PATTERNS:
        s = pat.sub(lambda m: (m.group(1) + "***REDACTED***"), s)
    # KEY=value / KEY: value for known-secret env keys.
    for key in _SECRET_ENV_KEYS:
        s = re.sub(rf"(?im)^(\s*{re.escape(key)}\s*[:=]\s*).+$",
                   r"\1***REDACTED***", s)
    return s


def _scrub_entry(entry: dict) -> dict:
    """Redact secrets from the free-text fields of an entry (never the keys)."""
    for k in ("message", "traceback", "step", "detail"):
        if entry.get(k):
            entry[k] = _redact(entry[k])
    return entry


# ---------------------------------------------------------------------------
# Local rotating log
# ---------------------------------------------------------------------------

def log_path() -> Path:
    return _app_dir() / LOG_NAME


def _queue_path() -> Path:
    return _app_dir() / _QUEUE_NAME


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    lg = logging.getLogger("agent_diagnostics.file")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    # Close any handler left from a previous configuration so we never hold two
    # open handles on the same file (which would break rotation on Windows).
    for h in list(lg.handlers):
        try:
            h.close()
        except Exception:
            pass
        lg.removeHandler(h)
    try:
        h = RotatingFileHandler(str(log_path()), maxBytes=_MAX_BYTES,
                                backupCount=_BACKUP_COUNT, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(h)
    except Exception:
        # If the file cannot be opened we still keep going (entries go to the
        # queue / stdout); a diagnostics failure must not break a job.
        pass
    _logger = lg
    return lg


# ---------------------------------------------------------------------------
# Setup + context
# ---------------------------------------------------------------------------

def configure(username: str = "", agent_name: str = "", agent_id: str = "",
              code_version: str = "", post=None) -> None:
    """Tell diagnostics who this agent is and how to reach the server.

    `post` is a callable(path, json=dict) -> response (with .status_code and
    .ok). Injected by agent.py so we reuse its authenticated _post without an
    import cycle."""
    if username is not None:
        _agent_ctx["username"] = username or _agent_ctx["username"]
    if agent_name:
        _agent_ctx["agent_name"] = agent_name
    if agent_id:
        _agent_ctx["agent_id"] = agent_id
    if code_version:
        _agent_ctx["code_version"] = code_version
    if post is not None:
        _agent_ctx["post"] = post


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_event(kind: str, message: str = "", *, job: dict | None = None,
                 job_id: str = "", workflow: str = "", step: str = "",
                 exc: BaseException | None = None, traceback: str = "",
                 detail: str = "") -> dict:
    """Record one diagnostic entry. `kind` is a short tag:
      ERROR | TIMEOUT | STALL | RETRY | FALLBACK | SESSION | UPLOAD

    Writes to the rotating local file AND queues it for the server. Never
    raises. Returns the (scrubbed) entry dict."""
    try:
        if job is not None:
            job_id = job_id or job.get("id", "")
            workflow = workflow or job.get("workflow", "")
            step = step or (job.get("stage_label") or (f"stage {job.get('stage')}"
                            if job.get("stage") is not None else ""))
        exc_type = ""
        if exc is not None:
            exc_type = type(exc).__name__
            if not message:
                message = str(exc)
            if not traceback:
                traceback = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        entry = {
            "ts": time.time(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": (kind or "ERROR").upper(),
            "username": _agent_ctx.get("username", ""),
            "agent_name": _agent_ctx.get("agent_name", ""),
            "agent_id": _agent_ctx.get("agent_id", ""),
            "code_version": _agent_ctx.get("code_version", ""),
            "job_id": job_id or "",
            "workflow": workflow or "",
            "step": step or "",
            "exc_type": exc_type,
            "message": message or "",
            "traceback": traceback or "",
            "detail": detail or "",
        }
        entry = _scrub_entry(entry)
        _write_local(entry)
        _enqueue_and_flush(entry)
        return entry
    except Exception as diag_exc:
        # Absolutely never let diagnostics break a job.
        try:
            print(f"[diag] failed to record event: {diag_exc}")
        except Exception:
            pass
        return {}


def _write_local(entry: dict) -> None:
    line = (f"{entry['time']} [{entry['kind']}] "
            f"job={entry.get('job_id') or '-'} wf={entry.get('workflow') or '-'} "
            f"step={entry.get('step') or '-'} "
            f"user={entry.get('username') or '-'}"
            + (f" exc={entry['exc_type']}" if entry.get("exc_type") else "")
            + (f" :: {entry['message']}" if entry.get("message") else "")
            + (f" | {entry['detail']}" if entry.get("detail") else ""))
    try:
        _get_logger().info(line)
        if entry.get("traceback"):
            _get_logger().info(entry["traceback"].rstrip())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Server upload with a persistent offline queue
# ---------------------------------------------------------------------------

def _enqueue_and_flush(entry: dict) -> None:
    """Append the entry to the persistent queue, then try to flush the queue to
    the server. A dropped network never loses an entry: it stays in the queue
    until a later flush (or the next event) succeeds."""
    with _lock:
        try:
            with open(_queue_path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
    flush()


def flush() -> None:
    """Send every queued entry to the server, oldest first. Stops at the first
    failure and leaves the rest queued for next time. Safe to call often."""
    post = _agent_ctx.get("post")
    if not post:
        return
    with _lock:
        qp = _queue_path()
        if not qp.exists():
            return
        try:
            lines = qp.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        remaining = list(lines)
        for i, ln in enumerate(lines):
            ln = ln.strip()
            if not ln:
                remaining.pop(0)
                continue
            try:
                entry = json.loads(ln)
            except Exception:
                remaining.pop(0)   # unreadable line — drop it, don't block
                continue
            try:
                r = post("/api/agent/diagnostics", json=entry)
                ok = getattr(r, "status_code", 0) == 200 or getattr(r, "ok", False)
            except Exception:
                ok = False
            if ok:
                remaining.pop(0)
            else:
                # Server unreachable — keep this and everything after it.
                break
        try:
            if remaining:
                qp.write_text("\n".join(remaining) + "\n", encoding="utf-8")
            else:
                qp.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Convenience wrappers for the common signals
# ---------------------------------------------------------------------------

def error(exc: BaseException, tb: str = "", *, job=None, job_id="", workflow="", step=""):
    return record_event("ERROR", exc=exc, traceback=tb, job=job,
                        job_id=job_id, workflow=workflow, step=step)


def timeout(message: str, *, job=None, step="", detail=""):
    return record_event("TIMEOUT", message, job=job, step=step, detail=detail)


def stall(step: str, seconds: float, *, job=None):
    return record_event("STALL", f"No progress on '{step}' for {seconds:.0f}s",
                        job=job, step=step, detail=f"stuck={seconds:.0f}s")


def retry(message: str, *, job=None, step="", detail=""):
    return record_event("RETRY", message, job=job, step=step, detail=detail)


def fallback(message: str, *, job=None, step="", detail=""):
    return record_event("FALLBACK", message, job=job, step=step, detail=detail)


def session(message: str, *, job=None, step="", detail=""):
    return record_event("SESSION", message, job=job, step=step, detail=detail)


def upload_issue(message: str, *, job=None, step="", detail=""):
    return record_event("UPLOAD", message, job=job, step=step, detail=detail)


# ---------------------------------------------------------------------------
# Stall watchdog
# ---------------------------------------------------------------------------
# Timeouts only fire when a wait GIVES UP. A job can also hang silently — the
# step never advances and no exception is raised. The watchdog runs alongside a
# job: if a step makes no progress for longer than its expected limit, it writes
# a STALL entry naming the step and how long it has been stuck, WITHOUT killing
# the job (the job may still recover; we only want the trace).

# Expected per-step ceilings (seconds). Matched loosely by substring against the
# step label so we do not have to enumerate every exact label. A text reply
# should be quick; an image turn can legitimately take minutes.
_STEP_LIMITS = [
    ("reading text", 90),          # text-from-image extraction
    ("read wording", 90),
    ("detect", 120),               # object detection (text reply)
    ("aspect", 150),               # aspect advice (text reply)
    ("upload", 120),               # attaching a file to the composer
    ("variation", 420),            # image generation turns
    ("colour", 420),
    ("style", 420),
    ("final", 420),
    ("collage", 420),
    ("regenerat", 420),
    ("extract", 420),              # artwork extraction (image turn)
    ("step", 420),                 # custom "Step N of M — ..."
]
_DEFAULT_STEP_LIMIT = 300          # anything unrecognised: 5 minutes


def _limit_for(step: str) -> int:
    s = (step or "").lower()
    for needle, secs in _STEP_LIMITS:
        if needle in s:
            return secs
    return _DEFAULT_STEP_LIMIT


class Watchdog:
    """Per-job stall detector. Call heartbeat() whenever the step changes or any
    progress is observed; call stop() when the job ends."""

    def __init__(self, job: dict, poll: float = 5.0):
        self._job = job
        self._poll = poll
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._step = job.get("stage_label") or (f"stage {job.get('stage')}"
                     if job.get("stage") is not None else "starting")
        self._since = time.monotonic()
        self._warned = False          # only one STALL per stuck step
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "Watchdog":
        self._thread.start()
        return self

    def heartbeat(self, step: str | None = None) -> None:
        """Progress observed. If the step label changed, reset the clock."""
        with self._lock:
            new_step = step if step is not None else (
                self._job.get("stage_label")
                or (f"stage {self._job.get('stage')}"
                    if self._job.get("stage") is not None else self._step))
            if new_step != self._step:
                self._step = new_step
                self._since = time.monotonic()
                self._warned = False

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            with self._lock:
                step = self._step
                stuck = time.monotonic() - self._since
                warned = self._warned
            # A job paused for an operator answer (awaiting_*) is not stalled —
            # it is legitimately waiting on a human. Do not flag those.
            status = self._job.get("status", "")
            if status.startswith("awaiting_"):
                with self._lock:
                    self._since = time.monotonic()   # reset so post-resume timing is fresh
                    self._warned = False
                continue
            if not warned and stuck >= _limit_for(step):
                with self._lock:
                    self._warned = True
                stall(step, stuck, job=self._job)

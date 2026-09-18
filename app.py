"""FastAPI web UI for artwork generation jobs.

Serves a single-page operator interface and exposes JSON API routes.
Generation runs on a client-side agent (agent.py) on the designer's PC, which
polls this server for jobs and drives Playwright/ChatGPT there. The server no
longer runs any browser worker itself.

Supports multi-turn workflows (Text workflow with operator decisions).
"""

from __future__ import annotations

import hashlib
import io
import mimetypes
import os
import re
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from config.agent_version import AGENT_CODE_VERSION
from config.job_options import JOB_OPTIONS, PARAMETERISED_OPTIONS
from config.workflows import TEXT_TURN_1, TEXT_TURN_2, TEXT_TURN_3, TEXT_REPLACE_COLLAGE, TEXT_REPLACE_FINAL, EXTRACT_CONTACT_SHEET, EXTRACT_SINGLE, ARTWORK_REGENERATE, CUSTOM_OPERATIONS
from src.aspect import image_info
from src.auth import (
    APP_USERNAME, APP_PASSWORD_HASH, verify_password, sign_cookie,
    get_current_user, check_rate_limit, record_failure, record_success,
    COOKIE_NAME, PUBLIC_PATHS, PUBLIC_PREFIXES,
)
from src.agent_tokens import token_name, get_or_create_for_name, touch_token, list_agents
from src import users
from src import nextcloud as nc
from src import printshop
from src.prompt_registry import (
    registry as prompt_registry, prompt_names_for_job, report_runs,
    PROMPT_KEYS, SOURCE_APP as PROMPT_SOURCE_APP, MODEL as PROMPT_MODEL,
    is_composed as is_prompt_block,
)
from src.nc_live import watcher as nc_watcher

# ---------------------------------------------------------------------------
# ARCHITECTURE NOTE
# The Playwright browser work does NOT run here any more. A client-side agent
# (agent.py) on the designer's PC polls this server, claims queued jobs, runs
# the workflows against a real logged-in Chrome, and posts progress/results
# back. The server owns the jobs table and the operator-facing web UI only.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# App & state
# ---------------------------------------------------------------------------

app = FastAPI(title="Artwork Automation")


def _cookie_user_active(username: str) -> bool:
    """True if the cookie's username maps to an existing, non-disabled user.

    Used by the middleware so a user disabled after login is shut out on their
    very next request. The .env seed admin is treated as active even before it
    is written into the store (migration safety)."""
    try:
        u = users.get_user(username)
    except Exception:
        u = None
    if u is not None:
        return not u.get("disabled")
    # Seed-admin fallback: env admin not yet represented in the store.
    return (username or "").strip().lower() == ((APP_USERNAME or "").strip().lower())


# Auth middleware
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Allow public paths
        if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return await call_next(request)
        # Agent endpoints authenticate with a bearer token, not the operator
        # cookie. The token is validated inside each agent endpoint.
        if path.startswith("/api/agent/"):
            return await call_next(request)
        # Allow login page
        if path == "/login":
            return await call_next(request)
        # Check auth. A valid cookie is necessary but not sufficient: a user
        # DISABLED after their cookie was issued must be shut out at once, so we
        # confirm the account still exists and is enabled (with the seed-admin
        # fallback for the .env admin not yet in the store).
        user = get_current_user(request)
        if user and _cookie_user_active(user):
            return await call_next(request)
        # Not authenticated / disabled. API calls get 401, page loads redirect.
        if path.startswith("/api/"):
            return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
        return RedirectResponse("/login", status_code=302)


app.add_middleware(AuthMiddleware)

# Seed the admin account from .env on first run so nobody is locked out during
# the move to multi-user login. Idempotent once any user exists.
try:
    users.ensure_seed_admin()
except Exception as _seed_exc:  # never block startup on the seed
    print(f"[users] seed skipped: {_seed_exc}")

jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

INPUT_DIR = Path("./input")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = Path("./downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Per-job agent console log is kept in memory, capped so a long or chatty job
# cannot grow the process without bound.
AGENT_LOG_MAX_LINES = 400


def _append_agent_log(job: dict, lines: list[str]) -> None:
    """Append agent console lines to the job's bounded log buffer."""
    if not lines:
        return
    buf = job.setdefault("agent_log", [])
    for ln in lines:
        if ln is None:
            continue
        buf.append(str(ln))
    if len(buf) > AGENT_LOG_MAX_LINES:
        del buf[:len(buf) - AGENT_LOG_MAX_LINES]
# The agent is now a one-dir build shipped as a ZIP (one-file cannot extract
# the large bundled Chromium at runtime). Kept .exe as a fallback name.
AGENT_ZIP_NAME = "ArtworkAgent.zip"
AGENT_EXE_NAME = "ArtworkAgent.exe"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
# Broader than ALLOWED_EXTENSIONS: these are shown in the per-customer artwork
# grid (Nextcloud renders previews for them) even though only the four above can
# be sent into a workflow.
VAULT_IMAGE_EXTENSIONS = ALLOWED_EXTENSIONS | {
    ".gif", ".bmp", ".tif", ".tiff", ".heic", ".avif"}
# Where "Save to Artwork Vault" puts generated files inside each customer folder.
VAULT_SAVE_SUBFOLDER = "AI Artwork"

# ---------------------------------------------------------------------------
# Agent presence: agents register and then poll. We consider the system able to
# generate when some agent has polled within AGENT_ONLINE_WINDOW seconds.
# ---------------------------------------------------------------------------
AGENT_ONLINE_WINDOW = 90.0          # seconds since last request to count as "connected"
AGENT_PROGRESS_TIMEOUT = 300.0      # 5 min without progress -> release the job
# A RUNNING job proves the agent is alive even if a long ChatGPT turn delays the
# next heartbeat — but only within this grace, never unlimited. A PAUSED
# (awaiting_*) job proves nothing: it is waiting on a human, so the agent is
# judged purely on its heartbeat. This is what stopped a shut-down PC whose job
# sat at an awaiting_* pause from showing "connected" forever.
AGENT_RUNNING_GRACE = 300.0         # 5 min: a running job extends presence this far past last_seen
agents: dict[str, dict[str, Any]] = {}   # agent_id -> {name, registered_at, last_seen, logged_in}


def _online_agents() -> list[dict]:
    """Every agent still within the online window, newest first.

    Multiple designers run their own agents on their own PCs, so "is an agent
    available" is a question about the whole fleet, not just the most recent
    registrant. Callers that only need one representative can take the first
    entry; callers deciding availability should look at the whole list.

    Presence is judged on the heartbeat (last_seen within AGENT_ONLINE_WINDOW).
    A currently RUNNING job extends that by AGENT_RUNNING_GRACE, so a long
    ChatGPT turn that delays the next heartbeat does not make a working agent
    look offline (the spurious-503 bug the heartbeat was meant to prevent).

    A job PAUSED at an awaiting_* status does NOT extend presence: it is waiting
    on a human, could sit for hours, and tells us nothing about whether the
    agent process is still alive. Counting a paused-job owner as online forever
    was what kept a shut-down PC showing "connected" indefinitely."""
    now = time.time()
    running_ids = _running_agent_ids()
    live = []
    for a in agents.values():
        seen = a.get("last_seen", 0)
        if now - seen <= AGENT_ONLINE_WINDOW:
            live.append(a)
        elif a.get("id") in running_ids and now - seen <= AGENT_RUNNING_GRACE:
            # Running a job and heartbeat is only a little stale — still alive.
            live.append(a)
    live.sort(key=lambda a: a.get("last_seen", 0), reverse=True)
    return live


def _online_agent() -> dict | None:
    """Return the most recently seen agent still within the online window.

    Kept for callers that just need a single representative (e.g. a name to
    show). Availability decisions should use _online_agents()."""
    live = _online_agents()
    return live[0] if live else None


def _any_agent_logged_in(live: list[dict] | None = None) -> bool:
    """True if ANY connected agent reports a signed-in ChatGPT session."""
    if live is None:
        live = _online_agents()
    return any(a.get("logged_in") for a in live)


def _heartbeat(agent_id: str | None) -> None:
    """Treat ANY authenticated agent request as a heartbeat.

    While a job runs the agent is busy-waiting on ChatGPT and does not poll
    next-job, so we must refresh last_seen from the progress / paused-input
    calls too — otherwise the agent looks offline mid-job."""
    if not agent_id:
        return
    a = agents.get(agent_id)
    if a:
        a["last_seen"] = time.time()


def _requeue_stale_jobs() -> None:
    """Release held jobs back to the queue when their agent is gone.

    Two triggers:
      1. A RUNNING job that has gone silent longer than AGENT_PROGRESS_TIMEOUT
         (the agent crashed/hung mid-turn).
      2. ANY held job (running OR paused at awaiting_*) whose owning agent is no
         longer online. A paused job's work lives in that agent's memory and
         open browser, so if the agent is dead the job cannot be resumed — hand
         it to another agent to re-run rather than leaving it stuck forever.
    Either way the job returns to "queued" and shows as waiting for an agent."""
    now = time.time()
    with _jobs_lock:
        online_ids = {a.get("id") for a in _online_agents()}
        for j in jobs.values():
            status = j.get("status")
            claimed_by = j.get("claimed_by")
            if not claimed_by or status not in _AGENT_BUSY_STATUSES:
                continue

            reason = None
            if status == "running":
                last = j.get("last_progress_at") or j.get("claimed_at") or 0
                if now - last > AGENT_PROGRESS_TIMEOUT and not j.get("awaiting_input"):
                    reason = f"agent silent > {AGENT_PROGRESS_TIMEOUT:.0f}s"
            if reason is None and claimed_by not in online_ids:
                # Owning agent has dropped offline (running past its grace, or
                # paused with a stale heartbeat). Its in-memory job state is lost.
                reason = "owning agent went offline"

            if reason:
                print(f"[server] Releasing job {j['id']} ({status}) — {reason}.")
                j["status"] = "queued"
                j["claimed_by"] = None
                j["claimed_by_name"] = ""
                j["claimed_at"] = None
                j["awaiting_input"] = False
                j["stage_label"] = "Requeued — waiting for an agent"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    files: list[str] = []
    options: list[str] = []
    params: dict[str, str] = {}
    custom_note: str = ""
    client: str = ""
    task_id: str = ""
    workflow: str = ""
    text: str = ""
    text_image: str = ""
    template_turn1: str = ""
    # Text workflow input mode: "typed" (default), "from_image", "replace",
    # "image_element", or "image_style".
    text_mode: str = "typed"
    # "replace" mode: recreate an uploaded design with new wording.
    replace_image: str = ""       # the finished design to recreate
    replace_new_text: str = ""    # the wording it should say instead
    replace_target: str = ""      # optional: which wording to replace (blank = all)
    template_replace_collage: str = ""  # operator-editable collage prompt
    template_replace_final: str = ""    # operator-editable final prompt
    # "image_element"/"image_style" modes (UC-3): wording plus a reference image
    # used either as a design element or as a style guide. Uploaded on turn 1.
    reference_image: str = ""
    # Mockup workflow fields
    mockup_image: str = ""
    extract_mode: str = "auto"  # "grid" or "auto"
    grid_cols: int = 4
    grid_rows: int = 2
    template_extract: str = ""
    template_regen: str = ""
    # Artwork Generation fields
    artwork_files: list[str] = []
    # Custom Operation fields
    custom_operations: list[str] = []
    custom_prompts: dict[str, str] = {}  # {operation_key: edited_prompt}
    aspect_dpi: int = 0  # operator-set DPI for Aspect Ratio Enhancement (0 = use file/300)
    halftone_settings: dict = {}  # {lpi, angle, dot} for the local Half Tone op
    blackout_settings: dict = {}  # {threshold} for the local Black Out op


class SelectionRequest(BaseModel):
    choice: int
    template: str | None = None


class RegenerateRequest(BaseModel):
    template: str | None = None


class CropActionRequest(BaseModel):
    action: str  # "accept", "reextract", "recrop", "redetect"
    mode: str = "auto"
    cols: int = 4
    rows: int = 2
    padding: int = 8
    boxes: list[dict] | None = None  # For "recrop": list of {x, y, w, h}


class MultiSelectRequest(BaseModel):
    choices: list[int]


class NumberSelectionRequest(BaseModel):
    numbers: list[int]


class RatioSelectionRequest(BaseModel):
    # Either a "W:H" ratio string, or explicit inch dimensions.
    ratio: str | None = None
    width: float | None = None
    height: float | None = None
    method: str = "pad"  # "pad" (local) or "regenerate" (ChatGPT)


class ObjectSelectionRequest(BaseModel):
    choices: list[dict]  # [{object: str, color: str}]


class TextConfirmRequest(BaseModel):
    text: str = ""
    # When True the operator asked to re-run the extraction (ChatGPT returned
    # no text). No `text` is required in that case.
    retry: bool = False


class OpenFolderRequest(BaseModel):
    path: str


# --- Agent-facing request models ---
class AgentRegisterRequest(BaseModel):
    agent_name: str = "designer"
    logged_in: bool = False
    code_version: str = ""


class AgentProgressRequest(BaseModel):
    agent_id: str | None = None  # so the server can heartbeat the right agent while a job runs
    stage: int | None = None
    stage_label: str | None = None
    active_time: float | None = None
    logged_in: bool | None = None
    # The agent posts the full job dict so the operator UI reflects live state.
    job: dict | None = None
    # New console log lines produced on the agent since the last sync, so the UI
    # can show what happened without the designer opening the agent terminal.
    log_lines: list[str] | None = None
    # Non-fatal problems that recovered (a retried navigation, a timeout that
    # was survived). Surfaced in the UI so testing shows what nearly failed.
    warnings: list[str] | None = None
    # The agent announces it has entered an operator pause. The server applies
    # this ONLY when the job is currently "running" (running -> awaiting_*), so
    # it can't override an already-answered/terminal job. Also carries the pause
    # metadata the operator UI needs (extracted_text, detected_objects, etc.).
    pause_status: str | None = None
    pause_fields: dict | None = None


class AgentErrorRequest(BaseModel):
    message: str
    step: int | None = None
    session_expired: bool = False
    # Rich diagnostics so a failure can be understood from the web UI alone.
    exc_type: str | None = None       # e.g. "GenerationTimeoutError"
    traceback: str | None = None      # full traceback.format_exc()
    step_name: str | None = None      # workflow step label that failed
    agent_name: str | None = None     # which designer's PC hit it
    timestamp: float | None = None    # epoch seconds on the agent
    warning: bool = False             # True = non-fatal (recovered), not a failure


class NextcloudImportRequest(BaseModel):
    # Vault paths to pull into ./input, and which workflow they are headed for.
    paths: list[str] = []
    target: str = "artwork"


class PrintshopVaultSaveRequest(BaseModel):
    # Ek generated file customer ke folder me, shop ki naming ke mutabiq.
    # `name` ./output ki file; `customer` uska Nextcloud folder; `lifecycle`
    # REF/SRC (naya design) ya WRK/FNL/FNLA (mojooda design ke saath), jis ke
    # liye `attach_to` me us design ki id chahiye.
    name: str = ""
    customer: str = ""
    lifecycle: str = "SRC"
    attach_to: str = ""


class PrintshopSaveRequest(BaseModel):
    # File a generated output back into PrintShop's vault as the design's next
    # working version. `name` is a file in ./output; `asset` is the id of the
    # vault row the design came from, carried in from the Design Studio link.
    name: str = ""
    asset: str = ""


class NextcloudSaveRequest(BaseModel):
    # Save a generated output file back into the vault, under the chosen
    # customer folder. `name` is a file in ./output; `customer` is the vault
    # customer folder (path or bare name); `filename` overrides the saved name.
    name: str = ""
    customer: str = ""
    filename: str = ""
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Job orchestration moved to agent.py (runs on the designer PC). See below for
# the agent API endpoints the client polls.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Job orchestration moved to agent.py (runs on the designer PC). See below for
# the agent API endpoints the client polls.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_filename(directory: Path, name: str) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    candidate = f"{stem}{suffix}"
    counter = 2
    while (directory / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


# A job in any of these statuses is actively held by an agent: it is either
# running on that agent's browser or paused waiting for an operator answer that
# the SAME agent will resume. An agent holds at most one such job at a time.
_AGENT_BUSY_STATUSES = (
    "running", "awaiting_selection", "awaiting_text_confirmation",
    "awaiting_crop_review", "awaiting_multi_selection",
    "awaiting_number_selection", "awaiting_object_selection",
    "awaiting_ratio_selection",
    # Session dropped mid-job: the agent holds the job paused and will resume it
    # once the designer signs in again. It must stay assigned to that agent and
    # the agent must not pick up other work while paused.
    "awaiting_signin",
)


def _busy_agent_ids() -> set[str]:
    """agent_ids currently holding a job (running or paused awaiting an answer).

    Used for CLAIM gating (an agent holding any job — running or paused — must
    not take a second job). NOT used for presence: see _running_agent_ids."""
    return {
        j.get("claimed_by")
        for j in jobs.values()
        if j.get("status") in _AGENT_BUSY_STATUSES and j.get("claimed_by")
    }


def _running_agent_ids() -> set[str]:
    """agent_ids owning a job that is actively RUNNING (not paused).

    Only a running job is proof the agent process is alive. A paused
    (awaiting_*) job is waiting on a human and says nothing about the agent, so
    it must not extend presence."""
    return {
        j.get("claimed_by")
        for j in jobs.values()
        if j.get("status") == "running" and j.get("claimed_by")
    }


def _free_agent_exists() -> bool:
    """True if at least one ONLINE agent is not already holding a job.

    This is the real "can a new job start" test now that each designer runs
    their own agent: refuse only when every online agent is busy, never merely
    because some job exists somewhere."""
    busy = _busy_agent_ids()
    for a in _online_agents():
        if a.get("id") not in busy:
            return True
    return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/login", response_class=HTMLResponse)
def serve_login():
    return HTMLResponse(content=Path("static/login.html").read_text(encoding="utf-8"))


class LoginRequest(BaseModel):
    username: str
    password: str


def _authenticate(username: str, password: str) -> dict | None:
    """Validate a login against the users store, with a seed-admin fallback.

    Primary path is the users store (multi-user). The fallback lets the .env
    admin log in even if the store is somehow empty/unseeded, so nobody is
    locked out during the migration — and only when that user does not already
    exist in the store (the store is authoritative once it has the account)."""
    u = users.verify(username, password)
    if u:
        return u
    # Fallback: env admin, only if not represented in the store yet.
    if (username or "").strip().lower() == (APP_USERNAME or "").strip().lower():
        if users.get_user(username) is None and verify_password(password, APP_PASSWORD_HASH):
            return {"username": (APP_USERNAME or "admin"),
                    "display_name": APP_USERNAME or "Admin", "role": "admin",
                    "disabled": False}
    return None


@app.post("/api/auth/login")
def auth_login(req: LoginRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in 5 minutes.")
    user = _authenticate(req.username, req.password)
    if not user:
        record_failure(ip)
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    record_success(ip)
    # Cookie identity stays the username; role/display are looked up per request
    # from the store so a role change takes effect without re-login.
    cookie_value = sign_cookie(user["username"])
    response = JSONResponse(content={
        "ok": True, "username": user["username"],
        "display_name": user.get("display_name", user["username"]),
        "role": user.get("role", "designer"),
    })
    response.set_cookie(
        key=COOKIE_NAME, value=cookie_value,
        httponly=True, samesite="lax", max_age=86400 * 7,
    )
    return response


@app.post("/api/auth/logout")
def auth_logout():
    response = JSONResponse(content={"ok": True})
    response.delete_cookie(key=COOKIE_NAME)
    return response


def _current_user_record(request: Request) -> dict | None:
    """Resolve the cookie to a full user record (username, display_name, role,
    disabled). Returns None if not authenticated or the account is disabled.

    Falls back to a synthetic admin record for the .env seed admin when it is
    not (yet) represented in the store, so the migration never locks anyone out.
    Looking the record up per request means a role change or a disable takes
    effect immediately, without forcing a re-login."""
    username = get_current_user(request)
    if not username:
        return None
    u = users.get_user(username)
    if u is not None:
        if u.get("disabled"):
            return None
        return u
    # Not in store: only the env seed admin is allowed as a fallback identity.
    if (username or "").strip().lower() == (APP_USERNAME or "").strip().lower():
        return {"username": (APP_USERNAME or "admin"),
                "display_name": APP_USERNAME or "Admin", "role": "admin",
                "disabled": False}
    return None


def _require_admin(request: Request) -> dict:
    """Return the current user record, or raise 403 unless they are an admin."""
    u = _current_user_record(request)
    if not u:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if u.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")
    return u


@app.get("/api/auth/me")
def auth_me(request: Request):
    u = _current_user_record(request)
    if not u:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "username": u["username"],
        "display_name": u.get("display_name", u["username"]),
        "role": u.get("role", "designer"),
    }


@app.get("/", response_class=HTMLResponse)
def serve_index():
    return HTMLResponse(content=Path("static/index.html").read_text(encoding="utf-8"))


@app.get("/api/inputs")
def list_inputs():
    files = []
    for p in sorted(INPUT_DIR.iterdir()):
        if p.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)})
    return files


@app.post("/api/upload")
def upload_files(files: list[UploadFile] = File(...)):
    saved, rejected = [], []
    for f in files:
        ext = Path(f.filename).suffix.lower() if f.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            rejected.append(f.filename or "unknown")
            continue
        safe_name = _unique_filename(INPUT_DIR, f.filename or f"upload{ext}")
        dest = INPUT_DIR / safe_name
        dest.write_bytes(f.file.read())
        saved.append({"name": safe_name, "size_kb": round(dest.stat().st_size / 1024, 1)})
    if rejected and not saved:
        raise HTTPException(status_code=400, detail=f"Only .png .jpg .jpeg .webp accepted. Rejected: {', '.join(rejected)}")
    return {"saved": saved, "rejected": rejected}


@app.get("/api/options")
def list_options():
    return [{"key": k, "label": k.replace("_", " ").title(), "needs_value": k in PARAMETERISED_OPTIONS} for k in JOB_OPTIONS]


# Prompt text comes from Decoinks Prompt Management (the live version of each
# prompt), falling back to config/workflows.py. See src/prompt_registry.py.
_CUSTOM_OP_TEMPLATES = {
    "reconstruct": {"template": "CUSTOM_RECONSTRUCT"},
    "remove_background": {"template": "CUSTOM_REMOVE_BACKGROUND"},
    "halo_removal": {"template": "CUSTOM_HALO_REMOVAL"},
    "change_object_color": {"template_detect": "CUSTOM_DETECT_OBJECTS", "template_apply": "CUSTOM_CHANGE_COLOR"},
    "aspect_ratio": {"template": "CUSTOM_ASPECT_ADVICE"},
}
# Keys the web UI uses in `custom_prompts` → the template each one overrides.
_CUSTOM_PROMPT_KEYS = {
    "reconstruct": "CUSTOM_RECONSTRUCT",
    "remove_background": "CUSTOM_REMOVE_BACKGROUND",
    "halo_removal": "CUSTOM_HALO_REMOVAL",
    "change_object_detect": "CUSTOM_DETECT_OBJECTS",
    "change_object_apply": "CUSTOM_CHANGE_COLOR",
    "aspect_ratio": "CUSTOM_ASPECT_ADVICE",
}


def _live_template(name: str) -> str:
    return prompt_registry.template(name)[0]


@app.get("/api/templates")
def get_templates():
    # NOTE: custom operations are served via /api/custom-operations with their templates embedded.
    # Every text/extraction/artwork prompt — including the two "replace text in a
    # design" prompts — goes through the live registry so they are managed from
    # Prompt Management, with config/workflows.py as the fallback.
    return {"turn1": _live_template("TEXT_TURN_1"),
            "turn2": _live_template("TEXT_TURN_2"),
            "turn3": _live_template("TEXT_TURN_3"),
            "replace_collage": _live_template("TEXT_REPLACE_COLLAGE"),
            "replace_final": _live_template("TEXT_REPLACE_FINAL"),
            "image_element_collage": _live_template("TEXT_IMAGE_ELEMENT_COLLAGE"),
            "image_style_collage": _live_template("TEXT_IMAGE_STYLE_COLLAGE"),
            "extract": _live_template("EXTRACT_CONTACT_SHEET"),
            "regen": _live_template("EXTRACT_SINGLE"),
            "artwork_regen": _live_template("ARTWORK_REGENERATE"),
            "artwork": _live_template("ARTWORK_REGENERATE")}


@app.get("/api/custom-operations")
def get_custom_operations():
    """The custom operations list, each carrying the live text of its prompts."""
    ops = []
    for op in CUSTOM_OPERATIONS:
        op = dict(op)
        for field, name in _CUSTOM_OP_TEMPLATES.get(op["key"], {}).items():
            op[field] = _live_template(name)
        ops.append(op)
    return ops


@app.get("/api/prompt-status")
def prompt_status():
    """Is Prompt Management reachable, and which live versions is this server using?"""
    status = prompt_registry.status()
    status["prompts"] = {
        name: {k: meta.get(k) for k in ("key", "version", "source", "rejected") if meta.get(k) is not None}
        for name, meta in prompt_registry.snapshot(list(PROMPT_KEYS))["meta"].items()
    }
    return status


@app.get("/api/artwork-info")
def artwork_info(file: str, dpi: int | None = None):
    """Return pixel dimensions, ratio and print size for an uploaded file.

    Lets the UI show the numbers the moment a file is uploaded. `dpi` optionally
    overrides the DPI used for the inch calculation (recalculates live)."""
    # Guard against path traversal — only files inside INPUT_DIR by basename.
    safe = Path(file).name
    path = INPUT_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        return image_info(path, dpi_override=dpi if (dpi and dpi > 0) else None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")


@app.post("/api/generate")
def create_job(req: GenerateRequest, request: Request):
    print(f"[create_job] workflow={req.workflow!r} mockup_image={req.mockup_image!r} files={req.files} artwork_files={req.artwork_files}")
    # Who is creating this job — recorded for attribution (shown in results and
    # history). Everyone still sees all jobs; this is identity only, not a
    # visibility restriction.
    _creator = _current_user_record(request)
    _created_by = _creator["username"] if _creator else ""
    _created_by_name = (_creator.get("display_name") or _creator["username"]) if _creator else ""
    if _online_agent() is None:
        raise HTTPException(status_code=503, detail="No agent running. Start the agent on your PC to generate.")
    if not req.client.strip():
        raise HTTPException(status_code=400, detail="Client name is required.")
    if not req.task_id.strip():
        raise HTTPException(status_code=400, detail="Job number is required.")
    # Per-agent limit: refuse only when every online agent is already busy, not
    # merely because some other designer's job exists. A queued job will be
    # picked up by whichever agent frees up next.
    if not _free_agent_exists():
        raise HTTPException(status_code=409, detail="All agents are busy. Wait for one to finish, or start another agent.")

    if req.workflow == "text":
        if req.text_mode == "replace":
            # Replace-text-in-a-design: needs the design image and the new wording.
            # Prompt fallbacks come from the live registry (managed from Prompt
            # Management), same as every other text prompt.
            if not req.replace_image.strip():
                raise HTTPException(status_code=400, detail="Upload the design image to replace text in.")
            if not req.replace_new_text.strip():
                raise HTTPException(status_code=400, detail="Enter the replacement text.")
            c1 = req.template_replace_collage.strip() or _live_template("TEXT_REPLACE_COLLAGE")
            if "{new_text}" not in c1 or "{target_clause}" not in c1:
                raise HTTPException(status_code=400, detail="Collage prompt must contain {new_text} and {target_clause} placeholders.")
            f1 = req.template_replace_final.strip() or _live_template("TEXT_REPLACE_FINAL")
            if "{n}" not in f1:
                raise HTTPException(status_code=400, detail="Final prompt must contain the {n} placeholder.")
        elif req.text_mode in ("image_element", "image_style"):
            # UC-3: wording plus a client reference image. Both required.
            if not req.reference_image.strip():
                raise HTTPException(status_code=400, detail="Upload the reference image.")
            if not req.text.strip():
                raise HTTPException(status_code=400, detail="Enter the wording for the design.")
            reg_name = "TEXT_IMAGE_ELEMENT_COLLAGE" if req.text_mode == "image_element" else "TEXT_IMAGE_STYLE_COLLAGE"
            t1 = req.template_turn1.strip() or _live_template(reg_name)
            if "{text}" not in t1:
                raise HTTPException(status_code=400, detail="Step 1 prompt must contain {text} placeholder.")
        else:
            if not req.text.strip() and not req.text_image.strip():
                raise HTTPException(status_code=400, detail="Enter design text or upload an image containing the text.")
            t1 = req.template_turn1.strip() if req.template_turn1.strip() else _live_template("TEXT_TURN_1")
            if "{text}" not in t1:
                raise HTTPException(status_code=400, detail="Step 1 prompt must contain {text} placeholder.")
    elif req.workflow == "mockup":
        if not req.mockup_image.strip():
            raise HTTPException(status_code=400, detail="Upload a mockup image first.")
        if req.extract_mode == "grid":
            if req.grid_cols < 1 or req.grid_rows < 1:
                raise HTTPException(status_code=400, detail="Grid cols and rows must be at least 1.")
    elif req.workflow == "artwork":
        if not req.artwork_files:
            raise HTTPException(status_code=400, detail="Upload at least one artwork file.")
    elif req.workflow == "custom":
        if not req.artwork_files:
            raise HTTPException(status_code=400, detail="Upload an artwork file.")
        if not req.custom_operations:
            raise HTTPException(status_code=400, detail="Select at least one operation.")
    else:
        if not req.files:
            raise HTTPException(status_code=400, detail="Select at least one image.")
        if not req.options:
            raise HTTPException(status_code=400, detail="Select at least one option.")

    # The prompts this job will run, frozen now: a version published while the
    # job is running does not change it half way. A box the designer changed is
    # kept as they wrote it and recorded as an edit.
    prompt_names = prompt_names_for_job(req.workflow, req.custom_operations, req.options, req.text_mode)
    managed = prompt_registry.snapshot(prompt_names)
    live = managed["templates"]
    edited: list[str] = []

    def _pick(name: str, submitted: str) -> str:
        if name not in live:
            return submitted
        if submitted and not prompt_registry.is_unedited(name, submitted):
            edited.append(name)
            return submitted
        return live[name]

    # Stage-1 template name depends on the text input mode. The image modes
    # (UC-3) use their own collage prompt at stage 1; typed/from_image use
    # TEXT_TURN_1. The operator edit (template_turn1) is attributed to whichever
    # registry name actually runs, so edit-tracking and run reporting are right.
    _stage1_name = {
        "image_element": "TEXT_IMAGE_ELEMENT_COLLAGE",
        "image_style": "TEXT_IMAGE_STYLE_COLLAGE",
    }.get(req.text_mode, "TEXT_TURN_1")
    template_turn1 = _pick(_stage1_name, req.template_turn1.strip())
    # "Replace text in a design" prompts are managed like the rest: an operator
    # edit is recorded, otherwise the live/registry version is used.
    template_replace_collage = _pick("TEXT_REPLACE_COLLAGE", req.template_replace_collage.strip())
    template_replace_final = _pick("TEXT_REPLACE_FINAL", req.template_replace_final.strip())
    template_extract = _pick("EXTRACT_CONTACT_SHEET", req.template_extract.strip())
    if req.workflow == "artwork":
        template_regen = _pick("ARTWORK_REGENERATE", req.template_regen.strip())
    elif req.workflow == "mockup":
        template_regen = _pick("EXTRACT_SINGLE", req.template_regen.strip())
    else:
        template_regen = req.template_regen.strip()
    custom_prompts = dict(req.custom_prompts or {})
    for ui_key, name in _CUSTOM_PROMPT_KEYS.items():
        if name in live:
            custom_prompts[ui_key] = _pick(name, (custom_prompts.get(ui_key) or "").strip())

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "id": job_id, "status": "queued", "workflow": req.workflow or "",
        "text": req.text.strip(), "text_image": req.text_image.strip(),
        "files": req.files, "options": req.options, "params": req.params,
        "custom_note": req.custom_note, "client": req.client.strip(),
        "task_id": req.task_id.strip(), "images": [], "stage_images": {},
        "vault_folder": None, "error": None,
        "started_at": None, "finished_at": None, "created_at": time.time(),
        # Attribution: which user created this job (identity only — everyone
        # still sees all jobs). No leading underscore so it survives _public_job.
        "created_by": _created_by, "created_by_name": _created_by_name,
        "stage": 0, "stage_label": "", "awaiting_input": False,
        "selection_prompt": "", "choices": [], "prompts": [],
        "extracted_text": "", "confirmed_text": "", "paused_at": None,
        # Team's Prompt Management-resolved templates.
        "template_turn1": template_turn1,
        "template_turn2": live.get("TEXT_TURN_2", ""), "template_turn3": live.get("TEXT_TURN_3", ""),
        "template_extract": template_extract,
        "template_regen": template_regen,
        # Prompt Management: the templates chosen for this job, which versions
        # they came from, and which ones the designer edited.
        "managed_prompts": live, "prompt_meta": managed["meta"], "prompt_edited": edited,
        # Text input mode + "replace text in a design" fields (registry-resolved
        # replace templates so they are managed like everything else).
        "text_mode": (req.text_mode or "typed").strip(),
        "replace_image": req.replace_image.strip(),
        "replace_new_text": req.replace_new_text.strip(),
        "replace_target": req.replace_target.strip(),
        "template_replace_collage": template_replace_collage,
        "template_replace_final": template_replace_final,
        "replace_similarity": None,   # source-vs-final, computed on result upload
        # UC-3 "wording with a client-supplied image" reference (turn 1 only).
        "reference_image": req.reference_image.strip(),
        "active_time": 0.0,
        "_regenerate": False, "_regen_template": None,
        # Mockup fields
        "mockup_image": req.mockup_image.strip(),
        "extract_mode": req.extract_mode,
        "grid_cols": req.grid_cols,
        "grid_rows": req.grid_rows,
        "crop_names": [], "crop_count": 0, "crop_warnings": [],
        "selected_crops": [], "final_names": [], "chosen_numbers": [],
        "artwork_files": req.artwork_files, "artwork_errors": [],
        "custom_operations": req.custom_operations, "custom_prompts": custom_prompts,
        "aspect_dpi": req.aspect_dpi, "aspect_info": None, "aspect_recommendations": "",
        "aspect_target": None, "aspect_method": "pad", "aspect_similarity": None,
        "aspect_original_file": "", "aspect_original_features": None,
        "aspect_baseline_file": "", "aspect_baseline_features": None,
        "halftone_settings": req.halftone_settings,
        "blackout_settings": req.blackout_settings,
        "custom_steps_done": [], "detected_objects": "",
        "_reextract": False, "_reextract_mode": "", "_reextract_cols": 4, "_reextract_rows": 2,
        "_recrop_boxes": None,
        "_step_start": None,
        # Agent claim tracking
        "claimed_by": None, "claimed_by_name": "", "claimed_at": None, "last_progress_at": None,
    }
    return {"job_id": job_id}


@app.post("/api/jobs/{job_id}/select")
def submit_selection(job_id: str, req: SelectionRequest):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "awaiting_selection":
        raise HTTPException(status_code=400, detail="This job is not waiting for input.")
    if req.choice < 1 or req.choice > 8:
        raise HTTPException(status_code=400, detail="Choice must be between 1 and 8.")

    # Validate optional template override
    stage = job.get("stage", 1)
    if req.template and req.template.strip():
        tpl = req.template.strip()
        if stage == 1 and "{n}" not in tpl:
            raise HTTPException(status_code=400, detail="Step 2 prompt must contain {n} placeholder.")
        if stage == 2 and "{m}" not in tpl:
            raise HTTPException(status_code=400, detail="Step 3 prompt must contain {m} placeholder.")
        # Store the template for the NEXT step
        name = "TEXT_TURN_2" if stage == 1 else "TEXT_TURN_3" if stage == 2 else None
        if name:
            if prompt_registry.is_unedited(name, tpl):
                tpl = (job.get("managed_prompts") or {}).get(name) or tpl
            elif name not in job.setdefault("prompt_edited", []):
                job["prompt_edited"].append(name)
            job["template_turn2" if stage == 1 else "template_turn3"] = tpl

    job["choices"].append(req.choice)
    job["_regenerate"] = False
    job.setdefault("_answered_pause_stages", []).append(job.get("stage"))
    job["status"] = "running"
    job["awaiting_input"] = False
    return {"ok": True, "choice": req.choice}


@app.post("/api/jobs/{job_id}/regenerate")
def regenerate_stage(job_id: str, req: RegenerateRequest):
    """Re-run the current stage's prompt. Does not advance."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "awaiting_selection":
        raise HTTPException(status_code=400, detail="This job is not waiting for input.")

    stage = job.get("stage", 1)
    if req.template and req.template.strip():
        tpl = req.template.strip()
        if stage == 1 and "{text}" not in tpl:
            raise HTTPException(status_code=400, detail="Step 1 prompt must contain {text} placeholder.")
        if stage == 2 and "{n}" not in tpl:
            raise HTTPException(status_code=400, detail="Step 2 prompt must contain {n} placeholder.")
        job["_regen_template"] = tpl
    else:
        job["_regen_template"] = None

    job["_regenerate"] = True
    job["status"] = "running"
    job["awaiting_input"] = False
    return {"ok": True}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a running or paused job, releasing the worker."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    old_status = job["status"]
    if old_status not in ("running", "awaiting_selection", "awaiting_text_confirmation", "awaiting_crop_review", "awaiting_multi_selection", "awaiting_number_selection", "awaiting_object_selection", "awaiting_ratio_selection", "queued"):
        raise HTTPException(status_code=400, detail="This job cannot be cancelled.")

    # Immediately mark as cancelled on the authoritative record
    job["status"] = "cancelled"
    job["finished_at"] = time.time()
    job["awaiting_input"] = False

    # The agent detects the "cancelled" status on its next poll (progress or
    # paused-input) and aborts the job cleanly. No local event to signal.
    print(f"[cancel] Job {job_id}: {old_status} -> cancelled")
    _report_prompt_runs(job)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/confirm-text")
def confirm_text(job_id: str, req: TextConfirmRequest):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "awaiting_text_confirmation":
        raise HTTPException(status_code=400, detail="This job is not waiting for text confirmation.")
    if req.retry:
        # Operator asked to re-run extraction (empty/wrong result). The agent's
        # turn-0 loop re-asks ChatGPT when it sees this flag on resume, then
        # RE-announces the confirmation pause at the same stage (0). Clear that
        # stage from the answered list so the re-announced pause is accepted
        # instead of being deduped as "already answered".
        job["_retry_extract"] = True
        job["confirmed_text"] = ""
        answered = job.setdefault("_answered_pause_stages", [])
        stg = job.get("stage")
        job["_answered_pause_stages"] = [s for s in answered if s != stg]
        job["status"] = "running"
        job["awaiting_input"] = False
        return {"ok": True, "retry": True}
    else:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        job["_retry_extract"] = False
        job["confirmed_text"] = req.text.strip()
    job.setdefault("_answered_pause_stages", []).append(job.get("stage"))
    job["status"] = "running"
    job["awaiting_input"] = False
    return {"ok": True}


@app.post("/api/jobs/{job_id}/crops")
def handle_crops(job_id: str, req: CropActionRequest):
    """Accept, re-extract, recrop with manual boxes, or redetect for a mockup job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "awaiting_crop_review":
        raise HTTPException(status_code=400, detail="This job is not waiting for crop review.")

    if req.action == "accept":
        job["_reextract"] = False
        job["_recrop_boxes"] = None
    elif req.action == "reextract":
        job["_reextract"] = True
        job["_reextract_mode"] = req.mode
        job["_reextract_cols"] = req.cols
        job["_reextract_rows"] = req.rows
        job["_reextract_padding"] = req.padding
        job["_recrop_boxes"] = None
    elif req.action == "recrop":
        # Operator manually adjusted boxes — recrop locally without ChatGPT
        if not req.boxes:
            raise HTTPException(status_code=400, detail="No boxes provided for recrop.")
        job["_reextract"] = True
        job["_reextract_mode"] = "manual_boxes"
        job["_recrop_boxes"] = req.boxes
        job["_reextract_padding"] = req.padding
    elif req.action == "redetect":
        # Re-run ChatGPT detection in a fresh chat
        job["_reextract"] = True
        job["_reextract_mode"] = "auto"
        job["_reextract_padding"] = req.padding
        job["_recrop_boxes"] = None
    else:
        raise HTTPException(status_code=400, detail="Action must be 'accept', 'reextract', 'recrop', or 'redetect'.")
    job.setdefault("_answered_pause_stages", []).append(job.get("stage"))
    job["status"] = "running"
    job["awaiting_input"] = False
    return {"ok": True}


@app.post("/api/jobs/{job_id}/select-multi")
def select_multi(job_id: str, req: MultiSelectRequest):
    """Operator selects which crops to regenerate."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "awaiting_multi_selection":
        raise HTTPException(status_code=400, detail="This job is not waiting for multi-selection.")
    if not req.choices:
        raise HTTPException(status_code=400, detail="Select at least one artwork.")
    crop_count = job.get("crop_count", 0)
    for c in req.choices:
        if c < 1 or c > crop_count:
            raise HTTPException(status_code=400, detail=f"Choice {c} is out of range (1-{crop_count}).")
    job["selected_crops"] = req.choices
    job.setdefault("_answered_pause_stages", []).append(job.get("stage"))
    job["status"] = "running"
    job["awaiting_input"] = False
    return {"ok": True}


@app.post("/api/jobs/{job_id}/select-numbers")
def select_numbers(job_id: str, req: NumberSelectionRequest):
    """Operator enters which design numbers the client chose from the contact sheet."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "awaiting_number_selection":
        raise HTTPException(status_code=400, detail="This job is not waiting for number selection.")
    if not req.numbers:
        raise HTTPException(status_code=400, detail="Enter at least one number.")
    for n in req.numbers:
        if n < 1:
            raise HTTPException(status_code=400, detail=f"Number {n} is invalid. Must be >= 1.")
    job["chosen_numbers"] = req.numbers
    job.setdefault("_answered_pause_stages", []).append(job.get("stage"))
    job["status"] = "running"
    job["awaiting_input"] = False
    return {"ok": True, "numbers": req.numbers}


@app.post("/api/jobs/{job_id}/select-objects")
def select_objects(job_id: str, req: ObjectSelectionRequest):
    """Operator selects which objects to recolour and their target colours."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "awaiting_object_selection":
        raise HTTPException(status_code=400, detail="This job is not waiting for object selection.")
    if not req.choices:
        raise HTTPException(status_code=400, detail="Select at least one object to recolour.")
    job["object_color_choices"] = req.choices
    job.setdefault("_answered_pause_stages", []).append(job.get("stage"))
    job["status"] = "running"
    job["awaiting_input"] = False
    return {"ok": True}


def _parse_ratio(req: "RatioSelectionRequest") -> tuple[float, float]:
    """Resolve the request into (w, h) ratio components. Raises ValueError on bad input."""
    if req.width is not None and req.height is not None:
        w, h = float(req.width), float(req.height)
        if w <= 0 or h <= 0:
            raise ValueError("Width and height must be positive.")
        return w, h
    if req.ratio:
        parts = req.ratio.replace("x", ":").replace("X", ":").split(":")
        if len(parts) != 2:
            raise ValueError("Ratio must be in the form W:H, e.g. 4:5.")
        try:
            w, h = float(parts[0]), float(parts[1])
        except ValueError:
            raise ValueError("Ratio must contain numbers, e.g. 4:5.")
        if w <= 0 or h <= 0:
            raise ValueError("Ratio values must be positive.")
        return w, h
    raise ValueError("Provide a ratio (e.g. 4:5) or width and height in inches.")


@app.post("/api/jobs/{job_id}/select-ratio")
def select_ratio(job_id: str, req: RatioSelectionRequest):
    """Operator picks the target aspect ratio; padding is applied locally by the worker."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "awaiting_ratio_selection":
        raise HTTPException(status_code=400, detail="This job is not waiting for ratio selection.")
    try:
        w, h = _parse_ratio(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Guard the ratio to a sane range (between 1:4 and 4:1).
    ratio_val = w / h
    if ratio_val < 0.25 or ratio_val > 4.0:
        raise HTTPException(status_code=400, detail="Ratio must be between 1:4 and 4:1.")
    method = (req.method or "pad").lower()
    if method not in ("pad", "regenerate"):
        raise HTTPException(status_code=400, detail="Method must be 'pad' or 'regenerate'.")
    job["aspect_target"] = {"w": w, "h": h}
    job["aspect_method"] = method
    job.setdefault("_answered_pause_stages", []).append(job.get("stage"))
    job["status"] = "running"
    job["awaiting_input"] = False
    return {"ok": True, "target": {"w": w, "h": h}, "method": method}


@app.post("/api/agent/job/{job_id}/signin-resume")
async def agent_signin_resume(job_id: str, request: Request):
    """The agent signed back in after a mid-job session drop. Flip the paused
    job back to 'running' so the agent's blocked worker can carry on.

    Mirrors the operator answer endpoints (they also move awaiting_* -> running),
    but this one is driven by the agent restoring its own session rather than an
    operator answering a prompt. Idempotent: a no-op if the job is not paused for
    sign-in."""
    _require_agent(request)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") == "awaiting_signin":
        job["status"] = "running"
        job["awaiting_input"] = False
        job["stage_label"] = "Resumed after sign-in"
        job["last_progress_at"] = time.time()
        print(f"[server] job {job_id} resumed after agent sign-in")
    return {"ok": True, "status": job.get("status")}


@app.get("/api/jobs/{job_id}/crop/{index}")
def serve_crop(job_id: str, index: int):
    """Serve a specific crop image by index (1-based)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    crop_names = job.get("crop_names", [])
    if index < 1 or index > len(crop_names):
        raise HTTPException(status_code=404, detail="Crop not found.")
    filename = crop_names[index - 1]
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk.")
    return FileResponse(str(path), media_type="image/png", filename=filename)


@app.get("/api/jobs/{job_id}/source")
def serve_source_image(job_id: str):
    """Serve the original mockup source image for the box editor."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    mockup_image = job.get("mockup_image", "")
    if not mockup_image:
        raise HTTPException(status_code=404, detail="No source image.")
    path = INPUT_DIR / mockup_image
    if not path.exists():
        raise HTTPException(status_code=404, detail="Source file not found.")
    return FileResponse(str(path), filename=mockup_image)


@app.get("/api/jobs/{job_id}/final/{index}")
def serve_final(job_id: str, index: int):
    """Serve a regenerated final image by index."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    final_names = job.get("final_names", [])
    # Find the one matching the index
    target = f"{job_id}_final_{index}.png"
    if target not in final_names:
        raise HTTPException(status_code=404, detail="Final image not found.")
    path = OUTPUT_DIR / target
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk.")
    return FileResponse(str(path), media_type="image/png", filename=target)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _public_job(job)


def _output_ready(filename: str) -> bool:
    """True if a generated output file actually exists on disk right now."""
    if not filename:
        return False
    try:
        return (OUTPUT_DIR / Path(filename).name).exists()
    except Exception:
        return False


def _public_job(job: dict) -> dict:
    """The job as the UI should see it: internal fields stripped, and — crucially
    — no image filename exposed until the file is actually on disk.

    Belt-and-braces against the stage-image 404 race: even if an agent announces
    a filename slightly before its upload lands, we withhold that name from the
    polled job until the matching file exists, so the UI never requests a 404.
    Once the upload arrives (a second or two later) the next poll reveals it."""
    safe = {k: v for k, v in job.items() if not k.startswith("_")}

    # stage_images: {key: filename} for single images, plus list-valued entries
    # like "finals". Drop any whose file is not yet present.
    si = safe.get("stage_images")
    if isinstance(si, dict):
        filtered = {}
        for k, v in si.items():
            if isinstance(v, str):
                if _output_ready(v):
                    filtered[k] = v
            elif isinstance(v, list):
                ready = [n for n in v if _output_ready(n)]
                if ready:
                    filtered[k] = ready
            else:
                filtered[k] = v
        safe["stage_images"] = filtered

    # images / final_names: only expose files that exist.
    for key in ("images", "final_names"):
        if isinstance(safe.get(key), list):
            safe[key] = [n for n in safe[key] if _output_ready(n)]

    # custom_steps_done: hide a step's file until it lands (the step row can show
    # once its image is servable).
    steps = safe.get("custom_steps_done")
    if isinstance(steps, list):
        safe["custom_steps_done"] = [s for s in steps if not s.get("file") or _output_ready(s.get("file"))]

    return safe


@app.get("/api/jobs")
def list_jobs():
    return sorted([_public_job(j) for j in jobs.values()], key=lambda j: j["created_at"], reverse=True)


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    """Return the agent's captured console log, warnings and error details for a
    job, so a designer can inspect what happened from the browser."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "log": job.get("agent_log", []),
        "warnings": job.get("warnings", []),
        "error_details": job.get("error_details", []),
        "error_screenshots": job.get("error_screenshots", []),
    }


@app.get("/api/jobs/{job_id}/error-screenshot/{name}")
def get_error_screenshot(job_id: str, name: str):
    """Serve a failure screenshot the agent uploaded for this job (from logs/)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    safe = Path(name).name
    # Only serve screenshots this job actually recorded — no arbitrary reads.
    if safe not in job.get("error_screenshots", []):
        raise HTTPException(status_code=404, detail="Screenshot not found for this job.")
    path = LOGS_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="Screenshot file missing.")
    return FileResponse(str(path), media_type="image/png", filename=safe)


@app.get("/api/output/{name}")
def serve_output(name: str):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="image/png", filename=name)


@app.get("/api/jobs/{job_id}/stage/{stage}.png")
def serve_stage_image(job_id: str, stage: str):
    """Serve a stage image. Resolves from ./output first, then the vault folder.

    A MISSING image is always a 404, never a 500. Callers routinely request
    stages a job does not have — e.g. /stage/finals.png on a failed or cancelled
    job that never produced any images, or while an upload is still in flight.
    Several such cases used to raise an unhandled exception (500):
      * `finals` maps to a LIST (final_names), so `OUTPUT_DIR / filename` hit a
        TypeError on a non-str path;
      * a bad/again-unavailable vault_folder made iterdir() raise OSError.
    We now treat every "cannot resolve a real file" outcome as a clean 404."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    filename = (job.get("stage_images") or {}).get(stage)
    # Only a single string names a servable image. Lists (e.g. "finals" ->
    # final_names) and anything non-string are not directly servable here — the
    # UI fetches those per-name via /api/output/<name>.
    if isinstance(filename, list):
        raise HTTPException(status_code=404,
                            detail=f"Stage '{stage}' is a set of images; fetch each by name.")
    if not isinstance(filename, str) or not filename:
        raise HTTPException(status_code=404, detail=f"No image for stage '{stage}'.")

    # Try ./output/ first, then the vault folder. Guard every filesystem touch so
    # an odd path or a vanished vault folder degrades to 404, not 500.
    try:
        path = OUTPUT_DIR / Path(filename).name
        if not path.exists():
            vault_folder = job.get("vault_folder")
            if vault_folder:
                vault_path = Path(vault_folder)
                found = None
                if vault_path.exists():
                    for f in vault_path.iterdir():
                        if f.name == filename:
                            found = f
                            break
                    if found is None:
                        # stage_images name may differ from the vault name — match
                        # by stage pattern.
                        pattern = {"stage1": "stage1", "stage2": "stage2",
                                   "final": "final"}.get(stage, stage)
                        for f in vault_path.iterdir():
                            if pattern in f.name and f.suffix == ".png" and "original" not in f.name:
                                found = f
                                break
                if found is not None:
                    path = found

        if not path.exists():
            raise HTTPException(status_code=404, detail="Image file not found on disk.")
    except HTTPException:
        raise
    except Exception as exc:
        # Any unexpected filesystem/type error -> 404, never a 500.
        print(f"[server] stage image resolve failed for {job_id}/{stage}: {exc}")
        raise HTTPException(status_code=404, detail="Image not available.")

    return FileResponse(str(path), media_type="image/png", filename=path.name)


@app.post("/api/open-folder")
def open_folder(req: OpenFolderRequest):
    folder = Path(req.path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Folder not found.")
    os.startfile(str(folder.resolve()))
    return {"ok": True}


@app.get("/api/status")
def get_status():
    _requeue_stale_jobs()
    live = _online_agents()
    # `worker_alive`/`logged_in` keys are kept for the existing UI: they now mean
    # "at least one agent is connected" and "at least one connected agent reports
    # a signed-in ChatGPT session". With multiple designers we count the whole
    # fleet, not just the newest registrant.
    online = len(live) > 0
    logged_in = _any_agent_logged_in(live)
    # Prefer a signed-in agent's name for the header label; else the newest.
    label_agent = next((a for a in live if a.get("logged_in")), live[0] if live else None)
    # Per-agent limit: a new job can start whenever SOME online agent is free.
    # We no longer advertise a single global "blocking_job_id" — that latched
    # every browser onto one designer's job and blocked the rest. Instead we
    # report who is busy so the UI can show which agent is on each job without
    # blocking anyone else.
    busy = _busy_agent_ids()
    active_jobs = [
        {"job_id": j.get("id"), "status": j.get("status"),
         "agent": j.get("claimed_by_name") or "", "agent_id": j.get("claimed_by")}
        for j in jobs.values() if j.get("status") in _AGENT_BUSY_STATUSES
    ]
    return {
        # Retained for backward-compat, but always null now: the client tracks
        # its own job locally and must not adopt another designer's job.
        "blocking_job_id": None,
        "free_agent": _free_agent_exists(),
        "busy_agent_count": len(busy),
        "active_jobs": active_jobs,
        "worker_alive": online,
        "worker_error": "" if online else "No agent running - start the agent on your PC to generate.",
        "logged_in": logged_in,
        "agent_connected": online,
        "agent_count": len(live),
        "agent_name": label_agent.get("name") if label_agent else "",
        "agent_names": [a.get("name", "") for a in live],
        # Code version of the labelled agent, and per-agent detail so the UI can
        # show the running version beside each connected agent.
        "agent_version": label_agent.get("code_version") if label_agent else "",
        "agents": [
            {"name": a.get("name", ""),
             "logged_in": bool(a.get("logged_in")),
             "code_version": a.get("code_version") or "unknown",
             # Epoch seconds of the last heartbeat, so the UI can show and
             # sanity-check "connected" at a glance.
             "last_seen": a.get("last_seen", 0)}
            for a in live
        ],
    }


@app.get("/api/session")
def get_session():
    """Report the fleet's ChatGPT session state (agent-reported). Signed-in if
    ANY connected agent reports a session."""
    live = _online_agents()
    label_agent = next((a for a in live if a.get("logged_in")), live[0] if live else None)
    return {"logged_in": _any_agent_logged_in(live), "account": "acct1",
            "agent_connected": len(live) > 0,
            "agent_name": label_agent.get("name") if label_agent else ""}


@app.post("/api/session/login")
def session_login():
    """Sign-in now happens on the designer's PC: they run login.py there. The
    server cannot drive the remote browser, so this just reports guidance."""
    if not _online_agents():
        raise HTTPException(status_code=503, detail="No agent running. Start the agent on your PC.")
    return {"ok": True, "message": "Run login.py on the PC where the agent runs to sign in to ChatGPT."}


@app.post("/api/session/confirm")
def session_confirm():
    """Re-report the fleet's session state."""
    return {"logged_in": _any_agent_logged_in()}


# ---------------------------------------------------------------------------
# OPERATOR: agent download + token surfacing (cookie-authed by the middleware)
# ---------------------------------------------------------------------------

@app.get("/api/my-agent-token")
def my_agent_token(request: Request, agent_name: str = ""):
    """Return the logged-in user's OWN agent token.

    Tokens are now per USER (issued when the account is created). This returns
    the caller's own token so they can paste it into their agent — on as many
    machines as they like; the identity is the token, and each machine still
    registers as its own agent entry. The `agent_name` query param is now just a
    display label for the agent window and no longer selects the token.

    Legacy fallback: an account that predates the users store (e.g. the .env
    seed admin before it was written) falls back to a name-keyed token so it is
    never left without one.
    """
    u = _current_user_record(request)
    if not u:
        raise HTTPException(status_code=401, detail="Not authenticated")
    rec = users.get_user(u["username"])
    if rec and rec.get("agent_token"):
        token = rec["agent_token"]
    else:
        # Legacy / seed-admin-not-yet-in-store: fall back to a name-keyed token.
        token = get_or_create_for_name(u.get("display_name") or u["username"])
    label = (agent_name or "").strip() or u.get("display_name") or u["username"]
    return {"token": token, "name": label, "server_url": str(request.base_url).rstrip("/")}


@app.get("/api/agent-tokens")
def agent_tokens(request: Request):
    """List issued tokens (name + last-seen, no secrets) so an admin can see who
    is set up. Cookie-authed by the middleware."""
    if not get_current_user(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    # Match each issued token to a currently-connected agent (by name) so the
    # admin panel can show the running code version next to who is online.
    live = _online_agents()
    ver_by_name = {a.get("name", ""): (a.get("code_version") or "") for a in live}
    online_names = set(ver_by_name.keys())
    out = []
    for entry in list_agents():
        nm = entry.get("name", "")
        entry = dict(entry)
        entry["online"] = nm in online_names
        entry["code_version"] = ver_by_name.get(nm, "")
        out.append(entry)
    return {"agents": out}


# ---------------------------------------------------------------------------
# Admin: user management. All endpoints are admin-only (role=admin). The cookie
# gate is already applied by AuthMiddleware; _require_admin adds the role check.
# ---------------------------------------------------------------------------

class CreateUserRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""
    role: str = "designer"


class RoleRequest(BaseModel):
    role: str


class DisableRequest(BaseModel):
    disabled: bool


class PasswordRequest(BaseModel):
    password: str


@app.get("/api/admin/users")
def admin_list_users(request: Request):
    """List all users (no secrets) for the admin screen, each annotated with
    whether one of their agents is currently connected (and its last-seen time
    and code version). This is the admin's fleet view — it used to live in the
    designer-facing agent-setup dialog, which no longer shows any of it."""
    me = _require_admin(request)
    # Most-recent online agent per owning user (a person may run several PCs).
    agent_by_user: dict[str, dict] = {}
    for a in _online_agents():
        uname = (a.get("username") or "").strip().lower()
        if not uname:
            continue
        prev = agent_by_user.get(uname)
        if prev is None or (a.get("last_seen", 0) > prev.get("last_seen", 0)):
            agent_by_user[uname] = a
    out = []
    for u in users.list_users():
        u = dict(u)
        a = agent_by_user.get((u.get("username") or "").strip().lower())
        u["agent_online"] = a is not None
        u["agent_last_seen"] = a.get("last_seen") if a else None
        u["agent_code_version"] = (a.get("code_version") if a else "") or ""
        out.append(u)
    return {"users": out, "me": me["username"]}


@app.post("/api/admin/users")
def admin_create_user(req: CreateUserRequest, request: Request):
    """Create a user with an email + password. The agent token is still issued
    and stored (the agent obtains it by signing in with the email+password), but
    it is NO LONGER returned or displayed anywhere — designers authenticate with
    their credentials, not a pasted token."""
    _require_admin(request)
    try:
        users.create_user(
            username=req.username, password=req.password,
            display_name=req.display_name or req.username, role=req.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "username": (req.username or "").strip().lower()}


@app.post("/api/admin/users/{username}/disable")
def admin_set_disabled(username: str, req: DisableRequest, request: Request):
    """Disable or re-enable a user. Disabling invalidates their agent token
    immediately (user_for_token rejects a disabled user). An admin cannot
    disable themselves."""
    me = _require_admin(request)
    if req.disabled and username.strip().lower() == me["username"].strip().lower():
        raise HTTPException(status_code=400, detail="You cannot disable your own account.")
    u = users.set_disabled(username, req.disabled)
    if not u:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"ok": True, "username": u["username"], "disabled": u["disabled"]}


@app.post("/api/admin/users/{username}/role")
def admin_set_role(username: str, req: RoleRequest, request: Request):
    """Change a user's role. An admin cannot demote themselves out of admin."""
    me = _require_admin(request)
    if (username.strip().lower() == me["username"].strip().lower()
            and req.role != "admin"):
        raise HTTPException(status_code=400, detail="You cannot remove your own admin role.")
    try:
        u = users.set_role(username, req.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not u:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"ok": True, "username": u["username"], "role": u["role"]}


@app.post("/api/admin/users/{username}/reset-password")
def admin_reset_password(username: str, req: PasswordRequest, request: Request):
    """Set a new password for a user."""
    _require_admin(request)
    try:
        u = users.reset_password(username, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not u:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"ok": True, "username": u["username"]}


@app.post("/api/admin/users/{username}/regenerate-token")
def admin_regenerate_token(username: str, request: Request):
    """Replace a user's agent token. Returns the NEW token ONCE (the old one
    stops working immediately)."""
    _require_admin(request)
    token = users.regenerate_token(username)
    if not token:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"ok": True, "username": username.strip().lower(),
            "agent_token": token, "token_shown_once": True}


@app.get("/api/download/agent")
def download_agent():
    """Serve the built agent. Prefers the one-dir ZIP; falls back to a lone exe.
    Operator auth enforced by middleware."""
    zip_path = DOWNLOADS_DIR / AGENT_ZIP_NAME
    if zip_path.exists():
        return FileResponse(
            str(zip_path),
            media_type="application/zip",
            filename=AGENT_ZIP_NAME,
            headers={"Content-Disposition": f'attachment; filename="{AGENT_ZIP_NAME}"'},
        )
    exe_path = DOWNLOADS_DIR / AGENT_EXE_NAME
    if exe_path.exists():
        return FileResponse(
            str(exe_path),
            media_type="application/vnd.microsoft.portable-executable",
            filename=AGENT_EXE_NAME,
            headers={"Content-Disposition": f'attachment; filename="{AGENT_EXE_NAME}"'},
        )
    # Designer-facing: no dev instructions (build_agent.bat means nothing to
    # them). The UI catches this and shows the friendly message in the modal
    # rather than navigating the browser to raw JSON.
    raise HTTPException(
        status_code=404,
        detail="The agent download isn't ready yet. Please contact your administrator — it will be available here shortly.",
    )


# ---------------------------------------------------------------------------
# AGENT SELF-UPDATE — code-only. The full ZIP (with Chromium) is for first
# installs; after that an agent pulls just its ~1 MB of Python via these two
# endpoints. Both require the agent bearer token.
# ---------------------------------------------------------------------------

# The Python the agent needs, relative to the project root. Only these travel;
# no Chromium, no venv, no build artefacts.
_AGENT_CODE_FILES = ("agent.py", "agent_gui.py")
_AGENT_CODE_DIRS = ("src", "config")
# Never ship these — caches, compiled artefacts, secrets, local state.
_BUNDLE_EXCLUDE_DIRS = {"__pycache__", ".git", "build", "dist", ".venv"}
_BUNDLE_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _build_code_bundle() -> bytes:
    """Build the code-only update zip on the fly from the working tree.

    Contains agent.py, agent_gui.py and the src/ and config/ trees — the exact
    set an agent imports — and nothing else. Deterministic-ish (sorted) so the
    same tree yields the same archive."""
    root = Path(__file__).parent
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in _AGENT_CODE_FILES:
            p = root / fname
            if p.is_file():
                zf.write(p, fname)
        for dname in _AGENT_CODE_DIRS:
            base = root / dname
            if not base.is_dir():
                continue
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                if any(part in _BUNDLE_EXCLUDE_DIRS for part in p.relative_to(root).parts):
                    continue
                if p.suffix.lower() in _BUNDLE_EXCLUDE_SUFFIXES:
                    continue
                zf.write(p, str(p.relative_to(root)).replace(os.sep, "/"))
    return buf.getvalue()


@app.get("/api/agent/code-version")
def agent_code_version(request: Request):
    """Current agent CODE version. Bearer-token authenticated."""
    _require_agent(request)
    return {"version": AGENT_CODE_VERSION}


@app.get("/api/agent/code-bundle")
def agent_code_bundle(request: Request):
    """Serve the code-only update zip. Bearer-token authenticated.

    Small (~1 MB): just the agent's Python. The Chromium runtime is only in the
    first-install ZIP and never changes, so it is deliberately excluded here."""
    _require_agent(request)
    data = _build_code_bundle()
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="agent_code.zip"',
            "X-Agent-Code-Version": AGENT_CODE_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# AGENT API — polled by the designer-PC agent. Bearer-token authenticated.
# Declared BEFORE app.mount() so the static mount does not shadow them.
# ---------------------------------------------------------------------------

class AgentAuthRequest(BaseModel):
    email: str
    password: str


@app.post("/api/agent/authenticate")
def agent_authenticate(req: AgentAuthRequest):
    """Exchange a designer's email + password for their agent token.

    The agent now signs in with the SAME credentials as the website instead of
    a pasted token; on success it uses the returned token as its bearer for all
    subsequent /api/agent/* calls. Failures are reported in plain terms — never
    a raw HTTP error — and distinguish bad credentials from a disabled account.
    The password is never logged or echoed back."""
    token, reason = users.authenticate_agent(req.email, req.password)
    if reason == "ok":
        print(f"[agent-auth] {(req.email or '').strip().lower()!r} authenticated")
        return {"token": token}
    if reason == "disabled":
        print(f"[agent-auth] {(req.email or '').strip().lower()!r} rejected: disabled")
        raise HTTPException(
            status_code=403,
            detail="This account has been disabled. Ask an admin to re-enable it.",
        )
    # bad_credentials (unknown email or wrong password)
    print(f"[agent-auth] {(req.email or '').strip().lower()!r} rejected: bad credentials")
    raise HTTPException(
        status_code=401,
        detail="Email or password is incorrect.",
    )


def _require_agent(request: Request) -> str:
    """Validate the agent bearer token; return the designer name or raise 401.

    Thin wrapper over _require_agent_identity for the many callers that only
    need the display name."""
    return _require_agent_identity(request)["name"]


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


def _require_agent_identity(request: Request) -> dict:
    """Validate the agent bearer token and return WHO it belongs to.

    Tokens are now PER USER (issued when an admin creates the user). Resolution
    order:
      1. users store — the authoritative source. A token belonging to a DISABLED
         user resolves to None there, so that agent is rejected immediately with
         no separate revocation step (a departed designer's agent stops working
         the moment they are disabled).
      2. legacy agent_tokens store — machine-name tokens issued before this
         change, so existing agents keep working until re-issued.

    Returns {"token", "username", "name"} where `name` is the display name.
    Raises 401 for an unknown/missing token or a disabled user's token."""
    token = _bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid or missing agent token.")

    u = users.user_for_token(token)   # None if unknown OR disabled
    if u is not None:
        # Best-effort last-seen for the legacy admin panel is not needed here;
        # the users store carries its own bookkeeping and per-agent presence is
        # tracked by agent_id below.
        return {"token": token,
                "username": u.get("username", ""),
                "name": u.get("display_name") or u.get("username") or "designer"}

    # Legacy machine-name token fallback.
    name = token_name(token)
    if not name:
        raise HTTPException(status_code=401, detail="Invalid or missing agent token.")
    touch_token(token)
    return {"token": token, "username": "", "name": name}


# Job fields that must never be overwritten by the agent's posted copy — the
# server owns the operator-supplied answers and the claim bookkeeping.
_SERVER_OWNED_FIELDS = {
    "confirmed_text", "choices", "chosen_numbers", "selected_crops",
    "object_color_choices", "aspect_target", "aspect_method",
    "_regenerate", "_regen_template", "_reextract", "_reextract_mode",
    "_reextract_cols", "_reextract_rows", "_reextract_padding", "_recrop_boxes",
    "claimed_by", "claimed_by_name", "claimed_at", "last_progress_at",
    # Lifecycle fields the SERVER owns. An agent's per-2s job echo must never
    # undo an operator's answer (the bug where choices=[4,4,4] landed but the
    # agent's stale "awaiting_selection" overwrote the server's "running").
    # `status` is handled specially below (terminal statuses are allowed).
    "awaiting_input", "paused_at",
    # Prompt Management bookkeeping is decided by the server.
    "managed_prompts", "prompt_meta", "prompt_edited",
}

# Terminal statuses the agent IS allowed to set.
_TERMINAL_STATUSES = {"done", "done_with_errors", "failed", "cancelled"}


def _merge_agent_job(job: dict, posted: dict) -> None:
    """Merge the agent's job copy into the authoritative record.

    The agent owns generation-produced fields (stage_label, images,
    stage_images, prompts, results, errors). The server owns operator answers,
    claim bookkeeping and the pause lifecycle — those are never overwritten.
    A cancellation on the server always wins.

    `status` is special: the agent may move the job to a TERMINAL status
    (done / failed / ...), but must never push it back to an awaiting_* or
    running value — that would undo an operator's answer.
    """
    if job.get("status") == "cancelled":
        return
    try:
        _merge_agent_fields(job, posted)
    finally:
        _report_prompt_runs(job)


def _merge_agent_fields(job: dict, posted: dict) -> None:
    for k, v in posted.items():
        if k in _SERVER_OWNED_FIELDS:
            continue
        if k == "status":
            server_status = job.get("status", "")
            # Only accept terminal statuses from the agent.
            if v in _TERMINAL_STATUSES:
                job["status"] = v
            elif v != server_status:
                # e.g. agent still carrying "awaiting_selection" while the
                # operator has already answered and the server is "running".
                print(f"[server] Ignored agent status {v!r} for job {job.get('id')} "
                      f"(server is {server_status!r}) — an agent sync cannot move a "
                      f"job back to a non-terminal status.")
            continue
        job[k] = v


# ── Prompt Management run log ────────────────────────────────────────────────
_prompt_runs_reported: set[str] = set()
_RUN_STATUS = {"done": "success", "done_with_errors": "success", "failed": "failed", "cancelled": "cancelled"}


def _template_pattern(template: str) -> "re.Pattern[str] | None":
    import string as _string
    try:
        parts = []
        for literal, field, _spec, _conv in _string.Formatter().parse(template):
            parts.append(re.escape(literal))
            if field is not None:
                parts.append(".*?")
        return re.compile("".join(parts), re.DOTALL)
    except (ValueError, re.error):
        return None


def _used_template(job: dict, name: str) -> str:
    """The template this job actually ran for `name` (the designer's edit, if any)."""
    custom = job.get("custom_prompts") or {}
    by_field = {
        "TEXT_TURN_1": job.get("template_turn1"), "TEXT_TURN_2": job.get("template_turn2"),
        "TEXT_TURN_3": job.get("template_turn3"), "EXTRACT_CONTACT_SHEET": job.get("template_extract"),
        "EXTRACT_SINGLE": job.get("template_regen"), "ARTWORK_REGENERATE": job.get("template_regen"),
        "TEXT_REPLACE_COLLAGE": job.get("template_replace_collage"),
        "TEXT_REPLACE_FINAL": job.get("template_replace_final"),
        # Image modes carry their stage-1 edit in template_turn1.
        "TEXT_IMAGE_ELEMENT_COLLAGE": job.get("template_turn1"),
        "TEXT_IMAGE_STYLE_COLLAGE": job.get("template_turn1"),
    }
    by_field.update({name_: custom.get(ui_key) for ui_key, name_ in _CUSTOM_PROMPT_KEYS.items()})
    return by_field.get(name) or (job.get("managed_prompts") or {}).get(name) or ""


def _report_prompt_runs(job: dict) -> None:
    """Once a job ends, tell Decoinks which prompt versions it ran and how it went.

    One record per prompt the job actually sent to ChatGPT, holding the exact
    text sent. Runs in the background and never affects the job."""
    job_id = job.get("id")
    status = job.get("status")
    if not job_id or status not in _RUN_STATUS or job_id in _prompt_runs_reported:
        return
    _prompt_runs_reported.add(job_id)
    try:
        names = prompt_names_for_job(job.get("workflow", ""), job.get("custom_operations"), job.get("options"), job.get("text_mode"))
        sent = [str(p) for p in job.get("prompts", []) if p]
        meta_all = job.get("prompt_meta") or {}
        edited = set(job.get("prompt_edited") or [])
        started = job.get("started_at")
        finished = job.get("finished_at") or time.time()
        files = [n for n in (job.get("images") or []) if isinstance(n, str)]
        files += [s.get("file") for s in (job.get("custom_steps_done") or []) if isinstance(s, dict) and s.get("file")]
        files = list(dict.fromkeys(files))[:100]
        errors = [job.get("error")] if job.get("error") else []
        errors += [str(e) for e in (job.get("artwork_errors") or [])]
        runs = []
        for name in names:
            candidates = [t for t in (_used_template(job, name), (job.get("managed_prompts") or {}).get(name)) if t]
            matched: list[str] = []
            for tpl in candidates:
                pat = _template_pattern(tpl)
                # A base/option block is one part of the composed prompt, so look inside it.
                hit = (lambda p: pat.search(p)) if is_prompt_block(name) else (lambda p: pat.fullmatch(p))
                matched = [p for p in sent if pat and hit(p)]
                if matched:
                    break
            if not matched:
                continue  # this prompt never reached ChatGPT in this job
            meta = meta_all.get(name) or {}
            runs.append({
                "prompt_key": PROMPT_KEYS[name],
                "prompt_version_id": meta.get("version_id"),
                "source_app": PROMPT_SOURCE_APP,
                "external_ref": job_id,
                "status": _RUN_STATUS[status],
                "prompt_source": "edited" if name in edited else (meta.get("source") or "built_in"),
                "input_variables": {
                    "workflow": job.get("workflow"), "client": job.get("client"), "job_number": job.get("task_id"),
                    "text": job.get("confirmed_text") or job.get("text") or None,
                    "choices": job.get("choices") or None, "chosen_numbers": job.get("chosen_numbers") or None,
                    "custom_operations": job.get("custom_operations") or None,
                    "template": name,
                },
                "resolved_prompt": "\n\n----- next send -----\n\n".join(matched)[:200000],
                "provider": PROMPT_MODEL["provider"], "model": PROMPT_MODEL["model"],
                "model_parameters": {"interface": "ChatGPT web, driven by the Artwork Agent",
                                     "agent": job.get("claimed_by_name") or None, "times_sent": len(matched)},
                "latency_ms": int((finished - started) * 1000) if started else None,
                "error_message": ("; ".join(errors))[:5000] or None,
                "output_reference": job.get("vault_folder"),
                "output_files": [{"file_name": f, "file_url": f"/api/output/{f}"} for f in files],
            })
        if runs:
            report_runs(runs)
            print(f"[prompts] job {job_id}: reporting {len(runs)} prompt run(s) to Decoinks")
    except Exception as exc:
        print(f"[prompts] could not build the run log for job {job_id}: {exc}")


# Warm the prompt cache so the first page load already shows the live text.
threading.Thread(target=prompt_registry.refresh, name="prompt-warmup", daemon=True).start()


@app.post("/api/agent/register")
def agent_register(req: AgentRegisterRequest, request: Request):
    ident = _require_agent_identity(request)
    # Identity now comes from the TOKEN (the user), not the self-reported name.
    # The agent keeps its "Your name" field purely as a machine label for the
    # fleet view; a person running two PCs registers two separate agent entries
    # (agent_id is a fresh uuid per register), which we want — the fleet stays
    # honest about how many machines are live.
    agent_id = uuid.uuid4().hex[:12]
    machine_label = req.agent_name or ident["name"]
    agents[agent_id] = {
        "id": agent_id,
        "name": machine_label,               # machine/display label for the fleet view
        "username": ident.get("username", ""),  # the OWNING user (token identity)
        "user_display": ident["name"],          # the user's display name
        "registered_at": time.time(), "last_seen": time.time(),
        "logged_in": req.logged_in,
        "code_version": (req.code_version or "").strip() or "unknown",
    }
    print(f"[server] Agent registered: machine={machine_label!r} "
          f"user={ident.get('username') or '(legacy)'} ({agent_id}) "
          f"code {agents[agent_id]['code_version']}")
    return {"agent_id": agent_id, "name": machine_label}


@app.get("/api/agent/next-job")
def agent_next_job(request: Request, agent_id: str, logged_in: bool = False, code_version: str = ""):
    _require_agent(request)
    a = agents.get(agent_id)
    if a:
        a["last_seen"] = time.time()
        a["logged_in"] = logged_in
        if code_version.strip():
            a["code_version"] = code_version.strip()
    _requeue_stale_jobs()
    # Claim the oldest queued job atomically. An agent holds at most one job at
    # a time, so if this agent is already running/paused on one, it takes
    # nothing new — the busy check and the claim happen under the same lock so
    # two rapid polls can't both slip a job onto one agent.
    with _jobs_lock:
        if agent_id in _busy_agent_ids():
            return Response(status_code=204)
        queued = sorted(
            [j for j in jobs.values() if j.get("status") == "queued"],
            key=lambda j: j.get("created_at", 0),
        )
        if not queued:
            return Response(status_code=204)
        job = queued[0]
        job["status"] = "running"
        job["started_at"] = time.time()
        job["claimed_by"] = agent_id
        job["claimed_by_name"] = a.get("name") if a else ""
        job["claimed_at"] = time.time()
        job["last_progress_at"] = time.time()
    # Return the full job plus file URLs the agent needs to download.
    input_files = []
    for key in ("files", "artwork_files"):
        input_files += [f for f in job.get(key, []) if f]
    for key in ("text_image", "mockup_image", "replace_image", "reference_image"):
        v = job.get(key)
        if v:
            input_files.append(v)
    return {"job": {k: v for k, v in job.items()}, "input_files": sorted(set(input_files))}


@app.post("/api/agent/job/{job_id}/progress")
def agent_progress(job_id: str, req: AgentProgressRequest, request: Request):
    _require_agent(request)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    # Heartbeat: keep the agent "online" while it is busy running this job.
    _heartbeat(req.agent_id or job.get("claimed_by"))
    if job.get("status") == "cancelled":
        return {"ok": True, "cancelled": True}
    job["last_progress_at"] = time.time()
    if req.job is not None:
        _merge_agent_job(job, req.job)
    # Console output + non-fatal warnings from the agent, so the UI can show
    # what happened (and what nearly failed) without the agent terminal.
    if req.log_lines:
        _append_agent_log(job, req.log_lines)
    if req.warnings:
        wbuf = job.setdefault("warnings", [])
        for w in req.warnings:
            entry = {"message": str(w), "timestamp": time.time(),
                     "stage_label": job.get("stage_label", "")}
            wbuf.append(entry)
            _append_agent_log(job, [f"[warning] {w}"])
    # Agent announces it has entered an operator pause. Apply ONLY when the job
    # is currently "running" AND the operator hasn't already answered a pause at
    # this stage — so a duplicate/stale announce can't re-open a resolved pause.
    if req.pause_status and req.pause_status.startswith("awaiting_"):
        announce_stage = req.stage if req.stage is not None else job.get("stage")
        answered = job.setdefault("_answered_pause_stages", [])
        if job.get("status") != "running":
            print(f"[server] Ignored pause announcement {req.pause_status!r} for job "
                  f"{job_id} (server status is {job.get('status')!r}, not running).")
        elif announce_stage in answered:
            print(f"[server] Ignored pause announcement {req.pause_status!r} for job "
                  f"{job_id} (stage {announce_stage} was already answered).")
        else:
            job["status"] = req.pause_status
            job["awaiting_input"] = True
            job["paused_at"] = time.time()
            # Pause metadata the operator UI needs (e.g. extracted_text,
            # detected_objects, aspect_recommendations). These are agent-owned.
            for k, v in (req.pause_fields or {}).items():
                job[k] = v
            print(f"[server] Job {job_id} entered pause: {req.pause_status} (stage {announce_stage})")
    if req.stage is not None:
        job["stage"] = req.stage
    if req.stage_label is not None:
        job["stage_label"] = req.stage_label
    if req.active_time is not None:
        job["active_time"] = req.active_time
    a = agents.get(req.agent_id or job.get("claimed_by"))
    if a and req.logged_in is not None:
        a["logged_in"] = req.logged_in
    return {"ok": True, "cancelled": False}


@app.post("/api/agent/job/{job_id}/result")
async def agent_result(job_id: str, request: Request):
    _require_agent(request)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    form = await request.form()
    saved = []
    for key, value in form.multi_items():
        if hasattr(value, "filename") and value.filename:
            data = await value.read()
            dest = OUTPUT_DIR / Path(value.filename).name
            dest.write_bytes(data)
            saved.append(dest.name)
    job["last_progress_at"] = time.time()
    _heartbeat(job.get("claimed_by"))
    _recompute_aspect_similarity(job)
    _recompute_replace_similarity(job)
    return {"ok": True, "saved": saved}


def _recompute_aspect_similarity(job: dict) -> None:
    """Compute the Aspect Ratio similarity on the SERVER from uploaded files.

    The agent produces the baseline + final images but no longer runs the cv2
    comparison. Here (server-side, where cv2 is available) we compare the final
    against the baseline and populate the fields the UI reads. Runs only for a
    regenerate aspect step and only when both files are present."""
    baseline = job.get("aspect_baseline_file")
    if not baseline:
        return
    final_step = None
    for step in job.get("custom_steps_done", []) or []:
        if step.get("op") == "aspect_ratio" and step.get("method") == "regenerate":
            final_step = step
    if not final_step or not final_step.get("file"):
        return
    if final_step.get("similarity"):  # already computed (e.g. by an older agent)
        return
    b_path = OUTPUT_DIR / baseline
    f_path = OUTPUT_DIR / final_step["file"]
    if not (b_path.exists() and f_path.exists()):
        return
    try:
        from src.compare import similarity as _similarity
        sim = _similarity(b_path.read_bytes(), f_path.read_bytes())
        final_step["similarity"] = sim
        job["aspect_similarity"] = sim
        print(f"[server] computed aspect similarity for job {job['id']}: "
              f"shape={sim.get('shape_pct')} detail={sim.get('detail_pct')}")
    except Exception as exc:
        print(f"[server] aspect similarity computation failed: {exc}")


def _recompute_replace_similarity(job: dict) -> None:
    """For a 'replace text in a design' job, compare the SOURCE design against
    the FINAL artwork so the operator can see how far the design drifted —
    preserving it is the whole point of this mode. Source is the uploaded input
    file; final is the recorded stage image. Runs server-side where cv2 lives."""
    if job.get("text_mode") != "replace":
        return
    if job.get("replace_similarity"):  # already computed
        return
    source = (job.get("replace_image") or "").strip()
    final = (job.get("stage_images") or {}).get("final")
    if not source or not final:
        return
    s_path = INPUT_DIR / Path(source).name
    f_path = OUTPUT_DIR / Path(final).name
    if not (s_path.exists() and f_path.exists()):
        return
    try:
        from src.compare import similarity as _similarity
        sim = _similarity(s_path.read_bytes(), f_path.read_bytes())
        job["replace_similarity"] = sim
        print(f"[server] computed replace similarity for job {job['id']}: "
              f"shape={sim.get('shape_pct')} detail={sim.get('detail_pct')}")
    except Exception as exc:
        print(f"[server] replace similarity computation failed: {exc}")


@app.post("/api/agent/job/{job_id}/error")
async def agent_error(job_id: str, request: Request):
    """Record a failure (or a non-fatal warning) with full diagnostics.

    Accepts multipart/form-data so a failure screenshot can ride along with the
    JSON detail fields. Falls back to a plain JSON body when no file is sent.
    Everything is stored on the job so the web UI can show the message plainly
    plus a collapsible technical view — the designer never needs the terminal.
    """
    _require_agent(request)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    ctype = request.headers.get("content-type", "")
    fields: dict[str, Any] = {}
    screenshot_name = None
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        for key, value in form.multi_items():
            if hasattr(value, "filename") and value.filename:
                data = await value.read()
                safe = Path(value.filename).name
                dest = LOGS_DIR / safe
                dest.write_bytes(data)
                screenshot_name = safe
            else:
                fields[key] = value
    else:
        try:
            fields = await request.json()
        except Exception:
            fields = {}

    message = (fields.get("message") or "Unknown error").strip() or "Unknown error"
    is_warning = str(fields.get("warning", "")).lower() in ("1", "true", "yes")

    detail = {
        "message": message,
        "exc_type": fields.get("exc_type") or "",
        "traceback": fields.get("traceback") or "",
        "step_name": fields.get("step_name") or job.get("stage_label", ""),
        "agent_name": fields.get("agent_name") or job.get("claimed_by_name", ""),
        "timestamp": _coerce_float(fields.get("timestamp")) or time.time(),
        "warning": is_warning,
        "screenshot": screenshot_name,
    }
    job.setdefault("error_details", []).append(detail)
    if screenshot_name:
        job.setdefault("error_screenshots", []).append(screenshot_name)
    # Mirror into the agent log so the single log view shows it in sequence.
    _append_agent_log(job, [f"[{'warning' if is_warning else 'error'}] "
                            f"{detail['step_name']}: {detail['exc_type']} {message}".strip()])

    # A warning is non-fatal — record it but do not fail the job.
    if not is_warning and job.get("status") != "cancelled":
        job["status"] = "failed"
        job["error"] = message
        job["finished_at"] = time.time()
        job["awaiting_input"] = False
        _report_prompt_runs(job)

    if str(fields.get("session_expired", "")).lower() in ("1", "true", "yes"):
        a = agents.get(job.get("claimed_by"))
        if a:
            a["logged_in"] = False

    print(f"[server] Agent {'warning' if is_warning else 'error'} on job {job_id}: {message}")
    return {"ok": True}


def _coerce_float(v) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


@app.get("/api/agent/job/{job_id}/paused-input")
def agent_paused_input(job_id: str, request: Request):
    """The agent polls this while blocked at a pause. Returns the operator's
    answer once available, or {ready:false} while still awaiting. Also surfaces
    a cancellation so the agent can abort."""
    _require_agent(request)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    job["last_progress_at"] = time.time()
    # Heartbeat: the agent polls this while paused at an operator prompt.
    _heartbeat(job.get("claimed_by"))
    status = job.get("status", "")
    if status == "cancelled":
        return {"ready": True, "cancelled": True}
    # Still awaiting the operator's answer?
    if status.startswith("awaiting_"):
        return {"ready": False, "cancelled": False, "status": status}
    # Operator has answered (status moved to running). Hand back every field the
    # workflows read after a pause.
    return {
        "ready": True, "cancelled": False, "status": status,
        "confirmed_text": job.get("confirmed_text", ""),
        "_retry_extract": job.get("_retry_extract", False),
        "choices": job.get("choices", []),
        "chosen_numbers": job.get("chosen_numbers", []),
        "selected_crops": job.get("selected_crops", []),
        "object_color_choices": job.get("object_color_choices", []),
        "aspect_target": job.get("aspect_target"),
        "aspect_method": job.get("aspect_method", "pad"),
        "template_turn2": job.get("template_turn2", ""),
        "template_turn3": job.get("template_turn3", ""),
        "_regenerate": job.get("_regenerate", False),
        "_regen_template": job.get("_regen_template"),
        "_reextract": job.get("_reextract", False),
        "_reextract_mode": job.get("_reextract_mode", ""),
        "_reextract_cols": job.get("_reextract_cols", 4),
        "_reextract_rows": job.get("_reextract_rows", 2),
        "_reextract_padding": job.get("_reextract_padding", 8),
        "_recrop_boxes": job.get("_recrop_boxes"),
    }


@app.get("/api/agent/file/{name}")
def agent_file(name: str, request: Request):
    _require_agent(request)
    safe = Path(name).name
    path = INPUT_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), filename=safe)

# ---------------------------------------------------------------------------
# Nextcloud vault (Leads 2.0)
# ---------------------------------------------------------------------------
#
# The operator's source artwork lives in Nextcloud, not on this box. These
# routes let them browse a customer's folder, watch it live, and send a file
# straight into one of the workflows. Nothing writes back: an import copies the
# file into ./input, the customer's folder is left exactly as it was.

MAX_IMPORT_BYTES = max(1, int(os.environ.get("NEXTCLOUD_MAX_IMPORT_MB", "80") or 80)) * 1024 * 1024
CHANGES_WAIT_SECONDS = 25.0

# Content hash -> file already sitting in ./input. Re-sending the same artwork
# (the usual case when an operator retries a job) reuses the copy instead of
# growing a pile of name_2.png, name_3.png.
_nc_imports: dict[str, str] = {}
_nc_import_lock = threading.Lock()

# AW-<CLIENT>-<NNNN> out of a vault filename, e.g. AW-JBR05-0004-OUT.jpg.
_ARTWORK_CODE = re.compile(r"(AW-[A-Za-z0-9]+-\d{4})", re.IGNORECASE)

WORKFLOW_TARGETS = {
    "artwork": {"label": "Artwork Generation", "multiple": True},
    "mockup": {"label": "Artwork Extraction", "multiple": False},
    "custom": {"label": "Custom Operation", "multiple": False},
    "text": {"label": "Text (read wording from image)", "multiple": False},
    # Text + Image: the vault file is the client's image; the operator picks the
    # mode (element / style / replace) in the tab before entering wording.
    "textimage": {"label": "Text + Image", "multiple": False},
}


def _nc_error(exc: nc.NextcloudError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=str(exc))


def _decorate(entry: nc.Entry, cfg: nc.NextcloudConfig) -> dict:
    """A listing row the UI can render without knowing the vault's layout."""
    item = entry.as_dict()
    ext = Path(entry.name).suffix.lower()
    customer = nc.customer_folder(entry.path, cfg)
    item.update({
        "ext": ext,
        "customer": customer,
        "customer_label": nc.display_name(customer),
        # Only these four reach the generator; everything else is shown but
        # cannot be sent, with the reason on the row.
        "importable": (not entry.is_dir) and ext in ALLOWED_EXTENSIONS,
        "reason": "" if entry.is_dir or ext in ALLOWED_EXTENSIONS
                  else f"{ext or 'This file type'} is not supported — use .png .jpg .jpeg .webp",
    })
    return item


@app.get("/api/nextcloud/status")
def nextcloud_status():
    """Connectivity plus live-watcher health, for the header indicator."""
    info = nc.test_connection()
    snap = nc_watcher.snapshot(since=nc_watcher.revision)
    info.update({
        "watching": snap["watching"], "stale": snap["stale"],
        "revision": snap["revision"], "poll_seconds": snap["poll_seconds"],
        "watch_error": snap["error"],
        "last_poll_ago": round(time.time() - snap["last_poll_at"], 1) if snap["last_poll_at"] else None,
        "targets": [{"key": k, **v} for k, v in WORKFLOW_TARGETS.items()],
    })
    return info


@app.get("/api/nextcloud/customers")
def nextcloud_customers(q: str = "", refresh: bool = False):
    """Every customer folder under the root, for the dropdown."""
    try:
        items = nc_watcher.customers(refresh=refresh)
    except nc.NextcloudError as exc:
        raise _nc_error(exc)
    needle = q.strip().lower()
    if needle:
        items = [c for c in items if needle in c["label"].lower() or needle in c["folder"].lower()]
    return {"count": len(items), "customers": items}


@app.get("/api/nextcloud/browse")
def nextcloud_browse(path: str = ""):
    """One folder: its sub-folders and its files, plus breadcrumbs back to the root."""
    cfg = nc.get_config()
    try:
        target = nc.safe_rel(path, cfg)
        entries = nc.list_folder(target)
    except nc.NextcloudError as exc:
        raise _nc_error(exc)

    folders = sorted((_decorate(e, cfg) for e in entries if e.is_dir),
                     key=lambda f: f["name"].lower())
    # Newest first: the file an operator wants is nearly always the one that
    # just arrived.
    files = sorted((_decorate(e, cfg) for e in entries if not e.is_dir),
                   key=lambda f: f["modified_ms"], reverse=True)

    rest = target[len(cfg.root):].strip("/")
    crumbs = [{"label": cfg.root, "path": cfg.root}]
    walked = cfg.root
    for part in [p for p in rest.split("/") if p]:
        walked = f"{walked}/{part}"
        crumbs.append({"label": nc.display_name(part) if walked.count("/") == cfg.root.count("/") + 1 else part,
                       "path": walked})

    customer = nc.customer_folder(target, cfg)
    parent = target.rsplit("/", 1)[0] if target != cfg.root and "/" in target else ""
    return {
        "path": target, "parent": parent, "is_root": target == cfg.root,
        "breadcrumbs": crumbs, "folders": folders, "files": files,
        "customer": customer, "customer_label": nc.display_name(customer),
        "revision": nc_watcher.revision,
    }


@app.get("/api/nextcloud/thumb")
def nextcloud_thumb(path: str, w: int = 320, h: int = 320):
    """Proxy Nextcloud's thumbnail so the browser never sees the credentials."""
    try:
        data, ctype = nc.preview(path, width=max(32, min(1024, w)), height=max(32, min(1024, h)))
    except nc.NextcloudError as exc:
        raise _nc_error(exc)
    return Response(content=data, media_type=ctype or "image/png",
                    headers={"Cache-Control": "private, max-age=300"})


@app.get("/api/nextcloud/changes")
def nextcloud_changes(since: int = 0, wait: bool = True, limit: int = 40):
    """Long-poll the change feed.

    The request parks on the watcher's condition variable and returns the
    instant a file lands, so the browser learns about it without a UI timer and
    without hammering the server between arrivals. It always returns within
    CHANGES_WAIT_SECONDS so proxies and sleeping laptops cannot leave it hanging.
    """
    # Park only when the client is exactly up to date. A `since` *ahead* of the
    # revision means the client is talking to a restarted process (the counter
    # resets); returning at once lets it resync instead of sitting out a full
    # timeout on every poll.
    if wait and nc_watcher.started and since == nc_watcher.revision:
        nc_watcher.wait_for_change(since, CHANGES_WAIT_SECONDS)
    return nc_watcher.snapshot(since=since, limit=max(1, min(200, limit)))


@app.post("/api/nextcloud/import")
def nextcloud_import(req: NextcloudImportRequest):
    """Copy the chosen vault files into ./input and say which workflow they suit."""
    target = (req.target or "artwork").strip()
    if target not in WORKFLOW_TARGETS:
        raise HTTPException(status_code=400, detail=f"Unknown workflow target {target!r}.")
    paths = [p for p in (req.paths or []) if str(p).strip()]
    if not paths:
        raise HTTPException(status_code=400, detail="Select at least one file.")
    if not WORKFLOW_TARGETS[target]["multiple"] and len(paths) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"{WORKFLOW_TARGETS[target]['label']} takes one file at a time.")

    cfg = nc.get_config()
    imported: list[dict] = []
    skipped: list[dict] = []
    for raw in paths:
        try:
            rel = nc.safe_rel(raw, cfg)
        except nc.NextcloudError as exc:
            skipped.append({"path": raw, "reason": str(exc)})
            continue
        name = Path(rel).name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append({"path": rel, "reason": "Only .png .jpg .jpeg .webp can be sent to a workflow."})
            continue
        try:
            data, _ = nc.download_file(rel, max_bytes=MAX_IMPORT_BYTES)
        except nc.NextcloudError as exc:
            skipped.append({"path": rel, "reason": str(exc)})
            continue

        digest = hashlib.sha256(data).hexdigest()
        with _nc_import_lock:
            existing = _nc_imports.get(digest)
            if existing and (INPUT_DIR / existing).exists():
                local = existing
            else:
                local = _unique_filename(INPUT_DIR, name)
                (INPUT_DIR / local).write_bytes(data)
                _nc_imports[digest] = local

        customer = nc.customer_folder(rel, cfg)
        code = _ARTWORK_CODE.search(name)
        imported.append({
            "name": local, "nc_path": rel, "nc_name": name,
            "customer": customer, "customer_label": nc.display_name(customer),
            "size_kb": round(len(data) / 1024, 1),
            "artwork_code": code.group(1).upper() if code else "",
        })

    if not imported:
        detail = skipped[0]["reason"] if skipped else "Nothing could be imported."
        raise HTTPException(status_code=400, detail=detail)

    return {
        "target": target,
        "target_label": WORKFLOW_TARGETS[target]["label"],
        "files": imported,
        "skipped": skipped,
        # Prefills for the job form, so the operator does not retype what the
        # vault path already says.
        "client": imported[0]["customer_label"],
        "task_id": imported[0]["artwork_code"],
    }


@app.get("/api/nextcloud/artworks")
def nextcloud_artworks(customer: str = "", limit: int = 500):
    """Every image under one customer folder, flattened, newest first.

    The tab shows a customer's artwork as one grid rather than making the
    operator open each order folder — one WebDAV SEARCH (Depth: infinity) walks
    the whole subtree, then we keep the image types and sort newest-first.
    """
    cfg = nc.get_config()
    if not str(customer or "").strip():
        raise HTTPException(status_code=400, detail="Select a customer first.")
    try:
        root = nc.safe_rel(customer, cfg)
        entries = nc.search_modified_since(0, limit=max(1, min(2000, limit)), rel_root=root)
    except nc.NextcloudError as exc:
        raise _nc_error(exc)

    files = [
        _decorate(e, cfg) for e in entries
        if not e.is_dir and Path(e.name).suffix.lower() in VAULT_IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda f: f["modified_ms"], reverse=True)
    return {
        "customer": nc.customer_folder(root, cfg),
        "customer_label": nc.display_name(nc.customer_folder(root, cfg)),
        "path": root, "count": len(files), "files": files,
        "revision": nc_watcher.revision,
    }


@app.post("/api/nextcloud/save-to-vault")
def nextcloud_save_to_vault(req: NextcloudSaveRequest):
    """Copy a generated output file into `<customer>/AI Artwork/` in the vault."""
    cfg = nc.get_config()

    # The output filename must resolve inside ./output and nowhere else.
    name = Path(str(req.name or "").strip()).name
    if not name:
        raise HTTPException(status_code=400, detail="No output file given.")
    src = OUTPUT_DIR / name
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404, detail="That generated file is no longer available.")

    customer = str(req.customer or "").strip()
    if not customer:
        raise HTTPException(status_code=400, detail="Choose a customer folder to save into.")
    try:
        cust_path = nc.safe_rel(customer, cfg)
    except nc.NextcloudError as exc:
        raise _nc_error(exc)
    # Must be a customer folder (one level under the root), not the root itself.
    if cust_path == cfg.root or nc.customer_folder(cust_path, cfg) == "":
        raise HTTPException(status_code=400, detail="Choose a customer folder, not the vault root.")

    data = src.read_bytes()
    ctype = mimetypes.guess_type(name)[0] or "image/png"

    # Preferred saved name: the operator's override, else the output's own name.
    desired = Path(str(req.filename or "").strip() or name).name
    if not Path(desired).suffix:
        desired += Path(name).suffix or ".png"

    folder = f"{cust_path}/{VAULT_SAVE_SUBFOLDER}"
    try:
        nc.ensure_folder(folder)
    except nc.NextcloudError as exc:
        raise _nc_error(exc)

    stem, suffix = Path(desired).stem, Path(desired).suffix
    # Bump the name on collision (…_v2, _v3) unless the caller asked to overwrite.
    for attempt in range(20):
        candidate = desired if attempt == 0 else f"{stem}_v{attempt + 1}{suffix}"
        try:
            saved = nc.upload_file(f"{folder}/{candidate}", data, ctype,
                                   overwrite=bool(req.overwrite))
        except nc.NextcloudError as exc:
            if exc.status == 412 and not req.overwrite:
                continue  # name taken — try the next version
            raise _nc_error(exc)
        return {
            "ok": True,
            "path": saved["path"],
            "name": candidate,
            "folder": folder,
            "customer": nc.customer_folder(cust_path, cfg),
            "customer_label": nc.display_name(nc.customer_folder(cust_path, cfg)),
            "size_kb": round(len(data) / 1024, 1),
        }
    raise HTTPException(status_code=409,
                        detail="Too many files with that name already — rename and try again.")


# ── Arriving from the Design Studio ─────────────────────────────────────────
@app.get("/api/printshop/handoff")
def printshop_handoff(asset: str = "", path: str = ""):
    """Open the vault on the one file the Design Studio sent over.

    The link carries the file's vault path and the id of its row in PrintShop's
    index. Neither is a credential — PrintShop will not act on that id without a
    signed token — so nothing sensitive rides in the URL of this plain-HTTP app.
    The path is all this side needs to show the file; the id is kept only so a
    later save can be filed against the same design.
    """
    if not str(asset or "").strip():
        raise HTTPException(status_code=400, detail="That link carries no artwork id.")
    cfg = nc.get_config()
    try:
        rel = nc.safe_rel(path, cfg)
    except nc.NextcloudError as exc:
        raise _nc_error(exc)

    folder = nc.customer_folder(rel, cfg)
    if not folder:
        raise HTTPException(status_code=400,
                            detail="That link does not point inside a customer folder.")
    name = Path(rel).name
    code = _ARTWORK_CODE.search(name)
    return {
        "asset": str(asset).strip(),
        "nc_path": rel,
        "file_name": name,
        "customer": folder,
        # The Nextcloud tab selects a customer by full path, not bare name.
        "customer_path": f"{cfg.root}/{folder}",
        "customer_label": nc.display_name(folder),
        "artwork_code": code.group(1).upper() if code else "",
        "importable": Path(name).suffix.lower() in ALLOWED_EXTENSIONS,
    }


@app.get("/api/printshop/designs")
def printshop_designs(customer: str = ""):
    """Is customer ke mojooda designs — jin ke saath nayi file jodi ja sakti hai."""
    try:
        key = printshop.entity_key(customer)
        return {"entity_key": key, "rows": printshop.list_designs(key)}
    except printshop.PrintshopError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@app.post("/api/printshop/save-vault")
def printshop_save_vault(req: PrintshopVaultSaveRequest):
    """Generated file ko customer ke folder me, shop ki naming ke saath rakho.

    Pehle yeh file `AI Artwork` naam ke alag folder me, apne hi naam ke saath
    girti thi — yani naming ke nizaam se bahar, aur us design se kati hui jis ka
    hissa thi. Ab PrintShop hi number deta hai: REF/SRC ko is client ka agla
    khaali number, aur WRK/FNL/FNLA ko usi design ka number jis ke saath jodi
    gayi. Ginti vault se aati hai, isliye kisi mojooda number se takra nahi
    sakti.
    """
    name = Path(str(req.name or "").strip()).name
    if not name:
        raise HTTPException(status_code=400, detail="No generated file given.")
    src = OUTPUT_DIR / name
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404,
                            detail="That generated file is no longer available.")

    data = src.read_bytes()
    mime = mimetypes.guess_type(name)[0] or "image/png"
    try:
        key = printshop.entity_key(req.customer)
        saved = printshop.upload_to_vault(key, req.lifecycle, req.attach_to, name, data, mime)
    except printshop.PrintshopError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

    return {"ok": True, "size_kb": round(len(data) / 1024, 1), **saved}


@app.post("/api/printshop/save-wrk")
def printshop_save_wrk(req: PrintshopSaveRequest):
    """Save a generated image as the design's next WRK version.

    PrintShop decides the name, the folder and the version number and indexes
    the result, so the file appears in the vault on its own — see
    src/printshop.py for why none of that is worked out here.
    """
    name = Path(str(req.name or "").strip()).name
    if not name:
        raise HTTPException(status_code=400, detail="No generated file given.")
    src = OUTPUT_DIR / name
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404,
                            detail="That generated file is no longer available.")

    data = src.read_bytes()
    mime = mimetypes.guess_type(name)[0] or "image/png"
    try:
        saved = printshop.save_working_file(req.asset, name, data, mime)
    except printshop.PrintshopError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

    return {"ok": True, "size_kb": round(len(data) / 1024, 1), **saved}


# The watcher runs whether or not anyone is looking at the Nextcloud tab, so a
# file that lands while the operator is mid-job is already in the feed when
# they switch to it.
nc_watcher.start()


app.mount("/input", StaticFiles(directory="input"), name="input")
app.mount("/static", StaticFiles(directory="static"), name="static")

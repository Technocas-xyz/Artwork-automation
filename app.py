"""FastAPI web UI for artwork generation jobs.

Serves a single-page operator interface and exposes JSON API routes.
Generation runs in a dedicated background worker thread to avoid blocking
the async event loop (Playwright sync API is not async-compatible).

Supports multi-turn workflows (Text workflow with operator decisions).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from config.job_options import JOB_OPTIONS, PARAMETERISED_OPTIONS
from config.workflows import TEXT_TURN_1, TEXT_TURN_2, TEXT_TURN_3, EXTRACT_CONTACT_SHEET, EXTRACT_SINGLE, ARTWORK_REGENERATE, CUSTOM_OPERATIONS
from src.aspect import image_info
from src.auth import (
    APP_USERNAME, APP_PASSWORD_HASH, verify_password, sign_cookie,
    get_current_user, check_rate_limit, record_failure, record_success,
    COOKIE_NAME, PUBLIC_PATHS, PUBLIC_PREFIXES,
)
from src.agent_tokens import token_name, get_or_create_for_name

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
        # Check auth
        user = get_current_user(request)
        if not user:
            # API calls get 401, page loads get redirect
            if path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
            return RedirectResponse("/login", status_code=302)
        # Authenticated — continue
        return await call_next(request)


app.add_middleware(AuthMiddleware)

jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

INPUT_DIR = Path("./input")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = Path("./downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
# The agent is now a one-dir build shipped as a ZIP (one-file cannot extract
# the large bundled Chromium at runtime). Kept .exe as a fallback name.
AGENT_ZIP_NAME = "ArtworkAgent.zip"
AGENT_EXE_NAME = "ArtworkAgent.exe"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# ---------------------------------------------------------------------------
# Agent presence: agents register and then poll. We consider the system able to
# generate when some agent has polled within AGENT_ONLINE_WINDOW seconds.
# ---------------------------------------------------------------------------
AGENT_ONLINE_WINDOW = 30.0          # seconds since last poll to count as "connected"
AGENT_PROGRESS_TIMEOUT = 300.0      # 5 min without progress -> release the job
agents: dict[str, dict[str, Any]] = {}   # agent_id -> {name, registered_at, last_seen, logged_in}


def _online_agent() -> dict | None:
    """Return the most recently seen agent still within the online window."""
    now = time.time()
    best = None
    for a in agents.values():
        if now - a.get("last_seen", 0) <= AGENT_ONLINE_WINDOW:
            if best is None or a["last_seen"] > best["last_seen"]:
                best = a
    return best


def _requeue_stale_jobs() -> None:
    """Release jobs whose claiming agent stopped sending progress."""
    now = time.time()
    with _jobs_lock:
        for j in jobs.values():
            if j.get("status") == "running" and j.get("claimed_by"):
                last = j.get("last_progress_at") or j.get("claimed_at") or 0
                if now - last > AGENT_PROGRESS_TIMEOUT and not j.get("awaiting_input"):
                    print(f"[server] Releasing stale job {j['id']} (agent silent > {AGENT_PROGRESS_TIMEOUT}s)")
                    j["status"] = "queued"
                    j["claimed_by"] = None
                    j["claimed_at"] = None
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
    text: str


class OpenFolderRequest(BaseModel):
    path: str


# --- Agent-facing request models ---
class AgentRegisterRequest(BaseModel):
    agent_name: str = "designer"
    logged_in: bool = False


class AgentProgressRequest(BaseModel):
    stage: int | None = None
    stage_label: str | None = None
    active_time: float | None = None
    logged_in: bool | None = None
    # The agent posts the full job dict so the operator UI reflects live state.
    job: dict | None = None


class AgentErrorRequest(BaseModel):
    message: str
    step: int | None = None
    session_expired: bool = False


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


def _any_job_blocking() -> str | None:
    for jid, j in jobs.items():
        if j.get("status") in ("awaiting_selection", "awaiting_text_confirmation", "awaiting_crop_review", "awaiting_multi_selection", "awaiting_number_selection", "awaiting_object_selection", "awaiting_ratio_selection", "running"):
            return jid
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/login", response_class=HTMLResponse)
def serve_login():
    return HTMLResponse(content=Path("static/login.html").read_text(encoding="utf-8"))


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def auth_login(req: LoginRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in 5 minutes.")
    if req.username != APP_USERNAME or not verify_password(req.password, APP_PASSWORD_HASH):
        record_failure(ip)
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    record_success(ip)
    cookie_value = sign_cookie(req.username)
    response = JSONResponse(content={"ok": True, "username": req.username})
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


@app.get("/api/auth/me")
def auth_me(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"username": user}


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


@app.get("/api/templates")
def get_templates():
    # NOTE: custom operations are served via /api/custom-operations with their templates embedded.
    # TODO: Migrate text/extraction/artwork templates to the same single-sourced pattern.
    return {"turn1": TEXT_TURN_1, "turn2": TEXT_TURN_2, "turn3": TEXT_TURN_3, "extract": EXTRACT_CONTACT_SHEET, "regen": EXTRACT_SINGLE, "artwork_regen": ARTWORK_REGENERATE, "artwork": ARTWORK_REGENERATE}


@app.get("/api/custom-operations")
def get_custom_operations():
    """Serve the single-sourced custom operations list with embedded templates."""
    return CUSTOM_OPERATIONS


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
def create_job(req: GenerateRequest):
    print(f"[create_job] workflow={req.workflow!r} mockup_image={req.mockup_image!r} files={req.files} artwork_files={req.artwork_files}")
    if _online_agent() is None:
        raise HTTPException(status_code=503, detail="No agent running. Start the agent on your PC to generate.")
    if not req.client.strip():
        raise HTTPException(status_code=400, detail="Client name is required.")
    if not req.task_id.strip():
        raise HTTPException(status_code=400, detail="Job number is required.")
    blocking = _any_job_blocking()
    if blocking:
        raise HTTPException(status_code=409, detail="A job is waiting for the client's choice. Cancel it or complete it to start a new one.")

    if req.workflow == "text":
        if not req.text.strip() and not req.text_image.strip():
            raise HTTPException(status_code=400, detail="Enter design text or upload an image containing the text.")
        t1 = req.template_turn1.strip() if req.template_turn1.strip() else TEXT_TURN_1
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

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "id": job_id, "status": "queued", "workflow": req.workflow or "",
        "text": req.text.strip(), "text_image": req.text_image.strip(),
        "files": req.files, "options": req.options, "params": req.params,
        "custom_note": req.custom_note, "client": req.client.strip(),
        "task_id": req.task_id.strip(), "images": [], "stage_images": {},
        "vault_folder": None, "error": None,
        "started_at": None, "finished_at": None, "created_at": time.time(),
        "stage": 0, "stage_label": "", "awaiting_input": False,
        "selection_prompt": "", "choices": [], "prompts": [],
        "extracted_text": "", "confirmed_text": "", "paused_at": None,
        "template_turn1": req.template_turn1.strip(),
        "template_turn2": "", "template_turn3": "",
        "template_extract": req.template_extract.strip() if hasattr(req, 'template_extract') else "",
        "template_regen": req.template_regen.strip() if hasattr(req, 'template_regen') else "",
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
        "custom_operations": req.custom_operations, "custom_prompts": req.custom_prompts,
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
        if stage == 1:
            job["template_turn2"] = tpl
        elif stage == 2:
            job["template_turn3"] = tpl

    job["choices"].append(req.choice)
    job["_regenerate"] = False
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
    return {"ok": True}


@app.post("/api/jobs/{job_id}/confirm-text")
def confirm_text(job_id: str, req: TextConfirmRequest):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "awaiting_text_confirmation":
        raise HTTPException(status_code=400, detail="This job is not waiting for text confirmation.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    job["confirmed_text"] = req.text.strip()
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
    job["status"] = "running"
    job["awaiting_input"] = False
    return {"ok": True, "target": {"w": w, "h": h}, "method": method}


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
    # Don't expose internal fields
    safe = {k: v for k, v in job.items() if not k.startswith("_")}
    return safe


@app.get("/api/jobs")
def list_jobs():
    return sorted([{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs.values()], key=lambda j: j["created_at"], reverse=True)


@app.get("/api/output/{name}")
def serve_output(name: str):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="image/png", filename=name)


@app.get("/api/jobs/{job_id}/stage/{stage}.png")
def serve_stage_image(job_id: str, stage: str):
    """Serve a stage image. Resolves from vault folder using pathlib."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    filename = job.get("stage_images", {}).get(stage)
    if not filename:
        raise HTTPException(status_code=404, detail=f"No image for stage '{stage}'.")

    # Try ./output/ first, then vault folder
    path = OUTPUT_DIR / filename
    if not path.exists():
        # Resolve from vault folder
        vault_folder = job.get("vault_folder")
        if vault_folder:
            vault_path = Path(vault_folder)
            # Search for the file in the vault folder
            for f in vault_path.iterdir() if vault_path.exists() else []:
                if f.name == filename:
                    path = f
                    break
            else:
                # Filename in stage_images might differ from vault name — find by stage pattern
                stage_patterns = {"stage1": "stage1", "stage2": "stage2", "final": "final"}
                pattern = stage_patterns.get(stage, stage)
                for f in vault_path.iterdir() if vault_path.exists() else []:
                    if pattern in f.name and f.suffix == ".png" and "original" not in f.name:
                        path = f
                        break

    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk.")
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
    blocking = _any_job_blocking()
    agent = _online_agent()
    # `worker_alive`/`logged_in` keys are kept for the existing UI: they now mean
    # "an agent is connected" and "that agent reports a logged-in ChatGPT session".
    online = agent is not None
    return {
        "blocking_job_id": blocking,
        "worker_alive": online,
        "worker_error": "" if online else "No agent running - start the agent on your PC to generate.",
        "logged_in": bool(agent and agent.get("logged_in")),
        "agent_connected": online,
        "agent_name": agent.get("name") if agent else "",
    }


@app.get("/api/session")
def get_session():
    """Report the connected agent's ChatGPT session state (agent-reported)."""
    agent = _online_agent()
    return {"logged_in": bool(agent and agent.get("logged_in")), "account": "acct1",
            "agent_connected": agent is not None, "agent_name": agent.get("name") if agent else ""}


@app.post("/api/session/login")
def session_login():
    """Sign-in now happens on the designer's PC: they run login.py there. The
    server cannot drive the remote browser, so this just reports guidance."""
    agent = _online_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="No agent running. Start the agent on your PC.")
    return {"ok": True, "message": "Run login.py on the PC where the agent runs to sign in to ChatGPT."}


@app.post("/api/session/confirm")
def session_confirm():
    """Re-report the agent's session state."""
    agent = _online_agent()
    return {"logged_in": bool(agent and agent.get("logged_in"))}


# ---------------------------------------------------------------------------
# OPERATOR: agent download + token surfacing (cookie-authed by the middleware)
# ---------------------------------------------------------------------------

@app.get("/api/my-agent-token")
def my_agent_token(request: Request):
    """Return (creating if needed) the stable agent token for the logged-in user,
    so the UI can show it with a copy button. The token still originates from the
    same store make_agent_token.py uses."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = get_or_create_for_name(user)
    return {"token": token, "name": user, "server_url": str(request.base_url).rstrip("/")}


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
    raise HTTPException(status_code=404, detail="Agent build not found. Run build_agent.bat on a Windows machine and place ArtworkAgent.zip in downloads/.")


# ---------------------------------------------------------------------------
# AGENT API — polled by the designer-PC agent. Bearer-token authenticated.
# Declared BEFORE app.mount() so the static mount does not shadow them.
# ---------------------------------------------------------------------------

def _require_agent(request: Request) -> str:
    """Validate the agent bearer token; return the designer name or raise 401."""
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    name = token_name(token)
    if not name:
        raise HTTPException(status_code=401, detail="Invalid or missing agent token.")
    return name


# Job fields that must never be overwritten by the agent's posted copy — the
# server owns the operator-supplied answers and the claim bookkeeping.
_SERVER_OWNED_FIELDS = {
    "confirmed_text", "choices", "chosen_numbers", "selected_crops",
    "object_color_choices", "aspect_target", "aspect_method",
    "_regenerate", "_regen_template", "_reextract", "_reextract_mode",
    "_reextract_cols", "_reextract_rows", "_reextract_padding", "_recrop_boxes",
    "claimed_by", "claimed_by_name", "claimed_at", "last_progress_at",
}


def _merge_agent_job(job: dict, posted: dict) -> None:
    """Merge the agent's job copy into the authoritative record.

    The agent owns generation-produced fields (stage, images, results, etc.).
    The server owns operator answers and claim bookkeeping — those are never
    overwritten. A cancellation on the server also always wins.
    """
    if job.get("status") == "cancelled":
        return
    for k, v in posted.items():
        if k in _SERVER_OWNED_FIELDS:
            continue
        job[k] = v


@app.post("/api/agent/register")
def agent_register(req: AgentRegisterRequest, request: Request):
    name = _require_agent(request)
    agent_id = uuid.uuid4().hex[:12]
    agents[agent_id] = {
        "id": agent_id, "name": req.agent_name or name,
        "registered_at": time.time(), "last_seen": time.time(),
        "logged_in": req.logged_in,
    }
    print(f"[server] Agent registered: {req.agent_name!r} ({agent_id})")
    return {"agent_id": agent_id, "name": req.agent_name or name}


@app.get("/api/agent/next-job")
def agent_next_job(request: Request, agent_id: str, logged_in: bool = False):
    _require_agent(request)
    a = agents.get(agent_id)
    if a:
        a["last_seen"] = time.time()
        a["logged_in"] = logged_in
    _requeue_stale_jobs()
    # Claim the oldest queued job atomically.
    with _jobs_lock:
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
    for key in ("text_image", "mockup_image"):
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
    if job.get("status") == "cancelled":
        return {"ok": True, "cancelled": True}
    job["last_progress_at"] = time.time()
    if req.job is not None:
        _merge_agent_job(job, req.job)
    if req.stage is not None:
        job["stage"] = req.stage
    if req.stage_label is not None:
        job["stage_label"] = req.stage_label
    if req.active_time is not None:
        job["active_time"] = req.active_time
    a = agents.get(job.get("claimed_by"))
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
    _recompute_aspect_similarity(job)
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


@app.post("/api/agent/job/{job_id}/error")
def agent_error(job_id: str, req: AgentErrorRequest, request: Request):
    _require_agent(request)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "cancelled":
        job["status"] = "failed"
        job["error"] = req.message
        job["finished_at"] = time.time()
        job["awaiting_input"] = False
    if req.session_expired:
        a = agents.get(job.get("claimed_by"))
        if a:
            a["logged_in"] = False
    print(f"[server] Agent error on job {job_id}: {req.message}")
    return {"ok": True}


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


app.mount("/static", StaticFiles(directory="static"), name="static")

"""FastAPI web UI for artwork generation jobs.

Serves a single-page operator interface and exposes JSON API routes.
Generation runs in a dedicated background worker thread to avoid blocking
the async event loop (Playwright sync API is not async-compatible).

Supports multi-turn workflows (Text workflow with operator decisions).
"""

from __future__ import annotations

import os
import threading
import traceback
import time
import uuid
from pathlib import Path
from queue import Queue
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from config.job_options import JOB_OPTIONS, PARAMETERISED_OPTIONS
from config.workflows import TEXT_TURN_0, TEXT_TURN_1, TEXT_TURN_2, TEXT_TURN_3, MOCKUP_REGENERATE, EXTRACT_BOXES, EXTRACT_ARTWORKS, EXTRACT_CONTACT_SHEET, EXTRACT_SINGLE, ARTWORK_REGENERATE
from src.auth import (
    APP_USERNAME, APP_PASSWORD_HASH, verify_password, sign_cookie,
    get_current_user, check_rate_limit, record_failure, record_success,
    COOKIE_NAME, PUBLIC_PATHS, PUBLIC_PREFIXES,
)
from src.browser import launch_context, is_logged_in
from src.extract import crop_boxes, grid_split, validate_crops, ExtractionError
from src.generator import generate, open_chat, send_turn, send_text_turn, get_last_text_reply, get_boxes, extract_artwork_images
from src.postprocess import is_opaque_white_bg, remove_white_background
from src.prompt_builder import build_prompt
from src.vault import save_to_vault

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
job_queue: Queue[str] = Queue()

INPUT_DIR = Path("./input")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

_worker_alive: bool = True
_worker_error: str = ""

# Session state (updated by the worker thread)
_session_logged_in: bool = False
_session_account: str = "acct1"
_session_check_time: float = 0.0
_session_action: str = ""  # "", "check", "login"
_session_result: threading.Event = threading.Event()

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


class TextConfirmRequest(BaseModel):
    text: str


class OpenFolderRequest(BaseModel):
    path: str


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

_resume_event = threading.Event()
# Signal for cancel: worker checks this after resuming
_cancel_flag: dict[str, bool] = {}


def _worker() -> None:
    global _worker_alive, _worker_error, _session_logged_in, _session_check_time

    try:
        context = launch_context("acct1")
        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://chatgpt.com", wait_until="domcontentloaded")
        _session_logged_in = is_logged_in(page)
        if not _session_logged_in:
            print("[worker] WARNING: Not logged in. Use the Sign In button in the UI.")
        else:
            print("[worker] Session OK — logged in.")
    except Exception as exc:
        _worker_alive = False
        _worker_error = str(exc)
        print(f"[worker] FATAL: {exc}")
        traceback.print_exc()
        return

    while True:
        try:
            job_id: str = job_queue.get()

            # Handle session actions
            if job_id == "__session_check__":
                _session_logged_in = is_logged_in(page)
                _session_check_time = time.time()
                _session_result.set()
                continue
            elif job_id == "__session_login__":
                page.goto("https://chatgpt.com", wait_until="domcontentloaded")
                try:
                    page.bring_to_front()
                except Exception:
                    pass
                _session_result.set()
                continue

            job = jobs.get(job_id)
            if job is None:
                continue
            job["status"] = "running"
            job["started_at"] = time.time()
            try:
                if job.get("workflow") == "text":
                    _run_text_workflow(page, job)
                elif job.get("workflow") == "mockup":
                    _run_mockup_workflow(page, job)
                elif job.get("workflow") == "artwork":
                    _run_artwork_workflow(page, job)
                else:
                    _run_legacy_job(page, job)
            except _CancelledError:
                if job["status"] != "cancelled":
                    job["status"] = "cancelled"
                    job["finished_at"] = time.time()
            except Exception as exc:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = time.time()
        except Exception as exc:
            _worker_alive = False
            _worker_error = f"Worker crashed: {exc}"
            print(f"[worker] FATAL: {exc}")
            traceback.print_exc()
            return


class _CancelledError(Exception):
    pass


def _wait_for_resume(job_id: str) -> None:
    """Block until resume event. Raises _CancelledError if job was cancelled."""
    _resume_event.clear()
    _resume_event.wait()
    if _cancel_flag.get(job_id):
        raise _CancelledError()


def _track_start(job: dict) -> None:
    """Mark the start of an active processing step."""
    job["_step_start"] = time.time()


def _track_end(job: dict) -> None:
    """Accumulate active processing time from the last _track_start."""
    start = job.get("_step_start")
    if start:
        job["active_time"] = job.get("active_time", 0.0) + (time.time() - start)
        job["_step_start"] = None


def _run_legacy_job(page: Any, job: dict[str, Any]) -> None:
    try:
        prompt = build_prompt(options=job["options"], params=job["params"], custom_note=job["custom_note"])
        image_paths = [str(INPUT_DIR / f) for f in job["files"]]
        images = generate(page=page, image_paths=image_paths, prompt=prompt, run_id=job["id"])
        originals = list(images)
        processed = [remove_white_background(d) if is_opaque_white_bg(d) else d for d in images]
        output_names = []
        for idx, data in enumerate(processed, 1):
            name = f"{job['id']}_V{idx}.png"
            (OUTPUT_DIR / name).write_bytes(data)
            output_names.append(name)
        job["images"] = output_names
        vault_paths = []
        if job.get("client") and job.get("task_id"):
            vault_paths = save_to_vault(images=processed, originals=originals, client=job["client"], task_id=job["task_id"])
        job["vault_folder"] = str(vault_paths[0].parent) if vault_paths else None
        job["status"] = "done"
        job["finished_at"] = time.time()
    except Exception:
        job["finished_at"] = time.time()
        raise


def _run_text_workflow(page: Any, job: dict[str, Any]) -> None:
    job_id = job["id"]
    client = job["client"]
    task_id = job["task_id"]
    text = job["text"]
    text_image = job.get("text_image", "")

    tpl_turn1 = job.get("template_turn1") or TEXT_TURN_1
    tpl_turn2 = job.get("template_turn2") or TEXT_TURN_2
    tpl_turn3 = job.get("template_turn3") or TEXT_TURN_3

    from src.vault import VAULT_DIR
    task_dir = VAULT_DIR / client / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_number = len(existing_runs) + 1
    run_dir = task_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    job["vault_folder"] = str(run_dir)

    _track_start(job)
    open_chat(page)
    _track_end(job)

    # --- TURN 0 (optional): Extract text from image ---
    if text_image and not text:
        job["stage"] = 0
        job["stage_label"] = "Reading text from image"
        prompt0 = TEXT_TURN_0
        extracted_text = send_text_turn(page, prompt=prompt0, image_paths=[str(INPUT_DIR / text_image)], run_id=f"{job_id}_t0")
        job.setdefault("prompts", []).append(prompt0)
        job["extracted_text"] = extracted_text
        job["status"] = "awaiting_text_confirmation"
        job["awaiting_input"] = True
        job["paused_at"] = time.time()
        _wait_for_resume(job_id)
        text = job["confirmed_text"]
        job["text"] = text
        job["status"] = "running"
        job["awaiting_input"] = False

    # --- TURN 1: Style variations (with regenerate loop) ---
    job["stage"] = 1
    job["stage_label"] = "Generating style variations"
    attempt = 1

    while True:
        prompt1 = tpl_turn1.format(text=text)
        job.setdefault("prompts", []).append(prompt1)
        _track_start(job)
        images1 = send_turn(page, prompt=prompt1, image_paths=None, run_id=f"{job_id}_t1_{attempt}")
        _track_end(job)

        if images1:
            suffix = f"_attempt{attempt}" if attempt > 1 else ""
            stage1_output = f"{job_id}_stage1{suffix}.png"
            (OUTPUT_DIR / stage1_output).write_bytes(images1[0])
            job.setdefault("stage_images", {})["stage1"] = stage1_output
            job["images"] = [stage1_output]
            vault_name = f"{task_id}_R{run_number}_stage1_styles{suffix}.png"
            (run_dir / vault_name).write_bytes(images1[0])

        job["stage"] = 1
        job["awaiting_input"] = True
        job["paused_at"] = time.time()
        job["template_turn2"] = job.get("template_turn2") or TEXT_TURN_2
        job["status"] = "awaiting_selection"
        _wait_for_resume(job_id)

        if job.get("_regenerate"):
            job["_regenerate"] = False
            tpl_turn1 = job.get("_regen_template") or tpl_turn1
            job["_regen_template"] = None
            attempt += 1
            job["status"] = "running"
            job["awaiting_input"] = False
            job["stage_label"] = f"Regenerating style variations (attempt {attempt})"
            continue
        break

    # --- TURN 2: Colour variations (with regenerate loop) ---
    job["status"] = "running"
    job["awaiting_input"] = False
    job["stage"] = 2
    job["stage_label"] = "Generating colour variations"
    style_choice = job["choices"][-1]  # last choice for stage 1
    tpl_turn2 = job.get("template_turn2") or tpl_turn2
    attempt = 1

    while True:
        prompt2 = tpl_turn2.format(n=style_choice)
        job.setdefault("prompts", []).append(prompt2)
        _track_start(job)
        images2 = send_turn(page, prompt=prompt2, image_paths=None, run_id=f"{job_id}_t2_{attempt}")
        _track_end(job)

        if images2:
            suffix = f"_attempt{attempt}" if attempt > 1 else ""
            stage2_output = f"{job_id}_stage2{suffix}.png"
            (OUTPUT_DIR / stage2_output).write_bytes(images2[0])
            job.setdefault("stage_images", {})["stage2"] = stage2_output
            job["images"] = [stage2_output]
            vault_name = f"{task_id}_R{run_number}_stage2_colours{suffix}.png"
            (run_dir / vault_name).write_bytes(images2[0])

        job["stage"] = 2
        job["awaiting_input"] = True
        job["paused_at"] = time.time()
        job["template_turn3"] = job.get("template_turn3") or TEXT_TURN_3
        job["status"] = "awaiting_selection"
        _wait_for_resume(job_id)

        if job.get("_regenerate"):
            job["_regenerate"] = False
            tpl_turn2 = job.get("_regen_template") or tpl_turn2
            job["_regen_template"] = None
            attempt += 1
            job["status"] = "running"
            job["awaiting_input"] = False
            job["stage_label"] = f"Regenerating colour variations (attempt {attempt})"
            continue
        break

    # --- TURN 3: Final artwork ---
    job["status"] = "running"
    job["awaiting_input"] = False
    job["stage"] = 3
    job["stage_label"] = "Generating final artwork"
    colour_choice = job["choices"][-1]  # last choice for stage 2
    tpl_turn3 = job.get("template_turn3") or tpl_turn3

    prompt3 = tpl_turn3.format(m=colour_choice)
    job.setdefault("prompts", []).append(prompt3)
    _track_start(job)
    images3 = send_turn(page, prompt=prompt3, image_paths=None, run_id=f"{job_id}_t3")
    _track_end(job)

    if images3:
        final_data = images3[0]
        original_data = final_data
        if is_opaque_white_bg(final_data):
            final_data = remove_white_background(final_data)
        final_output = f"{job_id}_final.png"
        (OUTPUT_DIR / final_output).write_bytes(final_data)
        job.setdefault("stage_images", {})["final"] = final_output
        job["images"] = [final_output]
        (run_dir / f"{task_id}_R{run_number}_final.png").write_bytes(final_data)
        (run_dir / f"{task_id}_R{run_number}_final_original.png").write_bytes(original_data)

    job["status"] = "done"
    job["awaiting_input"] = False
    job["finished_at"] = time.time()


def _run_mockup_workflow(page: Any, job: dict[str, Any]) -> None:
    """Mockup workflow: contact sheet → operator picks numbers → generate chosen designs."""
    job_id = job["id"]
    client = job["client"]
    task_id = job["task_id"]
    mockup_image = job.get("mockup_image", "")

    from src.vault import VAULT_DIR
    task_dir = VAULT_DIR / client / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_number = len(existing_runs) + 1
    run_dir = task_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    job["vault_folder"] = str(run_dir)

    image_path = INPUT_DIR / mockup_image

    # --- TURN 1: Contact sheet ---
    job["stage"] = 1
    job["stage_label"] = "Generating numbered contact sheet..."

    _track_start(job)
    open_chat(page)

    extract_tpl = job.get("template_extract") or EXTRACT_CONTACT_SHEET
    job.setdefault("prompts", []).append(extract_tpl)

    images1 = send_turn(page, prompt=extract_tpl, image_paths=[str(image_path)], run_id=f"{job_id}_contact")
    _track_end(job)

    if images1:
        contact_output = f"{job_id}_contact_sheet.png"
        (OUTPUT_DIR / contact_output).write_bytes(images1[0])
        job.setdefault("stage_images", {})["contact_sheet"] = contact_output
        # Save to vault
        (run_dir / f"{task_id}_R{run_number}_contact_sheet.png").write_bytes(images1[0])

    # PAUSE: operator picks numbers
    job["stage"] = 1
    job["stage_label"] = "Send contact sheet to client"
    job["awaiting_input"] = True
    job["paused_at"] = time.time()
    job["status"] = "awaiting_number_selection"
    print(f"[worker] Job {job_id} awaiting number selection.")

    _wait_for_resume(job_id)

    # --- TURNS: Generate each chosen design ---
    job["status"] = "running"
    job["awaiting_input"] = False
    job["stage"] = 2
    chosen_numbers: list[int] = job.get("chosen_numbers", [])
    total = len(chosen_numbers)
    job["stage_label"] = f"Generating {total} design(s)"

    regen_tpl = job.get("template_regen") or EXTRACT_SINGLE
    final_names: list[str] = []

    for i, n in enumerate(chosen_numbers, 1):
        job["stage_label"] = f"Generating design {i} of {total} (#{n})"
        prompt = regen_tpl.format(n=n)
        job.setdefault("prompts", []).append(prompt)

        try:
            _track_start(job)
            result_images = send_turn(page, prompt=prompt, image_paths=None, run_id=f"{job_id}_design_{n}")
            _track_end(job)

            if result_images:
                final_data = result_images[0]
                original_data = final_data
                if is_opaque_white_bg(final_data):
                    final_data = remove_white_background(final_data)

                final_name = f"{job_id}_design_{n}.png"
                (OUTPUT_DIR / final_name).write_bytes(final_data)
                final_names.append(final_name)

                (run_dir / f"{task_id}_R{run_number}_design_{n}.png").write_bytes(final_data)
                (run_dir / f"{task_id}_R{run_number}_design_{n}_original.png").write_bytes(original_data)
        except Exception as exc:
            _track_end(job)
            print(f"[worker] Generation failed for design #{n}: {exc}")

    job.setdefault("stage_images", {})["finals"] = final_names
    job["final_names"] = final_names
    job["status"] = "done"
    job["awaiting_input"] = False
    job["finished_at"] = time.time()


def _run_artwork_workflow(page: Any, job: dict[str, Any]) -> None:
    """Artwork Generation workflow: clean up client-supplied artwork files for DTF printing."""
    job_id = job["id"]
    client = job["client"]
    task_id = job["task_id"]
    artwork_files: list[str] = job.get("artwork_files", [])

    from src.vault import VAULT_DIR
    task_dir = VAULT_DIR / client / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_number = len(existing_runs) + 1
    run_dir = task_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    job["vault_folder"] = str(run_dir)

    regen_tpl = job.get("template_regen") or ARTWORK_REGENERATE
    total = len(artwork_files)
    job["stage"] = 1
    job["stage_label"] = f"Regenerating {total} artwork(s)"

    _track_start(job)
    open_chat(page)
    _track_end(job)

    final_names: list[str] = []
    errors: list[str] = []

    for i, filename in enumerate(artwork_files, 1):
        job["stage_label"] = f"Regenerating {i} of {total}"
        file_path = str(INPUT_DIR / filename)

        try:
            _track_start(job)
            result_images = send_turn(
                page,
                prompt=regen_tpl,
                image_paths=[file_path],
                run_id=f"{job_id}_art_{i}",
            )
            _track_end(job)

            if result_images:
                final_data = result_images[0]
                original_data = final_data
                if is_opaque_white_bg(final_data):
                    final_data = remove_white_background(final_data)

                final_name = f"{job_id}_final_{i}.png"
                (OUTPUT_DIR / final_name).write_bytes(final_data)
                final_names.append(final_name)

                (run_dir / f"{task_id}_R{run_number}_final_{i}.png").write_bytes(final_data)
                (run_dir / f"{task_id}_R{run_number}_final_{i}_original.png").write_bytes(original_data)
            else:
                errors.append(f"File {i} ({filename}): no image returned")
        except Exception as exc:
            _track_end(job)
            errors.append(f"File {i} ({filename}): {exc}")
            print(f"[worker] Artwork regen failed for {filename}: {exc}")

    job["final_names"] = final_names
    job["artwork_files"] = artwork_files
    job["artwork_errors"] = errors
    job.setdefault("prompts", []).append(regen_tpl)

    if errors and not final_names:
        job["status"] = "failed"
        job["error"] = "; ".join(errors)
    elif errors:
        job["status"] = "done_with_errors"
    else:
        job["status"] = "done"

    job["awaiting_input"] = False
    job["finished_at"] = time.time()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()

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
        if j.get("status") in ("awaiting_selection", "awaiting_text_confirmation", "awaiting_crop_review", "awaiting_multi_selection", "awaiting_number_selection", "running"):
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
    return {"turn1": TEXT_TURN_1, "turn2": TEXT_TURN_2, "turn3": TEXT_TURN_3, "extract": EXTRACT_CONTACT_SHEET, "regen": EXTRACT_SINGLE, "artwork_regen": ARTWORK_REGENERATE, "artwork": ARTWORK_REGENERATE}


@app.post("/api/generate")
def create_job(req: GenerateRequest):
    print(f"[create_job] workflow={req.workflow!r} mockup_image={req.mockup_image!r} files={req.files} artwork_files={req.artwork_files}")
    if not _worker_alive:
        raise HTTPException(status_code=503, detail=f"Worker is not running: {_worker_error}")
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
        "_reextract": False, "_reextract_mode": "", "_reextract_cols": 4, "_reextract_rows": 2,
        "_recrop_boxes": None,
        "_step_start": None,
    }
    job_queue.put(job_id)
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
    _resume_event.set()
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
    _resume_event.set()
    return {"ok": True}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a running or paused job, releasing the worker."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    old_status = job["status"]
    if old_status not in ("running", "awaiting_selection", "awaiting_text_confirmation", "awaiting_crop_review", "awaiting_multi_selection", "awaiting_number_selection", "queued"):
        raise HTTPException(status_code=400, detail="This job cannot be cancelled.")

    # Immediately mark as cancelled on the authoritative record
    job["status"] = "cancelled"
    job["finished_at"] = time.time()
    job["awaiting_input"] = False

    # Set the cancel flag so the worker exits cleanly when it resumes
    _cancel_flag[job_id] = True

    # Signal the resume event to unblock the worker if it's waiting
    if old_status in ("awaiting_selection", "awaiting_text_confirmation", "awaiting_crop_review", "awaiting_multi_selection", "awaiting_number_selection"):
        _resume_event.set()

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
    _resume_event.set()
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
        _resume_event.set()
    elif req.action == "reextract":
        job["_reextract"] = True
        job["_reextract_mode"] = req.mode
        job["_reextract_cols"] = req.cols
        job["_reextract_rows"] = req.rows
        job["_reextract_padding"] = req.padding
        job["_recrop_boxes"] = None
        _resume_event.set()
    elif req.action == "recrop":
        # Operator manually adjusted boxes — recrop locally without ChatGPT
        if not req.boxes:
            raise HTTPException(status_code=400, detail="No boxes provided for recrop.")
        job["_reextract"] = True
        job["_reextract_mode"] = "manual_boxes"
        job["_recrop_boxes"] = req.boxes
        job["_reextract_padding"] = req.padding
        _resume_event.set()
    elif req.action == "redetect":
        # Re-run ChatGPT detection in a fresh chat
        job["_reextract"] = True
        job["_reextract_mode"] = "auto"
        job["_reextract_padding"] = req.padding
        job["_recrop_boxes"] = None
        _resume_event.set()
    else:
        raise HTTPException(status_code=400, detail="Action must be 'accept', 'reextract', 'recrop', or 'redetect'.")
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
    _resume_event.set()
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
    _resume_event.set()
    return {"ok": True, "numbers": req.numbers}


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
    blocking = _any_job_blocking()
    return {"blocking_job_id": blocking, "worker_alive": _worker_alive, "worker_error": _worker_error, "logged_in": _session_logged_in}


@app.get("/api/session")
def get_session():
    """Check if the browser session is logged in. Uses cached result if recent."""
    if time.time() - _session_check_time > 30:
        # Ask worker to re-check (non-blocking if worker is busy)
        _session_result.clear()
        job_queue.put("__session_check__")
        _session_result.wait(timeout=10)
    return {"logged_in": _session_logged_in, "account": _session_account}


@app.post("/api/session/login")
def session_login():
    """Navigate the worker's browser to ChatGPT login page and bring to front."""
    if not _worker_alive:
        raise HTTPException(status_code=503, detail="Worker is not running.")
    _session_result.clear()
    job_queue.put("__session_login__")
    _session_result.wait(timeout=15)
    return {"ok": True}


@app.post("/api/session/confirm")
def session_confirm():
    """Re-check login status after the operator logged in manually."""
    _session_result.clear()
    job_queue.put("__session_check__")
    _session_result.wait(timeout=10)
    return {"logged_in": _session_logged_in}


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/input", StaticFiles(directory="input"), name="input")

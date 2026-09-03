"""FastAPI web UI for artwork generation jobs.

Serves a single-page operator interface and exposes JSON API routes.
Generation runs in a dedicated background worker thread to avoid blocking
the async event loop (Playwright sync API is not async-compatible).

Supports multi-turn workflows (Text workflow with operator decisions).
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
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
from config.workflows import TEXT_TURN_0, TEXT_TURN_1, TEXT_TURN_2, TEXT_TURN_3, MOCKUP_REGENERATE, EXTRACT_BOXES, EXTRACT_ARTWORKS, EXTRACT_CONTACT_SHEET, EXTRACT_SINGLE, ARTWORK_REGENERATE, CUSTOM_RECONSTRUCT, CUSTOM_REMOVE_BACKGROUND, CUSTOM_HALO_REMOVAL, CUSTOM_BLACK_OUT, CUSTOM_HALF_TONE, CUSTOM_DETECT_OBJECTS, CUSTOM_CHANGE_COLOR, CUSTOM_ASPECT_ADVICE, CUSTOM_ASPECT_BASELINE, CUSTOM_ASPECT_REGENERATE, normalise_ratio, CUSTOM_OPERATIONS_ORDER, CUSTOM_OPERATIONS_LABELS, CUSTOM_OPERATIONS
from src.aspect import image_info, fit_to_ratio
from src.postprocess import black_out, half_tone, similarity, extract_features
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
from src import nextcloud as nc
from src.nc_live import watcher as nc_watcher

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
# Broader than ALLOWED_EXTENSIONS: these are shown in the per-customer artwork
# grid (Nextcloud renders previews for them) even though only the four above can
# be sent into a workflow.
VAULT_IMAGE_EXTENSIONS = ALLOWED_EXTENSIONS | {
    ".gif", ".bmp", ".tif", ".tiff", ".heic", ".avif"}
# Where "Save to Artwork Vault" puts generated files inside each customer folder.
VAULT_SAVE_SUBFOLDER = "AI Artwork"

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


class NextcloudImportRequest(BaseModel):
    # Vault paths to pull into ./input, and which workflow they are headed for.
    paths: list[str] = []
    target: str = "artwork"


class NextcloudSaveRequest(BaseModel):
    # Save a generated output file back into the vault, under the chosen
    # customer folder. `name` is a file in ./output; `customer` is the vault
    # customer folder (path or bare name); `filename` overrides the saved name.
    name: str = ""
    customer: str = ""
    filename: str = ""
    overwrite: bool = False


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
                elif job.get("workflow") == "custom":
                    _run_custom_workflow(page, job)
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

    # The colour collage is a fixed 4-wide grid, numbered left-to-right then
    # top-to-bottom. Derive the row/position so the prompt names the exact cell
    # and ChatGPT cannot miscount its own collage (e.g. #8 -> row 2, position 4).
    COLOUR_GRID_COLS = 4
    try:
        m_int = int(colour_choice)
    except (TypeError, ValueError):
        m_int = 0
    if m_int >= 1:
        row = ((m_int - 1) // COLOUR_GRID_COLS) + 1
        col = ((m_int - 1) % COLOUR_GRID_COLS) + 1
    else:
        row = col = 0

    # Substitute {m}/{row}/{col}; tolerate operator-edited templates missing keys.
    prompt3 = tpl_turn3.replace("{m}", str(colour_choice)).replace("{row}", str(row)).replace("{col}", str(col))
    print(f"[text] Turn 3 final prompt (m={colour_choice}, row={row}, col={col}):\n{prompt3}")
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


def _run_custom_workflow(page: Any, job: dict[str, Any]) -> None:
    """Custom Operation workflow: apply selected operations in fixed order."""
    job_id = job["id"]
    client = job["client"]
    task_id = job["task_id"]
    artwork_file = job.get("artwork_files", [""])[0]
    operations = job.get("custom_operations", [])
    custom_prompts = job.get("custom_prompts", {})

    from src.vault import VAULT_DIR
    task_dir = VAULT_DIR / client / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_number = len(existing_runs) + 1
    run_dir = task_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    job["vault_folder"] = str(run_dir)

    # Sort operations into fixed execution order
    ordered_ops = [op for op in CUSTOM_OPERATIONS_ORDER if op in operations]
    total = len(ordered_ops)
    job["stage_label"] = f"Running {total} operation(s)"

    # Default prompts for each operation
    default_prompts = {
        "reconstruct": CUSTOM_RECONSTRUCT,
        "remove_background": CUSTOM_REMOVE_BACKGROUND,
        "halo_removal": CUSTOM_HALO_REMOVAL,
        # black_out and half_tone are deterministic LOCAL ops (src/postprocess.py) — no prompt.
        # change_object_color is handled separately (two turns, two templates); not a single-turn default.
    }

    _track_start(job)
    open_chat(page)
    _track_end(job)

    # The current reference image path — starts as the uploaded file, updated after each step
    current_image_path = str(INPUT_DIR / artwork_file)
    if not artwork_file or not Path(current_image_path).exists():
        job["status"] = "failed"
        job["error"] = f"Input file not found: {artwork_file!r}"
        job["finished_at"] = time.time()
        return

    steps_done: list[dict] = []
    errors: list[str] = []

    for i, op in enumerate(ordered_ops, 1):
        job["stage"] = i
        label = CUSTOM_OPERATIONS_LABELS.get(op, op)
        job["stage_label"] = f"Step {i} of {total} \u2014 {label}"
        print(f"[custom] Step {i}/{total}: {op} | input: {current_image_path}")

        if op == "change_object_color":
            # TWO-TURN operation with TWO distinct templates/keys, kept separate end to end.
            #   change_object_detect -> Step A (list objects, text reply)
            #   change_object_apply  -> Step B (apply colours, {changes}, image reply)
            # Turn A: detect objects (text reply)
            detect_prompt = custom_prompts.get("change_object_detect", CUSTOM_DETECT_OBJECTS)
            print(f"[custom] Turn A (detect) prompt [:200]:\n{detect_prompt[:200]}")
            job.setdefault("prompts", []).append(detect_prompt)

            try:
                _track_start(job)
                detected_text = send_text_turn(page, prompt=detect_prompt, image_paths=[current_image_path], run_id=f"{job_id}_detect_obj")
                _track_end(job)
            except Exception as exc:
                _track_end(job)
                errors.append(f"Step {i} ({label}) detect: {exc}")
                break

            job["detected_objects"] = detected_text

            # PAUSE for operator to assign colours
            job["awaiting_input"] = True
            job["paused_at"] = time.time()
            job["status"] = "awaiting_object_selection"
            print(f"[worker] Job {job_id} awaiting object colour selection.")
            _wait_for_resume(job_id)

            # Turn B: apply colour changes
            job["status"] = "running"
            job["awaiting_input"] = False
            job["stage_label"] = f"Step {i} of {total} \u2014 Applying colour changes"

            object_choices = job.get("object_color_choices", [])
            changes_text = "\n".join(f"- {c['object']} \u2192 {c['color']}" for c in object_choices)
            print(f"[custom] Change colour changes_text:\n{changes_text}")
            # Turn B uses its OWN template (change_object_apply), never the detection one.
            apply_template = custom_prompts.get("change_object_apply", CUSTOM_CHANGE_COLOR)
            if "{changes}" not in apply_template:
                # Operator edited out the placeholder; append the changes so they are never lost.
                apply_template = apply_template + "\n\n{changes}"
            color_prompt = apply_template.format(changes=changes_text)
            print(f"[custom] Turn B (apply) prompt [:200]:\n{color_prompt[:200]}")
            job.setdefault("prompts", []).append(color_prompt)

            try:
                _track_start(job)
                result_images = send_turn(page, prompt=color_prompt, image_paths=[current_image_path], run_id=f"{job_id}_color_{i}")
                _track_end(job)
            except Exception as exc:
                _track_end(job)
                errors.append(f"Step {i} ({label}) color: {exc}")
                break

            if result_images:
                step_name = f"{job_id}_step{i}_{op}.png"
                final_data = result_images[0]
                if is_opaque_white_bg(final_data):
                    final_data = remove_white_background(final_data)
                (OUTPUT_DIR / step_name).write_bytes(final_data)
                (run_dir / f"{task_id}_R{run_number}_step{i}_{op}.png").write_bytes(final_data)
                current_image_path = str(OUTPUT_DIR / step_name)
                steps_done.append({"step": i, "op": op, "label": label, "file": step_name, "prompt": color_prompt})
            else:
                errors.append(f"Step {i} ({label}): colour change returned no image")
                break

        elif op == "aspect_ratio":
            # Aspect Ratio Enhancement: TWO ChatGPT regenerations.
            #   Turn A - regenerate at CURRENT ratio, background removed -> BASELINE.
            #   Turn B - recommend target ratios (text).
            #   PAUSE  - operator picks target ratio + method.
            #   Turn C - regenerate at target ratio, working from the baseline.
            # Similarity compares the RESULT against the BASELINE (both are
            # background-free), never against the opaque original upload.
            info = image_info(current_image_path, dpi_override=job.get("aspect_dpi") or None)
            job["aspect_info"] = info
            # Record the ORIGINAL upload for the results display.
            job["aspect_original_file"] = Path(current_image_path).name
            try:
                job["aspect_original_features"] = extract_features(Path(current_image_path).read_bytes())
            except Exception as exc:
                print(f"[custom] original feature extraction failed: {exc}")

            # --- Turn A: baseline (clean, current ratio, background removed) ---
            job["stage_label"] = f"Step {i} of {total} \u2014 Cleaning artwork (baseline)"
            baseline_prompt = CUSTOM_ASPECT_BASELINE
            print(f"[custom] Aspect baseline prompt [:200]:\n{baseline_prompt[:200]}")
            job.setdefault("prompts", []).append(baseline_prompt)
            try:
                _track_start(job)
                baseline_images = send_turn(page, prompt=baseline_prompt, image_paths=[current_image_path], run_id=f"{job_id}_aspect_baseline")
                _track_end(job)
            except Exception as exc:
                _track_end(job)
                errors.append(f"Step {i} ({label}) baseline: {exc}")
                break
            if not baseline_images:
                errors.append(f"Step {i} ({label}): baseline regeneration returned no image")
                break
            baseline_data = baseline_images[0]
            if is_opaque_white_bg(baseline_data):
                baseline_data = remove_white_background(baseline_data)
            baseline_name = f"{job_id}_step{i}_aspect_baseline.png"
            (OUTPUT_DIR / baseline_name).write_bytes(baseline_data)
            (run_dir / f"{task_id}_R{run_number}_step{i}_aspect_baseline.png").write_bytes(baseline_data)
            baseline_path = str(OUTPUT_DIR / baseline_name)
            try:
                job["aspect_baseline_features"] = extract_features(baseline_data)
            except Exception as exc:
                print(f"[custom] baseline feature extraction failed: {exc}")
            job["aspect_baseline_file"] = baseline_name
            # Baseline is a shown stage in its own right.
            steps_done.append({"step": i, "op": op, "label": "Baseline (cleaned, current ratio)",
                               "file": baseline_name, "prompt": baseline_prompt, "method": "baseline"})

            # --- Turn B: recommend target ratios (text), based on the baseline ---
            b_info = image_info(baseline_path, dpi_override=job.get("aspect_dpi") or None)
            advice_tpl = custom_prompts.get("aspect_ratio", CUSTOM_ASPECT_ADVICE)
            advice_prompt = advice_tpl.format(
                width=b_info["width"], height=b_info["height"], ratio=b_info["ratio"],
                inches_w=b_info["inches_w"], inches_h=b_info["inches_h"], dpi=b_info["dpi"],
            )
            print(f"[custom] Aspect advice prompt [:200]:\n{advice_prompt[:200]}")
            job.setdefault("prompts", []).append(advice_prompt)
            try:
                _track_start(job)
                advice_text = send_text_turn(page, prompt=advice_prompt, image_paths=[baseline_path], run_id=f"{job_id}_aspect_advice")
                _track_end(job)
            except Exception as exc:
                _track_end(job)
                errors.append(f"Step {i} ({label}) advice: {exc}")
                break
            job["aspect_recommendations"] = advice_text

            # --- PAUSE for the operator to pick a target ratio + method ---
            job["awaiting_input"] = True
            job["paused_at"] = time.time()
            job["status"] = "awaiting_ratio_selection"
            print(f"[worker] Job {job_id} awaiting aspect ratio selection.")
            _wait_for_resume(job_id)

            job["status"] = "running"
            job["awaiting_input"] = False

            target = job.get("aspect_target")  # {"w": float, "h": float}
            method = (job.get("aspect_method") or "pad").lower()
            if not target or target.get("w", 0) <= 0 or target.get("h", 0) <= 0:
                errors.append(f"Step {i} ({label}): no valid target ratio selected")
                break

            tw, th = float(target["w"]), float(target["h"])
            baseline_bytes = Path(baseline_path).read_bytes()
            step_name = f"{job_id}_step{i}_{op}.png"

            if method == "regenerate":
                # --- Turn C: regenerate at the target ratio, FROM the baseline ---
                job["stage_label"] = f"Step {i} of {total} \u2014 Regenerating at target ratio"
                dpi = (b_info.get("dpi") if b_info else None) or 300
                ratio_str = normalise_ratio(tw, th)
                if tw >= th:
                    in_w = round(max(b_info.get("inches_w", 0), b_info.get("inches_h", 0)) or tw, 2)
                    in_h = round(in_w * (th / tw), 2)
                else:
                    in_h = round(max(b_info.get("inches_w", 0), b_info.get("inches_h", 0)) or th, 2)
                    in_w = round(in_h * (tw / th), 2)
                regen_prompt = CUSTOM_ASPECT_REGENERATE.format(
                    ratio=ratio_str, inches_w=in_w, inches_h=in_h, dpi=dpi,
                )
                print(f"[custom] Aspect target prompt [:200]:\n{regen_prompt[:200]}")
                job.setdefault("prompts", []).append(regen_prompt)
                try:
                    _track_start(job)
                    regen_images = send_turn(page, prompt=regen_prompt, image_paths=[baseline_path], run_id=f"{job_id}_aspect_target")
                    _track_end(job)
                except Exception as exc:
                    _track_end(job)
                    errors.append(f"Step {i} ({label}) regenerate: {exc}")
                    break
                if not regen_images:
                    errors.append(f"Step {i} ({label}): regeneration returned no image")
                    break
                gen_data = regen_images[0]
                if is_opaque_white_bg(gen_data):
                    gen_data = remove_white_background(gen_data)
                (OUTPUT_DIR / step_name).write_bytes(gen_data)
                (run_dir / f"{task_id}_R{run_number}_step{i}_{op}.png").write_bytes(gen_data)
                # Similarity: RESULT vs BASELINE (both background-free) — like for like.
                try:
                    sim = similarity(baseline_bytes, gen_data)
                except Exception as exc:
                    print(f"[custom] similarity failed: {exc}")
                    sim = None
                job["aspect_similarity"] = sim
                steps_done.append({"step": i, "op": op, "label": "Final (target ratio)",
                                   "file": step_name, "prompt": regen_prompt, "method": "regenerate",
                                   "compare_against": "baseline", "similarity": sim})
                current_image_path = str(OUTPUT_DIR / step_name)
            else:
                # PAD the BASELINE locally — pixels preserved exactly, no comparison.
                job["stage_label"] = f"Step {i} of {total} \u2014 Padding baseline to target ratio"
                try:
                    padded = fit_to_ratio(baseline_bytes, tw, th)
                except Exception as exc:
                    errors.append(f"Step {i} ({label}): padding failed: {exc}")
                    break
                (OUTPUT_DIR / step_name).write_bytes(padded)
                (run_dir / f"{task_id}_R{run_number}_step{i}_{op}.png").write_bytes(padded)
                current_image_path = str(OUTPUT_DIR / step_name)
                steps_done.append({"step": i, "op": op, "label": "Final (target ratio)",
                                   "file": step_name, "method": "pad"})

        elif op in ("black_out", "half_tone"):
            # DETERMINISTIC LOCAL operations — no ChatGPT turn, no prompt, instant.
            # The model would redraw the artwork and lose the exact silhouette;
            # these transform the actual pixels instead.
            job["stage_label"] = f"Step {i} of {total} \u2014 {label} (local)"
            try:
                src_bytes = Path(current_image_path).read_bytes()
                if op == "black_out":
                    # No opaque-background guard: Black Out drops black pixels and
                    # works fine on artwork that still has a (non-black) background.
                    bo = job.get("blackout_settings") or {}
                    result_bytes = black_out(src_bytes, threshold=int(bo.get("threshold", 40)))
                else:
                    # Half Tone needs transparency to work against — an opaque
                    # background gives nothing to reproduce. Fail clearly.
                    if is_opaque_white_bg(src_bytes):
                        errors.append(f"Step {i} ({label}): Input has an opaque background - run Remove Background first")
                        break
                    ht = job.get("halftone_settings") or {}
                    result_bytes = half_tone(
                        src_bytes,
                        lpi=float(ht.get("lpi", 40)),
                        angle=float(ht.get("angle", 22.5)),
                        dpi=float(ht.get("dpi", 300)),
                        dot=str(ht.get("dot", "round")),
                    )
            except Exception as exc:
                errors.append(f"Step {i} ({label}): {exc}")
                break

            step_name = f"{job_id}_step{i}_{op}.png"
            (OUTPUT_DIR / step_name).write_bytes(result_bytes)
            (run_dir / f"{task_id}_R{run_number}_step{i}_{op}.png").write_bytes(result_bytes)
            current_image_path = str(OUTPUT_DIR / step_name)
            steps_done.append({"step": i, "op": op, "label": label, "file": step_name})

        else:
            # Single-turn operation
            prompt = custom_prompts.get(op, default_prompts.get(op, ""))
            job.setdefault("prompts", []).append(prompt)

            try:
                _track_start(job)
                result_images = send_turn(page, prompt=prompt, image_paths=[current_image_path], run_id=f"{job_id}_{op}_{i}")
                _track_end(job)
            except Exception as exc:
                _track_end(job)
                errors.append(f"Step {i} ({label}): {exc}")
                break

            if result_images:
                step_name = f"{job_id}_step{i}_{op}.png"
                final_data = result_images[0]
                if op in ("remove_background", "halo_removal") and is_opaque_white_bg(final_data):
                    final_data = remove_white_background(final_data)
                (OUTPUT_DIR / step_name).write_bytes(final_data)
                (run_dir / f"{task_id}_R{run_number}_step{i}_{op}.png").write_bytes(final_data)
                current_image_path = str(OUTPUT_DIR / step_name)
                steps_done.append({"step": i, "op": op, "label": label, "file": step_name, "prompt": prompt})
            else:
                errors.append(f"Step {i} ({label}): no image returned")
                break

    # Save final
    if steps_done:
        last_file = steps_done[-1]["file"]
        final_name = f"{job_id}_final.png"
        import shutil
        shutil.copy2(str(OUTPUT_DIR / last_file), str(OUTPUT_DIR / final_name))
        (run_dir / f"{task_id}_R{run_number}_final.png").write_bytes((OUTPUT_DIR / last_file).read_bytes())
        job["final_names"] = [final_name]

    job["custom_steps_done"] = steps_done
    job["artwork_errors"] = errors

    if errors and not steps_done:
        job["status"] = "failed"
        job["error"] = "; ".join(errors)
    elif errors:
        job["status"] = "done_with_errors"
    elif len(steps_done) < total:
        # Some operations produced no output without raising an error
        job["status"] = "failed"
        job["error"] = f"Only {len(steps_done)} of {total} operations produced output. Check the server logs."
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
    if old_status not in ("running", "awaiting_selection", "awaiting_text_confirmation", "awaiting_crop_review", "awaiting_multi_selection", "awaiting_number_selection", "awaiting_object_selection", "awaiting_ratio_selection", "queued"):
        raise HTTPException(status_code=400, detail="This job cannot be cancelled.")

    # Immediately mark as cancelled on the authoritative record
    job["status"] = "cancelled"
    job["finished_at"] = time.time()
    job["awaiting_input"] = False

    # Set the cancel flag so the worker exits cleanly when it resumes
    _cancel_flag[job_id] = True

    # Signal the resume event to unblock the worker if it's waiting
    if old_status in ("awaiting_selection", "awaiting_text_confirmation", "awaiting_crop_review", "awaiting_multi_selection", "awaiting_number_selection", "awaiting_object_selection", "awaiting_ratio_selection"):
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
    _resume_event.set()
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
    _resume_event.set()
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
    "text": {"label": "Text (as text image)", "multiple": False},
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


# The watcher runs whether or not anyone is looking at the Nextcloud tab, so a
# file that lands while the operator is mid-job is already in the feed when
# they switch to it.
nc_watcher.start()


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/input", StaticFiles(directory="input"), name="input")

"""Client-side worker agent.

Runs on the DESIGNER'S PC, where a real logged-in Chrome/ChatGPT session lives.
It polls the server, claims queued jobs, runs the SAME workflow functions the
server used to run in-process, and posts progress/results back. The server
never calls the agent — so no open ports or firewall changes are needed here.

Only the ORCHESTRATION lives here. The workflow bodies below are the exact
functions that previously ran inside app.py, unchanged, reused as-is.

Setup: see SETUP.md. Needs a .env beside this file with:
    SERVER_URL=https://your-vps
    AGENT_TOKEN=agt_...
    AGENT_NAME=Some Designer   (optional)
"""
from __future__ import annotations

import io
import os
import sys
import time
import shutil
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# --- Shared, UNCHANGED library code (same modules the server used) ---
from config.workflows import (
    TEXT_TURN_0, TEXT_TURN_1, TEXT_TURN_2, TEXT_TURN_3,
    EXTRACT_CONTACT_SHEET, EXTRACT_SINGLE, ARTWORK_REGENERATE,
    CUSTOM_RECONSTRUCT, CUSTOM_REMOVE_BACKGROUND, CUSTOM_HALO_REMOVAL,
    CUSTOM_BLACK_OUT, CUSTOM_HALF_TONE, CUSTOM_DETECT_OBJECTS, CUSTOM_CHANGE_COLOR,
    CUSTOM_ASPECT_ADVICE, CUSTOM_ASPECT_BASELINE, CUSTOM_ASPECT_REGENERATE,
    normalise_ratio, CUSTOM_OPERATIONS_ORDER, CUSTOM_OPERATIONS_LABELS,
)
from src.aspect import image_info, fit_to_ratio
from src.browser import launch_context, is_logged_in
from src.generator import (
    generate, open_chat, rename_chat, send_turn, send_text_turn,
    get_last_text_reply, get_boxes, extract_artwork_images,
)
from src.postprocess import (
    is_opaque_white_bg, remove_white_background,
    black_out, half_tone,
)
# similarity / extract_features live in src.compare (which needs cv2). The
# packaged agent build EXCLUDES cv2 because the aspect-ratio *comparison* is a
# server-side concern; the agent only produces the images. Import them
# optionally so a cv2-free build still loads — the workflow already wraps every
# call to these in try/except, so the feature degrades gracefully rather than
# breaking a job.
try:
    from src.compare import similarity, extract_features  # noqa: F401
except Exception as _cmp_exc:  # cv2 absent in the packaged agent build
    print(f"[agent] comparison features unavailable (cv2 not bundled): {_cmp_exc}")

    def similarity(*_a, **_k):  # type: ignore
        raise RuntimeError("similarity is server-side only; cv2 not available in the agent build")

    def extract_features(*_a, **_k):  # type: ignore
        raise RuntimeError("extract_features is server-side only; cv2 not available in the agent build")
from src.prompt_builder import build_prompt
from src.vault import save_to_vault
from src import self_update

load_dotenv()

# How often the idle claim loop checks the server for a newer agent build.
UPDATE_CHECK_INTERVAL = 3600.0   # seconds (hourly)

SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
AGENT_TOKEN = os.getenv("AGENT_TOKEN", "")
AGENT_NAME = os.getenv("AGENT_NAME", os.getenv("COMPUTERNAME", "designer"))
POLL_INTERVAL = 3.0          # seconds between next-job polls
SYNC_INTERVAL = 2.0          # seconds between job-state syncs while running
PAUSE_POLL_INTERVAL = 3.0    # seconds between paused-input polls


def configure(server_url: str | None = None, token: str | None = None, name: str | None = None) -> None:
    """Override the connection settings at runtime (used by the GUI wrapper).

    This only changes the module-level connection config and auth header; the
    job loop and workflow logic are untouched.
    """
    global SERVER_URL, AGENT_TOKEN, AGENT_NAME, _HEADERS
    if server_url:
        SERVER_URL = server_url.rstrip("/")
    if token is not None:
        AGENT_TOKEN = token
    if name:
        AGENT_NAME = name
    _HEADERS = {"Authorization": f"Bearer {AGENT_TOKEN}"}

# Local working directories (mirror the server's ./input and ./output layout so
# the unchanged workflow code keeps writing to INPUT_DIR / OUTPUT_DIR).
_BASE = Path(tempfile.gettempdir()) / "artwork_agent"
INPUT_DIR = _BASE / "input"
OUTPUT_DIR = _BASE / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {"Authorization": f"Bearer {AGENT_TOKEN}"}

# Shared state used by the pause/sync machinery. The workflow code mutates a
# module-global `job` dict via the functions below; we track the current job id
# so the sync thread and pause poller know which one to talk to the server about.
_current_job_id: str | None = None
_current_job: dict | None = None
_current_agent_id: str | None = None  # so every progress post carries the agent id (heartbeat)
_sync_stop = threading.Event()

# --- Per-job console capture -------------------------------------------------
# Everything the agent prints while a job runs is buffered here and shipped to
# the server on each progress sync, so a designer can read the log in the web UI
# without opening the agent terminal. Bounded so a long job stays memory-safe.
_LOG_BUFFER_MAX = 400
_log_lock = threading.Lock()
_log_buffer: list[str] = []      # lines not yet sent to the server
_warn_buffer: list[str] = []     # non-fatal warnings not yet sent


class _Tee:
    """Wrap a stream so every line printed is also captured for the current job."""
    def __init__(self, stream):
        self._stream = stream

    def write(self, s):
        try:
            self._stream.write(s)
        except Exception:
            pass
        if s and _current_job_id:
            for line in str(s).splitlines():
                if line.strip() == "":
                    continue
                with _log_lock:
                    _log_buffer.append(line)
                    if len(_log_buffer) > _LOG_BUFFER_MAX:
                        del _log_buffer[:len(_log_buffer) - _LOG_BUFFER_MAX]

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass


# Install the tee once, at import, so all print() output is captured.
sys.stdout = _Tee(sys.stdout)
sys.stderr = _Tee(sys.stderr)


def _drain_log() -> list[str]:
    """Return and clear the buffered console lines pending upload."""
    with _log_lock:
        lines = _log_buffer[:]
        _log_buffer.clear()
        return lines


def _drain_warnings() -> list[str]:
    with _log_lock:
        w = _warn_buffer[:]
        _warn_buffer.clear()
        return w


def report_warning(message: str) -> None:
    """Record a non-fatal problem (a retried nav, a recovered timeout) so the UI
    shows what nearly failed. Also prints it (and thus captures it in the log)."""
    print(f"[warning] {message}")
    with _log_lock:
        _warn_buffer.append(str(message))


def _reset_job_buffers() -> None:
    with _log_lock:
        _log_buffer.clear()
        _warn_buffer.clear()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, **kw) -> requests.Response:
    return requests.post(SERVER_URL + path, headers=_HEADERS, timeout=30, **kw)


def _get(path: str, **kw) -> requests.Response:
    return requests.get(SERVER_URL + path, headers=_HEADERS, timeout=30, **kw)


# ---------------------------------------------------------------------------
# Pause / cancel machinery — same contract as the old in-process worker, but
# the "resume" now comes from the server (the operator answers in the web UI).
# ---------------------------------------------------------------------------

class _CancelledError(Exception):
    pass


# Only the fields the AGENT owns. We deliberately do NOT send status,
# awaiting_input, paused_at, choices, confirmed_text, etc. — those belong to the
# server/operator. Sending the whole dict previously let a stale sync overwrite
# an operator's answer (status flipped back to awaiting_*) and also caused
# stage_images JSON jitter that re-rendered the pause panel. Terminal statuses
# are sent explicitly at the end of a job, not from the periodic sync.
_AGENT_OWNED_FIELDS = (
    "stage", "stage_label", "active_time",
    "images", "stage_images", "prompts",
    "final_names", "custom_steps_done", "artwork_errors",
    "detected_objects", "aspect_recommendations", "aspect_info",
    "aspect_original_file", "aspect_original_features",
    "aspect_baseline_file", "aspect_baseline_features",
    "aspect_similarity", "vault_folder", "crop_names", "crop_count",
    "crop_warnings", "extracted_text",
)


def _agent_job_blob(job: dict, include_status: bool = False) -> dict:
    """Build a sync payload containing ONLY agent-owned fields."""
    blob = {k: job.get(k) for k in _AGENT_OWNED_FIELDS if k in job}
    if include_status:
        # Only used for the final post so the server records the terminal status.
        blob["status"] = job.get("status")
    return blob


def _sync_job() -> None:
    """Push the agent-owned fields of the current job to the server (heartbeat + progress)."""
    if not _current_job_id or _current_job is None:
        return
    try:
        payload = {
            "agent_id": _current_agent_id,  # heartbeat: keep this agent "online" mid-job
            "stage": _current_job.get("stage"),
            "stage_label": _current_job.get("stage_label"),
            "active_time": _current_job.get("active_time"),
            "logged_in": True,
            # Slim blob — agent-owned fields only, never status/awaiting_input.
            "job": _agent_job_blob(_current_job, include_status=False),
            # Console output + warnings since the last sync, for the web UI log.
            "log_lines": _drain_log(),
            "warnings": _drain_warnings(),
        }
        r = _post(f"/api/agent/job/{_current_job_id}/progress", json=payload)
        if r.ok and r.json().get("cancelled"):
            raise _CancelledError()
    except _CancelledError:
        raise
    except Exception as exc:
        # Non-fatal: the next sync will retry. Surface it as a warning so the UI
        # shows a hiccup even though the job recovered.
        report_warning(f"progress sync failed, will retry: {exc}")


def _sync_loop() -> None:
    """Background thread: periodically sync job state while a job runs."""
    while not _sync_stop.is_set():
        try:
            _sync_job()
        except _CancelledError:
            pass  # the workflow thread will observe cancellation at its next pause/turn
        _sync_stop.wait(SYNC_INTERVAL)


# Every field the operator can answer at a pause; copied from paused-input
# back onto the local job so the workflow (unchanged) reads them normally.
_ANSWER_FIELDS = (
    "confirmed_text", "choices", "chosen_numbers", "selected_crops",
    "object_color_choices", "aspect_target", "aspect_method",
    "template_turn2", "template_turn3",
    "_regenerate", "_regen_template",
    "_reextract", "_reextract_mode", "_reextract_cols",
    "_reextract_rows", "_reextract_padding", "_recrop_boxes",
)

# Pause metadata the operator UI needs to render the prompt (agent-owned).
_PAUSE_FIELDS = (
    "extracted_text", "detected_objects", "aspect_recommendations",
    "aspect_info", "aspect_original_file", "aspect_original_features",
    "aspect_baseline_file", "aspect_baseline_features",
    "stage_images", "images", "crop_names", "crop_count", "crop_warnings",
    "prompts", "template_turn2", "template_turn3",
)


def _announce_pause(job_id: str) -> None:
    """Tell the server this job has entered an operator pause.

    The periodic sync no longer sends `status`, so without this the server would
    still think the job is "running" and paused-input would return ready:true
    immediately with an empty answer. We send the local awaiting_* status plus
    the pause metadata; the server applies it only on a running -> awaiting_*
    transition."""
    if _current_job is None:
        return
    pstatus = _current_job.get("status", "")
    if not pstatus.startswith("awaiting_"):
        return
    fields = {k: _current_job.get(k) for k in _PAUSE_FIELDS if k in _current_job}
    try:
        _post(f"/api/agent/job/{job_id}/progress", json={
            "agent_id": _current_agent_id,
            "stage": _current_job.get("stage"),
            "stage_label": _current_job.get("stage_label"),
            "active_time": _current_job.get("active_time"),
            "logged_in": True,
            "pause_status": pstatus,
            "pause_fields": fields,
        })
        print(f"[agent] announced pause {pstatus} for job {job_id}")
    except Exception as exc:
        print(f"[agent] pause announce failed (will retry via poll): {exc}")


def _wait_for_resume(job_id: str) -> None:
    """Block until the operator answers this pause on the server, then copy the
    answer back onto the local job dict. Raises _CancelledError if cancelled.

    This mirrors the old _wait_for_resume(job_id) signature exactly, so the
    workflow bodies below are unchanged.
    """
    # Tell the server we are paused (the slim sync no longer carries status).
    _announce_pause(job_id)
    while True:
        try:
            r = _get(f"/api/agent/job/{job_id}/paused-input")
            if r.status_code == 404:
                raise _CancelledError()
            data = r.json()
        except _CancelledError:
            raise
        except Exception as exc:
            report_warning(f"paused-input poll failed, will retry: {exc}")
            time.sleep(PAUSE_POLL_INTERVAL)
            continue

        if data.get("cancelled"):
            raise _CancelledError()
        if data.get("ready"):
            # Copy EVERY operator answer field onto the local job dict so the
            # unchanged workflow code reads them as before.
            copied = {}
            if _current_job is not None:
                for k in _ANSWER_FIELDS:
                    if k in data:
                        _current_job[k] = data[k]
                        copied[k] = data[k]
            print(f"[agent] resume for job {job_id}; copied answer fields: {copied}")
            return
        time.sleep(PAUSE_POLL_INTERVAL)


def _track_start(job: dict) -> None:
    job["_step_start"] = time.time()


def _track_end(job: dict) -> None:
    start = job.get("_step_start")
    if start:
        job["active_time"] = job.get("active_time", 0.0) + (time.time() - start)
        job["_step_start"] = None


def _record_stage_image(job: dict, key: str, data: bytes, filename: str,
                        vault_dir: "Path | None" = None, vault_name: str | None = None) -> None:
    """Write a stage image to OUTPUT_DIR and record it under the SAME name.

    The filename used on disk, in stage_images[key], and in job['images'] all
    come from the single `filename` argument, so the recorded name can never
    drift from the file that is actually uploaded and served (the bug where
    stage_images said 'stage1.png' but the saved file was '<jobid>_stage1.png').
    Every stage calls this as it completes, so no stage entry is ever skipped.
    """
    (OUTPUT_DIR / filename).write_bytes(data)
    job.setdefault("stage_images", {})[key] = filename
    job["images"] = [filename]
    if vault_dir is not None and vault_name:
        (vault_dir / vault_name).write_bytes(data)
    print(f"[stage-image] recorded {key} -> {filename}")


def _open_chat_for(page: Any, job: dict) -> None:
    """Open the chat in this job's per-workflow ChatGPT project.

    A project link that fails to load falls back to plain chatgpt.com inside
    open_chat(); when that happens we record a warning on the job (surfaced in
    the web UI) rather than failing the run."""
    result = open_chat(page, job.get("workflow"))
    if isinstance(result, dict) and result.get("fell_back") and result.get("warning"):
        report_warning(result["warning"])


def _rename_chat_once(page: Any, job: dict) -> None:
    """Name the chat "<client> - <task_id>" once, best-effort, for findability.

    Guarded so it runs a single time per job and never raises — renaming is
    cosmetic and must not fail the job."""
    if job.get("_chat_renamed"):
        return
    job["_chat_renamed"] = True
    client = (job.get("client") or "").strip()
    task_id = (job.get("task_id") or "").strip()
    title = " - ".join([p for p in (client, task_id) if p])
    if not title:
        return
    try:
        rename_chat(page, title)
    except Exception as exc:
        print(f"[agent] chat rename skipped: {exc}")


# ===========================================================================
# WORKFLOW FUNCTIONS — copied verbatim from the old app.py in-process worker.
# Do not edit their logic. They read/write the module-global INPUT_DIR /
# OUTPUT_DIR and call _track_start/_end and _wait_for_resume defined above.
# ===========================================================================

def _run_legacy_job(page: Any, job: dict[str, Any]) -> None:
    try:
        prompt = build_prompt(options=job["options"], params=job["params"], custom_note=job["custom_note"])
        image_paths = [str(INPUT_DIR / f) for f in job["files"]]
        images = generate(page=page, image_paths=image_paths, prompt=prompt, run_id=job["id"], workflow=job.get("workflow"))
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
    _open_chat_for(page, job)
    _track_end(job)

    # --- TURN 0 (optional): Extract text from image ---
    if text_image and not text:
        job["stage"] = 0
        job["stage_label"] = "Reading text from image"
        prompt0 = TEXT_TURN_0
        extracted_text = send_text_turn(page, prompt=prompt0, image_paths=[str(INPUT_DIR / text_image)], run_id=f"{job_id}_t0")
        _rename_chat_once(page, job)
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
        _rename_chat_once(page, job)

        if images1:
            suffix = f"_attempt{attempt}" if attempt > 1 else ""
            stage1_output = f"{job_id}_stage1{suffix}.png"
            _record_stage_image(
                job, "stage1", images1[0], stage1_output,
                vault_dir=run_dir, vault_name=f"{task_id}_R{run_number}_stage1_styles{suffix}.png")

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
    # Guard: the operator's style choice must have been copied back from the
    # server (see _wait_for_resume). An empty list here means the answer never
    # arrived — fail clearly instead of an IndexError.
    if not job.get("choices"):
        raise RuntimeError("No style number was received from the operator (choices is empty) before stage 2.")
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
            _record_stage_image(
                job, "stage2", images2[0], stage2_output,
                vault_dir=run_dir, vault_name=f"{task_id}_R{run_number}_stage2_colours{suffix}.png")

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
    # Guard: the operator's colour choice must have arrived from the server.
    if not job.get("choices"):
        raise RuntimeError("No colour number was received from the operator (choices is empty) before the final step.")
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
        _record_stage_image(
            job, "final", final_data, final_output,
            vault_dir=run_dir, vault_name=f"{task_id}_R{run_number}_final.png")
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
    _open_chat_for(page, job)

    extract_tpl = job.get("template_extract") or EXTRACT_CONTACT_SHEET
    job.setdefault("prompts", []).append(extract_tpl)

    images1 = send_turn(page, prompt=extract_tpl, image_paths=[str(image_path)], run_id=f"{job_id}_contact")
    _track_end(job)
    _rename_chat_once(page, job)

    if images1:
        # Single source of truth for the name — recorded and saved identically.
        contact_output = f"{job_id}_contact_sheet.png"
        (OUTPUT_DIR / contact_output).write_bytes(images1[0])
        job.setdefault("stage_images", {})["contact_sheet"] = contact_output
        print(f"[stage-image] recorded contact_sheet -> {contact_output}")
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
    _open_chat_for(page, job)
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
            _rename_chat_once(page, job)

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
    _open_chat_for(page, job)
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
                _rename_chat_once(page, job)
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
                _rename_chat_once(page, job)
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
                _rename_chat_once(page, job)
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



# ===========================================================================
# ORCHESTRATION
# ===========================================================================

def _download_inputs(input_files: list[str]) -> None:
    """Fetch each input file from the server into the local INPUT_DIR."""
    for name in input_files:
        safe = Path(name).name
        dest = INPUT_DIR / safe
        try:
            r = _get(f"/api/agent/file/{safe}")
            if r.ok:
                dest.write_bytes(r.content)
            else:
                print(f"[agent] could not download input {safe}: HTTP {r.status_code}")
        except Exception as exc:
            print(f"[agent] download error for {safe}: {exc}")


def _collect_output_files(job: dict) -> set[str]:
    """Every OUTPUT_DIR filename this job produced, gathered from its fields."""
    names: set[str] = set()
    for key in ("images", "final_names"):
        for n in job.get(key, []) or []:
            if n:
                names.add(n)
    for n in (job.get("stage_images") or {}).get("finals", []) or []:
        names.add(n)
    si = job.get("stage_images") or {}
    for k, v in si.items():
        if isinstance(v, str) and v:
            names.add(v)
    for step in job.get("custom_steps_done", []) or []:
        if step.get("file"):
            names.add(step["file"])
    for extra in ("aspect_baseline_file",):
        if job.get(extra):
            names.add(job[extra])
    # Only files that actually exist locally.
    return {n for n in names if (OUTPUT_DIR / n).exists()}


def _upload_outputs(job_id: str, job: dict) -> None:
    """Upload all generated output images for this job to the server."""
    files = _collect_output_files(job)
    if not files:
        return
    multipart = []
    handles = []
    try:
        for name in sorted(files):
            fh = open(OUTPUT_DIR / name, "rb")
            handles.append(fh)
            multipart.append(("files", (name, fh, "image/png")))
        r = _post(f"/api/agent/job/{job_id}/result", files=multipart)
        if r.ok:
            print(f"[agent] uploaded {len(files)} output file(s) for {job_id}")
        else:
            print(f"[agent] result upload HTTP {r.status_code}: {r.text[:200]}")
    finally:
        for fh in handles:
            try:
                fh.close()
            except Exception:
                pass


def _run_job(page: Any, job: dict) -> None:
    """Dispatch to the correct workflow — same mapping the old worker used."""
    wf = job.get("workflow")
    if wf == "text":
        _run_text_workflow(page, job)
    elif wf == "mockup":
        _run_mockup_workflow(page, job)
    elif wf == "artwork":
        _run_artwork_workflow(page, job)
    elif wf == "custom":
        _run_custom_workflow(page, job)
    else:
        _run_legacy_job(page, job)


def _handle_claimed_job(page: Any, claim: dict) -> None:
    global _current_job_id, _current_job
    job = claim["job"]
    job_id = job["id"]
    _download_inputs(claim.get("input_files", []))

    _current_job_id = job_id
    _current_job = job
    _sync_stop.clear()
    sync_thread = threading.Thread(target=_sync_loop, daemon=True)
    sync_thread.start()

    _reset_job_buffers()
    print(f"[agent] running job {job_id} ({job.get('workflow')})")
    try:
        _run_job(page, job)
    except _CancelledError:
        print(f"[agent] job {job_id} cancelled by operator")
        job["status"] = "cancelled"
    except Exception as exc:
        # Capture the FULL traceback (not just str(exc)) plus which step failed,
        # and post it so the failure is fully visible in the web UI.
        tb = traceback.format_exc()
        print(tb)
        job["status"] = "failed"
        job["error"] = str(exc)
        job["finished_at"] = time.time()
        _report_error(job_id, job, exc, tb)
    finally:
        _sync_stop.set()
        sync_thread.join(timeout=5)
        # Upload outputs, then one final state sync so the terminal status lands.
        try:
            _upload_outputs(job_id, job)
        except Exception as exc:
            print(f"[agent] output upload failed: {exc}")
        try:
            # Final post carries the TERMINAL status (done/failed/...) — the
            # server accepts status only when it is terminal. Flush any last
            # console output so the tail of the log reaches the UI.
            _post(f"/api/agent/job/{job_id}/progress", json={
                "agent_id": _current_agent_id,
                "stage": job.get("stage"), "stage_label": job.get("stage_label"),
                "active_time": job.get("active_time"), "logged_in": True,
                "job": _agent_job_blob(job, include_status=True),
                "log_lines": _drain_log(),
                "warnings": _drain_warnings(),
            })
        except Exception as exc:
            print(f"[agent] final sync failed: {exc}")
        _current_job_id = None
        _current_job = None


def _find_error_screenshot(job_id: str):
    """Newest logs/{job_id}*_error.png the workflow saved for this failure, if any."""
    try:
        logs = Path("logs")
        shots = sorted(logs.glob(f"{job_id}*_error.png"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        return shots[0] if shots else None
    except Exception:
        return None


def _report_error(job_id: str, job: dict, exc: Exception, tb: str) -> None:
    """Post rich failure diagnostics (+ screenshot when present) to the server."""
    fields = {
        "message": str(exc),
        "exc_type": type(exc).__name__,
        "traceback": tb,
        "step_name": job.get("stage_label") or f"stage {job.get('stage', '?')}",
        "agent_name": AGENT_NAME,
        "timestamp": str(time.time()),
        "session_expired": "true" if isinstance(exc, _session_expired_types()) else "false",
    }
    # Include the last of the captured console log inline with the error too.
    shot = _find_error_screenshot(job_id)
    try:
        if shot is not None:
            with open(shot, "rb") as fh:
                _post(f"/api/agent/job/{job_id}/error",
                      data=fields,
                      files=[("screenshot", (shot.name, fh, "image/png"))])
        else:
            _post(f"/api/agent/job/{job_id}/error", data=fields)
    except Exception as post_exc:
        print(f"[agent] failed to report error to server: {post_exc}")


def _session_expired_types():
    """Exception types that mean the ChatGPT session expired, if importable."""
    types = []
    try:
        from src.browser import SessionExpiredError
        types.append(SessionExpiredError)
    except Exception:
        pass
    return tuple(types) or (type(None),)


def open_browser_context():
    """Launch the persistent Chrome context and return (context, page).

    Used both by the console `main()` and the GUI. The GUI reuses this single
    context for the "Sign in to ChatGPT" flow and for the claim loop so there is
    only ever one profile lock.
    """
    # A SECOND launch here while one is already open would lock the profile and
    # break the session check. This line makes any accidental re-launch obvious.
    print("[agent] launch_persistent_context('acct1') — opening browser context")
    context = launch_context("acct1")
    page = context.pages[0] if context.pages else context.new_page()
    page.goto("https://chatgpt.com", wait_until="domcontentloaded")
    return context, page


def register(logged_in: bool):
    """Register with the server; returns agent_id or raises."""
    r = _post("/api/agent/register", json={"agent_name": AGENT_NAME, "logged_in": logged_in})
    r.raise_for_status()
    return r.json()["agent_id"]


def _apply_and_restart(log=print, status=None) -> bool:
    """Apply a staged update and re-exec. Returns False (and stays on the old,
    working code) if applying fails — a broken update never breaks the agent.

    Caller MUST ensure no job is running before invoking this."""
    st = self_update.apply_if_staged()
    if st.error:
        log(f"[update] apply failed, staying on current code: {st.error}")
        if status:
            status("error", f"Update failed: {st.error}")
        return False
    if not st.applied:
        return False
    log(f"[update] applied code {st.local_version}; restarting…")
    if status:
        status("connected", "Updating — restarting…")
    self_update.restart()
    return True


def apply_pending_update() -> dict:
    """GUI Restart button entry point: apply a staged update and re-exec.

    Returns a status dict; on failure the agent keeps its current code and the
    dict carries the error so the GUI can surface it. Never raises."""
    st = self_update.apply_if_staged()
    if st.applied and not st.error:
        self_update.restart()  # does not return on success
    return st.as_dict()


def run_loop(page, agent_id, stop_event=None, on_status=None, on_log=None,
             on_update=None, auto_restart=False) -> None:
    """The claim loop. Identical job/workflow behaviour to the original; the
    additions are an optional stop_event for clean GUI stop, status/log
    callbacks, and self-update.

    Self-update runs ONLY between jobs (in the idle branch), so an update can
    never happen mid-job. `on_update(status_dict)` is called when a newer build
    has been staged and is ready; if `auto_restart` is True (headless/console)
    the loop applies the staged update and re-execs immediately while idle."""
    global _current_agent_id
    _current_agent_id = agent_id  # so the sync-thread progress posts carry it (heartbeat)

    def log(msg):
        print(msg)
        if on_log:
            try:
                on_log(msg)
            except Exception:
                pass

    def status(state, detail=""):
        if on_status:
            try:
                on_status(state, detail)
            except Exception:
                pass

    def notify_update(st):
        if on_update:
            try:
                on_update(st.as_dict() if hasattr(st, "as_dict") else st)
            except Exception:
                pass

    # Update-check bookkeeping. First check fires on the first idle poll; then
    # hourly. Tracked here so it only ever runs while idle (between jobs).
    last_update_check = [0.0]
    update_staged = [False]

    def maybe_self_update():
        """Called only from the IDLE branch — never during a job.

        Checks the server hourly; when a newer build is staged, notifies the
        caller. In auto_restart mode (console) it applies + re-execs right away
        since we are provably idle here."""
        now = time.time()
        if update_staged[0]:
            # Already staged and waiting. In auto mode, apply now (we're idle).
            if auto_restart:
                _apply_and_restart(log, status)
            return
        if now - last_update_check[0] < UPDATE_CHECK_INTERVAL:
            return
        last_update_check[0] = now
        try:
            st = self_update.check_and_stage(SERVER_URL, AGENT_TOKEN)
        except Exception as exc:
            log(f"[update] check failed (keeping current code): {exc}")
            return
        # An older server without the endpoints (404) is not worth a word.
        if st.reason == "not_supported":
            return
        if st.error:
            # unreachable -> muted; error -> real problem. Either way the agent
            # keeps its current code; the GUI decides how loudly to show it.
            log(f"[update] {st.error} (keeping current code)")
            notify_update(st)
            return
        if st.staged:
            update_staged[0] = True
            log(f"[update] new agent code {st.server_version} staged "
                f"(current {st.local_version}).")
            notify_update(st)
            if auto_restart:
                _apply_and_restart(log, status)

    while stop_event is None or not stop_event.is_set():
        try:
            # Re-check session cheaply so the server's status stays accurate.
            try:
                logged_in = is_logged_in(page)
            except Exception:
                logged_in = False

            status("connected" if logged_in else "not_signed_in",
                   "Waiting for jobs" if logged_in else "Not signed in to ChatGPT")

            r = _get("/api/agent/next-job", params={"agent_id": agent_id, "logged_in": str(logged_in).lower()})
            if r.status_code == 204:
                # IDLE: the only place a self-update may run, so it can never
                # interrupt a job. This may re-exec the process (auto mode).
                maybe_self_update()
                if stop_event is not None:
                    stop_event.wait(POLL_INTERVAL)
                else:
                    time.sleep(POLL_INTERVAL)
                continue
            if not r.ok:
                log(f"[agent] next-job HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(POLL_INTERVAL)
                continue

            if not logged_in:
                # We claimed a job but the session is gone — fail it clearly.
                claim = r.json()
                jid = claim["job"]["id"]
                log("[agent] session expired — sign in to ChatGPT.")
                status("not_signed_in", "Session expired — sign in to ChatGPT")
                _post(f"/api/agent/job/{jid}/error", json={
                    "message": "ChatGPT session expired on the agent PC. Sign in and retry.",
                    "session_expired": True,
                })
                time.sleep(POLL_INTERVAL)
                continue

            claim = r.json()
            status("running", f"Running job {claim['job']['id']} ({claim['job'].get('workflow')})")
            log(f"[agent] running job {claim['job']['id']} ({claim['job'].get('workflow')})")
            _handle_claimed_job(page, claim)
            status("connected", "Job finished — waiting for jobs")
        except KeyboardInterrupt:
            log("\n[agent] stopping.")
            break
        except Exception as exc:
            log(f"[agent] loop error: {exc}")
            status("error", str(exc))
            traceback.print_exc()
            time.sleep(POLL_INTERVAL)


def main() -> None:
    if not AGENT_TOKEN:
        print("ERROR: AGENT_TOKEN is not set. Paste your token into the .env beside agent.py.")
        return

    print(f"[agent] server: {SERVER_URL}")
    print(f"[agent] name:   {AGENT_NAME}")

    # Self-update at startup, BEFORE any job is claimed and before Chrome opens.
    # 1) Apply anything a previous run already staged.
    # 2) Then check the server; if newer, stage + apply + re-exec now. A failed
    #    update is swallowed so the agent always continues on its current code.
    try:
        _startup_self_update()
    except Exception as exc:
        print(f"[update] startup self-update skipped ({exc}); continuing on current code.")

    try:
        context, page = open_browser_context()
        logged_in = is_logged_in(page)
    except Exception as exc:
        print(f"[agent] FATAL: could not launch Chrome: {exc}")
        traceback.print_exc()
        return

    if not logged_in:
        print("=" * 64)
        print("  NOT LOGGED IN to ChatGPT.")
        print("  Close this window and run:  python login.py")
        print("  Then start the agent again.")
        print("=" * 64)
    else:
        print("[agent] ChatGPT session OK — logged in.")

    try:
        agent_id = register(logged_in)
        print(f"[agent] registered as {agent_id}")
    except Exception as exc:
        print(f"[agent] FATAL: could not register with server: {exc}")
        return

    # Console mode: apply staged updates automatically while idle (no GUI to
    # click a Restart button). Never mid-job — the check runs in the idle branch.
    run_loop(page, agent_id, auto_restart=True)


def _startup_self_update() -> None:
    """Apply a pending update, then check for a newer one — all before any job.

    Runs at process start when we are provably idle. Any failure leaves the
    agent on its current, working code."""
    if not AGENT_TOKEN:
        return
    # Apply whatever a prior run staged (then it re-execs and never returns).
    if self_update.has_staged_update():
        st = self_update.apply_if_staged()
        if st.applied and not st.error:
            print(f"[update] applied staged code {st.local_version}; restarting…")
            self_update.restart()
            return
        if st.error:
            print(f"[update] could not apply staged update ({st.error}); on current code.")
    # Fresh check: stage + apply now if the server is newer.
    st = self_update.check_and_stage(SERVER_URL, AGENT_TOKEN)
    if st.reason == "not_supported":
        # Older server without self-update endpoints — nothing to do, stay quiet.
        return
    if st.error:
        print(f"[update] {st.error}; on current code {st.local_version}.")
        return
    if st.staged:
        print(f"[update] new code {st.server_version} staged (current {st.local_version}); applying…")
        ast = self_update.apply_if_staged()
        if ast.applied and not ast.error:
            print("[update] applied; restarting…")
            self_update.restart()
        elif ast.error:
            print(f"[update] apply failed ({ast.error}); on current code.")
    else:
        print(f"[update] up to date (code {st.local_version}).")


if __name__ == "__main__":
    main()

"""Write-through persistence for the in-memory job dict.

The server keeps the `jobs` dict in app.py as the hot cache that the 2s poll and
the agent claim read from. This module mirrors each job into Postgres on every
state transition so nothing is lost on restart, and loads jobs back into the
cache on startup.

Design:
  * to_row()/from_row() are LOSSLESS round-trips. Promoted (indexed/queried)
    fields become real columns; everything else is partitioned into four JSONB
    buckets — request / ui_state / prompt_bundle / agent_log — so the full
    ~200-field job dict survives verbatim and new fields need no migration.
  * persist_job() is a single-row upsert; it is the only DB write on the hot
    path and is deliberately cheap.
  * All writes are best-effort and never raise into the request path: a DB blip
    must not break a live job (the cache remains the source of truth in-process).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from db.models import ArtworkFile, Job, JobStep, VaultSave
from db.session import get_session

# Fields promoted to real columns on the jobs table (indexed / queried).
_PROMOTED = {
    "id", "workflow", "status", "stage", "stage_label", "text_mode",
    "created_by", "created_by_name", "client", "error", "interrupted",
    "created_at", "started_at", "finished_at", "paused_at", "last_progress_at",
}
# These two map to differently-named columns.
#   task_id    -> job_number
#   claimed_by -> claimed_by_agent

# Which job-dict keys go into the prompt_bundle bucket.
_PROMPT_KEYS = {"managed_prompts", "prompt_meta", "prompt_edited"}
# Which job-dict keys go into the agent_log bucket.
_LOG_KEYS = {"agent_log", "warnings", "_answered_pause_stages"}
# ui_state keys: live/output/pause state the UI reads. Everything NOT promoted,
# NOT prompt, NOT log and NOT one of these request-ish inputs goes to ui_state
# by default; we route a known input set to `request` for clarity.
_REQUEST_KEYS = {
    "text", "text_image", "files", "options", "params", "custom_note",
    "replace_image", "replace_new_text", "replace_target", "reference_image",
    "mockup_image", "extract_mode", "grid_cols", "grid_rows",
    "artwork_files", "custom_operations", "custom_prompts",
    "custom_operation_labels", "identify_mode", "aspect_dpi",
    "halftone_settings", "blackout_settings",
    # geometry operator inputs
    "geometry_reconstruction_mode", "geometry_print_location",
    "geometry_garment_style", "geometry_garment_color",
    "geometry_target_print_width_in", "geometry_target_dpi",
    "geometry_aspect_ratio_policy", "geometry_background_requirement",
    "geometry_white_elements_policy", "geometry_black_elements_policy",
    "geometry_semi_transparency_allowed", "geometry_distress_policy",
    # template overrides the operator may have typed
    "template_turn1", "template_turn2", "template_turn3", "template_extract",
    "template_regen", "template_replace_collage", "template_replace_final",
    "template_identify", "template_identify_regen",
}


def _to_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        return None


def _to_epoch(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.timestamp()
    try:
        return float(v)
    except Exception:
        return None


def to_row(job: dict) -> dict:
    """Split a job dict into the Job table's column values."""
    request: dict = {}
    ui_state: dict = {}
    prompt_bundle: dict = {}
    agent_log: dict = {}

    for k, v in job.items():
        if k in _PROMOTED or k in ("task_id", "claimed_by", "claimed_by_name", "claimed_at", "job_number"):
            continue
        if k in _PROMPT_KEYS:
            prompt_bundle[k] = v
        elif k in _LOG_KEYS:
            agent_log[k] = v
        elif k in _REQUEST_KEYS:
            request[k] = v
        else:
            ui_state[k] = v

    return {
        "id": job["id"],
        "workflow": job.get("workflow") or "",
        "status": job.get("status") or "",
        "stage": int(job.get("stage") or 0),
        "stage_label": job.get("stage_label") or "",
        "text_mode": job.get("text_mode"),
        "created_by": job.get("created_by"),
        "created_by_name": job.get("created_by_name") or "",
        "client": job.get("client") or "",
        "job_number": job.get("task_id") or "",
        "error": job.get("error"),
        "interrupted": bool(job.get("interrupted", False)),
        "created_at": _to_dt(job.get("created_at")) or datetime.now(timezone.utc),
        "started_at": _to_dt(job.get("started_at")),
        "finished_at": _to_dt(job.get("finished_at")),
        "paused_at": _to_dt(job.get("paused_at")),
        "last_progress_at": _to_dt(job.get("last_progress_at")),
        "claimed_by_agent": job.get("claimed_by"),
        "claimed_by_name": job.get("claimed_by_name") or "",
        "request": request,
        "ui_state": ui_state,
        "prompt_bundle": prompt_bundle,
        "agent_log": agent_log,
    }


def from_row(row: Job) -> dict:
    """Rebuild the in-memory job dict from a Job row (inverse of to_row)."""
    job: dict = {}
    job.update(row.request or {})
    job.update(row.ui_state or {})
    job.update(row.prompt_bundle or {})
    job.update(row.agent_log or {})
    # Promoted columns (authoritative; overwrite anything in the blobs).
    job["id"] = row.id
    job["workflow"] = row.workflow
    job["status"] = row.status
    job["stage"] = row.stage
    job["stage_label"] = row.stage_label
    job["text_mode"] = row.text_mode
    job["created_by"] = row.created_by
    job["created_by_name"] = row.created_by_name
    job["client"] = row.client
    job["task_id"] = row.job_number
    job["error"] = row.error
    job["interrupted"] = row.interrupted
    job["created_at"] = _to_epoch(row.created_at)
    job["started_at"] = _to_epoch(row.started_at)
    job["finished_at"] = _to_epoch(row.finished_at)
    job["paused_at"] = _to_epoch(row.paused_at)
    job["last_progress_at"] = _to_epoch(row.last_progress_at)
    job["claimed_by"] = row.claimed_by_agent
    job["claimed_by_name"] = row.claimed_by_name
    return job


import threading

# IDs of jobs whose last write-through failed (DB was down). A tiny background
# worker retries them so the row catches up once the DB returns. We store IDs,
# not snapshots, and re-read from the live cache at flush time via a resolver,
# so the retry always writes the LATEST job state (not a stale snapshot).
_dirty: set[str] = set()
_dirty_lock = threading.Lock()
_job_resolver = None  # set by app.py: id -> job dict (the in-memory cache lookup)
_retry_started = False


def set_job_resolver(fn) -> None:
    """app.py registers a callable(job_id)->job_dict|None so the retry worker can
    fetch the current job state from the authoritative in-memory cache."""
    global _job_resolver
    _job_resolver = fn


def persist_job(job: dict) -> bool:
    """Upsert one job. NEVER raises into the caller — on any DB error the job is
    marked dirty for the background retry and the in-memory cache stays the
    source of truth. Returns True on success, False if the write was deferred."""
    jid = job.get("id")
    try:
        vals = to_row(job)
        with get_session() as s:
            existing = s.get(Job, vals["id"])
            if existing is None:
                s.add(Job(**vals))
            else:
                for k, v in vals.items():
                    setattr(existing, k, v)
            s.commit()
            _sync_steps(s, job)
            s.commit()
        if jid:
            with _dirty_lock:
                _dirty.discard(jid)
        return True
    except Exception as exc:
        if jid:
            with _dirty_lock:
                _dirty.add(jid)
        print(f"[db] persist_job({jid}) deferred (cache still authoritative; will retry): {exc}")
        return False


def start_retry_worker(interval: float = 20.0) -> None:
    """Start a daemon that re-writes any deferred (dirty) jobs once the DB is
    reachable again. Idempotent — safe to call once at startup."""
    global _retry_started
    if _retry_started:
        return
    _retry_started = True

    def _loop():
        import time as _t
        while True:
            _t.sleep(interval)
            with _dirty_lock:
                ids = list(_dirty)
            if not ids or _job_resolver is None:
                continue
            for jid in ids:
                job = None
                try:
                    job = _job_resolver(jid)
                except Exception:
                    job = None
                if job is None:
                    with _dirty_lock:
                        _dirty.discard(jid)   # gone from cache; nothing to write
                    continue
                if persist_job(job):
                    print(f"[db] retry: job {jid} caught up")

    threading.Thread(target=_loop, name="db-persist-retry", daemon=True).start()


def _sync_steps(s, job: dict) -> None:
    """Replace this job's job_steps rows from custom_steps_done. Idempotent:
    we delete and re-insert, which is cheap (a handful of rows per job)."""
    steps = job.get("custom_steps_done") or []
    if not steps:
        return
    jid = job["id"]
    for existing in s.scalars(select(JobStep).where(JobStep.job_id == jid)).all():
        s.delete(existing)
    s.flush()
    for i, st in enumerate(steps):
        sim = st.get("similarity")
        s.add(JobStep(
            job_id=jid,
            step_index=int(st.get("step", i)),
            op=st.get("op"),
            label=st.get("label") or "",
            method=st.get("method"),
            prompt_key=st.get("prompt_key"),
            prompt_version=st.get("prompt_version"),
            prompt_source=st.get("prompt_source"),
            prompt_text_used=st.get("prompt"),
            operator_override_text=st.get("operator_override_text"),
            result_file=st.get("file"),
            similarity=sim if isinstance(sim, dict) else None,
        ))


def record_artwork_file(job_id: str | None, path: str, role: str = "result",
                        width: int | None = None, height: int | None = None,
                        dpi: int | None = None, size_bytes: int | None = None,
                        sha256: str | None = None, derived_from_id: int | None = None) -> None:
    """Index a generated/uploaded file. Best-effort."""
    try:
        with get_session() as s:
            s.add(ArtworkFile(
                job_id=job_id, path=path, role=role, width=width, height=height,
                dpi=dpi, size_bytes=size_bytes, sha256=sha256,
                derived_from_id=derived_from_id,
            ))
            s.commit()
    except Exception as exc:
        print(f"[db] record_artwork_file({path}) failed: {exc}")


def record_vault_save(job_id: str | None, client: str, job_number: str,
                      vault_folder: str, file_name: str, lifecycle: str | None = None,
                      customer: str | None = None, saved_by: str | None = None,
                      run_number: int | None = None) -> None:
    """Index a vault save. Best-effort."""
    try:
        with get_session() as s:
            s.add(VaultSave(
                job_id=job_id, client=client or "", job_number=job_number or "",
                run_number=run_number, vault_folder=vault_folder or "",
                file_name=file_name or "", lifecycle=lifecycle, customer=customer,
                saved_by=saved_by,
            ))
            s.commit()
    except Exception as exc:
        print(f"[db] record_vault_save({file_name}) failed: {exc}")


def record_prompt_run(run: dict) -> None:
    """Mirror a Decoinks run record into our prompt_runs table. Best-effort."""
    try:
        from db.models import PromptRun
        with get_session() as s:
            s.add(PromptRun(
                job_id=run.get("external_ref"),
                prompt_key=run.get("prompt_key"),
                prompt_version_id=run.get("prompt_version_id"),
                prompt_source=run.get("prompt_source"),
                status=run.get("status"),
                resolved_prompt=run.get("resolved_prompt"),
                input_variables=run.get("input_variables"),
                latency_ms=run.get("latency_ms"),
                error_message=run.get("error_message"),
                output_reference=run.get("output_reference"),
                output_files={"files": run.get("output_files")} if run.get("output_files") else None,
            ))
            s.commit()
    except Exception as exc:
        print(f"[db] record_prompt_run failed: {exc}")


# ---------------------------------------------------------------------------
# Startup load + restart recovery
# ---------------------------------------------------------------------------

_ACTIVE_STATUSES = ("queued", "running", "preparing")


def load_jobs_into_cache(cache: dict, grace_seconds: float = 600.0,
                         finished_window_seconds: float = 24 * 3600) -> dict:
    """Load jobs from the DB into the in-memory cache on startup — but only the
    ones worth keeping hot: all ACTIVE jobs (queued / running / preparing / any
    awaiting_*) plus FINISHED jobs from the last `finished_window_seconds`
    (default 24h). Older finished jobs stay in the DB and are served by the
    history endpoint, not the cache, so memory stays bounded.

    Restart-recovery rules:
      * running   -> KEEP running within the grace window (a restart is seconds;
                     the agent uploads its result). No contact past grace -> mark
                     interrupted.
      * awaiting_* -> keep the pause exactly (answers are persisted).
      * queued/preparing -> stays claimable.
      * finished (<=window) -> loaded as recent history.
    """
    summary = {"loaded": 0, "running_kept": 0, "interrupted": 0, "skipped_old": 0}
    now = time.time()
    from datetime import datetime, timezone
    cutoff = datetime.fromtimestamp(now - finished_window_seconds, tz=timezone.utc)
    try:
        with get_session() as s:
            # Active (any status that is not a terminal one) OR finished recently.
            terminal = ("done", "done_with_errors", "failed", "cancelled")
            q = select(Job).where(
                (Job.status.notin_(terminal)) |
                (Job.finished_at.is_(None)) |
                (Job.finished_at >= cutoff)
            )
            for row in s.scalars(q).all():
                job = from_row(row)
                status = job.get("status") or ""
                if status == "running":
                    last = job.get("last_progress_at") or job.get("started_at") or job.get("created_at") or now
                    if (now - float(last)) > grace_seconds:
                        job["status"] = "failed"
                        job["interrupted"] = True
                        job["error"] = ("Interrupted by a server restart; the agent did not "
                                        "report back within the grace period.")
                        summary["interrupted"] += 1
                        persist_job(job)
                    else:
                        summary["running_kept"] += 1
                cache[job["id"]] = job
                summary["loaded"] += 1
    except Exception as exc:
        print(f"[db] load_jobs_into_cache failed (starting with empty cache): {exc}")
    return summary


def load_one(job_id: str) -> dict | None:
    """Load a single job from the DB as a job dict, or None if absent. Used to
    serve an OLD finished job that was evicted from the cache (opened from
    Previous Runs) without a 404. Never raises."""
    if not job_id:
        return None
    try:
        with get_session() as s:
            row = s.get(Job, job_id)
            return from_row(row) if row is not None else None
    except Exception as exc:
        print(f"[db] load_one({job_id}) failed: {exc}")
        return None


def query_history(created_by: str | None, is_admin: bool, page: int, page_size: int) -> dict:
    """Previous Runs, straight from the DB (not the cache) with role scope and
    LIMIT/OFFSET. Returns {items(raw job dicts), page, page_size, total,
    has_more}. The caller shapes each dict via _public_job."""
    from sqlalchemy import func
    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 20), 100))
    offset = (page - 1) * page_size
    out = {"items": [], "page": page, "page_size": page_size, "total": 0, "has_more": False}
    try:
        with get_session() as s:
            base = select(Job)
            cnt = select(func.count()).select_from(Job)
            if not is_admin:
                uname = (created_by or "").strip().lower()
                base = base.where(func.lower(Job.created_by) == uname)
                cnt = cnt.where(func.lower(Job.created_by) == uname)
            total = int(s.scalar(cnt) or 0)
            rows = s.scalars(
                base.order_by(Job.created_at.desc()).offset(offset).limit(page_size)
            ).all()
            out["items"] = [from_row(r) for r in rows]
            out["total"] = total
            out["has_more"] = offset + page_size < total
    except Exception as exc:
        print(f"[db] query_history failed: {exc}")
    return out


def sweep_running_grace(cache: dict, grace_seconds: float = 600.0) -> int:
    """Flip any still-running job that has had no agent contact beyond the grace
    period to interrupted. Meant to be called periodically after startup so a
    job whose agent never comes back is not stuck 'running' forever. Returns the
    number flipped."""
    now = time.time()
    flipped = 0
    for job in list(cache.values()):
        if job.get("status") != "running":
            continue
        last = job.get("last_progress_at") or job.get("started_at") or job.get("created_at") or now
        if (now - float(last)) > grace_seconds:
            job["status"] = "failed"
            job["interrupted"] = True
            job["error"] = ("Interrupted: the agent stopped reporting and did not "
                            "complete within the grace period.")
            persist_job(job)
            flipped += 1
    return flipped

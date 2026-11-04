"""Print Ready: PhotoCraft must run only when ticked.

Server: an explicit pr_via_photocraft=False is never overridden (missing -> True
for the printready workflow only).
Agent: OFF -> QA text turn only, nothing PhotoCraft logged, assessment stored.
       ON  -> QA text turn FIRST, then the PhotoCraft step; assessment + PNG.
Run:  python tests/test_printready_flow.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- server ----
import app as server  # noqa: E402

req_missing = server.GenerateRequest(workflow="printready", artwork_files=["a.png"])
req_false = server.GenerateRequest(workflow="printready", artwork_files=["a.png"],
                                   pr_via_photocraft=False)
req_true = server.GenerateRequest(workflow="printready", artwork_files=["a.png"],
                                  pr_via_photocraft=True)
check("request model: missing field is None (not False)", req_missing.pr_via_photocraft is None)
check("request model: explicit False kept", req_false.pr_via_photocraft is False)
check("request model: explicit True kept", req_true.pr_via_photocraft is True)

# Drive create_job end to end with its collaborators stubbed out.
src = Path(server.INPUT_DIR) / "_pr_flow_test.png"
Image.new("RGBA", (64, 64), (200, 30, 40, 255)).save(src)


def make_job(req) -> dict | None:
    server.jobs.clear()
    saved = {}
    for name, fn in (
        ("_current_user_record", lambda r: {"username": "tester", "display_name": "Tester"}),
        ("_user_has_online_agent", lambda u: True),
        ("_user_has_free_agent", lambda u: True),
        ("_validate_workflow_request", lambda r, u: {}),
    ):
        saved[name] = getattr(server, name)
        setattr(server, name, fn)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                out = server.create_job(req, type("R", (), {"headers": {}, "cookies": {}})())
            except server.HTTPException as exc:
                return {"_http": exc.status_code, "_detail": exc.detail}
        jid = out.get("id") if isinstance(out, dict) else None
        return server.jobs.get(jid) if jid else (next(iter(server.jobs.values()), None))
    finally:
        for name, fn in saved.items():
            setattr(server, name, fn)


try:
    for label, rq, want in (("missing -> True", req_missing, True),
                            ("explicit False stays False", req_false, False),
                            ("explicit True stays True", req_true, True)):
        rq = rq.model_copy(update={"artwork_files": [src.name]})
        job = make_job(rq)
        got = job.get("pr_via_photocraft") if job and "_http" not in job else "HTTP"
        check(f"create_job printready: {label}", got is want, f"got={got!r}")
    other = server.GenerateRequest(workflow="text", pr_via_photocraft=None)
    check("other workflows default OFF", bool(other.pr_via_photocraft) is False)
finally:
    server.jobs.clear()
    try:
        src.unlink()
    except OSError:
        pass

# ----------------------------------------------------------------- agent ----
import agent  # noqa: E402

tmp = Path(tempfile.mkdtemp())
agent.INPUT_DIR = tmp
agent.OUTPUT_DIR = tmp
(tmp / "art.png").write_bytes(b"x")

REPORT = {"score": 88, "band": "fixable", "checks": [
    {"label": "Resolution (DPI)", "value_str": "300 DPI", "status": "pass"}]}

calls: list[str] = []


def run_flow(via: bool, report=REPORT) -> tuple[dict, str]:
    calls.clear()
    job = {"id": "J1", "artwork_files": ["art.png"], "pr_via_photocraft": via,
           "printready_measurements": report, "printready_report": report,
           "status": "running", "prompts": []}

    def fake_text_turn(page, prompt, image_paths=None, run_id=""):
        calls.append("qa_turn")
        return "Looks good. Minor halo."

    def fake_pc_step(page, job_, input_path, run_id):
        calls.append("photocraft_step")
        print("[photocraft J1] select tool")
        return b"PNGBYTES", {"summary": "Print ready 3225x4030", "width_in": 10.75,
                             "height_in": 13.4, "dpi": 300, "to_size": (3225, 4030)}

    patches = {
        "send_text_turn": fake_text_turn,
        "_photocraft_print_ready_step": fake_pc_step,
        "_open_chat_for": lambda page, j: calls.append("open_chat"),
        "_track_start": lambda j: None, "_track_end": lambda j: None,
        "_rename_chat_once": lambda page, j: None,
        "_tpl": lambda j, key, default: default,
        "_upload_files": lambda jid, names: names,
        "_mark_uploaded": lambda j, r: None,
    }
    saved = {k: getattr(agent, k) for k in patches}
    for k, v in patches.items():
        setattr(agent, k, v)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            agent._run_printready_workflow(object(), job)
    finally:
        for k, v in saved.items():
            setattr(agent, k, v)
    return job, buf.getvalue()


job, log = run_flow(False)
check("OFF: done", job["status"] == "done", job.get("error", ""))
check("OFF: only the QA turn ran", calls == ["open_chat", "qa_turn"], str(calls))
check("OFF: assessment stored", job.get("printready_assessment") == "Looks good. Minor halo.")
check("OFF: no '[photocraft' in the log", "[photocraft" not in log.lower(), log[:200])
check("OFF: no print-ready PNG / summary", not job.get("final_names") and not job.get("printready_summary"))

job, log = run_flow(True)
check("ON: done", job["status"] == "done", job.get("error", ""))
check("ON: QA first, then PhotoCraft, same chat (one open_chat)",
      calls == ["open_chat", "qa_turn", "photocraft_step"], str(calls))
check("ON: assessment stored", job.get("printready_assessment") == "Looks good. Minor halo.")
check("ON: PNG + summary stored", job.get("final_names") == ["J1_printready.png"]
      and bool(job.get("printready_summary")))
check("ON: PNG written", (tmp / "J1_printready.png").read_bytes() == b"PNGBYTES")

job, log = run_flow(False, report=None)
check("OFF without measurements fails clearly", job["status"] == "failed" and bool(job.get("error")))
check("OFF without measurements: no turns", calls == [], str(calls))

job, log = run_flow(True, report=None)
check("ON without measurements: QA skipped, PhotoCraft still runs",
      job["status"] == "done" and calls == ["open_chat", "photocraft_step"], str(calls))

print("RESULT: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED"))
sys.exit(1 if FAILS else 0)

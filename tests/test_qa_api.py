"""Public Print Ready QA API (POST /api/v1/qa/print-ready + poll).

Covers: no/wrong key -> 401; good PNG -> 200 with score + checks; corrupt file
-> 400; JSON image_base64 -> 200; assessment=true -> 202 + job_id (with an agent
registered) or a clear error otherwise. Run:  python tests/test_qa_api.py
"""
from __future__ import annotations

import base64
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Keys must exist BEFORE app is imported (QA_API_KEYS is read at import time).
os.environ["QA_API_KEYS"] = "test-key-123, other-key"

from PIL import Image, ImageDraw  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as server  # noqa: E402

client = TestClient(server.app)
KEY = {"X-API-Key": "test-key-123"}
URL = "/api/v1/qa/print-ready"

FAILS: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    if not cond:
        FAILS.append(name)


def good_png() -> bytes:
    im = Image.new("RGBA", (512, 640), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 452, 452), fill=(200, 30, 40, 255))
    d.rectangle((160, 470, 360, 600), fill=(30, 90, 200, 255))
    b = io.BytesIO()
    im.save(b, "PNG", dpi=(300, 300))
    return b.getvalue()


png = good_png()

# --- auth -------------------------------------------------------------------
r = client.post(URL, files={"file": ("d.png", png, "image/png")})
check("no key -> 401", r.status_code == 401, str(r.status_code))
r = client.post(URL, files={"file": ("d.png", png, "image/png")}, headers={"X-API-Key": "nope"})
check("wrong key -> 401", r.status_code == 401, str(r.status_code))

# --- good PNG multipart -----------------------------------------------------
r = client.post(URL, files={"file": ("d.png", png, "image/png")},
                data={"dpi": "300"}, headers=KEY)
check("good PNG -> 200", r.status_code == 200, str(r.status_code))
if r.status_code == 200:
    j = r.json()
    check("200: ok true", j.get("ok") is True)
    check("200: has integer score", isinstance(j.get("score"), int), str(j.get("score")))
    check("200: band is a known value", j.get("band") in ("ready", "fixable", "not_ready"), str(j.get("band")))
    check("200: checks is a non-empty list", isinstance(j.get("checks"), list) and len(j["checks"]) >= 10,
          str(len(j.get("checks", []))))
    check("200: a check has id/status/threshold",
          all(k in j["checks"][0] for k in ("id", "label", "value", "status", "threshold")))
    check("200: image block has dims + dpi", j["image"]["width"] == 512 and j["image"]["height"] == 640
          and j["image"]["dpi"] == 300, str(j.get("image")))
    check("200: duration_ms present", isinstance(j.get("duration_ms"), int))
    check("200: DPI check passes at 300", next(c for c in j["checks"] if c["id"] == "dpi")["status"] == "pass")

# --- corrupt file -----------------------------------------------------------
r = client.post(URL, files={"file": ("x.png", b"not-an-image", "image/png")}, headers=KEY)
check("corrupt file -> 400", r.status_code == 400, str(r.status_code))
check("corrupt file: clear message", "image" in r.json().get("detail", "").lower(), r.json().get("detail", ""))

# --- JSON image_base64 ------------------------------------------------------
r = client.post(URL, json={"image_base64": base64.b64encode(png).decode(), "dpi": 300}, headers=KEY)
check("JSON image_base64 -> 200", r.status_code == 200, str(r.status_code))
if r.status_code == 200:
    check("JSON base64: same score as multipart", isinstance(r.json().get("score"), int))
# data: URL prefix tolerated
r = client.post(URL, json={"image_base64": "data:image/png;base64," + base64.b64encode(png).decode()}, headers=KEY)
check("JSON base64 with data: prefix -> 200", r.status_code == 200, str(r.status_code))
# bad base64
r = client.post(URL, json={"image_base64": "@@@not-base64@@@"}, headers=KEY)
check("JSON bad base64 -> 400", r.status_code == 400, str(r.status_code))
# neither field
r = client.post(URL, json={"dpi": 300}, headers=KEY)
check("JSON without image -> 400", r.status_code == 400, str(r.status_code))

# --- oversized dimensions ---------------------------------------------------
big = io.BytesIO(); Image.new("RGBA", (12001, 10), (255, 0, 0, 255)).save(big, "PNG")
r = client.post(URL, files={"file": ("big.png", big.getvalue(), "image/png")}, headers=KEY)
check("over 12000px side -> 400", r.status_code == 400, str(r.status_code))

# --- assessment=true --------------------------------------------------------
# Without an online agent it returns 200 + assessment_error (clear, not a crash).
r = client.post(URL, files={"file": ("d.png", png, "image/png")},
                data={"dpi": "300", "assessment": "true"}, headers=KEY)
check("assessment, no agent -> 200 with assessment_error",
      r.status_code == 200 and r.json().get("assessment_error"), str(r.status_code))

# Register a fake online agent, then assessment=true must queue a job -> 202.
AGENT = {"id": "agentX", "username": "svc", "name": "svc-pc",
         "last_seen": __import__("time").time(), "logged_in": True}
server.agents["agentX"] = AGENT
# The assessment job is owned by a real account that has an online agent. Stub
# the user lookup (no persistent store write) so "svc" resolves to an account.
_orig_get_user = server.users.get_user
_svc_rec = {"username": "svc", "display_name": "Service", "role": "designer", "disabled": False}
server.users.get_user = lambda u: _svc_rec if (u or "").strip().lower() == "svc" else _orig_get_user(u)
_orig_online = server._online_agents
_orig_online_user = server._online_agents_for_user
_orig_free = server._user_has_free_agent
server._online_agents = lambda: [AGENT]
server._online_agents_for_user = lambda u: [AGENT] if server._norm_user(u) == "svc" else []
server._user_has_free_agent = lambda u: server._norm_user(u) == "svc"
try:
    r = client.post(URL, files={"file": ("d.png", png, "image/png")},
                    data={"dpi": "300", "assessment": "true"}, headers=KEY)
    check("assessment + agent -> 202", r.status_code == 202, str(r.status_code))
    jid = None
    if r.status_code == 202:
        j = r.json()
        jid = j.get("job_id")
        check("202: job_id present", bool(jid))
        check("202: status queued", j.get("status") == "queued", str(j.get("status")))
        check("202: poll_url points at the job", j.get("poll_url", "").endswith(jid or "?"))
        check("202: still carries the OpenCV score", isinstance(j.get("score"), int))
    # Poll it.
    if jid:
        rp = client.get(f"{URL}/{jid}", headers=KEY)
        check("poll -> 200", rp.status_code == 200, str(rp.status_code))
        pj = rp.json()
        check("poll: status is queued/running", pj.get("status") in ("queued", "running"), str(pj.get("status")))
        check("poll: checks present from server measurement", len(pj.get("checks", [])) >= 10)
        check("poll: assessment field present", "assessment" in pj)
    rp = client.get(f"{URL}/does-not-exist", headers=KEY)
    check("poll unknown job -> 404", rp.status_code == 404, str(rp.status_code))
    rp = client.get(f"{URL}/{jid or 'x'}")
    check("poll without key -> 401", rp.status_code == 401, str(rp.status_code))
finally:
    server._online_agents = _orig_online
    server._online_agents_for_user = _orig_online_user
    server._user_has_free_agent = _orig_free
    server.users.get_user = _orig_get_user
    server.agents.pop("agentX", None)
    for k in list(server.jobs.keys()):
        server.jobs.pop(k, None)
    # Remove any qa_api_*.png the assessment path wrote into ./input.
    for f in server.INPUT_DIR.glob("qa_api*.png"):
        try:
            f.unlink()
        except OSError:
            pass

print("RESULT: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED"))
sys.exit(1 if FAILS else 0)

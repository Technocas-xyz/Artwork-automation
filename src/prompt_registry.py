"""Prompts from Decoinks Prompt Management, with the built-in text as a safety net.

The words ChatGPT is given used to live only in config/workflows.py, so changing
one meant a code change and a redeploy, and nothing recorded which wording
produced which artwork. Now the LIVE version of each prompt is published in
Decoinks (Printshop → Prompt Management) and this module fetches it.

How it behaves:
  * The server (app.py) asks Decoinks for every prompt this app runs, at most
    once a minute, and hands the resolved templates to the agent inside each
    job. Designer PCs never talk to Decoinks and never hold its credential.
  * The last good answer is kept in memory and on disk. If Decoinks is slow or
    down, jobs keep running on that copy; with no copy at all, on the built-in
    text in config/workflows.py. A job is never blocked by Prompt Management.
  * A published prompt that asks for a placeholder the code cannot supply is
    refused (the built-in text is used instead), because str.format would crash
    the job half way through.
  * After a job ends, one run per prompt it used is reported back to Decoinks —
    which version, whether a designer edited it, how long it took, what files
    came out — without ever delaying or failing the job.

Placeholders: Decoinks writes {{text}}; Python templates here write {text}, with
literal braces doubled. to_python_template / from_python_template convert
between the two exactly.
"""

from __future__ import annotations

import json
import os
import re
import string
import threading
import time
from pathlib import Path
from typing import Any

import requests

BACKEND = os.getenv("PRINTSHOP_BACKEND", "http://172.17.0.1:8094").rstrip("/")
SOURCE_APP = "artwork-automation"
TTL_SECONDS = float(os.getenv("PROMPT_CACHE_SECONDS", "60"))
RETRY_SECONDS = 15.0
_CACHE_FILE = Path(os.getenv(
    "PROMPT_CACHE_FILE",
    str(Path(__file__).resolve().parent.parent / "logs" / "prompt_cache.json"),
))

# Template name used in this code  →  prompt key in Decoinks Prompt Management.
PROMPT_KEYS: dict[str, str] = {
    "TEXT_TURN_0": "AIS.TEXT.EXTRACT",
    "TEXT_TURN_1": "AIS.TEXT.COLLAGE",
    "TEXT_TURN_2": "AIS.COLORWAY.GENERATE",
    "TEXT_TURN_3": "AIS.TEXT.FINAL",
    # "Replace text in a design" mode (Text workflow, third input mode).
    "TEXT_REPLACE_COLLAGE": "AIS.TEXT.REPLACE_COLLAGE",
    "TEXT_REPLACE_FINAL": "AIS.TEXT.REPLACE_FINAL",
    # "Wording with a client-supplied image" modes (UC-3): stage-1 collage only.
    # Turns 2/3 reuse TEXT_TURN_2 / TEXT_TURN_3.
    "TEXT_IMAGE_ELEMENT_COLLAGE": "AIS.TEXT.IMAGE_ELEMENT_COLLAGE",
    "TEXT_IMAGE_STYLE_COLLAGE": "AIS.TEXT.IMAGE_STYLE_COLLAGE",
    "EXTRACT_CONTACT_SHEET": "AIS.EXTRACT.DETECT",
    "EXTRACT_SINGLE": "AIS.EXTRACT.SINGLE",
    "ARTWORK_REGENERATE": "AIS.RECREATE.CLEANUP",
    "CUSTOM_RECONSTRUCT": "AIS.RECREATE.GENERATE",
    "CUSTOM_REMOVE_BACKGROUND": "AIS.PRINTREADY.REMOVE_BACKGROUND",
    "CUSTOM_HALO_REMOVAL": "AIS.PRINTREADY.HALO_REMOVAL",
    "CUSTOM_DETECT_OBJECTS": "AIS.COLORWAY.DETECT_OBJECTS",
    "CUSTOM_CHANGE_COLOR": "AIS.COLORWAY.CHANGE_COLOR",
    "CUSTOM_ASPECT_ADVICE": "AIS.RATIO.PLAN",
    "CUSTOM_ASPECT_BASELINE": "AIS.RATIO.BASELINE",
    "CUSTOM_ASPECT_REGENERATE": "AIS.RATIO.REGENERATE",
    # In config/workflows.py and kept in the library, but no screen sends them yet.
    "EXTRACT_BOXES": "AIS.EXTRACT.BOXES",
    "EXTRACT_ARTWORKS": "AIS.EXTRACT.SEPARATE",
    "MOCKUP_REGENERATE": "AIS.EXTRACT.REGENERATE",
    # Black Out and Half Tone now run locally (Pillow); their ChatGPT wording is kept.
    "CUSTOM_BLACK_OUT": "AIS.PRINTREADY.BLACK_OUT",
    "CUSTOM_HALF_TONE": "AIS.PRINTREADY.HALF_TONE",
    # The original edit-options job (API only): base block + one block per option.
    "BASE_INSTRUCTION": "AIS.EDIT.BASE",
    "JOB_OPTION_TEXT_ONLY": "AIS.EDIT.TEXT_ONLY",
    "JOB_OPTION_REMOVE_BACKGROUND": "AIS.EDIT.REMOVE_BACKGROUND",
    "JOB_OPTION_CHANGE_BACKGROUND": "AIS.EDIT.CHANGE_BACKGROUND",
    "JOB_OPTION_BLUR_BACKGROUND": "AIS.EDIT.BLUR_BACKGROUND",
    "JOB_OPTION_RECOLOUR": "AIS.EDIT.RECOLOUR",
    "JOB_OPTION_UPSCALE_CLEANUP": "AIS.EDIT.UPSCALE_CLEANUP",
}

# Every live prompt under this prefix is a one-step Custom operation (artwork in,
# image out) with no code change: publish AIS.CUSTOM_OPERATIONS.<NAME> in Prompt
# Management and it appears on the Custom screen within a minute. Such a prompt
# must not use {{placeholders}}; nothing here would fill them.
DYNAMIC_PREFIX = "AIS.CUSTOM_OPERATIONS."
DYNAMIC_OP_PREFIX = "pm_"          # the operation key the UI and agent use
_DYNAMIC_NAME = "PM_OP:"           # template name: PM_OP:<prompt key>
_SAFE_SUFFIX = re.compile(r"^[A-Z0-9_]{1,80}$")


def is_dynamic(name: str) -> bool:
    return name.startswith(_DYNAMIC_NAME)


def dynamic_op_key(prompt_key: str) -> str | None:
    suffix = prompt_key[len(DYNAMIC_PREFIX):] if prompt_key.startswith(DYNAMIC_PREFIX) else ""
    return DYNAMIC_OP_PREFIX + suffix.lower() if _SAFE_SUFFIX.match(suffix) else None


def dynamic_name(op_key: str) -> str:
    """pm_foo_bar → PM_OP:AIS.CUSTOM_OPERATIONS.FOO_BAR"""
    return _DYNAMIC_NAME + DYNAMIC_PREFIX + op_key[len(DYNAMIC_OP_PREFIX):].upper()


def key_for(name: str) -> str | None:
    """The Prompt Management key behind a template name."""
    return name[len(_DYNAMIC_NAME):] if is_dynamic(name) else PROMPT_KEYS.get(name)


# Which prompts each workflow can use, for the per-job snapshot and the run log.
WORKFLOW_PROMPTS: dict[str, list[str]] = {
    # Default (superset) for the Text workflow: the typed/from-image turns plus
    # the "replace text in a design" prompts. prompt_names_for_job narrows this
    # to the exact prompts a job uses when the text_mode is known.
    "text": ["TEXT_TURN_0", "TEXT_TURN_1", "TEXT_TURN_2", "TEXT_TURN_3",
             "TEXT_REPLACE_COLLAGE", "TEXT_REPLACE_FINAL",
             "TEXT_IMAGE_ELEMENT_COLLAGE", "TEXT_IMAGE_STYLE_COLLAGE"],
    "mockup": ["EXTRACT_CONTACT_SHEET", "EXTRACT_SINGLE"],
    "artwork": ["ARTWORK_REGENERATE"],
}

# The prompts each Text input mode actually runs.
TEXT_MODE_PROMPTS: dict[str, list[str]] = {
    "typed": ["TEXT_TURN_1", "TEXT_TURN_2", "TEXT_TURN_3"],
    "from_image": ["TEXT_TURN_0", "TEXT_TURN_1", "TEXT_TURN_2", "TEXT_TURN_3"],
    "replace": ["TEXT_REPLACE_COLLAGE", "TEXT_REPLACE_FINAL"],
    # Image modes: their own stage-1 collage, then the shared colour + final.
    "image_element": ["TEXT_IMAGE_ELEMENT_COLLAGE", "TEXT_TURN_2", "TEXT_TURN_3"],
    "image_style": ["TEXT_IMAGE_STYLE_COLLAGE", "TEXT_TURN_2", "TEXT_TURN_3"],
}
CUSTOM_OPERATION_PROMPTS: dict[str, list[str]] = {
    "reconstruct": ["CUSTOM_RECONSTRUCT"],
    "remove_background": ["CUSTOM_REMOVE_BACKGROUND"],
    "halo_removal": ["CUSTOM_HALO_REMOVAL"],
    "change_object_color": ["CUSTOM_DETECT_OBJECTS", "CUSTOM_CHANGE_COLOR"],
    "aspect_ratio": ["CUSTOM_ASPECT_ADVICE", "CUSTOM_ASPECT_BASELINE", "CUSTOM_ASPECT_REGENERATE"],
}

# The model these prompts run on: ChatGPT, driven through the designer's browser.
MODEL = {"provider": "OpenAI", "model": "chatgpt-web"}

_VAR = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


def _secret() -> str:
    return os.getenv("DECOINKS_SERVICE_SECRET", "").strip()


def to_python_template(text: str) -> str:
    """Decoinks text ({{name}}) → a str.format template ({name}, literal braces doubled)."""
    out: list[str] = []
    pos = 0
    for m in _VAR.finditer(text):
        out.append(text[pos:m.start()].replace("{", "{{").replace("}", "}}"))
        out.append("{" + m.group(1) + "}")
        pos = m.end()
    out.append(text[pos:].replace("{", "{{").replace("}", "}}"))
    return "".join(out)


def from_python_template(template: str) -> str:
    """A str.format template → Decoinks text. The exact inverse of to_python_template."""
    parts: list[str] = []
    for literal, field, _spec, _conv in string.Formatter().parse(template):
        parts.append(literal)
        if field is not None:
            parts.append("{{" + field + "}}")
    return "".join(parts)


def template_fields(template: str) -> set[str]:
    try:
        return {f for _, f, _, _ in string.Formatter().parse(template) if f}
    except ValueError:
        return {"<malformed>"}


OPTION_PREFIX = "JOB_OPTION_"


def option_template_name(option_key: str) -> str:
    return OPTION_PREFIX + option_key.upper()


def is_composed(name: str) -> bool:
    """Blocks that are joined into one bigger prompt rather than sent on their own."""
    return name == "BASE_INSTRUCTION" or name.startswith(OPTION_PREFIX)


def _builtin(name: str) -> str:
    if is_dynamic(name):
        return ""   # a Prompt Management operation has no built-in text
    if name == "BASE_INSTRUCTION":
        from src.prompt_builder import BASE_INSTRUCTION
        return BASE_INSTRUCTION
    if name.startswith(OPTION_PREFIX):
        from config.job_options import JOB_OPTIONS
        return JOB_OPTIONS[name[len(OPTION_PREFIX):].lower()]
    from config import workflows
    return getattr(workflows, name)


class PromptRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] | None = None
        self._fetched_at = 0.0
        self._next_try = 0.0
        self.last_error: str | None = None
        self.last_success_at: float | None = None
        # Every template text handed out per name, so a job carrying one of them
        # back (the web UI pre-fills its prompt boxes) is not counted as edited.
        self._served: dict[str, set[str]] = {}
        self._manifest_sent = ""
        self._manifest_at = 0.0
        self._warned: set[tuple] = set()

    # ── fetching ────────────────────────────────────────────────────────────
    def _load_disk(self) -> dict[str, Any] | None:
        try:
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_disk(self, data: dict[str, Any]) -> None:
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(_CACHE_FILE)
        except Exception as exc:
            print(f"[prompts] could not write prompt cache: {exc}")

    def refresh(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if self._data is None:
                self._data = self._load_disk()
            fresh = self._data is not None and now - self._fetched_at < TTL_SECONDS
            if (fresh and not force) or (not force and now < self._next_try) or not _secret():
                return self._data or {}
        try:
            r = requests.get(
                f"{BACKEND}/api/ai/prompts",
                params={"keys": ",".join(sorted(set(PROMPT_KEYS.values()))), "prefix": DYNAMIC_PREFIX},
                headers={"x-decoinks-sso-secret": _secret()},
                timeout=5,
            )
            r.raise_for_status()
            body = (r.json() or {}).get("data") or {}
            data = {
                "revision": body.get("revision"),
                "prompts": {p["prompt_key"]: p for p in body.get("prompts", []) if p.get("prompt_key")},
                "missing": body.get("missing", []),
                "fetched_at": time.time(),
            }
            with self._lock:
                changed = (self._data or {}).get("revision") != data["revision"]
                self._data = data
                self._fetched_at = time.time()
                self.last_error = None
                self.last_success_at = self._fetched_at
            if changed:
                self._save_disk(data)
                print(f"[prompts] loaded {len(data['prompts'])} live prompts from Decoinks (revision {data['revision']})")
            return data
        except Exception as exc:
            with self._lock:
                self.last_error = str(exc)
                self._next_try = time.time() + RETRY_SECONDS
                print(f"[prompts] Decoinks unreachable, using {'cached' if self._data else 'built-in'} prompts: {exc}")
                return self._data or {}

    # ── resolving ───────────────────────────────────────────────────────────
    def template(self, name: str) -> tuple[str, dict[str, Any]]:
        """The template to run for `name`, and where it came from."""
        builtin = _builtin(name)
        key = key_for(name)
        meta: dict[str, Any] = {"name": name, "key": key, "version_id": None, "version": None, "source": "built_in"}
        if not key:
            return builtin, meta
        p = (self.refresh().get("prompts") or {}).get(key)
        text = (p or {}).get("text") or ""
        if not text.strip():
            return builtin, meta
        if is_dynamic(name):
            # Sent to ChatGPT exactly as written, never through str.format.
            if _VAR.search(text):
                meta["rejected"] = {"unknown": sorted(set(_VAR.findall(text))), "missing": []}
                return "", meta
            meta.update(version_id=p["version"]["id"], version=p["version"].get("number"), source="managed")
            self._remember(name, text)
            return text, meta
        tpl = to_python_template(text)
        have, need = template_fields(tpl), template_fields(builtin)
        if have != need:
            # The code fills exactly these placeholders. An extra one would crash
            # str.format mid-job; a missing one would silently drop the designer's
            # input (the text, the chosen number, the colour changes).
            warn = (key, p["version"].get("id"))
            if warn not in self._warned:
                self._warned.add(warn)
                print(f"[prompts] {key} v{p['version'].get('number')} has placeholders {sorted(have)}, "
                      f"but {name} fills {sorted(need)}; using the built-in text")
            meta["rejected"] = {"unknown": sorted(have - need), "missing": sorted(need - have)}
            self._remember(name, builtin)
            return builtin, meta
        meta.update(version_id=p["version"]["id"], version=p["version"].get("number"), source="managed")
        self._remember(name, tpl)
        return tpl, meta

    @staticmethod
    def _norm(text: str) -> str:
        return (text or "").replace("\r\n", "\n").strip()

    def _remember(self, name: str, tpl: str) -> None:
        with self._lock:
            self._served.setdefault(name, set()).add(self._norm(tpl))

    def is_unedited(self, name: str, submitted: str) -> bool:
        """True when `submitted` is text this server served (or the built-in), not a designer's edit."""
        n = self._norm(submitted)
        if not n or n == self._norm(_builtin(name)):
            return True
        with self._lock:
            return n in self._served.get(name, set())

    def snapshot(self, names: list[str]) -> dict[str, Any]:
        templates: dict[str, str] = {}
        meta: dict[str, Any] = {}
        for name in dict.fromkeys(names):
            templates[name], meta[name] = self.template(name)
        return {"templates": templates, "meta": meta}

    def dynamic_operations(self) -> list[dict[str, Any]]:
        """Custom operations published in Prompt Management, in the Custom screen's shape."""
        ops = []
        prompts = self.refresh().get("prompts") or {}
        for key in sorted(prompts):
            op_key = dynamic_op_key(key)
            if not op_key:
                continue
            text, meta = self.template(dynamic_name(op_key))
            if meta["source"] != "managed":
                continue
            p = prompts[key]
            ops.append({"key": op_key, "label": (p.get("name") or op_key)[:80],
                        "desc": (p.get("description") or "From Prompt Management")[:200],
                        "template": text, "prompt_key": key, "from_prompt_management": True})
        return ops

    def manifest(self) -> list[dict[str, Any]]:
        """What this app reads and runs, for Decoinks to check publishes against."""
        rows = []
        names = list(PROMPT_KEYS) + [dynamic_name(o["key"]) for o in self.dynamic_operations()]
        for name in names:
            _, meta = self.template(name)
            rows.append({
                "prompt_key": key_for(name), "template_name": name,
                "placeholders": [] if is_dynamic(name) else sorted(template_fields(_builtin(name))),
                "running": {k: meta.get(k) for k in ("version", "version_id", "source", "rejected") if meta.get(k) is not None},
            })
        rows.append({"prompt_key": DYNAMIC_PREFIX + "*", "template_name": "PM_OP:*",
                     "label": "New Custom operation", "placeholders": [], "running": {}})
        return rows

    def publish_manifest(self, force: bool = False) -> None:
        """Tell Decoinks what is running. Only when it changed, or every 10 minutes."""
        if not _secret():
            return
        rows = self.manifest()
        digest = json.dumps(rows, sort_keys=True)
        if not force and digest == self._manifest_sent and time.time() - self._manifest_at < 600:
            return
        try:
            r = requests.put(f"{BACKEND}/api/ai/apps/{SOURCE_APP}/prompts", json={"prompts": rows},
                             headers={"x-decoinks-sso-secret": _secret()}, timeout=10)
            if r.status_code >= 400:
                print(f"[prompts] Decoinks refused the manifest ({r.status_code}): {r.text[:300]}")
                return
            self._manifest_sent, self._manifest_at = digest, time.time()
        except Exception as exc:
            print(f"[prompts] could not send the manifest: {exc}")

    def status(self) -> dict[str, Any]:
        data = self._data or {}
        return {
            "configured": bool(_secret()),
            "live_prompts": len(data.get("prompts") or {}),
            "revision": data.get("revision"),
            "missing": data.get("missing", []),
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
        }


registry = PromptRegistry()


def prompt_names_for_job(workflow: str, custom_operations: list[str] | None = None,
                         options: list[str] | None = None,
                         text_mode: str | None = None) -> list[str]:
    if workflow == "custom":
        names: list[str] = []
        for op in custom_operations or []:
            if op.startswith(DYNAMIC_OP_PREFIX):
                names.append(dynamic_name(op))
            names.extend(CUSTOM_OPERATION_PROMPTS.get(op, []))
        return names
    if not workflow and options:
        # The edit-options job: the base block, then one block per chosen option.
        return ["BASE_INSTRUCTION"] + [option_template_name(o) for o in options
                                       if option_template_name(o) in PROMPT_KEYS]
    if workflow == "text" and text_mode:
        # Narrow to the prompts this input mode actually runs; unknown modes fall
        # back to the full text superset so nothing is dropped.
        return list(TEXT_MODE_PROMPTS.get(text_mode, WORKFLOW_PROMPTS["text"]))
    return list(WORKFLOW_PROMPTS.get(workflow, []))


def report_runs(runs: list[dict[str, Any]]) -> None:
    """Send run records to Decoinks in the background. Never raises."""
    if not runs or not _secret():
        return

    def _send() -> None:
        for attempt in range(4):
            try:
                r = requests.post(
                    f"{BACKEND}/api/ai/generations",
                    json={"runs": runs},
                    headers={"x-decoinks-sso-secret": _secret()},
                    timeout=10,
                )
                if r.status_code < 500:
                    if r.status_code >= 400:
                        print(f"[prompts] Decoinks refused the run log ({r.status_code}): {r.text[:300]}")
                    return
            except Exception as exc:
                print(f"[prompts] run log attempt {attempt + 1} failed: {exc}")
            time.sleep(3 * (attempt + 1))

    threading.Thread(target=_send, name="prompt-run-log", daemon=True).start()

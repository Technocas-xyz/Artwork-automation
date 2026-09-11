"""Agent token store.

A designer's PC-side agent authenticates to the server with a bearer token in
addition to being on the trusted network. Tokens are stored in a small JSON
file so they survive restarts. Generate one per designer with
make_agent_token.py.

Tokens are keyed on a designer/machine NAME, not on the operator's website
login. Every designer logs into the site as the same shared "admin" account, so
keying on the login handed everyone the identical token and two agents then
collided on it. Keying on the name each PC reports (defaulting to the machine
name) gives every PC a distinct token.
"""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from threading import Lock

_TOKENS_FILE = Path("agent_tokens.json")
_lock = Lock()


def _load() -> dict:
    if not _TOKENS_FILE.exists():
        return {}
    try:
        return json.loads(_TOKENS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    _TOKENS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_token(name: str) -> str:
    """Create and persist a new agent token for `name`. Returns the token."""
    token = "agt_" + secrets.token_urlsafe(24)
    with _lock:
        data = _load()
        data[token] = {"name": name, "created_at": time.time()}
        _save(data)
    return token


def token_name(token: str) -> str | None:
    """Return the designer name for a token, or None if unknown."""
    if not token:
        return None
    with _lock:
        data = _load()
    entry = data.get(token)
    return entry.get("name") if entry else None


def touch_token(token: str) -> None:
    """Record that `token` was just used, for the admin "who's set up" view.

    Cheap best-effort write on every authenticated agent call; failures are
    swallowed so a token store hiccup never breaks a live agent request."""
    if not token:
        return
    try:
        with _lock:
            data = _load()
            entry = data.get(token)
            if entry is None:
                return
            entry["last_seen"] = time.time()
            _save(data)
    except Exception:
        pass


def token_for_name(name: str) -> str | None:
    """Return an existing token issued to `name`, or None."""
    with _lock:
        data = _load()
    for tok, entry in data.items():
        if entry.get("name") == name:
            return tok
    return None


def get_or_create_for_name(name: str) -> str:
    """Return the stable token for `name`, creating one if none exists yet.

    Lets the web UI surface a logged-in user's token without them having to ask.
    """
    existing = token_for_name(name)
    if existing:
        return existing
    return create_token(name)


def list_tokens() -> dict:
    with _lock:
        return _load()


def list_agents() -> list[dict]:
    """Return one entry per issued token for the admin panel: name, last-seen
    and a short token preview. The full token is NOT included — the panel only
    needs to show WHO is set up, not hand every operator every secret."""
    with _lock:
        data = _load()
    out = []
    for tok, entry in data.items():
        out.append({
            "name": entry.get("name", ""),
            "last_seen": entry.get("last_seen"),
            "token_preview": (tok[:10] + "…") if tok else "",
        })
    out.sort(key=lambda e: (e["last_seen"] or 0), reverse=True)
    return out

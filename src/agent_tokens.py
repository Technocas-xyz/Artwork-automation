"""Agent token store.

A designer's PC-side agent authenticates to the server with a bearer token in
addition to being on the trusted network. Tokens are stored in a small JSON
file so they survive restarts. Generate one per designer with
make_agent_token.py.
"""
from __future__ import annotations

import json
import secrets
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
        data[token] = {"name": name}
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

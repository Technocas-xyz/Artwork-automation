"""DB-backed user + agent-token store.

Replaces the users.json / agent_tokens.json files as the source of truth. The
public functions mirror the old src/users.py and src/agent_tokens.py APIs
exactly, so those modules can delegate here and every existing app.py call site
keeps working unchanged. Records are returned as plain dicts with the SAME keys
the old stores used (username, password_hash, display_name, role, created_at,
disabled, last_login, agent_token).
"""
from __future__ import annotations

import secrets
import time
from datetime import datetime, timezone

from sqlalchemy import func, select

from db.models import AgentToken, User
from db.session import get_session

VALID_ROLES = ("admin", "designer")


def _norm(username: str) -> str:
    return (username or "").strip().lower()


def _new_token() -> str:
    return "agt_" + secrets.token_urlsafe(24)


def _epoch(dt: datetime | None) -> float | None:
    return dt.timestamp() if isinstance(dt, datetime) else None


def _as_dict(u: User) -> dict:
    return {
        "username": u.username,
        "password_hash": u.password_hash or "",
        "display_name": u.display_name or "",
        "role": u.role or "designer",
        "created_at": _epoch(u.created_at),
        "disabled": bool(u.disabled),
        "last_login": _epoch(u.last_login),
        "agent_token": u.agent_token or "",
    }


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def ensure_seed_admin() -> None:
    """Seed the admin from .env on an empty users table. Idempotent."""
    from src.auth import APP_USERNAME, APP_PASSWORD_HASH
    with get_session() as s:
        if s.scalar(select(func.count()).select_from(User)):
            return
        uname = _norm(APP_USERNAME) or "admin"
        s.add(User(
            username=uname,
            password_hash=APP_PASSWORD_HASH or "",
            display_name=APP_USERNAME or "Admin",
            role="admin",
            disabled=False,
            agent_token=_new_token(),
        ))
        s.commit()
        print(f"[users] seeded admin account {uname!r} from .env (DB)")


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_user(username: str) -> dict | None:
    with get_session() as s:
        u = s.get(User, _norm(username))
        return _as_dict(u) if u else None


def list_users() -> list[dict]:
    with get_session() as s:
        users = s.scalars(select(User)).all()
    out = []
    for u in users:
        out.append({
            "username": u.username,
            "display_name": u.display_name or "",
            "role": u.role or "designer",
            "created_at": _epoch(u.created_at),
            "last_login": _epoch(u.last_login),
            "disabled": bool(u.disabled),
            "token_preview": (u.agent_token[:10] + "…") if u.agent_token else "",
        })
    out.sort(key=lambda e: (e.get("created_at") or 0))
    return out


def user_for_token(token: str) -> dict | None:
    """Resolve a bearer token to its user record (None for unknown OR disabled —
    disabling a user kills their agent immediately)."""
    if not token:
        return None
    with get_session() as s:
        u = s.scalar(select(User).where(User.agent_token == token))
        if not u or u.disabled:
            return None
        return _as_dict(u)


def verify(username: str, password: str) -> dict | None:
    from src.auth import verify_password
    with get_session() as s:
        u = s.get(User, _norm(username))
        if not u or u.disabled:
            return None
        if not verify_password(password, u.password_hash or ""):
            return None
        u.last_login = datetime.now(timezone.utc)
        s.commit()
        return _as_dict(u)


def authenticate_agent(email: str, password: str) -> tuple[str | None, str]:
    from src.auth import verify_password
    with get_session() as s:
        u = s.get(User, _norm(email))
        if not u or not verify_password(password, u.password_hash or ""):
            return None, "bad_credentials"
        if u.disabled:
            return None, "disabled"
        u.last_login = datetime.now(timezone.utc)
        s.commit()
        return (u.agent_token or ""), "ok"


def count() -> int:
    with get_session() as s:
        return int(s.scalar(select(func.count()).select_from(User)) or 0)


# ---------------------------------------------------------------------------
# Mutations (admin)
# ---------------------------------------------------------------------------

def is_valid_email(email: str) -> bool:
    import re
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (email or "").strip()))


def create_user(username: str, password: str, display_name: str,
                role: str = "designer") -> dict:
    from set_password import hash_password
    uname = _norm(username)
    if not uname:
        raise ValueError("Email is required.")
    if not is_valid_email(uname):
        raise ValueError(f"{username!r} is not a valid email address.")
    if role not in VALID_ROLES:
        raise ValueError(f"Role must be one of {VALID_ROLES}.")
    with get_session() as s:
        if s.get(User, uname) is not None:
            raise ValueError(f"A user with the email {uname!r} already exists.")
        u = User(
            username=uname,
            password_hash=hash_password(password) if password else "",
            display_name=(display_name or username).strip(),
            role=role,
            disabled=False,
            agent_token=_new_token(),
        )
        s.add(u)
        s.commit()
        print(f"[users] created user {uname!r} (role={role}) (DB)")
        return _as_dict(u)


def set_disabled(username: str, disabled: bool) -> dict | None:
    with get_session() as s:
        u = s.get(User, _norm(username))
        if not u:
            return None
        u.disabled = bool(disabled)
        s.commit()
        return _as_dict(u)


def set_role(username: str, role: str) -> dict | None:
    if role not in VALID_ROLES:
        raise ValueError(f"Role must be one of {VALID_ROLES}.")
    with get_session() as s:
        u = s.get(User, _norm(username))
        if not u:
            return None
        u.role = role
        s.commit()
        return _as_dict(u)


def reset_password(username: str, new_password: str) -> dict | None:
    from set_password import hash_password
    if not new_password:
        raise ValueError("A new password is required.")
    with get_session() as s:
        u = s.get(User, _norm(username))
        if not u:
            return None
        u.password_hash = hash_password(new_password)
        s.commit()
        return _as_dict(u)


def regenerate_token(username: str) -> str | None:
    with get_session() as s:
        u = s.get(User, _norm(username))
        if not u:
            return None
        token = _new_token()
        u.agent_token = token
        s.commit()
        print(f"[users] regenerated token for {_norm(username)!r} (DB)")
        return token


def touch_login(username: str) -> None:
    try:
        with get_session() as s:
            u = s.get(User, _norm(username))
            if u:
                u.last_login = datetime.now(timezone.utc)
                s.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Standalone agent tokens (migrated from agent_tokens.json)
# ---------------------------------------------------------------------------

def create_agent_token(name: str) -> str:
    token = _new_token()
    with get_session() as s:
        s.add(AgentToken(token=token, name=name))
        s.commit()
    return token


def token_name(token: str) -> str | None:
    """Return the owner name for a token — checks user tokens first, then the
    standalone agent_tokens table (so both kinds resolve)."""
    if not token:
        return None
    with get_session() as s:
        u = s.scalar(select(User).where(User.agent_token == token))
        if u:
            return u.display_name or u.username
        t = s.get(AgentToken, token)
        return t.name if t else None


def touch_token(token: str) -> None:
    if not token:
        return
    try:
        with get_session() as s:
            t = s.get(AgentToken, token)
            if t is not None:
                t.last_seen = datetime.now(timezone.utc)
                s.commit()
    except Exception:
        pass


def token_for_name(name: str) -> str | None:
    with get_session() as s:
        t = s.scalar(select(AgentToken).where(AgentToken.name == name))
        return t.token if t else None


def get_or_create_for_name(name: str) -> str:
    existing = token_for_name(name)
    return existing if existing else create_agent_token(name)


def list_agent_tokens() -> list[dict]:
    with get_session() as s:
        rows = s.scalars(select(AgentToken)).all()
    out = [{
        "name": t.name or "",
        "last_seen": _epoch(t.last_seen),
        "token_preview": (t.token[:10] + "…") if t.token else "",
    } for t in rows]
    out.sort(key=lambda e: (e["last_seen"] or 0), reverse=True)
    return out

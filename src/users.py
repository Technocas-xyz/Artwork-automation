"""User store — multi-user login and per-user identity.

Everyone used to sign in as the same shared "admin" account, so tokens and jobs
could not be attributed to a person. This store gives every operator their own
account and, crucially, their own agent token: creating a user issues the token,
mapped to them; regenerating replaces it; disabling a user invalidates it
immediately so a departed designer's agent stops working.

Persistence mirrors src/agent_tokens.py exactly — a small JSON file beside it,
guarded by a module-level lock, load/save helpers of the same shape. No database
in this project.

Each record (keyed by username) holds:
    username        the login (store key; lowercased)
    password_hash   PBKDF2-SHA256 salt:key hex — SAME format as set_password.py
    display_name    shown in the UI ("Sahar Shah")
    role            "admin" | "designer"
    created_at      epoch seconds
    disabled        bool — a disabled user cannot log in and their token is dead
    last_login      epoch seconds of the last successful login (None until then)
    agent_token     the user's bearer token ("agt_..."), issued on creation

The seed admin: on first run (empty store) we import the existing
APP_USERNAME / APP_PASSWORD_HASH from .env so nobody is locked out during the
change. That account is role=admin.
"""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from threading import Lock

# Reuse the EXACT hashing set_password.py uses, so a hash made either way is
# interoperable with src.auth.verify_password (PBKDF2-SHA256, 260000, salt:key).
from set_password import hash_password

# Store beside the code (like agent_tokens.json lives at the project root). We
# resolve relative to the project root so it sits next to agent_tokens.json.
_USERS_FILE = Path("users.json")
_lock = Lock()

VALID_ROLES = ("admin", "designer")

# Usernames are email addresses. A pragmatic single-@ check with a dotted domain
# — enough to reject obvious typos without trying to fully implement RFC 5322.
import re as _re
_EMAIL_RE = _re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match((email or "").strip()))


def _load() -> dict:
    if not _USERS_FILE.exists():
        return {}
    try:
        return json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    _USERS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _new_token() -> str:
    """Same token shape as agent_tokens.create_token."""
    return "agt_" + secrets.token_urlsafe(24)


def _norm(username: str) -> str:
    return (username or "").strip().lower()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def ensure_seed_admin() -> None:
    """On first run (empty store) seed the admin from .env so nobody is locked
    out mid-migration. Idempotent: does nothing once any user exists."""
    from src.auth import APP_USERNAME, APP_PASSWORD_HASH
    with _lock:
        data = _load()
        if data:
            return
        uname = _norm(APP_USERNAME) or "admin"
        data[uname] = {
            "username": uname,
            # Reuse the existing env hash verbatim so the current password keeps
            # working. If it's blank the admin must reset via set_password.py.
            "password_hash": APP_PASSWORD_HASH or "",
            "display_name": APP_USERNAME or "Admin",
            "role": "admin",
            "created_at": time.time(),
            "disabled": False,
            "last_login": None,
            "agent_token": _new_token(),
        }
        _save(data)
        print(f"[users] seeded admin account {uname!r} from .env")


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_user(username: str) -> dict | None:
    """Return a copy of the user record, or None."""
    with _lock:
        data = _load()
    u = data.get(_norm(username))
    return dict(u) if u else None


def list_users() -> list[dict]:
    """All users (without password hashes or full tokens) for the admin panel."""
    with _lock:
        data = _load()
    out = []
    for u in data.values():
        out.append({
            "username": u.get("username", ""),
            "display_name": u.get("display_name", ""),
            "role": u.get("role", "designer"),
            "created_at": u.get("created_at"),
            "last_login": u.get("last_login"),
            "disabled": bool(u.get("disabled")),
            # Never the full token here — only whether one exists + a preview.
            "token_preview": (u.get("agent_token", "")[:10] + "…") if u.get("agent_token") else "",
        })
    out.sort(key=lambda e: (e.get("created_at") or 0))
    return out


def user_for_token(token: str) -> dict | None:
    """Resolve a bearer token to its (copy of the) user record, or None.

    Returns None for an unknown token AND for a DISABLED user's token — so
    disabling a user kills their agent immediately, with no separate step."""
    if not token:
        return None
    with _lock:
        data = _load()
    for u in data.values():
        if u.get("agent_token") == token:
            if u.get("disabled"):
                return None
            return dict(u)
    return None


def verify(username: str, password: str) -> dict | None:
    """Validate a login. Returns the user record on success, else None.

    A disabled user never authenticates. Records last_login on success."""
    from src.auth import verify_password
    uname = _norm(username)
    with _lock:
        data = _load()
        u = data.get(uname)
        if not u or u.get("disabled"):
            return None
        if not verify_password(password, u.get("password_hash", "")):
            return None
        u["last_login"] = time.time()
        _save(data)
        return dict(u)


def authenticate_agent(email: str, password: str) -> tuple[str | None, str]:
    """Authenticate an agent by email + password and return its agent token.

    Returns (token, "ok") on success, or (None, reason) where reason is one of:
      "bad_credentials" — unknown email or wrong password
      "disabled"        — the account exists and the password is correct but the
                          user is disabled (so the caller can say so plainly, and
                          a disabled user is rejected the moment they try again)

    Distinguishing "disabled" from "bad_credentials" only leaks that a valid
    email+password pair belongs to a disabled account, which is acceptable here:
    the designer needs to know whether to fix their password or ask an admin to
    re-enable them. Records last_login on success."""
    from src.auth import verify_password
    uname = _norm(email)
    with _lock:
        data = _load()
        u = data.get(uname)
        if not u or not verify_password(password, u.get("password_hash", "")):
            return None, "bad_credentials"
        if u.get("disabled"):
            return None, "disabled"
        u["last_login"] = time.time()
        _save(data)
        return u.get("agent_token") or "", "ok"


# ---------------------------------------------------------------------------
# Mutations (admin)
# ---------------------------------------------------------------------------

def create_user(username: str, password: str, display_name: str,
                role: str = "designer") -> dict:
    """Create a user and issue their agent token. Returns the full record
    INCLUDING the plaintext agent_token (shown once on the admin screen).

    The username IS the email address: it must be a valid email and unique — the
    same email can never be registered twice.

    Raises ValueError on an invalid email, a duplicate email (naming the
    conflict), or an invalid role."""
    uname = _norm(username)
    if not uname:
        raise ValueError("Email is required.")
    if not is_valid_email(uname):
        raise ValueError(f"{username!r} is not a valid email address.")
    if role not in VALID_ROLES:
        raise ValueError(f"Role must be one of {VALID_ROLES}.")
    with _lock:
        data = _load()
        if uname in data:
            raise ValueError(f"A user with the email {uname!r} already exists.")
        token = _new_token()
        rec = {
            "username": uname,
            "password_hash": hash_password(password) if password else "",
            "display_name": (display_name or username).strip(),
            "role": role,
            "created_at": time.time(),
            "disabled": False,
            "last_login": None,
            "agent_token": token,
        }
        data[uname] = rec
        _save(data)
        print(f"[users] created user {uname!r} (role={role})")
        return dict(rec)


def set_disabled(username: str, disabled: bool) -> dict | None:
    """Disable or re-enable a user. Disabling invalidates their token at once
    (user_for_token/verify both reject a disabled user)."""
    uname = _norm(username)
    with _lock:
        data = _load()
        u = data.get(uname)
        if not u:
            return None
        u["disabled"] = bool(disabled)
        _save(data)
        return dict(u)


def set_role(username: str, role: str) -> dict | None:
    if role not in VALID_ROLES:
        raise ValueError(f"Role must be one of {VALID_ROLES}.")
    uname = _norm(username)
    with _lock:
        data = _load()
        u = data.get(uname)
        if not u:
            return None
        u["role"] = role
        _save(data)
        return dict(u)


def reset_password(username: str, new_password: str) -> dict | None:
    if not new_password:
        raise ValueError("A new password is required.")
    uname = _norm(username)
    with _lock:
        data = _load()
        u = data.get(uname)
        if not u:
            return None
        u["password_hash"] = hash_password(new_password)
        _save(data)
        return dict(u)


def regenerate_token(username: str) -> str | None:
    """Replace the user's agent token. Returns the NEW token (shown once)."""
    uname = _norm(username)
    with _lock:
        data = _load()
        u = data.get(uname)
        if not u:
            return None
        token = _new_token()
        u["agent_token"] = token
        _save(data)
        print(f"[users] regenerated token for {uname!r}")
        return token


def touch_login(username: str) -> None:
    """Best-effort last_login update (used by cookie-only paths if needed)."""
    uname = _norm(username)
    try:
        with _lock:
            data = _load()
            u = data.get(uname)
            if not u:
                return
            u["last_login"] = time.time()
            _save(data)
    except Exception:
        pass


def count() -> int:
    with _lock:
        return len(_load())

"""Authentication middleware for the Artwork Generator app.

Uses signed cookies with HMAC, PBKDF2 password hashing, and rate limiting.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path

from fastapi import Request

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed — rely on env vars being set

APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD_HASH = os.getenv("APP_PASSWORD_HASH", "")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "default-insecure-change-me")

COOKIE_NAME = "session"
COOKIE_MAX_AGE = 86400 * 7  # 7 days

# Rate limiting: {ip: (fail_count, lockout_until)}
_login_attempts: dict[str, tuple[int, float]] = {}
MAX_FAILURES = 5
LOCKOUT_SECONDS = 300


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored PBKDF2-SHA256 hash (salt:key hex)."""
    if not stored_hash or ":" not in stored_hash:
        return False
    try:
        salt_hex, key_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260000)
        return hmac.compare_digest(key.hex(), key_hex)
    except Exception:
        return False


def sign_cookie(username: str) -> str:
    """Create a signed session cookie value."""
    payload = json.dumps({"u": username, "t": int(time.time())})
    sig = hmac.HMAC(APP_SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return payload + "." + sig


def verify_cookie(value: str) -> str | None:
    """Verify a signed session cookie. Returns username or None."""
    if not value or "." not in value:
        return None
    parts = value.rsplit(".", 1)
    if len(parts) != 2:
        return None
    payload, sig = parts
    expected = hmac.HMAC(APP_SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(payload)
        if time.time() - data.get("t", 0) > COOKIE_MAX_AGE:
            return None
        return data.get("u")
    except Exception:
        return None


def check_rate_limit(ip: str) -> bool:
    """Returns True if the IP is allowed to attempt login."""
    entry = _login_attempts.get(ip)
    if not entry:
        return True
    count, lockout_until = entry
    if lockout_until and time.time() < lockout_until:
        return False
    if lockout_until and time.time() >= lockout_until:
        del _login_attempts[ip]
        return True
    return True


def record_failure(ip: str) -> None:
    """Record a failed login attempt."""
    entry = _login_attempts.get(ip, (0, 0.0))
    count = entry[0] + 1
    if count >= MAX_FAILURES:
        _login_attempts[ip] = (count, time.time() + LOCKOUT_SECONDS)
    else:
        _login_attempts[ip] = (count, 0.0)


def record_success(ip: str) -> None:
    """Clear rate limit on successful login."""
    _login_attempts.pop(ip, None)


def get_current_user(request: Request) -> str | None:
    """Extract username from session cookie. Returns None if not authenticated."""
    cookie = request.cookies.get(COOKIE_NAME)
    if not cookie:
        return None
    return verify_cookie(cookie)


# Paths that don't require auth
PUBLIC_PATHS = {"/login", "/api/auth/login", "/api/auth/logout"}
# /input/ serves uploaded artwork previews as <img> tags. Like /static/, these
# are opaque, non-sensitive filenames and must not be gated by the auth
# middleware — otherwise a missing/late cookie turns the image request into a
# 302 -> /login and the browser renders a broken-image icon.
PUBLIC_PREFIXES = ("/static/", "/input/")

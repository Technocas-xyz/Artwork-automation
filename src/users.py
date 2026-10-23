"""User store — now backed by PostgreSQL.

This module used to persist users in users.json. It now DELEGATES every call to
db.user_store so the rest of the app (which imports `from src import users`) is
unchanged while the source of truth is the `users` table. The one-time import of
an existing users.json into the DB is handled by scripts/first_start_import.py.

The public API (get_user, list_users, user_for_token, verify,
authenticate_agent, create_user, set_disabled, set_role, reset_password,
regenerate_token, touch_login, count, ensure_seed_admin, is_valid_email,
VALID_ROLES) is preserved exactly.
"""
from __future__ import annotations

from db import user_store as _store

VALID_ROLES = _store.VALID_ROLES


def is_valid_email(email: str) -> bool:
    return _store.is_valid_email(email)


def ensure_seed_admin() -> None:
    return _store.ensure_seed_admin()


def get_user(username: str):
    return _store.get_user(username)


def list_users():
    return _store.list_users()


def user_for_token(token: str):
    return _store.user_for_token(token)


def verify(username: str, password: str):
    return _store.verify(username, password)


def authenticate_agent(email: str, password: str):
    return _store.authenticate_agent(email, password)


def create_user(username: str, password: str, display_name: str, role: str = "designer"):
    return _store.create_user(username, password, display_name, role)


def set_disabled(username: str, disabled: bool):
    return _store.set_disabled(username, disabled)


def set_role(username: str, role: str):
    return _store.set_role(username, role)


def reset_password(username: str, new_password: str):
    return _store.reset_password(username, new_password)


def regenerate_token(username: str):
    return _store.regenerate_token(username)


def touch_login(username: str) -> None:
    return _store.touch_login(username)


def count() -> int:
    return _store.count()

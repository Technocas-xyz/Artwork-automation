"""Agent token store — now backed by PostgreSQL.

Delegates to db.user_store so existing callers (`from src import agent_tokens`)
are unchanged while tokens live in the DB. User-owned tokens live on the users
table; standalone machine-named tokens live in agent_tokens. token_name() and
touch_token() resolve across both.
"""
from __future__ import annotations

from db import user_store as _store


def create_token(name: str) -> str:
    return _store.create_agent_token(name)


def token_name(token: str):
    return _store.token_name(token)


def touch_token(token: str) -> None:
    return _store.touch_token(token)


def token_for_name(name: str):
    return _store.token_for_name(name)


def get_or_create_for_name(name: str) -> str:
    return _store.get_or_create_for_name(name)


def list_tokens() -> dict:
    # Legacy shape: {token: {...}}. We no longer expose raw tokens in bulk; the
    # admin view uses list_agents(). Return an empty map for any old caller.
    return {}


def list_agents() -> list[dict]:
    return _store.list_agent_tokens()

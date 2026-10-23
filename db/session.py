"""Engine + session factory, driven by the DATABASE_URL env var.

Keeps local dev working without Docker: set DATABASE_URL to a local Postgres
(e.g. postgresql+psycopg://user:pass@localhost:5432/artwork). In Compose it
points at the `db` service.

The engine uses pool_pre_ping so a connection dropped by a db restart is
transparently recycled rather than raising on the next query — important
because the app process outlives db restarts (verification step: restart the
db container, data still served).
"""
from __future__ import annotations

import os
import time

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# A single default is deliberately NOT a silent localhost — a missing
# DATABASE_URL is a deployment mistake we want surfaced. But to keep imports
# cheap (and tooling/tests importable without a DB), we fall back to a local
# dev URL and only connect lazily.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://artwork:artwork@localhost:5432/artwork",
)

_engine = None
_SessionLocal: sessionmaker | None = None


def engine():
    """Lazily create the process-wide engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,      # survive db restarts without app errors
            pool_size=10,
            max_overflow=20,
            future=True,
        )
    return _engine


def _session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=engine(), class_=Session, expire_on_commit=False, future=True
        )
    return _SessionLocal


def get_session() -> Session:
    """A new ORM session. Caller is responsible for closing it (use the
    context manager ``session_scope`` for automatic commit/rollback/close)."""
    return _session_factory()()


class session_scope:
    """Context manager: commit on success, rollback on error, always close.

    Usage:
        with session_scope() as s:
            s.add(obj)
    """

    def __enter__(self) -> Session:
        self._s = get_session()
        return self._s

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._s.commit()
            else:
                self._s.rollback()
        finally:
            self._s.close()
        return False


def wait_for_db(max_seconds: float = 60.0, interval: float = 2.0) -> bool:
    """Block until the database answers SELECT 1, or the timeout elapses.

    Used by the entrypoint (before alembic) and at app startup so the app comes
    up cleanly on `docker compose up` where the db may still be initialising.
    Returns True if the db became reachable, False on timeout."""
    deadline = time.time() + max_seconds
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # pragma: no cover - timing dependent
            last_err = exc
            print(f"[db] not ready yet ({exc.__class__.__name__}); retrying…")
            time.sleep(interval)
    print(f"[db] still unreachable after {max_seconds:.0f}s: {last_err}")
    return False

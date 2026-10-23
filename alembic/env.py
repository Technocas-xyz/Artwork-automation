"""Alembic environment.

Reads the database URL from the DATABASE_URL env var (never from alembic.ini),
and uses the ORM metadata in db.models so `alembic revision --autogenerate`
and `alembic upgrade head` stay in sync with the models.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the project root importable (db/, config/, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import Base  # noqa: E402

config = context.config

# DATABASE_URL wins over anything in the ini. Fall back to the dev default used
# by db.session so offline SQL generation works without the env set.
_url = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://artwork:artwork@localhost:5432/artwork",
)
config.set_main_option("sqlalchemy.url", _url)

if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        pass

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

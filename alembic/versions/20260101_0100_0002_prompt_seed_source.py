"""add prompts.seed_source

Records where each prompt's text originally came from (decoinks / config /
config(forced) / edited), so it is auditable on the Prompts page.

Revision ID: 0002_seed_source
Revises: 0001_initial
Create Date: 2026-01-01 01:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002_seed_source"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("prompts", sa.Column("seed_source", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("prompts", "seed_source")

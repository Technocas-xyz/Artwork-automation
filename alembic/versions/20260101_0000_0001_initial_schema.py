"""initial schema: users, agent_tokens, jobs, job_steps, artwork_files,
vault_saves, prompts, prompt_versions, prompt_runs

Revision ID: 0001_initial
Revises:
Create Date: 2026-01-01 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("username", sa.String(320), primary_key=True),
        sa.Column("password_hash", sa.Text(), server_default=""),
        sa.Column("display_name", sa.String(200), server_default=""),
        sa.Column("role", sa.String(20), server_default="designer"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("disabled", sa.Boolean(), server_default=sa.false()),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.Column("agent_token", sa.String(80), nullable=True),
    )
    op.create_index("ix_users_agent_token", "users", ["agent_token"], unique=True)

    op.create_table(
        "agent_tokens",
        sa.Column("token", sa.String(80), primary_key=True),
        sa.Column("name", sa.String(200), server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("workflow", sa.String(40)),
        sa.Column("status", sa.String(40)),
        sa.Column("stage", sa.Integer(), server_default="0"),
        sa.Column("stage_label", sa.Text(), server_default=""),
        sa.Column("text_mode", sa.String(40), nullable=True),
        sa.Column("created_by", sa.String(320), nullable=True),
        sa.Column("created_by_name", sa.String(200), server_default=""),
        sa.Column("client", sa.String(300), server_default=""),
        sa.Column("job_number", sa.String(120), server_default=""),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("interrupted", sa.Boolean(), server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_progress_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_by_agent", sa.String(32), nullable=True),
        sa.Column("claimed_by_name", sa.String(200), server_default=""),
        sa.Column("request", JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("ui_state", JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("prompt_bundle", JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("agent_log", JSONB, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("ix_jobs_workflow", "jobs", ["workflow"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_created_by", "jobs", ["created_by"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])
    op.create_index("ix_jobs_job_number", "jobs", ["job_number"])
    op.create_index("ix_jobs_claimed_by_agent", "jobs", ["claimed_by_agent"])
    op.create_index("ix_jobs_created_by_created_at", "jobs", ["created_by", "created_at"])

    op.create_table(
        "job_steps",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(32), sa.ForeignKey("jobs.id", ondelete="CASCADE")),
        sa.Column("step_index", sa.Integer(), server_default="0"),
        sa.Column("op", sa.String(80), nullable=True),
        sa.Column("label", sa.Text(), server_default=""),
        sa.Column("method", sa.String(40), nullable=True),
        sa.Column("prompt_key", sa.String(120), nullable=True),
        sa.Column("prompt_version", sa.Integer(), nullable=True),
        sa.Column("prompt_source", sa.String(20), nullable=True),
        sa.Column("prompt_text_used", sa.Text(), nullable=True),
        sa.Column("operator_override_text", sa.Text(), nullable=True),
        sa.Column("result_file", sa.Text(), nullable=True),
        sa.Column("similarity", JSONB, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_job_steps_job_id", "job_steps", ["job_id"])
    op.create_index("ix_job_steps_job_step", "job_steps", ["job_id", "step_index"])

    op.create_table(
        "artwork_files",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(32), sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("step_id", sa.BigInteger(), sa.ForeignKey("job_steps.id", ondelete="SET NULL"), nullable=True),
        sa.Column("path", sa.Text()),
        sa.Column("role", sa.String(40), server_default="result"),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("dpi", sa.Integer(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("derived_from_id", sa.BigInteger(), sa.ForeignKey("artwork_files.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_artwork_files_job_id", "artwork_files", ["job_id"])
    op.create_index("ix_artwork_files_sha256", "artwork_files", ["sha256"])
    op.create_index("ix_artwork_files_derived_from_id", "artwork_files", ["derived_from_id"])

    op.create_table(
        "vault_saves",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(32), sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("client", sa.String(300), server_default=""),
        sa.Column("job_number", sa.String(120), server_default=""),
        sa.Column("run_number", sa.Integer(), nullable=True),
        sa.Column("vault_folder", sa.Text(), server_default=""),
        sa.Column("file_name", sa.Text(), server_default=""),
        sa.Column("lifecycle", sa.String(20), nullable=True),
        sa.Column("customer", sa.String(300), nullable=True),
        sa.Column("saved_by", sa.String(320), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_vault_saves_job_id", "vault_saves", ["job_id"])

    op.create_table(
        "prompts",
        sa.Column("key", sa.String(160), primary_key=True),
        sa.Column("workflow", sa.String(60), server_default=""),
        sa.Column("label", sa.Text(), server_default=""),
        sa.Column("text", sa.Text(), server_default=""),
        sa.Column("placeholders", JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("active", sa.Boolean(), server_default=sa.true()),
        sa.Column("version", sa.Integer(), server_default="1"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_by", sa.String(320), nullable=True),
    )
    op.create_index("ix_prompts_workflow", "prompts", ["workflow"])

    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(160), sa.ForeignKey("prompts.key", ondelete="CASCADE")),
        sa.Column("version", sa.Integer()),
        sa.Column("text", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("created_by", sa.String(320), nullable=True),
        sa.UniqueConstraint("key", "version", name="uq_prompt_versions_key_version"),
    )
    op.create_index("ix_prompt_versions_key", "prompt_versions", ["key"])

    op.create_table(
        "prompt_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(32), nullable=True),
        sa.Column("prompt_key", sa.String(160), nullable=True),
        sa.Column("prompt_version_id", sa.String(80), nullable=True),
        sa.Column("prompt_source", sa.String(20), nullable=True),
        sa.Column("status", sa.String(40), nullable=True),
        sa.Column("resolved_prompt", sa.Text(), nullable=True),
        sa.Column("input_variables", JSONB, nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("output_reference", sa.Text(), nullable=True),
        sa.Column("output_files", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_prompt_runs_job_id", "prompt_runs", ["job_id"])
    op.create_index("ix_prompt_runs_prompt_key", "prompt_runs", ["prompt_key"])


def downgrade() -> None:
    op.drop_table("prompt_runs")
    op.drop_table("prompt_versions")
    op.drop_table("prompts")
    op.drop_table("vault_saves")
    op.drop_table("artwork_files")
    op.drop_table("job_steps")
    op.drop_table("jobs")
    op.drop_table("agent_tokens")
    op.drop_table("users")

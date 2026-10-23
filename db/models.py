"""SQLAlchemy 2.x ORM models — the persistent schema.

Design goals:
  * Faithfully persist the ~200-field in-memory job dict without losing anything:
    stable, queried fields are real columns; the long tail lives in JSONB blobs
    (request / ui_state / prompt_bundle), which the existing `_public_job`
    shaper already filters for the API.
  * Prompts are the DB's own source of truth (prompts + prompt_versions).
  * Artwork lineage (Detected -> Extracted -> Reconstructed -> Ratio -> Print
    Ready) can be added later with zero rewrite via artwork_files.derived_from_id.
  * Keep the 2s poll fast: indexes on jobs(status), jobs(created_by),
    jobs(created_at).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Users + agent tokens (migrated out of users.json / agent_tokens.json)
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(320), primary_key=True)  # email, lowercased
    password_hash: Mapped[str] = mapped_column(Text, default="")
    display_name: Mapped[str] = mapped_column(String(200), default="")
    role: Mapped[str] = mapped_column(String(20), default="designer")  # admin | designer
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    agent_token: Mapped[str | None] = mapped_column(String(80), unique=True, nullable=True, index=True)


class AgentToken(Base):
    """Agent tokens that are NOT tied to a user account (migrated from
    agent_tokens.json, which keyed on a machine/designer name). User-owned
    tokens live on users.agent_token; this table covers the standalone ones."""
    __tablename__ = "agent_tokens"

    token: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Jobs + steps + artwork files + vault saves
# ---------------------------------------------------------------------------

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # uuid4().hex[:12]
    workflow: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    stage: Mapped[int] = mapped_column(Integer, default=0)
    stage_label: Mapped[str] = mapped_column(Text, default="")
    text_mode: Mapped[str | None] = mapped_column(String(40), nullable=True)

    created_by: Mapped[str | None] = mapped_column(String(320), index=True, nullable=True)
    created_by_name: Mapped[str] = mapped_column(String(200), default="")
    client: Mapped[str] = mapped_column(String(300), default="")
    job_number: Mapped[str] = mapped_column(String(120), default="", index=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    interrupted: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_progress_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    claimed_by_agent: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    claimed_by_name: Mapped[str] = mapped_column(String(200), default="")

    # The long tail of the in-memory job dict, kept faithfully as JSONB:
    #   request      — echoed inputs (options, params, custom_operations,
    #                  custom_prompts, geometry_*, aspect_*, text/replace fields…)
    #   ui_state     — awaiting_input, selection_prompt, choices, pause answers,
    #                  stage_images, images, final_names, vault_folder, *_similarity…
    #   prompt_bundle— managed_prompts, prompt_meta, prompt_edited (what was injected)
    #   agent_log    — capped console lines + warnings
    request: Mapped[dict] = mapped_column(JSONB, default=dict)
    ui_state: Mapped[dict] = mapped_column(JSONB, default=dict)
    prompt_bundle: Mapped[dict] = mapped_column(JSONB, default=dict)
    agent_log: Mapped[dict] = mapped_column(JSONB, default=dict)

    steps: Mapped[list["JobStep"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobStep.step_index"
    )

    __table_args__ = (
        Index("ix_jobs_created_by_created_at", "created_by", "created_at"),
    )


class JobStep(Base):
    __tablename__ = "job_steps"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    step_index: Mapped[int] = mapped_column(Integer, default=0)
    op: Mapped[str | None] = mapped_column(String(80), nullable=True)
    label: Mapped[str] = mapped_column(Text, default="")
    method: Mapped[str | None] = mapped_column(String(40), nullable=True)

    # The prompt actually used on this step and WHERE it came from.
    prompt_key: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prompt_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_source: Mapped[str | None] = mapped_column(String(20), nullable=True)  # override|db|built_in
    prompt_text_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    operator_override_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    result_file: Mapped[str | None] = mapped_column(Text, nullable=True)
    similarity: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # incl. overall_pct

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    job: Mapped["Job"] = relationship(back_populates="steps")

    __table_args__ = (
        Index("ix_job_steps_job_step", "job_id", "step_index"),
    )


class ArtworkFile(Base):
    __tablename__ = "artwork_files"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"), index=True, nullable=True)
    step_id: Mapped[int | None] = mapped_column(ForeignKey("job_steps.id", ondelete="SET NULL"), nullable=True)

    path: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(40), default="result")  # original|baseline|result|final|annotated|grid|mockup…
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dpi: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    # Lineage: a later stage's file points at the file it was derived from, so
    # Detected -> Extracted -> Reconstructed -> Ratio -> Print Ready chains later
    # without a schema change.
    derived_from_id: Mapped[int | None] = mapped_column(
        ForeignKey("artwork_files.id", ondelete="SET NULL"), index=True, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class VaultSave(Base):
    __tablename__ = "vault_saves"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"), index=True, nullable=True)
    client: Mapped[str] = mapped_column(String(300), default="")
    job_number: Mapped[str] = mapped_column(String(120), default="")
    run_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vault_folder: Mapped[str] = mapped_column(Text, default="")
    file_name: Mapped[str] = mapped_column(Text, default="")
    lifecycle: Mapped[str | None] = mapped_column(String(20), nullable=True)  # WRK|FNL|FNLA…
    customer: Mapped[str | None] = mapped_column(String(300), nullable=True)
    saved_by: Mapped[str | None] = mapped_column(String(320), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Prompts (DB is the source of truth) + version history
# ---------------------------------------------------------------------------

class Prompt(Base):
    __tablename__ = "prompts"

    # The Decoinks-style key is the stable identity (e.g. "AIS.RATIO.PLAN").
    key: Mapped[str] = mapped_column(String(160), primary_key=True)
    workflow: Mapped[str] = mapped_column(String(60), default="", index=True)
    label: Mapped[str] = mapped_column(Text, default="")
    # Canonical text uses Decoinks-style {{name}} placeholders.
    text: Mapped[str] = mapped_column(Text, default="")
    placeholders: Mapped[dict] = mapped_column(JSONB, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(Integer, default=1)  # current version number
    # Where this key's text originally came from when it entered the DB:
    #   "decoinks" | "config" | "config(forced)" | "edited" (set when an admin
    #   saves/restores via the Prompts page). Auditable on the Prompts page.
    seed_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    updated_by: Mapped[str | None] = mapped_column(String(320), nullable=True)


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(ForeignKey("prompts.key", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_by: Mapped[str | None] = mapped_column(String(320), nullable=True)

    __table_args__ = (
        UniqueConstraint("key", "version", name="uq_prompt_versions_key_version"),
    )


class PromptRun(Base):
    """Our own copy of the per-job prompt-usage record that is also POSTed to
    Decoinks (/api/ai/generations). Decoinks logging stays unchanged; this is
    the local mirror so the data survives here too."""
    __tablename__ = "prompt_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    prompt_key: Mapped[str | None] = mapped_column(String(160), index=True, nullable=True)
    prompt_version_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    prompt_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    resolved_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_variables: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_files: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

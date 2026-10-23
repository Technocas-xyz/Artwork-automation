"""One-time, idempotent first-start imports.

Run by the Docker entrypoint after `alembic upgrade head`, on every boot. Each
step is a no-op once its data is already present, so running it repeatedly is
safe.

Steps:
  1. Import users.json      -> users table (skip existing usernames).
  2. Import agent_tokens.json -> agent_tokens table (skip existing tokens).
  3. Seed prompts           -> prompts + prompt_versions (Decoinks-wins except
                               CUSTOM_ASPECT_ADVICE; see scripts.prompt_seed).
  4. Scan vault/<client>/<task>/run_N folders -> vault_saves index (skip rows
     already present for a file path).
  5. Rename imported users.json / agent_tokens.json to *.imported as a backup
     (never delete) — ONLY after a successful user/token import.

Nothing here raises fatally: a failed sub-step is logged and the others proceed,
so a cosmetic import hiccup never blocks app startup.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from sqlalchemy import select

from db.models import AgentToken, Prompt, User, VaultSave
from db.session import get_session

USERS_JSON = Path("users.json")
TOKENS_JSON = Path("agent_tokens.json")
VAULT_DIR = Path("vault")


def _dt(epoch):
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc)
    except Exception:
        return None


def import_users(session) -> int:
    if not USERS_JSON.exists():
        return 0
    try:
        data = json.loads(USERS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[import] could not read {USERS_JSON}: {exc}")
        return 0
    existing = set(session.scalars(select(User.username)).all())
    added = 0
    for uname, rec in (data or {}).items():
        key = (rec.get("username") or uname or "").strip().lower()
        if not key or key in existing:
            continue
        session.add(User(
            username=key,
            password_hash=rec.get("password_hash", ""),
            display_name=rec.get("display_name", ""),
            role=rec.get("role", "designer"),
            disabled=bool(rec.get("disabled", False)),
            last_login=_dt(rec.get("last_login")),
            agent_token=rec.get("agent_token") or None,
        ))
        added += 1
    session.commit()
    print(f"[import] users.json -> users: {added} new, {len(existing)} already present")
    return added


def import_tokens(session) -> int:
    if not TOKENS_JSON.exists():
        return 0
    try:
        data = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[import] could not read {TOKENS_JSON}: {exc}")
        return 0
    existing = set(session.scalars(select(AgentToken.token)).all())
    # Also skip tokens already owned by a user (they belong on the users table).
    user_tokens = set(t for t in session.scalars(select(User.agent_token)).all() if t)
    added = 0
    for token, rec in (data or {}).items():
        if not token or token in existing or token in user_tokens:
            continue
        session.add(AgentToken(
            token=token,
            name=(rec or {}).get("name", ""),
            last_seen=_dt((rec or {}).get("last_seen")),
        ))
        added += 1
    session.commit()
    print(f"[import] agent_tokens.json -> agent_tokens: {added} new")
    return added


def seed_prompts_step(session) -> dict:
    from scripts.prompt_seed import seed_prompts
    summary = seed_prompts(session, created_by="first_start_import")
    session.commit()
    status = summary.get("status")
    if status == "skipped_unreachable":
        print("[import] prompts NOT seeded (Decoinks configured but unreachable); "
              "will retry on the next start. App serves built-in text meanwhile.")
    else:
        print(f"[import] prompts seeded: status={status} inserted={summary['inserted']} "
              f"skipped={summary['skipped']} by_source={summary['by_source']}")
    return summary


def import_vault_folders(session) -> int:
    """Index existing vault/<client>/<task>/run_N/*.png into vault_saves, so the
    history of past saves is visible. Idempotent on (vault_folder, file_name)."""
    if not VAULT_DIR.is_dir():
        return 0
    existing = {
        (v.vault_folder, v.file_name)
        for v in session.scalars(select(VaultSave)).all()
    }
    added = 0
    for client_dir in VAULT_DIR.iterdir():
        if not client_dir.is_dir():
            continue
        for task_dir in client_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for run_dir in task_dir.iterdir():
                if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                    continue
                try:
                    run_number = int(run_dir.name.split("_", 1)[1])
                except Exception:
                    run_number = None
                for f in run_dir.iterdir():
                    if not f.is_file():
                        continue
                    folder = str(run_dir)
                    if (folder, f.name) in existing:
                        continue
                    session.add(VaultSave(
                        client=client_dir.name, job_number=task_dir.name,
                        run_number=run_number, vault_folder=folder, file_name=f.name,
                    ))
                    added += 1
    session.commit()
    if added:
        print(f"[import] vault folders -> vault_saves: {added} new rows")
    return added


def _archive(path: Path) -> None:
    """Back up an imported JSON to *.imported.<ts> (never delete the data).

    In Docker these files are BIND-MOUNTED single files, so an os.rename of the
    mount point fails with EBUSY/EINVAL. We therefore COPY to the backup name
    (which is itself inside the bind-mounted dir or app dir) and leave the
    original in place — harmless, because the DB is the source of truth now and
    the stale JSON is no longer read. A once-only marker stops us re-copying on
    every boot. DEPLOY.md documents removing the json bind mounts next release.
    """
    if not path.exists():
        return
    marker = path.with_suffix(path.suffix + ".imported")
    if marker.exists():
        return  # already archived once
    dest = path.with_suffix(path.suffix + f".imported.{time.strftime('%Y%m%d_%H%M%S')}")
    try:
        import shutil
        shutil.copy2(path, dest)
        marker.write_text(f"imported {time.strftime('%Y-%m-%d %H:%M:%S')} -> {dest.name}\n",
                          encoding="utf-8")
        print(f"[import] backed up {path} -> {dest} (original left in place; DB is now source of truth)")
    except Exception as exc:
        print(f"[import] could not back up {path}: {exc}")


def main() -> None:
    print("[import] first-start import starting…")
    users_added = tokens_added = 0
    try:
        with get_session() as s:
            users_added = import_users(s)
    except Exception as exc:
        print(f"[import] users step failed: {exc}")
    try:
        with get_session() as s:
            tokens_added = import_tokens(s)
    except Exception as exc:
        print(f"[import] tokens step failed: {exc}")
    try:
        with get_session() as s:
            seed_prompts_step(s)
    except Exception as exc:
        print(f"[import] prompt seed step failed: {exc}")
    try:
        with get_session() as s:
            import_vault_folders(s)
    except Exception as exc:
        print(f"[import] vault step failed: {exc}")

    # Archive the JSON files only after a clean users/tokens import so a rollback
    # (which reads the JSON) still has them this release. They are renamed, not
    # deleted — see DEPLOY.md for when it is safe to remove the bind mounts.
    try:
        _archive(USERS_JSON)
        _archive(TOKENS_JSON)
    except Exception as exc:
        print(f"[import] archive step failed: {exc}")
    print("[import] first-start import done.")


if __name__ == "__main__":
    main()

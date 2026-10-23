"""Prompt seeding + Decoinks-vs-config comparison.

Produces the authoritative set of prompts to put in our DB, keyed by the stable
Decoinks-style key (e.g. "AIS.RATIO.PLAN"). Two sources:

  * CONFIG  — the built-in text in config/workflows.py (always available).
  * DECOINKS — the manager's live/edited text, fetched once if reachable.

Canonical storage uses Decoinks-style {{name}} placeholders. Config constants
are Python {name} templates, so they are converted with
prompt_registry.from_python_template before comparison/seeding.

Precedence when seeding (agreed with the owner):
  * Default: DECOINKS wins (the manager may have edited prompts there).
  * Exception: CONFIG wins for keys in CONFIG_WINS (CUSTOM_ASPECT_ADVICE /
    AIS.RATIO.PLAN, which was just rewritten in config).
  * Only in config  -> seed from config.
  * Only in Decoinks -> seed from Decoinks.

Nothing here writes until seed_prompts() is called; build_report() is read-only
so the per-key report can be shown first.
"""
from __future__ import annotations

import difflib
import json
import os
import time
from pathlib import Path

from src import prompt_registry as reg

# Keys (Decoinks-style) where the CONFIG text must win over Decoinks.
CONFIG_WINS_KEYS = {"AIS.RATIO.PLAN"}  # CUSTOM_ASPECT_ADVICE, just rewritten

BACKUPS_DIR = Path("backups")


# ---------------------------------------------------------------------------
# Config side: every built-in prompt, keyed by AIS.* key, as {{name}} text.
# ---------------------------------------------------------------------------

def _workflow_for_constant(name: str) -> str:
    """Best-effort workflow label for a config constant name (for the UI's
    grouping on the Prompts page). Derived from the known groupings."""
    for wf, names in reg.WORKFLOW_PROMPTS.items():
        if name in names:
            return wf
    if name.startswith("TEXT_"):
        return "text"
    if name.startswith("EXTRACT_") or name.startswith("MOCKUP_"):
        return "mockup"
    if name.startswith("ARTWORK_"):
        return "artwork"
    if name.startswith("IDENTIFY_"):
        return "identify"
    if name.startswith("PRINTREADY_"):
        return "printready"
    if name.startswith("COLORWAY_"):
        return "colorway"
    if name.startswith("GEOMETRY_"):
        return "geometry"
    if name.startswith("CUSTOM_"):
        return "custom"
    if name.startswith("JOB_OPTION_") or name == "BASE_INSTRUCTION":
        return "edit"
    return "other"


def config_prompts() -> dict[str, dict]:
    """Return {AIS_key: {key, name, workflow, text({{}}), placeholders}} for
    every PROMPT_KEYS entry whose built-in text exists in config."""
    out: dict[str, dict] = {}
    for name, ais_key in reg.PROMPT_KEYS.items():
        try:
            builtin = reg._builtin(name)  # Python {name} template text
        except Exception:
            builtin = ""
        if not builtin:
            continue
        canonical = reg.from_python_template(builtin)  # -> {{name}}
        out[ais_key] = {
            "key": ais_key,
            "name": name,
            "workflow": _workflow_for_constant(name),
            "text": canonical,
            "placeholders": sorted(reg.template_fields(builtin)),
        }
    return out


# ---------------------------------------------------------------------------
# Decoinks side: fetch once, save a backup, return {AIS_key: text({{}})}.
# ---------------------------------------------------------------------------

def fetch_decoinks(save_backup: bool = True) -> tuple[dict[str, str], str | None]:
    """Fetch live prompt text from Decoinks for every known key.

    Returns ({AIS_key: text_with_{{}}}, error_or_None). Never raises: on any
    problem (no secret, unreachable, bad response) returns ({}, reason) so the
    caller can fall back to config-only seeding and report it.
    """
    secret = os.getenv("DECOINKS_SERVICE_SECRET", "").strip()
    if not secret:
        return {}, "no DECOINKS_SERVICE_SECRET set (cannot reach Decoinks)"
    try:
        import requests
        keys = sorted(set(reg.PROMPT_KEYS.values()))
        url = f"{reg.BACKEND}/api/ai/prompts"
        resp = requests.get(
            url,
            params={"keys": ",".join(keys), "prefix": reg.DYNAMIC_PREFIX},
            headers={"x-decoinks-sso-secret": secret},
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return {}, f"{exc.__class__.__name__}: {exc}"

    prompts = (payload or {}).get("prompts") or {}
    out: dict[str, str] = {}
    for k, meta in prompts.items():
        text = (meta or {}).get("text") or ""
        if text:
            out[k] = text  # already {{name}} style in Decoinks
    if save_backup:
        try:
            BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            dest = BACKUPS_DIR / f"decoinks_prompts_export_{stamp}.json"
            dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[prompt_seed] saved Decoinks export -> {dest}")
        except Exception as exc:
            print(f"[prompt_seed] could not save Decoinks backup: {exc}")
    return out, None


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def _short_diff(a: str, b: str, context: int = 1) -> str:
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    diff = list(difflib.unified_diff(
        a_lines, b_lines, fromfile="config", tofile="decoinks",
        n=context, lineterm=""))
    # Keep it short for the report.
    return "\n".join(diff[:24])


def build_report() -> dict:
    """Read-only per-key comparison. Returns a dict with a human summary and a
    per-key list: status in {same, differs, only_config, only_decoinks}, which
    source would WIN under the agreed precedence, and a short diff when differ."""
    cfg = config_prompts()
    deco, deco_err = fetch_decoinks(save_backup=True)

    all_keys = sorted(set(cfg) | set(deco))
    rows = []
    counts = {"same": 0, "differs": 0, "only_config": 0, "only_decoinks": 0}
    for key in all_keys:
        c = cfg.get(key, {}).get("text")
        d = deco.get(key)
        if c is not None and d is not None:
            status = "same" if c.strip() == d.strip() else "differs"
        elif c is not None:
            status = "only_config"
        else:
            status = "only_decoinks"
        counts[status] += 1

        # Winner under precedence.
        if key in CONFIG_WINS_KEYS and c is not None:
            winner = "config (forced)"
        elif status == "same":
            winner = "either (identical)"
        elif status == "differs":
            winner = "decoinks"
        elif status == "only_config":
            winner = "config"
        else:
            winner = "decoinks"

        row = {"key": key, "status": status, "winner": winner,
               "config_name": cfg.get(key, {}).get("name", "")}
        if status == "differs":
            row["diff"] = _short_diff(c, d)
        rows.append(row)

    return {
        "decoinks_reachable": deco_err is None,
        "decoinks_error": deco_err,
        "counts": counts,
        "config_keys": len(cfg),
        "decoinks_keys": len(deco),
        "rows": rows,
    }


def print_report(report: dict) -> None:
    print("=" * 72)
    print("PROMPT SEED REPORT — config/workflows.py vs Decoinks")
    print("=" * 72)
    if report["decoinks_reachable"]:
        print(f"Decoinks: REACHABLE ({report['decoinks_keys']} keys returned)")
    else:
        print(f"Decoinks: NOT REACHABLE — {report['decoinks_error']}")
        print("          -> seeding from config only; re-run on the deploy box "
              "with DECOINKS_SERVICE_SECRET set to pull manager edits.")
    print(f"Config built-in keys: {report['config_keys']}")
    c = report["counts"]
    print(f"same={c['same']}  differs={c['differs']}  "
          f"only_config={c['only_config']}  only_decoinks={c['only_decoinks']}")
    print("-" * 72)
    for row in report["rows"]:
        print(f"[{row['status']:>13}] {row['key']:<34} winner={row['winner']}")
        if row.get("diff"):
            for line in row["diff"].splitlines():
                print(f"        {line}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Seeding (idempotent). Called by scripts.first_start_import.
# ---------------------------------------------------------------------------

def resolve_seed_set() -> tuple[dict[str, dict], dict]:
    """Decide the final text per key under the agreed precedence. Returns
    ({AIS_key: {key, workflow, label, text({{}}), placeholders, source}}, report)."""
    cfg = config_prompts()
    deco, deco_err = fetch_decoinks(save_backup=True)
    report = build_report_from(cfg, deco, deco_err)

    final: dict[str, dict] = {}
    for key in sorted(set(cfg) | set(deco)):
        c = cfg.get(key)
        d = deco.get(key)
        if key in CONFIG_WINS_KEYS and c is not None:
            text, source = c["text"], "config(forced)"
        elif d is not None:
            text, source = d, "decoinks"
        elif c is not None:
            text, source = c["text"], "config"
        else:
            continue
        meta = c or {}
        final[key] = {
            "key": key,
            "workflow": meta.get("workflow", ""),
            "label": meta.get("name", key),
            "text": text,
            "placeholders": sorted(reg.template_fields(reg.to_python_template(text))),
            "source": source,
        }
    return final, report


def build_report_from(cfg: dict, deco: dict, deco_err: str | None) -> dict:
    """Same shape as build_report() but from already-fetched data (so we don't
    hit Decoinks twice during a seed)."""
    all_keys = sorted(set(cfg) | set(deco))
    rows = []
    counts = {"same": 0, "differs": 0, "only_config": 0, "only_decoinks": 0}
    for key in all_keys:
        c = cfg.get(key, {}).get("text")
        d = deco.get(key)
        if c is not None and d is not None:
            status = "same" if c.strip() == d.strip() else "differs"
        elif c is not None:
            status = "only_config"
        else:
            status = "only_decoinks"
        counts[status] += 1
        rows.append({"key": key, "status": status})
    return {"decoinks_reachable": deco_err is None, "decoinks_error": deco_err,
            "counts": counts, "config_keys": len(cfg), "decoinks_keys": len(deco),
            "rows": rows}


def decoinks_configured() -> bool:
    """True if a Decoinks secret is set (i.e. Decoinks is the intended source)."""
    return bool(os.getenv("DECOINKS_SERVICE_SECRET", "").strip())


def seed_prompts(session, created_by: str = "system") -> dict:
    """Idempotently seed the prompts + prompt_versions tables.

    SEEDING TRAP GUARD (owner requirement): if Decoinks is CONFIGURED (a secret
    is set) but UNREACHABLE on this run, we must NOT fall back to config —
    otherwise the manager's Decoinks edits would be permanently masked by
    config text once the table is populated (seeding never overwrites). Instead
    we seed NOTHING, log loudly, and the runtime safely falls back to the
    built-in text exactly like before. The next start retries, and once Decoinks
    answers it seeds from there.

    Only when Decoinks is DISABLED (no secret) do we seed from config alone.

    Idempotent rule: a key already present is NEVER overwritten (manager edits
    via the Prompts page survive reboots). Each key records seed_source.

    Returns {inserted, skipped, by_source, status} where status is one of:
      "seeded"            — rows inserted (or nothing missing)
      "skipped_unreachable"— Decoinks configured but down; left empty, retry next start
    """
    from sqlalchemy import select
    from db.models import Prompt, PromptVersion

    cfg = config_prompts()
    deco, deco_err = fetch_decoinks(save_backup=True)

    if decoinks_configured() and deco_err is not None:
        # Configured but unreachable -> refuse to seed from config.
        existing = set(session.scalars(select(Prompt.key)).all())
        if existing:
            # Already seeded on a previous (reachable) run; nothing to do.
            print("[prompt_seed] Decoinks configured but unreachable this run; "
                  "prompts already seeded previously — leaving as-is.")
            return {"inserted": 0, "skipped": len(existing), "by_source": {},
                    "status": "already_seeded"}
        print("=" * 72)
        print("[prompt_seed] ERROR: Decoinks is CONFIGURED but UNREACHABLE and the "
              "prompts table is EMPTY.")
        print(f"[prompt_seed]   reason: {deco_err}")
        print("[prompt_seed]   NOT seeding from config (would mask the manager's "
              "Decoinks edits permanently).")
        print("[prompt_seed]   The app will serve BUILT-IN prompt text until Decoinks "
              "is reachable; seeding retries on the next start.")
        print("=" * 72)
        return {"inserted": 0, "skipped": 0, "by_source": {},
                "status": "skipped_unreachable"}

    if not decoinks_configured():
        print("[prompt_seed] Decoinks is DISABLED (no DECOINKS_SERVICE_SECRET); "
              "seeding prompts from config/workflows.py built-ins.")

    # Build the final set under precedence (Decoinks reachable, or disabled).
    final: dict[str, dict] = {}
    for key in sorted(set(cfg) | set(deco)):
        c = cfg.get(key)
        d = deco.get(key)
        if key in CONFIG_WINS_KEYS and c is not None:
            text, source = c["text"], "config(forced)"
        elif d is not None:
            text, source = d, "decoinks"
        elif c is not None:
            text, source = c["text"], "config"
        else:
            continue
        meta = c or {}
        final[key] = {
            "key": key, "workflow": meta.get("workflow", ""),
            "label": meta.get("name", key), "text": text,
            "placeholders": sorted(reg.template_fields(reg.to_python_template(text))),
            "source": source,
        }

    existing = set(session.scalars(select(Prompt.key)).all())
    inserted, skipped = 0, 0
    by_source: dict[str, int] = {}
    for key, row in final.items():
        if key in existing:
            skipped += 1
            continue
        session.add(Prompt(
            key=key, workflow=row["workflow"], label=row["label"],
            text=row["text"], placeholders={"fields": row["placeholders"]},
            active=True, version=1, seed_source=row["source"], updated_by=created_by,
        ))
        session.add(PromptVersion(key=key, version=1, text=row["text"], created_by=created_by))
        inserted += 1
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1
    return {"inserted": inserted, "skipped": skipped, "by_source": by_source,
            "status": "seeded"}


if __name__ == "__main__":
    # Read-only report for the owner to review before anything is written.
    print_report(build_report())

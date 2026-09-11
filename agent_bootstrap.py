"""Frozen entry point for the packaged agent.

This tiny bootstrap is the ONLY code baked permanently into ArtworkAgent.exe.
Everything that changes day to day — agent.py, agent_gui.py, src/, config/ —
lives in a WRITABLE `code/` folder next to the exe so it can be self-updated
without shipping the 250 MB Chromium runtime again.

Responsibilities (kept deliberately minimal so this file never needs updating):
  1. Seed `<app>/code` from the read-only baseline shipped inside `_internal/`
     on first run (fresh install has code before it ever contacts the server).
  2. Put `<app>/code` FIRST on sys.path so the loose, updatable modules win
     over anything frozen.
  3. Hand off to agent_gui.main().

If anything here fails, we fall back to importing agent_gui directly from the
frozen bundle, so a first launch can never be bricked by a code/ problem.
"""
import os
import shutil
import sys
from pathlib import Path


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _baseline_dir() -> Path | None:
    """Read-only code baseline shipped in the build (under _internal/)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        b = Path(meipass) / "code_baseline"
        if b.is_dir():
            return b
    # Source/dev run: the project root itself is the code.
    return None


def _seed_code_dir() -> Path:
    """Ensure <app>/code exists and holds code. Returns the code dir path."""
    code = _app_dir() / "code"
    baseline = _baseline_dir()
    try:
        code.mkdir(parents=True, exist_ok=True)
        # Seed only when the code dir has no agent_gui yet (fresh install or a
        # wiped code dir). We never overwrite an already-updated code dir.
        if baseline and not (code / "agent_gui.py").is_file():
            for entry in baseline.iterdir():
                dst = code / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(entry, dst)
    except Exception as exc:
        print(f"[bootstrap] could not seed code dir: {exc}")
    return code


def main() -> None:
    try:
        code = _seed_code_dir()
        p = str(code)
        # Loose, updatable code must win over any frozen copy.
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    except Exception as exc:
        print(f"[bootstrap] path setup failed, using frozen code: {exc}")

    # Import the (possibly updated) GUI and run it.
    import agent_gui
    agent_gui.main()


if __name__ == "__main__":
    main()

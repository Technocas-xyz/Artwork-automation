"""Single source of truth for the agent's CODE version.

Bump this whenever the agent's Python code (agent.py, agent_gui.py, src/,
config/) changes and you want deployed agents to self-update. The Chromium
runtime shipped in the one-dir build is NOT versioned here — it never changes,
so only the ~1 MB of code travels on each update.

Format: a dotted string compared with packaging-style ordering when available,
falling back to a plain tuple-of-ints comparison.
"""
from __future__ import annotations

AGENT_CODE_VERSION = "1.4.1"


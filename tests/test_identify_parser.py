"""Unit tests for the Artwork Identification reply parser.

The list-objects turn asks ChatGPT for JSON, but ChatGPT frequently replies with
a plain numbered/bulleted list (or an older prompt is still served from the DB).
The parser must handle both, in order: JSON first, then a tolerant list parser.
These tests lock that behaviour so "No items were identified" cannot silently
return when ChatGPT clearly did answer.

Run from the project root:  python tests/test_identify_parser.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# agent.py imports several heavy third parties at module load. Stub the ones that
# are not needed to exercise the pure parser functions, so the test runs without
# a full agent environment. (If a real one is installed, the real module wins.)
for _name in ("requests", "dotenv", "playwright", "playwright.sync_api"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            mod = types.ModuleType(_name)
            if _name == "dotenv":
                mod.load_dotenv = lambda *a, **k: None
            if _name == "playwright.sync_api":
                mod.Page = object
                mod.TimeoutError = type("TimeoutError", (Exception,), {})
            sys.modules[_name] = mod

import agent  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, extra: object = "") -> None:
    print(("PASS " if cond else "FAIL ") + label + ("" if cond else f"  [{extra}]"))
    if not cond:
        FAILS.append(label)


def parse_reply(reply: str):
    """Mirror the agent's parse order: JSON first, then the tolerant list.

    Returns (names, source) where source is 'json' | 'list' | 'none'."""
    parsed = agent._parse_identify_json(reply)
    if parsed:
        return [o["name"] for o in parsed], "json"
    names = agent._parse_numbered_list(reply)
    if names:
        return names, "list"
    return [], "none"


# 1) A clean JSON array (the prompt's intended format).
json_reply = (
    '[{"id": 1, "name": "PRINT ON DEMAND text", "cells": ["C3", "D3"]}, '
    '{"id": 2, "name": "Person holding tablet", "cells": ["E5"]}, '
    '{"id": 3, "name": "Large printer machine", "cells": ["G7"]}]'
)
names, src = parse_reply(json_reply)
check("JSON array parses via json", src == "json", src)
check("JSON array -> 3 names", names == ["PRINT ON DEMAND text", "Person holding tablet", "Large printer machine"], names)

# 2) The same JSON wrapped in ```json code fences + prose.
fenced_reply = (
    "Sure, here you go:\n```json\n"
    '[{"id": 1, "name": "Lion", "cells": ["D4"]}, {"id": 2, "name": "Sky", "cells": ["A1"]}]'
    "\n```\nLet me know!"
)
names, src = parse_reply(fenced_reply)
check("fenced JSON parses via json", src == "json", src)
check("fenced JSON -> [Lion, Sky]", names == ["Lion", "Sky"], names)

# 3) The exact plain list from the bug report (no JSON at all).
plain_reply = '1. "PRINT ON DEMAND" text\n2. Person holding tablet\n3. Large printer machine'
names, src = parse_reply(plain_reply)
check("plain list parses via list fallback", src == "list", src)
check("plain list -> 3 names (inner quotes kept)",
      names == ['"PRINT ON DEMAND" text', "Person holding tablet", "Large printer machine"], names)

# 4) Extra tolerance: bullets + markdown bold still parse (regression guard).
check("bulleted + bold list parses",
      agent._parse_numbered_list("- **Logo**\n* Tagline\n\u2022 Red star")
      == ["Logo", "Tagline", "Red star"],
      agent._parse_numbered_list("- **Logo**\n* Tagline\n\u2022 Red star"))

# 5) Prose-only / empty -> nothing (caller then shows the raw reply, not silent).
check("prose-only -> none", parse_reply("I could not find any objects.") == ([], "none"))
check("empty -> none", parse_reply("") == ([], "none"))

print("\nRESULT: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

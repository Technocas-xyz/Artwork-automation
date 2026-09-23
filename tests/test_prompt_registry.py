"""Checks for the Decoinks Prompt Management wiring. No network, no browser.

Run from the project root:  python tests/test_prompt_registry.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import requests  # noqa: F401
except ImportError:  # the registry only needs it for real fetches
    sys.modules["requests"] = types.ModuleType("requests")

from config import workflows as W  # noqa: E402
from config.job_options import JOB_OPTIONS, PARAMETERISED_OPTIONS  # noqa: E402
from src import prompt_builder  # noqa: E402
from src.prompt_registry import (  # noqa: E402
    PROMPT_KEYS, PromptRegistry, _builtin, from_python_template, option_template_name,
    prompt_names_for_job, template_fields, to_python_template,
)

FAILS: list[str] = []


def check(label: str, cond: bool, extra: object = "") -> None:
    print(("PASS " if cond else "FAIL ") + label + ("" if cond else f"  [{extra}]"))
    if not cond:
        FAILS.append(label)


def fake_registry(texts: dict[str, str]) -> PromptRegistry:
    """A registry whose "Decoinks" answer is `texts` (key -> {{}} text), never refetched."""
    reg = PromptRegistry()
    reg._data = {"revision": "t", "prompts": {
        k: {"prompt_key": k, "text": t, "version": {"id": f"v-{i}", "number": 1}}
        for i, (k, t) in enumerate(texts.items())}}
    reg._fetched_at = float("inf")
    return reg


# 1. Every prompt text in the code is mapped to a key in the library.
code_prompts = {n for n, v in vars(W).items() if n.isupper() and isinstance(v, str) and len(v) > 60}
code_prompts |= {"BASE_INSTRUCTION"} | {option_template_name(k) for k in JOB_OPTIONS}
check("every prompt in the code has a Prompt Management key", code_prompts <= set(PROMPT_KEYS),
      sorted(code_prompts - set(PROMPT_KEYS)))
check("31 prompts mapped, keys unique", len(PROMPT_KEYS) == 31 and len(set(PROMPT_KEYS.values())) == 31, len(PROMPT_KEYS))

# 2. Conversion to Decoinks {{}} text and back is exact for every one of them.
bad = [n for n in PROMPT_KEYS if to_python_template(from_python_template(_builtin(n))) != _builtin(n)]
check("round trip {x} <-> {{x}} is byte-identical for all 31", not bad, bad)

# 3. Published text identical to the code -> the automation runs the code's text, from Decoinks.
same = {PROMPT_KEYS[n]: from_python_template(_builtin(n)) for n in PROMPT_KEYS}
reg = fake_registry(same)
served = {n: reg.template(n) for n in PROMPT_KEYS}
check("all 31 served as managed and identical to the built-in text",
      all(t == _builtin(n) and m["source"] == "managed" for n, (t, m) in served.items()))

# 4. A version that adds or drops a placeholder is refused.
changed = dict(same)
changed["AIS.EDIT.RECOLOUR"] = "Recolour it"            # drops {{value}}
changed["AIS.EXTRACT.BOXES"] = same["AIS.EXTRACT.BOXES"] + " {{dpi}}"  # adds one
reg2 = fake_registry(changed)
t, m = reg2.template("JOB_OPTION_RECOLOUR")
check("dropping {{value}} is refused", m["source"] == "built_in" and t == JOB_OPTIONS["recolour"], m)
t, m = reg2.template("EXTRACT_BOXES")
check("adding {{dpi}} is refused", m["source"] == "built_in" and t == W.EXTRACT_BOXES, m)

# 4b. A prompt retired (Archived) in Decoinks is not served: the built-in text runs.
retired = {k: t for k, t in same.items() if k not in ("AIS.EXTRACT.BOXES", "AIS.PRINTREADY.BLACK_OUT")}
reg3 = fake_registry(retired)
t, m = reg3.template("CUSTOM_BLACK_OUT")
check("a retired prompt falls back to the built-in text", m["source"] == "built_in" and t == W.CUSTOM_BLACK_OUT, m)
t, m = reg3.template("TEXT_TURN_1")
check("the others are still served from Prompt Management", m["source"] == "managed" and t == W.TEXT_TURN_1, m)

# 5. Which prompts each job uses.
check("text job without a mode uses every text prompt", prompt_names_for_job("text") ==
      ["TEXT_TURN_0", "TEXT_TURN_1", "TEXT_TURN_2", "TEXT_TURN_3", "TEXT_REPLACE_COLLAGE", "TEXT_REPLACE_FINAL",
       "TEXT_IMAGE_ELEMENT_COLLAGE", "TEXT_IMAGE_STYLE_COLLAGE"])
check("text job, typed", prompt_names_for_job("text", text_mode="typed") == ["TEXT_TURN_1", "TEXT_TURN_2", "TEXT_TURN_3"])
check("text job, from image", prompt_names_for_job("text", text_mode="from_image") == ["TEXT_TURN_0", "TEXT_TURN_1", "TEXT_TURN_2", "TEXT_TURN_3"])
check("text job, replace", prompt_names_for_job("text", text_mode="replace") == ["TEXT_REPLACE_COLLAGE", "TEXT_REPLACE_FINAL"])
check("text job, image element", prompt_names_for_job("text", text_mode="image_element") == ["TEXT_IMAGE_ELEMENT_COLLAGE", "TEXT_TURN_2", "TEXT_TURN_3"])
check("text job, image style", prompt_names_for_job("text", text_mode="image_style") == ["TEXT_IMAGE_STYLE_COLLAGE", "TEXT_TURN_2", "TEXT_TURN_3"])
check("custom job", prompt_names_for_job("custom", ["reconstruct", "aspect_ratio"]) ==
      ["CUSTOM_RECONSTRUCT", "CUSTOM_ASPECT_ADVICE", "CUSTOM_ASPECT_BASELINE", "CUSTOM_ASPECT_REGENERATE"])
check("edit-options job", prompt_names_for_job("", options=["recolour", "text_only"]) ==
      ["BASE_INSTRUCTION", "JOB_OPTION_RECOLOUR", "JOB_OPTION_TEXT_ONLY"])

# 6. The edit-options prompt is built from the managed blocks.
opts, params = ["remove_background", "recolour"], {"recolour": "navy"}
check("build_prompt with no managed text is unchanged",
      prompt_builder.build_prompt(opts, params, "note") == prompt_builder.build_prompt(opts, params, "note", templates={}))
managed = {"BASE_INSTRUCTION": "BASE v2", "JOB_OPTION_RECOLOUR": "Recolour to {value} v2"}
built = prompt_builder.build_prompt(opts, params, "note", templates=managed)
check("build_prompt uses the managed base and option blocks",
      built == "BASE v2\n\n" + JOB_OPTIONS["remove_background"] + "\n\nRecolour to navy v2\n\nnote", built)
check("parameterised options still need {value}",
      all("{value}" in JOB_OPTIONS[k] and template_fields(JOB_OPTIONS[k]) == {"value"} for k in PARAMETERISED_OPTIONS))

# 7. A prompt published in the Custom Operations module becomes an operation, no code change.
from src.prompt_registry import dynamic_name, key_for  # noqa: E402
reg4 = fake_registry({**same,
                      "AIS.CUSTOM_OPERATIONS.SEPIA_TONE": "Make it sepia. Keep {braces} literal.",
                      "AIS.CUSTOM_OPERATIONS.BAD_ONE": "Uses {{colour}}, which nothing fills",
                      "AIS.CUSTOM_OPERATIONS.bad-key": "Unsafe key"})
reg4._data["prompts"]["AIS.CUSTOM_OPERATIONS.SEPIA_TONE"]["name"] = "Sepia Tone"
ops = reg4.dynamic_operations()
check("a published Custom Operations prompt is listed as an operation",
      [o["key"] for o in ops] == ["pm_sepia_tone"] and ops[0]["label"] == "Sepia Tone", ops)
check("its text is sent exactly as written", ops[0]["template"] == "Make it sepia. Keep {braces} literal.")
check("one with {{placeholders}} is not offered", all(o["key"] != "pm_bad_one" for o in ops))
check("job names map back to the key", key_for(dynamic_name("pm_sepia_tone")) == "AIS.CUSTOM_OPERATIONS.SEPIA_TONE")
check("custom job with a Prompt Management op",
      prompt_names_for_job("custom", ["pm_sepia_tone", "aspect_ratio"]) ==
      ["PM_OP:AIS.CUSTOM_OPERATIONS.SEPIA_TONE", "CUSTOM_ASPECT_ADVICE", "CUSTOM_ASPECT_BASELINE", "CUSTOM_ASPECT_REGENERATE"])
check("unedited op text is not counted as an edit",
      reg4.is_unedited(dynamic_name("pm_sepia_tone"), "Make it sepia. Keep {braces} literal."))

# 8. The manifest Decoinks checks publishes against.
man = {r["prompt_key"]: r for r in reg4.manifest()}
check("manifest lists every mapped key plus live ops and the pattern",
      set(PROMPT_KEYS.values()) | {"AIS.CUSTOM_OPERATIONS.SEPIA_TONE", "AIS.CUSTOM_OPERATIONS.*"} == set(man), sorted(man))
check("manifest placeholders are what the code fills",
      man["AIS.EDIT.RECOLOUR"]["placeholders"] == ["value"] and man["AIS.CUSTOM_OPERATIONS.SEPIA_TONE"]["placeholders"] == [])
check("manifest says what is running", man["AIS.TEXT.COLLAGE"]["running"]["source"] == "managed")

print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)

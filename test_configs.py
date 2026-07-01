#!/usr/bin/env python3
"""
Validates the generic per-model configs in configs/*.md.

omodel-manager owns these files but doesn't render them; this test is the
"store + validate" guarantee: every config parses and has the required shape,
so downstream adapters (omodel-wire, …) can trust them.

Run:  python3 -m unittest test_configs   (or the whole suite: python3 -m unittest)
"""

import json
import pathlib
import re
import unittest

CONFIGS = pathlib.Path(__file__).resolve().parent / "configs"
JSON_BLOCK = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)

REQUIRED_PRESETS = {"reason", "code", "agent", "instruct"}
KNOWN_THINKING_CONTROL = {"enable_thinking", "reasoning_effort", "soft_switch", "none"}
KNOWN_SAMPLING = {"temperature", "top_p", "top_k", "min_p", "presence_penalty",
                  "frequency_penalty", "repetition_penalty"}


def _config_files():
    if not CONFIGS.is_dir():
        return []
    return [p for p in sorted(CONFIGS.glob("*.md")) if p.name.lower() != "readme.md"]


def _load(p):
    m = JSON_BLOCK.search(p.read_text(encoding="utf-8"))
    assert m, f"{p.name}: no ```json block"
    return json.loads(m.group(1))


class ConfigValidityTests(unittest.TestCase):
    def test_at_least_one_config(self):
        self.assertTrue(_config_files(), "no configs/*.md found")

    def test_each_config_valid(self):
        for p in _config_files():
            with self.subTest(config=p.name):
                r = _load(p)  # must be valid JSON

                match = r.get("match")
                self.assertTrue(match, "match required")
                match = match if isinstance(match, list) else [match]

                # filename stem must be one of the match keys (config keyed to its name)
                stem = p.stem
                self.assertIn(stem, match,
                              f"filename '{stem}' should appear in match {match}")

                caps = r.get("capabilities", {})
                self.assertIn("reasoning", caps)
                self.assertIn("tool_call", caps)
                tc = caps.get("thinking_control", r.get("thinking_control"))
                if tc is not None:
                    self.assertIn(tc, KNOWN_THINKING_CONTROL)

                presets = r.get("presets", {})
                self.assertTrue(REQUIRED_PRESETS.issubset(presets),
                                f"missing presets: {REQUIRED_PRESETS - set(presets)}")
                for pk, preset in presets.items():
                    self.assertIn("thinking", preset, f"{pk}: needs thinking flag")
                    s = preset.get("sampling", {})
                    self.assertIsInstance(s, dict, f"{pk}: sampling must be an object")
                    unknown = set(s) - KNOWN_SAMPLING
                    self.assertFalse(unknown, f"{pk}: unknown sampling keys {unknown}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

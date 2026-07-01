#!/usr/bin/env python3
"""
Validates the generic per-model configs in configs/*.toml.

omodel-manager owns these files but doesn't render them; this test is the
"store + validate" guarantee: every config parses and has the required shape,
so downstream adapters (omodel-wire, …) can trust them.

Requires Python 3.11+ (stdlib tomllib). Run: python3 -m unittest test_configs
"""

import pathlib
import tomllib
import unittest

CONFIGS = pathlib.Path(__file__).resolve().parent / "configs"

REQUIRED_PRESETS = {"reason", "code", "agent", "instruct"}
KNOWN_THINKING_CONTROL = {"enable_thinking", "reasoning_effort", "soft_switch", "none"}
KNOWN_SAMPLING = {"temperature", "top_p", "top_k", "min_p", "presence_penalty",
                  "frequency_penalty", "repetition_penalty"}


def _config_files():
    return sorted(CONFIGS.glob("*.toml")) if CONFIGS.is_dir() else []


def _load(p):
    with open(p, "rb") as f:
        return tomllib.load(f)


class ConfigValidityTests(unittest.TestCase):
    def test_at_least_one_config(self):
        self.assertTrue(_config_files(), "no configs/*.toml found")

    def test_each_config_valid(self):
        for p in _config_files():
            with self.subTest(config=p.name):
                r = _load(p)  # valid TOML

                match = r.get("match")
                self.assertTrue(match, "match required")
                match = match if isinstance(match, list) else [match]
                self.assertIn(p.stem, match,
                              f"filename '{p.stem}' should appear in match {match}")

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
                    self.assertIsInstance(s, dict, f"{pk}: sampling must be a table")
                    unknown = set(s) - KNOWN_SAMPLING
                    self.assertFalse(unknown, f"{pk}: unknown sampling keys {unknown}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

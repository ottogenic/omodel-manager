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

REQUIRED_PRESETS = {"plan", "build"}
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
                self.assertEqual(set(presets), REQUIRED_PRESETS,
                                 f"presets must be exactly {sorted(REQUIRED_PRESETS)}")
                for pk, preset in presets.items():
                    self.assertIn("thinking", preset, f"{pk}: needs thinking flag")
                    s = preset.get("sampling", {})
                    self.assertIsInstance(s, dict, f"{pk}: sampling must be a table")
                    unknown = set(s) - KNOWN_SAMPLING
                    self.assertFalse(unknown, f"{pk}: unknown sampling keys {unknown}")

    def test_match_patterns_dont_cross_match(self):
        # A config's match patterns must uniquely identify ITS model. Downstream
        # (omw) does LIVE detection by SUBSTRING (pattern in served_id), so if one
        # config's pattern is a substring of another's, the wrong config lights up
        # when a SIBLING quant is running -- e.g. an nvfp4 config showing LIVE
        # because the fp8 sibling's served id contains a bare "Qwen3.6-35B". Guard it.
        configs = {}
        for p in _config_files():
            m = _load(p).get("match") or []
            configs[p.stem] = [str(x).lower() for x in (m if isinstance(m, list) else [m])]
        for a, apats in configs.items():
            for b, bpats in configs.items():
                if a == b:
                    continue
                for ap in apats:
                    for bp in bpats:
                        self.assertNotIn(ap, bp,
                            f"'{a}' pattern '{ap}' is a substring of '{b}' pattern "
                            f"'{bp}' -> they'll cross-match in omw's LIVE detection")

    def test_laguna_rc2_configs_use_checkpoint_sampling_and_validated_limits(self):
        expected = {
            "laguna-s-2.1-nvfp4": 229376,
            "laguna-dflash-s-2.1-nvfp4": 131072,
        }
        for name, context in expected.items():
            with self.subTest(config=name):
                cfg = _load(CONFIGS / f"{name}.toml")
                self.assertEqual(cfg["context"]["native"], context)
                self.assertEqual(cfg["capabilities"]["concurrency"], 1)
                for preset in REQUIRED_PRESETS:
                    sampling = cfg["presets"][preset]["sampling"]
                    self.assertEqual(sampling,
                                     {"temperature": 1.0, "top_p": 1.0, "top_k": 20})

    def test_deepseek_plan_build_modes_and_variants(self):
        cfg = _load(CONFIGS / "deepseek-v4-flash-0731.toml")
        self.assertEqual(cfg["capabilities"]["concurrency"], 12)
        self.assertEqual(cfg["capabilities"]["thinking_control"], "none")
        self.assertEqual(cfg["context"], {"native": 1048576, "min_thinking": 393216})
        self.assertEqual(cfg["presets"]["plan"]["options"]["chat_template_kwargs"],
                         {"thinking": True, "reasoning_effort": "max"})
        self.assertEqual(cfg["presets"]["build"]["options"]["chat_template_kwargs"],
                         {"thinking": True, "reasoning_effort": "high"})
        self.assertEqual(set(cfg["variants"]), {"low", "high", "max", "no-think"})
        self.assertEqual(cfg["variants"]["no-think"]["options"]["chat_template_kwargs"],
                         {"thinking": False})

    def test_qwen38_bf16_config_matches_multimodal_profiles(self):
        cfg = _load(CONFIGS / "qwen3.8-27b-bf16.toml")
        self.assertEqual(set(cfg["match"]), {
            "qwen3.8-27b-bf16", "qwen3.8-27b-bf16-mtp",
        })
        self.assertEqual(cfg["capabilities"]["vision"], {
            "input": ["text", "image", "video"], "output": ["text"],
        })
        self.assertEqual(cfg["capabilities"]["concurrency"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

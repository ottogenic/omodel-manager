#!/usr/bin/env python3
"""
Fast, offline test suite for omodel-manager.

Run:  python3 -m unittest        (or: python3 test_omodel_manager.py)

These tests DO NOT launch containers, hit the network, or touch $HOME. They
exercise the pure builders (config merge / extends / argv / masking / paths) and
mock the one Docker choke point (`docker()`) where needed. Keep them fast.
"""

import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from types import SimpleNamespace

# The tool is a hyphenated, extension-less executable ("omodel-manager"), so it
# can't be `import`ed by name -- load it from the sibling file with an explicit
# source loader (spec_from_file_location auto-detects by ".py", which we lack).
_path = pathlib.Path(__file__).resolve().parent / "omodel-manager"
_loader = SourceFileLoader("omodel_manager", str(_path))
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
mm = importlib.util.module_from_spec(_spec)
_loader.exec_module(mm)


def fake_docker(stdout="", returncode=0):
    """A stand-in for mm.docker() returning a CompletedProcess-like object."""
    def _run(args, capture=False, check=False, tty=False):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    return _run


class ConfigSeedTests(unittest.TestCase):
    def test_seed_equals_committed_file(self):
        """DEFAULT_CONFIG must match the on-disk model_manager.json (seed == file)."""
        on_disk = mm.load_config()
        seed = json.loads(json.dumps(mm.DEFAULT_CONFIG))  # normalize None->null etc.
        self.assertEqual(seed, on_disk,
                         "DEFAULT_CONFIG and model_manager.json diverged -- "
                         "regenerate with `config --init --force` or update both.")

    def test_defaults_force_color(self):
        env = mm.DEFAULT_CONFIG["defaults"]["env"]
        self.assertEqual(env.get("VLLM_LOGGING_COLOR"), "1")

    def test_no_secret_literals_in_config(self):
        blob = json.dumps(mm.DEFAULT_CONFIG)
        self.assertNotIn("hf_", blob.replace("hf_cache", ""))  # no hf_ token strings
        self.assertIsNone(mm.DEFAULT_CONFIG["defaults"]["env"]["HF_TOKEN"])


class ProfileTests(unittest.TestCase):
    """Every profile must merge and build a sane docker argv."""

    def setUp(self):
        self.cfg = mm.load_config()
        self.keys = list(self.cfg["models"])

    def test_every_profile_builds(self):
        self.assertTrue(self.keys)
        for key in self.keys:
            with self.subTest(profile=key):
                merged = mm.merge_model(self.cfg, key)
                self.assertTrue(merged["model"], "model required")
                self.assertTrue(merged["port"], "port required")
                name, argv, warnings = mm.build_run_argv(key, merged, target=None)
                self.assertEqual(argv[0], "run")
                self.assertIn(merged["image"], argv)
                self.assertIn("--model", argv)
                self.assertIn("--port", argv)
                self.assertNotIn(None, argv)
                # format_run must not raise for any profile
                self.assertIsInstance(mm.format_run(argv), str)

    def test_container_names_unique(self):
        names = [mm.container_name(k) for k in self.keys]
        self.assertEqual(len(names), len(set(names)))


class ExtendsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = mm.load_config()

    def test_deep_merge(self):
        base = {"a": 1, "b": {"x": 1, "y": 2}, "lst": [1]}
        over = {"b": {"y": 3, "z": 4}, "c": 5, "lst": [9]}
        out = mm._deep_merge(base, over)
        self.assertEqual(out, {"a": 1, "b": {"x": 1, "y": 3, "z": 4}, "c": 5, "lst": [9]})
        self.assertEqual(base["b"], {"x": 1, "y": 2})  # base not mutated

    def test_nemotron_1m_extends_256k(self):
        m = mm.merge_model(self.cfg, "nemotron-3-super-120b-nvfp4-1m")
        base = mm.merge_model(self.cfg, "nemotron-3-super-120b-nvfp4-256k")
        # inherited from base
        self.assertEqual(m["model"], base["model"])
        self.assertEqual(m["image"], base["image"])
        self.assertEqual([a["url"] for a in m["assets"]],
                         [a["url"] for a in base["assets"]])
        # overridden
        self.assertEqual(m["vllm_args"]["max-model-len"], 1048576)
        self.assertEqual(m["vllm_args"]["max-num-seqs"], 2)
        self.assertEqual(base["vllm_args"]["max-model-len"], 262144)


class BuildArgvTests(unittest.TestCase):
    def setUp(self):
        self.cfg = mm.load_config()

    def test_bool_and_none_flags(self):
        m = mm.merge_model(self.cfg, "glm-4.7-flash")
        m["vllm_args"]["a-bare-flag"] = True
        m["vllm_args"]["a-off-flag"] = False
        m["vllm_args"]["a-null-flag"] = None
        _, argv, _ = mm.build_run_argv("glm-4.7-flash", m, target=None)
        self.assertIn("--a-bare-flag", argv)
        self.assertNotIn("--a-off-flag", argv)
        self.assertNotIn("--a-null-flag", argv)

    def test_json_arg_is_single_token(self):
        m = mm.merge_model(self.cfg, "qwen3.6-27b-nvfp4-256k")
        _, argv, _ = mm.build_run_argv("qwen3.6-27b-nvfp4-256k", m, target=None)
        i = argv.index("--speculative-config")
        val = argv[i + 1]
        self.assertEqual(json.loads(val),
                         {"method": "mtp", "num_speculative_tokens": 3})

    def test_hf_overrides_intact(self):
        m = mm.merge_model(self.cfg, "qwen3.6-27b-nvfp4-512k")
        _, argv, _ = mm.build_run_argv("qwen3.6-27b-nvfp4-512k", m, target=None)
        i = argv.index("--hf-overrides")
        j = json.loads(argv[i + 1])
        self.assertEqual(j["text_config"]["rope_parameters"]["rope_type"], "yarn")


class TokenTests(unittest.TestCase):
    """HF_TOKEN sourcing/forwarding, with env + token store mocked (no $HOME writes)."""

    def setUp(self):
        self.cfg = mm.load_config()
        self._env = os.environ.get("HF_TOKEN")
        self._tokfile = mm.HF_TOKEN_FILE
        os.environ.pop("HF_TOKEN", None)
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        os.remove(self._tmp.name)  # start absent
        mm.HF_TOKEN_FILE = self._tmp.name

    def tearDown(self):
        mm.HF_TOKEN_FILE = self._tokfile
        if self._env is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = self._env
        if os.path.exists(self._tmp.name):
            os.remove(self._tmp.name)

    def _argv(self, target):
        m = mm.merge_model(self.cfg, "glm-4.7-flash")
        _, argv, _ = mm.build_run_argv("glm-4.7-flash", m, target=target)
        return argv

    def test_local_inherits_bare(self):
        os.environ["HF_TOKEN"] = "hf_LOCAL"
        argv = self._argv(target=None)
        i = argv.index("HF_TOKEN")
        self.assertEqual(argv[i - 1], "-e")  # bare `-e HF_TOKEN`, value not in argv

    def test_remote_forwards_value(self):
        os.environ["HF_TOKEN"] = "hf_REMOTE"
        argv = self._argv(target="user@host")
        self.assertIn("HF_TOKEN=hf_REMOTE", argv)

    def test_store_used_when_env_absent(self):
        with open(self._tmp.name, "w") as f:
            f.write("hf_STORED\n")
        argv = self._argv(target=None)
        self.assertIn("HF_TOKEN=hf_STORED", argv)

    def test_format_run_masks_token(self):
        os.environ["HF_TOKEN"] = "hf_SUPERSECRETVALUE"
        argv = self._argv(target="user@host")
        out = mm.format_run(argv, target="user@host")
        self.assertNotIn("hf_SUPERSECRETVALUE", out)
        self.assertIn("(masked)", out)


class PathTests(unittest.TestCase):
    def tearDown(self):
        mm._remote_home_cache.clear()

    def test_local_expands(self):
        # separator-agnostic (Windows may mix \ and /; on WSL/Linux it's all /)
        p = mm.host_path("~/x")
        self.assertTrue(p.startswith(os.path.expanduser("~")))
        self.assertTrue(p.endswith("x"))
        self.assertNotIn("~", p)
        self.assertTrue(mm.host_path("${PWD}/y").endswith("y"))
        self.assertNotIn("${PWD}", mm.host_path("${PWD}/y"))

    def test_remote_uses_remote_home(self):
        mm._remote_home_cache["u@h"] = "/home/otto"
        self.assertEqual(mm.host_path("~/x", target="u@h"), "/home/otto/x")
        self.assertEqual(mm.host_path("~/.cache/hf", target="u@h"), "/home/otto/.cache/hf")


class SshOptsTests(unittest.TestCase):
    def setUp(self):
        self._key = mm.OTOOLS_SSH_KEY

    def tearDown(self):
        mm.OTOOLS_SSH_KEY = self._key

    def test_no_key_no_identity(self):
        mm.OTOOLS_SSH_KEY = os.path.join(tempfile.gettempdir(), "definitely-missing-key")
        opts = mm.ssh_opts()
        self.assertNotIn("-i", opts)

    def test_key_present_pins_identity(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(b"x")
        tmp.close()
        try:
            mm.OTOOLS_SSH_KEY = tmp.name
            opts = mm.ssh_opts()
            self.assertIn("-i", opts)
            self.assertIn("IdentitiesOnly=yes", opts)
        finally:
            os.remove(tmp.name)


class ResolveTargetTests(unittest.TestCase):
    def setUp(self):
        self._docker = mm.docker

    def tearDown(self):
        mm.docker = self._docker

    def test_key_maps_to_container_name(self):
        name = mm.container_name("glm-4.7-flash")  # otools-vllm-glm-4.7-flash
        mm.docker = fake_docker(stdout=f"other\n{name}\nmore\n")
        self.assertEqual(mm.resolve_target("glm-4.7-flash"), name)

    def test_raw_name_passthrough(self):
        mm.docker = fake_docker(stdout="otools-vllm-glm-4.7-flash\n")
        self.assertEqual(mm.resolve_target("otools-vllm-glm-4.7-flash"),
                         "otools-vllm-glm-4.7-flash")

    def test_unknown_returns_input(self):
        mm.docker = fake_docker(stdout="something-else\n")
        self.assertEqual(mm.resolve_target("ghost"), "ghost")


class FmtTokensTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(mm._fmt_tokens(262144), "256K")
        self.assertEqual(mm._fmt_tokens(1048576), "1M")
        self.assertEqual(mm._fmt_tokens(202752), "198K")
        self.assertEqual(mm._fmt_tokens(524288), "512K")
        self.assertEqual(mm._fmt_tokens(None), "?")
        self.assertEqual(mm._fmt_tokens(1000), "1000")


if __name__ == "__main__":
    unittest.main(verbosity=2)

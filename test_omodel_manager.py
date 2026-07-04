#!/usr/bin/env python3
"""
Fast, offline test suite for omodel-manager.

Run:  python3 -m unittest        (or: python3 test_omodel_manager.py)

These tests DO NOT launch containers, hit the network, or touch $HOME. They
exercise the pure builders (config merge / extends / argv / masking / paths) and
mock the one Docker choke point (`docker()`) where needed. Keep them fast.
"""

import contextlib
import importlib.util
import io
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
    def test_defaults_force_color(self):
        env = mm.DEFAULT_CONFIG["defaults"]["env"]
        self.assertEqual(env.get("VLLM_LOGGING_COLOR"), "1")

    def test_no_secret_literals_in_config(self):
        blob = json.dumps(mm.DEFAULT_CONFIG)
        self.assertNotIn("hf_", blob.replace("hf_cache", ""))  # no hf_ token strings
        self.assertIsNone(mm.DEFAULT_CONFIG["defaults"]["env"]["HF_TOKEN"])

    def test_defaults_have_no_remotes(self):
        # host lists are operator/machine-specific -> ~/.config/otools/hosts, not config
        self.assertNotIn("remotes", mm.DEFAULT_CONFIG["defaults"])


class HostsStoreTests(unittest.TestCase):
    """~/.config/otools/hosts -- alias-aware; managed by `install`, read by `ps`."""

    def _tmp_hosts(self):
        return os.path.join(tempfile.mkdtemp(), "hosts")

    def setUp(self):
        self._old, mm.HOSTS_FILE = mm.HOSTS_FILE, self._tmp_hosts()

    def tearDown(self):
        mm.HOSTS_FILE = self._old

    def test_bare_roundtrip_and_dedup(self):
        # Bare user@host entries store as (target, target) pairs; dupes dropped, order kept.
        mm.save_hosts(["otto@a", "otto@b", "otto@a"])
        self.assertEqual(mm.load_hosts(), [("otto@a", "otto@a"), ("otto@b", "otto@b")])
        self.assertEqual(mm.host_targets(), ["otto@a", "otto@b"])

    def test_alias_roundtrip(self):
        mm.save_hosts([("dgx1", "otto@a"), ("dgx2", "otto@b")])
        self.assertEqual(mm.load_hosts(), [("dgx1", "otto@a"), ("dgx2", "otto@b")])

    def test_dedup_by_target_keeps_first_alias(self):
        mm.save_hosts([("dgx1", "otto@a"), ("other", "otto@a")])  # same target twice
        self.assertEqual(mm.load_hosts(), [("dgx1", "otto@a")])

    def test_resolve_host_alias_and_passthrough(self):
        mm.save_hosts([("dgx1", "otto@a")])
        self.assertEqual(mm.resolve_host("dgx1"), "otto@a")     # alias -> target
        self.assertEqual(mm.resolve_host("otto@b"), "otto@b")   # unknown/raw passes through
        self.assertEqual(mm.resolve_host(""), "")               # empty passes through
        self.assertIsNone(mm.resolve_host(None))

    def test_missing_file_is_empty(self):
        self.assertEqual(mm.load_hosts(), [])
        self.assertEqual(mm.host_targets(), [])

    def test_host_label_prefers_alias(self):
        # suggested commands should echo the alias, not the raw user@ip
        mm.save_hosts([("dgx1", "otto@192.168.50.101")])
        self.assertEqual(mm._host_label("otto@192.168.50.101"), "dgx1")
        self.assertEqual(mm._host_label("otto@unknown"), "otto@unknown")  # no alias -> raw
        self.assertIsNone(mm._host_label(None))


class NonBlockingLaunchTests(unittest.TestCase):
    """Background pull+run helpers and the pull-status log classifier."""

    def setUp(self):
        self._docker = mm.docker

    def tearDown(self):
        mm.docker = self._docker

    def test_image_present_true_false(self):
        mm.docker = fake_docker(returncode=0)
        self.assertTrue(mm._image_present("vllm/x:tag"))
        mm.docker = fake_docker(returncode=1)
        self.assertFalse(mm._image_present("vllm/x:tag"))

    def test_bg_command_shape(self):
        argv = ["run", "-d", "--rm", "--name", "otools-vllm-k", "img:tag", "--model", "m"]
        cmd = mm._launch_bg_command("k", "img:tag", argv)
        self.assertIn("docker pull", cmd)
        self.assertIn("docker run", cmd)
        self.assertIn("OTOOLS_LAUNCH_OK", cmd)
        self.assertIn("OTOOLS_LAUNCH_FAILED", cmd)
        self.assertIn("launch-k.log", cmd)
        self.assertTrue(cmd.rstrip().endswith("&"))   # backgrounds itself
        self.assertIn("nohup", cmd)

    def test_bg_command_quotes_json_arg(self):
        # A JSON vLLM arg must survive as one shell token inside the backgrounded run.
        argv = ["run", "img:tag", "--speculative-config", '{"method":"mtp"}']
        cmd = mm._launch_bg_command("k", "img:tag", argv)
        self.assertIn('method', cmd)
        # the whole command must be valid shell (balanced quoting) -> shlex parses it
        import shlex as _sh
        self.assertIsInstance(_sh.split(cmd.replace("&", "")), list)

    def test_launch_status_classifier(self):
        self.assertEqual(mm._launch_status(""), "none")
        self.assertEqual(mm._launch_status("   \n"), "none")
        self.assertEqual(mm._launch_status("pulling...\nOTOOLS_LAUNCH_OK\n"), "ok")
        self.assertEqual(mm._launch_status("err\nOTOOLS_LAUNCH_FAILED\n"), "failed")
        self.assertEqual(mm._launch_status("Pulling fs layer 40%%..."), "running")

    def test_pulling_keys_local(self):
        import shutil as _sh
        if not _sh.which("sh"):
            self.skipTest("needs a POSIX sh")
        home = tempfile.mkdtemp()
        d = os.path.join(home, ".config", "otools")
        os.makedirs(d)
        with open(os.path.join(d, "launch-modelA.log"), "w") as f:
            f.write("Pulling fs layer...\n")            # no marker -> mid-pull
        with open(os.path.join(d, "launch-modelB.log"), "w") as f:
            f.write("done\nOTOOLS_LAUNCH_OK\n")          # completed -> not pulling
        old = os.environ.get("HOME")
        os.environ["HOME"] = home
        try:
            keys = mm._pulling_keys(None)
        finally:
            if old is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old
        self.assertEqual(keys, ["modelA"])

    def test_pulling_keys_none_when_empty(self):
        import shutil as _sh
        if not _sh.which("sh"):
            self.skipTest("needs a POSIX sh")
        home = tempfile.mkdtemp()          # no ~/.config/otools at all
        old = os.environ.get("HOME")
        os.environ["HOME"] = home
        try:
            self.assertEqual(mm._pulling_keys(None), [])
        finally:
            if old is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old


class LaunchGuardTests(unittest.TestCase):
    """`launch` must refuse a silent local run when hosts are registered."""

    def setUp(self):
        self._hosts, self._remote, self._need = mm.HOSTS_FILE, mm.REMOTE, mm.need_docker
        mm.HOSTS_FILE = os.path.join(tempfile.mkdtemp(), "hosts")
        mm.REMOTE = None
        mm.need_docker = lambda: None      # isolate from a real docker/ssh
        self.cfg = mm.load_config()
        self.key = "glm-4.7-flash"         # a profile with no per-model remote / assets

    def tearDown(self):
        mm.HOSTS_FILE, mm.REMOTE, mm.need_docker = self._hosts, self._remote, self._need

    def _launch(self, **kw):
        base = dict(key=self.key, remote=None, local=False, dry_run=True,
                    foreground=False, keep=False, force=False, no_fetch=True,
                    refresh=False, wait=False)
        base.update(kw)
        # Swallow the dry-run banner so the test suite stays quiet.
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            mm.cmd_launch(SimpleNamespace(**base))

    def test_guard_blocks_local_when_hosts_registered(self):
        mm.save_hosts([("dgx1", "otto@a")])
        with self.assertRaises(SystemExit):
            self._launch()

    def test_local_flag_bypasses_guard(self):
        mm.save_hosts([("dgx1", "otto@a")])
        self._launch(local=True)   # --local + --dry-run: must not raise

    def test_no_hosts_allows_local(self):
        self._launch()             # empty registry: local is fine


class DropCachesTests(unittest.TestCase):
    """Automatic page-cache drop before every launch (UMA guard, vLLM #35313)."""

    def test_cmd_is_non_interactive(self):
        # `sudo -n` must never prompt: a missing NOPASSWD rule fails fast, not hangs.
        self.assertTrue(mm.DROP_CACHES_CMD.startswith("sudo -n "))
        self.assertIn(mm.DROP_CACHES_HELPER, mm.DROP_CACHES_CMD)

    def test_bg_command_always_injects(self):
        bg = mm._launch_bg_command("k", "img", ["run", "--name", "x"])
        # ordering: pull the image, then drop the cache, then run the container.
        self.assertLess(bg.index("pull"), bg.index(mm.DROP_CACHES_HELPER))
        self.assertLess(bg.index(mm.DROP_CACHES_HELPER), bg.index("--name"))
        # grouped `|| true` so a sudo failure can't abort the launch.
        self.assertIn("|| true", bg)

    def test_install_cmd_scopes_the_rule(self):
        cmd = mm._drop_caches_install_cmd()
        self.assertIn("NOPASSWD: %s" % mm.DROP_CACHES_HELPER, cmd)
        self.assertIn(mm.DROP_CACHES_SUDOERS, cmd)
        self.assertIn("visudo -cf", cmd)          # validates before it can break sudo


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
                # model is a POSITIONAL arg right after the image (`vllm serve <model>`),
                # not the deprecated `--model` flag.
                self.assertEqual(argv[argv.index(merged["image"]) + 1], merged["model"])
                self.assertNotIn("--model", argv)
                self.assertIn("--port", argv)
                self.assertNotIn(None, argv)
                # format_run must not raise for any profile
                self.assertIsInstance(mm.format_run(argv), str)

    def test_container_names_unique(self):
        names = [mm.container_name(k) for k in self.keys]
        self.assertEqual(len(names), len(set(names)))


class ExtendsTests(unittest.TestCase):
    def setUp(self):
        # Test against the committed source of truth (DEFAULT_CONFIG), NOT load_config()
        # — the latter reads the git-ignored model_manager.json sandbox, which can be
        # stale (e.g. predating a profile rename), making this test machine-dependent.
        self.cfg = json.loads(json.dumps(mm.DEFAULT_CONFIG))

    def test_deep_merge(self):
        base = {"a": 1, "b": {"x": 1, "y": 2}, "lst": [1]}
        over = {"b": {"y": 3, "z": 4}, "c": 5, "lst": [9]}
        out = mm._deep_merge(base, over)
        self.assertEqual(out, {"a": 1, "b": {"x": 1, "y": 3, "z": 4}, "c": 5, "lst": [9]})
        self.assertEqual(base["b"], {"x": 1, "y": 2})  # base not mutated

    def test_nemotron_1m_extends_256k(self):
        m = mm.merge_model(self.cfg, "nemotron-3-super-120b-a12b-nvfp4-1m")
        base = mm.merge_model(self.cfg, "nemotron-3-super-120b-a12b-nvfp4-256k")
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
        parsed = json.loads(argv[i + 1])   # the whole JSON is one argv token, and valid
        self.assertEqual(parsed["method"], "mtp")
        self.assertIn("num_speculative_tokens", parsed)   # value is tunable; don't hardcode it

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


class HostAddressingTests(unittest.TestCase):
    """A bare hostname addresses the model on that box: `logs dgx-2` needs no model name."""

    def setUp(self):
        self._hosts, self._lm = mm.HOSTS_FILE, mm.list_managed
        mm.HOSTS_FILE = os.path.join(tempfile.mkdtemp(), "hosts")
        with open(mm.HOSTS_FILE, "w") as f:
            f.write("dgx-2\totto@192.168.50.102\n")

    def tearDown(self):
        mm.HOSTS_FILE, mm.list_managed = self._hosts, self._lm

    def test_host_of_recognizes_alias_ip_userhost(self):
        self.assertEqual(mm.host_of("dgx-2"), "dgx-2")               # registered alias
        self.assertEqual(mm.host_of("otto@1.2.3.4"), "otto@1.2.3.4")  # user@host
        self.assertEqual(mm.host_of("192.168.50.103"), "192.168.50.103")  # bare IP

    def test_host_of_rejects_model_and_container_names(self):
        self.assertIsNone(mm.host_of("qwen3.6-35b-nvfp4"))
        self.assertIsNone(mm.host_of("otools-vllm-glm-4.7-flash"))
        self.assertIsNone(mm.host_of("dgx-9"))       # not a registered alias
        self.assertIsNone(mm.host_of(""))
        self.assertIsNone(mm.host_of(None))

    def test_managed_on_single(self):
        mm.list_managed = lambda **k: [{"Names": "otools-vllm-m1", "Labels": "otools.model=m1"}]
        self.assertEqual(mm._managed_on("otto@a"), ("otools-vllm-m1", "m1"))

    def test_managed_on_none_errors(self):
        mm.list_managed = lambda **k: []
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mm._managed_on("otto@a")

    def test_managed_on_ambiguous_errors(self):
        mm.list_managed = lambda **k: [{"Names": "a"}, {"Names": "b"}]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mm._managed_on("otto@a")


class FmtTokensTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(mm._fmt_tokens(262144), "256K")
        self.assertEqual(mm._fmt_tokens(1048576), "1M")
        self.assertEqual(mm._fmt_tokens(202752), "198K")
        self.assertEqual(mm._fmt_tokens(524288), "512K")
        self.assertEqual(mm._fmt_tokens(None), "?")
        self.assertEqual(mm._fmt_tokens(1000), "1000")

    def test_tokps(self):
        self.assertEqual(mm._fmt_tokps(42), "42")
        self.assertEqual(mm._fmt_tokps(41.6), "42")     # rounds
        self.assertEqual(mm._fmt_tokps(None), "—")      # unmeasured
        self.assertEqual(mm._fmt_tokps("nope"), "—")


if __name__ == "__main__":
    unittest.main(verbosity=2)

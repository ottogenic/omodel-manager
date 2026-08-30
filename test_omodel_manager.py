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
import socket
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from types import SimpleNamespace
from unittest import mock

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
        mm.save_hosts(["user@a", "user@b", "user@a"])
        self.assertEqual(mm.load_hosts(), [("user@a", "user@a"), ("user@b", "user@b")])
        self.assertEqual(mm.host_targets(), ["user@a", "user@b"])

    def test_alias_roundtrip(self):
        mm.save_hosts([("dgx1", "user@a"), ("dgx2", "user@b")])
        self.assertEqual(mm.load_hosts(), [("dgx1", "user@a"), ("dgx2", "user@b")])

    def test_dedup_by_target_keeps_first_alias(self):
        mm.save_hosts([("dgx1", "user@a"), ("other", "user@a")])  # same target twice
        self.assertEqual(mm.load_hosts(), [("dgx1", "user@a")])

    def test_resolve_host_alias_and_passthrough(self):
        mm.save_hosts([("dgx1", "user@a")])
        self.assertEqual(mm.resolve_host("dgx1"), "user@a")     # alias -> target
        self.assertEqual(mm.resolve_host("user@b"), "user@b")   # unknown/raw passes through
        self.assertEqual(mm.resolve_host(""), "")               # empty passes through
        self.assertIsNone(mm.resolve_host(None))

    def test_missing_file_is_empty(self):
        self.assertEqual(mm.load_hosts(), [])
        self.assertEqual(mm.host_targets(), [])

    def test_host_label_prefers_alias(self):
        # suggested commands should echo the alias, not the raw user@ip
        mm.save_hosts([("dgx1", "user@192.0.2.101")])
        self.assertEqual(mm._host_label("user@192.0.2.101"), "dgx1")
        self.assertEqual(mm._host_label("user@unknown"), "user@unknown")  # no alias -> raw
        self.assertIsNone(mm._host_label(None))

    def test_dedupe_host_targets_collapses_aliases_for_same_machine(self):
        identities = {"user@lan": "machine-a", "user@fabric": "machine-a",
                      "user@other": "machine-b"}
        with mock.patch.object(mm, "_host_machine_id", side_effect=identities.get):
            self.assertEqual(mm._dedupe_host_targets(
                ["user@lan", "user@fabric", "user@other"]),
                ["user@lan", "user@other"])

    def test_dedupe_host_targets_preserves_unknown_targets(self):
        with mock.patch.object(mm, "_host_machine_id", return_value=None):
            self.assertEqual(mm._dedupe_host_targets(["user@a", "user@b"]),
                             ["user@a", "user@b"])


class InstallTests(unittest.TestCase):
    """`install` targets this machine when no remote is named."""

    def setUp(self):
        self._hosts, self._remote = mm.HOSTS_FILE, mm.REMOTE
        mm.HOSTS_FILE = os.path.join(tempfile.mkdtemp(), "hosts")
        mm.REMOTE = "user@configured-default"

    def tearDown(self):
        mm.HOSTS_FILE, mm.REMOTE = self._hosts, self._remote

    def test_no_target_installs_locally_without_ssh_or_registration(self):
        args = SimpleNamespace(target=None, alias=None, fix=True)
        with mock.patch.object(mm, "_setup_host", return_value=(True, True)) as setup, \
                mock.patch.object(mm.shutil, "which") as which, \
                mock.patch.object(mm, "save_hosts") as save:
            with self.assertRaises(SystemExit) as raised:
                mm.cmd_install(args)
        self.assertEqual(raised.exception.code, 0)
        setup.assert_called_once_with(None, True)
        which.assert_not_called()
        save.assert_not_called()

    def test_explicit_target_keeps_remote_setup_and_registration(self):
        args = SimpleNamespace(target="user@192.0.2.101", alias="dgx1", fix=True)
        with mock.patch.object(mm, "_setup_host", return_value=(True, True)) as setup, \
                mock.patch.object(mm.shutil, "which", return_value="/usr/bin/ssh"), \
                mock.patch.object(mm, "load_hosts", return_value=[]), \
                mock.patch.object(mm, "save_hosts") as save, \
                mock.patch.object(mm, "auto_register_dgx_cluster") as discover:
            with self.assertRaises(SystemExit) as raised:
                mm.cmd_install(args)
        self.assertEqual(raised.exception.code, 0)
        setup.assert_called_once_with("user@192.0.2.101", True)
        save.assert_called_once_with([("dgx1", "user@192.0.2.101")])
        discover.assert_called_once_with([("dgx1", "user@192.0.2.101")],
                                         preferred_aliases=["dgx1"])

    def test_local_setup_skips_ssh_prerequisites(self):
        def setup_ok(target, cmd):
            self.assertIsNone(target)
            if cmd == "docker --version":
                return True, "Docker version 29"
            if cmd == "id -nG":
                return True, "user docker"
            if cmd.startswith("nvidia-smi"):
                return True, "GPU 0: NVIDIA GB10"
            if "Runtimes" in cmd:
                return True, '{"nvidia": {}}'
            return True, ""

        out = io.StringIO()
        with mock.patch.object(mm, "hf_token", return_value="hf_test"), \
                mock.patch.object(mm, "_setup_ok", side_effect=setup_ok), \
                contextlib.redirect_stdout(out):
            ready, local_ok = mm._setup_host(None, False)
        self.assertTrue(ready)
        self.assertTrue(local_ok)
        self.assertIn("Setup for this machine", out.getvalue())
        self.assertNotIn("SSH", out.getvalue())

    def test_remote_setup_requires_curl_for_health_checks(self):
        def setup_ok(target, cmd):
            self.assertEqual(target, "user@host")
            if cmd == "command -v curl":
                return False, ""
            if cmd == "docker --version":
                return True, "Docker version 29"
            if cmd == "id -nG":
                return True, "user docker"
            if cmd.startswith("nvidia-smi"):
                return True, "GPU 0: NVIDIA GB10"
            if "Runtimes" in cmd:
                return True, '{"nvidia": {}}'
            return True, "ok"

        out = io.StringIO()
        with mock.patch.object(mm, "hf_token", return_value="hf_test"), \
                mock.patch.object(mm.os.path, "exists", return_value=True), \
                mock.patch.object(mm, "_setup_ok", side_effect=setup_ok) as check, \
                contextlib.redirect_stdout(out):
            ready, ssh_ok = mm._setup_host("user@host", False)
        self.assertFalse(ready)
        self.assertTrue(ssh_ok)
        self.assertIn(mock.call("user@host", "command -v curl"), check.call_args_list)
        self.assertIn("[FAIL] curl", out.getvalue())

    def test_setup_runner_executes_locally_without_ssh(self):
        completed = SimpleNamespace(returncode=0, stdout="ok\n")
        with mock.patch.object(mm.subprocess, "run", return_value=completed) as run, \
                mock.patch.object(mm, "run_remote") as remote:
            result = mm._setup_run(None, "command -v docker", capture=True)
        self.assertIs(result, completed)
        run.assert_called_once_with(["sh", "-c", "command -v docker"], text=True,
                                    capture_output=True)
        remote.assert_not_called()


class DgxClusterDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self._clusters = mm.CLUSTERS_FILE
        mm.CLUSTERS_FILE = os.path.join(tempfile.mkdtemp(), "clusters.json")
        self.hosts = [("dgx3", "otto@dgx3"), ("dgx4", "otto@dgx4")]

    def tearDown(self):
        mm.CLUSTERS_FILE = self._clusters

    @staticmethod
    def host(alias, target, machine_id, suffix):
        return {
            "alias": alias,
            "target": target,
            "hostname": f"otto-{alias}",
            "machine_id": machine_id,
            "fabric": {
                "enp1s0f1np1": {
                    "ip": f"10.100.176.{suffix}", "network": "10.100.176.0/24",
                    "ucx_device": "rocep1s0f1:1", "mtu": 1500,
                },
                "enP2p1s0f1np1": {
                    "ip": f"10.100.177.{suffix}", "network": "10.100.177.0/24",
                    "ucx_device": "roceP2p1s0f1:1", "mtu": 1500,
                },
            },
        }

    def test_reads_identity_and_active_fabric_from_dgx(self):
        links = [{
            "ifname": "enp1s0f1np1", "operstate": "UP", "mtu": 1500,
            "addr_info": [{"family": "inet", "scope": "global",
                           "local": "10.100.176.2", "prefixlen": 24}],
        }]

        def host_text(target, args):
            responses = {
                ("test", "-f", "/etc/netplan/99-nvidia-sync-cluster.yaml"): (True, ""),
                ("uname", "-m"): (True, "aarch64"),
                ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"):
                    (True, "NVIDIA GB10"),
                ("hostnamectl", "--static"): (True, "otto-dgx-3"),
                ("cat", "/etc/machine-id"): (True, "machine-three"),
                ("ibdev2netdev",):
                    (True, "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"),
                ("ip", "-j", "address", "show"): (True, json.dumps(links)),
                ("cat", "/sys/class/net/enp1s0f1np1/address"):
                    (True, "fc:9d:05:13:86:8d"),
            }
            return responses[tuple(args)]

        with mock.patch.object(mm, "_host_text", side_effect=host_text):
            host = mm._dgx_cluster_host("dgx3", "otto@dgx3")
        self.assertEqual(host["hostname"], "otto-dgx-3")
        self.assertEqual(host["machine_id"], "machine-three")
        self.assertEqual(host["fabric"]["enp1s0f1np1"]["ucx_device"],
                         "rocep1s0f1:1")

    def test_pair_requires_bidirectional_interface_bound_ping(self):
        head = self.host("dgx3", "otto@dgx3", "three", 2)
        worker = self.host("dgx4", "otto@dgx4", "four", 1)
        with mock.patch.object(mm, "_host_text", return_value=(True, "")) as probe:
            cfg = mm._dgx_pair_config(head, worker)
        self.assertEqual(cfg["fabric"]["head_ips"],
                         ["10.100.176.2", "10.100.177.2"])
        self.assertEqual(cfg["fabric"]["worker_ips"],
                         ["10.100.176.1", "10.100.177.1"])
        self.assertEqual(probe.call_count, 4)
        for call in probe.call_args_list:
            self.assertEqual(call.args[1][0], "ping")
            self.assertIn("-I", call.args[1])

    def test_pair_is_rejected_when_a_rail_ping_fails(self):
        head = self.host("dgx3", "otto@dgx3", "three", 2)
        worker = self.host("dgx4", "otto@dgx4", "four", 1)
        with mock.patch.object(mm, "_host_text",
                               side_effect=[(True, ""), (False, "unreachable")]):
            self.assertIsNone(mm._dgx_pair_config(head, worker))

    def test_pair_is_rejected_when_neighbor_mac_is_not_candidate_peer(self):
        head = self.host("dgx3", "otto@dgx3", "three", 2)
        worker = self.host("dgx4", "otto@dgx4", "four", 1)
        for host, mac in ((head, "aa:aa:aa:aa:aa:03"),
                          (worker, "aa:aa:aa:aa:aa:04")):
            for fabric in host["fabric"].values():
                fabric["mac"] = mac
        responses = [(True, ""), (True, ""),
                     (True, "10.100.176.1 lladdr bb:bb:bb:bb:bb:bb REACHABLE")]
        with mock.patch.object(mm, "_host_text", side_effect=responses):
            self.assertIsNone(mm._dgx_pair_config(head, worker))

    def test_registers_one_discovered_pair_with_deterministic_name(self):
        found = {
            "otto@dgx3": self.host("dgx3", "otto@dgx3", "three", 2),
            "otto@dgx4": self.host("dgx4", "otto@dgx4", "four", 1),
        }
        with mock.patch.object(mm, "_dgx_cluster_host",
                               side_effect=lambda alias, target: found[target]), \
                mock.patch.object(mm, "_host_text", return_value=(True, "")):
            added = mm.auto_register_dgx_cluster(self.hosts)
        self.assertEqual(added, ["dgx3-dgx4"])
        saved = mm.load_clusters()["dgx3-dgx4"]
        self.assertEqual(saved["head"], "dgx3")
        self.assertEqual(saved["worker"], "dgx4")
        self.assertEqual(saved["head_machine_id"], "three")
        self.assertEqual(saved["worker_machine_id"], "four")

    def test_existing_pair_is_not_prompted_or_overwritten(self):
        existing = {"Beebo": {"head": "dgx3", "worker": "dgx4", "fabric": {}}}
        mm.save_clusters(existing)
        found = {
            "otto@dgx3": self.host("dgx3", "otto@dgx3", "three", 2),
            "otto@dgx4": self.host("dgx4", "otto@dgx4", "four", 1),
        }
        with mock.patch.object(mm, "_dgx_cluster_host",
                               side_effect=lambda alias, target: found[target]), \
                mock.patch.object(mm, "_host_text", return_value=(True, "")):
            added = mm.auto_register_dgx_cluster(self.hosts)
        self.assertEqual(added, [])
        self.assertEqual(mm.load_clusters(), existing)

    def test_existing_pair_matches_alternate_aliases_by_machine_id(self):
        hosts = [("dgx3", "otto@dgx3-mgmt"), ("dgx3-fabric", "otto@dgx3-fabric"),
                 ("dgx4", "otto@dgx4-mgmt"), ("dgx4-fabric", "otto@dgx4-fabric")]
        mm.save_clusters({"Beebo": {"head": "dgx3-fabric", "worker": "dgx4-fabric",
                                     "fabric": {}}})
        found = {
            target: self.host(alias, target, "three" if "dgx3" in target else "four",
                              2 if "dgx3" in target else 1)
            for alias, target in hosts
        }
        with mock.patch.object(mm, "_dgx_cluster_host",
                               side_effect=lambda alias, target: found[target]), \
                mock.patch.object(mm, "_host_text", return_value=(True, "")):
            added = mm.auto_register_dgx_cluster(hosts)
        self.assertEqual(added, [])

    def test_multiple_possible_pairs_are_not_guessed(self):
        hosts = self.hosts + [("dgx5", "otto@dgx5")]
        found = {
            target: self.host(alias, target, alias, index)
            for index, (alias, target) in enumerate(hosts, 1)
        }
        err = io.StringIO()
        with mock.patch.object(mm, "_dgx_cluster_host",
                               side_effect=lambda alias, target: found[target]), \
                mock.patch.object(mm, "_host_text", return_value=(True, "")), \
                contextlib.redirect_stderr(err):
            added = mm.auto_register_dgx_cluster(hosts)
        self.assertEqual(added, [])
        self.assertIn("multiple possible", err.getvalue())

    def test_preferred_new_host_disambiguates_multiple_pairs(self):
        hosts = self.hosts + [("dgx5", "otto@dgx5")]
        found = {
            target: self.host(alias, target, alias, index)
            for index, (alias, target) in enumerate(hosts, 1)
        }
        with mock.patch.object(mm, "_dgx_cluster_host",
                               side_effect=lambda alias, target: found[target]), \
                mock.patch.object(mm, "_host_text", return_value=(True, "")):
            added = mm.auto_register_dgx_cluster(hosts, preferred_aliases=["dgx5"])
        self.assertEqual(added, [])  # dgx5 still has two possible peers, so remains safe.


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


class LaunchTargetTests(unittest.TestCase):
    """`launch` with no host runs locally -- visibly when hosts are registered."""

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
        base = dict(key=self.key, host=None, remote=None, local=False, dry_run=True,
                    foreground=False, keep=False, force=False, no_fetch=True,
                    refresh=False, wait=False)
        base.update(kw)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            mm.cmd_launch(SimpleNamespace(**base))
        return out.getvalue()

    def test_default_is_local_with_a_visible_note(self):
        # Hosts registered, none chosen -> launch locally and say so (was a hard error).
        mm.save_hosts([("dgx1", "user@a")])
        out = self._launch()
        self.assertIn("Launching locally", out)
        self.assertIn("dgx1", out)                 # the note names the registered hosts

    def test_local_flag_overrides_default_remote(self):
        # --local must undo a defaults.remote that main() already resolved into REMOTE.
        mm.save_hosts([("dgx1", "user@a")])
        mm.REMOTE = "user@a"
        out = self._launch(local=True)
        self.assertIsNone(mm.REMOTE)
        self.assertIn("locally", out)              # dry-run banner confirms a local launch

    def test_local_plus_explicit_host_conflicts(self):
        with self.assertRaises(SystemExit):
            self._launch(local=True, remote="dgx1")

    def test_no_hosts_is_quietly_local(self):
        out = self._launch()                       # empty registry: local, no note
        self.assertNotIn("Registered hosts", out)


class LaunchClusterDispatchTests(unittest.TestCase):
    def setUp(self):
        self._clusters = mm.CLUSTERS_FILE
        mm.CLUSTERS_FILE = os.path.join(tempfile.mkdtemp(), "clusters.json")
        mm.save_clusters({"Beebo": {"head": "dgx3", "worker": "dgx4"}})

    def tearDown(self):
        mm.CLUSTERS_FILE = self._clusters

    def args(self, **kw):
        base = dict(key="deepseek-v4-flash-0731", host="beebo", remote=None,
                    local=False, dry_run=True, foreground=False, keep=False,
                    force=False, no_fetch=False, refresh=False, wait=False)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_cluster_profile_dispatches_with_canonical_cluster_name(self):
        with mock.patch.object(mm, "cmd_cluster_launch") as launch:
            mm.cmd_launch(self.args())
        forwarded = launch.call_args.args[0]
        self.assertEqual(forwarded.profile, "deepseek-v4-flash-0731")
        self.assertEqual(mm.cluster_config(forwarded.name)["name"], "Beebo")
        self.assertTrue(forwarded.dry_run)
        self.assertEqual(forwarded.startup_timeout, 1800)

    def test_cluster_profile_accepts_shared_host_option(self):
        with mock.patch.object(mm, "cmd_cluster_launch") as launch:
            mm.cmd_launch(self.args(host=None, remote="beebo"))
        self.assertEqual(launch.call_args.args[0].name, "beebo")

    def test_vllm_cluster_profile_forwards_keep(self):
        with mock.patch.object(mm, "cmd_cluster_launch") as launch:
            mm.cmd_launch(self.args(key="qwen3.8-flash-next-fp8", keep=True))
        self.assertTrue(launch.call_args.args[0].keep)

    def test_cluster_status_uses_canonical_name_for_labels(self):
        cfg = {"name": "Beebo", "head": "dgx3", "worker": "dgx4"}
        row = {"Names": "rank", "Labels": f"{mm.LABEL_CLUSTER}=Beebo", "Status": "Up"}
        args = SimpleNamespace(name="beebo")
        out = io.StringIO()
        with mock.patch.object(mm, "cluster_config", return_value=cfg), \
                mock.patch.object(mm, "_cluster_targets", return_value=("head", "worker")), \
                mock.patch.object(mm, "list_managed", return_value=[row]), \
                contextlib.redirect_stdout(out):
            mm.cmd_cluster_status(args)
        self.assertIn("rank", out.getvalue())
        self.assertNotIn("busy (other model)", out.getvalue())

    def test_cluster_profile_requires_a_cluster_target(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mm.cmd_launch(self.args(host=None))

    def test_cluster_profile_rejects_local_flag(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mm.cmd_launch(self.args(local=True))

    def test_cluster_profile_rejects_single_host_options(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mm.cmd_launch(self.args(foreground=True))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mm.cmd_launch(self.args(keep=True))


class ListManagedTests(unittest.TestCase):
    """list_managed tells 'can't query the host' (None) from 'nothing running' ([])."""

    def setUp(self):
        self._d = mm._docker_on

    def tearDown(self):
        mm._docker_on = self._d

    def test_failure_returns_none(self):
        mm._docker_on = lambda t, a, capture=False: SimpleNamespace(
            returncode=255, stdout="", stderr="ssh: connect to host user@down: refused")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(mm.list_managed(remote="user@down"))

    def test_empty_returns_list(self):
        mm._docker_on = lambda t, a, capture=False: SimpleNamespace(
            returncode=0, stdout="", stderr="")
        self.assertEqual(mm.list_managed(remote="user@a"), [])

    def test_rows_parsed(self):
        row = json.dumps({"Names": "otools-vllm-x", "Labels": "otools.model=x"})
        mm._docker_on = lambda t, a, capture=False: SimpleNamespace(
            returncode=0, stdout=row + "\n", stderr="")
        self.assertEqual(mm.list_managed(remote="user@a")[0]["Names"], "otools-vllm-x")

    def test_include_stopped_adds_all_flag(self):
        calls = []
        mm._docker_on = lambda t, a, capture=False: (
            calls.append(a) or SimpleNamespace(returncode=0, stdout="", stderr=""))
        mm.list_managed(remote="user@a", include_stopped=True)
        self.assertIn("-a", calls[0])


class PsUnreachableTests(unittest.TestCase):
    """A host that can't be queried shows as 'unreachable', not 'idle' (README promise)."""

    def setUp(self):
        self._hosts, self._clusters, self._remote = mm.HOSTS_FILE, mm.CLUSTERS_FILE, mm.REMOTE
        self._lm, self._shutil = mm.list_managed, mm.shutil
        root = tempfile.mkdtemp()
        mm.HOSTS_FILE = os.path.join(root, "hosts")
        mm.CLUSTERS_FILE = os.path.join(root, "clusters.json")
        mm.REMOTE = None
        mm.save_hosts([("dgx1", "user@a")])
        mm.shutil = SimpleNamespace(which=lambda n: "/bin/" + n)   # pretend ssh exists

    def tearDown(self):
        mm.HOSTS_FILE, mm.CLUSTERS_FILE, mm.REMOTE = self._hosts, self._clusters, self._remote
        mm.list_managed, mm.shutil = self._lm, self._shutil

    def test_down_host_reads_unreachable(self):
        mm.list_managed = lambda **k: None         # SSH/docker failure on every host
        out = io.StringIO()
        with mock.patch.object(mm, "_host_machine_id", return_value="machine-a"), \
                contextlib.redirect_stdout(out):
            mm.cmd_ps(SimpleNamespace(all=False, hosts=None))
        self.assertIn("unreachable", out.getvalue())
        self.assertNotIn("idle", out.getvalue())

    def test_table_columns_expand_for_cluster_names(self):
        rows = [
            ("local-head", "test-cluster", "qwen3-235b-a22b-fp4-test-cluster-head",
              "qwen3-235b-a22b-fp4", "8355", "Up 12 hours"),
            ("worker-host", "test-cluster", "qwen3-235b-a22b-fp4-test-cluster-worker",
              "qwen3-235b-a22b-fp4", "?", "Up 12 hours"),
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mm._print_ps_table(rows)
        lines = out.getvalue().splitlines()
        cluster_column = lines[0].index("CLUSTER")
        model_column = lines[0].index("MODEL")
        port_column = lines[0].index("PORT")
        self.assertTrue(lines[1][cluster_column:].startswith(rows[0][1]))
        self.assertTrue(lines[1][model_column:].startswith(rows[0][3]))
        self.assertTrue(lines[2][model_column:].startswith(rows[1][3]))
        self.assertTrue(lines[1][port_column:].startswith(rows[0][4]))

    def test_cluster_rows_show_name_and_cluster_commands(self):
        mm.list_managed = lambda **k: [{
            "Names": "otools-vllm-deepseek-Beebo-head",
            "Labels": (f"{mm.LABEL_MODEL}=deepseek-v4-flash-0731,"
                       f"{mm.LABEL_CLUSTER}=Beebo,{mm.LABEL_PORT}=8000"),
            "Status": "Up 1 hour",
        }]
        out = io.StringIO()
        with mock.patch.object(mm, "_host_machine_id", return_value="machine-a"), \
                mock.patch.object(mm, "_pulling_keys", return_value=[]), \
                contextlib.redirect_stdout(out):
            mm.cmd_ps(SimpleNamespace(all=False, hosts=None))
        body = out.getvalue()
        self.assertLess(body.index("HOST"), body.index("CLUSTER"))
        self.assertIn("Beebo", body)
        self.assertIn("omm cluster status Beebo", body)
        self.assertIn("omm cluster health Beebo deepseek-v4-flash-0731", body)

    def test_single_host_model_shows_registered_cluster_membership(self):
        mm.save_clusters({"Beebo": {"head": "dgx1", "worker": "user@b", "fabric": {}}})
        mm.list_managed = lambda **k: [{
            "Names": "otools-vllm-model-a",
            "Labels": f"{mm.LABEL_MODEL}=model-a,{mm.LABEL_PORT}=8000",
            "Status": "Up 1 hour",
        }]
        out = io.StringIO()
        with mock.patch.object(mm, "_host_machine_id", return_value="machine-a"), \
                mock.patch.object(mm, "_pulling_keys", return_value=[]), \
                contextlib.redirect_stdout(out):
            mm.cmd_ps(SimpleNamespace(all=False, hosts=None))
        row = out.getvalue().splitlines()[1]
        self.assertIn("dgx1", row)
        self.assertIn("Beebo", row)
        self.assertIn("omm logs dgx1 -f", out.getvalue())

    def test_membership_survives_alternate_alias_deduplication(self):
        mm.save_hosts([("dgx-mgmt", "user@mgmt"), ("dgx-fabric", "user@fabric")])
        mm.save_clusters({"Beebo": {"head": "dgx-fabric", "worker": "user@worker",
                                     "fabric": {}}})
        mm.list_managed = lambda **k: [{
            "Names": "otools-vllm-model-a",
            "Labels": f"{mm.LABEL_MODEL}=model-a,{mm.LABEL_PORT}=8000",
            "Status": "Up 1 hour",
        }]
        identities = {"user@mgmt": "same-machine", "user@fabric": "same-machine",
                      "user@worker": "worker-machine"}
        out = io.StringIO()
        with mock.patch.object(mm, "_host_machine_id", side_effect=identities.get), \
                mock.patch.object(mm, "_pulling_keys", return_value=[]), \
                contextlib.redirect_stdout(out):
            mm.cmd_ps(SimpleNamespace(all=False, hosts=None))
        row = out.getvalue().splitlines()[1]
        self.assertIn("dgx-mgmt", row)
        self.assertIn("Beebo", row)


class SyncTests(unittest.TestCase):
    """`sync` resets model_manager.json from DEFAULT_CONFIG, backing up local edits."""

    def setUp(self):
        self._cfgpath = mm.CONFIG_PATH
        mm.CONFIG_PATH = os.path.join(tempfile.mkdtemp(), "model_manager.json")

    def tearDown(self):
        mm.CONFIG_PATH = self._cfgpath

    def _sync(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mm.cmd_sync(SimpleNamespace())
        return out.getvalue()

    def test_creates_when_missing(self):
        out = self._sync()
        self.assertIn("Synced", out)
        with open(mm.CONFIG_PATH) as f:
            self.assertEqual(json.load(f), mm.DEFAULT_CONFIG)
        self.assertFalse(os.path.exists(mm.CONFIG_PATH + ".bak"))  # nothing to back up

    def test_backs_up_and_replaces_stale_config(self):
        stale = json.loads(json.dumps(mm.DEFAULT_CONFIG))
        stale["models"].pop(next(iter(stale["models"])))       # simulate pre-merge sandbox
        stale["models"]["local-experiment"] = {"model": "x/y", "port": 8000}
        mm.save_config(stale)
        out = self._sync()
        self.assertIn("Backed up", out)
        self.assertIn("new profiles:", out)
        self.assertIn("local-experiment", out)                 # surfaced as removed
        self.assertTrue(os.path.exists(mm.CONFIG_PATH + ".bak"))
        with open(mm.CONFIG_PATH + ".bak") as f:
            self.assertEqual(json.load(f), stale)              # old content preserved
        with open(mm.CONFIG_PATH) as f:
            self.assertEqual(json.load(f), mm.DEFAULT_CONFIG)

    def test_in_sync_is_a_noop(self):
        mm.save_config(mm.DEFAULT_CONFIG)
        out = self._sync()
        self.assertIn("Already in sync", out)
        self.assertFalse(os.path.exists(mm.CONFIG_PATH + ".bak"))


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


class LagunaProfileTests(unittest.TestCase):
    """The RC2 target and matched DFlash draft stay pinned and independently usable."""

    def setUp(self):
        self.cfg = json.loads(json.dumps(mm.DEFAULT_CONFIG))

    def test_target_only_profile(self):
        profile = mm.merge_model(self.cfg, "laguna-s-2.1-nvfp4")
        args = profile["vllm_args"]
        self.assertIn("@sha256:", profile["image"])
        self.assertEqual(args["revision"], "f8fdfcdc4e7b0c474a0102430a8cae0a3a358669")
        self.assertFalse(args["trust-remote-code"])
        self.assertEqual(args["kv-cache-dtype"], "fp8")
        self.assertEqual(args["max-model-len"], 229376)
        self.assertEqual(args["max-num-seqs"], 1)
        self.assertTrue(args["enable-prefix-caching"])
        self.assertEqual(json.loads(args["reasoning-config"]), {
            "reasoning_start_str": "<think>", "reasoning_end_str": "</think>"})
        self.assertNotIn("attention-backend", args)
        self.assertNotIn("moe-backend", args)
        self.assertNotIn("speculative-config", args)

    def test_dflash_profile_pins_matched_draft(self):
        profile = mm.merge_model(self.cfg, "laguna-dflash-s-2.1-nvfp4")
        args = profile["vllm_args"]
        spec = json.loads(args["speculative-config"])
        self.assertEqual(args["revision"], "f8fdfcdc4e7b0c474a0102430a8cae0a3a358669")
        self.assertEqual(args["served-model-name"], "laguna-dflash-s-2.1-nvfp4")
        self.assertEqual(args["max-model-len"], 131072)
        self.assertEqual(args["max-num-batched-tokens"], 9216)
        self.assertEqual(spec["method"], "dflash")
        self.assertEqual(spec["model"], "poolside/Laguna-S-2.1-DFlash-NVFP4")
        self.assertEqual(spec["revision"], "b3b5921a900b9e0a1e27e50bdaeb480692a6d19b")
        self.assertEqual(spec["num_speculative_tokens"], 7)


class NewModelProfileTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(json.dumps(mm.DEFAULT_CONFIG))

    def test_muse_uses_validated_dflash_settings(self):
        profile = mm.merge_model(self.cfg, "muse-glimmer-30b-nvfp4")
        args = profile["vllm_args"]
        spec = json.loads(args["speculative-config"])
        self.assertEqual(self.cfg["models"]["muse-glimmer-30b-nvfp4"]["tok_s"], 27)
        self.assertFalse(args["trust-remote-code"])
        self.assertEqual(args["max-model-len"], 131072)
        self.assertEqual(args["max-num-seqs"], 8)
        self.assertEqual(args["max-num-batched-tokens"], 2048)
        self.assertEqual(args["reasoning-parser"], "muse_glimmer")
        self.assertEqual(args["tool-call-parser"], "muse_glimmer")
        self.assertEqual(spec, {
            "method": "dflash",
            "model": "meta-models/Muse-Glimmer-30B-assistant",
            "num_speculative_tokens": 15,
        })

    def test_lightning_uses_spark_recipe_and_1m_variant_inherits_it(self):
        profile = mm.merge_model(self.cfg, "nemotron-3.5-lightning-30b-a3b-nvfp4")
        args = profile["vllm_args"]
        spec = json.loads(args["speculative-config"])
        self.assertEqual(
            self.cfg["models"]["nemotron-3.5-lightning-30b-a3b-nvfp4"]["tok_s"], 94)
        self.assertFalse(args["trust-remote-code"])
        self.assertEqual(args["mamba-cache-mode"], "align")
        self.assertEqual(args["mamba-ssm-cache-dtype"], "float32")
        self.assertEqual(args["moe-backend"], "marlin")
        self.assertEqual(args["kv-cache-dtype"], "fp8")
        self.assertEqual(args["max-num-batched-tokens"], 16384)
        self.assertEqual(spec["method"], "dspark")
        self.assertEqual(spec["num_speculative_tokens"], 3)

        long_profile = mm.merge_model(
            self.cfg, "nemotron-3.5-lightning-30b-a3b-nvfp4-1m")
        long_args = long_profile["vllm_args"]
        self.assertEqual(long_args["max-model-len"], 1048576)
        self.assertEqual(long_args["max-num-seqs"], 2)
        self.assertEqual(long_args["speculative-config"], args["speculative-config"])


class Qwen38Bf16ProfileTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(json.dumps(mm.DEFAULT_CONFIG))

    def test_quality_profile_keeps_full_precision_and_vision(self):
        profile = mm.merge_model(self.cfg, "qwen3.8-27b-bf16")
        args = profile["vllm_args"]
        self.assertEqual(profile["model"], "Qwen/Qwen3.8-27B")
        self.assertIn("@sha256:", profile["image"])
        self.assertEqual(args["revision"],
                         "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
        self.assertEqual(args["tokenizer"], "Qwen/Qwen3.8-27B-FP8")
        self.assertEqual(args["dtype"], "bfloat16")
        self.assertEqual(args["max-model-len"], 262144)
        self.assertEqual(args["max-num-seqs"], 2)
        self.assertNotIn("language-model-only", args)
        self.assertNotIn("speculative-config", args)
        self.assertIn("Vision", self.cfg["models"]["qwen3.8-27b-bf16"]["usecase"])

    def test_mtp_profile_is_an_isolated_speed_variant(self):
        profile = mm.merge_model(self.cfg, "qwen3.8-27b-bf16-mtp")
        args = profile["vllm_args"]
        self.assertEqual(args["served-model-name"], "qwen3.8-27b-bf16-mtp")
        self.assertTrue(args["no-enable-prefix-caching"])
        self.assertEqual(json.loads(args["speculative-config"]),
                         {"method": "mtp", "num_speculative_tokens": 2})
        self.assertNotIn("language-model-only", args)


class Qwen38Nvfp4DflashProfileTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(json.dumps(mm.DEFAULT_CONFIG))

    def test_profile_keeps_qualified_pins_and_scheduler(self):
        profile = mm.merge_model(self.cfg, "qwen3.8-27b-nvfp4-vllm-dflash2")
        args = profile["vllm_args"]
        self.assertEqual(
            self.cfg["models"]["qwen3.8-27b-nvfp4-vllm-dflash2"]["tok_s"], 27)
        self.assertEqual(profile["model"], "RadixArk/Qwen3.8-27B-NVFP4")
        self.assertEqual(
            profile["image"],
            "vllm/vllm-openai@sha256:6630695e452bd4d167f3b8bc3bf3151f93977997ff5dc7cd7d6086037f42a052")
        self.assertEqual(args["revision"],
                         "319f741cce68d7914884900c138a1fbb70a42f30")
        self.assertFalse(args["trust-remote-code"])
        self.assertEqual(args["max-model-len"], 262144)
        self.assertEqual(args["max-num-seqs"], 1)
        self.assertEqual(json.loads(args["speculative-config"]), {
            "method": "dflash",
            "model": "incoai/Qwen3.8-27B-DFlash2",
            "revision": "dedf8df68adfb1afeaf7b7480c0a0243108177b4",
            "num_speculative_tokens": 7,
        })


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


class HostBindTests(unittest.TestCase):
    """Loopback is the default bind; LAN exposure is an explicit per-profile opt-in."""

    def setUp(self):
        # Committed source of truth, not the git-ignored sandbox (machine-dependent).
        self.cfg = json.loads(json.dumps(mm.DEFAULT_CONFIG))

    def test_all_profiles_default_to_loopback(self):
        for key in self.cfg["models"]:
            with self.subTest(profile=key):
                merged = mm.merge_model(self.cfg, key)
                self.assertEqual(merged["host"], "127.0.0.1")

    def test_loopback_reaches_the_argv(self):
        m = mm.merge_model(self.cfg, "glm-4.7-flash")
        _, argv, _ = mm.build_run_argv("glm-4.7-flash", m, target=None)
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")

    def test_profile_can_opt_into_lan(self):
        self.cfg["models"]["glm-4.7-flash"]["host"] = "0.0.0.0"
        m = mm.merge_model(self.cfg, "glm-4.7-flash")
        self.assertEqual(m["host"], "0.0.0.0")
        _, argv, _ = mm.build_run_argv("glm-4.7-flash", m, target=None)
        self.assertEqual(argv[argv.index("--host") + 1], "0.0.0.0")


class RemoteHealthTests(unittest.TestCase):
    """`health` on a remote box probes that box's loopback over SSH (curl), since a
    loopback-bound server is invisible on the LAN."""

    def _probe(self, returncode, stdout, stderr=""):
        with mock.patch.object(mm, "run_remote",
                               return_value=SimpleNamespace(returncode=returncode,
                                                            stdout=stdout, stderr=stderr)) as run:
            result = mm.remote_http_models("user@host", 8000)
        command = run.call_args.args[1]
        self.assertIn("exit $rc", command)
        self.assertIn("--noproxy '*'", command)
        return result

    def test_ready_lists_model_ids(self):
        body = json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]})
        self.assertEqual(self._probe(0, body + "\nHTTP_CODE:200"), ("ready", "m1, m2"))

    def test_http_error_is_surfaced(self):
        self.assertEqual(self._probe(0, "boom\nHTTP_CODE:500"), ("error", "HTTP 500"))

    def test_no_listener(self):
        self.assertEqual(self._probe(7, "", "connection refused"),
                         ("starting", "server not listening yet"))

    def test_timeout_reads_as_starting(self):
        self.assertEqual(self._probe(28, "", "timed out"),
                         ("starting", "server not listening yet"))

    def test_missing_curl_is_an_error(self):
        phase, detail = self._probe(127, "", "curl: not found")
        self.assertEqual(phase, "error")
        self.assertIn("curl not found", detail)

    def test_ssh_failure_is_an_error(self):
        phase, detail = self._probe(255, "", "ssh: connection reset")
        self.assertEqual(phase, "error")
        self.assertIn("SSH health probe failed", detail)
        self.assertIn("connection reset", detail)

    def test_other_curl_failure_is_an_error(self):
        phase, detail = self._probe(35, "", "TLS failure")
        self.assertEqual(phase, "error")
        self.assertIn("exit 35", detail)

    def test_malformed_success_is_an_error(self):
        phase, detail = self._probe(0, "not the marked response")
        self.assertEqual(phase, "error")
        self.assertIn("invalid response", detail)


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
        mm._remote_home_cache["u@h"] = "/home/user"
        self.assertEqual(mm.host_path("~/x", target="u@h"), "/home/user/x")
        self.assertEqual(mm.host_path("~/.cache/hf", target="u@h"), "/home/user/.cache/hf")


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
            f.write("dgx-2\tuser@192.0.2.102\n")

    def tearDown(self):
        mm.HOSTS_FILE, mm.list_managed = self._hosts, self._lm

    def test_host_of_recognizes_alias_ip_userhost(self):
        self.assertEqual(mm.host_of("dgx-2"), "dgx-2")               # registered alias
        self.assertEqual(mm.host_of("user@1.2.3.4"), "user@1.2.3.4")  # user@host
        self.assertEqual(mm.host_of("192.0.2.103"), "192.0.2.103")  # bare IP

    def test_host_of_rejects_model_and_container_names(self):
        self.assertIsNone(mm.host_of("qwen3.6-35b-nvfp4"))
        self.assertIsNone(mm.host_of("otools-vllm-glm-4.7-flash"))
        self.assertIsNone(mm.host_of("dgx-9"))       # not a registered alias
        self.assertIsNone(mm.host_of(""))
        self.assertIsNone(mm.host_of(None))

    def test_managed_on_single(self):
        mm.list_managed = lambda **k: [{"Names": "otools-vllm-m1", "Labels": "otools.model=m1"}]
        self.assertEqual(mm._managed_on("user@a"), ("otools-vllm-m1", "m1"))

    def test_managed_on_none_errors(self):
        mm.list_managed = lambda **k: []
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mm._managed_on("user@a")

    def test_managed_on_ambiguous_errors(self):
        mm.list_managed = lambda **k: [{"Names": "a"}, {"Names": "b"}]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mm._managed_on("user@a")


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


class HttpModelsTests(unittest.TestCase):
    def test_refused_connection_is_starting(self):
        # A container that's up but still loading refuses connections; that must read as
        # 'starting' (keep polling), NOT 'error' — otherwise agents thrash. Grab a free
        # port and release it so nothing is listening on it.
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        phase, _ = mm.http_models("127.0.0.1", port, timeout=1.0)
        self.assertEqual(phase, "starting")


class ClusterProfileTests(unittest.TestCase):
    def test_qwen_artifacts_are_first_party_and_immutable(self):
        for profile in mm.CLUSTER_PROFILES.values():
            if profile["backend"] != "trtllm-mpi":
                continue
            self.assertTrue(profile["model"].startswith("nvidia/"))
            self.assertEqual(len(profile["revision"]), 40)
            image, digest = profile["image"].split("@sha256:")
            self.assertEqual(image, "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc8")
            self.assertEqual(len(digest), 64)
            self.assertEqual(profile["runtime_image"],
                             "otools/trtllm-mpi:1.3.0rc8-reviewed")

    def test_deepseek_uses_official_weights_and_c8r_source_pins(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        self.assertEqual(profile["model"], "deepseek-ai/DeepSeek-V4-Flash-0731")
        self.assertEqual(profile["served_model_name"], "deepseek-v4-flash-dspark")
        self.assertEqual(profile["status"], "validated")
        self.assertEqual(profile["revision"], "9e165c30e2704aec5d9d593cce3eebd58bbef1cb")
        self.assertEqual(profile["runtime_lane"], "c8r")
        self.assertEqual(profile["runtime_revision"],
                          "46eb0fcbadf0e4e0be8838b18f6aa85087ed8839")
        self.assertEqual(profile["runtime_overlay_tree"],
                         "94f81f56d9cface7c7719d4cd6d1e0954bce2c8f")
        self.assertEqual(profile["vllm_revision"],
                         "48bada6ea49ad7f3ecbe03128aa76562089c8b00")
        self.assertEqual(profile["deepgemm_revision"],
                         "a6b593d2826719dcf4892609af7b84ee23aaf32a")
        self.assertEqual(profile["b12x_revision"],
                         "7dc6fb8fcc6446ea093537d1657df81985fa5f43")
        self.assertIn("operator accepted", profile["b12x_license_provenance"])
        self.assertEqual(profile["flashinfer_version"], "0.6.16.post3")
        self.assertIn("Reederey87/dgx-spark-2x-deepseek-v4-flash", profile["runtime_repo"])
        self.assertEqual(profile["image"],
                         "otools/vllm-deepseek-v4-flash-0731:c8r-reviewed")
        self.assertEqual(len(profile["image_signature"]), 64)
        self.assertEqual(profile["max_model_len"], 1048576)
        self.assertEqual(profile["max_num_seqs"], 12)
        self.assertTrue(profile["enable_prefix_caching"])

    def test_deepseek_cand7_remains_an_isolated_rollback(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731-cand7"]
        self.assertEqual(profile["runtime_lane"], "cand7")
        self.assertEqual(profile["status"], "rollback")
        self.assertEqual(profile["runtime_revision"],
                         "15f29b7bd91d45a1678b3b8600a56512c36f13f2")
        self.assertEqual(profile["image"],
                         "otools/vllm-deepseek-v4-flash-0731:cand7-reviewed")
        self.assertFalse(profile["enable_prefix_caching"])

    def test_qwen38_flash_next_uses_only_pinned_official_artifacts(self):
        profile = mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"]
        self.assertEqual(profile["backend"], "vllm-mp")
        self.assertEqual(profile["model"], "Qwen/Qwen3.8-Flash-Next-FP8")
        self.assertEqual(profile["revision"],
                         "970c569adaca6b35532111fd6b27351b2baefe50")
        self.assertEqual(profile["model_size"], 185553536918)
        self.assertEqual(
            profile["image"],
            "vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8")
        self.assertEqual(profile["arm64_image_digest"],
                         "sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e")
        self.assertEqual(profile["image_signature"],
                         "483da4d4cdbd8cb6b2094ef3a9b205307b65d8e61120f043db61a4156a750d0b")
        self.assertEqual(profile["vllm_version"], "0.1.dev20073+g8e685d198")
        self.assertEqual(profile["tok_s"], 20)
        self.assertEqual(profile["status"], "validated")

    def test_deepseek_build_uses_only_verified_local_archives(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731-cand7"]
        dockerfile = mm._deepseek_dockerfile(
            profile, "/usr/local/lib/python3.12/dist-packages/vllm")
        self.assertNotIn("github.com/lukealonso/b12x/archive", dockerfile)
        self.assertNotIn("github.com/flashinfer-ai/flashinfer/archive", dockerfile)
        self.assertIn("sha256sum -c SHA256SUMS", dockerfile)
        self.assertIn("BUILD_NVEP=0", dockerfile)
        self.assertIn("/opt/otools/sources/b12x-", dockerfile)
        self.assertIn("/opt/otools/sources/flashinfer-", dockerfile)
        self.assertEqual(len(mm.DEEPSEEK_VLLM_PREIMAGES), 14)

    def test_deepseek_manifest_binds_all_supply_chain_and_model_inputs(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        manifest = mm._deepseek_manifest_inputs(profile)
        self.assertEqual(manifest["build_schema"], 4)
        self.assertEqual(manifest["runtime_lane"], "c8r")
        self.assertEqual(manifest["runtime_overlay_tree"], profile["runtime_overlay_tree"])
        self.assertEqual(manifest["vllm_commit"], profile["vllm_revision"])
        self.assertEqual(manifest["image_signature"], profile["image_signature"])
        self.assertEqual(manifest["model_file_count"], 74)
        self.assertEqual(manifest["model_shards"], 48)
        self.assertEqual(manifest["model_size"], 166898660330)

    def test_deepseek_model_hash_script_contains_no_literal_nul(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        payload = json.dumps({"sha256": "a" * 64, "files": 74,
                              "shards": 48, "size": 166898660330})
        completed = SimpleNamespace(returncode=0, stdout=payload, stderr="")
        with mock.patch.object(mm, "remote_home", return_value="/home/user"), \
                mock.patch.object(mm, "_host_exec", return_value=completed) as execute:
            signature, detail = mm._deepseek_model_signature("user@worker", profile)
        script = execute.call_args.args[1][2]
        self.assertNotIn("\0", script)
        self.assertIn('aggregate.update(b"\\0")', script)
        self.assertEqual(signature["sha256"], "a" * 64)
        self.assertEqual(detail, "")


class ClusterRegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.old = mm.CLUSTERS_FILE, mm.CLUSTER_DATA_DIR, mm.HOSTS_FILE
        mm.CLUSTERS_FILE = os.path.join(self.root, "clusters.json")
        mm.CLUSTER_DATA_DIR = os.path.join(self.root, "data")
        mm.HOSTS_FILE = os.path.join(self.root, "hosts")
        mm.save_hosts([("dgx4", "user@192.0.2.104")])

    def tearDown(self):
        mm.CLUSTERS_FILE, mm.CLUSTER_DATA_DIR, mm.HOSTS_FILE = self.old

    def config(self):
        return {
            "name": "spark",
            "head": "local",
            "worker": "dgx4",
            "fabric": {
                "interfaces": ["enP7s7"],
                "ucx_devices": ["mlx5_0:1"],
                "head_ips": ["10.10.10.1"],
                "worker_ips": ["10.10.10.2"],
                "mtu": 9000,
            },
        }

    def vllm_identity_fixture(self):
        profile = dict(mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"])
        metadata = {
            "Architecture": "arm64",
            "Os": "linux",
            "RepoDigests": [profile["image"]],
            "Id": "sha256:local-store-id",
            "Config": {"Entrypoint": ["vllm", "serve"], "Env": ["A=B"]},
            "RootFS": {"Type": "layers", "Layers": ["sha256:layer"]},
        }
        content = {
            "architecture": metadata["Architecture"],
            "os": metadata["Os"],
            "config": metadata["Config"],
            "rootfs": metadata["RootFS"],
        }
        profile["image_signature"] = mm.hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return profile, metadata

    def test_vllm_image_identity_accepts_pinned_canonical_content(self):
        profile, metadata = self.vllm_identity_fixture()
        responses = [
            SimpleNamespace(returncode=0, stdout=json.dumps([metadata]), stderr=""),
            SimpleNamespace(returncode=0, stdout=profile["vllm_version"] + "\n", stderr=""),
        ]
        with mock.patch.object(mm, "_host_exec", side_effect=responses):
            identity = mm._vllm_image_identity("user@host", profile)
        self.assertEqual(identity, {
            "id": "sha256:local-store-id",
            "content_sha256": profile["image_signature"],
        })

    def test_vllm_image_identity_rejects_missing_pinned_digest(self):
        profile, metadata = self.vllm_identity_fixture()
        metadata["RepoDigests"] = []
        inspected = SimpleNamespace(returncode=0, stdout=json.dumps([metadata]), stderr="")
        with mock.patch.object(mm, "_host_exec", return_value=inspected), \
                self.assertRaisesRegex(RuntimeError, "pinned registry digest"):
            mm._vllm_image_identity("user@host", profile)

    def test_vllm_image_identity_rejects_changed_content(self):
        profile, metadata = self.vllm_identity_fixture()
        metadata["Config"]["Env"].append("CHANGED=1")
        inspected = SimpleNamespace(returncode=0, stdout=json.dumps([metadata]), stderr="")
        with mock.patch.object(mm, "_host_exec", return_value=inspected), \
                self.assertRaisesRegex(RuntimeError, "image content mismatch"):
            mm._vllm_image_identity("user@host", profile)

    def test_roundtrip_and_local_head_resolution(self):
        cfg = self.config()
        stored = dict(cfg)
        stored.pop("name")
        mm.save_clusters({"spark": stored})
        self.assertEqual(mm.cluster_config("spark"), cfg)
        self.assertEqual(mm._cluster_targets(cfg), (None, "user@192.0.2.104"))

    def test_two_local_aliases_are_rejected(self):
        cfg = self.config()
        cfg["worker"] = "localhost"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mm._cluster_targets(cfg)

    def test_cluster_rename_moves_registry_and_cached_state(self):
        cfg = self.config()
        stored = dict(cfg)
        stored.pop("name")
        mm.save_clusters({"spark": stored})
        old_state = mm._cluster_state_dir("spark")
        os.makedirs(old_state)
        with open(os.path.join(old_state, "marker"), "w") as stream:
            stream.write("ok")
        args = SimpleNamespace(name="spark", new_name="studio")
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(mm, "_host_text", return_value=(True, "")), \
                mock.patch.object(mm, "_host_exec", return_value=completed), \
                mock.patch.object(mm, "remote_home", return_value="/home/user"), \
                contextlib.redirect_stdout(io.StringIO()):
            mm.cmd_cluster_rename(args)
        self.assertNotIn("spark", mm.load_clusters())
        self.assertIn("studio", mm.load_clusters())
        self.assertTrue(os.path.isfile(os.path.join(mm._cluster_state_dir("studio"), "marker")))

    def test_cluster_rename_checks_stopped_rank_containers(self):
        cfg = self.config()
        stored = dict(cfg)
        stored.pop("name")
        mm.save_clusters({"spark": stored})
        probe = mock.Mock(return_value=(True, "retained-rank-id"))
        args = SimpleNamespace(name="spark", new_name="studio")
        with mock.patch.object(mm, "_host_text", probe), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mm.cmd_cluster_rename(args)
        self.assertEqual(probe.call_args.args[1], [
            "docker", "ps", "-a", "-q", "--filter", f"label={mm.LABEL_CLUSTER}=spark",
        ])

    def test_certificate_fast_path_uses_inventory_without_full_hash(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        snapshot_identity = {"files": 74, "shards": 48, "size": 166898660330,
                             "metadata_sha256": "b" * 64}
        certificate = {
            "profile_fingerprint": mm._deepseek_profile_fingerprint(profile),
            "machine_id": "machine-a",
            "image_id": "sha256:image",
            "image_signature": profile["image_signature"],
            "model_signature": {"sha256": "a" * 64, "files": 74,
                                "shards": 48, "size": 166898660330},
            "snapshot_identity": snapshot_identity,
        }
        responses = [
            SimpleNamespace(returncode=0, stdout="machine-a\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="sha256:image\n", stderr=""),
        ]
        with mock.patch.object(mm, "_host_exec", side_effect=responses), \
                mock.patch.object(mm, "_deepseek_snapshot_identity",
                                  return_value=snapshot_identity):
            self.assertTrue(mm._deepseek_certificate_valid("user@host", profile, certificate))

    def test_ensure_deepseek_rebuilds_local_manifest_from_host_certificates(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        cfg = self.config()
        cfg["head"] = "user@head"
        cfg["worker"] = "user@worker"
        model_signature = {"sha256": "a" * 64, "files": 74,
                           "shards": 48, "size": 166898660330}
        certificates = {
            "user@head": {"image_id": "sha256:head", "image_signature": profile["image_signature"],
                          "runtime_signature": "runtime", "model_signature": model_signature},
            "user@worker": {"image_id": "sha256:worker", "image_signature": profile["image_signature"],
                            "runtime_signature": "runtime", "model_signature": model_signature},
        }
        image_present = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(mm, "_deepseek_snapshot_identity", return_value={
                "files": 74, "shards": 48, "size": 166898660330}), \
                mock.patch.object(mm, "_host_exec", return_value=image_present), \
                mock.patch.object(mm, "_read_host_json",
                                  side_effect=lambda target, path: certificates[target]), \
                mock.patch.object(mm, "_deepseek_certificate_valid", return_value=True), \
                mock.patch.object(mm, "_download_cluster_snapshot") as download, \
                mock.patch.object(mm, "_certify_deepseek_node") as certify:
            result = mm._ensure_deepseek_deployment(cfg, profile)
        self.assertEqual(set(result), {"head", "worker"})
        download.assert_not_called()
        certify.assert_not_called()
        path = os.path.join(mm._deepseek_build_dir(profile), "manifest.json")
        self.assertTrue(os.path.isfile(path))

    def test_qwen_argv_keeps_rdma_and_socket_devices_separate(self):
        cfg = self.config()
        profile = mm.CLUSTER_PROFILES["qwen3-235b-a22b-fp4"]
        name, argv = mm.build_qwen_cluster_argv(
            "qwen3-235b-a22b-fp4", profile, cfg, "head", None)
        self.assertEqual(name, "otools-vllm-qwen3-235b-a22b-fp4-spark-head")
        self.assertIn("UCX_NET_DEVICES=mlx5_0:1", argv)
        self.assertIn("NCCL_IB_HCA==mlx5_0:1", argv)
        self.assertIn("NCCL_SOCKET_IFNAME==enP7s7", argv)
        self.assertIn("NCCL_NET=IB", argv)
        self.assertIn("OMPI_MCA_btl_tcp_if_include=enP7s7", argv)
        self.assertIn("OMPI_MCA_oob_tcp_if_include=enP7s7", argv)
        self.assertIn("HF_HUB_OFFLINE=1", argv)
        self.assertIn("NCCL_IB_ADDR_FAMILY=AF_INET", argv)
        self.assertIn("SSH_LISTEN_ADDRESS=10.10.10.1", argv)
        self.assertNotIn("UCX_NET_DEVICES=enP7s7", argv)
        self.assertIn("otools.cluster=spark", argv)
        self.assertIn(profile["runtime_image"], argv)

    def test_vllm_mp_argv_is_pinned_conservative_and_role_aware(self):
        cfg = self.config()
        profile = mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"]
        with mock.patch.object(mm, "remote_home", return_value="/home/user"):
            _, head = mm.build_vllm_cluster_argv(
                "qwen3.8-flash-next-fp8", profile, cfg, "head", None)
            _, worker = mm.build_vllm_cluster_argv(
                "qwen3.8-flash-next-fp8", profile, cfg, "worker", "user@worker")
        self.assertIn(profile["image"], head)
        self.assertIn(mm._vllm_snapshot_path(profile), head)
        self.assertNotIn("--trust-remote-code", head)
        self.assertNotIn("--speculative-config", head)
        self.assertIn("VLLM_USE_DEEP_GEMM=0", head)
        self.assertIn("NCCL_NET=IB", head)
        self.assertIn("NCCL_IB_HCA==mlx5_0:1", head)
        self.assertIn("NCCL_SOCKET_IFNAME==enP7s7", head)
        self.assertIn("GLOO_SOCKET_IFNAME=enP7s7", head)
        self.assertIn("--enable-expert-parallel", head)
        self.assertIn("--no-enable-prefix-caching", head)
        self.assertIn("--no-async-scheduling", head)
        self.assertIn("--no-enable-flashinfer-autotune", head)
        self.assertIn("--enforce-eager", head)
        self.assertIn("--enable-log-requests", head)
        self.assertEqual(head[head.index("--tool-call-parser") + 1], "qwen3_xml")
        self.assertEqual(head[head.index("--reasoning-parser") + 1], "qwen3")
        self.assertEqual(head[head.index("--node-rank") + 1], "0")
        self.assertEqual(worker[worker.index("--node-rank") + 1], "1")
        self.assertNotIn("--headless", head)
        self.assertIn("--headless", worker)
        self.assertIn("--rm", head)
        with mock.patch.object(mm, "remote_home", return_value="/home/user"):
            _, kept = mm.build_vllm_cluster_argv(
                "qwen3.8-flash-next-fp8", profile, cfg, "head", None, keep=True)
        self.assertNotIn("--rm", kept)

    def test_vllm_mp_warmups_cover_reasoning_tools_vision_and_streaming(self):
        profile = mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"]
        requests = mm._vllm_warmup_requests(profile)
        self.assertEqual([label for label, _, _ in requests],
                         ["direct chat", "reasoning", "tool parser", "vision", "streaming"])
        payloads = {label: payload for label, _, payload in requests}
        self.assertFalse(payloads["direct chat"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payloads["reasoning"]["reasoning_effort"], "low")
        image = payloads["vision"]["messages"][0]["content"][0]["image_url"]["url"]
        self.assertTrue(image.startswith("data:image/png;base64,"))
        self.assertTrue(payloads["streaming"]["stream"])

    def test_vllm_warmup_rejects_non_object_tool_arguments(self):
        profile = mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"]
        for raw_arguments in ("[]", "null"):
            body = json.dumps({"choices": [{"message": {"tool_calls": [{
                "function": {"name": "get_weather", "arguments": raw_arguments},
            }]}}]})
            with self.subTest(arguments=raw_arguments), \
                    mock.patch.object(mm, "_vllm_warmup_requests", return_value=[
                        ("tool parser", 10, {})]), \
                    mock.patch.object(mm, "_host_text", return_value=(True, body)), \
                    mock.patch.object(mm.time, "monotonic", return_value=0), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, "invalid arguments"):
                mm._warm_vllm_cluster(None, profile, deadline=10)

    def test_vllm_warmup_caps_curl_at_launch_deadline(self):
        profile = mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"]
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]})
        with mock.patch.object(mm, "_vllm_warmup_requests", return_value=[
                ("direct chat", 300, {})]), \
                mock.patch.object(mm, "_host_text", return_value=(True, body)) as request, \
                mock.patch.object(mm.time, "monotonic", side_effect=[10.0, 10.3]), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "deadline exhausted during direct chat"):
            mm._warm_vllm_cluster(None, profile, deadline=10.2)
        argv = request.call_args.args[1]
        self.assertLessEqual(float(argv[argv.index("--max-time") + 1]), 0.2)

    def test_vllm_warmup_rejects_finish_only_stream(self):
        profile = mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"]
        body = ("data: " + json.dumps({"choices": [{
            "delta": {}, "finish_reason": "stop",
        }]}) + "\n\ndata: [DONE]\n")
        with mock.patch.object(mm, "_vllm_warmup_requests", return_value=[
                ("streaming", 10, {"stream": True})]), \
                mock.patch.object(mm, "_host_text", return_value=(True, body)), \
                mock.patch.object(mm.time, "monotonic", return_value=0), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "incomplete stream"):
            mm._warm_vllm_cluster(None, profile, deadline=10)

    def test_vllm_launch_rejects_existing_rank_before_verification(self):
        cfg = self.config()
        profile = mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"]
        existing = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        stderr = io.StringIO()
        with mock.patch.object(mm, "cluster_preflight", return_value=(True, {})), \
                mock.patch.object(mm, "_ensure_cluster_idle"), \
                mock.patch.object(mm, "remote_home", return_value="/home/user"), \
                mock.patch.object(mm, "_host_exec", return_value=existing) as execute, \
                mock.patch.object(mm, "_vllm_image_identity") as verify, \
                mock.patch.object(mm, "_drop_caches") as drop, \
                contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            mm._launch_vllm_cluster("qwen3.8-flash-next-fp8", profile, cfg)
        commands = [call.args[1] for call in execute.call_args_list]
        self.assertTrue(commands)
        self.assertTrue(all(command[:3] == ["docker", "container", "inspect"]
                            for command in commands))
        self.assertIn("cluster stop spark -y", stderr.getvalue())
        verify.assert_not_called()
        drop.assert_not_called()

    def test_vllm_keep_failure_stops_without_removing_ranks(self):
        cfg = self.config()
        profile = mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"]
        commands = []

        def execute(target, argv, capture=False, check=False, tty=False):
            commands.append(argv)
            if argv[:3] == ["docker", "container", "inspect"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="not found")
            if argv[:3] == ["docker", "inspect", "--format"]:
                return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
            if argv[:2] == ["docker", "logs"]:
                return SimpleNamespace(returncode=0, stdout="NCCL NET/IB\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        stderr = io.StringIO()
        with mock.patch.object(mm, "cluster_preflight", return_value=(True, {})), \
                mock.patch.object(mm, "_ensure_cluster_idle"), \
                mock.patch.object(mm, "remote_home", return_value="/home/user"), \
                mock.patch.object(mm, "_host_exec", side_effect=execute), \
                mock.patch.object(mm, "_vllm_image_identity", return_value={
                    "id": "sha256:image", "content_sha256": "same"}), \
                mock.patch.object(mm, "_drop_caches"), \
                mock.patch.object(mm, "_host_text", return_value=(True, "ok")), \
                mock.patch.object(mm, "_warm_vllm_cluster",
                                  side_effect=RuntimeError("malformed warmup")), \
                mock.patch.object(mm.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            mm._launch_vllm_cluster(
                "qwen3.8-flash-next-fp8", profile, cfg, startup_timeout=10, keep=True)
        stopped = {command[-1] for command in commands if command[:2] == ["docker", "stop"]}
        self.assertEqual(stopped, {
            "otools-vllm-qwen3.8-flash-next-fp8-spark-head",
            "otools-vllm-qwen3.8-flash-next-fp8-spark-worker",
        })
        self.assertFalse(any(command[:3] == ["docker", "rm", "-f"] for command in commands))
        self.assertIn("Retained rank containers", stderr.getvalue())
        self.assertIn("cluster stop spark -y", stderr.getvalue())

    def test_vllm_default_cleanup_accepts_auto_removed_rank(self):
        cfg = self.config()
        profile = mm.CLUSTER_PROFILES["qwen3.8-flash-next-fp8"]
        commands = []

        def execute(target, argv, capture=False, check=False, tty=False):
            commands.append(argv)
            name = argv[-1]
            missing = f"Error response from daemon: No such container: {name}\n"
            if argv[:3] == ["docker", "container", "inspect"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="not found")
            if argv[:3] == ["docker", "inspect", "--format"]:
                return SimpleNamespace(returncode=1, stdout="", stderr=missing)
            if argv[:4] == ["docker", "logs", "--tail", "160"]:
                return SimpleNamespace(returncode=1, stdout="", stderr=missing)
            if argv[:3] == ["docker", "rm", "-f"]:
                return SimpleNamespace(returncode=1, stdout="", stderr=missing)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        stderr = io.StringIO()
        with mock.patch.object(mm, "cluster_preflight", return_value=(True, {})), \
                mock.patch.object(mm, "_ensure_cluster_idle"), \
                mock.patch.object(mm, "remote_home", return_value="/home/user"), \
                mock.patch.object(mm, "_host_exec", side_effect=execute), \
                mock.patch.object(mm, "_vllm_image_identity", return_value={
                    "id": "sha256:image", "content_sha256": "same"}), \
                mock.patch.object(mm, "_drop_caches"), \
                mock.patch.object(mm.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            mm._launch_vllm_cluster(
                "qwen3.8-flash-next-fp8", profile, cfg, startup_timeout=10)
        removed = [command for command in commands if command[:3] == ["docker", "rm", "-f"]]
        self.assertEqual(removed, [[
            "docker", "rm", "-f", "otools-vllm-qwen3.8-flash-next-fp8-spark-worker",
        ]])
        self.assertNotIn("cleanup could not remove", stderr.getvalue())

    def test_cluster_launch_dispatches_vllm_without_deepseek_fallback(self):
        args = SimpleNamespace(name="spark", profile="qwen3.8-flash-next-fp8",
                               dry_run=True, startup_timeout=1800)
        with mock.patch.object(mm, "cluster_config", return_value=self.config()), \
                mock.patch.object(mm, "_launch_vllm_cluster") as vllm_launch, \
                mock.patch.object(mm, "_launch_deepseek_cluster") as deepseek_launch:
            mm.cmd_cluster_launch(args)
        vllm_launch.assert_called_once()
        deepseek_launch.assert_not_called()

    def test_qwen_warmups_cover_chat_tools_and_streaming(self):
        profile = mm.CLUSTER_PROFILES["qwen3-235b-a22b-fp4"]
        requests = mm._qwen_warmup_requests(profile)
        self.assertEqual([label for label, _, _ in requests],
                         ["plain chat", "tool parser", "streaming"])
        payloads = {label: payload for label, _, payload in requests}
        self.assertEqual(payloads["tool parser"]["model"], profile["model"])
        self.assertTrue(payloads["streaming"]["stream"])

    def test_qwen_warmup_rejects_malformed_tool_arguments(self):
        profile = mm.CLUSTER_PROFILES["qwen3-235b-a22b-thinking-2507-nvfp4"]
        body = json.dumps({"choices": [{"message": {"tool_calls": [{
            "function": {"name": "get_weather", "arguments": "<tool_call>{}"},
        }]}}]})
        with mock.patch.object(mm, "_qwen_warmup_requests", return_value=[
                ("tool parser", 10, {})]), \
                mock.patch.object(mm, "_host_text", return_value=(True, body)), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            mm._warm_qwen_cluster(None, profile)

    def test_qwen_cluster_health_checks_generation(self):
        args = SimpleNamespace(name="spark", profile="qwen3-235b-a22b-fp4")
        cfg = self.config()
        with mock.patch.object(mm, "cluster_config", return_value=cfg), \
                mock.patch.object(mm, "_host_text", return_value=(True, "ok")) as health, \
                contextlib.redirect_stdout(io.StringIO()):
            mm.cmd_cluster_health(args)
        self.assertEqual(health.call_args.args[1][-1],
                         "http://127.0.0.1:8355/health_generate")

    def test_qwen_profiles_use_conservative_fp4_startup(self):
        base = mm._qwen_extra_config(mm.CLUSTER_PROFILES["qwen3-235b-a22b-fp4"])
        instruct = mm._qwen_extra_config(
            mm.CLUSTER_PROFILES["qwen3-235b-a22b-instruct-2507-nvfp4"])
        self.assertIn("allowed_backends:\n    - cublaslt", base)
        self.assertIn("enable_autotuner: false", base)
        self.assertIn("cuda_graph_config: null", base)
        self.assertIn("free_gpu_memory_fraction: 0.75", base)
        self.assertIn("allowed_backends:\n    - cublaslt", instruct)
        self.assertIn("enable_autotuner: false", instruct)
        self.assertIn("cuda_graph_config: null", instruct)
        self.assertIn("free_gpu_memory_fraction: 0.75", instruct)
        thinking = mm._qwen_extra_config(
            mm.CLUSTER_PROFILES["qwen3-235b-a22b-thinking-2507-nvfp4"])
        self.assertIn("disable_flashinfer_sampling: true", thinking)
        self.assertIn("cuda_graph_config: null", thinking)
        self.assertIn("allowed_backends:\n    - cublaslt", thinking)
        self.assertIn("enable_autotuner: false", thinking)
        self.assertIn("free_gpu_memory_fraction: 0.95", thinking)
        self.assertNotIn("disable_flashinfer_sampling: true", instruct)
        self.assertEqual(
            mm.CLUSTER_PROFILES["qwen3-235b-a22b-thinking-2507-nvfp4"]["max_model_len"],
            48064)
        self.assertEqual(
            mm.CLUSTER_PROFILES["qwen3-235b-a22b-thinking-2507-nvfp4"]["max_num_tokens"],
            16384)
        self.assertEqual(
            mm.CLUSTER_PROFILES["qwen3-235b-a22b-thinking-2507-nvfp4"]["max_batch_size"],
            4)
        self.assertEqual(
            mm.CLUSTER_PROFILES["qwen3-235b-a22b-instruct-2507-nvfp4"]["max_num_tokens"],
            8192)

    def test_qwen_profiles_share_the_reviewed_rc8_manifest(self):
        root = "/tmp/qwen-runtime"
        base = mm.CLUSTER_PROFILES["qwen3-235b-a22b-fp4"]
        instruct = mm.CLUSTER_PROFILES["qwen3-235b-a22b-instruct-2507-nvfp4"]
        thinking = mm.CLUSTER_PROFILES["qwen3-235b-a22b-thinking-2507-nvfp4"]
        self.assertEqual(mm._qwen_manifest_path(root, base),
                         mm._qwen_manifest_path(root, instruct))
        self.assertEqual(mm._qwen_manifest_path(root, base),
                         mm._qwen_manifest_path(root, thinking))

    def test_deepseek_argv_is_offline_pinned_and_role_aware(self):
        cfg = self.config()
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        _, head = mm.build_deepseek_cluster_argv(
            "deepseek-v4-flash-0731", profile, cfg, "head", None)
        _, worker = mm.build_deepseek_cluster_argv(
            "deepseek-v4-flash-0731", profile, cfg, "worker", None)
        self.assertIn(mm._deepseek_snapshot_path(profile), head)
        self.assertIn("NCCL_NET=IB", head)
        self.assertIn("NCCL_IB_HCA==mlx5_0:1", head)
        self.assertIn("NCCL_SOCKET_IFNAME==enP7s7", head)
        self.assertIn("HF_HUB_OFFLINE=1", head)
        self.assertIn("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600", head)
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER=1", head)
        self.assertIn("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256", head)
        self.assertIn("TILELANG_CLEANUP_TEMP_FILES=1", head)
        self.assertNotIn("--trust-remote-code", head)
        spec = json.loads(head[head.index("--speculative-config") + 1])
        self.assertEqual(spec, {"method": "dspark", "num_speculative_tokens": 2,
                                "draft_sample_method": "probabilistic"})
        self.assertEqual(head[head.index("--served-model-name") + 1],
                         profile["served_model_name"])
        self.assertEqual(head[head.index("--kv-cache-memory-bytes") + 1], "21316272128")
        self.assertEqual(head[head.index("--max-model-len") + 1], "1048576")
        self.assertEqual(head[head.index("--max-num-seqs") + 1], "12")
        self.assertEqual(head[head.index("--max-cudagraph-capture-size") + 1], "72")
        self.assertEqual(head[head.index("--gpu-memory-utilization") + 1], "0.85")
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", head)
        self.assertIn("--enable-prefix-caching", head)
        self.assertIn("--enable-prompt-tokens-details", head)
        self.assertEqual(head[head.index("--tokenizer-mode") + 1], "deepseek_v4")
        self.assertEqual(head[head.index("--tool-call-parser") + 1], "deepseek_v4")
        self.assertEqual(head[head.index("--reasoning-parser") + 1], "deepseek_v4")
        reasoning = json.loads(head[head.index("--reasoning-config") + 1])
        self.assertEqual(reasoning, {
            "reasoning_parser": "deepseek_v4",
            "reasoning_start_str": "<think>",
            "reasoning_end_str": "</think>",
        })
        defaults = json.loads(head[head.index("--default-chat-template-kwargs") + 1])
        self.assertEqual(defaults, {"thinking": True, "reasoning_effort": "high"})
        self.assertIn("VLLM_CACHE_ROOT=/cache/runtime/vllm-cache-c8r", head)
        self.assertIn("NCCL_IB_ADDR_FAMILY=AF_INET", head)
        self.assertIn("--distributed-timeout-seconds", head)
        self.assertNotIn("--enforce-eager", head)
        self.assertEqual(head[head.index("--node-rank") + 1], "0")
        self.assertEqual(worker[worker.index("--node-rank") + 1], "1")
        self.assertNotIn("--headless", head)
        self.assertIn("--headless", worker)

        rollback = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731-cand7"]
        _, rollback_head = mm.build_deepseek_cluster_argv(
            "deepseek-v4-flash-0731-cand7", rollback, cfg, "head", None)
        self.assertIn("VLLM_CACHE_ROOT=/cache/runtime/vllm-cache-cand7", rollback_head)
        self.assertNotIn("--enable-prefix-caching", rollback_head)

    def test_deepseek_warmups_cover_agent_paths(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        requests = mm._deepseek_warmup_requests(profile)
        labels = [label for label, _, _ in requests]
        self.assertEqual(labels, ["non-thinking decode", "think high", "think max",
                                  "tool parser", "tool result", "long prefill",
                                  "sampled streaming", "long agent streaming"])
        payloads = {label: payload for label, _, payload in requests}
        timeouts = {label: timeout for label, timeout, _ in requests}
        self.assertEqual(payloads["non-thinking decode"]["chat_template_kwargs"],
                         {"thinking": False})
        self.assertEqual(payloads["think high"]["chat_template_kwargs"],
                         {"thinking": True, "reasoning_effort": "high"})
        self.assertEqual(payloads["think max"]["chat_template_kwargs"],
                         {"thinking": True, "reasoning_effort": "max"})
        self.assertGreater(len(payloads["long prefill"]["messages"][0]["content"]), 30000)
        self.assertGreaterEqual(timeouts["long prefill"], 600)
        self.assertTrue(payloads["sampled streaming"]["stream"])
        self.assertTrue(payloads["long agent streaming"]["stream"])
        self.assertEqual(payloads["tool result"]["messages"][2]["role"], "tool")

    def test_deepseek_warmup_rejects_incomplete_stream(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        response = SimpleNamespace(returncode=0, stdout="data: {}\n", stderr="")
        with mock.patch.object(mm, "_deepseek_warmup_requests", return_value=[
                ("sampled streaming", 10, {"stream": True})]), \
                mock.patch.object(mm, "_host_exec", return_value=response), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            mm._warm_deepseek_cluster(None, profile)

    def test_deepseek_warmup_surfaces_stream_error_detail(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        response = SimpleNamespace(
            returncode=0,
            stdout='data: {"error":{"message":"worker failed"}}\n\ndata: [DONE]\n',
            stderr="")
        with mock.patch.object(mm, "_deepseek_warmup_requests", return_value=[
                ("sampled streaming", 10, {"stream": True})]), \
                mock.patch.object(mm, "_host_exec", return_value=response), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "worker failed"):
            mm._warm_deepseek_cluster(None, profile)

    def test_deepseek_thinking_warmup_requires_reasoning_and_final_content(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        response = SimpleNamespace(returncode=0, stdout=json.dumps({
            "choices": [{"message": {"content": "4"}}]}), stderr="")
        with mock.patch.object(mm, "_deepseek_warmup_requests", return_value=[
                ("think high", 10, {})]), \
                mock.patch.object(mm, "_host_exec", return_value=response), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "emitted no reasoning"):
            mm._warm_deepseek_cluster(None, profile)

    def test_deepseek_non_thinking_warmup_rejects_reasoning(self):
        profile = mm.CLUSTER_PROFILES["deepseek-v4-flash-0731"]
        response = SimpleNamespace(returncode=0, stdout=json.dumps({
            "choices": [{"message": {"content": "ok", "reasoning": "unexpected"}}]}),
            stderr="")
        with mock.patch.object(mm, "_deepseek_warmup_requests", return_value=[
                ("non-thinking decode", 10, {})]), \
                mock.patch.object(mm, "_host_exec", return_value=response), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "unexpectedly emitted reasoning"):
            mm._warm_deepseek_cluster(None, profile)

    def test_local_runtime_sync_never_removes_its_source(self):
        cfg = self.config()
        local_root = os.path.join(self.root, "staging", "qwen-runtime")
        os.makedirs(local_root)
        with open(os.path.join(local_root, "Dockerfile"), "w") as f:
            f.write("FROM scratch\n")
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(mm, "remote_home", return_value="/home/user"), \
                mock.patch.object(mm, "_host_exec", return_value=completed) as execute, \
                mock.patch.object(mm.subprocess, "run", return_value=completed) as copy:
            mm._sync_cluster_files(cfg, local_root, "qwen-runtime")
        self.assertTrue(os.path.isfile(os.path.join(local_root, "Dockerfile")))
        copied = os.path.join(mm.CLUSTER_DATA_DIR, "spark", "qwen-runtime", "Dockerfile")
        self.assertTrue(os.path.isfile(copied))
        self.assertTrue(all(call.args[0] == "user@192.0.2.104" for call in execute.call_args_list))
        self.assertEqual(copy.call_args.args[0][-1],
                         "user@192.0.2.104:/home/user/.local/share/otools/clusters/spark/qwen-runtime")


class ClusterSafetyTests(unittest.TestCase):
    def config(self):
        return {
            "name": "spark", "head": "local", "worker": "user@worker",
            "fabric": {"interfaces": ["enP7s7"], "ucx_devices": ["mlx5_0:1"],
                       "head_ips": ["10.10.10.1"], "worker_ips": ["10.10.10.2"],
                       "mtu": 9000},
        }

    def test_preflight_rejects_management_route_even_if_ping_succeeds(self):
        calls = []

        def host_text(target, args):
            calls.append(args)
            if args[:2] == ["hostnamectl", "--static"]:
                return True, "head" if target is None else "worker"
            if args[:2] == ["uname", "-m"]:
                return True, "aarch64"
            if args[0] == "nvidia-smi":
                return True, "NVIDIA GB10, 580.1"
            if args[:2] == ["docker", "version"]:
                return True, "29.0"
            if args[0] == "df":
                return True, "Avail\n500G"
            if args[0] == "ibdev2netdev":
                return True, "mlx5_0 port 1 ==> enP7s7 (Up)"
            if args[:3] == ["ip", "-o", "link"]:
                return True, "enP7s7: <UP,LOWER_UP> mtu 9000"
            if args[:4] == ["ip", "-o", "-4", "addr"]:
                ip = "10.10.10.1" if target is None else "10.10.10.2"
                return True, f" enP7s7    inet {ip}/24"
            if args[0] == "cat":
                return True, "9000"
            if args[:3] == ["ip", "route", "get"]:
                return True, f"{args[3]} via 192.168.1.1 dev wlP9s9"
            if args[0] == "ping":
                return True, "ok"
            raise AssertionError(args)

        with mock.patch.object(mm, "_host_text", side_effect=host_text), \
                mock.patch.object(mm, "remote_home", return_value="/home/user"):
            ok, results = mm.cluster_preflight(self.config(), require_fabric=True, quiet=True)
        self.assertFalse(ok)
        self.assertTrue(any(name.endswith("-route") and not passed
                            for name, passed, _ in results))
        pings = [args for args in calls if args[0] == "ping"]
        self.assertTrue(all(args[1:3] == ["-I", "enP7s7"] for args in pings))

    def test_preflight_route_match_does_not_accept_interface_prefix(self):
        route = "10.10.10.2 dev enP7s70 src 10.10.10.1"
        self.assertFalse(mm._route_uses_interface(route, "enP7s7"))
        self.assertTrue(mm._route_uses_interface(route, "enP7s70"))

    def test_busy_detection_includes_unmanaged_gpu_containers(self):
        replies = [
            (True, "abc\ndef"),
            (True, "/manual-train\t[{\"Capabilities\":[[\"gpu\"]]}]\t[]\n"
                   "/web\t[]\t[]"),
        ]
        with mock.patch.object(mm, "_host_text", side_effect=replies):
            self.assertEqual(mm._cluster_busy(None), ["manual-train"])

    def test_busy_detection_includes_native_gpu_processes(self):
        replies = [(True, ""), (True, "1234, native-trainer")]
        with mock.patch.object(mm, "_host_text", side_effect=replies):
            self.assertEqual(mm._cluster_busy(None), ["host-gpu:1234:native-trainer"])

    def test_prepare_refuses_heavy_work_on_busy_nodes(self):
        args = SimpleNamespace(name="spark", profile="qwen3-235b-a22b-fp4",
                               build=True, weights=True, allow_busy=False, dry_run=True)
        with mock.patch.object(mm, "cluster_config", return_value=self.config()), \
                mock.patch.object(mm, "cluster_preflight", return_value=(True, [])), \
                mock.patch.object(mm, "_cluster_busy", side_effect=[["model-a"], ["model-b"]]), \
                mock.patch.object(mm, "_prepare_qwen") as prepare, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mm.cmd_cluster_prepare(args)
        prepare.assert_not_called()

    def test_prepare_refuses_unknown_busy_state(self):
        args = SimpleNamespace(name="spark", profile="qwen3-235b-a22b-fp4",
                               build=False, weights=False, allow_busy=False, dry_run=True)
        with mock.patch.object(mm, "cluster_config", return_value=self.config()), \
                mock.patch.object(mm, "cluster_preflight", return_value=(True, [])), \
                mock.patch.object(mm, "_cluster_busy", side_effect=[[], None]), \
                mock.patch.object(mm, "_prepare_qwen") as prepare, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mm.cmd_cluster_prepare(args)
        prepare.assert_not_called()

    def test_stop_fails_when_a_node_is_unreachable(self):
        args = SimpleNamespace(name="spark", yes=True)
        with mock.patch.object(mm, "cluster_config", return_value=self.config()), \
                mock.patch.object(mm, "list_managed", side_effect=[[], None]), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mm.cmd_cluster_stop(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)

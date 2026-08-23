#!/usr/bin/env python3
"""Offline tests for the Intel Arc B70 bring-up utility."""

import os
import tempfile
import types
import unittest
from unittest import mock

from utils import arc_b70_setup as arc


def option_value(argv, option):
    return argv[argv.index(option) + 1]


class ArcB70SetupTests(unittest.TestCase):
    def test_parse_supported_os_release(self):
        release = arc.parse_os_release(
            'ID=ubuntu\nVERSION_ID="26.04"\nVERSION_CODENAME=resolute\n'
        )
        self.assertEqual(release["ID"], "ubuntu")
        self.assertEqual(release["VERSION_ID"], "26.04")
        self.assertEqual(release["VERSION_CODENAME"], "resolute")

    def test_parse_b70_pci_address_and_driver(self):
        output = """0000:03:00.0 Display controller [0380]: Intel Corporation Device [8086:e223]
        Kernel driver in use: xe
        Kernel modules: xe
"""
        self.assertEqual(arc.parse_b70_lspci(output), ("0000:03:00.0", "xe"))

    def test_production_profile_is_full_context_q4_with_q8_kv(self):
        argv = arc.server_argv("~/llama.cpp", "~/models", "production")
        self.assertEqual(
            option_value(argv, "--model"),
            os.path.expanduser("~/models/Qwen3.8-27B-Q4_K_S.gguf"),
        )
        self.assertEqual(option_value(argv, "--ctx-size"), "262144")
        self.assertEqual(option_value(argv, "--parallel"), "1")
        self.assertEqual(option_value(argv, "--cache-type-k"), "q8_0")
        self.assertEqual(option_value(argv, "--cache-type-v"), "q8_0")
        self.assertEqual(option_value(argv, "--alias"), arc.SERVED_MODEL)
        self.assertEqual(option_value(argv, "--host"), "127.0.0.1")
        self.assertIn("--no-mmproj", argv)
        self.assertEqual(option_value(argv, "--gpu-layers"), "all")

    def test_q5_is_explicit_experiment(self):
        argv = arc.server_argv("~/llama.cpp", "~/models", "q5-experiment")
        self.assertTrue(option_value(argv, "--model").endswith("Qwen3.8-27B-Q5_K_M.gguf"))
        self.assertIn("may exceed practical VRAM", arc.PROFILES["q5-experiment"]["description"])

    def test_setup_plan_pins_commit_and_disables_host_fallback(self):
        source = os.path.abspath("/tmp/llama.cpp")
        commands = arc.setup_commands(source)
        flattened = [item for command in commands for item in command]
        self.assertIn(arc.LLAMA_TAG, flattened)
        self.assertIn(arc.LLAMA_COMMIT, flattened)
        self.assertIn("-DGGML_SYCL_HOST_MEM_FALLBACK=OFF", flattened)
        self.assertIn(
            f"-DCMAKE_CXX_COMPILER={arc.ONEAPI_COMPILER}/bin/icpx", flattened,
        )

    def test_systemd_unit_uses_loopback_production_profile(self):
        unit = arc.systemd_unit(
            "/home/test/llama.cpp", "/home/test/models", "production", "127.0.0.1", 8000,
            script_path="/repo/utils/arc_b70_setup.py", python_path="/usr/bin/python3",
        )
        self.assertIn("--profile production --host 127.0.0.1 --port 8000", unit)
        self.assertIn("Environment=ONEAPI_DEVICE_SELECTOR=level_zero:0", unit)
        self.assertIn("--model-dir /home/test/models", unit)
        self.assertIn("StartLimitBurst=3", unit)
        self.assertNotIn("network-online.target", unit)
        self.assertNotIn("0.0.0.0", unit)

    def test_model_pin_matches_exact_revision_and_hash(self):
        info = arc.MODEL_FILES["Q4_K_S"]
        self.assertEqual(arc.MODEL_REVISION, "f0eec4a4bb4975114a030d048952d83c0a53c034")
        self.assertEqual(info["size"], 16_713_148_000)
        self.assertEqual(
            info["sha256"],
            "9282674b002aac8d9d5eda7f53f5114d7fc91725f5a6962a03738571afb2218d",
        )

    def test_verify_model_rejects_wrong_content(self):
        with tempfile.NamedTemporaryFile() as model, \
                mock.patch.dict(arc.MODEL_FILES, {"TEST": {
                    "filename": "test.gguf", "size": 3, "sha256": "0" * 64,
                }}):
            model.write(b"bad")
            model.flush()
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                arc.verify_model(model.name, "TEST")

    def test_conflicting_intel_source_is_rejected(self):
        with tempfile.NamedTemporaryFile("w") as source:
            source.write("deb https://repositories.intel.com/gpu/ubuntu resolute client\n")
            source.flush()
            self.assertEqual(len(arc.conflicting_intel_sources("resolute", [source.name])), 1)

    def test_exact_omix_source_is_not_a_conflict(self):
        with tempfile.NamedTemporaryFile("w") as source:
            source.write(arc.omix_source("resolute"))
            source.flush()
            self.assertEqual(arc.conflicting_intel_sources("resolute", [source.name]), [])

    def test_wrong_codename_omix_source_is_a_conflict(self):
        with tempfile.NamedTemporaryFile("w") as source:
            source.write(arc.omix_source("noble"))
            source.flush()
            self.assertEqual(len(arc.conflicting_intel_sources("resolute", [source.name])), 1)

    def test_build_refuses_dirty_existing_checkout(self):
        with tempfile.TemporaryDirectory() as source:
            os.mkdir(os.path.join(source, ".git"))
            responses = [
                types.SimpleNamespace(stdout=arc.LLAMA_REPO + "\n"),
                types.SimpleNamespace(stdout=" M ggml/src/file.cpp\n"),
            ]
            with mock.patch.object(arc, "run", side_effect=responses):
                with self.assertRaisesRegex(RuntimeError, "modified llama.cpp checkout"):
                    arc.build_llama(source)

    def test_runtime_checks_require_pinned_version_and_sycl_device(self):
        with tempfile.NamedTemporaryFile() as server, \
                mock.patch.object(arc, "oneapi_environment", return_value={}), \
                mock.patch.object(arc, "command_output", side_effect=[
                    ("[level_zero:gpu] Intel Arc", ""),
                    (f"version {arc.LLAMA_TAG} ({arc.LLAMA_COMMIT[:8]})", ""),
                    ("SYCL0: Intel Arc Pro B70 (32768 MiB, 32000 MiB free)", ""),
                ]):
            checks = arc.runtime_checks(server.name)
        self.assertTrue(all(check["status"] == "PASS" for check in checks))

    def test_intel_key_bundle_rejects_unreviewed_primary_key(self):
        key_info = "\n".join([
            "pub:::::::::",
            "fpr:::::::::4E9EFCDEF82800256C1E7C64B02DB9BD8C321DCB:",
            "sub:::::::::",
            "fpr:::::::::FEFCA31BD4A679861EB7080C28DA432DAAC8BAEA:",
            "pub:::::::::",
            "fpr:::::::::AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA:",
        ])
        self.assertNotEqual(arc.primary_fingerprints(key_info), arc.INTEL_KEY_FINGERPRINTS)

    def test_oneapi_environment_uses_canonical_setvars_output(self):
        result = types.SimpleNamespace(
            returncode=0,
            stdout=b"PATH=/oneapi/bin:/usr/bin\0MKLROOT=/oneapi/mkl\0",
            stderr=b"",
        )
        with mock.patch.object(arc.os.path, "isfile", return_value=True), \
                mock.patch.object(arc.subprocess, "run", return_value=result) as run:
            env = arc.oneapi_environment()
        self.assertEqual(env["MKLROOT"], "/oneapi/mkl")
        self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:0")
        self.assertIn("source", run.call_args.args[0][4])


if __name__ == "__main__":
    unittest.main(verbosity=2)

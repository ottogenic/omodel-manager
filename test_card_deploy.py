#!/usr/bin/env python3
"""Offline contract tests for the checked-in B70 deployment helper."""

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from utils.card import deploy_b70_vllm as deploy


class CardDeployCommandTests(unittest.TestCase):
    def test_exact_served_id_and_qualified_vllm_command(self):
        self.assertEqual(deploy.MODEL_ID, "qwen3.8-27b-gptq-int4-b70")
        index = deploy.VLLM_ARGS.index("--served-model-name")
        self.assertEqual(deploy.VLLM_ARGS[index + 1], deploy.MODEL_ID)
        self.assertIn("/bench/utils/card/launch_vllm_xpu.py", deploy.VLLM_ARGS)

    def test_both_containers_have_standard_ownership_labels(self):
        for command, role in (
            (deploy.model_create_command(Path("/repo"), Path("/model"), 990),
             "model-server"),
            (deploy.proxy_create_command(Path("/repo")), "loopback-proxy"),
        ):
            labels = {
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--label"
            }
            self.assertTrue({
                "otools.manager=model_manager",
                "otools.model=qwen3.8-27b-gptq-int4-b70",
                "otools.device=b70",
                "otools.backend=vllm-xpu-docker",
                f"otools.role={role}",
            }.issubset(labels))

    def test_model_has_render_only_access_and_resource_guards(self):
        command = deploy.model_create_command(Path("/repo"), Path("/model"), 990)
        devices = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--device"
        ]
        self.assertEqual(devices, ["/dev/dri/renderD128:/dev/dri/renderD128"])
        self.assertNotIn("/dev/dri:/dev/dri", command)
        self.assertIn("24g", command)
        self.assertIn("25g", command)

    def test_proxy_publishes_only_to_loopback(self):
        command = deploy.proxy_create_command(Path("/repo"))
        publish = command.index("--publish")
        self.assertEqual(command[publish + 1], "127.0.0.1:8000:8000")
        self.assertIn("--read-only", command)
        self.assertIn("65534:65534", command)

    def test_plan_never_executes_subprocess_or_docker(self):
        output = io.StringIO()
        with mock.patch.object(deploy.subprocess, "run", side_effect=AssertionError), \
                contextlib.redirect_stdout(output):
            self.assertEqual(deploy.main([
                "plan", "b70", "qwen3.8-27b-gptq-int4-b70",
            ]), 0)
        body = output.getvalue()
        self.assertIn("docker create --name b70-vllm-qwen38-262k", body)
        self.assertIn(deploy.IMAGE, body)
        self.assertIn(deploy.REVISION, body)

    def test_unrelated_same_name_container_is_refused(self):
        inspect = {
            "Config": {
                "Image": deploy.IMAGE,
                "Cmd": deploy.VLLM_ARGS,
                "Labels": {"otools.manager": "someone-else"},
            },
        }
        with self.assertRaisesRegex(deploy.DeployError, "refusing unrelated"):
            deploy.validate_stop_ownership(deploy.MODEL_CONTAINER, inspect)

    def test_standard_labels_do_not_override_command_drift(self):
        inspect = {
            "Config": {
                "Image": deploy.IMAGE,
                "Cmd": ["unrelated"],
                "Labels": deploy.ownership_labels("model-server"),
            },
        }
        with self.assertRaisesRegex(deploy.DeployError, "refusing drifted"):
            deploy.validate_stop_ownership(deploy.MODEL_CONTAINER, inspect)

    def test_prior_profile_label_is_a_narrow_adoption_path(self):
        inspect = {
            "Config": {
                "Image": deploy.IMAGE,
                "Cmd": deploy.LEGACY_VLLM_ARGS,
                "Labels": {deploy.LEGACY_LABEL: deploy.PROFILE},
            },
        }
        deploy.validate_stop_ownership(deploy.MODEL_CONTAINER, inspect)
        inspect["Config"]["Cmd"] = ["unrelated"]
        with self.assertRaisesRegex(deploy.DeployError, "refusing unrelated"):
            deploy.validate_stop_ownership(deploy.MODEL_CONTAINER, inspect)

    def test_docker_inspect_error_is_not_treated_as_absent(self):
        result = SimpleNamespace(returncode=1, stdout="", stderr="permission denied")
        with mock.patch.object(deploy, "run", return_value=result):
            with self.assertRaisesRegex(deploy.DeployError, "permission denied"):
                deploy.inspect_container("container")


if __name__ == "__main__":
    unittest.main()

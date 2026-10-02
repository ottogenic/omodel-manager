"""Offline lifecycle checks for the RTX 4070 ASR slot."""

import unittest
from unittest import mock

from utils.card import deploy_rtx4070_qwen_asr as asr


class RTX4070ASRTests(unittest.TestCase):
    def test_official_checkpoint_is_isolated_from_cosyvoice(self):
        command = asr.create_command("GPU-test")
        self.assertEqual(asr.REPOSITORY, "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(command[command.index("--publish") + 1], "127.0.0.1:8003:8000")
        self.assertIn("device=GPU-test", command)
        self.assertIn("/model:ro", " ".join(command))
        self.assertIn("qwen-asr-serve", command)
        self.assertIn("--enforce-eager", command)
        self.assertEqual(asr.labels()["otools.device"], "rtx4070-asr")

    def test_docker_cannot_bypass_manager(self):
        with mock.patch.object(asr, "DOCKER_RUNNER", None):
            with self.assertRaisesRegex(asr.DeployError, "omodel-manager"):
                asr.docker(["ps"])

    def test_rejects_unowned_container(self):
        with self.assertRaisesRegex(asr.DeployError, "unrelated"):
            asr.require_ownership({"Config": {"Labels": {"otools.manager": "other"}}})

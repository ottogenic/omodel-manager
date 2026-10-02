"""RTX 4070 co-hosted deployments must select exactly the intended GPU."""

import unittest
from types import SimpleNamespace
from unittest import mock

from utils.card import rtx4070_gpu


class RTX4070GPUTests(unittest.TestCase):
    def test_selects_4070_uuid_not_another_card(self):
        output = "NVIDIA RTX A6000, GPU-other\nNVIDIA GeForce RTX 4070, GPU-selected\n"
        with mock.patch.object(rtx4070_gpu.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout=output)):
            self.assertEqual(rtx4070_gpu.gpu_uuid(), "GPU-selected")

    def test_rejects_ambiguous_4070_selection(self):
        output = "NVIDIA RTX 4070, GPU-first\nNVIDIA RTX 4070, GPU-second\n"
        with mock.patch.object(rtx4070_gpu.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout=output)):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                rtx4070_gpu.gpu_uuid()

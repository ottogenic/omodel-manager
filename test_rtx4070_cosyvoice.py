"""Experimental CosyVoice launch modes retain an explicit native rollback."""
import os
import unittest
from unittest.mock import patch

from utils.card import deploy_rtx4070_cosyvoice as cosy


class CosyVoiceModes(unittest.TestCase):
    def test_default_is_fp16_on_the_existing_image(self):
        with patch.dict(os.environ, {}, clear=True):
            command = cosy.create_command("gpu-test")
        self.assertIn(f"{cosy.MODEL_DIR}:/model:ro", command)
        self.assertEqual(command[-1], cosy.IMAGE)
        self.assertIn("COSYVOICE_ACCEL=fp16", command)

    def test_native_rollback_does_not_replace_image(self):
        with patch.dict(os.environ, {"COSYVOICE_ACCEL": "native"}):
            command = cosy.create_command("gpu-test")
        self.assertEqual(command[-1], cosy.IMAGE)
        self.assertIn("COSYVOICE_ACCEL=native", command)
        self.assertIn(f"{cosy.MODEL_DIR}:/model:ro", command)

    def test_accelerated_modes_use_isolated_images(self):
        for mode, image in (("trt", cosy.TRT_IMAGE), ("vllm", cosy.VLLM_IMAGE)):
            with self.subTest(mode=mode), patch.dict(os.environ, {"COSYVOICE_ACCEL": mode}):
                command = cosy.create_command("gpu-test")
                self.assertEqual(command[-1], image)
                self.assertIn(f"{cosy.MODEL_DIR}:/model:rw", command)

    def test_unknown_mode_refused(self):
        with patch.dict(os.environ, {"COSYVOICE_ACCEL": "unknown"}):
            with self.assertRaisesRegex(cosy.DeployError, "COSYVOICE_ACCEL"):
                cosy.create_command("gpu-test")


if __name__ == "__main__":
    unittest.main()

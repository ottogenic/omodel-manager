#!/usr/bin/env python3
"""Offline tests for the concurrent endpoint benchmark."""

import unittest
from unittest import mock

from utils import benchmark_concurrent as benchmark


class _Response:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, exc_type, exc, tb):
        return False


class BenchmarkTests(unittest.TestCase):
    def test_reasoning_stream_counts_as_output(self):
        response = _Response([
            b'data: {"choices":[{"delta":{"reasoning":"first"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"answer"}}],'
            b'"usage":{"prompt_tokens":100,"completion_tokens":6}}\n',
            b'data: [DONE]\n',
        ])
        with mock.patch.object(benchmark.urllib.request, "urlopen", return_value=response), \
                mock.patch.object(benchmark.time, "time", side_effect=[0.0, 2.0, 3.0]):
            result = benchmark.run_one("http://example.test", "model", "story", 10, 30)

        self.assertTrue(result["ok"])
        self.assertEqual(result["ttft"], 2.0)
        self.assertEqual(result["in_toks"], 100)
        self.assertEqual(result["out_toks"], 6)
        self.assertEqual(result["decode_tps"], 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

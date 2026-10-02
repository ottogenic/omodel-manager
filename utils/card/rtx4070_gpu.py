"""Identify the local RTX 4070 for co-hosted card deployments."""

import subprocess


def gpu_uuid():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    matches = [line.split(",", 1)[-1].strip() for line in result.stdout.splitlines()
               if "RTX 4070" in line]
    if result.returncode or len(matches) != 1 or not matches[0].startswith("GPU-"):
        raise ValueError("expected exactly one NVIDIA RTX 4070 visible to nvidia-smi")
    return matches[0]

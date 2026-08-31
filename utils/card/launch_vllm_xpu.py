#!/usr/bin/env python3
"""Apply the pinned single-XPU warmup guard, then exec vLLM."""

import hashlib
import os
from pathlib import Path
import sys

if __package__:
    from .patch_vllm_mtp_boundary import apply_patch as apply_mtp_boundary_patch
else:
    from patch_vllm_mtp_boundary import apply_patch as apply_mtp_boundary_patch


TARGET = Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/xpu_worker.py")
VLLM = "/opt/venv/bin/vllm"
EXPECTED_SHA256 = "bd3b35ad0e5ce23348810b9782f0934e9d9c9ad09398ff4808b3a789b0f9bce8"
PATCHED_SHA256 = "115273d63c4273489ee210998f6153d6a4dea503e4623aebc759f9ef09d98d6d"
ORIGINAL = """        if torch.distributed.is_xccl_available():
            torch.distributed.all_reduce(torch.zeros(1).xpu())
"""
PATCHED = """        if (torch.distributed.is_xccl_available()
                and self.parallel_config.world_size > 1):
            torch.distributed.all_reduce(torch.zeros(1).xpu())
"""


def patch_source(source):
    if source.count(ORIGINAL) != 1:
        raise RuntimeError("expected exactly one pinned XCCL warmup block")
    return source.replace(ORIGINAL, PATCHED, 1)


def main():
    source = TARGET.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == PATCHED_SHA256:
        compile(source, str(TARGET), "exec")
        print(f"single-XPU XCCL warmup already guarded; sha256={digest}", flush=True)
        apply_mtp_boundary_patch()
        os.execv(VLLM, [VLLM, *sys.argv[1:]])
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"refusing to patch {TARGET}: expected {EXPECTED_SHA256} or "
            f"{PATCHED_SHA256}, got {digest}"
        )

    patched = patch_source(source)
    compile(patched, str(TARGET), "exec")
    TARGET.write_text(patched, encoding="utf-8")
    patched_digest = hashlib.sha256(patched.encode()).hexdigest()
    if patched_digest != PATCHED_SHA256:
        raise RuntimeError(
            f"patched {TARGET} has unexpected SHA-256 {patched_digest}"
        )
    print(f"guarded single-XPU XCCL warmup; patched_sha256={patched_digest}", flush=True)
    apply_mtp_boundary_patch()
    os.execv(VLLM, [VLLM, *sys.argv[1:]])


if __name__ == "__main__":
    main()

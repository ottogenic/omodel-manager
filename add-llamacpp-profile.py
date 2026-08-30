#!/usr/bin/env python3
"""Add the qwen3-coder-next-q8-llamacpp profile to model_manager.json (backup first).

Profile-only integration -- no omodel-manager code changes. Two documented tricks:
  1. `model` is a positional arg in omm's vllm-shaped scaffold; llama-server has no
     positional model, so we park a harmless flag ("--jinja") in that slot.
  2. Everything else rides in vllm_args, which emits `--flag value` pairs that
     llama-server understands (--model, --alias, --ctx-size, --parallel, ...).
"""
import json, shutil, sys, glob, os

GGUF_GLOB = "/home/otto/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-Next-GGUF/snapshots/*/Q8_0/*Q8_0-00001-of-*.gguf"
ENTRYPOINT = sys.argv[1] if len(sys.argv) > 1 else "llama-server"

hits = sorted(glob.glob(GGUF_GLOB)) or sorted(glob.glob(GGUF_GLOB.replace("-00001-of-*", "*")))
assert hits, "Q8_0 GGUF not found -- download still running?"
host_gguf = hits[0]
container_gguf = host_gguf.replace("/home/otto/.cache/huggingface", "/root/.cache/huggingface")

P = "/home/otto/Documents/omodel-manager/model_manager.json"
shutil.copy2(P, P + ".pre-llamacpp.bak")
cfg = json.load(open(P))

def find_models_dict(o):
    if isinstance(o, dict):
        if "qwen3-coder-next-fp8" in o:
            return o
        for v in o.values():
            r = find_models_dict(v)
            if r is not None:
                return r
    return None

models = find_models_dict(cfg)
assert models is not None, "could not locate models dict"

models["qwen3-coder-next-q8-llamacpp"] = {
    "image": "scitrera/dgx-spark-llama-cpp:b10107-cu131",
    "model": "--jinja",
    "port": 8000,
    "usecase": ["Coding", "Agentic", "Quality-reference"],
    "docker_flags": ["--entrypoint", ENTRYPOINT],
    "vllm_args": {
        "model": container_gguf,
        "alias": "qwen3-coder-next-q8-llamacpp",
        "ctx-size": 131072,
        "parallel": 1,
        "n-gpu-layers": 999,
        "trust-remote-code": False,
    },
}
json.dump(cfg, open(P, "w"), indent=1)
print("profile added; gguf:", container_gguf)

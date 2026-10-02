"""Local Qwen3-ASR deployment sharing the RTX 4070 with CosyVoice.

All Docker operations are injected through omodel-manager's Docker choke point.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from utils.card import rtx4070_gpu


DEVICE = "rtx4070-asr"
PROFILE = "qwen3-asr-0.6b-rtx4070"
REPOSITORY = "Qwen/Qwen3-ASR-0.6B"
REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
IMAGE = "otools-qwen3-asr:0.0.7"
NAME = "otools-vllm-qwen3-asr-rtx4070"
PORT = 8003
MODEL_DIR = os.path.expanduser("~/.cache/otools/models/qwen3-asr-0.6b")
DOCKERFILE = os.path.join(os.path.dirname(__file__), "Dockerfile.qwen_asr")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DOCKER_RUNNER = None


class DeployError(RuntimeError):
    pass


def docker(argv, capture=True):
    if DOCKER_RUNNER is None:
        raise DeployError("Docker must be dispatched through omodel-manager")
    result = DOCKER_RUNNER(argv, capture=capture)
    if result.returncode:
        raise DeployError((result.stderr or result.stdout or "Docker failed").strip())
    return result


def labels():
    return {"otools.manager": "model_manager", "otools.model": PROFILE,
            "otools.device": DEVICE, "otools.backend": "qwen-asr-vllm",
            "otools.role": "model-server", "otools.port": str(PORT)}


def create_command(uuid):
    argv = ["create", "--name", NAME, "--gpus", f"device={uuid}",
            "--restart", "unless-stopped", "--shm-size", "2g",
            "--publish", f"127.0.0.1:{PORT}:8000"]
    for key, value in labels().items():
        argv += ["--label", f"{key}={value}"]
    return [*argv, "--volume", f"{MODEL_DIR}:/model:ro", "--env", "HF_HUB_OFFLINE=1",
            "--entrypoint", "qwen-asr-serve", IMAGE, "/model", "--host", "0.0.0.0",
            "--port", "8000", "--served-model-name", PROFILE,
            "--gpu-memory-utilization", "0.35", "--max-model-len", "4096",
            "--max-num-seqs", "1", "--enforce-eager"]


def inspect():
    result = DOCKER_RUNNER(["inspect", "--format", "{{json .}}", NAME], capture=True)
    if result.returncode:
        if "no such object" in (result.stderr or "").lower():
            return None
        raise DeployError((result.stderr or "Docker inspect failed").strip())
    return json.loads(result.stdout)


def require_ownership(info):
    found = info["Config"].get("Labels") or {}
    if any(found.get(key) != value for key, value in labels().items()):
        raise DeployError(f"refusing unrelated or drifted container {NAME}")


def plan():
    print(f"Qualified card plan: device={DEVICE} profile={PROFILE}")
    print(f"source: {REPOSITORY}@{REVISION} (official Qwen, Apache-2.0)")
    print(f"image: {IMAGE} built from {DOCKERFILE}; CosyVoice shares the same RTX 4070")
    print(f"download: hf download {REPOSITORY} --revision {REVISION} --local-dir /model")
    print("create: " + " ".join(create_command("<RTX-4070-GPU-UUID>")))
    print(f"transcriptions: http://127.0.0.1:{PORT}/v1/audio/transcriptions")


def launch():
    uuid = rtx4070_gpu.gpu_uuid()
    current = inspect()
    if current:
        require_ownership(current)
        if current["State"]["Running"]:
            raise DeployError(f"{NAME} is already running")
        docker(["rm", NAME])
    if DOCKER_RUNNER(["image", "inspect", IMAGE], capture=True).returncode:
        docker(["build", "-f", DOCKERFILE, "-t", IMAGE, ROOT], capture=False)
    os.makedirs(MODEL_DIR, exist_ok=True)
    docker(["run", "--rm", "--volume", f"{MODEL_DIR}:/model",
            "--entrypoint", "hf", IMAGE, "download", REPOSITORY,
            "--revision", REVISION, "--local-dir", "/model"], capture=False)
    docker(create_command(uuid))
    docker(["start", NAME])
    print(f"starting: http://127.0.0.1:{PORT}/v1 ({PROFILE})")
    print(f"Check readiness: omm health {DEVICE}; logs: omm logs {DEVICE}")


def health():
    current = inspect()
    if not current:
        raise DeployError("ASR container is absent")
    require_ownership(current)
    if not current["State"]["Running"]:
        raise DeployError(f"ASR container exited ({current['State'].get('ExitCode')}); see logs")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as response:
            ids = [item.get("id") for item in json.load(response).get("data", [])]
        if PROFILE not in ids:
            raise DeployError("ASR model is not yet served")
    except (OSError, urllib.error.URLError) as exc:
        raise DeployError(f"ASR is still loading: {exc}") from exc
    print(f"READY: {PROFILE} at http://127.0.0.1:{PORT}/v1")


def logs(follow=False):
    current = inspect()
    if not current:
        raise DeployError("ASR container is absent")
    require_ownership(current)
    docker(["logs", "--tail", "150", *(["-f"] if follow else []), NAME], capture=False)


def stop():
    current = inspect()
    if not current:
        raise DeployError("ASR container is absent")
    require_ownership(current)
    if current["State"]["Running"]:
        docker(["stop", NAME])
    print(f"stopped: {NAME} (container retained for logs)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "launch", "health", "logs", "stop"))
    parser.add_argument("device")
    parser.add_argument("profile", nargs="?")
    parser.add_argument("-f", "--follow", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.device != DEVICE or (args.profile and args.profile != PROFILE):
        parser.error("unknown qualified card or profile")
    try:
        if args.action == "plan":
            plan()
        elif args.action == "launch":
            launch()
        elif args.action == "health":
            health()
        elif args.action == "logs":
            logs(args.follow)
        else:
            stop()
    except (DeployError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0

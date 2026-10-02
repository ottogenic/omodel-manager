"""Experimental native CosyVoice 3 deployment; Docker injected by manager."""

import argparse
import json
import os
import sys
import urllib.request

from utils.card import rtx4070_gpu


class DeployError(RuntimeError):
    pass

DEVICE = "rtx4070"
PROFILE = "cosyvoice3-0.5b-rtx4070"
REPOSITORY = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
IMAGE = "otools-cosyvoice3:0.1.1"
TRT_IMAGE = "otools-cosyvoice3:0.1.1-trt-test"
VLLM_IMAGE = "otools-cosyvoice3:0.1.1-vllm-test"
NAME = "otools-cosyvoice3-rtx4070"
PORT = 8001  # Existing tailnet-only HTTPS route; Qwen Gradio rollback is stopped.
MODEL_DIR = os.path.expanduser("~/.cache/otools/models/cosyvoice3-0.5b-2512")
VOICES = os.path.expanduser("~/.local/share/otools/cosyvoice3/voices")
DOCKERFILE = os.path.join(os.path.dirname(__file__), "Dockerfile.cosyvoice")
UI = os.path.join(os.path.dirname(__file__), "cosyvoice_ui.py")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DOCKER_RUNNER = None


def docker(argv, capture=True):
    if DOCKER_RUNNER is None:
        raise DeployError("Docker must be dispatched through omodel-manager")
    result = DOCKER_RUNNER(argv, capture=capture)
    if result.returncode:
        raise DeployError((result.stderr or result.stdout or "Docker failed").strip())
    return result


def labels():
    return {"otools.manager": "model_manager", "otools.model": PROFILE,
            "otools.device": DEVICE, "otools.backend": "cosyvoice-native",
            "otools.role": "model-server", "otools.port": str(PORT)}


def create_command(uuid):
    argv = ["create", "--name", NAME, "--gpus", f"device={uuid}",
            "--restart", "unless-stopped", "--shm-size", "2g",
            "--publish", f"127.0.0.1:{PORT}:8000"]
    for key, value in labels().items():
        argv += ["--label", f"{key}={value}"]
    accel = os.environ.get("COSYVOICE_ACCEL", "fp16")
    if accel not in ("native", "fp16", "vllm", "trt"):
        raise DeployError("COSYVOICE_ACCEL must be native, fp16, vllm or trt")
    image = {"trt": TRT_IMAGE, "vllm": VLLM_IMAGE}.get(accel, IMAGE)
    # Accelerators export engines into a subdirectory of the model cache.
    model_mount = f"{MODEL_DIR}:/model:{'rw' if accel in ('trt', 'vllm') else 'ro'}"
    accel_env = (["--env", "MKL_THREADING_LAYER=GNU",
                  "--env", "VLLM_ENABLE_V1_MULTIPROCESSING=0"] if accel == "vllm" else [])
    return [*argv, "--volume", model_mount, "--volume", f"{VOICES}:/voices",
             "--volume", f"{UI}:/opt/CosyVoice/cosyvoice_ui.py:ro",
             "--env", "HF_HUB_OFFLINE=1", "--env", "GRADIO_ANALYTICS_ENABLED=False",
             "--env", f"COSYVOICE_ACCEL={accel}", *accel_env,
             image]


def inspect():
    if DOCKER_RUNNER is None:
        raise DeployError("Docker must be dispatched through omodel-manager")
    result = DOCKER_RUNNER(["inspect", "--format", "{{json .}}", NAME], capture=True)
    if result.returncode:
        if "no such object" in (result.stderr or "").lower():
            return None
        raise DeployError((result.stderr or "Docker inspect failed").strip())
    return json.loads(result.stdout)


def require_ownership(info):
    found = info["Config"].get("Labels") or {}
    if (any(found.get(k) != v for k, v in labels().items() if k != "otools.port")
            or found.get("otools.port") not in ("8005", str(PORT))):
        raise DeployError(f"refusing unrelated or drifted container {NAME}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "launch", "health", "logs", "stop"))
    parser.add_argument("device")
    parser.add_argument("profile", nargs="?")
    parser.add_argument("-f", "--follow", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.device != DEVICE or (args.profile and args.profile != PROFILE):
        parser.error("unknown experimental card/profile")
    try:
        if args.action == "plan":
            accel = os.environ.get("COSYVOICE_ACCEL", "fp16")
            print(f"Experimental native CosyVoice 3: {REPOSITORY}@{REVISION}")
            image = {"trt": TRT_IMAGE, "vllm": VLLM_IMAGE}.get(accel, IMAGE)
            dockerfile = {"trt": "Dockerfile.cosyvoice_trt", "vllm": "Dockerfile.cosyvoice_vllm"}.get(accel)
            print(f"build: {os.path.join(os.path.dirname(__file__), dockerfile) if dockerfile else DOCKERFILE} -> {image}")
            print("create: " + " ".join(create_command("<RTX-4070-GPU-UUID>")))
            return 0
        info = inspect()
        if info:
            require_ownership(info)
        if args.action == "launch":
            if info and info["State"]["Running"]:
                raise DeployError("CosyVoice is already running")
            # Keep the current device deployment until the image and weights are ready.
            accel = os.environ.get("COSYVOICE_ACCEL", "fp16")
            image = {"trt": TRT_IMAGE, "vllm": VLLM_IMAGE}.get(accel, IMAGE)
            if DOCKER_RUNNER(["image", "inspect", image], capture=True).returncode:
                dockerfile = {"trt": "Dockerfile.cosyvoice_trt", "vllm": "Dockerfile.cosyvoice_vllm"}.get(accel)
                docker(["build", "-f", os.path.join(os.path.dirname(__file__), dockerfile) if dockerfile else DOCKERFILE,
                        "-t", image, ROOT], capture=False)
            os.makedirs(MODEL_DIR, exist_ok=True)
            if not os.path.isfile(os.path.join(MODEL_DIR, "cosyvoice3.yaml")):
                docker(["run", "--rm", "--volume", f"{MODEL_DIR}:/model",
                        "--entrypoint", "hf", IMAGE, "download", REPOSITORY,
                        "--revision", REVISION, "--local-dir", "/model"], capture=False)
            os.makedirs(VOICES, mode=0o700, exist_ok=True)
            if info:
                docker(["rm", NAME])
            docker(create_command(rtx4070_gpu.gpu_uuid()))
            docker(["start", NAME])
            print(f"starting at http://127.0.0.1:{PORT}/")
        elif args.action == "health":
            if not info or not info["State"]["Running"]:
                raise DeployError("CosyVoice is not running; inspect logs")
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=10) as response:
                if response.status != 200 or b"CosyVoice 3" not in response.read(200_000):
                    raise DeployError("CosyVoice UI is not ready")
            print(f"READY: {PROFILE} at http://127.0.0.1:{PORT}/")
        elif args.action == "logs":
            if not info:
                raise DeployError("CosyVoice container absent")
            docker(["logs", "--tail", "150", *(["-f"] if args.follow else []), NAME], capture=False)
        else:
            if not info:
                raise DeployError("CosyVoice container absent")
            if info["State"]["Running"]:
                docker(["stop", NAME])
            print("CosyVoice stopped; container retained")
    except (DeployError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0

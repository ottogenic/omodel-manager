#!/usr/bin/env python3
"""Manage the qualified B70 Qwen3.8 vLLM deployment."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    from .download_qwen38_vllm_model import (
        DEFAULT_DESTINATION, FILES, REPOSITORY, REVISION,
    )
    from .launch_vllm_xpu import (
        EXPECTED_SHA256 as XPU_EXPECTED_SHA256,
        PATCHED_SHA256 as XPU_PATCHED_SHA256,
    )
    from .patch_vllm_mtp_boundary import (
        EXPECTED_SHA256 as MTP_EXPECTED_SHA256,
        PATCHED_SHA256 as MTP_PATCHED_SHA256,
    )
except ImportError:
    from download_qwen38_vllm_model import (
        DEFAULT_DESTINATION, FILES, REPOSITORY, REVISION,
    )
    from launch_vllm_xpu import (
        EXPECTED_SHA256 as XPU_EXPECTED_SHA256,
        PATCHED_SHA256 as XPU_PATCHED_SHA256,
    )
    from patch_vllm_mtp_boundary import (
        EXPECTED_SHA256 as MTP_EXPECTED_SHA256,
        PATCHED_SHA256 as MTP_PATCHED_SHA256,
    )


IMAGE = (
    "vllm/vllm-openai-xpu@"
    "sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
)
NETWORK = "b70-vllm-isolated"
MODEL_CONTAINER = "b70-vllm-qwen38-262k"
PROXY_CONTAINER = "b70-vllm-loopback"
MODEL_ID = "qwen3.8-27b-gptq-int4-b70"
DEVICE = "b70"
MODEL_PROFILE = MODEL_ID
CONTEXT_LENGTH = 262_144
PROFILE = "b70-qwen38-vllm-262k-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(DEFAULT_DESTINATION).expanduser()
RENDER_DEVICE = Path("/dev/dri/renderD128")

LABEL_MANAGER = "otools.manager"
LABEL_MODEL = "otools.model"
LABEL_DEVICE = "otools.device"
LABEL_BACKEND = "otools.backend"
LABEL_ROLE = "otools.role"
LABEL_VALUE = "model_manager"
BACKEND = "vllm-xpu-docker"
LEGACY_LABEL = "org.omodel-card.profile"
DOCKER_RUNNER = None

MODEL_ENV = [
    "CCL_ZE_IPC_EXCHANGE=sockets",
    "VLLM_TARGET_DEVICE=xpu",
    "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
    "ZE_AFFINITY_MASK=0",
    "VLLM_XPU_ENABLE_XPU_GRAPH=1",
    "PYTORCH_ALLOC_CONF=expandable_segments:True",
]
VLLM_ARGS = [
    "/bench/utils/card/launch_vllm_xpu.py",
    "serve",
    "/model",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--quantization",
    "gptq",
    "--dtype",
    "float16",
    "--max-model-len",
    str(CONTEXT_LENGTH),
    "--kv-cache-memory-bytes",
    "10737418240",
    "--kv-cache-dtype",
    "fp8",
    "--max-num-seqs",
    "1",
    "--max-num-batched-tokens",
    "8192",
    "--enable-prefix-caching",
    "--served-model-name",
    MODEL_ID,
    "--speculative-config",
    '{"method":"mtp","num_speculative_tokens":4}',
    "--default-chat-template-kwargs",
    '{"enable_thinking":true,"reasoning_effort":"medium"}',
    "--reasoning-parser",
    "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "qwen3_coder",
]
TEXT_ONLY_VLLM_ARGS = VLLM_ARGS.copy()
TEXT_ONLY_VLLM_ARGS.insert(
    TEXT_ONLY_VLLM_ARGS.index("--speculative-config"), "--language-model-only",
)
LEGACY_VLLM_ARGS = [
    "/bench/utils/launch_vllm_xpu.py" if value == VLLM_ARGS[0]
    else "qwen3.8-27b" if value == MODEL_ID else value
    for value in TEXT_ONLY_VLLM_ARGS
]


class DeployError(RuntimeError):
    pass


def run(command, check=True, capture=True):
    if command and command[0] == "docker" and DOCKER_RUNNER is not None:
        result = DOCKER_RUNNER(command[1:], capture=capture, check=False)
    else:
        result = subprocess.run(
            command,
            capture_output=capture,
            text=True,
            check=False,
        )
    if check and result.returncode:
        detail = ((result.stderr or result.stdout).strip() if capture else "")
        raise DeployError(f"command failed ({result.returncode}): {detail}")
    return result


def ownership_labels(role):
    return {
        LABEL_MANAGER: LABEL_VALUE,
        LABEL_MODEL: MODEL_PROFILE,
        LABEL_DEVICE: DEVICE,
        LABEL_BACKEND: BACKEND,
        LABEL_ROLE: role,
    }


def append_labels(command, role):
    for key, value in ownership_labels(role).items():
        command.extend(["--label", f"{key}={value}"])


def model_create_command(repo_root, model_path, render_group):
    command = [
        "docker", "create", "--name", MODEL_CONTAINER,
        "--network", NETWORK,
        "--memory", "24g", "--memory-swap", "25g", "--shm-size", "2g",
        "--device", f"{RENDER_DEVICE}:{RENDER_DEVICE}",
        "--group-add", str(render_group),
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    ]
    append_labels(command, "model-server")
    command.extend([
        "--volume", f"{model_path}:/model:ro",
        "--volume", f"{repo_root}:/bench:ro",
        "--entrypoint", "python",
    ])
    for value in MODEL_ENV:
        command.extend(["--env", value])
    return command + [IMAGE, *VLLM_ARGS]


def proxy_create_command(repo_root):
    command = [
        "docker", "create", "--name", PROXY_CONTAINER,
        "--network", "bridge",
        "--publish", "127.0.0.1:8000:8000",
        "--memory", "64m", "--memory-swap", "64m", "--pids-limit", "32",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=4m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
    ]
    append_labels(command, "loopback-proxy")
    command.extend([
        "--volume", f"{repo_root / 'utils' / 'card' / 'tcp_proxy.py'}:/proxy.py:ro",
        "--entrypoint", "python",
        IMAGE, "-I", "/proxy.py", MODEL_CONTAINER, "8000",
    ])
    return command


def inspect_container(name):
    result = run(["docker", "container", "inspect", name], check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if "No such container" in detail or "No such object" in detail:
            return None
        raise DeployError(f"cannot inspect {name}: {detail}")
    return json.loads(result.stdout)[0]


def environment_map(values):
    return dict(value.split("=", 1) for value in values)


def expected_environment(overrides):
    inspect = json.loads(run(["docker", "image", "inspect", IMAGE]).stdout)[0]
    environment = environment_map(inspect["Config"].get("Env") or [])
    environment.update(environment_map(overrides))
    return environment


def require_ownership(inspect, name, role, allowed_commands, allow_legacy=True):
    labels = inspect["Config"].get("Labels") or {}
    expected = ownership_labels(role)
    standard_present = any(key in labels for key in expected)
    if standard_present:
        if any(labels.get(key) != value for key, value in expected.items()):
            raise DeployError(f"refusing unrelated or partially owned container: {name}")
        if (inspect["Config"].get("Image") != IMAGE
                or inspect["Config"].get("Cmd") not in allowed_commands):
            raise DeployError(f"refusing drifted owned container: {name}")
        return
    legacy = (
        allow_legacy
        and labels.get(LEGACY_LABEL) == PROFILE
        and inspect["Config"].get("Image") == IMAGE
        and inspect["Config"].get("Cmd") in allowed_commands
    )
    if not legacy:
        raise DeployError(f"refusing unrelated container: {name}")


def require_common_security(inspect, name, role, commands, memory, memory_swap):
    host = inspect["HostConfig"]
    require_ownership(inspect, name, role, commands)
    if inspect["Config"]["Image"] != IMAGE:
        raise DeployError(f"{name} uses a different image")
    if host["Memory"] != memory or host["MemorySwap"] != memory_swap:
        raise DeployError(f"{name} has different memory guards")
    if host.get("CapAdd") is not None or host.get("CapDrop") != ["ALL"]:
        raise DeployError(f"{name} does not drop all capabilities")
    if host.get("SecurityOpt") != ["no-new-privileges"]:
        raise DeployError(f"{name} lacks no-new-privileges")
    if host.get("Privileged") or host.get("PublishAllPorts"):
        raise DeployError(f"{name} has privileged Docker settings")
    if host.get("RestartPolicy") != {"Name": "no", "MaximumRetryCount": 0}:
        raise DeployError(f"{name} has an unexpected restart policy")


def verify_model_container(inspect, repo_root, model_path, render_group, expected_env):
    require_common_security(
        inspect, MODEL_CONTAINER, "model-server", [VLLM_ARGS],
        24 * 1024**3, 25 * 1024**3,
    )
    config = inspect["Config"]
    host = inspect["HostConfig"]
    if config.get("Entrypoint") != ["python"] or config.get("Cmd") != VLLM_ARGS:
        raise DeployError(f"{MODEL_CONTAINER} has different vLLM arguments")
    if (
        host.get("NetworkMode") != NETWORK
        or set(inspect["NetworkSettings"]["Networks"]) != {NETWORK}
        or host.get("ShmSize") != 2 * 1024**3
        or host.get("IpcMode") != "private"
        or host.get("PidMode") != ""
        or host.get("ReadonlyRootfs")
        or host.get("PidsLimit") is not None
    ):
        raise DeployError(f"{MODEL_CONTAINER} has different runtime isolation")
    if host.get("GroupAdd") != [str(render_group)]:
        raise DeployError(f"{MODEL_CONTAINER} has a different render group")
    expected_devices = [{
        "PathOnHost": os.fspath(RENDER_DEVICE),
        "PathInContainer": os.fspath(RENDER_DEVICE),
        "CgroupPermissions": "rwm",
    }]
    if host.get("Devices") != expected_devices or host.get("DeviceRequests") is not None:
        raise DeployError(f"{MODEL_CONTAINER} lacks render-only GPU access")
    expected_binds = {
        f"{model_path}:/model:ro",
        f"{repo_root}:/bench:ro",
    }
    if set(host.get("Binds") or []) != expected_binds:
        raise DeployError(f"{MODEL_CONTAINER} has different bind mounts")
    if environment_map(config.get("Env") or []) != expected_env:
        raise DeployError(f"{MODEL_CONTAINER} has different XPU environment")


def verify_proxy_container(inspect, repo_root, expected_env):
    expected_command = ["-I", "/proxy.py", MODEL_CONTAINER, "8000"]
    require_common_security(
        inspect, PROXY_CONTAINER, "loopback-proxy", [expected_command],
        64 * 1024**2, 64 * 1024**2,
    )
    config = inspect["Config"]
    host = inspect["HostConfig"]
    if config.get("Entrypoint") != ["python"] or config.get("Cmd") != expected_command:
        raise DeployError(f"{PROXY_CONTAINER} has different proxy arguments")
    if (
        config.get("User") != "65534:65534"
        or environment_map(config.get("Env") or []) != expected_env
        or not host.get("ReadonlyRootfs")
        or host.get("NetworkMode") != "bridge"
        or set(inspect["NetworkSettings"]["Networks"]) != {"bridge", NETWORK}
        or host.get("IpcMode") != "private"
        or host.get("PidMode") != ""
        or host.get("PidsLimit") != 32
        or host.get("GroupAdd") is not None
        or host.get("Devices") != []
        or host.get("DeviceRequests") is not None
        or host.get("Tmpfs") != {"/tmp": "rw,noexec,nosuid,size=4m"}
    ):
        raise DeployError(f"{PROXY_CONTAINER} has different process isolation")
    bindings = (host.get("PortBindings") or {}).get("8000/tcp") or []
    if bindings != [{"HostIp": "127.0.0.1", "HostPort": "8000"}]:
        raise DeployError(f"{PROXY_CONTAINER} is not loopback-only")
    expected_bind = f"{repo_root / 'utils' / 'card' / 'tcp_proxy.py'}:/proxy.py:ro"
    if host.get("Binds") != [expected_bind]:
        raise DeployError(f"{PROXY_CONTAINER} has a different proxy mount")


def ensure_network():
    result = run(["docker", "network", "inspect", NETWORK], check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if "not found" in detail:
            run(["docker", "network", "create", "--internal", NETWORK])
            return
        raise DeployError(f"cannot inspect {NETWORK}: {detail}")
    network = json.loads(result.stdout)[0]
    if network.get("Driver") != "bridge" or not network.get("Internal"):
        raise DeployError(f"{NETWORK} exists but is not an internal bridge")


def ensure_image():
    result = run(["docker", "image", "inspect", IMAGE], check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if "No such image" not in detail:
            raise DeployError(f"cannot inspect pinned image: {detail}")
        run(["docker", "pull", IMAGE])


def verify_model_files(model_path):
    command = [
        sys.executable,
        os.fspath(REPOSITORY_ROOT / "utils" / "card" / "download_qwen38_vllm_model.py"),
        "--destination", os.fspath(model_path),
        "--verify-only",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise DeployError("pinned model verification failed")


def ensure_container(name, create_command, verify):
    inspect = inspect_container(name)
    if inspect is None:
        run(create_command)
        inspect = inspect_container(name)
    verify(inspect)
    if not inspect["State"]["Running"]:
        run(["docker", "start", name])


def refuse_unowned_existing_containers():
    for name in (MODEL_CONTAINER, PROXY_CONTAINER):
        inspect = inspect_container(name)
        if inspect is not None:
            validate_stop_ownership(name, inspect)


def replace_previous_text_only_container():
    inspect = inspect_container(MODEL_CONTAINER)
    if inspect is None or inspect["Config"].get("Cmd") != TEXT_ONLY_VLLM_ARGS:
        return
    validate_stop_ownership(MODEL_CONTAINER, inspect)
    if inspect["State"]["Running"]:
        run(["docker", "stop", MODEL_CONTAINER])
    run(["docker", "rm", MODEL_CONTAINER])


def replace_model_with_different_repo_mount(repo_root, model_path):
    inspect = inspect_container(MODEL_CONTAINER)
    if inspect is None or inspect["Config"].get("Cmd") != VLLM_ARGS:
        return
    binds = inspect["HostConfig"].get("Binds") or []
    expected_model = f"{model_path}:/model:ro"
    expected_repo = f"{repo_root}:/bench:ro"
    if set(binds) == {expected_model, expected_repo}:
        return
    if len(binds) != 2 or expected_model not in binds or not any(
            value.endswith(":/bench:ro") for value in binds):
        return
    validate_stop_ownership(MODEL_CONTAINER, inspect)
    if inspect["State"]["Running"]:
        run(["docker", "stop", MODEL_CONTAINER])
    run(["docker", "rm", MODEL_CONTAINER])


def replace_proxy_with_different_mount(repo_root):
    inspect = inspect_container(PROXY_CONTAINER)
    if inspect is None:
        return
    expected = f"{repo_root / 'utils' / 'card' / 'tcp_proxy.py'}:/proxy.py:ro"
    if inspect["HostConfig"].get("Binds") == [expected]:
        return
    validate_stop_ownership(PROXY_CONTAINER, inspect)
    if inspect["State"]["Running"]:
        run(["docker", "stop", PROXY_CONTAINER])
    run(["docker", "rm", PROXY_CONTAINER])


def check_service_once():
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
        if response.status != 200:
            raise DeployError(f"health endpoint returned HTTP {response.status}")
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=15) as response:
        models = json.load(response).get("data", [])
    selected = [item for item in models if item.get("id") == MODEL_ID]
    if len(selected) != 1 or selected[0].get("max_model_len") != CONTEXT_LENGTH:
        raise DeployError("served model identity or context length does not match")


def wait_for_service(timeout):
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        try:
            check_service_once()
            return
        except (DeployError, OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise DeployError(f"service did not become healthy: {last_error}")


def launch(timeout=600):
    if not MODEL_PATH.is_dir():
        raise DeployError(
            f"model is missing at {MODEL_PATH}; run "
            "python3 utils/card/download_qwen38_vllm_model.py"
        )
    if not RENDER_DEVICE.exists():
        raise DeployError(f"render device is missing: {RENDER_DEVICE}")
    refuse_unowned_existing_containers()
    replace_previous_text_only_container()
    replace_model_with_different_repo_mount(REPOSITORY_ROOT, MODEL_PATH)
    verify_model_files(MODEL_PATH)
    ensure_image()
    ensure_network()
    render_group = RENDER_DEVICE.stat().st_gid
    model_environment = expected_environment(MODEL_ENV)
    proxy_environment = expected_environment(["PYTHONDONTWRITEBYTECODE=1"])
    ensure_container(
        MODEL_CONTAINER,
        model_create_command(REPOSITORY_ROOT, MODEL_PATH, render_group),
        lambda inspect: verify_model_container(
            inspect, REPOSITORY_ROOT, MODEL_PATH, render_group, model_environment,
        ),
    )
    replace_proxy_with_different_mount(REPOSITORY_ROOT)
    proxy = inspect_container(PROXY_CONTAINER)
    if proxy is None:
        run(proxy_create_command(REPOSITORY_ROOT))
        run(["docker", "network", "connect", NETWORK, PROXY_CONTAINER])
        proxy = inspect_container(PROXY_CONTAINER)
    verify_proxy_container(proxy, REPOSITORY_ROOT, proxy_environment)
    if not proxy["State"]["Running"]:
        run(["docker", "start", PROXY_CONTAINER])
    wait_for_service(timeout)
    print(f"ready: http://127.0.0.1:8000/v1 ({MODEL_ID}, {CONTEXT_LENGTH} context)")


def validate_stop_ownership(name, inspect):
    if name == MODEL_CONTAINER:
        labels = inspect["Config"].get("Labels") or {}
        standard_present = any(key in labels for key in ownership_labels("model-server"))
        commands = [VLLM_ARGS, TEXT_ONLY_VLLM_ARGS]
        if not standard_present:
            commands.append(LEGACY_VLLM_ARGS)
        role = "model-server"
    else:
        commands = [["-I", "/proxy.py", MODEL_CONTAINER, "8000"]]
        role = "loopback-proxy"
    require_ownership(inspect, name, role, commands)


def stop(yes=False):
    inspected = []
    for name in (PROXY_CONTAINER, MODEL_CONTAINER):
        value = inspect_container(name)
        if value is not None:
            validate_stop_ownership(name, value)
            inspected.append((name, value))
    if not yes:
        reply = input(f"Stop the qualified deployment on {DEVICE}? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return False
    for name, inspect in inspected:
        if inspect["State"]["Running"]:
            run(["docker", "stop", name])
    print(f"stopped: {len(inspected)} retained container(s)")
    return True


def logs(follow=False):
    inspect = inspect_container(MODEL_CONTAINER)
    if inspect is None:
        raise DeployError(f"{MODEL_CONTAINER} does not exist")
    validate_stop_ownership(MODEL_CONTAINER, inspect)
    command = ["docker", "logs"]
    if follow:
        command.append("-f")
    command.append(MODEL_CONTAINER)
    result = run(command, check=False, capture=False)
    if result.returncode:
        raise DeployError(f"docker logs failed with status {result.returncode}")


def health():
    model = inspect_container(MODEL_CONTAINER)
    proxy = inspect_container(PROXY_CONTAINER)
    if model is None or proxy is None:
        raise DeployError("deployment containers are absent")
    render_group = RENDER_DEVICE.stat().st_gid
    verify_model_container(
        model, REPOSITORY_ROOT, MODEL_PATH, render_group, expected_environment(MODEL_ENV),
    )
    verify_proxy_container(
        proxy, REPOSITORY_ROOT,
        expected_environment(["PYTHONDONTWRITEBYTECODE=1"]),
    )
    if not model["State"]["Running"] or not proxy["State"]["Running"]:
        raise DeployError("deployment containers are not both running")
    check_service_once()
    print(f"READY: {MODEL_ID} ({CONTEXT_LENGTH} context)")


def plan():
    render_group = "<render-group-gid>"
    commands = [
        ["docker", "network", "create", "--internal", NETWORK],
        model_create_command(REPOSITORY_ROOT, MODEL_PATH, render_group),
        ["docker", "start", MODEL_CONTAINER],
        proxy_create_command(REPOSITORY_ROOT),
        ["docker", "network", "connect", NETWORK, PROXY_CONTAINER],
        ["docker", "start", PROXY_CONTAINER],
    ]
    print(f"Qualified card plan: device={DEVICE} profile={MODEL_PROFILE}")
    print(f"image: {IMAGE}")
    print(f"model: {REPOSITORY}@{REVISION}")
    print(f"served identity: {MODEL_ID} max_model_len={CONTEXT_LENGTH}")
    print(f"model files: {len(FILES)} pinned size/SHA-256 identities")
    for filename, (size, digest) in FILES.items():
        print(f"  {filename}  {size}  sha256:{digest}")
    print(f"XPU patch: {XPU_EXPECTED_SHA256} -> {XPU_PATCHED_SHA256}")
    print(f"MTP patch: {MTP_EXPECTED_SHA256} -> {MTP_PATCHED_SHA256}")
    print("Fresh-deployment Docker operations (existing objects are inspected and reused only "
          "after exact drift checks):")
    for command in commands:
        print(shlex.join(command))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "launch", "health"):
        child = subparsers.add_parser(action)
        child.add_argument("device")
        child.add_argument("profile")
    child = subparsers.add_parser("logs")
    child.add_argument("device")
    child.add_argument("profile")
    child.add_argument("-f", "--follow", action="store_true")
    child = subparsers.add_parser("stop")
    child.add_argument("device")
    child.add_argument("profile")
    child.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.device.casefold() != DEVICE or args.profile != MODEL_PROFILE:
        parser.error(f"only DEVICE={DEVICE} PROFILE={MODEL_PROFILE} is qualified")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.action == "plan":
            plan()
        elif args.action == "launch":
            launch()
        elif args.action == "logs":
            logs(args.follow)
        elif args.action == "stop":
            if stop(args.yes) is False:
                return 0
        else:
            health()
    except (DeployError, OSError, ValueError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

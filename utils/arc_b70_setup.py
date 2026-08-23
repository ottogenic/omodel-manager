#!/usr/bin/env python3
"""Install, preflight, and run the pinned Arc Pro B70 llama.cpp baseline.

This is deliberately separate from omodel-manager's NVIDIA/Docker path. All
commands are argv lists, setup is a dry-run unless --apply is supplied, and the
default server bind is loopback-only.
"""

import argparse
import glob
import hashlib
import os
import platform
import pwd
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


B70_PCI_ID = "8086:e223"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_TAG = "b10425"
LLAMA_COMMIT = "3d93885352a0049c8388a0da0698ec1a69e60d90"
MODEL_REPO = "bartowski/Qwen3.8-27B-GGUF"
MODEL_REVISION = "f0eec4a4bb4975114a030d048952d83c0a53c034"
SERVED_MODEL = "qwen3.8-27b-arc-gguf"
OMIX_KEY_URL = "https://repositories.intel.com/gpu/intel-graphics.key"
OMIX_RELEASE = "0.3.0"
INTEL_KEY_FINGERPRINTS = {
    "E0258B57D9C442D5DB1855C271740E4DE392BFE3",
    "4E9EFCDEF82800256C1E7C64B02DB9BD8C321DCB",
}
DEFAULT_SOURCE_DIR = "~/.local/src/llama.cpp"
DEFAULT_MODEL_DIR = f"~/.cache/otools/models/{SERVED_MODEL}/{MODEL_REVISION}"
DEFAULT_UNIT = "omodel-arc-b70.service"
ONEAPI_COMPILER = "/opt/intel/oneapi/compiler/latest"
ONEAPI_SETVARS = "/opt/intel/oneapi/setvars.sh"
BUILD_DIR_NAME = f"build-arc-b70-{LLAMA_TAG}"

MODEL_FILES = {
    "Q4_K_S": {
        "filename": "Qwen3.8-27B-Q4_K_S.gguf",
        "size": 16_713_148_000,
        "sha256": "9282674b002aac8d9d5eda7f53f5114d7fc91725f5a6962a03738571afb2218d",
    },
    "Q5_K_M": {
        "filename": "Qwen3.8-27B-Q5_K_M.gguf",
        "size": 20_752_787_040,
        "sha256": "e731e180460b906f373294a4e2de10541e80ee676af7f8c949a84dbb6ed3caa8",
    },
}

PROFILES = {
    "production": {
        "quant": "Q4_K_S",
        "context": 262_144,
        "parallel": 1,
        "kv": "q8_0",
        "description": "full native context candidate; qualify on the physical B70",
    },
    "smoke": {
        "quant": "Q4_K_S",
        "context": 32_768,
        "parallel": 1,
        "kv": "f16",
        "description": "short-lived correctness and offload check",
    },
    "q5-experiment": {
        "quant": "Q5_K_M",
        "context": 262_144,
        "parallel": 1,
        "kv": "q8_0",
        "description": "quality experiment; may exceed practical VRAM headroom",
    },
}


def display_command(argv):
    return shlex.join([os.fspath(arg) for arg in argv])


def run(argv, *, cwd=None, env=None, capture=False):
    print(f"+ {display_command(argv)}", flush=True)
    return subprocess.run(
        [os.fspath(arg) for arg in argv], cwd=cwd, env=env, check=True,
        text=True, capture_output=capture,
    )


def parse_os_release(text):
    values = {}
    for line in text.splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def read_os_release(path="/etc/os-release"):
    try:
        with open(path, encoding="utf-8") as handle:
            return parse_os_release(handle.read())
    except OSError:
        return {}


def kernel_version(release):
    match = re.match(r"(\d+)\.(\d+)", release)
    return tuple(map(int, match.groups())) if match else (0, 0)


def command_output(argv, *, env=None):
    try:
        result = subprocess.run(argv, check=False, text=True, capture_output=True, env=env)
    except OSError as exc:
        return None, str(exc)
    output = (result.stdout + result.stderr).strip()
    return (output if result.returncode == 0 else None), output


def parse_b70_lspci(output):
    if not output:
        return None, None
    bdf_match = re.search(
        r"(?im)^([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]).*8086:e223", output
    )
    driver_match = re.search(r"(?im)^\s*Kernel driver in use:\s*(\S+)", output)
    return (
        bdf_match.group(1).lower() if bdf_match else None,
        driver_match.group(1) if driver_match else None,
    )


def add_check(checks, ok, name, detail, *, warning=False):
    checks.append({
        "status": "PASS" if ok else ("WARN" if warning else "FAIL"),
        "name": name,
        "detail": detail,
    })


def sysfs_value(bdf, name):
    try:
        return Path(f"/sys/bus/pci/devices/{bdf}/{name}").read_text().strip()
    except OSError:
        return None


def find_render_node(bdf):
    matches = glob.glob(f"/dev/dri/by-path/pci-{bdf}-render")
    if not matches:
        return None
    return os.path.realpath(matches[0])


def hardware_checks():
    checks = []
    release = read_os_release()
    os_ok = release.get("ID") == "ubuntu" and release.get("VERSION_ID") == "26.04"
    add_check(
        checks, os_ok, "Ubuntu 26.04",
        f"{release.get('PRETTY_NAME') or release.get('VERSION_ID') or 'unknown OS'}",
    )

    kernel = platform.release()
    add_check(checks, kernel_version(kernel) >= (6, 17), "kernel >= 6.17", kernel)

    output, error = command_output(["lspci", "-Dnnk", "-d", B70_PCI_ID])
    bdf, driver = parse_b70_lspci(output)
    add_check(checks, bool(bdf), "Arc Pro B70 PCI device", bdf or error or "not found")
    add_check(checks, driver == "xe", "xe kernel driver", driver or "not bound")
    if not bdf:
        return checks, None

    max_speed = sysfs_value(bdf, "max_link_speed")
    max_width = sysfs_value(bdf, "max_link_width")
    current_speed = sysfs_value(bdf, "current_link_speed")
    current_width = sysfs_value(bdf, "current_link_width")
    max_ok = max_width == "4" and bool(max_speed and "16.0 GT/s" in max_speed)
    add_check(
        checks, max_ok, "PCIe capability",
        f"max {max_speed or '?'} x{max_width or '?'} (expected 16.0 GT/s x4)",
    )
    current_ok = current_width == "4" and bool(current_speed and "16.0 GT/s" in current_speed)
    add_check(
        checks, current_ok, "PCIe current link",
        f"current {current_speed or '?'} x{current_width or '?'}; recheck under load",
        warning=True,
    )

    render = find_render_node(bdf)
    add_check(checks, bool(render), "stable DRM render node", render or "not found")
    if render:
        accessible = os.access(render, os.R_OK | os.W_OK)
        try:
            mode = stat.filemode(os.stat(render).st_mode)
        except OSError:
            mode = "unknown permissions"
        add_check(checks, accessible, "render-node access", f"{render} {mode}")
    return checks, bdf


def runtime_checks(llama_server):
    checks = []
    try:
        env = oneapi_environment()
    except RuntimeError as exc:
        add_check(checks, False, "oneAPI environment", str(exc))
        return checks
    add_check(checks, True, "oneAPI environment", ONEAPI_SETVARS)
    sycl_ls = os.path.join(ONEAPI_COMPILER, "bin", "sycl-ls")
    sycl, sycl_error = command_output(
        [sycl_ls if os.path.isfile(sycl_ls) else "sycl-ls"], env=env,
    )
    sycl_ok = bool(sycl and re.search(r"level[_ -]?zero", sycl, re.IGNORECASE))
    add_check(checks, sycl_ok, "Level Zero SYCL device", sycl or sycl_error or "not found")

    server = os.path.expanduser(llama_server)
    if not os.path.isfile(server):
        add_check(checks, False, "pinned llama-server", f"missing: {server}")
        return checks
    version, version_error = command_output([server, "--version"], env=env)
    version_ok = bool(version and (LLAMA_COMMIT in version or LLAMA_COMMIT[:8] in version))
    add_check(
        checks, version_ok, "llama.cpp pinned commit",
        version or version_error or f"expected {LLAMA_COMMIT}",
    )
    devices, device_error = command_output([server, "--list-devices"], env=env)
    identity_ok = bool(devices and re.search(r"(?i)SYCL0[^\n]*(?:B70|e223)", devices))
    add_check(checks, identity_ok, "llama.cpp SYCL0 is B70",
              devices or device_error or "SYCL0 B70 not listed")
    memory_values = []
    if devices:
        for amount, unit in re.findall(r"(?i)(\d+(?:\.\d+)?)\s*(GiB|MiB)", devices):
            gib = float(amount) if unit.lower() == "gib" else float(amount) / 1024
            memory_values.append(gib)
    memory_ok = any(28 <= value <= 34 for value in memory_values)
    detail = ", ".join(f"{value:.1f} GiB" for value in memory_values) or "memory not reported"
    add_check(checks, memory_ok, "B70 memory approximately 32 GiB", detail)
    return checks


def print_checks(checks):
    for check in checks:
        detail = check["detail"].replace("\n", " | ")
        print(f"{check['status']:4}  {check['name']}: {detail}")
    failures = sum(check["status"] == "FAIL" for check in checks)
    warnings = sum(check["status"] == "WARN" for check in checks)
    print(f"\n{len(checks) - failures - warnings} passed, {warnings} warnings, {failures} failed")
    return 1 if failures else 0


def omix_source(codename):
    return (
        "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] "
        f"https://repositories.intel.com/gpu/ubuntu {codename}/intel-omix/{OMIX_RELEASE} unified\n"
    )


def setup_commands(source_dir):
    source_dir = os.path.abspath(os.path.expanduser(source_dir))
    icx = os.path.join(ONEAPI_COMPILER, "bin", "icx")
    icpx = os.path.join(ONEAPI_COMPILER, "bin", "icpx")
    return [
        ["sudo", "apt-get", "update"],
        ["sudo", "apt-get", "install", "-y", "ca-certificates", "gnupg", "git", "cmake",
         "build-essential", "libssl-dev", "pciutils"],
        ["sudo", "apt-get", "update"],
        ["sudo", "apt-get", "install", "-y", "intel-omix", "intel-omix-dev", "clinfo"],
        ["git", "clone", LLAMA_REPO, source_dir],
        ["git", "-C", source_dir, "fetch", "origin", "tag", LLAMA_TAG],
        ["git", "-C", source_dir, "checkout", "--detach", LLAMA_COMMIT],
        [
            "cmake", "-S", source_dir, "-B", os.path.join(source_dir, BUILD_DIR_NAME),
            "-DGGML_SYCL=ON", "-DGGML_SYCL_F16=ON",
            "-DGGML_SYCL_HOST_MEM_FALLBACK=OFF",
            f"-DCMAKE_C_COMPILER={icx}", f"-DCMAKE_CXX_COMPILER={icpx}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", os.path.join(source_dir, BUILD_DIR_NAME), "--config", "Release", "-j",
         str(os.cpu_count() or 1)],
    ]


def ensure_supported_os(allow_unsupported=False):
    release = read_os_release()
    supported = release.get("ID") == "ubuntu" and release.get("VERSION_ID") == "26.04"
    if not supported and not allow_unsupported:
        name = release.get("PRETTY_NAME") or "unknown OS"
        raise RuntimeError(f"setup requires Ubuntu 26.04 (found {name}); use --allow-unsupported-os to override")
    codename = release.get("VERSION_CODENAME")
    if not codename:
        raise RuntimeError("VERSION_CODENAME is missing from /etc/os-release")
    return codename


def conflicting_intel_sources(codename, paths=None):
    if paths is None:
        paths = ["/etc/apt/sources.list"] + glob.glob("/etc/apt/sources.list.d/*")
    conflicts = []
    expected = " ".join(omix_source(codename).strip().lower().split())
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        except (OSError, UnicodeError):
            continue
        for number, line in enumerate(lines, 1):
            active = line.strip().lower()
            if not active or active.startswith("#") or "intel" not in active:
                continue
            normalized = " ".join(active.split())
            if normalized == expected:
                continue
            conflicts.append(f"{path}:{number}: {line.strip()}")
    return conflicts


def installed_intel_compute_packages():
    output, _ = command_output(["dpkg-query", "-W", "-f=${binary:Package}\n"])
    if not output:
        return []
    patterns = (
        "intel-omix", "intel-gpu", "intel-opencl", "intel-level-zero", "intel-ocloc",
        "intel-igc", "intel-oneapi", "level-zero", "libze", "libigd", "libigc", "libigdfcl",
        "libigfxcmrt", "libigsc", "libmetee", "libxpu", "igsc", "metee", "xpu",
    )
    return sorted({name for name in output.splitlines()
                   if name.lower().startswith(patterns)})


def primary_fingerprints(gpg_colons):
    fingerprints = set()
    awaiting = False
    for line in gpg_colons.splitlines():
        record = line.split(":", 1)[0]
        if record == "pub":
            awaiting = True
        elif record == "sub":
            awaiting = False
        elif record == "fpr" and awaiting:
            fields = line.split(":")
            if len(fields) > 9:
                fingerprints.add(fields[9].upper())
            awaiting = False
    return fingerprints


def install_omix(codename):
    run(["sudo", "apt-get", "update"])
    run(["sudo", "apt-get", "install", "-y", "ca-certificates", "gnupg", "git", "cmake",
         "build-essential", "libssl-dev", "pciutils"])
    with tempfile.TemporaryDirectory(prefix="arc-b70-") as temp_dir:
        key = os.path.join(temp_dir, "intel-graphics.key")
        keyring = os.path.join(temp_dir, "intel-graphics.gpg")
        source = os.path.join(temp_dir, "intel-gpu.list")
        urllib.request.urlretrieve(OMIX_KEY_URL, key)
        key_info = run(
            ["gpg", "--batch", "--show-keys", "--with-colons", key], capture=True,
        ).stdout
        fingerprints = primary_fingerprints(key_info)
        if fingerprints != INTEL_KEY_FINGERPRINTS:
            raise RuntimeError(
                f"downloaded Intel key fingerprints {sorted(fingerprints)} do not match "
                f"the reviewed bundle {sorted(INTEL_KEY_FINGERPRINTS)}"
            )
        run(["gpg", "--batch", "--yes", "--dearmor", "--output", keyring, key])
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(omix_source(codename))
        run(["sudo", "install", "-m", "0644", keyring, "/usr/share/keyrings/intel-graphics.gpg"])
        run(["sudo", "install", "-m", "0644", source,
             f"/etc/apt/sources.list.d/intel-gpu-{codename}.list"])
    run(["sudo", "apt-get", "update"])
    run(["sudo", "apt-get", "install", "-y", "intel-omix", "intel-omix-dev", "clinfo"])


def build_llama(source_dir):
    source = Path(source_dir).expanduser().resolve()
    if source.exists() and not (source / ".git").is_dir():
        raise RuntimeError(f"refusing to replace non-git directory: {source}")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", LLAMA_REPO, source])
    remote = run(["git", "remote", "get-url", "origin"], cwd=source, capture=True).stdout.strip()
    normalized_remote = remote.rstrip("/")
    normalized_repo = LLAMA_REPO.rstrip("/")
    if normalized_remote.endswith(".git"):
        normalized_remote = normalized_remote[:-4]
    if normalized_repo.endswith(".git"):
        normalized_repo = normalized_repo[:-4]
    if normalized_remote != normalized_repo:
        raise RuntimeError(f"unexpected llama.cpp origin: {remote}")
    dirty = run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=source,
                capture=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"refusing to build a modified llama.cpp checkout:\n{dirty}")
    run(["git", "fetch", "origin", "tag", LLAMA_TAG], cwd=source)
    tag_commit = run(["git", "rev-parse", f"{LLAMA_TAG}^{{commit}}"], cwd=source,
                     capture=True).stdout.strip()
    if tag_commit != LLAMA_COMMIT:
        raise RuntimeError(f"{LLAMA_TAG} resolved to {tag_commit}, expected {LLAMA_COMMIT}")
    run(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=source)
    build_dir = source / BUILD_DIR_NAME
    marker = build_dir / ".arc-b70-generated-build"
    if build_dir.exists():
        try:
            generated_for = marker.read_text(encoding="utf-8").strip()
        except OSError:
            generated_for = None
        if generated_for != LLAMA_COMMIT:
            raise RuntimeError(f"refusing to replace unrecognized build directory: {build_dir}")
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    marker.write_text(LLAMA_COMMIT + "\n", encoding="utf-8")
    env = oneapi_environment()
    icx = os.path.join(ONEAPI_COMPILER, "bin", "icx")
    icpx = os.path.join(ONEAPI_COMPILER, "bin", "icpx")
    run([
        "cmake", "-B", build_dir, "-DGGML_SYCL=ON", "-DGGML_SYCL_F16=ON",
        "-DGGML_SYCL_HOST_MEM_FALLBACK=OFF", f"-DCMAKE_C_COMPILER={icx}",
        f"-DCMAKE_CXX_COMPILER={icpx}", "-DCMAKE_BUILD_TYPE=Release",
    ], cwd=source, env=env)
    run(["cmake", "--build", build_dir, "--config", "Release", "-j",
         str(os.cpu_count() or 1)], cwd=source, env=env)


def model_path(model_dir, profile_name):
    quant = PROFILES[profile_name]["quant"]
    return os.path.join(os.path.abspath(os.path.expanduser(model_dir)), MODEL_FILES[quant]["filename"])


def verify_model(path, quant):
    expected = MODEL_FILES[quant]
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        profile = "q5-experiment" if quant == "Q5_K_M" else "production"
        raise RuntimeError(
            f"model is missing: {path}\nrun: python3 utils/arc_b70_setup.py "
            f"download --profile {profile}"
        ) from exc
    if size != expected["size"]:
        raise RuntimeError(f"model size mismatch for {path}: {size} != {expected['size']}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected["sha256"]:
        raise RuntimeError(f"model SHA-256 mismatch for {path}: {actual}")


def download_model(model_dir, profile_name):
    quant = PROFILES[profile_name]["quant"]
    expected = MODEL_FILES[quant]
    destination = Path(model_path(model_dir, profile_name))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Verifying existing {destination}")
        verify_model(destination, quant)
        return destination
    url = (f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/"
           f"{expected['filename']}?download=true")
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    downloaded = 0
    print(f"Downloading pinned {MODEL_REPO}@{MODEL_REVISION}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                print(f"  {downloaded / (1024 ** 3):.1f} / {expected['size'] / (1024 ** 3):.1f} GiB",
                      end="\r", flush=True)
        print()
        if downloaded != expected["size"] or digest.hexdigest() != expected["sha256"]:
            raise RuntimeError("downloaded model failed pinned size/SHA-256 verification")
        os.replace(partial, destination)
    except Exception:
        try:
            partial.unlink()
        except OSError:
            pass
        raise
    return destination


def server_argv(source_dir, model_dir, profile_name, host="127.0.0.1", port=8000,
                context=None, parallel=None):
    profile = PROFILES[profile_name]
    server = os.path.join(os.path.abspath(os.path.expanduser(source_dir)), BUILD_DIR_NAME, "bin",
                          "llama-server")
    context = profile["context"] if context is None else context
    parallel = profile["parallel"] if parallel is None else parallel
    if context < 1 or context > 262_144:
        raise ValueError("--context must be between 1 and 262144 total tokens")
    if parallel < 1:
        raise ValueError("--parallel must be at least 1")
    return [
        server,
        "--model", model_path(model_dir, profile_name),
        "--no-mmproj",
        "--device", "SYCL0",
        "--split-mode", "none",
        "--main-gpu", "0",
        "--gpu-layers", "all",
        "--fit", "off",
        "--ctx-size", str(context),
        "--parallel", str(parallel),
        "--flash-attn", "on",
        "--cache-type-k", profile["kv"],
        "--cache-type-v", profile["kv"],
        "--jinja",
        "--reasoning-format", "deepseek",
        "--alias", SERVED_MODEL,
        "--host", host,
        "--port", str(port),
    ]


def oneapi_environment():
    if not os.path.isfile(ONEAPI_SETVARS):
        raise RuntimeError(f"missing oneAPI environment script: {ONEAPI_SETVARS}")
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c",
         'source "$1" >/dev/null && env -0', "arc-b70-setvars", ONEAPI_SETVARS],
        check=False, capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"oneAPI setvars.sh failed: {detail or result.returncode}")
    env = {}
    for entry in result.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        env[os.fsdecode(key)] = os.fsdecode(value)
    env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"
    env["ZES_ENABLE_SYSMAN"] = "1"
    return env


def systemd_unit(source_dir, model_dir, profile, host, port, script_path=None, python_path=None):
    script = os.path.abspath(script_path or __file__)
    python = os.path.abspath(python_path or sys.executable)
    values = [script, python, os.path.abspath(os.path.expanduser(source_dir)),
              os.path.abspath(os.path.expanduser(model_dir))]
    if any("\n" in value or " " in value for value in values):
        raise ValueError("systemd paths containing spaces or newlines are not supported")
    command = [python, script, "serve", "--source-dir", values[2], "--model-dir", values[3],
               "--profile", profile, "--host", host, "--port", str(port)]
    return f"""[Unit]
Description=Qwen3.8-27B on Intel Arc Pro B70
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
Environment=ONEAPI_DEVICE_SELECTOR=level_zero:0
Environment=ZES_ENABLE_SYSMAN=1
ExecStart={display_command(command)}
Restart=on-failure
RestartSec=10
TimeoutStopSec=30

[Install]
WantedBy=default.target
"""


def cmd_preflight(args):
    checks = []
    if args.stage in ("hardware", "all"):
        hardware, _ = hardware_checks()
        checks.extend(hardware)
    if args.stage in ("runtime", "all"):
        checks.extend(runtime_checks(args.llama_server))
    return print_checks(checks)


def cmd_install(args):
    source = os.path.abspath(os.path.expanduser(args.source_dir))
    if not args.apply:
        print(f"Pinned OMIX release: {OMIX_RELEASE}")
        print(f"Pinned llama.cpp:    {LLAMA_TAG} ({LLAMA_COMMIT})")
        print(f"Source directory:    {source}\n")
        commands = setup_commands(source)
        for argv in commands[:2]:
            print(f"+ {display_command(argv)}")
        print(f"# Download {OMIX_KEY_URL} over HTTPS and dearmor it with gpg")
        print(f"# Require key fingerprints {', '.join(sorted(INTEL_KEY_FINGERPRINTS))}")
        print("# Install the key as /usr/share/keyrings/intel-graphics.gpg")
        print(f"# Write the Ubuntu codename's Intel OMIX {OMIX_RELEASE} apt source")
        for argv in commands[2:]:
            print(f"+ {display_command(argv)}")
        print("\nDry-run only. Re-run with --apply on the Ubuntu B70 host.")
        return 0
    codename = ensure_supported_os(args.allow_unsupported_os)
    conflicts = conflicting_intel_sources(codename)
    if conflicts and not args.allow_conflicting_intel:
        joined = "\n".join(conflicts)
        raise RuntimeError(
            "conflicting Intel apt sources found; use a clean OMIX host or review "
            f"with --allow-conflicting-intel:\n{joined}"
        )
    packages = installed_intel_compute_packages()
    if packages and not args.allow_existing_intel_packages:
        raise RuntimeError(
            "pre-existing Intel compute packages require manual review; "
            f"review with --allow-existing-intel-packages: {', '.join(packages)}"
        )
    install_omix(codename)
    build_llama(source)
    print("\nSetup complete. Reboot, then run:")
    server = os.path.join(source, BUILD_DIR_NAME, "bin", "llama-server")
    print(f"  {display_command([sys.executable, __file__, 'preflight', '--llama-server', server])}")
    return 0


def cmd_download(args):
    destination = download_model(args.model_dir, args.profile)
    print(f"Verified model: {destination}")
    return 0


def cmd_serve(args):
    argv = server_argv(
        args.source_dir, args.model_dir, args.profile, args.host, args.port,
        args.context, args.parallel,
    )
    profile = PROFILES[args.profile]
    print(f"Profile: {args.profile} ({profile['description']})", file=sys.stderr)
    if args.host != "127.0.0.1":
        print(f"WARNING: the unauthenticated API will bind to {args.host}", file=sys.stderr)
    if args.dry_run:
        quant = profile["quant"]
        expected = MODEL_FILES[quant]
        print(f"# Model pin: {MODEL_REPO}@{MODEL_REVISION} {expected['sha256']}")
        print(f"ONEAPI_DEVICE_SELECTOR=level_zero:0 ZES_ENABLE_SYSMAN=1 {display_command(argv)}")
        return 0
    path = model_path(args.model_dir, args.profile)
    print(f"Verifying pinned model: {path}", file=sys.stderr, flush=True)
    verify_model(path, profile["quant"])
    os.execvpe(argv[0], argv, oneapi_environment())


def cmd_systemd(args):
    unit = systemd_unit(args.source_dir, args.model_dir, args.profile, args.host, args.port)
    if not args.install:
        print(unit, end="")
        return 0
    profile = PROFILES[args.profile]
    verify_model(model_path(args.model_dir, args.profile), profile["quant"])
    account = pwd.getpwuid(os.getuid()).pw_name
    if os.getuid() == 0:
        raise RuntimeError("install the systemd user service as its non-root owner")
    unit_dir = Path("~/.config/systemd/user").expanduser()
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / DEFAULT_UNIT
    unit_path.write_text(unit, encoding="utf-8")
    os.chmod(unit_path, 0o644)
    run(["sudo", "loginctl", "enable-linger", account])
    run(["systemctl", "--user", "daemon-reload"])
    run(["systemctl", "--user", "enable", DEFAULT_UNIT])
    print(f"Installed {unit_path}")
    if args.start:
        run(["systemctl", "--user", "restart", DEFAULT_UNIT])
    else:
        print(f"Start after preflight with: systemctl --user start {DEFAULT_UNIT}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="check B70 hardware and SYCL runtime")
    preflight.add_argument("--stage", choices=("hardware", "runtime", "all"), default="all")
    preflight.add_argument(
        "--llama-server",
        default=os.path.expanduser(f"{DEFAULT_SOURCE_DIR}/{BUILD_DIR_NAME}/bin/llama-server"),
    )
    preflight.set_defaults(func=cmd_preflight)

    install = sub.add_parser("install", help="install pinned OMIX and build llama.cpp")
    install.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    install.add_argument("--apply", action="store_true", help="execute the displayed setup")
    install.add_argument("--allow-unsupported-os", action="store_true")
    install.add_argument("--allow-conflicting-intel", action="store_true")
    install.add_argument("--allow-existing-intel-packages", action="store_true")
    install.set_defaults(func=cmd_install)

    download = sub.add_parser("download", help="download and verify a pinned GGUF")
    download.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    download.add_argument("--profile", choices=tuple(PROFILES), default="production")
    download.set_defaults(func=cmd_download)

    serve = sub.add_parser("serve", help="run a reproducible Qwen3.8 profile")
    serve.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    serve.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    serve.add_argument("--profile", choices=tuple(PROFILES), default="production")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--context", type=int, help="override total server context")
    serve.add_argument("--parallel", type=int, help="override slot count")
    serve.add_argument("--dry-run", action="store_true")
    serve.set_defaults(func=cmd_serve)

    systemd = sub.add_parser("systemd", help="print or install a user service")
    systemd.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    systemd.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    systemd.add_argument("--profile", choices=tuple(PROFILES), default="production")
    systemd.add_argument("--host", default="127.0.0.1")
    systemd.add_argument("--port", type=int, default=8000)
    systemd.add_argument("--install", action="store_true")
    systemd.add_argument("--start", action="store_true", help="start after installing")
    systemd.set_defaults(func=cmd_systemd)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "start", False) and not args.install:
        raise SystemExit("--start requires --install")
    try:
        return args.func(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

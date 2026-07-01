# AGENTS.md — architecture & working notes for `omodel-manager`

This file is the machine-and-human oriented map of the codebase. Read it before
editing. It complements `README.md` (user-facing) — here we cover *how* the tool is
built and the invariants you must not break.

## What this is

A single-file, stdlib-only Python CLI (`omodel-manager`) that launches and manages
vLLM Docker containers from an editable JSON config (`model_manager.json`), locally or
on a remote GPU host over SSH. It is a *launcher/manager*, not an inference client.

This repo also **owns the generic, harness-agnostic per-model configs** in
`configs/` (capabilities + per-mode sampling + tuning READMEs). The manager only
**stores + validates** them; downstream adapters (omodel-wire → OpenCode, pi.dev,
Claude Code, …) consume them and render harness-specific output. Keep `configs/`
free of any harness-specific keys. See `configs/README.md` for the format.

**Constraints (do not violate):**
- **Standard library only.** No third-party imports, ever. Runs with a bare `python3`.
- **Single script + data files.** The tool is `omodel-manager`; `model_manager.json`
  holds launch profiles; `configs/*.md` hold the generic model configs.
- **No local shell.** Build a `docker` **argv list** and run via `subprocess.run([...])`
  (through `docker()`), never `shell=True`. Over SSH the argv is `shlex.quote`d into a
  single remote command — this is what keeps JSON args like `--speculative-config` and
  `--hf-overrides` intact.
- **Secrets never touch the repo or the displayed command.** Tokens/keys come from env
  or `~/.config/otools`; `format_run` masks `*TOKEN*/*KEY*/*SECRET*` values.
- **Runtime identifiers are stable (backward-compat).** The `otools.*` labels, the
  `otools-vllm-` container name prefix, the SSH key `~/.ssh/otools_model_manager_ed25519`,
  and the token store `~/.config/otools/hf_token` are load-bearing for existing
  deployments. Renaming any of them orphans running containers / installed keys. Don't.
- **Cross-platform paths.** Runs from WSL/Linux and Windows; use `os.path`/`expanduser`.

## Layout of `omodel-manager`

Top-to-bottom, the meaningful sections:

- **Constants & seed** — `__version__`, `_default_config_path()` (resolves `--config` /
  `$OMODEL_MANAGER_CONFIG` / legacy `$OTOOLS_MODEL_MANAGER_CONFIG` / sibling file),
  `HF_TOKEN_FILE`, label/name constants, `DEFAULT_IMAGE`, and **`DEFAULT_CONFIG`** — the
  embedded seed written to `model_manager.json` on first run. **Invariant:**
  `DEFAULT_CONFIG` must stay equal to `model_manager.json` (seed == file). Edit one → edit
  both (or regenerate with `config --init --force`). The test suite enforces this.
- **Config** — `load_config()` / `save_config()`, `_deep_merge()`, `resolve_entry()`
  (applies the `extends` chain), `merge_model()` (merges `defaults` under the resolved
  entry), `container_name()`.
- **Secrets** — `_stored_hf_token()` / `hf_token()` / `save_hf_token()`.
- **Assets** — `assets_dir()`, `asset_local_path()`, `download_file()`,
  `resolve_assets()` (fetch + cache locally), `push_assets_remote()` (scp to the box).
- **Remote (SSH)** — module global `REMOTE`, `ssh_opts()` (pins the dedicated key once it
  exists), `ssh_argv()`, `run_remote()`, `remote_home()` (caches remote `$HOME`),
  `host_path()` (expand `~`/`${PWD}` for local vs remote), `need_docker()`, and **`docker()`
  — THE single choke point** (local `docker …` or `ssh … 'docker …'`).
- **Build** — `build_run_argv()` (config → `docker run` argv), `format_run()` (readable,
  secret-masked, multi-line rendering for dry-run/echo).
- **Commands** — `cmd_home` (no-arg landing), `cmd_ps`, `cmd_models`/`list`, `cmd_launch`,
  `cmd_fetch`, `cmd_stop`, `cmd_logs`, `cmd_health`, `cmd_setup`, `cmd_config`,
  `cmd_install_aliases`. Helpers: `_label`, `_fmt_tokens`, `_suggest` (breadcrumbs),
  `resolve_target` (key ↔ container name, exact match).
- **CLI** — `main()` (argparse subparsers; `--version`; a shared `--remote` parent).

## Reference — Docker / vLLM / SSH (WebFetch pointer lines)

Editing profiles or launch behavior means knowing how Docker and vLLM interpret the
flags. Fetch these when you need current detail (they move; treat the live docs as
authoritative over this file). Paste a line to an AI tool or fetch it yourself:

- use webfetch to find documentation/info on topic **docker run flags (-d/-t/--rm/--name/--label/-e/-v/--gpus)**: https://docs.docker.com/reference/cli/docker/container/run/
- use webfetch to find documentation/info on topic **docker GPU access (--gpus, device requests)**: https://docs.docker.com/engine/containers/resource_constraints/#gpu
- use webfetch to find documentation/info on topic **NVIDIA Container Toolkit install + CDI (why modern Docker has no named "nvidia" runtime)**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- use webfetch to find documentation/info on topic **vLLM server / `vllm serve` CLI args**: https://docs.vllm.ai/en/latest/cli/serve/
- use webfetch to find documentation/info on topic **vLLM environment variables (VLLM_LOGGING_COLOR, VLLM_ATTENTION_BACKEND, VLLM_ALLOW_LONG_MAX_MODEL_LEN, VLLM_NVFP4_GEMM_BACKEND)**: https://docs.vllm.ai/en/stable/configuration/env_vars/
- use webfetch to find documentation/info on topic **vLLM reasoning outputs (reasoning_effort, chat_template_kwargs.enable_thinking)**: https://docs.vllm.ai/en/latest/features/reasoning_outputs/
- use webfetch to find documentation/info on topic **vLLM using Docker (image, entrypoint, args)**: https://docs.vllm.ai/en/stable/deployment/docker/
- use webfetch to find documentation/info on topic **vLLM quantization (NVFP4 / ModelOpt auto-detect, --quantization)**: https://docs.vllm.ai/en/latest/features/quantization/
- use webfetch to find documentation/info on topic **HuggingFace resolve URL pattern (for `assets` downloads: /resolve/main/<file>)**: https://huggingface.co/docs/hub/en/how-to-downstream
- use webfetch to find documentation/info on topic **OpenSSH IdentitiesOnly / -i key pinning (ssh_config)**: https://man.openbsd.org/ssh_config
- use webfetch to find documentation/info on topic **ssh-copy-id (installing the public key)**: https://man.openbsd.org/ssh-copy-id
- use webfetch to find documentation/info on topic **DOCKER_HOST=ssh:// & docker context (alternative remote transport)**: https://docs.docker.com/engine/manage-resources/contexts/

## Mechanisms & gotchas this tool handles (vetted in development)

- **`--rm` eats crash logs.** Detached launches use `--rm`, so a container that crashes
  during startup is auto-removed with its logs. `--keep` omits `--rm` so it persists for
  `logs`; `--foreground` streams the crash live. New profiles: always `--dry-run`, then
  launch and watch (`--keep`/`--foreground`) the first time.
- **`docker --filter name=^x$` is unreliable** — container names are stored with a leading
  `/`, so the `^` anchor misses. `resolve_target` lists names and matches exactly.
- **Colored logs need an env var, not a TTY.** Most output comes from the vLLM
  **EngineCore subprocess**, forwarded over pipes; a container TTY only reaches the main
  process. `VLLM_LOGGING_COLOR=1` (in `defaults.env`, inherited by the subprocess) forces
  color regardless. `-d` (no `-t`) is used to avoid storing control-char noise.
- **Bind-mount paths belong to the docker *daemon host*.** For remote, `~`/`hf_cache`
  resolve against the **remote** `$HOME` (`remote_home()`), and declared `assets` are
  downloaded locally then `scp`'d to the box.
- **HF_TOKEN delivery.** Locally it's `-e HF_TOKEN` (inherited — stays out of argv).
  Remotely it's forwarded by value (`-e HF_TOKEN=…`), which IS visible in `docker inspect`
  on the box — acceptable (you own the box), documented, and masked in `format_run`.
- **GPU-runtime check is multi-signal.** Modern Docker (25+/29) uses **CDI**, so
  `docker info`'s `.Runtimes` has no `nvidia`. `setup` passes if any of: registered nvidia
  runtime, `nvidia-ctk` present, or a CDI spec in `/etc/cdi`.
- **`extends` can't remove keys.** Deep-merge only overrides/adds. When one profile must
  *omit* something a sibling has (e.g. 512K drops `--speculative-config`), write two full
  profiles rather than extending.
- **`defaults` are additive.** `docker_flags`/`volumes`/`assets` concatenate; `env`/
  `vllm_args` merge (profile wins). A profile inherits `--privileged`, ulimits, and
  `--trust-remote-code` even if its source recipe omitted them (harmless, more permissive).
- **Ctrl-C never dumps a traceback** (global handler + a `logs` handler that reassures the
  container is still running).

## Verify against reality (don't assume)

- **vLLM flags are image/version-specific.** A working recipe on one image can reject a
  flag on another (a real crash we hit). Confirm a new profile with `launch --dry-run`,
  then `--keep`/`--foreground` and read the startup log before trusting it.
- **`--quantization` is usually auto-detected** for NVFP4/ModelOpt checkpoints — omit it
  unless a model needs it explicitly.
- **YaRN long-context extrapolates** past the trained window (e.g. 27B 512K, `factor 2.0`);
  memory is cheap but verify output quality at very long contexts.
- **`setup` does NOT auto-install nvidia-container-toolkit** — it checks and advises with
  the install link (distro-specific; left to the operator).

## How to extend

- **New model:** add a `models.<key>` entry (+ `usecase`) to `DEFAULT_CONFIG`, keep it in
  sync with `model_manager.json` (regenerate via `config --init --force`). Verify with
  `list`, `launch <key> --dry-run`, and the tests.
- **New config field:** thread it through `merge_model` (and `resolve_entry` if it should
  be inheritable via `extends`), then consume it in `build_run_argv`.
- **New command:** add a subparser in `main()`, a `cmd_*` that talks to Docker **only**
  via `docker()`, and end it with a `_suggest([...])` breadcrumb block.
- **New remote transport** (e.g. `DOCKER_HOST=ssh://`): everything funnels through
  `docker()` — wrap there; don't spread SSH logic into commands.

## Verify after changes

```bash
python3 -m py_compile omodel-manager     # must pass
python3 -m unittest                          # test suite (no docker, no network, fast)
python3 omodel-manager list               # profile table renders
python3 omodel-manager launch <profile> --dry-run   # exact docker command, masked
```

Prefer `--dry-run` while iterating. Tests must not launch containers, hit the network, or
write to `$HOME` dotfiles (`install-aliases`/`save_hf_token` write real files).

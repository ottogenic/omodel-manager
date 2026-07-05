---
name: code-changes
description: Make a code change to the omodel-manager script — the top-to-bottom file/section map, how to extend (new model, config field, command, or remote transport), and the verify-after-changes checks. Use before editing omodel-manager's Python.
---

`omodel-manager` is a single-file, stdlib-only Python CLI that launches and manages
vLLM Docker containers from an editable JSON config, locally or on a remote GPU host
over SSH. Use this skill before editing its Python.

## Layout of `omodel-manager`

Top-to-bottom, the meaningful sections:

- **Constants & seed** — `__version__`, `_default_config_path()` (resolves `--config` /
  `$OMODEL_MANAGER_CONFIG` / legacy `$OTOOLS_MODEL_MANAGER_CONFIG` / sibling file),
  `HF_TOKEN_FILE`, `HOSTS_FILE`, label/name constants, `DEFAULT_IMAGE`, and
  **`DEFAULT_CONFIG`** — the committed **source of truth** for curated profiles.
  `config --init` writes it to a LOCAL, **git-ignored** `model_manager.json` that the tool
  reads and you tune. The two are deliberately NOT kept in lockstep (see the config
  sandbox invariant below). `model_manager.json` is never committed.
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
  `cmd_pull`, `cmd_pull_status`, `cmd_fetch`, `cmd_stop`, `cmd_logs`, `cmd_health`,
  `cmd_install` (a.k.a. `setup`), `cmd_uninstall`, `cmd_config`, `cmd_shell_init` (a.k.a.
  `install-aliases`). Helpers: `_label`, `_fmt_tokens`, `_suggest` (breadcrumbs),
  `resolve_target` (key ↔ container name), `resolve_host` (alias → `user@host`),
  `load_hosts`/`save_hosts`/`host_targets`, and `_image_present`/`_launch_bg_command`/
  `_launch_status` (the non-blocking launch path).
- **CLI** — `main()` (argparse subparsers; `--version`; a shared `--host`/`--remote` parent).

## How to extend

- **New model:** prototype it in your **local `model_manager.json`** (`launch <key>
  --dry-run`, then launch to validate). Once proven and approved, **promote** it: add the
  `models.<key>` entry (+ `usecase`) to **`DEFAULT_CONFIG`** and commit. `config --init
  --force` regenerates the local JSON from the updated defaults. Verify with `list` and the
  tests. This is the **add-a-model** skill's flow — follow it for onboarding. Respect the
  config sandbox invariant below (`model_manager.json` is throwaway/local; `DEFAULT_CONFIG`
  is the vetted, committed default — they are meant to differ while you iterate).
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
write to `$HOME` dotfiles (`shell-init`/`save_hf_token` write real files).

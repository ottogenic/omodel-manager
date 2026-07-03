# omodel-manager

Manage your vLLM Docker containers from an editable config — locally or on a
remote GPU box over SSH. List curated model profiles, launch one, watch its
logs, health-check it, and stop it, all with copy-pasteable next-step hints.

Built for a DGX Spark setup but works against any host with Docker + NVIDIA GPUs.

- **Stdlib only.** No `pip install`, no dependencies — just Python 3.
- **Config-driven.** Curated launch profiles are the committed source of truth in the
  script's `DEFAULT_CONFIG`; `config --init` writes them to a **local, git-ignored**
  `model_manager.json` you freely tune (reset anytime with `config --init --force`).
- **Local or remote.** Run Docker here, or on a GPU box over SSH (`--host`); one
  `install` bootstraps the box and gives it a short alias (`dgx1`).

---

## Requirements

- Python 3.8+ (stdlib only)
- Local: Docker with the NVIDIA container runtime/CDI
- Remote: an `ssh` client locally; Docker on the remote (see `install`)

## Quick start

```bash
# See what's available (context / concurrency / use-case)
python omodel-manager list

# Run one locally
python omodel-manager launch qwen3.6-35b-nvfp4

# ...or on a remote box: bootstrap + name it once, then use the alias
python omodel-manager install otto@192.168.50.102 dgx1 --fix
python omodel-manager launch qwen3.6-35b-nvfp4 --host dgx1
python omodel-manager logs   qwen3.6-35b-nvfp4 --host dgx1 -f
python omodel-manager health qwen3.6-35b-nvfp4 --host dgx1
```

Run with no arguments for a status/home screen with suggested next steps. Every
command prints a **Next steps** block so you never have to memorize the vocabulary.

Optional: `python omodel-manager shell-init` adds an `omm` shell alias.

## Commands

| Command | What it does |
|---|---|
| `list` (alias `models`) | Table of profiles: context, concurrency, use-case |
| `launch <profile> [host\|id]` | Start a profile (detached) — optional positional targets a host by alias or `ps` **ID** (e.g. an idle row). `--dry-run`, `--foreground`, `--keep`, `--force`, `--wait`, `--local`. Uncached image → pulls in the background and returns immediately |
| `pull <profile>` | Pre-pull a profile's image so `launch` starts instantly |
| `pull-status <profile\|id>` | Progress of a backgrounded launch/pull |
| `logs <profile\|id> [-f]` | Show/follow a container's logs (Ctrl-C detaches cleanly) |
| `health [<profile\|id>]` | `GET /v1/models` on running containers |
| `ps [--all]` | Numbered (**ID**) rows: running containers **plus** every registered host, each `running`/`idle`/`pulling`/`unreachable` |
| `stop <profile\|id>` (alias `kill`) | Stop + remove a container (`-y` to skip confirm) |
| `fetch <profile>` | Pre-download a profile's declared assets |
| `install <user@ip> [alias] [--fix]` (alias `setup`) | Bootstrap a remote (SSH, docker, group, HF token, drop-caches sudo rule) + register it under an alias |
| `uninstall <alias\|host> [--purge]` | Unregister a host + revoke the otools key (`--purge` also drops docker-group/containers + drop-caches rule) |
| `config [--path/--init/--edit]` | Show/init/edit the config file |
| `shell-init` (alias `install-aliases`) | Add the `omm` shell alias |

`--host ALIAS|USER@HOST` runs any docker-touching command on that host over SSH —
an alias from `install` (e.g. `dgx1`) or a raw `user@ip`. (`--remote` is a legacy
alias for `--host`.) Set `defaults.remote` in the config to make it the default. If
any host is registered, `launch` won't silently run local — pick a `--host`, or pass
`--local` to force local.

**Reference a `ps` row by its ID** instead of typing a model name + `--host`. Run `ps`,
then `logs 2` / `stop 2` / `health 2` (or `--id 2`) — the ID resolves to that row's host
and container. Works on `logs`, `stop`/`kill`, `health`, and `pull-status`. And
`launch <profile> 3` launches onto the host at `ps` row 3 (e.g. an idle box).

**Every `launch` drops the host's OS page cache right before `docker run`** — the DGX
Spark / UMA false-OOM-&-freeze guard (vLLM #35313). There's no flag: `install` sets up a
scoped `NOPASSWD` sudo rule (`/usr/local/sbin/otools-drop-caches`) so it runs unattended,
and launch uses `sudo -n`, so a host that wasn't installed just warns and skips it rather
than ever prompting. See [SPARK_NOTES.md](SPARK_NOTES.md) for the hardware background.

## The config

The committed **source of truth** is `DEFAULT_CONFIG` inside the `omodel-manager` script.
`config --init` writes it to a **local, git-ignored `model_manager.json`** next to the
script (override with `--config PATH` or `$OMODEL_MANAGER_CONFIG`) — that file is your
editable sandbox and is what the tool reads. Tune it freely; `config --init --force`
resets it from the hardcoded defaults. **Promote** a vetted change by editing
`DEFAULT_CONFIG` and committing — never commit `model_manager.json`. Structure:

```jsonc
{
  "defaults": {                     // merged UNDER every profile
    "image": "...", "host": "0.0.0.0", "gpus": "all",
    "remote": null,                 // "user@host" to default all commands remote
    "hf_cache": "~/.cache/huggingface",
    "env": { "HF_TOKEN": null, "VLLM_LOGGING_COLOR": "1" },
    "docker_flags": ["--privileged", "--network", "host", ...],
    "vllm_args": { "trust-remote-code": true }
  },
  "models": {
    "my-model": {
      "image": "vllm/vllm-openai:v0.20.0",   // per-profile override
      "model": "org/Model", "served-model-name": "...", "port": 8000,
      "usecase": ["Coding", "Agentic"],
      "env": { "VLLM_X": "1" },
      "volumes": ["${PWD}/file:/app/file"],
      "assets": [{ "url": "https://.../parser.py", "container": "/app/parser.py" }],
      "vllm_args": {                          // true => bare flag; value => --flag value
        "max-model-len": 262144, "max-num-seqs": 2,
        "speculative-config": "{\"method\":\"mtp\",\"num_speculative_tokens\":3}"
      }
    },
    "my-model-long": { "extends": "my-model", "vllm_args": { "max-model-len": 1048576 } }
  }
}
```

- **`vllm_args`**: `true` renders a bare flag (`--trust-remote-code`); any other
  value renders `--flag value`. JSON-string values (e.g. `--speculative-config`,
  `--hf-overrides`) pass through as a single intact argument.
- **`extends`**: inherit another profile and override only what differs.
- **`assets`**: files auto-downloaded before launch (cached in `assets/<key>/`)
  and mounted; for remote launches they're `scp`'d to the box.
- **`env`**: `null`/`"inherit"` inherits from your shell; a value is passed explicitly.

## Remote, install & uninstall

`install <user@ip> [alias] --fix` bootstraps a box: generates a **dedicated** SSH key
(`~/.ssh/otools_model_manager_ed25519`, clearly named so it's easy to revoke),
installs it, installs Docker if missing, adds you to the `docker` group, checks
the NVIDIA driver + container runtime (CDI-aware), and prompts for an HF token if
none is set. Run it without `--fix` for a read-only status report. It also **registers**
the host — under `alias` (e.g. `dgx1`) — in `~/.config/otools/hosts`, **merging** into
that list so other hosts stay. Then `ps` fans across every host by default and
`--host dgx1` resolves the alias. A bare `user@host` line works too; the file is safe to
hand-edit.

`uninstall <alias|host>` reverses it: drops the host from the registry and revokes the
otools key from the remote's `authorized_keys`. It leaves Docker, the docker group, and
the shared local key alone by default; `--purge` additionally stops the box's otools
containers and removes its docker-group membership.

## HF token

Needed for gated models (e.g. `nvidia/*`). Provide via `$HF_TOKEN` or let
`install ... --fix` store it at `~/.config/otools/hf_token` (chmod 600). Launches
forward it automatically (inherited locally; by value to the remote). It's never
written into the repo or shown unmasked in dry-run output.

## Notes

- Containers are labeled `otools.manager=model_manager` and named
  `otools-vllm-<profile>` so the tool only ever touches its own.
- Profiles share port 8000 by default → run one at a time per box.
- `--version` prints the version; see `CHANGELOG.md`.

## Development

Stdlib only — no virtualenv needed. The test suite is fast and offline (it never
launches containers, hits the network, or writes to `$HOME`):

```bash
python3 -m py_compile omodel-manager
python3 -m unittest                    # runs test_omodel_manager.py
```

See `CONTRIBUTING.md` for workflow and `AGENTS.md` for architecture + the invariants
to preserve.

Proprietary — see `LICENSE`.

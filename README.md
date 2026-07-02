# omodel-manager

Manage your vLLM Docker containers from an editable config — locally or on a
remote GPU box over SSH. List curated model profiles, launch one, watch its
logs, health-check it, and stop it, all with copy-pasteable next-step hints.

Built for a DGX Spark setup but works against any host with Docker + NVIDIA GPUs.

- **Stdlib only.** No `pip install`, no dependencies — just Python 3.
- **Config-driven.** Curated launch profiles are the committed source of truth in the
  script's `DEFAULT_CONFIG`; `config --init` writes them to a **local, git-ignored**
  `model_manager.json` you freely tune (reset anytime with `config --init --force`).
- **Local or remote.** Run Docker here, or on a GPU box over SSH (`--remote`); one
  `setup` bootstraps the box.

---

## Requirements

- Python 3.8+ (stdlib only)
- Local: Docker with the NVIDIA container runtime/CDI
- Remote: an `ssh` client locally; Docker on the remote (see `setup`)

## Quick start

```bash
# See what's available (context / concurrency / use-case)
python omodel-manager list

# Run one locally
python omodel-manager launch qwen3.6-35b-nvfp4

# ...or on a remote box (bootstrap it first)
python omodel-manager setup otto@192.168.50.102 --fix
python omodel-manager launch qwen3.6-35b-nvfp4 --remote otto@192.168.50.102
python omodel-manager logs   qwen3.6-35b-nvfp4 --remote otto@192.168.50.102 -f
python omodel-manager health qwen3.6-35b-nvfp4 --remote otto@192.168.50.102
```

Run with no arguments for a status/home screen with suggested next steps. Every
command prints a **Next steps** block so you never have to memorize the vocabulary.

Optional: `python omodel-manager install-aliases` adds an `omm` shell alias.

## Commands

| Command | What it does |
|---|---|
| `list` (alias `models`) | Table of profiles: context, concurrency, use-case |
| `launch <profile>` | Start a profile (detached). `--dry-run`, `--foreground`, `--keep`, `--force` |
| `logs <profile> [-f]` | Show/follow a container's logs (Ctrl-C detaches cleanly) |
| `health [<profile>]` | `GET /v1/models` on running containers |
| `ps [--all]` | List running managed containers |
| `stop <profile>` (alias `kill`) | Stop + remove a container (`-y` to skip confirm) |
| `fetch <profile>` | Pre-download a profile's declared assets |
| `setup [host] [--fix]` | Check/bootstrap a remote (SSH, docker, group, HF token) |
| `config [--path/--init/--edit]` | Show/init/edit the config file |
| `install-aliases` | Add the `omm` shell alias |

`--remote USER@HOST` runs any docker-touching command on that host over SSH.
Set `defaults.remote` in the config to make it the default.

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

## Remote & setup

`setup [host] --fix` bootstraps a box: generates a **dedicated** SSH key
(`~/.ssh/otools_model_manager_ed25519`, clearly named so it's easy to revoke),
installs it, installs Docker if missing, adds you to the `docker` group, checks
the NVIDIA driver + container runtime (CDI-aware), and prompts for an HF token if
none is set. Run it without `--fix` for a read-only status report. Comma-separate several
hosts (`setup otto@a,otto@b`); reachable ones are saved to `~/.config/otools/hosts` so
`ps` fans across them by default (no flag).

## HF token

Needed for gated models (e.g. `nvidia/*`). Provide via `$HF_TOKEN` or let
`setup --fix` store it at `~/.config/otools/hf_token` (chmod 600). Launches
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

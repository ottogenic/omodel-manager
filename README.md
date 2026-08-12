# omodel-manager

Manage your vLLM Docker containers from an editable config — locally or on a
remote GPU box over SSH. List curated model profiles, launch one, watch its
logs, health-check it, and stop it, all with copy-pasteable next-step hints.

Built for a DGX Spark setup but works against any host with Docker + NVIDIA GPUs.

- **Stdlib only.** No `pip install`, no dependencies — just Python 3.
- **Config-driven.** Curated launch profiles are the committed source of truth in the
  script's `DEFAULT_CONFIG`; `config --init` writes them to a **local, git-ignored**
  `model_manager.json` you freely tune (after a `git pull`, `omm sync` refreshes it from
  the committed defaults — with a `.bak` of your old file).
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

# On the DGX itself: check/provision local prerequisites, then run one locally
python omodel-manager install --fix
python omodel-manager launch qwen3.6-35b-a3b-nvfp4

# ...or on a remote box: bootstrap + name it once, then use the alias
python omodel-manager install user@192.0.2.102 dgx1 --fix
python omodel-manager launch qwen3.6-35b-a3b-nvfp4 --host dgx1
python omodel-manager logs   qwen3.6-35b-a3b-nvfp4 --host dgx1 -f
python omodel-manager health qwen3.6-35b-a3b-nvfp4 --host dgx1
```

Run with no arguments for a status/home screen with suggested next steps. Every
command prints a **Next steps** block so you never have to memorize the vocabulary.

Optional: `python omodel-manager shell-init` adds an `omm` shell alias.

## Commands

| Command | What it does |
|---|---|
| `list` (alias `models`) | Table of profiles: context, concurrency, use-case |
| `launch <profile> [host]` | Start a profile (detached) — optional positional host (alias/user@ip) to launch on. `--dry-run`, `--foreground`, `--keep`, `--force`, `--wait`, `--local`. Uncached image → pulls in the background and returns immediately |
| `pull <profile>` | Pre-pull a profile's image so `launch` starts instantly |
| `pull-status <profile\|host>` | Progress of a backgrounded launch/pull |
| `logs <profile\|host> [-f]` | Show/follow a container's logs (Ctrl-C detaches cleanly) |
| `health [<profile\|host>]` | `GET /v1/models` on running containers |
| `cluster <subcommand>` | Register, preflight, prepare, launch, inspect, and stop one model across two DGX Sparks |
| `ps [--all]` | Running containers **plus** every registered host, each `running`/`idle`/`pulling`/`unreachable` |
| `stop <profile\|host>` (alias `kill`) | Stop + remove a container (`-y` to skip confirm) |
| `fetch <profile>` | Pre-download a profile's declared assets |
| `install [<user@ip> [alias]] [--fix]` (alias `setup`) | Bootstrap this machine when no host is given; otherwise bootstrap a remote + register it under an alias |
| `uninstall <alias\|host> [--purge]` | Unregister a host + revoke the otools key (`--purge` also drops docker-group/containers + drop-caches rule) |
| `sync` | Reset `model_manager.json` from the committed `DEFAULT_CONFIG` — run after `git pull` to pick up newly merged profiles (backs up a differing old file to `.bak`; pairs with `omw sync`) |
| `config [--path/--init/--edit]` | Show/init/edit the config file |
| `shell-init` (alias `install-aliases`) | Add the `omm` shell alias |

`--host ALIAS|USER@HOST` runs any docker-touching command on that host over SSH —
an alias from `install` (e.g. `dgx1`) or a raw `user@ip`. (`--remote` is a legacy
alias for `--host`.) Set `defaults.remote` in the config to make it the default. With
no host given, `launch` runs locally — when hosts are registered it prints a one-line
"Launching locally (registered hosts: …)" reminder so a forgotten host stays visible.
`--local` forces a local launch even when `defaults.remote` is set (combining it with
an explicit host is an error).

**Address a running model by its host** instead of typing a model name + `--host` — with
one model per box, the hostname is unambiguous (and stable). Run `ps`, then `logs dgx-2` /
`stop dgx-2` / `health dgx-2` — the host (an `install` alias, a `user@ip`, or a bare IP)
resolves to the single container on it. Works on `logs`, `stop`/`kill`, `health`, and
`pull-status`. And `launch <profile> dgx-1` launches onto that host.

**Every `launch` drops the host's OS page cache right before `docker run`** — the DGX
Spark / UMA false-OOM-&-freeze guard (vLLM #35313). There's no flag: `install` sets up a
scoped `NOPASSWD` sudo rule (`/usr/local/sbin/otools-drop-caches`) so it runs unattended,
and launch uses `sudo -n`, so a host that wasn't installed just warns and skips it rather
than ever prompting. See [SPARK_NOTES.md](SPARK_NOTES.md) for the hardware background.

## Two-Spark clusters

`cluster` is a separate lifecycle for models that require both Sparks. It never turns a
normal one-container profile into a distributed deployment. Cluster definitions live in
`~/.config/otools/clusters.json`; runtime state lives under
`~/.local/share/otools/clusters/`. Override those paths for automation with
`$OMODEL_MANAGER_CLUSTERS` and `$OMODEL_MANAGER_CLUSTER_DATA`.

Register the local Spark as the head and a previously installed host alias as the worker.
The Linux interface and UCX RDMA device are deliberately separate: confirm their mapping
with `ibdev2netdev` after connecting the approved QSFP112 cable.

```bash
# Example one-rail CX-7 definition; use the addresses configured on your fabric.
omm cluster add spark2 local dgx4 \
  --interface enP7s7 --ucx-device mlx5_0:1 \
  --head-ip 192.168.177.10 --worker-ip 192.168.177.11

# Safe before the cable is connected: checks SSH, GB10, drivers, Docker, and disk.
omm cluster preflight spark2 --management-only

# After cable, IP, MTU, and RoCE setup: also requires link, exact route, jumbo ping,
# and mlx5-to-netdev mapping. A route over Wi-Fi/Ethernet is a hard failure.
omm cluster preflight spark2

# Stop existing one-node inference first. Heavy preparation refuses busy nodes by default.
omm cluster prepare spark2 qwen3-235b-a22b-fp4 --build --weights
omm cluster launch spark2 qwen3-235b-a22b-fp4
omm cluster status spark2
omm cluster health spark2 qwen3-235b-a22b-fp4
omm cluster logs spark2 qwen3-235b-a22b-fp4 -f
omm cluster stop spark2 -y
```

The Qwen path uses official NVIDIA checkpoints pinned to exact Hugging Face revisions and
ARM64 TensorRT-LLM bases pinned by digest. The manager builds the small SSH/MPI derivatives
locally, records separate immutable build manifests per runtime, uses a deployment-specific
SSH key, and serves absolute snapshot paths. Base, Instruct-2507, and Thinking-2507 were all
physically validated on two GB10 Sparks on 2026-08-11 with tool loops and 12,515-token streams.
All three profiles use rc8 for NVIDIA's Blackwell CUTLASS TMA fix. Thinking additionally uses
an explicit pinned tokenizer, torch sampling, and a conservative 0.60 KV fraction.

DeepSeek V4 Flash 0731 uses the reviewed c8r full-source lane: orchestration commit
`46eb0fcbadf0e4e0be8838b18f6aa85087ed8839`, vLLM commit
`48bada6ea49ad7f3ecbe03128aa76562089c8b00`, the pinned 17-file gx10 overlay, native
FlashInfer `0.6.16.post3`, and the SM120-capable DeepGEMM pin. The multi-hour source build is
performed by that pinned kit. `cluster prepare ... --build` then certifies the resident image
configuration, rootfs layers, and installed runtime contents on both nodes before recording
role-specific image IDs. It also hashes all 74 model files (166,898,660,330 bytes) on each node
and requires parity. C8r uses dedicated `-c8r` vLLM, Triton, and TileLang caches; the reviewed
cand7 image remains available as `deepseek-v4-flash-0731-cand7` with its own cache roots. The
operator accepts `b12x`'s Apache-2.0 package metadata despite its missing bundled license file.
Launch still requires the QSFP/RoCE preflight, API health, warmup battery, and `NCCL NET/IB`
evidence or rolls both ranks back. The c8r lane passed those gates plus sustained local OpenCode
tool traffic and a 20K-token stream on the qualified pair. No community model image or quant is
substituted.

See [DUAL_SPARK_MODEL_RESEARCH.md](DUAL_SPARK_MODEL_RESEARCH.md) for model selection and
primary-source links.

## The config

The committed **source of truth** is `DEFAULT_CONFIG` inside the `omodel-manager` script.
`config --init` writes it to a **local, git-ignored `model_manager.json`** next to the
script (override with `--config PATH` or `$OMODEL_MANAGER_CONFIG`) — that file is your
editable sandbox and is what the tool reads. Tune it freely; **after every `git pull`, run
`omm sync`** to refresh it from the committed defaults (it backs up a differing old file
to `.bak` first; `config --init --force` is the older spelling of the same reset).
**Promote** a vetted change by editing `DEFAULT_CONFIG` and committing — never commit
`model_manager.json`. Structure:

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
      "tok_s": 38,                            // decode tok/s, 1 user @ ~50k ctx (from benchmark)
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
- **`tok_s`** (optional): recorded decode speed — **single user at the benchmark's default
  context (~50k)** — shown in the `Tk/s` column of `list`/`models` so you can compare how fast
  each model is. Populate it from the benchmark (the `add-a-model` / `benchmark-model` skills);
  unmeasured profiles show `—`.

## Install, remote & uninstall

`install --fix` bootstraps the machine where the command runs: it installs Docker if
missing, adds the current user to the `docker` group, checks the NVIDIA driver + container
runtime, configures the drop-caches sudo rule, and prompts for an HF token if none is set.
It skips SSH setup and does not add the local machine to the remote hosts registry. Run
without `--fix` for a read-only local status report.

`install <user@ip> [alias] --fix` bootstraps a remote box: generates a **dedicated** SSH key
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

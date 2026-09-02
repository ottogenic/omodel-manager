# omodel-manager

Manage your vLLM Docker containers from an editable config — locally or on a
remote GPU box over SSH. List curated model profiles, launch one, watch its
logs, health-check it, and stop it, all with copy-pasteable next-step hints.

Built for DGX Spark nodes and clusters, with a checked-in qualified Docker/vLLM
deployment for the Intel Arc Pro B70.

- **Stdlib only.** No `pip install`, no dependencies — just Python 3.
- **Config-driven.** Curated launch profiles are the committed source of truth in the
  script's `DEFAULT_CONFIG`; `config --init` writes them to a **local, git-ignored**
  `model_manager.json` you freely tune (after a `git pull`, `omm sync` refreshes it from
  the committed defaults — with a `.bak` of your old file).
- **Local or remote.** Choose `local` or a registered device alias; one `install`
  bootstraps a remote box and gives it a short name (`dgx1`).

---

## Requirements

- Python 3.11+ (stdlib only; model-config resolution uses `tomllib`)
- Local nodes: Docker with the NVIDIA container runtime/CDI
- B70 card: Docker, `/dev/dri/renderD128`, and the pinned model snapshot described
  in `notes/card/b70-qwen3.8-vllm.md`
- Remote: an `ssh` client locally; Docker and `curl` on the remote (see `install`)

## Quick start

```bash
# See the unified model and device inventories
python omodel-manager models
python omodel-manager devices

# On the DGX itself: check/provision local prerequisites, then run one locally
python omodel-manager install --fix
python omodel-manager plan local qwen3.6-35b-a3b-nvfp4
python omodel-manager launch local qwen3.6-35b-a3b-nvfp4

# ...or on a remote box: bootstrap + name it once, then use the alias
python omodel-manager install user@192.0.2.102 dgx1 --fix
python omodel-manager launch dgx1 qwen3.6-35b-a3b-nvfp4
python omodel-manager logs dgx1 -f
python omodel-manager health dgx1
```

Run with no arguments for a status/home screen with suggested next steps. Every
command prints a **Next steps** block so you never have to memorize the vocabulary.

Optional: `python omodel-manager shell-init` adds an `omm` shell alias.

## Card And eGPU Qualification

The qualified B70 Docker/vLLM deployment is checked in under `utils/card/` and
is managed through the same device-first lifecycle as node and cluster models.
Historical native-runtime, driver, and OCuLink qualification records are retained
under `notes/card/`; those non-vLLM paths are evidence, not normal deployments.

## Commands

| Command | What it does |
|---|---|
| `models [card\|node\|cluster]` (alias `list`) | Unified profile table, optionally filtered by compatible device kind |
| `devices [card\|node\|cluster]` | Built-in `local` and `b70`, registered hosts, configured clusters, and explicit device overrides |
| `launch [DEVICE] [MODEL]` | Device-first deterministic launch. Missing operands show concrete choices instead of an argparse error |
| `plan DEVICE MODEL` | Print the existing node/cluster dry-run or exact card deployment plan without changing deployment state |
| `pull <profile>` | Pre-pull a profile's image so `launch` starts instantly |
| `pull-status <profile\|host>` | Progress of a backgrounded launch/pull |
| `logs DEVICE [head\|worker] [-f]` | Show/follow the current deployment logs; cluster role defaults to `head` |
| `health DEVICE` | Check the current deployment on a device |
| `cluster <subcommand>` | Register, inspect, preflight, and prepare two-node clusters; lifecycle uses the top-level device commands |
| `ps [--all]` | Running containers plus every registered host, with host and cluster columns and each target `running`/`idle`/`pulling`/`unreachable` |
| `stop DEVICE` | Stop the device's current deployment (`-y` to skip confirmation) |
| `fetch <profile>` | Pre-download a profile's declared assets |
| `install [<user@ip> [alias]] [--fix] [--card b70]` (alias `setup`) | Bootstrap this machine; for remotes, register the host/card and discover unambiguous DGX cluster pairs |
| `uninstall <device\|alias\|host> [--purge]` | Remove an explicit device registration, or unregister a host + revoke its otools key (`--purge` is host-only) |
| `sync` | Reset `model_manager.json` from the committed `DEFAULT_CONFIG` — run after `git pull` to pick up newly merged profiles (backs up a differing old file to `.bak`; pairs with `omw sync`) |
| `config [--path/--init/--edit]` | Show/init/edit the config file |
| `shell-init` (alias `install-aliases`) | Add the `omm` shell alias |

Lifecycle commands resolve the device first and inspect that device's live Docker state over
SSH. `local` always means this machine, host aliases come from `install`, and cluster names
come from the cluster registry. Names are case-insensitive and globally unambiguous.
Machine-local extensions live in versioned `~/.config/otools/devices.json` (override with
`$OMODEL_MANAGER_DEVICES`). No caller-local deployment file is authoritative: any controller
with the same targets registered can launch, inspect, and stop their models independently.
Remote vLLM listeners remain loopback-only. Their published `base_url` uses the device/head
target and assumes the operator's Tailgate route exposes that port; `omm` does not create
tunnels or broaden the listener. Local node and card URLs remain on `127.0.0.1`.

Card lifecycle uses the checked-in stdlib helper `utils/card/deploy_b70_vllm.py`.
`omm plan b70 qwen3.8-27b-gptq-int4-b70` works offline without Docker or a B70;
launch verifies the pinned image, model files, isolation, and container identities.
`$OMODEL_MANAGER_CARD_HELPER` may override the helper with an executable.

**Every `launch` drops the host's OS page cache right before `docker run`** — the DGX
Spark / UMA false-OOM-&-freeze guard (vLLM #35313). There's no flag: `install` sets up a
scoped `NOPASSWD` sudo rule (`/usr/local/sbin/otools-drop-caches`) so it runs unattended,
and launch uses `sudo -n`, so a host that wasn't installed just warns and skips it rather
than ever prompting.

## Two-Spark clusters

`cluster` is a separate lifecycle for models that require both Sparks. It never turns a
normal one-container profile into a distributed deployment. Cluster definitions live in
`~/.config/otools/clusters.json`; runtime state lives under
`~/.local/share/otools/clusters/`. Override those paths for automation with
`$OMODEL_MANAGER_CLUSTERS` and `$OMODEL_MANAGER_CLUSTER_DATA`.

Remote `install` discovers cluster pairs directly from the registered Linux DGX hosts; it
does not read NVIDIA Sync or other state from the client machine. A host qualifies only
when it is an ARM64 GB10 with NVIDIA Sync's DGX-side netplan marker and active IPv4 RoCE
interfaces. The manager requires matching subnets, UCX mappings and MTU on both hosts,
then verifies every rail with bidirectional interface-bound pings and physical peer MACs.
After the second host is installed, an unambiguous pair is registered automatically as
`<head-alias>-<worker-alias>`. Existing pairs are left unchanged, and ambiguous topologies
are never guessed. Rename a stopped pair with `omm cluster rename OLD NEW`.

The normal new-cluster flow is therefore:

```bash
omm install otto@new-dgx-1 dgx-5 --fix
omm install otto@new-dgx-2 dgx-6 --fix
omm cluster rename dgx-5-dgx-6 studio  # optional
omm launch studio deepseek-v4-flash-0731
```

DeepSeek launch downloads missing pinned weights, reuses an available reviewed image, and
certifies image, runtime, and model content on both nodes.

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

# Optional explicit preparation; normal DeepSeek launch now ensures missing artifacts itself.
omm cluster prepare spark2 qwen3-235b-a22b-fp4 --build --weights
# Lifecycle is device-first and cluster names resolve case-insensitively.
omm plan spark2 qwen3-235b-a22b-fp4
omm launch spark2 qwen3-235b-a22b-fp4
omm health spark2
omm logs spark2 head -f
omm stop spark2 -y
```

Qwen3.8 Flash Next requires explicit preparation on first use:

```bash
omm cluster prepare CLUSTER qwen3.8-flash-next-fp8 --build --weights
omm launch CLUSTER qwen3.8-flash-next-fp8
```

Facade launches retain failed rank logs where the backend supports it; `ps` and
`stop` include retained containers.

Cluster profiles keep their immutable model and runtime identities in `omodel-manager`.
Model-specific build history and findings belong in `notes/<profile>.md`.

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
    "image": "...", "host": "127.0.0.1", "gpus": "all",
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
- **`host`** (bind address): defaults to **`127.0.0.1` — loopback only**, so a launch can
  never accidentally expose the endpoint to the LAN. The server is reachable only on the box
  it runs on (local clients, and `health`/benchmarks that run there or probe over SSH). Set
  `"host": "0.0.0.0"` on a profile to expose it to the LAN on purpose — e.g. when a client
  on another machine (OpenCode on your laptop) must reach it directly.
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
that list so other hosts stay. Then `ps` fans across every host by default and lifecycle
commands accept `dgx1` as the device. A bare `user@host` line works too; the file is safe to
hand-edit. After registration, `install` probes the registered DGXs directly and offers to
name a newly discovered two-node cluster once both members are present.

For a host with the qualified Intel Arc Pro B70, register both the host and its card on every
controller that should manage it:

```bash
omm install otto@otto-home otto-home --card b70 --fix
omm launch otto-home-b70 qwen3.8-27b-gptq-int4-b70
omm health otto-home-b70
omm stop otto-home-b70 -y
```

Remote card actions stage the checked-in manager bundle and run the normal qualified `b70`
lifecycle on that host. The serving proxy remains loopback-only; clients on another machine
still need the documented Tailgate route (or another deliberate route) for `otto-home:8000`.

`uninstall <alias|host>` reverses it: drops the host from the registry and revokes the
otools key from the remote's `authorized_keys`. It leaves Docker, the docker group, and
the shared local key alone by default; `--purge` additionally stops the box's otools
containers and removes its docker-group membership.

`uninstall <device>` removes only an explicit `devices.json` entry. For example,
`omm uninstall otto-home-b70` forgets that remote card while preserving the `otto-home`
host registration, SSH access, and running containers.

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

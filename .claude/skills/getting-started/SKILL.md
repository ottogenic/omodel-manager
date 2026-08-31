---
name: getting-started
description: End-to-end onboarding for the otools DGX stack - serve local models on a DGX host with omodel-manager (omm) and wire them into OpenCode Build/Plan with omodel-wire (omw). Use when a user is setting up from scratch or asks how to install, provision a host, launch a model, install OpenCode, sync models, or "help me get started / onboard / set this up".
---

# Getting started with the otools DGX stack

Two sibling tools, run in this order:

| Tool | Alias | Job |
| --- | --- | --- |
| **omodel-manager** | `omm` | Launches & manages vLLM model containers on your DGX host(s). |
| **omodel-wire** | `omw` | Discovers live models and configures OpenCode's native Build and Plan agents. |

`omw` reads `omm`'s model configs from the sibling `../omodel-manager/configs` (or `--configs` / `$OMODEL_CONFIGS`), so keep both repos checked out side by side.

## How to drive this guide (for the AI assistant)

- **Run the read-only / inspection commands yourself and show the output** - `omm list`, `omm ps`, `omm health ...`, `omw sync --dry-run`, and `omw verify`. These are safe.
- For anything that **touches a remote host, starts a container, or is interactive** — `omm install`, `omm launch`, the OpenCode installer, `shell-init` (edits their rc file) — **give a copy-paste block and offer to run it**, but let the user confirm first (these do real work / may prompt).
- The shell env is **WSL/Linux with `python3`** (there is no bare `python`). Use `python3` for the one-time bootstrap; after `shell-init` the `omm`/`omw` aliases work from anywhere.

---

## 1. Install the shell aliases (one-time)

Run each from inside its repo, then re-open the shell:

```bash
# in the omodel-manager repo
python3 omodel-manager shell-init      # installs the `omm` alias
# in the omodel-wire repo
python3 omodel-wire.py shell-init      # installs the `omw` alias
```

This appends an alias to your shell rc (`~/.bashrc` / `~/.zshrc` / fish config). Idempotent — safe to re-run. **Open a new shell** so `omm` and `omw` resolve.

---

## 2. Provision a DGX host (one-time per host)

`omm install` bootstraps a remote host over SSH — generates a dedicated key (`~/.ssh/otools_model_manager_ed25519`), installs Docker + checks the NVIDIA runtime/CDI, and **registers the host under a short alias** in `~/.config/otools/hosts`.

```bash
omm install user@<host-ip> <alias> --fix     # e.g. user@192.0.2.101 dgx1 --fix
```

- `<alias>` (e.g. `dgx1`) is the device name used as the first lifecycle operand.
- `--fix` remediates what's missing (installs Docker, adds you to the `docker` group, sets up the scoped sudo rule, and **prompts for a HuggingFace token if none is set** — see step 4).
- This is interactive and SSHes into the box — **the user should run it.** Offer the command; don't run it for them.

Check what's registered / reachable:

```bash
omm ps            # all registered hosts + any running containers (safe — run this to confirm)
```

---

## 3. Launch your first model + watch it load

Recommended first model: **`unsloth-qwen3-coder-next-fp8`** (a fast, capable coder).

```bash
omm plan   dgx1 unsloth-qwen3-coder-next-fp8     # inspect the launch without mutation
omm launch dgx1 unsloth-qwen3-coder-next-fp8     # start the model container
omm logs   dgx1 -f                               # follow startup logs (Ctrl-C detaches)
omm health dgx1                                  # confirm it is serving
```

- `launch` returns immediately if the image isn't cached yet (it pulls in the background — poll with `omm pull-status`), then starts serving. Watch `logs -f` until vLLM prints that it's ready.
- See every available model to launch: `omm list` (safe — run it to show the catalogue).
- `launch`/`logs` reach the DGX — offer the commands and run them **if the user is ready to start a container**; otherwise hand them the copy-paste.

---

## 4. HuggingFace token (optional)

Only needed for **gated** models (e.g. `nvidia/*`). The recommended `unsloth-qwen3-coder-next-fp8` does **not** need one.

- Provide it via `$HF_TOKEN`, or let `omm install … --fix` store it at `~/.config/otools/hf_token` (chmod 600).
- Launches forward it automatically. It's never written into the repo or shown in dry-run output.

```bash
# optional — only for gated models:
umask 077; printf %s '<hf_token>' > ~/.config/otools/hf_token
```

---

## 5. Install OpenCode

OpenCode is the agent runtime that `omw` configures. Install (recommended one-liner):

```bash
curl -fsSL https://opencode.ai/install | bash      # installs to ~/.opencode/bin
```

Alternatives: `npm install -g opencode-ai` (or `bun`/`pnpm`). Docs: <https://opencode.ai/docs/>. Re-open your shell so `opencode` is on `PATH`.

---

## 6. Sync OpenCode Build and Plan (omw)

With at least one model live (step 3) and OpenCode installed, wire it up:

```bash
omw sync         # probes your live endpoints, reads omm's configs, writes ~/.config/opencode/opencode.json
```

`omw sync` adds the live DGX providers, assigns model-specific `build` and `plan`
presets to OpenCode's native agents, and writes the sampling plugin for parameters
OpenCode cannot express directly. It does not create custom agents, prompts, or
workflows. Useful flags: `--dry-run`, `--build-model REF`, and `--plan-model REF`.

Confirm it worked (safe — run these and show output):

```bash
omw verify       # compare the live endpoint with its model declaration
opencode         # launch OpenCode; Tab switches native Build and Plan
```

---

## 7. Choose Build and Plan models

`omw sync` prints every discovered device-qualified model reference. Preview the
result first, then optionally choose different live models for Build and Plan:

```bash
omw sync --dry-run
omw sync --build-model otools-dgx1/model-id --plan-model otools-dgx2/model-id
```

Without explicit flags, a still-live existing selection is preserved; otherwise
the first live model with the corresponding TOML preset is selected.

---

## Quick recap (copy-paste the whole path)

```bash
# 1. aliases (once, then re-open shell)
python3 omodel-manager shell-init ; python3 omodel-wire.py shell-init
# 2. provision a host (once per host — user runs; interactive)
omm install user@<host-ip> dgx1 --fix
# 3. launch + watch
omm launch dgx1 unsloth-qwen3-coder-next-fp8
omm logs   dgx1 -f
# 4. install OpenCode
curl -fsSL https://opencode.ai/install | bash
# 5. wire it up + verify
omw sync ; omw verify ; opencode
```

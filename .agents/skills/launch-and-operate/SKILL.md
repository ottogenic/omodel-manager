---
name: launch-and-operate
description: Launch, inspect, and manage running vLLM containers on remote GPU hosts — pick an idle host with ps, target hosts explicitly, non-blocking cold launches, and address a container by its hostname (logs/health/stop/pull-status). Use for any launch or day-to-day host operation.
---

# Launch and operate remote vLLM hosts

Operational rules for launching and managing vLLM Docker containers on remote GPU hosts with `omodel-manager`. Follow these exactly.

## Use the tool's built-in commands, not manual SSH

When a task requires remote action, use `omodel-manager` subcommands (`ps`, `launch`, `logs`, `health`, `docker`, …) rather than `ssh`-ing into the box manually. The tool's `--host` flag (an alias from `install`, or `user@ip`) handles SSH, key pinning, and path resolution — manual SSH bypasses those and can produce inconsistent results.

## Always target a host explicitly

Pass `--host <alias>` on every `launch`/`ps`/`logs`/`health` that should run remotely. Don't rely on habit or launch locally by accident: if hosts are registered, `launch` refuses a host-less run and tells you to pick one (or pass `--local` to force local). Prefer aliases (`dgx1`) over raw `user@ip`.

## Find a free host with `ps`, don't grep the config

`ps` lists every registered host and marks each `running` / `idle` / `unreachable`. Read it to pick an idle box — do NOT open `model_manager.json` or the configs to hunt for a host address.

## Address a running model by its host, not a model name

With one model per box, a hostname uniquely (and *stably*) identifies the container. `logs dgx-2` / `stop dgx-2` / `health dgx-2` / `pull-status dgx-2` (an `install` alias, `user@ip`, or bare IP) resolve the single container on that host — no `--host` or model name needed. `launch <profile> dgx-1` launches onto that host. Prefer this over model names when two boxes run the same model. (Do NOT use positional numbers — there is no index; a hostname is the stable handle.)

## `launch` is non-blocking on a cold image

If the image isn't cached, `launch` starts the pull+run in the background and returns immediately (so the tool call won't time out). Then poll `pull-status <key> --host <alias>` until it reports the container started, and `health` for readiness. Use `launch --wait` only when you deliberately want it to block; `pull <key> --host <alias>` pre-caches an image.

Crash-log details (`--keep` to persist a crashed container's logs, `--foreground` to stream a crash live) live in the **edit-launch-profiles** skill.

## Hosts registry

`ps` and `--host` resolve against the hosts registered by `install` in `~/.config/otools/hosts` — one `alias<TAB>user@host` per line (a bare `user@host` is also valid). `install user@ip [alias]` bootstraps a box and **merges** it into the store (other hosts stay); `uninstall <alias|host>` drops it and revokes the otools key from the remote. These are machine-specific and are NOT stored in `model_manager.json` or `DEFAULT_CONFIG`.

---
name: launch-and-operate
description: Launch, inspect, and manage running vLLM deployments by device — pick an idle device with ps/devices, plan or launch a compatible model, and use device-first logs/health/stop. Use for any launch or day-to-day host operation.
---

# Launch and operate remote vLLM hosts

Operational rules for launching and managing vLLM Docker containers on remote GPU hosts with `omodel-manager`. Follow these exactly.

## Use the tool's built-in commands, not manual SSH

When a task requires remote action, use `omodel-manager` subcommands (`ps`, `devices`,
`plan`, `launch`, `logs`, `health`, `stop`) rather than `ssh`-ing into the box manually.
The device registry handles SSH aliases, key pinning, clusters, and local cards; manual SSH
bypasses those contracts and can produce inconsistent state.

## Always name the device

Use `launch DEVICE MODEL` and `plan DEVICE MODEL`. Use the same device name for
`logs DEVICE`, `health DEVICE`, and `stop DEVICE`; use `local` for the local node and
`b70` for the qualified local card, or `<host-alias>-b70` for a card registered with
`install HOST ALIAS --card b70`. Prefer installed aliases (`dgx1`) over raw targets.

## Find a free host with `ps`, don't grep the config

`ps` lists every registered host and marks each `running` / `idle` / `unreachable`. Read it to pick an idle box — do NOT open `model_manager.json` or the configs to hunt for a host address.

## Address a deployment by its device

One active deployment owns a device, and a cluster owns both member nodes. Use
`logs dgx-2`, `stop dgx-2`, or `health dgx-2`; do not address lifecycle commands by model.

## `launch` is non-blocking on a cold image

If a node image is not cached, `launch` starts the pull+run in the background and returns
immediately. Poll `health DEVICE` and inspect `logs DEVICE -f`; lifecycle infers the profile
from target-host pull state and Docker labels.

Node launches retain their container so startup crash logs remain inspectable.

## Hosts registry

`devices` merges built-ins, installed hosts, clusters, and
`~/.config/otools/devices.json`. `install user@ip [alias]` bootstraps and names a node;
add `--card b70` to register its qualified card as `<alias>-b70`;
`uninstall <alias|host>` removes it and revokes the dedicated key. These records are
machine-specific and are not stored in `model_manager.json` or `DEFAULT_CONFIG`.

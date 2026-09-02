# AGENTS.md — `omodel-manager`

Single-file, stdlib-only Python CLI (`omodel-manager`) that launches and manages vLLM
Docker containers from an editable JSON config, locally or on a remote GPU host over SSH.
A *launcher/manager*, not an inference client. It also **owns** the harness-agnostic
per-model configs in `configs/*.toml` — stored + validated here, consumed by downstream
adapters (omodel-wire → OpenCode, …).

This file is intentionally short: the invariants below bind **every** task; anything
task-specific lives in a **skill** (see the index at the bottom) that loads on demand.

## Invariants — never violate

- **Stdlib only.** No third-party imports, ever. Runs on a bare `python3`.
- **Single manager script + checked-in stdlib card helpers.** Tool is `omodel-manager`;
  `utils/card/*.py` contains the qualified B70 deployment modules; `model_manager.json`
  holds launch profiles; versioned `~/.config/otools/devices.json` extends the machine-local
  device inventory; `configs/**/*.toml` hold generic
  model configs — keep them **harness-agnostic** (no OpenCode/harness keys; see
  `configs/README.md`).
- **No local shell.** All Docker goes through the `docker()` choke point as an **argv list**
  (`subprocess.run([...])`), never `shell=True`. Over SSH the argv is `shlex.quote`d.
- **Secrets never touch the repo or the displayed command.** Tokens/keys come from env or
  `~/.config/otools`; `format_run` masks `*TOKEN*/*KEY*/*SECRET*`.
- **`otools.*` runtime identifiers are load-bearing** — the labels, the `otools-vllm-`
  container prefix, the SSH key `~/.ssh/otools_model_manager_ed25519`, and the token store
  `~/.config/otools/hf_token`. Renaming any orphans running containers / installed keys.
- **`model_manager.json` is a git-ignored SANDBOX; `DEFAULT_CONFIG` (a dict literal in the
  script) is the committed source of truth.** Edit the JSON to test freely; promote a change
  into `DEFAULT_CONFIG` **only after it's validated and approved**. Never commit the JSON,
  never edit `DEFAULT_CONFIG` just to test, never edit both for a change-in-progress.
- **Cross-platform paths** (WSL/Linux + Windows): use `os.path` / `expanduser`.
- **Public lifecycle is device-first.** Use `launch DEVICE MODEL`, `plan DEVICE MODEL`, and
  `logs`/`health`/`stop DEVICE`; adapt the existing node/cluster internals rather than duplicating them.
  Device names are globally unique case-insensitively, and one active deployment owns a device
  (a cluster also owns its member nodes).
- **Runtime state is authoritative.** Lifecycle and downstream discovery inspect registered hosts;
  never couple them to caller-local launch intent or a `deployments.json` file.

## Working agreement

- **Todo tracking.** For any task with >2 discrete steps, use your harness's todo tool
  (not a chat-markdown plan). One item `in_progress` at a time; mark `completed` as you go.
  A single-step task or a plain question needs no list.
- **Plan approval is for the whole plan.** After a "go"/"proceed", execute all remaining
  steps in sequence without stopping to re-ask between them; discover prerequisites yourself.
- **Always name the device for a real launch.** Use `launch local <profile>` locally or
  `launch <alias> <profile>` remotely; pick an idle device from `ps`/`devices`, don't grep
  the config. (Details in the `launch-and-operate` skill.)
- **Solo git by default.** This is a single-maintainer repo unless told otherwise. Work in the
  canonical checkout and publish directly to `main` when the maintainer requests publishing.
  Use a branch/worktree only when explicitly requested, when concurrent work is actually active,
  or when isolation protects a working version. Always start clean, stage **explicit paths**
  (never `git add -A`), and inspect status, diff, and recent history before committing or pushing.
- **Publishing is not complete until the current checkout is synchronized.** If work happened on a
  branch or worktree, integrate it into the canonical checkout, push the requested destination, and
  verify the current `HEAD`, local `main`, and `origin/main` agree. Test with
  `python3 ./omodel-manager …` before integration; verify the alias only afterward.

## Skills — load the one matching your task first

Skill bodies load on demand; load the match before you start (OpenCode also surfaces them
via the `skill` tool).

| To… | Skill |
| --- | --- |
| Add or update a model (research → profile + config → prove on hardware → promote) | **`add-a-model`** |
| Benchmark a model's tok/s + concurrency cost | **`benchmark-model`** |
| Launch / inspect / stop containers on remote hosts | **`launch-and-operate`** |
| Edit launch profiles or launch behavior (Docker/vLLM/SSH reference + vetted gotchas) | **`edit-launch-profiles`** |
| Make a code change (file layout, how to extend, checks to run) | **`code-changes`** |
| Review / approve / merge an explicitly requested PR | delegate to **`agent-review`** (see the note below the table) |

Other docs (read directly when relevant): `notes/README.md` (per-model build records),
`configs/README.md` (config format), `README.md` (user-facing).

**When a PR is explicitly requested**, reviews are handled by the **`agent-review`** subagent.
It checks the PR against `REVIEW.md`
(the repo's bar), returns an itemized list of issues + suggested fixes, and merges only when the
review is clean. **If you have the `task` tool, delegate the review to `agent-review`** (call it by
name — `@agent-review` is only for when a human types it) rather than reviewing inline; when it
reports issues, get them fixed and re-delegate (same `task_id`) to re-review.

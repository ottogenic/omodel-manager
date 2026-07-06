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
- **Single script + data files.** Tool is `omodel-manager`; `model_manager.json` holds
  launch profiles; `configs/*.toml` hold the generic model configs — keep them
  **harness-agnostic** (no OpenCode/harness keys; see `configs/README.md`).
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
- **You open PRs — you never merge, push to `main`, or approve.** (See the `open-a-pr` skill.)
- **Cross-platform paths** (WSL/Linux + Windows): use `os.path` / `expanduser`.

## Working agreement

- **Todo tracking.** For any task with >2 discrete steps, use your harness's todo tool
  (not a chat-markdown plan). One item `in_progress` at a time; mark `completed` as you go.
  A single-step task or a plain question needs no list.
- **Plan approval is for the whole plan.** After a "go"/"proceed", execute all remaining
  steps in sequence without stopping to re-ask between them; discover prerequisites yourself.
- **Always target a host explicitly** on remote `launch`/`ps`/`logs`/`health`
  (`--host <alias>`) — never launch locally by accident. Pick an idle box from `ps`, don't
  grep the config. (Details in the `launch-and-operate` skill.)
- **Parallel-safe git.** Other agents may share this repo. Work in your own `git worktree`,
  branch first (`git switch -c …`), stage **explicit paths** (never `git add -A`), and start
  from a clean tree — if `git status` shows work that isn't yours, stop and surface it. Test
  via `python3 ./omodel-manager …` from your checkout, **not** the `omm` alias (it runs the
  main clone + its config, not your branch). Full flow: the `open-a-pr` skill.

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
| Commit + open a pull request | **`open-a-pr`** |
| Review / approve / merge an open PR | delegate to **`agent-review`** (see the note below the table) |

Other docs (read directly when relevant): `SPARK_NOTES.md` (DGX Spark GB10/sm_121 hardware
traps), `configs/README.md` (config format), `README.md` (user-facing).

**PR reviews** are handled by the **`agent-review`** subagent — it checks the PR against `REVIEW.md`
(the repo's bar), returns an itemized list of issues + suggested fixes, and merges only when the
review is clean. **If you have the `task` tool, delegate the review to `agent-review`** (call it by
name — `@agent-review` is only for when a human types it) rather than reviewing inline; when it
reports issues, get them fixed and re-delegate (same `task_id`) to re-review.

# REVIEW.md — the code-review bar for omodel-manager

The repo-specific standard a pull request must meet to merge. The review **process** — how to
run a review, what to report, and when to merge — lives in the **`pr-review` skill**; this file
is only the bar the skill checks against.

## Checks (must pass)

    python3 -m py_compile omodel-manager
    python3 -m unittest           # test_omodel_manager.py + test_configs.py — offline, no docker/network

## Invariants — a diff that breaks any of these is NOT mergeable

- **Stdlib only** — no third-party imports.
- **Single script** — the tool stays in `omodel-manager` (+ `configs/*.toml`, `test_*.py`).
- **Tests stay offline** — no containers, no network, no `$HOME` dotfile writes.
- `model_manager.json` is a git-ignored sandbox; curated profiles live in `DEFAULT_CONFIG`
  (the committed source of truth). A PR must not commit `model_manager.json`.
- `configs/*.toml` stay harness-agnostic — no OpenCode/harness-specific keys. New models
  follow the `add-a-model` skill.
- All Docker access goes through `docker()` — argv lists, never `shell=True`.
- `otools.*` runtime identifiers (labels, container prefix, SSH key + token paths) are
  load-bearing — a PR must not rename them.
- LF endings; kebab-case CLI, snake_case importable Python (`test_*.py`).

## Mergeable when ALL hold

1. Both checks pass.
2. The diff does only what the PR claims — no unrelated churn.
3. No correctness bug (read the diff; green tests aren't enough).
4. `CHANGELOG.md [Unreleased]` updated for anything user-facing.
5. Doesn't touch `LICENSE`, `__version__` / tags, `.github/`, or `otools.*` identifiers
   without explicit user approval.

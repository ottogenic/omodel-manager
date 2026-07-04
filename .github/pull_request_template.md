<!-- Read AGENTS.md → "Contributing changes (for AI agents / local models)" before opening. -->

## What & why
<!-- One or two sentences: what this changes and why. Link the issue/task if any. -->

## Scope
- [ ] One concern; small, focused diff
- [ ] No unrelated reformatting or churn
- [ ] Does not touch `LICENSE` / `__version__` / release tags / `.github/` / `otools.*` identifiers (or explicitly approved)

## Tests (paste the output)
```
$ python -m py_compile omodel-manager
$ python -m unittest
...
```

## Checklist
- [ ] Stdlib only — no new third-party imports
- [ ] Offline tests only — no containers, no network, no `$HOME` writes
- [ ] `model_manager.json` not committed (sandbox); curated profiles go in `DEFAULT_CONFIG`
- [ ] `configs/*.toml` stay harness-agnostic (see `configs/README.md`)
- [ ] `CHANGELOG.md` `[Unreleased]` updated (if user-facing)
- [ ] Branch named `feat/|fix/|chore/…`, Conventional-Commit title

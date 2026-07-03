# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Documentation
- **Naming conventions codified** in `CONTRIBUTING.md`: kebab-case for the CLI surface
  (executable, subcommands, flags), snake_case for imported Python files (modules/tests) —
  the latter required, since `python -m unittest <name>` / `import` reject hyphenated
  module names. Documents why the existing split is intentional, not accidental.
- Corrected two stale `CONTRIBUTING.md` notes that still described the pre-refactor
  `model_manager.json` as committed and kept identical to `DEFAULT_CONFIG`; it is now the
  git-ignored local sandbox (source of truth is `DEFAULT_CONFIG`).

### Added
- **Remote UX overhaul.**
  - **`--host ALIAS|USER@HOST`** replaces `--remote` (kept as a hidden legacy alias) and
    resolves aliases from the hosts store.
  - **`install <user@ip> [alias]`** (renames `setup`, kept as an alias): bootstraps a box
    *and* registers it under an alias, **merging** into `~/.config/otools/hosts` (other
    hosts stay) rather than overwriting.
  - **`uninstall <alias|host>`** (`--purge`): unregister a host and revoke the otools key
    from its `authorized_keys`; `--purge` also stops its otools containers and drops the
    docker-group membership. The shared local key is left in place.
  - **Non-blocking launch:** `launch` pre-checks the image and, if uncached, pulls + runs
    in the **background** and returns immediately (so an agent's tool call won't time out).
    New **`pull`** (pre-cache an image) and **`pull-status`** (poll a backgrounded launch
    via `~/.config/otools/launch-<key>.log`) subcommands, plus **`launch --wait`** to pull
    inline and block.
  - **`launch --local` guard:** with hosts registered, a host-less `launch` is refused —
    pick a `--host`, or pass `--local` to force local — preventing accidental local runs.
  - **Alias-aware hosts store:** `~/.config/otools/hosts` now takes `alias<TAB>user@host`
    lines (a bare `user@host` still works); `resolve_host()` maps aliases everywhere.
  - **`shell-init`** renames `install-aliases` (kept as an alias) to disambiguate from the
    new host `install`.
- `configs/` — generic, **harness-agnostic** per-model configs (capabilities +
  per-mode `presets` sampling + a tuning README), one `.toml` per model keyed to a
  `model_manager.json` profile. This is the source of truth that downstream
  adapters consume (omodel-wire → OpenCode; pi.dev / Claude Code later). Manager
  only stores + validates them (`test_configs.py`). First config: `qwen3.6-35b-nvfp4`.
- **`qwen3.6-35b-a3b-fp8`** launch profile + `configs/qwen3.6-35b-a3b-fp8.toml`:
  FP8 (e4m3) Qwen3.6-35B-A3B (hybrid GDN/Mamba MoE, multimodal, reasoning, tools),
  proven end-to-end on DGX Spark. Sets **`VLLM_USE_DEEP_GEMM=0`** — REQUIRED on
  Blackwell/sm_121, where DeepGEMM's E8M0 FP8-MoE weight processing crashes at load
  (`Unknown SF transformation`) and drops accuracy ~4% (vLLM #37804); MoE then falls
  back to TRITON, the working backend on sm_121 (#43507). Uses
  `tool-call-parser qwen3_coder` (per the model card) and `served-model-name` in
  `vllm_args` so the served id matches the config key.
- `utils/benchmark_concurrent.py` — sweep concurrency (1..N parallel requests) against a
  live endpoint to find the `max-num-seqs` throughput sweet spot (documented in
  `ADD_A_MODEL.md` §6).
- **Automatic page-cache drop before every `launch`.** `launch` now runs
  `sync; echo 3 > /proc/sys/vm/drop_caches` on the target right before `docker run` —
  no flag, no config, it's just how launch works. On UMA boxes (GB10 / DGX Spark) vLLM's
  `cudaMemGetInfo` can't see reclaimable cache, so a warm cache reads as a false OOM at
  load or freezes the box (vLLM #35313); a clean cache restores the read. `install`
  provisions a **scoped `NOPASSWD` sudoers rule** (`/usr/local/sbin/otools-drop-caches`)
  and launch calls it via `sudo -n`, so it runs unattended and *never prompts* — a host
  that wasn't installed just warns and skips (best-effort, never aborts). Honored on both
  the inline and background launch paths (pull → drop → run, grouped `|| true`).
  `install` checks/repairs the rule (`--fix`); `uninstall --purge` removes it.
- **`utils/benchmark_concurrent.py --host`** — accepts an `omm install` **alias**
  (resolved via `~/.config/otools/hosts`), a `user@ip`, or a bare ip, matching
  `omm --host`. (`--remote` kept as a legacy alias.)
- **`gemma4-26b-a4b`** launch profile + `configs/gemma4-26b-a4b.toml`: the MoE sibling of
  the dense 31B (`nvidia/Gemma-4-26B-A4B-NVFP4`, 25.2B total / 3.8B active), multimodal,
  reasoning + tools, 256K context. Only 3.8B active params per decode step → **~52 tok/s
  on DGX Spark vs the dense 31B's ~7** (ai-muninn benchmark). Uses the `gemma4-cu130`
  image + `gemma4` parsers, omits `--quantization` (auto-detected, vLLM #40291) and
  kv-cache fp8, and sets **`VLLM_USE_FLASHINFER_MOE_FP4=0`** to force the working Marlin
  FP4 path on sm_121 (no native FP4 MoE kernels on GB10).
- **`SPARK_NOTES.md`** — DGX Spark (GB10/sm_121) hard-won notes: a traps-&-fixes table
  (UMA #35313, FP8-MoE #37804/#43507, NVFP4 Marlin #43906, Gemma no-`--quantization`
  #40291, model-specific fp8-KV #35577, …) and a **watch-list** of open threads to
  revisit when upstream lands fixes (Marlin-vs-CUTLASS NVFP4-MoE A/B, MTP depth,
  gpu-util headroom, unvalidated tuning). `ADD_A_MODEL.md` and model `notes` link to it
  instead of re-listing the traps.
- `ADD_A_MODEL.md` — a **Tools & roles** note (document tools for research/authoring +
  `omm` for hardware; never raw `ssh`), **parallel-vs-serial** guidance (fan out §1
  research; keep §4–§6 on-hardware strictly serial) with tagged checklist items, and
  corrections from the DGX review (rolling `:nightly` over pins, validated env vars,
  auto-detected `--quantization`, model-specific kv-cache, automatic page-cache drop,
  benchmark `--host`).

### Changed
- **Config split into source-of-truth (code) + local sandbox (JSON).** `DEFAULT_CONFIG` in
  the script is the committed source of truth; `model_manager.json` is now a LOCAL,
  **git-ignored** file `config --init` generates from it. Tune/test in the JSON freely;
  promote validated changes into `DEFAULT_CONFIG` and commit. `config --init --force` resets
  the local file from the hardcoded defaults. The old "seed == file" invariant is gone.
- **`ps` now shows the whole fleet.** It lists every registered host, marking each
  `running` / `idle` / `pulling` / `unreachable` — a box with a background image pull in
  progress reads `pulling image` (not `idle`), so you don't double-launch on it. Rows are
  labeled by alias and the `NAME` column shows the short profile key (the `otools-vllm-`
  prefix is stripped for display; the container name is unchanged). Hosts are registered by
  `install` (was `setup`) in `~/.config/otools/hosts` instead of `config.defaults.remotes`.

### Removed
- `defaults.remotes` from the config — host lists are machine-specific and now live in
  `~/.config/otools/hosts` (managed by `install`).
- The `test_seed_equals_committed_file` test — `model_manager.json` is git-ignored, so
  there is no committed copy to match.

### Fixed
- `configs/qwen3.6-35b-nvfp4.toml`: `vision` corrected `false` → the multimodal
  modalities table `{ input = ["text","image"], output = ["text"] }` (verified live:
  4×4 blue PNG → "Blue"). Qwen3.6-35B-A3B is image-text-to-text, not text-only.
- `reason` preset (both `qwen3.6-35b-a3b-fp8` and `qwen3.6-35b-nvfp4`): corrected to
  the model card's "Thinking / General" recommendation — `temperature 0.6 → 1.0`,
  `presence_penalty 0.8 → 1.5`, `max_output 16384 → 32768`. The prior values were a
  precise-coding holdover; general reasoning/research runs hotter with a stronger
  anti-loop penalty. (`code`/`agent`/`instruct` already matched the card.)

## [0.1.0] - 2026-06-30

Initial packaged release. Extracted from the `otools` suite and renamed to
`omodel-manager`.

### Added
- Single-file, stdlib-only vLLM Docker manager: `list`/`launch`/`logs`/`health`/
  `ps`/`stop`/`fetch`/`setup`/`config`/`install-aliases`.
- Editable `model_manager.json`: `defaults` + per-profile `vllm_args`, `env`,
  `volumes`, `assets`, and `extends` (profile inheritance).
- Curated profiles: `qwen3.6-35b-nvfp4`, `nemotron-3-super-120b-nvfp4-256k`,
  `nemotron-3-super-120b-nvfp4-1m`, `qwen3.6-27b-nvfp4-256k`,
  `qwen3.6-27b-nvfp4-512k`, `glm-4.7-flash`.
- Remote execution over SSH (`--remote` / `defaults.remote` / per-model `remote`):
  paths resolve against the remote `$HOME`, assets are `scp`'d over, `HF_TOKEN`
  forwarded, `health` targets the remote host.
- `setup` bootstrap: dedicated revocable SSH key, docker install, docker group,
  CDI-aware GPU-runtime check, HF-token prompt/store.
- Auto-downloaded `assets` (e.g. custom reasoning-parser plugins), cached and mounted.
- UX: no-arg home screen, contextual "Next steps" breadcrumbs, readable
  secret-masked dry-run output, clean Ctrl-C, colored logs
  (`VLLM_LOGGING_COLOR=1`), `--keep` for post-mortem on crashed containers.
- `__version__` + `--version`; `install-aliases` (installs the `omm` alias).
- Offline test suite (`python3 -m unittest`).

[Unreleased]: https://example.invalid/omodel-manager/compare/v0.1.0...HEAD
[0.1.0]: https://example.invalid/omodel-manager/releases/tag/v0.1.0

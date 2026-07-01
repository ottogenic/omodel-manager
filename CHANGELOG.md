# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `configs/` — generic, **harness-agnostic** per-model configs (capabilities +
  per-mode `presets` sampling + a tuning README), one `.md` per model keyed to a
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

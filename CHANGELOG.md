# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **`utils/quality_eval.py` now accepts `omm install` aliases** (`dgx-3`), matching its own usage
  text and `benchmark_concurrent.py` — it previously skipped the shared hosts registry and failed
  with a DNS error on aliases. Also handles `alias:port`; raw `user@ip` / `ip` / `host:port`
  unchanged.

### Added
- **Pinned Laguna-S-2.1 RC2 target-only profile + separately validated DFlash variant.**
  `laguna-s-2.1-nvfp4` now serves Poolside's August 99.7 GB BF16-tail RC2 target pinned at
  `f8fdfcdc`, without speculative decoding; `laguna-dflash-s-2.1-nvfp4` adds the matched draft
  pinned at `b3b5921a` with seven proposals. Both use the immutable stock vLLM 0.25.1 image,
  auto-selected native `FLASHINFER_CUTLASS` MoE, explicit FP8 KV, one sequence, and independently
  validated context limits (229,376 target-only / 131,072 DFlash). On dgx4: target-only passed
  4/4 tool + 4/4 executable-code checks and measured 16.6 tok/s at ~55K; DFlash measured 18.4
  tok/s with workload-dependent acceptance (~18% prose, ~38% quality battery) and passed a repeat
  quality run. Harness configs now use RC2's authoritative 1.0 / 1.0 / top-k 20 sampling.
- **Targetless `install` now provisions the local machine.** On a DGX host itself,
  `omm install` runs the read-only prerequisite report and `omm install --fix` installs/repairs
  Docker, docker-group membership, the HF token, and the scoped drop-caches sudo rule directly.
  SSH setup and remote-host registration are skipped; explicit `install <user@host> [alias]`
  behavior is unchanged. No new mode flag was added.
- **`gemma4-26b-fp8-mtp`: coding-optimized MTP variant of `gemma4-26b-fp8`.** Same triton FP8 base
  + speculative decode (`num_speculative_tokens: 8`, drafter `google/gemma-4-26B-A4B-it-assistant`,
  native on the nightly — `Gemma4MTPModel`, no patch). **On-box dgx-3 2026-07-26:** 89% draft
  acceptance on code → **~54 tok/s @ ~41k code (1.5×)** and **~79 tok/s short (2.2×)** vs the
  non-MTP 36. Content-dependent: prose acceptance is ~46%, so prose@50k is a slight *loss* — hence
  a separate variant (use the base for prose/vision/general). Records `tok_s: 54` (code @ ~40k).
- **`north-mini-code-w4a16`: Cohere North-Mini-Code 1.0 (30B / 3B-active agentic-coding MoE) launch
  profile.** Uses the **official** `CohereLabs/North-Mini-Code-1.0-w4a16` (compressed-tensors
  nvfp4-pack W4A16, experts-only via QAD, ~18–20GB, HumanEval ~90.2 class) — chosen over the
  third-party `XanuNetworks/…-NVFP4` the Spark-Arena forum benchmarked, since W4A16 is the
  `--moe-backend marlin` dequant path we force on sm_121 anyway (so ~no speed lost) and the
  publisher is trusted. `trust-remote-code: False` — `config.json` has no `auto_map`, so vLLM loads
  the arch **natively** and no repo Python ever executes. Requires a **custom local image**
  (`otools/vllm-cohere-melody:nightly` = `vllm/vllm-openai:nightly` + `pip install
  "cohere_melody>=0.9.0"`): the `cohere_command4` tool/reasoning parsers ship in the `cohere_melody`
  plugin, not stock vLLM, and omm can't pip-install at launch — so bake it into the image. Also needs
  vLLM >= 0.21 / main for `Cohere2MoeForCausalLM` (the 0.23 nightly has it). GB10 config mirrors
  `gpt-oss-120b`: `--moe-backend marlin` + `VLLM_MARLIN_USE_ATOMIC_ADD=1` (avoid the sm_121
  CUTLASS-FP4-MoE `!!!!` garbage), bf16 KV, 256K context, `cohere_command4` parsers. **On-box
  2026-07-26 (dgx-3): config verified correct** (arch resolves natively with TRC off, MARLIN MoE
  selected, compressed-tensors auto-detected, weights download + melody image builds), **but it
  cannot load yet** — vLLM's FusedMoE loader throws `AttributeError: 'RoutedExperts' object has no
  attribute 'w2_bias'` (no Cohere2Moe per-expert-bias support in the `dev748` nightly). This is the
  upstream gap Cohere flags with "use vLLM main until a new release"; the profile carries a ⚠️ note
  and should not be launched until the melody image is rebuilt on a vLLM that supports Cohere2Moe
  expert biases.

### Changed
- **`gemma4-26b-fp8`: MoE backend `marlin` → `triton` (native FP8); benchmark now counts tokens,
  not chunks.** GB10 *has* native FP8 (unlike FP4), so forcing `--moe-backend marlin` was the wrong
  dequant path (it logged "no native FP8 support" and skipped the FP8 tensor cores). On-box dgx-3
  2026-07-26: triton and marlin measure identical (36 tok/s) @ ~53k — decode is KV-bound there —
  but triton is the correct native path (vLLM's own auto-pick) and wins under load. Confirmed
  `VLLM_USE_DEEP_GEMM=0` is a no-op for this per-channel FP8 (DeepGEMM never engages; it's only for
  block-quantized FP8), and corrected the misleading "native FP8 / FP4-MoE-unavailable" notes.
  **`benchmark_concurrent.py`**: `decode_tps` now uses `usage.completion_tokens` instead of
  SSE-chunk count — chunk-counting under-reported speculative-decode throughput by up to ~N×.
- **`gemma4-26b-a4b-nvfp4`: switched to the mainline `vllm/vllm-openai:nightly` image — now loads
  and works on GB10.** The pinned `gemma4-cu130` (vLLM 0.19.x) couldn't load the NVFP4 quant
  (per-expert scale `KeyError`, vLLM PR #41683); Gemma-4 is now native in mainline vLLM, and the
  nightly loads it cleanly. **Validated on-box (dgx-3, 2026-07-26):** `Gemma4ForConditionalGeneration`
  + MARLIN NvFp4 MoE, clean code generation + `gemma4` tool-calls (no `!!!!`), decode **28 tok/s @
  53k ctx** N=1 (TTFT 29 s / prefill 1,833 tok/s; ~52 tok/s at short ctx). Recorded `tok_s: 28`;
  dropped the stale ⚠️ launch-blocker note.
- **Laguna-S-2.1: `trust-remote-code` disabled; revision pin removed — now tracks latest.**
  Validated on-box 2026-07-26: vLLM v0.25.1 loads the architecture **natively** (no TRC needed;
  functional checks pass identically). With the repo's custom Python never executing, future
  poolside pushes can only alter *data* (weights/configs/template), so the security rationale for
  pinning evaporates — and poolside's promised "final checkpoint" looping fix will now be picked up
  automatically on the next container restart instead of requiring a manual re-audit-and-bump.
- **Laguna-S-2.1: sourced re-thinking-loop guidance + `preserve_thinking` for agent loops.**
  Research pass (official card/blog/generation_config + HF discussions, all linked in the TOML/notes)
  confirms: the identical sampling across presets **is** poolside's best practice (exactly one
  official rec — temp 0.7 / top_p 0.95 / top_k 20 — no coding-vs-reasoning split, no effort levels
  this release); mild "But wait…" re-verification is the model's intended style, while **hard loops
  are a staff-confirmed defect worst in 4-bit quants** — our pinned revision already is the 07-22
  fix re-quant (watch for the promised "final checkpoint"; re-audit + bump then). Config changes:
  the `agent` preset now sets `preserve_thinking = true` (official agentic rec: the model "may stop
  reasoning in follow-up steps if prior thinking blocks are dropped"); documented per-request
  `repetition_penalty 1.10–1.15` as the evidenced partial mitigation (deliberately not a default —
  it penalizes code tokens) and `thinking_token_budget` as a documented non-fix; noted the 07-21
  DFlash draft predates the re-quant target (collapsed acceptance = speed-only).

### Added
- **`omm sync`** — one command to refresh `model_manager.json` from the committed
  `DEFAULT_CONFIG` after a `git pull` (named to pair with `omw sync`). Newly merged profiles
  showed up in `DEFAULT_CONFIG` but not in `omm list` until you remembered
  `config --init --force`; `sync` is the discoverable spelling of that reset, prints which
  profiles were added/removed, and **backs up a differing old config to `.bak`** so local
  tweaks are never silently lost. (`config --init --force` still works.)
- **`laguna-s-2.1-nvfp4` launch profile + config** — poolside's Laguna-S-2.1 agentic coding model
  (118B total / ~8.5B active MoE), served from poolside's own NVFP4 quant (~72GB, the only variant
  that fits 128GB). Image pinned `v0.25.1` (laguna arch needs vLLM ≥ 0.25.0). DFlash speculative
  decoding on (`num_speculative_tokens 7` — output-identical, community-validated), `poolside_v1`
  tool/reasoning parsers, thinking on by default, 262144 context (poolside's quantized-quality
  ceiling). **Security-audited before download**: all-safetensors across base/NVFP4/DFlash repos;
  the trust-remote-code custom `.py` files read in full (transformers-only imports, no I/O/network/
  exec); verified poolside org; `--revision` pins the audited commit so future repo pushes can't
  inject new executable code. Validated on-box dgx-3 2026-07-23 (see profile notes for numbers).

### Documentation
- **Audited every externally-checkable DGX Spark claim in `SPARK_NOTES.md` and the `DEFAULT_CONFIG`
  model notes against upstream vLLM/GitHub, NVIDIA/vendor specs, and Docker Hub (2026-07-22), and
  corrected the wrong ones** — notes only, no launch behavior changed:
  - **Misattributed citations fixed:** `#43906` (cited for the NVFP4-MoE `!!!!` garbage — it's
    actually an MXFP8 issue), `#40291` (cited for a Gemma `--quantization` ValueError — it's an
    unrelated OOM bug), PR `#31740` (described as *merged* sm_121 support — still **open**; support
    ships via nightly/NGC builds), PR `#40708` ("≥ 0.19.1" is impossible — it merged *after* 0.19.1),
    and the FP8-MoE load crash (re-cited to `#47436`; the ~4% accuracy figure belongs to `#37804`).
  - **Facts corrected:** MSI **EdgeXpert** (was "EdgeExpert"); `:nightly-aarch64` is **not** a
    separate build lineage (identical digest to the arm64 half of `:nightly` as of 2026-07-22); the
    bandwidth roofline is a **ceiling** (measured decode ~35–70% of it); `VLLM_TEST_FORCE_FP8_MARLIN`
    affects the FP8 **linear/GEMM** path, not attention; `VLLM_MXFP4_BACKEND` is **unknown** on the
    0.23.x nightly (verified on-box during gpt-oss bring-up).
  - **New watch-list hooks:** native FP4 MoE kernels have landed (FlashInfer b12x, PR #40082) so the
    Marlin-vs-`auto` A/B is now live and overdue; `#35313` (the UMA false-OOM behind the pre-launch
    drop-caches guard) was closed 2026-04-13 → retest whether the guard is still needed; retest the
    modelopt-NVFP4 `!!!!` case with `VLLM_MARLIN_USE_ATOMIC_ADD=1` (its signature matches the
    documented Marlin-MoE atomic-add race, which that profile doesn't set).

### Changed
- **Quality-first rework of the FP8 profiles** (`qwen3-coder-next-fp8`, `unsloth-qwen3-coder-next-fp8`,
  `qwen3.6-27b-fp8-256k`; notes-only warning on `-512k`) — **validated on-box dgx-3 2026-07-23**:
  coder-next quality_eval 100% tool + 100% code (runs=2), functional PASS, log free of the
  uncalibrated-scale warnings, 36.6 tok/s @ ~54k (bf16-KV price ~13% vs 42) with TTFT/prefill
  massively improved (14.2s / 3,835 tok/s); 27b-256k functional PASS incl. reasoning, 6.5 tok/s
  @ ~54k (unchanged vs fp8 KV — the fix was free); 27b-512k READY + clean generation. The unsloth
  coder-next sibling carries the same config, not separately re-benchmarked:
  - **fp8 KV cache removed** on the three profiles: Qwen FP8 checkpoints ship no k/v/q scales, so
    fp8 KV ran **uncalibrated scale-1.0 fp8 attention** — vLLM's own startup log warns "may cause
    accuracy issues" (observed live on dgx-3). **Full 262144 context retained everywhere**: measured
    on-box, bf16 KV still yields a 1,050,664-token KV pool on the DeltaNet hybrid (only 12/48 layers
    hold KV) — ~4 full-context sequences; the dense 27B also fits full context in bf16.
    `qwen3.6-27b-fp8-512k` deliberately keeps fp8 KV (bf16 doesn't fit 524288) and now carries a
    "quality price of 512K + static-YaRN short-context cost" warning steering daily use to `-256k`.
  - **Prefix caching removed** on the coder-next hybrids: vLLM labels Mamba/GDN "align" mode
    experimental (live log), reuse on hybrids is ~zero, and state-corruption edge cases are reported.
  - **`--load-format fastsafetensors` removed** (load-time only; GDS unavailable on UMA; NGC
    ImportError reports) and **`max-num-batched-tokens 16384` added** on the coder-next pair (GDN
    mamba-align `block_size 2096 > 2048` guard, vLLM #36697; matches the 35B family's value).
  - **`VLLM_MARLIN_USE_ATOMIC_ADD=1` added** on the coder-next pair — inert on the log-confirmed
    TRITON FP8-MoE path, load-bearing if a future nightly auto-selects MARLIN (known `!!!!` race).
  - `tok_s` re-measured post-change per the re-measure rule: coder-next **37** (was 42), 27b-256k
    **7** (unchanged); unsloth sibling left unset pending its own run (old 46 preserved in notes).
    `SPARK_NOTES.md` fp8-KV row rewritten: fp8 KV is a capacity tool with a real quality cost, not
    a Qwen default.

### Changed
- **`launch` with no host now runs locally** — the registered-hosts guard (a hard error demanding
  `--host`/`--local`) is gone. On a GPU box itself, `omm launch <profile>` just works; when hosts
  are registered it prints a one-line "Launching locally (registered hosts: …)" reminder instead of
  erroring. Remote launch is unchanged (`launch <profile> <host>`, `--host`, or `defaults.remote`
  in the per-machine config). `--local` remains as the override for `defaults.remote` / a
  per-model `remote`; combining it with an explicit host is now an error.

### Fixed
- **`ps` now actually reports `unreachable`** — an SSH/docker failure used to fall through to
  `idle` (the unreachable branch was dead code), so a powered-off box looked free to launch on.
  `list_managed` now distinguishes "can't query the host" (`None`) from "nothing running" (`[]`).
- **Host-addressed `logs` no longer prints `omm logs None -f` in its Ctrl-C hints** — the
  suggestions echo the host (or model key) that addressed the container.
- **`logs`/`stop`/`health` honor a per-model `remote`** from the config, matching
  `launch`/`pull`/`pull-status`.
- Cosmetics: the container-already-exists error suggests `omm stop …` instead of the raw script
  path, and the `ps --hosts` help no longer claims omitting it queries local only.

### Added
- **`gpt-oss-120b` launch profile + config** — OpenAI's gpt-oss-120b (116.8B total / 5.1B active,
  native MXFP4 MoE; harmony reasoning + tool-calling, 131K context). Validated on-box dgx-3
  (GB10/sm_121) 2026-07-23: clean generation, reasoning-field separation + tool calls both work;
  **~31.5 tok/s decode @ ~49k ctx (N=1), ~17/req @ N=4**. Serves the **OpenAI** checkpoint, not
  Unsloth — Unsloth's Aug-2025 harmony/template fixes were merged upstream within days and its
  safetensors repo has an open vLLM tool-call failure report (`unslothai #5162`); weights are
  identical MXFP4. sm_121 specifics: `--moe-backend marlin` + `VLLM_MARLIN_USE_ATOMIC_ADD=1` (the
  stock CUTLASS/FlashInfer FP4 MoE path emits silent `!!!!`), `--attention-backend TRITON_ATTN`;
  the harmony `o200k_base.tiktoken` vocab is shipped as a mounted **asset** (the DGX can't reach
  `openaipublic.blob.core.windows.net`, so without it every chat request 500s with HarmonyError).
- **`qwen3.6-35b-a3b-nvfp4-unsloth` launch profile + config** — Unsloth's compressed-tensors
  NVFP4 (W4A4) quant of Qwen3.6-35B-A3B. Validated on DGX Spark (GB10/sm_121): 100% tool-call and
  100% code-quality over 10 runs each, no looping, ~59 tok/s single-user. Forces `--moe-backend
  marlin` (the non-Marlin FP4 MoE path garbages on sm_121; Unsloth's "don't use Marlin" is a B200
  claim that doesn't transfer). Recommended default for the 35B-A3B slot.
- **`utils/quality_eval.py`** — a stdlib-only, repeatable quality battery: a graded tool-call suite
  (selection/args/hallucination/loop-detection) and a code-quality suite (executable unit tests),
  run N times each and scored as pass-rates. Built for the 35B-A3B A/B; reusable for any model.

### Changed
- **Qwen3.6-35B-A3B profiles are now quality-first (MTP disabled).** `qwen3.6-35b-a3b-fp8` and the
  NVFP4 siblings now disable MTP speculative decoding: it reproducibly degenerates into
  repetition/garbage loops on this hybrid GDN/Mamba MoE (vLLM #47087/#47194), confirmed on-box. All
  three also add `--max-num-batched-tokens 16384` (QoS + clears the GDN mamba-align assertion).
  Anti-loop sampling stays harness-side in the config presets (card-recommended values).
- **Renamed `qwen3.6-35b-a3b-nvfp4` -> `qwen3.6-35b-a3b-nvfp4-nvidia`** (profile + config) to
  disambiguate the NVIDIA/modelopt checkpoint from the new Unsloth one and avoid served-id
  cross-matching. Marked BROKEN on the current nightly: it emits `!!!!` garbage despite a clean
  startup (NVFP4-MoE Marlin kernel miscompile for the modelopt checkpoint; the Unsloth checkpoint
  on the same nightly is clean).

### Added
- **`.claude/skills/getting-started` onboarding skill** — an end-to-end setup guide (shell aliases,
  DGX provisioning, launching a first model, the HF token, installing OpenCode, syncing + tweaking
  the roster) for Claude to walk a new user through, with copy-paste commands at each step.

### Fixed
- **PR-review tooling:** the `REVIEW.md` checks and the `pr-review` skill now use `python3` (the
  WSL env has no `python`), and the skill documents worktree-safe PR checkout plus a fallback for
  `gh pr merge`'s local post-merge error when `main` is checked out in a sibling worktree.

### Changed
- **`AGENTS.md` now tells agents to *delegate* PR reviews to `agent-review`** (via the `task` tool,
  by name) rather than reviewing inline — placed in `AGENTS.md` so it reaches every agent's context,
  including the prompt-free primaries. So "please review this PR" routes to `agent-review` from any agent.
- **PR-review workflow split: `REVIEW.md` is now just the repo's *bar*; the review *process* moved
  to a new `pr-review` skill.** `REVIEW.md` keeps only the checks, invariants, and merge conditions;
  the `pr-review` skill holds the process — review first, hand the parent agent an itemized list of
  issues + suggested fixes, and merge only when clean. (This is what `agent-review` runs.)

### Added
- **`gemma4-26b-fp8` launch profile** — RedHatAI's FP8 quantized Gemma 4 26B-A4B MoE variant.
  26B total / ~3.8B active per token, multimodal (text+image), reasoning + tool-calling.
  Measured ~37 tok/s decode single-user on dgx-1 (GB10, sm_121). At N=4 concurrency,
  decode speed drops to ~1.7-22 tok/s due to memory-bandwidth limitations.
  Uses `vllm/vllm-openai:nightly` (gemma4-cu130 lacks compressed-tensors MoE fixes).
- **`unsloth-qwen3-coder-next-fp8` launch profile** — the unsloth FP8 build of Qwen3-Coder-Next
  (hybrid GDN/DeltaNet MoE, ~80B total / ~3B active, 262K context) tuned for GB10/sm_121
  (`VLLM_USE_DEEP_GEMM=0` → TRITON FP8 MoE, FlashInfer attention, no `--quantization`). It shares
  the `qwen3-coder-next-fp8` config — that config's `match` now also covers the
  `unsloth/qwen3-coder-next-fp8` served id. Measured ~46 tok/s decode single-user on dgx-2.

### Changed
- **`git-new-worktree --delete` now also cleans up orphaned worktree folders.** If a folder is a
  sibling of the repo but git no longer tracks it as a worktree (its registration was pruned — e.g.
  by a cross-OS `git worktree prune`), or it's just a stray sibling directory, `--delete` offers to
  remove the folder after a clear "deletion is PERMANENT / can't verify unsaved work" warning.
  Restricted to siblings; refuses `.`/`..`.
- **New `git-sync-main` helper + renamed `new-worktree` → `git-new-worktree`.** `./git-sync-main`
  brings the current clone's `main` up to date with origin (fetch --prune → switch to main →
  fast-forward) — "make sure this is up to date." It **refuses inside a linked worktree and on a
  dirty tree**, so it never disturbs feature work. Both helpers now carry the `git-` prefix (also
  usable as `git new-worktree` / `git sync-main` when the repo is on `PATH`).
- **`./git-new-worktree` gains teardown**: `--delete <folder>` (aliases `--undo`/`--rm`) removes the
  worktree + its **local** branch only — **safe by default**: it never touches an open PR or the
  remote branch, so a submitted-but-unmerged PR (or a merged one) is untouched. `--abort <folder>`
  is the throw-it-all-away version — it also closes the open PR + deletes the remote branch. `-y`
  skips the prompt.
- **Genericized example host addresses/usernames** in help text, README, comments, and tests to
  the RFC 5737 documentation range (`192.0.2.0/24`) + a `user@` placeholder — no specific private
  LAN in the shipped code. Also ignore `hosts`/`wire.json` defensively and corrected a stale
  `.gitignore` note (`model_manager.json` is the git-ignored sandbox; `DEFAULT_CONFIG` is the
  committed source of truth).
- **Recorded `tok_s` (single-user decode speed) on every benchmarked launch profile** — from
  the overnight GB10 benchmarking run, committed to `DEFAULT_CONFIG` so they persist across
  `config --init` (previously they lived only in the local sandbox and were lost on a reset).
  `gemma4-26b-a4b-nvfp4` remains unmeasured.
- **`AGENTS.md` slimmed to invariants + a skill index; task detail moved to lazy-loaded
  skills.** The full `AGENTS.md` was injected into every model request (~5k tokens) even for
  trivial turns. It's now a lean always-on core (invariants + working agreement + a skill
  index); the workflows and reference material moved into OpenCode **skills** under
  `.agents/skills/` (the vendor-neutral discovery path): `add-a-model`, `benchmark-model`,
  `launch-and-operate`, `edit-launch-profiles`, `code-changes`, `open-a-pr`. Only each skill's
  name + description is advertised up front; the body loads on demand via the `skill` tool.
  `ADD_A_MODEL.md` moved into the `add-a-model` skill. Cuts per-request prompt overhead by
  ~4k tokens with no loss of guidance.

### Fixed
- **NVFP4 configs no longer cross-match their FP8 siblings in `omw models`.** The nvfp4
  configs carried quant-agnostic `match` patterns (`Qwen3.6-35B`, `Qwen3.6-27B`,
  `Qwen3-Coder-Next`, `Gemma-4-26B-A4B`) which — via omw's substring-based LIVE detection —
  matched the *fp8* siblings' served ids, so an nvfp4 model showed LIVE when only its fp8
  sibling was actually running. Tightened the four nvfp4 configs to precise, quant-specific
  patterns; added a `test_configs` guard that no config's pattern is a substring of another's.
- **`qwen3.6-35b-a3b-nvfp4`: force the sm_121 Marlin path + fix tool parser.** This profile was
  the only NVFP4/FP8 one with an empty `env` — it ran the FlashInfer/DeepGEMM FP8 kernels
  (`DeepGEMM E8M0 enabled`) that GB10/sm_121 mishandles (no native FP4), producing degenerate
  `!!!!` output under load. Added the FP8/DeepGEMM env its siblings carry
  (`VLLM_TEST_FORCE_FP8_MARLIN=1`, `VLLM_USE_DEEP_GEMM=0`) and pinned the MoE to Marlin via
  `--moe-backend marlin`. Also corrected `tool-call-parser` `qwen3_coder` → `qwen3_xml` (per the
  model card; `qwen3_coder` broke tool calls). The card omits these env vars because it targets
  datacenter Blackwell with native FP4 — they're required on the Spark.
- **`launch`/`health`/`pull-status`: clear startup state so a cached launch doesn't loop agents.**
  A fast (cached-image) `launch` returned before docker fully registered the container, and
  `health` then printed `not ready  Connection refused` — which reads like an *error* while the
  vLLM server is merely still loading — so agents polled/thrashed for 10+ minutes. Now: `launch`
  **settles ~5s** and verifies the container is running (catching an instant crash) before
  reporting success, and says plainly that the server is initializing and to poll `health`;
  **`health` distinguishes `STARTING`** (container up, server still loading — "NOT an error, keep
  polling") from a genuine error or a `DOWN` container; and **`pull-status`** on a directly
  (cached) launched model says "launched directly; container running — poll health" instead of
  the misleading "nothing is pulling".

### Changed
- **NVFP4 profiles: drop the deprecated `VLLM_NVFP4_GEMM_BACKEND` / `VLLM_USE_FLASHINFER_MOE_FP4`
  env vars; pin the MoE via the `--moe-backend marlin` flag instead.** Current vLLM nightlies
  (dev748+) log `Unknown vLLM environment variable` for both — they're superseded by the
  `--moe-backend` / `--linear-backend` CLI flags (vLLM DGX Spark blog). On GB10/sm_121 the correct
  split is **Marlin MoE + FlashInfer-CUTLASS linear (`auto`)**; forcing the *linear* GEMM to Marlin
  is wrong (empty/garbage output, per the ai-muninn Gemma writeup). Profiles touched:
  `qwen3.6-35b-a3b-nvfp4`, `nemotron-3-super-120b-a12b-nvfp4-256k` (+ `-1m`),
  `qwen3.6-27b-nvfp4-256k`, `qwen3.6-27b-nvfp4-512k`, `gemma4-26b-a4b-nvfp4`,
  `qwen3-coder-next-nvfp4`. The MoE profiles that already set `--moe-backend marlin` (35b,
  nemotron, coder-next) just shed the dead env vars — **no runtime change** on nightly;
  `gemma4-26b-a4b` **gains** `--moe-backend marlin` (its only backend pin had been the deprecated
  env var); the two **dense** `qwen3.6-27b-nvfp4-*` get no MoE flag (dense → the MoE var was
  always a no-op). Other env vars (`VLLM_TEST_FORCE_FP8_MARLIN`, `VLLM_USE_DEEP_GEMM`,
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN`, `VLLM_FLASHINFER_ALLREDUCE_BACKEND`, `VLLM_MARLIN_USE_ATOMIC_ADD`)
  are untouched.
- **`benchmark_concurrent.py` rewritten simple: `benchmark <endpoint> [N]`.** Replaces the
  growing-sessions / fixed-sweep / `--scenario` / `--quick` / `/metrics` machinery with one job:
  send **N unique ~50k-token prompts at once** (a generated story to summarize; unique so prefix
  caching can't skip the prefill) and report per request **TTFT** (input-processing time),
  **prefill tok/s**, and **decode tok/s**. `<endpoint>` is an alias / `user@ip` / `ip` / `host:port`;
  `N` (positional, default 1) is the number of simultaneous prompts. `--context` default **50000**
  (≈everyday coding; `--context 100000` for a big-repo stress). A small warm-up fires first so the
  numbers aren't a cold-start outlier and runs are comparable. Run `<host> 1` for single-user speed
  (→ `tok_s`), then `<host> 2`/`4` for the parallel slowdown.
- **Profile names standardized to upstream + `list` sorted.** Six keys renamed to match the
  published HF names (breaking — key = served-model-name = container name): `qwen3.6-35b-nvfp4`
  → `qwen3.6-35b-a3b-nvfp4`, `qwen3.6-35b-bf16` → `qwen3.6-35b-a3b-bf16`, both
  `nemotron-3-super-120b-nvfp4-*` → `…-120b-a12b-nvfp4-*`, `gemma4-31b-nvfp4` →
  `gemma4-31b-it-nvfp4`, `gemma4-26b-a4b` → `gemma4-26b-a4b-nvfp4`. Their `configs/*.toml` were
  renamed + `match` updated; `list`/`models` now sorts rows alphabetically so families group.
  **Migration:** relaunch any of these under the new name (`omm launch <new-key> --host …`) and
  re-run `omw --profiles` to re-sync agent providers.

### Added
- **Qwen3.6-27B-FP8** (`qwen3.6-27b-fp8-256k` / `-512k`): Official Qwen-team dense 27B FP8 (e4m3), 256K/512K context variants. Reasoning model (thinking ON by default), multimodal (vision input), tool-calling. No MoE → no VLLM_USE_DEEP_GEMM needed.
- **Qwen3-Coder-Next-FP8** (`qwen3-coder-next-fp8`): 80B/3B hybrid MoE (DeltaNet), FP8 quantized, 262K context, coding/agentic. No thinking mode. Requires `VLLM_USE_DEEP_GEMM=0` on GB10/sm_121.
- **`Tk/s` column in `list`/`models`.** Each profile can carry an optional `tok_s` — recorded
  decode speed at a **single user @ ~100k context** — so `omm models` shows at a glance how fast
  each model is. Populate it from the benchmark (`ADD_A_MODEL.md` §6 — `benchmark <host> 1`);
  unmeasured profiles show `—` (no guessed numbers).

## [0.2.0] - 2026-07-03

### Documentation
- **Naming conventions codified** in `CONTRIBUTING.md`: kebab-case for the CLI surface
  (executable, subcommands, flags), snake_case for imported Python files (modules/tests) —
  the latter required, since `python -m unittest <name>` / `import` reject hyphenated
  module names. Documents why the existing split is intentional, not accidental.
- Corrected two stale `CONTRIBUTING.md` notes that still described the pre-refactor
  `model_manager.json` as committed and kept identical to `DEFAULT_CONFIG`; it is now the
  git-ignored local sandbox (source of truth is `DEFAULT_CONFIG`).

### Added
- **New model: `qwen3-coder-next-nvfp4`.** RedHatAI NVFP4 quant of Qwen3-Coder-Next
  (hybrid DeltaNet MoE, ~79.7B total / ~3B active, 262K native context, pure text,
  `qwen3_coder` tool-call parser). Promoted into `DEFAULT_CONFIG` and shipped with a
  generic `configs/qwen3-coder-next-nvfp4.toml`. Validated on DGX Spark (GB10/sm_121):
  Marlin NVFP4 path, fp8 KV cache, clean growing-context benchmarks at 1-4 concurrent
  sessions (0 preemptions, KV peak 9% at 4 sessions).
- **Address a running model by its host.** `logs`, `stop`/`kill`, `health`, and `pull-status`
  accept a hostname (`omm logs dgx-2`) — an `install` alias, a `user@ip`, or a bare IP — and
  resolve the single container on that box (one model per host), so no `--host` or model name
  is needed. `launch <profile> dgx-1` takes a positional host to launch on. A hostname is a
  stable handle (unlike a transient `ps` row number) and disambiguates two boxes running the
  same model.
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
- `utils/benchmark_concurrent.py` — **generic** (config-free) throughput probe: point it at
  `--host` and it auto-discovers the served model from `/v1/models`, then drives a realistic
  growing-context load (concurrent multi-turn sessions to ~100k, streaming TTFT/TPOT, `/metrics`
  KV-pressure + preemptions); `--quick` runs a fast concurrency sweep. Documented in
  `ADD_A_MODEL.md` §6.
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
- **`benchmark_concurrent.py` now models a realistic growing-context load.** The default is
  **N concurrent multi-turn sessions whose context grows toward ~100k tokens** (unique code
  per turn, so prefix caching can't hide prefill), streaming **TTFT/TPOT bucketed by context
  size** and scraping the server's `/metrics` for **KV-cache pressure and preemptions** — this
  reproduces the real "two sessions doing real work crawl" that the old short identical-prompt
  sweep hid (prefix-cache hit + tiny KV → falsely flat). Small flag surface: `--sessions`,
  `--grow-to`, `--scenario coding|agent`, `--no-think`; the old sweep is preserved behind
  `--quick`. Thinking defaults **on** (representative for reasoning models). A non-zero
  `preemptions during run` pinpoints KV overflow as the cause of a slowdown.
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

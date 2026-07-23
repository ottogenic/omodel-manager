# DGX Spark (GB10 / sm_121) — hard-won notes

Hardware-specific traps and open threads for serving on the **MSI EdgeXpert DGX
Spark** (MS-C931): NVIDIA GB10 Grace-Blackwell, compute capability **sm_121** (not sm_120,
though the two are binary-compatible — sm_120 builds run),
**128 GB LPDDR5X *unified* memory** (CPU+GPU share it), ~**273 GB/s** bandwidth,
aarch64. This is the single source of truth for the gotchas that keep biting us —
model `notes` and the `add-a-model` skill link here instead of re-listing them.

**The one mental model that explains most of this:** decode on GB10 is
**memory-bandwidth-bound**, not compute-bound. tok/s ≈ 273 GB/s ÷ (bytes read per
token) ≈ 273 ÷ (active-params × bytes/weight). So **fewer active params and fewer
bytes/weight win**: NVFP4 ≫ FP8 ≫ BF16, MoE (few active experts) ≫ dense, and
speculative decoding (MTP) is a real multiplier. **Treat the formula as a ceiling, not a
prediction:** measured decode lands ~35–70% of it (dense-large models come closest; MoE and
small models sit lowest — e.g. gpt-oss-120b MXFP4 ~35–47% of roofline), with kernel efficiency
and attention/overhead eating the rest. Quality, though, moves the other way (4-bit experts
lose precision) — hence per-model, per-quant tuning.

---

## Traps & fixes

| Area | Symptom | Fix (what we do) | Source |
|------|---------|------------------|--------|
| **UMA memory mis-detection** | vLLM's `cudaMemGetInfo` can't see reclaimable page cache → false OOM at model load, or a full system freeze | **Every `launch` drops the page cache** (`sync; echo 3 > drop_caches`) right before `docker run` — automatic, set up by `install` (scoped NOPASSWD sudo). Also keep `gpu-memory-utilization` ≈ **0.85**. | vLLM #35313 (closed 2026-04-13 — retest) |
| **FP8-MoE (Qwen3.5/3.6)** | DeepGEMM E8M0 scale crash at load (`Unknown SF transformation`) on sm_121; separately, a ~4% accuracy drop | env **`VLLM_USE_DEEP_GEMM=0`** → MoE falls back to **TRITON** (the working backend on sm_121). CUTLASS MoE is *unavailable* on sm_121 — don't force it. | #47436 (sm_121 load crash), #37804 (the ~4% accuracy drop, on B200), #43507 (CUTLASS-MoE gap) |
| **NVFP4 / MXFP4 MoE** | Stock CUTLASS/FlashInfer FP4 MoE path emits **silent `!!!!` garbage** on sm_121 | Pin the MoE to the working **Marlin** dequant path via the **`--moe-backend marlin`** flag, **plus `VLLM_MARLIN_USE_ATOMIC_ADD=1`** (defuses a Marlin-MoE shared-memory race that *itself* prints `!!!!` at TP=1). Leave the FP4 **linear** GEMM on `auto` (→ FlashInfer CUTLASS, the correct sm_121 path — forcing Marlin there is a scale-bug/garbage risk, #34694). Deprecated env vars: `VLLM_NVFP4_GEMM_BACKEND` / `VLLM_USE_FLASHINFER_MOE_FP4` (and `VLLM_MXFP4_BACKEND` is simply **unknown** on the 0.23.x nightly — verified on-box) → use the `--moe-backend` / `--linear-backend` flags. Native FP4 MoE kernels *have* since landed (see watch-list). Separately, `VLLM_TEST_FORCE_FP8_MARLIN=1` routes the **FP8 linear/GEMM** path to Marlin (NOT attention, as earlier notes implied); its benefit on sm_121 is unverified. | ai-muninn/conselara sm121 writeups; vLLM #38718/#47365 (the `!!!!` garbage); #34694 (linear Marlin scale bug) |
| **NVFP4 MoE — Marlin not always enough (checkpoint×nightly)** | `nvidia/Qwen3.6-35B-A3B-NVFP4` (modelopt) emitted **pure `!!!!`** on nightly `v0.23.1rc1.dev748` **even with `--moe-backend marlin` selected** and a clean startup log — garbage persisted with fp8-KV and forced-FP8-Marlin removed, so it's the NVFP4-MoE **Marlin kernel itself** miscompiling for the modelopt packing. The **Unsloth compressed-tensors** NVFP4 of the same base model on the **same nightly** was clean. | Prefer the **Unsloth NVFP4** checkpoint on Spark; treat the nvidia/modelopt NVFP4 as image-fragile. **Always smoke-test *generation* (not just startup) after launch / every image bump** — assert output isn't `!!!!`/repeats/empty-after-thinking before routing traffic. Don't force `VLLM_TEST_FORCE_FP8_MARLIN` on a checkpoint that FP8-quantizes `lm_head` (drags `ParallelLMHead` into `prepare_fp8_layer_for_marlin` → `AttributeError` at load). | this repo, 2026-07-19 |
| **MTP on Qwen3.6 hybrid MoE** | MTP speculative decode degenerates into **repetition/garbage loops** on deep agentic runs (depth 2 *and* 3); MTP+prefix-caching also leaks tool-call XML as plain text and kills recall | **Disable MTP for quality-first serving** (remove `--speculative-config`) — reproduced on-box (looping returns when re-enabled). MTP is a *speed* feature (quality-neutral when it works); re-add only after validating, and never with prefix-caching on the hybrid. | vLLM #47087, #47194, #44734 |
| **Gemma-4 NVFP4** | Explicit `--quantization` reported to error at startup (auto-detect is safe) | **Omit `--quantization`** — vLLM auto-detects it. Also needs the Gemma4 image (`gemma4-cu130`). | model card (the earlier cite, #40291, is an unrelated OOM bug — no ValueError issue located; note explicit `--quantization mxfp4` on gpt-oss was accepted fine) |
| **fp8 KV cache** | A **capacity/speed tool with a real quality cost on our checkpoints**: Qwen FP8 checkpoints ship no k/v/q scales, so fp8 KV runs **uncalibrated scale-1.0 fp8 attention** — vLLM's own startup log warns "may cause accuracy issues" (observed live, coder-next-fp8, 2026-07-23). Also **crashes GLM-MLA** (`NotImplementedError`), hurts Gemma quality | **Quality-first serving: leave KV at bf16** and size `max-model-len` to fit — measured on-box: coder-next keeps its FULL 262K in bf16 (1.05M-token pool; only 12/48 layers hold KV) and 27b-256k fits full 262K too. Keep fp8 KV **only** where the target context can't fit otherwise (27b-512k) — and say so in the profile's notes. GLM (MLA) and Gemma: **omit it** always. | vLLM #35577; on-box startup-log warnings + KV-pool measurements 2026-07-23 |
| **MLA attention (GLM)** | Other MLA backends error on sm_121 | Use **`TRITON_MLA`** — the only working MLA backend here. | (on-box) |
| **FlashInfer + Gemma** | FlashInfer rejects some `head_size`s | Don't pin FlashInfer attention for Gemma; let vLLM pick. | (on-box) |
| **Nemotron-H Mamba cache** | `mamba_ssm_cache_dtype float16` is TRT-LLM-only (stochastic rounding) → accuracy risk on vLLM | Set **`float32`** (NVIDIA Spark cookbook). | NVIDIA Spark cookbook |
| **Container images** | Rolling nightly silently regressed Qwen3.6-NVFP4 to pure-garbage output (`dev601`). (`:nightly-aarch64` is **not** a separate lineage — as of 2026-07-22 its digest is identical to the arm64 half of the multi-arch `:nightly`; verify with `docker manifest inspect`.) | **Pin a known-good build by digest; nightly is opt-in.** Bump only when a feature needs it, re-validate *generation* (not just startup) on-box first, and comment the pin with its version. (Gemma4 still needs `gemma4-cu130`.) | this repo, 2026-07-03; Docker Hub tag digests, 2026-07-22 |
| **Two Sparks** | Need >128 GB / bigger models | Link via ConnectX-7 200GbE (RoCE, *not* NVLink) for **TP=2** (up to ~405B). Single box is **TP=1**. | (hardware) |

---

## Watch-list — revisit when merged / validated

Loose ends where the *current* config is a deliberate hold, not a final answer. Re-check
when the referenced fix lands or when you next have the box.

> **Citations audited 2026-07-22** against upstream vLLM/GitHub, NVIDIA/vendor specs, and Docker
> Hub — several issue numbers in the trap table were corrected. On-box observations (the
> modelopt-NVFP4 garbage, the MTP acceptance rates, the 0.7 gpu-mem-util value) were left as-is:
> web research can't dispute them, but the retest hooks below flag where a newer build may have
> moved things.

1. **Marlin vs auto/native-FP4 for FP4-MoE — now LIVE, run it.** We force Marlin
   (`--moe-backend marlin`) as the working FP4-MoE path. Native FP4 MoE kernels have since
   landed — FlashInfer **b12x** via PR #40082 (merged 2026-05-20), auto-selected on sm_121 — and
   the **official vLLM DGX Spark blog now recommends leaving backends on `auto`**. But community
   measurements still put Marlin ~16% ahead, and b12x had a TP/PP garbage regression (#47365). So
   the A/B is overdue: on the current nightly launch one profile on `auto` and one on
   `--moe-backend marlin`, compare *generation quality* and tok/s, and drop the flag only if auto
   wins clean. (The old #43906 cite for this was wrong — it's an MXFP8 issue.)
2. **MTP speculative-decode depth — resolved: use 2.** On-box (35B-NVFP4, GB10) per-position
   acceptance is ~89% / ~72% at depth 1/2 → `num_speculative_tokens=2` gives ~143 tok/s
   (≈2× spec-off); 3 over-drafts the single MTP head (position-3 acceptance craters). All
   Qwen3.6 profiles set to **2**. Nemotron (different arch) left at 3 pending its own check.
3. **gpu-memory-utilization — LOWER with MTP, not higher.** 0.85 OOMs the 35B-NVFP4+MTP
   config: the UMA graph-mem estimator reads **negative** (~−20 GiB, #35313) and over-sizes
   the KV cache past physical memory. **0.7 is the validated value** (~21 GiB real headroom).
   Drop-caches doesn't help here — it's anonymous graph memory, not reclaimable page cache.
4. **Unvalidated-on-hardware values.** These shipped from research + roofline, marked for
   live validation: `gemma4-26b-a4b-nvfp4` `max-num-seqs 8`; the 26B/31B Gemma tuning generally;
   the 35B MTP configs. Run §5–§6 of ADD_A_MODEL against each before trusting them.
5. **fp8 KV per new model.** It's not a global default (see the trap table). Each new model
   needs its own decision — don't copy a sibling's `kv-cache-dtype` blindly.
6. **Is the pre-launch drop-caches guard still needed?** #35313 (the UMA false-OOM the guard
   works around) was **closed 2026-04-13** — upstream may now read reclaimable memory correctly.
   The drop is harmless, but retest on a current nightly whether it (and the 0.7 gpu-mem-util
   fallback in item 3) are still load-bearing, or can be relaxed.
7. **Retest modelopt NVFP4 with `VLLM_MARLIN_USE_ATOMIC_ADD=1`.** The 2026-07-19 "pure `!!!!`
   *even with* `--moe-backend marlin`" observation on `nvidia/Qwen3.6-35B-A3B-NVFP4` matches the
   signature of the documented Marlin-MoE shared-memory race — and that profile does **not** set
   `VLLM_MARLIN_USE_ATOMIC_ADD=1`. Set it and re-run before writing the modelopt checkpoint off
   as image-fragile.

_When you close one of these, delete it here and fold the result into the trap table
(or the model's `notes`)._

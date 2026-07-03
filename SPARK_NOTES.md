# DGX Spark (GB10 / sm_121) — hard-won notes

Hardware-specific traps and open threads for serving on the **MSI EdgeExpert DGX
Spark**: NVIDIA GB10 Grace-Blackwell, compute capability **sm_121** (not sm_120),
**128 GB LPDDR5X *unified* memory** (CPU+GPU share it), ~**273 GB/s** bandwidth,
aarch64. This is the single source of truth for the gotchas that keep biting us —
model `notes` and [ADD_A_MODEL.md](ADD_A_MODEL.md) link here instead of re-listing them.

**The one mental model that explains most of this:** decode on GB10 is
**memory-bandwidth-bound**, not compute-bound. tok/s ≈ 273 GB/s ÷ (bytes read per
token) ≈ 273 ÷ (active-params × bytes/weight). So **fewer active params and fewer
bytes/weight win**: NVFP4 ≫ FP8 ≫ BF16, MoE (few active experts) ≫ dense, and
speculative decoding (MTP) is a real multiplier. Quality, though, moves the other way
(4-bit experts lose precision) — hence per-model, per-quant tuning.

---

## Traps & fixes

| Area | Symptom | Fix (what we do) | Source |
|------|---------|------------------|--------|
| **UMA memory mis-detection** | vLLM's `cudaMemGetInfo` can't see reclaimable page cache → false OOM at model load, or a full system freeze | **Every `launch` drops the page cache** (`sync; echo 3 > drop_caches`) right before `docker run` — automatic, set up by `install` (scoped NOPASSWD sudo). Also keep `gpu-memory-utilization` ≈ **0.85**. | vLLM #35313 |
| **FP8-MoE (Qwen3.5/3.6)** | Crash at load `Unknown SF transformation`; ~4% accuracy drop | env **`VLLM_USE_DEEP_GEMM=0`** → MoE falls back to **TRITON** (the working backend on sm_121). CUTLASS MoE is *unavailable* on sm_121 — don't force it. | vLLM #37804, #43507 |
| **NVFP4 MoE** | No **native FP4 MoE kernels** on GB10 | Force the working **Marlin** dequant path: env **`VLLM_USE_FLASHINFER_MOE_FP4=0`** (+ `VLLM_TEST_FORCE_FP8_MARLIN=1` where applicable). Marlin is the *fastest working* path here, not a fallback. | vLLM #43906, NVIDIA dev-forum "Marlin fix" |
| **Gemma-4 NVFP4** | Explicit `--quantization` → `ValueError` at startup | **Omit `--quantization`** — vLLM auto-detects it. Also needs the Gemma4 image (`gemma4-cu130`). | vLLM #40291 |
| **fp8 KV cache** | Model-specific: helps Qwen, **crashes GLM-MLA** (`NotImplementedError`), hurts Gemma quality | Decide per model. Qwen: keep `kv-cache-dtype fp8`. GLM (MLA) and Gemma: **omit it**. | vLLM #35577 |
| **MLA attention (GLM)** | Other MLA backends error on sm_121 | Use **`TRITON_MLA`** — the only working MLA backend here. | (on-box) |
| **FlashInfer + Gemma** | FlashInfer rejects some `head_size`s | Don't pin FlashInfer attention for Gemma; let vLLM pick. | (on-box) |
| **Nemotron-H Mamba cache** | `mamba_ssm_cache_dtype float16` is TRT-LLM-only (stochastic rounding) → accuracy risk on vLLM | Set **`float32`** (NVIDIA Spark cookbook). | NVIDIA Spark cookbook |
| **Container images** | Rolling **`:nightly-aarch64`** silently regressed Qwen3.6-NVFP4 to pure-garbage output (`dev601`) — and it's an arm64-only build, a *different lineage* than the multi-arch default (verify with `docker manifest inspect`) | **Pin a known-good build by digest; nightly is opt-in.** Bump only when a feature needs it, re-validate *generation* (not just startup) on-box first, and comment the pin with its version. (Gemma4 still needs `gemma4-cu130`.) | this repo, 2026-07-03 |
| **Two Sparks** | Need >128 GB / bigger models | Link via ConnectX-7 200GbE (RoCE, *not* NVLink) for **TP=2** (up to ~405B). Single box is **TP=1**. | (hardware) |

---

## Watch-list — revisit when merged / validated

Loose ends where the *current* config is a deliberate hold, not a final answer. Re-check
when the referenced fix lands or when you next have the box.

1. **Marlin vs auto/CUTLASS for NVFP4-MoE.** We force Marlin (`VLLM_USE_FLASHINFER_MOE_FP4=0`)
   because GB10 has no native FP4 MoE kernels (#43906). **When a nightly lands native
   FP4 / CUTLASS MoE for sm_121, A/B it on-box vs Marlin; if it wins, drop the env var.**
   This is the on-box A/B deferred in the DGX tuning commit — not yet run.
2. **MTP speculative-decode depth — resolved: use 2.** On-box (35B-NVFP4, GB10) per-position
   acceptance is ~89% / ~72% at depth 1/2 → `num_speculative_tokens=2` gives ~143 tok/s
   (≈2× spec-off); 3 over-drafts the single MTP head (position-3 acceptance craters). All
   Qwen3.6 profiles set to **2**. Nemotron (different arch) left at 3 pending its own check.
3. **gpu-memory-utilization — LOWER with MTP, not higher.** 0.85 OOMs the 35B-NVFP4+MTP
   config: the UMA graph-mem estimator reads **negative** (~−20 GiB, #35313) and over-sizes
   the KV cache past physical memory. **0.7 is the validated value** (~21 GiB real headroom).
   Drop-caches doesn't help here — it's anonymous graph memory, not reclaimable page cache.
4. **Unvalidated-on-hardware values.** These shipped from research + roofline, marked for
   live validation: `gemma4-26b-a4b` `max-num-seqs 8`; the 26B/31B Gemma tuning generally;
   the 35B MTP configs. Run §5–§6 of ADD_A_MODEL against each before trusting them.
5. **fp8 KV per new model.** It's not a global default (see the trap table). Each new model
   needs its own decision — don't copy a sibling's `kv-cache-dtype` blindly.

_When you close one of these, delete it here and fold the result into the trap table
(or the model's `notes`)._

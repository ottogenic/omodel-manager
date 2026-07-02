# Onboarding a model — add or update (AI guide)

A repeatable, AI-runnable workflow to **add a new model** to the otools stack — or
to **update an existing one** that was added quickly and never fully vetted. Each
model has a **launch profile** in omodel-manager (`model_manager.json`) and a
**generic config** in `configs/*.toml` that omodel-wire (and future adapters)
consume. Nothing is committed until the model is proven on real hardware.

**Input:** a HuggingFace repo link (e.g. `https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8`).

**Add vs update — same workflow, different starting point:**
- **Add:** no profile/config yet. Do every step below: research → draft profile →
  draft config → prove on hardware → verify params → commit.
- **Update:** a profile/config already exists but wasn't given the full treatment
  (params guessed, capabilities unverified, or drifted from the model card). Run the
  **same** research + live-test steps against the existing profile, then reconcile
  each field to what you observe and to the card — don't assume the existing values
  are correct. Example from this repo: `qwen3.6-35b-nvfp4` shipped with
  `tool-call-parser qwen3_xml` and `vision = false`, but the card + live tests show
  `qwen3_coder` and multimodal. An update pass is exactly what catches that.

> **Future (not built yet):** collapse "update" into one command — point the tool at
> an existing model key, and it re-fetches the card, re-runs the live validation
> battery, diffs declared-vs-observed, and proposes the param/capability corrections
> automatically. Until then, this guide is the manual version of that loop.

**Golden rule:** *declare nothing you haven't observed.* Research proposes values;
the live tests (below) confirm them. If a step's evidence contradicts the card,
trust the evidence and note it.

---

## 1. Research the model

1. **Fetch `config.json`** (`<repo>/raw/main/config.json`). Extract: `model_type`,
   `architectures`, `max_position_embeddings`, `rope_theta`/`rope_scaling`,
   `num_key_value_heads`/`head_dim` (KV-cache size), `quantization_config`
   (`fp8`/`nvfp4`/`gptq`/…), and **any vision/mm fields** (a `vision_config`,
   `image_token_id`, or an `architectures` name ending `ForConditionalGeneration`
   means it's **multimodal** — plan to declare `vision`).
2. **Read the model card** (`<repo>/raw/main/README.md`): recommended sampling per
   mode, thinking control (`enable_thinking` / `reasoning_effort` / `/think`),
   tool-call/reasoning parser names, context/long-context instructions, license/gating.
3. **Deep web research** (websearch + fetch) for **errors, issues, community
   feedback** — especially your hardware (DGX Spark = GB10/**Blackwell**/sm_121):
   - vLLM/SGLang GitHub issues for the exact model + quant (loading crashes, wrong
     flags, OOM, `max-num-seqs`/Mamba-cache limits, MoE-backend perf).
   - Search `"<model> vLLM"`, `"<model> blackwell"`, `"<model> fp8 throughput"`.
   - Note anything that changes launch flags or expectations (e.g. FP8-MoE decode
     is slow on Blackwell → NVFP4 may be the better serve).
   - **Known Blackwell/sm_121 FP8-MoE trap:** DeepGEMM's E8M0 scale-factor path
     crashes at load (`Unknown SF transformation`) and hurts accuracy for Qwen3.5/3.6
     FP8 MoE (vLLM #37804/#43507). Fix: env `VLLM_USE_DEEP_GEMM=0` → MoE falls back to
     TRITON (the working backend on sm_121). CUTLASS MoE is *unavailable* on sm_121 —
     don't try to force it.

Write down: capabilities (vision/reasoning/tool_call), thinking mechanism, native
context, quant + the vLLM flags it implies, and any known-issue mitigations.

## 2. Draft the launch profile (omodel-manager)

Add a profile to `DEFAULT_CONFIG` in `omodel-manager` **and** `model_manager.json`
(they must stay identical — the test enforces it). Base it on the closest existing
profile; change only what the quant/model needs:

- `image`: the pinned vLLM image that supports this model/quant.
- `model` + `served-model-name` — put `served-model-name` **inside `vllm_args`**, set
  to the config key, so the served id matches the config's `match` (a top-level
  `served-model-name` is silently ignored and the model serves under its full HF id).
- `port` (default 8000 — one model per box at a time).
- `env`: quant/runtime vars (e.g. `VLLM_NVFP4_GEMM_BACKEND`, `VLLM_ATTENTION_BACKEND`,
  and on Blackwell FP8-MoE `VLLM_USE_DEEP_GEMM=0` — see §1).
- `vllm_args`: `--quantization` (often auto-detected — omit unless required),
  `--kv-cache-dtype`, `--max-model-len`, `--max-num-seqs` (respect Mamba-cache limits),
  `--reasoning-parser`, `--tool-call-parser`, `--enable-auto-tool-choice`, spec-decode, etc.
- `usecase` tags; `notes` capturing the research (pinned image reason, known issues).
- `assets` if the model needs side files (custom parser plugin, chat template).

Do **not** commit yet.

## 3. Draft the generic config (configs/<key>.toml)

One TOML per model (see `configs/README.md` for the schema). This is what adapters
consume — keep it harness-agnostic. Fill from research; the live tests will confirm:

- `match`: the served-model-id(s) + the config key (filename stem must be in `match`).
- `[capabilities]`: `vision` (a modalities table if multimodal, else `false`),
  `reasoning`, `tool_call`, `thinking_control`.
- `[context]`: `native`, `min_thinking`.
- `[presets.*]` (`reason`/`code`/`agent`/`instruct`): `thinking`, `max_output`,
  `[presets.*.sampling]`, and any `options.chat_template_kwargs`.
- If it's the same base model as an existing config (different quant), consider
  **one config matching both** served ids rather than a duplicate.

---

## Testing (on real hardware, via omodel-manager)

### 4. Make the model work

1. **Find a free node.** `omodel-manager ps --remote <host>` on each box. If none is
   free, **ask the operator which running model to stop** — don't evict blindly.
2. **Launch** the dry-run first, then for real, watching startup:
   `omodel-manager launch <key> --remote <host> --keep`  — `--keep` is **not optional**
   on a first launch: the detached default uses `--rm`, which deletes a crashed
   container *and its logs*, so a startup crash leaves you with nothing to read. Then
   `omodel-manager logs <key> --remote <host> -f`.
3. **Review the logs** for: config/flag rejections, OOM / KV-cache / Mamba-cache
   warnings, quant/backend fallbacks, and `Application startup complete`.
4. **Functional test** — one request; confirm a clean completion in the logs (no
   errors). Serve with `--enable-log-requests` so the merged `SamplingParams(...)`
   line is logged (see §6).
5. **Concurrency test** — fire N parallel requests matching the profile's
   `--max-num-seqs`; check logs for preemption/cache errors. **Compare single-prompt
   vs multi-prompt decode tok/s** (this is where Blackwell FP8-MoE perf shows up).

### 5. Validate features & tunable params

Run each check against the **live** endpoint and read the logged `SamplingParams` /
response to confirm it actually took effect — *don't infer*.

1. **Thinking** — toggle `chat_template_kwargs.enable_thinking` on/off; confirm the
   reasoning field appears/disappears. **Field-name drift:** newer vLLM returns the
   trace in `reasoning`, older builds in `reasoning_content` — dump the raw message
   keys and check both (a jumping `completion_tokens` count next to an "empty" field
   means you're reading the wrong key). Test any special option you plan to declare,
   e.g. `{"chat_template_kwargs":{"preserve_thinking":true}}`.
2. **Vision** (only if the config declares it) — POST a **4×4 solid-blue PNG** with
   `"Describe this color in one word."`; a real vision model answers "blue". If the
   config says vision but the model can't, fix the config.
3. **Sampling params — test each independently.** Send one param at a time (isolated,
   one per request) and grep the logged `SamplingParams(...)` line to confirm it
   landed; give each a distinctive value so you can fingerprint its own log line:
   `temperature, top_p, top_k, min_p, presence_penalty, frequency_penalty,
   repetition_penalty, max_tokens, min_tokens, seed, stop, stop_token_ids, logprobs,
   top_logprobs, thinking_token_budget, repetition_detection`.
   **If one doesn't appear in the log, do NOT assume the rest worked** — remove that
   param and retry the others. Gotchas seen live: a rejection can be a *type* error,
   not "unsupported" (e.g. `repetition_detection` takes a `RepetitionDetectionParams`
   object `{max_pattern_size, min_pattern_size, min_count}`, not a bool — and it does
   not echo in the `SamplingParams` repr, so it's only confirmable by accept/reject).
   Record which are honored (they inform what the config and the chat.params plugin
   may safely set).

Confirm the merged truth from logs:
```bash
omodel-manager logs <container> --remote <host> 2>&1 | grep -i sampling
```

### 5b. Benchmark concurrency (`max-num-seqs`)

Before finalizing `max-num-seqs`, run the concurrency benchmark to find the throughput sweet spot:

```bash
python3 utils/benchmark_concurrent.py
```

This script sends 1, 2, 4, 6, 8, 10 concurrent requests to the live endpoint, measures wall time and per-request latency, and prints a summary table. It uses 256 tokens, thinking off, and a general reasoning prompt — good for baseline comparison.

- **System throughput** (total tok/s across all requests) should peak before dropping.
- **Per-request latency** should stay reasonable (watch for high variance = queuing/preemption).
- Increase `max-num-seqs` to the highest level before throughput drops or latency becomes inconsistent.
- After updating, restart the container and verify with `health`.

### 6. Finalize

Only after the model runs clean and you know what's tunable:
- Correct the `configs/<key>.toml` to match observed reality (esp. `capabilities`).
- **Promote** the vetted launch profile into **`DEFAULT_CONFIG`** (the committed source of
  truth) and commit it together with the `configs/<key>.toml`. Do **not** commit
  `model_manager.json` — it's your local, git-ignored sandbox where you prototyped and
  tested; `config --init --force` regenerates it from `DEFAULT_CONFIG`. Run
  `python3 -m unittest` in both repos.
- Optionally `omodel-wire --verify --remote <host>` to diff declared vs live.
- Add a `CHANGELOG.md` entry in each repo.

---

## Checklist

- [ ] `config.json` + card read; capabilities & flags noted
- [ ] deep research done; known issues (esp. Blackwell) captured
- [ ] launch profile drafted (not committed)
- [ ] `configs/<key>.toml` drafted
- [ ] launched on a free node; startup logs clean
- [ ] functional + concurrency test pass; single/multi tok/s noted
- [ ] thinking / vision / each sampling param verified in logs
- [ ] config corrected to observed reality; both repos committed + tests green

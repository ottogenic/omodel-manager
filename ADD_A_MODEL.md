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

> **Tools & roles — how you (the agent) work here.** Two tool families, nothing else:
> - **Document tools** (`WebFetch`/`WebSearch`, `Read`, `Write`, `Edit`) for §1–§3:
>   research the model and author the profile + config. This is the bulk of the job.
> - **The `omm` CLI** (`omodel-manager …`) for anything on hardware — `ps`, `launch`,
>   `logs`, `health`, `stop`. **You never run `ssh`, `scp`, or `docker` directly**;
>   `omm --host <alias>` does the remote plumbing for you. If you're reaching for a raw
>   `ssh`, stop — there's an `omm` subcommand for it.
>
> **Hardware context lives in [SPARK_NOTES.md](SPARK_NOTES.md)** — the DGX Spark
> (GB10/sm_121) trap table + open watch-list. Read it before §1; it's why several
> flags below are what they are, and it's where you log anything new you learn.

---

> **Track it with a todo list.** This is a multi-step workflow. Before you start,
> create a tracked checklist with your harness's todo tool (e.g. `todowrite`, or
> `TaskCreate`/`TaskUpdate`) covering the steps below — §0 prep, §1 research, §2
> profile, §3 config, §4 make-it-work, §5 validate params, §6 benchmark, §7 finalize
> — and keep exactly one item in progress. Don't run this from memory.
>
> **Parallel vs serial — tag each todo.** If your harness can run sub-agents in
> parallel, split the work by phase:
> - **`[parallel-ok]` §1 research** — the web lookups are independent; fan them out
>   (see §1) and synthesize. §2 and §3 drafting can also overlap once research is in.
> - **`[serial]` §4–§6 on-hardware** — **must run sequentially.** One model per box,
>   one shared container and log stream, and §5 fingerprints *one* `SamplingParams`
>   line per request — parallel requests corrupt that. Never fan these out.
>
> Rule of thumb: **parallelize the reading, serialize the hardware.**

## 0. Pick and prepare a host

1. **List hosts.** `omodel-manager ps` shows every registered host and marks each
   `running` / `idle` / `unreachable`. Pick an idle box and use **its alias** from
   here on. Don't grep `model_manager.json` or the configs for a host address.
2. **No host registered yet?** Bootstrap and name one:
   `omodel-manager install user@ip <alias> --fix` (remediates SSH / docker / docker
   group and prompts for an HF token). This stores `alias → user@ip` in
   `~/.config/otools/hosts` so every later step can use `--host <alias>`.
3. From here on, pass **`--host <alias>`** on every remote command. If any host is
   registered, `launch` refuses a host-less run (pass `--local` only to force local).

## 1. Research the model

> **Parallelize this step.** These lookups are independent — if you can spawn
> sub-agents, fan them out and have each return a short structured finding, then
> synthesize. Good split: (1) fetch `config.json`, (2) fetch the model card, (3) vLLM/
> SGLang GitHub issues for `<model> + <quant>`, (4) Blackwell/sm_121 reports,
> (5) throughput/perf benchmarks, (6) quant-specific gotchas. Read
> [SPARK_NOTES.md](SPARK_NOTES.md) first so you know the traps you're checking against.

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
   - **Check your findings against [SPARK_NOTES.md](SPARK_NOTES.md)'s trap table**
     before drafting flags — most GB10/sm_121 surprises are already logged there
     (FP8-MoE `VLLM_USE_DEEP_GEMM=0`, NVFP4 Marlin path, fp8-KV per-model, Gemma
     no-`--quantization`, …). If you hit a *new* one, add it there in §7.

Write down: capabilities (vision/reasoning/tool_call), thinking mechanism, native
context, quant + the vLLM flags it implies, and any known-issue mitigations.

## 2. Draft the launch profile (omodel-manager)

Add a profile to **`model_manager.json` only** (the local, git-ignored sandbox).
**Do NOT edit `DEFAULT_CONFIG` yet** — that is the committed source of truth and
should only be changed after the model is proven. Base it on the closest existing
profile; change only what the quant/model needs:

- `image`: **default to the rolling `:nightly-aarch64`** (omit `image` to inherit the
  config default) — this space moves weekly and pins go stale fast. **Pin only when a
  build is genuinely required** and say why in `notes` (e.g. Gemma-4 needs
  `gemma4-cu130`). Don't pin "to be safe."
- `model` + `served-model-name` — put `served-model-name` **inside `vllm_args`**, set
   to the config key, so the served id matches the config's `match` (a top-level
   `served-model-name` is silently ignored and the model serves under its full HF id).
   **Important:** `served-model-name` is a vLLM launch flag (lives in `vllm_args`).
   It is NOT the same as the config's top-level `model` field (the HF repo ID).
   Downstream tools that call the API must use the served name, not the HF ID.
- `port` (default 8000 — one model per box at a time).
- `env`: quant/runtime vars. On DGX Spark the validated ones (see
  [SPARK_NOTES.md](SPARK_NOTES.md)) are **`VLLM_USE_DEEP_GEMM=0`** (FP8-MoE),
  **`VLLM_USE_FLASHINFER_MOE_FP4=0`** + **`VLLM_TEST_FORCE_FP8_MARLIN=1`** (NVFP4 Marlin
  path). Don't invent env vars — confirm one exists before adding it.
- `vllm_args`: `--quantization` (**auto-detected — omit it**; an explicit value can crash
  startup, e.g. Gemma-4 `ValueError`, vLLM #40291), `--kv-cache-dtype` (**model-specific**
  on Spark — helps Qwen, crashes GLM-MLA, hurts Gemma; see SPARK_NOTES), `--max-model-len`,
  `--max-num-seqs` (respect Mamba-cache limits), `--reasoning-parser`, `--tool-call-parser`,
  `--enable-auto-tool-choice`, spec-decode, and **`--gpu-memory-utilization 0.85`** (UMA
  safety default). Leave the page-cache drop to `launch` — it's automatic (see §4).
- `usecase` tags; `notes` capturing the research (why any image is pinned, known issues).
- `assets` if the model needs side files (custom parser plugin, chat template).

**Testing only — do these things:**
- Edit `model_manager.json` only. Never touch `DEFAULT_CONFIG` during testing.
- **Never run `config --init` or `config --init --force`** during testing — it
  overwrites `model_manager.json` from `DEFAULT_CONFIG`, destroying your local edits.
- Validate with `launch <key> --dry-run`, then `launch <key> --host <host> --keep`.

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

 1. **Confirm a free node.** You picked one in §0 — re-run `omodel-manager ps` to be
    sure it's still `idle` (nothing grabbed it since). If none is free, **ask the
    operator which running model to stop** — don't evict blindly.
 2. **Launch** the dry-run first, then for real, watching startup:
    `omodel-manager launch <key> --host <host> --keep`  — `--keep` is **not optional**
    on a first launch: the detached default uses `--rm`, which deletes a crashed
    container *and its logs*, so a startup crash leaves you with nothing to read. If
    the image isn't cached yet, `launch` pulls it in the background and returns at once
    — poll `omodel-manager pull-status <key> --host <host>` until it says the container
    started. Then `omodel-manager logs <key> --host <host> -f`.
    - **Page cache is dropped automatically** right before the container starts (the UMA
      false-OOM/freeze guard, vLLM #35313 — see SPARK_NOTES). No flag: `install` set up a
      scoped NOPASSWD sudo rule for it. If a launch prints `warning: drop-caches skipped`,
      the host wasn't fully installed — run `omodel-manager install <host> --fix` and it
      goes away. Don't work around it with raw `ssh`.
 3. **Review the logs** for: config/flag rejections, OOM / KV-cache / Mamba-cache
    warnings, quant/backend fallbacks, and `Application startup complete`.
 4. **Wait for readiness.** The model takes 1–3 minutes to start (weights + torch.compile
    + flashinfer autotuning + CUDA graph capture). Poll `health` until READY before
    proceeding:
    ```bash
    while ! omm health <key> --host <host> 2>&1 | grep -q READY; do sleep 15; done
    ```
 5. **Functional test** — one request; confirm a clean completion in the logs (no
    errors). Serve with `--enable-log-requests` so the merged `SamplingParams(...)`
    line is logged (see §5).
 6. **Concurrency smoke test** — fire a handful of parallel requests to confirm the
    server survives concurrency (no preemption/cache errors in the logs). Save the real
    throughput sweep — single vs multi-prompt decode tok/s, tuning `--max-num-seqs`,
    where Blackwell FP8-MoE perf shows up — for the benchmark in **§6**; don't duplicate
    it here.

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
omodel-manager logs <container> --host <host> 2>&1 | grep -i sampling
```

### 6. Benchmark at a fixed 100k context (sweep concurrency)

A short identical-prompt sweep flatters every model (prefix caching + tiny KV). What matters
on a memory-bandwidth-bound box (DGX Spark) is speed at a **full working context** and how
many concurrent users it survives there. Benchmark that:

```bash
python3 utils/benchmark_concurrent.py --host <host>
```

It's a **generic** throughput probe — it doesn't read the config; it auto-discovers the served
model from `/v1/models`. It sends one big **~100k-token prompt** (unique code, so prefix caching
can't fold it) at **concurrency 1**, then **2**, then **3** … up to `--sessions`, streaming each
to measure **TTFT** and **TPOT** (time per output token), and scraping `/metrics` for KV pressure
and **preemptions**. Each level is one fast round — **not** a slow growing conversation — so on a
slow model results still arrive; if a level **fails twice in a row** (timeout/preemption, or the
server drops) the sweep **stops** and recommends the last level that completed.

- `--host` takes the same alias as `omm --host` (or `user@ip` / bare ip; `--remote` is legacy).
- `--context N` sets the fixed prompt size (default 100000); `--sessions N` the max concurrency to
  sweep to; `--req-timeout S` the per-request patience; `--scenario agent` shapes the prompt;
  `--no-think` disables thinking; `--quick` runs the old short-prompt smoke test.
- **Run it against the box's IP** (not an SSH alias that only proxies docker) so the `/metrics`
  endpoint is reachable — otherwise KV/preemption data shows `n/a`.

**The report hands you the two numbers to record:**
- **`Tk/s (1 user @ ~100k)`** — the concurrency-1 decode tok/s. Put it in the profile as
  **`tok_s`** (the `omm models` **Tk/s** column). It's a recorded observation — leave it unset
  rather than guess.
- **`Recommended max-num-seqs`** — the highest concurrency that completed at ~100k. If a level
  failed twice, that's the ceiling; use the last completed level. Set `max-num-seqs` to it,
  restart the container, and re-`health`.
- **`preemptions during run` > 0** means KV overflowed (eviction/recompute) — lower the session
  budget, shrink `--max-model-len`, or raise `gpu-memory-utilization`; don't bump `max-num-seqs`.

**Record the `Tk/s` figure for the `list` table.** So `omm models` can show how fast each model
is, capture its decode speed at **a single user with ~100k of context** — a consistent
apples-to-apples "how fast is this model" number:

```bash
python3 utils/benchmark_concurrent.py --host <host> --sessions 1 --grow-to 100000
```

From the result table, take the **decode tok/s** of the largest context bucket reached
(`64-100k`, or `>=100k` if it got there) — that's steady-state speed at a full working context,
not the flattering small-context number. Put it in the profile as a top-level integer:

```python
"qwen3.6-27b-nvfp4-256k": {
    "tok_s": 38,                     # decode tok/s, 1 user @ ~100k ctx (benchmarked YYYY-MM-DD)
    "image": "...",
    ...
}
```

`omm models` renders it in the **Tk/s** column (profiles without `tok_s` show `—`). Re-measure
and update it whenever the profile's quant, KV dtype, or `max-model-len` changes — it's a
recorded observation, so keep it honest (leave it unset rather than guess).

**Prerequisite:** the model must be READY (see §4 step 4) — the script doesn't wait for startup.
The model is auto-discovered from `/v1/models`; pass `--model <id>` only to override.

### 7. Finalize

Only after the model runs clean and you know what's tunable:
- Correct the `configs/<key>.toml` to match observed reality (esp. `capabilities`).
- **Promote** the vetted launch profile into **`DEFAULT_CONFIG`** (the committed source of
  truth) by copying the entry from `model_manager.json` into the `DEFAULT_CONFIG` dict
  in `omodel-manager`. Then run `config --init --force` to regenerate `model_manager.json`
  from the updated defaults (this resets it; re-add any local tweaks if needed).
- Commit `DEFAULT_CONFIG` and `configs/<key>.toml` together. Do **not** commit
  `model_manager.json` — it's your local, git-ignored sandbox. Run
  `python3 -m unittest` in both repos.
- Optionally `omodel-wire --verify --remote <host>` to diff declared vs live.
- Add a `CHANGELOG.md` entry in each repo.

---

## Checklist

Tags: `[parallel-ok]` = fan out to sub-agents if you can; `[serial]` = one at a time on
the box (see the parallel-vs-serial note up top).

- [ ] `[parallel-ok]` `config.json` + card read; capabilities & flags noted
- [ ] `[parallel-ok]` deep research done; findings checked against SPARK_NOTES.md
- [ ] launch profile drafted in `model_manager.json` only (not committed, not in `DEFAULT_CONFIG`)
- [ ] `configs/<key>.toml` drafted
- [ ] `[serial]` launched on a free node; startup logs clean
- [ ] `[serial]` functional + concurrency test pass; single/multi tok/s noted
- [ ] `[serial]` thinking / vision / each sampling param verified in logs
- [ ] config corrected to observed reality; both repos committed + tests green

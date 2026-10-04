# GLM-5.3-Flash: client policy and performance research

Research date: 2026-10-03. This is a recommendation record, not a qualified new
profile. Operator reports several hours of successful screenshot/web-development
work after the MessagePack and Bun-timeout fixes.

## Conclusions

1. Make client transport policy durable in **omodel-wire**. OpenCode 2.0.22
   deliberately scopes HTTP timeout configuration to a provider. A provider
   containing only this model gives effective model isolation. Bun's current
   environment workaround is process-wide, not model-scoped.
2. The running server is **not continuously decoding at 3 tok/s**. Its retained
   single-request decode intervals measure medians of 11.12 tok/s below 25K,
   9.42 at 50–75K and 8.00 at 125–150K. End-to-end useful output can nevertheless
   be around 3 tok/s after prefill and reasoning.
3. NVIDIA's current GB10 profile uses dense Triton attention, Marlin experts,
   BF16 KV, no speculation and decode CUDA graphs. This is a working baseline,
   not evidence of the maximum performance attainable on two Sparks.
4. Speculation is a substantial potential improvement, but **neither native
   NEXTN/MTP nor DFlash2 is a clean flag-only upgrade in this pinned image**:
   its draft loader contains the ModelOpt quantization override implicated in
   an upstream failure, and its GLM class lacks the required DFlash capture hook.
5. The strongest next runtime comparison is an audited **vLLM + DFlash2 + sparse
   attention** build, initially retaining publisher-owned weights and full image
   resolution. Public two-Spark reports support roughly 20–35 tok/s for less
   predictable workloads, with much higher short/structured scores. None of the
   reviewed reports establishes our complete 400–500K, ten-image, multi-session
   workload. Preserve the proven NIM profile as the comparison/rollback baseline.

## 1. What omodel-wire should manage

### Actual scopes, verified against V2 docs and installed-version source

| Setting | Current scope | Implication |
| --- | --- | --- |
| `settings.headerTimeout` | OpenCode provider; milliseconds | Set 1,200,000 on the GLM provider |
| `settings.chunkTimeout` | OpenCode provider; milliseconds | Independent inter-chunk limit; no evidence yet that GLM needs this raised |
| `settings.timeout` | OpenCode provider; whole request | Leave unset unless a whole-response deadline is intended |
| `BUN_CONFIG_HTTP_IDLE_TIMEOUT` | Entire service process; seconds | The applied 1,200-second workaround cannot distinguish models |

[OpenCode V2 provider docs](https://opencode.ai/v2/docs/providers/#timeouts)
explicitly say timeouts apply provider-wide, not per model or variant.
[2.0.22 `provider.ts:136–164`](https://github.com/anomalyco/opencode/blob/v2.0.22/packages/core/src/provider.ts#L136-L164)
confirms that `modelSettings()` strips `headerTimeout`, `chunkTimeout`, `timeout`
and `transport`. Placing those keys in a model's settings is not a solution.

The native route uses Effect's fetch client. The application timer does not
override Bun's socket timer. The earlier loopback experiment demonstrated this
with the **installed executable**: default fetch died at 360.311 seconds;
`timeout:false` and the environment override both completed at 400 seconds.
See [timeout evidence](glm-5.3-flash-connection-timeout.md).

Rechecked during this research: service 2.0.22, PID 148637, actual process
environment `BUN_CONFIG_HTTP_IDLE_TIMEOUT=1200`; disk config retains the provider's
legacy `options.headerTimeout=1200000`. Operator's sustained run is additional
end-to-end evidence beyond that bounded local experiment.

### A concrete sync bug to address

`../omodel-wire/omodel-wire.py` currently:

- emits only `baseURL` and `apiKey` in `oc_build_providers()`'s provider options
  (around lines 2373–2379);
- removes/rebuilds managed providers in `oc_provider_sync()` (around
  lines 3124–3131), rather than preserving the hand-added timeout.

Thus a normal sync can erase the manual application timeout. The service-level
Bun environment would survive, but OpenCode's own five-minute default would
return. This is a source-audited consequence; sync was not run against live config.

### Recommended design

- Store deployment/client transport policy in **omodel-wire's configuration**,
  matched by managed deployment/endpoint and model. Timeout need depends on the
  hardware/runtime and context, not just the model's identity. Avoid hardcoding
  `glm` substring checks. This requires a new policy field; it is not a currently
  supported wire.json feature.
- Generate the provider timeout deterministically on every sync. Preserve
  explicit user policy and unrelated provider fields, with defined precedence.
  Keep provider IDs stable. If one endpoint exposes models with different timeout
  requirements, separate logical providers by timeout policy while preserving the
  real API model ID. Chachi presently exposes one model, so no split is necessary.
- Keep OpenCode and Bun names out of the harness-agnostic model TOML. If the
  generic schema later carries recommended response-wait metadata, the adapter
  should translate it and allow a deployment-specific override.
- Treat Bun's environment as a **version-specific service prerequisite**. A
  normal sync should inspect/report it. An explicit client-setup action can use
  OpenCode's managed-service environment API and explain the restart. Do not
  write a shell export and assume an already-running service inherited it.
- Preserve an already sufficient user-set native timeout; do not toggle the
  environment when users switch models. Verify the active service environment
  and resolved provider configuration after application, not just disk JSON.
- Continue to use finite provider limits: raising the native ceiling does not
  automatically grant every other provider a 20-minute application timeout.
  Other native providers still retain their configured/default five-minute
  header and chunk limits. Non-provider Bun HTTP traffic is affected globally.

**Long-term fix:** make the native OpenCode transport disable/override Bun's
socket timeout per outgoing model request and rely on OpenCode's provider-level
header/chunk/whole-request timers. Its AI-SDK adapter already passes
`timeout:false` to fetch ([`aisdk.ts:162–168`](https://github.com/anomalyco/opencode/blob/v2.0.22/packages/core/src/aisdk.ts#L162-L168));
the native Effect path lacks that equivalent configuration. With this fixed,
the single-model provider alone would be sufficient and the global environment
workaround could be removed after verification. This is a proposed upstream
correction, not a patch tested through the complete native request stack.

Do not assume changing the package string back to `@ai-sdk/openai-compatible`
will select that adapter: 2.0.22's `aisdk-native.ts` explicitly rewrites known
compatible SDK packages to the native runtime. Likewise, sampling/body hooks do
not configure a socket timeout. A custom provider/transport is possible software
work, not an existing model-JSON knob.

Necessary future checks: sync idempotence; preservation/migration of explicit
timeouts; provider isolation; delayed-header request crossing six minutes;
ordinary provider deadline still firing; user cancellation closing the request;
streaming beyond twenty minutes with regular chunks (no accidental whole-request
cap). The already captured loopback test is the regression-test seed.

## 2. What speed Chachi actually delivered

Read-only inspection at approximately **17:45 UTC**:

- Both containers running, original PIDs 909775/771814, `OOMKilled=false`.
- Head available RAM approximately 6.47 GiB at inspection; head GPU idle, so its
  611 MHz clock is not evidence of an underclocked active workload.
- Retained last-12-hour head logs collected, with no inference submitted.

Continuous decode filtering: TP0 only, one running request, consecutive token
counts differ by 40, timestamps less than 60 seconds apart; reset at intervening
prefill records. These are scheduler-rate observations, not independent benchmark
trials or measurements of the UI's throughput formula.

| Occupied full-token count | Intervals | Median generated tok/s | Observed range |
| --- | ---: | ---: | ---: |
| 0–25K | 113 | 11.12 | 8.59–11.99 |
| 25–50K | 214 | 10.14 | 7.52–10.73 |
| 50–75K | 238 | 9.42 | 7.01–9.90 |
| 75–100K | 168 | 9.05 | 6.96–9.27 |
| 100–125K | 386 | 8.46 | 6.34–8.82 |
| 125–150K | 124 | 8.00 | 6.10–8.35 |

Artifacts: `/tmp/opencode/glm53-speed-research/head-12h.log`,
`decode-samples.json`, `continuous-decode-summary.json`. Context bins describe
the log's occupied-token count during single-request decode, not the configured
400K maximum. There is no measured 400K decode result here.

An already-completed real-image replay illustrates why **3 tok/s can still be
the experienced useful rate**:

- 79,540 input tokens / ten full-resolution screenshots.
- First response after 22.933 s; total 113.832 s.
- 815 generated tokens, including 461 reasoning tokens.
- All generated tokens / total time = **7.16 tok/s**.
- Non-reasoning tokens / total time = **3.11 tok/s**.
- Approximate generated rate after first response = **8.97 tok/s**.

This is not proof of how OpenChamber calculated the user's displayed number.
It demonstrates the distinction using our own measured data. Cold replay first
response was 642 seconds; speculation cannot erase image preprocessing or cold
prefill. Future comparisons must separate first event, first answer/tool,
all-token decode, non-reasoning output, and completed-task wall time.

## 3. Why the current NIM profile is slower

Source: extracted pinned image's
`/opt/nim/tuning_configs/gb10_throughput_nvfp4_1.yaml.j2` and effective startup
arguments in `/tmp/opencode/glm53-msgpack-startup-head.log`.

| Component | Live effective setting |
| --- | --- |
| Tensor parallelism | 2, across two Sparks |
| Full-attention backend | `triton` |
| Linear-attention backend | `triton` |
| MoE runner | `marlin` |
| Sparse-attention selection | `index_topk=null`, including nested text config |
| KV | BF16, 420K allocated tokens / 4.41 GiB per rank |
| Speculation | None |
| Decode graph | Captured batch 1, enabled |
| Prefill graph | Explicitly disabled by KDA compatibility resolution |
| Overlap scheduler / prefix cache | Enabled |
| Chunked prefill | 4096 |
| Running requests | 1 |

NVIDIA's profile disables sparse selection; the hybrid linear layers remain
linear. Full-attention cost consequently grows with retained context rather than
using the model's sparse top-k path. That is consistent with the measured slope,
but this research does not isolate its exact percentage of latency. TP collective
latency, dense/shared layers and GB10 memory bandwidth also matter. A 320B MoE
does not read all 320B parameters on every token; simplistic total-weight/bandwidth
division is not a reliable performance prediction.

The SGLang process has both RoCE HCA/interface names, GID index 3 and
`NCCL_IB_MERGE_NICS=1`. This verifies configuration, **not** that every collective
uses RDMA without fallback. Check active fabric counters/NCCL transport before
claiming a network bottleneck. NVIDIA's recipe asks users to let its hardware
helper discover these values; do not copy another cluster's NIC names.

## 4. Speculative decoding: exact pinned-build limitations

### Native MTP/NEXTN

The bundled checkpoint declares one NextN layer and its index contains **2,617
`model.language_model.layers.45.*` tensors**, including quantized expert weights
and scales. The feature's weights are present.

But pinned `models/deepseek_nextn.py:119–123` unconditionally changes
`modelopt_fp4` quantization to `None`. `glm5_next_nextn.py` inherits this draft
implementation. [Issue #36653](https://github.com/sgl-project/sglang/issues/36653)
reports the same quantized GLM family on TP2 GB10 failing to load: packed FP4
hidden dimension 2048 versus unquantized allocation 4096. The issue's original
TP-sharding hypothesis was corrected in discussion.

[PR #37322](https://github.com/sgl-project/sglang/pull/37322) addresses the
checkpoint-dependent quantization selection; checked **open/unmerged** on
2026-10-03. Its own evidence is unit tests, not a full checkpoint benchmark.
The pinned draft forward also lacks the later multimodal embedding correction
discussed in [#37548](https://github.com/sgl-project/sglang/issues/37548), whose
maintainer identified image sentinel IDs, not the reporter's TP8 theory, as the
root cause. The correction landed on the GLM branch after our pinned snapshot.

Therefore a `NEXTN` 1/1/2 or 3/1/4 trial is not the first cheap experiment on
this image. Updating/reviewing the runtime comes first. Use NEXTN for the native
head, not EAGLE3 with invented hidden-state reshaping; #36829's EAGLE3 experiment
does not establish that correctly configured native MTP is universally broken.

### DFlash2

The pinned tree contains generic DFLASH infrastructure, **but its GLM class lacks
`set_dflash_layers_to_capture`** (it inherits directly from `nn.Module`). The
capture setup function in
`model_executor/model_runner_components/attention_backend_setup.py:40–66`
requires this method. A CPU-only, stdlib AST extraction of that exact validation
function, with a stand-in target exposing the same absence, raises:

```text
Model Glm5NextForConditionalGeneration implements neither
set_dspark_layers_to_capture nor set_dflash_layers_to_capture,
one of which is required for DFLASH/DSPARK.
```

The running head's files were independently SHA-256 matched to the extracted
source before relying on this finding:

- `glm5_next.py`: `0a141565e73252ddb7f1773f30f0c48e001b7dce21a5ca7864b4ea6ae51d0ccd`
- `deepseek_nextn.py`: `68d1bcda68ae5488a00b24a55b84acda711ed7688e7461759b216d689b9131b6`

Later upstream DFlash adapter work and
[#36755](https://github.com/sgl-project/sglang/pull/36755) also handle mHC widened
hidden states and `residual=None`; these are absent from the pinned GLM capture
path, which still adds `hidden_states + residual` directly. DFLASH therefore
requires updated/reviewed model code, not merely an extra download and flags.

External draft model: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
It avoids this particular native-MTP expert-loading issue because the drafter is
a separate small model. It still needs draft weights, verify/graph/recurrent
scratch and a compatible attention path. The approximately 2 GiB weight download
is **not** its complete runtime memory cost. Our earlier multimodal minimum
headroom of roughly 2–3 GiB makes this material.

SGLang trial shape **after updating to a compatible runtime**, not launchable as
a flag-only change in the current NIM:

```text
--speculative-algorithm DFLASH
--speculative-draft-model-path <revision-pinned local draft>
--speculative-num-draft-tokens 4  # then sweep 5 and 6 if correct and faster
```

SGLang's total draft-window count includes the bonus token; do not equate its
number blindly with vLLM's `num_speculative_tokens`. Draft acceptance depends on
the workload. Evaluate reasoning and real tool/code output, not just counting.
The drafter card currently specifies CC BY-NC-ND 4.0/research-evaluation, separate
from the target's MIT license.

An updated SGLang runtime could preserve the target weights, but no evidence yet
proves that its dense-Triton path gets the sparse community recipe's speed or fits
our ten-image workload. Prefer a coherent, pinned alternative runtime over an
accumulating set of patches to the newly stable NIM.

### FP8 KV and sparse attention

FP8 KV primarily buys memory capacity. The community SGLang A/B measured
29.2 versus 29.8 tok/s, effectively parity, with better prefill. It is not a
promised 2x decode gain.

The pinned CUDA TileLang DSA validator rejects FP8 KV. Its proposed unlock
[#36904](https://github.com/sgl-project/sglang/pull/36904) is **closed without
merging**, not an upstream-shipped fix. A follow-up tester reports HumanEval
85.4% versus 90.2% with raw/unscaled FP8 KV; this is a community measurement,
not our validation. The newer scaled-layout [#39349](https://github.com/sgl-project/sglang/pull/39349)
is open and reports capacity benefits on H20, not a material throughput gain
or GB10 qualification. Sparse selection, KV dtype and the compatible kernels
must be reviewed as one path; removing NVIDIA's `index_topk=null` is not enough.

## 5. Other deployments: what the numbers really mean

All external numbers below are **author-reported**, not independently reproduced
on Chachi. Configured context and KV-pool capacity are not proof of an input of
that length. The same repository can contain both old failures and newer fixes;
dated results take precedence over stale quickstarts.

| Stack / report | Reported single-stream performance | Workload / qualification limits |
| --- | --- | --- |
| Our NIM, BF16/no speculation | 8–11 tok/s continuous decode | Actual retained agent workload through 125–150K; full screenshots |
| SGLang NVFP4, no speculation | ~14.7 tok/s | Community short-prompt baseline, sparse TileLang path |
| SGLang + DFlash2 | 28.6 code / 23.6 prose | Warm short prompts; 131K configured; separate 102,644-token cold prefill passed in 104 s |
| vLLM NVFP4 + native MTP | ~20–23 tok/s | Different checkpoints/patches; typical published baseline ~14–15 |
| vLLM NVIDIA quant + DFlash2, long-context recipe | ~28–35 tok/s | 0rand/pilcothink; actual depth-262K test 28.9 tok/s on experimental display-KV variant; ordinary profile warm 32.0 |
| Newer custom vLLM + DFlash2 speed stack | 35.3 prose / 68.9 code | tonyd2wild/knapcio TP2 port, short prompts; 114K single runs 26.9–41.1; additional dense-layer quantization and a **two-image cap** in the 6 GiB profile |
| EXL3 custom vLLM + DFlash2 | ~25–29 prose, much higher structured/code | Different quantization; not a runtime-only comparison |

Primary sources:

- [SGLang recipe](https://huggingface.co/randomllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark),
  [results](https://huggingface.co/randomllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark/blob/main/RESULTS.md),
  [dated ladder/corrections](https://huggingface.co/randomllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark/blob/main/LADDER.md).
  Greedy on/off outputs diverged on 19/20 prompts; do not repeat the blanket
  bit-identical claim as a tested property of that quantized stack.
- [vLLM MTP tuning ladder](https://huggingface.co/datasets/H-K-B/glm-5.3-flash-dgx-spark-vllm):
  TCP/eager 10–12, RDMA 15.1, graphs 15.3, MTP3 20–23. It uses an abliterated
  requant and custom NVFP4-KV patches; useful mechanism evidence, not a preferred
  checkpoint recommendation.
- [0rand long-context NVIDIA/DFlash2 recipe](https://github.com/0rand/glm-5.3-flash-nvidia-nvfp4-dflash-2x-dgx-sparks).
  Its 700K/1M startup and tool-evaluation labels alone do not prove all tests used
  that many input tokens. The explicit depth tests provide stronger evidence.
  The display-memory allocator is a separate experiment, not needed for a first
  400K runtime comparison.
- [tonyd2wild current TP2 report](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark),
  dated 2026-09-29 knapcio port. Its 96–101 tok/s peaks are predictable short
  counting/tool cases. The two-image cap excludes our target workload.
- [MiaAI EXL3 report](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).
- [Intel W4A16/MTP3 recipe](https://github.com/taoofshawn/spark-recipes/tree/main/glm-v53-flash-intel-w4a16):
  publisher-owned alternative with a documented patch list; warns that ~310K+
  prompts have hung hosts in upstream tests and vision concurrency needs testing.
- [NVIDIA optimization forum](https://forums.developer.nvidia.com/t/lets-optimize-nvidia-glm-5-3-flash-nvfp4-for-2x-dgx-spark/382939/26)
  includes real depth ladders and large variation by prompt type/checkpoint.

**Source correction:** 0rand's README claims the NVIDIA checkpoint has no MTP
head based on searching for `mtp`/`nextn` tensor names. Direct inspection of the
current NVIDIA index at revision `da920bb0b9f4a06727223a349e55468e38352348` found
889 `model.language_model.layers.45.*` tensors. Presence and compatible loading
are separate questions. Do not copy the no-head claim into our configs.

The public NVIDIA HuggingFace quant is also a different artifact from NIM's
currently bundled checkpoint. NVIDIA's [model card](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
tests GB200; it does not by itself qualify any of these community GB10 kernels.
[Official vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) uses
datacenter hardware and native MTP. GB10-specific NoPE support work
[#53969](https://github.com/vllm-project/vllm/pull/53969) remains open, so
"switch to stock latest vLLM" is not a proven recipe for this exact workload.

## 6. Ranked experiment plan

### Small changes within the proven deployment

1. Establish matched measurements at ~50K and ~125K, with full-resolution images,
   default reasoning, fixed sampling and the same output task. Report cold and
   cached TTFT separately, reasoning/all-output counts, decode rate and task wall
   time. Retained logs provide a starting baseline; do not flush a user's active
   prefix cache merely to collect an artificial benchmark.
2. A/B `--chunked-prefill-size 4096` versus **2048**. This is the cleanest
   no-new-weights launch-setting experiment. Expected benefit is prefill pressure
   and latency, not a multiplication of decode speed. Community evidence favors
   2048 at long context, but its indexer allocation mechanism differs from our
   dense NIM path; direction and magnitude need measuring here. Return to 4096
   if it merely adds scheduling overhead.
3. Compare per-request **high versus max reasoning effort** on representative
   coding/tool tasks. This can reduce time to useful output, not raw kernel
   latency. Preserve correctness as the deciding metric: our previous low-effort
   smoke test passed only 4/6 versus 6/6 at default. The official template always
   reasons; `enable_thinking=false` is not a supported shortcut for this artifact.
   The candidate TOML already specifies high for Build and max for Plan, so first
   verify which effort reaches the actual request; this may already be applied.
4. Check RDMA traffic and active clocks while measuring. Existing interface
   configuration and idle clock samples are insufficient to promise a fabric
   or clock-setting speedup.

Already enabled: prefix caching, overlap scheduling and batch-1 decode CUDA
graphs. Raising maximum graph batch size or concurrency cannot accelerate an
already single active request by itself. Increasing static/KV memory reservation
is not a compute-speed switch. Raising prefill to 8192 is not the first experiment
given the observed full-resolution multimodal headroom.

### Meaningful decode acceleration

5. Require a compatible, audited runtime before enabling MTP or DFlash: both have
   concrete blockers in the running image. A newer SGLang image with the proper
   GLM capture implementation is one comparison candidate. Keep weights constant
   where supported to distinguish runtime gains from quantization changes.
6. Prioritize a coherent audited **vLLM NVIDIA-quant + DFlash2 sparse-attention**
   comparison. Sweep modest draft windows and measure accepted tokens per verify
   step as well as wall time; more draft tokens is not always better. Start with the
   conservative long-context recipe rather than the headline speed stack's
   extra dense quantization/two-image restriction. Pin all code, image and weight
   revisions; verify tool parsing, full-image handling and actual prefix reuse.
7. Progress actual inputs through 128K → 256K → 400K with the ten screenshots,
   then 500K and two sessions if memory permits. Judge against task completion
   time and quality. A configured 1M limit or a counting score is not acceptance.

No deployment, checkpoint, client config or host tuning was changed by this
research. No external runtime patch was executed and no additional model
inference was submitted. Only research/chronology notes were added or updated.

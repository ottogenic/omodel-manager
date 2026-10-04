# GLM-5.3-Flash: Eugr B12X trial on Chachi

## Authorization and scope — 2026-10-03

Operator approved the Eugr B12X image, Spark checkpoint and recipe settings,
deployed with omodel-manager rather than the upstream launcher. Chachi's NIM
profile remains the rollback. Existing unrelated/uncommitted work is preserved.
The profile was live-qualified in the sandbox. On 2026-10-04 the operator approved
preserving both deployments in Git; the tested settings are now promoted into
`DEFAULT_CONFIG.cluster_models` alongside the NIM/SGLang baseline.

## Exact inputs

- Recipe: `eugr/spark-vllm-docker@cb51860d98e956d9df71431a8785cc0cf6f65b53`,
  `recipes/glm-5.3-flash.yaml`.
- Image: `eugr/spark-vllm-b12x@sha256:1803d1c23f746f2f559b1d119deb4517b857750c4fb283d0e96d500f681f1af8`.
- OCI config: `sha256:4aef4572b26eb283a1757aeedddc24f5b2bce3ff2c16d961f7334b0f480c3150`.
- Embedded vLLM: `local-inference-lab/vllm@ab86b70734009c34e91db456bb96340c3faf3d4e`,
  version `0.1.dev21553+gab86b7073.d20261003`.
- Embedded FlashInfer: `1b578de52924c973bd229fe4fd147499ead2781d`.
- Embedded B12X: `a321f9a6e32f4119486c0ea9dd961595ffc87daf` (package 1.5.0).
- Checkpoint: `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark@a608241037e4c2565356bff7ca293f2133888f88`.
  44 referenced safetensors shards, 187,637,721,616 bytes (~174.75 GiB).

The image source identities were recovered from hashed registry layers. This is
builder-authored provenance, not a signed source-to-binary attestation. The image
deliberately pins CUTLASS DSL 4.7.0 despite the source requesting 4.7.1. Use the
immutable image for this trial rather than rebuilding floating branches.

## Recipe translation

Preserve TP2, PP1, DCP1, B12X loader/attention/MoE/linear kernels, modelopt_mixed,
FP8 KV, block 256, Mamba align, prefix cache, chunked prefill 4096, 500000 context,
four sequences, 0.87 GPU memory utilization, MTP5 with humming/B12X draft kernels,
glm45 reasoning parser and glm47 tools. Preserve the recipe environment.

The upstream YAML's `kv_cache_memory_bytes: 8G` is unused by its command; do not
invent an explicit 8G allocation. Omitted scheduling and generation settings use
runtime defaults. No eager-mode override, expert parallelism, external drafter,
image reduction or remote model Python is added.

Launcher adaptations: exact local HF snapshot, existing served ID
`zai-org/GLM-5.3-Flash`, loopback API :8000 behind existing Tailscale TCP forwarding,
registered fabric interfaces, native multiprocess two-node rendezvous, device
labels, retained crash containers, offline weights and persistent runtime cache.
Request logging and token-details diagnostics are enabled by our launcher.

Source audit confirms the B12X loader accepts a read-only absolute HF snapshot,
including its blob symlinks; backing storage must support O_DIRECT. Set
`XDG_CACHE_HOME=/cache/runtime` to persist its native loader, CuTe and RoCE
compilation caches. Match Eugr's launcher with `NCCL_IGNORE_CPU_AFFINITY=1` and
`TP_SOCKET_IFNAME` on the first fabric rail; explicitly retain GID 3, already used
by this fabric. The Python frontend has no discovered total request-body cap;
the Rust frontend is not enabled. Native image-count default is 999; no image
count/resolution reduction is introduced.

Generic vLLM cluster configuration now supports external-recipe loader, eager,
quantization, linear, DCP and Mamba settings. Registry images may pin the immutable
Docker image ID in addition to their manifest and runtime version, avoiding
Docker-store-dependent inspect serialization signatures.

Both hosts run Docker 29's containerd image store: their inspected image ID and
Descriptor are the pinned **manifest** digest, not the OCI config digest. The
first preparation stopped at the incorrectly assumed config-ID check after
successful pulls. The manager now pins the observed Docker ID explicitly.

## Work record

### Startup findings

- Attempt 1: B12X's io_uring bounce loader was denied by Docker's default
  seccomp policy. Candidate now uses `security_opt = ["seccomp=unconfined"]`;
  both ranks subsequently loaded 88.14 GiB of model weights successfully.
- Attempt 2: cold compilation diverged between ranks. Worker finished JIT
  warmup at 20:11:05 UTC and timed out at RoCE sequence 1061 at 20:11:51;
  head was still warming kernels and later failed at sequence 1062. No OOM
  was reported. B12X's independent default is 20 million polls (~20 seconds
  per its source), unrelated to NCCL/Gloo timeout settings. Candidate now
  sets `B12X_ROCE_SPIN_LIMIT=600000000` on both nodes to tolerate cold JIT.
  This is an explicit deviation from the recipe, without changing model,
  context, concurrency, speculation or memory settings.
- At attempt 2, vLLM reported 9.72/9.79 GiB available for KV cache and an
  estimated 1,186,567-token capacity / 2.37x at 500K. These are allocator
  estimates, not successful long-context request measurements.

- Source/provenance and checkpoint review: complete.
- Profile translation: complete; 264 unit tests and py_compile pass.
- Hardware/fabric preflight: passed both nodes (driver 580.178.04, Docker 29.6.2,
  existing two 1500-MTU RoCE rails). No host networking changes.
- Image/weight preparation: complete on both nodes; NIM rollback preserved.
- Launch and functional qualification: passed on attempt 3.
- Full-image replay, context ladder and speed/concurrency: completed below.
- Final evidence/config reconciliation: complete; promoted profile records `tok_s=18`.

Configured context/concurrency are not measurements. The Spark checkpoint has
additional MXFP8 attention/shared-expert quantization compared with NVIDIA's
checkpoint; quality must be checked as well as speed. Native image preprocessing
retains its 8000-token per-image budget; trial screenshots are not downscaled by
our client.

## Live qualification results — 2026-10-03

Attempt 3 started the rank containers at 20:28:17/20:28:21 UTC and reached
readiness around 20:39 UTC. Logs confirm CUDA graphs, live RoCEnante all-reduce
and all-gather on both rails, NCCL NET/IB fallback, and active MTP acceptance
metrics. Final allocation estimate was 1,208,955 tokens / 2.42x at 500K.

- Manager readiness: direct chat, reasoning, parsed tools, vision and SSE passed.
- Arithmetic: 137 × 29 = 3973. Python code reading returned `[1, 9]`.
- JavaScript fix preserved zero, false and empty strings while excluding null
  and undefined. Required tool call returned `record_result({"value":3973})`.
- Every generated replay tool call was parsed as JSON; none was executed.
- Endpoint remains `http://otto-dgx-1.tail1faf4a.ts.net:8000/v1`, served ID
  `zai-org/GLM-5.3-Flash`, max total context 500000, loopback binding preserved.

### Full-resolution ten-image replay

Same 130-message coding history, six tool definitions and ten original images
as the NIM fixture, with a unique initial replay ID to prevent prefix reuse on
the cold request. Payload ~8.04 MB; native image tokens 28404.

| Request | Actual input | Cached input | First delta | First content/tool delta | Total | Output incl. reasoning | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold, full completion | 79562 | 0 | 91.24 s | 149.84 s | 169.50 s | 1570 | 20.08 |
| Cached, full completion | 79562 | 72960 | 11.40 s | 107.88 s | 125.10 s | 2148 | 18.90 |

Both produced a visible explanation plus two complete edit tool calls and ended
naturally below the 8192 output allowance. Earlier 2048-token replays returned
parseable calls but hit that allowance; the parser reported `tool_calls` rather
than `length`. They are **not** evidence of natural completion. The qualification
runner now checks usage against the allowance as well as finish_reason.

Historical NIM first-delta times were 642.22 s cold and 22.93 s cached. Different
sampled reasoning/output lengths mean total wall times are not a controlled
apples-to-apples speedup comparison. This trial also changes the checkpoint.

### Actual long context with all ten images

Three synthetic release labels were placed throughout appended report text;
the original conversation/images were retained. Correct labels were returned at:

| Actual input | First delta | Total | Output | Retrieval |
| ---: | ---: | ---: | ---: | --- |
| 150944 | 171.73 s | 173.39 s | 51 | 3/3 |
| 255851 | 293.93 s | 296.22 s | 86 | 3/3 |
| 399775 | 473.08 s | 476.67 s | 118 | 3/3 |
| 497704 | 605.31 s | 608.93 s | 115 | 3/3 |

All were cold (`cached_tokens=0`). The near-500K request reserved 2048 output
tokens, keeping its permitted total inside the 500000 limit. This demonstrates
capacity and simple retrieval, not full coding-quality equivalence at 500K.

One earlier 399818-token test **failed the answer check**: the model interpreted
the synthetic `ALPHA_KEY`/`BETA_KEY`/`GAMMA_KEY` request, mixed with repeated
“no actions requested” prose, as an injection and refused to return them.
Transport and generation completed. The retained corrected tests explicitly
describe public fictional release labels and avoid the contradictory wording.
This failure remains part of the quality record.

`/tokenize` undercounted this ten-image input by 23100 tokens compared with real
inference usage. Subsequent sizing used that measured correction; the table uses
**response usage**, not the requested target or tokenizer estimate.

### Speed and concurrency

`utils/benchmark_concurrent.py`, unique generated stories, `reasoning_effort=high`,
max output 2048, timeout 1200, image's default sampling. All returned `ok`.

| Parallel requests | Actual input per request | TTFT | Per-request decode | Wall |
| ---: | ---: | --- | --- | ---: |
| 1 | 50862 | 52.4 s | 18.3 tok/s | 64 s |
| 2 | 50862–51143 | 58.8–105.8 s | 4.3–14.2 tok/s | 122 s |
| 4 | 50863–51151 | 58.2–211.3 s | 1.4–13.1 tok/s | 229 s |

Decode includes reasoning and scheduling interference from other requests'
prefill. Four sequence slots work, but simultaneous cold prefills impose severe
interactive slowdown. Four simultaneous 400–500K sessions were not qualified;
the KV allocator itself estimates only about 2.42 full 500K requests.

### Tunable-parameter checks

Sequential isolated probes and their logged `SamplingParams` confirm delivery
of temperature (0.731), top_p (0.823), top_k (17), presence penalty (0.37),
frequency penalty (0.29), repetition penalty (1.13), max_tokens (137),
min_tokens (8), seed (7391), stop string, stop-token IDs, logprobs and top_logprobs.
These transport checks do not establish statistical effects of each setting.

- `min_p=0.071`: HTTP 400, explicitly unsupported with speculative decoding.
- `enable_thinking=true`: reasoning field and correct visible answer.
- `enable_thinking=false`: empty reasoning field but reasoning leaked into
  content and the integer was duplicated. No clean non-thinking mode is declared.
- Native low/high/max effort options: accepted with correct answers and reasoning;
  these short probes do not measure relative reasoning quality.
- `thinking_token_budget=83`: logged as 83 and its kernel compiled, but response
  consumed 256 reasoning tokens and exhausted output. Not qualified as a useful cap.
- `repetition_detection`: object accepted, absent from SamplingParams repr;
  actual repetition-stopping behavior remains unqualified and is not configured.

### Final state and evidence

Both rank containers remained running through replay, long-context, concurrent
and parameter testing with zero restarts. Post-concurrency diagnostics showed
`OOMKilled=false` on both, no new kernel OOM evidence, and 3.7/7.1 GiB available
host memory. These are point samples, not measured minimum headroom. No runtime
crash occurred after the successful launch.

Runtime warnings retained: unavailable optional DeepSelect/SymmMem paths, MTP
draft KV-group identification, text-only draft inputs for multimodal requests,
and some first-use Triton compilation. Target vision remained enabled and passed.
Do not interpret this small battery as quantization-quality equivalence to NIM.

Evidence directory: `/tmp/opencode/glm53-eugr-trial/`:
`launch-3.log`, `results.jsonl`, `replay-full.log`, `context-*.log`,
`benchmark-n{1,2,4}.log`, `parameter-results.jsonl`, `parameter-head.log`,
`final-{head,worker}.log`, `final-diagnostics.log`. Raw requests contain private
conversation/screenshots and remain outside the repo.

Rollback: `python3 ./omodel-manager stop Chachi --yes`, then
`python3 ./omodel-manager launch Chachi glm-5.3-flash-nim`. NIM's profile,
image and cached bundled weights remain available. Its concurrency is one; the
shared generic config now uses a conservative one-worker adapter cap for safe
switching, while the Eugr server retains four sequence slots. Both profiles are
now curated in `DEFAULT_CONFIG.cluster_models` and survive `omm sync`.
See [comparison setup](glm-5.3-flash-deployments.md).

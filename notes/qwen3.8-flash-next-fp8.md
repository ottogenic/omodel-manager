# Qwen3.8-Flash-Next FP8 dual-node notes

## Pinned inputs

- Model: `Qwen/Qwen3.8-Flash-Next-FP8`
- Model revision: `970c569adaca6b35532111fd6b27351b2baefe50`
- Model size: 185,553,536,918 bytes (172.8 GiB)
- Quantization: dynamic FP8, 128x128 weight blocks
- Image: `vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`
- ARM64 manifest: `sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`
- Canonical ARM64 content: `sha256:483da4d4cdbd8cb6b2094ef3a9b205307b65d8e61120f043db61a4156a750d0b`
- Reported runtime: `0.1.dev20073+g8e685d198`
- Installed vLLM wheel: `sha256:2dcbdc4aaccfaebfdd07e1585ea037eb216e01dee4283122a8b4337ca7febc05`
- License: Qwen Community License Agreement 1.0

The checkpoint is first-party, contains declarative model/tokenizer/Safetensors files, and is
loaded by the image's native `Qwen4ExpForConditionalGeneration` implementation. The launch does
not enable Hugging Face remote code.

## Research, 2026-08-29

The official card describes a 125B-parameter model with 6B active parameters, a separate 51B
n-gram embedding table, a 4B MTP module, multimodal input, and a native 262,144-token context.
The official vLLM recipe requires the dedicated `qwen38-flash-next` preview image and identifies
TP2 as the minimum validated FP8 layout on datacenter Blackwell. The exact dual-Spark community
recipe uses multi-node multiprocessing, TP2, expert parallelism, lazy Safetensors loading, an
8,192-token scheduler budget, and 0.83 memory utilization.

Selected sources:

- https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8
- https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
- https://github.com/vllm-project/vllm/pull/53896
- https://forums.developer.nvidia.com/t/qwen3-8-flash-next/381228/97
- https://forums.developer.nvidia.com/t/qwen3-8-flash-fp8-dual-sparks/381440/23

The initial manager profile intentionally differs from the fastest community command. MTP and
prefix caching are disabled, async scheduling is disabled, FlashInfer autotuning is disabled, and
eager execution is enabled. This establishes a conservative quality and memory baseline before
performance tuning.

Open Spark-specific issues behind those guards:

- DeepGEMM advertises SM121 support but faults on block FP8; `VLLM_USE_DEEP_GEMM=0` selects the
  working fallback: https://github.com/vllm-project/vllm/issues/54125
- Growing shared-prefix conversations can fault in the GDN cache path; disabling prefix caching
  is the demonstrated workaround: https://github.com/vllm-project/vllm/issues/54173
- Cold compilation can inflate profiled activation memory and shrink KV capacity until restart:
  https://github.com/vllm-project/vllm/issues/54122

The vLLM support PR is open and its own description calls the branch potentially unstable. This
profile remains `experimental` until the pinned image passes package inspection, dual-Spark text,
reasoning, tools, image input, long-context, memory, and benchmark gates.

## Checkpoint and runtime audit, 2026-08-29

The pinned checkpoint revision contains 144 root files: 131 Safetensors shards and 13 declarative
model, tokenizer, and metadata files. It contains no Python, executable, pickle, `bin`, `pt`, or
`pth` payloads; no `auto_map`, custom objects, symlinks, or path escapes; and no chat-template
import, include, eval, or attribute-escape primitives. The launch mounts that snapshot read-only,
runs Hugging Face and Transformers offline, and does not enable remote code.

Artifact identity is strong but source reproducibility is incomplete. Docker Hub resolves the
pinned index to the pinned ARM64 manifest and image config
`sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`. The manager also
checks the canonical config/rootfs fingerprint above and runtime version before preparation or
launch. The image labels nevertheless report both source revisions as `unknown` and the pipeline
as `local`; no OCI referrer, cosign-tag signature, Notary metadata, SBOM, or build attestation was
found. The embedded short revision `8e685d198` could not be resolved to a public full commit and
must not be expanded by inference.

Public PRs document related work but are not the exact image source. vLLM PRs 53896 and 53899 were
opened after the image was built, and PR 53896 uses a different installed package layout. The
official recipes merge names the dedicated image tag but publishes no source SHA or build record.
Community fork `blazux/qwen3.8-Flash-DGX` commit
`82ed48d373d8a2c03d142d203f07bce0a6b69125` explicitly starts from this exact image digest; it
therefore consumes, rather than proves the source of, the official artifact. The fork commit and
related implementation commits are unsigned. Inference Atlas independently records the same
runtime abbreviation and fork commit, but that contributor-supplied record has no output-image
digest. Relevant provenance sources:

- https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/qwen38-flash-next
- https://github.com/vllm-project/recipes/pull/848
- https://github.com/blazux/qwen3.8-Flash-DGX/commit/82ed48d373d8a2c03d142d203f07bce0a6b69125
- https://github.com/0xBakeer/inference-atlas/commit/a2959b3a1cf839e638ba675ef42db024c0eb4263

Manual package inspection found only two `pip check` findings. Automated image scanners were not
available, so this is not a substitute for an SBOM or vulnerability scan.

- PyTorch metadata requires `nvidia-nccl-cu13==2.29.7`, while the image deliberately installs
  2.30.7. Upstream vLLM commit `5d8e90a96616c4fe339ff0b0c2a2d470f6eb24bf` sets 2.30.7 because
  DeepEPv2 requires NCCL 2.30.4 or newer. A manager-mediated two-rank cluster smoke test completed an
  NCCL all-reduce with value 3.0 on both ranks, reported runtime 23007, selected `NET/IB` over both
  RoCE devices, reached `Init COMPLETE`, and exited zero. Do not downgrade or run package repair.
- `nvidia-cusparselt-cu13==0.8.1` reports unsupported because the installed wheel metadata uses the
  obsolete `manylinux2014_sbsa` tag instead of the published ARM64 wheel's
  `manylinux2014_aarch64`. Its ELF payload is ARM64. On one rank, CUDA allocation plus
  `cusparseLtInit`, version/property queries, and `cusparseLtDestroy` all returned success and
  reported 0.8.1. This was an API lifecycle smoke, not a numerical sparse-matmul test.
- There is no plain `tokenspeed` distribution. The image contains the vLLM-requested
  `tokenspeed-mla==0.1.8` and its `tokenspeed-triton==3.8.10.post20260721` dependency.
- Python startup hooks are limited to `distutils-precedence.pth` and
  `nvidia_cutlass_dsl_packages.pth`.

Decision: the package and provenance gate passes for an isolated launch because
the official artifact, platform contents, and native libraries are pinned and checked, and the
untrusted model snapshot cannot execute code. This does not establish reproducible source
provenance or production trust. Keep the API loopback-only.

## Hardware qualification

- Cluster: two DGX Sparks connected over dual-rail NVIDIA Sync/RoCE
- Status: validated for the pinned text and image surface; video is excluded
- Baseline launch: `omm cluster launch CLUSTER qwen3.8-flash-next-fp8`
- Preparation: `omm cluster prepare CLUSTER qwen3.8-flash-next-fp8 --build --weights`

### Startup and memory, 2026-08-30

The exact 144-file snapshot and pinned image were prepared on both nodes. The manager verified
vLLM `0.1.dev20073+g8e685d198` and canonical image content
`sha256:483da4d4cdbd8cb6b2094ef3a9b205307b65d8e61120f043db61a4156a750d0b` before launch. Full
fabric preflight passed on both configured private RoCE rails. NCCL 2.30.7 selected
`NET/IB` on both devices and reached `Init COMPLETE` on both ranks.

The 172.78 GiB checkpoint loaded in approximately 11.3-11.6 minutes. Each rank reported 86.74 GiB
of model memory. At 0.83 memory utilization the runs allocated 9.25-10.99 GiB per-rank KV memory;
the head reported 747,858-786,432 cache tokens, enough for 2.85-3.0 full 262,144-token sequences.
The retained conservative profile still limits scheduling to one sequence. Observed kernels:

- Triton FP8 MoE with 256 of 512 experts on each EP rank
- Triton/FLA GDN prefill and CUDA GDN decode
- FlashAttention 2 for attention and FlashInfer top-k/top-p sampling
- PYNCCL collectives over both RoCE rails

Expected warnings were eager mode disabling compile/CUDA graphs, unsupported symmetric-memory and
custom multi-node collectives on SM121, no GB10-specific FP8 MoE tuning file, first-use Triton JIT,
and undocumented `min_frames`/`max_frames` processor fields. There were no OOMs, preemptions,
DeepGEMM faults, tracebacks, or communication errors during the final text/image run.

### Features and parameters, 2026-08-30

Mandatory direct chat, separated reasoning, one exact `qwen3_xml` tool call, solid-blue PNG image,
streaming, and post-launch health gates passed. The runtime emits `reasoning` rather than
`reasoning_content`. `reasoning_effort` values `xhigh`, `medium`, and `low` each produced reasoning
plus final content with `enable_thinking=true`; all three suppressed reasoning with
`enable_thinking=false`.

Distinctive one-parameter requests were confirmed in logged `SamplingParams` for `temperature`,
`top_p`, `top_k`, `min_p`, presence and frequency penalties, repetition penalty, `max_tokens`,
`min_tokens`, seed, stop string, stop-token ID, logprobs/top-logprobs, and
`thinking_token_budget`. The API accepted structured `repetition_detection`, although that value
does not appear in the repr. Output-ceiling requests were separately verified at both the
initial `max_tokens=32768` and the final `max_tokens=131072` setting; each was logged, returned
HTTP 200 and `OK`, and stopped normally. The final value matches the local Qwen3.8-27B OpenCode
configuration. These checks prove request acceptance, not sustained generation to either limit.

The existing quality battery ran each case twice. Weather, arithmetic, three-tool selection, and
code-search calls passed 8/8 with valid arguments and no hallucinated tool. Executed FizzBuzz,
interval merge, Roman numeral, and balanced-bracket solutions passed 8/8. No response looped,
degenerated, or ran to its token limit.

Video is intentionally not declared. A valid in-memory 16x16, two-frame AVI request returned an
empty HTTP response and the head exited 139 (SIGSEGV); Docker auto-removal erased that first crash
log and the worker then reported the expected broken control-store connection. This prompted the
manager's `cluster launch --keep` support so future exited rank diagnostics survive. The final
relaunch used `--keep` and remained healthy, but video was not retried.

### Context and throughput, 2026-08-30

Unique-prompt streaming benchmarks with thinking disabled produced:

| Load | Actual input | TTFT | Prefill | Decode | Wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=1, ~50K | 49,985 | 19.19 s | 2,604 tok/s | 19.93 tok/s | 23.21 s |
| N=2, ~50K | 49,985-50,005 | 16.99-41.40 s | queued | 19.59-19.76 tok/s | 48.59 s |
| N=4, ~50K | 49,985-50,011 | 16.66-90.86 s | queued | 19.81-20.02 tok/s | 97.13 s |
| N=1, ~100K | 99,965 | 35.39 s | 2,825 tok/s | 20.33 tok/s | 41.10 s |

With `max_num_seqs=1`, N=2 and N=4 serialized exactly rather than sharing a batch. Once scheduled,
each 50K prefill took approximately 16.7-16.9 seconds and decode stayed near 20 tok/s; later TTFT
includes queue time. The recorded profile speed is therefore `tok_s=20` from N=1 at 49,985 input
tokens.

A near-native retrieval request used 239,955 actual prompt tokens plus a 128-token output ceiling,
leaving 22,061 tokens of headroom. TTFT was 94.06 seconds and total time 96.58 seconds. The model
returned both exact random markers from the beginning and end of the prompt and stopped normally.

### Promotion decision

The pinned profile is `validated` for its declared text, image, reasoning, and tool surface. Keep
`gpu_memory_utilization=0.83`, `max_num_seqs=1`, eager mode, and disabled MTP, prefix caching,
async scheduling, and FlashInfer autotuning: the baseline already fits native context, and none of
those riskier speed levers was A/B qualified. This status is artifact-specific and does not include
video, reproducible source provenance, or a production-trust claim.

# DeepSeek V4 Flash Vision Exp build notes

## 2026-09-02 official preview candidate

### Scope and provenance

- Hardware target: two DGX Spark GB10 nodes (one head and one worker).
- Official weights: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at
  `6821d6ad3681a4b137b066b76094fa82ebd0a380`, 48 shards and 167,819,616,863
  bytes in Hugging Face storage. The model is MIT licensed.
- Official vLLM ARM64 preview base:
  `vllm/vllm-openai@sha256:8568b4bbc821903d93a0a9c17dd80382fdc0ba78eaa128e3eb5cb71c3bf06b79`.
  The multi-architecture tag manifest is
  `sha256:0075fd82e3b6d943b0aa91e35da8dbca63d88516c607745131055e9d81f37ebb`.
- vLLM Vision support is pre-release PR #54566 at
  `3e3c938ebb837efcf3535e8644d21a413ed08c0d`; its official validation is TP4
  on GB200, not GB10.
- GB10 image-prefill compatibility uses only FlashInfer PR #4850 at
  `7a55473b9e3f81a1ce7c862c3b993f6ef23cab5f`. The reviewed dispatcher source
  SHA-256 is `7cc455ade8df4084174e2f837aa0f3c7d511c389d0bcc4020c8571f54cb36811`.
- FP8 E8M0 linear compatibility uses vLLM PR #47988 at
  `23556d7a6e69e4837e74f2c00004d51b550c0eb8`. Its patched CUTLASS source is
  pinned at SHA-256 `c6282c9290b13bde0bce7af8bcb1c05831c75c948648a064eb160b683c26dd0b`.
  The Triton change is applied fail-closed to the exact Vision source at
  `3e3c938ebb837efcf3535e8644d21a413ed08c0d` and verifies the merged file as
  SHA-256 `7bbbbfb8061a575eb9db0d7657eef54e31829822a2b42bf801f6c1329822b097`.
- Because the profile deliberately disables DeepGEMM linear kernels, `wo_a` retains the normal
  2-D FP8 layout while the model's unconditional output-projection einsum expects DeepGEMM's
  3-D post-load layout. A target-only fallback uses ordinary FP32 block scales, dequantizes and
  caches each `wo_a` once in BF16, then applies `torch.bmm`. It is derived from the previously
  audited and numerically checked SM12x output-projection fallback, narrowed to 2-D weights. The
  exact Vision source and patched result are pinned at SHA-256
  `ffb28e7cc44124bb2878e596617b70ec2659c7b26a2048a06facc343fa4d24e1` and
  `c958daaec754d0933659d639317fc13065e7a1867aeef06878331e400e9fc581`.

### Security decision

The MiaAI-Lab recipe at `d97c808ec1c71b496badee6805dfd4818a8455d7` was reviewed but not used.
It starts from a third-party Anemll image and mutates many installed runtime files at
container startup. Its patch train mixes Vision support, scheduler behavior, API policy,
performance backports, and optional experimental plugins; its advertised baseline scripts
do not actually isolate those patches. That is too broad a trust and regression surface.

The candidate instead builds with networking disabled over the immutable official ARM64
image. It replaces one NVIDIA-authored FlashInfer dispatcher source file, removes the stale
precompiled copy so FlashInfer JIT compiles the reviewed source, and overlays the two reviewed
vLLM Python files needed to losslessly upcast E8M0 scales and avoid unsupported SM12x CUTLASS
shapes. The build verifies all source hashes, all four required TP1/TP2 dispatch arms, and
`nvcc`. No arbitrary install hooks, package indexes, third-party wheels, runtime patch scripts,
or remote code are used.

### Initial serve shape

- TP2 plus expert parallel over the registered RoCE fabric.
- FP8 KV cache, block size 256, 262,144-token operational ceiling, one active sequence slot.
- Prefix caching and async scheduling disabled during qualification.
- FlashInfer autotuning disabled; CUDA graphs remain enabled.
- DSpark speculative decoding disabled for the first correctness and speed baseline. It will
  only be enabled at depth 3 if the target-only lane passes and measured acceptance improves
  end-user decode speed.

### Status

The reviewed E8M0 overlay built successfully on both qualification nodes. Both report vLLM
`0.28.1rc1.dev137+g5ab628dd1`. Docker's classic and containerd stores expose distinct inspect
signatures, pinned as `59706219fb144af5dac42adc85e5bbf15ad35939abfd0dfb7c405040bc4a0972`
and `8f2079505aa02a5e24f6bb835bf645ed6a77b5544cda9fdca8f4e907bc6a500f` for
the E8M0-only build. After adding the output-projection fallback, the current signatures are
`899b8001539557f8095b4705863534ceb379e3ac93a7df6c22a079c909a4e13e` and
`fa6bdaf6dd184f26086f7309be6b0152bb00e8cd701c3bd5d6f1f35d45c4886b`.

The first launch loaded all weights but failed during memory profiling because CUTLASS requires
fp32 scales and the checkpoint stores lossless power-of-two scales as E8M0. The error was
`scaled_mm_helper.hpp:17: a_scales.scalar_type() == at::ScalarType::Float`; it matches vLLM
issue #47818 and is addressed by the reviewed PR #47988 overlay above. Qualification after that
overlay progressed to output projection, where the 2-D/3-D DeepGEMM layout mismatch failed at
`layout.hpp:40: t.dim() == N`; the target-only BF16 batched-matmul fallback above addresses that
second failure.

The final overlay launched cleanly and passed direct text, separated reasoning/content, exact
structured tool-call arguments, solid-blue image recognition, complete SSE streaming, and NCCL
NET/IB startup checks. A two-run quality battery passed all eight tool cases and all eight
executed-code cases. Every advertised sampling control was accepted independently: temperature,
top-p, top-k, min-p, presence/frequency/repetition penalties, max/min tokens, seed, stop strings,
stop token IDs, logprobs/top-logprobs, thinking token budget, and repetition detection. Request
logs confirmed the non-default values (repetition detection is accept/reject-only because its
object is not included in `SamplingParams` repr).

At about 50K input tokens, one request measured 33.8 s TTFT, 1,488 prefill tok/s, and 21.9 decode
tok/s. At about 101K it measured 56.7 s TTFT, 1,792 prefill tok/s, and 22.6 decode tok/s. Two
simultaneous 50K requests completed without errors, but shared scheduling was uneven: TTFT was
30.6-53.8 s and decode was 3.5-18.6 tok/s per request. The qualified profile therefore uses one
active sequence slot. The serialized N=2 rerun retained 21.7-22.8 decode tok/s, with TTFT of
33.3 s for the first request and 65.4 s for the second. Multiple clients therefore queue instead
of allowing a competing large prefill to starve an in-flight decode. The model's upstream
1,048,576-token limit is recorded as native, but this profile only serves and qualifies a
262,144-token ceiling.

## 2026-09-02 Anemll candidate

`deepseek-v4-flash-vision-anemll` is a separate, unvalidated candidate using Anemll source
`47503f8e38dadd4dededca798150db2619594fce` via immutable image
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
(vLLM `0.25.2.dev0+g752a3a504.d20260714`). Its build bakes the default-on patch train from
MiaAI-Lab commit `8494a492ad02423620c5740cab2a803ef54d0fb7`, the official model revision
`86f746b36186f0e567729a5c06a8c918caba82a9`, and that revision's pinned encoder. Image
builds completed on both qualification nodes with network disabled on 2026-09-02. The classic and
containerd Docker stores produced content signatures
`84ba1d3fc5b43188da20db113b054fb49b29275e2d8de64996a5c7b3bfe4548e` and
`d0c34dac3663a926125c03431bf562d0fef1d9d131fd4f759769c2e7335cd06b`; both are pinned as
allowed representations. The fallback remained live and healthy throughout the build.

Promotion builds on the deployment cluster produced the same patched vLLM version from the same offline context,
with additional Docker content representations
`a64c68527254b9a4aadbd6bc39f829825e20a41ba8c3e3baf326bfac2243e662` and
`bd1ff59fe48ffb07e14eb78af8a2db53160c9dd87c2a8718174005e0772895d4`. They are also pinned in
the allowlist. Trusted build discovery may report a newly built identity for review, while all
ordinary prepare and launch paths continue to reject unpinned content.

The first six-slot launch passed direct chat, reasoning, tools, Vision, streaming, and NCCL
NET/IB warmups. At about 50K input tokens, N=1 measured 52.3 s TTFT, 964 prefill tok/s, and
32.4 decode tok/s. At about 101K it measured 73.7 s TTFT, 1,380 prefill tok/s, and 35.3 decode
tok/s. Two simultaneous 50K requests exposed the same starvation pattern as the fallback:
TTFT was 38.7-79.7 s and decode was 6.3-26.9 tok/s. The candidate therefore serves one active
sequence. Its CUDA graph capture ceiling retains the recipe's qualified value of 42 rather than
shrinking with the serving queue: a trial at 7 reduced repeated N=1 decode to 23.9 tok/s.

The final one-slot/42-capture shape measured 32.5 s TTFT, 1,550 prefill tok/s, and 36.5 decode
tok/s at about 50K. Queued N=2 retained 34.7-35.7 decode tok/s with TTFT of 32.2-72.5 s. Queued
N=4 retained 33.2-39.5 decode tok/s with TTFT of 32.0-153.1 s. DSpark's acceptance varies with
the generated text; the 50K validation run reported 24.7% draft-token acceptance and mean
acceptance length 2.48 in its sampled metrics. Two quality-evaluation runs passed every tool and
executable-code case (100%/100%). The profile is qualified at 36 tok/s and remains the selected
qualification lane; the 22 tok/s preview profile remains the rollback.

All configured presets work. Independent API checks also accepted `temperature`, `top_p`,
`top_k`, presence/frequency/repetition penalties, max/min tokens, seed, string and token-ID stops,
logprobs/top-logprobs, and structured repetition detection. This speculative V2 runtime rejects
`min_p` and `thinking_token_budget`; neither is declared by its generic config. Direct/reasoning,
tool, Vision, streaming, and NCCL NET/IB launch warmups passed.

## Post-promotion tuning

The secondary cluster is the isolated tuning lane after promotion. A `k=4` probabilistic DSpark
trial was rejected during argument validation because this checkpoint's MTP has `n_predict=3`;
`num_speculative_tokens` must be divisible by 3. No model weights loaded and the retained rank
containers were removed before the next trial.

The valid `k=3` probabilistic trial passed all launch warmups. Three N=1 runs at about 50K
measured 40.4, 35.2, and 43.8 decode tok/s (median 40.4), with TTFT of 47.8, 32.4, and 33.2 s.
Sampled draft acceptance during the benchmark settled between 45.8% and 53.3%, roughly double
the `k=6` validation run. Queued N=2 retained 37.4-41.2 decode tok/s with 37.9-78.3 s TTFT.

The `k=3` greedy trial was more stable but slower: three N=1 runs measured 38.5, 38.5, and
39.3 decode tok/s, with 31.8-32.0 s TTFT.

Without speculative decoding, three N=1 runs measured only 27.0, 25.7, and 26.2 decode tok/s
(median 26.2), with 31.4-32.9 s TTFT. DSpark therefore provides a large net decode benefit on
this runtime.

Restoring `k=3` probabilistic and raising `max_num_batched_tokens` from 8192 to 8256 produced
41.8, 46.4, and 43.8 decode tok/s at about 50K (median 43.8), with stable 32.1-32.7 s TTFT.
Queued N=2 retained 37.6-39.4 decode tok/s. At about 101K, it measured 67.2 s TTFT, 1,514
prefill tok/s, and 39.6 decode tok/s. Two quality runs again passed 100% tools and 100% code;
runtime logs remained error-free.

Increasing the same lane to `k=9` reduced N=1 decode to 32.7 and 32.6 tok/s in two runs,
with 32.4-32.7 s TTFT. Together with the invalid `k=4` result and the slower `k=6`
checkpoint, this selects `k=3` probabilistic with 8256 batch tokens as the best tested shape.
After restoring that shape, a confirmation run measured 41.3 decode tok/s with 32.9 s TTFT.

## 2026-09-03 selective-patch comparison and promotion

A separately tagged image tested MiaAI-Lab source
`d828ddd89708b0216a3af124a57e44dd5c09cb37` with only its SHM ring-buffer recovery and
sequence-parallel prefill indexer patches added to the qualified patch train. The candidate
kept the winning `k=3` probabilistic and 8256-token batch shape. It launched cleanly, produced
error-free runtime logs, and passed two quality runs at 100% tools and 100% executable code.

The candidate did not produce a repeatable performance gain. After its one-time indexer compile,
two 50K runs measured 37.8 and 38.1 decode tok/s, 1,509-1,530 prefill tok/s, and 32.9-33.4 s
TTFT. The unchanged runtime measured 39.3-41.7 decode tok/s, 1,549-1,568 prefill tok/s, and
32.2-32.6 s TTFT in the same session. At about 101K, candidate runs measured 38.5-39.6 decode
tok/s, 1,464-1,482 prefill tok/s, and 68.7-69.5 s TTFT; unchanged controls measured 37.4-38.6
decode tok/s, 1,485-1,500 prefill tok/s, and 67.8-68.5 s TTFT. The mixed decode noise at 101K
did not offset the candidate's consistent prefill and TTFT regression or its 49.5 s cold compile.

The selective image was rejected. The existing reviewed image with `k=3` probabilistic and
8256 batch tokens is promoted as `deepseek-v4-flash-vision-anemll`; its conservative inventory
rating is 40 tok/s. Temporary tuning and selective profile names are not shipped.

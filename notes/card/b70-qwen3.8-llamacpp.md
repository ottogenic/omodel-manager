# Arc Pro B70 / Qwen3.8 llama.cpp Qualification

This is retained native-runtime evidence, not a normal `omodel-manager`
deployment. The supported B70 deployment is the qualified vLLM path documented
in `b70-qwen3.8-vllm.md`.

## Preserved Baseline

Snapshot: 2026-08-23 after the B70 host was cold-booted following an abandoned
RTX 4090 swap.

- GPU: Intel Arc Pro B70, `8086:e223`, BDF `0000:03:00.0`.
- Driver/render node: `xe`, `/dev/dri/renderD128`.
- VRAM: 32,656 MiB total; PCIe 4.0 x4.
- OS/runtime: Ubuntu 26.04, kernel `7.0.0-30-generic`, OMIX
  `0.3.0-9~26.04`.
- llama.cpp: `b10425`, commit
  `3d93885352a0049c8388a0da0698ec1a69e60d90`, IntelLLVM 2026.1.0.
- Server SHA-256:
  `f885280b5ff9e179d46f53d930780e782bf3ff5cdbd87bdcc4142da33c5e8ae0`.
- Model: `lmstudio-community/Qwen3.8-27B-GGUF` revision
  `5a7da681f60570ab5b439a587e912d2e5eddb582`, file
  `Qwen3.8-27B-Q4_K_M.gguf`, 16,810,714,336 bytes, SHA-256
  `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520`.

The retained launch used full offload, fitting disabled, 262,144 total context,
one slot, 2048/512 batch and ubatch, Flash Attention, Q4 or Q8 KV experiments,
embedded MTP2, loopback port 8000, and served alias `qwen3.8-27b`.

## Matched Results

| Prompt | Q4 weights / Q4 KV | Q4 weights / Q8 KV |
| --- | ---: | ---: |
| ~55K prefill | 406 tok/s | 408 tok/s |
| ~55K decode | 15.3 tok/s | 15.5 tok/s |
| ~110K prefill | 265 tok/s | 266 tok/s |
| ~110K decode | 11.3 tok/s | 10.4 tok/s |

Q8 KV used about 4.8 GiB more VRAM. It was later selected for coding-session
quality despite the throughput result; Q4_K_M/Q8 loaded at full context with
26,391 MiB used and 6,265 MiB free.

Q8_0 weights with Q8 KV were rejected. At 65,536 context they used 29,774.5 MiB
idle and produced 308 tok/s prefill and 13.5 tok/s decode at ~55K, versus 406
and 15.3 for Q4_K_M/Q4. The larger weights reduced context and performance.

## Later SYCL Builds

llama.cpp `b10427`, commit `65091386227039bfb81ee3426537656e3b4a3f83`,
improved matched decode to 17.3 tok/s at ~55K and 11.2 tok/s at ~110K. A later
combined `b10603` plus pending TILE/GDN work reached:

| Candidate | ~55K decode | ~110K decode |
| --- | ---: | ---: |
| retained b10427, Q8 KV | 17.3 | 11.2 |
| clean b10603, Q8 KV | 16.5 | 11.1 |
| TILE, Q8 KV | 22.4 | 15.8 |
| combined MTP4/Q8 | 26.1 | 17.8 |
| combined MTP4/F16, 131K | 28.5 | 21.9 |

The generic SYCL backend suite was not fully clean, so these remain targeted
hardware results rather than a general backend qualification.

## Safety Findings

- F16 KV at 131,072 loaded safely and was faster than Q8 on this implementation.
- An `8192/8192` batch/ubatch attempt exhausted VRAM, triggered Xe TTM eviction,
  consumed host RAM/swap, killed processes, and reset the compute engine. It is
  permanently rejected.
- A guarded 262K F16 attempt still caused driver-managed TTM pressure and an
  unclean host restart. User-service memory/swap limits did not contain the
  kernel eviction path. Do not retry without a new explicit safety design.
- 220,160 F16/no-MTP loaded safely with conservative `512/128`; a ~55K prompt
  measured 144 tok/s prefill and 16.5 tok/s decode.

Further native tuning was paused when the pinned vLLM GPTQ candidate materially
outperformed this lane. Raw retained comparisons are under `results/card/`.

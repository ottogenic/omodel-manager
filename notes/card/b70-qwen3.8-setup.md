# Arc Pro B70 / Qwen3.8 Setup Qualification

Research date: 2026-08-21. This is the reviewed pre-arrival and host bring-up
record. The resulting normal deployment is the checked-in vLLM helper; native
llama.cpp instructions below are historical qualification, not an `omm` backend.

## Hardware Gates

- Card: Intel Arc Pro B70, PCI ID `8086:e223`, expected on the upstream `xe`
  driver with an accessible stable render node.
- Host path: cold-plug OCuLink only. Never connect or disconnect either side
  while powered.
- Link target: PCIe 4.0 x4 (`16 GT/s`, width x4). Repeated AER errors, reduced
  width, intermittent enumeration, or reset faults are hardware gates.
- Keep the display on the host iGPU during bring-up. Power the dock before the
  host unless its manual specifies another sequence.
- Require at least three clean cold boots before runtime work and ten plus an
  8-24 hour soak before declaring the setup stable.

The qualified host later measured Ubuntu 26.04, kernel `7.0.0-30-generic`, OMIX
`0.3.0-9~26.04`, BDF `0000:03:00.0`, and a stable `/dev/dri/renderD128`.

## Software Baseline

The original baseline used native llama.cpp SYCL because it offered a direct,
debuggable Level Zero path before the vLLM candidate was proven. The reviewed
build was llama.cpp `b10425`, commit
`3d93885352a0049c8388a0da0698ec1a69e60d90`, with SYCL and F16 enabled and
host-memory fallback disabled. Full GPU offload and clean kernel logs were hard
requirements. See `b70-qwen3.8-llamacpp.md` for results.

The original model-fit estimate for 32 GiB was:

| Format | Approximate size | Initial conclusion |
| --- | ---: | --- |
| BF16 | 50.9 GiB | cannot fit |
| official FP8 | 28.8 GiB | too little runtime/cache margin |
| GGUF Q8_0 | 27.1 GiB | too tight for useful context |
| GGUF Q5_K_M | 19.3 GiB | explicit experiment only |
| GGUF Q4_K_M | 16.6 GiB | practical baseline |
| GGUF Q4_K_S | 15.6 GiB | original full-context candidate |

For 16 full-attention layers, F16 KV was estimated near 64 KiB/token; Q8 KV
roughly halves that portion. No result was accepted if weights or KV spilled to
host memory over the x4 link.

## Validation Order

1. Verify PCI identity, `xe`, link width/speed, render permissions, and clean
   kernel logs.
2. Verify the exact runtime and model artifact before launch.
3. Require deterministic generation, thinking on/off, reasoning effort, parsed
   tool calls, long output, and complete GPU offload.
4. Test 50K, 100K, and 200K contexts while observing VRAM, host memory, swap,
   power, temperature, PCIe state, and Xe/TTM/AER diagnostics.
5. Change quantization, KV type, MTP, vision, and concurrency one at a time.
6. Repeat warm reboot, cold boot, and soak gates before promotion.

The production path superseding this plan is:

```bash
python3 utils/card/deploy_b70_vllm.py plan b70 qwen3.8-27b-gptq-int4-b70
```

That plan is offline. A real launch additionally verifies the pinned model files,
render node, image, network, and retained container configuration.

## Primary Sources

- <https://www.asrock.com/Graphics-Card/Intel/Intel%20Arc%20Pro%20B70%20Creator%2032GB/>
- <https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html>
- <https://dgpu-docs.intel.com/installation-guides/installing-omix.html>
- <https://huggingface.co/Qwen/Qwen3.8-27B>
- <https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md>
- <https://docs.vllm.ai/en/latest/features/quantization/>

# Preserved GLM-5.3-Flash deployments

## Publication — 2026-10-04

The operator approved preserving both qualified builds for later quality and
accuracy comparisons. Their source of truth is the literal
`DEFAULT_CONFIG["cluster_models"]` in `omodel-manager`. Local sandbox overrides
are optional; a fresh checkout and `omm sync` retain both builds.

| | NVIDIA baseline | Eugr candidate |
| --- | --- | --- |
| Profile | `glm-5.3-flash-nim` | `glm-5.3-flash-eugr` |
| Backend | NIM/SGLang | vLLM/B12X |
| Registry image | `nvcr.io/nim/zai-org/glm-5.3-flash` | `eugr/spark-vllm-b12x` |
| Manifest SHA256 | `0bd2a1f4ffacf4ee61e1f8c1b48071a6713b51bad954ab0ca627c24be84f43f8` | `1803d1c23f746f2f559b1d119deb4517b857750c4fb283d0e96d500f681f1af8` |
| Weights | bundled `nim/zai-org/glm-5.3-flash:nim-aa28e1f-nvfp4` | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark@a608241037e4c2565356bff7ca293f2133888f88` |
| KV precision | BF16 | FP8 |
| Speculation | none | native MTP5 |
| Configured total context | 400000 | 500000 |
| Runtime sequence slots | 1 | 4 |
| Runtime cache under remote `~/.cache/otools/` | `glm-5.3-flash-nim/` | `zai-org/GLM-5.3-Flash/` |
| Detailed evidence | [NIM/SGLang](glm-5.3-flash-chachi.md) | [Eugr vLLM](glm-5.3-flash-eugr.md) |

Both serve `zai-org/GLM-5.3-Flash` on loopback port 8000 behind the existing
Tailscale forwarding. They own the whole two-Spark device, so only one runs on
Chachi at a time. The shared TOML retains native model context 1048576 as an
architecture fact; clients must use the live `/v1/models` deployed limit.
Its conservative one-worker cap is compatible with both servers.

## Prepare and switch

For an idle, registered two-Spark device needing artifacts (substitute its name):

```bash
python3 ./omodel-manager cluster prepare Chachi glm-5.3-flash-nim --build --weights
python3 ./omodel-manager cluster prepare Chachi glm-5.3-flash-eugr --build --weights
```

The first pulls the pinned NIM image, stages its hash-checked loopback/64 MiB
nginx template and pre-caches NVIDIA's bundled profile. The second pulls the
pinned Eugr image and exact HF snapshot on both hosts. Already cached artifacts
are reused. Source pins and recipe provenance are recorded in the detailed notes.
No floating `latest` tag or external launcher is needed.

For Chachi, both sets of artifacts are already cached. Save any desired logs,
stop the current deployment, and select one profile:

```bash
python3 ./omodel-manager logs Chachi head --tail 10000
python3 ./omodel-manager stop Chachi
python3 ./omodel-manager plan Chachi glm-5.3-flash-nim
python3 ./omodel-manager launch Chachi glm-5.3-flash-nim
python3 ./omodel-manager health Chachi

# Switch back to vLLM when ready:
python3 ./omodel-manager stop Chachi
python3 ./omodel-manager launch Chachi glm-5.3-flash-eugr
python3 ./omodel-manager health Chachi
```

`stop` removes deployment containers, not their images, bundled weights, HF
snapshots or persistent compilation caches. Failed launches retain rank logs.
NIM keeps `SGLANG_USE_PICKLE_IPC=0`; Eugr keeps `seccomp=unconfined` for io_uring
and `B12X_ROCE_SPIN_LIMIT=600000000` for cold-compilation rank skew.

## Comparison protocol

1. Record profile, image digest, weight revision, actual input usage, sampling,
   reasoning effort and output allowance with each result.
2. Replay identical text, tool schemas and original image bytes. Start with a
   context both have actually handled (~50–80K); qualify longer NIM inputs before
   using them as an accuracy comparison. NIM's 400K setting is not a passed 400K test.
3. Use one request at a time for accuracy. Repeat multiple samples; score answers
   and parsed tool arguments against the same expected results. Do not execute
   generated tools during an inference-only replay.
4. Keep cold and prefix-cached latency separate. A fresh system-prefix nonce can
   prevent prefix hits; preserve the rest of the prompt. Check runtime evidence
   because the earlier NIM API reported zero cached tokens despite logged reuse.
5. Record reasoning-inclusive decode speed, first visible answer and total wall
   time separately. Check natural completion: vLLM's tool parser can report
   `tool_calls` even when the output allowance was exhausted.

This is a comparison of **complete deployments**, not a controlled SGLang versus
vLLM engine test: quantized weights, KV precision, kernels and speculation differ.
NVIDIA's standalone HF quant is also distinct from the bundled NIM checkpoint.

Qualification establishes working chat/tools/vision/streaming, NIM's full-image
~80K replay, and Eugr's ten-image retrieval up to 497704 actual input tokens.
It does not establish checkpoint accuracy equivalence. One ambiguously worded
399818-token Eugr retrieval probe was refused; the corrected fictional-label
probe passed. That failure remains documented rather than omitted.

## Client transport

Near-500K cold vLLM prefill took about ten minutes. The existing client correction
is documented in [the Bun timeout investigation](glm-5.3-flash-connection-timeout.md).
Provider timeouts and the service's `BUN_CONFIG_HTTP_IDLE_TIMEOUT=1200` are
machine-local settings, not image contents. Updating a client must retain them;
the durable omodel-wire timeout policy remains separate work.

Raw replay requests and screenshots stay outside Git. The repository contains
configuration, source changes, tests and evidence summaries, not container layers
or model weight files.

## Publication checks

On 2026-10-04, both promoted profiles' head and worker Docker argv were compared
against the qualified sandbox and matched exactly. NIM's corrected API metadata
and explicit cache-name field change neither command nor existing cache path.
All 267 offline tests and Python compilation passed. After `omm sync`, both
device-first plans rendered and `health Chachi` reported the running vLLM model
ready with max context 500000. Publication did not require a serving restart.

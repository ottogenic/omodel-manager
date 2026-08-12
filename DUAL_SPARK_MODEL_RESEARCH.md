# Dual DGX Spark Model Research

Research date: 2026-08-09

## Objective

Identify models from the supplied OpenRouter catalog that meet all of these
requirements:

1. Downloadable weights are available for self-hosting.
2. The model cannot be served usefully on one 128 GB DGX Spark.
3. The model can be served usefully on two connected 128 GB DGX Sparks.
4. The model has credible coding, tool-use, or agentic capabilities.

This is a practical serving assessment, not a raw parameter-count filter.
"Fits" means that an approximately 4-bit checkpoint leaves enough memory for
the operating systems, serving runtime, activations, and a useful KV cache. A
model that only loads with a 64-token context is not considered a useful
coding-agent deployment.

For Mixture-of-Experts (MoE) models, all experts must remain in memory. The
active parameter count affects inference compute and speed, but does not reduce
the weight-storage requirement.

## Recommended Models

These models satisfy the selection criteria. Deployment maturity ranges from
official NVIDIA support to credible but unvalidated community configurations.

| OpenRouter ID | Total / active parameters | Practical checkpoint size | Two-Spark path | Assessment |
|---|---:|---:|---|---|
| `deepseek/deepseek-v4-flash-0731` | 284B / 13B | 166.9 GB mixed FP4/FP8 | Patched vLLM or SGLang, TP=2 | Excellent coding and agents; demonstrated on two GB10 systems |
| `deepseek/deepseek-v4-flash` | 284B / 13B | Similar | Same as 0731 | Older preview; prefer 0731 |
| `~deepseek/deepseek-v4-flash-latest` | Alias | Same as current target | Same as current target | Currently aliases 0731; pin 0731 for reproducibility |
| `qwen/qwen3-235b-a22b` | 235B / 22B | 134.1 GB NVFP4 | TensorRT-LLM TP=2 | Best officially documented two-Spark path |
| `qwen/qwen3-235b-a22b-2507` | 235B / 22B | 139.2 GB NVFP4 | TensorRT-LLM TP=2 | Strong non-thinking coding and tool loops |
| `qwen/qwen3-235b-a22b-thinking-2507` | 235B / 22B | 139.2 GB NVFP4 | TensorRT-LLM TP=2 | Strongest Qwen reasoning/coding option in this size class |
| `qwen/qwen3-vl-235b-a22b-instruct` | 236B / 22B | 135.3 GB NVFP4 | vLLM TP=2 | Visual coding, screenshot, GUI, and document agents |
| `qwen/qwen3-vl-235b-a22b-thinking` | 236B / 22B | 135.3 GB NVFP4 | vLLM TP=2 | Qualifies, but currently depends on a community quant |
| `tencent/hy3` | 295B / 21B | 169.6 GB NVFP4 | veloGB10 TP=2 | Explicitly demonstrated on two DGX Sparks |
| `thinkingmachines/inkling-small` | 276B / 12B | 170.7 GB NVFP4 | vLLM or SGLang TP=2 | Strong multimodal coding and tool capability |
| `z-ai/glm-4.5` | About 355B / 32B | About 192 GB IQ4/AWQ | llama.cpp RPC or vLLM | Good coding agent; community deployment path |
| `z-ai/glm-4.6` | About 357B / 32B | About 201 GB NVFP4 | vLLM TP=2 | Fits with less context and runtime headroom |
| `z-ai/glm-4.7` | About 358B / 32B | About 194 GB AWQ | vLLM TP=2 | Preferred GLM version; strong coding and terminal results |
| `minimax/minimax-m2` | 230B / 10B | 120.8 GB AWQ | vLLM or SGLang TP=2 | Coding/agent focused; one Spark is too constrained |
| `minimax/minimax-m2.1` | 230B / 10B | 124.9 GB AWQ | vLLM or SGLang TP=2 | Strong coding, planning, and tool use |
| `minimax/minimax-m2.5` | 230B / 10B | 130.2 GB AWQ | vLLM or SGLang TP=2 | Best generally usable MiniMax M2 choice |

Although the MiniMax M2 and M2.1 AWQ weights are nominally below 128 GB,
they leave inadequate space on one Spark for the OS, runtime, activations, and
a useful KV cache. They are operationally two-Spark models.

## Conditional Models

These models meet the memory window only with a particular quantization or
depend on an immature distributed runtime. They are useful experiments but are
not recommended as initial deployment targets.

| OpenRouter ID | Conditional assessment |
|---|---|
| `minimax/minimax-m2.7` | AWQ/NVFP4 needs two Sparks, but an aggressive 101 GiB IQ4 build can fit one. The model license is non-commercial without authorization. |
| `xiaomi/mimo-v2.5` | An approximately 185 GB GGUF fits two and the model is strongly agentic/multimodal. Optimized SGLang kernels do not support GB10, making experimental llama.cpp RPC the likely path. |
| `arcee-ai/trinity-large-thinking` | Official W4A16 is approximately 213 GB. It may load, but runtime and long-context headroom are limited. |
| `meta-llama/llama-4-maverick` | A prototype AWQ is approximately 215 GB. The more credible NVFP4 is approximately 234 GB and too tight. No validated Spark recipe was found. |
| `nousresearch/hermes-3-llama-3.1-405b` | Approximately 217 GB with IQ4_XS. Dense 405B decoding would be slow, and llama.cpp RPC remains experimental. |
| `minimax/minimax-m3` | GPTQ is approximately 224 GB. A constrained test may work, but runtime support and useful context capacity remain questionable. |

NVIDIA's own two-Spark 405B AWQ example limits the model length to 64 tokens
and warns that the deployment has insufficient production headroom. Normal
400B and 405B checkpoints are therefore not treated as reliable local coding
agents on two Sparks.

## Recommended Test Order

1. `deepseek/deepseek-v4-flash-0731` for coding-agent capability and fast MoE
   decoding, accepting that the demonstrated GB10 route uses a patched runtime.
2. `qwen/qwen3-235b-a22b-thinking-2507` for strong reasoning through the most
   NVIDIA-native model family in this capacity class.
3. `qwen/qwen3-235b-a22b-2507` for lower-latency, non-thinking tool loops.
4. `tencent/hy3` for another model with an explicitly demonstrated two-Spark
   path.
5. `thinkingmachines/inkling-small` for multimodal coding agents.
6. `z-ai/glm-4.7` for strong coding quality with more deployment work.
7. `minimax/minimax-m2.5` for fast 10B-active agent workloads.

## Exclusions

### API-Only Or Closed

The following families do not provide downloadable weights for the listed
hosted models:

- Anthropic Claude
- Google Gemini
- OpenAI GPT and o-series, except the GPT-OSS models
- xAI Grok
- Amazon Nova
- Perplexity Sonar
- Most hosted `Plus`, `Max`, `Pro`, `Flash`, and `latest` aliases

### Fit One Spark

These models may be useful locally, but do not require two Sparks at a practical
4-bit precision:

- 70B and 72B Llama, Hermes, Qwen, and roleplay derivatives
- Llama 4 Scout
- Qwen3-Next-80B
- Qwen3.5-122B-A10B
- NVIDIA Nemotron Super 120B
- GPT-OSS-120B
- Mistral Large 123B
- Mixtral 8x22B
- Cohere Command A
- Step Flash
- Poolside Laguna S and XS

### Too Large For Two Sparks

The following models require more than two Sparks at useful precision and
context sizes:

- DeepSeek V3, V3.1, V3.2, and R1 671B models
- DeepCogito 671B
- Moonshot Kimi K2 1T models
- NVIDIA Nemotron Ultra 550B
- Qwen3-Coder 480B-A35B
- GLM-5 744B models
- InclusionAI Ling and Ring 1T models
- LongCat 560B and 1.6T models
- MiMo-V2.5-Pro 1T
- Qwen3.5-397B-A17B at the official 251 GB NVFP4 size
- Hermes 4 405B at useful agent context lengths

### Missing A Practical Checkpoint Or Runtime

- ERNIE-4.5-VL-424B-A47B has a numerically plausible quant but no validated
  serving runtime for this deployment.
- MiniMax-01 and MiniMax-M1 lack a suitable released Linux 4-bit deployment
  checkpoint.
- Jamba Large requires more aggressive quantization and has a less practical
  Spark runtime path.

### Not Coding Or Agent Focused

Roleplay-oriented models such as Magnum, Euryale, Mythomax, M2-Her, and similar
finetunes were excluded even when their weights were downloadable.

## Networking And Runtime Notes

- Connect the Sparks using one approved QSFP112 cable over the ConnectX-7
  interfaces. NVIDIA's current guidance says a second cable does not improve
  two-node performance.
- The two systems do not form one coherent 256 GB CUDA memory pool. The serving
  engine must shard the model with tensor or pipeline parallelism.
- TensorRT-LLM TP=2 is the preferred path for NVIDIA Qwen NVFP4 checkpoints.
- NVIDIA's vLLM Ray playbook is the baseline for other supported architectures.
- llama.cpp RPC can split GGUF weights across hosts and may use RDMA, but its RPC
  backend is still described as experimental and should not be exposed to an
  untrusted network.
- Start validation at batch size one and 8K to 32K context. Advertised 128K,
  256K, or 1M contexts do not imply that the full context fits this hardware.

## Manager Implementation Status

`omodel-manager` v0.3 prework implements a separate two-node registry, management and
fabric preflight, exact-source preparation, and coordinated rank lifecycle. It refuses
heavy preparation while either Spark is serving another model and refuses launch unless
the configured peer route, jumbo ping, and active `mlx5` mapping all use the selected
QSFP/RoCE interface.

All three Qwen TensorRT-LLM profiles were physically qualified on a dual-Spark pair on 2026-08-11.
Their official NVIDIA weights are pinned to exact revisions. All three use the rc8 ARM64 digest
`sha256:c7297fdfb2e947296a4b29c8bdf8f5122e565179d65f806f85ff8fb4a478606e`,
which includes TensorRT-LLM PR #11956's Blackwell CUTLASS TMA fix. The manager builds and
records reviewed MPI derivatives separately for each immutable runtime. Base and Instruct
originally passed structured tool calls, tool-result continuation, and 12,515-token coding streams
on rc5, but Base later reproduced the upstream intermittent TMA descriptor crash and was moved to
rc8 before requalification.
Thinking passed reasoning/content separation, an automatic tool call and result loop, two
12,515-token sampled streams, post-run generation health, and clean kernel logs on both nodes.
Its final profile uses an explicit pinned tokenizer, disables FlashInfer sampling, reserves
KV at 0.60, and retains a 32K sequence limit with 8K chunked prefill.

DeepSeek V4 Flash 0731's current primary lane is c8r, pinned to the reviewed deployment kit at
`46eb0fcbadf0e4e0be8838b18f6aa85087ed8839` and full-source vLLM at
`48bada6ea49ad7f3ecbe03128aa76562089c8b00`. It carries 17 whole-file gx10 overlays,
FlashInfer `0.6.16.post3`, DeepGEMM `a6b593d2`, and separate `-c8r` compile caches. The
manager certifies matching image configuration/rootfs signatures and installed-content hashes
across the nodes despite their different Docker storage backends. Both 74-file model snapshots
must hash to the same 166,898,660,330-byte manifest. The prior cand7 source remains pinned at
`15f29b7bd91d45a1678b3b8600a56512c36f13f2` as the first explicit rollback rung with isolated
cache roots. Cand4 and cand7 stalled near 13.8K computed tokens under CUDA graphs, and later
real-agent traffic also wedged cand7 after eager-mode qualification, so cand7 is not treated as
the promoted stable lane. Worker-first launch refuses to run until the QSFP/RoCE route,
MTU-sized ping, and active HCA mapping pass; it then requires API health, the complete agent
warmup battery, and `NCCL NET/IB` evidence. No community model image or quant is used as a
fallback. C8r subsequently passed those physical gates, a roughly 100K-context local OpenCode
audit with repeated tools and 97% prefix-cache reuse, and a completed 20K-token stream with both
ranks healthy afterward. The 20K numeric prompt exhausted its budget in coherent reasoning and
returned empty visible content, matching the model's documented long numeric-constraint trait;
this is a quality limitation, not a runtime hang.

## Primary Sources

- [NVIDIA two-Spark TensorRT-LLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/trt-llm)
- [NVIDIA two-Spark vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/vllm)
- [NVIDIA ConnectX-7 clustering guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
- [DeepSeek V4 Flash 0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [Qwen3-235B-A22B NVFP4](https://huggingface.co/nvidia/Qwen3-235B-A22B-NVFP4)
- [Qwen3-235B-A22B Instruct 2507 NVFP4](https://huggingface.co/nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4)
- [Qwen3-235B-A22B Thinking 2507 NVFP4](https://huggingface.co/nvidia/Qwen3-235B-A22B-Thinking-2507-NVFP4)
- [Qwen3-VL-235B-A22B Instruct NVFP4](https://huggingface.co/nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4)
- [Tencent Hy3](https://huggingface.co/tencent/Hy3)
- [Thinking Machines Inkling Small NVFP4](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)
- [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7)
- [MiniMax M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [llama.cpp RPC documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc)

## Maintenance Note

Checkpoint availability, GB10 kernels, and distributed-serving support are
changing quickly. Recheck model cards, artifact sizes, licenses, and the latest
NVIDIA playbooks before implementing a profile or downloading hundreds of
gigabytes of weights.

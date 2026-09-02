# Arc Pro B70 / Qwen3.8 vLLM Qualification

This record documents the production B70 profile integrated into
`omodel-manager`.

## Pinned Identities

- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- Observed runtime: vLLM `0.27.2rc1.dev77+gac7509e2b.xpu`, XPU kernels
  `0.1.12.3`, Torch `2.13.0+xpu`.
- Model: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` at revision
  `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`.
- Served ID: `qwen3.8-27b-gptq-int4-b70`.
- Context: 262,144; GPTQ INT4 weights, BF16 MTP4, FP8 KV, one sequence, 8,192
  scheduler budget, 10 GiB explicit KV cache, prefix caching.

The model publisher is a third party. Qualification verifies every required
file size and SHA-256, contiguous Safetensors data, exact index membership, 2,399
tensors, and 15 BF16 MTP tensors. It does not independently prove derivation
from official Qwen weights.

## Isolation And Drift Gates

- Only `/dev/dri/renderD128` is exposed; no primary DRM node or broad
  `/dev/dri` access.
- The model uses an internal Docker bridge with no external route.
- A constrained read-only proxy publishes only `127.0.0.1:8000`.
- Both containers drop all capabilities, set `no-new-privileges`, have no
  restart policy, and carry exact manager/device/model/backend/role labels.
- The model container is limited to 24 GiB host memory and 25 GiB including
  swap; proxy limits are 64 MiB memory/swap and 32 PIDs.
- Existing same-name containers are reused only after exact image, argv,
  environment, mount, network, device, resource, and security verification.
- The old `org.omodel-card.profile` label is accepted only with the exact pinned
  legacy image and command so an earlier deployment can be stopped safely.

The official image needed two fail-closed patches. The single-rank XCCL warmup
guard changes `xpu_worker.py` from SHA-256
`bd3b35ad0e5ce23348810b9782f0934e9d9c9ad09398ff4808b3a789b0f9bce8` to
`115273d63c4273489ee210998f6153d6a4dea503e4623aebc759f9ef09d98d6d`.
The partial-MTP boundary fix changes `gdn_attn.py` from
`fda86b96ab5daaf50bd02d022518779c220401dbedc7b28cf478f4c48e72d3d3` to
`135799921da0d842aae828a23bdbce010ca08ab76848ea08b1e1c1736caf401a`.
Both refuse source drift.

## Qualification Results

The initial 100K gate loaded with 24,897 MiB observed VRAM and passed health,
identity, deterministic output, full-offload, host-memory, swap, and kernel-log
checks. A 4,281-token screen measured 1,815 tok/s prefill and 52.0 tok/s decode.

The 250K matched candidate used 28,385 MiB loaded VRAM and measured:

| Actual input | TTFT | Prefill | Decode |
| ---: | ---: | ---: | ---: |
| 54,744 | 42.6 s | 1,284 tok/s | 40.5 tok/s |
| 110,707 | 119.6 s | 925 tok/s | 36.8 tok/s |

The retained 262,144 profile passed an exact 262,016-token prompt plus 128-token
output request after the boundary patch. Coding-quality runs retained under
`results/card/` measured median B70 decode near 66 tok/s in the uncapped medium
suite, with 14/15 all-check generations and 44/45 hidden checks.

## Operation

```bash
omm plan b70 qwen3.8-27b-gptq-int4-b70
omm launch b70 qwen3.8-27b-gptq-int4-b70
omm logs b70 -f
omm health b70
omm stop b70 -y
```

The direct helper contract is available for diagnosis:

```bash
python3 utils/card/deploy_b70_vllm.py plan b70 qwen3.8-27b-gptq-int4-b70
```

Normal lifecycle should use `omm` so ownership, drift checks, and local or remote Docker
transport remain consistent.

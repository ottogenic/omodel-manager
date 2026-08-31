# DeepSeek V4 Flash 0731 build notes

## 2026-08-30 smpcache qualification candidate

### Scope

- Hardware: two DGX Spark GB10 nodes, Beebo (`thing-1` head, `thing-2` worker).
- Weights: `deepseek-ai/DeepSeek-V4-Flash-0731` at
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.
- Base runtime: existing reviewed c8r image, vLLM `48bada6e`, DeepGEMM `a6b593d`,
  B12X `7dc6fb8`, and FlashInfer `0.6.16.post3`.
- Candidate source: `Reederey87/dgx-spark-2x-deepseek-v4-flash` at
  `cc0d826b00629240f918bdf9b943feb59074c0d6`.
- Qualified image: `otools/vllm-deepseek-v4-flash-0731:smpcache-reviewed`.

The candidate applies the three effective final postimages from Reederey's promoted
`c8r-tbfix-ixfix-c128arev-smpcache` chain directly over the exact c8r base. The final
`sampler.py` includes the thinking-budget predicate and therefore subsumes the intermediate
tbfix file. The build manifest nevertheless binds all four source-kit trees and all three
postimage SHA-256 values.

### Promoted fixes under test

- `ixfix`, upstream vLLM #52492: keep learned indexer scoring active during breakable CUDA
  graph capture. Without it, a short dummy capture can silently constrain replayed cached
  prefixes to the first sparse candidates.
- `c128arev`, upstream vLLM #51318: restore fixed-capacity C128A metadata rows so eager
  writers and graph-captured consumers agree on row stride under high concurrency.
- `smpcache`, upstream vLLM #52329: cache per-slot logits-processing state, including
  `thinking_token_budget`, instead of scanning it each decode step.

### Settings intentionally unchanged

- `nvfp4_ds_mla`, explicit 19.85 GiB KV allocation, and 1,048,576 context.
- DSpark probabilistic draft length 2.
- `max_num_seqs=12`, batched-token budget 8192, long-prefill threshold 4096.
- Breakable CUDA graphs with capture size 72 and prefix caching enabled.
- Async scheduling disabled; no vLLM 0.28 rebase.

### Required gates

- Build and runtime signatures match on both ranks; candidate uses fresh generated caches.
- `/tokenize` produces three distinct, increasing low/high/max prompt encodings. This guards
  the declared Plan=max and Build=high policy against the known DeepSeek V4 mapping bug.
- Non-thinking, high, max, structured tool call, tool-result continuation, long prefill,
  sampled streaming, and long-agent streaming pass during launch.
- Repeat the prior long-generation failure boundary beyond 13.8K generated tokens with
  slot reuse and concurrency 10-12; inspect both ranks for graph replay, NCCL, Xid, and swap.
- A/B against c8r at realistic context and concurrency before promotion.

### Status

The candidate was built and qualified on Beebo on 2026-08-31. Both ranks had identical image,
runtime, and model signatures; each model snapshot was exactly 166,898,660,330 bytes. Launch
completed every tokenizer and request-path gate, captured CUDA graphs, enabled prefix caching,
allocated 3,027,217 KV-cache tokens, and established NCCL NET/IB between the nodes.

Quality and behavior results:

- `utils/quality_eval.py` passed all four tool tasks and all four coding tasks (100% each).
- Low, high, and max thinking budgets encoded distinctly. Requested 512, 1024, and 2048 token
  budgets produced increasing reasoning lengths, confirming that per-request budget state is
  active.
- A long agent response generated 15,135 completion tokens in 395.7 seconds and stopped
  normally, crossing cand7's prior approximately 13.8K-token failure boundary. It contained
  22,868 reasoning characters and 51,543 answer characters.
- A 12-request, approximately 10K-token concurrency run completed all requests. Together with
  the long response and the repeated cached-prefix probes below, this exercises full slot
  occupancy, slot reuse, and generation beyond the prior boundary.
- The cached-prefix indexer probe recalled 12/12 facts at 200,145 prompt tokens, split 6/6
  early and 6/6 late. All requests reused at least 199,936 cached tokens.
- The same probe recalled 6/6 facts at 943,401 prompt tokens, split 3/3 early and 3/3 late.
  All requests reused 943,104 cached tokens. Neither probe emitted special-token leakage,
  non-English garbage, mojibake, or repeated-line attractors.

Performance results:

| Context / concurrency | TTFT | Decode speed |
| --- | ---: | ---: |
| approximately 50K / 1 | 36.1 s | 39.7 tok/s |
| approximately 50K / 2 | 45.2-46.2 s | 25.5-28.3 tok/s |
| approximately 50K / 4 | 47.5-93.3 s | 1.7-15.7 tok/s |
| approximately 128K / 1 | 68.6 s | 35.9 tok/s |
| approximately 10K / 12 | 9.0-54.2 s | 2.8-10.5 tok/s |

The final controlled c8r/smpcache A/B used fresh, unique approximately 50K-token prompts after
each lane completed the same launch warmups:

| Lane / concurrency | TTFT | Decode speed |
| --- | ---: | ---: |
| c8r / 1 | 24.0 s | 38.8 tok/s |
| smpcache / 1 | 24.6 s | 37.9 tok/s |
| c8r / 2 | 45.4-46.3 s | 25.8-29.4 tok/s |
| smpcache / 2 | 46.0-47.0 s | 27.7-30.3 tok/s |

The lanes are effectively tied at concurrency 1. The candidate retains comparable TTFT and
slightly higher decode throughput at concurrency 2, with no observed performance regression.

After all qualification traffic, both rank logs were clean of Xid, CUDA, NCCL failure, MMU,
invalid-token, swap, traceback, engine-death, and OOM events, and `omm health Beebo` remained
`READY`. Startup produced expected first-shape Triton/TileLang JIT latency warnings only.

All technical qualification gates are complete. On promotion, smpcache became the validated
`deepseek-v4-flash-0731` preferred profile with a measured 38 tok/s at approximately 50K
context. The former c8r deployment remains available as `deepseek-v4-flash-0731-c8r`, and cand7
remains the deeper rollback.

---
name: benchmark-model
description: Benchmark a model's decode speed (tok/s) and TTFT at a real context and under concurrency using utils/benchmark_concurrent.py. Use to measure single-user speed for a profile's tok_s and to check the parallel slowdown.
---

# benchmark-model

Measure a running model's real-world speed: decode throughput (tok/s) and
time-to-first-token (TTFT) at a working context, and how much both degrade under
concurrency.

## Why this matters

Decode on this box is **memory-bandwidth-bound**, so a synthetic short-prompt
benchmark tells you nothing useful. What actually matters is:

- **tok/s at a working context** — the output speed a real user sees once a
  meaningful amount of context is loaded.
- **how much that drops under concurrency** — serving several requests at once
  competes for the same memory bandwidth, so throughput per request falls.

## When to use

- You've launched a model (see the **add-a-model** skill — this is invoked from
  its benchmarking step) and need to record the profile's `tok_s`.
- You want to know how a box holds up when more than one request runs at once.
- You want a stress test that mimics a large real repo/context.

## The tool

`utils/benchmark_concurrent.py <host> N` sends **N unique ~50k-token prompts at
once**. Each prompt is a generated story to summarize, and the stories are
**unique** so vLLM's prefix caching can't skip the prefill — every request pays
the full input-processing cost, the way a real distinct request would. It streams
the responses and reports, per request:

- **TTFT** — input-processing time (how long until the first token arrives).
- **decode tok/s** — output speed (tokens generated per second once decoding starts).

A **warm-up fires first**, so successive runs are directly comparable.

## How to run it

Single-user speed first — this is the number you record as the profile's `tok_s`:

```bash
utils/benchmark_concurrent.py <host> 1
```

Then raise the concurrency to see the slowdown:

```bash
utils/benchmark_concurrent.py <host> 2
utils/benchmark_concurrent.py <host> 4
```

Compare the per-request decode tok/s across 1 / 2 / 4 to quantify the parallel cost.

### Big-repo stress test

To mimic a large real context (e.g. a big repository), bump the context size:

```bash
utils/benchmark_concurrent.py <host> 1 --context 100000
```

## What to do with the results

- Record the `<host> 1` decode tok/s as the profile's `tok_s`.
- Note how much tok/s falls at `2` and `4` — that's the parallel slowdown you can
  expect when the box serves multiple users.

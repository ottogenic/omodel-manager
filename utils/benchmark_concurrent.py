#!/usr/bin/env python3
"""Benchmark concurrent throughput for all models in omodel-manager config.

Reads model profiles from model_manager.json, discovers their ports and
max-num-seqs, then sweeps concurrency levels to find the throughput sweet spot.

Usage:
    # Benchmark all models (reads config, probes each port)
    python3 utils/benchmark_concurrent.py

    # Benchmark specific profiles only
    python3 utils/benchmark_concurrent.py --profiles qwen3.6-35b-nvfp4 qwen3.6-35b-a3b-fp8

    # Benchmark with custom concurrency sweep
    python3 utils/benchmark_concurrent.py --max-concurrency 8 --concurrency-step 1

    # Use a specific config file
    python3 utils/benchmark_concurrent.py --config /path/to/model_manager.json

    # Skip warmup for faster runs
    python3 utils/benchmark_concurrent.py --skip-warmup

    # Custom prompt and token count
    python3 utils/benchmark_concurrent.py --max-tokens 512 --prompt "Your custom prompt here"
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import concurrent.futures
from pathlib import Path

# Defaults
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "model_manager.json")
DEFAULT_PROMPT = (
    "Solve this step by step: "
    "A train leaves Station A at 60 km/h. Two hours later, another train leaves Station A "
    "at 90 km/h in the same direction. Station B is 300 km from Station A. "
    "How long after the second train leaves will it catch up to the first train? "
    "Show your work."
)
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95

# Module-level prompt (set in main())
PROMPT = DEFAULT_PROMPT


def load_config(path):
    """Load the omodel-manager config file."""
    if not os.path.exists(path):
        print(f"ERROR: config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def merge_model(cfg, key):
    """Resolve a model profile with extends chain and defaults merge."""
    defaults = cfg.get("defaults", {})
    models = cfg.get("models", {})

    if key not in models:
        print(f"ERROR: no model '{key}' in config. Available: {', '.join(sorted(models))}",
              file=sys.stderr)
        sys.exit(1)

    # Resolve extends chain
    def resolve_entry(k, seen=None):
        seen = seen or set()
        if k in seen:
            print(f"ERROR: circular extends involving '{k}'", file=sys.stderr)
            sys.exit(1)
        seen.add(k)
        m = models[k]
        if m.get("extends"):
            base = resolve_entry(m["extends"], seen)
            child = {kk: vv for kk, vv in m.items() if kk != "extends"}
            return deep_merge(base, child)
        return dict(m)

    m = resolve_entry(key)

    # Merge with defaults
    merged = {
        "image": m.get("image", defaults.get("image", "")),
        "host": m.get("host", defaults.get("host", "0.0.0.0")),
        "port": m.get("port"),
        "model": m.get("model"),
        "vllm_args": {**defaults.get("vllm_args", {}), **m.get("vllm_args", {})},
        "env": {**defaults.get("env", {}), **m.get("env", {})},
    }

    if not merged["port"]:
        merged["port"] = 8000  # default port

    return merged


def deep_merge(base, over):
    """Deep merge two dicts."""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def make_request(idx, model, base_url, max_tokens, temperature, top_p):
    """Send a single chat completion request and return (idx, elapsed, output_tokens, success, error)."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "user", "content": PROMPT},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }).encode()
    req = urllib.request.Request(
        base_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        elapsed = time.time() - start
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
        return (idx, elapsed, completion_tokens, True, None)
    except Exception as e:
        elapsed = time.time() - start
        return (idx, elapsed, 0, False, str(e))


def run_benchmark(concurrency, model, base_url, max_tokens, temperature, top_p):
    """Run `concurrency` simultaneous requests and return aggregate stats."""
    print(f"  Concurrency {concurrency}: ", end="", flush=True)

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for i in range(concurrency):
            futures.append(pool.submit(make_request, i, model, base_url,
                                       max_tokens, temperature, top_p))

        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    successful = [r for r in results if r[3]]
    failed = [r for r in results if not r[3]]

    if not successful:
        return None

    total_time = max(r[1] for r in results)
    total_output_tokens = sum(r[2] for r in successful)
    avg_time = sum(r[1] for r in successful) / len(successful)
    throughput = total_output_tokens / total_time if total_time > 0 else 0

    status = "OK" if not failed else f"FAIL({len(failed)})"
    print(f"{concurrency:>2}x | {throughput:>6.1f} tok/s | {avg_time:>6.2f}s avg | {status}")

    return {
        "concurrency": concurrency,
        "total_time": total_time,
        "avg_latency": avg_time,
        "total_tokens": total_output_tokens,
        "throughput": throughput,
        "successful": len(successful),
        "failed": len(failed),
    }


def sweep_concurrency(model_name, model_info, base_url, max_tokens, temperature, top_p,
                      max_concurrency, step, warmup_count, skip_warmup, model_override=None):
    """Sweep concurrency levels for a single model and return results."""
    max_seqs = model_info["vllm_args"].get("max-num-seqs", 4)
    model_id = model_override if model_override else model_info.get("model", model_name)

    # Determine sweep range: from 1 up to max_seqs (or max_concurrency, whichever is lower)
    sweep_max = min(max_seqs, max_concurrency)
    levels = list(range(1, sweep_max + 1, step))
    if 1 not in levels:
        levels.insert(0, 1)

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"  HF ID: {model_id}")
    print(f"  max-num-seqs (config): {max_seqs}")
    print(f"  Sweeping concurrency: {levels}")
    print(f"{'='*70}")

    # Warmup
    if not skip_warmup:
        print(f"\n  Warmup ({warmup_count} requests)...")
        run_benchmark(min(warmup_count, 2), model_id, base_url,
                      max_tokens, temperature, top_p)
        time.sleep(2)

    results = []
    for level in levels:
        result = run_benchmark(level, model_id, base_url,
                               max_tokens, temperature, top_p)
        if result is None:
            print(f"  *** All requests failed at concurrency={level}, stopping. ***")
            break
        results.append(result)

        # Early exit if throughput drops significantly
        if len(results) >= 2:
            prev_tp = results[-2]["throughput"]
            curr_tp = results[-1]["throughput"]
            if curr_tp < prev_tp * 0.5:
                print(f"  *** Throughput dropped >50% from {prev_tp:.1f} to {curr_tp:.1f} tok/s. Stopping. ***")
                break

        if result["failed"] > 0:
            print(f"  *** {result['failed']} failures at concurrency={level}. Stopping. ***")
            break

        time.sleep(1)

    return results


def print_summary(model_results):
    """Print consolidated summary across all models."""
    print(f"\n{'='*70}")
    print("CONSOLIDATED SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Peak Throughput':>14} {'Best Concurrency':>16} {'Avg Latency':>12} {'Status'}")
    print("-" * 70)

    for model_name, results in model_results.items():
        if not results:
            print(f"{model_name:<30} {'N/A':>14} {'N/A':>16} {'N/A':>12} {'FAILED'}")
            continue

        best_tp = max(results, key=lambda r: r["throughput"])
        best_lat = min(results, key=lambda r: r["avg_latency"])
        status = "OK" if best_tp["failed"] == 0 else f"FAIL({best_tp['failed']})"

        print(f"{model_name:<30} {best_tp['throughput']:>13.1f} tok/s {best_tp['concurrency']:>16}x {best_lat['avg_latency']:>11.2f}s {status}")

    print(f"{'='*70}")
    print("\nRECOMMENDATION:")
    print("  Set each model's max-num-seqs to its peak throughput concurrency.")
    print("  Restart the container after updating the config.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark concurrent throughput for all models in omodel-manager config")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="path to model_manager.json (default: %(default)s)")
    parser.add_argument("--profiles", nargs="+", default=None,
                        help="specific profile keys to benchmark (default: all)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="max output tokens per request (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="sampling temperature (default: %(default)s)")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                        help="top-p sampling (default: %(default)s)")
    parser.add_argument("--max-concurrency", type=int, default=None,
                        help="max concurrency to test (default: profile's max-num-seqs)")
    parser.add_argument("--concurrency-step", type=int, default=1,
                        help="step between concurrency levels (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="number of warmup requests (default: %(default)s)")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="skip warmup")
    parser.add_argument("--prompt", default=None,
                        help="custom prompt (default: standard reasoning prompt)")
    parser.add_argument("--remote", default=None,
                        help="remote host to benchmark against (e.g. 192.168.50.102)")
    parser.add_argument("--model", default=None,
                        help="served model name to use in API requests (overrides config)")
    args = parser.parse_args()

    global PROMPT
    PROMPT = args.prompt if args.prompt else DEFAULT_PROMPT

    print(f"Benchmarking omodel-manager models")
    print(f"Config: {args.config}")
    print(f"Tokens: {args.max_tokens} | Temp: {args.temperature} | Top-p: {args.top_p}")
    print(f"Prompt: {PROMPT[:60]}...")
    if args.remote:
        print(f"Remote host: {args.remote}")

    # Load config
    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    models = cfg.get("models", {})

    if not models:
        print("ERROR: no models defined in config.", file=sys.stderr)
        sys.exit(1)

    # Determine which profiles to benchmark
    if args.profiles:
        profile_keys = args.profiles
    else:
        profile_keys = sorted(models.keys())

    # Resolve and benchmark each profile
    model_results = {}
    for key in profile_keys:
        try:
            model_info = merge_model(cfg, key)
        except SystemExit:
            continue

        # Build base URL from config (host:port), override with --remote if given
        host = args.remote if args.remote else model_info.get("host", "0.0.0.0")
        port = model_info.get("port", 8000)
        base_url = f"http://{host}:{port}/v1/chat/completions"

        results = sweep_concurrency(
            key, model_info, base_url,
            args.max_tokens, args.temperature, args.top_p,
            args.max_concurrency, args.concurrency_step,
            args.warmup, args.skip_warmup, args.model
        )
        model_results[key] = results

    # Print consolidated summary
    print_summary(model_results)


if __name__ == "__main__":
    main()
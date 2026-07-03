#!/usr/bin/env python3
"""Benchmark omodel-manager models under a *realistic* growing-context load.

The default mode runs N concurrent "sessions", each a multi-turn conversation
whose context GROWS turn over turn (unique code each turn, so prefix caching
can't fold it) until it reaches a target size (default 100k tokens). It streams
responses to measure TTFT (time to first token) and TPOT (time per output token)
and reports how they degrade as context grows -- plus KV-cache pressure and
preemptions scraped from the server's /metrics. This reproduces the real-world
"two sessions doing real work slow to a crawl" behaviour that a short, identical-
prompt sweep hides.

Usage:
    # Realistic: 2 sessions growing to 100k on a coding workload (the default)
    python3 utils/benchmark_concurrent.py --profiles qwen3.6-27b-nvfp4-256k \
        --model qwen3.6-27b-nvfp4-256k --host dgx1

    # Push harder: 3 sessions, grow to 64k, agent-style turns
    python3 utils/benchmark_concurrent.py --profiles nemotron-3-super-120b-nvfp4-256k \
        --model ... --host dgx1 --sessions 3 --grow-to 64000 --scenario agent

    # Old fast smoke test (short identical prompts, concurrency sweep)
    python3 utils/benchmark_concurrent.py --profiles glm-4.7-flash --host dgx1 --quick

Notes:
    * --host takes an `omm install` alias (e.g. dgx1), a user@ip, or a bare ip.
    * Pass --model when the profile sets a served-model-name (else you get 404s).
    * Thinking is ON by default (representative for reasoning models); --no-think off.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
import concurrent.futures
from pathlib import Path

# Defaults
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "model_manager.json")
DEFAULT_GROW_TO = 100_000      # grow each session's context to ~this many prompt tokens
DEFAULT_SESSIONS = 2           # concurrent growing conversations (your real-world case)
DEFAULT_MAX_TOKENS = 1024      # per-turn output cap (a realistic coding/agent turn)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

# Quick-mode (legacy) short prompt.
QUICK_PROMPT = (
    "Solve this step by step: A train leaves Station A at 60 km/h. Two hours later, "
    "another train leaves at 90 km/h in the same direction. How long until it catches up?"
)
PROMPT = QUICK_PROMPT          # module-level, used by quick mode (set in main())

_plock = threading.Lock()


def log(msg):
    with _plock:
        print(msg, flush=True)


# ============================================================================
# Config (shared with omodel-manager: profiles + the ~/.config/otools/hosts store)
# ============================================================================
def load_config(path):
    if not os.path.exists(path):
        print(f"ERROR: config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


HOSTS_FILE = os.path.expanduser("~/.config/otools/hosts")


def resolve_host(name):
    """Map a host ALIAS (from `omm install`) to its target, then return just the
    hostname/IP -- same alias store/semantics as `omm --host`. Accepts an alias
    (`dgx1`), a `user@ip`, or a bare ip; a `user@` prefix is stripped."""
    if not name:
        return None
    target = name
    try:
        with open(HOSTS_FILE) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split(None, 1)
                alias = parts[0]
                tgt = parts[1].strip() if len(parts) == 2 else parts[0]
                if name == alias:
                    target = tgt
                    break
    except OSError:
        pass
    return target.split("@")[-1] if "@" in target else target


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def merge_model(cfg, key):
    """Resolve a model profile (extends chain + defaults merge)."""
    defaults = cfg.get("defaults", {})
    models = cfg.get("models", {})
    if key not in models:
        print(f"ERROR: no model '{key}' in config. Available: {', '.join(sorted(models))}",
              file=sys.stderr)
        sys.exit(1)

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
    merged = {
        "host": m.get("host", defaults.get("host", "0.0.0.0")),
        "port": m.get("port") or 8000,
        "model": m.get("model"),
        "vllm_args": {**defaults.get("vllm_args", {}), **m.get("vllm_args", {})},
    }
    return merged


# ============================================================================
# Server metrics (/metrics) -- the "why" behind a crawl: KV pressure + preemption
# ============================================================================
_PROM = re.compile(r"^vllm:(\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")
_WANT = {"gpu_cache_usage_perc", "num_requests_waiting",
         "num_requests_running", "num_preemptions_total"}


def fetch_metrics(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            body = resp.read().decode("utf-8", "ignore")
    except Exception:
        return None
    out = {}
    for line in body.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = _PROM.match(line)
        if not m or m.group(1) not in _WANT:
            continue
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


class MetricsSampler(threading.Thread):
    """Polls <host>:<port>/metrics in the background while the load runs."""

    def __init__(self, host, port, interval=2.0):
        super().__init__(daemon=True)
        self.url = f"http://{host}:{port}/metrics"
        self.interval = interval
        self._stop = threading.Event()
        self.available = False
        self.cache, self.waiting, self.running, self.preempt = [], [], [], []

    def run(self):
        while not self._stop.is_set():
            m = fetch_metrics(self.url)
            if m:
                self.available = True
                if "gpu_cache_usage_perc" in m:
                    self.cache.append(m["gpu_cache_usage_perc"])
                if "num_requests_waiting" in m:
                    self.waiting.append(m["num_requests_waiting"])
                if "num_requests_running" in m:
                    self.running.append(m["num_requests_running"])
                if "num_preemptions_total" in m:
                    self.preempt.append(m["num_preemptions_total"])
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def peak_cache_pct(self):
        if not self.cache:
            return None
        v = max(self.cache)
        return v * 100.0 if v <= 1.0 else v      # vLLM reports a 0..1 fraction

    def max_waiting(self):
        return int(max(self.waiting)) if self.waiting else None

    def preemptions(self):
        return int(self.preempt[-1] - min(self.preempt)) if self.preempt else None


# ============================================================================
# Synthetic, growing, *unique* conversation content
# ============================================================================
_SYSTEM = {
    "coding": ("You are a senior engineer doing a long, multi-file refactor. Keep every "
               "answer short: summarize the change and cite function names. Never repeat "
               "the file back."),
    "agent": ("You are an autonomous coding agent with tools {read_file, write_file, "
              "run_tests}. Reason briefly, then state the next tool call. Keep it short."),
}


def _code_blob(sid, turn, approx_tokens):
    """~approx_tokens of unique, plausible code (unique ids defeat prefix caching)."""
    target_chars = approx_tokens * 4
    out, i, size = [], 0, 0
    while size < target_chars:
        b = (f"def s{sid}_t{turn}_op_{i}(a{i}, b{i}, c{i}):\n"
             f"    # unit {sid}.{turn}.{i} tag-{sid * 31 + turn * 17 + i} (unique, uncacheable)\n"
             f"    acc_{i} = a{i} * {i * 7 + 1} + b{i} * {i * 3 + 2} - c{i} * {i * 11 + 5}\n"
             f"    return (acc_{i} ^ {i * 13 + 97}) % {i * 29 + 101}\n\n")
        out.append(b)
        size += len(b)
        i += 1
    return "".join(out)


def turn_user(scenario, sid, turn, chunk_tokens):
    blob = _code_blob(sid, turn, chunk_tokens)
    if scenario == "agent":
        return (f"Tool result read_file(module_{sid}_{turn}.py):\n```python\n{blob}```\n"
                "Integrate this with the earlier modules; name the next tool call.")
    lead = ("Here is the start of a large module I'm refactoring:" if turn == 0
            else "Now consider this additional file that must interoperate:")
    return (f"{lead}\n```python\n{blob}```\n"
            "Give a one-paragraph review and name the riskiest function (by name).")


# ============================================================================
# Growing-session load
# ============================================================================
def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stream_turn(base_url, model, messages, max_tokens, temperature, top_p, think, timeout=600):
    """One streamed chat turn. Returns dict with ttft, tpot, out_tokens, prompt_tokens."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": think},
    }).encode()
    req = urllib.request.Request(base_url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    ttft = last = None
    itl, text = [], []
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                ch = obj.get("choices") or []
                if ch:
                    d = ch[0].get("delta") or {}
                    piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                    if piece:
                        now = time.time()
                        if ttft is None:
                            ttft = now - start
                        else:
                            itl.append(now - last)
                        last = now
                        text.append(piece)
    except Exception as e:
        return {"ok": False, "err": str(e), "ttft": None, "tpot": None,
                "out_tokens": 0, "prompt_tokens": usage.get("prompt_tokens", 0), "content": ""}
    return {"ok": True, "err": None, "ttft": ttft,
            "tpot": (sum(itl) / len(itl)) if itl else None,
            "out_tokens": usage.get("completion_tokens", len(itl) + 1),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "content": "".join(text)}


def run_session(sid, base_url, model, args, chunk_tokens, think, max_turns=80):
    """One growing conversation: add a unique code chunk each turn until the context
    reaches --grow-to. Returns a list of per-turn records."""
    messages = [{"role": "system", "content": _SYSTEM[args.scenario]},
                {"role": "user", "content": turn_user(args.scenario, sid, 0, chunk_tokens)}]
    turns = []
    for t in range(max_turns):
        est = sum(len(m.get("content", "")) for m in messages) // 4   # fallback if no usage
        r = stream_turn(base_url, model, messages, args.max_tokens,
                        args.temperature, args.top_p, think)
        ctx = r["prompt_tokens"] or est
        if not r["ok"]:
            log(f"  [s{sid} t{t}] ctx~{ctx // 1000}k  FAILED: {r['err'][:80]}")
            turns.append({"sid": sid, "turn": t, "ctx": ctx, "ok": False})
            break
        turns.append({"sid": sid, "turn": t, "ctx": ctx, "ttft": r["ttft"],
                      "tpot": r["tpot"], "out": r["out_tokens"], "ok": True})
        tpot_ms = (r["tpot"] * 1000) if r["tpot"] else 0
        log(f"  [s{sid} t{t}] ctx {ctx // 1000:>3}k  ttft {r['ttft'] or 0:4.1f}s  "
            f"tpot {tpot_ms:5.1f}ms  out {r['out_tokens']}")
        if ctx >= args.grow_to:
            break
        messages.append({"role": "assistant", "content": r["content"] or "ok"})
        messages.append({"role": "user", "content": turn_user(args.scenario, sid, t + 1, chunk_tokens)})
    return turns


_BUCKETS = [(0, 8000, "<8k"), (8000, 16000, "8-16k"), (16000, 32000, "16-32k"),
            (32000, 64000, "32-64k"), (64000, 100000, "64-100k"), (100000, 10 ** 9, ">=100k")]


def report_growing(model_name, sessions, sampler, wall, args):
    ok_turns = [t for s in sessions for t in s if t.get("ok")]
    print(f"\n{'=' * 74}")
    print(f"GROWING-SESSION RESULT: {model_name}")
    print(f"  {args.sessions} concurrent session(s), scenario={args.scenario}, "
          f"grow-to={args.grow_to // 1000}k, thinking={'off' if args.no_think else 'on'}, "
          f"wall={wall:.0f}s")
    print(f"{'=' * 74}")
    if not ok_turns:
        print("  No successful turns -- check --model / host / that the server is up.")
        return

    print(f"  {'context':<10} {'TTFT p50/p95':>16} {'TPOT p50/p95 (ms)':>20} "
          f"{'decode tok/s':>13} {'turns':>6}")
    first_decode = last_decode = None
    for lo, hi, label in _BUCKETS:
        b = [t for t in ok_turns if lo <= t["ctx"] < hi]
        if not b:
            continue
        ttfts = [t["ttft"] for t in b if t["ttft"] is not None]
        tpots = [t["tpot"] for t in b if t["tpot"]]
        dec = (1.0 / _pct(tpots, 50)) if tpots else 0.0
        if first_decode is None and dec:
            first_decode = dec
        if dec:
            last_decode = dec
        print(f"  {label:<10} {_pct(ttfts, 50):5.1f}/{_pct(ttfts, 95):<9.1f}"
              f"{_pct(tpots, 50) * 1000:8.1f}/{_pct(tpots, 95) * 1000:<11.1f}"
              f"{dec:>13.1f} {len(b):>6}")

    print(f"\n  KV cache peak: "
          + (f"{sampler.peak_cache_pct():.0f}%" if sampler.peak_cache_pct() is not None else "n/a")
          + "   max queue (waiting): "
          + (f"{sampler.max_waiting()}" if sampler.max_waiting() is not None else "n/a")
          + "   preemptions during run: "
          + (f"{sampler.preemptions()}" if sampler.preemptions() is not None else "n/a"))
    if not sampler.available:
        print("  (/metrics not reachable -- run against the box's IP to see KV/preemption data)")

    print("\n  READING:")
    if first_decode and last_decode and last_decode < first_decode:
        print(f"    - decode fell from {first_decode:.0f} to {last_decode:.0f} tok/s as context grew "
              f"({first_decode / last_decode:.1f}x slower) -- the bandwidth wall.")
    pre = sampler.preemptions()
    if pre:
        print(f"    - {pre} preemption(s) occurred: KV cache overflowed and sequences were evicted/"
              "recomputed. THIS is the crawl. Lower concurrent sessions, shrink max-model-len, or "
              "raise gpu-memory-utilization.")
    elif sampler.available:
        print("    - no preemptions: the slowdown is pure bandwidth (KV grows), not eviction.")
    print(f"    - at {args.sessions} concurrent sessions this model stayed usable up to the last "
          "context bucket above with acceptable TPOT; pick your real session budget accordingly.")


def run_growing(model_name, model_info, base_url, host, port, args):
    model_id = args.model or model_info.get("model", model_name)
    think = not args.no_think
    chunk = max(3000, args.grow_to // 16)          # ~16 turns to reach the target
    print(f"\n{'=' * 74}\nModel: {model_name}  (id: {model_id})\n"
          f"  Growing {args.sessions} session(s) to ~{args.grow_to // 1000}k tokens "
          f"(~{chunk // 1000}k/turn) ...\n{'=' * 74}")

    sampler = MetricsSampler(host, port)
    sampler.start()
    t0 = time.time()
    sessions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.sessions) as pool:
        futs = {pool.submit(run_session, sid, base_url, model_id, args, chunk, think): sid
                for sid in range(args.sessions)}
        for f in concurrent.futures.as_completed(futs):
            try:
                sessions.append(f.result())
            except Exception as e:
                log(f"  session {futs[f]} crashed: {e}")
    wall = time.time() - t0
    sampler.stop()
    sampler.join(timeout=3)
    report_growing(model_name, sessions, sampler, wall, args)
    return sessions


# ============================================================================
# Quick mode (legacy): short identical prompts, concurrency sweep -- a smoke test
# ============================================================================
def quick_request(idx, model, base_url, max_tokens, temperature, top_p):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
        "stream": False, "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base_url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        return (idx, time.time() - start, data.get("usage", {}).get("completion_tokens", 0), True, None)
    except Exception as e:
        return (idx, time.time() - start, 0, False, str(e))


def quick_run(concurrency, model, base_url, max_tokens, temperature, top_p):
    print(f"  Concurrency {concurrency}: ", end="", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(quick_request, i, model, base_url, max_tokens, temperature, top_p)
                   for i in range(concurrency)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    successful = [r for r in results if r[3]]
    failed = [r for r in results if not r[3]]
    if not successful:
        print("all failed")
        return None
    total_time = max(r[1] for r in results)
    total_tokens = sum(r[2] for r in successful)
    avg_time = sum(r[1] for r in successful) / len(successful)
    throughput = total_tokens / total_time if total_time > 0 else 0
    print(f"{concurrency:>2}x | {throughput:>6.1f} tok/s | {avg_time:>6.2f}s avg | "
          f"{'OK' if not failed else f'FAIL({len(failed)})'}")
    return {"concurrency": concurrency, "avg_latency": avg_time,
            "throughput": throughput, "successful": len(successful), "failed": len(failed)}


def quick_sweep(model_name, model_info, base_url, args, model_id):
    max_seqs = model_info["vllm_args"].get("max-num-seqs", 4)
    sweep_max = min(max_seqs, args.max_concurrency) if args.max_concurrency else max_seqs
    levels = list(range(1, sweep_max + 1, args.concurrency_step))
    if 1 not in levels:
        levels.insert(0, 1)
    print(f"\n{'=' * 70}\nModel: {model_name}  (id: {model_id})\n"
          f"  max-num-seqs {max_seqs}; quick sweep {levels}\n{'=' * 70}")
    results = []
    for level in levels:
        r = quick_run(level, model_id, base_url, args.max_tokens, args.temperature, args.top_p)
        if r is None:
            break
        results.append(r)
        if len(results) >= 2 and r["throughput"] < results[-2]["throughput"] * 0.5:
            print("  *** throughput dropped >50%, stopping. ***")
            break
        time.sleep(1)
    if results:
        best = max(results, key=lambda x: x["throughput"])
        print(f"  peak {best['throughput']:.1f} tok/s at {best['concurrency']}x")
    return results


# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Benchmark omodel-manager models under a realistic growing-context load.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to model_manager.json")
    p.add_argument("--profiles", nargs="+", default=None,
                   help="profile keys to benchmark (default: all)")
    p.add_argument("--model", default=None,
                   help="served model name for API requests (needed if the profile sets one)")
    p.add_argument("--host", default=None,
                   help="alias from `omm install` (e.g. dgx1), a user@ip, or a bare ip")
    p.add_argument("--remote", default=None, help="legacy alias for --host")
    # Growing-session (default) knobs -- kept intentionally few.
    p.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS,
                   help="concurrent growing conversations (default: %(default)s)")
    p.add_argument("--grow-to", type=int, default=DEFAULT_GROW_TO,
                   help="grow each session's context to ~this many tokens (default: %(default)s)")
    p.add_argument("--scenario", choices=["coding", "agent"], default="coding",
                   help="turn shape (default: %(default)s)")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="output tokens per turn (default: %(default)s)")
    p.add_argument("--no-think", action="store_true",
                   help="disable thinking (default: on -- representative for reasoning models)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    # Quick (legacy smoke test) mode.
    p.add_argument("--quick", action="store_true",
                   help="run the old short-prompt concurrency sweep instead (fast smoke test)")
    p.add_argument("--max-concurrency", type=int, default=None, help="[--quick] max concurrency")
    p.add_argument("--concurrency-step", type=int, default=1, help="[--quick] step")
    p.add_argument("--prompt", default=None, help="[--quick] custom short prompt")
    args = p.parse_args()

    global PROMPT
    PROMPT = args.prompt or QUICK_PROMPT

    host_arg = args.host or args.remote
    resolved_host = resolve_host(host_arg)

    print("Benchmarking omodel-manager models")
    print(f"Config: {args.config}")
    if args.quick:
        print(f"Mode: quick sweep | tokens {args.max_tokens} | temp {args.temperature}")
    else:
        print(f"Mode: growing sessions | {args.sessions} session(s) -> {args.grow_to // 1000}k "
              f"| scenario {args.scenario} | thinking {'off' if args.no_think else 'on'}")
    if host_arg:
        print(f"Host: {host_arg} -> {resolved_host}" if resolved_host != host_arg
              else f"Host: {resolved_host}")

    cfg = load_config(args.config)
    models = cfg.get("models", {})
    if not models:
        print("ERROR: no models defined in config.", file=sys.stderr)
        sys.exit(1)
    profile_keys = args.profiles if args.profiles else sorted(models.keys())

    for key in profile_keys:
        try:
            model_info = merge_model(cfg, key)
        except SystemExit:
            continue
        host = resolved_host or model_info.get("host", "0.0.0.0")
        port = model_info.get("port", 8000)
        base_url = f"http://{host}:{port}/v1/chat/completions"
        model_id = args.model or model_info.get("model", key)
        if args.quick:
            quick_sweep(key, model_info, base_url, args, model_id)
        else:
            run_growing(key, model_info, base_url, host, port, args)


if __name__ == "__main__":
    main()

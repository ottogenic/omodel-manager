#!/usr/bin/env python3
"""Benchmark ANY running vLLM/OpenAI endpoint at a FIXED large context.

This is a generic throughput probe -- it does NOT read the omodel-manager config or
care which profile is running. Point it at a host; it auto-discovers the served model
via /v1/models. It sends a single big prompt (~100k tokens of unique code, so prefix
caching can't fold it) at **concurrency 1**, then **2**, then **3** ... measuring TTFT
(time to first token) and TPOT (time per output token) at each level, and scraping the
server's /metrics for KV-cache pressure and preemptions. From that you get the two
numbers that matter: single-user decode speed at a full context (the `tok_s` column),
and how many concurrent users survive before it degrades (`max-num-seqs`).

Each concurrency level is ONE round (not a long growing conversation), so results
arrive fast and bounded -- if a level times out/fails twice in a row the sweep stops
and recommends the last level that completed. This avoids the "never finishes on a slow
model" problem of walking the context up turn by turn.

Usage:
    # Default: 100k context, sweep concurrency 1..4, model auto-discovered
    python3 utils/benchmark_concurrent.py --host dgx1

    # Lighter/heavier: smaller context, sweep to 6, agent-style prompt
    python3 utils/benchmark_concurrent.py --host 192.168.50.102 --context 64000 \
        --sessions 6 --scenario agent

    # Old fast smoke test (short identical prompts, concurrency sweep)
    python3 utils/benchmark_concurrent.py --host dgx1 --quick

Notes:
    * --host takes an `omm install` alias (e.g. dgx1), a user@ip, or a bare ip.
    * --model is optional -- auto-discovered from /v1/models; pass it to override.
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

# Defaults
DEFAULT_CONTEXT = 100_000      # fixed prompt size (~tokens) every request runs at
DEFAULT_SESSIONS = 4           # sweep concurrency 1..this many parallel requests
DEFAULT_MAX_TOKENS = 256       # output tokens per request (enough to measure TPOT; keeps it fast)
DEFAULT_REQ_TIMEOUT = 300      # seconds per request before it's counted a failure
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_PORT = 8000

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
# Host resolution (shared with omodel-manager's ~/.config/otools/hosts store)
# ============================================================================
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


def discover_model(host, port):
    """Return the first served model id from /v1/models, or None."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=10) as r:
            data = json.loads(r.read().decode())
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return ids[0] if ids else None
    except Exception:
        return None


# ============================================================================
# Server metrics (/metrics) -- the "why" behind a crawl: KV pressure + preemption
# ============================================================================
_PROM = re.compile(r"^vllm:(\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")
# vLLM V1 renamed gpu_cache_usage_perc -> kv_cache_usage_perc; accept both.
_WANT = {"gpu_cache_usage_perc", "kv_cache_usage_perc", "num_requests_waiting",
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
        self._done = threading.Event()
        self.available = False
        self.cache, self.waiting, self.running, self.preempt = [], [], [], []

    def run(self):
        while not self._done.is_set():
            m = fetch_metrics(self.url)
            if m:
                self.available = True
                if "kv_cache_usage_perc" in m:
                    self.cache.append(m["kv_cache_usage_perc"])
                elif "gpu_cache_usage_perc" in m:
                    self.cache.append(m["gpu_cache_usage_perc"])
                if "num_requests_waiting" in m:
                    self.waiting.append(m["num_requests_waiting"])
                if "num_requests_running" in m:
                    self.running.append(m["num_requests_running"])
                if "num_preemptions_total" in m:
                    self.preempt.append(m["num_preemptions_total"])
            self._done.wait(self.interval)

    def stop(self):
        self._done.set()

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


def build_prompt(scenario, sid, context_tokens):
    """One big ~context_tokens prompt (unique code + an instruction) as a messages list."""
    blob = _code_blob(sid, 0, context_tokens)
    if scenario == "agent":
        user = (f"Tool result read_file(module_{sid}.py):\n```python\n{blob}```\n"
                "Summarize what this module does and name the next tool call.")
    else:
        user = (f"Here is a large module to review:\n```python\n{blob}```\n"
                "Give a one-paragraph review and name the riskiest function (by name).")
    return [{"role": "system", "content": _SYSTEM[scenario]},
            {"role": "user", "content": user}]


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


def _run_level(n, base_url, model, ctx, args, think):
    """Fire n parallel single-shot requests, each a fresh ~ctx prompt. Returns
    (ok, record). A level fails if ANY request errors/times out -- that's the ceiling."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(stream_turn, base_url, model,
                            build_prompt(args.scenario, sid, ctx), args.max_tokens,
                            args.temperature, args.top_p, think, args.req_timeout)
                for sid in range(n)]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    good = [r for r in results if r["ok"] and r["ttft"] is not None]
    for r in results:
        if not r["ok"]:
            log(f"    request failed: {(r['err'] or 'unknown')[:70]}")
    if len(good) < n:
        return False, None
    for r in good:
        actual = (r["prompt_tokens"] or ctx) // 1000
        log(f"    ctx {actual:>3}k  ttft {r['ttft']:5.1f}s  "
            f"tpot {(r['tpot'] or 0) * 1000:5.0f}ms  out {r['out_tokens']}")
    ttfts = [r["ttft"] for r in good]
    tpots = [r["tpot"] for r in good if r["tpot"]]
    return True, {"n": n, "ttft50": _pct(ttfts, 50), "ttft95": _pct(ttfts, 95),
                  "tpot50": _pct(tpots, 50), "tpot95": _pct(tpots, 95),
                  "decode": (1.0 / _pct(tpots, 50)) if tpots else 0.0, "status": "OK"}


def sweep_fixed(model_id, base_url, host, port, args):
    """Sweep concurrency 1..--sessions, each at a FIXED ~--context prompt. Stops after a
    level fails twice in a row and recommends the last level that completed."""
    ctx = args.context
    think = not args.no_think
    print(f"\n{'=' * 74}\nModel: {model_id}\n"
          f"  Fixed ~{ctx // 1000}k context, sweeping concurrency 1..{args.sessions} "
          f"(scenario={args.scenario}, thinking={'off' if args.no_think else 'on'}) ...\n{'=' * 74}")
    sampler = MetricsSampler(host, port)
    sampler.start()
    t0 = time.time()
    rows, recommended = [], 0
    for n in range(1, args.sessions + 1):
        log(f"-- concurrency {n} @ ~{ctx // 1000}k ...")
        ok, rec = _run_level(n, base_url, model_id, ctx, args, think)
        if not ok:
            log(f"   concurrency {n} failed; retrying once ...")
            ok, rec = _run_level(n, base_url, model_id, ctx, args, think)
        if not ok:
            log(f"   concurrency {n} failed twice -- stopping the sweep.")
            rows.append({"n": n, "status": "FAILED x2"})
            break
        rows.append(rec)
        recommended = n
    wall = time.time() - t0
    sampler.stop()
    sampler.join(timeout=3)
    report_fixed(model_id, ctx, rows, recommended, sampler, wall, args)
    return rows


def report_fixed(model_id, ctx, rows, recommended, sampler, wall, args):
    print(f"\n{'=' * 74}")
    print(f"FIXED-CONTEXT RESULT: {model_id}  (~{ctx // 1000}k context, wall={wall:.0f}s)")
    print(f"{'=' * 74}")
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    if not ok_rows:
        print("  No concurrency level completed -- check --model / host / that the server is up, "
              "or try a smaller --context.")
        return
    print(f"  {'users':<6} {'TTFT p50/p95':>16} {'TPOT p50/p95 (ms)':>20} "
          f"{'decode tok/s':>13} {'status'}")
    for r in rows:
        if r.get("status", "").startswith("FAILED"):
            print(f"  {r['n']:<6} {'—':>16} {'—':>20} {'—':>13} {r['status']}")
        else:
            print(f"  {r['n']:<6} {r['ttft50']:5.1f}/{r['ttft95']:<9.1f}"
                  f"{r['tpot50'] * 1000:8.0f}/{r['tpot95'] * 1000:<11.0f}"
                  f"{r['decode']:>13.1f} OK")

    print(f"\n  KV cache peak: "
          + (f"{sampler.peak_cache_pct():.0f}%" if sampler.peak_cache_pct() is not None else "n/a")
          + "   preemptions during run: "
          + (f"{sampler.preemptions()}" if sampler.preemptions() is not None else "n/a"))
    if not sampler.available:
        print("  (/metrics not reachable -- run against the box's IP to see KV/preemption data)")

    single = next((r for r in ok_rows if r["n"] == 1), None)
    print("\n  READING:")
    if single:
        print(f"    - Tk/s (1 user @ ~{ctx // 1000}k): {single['decode']:.0f}  "
              "-> record as the profile's `tok_s` (the `list` Tk/s column).")
    print(f"    - Recommended max-num-seqs: {recommended}  "
          f"(highest concurrency that completed at ~{ctx // 1000}k).")
    if any(r.get("status", "").startswith("FAILED") for r in rows):
        print("    - the sweep stopped after a level failed twice (see the errors above -- a timeout"
              "/preemption is a real ceiling; a dropped connection means the server restarted, so "
              "re-run). Use the last completed level as `max-num-seqs`.")


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


def quick_sweep(model_id, base_url, args):
    sweep_max = args.max_concurrency or 8
    levels = list(range(1, sweep_max + 1, args.concurrency_step))
    if 1 not in levels:
        levels.insert(0, 1)
    print(f"\n{'=' * 70}\nModel: {model_id}\n  quick sweep {levels}\n{'=' * 70}")
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
        description="Benchmark any running vLLM endpoint at a fixed large context, sweeping concurrency.")
    p.add_argument("--host", default=None,
                   help="alias from `omm install` (e.g. dgx1), a user@ip, or a bare ip")
    p.add_argument("--remote", default=None, help="legacy alias for --host")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="server port (default: %(default)s)")
    p.add_argument("--model", default=None,
                   help="served model name (default: auto-discovered from /v1/models)")
    # Fixed-context sweep (default) knobs -- kept intentionally few.
    p.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS,
                   help="sweep concurrency 1..this many parallel requests (default: %(default)s)")
    p.add_argument("--context", "--grow-to", type=int, default=DEFAULT_CONTEXT, dest="context",
                   help="fixed prompt context, ~tokens, for every request (default: %(default)s)")
    p.add_argument("--req-timeout", type=int, default=DEFAULT_REQ_TIMEOUT, dest="req_timeout",
                   help="seconds per request before it counts as a failure (default: %(default)s)")
    p.add_argument("--scenario", choices=["coding", "agent"], default="coding",
                   help="prompt shape (default: %(default)s)")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="output tokens per request (default: %(default)s)")
    p.add_argument("--no-think", action="store_true",
                   help="disable thinking (default: on -- representative for reasoning models)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    # Quick (legacy smoke test) mode.
    p.add_argument("--quick", action="store_true",
                   help="run the old short-prompt concurrency sweep instead (fast smoke test)")
    p.add_argument("--max-concurrency", type=int, default=None, help="[--quick] max concurrency (default 8)")
    p.add_argument("--concurrency-step", type=int, default=1, help="[--quick] step")
    p.add_argument("--prompt", default=None, help="[--quick] custom short prompt")
    args = p.parse_args()

    global PROMPT
    PROMPT = args.prompt or QUICK_PROMPT

    host_arg = args.host or args.remote
    if not host_arg:
        print("ERROR: pass --host (an alias from `omm install`, a user@ip, or an ip).", file=sys.stderr)
        sys.exit(1)
    host = resolve_host(host_arg)
    port = args.port
    base_url = f"http://{host}:{port}/v1/chat/completions"

    model_id = args.model or discover_model(host, port)
    if not model_id:
        print(f"ERROR: couldn't discover a model at http://{host}:{port}/v1/models. "
              "Is the server up? Pass --model to override.", file=sys.stderr)
        sys.exit(1)

    print("Benchmarking a live vLLM endpoint")
    print(f"Host: {host_arg} -> {host}:{port}" if host_arg != host else f"Host: {host}:{port}")
    print(f"Model: {model_id}" + ("  (auto-discovered)" if not args.model else ""))
    if args.quick:
        print(f"Mode: quick sweep | tokens {args.max_tokens} | temp {args.temperature}")
        quick_sweep(model_id, base_url, args)
    else:
        print(f"Mode: fixed context | ~{args.context // 1000}k | sweep 1..{args.sessions} users "
              f"| scenario {args.scenario} | thinking {'off' if args.no_think else 'on'}")
        sweep_fixed(model_id, base_url, host, port, args)


if __name__ == "__main__":
    main()

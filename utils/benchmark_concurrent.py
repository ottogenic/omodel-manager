#!/usr/bin/env python3
"""Measure how fast a vLLM/OpenAI endpoint handles a big prompt.

Sends N **unique** ~<context>-token prompts (a generated story) at the SAME time and
asks each to summarize it in a paragraph, streaming the reply. The text is unique per
request so prefix caching can't skip the prefill. Per request it reports:

  - TTFT (time to first token)  = how long to PROCESS the input (prefill)
  - prefill tok/s = input_tokens / TTFT
  - decode  tok/s = output_tokens / (first-to-last-token time)   -> "how fast it types"

Run at N=1 for single-user speed, then N=2, N=4 to see the parallel slowdown. A small
warm-up request fires first so the timed numbers aren't a cold-start outlier (and runs
stay comparable to each other).

Usage:
    benchmark_concurrent.py <endpoint> [N]
    benchmark_concurrent.py dgx-3 1                   # one ~50k prompt, timed
    benchmark_concurrent.py dgx-3 2                   # two ~50k prompts at the same time
    benchmark_concurrent.py dgx-3 4 --context 100000  # bigger, big-repo stress size

    <endpoint> = an `omm install` alias (dgx-3), a user@ip, an ip, or host:port.
    --model is auto-discovered from /v1/models; --context sets the prompt size (~tokens).
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.request

DEFAULT_CONTEXT = 50_000       # ~everyday coding context; use --context 100000 for big-repo stress
DEFAULT_MAX_TOKENS = 300       # summary length -> enough output tokens to time decode
DEFAULT_TIMEOUT = 300          # wall-clock seconds per request
DEFAULT_PORT = 8000

HOSTS_FILE = os.path.expanduser("~/.config/otools/hosts")


def resolve_host(name):
    """An `omm install` alias / user@ip / ip / host -> the bare host to connect to."""
    target = name
    try:
        with open(HOSTS_FILE) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split(None, 1)
                if parts[0] == name:
                    target = parts[1].strip() if len(parts) == 2 else parts[0]
                    break
    except OSError:
        pass
    return target.split("@")[-1] if "@" in target else target


def discover_model(host, port):
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=10) as r:
            ids = [m.get("id") for m in json.load(r).get("data", []) if m.get("id")]
        return ids[0] if ids else None
    except Exception:
        return None


def make_story(sid, approx_tokens):
    """~approx_tokens of UNIQUE prose (so prefix caching can't fold the prefill). `sid`
    (which request) and the paragraph number are baked into every line, so each of the
    N concurrent prompts is different."""
    names = ["Mira", "Tomas", "Aisha", "Ravi", "Lena", "Odell", "Priya", "Sven"]
    places = ["the harbor at Kessel", "a mountain waystation", "the glass markets of Yara",
              "an orbital greenhouse", "the flooded archive", "the salt flats of Dune-9"]
    out, i, size = [], 0, 0
    target = approx_tokens * 4      # ~4 chars/token
    while size < target:
        para = (f"Chapter {i} (thread {sid}). {names[(sid + i) % len(names)]} met "
                f"{names[(sid + i + 3) % len(names)]} at {places[(sid * 2 + i) % len(places)]}, "
                f"carrying ledger {sid * 100003 + i}. They argued over the {i * 7 % 13} missing "
                f"crates and a promise made {i * 3 + sid} winters ago; nothing was resolved, and "
                f"the tide rose by {i % 9} spans while the lamps guttered low over the water.\n")
        out.append(para)
        size += len(para)
        i += 1
    return "".join(out)


def run_one(base_url, model, story, max_tokens, timeout):
    """Stream one summarize request; return timing/token stats."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": f"{story}\nSummarize the text above in one short paragraph."}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base_url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    first = last = None
    n = 0
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
                delta = (ch[0].get("delta") or {}) if ch else {}
                # Reasoning models may stream their entire completion before the
                # final answer through either reasoning field.
                if (delta.get("content") or delta.get("reasoning")
                        or delta.get("reasoning_content")):
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    n += 1
    except Exception as e:
        return {"ok": False, "err": str(e)}
    if first is None:
        return {"ok": False, "err": "no output tokens"}
    ttft = first - start
    gen = (last - first) if (last and n > 1) else 0.0
    # Count TOKENS, not SSE chunks. With speculative decoding vLLM emits several accepted
    # tokens per chunk, so chunk-counting (n) under-reports decode speed by up to ~Nx.
    # usage.completion_tokens is the true count; fall back to n only if usage is absent.
    out_toks = usage.get("completion_tokens", n)
    return {"ok": True, "ttft": ttft,
            "in_toks": usage.get("prompt_tokens", 0),
            "out_toks": out_toks,
            "prefill_tps": (usage.get("prompt_tokens", 0) / ttft) if ttft else 0.0,
            "decode_tps": ((out_toks - 1) / gen) if gen > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser(
        description="Measure how fast an endpoint processes a big prompt, at N parallel requests.")
    ap.add_argument("endpoint", help="alias from `omm install`, or user@ip / ip / host[:port]")
    ap.add_argument("n", nargs="?", type=int, default=1,
                    help="number of simultaneous prompts (default: 1)")
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                    help="prompt size in ~tokens (default: %(default)s)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="port (default: %(default)s)")
    ap.add_argument("--model", default=None, help="served model (default: auto-discover)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="summary length cap (default: %(default)s)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help="wall-clock seconds per request (default: %(default)s)")
    args = ap.parse_args()

    ep, port = args.endpoint, args.port
    if ":" in ep:                                  # accept host:port in the endpoint
        head, _, tail = ep.rpartition(":")
        if tail.isdigit():
            ep, port = head, int(tail)
    host = resolve_host(ep)
    base_url = f"http://{host}:{port}/v1/chat/completions"

    model = args.model or discover_model(host, port)
    if not model:
        print(f"ERROR: no model at http://{host}:{port}/v1/models (is it up? pass --model).",
              file=sys.stderr)
        sys.exit(1)

    print(f"Endpoint: {ep} -> {host}:{port}")
    print(f"Model:    {model}" + ("  (auto)" if not args.model else ""))
    print(f"Prompt:   ~{args.context // 1000}k tokens (unique), summarize | "
          f"{args.n} simultaneous request(s)")

    # Warm-up: one tiny request so the timed numbers aren't a cold-start outlier and
    # runs stay comparable across N. Timing discarded.
    print("  warming up ...", flush=True)
    run_one(base_url, model, make_story(999, 200), max_tokens=16, timeout=min(args.timeout, 60))

    # Make repeated benchmark invocations unique too. Without a run nonce, a
    # server with prefix caching can reuse the same sid=0 story from an earlier
    # N=1 run and report a meaningless near-zero TTFT.
    nonce = time.time_ns()
    stories = [f"Benchmark run {nonce}, request {sid}.\n" + make_story(sid, args.context)
               for sid in range(args.n)]
    results = [None] * args.n
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as pool:
        futs = {pool.submit(run_one, base_url, model, stories[s], args.max_tokens, args.timeout): s
                for s in range(args.n)}
        pending = set(futs)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=15)
            for f in done:
                results[futs[f]] = f.result()
            if pending:
                print(f"  ... {len(pending)}/{args.n} still processing "
                      f"({time.time() - t0:.0f}s elapsed) ...", flush=True)
    wall = time.time() - t0

    print(f"\n  {'req':<4} {'input':>9} {'TTFT':>8} {'prefill tok/s':>14} {'decode tok/s':>13}  status")
    oks = []
    for i, r in enumerate(results, 1):
        if not r or not r.get("ok"):
            print(f"  {i:<4} {'-':>9} {'-':>8} {'-':>14} {'-':>13}  FAIL: "
                  f"{(r or {}).get('err', '?')[:45]}")
            continue
        oks.append(r)
        print(f"  {i:<4} {r['in_toks']:>9,} {r['ttft']:>7.1f}s {r['prefill_tps']:>14,.0f} "
              f"{r['decode_tps']:>13.1f}  ok")

    if oks:
        ttfts = [r["ttft"] for r in oks]
        pf = [r["prefill_tps"] for r in oks]
        dc = [r["decode_tps"] for r in oks]
        inavg = sum(r["in_toks"] for r in oks) // len(oks)
        print(f"\n  Summary (N={args.n} @ ~{inavg // 1000}k input, wall {wall:.0f}s):")
        print(f"    TTFT (time to process the input):  {min(ttfts):.1f}-{max(ttfts):.1f} s")
        print(f"    prefill (input) throughput:        {min(pf):,.0f}-{max(pf):,.0f} tok/s")
        print(f"    decode (output) speed:             {min(dc):.1f}-{max(dc):.1f} tok/s")
        print("\n  Tip: decode tok/s at N=1 -> a profile's `tok_s` (the `list` Tk/s column). "
              "Compare N=1 vs N=2/4 for the parallel slowdown.")


if __name__ == "__main__":
    main()

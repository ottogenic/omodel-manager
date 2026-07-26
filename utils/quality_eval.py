#!/usr/bin/env python3
"""
quality_eval.py -- repeatable QUALITY battery for the Qwen3.6-35B-A3B A/B.

Stdlib only. Runs two graded suites against an OpenAI-compatible vLLM endpoint,
N times each (quality is intermittent -- see vLLM #47087 -- so we score over runs,
not single-shot):

  1. TOOL-CALL battery: scripted /v1/chat/completions requests with `tools=[...]`.
     Grades: correct tool selected, args well-formed JSON, no hallucinated tool
     name, multi-step chains, and NO degenerate looping (response != max_tokens
     of repeated text).

  2. CODE-QUALITY suite: self-contained coding problems. The model's fenced code
     block is extracted and executed against hidden unit tests in a subprocess.
     Scored pass/fail per run -> pass-rate over N runs.

Usage:
    python3 utils/quality_eval.py <host> [--runs 5] [--model NAME]
    <host>: alias/user@ip/ip/host:port (same forms as benchmark_concurrent.py)

Writes a JSON report to ./quality_eval_<served-id>_<ts>.json and prints a summary.
No third-party deps; talks HTTP via urllib.
"""
import argparse, json, re, sys, time, urllib.request, urllib.error, subprocess, tempfile, os, statistics

# ---------------------------------------------------------------- endpoint utils
HOSTS_FILE = os.path.expanduser("~/.config/otools/hosts")  # shared with omm (`install` writes it)


def resolve_host(name):
    """An `omm install` alias / user@ip / ip / host -> the registered target (or itself).
    Mirrors utils/benchmark_concurrent.py so both tools accept the same endpoint forms."""
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
    return target


def resolve_base(host):
    if "://" in host:
        host = host.split("://", 1)[1]
    name, _, port = host.partition(":")
    name = resolve_host(name)              # alias -> user@ip; raw hosts pass through
    if "@" in name:
        name = name.split("@", 1)[1]
    return f"http://{name}:{port or 8000}/v1"

def http_json(url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def discover_model(base):
    return http_json(base + "/models")["data"][0]["id"]

def chat(base, model, messages, tools=None, max_tokens=32768, temperature=0.6,
         top_p=0.95, extra=None, timeout=900):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p,
            "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}}
    # flatten extra_body into request (vLLM accepts top_k/chat_template_kwargs top-level via extra_body)
    req = {"model": model, "messages": messages, "max_tokens": max_tokens,
           "temperature": temperature, "top_p": top_p, "top_k": 20,
           "presence_penalty": 0.5,
           "chat_template_kwargs": {"enable_thinking": True}}
    if tools:
        req["tools"] = tools
        req["tool_choice"] = "auto"
    if extra:
        req.update(extra)
    t0 = time.time()
    resp = http_json(base + "/chat/completions", req, timeout=timeout)
    dt = time.time() - t0
    return resp, dt

# ---------------------------------------------------------------- loop detector
def looks_degenerate(text, finish_reason, completion_tokens):
    """Heuristic for the #47087 failure mode: ran to max_tokens with repetition,
    or empty-after-thinking."""
    if text is None:
        text = ""
    t = text.strip()
    if finish_reason == "length" and completion_tokens and completion_tokens > 4000:
        # ran to the cap -- check for repetition
        words = t.split()
        if len(words) > 40:
            # any 8-gram repeated 3+ times?
            grams = [" ".join(words[i:i+8]) for i in range(len(words)-8)]
            from collections import Counter
            c = Counter(grams)
            if c and c.most_common(1)[0][1] >= 3:
                return True, "repetition-to-length"
        return True, "ran-to-length"
    if not t and (completion_tokens or 0) > 200:
        return True, "empty-after-thinking"
    if re.search(r"(!{6,}|(.)\2{40,})", t):
        return True, "degenerate-chars"
    return False, ""

# ---------------------------------------------------------------- tool-call suite
WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}
ADD_TOOL = {"type": "function", "function": {
    "name": "add_numbers",
    "description": "Add two integers and return the sum.",
    "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"]}}}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_files",
    "description": "Search the codebase for a regex pattern.",
    "parameters": {"type": "object", "properties": {
        "pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern"]}}}

TOOL_CASES = [
    {"name": "single_weather", "tools": [WEATHER_TOOL],
     "prompt": "What's the weather in Paris right now? Use the tool.",
     "expect_tool": "get_weather", "expect_args_keys": ["city"]},
    {"name": "single_add", "tools": [ADD_TOOL],
     "prompt": "Use the tool to add 4839 and 1725.",
     "expect_tool": "add_numbers", "expect_args_keys": ["a", "b"],
     "expect_args": {"a": 4839, "b": 1725}},
    {"name": "select_among_three", "tools": [WEATHER_TOOL, ADD_TOOL, SEARCH_TOOL],
     "prompt": "Find every function definition (regex 'def ') in the src/ directory.",
     "expect_tool": "search_files", "expect_args_keys": ["pattern"]},
    {"name": "no_hallucinated_tool", "tools": [WEATHER_TOOL, ADD_TOOL, SEARCH_TOOL],
     "prompt": "Search the repo for the string TODO under the lib/ folder.",
     "expect_tool": "search_files", "expect_args_keys": ["pattern"]},
]

def grade_tool_case(resp, case):
    """Return (passed:bool, reason:str)."""
    try:
        msg = resp["choices"][0]["message"]
        fr = resp["choices"][0].get("finish_reason")
        usage = resp.get("usage", {})
    except Exception as e:
        return False, f"malformed-response:{e}"
    ct = usage.get("completion_tokens")
    calls = msg.get("tool_calls") or []
    # Only run the loop/degeneracy detector when there is NO tool call. A valid
    # tool-call response legitimately has empty `content`, which would otherwise
    # trip the empty-after-thinking heuristic (false positive).
    if not calls:
        deg, why = looks_degenerate(msg.get("content"), fr, ct)
        if deg:
            return False, f"looping:{why}"
        return False, "no-tool-call"
    fn = calls[0]["function"]["name"]
    valid_names = {t["function"]["name"] for t in case["tools"]}
    if fn not in valid_names:
        return False, f"hallucinated-tool:{fn}"
    if fn != case["expect_tool"]:
        return False, f"wrong-tool:{fn}!={case['expect_tool']}"
    raw = calls[0]["function"].get("arguments", "")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return False, f"invalid-json-args:{raw[:60]!r}"
    for k in case["expect_args_keys"]:
        if k not in args:
            return False, f"missing-arg:{k}"
    if "expect_args" in case:
        for k, v in case["expect_args"].items():
            if str(args.get(k)) != str(v):
                return False, f"wrong-arg-value:{k}={args.get(k)}!={v}"
    return True, "ok"

# ---------------------------------------------------------------- code-quality suite
def extract_code(text):
    if not text:
        return ""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else ""

CODE_CASES = [
    {"name": "fizzbuzz",
     "prompt": ("Write a Python function `fizzbuzz(n)` that returns a list of strings "
                "for 1..n: 'Fizz' if divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if both, "
                "else the number as a string. Return ONLY a ```python code block."),
     "test": ("assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']\n"
              "assert fizzbuzz(15)[-1] == 'FizzBuzz'\n"
              "assert fizzbuzz(15)[2] == 'Fizz' and fizzbuzz(15)[4] == 'Buzz'\n")},
    {"name": "merge_intervals",
     "prompt": ("Write a Python function `merge(intervals)` that merges overlapping "
                "intervals given as a list of [start,end] and returns the merged list "
                "sorted by start. Return ONLY a ```python code block."),
     "test": ("assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
              "assert merge([[1,4],[4,5]]) == [[1,5]]\n"
              "assert merge([]) == []\n")},
    {"name": "roman",
     "prompt": ("Write a Python function `to_roman(n)` converting an integer 1..3999 to "
                "a Roman numeral string. Return ONLY a ```python code block."),
     "test": ("assert to_roman(4)=='IV'\nassert to_roman(9)=='IX'\n"
              "assert to_roman(58)=='LVIII'\nassert to_roman(1994)=='MCMXCIV'\n")},
    {"name": "balanced_parens",
     "prompt": ("Write a Python function `is_balanced(s)` returning True iff the brackets "
                "in s (), [], {} are correctly balanced/nested. Return ONLY a ```python code block."),
     "test": ("assert is_balanced('([]{})')==True\nassert is_balanced('([)]')==False\n"
              "assert is_balanced('')==True\nassert is_balanced('(((')==False\n")},
]

def run_code(code, test):
    if not code:
        return False, "no-code-extracted"
    prog = code + "\n\n" + test + "\nprint('ALL_PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=15)
        ok = "ALL_PASS" in r.stdout
        return ok, ("ok" if ok else (r.stderr.strip().splitlines() or ["fail"])[-1][:120])
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)

# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    base = resolve_base(args.host)
    model = args.model or discover_model(base)
    print(f"endpoint={base}  model={model}  runs={args.runs}", flush=True)

    # warm-up
    try:
        chat(base, model, [{"role": "user", "content": "hi"}], max_tokens=8, timeout=300)
    except Exception as e:
        print(f"warm-up failed: {e}", file=sys.stderr)

    report = {"model": model, "endpoint": base, "runs": args.runs,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "tool": {}, "code": {}}

    # tool battery
    print("\n== TOOL-CALL BATTERY ==", flush=True)
    for case in TOOL_CASES:
        passes, reasons = 0, []
        for i in range(args.runs):
            try:
                resp, dt = chat(base, model,
                                [{"role": "user", "content": case["prompt"]}],
                                tools=case["tools"], max_tokens=args.max_tokens)
                ok, why = grade_tool_case(resp, case)
            except Exception as e:
                ok, why = False, f"exc:{type(e).__name__}:{e}"
            passes += ok
            if not ok:
                reasons.append(why)
        rate = passes / args.runs
        report["tool"][case["name"]] = {"pass": passes, "n": args.runs,
                                        "rate": rate, "fails": reasons}
        print(f"  {case['name']:22s} {passes}/{args.runs}  "
              f"{'FAIL '+';'.join(sorted(set(reasons))) if reasons else 'ok'}", flush=True)

    # code battery
    print("\n== CODE-QUALITY SUITE ==", flush=True)
    for case in CODE_CASES:
        passes, reasons = 0, []
        for i in range(args.runs):
            try:
                resp, dt = chat(base, model,
                                [{"role": "user", "content": case["prompt"]}],
                                max_tokens=args.max_tokens)
                msg = resp["choices"][0]["message"]
                fr = resp["choices"][0].get("finish_reason")
                ct = resp.get("usage", {}).get("completion_tokens")
                deg, why = looks_degenerate(msg.get("content"), fr, ct)
                if deg:
                    ok, why = False, f"looping:{why}"
                else:
                    ok, why = run_code(extract_code(msg.get("content")), case["test"])
            except Exception as e:
                ok, why = False, f"exc:{type(e).__name__}:{e}"
            passes += ok
            if not ok:
                reasons.append(why)
        rate = passes / args.runs
        report["code"][case["name"]] = {"pass": passes, "n": args.runs,
                                        "rate": rate, "fails": reasons}
        print(f"  {case['name']:22s} {passes}/{args.runs}  "
              f"{'FAIL '+';'.join(sorted(set(reasons))) if reasons else 'ok'}", flush=True)

    tool_rate = statistics.mean([v["rate"] for v in report["tool"].values()])
    code_rate = statistics.mean([v["rate"] for v in report["code"].values()])
    report["summary"] = {"tool_pass_rate": tool_rate, "code_pass_rate": code_rate}
    print(f"\nSUMMARY  tool={tool_rate:.0%}  code={code_rate:.0%}", flush=True)

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    out = f"quality_eval_{safe}_{int(time.time())}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"report -> {out}", flush=True)

if __name__ == "__main__":
    main()

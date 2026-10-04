# GLM-5.3-Flash on Chachi

## Preserved baseline — 2026-10-04

The operator approved keeping the tested MessagePack-enabled NVIDIA NIM/SGLang
build alongside the Eugr vLLM build for later quality comparisons. The exact
baseline is now curated as `DEFAULT_CONFIG.cluster_models.glm-5.3-flash-nim`;
`omm sync` preserves it. Its API ID is `zai-org/GLM-5.3-Flash`, while an explicit
`runtime_cache_name=glm-5.3-flash-nim` retains the qualified bundled-weight cache.
Configured context/sequence metadata is 400000/1; launch still uses the tested
NIM passthrough arguments rather than `NIM_MAX_MODEL_LEN`.

The dated investigation below is historical; later findings supersede earlier
candidate settings and unresolved diagnoses. [Comparison setup and current
profiles](glm-5.3-flash-deployments.md) summarize what is deployable now.

## Qualification status — 2026-10-02

### MessagePack serving trial — 2026-10-03, operator approved

**Sustained-run follow-up, 2026-10-03:** operator reports several hours of
screenshot-driven webpage work without crash or timeout after applying the Bun
environment setting. Read-only inspection around 17:45 UTC found both original
container processes still running with `OOMKilled=false`. Retained continuous
single-request decode rates ranged from medians 11.12 tok/s below 25K to 8.00
tok/s at 125–150K. See [client-policy and speed research](glm-5.3-flash-client-and-speed-research.md)
for the measurement method, omodel-wire timeout ownership, speculative-decoding
blockers and candidate runtime comparisons. This does not qualify an actual
400K multimodal input yet.

**Connection-cutoff root cause subsequently isolated:** the installed OpenCode
executable's Bun 1.4.2 times out a loopback-only delayed-header request after
360.311 seconds. The same executable succeeds after 400 seconds with
`BUN_CONFIG_HTTP_IDLE_TIMEOUT=1200` or per-fetch `timeout:false`. NVIDIA and
Tailscale are absent from this reproduction. Live nginx inference timeout is
four hours. The operator subsequently applied the service environment workaround;
the restarted service (2.0.22, PID 148637) was verified with
`BUN_CONFIG_HTTP_IDLE_TIMEOUT=1200` in its actual process environment. See
[connection-timeout evidence and correction](glm-5.3-flash-connection-timeout.md).

**Subsequent real-session observation, 03:29:29–03:39:29 UTC:** 54 samples;
minimum MemAvailable 2.829/3.390 GiB, ending at 4.222/4.976 GiB. Head swap used
4.398→4.444 GiB; worker stayed 3.270 GiB. No observed container OOM/restart or
monitor alert. Memory pressure was not absent: peak sampled full-pressure avg10
was 5.67% head / 56.84% worker, indicating a transient worker stall despite
remaining RAM. Evidence: `/tmp/opencode/glm53-live-watch-20261003-032929/` and
`/tmp/opencode/glm53-live-watch-run.log`.

The user completed a real OpenCode service restart: `/api/info` now reports
2.0.22/PID 144802 and the provider exposes headerTimeout=1200000. Nevertheless,
the real session still retries with generic `provider.transport` errors; nginx
recorded HTTP 499 at 03:30:20 and 03:36:20. Thus activating 2.0.22 did not resolve
the six-minute connection cutoff. Root cause remains unresolved; do not claim
the prior timeout adjustment is an end-to-end fix.

Retries are new inference submissions. Scheduler logs show prefix reuse of
53,248 tokens at 03:32:35 and 77,824 at 03:39:24, with queue count zero in observed
prefill lines. These support re-admission/cache reuse, not a growing queue of
simultaneous full inference jobs. The pinned tokenizer detects disconnects while
waiting and dispatches abort; scheduler marks chunked requests for deferred
abort. Frontend image work and in-flight chunks can still overlap/waste work
during cancellation. No per-request-ID trace proves all cleanup timing here.

Operator approved relaunching and testing after restarting their OpenCode interface.
Added only `SGLANG_USE_PICKLE_IPC=0` to the ignored sandbox profile and confirmed
the device-first plan includes it on both ranks. Both hosts were idle. Retained
exited containers still reserved Chachi, so the initial launch refused; cleared
them with `stop Chachi --yes` after the prior OOM evidence had been preserved,
then launched `Chachi glm-5.3-flash-nim` again. Container inspection confirms the
flag on both ranks, with the same 400K context/420K BF16 KV pool settings,
single active request, pinned image/checkpoint and loopback binding.

Launch evidence: `/tmp/opencode/glm53-msgpack-launch-2.log`; selected environments:
`/tmp/opencode/glm53-msgpack-{head,worker}-env.json`. Replay diagnostic prepared at
`/tmp/opencode/glm53_msgpack_retest.py`: exact saved nine-image request, reconstructed
ten-image cold request and cached repeat, with five-second both-host memory
sampling and a low-memory watchdog. Generated tools are recorded, never executed.
Launch completed successfully: stock readiness checks passed chat, reasoning,
tools, vision, streaming and both ranks; a separate `health Chachi` also passed.
Both ranks allocated 420,000 BF16 KV tokens/4.41 GiB and ten Mamba slots/0.76 GiB;
available memory after graph capture was 7.65/7.76 GiB. Startup logs are preserved
at `/tmp/opencode/glm53-msgpack-startup-{head,worker}.log`.

Full-resolution replay results (same original nine image bytes; tenth screenshot
retained at 2000x1250). HTTP requests use simplified reconstructed system/tool
schemas, so the ten-image request measures 79,540 input tokens rather than the
original OpenCode request's roughly 83K. Cold requests use distinct system prefixes.

| Request | Input/image tokens | First response | Result | Minimum available RAM head/worker |
| --- | ---: | ---: | --- | ---: |
| Nine images, cold | 75,940 / 25,164 fixture | 577.712 s | Test-reader exception after first content; server healthy | 2.964 / 5.254 GiB |
| Nine images, cached retry | 75,940 / 25,164 | 19.513 s | 225 output tokens, parsed tool call, DONE and health pass | 2.558 / 4.881 GiB |
| Ten images, cold | 79,540 / 28,404 | 642.216 s | 512 reasoning tokens, length cap, DONE and health pass | 2.205 / 5.019 GiB |
| Ten images, cached repeat | 79,540 / 28,404 | 24.350 s | 512 reasoning tokens, length cap, DONE and health pass | 2.413 / 4.162 GiB |
| Ten images, cached, 2,048 output allowance | 79,540 / 28,404 | 22.933 s | 815 output tokens, visible response plus two parsed tool calls, DONE and health pass | 2.599 / 4.854 GiB |

The first diagnostic reader incorrectly iterated `tool_calls: null`, a valid
stream delta. Corrected to treat nullable optional arrays as empty and checked
the captured 151 events before resuming. This was a diagnostic failure, not a
server OOM or protocol error. The nine-image retry reused 75,904 tokens; the
ten-image repeat reused 79,488 tokens (52 new), proven by scheduler logs despite
the API's usage field reporting `cached_tokens: 0`.

No memory-watchdog stop occurred. Both containers remained running with
`OOMKilled=false`; final device health passed. During the ten-image cold request,
head swap used increased 4.097→4.193 GiB and worker stayed about 3.30 GiB; repeat
ended at 4.251/3.301 GiB. The prior nine-image pickle replay's minimum available
RAM was 0.19/0.36 GiB, versus 2.964/5.254 GiB here. Different initial swap/cache
conditions and sampling intervals prevent treating that delta as an exact isolated
allocation measurement. The improvement is nevertheless observed with the full
image bytes retained and unchanged weight/KV precision.

The 512-token cap stopped both initial ten-image runs before visible content. A
final cached pass with 2,048 output tokens completed normally in 113.832 seconds,
using 815 output tokens (461 reasoning), with visible text and two JSON-parsed
edit calls. Those calls were recorded only, not executed. Head swap ended at
4.300 GiB; worker stayed 3.301 GiB. Evidence:
`/tmp/opencode/glm53-msgpack-finish.log` and
`/tmp/opencode/glm53-msgpack-retest-20261002-211812/`.
This trial does not qualify 400K input, extended uptime or multi-session capacity.
Evidence: `/tmp/opencode/glm53-msgpack-retest-run{,-2}.log`, replay directories
`/tmp/opencode/glm53-msgpack-retest-20261002-{205150,210305}/`, and preserved
`/tmp/opencode/glm53-msgpack-after-{head,worker}.log`.

New process evidence: both live schedulers have a TorchInductor
`compile_worker/__main__.py --kind=fork --workers=20` subprocess. This confirms an
independent compiler pool configured for twenty workers, matching the previous
unattributed pool-size observation. No direct children of those pool processes
were present in this particular snapshot; it does not retrospectively identify
every PID in the previous OOM table. Process snapshots:
`/tmp/opencode/glm53-msgpack-processes-{head,worker}.log`.

The post-restart OpenCode API still reports service 2.0.21/PID 58886, while the CLI
is 2.0.22. The replay uses a direct long-timeout client; no OpenCode service restart
was performed by this trial.

### Deep image-memory research — after the second OOM

See [the source audit, measured reproduction and validation plan](glm-5.3-flash-multimodal-memory.md).
The pinned image's CPU-only diagnostic reproduced **10x serialization/storage
amplification** for ten image views of a packed tensor. Its bundled MessagePack
codec reduced both modeled IPC hops to approximately logical tensor size with
exactly preserved values. `SGLANG_USE_PICKLE_IPC=0` is the leading configuration-only
test candidate; full request-schema and serving compatibility remain unqualified.
The real ten-image payload is calculated at approximately 0.498 GiB raw patches,
potentially 4.977 GiB per amplified serialized payload before additional copies.
This was not a capture of the failing request's allocations.

Important corrections: prefix hits do not skip full-history preprocessing and
transport, but normal ViT execution is chunk-aware. Missing prefill logs do not
establish a failure before vision execution. The GLM loader does not use the CPU
process pool bounded by `SGLANG_CPU_WORKERS`; compiler children remain unattributed.
The currently queried OpenCode service reports 2.0.21, while timeout propagation
and enforcement were added in 2.0.22. A live config value alone did not verify the
earlier timeout repair. No serving restart/change was made during this research.

### Second confirmed OOM — 2026-10-02 23:22–23:28 UTC

The single-active-request/400K candidate failed during real WoW session use.
Both retained containers report `OOMKilled=true`, `Status=exited`. The worker
kernel killed `sglang::schedul` PID 537278 at 23:22:20 UTC with all 16 GiB of swap
exhausted; the head killed scheduler PID 653428 at 23:28:28. The head proxy then
logged an upstream connection closing before response headers at 23:28:56.
Both hosts had about 117 GiB available after the containers exited.

The session's last successful request had 83,026 input / 44 output tokens.
Scheduler logs at 23:15:46 show 82,688 cached tokens, only 338 new tokens, KV
usage 0.20 and Mamba usage 0.40. That response invoked `read` on another screenshot:
the backed-up tool result contains a 2000x1250 PNG of 1,581,920 bytes, bringing the
historical image count to ten. No subsequent prefill batch is logged before the
crashes. The UI's 83,070 total therefore describes the last successful step, not
a measured token count for the failing image-containing request.

Startup evidence: weight loading reported 96.79 GB memory usage on the head;
the entire 420K-token KV pool was preallocated at 4.41 GB, plus 0.76 GB recurrent
state. The free slots inside that pool do not constitute free system RAM for
vision buffers. Startup headroom after initialization was only about 7.6 GiB.
Earlier controlled image replay already demonstrated near-zero available RAM.
This establishes inadequate memory headroom for the real multimodal workload,
but does not yet distinguish transient image allocations, retained buffers,
allocator behavior, or a leak. The killed process's CPU RSS does not account for
the entire unified GPU/CPU memory footprint. Both container environments still
contain `SGLANG_CPU_WORKERS=2`; additional Python processes in the OOM task table
have not been attributed to a particular pool.

Evidence is preserved under `/tmp/opencode/glm53-second-oom/`: full head/worker
container logs, inspect state, kernel journals and a private consistent session
database backup. No restart or serving configuration change was performed during
this investigation. The 400K limit remains a configured token ceiling, **not a
qualified usable multimodal capacity**. The earlier timeout adjustment does not
resolve this failure.

### Single-active-request, bounded-context trial — 2026-10-02 evening

Operator approved restarting Chachi with one active request and an explicit
deployment context limit, then syncing that live limit into OpenCode. Candidate:
`NIM_KV_CACHE_PERCENT=0.93`, `SGLANG_CPU_WORKERS=2`, and
`NIM_PASSTHROUGH_ARGS="--context-length 400000 --max-running-requests 1
--max-mamba-cache-size 10 --cuda-graph-max-bs-decode 1 --max-total-tokens 420000"`.
Keep BF16 KV, prefix caching, the bundled checkpoint, and the pinned image.
The 420K pool ceiling is separate from the 400K total per-request context limit;
actual allocation must be checked before advertising that limit. Ten Mamba slots
are a candidate for one active request (five slots required) plus retained states.
No 400K/500K workload stability claim is made by this configuration alone.

First launch failed before weight loading: setting `NIM_MAX_MODEL_LEN=400000`
causes legacy NIM profile compatibility filtering to reject all profiles, including
GB10. `nimlib/llm/profile_utils.py:1278–1286` gates that check on the environment
variable. Use the supported passthrough `--context-length` instead; the NIM
SystemConfig/SGLang translator maps it to the same backend context parameter.
Evidence: `/tmp/opencode/glm53-single-session-launch.log` (failure),
`/tmp/opencode/glm53-single-session-launch-2.log` (successful retry).

Restart completed and both ranks passed readiness. Effective scheduler settings:
`max_running_requests=1`, `context_len=400000`, `max_total_num_tokens=420000`.
Each node allocated 4.41 GiB BF16 KV plus 0.76 GiB recurrent states (10 slots).
Decode graph capture used 0.06 GiB; available memory after capture was 7.63 GiB
head / 7.78 GiB worker. Evidence: `/tmp/opencode/glm53-single-session-{head,worker}.log`.

Bounded verification passed: streaming/reasoning, tool call, six accumulated
1280x720 image turns, and final generation health. HTTP guards rejected an
800,032-token input and a 400,001-token output request with HTTP 400; the input
error explicitly named the 400,000-token context limit. This verifies rejection,
not successful inference at 400K. At the end, host MemAvailable was 3.8 GiB head /
6.3 GiB worker, swap used 3.8/2.8 GiB, with recent memory-pressure stalls but no
new OOM/restarts. Full 400K-context and sustained multimodal stability remain
unqualified; do not describe this as an OOM-proof deployment.
Evidence: `/tmp/opencode/glm53-single-session-check-run.log` and
`/tmp/opencode/glm53-single-session-check-20261002-125501-*`.

Sync contract: `omodel-wire sync` reads `/v1/models.max_model_len` into the
OpenCode model's `limit.context` (`oc_build_providers`). The generic TOML's
`context.native=1048576` documents the architecture, not the deployed limit.
`omodel-manager sync` resets the sandbox from DEFAULT_CONFIG and must not be
used to update OpenCode or to preserve this unpromoted trial.
Executed `omodel-wire.py sync` after reviewing its dry run. The existing GLM
provider ID was preserved; the generated model limits are context 400000 / output
65536 (Build/Plan TOML output preset remains 32768). Queried OpenCode's live
`/api/model` with `location[directory]=/home/otto/projects/wow` and confirmed it
has loaded `limit.context=400000`, not 1048576. OpenCode's documented automatic
compaction uses that deployed budget with output/buffer headroom.

Separate retry finding: OpenCode V2 documents a default five-minute header timeout
and five-minute stream-chunk timeout, followed by up to three retries. No timeout
settings were changed in this trial. Slow cold long-context prefill can therefore
still exceed client timeouts even when it fits the cache.

The prior 0.90 recovery passed the bounded image/history test through 84,522 input
tokens, but real OpenCode traffic still failed. At 18:21:18 UTC nginx logged an
OpenCode HTTP 499 with zero response bytes while the scheduler was processing
an approximately 83K-token prompt with 49,152 cached tokens. At 18:21:53 its
active allocation was released before prefill completed; at 18:23:29 generation
health failed waiting for the detokenizer. No new OOM kill/restart was observed;
head swap had grown to 8.6 GiB. The exact disconnect/stall cause is unresolved.
Prior head log: `/tmp/opencode/glm53-pre-single-session-head.log`.

### Real WoW session replay — 2026-10-02 evening

Read the session from a consistent, permission-restricted SQLite backup without
editing the live database. The latest stored error explicitly says
`provider.transport: Timed out waiting for response headers`. Nginx logs four
OpenCode 2.0.22 HTTP 499 responses at 19:56:11, 20:01:13, 20:06:17, and 20:11:24
UTC after the 19:51:11 resume. The scheduler was still doing prefill each time;
there was no new OOM kill or container restart. This establishes a client header
timeout for this failure, rather than inferring it solely from an HTTP 499.

Stored history includes nine images: six 2000x1250 PNGs and three 1004x1458 JPEGs,
19,391,496 total pixels (3.51x the earlier six 1280x720 synthetic-image workload).
Reconstructed successful user/assistant messages, reasoning, tool calls/results,
and image bytes; failed empty assistant messages were excluded. The replay uses
simplified system/tool declarations, so it is not a byte-identical original request
(75,940 measured input tokens versus about 83K for the actual OpenCode request).
Both comparison arms use different initial prefixes to avoid cross-arm KV reuse;
returned tool calls are recorded only, never executed.

| Replay | Actual input tokens | First token | Completion | Sampled minimum MemAvailable head/worker |
| --- | ---: | ---: | --- | --- |
| Full history, nine images | 75,940 (25,164 image tokens) | 749.657 s | 152 output tokens, clean tool call | 0.19 / 0.36 GiB |
| Same text, images omitted | 50,829 | 260.096 s | 82 output tokens, clean tool call | 6.68 / 6.66 GiB |
| Same text, newest image only | 54,062 (3,240 image tokens) | 307.159 s | 239 output tokens, clean tool call | 6.16 / 6.65 GiB |

Both returned HTTP 200, streamed through `[DONE]`, and passed generation health
afterward. Full-image replay increased swap usage 5.76→7.26 GiB on the head and
4.37→5.20 GiB on the worker. Text-only swap stayed approximately flat. The
20-second memory sampler caught the severe image-run troughs at 21:04:55–21:07:06,
before the first logged 4096-token prefill chunk at 21:07:37. This is evidence of
multimodal-path memory pressure, not an exact allocator-level attribution.
Dropping images also removes 25K tokens, so do not assign the entire latency
difference to preprocessing alone.

Evidence: `/tmp/opencode/glm53-session-debug-20261002-150056.db` (private backup),
`/tmp/opencode/glm53_wow_replay.py`, `/tmp/opencode/glm53-wow-replay-run-2.log`,
`/tmp/opencode/glm53-wow-replay-20261002-150347-*`, and
`/tmp/opencode/glm53-wow-replay-{head,worker}.log`.
The first reconstruction attempt got HTTP 400 because the test serializer did
not stringify a historical tool error object; fixed before inference comparison.
The newest-image-only comparison also completed through `[DONE]` and passed
generation health, with zero cached input tokens. Headers and first token arrived
at 307.159 s: still seven seconds beyond OpenCode's default timeout. Head swap
stayed at 7.25 GiB; worker swap decreased from 5.20 to 4.42 GiB. Sampled available
memory stayed above 6.16/6.65 GiB, supporting removal of historical images as a
memory mitigation for this workload, but not proof of sustained stability.
Evidence: `/tmp/opencode/glm53-wow-replay-latest-image.log` and
`/tmp/opencode/glm53-wow-replay-20261002-152157-*`.
No session history, live provider timeout, or model configuration has been changed
by these replay tests. A provider-wide header timeout increase (for example,
20 minutes for this measured workload) would accommodate the observed response
delay, but cannot itself resolve multimodal memory pressure or qualify 400K input.

Operator subsequently requested the timeout adjustment. Set
`provider.dgx-otto-dgx-1-8000.options.headerTimeout=1200000` (20 minutes) in
`~/.config/opencode/opencode.json`, retaining the existing legacy-format provider.
OpenCode V2 normalizes this to provider `settings.headerTimeout`; verified the
live `/api/provider/dgx-otto-dgx-1-8000` for the WoW project reports `1200000`.
The normalized configuration differs only in that field; model/endpoint settings
are preserved. A timestamped backup precedes the edit. No service restart was
performed. Later version/source investigation showed that this API check did not
prove timeout enforcement; see the deep-research correction above. The
streamed-chunk timeout setting was not changed.

### OOM recovery — 2026-10-02 afternoon

Investigation completed: Docker marks the head `OOMKilled=true`. Head kernel
journal records a global OOM and kills `sglang::schedul` (host PID 293121) at
16:04:35. Container exit code 0 is misleading: the wrapper exits cleanly after
the scheduler receives SIGKILL. Worker reports peer loss; its OOMKilled is false.
Last logged request was approximately 79K tokens, only 20% KV-pool occupancy.
The scheduler had approximately 12.4 GiB swapped plus 5.3 GiB RSS; about twenty
Python pool processes also had roughly 380 MiB swapped apiece. CPU RSS and
GPU shared memory must not be added naively on GB10. The exact allocator or
retention mechanism behind growth is not yet established.

Recovery validation: **bounded test passed; real-session failure remains**.
Restored the missing candidate profile into
the sandbox, keep the stock profile in `model_manager.json.bak`, and try vendor
`NIM_KV_CACHE_PERCENT=0.90` (was 0.94) plus `SGLANG_CPU_WORKERS=2` (was CPU count).
The pinned NIM registry/adapter maps the first to SGLang `mem_fraction_static`;
`base_processor.py:354–356` implements the second. This reduces cache capacity
to leave more shared-memory headroom and bounds one processor pool. Later source
inspection found that GLM's image loader uses an I/O thread pool instead; the
setting does not establish a bound on GLM/compiler child-process overhead.
These are mitigations pending a repeated-image/long-context test, not a proven
memory-leak repair. The 64 MiB proxy limit remains applied. Evidence:
`/tmp/opencode/glm53-reset-{head,worker}.log` and
`/tmp/opencode/glm53-reset-diagnostics.log`.

Recovery launch passed chat, reasoning, tools, vision, streaming, and two-rank
readiness. Both ranks log effective `mem_fraction_static=0.9`. New allocation:
158,735 shared KV tokens (1.67 GiB), 19 Mamba state slots (~1.38 GiB), automatically
capped to 3 active requests because each request needs 5 Mamba slots. Head reports
8.90 GiB available after graph capture. The crashed loopback launch actually had
400,561 shared tokens and 4.20 GiB free after capture (the earlier 419,760 figure
was a previous baseline launch). The API's advertised 1M context remains a model
limit, not the deployment's physical cache capacity. Evidence:
`/tmp/opencode/glm53-oom-recovery-{launch,head,worker}.log`.
The bounded repeated-image and ~80K-token cached-followup test completed via
`/tmp/opencode/glm53_oom_retest.py`; see `/tmp/opencode/glm53-oom-retest-run.log`.
All requests passed, but later follow-ups slowed sharply while OpenCode requests
also occupied the scheduler. This does not establish long-run memory stability.

**Original baseline: NVIDIA's published two-Spark NIM recipe, bundled checkpoint,
and stock inference tuning.** The operator explicitly requested this baseline
after questioning the earlier HF substitution and speculative tuning changes.
The earlier candidate and rationale below are historical, not the current plan.

Sandbox profile: `glm-5.3-flash-nim`. Baseline tests used the digest-pinned official image,
with no source overlays, HF substitution, context cap, CUDA-graph disable,
concurrency override, memory-fraction override, or custom-code-trust override.
The image supplies its own 0.94 memory fraction, four active requests, CUDA
graphs, and model context handling. NIM's manifest pins/validates bundled weights.

Observed stock-recipe issue: Tailscale binds its own IP on port 8000, conflicting
with NIM's wildcard bind (loopback:8000 itself was free).
API ports moved to 18000 (head) / 18002 (worker), backend to 18001 via documented
NIM settings. Evidence: `/tmp/opencode/glm53-nim-stock-launch-1.log`.
Automatic GB10 profile selection succeeded with the bundled model, without
`NIM_MODEL_PROFILE`; public NGC downloads began without a supplied API key.
Evidence: `/tmp/opencode/glm53-nim-stock-head-2.log`. The earlier auto-selection
failure therefore does not establish a stock-NIM failure.

Management-only differences: retained containers/otools labels, isolated
persistent download cache, and a two-hour cold-start allowance. Optional video
FFmpeg is not installed; video is not advertised as qualified. A foreground
orchestration timeout interrupted attempt 2 locally; its head was then stopped
through the manager and attempt 3 started in the background, reusing its cache.
Current launch evidence: `/tmp/opencode/glm53-nim-stock-launch-3.log`; serial
qualification starts only after launch warmups pass.

Cold-start staging observation: the head's bundled download finished at 04:52:59,
after roughly 32 minutes. It then reported a ten-minute worker-join window, while
the worker still had most of its 120 shards to download. The cold rendezvous was
interrupted and caches retained. Both hosts are now prepared with the image's
documented `download-to-cache --profile <GB10 profile> --use-cache --verify-checksums`
utility before starting the pair again. This changes staging, not inference
tuning. Evidence: `/tmp/opencode/glm53-nim-cache-prepare.log`; next launch:
`/tmp/opencode/glm53-nim-stock-launch-4.log`.

### Stock baseline live — 2026-10-02

Both caches passed NVIDIA's checksum-verifying downloader. Stock launch attempt 4
passed chat, separated reasoning, tool parsing, blue-image recognition, streaming,
and two-rank readiness. The served model is `zai-org/GLM-5.3-Flash`; `/v1/models`
advertises 1,048,576 context. Actual startup allocated 419,760 KV tokens across
requests, so the advertised limit is not a demonstrated usable million-token input.
Decode CUDA graphs for batches 1/2/4 are active; the vendor runtime automatically
disables prefill graphs for KDA. Worker logs report 97.91 GiB weight usage, 4.40
GiB KV cache, 55 Mamba cache slots, and 4.30 GiB available after graph capture.
Initial autotuning/compilation was lengthy. The worker also logged
`w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected.`
This is recorded for quality evaluation; no corrective patch has been applied.

Direct tailnet access to port 18000 timed out although head-local checks passed.
The manager's `tunnel Chachi --local-port 18000` now supplies a working local
`http://127.0.0.1:18000/v1` endpoint using the registered SSH key and runtime port
labels. Extended serial qualification is running through this path; the model
was not restarted to establish the tunnel. Evidence:
`/tmp/opencode/glm53-nim-stock-tests-2.log`, `glm53-nim-ready-head.log`, and
`glm53-nim-ready-worker.log` in the same evidence directory.

### Quality checks: effort-dependent malformed output

The first small functional suite passed 16/20 checks with low effort. Two
JSON-escaping failures used ambiguous punctuation; another failure incorrectly
required a nonempty low-effort reasoning trace. The remaining failure contained
the correct interval answer followed by reasoning-like text, another `</think>`,
and the answer again in the final `content` field.

Focused repeats reproduced malformed low-effort output; clarified JSON escaping
also failed once at low effort. Default/max and high interval repeats passed.
The initial runner used nested `chat_template_kwargs.reasoning_effort`, whereas
NVIDIA documents top-level `reasoning_effort`. Inspection of this image's
`serving_chat.py` lines 1063–1074 confirms normalization of nested to top-level.
The follow-up used only model/messages and optional top-level effort, leaving
sampling and output limits at vendor defaults: **6/6 default-effort checks passed;
low effort failed 2/6 checks**, both interval cases with stray reasoning delimiters.
No source patch or inference setting was changed. Evidence:
`/tmp/opencode/glm53-nvidia-api-retest.jsonl` and matching `.log`.
This rules out the nested field as a necessary cause; it does not isolate model,
parser, cache, or numeric behavior. Low effort is not quality-qualified.

The logged scale warning has a concrete numerical meaning in the pinned image:
`sglang/srt/layers/quantization/modelopt_quant.py:2441–2467` says Marlin supports
one shared gate/up scale, warns when the two checkpoint scales differ, and uses
`w13_weight_scale_2[:, 0]` (the gate scale) for the fused weight. Other backend
paths retain the two scales. This is a priority for controlled follow-up; it is
not yet established as the cause of the malformed responses. No scale repair,
checkpoint conversion, or backend swap has been applied during baseline testing.

Fix research (2026-10-02): vLLM PR
https://github.com/vllm-project/vllm/pull/55073 is still open, head
`c5233da3bf0951c2216cb69215e32bb962ffd7b8`. It chooses each expert's maximum
gate/up global scale and compensates each half's block scales by old/new global
scale, retaining the FP4 payload. Its E4M3 conversion introduces rounding and
requires accuracy evaluation. A commenter reports improved held-out NLL on a
different Qwen-derived checkpoint on two Sparks; this is not GLM/NIM validation.
The patch targets vLLM and cannot be applied directly to this SGLang image.
This image's `marlin_utils_fp4.py:503–520` converts block scales to activation
dtype before Marlin permutation, which is a candidate place to preserve separate
gate/up factors without another FP8 rounding step. Any adaptation must retain
the two original global scales until reconciliation, verify gate/up layout and
TP slicing, check finite/range behavior, and compare dequantized reference values
and end-to-end quality. This remains a proposed correction, not a deployed fix.

Broader upstream check (2026-10-02): SGLang PR
https://github.com/sgl-project/sglang/pull/27588 **merged 2026-06-15**, commit
`441b75ee69730687230214f5d92bf92c0c4bf7b7`. It fixes separate gate/up alphas for
the FlashInfer TRT-LLM backend and explicitly leaves Marlin's single gate-scale
collapse unchanged. The pinned NIM already contains that code. SGLang `main`
still contains the same Marlin warning/collapse as checked today. Related PRs
31330 (FP32 global scale) and 29931 (routing-factor double application) concern
different defects, not a demonstrated correction for separate gate/up scales.

Queried NVIDIA's public registry directly: the only runnable GLM tags returned
were `latest` and `2.1.2-variant` (other tags are SBOM/VEX/signature artifacts).
Both resolve to index
`sha256:75dc19a2c94d023a3aa13603e547c77aa54499781d2919413fff2290646ac827`, whose
ARM64 manifest is our exact pinned
`sha256:0bd2a1f4ffacf4ee61e1f8c1b48071a6713b51bad954ab0ca627c24be84f43f8`.
Thus repulling `latest` does not provide a newer ARM64 build at this check.
The GLM-specific release notes still describe the initial 2.1.2-variant release.
An existing fix for another backend is not automatically a viable GB10 backend
switch; compatibility and end-to-end accuracy must be established first.

Broader functional/sampling/context qualification now uses vendor-default effort
and sampling. The speed sweep explicitly requests max effort and counts reasoning
tokens. Sampling HTTP acceptance must not be described as proof of application:
stock NIM has request-parameter logging disabled.

The default functional suite completed **20/20 under its original validator**.
That validator stripped Markdown fences: `json_escape/0` and `vision/spatial`
returned fenced JSON despite JSON-only instructions. Thus two responses still
fail a strict raw-JSON consumer. Future runs now record semantic correctness and
strict JSON separately; historical results are retained unchanged.
Streaming, arithmetic, code tracing, Unicode, tool invocation/continuation, and
both image tasks worked in this small suite. Evidence:
`/tmp/opencode/glm53-default-functional.jsonl`.

Sampling probes returned HTTP success for all 16 fields; the stop probe exhausted
its 256-token budget before a normal completion. Its prompt never requested the
stop string, so this is not evidence that stopping failed. `logprobs` returned
per-token values, and `top_logprobs=3` returned three candidates per token.
`thinking_token_budget=101` produced 112 reasoning tokens; neither that field nor
`repetition_detection` is declared in this image's SGLang ChatCompletionRequest.
Their acceptance must not be advertised as support. Remaining controls are
accepted but need merged-parameter logging or targeted behavioral tests.

Recommended next qualification additions:

1. Strict JSON/schema checks, including streamed tool-argument assembly and
   multi-turn tool errors/recovery, with realistic coding tasks.
2. A larger effort matrix (default/high/low), streaming and non-streaming,
   unique cache salts versus reused prompts; save raw token IDs/delimiters to
   separate generation defects from reasoning-parser defects.
3. Audit gate/up scale ratios in the exact bundled checkpoint and ask NVIDIA
   about the stock Marlin warning before any controlled runtime/weight A/B.
   Similar behavior is reported in vLLM issue
   https://github.com/vllm-project/vllm/issues/54974, but its checkpoint and runtime
   differ; its measured ratios do not describe this deployment.
4. Multi-position retrieval at increasing **measured** token lengths, followed
   by quality checks under concurrent load; advertised context is not capacity.
5. Dedicated stop-string, output-cap, seed, and min-token behavioral probes;
   measure time to final answer as well as first reasoning token and decode speed.

### Completed baseline context and performance sweep

Retrieval of a random code from the middle of **47,465 input tokens** passed,
including strict JSON, in 330.235 seconds. This is one retrieval case, not
qualification of the advertised 1,048,576-token window. Startup KV capacity was
419,760 tokens shared across requests.

The unique-prompt benchmark used default sampling, explicit max effort, and a
300-token output cap. Timings include reasoning tokens and were measured through
the manager's SSH tunnel. All seven streams supplied finish reasons and `[DONE]`.

| Submitted concurrency | Actual input tokens/request | First-token wait (s) | Effective output tok/s/request | Wall time (s) |
| --- | --- | --- | --- | --- |
| 1 | 50,862 | 256.6 | 9.8 | 287 |
| 2 | 50,863–51,144 | 259.3–503.5 | 0.8–7.7 | 542 |
| 4 | 50,863–51,151 | 267.1–1,068.9 | 0.3–7.8 | 1,107 |

Five of seven outputs hit the cap; these are speed measurements, not evidence of
complete, correct summaries. The effective output rate includes pauses between
the first and last generated token. Logs show long prefill phases while existing
requests wait for decoding. Once prefill finishes, logged aggregate decode rates
are approximately 14 tok/s for two requests and 18 tok/s for three. The low
effective rates therefore must not be described as steady-state kernel speed.
Four requests were submitted, but the logs show only three actively decoding
together before a request finishes and the fourth is admitted.

The initial retrieval triggered additional Triton compilation; the subsequent
~51K single-request benchmark still needed 256.6 seconds to first output. Future
performance investigation should distinguish compilation, uncached prefill,
prefix-cache reuse, and prefill/decode scheduling. A context-size ladder and
short requests arriving during a long prefill are higher-priority next tests than
blindly increasing concurrency. Any tuning should be a one-variable A/B against
this baseline after confirming NVIDIA support for the exact model/backend.

Evidence: `/tmp/opencode/glm53-default-context.jsonl`,
`glm53-default-bench-n{1,2,4}.log`, and `glm53-post-benchmark-{head,worker}.log`.
Post-sweep health was READY; captured head logs contain no ERROR, traceback, OOM,
or preemption/retraction messages. The sandbox records rounded N=1 `tok_s=10`.
The profile remains a candidate: low-effort quality and interactive long-context
latency are unresolved. Nothing has been promoted into `DEFAULT_CONFIG`.

### Requested loopback endpoint

Operator requested the final API at **127.0.0.1:8000** for the existing Tailscale
Serve mapping. `ports Chachi 8000` confirmed only tailnet listeners on both hosts,
forwarding to loopback:8000. Previous loopback-only models could coexist; NIM's
wildcard listener conflicts. After baseline testing, both hosts passed preparation
of the exact-hash nginx loopback template. The network-only adaptation mounts that
template and sets the supported backend host to loopback. API/backend ports are
8000/8001 on the head, 8002/8001 on the worker. No model-code overlay or inference
tuning is involved. Restart completed and passed chat, reasoning, tools, vision,
streaming, listener checks, and two-rank readiness. Device health is READY.
Direct access through the existing Tailscale mapping returned the expected model
and a correct arithmetic response with separated reasoning and normal stop:
`http://otto-dgx-1.tail1faf4a.ts.net:8000/v1`, model `zai-org/GLM-5.3-Flash`.
Evidence: `/tmp/opencode/glm53-loopback-{prepare,plan,launch}.log` and
`/tmp/opencode/glm53-tailnet-final-smoke.json`. The numerical scale issue remains
unmodified. Repository checks passed: 259 unit tests, compilation, and diff check.

## Reported outage investigation — 2026-10-02 afternoon

Request-body limit remediation: **completed**. Operator requested a practical
limit for large chat histories and images; selected 64 MiB (67,108,864 bytes).
Updated the hash-checked persistent nginx template on both hosts, then applied
`reload-proxy Chachi`: config generation, nginx validation, and graceful reload.
The new template SHA-256 is
`183a7bcb6dee4ade35df55383249bf586ddda2abee5286347a220934251c3b40`.
The reload command discovers NIM rank identity from runtime labels and does not
depend on a caller-local profile. Generated configs are restored if validation
or the reload command fails. Weights stayed loaded; both containers still showed
eight hours uptime and readiness passed afterward.

Live tailnet verification: 2 MiB JSON body returned 323; 8 MiB JSON body containing
a small blue image returned Blue. Whitespace padding exercised HTTP byte limits
without a large-token inference workload. A declared body of 64 MiB + 1 byte was
rejected with HTTP 413 before upload. Evidence:
`/tmp/opencode/glm53-body-limit-verification.json` and
`/tmp/opencode/glm53-body-limit-prepare.log`. All 262 repository unit tests passed.

Both rank containers showed eight hours uptime. Tailnet readiness and `/v1/models`
returned HTTP 200; a real arithmetic request returned 323 with separated reasoning
and normal stop in 3.04 seconds. No current inference outage was reproduced.

The head log at 14:36:01 records `client intended to send too large body: 1430728
bytes` on `/v1/chat/completions`. The pinned nginx template has no
`client_max_body_size`, leaving nginx's default 1 MiB limit. A valid tiny-prompt
JSON request padded with whitespace to exactly 1,430,728 bytes reproduced HTTP
413 `Request Entity Too Large` through Tailscale, without a large inference job.
This is HTTP request rejection before inference, not evidence of a model crash or
token-context overflow. The remedy is a larger nginx body limit and a validated
graceful proxy reload, applied after the operator requested it as recorded above.
No model restart was needed. Initial evidence: `/tmp/opencode/glm53-crash-{head,worker}.log` and
`/tmp/opencode/glm53-crash-chat-check.json`.

Separately, the current local sandbox no longer contains the GLM profile; its
previous entry is intact in `model_manager.json.bak`. Default manager health/logs
therefore fail with `unknown cluster profile` despite live runtime discovery
seeing both ranks. Using `--config model_manager.json.bak` restored read-only
diagnostics and confirmed READY. The reset mechanism/user action was not proven;
no replacement of the current config was performed. This also exposes a remaining
cluster lifecycle dependency on caller-local profiles that should be corrected.

## Initial host preparation details (historical)

Target: registered `Chachi` cluster (`otto-dgx-1`, `otto-dgx-2`), two DGX Sparks.
Initial `omm ps` found an existing DeepSeek head container and an idle worker;
operator is updating the machines. Operator authorized testing once both machines
finish updating/reboot and the existing DeepSeek deployment is no longer running.
Subsequent inspection found the head temporarily unreachable, consistent with
maintenance; no host modifications have been made.

## Work tracking

The current harness exposes no todo tool; this record tracks the workflow instead.

| Step | State |
| --- | --- |
| Prepare: inspect checkout and registered devices | completed |
| Research trusted quantizations and exact-hardware runtimes | completed |
| Audit required external patches and pin all artifacts | completed |
| Draft sandbox cluster profile and generic config | completed |
| Deploy on idle Chachi; verify both ranks | completed |
| Baseline quality/tools/vision/sampling checks; record qualification gaps | completed |
| Benchmark real context and concurrency | completed |
| Restore loopback access, document results, and run project checks | completed |

## Planned acceptance evidence

- Both ranks healthy over the registered RoCE fabric; exact model/image identities.
- Text and streaming output; repeated deterministic probes for corruption.
- Reasoning effort low/high/max; correct reasoning/content separation. The pinned
  GLM template always thinks and does not implement `enable_thinking=false`.
- Tool arguments parse, tool-result continuation works, parallel requests complete.
- Solid-color image identification and a more discriminating image task before
  advertising vision.
- Independent sampling-parameter requests checked against runtime logs.
- Long-context retrieval and coding/task probes with objectively checkable answers.
- Warm single-user ~50K context TTFT/decode rate, then concurrency 2 and 4.
- Final clean restart/health and usable endpoint recorded with remaining limitations.

Published benchmark scores and third-party speed reports will be distinguished
from measurements on Chachi. A small local probe suite is not a replacement for
full benchmark evaluation of quantization quality.

## Quantization research

Publisher-owned candidates (revisions checked 2026-10-01):

| Publisher/model | Revision | Complete safetensors GiB | Role |
| --- | --- | ---: | --- |
| `nvidia/GLM-5.3-Flash-NVFP4` | `da920bb0b9f4a06727223a349e55468e38352348` | 190.40 | Quality-evidence-first candidate |
| `RedHatAI/GLM-5.3-Flash-NVFP4` | `18d55bfd5a2194887738da73753975c9d3842f46` | 184.26 | Smaller NVFP4, detailed TP2 operator reports |
| `Intel/GLM-5.3-Flash-W4A16-AutoRound` | `5eee1846f0321058ed73745f9aa16f2aaf0fc0a0` | 169.01 | Trusted INT4 capacity alternative |

The original Z.ai FP8 is 305.79 GiB and BF16 is 598.52 GiB; neither fits fully
resident on two Sparks. Four-bit file sizes exclude runtime, repacking, replicated
tensors, vision, KV/KDA state, activations, and OS. NVIDIA specifies 128 GB unified
memory and 273 GB/s bandwidth per Spark; two nodes do not form one unified allocator.

NVIDIA publishes paired BF16/NVFP4 results on GB200:

| Task | BF16 | NVFP4 |
| --- | ---: | ---: |
| GPQA Diamond | 92.17 | 92.11 |
| SciCode | 56.21 | 57.69 |
| MMMU Pro | 76.88 | 76.30 |
| AA-LCR | 71.00 | 71.06 |
| IFBench | 61.30 | 60.54 |
| Terminal Bench 2.1 | 82.58 | 83.15 |

Temperature 1.0, top-p 0.95, up to 327,680 output tokens (Terminal Bench uncapped).
These are publisher results, not Chachi measurements or proof of identical GB10
Marlin W4A16 quality. Small positive deltas do not establish improved quality.

Intel reports average 83.99 BF16 versus 83.86 INT4 across GSM8K/MMLU/PIQA/HellaSwag
(99.84% relative retention of that average). Red Hat reports GSM8K Platinum 97.74,
MATH-500 94.87, AIME 2025 86.67, GPQA Diamond 90.57, without a matched BF16 table.
Different evaluations/settings cannot establish NVIDIA-versus-Red-Hat-versus-Intel
quality rankings. No controlled official INT4/NVFP4 comparison was located.

### Precision details that affect the choice

- NVIDIA: ModelOpt NVFP4, group 16; routed experts and dense prefix MLPs quantized;
  shared experts/attention/vision remain higher precision. BF16 MTP tensors are
  present as layer 45. The pinned revision fixes their quantization exclusion.
- Red Hat: compressed-tensors NVFP4, group 16, routed experts only; FP8 MTP file.
- Intel: AutoRound 0.15.0, symmetric INT4 group 128, W4A16, GPTQ-style packing;
  attention/shared experts/dense prefix/vision remain higher precision.
- NVFP4 has approximately 4.5 bits/quantized weight including group scales; INT4
  with a BF16 scale per 128 weights is approximately 4.125, before other metadata.
  Module exclusions and MTP precision also materially change complete size.
- NVFP4 weights run through a W4A16 kernel remain FP4 weights; this does not turn
  the checkpoint into INT4. Native FP4 hardware alone does not guarantee an
  optimized/correct GLM sparse-MLA/KDA/MoE path on SM121.

### Model-native behavior

`Glm5NextForConditionalGeneration`, 320B total/18B active, 45 target layers
(34 KDA + 11 sparse MLA), 288 experts/8 selected, native context 1,048,576.
Artifacts contain text/image/video support; served modalities must be tested.
Templates always open `<think>`; `reasoning_effort` accepts low/high/max and
defaults to max. `clear_thinking` controls historical reasoning retention, not a
thinking-off switch. Sampling recommendation: temperature 1.0, top-p 0.95.

### Sources

- [NVIDIA pinned card and paired evaluation](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/da920bb0b9f4a06727223a349e55468e38352348/README.md)
- [NVIDIA pinned config](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/da920bb0b9f4a06727223a349e55468e38352348/config.json)
- [NVIDIA MTP metadata fix](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/discussions/11)
- [Red Hat pinned card](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4/blob/18d55bfd5a2194887738da73753975c9d3842f46/README.md)
- [Intel pinned card](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound/blob/5eee1846f0321058ed73745f9aa16f2aaf0fc0a0/README.md)
- [Z.ai pinned architecture](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)
- [NVIDIA Spark specifications](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA NVFP4 representation](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Red Hat general NVFP4 study](https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization)

General NVFP4 studies concern other models; they motivate testing but are not
GLM-5.3 quality measurements.

## Runtime selection

**First candidate: official NVIDIA NIM 2.1.2-variant (patched SGLang), NVIDIA
NVFP4 weights.** The NVIDIA guide updated 2026-09-30 explicitly supports two
DGX Sparks/GB10 with NVFP4. This is stronger exact-hardware evidence than a
generic Blackwell/GB200 recipe. No community image or patch has been executed.

**Provenance correction after inspecting the image:** the baked NGC artifact is
credited to **LibertAI**, not NVIDIA's HF quantization team. It will not be used.
The supported local-model override is required for our pinned NVIDIA HF checkpoint.
That replacement is not the vendor's exact tested model/runtime combination and
requires fresh qualification. The model source stays read-only and unchanged.

Pinned NIM ARM64 image:

```
nvcr.io/nim/zai-org/glm-5.3-flash@sha256:0bd2a1f4ffacf4ee61e1f8c1b48071a6713b51bad954ab0ca627c24be84f43f8
```

Registry labels identify SGLang commit `033446bb05f35c0943aed2750c443077ffc0b92c`,
patch profile `sglang-glm53f`, CUDA 13.0.3, FlashInfer 0.6.17, and sgl_kernel
0.4.6.post1. This is not stock SGLang and is not TensorRT-LLM. Artifact pinning
identifies bytes; it does not by itself prove correctness/security.

| Runtime | Evidence and decision |
| --- | --- |
| NVIDIA NIM | Official exact two-GB10 NVFP4 support; evaluate first |
| vLLM | Native model/ARM64 support, but open exact NVIDIA-weight TP2 indexer divergence and recent correctness fixes; fallback requires additional review |
| Stock SGLang | Native model support; generic recipes target datacenter GPUs; SM121 NoPE and ModelOpt ignore-list issues remain relevant |
| llama.cpp | Text/vision merged 2026-09-30; no established trusted-publisher GGUF plus tested dual-Spark recipe found |
| TensorRT-LLM | Model support merged 2026-09-28; published recipe is FP8 on four B200s, not NVFP4 on GB10 |
| EXL3/SparkInfer/DFlash hybrids | Fast community claims use different weights/drafters/custom kernels; not equivalent to unchanged NVIDIA/Red Hat/Intel weights |

No matched runtime speed ranking can be inferred from dissimilar published
prompts, contexts, quants, speculation, and concurrency. NIM throughput remains
unmeasured on Chachi.

### Correctness issues driving the decision

- [vLLM #58636](https://github.com/vllm-project/vllm/issues/58636): exact NVIDIA
  NVFP4/GB10 TP2 indexer normalization can differ across ranks after independent
  autotuning, leading to different sparse candidate selection. A determinism flag
  alone is reportedly insufficient due to a PyTorch state-restoration bug.
- [vLLM #57087](https://github.com/vllm-project/vllm/issues/57087): concurrent
  emoji/CJK corruption on GB10 TP2; checkpoint applicability must be tested.
- vLLM fixes merged after 0.30.0 include padded-stride indexer writes
  [#57477](https://github.com/vllm-project/vllm/pull/57477), speculative tail-ring
  overwrite [#58454](https://github.com/vllm-project/vllm/pull/58454), pooled-index
  selection [#58704](https://github.com/vllm-project/vllm/pull/58704), FP32 KDA
  state [#58846](https://github.com/vllm-project/vllm/pull/58846), and FP8 dtype/
  workspace [#55222](https://github.com/vllm-project/vllm/pull/55222).
- [SGLang #39302](https://github.com/sgl-project/sglang/issues/39302): SM120 NoPE
  support; [#36596](https://github.com/sgl-project/sglang/issues/36596): exact-GB10
  ModelOpt fused-name/prefix ignore matching.
- Successful image HTTP responses alone are inadequate: incompatible processors
  can silently answer only the text portion. Controlled image probes are required.

### Primary runtime sources

- [Official two-Spark NIM guide](https://docs.nvidia.com/nim/vision-language-models/latest/deploy-on-dgx-spark.html)
- [NIM exact-hardware support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html)
- [NIM GLM request/effort guide](https://docs.nvidia.com/nim/vision-language-models/latest/get-started/advanced/get-started-glm-5-3-flash.html)
- [NIM environment controls](https://docs.nvidia.com/nim/vision-language-models/latest/environment-variables.html)
- [Official vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)
- [Official SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash.md)
- [llama.cpp model support](https://github.com/ggml-org/llama.cpp/pull/27773)
- [TensorRT-LLM model support](https://github.com/NVIDIA/TensorRT-LLM/pull/19136)

### Host readiness and staging

Following operator authorization, both nodes became idle after maintenance.
`cluster preflight Chachi` passed: ARM64 GB10, driver 580.178.04 on both nodes,
Docker 29.6.2, both registered RoCE rails up/routed correctly, and >2 TB free
disk on each. Registered MTU remains 1500; no fabric configuration was changed.

The ignored sandbox contains `cluster_models.glm-5.3-flash-nvfp4`, initially
requesting 131072 served context and one sequence pending runtime verification.
Preparation uses the pinned official NIM image and the existing pinned NVIDIA
download-container mechanism for the exact HF revision. Heavy preparation began
only after `omm ps` showed both Chachi nodes idle.

### Inspected NIM controls and minimal adaptations

Static inspection of 847 MB of digest-verified relevant image layers confirmed:

- NIM runs as `nvs`, UID/GID 1000:1000, with native Transformers 5.16.1.
- Its SGLang translator really consumes `NIM_PASSTHROUGH_ARGS`.
- `NIM_KV_CACHE_PERCENT` controls **weights plus cache static memory fraction**;
  the candidate overrides the vendor's 0.94 with 0.85.
- Candidate concurrency is explicitly `--max-running-requests 1`, chunked prefill
  4096, CUDA graphs initially disabled, request parameter logging enabled.
- The vendor GB10 tuning selects Marlin, BF16 KV, TP2, and **sets index_topk=null**,
  disabling DSA classification. This is a different execution path from community
  sparse-MLA recipes and could materially affect long-context speed and outputs.
  Native 1M context must not be represented as a qualified served limit.
- Both nginx and the backend bind wildcard addresses by default. The candidate
  uses `NIM_BACKEND_HOST=127.0.0.1`, the API/health port kept identical, and a
  one-line nginx-template override binding the API to loopback.
- NVIDIA's quantization exclusions name `model.language_model.*` and `model.visual*`,
  while this SGLang implementation constructs `model.*` and `visual*`. A minimal
  class-level `WeightsMapper` mirrors the existing weight-loader renaming for the
  exclusion list. It changes runtime name matching, not weight values or quant
  calibration. Upstream's existing ModelOpt mapper hook consumes it.

The read-only overlays are generated from the pinned official image using
network-disabled CPU-only Python, with exact preimage and result SHA256 checks.
The manager verifies the results again before launch. No community CUDA kernel,
launcher, watchdog, external drafter, or model converter is imported.

Full security findings and the limits of the review:
[glm-5.3-flash-security.md](glm-5.3-flash-security.md).

### First startup failures — 2026-10-02

1. Automatic NIM profile selection failed on unsupported GB10 NVML memory fields
   (`NoneType` arithmetic), before launching SGLang. Explicitly selecting the baked
   GB10 profile with `NIM_MODEL_PROFILE=d3c5ed87ba2f6ed5424d46a191fde9d1c53f424d3d454372fae8873205a04b8b`
   bypassed that failure and both ranks launched. Evidence: `/tmp/opencode/glm53-launch-2.log`
   and `/tmp/opencode/glm53-launch-3.log`.
2. NIM's local workspace SDK enumerated zero files from the HF symlink snapshot;
   Transformers then rejected the empty workspace for missing `model_type`.
   A third hash-checked overlay makes `nim_serve` pass the existing absolute local
   model reference directly to SGLang. The original snapshot remains unchanged.
   Overlay details and exact hashes are in the security note.

All 250 manager tests passed before the third-overlay addition. Neither startup
failure is evidence of an inference-quality or memory-capacity result.

3. Attempt 4 loaded all 33 weight shards on both ranks using Marlin and NCCL
   NET/IB on both rails. Head load took 977 seconds, reported 96.27 GiB memory
   usage and 16.04 GiB available. Cache allocation then failed under the 0.85
   static-memory budget (`total_rest_memory=-1.42 GB`), before any inference.
   The next sandbox attempt raises that budget to 0.90 (vendor default 0.94).
   Evidence: `/tmp/opencode/glm53-launch-4.log`. The extended test runner stopped
   at its startup gate and sent no qualification requests.

4. Operator questioned deviating from NVIDIA's official 0.94 memory fraction.
   The 0.90 attempt was interrupted during startup, without a qualification result.
   The next attempt uses the official 0.94 baseline. The initial 0.85 came from
   the repository's generic UMA default rather than this model's vendor recipe;
   the NVIDIA HF versus bundled NGC checkpoint difference still requires live
   capacity and quality validation.

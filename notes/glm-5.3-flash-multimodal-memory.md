# GLM-5.3-Flash: repeated-image OOM investigation

Research following Chachi's 2026-10-02 OOM. Deployment history and retained
evidence: [Chachi qualification record](glm-5.3-flash-chachi.md).

**Subsequent approved serving test:** the 2026-10-03 MessagePack trial passed
startup checks and nine-/ten-image replay transport with materially better
memory floors and no OOM. See the qualification record for measured results,
the output-cap caveat and remaining capacity limits. The sections below preserve
the research evidence and recommendation as recorded before that deployment.

## Conclusion

**A CPU-only probe in the exact pinned NVIDIA image reproduced tenfold
multi-image tensor serialization amplification.** The image's existing MessagePack
codec eliminated this amplification in both modeled transport hops, preserving
tensor values, shapes and dtypes. This is a concrete mechanism and a promising
configuration-only remedy; it is not yet an end-to-end GLM fix qualification.

The recommended first serving experiment is `SGLANG_USE_PICKLE_IPC=0` on both
nodes, retaining the checkpoint, BF16 KV and image representations. This is a
bundled SGLang capability, not a separately documented NVIDIA guarantee. Full
request-schema compatibility and real-workload memory/latency remain to be tested.

No serving configuration, model container, checkpoint or OpenCode service was
restarted/changed during this research. A separate bounded CPU-only diagnostic
container ran from the already-cached image on Chachi's head and exited cleanly.

## 1. Exact runtime and upstream match

Image ARM64 digest:
`sha256:0bd2a1f4ffacf4ee61e1f8c1b48071a6713b51bad954ab0ca627c24be84f43f8`.
Extracted SGLang base commit:
`033446bb05f35c0943aed2750c443077ffc0b92c`.

[SGLang issue #33388](https://github.com/sgl-project/sglang/issues/33388)
reports per-image views retaining an entire packed parent tensor, with pickle
serializing that parent once per image. Its 518x reproduction uses hundreds of
images on H200s, not GLM/GB10. A follow-up reports approximately 31 slices of a
313 MiB parent becoming a 10.1 GiB received pickle payload, with allocations
attributed to `recv_pyobj`. Those are external mechanism reports, not our measured
payload sizes.

The pinned source contains the relevant complete chain:

1. Transformers' GLM processor concatenates all raw image patches.
2. `managers/mm_utils.py:935–943,1147–1151` splits with tensor slicing, retaining
   parent storage.
3. `multimodal/transport/__init__.py:10–20` selects default transport for multiple
   nodes; `mm_utils.py:1375–1380` consequently bypasses POSIX SHM wrapping.
4. `environ.py:343` defaults `SGLANG_USE_PICKLE_IPC=True`.
5. `managers/io_struct.py:2433–2462` uses `send_pyobj`/`recv_pyobj` when enabled.
6. `scheduler_components/request_receiver.py:88–102,209–215` receives locally,
   then broadcasts requests across TP ranks.
7. `utils/common.py:2438–2484` broadcasts using `pickle.dumps`. The receiver
   allocates a byte tensor, copies it into Python bytes, then unpickles tensors.

Selected environment inspection of both stopped containers confirmed the flag is
unset, so the default applies. The source audit compared eleven relevant files
byte-for-byte with the public base commit; transport, processor and serialization
files matched. Existing NIM patches do not alter those files.

Source root for the line references above:
[pinned SGLang SRT](https://github.com/sgl-project/sglang/tree/033446bb05f35c0943aed2750c443077ffc0b92c/python/sglang/srt).
Local extracted source: `/tmp/opencode/nim-glm53-inspect/rootfs/`.

## 2. Measured reproduction in the pinned image

Diagnostic artifacts:

- `/tmp/opencode/glm53-deep-research/probe_serialization.py`
- `/tmp/opencode/glm53-deep-research/serialization-probe.log`

The diagnostic used installed PyTorch `2.13.0+cu130`, installed msgspec, and the
installed SGLang tensor codec. CUDA was unavailable. It loaded no weights and
used no model-cache mounts; container RAM/swap limit was 2 GiB, with two CPUs,
no network and a read-only root. Docker execution used the manager's choke point.

Synthetic image features used the same packed-tensor/per-image-view structure,
at 256 KiB per image. Hop A modeled tokenizer-to-rank-0 serialization; hop B
performed the same `pickle.dumps([request])` operation used by TP broadcast,
after decoding hop A. This exercised serializer operations, not live ZeroMQ,
Gloo, full `TokenizedGenerateReqInput`, or an inference scheduler.

| Images | Mode | Logical feature MiB | Hop A MiB | Hop B MiB | Rank-0 feature storage MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | pickle | 0.25 | 0.2507 | 0.2507 | 0.25 |
| 2 | pickle | 0.50 | 1.0014 | 1.0014 | 1.00 |
| 10 | pickle | 2.50 | 25.0081 | 25.0081 | 25.00 |
| 10 | `.contiguous()` then pickle | 2.50 | 25.0081 | 25.0081 | 25.00 |
| 10 | compact `.clone()` then pickle | 2.50 | 2.5060 | 2.5060 | 2.50 |
| 10 | MessagePack then pickle | 2.50 | 2.5011 | 2.5063 | 2.50 |

All 12 cases (1/2/10 images × four modes) preserved exact tensor values, shape
and dtype through both hops. `.contiguous()` was verified to return the original
already-contiguous views; it does not shrink oversized backing storage.

MessagePack transmits logical tensor bytes and reconstructs each tensor from its
own right-sized bytearray (`utils/msgpack_utils.py:63–76,158–166`). Thus the
subsequent distributed hop still uses pickle, but sees compact storages and no
longer repeats the packed parent. Ordinary serialization/copy overhead remains.

The logged RSS is a process-wide cumulative high-water mark across modes. Do not
use it as an isolated per-mode memory comparison; the byte/storage measurements
are the controlled result.

### Scaling to the actual screenshots: calculated, not captured on the wire

Bundled processor configuration: patch size 14, temporal patch size 2, merge
size 2, FP32 patch vectors of width `3 × 2 × 14 × 14 = 1176`.

| Input | Aligned dimensions | Patches | Image tokens | Raw FP32 feature bytes |
| --- | --- | ---: | ---: | ---: |
| One 2000×1250 PNG | 2016×1260 | 12,960 | 3,240 | 60,963,840 |
| One 1004×1458 JPEG | 1008×1484 | 7,632 | 1,908 | 35,900,928 |
| Nine images: six PNG, three JPEG | — | 100,656 | 25,164 | 473,485,824 |
| Ten images: add one PNG | — | 113,616 | 28,404 | 534,449,664 |

The nine-image calculation matches the measured replay's 25,164 image tokens.
At the reproduced amplification ratios, the nine-image raw feature payload grows
from approximately **0.441 to 3.969 GiB**; ten images grow from **0.498 to
4.977 GiB**, excluding serialization metadata. The actual failing IPC payload
was not captured. Multiple received/serialized/reconstructed copies can coexist;
their sum is not a measured simultaneous peak.

This explains why adding one ordinary screenshot can cost much more than its
58 MiB raw patch tensor: it enlarges the parent tensor *and* adds another view
whose pickle contains that parent.

## 3. Prefix caching, vision execution and lifetime corrections

Full submitted image history is loaded/preprocessed/transported before scheduler
prefix matching. Therefore the 82,688 cached-token hit does not avoid that work.
GLM directly calls `process_and_combine_mm_data`; it does not use the generic
artifact-cache lookup path found in the Kimi implementation. Increasing
`--mm-preprocess-cache-size-mb` alone is not a demonstrated GLM remedy.

However, normal vision embedding execution is chunk-aware: per-image intervals
outside the current prefill chunk are skipped, and a local embedding cache is
consulted. Do not claim that every historical image is re-encoded by the ViT on
every cached continuation. Sources: `managers/mm_schedule.py:285–355,465–559`.

Other concrete contributors/fix points:

- CPU feature transport can still use CUDA torchvision preprocessing, then copy
  features to CPU. Both consume GB10's unified RAM.
- Preprocessing batches the submitted history and retains grouped/resized/
  normalized tensors plus concatenated patches. The raw patch inputs and final
  concatenation alone can coexist at roughly 1 GiB for ten screenshots.
- A request-scoped CUDA MemPool mitigation is already present. The merged
  [fast-processor allocator fix #36295](https://github.com/sgl-project/sglang/pull/36295)
  is not a missing upgrade in this NIM.
- GLM moves cache-miss features to CUDA as FP32, concatenates, then casts BF16
  (`models/glm5_next.py:1423–1435`), creating avoidable intermediate copies.
- Scheduler retains CPU raw features as chunked-prefill fallback; tokenizer can
  also retain its local request while waiting for the response. These are
  per-request lifetimes, not proof of an indefinite leak.
- `enable_mm_global_cache=False` leaves a separate local embedding cache active;
  its default is 100 MiB. Cached embedding views can retain larger batch storage
  than their logical-byte accounting records.
- `SGLANG_CPU_WORKERS=2` controls one processor pool, but the inspected GLM loader
  uses an I/O thread pool. It does not establish a bound on all Python children.
  GLM's independent `@torch.compile` vision SwiGLU path makes TorchInductor workers
  plausible; exact attribution of the twenty children needs PPID/cmdline evidence.
- Prefill statistics are emitted after batch-result processing. Missing prefill
  logs do **not** establish that the OOM happened before vision/first-batch execution.
- `max_running_requests=1` is scheduler admission, not a universal bound on
  outstanding frontend preprocessing, transmission, cancellation or retries.

## 4. Upstream repair status

Status checked during this investigation; open PRs are not released fixes.

| Source | Status and applicability |
| --- | --- |
| [#33391](https://github.com/sgl-project/sglang/pull/33391) | Open; recursively compacts CPU tensors, including metadata. Its `wrap_as_pickle` hook is obsolete for the normal multimodal path in this pinned build; blindly porting it can miss real requests. |
| [#41015](https://github.com/sgl-project/sglang/pull/41015) | Open; compacts features before tokenizer dispatch, an applicable interception point. Narrower metadata coverage; no exact GLM/GB10 qualification. |
| [#29656](https://github.com/sgl-project/sglang/pull/29656) | Merged August 20; native multimodal MessagePack is already bundled. Existing configuration candidate, now verified at codec level here. |
| [#37536](https://github.com/sgl-project/sglang/pull/37536) | Open; reduces raw-image tensor lifetime around embedding. Plausible additional relief; not locally qualified. |
| [#38036](https://github.com/sgl-project/sglang/pull/38036) | Open; proposes frontend preprocessing admission control. Relevant to retry amplification outside scheduler limits. |
| [#38214](https://github.com/sgl-project/sglang/pull/38214) | Open; GLM first-image vision JIT precompilation. Distinct possible peak; not evidence that compilation caused this tenth-image failure. |
| [#41572](https://github.com/sgl-project/sglang/pull/41572) | Merged PTX-KDA workspace fix, but this launch uses Triton linear attention. Do not assign this different backend's failure here. |

If MessagePack has a serving compatibility problem, a compact-storage patch at
the pre-dispatch location is a fallback to audit: ordinary CPU strided tensors,
skip subclasses/transport proxies, include nested metadata, preserve repeated
references. No numerical repair is needed for this serialization mechanism.

## 5. Client cancellation remains a separate problem

Retained nginx access logs show six-minute HTTP-499 cycles at
22:35:26, 22:41:26, 22:47:26, 22:53:26, 22:59:31 and 23:05:39 UTC, followed by
successful responses at 23:11:10 and 23:15:51. During the later OOM interval,
499s recur at 23:21:48 and 23:27:48, followed by a 502 at 23:28:56.
NIM's inference proxy timeout was four hours.

The provider API exposes `settings.headerTimeout=1200000`, but a subsequent
`/api/info` query reported service **2.0.21**, while the CLI reports **2.0.22**.
OpenCode [PR #49229](https://github.com/anomalyco/opencode/pull/49229), commit
`4c0d0ff478ca9150c163fb8b04a76395e4dccafe`, adds both provider-timeout propagation
in `model-resolver.ts` and enforcement in HTTP transport. It is in the
`v2.0.21...v2.0.22` comparison and absent from the inspected 2.0.21 files.

Thus exposing a timeout setting is insufficient to prove its enforcement. The
earlier conclusion that no service restart was needed was not justified. Verify
the actual service version, activate the updated service with approval, then
test an actual request exceeding the former cutoff. Earlier log user-agent
strings said 2.0.22; current service identity alone does not reconstruct every
historical process/version or prove the precise origin of the six-minute cutoff.

Repeated image work during cancellation/retry can compound memory pressure. Its
exact overlap and reclamation remain unmeasured; a longer timeout does not itself
fix image allocation. Evidence: `/tmp/opencode/glm53-second-oom/nginx-access.log`,
`nginx-nginx.conf`, and the preserved client log/database backup.

## 6. NVIDIA release and alternative-runtime research

Registry checks found no newer runnable GLM NIM ARM64 image. Both `latest` and
`2.1.2-variant` select the pinned digest above; their index is
`sha256:75dc19a2c94d023a3aa13603e547c77aa54499781d2919413fff2290646ac827`.
NGC still lists `nim-aa28e1f-nvfp4` as the latest NVFP4 artifact.
[NVIDIA release notes](https://docs.nvidia.com/nim/vision-language-models/latest/release-notes.html)
describe the initial GLM release. Repulling `latest` does not supply a repair.

No located alternative demonstrates the complete target: 400–500K actual coding
context, ten accumulated full-resolution screenshots, and useful concurrency.
The following are publisher/operator measurements, not local reproductions:

| Candidate | Strongest relevant evidence | Qualification gap |
| --- | --- | --- |
| MiaAI TensorFold | 981,841-token needle run, 967 s TTFT, 5.6/9.5 GiB minimum available; separate 50-image/~16K-token run, 12.5 s, 6.75 GiB head minimum | Community EXL3 + default dense Q4; image-token budgets reduce representation; no combined 500K/full-resolution screenshot test. Fresh image-history fixes and unresolved serving stalls. |
| Entrpi EXL3 | 499,245-token needle, 390.8 s; 1,029,486-token needle, 907.8 s with NVFP4 KV | Later production census only 0.5–1.5 GB available; earlier long-context concurrency hard wedge; multimodal profiling skipped. |
| Intel AutoRound + reviewed vLLM | Operator reports exact needles at 419K/836K/947K, 84.05 GiB weight load and >=4.3 GB headroom through 950K | Current reproducible recipe limits images to four; runtime image inherits previously rejected community provenance. Rebuild/review required. |
| eugr B12X/NVFP4-Spark | Operator reports hundreds of 200–500K coding requests; successful 600K cold prefill in 5.5 min | Reported periodic hangs include a 67K decode; experimental runtime and additional dense quantization; no matched screenshot stress qualification. |

Primary references:

- [TensorFold pinned source](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/cf28cc4f8038be322cdeda220c6f1c8ace8f27d1).
  Source committed October 2 labels its new changelog section October 3. Treat
  these as fresh source fixes, not a mature independently qualified release.
  Its image-history reuse/tool-result screenshot repairs are important. Total
  image-token budgeting can change old image representations as image count
  grows, invalidating prefix reuse. Default dense-Q4 quality evidence also shows
  an end-of-turn regression (12/48 truncated versus 1/48 BF16/FP8 in its test).
- [Entrpi pinned findings](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/blob/63f254f7dfc12fb3eaa5449ac84a94f0ab333163/docs/FINDINGS.md).
  Memory-floor figures belong to different revisions/settings; do not attach
  earlier favorable floors to the latest production census.
- [Intel operator report](https://forums.developer.nvidia.com/t/intel-glm-5-3-flash-w4a16-autoround/382041/5)
  and [recipe evidence corrections](https://github.com/florianbrede-ayet/spark-recipes/tree/main/tp2_glm53flash_autoround_mtp3_pmu128).
  Eight-context retention belongs to earlier MTP5 evidence, not direct MTP3
  qualification. Intel weights are the preferred publisher-owned fallback
  research direction if NIM cannot meet the workload.
- [eugr recipe](https://github.com/eugr/spark-vllm-docker/blob/53bd8e034a06290db1fe95ceda7c5a57ae624259/recipes/glm-5.3-flash.yaml)
  and [issue #409](https://github.com/eugr/spark-vllm-docker/issues/409).
  The author corrected an apparent 500K cliff after 600K succeeded; stalls
  remained at smaller contexts. Do not report a universal hard context limit.
- [SGLang two-Spark issue #36941](https://github.com/sgl-project/sglang/issues/36941).
  The author withdrew an initial >40K worker-death claim; unique 102,644-token
  input passed at a smaller chunk. Its sparse-indexer workspace path is not the
  active NIM path (`index_topk=null`).

Demand-backed KV (`kvcached`) remains an interesting separate capacity direction,
but no compatible GLM/NIM/GB10 integration was established. Free slots in the
existing fixed KV pool remain allocated RAM. Neither dynamic-KV claims nor FP8
KV savings address the now-reproduced image serializer defect by themselves.

## 7. Recommended controlled validation, pending approval

1. Exercise the actual typed request envelope through the installed native codec,
   including GLM grid/mrope metadata, sampling and wrapped time statistics.
   The small diagnostic has validated tensor codecs, not all protocol messages.
2. Restart the same NIM configuration with only `SGLANG_USE_PICKLE_IPC=0` added
   consistently on both nodes. Record effective environment and startup pools.
3. Replay the preserved full-resolution nine- and ten-image history over a direct
   long-lived client connection; compare cold and prefix-cached continuations.
   Record actual IPC sizes, both-host MemAvailable/swap/pressure, process tree,
   request IDs, TTFT and token counts. Use a watchdog to abort on sustained
   pressure before another global OOM and confirm both ranks remain healthy.
4. Test sequential repeated turns and controlled cancellation/retry with memory
   recovery checks. Separately activate/verify the updated OpenCode service and
   run an actual long-wait request, rather than relying on configuration output.
5. Only after the original workload is stable, measure 128K → 256K → 400K actual
   multimodal inputs with correctness checks; then 500K and two-session operation
   if preceding results justify them. Report latency as well as capacity.

No candidate promotion, DEFAULT_CONFIG change, publishing or external patch
execution was performed. The original exact failing allocation and any persistent
leak remain unproven; the serialization amplification itself is now reproduced.

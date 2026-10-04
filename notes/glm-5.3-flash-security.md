# GLM-5.3-Flash runtime security review — 2026-10-01

## Decision

**Updated 2026-10-02:** the operator approved NVIDIA's officially distributed and
recommended NIM combination as the baseline, including its bundled checkpoint.
LibertAI attribution does not negate NVIDIA distribution/recommendation. The
current `glm-5.3-flash-nim` profile runs the unmodified official image with its
bundled manifest-verified weights and default trust/listener/inference behavior.
None of the three overlays below is mounted in that profile. Their analysis and
the NVIDIA HF substitution below describe the earlier, superseded candidate.
The observed port conflict is handled using NIM port settings (18000–18002),
not a source patch. Community images/patches remain unused.

Evaluate the **official NVIDIA NIM image** first, with NVIDIA's unchanged,
revision-pinned NVFP4 weights. The community stack below was researched and
statically audited but **has not been executed**. Its complete deployment recipes
are not approved for Chachi. No obvious deliberate credential-exfiltration payload
was found in the reviewed custom patches; substantial unsafe execution paths exist
without requiring malicious intent.

This review distinguishes source inspection from binary provenance, GPU numerical
correctness, and live network verification. A pinned digest identifies an artifact;
it is not a security certification.

## Audited source scope

### Selected NVIDIA NIM: additional image inspection

847 MB of relevant image layers were downloaded and SHA256-verified, then examined
without executing them. Evidence is under `/tmp/opencode/nim-glm53-inspect/rootfs`.
The immutable ARM64 manifest is
`0bd2a1f4ffacf4ee61e1f8c1b48071a6713b51bad954ab0ca627c24be84f43f8`.

Important corrections to the initial publisher-based recommendation:

1. Its default `ngc://nim/zai-org/glm-5.3-flash:nim-aa28e1f-nvfp4` artifact README
   explicitly credits LibertAI. **That model is excluded.** The deployment uses the
   local, separately pinned NVIDIA HF snapshot. NIM's local override bypasses NGC
   manifest checksums, so HF source identity must remain independently pinned.
2. nginx defaults to wildcard API/health listeners, and SGLang defaults to
   `0.0.0.0:8001`, which bypasses nginx route/TLS controls. Both are bound to
   loopback in our candidate; separate wildcard health listeners are prevented by
   setting `NIM_HEALTH_PORT` equal to the role's `NIM_SERVER_PORT`.
3. The Spark tuning defaults to custom-code trust and 94% static memory. Our
   candidate explicitly disables custom model code and uses 85%.
4. No NIM peer ACL protects distributed rendezvous/Gloo/NCCL. Logged primary
   addresses are validated against the registered fabric; host-network data-plane
   exposure still needs live verification. API loopback does not secure the fabric.
5. Full runtime argv is logged. No secrets are put in passthrough arguments.
6. The vendor profile disables sparse-indexer classification using `index_topk=null`.
   This is a numerical/performance qualification concern, not a security fix.

#### Our three minimal overlays

`/opt/nim/etc/nginx.conf.template`: only change the listen address to loopback.

```
before: 63bdbb7aed7a6c36f0cd791b6a23920d1f89430010347d045ac1fc94973185de
after:  d949dbe4eef0b8780f83125035f6b5d1e2bae7c4d1f07d4eab0049aea80311a8
```

`/sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py`: import the runtime's
existing `WeightsMapper` and declare two class-level prefix translations:
`model.language_model.` → `model.`, `model.visual` → `visual`. This allows the
existing ModelOpt quantization-exclusion mapper to follow the same naming scheme
as this model's weight loader. It grants no additional execution/network/host access.

```
before: 0a141565e73252ddb7f1773f30f0c48e001b7dce21a5ca7864b4ea6ae51d0ccd
after:  77cb6a3e7a710990cff61c6a6c91537eb5bf7d134fb935a5ae1e2fb04f9200f6
```

`/usr/local/lib/python3.12/dist-packages/nim_stack/cli/actions.py`: when the
configured model reference is an existing absolute local directory, pass that
directory to SGLang instead of the SDK-generated workspace. The live SDK
enumerated zero files from the HF symlink snapshot and produced an empty workspace.
The correction uses the already hash-checked, read-only NVIDIA snapshot; it adds
no downloads, file writes, custom-code trust, or new execution mechanism.
Nonlocal references retain the original workspace behavior.

```
before: a7ea8e305ba694f6a90297a4cdd77a7b8b23ee2b4303b7b77b3a72cbe7bae52b
after:  ead9070a9afb365d9716d55fe3f34d343033eed66f5119494452699700d60ced
```

The generation step imports only Python stdlib, checks exact preimages, applies
single-occurrence replacements, checks resulting SHA256, and compiles but does not
execute the patched Python. Its container has no network/GPU and only a scoped
output mount. Serving mounts all three files read-only and verifies hashes beforehand.
The original NVIDIA weight/config files remain read-only and unchanged.
This is a statically reviewed compatibility candidate, not proof that the model
loads or preserves quality; live qualification is still required.

### Community alternatives

- **T:** [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/tree/96658cf02227a159a25b330b4ce33a80a548375d),
  commit `96658cf02227a159a25b330b4ce33a80a548375d`.
- **K:** [knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/tree/770d1153062aa916b06591411c61d5c593ea0f03),
  commit `770d1153062aa916b06591411c61d5c593ea0f03`. T's current runbook delegates
  deployment to this repository, making it part of the executed-source closure.

Static review covered executable control paths, build recipes, dynamic loading,
network/privilege boundaries, model conversion, and targeted upstream diffs.
Inventoried trees: approximately 15,079 lines/134 code-build files in T and
92,264 lines/230 in K. Not every native/GPU kernel or inherited dependency was
exhaustively verified. No fetched code, installers, containers, or hardware actions
were executed as part of this source audit.

## Findings

### Critical: host execution of model-generated Python

T `runs/2026-09-20-nvidia-lanes/tools/sr_quality.py:2,50–55` extracts completion
code and executes it via `subprocess.run([sys.executable, '-c', code], ...)`.
Its own documentation specifies root execution. `lane_run.sh:16` invokes it
automatically; another copy exists in the TP4-versus-DeepSeek tooling.

K `bench/qeval_tasks.py:61–79` uses `python -I -S` and a temporary working
directory. Those flags do not isolate filesystem/network/process privileges.
A timeout does not prevent credential access, destructive writes, or spawned
processes. Exploitation requires this evaluator to run and receive malicious code;
ordinary tool-call parsing is not itself host execution.

**Disposition:** do not use these evaluators. Chachi's initial qualification uses
objectively checkable response/JSON tasks and executes no model-generated code.

### High: unauthenticated listeners and enlarged runtime privileges

T `launch-glm53-tp4-24g.sh:55–64,78–85,114–116` and K `start.sh:79–106` combine
root containers, host network/IPC, GPUs/RDMA, unlimited memlock and writable
persistent caches. T's current `env.500k:17–18` overrides K's loopback default
to `0.0.0.0:8000`; the legacy launcher also enables remote model code.
Distributed rendezvous ports include 29521 and 29669.

No primary serving command was found mounting the Docker socket, host root,
SSH keys or the entire home directory, or enabling `--privileged`. Container-root
is not automatically unrestricted host-root. Nonetheless, network reachability
and GPU/RDMA/shared-IPC attack surface require explicit consideration.

**Disposition:** retain project loopback API behavior and ownership-scoped,
minimal mounts. Verify actual listeners. No community launcher is being imported.

### High: destructive fleet operations and host-wide changes

T `glm_boot.sh:13–17` and `glm_settle.sh:7–11` under the historical NVIDIA-lanes
tooling delete both their own and unrelated `vllm_dsv41` containers across
hardcoded hosts. Other recipes change swap/swappiness, compact memory, repeatedly
flush caches, and perform broad process-name termination. See `README.md:421–426`,
`fleet_watchdog.sh:60–65,74–86,120–133`, `flusher-unconditional.sh:12–25`.
No automatic cron/systemd/authorized_keys installation was found in reviewed paths.

**Disposition:** use only omodel-manager's registered-device lifecycle. Preserve
failed rank logs. Its existing scoped pre-launch cache drop is sufficient.

### High, conditional: shell configuration is executable code

T `glm_boot.sh:8–10,22,25` assembles unescaped single-quoted assignments and sends
them through remote shells/local `eval`; controlled knob values can inject commands.
K `start.sh:13,22,79–106` sources environment files and interpolates remote shell
commands. This requires control of configuration values, not just a model prompt.

**Disposition:** JSON profiles, local argv lists, and the existing quoted SSH
transport; no `source`, `eval`, or copied launcher wrappers.

### High: mutable dependencies and incomplete binary provenance

T Dockerfiles v3/v4/v5 use mutable base tags/package sources; v9 installs unpinned
`instanttensor`. K `Dockerfile.roce:16,23–29` pins its base manifest, but T's
2026-09-29 runbook overrides it with a tag without enforcing byte equivalence.
K's optional source-closure verifier expects files not present in the audited
checkout/default invocation (`candidate.env`, `SOURCE_CLOSURE.json`).

The inspected GHCR image resolves to manifest
`sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`;
configuration/image ID is
`sha256:35c6f70ffcba62fd67d7b9d4b4e8300ad177201792ce9cdb1ea18fd449bc23b6`.
These are distinct object types; the configuration digest is not a pullable
manifest. Image history includes inherited unhashed get-pip execution and mutable
package inputs. Inspected v11 history follows v8→DFlash2 rather than the README's
v1→v9 claim; v9's instanttensor installation was not observed in that history.

**Disposition:** do not import this binary chain. Official NIM is pinned by its
ARM64 manifest. NVIDIA binary/package provenance is still a publisher trust
boundary, not independently reconstructed source equivalence.

### High, conditional: persistent executable-cache poisoning

K `overlay/glm_l2_prefetch.py:101–120` loads an existing
`/cache/glm_l2pf/libglm_l2pf_v1.so` using `ctypes.CDLL` without validating its bytes.
The fixed filename and writable persistent cache permit someone with cache-write
access to preserve native code across container replacements. Other JIT caches
are also executable inputs. The RoCE proxy's source-addressed cache is better.

**Disposition:** do not reuse these community executable caches. Give the selected
runtime its own deployment cache. Source/digest pins do not authenticate an old
writable cache.

### High: externally staged replacement code is missing from the audit closure

T `runs/2026-09-20-nvidia-lanes/tools/glm53_tp4.sh:64–78,86–100` bind-mounts
prefix-cache/RoCE replacements from outside the repository. MD5 checks of original
image files are compatibility checks, not authentication of replacement files.

**Disposition:** these launch paths remain unqualified; no external mounts imported.

### Medium–high: model download/conversion does not consistently fail closed

Historical T tooling uses unrevisioned snapshots and `/resolve/main` URLs, a
first-64-byte equality check, checkpoint-controlled filenames, hardlinks, reused
output directories, and failures that print without exiting. References:
`bf_dl.py:11–16`, `fetch_dealign.py:15–21,45–47`, `build_nv.py:23–46,78–89`,
`bf_build.sh:47–64` in the historical run tools. Risks include stale/mixed weights,
path escape through an untrusted index, and modifications propagating via hardlinks.

Current CPU conversion improves isolation (no network, resource limits, caller
UID/GID, fresh output), but the models tree remains writable.

**Disposition:** use unchanged publisher safetensors at an exact HF revision;
do not run community requantization or fetch third-party draft weights.

### Medium: offline model flags do not block arbitrary network traffic

K `start.sh:88–92` sets HF/Transformers offline flags but not the reviewed vLLM
usage-statistics opt-outs. Upstream `usage_lib.py:50–68,187–268` can report
hardware/software/model-architecture metadata to stats.vllm.ai. This is not
evidence that prompts were transmitted. No live network observation was made.

**Disposition:** distinguish download-offline behavior from network isolation;
disable telemetry via documented supported controls and inspect actual runtime.

## Limited patch candidates, if a future vLLM fallback is needed

Small reviewed changes that can be independently transplanted into a controlled
build, subject to matching preimages and numerical/hardware testing:

- T v6 Dockerfile: SM12x PDL gating.
- T `patch_v7.py:6–40`: initialize top-k entries to -1 and bound pool indices.
- T `patch_v8_fp8.py:9–23`: shared-memory tile cap and architecture guard.
- T sparse indexer `:535–543,800–823`: sentinel initialization/SM-count dispatch.
- T historical `mkpatch.py:18–36`: preserve checkpoint-specific attention quant
  configuration. Reversed diffs reproduce original-file MD5 fingerprints; that
  confirms the narrow transformation, not replacement-file authenticity.

DFlash2 cache grouping/auxiliary state, exact-top-k alternatives, K's automatically
loaded `.pth`/`sitecustomize` overlays, custom CUDA kernels, and RoCE hooks require
a combined integration review. RoCE's `all_gather_object` entails pickle trust
among distributed peers. These are not standalone trivial patches.

### Independently checked provenance

- Base vLLM: `487ecf187d3dfe74d2cf6119a92881dba403c219`.
- Candidate KDA diff is limited to quant-config propagation; model diff additionally
  introduces auxiliary-state/interfaces; sparse indexer has three reviewed hunks.
- DFlash2 upstream merge `b389ac29465b33f9e9c534df221ea3c129e9793f`;
  candidate qwen3_dflash2 is byte-identical; speculator adds local Gumbel adaptation.
- All 30 vendored b12x files match recorded hashes and upstream
  `b58f34eaf978277621efced6678e6713fd7122e4`.
- RoCE wrapper full equivalence was not established.
- FlashKDA declares upstream `fb4dfc96cb22fdf2fff3986330031184b9dcf2b9`;
  its complete native diff and inherited CUTLASS headers were not certified.
- MoE Marlin declares upstream `d26f469737fef5c3ec6ee5b5a22b319e6b0c47ac`;
  declaration alone is not independently established equivalence.

## TP2 incompatibility of the complete audited stack

K `start.sh:97,124–129` hardcodes TP4/four ranks. Its enabled prefill sharding
defines TP=4 and explicitly rejects other sizes (`glm_prefill_shard.py:105,608–619`).
Dense-fast kernels document TP4-tuned shapes. Merely changing the launcher's TP
number does not make this stack applicable to Chachi. Sibling TP2 reports are
separate artifacts requiring their own review.

## Scope limitations

This audit did not certify all image layers, installed packages, weight tensors,
GPU memory safety, numerical equivalence, or host firewall state. K's newer
preflight/stop-preserving helpers, read-only model/source mounts, isolated CPU
conversion, and verified b12x vendoring are genuine improvements over older
recipes. They do not remove the findings above.

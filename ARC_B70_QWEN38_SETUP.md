# Arc Pro B70 / Qwen3.8-27B Setup Notes

Research date: 2026-08-21

Status: pre-arrival research. None of the recommendations below have been
validated on our hardware yet. Record observed versions, memory use, correctness,
and performance during bring-up rather than treating estimates as results.

## Next Steps When The B70 Arrives

Run these in order; stop at the first failed gate and record the evidence in the
results table below rather than tuning around it:

1. Assemble the cold-plug OCuLink setup, boot Ubuntu 26.04, and run
   `python3 utils/arc_b70_setup.py preflight --stage hardware`. Require the B70
   PCI ID, `xe`, PCIe 4.0 x4 capability, and an accessible stable render node.
2. Review `python3 utils/arc_b70_setup.py install`, then run it with `--apply`.
   Resolve conflicting Intel sources/packages explicitly; do not bypass a failure
   without identifying its cause.
3. Reboot, correct render-group membership if needed, and run the full
   `python3 utils/arc_b70_setup.py preflight`. Require the pinned llama.cpp commit,
   Level Zero, `SYCL0` identified as the B70, and approximately 32 GiB memory.
4. Run `python3 utils/arc_b70_setup.py download --profile production` to fetch and
   verify the immutable Q4_K_S GGUF.
5. Launch `python3 utils/arc_b70_setup.py serve --profile smoke`. Pass the health,
   deterministic generation, thinking, tool-call, long-output, full-offload, and
   clean-kernel-log gates before increasing context.
6. Launch `python3 utils/arc_b70_setup.py serve --profile production`. Confirm the
   Q4_K_S, Q8-KV, 262,144-token candidate allocates without host fallback; otherwise
   record the first clean context limit instead of spilling over OCuLink.
7. Benchmark 50K, 100K, and 200K prompts, then complete a representative OpenCode
   task. Test 50K concurrency only after the single-slot profile is stable.
8. Test Q5, MTP, vision, or additional slots one change at a time. Download Q5 with
   `python3 utils/arc_b70_setup.py download --profile q5-experiment` before using it.
9. Complete warm-reboot, ten-cold-boot, and 8-24-hour soak gates. Only then install
   and start `omodel-arc-b70.service`, create the harness-agnostic model config, and
   consider promotion into `DEFAULT_CONFIG`.

## Recommendation

Use this sequence:

1. Install Ubuntu Desktop 26.04 as the serving OS. Factory Windows is optional:
   use it only when a firmware update or an independent hardware diagnostic is
   useful, not as a prerequisite to starting Linux bring-up.
2. Start with native `llama.cpp` built with its SYCL backend. Use the repository's
   `utils/arc_b70_setup.py` so OMIX 0.3.0, llama.cpp `b10425`, compiler paths, and
   launch flags are reproducible.
3. Run the short Q4_K_S/F16 `smoke` profile only to prove output correctness and
   full GPU offload. The production candidate is Q4_K_S, the native 262,144-token
   context, Q8 KV, one slot, no vision projector, and no MTP speculative decoding.
4. Treat Q5_K_M at full context as an explicit quality experiment after measured
   Q4 headroom, not as the baseline. Never accept host-memory fallback to make it fit.
5. After correctness and stability pass, test MTP, vision, Q5, and concurrency one
   change at a time.
6. Treat Intel's Dockerized vLLM and SGLang XPU as experiments, not the first
   working path. Do not use Ollama as the primary Intel path yet.

This choice favors a working, debuggable server over the theoretically highest
throughput. A same-architecture Qwen3.6-27B Q4_K test on a B70 measured about
24 tok/s decode and 1,054 tok/s prompt processing in an upstream llama.cpp pull
request. That is useful evidence, but it is not a Qwen3.8 result or a promise for
our OCuLink system.

## Why Linux

### Ubuntu 26.04 advantages

- Intel says Ubuntu 26.04 and later provide full Battlemage support out of the
  box. The B70 (`8086:e223`) is fully supported by the upstream `xe` kernel
  driver starting with kernel 6.17.
- Intel's pinned OMIX stack explicitly supports the Arc Pro B70 and includes the
  Level Zero/OpenCL runtime, SYCL compiler, oneDNN, and oneMKL needed for AI
  workloads.
- Docker can expose the Intel DRM render node directly with `--device /dev/dri`.
- SSH, systemd, Docker, logs, restart policies, and headless operation fit this
  machine's intended role.
- The B70 compute path is independent of which GPU drives the monitor. The AMD
  890M can remain the display adapter.

### Ubuntu 26.04 disadvantages

- Minisforum officially lists Windows 11, not Linux, for the X1 Pro. Fingerprint,
  audio, Wi-Fi, suspend, and vendor update utilities may be less polished.
- The Intel compute stack is moving quickly. Pin known-good OS, OMIX, oneAPI, and
  llama.cpp versions instead of updating all layers independently.
- Native SYCL llama.cpp currently requires more setup than installing a desktop
  application.

### Windows 11 advantages

- It is Minisforum's supported OS and the easiest baseline for BIOS/firmware,
  fingerprint, sleep, and other X1 Pro features.
- Intel's native graphics driver and native Windows llama.cpp SYCL builds make it
  a useful hardware diagnostic path.
- LM Studio is the fastest way to prove a GGUF can load through a GUI.

### Windows 11 disadvantages

- Docker Desktop documents GPU passthrough only for NVIDIA GPUs through WSL2.
  It is not a supported Intel Arc container-serving route.
- Windows Update, desktop sessions, and GUI applications add variability to a
  headless service.
- LM Studio is convenient but offers less control and reproducibility than a
  pinned server build and system service.

### OS decision

Use Ubuntu Desktop 26.04 for the long-running host. Use Windows first, and keep it
available if disk space permits, because it separates a physical/firmware problem
from a Linux software-stack problem. Do not use Ubuntu 25.04; it is obsolete and
predates full B70 enablement. Ubuntu 24.04.4 with a 6.17 HWE kernel is an Intel
OMIX-supported fallback, but 26.04 is the cleaner new installation.

## Runtime Comparison

| Runtime | Performance potential | Convenience | Stability for this exact setup | Decision |
| --- | --- | --- | --- | --- |
| `llama.cpp` SYCL | High for GGUF on Intel; direct Level Zero path; B70 evidence on the same Qwen GDN architecture | Medium; source build on Linux, binary/GUI options on Windows | Best-supported combination of model format, memory fit, and Intel backend, but Qwen3.8/B70 still needs our validation | **Primary** |
| LM Studio | Usually llama.cpp-class performance when the correct Intel-capable runtime and full GPU offload are selected | Excellent GUI, model download, chat, and local API | Good native Windows fallback; Linux AppImage docs say Ubuntu newer than 22 is not well tested; exact B70 runtime selection must be checked | **Fast diagnostic/fallback** |
| Intel vLLM container / llm-scaler | Highest serving/concurrency potential if its optimized kernels and a supported quant work | Good Docker/OpenAI API workflow once working | Intel validates a vLLM 0.21 container on B70, but upstream vLLM marks FP8 W8A8 and GGUF unsupported on Intel GPU. An open B70 issue reports garbage FP8 output | **Experimental comparison** |
| SGLang XPU | Potentially high throughput | Low today: source build or custom Docker build | SGLang targets Arc Pro B-series, but its optimized list contains only small BF16 models and its XPU docs do not validate Qwen3.8 or a fitting quant. Speculative decode is not implemented on XPU | **Experimental, after vLLM** |
| Ollama | Vulkan can work, but a reported B50 comparison was about 3.3x slower than a local SYCL integration | Excellent daemon and model UX | Native SYCL support is still an open proposal; using Vulkan leaves substantial Intel performance untapped | **Do not use as primary** |
| OpenVINO / OVMS | Strong Intel deployment stack in general | Medium | No explicit Qwen3.8 architecture validation found for this setup | **Revisit later** |

Do not confuse Intel's native Linux Docker path with `--gpus all`; that flag and
the NVIDIA Container Toolkit are not the Arc mechanism. Intel containers use the
DRM device, normally `/dev/dri` or a specific render node.

## Model Fit In 32 GB

Qwen3.8-27B is a dense, native multimodal model with 64 language layers: 48
Gated DeltaNet layers and 16 full-attention layers. It has a native 262,144-token
context, thinking enabled by default, `reasoning_effort` control, tool use, and
an included MTP module.

The advertised context is not the practical starting context on a 32 GB card.

| Weights | Published file size | Approx. GiB | Initial assessment |
| --- | ---: | ---: | --- |
| BF16 | 54.66 GB | 50.9 GiB | Cannot fit in B70 VRAM |
| Official FP8 | about 30.9 GB | about 28.8 GiB | Too tight after cache/runtime allocations; Intel FP8 correctness is also suspect |
| GGUF Q8_0 | 29.12 GB | 27.1 GiB | Too tight for a useful cache and runtime buffers |
| GGUF Q6_K | 23.46 GB | 21.9 GiB | Plausible at modest context, but not the first test |
| GGUF Q5_K_M | 20.75 GB | 19.3 GiB | Full-context experiment only after measuring Q4 headroom |
| GGUF Q4_K_M | 17.77 GB | 16.6 GiB | Useful fallback/comparison quant |
| GGUF Q4_K_S | 16.71 GB | 15.56 GiB | Production full-context candidate |

For the 16 full-attention layers, F16 KV cache is approximately 64 KiB/token:

```text
16 layers * K/V * 4 KV heads * 256 head dimension * 2 bytes = 64 KiB/token
```

That is about 2 GiB at 32K, 4 GiB at 64K, 8 GiB at 128K, and 16 GiB at 262K for
one sequence, before GDN recurrent state, compute buffers, driver allocations,
and the optional vision projector. Q8 KV approximately halves the attention
cache, making its full-context attention cache about 8 GiB. The GDN state adds
roughly 154 MiB per sequence, and the vision projector adds about 0.9 GiB. GGUF
file size is not guaranteed to equal final device allocation.

Use Q4_K_S plus Q8 KV as the full-context candidate. The arithmetic suggests it
may fit, but only the live device allocation and logs can establish that. If it
does not fit cleanly, reduce context and record the actual limit rather than
accepting PCIe spill. The PCIe 4.0 x4 OCuLink link is adequate when weights and KV
stay resident in VRAM, but host fallback would make the link much more visible.

## Before Assembly

- Have an Ubuntu Desktop 26.04 installer USB, a Windows recovery path, Ethernet,
  a monitor connected to the X1 Pro's AMD iGPU, and a keyboard available.
- Back up the factory Windows recovery material before repartitioning.
- Download the latest X1 Pro BIOS and Windows driver packages from Minisforum,
  but record the shipped versions before updating.
- The SF1000 is ample and includes a native 12V-2x6/12VHPWR cable. Use only the
  cables shipped for this PSU; modular PSU cables are not interchangeable.
- Verify the dock supports the card's dimensions and cable bend clearance. The
  B70 is a two-slot card approximately 271 mm long. Do not sharply bend the GPU
  power cable at the connector.

## Arrival-Day Hardware Bring-Up

1. With the X1 Pro and SF1000 powered off and unplugged, seat the B70 in the dock,
   attach the dock's control/power connections exactly as its manual specifies,
   fully seat the native 12V-2x6 GPU cable, and connect OCuLink.
2. Treat OCuLink as cold-plug-only. Never connect or disconnect it while either
   side is powered.
3. Keep the display connected to the X1 Pro, not the B70, for initial testing.
4. If the dock manual does not prescribe another sequence, power the dock before
   booting the X1 Pro so the PCIe device exists during enumeration.
5. In BIOS, record defaults and current version. Check for UEFI/CSM, Above 4G
   Decoding, Resizable BAR, PCIe generation, and OCuLink settings. Prefer UEFI,
   Above 4G Decoding enabled, Resizable BAR enabled, and PCIe Gen4. Change only
   one setting at a time if enumeration fails.
6. Boot the intended Ubuntu installation. If PCIe enumeration or firmware remains
   ambiguous, use factory Windows as an independent diagnostic: install the Intel
   Arc Pro driver, verify approximately 32 GB dedicated memory, and run a native
   GPU workload. Update X1 firmware only if needed.
7. Perform at least three full shutdown/cold-boot cycles. A
   device that appears only after warm reboot is not a stable baseline.

## Ubuntu And OMIX Bring-Up

Use a clean Ubuntu Desktop 26.04 install. Do not add Intel's graphics PPA before
OMIX; Intel warns that newer PPA packages can conflict with OMIX's validated,
pinned package set.

The repository utility turns the commands in this section into a checked workflow:

```bash
python3 utils/arc_b70_setup.py install          # dry-run only
python3 utils/arc_b70_setup.py preflight --stage hardware
python3 utils/arc_b70_setup.py install --apply  # explicit apt/repo/build changes
sudo reboot
python3 utils/arc_b70_setup.py preflight
```

`install --apply` refuses non-Ubuntu-26.04 hosts, conflicting Intel apt sources,
and pre-existing Intel compute packages by default. It requires the exact reviewed
Intel repository-key bundle (`E0258B57D9C442D5DB1855C271740E4DE392BFE3` and
`4E9EFCDEF82800256C1E7C64B02DB9BD8C321DCB`), refuses a modified llama.cpp
checkout, and recreates only its own commit-specific build directory. Override
the OS/source/package checks only after manually explaining the host state; they
are intended to fail closed.

Before installing Intel user-mode packages:

```bash
uname -r
lspci -nn | grep -Ei 'VGA|DISPLAY'
ls -l /dev/dri /dev/dri/by-path
```

Expected B70 PCI ID: `8086:e223`. Obtain its PCI address from `lspci`, substitute
it for `<BDF>`, and verify the `xe` driver and link:

```bash
lspci -k -s <BDF>
sudo lspci -vv -s <BDF> | grep -E 'LnkCap|LnkSta'
journalctl -k -b | grep -Ei 'xe|drm|firmware|guc|aer|pcie|bar|reset|timeout'
```

The target link is PCIe 4.0 x4: `16GT/s` and `Width x4`. A lower link during an
idle power-saving state is not automatically a fault; check under load as well.
Repeated AER errors, x1/x2 training, or intermittent absence point to power,
cable, dock, firmware, or signal-integrity problems before they point to the
inference runtime.

Install the exact OMIX 0.3.0 release by following Intel's live instructions.
As of the research date, the documented commands are:

```bash
sudo apt update
sudo apt install -y gnupg wget
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
. /etc/os-release
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/intel-omix/0.3.0 unified" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu-${VERSION_CODENAME}.list
sudo apt update
sudo apt install -y intel-omix intel-omix-dev clinfo
sudo reboot
```

After reboot:

```bash
stat -c '%n %G' /dev/dri/render*
groups "$USER"
source /opt/intel/oneapi/setvars.sh
sycl-ls
clinfo -l
```

If the render node's group is `render` and the user is not a member:

```bash
sudo gpasswd -a "$USER" render
```

Log out and back in after changing groups. Because the X1 Pro also has an AMD
iGPU, do not assume `renderD128` is the B70. Map the PCI BDF to the stable name in
`/dev/dri/by-path`.

## Build The Baseline Server

Build a pinned llama.cpp version rather than an unknown rolling binary. Bartowski
created the quant with release `b10419`. Use `b10425`, whose commit includes the
B70 Gated DeltaNet optimization merged on 2026-08-14, or a deliberately recorded
newer version.

```bash
sudo apt install -y git cmake build-essential libssl-dev
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout --detach 3d93885352a0049c8388a0da0698ec1a69e60d90
source /opt/intel/oneapi/setvars.sh
cmake -B build \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_HOST_MEM_FALLBACK=OFF \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(nproc)"
./build/bin/llama-server --list-devices
```

Confirm the B70 is listed as `SYCL0` with approximately 32 GB. Do not proceed if
the server sees only a CPU, uses the AMD iGPU through another backend, or silently
falls back to host memory. Retain the CMake configure output and record whether it
found and enabled oneDNN; a SYCL build can still complete without the intended
oneDNN path. Retain the configure output and exact compiler/runtime package
versions for the results record.

## Baseline Qwen3.8 Launch

The model is ungated, but a moving `main` revision is not reproducible. The utility
downloads `Qwen3.8-27B-Q4_K_S.gguf` from immutable repository revision
`f0eec4a4bb4975114a030d048952d83c0a53c034`, requires size 16,713,148,000 bytes
and SHA-256 `9282674b002aac8d9d5eda7f53f5114d7fc91725f5a6962a03738571afb2218d`,
and rechecks the local file before every launch:

```bash
python3 utils/arc_b70_setup.py download --profile production
```

Inspect the production command without starting it:

```bash
python3 utils/arc_b70_setup.py serve --profile production --dry-run
```

The generated command is equivalent to:

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1
./build/bin/llama-server \
  --model ~/.cache/otools/models/qwen3.8-27b-arc-gguf/f0eec4a4bb4975114a030d048952d83c0a53c034/Qwen3.8-27B-Q4_K_S.gguf \
  --no-mmproj \
  --device SYCL0 \
  --split-mode none \
  --main-gpu 0 \
  --gpu-layers all \
  --fit off \
  --ctx-size 262144 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --reasoning-format deepseek \
  --alias qwen3.8-27b-arc-gguf \
  --host 127.0.0.1 \
  --port 8000
```

Before relying on this exact command, check `./build/bin/llama-server --help`;
the CLI changes quickly. The utility defaults to loopback. Passing
`--host 0.0.0.0` exposes the unauthenticated API, so do that only when port 8000
is restricted to the trusted LAN.

Run `--profile smoke` first, but use it only as a short correctness/offload gate.
If the production candidate fails to allocate or invokes host-memory fallback,
reduce context and record the first clean capacity. Do not solve a VRAM problem
by spilling weights or KV across OCuLink. Run `--profile q5-experiment` only after
Q4 measurements show enough headroom, and download its separately pinned GGUF first:

```bash
python3 utils/arc_b70_setup.py download --profile q5-experiment
python3 utils/arc_b70_setup.py serve --profile q5-experiment
```

After the production profile is qualified, print and review the user service, then
install it. Installation verifies the model again and enables systemd user
lingering so it starts at boot without an interactive login; it does not start the
server unless `--start` is explicit.

```bash
python3 utils/arc_b70_setup.py systemd
python3 utils/arc_b70_setup.py systemd --install
systemctl --user start omodel-arc-b70.service
journalctl --user -u omodel-arc-b70.service -f
```

## Validation Gates

Do these in order and retain the server log for every run:

1. Confirm the log names `SYCL0`, shows all model layers on the B70, and contains
   no host fallback, device loss, AER, reset, or allocation warning.
2. Check `http://127.0.0.1:8000/health` and `/v1/models`.
3. Run a factual prompt twice with `temperature: 0` (or the current server's
   documented greedy setting). Reject repeated punctuation, garbage tokens, empty
   content after thinking, or materially inconsistent output.
4. Test thinking on and off through `chat_template_kwargs.enable_thinking`. Pass
   `low`, `medium`, and `xhigh` through
   `chat_template_kwargs.reasoning_effort`, not only as a top-level API field.
   Confirm the rendered prompt changes and reasoning is separated from final
   content; an HTTP 200 alone does not prove the template consumed the option.
5. Test one forced tool call, one automatic tool call, and continuation after a
   tool result. Validate parsed JSON, not just readable XML in raw text.
6. Generate at least 4K output tokens and check for repetition or corruption.
7. Run 50K, 100K, and 200K requests while watching device memory, temperature,
   power, clocks, kernel logs, and PCIe errors. Then run one representative
   OpenCode coding task against a realistically populated repository context.
8. Benchmark one, two, and four concurrent requests. Restart with the matching
   number of server slots and enough total context for every slot. Record TTFT,
   prompt tok/s, per-request decode tok/s, wall time, peak VRAM, and power. The
   current repository benchmark does not calculate aggregate decode throughput.
9. Repeat after a warm reboot and at least ten cold boots. Then run an 8-24 hour
   soak test before calling the configuration stable.

The production candidate has one slot. Measure single-user behavior at realistic
working sizes with room for the template and output:

```bash
python3 utils/benchmark_concurrent.py 127.0.0.1 1 --context 50000
python3 utils/benchmark_concurrent.py 127.0.0.1 1 --context 100000
python3 utils/benchmark_concurrent.py 127.0.0.1 1 --context 200000
```

For a representative concurrency sweep, keep each prompt at 50K. Restart the
utility with `--context 65536 --parallel 1`, then `--context 131072 --parallel 2`,
then `--context 262144 --parallel 4`. Run each benchmark locally on the B70 host:

```bash
python3 utils/arc_b70_setup.py serve --profile production --context 65536 --parallel 1
python3 utils/benchmark_concurrent.py 127.0.0.1 1 --context 50000

python3 utils/arc_b70_setup.py serve --profile production --context 131072 --parallel 2
python3 utils/benchmark_concurrent.py 127.0.0.1 2 --context 50000

python3 utils/arc_b70_setup.py serve --profile production --context 262144 --parallel 4
python3 utils/benchmark_concurrent.py 127.0.0.1 4 --context 50000
```

Each server invocation occupies the terminal; stop it before launching the next
variant, and run the benchmark from a second local shell.

`--ctx-size` is total server context when slots do not use a unified dynamic cache;
simply increasing `--parallel` can divide the existing context among slots. Check
the startup log for actual per-slot capacity. If N=4 does not fit, record that as
the capacity result rather than reducing only that run's prompt and comparing
unlike workloads.

## Controlled Tuning Order

Change one item, rerun correctness, then benchmark:

1. Q4_K_S at 50K, 100K, 200K, and the full configured 262K capacity.
2. Q8 KV versus F16 KV at a context where both fit, checking output quality as
   well as memory and speed.
3. Q5_K_M versus Q4_K_S at the largest context each can hold without fallback.
4. MTP with `--spec-type draft-mtp`. Keep it only if generation quality remains
   clean and measured end-to-end latency improves.
5. Vision by removing `--no-mmproj`. Confirm the projector is loaded and test a
   known 4x4 solid-blue image. Measure its memory cost.
6. `--parallel 2`, then higher concurrency only if per-request latency remains
   acceptable. Increase total `--ctx-size` with the slot count so each request
   retains the intended capacity.
7. Q6 only if Q5 leaves abundant measured headroom and the quality gain matters.

Do not enable persistent SYCL JIT caching just because an online recipe suggests
it. llama.cpp explicitly warns that `SYCL_CACHE_PERSISTENT=1` can mix stale and
new binaries and cause crashes. Startup speed is not decode speed.

## Alternative Paths

### LM Studio

Use LM Studio on Windows to validate the model quickly:

- Import the same Bartowski Q4_K_M first.
- Select the Intel-capable llama.cpp/Vulkan runtime offered by the installation.
- Require full GPU offload and inspect logs; do not accept a CPU fallback.
- Set 32K context and one loaded model.
- Start its OpenAI-compatible local server and run the same correctness prompts.

If LM Studio works while Linux SYCL does not, the hardware is probably sound and
the problem is in the Linux/OMIX/build combination. If both fail or the B70 is
missing from both operating systems, return to power, BIOS, OCuLink, and PCIe
enumeration.

### Intel vLLM Container

Intel publishes `intel/vllm:0.21.0-ubuntu24.04-20260805` (amd64 digest
`sha256:a76ac6b89350e0b3f5abccbfbe38474a27635666f988b4c3c181d856e51beecc`),
validated on B70, but calls the image not intended for production. Its published
validation used an Ubuntu 24.04.4 host and OMIX 0.1.0 in the image, so our Ubuntu
26.04/OMIX 0.3.0 host is a new combination to verify. Test it only after the GGUF
baseline. Pass `/dev/dri`, not NVIDIA's `--gpus all`, and expose the stable B70
render node where possible.

The key blocker is a quant that fits and produces correct output. Upstream vLLM's
quantization table currently says:

- AWQ and GPTQ: supported on Intel GPU in general.
- LLM Compressor FP8 W8A8: not supported on Intel GPU.
- GGUF: not supported on Intel GPU.

The open B70 FP8 issue demonstrates that successful loading is not enough; compare
fixed prompt outputs against the known-good llama.cpp baseline before measuring
speed. Do not start with the official Qwen FP8 checkpoint.

### SGLang

SGLang's XPU backend explicitly targets Arc Pro B-series and provides a Dockerfile,
but its current verified list is limited to small BF16 models on B580. Qwen3.8 BF16
does not fit this card, and the fitting quant path is unproven. It is therefore a
later engineering experiment, not an arrival-day route.

### Ollama

Ollama can use Vulkan on Intel Arc, but native SYCL remains an open proposal. The
proposal's B50 test reported approximately 10 tok/s through Vulkan versus 33 tok/s
through a local SYCL integration. That is one test, not a B70 guarantee, but it is
enough reason not to make Ollama the performance baseline.

## omodel-manager Impact

The current `omodel-manager` is NVIDIA-specific for GPU launch and install checks:
it emits `--gpus`, checks `nvidia-smi`/CDI, and its committed Qwen3.8 profiles are
for DGX Spark CUDA images. Do not use those profiles on the B70.

After the native B70 baseline is proven, the minimal manager work will be:

- Add an explicit Intel accelerator path that passes the correct `/dev/dri`
  render node and groups without weakening the NVIDIA path.
- Add Intel-aware install/health diagnostics (`xe`, Level Zero/SYCL, render-node
  permissions) instead of pretending NVIDIA CDI is present.
- Add a pinned SYCL llama.cpp image/build and a Qwen3.8 GGUF launch profile in
  `model_manager.json` first.
- Create a GGUF-specific harness-agnostic config for the distinct
  `qwen3.8-27b-arc-gguf` served ID. Reuse only model capabilities observed on the
  live endpoint; its context, output limits, concurrency, and vision status must
  not inherit the larger DGX BF16/FP8 declarations.
- Promote into `DEFAULT_CONFIG` only after correctness, benchmark, reboot, and soak
  gates pass on the physical system.

## Results To Record

| Item | Observation |
| --- | --- |
| X1 BIOS version | |
| Ubuntu/kernel | |
| B70 PCI ID and BDF | |
| `xe` driver / OMIX / Level Zero versions | |
| PCIe link idle / loaded | |
| llama.cpp tag and commit | |
| GGUF repo revision and file SHA-256 | |
| Quant / KV type / context / slots | |
| Full GPU offload confirmed | |
| Peak VRAM / host RAM | |
| 50K N=1 TTFT / prefill / decode | |
| 100K N=1 TTFT / prefill / decode | |
| 200K N=1 TTFT / prefill / decode | |
| Representative OpenCode task | |
| 50K N=2 per-request / wall time | |
| 50K N=4 per-request / wall time | |
| Thinking / tools / long generation | |
| MTP result | |
| Vision result | |
| Cold boots passed | |
| Soak duration / errors | |

## Primary Sources

- [Minisforum AI X1 Pro specifications](https://www.minisforum.com/products/minisforum-ai-x1-pro)
- [ASRock Arc Pro B70 Creator specifications](https://www.asrock.com/Graphics-Card/Intel/Intel%20Arc%20Pro%20B70%20Creator%2032GB/)
- [Corsair SF1000 specifications and cable list](https://www.corsair.com/us/en/p/psu/cp-9020257-na/sf-series-sf1000-fully-modular-80-plus-platinum-sfx-power-supply-cp-9020257-na)
- [Intel installation-path guidance](https://dgpu-docs.intel.com/installation-guides/index.html)
- [Intel B70 `xe` driver support table](https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html)
- [Intel OMIX installation](https://dgpu-docs.intel.com/installation-guides/installing-omix.html)
- [Intel vLLM 0.21.0 B70 validation](https://dgpu-docs.intel.com/overview/release-notes/containers/vLLM/0.21.0.html)
- [Docker Desktop Windows GPU limitation](https://docs.docker.com/desktop/features/gpu/)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.8-27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)
- [Pinned Bartowski Qwen3.8-27B GGUF tree](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/tree/f0eec4a4bb4975114a030d048952d83c0a53c034)
- [Intel AutoRound Qwen3.8 Q4_K_M](https://huggingface.co/Intel/Qwen3.8-27B-q4km-AutoRound)
- [llama.cpp SYCL documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [llama.cpp B70/Qwen3.6-27B measurements](https://github.com/ggml-org/llama.cpp/pull/26643)
- [vLLM quantization hardware matrix](https://docs.vllm.ai/en/latest/features/quantization/)
- [Open vLLM B70 FP8 correctness issue](https://github.com/vllm-project/vllm/issues/48058)
- [SGLang XPU documentation](https://docs.sglang.io/platforms/xpu.html)
- [Open Ollama SYCL proposal and Vulkan comparison](https://github.com/ollama/ollama/issues/16930)
- [LM Studio system requirements](https://lmstudio.ai/docs/app/system-requirements)

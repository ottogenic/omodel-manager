---
name: edit-launch-profiles
description: Edit launch profiles or launch behavior in omodel-manager — Docker/vLLM/SSH documentation pointers and the vetted mechanisms & gotchas (--rm eats crash logs, CDI GPU check, extends can't remove keys, defaults are additive, colored-logs env var, etc.). Use when changing how containers are built or launched.
---

Use this skill when changing how omodel-manager builds or launches vLLM Docker
containers — editing launch profiles, `docker run` argv construction, or launch
behavior. It carries the Docker / vLLM / SSH documentation pointers and the
vetted mechanisms & gotchas so you don't re-learn them the hard way.

For prior build findings, read only the matching `notes/<profile>.md` record. Do not
generalize one model's findings into shared hardware guidance. For **benchmarking a
profile's real-context speed and parallel cost**, load the **benchmark-model** skill.

## Reference — Docker / vLLM / SSH (WebFetch pointer lines)

Editing profiles or launch behavior means knowing how Docker and vLLM interpret the
flags. Fetch these when you need current detail (they move; treat the live docs as
authoritative over this file). Paste a line to an AI tool or fetch it yourself:

- use webfetch to find documentation/info on topic **docker run flags (-d/-t/--rm/--name/--label/-e/-v/--gpus)**: https://docs.docker.com/reference/cli/docker/container/run/
- use webfetch to find documentation/info on topic **docker GPU access (--gpus, device requests)**: https://docs.docker.com/engine/containers/resource_constraints/#gpu
- use webfetch to find documentation/info on topic **NVIDIA Container Toolkit install + CDI (why modern Docker has no named "nvidia" runtime)**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- use webfetch to find documentation/info on topic **vLLM server / `vllm serve` CLI args**: https://docs.vllm.ai/en/latest/cli/serve/
- use webfetch to find documentation/info on topic **vLLM environment variables (VLLM_LOGGING_COLOR, VLLM_ATTENTION_BACKEND, VLLM_ALLOW_LONG_MAX_MODEL_LEN, VLLM_TEST_FORCE_FP8_MARLIN)**: https://docs.vllm.ai/en/stable/configuration/env_vars/
- use webfetch to find documentation/info on topic **vLLM NVFP4 backend flags (--moe-backend / --linear-backend; the marlin/auto values that replaced VLLM_NVFP4_GEMM_BACKEND & VLLM_USE_FLASHINFER_MOE_FP4)**: https://docs.vllm.ai/en/stable/cli/serve/
- use webfetch to find documentation/info on topic **vLLM reasoning outputs (reasoning_effort, chat_template_kwargs.enable_thinking)**: https://docs.vllm.ai/en/latest/features/reasoning_outputs/
- use webfetch to find documentation/info on topic **vLLM using Docker (image, entrypoint, args)**: https://docs.vllm.ai/en/stable/deployment/docker/
- use webfetch to find documentation/info on topic **vLLM quantization (NVFP4 / ModelOpt auto-detect, --quantization)**: https://docs.vllm.ai/en/latest/features/quantization/
- use webfetch to find documentation/info on topic **HuggingFace resolve URL pattern (for `assets` downloads: /resolve/main/<file>)**: https://huggingface.co/docs/hub/en/how-to-downstream
- use webfetch to find documentation/info on topic **OpenSSH IdentitiesOnly / -i key pinning (ssh_config)**: https://man.openbsd.org/ssh_config
- use webfetch to find documentation/info on topic **ssh-copy-id (installing the public key)**: https://man.openbsd.org/ssh-copy-id
- use webfetch to find documentation/info on topic **DOCKER_HOST=ssh:// & docker context (alternative remote transport)**: https://docs.docker.com/engine/manage-resources/contexts/

## Mechanisms & gotchas this tool handles (vetted in development)

- **`--rm` eats crash logs.** Detached launches use `--rm`, so a container that crashes
  during startup is auto-removed with its logs. `--keep` omits `--rm` so it persists for
  `logs`; `--foreground` streams the crash live. New profiles: always `--dry-run`, then
  launch and watch (`--keep`/`--foreground`) the first time.
- **`docker --filter name=^x$` is unreliable** — container names are stored with a leading
  `/`, so the `^` anchor misses. `resolve_target` lists names and matches exactly.
- **Colored logs need an env var, not a TTY.** Most output comes from the vLLM
  **EngineCore subprocess**, forwarded over pipes; a container TTY only reaches the main
  process. `VLLM_LOGGING_COLOR=1` (in `defaults.env`, inherited by the subprocess) forces
  color regardless. `-d` (no `-t`) is used to avoid storing control-char noise.
- **Bind-mount paths belong to the docker *daemon host*.** For remote, `~`/`hf_cache`
  resolve against the **remote** `$HOME` (`remote_home()`), and declared `assets` are
  downloaded locally then `scp`'d to the box.
- **HF_TOKEN delivery.** Locally it's `-e HF_TOKEN` (inherited — stays out of argv).
  Remotely it's forwarded by value (`-e HF_TOKEN=…`), which IS visible in `docker inspect`
  on the box — acceptable (you own the box), documented, and masked in `format_run`.
- **GPU-runtime check is multi-signal.** Modern Docker (25+/29) uses **CDI**, so
  `docker info`'s `.Runtimes` has no `nvidia`. `install` passes if any of: registered nvidia
  runtime, `nvidia-ctk` present, or a CDI spec in `/etc/cdi`.
- **`extends` can't remove keys.** Deep-merge only overrides/adds. When one profile must
  *omit* something a sibling has (e.g. 512K drops `--speculative-config`), write two full
  profiles rather than extending.
- **`defaults` are additive.** `docker_flags`/`volumes`/`assets` concatenate; `env`/
  `vllm_args` merge (profile wins). A profile inherits `--privileged`, ulimits, and
  `--trust-remote-code` even if its source recipe omitted them (harmless, more permissive).
- **Ctrl-C never dumps a traceback** (global handler + a `logs` handler that reassures the
  container is still running).

## Verify against reality (don't assume)

- **vLLM flags are image/version-specific.** A working recipe on one image can reject a
  flag on another (a real crash we hit). Confirm a new profile with `launch --dry-run`,
  then `--keep`/`--foreground` and read the startup log before trusting it.
- **`--quantization` is usually auto-detected** for NVFP4/ModelOpt checkpoints — omit it
  unless a model needs it explicitly.
- **YaRN long-context extrapolates** past the trained window (e.g. 27B 512K, `factor 2.0`);
  memory is cheap but verify output quality at very long contexts.
- **`install` does NOT auto-install nvidia-container-toolkit** — it checks and advises with
  the install link (distro-specific; left to the operator).
- **Benchmark speed at a real context, and check the parallel cost** — this is handled by a
  separate skill; load the **benchmark-model** skill.

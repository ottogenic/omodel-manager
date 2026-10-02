# Qwen3-ASR 0.6B on RTX 4070 — qualification record

## Sources and intent

- Official [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, Apache-2.0. Approximately 1.88 GB of BF16 weights including the audio encoder; marketed as 0.6B (Hugging Face counts 938M total parameters).
- Official [Qwen3-ASR runtime](https://github.com/QwenLM/Qwen3-ASR), commit `7c6daf77a2421100f5fb066495372c00129d39ff`, version 0.0.6. Its `qwen-asr-serve` entry point registers the ASR architecture with vLLM and provides `/v1/audio/transcriptions`.
- `vllm/vllm-openai:v0.14.0` image digest `sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd` has vLLM 0.14.0 and Transformers 4.57.6. Plain vLLM does **not** register `Qwen3ASRForConditionalGeneration`; the pinned Qwen package is necessary.
- Alternative official `Qwen/Qwen3-ASR-0.6B-hf` is native in Transformers >=5.13.0, but the original checkpoint plus vendor `qwen-asr-serve` is selected for direct OpenAI transcription compatibility.

## Co-hosting and interface

The machine's RTX 4070 is 12,282 MiB. The existing Qwen3-TTS 1.7B Base service occupies 4,614 MiB idle on this GPU. `rtx4070-asr` is a distinct lifecycle slot on the **same card** as the TTS slot `rtx4070`. ASR is host-loopback on 8003, with GPU memory fraction 0.35, one request sequence, eager execution and 4096-token serving limit to constrain resident memory. The existing B70 vLLM runs on another card on port 8000.

The ASR config declares `task = "asr"` and OpenAI audio-transcriptions interface, not chat presets or a tok/s benchmark. OpenChamber's server STT provider takes an OpenAI-compatible transcription URL; do not misregister ASR as a chat model in downstream adapters.

## Live validation

Initial build installed all `qwen-asr` extras including Gradio 6.17.3, which upgraded Starlette to 1.7.0. The vLLM 0.14.0 metrics middleware requires Starlette <1, causing `/v1/models` HTTP 500 (`_IncludedRouter` has no `.path`). The model itself loaded in 1.53 GiB, and vLLM reported `['generate', 'transcription']` tasks. Rebuilt the image with the pinned vendor source `--no-deps` and only the serving dependencies to preserve vLLM's FastAPI/Starlette versions. The first server was stopped by `omm stop rtx4070-asr` before replacing it. Qwen TTS remained healthy throughout.

With the corrected image (`otools-qwen3-asr:0.0.7`), `omm health rtx4070-asr` returned READY. The official English WAV sample (`asr_en.wav`, 48 kHz, 15.05 s) returned HTTP 200 and a plausible English transcript in 1.17 s initially, then 0.98 s warm; FLAC returned HTTP 200 in 0.97 s. The output text began `language English<asr_text>` before the spoken words. A WebM/Opus encode of the same sample returned HTTP 500 (`Format not recognised` from the audio loader). OpenChamber may upload `audio/webm`; direct integration needs conversion to WAV/FLAC and removal of the language prefix from the result. That conversion belongs in an integration layer, not in the model launcher.

During simultaneous ASR transcription and TTS voice cloning, both requests succeeded: TTS generated 57,600 frames of 24 kHz WAV, ASR returned its transcript, and host `nvidia-smi` sampled a peak of 8,376 MiB / 12,282 MiB (0.2 s polling). This is an observed sampled peak, not a guaranteed upper bound. `omm health rtx4070` and `omm health b70` both reported READY afterward. ASR, TTS and B70 were left running.

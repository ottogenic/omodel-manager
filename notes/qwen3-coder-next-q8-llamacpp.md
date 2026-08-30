# Qwen3-Coder-Next Q8 llama.cpp

Alternative engine record for isolating vLLM-specific issues. Tested 2026-07-25.

- Model: Unsloth Qwen3-Coder-Next Q8_0 GGUF, three shards (approximately 85 GB)
- Image: `scitrera/dgx-spark-llama-cpp:b10107-cu131`
- Profile helper: `python3 add-llamacpp-profile.py`
- Context: 131,072 tokens, one parallel request, all layers on GPU
- Server: `llama-server` with Jinja tool templates

The helper recreates the local `qwen3-coder-next-q8-llamacpp` sandbox profile. Its unusual
field mapping is intentional: the image entrypoint receives no positional model argument, so
the profile's model field carries `--jinja`; remote code is disabled and the GGUF path is passed
through the remaining server arguments.

Validation passed `/v1/models`, native tool-call parsing, and a full loom pipeline run. A
three-run coding comparison matched the vLLM FP8 profile closely enough that the observed
crashes and defects were treated as model-level rather than engine-specific.

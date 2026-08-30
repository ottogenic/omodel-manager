# Model build notes

Keep build research and hardware findings in one file per exact model/profile. Use the
profile key as the filename, for example:

- `notes/qwen3.8-flash-next-fp8.md`
- `notes/qwen3-coder-next-q8-llamacpp.md`

Each note should identify the exact model revision, image or source revision, build date,
hardware, commands or configuration tested, observed result, and unresolved follow-ups.
Date new findings instead of rewriting old observations without explanation.

Treat every finding as specific to that model, quantization, runtime revision, and build.
Do not turn one model's result into shared Spark guidance or a cross-model trap. Before
building, read only the note for the profile being changed and verify current upstream
documentation independently.

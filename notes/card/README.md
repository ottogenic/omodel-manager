# Card Qualification Records

This directory retains the hardware, runtime, and model evidence that qualified
the checked-in Intel Arc Pro B70 deployment. The supported manager lifecycle is:

```bash
omm plan b70 qwen3.8-27b-gptq-int4-b70
omm launch b70 qwen3.8-27b-gptq-int4-b70
omm logs b70 -f
omm health b70
omm stop b70 -y
```

The production implementation is stdlib-only code in `utils/card/`. It pins the
vLLM XPU image and model revision, verifies every model file, exposes only the
B70 render node, places the model on an internal Docker network, and publishes
the API only on host loopback through a constrained proxy.

The llama.cpp and RTX 4090 records describe qualification and comparison work.
They are not normal `omodel-manager` deployment paths and their old setup
utilities are intentionally not part of the production helper set.

- `b70-qwen3.8-setup.md`: reviewed arrival-day and host bring-up procedure.
- `b70-qwen3.8-llamacpp.md`: retained native SYCL experiments and safety findings.
- `b70-qwen3.8-vllm.md`: production vLLM qualification and exact identities.
- `rtx4090-qwen3.8-llamacpp.md`: OCuLink comparison procedure and baseline.
- `../../results/card/`: retained raw benchmark and boundary evidence.

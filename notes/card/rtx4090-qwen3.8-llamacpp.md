# RTX 4090 / DEG2 OCuLink llama.cpp Qualification

Research date: 2026-08-23. This was a hardware-comparison procedure, not an
`omodel-manager` launch profile. The setup utilities from the qualification repo
are intentionally not included under `utils/card/` because this path was not
promoted to production.

## Safety Boundary

- The RTX 4090 temporarily replaced the B70 in the existing DEG2 dock.
- OCuLink was treated as cold-swap only: shut down host, power off dock and PSU,
  unplug AC, wait for all power indications to stop, then change hardware.
- The card required a fully seated native 12VHPWR/12V-2x6 cable from the exact
  PSU cable set, suitable bend clearance, and physical support.
- Dock/PSU power came up before the host. The monitor remained on the host iGPU.

## Qualification Gates

- PCI ID `10de:2684`, bound to `nvidia`.
- RTX 4090 identity, compute capability 8.9, and at least 24,000 MiB from
  `nvidia-smi`.
- Host-facing PCIe 4.0 x4 capability and active x4 width under load.
- Native llama.cpp identifies `CUDA0`; the launch binds the verified PCI GPU
  identity rather than trusting an ambient ordinal.
- No NVRM Xid, AER, reset, timeout, fallen-off-bus, host-memory, or CUDA unified
  memory spill.

The matched historical launch used llama.cpp `b10425` commit
`3d93885352a0049c8388a0da0698ec1a69e60d90`, Ada `sm_89`, the same immutable
Qwen3.8 Q4_K_M artifact as the B70 lane, full GPU offload, fitting disabled,
262,144 context, one slot, 2048/512 batch settings, Q4 KV, Flash Attention, and
embedded MTP2. The 262K profile was expected to be tight on 24 GiB and any OOM
was to be accepted rather than hidden by fitting, partial offload, or spill.

## Matched Baseline

| Prompt | DEG2 B70 | Remote RTX 4090 (`cpgaming`) |
| --- | ---: | ---: |
| ~55K decode | 15.3 tok/s | 40.1 tok/s |
| ~55K prefill | 406 tok/s | 1,532 tok/s |
| ~110K decode | 11.3 tok/s | 26.5 tok/s |
| ~110K prefill | 265 tok/s | 1,334 tok/s |

The later qualified B70 vLLM combination reached 40.5 tok/s decode at ~55K and
36.8 tok/s at ~110K while preserving the card's 262K deployment goal. That
result, not this temporary CUDA comparison, is the supported card path.

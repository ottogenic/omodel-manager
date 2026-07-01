# `configs/` — generic per-model configs (source of truth)

These are **harness-agnostic** model configs: what a model can do (capabilities)
and how to sample it per task (presets), with a human tuning README. One file per
model, keyed to a launch profile in `model_manager.json`.

**omodel-manager owns and stores these.** It does not render them — it just holds
the files and validates them (see `test_configs.py`). Downstream **adapters
consume them** and translate to a specific agent harness:

- `omodel-wire` → OpenCode (agents, permissions, colors, `chat.params` plugin)
- future adapters → pi.dev, Claude Code, …

So keep these files **generic**: no OpenCode/pi.dev-specific keys (agent names,
permissions, colors). Those belong in the adapter.

## One file per model

- Named to match a `model_manager.json` profile/model key
  (`qwen3.6-35b-nvfp4.md` ↔ the `qwen3.6-35b-nvfp4` launch profile).
- Each file is **both a tuning README and a machine config**: prose docs + a
  single fenced ` ```json ` block (the doc is the comment JSON lacks).
- Adapters extract the **first ` ```json ` block**, `json.loads` it, and match a
  discovered served-model-id against its `match` list. `README.md` is skipped.

## The machine block

| Key | Meaning |
|-----|---------|
| `match` | Substrings matched (case-insensitive) against the served-model-id from `/v1/models`. Include the manager key **and** the HF/served id(s). |
| `source` | Model-card URL the tuning came from (docs only). |
| `capabilities.vision` | `false`, or `{ "input": ["text","image"], "output": ["text"] }`. Adapters that support images write the harness's equivalent. **Replaces live vision probing.** |
| `capabilities.reasoning` | `true`/`false` — does the model emit chain-of-thought. **Replaces reasoning probing.** |
| `capabilities.tool_call` | `true`/`false`. |
| `capabilities.thinking_control` | How thinking toggles: `"enable_thinking"` (Qwen `chat_template_kwargs`), `"reasoning_effort"` (vLLM low/med/high), `"soft_switch"` (`/think` `/no_think`), or `"none"` (template default; per-preset `options` carry knobs). |
| `context.native` / `min_thinking` | Context facts (docs + sanity). |
| `presets` | Harness-agnostic **task modes**: `reason`, `code`, `agent`, `instruct`. Each: `thinking` (bool), optional `max_output`, a `sampling` block, optional raw `options`. An adapter maps these modes onto its own agents/subagents. |
| `variants` | Optional thinking-depth presets (raw `options` only). |

### Supported `sampling` params (generic)

`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
`repetition_penalty`, `max_output`. Non-standard body params (e.g.
`thinking_token_budget`, `chat_template_kwargs`) go under a preset's `options`.

## Contract with adapters

- **Match on `match`.** Filename == manager profile key so the two line up.
- Two launch profiles serving the **same** model (e.g. 256k vs 512k context)
  share **one** config here — sampling is per-model, not per-launch.
- A model with no matching config falls back to the adapter's default.
- Run `python3 -m unittest` after editing — the validation test parses every
  config and checks required keys.

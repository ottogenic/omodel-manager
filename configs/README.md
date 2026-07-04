# `configs/` — generic per-model configs (source of truth)

**Harness-agnostic** model configs: what a model can do (capabilities) and how to
sample it per task (presets). One **TOML** file per model, keyed to a launch
profile in `model_manager.json`. TOML so the files `cat`/`vi` cleanly and the
tuning guidance lives inline as `#` comments (see `qwen3.6-35b-a3b-nvfp4.toml`).

**omodel-manager owns and stores these.** It only holds + validates them
(`test_configs.py`, needs Python 3.11+ for `tomllib`). Downstream **adapters
consume them** and translate to a specific agent harness:

- `omodel-wire` → OpenCode (agents, permissions, colors, `chat.params` plugin)
- future adapters → pi.dev, Claude Code, …

Keep these files **generic** — no OpenCode/pi.dev-specific keys (agent names,
permissions, colors). Those belong in the adapter.

## One file per model

- `configs/<key>.toml`, named to match a `model_manager.json` profile/model key
  (`qwen3.6-35b-a3b-nvfp4.toml` ↔ the `qwen3.6-35b-a3b-nvfp4` launch profile).
- Adapters `tomllib.load` each file and match a discovered served-model-id against
  its `match` list. Non-`.toml` files (this README) are ignored.

## Schema

| Key | Meaning |
|-----|---------|
| `match` | Array of substrings matched (case-insensitive) against the served-model-id from `/v1/models`. Include the manager key **and** the HF/served id(s). The filename stem must be one of them. |
| `source` | Model-card URL the tuning came from (docs only). |
| `[capabilities]` `vision` | `false`, or a table `{ input = ["text","image"], output = ["text"] }`. **Replaces live vision probing.** |
| `[capabilities]` `reasoning` | `true`/`false` — does the model emit chain-of-thought. **Replaces reasoning probing.** |
| `[capabilities]` `tool_call` | `true`/`false`. |
| `[capabilities]` `concurrency` | Optional int mirroring the launch profile's `max-num-seqs` (parallel sequence slots). Adapters use it as the default cap on parallel workers (e.g. omodel-wire's team work-budget). Omit if unknown. |
| `[capabilities]` `thinking_control` | `"enable_thinking"` (Qwen `chat_template_kwargs`), `"reasoning_effort"` (vLLM low/med/high), `"soft_switch"` (`/think` `/no_think`), or `"none"` (template default; per-preset `options` carry knobs). |
| `[context]` `native` / `min_thinking` | Context facts (docs + sanity). |
| `[presets.<mode>]` | Harness-agnostic **task modes**: `reason`, `code`, `agent`, `instruct`. Each has `thinking` (bool), optional `max_output`, a `[presets.<mode>.sampling]` table, and optional `options` (e.g. `options.chat_template_kwargs`). Adapters map these modes onto their own agents/subagents. |
| `[variants.<name>]` | Optional thinking-depth presets (raw `options` only). |

### Supported `sampling` keys (generic)

`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
`repetition_penalty` (+ preset-level `max_output`). Non-standard body params (e.g.
`thinking_token_budget`, `chat_template_kwargs`) go under a preset's `options`.

## Contract with adapters

- **Match on `match`.** Filename stem == manager profile key so the two line up.
- Two launch profiles serving the **same** model (e.g. 256k vs 512k context) share
  **one** config here — sampling is per-model, not per-launch.
- A model with no matching config falls back to the adapter's default.
- Run `python3 -m unittest` after editing — the validator parses every config and
  checks required keys.

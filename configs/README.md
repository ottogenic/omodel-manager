# `configs/` — generic per-model configs (source of truth)

**Harness-agnostic** model configs: what a model can do (capabilities) and how to
sample it per task (presets). One **TOML** file per model, keyed to a launch
profile in `model_manager.json` and grouped by deployment kind. TOML keeps the
files readable and lets tuning guidance live inline as `#` comments.

**omodel-manager owns and stores these.** It only holds + validates them
(`test_configs.py`, needs Python 3.11+ for `tomllib`). Downstream **adapters
consume them** and translate to a specific agent harness:

- `omodel-wire` → OpenCode providers, native Build/Plan presets, and `chat.params` plugin
- future adapters → pi.dev, Claude Code, …

Keep these files **generic** — no OpenCode/pi.dev-specific keys (agent names,
permissions, colors). Those belong in the adapter.

## Deployment-kind directories

- `configs/node/<key>.toml`: deployments served by one compute node.
- `configs/cluster/<key>.toml`: deployments spanning multiple compute nodes.
- `configs/card/<key>.toml`: deployments qualified for a specific accelerator
  card target.
- Every TOML is exactly one level below `configs/`, in one of these known kind
  directories. Its filename stem appears in `match`; one TOML may cover multiple
  launch variants of the same base model.
- Every launch profile must resolve to exactly one TOML of its deployment kind by
  profile key, served-model id, or model id. Deployment records publish that exact
  configs-relative path for adapters.
- Adapters discover `configs/**/*.toml` recursively and match a served-model-id
  against each file's `match` list. Non-TOML files are ignored.

Directory placement describes the **deployment kind**, not the config schema.
TOML contents remain harness-agnostic and describe only model capabilities,
context, sampling, and model-native options. Deployment/runtime and harness keys
do not belong in these files.

## Schema

| Key | Meaning |
|-----|---------|
| `match` | Array of substrings matched (case-insensitive) against the served-model-id from `/v1/models`. Include the manager key **and** the HF/served id(s). The filename stem must be one of them. |
| `source` | Model-card URL the tuning came from (docs only). |
| `name` | Human-readable model label shown in harness pickers (OpenCode/Chamber). Display only — matching still uses `match`; an adapter falls back to the served-model-id when a config omits it. Keep it **bare** (family/size, plus a genuine model variant like Instruct/Thinking/Vision) — quant/build/serial details stay in the launch profile, not here. |
| `[capabilities]` `vision` | `false`, or a table `{ input = ["text","image"], output = ["text"] }`. **Replaces live vision probing.** |
| `[capabilities]` `reasoning` | `true`/`false` — does the model emit chain-of-thought. **Replaces reasoning probing.** |
| `[capabilities]` `tool_call` | `true`/`false`. |
| `[capabilities]` `concurrency` | Optional int mirroring the launch profile's `max-num-seqs` (parallel sequence slots). Adapters use it as the default cap on parallel workers (e.g. omodel-wire's team work-budget). Omit if unknown. |
| `[capabilities]` `thinking_control` | `"enable_thinking"` (Qwen `chat_template_kwargs`), `"reasoning_effort"` (vLLM low/med/high), `"soft_switch"` (`/think` `/no_think`), or `"none"` (template default; per-preset `options` carry knobs). |
| `[context]` `native` / `min_thinking` | Context facts (docs + sanity). |
| `[presets.<mode>]` | Two harness-agnostic task modes: `plan` and `build`. Each has `thinking` (bool), optional `max_output`, a `[presets.<mode>.sampling]` table, and optional model-native `options` (e.g. `options.chat_template_kwargs`). Adapters map them onto their native planning and implementation modes. |
| `[variants.<name>]` | Optional thinking-depth presets (raw `options` only). |

### Supported `sampling` keys (generic)

`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
`repetition_penalty` (+ preset-level `max_output`). Non-standard body params (e.g.
`thinking_token_budget`, `chat_template_kwargs`) go under a preset's `options`.

## Contract with adapters

- **Match on `match`.** Filename stem == manager profile key so the two line up;
  directory kind does not participate in matching.
- **Display via `name`.** Adapters label a model with its `name` (the bare model
  label in the schema above), never the raw served-model-id. A config without a
  `name` falls back to the served-model-id; every shipped config sets one.
- Two launch profiles serving the **same** model (e.g. 256k vs 512k context) share
  **one** config here — sampling is per-model, not per-launch.
- A model with no matching config falls back to the adapter's default.
- Run `python3 -m unittest test_configs -v` after editing. The validator parses
  every config recursively and checks placement and required keys.

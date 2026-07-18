# DeepSeek V4 Q0 fixtures

Copied from the ds4 reference repository's `tests/test-vectors` directory so the
Q0 renderer/tokenizer gate is reproducible from this repository alone.

Included for Q0:

- `official.vec`: compact official-vector fixture with prompt paths and selected/top
  token bytes.
- `prompts/*.txt`: the five prompt files referenced by `official.vec`.

Included for Q1:

- `manifest.json`: the five-prompt test-vector manifest.
- `official/*.official.json`: official DeepSeek-V4-Flash greedy, thinking-off
  top-20 records for those prompts.

The official JSON records expose selected-token agreement and top-20 candidate
sets. They do not expose calibrated non-selected logprobs: the copied records use
`-9999.0` sentinels for non-selected alternatives, so Q1 must not treat those as
usable probability deltas.

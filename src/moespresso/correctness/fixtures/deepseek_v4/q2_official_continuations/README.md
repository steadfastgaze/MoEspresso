# DeepSeek-V4 Q2 prompts

`prompts.jsonl` is tracked because it is byte-identical to the public DS4
quality-testing prompt set.

The official continuation/top-logprob capture is intentionally not tracked.
Create or provide it at:

`src/moespresso/correctness/fixtures/deepseek_v4/private/q2_official_continuations/official_continuations.json`

Use `moespresso-ds4-quality q2-capture` to refresh that local ignored file, or
pass `--reference` to `moespresso-ds4-quality q2`.

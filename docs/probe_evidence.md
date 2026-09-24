# Calibration and probe evidence

Calibration readers supply measured activation statistics and their identity
to model-specific package builders. They do not select rule-based quantization
recipes.

## Providers

- `probe/calibration.py` reads GGUF and legacy imatrix vectors, records their
  identity, and exposes expert routing counts for cold-start hotlists.
- `probe/qwen4/calibration.py` validates routed-expert and dense-tensor teacher
  captures, checking source identity, tensor coverage, sample counts and
  calibration policy before Qwen4 conversion.
- `probe/deepseek_v4/` contains source codecs, expert adapters and evidence
  helpers for calibrated DeepSeek workflows.
- `probe/quality.py` and `probe/weight_io.py` provide shared quality math and
  bounded source reads.

`probe_evidence` and `optimizer_decision` artifacts carry provenance for
calibrated workflows.

The Qwen4 converter applies a validated policy to experts with zero observed
routes. Whole-model conversion still requires calibration.

See [source inventory](source_inventory.md),
[allocation artifacts](optimizer_decision.md) and
[package format](package_format.md).

# Allocation decision artifacts

An `optimizer_decision` records an allocation from a calibrated conversion or
allocation tool. DeepSeek and Qwen IQ_K workflows use this artifact contract.
The runtime does not run an optimizer.

Package builders consume explicit GGUF recipes or validated converted-expert
artifacts with their allocation decisions.

## Package-plan boundary

`package.plan.package_plan_from_decision` transfers the decision and probe
identities, required features, constraints and per-tensor allocation into a
shared `package_plan`. The writer follows that plan's codec choices.

Before encoding, Qwen4's IQ_K converter validates the decision against the
source inventory and teacher calibration identity. Its package builder checks
the converted cells, per-expert calibration policy and exact codec geometry.
DeepSeek's IQ_K builder consumes each routed cell's declared member and encoded
bytes. GGUF recipe builders record recipe provenance without creating
optimizer evidence.

Package-plan force overrides apply at build time. Unknown formats and unmatched
patterns fail closed. A dry-run preview reports each matched tensor.

See [artifact contracts](artifact_contract.md),
[package format](package_format.md) and
[DeepSeek package construction](deepseek_v4_package_recipe.md).

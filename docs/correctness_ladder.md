# The correctness ladder

`src/moespresso/correctness/` contains four generic package checks and the
model-specific quality gates. The generic checks inspect ownership, storage,
reconstruction, and small numerical primitives without loading a complete model.

Plausible output cannot establish numerical correctness. These checks compare
stored bytes and declared transforms with an independent or explicitly classified
reference, then record the result in a `correctness_evidence` artifact. Any blocking
finding makes the artifact invalid.

## Reference strength

Each finding names the reference used:

- `independent`: project-owned reference math that avoids the production path under
  test.
- `external_codec`: the library that owns the format, such as MLX dequantization for
  affine tensors.
- `shared_code`: a production helper reused as a smoke check.

Shared code cannot provide the sole evidence for its own correctness.

## Implemented checks

### L0: static contract

`l0_static_contract` compares the architecture profile, source inventory, and
package manifest without reading tensor data. It requires every source tensor to be
owned or explicitly excluded, every owned tensor to appear in the package, and every
manifest format to match the declared role. Unknown package tensors, carried
exclusions, missing tensors, and format mismatches are blocking. Fused gate/up
entries are checked separately, so one valid half cannot hide the other.

### L0b: header storage contract

`l0b_norm_shift_contract` reads safetensors headers and checks the coupled conv1d and
RMSNorm rule. Norms remain unshifted on disk; the runtime adds the required shift
when the stored conv1d shape activates the sanitizer. L0b verifies that shape and
requires the conv1d tensor when the family profile declares it mandatory.

### L1: tensor reconstruction

`l1_tensor_reconstruction` samples source and package tensors deterministically,
reconstructs the stored representation, and compares it with the source. Structural
passthrough tensors use an independent reference, while affine tensors use the
external MLX codec. Sampling favors high-risk roles such as embeddings, the
vocabulary head, input projections, conv1d, norms, and fused gate/up weights. A
declared format with no successful sample is a blocking result.

### L2: micro-goldens

`l2_micro_goldens` runs independent examples for the fused gate/up split, the conv1d
shape that controls the norm shift, and affine sidecar shapes. These small checks
pin the primitives used by L1.

## Running the checks

The four checks are library functions exercised directly by the correctness tests:

- `correctness/ladder.py`: L0, L0b, and evidence construction.
- `correctness/reconstruct.py`: L1.
- `correctness/goldens.py`: L2.

Package builders do not run this sequence automatically. Release acceptance also
requires the relevant model-family gate and a served-path check.

## Model-family gates

- **DeepSeek-V4:** `moespresso-ds4-quality` covers Q0 renderer and tokenizer
  goldens, Q1 selected-token identity, Q2 teacher-forced NLL, Q3 long-context
  recall. `moespresso-ds4-q4` runs the KL panels, while
  `moespresso-ds4-wikitext-ppl` supplies the separate perplexity gate.
- **Ornith:** `moespresso-ornith-gate` publicly covers sandboxed agentic coding and
  exact-scored long-context tasks. Private fixtures add the hard-reasoning gate.
- **Qwen 35B:** `moespresso-qwen35-hard-questions` runs the exact-answer package
  comparison.

`correctness/environment.py` records the MLX wheel tag with measured evidence because
wheel variants can move low-margin logits even when the package is unchanged.

//! One-token QSA attention over already selected BF16 physical rows.
//!
//! The caller owns row selection and KV storage. Ordinary device execution
//! reconstructs selected rows inside FP32 online attention,
//! gates once and projects to hidden width. The BF16 diagnostic path remains
//! available for numerical stage comparisons.

use anyhow::{Result, ensure};
use half::bf16;

pub const QUERY_HEADS: usize = 24;
pub const KV_HEADS: usize = 2;
pub const HEAD_WIDTH: usize = 256;
pub const HIDDEN: usize = 2560;
pub const SELECTED_WIDTH: usize = 2051;

/// Validate the fixed-width selected-row input before GPU publication.
pub fn validate_selected_rows(
    queries_bf16: &[u16],
    gate_bf16: &[u16],
    keys_bf16: &[u16],
    values_bf16: &[u16],
    valid: &[u8],
) -> Result<()> {
    ensure!(
        queries_bf16.len() == QUERY_HEADS * HEAD_WIDTH,
        "QSA query shape differs from 24x256"
    );
    ensure!(
        gate_bf16.len() == QUERY_HEADS * HEAD_WIDTH,
        "QSA gate shape differs from 6144"
    );
    ensure!(
        keys_bf16.len() == KV_HEADS * SELECTED_WIDTH * HEAD_WIDTH,
        "QSA selected K shape differs from 2x2051x256"
    );
    ensure!(
        values_bf16.len() == keys_bf16.len(),
        "QSA selected V shape differs from K"
    );
    ensure!(
        valid.len() == SELECTED_WIDTH,
        "QSA selected validity shape differs from 2051"
    );
    ensure!(
        valid.iter().all(|&v| v <= 1),
        "QSA validity must contain only zero or one"
    );
    ensure!(
        valid.contains(&1),
        "QSA one-token selection must contain a visible row"
    );
    ensure!(
        valid.windows(2).all(|v| v != [0, 1]),
        "QSA valid rows must precede padding"
    );
    // Scan every word with a non-short-circuit reduction to enable vectorization.
    ensure!(
        queries_bf16
            .iter()
            .chain(gate_bf16)
            .chain(keys_bf16)
            .chain(values_bf16)
            .fold(true, |ok, &word| ok & bf16::from_bits(word).is_finite()),
        "QSA selected inputs must be finite BF16"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use anyhow::{Context, Result, ensure};

    use crate::kquant::{DenseMatvec, OutputPrecision};
    use crate::kvarn::DeviceKvarnState;
    #[cfg(test)]
    use crate::kvarn_pending::PendingKvarnView;
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use crate::package::{Package, WeightFormat};

    const MSL: &str = include_str!("../kernels/metal/qsa_attention.metal");
    const ONLINE_MSL: &str = include_str!("../kernels/metal/qsa_online.metal");
    const MAX_SPLITS: usize = 64;
    const PART_STRIDE: usize = 2 + HEAD_WIDTH;

    fn words_bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_words(buffer: &Buffer, words: usize) -> Result<Vec<u16>> {
        let mut bytes = vec![0; words * 2];
        buffer.read_bytes(0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect())
    }

    struct Scratch {
        queries: Buffer,
        gate: Buffer,
        keys: Buffer,
        values: Buffer,
        valid: Buffer,
        rows: Buffer,
        logits: Buffer,
        scaled: Buffer,
        masked: Buffer,
        probabilities: Buffer,
        attention: Buffer,
        sigmoid: Buffer,
        gated: Buffer,
        online_partials: Buffer,
        online_count: Buffer,
        output: Buffer,
    }

    pub struct QsaAttentionStages {
        pub logits_bf16: Vec<u16>,
        pub scaled_logits_bf16: Vec<u16>,
        pub probabilities_bf16: Vec<u16>,
        pub attention_bf16: Vec<u16>,
        pub gate_sigmoid_bf16: Vec<u16>,
        pub gated_bf16: Vec<u16>,
        pub output_bf16: Vec<u16>,
    }

    pub struct QsaAttention {
        metal: Metal,
        output_projection: DenseMatvec,
        logits_pipeline: Pipeline,
        softmax_pipeline: Pipeline,
        value_gate_pipeline: Pipeline,
        kvarn_online_tile_pipeline: Pipeline,
        selection_guard_pipeline: Pipeline,
        online_merge_pipeline: Pipeline,
        scratch: Scratch,
    }

    impl QsaAttention {
        pub(crate) fn shared_matrix(&self) -> crate::kquant::DeviceMatrix {
            self.output_projection.shared_matrix()
        }
        pub fn new(metal: &Metal, package: &Package, layer: usize) -> Result<Self> {
            ensure!(
                layer < 48 && layer % 4 == 3,
                "released QSA uses layers 3, 7, ..., 47"
            );
            let config = &package.architecture()["config"];
            ensure!(
                config["indexer_budget"].as_u64() == Some(2048)
                    && config["indexer_compress_ratio"].as_u64() == Some(4)
                    && config["head_dim"].as_u64() == Some(HEAD_WIDTH as u64),
                "QSA package selected-row geometry differs from 2051x256"
            );
            let path = format!("layers.{layer}.mixer.module.o_proj");
            let weight = package.direct_weight(&path)?;
            ensure!(
                weight.format == WeightFormat::Q6K
                    && weight.shape == [HIDDEN, QUERY_HEADS * HEAD_WIDTH],
                "{path} differs from the released Q6_K QSA output projection"
            );
            let output_projection = DenseMatvec::new(metal, &weight)?;
            Self::from_projection(metal, output_projection)
        }

        fn from_projection(metal: &Metal, output_projection: DenseMatvec) -> Result<Self> {
            let logits_pipeline = metal.compile_msl_strict(MSL, "qsa_selected_logits")?;
            let softmax_pipeline = metal.compile_msl_strict(MSL, "qsa_selected_softmax")?;
            let value_gate_pipeline = metal.compile_msl_strict(MSL, "qsa_selected_value_gate")?;
            let kvarn_online_tile_pipeline =
                metal.compile_msl_strict(ONLINE_MSL, "qsa_kvarn_online_tile")?;
            let selection_guard_pipeline = metal.compile_msl_strict(
                include_str!("../kernels/metal/kvarn_pending.metal"),
                "kvarn_pending_selection_guard",
            )?;
            let online_merge_pipeline = metal.compile_msl_strict(ONLINE_MSL, "qsa_online_merge")?;
            ensure!(
                softmax_pipeline.max_threads_per_threadgroup() >= 544
                    && kvarn_online_tile_pipeline.max_threads_per_threadgroup() >= 128
                    && online_merge_pipeline.max_threads_per_threadgroup() >= 32,
                "QSA attention threadgroup geometry is unsupported"
            );
            let scratch = Scratch {
                queries: metal.shared_buffer(QUERY_HEADS * HEAD_WIDTH * 2)?,
                gate: metal.shared_buffer(QUERY_HEADS * HEAD_WIDTH * 2)?,
                keys: metal.shared_buffer(KV_HEADS * SELECTED_WIDTH * HEAD_WIDTH * 2)?,
                values: metal.shared_buffer(KV_HEADS * SELECTED_WIDTH * HEAD_WIDTH * 2)?,
                valid: metal.shared_buffer(SELECTED_WIDTH)?,
                rows: metal.shared_buffer(8)?,
                logits: metal.shared_buffer(QUERY_HEADS * SELECTED_WIDTH * 2)?,
                scaled: metal.shared_buffer(QUERY_HEADS * SELECTED_WIDTH * 2)?,
                masked: metal.shared_buffer(QUERY_HEADS * SELECTED_WIDTH * 4)?,
                probabilities: metal.shared_buffer(QUERY_HEADS * SELECTED_WIDTH * 2)?,
                attention: metal.shared_buffer(QUERY_HEADS * HEAD_WIDTH * 2)?,
                sigmoid: metal.shared_buffer(QUERY_HEADS * HEAD_WIDTH * 2)?,
                gated: metal.shared_buffer(QUERY_HEADS * HEAD_WIDTH * 2)?,
                online_partials: metal.shared_buffer(QUERY_HEADS * MAX_SPLITS * PART_STRIDE * 4)?,
                online_count: metal.shared_buffer(4)?,
                output: metal.shared_buffer(HIDDEN * 2)?,
            };
            scratch.rows.write_bytes(
                0,
                &[SELECTED_WIDTH as u32, 1]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            Ok(Self {
                metal: metal.clone(),
                output_projection,
                logits_pipeline,
                softmax_pipeline,
                value_gate_pipeline,
                kvarn_online_tile_pipeline,
                selection_guard_pipeline,
                online_merge_pipeline,
                scratch,
            })
        }

        pub fn run(
            &mut self,
            queries_bf16: &[u16],
            gate_bf16: &[u16],
            keys_bf16: &[u16],
            values_bf16: &[u16],
            valid: &[u8],
        ) -> Result<Vec<u16>> {
            Ok(self
                .execute(
                    queries_bf16,
                    gate_bf16,
                    keys_bf16,
                    values_bf16,
                    valid,
                    false,
                )?
                .0)
        }

        pub fn run_stages(
            &mut self,
            queries_bf16: &[u16],
            gate_bf16: &[u16],
            keys_bf16: &[u16],
            values_bf16: &[u16],
            valid: &[u8],
        ) -> Result<QsaAttentionStages> {
            self.execute(queries_bf16, gate_bf16, keys_bf16, values_bf16, valid, true)?
                .1
                .context("QSA attention stages were not requested")
        }

        /// Attend a committed device cache plus its exact pending row, without
        /// publishing that row or reading selected K/V back for another upload.
        #[cfg(test)]
        pub(crate) fn run_pending(
            &mut self,
            state: &DeviceKvarnState,
            selected: &[i32],
            queries: &[u16],
            gate: &[u16],
            pending_keys: &[u8],
            pending_values: &[u8],
        ) -> Result<Vec<u16>> {
            ensure!(
                queries.len() == QUERY_HEADS * HEAD_WIDTH && gate.len() == queries.len(),
                "QSA query/gate shape differs"
            );
            ensure!(
                queries
                    .iter()
                    .chain(gate)
                    .all(|&v| bf16::from_bits(v).is_finite()),
                "QSA selected inputs must be finite BF16"
            );
            ensure!(
                pending_keys.len() == 1024 && pending_values.len() == 1024,
                "KVarN pending row must contain two BF16 heads"
            );
            for bytes in [pending_keys, pending_values] {
                ensure!(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|v| bf16::from_bits(u16::from_le_bytes(*v)).is_finite()),
                    "KVarN pending row contains nonfinite BF16"
                );
            }
            ensure!(
                selected.len() == SELECTED_WIDTH,
                "QSA selected width differs"
            );
            self.scratch.queries.write_bytes(0, &words_bytes(queries))?;
            self.scratch.gate.write_bytes(0, &words_bytes(gate))?;
            let keys = self.metal.shared_buffer(1024)?;
            let values = self.metal.shared_buffer(1024)?;
            keys.write_bytes(0, pending_keys)?;
            values.write_bytes(0, pending_values)?;
            // SAFETY: A singleton has identical token/head-major layout. Host
            // inputs were validated above and the view retains all its readers.
            let pending =
                unsafe { PendingKvarnView::from_device(&self.metal, state, &keys, &values, 1)? };
            let mut batch = self.metal.batch()?;
            let rows = pending.encode_rows(&mut batch, selected, 0)?;
            self.encode_attention(&mut batch, &rows.keys, &rows.values, &rows.valid)?;
            batch
                .submit()
                .wait()
                .context("QSA pending selected attention command")?;
            // Retain selected-input validation after completion and before
            // output or append acceptance, without a second selected-row upload.
            let mut scratch = [0u8; 16_384];
            for buffer in [&rows.keys, &rows.values] {
                for offset in (0..buffer.byte_len()).step_by(scratch.len()) {
                    let len = scratch.len().min(buffer.byte_len() - offset);
                    buffer.read_bytes(offset, &mut scratch[..len])?;
                    ensure!(
                        scratch[..len]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .fold(true, |ok, v| ok
                                & bf16::from_bits(u16::from_le_bytes(*v)).is_finite()),
                        "QSA selected inputs must be finite BF16"
                    );
                }
            }
            let output = read_words(&self.scratch.output, HIDDEN)?;
            ensure!(
                output.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "QSA output must be finite BF16"
            );
            Ok(output)
        }

        fn execute(
            &mut self,
            queries_bf16: &[u16],
            gate_bf16: &[u16],
            keys_bf16: &[u16],
            values_bf16: &[u16],
            valid: &[u8],
            capture: bool,
        ) -> Result<(Vec<u16>, Option<QsaAttentionStages>)> {
            validate_selected_rows(queries_bf16, gate_bf16, keys_bf16, values_bf16, valid)?;
            self.scratch
                .queries
                .write_bytes(0, &words_bytes(queries_bf16))?;
            self.scratch.gate.write_bytes(0, &words_bytes(gate_bf16))?;
            self.scratch.keys.write_bytes(0, &words_bytes(keys_bf16))?;
            self.scratch
                .values
                .write_bytes(0, &words_bytes(values_bf16))?;
            self.scratch.valid.write_bytes(0, valid)?;
            let mut batch: Batch = self.metal.batch()?;
            self.encode_attention(
                &mut batch,
                &self.scratch.keys,
                &self.scratch.values,
                &self.scratch.valid,
            )?;
            batch
                .submit()
                .wait()
                .context("QSA selected attention command")?;
            let output = read_words(&self.scratch.output, HIDDEN)?;
            if !capture {
                return Ok((output, None));
            }
            let stages = QsaAttentionStages {
                logits_bf16: read_words(&self.scratch.logits, QUERY_HEADS * SELECTED_WIDTH)?,
                scaled_logits_bf16: read_words(&self.scratch.scaled, QUERY_HEADS * SELECTED_WIDTH)?,
                probabilities_bf16: read_words(
                    &self.scratch.probabilities,
                    QUERY_HEADS * SELECTED_WIDTH,
                )?,
                attention_bf16: read_words(&self.scratch.attention, QUERY_HEADS * HEAD_WIDTH)?,
                gate_sigmoid_bf16: read_words(&self.scratch.sigmoid, QUERY_HEADS * HEAD_WIDTH)?,
                gated_bf16: read_words(&self.scratch.gated, QUERY_HEADS * HEAD_WIDTH)?,
                output_bf16: output,
            };
            Ok((stages.output_bf16.clone(), Some(stages)))
        }

        fn encode_attention(
            &self,
            batch: &mut Batch,
            keys: &Buffer,
            values: &Buffer,
            valid: &Buffer,
        ) -> Result<()> {
            self.encode_attention_device(
                batch,
                &self.scratch.queries,
                &self.scratch.gate,
                keys,
                values,
                valid,
            )
        }

        fn encode_attention_device(
            &self,
            batch: &mut Batch,
            queries: &Buffer,
            gate: &Buffer,
            keys: &Buffer,
            values: &Buffer,
            valid: &Buffer,
        ) -> Result<()> {
            ensure!(
                keys.byte_len() == KV_HEADS * SELECTED_WIDTH * HEAD_WIDTH * 2
                    && values.byte_len() == keys.byte_len()
                    && valid.byte_len() == SELECTED_WIDTH
                    && queries.byte_len() == QUERY_HEADS * HEAD_WIDTH * 2
                    && gate.byte_len() == queries.byte_len(),
                "QSA selected buffer extent differs"
            );
            // SAFETY: Every stage uses the fixed, validated QSA geometry and
            // disjoint scratch outputs. Nonfinite values cannot affect addresses;
            // the caller rejects them before accepting the completed output.
            unsafe {
                let logits = [
                    queries,
                    keys,
                    valid,
                    &self.scratch.logits,
                    &self.scratch.scaled,
                    &self.scratch.masked,
                    &self.scratch.rows,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(
                    &self.logits_pipeline,
                    &logits,
                    QUERY_HEADS * SELECTED_WIDTH,
                    128,
                )?;
                let softmax = [
                    &self.scratch.masked,
                    &self.scratch.probabilities,
                    &self.scratch.rows,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.softmax_pipeline, &softmax, QUERY_HEADS * 544, 544)?;
                let value_gate = [
                    &self.scratch.probabilities,
                    values,
                    gate,
                    &self.scratch.attention,
                    &self.scratch.sigmoid,
                    &self.scratch.gated,
                    &self.scratch.rows,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(
                    &self.value_gate_pipeline,
                    &value_gate,
                    QUERY_HEADS * HEAD_WIDTH,
                    128,
                )?;
                self.output_projection.encode(
                    batch,
                    &self.scratch.gated,
                    &self.scratch.output,
                    OutputPrecision::Bf16,
                )?;
            }
            Ok(())
        }

        /// Encode selected attention without materializing query, KV or output
        /// rows on the host. The stream owns all producer/consumer ordering.
        #[allow(clippy::too_many_arguments)]
        #[cfg(test)]
        pub(crate) fn encode_pending_device(
            &self,
            batch: &mut Batch,
            state: &DeviceKvarnState,
            selected: &[i32],
            queries: &Buffer,
            gate: &Buffer,
            pending_keys: &Buffer,
            pending_values: &Buffer,
            status: &crate::device_status::DeviceStatus,
        ) -> Result<Buffer> {
            // SAFETY: Fixed token-major rows are produced earlier in the same
            // ordered stream. The caller enqueues append after these readers.
            let pending = unsafe {
                PendingKvarnView::from_device(&self.metal, state, pending_keys, pending_values, 1)?
            };
            pending.encode_rows_into(
                batch,
                selected,
                0,
                &self.scratch.keys,
                &self.scratch.values,
                &self.scratch.valid,
            )?;
            self.encode_attention_device(
                batch,
                queries,
                gate,
                &self.scratch.keys,
                &self.scratch.values,
                &self.scratch.valid,
            )?;
            status.encode_bf16(batch, &self.scratch.output)?;
            Ok(self.scratch.output.clone())
        }

        /// Decode the GPU-selected KVarN row with split FP32 online softmax.
        /// Only the final gated vector rounds to BF16 before Q6 projection.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn encode_pending_online_device(
            &self,
            batch: &mut Batch,
            state: &DeviceKvarnState,
            selected_ids: &Buffer,
            selected_valid: &Buffer,
            selected_count: &Buffer,
            selection_status: &Buffer,
            queries: &Buffer,
            gate: &Buffer,
            pending_keys: &Buffer,
            pending_values: &Buffer,
            status: &crate::device_status::DeviceStatus,
        ) -> Result<Buffer> {
            ensure!(
                queries.byte_len() == QUERY_HEADS * HEAD_WIDTH * 2
                    && gate.byte_len() == queries.byte_len()
                    && pending_keys.byte_len() >= 1024
                    && pending_values.byte_len() >= 1024
                    && selected_ids.byte_len() == SELECTED_WIDTH * 4
                    && selected_valid.byte_len() == SELECTED_WIDTH
                    && selected_count.byte_len() == 4
                    && selection_status.byte_len() == 4,
                "QSA online input extent differs"
            );
            let frontier = state.frontier()?;
            let visible = frontier
                .checked_add(1)
                .context("QSA pending frontier overflow")?;
            ensure!(
                visible <= state.context_limit(),
                "QSA pending row exceeds context"
            );
            let expected_count = if visible <= SELECTED_WIDTH {
                visible
            } else {
                2048 + visible % 4
            };
            let splits = expected_count.div_ceil(32).clamp(1, MAX_SPLITS);
            let geometry = self.metal.shared_buffer(20)?;
            geometry.write_bytes(
                0,
                &[
                    u32::try_from(frontier)?,
                    u32::try_from(state.record_count())?,
                    u32::try_from(frontier)?,
                    1,
                    u32::try_from(expected_count)?,
                ]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            )?;
            let committed = state.buffers()?;
            // SAFETY: The guard validates all IDs before direct storage reads.
            // Committed and pending allocations remain immutable on the stream
            // until these readers finish; append follows attention in the caller.
            unsafe {
                let guard = [
                    (selected_ids, 0),
                    (selected_valid, 0),
                    (selected_count, 0),
                    (&geometry, 8),
                    (&self.scratch.valid, 0),
                    (&self.scratch.online_count, 0),
                    (selection_status, 0),
                ]
                .map(|(buffer, offset)| Binding { buffer, offset });
                batch.dispatch_1d(&self.selection_guard_pipeline, &guard, SELECTED_WIDTH, 128)?;
                let tile = [
                    queries,
                    committed[0].0,
                    committed[1].0,
                    committed[2].0,
                    committed[3].0,
                    committed[4].0,
                    pending_keys,
                    pending_values,
                    selected_ids,
                    &self.scratch.valid,
                    &self.scratch.online_count,
                    &geometry,
                    &self.scratch.online_partials,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(
                    &self.kvarn_online_tile_pipeline,
                    &tile,
                    splits * KV_HEADS * 128,
                    128,
                )?;
                let merge = [
                    &self.scratch.online_partials,
                    gate,
                    &self.scratch.online_count,
                    &self.scratch.gated,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.online_merge_pipeline, &merge, QUERY_HEADS * 32, 32)?;
                self.output_projection.encode(
                    batch,
                    &self.scratch.gated,
                    &self.scratch.output,
                    OutputPrecision::Bf16,
                )?;
            }
            status.encode_bf16(batch, &self.scratch.output)?;
            Ok(self.scratch.output.clone())
        }
    }

    #[cfg(test)]
    mod device_tests {
        use super::*;
        use crate::kvarn::{KvarnDecoder, KvarnState, SINK_BYTES, TAIL_BYTES, TILE_RECORD_BYTES};
        use crate::package::DirectWeight;

        fn fp64_attention_reference(
            queries: &[u16],
            gate: &[u16],
            keys: &[u16],
            values: &[u16],
            valid: &[u8],
            count: usize,
        ) -> Vec<f64> {
            let decode = |word: u16| bf16::from_bits(word).to_f32() as f64;
            let mut output = vec![0.0f64; QUERY_HEADS * HEAD_WIDTH];
            for head in 0..QUERY_HEADS {
                let kv_head = head / 12;
                let mut scores = Vec::new();
                for row in 0..count {
                    if valid[row] == 0 {
                        continue;
                    }
                    let mut score = 0.0f64;
                    for d in 0..HEAD_WIDTH {
                        score += decode(queries[head * HEAD_WIDTH + d])
                            * decode(keys[(kv_head * SELECTED_WIDTH + row) * HEAD_WIDTH + d]);
                    }
                    scores.push((row, score / 16.0));
                }
                let maximum = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<(usize, f64)> = scores
                    .iter()
                    .map(|&(row, score)| (row, (score - maximum).exp()))
                    .collect();
                let total: f64 = weights.iter().map(|(_, weight)| *weight).sum();
                for d in 0..HEAD_WIDTH {
                    let mut attended = 0.0f64;
                    for &(row, weight) in &weights {
                        attended += weight
                            * decode(values[(kv_head * SELECTED_WIDTH + row) * HEAD_WIDTH + d]);
                    }
                    let g = decode(gate[head * HEAD_WIDTH + d]);
                    let logistic = if g >= 0.0 {
                        1.0 / (1.0 + (-g).exp())
                    } else {
                        let e = g.exp();
                        e / (1.0 + e)
                    };
                    output[head * HEAD_WIDTH + d] = attended / total * logistic;
                }
            }
            output
        }

        fn assert_online_error(actual: &[u16], reference: &[f64], count: usize, values: &[u16]) {
            assert_eq!(actual.len(), reference.len());
            let value_scale = values
                .iter()
                .map(|&word| bf16::from_bits(word).to_f32().abs() as f64)
                .fold(0.0f64, f64::max);
            let accumulation_floor = f32::EPSILON as f64 * count as f64 * value_scale;
            let mut maximum = 0.0f64;
            let mut squared = 0.0f64;
            let mut max_ratio = 0.0f64;
            let mut mismatched_words = 0usize;
            for (&word, &expected) in actual.iter().zip(reference) {
                let observed = bf16::from_bits(word).to_f32() as f64;
                let error = (observed - expected).abs();
                let rounded = bf16::from_f32(expected.abs() as f32).to_bits() & 0x7fff;
                let current = bf16::from_bits(rounded).to_f32() as f64;
                let next = bf16::from_bits(rounded + 1).to_f32() as f64;
                let ulp = next - current;
                // One BF16 ULP permits a rounding-boundary flip; the FP32
                // accumulation allowance scales with the number of keys.
                let allowance = ulp + accumulation_floor;
                maximum = maximum.max(error);
                squared += error * error;
                max_ratio = max_ratio.max(error / allowance);
                mismatched_words += usize::from(word != bf16::from_f32(expected as f32).to_bits());
            }
            let rms = (squared / actual.len() as f64).sqrt();
            eprintln!(
                "online attention count={count}: max_abs={maximum:.8}, rms={rms:.8}, max_allowance_ratio={max_ratio:.4}, mismatched_bf16_words={mismatched_words}"
            );
            assert!(
                max_ratio <= 1.0,
                "online attention exceeds BF16 ULP plus FP32 accumulation allowance"
            );
        }

        #[test]
        fn online_attention_matches_fp64_on_reconstructed_kvarn_and_guards_ids() {
            let metal = Metal::new().unwrap();
            let data = |bytes: usize, salt: usize| {
                (0..bytes / 2)
                    .flat_map(|i| {
                        let head = i / (bytes / 4);
                        bf16::from_f32(
                            ((i * 37 + salt + head * 53) % 127) as f32 / 512. - 0.125
                                + head as f32 * 0.03125,
                        )
                        .to_bits()
                        .to_le_bytes()
                    })
                    .collect::<Vec<_>>()
            };
            let sink_keys = data(SINK_BYTES, 1);
            let sink_values = data(SINK_BYTES, 17);
            let tail_keys = data(TAIL_BYTES, 9);
            let tail_values = data(TAIL_BYTES, 43);
            for frontier in [63usize, 2047, 8448, 16899] {
                let record_count = crate::kvarn::partition(frontier).1.saturating_sub(128) / 128;
                let mut records = vec![0; record_count * TILE_RECORD_BYTES];
                for (record_index, record) in records
                    .as_chunks_mut::<{ crate::kvarn::HEAD_RECORD_BYTES }>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    for (i, byte) in record[..16384].iter_mut().enumerate() {
                        *byte = (i.wrapping_mul(37).wrapping_add(record_index * 29) % 256) as u8;
                    }
                    for (i, byte) in record[17664..34048].iter_mut().enumerate() {
                        *byte =
                            (i.wrapping_mul(71).wrapping_add(record_index * 47 + 19) % 256) as u8;
                    }
                    for (range, value) in [
                        (16384..16896, 0.015625),
                        (16896..17408, -0.125),
                        (17408..17664, 0.25),
                        (34048..34560, 0.03125),
                        (34560..34816, 0.125),
                        (34816..35072, -0.875),
                    ] {
                        for (index, pair) in
                            record[range].as_chunks_mut::<2>().0.iter_mut().enumerate()
                        {
                            let value =
                                value * (1.0 + ((record_index * 3 + index) % 17) as f32 / 32.0);
                            pair.copy_from_slice(
                                &half::f16::from_f32(value).to_bits().to_le_bytes(),
                            );
                        }
                    }
                }
                let host = KvarnState::restore(
                    frontier + 1,
                    frontier,
                    &records,
                    &sink_keys,
                    &sink_values,
                    &tail_keys,
                    &tail_values,
                )
                .unwrap();
                let device = DeviceKvarnState::restore(
                    &metal,
                    frontier + 1,
                    frontier,
                    host.records(),
                    &sink_keys,
                    &sink_values,
                    &tail_keys,
                    &tail_values,
                )
                .unwrap();
                let pending_keys = data(1024, 3);
                let pending_values = data(1024, 61);
                let mut selected = vec![-1i32; SELECTED_WIDTH];
                let count = if frontier < SELECTED_WIDTH {
                    frontier + 1
                } else {
                    2048 + (frontier + 1) % 4
                };
                let mut active = (0..count)
                    .map(|i| (i * frontier / (count - 1)) as i32)
                    .collect::<Vec<_>>();
                active.reverse();
                selected[..active.len()].copy_from_slice(&active);
                let selected_rows = KvarnDecoder::new(&metal)
                    .unwrap()
                    .gather_pending_row(&host, &selected, &pending_keys, &pending_values)
                    .unwrap();
                let words = |bytes: &[u8]| {
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|pair| u16::from_le_bytes(*pair))
                        .collect::<Vec<_>>()
                };
                let host_keys = words(&selected_rows.keys_bf16);
                let host_values = words(&selected_rows.values_bf16);
                let valid = selected_rows
                    .valid
                    .iter()
                    .map(|&v| u8::from(v))
                    .collect::<Vec<_>>();
                let mut queries = (0..QUERY_HEADS * HEAD_WIDTH)
                    .map(|d| bf16::from_f32((d % 97) as f32 / 256. - 0.1875).to_bits())
                    .collect::<Vec<_>>();
                queries[..HEAD_WIDTH].fill(0); // Exact tied logits for head zero.
                queries[HEAD_WIDTH..2 * HEAD_WIDTH].fill(bf16::from_f32(16.0).to_bits());
                let gate = (0..queries.len())
                    .map(|d| bf16::from_f32((d % 37) as f32 / 4. - 4.5).to_bits())
                    .collect::<Vec<_>>();
                let reference = fp64_attention_reference(
                    &queries,
                    &gate,
                    &host_keys,
                    &host_values,
                    &valid,
                    active.len(),
                );
                let upload = |bytes: &[u8]| {
                    let buffer = metal.shared_buffer(bytes.len()).unwrap();
                    buffer.write_bytes(0, bytes).unwrap();
                    buffer
                };
                let mut matrix = vec![0u8; HIDDEN * (QUERY_HEADS * HEAD_WIDTH / 256) * 210];
                for block in matrix.as_chunks_mut::<210>().0 {
                    block[..192].fill(0x55);
                    block[192..208].fill(1);
                    block[208..]
                        .copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                }
                let projection = DenseMatvec::new(
                    &metal,
                    &DirectWeight {
                        role: "test".into(),
                        format: WeightFormat::Q6K,
                        shape: vec![HIDDEN, QUERY_HEADS * HEAD_WIDTH],
                        bytes: matrix,
                    },
                )
                .unwrap();
                let attention = QsaAttention::from_projection(&metal, projection).unwrap();
                let ids = upload(
                    &selected
                        .iter()
                        .flat_map(|id| id.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                let selected_valid = upload(&valid);
                let selected_count = upload(&(active.len() as u32).to_le_bytes());
                let selection_status = upload(&0u32.to_le_bytes());
                let device_queries = upload(&words_bytes(&queries));
                let device_gate = upload(&words_bytes(&gate));
                let device_pending_keys = upload(&pending_keys);
                let device_pending_values = upload(&pending_values);
                let status = crate::device_status::DeviceStatus::new(&metal).unwrap();
                let mut stream = metal.stream().unwrap();
                attention
                    .encode_pending_online_device(
                        stream.batch().unwrap(),
                        &device,
                        &ids,
                        &selected_valid,
                        &selected_count,
                        &selection_status,
                        &device_queries,
                        &device_gate,
                        &device_pending_keys,
                        &device_pending_values,
                        &status,
                    )
                    .unwrap();
                stream.wait().unwrap();
                status.check().unwrap();
                let actual = read_words(&attention.scratch.gated, reference.len()).unwrap();
                assert_online_error(&actual, &reference, active.len(), &host_values);
                let mut error = [0u8; 4];
                selection_status.read_bytes(0, &mut error).unwrap();
                assert_eq!(u32::from_le_bytes(error), 0);

                ids.write_bytes(0, &i32::MAX.to_le_bytes()).unwrap();
                selection_status
                    .write_bytes(0, &0u32.to_le_bytes())
                    .unwrap();
                status.reset().unwrap();
                let mut stream = metal.stream().unwrap();
                attention
                    .encode_pending_online_device(
                        stream.batch().unwrap(),
                        &device,
                        &ids,
                        &selected_valid,
                        &selected_count,
                        &selection_status,
                        &device_queries,
                        &device_gate,
                        &device_pending_keys,
                        &device_pending_values,
                        &status,
                    )
                    .unwrap();
                stream.wait().unwrap();
                selection_status.read_bytes(0, &mut error).unwrap();
                assert_ne!(u32::from_le_bytes(error), 0);
                let mut guarded = [1u8; 1];
                attention.scratch.valid.read_bytes(0, &mut guarded).unwrap();
                assert_eq!(guarded, [0]);

                ids.write_bytes(0, &selected[0].to_le_bytes()).unwrap();
                selected_valid
                    .write_bytes(0, &vec![0; SELECTED_WIDTH])
                    .unwrap();
                selection_status
                    .write_bytes(0, &0u32.to_le_bytes())
                    .unwrap();
                status.reset().unwrap();
                let mut stream = metal.stream().unwrap();
                attention
                    .encode_pending_online_device(
                        stream.batch().unwrap(),
                        &device,
                        &ids,
                        &selected_valid,
                        &selected_count,
                        &selection_status,
                        &device_queries,
                        &device_gate,
                        &device_pending_keys,
                        &device_pending_values,
                        &status,
                    )
                    .unwrap();
                stream.wait().unwrap();
                selection_status.read_bytes(0, &mut error).unwrap();
                assert_ne!(u32::from_le_bytes(error), 0);
                assert!(
                    read_words(&attention.scratch.gated, QUERY_HEADS * HEAD_WIDTH)
                        .unwrap()
                        .iter()
                        .all(|&word| word == 0)
                );

                selected_valid.write_bytes(0, &valid).unwrap();
                for malformed in [0u32, 2052u32] {
                    selected_count
                        .write_bytes(0, &malformed.to_le_bytes())
                        .unwrap();
                    selection_status
                        .write_bytes(0, &0u32.to_le_bytes())
                        .unwrap();
                    status.reset().unwrap();
                    let mut stream = metal.stream().unwrap();
                    attention
                        .encode_pending_online_device(
                            stream.batch().unwrap(),
                            &device,
                            &ids,
                            &selected_valid,
                            &selected_count,
                            &selection_status,
                            &device_queries,
                            &device_gate,
                            &device_pending_keys,
                            &device_pending_values,
                            &status,
                        )
                        .unwrap();
                    stream.wait().unwrap();
                    selection_status.read_bytes(0, &mut error).unwrap();
                    assert_ne!(u32::from_le_bytes(error), 0);
                    let mut safe = [1u8; 4];
                    attention
                        .scratch
                        .online_count
                        .read_bytes(0, &mut safe)
                        .unwrap();
                    assert_eq!(u32::from_le_bytes(safe), 0);
                    assert!(
                        read_words(&attention.scratch.gated, QUERY_HEADS * HEAD_WIDTH)
                            .unwrap()
                            .iter()
                            .all(|&word| word == 0)
                    );
                }
            }
        }

        #[test]
        fn online_attention_split_boundaries_match_fp64() {
            let metal = Metal::new().unwrap();
            let tile = metal
                .compile_msl_strict(ONLINE_MSL, "qsa_online_tile")
                .unwrap();
            let merge = metal
                .compile_msl_strict(ONLINE_MSL, "qsa_online_merge")
                .unwrap();
            let upload = |bytes: &[u8]| {
                let buffer = metal.shared_buffer(bytes.len()).unwrap();
                buffer.write_bytes(0, bytes).unwrap();
                buffer
            };
            let mut queries = (0..QUERY_HEADS * HEAD_WIDTH)
                .map(|index| bf16::from_f32((index % 29) as f32 / 64. - 0.2).to_bits())
                .collect::<Vec<_>>();
            queries[..HEAD_WIDTH].fill(0);
            queries[HEAD_WIDTH..2 * HEAD_WIDTH].fill(bf16::from_f32(64.0).to_bits());
            let gate = (0..queries.len())
                .map(|index| bf16::from_f32((index % 23) as f32 / 2. - 5.0).to_bits())
                .collect::<Vec<_>>();
            let mut keys = (0..KV_HEADS * SELECTED_WIDTH * HEAD_WIDTH)
                .map(|index| bf16::from_f32((index % 211) as f32 / 512. - 0.2).to_bits())
                .collect::<Vec<_>>();
            for kv_head in 0..KV_HEADS {
                let first = kv_head * SELECTED_WIDTH * HEAD_WIDTH;
                keys[first..first + HEAD_WIDTH].fill(bf16::from_f32(2.0).to_bits());
                keys[first + HEAD_WIDTH..first + 2 * HEAD_WIDTH]
                    .fill(bf16::from_f32(-2.0).to_bits());
            }
            let values = (0..keys.len())
                .map(|index| bf16::from_f32((index % 173) as f32 / 400. - 0.2).to_bits())
                .collect::<Vec<_>>();
            let q = upload(&words_bytes(&queries));
            let g = upload(&words_bytes(&gate));
            let k = upload(&words_bytes(&keys));
            let v = upload(&words_bytes(&values));
            let validity = metal.shared_buffer(SELECTED_WIDTH).unwrap();
            let count_buffer = metal.shared_buffer(4).unwrap();
            let partials = metal
                .shared_buffer(QUERY_HEADS * MAX_SPLITS * PART_STRIDE * 4)
                .unwrap();
            let output = metal.shared_buffer(QUERY_HEADS * HEAD_WIDTH * 2).unwrap();
            for count in [1usize, 32, 33, 65, 2051] {
                let mut valid = vec![0u8; SELECTED_WIDTH];
                valid[..count].fill(1);
                validity.write_bytes(0, &valid).unwrap();
                count_buffer
                    .write_bytes(0, &(count as u32).to_le_bytes())
                    .unwrap();
                let splits = count.div_ceil(32).clamp(1, MAX_SPLITS);
                let mut batch = metal.batch().unwrap();
                let tile_bindings = [&q, &k, &v, &validity, &count_buffer, &partials]
                    .map(|buffer| Binding { buffer, offset: 0 });
                let merge_bindings = [&partials, &g, &count_buffer, &output]
                    .map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: The active split count bounds all partial writes;
                // the merge reads precisely those initialized split records.
                unsafe {
                    batch
                        .dispatch_1d(&tile, &tile_bindings, splits * KV_HEADS * 128, 128)
                        .unwrap();
                    batch
                        .dispatch_1d(&merge, &merge_bindings, QUERY_HEADS * 32, 32)
                        .unwrap();
                }
                batch.submit().wait().unwrap();
                let actual = read_words(&output, QUERY_HEADS * HEAD_WIDTH).unwrap();
                let reference =
                    fp64_attention_reference(&queries, &gate, &keys, &values, &valid, count);
                assert_online_error(&actual, &reference, count, &values);
            }
        }

        #[test]
        fn pending_device_attention_matches_host_stages_and_rejects_bad_rows() {
            let metal = Metal::new().unwrap();
            let mut matrix = vec![0u8; HIDDEN * (QUERY_HEADS * HEAD_WIDTH / 256) * 210];
            for block in matrix.as_chunks_mut::<210>().0 {
                block[..192].fill(0x55);
                block[192..208].fill(1);
                block[208..].copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
            }
            let weight = DirectWeight {
                role: "test".into(),
                format: WeightFormat::Q6K,
                shape: vec![HIDDEN, QUERY_HEADS * HEAD_WIDTH],
                bytes: matrix,
            };
            let projection = DenseMatvec::new(&metal, &weight).unwrap();
            let mut attention = QsaAttention::from_projection(&metal, projection).unwrap();
            let data = |bytes: usize, salt: usize| {
                (0..bytes / 2)
                    .flat_map(|i| {
                        bf16::from_f32(((i * 37 + salt) % 127) as f32 / 512. - 0.125)
                            .to_bits()
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>()
            };
            let sink = data(SINK_BYTES, 1);
            let tail = data(TAIL_BYTES, 9);
            let host = KvarnState::restore(
                9000,
                8448,
                &vec![0; TILE_RECORD_BYTES],
                &sink,
                &sink,
                &tail,
                &tail,
            )
            .unwrap();
            let device = DeviceKvarnState::restore(
                &metal,
                9000,
                8448,
                host.records(),
                &sink,
                &sink,
                &tail,
                &tail,
            )
            .unwrap();
            let pending = data(1024, 3);
            let queries = data(QUERY_HEADS * HEAD_WIDTH * 2, 13)
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes(*v))
                .collect::<Vec<_>>();
            let gate = queries.clone();
            let mut ids = vec![-1; SELECTED_WIDTH];
            ids[..4].copy_from_slice(&[0, 128, 8430, 8448]);
            let rows = KvarnDecoder::new(&metal)
                .unwrap()
                .gather_pending_row(&host, &ids, &pending, &pending)
                .unwrap();
            let words = |bytes: &[u8]| {
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| u16::from_le_bytes(*v))
                    .collect::<Vec<_>>()
            };
            let expected = attention
                .run_stages(
                    &queries,
                    &gate,
                    &words(&rows.keys_bf16),
                    &words(&rows.values_bf16),
                    &rows.valid.iter().map(|&v| u8::from(v)).collect::<Vec<_>>(),
                )
                .unwrap();
            let output = attention
                .run_pending(&device, &ids, &queries, &gate, &pending, &pending)
                .unwrap();
            assert_eq!(output, expected.output_bf16);
            let upload = |bytes: &[u8]| {
                let buffer = metal.shared_buffer(bytes.len()).unwrap();
                buffer.write_bytes(0, bytes).unwrap();
                buffer
            };
            let device_queries = upload(&words_bytes(&queries));
            let device_gate = upload(&words_bytes(&gate));
            let device_pending = upload(&pending);
            let status = crate::device_status::DeviceStatus::new(&metal).unwrap();
            let mut stream = metal.stream().unwrap();
            let encoded_output = attention
                .encode_pending_device(
                    stream.batch().unwrap(),
                    &device,
                    &ids,
                    &device_queries,
                    &device_gate,
                    &device_pending,
                    &device_pending,
                    &status,
                )
                .unwrap();
            assert!(encoded_output.read_bytes(0, &mut [0; 2]).is_err());
            stream.flush().unwrap();
            status
                .encode_bf16(stream.batch().unwrap(), &encoded_output)
                .unwrap();
            stream.wait().unwrap();
            status.check().unwrap();
            assert_eq!(
                read_words(&encoded_output, HIDDEN).unwrap(),
                expected.output_bf16
            );
            for (buffer, words) in [
                (&attention.scratch.logits, expected.logits_bf16),
                (&attention.scratch.scaled, expected.scaled_logits_bf16),
                (
                    &attention.scratch.probabilities,
                    expected.probabilities_bf16,
                ),
                (&attention.scratch.attention, expected.attention_bf16),
                (&attention.scratch.sigmoid, expected.gate_sigmoid_bf16),
                (&attention.scratch.gated, expected.gated_bf16),
            ] {
                assert_eq!(read_words(buffer, words.len()).unwrap(), words);
            }
            let mut bad_pending = pending.clone();
            bad_pending[..2].copy_from_slice(&0x7fc0u16.to_le_bytes());
            assert!(
                attention
                    .run_pending(&device, &ids, &queries, &gate, &bad_pending, &pending)
                    .is_err()
            );
            let mut bad_ids = ids.clone();
            bad_ids[1] = 0;
            assert!(
                attention
                    .run_pending(&device, &bad_ids, &queries, &gate, &pending, &pending)
                    .is_err()
            );
            // Deliberate test-only corruption bypasses the validated restore API.
            device.buffers().unwrap()[0]
                .0
                .write_bytes(16384, &0x7e00u16.to_le_bytes())
                .unwrap();
            assert!(
                attention
                    .run_pending(&device, &ids, &queries, &gate, &pending, &pending)
                    .is_err()
            );
            assert_eq!(device.frontier().unwrap(), 8448);
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{QsaAttention, QsaAttentionStages};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_row_shape_and_validity_fail_closed() {
        let queries = vec![0u16; QUERY_HEADS * HEAD_WIDTH];
        let gate = queries.clone();
        let keys = vec![0u16; KV_HEADS * SELECTED_WIDTH * HEAD_WIDTH];
        let values = keys.clone();
        let mut valid = vec![0u8; SELECTED_WIDTH];
        assert!(validate_selected_rows(&queries, &gate, &keys, &values, &valid).is_err());
        valid[0] = 1;
        assert!(validate_selected_rows(&queries, &gate, &keys, &values, &valid).is_ok());
        valid[2] = 1;
        assert!(validate_selected_rows(&queries, &gate, &keys, &values, &valid).is_err());
        valid[2] = 2;
        assert!(validate_selected_rows(&queries, &gate, &keys, &values, &valid).is_err());
    }
}

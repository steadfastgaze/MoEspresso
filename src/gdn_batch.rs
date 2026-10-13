//! Multi-row GDN over explicit candidate state and reusable direct projections.

use anyhow::{Result, ensure};

pub const HIDDEN: usize = 2560;
pub const CONV: usize = 10240;
pub const KEY: usize = 2048;
pub const VALUE: usize = 6144;
pub const HEADS: usize = 48;
pub const RECURRENT_BYTES: usize = 48 * 128 * 128 * 4;
pub const CONV_BYTES: usize = 3 * CONV * 2;
pub const METAL_SOURCE: &str = include_str!("../kernels/metal/gdn_batch.metal");

pub fn next_frontier(frontier: u64, rows: usize) -> Result<u64> {
    ensure!(
        (2..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
        "GDN batch supports 2 through 4096 rows"
    );
    frontier
        .checked_add(rows as u64)
        .ok_or_else(|| anyhow::anyhow!("GDN batch frontier overflow"))
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use crate::dense_batch::DenseBatch;
    use crate::dense_batch::DenseBatchScratch;
    use crate::device_status::DeviceStatus;
    use crate::gdn::{GdnState, GdnWeights};
    use crate::metal::{Batch, Binding, Buffer, InFlight, Metal, Pipeline};
    use crate::package::{DirectWeight, WeightFormat};
    use anyhow::Context;
    use half::bf16;
    use std::collections::BTreeMap;

    fn upload(metal: &Metal, bytes: &[u8]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(bytes.len())?;
        buffer.write_bytes(0, bytes)?;
        Ok(buffer)
    }
    fn finite_bf16(bytes: &[u8], words: usize, name: &str) -> Result<()> {
        ensure!(bytes.len() == words * 2, "{name}: BF16 extent differs");
        ensure!(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .all(|v| bf16::from_bits(u16::from_le_bytes(*v)).is_finite()),
            "{name}: nonfinite BF16 value"
        );
        Ok(())
    }
    fn raw(weight: &DirectWeight, shape: &[usize], name: &str) -> Result<()> {
        ensure!(
            weight.format == WeightFormat::Bf16 && weight.shape == shape,
            "{name}: BF16 parameter geometry differs"
        );
        finite_bf16(&weight.bytes, shape.iter().product(), name)
    }
    fn read(buffer: &Buffer) -> Result<Vec<u8>> {
        let mut bytes = vec![0; buffer.byte_len()];
        buffer.read_bytes(0, &mut bytes)?;
        Ok(bytes)
    }

    struct Scratch {
        qkv: Buffer,
        gate: Buffer,
        beta_logits: Buffer,
        decay_logits: Buffer,
        conv: Buffer,
        activated: Buffer,
        query: Buffer,
        key: Buffer,
        value: Buffer,
        beta: Buffer,
        decay: Buffer,
        recurrence: Buffer,
        norm_gate: Buffer,
        output: Buffer,
        next_conv: Buffer,
        next_recurrent: Buffer,
    }
    impl Scratch {
        fn new(metal: &Metal, rows: usize) -> Result<Self> {
            Self::allocate(metal, rows, RECURRENT_BYTES)
        }

        fn allocate(metal: &Metal, rows: usize, recurrent_bytes: usize) -> Result<Self> {
            let row = |width| metal.shared_buffer(rows * width * 2);
            Ok(Self {
                qkv: row(CONV)?,
                gate: row(VALUE)?,
                beta_logits: row(HEADS)?,
                decay_logits: row(HEADS)?,
                conv: row(CONV)?,
                activated: row(CONV)?,
                query: row(KEY)?,
                key: row(KEY)?,
                value: row(VALUE)?,
                beta: row(HEADS)?,
                decay: metal.shared_buffer(rows * HEADS * 4)?,
                recurrence: row(VALUE)?,
                norm_gate: row(VALUE)?,
                output: row(HIDDEN)?,
                next_conv: metal.shared_buffer(CONV_BYTES)?,
                next_recurrent: metal.shared_buffer(recurrent_bytes)?,
            })
        }
        fn stages(&self) -> Result<BTreeMap<&'static str, Vec<u8>>> {
            [
                ("qkv_bf16", &self.qkv),
                ("gate_bf16", &self.gate),
                ("beta_logits_bf16", &self.beta_logits),
                ("decay_logits_bf16", &self.decay_logits),
                ("conv_bf16", &self.conv),
                ("activated_bf16", &self.activated),
                ("query_bf16", &self.query),
                ("key_bf16", &self.key),
                ("value_bf16", &self.value),
                ("beta_bf16", &self.beta),
                ("decay_f32", &self.decay),
                ("recurrence_bf16", &self.recurrence),
                ("norm_gate_bf16", &self.norm_gate),
                ("output_bf16", &self.output),
                ("conv_state_bf16", &self.next_conv),
                ("recurrent_state_f32", &self.next_recurrent),
            ]
            .into_iter()
            .map(|(name, buffer)| Ok((name, read(buffer)?)))
            .collect()
        }
    }

    /// One layer's working storage, reused across all GDN layers.
    pub struct GdnBatchScratch {
        capacity: usize,
        scratch: Scratch,
    }

    impl GdnBatchScratch {
        pub fn new(metal: &Metal, capacity: usize) -> Result<Self> {
            next_frontier(0, capacity)?;
            Ok(Self {
                capacity,
                scratch: Scratch::allocate(metal, capacity, 4)?,
            })
        }

        pub fn byte_len(&self) -> usize {
            let s = &self.scratch;
            [
                &s.qkv,
                &s.gate,
                &s.beta_logits,
                &s.decay_logits,
                &s.conv,
                &s.activated,
                &s.query,
                &s.key,
                &s.value,
                &s.beta,
                &s.decay,
                &s.recurrence,
                &s.norm_gate,
                &s.output,
                &s.next_conv,
                &s.next_recurrent,
            ]
            .iter()
            .map(|v| v.byte_len())
            .sum()
        }
    }

    pub struct GdnBatch {
        metal: Metal,
        qkv: DenseBatch,
        gate: DenseBatch,
        beta: DenseBatch,
        decay: DenseBatch,
        output: DenseBatch,
        conv_weight: Buffer,
        a_log: Buffer,
        dt_bias: Buffer,
        norm_weight: Buffer,
        eps: Buffer,
        conv: Pipeline,
        prepare: Pipeline,
        recurrence: Pipeline,
        recurrence_r4: Pipeline,
        norm_gate: Pipeline,
    }

    /// Completed output and unpublished candidate state. The caller commits
    /// the candidate only after its entire layer or model chunk succeeds.
    pub struct GdnBatchOutput {
        pub output: Buffer,
        pub state: GdnState,
        pub rows: usize,
    }
    pub struct GdnBatchStages {
        pub output: GdnBatchOutput,
        pub stages: BTreeMap<&'static str, Vec<u8>>,
    }
    /// Dropping an unfinished result waits for its GPU readers but publishes
    /// no candidate. The original state is never modified by this operation.
    pub struct PendingGdnBatch {
        flight: InFlight,
        scratch: Scratch,
        rows: usize,
        frontier: u64,
    }
    impl PendingGdnBatch {
        pub fn wait(self) -> Result<GdnBatchOutput> {
            let Self {
                flight,
                scratch,
                rows,
                frontier,
            } = self;
            flight.wait().context("GDN batch command")?;
            Ok(GdnBatchOutput {
                output: scratch.output,
                state: GdnState::from_buffers(scratch.next_conv, scratch.next_recurrent, frontier)?,
                rows,
            })
        }
        pub fn wait_stages(self) -> Result<GdnBatchStages> {
            let Self {
                flight,
                scratch,
                rows,
                frontier,
            } = self;
            flight.wait().context("GDN batch command")?;
            let stages = scratch.stages()?;
            Ok(GdnBatchStages {
                output: GdnBatchOutput {
                    output: scratch.output,
                    state: GdnState::from_buffers(
                        scratch.next_conv,
                        scratch.next_recurrent,
                        frontier,
                    )?,
                    rows,
                },
                stages,
            })
        }
    }

    impl GdnBatch {
        pub fn dense_scratch_bytes(&self, capacity: usize) -> Result<usize> {
            [&self.qkv, &self.gate, &self.beta, &self.decay, &self.output]
                .iter()
                .try_fold(4, |bytes, op| Ok(bytes.max(op.scratch_bytes(capacity)?)))
        }

        /// Encode a chunk against canonical state. Publication follows the
        /// request's final completion and finite-status check.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn encode_chunk(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            rows: usize,
            state: &mut GdnState,
            workspace: &mut GdnBatchScratch,
            dense: &DenseBatchScratch,
            status: &DeviceStatus,
        ) -> Result<Buffer> {
            next_frontier(state.frontier()?, rows)?;
            ensure!(
                rows <= workspace.capacity && hidden.byte_len() >= rows * HIDDEN * 2,
                "GDN chunk exceeds workspace"
            );
            let scratch = &mut workspace.scratch;
            let (old_conv, recurrent) = state.buffers()?;
            let count = (rows as u32).to_le_bytes();
            // SAFETY: Scratch is shared only between ordered layers. Each SIMD
            // owner reads and writes exclusive recurrent rows. Convolution
            // history has a distinct destination until its final reader ends.
            unsafe {
                self.qkv
                    .encode_rows_into(batch, hidden, &scratch.qkv, rows, dense)?;
                self.gate
                    .encode_rows_into(batch, hidden, &scratch.gate, rows, dense)?;
                self.beta
                    .encode_rows_into(batch, hidden, &scratch.beta_logits, rows, dense)?;
                self.decay
                    .encode_rows_into(batch, hidden, &scratch.decay_logits, rows, dense)?;
                let conv = [
                    &scratch.qkv,
                    old_conv,
                    &self.conv_weight,
                    &scratch.conv,
                    &scratch.activated,
                    &scratch.next_conv,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d_with_inline(
                    &self.conv,
                    &conv,
                    &[&count],
                    rows.max(3) * CONV,
                    128,
                )?;
                let prepare = [
                    &scratch.activated,
                    &scratch.beta_logits,
                    &scratch.decay_logits,
                    &self.a_log,
                    &self.dt_bias,
                    &scratch.query,
                    &scratch.key,
                    &scratch.value,
                    &scratch.beta,
                    &scratch.decay,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.prepare, &prepare, rows * 80 * 32, 32)?;
                let recurrence = [
                    &scratch.query,
                    &scratch.key,
                    &scratch.value,
                    &scratch.decay,
                    &scratch.beta,
                    recurrent,
                    recurrent,
                    &scratch.recurrence,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                let (scan, threads) = if rows > 8 {
                    (&self.recurrence_r4, 32)
                } else {
                    (&self.recurrence, 128)
                };
                batch.dispatch_1d_with_inline(
                    scan,
                    &recurrence,
                    &[&count],
                    48 * 32 * threads,
                    threads,
                )?;
                let norm = [
                    &scratch.recurrence,
                    &scratch.gate,
                    &self.norm_weight,
                    &scratch.norm_gate,
                    &self.eps,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.norm_gate, &norm, rows * HEADS * 32, 32)?;
                self.output.encode_rows_into(
                    batch,
                    &scratch.norm_gate,
                    &scratch.output,
                    rows,
                    dense,
                )?;
            }
            status.encode_f32(batch, recurrent)?;
            status.encode_bf16(batch, &scratch.next_conv)?;
            state.stage_chunk_history(&mut scratch.next_conv)?;
            Ok(scratch.output.clone())
        }
        pub(crate) fn from_decode(metal: &Metal, source: &crate::gdn::GdnCell) -> Result<Self> {
            let weights = source.shared_weights();
            let [qkv, gate, beta, decay, output] = weights.matrices;
            let source = format!(
                "{}\n{}",
                include_str!("../kernels/metal/gdn.metal"),
                METAL_SOURCE
            );
            let pipeline = |name, threads| -> Result<Pipeline> {
                let p = metal.compile_msl_strict(&source, name)?;
                ensure!(
                    p.max_threads_per_threadgroup() >= threads,
                    "GDN batch threadgroup unsupported"
                );
                Ok(p)
            };
            Ok(Self {
                metal: metal.clone(),
                qkv: DenseBatch::from_matrix(qkv)?,
                gate: DenseBatch::from_matrix(gate)?,
                beta: DenseBatch::from_matrix(beta)?,
                decay: DenseBatch::from_matrix(decay)?,
                output: DenseBatch::from_matrix(output)?,
                conv_weight: weights.conv,
                a_log: weights.a_log,
                dt_bias: weights.dt_bias,
                norm_weight: weights.norm,
                eps: weights.epsilon,
                conv: pipeline("gdn_batch_conv", 128)?,
                prepare: pipeline("gdn_batch_prepare", 32)?,
                recurrence: pipeline("gdn_batch_recurrence", 128)?,
                recurrence_r4: pipeline("gdn_batch_recurrence_r4", 32)?,
                norm_gate: pipeline("gdn_norm_gate", 32)?,
            })
        }
        pub fn new(metal: &Metal, weights: GdnWeights<'_>) -> Result<Self> {
            for (weight, shape, name) in [
                (weights.qkv, [CONV, HIDDEN], "qkv"),
                (weights.gate, [VALUE, HIDDEN], "gate"),
                (weights.beta, [HEADS, HIDDEN], "beta"),
                (weights.decay, [HEADS, HIDDEN], "decay"),
                (weights.output, [HIDDEN, VALUE], "output"),
            ] {
                ensure!(
                    weight.format == WeightFormat::Q6K && weight.shape == shape,
                    "GDN {name}: Q6_K projection geometry differs"
                );
            }
            ensure!(
                weights.conv.format == WeightFormat::Bf16
                    && (weights.conv.shape == [CONV, 1, 4] || weights.conv.shape == [CONV, 4, 1]),
                "GDN convolution parameter geometry differs"
            );
            finite_bf16(&weights.conv.bytes, CONV * 4, "GDN convolution weight")?;
            raw(weights.a_log, &[HEADS], "GDN A_log")?;
            raw(weights.dt_bias, &[HEADS], "GDN dt_bias")?;
            raw(weights.norm, &[128], "GDN norm")?;
            ensure!(
                weights.eps.is_finite() && weights.eps > 0.0,
                "GDN epsilon must be finite and positive"
            );
            let source = format!(
                "{}\n{}",
                include_str!("../kernels/metal/gdn.metal"),
                METAL_SOURCE
            );
            let pipeline = |name, threads| -> Result<Pipeline> {
                let p = metal.compile_msl_strict(&source, name)?;
                ensure!(
                    p.max_threads_per_threadgroup() >= threads,
                    "GDN batch threadgroup unsupported"
                );
                Ok(p)
            };
            Ok(Self {
                metal: metal.clone(),
                qkv: DenseBatch::new(metal, weights.qkv)?,
                gate: DenseBatch::new(metal, weights.gate)?,
                beta: DenseBatch::new(metal, weights.beta)?,
                decay: DenseBatch::new(metal, weights.decay)?,
                output: DenseBatch::new(metal, weights.output)?,
                conv_weight: upload(metal, &weights.conv.bytes)?,
                a_log: upload(metal, &weights.a_log.bytes)?,
                dt_bias: upload(metal, &weights.dt_bias.bytes)?,
                norm_weight: upload(metal, &weights.norm.bytes)?,
                eps: upload(metal, &weights.eps.to_le_bytes())?,
                conv: pipeline("gdn_batch_conv", 128)?,
                prepare: pipeline("gdn_batch_prepare", 32)?,
                recurrence: pipeline("gdn_batch_recurrence", 128)?,
                recurrence_r4: pipeline("gdn_batch_recurrence_r4", 32)?,
                norm_gate: pipeline("gdn_norm_gate", 32)?,
            })
        }

        pub fn submit(
            &self,
            hidden_bf16: &[u8],
            rows: usize,
            state: &GdnState,
        ) -> Result<PendingGdnBatch> {
            next_frontier(state.frontier()?, rows)?;
            finite_bf16(hidden_bf16, rows * HIDDEN, "GDN batch hidden")?;
            let hidden = upload(&self.metal, hidden_bf16)?;
            // SAFETY: Input has checked finite BF16 values and declared extent.
            unsafe { self.submit_buffer(&hidden, rows, state) }
        }

        /// Execute the entire cell in one command batch, retaining GPU input.
        ///
        /// # Safety
        /// Input must contain finite BF16 `[rows,2560]` values produced before
        /// this submission. Its storage must not alias parameter or state
        /// buffers. The command owner excludes concurrent host writes.
        pub unsafe fn submit_buffer(
            &self,
            hidden: &Buffer,
            rows: usize,
            state: &GdnState,
        ) -> Result<PendingGdnBatch> {
            let frontier = next_frontier(state.frontier()?, rows)?;
            ensure!(
                hidden.byte_len() >= rows * HIDDEN * 2,
                "GDN batch hidden buffer too small"
            );
            let (old_conv, old_recurrent) = state.buffers()?;
            let scratch = Scratch::new(&self.metal, rows)?;
            let count = upload(&self.metal, &(rows as u32).to_le_bytes())?;
            let mut batch = self.metal.batch()?;
            // SAFETY: All scratch buffers have the fixed geometry above,
            // candidate state is disjoint, and dispatches share one owner.
            unsafe {
                self.qkv
                    .encode_rows(&mut batch, hidden, &scratch.qkv, rows)?;
                self.gate
                    .encode_rows(&mut batch, hidden, &scratch.gate, rows)?;
                self.beta
                    .encode_rows(&mut batch, hidden, &scratch.beta_logits, rows)?;
                self.decay
                    .encode_rows(&mut batch, hidden, &scratch.decay_logits, rows)?;
                let conv = [
                    &scratch.qkv,
                    old_conv,
                    &self.conv_weight,
                    &scratch.conv,
                    &scratch.activated,
                    &scratch.next_conv,
                    &count,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.conv, &conv, rows.max(3) * CONV, 128)?;
                let prepare = [
                    &scratch.activated,
                    &scratch.beta_logits,
                    &scratch.decay_logits,
                    &self.a_log,
                    &self.dt_bias,
                    &scratch.query,
                    &scratch.key,
                    &scratch.value,
                    &scratch.beta,
                    &scratch.decay,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.prepare, &prepare, rows * 80 * 32, 32)?;
                let recurrence = [
                    &scratch.query,
                    &scratch.key,
                    &scratch.value,
                    &scratch.decay,
                    &scratch.beta,
                    old_recurrent,
                    &scratch.next_recurrent,
                    &scratch.recurrence,
                    &count,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.recurrence, &recurrence, 48 * 32 * 128, 128)?;
                let norm = [
                    &scratch.recurrence,
                    &scratch.gate,
                    &self.norm_weight,
                    &scratch.norm_gate,
                    &self.eps,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.norm_gate, &norm, rows * HEADS * 32, 32)?;
                self.output
                    .encode_rows(&mut batch, &scratch.norm_gate, &scratch.output, rows)?;
            }
            Ok(PendingGdnBatch {
                flight: batch.submit(),
                scratch,
                rows,
                frontier,
            })
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{GdnBatch, GdnBatchOutput, GdnBatchScratch, GdnBatchStages, PendingGdnBatch};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_frontier_is_bounded_and_checked() {
        assert_eq!(next_frontier(7, 64).unwrap(), 71);
        assert_eq!(next_frontier(0, 512).unwrap(), 512);
        assert_eq!(next_frontier(17, 4096).unwrap(), 4113);
        assert_eq!(next_frontier(7, 2).unwrap(), 9);
        assert_eq!(next_frontier(7, 3).unwrap(), 10);
        for rows in [0, 1, 4097, usize::MAX] {
            assert!(next_frontier(0, rows).is_err());
        }
        assert!(next_frontier(u64::MAX - 63, 64).is_err());
    }
}

//! Generic multi-row Qwen residual reads with explicit BF16 boundaries.

#[cfg(target_os = "macos")]
mod gpu {
    use anyhow::{Result, ensure};
    use half::bf16;

    use crate::dense_batch::{DenseBatch, DenseBatchScratch};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use crate::package::{Package, WeightFormat};

    const SOURCE: &str = include_str!("../kernels/metal/hc_batch.metal");
    const WIDTH: usize = 2560;
    const EXPANDED: usize = WIDTH * 4;
    const LOWRANK: usize = 320;

    /// Temporaries reused by every residual read on the ordered stream.
    pub struct HcBatchScratch {
        capacity: usize,
        normalized: Buffer,
        projected_down: Buffer,
        projected_up: Buffer,
        projected_injection: Buffer,
        lowrank: Buffer,
        dummy: Buffer,
    }

    /// Two instances alternate between attention and MLP residual reads.
    pub struct HcBatchRows {
        pub updated: Buffer,
        pub mixed: Buffer,
        pub injection: Buffer,
    }

    impl HcBatchScratch {
        pub fn new(metal: &Metal, capacity: usize) -> Result<Self> {
            ensure!(
                (2..=crate::prefill_capacity::MAX_ROWS).contains(&capacity),
                "HC workspace capacity is invalid"
            );
            let row = |width| metal.shared_buffer(capacity * width * 2);
            Ok(Self {
                capacity,
                normalized: row(EXPANDED)?,
                projected_down: row(LOWRANK)?,
                projected_up: row(EXPANDED)?,
                projected_injection: row(4)?,
                lowrank: row(LOWRANK)?,
                dummy: metal.shared_buffer(2)?,
            })
        }

        pub fn byte_len(&self) -> usize {
            [
                &self.normalized,
                &self.projected_down,
                &self.projected_up,
                &self.projected_injection,
                &self.lowrank,
                &self.dummy,
            ]
            .iter()
            .map(|buffer| buffer.byte_len())
            .sum()
        }
    }

    impl HcBatchRows {
        pub fn new(metal: &Metal, capacity: usize) -> Result<Self> {
            ensure!(
                (2..=crate::prefill_capacity::MAX_ROWS).contains(&capacity),
                "HC workspace capacity is invalid"
            );
            Ok(Self {
                updated: metal.shared_buffer(capacity * EXPANDED * 2)?,
                mixed: metal.shared_buffer(capacity * WIDTH * 2)?,
                injection: metal.shared_buffer(capacity * 4 * 2)?,
            })
        }

        pub fn byte_len(&self) -> usize {
            self.updated.byte_len() + self.mixed.byte_len() + self.injection.byte_len()
        }
    }

    pub struct HcBatch {
        metal: Metal,
        down: DenseBatch,
        up: DenseBatch,
        injection: Option<DenseBatch>,
        norm_weight: Buffer,
        epsilon: Buffer,
        norm: Pipeline,
        front: Pipeline,
        mix: Pipeline,
    }

    /// Each buffer is retained by the submitted batch until completion.
    pub struct HcBatchOutput {
        pub updated: Buffer,
        pub mixed: Buffer,
        pub injection: Option<Buffer>,
        normalized: Buffer,
        projected_down: Buffer,
        projected_up: Buffer,
        projected_injection: Buffer,
        lowrank: Buffer,
        rows: usize,
    }

    pub struct HcBatchStages {
        pub updated: Vec<u16>,
        pub normalized: Vec<u16>,
        pub projected_down: Vec<u16>,
        pub lowrank: Vec<u16>,
        pub projected_up: Vec<u16>,
        pub mixed: Vec<u16>,
        pub projected_injection: Option<Vec<u16>>,
        pub injection: Option<Vec<u16>>,
    }

    fn checked_words(values: &[u16], expected: usize) -> Result<()> {
        ensure!(
            values.len() == expected,
            "HC batch input length differs from geometry"
        );
        ensure!(
            values.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "HC batch requires finite inputs"
        );
        Ok(())
    }

    fn read(buffer: &Buffer, words: usize) -> Result<Vec<u16>> {
        let mut bytes = vec![0; words * 2];
        buffer.read_bytes(0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes(*v))
            .collect())
    }

    impl HcBatchOutput {
        /// Read diagnostics after device completion; reserved buffers fail closed.
        ///
        /// # Safety
        /// The command that encoded these outputs must have been submitted
        /// and successfully waited. Dropping an unsubmitted batch is not enough.
        pub unsafe fn stages(&self) -> Result<HcBatchStages> {
            Ok(HcBatchStages {
                updated: read(&self.updated, self.rows * EXPANDED)?,
                normalized: read(&self.normalized, self.rows * EXPANDED)?,
                projected_down: read(&self.projected_down, self.rows * LOWRANK)?,
                lowrank: read(&self.lowrank, self.rows * LOWRANK)?,
                projected_up: read(&self.projected_up, self.rows * EXPANDED)?,
                mixed: read(&self.mixed, self.rows * WIDTH)?,
                projected_injection: self
                    .injection
                    .as_ref()
                    .map(|_| read(&self.projected_injection, self.rows * 4))
                    .transpose()?,
                injection: self
                    .injection
                    .as_ref()
                    .map(|v| read(v, self.rows * 4))
                    .transpose()?,
            })
        }
    }

    impl HcBatch {
        pub fn dense_scratch_bytes(&self, capacity: usize) -> Result<usize> {
            let bytes = self
                .down
                .scratch_bytes(capacity)?
                .max(self.up.scratch_bytes(capacity)?);
            Ok(match &self.injection {
                Some(p) => bytes.max(p.scratch_bytes(capacity)?),
                None => bytes,
            })
        }
        pub(crate) fn from_decode(metal: &Metal, source: &crate::hc::HcBlock) -> Result<Self> {
            let weights = source.shared_weights();
            let norm = metal.compile_msl_strict(SOURCE, "hc_batch_norm")?;
            let front = metal.compile_msl_strict(SOURCE, "hc_batch_front")?;
            let mix = metal.compile_msl_strict(SOURCE, "hc_batch_mix")?;
            ensure!(
                norm.max_threads_per_threadgroup() >= 640
                    && front.max_threads_per_threadgroup() >= 64
                    && mix.max_threads_per_threadgroup() >= 256,
                "unsupported HC batch threadgroups"
            );
            Ok(Self {
                metal: metal.clone(),
                down: DenseBatch::from_matrix(weights.down)?,
                up: DenseBatch::from_matrix(weights.up)?,
                injection: weights.injection.map(DenseBatch::from_matrix).transpose()?,
                norm_weight: weights.norm,
                epsilon: weights.epsilon,
                norm,
                front,
                mix,
            })
        }
        pub fn new(
            metal: &Metal,
            package: &Package,
            prefix: &str,
            with_injection: bool,
            eps: f32,
        ) -> Result<Self> {
            ensure!(
                eps.is_finite() && eps > 0.0,
                "HC epsilon must be positive and finite"
            );
            let matrix = |name: &str, format, shape: [usize; 2]| -> Result<DenseBatch> {
                let weight = package.direct_weight(&format!("{prefix}.{name}"))?;
                ensure!(
                    weight.format == format && weight.shape == shape,
                    "HC batch {name} has unsupported geometry"
                );
                DenseBatch::new(metal, &weight)
            };
            let down = matrix(
                "input_mix_weight_down",
                WeightFormat::Q6K,
                [LOWRANK, EXPANDED],
            )?;
            let up = matrix(
                "input_mix_weight_up",
                WeightFormat::Q8_0,
                [EXPANDED, LOWRANK],
            )?;
            let injection = with_injection
                .then(|| matrix("block_inject_weight", WeightFormat::Q6K, [4, EXPANDED]))
                .transpose()?;
            let weight = package.direct_weight(&format!("{prefix}.hc_norm"))?;
            ensure!(
                weight.format == WeightFormat::Bf16 && weight.shape == [EXPANDED],
                "HC norm geometry differs"
            );
            ensure!(
                weight
                    .bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|v| bf16::from_bits(u16::from_le_bytes(*v)).is_finite()),
                "HC norm weight contains nonfinite values"
            );
            let norm_weight = metal.shared_buffer(EXPANDED * 2)?;
            norm_weight.write_bytes(0, &weight.bytes)?;
            let epsilon = metal.shared_buffer(4)?;
            epsilon.write_bytes(0, &eps.to_le_bytes())?;
            let norm = metal.compile_msl_strict(SOURCE, "hc_batch_norm")?;
            let front = metal.compile_msl_strict(SOURCE, "hc_batch_front")?;
            let mix = metal.compile_msl_strict(SOURCE, "hc_batch_mix")?;
            ensure!(
                norm.max_threads_per_threadgroup() >= 640
                    && front.max_threads_per_threadgroup() >= 64
                    && mix.max_threads_per_threadgroup() >= 256,
                "unsupported HC batch threadgroups"
            );
            Ok(Self {
                metal: metal.clone(),
                down,
                up,
                injection,
                norm_weight,
                epsilon,
                norm,
                front,
                mix,
            })
        }

        /// Encode a multi-row read without submitting or waiting.
        ///
        /// # Safety
        /// Inputs must contain finite BF16 rows with the declared geometry.
        /// Earlier writes must be complete or ordered within this batch.
        /// Pending output and injection have widths 2560 and 4 respectively.
        pub unsafe fn encode(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            pending: Option<(&Buffer, &Buffer)>,
            rows: usize,
        ) -> Result<HcBatchOutput> {
            let scratch = HcBatchScratch::new(&self.metal, rows)?;
            let output = HcBatchRows::new(&self.metal, rows)?;
            let dense = DenseBatchScratch::new(&self.metal, self.dense_scratch_bytes(rows)?)?;
            // SAFETY: All scratch is private; the caller supplies initialized rows.
            unsafe { self.encode_into(batch, input, pending, rows, &scratch, &output, &dense) }
        }

        /// Encode into shared temporaries and one alternating output workspace.
        ///
        /// # Safety
        /// Output storage must not alias inputs. All earlier workspace readers
        /// precede this call on the same ordered stream.
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn encode_into(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            pending: Option<(&Buffer, &Buffer)>,
            rows: usize,
            scratch: &HcBatchScratch,
            output: &HcBatchRows,
            dense: &DenseBatchScratch,
        ) -> Result<HcBatchOutput> {
            ensure!(
                (2..=scratch.capacity).contains(&rows),
                "HC batch supports 2 through 4096 rows"
            );
            ensure!(
                input.byte_len() >= rows * EXPANDED * 2,
                "HC batch residual buffer is too small"
            );
            if let Some((output, injection)) = pending {
                ensure!(
                    output.byte_len() >= rows * WIDTH * 2 && injection.byte_len() >= rows * 4 * 2,
                    "HC pending buffers are too small"
                );
            }
            ensure!(
                output.updated.byte_len() >= rows * EXPANDED * 2
                    && output.mixed.byte_len() >= rows * WIDTH * 2
                    && output.injection.byte_len() >= rows * 4 * 2,
                "HC output workspace is too small"
            );
            let (pending_output, pending_injection) =
                pending.unwrap_or((&scratch.dummy, &scratch.dummy));
            let flags = [
                u32::from(pending.is_some()).to_le_bytes(),
                u32::from(self.injection.is_some()).to_le_bytes(),
            ]
            .concat();
            let updated = &output.updated;
            let normalized = &scratch.normalized;
            let projected_down = &scratch.projected_down;
            let projected_up = &scratch.projected_up;
            let projected_injection = &scratch.projected_injection;
            let injection = &output.injection;
            let lowrank = &scratch.lowrank;
            let mixed = &output.mixed;
            // SAFETY: Fixed row bounds and disjoint allocations cover every
            // kernel index. The batch retains intermediates and orders stages.
            unsafe {
                batch.dispatch_1d_with_inline(
                    &self.norm,
                    &[
                        input,
                        &self.norm_weight,
                        pending_output,
                        pending_injection,
                        normalized,
                        updated,
                        &self.epsilon,
                    ]
                    .map(|buffer| Binding { buffer, offset: 0 }),
                    &[&flags],
                    rows * 640,
                    640,
                )?;
                self.down
                    .encode_rows_into(batch, normalized, projected_down, rows, dense)?;
                if let Some(project) = &self.injection {
                    project.encode_rows_into(
                        batch,
                        normalized,
                        projected_injection,
                        rows,
                        dense,
                    )?;
                }
                batch.dispatch_1d_with_inline(
                    &self.front,
                    &[projected_down, projected_injection, lowrank, injection]
                        .map(|buffer| Binding { buffer, offset: 0 }),
                    &[&flags],
                    rows * LOWRANK,
                    64,
                )?;
                self.up
                    .encode_rows_into(batch, lowrank, projected_up, rows, dense)?;
                batch.dispatch_1d(
                    &self.mix,
                    &[projected_up, normalized, mixed].map(|buffer| Binding { buffer, offset: 0 }),
                    rows * WIDTH,
                    256,
                )?;
            }
            Ok(HcBatchOutput {
                updated: updated.clone(),
                mixed: mixed.clone(),
                injection: self.injection.as_ref().map(|_| injection.clone()),
                normalized: normalized.clone(),
                projected_down: projected_down.clone(),
                projected_up: projected_up.clone(),
                projected_injection: projected_injection.clone(),
                lowrank: lowrank.clone(),
                rows,
            })
        }

        pub fn run_stages(
            &self,
            residual: &[u16],
            pending: Option<(&[u16], &[u16])>,
            rows: usize,
        ) -> Result<HcBatchStages> {
            ensure!(
                (2..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
                "HC batch supports 2 through 4096 rows"
            );
            checked_words(residual, rows * EXPANDED)?;
            if let Some((output, injection)) = pending {
                checked_words(output, rows * WIDTH)?;
                checked_words(injection, rows * 4)?;
            }
            let upload = |words: &[u16]| -> Result<Buffer> {
                let buffer = self.metal.shared_buffer(words.len() * 2)?;
                buffer.write_bytes(
                    0,
                    &words
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                Ok(buffer)
            };
            let input = upload(residual)?;
            let pending = pending
                .map(|(o, i)| Ok::<_, anyhow::Error>((upload(o)?, upload(i)?)))
                .transpose()?;
            let mut batch = self.metal.batch()?;
            // SAFETY: Host inputs were validated and uploaded into disjoint buffers.
            let output = unsafe {
                self.encode(
                    &mut batch,
                    &input,
                    pending.as_ref().map(|(o, i)| (o, i)),
                    rows,
                )?
            };
            batch.submit().wait()?;
            // SAFETY: The producing command completed successfully above.
            unsafe { output.stages() }
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{HcBatch, HcBatchOutput, HcBatchRows, HcBatchScratch, HcBatchStages};

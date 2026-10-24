//! Multirow PLE lookup, projection and convolution state updates.

#[cfg(target_os = "macos")]
mod gpu {
    use std::collections::BTreeMap;

    use anyhow::{Context, Result, ensure};
    use half::bf16;

    use crate::dense_batch::{DenseBatch, DenseBatchScratch};
    use crate::device_status::DeviceStatus;
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline, Stream};
    use crate::model::{MixerKind, QwenConfig};
    use crate::package::{DirectWeight, Package, WeightFormat};
    use crate::ple::{DevicePleState, PleRows, PleState};

    const SOURCE: &str = include_str!("../kernels/metal/ple_batch.metal");
    const NORM_SOURCE: &str = include_str!("../kernels/metal/hc_batch.metal");
    const WIDTH: usize = 2560;
    const BRANCHES: usize = 4;
    const EXPANDED: usize = WIDTH * BRANCHES;
    const HEADS: usize = 16;
    const HISTORY: usize = 9;
    const VOCAB: u32 = 248_320;

    fn words(bytes: &[u8]) -> Vec<u16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect()
    }

    fn upload_words(metal: &Metal, input: &[u16]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(input.len() * 2)?;
        buffer.write_bytes(
            0,
            &input
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        Ok(buffer)
    }

    fn read_words(buffer: &Buffer, count: usize) -> Result<Vec<u16>> {
        let byte_len = count
            .checked_mul(2)
            .context("PLE read byte count overflow")?;
        ensure!(
            buffer.byte_len() >= byte_len,
            "PLE read exceeds completed buffer"
        );
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .context("PLE read byte reservation failed")?;
        bytes.resize(byte_len, 0);
        buffer.read_bytes(0, &mut bytes)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .context("PLE read word reservation failed")?;
        result.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| u16::from_le_bytes(*word)),
        );
        Ok(result)
    }

    fn finite(values: &[u16], label: &str) -> Result<()> {
        ensure!(
            values.iter().all(|word| bf16::from_bits(*word).is_finite()),
            "{label} contains nonfinite BF16 values"
        );
        Ok(())
    }

    fn weight(
        package: &Package,
        name: &str,
        format: WeightFormat,
        shape: &[usize],
    ) -> Result<DirectWeight> {
        let weight = package.direct_weight(name)?;
        ensure!(
            weight.format == format && weight.shape == shape,
            "PLE batch {name} parameter geometry differs"
        );
        if format == WeightFormat::Bf16 {
            finite(&words(&weight.bytes), name)?;
        }
        Ok(weight)
    }

    struct Scratch {
        key_projected: Buffer,
        key: Buffer,
        value: Buffer,
        query: Buffer,
        dot: Buffer,
        gate: Buffer,
        gain: Buffer,
        flat: Buffer,
        normalized: Buffer,
        conv: Buffer,
        output: Buffer,
        next_conv: Buffer,
        norm_updated: Buffer,
    }

    impl Scratch {
        fn new(metal: &Metal, rows: usize) -> Result<Self> {
            let matrix = |width| metal.shared_buffer(rows * width * 2);
            Ok(Self {
                key_projected: matrix(EXPANDED)?,
                key: matrix(EXPANDED)?,
                value: matrix(WIDTH)?,
                query: matrix(EXPANDED)?,
                dot: matrix(BRANCHES)?,
                gate: matrix(BRANCHES)?,
                gain: matrix(BRANCHES)?,
                flat: matrix(EXPANDED)?,
                normalized: matrix(EXPANDED)?,
                conv: matrix(EXPANDED)?,
                output: matrix(EXPANDED)?,
                next_conv: metal.shared_buffer(HISTORY * EXPANDED * 2)?,
                norm_updated: matrix(EXPANDED)?,
            })
        }

        fn stages(
            &self,
            rows: usize,
            embedding: &Buffer,
            hidden: &Buffer,
            previous: &Buffer,
        ) -> Result<BTreeMap<&'static str, Vec<u16>>> {
            Ok(BTreeMap::from([
                ("hidden", read_words(hidden, rows * EXPANDED)?),
                ("previous_conv", read_words(previous, HISTORY * EXPANDED)?),
                ("embedding", read_words(embedding, rows * WIDTH)?),
                (
                    "key_projection",
                    read_words(&self.key_projected, rows * EXPANDED)?,
                ),
                ("key", read_words(&self.key, rows * EXPANDED)?),
                ("value", read_words(&self.value, rows * WIDTH)?),
                ("query", read_words(&self.query, rows * EXPANDED)?),
                ("gate_dot", read_words(&self.dot, rows * BRANCHES)?),
                ("gate", read_words(&self.gate, rows * BRANCHES)?),
                ("flat", read_words(&self.flat, rows * EXPANDED)?),
                ("normalized", read_words(&self.normalized, rows * EXPANDED)?),
                ("conv", read_words(&self.conv, rows * EXPANDED)?),
                ("output", read_words(&self.output, rows * EXPANDED)?),
                (
                    "conv_state",
                    read_words(&self.next_conv, HISTORY * EXPANDED)?,
                ),
            ]))
        }
    }

    /// Reused across PLE layers and chunks on one ordered execution stream.
    pub struct PleBatchScratch {
        capacity: usize,
        embedding: Buffer,
        key_projected: Buffer,
        value: Buffer,
        gated: Buffer,
        normed: Buffer,
        output: Buffer,
    }

    impl PleBatchScratch {
        pub fn new(metal: &Metal, capacity: usize) -> Result<Self> {
            ensure!(
                (2..=crate::prefill_capacity::MAX_ROWS).contains(&capacity),
                "PLE workspace capacity is invalid"
            );
            let matrix = |width| metal.shared_buffer(capacity * width * 2);
            Ok(Self {
                capacity,
                embedding: matrix(WIDTH)?,
                key_projected: matrix(EXPANDED)?,
                value: matrix(WIDTH)?,
                gated: matrix(EXPANDED)?,
                normed: matrix(EXPANDED)?,
                output: matrix(EXPANDED)?,
            })
        }

        pub fn byte_len(&self) -> usize {
            self.embedding.byte_len()
                + self.key_projected.byte_len()
                + self.value.byte_len()
                + self.gated.byte_len()
                + self.normed.byte_len()
                + self.output.byte_len()
        }
    }

    /// One complete multirow PLE result. Its state is a candidate until the
    /// calling layer or model chunk commits it.
    pub struct PleBatchResult {
        pub output_bf16: Vec<u16>,
        pub state: PleState,
        pub row_ids: Vec<[u64; HEADS]>,
        pub stages: Option<BTreeMap<&'static str, Vec<u16>>>,
    }

    /// GPU output and candidate metadata retained by a caller-owned batch.
    /// Finish it only after that batch completes successfully.
    pub struct PleBatchEncoded {
        scratch: Scratch,
        hidden: Buffer,
        embedding: Buffer,
        previous: Buffer,
        context: [u32; 2],
        frontier: u64,
        row_ids: Vec<[u64; HEADS]>,
        rows: usize,
    }

    impl PleBatchEncoded {
        /// The BF16 `[rows,10240]` residual output for ordered GPU consumers.
        pub fn output_buffer(&self) -> &Buffer {
            &self.scratch.output
        }

        /// The BF16 nine-row candidate convolution history.
        pub fn conv_state_buffer(&self) -> &Buffer {
            &self.scratch.next_conv
        }

        /// Read the completed result. Reserved buffers fail closed before wait.
        ///
        /// # Safety
        /// The command batch containing this encoding must have been submitted
        /// and must have completed successfully. Dropping an unsubmitted batch
        /// also releases buffer reservations, but does not initialize outputs.
        pub unsafe fn finish(self, capture_stages: bool) -> Result<PleBatchResult> {
            let output_bf16 = read_words(&self.scratch.output, self.rows * EXPANDED)?;
            let next_conv = read_words(&self.scratch.next_conv, HISTORY * EXPANDED)?;
            let stages = capture_stages
                .then(|| {
                    self.scratch
                        .stages(self.rows, &self.embedding, &self.hidden, &self.previous)
                })
                .transpose()?;
            Ok(PleBatchResult {
                output_bf16,
                state: PleState {
                    context: self.context,
                    conv: next_conv,
                    offset: self.frontier,
                },
                row_ids: self.row_ids,
                stages,
            })
        }
    }

    pub struct PleBatch {
        metal: Metal,
        rows: PleRows,
        key: DenseBatch,
        value: DenseBatch,
        norm_key: Buffer,
        norm_query: Buffer,
        norm_conv: Buffer,
        conv_weight: Buffer,
        epsilon: Buffer,
        flags: Buffer,
        dummy: Buffer,
        norm: Pipeline,
        dot: Pipeline,
        gate: Pipeline,
        flat: Pipeline,
        conv: Pipeline,
        fused_gate: Pipeline,
        fused_conv: Pipeline,
        max_positions: usize,
    }

    impl PleBatch {
        pub fn new(metal: &Metal, package: &Package) -> Result<Self> {
            Self::build(metal, package, None)
        }
        pub(crate) fn from_decode(
            metal: &Metal,
            package: &Package,
            source: &crate::ple::Qwen4Ple,
        ) -> Result<Self> {
            Self::build(metal, package, Some(source.shared_matrices()))
        }
        fn build(
            metal: &Metal,
            package: &Package,
            shared: Option<[crate::kquant::DeviceMatrix; 2]>,
        ) -> Result<Self> {
            let config = QwenConfig::from_package(package)?;
            ensure!(
                config.ple_layer_index == 1 && config.layers[1] == MixerKind::Gdn,
                "PLE batch must precede Qwen layer 1 GDN"
            );
            let rows = PleRows::new(package)?;
            let matrix = |slot: usize, name: &str, shape: [usize; 2]| -> Result<DenseBatch> {
                if let Some(ref matrices) = shared {
                    matrices[slot].check(WeightFormat::Q6K, shape)?;
                    DenseBatch::from_matrix(matrices[slot].clone())
                } else {
                    DenseBatch::new(metal, &weight(package, name, WeightFormat::Q6K, &shape)?)
                }
            };
            let key = matrix(0, "layers.1.ple.key_proj", [EXPANDED, WIDTH])?;
            let value = matrix(1, "layers.1.ple.value_proj", [WIDTH, WIDTH])?;
            let norm_key = weight(
                package,
                "layers.1.ple.norm_key",
                WeightFormat::Bf16,
                &[EXPANDED],
            )?;
            let norm_query = weight(
                package,
                "layers.1.ple.norm_query",
                WeightFormat::Bf16,
                &[EXPANDED],
            )?;
            let norm_conv = weight(
                package,
                "layers.1.ple.norm_conv",
                WeightFormat::Bf16,
                &[EXPANDED],
            )?;
            let conv = weight(
                package,
                "layers.1.ple.conv1d",
                WeightFormat::Bf16,
                &[EXPANDED, 1, 4],
            )?;
            let source = format!("{NORM_SOURCE}\n{SOURCE}");
            let pipeline = |name: &str, threads: usize| -> Result<Pipeline> {
                let pipeline = metal.compile_msl_strict(&source, name)?;
                ensure!(
                    pipeline.max_threads_per_threadgroup() >= threads,
                    "PLE batch threadgroup unsupported"
                );
                Ok(pipeline)
            };
            let epsilon = metal.shared_buffer(4)?;
            epsilon.write_bytes(0, &1e-6f32.to_le_bytes())?;
            let flags = metal.shared_buffer(8)?;
            flags.write_bytes(0, &[0; 8])?;
            Ok(Self {
                metal: metal.clone(),
                rows,
                key,
                value,
                norm_key: upload_words(metal, &words(&norm_key.bytes))?,
                norm_query: upload_words(metal, &words(&norm_query.bytes))?,
                norm_conv: upload_words(metal, &words(&norm_conv.bytes))?,
                conv_weight: upload_words(metal, &words(&conv.bytes))?,
                epsilon,
                flags,
                dummy: metal.shared_buffer(2)?,
                norm: pipeline("hc_batch_norm", 640)?,
                dot: pipeline("ple_batch_gate_dot", 640)?,
                gate: pipeline("ple_batch_gate_scalar", 64)?,
                flat: pipeline("ple_batch_flat", 256)?,
                conv: pipeline("ple_batch_conv", 128)?,
                fused_gate: pipeline("ple_prefill_fused_gate", 256)?,
                fused_conv: pipeline("ple_prefill_fused_conv_residual", 128)?,
                max_positions: config.max_positions,
            })
        }

        pub fn dense_scratch_bytes(&self, capacity: usize) -> Result<usize> {
            Ok(self
                .key
                .scratch_bytes(capacity)?
                .max(self.value.scratch_bytes(capacity)?))
        }

        /// Encode a PLE chunk with canonical device history and a reusable
        /// workspace. The output includes the expanded residual.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn encode_chunk(
            &mut self,
            stream: &mut Stream,
            state: &mut DevicePleState,
            tokens: &[u32],
            hidden: &Buffer,
            scratch: &mut PleBatchScratch,
            dense: &DenseBatchScratch,
            status: &DeviceStatus,
        ) -> Result<Buffer> {
            let rows = tokens.len();
            ensure!(
                (2..=scratch.capacity).contains(&rows),
                "PLE chunk size is invalid"
            );
            ensure!(
                hidden.byte_len() >= rows * EXPANDED * 2,
                "PLE chunk hidden buffer is too small"
            );
            ensure!(
                state.conv_byte_len() == HISTORY * EXPANDED * 2,
                "PLE device history geometry differs"
            );
            let frontier = state
                .offset
                .checked_add(rows as u64)
                .context("PLE frontier overflow")?;
            ensure!(
                frontier <= self.max_positions as u64,
                "PLE chunk exceeds model context limit"
            );
            ensure!(
                state.context.iter().all(|&token| token < VOCAB),
                "PLE token context is invalid"
            );

            let mut embedding = Vec::new();
            embedding
                .try_reserve_exact(rows * WIDTH)
                .context("PLE row staging reservation failed")?;
            let mut context = state.context;
            for &token in tokens {
                let ids = self.rows.row_ids(token, context)?;
                embedding.extend(self.rows.lookup(&ids)?);
                context = [context[1], token];
            }
            ensure!(
                embedding.len() == rows * WIDTH,
                "PLE selected rows have invalid width"
            );
            finite(&embedding, "PLE selected rows")?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(embedding.len() * 2)
                .context("PLE row upload reservation failed")?;
            bytes.extend(embedding.iter().flat_map(|word| word.to_le_bytes()));
            scratch.embedding.write_bytes(0, &bytes)?;

            let batch = stream.batch()?;
            // SAFETY: The staged embedding has a checked active extent. Each
            // projection and fused consumer is ordered within this stream.
            unsafe {
                self.key.encode_rows_into(
                    batch,
                    &scratch.embedding,
                    &scratch.key_projected,
                    rows,
                    dense,
                )?;
                self.value.encode_rows_into(
                    batch,
                    &scratch.embedding,
                    &scratch.value,
                    rows,
                    dense,
                )?;
            }
            let gate = [
                hidden,
                &scratch.key_projected,
                &scratch.value,
                &self.norm_key,
                &self.norm_query,
                &self.norm_conv,
                &scratch.gated,
                &scratch.normed,
                &self.epsilon,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: One 256-thread group owns each active row. Its four
            // branches write disjoint 2560-cell regions.
            unsafe {
                batch.dispatch_1d(&self.fused_gate, &gate, rows * 256, 256)?;
            }
            let conv = [
                hidden,
                &scratch.gated,
                &scratch.normed,
                state.device_history(),
                &self.conv_weight,
                &scratch.output,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            let rows_inline = (rows as u32).to_le_bytes();
            // SAFETY: Each channel owns distinct history cells and output
            // columns. The group reads all old history into thread-local
            // storage before writing the same canonical allocation.
            unsafe {
                batch.dispatch_1d_with_inline(
                    &self.fused_conv,
                    &conv,
                    &[&rows_inline],
                    EXPANDED,
                    128,
                )?;
            }
            status.encode_bf16_count(batch, &scratch.output, rows * EXPANDED)?;
            status.encode_bf16(batch, state.device_history())?;
            state.advance_chunk_in_place(tokens)?;
            Ok(scratch.output.clone())
        }

        fn encode_norm(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            weight: &Buffer,
            output: &Buffer,
            updated: &Buffer,
            rows: usize,
        ) -> Result<()> {
            let bindings = [
                input,
                weight,
                &self.dummy,
                &self.dummy,
                output,
                updated,
                &self.epsilon,
                &self.flags,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: The 640-thread groups cover four 2560-wide streams per
            // row. Pending reads are disabled; outputs have distinct storage.
            unsafe { batch.dispatch_1d(&self.norm, &bindings, rows * 640, 640) }
        }

        pub fn run(
            &mut self,
            state: &PleState,
            tokens: &[u32],
            hidden_bf16: &[u16],
        ) -> Result<PleBatchResult> {
            self.execute(state, tokens, hidden_bf16, false)
        }

        pub fn run_stages(
            &mut self,
            state: &PleState,
            tokens: &[u32],
            hidden_bf16: &[u16],
        ) -> Result<PleBatchResult> {
            self.execute(state, tokens, hidden_bf16, true)
        }

        fn execute(
            &mut self,
            state: &PleState,
            tokens: &[u32],
            hidden_bf16: &[u16],
            capture: bool,
        ) -> Result<PleBatchResult> {
            let rows = tokens.len();
            ensure!(
                (2..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
                "PLE batch row count exceeds supported prefill capacity"
            );
            ensure!(
                hidden_bf16.len() == rows * EXPANDED,
                "PLE batch hidden geometry differs"
            );
            finite(hidden_bf16, "PLE batch hidden")?;
            let hidden = upload_words(&self.metal, hidden_bf16)?;
            let mut batch = self.metal.batch()?;
            // SAFETY: Host values were checked finite and uploaded to a
            // disjoint buffer; every previous producer is complete.
            let encoded = unsafe { self.encode(&mut batch, &hidden, tokens, state)? };
            batch.submit().wait().context("PLE batch command")?;
            // SAFETY: A successful wait above completed every encoded writer.
            unsafe { encoded.finish(capture) }
        }

        /// Encode one multirow PLE candidate into an ordered caller batch.
        /// The returned output buffer may feed later GPU work in that batch.
        /// Finish the candidate only after successful batch completion.
        ///
        /// # Safety
        /// `hidden` must contain finite BF16 `[rows,10240]` values, must not
        /// alias PLE parameters or state, and its producers must precede this
        /// encoding. The caller must not modify bound buffers until completion.
        pub unsafe fn encode(
            &mut self,
            batch: &mut Batch,
            hidden: &Buffer,
            tokens: &[u32],
            state: &PleState,
        ) -> Result<PleBatchEncoded> {
            let rows = tokens.len();
            ensure!(
                (2..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
                "PLE batch row count exceeds supported prefill capacity"
            );
            ensure!(
                hidden.byte_len() >= rows * EXPANDED * 2,
                "PLE batch hidden buffer is too small"
            );
            ensure!(
                state.conv.len() == HISTORY * EXPANDED,
                "PLE batch convolution history differs"
            );
            ensure!(
                state.context.iter().all(|&token| token < VOCAB),
                "PLE batch token context is invalid"
            );
            let frontier = state
                .offset
                .checked_add(rows as u64)
                .context("PLE batch frontier overflow")?;
            ensure!(
                frontier <= self.max_positions as u64,
                "PLE batch exceeds model context limit"
            );
            finite(&state.conv, "PLE batch convolution history")?;
            let mut row_ids = Vec::with_capacity(rows);
            let mut embedding = Vec::with_capacity(rows * WIDTH);
            let mut context = state.context;
            for &token in tokens {
                let selected = self.rows.row_ids(token, context)?;
                embedding.extend(self.rows.lookup(&selected)?);
                row_ids.push(selected);
                context = [context[1], token];
            }
            ensure!(
                embedding.len() == rows * WIDTH,
                "PLE selected rows have invalid width"
            );
            finite(&embedding, "PLE selected rows")?;
            let previous = upload_words(&self.metal, &state.conv)?;
            let embeddings = upload_words(&self.metal, &embedding)?;
            let scratch = Scratch::new(&self.metal, rows)?;
            let count = self.metal.shared_buffer(4)?;
            count.write_bytes(0, &(rows as u32).to_le_bytes())?;
            // SAFETY: All input and output buffers have checked fixed extents;
            // each producer precedes its consumer in this one command batch.
            unsafe {
                self.key
                    .encode_rows(batch, &embeddings, &scratch.key_projected, rows)?;
                self.value
                    .encode_rows(batch, &embeddings, &scratch.value, rows)?;
            }
            self.encode_norm(
                batch,
                &scratch.key_projected,
                &self.norm_key,
                &scratch.key,
                &scratch.norm_updated,
                rows,
            )?;
            self.encode_norm(
                batch,
                hidden,
                &self.norm_query,
                &scratch.query,
                &scratch.norm_updated,
                rows,
            )?;
            let dot = [&scratch.key, &scratch.query, &scratch.dot]
                .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: One 640-thread group reduces one row/branch; the grid
            // contains exactly rows*4 groups and writes distinct dot cells.
            unsafe {
                batch.dispatch_1d(&self.dot, &dot, rows * BRANCHES * 640, 640)?;
            }
            let gate = [&scratch.dot, &scratch.gate, &scratch.gain]
                .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: Each gate index is in the rows*4-sized buffers.
            unsafe {
                batch.dispatch_1d(&self.gate, &gate, rows * BRANCHES, 64)?;
            }
            let flat = [&scratch.gain, &scratch.value, &scratch.flat]
                .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: Each flattened cell reads one valid gain/value and
            // writes its own BF16 output cell.
            unsafe {
                batch.dispatch_1d(&self.flat, &flat, rows * EXPANDED, 256)?;
            }
            self.encode_norm(
                batch,
                &scratch.flat,
                &self.norm_conv,
                &scratch.normalized,
                &scratch.norm_updated,
                rows,
            )?;
            let conv = [
                &previous,
                &scratch.normalized,
                &self.conv_weight,
                &scratch.flat,
                &scratch.conv,
                &scratch.output,
                &scratch.next_conv,
                &count,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: max(rows,9) initializes every candidate-history cell;
            // rows groups independently compute 4-tap dilated convolution.
            unsafe {
                batch.dispatch_1d(&self.conv, &conv, rows.max(HISTORY) * EXPANDED, 128)?;
            }
            Ok(PleBatchEncoded {
                scratch,
                hidden: hidden.clone(),
                embedding: embeddings,
                previous,
                context,
                frontier,
                row_ids,
                rows,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn row_input(metal: &Metal, values: &[f32]) -> Result<Buffer> {
            let mut words = vec![0u16; values.len() * EXPANDED];
            for (row, &value) in values.iter().enumerate() {
                words[row * EXPANDED] = bf16::from_f32(value).to_bits();
            }
            upload_words(metal, &words)
        }

        fn history(metal: &Metal, values: &[f32; HISTORY]) -> Result<Buffer> {
            let mut words = vec![0u16; HISTORY * EXPANDED];
            for (row, &value) in values.iter().enumerate() {
                words[row * EXPANDED] = bf16::from_f32(value).to_bits();
            }
            upload_words(metal, &words)
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_conv(
            batch: &mut Batch,
            kernel: &Pipeline,
            residual: &Buffer,
            gated: &Buffer,
            normed: &Buffer,
            history: &Buffer,
            weight: &Buffer,
            output: &Buffer,
            rows: usize,
        ) -> Result<()> {
            let bindings = [residual, gated, normed, history, weight, output]
                .map(|buffer| Binding { buffer, offset: 0 });
            let count = (rows as u32).to_le_bytes();
            // SAFETY: The test allocates complete independent rows and each
            // channel updates only its own nine history cells.
            unsafe { batch.dispatch_1d_with_inline(kernel, &bindings, &[&count], EXPANDED, 128) }
        }

        #[test]
        fn fused_conv_advances_repeated_and_interleaved_histories() -> Result<()> {
            let metal = Metal::new()?;
            let source = format!("{NORM_SOURCE}\n{SOURCE}");
            let kernel = metal.compile_msl_strict(&source, "ple_prefill_fused_conv_residual")?;
            let mut weight = vec![0u16; EXPANDED * 4];
            weight[0] = bf16::from_f32(1.0).to_bits();
            let weight = upload_words(&metal, &weight)?;
            let a_history_values = std::array::from_fn(|row| -0.75 + row as f32 * 0.125);
            let b_history_values = std::array::from_fn(|row| 1.0 + row as f32 * 0.125);
            let a_history = history(&metal, &a_history_values)?;
            let b_history = history(&metal, &b_history_values)?;
            let a_first = (1..=9).map(|i| i as f32 * 0.125).collect::<Vec<_>>();
            let a_second = [1.375, 1.5];
            let b_first = [0.25, 0.5, 0.75];
            let a_first_normed = row_input(&metal, &a_first)?;
            let a_second_normed = row_input(&metal, &a_second)?;
            let b_first_normed = row_input(&metal, &b_first)?;
            let a_first_zero = row_input(&metal, &[0.0; 9])?;
            let a_second_zero = row_input(&metal, &[0.0; 2])?;
            let b_first_zero = row_input(&metal, &[0.0; 3])?;
            let a_first_output = metal.shared_buffer(9 * EXPANDED * 2)?;
            let a_second_output = metal.shared_buffer(2 * EXPANDED * 2)?;
            let b_first_output = metal.shared_buffer(3 * EXPANDED * 2)?;
            let mut batch = metal.batch()?;
            encode_conv(
                &mut batch,
                &kernel,
                &a_first_zero,
                &a_first_zero,
                &a_first_normed,
                &a_history,
                &weight,
                &a_first_output,
                9,
            )?;
            encode_conv(
                &mut batch,
                &kernel,
                &b_first_zero,
                &b_first_zero,
                &b_first_normed,
                &b_history,
                &weight,
                &b_first_output,
                3,
            )?;
            encode_conv(
                &mut batch,
                &kernel,
                &a_second_zero,
                &a_second_zero,
                &a_second_normed,
                &a_history,
                &weight,
                &a_second_output,
                2,
            )?;
            batch.submit().wait()?;

            let silu = |x: f32| x / (1.0 + (-x).exp());
            for (output, expected) in [
                (&a_first_output, a_history_values.as_slice()),
                (&b_first_output, &b_history_values[..3]),
                (&a_second_output, &a_first[..2]),
            ] {
                for (row, &value) in expected.iter().enumerate() {
                    let mut bytes = [0u8; 2];
                    output.read_bytes(row * EXPANDED * 2, &mut bytes)?;
                    let actual = bf16::from_bits(u16::from_le_bytes(bytes)).to_f32();
                    let reference = silu(value);
                    let tolerance = reference.abs().max(1.0) * (1.0 / 128.0);
                    assert!(
                        (actual - reference).abs() <= tolerance,
                        "PLE row {row}: {actual} differs from {reference}"
                    );
                }
            }
            for (buffer, expected) in [
                (&a_history, [&a_first[2..], &a_second].concat()),
                (&b_history, [&b_history_values[3..], &b_first].concat()),
            ] {
                for (row, &value) in expected.iter().enumerate() {
                    let mut bytes = [0u8; 2];
                    buffer.read_bytes(row * EXPANDED * 2, &mut bytes)?;
                    assert_eq!(u16::from_le_bytes(bytes), bf16::from_f32(value).to_bits());
                }
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{PleBatch, PleBatchEncoded, PleBatchResult, PleBatchScratch};

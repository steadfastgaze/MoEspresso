//! Bounded selected-row attention tiles followed by a multirow output projection.

#[cfg(target_os = "macos")]
mod gpu {
    use crate::dense_batch::{DenseBatch, DenseBatchScratch};
    use crate::kvarn_pending::{PendingKvarnRows, ReconstructedKvarnRows};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use crate::package::{Package, WeightFormat};
    use crate::qsa_batch_projection::QsaBatchProjected;
    use crate::qsa_index::QsaChunkIndexScratch;
    use anyhow::{Result, ensure};
    use std::collections::BTreeMap;

    const SELECTED: usize = 2051;
    const HIDDEN: usize = 2560;
    const EXPANDED: usize = 6144;
    const SELECTED_BYTES: usize = 2 * SELECTED * 256 * 2;
    const SOURCE: &str = include_str!("../kernels/metal/qsa_attention.metal");

    pub struct QsaBatchAttentionOutput {
        pub attention: Buffer,
        pub sigmoid: Buffer,
        pub gated: Buffer,
        pub output: Buffer,
        rows: usize,
    }

    impl QsaBatchAttentionOutput {
        pub fn new(metal: &Metal, capacity: usize) -> Result<Self> {
            ensure!(
                (1..=crate::prefill_capacity::MAX_ROWS).contains(&capacity),
                "QSA output capacity differs"
            );
            Ok(Self {
                attention: metal.shared_buffer(capacity * EXPANDED * 2)?,
                sigmoid: metal.shared_buffer(capacity * EXPANDED * 2)?,
                gated: metal.shared_buffer(capacity * EXPANDED * 2)?,
                output: metal.shared_buffer(capacity * HIDDEN * 2)?,
                rows: capacity,
            })
        }

        pub fn byte_len(&self) -> usize {
            self.attention.byte_len()
                + self.sigmoid.byte_len()
                + self.gated.byte_len()
                + self.output.byte_len()
        }

        fn new_device(metal: &Metal, capacity: usize) -> Result<Self> {
            ensure!(
                (1..=crate::prefill_capacity::MAX_ROWS).contains(&capacity),
                "QSA output capacity differs"
            );
            Ok(Self {
                attention: metal.shared_buffer(2)?,
                sigmoid: metal.shared_buffer(2)?,
                gated: metal.shared_buffer(capacity * EXPANDED * 2)?,
                output: metal.shared_buffer(capacity * HIDDEN * 2)?,
                rows: capacity,
            })
        }

        /// Read diagnostic arrays after every tile and projection completed.
        ///
        /// # Safety
        /// Every row must have been encoded exactly once and the producing
        /// commands successfully submitted and waited before this call.
        pub unsafe fn stages(&self) -> Result<BTreeMap<&'static str, Vec<u8>>> {
            [
                ("attention", &self.attention),
                ("sigmoid", &self.sigmoid),
                ("gated", &self.gated),
                ("output", &self.output),
            ]
            .into_iter()
            .map(|(name, buffer)| {
                let mut bytes = vec![0; buffer.byte_len()];
                buffer.read_bytes(0, &mut bytes)?;
                Ok((name, bytes))
            })
            .collect()
        }
    }

    /// One bounded QSA prefill workspace reused across all sparse layers.
    pub struct QsaPrefillScratch {
        pub projected: QsaBatchProjected,
        pub index: QsaChunkIndexScratch,
        pub selected: PendingKvarnRows,
        pub effective_valid: Buffer,
        pub safe_counts: Buffer,
        pub online_partials: Buffer,
        pub output: QsaBatchAttentionOutput,
    }

    impl QsaPrefillScratch {
        pub fn new(metal: &Metal, capacity: usize, max_positions: usize) -> Result<Self> {
            Ok(Self {
                projected: QsaBatchProjected::new(metal, capacity)?,
                index: QsaChunkIndexScratch::new(metal, capacity, max_positions)?,
                selected: PendingKvarnRows::new(metal, 64)?,
                effective_valid: metal.shared_buffer(64 * SELECTED)?,
                safe_counts: metal.shared_buffer(64 * 4)?,
                online_partials: metal.shared_buffer(24 * 64 * (2 + 256) * 4)?,
                output: QsaBatchAttentionOutput::new_device(metal, capacity)?,
            })
        }

        pub fn byte_len(&self) -> usize {
            self.projected.byte_len()
                + self.index.byte_len()
                + self.selected.byte_len()
                + self.effective_valid.byte_len()
                + self.safe_counts.byte_len()
                + self.online_partials.byte_len()
                + self.output.byte_len()
        }
    }

    pub struct QsaBatchAttention {
        metal: Metal,
        projection: DenseBatch,
        logits: Pipeline,
        softmax: Pipeline,
        value_gate: Pipeline,
        online_mm: Pipeline,
        indexed_mm: Pipeline,
        online_tile: Pipeline,
        online_merge: Pipeline,
    }

    impl QsaBatchAttention {
        pub(crate) fn from_decode(
            metal: &Metal,
            source: &crate::qsa_attention::QsaAttention,
        ) -> Result<Self> {
            Self::from_projection(metal, DenseBatch::from_matrix(source.shared_matrix())?)
        }
        pub fn new(metal: &Metal, package: &Package, layer: usize) -> Result<Self> {
            let config = crate::model::QwenConfig::from_package(package)?;
            ensure!(
                config.layers.get(layer) == Some(&crate::model::MixerKind::Qsa),
                "attention layer is not QSA"
            );
            let weight = package.direct_weight(&format!("layers.{layer}.mixer.module.o_proj"))?;
            ensure!(
                weight.format == WeightFormat::Q6K && weight.shape == [HIDDEN, EXPANDED],
                "QSA output projection geometry differs"
            );
            Self::from_projection(metal, DenseBatch::new(metal, &weight)?)
        }
        fn from_projection(metal: &Metal, projection: DenseBatch) -> Result<Self> {
            let logits = metal.compile_msl_strict(SOURCE, "qsa_selected_logits")?;
            let softmax = metal.compile_msl_strict(SOURCE, "qsa_selected_softmax")?;
            let value_gate = metal.compile_msl_strict(SOURCE, "qsa_selected_value_gate")?;
            let online_mm = metal.compile_msl_strict(SOURCE, "qsa_prefill_online_mm")?;
            let indexed_mm = metal.compile_msl_strict(SOURCE, "qsa_prefill_indexed_mm")?;
            let online_tile = metal.compile_msl_strict(
                include_str!("../kernels/metal/qsa_online.metal"),
                "qsa_online_tile",
            )?;
            let online_merge = metal.compile_msl_strict(
                include_str!("../kernels/metal/qsa_online.metal"),
                "qsa_online_merge",
            )?;
            ensure!(
                logits.max_threads_per_threadgroup() >= 128
                    && softmax.max_threads_per_threadgroup() >= 544
                    && value_gate.max_threads_per_threadgroup() >= 128,
                "QSA attention threadgroup unsupported"
            );
            Ok(Self {
                metal: metal.clone(),
                projection,
                logits,
                softmax,
                value_gate,
                online_mm,
                indexed_mm,
                online_tile,
                online_merge,
            })
        }

        /// Run the DS4 matrix online path over a gathered multiquery tile.
        /// Short tails retain the established decode split-online arithmetic.
        ///
        /// # Safety
        /// The query, gate, selected K/V, validity and counts are complete or
        /// produced earlier on this stream. The output rows and partials have
        /// no unordered readers, and status is a sticky zero-initialized flag.
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn encode_tile_online(
            &self,
            batch: &mut Batch,
            queries: &Buffer,
            gates: &Buffer,
            selected: &PendingKvarnRows,
            effective_valid: &Buffer,
            safe_counts: &Buffer,
            tile_offset: usize,
            rows: usize,
            output: &QsaBatchAttentionOutput,
            status: &Buffer,
            online_partials: &Buffer,
        ) -> Result<()> {
            ensure!(
                (1..=64).contains(&rows)
                    && rows <= selected.queries
                    && tile_offset
                        .checked_add(rows)
                        .is_some_and(|end| end <= output.rows),
                "QSA online tile exceeds output capacity"
            );
            ensure!(
                queries.byte_len() >= (tile_offset + rows) * EXPANDED * 2
                    && gates.byte_len() >= (tile_offset + rows) * EXPANDED * 2
                    && selected.keys.byte_len() >= rows * SELECTED_BYTES
                    && selected.values.byte_len() >= rows * SELECTED_BYTES
                    && effective_valid.byte_len() >= rows * SELECTED
                    && safe_counts.byte_len() >= rows * 4
                    && status.byte_len() >= 4
                    && online_partials.byte_len() >= 24 * 64 * (2 + 256) * 4,
                "QSA online tile geometry differs"
            );
            let q_offset = tile_offset * EXPANDED * 2;
            if rows > 8 {
                let count = self.metal.shared_buffer(8)?;
                count.write_bytes(
                    0,
                    &[u32::try_from(rows)?, 0]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                let bindings = [
                    (queries, q_offset),
                    (gates, q_offset),
                    (&selected.keys, 0),
                    (&selected.values, 0),
                    (effective_valid, 0),
                    (safe_counts, 0),
                    // Capture is disabled; these are two-byte placeholders.
                    (&output.attention, 0),
                    (&output.sigmoid, 0),
                    (&output.gated, q_offset),
                    (status, 0),
                    (&count, 0),
                ]
                .map(|(buffer, offset)| Binding { buffer, offset });
                // SAFETY: One 128-thread group owns each query/KV-head pair;
                // every output row is disjoint and bounded by the preflight.
                unsafe { batch.dispatch_1d(&self.online_mm, &bindings, rows * 2 * 128, 128)? };
            } else {
                for row in 0..rows {
                    let q = q_offset + row * EXPANDED * 2;
                    let selected_offset = row * SELECTED_BYTES;
                    let bindings = [
                        (queries, q),
                        (&selected.keys, selected_offset),
                        (&selected.values, selected_offset),
                        (effective_valid, row * SELECTED),
                        (safe_counts, row * 4),
                        (online_partials, 0),
                    ]
                    .map(|(buffer, offset)| Binding { buffer, offset });
                    // SAFETY: Reused partials are overwritten before the next
                    // row's merge and no other command uses them concurrently.
                    unsafe { batch.dispatch_1d(&self.online_tile, &bindings, 64 * 2 * 128, 128)? };
                    let bindings = [
                        (online_partials, 0),
                        (gates, q),
                        (safe_counts, row * 4),
                        (&output.gated, q),
                    ]
                    .map(|(buffer, offset)| Binding { buffer, offset });
                    unsafe { batch.dispatch_1d(&self.online_merge, &bindings, 24 * 32, 32)? };
                }
            }
            Ok(())
        }

        /// Native arithmetic with logical-ID addressing into ordered reconstructed rows.
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn encode_tile_reconstructed(
            &self,
            batch: &mut Batch,
            queries: &Buffer,
            gates: &Buffer,
            raw: &ReconstructedKvarnRows<'_>,
            valid: &Buffer,
            counts: &Buffer,
            ids: &Buffer,
            tile_offset: usize,
            rows: usize,
            output: &QsaBatchAttentionOutput,
            status: &Buffer,
            deferred: Option<(&Pipeline, &Buffer)>,
        ) -> Result<()> {
            let frontier = raw.frontier;
            let reconstructed_rows = raw.total_rows;
            let end = tile_offset
                .checked_add(rows)
                .ok_or_else(|| anyhow::anyhow!("indexed tile overflow"))?;
            let visible = frontier
                .checked_add(end)
                .ok_or_else(|| anyhow::anyhow!("indexed frontier overflow"))?;
            let query_bytes = end
                .checked_mul(EXPANDED * 2)
                .ok_or_else(|| anyhow::anyhow!("indexed query extent overflow"))?;
            let ids_bytes = end
                .checked_mul(2051 * 4)
                .ok_or_else(|| anyhow::anyhow!("indexed selector extent overflow"))?;
            ensure!(
                (9..=64).contains(&rows) && end <= output.rows,
                "indexed attention requires a matrix tile"
            );
            ensure!(
                crate::kvarn_pending::reconstruction_fits(
                    0,
                    reconstructed_rows,
                    raw.keys.byte_len(),
                    raw.values.byte_len()
                ) && visible <= reconstructed_rows
                    && queries.byte_len() >= query_bytes
                    && gates.byte_len() >= query_bytes
                    && ids.byte_len() >= ids_bytes
                    && valid.byte_len() >= rows * 2051
                    && counts.byte_len() >= rows * 4
                    && status.byte_len() >= 4,
                "indexed attention extent differs"
            );
            let count = self.metal.shared_buffer(8)?;
            count.write_bytes(
                0,
                &[u32::try_from(rows)?, 0]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let geometry = self.metal.shared_buffer(12)?;
            geometry.write_bytes(
                0,
                &[
                    u32::try_from(frontier)?,
                    u32::try_from(tile_offset)?,
                    u32::try_from(reconstructed_rows)?,
                ]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
            )?;
            let offset = tile_offset * EXPANDED * 2;
            let (pipeline, destination, destination_offset) =
                if let Some((pipeline, buffer)) = deferred {
                    ensure!(
                        buffer.byte_len() >= end * EXPANDED * 4,
                        "deferred QSA FP32 extent differs"
                    );
                    (pipeline, buffer, tile_offset * EXPANDED * 4)
                } else {
                    (&self.indexed_mm, &output.gated, offset)
                };
            let bindings = [
                (queries, offset),
                (gates, offset),
                (raw.keys, 0),
                (raw.values, 0),
                (valid, 0),
                (counts, 0),
                (&output.attention, 0),
                (&output.sigmoid, 0),
                (destination, destination_offset),
                (status, 0),
                (&count, 0),
                (ids, tile_offset * 2051 * 4),
                (&geometry, 0),
            ]
            .map(|(buffer, offset)| Binding { buffer, offset });
            // SAFETY: Guarded IDs precede this dispatch and the kernel also
            // rejects negative/future/out-of-reconstruction IDs before access.
            unsafe {
                batch.dispatch_1d(pipeline, &bindings, rows * 2 * 128, 128)?;
            }
            Ok(())
        }

        pub fn dense_scratch_bytes(&self, capacity: usize) -> Result<usize> {
            self.projection.scratch_bytes(capacity)
        }

        pub fn allocate_output(&self, rows: usize) -> Result<QsaBatchAttentionOutput> {
            ensure!(
                (2..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
                "QSA batch row count exceeds supported prefill capacity"
            );
            QsaBatchAttentionOutput::new(&self.metal, rows)
        }

        /// Encode one query tile with disjoint per-query scratch.
        ///
        /// # Safety
        /// Queries/gates are finite BF16 `[output.rows,24,256]`. Selected rows
        /// have query-major finite K/V and a nonempty valid-prefix mask per row.
        /// Producers precede these commands; inputs do not alias output/scratch.
        /// Each output row is encoded once before projecting the full chunk.
        pub unsafe fn encode_tile(
            &self,
            batch: &mut Batch,
            queries: &Buffer,
            gates: &Buffer,
            selected: &PendingKvarnRows,
            offset: usize,
            output: &QsaBatchAttentionOutput,
        ) -> Result<()> {
            ensure!(
                (1..=64).contains(&selected.queries)
                    && offset
                        .checked_add(selected.queries)
                        .is_some_and(|end| end <= output.rows),
                "QSA attention query tile exceeds output chunk"
            );
            ensure!(
                queries.byte_len() >= output.rows * EXPANDED * 2
                    && gates.byte_len() >= output.rows * EXPANDED * 2,
                "QSA attention query or gate extent differs"
            );
            ensure!(
                selected.keys.byte_len() >= selected.queries * SELECTED_BYTES
                    && selected.values.byte_len() >= selected.queries * SELECTED_BYTES
                    && selected.valid.byte_len() >= selected.queries * SELECTED,
                "QSA attention selected-row extent differs"
            );
            let rows = selected.queries;
            let logits = self.metal.shared_buffer(rows * 24 * SELECTED * 2)?;
            let scaled = self.metal.shared_buffer(rows * 24 * SELECTED * 2)?;
            let masked = self.metal.shared_buffer(rows * 24 * SELECTED * 4)?;
            let probabilities = self.metal.shared_buffer(rows * 24 * SELECTED * 2)?;
            let shape = self.metal.shared_buffer(8)?;
            shape.write_bytes(
                0,
                &[SELECTED as u32, rows as u32]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let q = offset * EXPANDED * 2;
            // Each query/head owns its scratch and output; dispatch boundaries
            // order logits, softmax and value/gate without changing reductions.
            unsafe {
                let bindings = [
                    (queries, q),
                    (&selected.keys, 0),
                    (&selected.valid, 0),
                    (&logits, 0),
                    (&scaled, 0),
                    (&masked, 0),
                    (&shape, 0),
                ]
                .map(|(buffer, offset)| Binding { buffer, offset });
                batch.dispatch_1d(&self.logits, &bindings, rows * 24 * SELECTED, 128)?;
                let bindings = [(&masked, 0), (&probabilities, 0), (&shape, 0)]
                    .map(|(buffer, offset)| Binding { buffer, offset });
                batch.dispatch_1d(&self.softmax, &bindings, rows * 24 * 544, 544)?;
                let bindings = [
                    (&probabilities, 0),
                    (&selected.values, 0),
                    (gates, q),
                    (&output.attention, q),
                    (&output.sigmoid, q),
                    (&output.gated, q),
                    (&shape, 0),
                ]
                .map(|(buffer, offset)| Binding { buffer, offset });
                batch.dispatch_1d(&self.value_gate, &bindings, rows * EXPANDED, 128)?;
            }
            Ok(())
        }

        /// Project a completed set of gated rows using the batch matrix path.
        ///
        /// # Safety
        /// Every gated row was produced before this encoding, is finite BF16,
        /// and has no concurrent writer. Do not replace this with a last-row
        /// singleton projection; its quantized arithmetic can differ.
        pub unsafe fn encode_projection(
            &self,
            batch: &mut Batch,
            output: &QsaBatchAttentionOutput,
        ) -> Result<()> {
            // SAFETY: Caller establishes completed gated inputs and disjoint output.
            unsafe {
                self.projection
                    .encode_rows(batch, &output.gated, &output.output, output.rows)
            }
        }

        /// Project only active rows in a capacity-sized output allocation.
        ///
        /// # Safety
        /// Every active gated row is finite and was written earlier in the
        /// same stream; scratch has no unordered readers or writers.
        pub unsafe fn encode_projection_into(
            &self,
            batch: &mut Batch,
            output: &QsaBatchAttentionOutput,
            rows: usize,
            dense: &DenseBatchScratch,
        ) -> Result<()> {
            ensure!(
                (1..=output.rows).contains(&rows),
                "QSA projection rows exceed capacity"
            );
            // SAFETY: Caller establishes active inputs and ordered scratch reuse.
            unsafe {
                self.projection
                    .encode_rows_into(batch, &output.gated, &output.output, rows, dense)
            }
        }
    }

    #[cfg(test)]
    mod prefill_tests;
}

#[cfg(target_os = "macos")]
pub use gpu::{QsaBatchAttention, QsaBatchAttentionOutput, QsaPrefillScratch};

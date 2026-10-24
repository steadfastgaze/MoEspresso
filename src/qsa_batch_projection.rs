//! Multirow QSA projections with token-major output and shared Metal arithmetic.

#[cfg(target_os = "macos")]
mod gpu {
    use anyhow::{Context, Result, ensure};
    use half::bf16;
    use std::collections::BTreeMap;

    use crate::dense_batch::{DenseBatch, DenseBatchScratch};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use crate::package::{Package, WeightFormat};

    const SOURCE: &str = include_str!("../kernels/metal/qsa_projection.metal");

    fn upload(metal: &Metal, data: &[u8]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(data.len())?;
        buffer.write_bytes(0, data)?;
        Ok(buffer)
    }

    /// Scratch and projected values retain token-major layout for every head.
    pub struct QsaBatchProjected {
        pub index_queries: Buffer,
        pub raw_index_key: Buffer,
        pub queries: Buffer,
        pub query_gate: Buffer,
        pub keys: Buffer,
        pub values: Buffer,
        index_projection: Buffer,
        query_projection: Buffer,
        key_projection: Buffer,
        cosine: Buffer,
        sine: Buffer,
    }

    impl QsaBatchProjected {
        pub fn new(metal: &Metal, capacity: usize) -> Result<Self> {
            ensure!(
                (1..=crate::prefill_capacity::MAX_ROWS).contains(&capacity),
                "QSA projection capacity differs"
            );
            let row = |width| metal.shared_buffer(capacity * width * 2);
            Ok(Self {
                index_projection: row(640)?,
                query_projection: row(12288)?,
                key_projection: row(512)?,
                values: row(512)?,
                cosine: row(64)?,
                sine: row(64)?,
                index_queries: row(512)?,
                raw_index_key: row(128)?,
                queries: row(6144)?,
                query_gate: row(6144)?,
                keys: row(512)?,
            })
        }

        pub fn byte_len(&self) -> usize {
            [
                &self.index_projection,
                &self.query_projection,
                &self.key_projection,
                &self.values,
                &self.cosine,
                &self.sine,
                &self.index_queries,
                &self.raw_index_key,
                &self.queries,
                &self.query_gate,
                &self.keys,
            ]
            .iter()
            .map(|buffer| buffer.byte_len())
            .sum()
        }

        fn check_rows(&self, rows: usize) -> Result<()> {
            ensure!(
                (1..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
                "QSA projection row count differs"
            );
            ensure!(
                [
                    (&self.index_projection, 640),
                    (&self.query_projection, 12288),
                    (&self.key_projection, 512),
                    (&self.values, 512),
                    (&self.cosine, 64),
                    (&self.sine, 64),
                    (&self.index_queries, 512),
                    (&self.raw_index_key, 128),
                    (&self.queries, 6144),
                    (&self.query_gate, 6144),
                    (&self.keys, 512),
                ]
                .into_iter()
                .all(|(buffer, width)| buffer.byte_len() >= rows * width * 2),
                "QSA projection workspace is too small"
            );
            Ok(())
        }

        /// Read diagnostic arrays after successful GPU completion.
        ///
        /// # Safety
        /// The producing batch must have been submitted and successfully waited;
        /// dropping an unsubmitted batch does not initialize its output arrays.
        pub unsafe fn stages(&self) -> Result<BTreeMap<&'static str, Vec<u8>>> {
            [
                ("index_projection", &self.index_projection),
                ("query_projection", &self.query_projection),
                ("key_projection", &self.key_projection),
                ("value_projection", &self.values),
                ("cosine", &self.cosine),
                ("sine", &self.sine),
                ("index_queries", &self.index_queries),
                ("raw_index_key", &self.raw_index_key),
                ("queries", &self.queries),
                ("query_gate", &self.query_gate),
                ("keys", &self.keys),
                ("values", &self.values),
            ]
            .into_iter()
            .map(|(name, buffer)| {
                let mut data = vec![0; buffer.byte_len()];
                buffer.read_bytes(0, &mut data)?;
                Ok((name, data))
            })
            .collect()
        }
    }

    /// Encode independent rows without changing per-head arithmetic or layout.
    ///
    /// # Safety
    /// Every projected input and norm is initialized with the declared geometry.
    /// All output extents cover `rows`; positions contains three i64 values per
    /// row. Inputs and outputs are disjoint and retained by the command owner.
    unsafe fn encode_preparation(
        batch: &mut Batch,
        factors: &Pipeline,
        project: &Pipeline,
        positions: &Buffer,
        norms: [&Buffer; 4],
        out: &QsaBatchProjected,
        rows: usize,
    ) -> Result<()> {
        out.check_rows(rows)?;
        ensure!(
            positions.byte_len() >= rows * 24,
            "QSA position extent differs"
        );
        let [index_norm, query_norm, key_norm, epsilon] = norms;
        ensure!(
            index_norm.byte_len() >= 128 * 2
                && query_norm.byte_len() >= 256 * 2
                && key_norm.byte_len() >= 256 * 2
                && epsilon.byte_len() >= 4,
            "QSA normalization extent differs"
        );
        let binding = |buffer| Binding { buffer, offset: 0 };
        // SAFETY: Each factor thread addresses one of 32 pairs in one row;
        // each subsequent 32-thread group owns one of that row's 30 heads.
        unsafe {
            batch.dispatch_1d(
                factors,
                &[binding(positions), binding(&out.cosine), binding(&out.sine)],
                rows * 32,
                32,
            )?;
            let bindings = [
                &out.index_projection,
                &out.query_projection,
                &out.key_projection,
                index_norm,
                query_norm,
                key_norm,
                &out.cosine,
                &out.sine,
                &out.index_queries,
                &out.raw_index_key,
                &out.queries,
                &out.query_gate,
                &out.keys,
                epsilon,
            ]
            .map(binding);
            batch.dispatch_1d(project, &bindings, rows * 30 * 32, 32)?;
        }
        Ok(())
    }

    pub struct QsaBatchProjector {
        metal: Metal,
        index: DenseBatch,
        query: DenseBatch,
        key: DenseBatch,
        value: DenseBatch,
        index_norm: Buffer,
        query_norm: Buffer,
        key_norm: Buffer,
        epsilon: Buffer,
        factors: Pipeline,
        project: Pipeline,
    }

    impl QsaBatchProjector {
        pub(crate) fn gate_weight_binding(&self) -> Result<Binding<'_>> {
            let geometry = self.query.geometry();
            ensure!(
                geometry.format() == WeightFormat::Q6K
                    && geometry.input() == 2560
                    && geometry.output() == 12288,
                "ANE QSA gate requires a Q6_K [12288,2560] matrix"
            );
            Ok(self.query.packed_weight_binding())
        }

        /// Mapped query GEMM and unchanged head preparation for an external gate.
        ///
        /// # Safety
        /// Inputs, output and dense scratch satisfy encode_into's ordered ownership.
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn encode_external_gate(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            rows: usize,
            start: usize,
            out: &QsaBatchProjected,
            dense: &DenseBatchScratch,
            pipelines: [&Pipeline; 2],
        ) -> Result<()> {
            ensure!(
                (4096..=8192).contains(&rows),
                "QSA external gate rows differ"
            );
            out.check_rows(rows)?;
            let end = start.checked_add(rows).context("QSA positions overflow")?;
            let [query, prepare] = pipelines;
            unsafe {
                self.index
                    .encode_rows_into(batch, hidden, &out.index_projection, rows, dense)?;
                self.query.encode_qsa_half(
                    batch,
                    hidden,
                    &out.query_projection,
                    rows,
                    dense,
                    query,
                )?;
                self.key
                    .encode_rows_into(batch, hidden, &out.key_projection, rows, dense)?;
                self.value
                    .encode_rows_into(batch, hidden, &out.values, rows, dense)?;
                let positions: Vec<u8> = (start..end)
                    .flat_map(|p| (0..3).flat_map(move |_| (p as i64).to_le_bytes()))
                    .collect();
                let positions = upload(&self.metal, &positions)?;
                encode_preparation(
                    batch,
                    &self.factors,
                    prepare,
                    &positions,
                    [
                        &self.index_norm,
                        &self.query_norm,
                        &self.key_norm,
                        &self.epsilon,
                    ],
                    out,
                    rows,
                )
            }
        }

        pub fn dense_scratch_bytes(&self, capacity: usize) -> Result<usize> {
            [&self.index, &self.query, &self.key, &self.value]
                .iter()
                .try_fold(4, |bytes, matrix| {
                    Ok(bytes.max(matrix.scratch_bytes(capacity)?))
                })
        }

        pub(crate) fn from_decode(
            metal: &Metal,
            source: &crate::qsa_projection::QsaProjector,
        ) -> Result<Self> {
            let weights = source.shared_weights();
            let [index, query, key, value] = weights.matrices;
            let factors = metal.compile_msl_strict(SOURCE, "qsa_rope_factors_rows")?;
            let project = metal.compile_msl_strict(SOURCE, "qsa_project_rope_rows")?;
            ensure!(
                factors.max_threads_per_threadgroup() >= 32
                    && project.max_threads_per_threadgroup() >= 32,
                "QSA projection requires 32-thread SIMD groups"
            );
            Ok(Self {
                metal: metal.clone(),
                index: DenseBatch::from_matrix(index)?,
                query: DenseBatch::from_matrix(query)?,
                key: DenseBatch::from_matrix(key)?,
                value: DenseBatch::from_matrix(value)?,
                index_norm: weights.index_norm,
                query_norm: weights.query_norm,
                key_norm: weights.key_norm,
                epsilon: weights.epsilon,
                factors,
                project,
            })
        }
        pub fn new(metal: &Metal, package: &Package, layer: usize) -> Result<Self> {
            let config = crate::model::QwenConfig::from_package(package)?;
            ensure!(
                config.layers.get(layer) == Some(&crate::model::MixerKind::Qsa),
                "QSA projection layer is not sparse attention"
            );
            let prefix = format!("layers.{layer}.mixer.module");
            let projection = |leaf, width| -> Result<DenseBatch> {
                let weight = package.direct_weight(&format!("{prefix}.{leaf}"))?;
                ensure!(
                    weight.format == WeightFormat::Q6K && weight.shape == [width, 2560],
                    "QSA batch projection geometry differs"
                );
                DenseBatch::new(metal, &weight)
            };
            let norm = |leaf, width| -> Result<Buffer> {
                let weight = package.direct_weight(&format!("{prefix}.{leaf}"))?;
                ensure!(
                    weight.format == WeightFormat::Bf16 && weight.shape == [width],
                    "QSA batch normalization geometry differs"
                );
                ensure!(
                    weight
                        .bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|v| bf16::from_bits(u16::from_le_bytes(*v)).is_finite()),
                    "QSA batch norm contains nonfinite values"
                );
                upload(metal, &weight.bytes)
            };
            let factors = metal.compile_msl_strict(SOURCE, "qsa_rope_factors_rows")?;
            let project = metal.compile_msl_strict(SOURCE, "qsa_project_rope_rows")?;
            ensure!(
                factors.max_threads_per_threadgroup() >= 32
                    && project.max_threads_per_threadgroup() >= 32,
                "QSA projection requires 32-thread SIMD groups"
            );
            Ok(Self {
                metal: metal.clone(),
                index: projection("indexer.index_qk_proj", 640)?,
                query: projection("q_proj", 12288)?,
                key: projection("k_proj", 512)?,
                value: projection("v_proj", 512)?,
                index_norm: norm("indexer.q_layernorm", 128)?,
                query_norm: norm("q_norm", 256)?,
                key_norm: norm("k_norm", 256)?,
                epsilon: upload(metal, &config.eps.to_le_bytes())?,
                factors,
                project,
            })
        }

        /// Encode multirow projections and normalization without host readbacks.
        ///
        /// # Safety
        /// Hidden contains finite BF16 `[rows,2560]` values ready before these
        /// commands. It must not alias parameters or outputs. The command owner
        /// excludes host writes until completion.
        pub unsafe fn encode(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            rows: usize,
            start: usize,
        ) -> Result<QsaBatchProjected> {
            ensure!(
                (2..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
                "QSA batch row count exceeds supported prefill capacity"
            );
            ensure!(
                start.checked_add(rows).is_some_and(|end| end <= 262144),
                "QSA batch exceeds model context"
            );
            ensure!(
                hidden.byte_len() >= rows * 2560 * 2,
                "QSA batch hidden extent differs"
            );
            let out = QsaBatchProjected::new(&self.metal, rows)?;
            let dense = DenseBatchScratch::new(&self.metal, self.dense_scratch_bytes(rows)?)?;
            // SAFETY: The newly allocated output and dense scratch are disjoint.
            unsafe { self.encode_into(batch, hidden, rows, start, &out, &dense)? };
            Ok(out)
        }

        /// Encode into reusable projection allocations owned by the prefill workspace.
        ///
        /// # Safety
        /// The active hidden rows are finite BF16. The output and shared dense
        /// scratch are disjoint; earlier uses are ordered on the same stream.
        pub unsafe fn encode_into(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            rows: usize,
            start: usize,
            out: &QsaBatchProjected,
            dense: &DenseBatchScratch,
        ) -> Result<()> {
            ensure!(
                (1..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
                "QSA projection row count differs"
            );
            ensure!(
                start.checked_add(rows).is_some_and(|end| end <= 262144),
                "QSA projection exceeds model context"
            );
            ensure!(
                hidden.byte_len() >= rows * 2560 * 2,
                "QSA hidden extent differs"
            );
            out.check_rows(rows)?;
            let positions: Vec<u8> = (start..start + rows)
                .flat_map(|position| (0..3).flat_map(move |_| (position as i64).to_le_bytes()))
                .collect();
            let positions = upload(&self.metal, &positions)?;
            // SAFETY: Fixed disjoint scratch extents cover each projection and
            // row grid. Batch retains every allocation through completion.
            unsafe {
                for (projection, target) in [
                    (&self.index, &out.index_projection),
                    (&self.query, &out.query_projection),
                    (&self.key, &out.key_projection),
                    (&self.value, &out.values),
                ] {
                    projection.encode_rows_into(batch, hidden, target, rows, dense)?;
                }
                encode_preparation(
                    batch,
                    &self.factors,
                    &self.project,
                    &positions,
                    [
                        &self.index_norm,
                        &self.query_norm,
                        &self.key_norm,
                        &self.epsilon,
                    ],
                    out,
                    rows,
                )?;
            }
            Ok(())
        }

        pub fn run_stages(
            &self,
            hidden: &[u8],
            rows: usize,
            start: usize,
        ) -> Result<BTreeMap<&'static str, Vec<u8>>> {
            ensure!(
                (2..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
                "QSA batch row count exceeds supported prefill capacity"
            );
            ensure!(
                hidden.len() == rows * 2560 * 2,
                "QSA batch hidden extent differs"
            );
            ensure!(
                hidden
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|v| bf16::from_bits(u16::from_le_bytes(*v)).is_finite()),
                "QSA batch input is nonfinite"
            );
            let input = upload(&self.metal, hidden)?;
            let mut batch = self.metal.batch()?;
            // SAFETY: Input shape and finite values were validated above.
            let out = unsafe { self.encode(&mut batch, &input, rows, start)? };
            batch.submit().wait().context("QSA batch projection")?;
            // SAFETY: The producing command completed successfully above.
            unsafe { out.stages() }
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;

        fn fill(buffer: &Buffer, seed: usize) -> Result<()> {
            let bytes: Vec<u8> = (0..buffer.byte_len() / 2)
                .flat_map(|index| {
                    let value = ((index * 17 + seed) % 1021) as f32 / 128.0 - 4.0;
                    bf16::from_f32(value).to_bits().to_le_bytes()
                })
                .collect();
            buffer.write_bytes(0, &bytes)
        }

        fn initialize(out: &QsaBatchProjected) -> Result<()> {
            for (index, buffer) in [
                &out.index_projection,
                &out.query_projection,
                &out.key_projection,
                &out.values,
                &out.cosine,
                &out.sine,
                &out.index_queries,
                &out.raw_index_key,
                &out.queries,
                &out.query_gate,
                &out.keys,
            ]
            .into_iter()
            .enumerate()
            {
                fill(buffer, index * 31)?;
            }
            Ok(())
        }

        #[test]
        fn preparation_row_grids_match_per_row_kernels_and_preserve_unused_rows() -> Result<()> {
            let metal = Metal::new()?;
            let factors = metal.compile_msl_strict(SOURCE, "qsa_rope_factors_rows")?;
            let project = metal.compile_msl_strict(SOURCE, "qsa_project_rope_rows")?;
            let old_factors = metal.compile_msl_strict(SOURCE, "qsa_rope_factors")?;
            let old_project = metal.compile_msl_strict(SOURCE, "qsa_project_rope_one")?;
            let index_norm = metal.shared_buffer(128 * 2)?;
            let query_norm = metal.shared_buffer(256 * 2)?;
            let key_norm = metal.shared_buffer(256 * 2)?;
            for (index, norm) in [&index_norm, &query_norm, &key_norm]
                .into_iter()
                .enumerate()
            {
                fill(norm, 101 + index * 71)?;
            }
            let epsilon = upload(&metal, &1e-6f32.to_le_bytes())?;
            let norms = [&index_norm, &query_norm, &key_norm, &epsilon];
            for (rows, start) in [(1, 0), (1, 262143), (3, 2045), (65, 8430), (2048, 260096)] {
                // A spare initialized row detects writes beyond the active
                // extent for every case below the maximum workspace width.
                let capacity = (rows + 1).min(crate::prefill_capacity::MAX_ROWS);
                let expected = QsaBatchProjected::new(&metal, capacity)?;
                let actual = QsaBatchProjected::new(&metal, capacity)?;
                initialize(&expected)?;
                initialize(&actual)?;
                let positions: Vec<u8> = (start..start + rows)
                    .flat_map(|position| (0..3).flat_map(move |_| (position as i64).to_le_bytes()))
                    .collect();
                let positions = upload(&metal, &positions)?;
                let mut batch = metal.batch()?;
                // SAFETY: Distinct initialized workspaces have the full active
                // row geometry; the baseline is the original per-row schedule.
                unsafe {
                    for row in 0..rows {
                        let factors_bindings = [
                            (&positions, row * 24),
                            (&expected.cosine, row * 128),
                            (&expected.sine, row * 128),
                        ]
                        .map(|(buffer, offset)| Binding { buffer, offset });
                        batch.dispatch_1d(&old_factors, &factors_bindings, 32, 32)?;
                        let bindings = [
                            (&expected.index_projection, row * 640 * 2),
                            (&expected.query_projection, row * 12288 * 2),
                            (&expected.key_projection, row * 512 * 2),
                            (&index_norm, 0),
                            (&query_norm, 0),
                            (&key_norm, 0),
                            (&expected.cosine, row * 128),
                            (&expected.sine, row * 128),
                            (&expected.index_queries, row * 512 * 2),
                            (&expected.raw_index_key, row * 128 * 2),
                            (&expected.queries, row * 6144 * 2),
                            (&expected.query_gate, row * 6144 * 2),
                            (&expected.keys, row * 512 * 2),
                            (&epsilon, 0),
                        ]
                        .map(|(buffer, offset)| Binding { buffer, offset });
                        batch.dispatch_1d(&old_project, &bindings, 30 * 32, 32)?;
                    }
                    encode_preparation(
                        &mut batch, &factors, &project, &positions, norms, &actual, rows,
                    )?;
                }
                assert!(actual.cosine.read_bytes(0, &mut [0; 2]).is_err());
                batch.submit().wait()?;
                // SAFETY: Both complete workspaces were initialized on the
                // host, and all producing commands completed successfully.
                let (expected, actual) = unsafe { (expected.stages()?, actual.stages()?) };
                for (name, reference) in expected {
                    ensure!(
                        actual.get(name) == Some(&reference),
                        "{rows} rows at {start}: {name} differs from per-row preparation"
                    );
                }
            }
            // Whole-buffer dispatches must reject either short factor array
            // before any GPU work can address it.
            for short_cosine in [false, true] {
                let mut out = QsaBatchProjected::new(&metal, 1)?;
                if short_cosine {
                    out.cosine = metal.shared_buffer(126)?;
                } else {
                    out.sine = metal.shared_buffer(126)?;
                }
                let positions = upload(&metal, &[0; 24])?;
                let mut batch = metal.batch()?;
                // SAFETY: The invalid extent is rejected before dispatch;
                // this call must not read the otherwise uninitialized inputs.
                let error = unsafe {
                    encode_preparation(&mut batch, &factors, &project, &positions, norms, &out, 1)
                }
                .unwrap_err();
                assert_eq!(error.to_string(), "QSA projection workspace is too small");
                out.cosine.write_bytes(0, &[0; 2])?;
                out.sine.write_bytes(0, &[0; 2])?;
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{QsaBatchProjected, QsaBatchProjector};

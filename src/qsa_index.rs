//! Four-token QSA index compression and stable physical-row selection.
//!
//! The all-visible text stream has one position per physical token. A step
//! returns a candidate state, which the decoder commits after the full layer.

use anyhow::{Context, Result, ensure};
use half::bf16;

pub const INDEX_WIDTH: usize = 128;
pub const COMPRESS_RATIO: usize = 4;
pub const SELECTED_WIDTH: usize = 2051;
const QUERY_WORDS: usize = 4 * INDEX_WIDTH;
const BLOCK_BUDGET: usize = 512;
const MAX_POSITIONS: u64 = 262_144;

fn compressed_growth_target(
    groups: usize,
    capacity_bytes: usize,
    max_groups: usize,
) -> Result<Option<usize>> {
    let row_bytes = INDEX_WIDTH * 2;
    ensure!(
        capacity_bytes.is_multiple_of(row_bytes) && groups <= max_groups,
        "QSA compressed growth geometry differs"
    );
    let capacity = capacity_bytes / row_bytes;
    if groups <= capacity {
        return Ok(None);
    }
    let target = groups.max(capacity.saturating_mul(2).min(max_groups));
    Ok(Some(
        target
            .checked_mul(row_bytes)
            .context("QSA compressed growth overflow")?,
    ))
}

fn finite(values: &[u16]) -> bool {
    values.iter().all(|&v| bf16::from_bits(v).is_finite())
}

fn candidate_lengths(frontier: u64, limit: u64) -> Result<(usize, usize)> {
    ensure!(
        frontier <= limit && frontier <= MAX_POSITIONS,
        "QSA index candidate exceeds context limit"
    );
    let frontier = usize::try_from(frontier).context("QSA index frontier exceeds host size")?;
    let compressed = (frontier / COMPRESS_RATIO)
        .checked_mul(INDEX_WIDTH)
        .context("QSA compressed index size overflow")?;
    let pending = (frontier % COMPRESS_RATIO)
        .checked_mul(INDEX_WIDTH)
        .context("QSA pending index size overflow")?;
    Ok((compressed, pending))
}

fn copied_words(source: &[u16], capacity: usize) -> Result<Vec<u16>> {
    ensure!(
        source.len() <= capacity,
        "QSA index copy exceeds candidate size"
    );
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .context("QSA index candidate reservation failed")?;
    result.extend_from_slice(source);
    Ok(result)
}

fn chunk_rows(frontier: u64, raw_words: usize, query_words: usize, limit: u64) -> Result<usize> {
    ensure!(
        raw_words.is_multiple_of(INDEX_WIDTH),
        "QSA index chunk has partial raw rows"
    );
    let rows = raw_words / INDEX_WIDTH;
    ensure!(
        (2..=crate::prefill_capacity::MAX_ROWS).contains(&rows)
            && query_words == rows * QUERY_WORDS,
        "QSA index chunk requires 2 through {} complete query/key rows",
        crate::prefill_capacity::MAX_ROWS
    );
    ensure!(
        frontier
            .checked_add(rows as u64)
            .is_some_and(|end| end <= limit),
        "QSA index chunk exceeds context limit"
    );
    Ok(rows)
}

#[derive(Clone, Default)]
pub struct QsaIndexState {
    frontier: u64,
    compressed: Vec<u16>,
    pending: Vec<u16>,
}

impl QsaIndexState {
    pub fn zero() -> Self {
        Self::default()
    }

    /// Restore a validated snapshot with complete groups and a zero-to-three-row tail.
    pub fn from_parts(frontier: u64, compressed: Vec<u16>, pending: Vec<u16>) -> Result<Self> {
        Self::validate_parts(frontier, &compressed, &pending)?;
        Ok(Self {
            frontier,
            compressed,
            pending,
        })
    }

    /// Validate and fallibly copy a borrowed snapshot into independent state.
    #[cfg(test)]
    pub(crate) fn try_from_parts(
        frontier: u64,
        compressed: &[u16],
        pending: &[u16],
    ) -> Result<Self> {
        Self::validate_parts(frontier, compressed, pending)?;
        Ok(Self {
            frontier,
            compressed: copied_words(compressed, compressed.len())?,
            pending: copied_words(pending, pending.len())?,
        })
    }

    /// Validate borrowed index rows without copying either payload.
    pub(crate) fn validate_parts(frontier: u64, compressed: &[u16], pending: &[u16]) -> Result<()> {
        ensure!(
            frontier <= MAX_POSITIONS,
            "QSA index frontier exceeds supported context"
        );
        ensure!(
            compressed.len() == frontier as usize / COMPRESS_RATIO * INDEX_WIDTH
                && pending.len() == frontier as usize % COMPRESS_RATIO * INDEX_WIDTH,
            "QSA index snapshot lengths differ from frontier"
        );
        ensure!(
            finite(compressed) && finite(pending),
            "QSA index snapshot contains nonfinite values"
        );
        Ok(())
    }

    pub fn frontier(&self) -> u64 {
        self.frontier
    }
    pub fn compressed_bf16(&self) -> &[u16] {
        &self.compressed
    }
    pub fn pending_bf16(&self) -> &[u16] {
        &self.pending
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn into_parts(self) -> (u64, Vec<u16>, Vec<u16>) {
        (self.frontier, self.compressed, self.pending)
    }
}

/// Select the strongest complete groups, breaking score ties by lower group ID.
/// Returned rows retain physical context order and have trailing -1 padding.
pub fn select_scores(scores: &[f32], frontier: u64) -> Result<Vec<i32>> {
    ensure!(
        frontier > 0 && frontier <= MAX_POSITIONS,
        "invalid QSA selection frontier"
    );
    let groups = frontier as usize / COMPRESS_RATIO;
    ensure!(
        scores.len() == groups,
        "QSA score count differs from frontier"
    );
    ensure!(
        scores.iter().all(|&v| v.is_finite() && v >= 0.0),
        "QSA selection requires finite nonnegative scores"
    );
    let mut chosen: Vec<usize> = (0..groups).collect();
    if groups > BLOCK_BUDGET {
        chosen
            .sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
        chosen.truncate(BLOCK_BUDGET);
        chosen.sort_unstable();
    }
    let mut selected = Vec::with_capacity(SELECTED_WIDTH);
    for group in chosen {
        selected.extend((group * COMPRESS_RATIO..(group + 1) * COMPRESS_RATIO).map(|v| v as i32));
    }
    selected.extend((groups * COMPRESS_RATIO..frontier as usize).map(|v| v as i32));
    selected.resize(SELECTED_WIDTH, -1);
    Ok(selected)
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use crate::device_status::DeviceStatus;
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use crate::model::{MixerKind, QwenConfig};
    use crate::package::{Package, WeightFormat};
    use crate::qsa_projection::QsaRope;
    use anyhow::Context;

    const SOURCE: &str = include_str!("../kernels/metal/qsa_index.metal");
    const DEVICE_SOURCE: &str = include_str!("../kernels/metal/qsa_index_device.metal");

    struct DeviceIndexKernels {
        copy: Pipeline,
        pending: Pipeline,
        compress: Pipeline,
        compress_chunk: Pipeline,
        pending_tail: Pipeline,
        score: Pipeline,
        score_mm: Pipeline,
        select: Pipeline,
        select_tile: Pipeline,
        expand: Pipeline,
        expand_tile: Pipeline,
    }

    impl DeviceIndexKernels {
        fn new(metal: &Metal) -> Result<Self> {
            Ok(Self {
                copy: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_copy")?,
                pending: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_pending")?,
                compress: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_compress")?,
                compress_chunk: metal
                    .compile_msl_strict(DEVICE_SOURCE, "qsa_device_compress_chunk")?,
                pending_tail: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_pending_tail")?,
                score: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_score")?,
                score_mm: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_score_mm")?,
                select: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_select")?,
                select_tile: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_select_tile")?,
                expand: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_expand")?,
                expand_tile: metal.compile_msl_strict(DEVICE_SOURCE, "qsa_device_expand_tile")?,
            })
        }
    }

    struct DeviceIndexScratch {
        selected_groups: Buffer,
        ids: Buffer,
        valid: Buffer,
        count: Buffer,
        status: Buffer,
        group_count: Buffer,
        context: Buffer,
        positions: Buffer,
        cosine: Buffer,
        sine: Buffer,
        slot: Buffer,
        scores: Buffer,
    }

    impl DeviceIndexScratch {
        fn new(metal: &Metal) -> Result<Self> {
            Ok(Self {
                selected_groups: metal.shared_buffer(BLOCK_BUDGET * 4)?,
                ids: metal.shared_buffer(SELECTED_WIDTH * 4)?,
                valid: metal.shared_buffer(SELECTED_WIDTH)?,
                count: metal.shared_buffer(4)?,
                status: metal.shared_buffer(4)?,
                group_count: metal.shared_buffer(4)?,
                context: metal.shared_buffer(8)?,
                positions: metal.shared_buffer(3 * 8)?,
                cosine: metal.shared_buffer(64 * 2)?,
                sine: metal.shared_buffer(64 * 2)?,
                slot: metal.shared_buffer(4)?,
                scores: metal.shared_buffer(4)?,
            })
        }

        fn fit_scores(&mut self, metal: &Metal, groups: usize, max_groups: usize) -> Result<()> {
            let needed = groups * 4;
            if needed > self.scores.byte_len() {
                let capacity =
                    needed.max(self.scores.byte_len().saturating_mul(2).min(max_groups * 4));
                self.scores = metal.shared_buffer(capacity)?;
            }
            Ok(())
        }
    }

    /// Canonical device index. Host copies are explicit snapshot/prefill seams.
    pub struct DeviceQsaIndexState {
        max_positions: u64,
        frontier: u64,
        valid: bool,
        compressed: Buffer,
        pending: Buffer,
        pending_frontier: Option<u64>,
        pending_status: Option<Buffer>,
    }

    pub struct DeviceQsaSelection {
        pub ids: Buffer,
        pub valid: Buffer,
        pub count: Buffer,
        pub status: Buffer,
    }

    /// Reused across QSA layers. Score storage covers one 64-query tile;
    /// selections cover the full chunk because attention consumes later tiles.
    pub struct QsaChunkIndexScratch {
        ids: Buffer,
        valid: Buffer,
        count: Buffer,
        selected_groups: Buffer,
        scores: Buffer,
        capacity: usize,
        max_groups: usize,
    }

    impl QsaChunkIndexScratch {
        pub fn new(metal: &Metal, capacity: usize, max_positions: usize) -> Result<Self> {
            ensure!(
                (1..=crate::prefill_capacity::MAX_ROWS).contains(&capacity)
                    && (1..=262144).contains(&max_positions),
                "QSA chunk index capacity differs"
            );
            let max_groups = max_positions / COMPRESS_RATIO;
            Ok(Self {
                ids: metal.shared_buffer(capacity * SELECTED_WIDTH * 4)?,
                valid: metal.shared_buffer(capacity * SELECTED_WIDTH)?,
                count: metal.shared_buffer(capacity * 4)?,
                selected_groups: metal.shared_buffer(64 * BLOCK_BUDGET * 4)?,
                scores: metal.shared_buffer(64 * max_groups.max(1) * 4)?,
                capacity,
                max_groups,
            })
        }

        pub fn byte_len(&self) -> usize {
            [
                &self.ids,
                &self.valid,
                &self.count,
                &self.selected_groups,
                &self.scores,
            ]
            .iter()
            .map(|buffer| buffer.byte_len())
            .sum()
        }
    }

    pub struct DeviceQsaChunkSelection {
        pub ids: Buffer,
        pub valid: Buffer,
        pub count: Buffer,
        pub status: Buffer,
        pub rows: usize,
    }

    fn bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn upload(metal: &Metal, bytes: &[u8]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(bytes.len())?;
        buffer.write_bytes(0, bytes)?;
        Ok(buffer)
    }

    pub struct QsaIndex {
        metal: Metal,
        rope: QsaRope,
        norm: Buffer,
        compress: Pipeline,
        score: Pipeline,
        factors: Pipeline,
        max_positions: u64,
        device: Option<DeviceIndexKernels>,
        device_scratch: Option<DeviceIndexScratch>,
    }

    pub struct QsaIndexStep {
        pub state: QsaIndexState,
        pub selected: Vec<i32>,
        /// Empty for short contexts, where all rows are selected without scoring.
        pub scores: Vec<f32>,
    }

    /// One uncommitted candidate and query-major physical IDs `[rows,2051]`.
    pub struct QsaIndexChunk {
        pub state: QsaIndexState,
        pub selected: Vec<i32>,
        pub rows: usize,
    }

    type ScoreObserver<'a> = &'a mut dyn FnMut(usize, &[f32], &[f32]) -> Result<()>;

    impl QsaIndex {
        pub fn new(metal: &Metal, package: &Package, layer: usize) -> Result<Self> {
            let config = QwenConfig::from_package(package)?;
            ensure!(
                config.layers.get(layer) == Some(&MixerKind::Qsa),
                "QSA index requires a declared QSA layer"
            );
            let weight = package
                .direct_weight(&format!("layers.{layer}.mixer.module.indexer.k_layernorm"))?;
            ensure!(
                weight.format == WeightFormat::Bf16
                    && weight.shape == [INDEX_WIDTH]
                    && weight.bytes.len() == INDEX_WIDTH * 2,
                "QSA index norm has incompatible geometry"
            );
            ensure!(
                weight
                    .bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|w| bf16::from_bits(u16::from_le_bytes(*w)).is_finite()),
                "QSA index norm contains nonfinite values"
            );
            let compress = metal.compile_msl_strict(SOURCE, "qsa_index_compress")?;
            let score = metal.compile_msl_strict(SOURCE, "qsa_index_scores")?;
            ensure!(
                compress.max_threads_per_threadgroup() >= 32
                    && score.max_threads_per_threadgroup() >= 32,
                "QSA index requires 32-thread Metal groups"
            );
            Ok(Self {
                metal: metal.clone(),
                rope: QsaRope::new(metal)?,
                norm: upload(metal, &weight.bytes)?,
                compress,
                score,
                factors: metal.compile_msl_strict(
                    include_str!("../kernels/metal/qsa_projection.metal"),
                    "qsa_rope_factors",
                )?,
                max_positions: config.max_positions as u64,
                device: None,
                device_scratch: None,
            })
        }

        fn compress_group(&mut self, raw: &[u16], start: u64) -> Result<Vec<u16>> {
            ensure!(
                raw.len() == COMPRESS_RATIO * INDEX_WIDTH && finite(raw),
                "invalid QSA raw index group"
            );
            let factors = self.rope.at([start as i64; 3])?;
            let input = upload(&self.metal, &bytes(raw))?;
            let cosine = upload(&self.metal, &bytes(&factors.cosine_bf16))?;
            let sine = upload(&self.metal, &bytes(&factors.sine_bf16))?;
            let output = self.metal.shared_buffer(INDEX_WIDTH * 2)?;
            let bindings = [&input, &self.norm, &cosine, &sine, &output]
                .map(|buffer| Binding { buffer, offset: 0 });
            let mut batch = self.metal.batch()?;
            // SAFETY: One SIMD group reads four 128-wide rows, a 128-wide norm,
            // 64-wide factors and writes one disjoint 128-wide BF16 row.
            unsafe {
                batch.dispatch_1d(&self.compress, &bindings, 32, 32)?;
            }
            batch
                .submit()
                .wait()
                .context("QSA index compression command")?;
            let mut result = vec![0; INDEX_WIDTH * 2];
            output.read_bytes(0, &mut result)?;
            let words: Vec<u16> = result
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes(*v))
                .collect();
            ensure!(
                finite(&words),
                "QSA index compression produced nonfinite values"
            );
            Ok(words)
        }

        /// Score a budgeted context. Short contexts select every row without scoring.
        pub fn scores(&self, queries: &[u16], compressed: &[u16]) -> Result<Vec<f32>> {
            Ok(self.score_stages(queries, compressed)?.0)
        }

        pub fn score_stages(
            &self,
            queries: &[u16],
            compressed: &[u16],
        ) -> Result<(Vec<f32>, Vec<f32>)> {
            ensure!(
                queries.len() == QUERY_WORDS && finite(queries),
                "invalid QSA index query row"
            );
            ensure!(
                compressed.len().is_multiple_of(INDEX_WIDTH)
                    && finite(compressed)
                    && compressed.len() <= MAX_POSITIONS as usize / 4 * INDEX_WIDTH,
                "invalid QSA compressed index rows"
            );
            let count = compressed.len() / INDEX_WIDTH;
            ensure!(
                count >= BLOCK_BUDGET,
                "QSA index scores require at least 512 complete groups"
            );
            let mut query_values: Vec<f32> = queries
                .iter()
                .map(|&v| bf16::from_bits(v).to_f32())
                .collect();
            query_values.resize(8 * INDEX_WIDTH, 0.0);
            let mut key_values: Vec<f32> = compressed
                .iter()
                .map(|&v| bf16::from_bits(v).to_f32())
                .collect();
            key_values.resize(count.div_ceil(8) * 8 * INDEX_WIDTH, 0.0);
            let input = upload(
                &self.metal,
                &query_values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let keys = upload(
                &self.metal,
                &key_values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let counts = upload(&self.metal, &(count as u32).to_le_bytes())?;
            let output = self.metal.shared_buffer(count * 4)?;
            let raw_output = self.metal.shared_buffer(count * 4 * 4)?;
            let bindings = [&input, &keys, &output, &counts, &raw_output]
                .map(|buffer| Binding { buffer, offset: 0 });
            let mut batch = self.metal.batch()?;
            // SAFETY: Query and key matrices are padded to complete 8x8 tiles;
            // each SIMD group writes at most eight checked output elements.
            unsafe {
                batch.dispatch_1d(&self.score, &bindings, count.div_ceil(8) * 32, 32)?;
            }
            batch.submit().wait().context("QSA index scoring command")?;
            let mut result = vec![0; count * 4];
            output.read_bytes(0, &mut result)?;
            let values: Vec<f32> = result
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            ensure!(
                values.iter().all(|&v| v.is_finite() && v >= 0.0),
                "QSA index score is nonfinite or negative"
            );
            let mut raw = vec![0; count * 4 * 4];
            raw_output.read_bytes(0, &mut raw)?;
            Ok((
                values,
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect(),
            ))
        }

        pub fn step(
            &mut self,
            state: &QsaIndexState,
            raw_key_bf16: &[u16],
            queries_bf16: &[u16],
        ) -> Result<QsaIndexStep> {
            ensure!(
                state.frontier < self.max_positions,
                "QSA index context limit reached"
            );
            ensure!(
                raw_key_bf16.len() == INDEX_WIDTH && finite(raw_key_bf16),
                "invalid QSA raw index key"
            );
            ensure!(
                queries_bf16.len() == QUERY_WORDS && finite(queries_bf16),
                "invalid QSA index query row"
            );
            let (old_compressed, old_pending) =
                candidate_lengths(state.frontier, self.max_positions)?;
            ensure!(
                state.compressed.len() == old_compressed && state.pending.len() == old_pending,
                "QSA index state lengths differ from frontier"
            );
            let frontier = state
                .frontier
                .checked_add(1)
                .context("QSA index frontier overflow")?;
            let (compressed_words, pending_words) =
                candidate_lengths(frontier, self.max_positions)?;
            let group_words = state
                .pending
                .len()
                .checked_add(INDEX_WIDTH)
                .context("QSA pending group size overflow")?;
            ensure!(
                group_words <= COMPRESS_RATIO * INDEX_WIDTH,
                "QSA pending group exceeds four rows"
            );
            let mut group = [0u16; COMPRESS_RATIO * INDEX_WIDTH];
            group[..state.pending.len()].copy_from_slice(&state.pending);
            group[state.pending.len()..group_words].copy_from_slice(raw_key_bf16);
            let mut compressed = copied_words(&state.compressed, compressed_words)?;
            let pending = copied_words(&group[..pending_words], pending_words)?;
            if group_words == COMPRESS_RATIO * INDEX_WIDTH {
                let encoded = self.compress_group(&group, frontier - COMPRESS_RATIO as u64)?;
                ensure!(
                    encoded.len() == INDEX_WIDTH,
                    "QSA compressed group width differs"
                );
                compressed.extend_from_slice(&encoded);
            }
            ensure!(
                compressed.len() == compressed_words,
                "QSA compressed candidate length differs"
            );
            let next = QsaIndexState {
                frontier,
                compressed,
                pending,
            };
            let scores = if next.frontier > 2048 {
                self.scores(queries_bf16, &next.compressed)?
            } else {
                Vec::new()
            };
            let selected = if next.frontier <= 2048 {
                let mut ids: Vec<i32> = (0..next.frontier as i32).collect();
                ids.resize(SELECTED_WIDTH, -1);
                ids
            } else {
                select_scores(&scores, next.frontier)?
            };
            Ok(QsaIndexStep {
                state: next,
                selected,
                scores,
            })
        }

        /// Build a causal candidate without changing the committed input state.
        /// Old compressed keys are copied and uploaded once for the whole chunk.
        pub fn chunk(
            &mut self,
            state: &QsaIndexState,
            raw: &[u16],
            queries: &[u16],
        ) -> Result<QsaIndexChunk> {
            self.chunk_impl(state, raw, queries, None)
        }

        /// Compare scores without retaining an unbounded full-chunk score matrix.
        /// The observer sees one causal score row and head-major raw `[4,groups]`
        /// products. Short contexts are scored for this diagnostic entry point.
        /// An observer error drops the candidate and leaves `state` unchanged.
        pub fn chunk_with_score_observer(
            &mut self,
            state: &QsaIndexState,
            raw: &[u16],
            queries: &[u16],
            observer: ScoreObserver<'_>,
        ) -> Result<QsaIndexChunk> {
            self.chunk_impl(state, raw, queries, Some(observer))
        }

        fn chunk_impl(
            &mut self,
            state: &QsaIndexState,
            raw: &[u16],
            queries: &[u16],
            mut observer: Option<ScoreObserver<'_>>,
        ) -> Result<QsaIndexChunk> {
            let rows = chunk_rows(state.frontier, raw.len(), queries.len(), self.max_positions)?;
            ensure!(
                finite(raw) && finite(queries),
                "QSA index chunk contains nonfinite input"
            );
            let (old_compressed, old_pending) =
                candidate_lengths(state.frontier, self.max_positions)?;
            ensure!(
                state.compressed.len() == old_compressed && state.pending.len() == old_pending,
                "QSA index state lengths differ from frontier"
            );
            let frontier = state
                .frontier
                .checked_add(rows as u64)
                .context("QSA index frontier overflow")?;
            let (compressed_words, pending_words) =
                candidate_lengths(frontier, self.max_positions)?;
            let joined_words = state
                .pending
                .len()
                .checked_add(raw.len())
                .context("QSA joined key size overflow")?;
            let mut joined = Vec::new();
            joined
                .try_reserve_exact(joined_words)
                .context("QSA joined key reservation failed")?;
            joined.extend_from_slice(&state.pending);
            joined.extend_from_slice(raw);
            let new_groups = joined.len() / (COMPRESS_RATIO * INDEX_WIDTH);
            let complete_words = new_groups
                .checked_mul(COMPRESS_RATIO * INDEX_WIDTH)
                .context("QSA complete group size overflow")?;
            ensure!(
                joined.len() - complete_words == pending_words,
                "QSA pending tail differs from frontier"
            );
            let mut compressed = copied_words(&state.compressed, compressed_words)?;
            let pending = copied_words(&joined[complete_words..], pending_words)?;
            if new_groups > 0 {
                let start = state.frontier - state.frontier % 4;
                let positions: Vec<_> = (0..new_groups)
                    .flat_map(|group| [((start + group as u64 * 4) as i64); 3])
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let positions = upload(&self.metal, &positions)?;
                let input = upload(&self.metal, &bytes(&joined[..complete_words]))?;
                let cosine = self.metal.shared_buffer(new_groups * 64 * 2)?;
                let sine = self.metal.shared_buffer(new_groups * 64 * 2)?;
                let output = self.metal.shared_buffer(new_groups * INDEX_WIDTH * 2)?;
                let mut batch = self.metal.batch()?;
                for group in 0..new_groups {
                    let factors = [
                        (&positions, group * 24),
                        (&cosine, group * 128),
                        (&sine, group * 128),
                    ]
                    .map(|(buffer, offset)| Binding { buffer, offset });
                    let bindings = [
                        (&input, group * 4 * INDEX_WIDTH * 2),
                        (&self.norm, 0),
                        (&cosine, group * 128),
                        (&sine, group * 128),
                        (&output, group * INDEX_WIDTH * 2),
                    ]
                    .map(|(buffer, offset)| Binding { buffer, offset });
                    // SAFETY: Disjoint complete groups use the same proven factor
                    // and compression kernels, ordered in one retained batch.
                    unsafe {
                        batch.dispatch_1d(&self.factors, &factors, 32, 32)?;
                        batch.dispatch_1d(&self.compress, &bindings, 32, 32)?;
                    }
                }
                batch
                    .submit()
                    .wait()
                    .context("QSA index chunk compression")?;
                let mut encoded = vec![0; new_groups * INDEX_WIDTH * 2];
                output.read_bytes(0, &mut encoded)?;
                compressed.extend(
                    encoded
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|v| u16::from_le_bytes(*v)),
                );
                ensure!(
                    finite(&compressed[state.compressed.len()..]),
                    "QSA index chunk compression produced nonfinite values"
                );
            }
            ensure!(
                compressed.len() == compressed_words,
                "QSA compressed candidate length differs"
            );
            let next = QsaIndexState {
                frontier,
                compressed,
                pending,
            };
            let selected_words = rows
                .checked_mul(SELECTED_WIDTH)
                .context("QSA selected row size overflow")?;
            let mut selected = Vec::new();
            selected
                .try_reserve_exact(selected_words)
                .context("QSA selected row reservation failed")?;
            if frontier <= 2048 && observer.is_none() {
                for row in 0..rows {
                    let visible = state.frontier + row as u64 + 1;
                    selected.extend(0..visible as i32);
                    selected.resize((row + 1) * SELECTED_WIDTH, -1);
                }
                return Ok(QsaIndexChunk {
                    state: next,
                    selected,
                    rows,
                });
            }
            let groups = next.compressed.len() / INDEX_WIDTH;
            let padded_groups = groups.div_ceil(8) * 8;
            let mut key_bytes = Vec::with_capacity(padded_groups * INDEX_WIDTH * 4);
            for &word in &next.compressed {
                key_bytes.extend_from_slice(&bf16::from_bits(word).to_f32().to_le_bytes());
            }
            key_bytes.resize((padded_groups * INDEX_WIDTH * 4).max(4), 0);
            let keys = upload(&self.metal, &key_bytes)?;
            drop(key_bytes);
            for tile_start in (0..rows).step_by(64) {
                let tile_rows = (rows - tile_start).min(64);
                let mut query_bytes = Vec::with_capacity(tile_rows * 8 * INDEX_WIDTH * 4);
                let mut counts = Vec::with_capacity(tile_rows);
                for local in 0..tile_rows {
                    let row = tile_start + local;
                    for &word in &queries[row * QUERY_WORDS..(row + 1) * QUERY_WORDS] {
                        query_bytes
                            .extend_from_slice(&bf16::from_bits(word).to_f32().to_le_bytes());
                    }
                    query_bytes.resize((local + 1) * 8 * INDEX_WIDTH * 4, 0);
                    let visible = state.frontier + row as u64 + 1;
                    counts.push(if visible > 2048 || observer.is_some() {
                        visible as usize / 4
                    } else {
                        0
                    });
                }
                let max_count = counts.iter().copied().max().unwrap_or(0);
                let input = upload(&self.metal, &query_bytes)?;
                let dimensions = upload(
                    &self.metal,
                    &counts
                        .iter()
                        .flat_map(|&v| (v as u32).to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                let output = self
                    .metal
                    .shared_buffer((tile_rows * max_count * 4).max(4))?;
                let raw_output = self
                    .metal
                    .shared_buffer((tile_rows * max_count * 16).max(4))?;
                let mut batch = self.metal.batch()?;
                for (local, &count) in counts.iter().enumerate() {
                    if count == 0 {
                        continue;
                    }
                    let bindings = [
                        (&input, local * 8 * INDEX_WIDTH * 4),
                        (&keys, 0),
                        (&output, local * max_count * 4),
                        (&dimensions, local * 4),
                        (&raw_output, local * max_count * 16),
                    ]
                    .map(|(buffer, offset)| Binding { buffer, offset });
                    // SAFETY: Final keys are padded once; each causal count only
                    // publishes completed groups into its disjoint query slice.
                    unsafe {
                        batch.dispatch_1d(&self.score, &bindings, count.div_ceil(8) * 32, 32)?;
                    }
                }
                batch.submit().wait().context("QSA index chunk scoring")?;
                for (local, &count) in counts.iter().enumerate() {
                    let row = tile_start + local;
                    let visible = state.frontier + row as u64 + 1;
                    let mut encoded = vec![0; count * 4];
                    output.read_bytes(local * max_count * 4, &mut encoded)?;
                    let scores: Vec<_> = encoded
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| f32::from_le_bytes(*v))
                        .collect();
                    ensure!(
                        scores.iter().all(|v| v.is_finite() && *v >= 0.0),
                        "QSA index chunk score is nonfinite or negative"
                    );
                    if let Some(ref mut observe) = observer {
                        let mut encoded = vec![0; count * 16];
                        raw_output.read_bytes(local * max_count * 16, &mut encoded)?;
                        let raw: Vec<_> = encoded
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|v| f32::from_le_bytes(*v))
                            .collect();
                        observe(row, &scores, &raw)?;
                    }
                    if visible <= 2048 {
                        selected.extend(0..visible as i32);
                        selected.resize((row + 1) * SELECTED_WIDTH, -1);
                    } else {
                        selected.extend(select_scores(&scores, visible)?);
                    }
                }
            }
            Ok(QsaIndexChunk {
                state: next,
                selected,
                rows,
            })
        }

        /// Encode one QSA index update and causal selected token list on-device.
        /// The state remains unpublished until the request owner completes.
        pub(crate) fn encode_step_device(
            &mut self,
            batch: &mut Batch,
            state: &mut DeviceQsaIndexState,
            raw: &Buffer,
            queries: &Buffer,
            status: &DeviceStatus,
        ) -> Result<DeviceQsaSelection> {
            let frontier = state.frontier()?;
            ensure!(
                state.max_positions <= self.max_positions && frontier < state.max_positions,
                "QSA device index context limit differs"
            );
            ensure!(
                raw.byte_len() == INDEX_WIDTH * 2 && queries.byte_len() == QUERY_WORDS * 2,
                "QSA device index projection geometry differs"
            );
            if self.device.is_none() {
                self.device = Some(DeviceIndexKernels::new(&self.metal)?);
            }
            let kernels = self
                .device
                .as_ref()
                .expect("device index kernels initialized");
            ensure!(
                kernels.select.max_threads_per_threadgroup() >= 1024
                    && kernels.score.max_threads_per_threadgroup() >= 128,
                "QSA device index requires DS4 selector geometry"
            );
            let next = frontier + 1;
            let old_groups = usize::try_from(frontier / COMPRESS_RATIO as u64)?;
            let groups = usize::try_from(next / COMPRESS_RATIO as u64)?;
            if self.device_scratch.is_none() {
                self.device_scratch = Some(DeviceIndexScratch::new(&self.metal)?);
            }
            let scratch = self
                .device_scratch
                .as_mut()
                .expect("device index scratch initialized");
            if groups > BLOCK_BUDGET {
                scratch.fit_scores(
                    &self.metal,
                    groups,
                    usize::try_from(state.max_positions / COMPRESS_RATIO as u64)?,
                )?;
            }
            // Reuse is permitted only after the prior request completed.
            scratch.status.write_bytes(0, &0u32.to_le_bytes())?;
            scratch
                .group_count
                .write_bytes(0, &u32::try_from(groups)?.to_le_bytes())?;
            scratch.context.write_bytes(
                0,
                &[
                    u32::try_from(next)?.to_le_bytes(),
                    u32::try_from(groups)?.to_le_bytes(),
                ]
                .concat(),
            )?;
            let grown = state
                .compressed_growth_bytes(groups)?
                .map(|bytes| self.metal.shared_buffer(bytes))
                .transpose()?;

            state.valid = false;
            status.encode_bf16(batch, raw)?;
            status.encode_bf16(batch, queries)?;
            if let Some(new_compressed) = grown {
                if old_groups > 0 {
                    let bindings = [&state.compressed, &new_compressed]
                        .map(|buffer| Binding { buffer, offset: 0 });
                    // SAFETY: The published prefix is copied before this batch
                    // writes the next group; old backing remains retained.
                    unsafe {
                        batch.dispatch_1d(
                            &kernels.copy,
                            &bindings,
                            old_groups * INDEX_WIDTH * 2,
                            256,
                        )?;
                    }
                }
                state.compressed = new_compressed;
            }
            if groups > old_groups {
                let position = (frontier - 3) as i64;
                scratch.positions.write_bytes(
                    0,
                    &[position, position, position]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                scratch
                    .slot
                    .write_bytes(0, &u32::try_from(old_groups)?.to_le_bytes())?;
                let factor_bindings = [&scratch.positions, &scratch.cosine, &scratch.sine]
                    .map(|buffer| Binding { buffer, offset: 0 });
                let bindings = [
                    &state.pending,
                    raw,
                    &self.norm,
                    &scratch.cosine,
                    &scratch.sine,
                    &state.compressed,
                    &scratch.slot,
                    &scratch.status,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: The three saved rows and new row complete one group.
                // Factor generation and compression are ordered in this batch.
                unsafe {
                    batch.dispatch_1d(&self.factors, &factor_bindings, 32, 32)?;
                    batch.dispatch_1d(&kernels.compress, &bindings, 32, 32)?;
                }
            } else {
                scratch.slot.write_bytes(
                    0,
                    &u32::try_from(frontier % COMPRESS_RATIO as u64)?.to_le_bytes(),
                )?;
                let bindings = [raw, &state.pending, &scratch.slot, &scratch.status]
                    .map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: Only the next pending row is written.
                unsafe { batch.dispatch_1d(&kernels.pending, &bindings, INDEX_WIDTH, 128)? };
            }
            if groups > BLOCK_BUDGET {
                let bindings = [
                    queries,
                    &state.compressed,
                    &scratch.scores,
                    &scratch.group_count,
                    &scratch.status,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: One scalar scorer owns each complete group and reads
                // only the compressed prefix written before it.
                unsafe {
                    batch.dispatch_1d(&kernels.score, &bindings, groups, 128)?;
                }
                let bindings = [
                    &scratch.scores,
                    &scratch.group_count,
                    &scratch.selected_groups,
                    &scratch.status,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: A single 1024-thread group ranks the finite score row.
                unsafe { batch.dispatch_1d(&kernels.select, &bindings, 1024, 1024)? };
            }
            let bindings = [
                &scratch.selected_groups,
                &scratch.ids,
                &scratch.valid,
                &scratch.count,
                &scratch.context,
                &scratch.status,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: Each of the 2051 output slots is written exactly once;
            // sparse group IDs and causal addresses are checked before use.
            unsafe { batch.dispatch_1d(&kernels.expand, &bindings, SELECTED_WIDTH, 256)? };
            state.pending_frontier = Some(next);
            state.pending_status = Some(scratch.status.clone());
            Ok(DeviceQsaSelection {
                ids: scratch.ids.clone(),
                valid: scratch.valid.clone(),
                count: scratch.count.clone(),
                status: scratch.status.clone(),
            })
        }

        /// Encode a complete causal prefill index chunk without a host mirror.
        /// The selected IDs remain in device storage until attention consumes them.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn encode_chunk_device(
            &mut self,
            batch: &mut Batch,
            state: &mut DeviceQsaIndexState,
            raw: &Buffer,
            queries: &Buffer,
            rows: usize,
            scratch: &mut QsaChunkIndexScratch,
            status: &DeviceStatus,
        ) -> Result<DeviceQsaChunkSelection> {
            let frontier = state.frontier()?;
            ensure!(
                (1..=scratch.capacity).contains(&rows)
                    && state.max_positions <= self.max_positions
                    && frontier
                        .checked_add(rows as u64)
                        .is_some_and(|end| end <= state.max_positions),
                "QSA device chunk exceeds context capacity"
            );
            ensure!(
                raw.byte_len() >= rows * INDEX_WIDTH * 2
                    && queries.byte_len() >= rows * QUERY_WORDS * 2,
                "QSA device chunk projection geometry differs"
            );
            let old_groups = usize::try_from(frontier / COMPRESS_RATIO as u64)?;
            let next = frontier + rows as u64;
            let groups = usize::try_from(next / COMPRESS_RATIO as u64)?;
            ensure!(
                groups <= scratch.max_groups,
                "QSA score workspace context differs"
            );
            if self.device.is_none() {
                self.device = Some(DeviceIndexKernels::new(&self.metal)?);
            }
            let kernels = self
                .device
                .as_ref()
                .expect("device index kernels initialized");
            ensure!(
                kernels.select.max_threads_per_threadgroup() >= 1024
                    && kernels.score_mm.max_threads_per_threadgroup() >= 128,
                "QSA device chunk requires DS4 matrix scorer and selector geometry"
            );
            let flag = status.flag_buffer();
            let mut contexts = Vec::with_capacity(rows * 8);
            for row in 0..rows {
                let visible = usize::try_from(frontier)? + row + 1;
                contexts.extend_from_slice(&u32::try_from(visible)?.to_le_bytes());
                contexts.extend_from_slice(&u32::try_from(visible / COMPRESS_RATIO)?.to_le_bytes());
            }
            // Immutable per-layer causal counts are retained by the command
            // owner. The reusable scratch has no CPU writes while queued.
            let contexts = upload(&self.metal, &contexts)?;
            let grown = state
                .compressed_growth_bytes(groups)?
                .map(|bytes| self.metal.shared_buffer(bytes))
                .transpose()?;

            // Any encoding error leaves the session-owned state inaccessible.
            state.valid = false;
            if let Some(new_compressed) = grown {
                if old_groups > 0 {
                    let bindings = [&state.compressed, &new_compressed]
                        .map(|buffer| Binding { buffer, offset: 0 });
                    // SAFETY: Old published rows are copied before subsequent writes.
                    unsafe {
                        batch.dispatch_1d(
                            &kernels.copy,
                            &bindings,
                            old_groups * INDEX_WIDTH * 2,
                            256,
                        )?
                    };
                }
                state.compressed = new_compressed;
            }
            let compress_args = [
                u32::try_from(frontier)?,
                u32::try_from(rows)?,
                u32::try_from(old_groups)?,
                u32::try_from(groups)?,
            ];
            let compress_args = upload(
                &self.metal,
                &compress_args
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            if groups > old_groups {
                let bindings = [
                    &state.pending,
                    raw,
                    &self.norm,
                    &state.compressed,
                    flag,
                    &compress_args,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: Each group reads its four raw rows from either the
                // old incomplete pending group or this projected chunk.
                unsafe {
                    batch.dispatch_1d(
                        &kernels.compress_chunk,
                        &bindings,
                        (groups - old_groups) * 32,
                        32,
                    )?
                };
            }
            let pending_rows = usize::try_from(next)? % COMPRESS_RATIO;
            if pending_rows > 0 {
                let tail_args = [u32::try_from(frontier)?, u32::try_from(next)?];
                let tail_args = upload(
                    &self.metal,
                    &tail_args
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                let bindings = [raw, &state.pending, flag, &tail_args]
                    .map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: Completed-group readers run before this final
                // partial-block update on the same ordered command stream.
                unsafe {
                    batch.dispatch_1d(
                        &kernels.pending_tail,
                        &bindings,
                        pending_rows * INDEX_WIDTH,
                        128,
                    )?
                };
            }
            for tile in (0..rows).step_by(64) {
                let tile_rows = (rows - tile).min(64);
                ensure!(
                    scratch.selected_groups.byte_len() >= tile_rows * BLOCK_BUDGET * 4,
                    "QSA selected-group tile scratch is too small"
                );
                if groups > BLOCK_BUDGET && tile_rows > 2 {
                    let params = [
                        u32::try_from(tile_rows)?,
                        u32::try_from(groups)?,
                        u32::try_from(frontier + tile as u64)?,
                    ];
                    let args = upload(
                        &self.metal,
                        &params
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let bindings = [
                        Binding {
                            buffer: queries,
                            offset: tile * QUERY_WORDS * 2,
                        },
                        Binding {
                            buffer: &state.compressed,
                            offset: 0,
                        },
                        Binding {
                            buffer: &scratch.scores,
                            offset: 0,
                        },
                        Binding {
                            buffer: flag,
                            offset: 0,
                        },
                        Binding {
                            buffer: &args,
                            offset: 0,
                        },
                    ];
                    // SAFETY: Each matrix tile owns 16 query rows by 64
                    // group scores; later selectors read only causal groups.
                    unsafe {
                        batch.dispatch_1d(
                            &kernels.score_mm,
                            &bindings,
                            groups.div_ceil(64) * tile_rows.div_ceil(16) * 128,
                            128,
                        )?
                    };
                } else if groups > BLOCK_BUDGET {
                    for local in 0..tile_rows {
                        let row = tile + local;
                        let visible = usize::try_from(frontier)? + row + 1;
                        if visible / COMPRESS_RATIO <= BLOCK_BUDGET {
                            continue;
                        }
                        let bindings = [
                            Binding {
                                buffer: queries,
                                offset: row * QUERY_WORDS * 2,
                            },
                            Binding {
                                buffer: &state.compressed,
                                offset: 0,
                            },
                            Binding {
                                buffer: &scratch.scores,
                                offset: local * groups * 4,
                            },
                            Binding {
                                buffer: &contexts,
                                offset: row * 8 + 4,
                            },
                            Binding {
                                buffer: flag,
                                offset: 0,
                            },
                        ];
                        unsafe {
                            batch.dispatch_1d(
                                &kernels.score,
                                &bindings,
                                visible / COMPRESS_RATIO,
                                128,
                            )?
                        };
                    }
                }
                if groups > BLOCK_BUDGET {
                    let args = [u32::try_from(groups)?, u32::try_from(tile_rows)?];
                    let args = upload(
                        &self.metal,
                        &args
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let bindings = [
                        Binding {
                            buffer: &scratch.scores,
                            offset: 0,
                        },
                        Binding {
                            buffer: &contexts,
                            offset: tile * 8,
                        },
                        Binding {
                            buffer: &scratch.selected_groups,
                            offset: 0,
                        },
                        Binding {
                            buffer: flag,
                            offset: 0,
                        },
                        Binding {
                            buffer: &args,
                            offset: 0,
                        },
                    ];
                    // SAFETY: One independent DS4 1024-thread radix group
                    // owns each query row; dense rows return without reads.
                    unsafe {
                        batch.dispatch_1d(
                            &kernels.select_tile,
                            &bindings,
                            tile_rows * 1024,
                            1024,
                        )?
                    };
                }
                let row_arg = upload(&self.metal, &u32::try_from(tile_rows)?.to_le_bytes())?;
                let bindings = [
                    Binding {
                        buffer: &scratch.selected_groups,
                        offset: 0,
                    },
                    Binding {
                        buffer: &scratch.ids,
                        offset: tile * SELECTED_WIDTH * 4,
                    },
                    Binding {
                        buffer: &scratch.valid,
                        offset: tile * SELECTED_WIDTH,
                    },
                    Binding {
                        buffer: &scratch.count,
                        offset: tile * 4,
                    },
                    Binding {
                        buffer: &contexts,
                        offset: tile * 8,
                    },
                    Binding {
                        buffer: flag,
                        offset: 0,
                    },
                    Binding {
                        buffer: &row_arg,
                        offset: 0,
                    },
                ];
                // SAFETY: One group writes each 2051-slot causal row.
                unsafe {
                    batch.dispatch_1d(&kernels.expand_tile, &bindings, tile_rows * 256, 256)?
                };
            }
            state.pending_frontier = Some(next);
            state.pending_status = Some(flag.clone());
            Ok(DeviceQsaChunkSelection {
                ids: scratch.ids.clone(),
                valid: scratch.valid.clone(),
                count: scratch.count.clone(),
                status: flag.clone(),
                rows,
            })
        }
    }

    impl DeviceQsaIndexState {
        fn compressed_growth_bytes(&self, groups: usize) -> Result<Option<usize>> {
            compressed_growth_target(
                groups,
                self.compressed.byte_len(),
                usize::try_from(self.max_positions / COMPRESS_RATIO as u64)?,
            )
        }

        pub(crate) fn next_token_backing_growth(&self) -> Result<Option<(usize, usize)>> {
            let next = self
                .frontier()?
                .checked_add(1)
                .context("QSA frontier overflow")?;
            ensure!(
                next <= self.max_positions,
                "QSA context capacity is exhausted"
            );
            Ok(self
                .compressed_growth_bytes(usize::try_from(next / COMPRESS_RATIO as u64)?)?
                .map(|target| (self.compressed.byte_len(), target)))
        }
        pub(crate) fn zero(metal: &Metal, max_positions: usize) -> Result<Self> {
            Self::from_parts(metal, 0, &[], &[], max_positions)
        }

        pub(crate) fn from_parts(
            metal: &Metal,
            frontier: u64,
            compressed: &[u16],
            pending: &[u16],
            max_positions: usize,
        ) -> Result<Self> {
            let max_positions = u64::try_from(max_positions)?;
            ensure!(
                max_positions > 0 && max_positions <= MAX_POSITIONS && frontier <= max_positions,
                "QSA device index context limit differs"
            );
            QsaIndexState::validate_parts(frontier, compressed, pending)?;
            let compressed_capacity = compressed.len().max(INDEX_WIDTH) * 2;
            let compressed_buffer = metal.shared_buffer(compressed_capacity)?;
            let write_words = |buffer: &Buffer, words: &[u16]| -> Result<()> {
                let mut scratch = [0u8; 16_384];
                for start in (0..words.len()).step_by(scratch.len() / 2) {
                    let end = (start + scratch.len() / 2).min(words.len());
                    for (index, &word) in words[start..end].iter().enumerate() {
                        scratch[index * 2..index * 2 + 2].copy_from_slice(&word.to_le_bytes());
                    }
                    buffer.write_bytes(start * 2, &scratch[..(end - start) * 2])?;
                }
                Ok(())
            };
            write_words(&compressed_buffer, compressed)?;
            let pending_buffer = metal.shared_buffer(COMPRESS_RATIO * INDEX_WIDTH * 2)?;
            write_words(&pending_buffer, pending)?;
            Ok(Self {
                max_positions,
                frontier,
                valid: true,
                compressed: compressed_buffer,
                pending: pending_buffer,
                pending_frontier: None,
                pending_status: None,
            })
        }

        pub(crate) fn frontier(&self) -> Result<u64> {
            ensure!(self.valid, "QSA device index has an uncommitted append");
            Ok(self.frontier)
        }

        pub(crate) fn logical_parts_byte_lengths(&self) -> Result<(usize, usize)> {
            let frontier = self.frontier()?;
            Ok((
                usize::try_from(frontier / COMPRESS_RATIO as u64)? * INDEX_WIDTH * 2,
                usize::try_from(frontier % COMPRESS_RATIO as u64)? * INDEX_WIDTH * 2,
            ))
        }

        pub(crate) fn checkpoint_buffers(&self) -> Result<[(&Buffer, usize); 2]> {
            let (compressed, pending) = self.logical_parts_byte_lengths()?;
            Ok([(&self.compressed, compressed), (&self.pending, pending)])
        }

        /// Import validated, quiescent storage without a second host copy.
        pub(crate) fn from_buffers(
            frontier: u64,
            compressed: Buffer,
            pending: Buffer,
            max_positions: usize,
        ) -> Result<Self> {
            ensure!(
                max_positions > 0
                    && max_positions as u64 <= MAX_POSITIONS
                    && frontier <= max_positions as u64,
                "QSA restored frontier exceeds context"
            );
            let bytes = usize::try_from(frontier / COMPRESS_RATIO as u64)? * INDEX_WIDTH * 2;
            ensure!(
                compressed.byte_len() == bytes.max(INDEX_WIDTH * 2)
                    && pending.byte_len() == COMPRESS_RATIO * INDEX_WIDTH * 2,
                "QSA restored buffer geometry differs"
            );
            compressed.read_bytes(0, &mut [])?;
            pending.read_bytes(0, &mut [])?;
            Ok(Self {
                max_positions: max_positions as u64,
                frontier,
                valid: true,
                compressed,
                pending,
                pending_frontier: None,
                pending_status: None,
            })
        }

        pub(crate) fn export_host(&self) -> Result<QsaIndexState> {
            let (compressed_bytes, pending_bytes) = self.logical_parts_byte_lengths()?;
            let read = |buffer: &Buffer, count: usize| -> Result<Vec<u16>> {
                let mut words = Vec::new();
                words.try_reserve_exact(count / 2)?;
                let mut scratch = [0u8; 16_384];
                for start in (0..count).step_by(scratch.len()) {
                    let width = (count - start).min(scratch.len());
                    buffer.read_bytes(start, &mut scratch[..width])?;
                    words.extend(
                        scratch[..width]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|word| u16::from_le_bytes(*word)),
                    );
                }
                Ok(words)
            };
            QsaIndexState::from_parts(
                self.frontier,
                read(&self.compressed, compressed_bytes)?,
                read(&self.pending, pending_bytes)?,
            )
        }

        pub(crate) fn publish_after_wait(&mut self) -> Result<u64> {
            let next = self
                .pending_frontier
                .context("QSA device index has no encoded append")?;
            ensure!(!self.valid, "QSA device index append was already published");
            self.compressed.read_bytes(0, &mut [])?;
            self.pending.read_bytes(0, &mut [])?;
            let flag = self
                .pending_status
                .as_ref()
                .context("QSA device index lacks selector status")?;
            let mut bytes = [0; 4];
            flag.read_bytes(0, &mut bytes)?;
            ensure!(
                u32::from_le_bytes(bytes) == 0,
                "QSA device index selected invalid or nonfinite values"
            );
            self.frontier = next;
            self.pending_frontier = None;
            self.pending_status = None;
            self.valid = true;
            Ok(next)
        }
    }

    #[cfg(test)]
    mod device_tests {
        use super::*;

        #[test]
        fn device_index_and_kvarn_growth_retire_at_stream_boundaries() -> Result<()> {
            use crate::kvarn::{
                DeviceKvarnState, KvarnEncoder, SINK_BYTES, TAIL_BYTES, TILE_RECORD_BYTES,
                TOKEN_BYTES,
            };
            use crate::metal::{MemoryStats, Stream};
            use objc2::rc::autoreleasepool;

            const FRONTIER: usize = 12_287;
            const CONTEXT: usize = 16_384;
            const ROWS: usize = 2;
            const OLD_BYTES: u64 = (3_071 * INDEX_WIDTH * 2 + 30 * TILE_RECORD_BYTES) as u64;
            const NEW_BYTES: u64 = (4_096 * INDEX_WIDTH * 2 + 60 * TILE_RECORD_BYTES) as u64;
            const GROWTH_BYTES: u64 = NEW_BYTES - OLD_BYTES;

            fn state(metal: &Metal) -> Result<(DeviceQsaIndexState, DeviceKvarnState)> {
                let values = |len: usize| {
                    (0..len / 2)
                        .flat_map(|i| {
                            bf16::from_f32((i % 31) as f32 / 64.0 + 0.25)
                                .to_bits()
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>()
                };
                let sink = values(SINK_BYTES);
                let tail = values(TAIL_BYTES);
                Ok((
                    DeviceQsaIndexState::from_parts(
                        metal,
                        FRONTIER as u64,
                        &vec![0; 3_071 * INDEX_WIDTH],
                        &[0; 3 * INDEX_WIDTH],
                        CONTEXT,
                    )?,
                    DeviceKvarnState::restore(
                        metal,
                        CONTEXT,
                        FRONTIER,
                        &vec![0; 30 * TILE_RECORD_BYTES],
                        &sink,
                        &sink,
                        &tail,
                        &tail,
                    )?,
                ))
            }

            fn observe(
                metal: &Metal,
                case: &str,
                stage: &str,
                baseline: MemoryStats,
            ) -> Result<MemoryStats> {
                let stats = metal.memory_stats()?;
                println!(
                    "qsa_growth_lifetime case={case} stage={stage} live_requested={} peak_requested={} device_allocated={} device_delta={}",
                    stats.live_requested_buffer_bytes,
                    stats.lifetime_peak_requested_buffer_bytes,
                    stats.device_current_allocated_bytes,
                    i128::from(stats.device_current_allocated_bytes)
                        - i128::from(baseline.device_current_allocated_bytes),
                );
                Ok(stats)
            }

            for case in ["bounded", "queued", "error"] {
                let metal = Metal::new()?;
                let mut index = QsaIndex {
                    metal: metal.clone(),
                    rope: QsaRope::new(&metal)?,
                    norm: upload(&metal, &bytes(&[0; INDEX_WIDTH]))?,
                    compress: metal.compile_msl_strict(SOURCE, "qsa_index_compress")?,
                    score: metal.compile_msl_strict(SOURCE, "qsa_index_scores")?,
                    factors: metal.compile_msl_strict(
                        include_str!("../kernels/metal/qsa_projection.metal"),
                        "qsa_rope_factors",
                    )?,
                    max_positions: MAX_POSITIONS,
                    device: None,
                    device_scratch: None,
                };
                let encoder = KvarnEncoder::new(&metal)?;
                let raw = upload(&metal, &vec![0; ROWS * INDEX_WIDTH * 2])?;
                let queries = upload(&metal, &vec![0; ROWS * QUERY_WORDS * 2])?;
                let kv = upload(&metal, &vec![0; ROWS * TOKEN_BYTES])?;
                let status = DeviceStatus::new(&metal)?;
                let mut scratch = QsaChunkIndexScratch::new(&metal, ROWS, CONTEXT)?;
                let mut encode = |stream: &mut Stream,
                                  state: &mut (DeviceQsaIndexState, DeviceKvarnState)|
                 -> Result<()> {
                    let selected = index.encode_chunk_device(
                        stream.batch()?,
                        &mut state.0,
                        &raw,
                        &queries,
                        ROWS,
                        &mut scratch,
                        &status,
                    )?;
                    drop(selected);
                    state
                        .1
                        .encode_append_chunk(stream.batch()?, &encoder, &kv, &kv, ROWS)?;
                    assert!(state.0.frontier().is_err());
                    assert!(state.1.frontier().is_err());
                    Ok(())
                };
                let publish = |state: &mut (DeviceQsaIndexState, DeviceKvarnState)| -> Result<()> {
                    assert_eq!(state.0.publish_after_wait()?, (FRONTIER + ROWS) as u64);
                    assert_eq!(state.1.publish_after_wait()?, FRONTIER + ROWS);
                    assert_eq!(state.0.compressed.byte_len(), 4_096 * INDEX_WIDTH * 2);
                    assert_eq!(state.1.buffers()?[0].0.byte_len(), 60 * TILE_RECORD_BYTES);
                    Ok(())
                };

                // Initialize lazy kernels and seal scratch outside the measured pool.
                autoreleasepool(|_| -> Result<()> {
                    let mut warm = state(&metal)?;
                    let mut stream = metal.stream()?;
                    encode(&mut stream, &mut warm)?;
                    stream.wait()?;
                    status.check()?;
                    publish(&mut warm)
                })?;
                let fixture_bytes = metal.memory_stats()?.live_requested_buffer_bytes;
                let mut states = [state(&metal)?, state(&metal)?];
                let baseline = metal.memory_stats()?;
                observe(&metal, case, "baseline", baseline)?;

                // Both layers share this pool, matching a complete engine call.
                let inside = autoreleasepool(|_| -> Result<MemoryStats> {
                    let result = (|| -> Result<()> {
                        let mut stream = metal.stream()?;
                        for (layer, state) in states.iter_mut().enumerate() {
                            encode(&mut stream, state)?;
                            if case != "error" || layer == 0 {
                                stream.flush()?;
                            }
                            let before = observe(&metal, case, "encoded", baseline)?;
                            let prior_growth = if case == "bounded" {
                                layer as u64 * GROWTH_BYTES
                            } else {
                                layer as u64 * NEW_BYTES
                            };
                            // Small queued arguments may add bytes above retained old/new state.
                            assert!(
                                before.live_requested_buffer_bytes
                                    >= baseline.live_requested_buffer_bytes
                                        + prior_growth
                                        + NEW_BYTES
                            );
                            if case == "bounded" {
                                stream.wait()?;
                                status.check()?;
                                publish(state)?;
                                let after = observe(&metal, case, "retired", baseline)?;
                                assert_eq!(
                                    after.live_requested_buffer_bytes,
                                    baseline.live_requested_buffer_bytes
                                        + (layer as u64 + 1) * GROWTH_BYTES,
                                );
                            }
                        }
                        if case == "error" {
                            // A submitted first layer drains; the active second layer is canceled.
                            anyhow::bail!("injected post-encode failure");
                        }
                        stream.wait()?;
                        status.check()?;
                        if case == "queued" {
                            for state in &mut states {
                                publish(state)?;
                            }
                        }
                        Ok(())
                    })();
                    if case == "error" {
                        assert_eq!(
                            result.unwrap_err().to_string(),
                            "injected post-encode failure"
                        );
                        for state in &states {
                            assert!(state.0.frontier().is_err());
                            assert!(state.1.frontier().is_err());
                        }
                    } else {
                        result?;
                    }
                    let after = observe(&metal, case, "inside_pool", baseline)?;
                    // Failed states retain their unpublished four-byte seal flags.
                    let pending_flags = if case == "error" { 2 * 4 } else { 0 };
                    assert_eq!(
                        after.live_requested_buffer_bytes,
                        baseline.live_requested_buffer_bytes + 2 * GROWTH_BYTES + pending_flags,
                    );
                    Ok(after)
                })?;
                let outside = observe(&metal, case, "after_pool", baseline)?;
                assert_eq!(
                    inside.live_requested_buffer_bytes,
                    outside.live_requested_buffer_bytes,
                );
                println!(
                    "qsa_growth_lifetime case={case} device_delta_across_pool={}",
                    i128::from(outside.device_current_allocated_bytes)
                        - i128::from(inside.device_current_allocated_bytes),
                );
                drop(states);
                assert_eq!(
                    metal.memory_stats()?.live_requested_buffer_bytes,
                    fixture_bytes
                );
            }
            Ok(())
        }

        #[test]
        fn borrowed_import_exports_only_logical_index_rows_at_reduced_context() -> Result<()> {
            let metal = Metal::new()?;
            let group = vec![bf16::ONE.to_bits(); INDEX_WIDTH];
            let tail = vec![bf16::ZERO.to_bits(); INDEX_WIDTH];
            let state = DeviceQsaIndexState::from_parts(&metal, 5, &group, &tail, 4096)?;
            assert_eq!(state.frontier()?, 5);
            assert_eq!(state.logical_parts_byte_lengths()?, (256, 256));
            let host = state.export_host()?;
            assert_eq!(host.compressed_bf16(), group);
            assert_eq!(host.pending_bf16(), tail);
            assert!(DeviceQsaIndexState::from_parts(&metal, 5, &group, &tail, 4).is_err());
            assert!(DeviceQsaIndexState::zero(&metal, 0).is_err());
            Ok(())
        }

        #[test]
        fn device_step_accepts_engine_context_below_model_limit() -> Result<()> {
            let metal = Metal::new()?;
            let mut index = QsaIndex {
                metal: metal.clone(),
                rope: QsaRope::new(&metal)?,
                norm: metal.shared_buffer(INDEX_WIDTH * 2)?,
                compress: metal.compile_msl_strict(SOURCE, "qsa_index_compress")?,
                score: metal.compile_msl_strict(SOURCE, "qsa_index_scores")?,
                factors: metal.compile_msl_strict(
                    include_str!("../kernels/metal/qsa_projection.metal"),
                    "qsa_rope_factors",
                )?,
                max_positions: MAX_POSITIONS,
                device: None,
                device_scratch: None,
            };
            let mut state = DeviceQsaIndexState::zero(&metal, 4096)?;
            let raw = metal.shared_buffer(INDEX_WIDTH * 2)?;
            let queries = metal.shared_buffer(QUERY_WORDS * 2)?;
            raw.write_bytes(0, &vec![0; INDEX_WIDTH * 2])?;
            queries.write_bytes(0, &vec![0; QUERY_WORDS * 2])?;
            let status = DeviceStatus::new(&metal)?;
            let mut stream = metal.stream()?;
            let selected =
                index.encode_step_device(stream.batch()?, &mut state, &raw, &queries, &status)?;
            assert!(state.frontier().is_err());
            assert!(state.publish_after_wait().is_err());
            stream.wait()?;
            status.check()?;
            assert_eq!(state.publish_after_wait()?, 1);
            let mut count = [0; 4];
            selected.count.read_bytes(0, &mut count)?;
            assert_eq!(u32::from_le_bytes(count), 1);
            selected.ids.read_bytes(0, &mut count)?;
            assert_eq!(i32::from_le_bytes(count), 0);
            Ok(())
        }

        #[test]
        fn device_compression_preserves_released_bf16_group_bytes() -> Result<()> {
            let metal = Metal::new()?;
            let kernels = DeviceIndexKernels::new(&metal)?;
            let reference = metal.compile_msl_strict(SOURCE, "qsa_index_compress")?;
            let words: Vec<u16> = (0..4 * INDEX_WIDTH)
                .map(|i| bf16::from_f32((i % 29) as f32 / 32.0 - 0.4).to_bits())
                .collect();
            let raw = upload(&metal, &bytes(&words))?;
            let pending = upload(&metal, &bytes(&words[..3 * INDEX_WIDTH]))?;
            let current = upload(&metal, &bytes(&words[3 * INDEX_WIDTH..]))?;
            let weight = upload(&metal, &bytes(&vec![0; INDEX_WIDTH]))?;
            let cosine = upload(&metal, &bytes(&[bf16::ONE.to_bits(); 64]))?;
            let sine = upload(&metal, &bytes(&[0; 64]))?;
            let group = upload(&metal, &0u32.to_le_bytes())?;
            let status = upload(&metal, &0u32.to_le_bytes())?;
            let old = metal.shared_buffer(INDEX_WIDTH * 2)?;
            let new = metal.shared_buffer(INDEX_WIDTH * 2)?;
            let mut batch = metal.batch()?;
            let old_bindings =
                [&raw, &weight, &cosine, &sine, &old].map(|buffer| Binding { buffer, offset: 0 });
            let new_bindings = [
                &pending, &current, &weight, &cosine, &sine, &new, &group, &status,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: Fixed complete four-row source and disjoint outputs.
            unsafe {
                batch.dispatch_1d(&reference, &old_bindings, 32, 32)?;
                batch.dispatch_1d(&kernels.compress, &new_bindings, 32, 32)?;
            }
            batch.submit().wait()?;
            let mut expected = vec![0; INDEX_WIDTH * 2];
            let mut actual = vec![0; INDEX_WIDTH * 2];
            old.read_bytes(0, &mut expected)?;
            new.read_bytes(0, &mut actual)?;
            assert_eq!(actual, expected);
            let mut flag = [0; 4];
            status.read_bytes(0, &mut flag)?;
            assert_eq!(u32::from_le_bytes(flag), 0);
            Ok(())
        }

        #[test]
        fn device_chunk_compression_and_dense_selection_match_host_candidate() -> Result<()> {
            let metal = Metal::new()?;
            let mut index = QsaIndex {
                metal: metal.clone(),
                rope: QsaRope::new(&metal)?,
                norm: upload(&metal, &bytes(&[0; INDEX_WIDTH]))?,
                compress: metal.compile_msl_strict(SOURCE, "qsa_index_compress")?,
                score: metal.compile_msl_strict(SOURCE, "qsa_index_scores")?,
                factors: metal.compile_msl_strict(
                    include_str!("../kernels/metal/qsa_projection.metal"),
                    "qsa_rope_factors",
                )?,
                max_positions: MAX_POSITIONS,
                device: None,
                device_scratch: None,
            };
            let old_words: Vec<u16> = (0..3 * INDEX_WIDTH)
                .map(|i| bf16::from_f32((i % 37) as f32 / 32.0 - 0.5).to_bits())
                .collect();
            let raw_words: Vec<u16> = (0..10 * INDEX_WIDTH)
                .map(|i| bf16::from_f32((i % 43) as f32 / 64.0 - 0.3).to_bits())
                .collect();
            let query_words = vec![0u16; 10 * QUERY_WORDS];
            let host = QsaIndexState::from_parts(3, Vec::new(), old_words.clone())?;
            let expected = index.chunk(&host, &raw_words, &query_words)?;
            let mut device = DeviceQsaIndexState::from_parts(&metal, 3, &[], &old_words, 128)?;
            let raw = upload(&metal, &bytes(&raw_words))?;
            let queries = upload(&metal, &bytes(&query_words))?;
            let status = DeviceStatus::new(&metal)?;
            let mut scratch = QsaChunkIndexScratch::new(&metal, 10, 128)?;
            let mut stream = metal.stream()?;
            let selection = index.encode_chunk_device(
                stream.batch()?,
                &mut device,
                &raw,
                &queries,
                10,
                &mut scratch,
                &status,
            )?;
            assert!(device.frontier().is_err());
            stream.wait()?;
            status.check()?;
            assert_eq!(device.publish_after_wait()?, 13);
            let actual = device.export_host()?;
            assert_eq!(actual.compressed_bf16(), expected.state.compressed_bf16());
            assert_eq!(actual.pending_bf16(), expected.state.pending_bf16());
            let mut ids = vec![0u8; 10 * SELECTED_WIDTH * 4];
            selection.ids.read_bytes(0, &mut ids)?;
            let ids: Vec<i32> = ids
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| i32::from_le_bytes(*word))
                .collect();
            assert_eq!(ids, expected.selected);
            let mut counts = [0u8; 10 * 4];
            selection.count.read_bytes(0, &mut counts)?;
            for (row, count) in counts.as_chunks::<4>().0.iter().enumerate() {
                assert_eq!(u32::from_le_bytes(*count), (4 + row) as u32);
            }
            Ok(())
        }

        #[test]
        fn device_chunk_tied_sparse_rows_cross_causal_and_score_tile_boundaries() -> Result<()> {
            let metal = Metal::new()?;
            let mut index = QsaIndex {
                metal: metal.clone(),
                rope: QsaRope::new(&metal)?,
                norm: upload(&metal, &bytes(&[0; INDEX_WIDTH]))?,
                compress: metal.compile_msl_strict(SOURCE, "qsa_index_compress")?,
                score: metal.compile_msl_strict(SOURCE, "qsa_index_scores")?,
                factors: metal.compile_msl_strict(
                    include_str!("../kernels/metal/qsa_projection.metal"),
                    "qsa_rope_factors",
                )?,
                max_positions: MAX_POSITIONS,
                device: None,
                device_scratch: None,
            };
            let frontier = 2047usize;
            let rows = 70usize;
            let compressed = vec![0u16; (frontier / 4) * INDEX_WIDTH];
            let pending = vec![0u16; (frontier % 4) * INDEX_WIDTH];
            let host =
                QsaIndexState::from_parts(frontier as u64, compressed.clone(), pending.clone())?;
            let mut state = DeviceQsaIndexState::from_parts(
                &metal,
                frontier as u64,
                &compressed,
                &pending,
                4096,
            )?;
            let raw_words = vec![0u16; rows * INDEX_WIDTH];
            let query_words = vec![0u16; rows * QUERY_WORDS];
            let raw = upload(&metal, &bytes(&raw_words))?;
            let queries = upload(&metal, &bytes(&query_words))?;
            let status = DeviceStatus::new(&metal)?;
            let mut scratch = QsaChunkIndexScratch::new(&metal, rows, 4096)?;
            let mut stream = metal.stream()?;
            let selection = index.encode_chunk_device(
                stream.batch()?,
                &mut state,
                &raw,
                &queries,
                rows,
                &mut scratch,
                &status,
            )?;
            stream.wait()?;
            status.check()?;
            assert_eq!(state.publish_after_wait()?, (frontier + rows) as u64);
            let mut ids = vec![0; rows * SELECTED_WIDTH * 4];
            let mut valid = vec![0; rows * SELECTED_WIDTH];
            let mut counts = vec![0; rows * 4];
            selection.ids.read_bytes(0, &mut ids)?;
            selection.valid.read_bytes(0, &mut valid)?;
            selection.count.read_bytes(0, &mut counts)?;
            for row in 0..rows {
                let visible = frontier + row + 1;
                let groups = visible / 4;
                let mut expected: Vec<i32> = (0..visible.min(2048) as i32).collect();
                if groups > BLOCK_BUDGET {
                    expected.extend((groups * 4..visible).map(|id| id as i32));
                } else if visible > 2048 {
                    expected.extend((2048..visible).map(|id| id as i32));
                }
                let count = u32::from_le_bytes(counts[row * 4..row * 4 + 4].try_into()?);
                assert_eq!(count as usize, expected.len(), "row {row}");
                let words = ids[row * SELECTED_WIDTH * 4..(row + 1) * SELECTED_WIDTH * 4]
                    .as_chunks::<4>()
                    .0;
                let mut actual = Vec::new();
                for (slot, word) in words.iter().enumerate() {
                    let id = i32::from_le_bytes(*word);
                    assert_eq!(
                        valid[row * SELECTED_WIDTH + slot],
                        u8::from(slot < expected.len())
                    );
                    if slot < expected.len() {
                        actual.push(id);
                    } else {
                        assert_eq!(id, -1);
                    }
                }
                actual.sort_unstable();
                assert_eq!(actual, expected, "row {row}");
            }
            let exported = state.export_host()?;
            let reference = index.chunk(&host, &raw_words, &query_words)?;
            assert_eq!(
                exported.compressed_bf16(),
                reference.state.compressed_bf16()
            );
            assert_eq!(exported.pending_bf16(), reference.state.pending_bf16());
            Ok(())
        }

        #[test]
        fn device_radix_matches_selected_sets_and_lower_id_ties() -> Result<()> {
            let metal = Metal::new()?;
            let kernels = DeviceIndexKernels::new(&metal)?;
            for group_total in [513usize, 65_536] {
                for winner in [None, Some(group_total - 1)] {
                    let mut scores = vec![0.0f32; group_total];
                    if let Some(winner) = winner {
                        scores[winner] = 1.0;
                    }
                    let source = upload(
                        &metal,
                        &scores
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let groups = upload(&metal, &u32::try_from(group_total)?.to_le_bytes())?;
                    let frontier = u32::try_from(group_total * COMPRESS_RATIO)?;
                    let context = upload(
                        &metal,
                        &[
                            frontier.to_le_bytes(),
                            u32::try_from(group_total)?.to_le_bytes(),
                        ]
                        .concat(),
                    )?;
                    let selected = metal.shared_buffer(BLOCK_BUDGET * 4)?;
                    let ids = metal.shared_buffer(SELECTED_WIDTH * 4)?;
                    let valid = metal.shared_buffer(SELECTED_WIDTH)?;
                    let count = metal.shared_buffer(4)?;
                    let status = upload(&metal, &0u32.to_le_bytes())?;
                    let mut batch = metal.batch()?;
                    let select_bindings = [&source, &groups, &selected, &status]
                        .map(|buffer| Binding { buffer, offset: 0 });
                    let expand_bindings = [&selected, &ids, &valid, &count, &context, &status]
                        .map(|buffer| Binding { buffer, offset: 0 });
                    // SAFETY: Scratch sizes cover both the first sparse row and
                    // the maximum supported complete-group row.
                    unsafe {
                        batch.dispatch_1d(&kernels.select, &select_bindings, 1024, 1024)?;
                        batch.dispatch_1d(
                            &kernels.expand,
                            &expand_bindings,
                            SELECTED_WIDTH,
                            256,
                        )?;
                    }
                    batch.submit().wait()?;
                    let mut flag = [0; 4];
                    status.read_bytes(0, &mut flag)?;
                    assert_eq!(u32::from_le_bytes(flag), 0);
                    count.read_bytes(0, &mut flag)?;
                    assert_eq!(u32::from_le_bytes(flag), 2048);
                    let mut result = vec![0; SELECTED_WIDTH * 4];
                    ids.read_bytes(0, &mut result)?;
                    let mut selected_ids: Vec<i32> = result
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .take(2048)
                        .map(|word| i32::from_le_bytes(*word))
                        .collect();
                    if winner.is_some() {
                        let first = i32::try_from((group_total - 1) * COMPRESS_RATIO)?;
                        assert_eq!(
                            &selected_ids[..4],
                            &[first, first + 1, first + 2, first + 3]
                        );
                    }
                    selected_ids.sort_unstable();
                    assert_eq!(
                        selected_ids,
                        select_scores(&scores, u64::from(frontier))?[..2048]
                    );
                    let mut flags = vec![0; SELECTED_WIDTH];
                    valid.read_bytes(0, &mut flags)?;
                    assert!(flags[..2048].iter().all(|&flag| flag == 1));
                    assert!(flags[2048..].iter().all(|&flag| flag == 0));
                }
            }
            Ok(())
        }

        #[test]
        fn tiled_radix_expansion_keeps_causal_boundary_and_lower_id_ties() -> Result<()> {
            let metal = Metal::new()?;
            let kernels = DeviceIndexKernels::new(&metal)?;
            let groups = 513usize;
            let frontiers = [2051u32, 2052, 2053, 2054];
            let mut scores = vec![0.0f32; frontiers.len() * groups];
            scores[2 * groups + 512] = 1.0;
            scores[3 * groups + 512] = 1.0;
            let source = upload(
                &metal,
                &scores
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let context = upload(
                &metal,
                &frontiers
                    .iter()
                    .flat_map(|&frontier| {
                        [frontier, frontier / 4]
                            .into_iter()
                            .flat_map(|word| word.to_le_bytes())
                    })
                    .collect::<Vec<_>>(),
            )?;
            let selected = metal.shared_buffer(64 * BLOCK_BUDGET * 4)?;
            let ids = metal.shared_buffer(frontiers.len() * SELECTED_WIDTH * 4)?;
            let valid = metal.shared_buffer(frontiers.len() * SELECTED_WIDTH)?;
            let count = metal.shared_buffer(frontiers.len() * 4)?;
            let status = upload(&metal, &0u32.to_le_bytes())?;
            let selector_args = upload(
                &metal,
                &[groups as u32, frontiers.len() as u32]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let row_arg = upload(&metal, &(frontiers.len() as u32).to_le_bytes())?;
            let selector = [&source, &context, &selected, &status, &selector_args]
                .map(|buffer| Binding { buffer, offset: 0 });
            let expansion = [&selected, &ids, &valid, &count, &context, &status, &row_arg]
                .map(|buffer| Binding { buffer, offset: 0 });
            let mut batch = metal.batch()?;
            // SAFETY: Four score rows use a fixed 513-group stride and the
            // selected-group workspace covers the whole tile.
            unsafe {
                batch.dispatch_1d(
                    &kernels.select_tile,
                    &selector,
                    frontiers.len() * 1024,
                    1024,
                )?;
                batch.dispatch_1d(&kernels.expand_tile, &expansion, frontiers.len() * 256, 256)?;
            }
            batch.submit().wait()?;
            let mut flag = [0; 4];
            status.read_bytes(0, &mut flag)?;
            assert_eq!(u32::from_le_bytes(flag), 0);
            let mut raw_ids = vec![0; frontiers.len() * SELECTED_WIDTH * 4];
            ids.read_bytes(0, &mut raw_ids)?;
            let mut raw_counts = vec![0; frontiers.len() * 4];
            count.read_bytes(0, &mut raw_counts)?;
            for (row, &frontier) in frontiers.iter().enumerate() {
                let count = u32::from_le_bytes(raw_counts[row * 4..row * 4 + 4].try_into()?);
                let expected = select_scores(
                    &scores[row * groups..row * groups + frontier as usize / 4],
                    u64::from(frontier),
                )?;
                assert_eq!(
                    count as usize,
                    expected.iter().filter(|&&id| id >= 0).count()
                );
                let mut actual: Vec<i32> = raw_ids
                    [row * SELECTED_WIDTH * 4..(row + 1) * SELECTED_WIDTH * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| i32::from_le_bytes(*word))
                    .filter(|&id| id >= 0)
                    .collect();
                let mut expected: Vec<i32> = expected.into_iter().filter(|&id| id >= 0).collect();
                actual.sort_unstable();
                expected.sort_unstable();
                assert_eq!(actual, expected);
            }
            Ok(())
        }

        #[test]
        fn device_scalar_scores_match_signed_four_head_reference_and_flag_nan() -> Result<()> {
            let metal = Metal::new()?;
            let kernels = DeviceIndexKernels::new(&metal)?;
            let groups = 517usize;
            let query_words: Vec<u16> = (0..QUERY_WORDS)
                .map(|i| {
                    let code = (i * 7 % 31) as i32 - 15;
                    bf16::from_f32(code as f32 / 16.0).to_bits()
                })
                .collect();
            let key_words: Vec<u16> = (0..groups * INDEX_WIDTH)
                .map(|i| {
                    let block = i / INDEX_WIDTH;
                    let channel = i % INDEX_WIDTH;
                    let code = ((block * 13 + channel * 5) % 37) as i32 - 18;
                    bf16::from_f32(code as f32 / 32.0).to_bits()
                })
                .collect();
            let query = upload(&metal, &bytes(&query_words))?;
            let keys = upload(&metal, &bytes(&key_words))?;
            let group_count = upload(&metal, &u32::try_from(groups)?.to_le_bytes())?;
            let output = metal.shared_buffer(groups * 4)?;
            let status = upload(&metal, &0u32.to_le_bytes())?;
            let bindings = [&query, &keys, &output, &group_count, &status]
                .map(|buffer| Binding { buffer, offset: 0 });
            let mut batch = metal.batch()?;
            // SAFETY: Complete BF16 query and 517 complete BF16 key groups.
            unsafe { batch.dispatch_1d(&kernels.score, &bindings, groups, 128)? };
            batch.submit().wait()?;
            let mut flag = [0; 4];
            status.read_bytes(0, &mut flag)?;
            assert_eq!(u32::from_le_bytes(flag), 0);
            let mut raw = vec![0; groups * 4];
            output.read_bytes(0, &mut raw)?;
            let actual: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect();
            let reference: Vec<f64> = (0..groups)
                .map(|block| {
                    let mut total = 0.0f64;
                    for head in 0..4 {
                        let mut dot = 0.0f64;
                        for channel in 0..INDEX_WIDTH {
                            let q = bf16::from_bits(query_words[head * INDEX_WIDTH + channel]);
                            let k = bf16::from_bits(key_words[block * INDEX_WIDTH + channel]);
                            dot += f64::from(q.to_f32()) * f64::from(k.to_f32());
                        }
                        total += dot.max(0.0);
                    }
                    total / (INDEX_WIDTH as f64).sqrt()
                })
                .collect();
            assert!(reference.iter().any(|&score| score != reference[0]));
            for (got, expected) in actual.iter().zip(reference) {
                let tolerance = 1e-4 * expected.abs().max(1.0);
                assert!(
                    (f64::from(*got) - expected).abs() <= tolerance,
                    "QSA scalar score differs from four-head reference"
                );
            }

            query.write_bytes(0, &bf16::NAN.to_bits().to_le_bytes())?;
            status.write_bytes(0, &0u32.to_le_bytes())?;
            let mut batch = metal.batch()?;
            // SAFETY: The malformed BF16 value is checked before score publish.
            unsafe { batch.dispatch_1d(&kernels.score, &bindings, groups, 128)? };
            batch.submit().wait()?;
            status.read_bytes(0, &mut flag)?;
            assert_ne!(u32::from_le_bytes(flag), 0);
            Ok(())
        }

        #[test]
        fn device_matrix_scores_match_signed_causal_reference() -> Result<()> {
            let metal = Metal::new()?;
            let kernels = DeviceIndexKernels::new(&metal)?;
            let rows = 5usize;
            let groups = 517usize;
            let frontier = 2050usize;
            let query_words: Vec<u16> = (0..rows * QUERY_WORDS)
                .map(|i| {
                    let code = ((i * 11 + i / QUERY_WORDS * 7) % 43) as i32 - 21;
                    bf16::from_f32(code as f32 / 32.0).to_bits()
                })
                .collect();
            let key_words: Vec<u16> = (0..groups * INDEX_WIDTH)
                .map(|i| {
                    let code = ((i * 13 + i / INDEX_WIDTH * 5) % 47) as i32 - 23;
                    bf16::from_f32(code as f32 / 32.0).to_bits()
                })
                .collect();
            let queries = upload(&metal, &bytes(&query_words))?;
            let keys = upload(&metal, &bytes(&key_words))?;
            let scores = metal.shared_buffer(rows * groups * 4)?;
            let status = upload(&metal, &0u32.to_le_bytes())?;
            let args = upload(
                &metal,
                &[rows as u32, groups as u32, frontier as u32]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let bindings = [&queries, &keys, &scores, &status, &args]
                .map(|buffer| Binding { buffer, offset: 0 });
            let mut batch = metal.batch()?;
            // SAFETY: Five BF16 query rows and 517 compressed groups cover the
            // matrix scorer's 16-by-64 tile geometry and causal tail masking.
            unsafe {
                batch.dispatch_1d(
                    &kernels.score_mm,
                    &bindings,
                    groups.div_ceil(64) * rows.div_ceil(16) * 128,
                    128,
                )?
            };
            batch.submit().wait()?;
            let mut flag = [0; 4];
            status.read_bytes(0, &mut flag)?;
            assert_eq!(u32::from_le_bytes(flag), 0);
            let mut output = vec![0; rows * groups * 4];
            scores.read_bytes(0, &mut output)?;
            let unit_roundoff = f64::from(f32::EPSILON) * 0.5;
            let gamma_128 = 128.0 * unit_roundoff / (1.0 - 128.0 * unit_roundoff);
            let gamma_4 = 4.0 * unit_roundoff / (1.0 - 4.0 * unit_roundoff);
            let scale = 0.08838834764831845f64;
            let scale_f32 = f64::from(0.088_388_346_f32);
            let mut max_error = 0.0f64;
            for row in 0..rows {
                let visible = (frontier + row + 1) / 4;
                for block in 0..groups {
                    let actual =
                        f32::from_le_bytes(output[(row * groups + block) * 4..][..4].try_into()?);
                    if block >= visible {
                        assert_eq!(
                            actual.to_bits(),
                            0.0f32.to_bits(),
                            "future group was scored at row={row} block={block}"
                        );
                        continue;
                    }
                    let mut sum_relu = 0.0f64;
                    let mut sum_abs_products = 0.0f64;
                    for head in 0..4 {
                        let mut dot = 0.0f64;
                        for channel in 0..INDEX_WIDTH {
                            let q = bf16::from_bits(
                                query_words[row * QUERY_WORDS + head * INDEX_WIDTH + channel],
                            )
                            .to_f32() as f64;
                            let k = bf16::from_bits(key_words[block * INDEX_WIDTH + channel])
                                .to_f32() as f64;
                            assert_eq!(f64::from(half::f16::from_f32(q as f32).to_f32()), q);
                            assert_eq!(f64::from(half::f16::from_f32(k as f32).to_f32()), k);
                            dot += q * k;
                            sum_abs_products += (q * k).abs();
                        }
                        sum_relu += dot.max(0.0);
                    }
                    let expected = sum_relu * scale;
                    // Each 128-term FP32 dot is bounded by gamma_128 times
                    // its absolute products. Four ReLU heads, their addition,
                    // and the final FP32 scale account for the remaining bound.
                    let allowance = scale * (gamma_128 * sum_abs_products + gamma_4 * sum_relu)
                        + (scale_f32 - scale).abs() * sum_relu
                        + unit_roundoff * expected.abs();
                    let error = (f64::from(actual) - expected).abs();
                    max_error = max_error.max(error);
                    assert!(
                        error <= allowance,
                        "QSA matrix score row={row} block={block}: {actual} vs {expected}, error={error}, bound={allowance}"
                    );
                }
            }
            eprintln!("QSA matrix scorer maximum absolute error: {max_error:.9}");
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{
    DeviceQsaChunkSelection, DeviceQsaIndexState, DeviceQsaSelection, QsaChunkIndexScratch,
    QsaIndex, QsaIndexChunk, QsaIndexStep,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_query_uses_actual_capacity_and_clamped_allocation_target() {
        assert_eq!(compressed_growth_target(0, 256, 0).unwrap(), None);
        assert_eq!(
            compressed_growth_target(2048, 2048 * 256, 3072).unwrap(),
            None
        );
        assert_eq!(
            compressed_growth_target(2049, 2048 * 256, 3072).unwrap(),
            Some(3072 * 256)
        );
        assert_eq!(
            compressed_growth_target(2049, 4096 * 256, 4096).unwrap(),
            None
        );
        assert!(compressed_growth_target(4, 256, 3).is_err());
        assert!(compressed_growth_target(4, 257, 8).is_err());
    }

    #[test]
    fn candidate_lengths_follow_four_row_boundaries_and_reject_limits() {
        for (frontier, compressed, pending) in [
            (0, 0, 0),
            (1, 0, INDEX_WIDTH),
            (3, 0, 3 * INDEX_WIDTH),
            (4, INDEX_WIDTH, 0),
            (
                MAX_POSITIONS,
                MAX_POSITIONS as usize / COMPRESS_RATIO * INDEX_WIDTH,
                0,
            ),
        ] {
            assert_eq!(
                candidate_lengths(frontier, MAX_POSITIONS).unwrap(),
                (compressed, pending)
            );
        }
        assert!(candidate_lengths(4, 3).is_err());
        assert!(candidate_lengths(MAX_POSITIONS + 1, u64::MAX).is_err());
        assert!(candidate_lengths(u64::MAX, u64::MAX).is_err());
    }

    #[test]
    fn borrowed_candidate_copy_is_independent_and_invalid_input_leaves_source_unchanged() {
        let compressed = vec![bf16::ONE.to_bits(); INDEX_WIDTH];
        let pending = vec![bf16::ONE.to_bits(); INDEX_WIDTH];
        let original = compressed.clone();
        assert!(QsaIndexState::try_from_parts(4, &compressed, &pending).is_err());
        assert_eq!(compressed, original);

        let mut copied = QsaIndexState::try_from_parts(5, &compressed, &pending).unwrap();
        assert_eq!(copied.frontier(), 5);
        assert!(copied.compressed.capacity() >= INDEX_WIDTH);
        assert!(copied.pending.capacity() >= INDEX_WIDTH);
        copied.compressed[0] = bf16::ZERO.to_bits();
        assert_eq!(compressed, original);
        assert!(copied_words(&compressed, INDEX_WIDTH - 1).is_err());

        let mut reserved = copied_words(&compressed, 2 * INDEX_WIDTH).unwrap();
        let allocation = reserved.as_ptr();
        reserved.extend_from_slice(&compressed);
        assert_eq!(reserved.as_ptr(), allocation);
        assert_eq!(reserved.len(), 2 * INDEX_WIDTH);
    }

    #[test]
    fn maximum_context_restore_reserves_the_declared_index_geometry() {
        let (compressed_words, pending_words) =
            candidate_lengths(MAX_POSITIONS, MAX_POSITIONS).unwrap();
        assert_eq!(pending_words, 0);
        let compressed = vec![0u16; compressed_words];
        let restored = QsaIndexState::try_from_parts(MAX_POSITIONS, &compressed, &[]).unwrap();
        assert_eq!(restored.compressed.len(), compressed_words);
        assert!(restored.compressed.capacity() >= compressed_words);
        assert!(restored.pending.is_empty());
        assert_ne!(restored.compressed.as_ptr(), compressed.as_ptr());
    }

    #[test]
    fn borrowed_parts_validate_index_frontier_shape_and_finiteness() {
        QsaIndexState::validate_parts(0, &[], &[]).unwrap();
        assert!(QsaIndexState::validate_parts(MAX_POSITIONS + 1, &[], &[]).is_err());
        let mut compressed = vec![0u16; INDEX_WIDTH];
        QsaIndexState::validate_parts(4, &compressed, &[]).unwrap();
        assert!(QsaIndexState::validate_parts(4, &compressed[..INDEX_WIDTH - 1], &[]).is_err());
        assert!(QsaIndexState::validate_parts(4, &compressed, &[0]).is_err());
        compressed[INDEX_WIDTH - 1] = bf16::NAN.to_bits();
        assert!(QsaIndexState::validate_parts(4, &compressed, &[]).is_err());
        let mut pending = vec![0u16; 3 * INDEX_WIDTH];
        QsaIndexState::validate_parts(3, &[], &pending).unwrap();
        pending[0] = bf16::NEG_INFINITY.to_bits();
        assert!(QsaIndexState::validate_parts(3, &[], &pending).is_err());
    }

    #[test]
    fn chunks_check_rows_and_frontier_before_arithmetic_or_allocation() {
        {
            assert_eq!(
                chunk_rows(MAX_POSITIONS - 8192, 8192 * 128, 8192 * 512, MAX_POSITIONS).unwrap(),
                8192
            );
            assert!(
                chunk_rows(MAX_POSITIONS - 8191, 8192 * 128, 8192 * 512, MAX_POSITIONS).is_err()
            );
            assert!(chunk_rows(0, 8192 * 128, 8192 * 512 - 1, MAX_POSITIONS).is_err());
        }
        for rows in [2, 3, 64, 512, 1024, 2048, 2049, 4096] {
            assert_eq!(
                chunk_rows(0, rows * 128, rows * 512, MAX_POSITIONS).unwrap(),
                rows
            );
        }
        assert_eq!(
            chunk_rows(MAX_POSITIONS - 512, 512 * 128, 512 * 512, MAX_POSITIONS).unwrap(),
            512
        );
        assert_eq!(
            chunk_rows(MAX_POSITIONS - 4096, 4096 * 128, 4096 * 512, MAX_POSITIONS).unwrap(),
            4096
        );
        assert!(chunk_rows(MAX_POSITIONS - 4095, 4096 * 128, 4096 * 512, MAX_POSITIONS).is_err());
        for rows in [0, 1, crate::prefill_capacity::MAX_ROWS + 1] {
            assert!(chunk_rows(0, rows * 128, rows * 512, MAX_POSITIONS).is_err());
        }
        assert!(chunk_rows(0, 257, 1024, MAX_POSITIONS).is_err());
        assert!(chunk_rows(0, 256, 1023, MAX_POSITIONS).is_err());
        assert!(chunk_rows(MAX_POSITIONS - 1, 256, 1024, MAX_POSITIONS).is_err());
        assert!(chunk_rows(u64::MAX, 256, 1024, MAX_POSITIONS).is_err());
        assert!(chunk_rows(0, usize::MAX, usize::MAX, MAX_POSITIONS).is_err());
    }

    #[test]
    fn state_rejects_inconsistent_lengths_and_nonfinite_values() {
        assert!(QsaIndexState::from_parts(4, Vec::new(), Vec::new()).is_err());
        assert!(QsaIndexState::from_parts(1, Vec::new(), vec![0x7fc0; 128]).is_err());
        assert!(QsaIndexState::from_parts(3, Vec::new(), vec![0; 384]).is_ok());
    }

    #[test]
    fn ties_prefer_low_groups_and_keep_tail_after_selected_groups() {
        let selected = select_scores(&vec![0.0; 513], 2055).unwrap();
        assert_eq!(&selected[..2048], &(0..2048).collect::<Vec<i32>>());
        assert_eq!(&selected[2048..], &[2052, 2053, 2054]);
        assert!(select_scores(&[f32::NAN], 4).is_err());
    }
}

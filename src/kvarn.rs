//! KVarN adapted for fixed K4/V4 records and exact sink/tail state in Qwen attention.
//! See THIRD-PARTY-NOTICES for the Apache-2.0 license attribution.

use anyhow::{Context, Result, ensure};
use half::{bf16, f16};

#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
pub use device::DeviceKvarnState;

pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 256;
pub const TILE_TOKENS: usize = 128;
pub const SINK_TOKENS: usize = 128;
pub const EXACT_SUFFIX: usize = 8192;
pub const TAIL_CAPACITY: usize = EXACT_SUFFIX + TILE_TOKENS;
pub const HEAD_RECORD_BYTES: usize = 35072;
pub const TILE_RECORD_BYTES: usize = KV_HEADS * HEAD_RECORD_BYTES;
pub const TOKEN_BYTES: usize = KV_HEADS * HEAD_DIM * 2;
pub const SINK_BYTES: usize = SINK_TOKENS * TOKEN_BYTES;
pub const TAIL_BYTES: usize = TAIL_CAPACITY * TOKEN_BYTES;
/// One encoder-owned seal workspace, shared by ordered layers and requests.
pub const DEVICE_SEAL_WORKSPACE_BYTES: usize = 1_060_872;
const MAX_CONTEXT_TOKENS: usize = 262_144;
const MAX_SELECTED_ROWS: usize = u32::MAX as usize / (KV_HEADS * HEAD_DIM);

const CODES_BYTES: usize = TILE_TOKENS * HEAD_DIM / 2;
const K_SCALE: usize = CODES_BYTES;
const K_ZERO: usize = K_SCALE + HEAD_DIM * 2;
const K_TOKEN_SCALE: usize = K_ZERO + HEAD_DIM * 2;
const V_CODES: usize = K_TOKEN_SCALE + TILE_TOKENS * 2;
const V_CHANNEL_SCALE: usize = V_CODES + CODES_BYTES;
const V_TOKEN_SCALE: usize = V_CHANNEL_SCALE + HEAD_DIM * 2;
const V_ZERO: usize = V_TOKEN_SCALE + TILE_TOKENS * 2;

fn selected_row_bytes(rows: usize) -> Result<usize> {
    ensure!(
        rows <= MAX_SELECTED_ROWS,
        "KVarN selected rows exceed Metal uint output indexing"
    );
    rows.checked_mul(TOKEN_BYTES)
        .context("KVarN selected-row byte count overflow")
}

/// Return exact sink end, packed body end, and exact tail start.
pub fn partition(frontier: usize) -> (usize, usize, usize) {
    let sink = frontier.min(SINK_TOKENS);
    let body = if frontier <= SINK_TOKENS {
        frontier
    } else {
        let eligible = frontier.saturating_sub(SINK_TOKENS + EXACT_SUFFIX);
        SINK_TOKENS + eligible / TILE_TOKENS * TILE_TOKENS
    };
    (sink, body, body)
}

fn record_growth_target(
    required_records: usize,
    capacity_bytes: usize,
    max_records: usize,
) -> Result<Option<usize>> {
    ensure!(
        required_records <= max_records,
        "KVarN record count exceeds context capacity"
    );
    let required_bytes = required_records
        .checked_mul(TILE_RECORD_BYTES)
        .context("KVarN record byte count overflow")?;
    if capacity_bytes >= required_bytes {
        return Ok(None);
    }
    let doubled = (capacity_bytes / TILE_RECORD_BYTES)
        .max(1)
        .checked_mul(2)
        .context("KVarN record growth overflow")?;
    let target_records = required_records.max(doubled.min(max_records));
    let target_bytes = target_records
        .checked_mul(TILE_RECORD_BYTES)
        .context("KVarN record reservation overflow")?;
    Ok(Some(target_bytes))
}

fn record_growth_at_frontier(
    next: usize,
    capacity_bytes: usize,
    max_context: usize,
) -> Result<Option<usize>> {
    ensure!(next <= max_context, "KVarN context capacity is exhausted");
    let records = partition(next).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
    let maximum = partition(max_context).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
    record_growth_target(records, capacity_bytes, maximum)
}

#[test]
fn record_growth_query_distinguishes_seals_from_backing_growth() {
    assert_eq!(record_growth_at_frontier(8447, 2, 9216).unwrap(), None);
    assert_eq!(
        record_growth_at_frontier(8448, 2, 9216).unwrap(),
        Some(2 * TILE_RECORD_BYTES)
    );
    assert_eq!(
        record_growth_at_frontier(8576, 2 * TILE_RECORD_BYTES, 9216).unwrap(),
        None
    );
    assert_eq!(
        record_growth_at_frontier(8704, 2 * TILE_RECORD_BYTES, 9216).unwrap(),
        Some(4 * TILE_RECORD_BYTES)
    );
    assert_eq!(
        record_growth_at_frontier(8704, 4 * TILE_RECORD_BYTES, 9216).unwrap(),
        None
    );
    assert!(record_growth_at_frontier(9217, 4 * TILE_RECORD_BYTES, 9216).is_err());
}

fn finite_bf16(bytes: &[u8], name: &str) -> Result<()> {
    ensure!(
        bytes.len().is_multiple_of(2),
        "{name} has a partial BF16 word"
    );
    ensure!(
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|word| bf16::from_bits(u16::from_le_bytes(*word)).is_finite()),
        "{name} contains nonfinite BF16 values"
    );
    Ok(())
}

fn read_bf16(bytes: &[u8], word: usize) -> f32 {
    let offset = word * 2;
    bf16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn write_f16(bytes: &mut [u8], offset: usize, value: f32) -> Result<()> {
    let half = f16::from_f32(value);
    ensure!(half.is_finite(), "KVarN metadata overflows FP16");
    bytes[offset..offset + 2].copy_from_slice(&half.to_bits().to_le_bytes());
    Ok(())
}

pub(crate) fn finite_record_metadata(records: &[u8]) -> Result<()> {
    ensure!(
        records.len().is_multiple_of(TILE_RECORD_BYTES),
        "KVarN records contain a partial tile"
    );
    for head in records.as_chunks::<HEAD_RECORD_BYTES>().0 {
        for (start, end) in [
            (K_SCALE, K_TOKEN_SCALE + TILE_TOKENS * 2),
            (V_CHANNEL_SCALE, HEAD_RECORD_BYTES),
        ] {
            ensure!(
                head[start..end]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|word| f16::from_bits(u16::from_le_bytes(*word)).is_finite()),
                "KVarN record metadata contains nonfinite FP16 values"
            );
        }
    }
    Ok(())
}

fn hadamard(row: &mut [f32; HEAD_DIM]) {
    let mut stride = 1;
    while stride < HEAD_DIM {
        for base in (0..HEAD_DIM).step_by(2 * stride) {
            for offset in 0..stride {
                let a = row[base + offset];
                let b = row[base + offset + stride];
                row[base + offset] = a + b;
                row[base + offset + stride] = a - b;
            }
        }
        stride *= 2;
    }
    for value in row.iter_mut() {
        *value *= 0.0625;
    }
}

fn quantize_rows(values: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut codes = vec![0u8; values.len()];
    let mut scales = vec![0.0f32; rows];
    let mut zeros = vec![0.0f32; rows];
    for row in 0..rows {
        let values = &values[row * cols..(row + 1) * cols];
        let low = values.iter().copied().fold(f32::INFINITY, f32::min);
        let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = ((high - low) / 15.0).max(1e-10);
        scales[row] = scale;
        zeros[row] = low;
        for (col, value) in values.iter().enumerate() {
            codes[row * cols + col] =
                ((*value - low) / scale).round_ties_even().clamp(0.0, 15.0) as u8;
        }
    }
    (codes, scales, zeros)
}

fn encode_tile_inner<F>(keys: &[u8], values: &[u8], normalize: &F) -> Result<Vec<u8>>
where
    F: Fn(&[f32], usize, usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)>,
{
    ensure!(
        keys.len() == TILE_TOKENS * TOKEN_BYTES,
        "KVarN tile key byte count differs"
    );
    ensure!(
        values.len() == keys.len(),
        "KVarN tile value byte count differs"
    );
    finite_bf16(keys, "KVarN tile keys")?;
    finite_bf16(values, "KVarN tile values")?;
    let mut records = vec![0u8; TILE_RECORD_BYTES];
    for head in 0..KV_HEADS {
        let record = &mut records[head * HEAD_RECORD_BYTES..(head + 1) * HEAD_RECORD_BYTES];
        let mut key_rotated = vec![0.0f32; TILE_TOKENS * HEAD_DIM];
        let mut value_rotated = vec![0.0f32; TILE_TOKENS * HEAD_DIM];
        for token in 0..TILE_TOKENS {
            let mut key_row = [0.0f32; HEAD_DIM];
            let mut value_row = [0.0f32; HEAD_DIM];
            for channel in 0..HEAD_DIM {
                let word = (token * KV_HEADS + head) * HEAD_DIM + channel;
                key_row[channel] = read_bf16(keys, word);
                value_row[channel] = read_bf16(values, word);
            }
            hadamard(&mut key_row);
            hadamard(&mut value_row);
            key_rotated[token * HEAD_DIM..(token + 1) * HEAD_DIM].copy_from_slice(&key_row);
            value_rotated[token * HEAD_DIM..(token + 1) * HEAD_DIM].copy_from_slice(&value_row);
        }
        let mut key_matrix = vec![0.0f32; HEAD_DIM * TILE_TOKENS];
        for token in 0..TILE_TOKENS {
            for channel in 0..HEAD_DIM {
                key_matrix[channel * TILE_TOKENS + token] = key_rotated[token * HEAD_DIM + channel];
            }
        }
        let (balanced_k, token_scale_k, channel_scale_k) =
            normalize(&key_matrix, HEAD_DIM, TILE_TOKENS)?;
        let (codes_k, rtn_scale_k, rtn_zero_k) = quantize_rows(&balanced_k, HEAD_DIM, TILE_TOKENS);
        for token in 0..TILE_TOKENS {
            for pair in 0..HEAD_DIM / 2 {
                let low = codes_k[(pair * 2) * TILE_TOKENS + token];
                let high = codes_k[(pair * 2 + 1) * TILE_TOKENS + token];
                record[token * (HEAD_DIM / 2) + pair] = low | (high << 4);
            }
        }
        for channel in 0..HEAD_DIM {
            write_f16(
                record,
                K_SCALE + channel * 2,
                channel_scale_k[channel] * rtn_scale_k[channel],
            )?;
            write_f16(
                record,
                K_ZERO + channel * 2,
                channel_scale_k[channel] * rtn_zero_k[channel],
            )?;
        }
        for (token, &scale) in token_scale_k.iter().enumerate() {
            write_f16(record, K_TOKEN_SCALE + token * 2, scale)?;
        }

        let (balanced_v, channel_scale_v, token_scale_v) =
            normalize(&value_rotated, TILE_TOKENS, HEAD_DIM)?;
        let (codes_v, rtn_scale_v, rtn_zero_v) = quantize_rows(&balanced_v, TILE_TOKENS, HEAD_DIM);
        for token in 0..TILE_TOKENS {
            for pair in 0..HEAD_DIM / 2 {
                let low = codes_v[token * HEAD_DIM + pair * 2];
                let high = codes_v[token * HEAD_DIM + pair * 2 + 1];
                record[V_CODES + token * (HEAD_DIM / 2) + pair] = low | (high << 4);
            }
        }
        for (channel, &scale) in channel_scale_v.iter().enumerate() {
            write_f16(record, V_CHANNEL_SCALE + channel * 2, scale)?;
        }
        for token in 0..TILE_TOKENS {
            write_f16(
                record,
                V_TOKEN_SCALE + token * 2,
                token_scale_v[token] * rtn_scale_v[token],
            )?;
            write_f16(
                record,
                V_ZERO + token * 2,
                token_scale_v[token] * rtn_zero_v[token],
            )?;
        }
    }
    Ok(records)
}

/// Fixed-capacity single-owner K/V storage. Index keys are a separate seam.
pub struct KvarnState {
    max_context: usize,
    frontier: usize,
    records: Vec<u8>,
    sink_keys: Vec<u8>,
    sink_values: Vec<u8>,
    tail_keys: Vec<u8>,
    tail_values: Vec<u8>,
}

impl KvarnState {
    pub fn zero(max_context: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_CONTEXT_TOKENS).contains(&max_context),
            "KVarN context capacity is outside the released range"
        );
        Ok(Self {
            max_context,
            frontier: 0,
            records: Vec::new(),
            sink_keys: vec![0; SINK_BYTES],
            sink_values: vec![0; SINK_BYTES],
            tail_keys: vec![0; TAIL_BYTES],
            tail_values: vec![0; TAIL_BYTES],
        })
    }

    pub fn restore(
        max_context: usize,
        frontier: usize,
        records: &[u8],
        sink_keys: &[u8],
        sink_values: &[u8],
        tail_keys: &[u8],
        tail_values: &[u8],
    ) -> Result<Self> {
        Self::validate_parts(
            max_context,
            frontier,
            records,
            sink_keys,
            sink_values,
            tail_keys,
            tail_values,
        )?;
        let mut state = Self::zero(max_context)?;
        state
            .records
            .try_reserve_exact(records.len())
            .context("KVarN record restore reservation failed")?;
        state.frontier = frontier;
        state.records.extend_from_slice(records);
        state.sink_keys.copy_from_slice(sink_keys);
        state.sink_values.copy_from_slice(sink_values);
        state.tail_keys.copy_from_slice(tail_keys);
        state.tail_values.copy_from_slice(tail_values);
        Ok(state)
    }

    /// Validate borrowed storage without allocating a restored state.
    pub(crate) fn validate_parts(
        max_context: usize,
        frontier: usize,
        records: &[u8],
        sink_keys: &[u8],
        sink_values: &[u8],
        tail_keys: &[u8],
        tail_values: &[u8],
    ) -> Result<()> {
        ensure!(
            (1..=MAX_CONTEXT_TOKENS).contains(&max_context),
            "KVarN context capacity is outside the released range"
        );
        ensure!(
            frontier <= max_context,
            "KVarN frontier exceeds context capacity"
        );
        let (_, body, _) = partition(frontier);
        let expected = body.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
        ensure!(
            records.len() == expected * TILE_RECORD_BYTES,
            "KVarN record count differs from frontier"
        );
        finite_record_metadata(records)?;
        ensure!(
            sink_keys.len() == SINK_BYTES && sink_values.len() == SINK_BYTES,
            "KVarN sink geometry differs"
        );
        ensure!(
            tail_keys.len() == TAIL_BYTES && tail_values.len() == TAIL_BYTES,
            "KVarN tail ring geometry differs"
        );
        finite_bf16(sink_keys, "KVarN sink keys")?;
        finite_bf16(sink_values, "KVarN sink values")?;
        finite_bf16(tail_keys, "KVarN tail keys")?;
        finite_bf16(tail_values, "KVarN tail values")?;
        Ok(())
    }

    pub fn frontier(&self) -> usize {
        self.frontier
    }
    pub fn context_limit(&self) -> usize {
        self.max_context
    }
    pub fn body_frontier(&self) -> usize {
        partition(self.frontier).1
    }
    pub fn record_count(&self) -> usize {
        self.records.len() / TILE_RECORD_BYTES
    }
    pub fn records(&self) -> &[u8] {
        &self.records
    }
    pub fn sink_keys(&self) -> &[u8] {
        &self.sink_keys
    }
    pub fn sink_values(&self) -> &[u8] {
        &self.sink_values
    }
    pub fn tail_keys(&self) -> &[u8] {
        &self.tail_keys
    }
    pub fn tail_values(&self) -> &[u8] {
        &self.tail_values
    }

    pub fn safe_chunk_tokens(&self, requested: usize) -> Result<usize> {
        ensure!(requested > 0, "KVarN append width must be positive");
        let next_seal = SINK_TOKENS + EXACT_SUFFIX + (self.record_count() + 1) * TILE_TOKENS;
        Ok(requested.min(next_seal - self.frontier))
    }

    /// Append complete finite head-major BF16 K/V rows [2, tokens, 256], sealing at most one tile.
    #[cfg(target_os = "macos")]
    pub fn append(&mut self, encoder: &KvarnEncoder, keys: &[u8], values: &[u8]) -> Result<()> {
        ensure!(
            !keys.is_empty() && keys.len().is_multiple_of(TOKEN_BYTES),
            "KVarN append keys must contain complete tokens"
        );
        ensure!(
            values.len() == keys.len(),
            "KVarN append values differ in shape"
        );
        finite_bf16(keys, "KVarN append keys")?;
        finite_bf16(values, "KVarN append values")?;
        let tokens = keys.len() / TOKEN_BYTES;
        let next = self
            .frontier
            .checked_add(tokens)
            .context("KVarN frontier overflow")?;
        ensure!(
            next <= self.max_context,
            "KVarN context capacity is exhausted"
        );
        ensure!(
            tokens <= self.safe_chunk_tokens(tokens)?,
            "KVarN append crosses a tile seal; split at safe_chunk_tokens"
        );
        let (_, next_body, _) = partition(next);
        let next_records = next_body.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
        ensure!(
            next_records <= self.record_count() + 1,
            "KVarN append crosses multiple tile seals"
        );
        let max_records = partition(self.max_context).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
        // Reserve before encoding or changing the sink and tail. Doubling in
        // record units avoids a backing move at every tile seal.
        if let Some(target_bytes) =
            record_growth_target(next_records, self.records.capacity(), max_records)?
        {
            let additional = target_bytes
                .checked_sub(self.records.len())
                .context("KVarN record reservation is below the occupied length")?;
            self.records
                .try_reserve_exact(additional)
                .context("KVarN record append reservation failed")?;
        }
        let record = if next_records > self.record_count() {
            let tile_start = SINK_TOKENS + self.record_count() * TILE_TOKENS;
            let mut tile_keys = vec![0; TILE_TOKENS * TOKEN_BYTES];
            let mut tile_values = vec![0; TILE_TOKENS * TOKEN_BYTES];
            for tile_token in 0..TILE_TOKENS {
                let logical = tile_start + tile_token;
                for head in 0..KV_HEADS {
                    for channel in 0..HEAD_DIM {
                        let target = ((tile_token * KV_HEADS + head) * HEAD_DIM + channel) * 2;
                        let source = if logical < self.frontier {
                            let offset = (head * TAIL_CAPACITY
                                + (logical - SINK_TOKENS) % TAIL_CAPACITY)
                                * HEAD_DIM
                                * 2
                                + channel * 2;
                            (
                                &self.tail_keys[offset..offset + 2],
                                &self.tail_values[offset..offset + 2],
                            )
                        } else {
                            let offset = (head * tokens * HEAD_DIM
                                + (logical - self.frontier) * HEAD_DIM
                                + channel)
                                * 2;
                            (&keys[offset..offset + 2], &values[offset..offset + 2])
                        };
                        tile_keys[target..target + 2].copy_from_slice(source.0);
                        tile_values[target..target + 2].copy_from_slice(source.1);
                    }
                }
            }
            let record = encoder.encode_tile(&tile_keys, &tile_values)?;
            ensure!(
                record.len() == TILE_RECORD_BYTES,
                "KVarN encoder returned an incomplete tile record"
            );
            Some(record)
        } else {
            None
        };
        // No fallible operation follows this point. Complete the physical
        // writes before publishing the new logical frontier.
        for token in 0..tokens {
            let logical = self.frontier + token;
            for head in 0..KV_HEADS {
                let source = (head * tokens + token) * HEAD_DIM * 2;
                let (target_keys, target_values, index) = if logical < SINK_TOKENS {
                    (
                        &mut self.sink_keys,
                        &mut self.sink_values,
                        head * SINK_TOKENS + logical,
                    )
                } else {
                    let slot = (logical - SINK_TOKENS) % TAIL_CAPACITY;
                    (
                        &mut self.tail_keys,
                        &mut self.tail_values,
                        head * TAIL_CAPACITY + slot,
                    )
                };
                let target = index * HEAD_DIM * 2;
                target_keys[target..target + HEAD_DIM * 2]
                    .copy_from_slice(&keys[source..source + HEAD_DIM * 2]);
                target_values[target..target + HEAD_DIM * 2]
                    .copy_from_slice(&values[source..source + HEAD_DIM * 2]);
            }
        }
        if let Some(record) = record {
            self.records.extend_from_slice(&record);
        }
        self.frontier = next;
        Ok(())
    }
}

/// BF16 rows are head-major [2, selected_width, 256].
pub struct KvarnSelectedRows {
    pub keys_bf16: Vec<u8>,
    pub values_bf16: Vec<u8>,
    pub logical_ids: Vec<i32>,
    pub valid: Vec<bool>,
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::{
        DEVICE_SEAL_WORKSPACE_BYTES, HEAD_DIM, KV_HEADS, KvarnSelectedRows, KvarnState,
        SINK_TOKENS, TILE_RECORD_BYTES, TILE_TOKENS,
    };
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use anyhow::{Context, Result, ensure};

    const SOURCE: &str = include_str!("../kernels/metal/kvarn_decode.metal");
    const ENCODE_SOURCE: &str = include_str!("../kernels/metal/kvarn_encode.metal");

    unsafe fn launch(
        batch: &mut Batch,
        pipeline: &Pipeline,
        buffers: &[&Buffer],
        args: &[u8],
        grid: usize,
        group: usize,
    ) -> Result<()> {
        let bindings: Vec<_> = buffers
            .iter()
            .map(|buffer| Binding { buffer, offset: 0 })
            .collect();
        // SAFETY: Callers supply fixed-size, typed buffers for each pipeline.
        unsafe { batch.dispatch_1d_with_inline(pipeline, &bindings, &[args], grid, group) }
    }

    struct NormalizeScratch {
        current: Buffer,
        balanced: Buffer,
        col_std: Buffer,
        row_std: Buffer,
        log_col: Buffer,
        log_row: Buffer,
        best_col: Buffer,
        best_row: Buffer,
        best_score: Buffer,
    }

    impl NormalizeScratch {
        fn new(metal: &Metal, rows: usize, cols: usize) -> Result<Self> {
            ensure!(
                [128, 256].contains(&rows) && [128, 256].contains(&cols),
                "KVarN normalization workspace geometry differs"
            );
            Ok(Self {
                current: metal.shared_buffer(KV_HEADS * HEAD_DIM * TILE_TOKENS * 4)?,
                balanced: metal.shared_buffer(KV_HEADS * HEAD_DIM * TILE_TOKENS * 4)?,
                col_std: metal.shared_buffer(KV_HEADS * cols * 4)?,
                row_std: metal.shared_buffer(KV_HEADS * rows * 4)?,
                log_col: metal.shared_buffer(KV_HEADS * cols * 4)?,
                log_row: metal.shared_buffer(KV_HEADS * rows * 4)?,
                best_col: metal.shared_buffer(KV_HEADS * cols * 4)?,
                best_row: metal.shared_buffer(KV_HEADS * rows * 4)?,
                best_score: metal.shared_buffer(KV_HEADS * 4)?,
            })
        }

        fn byte_len(&self) -> usize {
            [
                &self.current,
                &self.balanced,
                &self.col_std,
                &self.row_std,
                &self.log_col,
                &self.log_row,
                &self.best_col,
                &self.best_row,
                &self.best_score,
            ]
            .iter()
            .map(|b| b.byte_len())
            .sum()
        }
    }

    struct SealScratch {
        keys: Buffer,
        values: Buffer,
        normalize: NormalizeScratch,
    }

    impl SealScratch {
        fn new(metal: &Metal) -> Result<Self> {
            let scratch = Self {
                keys: metal.shared_buffer(KV_HEADS * HEAD_DIM * TILE_TOKENS * 4)?,
                values: metal.shared_buffer(KV_HEADS * HEAD_DIM * TILE_TOKENS * 4)?,
                normalize: NormalizeScratch::new(metal, HEAD_DIM, HEAD_DIM)?,
            };
            ensure!(
                scratch.keys.byte_len() + scratch.values.byte_len() + scratch.normalize.byte_len()
                    == DEVICE_SEAL_WORKSPACE_BYTES,
                "KVarN seal workspace differs from admission accounting"
            );
            Ok(scratch)
        }
    }

    pub struct KvarnEncoder {
        metal: Metal,
        std_col: Pipeline,
        std_row: Pipeline,
        init: Pipeline,
        score_init: Pipeline,
        update_col_log: Pipeline,
        update_row_log: Pipeline,
        update_map: Pipeline,
        score_select: Pipeline,
        finalize: Pipeline,
        append_kernels: std::cell::OnceCell<super::device::DeviceAppendKernels>,
        seal_scratch: std::cell::OnceCell<SealScratch>,
    }

    impl KvarnEncoder {
        pub fn new(metal: &Metal) -> Result<Self> {
            let std_col = metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_std_col")?;
            let std_row = metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_std_row")?;
            let init = metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_init")?;
            let score_init = metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_score_init")?;
            let update_col_log = metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_update_col_log")?;
            let update_row_log = metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_update_row_log")?;
            let update_map = metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_update_map")?;
            let score_select = metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_score_select")?;
            let finalize = metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_finalize")?;
            ensure!(
                std_col.max_threads_per_threadgroup() >= 256
                    && std_row.max_threads_per_threadgroup() >= 32
                    && init.max_threads_per_threadgroup() >= 256
                    && update_col_log.max_threads_per_threadgroup() >= 128
                    && update_row_log.max_threads_per_threadgroup() >= 128
                    && update_map.max_threads_per_threadgroup() >= 256
                    && finalize.max_threads_per_threadgroup() >= 256,
                "Metal cannot dispatch KVarN reduction or map geometry"
            );
            Ok(Self {
                metal: metal.clone(),
                std_col,
                std_row,
                init,
                score_init,
                update_col_log,
                update_row_log,
                update_map,
                score_select,
                finalize,
                append_kernels: std::cell::OnceCell::new(),
                seal_scratch: std::cell::OnceCell::new(),
            })
        }

        pub(super) fn append_kernels(&self) -> Result<&super::device::DeviceAppendKernels> {
            if self.append_kernels.get().is_none() {
                let _ = self
                    .append_kernels
                    .set(super::device::DeviceAppendKernels::new(&self.metal)?);
            }
            Ok(self
                .append_kernels
                .get()
                .expect("KVarN append kernels initialized"))
        }

        /// Prepare both transforms, then pack each orientation before its
        /// normalization storage is reused. The caller orders every reader.
        #[allow(clippy::too_many_arguments)]
        pub(super) fn encode_seal(
            &self,
            batch: &mut Batch,
            tail_keys: &Buffer,
            tail_values: &Buffer,
            records: &Buffer,
            tile_start: usize,
            record_offset: usize,
            status: &Buffer,
        ) -> Result<()> {
            if self.seal_scratch.get().is_none() {
                let _ = self.seal_scratch.set(SealScratch::new(&self.metal)?);
            }
            let scratch = self
                .seal_scratch
                .get()
                .expect("KVarN seal scratch initialized");
            let kernels = self.append_kernels()?;
            let tile_start = u32::try_from(tile_start)?.to_le_bytes();
            let record_offset = u32::try_from(record_offset)?.to_le_bytes();
            // SAFETY: Each transform writes disjoint entries, and each pack
            // consumes normalization outputs before the next orientation.
            unsafe {
                launch(
                    batch,
                    &kernels.prepare,
                    &[tail_keys, tail_values, &scratch.keys, &scratch.values],
                    &tile_start,
                    KV_HEADS * TILE_TOKENS * HEAD_DIM,
                    HEAD_DIM,
                )?;
            }
            self.encode_normalize_into(
                batch,
                &scratch.keys,
                HEAD_DIM,
                TILE_TOKENS,
                &scratch.normalize,
            )?;
            unsafe {
                launch(
                    batch,
                    &kernels.pack_key,
                    &[
                        &scratch.normalize.balanced,
                        &scratch.normalize.best_col,
                        &scratch.normalize.best_row,
                        records,
                        status,
                    ],
                    &record_offset,
                    KV_HEADS * HEAD_DIM / 2,
                    128,
                )?;
            }
            self.encode_normalize_into(
                batch,
                &scratch.values,
                TILE_TOKENS,
                HEAD_DIM,
                &scratch.normalize,
            )?;
            unsafe {
                launch(
                    batch,
                    &kernels.pack_value,
                    &[
                        &scratch.normalize.balanced,
                        &scratch.normalize.best_col,
                        &scratch.normalize.best_row,
                        records,
                        status,
                    ],
                    &record_offset,
                    KV_HEADS * TILE_TOKENS,
                    128,
                )?;
            }
            Ok(())
        }

        /// Encode one BF16 token-major tile with Metal's exact normalization.
        pub fn encode_tile(&self, keys: &[u8], values: &[u8]) -> Result<Vec<u8>> {
            let normalize = |matrix: &[f32], rows: usize, cols: usize| {
                let mut paired = Vec::with_capacity(matrix.len() * 2);
                paired.extend_from_slice(matrix);
                paired.extend_from_slice(matrix);
                let (balanced, columns, rows_out) = self.normalize_graph(&paired, rows, cols)?;
                Ok((
                    balanced[..matrix.len()].to_vec(),
                    columns[..cols].to_vec(),
                    rows_out[..rows].to_vec(),
                ))
            };
            super::encode_tile_inner(keys, values, &normalize)
        }

        fn normalize_graph(
            &self,
            source: &[f32],
            rows: usize,
            cols: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
            ensure!(
                (rows, cols) == (256, 128) || (rows, cols) == (128, 256),
                "KVarN normalization geometry differs"
            );
            ensure!(
                source.len() == 2 * rows * cols && source.iter().all(|v| v.is_finite()),
                "KVarN normalization source differs"
            );
            let source_bytes: Vec<u8> = source.iter().flat_map(|v| v.to_le_bytes()).collect();
            let input = self.metal.shared_buffer(source_bytes.len())?;
            input.write_bytes(0, &source_bytes)?;
            let mut batch = self.metal.batch()?;
            let (balanced, best_col, best_row) =
                self.encode_normalize_graph(&mut batch, &input, rows, cols)?;
            batch.submit().wait()?;
            let read = |buffer: &Buffer| -> Result<Vec<f32>> {
                let mut bytes = vec![0; buffer.byte_len()];
                buffer.read_bytes(0, &mut bytes)?;
                Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect())
            };
            Ok((read(&balanced)?, read(&best_col)?, read(&best_row)?))
        }

        /// Encode variance normalization into an ordered device batch.
        /// The returned buffers stay live through the batch's retained bindings.
        pub(crate) fn encode_normalize_graph(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            rows: usize,
            cols: usize,
        ) -> Result<(Buffer, Buffer, Buffer)> {
            let scratch = NormalizeScratch::new(&self.metal, rows, cols)?;
            self.encode_normalize_into(batch, input, rows, cols, &scratch)?;
            Ok((scratch.balanced, scratch.best_col, scratch.best_row))
        }

        fn encode_normalize_into(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            rows: usize,
            cols: usize,
            scratch: &NormalizeScratch,
        ) -> Result<()> {
            ensure!(
                (rows, cols) == (256, 128) || (rows, cols) == (128, 256),
                "KVarN normalization geometry differs"
            );
            ensure!(
                input.byte_len() >= KV_HEADS * rows * cols * 4
                    && scratch.col_std.byte_len() >= KV_HEADS * cols * 4
                    && scratch.row_std.byte_len() >= KV_HEADS * rows * 4,
                "KVarN normalization device source or workspace differs"
            );
            let dims = [
                u32::try_from(rows)?.to_le_bytes(),
                u32::try_from(cols)?.to_le_bytes(),
            ]
            .concat();
            let s = scratch;
            let matrix_grid = KV_HEADS * rows * cols;
            let col_log_grid = KV_HEADS * cols;
            let row_log_grid = KV_HEADS * rows;
            let col_grid = KV_HEADS * (cols / 32) * 256;
            let row_grid = KV_HEADS * (rows / 4) * 32;
            // SAFETY: Fixed geometry preserves the eight-iteration
            // graph. Each output has exclusive per-head ownership, and scratch
            // is reused only after its preceding consumers on this stream.
            unsafe {
                launch(
                    batch,
                    &self.init,
                    &[
                        input,
                        &s.current,
                        &s.log_col,
                        &s.log_row,
                        &s.best_col,
                        &s.best_row,
                    ],
                    &dims,
                    matrix_grid,
                    256,
                )?;
                launch(
                    batch,
                    &self.std_col,
                    &[&s.current, &s.col_std],
                    &dims,
                    col_grid,
                    256,
                )?;
                launch(
                    batch,
                    &self.std_row,
                    &[&s.current, &s.row_std],
                    &dims,
                    row_grid,
                    32,
                )?;
                launch(
                    batch,
                    &self.score_init,
                    &[&s.col_std, &s.row_std, &s.best_score],
                    &dims,
                    KV_HEADS,
                    1,
                )?;
                for _ in 0..8 {
                    launch(
                        batch,
                        &self.std_col,
                        &[&s.current, &s.col_std],
                        &dims,
                        col_grid,
                        256,
                    )?;
                    launch(
                        batch,
                        &self.update_col_log,
                        &[&s.col_std, &s.log_col],
                        &dims,
                        col_log_grid,
                        128,
                    )?;
                    launch(
                        batch,
                        &self.update_map,
                        &[input, &s.current, &s.log_col, &s.log_row],
                        &dims,
                        matrix_grid,
                        256,
                    )?;
                    launch(
                        batch,
                        &self.std_row,
                        &[&s.current, &s.row_std],
                        &dims,
                        row_grid,
                        32,
                    )?;
                    launch(
                        batch,
                        &self.update_row_log,
                        &[&s.row_std, &s.log_row],
                        &dims,
                        row_log_grid,
                        128,
                    )?;
                    launch(
                        batch,
                        &self.update_map,
                        &[input, &s.current, &s.log_col, &s.log_row],
                        &dims,
                        matrix_grid,
                        256,
                    )?;
                    launch(
                        batch,
                        &self.std_col,
                        &[&s.current, &s.col_std],
                        &dims,
                        col_grid,
                        256,
                    )?;
                    launch(
                        batch,
                        &self.std_row,
                        &[&s.current, &s.row_std],
                        &dims,
                        row_grid,
                        32,
                    )?;
                    launch(
                        batch,
                        &self.score_select,
                        &[
                            &s.col_std,
                            &s.row_std,
                            &s.log_col,
                            &s.log_row,
                            &s.best_col,
                            &s.best_row,
                            &s.best_score,
                        ],
                        &dims,
                        KV_HEADS,
                        1,
                    )?;
                }
                launch(
                    batch,
                    &self.finalize,
                    &[input, &s.best_col, &s.best_row, &s.balanced],
                    &dims,
                    matrix_grid,
                    256,
                )?;
            }
            Ok(())
        }
    }

    pub struct KvarnDecoder {
        metal: Metal,
        pipeline: Pipeline,
    }

    impl KvarnDecoder {
        pub fn new(metal: &Metal) -> Result<Self> {
            let pipeline = metal.compile_msl_strict(SOURCE, "kvarn_gather")?;
            ensure!(
                pipeline.max_threads_per_threadgroup() >= 256,
                "Metal cannot dispatch KVarN selected-row geometry"
            );
            Ok(Self {
                metal: metal.clone(),
                pipeline,
            })
        }

        fn upload(&self, bytes: &[u8]) -> Result<Buffer> {
            let buffer = self.metal.shared_buffer(bytes.len().max(1))?;
            if !bytes.is_empty() {
                buffer.write_bytes(0, bytes)?;
            }
            Ok(buffer)
        }

        fn gather(
            &self,
            state: &KvarnState,
            logical_ids: Vec<i32>,
            valid: Vec<bool>,
        ) -> Result<KvarnSelectedRows> {
            ensure!(
                logical_ids.len() == valid.len(),
                "KVarN selection validity differs"
            );
            if logical_ids.is_empty() {
                return Ok(KvarnSelectedRows {
                    keys_bf16: Vec::new(),
                    values_bf16: Vec::new(),
                    logical_ids,
                    valid,
                });
            }
            let row_count =
                u32::try_from(logical_ids.len()).context("KVarN selection is too wide")?;
            let frontier =
                u32::try_from(state.frontier).context("KVarN frontier exceeds Metal index")?;
            let records = u32::try_from(state.record_count())
                .context("KVarN record count exceeds Metal index")?;
            ensure!(
                state.records.len() == state.record_count() * TILE_RECORD_BYTES,
                "KVarN records are not complete"
            );
            let bytes = super::selected_row_bytes(logical_ids.len())?;
            let packed = self.upload(&state.records)?;
            let sink_keys = self.upload(&state.sink_keys)?;
            let sink_values = self.upload(&state.sink_values)?;
            // At sink-only frontiers every valid ID is below 128. The gather
            // kernel returns for invalid rows before reading either tail.
            let (tail_keys, tail_values) = if state.frontier <= SINK_TOKENS {
                let unused_tail = self.metal.shared_buffer(2)?;
                (unused_tail.clone(), unused_tail)
            } else {
                (
                    self.upload(&state.tail_keys)?,
                    self.upload(&state.tail_values)?,
                )
            };
            let ids = self.upload(
                &logical_ids
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let flags = self.upload(&valid.iter().map(|v| u8::from(*v)).collect::<Vec<_>>())?;
            let counters = self.upload(
                &[
                    frontier.to_le_bytes(),
                    records.to_le_bytes(),
                    row_count.to_le_bytes(),
                ]
                .concat(),
            )?;
            let keys = self.metal.shared_buffer(bytes)?;
            let values = self.metal.shared_buffer(bytes)?;
            let mut batch = self.metal.batch()?;
            // SAFETY: Fixed record/sink/tail sizes and validated counters
            // bound every branch. Each group writes one distinct row/head.
            unsafe {
                batch.dispatch_1d(
                    &self.pipeline,
                    &[
                        Binding {
                            buffer: &packed,
                            offset: 0,
                        },
                        Binding {
                            buffer: &sink_keys,
                            offset: 0,
                        },
                        Binding {
                            buffer: &sink_values,
                            offset: 0,
                        },
                        Binding {
                            buffer: &tail_keys,
                            offset: 0,
                        },
                        Binding {
                            buffer: &tail_values,
                            offset: 0,
                        },
                        Binding {
                            buffer: &ids,
                            offset: 0,
                        },
                        Binding {
                            buffer: &flags,
                            offset: 0,
                        },
                        Binding {
                            buffer: &counters,
                            offset: 0,
                        },
                        Binding {
                            buffer: &keys,
                            offset: 0,
                        },
                        Binding {
                            buffer: &values,
                            offset: 0,
                        },
                    ],
                    logical_ids.len() * KV_HEADS * 256,
                    256,
                )?;
            }
            batch.submit().wait()?;
            let mut keys_bf16 = vec![0; bytes];
            let mut values_bf16 = vec![0; bytes];
            keys.read_bytes(0, &mut keys_bf16)?;
            values.read_bytes(0, &mut values_bf16)?;
            Ok(KvarnSelectedRows {
                keys_bf16,
                values_bf16,
                logical_ids,
                valid,
            })
        }

        /// Preserve requested physical-row order and duplicate positions.
        pub fn gather_rows(
            &self,
            state: &KvarnState,
            logical_ids: &[i32],
        ) -> Result<KvarnSelectedRows> {
            let valid = logical_ids
                .iter()
                .map(|&id| id >= 0 && (id as usize) < state.frontier)
                .collect();
            self.gather(state, logical_ids.to_vec(), valid)
        }

        /// Gather committed rows plus one exact pending row without advancing
        /// the cache. A sealing append must follow attention so its new packed
        /// tile does not change the current query's older exact rows.
        pub fn gather_pending_row(
            &self,
            state: &KvarnState,
            logical_ids: &[i32],
            pending_keys: &[u8],
            pending_values: &[u8],
        ) -> Result<KvarnSelectedRows> {
            ensure!(
                pending_keys.len() == super::TOKEN_BYTES
                    && pending_values.len() == super::TOKEN_BYTES,
                "KVarN pending row must contain two BF16 heads"
            );
            ensure!(
                state.frontier < state.max_context,
                "KVarN context capacity is exhausted"
            );
            super::finite_bf16(pending_keys, "KVarN pending keys")?;
            super::finite_bf16(pending_values, "KVarN pending values")?;
            let mut selected = self.gather_rows(state, logical_ids)?;
            let head_bytes = super::HEAD_DIM * 2;
            for (lane, &id) in logical_ids.iter().enumerate() {
                if id < 0 || id as usize != state.frontier {
                    continue;
                }
                selected.valid[lane] = true;
                for head in 0..KV_HEADS {
                    let source = head * head_bytes;
                    let target = (head * logical_ids.len() + lane) * head_bytes;
                    selected.keys_bf16[target..target + head_bytes]
                        .copy_from_slice(&pending_keys[source..source + head_bytes]);
                    selected.values_bf16[target..target + head_bytes]
                        .copy_from_slice(&pending_values[source..source + head_bytes]);
                }
            }
            Ok(selected)
        }

        /// Apply QSA ascending-set normalization before gathering BF16 rows.
        pub fn gather_qsa_selection(
            &self,
            state: &KvarnState,
            logical_ids: &[i32],
        ) -> Result<KvarnSelectedRows> {
            let sentinel =
                i32::try_from(state.frontier).context("KVarN frontier exceeds QSA index")?;
            let mut normalized: Vec<_> = logical_ids
                .iter()
                .map(|&id| {
                    if id >= 0 && id < sentinel {
                        id
                    } else {
                        sentinel
                    }
                })
                .collect();
            normalized.sort_unstable();
            let valid = normalized
                .iter()
                .enumerate()
                .map(|(index, &id)| id < sentinel && (index == 0 || id != normalized[index - 1]))
                .collect();
            self.gather(state, normalized, valid)
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::KvarnDecoder;
#[cfg(target_os = "macos")]
pub use gpu::KvarnEncoder;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn balanced_map_preserves_head_row_and_column_indices() -> Result<()> {
        use crate::metal::{Buffer, Metal};

        fn words(buffer: &Buffer) -> Result<Vec<f32>> {
            let mut bytes = vec![0; buffer.byte_len()];
            buffer.read_bytes(0, &mut bytes)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect())
        }

        let metal = Metal::new()?;
        let encoder = KvarnEncoder::new(&metal)?;
        for (rows, cols) in [(256, 128), (128, 256)] {
            let elements = rows * cols;
            let source: Vec<f32> = (0..KV_HEADS * elements)
                .map(|index| {
                    let head = index / elements;
                    let row = index % elements / cols;
                    let col = index % cols;
                    ((row * 17 + col * 29 + row * col * 3 + head * 97) % 257) as f32 / 64.0 - 2.0
                })
                .collect();
            let input = metal.shared_buffer(source.len() * 4)?;
            input.write_bytes(
                0,
                &source
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let mut batch = metal.batch()?;
            let (balanced, columns, rows_out) =
                encoder.encode_normalize_graph(&mut batch, &input, rows, cols)?;
            batch.submit().wait()?;
            let balanced = words(&balanced)?;
            let columns = words(&columns)?;
            let rows_out = words(&rows_out)?;
            assert_eq!(balanced.len(), source.len());
            assert!(columns.iter().all(|v| v.is_finite() && *v > 0.0));
            assert!(rows_out.iter().all(|v| v.is_finite() && *v > 0.0));
            for (index, &value) in source.iter().enumerate() {
                let head = index / elements;
                let row = index % elements / cols;
                let col = index % cols;
                let expected = value / columns[head * cols + col] / rows_out[head * rows + row];
                let error = (balanced[index] - expected).abs();
                assert!(
                    error <= 1e-4 * expected.abs().max(1.0),
                    "balanced map differs at head={head}, row={row}, col={col}"
                );
            }
            assert_ne!(balanced[0], balanced[elements]);
        }
        Ok(())
    }

    #[test]
    fn borrowed_parts_validate_kvarn_geometry_and_finiteness() {
        let mut sink = vec![0u8; SINK_BYTES];
        let mut tail = vec![0u8; TAIL_BYTES];
        KvarnState::validate_parts(9000, 512, &[], &sink, &sink, &tail, &tail).unwrap();
        assert!(KvarnState::validate_parts(0, 0, &[], &[], &[], &[], &[]).is_err());
        assert!(
            KvarnState::validate_parts(MAX_CONTEXT_TOKENS + 1, 0, &[], &[], &[], &[], &[]).is_err()
        );
        assert!(KvarnState::validate_parts(9000, 9001, &[], &sink, &sink, &tail, &tail).is_err());
        assert!(
            KvarnState::validate_parts(
                9000,
                512,
                &[],
                &sink[..SINK_BYTES - 2],
                &sink,
                &tail,
                &tail
            )
            .is_err()
        );
        assert!(
            KvarnState::validate_parts(
                9000,
                512,
                &[],
                &sink,
                &sink,
                &tail[..TAIL_BYTES - 2],
                &tail
            )
            .is_err()
        );
        sink[..2].copy_from_slice(&bf16::NAN.to_bits().to_le_bytes());
        assert!(KvarnState::validate_parts(9000, 512, &[], &sink, &sink, &tail, &tail).is_err());
        sink[..2].fill(0);
        tail[TAIL_BYTES - 2..].copy_from_slice(&bf16::INFINITY.to_bits().to_le_bytes());
        assert!(KvarnState::validate_parts(9000, 512, &[], &sink, &sink, &tail, &tail).is_err());
        tail[TAIL_BYTES - 2..].fill(0);
        let mut record = vec![0u8; TILE_RECORD_BYTES];
        KvarnState::validate_parts(9000, 8448, &record, &sink, &sink, &tail, &tail).unwrap();
        record[K_SCALE..K_SCALE + 2].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
        assert!(
            KvarnState::validate_parts(9000, 8448, &record, &sink, &sink, &tail, &tail).is_err()
        );
        assert!(
            KvarnState::validate_parts(9000, 8447, &record, &sink, &sink, &tail, &tail).is_err()
        );
    }

    #[test]
    fn partition_crosses_only_complete_body_tiles() {
        assert_eq!(partition(0), (0, 0, 0));
        assert_eq!(partition(128), (128, 128, 128));
        assert_eq!(partition(8447), (128, 128, 128));
        assert_eq!(partition(8448), (128, 256, 256));
        assert_eq!(partition(8451), (128, 256, 256));
        assert_eq!(partition(8576), (128, 384, 384));
        let state = KvarnState::zero(9000).unwrap();
        assert_eq!(state.safe_chunk_tokens(9000).unwrap(), 8448);
        assert!(state.safe_chunk_tokens(0).is_err());
    }

    #[test]
    fn record_growth_requests_are_capped_and_logarithmic() -> Result<()> {
        for (context, expected_records) in [
            (1, 0),
            (128, 0),
            (8447, 0),
            (8448, 1),
            (MAX_CONTEXT_TOKENS, 1983),
        ] {
            let state = KvarnState::zero(context)?;
            let max_records = partition(context).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
            assert_eq!(max_records, expected_records);
            assert_eq!(state.records.len(), 0);
            assert_eq!(state.records.capacity(), 0);
            assert_eq!(record_growth_target(0, 0, max_records)?, None);
        }

        let max_records = 1983;
        let mut requested_capacity = 0;
        let mut copied_old_backing = 0;
        let mut targets = Vec::new();
        for required in 1..=max_records {
            if let Some(target) = record_growth_target(required, requested_capacity, max_records)? {
                copied_old_backing += requested_capacity;
                requested_capacity = target;
                targets.push(target / TILE_RECORD_BYTES);
            }
        }
        assert_eq!(targets, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1983]);
        assert_eq!(requested_capacity, max_records * TILE_RECORD_BYTES);
        // This bounds the requested schedule if each growth moves its old
        // backing. Allocator-rounded capacity may differ from each request.
        assert_eq!(copied_old_backing, 143_514_624);
        assert!(copied_old_backing < 2 * requested_capacity);
        Ok(())
    }

    #[test]
    fn record_growth_handles_restored_and_pathological_capacities() -> Result<()> {
        assert_eq!(record_growth_target(1, 0, 1)?, Some(TILE_RECORD_BYTES));
        assert_eq!(
            record_growth_target(78, 77 * TILE_RECORD_BYTES + 1, 200)?,
            Some(154 * TILE_RECORD_BYTES)
        );
        assert_eq!(
            record_growth_target(3, 2 * TILE_RECORD_BYTES, 3)?,
            Some(3 * TILE_RECORD_BYTES)
        );
        assert_eq!(record_growth_target(3, 3 * TILE_RECORD_BYTES + 1, 3)?, None);
        assert!(record_growth_target(4, 0, 3).is_err());
        assert!(record_growth_target(usize::MAX, 0, usize::MAX).is_err());

        let mut state = KvarnState::zero(8576)?;
        let target = record_growth_target(1, state.records.capacity(), 2)?.unwrap();
        state.records.try_reserve_exact(target)?;
        let reserved_capacity = state.records.capacity();
        assert!(reserved_capacity >= 2 * TILE_RECORD_BYTES);
        let backing = state.records.as_ptr();
        let tile = vec![0; TILE_RECORD_BYTES];
        state.records.extend_from_slice(&tile);
        state.records.extend_from_slice(&tile);
        assert_eq!(state.records.as_ptr(), backing);
        assert_eq!(state.records.capacity(), reserved_capacity);
        assert_eq!(record_growth_target(2, reserved_capacity, 2)?, None);
        Ok(())
    }

    #[test]
    fn restore_rejects_partial_or_nonfinite_records_and_bad_frontiers() {
        let sink = vec![0; SINK_BYTES];
        let tail = vec![0; TAIL_BYTES];
        assert!(KvarnState::zero(0).is_err());
        assert!(KvarnState::restore(9000, 9001, &[], &sink, &sink, &tail, &tail).is_err());
        assert!(KvarnState::restore(9000, 8448, &[], &sink, &sink, &tail, &tail).is_err());
        let mut record = vec![0; TILE_RECORD_BYTES];
        record[K_SCALE..K_SCALE + 2].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(KvarnState::restore(9000, 8448, &record, &sink, &sink, &tail, &tail).is_err());
        record[K_SCALE..K_SCALE + 2].copy_from_slice(&0u16.to_le_bytes());
        let restored =
            KvarnState::restore(9000, 8448, &record, &sink, &sink, &tail, &tail).unwrap();
        assert!(restored.records.capacity() >= record.len());
        assert_eq!(
            (
                restored.frontier(),
                restored.body_frontier(),
                restored.record_count()
            ),
            (8448, 256, 1)
        );
    }

    #[test]
    fn context_capacity_accepts_short_sink_only_sessions_and_caps_packed_addresses() {
        for capacity in [1, 2, 3, MAX_CONTEXT_TOKENS] {
            let state = KvarnState::zero(capacity).unwrap();
            assert_eq!(state.frontier(), 0);
            assert_eq!(state.record_count(), 0);
        }
        for capacity in [0, MAX_CONTEXT_TOKENS + 1, i32::MAX as usize, usize::MAX] {
            assert!(
                KvarnState::zero(capacity).is_err(),
                "accepted capacity {capacity}"
            );
        }
        let record_count = (partition(MAX_CONTEXT_TOKENS).1 - SINK_TOKENS) / TILE_TOKENS;
        assert!(record_count * TILE_RECORD_BYTES <= u32::MAX as usize);
    }

    #[test]
    fn selected_row_extent_rejects_uint_wrap_without_allocating_buffers() {
        assert_eq!(selected_row_bytes(0).unwrap(), 0);
        assert_eq!(selected_row_bytes(2051).unwrap(), 2051 * TOKEN_BYTES);
        assert_eq!(
            selected_row_bytes(MAX_SELECTED_ROWS).unwrap(),
            MAX_SELECTED_ROWS * TOKEN_BYTES
        );
        assert!(selected_row_bytes(MAX_SELECTED_ROWS + 1).is_err());
        assert!(selected_row_bytes(usize::MAX).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gather_sink_tail_boundary_preserves_rows_and_invalid_mask() -> Result<()> {
        use crate::metal::Metal;

        fn word(pos: usize, head: usize, channel: usize, value: bool) -> [u8; 2] {
            let offset = usize::from(value) * 113;
            bf16::from_f32(((pos * 5 + head * 7 + channel + offset) % 509) as f32 / 16.0)
                .to_bits()
                .to_le_bytes()
        }

        let metal = Metal::new()?;
        let decoder = KvarnDecoder::new(&metal)?;
        let mut state = KvarnState::zero(129)?;
        for pos in 0..129 {
            for head in 0..KV_HEADS {
                for channel in 0..HEAD_DIM {
                    let offset = if pos < SINK_TOKENS {
                        (head * SINK_TOKENS + pos) * HEAD_DIM * 2 + channel * 2
                    } else {
                        (head * TAIL_CAPACITY + pos - SINK_TOKENS) * HEAD_DIM * 2 + channel * 2
                    };
                    let (keys, values) = if pos < SINK_TOKENS {
                        (&mut state.sink_keys, &mut state.sink_values)
                    } else {
                        (&mut state.tail_keys, &mut state.tail_values)
                    };
                    keys[offset..offset + 2].copy_from_slice(&word(pos, head, channel, false));
                    values[offset..offset + 2].copy_from_slice(&word(pos, head, channel, true));
                }
            }
        }
        for frontier in [1, 128, 129] {
            state.frontier = frontier;
            let ids = [
                0,
                frontier as i32 - 1,
                frontier as i32 - 1,
                -1,
                frontier as i32,
            ];
            let gathered = decoder.gather_rows(&state, &ids)?;
            assert_eq!(gathered.valid, [true, true, true, false, false]);
            let mut expected_keys = vec![0; ids.len() * TOKEN_BYTES];
            let mut expected_values = vec![0; ids.len() * TOKEN_BYTES];
            for head in 0..KV_HEADS {
                for (row, &id) in ids.iter().enumerate() {
                    if id < 0 || id as usize >= frontier {
                        continue;
                    }
                    for channel in 0..HEAD_DIM {
                        let offset = (head * ids.len() + row) * HEAD_DIM * 2 + channel * 2;
                        expected_keys[offset..offset + 2].copy_from_slice(&word(
                            id as usize,
                            head,
                            channel,
                            false,
                        ));
                        expected_values[offset..offset + 2].copy_from_slice(&word(
                            id as usize,
                            head,
                            channel,
                            true,
                        ));
                    }
                }
            }
            assert_eq!(
                gathered.keys_bf16, expected_keys,
                "K at frontier {frontier}"
            );
            assert_eq!(
                gathered.values_bf16, expected_values,
                "V at frontier {frontier}"
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn pending_decode_row_keeps_preseal_rows_exact_until_attention_finishes() -> Result<()> {
        use crate::metal::Metal;

        fn word(pos: usize, head: usize, channel: usize, value: bool) -> [u8; 2] {
            let multiplier = if value { 18 } else { 17 };
            let integer = ((pos * multiplier + head * 31 + channel * 13) % 257) as i32 - 128;
            bf16::from_f32(integer as f32 / 128.0)
                .to_bits()
                .to_le_bytes()
        }

        let metal = Metal::new()?;
        let decoder = KvarnDecoder::new(&metal)?;
        let encoder = KvarnEncoder::new(&metal)?;
        for frontier in [0, 127, 8447] {
            let mut state = KvarnState::zero(frontier + 1)?;
            for pos in 0..frontier {
                for head in 0..KV_HEADS {
                    let (keys, values, offset) = if pos < SINK_TOKENS {
                        (
                            &mut state.sink_keys,
                            &mut state.sink_values,
                            (head * SINK_TOKENS + pos) * HEAD_DIM * 2,
                        )
                    } else {
                        (
                            &mut state.tail_keys,
                            &mut state.tail_values,
                            (head * TAIL_CAPACITY + pos - SINK_TOKENS) * HEAD_DIM * 2,
                        )
                    };
                    for channel in 0..HEAD_DIM {
                        let offset = offset + channel * 2;
                        keys[offset..offset + 2].copy_from_slice(&word(pos, head, channel, false));
                        values[offset..offset + 2].copy_from_slice(&word(pos, head, channel, true));
                    }
                }
            }
            state.frontier = frontier;
            let mut keys = vec![0; TOKEN_BYTES];
            let mut values = vec![0; TOKEN_BYTES];
            for head in 0..KV_HEADS {
                for channel in 0..HEAD_DIM {
                    let offset = (head * HEAD_DIM + channel) * 2;
                    keys[offset..offset + 2].copy_from_slice(&word(frontier, head, channel, false));
                    values[offset..offset + 2]
                        .copy_from_slice(&word(frontier, head, channel, true));
                }
            }
            let ids = [
                0,
                128,
                255,
                frontier as i32,
                frontier as i32,
                -1,
                frontier as i32 + 1,
            ];
            let gathered = decoder.gather_pending_row(&state, &ids, &keys, &values)?;
            for (lane, &id) in ids.iter().enumerate() {
                let valid = id >= 0 && id as usize <= frontier;
                assert_eq!(gathered.valid[lane], valid);
                for head in 0..KV_HEADS {
                    for channel in 0..HEAD_DIM {
                        let offset = ((head * ids.len() + lane) * HEAD_DIM + channel) * 2;
                        let expected_k = if valid {
                            word(id as usize, head, channel, false)
                        } else {
                            [0; 2]
                        };
                        let expected_v = if valid {
                            word(id as usize, head, channel, true)
                        } else {
                            [0; 2]
                        };
                        assert_eq!(gathered.keys_bf16[offset..offset + 2], expected_k);
                        assert_eq!(gathered.values_bf16[offset..offset + 2], expected_v);
                    }
                }
            }
            assert_eq!(state.frontier(), frontier);
            assert_eq!(state.record_count(), 0);
            assert!(
                decoder
                    .gather_pending_row(&state, &ids, &keys[..2], &values)
                    .is_err()
            );
            let mut nonfinite = keys.clone();
            nonfinite[..2].copy_from_slice(&bf16::NAN.to_bits().to_le_bytes());
            assert!(
                decoder
                    .gather_pending_row(&state, &ids, &nonfinite, &values)
                    .is_err()
            );
            state.append(&encoder, &keys, &values)?;
            let after = decoder.gather_rows(&state, &ids)?;
            if frontier == 8447 {
                assert_eq!(state.record_count(), 1);
                assert_ne!(after.keys_bf16, gathered.keys_bf16);
                assert_ne!(after.values_bf16, gathered.values_bf16);
            } else {
                assert_eq!(after.keys_bf16, gathered.keys_bf16);
                assert_eq!(after.values_bf16, gathered.values_bf16);
            }
            assert!(
                decoder
                    .gather_pending_row(&state, &ids, &keys, &values)
                    .is_err()
            );
        }
        Ok(())
    }
}

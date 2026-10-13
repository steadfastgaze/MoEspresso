//! Selected-row PLE lookup and one-token Qwen PLE state.
//!
//! The table is kept on disk. Each decode step reads only its sixteen selected
//! BF16 rows and returns a candidate state for the caller to commit.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::Deserialize;
use serde_json::Value;

use crate::package::{Package, WeightFormat};

const EOS: u32 = 248_044;
const VOCAB: u32 = 248_320;
const HIDDEN: usize = 2_560;
const BRANCHES: usize = 4;
const EXPANDED: usize = HIDDEN * BRANCHES;
const HEADS: usize = 16;
const ROW_WIDTH: usize = HIDDEN / HEADS;
const CONV_TAPS: usize = 4;
const DILATION: usize = 3;
const HISTORY: usize = (CONV_TAPS - 1) * DILATION;
const MIX_GAMMA: u64 = 0x9E3779B97F4A7C15;

fn config_integer(config: &Value, field: &str) -> Result<u64> {
    config
        .get(field)
        .and_then(Value::as_u64)
        .with_context(|| format!("invalid PLE architecture {field}"))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(MIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

fn prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

fn validate_architecture(package: &Package, component: &PleComponent) -> Result<()> {
    let config = package
        .architecture()
        .get("config")
        .context("missing PLE model config")?;
    let base = config_integer(config, "ngram_vocab_size_base")?;
    ensure!(base == 20_000_000, "unsupported PLE vocabulary base");
    let mut size = base - 1;
    let mut logical_rows = 0u64;
    for (&actual_size, &actual_offset) in component.table_sizes.iter().zip(&component.table_offsets)
    {
        loop {
            size = size.checked_add(1).context("PLE prime search overflow")?;
            if prime(size) {
                break;
            }
        }
        ensure!(
            actual_size == size && actual_offset == logical_rows,
            "PLE table geometry differs from architecture"
        );
        logical_rows = logical_rows
            .checked_add(size)
            .context("PLE row count overflow")?;
    }
    let divisor = config_integer(config, "make_ngram_vocab_size_divisible_by")?;
    ensure!(divisor > 0, "invalid PLE row divisor");
    let padded = logical_rows
        .div_ceil(divisor)
        .checked_mul(divisor)
        .context("PLE padding overflow")?;
    let count = config_integer(config, "split_ngram_parts")?;
    ensure!(
        count > 0 && padded.is_multiple_of(count),
        "PLE shard count does not divide padding"
    );
    ensure!(
        component.logical_rows == logical_rows
            && component.padded_rows == padded
            && component.rows_per_shard == padded / count
            && component.shards.len() as u64 == count,
        "PLE shard geometry differs from architecture"
    );
    let seed = config_integer(config, "seed")?;
    let vocab = config_integer(config, "vocab_size")?;
    ensure!(vocab > 0, "invalid PLE vocabulary size");
    let half_bound = ((i64::MAX as u64) / vocab / 2).max(1);
    for (index, &actual) in component.multipliers.iter().enumerate() {
        let value = seed.wrapping_add(MIX_GAMMA.wrapping_mul(index as u64 + 1));
        let expected = 2 * (splitmix64(value) % half_bound) + 1;
        ensure!(
            actual == expected,
            "PLE hash multiplier differs from architecture"
        );
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShardRecord {
    index: usize,
    path: String,
    row_start: u64,
    row_count: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PleComponent {
    schema: String,
    layer_index: usize,
    dtype: String,
    row_width: usize,
    row_bytes: usize,
    logical_rows: u64,
    padded_rows: u64,
    rows_per_shard: u64,
    ngram_size: usize,
    heads_per_ngram: usize,
    multipliers: Vec<u64>,
    table_sizes: Vec<u64>,
    table_offsets: Vec<u64>,
    shards: Vec<ShardRecord>,
}

struct Shard {
    file: File,
}

/// Manifest-bound direct-row reader. The 128 files are opened, never mapped or expanded.
pub struct PleRows {
    component: PleComponent,
    shards: Vec<Shard>,
}

impl PleRows {
    pub fn new(package: &Package) -> Result<Self> {
        let value = package
            .ple_provider()
            .context("package has no PLE provider")?;
        let component: PleComponent = serde_json::from_value(value.clone())?;
        ensure!(
            component.schema == "qwen4_ple_provider_v1",
            "unsupported PLE schema"
        );
        ensure!(
            component.layer_index == 1,
            "PLE belongs to zero-based layer 1"
        );
        ensure!(component.dtype == "BF16", "PLE rows must be BF16");
        ensure!(
            component.ngram_size == 3 && component.heads_per_ngram == 8,
            "unsupported PLE n-gram geometry"
        );
        ensure!(
            component.row_width == ROW_WIDTH && component.row_bytes == 2 * ROW_WIDTH,
            "unsupported PLE row geometry"
        );
        ensure!(
            component.multipliers.len() == 3
                && component.table_sizes.len() == HEADS
                && component.table_offsets.len() == HEADS,
            "invalid PLE hash metadata lengths"
        );
        validate_architecture(package, &component)?;
        let mut next = 0u64;
        for (&size, &offset) in component.table_sizes.iter().zip(&component.table_offsets) {
            ensure!(
                size > 0 && offset == next,
                "PLE table offsets are not contiguous"
            );
            next = next
                .checked_add(size)
                .context("PLE table length overflow")?;
        }
        ensure!(
            next == component.logical_rows,
            "PLE logical row count differs from tables"
        );
        ensure!(
            component.rows_per_shard > 0
                && !component.shards.is_empty()
                && component
                    .rows_per_shard
                    .checked_mul(component.shards.len() as u64)
                    == Some(component.padded_rows)
                && component.logical_rows <= component.padded_rows
                && component.padded_rows - component.logical_rows < component.rows_per_shard,
            "invalid PLE shard or padding geometry"
        );
        let expected_bytes = component
            .rows_per_shard
            .checked_mul(component.row_bytes as u64)
            .context("PLE shard byte size overflow")?;
        let mut shards = Vec::with_capacity(component.shards.len());
        let mut seen_paths = BTreeSet::new();
        for (index, record) in component.shards.iter().enumerate() {
            ensure!(seen_paths.insert(&record.path), "duplicate PLE shard path");
            ensure!(
                record.index == index
                    && record.row_start == index as u64 * component.rows_per_shard
                    && record.row_count == component.rows_per_shard,
                "PLE shard range is not contiguous"
            );
            let (path, declared_bytes) = package.declared_file(&record.path)?;
            ensure!(
                declared_bytes == expected_bytes,
                "PLE shard size differs from rows"
            );
            let file =
                File::open(&path).with_context(|| format!("open PLE shard {}", path.display()))?;
            ensure!(
                file.metadata()?.len() == declared_bytes,
                "PLE shard file size changed"
            );
            shards.push(Shard { file });
        }
        Ok(Self { component, shards })
    }

    /// Return physical row IDs in bigram-head then trigram-head order.
    pub fn row_ids(&self, token: u32, context: [u32; 2]) -> Result<[u64; HEADS]> {
        ensure!(token < VOCAB, "token ID exceeds released Qwen vocabulary");
        ensure!(
            context.iter().all(|&v| v < VOCAB),
            "invalid PLE token history"
        );
        let previous = if context[1] == EOS { EOS } else { context[1] };
        let older = if context[1] == EOS || context[0] == EOS {
            EOS
        } else {
            context[0]
        };
        let terms = [token as u64, previous as u64, older as u64];
        let mut mixed = terms[0].wrapping_mul(self.component.multipliers[0]);
        let mut rows = [0u64; HEADS];
        for ngram in 0..2 {
            mixed ^= terms[ngram + 1].wrapping_mul(self.component.multipliers[ngram + 1]);
            for head in 0..8 {
                let index = ngram * 8 + head;
                rows[index] =
                    self.component.table_offsets[index] + mixed % self.component.table_sizes[index];
                ensure!(
                    rows[index] < self.component.logical_rows,
                    "PLE hash exceeded table"
                );
            }
        }
        Ok(rows)
    }

    /// Read exactly 16 selected rows, preserving head order and BF16 bits.
    pub fn lookup(&mut self, row_ids: &[u64; HEADS]) -> Result<Vec<u16>> {
        let mut bytes = vec![0u8; HEADS * self.component.row_bytes];
        for (head, &row) in row_ids.iter().enumerate() {
            ensure!(
                row < self.component.logical_rows,
                "PLE row exceeds logical range"
            );
            let shard_index = (row / self.component.rows_per_shard) as usize;
            let local_row = row % self.component.rows_per_shard;
            let offset = local_row
                .checked_mul(self.component.row_bytes as u64)
                .context("PLE row byte offset overflow")?;
            let file = &mut self.shards[shard_index].file;
            file.seek(SeekFrom::Start(offset))?;
            let begin = head * self.component.row_bytes;
            file.read_exact(&mut bytes[begin..begin + self.component.row_bytes])?;
        }
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect())
    }
}

/// State at a committed public token frontier.
#[derive(Clone)]
pub struct PleState {
    pub context: [u32; 2],
    /// Oldest to newest, nine normalized BF16 rows of expanded width.
    pub conv: Vec<u16>,
    pub offset: u64,
}

impl Default for PleState {
    fn default() -> Self {
        Self {
            context: [EOS; 2],
            conv: vec![0; HISTORY * EXPANDED],
            offset: 0,
        }
    }
}

impl PleState {
    pub(crate) fn try_zero() -> Result<Self> {
        let words = HISTORY
            .checked_mul(EXPANDED)
            .context("PLE history size overflow")?;
        let mut conv = Vec::new();
        conv.try_reserve_exact(words)
            .context("PLE history reservation failed")?;
        conv.resize(words, 0);
        Ok(Self {
            context: [EOS; 2],
            conv,
            offset: 0,
        })
    }

    #[cfg(test)]
    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            context: self.context,
            conv: try_copy_history(&self.conv)?,
            offset: self.offset,
        })
    }

    pub(crate) fn validate_committed(&self, consumed: &[u32], eos: u32) -> Result<()> {
        ensure!(
            self.offset == consumed.len() as u64,
            "PLE offset differs from consumed token count"
        );
        ensure!(
            self.conv.len() == HISTORY * EXPANDED,
            "invalid PLE convolution history"
        );
        ensure!(
            self.conv.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "PLE convolution history contains nonfinite values"
        );
        ensure!(
            consumed.iter().all(|&token| token < VOCAB),
            "consumed token ID exceeds released Qwen vocabulary"
        );
        let expected = match consumed {
            [] => [eos, eos],
            [last] => [eos, *last],
            _ => [consumed[consumed.len() - 2], consumed[consumed.len() - 1]],
        };
        ensure!(
            self.context == expected,
            "PLE token history differs from consumed tokens"
        );
        Ok(())
    }
}

/// Canonical PLE convolution history for device execution. Host conversion is
/// reserved for quiescent snapshot, restore and external inspection boundaries.
#[cfg(target_os = "macos")]
pub struct DevicePleState {
    pub(crate) context: [u32; 2],
    conv: crate::metal::Buffer,
    pub(crate) offset: u64,
}

#[cfg(target_os = "macos")]
impl DevicePleState {
    /// Import already validated, quiescent convolution storage.
    pub(crate) fn from_buffer(
        conv: crate::metal::Buffer,
        consumed: &[u32],
        eos: u32,
    ) -> Result<Self> {
        let context = match consumed {
            [] => [eos, eos],
            [last] => [eos, *last],
            _ => [consumed[consumed.len() - 2], consumed[consumed.len() - 1]],
        };
        let state = Self {
            conv,
            context,
            offset: consumed.len() as u64,
        };
        state.conv.read_bytes(0, &mut [])?;
        state.validate_committed(consumed, eos)?;
        Ok(state)
    }

    pub fn zero(metal: &crate::metal::Metal) -> Result<Self> {
        Self::from_host(metal, &PleState::try_zero()?)
    }

    pub fn from_host(metal: &crate::metal::Metal, host: &PleState) -> Result<Self> {
        ensure!(
            host.context.iter().all(|&token| token < VOCAB)
                && host.conv.len() == HISTORY * EXPANDED
                && host
                    .conv
                    .iter()
                    .all(|&word| bf16::from_bits(word).is_finite()),
            "invalid PLE state for device import"
        );
        let conv = metal.shared_buffer(HISTORY * EXPANDED * 2)?;
        conv.write_bytes(
            0,
            &host
                .conv
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            context: host.context,
            conv,
            offset: host.offset,
        })
    }

    pub fn to_host(&self) -> Result<PleState> {
        let mut conv = Vec::new();
        conv.try_reserve_exact(HISTORY * EXPANDED)
            .context("PLE host history reservation failed")?;
        let mut bytes = [0u8; 16 * 1024];
        for offset in (0..HISTORY * EXPANDED * 2).step_by(bytes.len()) {
            let length = bytes.len().min(HISTORY * EXPANDED * 2 - offset);
            self.conv.read_bytes(offset, &mut bytes[..length])?;
            conv.extend(
                bytes[..length]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| u16::from_le_bytes(*pair)),
            );
        }
        ensure!(
            conv.iter().all(|&word| bf16::from_bits(word).is_finite()),
            "PLE device history contains nonfinite values"
        );
        Ok(PleState {
            context: self.context,
            conv,
            offset: self.offset,
        })
    }

    pub fn conv_byte_len(&self) -> usize {
        self.conv.byte_len()
    }

    pub(crate) fn device_history(&self) -> &crate::metal::Buffer {
        &self.conv
    }

    pub(crate) fn advance_chunk_in_place(&mut self, tokens: &[u32]) -> Result<()> {
        ensure!(
            tokens.iter().all(|&token| token < VOCAB),
            "invalid PLE token history"
        );
        ensure!(
            self.conv.byte_len() == HISTORY * EXPANDED * 2,
            "invalid PLE device convolution history"
        );
        let next_offset = self
            .offset
            .checked_add(tokens.len() as u64)
            .context("PLE frontier overflow")?;
        for &token in tokens {
            self.context = [self.context[1], token];
        }
        self.offset = next_offset;
        Ok(())
    }

    fn advance_history(&mut self, alternate: &mut crate::metal::Buffer, token: u32) -> Result<()> {
        ensure!(token < VOCAB, "invalid PLE token history");
        ensure!(
            alternate.byte_len() == HISTORY * EXPANDED * 2,
            "invalid PLE alternate convolution history"
        );
        let next_offset = self
            .offset
            .checked_add(1)
            .context("PLE frontier overflow")?;
        std::mem::swap(&mut self.conv, alternate);
        self.context = [self.context[1], token];
        self.offset = next_offset;
        Ok(())
    }

    pub(crate) fn validate_committed(&self, consumed: &[u32], eos: u32) -> Result<()> {
        ensure!(
            self.offset == consumed.len() as u64,
            "PLE offset differs from consumed token count"
        );
        ensure!(
            self.conv.byte_len() == HISTORY * EXPANDED * 2,
            "invalid PLE device convolution history"
        );
        ensure!(
            consumed.iter().all(|&token| token < VOCAB),
            "consumed token ID exceeds released Qwen vocabulary"
        );
        let expected = match consumed {
            [] => [eos, eos],
            [last] => [eos, *last],
            _ => [consumed[consumed.len() - 2], consumed[consumed.len() - 1]],
        };
        ensure!(
            self.context == expected,
            "PLE token history differs from consumed tokens"
        );
        Ok(())
    }
}

fn try_copy_history(source: &[u16]) -> Result<Vec<u16>> {
    ensure!(
        source.len() == HISTORY * EXPANDED,
        "PLE history length differs"
    );
    let mut copy = Vec::new();
    copy.try_reserve_exact(source.len())
        .context("PLE history copy reservation failed")?;
    copy.extend_from_slice(source);
    Ok(copy)
}

fn try_extended_history(source: &[u16]) -> Result<Vec<u16>> {
    let mut history = try_copy_history(source)?;
    history
        .try_reserve_exact(EXPANDED)
        .context("PLE extended history reservation failed")?;
    Ok(history)
}

fn check_weight(
    package: &Package,
    name: &str,
    format: WeightFormat,
    shape: &[usize],
) -> Result<crate::package::DirectWeight> {
    let weight = package.direct_weight(name)?;
    ensure!(
        weight.format == format && weight.shape == shape,
        "invalid {name} format or shape"
    );
    Ok(weight)
}

fn bf16_words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

fn bf(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}
fn round(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}

fn sigmoid(x: f32) -> f32 {
    let e = bf(round(x.abs().exp()));
    let denominator = bf(round(1.0 + e));
    let y = bf(round(1.0 / denominator));
    if x < 0.0 { y } else { bf(round(1.0 - y)) }
}
fn silu(x: f32) -> f32 {
    bf(round(x * sigmoid(x)))
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use crate::device_status::DeviceStatus;
    use crate::kquant::{DenseMatvec, OutputPrecision};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline, Stream};
    use crate::model::{MixerKind, QwenConfig};

    const NORM_SOURCE: &str = include_str!("../kernels/metal/hc.metal");
    const PLE_SOURCE: &str = include_str!("../kernels/metal/ple.metal");
    const DEVICE_NORM_SOURCE: &str = include_str!("../kernels/metal/hc_batch.metal");
    const DEVICE_PLE_SOURCE: &str = include_str!("../kernels/metal/ple_batch.metal");

    fn upload_words(metal: &Metal, words: &[u16]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(words.len() * 2)?;
        buffer.write_bytes(
            0,
            &words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        Ok(buffer)
    }

    struct DevicePleScratch {
        embedding: Buffer,
        key_projected: Buffer,
        key: Buffer,
        value: Buffer,
        query: Buffer,
        dot: Buffer,
        gate: Buffer,
        gain: Buffer,
        flat: Buffer,
        normalized: Buffer,
        norm_updated: Buffer,
        output: Buffer,
        next_conv: Buffer,
    }

    impl DevicePleScratch {
        fn new(metal: &Metal) -> Result<Self> {
            let row = |width| metal.shared_buffer(width * 2);
            Ok(Self {
                embedding: row(HIDDEN)?,
                key_projected: row(EXPANDED)?,
                key: row(EXPANDED)?,
                value: row(HIDDEN)?,
                query: row(EXPANDED)?,
                dot: row(BRANCHES)?,
                gate: row(BRANCHES)?,
                gain: row(BRANCHES)?,
                flat: row(EXPANDED)?,
                normalized: row(EXPANDED)?,
                norm_updated: row(EXPANDED)?,
                output: row(EXPANDED)?,
                next_conv: row(HISTORY * EXPANDED)?,
            })
        }
    }

    /// One-token PLE execution using bounded SSD reads and native projections.
    pub struct Qwen4Ple {
        metal: Metal,
        pub rows: PleRows,
        key: DenseMatvec,
        value: DenseMatvec,
        norm_key: Vec<u16>,
        norm_query: Vec<u16>,
        norm_conv: Vec<u16>,
        conv: Vec<u16>,
        norm_kernel: Pipeline,
        gate_dot_kernel: Pipeline,
        device_norm_key: Buffer,
        device_norm_query: Buffer,
        device_norm_conv: Buffer,
        device_conv_weight: Buffer,
        device_epsilon: Buffer,
        device_flags: Buffer,
        device_dummy: Buffer,
        device_norm: Pipeline,
        device_dot: Pipeline,
        device_gate: Pipeline,
        device_flat: Pipeline,
        device_conv: Pipeline,
        device_scratch: DevicePleScratch,
        max_positions: usize,
    }

    pub struct PleStep {
        pub output_bf16: Vec<u16>,
        pub state: PleState,
        pub row_ids: [u64; HEADS],
        pub stages: Option<PleStages>,
    }

    /// Optional numerical stages for an external oracle comparison.
    pub struct PleStages {
        pub embedding: Vec<u16>,
        pub key: Vec<u16>,
        pub value: Vec<u16>,
        pub query: Vec<u16>,
        pub gate_dot: Vec<u16>,
        pub gate: Vec<u16>,
        pub flat: Vec<u16>,
        pub normalized: Vec<u16>,
        pub conv: Vec<u16>,
    }

    impl Qwen4Ple {
        pub(crate) fn shared_matrices(&self) -> [crate::kquant::DeviceMatrix; 2] {
            [&self.key, &self.value].map(DenseMatvec::shared_matrix)
        }

        fn encode_device_norm(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            weight: &Buffer,
            output: &Buffer,
        ) -> Result<()> {
            let bindings = [
                input,
                weight,
                &self.device_dummy,
                &self.device_dummy,
                output,
                &self.device_scratch.norm_updated,
                &self.device_epsilon,
                &self.device_flags,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: One 640-thread group covers each of four 2560-wide
            // streams. Pending injection is disabled for PLE normalization.
            unsafe { batch.dispatch_1d(&self.device_norm, &bindings, 640, 640) }
        }

        /// Encode one PLE contribution and residual update without exporting
        /// the convolution history. The caller publishes the session only
        /// after its ordered stream completes and device status is checked.
        pub(crate) fn encode_step_device(
            &mut self,
            stream: &mut Stream,
            state: &mut DevicePleState,
            token: u32,
            residual: &Buffer,
            status: &DeviceStatus,
        ) -> Result<Buffer> {
            ensure!(
                residual.byte_len() == EXPANDED * 2
                    && state.conv.byte_len() == HISTORY * EXPANDED * 2,
                "PLE device row or history geometry differs"
            );
            ensure!(
                state.offset < self.max_positions as u64,
                "PLE frontier exceeds model context limit"
            );
            let row_ids = self.rows.row_ids(token, state.context)?;
            let embedding = self.rows.lookup(&row_ids)?;
            ensure!(
                embedding.len() == HIDDEN
                    && embedding
                        .iter()
                        .all(|&word| bf16::from_bits(word).is_finite()),
                "PLE selected rows contain nonfinite BF16 values"
            );
            self.device_scratch.embedding.write_bytes(
                0,
                &embedding
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let scratch = &self.device_scratch;
            let batch = stream.batch()?;
            // SAFETY: The selected BF16 row was validated and staged before
            // encoding. Every device output has disjoint storage and the
            // ordered batch retains inputs, outputs and previous history.
            unsafe {
                self.key.encode(
                    batch,
                    &scratch.embedding,
                    &scratch.key_projected,
                    OutputPrecision::Bf16,
                )?;
                self.value.encode(
                    batch,
                    &scratch.embedding,
                    &scratch.value,
                    OutputPrecision::Bf16,
                )?;
            }
            self.encode_device_norm(
                batch,
                &scratch.key_projected,
                &self.device_norm_key,
                &scratch.key,
            )?;
            self.encode_device_norm(batch, residual, &self.device_norm_query, &scratch.query)?;
            let dot = [&scratch.key, &scratch.query, &scratch.dot]
                .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: Four groups reduce disjoint branches into four BF16 cells.
            unsafe {
                batch.dispatch_1d(&self.device_dot, &dot, BRANCHES * 640, 640)?;
            }
            let gate = [&scratch.dot, &scratch.gate, &scratch.gain]
                .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: The four gate cells and 10240 flattened cells have
            // direct bounds from the fixed PLE geometry.
            unsafe {
                batch.dispatch_1d(&self.device_gate, &gate, BRANCHES, 64)?;
                let flat = [&scratch.gain, &scratch.value, &scratch.flat]
                    .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.device_flat, &flat, EXPANDED, 256)?;
            }
            self.encode_device_norm(
                batch,
                &scratch.flat,
                &self.device_norm_conv,
                &scratch.normalized,
            )?;
            let conv = [
                &state.conv,
                &scratch.normalized,
                &self.device_conv_weight,
                &scratch.flat,
                residual,
                &scratch.output,
                &scratch.next_conv,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: Each channel advances its own nine-row history and
            // writes one disjoint residual cell for the next layer.
            unsafe {
                batch.dispatch_1d(&self.device_conv, &conv, EXPANDED, 128)?;
            }
            status.encode_bf16(batch, &scratch.output)?;
            status.encode_bf16(batch, &scratch.next_conv)?;
            let output = scratch.output.clone();
            state.advance_history(&mut self.device_scratch.next_conv, token)?;
            Ok(output)
        }

        pub fn new(metal: &Metal, package: &Package) -> Result<Self> {
            let config = QwenConfig::from_package(package)?;
            ensure!(
                config.ple_layer_index == 1 && config.layers[1] == MixerKind::Gdn,
                "PLE insertion must precede Qwen layer 1 GDN"
            );
            let rows = PleRows::new(package)?;
            let key = check_weight(
                package,
                "layers.1.ple.key_proj",
                WeightFormat::Q6K,
                &[EXPANDED, HIDDEN],
            )?;
            let value = check_weight(
                package,
                "layers.1.ple.value_proj",
                WeightFormat::Q6K,
                &[HIDDEN, HIDDEN],
            )?;
            let norm_key = check_weight(
                package,
                "layers.1.ple.norm_key",
                WeightFormat::Bf16,
                &[EXPANDED],
            )?;
            let norm_query = check_weight(
                package,
                "layers.1.ple.norm_query",
                WeightFormat::Bf16,
                &[EXPANDED],
            )?;
            let norm_conv = check_weight(
                package,
                "layers.1.ple.norm_conv",
                WeightFormat::Bf16,
                &[EXPANDED],
            )?;
            let conv = check_weight(
                package,
                "layers.1.ple.conv1d",
                WeightFormat::Bf16,
                &[EXPANDED, 1, CONV_TAPS],
            )?;
            let norm_kernel = metal.compile_msl_strict(NORM_SOURCE, "hc_norm")?;
            ensure!(
                norm_kernel.max_threads_per_threadgroup() >= 640,
                "PLE grouped norm requires 640 Metal threads"
            );
            let gate_dot_kernel = metal.compile_msl_strict(PLE_SOURCE, "ple_gate_dot")?;
            ensure!(
                gate_dot_kernel.max_threads_per_threadgroup() >= 640,
                "PLE gate reduction requires 640 Metal threads"
            );
            let device_source = format!("{DEVICE_NORM_SOURCE}\n{DEVICE_PLE_SOURCE}");
            let device_pipeline = |name: &str, threads: usize| -> Result<Pipeline> {
                let pipeline = metal.compile_msl_strict(&device_source, name)?;
                ensure!(
                    pipeline.max_threads_per_threadgroup() >= threads,
                    "PLE device threadgroup unsupported"
                );
                Ok(pipeline)
            };
            let device_epsilon = metal.shared_buffer(4)?;
            device_epsilon.write_bytes(0, &1e-6f32.to_le_bytes())?;
            let device_flags = metal.shared_buffer(8)?;
            device_flags.write_bytes(0, &[0; 8])?;
            Ok(Self {
                metal: metal.clone(),
                rows,
                key: DenseMatvec::new(metal, &key)?,
                value: DenseMatvec::new(metal, &value)?,
                device_norm_key: upload_words(metal, &bf16_words(&norm_key.bytes))?,
                device_norm_query: upload_words(metal, &bf16_words(&norm_query.bytes))?,
                device_norm_conv: upload_words(metal, &bf16_words(&norm_conv.bytes))?,
                device_conv_weight: upload_words(metal, &bf16_words(&conv.bytes))?,
                device_epsilon,
                device_flags,
                device_dummy: metal.shared_buffer(2)?,
                device_norm: device_pipeline("hc_batch_norm", 640)?,
                device_dot: device_pipeline("ple_batch_gate_dot", 640)?,
                device_gate: device_pipeline("ple_batch_gate_scalar", 64)?,
                device_flat: device_pipeline("ple_batch_flat", 256)?,
                device_conv: device_pipeline("ple_decode_conv_residual", 128)?,
                device_scratch: DevicePleScratch::new(metal)?,
                norm_key: bf16_words(&norm_key.bytes),
                norm_query: bf16_words(&norm_query.bytes),
                norm_conv: bf16_words(&norm_conv.bytes),
                conv: bf16_words(&conv.bytes),
                norm_kernel,
                gate_dot_kernel,
                max_positions: config.max_positions,
            })
        }

        fn dot(&self, key: &[u16], query: &[u16]) -> Result<Vec<u16>> {
            let key_buffer = self.metal.shared_buffer(EXPANDED * 2)?;
            let query_buffer = self.metal.shared_buffer(EXPANDED * 2)?;
            let output = self.metal.shared_buffer(BRANCHES * 2)?;
            key_buffer.write_bytes(
                0,
                &key.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )?;
            query_buffer.write_bytes(
                0,
                &query
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let bindings =
                [&key_buffer, &query_buffer, &output].map(|buffer| Binding { buffer, offset: 0 });
            let mut batch = self.metal.batch()?;
            // SAFETY: Four independent groups each reduce 2560 validated BF16
            // values to one disjoint output. All buffers outlive completion.
            unsafe {
                batch.dispatch_1d(&self.gate_dot_kernel, &bindings, 4 * 640, 640)?;
            }
            batch
                .submit()
                .wait()
                .context("PLE gate reduction command")?;
            let mut bytes = vec![0u8; BRANCHES * 2];
            output.read_bytes(0, &mut bytes)?;
            Ok(bf16_words(&bytes))
        }

        fn norm(&self, input: &[u16], weight: &[u16]) -> Result<Vec<u16>> {
            ensure!(
                input.len() == EXPANDED && weight.len() == EXPANDED,
                "PLE grouped norm width changed"
            );
            let source = self.metal.shared_buffer(EXPANDED * 2)?;
            let scales = self.metal.shared_buffer(EXPANDED * 2)?;
            let dummy = self.metal.shared_buffer(HIDDEN * 2)?;
            let injection = self.metal.shared_buffer(BRANCHES * 2)?;
            let output = self.metal.shared_buffer(EXPANDED * 2)?;
            let updated = self.metal.shared_buffer(EXPANDED * 2)?;
            let eps = self.metal.shared_buffer(4)?;
            let has_pending = self.metal.shared_buffer(4)?;
            source.write_bytes(
                0,
                &input
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            scales.write_bytes(
                0,
                &weight
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            eps.write_bytes(0, &1e-6f32.to_le_bytes())?;
            has_pending.write_bytes(0, &0i32.to_le_bytes())?;
            let bindings = [
                &source,
                &scales,
                &dummy,
                &injection,
                &output,
                &updated,
                &eps,
                &has_pending,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            let mut batch = self.metal.batch()?;
            // SAFETY: hc_norm reads four 2560-element streams and writes four
            // disjoint streams. Pending inputs are unused with has_pending=0.
            unsafe {
                batch.dispatch_1d(&self.norm_kernel, &bindings, BRANCHES * 640, 640)?;
            }
            batch.submit().wait().context("PLE grouped norm command")?;
            let mut bytes = vec![0u8; EXPANDED * 2];
            output.read_bytes(0, &mut bytes)?;
            Ok(bf16_words(&bytes))
        }

        /// Compute a candidate PLE frontier. The caller commits `step.state`.
        pub fn step(
            &mut self,
            state: &PleState,
            token: u32,
            hidden_bf16: &[u16],
        ) -> Result<PleStep> {
            self.execute(state, token, hidden_bf16, false)
        }

        pub fn step_stages(
            &mut self,
            state: &PleState,
            token: u32,
            hidden_bf16: &[u16],
        ) -> Result<PleStep> {
            self.execute(state, token, hidden_bf16, true)
        }

        fn execute(
            &mut self,
            state: &PleState,
            token: u32,
            hidden_bf16: &[u16],
            capture: bool,
        ) -> Result<PleStep> {
            ensure!(
                state.conv.len() == HISTORY * EXPANDED,
                "invalid PLE convolution history"
            );
            ensure!(
                state.offset < self.max_positions as u64,
                "PLE frontier exceeds model context limit"
            );
            ensure!(
                hidden_bf16.len() == EXPANDED,
                "PLE requires one expanded residual row"
            );
            ensure!(
                hidden_bf16.iter().all(|&v| bf16::from_bits(v).is_finite())
                    && state.conv.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "PLE input or convolution state is nonfinite"
            );
            let row_ids = self.rows.row_ids(token, state.context)?;
            let embedding = self.rows.lookup(&row_ids)?;
            let key = self.key.run_bf16(&self.metal, &embedding)?;
            let value = self.value.run_bf16(&self.metal, &embedding)?;
            let key = self.norm(&key, &self.norm_key)?;
            let query = self.norm(hidden_bf16, &self.norm_query)?;
            let mut flat = vec![0u16; EXPANDED];
            let gate_dot = self.dot(&key, &query)?;
            let mut gate_stage = vec![0u16; BRANCHES];
            for stream in 0..BRANCHES {
                let base = stream * HIDDEN;
                let gate = round(bf(gate_dot[stream]) / bf(round((HIDDEN as f32).sqrt())));
                let signed = if bf(gate) == 0.0 {
                    0.0
                } else {
                    bf(gate).signum() * bf(round(bf(gate).abs().max(1e-6).sqrt()))
                };
                gate_stage[stream] = round(signed);
                let gain = round(sigmoid(signed));
                for d in 0..HIDDEN {
                    flat[base + d] = round(bf(gain) * bf(value[d]));
                }
            }
            let normalized = self.norm(&flat, &self.norm_conv)?;
            let mut history = try_extended_history(&state.conv)?;
            history.extend_from_slice(&normalized);
            let mut output = vec![0u16; EXPANDED];
            let mut conv_stage = vec![0u16; EXPANDED];
            for d in 0..EXPANDED {
                let mut sum = 0.0f32;
                for tap in 0..CONV_TAPS {
                    let index = tap * DILATION * EXPANDED + d;
                    sum += bf(history[index]) * bf(self.conv[d * CONV_TAPS + tap]);
                }
                conv_stage[d] = round(silu(bf(round(sum))));
                output[d] = round(bf(flat[d]) + bf(conv_stage[d]));
            }
            let stages = capture.then_some(PleStages {
                embedding,
                key,
                value,
                query,
                gate_dot,
                gate: gate_stage,
                flat,
                normalized,
                conv: conv_stage,
            });
            let next_conv = try_copy_history(&history[EXPANDED..])?;
            Ok(PleStep {
                output_bf16: output,
                state: PleState {
                    context: [state.context[1], token],
                    conv: next_conv,
                    offset: state
                        .offset
                        .checked_add(1)
                        .context("PLE frontier overflow")?,
                },
                row_ids,
                stages,
            })
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{PleStep, Qwen4Ple};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallible_history_copy_and_extension_reserve_before_growth() {
        let state = PleState::try_zero().unwrap();
        assert_eq!(state.conv.len(), HISTORY * EXPANDED);
        assert!(state.conv.capacity() >= state.conv.len());
        assert!(state.conv.iter().all(|&word| word == 0));

        let mut extended = try_extended_history(&state.conv).unwrap();
        assert_eq!(extended.len(), HISTORY * EXPANDED);
        assert!(extended.capacity() >= (HISTORY + 1) * EXPANDED);
        let allocation = extended.as_ptr();
        extended.extend_from_slice(&vec![7; EXPANDED]);
        assert_eq!(extended.as_ptr(), allocation);
        assert_eq!(extended.len(), (HISTORY + 1) * EXPANDED);

        let copy = state.try_clone().unwrap();
        assert_eq!(copy.conv, state.conv);
        assert_ne!(copy.conv.as_ptr(), state.conv.as_ptr());
        assert!(try_copy_history(&state.conv[..state.conv.len() - 1]).is_err());
        let mut invalid = state;
        invalid.conv.pop();
        let original = invalid.conv.clone();
        assert!(invalid.try_clone().is_err());
        assert_eq!(invalid.conv, original);
    }

    #[test]
    fn empty_state_has_released_geometry() {
        let state = PleState::default();
        assert_eq!(state.context, [EOS, EOS]);
        assert_eq!(state.conv.len(), 9 * 10_240);
        assert_eq!(state.offset, 0);
        state.validate_committed(&[], EOS).unwrap();
    }

    #[test]
    fn committed_state_rejects_malformed_history_and_context() {
        let mut state = PleState {
            offset: 2,
            context: [7, 8],
            ..PleState::default()
        };
        state.validate_committed(&[7, 8], EOS).unwrap();
        state.context[0] = EOS;
        assert!(state.validate_committed(&[7, 8], EOS).is_err());
        state.context[0] = 7;
        state.conv.pop();
        assert!(state.validate_committed(&[7, 8], EOS).is_err());
        state.conv.push(0x7fc0);
        assert!(state.validate_committed(&[7, 8], EOS).is_err());
        state.conv.pop();
        state.conv.push(0);
        assert!(state.validate_committed(&[7, VOCAB], EOS).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn device_history_repeated_tokens_and_interleaved_sessions() {
        use crate::metal::{Binding, Buffer, Metal, Pipeline};

        fn uploaded(metal: &Metal, words: &[u16]) -> Buffer {
            let buffer = metal.shared_buffer(words.len() * 2).unwrap();
            buffer
                .write_bytes(
                    0,
                    &words
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            buffer
        }

        fn step(
            metal: &Metal,
            kernel: &Pipeline,
            state: &mut DevicePleState,
            alternate: &mut Buffer,
            weight: &Buffer,
            token: u32,
            value: f32,
        ) -> f32 {
            assert!(!state.conv.shares_allocation(alternate));
            let normalized = uploaded(metal, &vec![bf16::from_f32(value).to_bits(); EXPANDED]);
            let zero = uploaded(metal, &vec![0; EXPANDED]);
            let output = metal.shared_buffer(EXPANDED * 2).unwrap();
            let bindings = [
                &state.conv,
                &normalized,
                weight,
                &zero,
                &zero,
                &output,
                &*alternate,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            let mut batch = metal.batch().unwrap();
            // SAFETY: Each channel reads prior history and writes one output
            // and nine distinct cells in the alternate history allocation.
            unsafe {
                batch.dispatch_1d(kernel, &bindings, EXPANDED, 128).unwrap();
            }
            state.advance_history(alternate, token).unwrap();
            batch.submit().wait().unwrap();
            assert!(!state.conv.shares_allocation(alternate));
            let mut first = [0u8; 2];
            output.read_bytes(0, &mut first).unwrap();
            bf16::from_bits(u16::from_le_bytes(first)).to_f32()
        }

        let metal = Metal::new().unwrap();
        let source = format!(
            "{}\n{}",
            include_str!("../kernels/metal/hc_batch.metal"),
            include_str!("../kernels/metal/ple_batch.metal")
        );
        let kernel = metal
            .compile_msl_strict(&source, "ple_decode_conv_residual")
            .unwrap();
        let mut taps = vec![0u16; EXPANDED * CONV_TAPS];
        for channel in 0..EXPANDED {
            taps[channel * CONV_TAPS + 3] = bf16::from_f32(1.0).to_bits();
        }
        let weight = uploaded(&metal, &taps);
        let mut alternate = metal.shared_buffer(HISTORY * EXPANDED * 2).unwrap();
        let mut a = DevicePleState::zero(&metal).unwrap();
        let mut b = DevicePleState::zero(&metal).unwrap();
        let a1 = step(&metal, &kernel, &mut a, &mut alternate, &weight, 101, 1.0);
        let b1 = step(&metal, &kernel, &mut b, &mut alternate, &weight, 201, 2.0);
        let a2 = step(&metal, &kernel, &mut a, &mut alternate, &weight, 102, 3.0);
        let b2 = step(&metal, &kernel, &mut b, &mut alternate, &weight, 202, 4.0);
        assert!(a1 > 0.5 && a1 < b1 && b1 < a2 && a2 < b2);
        for (state, context, prior, latest) in
            [(&a, [101, 102], 1.0, 3.0), (&b, [201, 202], 2.0, 4.0)]
        {
            let host = state.to_host().unwrap();
            assert_eq!(host.context, context);
            assert_eq!(host.offset, 2);
            assert!(host.conv[..7 * EXPANDED].iter().all(|&word| word == 0));
            assert!(
                host.conv[7 * EXPANDED..8 * EXPANDED]
                    .iter()
                    .all(|&word| word == bf16::from_f32(prior).to_bits())
            );
            assert!(
                host.conv[8 * EXPANDED..]
                    .iter()
                    .all(|&word| word == bf16::from_f32(latest).to_bits())
            );
        }
    }
}

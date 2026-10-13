//! BF16 Qwen sampling policy over one vocabulary row.
//!
//! Presence processing and log-probability construction precede filtering.
//! Stochastic filtering runs top-p, min-p and top-k in that order, with
//! temperature applied only to the surviving BF16 log probabilities.

use std::collections::BTreeSet;

use anyhow::{Context, Result, ensure};
use half::bf16;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use serde_json::Value;

use crate::metal::{Binding, Buffer, Metal, Pipeline};
use crate::package::Package;

const NEG_INF: u16 = 0xff80;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingOptions {
    pub temperature: f64,
    pub top_p: f64,
    pub min_p: f64,
    pub top_k: usize,
    pub presence_penalty: f64,
    /// Zero uses the full history, matching Python's `tokens[-0:]` slice.
    pub presence_context_size: usize,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 1.0,
            min_p: 0.0,
            top_k: 0,
            presence_penalty: 0.0,
            presence_context_size: 20,
        }
    }
}

fn optional_number(config: &Value, name: &str) -> Result<Option<f64>> {
    let Some(value) = config.get(name).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    Ok(Some(value.as_f64().with_context(|| {
        format!("generation field {name} must be a number")
    })?))
}

impl SamplingOptions {
    fn from_generation_config(config: &Value) -> Result<Self> {
        ensure!(
            config.is_object(),
            "generation configuration must be an object"
        );
        let mut options = Self::default();
        if let Some(value) = optional_number(config, "temperature")? {
            options.temperature = value;
        }
        if let Some(value) = optional_number(config, "top_p")? {
            options.top_p = value;
        }
        if let Some(value) = optional_number(config, "min_p")? {
            options.min_p = value;
        }
        if let Some(value) = config.get("top_k").filter(|value| !value.is_null()) {
            options.top_k = usize::try_from(
                value
                    .as_u64()
                    .context("generation field top_k must be a nonnegative integer")?,
            )?;
        }
        if let Some(value) = optional_number(config, "presence_penalty")? {
            options.presence_penalty = value;
        }
        if let Some(value) = optional_number(config, "repetition_penalty")? {
            ensure!(
                value == 1.0,
                "non-neutral repetition penalty is unsupported"
            );
        }
        let do_sample = match config.get("do_sample").filter(|value| !value.is_null()) {
            Some(value) => Some(
                value
                    .as_bool()
                    .context("generation field do_sample must be boolean")?,
            ),
            None => None,
        };
        options.validate()?;
        if do_sample == Some(false) {
            options.temperature = 0.0;
            options.top_p = 1.0;
            options.min_p = 0.0;
            options.top_k = 0;
            options.presence_penalty = 0.0;
        }
        Ok(options)
    }

    /// Read the package's declared generation defaults once at setup time.
    pub fn from_package(package: &Package) -> Result<Self> {
        let path = package.tokenizer_file("generation_config.json")?;
        let config: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        Self::from_generation_config(&config)
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "temperature must be finite and nonnegative"
        );
        if self.temperature > 0.0 {
            ensure!(
                scalar(1.0 / self.temperature).is_finite(),
                "temperature reciprocal is not representable as finite BF16"
            );
        }
        ensure!(
            self.top_p.is_finite() && (0.0..=1.0).contains(&self.top_p),
            "top_p must be finite and between zero and one"
        );
        ensure!(
            self.min_p.is_finite() && (0.0..=1.0).contains(&self.min_p),
            "min_p must be finite and between zero and one"
        );
        ensure!(
            self.presence_penalty.is_finite() && scalar(self.presence_penalty).is_finite(),
            "presence penalty must be finite in BF16"
        );
        Ok(())
    }
}

fn bf(value: u16) -> f32 {
    bf16::from_bits(value).to_f32()
}

fn round(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}

fn scalar(value: f64) -> f32 {
    bf(round(value as f32))
}

fn presence(logits: &mut [u16], history: &[u32], options: SamplingOptions) -> Result<()> {
    if options.presence_penalty == 0.0 {
        return Ok(());
    }
    let start = if options.presence_context_size == 0 {
        0
    } else {
        history.len().saturating_sub(options.presence_context_size)
    };
    let seen: BTreeSet<u32> = history[start..].iter().copied().collect();
    let penalty = scalar(options.presence_penalty);
    for id in seen {
        let slot = logits
            .get_mut(id as usize)
            .context("presence history token exceeds vocabulary")?;
        *slot = round(bf(*slot) - penalty);
    }
    Ok(())
}

#[cfg(test)]
fn cpu_logsumexp(logits: &[u16]) -> Result<u16> {
    let maximum = logits
        .iter()
        .map(|&word| bf(word))
        .fold(f32::NEG_INFINITY, f32::max);
    ensure!(maximum.is_finite(), "sampling logits must be finite");
    let sum: f64 = logits
        .iter()
        .map(|&word| f64::from(bf(word) - maximum).exp())
        .sum();
    let normalizer = round((f64::from(maximum) + sum.ln()) as f32);
    Ok(normalizer)
}

trait PreparationMath {
    fn logsumexp_bf16(&mut self, logits: &[u16]) -> Result<u16>;
    fn exp_bf16(&mut self, values: &[u16]) -> Result<Vec<u16>>;
    fn cumulative_bf16(&mut self, values: &[u16]) -> Result<Vec<u16>>;

    fn exp_bf16_into(&mut self, values: &[u16], output: &mut Vec<u16>) -> Result<()> {
        *output = self.exp_bf16(values)?;
        Ok(())
    }

    fn cumulative_bf16_into(&mut self, values: &[u16], output: &mut Vec<u16>) -> Result<()> {
        *output = self.cumulative_bf16(values)?;
        Ok(())
    }
}

#[cfg(test)]
struct CpuPreparationMath;

#[cfg(test)]
impl PreparationMath for CpuPreparationMath {
    fn logsumexp_bf16(&mut self, logits: &[u16]) -> Result<u16> {
        cpu_logsumexp(logits)
    }

    fn exp_bf16(&mut self, values: &[u16]) -> Result<Vec<u16>> {
        Ok(values.iter().map(|&word| round(bf(word).exp())).collect())
    }

    fn cumulative_bf16(&mut self, values: &[u16]) -> Result<Vec<u16>> {
        cpu_bf16_scan(values)
    }
}

struct TopPIntermediates {
    exponentials: Vec<u16>,
    cumulative: Vec<u16>,
}

fn top_p_filter(
    logprobs: &mut [u16],
    top_p: f64,
    math: &mut impl PreparationMath,
) -> Result<Option<TopPIntermediates>> {
    if top_p <= 0.0 || top_p >= 1.0 {
        return Ok(None);
    }
    let exponentials = math.exp_bf16(logprobs)?;
    ensure!(
        exponentials.len() == logprobs.len(),
        "top-p exp length differs"
    );
    let mut order: Vec<usize> = (0..logprobs.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        bf(logprobs[a])
            .partial_cmp(&bf(logprobs[b]))
            .unwrap()
            .then(a.cmp(&b))
    });
    let threshold = scalar(1.0 - top_p);
    let sorted: Vec<u16> = order.iter().map(|&id| exponentials[id]).collect();
    let cumulative = math.cumulative_bf16(&sorted)?;
    ensure!(
        cumulative.len() == logprobs.len(),
        "top-p scan length differs"
    );
    let mut keep = vec![false; logprobs.len()];
    for (position, id) in order.into_iter().enumerate() {
        keep[id] = bf(cumulative[position]) > threshold;
    }
    for (id, value) in logprobs.iter_mut().enumerate() {
        if !keep[id] {
            *value = NEG_INF;
        }
    }
    Ok(Some(TopPIntermediates {
        exponentials,
        cumulative,
    }))
}

#[cfg(test)]
fn cpu_bf16_scan(words: &[u16]) -> Result<Vec<u16>> {
    let mut total = 0.0f32;
    Ok(words
        .iter()
        .map(|&word| {
            total += bf(word);
            round(total)
        })
        .collect())
}

fn min_p_filter(logprobs: &mut [u16], min_p: f64) {
    if min_p == 0.0 {
        return;
    }
    let maximum = logprobs
        .iter()
        .map(|&word| bf(word))
        .fold(f32::NEG_INFINITY, f32::max);
    let threshold = bf(round(maximum + scalar(min_p.ln())));
    for value in logprobs {
        if bf(*value) < threshold {
            *value = NEG_INF;
        }
    }
}

fn top_k_filter(logprobs: &mut [u16], top_k: usize) -> Result<()> {
    if top_k == 0 {
        return Ok(());
    }
    ensure!(
        top_k < logprobs.len(),
        "top_k must be below vocabulary size"
    );
    let mut order: Vec<usize> = (0..logprobs.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        bf(logprobs[b])
            .partial_cmp(&bf(logprobs[a]))
            .unwrap()
            .then(a.cmp(&b))
    });
    let mut keep = vec![false; logprobs.len()];
    for id in order.into_iter().take(top_k) {
        keep[id] = true;
    }
    for (id, value) in logprobs.iter_mut().enumerate() {
        if !keep[id] {
            *value = NEG_INF;
        }
    }
    Ok(())
}

// Five BF16 rows, one byte per token for presence, one u32 ID order, one
// f64 categorical row, two BF16 staging byte rows, and fixed BF16 buckets.
// Vec spare capacity and allocator overhead are accounted separately.
pub fn sampling_host_workspace_bytes(vocabulary: usize) -> Result<u64> {
    u64::try_from(vocabulary)?
        .checked_mul(27)
        .and_then(|bytes| bytes.checked_add(65_536 * 4))
        .context("sampling host workspace byte count overflow")
}

#[derive(Default)]
struct SamplingWorkspace {
    adjusted: Vec<u16>,
    logprobs: Vec<u16>,
    presence_seen: Vec<u8>,
    filter: FilterWorkspace,
    weights: Vec<f64>,
}

#[derive(Default)]
struct FilterWorkspace {
    exponentials: Vec<u16>,
    sorted_exponentials: Vec<u16>,
    cumulative: Vec<u16>,
    order: Vec<u32>,
    buckets: Vec<u32>,
}

fn bf16_is_nan(word: u16) -> bool {
    word & 0x7f80 == 0x7f80 && word & 0x007f != 0
}

// Unsigned keys have the same ascending order as BF16 numerical values.
// Both zero signs share a key so ID order breaks that tie, as in f32 compare.
fn bf16_order_key(word: u16) -> usize {
    let normalized = if word & 0x7fff == 0 { 0 } else { word };
    usize::from(if normalized & 0x8000 != 0 {
        !normalized
    } else {
        normalized ^ 0x8000
    })
}

fn count_score_keys(words: &[u16], buckets: &mut Vec<u32>) -> Result<()> {
    ensure!(
        words.iter().all(|&word| !bf16_is_nan(word)),
        "sampling log probabilities contain NaN"
    );
    buckets.resize(65_536, 0);
    buckets.fill(0);
    for &word in words {
        let count = &mut buckets[bf16_order_key(word)];
        *count = count
            .checked_add(1)
            .context("sampling key count overflow")?;
    }
    Ok(())
}

fn stable_ascending_order(words: &[u16], workspace: &mut FilterWorkspace) -> Result<()> {
    count_score_keys(words, &mut workspace.buckets)?;
    let mut next = 0u32;
    for count in workspace.buckets.iter_mut() {
        let size = *count;
        *count = next;
        next = next
            .checked_add(size)
            .context("sampling key offset overflow")?;
    }
    ensure!(
        usize::try_from(next)? == words.len(),
        "sampling key count differs"
    );
    workspace.order.resize(words.len(), 0);
    for (id, &word) in words.iter().enumerate() {
        let slot = &mut workspace.buckets[bf16_order_key(word)];
        workspace.order[*slot as usize] = u32::try_from(id)?;
        *slot += 1;
    }
    Ok(())
}

fn top_p_filter_id(
    logprobs: &mut [u16],
    top_p: f64,
    math: &mut impl PreparationMath,
    workspace: &mut FilterWorkspace,
) -> Result<()> {
    if top_p <= 0.0 || top_p >= 1.0 {
        return Ok(());
    }
    math.exp_bf16_into(logprobs, &mut workspace.exponentials)?;
    ensure!(
        workspace.exponentials.len() == logprobs.len(),
        "top-p exp length differs"
    );
    stable_ascending_order(logprobs, workspace)?;
    workspace.sorted_exponentials.resize(logprobs.len(), 0);
    for (position, &id) in workspace.order.iter().enumerate() {
        workspace.sorted_exponentials[position] = workspace.exponentials[id as usize];
    }
    math.cumulative_bf16_into(&workspace.sorted_exponentials, &mut workspace.cumulative)?;
    ensure!(
        workspace.cumulative.len() == logprobs.len(),
        "top-p scan length differs"
    );
    let threshold = scalar(1.0 - top_p);
    for (position, &id) in workspace.order.iter().enumerate() {
        if bf(workspace.cumulative[position]) <= threshold {
            logprobs[id as usize] = NEG_INF;
        }
    }
    Ok(())
}

fn top_k_filter_id(
    logprobs: &mut [u16],
    top_k: usize,
    workspace: &mut FilterWorkspace,
) -> Result<()> {
    if top_k == 0 {
        return Ok(());
    }
    ensure!(
        top_k < logprobs.len(),
        "top_k must be below vocabulary size"
    );
    count_score_keys(logprobs, &mut workspace.buckets)?;
    let mut remaining = u32::try_from(top_k)?;
    let mut cutoff = None;
    for (key, &count) in workspace.buckets.iter().enumerate().rev() {
        if count >= remaining {
            cutoff = Some(key);
            break;
        }
        remaining -= count;
    }
    let cutoff = cutoff.context("top-k score cutoff missing")?;
    for value in logprobs {
        let key = bf16_order_key(*value);
        if key < cutoff || (key == cutoff && remaining == 0) {
            *value = NEG_INF;
        } else if key == cutoff {
            remaining -= 1;
        }
    }
    Ok(())
}

fn prepare_id_with_math(
    logits_bf16: &[u16],
    history: &[u32],
    options: SamplingOptions,
    math: &mut impl PreparationMath,
    workspace: &mut SamplingWorkspace,
) -> Result<()> {
    options.validate()?;
    ensure!(!logits_bf16.is_empty(), "sampling needs a vocabulary row");
    ensure!(
        logits_bf16.iter().all(|&word| bf(word).is_finite()),
        "sampling logits must be finite BF16"
    );
    ensure!(
        u32::try_from(logits_bf16.len()).is_ok(),
        "sampling vocabulary exceeds u32"
    );
    workspace.adjusted.clear();
    workspace.adjusted.extend_from_slice(logits_bf16);
    if options.presence_penalty != 0.0 {
        let start = if options.presence_context_size == 0 {
            0
        } else {
            history.len().saturating_sub(options.presence_context_size)
        };
        workspace.presence_seen.resize(logits_bf16.len(), 0);
        workspace.presence_seen.fill(0);
        for &id in &history[start..] {
            let seen = workspace
                .presence_seen
                .get_mut(id as usize)
                .context("presence history token exceeds vocabulary")?;
            *seen = 1;
        }
        let penalty = scalar(options.presence_penalty);
        for (id, &seen) in workspace.presence_seen.iter().enumerate() {
            if seen != 0 {
                let slot = &mut workspace.adjusted[id];
                *slot = round(bf(*slot) - penalty);
            }
        }
    }
    let normalizer = math.logsumexp_bf16(&workspace.adjusted)?;
    workspace.logprobs.clear();
    workspace.logprobs.extend(
        workspace
            .adjusted
            .iter()
            .map(|&word| round(bf(word) - bf(normalizer))),
    );
    ensure!(
        workspace.logprobs.iter().all(|&word| !bf16_is_nan(word)),
        "sampling log probabilities contain NaN"
    );
    if options.temperature > 0.0 {
        top_p_filter_id(
            &mut workspace.logprobs,
            options.top_p,
            math,
            &mut workspace.filter,
        )?;
        min_p_filter(&mut workspace.logprobs, options.min_p);
        top_k_filter_id(
            &mut workspace.logprobs,
            options.top_k,
            &mut workspace.filter,
        )?;
        let reciprocal = scalar(1.0 / options.temperature);
        for value in &mut workspace.logprobs {
            *value = round(bf(*value) * reciprocal);
        }
    }
    Ok(())
}

/// Numeric stage values remain BF16 at every public reference boundary.
pub struct SamplingStages {
    pub adjusted_logits_bf16: Vec<u16>,
    pub logsumexp_bf16: u16,
    pub logprobs_bf16: Vec<u16>,
    pub top_p_exp_bf16: Option<Vec<u16>>,
    /// Inclusive cumulative values in ascending log-probability order.
    pub top_p_cumulative_bf16: Option<Vec<u16>>,
    pub top_p_bf16: Vec<u16>,
    pub min_p_bf16: Vec<u16>,
    pub filtered_bf16: Vec<u16>,
    pub scaled_bf16: Option<Vec<u16>>,
}

/// Test-only host approximation. Production preparation uses Metal reductions.
#[cfg(test)]
fn prepare_cpu_diagnostic(
    logits_bf16: &[u16],
    history: &[u32],
    options: SamplingOptions,
) -> Result<SamplingStages> {
    prepare_with_math(logits_bf16, history, options, &mut CpuPreparationMath)
}

fn prepare_with_math(
    logits_bf16: &[u16],
    history: &[u32],
    options: SamplingOptions,
    math: &mut impl PreparationMath,
) -> Result<SamplingStages> {
    options.validate()?;
    ensure!(!logits_bf16.is_empty(), "sampling needs a vocabulary row");
    ensure!(
        logits_bf16.iter().all(|&word| bf(word).is_finite()),
        "sampling logits must be finite BF16"
    );
    let mut adjusted = logits_bf16.to_vec();
    presence(&mut adjusted, history, options)?;
    let logsumexp_bf16 = math.logsumexp_bf16(&adjusted)?;
    let logprobs_bf16: Vec<u16> = adjusted
        .iter()
        .map(|&word| round(bf(word) - bf(logsumexp_bf16)))
        .collect();
    let mut filtered = logprobs_bf16.clone();
    let top_p_intermediates = if options.temperature > 0.0 {
        top_p_filter(&mut filtered, options.top_p, math)?
    } else {
        None
    };
    let (top_p_exp_bf16, top_p_cumulative_bf16) = match top_p_intermediates {
        Some(stages) => (Some(stages.exponentials), Some(stages.cumulative)),
        None => (None, None),
    };
    let top_p_bf16 = filtered.clone();
    if options.temperature > 0.0 {
        min_p_filter(&mut filtered, options.min_p);
    }
    let min_p_bf16 = filtered.clone();
    if options.temperature > 0.0 {
        top_k_filter(&mut filtered, options.top_k)?;
    }
    let scaled_bf16 = (options.temperature > 0.0).then(|| {
        let reciprocal = scalar(1.0 / options.temperature);
        filtered
            .iter()
            .map(|&word| round(bf(word) * reciprocal))
            .collect()
    });
    Ok(SamplingStages {
        adjusted_logits_bf16: adjusted,
        logsumexp_bf16,
        logprobs_bf16,
        top_p_exp_bf16,
        top_p_cumulative_bf16,
        top_p_bf16,
        min_p_bf16,
        filtered_bf16: filtered,
        scaled_bf16,
    })
}

/// Reusable Metal reductions for the sampler's BF16 logsumexp and top-p scan.
///
/// Host filtering follows the package policy. Diagnostic stage capture and
/// production ID-only draws share these Metal reductions.
pub struct MetalSamplingPreparation {
    metal: Metal,
    lse_block_pipeline: Pipeline,
    lse_looped_pipeline: Pipeline,
    exp_pipeline: Pipeline,
    scan_pipeline: Pipeline,
    scan_input: Buffer,
    scan_output: Buffer,
    scan_count: Buffer,
    host_input_bytes: Vec<u8>,
    host_output_bytes: Vec<u8>,
    max_vocabulary: usize,
}

impl MetalSamplingPreparation {
    pub fn new(metal: &Metal, max_vocabulary: usize) -> Result<Self> {
        ensure!(
            max_vocabulary > 0 && max_vocabulary <= i32::MAX as usize,
            "sampling vocabulary must fit a nonzero i32"
        );
        let bytes = max_vocabulary
            .checked_mul(2)
            .context("sampling scan allocation exceeds usize")?;
        let scan_pipeline = metal.compile_msl_strict(
            include_str!("../kernels/metal/sampling_scan.metal"),
            "sampling_bf16_cumsum",
        )?;
        let lse_source = include_str!("../kernels/metal/sampling_lse.metal");
        let lse_block_pipeline =
            metal.compile_msl_strict(lse_source, "sampling_bf16_logsumexp_block")?;
        let lse_looped_pipeline =
            metal.compile_msl_strict(lse_source, "sampling_bf16_logsumexp_looped")?;
        let exp_pipeline = metal.compile_msl_strict(
            include_str!("../kernels/metal/sampling_exp.metal"),
            "sampling_bf16_exp",
        )?;
        ensure!(
            scan_pipeline.max_threads_per_threadgroup() >= 32
                && lse_block_pipeline.max_threads_per_threadgroup() >= 32
                && lse_looped_pipeline.max_threads_per_threadgroup() >= 32
                && exp_pipeline.max_threads_per_threadgroup() > 0,
            "Metal sampling reductions require one SIMD group"
        );
        Ok(Self {
            metal: metal.clone(),
            lse_block_pipeline,
            lse_looped_pipeline,
            exp_pipeline,
            scan_pipeline,
            scan_input: metal.shared_buffer(bytes)?,
            scan_output: metal.shared_buffer(bytes)?,
            scan_count: metal.shared_buffer(4)?,
            host_input_bytes: vec![0; bytes],
            host_output_bytes: vec![0; bytes],
            max_vocabulary,
        })
    }

    fn write_input(&mut self, words: &[u16]) -> Result<()> {
        let bytes = &mut self.host_input_bytes[..words.len() * 2];
        let (pairs, remainder) = bytes.as_chunks_mut::<2>();
        debug_assert!(remainder.is_empty());
        for (&word, pair) in words.iter().zip(pairs.iter_mut()) {
            pair.copy_from_slice(&word.to_le_bytes());
        }
        self.scan_input.write_bytes(0, bytes)?;
        self.scan_count
            .write_bytes(0, &u32::try_from(words.len())?.to_le_bytes())?;
        Ok(())
    }

    fn read_output(&mut self, output: &mut Vec<u16>, len: usize) -> Result<()> {
        let bytes = &mut self.host_output_bytes[..len * 2];
        self.scan_output.read_bytes(0, bytes)?;
        output.resize(len, 0);
        let (pairs, remainder) = bytes.as_chunks::<2>();
        debug_assert!(remainder.is_empty());
        for (word, pair) in output.iter_mut().zip(pairs.iter()) {
            *word = u16::from_le_bytes(*pair);
        }
        Ok(())
    }

    /// Convert BF16 log probabilities to BF16 top-p probabilities on Metal.
    pub fn exp_bf16(&mut self, words: &[u16]) -> Result<Vec<u16>> {
        let mut output = Vec::new();
        self.exp_bf16_into(words, &mut output)?;
        Ok(output)
    }

    fn exp_bf16_into(&mut self, words: &[u16], output: &mut Vec<u16>) -> Result<()> {
        ensure!(
            !words.is_empty() && words.len() <= self.max_vocabulary,
            "sampling exp row exceeds configured vocabulary"
        );
        self.write_input(words)?;
        let group_threads = self.exp_pipeline.max_threads_per_threadgroup().min(256);
        let mut batch = self.metal.batch()?;
        // SAFETY: input and output each hold count BF16 values, the count
        // binding holds a uint, and each kernel thread checks its index.
        unsafe {
            batch.dispatch_1d(
                &self.exp_pipeline,
                &[
                    Binding {
                        buffer: &self.scan_input,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.scan_output,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.scan_count,
                        offset: 0,
                    },
                ],
                words.len(),
                group_threads,
            )?;
        }
        batch.submit().wait()?;
        self.read_output(output, words.len())
    }

    /// Reduce one finite BF16 row to a BF16 logsumexp scalar.
    pub fn logsumexp_bf16(&mut self, words: &[u16]) -> Result<u16> {
        ensure!(
            !words.is_empty() && words.len() <= self.max_vocabulary,
            "sampling reduction row exceeds configured vocabulary"
        );
        ensure!(
            words.iter().all(|&word| bf(word).is_finite()),
            "sampling logits must be finite BF16"
        );
        self.write_input(words)?;
        let (pipeline, threads) = if words.len() <= 4096 {
            (&self.lse_block_pipeline, words.len().div_ceil(128) * 32)
        } else {
            (
                &self.lse_looped_pipeline,
                self.lse_looped_pipeline.max_threads_per_threadgroup(),
            )
        };
        ensure!(
            threads > 0
                && threads.is_multiple_of(32)
                && threads <= pipeline.max_threads_per_threadgroup(),
            "Metal logsumexp needs a supported whole number of SIMD groups"
        );
        let mut batch = self.metal.batch()?;
        // SAFETY: input contains count BF16 values, output has one BF16 value,
        // and the uint count binding matches the MSL signature. One validated
        // threadgroup bounds-checks all partial reads.
        unsafe {
            batch.dispatch_1d(
                pipeline,
                &[
                    Binding {
                        buffer: &self.scan_input,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.scan_output,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.scan_count,
                        offset: 0,
                    },
                ],
                threads,
                threads,
            )?;
        }
        batch.submit().wait()?;
        let mut output_bytes = [0u8; 2];
        self.scan_output.read_bytes(0, &mut output_bytes)?;
        Ok(u16::from_le_bytes(output_bytes))
    }

    /// Scan a BF16 row in input order, preserving the reference's GPU topology.
    pub fn cumulative_bf16(&mut self, words: &[u16]) -> Result<Vec<u16>> {
        let mut output = Vec::new();
        self.cumulative_bf16_into(words, &mut output)?;
        Ok(output)
    }

    fn cumulative_bf16_into(&mut self, words: &[u16], output: &mut Vec<u16>) -> Result<()> {
        ensure!(
            !words.is_empty() && words.len() <= self.max_vocabulary,
            "sampling scan row exceeds configured vocabulary"
        );
        self.write_input(words)?;
        let maximum = self.scan_pipeline.max_threads_per_threadgroup();
        let threads = if words.len() <= 4096 {
            words.len().div_ceil(128) * 32
        } else if words.len() <= 8192 {
            (words.len() / 2).div_ceil(128) * 32
        } else {
            maximum
        }
        .min(maximum);
        ensure!(
            threads > 0 && threads.is_multiple_of(32),
            "Metal scan needs a whole number of SIMD groups"
        );
        let mut batch = self.metal.batch()?;
        // SAFETY: input and output each hold at least count BF16 values, the
        // count binding holds a uint, and the single-group kernel checks tail
        // accesses. The group size is within the compiled pipeline limit.
        unsafe {
            batch.dispatch_1d(
                &self.scan_pipeline,
                &[
                    Binding {
                        buffer: &self.scan_input,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.scan_output,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.scan_count,
                        offset: 0,
                    },
                ],
                threads,
                threads,
            )?;
        }
        batch.submit().wait()?;
        self.read_output(output, words.len())
    }

    /// Prepare the supported filters using the Metal BF16 top-p scan.
    pub fn prepare(
        &mut self,
        logits_bf16: &[u16],
        history: &[u32],
        options: SamplingOptions,
    ) -> Result<SamplingStages> {
        ensure!(
            logits_bf16.len() <= self.max_vocabulary,
            "sampling row exceeds configured vocabulary"
        );
        prepare_with_math(logits_bf16, history, options, self)
    }
}

impl PreparationMath for MetalSamplingPreparation {
    fn logsumexp_bf16(&mut self, logits: &[u16]) -> Result<u16> {
        Self::logsumexp_bf16(self, logits)
    }

    fn exp_bf16(&mut self, values: &[u16]) -> Result<Vec<u16>> {
        Self::exp_bf16(self, values)
    }

    fn exp_bf16_into(&mut self, values: &[u16], output: &mut Vec<u16>) -> Result<()> {
        Self::exp_bf16_into(self, values, output)
    }

    fn cumulative_bf16(&mut self, values: &[u16]) -> Result<Vec<u16>> {
        Self::cumulative_bf16(self, values)
    }

    fn cumulative_bf16_into(&mut self, values: &[u16], output: &mut Vec<u16>) -> Result<()> {
        Self::cumulative_bf16_into(self, values, output)
    }
}

pub struct SampledToken {
    pub id: u32,
    pub logprobs_bf16: Vec<u16>,
}

pub struct Sampler {
    options: SamplingOptions,
    rng: ChaCha8Rng,
    workspace: SamplingWorkspace,
}

impl Sampler {
    pub fn new(options: SamplingOptions, seed: u64) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            options,
            rng: ChaCha8Rng::seed_from_u64(seed),
            workspace: SamplingWorkspace::default(),
        })
    }

    pub(crate) fn options(&self) -> SamplingOptions {
        self.options
    }

    /// Draw only the token ID, retaining host scratch for the next row.
    /// Diagnostic stage capture remains available through `sample`.
    pub fn sample_id(
        &mut self,
        preparation: &mut MetalSamplingPreparation,
        logits_bf16: &[u16],
        history: &[u32],
    ) -> Result<u32> {
        ensure!(
            logits_bf16.len() <= preparation.max_vocabulary,
            "sampling row exceeds configured vocabulary"
        );
        self.sample_id_with_math(preparation, logits_bf16, history)
    }

    fn sample_id_with_math(
        &mut self,
        math: &mut impl PreparationMath,
        logits_bf16: &[u16],
        history: &[u32],
    ) -> Result<u32> {
        prepare_id_with_math(
            logits_bf16,
            history,
            self.options,
            math,
            &mut self.workspace,
        )?;
        let logprobs = &self.workspace.logprobs;
        let id = if self.options.temperature > 0.0 {
            let maximum = logprobs
                .iter()
                .map(|&word| bf(word))
                .fold(f32::NEG_INFINITY, f32::max);
            ensure!(
                maximum.is_finite(),
                "sampling filters removed the entire vocabulary"
            );
            let weights = &mut self.workspace.weights;
            weights.clear();
            let mut total = 0.0f64;
            for &word in logprobs {
                let weight = f64::from(bf(word) - maximum).exp();
                weights.push(weight);
                total += weight;
            }
            ensure!(
                total.is_finite() && total > 0.0,
                "invalid categorical weights"
            );
            let unit = (self.rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let target = unit * total;
            let mut cumulative = 0.0;
            let mut chosen = None;
            for (index, &weight) in weights.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                cumulative += weight;
                if cumulative > target {
                    chosen = Some(index);
                    break;
                }
                chosen = Some(index);
            }
            chosen.context("categorical distribution has no positive weight")?
        } else {
            let mut best = 0usize;
            let mut score = bf(logprobs[0]);
            for (id, &word) in logprobs.iter().enumerate().skip(1) {
                let candidate = bf(word);
                if candidate > score {
                    best = id;
                    score = candidate;
                }
            }
            best
        };
        u32::try_from(id).context("sampled token ID exceeds u32")
    }

    /// Prepare with the reference Metal reductions, then draw one token.
    pub fn sample(
        &mut self,
        preparation: &mut MetalSamplingPreparation,
        logits_bf16: &[u16],
        history: &[u32],
    ) -> Result<SampledToken> {
        let stages = preparation.prepare(logits_bf16, history, self.options)?;
        self.draw_prepared(stages)
    }

    /// Draw from exp(BF16-filtered-logprob / temperature) in token-ID order.
    fn draw_prepared(&mut self, stages: SamplingStages) -> Result<SampledToken> {
        let id = if let Some(scaled) = &stages.scaled_bf16 {
            let maximum = scaled
                .iter()
                .map(|&word| bf(word))
                .fold(f32::NEG_INFINITY, f32::max);
            ensure!(
                maximum.is_finite(),
                "sampling filters removed the entire vocabulary"
            );
            let mut weights = Vec::with_capacity(scaled.len());
            let mut total = 0.0f64;
            for &word in scaled {
                let weight = f64::from(bf(word) - maximum).exp();
                weights.push(weight);
                total += weight;
            }
            ensure!(
                total.is_finite() && total > 0.0,
                "invalid categorical weights"
            );
            let unit = (self.rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let target = unit * total;
            let mut cumulative = 0.0;
            let mut chosen = None;
            for (index, weight) in weights.into_iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                cumulative += weight;
                if cumulative > target {
                    chosen = Some(index);
                    break;
                }
                chosen = Some(index);
            }
            chosen.context("categorical distribution has no positive weight")?
        } else {
            stages
                .logprobs_bf16
                .iter()
                .enumerate()
                .max_by(|a, b| bf(*a.1).partial_cmp(&bf(*b.1)).unwrap().then(b.0.cmp(&a.0)))
                .map(|(id, _)| id)
                .context("greedy sampling row is empty")?
        };
        Ok(SampledToken {
            id: u32::try_from(id).context("sampled token ID exceeds u32")?,
            logprobs_bf16: stages.logprobs_bf16,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bits(values: &[f32]) -> Vec<u16> {
        values.iter().map(|&value| round(value)).collect()
    }

    fn sample_cpu(sampler: &mut Sampler, row: &[u16], history: &[u32]) -> Result<SampledToken> {
        let stages = prepare_cpu_diagnostic(row, history, sampler.options)?;
        sampler.draw_prepared(stages)
    }

    #[test]
    fn package_defaults_and_greedy_override_are_validated() {
        let defaults = SamplingOptions::from_generation_config(&json!({
            "do_sample": true, "temperature": 1.0, "top_p": 0.95, "top_k": 20
        }))
        .unwrap();
        assert_eq!(defaults.temperature, 1.0);
        assert_eq!(defaults.top_p, 0.95);
        assert_eq!(defaults.top_k, 20);
        let greedy = SamplingOptions::from_generation_config(&json!({
            "do_sample": false, "temperature": 0.7, "top_p": 0.5,
            "top_k": 7, "min_p": 0.1, "presence_penalty": 0.4,
            "repetition_penalty": 1.0
        }))
        .unwrap();
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(greedy.top_p, 1.0);
        assert_eq!(greedy.min_p, 0.0);
        assert_eq!(greedy.top_k, 0);
        assert_eq!(greedy.presence_penalty, 0.0);
        assert!(
            SamplingOptions::from_generation_config(&json!({"repetition_penalty":1.2})).is_err()
        );
        assert!(SamplingOptions::from_generation_config(&json!({"temperature":-1})).is_err());
        assert!(SamplingOptions::from_generation_config(&json!({"top_p":true})).is_err());
        assert!(SamplingOptions::from_generation_config(&json!({"temperature":1e-100})).is_err());
        assert!(
            SamplingOptions::from_generation_config(&json!({"presence_penalty":1e100})).is_err()
        );
    }

    #[test]
    fn repeated_presence_counts_once_in_last_window() {
        let options = SamplingOptions {
            presence_penalty: 0.5,
            presence_context_size: 4,
            ..SamplingOptions::default()
        };
        let stages =
            prepare_cpu_diagnostic(&bits(&[4.0, 4.0, 1.0, 0.0]), &[2, 0, 0, 1, 0], options)
                .unwrap();
        assert_eq!(stages.adjusted_logits_bf16, bits(&[3.5, 3.5, 1.0, 0.0]));
        assert!(prepare_cpu_diagnostic(&bits(&[1.0]), &[1], options).is_err());
        let fractional = SamplingOptions {
            presence_penalty: 0.6,
            ..options
        };
        assert_eq!(
            prepare_cpu_diagnostic(&bits(&[2.734375, 0.0]), &[0], fractional)
                .unwrap()
                .adjusted_logits_bf16[0],
            round(2.125)
        );
    }

    #[test]
    fn top_p_strict_bf16_threshold_and_greedy_bypass() {
        let mut options = SamplingOptions {
            temperature: 1.0,
            top_p: 0.5,
            ..SamplingOptions::default()
        };
        let row = bits(&[0.0, 0.0]);
        assert_eq!(
            prepare_cpu_diagnostic(&row, &[], options)
                .unwrap()
                .filtered_bf16,
            vec![NEG_INF, round(-0.69140625)]
        );
        options.top_p = 0.5001;
        assert_eq!(
            prepare_cpu_diagnostic(&row, &[], options)
                .unwrap()
                .filtered_bf16[0],
            NEG_INF
        );
        options.temperature = 0.0;
        assert_eq!(
            prepare_cpu_diagnostic(&row, &[], options)
                .unwrap()
                .filtered_bf16,
            bits(&[-0.69140625, -0.69140625])
        );
        assert_eq!(
            sample_cpu(&mut Sampler::new(options, 3).unwrap(), &row, &[])
                .unwrap()
                .id,
            0
        );
    }

    #[test]
    fn top_k_tie_prefers_lower_id_and_filters_after_min_p() {
        let options = SamplingOptions {
            temperature: 1.0,
            top_k: 2,
            ..SamplingOptions::default()
        };
        assert_eq!(
            prepare_cpu_diagnostic(&bits(&[4.0, 3.0, 3.0, 3.0, 2.0]), &[], options)
                .unwrap()
                .filtered_bf16,
            vec![round(-0.8125), round(-1.8125), NEG_INF, NEG_INF, NEG_INF]
        );
        assert!(
            prepare_cpu_diagnostic(
                &bits(&[1.0, 0.0]),
                &[],
                SamplingOptions {
                    top_k: 2,
                    ..options
                }
            )
            .is_err()
        );
    }

    #[test]
    fn seeded_draws_are_reproducible_and_non_degenerate() {
        let options = SamplingOptions {
            temperature: 1.0,
            ..SamplingOptions::default()
        };
        let row = bits(&[0.0, 0.0]);
        let mut first = Sampler::new(options, 98245).unwrap();
        let mut second = Sampler::new(options, 98245).unwrap();
        let a: Vec<u32> = (0..64)
            .map(|_| sample_cpu(&mut first, &row, &[]).unwrap().id)
            .collect();
        let b: Vec<u32> = (0..64)
            .map(|_| sample_cpu(&mut second, &row, &[]).unwrap().id)
            .collect();
        assert_eq!(a, b);
        assert!(a.contains(&0) && a.contains(&1));
    }

    #[test]
    fn temperature_reciprocal_is_bf16_before_scaling() {
        let options = SamplingOptions {
            temperature: 1.0 / 1.0038,
            ..SamplingOptions::default()
        };
        let stages = prepare_cpu_diagnostic(&bits(&[0.0, -10.0]), &[], options).unwrap();
        assert_eq!(stages.logprobs_bf16[1], round(-10.0));
        assert_eq!(stages.scaled_bf16.unwrap()[1], round(-10.0));
    }

    #[test]
    fn empty_bf16_top_p_distribution_fails_closed() {
        let options = SamplingOptions {
            temperature: 1.0,
            top_p: 0.0001,
            ..SamplingOptions::default()
        };
        let row = bits(&[0.0, 0.0]);
        assert_eq!(
            prepare_cpu_diagnostic(&row, &[], options)
                .unwrap()
                .filtered_bf16,
            [NEG_INF; 2]
        );
        assert!(sample_cpu(&mut Sampler::new(options, 7).unwrap(), &row, &[]).is_err());
    }

    #[test]
    fn greedy_uses_rounded_logprobs_not_raw_logits() {
        let mut row = bits(&[0.0, 0.001]);
        row.resize(32, round(0.0));
        let options = SamplingOptions {
            temperature: 0.0,
            ..SamplingOptions::default()
        };
        let stages = prepare_cpu_diagnostic(&row, &[], options).unwrap();
        assert_eq!(stages.logprobs_bf16[0], stages.logprobs_bf16[1]);
        assert_eq!(
            sample_cpu(&mut Sampler::new(options, 0).unwrap(), &row, &[])
                .unwrap()
                .id,
            0
        );
    }

    #[test]
    fn bf16_keys_preserve_numeric_order_and_id_ties() {
        let words: Vec<u16> = (0..=u16::MAX).filter(|&word| !bf16_is_nan(word)).collect();
        let mut words_by_key = words.clone();
        words_by_key.sort_unstable_by_key(|&word| bf16_order_key(word));
        assert!(
            words_by_key
                .windows(2)
                .all(|pair| bf(pair[0]) <= bf(pair[1]))
        );
        let mut expected: Vec<usize> = (0..words.len()).collect();
        expected.sort_unstable_by(|&a, &b| {
            bf(words[a])
                .partial_cmp(&bf(words[b]))
                .unwrap()
                .then(a.cmp(&b))
        });
        let mut filter = FilterWorkspace::default();
        stable_ascending_order(&words, &mut filter).unwrap();
        assert_eq!(
            filter.order,
            expected.iter().map(|&id| id as u32).collect::<Vec<_>>()
        );
        assert_eq!(bf16_order_key(0x8000), bf16_order_key(0));
        let words = [NEG_INF, round(1.0), 0x8000, 0, round(-1.0), round(1.0)];
        stable_ascending_order(&words, &mut filter).unwrap();
        assert_eq!(filter.order, [0, 4, 2, 3, 1, 5]);
        assert!(count_score_keys(&[0x7f81], &mut filter.buckets).is_err());
        for top_k in 1..words.len() {
            let mut old = words.to_vec();
            let mut new = words.to_vec();
            top_k_filter(&mut old, top_k).unwrap();
            top_k_filter_id(&mut new, top_k, &mut filter).unwrap();
            assert_eq!(new, old);
        }
    }

    #[test]
    fn lean_bf16_filters_match_diagnostic_stages() {
        let mut state = 0x82b0_77a5_u32;
        let mut workspace = SamplingWorkspace::default();
        for len in [2usize, 7, 31, 257, 1024] {
            let mut row = Vec::with_capacity(len);
            for id in 0..len {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = ((state >> 17) % 41) as f32 / 8.0 - 2.5;
                row.push(if id % 13 == 0 { 0x8000 } else { round(value) });
            }
            let history = [0, 1, 1, (len - 1) as u32, 0];
            for (top_p, min_p, top_k, penalty) in [
                (1.0, 0.0, 0, 0.0),
                (0.7, 0.0, 0, 0.0),
                (0.93, 0.1, len - 1, 0.5),
                (0.4, 0.02, 1, -0.25),
            ] {
                let options = SamplingOptions {
                    temperature: 0.8,
                    top_p,
                    min_p,
                    top_k,
                    presence_penalty: penalty,
                    presence_context_size: 3,
                };
                let expected = prepare_cpu_diagnostic(&row, &history, options).unwrap();
                prepare_id_with_math(
                    &row,
                    &history,
                    options,
                    &mut CpuPreparationMath,
                    &mut workspace,
                )
                .unwrap();
                assert_eq!(workspace.adjusted, expected.adjusted_logits_bf16);
                if let Some(exponentials) = expected.top_p_exp_bf16 {
                    assert_eq!(workspace.filter.exponentials, exponentials);
                }
                if let Some(cumulative) = expected.top_p_cumulative_bf16 {
                    assert_eq!(workspace.filter.cumulative, cumulative);
                }
                assert_eq!(workspace.logprobs, expected.scaled_bf16.unwrap());
            }
            let greedy = SamplingOptions {
                temperature: 0.0,
                presence_penalty: 0.5,
                presence_context_size: 0,
                ..SamplingOptions::default()
            };
            let expected = prepare_cpu_diagnostic(&row, &history, greedy).unwrap();
            prepare_id_with_math(
                &row,
                &history,
                greedy,
                &mut CpuPreparationMath,
                &mut workspace,
            )
            .unwrap();
            assert_eq!(workspace.logprobs, expected.logprobs_bf16);
        }
    }

    #[test]
    fn lean_seeded_ids_and_error_paths_match_diagnostic() {
        let row = bits(&[2.0, 1.0, 1.0, 0.5, 0.0, -0.5, -2.0]);
        for options in [
            SamplingOptions {
                temperature: 0.0,
                presence_penalty: 0.25,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                temperature: 0.8,
                top_p: 0.9,
                min_p: 0.1,
                top_k: 4,
                presence_penalty: 0.25,
                ..SamplingOptions::default()
            },
        ] {
            let mut reference = Sampler::new(options, 823_947).unwrap();
            let mut lean = Sampler::new(options, 823_947).unwrap();
            for step in 0..64 {
                let history = [step % row.len() as u32, 2, 2, 0];
                let expected = sample_cpu(&mut reference, &row, &history).unwrap().id;
                let actual = lean
                    .sample_id_with_math(&mut CpuPreparationMath, &row, &history)
                    .unwrap();
                assert_eq!(actual, expected);
            }
        }
        let options = SamplingOptions {
            temperature: 1.0,
            top_p: 0.0001,
            ..SamplingOptions::default()
        };
        assert!(
            Sampler::new(options, 7)
                .unwrap()
                .sample_id_with_math(&mut CpuPreparationMath, &bits(&[0.0, 0.0]), &[])
                .is_err()
        );
        let mut sampler = Sampler::new(
            SamplingOptions {
                temperature: 1.0,
                top_k: 2,
                presence_penalty: 0.5,
                ..SamplingOptions::default()
            },
            1,
        )
        .unwrap();
        assert!(
            sampler
                .sample_id_with_math(&mut CpuPreparationMath, &[], &[])
                .is_err()
        );
        assert!(
            sampler
                .sample_id_with_math(&mut CpuPreparationMath, &bits(&[0.0, 0.0]), &[2])
                .is_err()
        );
        assert!(
            sampler
                .sample_id_with_math(&mut CpuPreparationMath, &bits(&[0.0, 0.0]), &[])
                .is_err()
        );
        assert!(
            sampler
                .sample_id_with_math(&mut CpuPreparationMath, &[0x7f80], &[])
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires Metal GPU ownership"]
    fn metal_id_only_matches_diagnostic_seeded_draws() -> Result<()> {
        let metal = Metal::new()?;
        let mut preparation = MetalSamplingPreparation::new(&metal, 257)?;
        let rows: Vec<Vec<u16>> = [0usize, 19]
            .into_iter()
            .map(|shift| {
                (0..257)
                    .map(|id| {
                        if id % 17 == 0 {
                            0x8000
                        } else {
                            round(((id * 37 + shift) % 73) as f32 / 8.0 - 4.5)
                        }
                    })
                    .collect()
            })
            .collect();
        let history = [0, 17, 17, 256, 5];
        for options in [
            SamplingOptions {
                temperature: 0.0,
                presence_penalty: 0.25,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                temperature: 0.8,
                top_p: 0.91,
                min_p: 0.02,
                top_k: 32,
                presence_penalty: 0.25,
                presence_context_size: 3,
            },
        ] {
            let mut diagnostic = Sampler::new(options, 237_871)?;
            let mut lean = Sampler::new(options, 237_871)?;
            for row in rows.iter().cycle().take(4) {
                let expected = diagnostic.sample(&mut preparation, row, &history)?.id;
                let actual = lean.sample_id(&mut preparation, row, &history)?;
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }
}

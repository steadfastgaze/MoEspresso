//! Fixed-shape GDN, QSA gate and shared MLP projection offload.
//!
//! One serialized inference owner retains a dynamic-weight ANE program. The
//! GPU packs activations and existing Q6_K gate weights into shared staging.
//! Each pack retires the preceding suffix before staging reuse. The combined
//! owner selects a 4096-row or 8192-row projection at startup. The 8192-row
//! shape also accepts fronts containing at least 4096 useful rows. Smaller
//! tails and decode stay on GPU.

use crate::ane::{DynamicLayout, DynamicProjection, ProjectionShape};
use crate::gdn_batch::{HIDDEN, VALUE};
use crate::metal::{Batch, Binding, Buffer, Completion, Metal, Pipeline, Stream};
use crate::model::{MixerKind, QwenConfig};
use crate::package::Package;
#[cfg(test)]
use crate::package::{DirectWeight, WeightFormat};
use anyhow::{Context, Result, bail, ensure};
#[cfg(test)]
use half::f16;

const PACK_SOURCE: &str = include_str!("ane_prefill/pack_q6.metal");
const PADDED_UNPACK_SOURCE: &str = include_str!("ane_prefill/padded_unpack.metal");
const LAYERS: usize = 48;
const TILE: usize = 32;
const THREADS: usize = 256;
const SHARED: usize = 640;
pub(super) const MAX_PHYSICAL_ROWS: usize = 8192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProjectionKind {
    Gdn,
    QsaGate,
    SharedFront,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Idle,
    Packing(StagedProjection),
    Input(StagedProjection),
    Output(StagedProjection),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProjectionRows {
    physical: usize,
    useful: usize,
}

impl ProjectionRows {
    fn gdn(physical: usize, useful: usize) -> Result<Self> {
        validate_rows(physical)?;
        ensure!(
            useful == physical || (physical == 8192 && (4096..=8192).contains(&useful)),
            "ANE GDN useful rows differ from the physical projection"
        );
        Ok(Self { physical, useful })
    }

    #[cfg(test)]
    fn full(physical: usize) -> Result<Self> {
        validate_rows(physical)?;
        Ok(Self {
            physical,
            useful: physical,
        })
    }

    fn qsa(physical: usize, useful: usize) -> Result<Self> {
        ensure!(
            matches!(physical, 4096 | MAX_PHYSICAL_ROWS),
            "ANE QSA physical rows differ"
        );
        Self::gdn(physical, useful)
    }

    fn bytes(self, channels: usize) -> Result<(usize, usize)> {
        ensure!(channels != 0, "ANE projection channels are empty");
        let extent = |rows: usize| {
            rows.checked_mul(channels)
                .and_then(|n| n.checked_mul(2))
                .filter(|&n| n <= u32::MAX as usize)
                .context("ANE projection byte extent exceeds uint")
        };
        Ok((extent(self.physical)?, extent(self.useful)?))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StagedProjection {
    kind: ProjectionKind,
    layer: usize,
    rows: ProjectionRows,
}

impl Phase {
    fn packed(self, kind: ProjectionKind, index: usize) -> Result<StagedProjection> {
        match self {
            Self::Packing(stage) if stage.kind == kind && stage.layer == index => Ok(stage),
            _ => bail!("ANE prefill packed layer differs from staged layer"),
        }
    }

    fn input(self, kind: ProjectionKind, index: usize) -> Result<StagedProjection> {
        match self {
            Self::Input(stage) if stage.kind == kind && stage.layer == index => Ok(stage),
            _ => bail!("ANE prefill input layer differs from evaluation layer"),
        }
    }

    fn output(self, kind: ProjectionKind, index: usize) -> Result<StagedProjection> {
        match self {
            Self::Output(stage) if stage.kind == kind && stage.layer == index => Ok(stage),
            _ => bail!("ANE prefill output is not staged"),
        }
    }
}

/// An exclusive fixed-shape owner. The ANE runtime makes it neither Send nor Sync.
pub(crate) struct AnePrefill {
    rows: usize,
    program: Option<DynamicProjection>,
    packed_input: Buffer,
    channel_output: Buffer,
    status: Buffer,
    pack: Pipeline,
    qsa_pack: Pipeline,
    shared_layout: (Pipeline, Pipeline),
    unpack: Pipeline,
    host_input: Vec<u16>,
    host_output: Vec<u16>,
    phase: Phase,
    poisoned: bool,
    #[cfg(test)]
    evaluated_layers: Vec<usize>,
    #[cfg(test)]
    evaluated_projections: Vec<(ProjectionKind, usize)>,
}

impl AnePrefill {
    pub(crate) fn load(metal: &Metal, package: &Package, rows: usize) -> Result<Self> {
        ProjectionRows::qsa(rows, rows)?;
        let preparation_start = std::time::Instant::now();
        report_process_memory("before");
        validate_rows(rows)?;
        ensure!(
            cfg!(target_endian = "little"),
            "ANE staging requires a little-endian host"
        );
        let config = QwenConfig::from_package(package)?;
        ensure!(
            config.layers.len() == LAYERS,
            "ANE prefill requires 48 model layers"
        );
        ensure!(
            config
                .layers
                .iter()
                .filter(|kind| **kind == MixerKind::Gdn)
                .count()
                == 36,
            "ANE GPU Q6 prefill requires 36 GDN layers"
        );
        let pack = metal.compile_msl_strict(PACK_SOURCE, "q6_combined_pack")?;
        let qsa_pack = metal.compile_msl_strict(PACK_SOURCE, "q6_qsa_gate_pack")?;
        let unpack = metal.compile_msl_strict(PADDED_UNPACK_SOURCE, "tail_unpack_bf16")?;
        let shared_layout = (
            metal.compile_msl_strict(PACK_SOURCE, "q6_shared_front_pack")?,
            metal.compile_msl_strict(PADDED_UNPACK_SOURCE, "tail_unpack_shared")?,
        );
        for pipeline in [
            &pack,
            &unpack,
            &qsa_pack,
            &shared_layout.0,
            &shared_layout.1,
        ] {
            ensure!(
                pipeline.max_threads_per_threadgroup() >= THREADS,
                "ANE layout threadgroup unsupported"
            );
        }
        let status = metal.shared_buffer(4)?;
        // Sticky for the entire owner lifetime. Resetting between layers could
        // race a previously submitted suffix that still reads this buffer.
        status.write_bytes(0, &0u32.to_le_bytes())?;
        let dynamic_layout = combined_layout(rows)?;
        let shape = dynamic_layout.shape;
        // Allocate fallible staging resources before loading the native program.
        let mut owner = Self {
            rows,
            program: None,
            packed_input: metal.shared_buffer(dynamic_layout.input_elements * 2)?,
            channel_output: metal.shared_buffer(rows * VALUE * 2)?,
            status,
            pack,
            qsa_pack,
            shared_layout,
            unpack,
            host_input: vec![0; dynamic_layout.input_elements],
            host_output: vec![0; rows * VALUE],
            phase: Phase::Idle,
            poisoned: false,
            #[cfg(test)]
            evaluated_layers: Vec::new(),
            #[cfg(test)]
            evaluated_projections: Vec::new(),
        };
        eprintln!("ANE dynamic prefill program compile started");
        let start = std::time::Instant::now();
        owner.program = Some(
            DynamicProjection::compile(shape)
                .context("compile reusable ANE combined projection")?,
        );
        let compile_seconds = start.elapsed().as_secs_f64();
        eprintln!("ANE dynamic prefill program compile completed");
        let host_input_bytes = owner.host_input.len() * 2;
        let host_output_bytes = owner.host_output.len() * 2;
        let gpu_staging_bytes = host_input_bytes + host_output_bytes + 4;
        let ane_io_bytes = host_input_bytes + host_output_bytes;
        eprintln!(
            "ANE GPU Q6 prefill preparation: weight_seconds=0 transpose_seconds=0 compile_seconds={compile_seconds:.6} total_seconds={:.6} programs={} retained_fp16_weight_bytes=0 additional_packed_weight_bytes=0 host_combined_input_bytes={host_input_bytes} host_output_bytes={host_output_bytes} gpu_staging_bytes={gpu_staging_bytes} ane_io_bytes={ane_io_bytes}",
            preparation_start.elapsed().as_secs_f64(),
            usize::from(owner.program.is_some())
        );
        report_process_memory("after");
        Ok(owner)
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn accepts_gdn_rows(&self, useful: usize) -> bool {
        ProjectionRows::gdn(self.rows, useful).is_ok()
    }

    pub(crate) fn accepts_shared_rows(&self, useful: usize) -> bool {
        self.accepts_gdn_rows(useful)
    }

    /// Package identity and chunk width remain separate compatibility inputs.
    pub(crate) fn cache_identity(&self) -> &'static str {
        producer_identity()
    }

    fn healthy(&self) -> Result<()> {
        ensure!(!self.poisoned, "ANE prefill controller is poisoned");
        Ok(())
    }

    /// Submit packing after its producer without delaying independent work.
    pub(crate) fn encode_input(
        &mut self,
        index: usize,
        useful_rows: usize,
        stream: &mut Stream,
        hidden: &Buffer,
        weight: Binding<'_>,
    ) -> Result<Completion> {
        self.encode_input_kind(
            index,
            useful_rows,
            stream,
            hidden,
            [weight; 2],
            ProjectionKind::Gdn,
        )
    }

    pub(crate) fn encode_qsa_input(
        &mut self,
        index: usize,
        useful_rows: usize,
        stream: &mut Stream,
        hidden: &Buffer,
        weight: Binding<'_>,
    ) -> Result<Completion> {
        self.encode_input_kind(
            index,
            useful_rows,
            stream,
            hidden,
            [weight; 2],
            ProjectionKind::QsaGate,
        )
    }

    pub(crate) fn encode_shared_input(
        &mut self,
        index: usize,
        useful_rows: usize,
        stream: &mut Stream,
        hidden: &Buffer,
        weights: [Binding<'_>; 2],
    ) -> Result<Completion> {
        self.encode_input_kind(
            index,
            useful_rows,
            stream,
            hidden,
            weights,
            ProjectionKind::SharedFront,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_input_kind(
        &mut self,
        index: usize,
        useful_rows: usize,
        stream: &mut Stream,
        hidden: &Buffer,
        weights: [Binding<'_>; 2],
        kind: ProjectionKind,
    ) -> Result<Completion> {
        let result = (|| {
            self.healthy()?;
            let rows = if kind == ProjectionKind::QsaGate {
                validate_qsa_layer(index)?;
                ProjectionRows::qsa(self.rows, useful_rows)?
            } else {
                if kind == ProjectionKind::SharedFront {
                    ensure!(index < LAYERS, "shared front is not configured");
                } else {
                    validate_layer(index)?;
                }
                ProjectionRows::gdn(self.rows, useful_rows)?
            };
            ensure!(
                self.phase == Phase::Idle,
                "ANE prefill has unfinished gate staging"
            );
            self.encode_combined(stream.batch()?, hidden, weights, kind, rows)?;
            let completion = stream.flush_marked()?;
            self.phase = Phase::Packing(StagedProjection {
                kind,
                layer: index,
                rows,
            });
            Ok(completion)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Retire the input producer while later independent GPU work stays queued.
    /// The marker also covers the preceding gate consumer before staging reuse.
    pub(crate) fn stage_input(
        &mut self,
        kind: ProjectionKind,
        index: usize,
        stream: &mut Stream,
        completion: Completion,
    ) -> Result<()> {
        let result = (|| {
            self.healthy()?;
            let stage = self.phase.packed(kind, index)?;
            self.read_packed_input(stream, completion)?;
            self.program
                .as_mut()
                .context("ANE combined program is absent")?
                .write_combined_input(&self.host_input)?;
            self.phase = Phase::Input(stage);
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn read_packed_input(&mut self, stream: &mut Stream, completion: Completion) -> Result<()> {
        stream
            .wait_through(completion)
            .context("complete ANE prefill input pack")?;
        self.read_status()?;
        self.packed_input
            .read_bytes(0, word_bytes_mut(&mut self.host_input))?;
        Ok(())
    }

    /// Run after the independent GDN prefix has been submitted asynchronously.
    /// The host publish touches only channel staging, not prefix-owned scratch.
    pub(crate) fn evaluate_and_stage(&mut self, kind: ProjectionKind, index: usize) -> Result<()> {
        let result = (|| {
            self.healthy()?;
            let stage = self.phase.input(kind, index)?;
            let program = self
                .program
                .as_mut()
                .context("ANE combined program is absent")?;
            program.evaluate()?;
            program.read_output_channel_major(&mut self.host_output)?;
            finite_fp16(&self.host_output)?;
            self.channel_output
                .write_bytes(0, word_bytes(&self.host_output))?;
            self.phase = Phase::Output(stage);
            #[cfg(test)]
            if kind != ProjectionKind::SharedFront {
                self.evaluated_layers.push(index);
            }
            #[cfg(test)]
            self.evaluated_projections.push((kind, index));
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Append unpacking before the suffix on the prefix's ordered stream.
    /// No wait is required between the already submitted prefix and this batch.
    pub(crate) fn encode_gate(
        &mut self,
        kind: ProjectionKind,
        index: usize,
        batch: &mut Batch,
        destinations: &[&Buffer],
    ) -> Result<()> {
        let result = (|| {
            self.healthy()?;
            let stage = self.phase.output(kind, index)?;
            let (channels, pipeline) = if kind == ProjectionKind::SharedFront {
                ensure!(
                    destinations.len() == 2,
                    "shared front requires gate/up publication"
                );
                (SHARED * 2, &self.shared_layout.1)
            } else {
                ensure!(
                    destinations.len() == 1,
                    "gate publication requires one buffer"
                );
                (VALUE, &self.unpack)
            };
            self.encode_layout(
                batch,
                &self.channel_output,
                destinations,
                stage.rows,
                channels,
                pipeline,
            )?;
            self.phase = Phase::Idle;
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Call after the final stream wait and before publishing committed state.
    pub(crate) fn check_status_after_completion(&mut self) -> Result<()> {
        let result = (|| {
            self.healthy()?;
            ensure!(
                self.phase == Phase::Idle,
                "ANE prefill ended with unfinished staging"
            );
            self.read_status()
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    #[cfg(test)]
    pub(super) fn test_state(&self) -> (&[usize], bool) {
        (&self.evaluated_layers, self.poisoned)
    }

    #[cfg(test)]
    pub(super) fn test_projection_trace(&self) -> &[(ProjectionKind, usize)] {
        &self.evaluated_projections
    }

    fn read_status(&self) -> Result<()> {
        let mut bytes = [0; 4];
        self.status.read_bytes(0, &mut bytes)?;
        let flags = u32::from_le_bytes(bytes);
        ensure!(
            flags == 0,
            "ANE prefill layout rejected values: flags={flags:#x} (1=nonfinite BF16, 2=FP16 overflow, 4=nonfinite FP16, 8=nonfinite Q6 multiplier, 16=nonfinite Q6 product)"
        );
        Ok(())
    }

    fn encode_combined(
        &self,
        batch: &mut Batch,
        hidden: &Buffer,
        weights: [Binding<'_>; 2],
        kind: ProjectionKind,
        rows: ProjectionRows,
    ) -> Result<()> {
        ensure!(rows.physical == self.rows, "ANE pack physical rows differ");
        let (_, input_bytes) = rows.bytes(HIDDEN)?;
        let qsa = kind == ProjectionKind::QsaGate;
        let shared = kind == ProjectionKind::SharedFront;
        let weight_bytes =
            HIDDEN / 256 * if shared { SHARED } else { VALUE } * 210 * if qsa { 2 } else { 1 };
        let pack = if shared {
            &self.shared_layout.0
        } else if qsa {
            &self.qsa_pack
        } else {
            &self.pack
        };
        ensure!(
            hidden.byte_len() >= input_bytes,
            "ANE combined activation buffer is too small"
        );
        for weight in &weights[..if shared { 2 } else { 1 }] {
            ensure!(
                weight
                    .offset
                    .checked_add(weight_bytes)
                    .is_some_and(|end| end <= weight.buffer.byte_len()),
                "ANE combined Q6_K weight buffer is too small"
            );
        }
        let combined_bytes = self.host_input.len() * 2;
        let dst = self.packed_input.gpu_address(0, combined_bytes)?;
        let src = hidden.gpu_address(0, input_bytes)?;
        ensure!(
            disjoint_gpu_spans(dst, combined_bytes, src, input_bytes)?,
            "ANE combined staging aliases an operand"
        );
        for weight in &weights[..if shared { 2 } else { 1 }] {
            ensure!(
                disjoint_gpu_spans(
                    dst,
                    combined_bytes,
                    weight.buffer.gpu_address(weight.offset, weight_bytes)?,
                    weight_bytes
                )?,
                "ANE combined staging aliases weights"
            );
        }
        let dims: Vec<_> = [
            rows.physical as u32,
            HIDDEN as u32,
            VALUE as u32,
            rows.useful as u32,
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
        let threads =
            (self.rows.div_ceil(TILE) + VALUE.div_ceil(TILE)) * HIDDEN.div_ceil(TILE) * THREADS;
        // SAFETY: Fixed checked dimensions cover the Q6_K pack indexing. The
        // stream retains the immutable weight, activation, staging and status
        // through the pack marker. No host access precedes marker completion.
        let mut bindings = vec![
            Binding {
                buffer: hidden,
                offset: 0,
            },
            weights[0],
            Binding {
                buffer: &self.packed_input,
                offset: 0,
            },
            Binding {
                buffer: &self.status,
                offset: 0,
            },
        ];
        if shared {
            bindings.push(weights[1]);
        }
        unsafe { batch.dispatch_1d_with_inline(pack, &bindings, &[&dims], threads, THREADS) }
    }

    fn encode_layout(
        &self,
        batch: &mut Batch,
        source: &Buffer,
        destinations: &[&Buffer],
        rows: ProjectionRows,
        channels: usize,
        pipeline: &Pipeline,
    ) -> Result<()> {
        ensure!(
            rows.physical == self.rows,
            "ANE unpack physical rows differ"
        );
        ensure!(
            matches!(destinations.len(), 1 | 2) && channels.is_multiple_of(destinations.len()),
            "ANE unpack destinations differ"
        );
        let (source_bytes, _) = rows.bytes(channels)?;
        let (_, destination_bytes) = rows.bytes(channels / destinations.len())?;
        let src = source.gpu_address(0, source_bytes)?;
        for (index, destination) in destinations.iter().enumerate() {
            let dst = destination.gpu_address(0, destination_bytes)?;
            ensure!(
                disjoint_gpu_spans(src, source_bytes, dst, destination_bytes)?,
                "ANE layout allocations alias"
            );
            for previous in &destinations[..index] {
                ensure!(
                    disjoint_gpu_spans(
                        dst,
                        destination_bytes,
                        previous.gpu_address(0, destination_bytes)?,
                        destination_bytes
                    )?,
                    "ANE split destinations alias"
                );
            }
        }
        let dims: Vec<_> = [rows.physical as u32, rows.useful as u32, channels as u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        // SAFETY: Fixed checked dimensions cover guarded shader indexing. The
        // same ordered owner retains distinct input/output and sticky status.
        let bindings: Vec<_> = std::iter::once(source)
            .chain(destinations.iter().copied())
            .chain(std::iter::once(&self.status))
            .map(|buffer| Binding { buffer, offset: 0 })
            .collect();
        unsafe {
            batch.dispatch_1d_with_inline(
                pipeline,
                &bindings,
                &[&dims],
                rows.useful.div_ceil(TILE) * channels.div_ceil(TILE) * THREADS,
                THREADS,
            )
        }
    }

    /// Unload the synchronous ANE program, including a poisoned owner.
    /// Submitted Metal batches retain their own staging-buffer references.
    pub(crate) fn close(mut self) -> Result<()> {
        if let Some(program) = self.program.take() {
            program
                .close()
                .context("ANE dynamic prefill unload failed")?;
        }
        Ok(())
    }
}

fn report_process_memory(stage: &str) {
    match crate::ane::memory_stats() {
        Ok((footprint, resident)) => eprintln!(
            "ANE dynamic prefill memory {stage}: footprint_bytes={footprint} resident_bytes={resident}"
        ),
        Err(error) => eprintln!("ANE dynamic prefill memory {stage}: unavailable: {error}"),
    }
}

fn validate_rows(rows: usize) -> Result<()> {
    ensure!(
        (64..=crate::prefill_capacity::MAX_ROWS).contains(&rows) && rows.is_multiple_of(16),
        "ANE prefill rows must be 64 through {} and divisible by 16",
        crate::prefill_capacity::MAX_ROWS
    );
    Ok(())
}

fn combined_layout(rows: usize) -> Result<DynamicLayout> {
    validate_rows(rows)?;
    Ok(DynamicLayout::new(ProjectionShape::new(
        rows, HIDDEN, VALUE,
    )?)?)
}

fn disjoint_gpu_spans(a: u64, an: usize, b: u64, bn: usize) -> Result<bool> {
    let ae = a
        .checked_add(u64::try_from(an)?)
        .context("ANE GPU span overflow")?;
    let be = b
        .checked_add(u64::try_from(bn)?)
        .context("ANE GPU span overflow")?;
    Ok(ae <= b || be <= a)
}

fn validate_layer(index: usize) -> Result<()> {
    ensure!(
        index < LAYERS && index % 4 != 3,
        "ANE prefill requires a GDN layer index"
    );
    Ok(())
}

fn producer_identity() -> &'static str {
    "ane-gdn-z-qsa-gate-shared-front-q6k-bf16-fp16-reuse-padded-gdn-qsa-v4"
}

#[cfg(test)]
fn validate_padding(values: &[u16], rows: ProjectionRows, channels: usize) -> Result<()> {
    let (bytes, _) = rows.bytes(channels)?;
    ensure!(
        values.len() == bytes / 2,
        "ANE physical output length differs"
    );
    for channel in values.chunks_exact(rows.physical) {
        ensure!(
            channel[rows.useful..]
                .iter()
                .all(|&bits| bits & 0x7fff == 0),
            "ANE padded output is not zero"
        );
    }
    Ok(())
}

fn validate_qsa_layer(index: usize) -> Result<()> {
    ensure!(
        index < LAYERS && index % 4 == 3,
        "ANE prefill requires a QSA layer index"
    );
    Ok(())
}

fn finite_fp16(values: &[u16]) -> Result<()> {
    if let Some(index) = values.iter().position(|&value| value & 0x7c00 == 0x7c00) {
        bail!("ANE prefill output contains nonfinite FP16 at element {index}");
    }
    Ok(())
}

#[cfg(test)]
fn gate_weights_fp16(weight: &DirectWeight) -> Result<Vec<u16>> {
    ensure!(
        weight.format == WeightFormat::Q6K && weight.shape == [VALUE, HIDDEN],
        "ANE GDN gate requires a Q6_K [6144,2560] matrix"
    );
    decode_q6_fp16(&weight.bytes, HIDDEN, VALUE)
}

#[cfg(test)]
fn transpose_weights(source: &[u16], input: usize, output: usize) -> Result<Vec<u16>> {
    ensure!(
        input > 0 && output > 0,
        "invalid ANE weight transpose geometry"
    );
    let elements = input
        .checked_mul(output)
        .context("ANE weight transpose overflow")?;
    ensure!(
        source.len() == elements,
        "ANE weight transpose length differs from geometry"
    );
    let mut result = vec![0; elements];
    for out_start in (0..output).step_by(TILE) {
        for in_start in (0..input).step_by(TILE) {
            for out in out_start..(out_start + TILE).min(output) {
                for inp in in_start..(in_start + TILE).min(input) {
                    result[inp * output + out] = source[out * input + inp];
                }
            }
        }
    }
    Ok(result)
}

// Follow the dense-batch tile loader's Q6_K block layout and BF16 rounding,
// credited to mlx-kquant (MIT) and ggml in dense_batch.metal and the notices.
#[cfg(test)]
fn decode_q6_fp16(packed: &[u8], input: usize, output: usize) -> Result<Vec<u16>> {
    ensure!(
        input > 0 && input.is_multiple_of(256) && output > 0,
        "invalid ANE Q6_K geometry"
    );
    let elements = input
        .checked_mul(output)
        .context("ANE weight shape overflow")?;
    let expected = (elements / 256)
        .checked_mul(210)
        .context("ANE packed weight size overflow")?;
    ensure!(
        packed.len() == expected,
        "ANE Q6_K payload length differs from geometry"
    );
    let mut result = Vec::with_capacity(elements);
    for block in packed.as_chunks::<210>().0 {
        let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
        ensure!(d.is_finite(), "ANE Q6_K block multiplier is nonfinite");
        for sub in 0..256 {
            let half = sub / 128;
            let quarter = sub % 128 / 32;
            let lane = sub % 32;
            let low = block[half * 64 + quarter % 2 * 32 + lane];
            let high = block[128 + half * 32 + lane];
            let code = (((low >> if quarter >= 2 { 4 } else { 0 }) & 15)
                | (((high >> (quarter * 2)) & 3) << 4)) as i32
                - 32;
            let scale = block[192 + sub / 16] as i8;
            let scaled = d * f32::from(scale);
            let value = scaled * code as f32;
            let bits = value.to_bits();
            let rounded = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
            let half = f16::from_f32(f32::from_bits(u32::from(rounded) << 16));
            ensure!(
                half.is_finite(),
                "ANE reconstructed gate weight is not finite FP16"
            );
            result.push(half.to_bits());
        }
    }
    Ok(result)
}

fn word_bytes(words: &[u16]) -> &[u8] {
    // SAFETY: Initialized u16 words are readable and construction checked LE.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}

fn word_bytes_mut(words: &mut [u16]) -> &mut [u8] {
    // SAFETY: Every byte pattern is a u16 and the exclusive slice covers all bytes.
    unsafe {
        std::slice::from_raw_parts_mut(words.as_mut_ptr().cast(), std::mem::size_of_val(words))
    }
}

#[cfg(test)]
mod padded_rows;

#[cfg(test)]
mod shared_front;

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn qsa_4096_physical_rows_accept_only_exact_useful_rows() {
        let rows = ProjectionRows::qsa(4096, 4096)
            .expect("the bounded QSA projection must accept exactly 4096 useful rows");
        assert_eq!(
            rows,
            ProjectionRows {
                physical: 4096,
                useful: 4096,
            }
        );
        for useful in [0, 1, 2048, 4095, 4097, 8192, usize::MAX] {
            assert!(
                ProjectionRows::qsa(4096, useful).is_err(),
                "a 4096-row QSA projection must reject {useful} useful rows"
            );
        }
    }

    #[test]
    fn fixed_8192_combined_layout_and_pack_indices_are_bounded() {
        assert_eq!(MAX_PHYSICAL_ROWS, 8192);
        let layout = combined_layout(MAX_PHYSICAL_ROWS).unwrap();
        assert_eq!(layout.spatial, 14336);
        assert_eq!(layout.input_elements * 2, 70 * 1024 * 1024);
        assert_eq!(
            layout.shape.output_elements().unwrap() * 2,
            96 * 1024 * 1024
        );
        let input_tiles = 8192usize.div_ceil(32);
        let weight_tiles = VALUE.div_ceil(32);
        let channel_tiles = HIDDEN.div_ceil(32);
        assert_eq!(
            (input_tiles + weight_tiles) * channel_tiles * 256,
            9_175_040
        );
        // The activation prefix and transposed weight suffix partition each
        // channel's combined row with no hole, alias or uint overflow.
        for channel in 0..HIDDEN {
            let first = channel * layout.spatial;
            let weight = first + 8192;
            let end = weight + VALUE;
            assert_eq!(end, (channel + 1) * layout.spatial);
            assert!(first < weight && end <= layout.input_elements);
        }
        assert_eq!(layout.input_elements, 36_700_160);
        assert!(layout.input_elements < u32::MAX as usize);
        assert!(8192 * VALUE < u32::MAX as usize);
        let staging_bytes =
            (layout.input_elements + layout.shape.output_elements().unwrap()) as u64 * 2;
        let accounted = crate::memory::ane_prefill_memory(MAX_PHYSICAL_ROWS).unwrap();
        assert_eq!(accounted.host_staging_bytes, staging_bytes);
        assert_eq!(accounted.metal_staging_bytes, staging_bytes + 4);
        assert_eq!(accounted.iosurface_payload_bytes, staging_bytes);
    }

    #[test]
    fn gpu_q6_combined_layout_has_no_retained_fp16_weight_geometry() {
        let layout = combined_layout(4096).unwrap();
        assert_eq!(layout.spatial, 10240);
        assert_eq!(layout.input_elements * 2, 50 * 1024 * 1024);
        assert_eq!(
            layout.shape.output_elements().unwrap() * 2,
            48 * 1024 * 1024
        );
        let staging_bytes =
            (layout.input_elements + layout.shape.output_elements().unwrap()) as u64 * 2;
        let accounted = crate::memory::ane_prefill_memory(4096).unwrap();
        assert_eq!(accounted.host_staging_bytes, staging_bytes);
        assert_eq!(accounted.metal_staging_bytes, staging_bytes + 4);
        assert_eq!(accounted.iosurface_payload_bytes, staging_bytes);
        let large = combined_layout(8192).unwrap();
        assert_eq!(large.spatial, 14336);
        assert_eq!(large.input_elements * 2, 70 * 1024 * 1024);
        assert_eq!(large.shape.output_elements().unwrap() * 2, 96 * 1024 * 1024);
        for bad in [0, 63, 65, 4097, usize::MAX] {
            assert!(combined_layout(bad).is_err());
        }
        assert!(disjoint_gpu_spans(100, 20, 120, 2).unwrap());
        assert!(!disjoint_gpu_spans(100, 20, 119, 2).unwrap());
        assert!(!disjoint_gpu_spans(100, 20, 100, 20).unwrap());
        assert!(disjoint_gpu_spans(u64::MAX, 2, 100, 1).is_err());
    }

    #[test]
    fn numerical_producer_scope_retains_the_combined_padded_identity() {
        assert_eq!(
            format!("{};rows={MAX_PHYSICAL_ROWS}", producer_identity()),
            "ane-gdn-z-qsa-gate-shared-front-q6k-bf16-fp16-reuse-padded-gdn-qsa-v4;rows=8192"
        );
        assert_eq!(
            format!("{};rows=4096", producer_identity()),
            "ane-gdn-z-qsa-gate-shared-front-q6k-bf16-fp16-reuse-padded-gdn-qsa-v4;rows=4096"
        );
        assert_eq!(
            (0..48).filter(|&i| validate_qsa_layer(i).is_ok()).count(),
            12
        );
        for layer in 0..48 {
            assert_ne!(
                validate_layer(layer).is_ok(),
                validate_qsa_layer(layer).is_ok()
            );
        }
        assert!(validate_qsa_layer(48).is_err());
        assert!(validate_qsa_layer(usize::MAX).is_err());
    }

    #[test]
    fn dynamic_weight_transpose_matches_independent_edges_without_value_changes() {
        for (input, output) in [(1, 1), (3, 5), (33, 65), (67, 35)] {
            let source: Vec<_> = (0..input * output)
                .map(|n| ((n * 137 + 17) % 65536) as u16)
                .collect();
            let actual = transpose_weights(&source, input, output).unwrap();
            for i in 0..input {
                for o in 0..output {
                    assert_eq!(actual[i * output + o], source[o * input + i]);
                }
            }
        }
        assert!(transpose_weights(&[], 0, 1).is_err());
        assert!(transpose_weights(&[1], 2, 1).is_err());
        assert!(transpose_weights(&[], usize::MAX, 2).is_err());
    }

    #[test]
    fn input_retirement_leaves_independent_prefix_reserved() -> Result<()> {
        let metal = Metal::new()?;
        let rows = 64;
        let status = metal.shared_buffer(4)?;
        status.write_bytes(0, &0u32.to_le_bytes())?;
        let mut owner = AnePrefill {
            rows,
            program: None,
            packed_input: metal.shared_buffer(combined_layout(rows)?.input_elements * 2)?,
            channel_output: metal.shared_buffer(rows * VALUE * 2)?,
            status,
            pack: metal.compile_msl_strict(PACK_SOURCE, "q6_combined_pack")?,
            qsa_pack: metal.compile_msl_strict(PACK_SOURCE, "q6_qsa_gate_pack")?,
            shared_layout: (
                metal.compile_msl_strict(PACK_SOURCE, "q6_shared_front_pack")?,
                metal.compile_msl_strict(PADDED_UNPACK_SOURCE, "tail_unpack_shared")?,
            ),
            unpack: metal.compile_msl_strict(PADDED_UNPACK_SOURCE, "tail_unpack_bf16")?,
            host_input: vec![0; combined_layout(rows)?.input_elements],
            host_output: vec![0; rows * VALUE],
            phase: Phase::Idle,
            poisoned: false,
            evaluated_layers: Vec::new(),
            evaluated_projections: Vec::new(),
        };
        let hidden = metal.shared_buffer(rows * HIDDEN * 2)?;
        hidden.write_bytes(0, word_bytes(&vec![bf16::ONE.to_bits(); rows * HIDDEN]))?;
        let weights = metal.shared_buffer(HIDDEN / 256 * VALUE * 210)?;
        weights.write_bytes(0, &vec![0; weights.byte_len()])?;
        let independent = metal.shared_buffer(4)?;
        let later = metal.compile_msl_strict(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void later(device uint* out [[buffer(0)]]) { out[0] = 7; }",
            "later",
        )?;
        let mut stream = metal.stream()?;
        let input_ready = owner.encode_input(
            0,
            rows,
            &mut stream,
            &hidden,
            Binding {
                buffer: &weights,
                offset: 0,
            },
        )?;
        // The submitted independent branch must retain its own reservations
        // while the host reads the already completed packed input.
        unsafe {
            stream.batch()?.dispatch_1d(
                &later,
                &[Binding {
                    buffer: &independent,
                    offset: 0,
                }],
                1,
                1,
            )?;
        }
        stream.flush()?;
        // This layout-only fixture has no native owner. Retire and read the
        // same producer boundary used before a real native input write.
        owner.read_packed_input(&mut stream, input_ready)?;
        assert!(independent.read_bytes(0, &mut [0; 4]).is_err());
        for channel in 0..HIDDEN {
            assert!(
                owner.host_input[channel * (rows + VALUE)..channel * (rows + VALUE) + rows]
                    .iter()
                    .all(|&v| v == f16::ONE.to_bits())
            );
        }
        assert!(matches!(owner.phase, Phase::Packing(_)));
        stream.wait()?;
        let mut result = [0; 4];
        independent.read_bytes(0, &mut result)?;
        assert_eq!(u32::from_le_bytes(result), 7);
        assert!(
            owner
                .stage_input(ProjectionKind::Gdn, 0, &mut stream, input_ready)
                .is_err()
        );
        assert!(owner.healthy().is_err());
        Ok(())
    }

    fn packed_fixture(multiplier: f32) -> (Vec<u8>, Vec<u16>) {
        let mut bytes = vec![0; 210];
        let scales: Vec<i8> = (0..16).map(|i| (i * 13 - 95) as i8).collect();
        let multiplier = f16::from_f32(multiplier);
        bytes[208..].copy_from_slice(&multiplier.to_bits().to_le_bytes());
        for (index, scale) in scales.iter().enumerate() {
            bytes[192 + index] = *scale as u8;
        }
        let mut expected = vec![0; 256];
        for group in 0..2 {
            for lane in 0..32 {
                for plane in 0..4 {
                    let index = group * 128 + plane * 32 + lane;
                    let code = ((index * 13 + 7) % 64) as u8;
                    bytes[group * 64 + lane + plane % 2 * 32] |= (code & 15) << (4 * (plane / 2));
                    bytes[128 + group * 32 + lane] |= (code >> 4) << (2 * plane);
                    let exact = f64::from(multiplier.to_f32())
                        * f64::from(scales[index / 16])
                        * f64::from(i32::from(code) - 32);
                    expected[index] =
                        f16::from_f32(bf16::from_f32(exact as f32).to_f32()).to_bits();
                }
            }
        }
        (bytes, expected)
    }

    #[test]
    fn codec_matches_independent_block_fixture_and_output_major_boundaries() {
        let (a, expected_a) = packed_fixture(0.0078125);
        let (b, expected_b) = packed_fixture(-0.0001373291);
        let packed = [a.clone(), b.clone(), b, a].concat();
        let expected = [
            expected_a.clone(),
            expected_b.clone(),
            expected_b,
            expected_a,
        ]
        .concat();
        assert_eq!(decode_q6_fp16(&packed, 512, 2).unwrap(), expected);
    }

    #[test]
    fn codec_rejects_geometry_payload_and_nonfinite_boundaries() {
        let (mut packed, _) = packed_fixture(0.0078125);
        assert!(decode_q6_fp16(&packed, 255, 1).is_err());
        assert!(decode_q6_fp16(&packed, 256, 0).is_err());
        assert!(decode_q6_fp16(&packed[..209], 256, 1).is_err());
        assert!(decode_q6_fp16(&packed, usize::MAX - 255, 256).is_err());
        packed[208..].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
        assert!(decode_q6_fp16(&packed, 256, 1).is_err());
        packed.fill(0);
        packed[192..208].fill(127);
        packed[208..].copy_from_slice(&f16::MAX.to_bits().to_le_bytes());
        assert!(decode_q6_fp16(&packed, 256, 1).is_err());
        let wrong = DirectWeight {
            role: "test".into(),
            format: WeightFormat::Bf16,
            shape: vec![VALUE, HIDDEN],
            bytes: Vec::new(),
        };
        assert!(gate_weights_fp16(&wrong).is_err());
    }

    #[test]
    fn fixed_shape_and_layer_checks_reject_unsupported_dispatches() {
        for rows in [64, 80, 2048, 4096, 8192] {
            validate_rows(rows).unwrap();
        }
        for rows in [0, 1, 63, 65, 4097, 8193, usize::MAX] {
            assert!(validate_rows(rows).is_err());
        }
        assert_eq!((0..48).filter(|&i| validate_layer(i).is_ok()).count(), 36);
        for layer in [3, 7, 47, 48, usize::MAX] {
            assert!(validate_layer(layer).is_err());
        }
        let stage = StagedProjection {
            kind: ProjectionKind::Gdn,
            layer: 2,
            rows: ProjectionRows::full(64).unwrap(),
        };
        assert_eq!(Phase::Input(stage), Phase::Input(stage));
        assert_ne!(Phase::Input(stage), Phase::Output(stage));
    }

    #[test]
    fn finite_check_covers_every_fp16_pattern_and_le_views_preserve_words() {
        for bits in 0..=u16::MAX {
            assert_eq!(
                finite_fp16(&[bits]).is_ok(),
                f16::from_bits(bits).is_finite()
            );
        }
        let values = [0x0001, 0x8000, 0x7bff];
        assert_eq!(word_bytes(&values), [1, 0, 0, 128, 255, 123]);
        let mut output = [0; 3];
        word_bytes_mut(&mut output).copy_from_slice(word_bytes(&values));
        assert_eq!(output, values);
    }

    // CPU copies of the shader integer operations are checked against half's
    // independent value conversions, including subnormal and signed-zero cases.
    fn pack_word(bits: u16) -> Option<u16> {
        let sign = u32::from(bits & 0x8000);
        let exponent = u32::from((bits >> 7) & 255);
        let fraction = u32::from(bits & 127);
        if exponent >= 143 {
            return None;
        }
        if exponent >= 113 {
            return Some((sign | ((exponent - 112) << 10) | (fraction << 3)) as u16);
        }
        if exponent < 102 {
            return Some(sign as u16);
        }
        let mantissa = 128 + fraction;
        if exponent >= 110 {
            return Some((sign | (mantissa << (exponent - 110))) as u16);
        }
        let shift = 110 - exponent;
        let mut rounded = mantissa >> shift;
        let remainder = mantissa & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        rounded += u32::from(remainder > halfway || (remainder == halfway && rounded & 1 != 0));
        Some((sign | rounded) as u16)
    }

    fn unpack_word(bits: u16) -> Option<u16> {
        let sign = u32::from(bits & 0x8000) << 16;
        let exponent = u32::from((bits >> 10) & 31);
        let fraction = u32::from(bits & 1023);
        if exponent == 31 {
            return None;
        }
        let word = if exponent != 0 {
            sign | ((exponent + 112) << 23) | (fraction << 13)
        } else if fraction == 0 {
            sign
        } else {
            let leading = 31 - fraction.leading_zeros();
            sign | ((103 + leading) << 23) | ((fraction - (1 << leading)) << (23 - leading))
        };
        Some((word.wrapping_add(0x7fff + ((word >> 16) & 1)) >> 16) as u16)
    }

    #[test]
    fn layout_integer_conversion_matches_independent_reference_for_all_words() {
        for bits in 0..=u16::MAX {
            let bf = bf16::from_bits(bits).to_f32();
            let fp = f16::from_f32(bf);
            assert_eq!(
                pack_word(bits),
                (bf.is_finite() && fp.is_finite()).then_some(fp.to_bits()),
                "BF16 {bits:#06x}"
            );
            let fp = f16::from_bits(bits);
            assert_eq!(
                unpack_word(bits),
                fp.is_finite()
                    .then(|| bf16::from_f32(fp.to_f32()).to_bits()),
                "FP16 {bits:#06x}"
            );
        }
    }

    #[test]
    fn layout_tiles_match_independent_nonsquare_row_channel_mapping() {
        for (rows, channels) in [(1_usize, 1_usize), (31, 33), (33, 65), (80, 97)] {
            let values: Vec<_> = (0..rows * channels)
                .map(|i| bf16::from_f32((i as i32 % 251 - 125) as f32 / 128.0).to_bits())
                .collect();
            let mut packed = vec![0; values.len()];
            let mut unpacked = vec![0; values.len()];
            for row_base in (0..rows).step_by(32) {
                for channel_base in (0..channels).step_by(32) {
                    let mut tile = [0; 32 * 33];
                    for tid in 0..256 {
                        let x = tid % 32;
                        let y = tid / 32;
                        for offset in (0..32).step_by(8) {
                            let row = row_base + y + offset;
                            let channel = channel_base + x;
                            if row < rows && channel < channels {
                                tile[(y + offset) * 33 + x] =
                                    pack_word(values[row * channels + channel]).unwrap();
                            }
                        }
                    }
                    for tid in 0..256 {
                        let x = tid % 32;
                        let y = tid / 32;
                        for offset in (0..32).step_by(8) {
                            let row = row_base + x;
                            let channel = channel_base + y + offset;
                            if row < rows && channel < channels {
                                packed[channel * rows + row] = tile[x * 33 + y + offset];
                            }
                        }
                    }
                }
            }
            for row in 0..rows {
                for channel in 0..channels {
                    assert_eq!(
                        packed[channel * rows + row],
                        f16::from_f32(bf16::from_bits(values[row * channels + channel]).to_f32())
                            .to_bits()
                    );
                }
            }
            for row_base in (0..rows).step_by(32) {
                for channel_base in (0..channels).step_by(32) {
                    let mut tile = [0; 32 * 33];
                    for tid in 0..256 {
                        let x = tid % 32;
                        let y = tid / 32;
                        for offset in (0..32).step_by(8) {
                            let row = row_base + x;
                            let channel = channel_base + y + offset;
                            if row < rows && channel < channels {
                                tile[(y + offset) * 33 + x] = packed[channel * rows + row];
                            }
                        }
                    }
                    for tid in 0..256 {
                        let x = tid % 32;
                        let y = tid / 32;
                        for offset in (0..32).step_by(8) {
                            let row = row_base + y + offset;
                            let channel = channel_base + x;
                            if row < rows && channel < channels {
                                unpacked[row * channels + channel] =
                                    unpack_word(tile[x * 33 + y + offset]).unwrap();
                            }
                        }
                    }
                }
            }
            for row in 0..rows {
                for channel in 0..channels {
                    assert_eq!(
                        unpacked[row * channels + channel],
                        bf16::from_f32(f16::from_bits(packed[channel * rows + row]).to_f32())
                            .to_bits()
                    );
                }
            }
        }
    }
}

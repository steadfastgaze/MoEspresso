//! Single-owner Qwen execution with one committed frontier for every layer.

use crate::decoder::{FinalProjection, GdnDecoderBlock, GdnLayerState, residual_reader};
use crate::device_status::DeviceStatus;
use crate::embedding::Embedding;
use crate::hc::HcBlock;
use crate::kvarn::{DeviceKvarnState, KvarnEncoder, KvarnState};
use crate::metal::{Batch, Binding, Buffer, MemoryStats, Metal, Pipeline};
use crate::mlp::Qwen4Mlp;
use crate::model::{MixerKind, QwenConfig};
use crate::package::Package;
use crate::ple::{DevicePleState, PleState, Qwen4Ple};
use crate::qsa_attention::QsaAttention;
use crate::qsa_index::{DeviceQsaIndexState, QsaIndex, QsaIndexState};
use crate::qsa_projection::QsaProjector;
use crate::routing::RoutingMode;
use crate::sampling::{MetalSamplingPreparation, Sampler, SamplingOptions};
use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::rc::autoreleasepool;
use std::cell::OnceCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

mod ane_prefill;
mod device_greedy;
mod expert_capacity;
mod prefill;
mod prefix_cache;
mod qsa_gate;
mod snapshot;

pub use expert_capacity::{ExpertCapacityRecovery, ExpertCapacityRelease};
pub use prefix_cache::{PrefixHit, PromptPrefixCache};
pub use snapshot::SessionSnapshot;
#[cfg(target_os = "macos")]
pub use snapshot::{
    DiskCacheMode, DiskCheckpointSlot, DiskCheckpointStore, DiskLimits, IO_BUFFER_BYTES,
    StoreLimits,
};

fn bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect()
}

struct Residual {
    value: Buffer,
    pending: Option<(Buffer, Buffer)>,
}
impl Residual {
    fn pending(&self) -> Option<(&Buffer, &Buffer)> {
        self.pending.as_ref().map(|(o, i)| (o, i))
    }
}

struct GraphScratch {
    input: Buffer,
    materialized: Buffer,
    count: Buffer,
    write: Pipeline,
    status: DeviceStatus,
}

impl GraphScratch {
    fn new(metal: &Metal) -> Result<Self> {
        let count = metal.shared_buffer(4)?;
        count.write_bytes(0, &10240u32.to_le_bytes())?;
        Ok(Self {
            input: metal.shared_buffer(10240 * 2)?,
            materialized: metal.shared_buffer(10240 * 2)?,
            count,
            write: metal.compile_msl_strict(
                include_str!("../kernels/metal/prefill.metal"),
                "prefill_residual_write",
            )?,
            status: DeviceStatus::new(metal)?,
        })
    }

    fn materialize(&self, batch: &mut Batch, residual: &mut Residual) -> Result<()> {
        if let Some((output, injection)) = residual.pending.take() {
            let bindings = [
                &residual.value,
                &output,
                &injection,
                &self.materialized,
                &self.count,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: All decoder outputs have fixed single-row HC geometry;
            // their writers precede this residual composition in the stream.
            unsafe {
                batch.dispatch_1d(&self.write, &bindings, 10240, 256)?;
            }
            residual.value = self.materialized.clone();
        }
        Ok(())
    }
}

struct QsaBlock {
    attention_read: HcBlock,
    projection: QsaProjector,
    index: QsaIndex,
    attention: QsaAttention,
    mlp_read: HcBlock,
    mlp: Qwen4Mlp,
}
struct QsaState {
    kv: DeviceKvarnState,
    index: DeviceQsaIndexState,
}
enum Layer {
    Gdn(Box<GdnDecoderBlock>),
    Qsa(Box<QsaBlock>),
}
enum LayerState {
    Gdn(GdnLayerState),
    Qsa(QsaState),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProducerSpan {
    Prefill {
        width: usize,
    },
    Steps {
        mode: RoutingMode,
        count: usize,
    },
    /// Committed state restored without replaying its historical operations.
    Restored {
        count: usize,
    },
}

impl ProducerSpan {
    fn width(self) -> usize {
        match self {
            Self::Prefill { width } => width,
            Self::Steps { count, .. } | Self::Restored { count } => count,
        }
    }
}
impl LayerState {
    fn frontier(&self) -> Result<usize> {
        match self {
            Self::Gdn(s) => Ok(usize::try_from(s.frontier()?)?),
            Self::Qsa(s) => {
                ensure!(
                    s.index.frontier()? == s.kv.frontier()? as u64,
                    "QSA index and KV frontiers differ"
                );
                s.kv.frontier()
            }
        }
    }
}

/// Failed tokens invalidate the session; partially advanced layers cannot resume.
pub struct Session {
    owner: Rc<()>,
    cache_origin: Rc<()>,
    frontier: usize,
    valid: bool,
    layers: Vec<LayerState>,
    ple: DevicePleState,
    eos_token_id: u32,
    consumed_tokens: Vec<u32>,
    logits_device: Buffer,
    last_logits_bf16: OnceCell<Vec<u16>>,
    producer_spans: Vec<ProducerSpan>,
}

/// Borrowed logical state at a committed layer frontier.
pub enum LayerStateView<'a> {
    Gdn {
        conv_bf16: &'a [u8],
        recurrent_f32: &'a [u8],
    },
    Qsa {
        index: &'a QsaIndexState,
        kv: &'a KvarnState,
    },
}

impl Session {
    pub fn frontier(&self) -> Result<usize> {
        ensure!(self.valid, "session contains an incomplete token");
        Ok(self.frontier)
    }

    fn validate_committed(&self) -> Result<()> {
        let frontier = self.frontier()?;
        ensure!(
            self.consumed_tokens.len() == frontier,
            "consumed token count differs from session frontier"
        );
        let span_width = self.producer_spans.iter().try_fold(0usize, |total, span| {
            ensure!(span.width() > 0, "empty numerical producer span");
            total
                .checked_add(span.width())
                .context("producer span overflow")
        })?;
        ensure!(
            span_width == frontier,
            "numerical producer spans differ from session frontier"
        );
        if frontier == 0 {
            ensure!(
                self.last_logits_bf16.get().is_none(),
                "empty session contains last-row logits"
            );
        } else {
            ensure!(
                self.logits_device.byte_len() == 248_320 * 2,
                "committed device logits have invalid shape"
            );
            if let Some(logits) = self.last_logits_bf16.get() {
                ensure!(
                    logits.len() == 248_320
                        && logits.iter().all(|&v| bf16::from_bits(v).is_finite()),
                    "committed logits have invalid shape or values"
                );
            }
        }
        self.ple
            .validate_committed(&self.consumed_tokens, self.eos_token_id)?;
        ensure!(
            self.ple.offset == frontier as u64,
            "PLE differs from session frontier"
        );
        for layer in &self.layers {
            ensure!(
                layer.frontier()? == frontier,
                "layer differs from session frontier"
            );
        }
        Ok(())
    }

    fn commit_device_output(
        &mut self,
        tokens: &[u32],
        producer: ProducerSpan,
        next: usize,
    ) -> Result<()> {
        ensure!(!self.valid, "cannot recommit a completed session");
        ensure!(
            producer.width() == tokens.len()
                && self.frontier.checked_add(tokens.len()) == Some(next)
                && tokens.iter().all(|&token| token < 248_320),
            "committed producer width differs from token frontier"
        );
        ensure!(
            self.logits_device.byte_len() == 248_320 * 2,
            "committed device logits have invalid shape"
        );
        self.consumed_tokens.extend_from_slice(tokens);
        self.last_logits_bf16.take();
        match (self.producer_spans.last_mut(), producer) {
            (
                Some(ProducerSpan::Steps {
                    mode: previous,
                    count,
                }),
                ProducerSpan::Steps { mode, count: 1 },
            ) if *previous == mode => *count += 1,
            _ => self.producer_spans.push(producer),
        }
        self.frontier = next;
        self.valid = true;
        Ok(())
    }

    #[cfg(test)]
    fn commit_output(
        &mut self,
        tokens: &[u32],
        logits_bf16: &[u16],
        producer: ProducerSpan,
        next: usize,
    ) -> Result<()> {
        ensure!(
            logits_bf16.len() == 248_320
                && logits_bf16.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "committed logits have invalid shape or values"
        );
        self.logits_device.write_bytes(0, &bytes(logits_bf16))?;
        self.commit_device_output(tokens, producer, next)?;
        let _ = self.last_logits_bf16.set(logits_bf16.to_vec());
        Ok(())
    }

    /// Exact token IDs consumed by completed engine calls, excluding emitted
    /// tokens that have not yet entered the model.
    pub fn consumed_tokens(&self) -> Result<&[u32]> {
        self.validate_committed()?;
        Ok(&self.consumed_tokens)
    }

    /// Last-row logits after the most recently consumed token.
    pub fn last_logits_bf16(&self) -> Result<&[u16]> {
        self.validate_committed()?;
        ensure!(self.frontier > 0, "empty session has no last-row logits");
        if self.last_logits_bf16.get().is_none() {
            let row = self.read_logits_bf16_transient()?;
            let _ = self.last_logits_bf16.set(row);
        }
        Ok(self.last_logits_bf16.get().expect("logits materialized"))
    }

    /// Read a completed row for a host sampler without retaining a snapshot row.
    pub(crate) fn read_logits_bf16_transient(&self) -> Result<Vec<u16>> {
        self.frontier()?;
        ensure!(self.frontier > 0, "empty session has no last-row logits");
        let mut row = vec![0u8; 248_320 * 2];
        self.logits_device.read_bytes(0, &mut row)?;
        let words = words(&row);
        ensure!(
            words.iter().all(|&v| bf16::from_bits(v).is_finite()),
            "committed logits have nonfinite values"
        );
        Ok(words)
    }

    /// Read back PLE state without changing the committed device session.
    pub fn ple_state(&self) -> Result<PleState> {
        self.validate_committed()?;
        self.ple.to_host()
    }

    /// Visit each committed layer through a temporary host readback. The
    /// readback is released after its callback and is never retained by Session.
    pub fn visit_state<F>(&self, mut visitor: F) -> Result<()>
    where
        F: for<'a> FnMut(usize, LayerStateView<'a>) -> Result<()>,
    {
        self.validate_committed()?;
        for (index, layer) in self.layers.iter().enumerate() {
            match layer {
                LayerState::Gdn(state) => {
                    let (conv_bf16, recurrent_f32) = state.snapshot()?;
                    visitor(
                        index,
                        LayerStateView::Gdn {
                            conv_bf16: &conv_bf16,
                            recurrent_f32: &recurrent_f32,
                        },
                    )?;
                }
                LayerState::Qsa(state) => {
                    let kv = state.kv.readback()?;
                    let index_state = state.index.export_host()?;
                    visitor(
                        index,
                        LayerStateView::Qsa {
                            index: &index_state,
                            kv: &kv,
                        },
                    )?;
                }
            }
        }
        Ok(())
    }
}
pub struct Engine {
    _expert_row_workspace: crate::package::ExpertRowWorkspace,
    identity: Rc<()>,
    cache_identity: Rc<()>,
    session_started: std::cell::Cell<bool>,
    ane_residency: Option<crate::metal::FixedResidency>,
    ane_residency_failed: bool,
    package_manifest_digest: [u8; 32],
    metal: Metal,
    config: QwenConfig,
    expert_capacity: usize,
    max_context: usize,
    embedding: Embedding,
    layers: Vec<Layer>,
    ple: Qwen4Ple,
    head: FinalProjection,
    kv_encoder: KvarnEncoder,
    prefill: prefill::Prefill,
    graph: GraphScratch,
    greedy: device_greedy::DeviceGreedy,
    sampling_preparation: Option<MetalSamplingPreparation>,
}
pub struct TokenOutput {
    pub logits_bf16: Vec<u16>,
    pub changed_routes: u32,
    pub frontier: usize,
}

pub(crate) struct GenerationStep {
    pub(crate) changed_routes: u32,
    pub(crate) frontier: usize,
    pub(crate) selected_id: Option<u32>,
}

pub(crate) struct GenerationControl<'a> {
    pub(crate) cancelled: &'a mut dyn FnMut() -> bool,
    pub(crate) boundary: &'a mut dyn FnMut(&mut Engine) -> Result<bool>,
    pub(crate) capacity_changed: bool,
}

fn prefill_control_due(
    expert_capacity: usize,
    rows: usize,
    first_layer: bool,
    elapsed: Duration,
) -> bool {
    expert_capacity < 512
        && ((first_layer && rows > crate::prefill_capacity::DEFAULT_ROWS)
            || elapsed >= Duration::from_secs(1))
}

#[test]
fn bounded_prefill_control_covers_slow_narrow_chunks_without_eager_waits() {
    for capacity in [64, 128, 511] {
        for rows in [2, 128, 512] {
            for first_layer in [false, true] {
                assert!(!prefill_control_due(
                    capacity,
                    rows,
                    first_layer,
                    Duration::from_millis(999)
                ));
                assert!(prefill_control_due(
                    capacity,
                    rows,
                    first_layer,
                    Duration::from_secs(1)
                ));
            }
        }
        assert!(prefill_control_due(capacity, 2048, true, Duration::ZERO));
        assert!(!prefill_control_due(
            capacity,
            2048,
            false,
            Duration::from_millis(999)
        ));
        assert!(prefill_control_due(
            capacity,
            2048,
            false,
            Duration::from_secs(1)
        ));
    }
    for rows in [2, 512, 2048] {
        for first_layer in [false, true] {
            assert!(!prefill_control_due(
                512,
                rows,
                first_layer,
                Duration::from_secs(60)
            ));
        }
    }
}

fn layer_routing(layer: usize, requested: RoutingMode) -> RoutingMode {
    // The Qwen Cache-Prior policy preserves original routing in its first
    // two layers, independently of expert residency or quantization format.
    if layer < 2 {
        RoutingMode::Unbiased
    } else {
        requested
    }
}

const EVENT_SERVICE_WINDOW: usize = 4;

fn service_event_demand(layers: &mut [Layer], start: usize) -> Result<u32> {
    let mut changed_routes = 0;
    for (offset, layer) in layers.iter_mut().enumerate() {
        let index = start + offset;
        let mlp = match layer {
            Layer::Gdn(block) => block.mlp_mut(),
            Layer::Qsa(block) => &mut block.mlp,
        };
        changed_routes += mlp
            .service_demand()
            .with_context(|| format!("expert demand at layer {index}"))?;
    }
    Ok(changed_routes)
}

#[test]
fn prefill_is_unbiased_and_decode_keeps_first_two_layers_original() {
    for layer in 0..48 {
        assert_eq!(
            layer_routing(layer, RoutingMode::Unbiased),
            RoutingMode::Unbiased
        );
        assert_eq!(
            layer_routing(layer, RoutingMode::CachePrior),
            if layer < 2 {
                RoutingMode::Unbiased
            } else {
                RoutingMode::CachePrior
            }
        );
    }
}

impl Engine {
    fn new_session_owner(&self) -> Rc<()> {
        self.session_started.set(true);
        self.identity.clone()
    }

    /// Configure combined GDN, QSA gate and shared gate/up offload before any
    /// session or cache attachment. Admission must reserve the ANE allowance.
    /// Initial GPU workspace width selects one retained physical program;
    /// later workspace replacement does not change its geometry or producer.
    /// Setup errors are returned without selecting a different producer.
    pub fn enable_ane(&mut self, package: &Package) -> Result<()> {
        self.validate_ane_startup(package)?;
        if let Err(error) = crate::ane::check(|_| Ok(())) {
            self.ane_residency_failed = !crate::ane::startup_fallback_allowed(&error);
            return Err(error).context("ANE startup runtime check failed");
        }
        let mut pins = Vec::new();
        if self.expert_capacity == 512 {
            for layer in &mut self.layers {
                let experts = match layer {
                    Layer::Gdn(block) => block.experts_mut(),
                    Layer::Qsa(block) => block.mlp.experts_mut(),
                };
                pins.extend(experts.full_resident_weight_buffers()?);
            }
        }
        if let Err(error) = self.prefill.enable_ane(package) {
            self.ane_residency_failed = !crate::ane::startup_fallback_allowed(&error);
            return Err(error);
        }
        if self.expert_capacity == 512 {
            match crate::metal::FixedResidency::new(&self.metal, pins) {
                Ok(owner) => self.ane_residency = Some(owner),
                Err(error) => {
                    return match self.prefill.close_ane() {
                        Ok(()) => Err(error).context("ANE expert residency setup failed"),
                        Err(cleanup) => {
                            self.ane_residency_failed = true;
                            Err(cleanup).with_context(|| {
                                format!("ANE expert residency setup failed: {error:#}; ANE cleanup failed")
                            })
                        }
                    };
                }
            }
        }
        self.identity = Rc::new(());
        self.cache_identity = Rc::new(());
        Ok(())
    }

    /// Prefer ANE during startup, warning and retaining GPU execution only
    /// after a setup failure with completed native cleanup. This does not
    /// recover an engine after session creation or an inference failure.
    pub fn prefer_ane(&mut self, package: &Package) -> Result<bool> {
        self.validate_ane_startup(package)?;
        match self.enable_ane(package) {
            Ok(()) => Ok(true),
            Err(error)
                if !self.ane_residency_failed && crate::ane::startup_fallback_allowed(&error) =>
            {
                eprintln!("⚠ ANE prefill unavailable; using GPU prefill: {error:#}");
                Ok(false)
            }
            Err(error) => Err(error),
        }
    }

    fn validate_ane_startup(&self, package: &Package) -> Result<()> {
        ensure!(
            !self.session_started.get(),
            "ANE configuration requires an engine with no previous sessions"
        );
        ensure!(
            !self.ane_residency_failed,
            "ANE residency requires a fresh engine after failure"
        );
        ensure!(!self.prefill.ane_active(), "ANE is already configured");
        ensure!(
            Rc::strong_count(&self.cache_identity) == 1,
            "ANE configuration must precede cache attachment"
        );
        ensure!(
            package.manifest_digest() == self.package_manifest_digest,
            "ANE package differs from engine package"
        );
        ensure!(
            !self.prefill.workspace_in_progress(),
            "prefill workspace is in progress"
        );
        Ok(())
    }

    fn ane_prefill_cache_scope(&self) -> Option<String> {
        self.prefill.ane_cache_scope()
    }

    /// Whether a healthy combined ANE owner is configured.
    pub fn ane_active(&self) -> bool {
        !self.ane_residency_failed && self.prefill.ane_active()
    }

    /// Consume the engine and report synchronous ANE unload failures.
    pub fn close_ane(mut self) -> Result<()> {
        ensure!(
            !self.prefill.workspace_in_progress(),
            "prefill workspace is in progress"
        );
        drop(self.ane_residency.take());
        self.prefill.close_ane()
    }

    fn finish_ane_call<T>(&mut self, result: Result<T>) -> Result<T> {
        // Inner streams and progress guards have retired before this boundary.
        if result.is_err() && self.prefill.ane_active() {
            drop(self.ane_residency.take());
            self.ane_residency_failed = true;
        }
        result
    }

    fn decode_backing_growth(&self, session: &Session) -> Result<bool> {
        for state in &session.layers {
            if let LayerState::Qsa(state) = state
                && (state.index.next_token_backing_growth()?.is_some()
                    || state.kv.next_token_backing_growth()?.is_some())
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub fn context_limit(&self) -> usize {
        self.max_context
    }

    /// Current physical expert slot capacity in each layer.
    pub fn expert_capacity(&self) -> usize {
        self.expert_capacity
    }

    /// On-demand Metal resource counters for this Engine's device identity.
    pub fn memory_stats(&self) -> Result<MemoryStats> {
        self.metal.memory_stats()
    }

    #[cfg(test)]
    pub(crate) fn test_metal(&self) -> Metal {
        self.metal.clone()
    }

    #[cfg(test)]
    pub(crate) fn test_prefill_workspace_in_progress(&self) -> bool {
        self.prefill.workspace_in_progress()
    }

    pub(crate) fn prefill_workspace_ready(&self) -> bool {
        self.prefill.workspace_ready()
    }

    pub(crate) fn replace_prefill_without_overlap(
        &mut self,
        capacity: usize,
        workspace_budget_bytes: u64,
    ) -> Result<(u64, u64)> {
        autoreleasepool(|_| {
            self.prefill.replace_workspace_without_overlap(
                self.max_context,
                capacity,
                workspace_budget_bytes,
            )
        })
    }

    /// Largest prompt chunk supported by bounded and full expert residency.
    pub fn prefill_chunk_size(&self) -> usize {
        self.prefill.capacity()
    }

    /// Published logical expert IDs, in source order for each layer.
    pub fn resident_experts(&self) -> Result<Vec<Vec<u32>>> {
        self.layers
            .iter()
            .map(|layer| {
                let map = match layer {
                    Layer::Gdn(block) => block.residency_map()?,
                    Layer::Qsa(block) => block.mlp.residency_map()?,
                };
                Ok(map
                    .iter()
                    .enumerate()
                    .filter_map(|(id, &slot)| (slot < 512).then_some(id as u32))
                    .collect())
            })
            .collect()
    }

    pub fn load(package: &Package, expert_capacity: usize, max_context: usize) -> Result<Self> {
        Self::load_with_prefill_capacity(
            package,
            expert_capacity,
            max_context,
            crate::prefill_capacity::DEFAULT_ROWS,
        )
    }

    pub fn load_with_prefill_capacity(
        package: &Package,
        expert_capacity: usize,
        max_context: usize,
        prefill_capacity: usize,
    ) -> Result<Self> {
        crate::prefill_capacity::validate_capacity(prefill_capacity)?;
        autoreleasepool(|_| {
            Self::load_inner(package, expert_capacity, max_context, prefill_capacity)
        })
    }

    fn load_inner(
        package: &Package,
        expert_capacity: usize,
        max_context: usize,
        prefill_capacity: usize,
    ) -> Result<Self> {
        ensure!(
            (10..=512).contains(&expert_capacity),
            "expert capacity must be between 10 and 512"
        );
        let config = QwenConfig::from_package(package)?;
        ensure!(
            max_context > 0 && max_context <= config.max_positions,
            "context limit exceeds model contract"
        );
        if expert_capacity < 512 {
            eprintln!("expert storage: mode=chunked capacity={expert_capacity}");
        }
        let expert_layers = (0..config.layers.len())
            .map(|layer| package.expert_layer(layer))
            .collect::<Result<Vec<_>>>()?;
        let largest_row = expert_layers
            .iter()
            .map(|layer| layer.row_bytes)
            .max()
            .context("engine has no expert layers")?;
        let expert_row_workspace = if expert_capacity < 512 {
            crate::package::ExpertRowWorkspace::with_prefill_reader(largest_row)?
        } else {
            crate::package::ExpertRowWorkspace::new(largest_row)?
        };
        let metal = Metal::for_engine(expert_capacity)?;
        let embedding = Embedding::new(package.direct_weight("embedding")?)?;
        let mut layers = Vec::with_capacity(config.layers.len());
        for ((index, kind), expert_layer) in config.layers.iter().enumerate().zip(expert_layers) {
            layers.push(match kind {
                MixerKind::Gdn => Layer::Gdn(Box::new(GdnDecoderBlock::with_expert_workspace(
                    &metal,
                    package,
                    index,
                    expert_capacity,
                    config.eps,
                    expert_layer,
                    expert_row_workspace.clone(),
                )?)),
                MixerKind::Qsa => {
                    let prefix = format!("layers.{index}");
                    Layer::Qsa(Box::new(QsaBlock {
                        attention_read: residual_reader(
                            &metal,
                            package,
                            &format!("{prefix}.attention_residual"),
                            config.eps,
                        )?,
                        projection: QsaProjector::new(&metal, package, index)?,
                        index: QsaIndex::new(&metal, package, index)?,
                        attention: QsaAttention::new(&metal, package, index)?,
                        mlp_read: residual_reader(
                            &metal,
                            package,
                            &format!("{prefix}.mlp_residual"),
                            config.eps,
                        )?,
                        mlp: Qwen4Mlp::with_expert_workspace(
                            &metal,
                            package,
                            index,
                            expert_capacity,
                            expert_layer,
                            expert_row_workspace.clone(),
                        )?,
                    }))
                }
            });
        }
        if expert_capacity == 512 {
            let sources: Vec<u32> = (0..512).collect();
            for layer in &mut layers {
                match layer {
                    Layer::Gdn(block) => block.seed_residents(&sources)?,
                    Layer::Qsa(block) => block.mlp.seed_residents(&sources)?,
                }
            }
        } else if let Some(hotlist) = package.expert_hotlist()? {
            for (index, layer) in layers.iter_mut().enumerate() {
                if let Some(counts) = hotlist.layers.get(&index) {
                    match layer {
                        Layer::Gdn(block) => block.seed_hotlist(counts)?,
                        Layer::Qsa(block) => block.mlp.seed_hotlist(counts)?,
                    }
                }
            }
        }
        let ple = Qwen4Ple::new(&metal, package)?;
        let head = FinalProjection::new(&metal, package, config.eps)?;
        let kv_encoder = KvarnEncoder::new(&metal)?;
        let prefill = prefill::Prefill::new(
            &metal,
            package,
            &config,
            &layers,
            &ple,
            max_context,
            prefill_capacity,
        )?;
        let graph = GraphScratch::new(&metal)?;
        let greedy = device_greedy::DeviceGreedy::new(&metal, 248_320)?;
        ensure!(
            greedy.requested_bytes() as u64 == crate::memory::DEVICE_GREEDY_REQUESTED_BYTES
                && greedy.host_bytes() as u64 == crate::memory::DEVICE_GREEDY_HOST_BYTES,
            "greedy sampling workspace differs from admission inventory"
        );
        Ok(Self {
            identity: Rc::new(()),
            cache_identity: Rc::new(()),
            session_started: std::cell::Cell::new(false),
            ane_residency: None,
            ane_residency_failed: false,
            _expert_row_workspace: expert_row_workspace,
            package_manifest_digest: package.manifest_digest(),
            metal,
            config,
            expert_capacity,
            max_context,
            embedding,
            layers,
            ple,
            head,
            kv_encoder,
            prefill,
            graph,
            greedy,
            sampling_preparation: None,
        })
    }

    pub fn start_session(&self) -> Result<Session> {
        ensure!(
            !self.ane_residency_failed,
            "ANE residency requires a fresh engine after failure"
        );
        autoreleasepool(|_| self.start_session_inner())
    }

    fn start_session_inner(&self) -> Result<Session> {
        let layers = self
            .config
            .layers
            .iter()
            .map(|kind| {
                Ok(match kind {
                    MixerKind::Gdn => LayerState::Gdn(GdnLayerState::zero(&self.metal)?),
                    MixerKind::Qsa => LayerState::Qsa(QsaState {
                        kv: DeviceKvarnState::zero(&self.metal, self.max_context)?,
                        index: DeviceQsaIndexState::zero(&self.metal, self.max_context)?,
                    }),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Session {
            owner: self.new_session_owner(),
            cache_origin: self.cache_identity.clone(),
            frontier: 0,
            valid: true,
            layers,
            ple: DevicePleState::zero(&self.metal)?,
            eos_token_id: self.config.eos_token_id,
            consumed_tokens: Vec::new(),
            logits_device: self.metal.shared_buffer(248_320 * 2)?,
            last_logits_bf16: OnceCell::new(),
            producer_spans: Vec::new(),
        })
    }

    /// Consume one unbiased, layer-major prompt chunk on the decode session.
    /// Chunk boundaries retain their batch arithmetic and expert demand order.
    pub fn prefill_chunk(&mut self, tokens: &[u32], session: &mut Session) -> Result<TokenOutput> {
        // Model/state resources retain their own owners. Temporary Objective-C
        // objects can drain after this chunk's GPU work has completed.
        let result = autoreleasepool(|_| self.prefill_chunk_inner(tokens, session, None, None))
            .and_then(|output| {
                let output = output.context("uncancellable prefill did not complete")?;
                Ok(TokenOutput {
                    logits_bf16: session.last_logits_bf16()?.to_vec(),
                    changed_routes: output.changed_routes,
                    frontier: output.frontier,
                })
            });
        self.finish_ane_call(result)
    }

    fn prefill_chunk_inner(
        &mut self,
        tokens: &[u32],
        session: &mut Session,
        greedy: Option<(&[u32], SamplingOptions)>,
        mut control: Option<&mut GenerationControl<'_>>,
    ) -> Result<Option<GenerationStep>> {
        ensure!(
            !self.ane_residency_failed,
            "ANE residency requires a fresh engine after failure"
        );
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.prefill_chunk_size(),
            "prefill chunk exceeds the Engine workspace"
        );
        if tokens.len() == 1 {
            return self
                .step_inner(tokens[0], RoutingMode::Unbiased, session, greedy)
                .map(Some);
        }
        ensure!(
            Rc::ptr_eq(&self.identity, &session.owner),
            "session belongs to another engine"
        );
        let frontier = session.frontier()?;
        let next = frontier
            .checked_add(tokens.len())
            .context("prefill frontier overflow")?;
        ensure!(next <= self.max_context, "prefill exceeds context limit");
        ensure!(
            tokens.iter().all(|&id| id < 248320),
            "prefill token exceeds vocabulary"
        );
        ensure!(
            session.ple.offset == frontier as u64,
            "PLE differs from session frontier"
        );
        for state in &session.layers {
            ensure!(
                state.frontier()? == frontier,
                "layer differs from session frontier"
            );
        }
        session.valid = false;
        let mut last_control = Instant::now();
        let mut progress =
            self.prefill
                .begin(tokens, frontier, &self.embedding, &self.graph.status)?;
        // Bounded chunks of any width can take several seconds. Check owner
        // control at layer boundaries about once per second, with an eager
        // first-layer check for wide chunks. Fast narrow chunks and full
        // residency retain their existing submission schedule.
        for index in 0..self.layers.len() {
            self.prefill.encode_layer(
                index,
                tokens,
                session,
                &mut self.layers[index],
                &self.kv_encoder,
                &self.graph.status,
                &mut progress,
            )?;
            if let Some(control) = control.as_deref_mut() {
                if (control.cancelled)() {
                    progress.stream.abort_and_wait()?;
                    return Ok(None);
                }
                if prefill_control_due(
                    self.expert_capacity,
                    tokens.len(),
                    index == 0,
                    last_control.elapsed(),
                ) {
                    progress.stream.wait()?;
                    self.graph.status.check()?;
                    control.capacity_changed |= (control.boundary)(self)?;
                    last_control = Instant::now();
                    if (control.cancelled)() {
                        return Ok(None);
                    }
                }
            }
        }
        self.prefill.finish(
            session,
            &self.head,
            &self.graph.status,
            greedy.map(|(history, options)| (&mut self.greedy, history, options)),
            &mut progress,
        )?;
        let selected_id = if greedy.is_some() {
            Some(self.greedy.read_id()?)
        } else {
            None
        };
        for state in &session.layers {
            ensure!(
                state.frontier()? == next,
                "prefill layer did not advance to next frontier"
            );
        }
        ensure!(
            session.ple.offset == next as u64,
            "prefill PLE did not advance to next frontier"
        );
        session.commit_device_output(
            tokens,
            ProducerSpan::Prefill {
                width: tokens.len(),
            },
            next,
        )?;
        Ok(Some(GenerationStep {
            changed_routes: 0,
            frontier: next,
            selected_id,
        }))
    }

    /// Consume one physical token. Use Unbiased for every prefill token.
    pub fn step(
        &mut self,
        token: u32,
        mode: RoutingMode,
        session: &mut Session,
    ) -> Result<TokenOutput> {
        let result =
            autoreleasepool(|_| self.step_inner(token, mode, session, None)).and_then(|output| {
                Ok(TokenOutput {
                    logits_bf16: session.last_logits_bf16()?.to_vec(),
                    changed_routes: output.changed_routes,
                    frontier: output.frontier,
                })
            });
        self.finish_ane_call(result)
    }

    fn step_inner(
        &mut self,
        token: u32,
        mode: RoutingMode,
        session: &mut Session,
        greedy: Option<(&[u32], SamplingOptions)>,
    ) -> Result<GenerationStep> {
        ensure!(
            !self.ane_residency_failed,
            "ANE residency requires a fresh engine after failure"
        );
        ensure!(
            Rc::ptr_eq(&self.identity, &session.owner),
            "session belongs to another engine"
        );
        let frontier = session.frontier()?;
        ensure!(frontier < self.max_context, "session reached context limit");
        for state in &session.layers {
            ensure!(
                state.frontier()? == frontier,
                "layer differs from session frontier"
            );
        }
        ensure!(
            session.ple.offset == frontier as u64,
            "PLE differs from session frontier"
        );
        // Bounded admission reserves one old QSA/KVarN backing pair during
        // growth. Retire each layer on those tokens; ordinary tokens can queue
        // every layer without retaining replaced cache backings.
        let event_decode = self.expert_capacity < 512 && !self.decode_backing_growth(session)?;
        let layer_count = self.layers.len();
        ensure!(
            layer_count > 0 && layer_count == session.layers.len(),
            "decoder and session layer counts differ"
        );
        let service_window = if event_decode {
            EVENT_SERVICE_WINDOW.min(layer_count)
        } else {
            layer_count
        };
        let mut embedding = vec![0u16; 2560];
        self.embedding.lookup_into(&[token], &mut embedding)?;
        self.graph.status.reset()?;
        self.graph
            .input
            .write_bytes(0, &bytes(&embedding).repeat(4))?;
        let mut residual = Residual {
            value: self.graph.input.clone(),
            pending: None,
        };
        session.valid = false;
        let mut stream = self.metal.stream()?;
        let mut ple_executed = false;
        let mut changed_routes = 0;
        let mut enqueued_layers = 0usize;
        let mut serviced_layers = 0usize;
        for (chunk, (layer_window, state_window)) in self
            .layers
            .chunks_mut(service_window)
            .zip(session.layers.chunks_mut(service_window))
            .enumerate()
        {
            let start = chunk * service_window;
            for (offset, (layer, state)) in layer_window
                .iter_mut()
                .zip(state_window.iter_mut())
                .enumerate()
            {
                let index = start + offset;
                let mode = layer_routing(index, mode);
                if index == self.config.ple_layer_index {
                    self.graph.materialize(stream.batch()?, &mut residual)?;
                    residual.value = self.ple.encode_step_device(
                        &mut stream,
                        &mut session.ple,
                        token,
                        &residual.value,
                        &self.graph.status,
                    )?;
                    ple_executed = true;
                }
                residual = match (layer, state) {
                    (Layer::Gdn(block), LayerState::Gdn(state)) => {
                        let output = if event_decode {
                            block.run_device_schedule(
                                &mut stream,
                                &residual.value,
                                residual.pending(),
                                mode,
                                state,
                                &self.graph.status,
                                true,
                            )
                        } else {
                            block.run_device(
                                &mut stream,
                                &residual.value,
                                residual.pending(),
                                mode,
                                state,
                                &self.graph.status,
                            )
                        }
                        .with_context(|| format!("GDN layer {index}"))?;
                        changed_routes += output.changed_routes;
                        Residual {
                            value: output.residual,
                            pending: Some((output.pending_output, output.pending_injection)),
                        }
                    }
                    (Layer::Qsa(block), LayerState::Qsa(state)) => {
                        let batch = stream.batch()?;
                        // SAFETY: Device residual dependencies precede this layer
                        // in the single ordered stream. HC checks fixed extents.
                        let read = unsafe {
                            block.attention_read.encode_device(
                                batch,
                                &residual.value,
                                residual.pending(),
                            )?
                        };
                        block
                            .projection
                            .encode_device(batch, &read.mixed, [frontier as i64; 3])?;
                        block
                            .projection
                            .encode_validation(batch, &self.graph.status)?;
                        let (raw_index_key, index_queries) = block.projection.index_buffers();
                        let selected = block.index.encode_step_device(
                            batch,
                            &mut state.index,
                            raw_index_key,
                            index_queries,
                            &self.graph.status,
                        )?;
                        let (queries, gate, keys, values) = block.projection.attention_buffers();
                        let attended = block.attention.encode_pending_online_device(
                            batch,
                            &state.kv,
                            &selected.ids,
                            &selected.valid,
                            &selected.count,
                            &selected.status,
                            queries,
                            gate,
                            keys,
                            values,
                            &self.graph.status,
                        )?;
                        state
                            .kv
                            .encode_append_token(batch, &self.kv_encoder, keys, values)?;
                        let mlp_read = unsafe {
                            block.mlp_read.encode_device(
                                batch,
                                &read.updated,
                                Some((
                                    &attended,
                                    read.injection.as_ref().context("QSA injection missing")?,
                                )),
                            )?
                        };
                        self.graph.status.encode_bf16(batch, &mlp_read.updated)?;
                        let output = if event_decode {
                            block.mlp.enqueue_device(
                                &mut stream,
                                &mlp_read.mixed,
                                mode,
                                &self.graph.status,
                            )?
                        } else {
                            block.mlp.run_device(
                                &mut stream,
                                &mlp_read.mixed,
                                mode,
                                &self.graph.status,
                            )?
                        };
                        changed_routes += output.changed_routes;
                        Residual {
                            value: mlp_read.updated,
                            pending: Some((
                                output.output,
                                mlp_read.injection.context("MLP injection missing")?,
                            )),
                        }
                    }
                    _ => anyhow::bail!("decoder and session layer types differ"),
                };
                if index == 1 && self.expert_capacity == 512 {
                    stream.flush()?;
                }
                if event_decode {
                    enqueued_layers += 1;
                }
            }
            if event_decode && service_window < layer_count {
                changed_routes += service_event_demand(layer_window, start)?;
                serviced_layers += layer_window.len();
            }
        }
        self.head.encode_device(
            stream.batch()?,
            &residual.value,
            residual.pending(),
            &self.graph.status,
            &session.logits_device,
        )?;
        if let Some((history, options)) = greedy {
            self.greedy.encode(
                stream.batch()?,
                &session.logits_device,
                history,
                options,
                &self.graph.status,
            )?;
        }
        if event_decode {
            if service_window == layer_count {
                changed_routes += service_event_demand(&mut self.layers, 0)?;
                serviced_layers += layer_count;
            }
            ensure!(
                serviced_layers == enqueued_layers && enqueued_layers == layer_count,
                "event decode left an expert demand unserviced"
            );
        }
        stream.wait().context("decode graph completion")?;
        self.graph.status.check()?;
        let selected_id = if greedy.is_some() {
            Some(self.greedy.read_id()?)
        } else {
            None
        };
        let next = frontier
            .checked_add(1)
            .context("session frontier overflow")?;
        for state in &mut session.layers {
            match state {
                LayerState::Gdn(state) => {
                    state.publish_device_after_wait()?;
                }
                LayerState::Qsa(state) => {
                    state.index.publish_after_wait()?;
                    state.kv.publish_after_wait()?;
                }
            }
        }
        for state in &session.layers {
            ensure!(
                state.frontier()? == next,
                "layer did not advance to next token"
            );
        }
        ensure!(ple_executed, "model did not execute declared PLE layer");
        ensure!(
            session.ple.offset == next as u64,
            "PLE did not advance to next token"
        );
        session.commit_device_output(&[token], ProducerSpan::Steps { mode, count: 1 }, next)?;
        Ok(GenerationStep {
            changed_routes,
            frontier: next,
            selected_id,
        })
    }

    /// Consume a generation chunk without materializing diagnostic logits.
    /// The final head and greedy selection share the ordered graph stream.
    pub(crate) fn consume_for_generation(
        &mut self,
        tokens: &[u32],
        mode: RoutingMode,
        session: &mut Session,
        history: &[u32],
        sampler: Option<&mut Sampler>,
        control: Option<&mut GenerationControl<'_>>,
    ) -> Result<Option<GenerationStep>> {
        let greedy = sampler
            .as_ref()
            .map(|sampler| sampler.options())
            .filter(|options| options.temperature == 0.0)
            .map(|options| (history, options));
        let result = autoreleasepool(|_| {
            if tokens.len() == 1 {
                self.step_inner(tokens[0], mode, session, greedy).map(Some)
            } else {
                ensure!(mode == RoutingMode::Unbiased, "decode requires one token");
                self.prefill_chunk_inner(tokens, session, greedy, control)
            }
        })
        .and_then(|mut output| {
            let Some(mut output) = output.take() else {
                return Ok(None);
            };
            if let Some(sampler) = sampler
                && output.selected_id.is_none()
            {
                output.selected_id = Some(self.sample_committed(session, history, sampler)?);
            }
            Ok(Some(output))
        });
        self.finish_ane_call(result)
    }

    /// Select from a restored or just-committed row without consuming a token.
    pub(crate) fn sample_committed(
        &mut self,
        session: &Session,
        history: &[u32],
        sampler: &mut Sampler,
    ) -> Result<u32> {
        ensure!(
            !self.ane_residency_failed,
            "ANE residency requires a fresh engine after failure"
        );
        ensure!(
            Rc::ptr_eq(&self.identity, &session.owner),
            "session belongs to another engine"
        );
        ensure!(
            session.frontier()? > 0,
            "empty session has no logits to sample"
        );
        let result = autoreleasepool(|_| {
            if sampler.options().temperature == 0.0 {
                self.graph.status.reset()?;
                let mut stream = self.metal.stream()?;
                self.greedy.encode(
                    stream.batch()?,
                    &session.logits_device,
                    history,
                    sampler.options(),
                    &self.graph.status,
                )?;
                stream.wait().context("greedy sampling completion")?;
                self.graph.status.check()?;
                self.greedy.read_id()
            } else {
                let row = session.read_logits_bf16_transient()?;
                if self.sampling_preparation.is_none() {
                    self.sampling_preparation =
                        Some(MetalSamplingPreparation::new(&self.metal, row.len())?);
                }
                sampler.sample_id(
                    self.sampling_preparation
                        .as_mut()
                        .expect("sampling preparation initialized"),
                    &row,
                    history,
                )
            }
        });
        self.finish_ane_call(result)
    }
}

#[cfg(test)]
mod failure_recovery_tests {
    use super::*;
    use crate::generation::{
        GenerationCaches, GenerationOptions, GenerationStats, generate_with_cache,
    };
    use crate::metal::TestFaultGuard;
    use crate::sampling::SamplingOptions;
    use crate::sampling::{MetalSamplingPreparation, Sampler};
    use crate::tokenizer::TextTokenizer;

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE, private ANE support and exclusive GPU ownership"]
    fn combined_expert_residency_rejects_expert_changes_and_releases_after_command_failure()
    -> Result<()> {
        expert_residency_failure_case()
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE, private ANE support and exclusive GPU ownership"]
    fn combined_qsa_evaluation_precedes_command_failure_and_residency_release() -> Result<()> {
        post_qsa_command_failure_case(4096)
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE, private ANE support and exclusive GPU ownership"]
    fn padded_qsa_evaluation_precedes_command_failure_and_residency_release() -> Result<()> {
        post_qsa_command_failure_case(4097)
    }

    fn post_qsa_command_failure_case(useful: usize) -> Result<()> {
        use objc2_metal::MTLResidencySet;

        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let mut engine = Engine::load_with_prefill_capacity(&package, 512, 8192, 8192)?;
        engine.enable_ane(&package)?;
        let metal = engine.metal.clone();
        let native = engine.ane_residency.as_ref().unwrap().test_set();
        assert_eq!(native.allocationCount(), 588);
        assert_eq!(engine.prefill.test_ane_state(), Some((&[][..], false)));
        let mut session = engine.start_session()?;
        let tokens: Vec<u32> = (0..useful).map(|i| 61 + (i % 32) as u32).collect();

        // Each mixer and shared front submits pack/prefix. The layer1 flush
        // adds one command. Completion17 is shared3's prefix, retired by the
        // next layer's pack wait after QSA3 and shared3 native evaluations.
        let fault = metal.test_fail_completed_command_after(16)?;
        let error = engine
            .prefill_chunk(&tokens, &mut session)
            .err()
            .context("post-QSA command injection unexpectedly succeeded")?;
        assert!(fault.fired());
        assert!(format!("{error:#}").contains("injected completed Metal command error"));
        drop(fault);
        assert_eq!(
            engine.prefill.test_ane_state(),
            Some((&[0, 1, 2, 3][..], true))
        );
        use ane_prefill::ProjectionKind::{Gdn, QsaGate, SharedFront};
        assert_eq!(
            engine.prefill.test_projection_trace(),
            &[
                (Gdn, 0),
                (SharedFront, 0),
                (Gdn, 1),
                (SharedFront, 1),
                (Gdn, 2),
                (SharedFront, 2),
                (QsaGate, 3),
                (SharedFront, 3),
            ]
        );
        assert!(!session.valid);
        assert!(session.frontier().is_err());
        assert_eq!(session.frontier, 0);
        assert!(session.consumed_tokens.is_empty());
        assert!(engine.ane_residency.is_none());
        assert!(engine.ane_residency_failed);
        assert_eq!(native.allocationCount(), 0);
        assert!(!engine.prefill.workspace_in_progress());
        engine._expert_row_workspace.check_idle()?;
        // Host writes fail while any submitted reader still owns this status.
        engine.graph.status.reset()?;
        let failed_bytes = engine.memory_stats()?.live_requested_buffer_bytes;
        assert!(engine.start_session().is_err());
        assert!(engine.prefill_chunk(&[7, 8], &mut session).is_err());
        assert!(engine.step(7, RoutingMode::Unbiased, &mut session).is_err());
        assert_eq!(
            engine.memory_stats()?.live_requested_buffer_bytes,
            failed_bytes
        );
        assert_eq!(
            engine.prefill.test_ane_state(),
            Some((&[0, 1, 2, 3][..], true))
        );
        drop(session);
        engine.close_ane()?;
        assert_eq!(native.allocationCount(), 0);
        assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, 0);
        Ok(())
    }

    fn expert_residency_failure_case() -> Result<()> {
        use objc2_metal::MTLResidencySet;

        fn state(session: &Session) -> Result<Vec<Vec<u8>>> {
            let ple = session.ple_state()?;
            let mut values = vec![
                session.frontier()?.to_le_bytes().to_vec(),
                ple.offset.to_le_bytes().to_vec(),
                ple.context.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ple.conv.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ];
            session.visit_state(|_, layer| {
                match layer {
                    LayerStateView::Gdn {
                        conv_bf16,
                        recurrent_f32,
                    } => values.extend([conv_bf16.to_vec(), recurrent_f32.to_vec()]),
                    LayerStateView::Qsa { index, kv } => values.extend([
                        index.frontier().to_le_bytes().to_vec(),
                        index
                            .compressed_bf16()
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                        index
                            .pending_bf16()
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                        kv.frontier().to_le_bytes().to_vec(),
                        kv.records().to_vec(),
                        kv.sink_keys().to_vec(),
                        kv.sink_values().to_vec(),
                        kv.tail_keys().to_vec(),
                        kv.tail_values().to_vec(),
                    ]),
                }
                Ok(())
            })?;
            Ok(values)
        }

        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let mut engine = Engine::load_with_prefill_capacity(&package, 512, 8192, 4096)?;
        engine.enable_ane(&package)?;
        let scope = engine
            .ane_prefill_cache_scope()
            .context("active ANE producer scope absent")?;
        assert!(scope.ends_with(";rows=4096"));
        let metal = engine.metal.clone();
        assert!(engine.ane_residency.is_some());
        let native = engine.ane_residency.as_ref().unwrap().test_set();
        assert_eq!(native.allocationCount(), 588);
        let mut session = engine.start_session()?;
        let before_state = state(&session)?;
        let before_routes = engine.resident_experts()?;
        let before_bytes = engine.memory_stats()?.live_requested_buffer_bytes;
        for error in [
            engine.shrink_expert_capacity(496).unwrap_err(),
            engine.recover_expert_capacity(513, u64::MAX).unwrap_err(),
        ] {
            assert!(error.to_string().contains("fixed ANE residency forbids"));
        }
        assert_eq!(engine.recover_expert_capacity(512, 0)?.new_capacity, 512);
        assert_eq!(engine.shrink_expert_capacity(512)?.new_capacity, 512);
        // Scratch is not part of the expert set and can be replaced at an
        // ordinary retired boundary without retaining its previous allocation.
        engine.replace_prefill_without_overlap(2048, u64::MAX)?;
        assert_eq!(engine.prefill_chunk_size(), 2048);
        assert_eq!(
            engine.ane_prefill_cache_scope().as_deref(),
            Some(scope.as_str())
        );
        assert_eq!(native.allocationCount(), 588);
        engine.replace_prefill_without_overlap(4096, u64::MAX)?;
        assert_eq!(
            engine.ane_prefill_cache_scope().as_deref(),
            Some(scope.as_str())
        );
        assert_eq!(
            (engine.expert_capacity(), engine.prefill_chunk_size()),
            (512, 4096)
        );
        assert_eq!(engine.resident_experts()?, before_routes);
        assert_eq!(state(&session)?, before_state);
        assert_eq!(
            engine.memory_stats()?.live_requested_buffer_bytes,
            before_bytes
        );
        let mut sampler = Sampler::new(SamplingOptions::default(), 0)?;
        assert!(
            engine
                .sample_committed(&session, &[], &mut sampler)
                .is_err()
        );
        assert!(engine.ane_residency.is_some());

        // A seven-row GPU tail fails before any fixed-shape ANE evaluation.
        let fault = metal.test_fail_completed_command_after(0)?;
        let error = engine
            .prefill_chunk(&[7, 8, 9, 10, 11, 12, 13], &mut session)
            .err()
            .context("fixed-residency command injection unexpectedly succeeded")?;
        assert!(fault.fired());
        assert!(format!("{error:#}").contains("injected completed Metal command error"));
        drop(fault);
        assert!(engine.ane_residency.is_none());
        assert!(engine.ane_residency_failed);
        assert_eq!(native.allocationCount(), 0);
        assert!(!engine.prefill.workspace_in_progress());
        engine._expert_row_workspace.check_idle()?;
        engine.graph.status.reset()?;
        assert!(engine.prefill_chunk(&[7, 8], &mut session).is_err());
        assert!(engine.step(7, RoutingMode::Unbiased, &mut session).is_err());
        let failed_bytes = engine.memory_stats()?.live_requested_buffer_bytes;
        assert!(engine.start_session().is_err());
        assert_eq!(
            engine.memory_stats()?.live_requested_buffer_bytes,
            failed_bytes
        );
        drop(session);
        engine.close_ane()?;
        assert_eq!(native.allocationCount(), 0);
        assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn bounded_backing_growth_uses_serial_decode() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let mut engine = Engine::load(&package, 64, 1024)?;
        let mut session = engine.start_session()?;
        let prefill = engine.prefill_chunk(&[7, 8, 9, 10, 11, 12, 13], &mut session)?;
        assert_eq!(prefill.frontier, 7);
        let mut growth_layers = Vec::new();
        for (index, state) in session.layers.iter().enumerate() {
            if let LayerState::Qsa(state) = state
                && let Some((old, target)) = state.index.next_token_backing_growth()?
            {
                growth_layers.push((index, old, target));
            }
        }
        ensure!(
            !growth_layers.is_empty() && engine.decode_backing_growth(&session)?,
            "frontier seven did not require QSA backing growth"
        );
        // Event handoff encoding would trigger this fault. Backing growth must
        // complete through the serial path without admitting any handoff.
        let fault = engine.metal.test_fail_handoff_after(0)?;
        let output = engine.step(14, RoutingMode::Unbiased, &mut session)?;
        assert!(!fault.fired());
        drop(fault);
        assert_eq!(output.frontier, 8);
        assert_eq!(session.frontier()?, 8);
        assert_eq!(session.ple_state()?.offset, 8);
        for state in &session.layers {
            assert_eq!(state.frontier()?, 8);
        }
        assert!(
            output
                .logits_bf16
                .iter()
                .all(|&word| half::bf16::from_bits(word).is_finite())
        );
        for (index, old, target) in growth_layers {
            let LayerState::Qsa(state) = &session.layers[index] else {
                anyhow::bail!("QSA growth layer type changed");
            };
            assert!(target > old);
            assert_eq!(state.index.checkpoint_buffers()?[0].0.byte_len(), target);
        }
        assert!(!engine.decode_backing_growth(&session)?);
        let fault = engine.metal.test_fail_handoff_after(0)?;
        let error = engine
            .step(15, RoutingMode::Unbiased, &mut session)
            .err()
            .context("ordinary decode did not resume event handoffs after backing growth")?;
        ensure!(
            fault.fired()
                && error
                    .chain()
                    .any(|cause| cause.to_string() == "injected decode handoff encoding error"),
            "unexpected post-growth handoff failure: {error:#}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn bounded_handoff_encoding_failure_allows_new_session_on_same_engine() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let mut engine = Engine::load(&package, 12, 1024)?;
        let pools = |engine: &mut Engine| -> Vec<_> {
            engine
                .layers
                .iter_mut()
                .map(|layer| match layer {
                    Layer::Gdn(block) => block.experts_mut().snapshot(),
                    Layer::Qsa(block) => block.mlp.experts_mut().snapshot(),
                })
                .collect()
        };
        let before = pools(&mut engine);
        let mut failed = engine.start_session()?;
        {
            // Two earlier MLP commands are committed behind their gates. The
            // third epoch is admitted but fails before router/wait encoding.
            let fault = engine.metal.test_fail_handoff_after(2)?;
            let error = engine
                .step(7, RoutingMode::Unbiased, &mut failed)
                .err()
                .context("injected handoff error was accepted")?;
            ensure!(
                fault.fired()
                    && error
                        .chain()
                        .any(|cause| cause.to_string() == "injected decode handoff encoding error"),
                "unexpected handoff failure: {error:#}"
            );
        }
        assert!(failed.frontier().is_err());
        assert!(!failed.valid);
        assert_eq!(failed.frontier, 0);
        assert!(failed.consumed_tokens.is_empty());
        assert_eq!(pools(&mut engine), before);
        let mut retry = engine.start_session()?;
        let recovered = engine.step(7, RoutingMode::Unbiased, &mut retry)?;
        assert_eq!(recovered.frontier, 1);
        assert_eq!(retry.ple_state()?.offset, 1);
        assert_eq!(retry.consumed_tokens()?, &[7]);
        let mut reference = engine.start_session()?;
        let control = engine.step(7, RoutingMode::Unbiased, &mut reference)?;
        assert_eq!(recovered.logits_bf16, control.logits_bf16);
        assert_eq!(recovered.changed_routes, control.changed_routes);
        // Both sessions reach the real backing-growth fallback after recovery.
        for session in [&mut retry, &mut reference] {
            engine.prefill_chunk(&[8, 9, 10, 11, 12, 13], session)?;
            assert!(engine.decode_backing_growth(session)?);
        }
        let recovered_growth = engine.step(14, RoutingMode::Unbiased, &mut retry)?;
        let control_growth = engine.step(14, RoutingMode::Unbiased, &mut reference)?;
        assert_eq!(recovered_growth.frontier, 8);
        assert_eq!(recovered_growth.logits_bf16, control_growth.logits_bf16);
        assert_eq!(
            recovered_growth.changed_routes,
            control_growth.changed_routes
        );
        assert!(failed.frontier().is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn windowed_handoff_failure_after_service_allows_same_engine_retry() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let mut engine = Engine::load(&package, 64, 1024)?;
        let pools = |engine: &mut Engine| -> Vec<_> {
            engine
                .layers
                .iter_mut()
                .map(|layer| match layer {
                    Layer::Gdn(block) => block.experts_mut().snapshot(),
                    Layer::Qsa(block) => block.mlp.experts_mut().snapshot(),
                })
                .collect()
        };
        let before = pools(&mut engine);
        let mut failed = engine.start_session()?;
        {
            // The first four demands have been serviced. Layer four is
            // enqueued when the next layer fails before router encoding.
            let fault = engine.metal.test_fail_handoff_after(5)?;
            let error = engine
                .step(7, RoutingMode::Unbiased, &mut failed)
                .err()
                .context("windowed handoff error was accepted")?;
            ensure!(
                fault.fired()
                    && error
                        .chain()
                        .any(|cause| cause.to_string() == "injected decode handoff encoding error"),
                "unexpected windowed handoff failure: {error:#}"
            );
        }
        assert!(failed.frontier().is_err());
        assert!(!failed.valid);
        assert_eq!(failed.frontier, 0);
        assert!(failed.consumed_tokens.is_empty());
        let after = pools(&mut engine);
        assert!(
            after[..4]
                .iter()
                .zip(&before[..4])
                .all(|(after, before)| after.clock > before.clock),
            "the first four expert demands must be serviced before the injected failure"
        );
        assert_eq!(&after[4..], &before[4..]);
        for layer in &mut engine.layers {
            let mlp = match layer {
                Layer::Gdn(block) => block.mlp_mut(),
                Layer::Qsa(block) => &mut block.mlp,
            };
            mlp.check_expert_quiescent()?;
        }
        let mut retry = engine.start_session()?;
        let recovered = engine.step(7, RoutingMode::Unbiased, &mut retry)?;
        assert_eq!(recovered.frontier, 1);
        assert!(
            recovered
                .logits_bf16
                .iter()
                .all(|&word| half::bf16::from_bits(word).is_finite())
        );
        let mut reference = engine.start_session()?;
        let control = engine.step(7, RoutingMode::Unbiased, &mut reference)?;
        assert_eq!(recovered.logits_bf16, control.logits_bf16);
        assert_eq!(recovered.changed_routes, control.changed_routes);
        // Both sessions reach the real backing-growth fallback after recovery.
        for session in [&mut retry, &mut reference] {
            engine.prefill_chunk(&[8, 9, 10, 11, 12, 13], session)?;
            assert!(engine.decode_backing_growth(session)?);
        }
        let recovered_growth = engine.step(14, RoutingMode::Unbiased, &mut retry)?;
        let control_growth = engine.step(14, RoutingMode::Unbiased, &mut reference)?;
        assert_eq!(recovered_growth.frontier, 8);
        assert_eq!(recovered_growth.logits_bf16, control_growth.logits_bf16);
        assert_eq!(
            recovered_growth.changed_routes,
            control_growth.changed_routes
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn independent_sessions_keep_unmaterialized_device_logits() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let mut engine = Engine::load(&package, 12, 1024)?;
        let mut first = engine.start_session()?;
        let first_output = engine.prefill_chunk(&(7..71).collect::<Vec<_>>(), &mut first)?;
        // The diagnostic result has already read the row. Clear only its host
        // cache so the later assertion must read the Session-owned allocation.
        first.last_logits_bf16.take();
        assert!(first.last_logits_bf16.get().is_none());

        let mut second = engine.start_session()?;
        assert!(!first.logits_device.shares_allocation(&second.logits_device));
        let second_output = engine.prefill_chunk(&(71..135).collect::<Vec<_>>(), &mut second)?;
        assert_ne!(first_output.logits_bf16, second_output.logits_bf16);
        assert!(first.last_logits_bf16.get().is_none());
        assert_eq!(first.last_logits_bf16()?, first_output.logits_bf16);

        let snapshot = engine.snapshot(&first)?;
        assert_eq!(snapshot.last_logits_bf16(), first_output.logits_bf16);
        let mut restored = engine.restore(&snapshot)?;
        assert!(
            !first
                .logits_device
                .shares_allocation(&restored.logits_device)
        );
        restored.last_logits_bf16.take();
        assert_eq!(restored.last_logits_bf16()?, first_output.logits_bf16);
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn generation_device_selection_matches_diagnostic_engines() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let prompt: Vec<u32> = (7..71).collect();
        let stochastic = SamplingOptions::from_package(&package)?;
        ensure!(
            stochastic.temperature > 0.0,
            "package default is not stochastic"
        );
        let greedy = SamplingOptions {
            temperature: 0.0,
            ..SamplingOptions::default()
        };
        for options in [greedy, stochastic] {
            let (reference_ids, reference_rows, reference_frontiers, reference_routes) = {
                let mut engine = Engine::load(&package, 12, 1024)?;
                let mut session = engine.start_session()?;
                let mut preparation = MetalSamplingPreparation::new(&engine.metal, 248_320)?;
                let mut sampler = Sampler::new(options, 39)?;
                let mut history = prompt.clone();
                let mut output = engine.prefill_chunk(&prompt, &mut session)?;
                let mut ids = Vec::new();
                let mut rows = Vec::new();
                let mut frontiers = Vec::new();
                let mut routes = Vec::new();
                for index in 0..4 {
                    let id = sampler
                        .sample(&mut preparation, &output.logits_bf16, &history)?
                        .id;
                    ids.push(id);
                    rows.push(std::mem::take(&mut output.logits_bf16));
                    frontiers.push(output.frontier);
                    routes.push(output.changed_routes);
                    if index < 3 {
                        history.push(id);
                        output = engine.step(id, RoutingMode::CachePrior, &mut session)?;
                    }
                }
                (ids, rows, frontiers, routes)
            };
            let mut engine = Engine::load(&package, 12, 1024)?;
            let mut session = engine.start_session()?;
            let mut sampler = Sampler::new(options, 39)?;
            let mut history = prompt.clone();
            let mut output = engine
                .consume_for_generation(
                    &prompt,
                    RoutingMode::Unbiased,
                    &mut session,
                    &history,
                    Some(&mut sampler),
                    None,
                )?
                .context("uncancellable generation did not complete")?;
            let restored = engine.restore(&engine.snapshot(&session)?)?;
            let mut restored_sampler = Sampler::new(options, 39)?;
            assert_eq!(
                engine.sample_committed(&restored, &history, &mut restored_sampler)?,
                reference_ids[0],
            );
            for index in 0..4 {
                let id = output
                    .selected_id
                    .context("generation did not select an ID")?;
                assert_eq!(id, reference_ids[index]);
                assert_eq!(output.frontier, reference_frontiers[index]);
                assert_eq!(output.changed_routes, reference_routes[index]);
                assert_eq!(session.last_logits_bf16()?, reference_rows[index]);
                if index < 3 {
                    history.push(id);
                    output = engine
                        .consume_for_generation(
                            &[id],
                            RoutingMode::CachePrior,
                            &mut session,
                            &history,
                            Some(&mut sampler),
                            None,
                        )?
                        .context("uncancellable generation did not complete")?;
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU access"]
    fn route_first_wide_shared_suffix_error_drains_and_recovers() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        // The fixed test package and prompt need more than 128 layer-zero
        // sources, so the bounded planner loads multiple groups in this layer.
        let mut engine = Engine::load(&package, 128, 1024)?;
        assert!(matches!(engine.layers.first(), Some(Layer::Gdn(_))));
        let prompt = (0..512).map(|index| 7 + index % 32).collect::<Vec<u32>>();
        let pools = |engine: &mut Engine| -> Vec<_> {
            engine
                .layers
                .iter_mut()
                .map(|layer| match layer {
                    Layer::Gdn(block) => block.experts_mut().snapshot(),
                    Layer::Qsa(block) => block.mlp.experts_mut().snapshot(),
                })
                .collect()
        };
        let (reference_prefill, reference_step) = {
            let mut session = engine.start_session()?;
            let prefill = engine.prefill_chunk(&prompt, &mut session)?;
            let step = engine.step(7, RoutingMode::Unbiased, &mut session)?;
            assert_eq!((prefill.frontier, step.frontier), (512, 513));
            (prefill.logits_bf16, step.logits_bf16)
        };
        let baseline = engine.memory_stats()?.live_requested_buffer_bytes;
        let before = pools(&mut engine);
        let mut failed = engine.start_session()?;
        // A fresh stream submits the router prefix first and shared suffix
        // second. Skip prefix completion to fail the submitted shared suffix.
        let guard = engine.metal.test_fail_completed_command_after(1)?;
        let error = engine
            .prefill_chunk(&prompt, &mut failed)
            .err()
            .context("wide suffix injection unexpectedly succeeded")?;
        let message = format!("{error:#}");
        assert!(guard.fired(), "wide suffix fault did not fire");
        assert!(message.contains("injected completed Metal command error"));
        assert!(
            !message.contains("bounded MLP route demand boundary"),
            "expected suffix retirement within layer-zero groups, not a later route wait"
        );
        let after = pools(&mut engine);
        assert!(
            after[0].clock > before[0].clock,
            "no valid pool loads preceded suffix error"
        );
        assert_ne!(
            after[0].slot_of, before[0].slot_of,
            "pool publication prefix was not exercised"
        );
        assert_eq!(
            &after[1..],
            &before[1..],
            "later layers ran after suffix error"
        );
        assert!(failed.frontier().is_err());
        assert!(failed.last_logits_bf16().is_err());
        assert!(engine.snapshot(&failed).is_err());
        drop(guard);
        drop(failed);
        assert_eq!(engine.memory_stats()?.live_requested_buffer_bytes, baseline);
        {
            let mut recovered = engine.start_session()?;
            let prefill = engine.prefill_chunk(&prompt, &mut recovered)?;
            assert_eq!(prefill.frontier, 512);
            assert_eq!(prefill.logits_bf16, reference_prefill);
            let step = engine.step(7, RoutingMode::Unbiased, &mut recovered)?;
            assert_eq!(step.frontier, 513);
            assert_eq!(step.logits_bf16, reference_step);
        }
        assert_eq!(engine.memory_stats()?.live_requested_buffer_bytes, baseline);
        Ok(())
    }

    #[derive(Clone, Copy, Debug)]
    enum Failure {
        Allocation,
        CompletedCommand,
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn explicit_4096_command_failure_preserves_prior_conversation() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let tokenizer = TextTokenizer::load(&package)?;
        let mut engine = Engine::load_with_prefill_capacity(&package, 128, 6144, 4096)?;
        let prior: Vec<u32> = (7..39).collect();
        let mut cache = PromptPrefixCache::new(&engine, 1 << 30);
        {
            let mut session = engine.start_session()?;
            engine.prefill_chunk(&prior, &mut session)?;
            assert!(cache.offer_prompt_checkpoint(&engine, &session, &prior)?);
        }
        let retained = cache.retained_bytes()?;
        let options = GenerationOptions {
            max_tokens: 2,
            max_context: 6144,
            decode_mode: RoutingMode::CachePrior,
            sampling: SamplingOptions {
                temperature: 0.0,
                ..SamplingOptions::default()
            },
            seed: 0,
        };
        let prompt: Vec<_> = (0..4096).map(|i| 61 + i % 32).collect();
        let fault = engine.metal.test_fail_completed_command_after(0)?;
        let error = generate_with_cache(
            &mut engine,
            &tokenizer,
            &prompt,
            options,
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |_| panic!("failed wide prefill must not emit"),
            || false,
        )
        .unwrap_err();
        assert!(fault.fired());
        assert!(format!("{error:#}").contains("injected completed Metal command error"));
        drop(fault);
        assert_eq!(cache.retained_bytes()?, retained);
        assert!(cache.best_match(&engine, &prompt).is_none());
        let mut followup = prior.clone();
        followup.extend([39, 40, 41]);
        let resumed = generate_with_cache(
            &mut engine,
            &tokenizer,
            &followup,
            options,
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |_| Ok(()),
            || false,
        )?;
        assert_eq!(resumed.prompt_cache.cached_tokens, prior.len());
        assert_eq!(resumed.prefill_tokens, 3);
        assert_eq!(resumed.prompt_consumed, followup.len());
        Ok(())
    }

    impl Failure {
        fn arm(self, metal: &Metal) -> Result<TestFaultGuard> {
            match self {
                // Permit one allocation so a failed request has a partially
                // constructed owner to release during Result propagation.
                Self::Allocation => metal.test_fail_allocation_after(1),
                Self::CompletedCommand => metal.test_fail_completed_command_after(0),
            }
        }

        fn message(self) -> &'static str {
            match self {
                Self::Allocation => "injected Metal allocation rejection",
                Self::CompletedCommand => "injected completed Metal command error",
            }
        }
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn injected_metal_errors_allow_next_request_on_same_engine() -> Result<()> {
        // These controlled Result failures do not simulate driver OOM, device
        // loss, host allocator exhaustion, or an HTTP wire exchange.
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let tokenizer = TextTokenizer::load(&package)?;
        let mut engine = Engine::load(&package, 12, 1024)?;
        let prompt: Vec<u32> = (7..71).collect();
        let options = GenerationOptions {
            max_tokens: 2,
            max_context: 1024,
            decode_mode: RoutingMode::Unbiased,
            sampling: SamplingOptions {
                temperature: 0.0,
                ..SamplingOptions::default()
            },
            seed: 0,
        };
        // Retain complete numerical references and warm reusable allocations,
        // even if the sampled request ends immediately at a stop token.
        let (reference_prefill, reference_step) = {
            let mut session = engine.start_session()?;
            let prefill = engine.prefill_chunk(&prompt, &mut session)?;
            let step = engine.step(71, RoutingMode::Unbiased, &mut session)?;
            (prefill.logits_bf16, step.logits_bf16)
        };
        let check_recovered_logits = |engine: &mut Engine| -> Result<()> {
            let mut session = engine.start_session()?;
            let prefill = engine.prefill_chunk(&prompt, &mut session)?;
            assert!(
                prefill.logits_bf16 == reference_prefill,
                "recovered prefill logits differ"
            );
            let step = engine.step(71, RoutingMode::Unbiased, &mut session)?;
            assert!(
                step.logits_bf16 == reference_step,
                "recovered decode logits differ"
            );
            Ok(())
        };
        let request = |engine: &mut Engine, text: &mut String| -> Result<GenerationStats> {
            generate_with_cache(
                engine,
                &tokenizer,
                &prompt,
                options,
                GenerationCaches {
                    memory: None,
                    disk: None,
                },
                |part| {
                    text.push_str(part);
                    Ok(())
                },
                || false,
            )
        };
        let mut reference_text = String::new();
        let reference = request(&mut engine, &mut reference_text)?;
        let baseline = engine.memory_stats()?.live_requested_buffer_bytes;

        for failure in [Failure::Allocation, Failure::CompletedCommand] {
            let mut session = engine.start_session()?;
            let guard = failure.arm(&engine.metal)?;
            let error = engine
                .prefill_chunk(&prompt, &mut session)
                .err()
                .context("injected prefill unexpectedly succeeded")?;
            assert!(guard.fired(), "fault did not fire: {failure:?}");
            assert!(format!("{error:#}").contains(failure.message()));
            assert!(session.frontier().is_err());
            assert!(session.last_logits_bf16().is_err());
            assert!(engine.snapshot(&session).is_err());
            drop(guard);
            drop(session);
            assert_eq!(engine.memory_stats()?.live_requested_buffer_bytes, baseline);
            check_recovered_logits(&mut engine)?;
            assert_eq!(engine.memory_stats()?.live_requested_buffer_bytes, baseline);

            // Exercise the generation function used by both HTTP handlers.
            // Its request-local Session must unwind without restarting Engine.
            let guard = failure.arm(&engine.metal)?;
            let mut failed_text = String::new();
            let error = request(&mut engine, &mut failed_text)
                .err()
                .context("injected generation unexpectedly succeeded")?;
            assert!(guard.fired(), "request fault did not fire: {failure:?}");
            assert!(format!("{error:#}").contains(failure.message()));
            assert!(failed_text.is_empty());
            drop(guard);
            assert_eq!(engine.memory_stats()?.live_requested_buffer_bytes, baseline);
            check_recovered_logits(&mut engine)?;
            assert_eq!(engine.memory_stats()?.live_requested_buffer_bytes, baseline);

            let mut recovered_text = String::new();
            let recovered = request(&mut engine, &mut recovered_text)?;
            assert_eq!(recovered_text, reference_text);
            assert_eq!(recovered.prompt_consumed, reference.prompt_consumed);
            assert_eq!(recovered.generated_tokens, reference.generated_tokens);
            assert_eq!(recovered.frontier, reference.frontier);
            assert_eq!(recovered.changed_routes, reference.changed_routes);
            assert_eq!(recovered.finish, reference.finish);
            assert_eq!(engine.memory_stats()?.live_requested_buffer_bytes, baseline);
        }
        Ok(())
    }
}

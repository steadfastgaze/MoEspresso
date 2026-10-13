//! Capacity-bounded layer-major prefill on the decode engine's ordered stream.

use super::{Layer, LayerState, Residual, Session, device_greedy::DeviceGreedy};
use crate::decoder::FinalProjection;
use crate::dense_batch::DenseBatchScratch;
use crate::device_status::DeviceStatus;
use crate::embedding::Embedding;
use crate::gdn_batch::{GdnBatch, GdnBatchScratch};
use crate::hc_batch::{HcBatch, HcBatchRows, HcBatchScratch};
use crate::kvarn::KvarnEncoder;
use crate::kvarn_pending::PendingKvarnView;
use crate::metal::{Binding, Metal, Pipeline, PrefillResidencyGuard, Stream};
use crate::mlp_batch::{MlpBatch, MlpBatchScratch};
use crate::model::{MixerKind, QwenConfig};
use crate::package::Package;
use crate::ple_batch::{PleBatch, PleBatchScratch};
use crate::qsa_batch_attention::{QsaBatchAttention, QsaPrefillScratch};
use crate::qsa_batch_projection::QsaBatchProjector;
use crate::sampling::SamplingOptions;
use anyhow::{Context, Result, ensure};

enum Mixer {
    Gdn(Box<GdnBatch>),
    Qsa(Box<(QsaBatchProjector, QsaBatchAttention)>),
}

struct BatchLayer {
    attention: HcBatch,
    mixer: Mixer,
    mlp_read: HcBatch,
    mlp: MlpBatch,
}

/// Outputs alternate A/B; HC temporaries and each mixer workspace have only
/// ordered consumers. No workspace is multiplied by the number of layers.
struct PrefillWorkspace {
    hc: HcBatchScratch,
    rows: [HcBatchRows; 2],
    dense: DenseBatchScratch,
    gdn: GdnBatchScratch,
    qsa: QsaPrefillScratch,
    mlp: MlpBatchScratch,
    ple: PleBatchScratch,
}

impl PrefillWorkspace {
    fn new(
        metal: &Metal,
        layers: &[BatchLayer],
        ple: &PleBatch,
        max_context: usize,
        capacity: usize,
    ) -> Result<Self> {
        let mut dense_bytes = 4;
        for layer in layers {
            dense_bytes = dense_bytes
                .max(layer.attention.dense_scratch_bytes(capacity)?)
                .max(layer.mlp_read.dense_scratch_bytes(capacity)?)
                .max(layer.mlp.dense_scratch_bytes(capacity)?);
            dense_bytes = dense_bytes.max(match &layer.mixer {
                Mixer::Gdn(mixer) => mixer.dense_scratch_bytes(capacity)?,
                Mixer::Qsa(mixer) => mixer
                    .0
                    .dense_scratch_bytes(capacity)?
                    .max(mixer.1.dense_scratch_bytes(capacity)?),
            });
        }
        dense_bytes = dense_bytes.max(ple.dense_scratch_bytes(capacity)?);
        let workspace = Self {
            hc: HcBatchScratch::new(metal, capacity)?,
            rows: [
                HcBatchRows::new(metal, capacity)?,
                HcBatchRows::new(metal, capacity)?,
            ],
            dense: DenseBatchScratch::new(metal, dense_bytes)?,
            gdn: GdnBatchScratch::new(metal, capacity)?,
            qsa: QsaPrefillScratch::new(metal, capacity, max_context)?,
            mlp: MlpBatchScratch::new(metal, capacity)?,
            ple: PleBatchScratch::new(metal, capacity)?,
        };
        let requested = workspace.byte_len();
        let accounted =
            crate::memory::native_prefill_workspace_bytes_for_capacity(max_context, capacity)?;
        ensure!(
            requested == accounted,
            "{capacity}-row prefill workspace requested {requested} bytes but admission accounts for {accounted}"
        );
        Ok(workspace)
    }

    fn byte_len(&self) -> usize {
        self.hc.byte_len()
            + self.rows.iter().map(HcBatchRows::byte_len).sum::<usize>()
            + self.dense.byte_len()
            + self.gdn.byte_len()
            + self.qsa.byte_len()
            + self.mlp.byte_len()
            + self.ple.byte_len()
    }
}

pub(super) struct Prefill {
    capacity: usize,
    metal: Metal,
    layers: Vec<BatchLayer>,
    ple: PleBatch,
    workspace: Option<PrefillWorkspace>,
    write: Pipeline,
    ple_index: usize,
    workspace_lifetime: std::rc::Rc<()>,
}

/// One in-progress chunk. Stream retirement precedes the residency guard's
/// release, including cancellation and failures before the final layer.
pub(super) struct PrefillProgress {
    pub(super) stream: Stream,
    residual: Residual,
    ple_executed: bool,
    rows: usize,
    frontier: usize,
    _residency: Option<PrefillResidencyGuard>,
    _workspace_lifetime: std::rc::Rc<()>,
}

impl Prefill {
    pub(super) fn workspace_in_progress(&self) -> bool {
        std::rc::Rc::strong_count(&self.workspace_lifetime) > 1
    }

    pub(super) fn workspace_ready(&self) -> bool {
        self.workspace.is_some()
    }

    fn workspace_requested_bytes(&self) -> u64 {
        self.workspace
            .as_ref()
            .map_or(0, |workspace| workspace.byte_len() as u64)
    }

    /// Replace only retired scratch, releasing it before any new allocation.
    /// A failed allocation leaves no scratch and keeps the previous width for
    /// a later rebuild. Committed session state is stored elsewhere.
    pub(super) fn replace_workspace_without_overlap(
        &mut self,
        max_context: usize,
        capacity: usize,
        workspace_budget_bytes: u64,
    ) -> Result<(u64, u64)> {
        crate::prefill_capacity::validate_capacity(capacity)?;
        ensure!(
            !self.workspace_in_progress(),
            "prefill workspace is in progress"
        );
        let old = self.workspace_requested_bytes();
        let new = crate::memory::native_prefill_workspace_bytes_for_capacity(max_context, capacity)?
            as u64;
        ensure!(
            new <= workspace_budget_bytes,
            "phase workspace exceeds its complete serving budget"
        );
        if capacity == self.capacity && old != 0 {
            return Ok((old, old));
        }
        self.workspace = None;
        let workspace =
            PrefillWorkspace::new(&self.metal, &self.layers, &self.ple, max_context, capacity)?;
        self.workspace = Some(workspace);
        self.capacity = capacity;
        Ok((old, new))
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn new(
        metal: &Metal,
        package: &Package,
        config: &QwenConfig,
        decode_layers: &[Layer],
        ple: &crate::ple::Qwen4Ple,
        max_context: usize,
        capacity: usize,
    ) -> Result<Self> {
        crate::prefill_capacity::validate_capacity(capacity)?;
        ensure!(
            decode_layers.len() == config.layers.len(),
            "prefill/decode layer count differs"
        );
        let mut layers = Vec::with_capacity(config.layers.len());
        for (index, (kind, decode)) in config.layers.iter().zip(decode_layers).enumerate() {
            let (attention, mixer, mlp_read, mlp) = match (kind, decode) {
                (MixerKind::Gdn, Layer::Gdn(block)) => {
                    let (attention, mixer, mlp_read, mlp) = block.shared_parts();
                    (
                        attention,
                        Mixer::Gdn(Box::new(GdnBatch::from_decode(metal, mixer)?)),
                        mlp_read,
                        mlp,
                    )
                }
                (MixerKind::Qsa, Layer::Qsa(block)) => (
                    &block.attention_read,
                    Mixer::Qsa(Box::new((
                        QsaBatchProjector::from_decode(metal, &block.projection)?,
                        QsaBatchAttention::from_decode(metal, &block.attention)?,
                    ))),
                    &block.mlp_read,
                    &block.mlp,
                ),
                _ => anyhow::bail!("prefill/decode layer kinds differ"),
            };
            let layer = BatchLayer {
                attention: HcBatch::from_decode(metal, attention)?,
                mixer,
                mlp_read: HcBatch::from_decode(metal, mlp_read)?,
                mlp: MlpBatch::from_decode(metal, package, index, mlp)?,
            };
            layers.push(layer);
        }
        let ple = PleBatch::from_decode(metal, package, ple)?;
        let workspace = PrefillWorkspace::new(metal, &layers, &ple, max_context, capacity)?;
        Ok(Self {
            capacity,
            metal: metal.clone(),
            layers,
            ple,
            workspace: Some(workspace),
            write: metal.compile_msl_strict(
                include_str!("../../kernels/metal/prefill.metal"),
                "prefill_residual_write",
            )?,
            ple_index: config.ple_layer_index,
            workspace_lifetime: std::rc::Rc::new(()),
        })
    }

    pub(super) fn begin(
        &mut self,
        tokens: &[u32],
        frontier: usize,
        embedding: &Embedding,
        status: &DeviceStatus,
    ) -> Result<PrefillProgress> {
        let rows = tokens.len();
        ensure!(
            (2..=self.capacity).contains(&rows),
            "prefill chunk exceeds workspace"
        );
        let workspace = self
            .workspace
            .as_mut()
            .context("prefill workspace is absent")?;
        status.reset()?;
        let mut embedded = vec![0u16; rows * 2560];
        embedding.lookup_into(tokens, &mut embedded)?;
        // B is the initial residual and later the MLP residual. Attention A
        // consumes it before B's first overwrite in the same serial stream.
        for (row, values) in embedded.as_chunks::<2560>().0.iter().enumerate() {
            let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            for branch in 0..4 {
                workspace.rows[1]
                    .updated
                    .write_bytes((row * 10240 + branch * 2560) * 2, &bytes)?;
            }
        }
        let residual = Residual {
            value: workspace.rows[1].updated.clone(),
            pending: None,
        };
        // Keep the prefill residency request outside the stream lifetime so
        // outstanding commands retire before the request ends, including errors.
        let residency = self.metal.prefill_residency_guard()?;
        let stream = self.metal.stream()?;
        Ok(PrefillProgress {
            stream,
            residual,
            ple_executed: false,
            rows,
            frontier,
            _residency: residency,
            _workspace_lifetime: self.workspace_lifetime.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_layer(
        &mut self,
        index: usize,
        tokens: &[u32],
        session: &mut Session,
        decode: &mut Layer,
        encoder: &KvarnEncoder,
        status: &DeviceStatus,
        progress: &mut PrefillProgress,
    ) -> Result<()> {
        let layer = &mut self.layers[index];
        let state = &mut session.layers[index];
        let workspace = self
            .workspace
            .as_mut()
            .context("prefill workspace is absent")?;
        let rows = progress.rows;
        let frontier = progress.frontier;
        let stream = &mut progress.stream;
        let residual = &mut progress.residual;
        if index == self.ple_index {
            if let Some((output, injection)) = residual.pending.take() {
                let target = &workspace.rows[0].updated;
                let count = ((rows * 10240) as u32).to_le_bytes();
                let bindings = [&residual.value, &output, &injection, target]
                    .map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: B residual and pending operands are disjoint from
                // A. PLE consumes A before attention reuses its output rows.
                unsafe {
                    stream.batch()?.dispatch_1d_with_inline(
                        &self.write,
                        &bindings,
                        &[&count],
                        rows * 10240,
                        256,
                    )?;
                }
                residual.value = target.clone();
            }
            residual.value = self.ple.encode_chunk(
                stream,
                &mut session.ple,
                tokens,
                &residual.value,
                &mut workspace.ple,
                &workspace.dense,
                status,
            )?;
            progress.ple_executed = true;
        }
        let attention = unsafe {
            layer.attention.encode_into(
                stream.batch()?,
                &residual.value,
                residual.pending(),
                rows,
                &workspace.hc,
                &workspace.rows[0],
                &workspace.dense,
            )?
        };
        let injection = attention
            .injection
            .as_ref()
            .context("attention injection missing")?;
        let mixed = match (&layer.mixer, &mut *state, &mut *decode) {
            (Mixer::Gdn(mixer), LayerState::Gdn(state), Layer::Gdn(_)) => mixer.encode_chunk(
                stream.batch()?,
                &attention.mixed,
                rows,
                state.begin_prefill()?,
                &mut workspace.gdn,
                &workspace.dense,
                status,
            )?,
            (Mixer::Qsa(mixer), LayerState::Qsa(state), Layer::Qsa(decode)) => {
                let scratch = &mut workspace.qsa;
                let batch = stream.batch()?;
                unsafe {
                    mixer.0.encode_into(
                        batch,
                        &attention.mixed,
                        rows,
                        frontier,
                        &scratch.projected,
                        &workspace.dense,
                    )?;
                }
                // Dense index prefixes bypass scoring, and short attention
                // tails share the decode consumer. Validate their logical
                // projection rows without scanning unused capacity.
                for (buffer, width) in [
                    (&scratch.projected.index_queries, 512),
                    (&scratch.projected.raw_index_key, 128),
                    (&scratch.projected.queries, 6144),
                    (&scratch.projected.query_gate, 6144),
                ] {
                    status.encode_bf16_count(batch, buffer, rows * width)?;
                }
                let selection = decode.index.encode_chunk_device(
                    batch,
                    &mut state.index,
                    &scratch.projected.raw_index_key,
                    &scratch.projected.index_queries,
                    rows,
                    &mut scratch.index,
                    status,
                )?;
                // SAFETY: Token-major projection producers precede all
                // gathers. Append occurs only after the final pending read.
                let pending = unsafe {
                    PendingKvarnView::from_device(
                        &self.metal,
                        &state.kv,
                        &scratch.projected.keys,
                        &scratch.projected.values,
                        rows,
                    )?
                };
                let mut reconstructed = pending.encode_reconstruct(batch, &scratch.selected)?;
                for offset in (0..rows).step_by(64) {
                    let tile_rows = (rows - offset).min(64);
                    if tile_rows <= 8 && reconstructed.is_some() {
                        ensure!(offset + tile_rows == rows, "short QSA tile is not last");
                        // Ending the Rust view is not GPU retirement. The
                        // old gather may overwrite this scratch only after
                        // the final indexed reader on this ordered stream.
                        reconstructed = None;
                    }
                    if let Some(raw) = &reconstructed {
                        pending.encode_selection_guard(
                            batch,
                            &selection,
                            offset,
                            tile_rows,
                            &scratch.selected,
                            &scratch.effective_valid,
                            &scratch.safe_counts,
                        )?;
                        // SAFETY: Reconstruction and selector guard precede
                        // every indexed read; no scratch overwrite occurs
                        // until all indexed tiles have been encoded.
                        unsafe {
                            mixer.1.encode_tile_reconstructed(
                                batch,
                                &scratch.projected.queries,
                                &scratch.projected.query_gate,
                                raw,
                                &scratch.effective_valid,
                                &scratch.safe_counts,
                                &selection.ids,
                                offset,
                                tile_rows,
                                &scratch.output,
                                &selection.status,
                            )?;
                        }
                    } else {
                        pending.encode_tile_device_into(
                            batch,
                            &selection,
                            offset,
                            tile_rows,
                            &scratch.selected,
                            &scratch.effective_valid,
                            &scratch.safe_counts,
                        )?;
                        // SAFETY: The existing gather precedes its consumer;
                        // earlier indexed readers precede this scratch reuse.
                        unsafe {
                            mixer.1.encode_tile_online(
                                batch,
                                &scratch.projected.queries,
                                &scratch.projected.query_gate,
                                &scratch.selected,
                                &scratch.effective_valid,
                                &scratch.safe_counts,
                                offset,
                                tile_rows,
                                &scratch.output,
                                &selection.status,
                                &scratch.online_partials,
                            )?;
                        }
                    }
                }
                unsafe {
                    mixer.1.encode_projection_into(
                        batch,
                        &scratch.output,
                        rows,
                        &workspace.dense,
                    )?;
                }
                status.encode_bf16_count(batch, &scratch.output.gated, rows * 6144)?;
                drop(pending);
                status.encode_bf16_count(batch, &scratch.projected.keys, rows * 512)?;
                status.encode_bf16_count(batch, &scratch.projected.values, rows * 512)?;
                state.kv.encode_append_chunk(
                    batch,
                    encoder,
                    &scratch.projected.keys,
                    &scratch.projected.values,
                    rows,
                )?;
                scratch.output.output.clone()
            }
            _ => anyhow::bail!("prefill layer/state kinds differ"),
        };
        let mlp_read = unsafe {
            layer.mlp_read.encode_into(
                stream.batch()?,
                &attention.updated,
                Some((&mixed, injection)),
                rows,
                &workspace.hc,
                &workspace.rows[1],
                &workspace.dense,
            )?
        };
        let experts = match &mut *decode {
            Layer::Gdn(block) => block.experts_mut(),
            Layer::Qsa(block) => block.mlp.experts_mut(),
        };
        let full_resident = experts.capacity() == 512;
        let output = layer.mlp.run_device(
            stream,
            &mlp_read.mixed,
            rows,
            &workspace.mlp,
            &workspace.dense,
            experts,
            status,
        )?;
        *residual = Residual {
            value: mlp_read.updated,
            pending: Some((output, mlp_read.injection.context("MLP injection missing")?)),
        };
        if index == 1 && full_resident {
            stream.flush()?;
        }
        Ok(())
    }

    pub(super) fn finish(
        &mut self,
        session: &mut Session,
        head: &FinalProjection,
        status: &DeviceStatus,
        greedy: Option<(&mut DeviceGreedy, &[u32], SamplingOptions)>,
        progress: &mut PrefillProgress,
    ) -> Result<()> {
        let rows = progress.rows;
        let residual = &progress.residual;
        let stream = &mut progress.stream;
        ensure!(
            progress.ple_executed,
            "model did not execute declared PLE layer"
        );
        head.encode_last_row(
            stream.batch()?,
            &residual.value,
            residual.pending(),
            rows,
            status,
            &session.logits_device,
        )?;
        if let Some((greedy, history, options)) = greedy {
            greedy.encode(
                stream.batch()?,
                &session.logits_device,
                history,
                options,
                status,
            )?;
        }
        stream.wait().context("prefill graph completion")?;
        status.check()?;
        for state in &mut session.layers {
            match state {
                LayerState::Gdn(state) => {
                    state.publish_prefill_after_wait(rows)?;
                }
                LayerState::Qsa(state) => {
                    state.index.publish_after_wait()?;
                    state.kv.publish_after_wait()?;
                }
            }
        }
        Ok(())
    }
}

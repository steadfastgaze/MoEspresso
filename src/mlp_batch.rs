//! Unbiased batched MLP with caller-owned expert residency.

use anyhow::{Result, ensure};
use half::bf16;
use std::collections::BTreeSet;

const HIDDEN: usize = 2560;
const ROUTES: usize = 10;
const EXPERTS: usize = 512;
const SHARED: usize = 640;
const DOWN_STORED: usize = 768;
const SORTED_PAIRS: usize = 4096;
const LOGITS_READBACK_BYTES: usize = 16 * 1024;

fn validate_rows(rows: usize) -> Result<()> {
    ensure!(
        (2..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
        "MLP batch requires 2 through 4096 rows"
    );
    Ok(())
}

// Read only the completed projection's logical rows. The selector may have
// produced IDs from nonfinite logits, but they cannot reach expert demand.
fn validate_projected_logits(
    rows: usize,
    mut read: impl FnMut(usize, &mut [u8]) -> Result<()>,
) -> Result<()> {
    validate_rows(rows)?;
    let total = rows * EXPERTS * 2;
    let mut words = [0u8; LOGITS_READBACK_BYTES];
    for offset in (0..total).step_by(LOGITS_READBACK_BYTES) {
        let chunk = &mut words[..(total - offset).min(LOGITS_READBACK_BYTES)];
        read(offset, chunk)?;
        ensure!(
            chunk.as_chunks::<2>().0.iter().fold(true, |ok, word| {
                ok & bf16::from_bits(u16::from_le_bytes(*word)).is_finite()
            }),
            "routing requires finite BF16 logits"
        );
    }
    Ok(())
}

fn require_full_identity(capacity: usize, map: &[u32]) -> Result<()> {
    ensure!(
        capacity == EXPERTS
            && map.len() == EXPERTS
            && map
                .iter()
                .enumerate()
                .all(|(id, &slot)| slot as usize == id),
        "full-resident MLP requires all 512 experts in identity slots"
    );
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DemandChunk {
    pub start: usize,
    pub end: usize,
    pub active: BTreeSet<u32>,
}

/// Partition contiguous token rows by the complete selected-source union.
pub fn demand_chunks(ids: &[u32], capacity: usize) -> Result<Vec<DemandChunk>> {
    ensure!(
        !ids.is_empty() && ids.len().is_multiple_of(ROUTES),
        "MLP selections need complete top-ten rows"
    );
    ensure!(
        (ROUTES..EXPERTS).contains(&capacity),
        "MLP batch requires a bounded expert capacity"
    );
    let mut chunks = Vec::new();
    let mut active = BTreeSet::new();
    let mut start = 0;
    for (row, selected) in ids.as_chunks::<ROUTES>().0.iter().enumerate() {
        let row_active: BTreeSet<_> = selected.iter().copied().collect();
        ensure!(
            row_active.len() == ROUTES && row_active.iter().all(|&id| (id as usize) < EXPERTS),
            "MLP row has duplicate or invalid expert IDs"
        );
        if row > start && active.union(&row_active).count() > capacity {
            chunks.push(DemandChunk {
                start,
                end: row,
                active,
            });
            start = row;
            active = row_active;
        } else {
            active.extend(row_active);
        }
    }
    chunks.push(DemandChunk {
        start,
        end: ids.len() / ROUTES,
        active,
    });
    Ok(chunks)
}

/// Sort token-expert pairs by source, then bound each group to half capacity.
fn sorted_chunks(ids: &[u32], capacity: usize) -> Result<(Vec<usize>, Vec<DemandChunk>)> {
    demand_chunks(ids, capacity)?;
    let mut order: Vec<_> = (0..ids.len()).collect();
    order.sort_by_key(|&i| ids[i]);
    let all: BTreeSet<_> = ids.iter().copied().collect();
    if all.len() <= capacity {
        return Ok((
            order,
            vec![DemandChunk {
                start: 0,
                end: ids.len(),
                active: all,
            }],
        ));
    }
    let mut chunks = Vec::new();
    let mut active = BTreeSet::new();
    let mut start = 0;
    for (position, &pair) in order.iter().enumerate() {
        if !active.contains(&ids[pair]) && active.len() == capacity / 2 {
            chunks.push(DemandChunk {
                start,
                end: position,
                active,
            });
            start = position;
            active = BTreeSet::new();
        }
        active.insert(ids[pair]);
    }
    chunks.push(DemandChunk {
        start,
        end: order.len(),
        active,
    });
    Ok((order, chunks))
}

// Whole-union-fit small prefill keeps row order. Larger or over-capacity
// prefill uses source-sorted groups with protected reader overlap.
fn planned_chunks(ids: &[u32], capacity: usize) -> Result<(Vec<usize>, Vec<DemandChunk>, bool)> {
    if ids.len() >= SORTED_PAIRS {
        let (order, chunks) = sorted_chunks(ids, capacity)?;
        return Ok((order, chunks, true));
    }
    let chunks = demand_chunks(ids, capacity)?;
    if chunks.len() == 1 {
        return Ok(((0..ids.len()).collect(), chunks, false));
    }
    drop(chunks);
    let (order, chunks) = sorted_chunks(ids, capacity)?;
    Ok((order, chunks, true))
}

fn packed_pair_range(chunk: &DemandChunk, sorted: bool) -> std::ops::Range<usize> {
    if sorted {
        chunk.start..chunk.end
    } else {
        chunk.start * ROUTES..chunk.end * ROUTES
    }
}

struct PackedChunkRoutes {
    chunk: usize,
    ids: Vec<u32>,
    original_pairs: Vec<u32>,
    slot_map: [u32; EXPERTS],
    live_sources: usize,
}

// Filter a logical group without changing any source's pair sequence. Physical
// children keep original global pair indices for the single final reduction.
fn partition_packed_group(
    ids: &[u32],
    original_pairs: &[usize],
    map: &[u32; EXPERTS],
    capacity: usize,
    ranges: &[std::ops::Range<usize>],
    original_bound: usize,
) -> Result<Vec<PackedChunkRoutes>> {
    ensure!(
        !ids.is_empty()
            && ids.len() == original_pairs.len()
            && (ROUTES..EXPERTS).contains(&capacity)
            && original_bound <= u32::MAX as usize,
        "packed chunk route geometry differs"
    );
    let mut locations = Vec::with_capacity(capacity);
    let mut children = Vec::with_capacity(ranges.len());
    for (chunk, range) in ranges.iter().enumerate() {
        ensure!(
            range.start == locations.len() && range.start < range.end && range.end <= capacity,
            "packed expert chunks do not cover capacity"
        );
        locations.extend((0..range.len()).map(|local| (chunk, local)));
        children.push(PackedChunkRoutes {
            chunk,
            ids: Vec::new(),
            original_pairs: Vec::new(),
            slot_map: [EXPERTS as u32; EXPERTS],
            live_sources: 0,
        });
    }
    ensure!(
        locations.len() == capacity,
        "packed expert chunks do not cover capacity"
    );
    let mut owners = vec![None; capacity];
    for (&id, &original) in ids.iter().zip(original_pairs) {
        ensure!(
            (id as usize) < EXPERTS && original < original_bound,
            "packed chunk source or original pair is invalid"
        );
        let slot = map[id as usize] as usize;
        ensure!(slot < capacity, "packed chunk source is not published");
        ensure!(
            owners[slot].is_none_or(|owner| owner == id),
            "packed chunk sources alias a physical slot"
        );
        owners[slot] = Some(id);
        let (chunk, local) = locations[slot];
        let child = &mut children[chunk];
        if child.slot_map[id as usize] == EXPERTS as u32 {
            child.slot_map[id as usize] = local as u32;
            child.live_sources += 1;
        }
        child.ids.push(id);
        child.original_pairs.push(original as u32);
    }
    children.retain(|child| !child.ids.is_empty());
    Ok(children)
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use crate::dense_batch::{DenseBatch, DenseBatchScratch};
    use crate::device_status::DeviceStatus;
    use crate::experts::{
        BoundedExperts, ExpertDemandCounts, ExpertPoolSnapshot, PrefillLoadScope,
        sigmoid_fp16_table,
    };
    use crate::iqk::{Codec, GemvDescriptor};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline, Stream};
    use crate::package::{ExpertLayer, Package, Projection, WeightFormat};
    use crate::routing::{CachePriorRouter, RoutingBuffers, RoutingGeometry, RoutingMode};
    use anyhow::Context;
    use std::cell::OnceCell;
    use std::collections::BTreeMap;

    fn upload(metal: &Metal, bytes: &[u8]) -> Result<Buffer> {
        let out = metal.shared_buffer(bytes.len())?;
        out.write_bytes(0, bytes)?;
        Ok(out)
    }
    fn read(buffer: &Buffer) -> Result<Vec<u8>> {
        let mut out = vec![0; buffer.byte_len()];
        buffer.read_bytes(0, &mut out)?;
        Ok(out)
    }

    #[derive(Clone, Debug)]
    pub struct GroupDemand {
        pub chunk: DemandChunk,
        pub before: ExpertPoolSnapshot,
        pub after: ExpertPoolSnapshot,
        pub counts: ExpertDemandCounts,
    }
    #[derive(Clone, Debug, Default)]
    pub struct MlpBatchCounts {
        /// No host route demand or pool accounting occurs in this mode.
        pub full_resident: bool,
        /// Full residency uses packed tiles; bounded residency uses ranges.
        pub sorted: bool,
        pub rows: usize,
        pub chunks: usize,
        pub unique_sources: usize,
        pub hits: usize,
        pub misses: usize,
        pub evictions: usize,
    }
    pub struct MlpBatchOutput {
        pub output: Buffer,
        pub rows: usize,
        pub counts: MlpBatchCounts,
    }
    pub struct MlpBatchStages {
        pub output: MlpBatchOutput,
        pub stages: BTreeMap<&'static str, Vec<u8>>,
        pub groups: Vec<GroupDemand>,
        pub before: ExpertPoolSnapshot,
        pub after: ExpertPoolSnapshot,
        pub sorted_first: BTreeMap<&'static str, Vec<u8>>,
    }
    type RoutedDiagnostics = (
        MlpBatchCounts,
        Vec<GroupDemand>,
        BTreeMap<&'static str, Vec<u8>>,
    );

    struct PackedTileOccupancy {
        source_pairs: [usize; EXPERTS],
    }

    impl PackedTileOccupancy {
        fn new(ids: &[u32]) -> Result<Self> {
            let mut source_pairs = [0; EXPERTS];
            for &id in ids {
                ensure!((id as usize) < EXPERTS, "packed tile source is invalid");
                source_pairs[id as usize] += 1;
            }
            Ok(Self { source_pairs })
        }

        fn tiles(&self, tile_tokens: usize) -> usize {
            self.source_pairs
                .iter()
                .map(|count| count.div_ceil(tile_tokens))
                .sum()
        }

        fn select_width(&self) -> usize {
            // Repeated IQ_K reconstruction has a fixed cost per tile. The
            // score prices it as one 32-column matrix tile and
            // adds the actual padded token columns for each candidate width.
            [32usize, 16, 8]
                .into_iter()
                .min_by_key(|&width| (self.tiles(width) * (32 + width), usize::MAX - width))
                .unwrap()
        }
    }

    struct Scratch {
        logits: Buffer,
        ids: Buffer,
        scores: Buffer,
        changed: Buffer,
        gate: Buffer,
        up: Buffer,
        activation: Buffer,
        down: Buffer,
        scalar: Buffer,
        routed: Buffer,
        shared: Buffer,
        output: Buffer,
    }
    impl Scratch {
        fn new(metal: &Metal, rows: usize) -> Result<Self> {
            let bf = |width| metal.shared_buffer(rows * width * 2);
            Ok(Self {
                logits: bf(EXPERTS)?,
                ids: metal.shared_buffer(rows * ROUTES * 4)?,
                scores: bf(ROUTES)?,
                changed: metal.shared_buffer(rows * 4)?,
                gate: bf(SHARED)?,
                up: bf(SHARED)?,
                activation: bf(SHARED)?,
                down: bf(HIDDEN)?,
                scalar: bf(1)?,
                routed: bf(HIDDEN)?,
                shared: bf(HIDDEN)?,
                output: bf(HIDDEN)?,
            })
        }
        fn stages(&self) -> Result<BTreeMap<&'static str, Vec<u8>>> {
            [
                ("router_logits_bf16", &self.logits),
                ("selected_source_ids_u32", &self.ids),
                ("selected_scores_bf16", &self.scores),
                ("shared_gate_proj_bf16", &self.gate),
                ("shared_up_proj_bf16", &self.up),
                ("shared_activation_bf16", &self.activation),
                ("shared_down_bf16", &self.down),
                ("shared_gate_logit_bf16", &self.scalar),
                ("routed_bf16", &self.routed),
                ("shared_bf16", &self.shared),
                ("final_bf16", &self.output),
            ]
            .into_iter()
            .map(|(name, value)| Ok((name, read(value)?)))
            .collect()
        }

        fn byte_len(&self) -> usize {
            [
                &self.logits,
                &self.ids,
                &self.scores,
                &self.changed,
                &self.gate,
                &self.up,
                &self.activation,
                &self.down,
                &self.scalar,
                &self.routed,
                &self.shared,
                &self.output,
            ]
            .iter()
            .map(|buffer| buffer.byte_len())
            .sum()
        }
    }

    /// Engine-owned capacity scratch reused by every MLP prefill layer.
    pub struct MlpBatchScratch {
        capacity: usize,
        base: Scratch,
        x_half: Buffer,
        packed_activation: Buffer,
        packed_down: Buffer,
        order: Buffer,
        schedule: Buffer,
        total: Buffer,
        route_order: Buffer,
    }

    impl MlpBatchScratch {
        pub fn new(metal: &Metal, capacity: usize) -> Result<Self> {
            validate_rows(capacity)?;
            let pairs = capacity * ROUTES;
            let max_tiles = EXPERTS + pairs.div_ceil(8);
            Ok(Self {
                capacity,
                base: Scratch::new(metal, capacity)?,
                x_half: metal.shared_buffer(capacity * HIDDEN * 2)?,
                packed_activation: metal.shared_buffer(pairs * SHARED * 2)?,
                packed_down: metal.shared_buffer(pairs * HIDDEN * 2)?,
                order: metal.shared_buffer(pairs * 4)?,
                schedule: metal.shared_buffer(max_tiles * 3 * 4)?,
                total: metal.shared_buffer(4)?,
                route_order: metal.shared_buffer(pairs * 4)?,
            })
        }

        pub fn byte_len(&self) -> usize {
            self.base.byte_len()
                + [
                    &self.x_half,
                    &self.packed_activation,
                    &self.packed_down,
                    &self.order,
                    &self.schedule,
                    &self.total,
                    &self.route_order,
                ]
                .iter()
                .map(|buffer| buffer.byte_len())
                .sum::<usize>()
        }
    }

    /// The pool remains caller-owned so prefill and decode use one published map.
    pub struct MlpBatch {
        metal: Metal,
        source: ExpertLayer,
        router: DenseBatch,
        selector: CachePriorRouter,
        shared_gate: DenseBatch,
        shared_up: DenseBatch,
        shared_down: DenseBatch,
        scalar: DenseBatch,
        activate: Pipeline,
        finish: Pipeline,
        to_half: Pipeline,
        swiglu: Pipeline,
        reduce: Pipeline,
        sigmoid: Buffer,
        dequant: Pipeline,
        sorted_mm: Pipeline,
        gather_half: Pipeline,
        scatter_half: Pipeline,
        full: [OnceCell<FullResidentPipelines>; 3],
        addressed_full: [OnceCell<FullResidentPipelines>; 3],
    }

    struct FullResidentPipelines {
        schedule: Pipeline,
        sorted_schedule: Pipeline,
        route_order: Pipeline,
        gate_up: Pipeline,
        down: Pipeline,
        schedule_slots: Pipeline,
        order_original_pairs: Pipeline,
    }

    impl MlpBatch {
        fn packed_tile_width(&self, ids: Option<&[u32]>) -> Result<usize> {
            let occupancy = ids.map(PackedTileOccupancy::new).transpose()?;
            // Full residency keeps routes on the device without a host histogram.
            Ok(occupancy
                .as_ref()
                .map_or(32, PackedTileOccupancy::select_width))
        }

        pub(crate) fn dense_scratch_bytes(&self, capacity: usize) -> Result<usize> {
            [
                &self.router,
                &self.shared_gate,
                &self.shared_up,
                &self.shared_down,
                &self.scalar,
            ]
            .into_iter()
            .try_fold(4, |bytes, matrix| {
                Ok(bytes.max(matrix.scratch_bytes(capacity)?))
            })
        }

        pub fn new(metal: &Metal, package: &Package, layer: usize) -> Result<Self> {
            Self::build(metal, package, layer, None, package.expert_layer(layer)?)
        }
        pub(crate) fn from_decode(
            metal: &Metal,
            package: &Package,
            layer: usize,
            source: &crate::mlp::Qwen4Mlp,
        ) -> Result<Self> {
            Self::build(
                metal,
                package,
                layer,
                Some(source.shared_matrices()),
                source.source_layer(),
            )
        }
        fn build(
            metal: &Metal,
            package: &Package,
            layer: usize,
            shared: Option<[crate::kquant::DeviceMatrix; 5]>,
            expert_layer: crate::package::ExpertLayer,
        ) -> Result<Self> {
            let prefix = format!("layers.{layer}.mlp");
            let matrix = |slot: usize, leaf, format, shape: [usize; 2]| -> Result<DenseBatch> {
                if let Some(ref matrices) = shared {
                    matrices[slot].check(format, shape)?;
                    return DenseBatch::from_matrix(matrices[slot].clone());
                }
                let weight = package.direct_weight(&format!("{prefix}.{leaf}"))?;
                ensure!(
                    weight.format == format && weight.shape == shape,
                    "MLP batch {leaf}: format/geometry differs"
                );
                DenseBatch::new(metal, &weight)
            };
            let source = format!(
                "{}\n{}\n{}",
                include_str!("../kernels/metal/mlp.metal"),
                include_str!("../kernels/metal/expert_block.metal"),
                include_str!("../kernels/metal/mlp_batch.metal")
            );
            let pipeline = |name| -> Result<Pipeline> {
                let p = metal.compile_msl(&source, name)?;
                ensure!(
                    p.max_threads_per_threadgroup() >= 256,
                    "MLP batch requires 256-thread groups"
                );
                Ok(p)
            };
            Ok(Self {
                metal: metal.clone(),
                source: expert_layer,
                router: matrix(0, "gate", WeightFormat::Bf16, [EXPERTS, HIDDEN])?,
                selector: CachePriorRouter::new(metal)?,
                shared_gate: matrix(
                    1,
                    "shared_expert.gate_proj",
                    WeightFormat::Q6K,
                    [SHARED, HIDDEN],
                )?,
                shared_up: matrix(
                    2,
                    "shared_expert.up_proj",
                    WeightFormat::Q6K,
                    [SHARED, HIDDEN],
                )?,
                shared_down: matrix(
                    3,
                    "shared_expert.down_proj",
                    WeightFormat::Q8_0,
                    [HIDDEN, SHARED],
                )?,
                scalar: matrix(4, "shared_expert_gate", WeightFormat::Q6K, [1, HIDDEN])?,
                activate: pipeline("mlp_batch_activate")?,
                finish: pipeline("mlp_batch_finish")?,
                to_half: pipeline("mlp_batch_bf16_to_fp16")?,
                swiglu: pipeline("swiglu_10")?,
                reduce: pipeline("mlp_batch_reduce")?,
                sigmoid: upload(metal, &sigmoid_fp16_table())?,
                dequant: pipeline("mlp_batch_dequant")?,
                sorted_mm: pipeline("mlp_batch_sorted_mm")?,
                gather_half: pipeline("mlp_batch_gather_half")?,
                scatter_half: pipeline("mlp_batch_scatter_half")?,
                full: std::array::from_fn(|_| OnceCell::new()),
                addressed_full: std::array::from_fn(|_| OnceCell::new()),
            })
        }

        /// Encode unbiased multirow MLP on the caller's ordered stream.
        /// A bounded pool waits for compact routes and retires victim readers
        /// before replacing slots; full residency keeps packed and shared work on
        /// the device without a demand boundary.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn run_device(
            &self,
            stream: &mut Stream,
            hidden: &Buffer,
            rows: usize,
            scratch: &MlpBatchScratch,
            dense: &DenseBatchScratch,
            experts: &mut BoundedExperts,
            status: &DeviceStatus,
        ) -> Result<Buffer> {
            validate_rows(rows)?;
            ensure!(
                rows <= scratch.capacity && hidden.byte_len() >= rows * HIDDEN * 2,
                "MLP device input or workspace is too small"
            );
            experts.check_layer(&self.source)?;
            let full = experts.capacity() == EXPERTS;
            let residency = experts.residency_map()?;
            if full {
                require_full_identity(experts.capacity(), &residency)?;
            }
            let map = upload(
                &self.metal,
                &residency
                    .iter()
                    .flat_map(|slot| slot.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let pipelines = self.full_pipelines()?;
            let base = &scratch.base;
            let batch = stream.batch()?;
            // SAFETY: Fixed capacity buffers hold every active row. The shared
            // partition scratch is reused only by serial producer commands.
            unsafe {
                self.router
                    .encode_rows_into(batch, hidden, &base.logits, rows, dense)?;
                self.selector.encode(
                    batch,
                    RoutingBuffers {
                        logits_bf16: &base.logits,
                        gate_slots: &map,
                        up_slots: &map,
                        down_slots: &map,
                        ids: &base.ids,
                        scores_bf16: &base.scores,
                        changed: &base.changed,
                    },
                    RoutingGeometry::new(rows)?,
                    RoutingMode::Unbiased,
                )?;
            }
            let route_done = if !full && rows >= 512 {
                // Route ordering also reads IDs, so it must retire before the
                // host demand read rather than keeping IDs in the shared suffix.
                unsafe {
                    batch.dispatch_1d(
                        &pipelines.route_order,
                        &[&base.ids, &scratch.route_order]
                            .map(|buffer| Binding { buffer, offset: 0 }),
                        rows,
                        rows.min(128),
                    )?;
                }
                status.encode_bf16_count(batch, hidden, rows * HIDDEN)?;
                status.encode_bf16_count(batch, &base.logits, rows * EXPERTS)?;
                Some(stream.flush_marked()?)
            } else {
                None
            };
            let batch = stream.batch()?;
            // SAFETY: These producers read only hidden/direct weights and
            // ordered partition scratch, not route IDs, status or expert pools.
            unsafe {
                self.shared_gate
                    .encode_rows_into(batch, hidden, &base.gate, rows, dense)?;
                self.shared_up
                    .encode_rows_into(batch, hidden, &base.up, rows, dense)?;
                self.scalar
                    .encode_rows_into(batch, hidden, &base.scalar, rows, dense)?;
                batch.dispatch_1d(
                    &self.activate,
                    &[&base.gate, &base.up, &base.activation]
                        .map(|buffer| Binding { buffer, offset: 0 }),
                    rows * SHARED,
                    256,
                )?;
                self.shared_down.encode_rows_into(
                    batch,
                    &base.activation,
                    &base.down,
                    rows,
                    dense,
                )?;
                batch.dispatch_1d(
                    &self.to_half,
                    &[hidden, &scratch.x_half].map(|buffer| Binding { buffer, offset: 0 }),
                    rows * HIDDEN,
                    256,
                )?;
                if route_done.is_none() {
                    batch.dispatch_1d(
                        &pipelines.route_order,
                        &[&base.ids, &scratch.route_order]
                            .map(|buffer| Binding { buffer, offset: 0 }),
                        rows,
                        rows.min(128),
                    )?;
                }
            }
            if route_done.is_some() {
                stream.flush()?;
            } else {
                status.encode_bf16_count(batch, hidden, rows * HIDDEN)?;
                status.encode_bf16_count(batch, &base.logits, rows * EXPERTS)?;
            }

            if full {
                self.encode_packed_chunk(
                    stream.batch()?,
                    scratch,
                    experts,
                    &base.ids,
                    0,
                    rows * ROUTES,
                    EXPERTS,
                    self.packed_tile_width(None)?,
                    None,
                    None,
                    false,
                    None,
                    false,
                )?;
            } else {
                if let Some(marker) = route_done {
                    stream.wait_through(marker)
                } else {
                    stream.wait()
                }
                .context("bounded MLP route demand boundary")?;
                status.check()?;
                let mut bytes = vec![0u8; rows * ROUTES * 4];
                base.ids.read_bytes(0, &mut bytes)?;
                let ids = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes(*word))
                    .collect::<Vec<_>>();
                let (pair_order, chunks, sorted) = planned_chunks(&ids, experts.capacity())?;
                let ordered_ids = pair_order.iter().map(|&pair| ids[pair]).collect::<Vec<_>>();
                let route_buffers = if experts.is_chunked() {
                    None
                } else {
                    Some((
                        upload(
                            &self.metal,
                            &ordered_ids
                                .iter()
                                .flat_map(|id| id.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )?,
                        upload(
                            &self.metal,
                            &pair_order
                                .iter()
                                .map(|&pair| pair as u32)
                                .flat_map(|pair| pair.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )?,
                    ))
                };
                if sorted && chunks.len() > 1 {
                    let mut scope = PrefillLoadScope::new(experts, stream)?;
                    for chunk in &chunks {
                        scope.retire_previous()?;
                        scope.load_group(&chunk.active)?;
                        let pairs = packed_pair_range(chunk, true);
                        // SAFETY: CPU-validated ordered IDs in this pair range
                        // have exactly chunk.active as their source set. The
                        // packed scheduler remaps those sources with this
                        // captured map, and gate/up/down read only those slots.
                        unsafe {
                            scope.submit_group(|batch, pool, map| {
                                self.encode_packed_group(
                                    batch,
                                    scratch,
                                    pool,
                                    &ordered_ids,
                                    &pair_order,
                                    route_buffers.as_ref(),
                                    pairs.clone(),
                                    chunk.active.len(),
                                    map,
                                    true,
                                )
                            })?;
                        }
                    }
                    scope.finish()?;
                } else {
                    let mut protect = BTreeSet::new();
                    for chunk in chunks {
                        let pairs = packed_pair_range(&chunk, sorted);
                        if chunk
                            .active
                            .iter()
                            .any(|&source| experts.slot_of(source).is_none())
                        {
                            stream
                                .wait()
                                .context("bounded MLP expert overwrite boundary")?;
                            status.check()?;
                        }
                        if sorted {
                            experts.ensure_group_protected(&chunk.active, &protect)?;
                            protect = chunk.active.clone();
                        } else {
                            experts.ensure_group(&chunk.active)?;
                        }
                        let map = experts.residency_map()?;
                        self.encode_packed_group(
                            stream.batch()?,
                            scratch,
                            experts,
                            &ordered_ids,
                            &pair_order,
                            route_buffers.as_ref(),
                            pairs,
                            chunk.active.len(),
                            &map,
                            sorted,
                        )?;
                    }
                }
            }
            // SAFETY: Every original pair index is written exactly once by the
            // packed groups; reduction follows all expert and shared producers.
            unsafe {
                stream.batch()?.dispatch_1d(
                    &self.reduce,
                    &[
                        &scratch.packed_down,
                        &base.scores,
                        &scratch.route_order,
                        &base.routed,
                    ]
                    .map(|buffer| Binding { buffer, offset: 0 }),
                    rows * HIDDEN,
                    256,
                )?;
                let finish = [
                    &base.routed,
                    &base.down,
                    &base.scalar,
                    &base.shared,
                    &base.output,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                stream
                    .batch()?
                    .dispatch_1d(&self.finish, &finish, rows * HIDDEN, 256)?;
            }
            status.encode_bf16_count(stream.batch()?, &base.output, rows * HIDDEN)?;
            Ok(base.output.clone())
        }

        /// Successful groups may update pool demand accounting before an error.
        /// A caller must invalidate its model session if execution fails.
        /// This diagnostic path requires contiguous expert storage.
        pub fn run_stages(
            &self,
            hidden: &[u8],
            rows: usize,
            experts: &mut BoundedExperts,
        ) -> Result<MlpBatchStages> {
            validate_rows(rows)?;
            ensure!(
                hidden.len() == rows * HIDDEN * 2
                    && hidden
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|v| bf16::from_bits(u16::from_le_bytes(*v)).is_finite()),
                "MLP batch requires complete finite BF16 rows"
            );
            let input = upload(&self.metal, hidden)?;
            // SAFETY: The host inputs were checked and uploaded to new storage.
            unsafe { self.execute(&input, rows, experts, true) }
        }

        /// Complete the device prefill path for fixture comparison, returning
        /// the output and selected source IDs after all producers finish.
        pub fn run_device_replay(
            &self,
            hidden: &[u8],
            rows: usize,
            experts: &mut BoundedExperts,
        ) -> Result<(Vec<u8>, Vec<u8>)> {
            validate_rows(rows)?;
            ensure!(
                hidden.len() == rows * HIDDEN * 2
                    && hidden
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|v| bf16::from_bits(u16::from_le_bytes(*v)).is_finite()),
                "MLP batch requires complete finite BF16 rows"
            );
            let input = upload(&self.metal, hidden)?;
            let scratch = MlpBatchScratch::new(&self.metal, rows)?;
            let dense = DenseBatchScratch::new(&self.metal, self.dense_scratch_bytes(rows)?)?;
            let status = DeviceStatus::new(&self.metal)?;
            let mut stream = self.metal.stream()?;
            let output = self.run_device(
                &mut stream,
                &input,
                rows,
                &scratch,
                &dense,
                experts,
                &status,
            )?;
            stream.wait()?;
            status.check()?;
            Ok((read(&output)?, read(&scratch.base.ids)?))
        }

        /// # Safety
        /// Input is finite contiguous BF16 `[rows,2560]`, with prior producers
        /// complete and no aliasing with writable pool or result allocations.
        /// An execution error requires invalidating the caller's model session.
        /// This diagnostic path requires contiguous expert storage.
        pub unsafe fn run_buffer(
            &self,
            hidden: &Buffer,
            rows: usize,
            experts: &mut BoundedExperts,
        ) -> Result<MlpBatchOutput> {
            // SAFETY: Forward the caller's device-buffer contract.
            Ok(unsafe { self.execute(hidden, rows, experts, false) }?.output)
        }

        unsafe fn execute(
            &self,
            hidden: &Buffer,
            rows: usize,
            experts: &mut BoundedExperts,
            capture: bool,
        ) -> Result<MlpBatchStages> {
            validate_rows(rows)?;
            ensure!(
                !experts.is_chunked(),
                "legacy MLP diagnostics require contiguous expert storage"
            );
            ensure!(
                hidden.byte_len() >= rows * HIDDEN * 2,
                "MLP batch input buffer is too small"
            );
            experts.check_layer(&self.source)?;
            let full_resident = experts.capacity() == EXPERTS;
            let residency = experts.residency_map()?;
            if full_resident {
                require_full_identity(experts.capacity(), &residency)?;
            }
            let before = experts.snapshot();
            let scratch = Scratch::new(&self.metal, rows)?;
            let maps = upload(
                &self.metal,
                &residency
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let mut batch = self.metal.batch()?;
            // SAFETY: Fixed geometry and disjoint fresh scratch; all input
            // producers precede use in the one ordered command batch.
            unsafe {
                self.router
                    .encode_rows(&mut batch, hidden, &scratch.logits, rows)?;
                self.selector.encode(
                    &mut batch,
                    RoutingBuffers {
                        logits_bf16: &scratch.logits,
                        gate_slots: &maps,
                        up_slots: &maps,
                        down_slots: &maps,
                        ids: &scratch.ids,
                        scores_bf16: &scratch.scores,
                        changed: &scratch.changed,
                    },
                    RoutingGeometry::new(rows)?,
                    RoutingMode::Unbiased,
                )?;
                self.shared_gate
                    .encode_rows(&mut batch, hidden, &scratch.gate, rows)?;
                self.shared_up
                    .encode_rows(&mut batch, hidden, &scratch.up, rows)?;
                self.scalar
                    .encode_rows(&mut batch, hidden, &scratch.scalar, rows)?;
                let bind = [&scratch.gate, &scratch.up, &scratch.activation]
                    .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.activate, &bind, rows * SHARED, 256)?;
                self.shared_down.encode_rows(
                    &mut batch,
                    &scratch.activation,
                    &scratch.down,
                    rows,
                )?;
            }
            batch.submit().wait()?;
            validate_projected_logits(rows, |offset, bytes| {
                scratch.logits.read_bytes(offset, bytes)
            })?;
            // Full residency leaves routes on-device and never enters demand
            // accounting. Diagnostic reads occur only after GPU completion.
            let ids: Vec<u32> = if full_resident {
                Vec::new()
            } else {
                read(&scratch.ids)?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_le_bytes(*v))
                    .collect()
            };
            let plan = if full_resident {
                None
            } else {
                Some(planned_chunks(&ids, experts.capacity())?)
            };
            let (counts, groups, mut sorted_first) = if full_resident {
                // SAFETY: All slots are published in identity order; the
                // completed router only produces source IDs below 512.
                unsafe { self.run_full_resident(hidden, rows, &scratch, experts, capture)? }
            } else if plan.as_ref().is_some_and(|(_, _, sorted)| *sorted) {
                // SAFETY: The shared input contract and fresh scratch were checked above.
                let (order, chunks, _) = plan.expect("bounded prefill has a plan");
                unsafe {
                    self.run_sorted(hidden, &ids, (order, chunks), &scratch, experts, capture)?
                }
            } else {
                let (_, chunks, _) = plan.expect("bounded prefill has a plan");
                let unique_sources = ids.iter().copied().collect::<BTreeSet<_>>().len();
                let mut counts = MlpBatchCounts {
                    rows,
                    chunks: if unique_sources > experts.capacity() {
                        chunks.len()
                    } else {
                        0
                    },
                    unique_sources,
                    ..Default::default()
                };
                let mut groups = Vec::new();
                for chunk in chunks {
                    let previous = capture.then(|| experts.snapshot());
                    let demand = experts.ensure_group(&chunk.active)?;
                    counts.hits += demand.hits;
                    counts.misses += demand.misses;
                    counts.evictions += demand.evictions;
                    if let Some(previous) = previous {
                        groups.push(GroupDemand {
                            chunk: chunk.clone(),
                            before: previous,
                            after: experts.snapshot(),
                            counts: demand,
                        });
                    }
                    let selected = &ids[chunk.start * ROUTES..chunk.end * ROUTES];
                    let slots: Vec<u32> = selected
                        .iter()
                        .map(|&id| {
                            experts
                                .slot_of(id)
                                .map(|v| v as u32)
                                .context("selected expert not resident")
                        })
                        .collect::<Result<_>>()?;
                    let pairs = selected.len();
                    let tokens = chunk.end - chunk.start;
                    let selectors = upload(
                        &self.metal,
                        &slots
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let order: Vec<u32> = selected
                        .as_chunks::<ROUTES>()
                        .0
                        .iter()
                        .flat_map(|row| {
                            let mut order: Vec<_> = (0..ROUTES as u32).collect();
                            order.sort_by_key(|&i| row[i as usize]);
                            order
                        })
                        .collect();
                    let order = upload(
                        &self.metal,
                        &order
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let x = self.metal.shared_buffer(tokens * HIDDEN * 2)?;
                    let gate = self.metal.shared_buffer(pairs * SHARED * 2)?;
                    let up = self.metal.shared_buffer(pairs * SHARED * 2)?;
                    let activation = self.metal.shared_buffer(pairs * DOWN_STORED * 2)?;
                    let down = self.metal.shared_buffer(pairs * HIDDEN * 2)?;
                    let mut batch = self.metal.batch()?;
                    let (gp, gg) = experts.projection(Projection::Gate)?;
                    let (up_project, ug) = experts.projection(Projection::Up)?;
                    let (dp, dg) = experts.projection(Projection::Down)?;
                    let gate_d = GemvDescriptor::new(gg, pairs, ROUTES)?;
                    let up_d = GemvDescriptor::new(ug, pairs, ROUTES)?;
                    let down_d = GemvDescriptor::new(dg, pairs, 1)?;
                    let gate_dims = upload(
                        &self.metal,
                        &gate_d
                            .dims
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let down_dims = upload(
                        &self.metal,
                        &down_d
                            .dims
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    // SAFETY: The whole selected union was published once. Kernel
                    // readers complete before any later group can overwrite slots.
                    unsafe {
                        batch.dispatch_1d(
                            &self.to_half,
                            &[
                                Binding {
                                    buffer: hidden,
                                    offset: chunk.start * HIDDEN * 2,
                                },
                                Binding {
                                    buffer: &x,
                                    offset: 0,
                                },
                            ],
                            tokens * HIDDEN,
                            256,
                        )?;
                        gp.encode(
                            &mut batch, gate_d, &slots, &x, &selectors, &gate_dims, &gate,
                        )?;
                        up_project
                            .encode(&mut batch, up_d, &slots, &x, &selectors, &gate_dims, &up)?;
                        let bind = [&gate, &up, &self.sigmoid, &activation]
                            .map(|buffer| Binding { buffer, offset: 0 });
                        batch.dispatch_1d(&self.swiglu, &bind, pairs * DOWN_STORED, 256)?;
                        dp.encode(
                            &mut batch,
                            down_d,
                            &slots,
                            &activation,
                            &selectors,
                            &down_dims,
                            &down,
                        )?;
                        batch.dispatch_1d(
                            &self.reduce,
                            &[
                                Binding {
                                    buffer: &down,
                                    offset: 0,
                                },
                                Binding {
                                    buffer: &scratch.scores,
                                    offset: chunk.start * ROUTES * 2,
                                },
                                Binding {
                                    buffer: &order,
                                    offset: 0,
                                },
                                Binding {
                                    buffer: &scratch.routed,
                                    offset: chunk.start * HIDDEN * 2,
                                },
                            ],
                            tokens * HIDDEN,
                            256,
                        )?;
                    }
                    batch.submit().wait()?;
                }
                (counts, groups, BTreeMap::new())
            };
            let mut batch = self.metal.batch()?;
            let bind = [
                &scratch.routed,
                &scratch.down,
                &scratch.scalar,
                &scratch.shared,
                &scratch.output,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: Every routed slice and shared row is complete; each
            // output index has one independent writer and all extents match.
            unsafe {
                batch.dispatch_1d(&self.finish, &bind, rows * HIDDEN, 256)?;
            }
            batch.submit().wait()?;
            let mut stages = if capture {
                scratch.stages()?
            } else {
                BTreeMap::new()
            };
            if full_resident {
                stages.append(&mut sorted_first);
            }
            Ok(MlpBatchStages {
                output: MlpBatchOutput {
                    output: scratch.output,
                    rows,
                    counts,
                },
                stages,
                groups,
                before,
                after: experts.snapshot(),
                sorted_first,
            })
        }

        fn full_pipelines(&self) -> Result<&FullResidentPipelines> {
            self.packed_pipelines(32)
        }

        fn packed_pipelines(&self, tile_tokens: usize) -> Result<&FullResidentPipelines> {
            let index = match tile_tokens {
                8 => 0,
                16 => 1,
                32 => 2,
                _ => anyhow::bail!("packed tile width must be 8, 16 or 32"),
            };
            let cell = &self.full[index];
            self.initialize_packed_pipelines(tile_tokens, cell, false)
        }

        fn addressed_pipelines(&self, tile_tokens: usize) -> Result<&FullResidentPipelines> {
            let index = match tile_tokens {
                8 => 0,
                16 => 1,
                32 => 2,
                _ => anyhow::bail!("packed tile width must be 8, 16 or 32"),
            };
            self.initialize_packed_pipelines(tile_tokens, &self.addressed_full[index], true)
        }

        fn initialize_packed_pipelines<'a>(
            &self,
            tile_tokens: usize,
            cell: &'a OnceCell<FullResidentPipelines>,
            addressed: bool,
        ) -> Result<&'a FullResidentPipelines> {
            if cell.get().is_none() {
                let source = format!(
                    "#define MLP_FULL_TILE_TOKENS {tile_tokens}\n#define MLP_PACKED_ADDRESSED {}\n\
                     {}",
                    u8::from(addressed),
                    include_str!("../kernels/metal/mlp_full_resident.metal")
                );
                ensure!(
                    cell.set(self.compile_packed_pipelines(&source)?).is_ok(),
                    "MLP pipelines already initialized"
                );
            }
            Ok(cell.get().unwrap())
        }

        fn compile_packed_pipelines(&self, source: &str) -> Result<FullResidentPipelines> {
            let compile = |name, threads| -> Result<Pipeline> {
                let pipeline = self.metal.compile_msl_strict(source, name)?;
                ensure!(
                    pipeline.max_threads_per_threadgroup() >= threads,
                    "full-resident MLP threadgroup unsupported"
                );
                Ok(pipeline)
            };
            Ok(FullResidentPipelines {
                schedule: compile("mlp_full_schedule", 512)?,
                sorted_schedule: compile("mlp_sorted_schedule", 512)?,
                route_order: compile("mlp_full_route_order", 128)?,
                gate_up: compile("mlp_full_gate_up", 128)?,
                down: compile("mlp_full_down", 128)?,
                schedule_slots: compile("mlp_schedule_slots", 128)?,
                order_original_pairs: compile("mlp_order_original_pairs", 128)?,
            })
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_packed_group(
            &self,
            batch: &mut Batch,
            scratch: &MlpBatchScratch,
            experts: &BoundedExperts,
            ordered_ids: &[u32],
            original_pairs: &[usize],
            route_buffers: Option<&(Buffer, Buffer)>,
            pairs: std::ops::Range<usize>,
            live_sources: usize,
            slot_map: &[u32; EXPERTS],
            source_sorted: bool,
        ) -> Result<()> {
            if experts.is_chunked() && ordered_ids.len() <= 64 * ROUTES {
                ensure!(
                    pairs.start < pairs.end
                        && pairs.end <= ordered_ids.len()
                        && pairs.end <= original_pairs.len(),
                    "addressed MLP logical group range differs"
                );
                let selected = &ordered_ids[pairs.clone()];
                ensure!(
                    selected.iter().all(|&source| {
                        (source as usize) < EXPERTS
                            && (slot_map[source as usize] as usize) < experts.capacity()
                    }),
                    "addressed MLP group contains an unpublished source"
                );
                let original = &original_pairs[pairs.clone()];
                ensure!(
                    original.iter().all(|&pair| pair < ordered_ids.len()),
                    "addressed MLP original pair exceeds active rows"
                );
                let ids = upload(
                    &self.metal,
                    &selected
                        .iter()
                        .flat_map(|id| id.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                let originals = upload(
                    &self.metal,
                    &original
                        .iter()
                        .flat_map(|&pair| (pair as u32).to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                self.encode_packed_chunk(
                    batch,
                    scratch,
                    experts,
                    &ids,
                    0,
                    selected.len(),
                    live_sources,
                    self.packed_tile_width(Some(selected))?,
                    Some(slot_map),
                    Some(&originals),
                    source_sorted,
                    None,
                    true,
                )?;
                return Ok(());
            }
            if !experts.is_chunked() {
                let (ids, original) =
                    route_buffers.context("contiguous packed MLP route buffers are missing")?;
                return self.encode_packed_chunk(
                    batch,
                    scratch,
                    experts,
                    ids,
                    pairs.start,
                    pairs.end,
                    live_sources,
                    self.packed_tile_width(Some(&ordered_ids[pairs.clone()]))?,
                    Some(slot_map),
                    Some(original),
                    source_sorted,
                    None,
                    false,
                );
            }
            ensure!(
                pairs.start < pairs.end
                    && pairs.end <= ordered_ids.len()
                    && pairs.end <= original_pairs.len(),
                "packed MLP logical group range differs"
            );
            let ranges = (0..experts.chunk_count())
                .map(|chunk| experts.chunk_range(chunk))
                .collect::<Result<Vec<_>>>()?;
            let children = partition_packed_group(
                &ordered_ids[pairs.clone()],
                &original_pairs[pairs],
                slot_map,
                experts.capacity(),
                &ranges,
                ordered_ids.len(),
            )?;
            ensure!(
                children
                    .iter()
                    .map(|child| child.live_sources)
                    .sum::<usize>()
                    == live_sources,
                "packed MLP logical source set differs"
            );
            // All physical children remain in the logical group's one batch.
            // Serial commands reuse schedule scratch and scatter to disjoint
            // original pairs before the next child replaces that scratch.
            for child in children {
                let ids = upload(
                    &self.metal,
                    &child
                        .ids
                        .iter()
                        .flat_map(|id| id.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                let original = upload(
                    &self.metal,
                    &child
                        .original_pairs
                        .iter()
                        .flat_map(|pair| pair.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                self.encode_packed_chunk(
                    batch,
                    scratch,
                    experts,
                    &ids,
                    0,
                    child.ids.len(),
                    child.live_sources,
                    self.packed_tile_width(Some(&child.ids))?,
                    Some(&child.slot_map),
                    Some(&original),
                    source_sorted,
                    Some(child.chunk),
                    false,
                )?;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_packed_chunk(
            &self,
            batch: &mut Batch,
            scratch: &MlpBatchScratch,
            experts: &BoundedExperts,
            sorted_ids: &Buffer,
            start: usize,
            end: usize,
            live_sources: usize,
            tile_tokens: usize,
            slot_map: Option<&[u32; EXPERTS]>,
            original_pairs: Option<&Buffer>,
            source_sorted: bool,
            physical_chunk: Option<usize>,
            addressed: bool,
        ) -> Result<()> {
            ensure!(
                start < end
                    && end <= scratch.capacity * ROUTES
                    && (1..=EXPERTS).contains(&live_sources),
                "packed MLP chunk geometry differs"
            );
            ensure!(
                sorted_ids.byte_len() >= end * 4,
                "packed MLP route buffer is too small"
            );
            if let Some(original) = original_pairs {
                ensure!(
                    original.byte_len() >= end * 4,
                    "packed MLP pair map is too small"
                );
            }
            ensure!(
                !source_sorted || (slot_map.is_some() && original_pairs.is_some()),
                "sorted packed MLP needs bounded source and pair maps"
            );
            let pairs = end - start;
            if addressed {
                ensure!(
                    end <= 64 * ROUTES,
                    "addressed packed producers require bounded short groups"
                );
            }
            let max_tiles = live_sources + pairs.div_ceil(tile_tokens);
            ensure!(
                max_tiles * 3 * 4 <= scratch.schedule.byte_len(),
                "packed MLP schedule exceeds workspace"
            );
            let pipelines = if addressed {
                self.addressed_pipelines(tile_tokens)?
            } else {
                self.packed_pipelines(tile_tokens)?
            };
            let projection = |kind| {
                if addressed {
                    experts.chunk_projection(0, kind)
                } else if let Some(chunk) = physical_chunk {
                    experts.chunk_projection(chunk, kind)
                } else {
                    experts.projection(kind)
                }
            };
            let (gate, gate_g) = projection(Projection::Gate)?;
            let (up, up_g) = projection(Projection::Up)?;
            let (down, down_g) = projection(Projection::Down)?;
            ensure!(
                gate_g.codec() == up_g.codec(),
                "packed gate/up IQ_K members differ"
            );
            let codec = |g: crate::iqk::Geometry| {
                u32::from(if g.codec() == Codec::Iq3K { 3u8 } else { 2u8 })
            };
            let dims = upload(
                &self.metal,
                &[pairs as u32, codec(gate_g), codec(down_g), 0]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let binding = |buffer| Binding { buffer, offset: 0 };
            // SAFETY: The router writes IDs before scheduling. Logical source
            // groups map each tile's expert address to a published physical
            // slot in bounded mode. Each tile's first word is the sole expert
            // address in gate/up and down:
            // each reads only that slot's 640 or 2560 stored output rows.
            // IQ2 never reads the absent qh/sch placeholder bindings. Batch
            // retains every stream, including aliases, until completion.
            unsafe {
                batch.dispatch_1d(
                    if source_sorted {
                        &pipelines.sorted_schedule
                    } else {
                        &pipelines.schedule
                    },
                    &[
                        Binding {
                            buffer: sorted_ids,
                            offset: start * 4,
                        },
                        binding(&dims),
                        binding(&scratch.order),
                        binding(&scratch.schedule),
                        binding(&scratch.total),
                    ],
                    EXPERTS,
                    EXPERTS,
                )?;
                if let Some(original) = original_pairs {
                    let pair_count = (pairs as u32).to_le_bytes();
                    batch.dispatch_1d_with_inline(
                        &pipelines.order_original_pairs,
                        &[
                            binding(&scratch.order),
                            Binding {
                                buffer: original,
                                offset: start * 4,
                            },
                        ],
                        &[&pair_count],
                        pairs,
                        128,
                    )?;
                }
                if let Some(map) = slot_map {
                    let map_bytes = map
                        .iter()
                        .flat_map(|slot| slot.to_le_bytes())
                        .collect::<Vec<_>>();
                    batch.dispatch_1d_with_inline(
                        &pipelines.schedule_slots,
                        &[binding(&scratch.schedule), binding(&scratch.total)],
                        &[&map_bytes],
                        max_tiles,
                        128,
                    )?;
                }
                let g = gate.stream_bindings();
                let u = up.stream_bindings();
                let d = down.stream_bindings();
                let mut gate_bindings = vec![Binding {
                    buffer: &scratch.x_half,
                    offset: 0,
                }];
                gate_bindings.extend(g);
                gate_bindings.extend(u.into_iter().take(6));
                gate_bindings.extend([
                    binding(&scratch.order),
                    binding(&scratch.schedule),
                    binding(&scratch.total),
                    binding(&self.sigmoid),
                    binding(&dims),
                    binding(&scratch.packed_activation),
                    binding(&scratch.base.changed),
                    binding(&scratch.base.changed),
                ]);
                if addressed {
                    let table = experts.retain_addressed_prefill(batch)?;
                    let capacity = u32::try_from(experts.capacity())?.to_le_bytes();
                    gate_bindings.push(binding(table));
                    batch.dispatch_1d_with_inline(
                        &pipelines.gate_up,
                        &gate_bindings,
                        &[&capacity],
                        max_tiles * 20 * 128,
                        128,
                    )?;
                } else {
                    batch.dispatch_1d(
                        &pipelines.gate_up,
                        &gate_bindings,
                        max_tiles * 20 * 128,
                        128,
                    )?;
                }
                let mut down_bindings = vec![binding(&scratch.packed_activation)];
                down_bindings.extend(d);
                down_bindings.extend([
                    binding(&scratch.order),
                    binding(&scratch.schedule),
                    binding(&scratch.total),
                    binding(&dims),
                    binding(&scratch.packed_down),
                ]);
                if addressed {
                    let table = experts.retain_addressed_prefill(batch)?;
                    let capacity = u32::try_from(experts.capacity())?.to_le_bytes();
                    down_bindings.push(binding(table));
                    batch.dispatch_1d_with_inline(
                        &pipelines.down,
                        &down_bindings,
                        &[&capacity],
                        max_tiles * 80 * 128,
                        128,
                    )?;
                } else {
                    batch.dispatch_1d(
                        &pipelines.down,
                        &down_bindings,
                        max_tiles * 80 * 128,
                        128,
                    )?;
                }
            }
            Ok(())
        }

        unsafe fn run_full_resident(
            &self,
            hidden: &Buffer,
            rows: usize,
            scratch: &Scratch,
            experts: &BoundedExperts,
            capture: bool,
        ) -> Result<RoutedDiagnostics> {
            let pipelines = self.full_pipelines()?;
            let pairs = rows * ROUTES;
            let packed = pairs >= SORTED_PAIRS;
            let (gp, gg) = experts.projection(Projection::Gate)?;
            let (up, ug) = experts.projection(Projection::Up)?;
            let (dp, dg) = experts.projection(Projection::Down)?;
            ensure!(
                !packed || gg.codec() == ug.codec(),
                "packed gate/up require matching IQ_K members"
            );
            let x = self.metal.shared_buffer(rows * HIDDEN * 2)?;
            let gate = self.metal.shared_buffer(if !packed || capture {
                pairs * SHARED * 2
            } else {
                2
            })?;
            let up_output = self.metal.shared_buffer(gate.byte_len())?;
            let activation_width = if packed { SHARED } else { DOWN_STORED };
            let activation = self.metal.shared_buffer(pairs * activation_width * 2)?;
            let down = self.metal.shared_buffer(pairs * HIDDEN * 2)?;
            let route_order = self.metal.shared_buffer(pairs * 4)?;
            let encoded = |words: &[u32]| {
                upload(
                    &self.metal,
                    &words
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            };
            let mut batch = self.metal.batch()?;
            // SAFETY: Fixed row extents, fresh disjoint scratch, and immutable
            // source IDs produced by the checked 512-expert router.
            unsafe {
                batch.dispatch_1d(
                    &self.to_half,
                    &[hidden, &x].map(|buffer| Binding { buffer, offset: 0 }),
                    rows * HIDDEN,
                    256,
                )?;
                batch.dispatch_1d(
                    &pipelines.route_order,
                    &[&scratch.ids, &route_order].map(|buffer| Binding { buffer, offset: 0 }),
                    rows,
                    rows.min(128),
                )?;
                if packed {
                    let max_tiles = EXPERTS + pairs.div_ceil(32);
                    let order = self.metal.shared_buffer(pairs * 4)?;
                    let schedule = self.metal.shared_buffer(max_tiles * 3 * 4)?;
                    let total = self.metal.shared_buffer(4)?;
                    let codec =
                        |g: crate::iqk::Geometry| if g.codec() == Codec::Iq3K { 3 } else { 2 };
                    let dims = encoded(&[pairs as u32, codec(gg), codec(dg), u32::from(capture)])?;
                    batch.dispatch_1d(
                        &pipelines.schedule,
                        &[&scratch.ids, &dims, &order, &schedule, &total]
                            .map(|buffer| Binding { buffer, offset: 0 }),
                        EXPERTS,
                        EXPERTS,
                    )?;
                    let mut bindings = vec![Binding {
                        buffer: &x,
                        offset: 0,
                    }];
                    bindings.extend(gp.stream_bindings());
                    // The gate/up members share one value table, but retain
                    // distinct read-only packed projection streams.
                    bindings.extend(up.stream_bindings().into_iter().take(6));
                    bindings.extend(
                        [
                            &order,
                            &schedule,
                            &total,
                            &self.sigmoid,
                            &dims,
                            &activation,
                            &gate,
                            &up_output,
                        ]
                        .map(|buffer| Binding { buffer, offset: 0 }),
                    );
                    batch.dispatch_1d(&pipelines.gate_up, &bindings, max_tiles * 20 * 128, 128)?;
                    let mut bindings = vec![Binding {
                        buffer: &activation,
                        offset: 0,
                    }];
                    bindings.extend(dp.stream_bindings());
                    bindings.extend(
                        [&order, &schedule, &total, &dims, &down]
                            .map(|buffer| Binding { buffer, offset: 0 }),
                    );
                    batch.dispatch_1d(&pipelines.down, &bindings, max_tiles * 80 * 128, 128)?;
                } else {
                    let gd = GemvDescriptor::new(gg, pairs, ROUTES)?;
                    let ud = GemvDescriptor::new(ug, pairs, ROUTES)?;
                    let dd = GemvDescriptor::new(dg, pairs, 1)?;
                    let gate_dims = encoded(&gd.dims)?;
                    let down_dims = encoded(&dd.dims)?;
                    gp.encode_device_selectors(
                        &mut batch,
                        gd,
                        &x,
                        &scratch.ids,
                        &gate_dims,
                        &gate,
                    )?;
                    up.encode_device_selectors(
                        &mut batch,
                        ud,
                        &x,
                        &scratch.ids,
                        &gate_dims,
                        &up_output,
                    )?;
                    batch.dispatch_1d(
                        &self.swiglu,
                        &[&gate, &up_output, &self.sigmoid, &activation]
                            .map(|buffer| Binding { buffer, offset: 0 }),
                        pairs * DOWN_STORED,
                        256,
                    )?;
                    dp.encode_device_selectors(
                        &mut batch,
                        dd,
                        &activation,
                        &scratch.ids,
                        &down_dims,
                        &down,
                    )?;
                }
                batch.dispatch_1d(
                    &self.reduce,
                    &[&down, &scratch.scores, &route_order, &scratch.routed]
                        .map(|buffer| Binding { buffer, offset: 0 }),
                    rows * HIDDEN,
                    256,
                )?;
            }
            batch.submit().wait()?;
            let mut stages = BTreeMap::new();
            if capture {
                for (name, buffer) in if packed {
                    [
                        ("debug_gate_f16", &gate),
                        ("debug_up_f16", &up_output),
                        ("packed_activation_f16", &activation),
                        ("packed_down_f16", &down),
                    ]
                } else {
                    [
                        ("small_gate_projection", &gate),
                        ("small_up_projection", &up_output),
                        ("small_activation_padded_f16", &activation),
                        ("small_down_projection", &down),
                    ]
                } {
                    stages.insert(name, read(buffer)?);
                }
            }
            // The reference full-resident branch deliberately leaves its
            // unique-source and demand counters untouched as well as the pool.
            Ok((
                MlpBatchCounts {
                    full_resident: true,
                    sorted: packed,
                    rows,
                    ..Default::default()
                },
                Vec::new(),
                stages,
            ))
        }

        /// Encode one projection using ten power-of-two output ranges. The
        /// temporary decoded range is reused only after its prior GPU readers.
        #[allow(clippy::too_many_arguments)]
        unsafe fn sorted_projection(
            &self,
            batch: &mut Batch,
            experts: &BoundedExperts,
            kind: Projection,
            slots: &[u32],
            tile_words: &[u32],
            input: &Buffer,
            output: &Buffer,
            capture_first: bool,
        ) -> Result<Buffer> {
            let (projection, geometry) = experts.projection(kind)?;
            let k = geometry.in_features();
            let n = geometry.out_features();
            let range = n / 10;
            let dequant_threads = slots.len() * range * (k / 32);
            ensure!(
                dequant_threads > 0 && dequant_threads.is_multiple_of(256),
                "sorted dequantization requires complete 256-thread groups"
            );
            let encoded = |words: &[u32]| {
                upload(
                    &self.metal,
                    &words
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            };
            let source_slots = encoded(slots)?;
            let tiles = encoded(tile_words)?;
            let weights = self.metal.shared_buffer(slots.len() * range * k * 2)?;
            // Only diagnostics need the first range after later ranges run.
            // Otherwise one allocation is reused in command order.
            let first = if capture_first {
                self.metal.shared_buffer(weights.byte_len())?
            } else {
                weights.clone()
            };
            for part in 0..10 {
                let base = part * range;
                let dims = encoded(&[
                    k as u32,
                    n as u32,
                    base as u32,
                    range as u32,
                    if geometry.codec() == Codec::Iq3K {
                        3
                    } else {
                        2
                    },
                ])?;
                let decoded = if part == 0 { &first } else { &weights };
                let mut bindings = projection.stream_bindings().to_vec();
                bindings.extend([
                    Binding {
                        buffer: &source_slots,
                        offset: 0,
                    },
                    Binding {
                        buffer: &dims,
                        offset: 0,
                    },
                    Binding {
                        buffer: decoded,
                        offset: 0,
                    },
                ]);
                // SAFETY: Fixed package geometry, checked pool slots and exact
                // range allocations; the batch retains all streams and scratch.
                unsafe {
                    batch.dispatch_1d(&self.dequant, &bindings, dequant_threads, 256)?;
                }
                let mm_dims = encoded(&[k as u32, n as u32, range as u32, base as u32])?;
                let bindings = [input, decoded, &tiles, &mm_dims, output]
                    .map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: Every tile describes at most sixteen source-homogeneous
                // rows; ranges cover only the declared projection output.
                unsafe {
                    batch.dispatch_1d(
                        &self.sorted_mm,
                        &bindings,
                        tile_words.len() / 3 * (range / 64) * 64,
                        64,
                    )?;
                }
            }
            Ok(first)
        }

        unsafe fn run_sorted(
            &self,
            hidden: &Buffer,
            ids: &[u32],
            plan: (Vec<usize>, Vec<DemandChunk>),
            scratch: &Scratch,
            experts: &mut BoundedExperts,
            capture: bool,
        ) -> Result<RoutedDiagnostics> {
            let (order, chunks) = plan;
            let unique_sources = ids.iter().copied().collect::<BTreeSet<_>>().len();
            let mut counts = MlpBatchCounts {
                sorted: true,
                rows: ids.len() / ROUTES,
                chunks: if unique_sources > experts.capacity() {
                    chunks.len()
                } else {
                    0
                },
                unique_sources,
                ..Default::default()
            };
            let mut groups = Vec::new();
            let mut first = BTreeMap::new();
            let mut protect = BTreeSet::new();
            let all_down = self.metal.shared_buffer(ids.len() * HIDDEN * 2)?;
            for (chunk_index, chunk) in chunks.into_iter().enumerate() {
                let previous = capture.then(|| experts.snapshot());
                let demand = experts.ensure_group_protected(&chunk.active, &protect)?;
                counts.hits += demand.hits;
                counts.misses += demand.misses;
                counts.evictions += demand.evictions;
                if let Some(before) = previous {
                    groups.push(GroupDemand {
                        chunk: chunk.clone(),
                        before,
                        after: experts.snapshot(),
                        counts: demand,
                    });
                }
                let pair_order = &order[chunk.start..chunk.end];
                let pairs = pair_order.len();
                let slots = chunk
                    .active
                    .iter()
                    .map(|&id| {
                        experts
                            .slot_of(id)
                            .map(|slot| slot as u32)
                            .context("sorted source is not resident")
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut tiles = Vec::new();
                let mut start = 0;
                for (source, &id) in chunk.active.iter().enumerate() {
                    let end = start
                        + pair_order[start..]
                            .iter()
                            .take_while(|&&p| ids[p] == id)
                            .count();
                    for row in (start..end).step_by(16) {
                        tiles.extend([row as u32, end as u32, source as u32]);
                    }
                    start = end;
                }
                ensure!(start == pairs, "sorted expert tiles do not cover all pairs");
                let pair_bytes: Vec<_> = pair_order
                    .iter()
                    .flat_map(|&v| (v as u32).to_le_bytes())
                    .collect();
                let pair_buffer = upload(&self.metal, &pair_bytes)?;
                let x = self.metal.shared_buffer(pairs * HIDDEN * 2)?;
                let gate = self.metal.shared_buffer(pairs * SHARED * 2)?;
                let up = self.metal.shared_buffer(pairs * SHARED * 2)?;
                let activation = self.metal.shared_buffer(pairs * DOWN_STORED * 2)?;
                let down = self.metal.shared_buffer(pairs * HIDDEN * 2)?;
                let capture_first = capture && chunk_index == 0;
                let mut batch = self.metal.batch()?;
                // SAFETY: Pair IDs came from complete checked top-ten rows;
                // all buffers have the fixed widths consumed by the kernels.
                let (gate_weights, up_weights, down_weights) = unsafe {
                    let bindings =
                        [hidden, &pair_buffer, &x].map(|buffer| Binding { buffer, offset: 0 });
                    batch.dispatch_1d(&self.gather_half, &bindings, pairs * HIDDEN, 256)?;
                    let gw = self.sorted_projection(
                        &mut batch,
                        experts,
                        Projection::Gate,
                        &slots,
                        &tiles,
                        &x,
                        &gate,
                        capture_first,
                    )?;
                    let uw = self.sorted_projection(
                        &mut batch,
                        experts,
                        Projection::Up,
                        &slots,
                        &tiles,
                        &x,
                        &up,
                        capture_first,
                    )?;
                    let bindings = [&gate, &up, &self.sigmoid, &activation]
                        .map(|buffer| Binding { buffer, offset: 0 });
                    batch.dispatch_1d(&self.swiglu, &bindings, pairs * DOWN_STORED, 256)?;
                    let dw = self.sorted_projection(
                        &mut batch,
                        experts,
                        Projection::Down,
                        &slots,
                        &tiles,
                        &activation,
                        &down,
                        capture_first,
                    )?;
                    let bindings = [&down, &pair_buffer, &all_down]
                        .map(|buffer| Binding { buffer, offset: 0 });
                    batch.dispatch_1d(&self.scatter_half, &bindings, pairs * HIDDEN, 256)?;
                    (gw, uw, dw)
                };
                batch.submit().wait()?;
                if capture_first {
                    first.insert("pair_order_u32", pair_bytes);
                    for (name, buffer) in [
                        ("input_f16", &x),
                        ("gate_f16", &gate),
                        ("up_f16", &up),
                        ("activation_f16", &activation),
                        ("down_f16", &down),
                        ("gate_weights_f16", &gate_weights),
                        ("up_weights_f16", &up_weights),
                        ("down_weights_f16", &down_weights),
                    ] {
                        first.insert(name, read(buffer)?);
                    }
                }
                protect = chunk.active;
            }
            let source_order: Vec<_> = ids
                .as_chunks::<ROUTES>()
                .0
                .iter()
                .flat_map(|row| {
                    let mut order: Vec<_> = (0..ROUTES as u32).collect();
                    order.sort_by_key(|&i| row[i as usize]);
                    order
                })
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let source_order = upload(&self.metal, &source_order)?;
            let bindings = [&all_down, &scratch.scores, &source_order, &scratch.routed]
                .map(|buffer| Binding { buffer, offset: 0 });
            let mut batch = self.metal.batch()?;
            // SAFETY: All pair outputs are complete and scattered into their
            // original top-ten row positions before source-ordered reduction.
            unsafe {
                batch.dispatch_1d(&self.reduce, &bindings, counts.rows * HIDDEN, 256)?;
            }
            batch.submit().wait()?;
            Ok((counts, groups, first))
        }
    }

    #[cfg(test)]
    mod device_mapping_tests {
        use super::*;

        include!("mlp_batch/wide_grouped_tests.rs");

        #[test]
        fn packed_tile_policy_uses_source_histogram_and_reconstruction_work() -> Result<()> {
            let one_per_source = (0..64).collect::<Vec<u32>>();
            assert_eq!(PackedTileOccupancy::new(&one_per_source)?.select_width(), 8);
            for (count, expected) in [(9, 16), (17, 32), (33, 32)] {
                let ids = (0..4)
                    .flat_map(|source| std::iter::repeat_n(source, count))
                    .collect::<Vec<_>>();
                let occupancy = PackedTileOccupancy::new(&ids)?;
                assert_eq!(occupancy.select_width(), expected);
                for width in [8, 16, 32] {
                    assert_eq!(occupancy.tiles(width), 4 * count.div_ceil(width));
                }
            }
            // A single concentrated source does not force every sparse source
            // to pay for unused token matrices in the same physical group.
            let mut mixed = vec![0; 33];
            mixed.extend(1..64);
            assert_eq!(PackedTileOccupancy::new(&mixed)?.select_width(), 8);
            assert!(PackedTileOccupancy::new(&[512]).is_err());
            Ok(())
        }

        #[test]
        fn packed_tile_widths_complete_routed_projection_with_original_pairs() -> Result<()> {
            use half::f16;

            let metal = Metal::new()?;
            let words = |values: &[u32]| -> Result<Buffer> {
                upload(
                    &metal,
                    &values
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            };
            let streams = |codec: Codec, width: usize, output: usize, divisor: f32| {
                let count = 4 * output;
                let mut qs = vec![if codec == Codec::Iq3K { 0 } else { 0xaa }; count * width / 4];
                if width == DOWN_STORED {
                    // Nonzero stored padding must not enter the logical down dot.
                    for row in 0..count {
                        qs[row * width / 4 + SHARED / 4..(row + 1) * width / 4].fill(0x55);
                    }
                }
                let qs = upload(&metal, &qs)?;
                let scl = upload(
                    &metal,
                    &vec![if codec == Codec::Iq3K { 0 } else { 0x99 }; count * width / 32],
                )?;
                let qh = if codec == Codec::Iq3K {
                    upload(&metal, &vec![0xff; count * width / 8])?
                } else {
                    qs.clone()
                };
                let sch = if codec == Codec::Iq3K {
                    upload(&metal, &vec![0; count * width / 128])?
                } else {
                    scl.clone()
                };
                let dv = (0..count)
                    .flat_map(|row| {
                        let scale = f16::from_f32((row / output + 1) as f32 / divisor)
                            .to_bits()
                            .to_le_bytes();
                        (0..width / 256).flat_map(move |_| scale)
                    })
                    .collect::<Vec<_>>();
                Ok::<_, anyhow::Error>([
                    qs,
                    qh,
                    scl,
                    sch,
                    upload(&metal, &vec![0; count * width / 128])?,
                    upload(&metal, &dv)?,
                    upload(&metal, &codec.value_table_fp16())?,
                ])
            };
            let rows = 7;
            let pairs = rows * ROUTES;
            let mut ids = vec![0u32; 33];
            ids.extend([1u32; 17]);
            ids.extend([3u32; 20]);
            ids.rotate_left(11);
            ids.reverse();
            let original = (0..pairs)
                .map(|pair| (pair * 11) % pairs)
                .collect::<Vec<_>>();
            let mut slots = [u32::MAX; EXPERTS];
            slots[0] = 3;
            slots[1] = 0;
            slots[3] = 2;
            let ids_buffer = words(&ids)?;
            let original_buffer =
                words(&original.iter().map(|&pair| pair as u32).collect::<Vec<_>>())?;
            let slot_buffer = words(&slots)?;
            let input = upload(
                &metal,
                &(0..rows)
                    .flat_map(|row| {
                        let value = f16::from_f32((row + 1) as f32 / 64.0)
                            .to_bits()
                            .to_le_bytes();
                        (0..HIDDEN).flat_map(move |_| value)
                    })
                    .collect::<Vec<_>>(),
            )?;
            let sigmoid_bytes = sigmoid_fp16_table();
            let sigmoid = upload(&metal, &sigmoid_bytes)?;
            let guard_bytes = 64;
            for codec in [Codec::Iq2K, Codec::Iq3K] {
                let gate = streams(codec, HIDDEN, SHARED, 2048.0)?;
                let up = streams(codec, HIDDEN, SHARED, 4096.0)?;
                let down = streams(codec, DOWN_STORED, HIDDEN, 2048.0)?;
                for tile_tokens in [8usize, 16, 32] {
                    let source = format!(
                        "#define MLP_FULL_TILE_TOKENS {tile_tokens}\n{}",
                        include_str!("../kernels/metal/mlp_full_resident.metal")
                    );
                    let compile = |name| metal.compile_msl_strict(&source, name);
                    let schedule_kernel = compile("mlp_full_schedule")?;
                    let remap = compile("mlp_schedule_slots")?;
                    let reorder = compile("mlp_order_original_pairs")?;
                    let gate_kernel = compile("mlp_full_gate_up")?;
                    let down_kernel = compile("mlp_full_down")?;
                    let order = metal.shared_buffer(pairs * 4)?;
                    let max_tiles = EXPERTS + pairs.div_ceil(tile_tokens);
                    let schedule = metal.shared_buffer(max_tiles * 3 * 4)?;
                    let total = words(&[u32::MAX])?;
                    let member = if codec == Codec::Iq3K { 3 } else { 2 };
                    let dims = words(&[pairs as u32, member, member, 0])?;
                    let pair_count = words(&[pairs as u32])?;
                    let activation = upload(&metal, &vec![0x5a; pairs * SHARED * 2 + guard_bytes])?;
                    let output = upload(&metal, &vec![0x5a; pairs * HIDDEN * 2 + guard_bytes])?;
                    let binding = |buffer| Binding { buffer, offset: 0 };
                    let mut batch = metal.batch()?;
                    // SAFETY: Complete synthetic stream-major projections cover
                    // all four slots. The bijective pair map addresses finite
                    // input rows and distinct guarded result ranges.
                    unsafe {
                        batch.dispatch_1d(
                            &schedule_kernel,
                            &[&ids_buffer, &dims, &order, &schedule, &total].map(binding),
                            EXPERTS,
                            EXPERTS,
                        )?;
                        batch.dispatch_1d(
                            &remap,
                            &[&schedule, &total, &slot_buffer].map(binding),
                            max_tiles,
                            128,
                        )?;
                        batch.dispatch_1d(
                            &reorder,
                            &[&order, &original_buffer, &pair_count].map(binding),
                            pairs,
                            128,
                        )?;
                        let mut bindings = vec![binding(&input)];
                        bindings.extend(gate.iter().map(binding));
                        bindings.extend(up.iter().take(6).map(binding));
                        bindings.extend(
                            [
                                &order,
                                &schedule,
                                &total,
                                &sigmoid,
                                &dims,
                                &activation,
                                &activation,
                                &activation,
                            ]
                            .map(binding),
                        );
                        batch.dispatch_1d(&gate_kernel, &bindings, max_tiles * 20 * 128, 128)?;
                        let mut bindings = vec![binding(&activation)];
                        bindings.extend(down.iter().map(binding));
                        bindings.extend([&order, &schedule, &total, &dims, &output].map(binding));
                        batch.dispatch_1d(&down_kernel, &bindings, max_tiles * 80 * 128, 128)?;
                    }
                    batch.submit().wait()?;
                    let mut expected_order = (0..pairs).collect::<Vec<_>>();
                    expected_order.sort_by_key(|&pair| ids[pair]);
                    let mut expected_schedule = Vec::new();
                    let mut start = 0;
                    for source in [0u32, 1, 3] {
                        let count = ids.iter().filter(|&&id| id == source).count();
                        for tile in 0..count.div_ceil(tile_tokens) {
                            expected_schedule.extend([
                                slots[source as usize],
                                (start + tile * tile_tokens) as u32,
                                (count - tile * tile_tokens).min(tile_tokens) as u32,
                            ]);
                        }
                        start += count;
                    }
                    let total = u32::from_le_bytes(read(&total)?.try_into().unwrap()) as usize;
                    assert_eq!(
                        total,
                        expected_schedule.len() / 3,
                        "{codec:?} tile {tile_tokens}"
                    );
                    assert_eq!(
                        read(&schedule)?[..total * 12],
                        expected_schedule
                            .iter()
                            .flat_map(|word| word.to_le_bytes())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        read(&order)?,
                        expected_order
                            .iter()
                            .flat_map(|&pair| (original[pair] as u32).to_le_bytes())
                            .collect::<Vec<_>>()
                    );
                    for (buffer, width) in [(&activation, SHARED), (&output, HIDDEN)] {
                        let bytes = read(buffer)?;
                        assert_eq!(&bytes[pairs * width * 2..], &[0x5a; 64]);
                        for (local, &global) in original.iter().enumerate() {
                            let slot = slots[ids[local] as usize] as f32 + 1.0;
                            let x = (global / ROUTES + 1) as f32 / 64.0;
                            let g = f16::from_f32(HIDDEN as f32 * x * slot / 2048.0);
                            let u = f16::from_f32(HIDDEN as f32 * x * slot / 4096.0);
                            let at = g.to_bits() as usize * 2;
                            let sigmoid = f16::from_bits(u16::from_le_bytes(
                                sigmoid_bytes[at..at + 2].try_into().unwrap(),
                            ));
                            let silu = f16::from_f32(g.to_f32() * sigmoid.to_f32());
                            let activation = f16::from_f32(silu.to_f32() * u.to_f32());
                            let expected = if width == SHARED {
                                activation.to_f32()
                            } else {
                                f16::from_f32(SHARED as f32 * activation.to_f32() * slot / 2048.0)
                                    .to_f32()
                            };
                            for column in 0..width {
                                let at = (global * width + column) * 2;
                                let value = f16::from_bits(u16::from_le_bytes(
                                    bytes[at..at + 2].try_into().unwrap(),
                                ))
                                .to_f32();
                                assert!(value.is_finite());
                                assert!(
                                    (value - expected).abs() <= 2e-5 + expected.abs() * 0.002,
                                    "{codec:?} tile {tile_tokens} width {width} pair {global} column {column}: {value} != {expected}"
                                );
                            }
                        }
                    }
                }
            }
            Ok(())
        }

        #[test]
        fn row_chunks_restore_global_original_pairs() -> Result<()> {
            let metal = Metal::new()?;
            let source = include_str!("../kernels/metal/mlp_full_resident.metal");
            let schedule = metal.compile_msl_strict(source, "mlp_full_schedule")?;
            let restore = metal.compile_msl_strict(source, "mlp_order_original_pairs")?;
            let ids: Vec<u32> = (0..10)
                .rev()
                .chain(0..10)
                .chain((10..20).rev())
                .chain((20..30).rev())
                .collect();
            let pair_order: Vec<_> = (0..ids.len()).collect();
            let chunks = demand_chunks(&ids, 12)?;
            assert!(chunks.len() > 1);
            let encoded = |values: &[u32]| -> Result<Buffer> {
                upload(
                    &metal,
                    &values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            };
            let ordered_ids = encoded(&pair_order.iter().map(|&p| ids[p]).collect::<Vec<_>>())?;
            let original_pairs = encoded(
                &pair_order
                    .iter()
                    .map(|&pair| pair as u32)
                    .collect::<Vec<_>>(),
            )?;
            let order = metal.shared_buffer(ids.len() * 4)?;
            let tiles = metal.shared_buffer((EXPERTS + ids.len().div_ceil(32)) * 3 * 4)?;
            let total = metal.shared_buffer(4)?;
            for chunk in chunks {
                let pair_range = packed_pair_range(&chunk, false);
                let start = pair_range.start;
                let end = pair_range.end;
                let pairs = end - start;
                let dims = encoded(&[pairs as u32, 0, 0, 0])?;
                let mut batch = metal.batch()?;
                // SAFETY: Every chunk is a complete set of valid rows and
                // the pair map spans the full original input.
                unsafe {
                    batch.dispatch_1d(
                        &schedule,
                        &[
                            Binding {
                                buffer: &ordered_ids,
                                offset: start * 4,
                            },
                            Binding {
                                buffer: &dims,
                                offset: 0,
                            },
                            Binding {
                                buffer: &order,
                                offset: 0,
                            },
                            Binding {
                                buffer: &tiles,
                                offset: 0,
                            },
                            Binding {
                                buffer: &total,
                                offset: 0,
                            },
                        ],
                        EXPERTS,
                        EXPERTS,
                    )?;
                    batch.dispatch_1d_with_inline(
                        &restore,
                        &[
                            Binding {
                                buffer: &order,
                                offset: 0,
                            },
                            Binding {
                                buffer: &original_pairs,
                                offset: start * 4,
                            },
                        ],
                        &[&(pairs as u32).to_le_bytes()],
                        pairs,
                        128,
                    )?;
                }
                batch.submit().wait()?;
                let mut mapped = vec![0; pairs * 4];
                order.read_bytes(0, &mut mapped)?;
                let mapped: Vec<_> = mapped
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes(*word) as usize)
                    .collect();
                let mut expected: Vec<_> = (start..end).collect();
                expected.sort_by_key(|&pair| ids[pair]);
                assert_eq!(mapped, expected);
            }
            Ok(())
        }

        #[test]
        fn source_groups_remap_slots_scatter_pairs_and_reduce_in_route_order() -> Result<()> {
            let metal = Metal::new()?;
            let schedule_source = format!(
                "{}\n{}",
                include_str!("../kernels/metal/mlp_full_resident.metal"),
                r#"kernel void mlp_test_pair_scatter(
                    const device uint* schedule [[buffer(0)]],
                    const device uint* order [[buffer(1)]],
                    const device uint* total [[buffer(2)]],
                    device half* down [[buffer(3)]],
                    uint tile [[thread_position_in_grid]]) {
                    if (tile >= total[0]) return;
                    uint slot = schedule[tile * 3u];
                    uint start = schedule[tile * 3u + 1u];
                    uint count = schedule[tile * 3u + 2u];
                    for (uint i = 0u; i < count; ++i) {
                        uint pair = order[start + i];
                        half value = half(float(slot) * 0.125f
                            + float(pair % 10u) * 0.03125f);
                        for (uint col = 0u; col < 2560u; ++col)
                            down[pair * 2560u + col] = value;
                    }
                }"#
            );
            let schedule = metal.compile_msl_strict(&schedule_source, "mlp_full_schedule")?;
            let remap = metal.compile_msl_strict(&schedule_source, "mlp_schedule_slots")?;
            let original =
                metal.compile_msl_strict(&schedule_source, "mlp_order_original_pairs")?;
            let scatter = metal.compile_msl_strict(&schedule_source, "mlp_test_pair_scatter")?;
            let route_order = metal.compile_msl_strict(&schedule_source, "mlp_full_route_order")?;
            let reduce_source = format!(
                "{}\n{}\n{}",
                include_str!("../kernels/metal/mlp.metal"),
                include_str!("../kernels/metal/expert_block.metal"),
                include_str!("../kernels/metal/mlp_batch.metal")
            );
            let reduce = metal.compile_msl_strict(&reduce_source, "mlp_batch_reduce")?;
            let ids: Vec<u32> = [
                [9, 4, 2, 6, 10, 12, 14, 16, 18, 20],
                [5, 3, 1, 7, 11, 13, 15, 17, 19, 21],
                [20, 18, 16, 14, 12, 10, 9, 6, 4, 2],
                [21, 19, 17, 15, 13, 11, 7, 5, 3, 1],
            ]
            .into_iter()
            .flatten()
            .collect();
            let (pair_order, groups) = sorted_chunks(&ids, 12)?;
            assert!(groups.len() > 2);
            let sorted_ids = pair_order.iter().map(|&pair| ids[pair]).collect::<Vec<_>>();
            let encoded = |values: &[u32]| -> Result<Buffer> {
                upload(
                    &metal,
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            };
            let selected = encoded(&ids)?;
            let sorted = encoded(&sorted_ids)?;
            let originals = encoded(
                &pair_order
                    .iter()
                    .map(|&pair| pair as u32)
                    .collect::<Vec<_>>(),
            )?;
            let order = metal.shared_buffer(ids.len() * 4)?;
            let tiles = metal.shared_buffer((EXPERTS + ids.len().div_ceil(32)) * 3 * 4)?;
            let total = metal.shared_buffer(4)?;
            let down = metal.shared_buffer(ids.len() * HIDDEN * 2)?;
            let source_order = metal.shared_buffer(ids.len() * 4)?;
            let scores = upload(
                &metal,
                &(0..ids.len())
                    .map(|pair| bf16::from_f32(((pair % ROUTES) + 1) as f32 * 0.125).to_bits())
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let routed = metal.shared_buffer(4 * HIDDEN * 2)?;
            let mut expected_slots = vec![u32::MAX; ids.len()];
            let mut batch = metal.batch()?;
            // SAFETY: Four complete rows have distinct route-order output.
            unsafe {
                batch.dispatch_1d(
                    &route_order,
                    &[&selected, &source_order].map(|buffer| Binding { buffer, offset: 0 }),
                    4,
                    4,
                )?;
            }
            for group in &groups {
                let pairs = group.end - group.start;
                let dims = encoded(&[pairs as u32, 0, 0, 0])?;
                let mut map = [u32::MAX; EXPERTS];
                for (rank, &source) in group.active.iter().enumerate() {
                    map[source as usize] = 11 - rank as u32;
                }
                for &pair in &pair_order[group.start..group.end] {
                    expected_slots[pair] = map[ids[pair] as usize];
                }
                let map_bytes = map
                    .iter()
                    .flat_map(|slot| slot.to_le_bytes())
                    .collect::<Vec<_>>();
                let count = (pairs as u32).to_le_bytes();
                let max_tiles = group.active.len() + pairs.div_ceil(32);
                // SAFETY: Each group has complete source IDs, the original
                // pair map is a permutation, and groups write disjoint pairs.
                unsafe {
                    batch.dispatch_1d(
                        &schedule,
                        &[
                            Binding {
                                buffer: &sorted,
                                offset: group.start * 4,
                            },
                            Binding {
                                buffer: &dims,
                                offset: 0,
                            },
                            Binding {
                                buffer: &order,
                                offset: 0,
                            },
                            Binding {
                                buffer: &tiles,
                                offset: 0,
                            },
                            Binding {
                                buffer: &total,
                                offset: 0,
                            },
                        ],
                        EXPERTS,
                        EXPERTS,
                    )?;
                    batch.dispatch_1d_with_inline(
                        &original,
                        &[
                            Binding {
                                buffer: &order,
                                offset: 0,
                            },
                            Binding {
                                buffer: &originals,
                                offset: group.start * 4,
                            },
                        ],
                        &[&count],
                        pairs,
                        128,
                    )?;
                    batch.dispatch_1d_with_inline(
                        &remap,
                        &[
                            Binding {
                                buffer: &tiles,
                                offset: 0,
                            },
                            Binding {
                                buffer: &total,
                                offset: 0,
                            },
                        ],
                        &[&map_bytes],
                        max_tiles,
                        128,
                    )?;
                    batch.dispatch_1d(
                        &scatter,
                        &[&tiles, &order, &total, &down]
                            .map(|buffer| Binding { buffer, offset: 0 }),
                        max_tiles,
                        128,
                    )?;
                }
            }
            // SAFETY: Every original pair was written once before reduction.
            unsafe {
                batch.dispatch_1d(
                    &reduce,
                    &[&down, &scores, &source_order, &routed]
                        .map(|buffer| Binding { buffer, offset: 0 }),
                    4 * HIDDEN,
                    256,
                )?;
            }
            batch.submit().wait()?;
            assert!(expected_slots.iter().all(|&slot| slot < 12));
            assert!(
                expected_slots
                    .iter()
                    .enumerate()
                    .any(|(pair, &slot)| slot != ids[pair])
            );
            for (pair, &slot) in expected_slots.iter().enumerate() {
                let mut bytes = [0u8; 2];
                down.read_bytes(pair * HIDDEN * 2, &mut bytes)?;
                let value = half::f16::from_bits(u16::from_le_bytes(bytes)).to_f32();
                let expected = slot as f32 * 0.125 + (pair % ROUTES) as f32 * 0.03125;
                assert_eq!(
                    value, expected,
                    "original pair {pair} was not scattered from its slot"
                );
            }
            let round = |value| bf16::from_f32(value).to_f32();
            for row in 0..4 {
                let mut routes = (0..ROUTES).collect::<Vec<_>>();
                routes.sort_by_key(|&route| ids[row * ROUTES + route]);
                let mut expected = 0.0f32;
                for route in routes {
                    let pair = row * ROUTES + route;
                    let value = expected_slots[pair] as f32 * 0.125 + route as f32 * 0.03125;
                    let score = (route + 1) as f32 * 0.125;
                    expected = round(expected + round(round(value) * score));
                }
                let mut bytes = [0u8; 2];
                routed.read_bytes(row * HIDDEN * 2, &mut bytes)?;
                assert_eq!(
                    u16::from_le_bytes(bytes),
                    bf16::from_f32(expected).to_bits(),
                    "row {row} used a different source reduction order"
                );
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{
    GroupDemand, MlpBatch, MlpBatchCounts, MlpBatchOutput, MlpBatchScratch, MlpBatchStages,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_words(rows: usize, words: &[u16]) -> Result<Vec<(usize, usize)>> {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let mut reads = Vec::new();
        validate_projected_logits(rows, |offset, destination| {
            destination.copy_from_slice(&bytes[offset..offset + destination.len()]);
            reads.push((offset, destination.len()));
            Ok(())
        })?;
        Ok(reads)
    }

    #[test]
    fn projected_logits_scan_exact_rows_and_reject_nonfinite_positions() {
        for rows in [2, 512, 1024, 2048, 4096] {
            let count = rows * EXPERTS;
            let mut words = vec![0u16; count];
            let reads = scan_words(rows, &words).unwrap();
            assert_eq!(
                reads.first(),
                Some(&(0, (count * 2).min(LOGITS_READBACK_BYTES)))
            );
            assert_eq!(reads.last().unwrap().0 + reads.last().unwrap().1, count * 2);
            assert!(
                reads
                    .iter()
                    .all(|(_, length)| *length <= LOGITS_READBACK_BYTES)
            );
            for (position, invalid) in [(0, 0x7f80), (count / 2, 0xff80), (count - 1, 0x7fc1)] {
                words[position] = invalid;
                let error = scan_words(rows, &words).unwrap_err();
                assert_eq!(error.to_string(), "routing requires finite BF16 logits");
                words[position] = 0;
            }
        }
    }

    #[test]
    fn projected_logits_scan_rejects_invalid_geometry_before_read() {
        for rows in [0, 1, 4097, usize::MAX] {
            let mut called = false;
            assert!(
                validate_projected_logits(rows, |_, _| {
                    called = true;
                    Ok(())
                })
                .is_err()
            );
            assert!(!called);
        }
    }

    #[test]
    fn full_residency_requires_every_identity_slot() {
        let mut map: Vec<u32> = (0..512).collect();
        assert!(require_full_identity(512, &map).is_ok());
        assert!(require_full_identity(511, &map).is_err());
        assert!(require_full_identity(512, &map[..511]).is_err());
        map[200] = 512;
        assert!(require_full_identity(512, &map).is_err());
        map[200] = 200;
        map.swap(7, 200);
        assert!(require_full_identity(512, &map).is_err());
    }
    #[test]
    fn rows_reject_singleton_and_overflow() {
        for rows in [2, 3, 64, 409, 410, 512, 1024, 2048, 2049, 4096] {
            assert!(validate_rows(rows).is_ok());
        }
        for rows in [0, 1, 4097, usize::MAX] {
            assert!(validate_rows(rows).is_err());
        }
    }
    #[test]
    fn wide_source_groups_preserve_every_pair_and_visit_each_source_once() {
        let rows = 4096;
        let ids: Vec<_> = (0..rows)
            .flat_map(|row| {
                (0..ROUTES).map(move |route| ((row * 13 + route * 37) % EXPERTS) as u32)
            })
            .collect();
        for capacity in [64, 188, 189, 202, 511] {
            let (order, chunks, sorted) = planned_chunks(&ids, capacity).unwrap();
            assert!(sorted);
            assert_eq!(order.len(), rows * ROUTES);
            let mut pairs = order.clone();
            pairs.sort_unstable();
            assert_eq!(pairs, (0..rows * ROUTES).collect::<Vec<_>>());
            let mut sources = BTreeSet::new();
            let mut end = 0;
            for chunk in &chunks {
                assert_eq!(chunk.start, end);
                assert!(chunk.active.len() <= capacity / 2);
                assert!(sources.is_disjoint(&chunk.active));
                sources.extend(chunk.active.iter().copied());
                assert!(
                    order[chunk.start..chunk.end]
                        .iter()
                        .all(|&pair| chunk.active.contains(&ids[pair]))
                );
                end = chunk.end;
            }
            assert_eq!(end, ids.len());
            assert_eq!(sources.len(), EXPERTS);
            assert_eq!(chunks.len(), EXPERTS.div_ceil(capacity / 2));
        }
    }
    #[test]
    fn sorted_groups_bound_sources_without_splitting_an_expert() {
        let ids: Vec<_> = (0..10).rev().chain(5..15).collect();
        let (order, groups) = sorted_chunks(&ids, 12).unwrap();
        assert!(order.windows(2).all(|w| ids[w[0]] <= ids[w[1]]));
        assert_eq!(
            groups.iter().map(|g| g.active.len()).collect::<Vec<_>>(),
            [6, 6, 3]
        );
        assert_eq!(groups.first().unwrap().start, 0);
        assert_eq!(groups.last().unwrap().end, ids.len());
        for group in &groups {
            assert_eq!(
                order[group.start..group.end]
                    .iter()
                    .map(|&p| ids[p])
                    .collect::<BTreeSet<_>>(),
                group.active
            );
        }
        let (_, direct) = sorted_chunks(&ids, 20).unwrap();
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0].active.len(), 15);
    }
    #[test]
    fn grouping_touches_a_repeated_union_once() {
        let ids: Vec<_> = (0..10).chain(0..10).chain(10..20).chain(10..20).collect();
        let groups = demand_chunks(&ids, 12).unwrap();
        assert_eq!(
            groups
                .iter()
                .map(|c| (c.start, c.end, c.active.len()))
                .collect::<Vec<_>>(),
            [(0, 2, 10), (2, 4, 10)]
        );
    }
    #[test]
    fn row_chunk_pair_map_preserves_global_scatter_positions() {
        let ids: Vec<_> = (0..10).chain(0..10).chain(10..20).chain(20..30).collect();
        let order: Vec<_> = (0..ids.len()).collect();
        let chunks = demand_chunks(&ids, 12).unwrap();
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| (chunk.start, chunk.end))
                .collect::<Vec<_>>(),
            [(0, 2), (2, 3), (3, 4)]
        );
        let mut scattered = vec![None; ids.len()];
        for chunk in chunks {
            let pair_range = packed_pair_range(&chunk, false);
            let start = pair_range.start;
            let end = pair_range.end;
            for (local, &original) in order[start..end].iter().enumerate() {
                assert!(original >= start && original < end);
                assert!(scattered[original].replace(ids[start + local]).is_none());
            }
        }
        assert_eq!(scattered, ids.into_iter().map(Some).collect::<Vec<_>>());
    }

    #[test]
    fn physical_chunk_filter_preserves_source_tiles_and_global_pairs() {
        let sources = [7, 31, 4, 100, 511, 90, 55, 42, 88, 201, 320, 2];
        let slots = [63, 64, 92, 2, 66, 5, 0, 80, 60, 70, 61, 91];
        let mut map = [EXPERTS as u32; EXPERTS];
        for (&source, &slot) in sources.iter().zip(&slots) {
            map[source as usize] = slot;
        }
        let ranges = [0..64, 64..93];
        for rows in [17, 512] {
            let ids: Vec<_> = (0..rows)
                .flat_map(|row| (0..ROUTES).map(move |route| sources[(row * 7 + route) % 12]))
                .collect();
            let (order, groups, sorted) = planned_chunks(&ids, 93).unwrap();
            assert_eq!(groups.len(), 1);
            assert_eq!(sorted, rows == 512);
            let ordered_ids: Vec<_> = order.iter().map(|&pair| ids[pair]).collect();
            let children =
                partition_packed_group(&ordered_ids, &order, &map, 93, &ranges, ids.len()).unwrap();
            assert_eq!(children.len(), 2);
            assert_eq!(children.iter().map(|c| c.live_sources).sum::<usize>(), 12);
            let mut scattered = vec![None; ids.len()];
            for child in &children {
                let range = &ranges[child.chunk];
                for (&source, &original) in child.ids.iter().zip(&child.original_pairs) {
                    assert_eq!(ids[original as usize], source);
                    assert!(scattered[original as usize].replace(source).is_none());
                    assert_eq!(
                        child.slot_map[source as usize] as usize,
                        map[source as usize] as usize - range.start
                    );
                    assert!((child.slot_map[source as usize] as usize) < range.len());
                }
                for (source, &local) in child.slot_map.iter().enumerate() {
                    if !child.ids.contains(&(source as u32)) {
                        assert_eq!(local, EXPERTS as u32);
                    }
                }
            }
            assert_eq!(scattered, ids.iter().copied().map(Some).collect::<Vec<_>>());
            for source in sources {
                let parent: Vec<_> = ordered_ids
                    .iter()
                    .zip(&order)
                    .filter_map(|(&id, &pair)| (id == source).then_some(pair as u32))
                    .collect();
                let child = children
                    .iter()
                    .find(|child| child.slot_map[source as usize] != EXPERTS as u32)
                    .unwrap();
                let filtered: Vec<_> = child
                    .ids
                    .iter()
                    .zip(&child.original_pairs)
                    .filter_map(|(&id, &pair)| (id == source).then_some(pair))
                    .collect();
                assert_eq!(filtered, parent);
                assert_eq!(
                    filtered.chunks(32).collect::<Vec<_>>(),
                    parent.chunks(32).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn physical_chunk_filter_handles_short_tails_and_skips_empty_chunks() {
        let ids: Vec<_> = (0..ROUTES as u32).rev().cycle().take(20).collect();
        let original: Vec<_> = (100..120).collect();
        for capacity in [12, 13, 64, 65, 93, 178] {
            let ranges: Vec<_> = (0..capacity)
                .step_by(64)
                .map(|start| start..(start + 64).min(capacity))
                .collect();
            let mut map = [EXPERTS as u32; EXPERTS];
            for (source, slot) in map.iter_mut().take(ROUTES).enumerate() {
                *slot = (capacity - 1 - source) as u32;
            }
            let children =
                partition_packed_group(&ids, &original, &map, capacity, &ranges, 120).unwrap();
            let mut restored = Vec::new();
            for child in children {
                assert!(!child.ids.is_empty());
                for (&source, &pair) in child.ids.iter().zip(&child.original_pairs) {
                    let global = map[source as usize] as usize;
                    assert_eq!(child.chunk, global / 64);
                    assert_eq!(child.slot_map[source as usize] as usize, global % 64);
                    assert!((global % 64) < ranges[child.chunk].len());
                    restored.push(pair);
                }
            }
            restored.sort_unstable();
            assert_eq!(restored, (100..120).collect::<Vec<_>>());
        }
    }

    #[test]
    fn physical_chunk_filter_rejects_invalid_ranges_maps_and_pairs() {
        let mut map = [EXPERTS as u32; EXPERTS];
        map[7] = 11;
        let valid = [0..8, 8..12];
        for ranges in [
            vec![],
            std::iter::once(0..8).collect(),
            vec![0..8, 9..12],
            vec![0..8, 7..12],
            std::iter::once(0..13).collect(),
            vec![0..0, 0..12],
        ] {
            assert!(partition_packed_group(&[7], &[20], &map, 12, &ranges, 21).is_err());
        }
        assert!(partition_packed_group(&[], &[], &map, 12, &valid, 21).is_err());
        assert!(partition_packed_group(&[7], &[], &map, 12, &valid, 21).is_err());
        assert!(partition_packed_group(&[7], &[21], &map, 12, &valid, 21).is_err());
        assert!(partition_packed_group(&[512], &[20], &map, 12, &valid, 21).is_err());
        assert!(partition_packed_group(&[8], &[20], &map, 12, &valid, 21).is_err());
        map[7] = 12;
        assert!(partition_packed_group(&[7], &[20], &map, 12, &valid, 21).is_err());
        map[7] = 11;
        map[8] = 11;
        assert!(partition_packed_group(&[7, 8], &[19, 20], &map, 12, &valid, 21).is_err());
    }

    #[test]
    fn small_prefill_deduplicates_sources_only_when_row_unions_exceed_capacity() {
        let capacity = 125;
        for rows in [20, 171] {
            let selected: Vec<_> = (0..rows)
                .flat_map(|row| {
                    (0..ROUTES).map(move |route| ((row * 37 + route * 11) % EXPERTS) as u32)
                })
                .collect();
            let sources: BTreeSet<_> = selected.iter().copied().collect();
            assert!(demand_chunks(&selected, capacity).unwrap().len() > 1);
            let (order, groups, sorted) = planned_chunks(&selected, capacity).unwrap();
            assert!(
                sorted,
                "over-capacity {rows}-row prefill must deduplicate sources"
            );
            assert_eq!(
                groups.iter().map(|group| group.active.len()).sum::<usize>(),
                sources.len()
            );
            assert!(
                groups
                    .iter()
                    .all(|group| group.active.len() <= capacity / 2)
            );
            let mut covered = vec![false; selected.len()];
            for group in &groups {
                let pairs = packed_pair_range(group, sorted);
                let grouped = &order[pairs];
                assert!(
                    grouped
                        .windows(2)
                        .all(|pair| selected[pair[0]] <= selected[pair[1]])
                );
                for &original in grouped {
                    assert!(!std::mem::replace(&mut covered[original], true));
                }
                for source in &group.active {
                    let original: Vec<_> = grouped
                        .iter()
                        .copied()
                        .filter(|&pair| selected[pair] == *source)
                        .collect();
                    assert!(original.windows(2).all(|pair| pair[0] < pair[1]));
                }
            }
            assert!(covered.into_iter().all(|covered| covered));

            let fit: Vec<_> = (0..rows)
                .flat_map(|row| (0..ROUTES).map(move |route| ((row * 13 + route * 17) % 64) as u32))
                .collect();
            let (order, groups, sorted) = planned_chunks(&fit, capacity).unwrap();
            assert!(
                !sorted,
                "whole-union-fit {rows}-row prefill keeps row order"
            );
            assert_eq!(groups.len(), 1);
            assert_eq!(order, (0..fit.len()).collect::<Vec<_>>());
            assert_eq!(packed_pair_range(&groups[0], sorted), 0..fit.len());
        }
    }

    #[test]
    fn large_prefill_plan_uses_sorted_groups_and_whole_active_fit() {
        let ids: Vec<_> = (0..410)
            .flat_map(|row| (0..ROUTES).map(move |route| ((row + route) % 32) as u32))
            .collect();
        let (order, chunks, sorted) = planned_chunks(&ids, 12).unwrap();
        assert!(sorted);
        assert!(order.windows(2).all(|pair| ids[pair[0]] <= ids[pair[1]]));
        assert!(chunks.iter().all(|chunk| chunk.active.len() <= 6));
        let (_, fit, sorted) = planned_chunks(&ids, 32).unwrap();
        assert!(sorted);
        assert_eq!(fit.len(), 1);
        assert_eq!(fit[0].active.len(), 32);
    }
    #[test]
    fn prefill_plans_keep_row_and_sorted_pair_range_units() {
        let ids: Vec<_> = (0..410)
            .flat_map(|row| (0..ROUTES).map(move |route| ((row + route) % 32) as u32))
            .collect();
        for (rows, capacity, expected_sorted) in [
            (409, 12, true),
            (410, 12, true),
            (409, 32, false),
            (410, 32, true),
        ] {
            let selected = &ids[..rows * ROUTES];
            assert_eq!(selected.iter().copied().collect::<BTreeSet<_>>().len(), 32);
            let (order, chunks, sorted) = planned_chunks(selected, capacity).unwrap();
            assert_eq!(sorted, expected_sorted);
            assert_eq!(order.len(), selected.len());
            let mut seen = vec![false; selected.len()];
            for chunk in &chunks {
                let pairs = packed_pair_range(chunk, sorted);
                assert!(pairs.start < pairs.end && pairs.end <= selected.len());
                if sorted {
                    assert!(
                        chunk.active.len()
                            <= if chunks.len() == 1 {
                                capacity
                            } else {
                                capacity / 2
                            }
                    );
                } else {
                    assert_eq!(pairs, chunk.start * ROUTES..chunk.end * ROUTES);
                    assert!(chunk.active.len() <= capacity);
                }
                for &original in &order[pairs] {
                    assert!(!std::mem::replace(&mut seen[original], true));
                }
            }
            assert!(seen.into_iter().all(|covered| covered));
            assert_eq!(
                packed_pair_range(chunks.last().unwrap(), sorted).end,
                selected.len()
            );
        }
    }
    #[test]
    fn grouping_rejects_invalid_rows_and_capacity() {
        for ids in [vec![], vec![0; 10], (0..9).collect(), (505..515).collect()] {
            assert!(demand_chunks(&ids, 12).is_err());
        }
        for capacity in [0, 9, 512, usize::MAX] {
            assert!(demand_chunks(&(0..10).collect::<Vec<_>>(), capacity).is_err());
        }
    }
}

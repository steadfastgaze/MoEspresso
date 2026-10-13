//! One bounded Qwen routed expert block with atomic slot publication.
//!
//! The three projection streams share physical slot numbers, while routing
//! continues to use package source expert IDs. A slot becomes addressable only
//! after its complete package row has filled all three projection buffers.

use half::f16;

#[cfg(target_os = "macos")]
mod chunks;

/// Sigmoid rounded to FP16 for each possible FP16 gate bit pattern.
pub fn sigmoid_fp16_table() -> Vec<u8> {
    let mut table = Vec::with_capacity(1 << 17);
    for bits in 0..=u16::MAX {
        let gate = f16::from_bits(bits).to_f32();
        if gate.is_nan() {
            table.extend_from_slice(&0x7e00u16.to_le_bytes());
            continue;
        }
        let exponential = f16::from_f32(gate.abs().exp()).to_f32();
        let denominator = f16::from_f32(1.0 + exponential).to_f32();
        let y = f16::from_f32(1.0 / denominator).to_f32();
        let sigmoid = if gate < 0.0 {
            y
        } else {
            f16::from_f32(1.0 - y).to_f32()
        };
        table.extend_from_slice(&f16::from_f32(sigmoid).to_bits().to_le_bytes());
    }
    table
}

#[cfg(target_os = "macos")]
mod gpu {
    use std::collections::{BTreeMap, BTreeSet};

    use anyhow::{Context, Result, ensure};
    use half::{bf16, f16};

    use super::chunks::FixedChunks;
    use super::chunks::PreparedChunksGrowth;
    use crate::iqk::{Codec, GemvDescriptor, Geometry, IqkGemv};
    use crate::metal::{Batch, Binding, Buffer, Completion, InFlight, Metal, Pipeline, Stream};
    use crate::package::{ExpertLayer, Projection};

    // Contiguous storage retains its existing inline projection ownership.
    #[allow(clippy::large_enum_variant)]
    enum ExpertStorage {
        Contiguous {
            gate: IqkGemv,
            up: IqkGemv,
            down: IqkGemv,
        },
        Chunked(FixedChunks),
    }

    const EXPERT_SOURCE: &str = include_str!("../kernels/metal/expert_block.metal");
    const ROUTED_SOURCE: &str = include_str!("../kernels/metal/iqk_routed.metal");
    const TOP_K: usize = 10;
    const HIDDEN: usize = 2560;
    const INTERMEDIATE: usize = 640;
    const DOWN_STORED: usize = 768;

    fn buffer(metal: &Metal, len: usize) -> Result<Buffer> {
        metal.shared_buffer(len)
    }

    fn least_frequent_slot(
        expert_at: &[Option<u32>],
        last_use: &[u64],
        frequency: &BTreeMap<u32, u64>,
        protected: &BTreeSet<u32>,
    ) -> Option<usize> {
        expert_at
            .iter()
            .enumerate()
            .filter_map(|(slot, &source)| {
                source.filter(|id| !protected.contains(id)).map(|id| {
                    (
                        slot,
                        (frequency.get(&id).copied().unwrap_or(0), last_use[slot]),
                    )
                })
            })
            .min_by_key(|(_, priority)| *priority)
            .map(|(slot, _)| slot)
    }

    fn publish_replacement(
        slot_of: &mut BTreeMap<u32, usize>,
        expert_at: &mut [Option<u32>],
        source: u32,
        slot: usize,
        write: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        {
            if let Some(previous) = expert_at[slot].take() {
                slot_of.remove(&previous);
            }
        }
        {
            write()?;
        }
        {
            expert_at[slot] = Some(source);
            slot_of.insert(source, slot);
        }
        Ok(())
    }

    struct Scratch {
        hidden: Buffer,
        slots: Buffer,
        gate_dims: Buffer,
        down_dims: Buffer,
        gate: Buffer,
        up: Buffer,
        activation: Buffer,
        down: Buffer,
        scores: Buffer,
        order: Buffer,
        output: Buffer,
        sigmoid_table: Buffer,
    }

    pub struct PendingOutput {
        flight: InFlight,
        gate: Buffer,
        up: Buffer,
        activation: Buffer,
        down: Buffer,
        output: Buffer,
    }

    pub struct RoutedStages {
        pub gate_fp16: Vec<u8>,
        pub up_fp16: Vec<u8>,
        pub activation_fp16_padded: Vec<u8>,
        pub down_fp16: Vec<u8>,
        pub final_bf16: Vec<u8>,
    }

    impl PendingOutput {
        /// Wait for the entire block and return one BF16 hidden row.
        pub fn wait(self) -> Result<Vec<u8>> {
            let Self { flight, output, .. } = self;
            flight.wait()?;
            let mut bytes = vec![0; HIDDEN * 2];
            output.read_bytes(0, &mut bytes)?;
            Ok(bytes)
        }

        pub fn wait_stages(self) -> Result<RoutedStages> {
            let Self {
                flight,
                gate,
                up,
                activation,
                down,
                output,
            } = self;
            flight.wait()?;
            fn read(buffer: Buffer) -> Result<Vec<u8>> {
                let mut bytes = vec![0; buffer.byte_len()];
                buffer.read_bytes(0, &mut bytes)?;
                Ok(bytes)
            }
            Ok(RoutedStages {
                gate_fp16: read(gate)?,
                up_fp16: read(up)?,
                activation_fp16_padded: read(activation)?,
                down_fp16: read(down)?,
                final_bf16: read(output)?,
            })
        }
    }

    pub struct BoundedExperts {
        metal: Metal,
        layer: ExpertLayer,
        row_workspace: crate::package::ExpertRowWorkspace,
        capacity: usize,
        storage: ExpertStorage,
        chunked_pipelines: Option<(Pipeline, Pipeline)>,
        activation_pipeline: Pipeline,
        reduction_pipeline: Pipeline,
        input_fp16_pipeline: Pipeline,
        resident_routes_pipeline: Pipeline,
        gate_up_pipeline: Pipeline,
        down_reduce_pipeline: Pipeline,
        bounded_pipelines: Option<(Pipeline, Pipeline)>,
        scratch: Scratch,
        slot_of: BTreeMap<u32, usize>,
        expert_at: Vec<Option<u32>>,
        last_use: Vec<u64>,
        frequency: BTreeMap<u32, u64>,
        clock: u64,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ExpertPoolSnapshot {
        pub clock: u64,
        pub slot_of: BTreeMap<u32, usize>,
        pub expert_at: Vec<Option<u32>>,
        pub frequency: BTreeMap<u32, u64>,
        pub last_use: Vec<u64>,
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct ExpertDemandCounts {
        pub hits: usize,
        pub misses: usize,
        pub evictions: usize,
    }

    /// Whole suffix allocations released at a completed inference boundary.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ExpertReleaseReport {
        pub old_capacity: usize,
        pub new_capacity: usize,
        pub released_requested_bytes: u64,
        pub evicted_sources: usize,
    }

    /// Net weight growth and the complete additional preparation peak.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ExpertGrowthReport {
        pub old_capacity: usize,
        pub new_capacity: usize,
        pub added_requested_bytes: u64,
        /// New suffix slabs and the replacement address table while the old
        /// address table remains held.
        pub preparation_requested_bytes: u64,
    }

    pub(crate) struct PreparedExpertSuffixGrowth<'a> {
        pool: &'a mut BoundedExperts,
        growth: PreparedChunksGrowth,
        report: ExpertGrowthReport,
    }

    impl PreparedExpertSuffixGrowth<'_> {
        pub(crate) fn report(&self) -> ExpertGrowthReport {
            self.report
        }

        pub(crate) fn commit(self) -> ExpertGrowthReport {
            let Self {
                pool,
                growth,
                report,
            } = self;
            let ExpertStorage::Chunked(chunks) = &mut pool.storage else {
                unreachable!("prepared expert growth lost fixed chunk storage");
            };
            chunks.commit_growth(growth);
            pool.expert_at.resize(report.new_capacity, None);
            pool.last_use.resize(report.new_capacity, 0);
            pool.capacity = report.new_capacity;
            report
        }
    }

    /// An exclusive validated pool borrow. Dropping preparation changes nothing.
    pub(crate) struct PreparedExpertSuffixRelease<'a> {
        pool: &'a mut BoundedExperts,
        keep: usize,
        report: ExpertReleaseReport,
    }

    impl PreparedExpertSuffixRelease<'_> {
        pub(crate) fn report(&self) -> ExpertReleaseReport {
            self.report
        }

        pub(crate) fn commit(self) -> ExpertReleaseReport {
            let Self { pool, keep, report } = self;
            let ExpertStorage::Chunked(chunks) = &mut pool.storage else {
                unreachable!("prepared expert release lost fixed chunk storage");
            };
            pool.slot_of.retain(|_, slot| *slot < report.new_capacity);
            pool.expert_at.truncate(report.new_capacity);
            pool.last_use.truncate(report.new_capacity);
            chunks.commit_suffix(keep, report.new_capacity);
            pool.capacity = report.new_capacity;
            report
        }
    }

    struct PrefillGroup {
        sources: BTreeSet<u32>,
        slots: BTreeSet<usize>,
        map: [u32; 512],
        completion: Option<Completion>,
    }

    impl PrefillGroup {
        fn capture(
            sources: &BTreeSet<u32>,
            published: &[u32; 512],
            capacity: usize,
        ) -> Result<Self> {
            ensure!(!sources.is_empty(), "prefill reader group is empty");
            let mut map = [512; 512];
            let mut slots = BTreeSet::new();
            for &source in sources {
                let slot = *published
                    .get(source as usize)
                    .context("prefill source is invalid")?;
                ensure!(
                    (slot as usize) < capacity && slots.insert(slot as usize),
                    "prefill reader map contains missing or aliased slots"
                );
                map[source as usize] = slot;
            }
            Ok(Self {
                sources: sources.clone(),
                slots,
                map,
                completion: None,
            })
        }
    }

    /// Exclusive pool/stream ownership for the bounded prefill overlap loop.
    /// Readers are recorded before encoding, including unsubmitted work.
    pub(crate) struct PrefillLoadScope<'a> {
        pool: &'a mut BoundedExperts,
        stream: &'a mut Stream,
        groups: Vec<PrefillGroup>,
        finished: bool,
        failed: bool,
    }

    impl<'a> PrefillLoadScope<'a> {
        pub(crate) fn new(pool: &'a mut BoundedExperts, stream: &'a mut Stream) -> Result<Self> {
            pool.check_writable()?;
            Ok(Self {
                pool,
                stream,
                groups: Vec::with_capacity(2),
                finished: false,
                failed: false,
            })
        }

        fn step<T>(&mut self, action: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
            ensure!(!self.failed, "prefill scope has already failed");
            let result = action(self);
            if result.is_err() {
                self.failed = true;
                // Preserve the initiating error, but drain even if a submitted
                // command also fails. Protection remains alive during cleanup.
                let _ = self.stream.abort_and_wait();
            }
            result
        }

        /// Retire the previous group while keeping the current one outstanding.
        pub(crate) fn retire_previous(&mut self) -> Result<()> {
            self.step(|scope| {
                if scope.groups.len() == 2 {
                    let marker = scope.groups[0]
                        .completion
                        .context("prefill previous group is unsubmitted")?;
                    {
                        scope.stream.wait_through(marker)?;
                    }
                    scope.groups.remove(0);
                }
                Ok(())
            })
        }

        pub(crate) fn load_group(&mut self, active: &BTreeSet<u32>) -> Result<()> {
            self.step(|scope| {
                ensure!(
                    scope.groups.len() < 2
                        && scope.groups.iter().all(|group| group.completion.is_some()),
                    "prefill load requires submitted current and retired previous groups"
                );
                let protect = scope
                    .groups
                    .iter()
                    .flat_map(|group| group.sources.iter().copied())
                    .collect();
                let slots = scope
                    .groups
                    .iter()
                    .flat_map(|group| group.slots.iter().copied())
                    .collect();
                // SAFETY: Entry excluded prior readers, the scope owns both pool
                // and stream, and all subsequent readers use the captured groups.
                // Unretired physical slots are independently excluded by loader.
                unsafe {
                    scope.pool.ensure_group_disjoint(active, &protect, &slots)?;
                }
                scope.groups.push(PrefillGroup::capture(
                    active,
                    &scope.pool.residency_map()?,
                    scope.pool.capacity,
                )?);
                Ok(())
            })
        }

        /// Encode and submit only the current group's captured slot footprint.
        ///
        /// # Safety
        /// The encoder must read pool projections only at the captured map's
        /// non-sentinel slots for this group's sources. It must not write pool
        /// storage, retain an encoder outside the call, or introduce a reader
        /// through another stream. This is the packed shader footprint boundary.
        pub(crate) unsafe fn submit_group(
            &mut self,
            encode: impl FnOnce(&mut Batch, &BoundedExperts, &[u32; 512]) -> Result<()>,
        ) -> Result<()> {
            self.step(|scope| {
                let group = scope
                    .groups
                    .last_mut()
                    .context("prefill has no loaded group")?;
                ensure!(
                    group.completion.is_none(),
                    "prefill group was already submitted"
                );
                encode(scope.stream.batch()?, scope.pool, &group.map)?;
                group.completion = Some(scope.stream.flush_marked()?);
                Ok(())
            })
        }

        /// Revoke disjoint writes; normal allocation guards retain final readers.
        pub(crate) fn finish(mut self) -> Result<()> {
            ensure!(
                !self.failed && self.groups.iter().all(|group| group.completion.is_some()),
                "prefill cannot finish with an unsubmitted reader"
            );
            self.finished = true;
            Ok(())
        }
    }

    impl Drop for PrefillLoadScope<'_> {
        fn drop(&mut self) {
            if !self.finished {
                // Keep group protection and the pool borrow until every
                // submitted command completes, even after a command error.
                let _ = self.stream.abort_and_wait();
            }
        }
    }

    impl BoundedExperts {
        pub fn new(metal: &Metal, layer: ExpertLayer, capacity: usize) -> Result<Self> {
            let workspace = crate::package::ExpertRowWorkspace::new(layer.row_bytes)?;
            Self::with_workspace_contiguous(metal, layer, capacity, workspace)
        }

        pub(crate) fn with_workspace(
            metal: &Metal,
            layer: ExpertLayer,
            capacity: usize,
            row_workspace: crate::package::ExpertRowWorkspace,
        ) -> Result<Self> {
            Self::with_storage(metal, layer, capacity, row_workspace, capacity < 512)
        }

        pub(crate) fn with_workspace_contiguous(
            metal: &Metal,
            layer: ExpertLayer,
            capacity: usize,
            row_workspace: crate::package::ExpertRowWorkspace,
        ) -> Result<Self> {
            Self::with_storage(metal, layer, capacity, row_workspace, false)
        }

        fn with_storage(
            metal: &Metal,
            layer: ExpertLayer,
            capacity: usize,
            row_workspace: crate::package::ExpertRowWorkspace,
            chunked: bool,
        ) -> Result<Self> {
            drop(row_workspace.row(layer.row_bytes)?);
            ensure!(
                capacity >= TOP_K && capacity <= layer.num_experts,
                "bounded IQ_K capacity must fit top-10 and package expert count"
            );
            let gate_g = Geometry::from_projection(layer.projection(Projection::Gate))?;
            let up_g = Geometry::from_projection(layer.projection(Projection::Up))?;
            let down_g = Geometry::from_projection(layer.projection(Projection::Down))?;
            ensure!(
                gate_g.in_features() == HIDDEN
                    && layer.projection(Projection::Gate).logical_in_features == HIDDEN
                    && layer.projection(Projection::Up).logical_in_features == HIDDEN
                    && gate_g.out_features() == INTERMEDIATE
                    && up_g == gate_g
                    && down_g.in_features() == DOWN_STORED
                    && down_g.out_features() == HIDDEN,
                "bounded Qwen expert projection geometry differs from released dimensions"
            );
            ensure!(
                matches!(
                    (gate_g.codec(), down_g.codec()),
                    (Codec::Iq2K, Codec::Iq2K)
                        | (Codec::Iq3K, Codec::Iq2K)
                        | (Codec::Iq3K, Codec::Iq3K)
                ),
                "unsupported bounded Qwen expert codec tuple"
            );
            ensure!(
                layer.projection(Projection::Down).logical_in_features == INTERMEDIATE,
                "bounded Qwen down projection requires logical width 640"
            );
            let storage = if chunked {
                ExpertStorage::Chunked(FixedChunks::new(metal, [gate_g, up_g, down_g], capacity)?)
            } else {
                ExpertStorage::Contiguous {
                    gate: IqkGemv::empty_slots(metal, gate_g, capacity)?,
                    up: IqkGemv::empty_slots(metal, up_g, capacity)?,
                    down: IqkGemv::empty_slots(metal, down_g, capacity)?,
                }
            };
            let chunked_pipelines = if chunked {
                let source = format!(
                    "{}\n{}",
                    ROUTED_SOURCE,
                    include_str!("../kernels/metal/iqk_chunked.metal")
                );
                let suffix = |codec| match codec {
                    Codec::Iq2K => "iq2",
                    Codec::Iq3K => "iq3",
                };
                Some((
                    metal.compile_msl_strict(
                        &source,
                        &format!("chunk_gate_{}", suffix(gate_g.codec())),
                    )?,
                    metal.compile_msl_strict(
                        &source,
                        &format!("chunk_down_{}", suffix(down_g.codec())),
                    )?,
                ))
            } else {
                None
            };
            let activation_pipeline = metal.compile_msl(EXPERT_SOURCE, "swiglu_10")?;
            let reduction_pipeline = metal.compile_msl(EXPERT_SOURCE, "reduce_10_bf16")?;
            let input_fp16_pipeline =
                metal.compile_msl_strict(ROUTED_SOURCE, "qwen4_routed_input_fp16")?;
            let resident_routes_pipeline =
                metal.compile_msl_strict(EXPERT_SOURCE, "prepare_resident_routes")?;
            let gate_up_pipeline = metal.compile_msl_strict(
                ROUTED_SOURCE,
                match gate_g.codec() {
                    Codec::Iq2K => "routed_gate_iq2",
                    Codec::Iq3K => "routed_gate_iq3",
                },
            )?;
            let down_reduce_pipeline = metal.compile_msl_strict(
                ROUTED_SOURCE,
                match down_g.codec() {
                    Codec::Iq2K => "routed_down_iq2",
                    Codec::Iq3K => "routed_down_iq3",
                },
            )?;
            ensure!(
                gate_up_pipeline.max_threads_per_threadgroup() >= 80
                    && down_reduce_pipeline.max_threads_per_threadgroup() >= 320,
                "Metal cannot dispatch routed IQ_K threadgroup geometry"
            );
            let sigmoid_table = buffer(metal, 1 << 17)?;
            sigmoid_table.write_bytes(0, &super::sigmoid_fp16_table())?;
            let scratch = Scratch {
                hidden: buffer(metal, HIDDEN * 2)?,
                slots: buffer(metal, TOP_K * 4)?,
                gate_dims: buffer(metal, 8)?,
                down_dims: buffer(metal, 8)?,
                gate: buffer(metal, TOP_K * INTERMEDIATE * 2)?,
                up: buffer(metal, TOP_K * INTERMEDIATE * 2)?,
                activation: buffer(metal, TOP_K * DOWN_STORED * 2)?,
                down: buffer(metal, TOP_K * HIDDEN * 2)?,
                scores: buffer(metal, TOP_K * 2)?,
                order: buffer(metal, TOP_K * 4)?,
                output: buffer(metal, HIDDEN * 2)?,
                sigmoid_table,
            };
            let policy_capacity = if chunked {
                super::chunks::MAX_BOUNDED_SLOTS
            } else {
                capacity
            };
            let mut expert_at = Vec::new();
            expert_at.try_reserve_exact(policy_capacity)?;
            expert_at.resize(capacity, None);
            let mut last_use = Vec::new();
            last_use.try_reserve_exact(policy_capacity)?;
            last_use.resize(capacity, 0);
            Ok(Self {
                metal: metal.clone(),
                layer,
                row_workspace,
                capacity,
                storage,
                chunked_pipelines,
                activation_pipeline,
                reduction_pipeline,
                input_fp16_pipeline,
                resident_routes_pipeline,
                gate_up_pipeline,
                down_reduce_pipeline,
                bounded_pipelines: if capacity < 512 {
                    let source = format!(
                        "{}\n{}",
                        ROUTED_SOURCE,
                        include_str!("../kernels/metal/iqk_bounded.metal")
                    );
                    let suffix = |codec| match codec {
                        Codec::Iq2K => "iq2",
                        Codec::Iq3K => "iq3",
                    };
                    Some((
                        metal.compile_msl_strict(
                            &source,
                            &format!("bounded_gate_{}", suffix(gate_g.codec())),
                        )?,
                        metal.compile_msl_strict(
                            &source,
                            &format!("bounded_down_{}", suffix(down_g.codec())),
                        )?,
                    ))
                } else {
                    None
                },
                scratch,
                slot_of: BTreeMap::new(),
                expert_at,
                last_use,
                frequency: BTreeMap::new(),
                clock: 0,
            })
        }

        pub fn capacity(&self) -> usize {
            self.capacity
        }

        /// Prepare a whole suffix release without allocating replacement weights.
        /// The owner must complete its stream/publication work and discard old
        /// captured schedules first. No access through earlier buffer aliases
        /// may be encoded while the exclusive prepared borrow remains alive.
        pub(crate) fn prepare_release_suffix(
            &mut self,
            target_capacity: usize,
        ) -> Result<PreparedExpertSuffixRelease<'_>> {
            ensure!(
                target_capacity >= TOP_K,
                "expert release target cannot fit top-ten routes"
            );
            let ExpertStorage::Chunked(chunks) = &self.storage else {
                anyhow::bail!("expert suffix release requires fixed chunk storage");
            };
            ensure!(
                chunks.capacity() == self.capacity,
                "expert storage capacity differs from slot policy"
            );
            let (keep, released_requested_bytes) = chunks.prepare_suffix(target_capacity)?;
            self.check_resize_idle()?;
            let report = ExpertReleaseReport {
                old_capacity: self.capacity,
                new_capacity: target_capacity,
                released_requested_bytes,
                evicted_sources: self
                    .slot_of
                    .values()
                    .filter(|&&slot| slot >= target_capacity)
                    .count(),
            };
            Ok(PreparedExpertSuffixRelease {
                pool: self,
                keep,
                report,
            })
        }

        fn check_resize_idle(&self) -> Result<()> {
            self.row_workspace.check_idle()?;
            let ExpertStorage::Chunked(chunks) = &self.storage else {
                anyhow::bail!("expert resizing requires fixed chunk storage");
            };
            chunks.check_quiescent()?;
            for buffer in [
                &self.scratch.hidden,
                &self.scratch.slots,
                &self.scratch.gate_dims,
                &self.scratch.down_dims,
                &self.scratch.gate,
                &self.scratch.up,
                &self.scratch.activation,
                &self.scratch.down,
                &self.scratch.scores,
                &self.scratch.order,
                &self.scratch.output,
                &self.scratch.sigmoid_table,
            ] {
                buffer.read_bytes(0, &mut [])?;
            }
            ensure!(
                self.expert_at.len() == self.capacity && self.last_use.len() == self.capacity,
                "expert slot policy extent differs from capacity"
            );
            for (&source, &slot) in &self.slot_of {
                ensure!(
                    (source as usize) < self.layer.num_experts
                        && slot < self.capacity
                        && self.expert_at[slot] == Some(source),
                    "expert published slot map is inconsistent"
                );
            }
            for (slot, source) in self.expert_at.iter().enumerate() {
                ensure!(
                    source.is_none_or(|source| self.slot_of.get(&source) == Some(&slot)),
                    "expert physical slot map is inconsistent"
                );
            }
            Ok(())
        }

        /// Quote all additional Metal backing before preparing bounded growth.
        pub(crate) fn quote_growth_suffix(
            &self,
            target_capacity: usize,
        ) -> Result<ExpertGrowthReport> {
            let ExpertStorage::Chunked(chunks) = &self.storage else {
                anyhow::bail!("expert suffix growth requires fixed chunk storage");
            };
            ensure!(
                chunks.capacity() == self.capacity
                    && self.expert_at.capacity() >= target_capacity
                    && self.last_use.capacity() >= target_capacity,
                "expert growth retained slot-policy capacity differs"
            );
            let quote = chunks.quote_growth(target_capacity)?;
            Ok(ExpertGrowthReport {
                old_capacity: self.capacity,
                new_capacity: target_capacity,
                added_requested_bytes: quote.added_requested_bytes,
                preparation_requested_bytes: quote.preparation_requested_bytes,
            })
        }

        /// Prepare missing suffix slots without replacing retained weight slabs.
        /// Cancellation preserves every published source and address row.
        pub(crate) fn prepare_growth_suffix(
            &mut self,
            target_capacity: usize,
        ) -> Result<PreparedExpertSuffixGrowth<'_>> {
            let report = self.quote_growth_suffix(target_capacity)?;
            self.check_resize_idle()?;
            let ExpertStorage::Chunked(chunks) = &self.storage else {
                unreachable!()
            };
            let growth = chunks.prepare_growth(&self.metal, target_capacity)?;
            ensure!(
                growth.quote.added_requested_bytes == report.added_requested_bytes
                    && growth.quote.preparation_requested_bytes
                        == report.preparation_requested_bytes,
                "expert growth preparation differs from quote"
            );
            Ok(PreparedExpertSuffixGrowth {
                pool: self,
                growth,
                report,
            })
        }

        pub(crate) fn is_chunked(&self) -> bool {
            matches!(self.storage, ExpertStorage::Chunked(_))
        }

        /// Retain every physical owner read through the published slot table
        /// until the addressed projection's command completes.
        pub(crate) fn retain_addressed_prefill(&self, batch: &mut Batch) -> Result<&Buffer> {
            let ExpertStorage::Chunked(chunks) = &self.storage else {
                anyhow::bail!("addressed prefill requires chunked expert storage");
            };
            ensure!(
                chunks.capacity() == self.capacity
                    && chunks.addresses.byte_len() >= self.capacity * 18 * 8,
                "addressed prefill table does not cover the current capacity"
            );
            let mut owners = [&chunks.addresses; super::chunks::MAX_CHUNKS];
            for (owner, buffer) in owners.iter_mut().zip(chunks.buffers()) {
                *owner = buffer;
            }
            batch.use_indirect_reads(&owners[..chunks.count()])?;
            Ok(&chunks.addresses)
        }

        pub(crate) fn chunk_count(&self) -> usize {
            match &self.storage {
                ExpertStorage::Chunked(chunks) => chunks.count(),
                _ => 0,
            }
        }

        pub(crate) fn chunk_range(&self, chunk: usize) -> Result<std::ops::Range<usize>> {
            match &self.storage {
                ExpertStorage::Chunked(chunks) => chunks.range(chunk),
                _ => anyhow::bail!("expert storage is contiguous"),
            }
        }

        pub(crate) fn chunk_projection(
            &self,
            chunk: usize,
            kind: Projection,
        ) -> Result<(&IqkGemv, Geometry)> {
            match &self.storage {
                ExpertStorage::Chunked(chunks) => Ok((
                    chunks.projection(chunk, kind)?,
                    Geometry::from_projection(self.layer.projection(kind))?,
                )),
                _ => anyhow::bail!("expert storage is contiguous"),
            }
        }

        fn check_writable(&self) -> Result<()> {
            match &self.storage {
                ExpertStorage::Contiguous { gate, up, down } => {
                    gate.check_writable()?;
                    up.check_writable()?;
                    down.check_writable()
                }
                ExpertStorage::Chunked(chunks) => chunks.check_writable(),
            }
        }

        fn projections_at(storage: &ExpertStorage, slot: usize) -> Result<([&IqkGemv; 3], usize)> {
            match storage {
                ExpertStorage::Contiguous { gate, up, down } => Ok(([gate, up, down], slot)),
                ExpertStorage::Chunked(chunks) => {
                    let (chunk, local) = chunks.location(slot)?;
                    Ok((
                        [
                            chunks.projection(chunk, Projection::Gate)?,
                            chunks.projection(chunk, Projection::Up)?,
                            chunks.projection(chunk, Projection::Down)?,
                        ],
                        local,
                    ))
                }
            }
        }

        pub(crate) fn check_layer(&self, expected: &ExpertLayer) -> Result<()> {
            ensure!(
                self.layer.same_source(expected),
                "expert pool belongs to another package or layer"
            );
            Ok(())
        }

        pub(crate) fn source_layer(&self) -> ExpertLayer {
            self.layer.clone()
        }

        pub fn snapshot(&self) -> ExpertPoolSnapshot {
            ExpertPoolSnapshot {
                clock: self.clock,
                slot_of: self.slot_of.clone(),
                expert_at: self.expert_at.clone(),
                frequency: self.frequency.clone(),
                last_use: self.last_use.clone(),
            }
        }

        /// One demand touch per source in a complete capacity-bounded union.
        pub(crate) fn ensure_group(
            &mut self,
            active: &BTreeSet<u32>,
        ) -> Result<ExpertDemandCounts> {
            self.ensure_group_protected(active, &BTreeSet::new())
        }

        pub(crate) fn ensure_group_protected(
            &mut self,
            active: &BTreeSet<u32>,
            protect: &BTreeSet<u32>,
        ) -> Result<ExpertDemandCounts> {
            self.validate_group(active, protect)?;
            let protected: BTreeSet<_> = active.union(protect).copied().collect();
            let workspace = self.row_workspace.clone();
            let mut rows = None;
            let mut reader_started = false;
            let mut counts = ExpertDemandCounts::default();
            for source in active.iter().copied() {
                let slot = match self.slot_of.get(&source).copied() {
                    Some(slot) => {
                        counts.hits += 1;
                        slot
                    }
                    None => {
                        let eviction = self.slot_of.len() == self.capacity;
                        // Preserve earlier resident touches if writability,
                        // workspace borrowing or the first row read fails.
                        self.check_writable()?;
                        if !reader_started {
                            let missing = active
                                .iter()
                                .copied()
                                .filter(|source| !self.slot_of.contains_key(source))
                                .collect();
                            rows = workspace.prefill_rows(&self.layer, missing)?;
                            reader_started = true;
                        }
                        let slot = if let Some(cursor) = &mut rows {
                            let row = cursor.next(source)?;
                            // SAFETY: The caller retired prior pool readers,
                            // and ordinary guarded projection copies apply.
                            unsafe { self.load_demand_row(source, row, &protected, None)? }
                        } else {
                            self.load_source(source, &protected)?
                        };
                        counts.misses += 1;
                        counts.evictions += usize::from(eviction);
                        slot
                    }
                };
                self.touch(source, slot)?;
            }
            if let Some(rows) = rows {
                rows.finish()?;
            }
            Ok(counts)
        }

        fn validate_group(&self, active: &BTreeSet<u32>, protect: &BTreeSet<u32>) -> Result<()> {
            ensure!(
                !active.is_empty() && active.len() <= self.capacity,
                "expert demand union must be nonempty and fit capacity"
            );
            ensure!(
                active
                    .iter()
                    .all(|&id| (id as usize) < self.layer.num_experts),
                "expert demand union contains an invalid source"
            );
            let protected: BTreeSet<_> = active.union(protect).copied().collect();
            ensure!(
                protected.len() <= self.capacity
                    && protected
                        .iter()
                        .all(|&id| (id as usize) < self.layer.num_experts),
                "expert protected union must fit capacity and source geometry"
            );
            Ok(())
        }

        /// # Safety
        /// `reader_slots` must cover every encoded/submitted pool reader;
        /// `protect` must contain their logical sources. Only the exclusive
        /// prefill scope establishes this contract while a projection is busy.
        unsafe fn ensure_group_disjoint(
            &mut self,
            active: &BTreeSet<u32>,
            protect: &BTreeSet<u32>,
            reader_slots: &BTreeSet<usize>,
        ) -> Result<()> {
            self.validate_group(active, protect)?;
            let protected = active.union(protect).copied().collect();
            let workspace = self.row_workspace.clone();
            let missing = active
                .iter()
                .copied()
                .filter(|source| !self.slot_of.contains_key(source))
                .collect();
            let mut rows = workspace.prefill_rows(&self.layer, missing)?;
            for source in active.iter().copied() {
                let slot = match self.slot_of.get(&source).copied() {
                    Some(slot) => slot,
                    None => {
                        if let Some(cursor) = &mut rows {
                            let row = cursor.next(source)?;
                            // SAFETY: The unchanged prefill scope excludes all readers.
                            unsafe {
                                self.load_read_row_disjoint(source, row, &protected, reader_slots)?
                            }
                        } else {
                            let mut row = workspace.row(self.layer.row_bytes)?;
                            {
                                self.layer.read_row_into(source as usize, &mut row)?;
                            }
                            // SAFETY: The unchanged prefill scope excludes all readers.
                            unsafe {
                                self.load_read_row_disjoint(source, &row, &protected, reader_slots)?
                            }
                        }
                    }
                };
                self.touch(source, slot)?;
            }
            if let Some(rows) = rows {
                rows.finish()?;
            }
            Ok(())
        }

        /// The caller excludes every recorded GPU/host reader of the victim.
        unsafe fn load_read_row_disjoint(
            &mut self,
            source: u32,
            row: &[u8],
            protected: &BTreeSet<u32>,
            reader_slots: &BTreeSet<usize>,
        ) -> Result<usize> {
            let slot = self.choose_slot(protected)?;
            ensure!(
                !reader_slots.contains(&slot),
                "prefill victim has an outstanding reader"
            );
            self.replace_source(source, slot, row, |projections, rows, local| {
                // SAFETY: The selected physical slot is outside every recorded
                // reader; every projection span is preflighted before invalidation.
                for (projection, row) in projections.into_iter().zip(rows) {
                    unsafe {
                        projection.write_slot_disjoint(local, row)?;
                    }
                }
                Ok(())
            })?;
            Ok(slot)
        }

        pub(crate) fn projection(&self, kind: Projection) -> Result<(&IqkGemv, Geometry)> {
            let geometry = Geometry::from_projection(self.layer.projection(kind))?;
            let projection = match (&self.storage, kind) {
                (ExpertStorage::Contiguous { gate, .. }, Projection::Gate) => gate,
                (ExpertStorage::Contiguous { up, .. }, Projection::Up) => up,
                (ExpertStorage::Contiguous { down, .. }, Projection::Down) => down,
                _ => anyhow::bail!("chunked expert storage requires an explicit chunk projection"),
            };
            Ok((projection, geometry))
        }
        pub fn slot_of(&self, source: u32) -> Option<usize> {
            self.slot_of.get(&source).copied()
        }

        /// Snapshot published source-to-slot hints before selecting routes.
        pub fn residency_map(&self) -> Result<[u32; 512]> {
            ensure!(
                self.layer.num_experts == 512,
                "cache-prior hints require 512 logical experts"
            );
            let mut map = [512u32; 512];
            for (&source, &slot) in &self.slot_of {
                ensure!(
                    (source as usize) < 512
                        && slot < self.capacity
                        && self.expert_at[slot] == Some(source),
                    "bounded IQ_K residency map is inconsistent"
                );
                map[source as usize] = u32::try_from(slot)?;
            }
            Ok(map)
        }

        fn check_sources(&self, sources: &[u32; TOP_K]) -> Result<BTreeSet<u32>> {
            let selected: BTreeSet<_> = sources.iter().copied().collect();
            ensure!(
                selected.len() == TOP_K,
                "routed source IDs must be distinct"
            );
            ensure!(
                selected
                    .iter()
                    .all(|&id| (id as usize) < self.layer.num_experts),
                "routed source ID exceeds package expert count"
            );
            Ok(selected)
        }

        fn choose_slot(&self, protected: &BTreeSet<u32>) -> Result<usize> {
            if let Some(slot) = self.expert_at.iter().position(Option::is_none) {
                return Ok(slot);
            }
            least_frequent_slot(&self.expert_at, &self.last_use, &self.frequency, protected)
                .context("bounded IQ_K has no evictable slot")
        }

        fn touch(&mut self, source: u32, slot: usize) -> Result<()> {
            self.clock = self
                .clock
                .checked_add(1)
                .context("IQ_K recency counter overflow")?;
            let count = self.frequency.entry(source).or_default();
            *count = count
                .checked_add(1)
                .context("IQ_K frequency counter overflow")?;
            self.last_use[slot] = self.clock;
            if self.clock.is_multiple_of(128) {
                for count in self.frequency.values_mut() {
                    *count >>= 1;
                }
            }
            Ok(())
        }

        fn load_source(&mut self, source: u32, protected: &BTreeSet<u32>) -> Result<usize> {
            // In-flight GPU readers make every slot immutable. All projection
            // buffers must be writable before changing the published map.
            self.check_writable()?;
            let workspace = self.row_workspace.clone();
            let mut row = workspace.row(self.layer.row_bytes)?;
            {
                self.layer.read_row_into(source as usize, &mut row)?;
            }
            let slot = self.choose_slot(protected)?;
            self.replace_source(source, slot, &row, |projections, rows, local| {
                for (projection, row) in projections.into_iter().zip(rows) {
                    projection.write_slot(local, row)?;
                }
                Ok(())
            })?;
            Ok(slot)
        }

        fn replace_source(
            &mut self,
            source: u32,
            slot: usize,
            row: &[u8],
            write: impl FnOnce([&IqkGemv; 3], [&[u8]; 3], usize) -> Result<()>,
        ) -> Result<()> {
            let gate = self.layer.projection(Projection::Gate).bytes(row)?;
            let up = self.layer.projection(Projection::Up).bytes(row)?;
            let down = self.layer.projection(Projection::Down).bytes(row)?;
            let (projections, local) = Self::projections_at(&self.storage, slot)?;
            for (projection, row) in projections.into_iter().zip([gate, up, down]) {
                projection.preflight_slot(local, row)?;
            }
            publish_replacement(&mut self.slot_of, &mut self.expert_at, source, slot, || {
                write(projections, [gate, up, down], local)
            })
        }

        /// Seed one initial published membership set with one touch per expert.
        /// Sources are loaded in source-ID order, matching the pool's demand
        /// allocator. A partially completed failure leaves valid published rows.
        pub fn seed_residents(&mut self, sources: &[u32]) -> Result<()> {
            ensure!(
                self.slot_of.is_empty(),
                "resident seeding requires an empty pool"
            );
            ensure!(
                sources.len() <= self.capacity,
                "resident seed exceeds capacity"
            );
            let selected: BTreeSet<_> = sources.iter().copied().collect();
            ensure!(
                selected.len() == sources.len()
                    && selected
                        .iter()
                        .all(|&id| (id as usize) < self.layer.num_experts),
                "resident seed has duplicate or invalid source IDs"
            );
            for source in selected.iter().copied() {
                let slot = self.load_source(source, &selected)?;
                self.touch(source, slot)?;
            }
            Ok(())
        }

        /// Install the package's bounded prior and preload its highest-ranked rows.
        pub fn seed_hotlist(&mut self, counts: &BTreeMap<u32, u64>) -> Result<()> {
            ensure!(
                self.slot_of.is_empty() && self.frequency.is_empty(),
                "hotlist requires an unused pool"
            );
            ensure!(
                counts
                    .keys()
                    .all(|&id| (id as usize) < self.layer.num_experts),
                "hotlist expert exceeds geometry"
            );
            let mut ranked: Vec<u32> = counts.keys().copied().collect();
            ranked.sort_unstable_by(|a, b| counts[b].cmp(&counts[a]).then(a.cmp(b)));
            ranked.truncate(self.capacity);
            let protected: BTreeSet<_> = ranked.iter().copied().collect();
            self.frequency = counts.clone();
            // Hot seeding preserves ranking order, unlike sorted runtime demands.
            for source in ranked {
                let slot = self.load_source(source, &protected)?;
                self.touch(source, slot)?;
            }
            let top = self.frequency.values().copied().max().unwrap_or(0);
            if top > 8 {
                let scale = 8.0 / top as f64;
                for count in self.frequency.values_mut() {
                    *count = ((*count as f64 * scale).round_ties_even() as u64).max(1);
                }
            }
            Ok(())
        }

        /// Read and publish all missing selected rows before exposing selectors.
        pub fn ensure(&mut self, sources: &[u32; TOP_K]) -> Result<[u32; TOP_K]> {
            // SAFETY: No gated write is requested; ordinary buffer guards apply.
            unsafe { self.ensure_with_gate(sources, None) }
        }

        unsafe fn ensure_with_gate(
            &mut self,
            sources: &[u32; TOP_K],
            gate: Option<&crate::metal::EventPoint>,
        ) -> Result<[u32; TOP_K]> {
            let protected = self.check_sources(sources)?;
            let workspace = self.row_workspace.clone();
            let missing: Vec<_> = protected
                .iter()
                .copied()
                .filter(|source| !self.slot_of.contains_key(source))
                .collect();
            if !missing.is_empty() {
                if let Some(event) = gate {
                    ensure!(!event.reached(), "expert publication was already released");
                } else {
                    self.check_writable()?;
                }
            }
            let mut rows = workspace.prefill_rows(&self.layer, missing)?;
            for source in protected.iter().copied() {
                let slot = match self.slot_of.get(&source).copied() {
                    Some(slot) => slot,
                    None => {
                        if let Some(cursor) = &mut rows {
                            let row = cursor.next(source)?;
                            // SAFETY: Every demand reader waits on the unchanged gate.
                            unsafe { self.load_demand_row(source, row, &protected, gate)? }
                        } else {
                            let mut row = workspace.row(self.layer.row_bytes)?;
                            {
                                self.layer.read_row_into(source as usize, &mut row)?;
                            }
                            // SAFETY: The same gate and exclusive inference owner apply.
                            unsafe { self.load_demand_row(source, &row, &protected, gate)? }
                        }
                    }
                };
                self.touch(source, slot)?;
            }
            if let Some(rows) = rows {
                rows.finish()?;
            }
            let mut slots = [0u32; TOP_K];
            for (i, source) in sources.iter().enumerate() {
                let slot = *self
                    .slot_of
                    .get(source)
                    .context("selected expert is not published")?;
                ensure!(
                    self.expert_at[slot] == Some(*source),
                    "stale bounded IQ_K slot mapping"
                );
                slots[i] = u32::try_from(slot)?;
            }
            Ok(slots)
        }

        /// The caller excludes prior readers and holds future readers behind
        /// `gate`, when supplied. Read-ahead never publishes or selects victims.
        unsafe fn load_demand_row(
            &mut self,
            source: u32,
            row: &[u8],
            protected: &BTreeSet<u32>,
            gate: Option<&crate::metal::EventPoint>,
        ) -> Result<usize> {
            if let Some(event) = gate {
                ensure!(!event.reached(), "expert publication was already released");
            } else {
                self.check_writable()?;
            }
            let slot = self.choose_slot(protected)?;
            self.replace_source(source, slot, row, |projections, rows, local| {
                for (projection, row) in projections.into_iter().zip(rows) {
                    if let Some(event) = gate {
                        // SAFETY: The caller excludes all readers behind this event.
                        unsafe {
                            projection.write_slot_gated(local, row, event)?;
                        }
                    } else {
                        projection.write_slot(local, row)?;
                    }
                }
                Ok(())
            })?;
            Ok(slot)
        }

        /// Publish complete bundle rows and existing selectors under one gate.
        ///
        /// # Safety
        /// Earlier pool readers have retired. Every future pool and selector
        /// read must wait on this exact unopened publication point.
        pub(crate) unsafe fn publish_gated(
            &mut self,
            sources: &[u32; TOP_K],
            scores: &[u16; TOP_K],
            event: &crate::metal::EventPoint,
        ) -> Result<()> {
            ensure!(!event.reached(), "expert publication gate already released");
            ensure!(
                scores.iter().all(|&word| bf16::from_bits(word).is_finite()),
                "nonfinite expert score"
            );
            // SAFETY: Same exclusive publication contract as this method.
            let slots = unsafe { self.ensure_with_gate(sources, Some(event))? };
            self.write_selectors(sources, scores, &slots, Some(event))
        }

        fn write_selectors(
            &self,
            sources: &[u32; TOP_K],
            scores: &[u16; TOP_K],
            slots: &[u32; TOP_K],
            event: Option<&crate::metal::EventPoint>,
        ) -> Result<()> {
            let mut order: Vec<u32> = (0..TOP_K as u32).collect();
            order.sort_unstable_by_key(|&route| sources[route as usize]);
            let slot_bytes: Vec<u8> = order
                .iter()
                .flat_map(|&route| slots[route as usize].to_le_bytes())
                .collect();
            let score_bytes: Vec<u8> = order
                .iter()
                .flat_map(|&route| scores[route as usize].to_le_bytes())
                .collect();
            let order_bytes: Vec<u8> = order.iter().flat_map(|route| route.to_le_bytes()).collect();
            for (buffer, bytes) in [
                (&self.scratch.slots, slot_bytes),
                (&self.scratch.scores, score_bytes),
                (&self.scratch.order, order_bytes),
            ] {
                if let Some(event) = event {
                    // SAFETY: Only publish_gated supplies an event after proving
                    // exclusive selector access until that event is released.
                    unsafe { crate::metal::PublicationLease::new(buffer, event)? }.write(&bytes)?;
                } else {
                    buffer.write_bytes(0, &bytes)?;
                }
            }
            Ok(())
        }

        pub(crate) fn encode_gated(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            flag: &Buffer,
        ) -> Result<Buffer> {
            ensure!(
                self.bounded_pipelines.is_some(),
                "bounded kernels require bounded residency"
            );
            self.encode_selected_with_guard(batch, hidden, Some(flag))
        }

        /// Encode the established two-stage IQ_K routed path from a device
        /// hidden row. Callers complete earlier GPU readers before demand can
        /// replace any physical slot in this bounded pool.
        pub(crate) fn encode_routed_device(
            &mut self,
            batch: &mut Batch,
            hidden: &Buffer,
            sources: &[u32; TOP_K],
            scores_bf16: &[u16; TOP_K],
        ) -> Result<Buffer> {
            ensure!(
                hidden.byte_len() == HIDDEN * 2,
                "routed hidden width differs"
            );
            ensure!(
                scores_bf16
                    .iter()
                    .all(|&word| bf16::from_bits(word).is_finite()),
                "bounded IQ_K route scores must be finite"
            );
            let slots = self.ensure(sources)?;
            self.write_selectors(sources, scores_bf16, &slots, None)?;
            self.encode_selected_device(batch, hidden)
        }

        /// Identity-resident weights require no loading or CPU route readback.
        /// The GPU guard validates selectors before they become addresses.
        pub(crate) fn encode_full_resident_device(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            ids: &Buffer,
            scores: &Buffer,
            changed: &Buffer,
            status: &Buffer,
        ) -> Result<Buffer> {
            ensure!(
                self.capacity == 512
                    && self
                        .expert_at
                        .iter()
                        .enumerate()
                        .all(|(slot, source)| *source == Some(slot as u32)),
                "device-only expert routing requires full identity residency"
            );
            ensure!(
                hidden.byte_len() == HIDDEN * 2
                    && ids.byte_len() >= TOP_K * 4
                    && scores.byte_len() >= TOP_K * 2
                    && changed.byte_len() >= 4
                    && status.byte_len() >= 4,
                "device-only expert selector extent differs"
            );
            let bindings = [
                ids,
                scores,
                changed,
                &self.scratch.slots,
                &self.scratch.scores,
                &self.scratch.order,
                status,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: One thread validates ten IDs/scores and prepares bounded
            // identity slots. Failure sets sticky status and uses safe slot0.
            unsafe {
                batch.dispatch_1d(&self.resident_routes_pipeline, &bindings, 1, 1)?;
            }
            self.encode_selected_device(batch, hidden)
        }

        fn encode_selected_device(&self, batch: &mut Batch, hidden: &Buffer) -> Result<Buffer> {
            self.encode_selected_with_guard(batch, hidden, None)
        }

        fn encode_selected_with_guard(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            publication: Option<&Buffer>,
        ) -> Result<Buffer> {
            if let ExpertStorage::Chunked(chunks) = &self.storage {
                return self.encode_chunked(
                    batch,
                    hidden,
                    publication.unwrap_or(&chunks.ready),
                    chunks,
                );
            }
            let gate = self.projection(Projection::Gate)?.0.stream_bindings();
            let up = self.projection(Projection::Up)?.0.stream_bindings();
            let down = self.projection(Projection::Down)?.0.stream_bindings();
            let binding = |buffer| Binding { buffer, offset: 0 };
            // SAFETY: Conversion reads the initialized hidden row and writes
            // exactly HIDDEN FP16 cells; it does not access expert selectors.
            unsafe {
                batch.dispatch_1d(
                    &self.input_fp16_pipeline,
                    &[binding(hidden), binding(&self.scratch.hidden)],
                    HIDDEN,
                    256,
                )?;
            }
            let guarded = self.bounded_pipelines.as_ref();
            let mut dispatch = |normal: &Pipeline,
                                guarded: Option<&Pipeline>,
                                bindings: &[Binding<'_>],
                                grid,
                                group|
             -> Result<()> {
                if let Some(flag) = publication {
                    let mut bindings = bindings.to_vec();
                    bindings.push(binding(flag));
                    // SAFETY: The extra binding is the publication guard of the
                    // selected bounded kernel; base geometry is unchanged.
                    unsafe {
                        batch.dispatch_1d(
                            guarded.context("missing bounded kernel")?,
                            &bindings,
                            grid,
                            group,
                        )
                    }
                } else {
                    // SAFETY: Existing routed bindings and exact kernel geometry.
                    unsafe { batch.dispatch_1d(normal, bindings, grid, group) }
                }
            };
            // The ordinary path has published selectors; the bounded path
            // waits for publication and validates its flag before addressing.
            dispatch(
                &self.gate_up_pipeline,
                guarded.map(|pipelines| &pipelines.0),
                &[
                    binding(&self.scratch.hidden),
                    gate[0],
                    gate[1],
                    gate[2],
                    gate[3],
                    gate[4],
                    gate[5],
                    gate[6],
                    up[0],
                    up[1],
                    up[2],
                    up[3],
                    up[4],
                    up[5],
                    binding(&self.scratch.slots),
                    binding(&self.scratch.order),
                    binding(&self.scratch.sigmoid_table),
                    binding(&self.scratch.gate),
                    binding(&self.scratch.up),
                    binding(&self.scratch.activation),
                ],
                TOP_K * (INTERMEDIATE / 16) * 80,
                80,
            )?;
            dispatch(
                &self.down_reduce_pipeline,
                guarded.map(|pipelines| &pipelines.1),
                &[
                    binding(&self.scratch.activation),
                    down[0],
                    down[1],
                    down[2],
                    down[3],
                    down[4],
                    down[5],
                    down[6],
                    binding(&self.scratch.slots),
                    binding(&self.scratch.scores),
                    binding(&self.scratch.order),
                    binding(&self.scratch.down),
                    binding(&self.scratch.output),
                ],
                (HIDDEN / 16) * 320,
                320,
            )?;
            Ok(self.scratch.output.clone())
        }

        fn encode_chunked(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            publication: &Buffer,
            chunks: &FixedChunks,
        ) -> Result<Buffer> {
            let (gate, down) = self
                .chunked_pipelines
                .as_ref()
                .context("missing chunked routed kernels")?;
            let binding = |buffer| Binding { buffer, offset: 0 };
            let gate_table = chunks.projection(0, Projection::Gate)?.stream_bindings()[6];
            let down_table = chunks.projection(0, Projection::Down)?.stream_bindings()[6];
            let capacity = u32::try_from(self.capacity)?.to_le_bytes();
            // SAFETY: Identical native input conversion and single-row geometry.
            unsafe {
                batch.dispatch_1d(
                    &self.input_fp16_pipeline,
                    &[binding(hidden), binding(&self.scratch.hidden)],
                    HIDDEN,
                    256,
                )?;
            }
            let mut owners = [&chunks.addresses; super::chunks::MAX_CHUNKS];
            for (owner, buffer) in owners.iter_mut().zip(chunks.buffers()) {
                *owner = buffer;
            }
            let owners = &owners[..chunks.count()];
            // The retained table's active prefix covers current capacity. The
            // copied capacity guard excludes retired rows before address reads,
            // and all remaining possible owners are retained through completion.
            batch.use_indirect_reads(owners)?;
            // SAFETY: The table contains checked stream spans, selected global
            // slots are published behind the guard, and native geometry is kept.
            unsafe {
                batch.dispatch_1d_with_inline(
                    gate,
                    &[
                        binding(&self.scratch.hidden),
                        binding(&chunks.addresses),
                        gate_table,
                        binding(&self.scratch.slots),
                        binding(&self.scratch.order),
                        binding(&self.scratch.sigmoid_table),
                        binding(&self.scratch.gate),
                        binding(&self.scratch.up),
                        binding(&self.scratch.activation),
                        binding(publication),
                    ],
                    &[&capacity],
                    TOP_K * (INTERMEDIATE / 16) * 80,
                    80,
                )?;
            }
            batch.use_indirect_reads(owners)?;
            // SAFETY: The same complete table/guard contract applies to down;
            // the source-order FP16/BF16 reduction is the native implementation.
            unsafe {
                batch.dispatch_1d_with_inline(
                    down,
                    &[
                        binding(&self.scratch.activation),
                        binding(&chunks.addresses),
                        down_table,
                        binding(&self.scratch.slots),
                        binding(&self.scratch.scores),
                        binding(&self.scratch.order),
                        binding(&self.scratch.down),
                        binding(&self.scratch.output),
                        binding(publication),
                    ],
                    &[&capacity],
                    (HIDDEN / 16) * 320,
                    320,
                )?;
            }
            Ok(self.scratch.output.clone())
        }

        /// Encode gate, up, activation, down and ordered reduction in one batch.
        pub(crate) fn encode_routed(
            &mut self,
            batch: &mut Batch,
            hidden_bf16: &[u8],
            sources: &[u32; TOP_K],
            scores_bf16: &[u8],
        ) -> Result<Buffer> {
            ensure!(
                !self.is_chunked(),
                "diagnostic routed GEMV requires contiguous expert storage"
            );
            ensure!(
                hidden_bf16.len() == HIDDEN * 2 && scores_bf16.len() == TOP_K * 2,
                "bounded IQ_K hidden or score BF16 byte count is wrong"
            );
            ensure!(
                scores_bf16
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes))
                        .to_f32()
                        .is_finite()),
                "bounded IQ_K route scores must be finite"
            );
            let slots = self.ensure(sources)?;
            let mut input = Vec::with_capacity(HIDDEN * 2);
            for bytes in hidden_bf16.as_chunks::<2>().0 {
                let word = u16::from_le_bytes(*bytes);
                input.extend_from_slice(
                    &f16::from_f32(bf16::from_bits(word).to_f32())
                        .to_bits()
                        .to_le_bytes(),
                );
            }
            self.scratch.hidden.write_bytes(0, &input)?;
            let slot_bytes: Vec<u8> = slots.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.scratch.slots.write_bytes(0, &slot_bytes)?;
            self.scratch
                .gate_dims
                .write_bytes(0, &[10u32.to_le_bytes(), 640u32.to_le_bytes()].concat())?;
            self.scratch
                .down_dims
                .write_bytes(0, &[1u32.to_le_bytes(), 2560u32.to_le_bytes()].concat())?;
            self.scratch.scores.write_bytes(0, scores_bf16)?;
            let mut order: Vec<u32> = (0..TOP_K as u32).collect();
            order.sort_by_key(|&route| sources[route as usize]);
            let order_bytes: Vec<u8> = order.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.scratch.order.write_bytes(0, &order_bytes)?;

            let gate_d = GemvDescriptor::new(
                Geometry::from_projection(self.layer.projection(Projection::Gate))?,
                TOP_K,
                TOP_K,
            )?;
            let up_d = GemvDescriptor::new(
                Geometry::from_projection(self.layer.projection(Projection::Up))?,
                TOP_K,
                TOP_K,
            )?;
            let down_d = GemvDescriptor::new(
                Geometry::from_projection(self.layer.projection(Projection::Down))?,
                TOP_K,
                1,
            )?;
            let gate = self.projection(Projection::Gate)?.0;
            let up = self.projection(Projection::Up)?.0;
            let down = self.projection(Projection::Down)?.0;
            // SAFETY: `ensure` validated and published every slot. The selector
            // and dimensions buffers were written above and are retained by the
            // first encoded dispatch, excluding host mutation until completion.
            unsafe {
                gate.encode(
                    batch,
                    gate_d,
                    &slots,
                    &self.scratch.hidden,
                    &self.scratch.slots,
                    &self.scratch.gate_dims,
                    &self.scratch.gate,
                )?;
                up.encode(
                    batch,
                    up_d,
                    &slots,
                    &self.scratch.hidden,
                    &self.scratch.slots,
                    &self.scratch.gate_dims,
                    &self.scratch.up,
                )?;
                batch.dispatch_1d(
                    &self.activation_pipeline,
                    &[
                        Binding {
                            buffer: &self.scratch.gate,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.up,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.sigmoid_table,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.activation,
                            offset: 0,
                        },
                    ],
                    TOP_K * DOWN_STORED,
                    256,
                )?;
                down.encode(
                    batch,
                    down_d,
                    &slots,
                    &self.scratch.activation,
                    &self.scratch.slots,
                    &self.scratch.down_dims,
                    &self.scratch.down,
                )?;
                batch.dispatch_1d(
                    &self.reduction_pipeline,
                    &[
                        Binding {
                            buffer: &self.scratch.down,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.scores,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.order,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.output,
                            offset: 0,
                        },
                    ],
                    HIDDEN,
                    256,
                )?;
            }
            Ok(self.scratch.output.clone())
        }

        pub fn submit(
            &mut self,
            hidden_bf16: &[u8],
            sources: &[u32; TOP_K],
            scores_bf16: &[u8],
        ) -> Result<PendingOutput> {
            let mut batch = self.metal.batch()?;
            let output = self.encode_routed(&mut batch, hidden_bf16, sources, scores_bf16)?;
            Ok(PendingOutput {
                flight: batch.submit(),
                gate: self.scratch.gate.clone(),
                up: self.scratch.up.clone(),
                activation: self.scratch.activation.clone(),
                down: self.scratch.down.clone(),
                output,
            })
        }

        pub fn run(
            &mut self,
            hidden_bf16: &[u8],
            sources: &[u32; TOP_K],
            scores_bf16: &[u8],
        ) -> Result<Vec<u8>> {
            self.submit(hidden_bf16, sources, scores_bf16)?.wait()
        }
    }

    #[cfg(test)]
    mod overlap_device_tests {
        use super::*;

        #[test]
        #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive Metal ownership"]
        fn suffix_release_is_allocation_free_and_preserves_prefix_policy() -> Result<()> {
            let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
            let metal = Metal::new()?;
            for layer in [0, 2] {
                for capacity in [65, 102, 159, 178] {
                    let target = (capacity - 1) / 16 * 16;
                    let source = package.expert_layer(layer)?;
                    let workspace =
                        crate::package::ExpertRowWorkspace::with_prefill_reader(source.row_bytes)?;
                    let mut pool = BoundedExperts::with_storage(
                        &metal,
                        source,
                        capacity,
                        workspace.clone(),
                        true,
                    )?;
                    pool.seed_residents(&(100..100 + capacity as u32).collect::<Vec<_>>())?;
                    let before = pool.snapshot();
                    let ExpertStorage::Chunked(chunks) = &pool.storage else {
                        unreachable!()
                    };
                    let table = chunks.addresses.clone();
                    let address = table.gpu_address(0, table.byte_len())?;
                    let aliases = [
                        chunks.buffers().last().unwrap().clone(),
                        table.clone(),
                        chunks.ready.clone(),
                        pool.scratch.slots.clone(),
                    ];
                    let requested = metal.memory_stats()?.live_requested_buffer_bytes;
                    for target in [0, 10, 63, 65, capacity, capacity + 64] {
                        assert!(pool.prepare_release_suffix(target).is_err());
                        assert_eq!(pool.snapshot(), before);
                    }
                    let row = workspace.row(pool.layer.row_bytes)?;
                    assert!(pool.prepare_release_suffix(target).is_err());
                    drop(row);
                    for alias in &aliases {
                        let mut batch = metal.batch()?;
                        batch.use_indirect_reads(&[alias])?;
                        assert!(pool.prepare_release_suffix(target).is_err());
                        assert_eq!(pool.snapshot(), before);
                        drop(batch);
                    }
                    drop(aliases);
                    let report = {
                        let plan = pool.prepare_release_suffix(target)?;
                        plan.report()
                    };
                    assert_eq!(report.old_capacity, capacity);
                    assert_eq!(report.new_capacity, target);
                    assert_eq!(report.evicted_sources, capacity - target);
                    assert_eq!(pool.snapshot(), before);
                    assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, requested);
                    // No Metal allocation is permitted by preparation or commit.
                    let allocation_failure = metal.test_fail_allocation_after(0)?;
                    assert_eq!(pool.prepare_release_suffix(target)?.commit(), report);
                    drop(allocation_failure);
                    assert_eq!(
                        requested - metal.memory_stats()?.live_requested_buffer_bytes,
                        report.released_requested_bytes
                    );
                    assert_eq!(pool.capacity(), target);
                    assert_eq!(pool.chunk_count(), target / 16);
                    assert_eq!(pool.chunk_range(0)?, 0..16);
                    assert_eq!(
                        pool.chunk_range(pool.chunk_count() - 1)?,
                        target - 16..target
                    );
                    assert_eq!(table.gpu_address(0, table.byte_len())?, address);
                    let mut expected = before.clone();
                    expected.slot_of.retain(|_, slot| *slot < target);
                    expected.expert_at.truncate(target);
                    expected.last_use.truncate(target);
                    assert_eq!(pool.snapshot(), expected);
                    let routes = std::array::from_fn(|index| {
                        if index == 9 {
                            100 + capacity as u32 - 1
                        } else {
                            100 + index as u32
                        }
                    });
                    assert!(
                        pool.ensure(&routes)?
                            .into_iter()
                            .all(|slot| slot < target as u32)
                    );
                    assert!(pool.slot_of(routes[9]).is_some());
                    pool.check_writable()?;
                    drop(table);
                    let grown_from = pool.snapshot();
                    let quote = pool.quote_growth_suffix(capacity)?;
                    let requested = metal.memory_stats()?.live_requested_buffer_bytes;
                    {
                        let plan = pool.prepare_growth_suffix(capacity)?;
                        assert_eq!(plan.report(), quote);
                        assert_eq!(
                            metal.memory_stats()?.live_requested_buffer_bytes - requested,
                            quote.preparation_requested_bytes
                        );
                    }
                    assert_eq!(pool.snapshot(), grown_from);
                    assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, requested);
                    let plan = pool.prepare_growth_suffix(capacity)?;
                    let allocation_failure = metal.test_fail_allocation_after(0)?;
                    assert_eq!(plan.commit(), quote);
                    drop(allocation_failure);
                    assert_eq!(
                        metal.memory_stats()?.live_requested_buffer_bytes - requested,
                        quote.added_requested_bytes
                    );
                    let mut expected = grown_from;
                    expected.expert_at.resize(capacity, None);
                    expected.last_use.resize(capacity, 0);
                    assert_eq!(pool.snapshot(), expected);
                    assert_eq!(pool.capacity(), capacity);
                    assert!(pool.quote_growth_suffix(capacity).is_err());
                    assert!(pool.quote_growth_suffix(capacity + 64).is_ok());
                }
            }
            let source = package.expert_layer(0)?;
            let mut contiguous = BoundedExperts::new(&metal, source, 65)?;
            assert!(contiguous.prepare_release_suffix(64).is_err());
            Ok(())
        }

        #[test]
        #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive Metal ownership"]
        fn appended_growth_preserves_policy_forced_routes_and_packed_prefill() -> Result<()> {
            let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
            let metal = Metal::new()?;
            let hidden: Vec<_> = (0..HIDDEN)
                .flat_map(|index| {
                    bf16::from_f32((index as i32 % 17 - 8) as f32 / 128.)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect();
            let hidden_buffer = metal.shared_buffer(hidden.len())?;
            hidden_buffer.write_bytes(0, &hidden)?;
            let scores = [bf16::from_f32(0.1).to_bits(); TOP_K];
            let run_routes = |pool: &mut BoundedExperts, sources: &[u32; TOP_K]| {
                let mut batch = metal.batch()?;
                let output =
                    pool.encode_routed_device(&mut batch, &hidden_buffer, sources, &scores)?;
                batch.submit().wait()?;
                let mut bytes = vec![0; output.byte_len()];
                output.read_bytes(0, &mut bytes)?;
                Ok::<_, anyhow::Error>(bytes)
            };
            let packed_hidden = hidden.repeat(512);
            for layer in [0, 2] {
                let mlp = crate::mlp_batch::MlpBatch::new(&metal, &package, layer)?;
                let mut reference = BoundedExperts::new(&metal, package.expert_layer(layer)?, 192)?;
                let expected_packed = mlp.run_device_replay(&packed_hidden, 512, &mut reference)?;
                let selected: BTreeSet<_> = expected_packed
                    .1
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| u32::from_le_bytes(*word))
                    .collect();
                assert_eq!(selected.len(), TOP_K);
                let source = package.expert_layer(layer)?;
                let per_slot = Projection::ALL.into_iter().try_fold(0u64, |total, kind| {
                    let geometry = Geometry::from_projection(source.projection(kind))?;
                    Ok::<_, anyhow::Error>(
                        total
                            + IqkGemv::slot_stream_bytes(geometry)?
                                .into_iter()
                                .sum::<usize>() as u64,
                    )
                })?;
                let workspace =
                    crate::package::ExpertRowWorkspace::with_prefill_reader(source.row_bytes)?;
                for initial in [95, 145] {
                    let mut pool = BoundedExperts::with_storage(
                        &metal,
                        source.clone(),
                        initial,
                        workspace.clone(),
                        true,
                    )?;
                    let filler_slots = initial - TOP_K;
                    let filler: BTreeSet<_> = (0..512)
                        .filter(|source| !selected.contains(source))
                        .take(filler_slots)
                        .collect();
                    pool.ensure_group(&filler)?;
                    pool.ensure_group(&selected)?;
                    assert!(
                        selected
                            .iter()
                            .all(|source| pool.slot_of(*source).unwrap() >= filler_slots)
                    );
                    let routes = [
                        0,
                        1,
                        63,
                        64,
                        65,
                        filler_slots,
                        filler_slots + 1,
                        initial - 3,
                        initial - 2,
                        initial - 1,
                    ]
                    .map(|slot| pool.expert_at[slot].unwrap());
                    let expected_routes = run_routes(&mut reference, &routes)?;
                    assert_eq!(run_routes(&mut pool, &routes)?, expected_routes);
                    assert_eq!(
                        mlp.run_device_replay(&packed_hidden, 512, &mut pool)?,
                        expected_packed
                    );
                    let targets: &[usize] = if initial == 145 {
                        &[160, 176, 192, 204]
                    } else {
                        &[128, 192, 223]
                    };
                    for &target in targets {
                        let before = pool.snapshot();
                        let old_capacity = pool.capacity();
                        let quote = pool.quote_growth_suffix(target)?;
                        assert_eq!(
                            quote.added_requested_bytes,
                            per_slot * (target - old_capacity) as u64
                        );
                        assert_eq!(
                            quote.preparation_requested_bytes,
                            quote.added_requested_bytes + (511 * 18 * 8) as u64
                        );
                        let requested = metal.memory_stats()?.live_requested_buffer_bytes;
                        let allocations = target.div_ceil(16) - old_capacity / 16 + 1;
                        for allocation in 0..allocations {
                            let fault = metal.test_fail_allocation_after(allocation)?;
                            assert!(pool.prepare_growth_suffix(target).is_err());
                            drop(fault);
                            assert_eq!(pool.snapshot(), before);
                            assert_eq!(
                                metal.memory_stats()?.live_requested_buffer_bytes,
                                requested
                            );
                        }
                        {
                            let prepared = pool.prepare_growth_suffix(target)?;
                            assert_eq!(prepared.report(), quote);
                            assert_eq!(
                                metal.memory_stats()?.live_requested_buffer_bytes - requested,
                                quote.preparation_requested_bytes
                            );
                        }
                        assert_eq!(pool.snapshot(), before);
                        assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, requested);
                        let prepared = pool.prepare_growth_suffix(target)?;
                        let fault = metal.test_fail_allocation_after(0)?;
                        assert_eq!(prepared.commit(), quote);
                        drop(fault);
                        let mut expected = before;
                        expected.expert_at.resize(target, None);
                        expected.last_use.resize(target, 0);
                        assert_eq!(pool.snapshot(), expected);
                        assert_eq!(
                            metal.memory_stats()?.live_requested_buffer_bytes - requested,
                            quote.added_requested_bytes
                        );
                        assert!(pool.bounded_pipelines.is_some());
                        assert_eq!(run_routes(&mut pool, &routes)?, expected_routes);
                        assert_eq!(
                            mlp.run_device_replay(&packed_hidden, 512, &mut pool)?,
                            expected_packed
                        );
                        let new_sources: BTreeSet<_> = (0..512)
                            .filter(|source| pool.slot_of(*source).is_none())
                            .take(target - pool.slot_of.len())
                            .collect();
                        pool.ensure_group(&new_sources)?;
                        assert_eq!(pool.slot_of.len(), target);
                        let new_routes = [
                            0,
                            63,
                            64,
                            old_capacity - 1,
                            old_capacity,
                            target - 5,
                            target - 4,
                            target - 3,
                            target - 2,
                            target - 1,
                        ]
                        .map(|slot| pool.expert_at[slot].unwrap());
                        assert_eq!(
                            run_routes(&mut pool, &new_routes)?,
                            run_routes(&mut reference, &new_routes)?
                        );
                    }
                    assert!(pool.quote_growth_suffix(512).is_err());
                }
            }
            Ok(())
        }

        #[test]
        #[ignore = "requires exclusive GPU ownership"]
        fn chunk_capacity_guards_retired_selectors_before_address_reads() -> Result<()> {
            let metal = Metal::new()?;
            let source = format!(
                "{}\n{}",
                ROUTED_SOURCE,
                include_str!("../kernels/metal/iqk_chunked.metal")
            );
            for codec in ["iq2", "iq3"] {
                let gate = metal.compile_msl_strict(&source, &format!("chunk_gate_{codec}"))?;
                let down = metal.compile_msl_strict(&source, &format!("chunk_down_{codec}"))?;
                for (capacity, invalid_slot, flag_value, last_only) in [
                    (0u32, 0u32, 1u32, false),
                    (9, 0, 1, false),
                    (512, 0, 1, false),
                    (u32::MAX, 0, 1, false),
                    (64, 64, 1, false),
                    (64, u32::MAX, 1, false),
                    (64, u32::MAX, 0, false),
                    (64, u32::MAX, 2, false),
                    (64, 64, 1, true),
                ] {
                    let poison = metal.shared_buffer(8)?;
                    poison.write_bytes(0, &u64::MAX.to_le_bytes())?;
                    let flag = metal.shared_buffer(4)?;
                    flag.write_bytes(0, &flag_value.to_le_bytes())?;
                    let slots = metal.shared_buffer(40)?;
                    slots.write_bytes(
                        0,
                        &(0..10)
                            .map(|index| {
                                if last_only && index != 9 {
                                    0
                                } else {
                                    invalid_slot
                                }
                            })
                            .flat_map(|slot| slot.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let stages = metal.shared_buffer(TOP_K * DOWN_STORED * 2 + 16)?;
                    stages.write_bytes(0, &vec![0xa5; stages.byte_len()])?;
                    let output = metal.shared_buffer(HIDDEN * 2 + 16)?;
                    output.write_bytes(0, &vec![0xa5; output.byte_len()])?;
                    let binding = |buffer| Binding { buffer, offset: 0 };
                    let limit = capacity.to_le_bytes();
                    let mut batch = metal.batch()?;
                    // SAFETY: Invalid capacity/slots or publication flag exits
                    // uniformly before reading any undersized poison binding.
                    unsafe {
                        if !last_only {
                            batch.dispatch_1d_with_inline(
                                &gate,
                                &[
                                    binding(&poison),
                                    binding(&poison),
                                    binding(&poison),
                                    binding(&slots),
                                    binding(&poison),
                                    binding(&poison),
                                    binding(&stages),
                                    binding(&stages),
                                    binding(&stages),
                                    binding(&flag),
                                ],
                                &[&limit],
                                TOP_K * (INTERMEDIATE / 16) * 80,
                                80,
                            )?;
                        }
                        batch.dispatch_1d_with_inline(
                            &down,
                            &[
                                binding(&poison),
                                binding(&poison),
                                binding(&poison),
                                binding(&slots),
                                binding(&poison),
                                binding(&poison),
                                binding(&stages),
                                binding(&output),
                                binding(&flag),
                            ],
                            &[&limit],
                            (HIDDEN / 16) * 320,
                            320,
                        )?;
                    }
                    batch.submit().wait()?;
                    let mut actual = vec![0; stages.byte_len()];
                    stages.read_bytes(0, &mut actual)?;
                    assert!(actual.iter().all(|&byte| byte == 0xa5));
                    let mut actual = vec![0; output.byte_len()];
                    output.read_bytes(0, &mut actual)?;
                    assert!(actual[..HIDDEN * 2].iter().all(|&byte| byte == 0));
                    assert!(actual[HIDDEN * 2..].iter().all(|&byte| byte == 0xa5));
                }
            }
            Ok(())
        }

        #[test]
        #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
        fn prefetch_eof_preserves_publication_prefix_and_retries() -> Result<()> {
            let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
            let metal = Metal::new()?;
            for layer in [0, 2] {
                let source = package.expert_layer(layer)?;
                let workspace =
                    crate::package::ExpertRowWorkspace::with_prefill_reader(source.row_bytes)?;
                let mut pool =
                    BoundedExperts::with_workspace(&metal, source, 12, workspace.clone())?;
                let mut serial = BoundedExperts::new(&metal, package.expert_layer(layer)?, 12)?;
                let seeds = (0..12).collect::<Vec<_>>();
                pool.seed_residents(&seeds)?;
                serial.seed_residents(&seeds)?;
                workspace.fail_next_prefetch_for_test();
                let active = BTreeSet::from([20, 21]);
                let mut stream = metal.stream()?;
                {
                    let mut scope = PrefillLoadScope::new(&mut pool, &mut stream)?;
                    assert!(scope.load_group(&active).is_err());
                }
                serial.ensure_group_protected(&BTreeSet::from([20]), &active)?;
                assert_eq!(pool.snapshot(), serial.snapshot());
                assert!(pool.slot_of(20).is_some() && pool.slot_of(21).is_none());
                pool.check_writable()?;
                {
                    let mut scope = PrefillLoadScope::new(&mut pool, &mut stream)?;
                    scope.load_group(&active)?;
                }
                serial.ensure_group(&active)?;
                assert_eq!(pool.snapshot(), serial.snapshot());
            }
            Ok(())
        }

        #[test]
        #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
        fn narrow_prefetch_failure_preserves_sorted_publication_and_retry() -> Result<()> {
            fn payloads(pool: &BoundedExperts) -> Result<Vec<Vec<u8>>> {
                (0..pool.capacity)
                    .map(|slot| {
                        let (projections, local) =
                            BoundedExperts::projections_at(&pool.storage, slot)?;
                        let mut row = Vec::new();
                        for (projection, kind) in projections.into_iter().zip(Projection::ALL) {
                            let geometry = Geometry::from_projection(pool.layer.projection(kind))?;
                            let bindings = projection.stream_bindings();
                            for (binding, bytes) in bindings
                                .into_iter()
                                .zip(IqkGemv::slot_stream_bytes(geometry)?)
                            {
                                if bytes != 0 {
                                    let start = row.len();
                                    row.resize(start + bytes, 0);
                                    binding.buffer.read_bytes(
                                        binding.offset + local * bytes,
                                        &mut row[start..],
                                    )?;
                                }
                            }
                        }
                        Ok(row)
                    })
                    .collect()
            }

            let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
            let metal = Metal::new()?;
            let active = BTreeSet::from([2, 3, 4, 5, 6, 7, 8, 9, 20, 21]);
            let completed = active.iter().copied().filter(|&id| id != 21).collect();
            for layer in [0, 2] {
                for chunked in [false, true] {
                    let source = package.expert_layer(layer)?;
                    let workspace =
                        crate::package::ExpertRowWorkspace::with_prefill_reader(source.row_bytes)?;
                    let mut pool = BoundedExperts::with_storage(
                        &metal,
                        source.clone(),
                        12,
                        workspace.clone(),
                        chunked,
                    )?;
                    let mut serial = BoundedExperts::with_storage(
                        &metal,
                        source.clone(),
                        12,
                        crate::package::ExpertRowWorkspace::new(source.row_bytes)?,
                        chunked,
                    )?;
                    let seeds = (0..12).collect::<Vec<_>>();
                    pool.seed_residents(&seeds)?;
                    serial.seed_residents(&seeds)?;
                    workspace.fail_next_prefetch_for_test();
                    assert!(pool.ensure_group(&active).is_err());
                    workspace.check_idle()?;
                    serial.ensure_group_protected(&completed, &active)?;
                    assert_eq!(pool.snapshot(), serial.snapshot());
                    assert_eq!(payloads(&pool)?, payloads(&serial)?);
                    assert!(pool.slot_of(20).is_some() && pool.slot_of(21).is_none());
                    pool.check_writable()?;
                    let actual = pool.ensure_group(&active)?;
                    let expected = serial.ensure_group(&active)?;
                    assert_eq!(actual, expected);
                    assert_eq!(pool.snapshot(), serial.snapshot());
                    assert_eq!(payloads(&pool)?, payloads(&serial)?);
                    workspace.check_idle()?;
                    pool.check_writable()?;

                    // Resident touches before the first missing row remain
                    // visible when the owner workspace cannot be borrowed.
                    let hits = BTreeSet::from([2, 3, 4, 5, 6, 7, 8, 9]);
                    let next = hits.union(&BTreeSet::from([30, 31])).copied().collect();
                    let borrow = workspace.row(source.row_bytes)?;
                    assert!(pool.ensure_group(&next).is_err());
                    serial.ensure_group_protected(&hits, &next)?;
                    assert_eq!(pool.snapshot(), serial.snapshot());
                    // An all-hit group does not need the owner row workspace.
                    assert_eq!(pool.ensure_group(&hits)?, serial.ensure_group(&hits)?);
                    assert_eq!(pool.snapshot(), serial.snapshot());
                    drop(borrow);
                    workspace.check_idle()?;
                }
            }
            Ok(())
        }

        #[test]
        #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
        fn decode_prefetch_failure_preserves_sorted_publication_and_retry() -> Result<()> {
            let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
            let metal = Metal::new()?;
            let routes = [21, 9, 8, 7, 20, 6, 5, 4, 3, 2];
            let active: BTreeSet<_> = routes.into_iter().collect();
            let completed: BTreeSet<_> = active.iter().copied().filter(|&id| id != 21).collect();
            for layer in [0, 2] {
                for gated in [false, true] {
                    let source = package.expert_layer(layer)?;
                    let workspace =
                        crate::package::ExpertRowWorkspace::with_prefill_reader(source.row_bytes)?;
                    let mut pool =
                        BoundedExperts::with_workspace(&metal, source, 12, workspace.clone())?;
                    let mut serial = BoundedExperts::new(&metal, package.expert_layer(layer)?, 12)?;
                    let seeds = (0..12).collect::<Vec<_>>();
                    pool.seed_residents(&seeds)?;
                    serial.seed_residents(&seeds)?;
                    let event = metal.shared_event()?.point(1)?;
                    workspace.fail_next_prefetch_for_test();
                    // SAFETY: This test has no submitted pool readers; the gate
                    // remains unopened throughout failure and retry.
                    assert!(
                        unsafe { pool.ensure_with_gate(&routes, gated.then_some(&event)) }.is_err()
                    );
                    workspace.check_idle()?;
                    serial.ensure_group_protected(&completed, &active)?;
                    assert_eq!(pool.snapshot(), serial.snapshot());
                    assert!(pool.slot_of(20).is_some() && pool.slot_of(21).is_none());
                    pool.check_writable()?;
                    // SAFETY: The same quiescent pool and unopened gate apply.
                    let slots = unsafe { pool.ensure_with_gate(&routes, gated.then_some(&event))? };
                    let expected = serial.ensure(&routes)?;
                    assert_eq!(slots, expected);
                    assert_eq!(pool.snapshot(), serial.snapshot());
                    workspace.check_idle()?;
                    pool.check_writable()?;
                }
            }
            Ok(())
        }

        #[test]
        #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
        fn retained_read_failure_and_workspace_conflict_preserve_pool() -> Result<()> {
            let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
            let metal = Metal::new()?;
            for layer in [0, 2] {
                let mut pool = BoundedExperts::new(&metal, package.expert_layer(layer)?, 12)?;
                pool.seed_residents(&(0..12).collect::<Vec<_>>())?;
                let before = pool.snapshot();
                let source = pool.layer.clone();
                pool.layer = source.at_eof_for_test()?;
                assert!(pool.ensure_group(&BTreeSet::from([20])).is_err());
                assert_eq!(pool.snapshot(), before);
                let mut stream = metal.stream()?;
                {
                    let mut scope = PrefillLoadScope::new(&mut pool, &mut stream)?;
                    assert!(scope.load_group(&BTreeSet::from([20])).is_err());
                }
                assert_eq!(pool.snapshot(), before);
                pool.layer = source;
                let workspace = pool.row_workspace.clone();
                let borrow = workspace.row(pool.layer.row_bytes)?;
                assert!(pool.ensure_group(&BTreeSet::from([20])).is_err());
                assert_eq!(pool.snapshot(), before);
                drop(borrow);
                pool.ensure_group(&BTreeSet::from([20]))?;
                assert!(pool.slot_of(20).is_some());
                pool.check_writable()?;
            }
            Ok(())
        }

        #[test]
        #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
        fn package_overlap_preserves_policy_and_scope_cleanup() -> Result<()> {
            let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
            let metal = Metal::new()?;
            let read = metal.compile_msl_strict(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void read_slot(device const uchar* g [[buffer(0)]], device const uchar* u [[buffer(1)]], device const uchar* d [[buffer(2)]], device uint* out [[buffer(3)]], constant uint4& p [[buffer(4)]]) { out[p.w * 3] = g[p.x]; out[p.w * 3 + 1] = u[p.y]; out[p.w * 3 + 2] = d[p.z]; }",
                "read_slot",
            )?;
            for (layer, capacity) in [
                (0, 12),
                (0, 13),
                (0, 65),
                (0, 102),
                (0, 178),
                (2, 12),
                (2, 13),
                (2, 65),
                (2, 102),
                (2, 178),
            ] {
                let source = package.expert_layer(layer)?;
                let workspace =
                    crate::package::ExpertRowWorkspace::with_prefill_reader(source.row_bytes)?;
                let mut pool = BoundedExperts::with_workspace(&metal, source, capacity, workspace)?;
                let mut serial =
                    BoundedExperts::new(&metal, package.expert_layer(layer)?, capacity)?;
                let width = (capacity / 2) as u32;
                let groups: Vec<BTreeSet<u32>> = [
                    (0, width),
                    (0, width),
                    (width, 2 * width),
                    (2 * width, 3 * width),
                    (3 * width, 4 * width),
                ]
                .into_iter()
                .map(|(start, end)| (start..end).collect())
                .collect();
                let output = metal.shared_buffer(groups.len() * 3 * 4)?;
                let encode = |batch: &mut Batch,
                              pool: &BoundedExperts,
                              map: &[u32; 512],
                              source: u32,
                              index: usize|
                 -> Result<()> {
                    let (projections, local) = BoundedExperts::projections_at(
                        &pool.storage,
                        map[source as usize] as usize,
                    )?;
                    let bindings = projections.map(|projection| projection.stream_bindings()[0]);
                    let mut args = Projection::ALL
                        .into_iter()
                        .map(|kind| {
                            Ok(u32::try_from(
                                local
                                    * IqkGemv::slot_stream_bytes(Geometry::from_projection(
                                        pool.layer.projection(kind),
                                    )?)?[0],
                            )?)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    args.push(index as u32);
                    let bytes: Vec<_> = args.iter().flat_map(|word| word.to_le_bytes()).collect();
                    // SAFETY: Each qs address is inside this source's captured
                    // physical slot; the output contains three words per group.
                    unsafe {
                        batch.dispatch_1d_with_inline(
                            &read,
                            &[
                                bindings[0],
                                bindings[1],
                                bindings[2],
                                Binding {
                                    buffer: &output,
                                    offset: 0,
                                },
                            ],
                            &[&bytes],
                            1,
                            1,
                        )
                    }
                };
                let mut stream = metal.stream()?;
                {
                    let mut scope = PrefillLoadScope::new(&mut pool, &mut stream)?;
                    for (index, group) in groups.iter().enumerate() {
                        scope.retire_previous()?;
                        scope.load_group(group)?;
                        let source = *group.first().unwrap();
                        // SAFETY: The helper reads only one recorded source.
                        unsafe {
                            scope.submit_group(|batch, pool, map| {
                                encode(batch, pool, map, source, index)
                            })?;
                        }
                    }
                    scope.finish()?;
                }
                assert!(pool.check_writable().is_err());
                assert!(PrefillLoadScope::new(&mut pool, &mut stream).is_err());
                assert!(pool.ensure_group(&BTreeSet::from([511])).is_err());
                stream.wait()?;
                let mut protected = BTreeSet::new();
                for group in &groups {
                    serial.ensure_group_protected(group, &protected)?;
                    protected = group.clone();
                }
                assert_eq!(pool.snapshot(), serial.snapshot());
                let mut actual = vec![0; output.byte_len()];
                output.read_bytes(0, &mut actual)?;
                let mut expected = Vec::new();
                for group in &groups {
                    let row = pool.layer.read_row(*group.first().unwrap() as usize)?;
                    for projection in Projection::ALL {
                        expected.extend_from_slice(
                            &(pool.layer.projection(projection).bytes(&row)?[0] as u32)
                                .to_le_bytes(),
                        );
                    }
                }
                assert_eq!(actual, expected);

                // Catch an encode error after two submitted groups. It cancels
                // the active reader, drains both submitted ones, and poisons
                // finish even when the caller retains the scope.
                {
                    let mut scope = PrefillLoadScope::new(&mut pool, &mut stream)?;
                    for (index, group) in groups.iter().take(2).enumerate() {
                        scope.load_group(group)?;
                        unsafe {
                            scope.submit_group(|batch, pool, map| {
                                encode(batch, pool, map, 0, index)
                            })?;
                        }
                    }
                    scope.retire_previous()?;
                    scope.load_group(&groups[2])?;
                    let error = unsafe {
                        scope.submit_group(|batch, pool, map| {
                            encode(batch, pool, map, *groups[2].first().unwrap(), 2)?;
                            anyhow::bail!("injected encode failure")
                        })
                    }
                    .unwrap_err();
                    assert_eq!(error.to_string(), "injected encode failure");
                    scope.pool.check_writable()?;
                    assert!(scope.finish().is_err());
                }
                pool.check_writable()?;
                // Dropping an unfinished scope drains outstanding readers too.
                {
                    let mut scope = PrefillLoadScope::new(&mut pool, &mut stream)?;
                    scope.load_group(&groups[0])?;
                    unsafe {
                        scope.submit_group(|batch, pool, map| encode(batch, pool, map, 0, 0))?;
                    }
                }
                pool.check_writable()?;
                pool.ensure_group(&BTreeSet::from([511]))?;
            }
            Ok(())
        }
    }

    #[cfg(test)]
    mod pool_tests {
        use super::*;

        #[test]
        fn prefill_captures_only_published_unique_physical_slots() {
            let sources = BTreeSet::from([4, 9, 17]);
            let mut map = [512; 512];
            map[4] = 11;
            map[9] = 0;
            map[17] = 5;
            map[20] = 3;
            let group = PrefillGroup::capture(&sources, &map, 12).unwrap();
            assert_eq!(group.slots, BTreeSet::from([0, 5, 11]));
            assert_eq!(group.sources, sources);
            assert_eq!(group.map[20], 512);
            assert!(group.completion.is_none());
            // The encoded-but-unsubmitted footprint is already captured.
            assert!(group.slots.contains(&(group.map[4] as usize)));
            map[9] = 11;
            assert!(PrefillGroup::capture(&sources, &map, 12).is_err());
            map[9] = 512;
            assert!(PrefillGroup::capture(&sources, &map, 12).is_err());
            assert!(PrefillGroup::capture(&BTreeSet::from([512]), &map, 12).is_err());
            assert!(PrefillGroup::capture(&BTreeSet::new(), &map, 12).is_err());
        }

        #[test]
        fn failed_projection_copy_unpublishes_victim_and_retry_fills_every_projection() {
            // Fail before any write, inside one projection, and between each
            // pair of projections. Previously completed slots remain published.
            for fail_after in 0..6 {
                let mut map = BTreeMap::from([(7, 0), (8, 1)]);
                let mut occupied = [Some(7), Some(8)];
                let mut bytes = [[7u8; 6], [8u8; 6]];
                let error = publish_replacement(&mut map, &mut occupied, 9, 0, || {
                    for (index, byte) in bytes[0].iter_mut().enumerate() {
                        if index == fail_after {
                            anyhow::bail!("injected projection failure");
                        }
                        *byte = 9;
                    }
                    Ok(())
                });
                assert!(error.is_err());
                assert_eq!(occupied, [None, Some(8)]);
                assert_eq!(map, BTreeMap::from([(8, 1)]));
                assert_eq!(bytes[1], [8; 6]);
                publish_replacement(&mut map, &mut occupied, 9, 0, || {
                    bytes[0].fill(9);
                    Ok(())
                })
                .unwrap();
                assert_eq!(occupied, [Some(9), Some(8)]);
                assert_eq!(map, BTreeMap::from([(8, 1), (9, 0)]));
                assert_eq!(bytes[0], [9; 6]);
            }
        }

        #[test]
        fn previous_and_current_slot_union_protects_shared_readers() {
            let residents = [Some(4), Some(9), Some(17), Some(20)];
            let recency = [1, 2, 3, 4];
            let frequency = BTreeMap::new();
            let mut map = [512; 512];
            for (slot, source) in residents.into_iter().enumerate() {
                map[source.unwrap() as usize] = slot as u32;
            }
            let previous = PrefillGroup::capture(&BTreeSet::from([4, 9]), &map, 4).unwrap();
            let current = PrefillGroup::capture(&BTreeSet::from([9, 17]), &map, 4).unwrap();
            let protected = previous.sources.union(&current.sources).copied().collect();
            assert_eq!(
                least_frequent_slot(&residents, &recency, &frequency, &protected),
                Some(3)
            );
            // Retiring previous permits slot 0 only; shared slot 1 still has
            // current's reader, independent of how many commands use it.
            assert_eq!(
                least_frequent_slot(&residents, &recency, &frequency, &current.sources),
                Some(0)
            );
            assert!(current.slots.contains(&1));
        }

        #[test]
        fn frequency_precedes_recency_and_protected_sources_survive() {
            let residents = [Some(4), Some(51), Some(58), Some(114)];
            let recency = [1, 2, 3, 4];
            let mut frequency = BTreeMap::from([(4, 1), (51, 1), (58, 3), (114, 1)]);
            let protected = BTreeSet::from([58]);
            assert_eq!(
                least_frequent_slot(&residents, &recency, &frequency, &protected),
                Some(0)
            );
            frequency.insert(4, 2);
            assert_eq!(
                least_frequent_slot(&residents, &recency, &frequency, &protected),
                Some(1)
            );
            assert_eq!(
                least_frequent_slot(
                    &residents,
                    &recency,
                    &frequency,
                    &BTreeSet::from([4, 51, 58, 114])
                ),
                None
            );
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{
    BoundedExperts, ExpertDemandCounts, ExpertGrowthReport, ExpertPoolSnapshot,
    ExpertReleaseReport, PendingOutput, RoutedStages,
};

#[cfg(target_os = "macos")]
pub(crate) use gpu::PrefillLoadScope;

#[cfg(target_os = "macos")]
pub(crate) use gpu::{PreparedExpertSuffixGrowth, PreparedExpertSuffixRelease};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires IQK_SIGMOID_ORACLE"]
    fn sigmoid_matches_external_fp16_oracle() {
        let path = std::env::var("IQK_SIGMOID_ORACLE")
            .expect("IQK_SIGMOID_ORACLE is required for this oracle check");
        let reference = std::fs::read(path).unwrap();
        let native = sigmoid_fp16_table();
        assert_eq!(native.len(), reference.len());
        let differing: Vec<_> = native
            .as_chunks::<2>()
            .0
            .iter()
            .zip(reference.as_chunks::<2>().0)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, u16::from_le_bytes(*a), u16::from_le_bytes(*b)))
            .collect();
        assert!(
            differing.is_empty(),
            "{} sigmoid mismatches, first: {:?}",
            differing.len(),
            &differing[..differing.len().min(12)]
        );
    }
}

//! Header-only payload accounting and a pure ordinary-path workspace estimate.
//!
//! Payload fields are logical lengths. The separate workspace estimate counts
//! source-level buffer requests and explicit reserves. Neither is process RSS,
//! device allocation size, or a universal peak-memory guarantee.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use crate::kvarn::{
    DEVICE_SEAL_WORKSPACE_BYTES, SINK_TOKENS, TAIL_CAPACITY, TILE_RECORD_BYTES, TILE_TOKENS,
    TOKEN_BYTES, partition,
};
use crate::model::{MixerKind, QwenConfig};
use crate::package::{Package, Projection, WeightFormat};
use crate::qsa_index::{COMPRESS_RATIO, INDEX_WIDTH};

const GENERATION_VOCABULARY: u64 = 248_320;
pub(crate) const DEVICE_GREEDY_REQUESTED_BYTES: u64 = 2 * GENERATION_VOCABULARY
    + 4 * (GENERATION_VOCABULARY / 32)
    + 8 * (GENERATION_VOCABULARY / 256)
    + 10;
pub(crate) const DEVICE_GREEDY_HOST_BYTES: u64 = 4 * (GENERATION_VOCABULARY / 32);
const SAMPLING_PREPARATION_DEVICE_BYTES: u64 = 4 * GENERATION_VOCABULARY + 4;
const SAMPLING_HOST_WORKSPACE_BYTES: u64 = 27 * GENERATION_VOCABULARY + 262_144;

/// Process, compiler, driver and allocator overhead outside labelled payloads.
/// This policy reserve is not a bound on native allocation or later pressure.
pub const PROCESS_OVERHEAD_ALLOWANCE_BYTES: u64 = 2 << 30;

/// Retained payloads of one combined ANE owner. The single dynamic
/// input includes activations and weights; no decoded weight copy is retained.
/// Native program, pipeline, compiler and allocation overhead remain within
/// the caller's separate process allowance. These lengths are not an RSS bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnePrefillMemory {
    pub host_staging_bytes: u64,
    pub metal_staging_bytes: u64,
    pub iosurface_payload_bytes: u64,
    pub qsa_attended_bytes: u64,
}

/// Logical payload lengths for one supported physical program shape.
pub fn ane_prefill_memory(physical_rows: usize) -> Result<AnePrefillMemory> {
    ensure!(
        matches!(physical_rows, 4096 | 8192),
        "ANE physical rows must be 4096 or 8192"
    );
    let rows = physical_rows as u64;
    let staging = 2 * 2560 * (rows + 6144) + 2 * 6144 * rows;
    Ok(AnePrefillMemory {
        host_staging_bytes: staging,
        metal_staging_bytes: staging + 4,
        iosurface_payload_bytes: staging,
        qsa_attended_bytes: 4 * 6144 * rows,
    })
}

/// Charge once, alongside the ordinary process allowance, before selecting
/// expert capacity or workspace phases. The physical owner does not resize
/// when ordinary GPU scratch changes between 2048 and 4096 rows.
pub fn ane_prefill_allowance_bytes(physical_rows: usize) -> Result<u64> {
    let memory = ane_prefill_memory(physical_rows)?;
    Ok(memory.host_staging_bytes
        + memory.metal_staging_bytes
        + memory.iosurface_payload_bytes
        + memory.qsa_attended_bytes)
}

// Persistent operator and session buffers plus one request's reusable
// sampling scratch, outside packed package tensors. This includes the IQ3
// lookup-table upper bound and must be revised with those operator buffers.
const ORDINARY_CORE_BUFFER_BYTES: u64 = 88_946_110
    + 12 * 2 // empty device-record sentinels
    + 40_968 // device graph scratch
    + 2 * GENERATION_VOCABULARY // Session-owned device final-head row
    + DEVICE_GREEDY_REQUESTED_BYTES // reusable greedy adjustment and reduction
    + DEVICE_GREEDY_HOST_BYTES // reusable host presence bitset
    + SAMPLING_PREPARATION_DEVICE_BYTES // lazy same-device stochastic reductions
    + SAMPLING_HOST_WORKSPACE_BYTES // ID-only sampler and preparation host scratch
    + 481_318 // PLE device weights, scratch and alternate history
    + 12 * (24 * 64 * 258 * 4 + 4) // online QSA partials and guarded count
    + 12 * (12_607 + 262_144) // reusable index scratch at maximum model context
    + 2 * GENERATION_VOCABULARY // optional cached Session host row
    + 48 * (1_024 + 2_048 + 40 + 20 + 4) // persistent per-layer decode router
    + 48 * 92 // bounded decode mailboxes and publication flags
    + DEVICE_SEAL_WORKSPACE_BYTES as u64; // one encoder-owned seal workspace
// Device index state keeps a full pending group and a nonempty compressed
// allocation even before a complete group exists. Logical payload accounting
// is separate, so retain this small conservative capacity allowance per layer.
const QSA_DEVICE_INDEX_FIXED_CAPACITY_BYTES: u64 =
    (COMPRESS_RATIO as u64 + 1) * INDEX_WIDTH as u64 * 2;

/// Requested bytes for the engine-owned capacity-512 prefill workspace.
/// All operator workspaces coexist even when the caller submits fewer rows.
/// Prefill construction checks this inventory against its actual buffers.
pub(crate) fn native_prefill_workspace_bytes(context: usize) -> Result<usize> {
    native_prefill_workspace_bytes_for_capacity(context, crate::prefill_capacity::DEFAULT_ROWS)
}

pub(crate) fn native_prefill_workspace_bytes_for_capacity(
    context: usize,
    capacity: usize,
) -> Result<usize> {
    crate::prefill_capacity::validate_capacity(capacity)?;
    ensure!(
        (1..=262_144).contains(&context),
        "prefill workspace context is invalid"
    );
    // Dense scratch is a maximum over every supported width. Both wider
    // workspaces peak at 1024 rows because the router then uses split-K.
    let dense = match capacity {
        512 => 2_097_152,
        2048 | 4096 => 4_194_304,
        8192 => 4_194_304,
        _ => unreachable!("validated prefill capacity"),
    };
    let fixed = capacity * 42_248 + 2 // HC transient
        + capacity * 51_216 // two HC result sets
        + capacity * 124_384 + 61_444 // GDN and one convolution spare
        + dense
        + capacity * 94_610 + (512 + (capacity * 10).div_ceil(8)) * 12 + 4 // MLP
        + capacity * 92_160; // PLE
    let projection =
        capacity * 2 * (640 + 12288 + 512 + 512 + 64 + 64 + 512 + 128 + 6144 + 6144 + 512);
    let index = capacity * 2051 * 5
        + capacity * 4 // selected counts
        + 64 * 512 * 4 // one selected-group row per query in a 64-row tile
        + 64 * (context / COMPRESS_RATIO).max(1) * 4;
    let gathered = 64 * (2 * (2 * 2051 * 256 * 2) + 2051);
    let attention = 64 * 2051 + 64 * 4 + 24 * 64 * 258 * 4 + 4 + capacity * (6144 + 2560) * 2;
    Ok(fixed + projection + index + gathered + attention)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectPayloadBytes {
    pub embedding_cpu_bytes: u64,
    pub matrix_count: usize,
    pub matrix_packed_bytes: u64,
    pub auxiliary_count: usize,
    pub auxiliary_bytes: u64,
    /// Largest single direct tensor payload returned by the package reader.
    pub largest_direct_payload_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertLayerBytes {
    pub layer: usize,
    pub experts: usize,
    /// Gate, up and down projection bytes for one physical slot.
    pub projection_bytes_per_slot: [u64; 3],
    pub packed_bytes_per_slot: u64,
    pub full_bundle_payload_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedSessionBytes {
    pub gdn_recurrent_f32: u64,
    pub gdn_convolution_bf16: u64,
    pub qsa_sink_kv_bf16: u64,
    pub qsa_tail_kv_bf16: u64,
    pub ple_history_bf16: u64,
}

/// Occupied KVarN record and QSA index payload at one committed token frontier.
/// The fixed sink/tail arrays are reported separately in `FixedSessionBytes`.
/// Vec spare capacity, temporary copies and allocation overhead are excluded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrowingSessionPayloadBytes {
    pub frontier_tokens: usize,
    pub kvarn_record_tiles_per_qsa_layer: usize,
    pub qsa_index_complete_groups_per_layer: usize,
    pub qsa_index_pending_rows_per_layer: usize,
    pub kvarn_records: u64,
    pub qsa_index_compressed_bf16: u64,
    pub qsa_index_pending_bf16: u64,
    pub total_logical_bytes: u64,
}

/// Package-backed byte inventory. It does not include decoded scratch,
/// lookup tables, allocator rounding or host copies. Growing body/index payload
/// is available separately at a requested frontier.
pub struct PackageFootprint {
    pub artifact_id: String,
    pub direct: DirectPayloadBytes,
    pub expert_layers: Vec<ExpertLayerBytes>,
    pub packed_expert_bytes_per_uniform_slot: u64,
    pub full_expert_bundle_payload_bytes: u64,
    pub fixed_session: FixedSessionBytes,
    max_expert_slots: usize,
    max_context_tokens: usize,
    qsa_layers: usize,
}

/// Trusted inputs for a bounded routed-expert pool calculation. All byte
/// estimates must include their allocations at the selected context limit.
/// In particular, `bounded_workspace_bytes` must be a conservative bound for
/// every candidate capacity through `max_slots`, not only the final choice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapacityInputs {
    /// Process memory envelope after any host or device reserve is applied.
    pub process_budget_bytes: u64,
    /// Optional explicit user byte ceiling on that envelope.
    pub user_memory_limit_bytes: Option<u64>,
    /// Persistent nonexpert bytes, including fixed model and session state.
    pub fixed_nonexpert_bytes: u64,
    /// Conservative growing-cache allowance at the intended context limit.
    pub context_reserve_bytes: u64,
    /// Bounded transient workspace, or `None` when it is not yet known.
    pub bounded_workspace_bytes: Option<u64>,
    /// Packed expert bytes for one uniform physical slot across all layers.
    pub packed_expert_bytes_per_slot: u64,
    /// At least ten slots are needed for the model's top-ten router.
    pub minimum_slots: usize,
    /// Number of source experts declared by the package, at most 512.
    pub max_slots: usize,
}

/// Sum of the supplied estimates at the selected slot count. This is not a
/// measured or universal process-memory peak.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapacityPlan {
    pub slots: usize,
    pub effective_budget_bytes: u64,
    pub accounted_peak_bytes: u64,
    pub remaining_budget_bytes: u64,
}

/// Immutable serving commitments from the initial bounded-pool admission.
/// Capacity changes credit or charge weights while context, caches,
/// workspaces and retained tables remain reserved at their admitted limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertCapacityControlBudget {
    initial_capacity: usize,
    maximum_capacity: usize,
    startup_budget_bytes: u64,
    effective_budget_bytes: u64,
    initial_serving_peak_bytes: u64,
    packed_expert_bytes_per_slot: u64,
    replacement_address_table_overlap_bytes: u64,
    process_overhead_allowance_bytes: u64,
    operating_growth: bool,
}

impl ExpertCapacityControlBudget {
    pub fn from_admission(
        footprint: &PackageFootprint,
        estimate: &OrdinaryPathEstimate,
        plan: &CapacityPlan,
    ) -> Result<Self> {
        ensure!(
            (10..512).contains(&plan.slots),
            "expert capacity control requires an initially bounded pool"
        );
        let phases = estimate.phase_component_sums()?;
        ensure!(
            phases.map(|(_, bytes)| bytes)
                == [
                    estimate.startup_phase_peak_bytes,
                    estimate.engine_construction_phase_peak_bytes,
                    estimate.serving_phase_peak_bytes,
                ]
                && largest_phase(phases)
                    == (
                        estimate.limiting_phase,
                        estimate.accounted_component_sum_bytes
                    ),
            "expert capacity control phase peaks differ from admission"
        );
        let selected_expert_bytes = footprint.packed_expert_bytes_at_capacity(plan.slots)?;
        let expert_surplus = estimate
            .expert_stream_payload_at_ceiling_bytes
            .checked_sub(selected_expert_bytes)
            .context("expert capacity control exceeds admission ceiling")?;
        ensure!(
            estimate
                .accounted_component_sum_bytes
                .checked_sub(expert_surplus)
                == Some(plan.accounted_peak_bytes)
                && plan.accounted_peak_bytes <= plan.effective_budget_bytes
                && plan
                    .effective_budget_bytes
                    .checked_sub(plan.accounted_peak_bytes)
                    == Some(plan.remaining_budget_bytes),
            "expert capacity control plan differs from admission"
        );
        let initial_metadata = footprint.expert_storage_metadata_bytes(plan.slots)?;
        ensure!(
            estimate.expert_storage_metadata_reserve_bytes >= initial_metadata,
            "expert capacity control metadata exceeds admission reserve"
        );
        let ready_bytes = mul(&[u64::try_from(footprint.expert_layers.len())?, 4])?;
        let replacement_address_table_overlap_bytes = initial_metadata
            .checked_sub(ready_bytes)
            .context("expert capacity control metadata omits publication buffers")?;
        ensure!(
            footprint.packed_expert_bytes_per_uniform_slot > 0,
            "expert capacity control slot byte count must be positive"
        );
        Ok(Self {
            initial_capacity: plan.slots,
            maximum_capacity: plan.slots,
            startup_budget_bytes: plan.effective_budget_bytes,
            effective_budget_bytes: plan.effective_budget_bytes,
            initial_serving_peak_bytes: estimate
                .serving_phase_peak_bytes
                .checked_sub(expert_surplus)
                .context("expert capacity control surplus exceeds serving peak")?,
            packed_expert_bytes_per_slot: footprint.packed_expert_bytes_per_uniform_slot,
            replacement_address_table_overlap_bytes,
            process_overhead_allowance_bytes: estimate.driver_allocator_allowance_bytes,
            operating_growth: false,
        })
    }

    pub fn initial_capacity(&self) -> usize {
        self.initial_capacity
    }

    pub fn maximum_capacity(&self) -> usize {
        self.maximum_capacity
    }

    pub fn startup_budget_bytes(&self) -> u64 {
        self.startup_budget_bytes
    }

    pub fn operating_budget_bytes(&self) -> u64 {
        self.effective_budget_bytes
    }

    pub fn initial_serving_peak_bytes(&self) -> u64 {
        self.initial_serving_peak_bytes
    }

    pub(crate) fn process_overhead_allowance_bytes(&self) -> u64 {
        self.process_overhead_allowance_bytes
    }

    /// Verify that a wider request workspace can retain at least 64 expert
    /// slots within the existing operating budget. Context, caches and the
    /// process allowance, including any reserved ANE payload, remain charged.
    /// This proof does not change the ordinary budget or allocate storage.
    pub fn validate_prefill_phase(
        &self,
        footprint: &PackageFootprint,
        context: usize,
        ordinary_rows: usize,
        wide_rows: usize,
    ) -> Result<()> {
        self.with_prefill_width(footprint, context, ordinary_rows, wide_rows)
            .map(|_| ())
    }

    /// A request phase changes scratch, chunk transients and experts.
    /// Context, retained state, process allowance and publication overlap keep
    /// their complete bounds. This does not admit a fresh engine construction.
    pub(crate) fn with_prefill_width(
        self,
        footprint: &PackageFootprint,
        context: usize,
        old_rows: usize,
        new_rows: usize,
    ) -> Result<Self> {
        crate::prefill_capacity::validate_capacity(old_rows)?;
        crate::prefill_capacity::validate_capacity(new_rows)?;
        ensure!(
            self.packed_expert_bytes_per_slot > 0,
            "phase expert slot credit is zero"
        );
        ensure!(
            footprint.packed_expert_bytes_per_uniform_slot == self.packed_expert_bytes_per_slot,
            "phase package differs from admitted expert credit"
        );
        let old = footprint.estimate_fixed_capacity_prefill(
            context,
            self.initial_capacity,
            old_rows,
            self.process_overhead_allowance_bytes,
        )?;
        let new = footprint.estimate_fixed_capacity_prefill(
            context,
            self.initial_capacity,
            new_rows,
            self.process_overhead_allowance_bytes,
        )?;
        let nonexpert = self
            .initial_serving_peak_bytes
            .checked_sub(mul(&[
                self.packed_expert_bytes_per_slot,
                self.initial_capacity as u64,
            ])?)
            .context("phase scratch differs from admitted serving commitments")?;
        let nonexpert = if new.serving_phase_peak_bytes >= old.serving_phase_peak_bytes {
            nonexpert.checked_add(new.serving_phase_peak_bytes - old.serving_phase_peak_bytes)
        } else {
            nonexpert.checked_sub(old.serving_phase_peak_bytes - new.serving_phase_peak_bytes)
        }
        .context("phase transients differ from admitted serving commitments")?;
        let available = self
            .effective_budget_bytes
            .checked_sub(nonexpert)
            .and_then(|bytes| bytes.checked_sub(self.replacement_address_table_overlap_bytes))
            .context("phase commitments exceed operating budget")?;
        let maximum = (available / self.packed_expert_bytes_per_slot) as usize;
        ensure!(
            maximum >= 64,
            "phase workspace cannot retain the adaptive expert floor"
        );
        let maximum = maximum.min(self.maximum_capacity);
        let initial = self.initial_capacity.min(maximum);
        Ok(Self {
            initial_capacity: initial,
            maximum_capacity: maximum,
            initial_serving_peak_bytes: add(
                nonexpert,
                mul(&[self.packed_expert_bytes_per_slot, initial as u64])?,
            )?,
            operating_growth: true,
            ..self
        })
    }

    pub(crate) fn workspace_budget_after_release(
        &self,
        context: usize,
        current_rows: usize,
        capacity: usize,
    ) -> Result<u64> {
        ensure!(
            (64..=self.maximum_capacity).contains(&capacity),
            "phase expert capacity is outside its extent"
        );
        let scratch = native_prefill_workspace_bytes_for_capacity(context, current_rows)? as u64;
        let nonexpert = self
            .initial_serving_peak_bytes
            .checked_sub(mul(&[
                self.packed_expert_bytes_per_slot,
                self.initial_capacity as u64,
            ])?)
            .and_then(|bytes| bytes.checked_sub(scratch))
            .context("phase scratch exceeds admitted nonexpert commitments")?;
        self.effective_budget_bytes
            .checked_sub(add(
                nonexpert,
                mul(&[self.packed_expert_bytes_per_slot, capacity as u64])?,
            )?)
            .context("phase workspace preparation exceeds its operating envelope")
    }

    #[cfg(test)]
    pub(crate) fn test_with_prefill_width(
        self,
        footprint: &PackageFootprint,
        context: usize,
        old_rows: usize,
        new_rows: usize,
    ) -> Result<Self> {
        self.with_prefill_width(footprint, context, old_rows, new_rows)
    }

    #[cfg(test)]
    pub(crate) fn test_workspace_budget_after_release(
        &self,
        context: usize,
        current_rows: usize,
        capacity: usize,
    ) -> Result<u64> {
        self.workspace_budget_after_release(context, current_rows, capacity)
    }

    #[cfg(test)]
    pub(crate) fn test_phase_expert_preparation(
        &self,
        current: usize,
        target: usize,
    ) -> Result<u64> {
        let mut budget = *self;
        budget.operating_growth = true;
        budget.preparation_budget_bytes(current, target)
    }

    /// Apply an independent operating envelope while retaining every serving
    /// commitment from startup. Load staging is complete before recovery, so
    /// the operating ceiling uses the serving phase alone. The immutable
    /// prefill workspace, context and cache reserves remain fully charged.
    pub fn with_operating_budget(
        mut self,
        footprint: &PackageFootprint,
        admitted: &OrdinaryPathEstimate,
        context_limit: usize,
        prefill_capacity: usize,
        operating_budget_bytes: u64,
    ) -> Result<Self> {
        ensure!(
            !self.operating_growth
                && self.maximum_capacity == self.initial_capacity
                && self.packed_expert_bytes_per_slot > 0,
            "expert operating budget was already resolved"
        );
        let admitted_expert_surplus = admitted
            .expert_stream_payload_at_ceiling_bytes
            .checked_sub(footprint.packed_expert_bytes_at_capacity(self.initial_capacity)?)
            .context("expert operating capacity exceeds startup estimate")?;
        ensure!(
            admitted
                .serving_phase_peak_bytes
                .checked_sub(admitted_expert_surplus)
                == Some(self.initial_serving_peak_bytes)
                && footprint.packed_expert_bytes_per_uniform_slot
                    == self.packed_expert_bytes_per_slot,
            "expert operating commitments differ from startup admission"
        );
        let operating = footprint.estimate_ordinary_path_with_prefill_capacity(
            OrdinaryPathInputs {
                context_limit,
                max_prefill_rows: context_limit.min(prefill_capacity),
                candidate_slot_ceiling: 511,
                uncounted_core_reserve_bytes: admitted.uncounted_core_reserve_bytes,
                vec_spare_growth_overlap_reserve_bytes: Some(
                    admitted.vec_spare_growth_overlap_reserve_bytes,
                ),
                prefix_retained_budget_bytes: admitted.prefix_retained_budget_bytes,
                snapshot_capture_restore_overlap_reserve_bytes: Some(
                    admitted.snapshot_capture_restore_overlap_reserve_bytes,
                ),
                driver_allocator_allowance_bytes: admitted.driver_allocator_allowance_bytes,
            },
            prefill_capacity,
        )?;
        let nonexpert_bytes = operating
            .serving_phase_peak_bytes
            .checked_sub(operating.expert_stream_payload_at_ceiling_bytes)
            .context("expert operating payload exceeds serving phase")?;
        let initial_serving_peak_bytes = add(
            nonexpert_bytes,
            footprint.packed_expert_bytes_at_capacity(self.initial_capacity)?,
        )?;
        self.initial_serving_peak_bytes = self
            .initial_serving_peak_bytes
            .max(initial_serving_peak_bytes);
        let nonexpert_bytes = self
            .initial_serving_peak_bytes
            .checked_sub(footprint.packed_expert_bytes_at_capacity(self.initial_capacity)?)
            .context("expert operating payload exceeds initial serving phase")?;
        let available_expert_bytes = operating_budget_bytes
            .checked_sub(nonexpert_bytes)
            .context("expert operating commitments exceed the operating envelope")?;
        ensure!(
            self.initial_serving_peak_bytes <= operating_budget_bytes,
            "expert operating envelope cannot preserve the initial serving capacity"
        );
        let recoverable_expert_bytes =
            available_expert_bytes.saturating_sub(self.replacement_address_table_overlap_bytes);
        let maximum_capacity = usize::try_from(
            (recoverable_expert_bytes / self.packed_expert_bytes_per_slot).min(511),
        )?
        .max(self.initial_capacity);
        self.maximum_capacity = maximum_capacity;
        self.effective_budget_bytes = operating_budget_bytes;
        self.operating_growth = true;
        Ok(self)
    }

    /// Validate actual serving commitments before loading an Engine. A smaller
    /// configuration may reuse the admitted envelope when its complete serving
    /// bound fits and its expert allocation credits remain identical.
    pub fn validate_serving(
        &self,
        footprint: &PackageFootprint,
        max_context: usize,
        prefill_capacity: usize,
        memory_retained_bytes: u64,
        disk_snapshot_bytes: u64,
    ) -> Result<()> {
        ensure!(
            footprint.packed_expert_bytes_per_uniform_slot == self.packed_expert_bytes_per_slot
                && self.packed_expert_bytes_per_slot > 0,
            "expert capacity control weight credit differs from serving package"
        );
        let actual_metadata = footprint.expert_storage_metadata_bytes(self.initial_capacity)?;
        let ready_bytes = mul(&[u64::try_from(footprint.expert_layers.len())?, 4])?;
        ensure!(
            actual_metadata.checked_sub(ready_bytes)
                == Some(self.replacement_address_table_overlap_bytes),
            "expert capacity control address overlap differs from serving package"
        );
        let actual = footprint.estimate_fixed_capacity_prefill(
            max_context,
            self.initial_capacity,
            prefill_capacity,
            self.process_overhead_allowance_bytes,
        )?;
        let actual_serving_peak = sum(&[
            actual.serving_phase_peak_bytes,
            memory_retained_bytes,
            disk_snapshot_bytes,
        ])?;
        ensure!(
            actual_serving_peak <= self.initial_serving_peak_bytes,
            "expert capacity control serving commitments need {actual_serving_peak} bytes, admitted bound is {}",
            self.initial_serving_peak_bytes
        );
        Ok(())
    }

    /// Additional Metal requested-buffer headroom for a recovery preparation.
    /// The Engine independently quotes all new allocations before using this
    /// budget. Appended chunks retain the existing partial tail, so preparation
    /// charges only added slots and replacement address tables. This component
    /// contract does not infer RSS or resample free RAM.
    pub fn preparation_budget_bytes(
        &self,
        current_capacity: usize,
        target_capacity: usize,
    ) -> Result<u64> {
        ensure!(
            (10..512).contains(&self.initial_capacity)
                && (self.initial_capacity..512).contains(&self.maximum_capacity)
                && (10..=self.maximum_capacity).contains(&current_capacity)
                && (self.operating_growth
                    || current_capacity == self.initial_capacity
                    || current_capacity.is_multiple_of(64))
                && (current_capacity..=self.maximum_capacity).contains(&target_capacity)
                && (self.operating_growth
                    || target_capacity == self.initial_capacity
                    || target_capacity.is_multiple_of(64)),
            "expert recovery capacity differs from the admitted chunk extent"
        );
        if target_capacity == current_capacity {
            return Ok(0);
        }
        ensure!(
            self.packed_expert_bytes_per_slot > 0,
            "expert recovery slot byte count must be positive"
        );
        let current_serving_peak = if current_capacity <= self.initial_capacity {
            let released_expert_bytes = mul(&[
                self.packed_expert_bytes_per_slot,
                u64::try_from(self.initial_capacity - current_capacity)?,
            ])?;
            self.initial_serving_peak_bytes
                .checked_sub(released_expert_bytes)
                .context("released expert bytes exceed the admitted serving peak")?
        } else {
            add(
                self.initial_serving_peak_bytes,
                mul(&[
                    self.packed_expert_bytes_per_slot,
                    u64::try_from(current_capacity - self.initial_capacity)?,
                ])?,
            )?
        };
        let added_expert_bytes = mul(&[
            self.packed_expert_bytes_per_slot,
            u64::try_from(target_capacity - current_capacity)?,
        ])?;
        let preparation_bytes = sum(&[
            added_expert_bytes,
            self.replacement_address_table_overlap_bytes,
        ])?;
        let preparation_peak = add(current_serving_peak, preparation_bytes)?;
        ensure!(
            preparation_peak <= self.effective_budget_bytes,
            "expert recovery preparation exceeds policy budget: current_slots={current_capacity} target_slots={target_capacity} table_overlap={}B needed={preparation_peak}B budget={}B",
            self.replacement_address_table_overlap_bytes,
            self.effective_budget_bytes
        );
        self.effective_budget_bytes
            .checked_sub(current_serving_peak)
            .context("admitted serving commitments exceed recovery budget")
    }
}

/// Explicit allowances beyond the package payload and ordinary operator
/// buffer requests. These values are supplied by a caller, not inferred from
/// physical memory. `None` means the overlap has not been bounded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrdinaryPathInputs {
    pub context_limit: usize,
    /// Largest requested prefill call; one means decode-only execution.
    pub max_prefill_rows: usize,
    /// Evaluate every pool size from ten through this ceiling, including 511.
    pub candidate_slot_ceiling: usize,
    /// Persistent operator buffers not represented by packed package payloads.
    pub uncounted_core_reserve_bytes: u64,
    /// Vec spare capacity and old/new backing storage during growth.
    pub vec_spare_growth_overlap_reserve_bytes: Option<u64>,
    /// Maximum retained host-owned prompt-prefix snapshots.
    pub prefix_retained_budget_bytes: u64,
    /// Candidate snapshot creation or restore while retained entries remain.
    pub snapshot_capture_restore_overlap_reserve_bytes: Option<u64>,
    /// Externally established driver, allocator and process overhead allowance.
    pub driver_allocator_allowance_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrdinaryWorkspaceStage {
    StartupDirect,
    StartupExpertPool,
    PrefillQueuedGraph,
    DecodeQueuedGraph,
}

/// One simultaneous stage envelope. Host values count known logical payloads;
/// Metal values are requested shared-buffer lengths, not device accounting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrdinaryWorkspaceBytes {
    pub stage: OrdinaryWorkspaceStage,
    pub capacity_slots: usize,
    /// Zero for load staging, one for decode, otherwise the prefill width.
    pub rows: usize,
    pub metal_requested_bytes: u64,
    pub host_logical_bytes: u64,
    pub combined_component_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrdinaryMemoryPhase {
    Startup,
    EngineConstruction,
    Serving,
}

/// Complete lifetime-phase peaks for explicit capacity planning, not a hard
/// RSS or device working-set bound. Each phase includes the full direct and
/// ceiling expert payloads plus core and process allowances. Workspace host
/// and Metal fields come from one labelled stage, not separate maxima.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OrdinaryPathEstimate {
    pub direct_wire_payload_bytes: u64,
    pub expert_stream_payload_at_ceiling_bytes: u64,
    /// Persistent chunk address and publication buffers. Bounded admission
    /// reserves the complete bounded table extent; suffix release retains the
    /// allocation independently of the current weight capacity.
    pub expert_storage_metadata_reserve_bytes: u64,
    pub fixed_session_logical_bytes: u64,
    pub growing_session_logical_bytes: u64,
    pub minimum_snapshot_logical_bytes: u64,
    /// Persistent prefill buffers, created after tensor loading and retained
    /// throughout serving, including one-token decode.
    pub prefill_workspace_requested_bytes: u64,
    /// Reader staging and bounded-worker storage retained by the Engine.
    pub expert_reader_workspace_across_candidates_bytes: u64,
    pub runtime_workspace_at_ceiling: OrdinaryWorkspaceBytes,
    pub runtime_workspace_across_candidates: OrdinaryWorkspaceBytes,
    pub startup_workspace_across_candidates: OrdinaryWorkspaceBytes,
    pub max_workspace_across_candidates: OrdinaryWorkspaceBytes,
    /// Independent maximum requested Metal component for comparison with a
    /// measured peak-minus-live counter. Do not add it to the workspace sum.
    pub runtime_metal_requested_upper_bytes: u64,
    pub uncounted_core_reserve_bytes: u64,
    pub vec_spare_growth_overlap_reserve_bytes: u64,
    pub prefix_retained_budget_bytes: u64,
    pub snapshot_capture_restore_overlap_reserve_bytes: u64,
    pub driver_allocator_allowance_bytes: u64,
    pub startup_phase_peak_bytes: u64,
    pub engine_construction_phase_peak_bytes: u64,
    pub serving_phase_peak_bytes: u64,
    pub limiting_phase: OrdinaryMemoryPhase,
    /// Largest complete phase peak, preserving the existing diagnostic field.
    pub accounted_component_sum_bytes: u64,
}

impl OrdinaryPathEstimate {
    fn phase_component_sums(&self) -> Result<[(OrdinaryMemoryPhase, u64); 3]> {
        let common = sum(&[
            self.direct_wire_payload_bytes,
            self.expert_stream_payload_at_ceiling_bytes,
            self.expert_storage_metadata_reserve_bytes,
            self.uncounted_core_reserve_bytes,
            self.driver_allocator_allowance_bytes,
        ])?;
        // Loading uses the complete final payload as a conservative bound
        // while direct or expert staging is live. The prefill workspace and
        // Session are constructed after these load stages have completed.
        let startup = add(
            common,
            self.startup_workspace_across_candidates
                .combined_component_bytes,
        )?;
        // Prefill operators share their Engine weights. Their workspaces are
        // created while the Engine-owned package reader remains alive.
        let construction = sum(&[
            common,
            self.prefill_workspace_requested_bytes,
            self.expert_reader_workspace_across_candidates_bytes,
        ])?;
        let serving = sum(&[
            common,
            self.prefill_workspace_requested_bytes,
            self.fixed_session_logical_bytes,
            self.growing_session_logical_bytes,
            self.vec_spare_growth_overlap_reserve_bytes,
            self.prefix_retained_budget_bytes,
            self.snapshot_capture_restore_overlap_reserve_bytes,
            self.runtime_workspace_across_candidates
                .combined_component_bytes,
        ])?;
        Ok([
            (OrdinaryMemoryPhase::Startup, startup),
            (OrdinaryMemoryPhase::EngineConstruction, construction),
            (OrdinaryMemoryPhase::Serving, serving),
        ])
    }

    fn refresh_phase_peaks(&mut self) -> Result<()> {
        let phases = self.phase_component_sums()?;
        self.startup_phase_peak_bytes = phases[0].1;
        self.engine_construction_phase_peak_bytes = phases[1].1;
        self.serving_phase_peak_bytes = phases[2].1;
        (self.limiting_phase, self.accounted_component_sum_bytes) = largest_phase(phases);
        Ok(())
    }
}

fn largest_phase(phases: [(OrdinaryMemoryPhase, u64); 3]) -> (OrdinaryMemoryPhase, u64) {
    phases
        .into_iter()
        .reduce(|largest, candidate| {
            if candidate.1 > largest.1 {
                candidate
            } else {
                largest
            }
        })
        .unwrap()
}

fn mul(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |result, &value| {
        result
            .checked_mul(value)
            .context("memory byte count overflow")
    })
}

fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .context("memory byte count overflow")
}

fn sum(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |total, &value| add(total, value))
}

fn plan_from_ordinary_estimate(
    estimate: &OrdinaryPathEstimate,
    packed_expert_bytes_per_slot: u64,
    max_slots: usize,
    process_budget_bytes: u64,
    user_memory_limit_bytes: u64,
) -> Result<CapacityPlan> {
    let phases = estimate.phase_component_sums()?;
    ensure!(
        phases.map(|(_, bytes)| bytes)
            == [
                estimate.startup_phase_peak_bytes,
                estimate.engine_construction_phase_peak_bytes,
                estimate.serving_phase_peak_bytes,
            ]
            && largest_phase(phases)
                == (
                    estimate.limiting_phase,
                    estimate.accounted_component_sum_bytes
                ),
        "ordinary phase peaks differ from component estimates"
    );
    // Every phase charges the same ceiling expert payload. Remove only that
    // common linear term; context and workspace need not coexist in the
    // limiting phase and cannot be subtracted as independent maxima.
    let mut nonexpert = 0;
    for (_, bytes) in phases {
        nonexpert = nonexpert.max(
            bytes
                .checked_sub(estimate.expert_stream_payload_at_ceiling_bytes)
                .context("expert payload exceeds ordinary phase peak")?,
        );
    }
    plan_capacity(CapacityInputs {
        process_budget_bytes,
        user_memory_limit_bytes: Some(user_memory_limit_bytes),
        fixed_nonexpert_bytes: nonexpert,
        context_reserve_bytes: 0,
        bounded_workspace_bytes: Some(0),
        packed_expert_bytes_per_slot,
        minimum_slots: 10,
        max_slots,
    })
}

fn finish_optional_prefill_upgrade(
    estimate: OrdinaryPathEstimate,
    plan: CapacityPlan,
    admitted_rows: usize,
    wide: Result<OrdinaryPathEstimate>,
    requested_rows: usize,
    packed_expert_bytes_per_slot: u64,
) -> (OrdinaryPathEstimate, CapacityPlan, usize) {
    let wide = match wide {
        Ok(wide) => wide,
        Err(error) => {
            eprintln!(
                "prefill admission: optional {requested_rows} estimate unavailable; retaining admitted {admitted_rows} plan: {error:#}"
            );
            return (estimate, plan, admitted_rows);
        }
    };
    let wider_plan = if plan.slots == 512 {
        if wide.accounted_component_sum_bytes > plan.effective_budget_bytes {
            return (estimate, plan, admitted_rows);
        }
        CapacityPlan {
            accounted_peak_bytes: wide.accounted_component_sum_bytes,
            remaining_budget_bytes: plan.effective_budget_bytes
                - wide.accounted_component_sum_bytes,
            ..plan
        }
    } else {
        let Ok(wider_plan) = plan_from_ordinary_estimate(
            &wide,
            packed_expert_bytes_per_slot,
            plan.slots,
            plan.effective_budget_bytes,
            plan.effective_budget_bytes,
        ) else {
            return (estimate, plan, admitted_rows);
        };
        // Wider prefill must not consume the expert floor retained by the
        // adaptive runtime. All context, cache and process reserves remain in
        // the wide estimate; only its expert payload is replanned.
        if wider_plan.slots < 64 {
            return (estimate, plan, admitted_rows);
        }
        wider_plan
    };
    (wide, wider_plan, requested_rows)
}

fn config_number(config: &Value, field: &str) -> Result<u64> {
    config[field]
        .as_u64()
        .with_context(|| format!("missing validated memory geometry {field}"))
}

fn direct_payload(package: &Package, config: &Value) -> Result<DirectPayloadBytes> {
    let mut result = DirectPayloadBytes {
        embedding_cpu_bytes: 0,
        matrix_count: 0,
        matrix_packed_bytes: 0,
        auxiliary_count: 0,
        auxiliary_bytes: 0,
        largest_direct_payload_bytes: 0,
    };
    let mut embeddings = 0;
    for header in package.direct_headers()? {
        let bytes = u64::try_from(header.bytes)?;
        result.largest_direct_payload_bytes = result.largest_direct_payload_bytes.max(bytes);
        if header.module_path == "embedding" {
            ensure!(
                header.format == WeightFormat::Q8_0
                    && header.shape
                        == [
                            usize::try_from(config_number(config, "vocab_size")?)?,
                            usize::try_from(config_number(config, "hidden_size")?)?,
                        ],
                "embedding package geometry differs from loader"
            );
            embeddings += 1;
            result.embedding_cpu_bytes = bytes;
        } else if header.shape.len() == 2 {
            result.matrix_count += 1;
            result.matrix_packed_bytes = add(result.matrix_packed_bytes, bytes)?;
        } else {
            result.auxiliary_count += 1;
            result.auxiliary_bytes = add(result.auxiliary_bytes, bytes)?;
        }
    }
    ensure!(embeddings == 1, "package requires one embedding parameter");
    Ok(result)
}

fn expert_payload(
    package: &Package,
    layers: usize,
    expected_experts: usize,
) -> Result<(Vec<ExpertLayerBytes>, u64, u64)> {
    let mut per_layer = Vec::with_capacity(layers);
    let mut one_slot = 0;
    let mut full_bundles = 0;
    for layer in 0..layers {
        let bundle = package.expert_layer(layer)?;
        ensure!(
            bundle.num_experts == expected_experts,
            "layer {layer} expert count differs from model contract"
        );
        let projections =
            Projection::ALL.map(|projection| u64::try_from(bundle.projection(projection).nbytes));
        let projections = projections
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let projection_bytes_per_slot: [u64; 3] =
            projections.try_into().expect("three projections");
        let packed_bytes_per_slot = projection_bytes_per_slot.into_iter().try_fold(0u64, add)?;
        ensure!(
            packed_bytes_per_slot == u64::try_from(bundle.row_bytes)?,
            "expert projections do not fill one packed slot"
        );
        let full_bundle_payload_bytes =
            mul(&[packed_bytes_per_slot, u64::try_from(bundle.num_experts)?])?;
        one_slot = add(one_slot, packed_bytes_per_slot)?;
        full_bundles = add(full_bundles, full_bundle_payload_bytes)?;
        per_layer.push(ExpertLayerBytes {
            layer,
            experts: bundle.num_experts,
            projection_bytes_per_slot,
            packed_bytes_per_slot,
            full_bundle_payload_bytes,
        });
    }
    Ok((per_layer, one_slot, full_bundles))
}

fn fixed_session(config: &Value, kinds: &[MixerKind]) -> Result<FixedSessionBytes> {
    let gdn = u64::try_from(kinds.iter().filter(|&&kind| kind == MixerKind::Gdn).count())?;
    let qsa = u64::try_from(kinds.iter().filter(|&&kind| kind == MixerKind::Qsa).count())?;
    let key_heads = config_number(config, "linear_num_key_heads")?;
    let value_heads = config_number(config, "linear_num_value_heads")?;
    let key_width = config_number(config, "linear_key_head_dim")?;
    let value_width = config_number(config, "linear_value_head_dim")?;
    let conv_taps = config_number(config, "linear_conv_kernel_dim")?;
    let conv_history = conv_taps.checked_sub(1).context("empty GDN convolution")?;
    let conv_width = add(
        mul(&[2, key_heads, key_width])?,
        mul(&[value_heads, value_width])?,
    )?;
    let kv_heads = config_number(config, "num_key_value_heads")?;
    let head_width = config_number(config, "head_dim")?;
    ensure!(
        mul(&[kv_heads, head_width, 2])? == u64::try_from(TOKEN_BYTES)?,
        "QSA package geometry differs from KVarN token wire"
    );
    let hc_count = config_number(config, "hc_count")?;
    let hidden = config_number(config, "hidden_size")?;
    let ple_taps = config_number(config, "ple_conv_kernel_size")?;
    let ple_history = ple_taps.checked_sub(1).context("empty PLE convolution")?;
    let ngram = config_number(config, "ngram_size")?;
    Ok(FixedSessionBytes {
        gdn_recurrent_f32: mul(&[gdn, value_heads, key_width, value_width, 4])?,
        gdn_convolution_bf16: mul(&[gdn, conv_history, conv_width, 2])?,
        qsa_sink_kv_bf16: mul(&[qsa, 2, u64::try_from(SINK_TOKENS)?, kv_heads, head_width, 2])?,
        qsa_tail_kv_bf16: mul(&[
            qsa,
            2,
            u64::try_from(TAIL_CAPACITY)?,
            kv_heads,
            head_width,
            2,
        ])?,
        ple_history_bf16: mul(&[ple_history, ngram, hc_count, hidden, 2])?,
    })
}

fn growing_session_payload(
    frontier: usize,
    max_context_tokens: usize,
    qsa_layers: usize,
) -> Result<GrowingSessionPayloadBytes> {
    ensure!(
        frontier <= max_context_tokens,
        "session frontier exceeds declared context limit"
    );
    let (_, body_end, _) = partition(frontier);
    let record_tokens = body_end.saturating_sub(SINK_TOKENS);
    ensure!(
        record_tokens.is_multiple_of(TILE_TOKENS),
        "KVarN partition produced a partial record tile"
    );
    let record_tiles = record_tokens / TILE_TOKENS;
    let complete_groups = frontier / COMPRESS_RATIO;
    let pending_rows = frontier % COMPRESS_RATIO;
    let layers = u64::try_from(qsa_layers)?;
    let kvarn_records = mul(&[
        layers,
        u64::try_from(record_tiles)?,
        u64::try_from(TILE_RECORD_BYTES)?,
    ])?;
    let qsa_index_compressed_bf16 = mul(&[
        layers,
        u64::try_from(complete_groups)?,
        u64::try_from(INDEX_WIDTH)?,
        2,
    ])?;
    let qsa_index_pending_bf16 = mul(&[
        layers,
        u64::try_from(pending_rows)?,
        u64::try_from(INDEX_WIDTH)?,
        2,
    ])?;
    let total_logical_bytes = add(
        add(kvarn_records, qsa_index_compressed_bf16)?,
        qsa_index_pending_bf16,
    )?;
    Ok(GrowingSessionPayloadBytes {
        frontier_tokens: frontier,
        kvarn_record_tiles_per_qsa_layer: record_tiles,
        qsa_index_complete_groups_per_layer: complete_groups,
        qsa_index_pending_rows_per_layer: pending_rows,
        kvarn_records,
        qsa_index_compressed_bf16,
        qsa_index_pending_bf16,
        total_logical_bytes,
    })
}

fn workspace(
    stage: OrdinaryWorkspaceStage,
    capacity_slots: usize,
    rows: usize,
    metal_requested_bytes: u64,
    host_logical_bytes: u64,
) -> Result<OrdinaryWorkspaceBytes> {
    Ok(OrdinaryWorkspaceBytes {
        stage,
        capacity_slots,
        rows,
        metal_requested_bytes,
        host_logical_bytes,
        combined_component_bytes: add(metal_requested_bytes, host_logical_bytes)?,
    })
}

fn larger(a: OrdinaryWorkspaceBytes, b: OrdinaryWorkspaceBytes) -> OrdinaryWorkspaceBytes {
    if b.combined_component_bytes > a.combined_component_bytes {
        b
    } else {
        a
    }
}

/// Short-lived requested bytes outside the persistent operator and prefill
/// workspaces. One ordered stream retains bound immutable parameters across
/// layers until a wait; bounded demand waits can only reduce this envelope.
fn runtime_workspaces(
    rows: usize,
    capacity: usize,
    largest_expert_row: u64,
) -> Result<Vec<OrdinaryWorkspaceBytes>> {
    let row_workspace = expert_reader_workspace_bytes(largest_expert_row, capacity)?;
    let qsa_layers = 12u64;
    let layers = 48u64;
    let decode_metal = mul(&[qsa_layers, 36 + 4])?; // pending-view counts and seal status
    // Two rows cover transient stochastic read bytes and decoded BF16 words.
    // Retain the conservative third-row allowance for diagnostic output.
    let decode_head_host = 5_120 + 3 * 248_320 * 2;
    let decode_demand_host = if capacity == 512 {
        0
    } else {
        sum(&[5_120, 20_480])?
    };
    let mut stages = vec![workspace(
        OrdinaryWorkspaceStage::DecodeQueuedGraph,
        capacity,
        1,
        decode_metal,
        add(row_workspace, decode_head_host.max(decode_demand_host))?,
    )?];
    if rows == 1 {
        return Ok(stages);
    }

    let r = u64::try_from(rows)?;
    let tiles = u64::try_from(rows.div_ceil(64))?;
    // Per QSA layer: projection positions, batched index contexts and
    // parameters, pending-view counts, tile arguments, and one seal flag.
    // Reconstruction adds one 12-byte shape per chunk and indexed tiles add
    // 12-byte geometry. Charge every tile even when a fallback uses less.
    let qsa_metal_per_layer = sum(&[mul(&[44, r])?, 64, mul(&[48, tiles])?])?;
    let qsa_metal = mul(&[qsa_layers, qsa_metal_per_layer])?;
    let bounded = capacity < 512;
    let source_groups = if bounded {
        512usize.div_ceil(capacity / 2)
    } else {
        1
    };
    let mlp_metal_per_layer = sum(&[
        2_048,                                      // immutable route slot map
        mul(&[16, u64::try_from(source_groups)?])?, // packed schedule dimensions
        if bounded { mul(&[80, r])? } else { 0 },   // sorted IDs and original pairs
    ])?;
    let metal = add(qsa_metal, mul(&[layers, mlp_metal_per_layer])?)?;

    // The initial embedding vector remains in scope through the chunk.
    // PLE stages BF16 words and upload bytes once; head readback holds both
    // byte and word vectors. A bounded expert read overlaps compact route
    // vectors. The persistent owner staging row overlaps every phase and is
    // charged outside the transient maximum. BTree route metadata is reserved
    // separately at 64 bytes per possible logical source.
    let embedding_host = mul(&[5_120, r])?;
    let ple_host = sum(&[embedding_host, mul(&[10_240, r])?, 10_240])?;
    // The two rows also cover transient stochastic read bytes and words.
    let head_host = add(embedding_host, 2 * 248_320 * 2)?;
    let bounded_host = if bounded {
        sum(&[embedding_host, mul(&[280, r])?, 64 * 512])?
    } else {
        0
    };
    stages.push(workspace(
        OrdinaryWorkspaceStage::PrefillQueuedGraph,
        capacity,
        rows,
        metal,
        add(row_workspace, ple_host.max(head_host).max(bounded_host))?,
    )?);
    Ok(stages)
}

fn expert_reader_workspace_bytes(row: u64, capacity: usize) -> Result<u64> {
    if capacity == 512 {
        return Ok(row);
    }
    // Owner and read-ahead rows, the requested persistent worker stacks, and bounded
    // channel/control/miss-list storage overlap every Engine execution phase.
    sum(&[
        mul(&[row, (crate::package::PREFILL_READ_ROWS + 1) as u64])?,
        mul(&[
            crate::package::PREFILL_READ_STACK_BYTES as u64,
            crate::package::PREFILL_READ_WORKERS as u64,
        ])?,
        crate::package::PREFILL_READ_CONTROL_BYTES as u64,
    ])
}

/// Select the largest affordable expert pool from explicit, conservative
/// byte estimates. No host observation or automatic serving policy is implied.
pub fn plan_capacity(input: CapacityInputs) -> Result<CapacityPlan> {
    ensure!(
        input.minimum_slots >= 10 && input.minimum_slots <= input.max_slots,
        "expert slot bounds must include the top-ten minimum"
    );
    ensure!(input.max_slots <= 512, "expert slot maximum exceeds 512");
    ensure!(
        input.packed_expert_bytes_per_slot > 0,
        "packed expert slot byte count must be positive"
    );
    let workspace = input
        .bounded_workspace_bytes
        .context("bounded workspace estimate is required for capacity planning")?;
    let effective_budget_bytes = input
        .user_memory_limit_bytes
        .map_or(input.process_budget_bytes, |limit| {
            input.process_budget_bytes.min(limit)
        });
    let fixed = add(input.fixed_nonexpert_bytes, input.context_reserve_bytes)?;
    let fixed = add(fixed, workspace)?;
    let minimum_experts = mul(&[
        input.packed_expert_bytes_per_slot,
        u64::try_from(input.minimum_slots)?,
    ])?;
    let minimum_total = add(fixed, minimum_experts)?;
    ensure!(
        minimum_total <= effective_budget_bytes,
        "memory budget cannot fit the top-ten expert minimum: need {minimum_total} bytes, have {effective_budget_bytes} bytes"
    );
    let affordable = (effective_budget_bytes - fixed) / input.packed_expert_bytes_per_slot;
    let slots = usize::try_from(affordable.min(u64::try_from(input.max_slots)?))?;
    let expert_bytes = mul(&[input.packed_expert_bytes_per_slot, u64::try_from(slots)?])?;
    let accounted_peak_bytes = add(fixed, expert_bytes)?;
    Ok(CapacityPlan {
        slots,
        effective_budget_bytes,
        accounted_peak_bytes,
        remaining_budget_bytes: effective_budget_bytes - accounted_peak_bytes,
    })
}

impl PackageFootprint {
    fn ordinary_vec_overlap_at_context(
        &self,
        context_limit: usize,
        full_resident: bool,
    ) -> Result<u64> {
        let growing = self.growing_session_payload_at_frontier(context_limit)?;
        // At this frontier, the logical record and complete-group counts are
        // also the declared capacity maxima. They cover all twelve live/new
        // backings, including spare capacity at every earlier frontier.
        let old_pair = sum(&[
            mul(&[
                u64::try_from(growing.kvarn_record_tiles_per_qsa_layer)?,
                u64::try_from(TILE_RECORD_BYTES)?,
            ])?,
            mul(&[
                u64::try_from(growing.qsa_index_complete_groups_per_layer)?,
                u64::try_from(INDEX_WIDTH * 2)?,
            ])?,
        ])?;
        let layers = u64::try_from(self.qsa_layers)?;
        let old_backings = if full_resident { layers } else { 1 };
        sum(&[
            mul(&[layers, QSA_DEVICE_INDEX_FIXED_CAPACITY_BYTES])?,
            mul(&[old_backings, old_pair])?,
        ])
    }

    /// Read the package manifest and safetensors headers without reading weight
    /// payloads or creating a Metal device.
    pub fn from_package(package: &Package) -> Result<Self> {
        let model = QwenConfig::from_package(package)?;
        let config = &package.architecture()["config"];
        let direct = direct_payload(package, config)?;
        let max_expert_slots = usize::try_from(config_number(config, "num_experts")?)?;
        let (expert_layers, packed_expert_bytes_per_uniform_slot, full_expert_bundle_payload_bytes) =
            expert_payload(package, model.layers.len(), max_expert_slots)?;
        let fixed_session = fixed_session(config, &model.layers)?;
        let qsa_layers = model
            .layers
            .iter()
            .filter(|&&kind| kind == MixerKind::Qsa)
            .count();
        Ok(Self {
            artifact_id: package.artifact_id.clone(),
            direct,
            expert_layers,
            packed_expert_bytes_per_uniform_slot,
            full_expert_bundle_payload_bytes,
            fixed_session,
            max_expert_slots,
            max_context_tokens: model.max_positions,
            qsa_layers,
        })
    }

    /// Logical occupied KVarN body records and QSA index words at `frontier`.
    /// This is not the allocated Vec capacity or a bound on copy/reallocation
    /// peaks while a session grows.
    pub fn growing_session_payload_at_frontier(
        &self,
        frontier: usize,
    ) -> Result<GrowingSessionPayloadBytes> {
        growing_session_payload(frontier, self.max_context_tokens, self.qsa_layers)
    }

    /// Packed expert stream payload for the same physical slot count in every
    /// layer. Lookup tables and per-layer scratch are not included.
    pub fn packed_expert_bytes_at_capacity(&self, capacity: usize) -> Result<u64> {
        ensure!(
            (10..=self.max_expert_slots).contains(&capacity),
            "expert capacity differs from engine range"
        );
        mul(&[
            self.packed_expert_bytes_per_uniform_slot,
            u64::try_from(capacity)?,
        ])
    }

    /// Requested address and publication bytes for the initial expert pool.
    /// Bounded tables retain the complete 511-slot extent through release and
    /// growth. Contiguous storage, including full residency, has no tables.
    pub fn expert_storage_metadata_bytes(&self, initial_capacity: usize) -> Result<u64> {
        ensure!(
            (10..=self.max_expert_slots.min(512)).contains(&initial_capacity),
            "expert metadata capacity differs from engine range"
        );
        if initial_capacity == 512 {
            return Ok(0);
        }
        mul(&[
            add(mul(&[511, 18, 8])?, 4)?,
            u64::try_from(self.expert_layers.len())?,
        ])
    }

    /// Plan only an explicitly budgeted ordinary CLI path. This accounts for
    /// validated Qwen operator requests, capacity-dependent old QSA backing
    /// overlap, PLE and pending-index overlap, and a caller-supplied absolute
    /// process allowance. It is a component estimate, not a process-memory
    /// guarantee.
    pub fn plan_explicit_ordinary_capacity(
        &self,
        context_limit: usize,
        process_budget_bytes: u64,
        user_memory_limit_bytes: u64,
        process_overhead_allowance_bytes: u64,
    ) -> Result<(OrdinaryPathEstimate, CapacityPlan)> {
        self.plan_explicit_ordinary_capacity_with_prefix_budget(
            context_limit,
            process_budget_bytes,
            user_memory_limit_bytes,
            process_overhead_allowance_bytes,
            0,
        )
    }

    /// Plan an explicitly retained in-memory prompt cache with no disk tier.
    /// The cache evicts a replaced entry before capture and limits the sum of
    /// retained entries plus a successful candidate to this one byte budget.
    /// The ordinary session component separately covers a restored live state.
    /// A declined capture can briefly allocate beyond its retained allowance
    /// before allocator-rounded capacity is checked. This is not an RSS bound.
    pub fn plan_explicit_ordinary_capacity_with_prefix_budget(
        &self,
        context_limit: usize,
        process_budget_bytes: u64,
        user_memory_limit_bytes: u64,
        process_overhead_allowance_bytes: u64,
        prefix_retained_budget_bytes: u64,
    ) -> Result<(OrdinaryPathEstimate, CapacityPlan)> {
        self.plan_ordinary_capacity_with_cache_budgets(
            context_limit,
            process_budget_bytes,
            user_memory_limit_bytes,
            process_overhead_allowance_bytes,
            prefix_retained_budget_bytes,
            0,
        )
    }

    /// Reserve one disk capture or decoded snapshot separately from retained
    /// memory entries. The ordinary component already includes the live Session.
    pub fn plan_ordinary_capacity_with_cache_budgets(
        &self,
        context_limit: usize,
        process_budget_bytes: u64,
        user_memory_limit_bytes: u64,
        process_overhead_allowance_bytes: u64,
        prefix_retained_budget_bytes: u64,
        disk_snapshot_overlap_bytes: u64,
    ) -> Result<(OrdinaryPathEstimate, CapacityPlan)> {
        ensure!(
            self.qsa_layers == 12 && self.expert_layers.len() == 48 && self.max_expert_slots == 512,
            "explicit ordinary memory policy requires validated Qwen layer and expert geometry"
        );
        ensure!(
            user_memory_limit_bytes > 0,
            "explicit memory limit must be positive"
        );
        let estimate_for = |candidate_slot_ceiling, full_resident| {
            // The full-resident branch evaluates contiguous storage only.
            // Its separate bounded fallback reserves the 511-slot tables.
            self.estimate_ordinary_path_with_metadata_capacity(
                OrdinaryPathInputs {
                    context_limit,
                    max_prefill_rows: context_limit.min(512),
                    candidate_slot_ceiling,
                    uncounted_core_reserve_bytes: ORDINARY_CORE_BUFFER_BYTES,
                    vec_spare_growth_overlap_reserve_bytes: Some(
                        self.ordinary_vec_overlap_at_context(context_limit, full_resident)?,
                    ),
                    prefix_retained_budget_bytes,
                    snapshot_capture_restore_overlap_reserve_bytes: Some(
                        disk_snapshot_overlap_bytes,
                    ),
                    driver_allocator_allowance_bytes: process_overhead_allowance_bytes,
                },
                crate::prefill_capacity::DEFAULT_ROWS,
                candidate_slot_ceiling,
            )
        };
        let full = estimate_for(512, true)?;
        if full.accounted_component_sum_bytes <= process_budget_bytes.min(user_memory_limit_bytes) {
            let plan = plan_from_ordinary_estimate(
                &full,
                self.packed_expert_bytes_per_uniform_slot,
                512,
                process_budget_bytes,
                user_memory_limit_bytes,
            )?;
            return Ok((full, plan));
        }
        let estimate = estimate_for(511, false)?;
        let plan = plan_from_ordinary_estimate(
            &estimate,
            self.packed_expert_bytes_per_uniform_slot,
            511,
            process_budget_bytes,
            user_memory_limit_bytes,
        )?;
        Ok((estimate, plan))
    }

    /// Account for ordinary load, prefill and one-token decode at explicit
    /// geometry and reserve inputs. Candidate pool sizes are all examined so
    /// the full-resident 512-slot kernel does not hide 511-slot workspace.
    /// The result is the largest complete lifetime-phase component sum, not
    /// a hard RSS bound or an automatic serving policy.
    pub fn estimate_ordinary_path(
        &self,
        input: OrdinaryPathInputs,
    ) -> Result<OrdinaryPathEstimate> {
        self.estimate_ordinary_path_with_prefill_capacity(
            input,
            crate::prefill_capacity::DEFAULT_ROWS,
        )
    }

    /// Prefer a wider workspace within the existing serving budget. Full
    /// residency is preserved; bounded expert slots share the budget with
    /// prefill while keeping context, cache and process allowances intact.
    pub fn upgrade_prefill_capacity(
        &self,
        context_limit: usize,
        estimate: OrdinaryPathEstimate,
        plan: CapacityPlan,
    ) -> Result<(OrdinaryPathEstimate, CapacityPlan, usize)> {
        self.upgrade_prefill_capacity_to(
            context_limit,
            estimate,
            plan,
            crate::prefill_capacity::AUTOMATIC_ROWS,
        )
    }

    /// Resolve an explicit workspace ceiling from an admitted 512-row plan.
    /// Each wider tier keeps the previous admitted tier on refusal, including
    /// full-residency and expert-floor checks. Automatic serving uses 2048.
    pub fn upgrade_prefill_capacity_to(
        &self,
        context_limit: usize,
        mut estimate: OrdinaryPathEstimate,
        mut plan: CapacityPlan,
        requested_rows: usize,
    ) -> Result<(OrdinaryPathEstimate, CapacityPlan, usize)> {
        crate::prefill_capacity::validate_capacity(requested_rows)?;
        ensure!(
            estimate.prefill_workspace_requested_bytes
                == u64::try_from(native_prefill_workspace_bytes(context_limit)?)?,
            "prefill upgrade requires an admitted 512-row estimate"
        );
        let mut admitted_rows = crate::prefill_capacity::DEFAULT_ROWS;
        if context_limit <= crate::prefill_capacity::DEFAULT_ROWS {
            return Ok((estimate, plan, admitted_rows));
        }
        for wider_rows in [crate::prefill_capacity::AUTOMATIC_ROWS, 4096, 8192]
            .into_iter()
            .filter(|&rows| rows <= crate::prefill_capacity::MAX_ROWS)
        {
            if wider_rows > requested_rows {
                break;
            }
            let wide = self.estimate_ordinary_path_with_metadata_capacity(
                OrdinaryPathInputs {
                    context_limit,
                    max_prefill_rows: context_limit.min(wider_rows),
                    candidate_slot_ceiling: plan.slots,
                    uncounted_core_reserve_bytes: estimate.uncounted_core_reserve_bytes,
                    vec_spare_growth_overlap_reserve_bytes: Some(
                        estimate.vec_spare_growth_overlap_reserve_bytes,
                    ),
                    prefix_retained_budget_bytes: estimate.prefix_retained_budget_bytes,
                    snapshot_capture_restore_overlap_reserve_bytes: Some(
                        estimate.snapshot_capture_restore_overlap_reserve_bytes,
                    ),
                    driver_allocator_allowance_bytes: estimate.driver_allocator_allowance_bytes,
                },
                wider_rows,
                plan.slots,
            );
            let upgraded = finish_optional_prefill_upgrade(
                estimate,
                plan,
                admitted_rows,
                wide,
                wider_rows,
                self.packed_expert_bytes_per_uniform_slot,
            );
            (estimate, plan, admitted_rows) = upgraded;
            if admitted_rows != wider_rows {
                break;
            }
        }
        Ok((estimate, plan, admitted_rows))
    }

    /// Explicit fixed-residency, cache-free component accounting.
    pub fn estimate_fixed_capacity_prefill(
        &self,
        context_limit: usize,
        expert_capacity: usize,
        prefill_capacity: usize,
        process_overhead_allowance_bytes: u64,
    ) -> Result<OrdinaryPathEstimate> {
        self.estimate_ordinary_path_with_metadata_capacity(
            OrdinaryPathInputs {
                context_limit,
                max_prefill_rows: context_limit.min(prefill_capacity),
                candidate_slot_ceiling: expert_capacity,
                uncounted_core_reserve_bytes: ORDINARY_CORE_BUFFER_BYTES,
                vec_spare_growth_overlap_reserve_bytes: Some(
                    self.ordinary_vec_overlap_at_context(context_limit, expert_capacity == 512)?,
                ),
                prefix_retained_budget_bytes: 0,
                snapshot_capture_restore_overlap_reserve_bytes: Some(0),
                driver_allocator_allowance_bytes: process_overhead_allowance_bytes,
            },
            prefill_capacity,
            expert_capacity,
        )
    }

    pub fn estimate_ordinary_path_with_prefill_capacity(
        &self,
        input: OrdinaryPathInputs,
        prefill_capacity: usize,
    ) -> Result<OrdinaryPathEstimate> {
        self.estimate_ordinary_path_with_metadata_capacity(
            input,
            prefill_capacity,
            input.candidate_slot_ceiling.min(511),
        )
    }

    fn estimate_ordinary_path_with_metadata_capacity(
        &self,
        input: OrdinaryPathInputs,
        prefill_capacity: usize,
        metadata_initial_capacity: usize,
    ) -> Result<OrdinaryPathEstimate> {
        crate::prefill_capacity::validate_capacity(prefill_capacity)?;
        ensure!(
            input.context_limit > 0 && input.context_limit <= self.max_context_tokens,
            "memory context limit differs from package range"
        );
        ensure!(
            (1..=prefill_capacity).contains(&input.max_prefill_rows)
                && input.max_prefill_rows <= input.context_limit,
            "memory prefill width differs from context or kernel range"
        );
        ensure!(
            (10..=self.max_expert_slots).contains(&input.candidate_slot_ceiling),
            "memory expert ceiling differs from package range"
        );
        let vec_reserve = input
            .vec_spare_growth_overlap_reserve_bytes
            .context("Vec spare and growth overlap reserve is required")?;
        let snapshot_overlap = input
            .snapshot_capture_restore_overlap_reserve_bytes
            .context("snapshot capture and restore overlap reserve is required")?;
        ensure!(
            self.qsa_layers > 0 && self.expert_layers.len() == 48,
            "memory estimate requires validated Qwen mixer geometry"
        );
        let fixed_session_logical_bytes = sum(&[
            self.fixed_session.gdn_recurrent_f32,
            self.fixed_session.gdn_convolution_bf16,
            self.fixed_session.qsa_sink_kv_bf16,
            self.fixed_session.qsa_tail_kv_bf16,
            self.fixed_session.ple_history_bf16,
        ])?;
        let growing = self.growing_session_payload_at_frontier(input.context_limit)?;
        let largest_expert_row = self
            .expert_layers
            .iter()
            .map(|layer| layer.packed_bytes_per_slot)
            .max()
            .context("memory estimate has no expert layers")?;
        let largest_expert_projection = self
            .expert_layers
            .iter()
            .flat_map(|layer| layer.projection_bytes_per_slot)
            .max()
            .context("memory estimate has no expert projections")?;
        // Storage is monotone within bounded capacities, but the reader worker
        // is absent at full residency. Check both sides of that discontinuity.
        let mut startup_workspace_across_candidates = None;
        let mut expert_reader_workspace_across_candidates_bytes = 0;
        for capacity in [
            input.candidate_slot_ceiling,
            input.candidate_slot_ceiling.min(511),
        ] {
            let row_workspace = expert_reader_workspace_bytes(largest_expert_row, capacity)?;
            expert_reader_workspace_across_candidates_bytes =
                expert_reader_workspace_across_candidates_bytes.max(row_workspace);
            let direct = workspace(
                OrdinaryWorkspaceStage::StartupDirect,
                capacity,
                0,
                0,
                add(row_workspace, self.direct.largest_direct_payload_bytes)?,
            )?;
            let expert = workspace(
                OrdinaryWorkspaceStage::StartupExpertPool,
                capacity,
                0,
                0,
                sum(&[
                    mul(&[largest_expert_projection, u64::try_from(capacity)?])?,
                    largest_expert_projection,
                    row_workspace,
                ])?,
            )?;
            let candidate = larger(direct, expert);
            startup_workspace_across_candidates = Some(
                startup_workspace_across_candidates
                    .map_or(candidate, |current| larger(current, candidate)),
            );
        }
        let startup_workspace_across_candidates = startup_workspace_across_candidates.unwrap();

        let mut runtime_workspace_across_candidates = None;
        let mut runtime_workspace_at_ceiling = None;
        let mut runtime_metal_requested_upper_bytes = 0;
        for capacity in 10..=input.candidate_slot_ceiling {
            let mut at_capacity = None;
            // Split-K scratch changes discretely with row count; scanning the
            // legal widths avoids assuming the largest chunk is the peak.
            for rows in 1..=input.max_prefill_rows {
                for stage in runtime_workspaces(rows, capacity, largest_expert_row)? {
                    // This independent comparison bound includes the supplied
                    // all-layer backing-growth reserve. The planner charges
                    // that reserve separately from this diagnostic field.
                    runtime_metal_requested_upper_bytes = runtime_metal_requested_upper_bytes
                        .max(sum(&[stage.metal_requested_bytes, vec_reserve])?);
                    at_capacity = Some(at_capacity.map_or(stage, |current| larger(current, stage)));
                }
            }
            let at_capacity = at_capacity.context("no ordinary execution stage")?;
            runtime_workspace_across_candidates = Some(
                runtime_workspace_across_candidates
                    .map_or(at_capacity, |current| larger(current, at_capacity)),
            );
            if capacity == input.candidate_slot_ceiling {
                runtime_workspace_at_ceiling = Some(at_capacity);
            }
        }
        let runtime_workspace_at_ceiling = runtime_workspace_at_ceiling.unwrap();
        let runtime_workspace_across_candidates = runtime_workspace_across_candidates.unwrap();
        let max_workspace_across_candidates = larger(
            runtime_workspace_across_candidates,
            startup_workspace_across_candidates,
        );
        let direct_wire_payload_bytes = sum(&[
            self.direct.embedding_cpu_bytes,
            self.direct.matrix_packed_bytes,
            self.direct.auxiliary_bytes,
        ])?;
        let expert_stream_payload_at_ceiling_bytes =
            self.packed_expert_bytes_at_capacity(input.candidate_slot_ceiling)?;
        let minimum_snapshot_logical_bytes = sum(&[
            fixed_session_logical_bytes,
            growing.total_logical_bytes,
            mul(&[u64::try_from(input.context_limit)?, 4])?,
            248_320 * 2,
        ])?;
        let prefill_workspace_requested_bytes = u64::try_from(
            if prefill_capacity == crate::prefill_capacity::DEFAULT_ROWS {
                native_prefill_workspace_bytes(input.context_limit)?
            } else {
                native_prefill_workspace_bytes_for_capacity(input.context_limit, prefill_capacity)?
            },
        )?;
        let mut estimate = OrdinaryPathEstimate {
            direct_wire_payload_bytes,
            expert_stream_payload_at_ceiling_bytes,
            expert_storage_metadata_reserve_bytes: self
                .expert_storage_metadata_bytes(metadata_initial_capacity)?,
            fixed_session_logical_bytes,
            growing_session_logical_bytes: growing.total_logical_bytes,
            minimum_snapshot_logical_bytes,
            prefill_workspace_requested_bytes,
            expert_reader_workspace_across_candidates_bytes,
            runtime_workspace_at_ceiling,
            runtime_workspace_across_candidates,
            startup_workspace_across_candidates,
            max_workspace_across_candidates,
            runtime_metal_requested_upper_bytes,
            uncounted_core_reserve_bytes: input.uncounted_core_reserve_bytes,
            vec_spare_growth_overlap_reserve_bytes: vec_reserve,
            prefix_retained_budget_bytes: input.prefix_retained_budget_bytes,
            snapshot_capture_restore_overlap_reserve_bytes: snapshot_overlap,
            driver_allocator_allowance_bytes: input.driver_allocator_allowance_bytes,
            startup_phase_peak_bytes: 0,
            engine_construction_phase_peak_bytes: 0,
            serving_phase_peak_bytes: 0,
            limiting_phase: OrdinaryMemoryPhase::Startup,
            accounted_component_sum_bytes: 0,
        };
        estimate.refresh_phase_peaks()?;
        Ok(estimate)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn bounded_reader_stack_and_control_are_lifetime_charged() {
        let row = 1_900_000;
        let control = crate::package::PREFILL_READ_WORKERS as u64
            * crate::package::PREFILL_READ_STACK_BYTES as u64
            + crate::package::PREFILL_READ_CONTROL_BYTES as u64;
        for capacity in [10, 178, 511] {
            assert_eq!(
                super::expert_reader_workspace_bytes(row, capacity).unwrap(),
                (crate::package::PREFILL_READ_ROWS as u64 + 1) * row + control
            );
        }
        assert_eq!(super::expert_reader_workspace_bytes(row, 512).unwrap(), row);
        assert!(super::expert_reader_workspace_bytes(u64::MAX, 511).is_err());
    }
    #[test]
    fn persistent_expert_rows_overlap_every_runtime_stage() {
        for capacity in [10, 178, 512] {
            for rows in [1, 512, 2048, 4096] {
                let without = super::runtime_workspaces(rows, capacity, 0).unwrap();
                let with = super::runtime_workspaces(rows, capacity, 1_900_000).unwrap();
                for (before, after) in without.into_iter().zip(with) {
                    assert_eq!(
                        after.host_logical_bytes - before.host_logical_bytes,
                        if capacity == 512 {
                            1_900_000
                        } else {
                            (crate::package::PREFILL_READ_ROWS as u64 + 1) * 1_900_000
                        }
                    );
                    assert_eq!(after.metal_requested_bytes, before.metal_requested_bytes);
                }
            }
        }
    }
    use super::*;
    use serde_json::json;

    #[test]
    fn combined_ane_payload_is_reserved_once_outside_workspace_phases() {
        let mib = 1024 * 1024;
        for (physical, staging, attended, total) in [(4096, 98, 96, 390), (8192, 166, 192, 690)] {
            let memory = ane_prefill_memory(physical).unwrap();
            assert_eq!(memory.host_staging_bytes, staging * mib);
            assert_eq!(memory.metal_staging_bytes, staging * mib + 4);
            assert_eq!(memory.iosurface_payload_bytes, staging * mib);
            assert_eq!(memory.qsa_attended_bytes, attended * mib);
            assert_eq!(
                ane_prefill_allowance_bytes(physical).unwrap(),
                total * mib + 4
            );
        }
        assert_eq!(
            ane_prefill_allowance_bytes(8192).unwrap() - ane_prefill_allowance_bytes(4096).unwrap(),
            300 * mib
        );
        for rows in [0, 2048, 4097, 8193, usize::MAX] {
            assert!(ane_prefill_memory(rows).is_err());
            assert!(ane_prefill_allowance_bytes(rows).is_err());
        }
        let report = ordinary_report();
        for rows in [2048, 4096, 8192] {
            let allowance = ane_prefill_allowance_bytes(
                crate::prefill_capacity::ane_physical_rows(rows).unwrap(),
            )
            .unwrap();
            let input = ordinary_input(131072, rows, 511);
            let ordinary = report
                .estimate_ordinary_path_with_prefill_capacity(input, rows)
                .unwrap();
            let reserved = report
                .estimate_ordinary_path_with_prefill_capacity(
                    OrdinaryPathInputs {
                        driver_allocator_allowance_bytes: input.driver_allocator_allowance_bytes
                            + allowance,
                        ..input
                    },
                    rows,
                )
                .unwrap();
            for (before, after) in ordinary
                .phase_component_sums()
                .unwrap()
                .into_iter()
                .zip(reserved.phase_component_sums().unwrap())
            {
                assert_eq!(before.0, after.0);
                assert_eq!(after.1 - before.1, allowance);
            }
        }
    }

    #[test]
    fn explicit_8192_workspace_retains_exact_linear_inventory_and_admission_cost() {
        for context in [1, 8192, 131072, 262144] {
            let ordinary = native_prefill_workspace_bytes_for_capacity(context, 4096).unwrap();
            let wider = native_prefill_workspace_bytes_for_capacity(context, 8192).unwrap();
            assert_eq!(wider - ordinary, 1_996_144_640);
            assert!(wider < u32::MAX as usize);
        }
        assert_eq!(
            native_prefill_workspace_bytes_for_capacity(131072, 8192).unwrap(),
            4_275_747_470
        );
        let report = ordinary_report();
        let estimate = |rows| {
            report
                .estimate_ordinary_path_with_prefill_capacity(
                    OrdinaryPathInputs {
                        prefix_retained_budget_bytes: 0,
                        snapshot_capture_restore_overlap_reserve_bytes: Some(0),
                        ..ordinary_input(131072, rows, 512)
                    },
                    rows,
                )
                .unwrap()
        };
        let base = estimate(4096);
        let wider = estimate(8192);
        assert_eq!(
            wider.prefill_workspace_requested_bytes - base.prefill_workspace_requested_bytes,
            1_996_144_640
        );
        assert_eq!(
            wider.engine_construction_phase_peak_bytes - base.engine_construction_phase_peak_bytes,
            1_996_144_640
        );
        assert_eq!(
            wider
                .runtime_workspace_across_candidates
                .combined_component_bytes
                - base
                    .runtime_workspace_across_candidates
                    .combined_component_bytes,
            80_842_752
        );
        assert_eq!(
            wider.serving_phase_peak_bytes - base.serving_phase_peak_bytes,
            2_076_987_392
        );
        let full_stage = |rows| {
            runtime_workspaces(rows, 512, 0)
                .unwrap()
                .into_iter()
                .find(|stage| stage.stage == OrdinaryWorkspaceStage::PrefillQueuedGraph)
                .unwrap()
        };
        assert_eq!(
            full_stage(8192).combined_component_bytes - full_stage(4096).combined_component_bytes,
            65_114_112
        );
        assert_eq!(
            wider.startup_phase_peak_bytes,
            base.startup_phase_peak_bytes
        );
    }

    #[test]
    fn generation_sampling_inventory_matches_sampler_workspace() {
        assert_eq!(
            crate::sampling::sampling_host_workspace_bytes(GENERATION_VOCABULARY as usize).unwrap(),
            SAMPLING_HOST_WORKSPACE_BYTES
        );
        assert_eq!(DEVICE_GREEDY_REQUESTED_BYTES, 535_450);
        assert_eq!(DEVICE_GREEDY_HOST_BYTES, 31_040);
    }

    #[test]
    fn admitted_prefill_capacities_have_distinct_workspace_bounds() {
        assert_eq!(
            native_prefill_workspace_bytes_for_capacity(32768, 512).unwrap(),
            524_587_662
        );
        assert_eq!(
            native_prefill_workspace_bytes_for_capacity(32768, 2048).unwrap(),
            1_275_239_054
        );
        assert_eq!(
            native_prefill_workspace_bytes_for_capacity(32768, 4096).unwrap(),
            2_273_311_374
        );
        assert!(native_prefill_workspace_bytes_for_capacity(32768, 513).is_err());
    }

    #[test]
    fn explicit_4096_workspace_charges_construction_and_complete_serving_peak() {
        let report = ordinary_report();
        let estimate = |rows| {
            report
                .estimate_ordinary_path_with_prefill_capacity(
                    OrdinaryPathInputs {
                        prefix_retained_budget_bytes: 1 << 30,
                        snapshot_capture_restore_overlap_reserve_bytes: Some(512 << 20),
                        ..ordinary_input(131072, rows, 178)
                    },
                    rows,
                )
                .unwrap()
        };
        let automatic = estimate(2048);
        let explicit = estimate(4096);
        let persistent_delta = 998_072_320;
        let transient_delta = 40_421_376;
        assert_eq!(
            explicit.prefill_workspace_requested_bytes
                - automatic.prefill_workspace_requested_bytes,
            persistent_delta
        );
        assert_eq!(
            explicit.startup_phase_peak_bytes,
            automatic.startup_phase_peak_bytes
        );
        assert_eq!(
            explicit.engine_construction_phase_peak_bytes
                - automatic.engine_construction_phase_peak_bytes,
            persistent_delta
        );
        assert_eq!(
            explicit
                .runtime_workspace_across_candidates
                .combined_component_bytes
                - automatic
                    .runtime_workspace_across_candidates
                    .combined_component_bytes,
            transient_delta
        );
        assert_eq!(
            explicit.serving_phase_peak_bytes - automatic.serving_phase_peak_bytes,
            persistent_delta + transient_delta
        );
        assert_eq!(
            explicit.prefix_retained_budget_bytes,
            automatic.prefix_retained_budget_bytes
        );
        assert_eq!(
            explicit.snapshot_capture_restore_overlap_reserve_bytes,
            automatic.snapshot_capture_restore_overlap_reserve_bytes
        );
    }
    #[test]
    fn fixed_state_math_matches_declared_loader_geometry() {
        let config = json!({
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "num_key_value_heads": 2,
            "head_dim": 256, "hc_count": 4, "hidden_size": 2560,
            "ple_conv_kernel_size": 4, "ngram_size": 3,
        });
        let kinds = (0..48)
            .map(|index| {
                if index % 4 == 3 {
                    MixerKind::Qsa
                } else {
                    MixerKind::Gdn
                }
            })
            .collect::<Vec<_>>();
        let bytes = fixed_session(&config, &kinds).unwrap();
        assert_eq!(bytes.gdn_recurrent_f32, 113_246_208);
        assert_eq!(bytes.gdn_convolution_bf16, 2_211_840);
        assert_eq!(bytes.qsa_sink_kv_bf16, 3_145_728);
        assert_eq!(bytes.qsa_tail_kv_bf16, 204_472_320);
        assert_eq!(bytes.ple_history_bf16, 184_320);

        let mut asymmetric = config.clone();
        asymmetric["linear_key_head_dim"] = json!(64);
        let bytes = fixed_session(&asymmetric, &kinds).unwrap();
        assert_eq!(bytes.gdn_recurrent_f32, 56_623_104);
        assert_eq!(bytes.gdn_convolution_bf16, 1_769_472);
    }

    #[test]
    fn fixed_geometry_rejects_mismatch_and_overflow() {
        let mut config = json!({
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "num_key_value_heads": 3,
            "head_dim": 256, "hc_count": 4, "hidden_size": 2560,
            "ple_conv_kernel_size": 4, "ngram_size": 3,
        });
        assert!(fixed_session(&config, &[MixerKind::Qsa]).is_err());
        config["num_key_value_heads"] = json!(2);
        config["hidden_size"] = json!(u64::MAX);
        assert!(fixed_session(&config, &[MixerKind::Gdn]).is_err());
    }

    #[test]
    fn capacity_rejects_underfilled_and_overfull_pools() {
        let report = PackageFootprint {
            artifact_id: String::new(),
            direct: DirectPayloadBytes {
                embedding_cpu_bytes: 0,
                matrix_count: 0,
                matrix_packed_bytes: 0,
                auxiliary_count: 0,
                auxiliary_bytes: 0,
                largest_direct_payload_bytes: 0,
            },
            expert_layers: Vec::new(),
            packed_expert_bytes_per_uniform_slot: 76_103_680,
            full_expert_bundle_payload_bytes: 0,
            fixed_session: FixedSessionBytes {
                gdn_recurrent_f32: 0,
                gdn_convolution_bf16: 0,
                qsa_sink_kv_bf16: 0,
                qsa_tail_kv_bf16: 0,
                ple_history_bf16: 0,
            },
            max_expert_slots: 512,
            max_context_tokens: 262_144,
            qsa_layers: 12,
        };
        assert_eq!(
            report.packed_expert_bytes_at_capacity(12).unwrap(),
            913_244_160
        );
        assert!(report.packed_expert_bytes_at_capacity(9).is_err());
        assert!(report.packed_expert_bytes_at_capacity(513).is_err());
    }

    #[test]
    fn growing_payload_crosses_sink_and_first_seal_exactly() {
        let empty = growing_session_payload(0, 262_144, 12).unwrap();
        assert_eq!(empty.total_logical_bytes, 0);
        for frontier in [127, 128, 129, 8_447] {
            let bytes = growing_session_payload(frontier, 262_144, 12).unwrap();
            assert_eq!(bytes.kvarn_record_tiles_per_qsa_layer, 0);
            assert_eq!(bytes.kvarn_records, 0);
            assert_eq!(
                bytes.qsa_index_complete_groups_per_layer * COMPRESS_RATIO
                    + bytes.qsa_index_pending_rows_per_layer,
                frontier
            );
        }
        let before = growing_session_payload(8_447, 262_144, 12).unwrap();
        assert_eq!(before.qsa_index_pending_rows_per_layer, 3);
        let sealed = growing_session_payload(8_448, 262_144, 12).unwrap();
        assert_eq!(sealed.kvarn_record_tiles_per_qsa_layer, 1);
        assert_eq!(sealed.kvarn_records, 12 * TILE_RECORD_BYTES as u64);
        assert_eq!(sealed.qsa_index_pending_rows_per_layer, 0);
        assert_eq!(sealed.qsa_index_compressed_bf16, 12 * 2_112 * 128 * 2);
    }

    #[test]
    fn growing_payload_max_context_is_checked_and_remains_logical() {
        let max = growing_session_payload(262_144, 262_144, 12).unwrap();
        assert_eq!(max.kvarn_record_tiles_per_qsa_layer, 1_983);
        assert_eq!(max.qsa_index_complete_groups_per_layer, 65_536);
        assert_eq!(max.qsa_index_pending_rows_per_layer, 0);
        assert_eq!(max.kvarn_records, 12 * 1_983 * TILE_RECORD_BYTES as u64);
        assert_eq!(max.qsa_index_compressed_bf16, 12 * 65_536 * 128 * 2);
        assert_eq!(
            max.total_logical_bytes,
            max.kvarn_records + max.qsa_index_compressed_bf16
        );
        assert!(growing_session_payload(262_145, 262_144, 12).is_err());
        assert!(growing_session_payload(1, 0, 12).is_err());
    }

    fn capacity_input() -> CapacityInputs {
        CapacityInputs {
            process_budget_bytes: 1_000,
            user_memory_limit_bytes: None,
            fixed_nonexpert_bytes: 100,
            context_reserve_bytes: 200,
            bounded_workspace_bytes: Some(100),
            packed_expert_bytes_per_slot: 50,
            minimum_slots: 10,
            max_slots: 512,
        }
    }

    #[test]
    fn planner_respects_exact_minimum_and_explicit_byte_ceiling() {
        let input = capacity_input();
        let plan = plan_capacity(input).unwrap();
        assert_eq!(plan.slots, 12);
        assert_eq!(plan.accounted_peak_bytes, 1_000);
        assert_eq!(plan.remaining_budget_bytes, 0);

        let capped = plan_capacity(CapacityInputs {
            user_memory_limit_bytes: Some(900),
            ..input
        })
        .unwrap();
        assert_eq!(capped.slots, 10);
        assert_eq!(capped.effective_budget_bytes, 900);
        assert_eq!(capped.accounted_peak_bytes, 900);
        assert_eq!(capped.remaining_budget_bytes, 0);
        let capped_with_remainder = plan_capacity(CapacityInputs {
            user_memory_limit_bytes: Some(901),
            ..input
        })
        .unwrap();
        assert_eq!(capped_with_remainder.slots, 10);
        assert_eq!(capped_with_remainder.remaining_budget_bytes, 1);
        assert!(
            plan_capacity(CapacityInputs {
                user_memory_limit_bytes: Some(899),
                ..input
            })
            .is_err()
        );
        assert!(
            plan_capacity(CapacityInputs {
                process_budget_bytes: 899,
                ..input
            })
            .is_err()
        );
    }

    #[test]
    fn planner_keeps_full_residency_and_rejects_unknown_or_overflowing_inputs() {
        let input = capacity_input();
        let full = plan_capacity(CapacityInputs {
            process_budget_bytes: 26_000,
            ..input
        })
        .unwrap();
        assert_eq!(full.slots, 512);
        assert_eq!(full.accounted_peak_bytes, 26_000);
        let package_capped = plan_capacity(CapacityInputs {
            process_budget_bytes: 26_000,
            max_slots: 511,
            ..input
        })
        .unwrap();
        assert_eq!(package_capped.slots, 511);

        assert!(
            plan_capacity(CapacityInputs {
                bounded_workspace_bytes: None,
                ..input
            })
            .is_err()
        );
        assert!(
            plan_capacity(CapacityInputs {
                packed_expert_bytes_per_slot: 0,
                ..input
            })
            .is_err()
        );
        assert!(
            plan_capacity(CapacityInputs {
                minimum_slots: 9,
                ..input
            })
            .is_err()
        );
        assert!(
            plan_capacity(CapacityInputs {
                max_slots: 513,
                ..input
            })
            .is_err()
        );
        assert!(
            plan_capacity(CapacityInputs {
                fixed_nonexpert_bytes: u64::MAX,
                ..input
            })
            .is_err()
        );
        assert!(
            plan_capacity(CapacityInputs {
                packed_expert_bytes_per_slot: u64::MAX,
                ..input
            })
            .is_err()
        );
    }

    fn ordinary_report() -> PackageFootprint {
        let projection = [600_000, 600_000, 356_480];
        PackageFootprint {
            artifact_id: "fixture".into(),
            direct: DirectPayloadBytes {
                embedding_cpu_bytes: 675_430_400,
                matrix_count: 773,
                matrix_packed_bytes: 3_705_334_720,
                auxiliary_count: 1,
                auxiliary_bytes: 100,
                largest_direct_payload_bytes: 520_000_000,
            },
            expert_layers: (0..48)
                .map(|layer| ExpertLayerBytes {
                    layer,
                    experts: 512,
                    projection_bytes_per_slot: projection,
                    packed_bytes_per_slot: projection.into_iter().sum(),
                    full_bundle_payload_bytes: 512 * projection.into_iter().sum::<u64>(),
                })
                .collect(),
            packed_expert_bytes_per_uniform_slot: 48 * projection.into_iter().sum::<u64>(),
            full_expert_bundle_payload_bytes: 512 * 48 * projection.into_iter().sum::<u64>(),
            fixed_session: FixedSessionBytes {
                gdn_recurrent_f32: 113_246_208,
                gdn_convolution_bf16: 2_211_840,
                qsa_sink_kv_bf16: 3_145_728,
                qsa_tail_kv_bf16: 204_472_320,
                ple_history_bf16: 184_320,
            },
            max_expert_slots: 512,
            max_context_tokens: 262_144,
            qsa_layers: 12,
        }
    }

    fn ordinary_input(
        context_limit: usize,
        max_prefill_rows: usize,
        ceiling: usize,
    ) -> OrdinaryPathInputs {
        OrdinaryPathInputs {
            context_limit,
            max_prefill_rows,
            candidate_slot_ceiling: ceiling,
            uncounted_core_reserve_bytes: 1_000_000,
            vec_spare_growth_overlap_reserve_bytes: Some(2_000_000),
            prefix_retained_budget_bytes: 0,
            snapshot_capture_restore_overlap_reserve_bytes: Some(0),
            driver_allocator_allowance_bytes: 3_000_000,
        }
    }

    fn legacy_component_sum(estimate: OrdinaryPathEstimate) -> u64 {
        sum(&[
            estimate.direct_wire_payload_bytes,
            estimate.expert_stream_payload_at_ceiling_bytes,
            estimate.fixed_session_logical_bytes,
            estimate.growing_session_logical_bytes,
            estimate.prefill_workspace_requested_bytes,
            estimate
                .max_workspace_across_candidates
                .combined_component_bytes,
            estimate.uncounted_core_reserve_bytes,
            estimate.vec_spare_growth_overlap_reserve_bytes,
            estimate.prefix_retained_budget_bytes,
            estimate.snapshot_capture_restore_overlap_reserve_bytes,
            estimate.driver_allocator_allowance_bytes,
        ])
        .unwrap()
    }

    #[test]
    fn chunk_metadata_reserves_the_same_bounded_extent_at_every_capacity() {
        let report = ordinary_report();
        let expected = 3_532_224;
        assert_eq!(report.expert_storage_metadata_bytes(178).unwrap(), expected);
        assert_eq!(report.expert_storage_metadata_bytes(512).unwrap(), 0);
        assert!(report.expert_storage_metadata_bytes(9).is_err());
        assert!(report.expert_storage_metadata_bytes(513).is_err());

        let all_candidates = report
            .estimate_ordinary_path(ordinary_input(1024, 512, 512))
            .unwrap();
        let bounded_reserve = 3_532_224;
        assert_eq!(
            all_candidates.expert_storage_metadata_reserve_bytes,
            bounded_reserve
        );
        let mut without_metadata = all_candidates;
        without_metadata.expert_storage_metadata_reserve_bytes = 0;
        without_metadata.refresh_phase_peaks().unwrap();
        for ((phase, bytes), (baseline_phase, baseline)) in all_candidates
            .phase_component_sums()
            .unwrap()
            .into_iter()
            .zip(without_metadata.phase_component_sums().unwrap())
        {
            assert_eq!(phase, baseline_phase);
            assert_eq!(bytes, baseline + bounded_reserve);
        }

        for rows in [512, 2048] {
            let bounded = report
                .estimate_fixed_capacity_prefill(32768, 178, rows, 2 << 30)
                .unwrap();
            assert_eq!(bounded.expert_storage_metadata_reserve_bytes, expected);
            let full = report
                .estimate_fixed_capacity_prefill(32768, 512, rows, 2 << 30)
                .unwrap();
            assert_eq!(full.expert_storage_metadata_reserve_bytes, 0);
        }
    }

    #[test]
    fn chunk_metadata_preserves_exact_full_fit_and_bounds_the_bounded_fallback() {
        let report = ordinary_report();
        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let metadata = report.expert_storage_metadata_bytes(511).unwrap();
        for context in [1, 131072] {
            let (full, full_plan) = report
                .plan_explicit_ordinary_capacity(context, u64::MAX, u64::MAX, 2 << 30)
                .unwrap();
            assert_eq!(full_plan.slots, 512);
            assert_eq!(full.expert_storage_metadata_reserve_bytes, 0);
            let full_budget = full_plan.accounted_peak_bytes;
            let (exact, exact_plan) = report
                .plan_explicit_ordinary_capacity(context, full_budget, full_budget, 2 << 30)
                .unwrap();
            assert_eq!(exact, full);
            assert_eq!(exact_plan.slots, 512);
            assert_eq!(exact_plan.remaining_budget_bytes, 0);

            let (bounded, bounded_plan) = report
                .plan_explicit_ordinary_capacity(context, u64::MAX, full_budget - 1, 2 << 30)
                .unwrap();
            assert_eq!(bounded_plan.slots, 511);
            assert_eq!(bounded.expert_storage_metadata_reserve_bytes, metadata);
            let nonexpert = bounded.accounted_component_sum_bytes
                - bounded.expert_stream_payload_at_ceiling_bytes;
            for slots in [10, 178, 511] {
                let budget = nonexpert + slots * per_slot;
                let (selected, selected_plan) = report
                    .plan_explicit_ordinary_capacity(context, u64::MAX, budget, 2 << 30)
                    .unwrap();
                assert_eq!(selected_plan.slots, slots as usize);
                assert_eq!(selected_plan.remaining_budget_bytes, 0);
                assert_eq!(selected.expert_storage_metadata_reserve_bytes, metadata);
                assert!(
                    selected.expert_storage_metadata_reserve_bytes
                        >= report
                            .expert_storage_metadata_bytes(selected_plan.slots)
                            .unwrap()
                );
                let expert_surplus =
                    selected.expert_stream_payload_at_ceiling_bytes - slots * per_slot;
                assert_eq!(
                    selected.accounted_component_sum_bytes - expert_surplus,
                    selected_plan.accounted_peak_bytes
                );
                if slots > 10 {
                    let (_, below) = report
                        .plan_explicit_ordinary_capacity(context, u64::MAX, budget - 1, 2 << 30)
                        .unwrap();
                    assert_eq!(below.slots, slots as usize - 1);
                }
            }
            let minimum_budget = nonexpert + 10 * per_slot;
            assert!(
                report
                    .plan_explicit_ordinary_capacity(
                        context,
                        u64::MAX,
                        minimum_budget - 1,
                        2 << 30,
                    )
                    .is_err()
            );
            if metadata > 0 {
                assert!(
                    report
                        .plan_explicit_ordinary_capacity(
                            context,
                            u64::MAX,
                            minimum_budget - metadata,
                            2 << 30,
                        )
                        .is_err()
                );
            }
        }

        // Contiguous full residency does not allocate bounded tables, even
        // when their addition would overflow the component arithmetic.
        let (without_allowance, _) = report
            .plan_explicit_ordinary_capacity(131072, u64::MAX, u64::MAX, 0)
            .unwrap();
        let allowance = u64::MAX - without_allowance.accounted_component_sum_bytes;
        let (exact_max, exact_max_plan) = report
            .plan_explicit_ordinary_capacity(131072, u64::MAX, u64::MAX, allowance)
            .unwrap();
        assert_eq!(exact_max_plan.slots, 512);
        assert_eq!(exact_max_plan.accounted_peak_bytes, u64::MAX);
        assert_eq!(exact_max.expert_storage_metadata_reserve_bytes, 0);
    }

    #[test]
    fn workspace_phase_keeps_complete_reserves_and_quotes_both_returns() {
        let footprint = ordinary_report();
        let old = native_prefill_workspace_bytes_for_capacity(131072, 2048).unwrap() as u64;
        let new = native_prefill_workspace_bytes_for_capacity(131072, 4096).unwrap() as u64;
        let per_slot = footprint.packed_expert_bytes_per_uniform_slot;
        let overlap = 3_532_032;
        let other = 8u64 << 30;
        let process_allowance =
            PROCESS_OVERHEAD_ALLOWANCE_BYTES + ane_prefill_allowance_bytes(4096).unwrap();
        let root = ExpertCapacityControlBudget {
            initial_capacity: 96,
            maximum_capacity: 96,
            startup_budget_bytes: other + old + 96 * per_slot,
            effective_budget_bytes: other + old + 96 * per_slot + overlap,
            initial_serving_peak_bytes: other + old + 96 * per_slot,
            packed_expert_bytes_per_slot: per_slot,
            replacement_address_table_overlap_bytes: overlap,
            process_overhead_allowance_bytes: process_allowance,
            operating_growth: true,
        };
        let wide = root
            .test_with_prefill_width(&footprint, 131072, 2048, 4096)
            .unwrap();
        let unchanged = root;
        root.validate_prefill_phase(&footprint, 131072, 2048, 4096)
            .unwrap();
        assert_eq!(root, unchanged);
        assert_eq!(wide.maximum_capacity(), 82);
        assert_eq!(
            wide.initial_serving_peak_bytes(),
            other + new + 40_421_376 + 82 * per_slot
        );
        let scratch_budget = root
            .test_workspace_budget_after_release(131072, 2048, 82)
            .unwrap();
        assert!(new <= scratch_budget && new + old > scratch_budget);
        assert!(wide.test_phase_expert_preparation(80, 82).unwrap() >= 2 * per_slot + overlap);
        assert!(root.test_phase_expert_preparation(82, 96).unwrap() >= 14 * per_slot + overlap);
        assert_eq!(root.operating_budget_bytes(), wide.operating_budget_bytes());
        assert_eq!(root.startup_budget_bytes(), wide.startup_budget_bytes());
        assert_eq!(
            root.process_overhead_allowance_bytes,
            wide.process_overhead_allowance_bytes
        );
        assert_eq!(wide.process_overhead_allowance_bytes, process_allowance);
        assert!(
            root.test_with_prefill_width(&footprint, 131072, 2048, 513)
                .is_err()
        );
        let tight = ExpertCapacityControlBudget {
            initial_capacity: 64,
            maximum_capacity: 64,
            initial_serving_peak_bytes: other + old + 64 * per_slot,
            effective_budget_bytes: other + old + 64 * per_slot + overlap,
            ..root
        };
        assert!(
            tight
                .test_with_prefill_width(&footprint, 131072, 2048, 4096)
                .is_err()
        );
        assert!(
            tight
                .validate_prefill_phase(&footprint, 131072, 2048, 4096)
                .is_err()
        );
    }

    fn recovery_budget() -> ExpertCapacityControlBudget {
        ExpertCapacityControlBudget {
            initial_capacity: 178,
            maximum_capacity: 178,
            startup_budget_bytes: 18_450,
            effective_budget_bytes: 18_450,
            initial_serving_peak_bytes: 23_400,
            packed_expert_bytes_per_slot: 100,
            replacement_address_table_overlap_bytes: 50,
            process_overhead_allowance_bytes: 0,
            operating_growth: false,
        }
    }

    #[test]
    fn recovery_budget_funds_partial_and_original_preparation_exactly() {
        let budget = recovery_budget();
        assert_eq!(budget.initial_capacity(), 178);
        assert_eq!(budget.maximum_capacity(), 178);
        assert_eq!(budget.startup_budget_bytes(), 18_450);
        assert_eq!(budget.operating_budget_bytes(), 18_450);
        assert_eq!(budget.preparation_budget_bytes(64, 128).unwrap(), 6_450);
        let mut short = budget;
        short.effective_budget_bytes -= 1;
        assert!(short.preparation_budget_bytes(64, 128).is_err());
        assert!(budget.preparation_budget_bytes(64, 178).is_err());

        let mut original = budget;
        original.effective_budget_bytes =
            original.initial_serving_peak_bytes + original.replacement_address_table_overlap_bytes;
        assert_eq!(original.preparation_budget_bytes(64, 178).unwrap(), 11_450);
        original.effective_budget_bytes -= 1;
        assert!(original.preparation_budget_bytes(64, 178).is_err());
        assert_eq!(budget.preparation_budget_bytes(64, 64).unwrap(), 0);
        assert_eq!(budget.preparation_budget_bytes(178, 178).unwrap(), 0);
    }

    #[test]
    fn recovery_budget_preserves_context_and_cache_commitments() {
        let mut plain = recovery_budget();
        // The serving envelope also contains 5,600 nonweight bytes. Removing
        // 1,100 bytes models an admission without retained and disk snapshots.
        let prefix_and_disk_reserve = 700 + 400;
        plain.initial_serving_peak_bytes -= prefix_and_disk_reserve;
        plain.effective_budget_bytes -= prefix_and_disk_reserve;
        let permitted = plain.preparation_budget_bytes(64, 128).unwrap();
        let mut cached = plain;
        cached.initial_serving_peak_bytes += prefix_and_disk_reserve;
        let original_cached_bound = cached.initial_serving_peak_bytes;
        assert!(cached.preparation_budget_bytes(64, 128).is_err());
        cached.effective_budget_bytes += prefix_and_disk_reserve;
        assert_eq!(cached.preparation_budget_bytes(64, 128).unwrap(), permitted);
        assert_eq!(cached.initial_serving_peak_bytes, original_cached_bound);
    }

    #[test]
    fn operating_preparation_retains_partial_tail_without_replacement_charge() {
        let mut budget = recovery_budget();
        budget.maximum_capacity = 250;
        budget.operating_growth = true;
        budget.effective_budget_bytes = 24_850;
        // Growing 178 to 192 retains the fifty-slot tail and appends fourteen
        // slots while old and replacement address tables remain live.
        assert_eq!(budget.preparation_budget_bytes(178, 192).unwrap(), 1_450);
        budget.effective_budget_bytes -= 1;
        assert!(budget.preparation_budget_bytes(178, 192).is_err());
        budget.effective_budget_bytes = 30_650;
        assert_eq!(budget.preparation_budget_bytes(192, 250).unwrap(), 5_850);
        assert!(budget.preparation_budget_bytes(192, 251).is_err());
        assert!(budget.preparation_budget_bytes(250, 256).is_err());
        assert_eq!(budget.preparation_budget_bytes(250, 250).unwrap(), 0);
    }

    #[test]
    fn operating_preparation_replays_32gib_145_slot_admission() {
        let budget = ExpertCapacityControlBudget {
            initial_capacity: 145,
            maximum_capacity: 204,
            startup_budget_bytes: 21_230_125_056,
            effective_budget_bytes: 25_726_861_312,
            initial_serving_peak_bytes: 21_157_819_868,
            packed_expert_bytes_per_slot: 76_103_680,
            replacement_address_table_overlap_bytes: 3_532_032,
            process_overhead_allowance_bytes: 2 << 30,
            operating_growth: true,
        };
        assert_eq!(
            budget.preparation_budget_bytes(145, 192).unwrap(),
            4_569_041_444
        );
        assert_eq!(
            budget.preparation_budget_bytes(192, 204).unwrap(),
            992_168_484
        );
        assert!(budget.preparation_budget_bytes(192, 205).is_err());
        assert!(budget.preparation_budget_bytes(192, 512).is_err());

        // Operating policy may use smaller steps and a non-aligned backoff cap.
        // Each quote still funds the complete target plus table overlap.
        for (current, target) in [(145, 160), (160, 176), (176, 192), (128, 129), (129, 144)] {
            let target_peak = budget.initial_serving_peak_bytes as i128
                + (target as i128 - budget.initial_capacity as i128)
                    * budget.packed_expert_bytes_per_slot as i128
                + budget.replacement_address_table_overlap_bytes as i128;
            let mut exact = budget;
            exact.effective_budget_bytes = u64::try_from(target_peak).unwrap();
            assert_eq!(
                exact.preparation_budget_bytes(current, target).unwrap(),
                (target - current) as u64 * budget.packed_expert_bytes_per_slot
                    + budget.replacement_address_table_overlap_bytes
            );
            exact.effective_budget_bytes -= 1;
            assert!(exact.preparation_budget_bytes(current, target).is_err());
        }

        for (current, target, exact_peak, preparation_bytes) in [
            (145, 192, 24_738_224_860, 3_580_404_992),
            (192, 204, 25_651_469_020, 916_776_192),
        ] {
            let mut exact = budget;
            exact.effective_budget_bytes = exact_peak;
            assert_eq!(
                exact.preparation_budget_bytes(current, target).unwrap(),
                preparation_bytes
            );
            exact.effective_budget_bytes -= 1;
            assert!(exact.preparation_budget_bytes(current, target).is_err());

            let mut no_table_overlap = budget;
            no_table_overlap.effective_budget_bytes =
                exact_peak - budget.replacement_address_table_overlap_bytes;
            assert!(
                no_table_overlap
                    .preparation_budget_bytes(current, target)
                    .is_err()
            );
        }
    }

    fn operating_admission(
        context: usize,
        retained: u64,
        disk: u64,
        prefill_capacity: usize,
    ) -> (
        PackageFootprint,
        OrdinaryPathEstimate,
        ExpertCapacityControlBudget,
    ) {
        let report = ordinary_report();
        let estimate = report
            .estimate_ordinary_path_with_prefill_capacity(
                OrdinaryPathInputs {
                    uncounted_core_reserve_bytes: ORDINARY_CORE_BUFFER_BYTES,
                    vec_spare_growth_overlap_reserve_bytes: Some(
                        report
                            .ordinary_vec_overlap_at_context(context, false)
                            .unwrap(),
                    ),
                    prefix_retained_budget_bytes: retained,
                    snapshot_capture_restore_overlap_reserve_bytes: Some(disk),
                    driver_allocator_allowance_bytes: 2 << 30,
                    ..ordinary_input(context, context.min(prefill_capacity), 511)
                },
                prefill_capacity,
            )
            .unwrap();
        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let startup_budget = estimate.accounted_component_sum_bytes
            - estimate.expert_stream_payload_at_ceiling_bytes
            + 117 * per_slot;
        let plan =
            plan_from_ordinary_estimate(&estimate, per_slot, 511, startup_budget, startup_budget)
                .unwrap();
        assert_eq!(plan.slots, 117);
        let budget =
            ExpertCapacityControlBudget::from_admission(&report, &estimate, &plan).unwrap();
        (report, estimate, budget)
    }

    #[test]
    fn operating_budget_preserves_serving_promises_and_caps_below_full_residency() {
        let (report, admitted, startup) = operating_admission(131072, 1 << 30, 1 << 29, 512);
        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let nonexpert =
            admitted.serving_phase_peak_bytes - admitted.expert_stream_payload_at_ceiling_bytes;
        let ceiling = nonexpert + 250 * per_slot + startup.replacement_address_table_overlap_bytes;
        let operating = startup
            .with_operating_budget(&report, &admitted, 131072, 512, ceiling)
            .unwrap();
        assert_eq!(startup.maximum_capacity(), 117);
        assert_eq!(operating.initial_capacity(), 117);
        assert_eq!(operating.maximum_capacity(), 250);
        let one_byte_short = startup
            .with_operating_budget(&report, &admitted, 131072, 512, ceiling - 1)
            .unwrap();
        assert_eq!(one_byte_short.maximum_capacity(), 249);
        let no_table_overlap = startup
            .with_operating_budget(
                &report,
                &admitted,
                131072,
                512,
                ceiling - startup.replacement_address_table_overlap_bytes,
            )
            .unwrap();
        assert_eq!(no_table_overlap.maximum_capacity(), 249);
        assert_eq!(
            operating.startup_budget_bytes(),
            startup.startup_budget_bytes()
        );
        assert_eq!(operating.operating_budget_bytes(), ceiling);
        assert_eq!(
            operating.initial_serving_peak_bytes(),
            startup.initial_serving_peak_bytes()
        );
        operating
            .validate_serving(&report, 131072, 512, 1 << 30, 1 << 29)
            .unwrap();
        assert!(
            operating
                .validate_serving(&report, 262144, 512, 1 << 30, 1 << 29)
                .is_err()
        );
        assert!(
            operating
                .validate_serving(&report, 131072, 2048, 1 << 30, 1 << 29)
                .is_err()
        );
        assert!(
            operating
                .validate_serving(&report, 131072, 512, 1 << 30, (1 << 29) + 1_000_000)
                .is_err()
        );

        let (_, uncached, uncached_startup) = operating_admission(131072, 0, 0, 512);
        let uncached_operating = uncached_startup
            .with_operating_budget(&report, &uncached, 131072, 512, ceiling)
            .unwrap();
        assert!(uncached_operating.maximum_capacity() > operating.maximum_capacity());
        let full_budget = startup
            .with_operating_budget(&report, &admitted, 131072, 512, u64::MAX)
            .unwrap();
        assert_eq!(full_budget.maximum_capacity(), 511);
        assert!(full_budget.preparation_budget_bytes(448, 512).is_err());
        assert!(
            full_budget
                .with_operating_budget(&report, &admitted, 131072, 512, u64::MAX)
                .is_err()
        );
        assert!(
            startup
                .with_operating_budget(
                    &report,
                    &admitted,
                    131072,
                    512,
                    startup.initial_serving_peak_bytes() - 1
                )
                .is_err()
        );
        assert!(
            startup
                .with_operating_budget(&report, &admitted, 131072, 512, 0)
                .is_err()
        );
    }

    #[test]
    fn operating_budget_rechecks_wide_workspace_and_ignores_completed_load_staging() {
        let (mut report, admitted, startup) = operating_admission(32768, 0, 0, 2048);
        let wide = startup
            .with_operating_budget(&report, &admitted, 32768, 2048, u64::MAX)
            .unwrap();
        wide.validate_serving(&report, 32768, 2048, 0, 0).unwrap();

        report.direct.largest_direct_payload_bytes = 8 << 30;
        let staged = report
            .estimate_ordinary_path_with_prefill_capacity(
                OrdinaryPathInputs {
                    uncounted_core_reserve_bytes: admitted.uncounted_core_reserve_bytes,
                    vec_spare_growth_overlap_reserve_bytes: Some(
                        admitted.vec_spare_growth_overlap_reserve_bytes,
                    ),
                    driver_allocator_allowance_bytes: admitted.driver_allocator_allowance_bytes,
                    ..ordinary_input(32768, 2048, 511)
                },
                2048,
            )
            .unwrap();
        assert_eq!(staged.limiting_phase, OrdinaryMemoryPhase::Startup);
        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let startup_bytes = staged.accounted_component_sum_bytes
            - staged.expert_stream_payload_at_ceiling_bytes
            + 117 * per_slot;
        let plan =
            plan_from_ordinary_estimate(&staged, per_slot, 511, startup_bytes, startup_bytes)
                .unwrap();
        let staged_budget =
            ExpertCapacityControlBudget::from_admission(&report, &staged, &plan).unwrap();
        let operating_bytes = staged.serving_phase_peak_bytes
            - staged.expert_stream_payload_at_ceiling_bytes
            + 200 * per_slot
            + staged_budget.replacement_address_table_overlap_bytes;
        assert!(operating_bytes < startup_bytes);
        let operating = staged_budget
            .with_operating_budget(&report, &staged, 32768, 2048, operating_bytes)
            .unwrap();
        assert_eq!(operating.maximum_capacity(), 200);
        assert!(operating.preparation_budget_bytes(117, 128).is_ok());
        assert!(staged_budget.preparation_budget_bytes(117, 128).is_err());

        let mut overflowing = staged;
        overflowing.serving_phase_peak_bytes = u64::MAX;
        assert!(
            staged_budget
                .with_operating_budget(&report, &overflowing, 32768, 2048, u64::MAX)
                .is_err()
        );
        let mut wrong_credit = staged_budget;
        wrong_credit.packed_expert_bytes_per_slot = 0;
        assert!(
            wrong_credit
                .with_operating_budget(&report, &staged, 32768, 2048, u64::MAX)
                .is_err()
        );
    }

    #[test]
    fn recovery_budget_rejects_invalid_extents_and_checked_arithmetic_failures() {
        let budget = recovery_budget();
        for (current, target) in [
            (0, 128),
            (63, 128),
            (65, 128),
            (179, 179),
            (64, 63),
            (64, 65),
            (64, 127),
            (64, 179),
            (64, 512),
        ] {
            assert!(budget.preparation_budget_bytes(current, target).is_err());
        }
        for initial in [9, 512] {
            let mut invalid = budget;
            invalid.initial_capacity = initial;
            assert!(invalid.preparation_budget_bytes(64, 128).is_err());
        }
        let mut invalid = budget;
        invalid.packed_expert_bytes_per_slot = 0;
        assert!(invalid.preparation_budget_bytes(64, 128).is_err());
        invalid = budget;
        invalid.packed_expert_bytes_per_slot = u64::MAX;
        assert!(invalid.preparation_budget_bytes(64, 128).is_err());
        invalid = budget;
        invalid.initial_serving_peak_bytes = 100;
        assert!(invalid.preparation_budget_bytes(64, 128).is_err());
        invalid = budget;
        invalid.effective_budget_bytes = 11_999;
        assert!(invalid.preparation_budget_bytes(64, 128).is_err());
        invalid = budget;
        invalid.initial_serving_peak_bytes = u64::MAX;
        invalid.effective_budget_bytes = u64::MAX;
        assert!(invalid.preparation_budget_bytes(64, 178).is_err());
        invalid = budget;
        invalid.initial_capacity = 128;
        invalid.initial_serving_peak_bytes = u64::MAX;
        invalid.effective_budget_bytes = u64::MAX;
        invalid.packed_expert_bytes_per_slot = u64::MAX / 64;
        invalid.replacement_address_table_overlap_bytes = 64;
        assert!(invalid.preparation_budget_bytes(64, 128).is_err());
    }

    #[test]
    fn recovery_budget_derives_selected_serving_and_actual_table_overlap() {
        let report = ordinary_report();
        let (_, full) = report
            .plan_explicit_ordinary_capacity(131072, u64::MAX, u64::MAX, 2 << 30)
            .unwrap();
        let (bounded, _) = report
            .plan_explicit_ordinary_capacity(
                131072,
                u64::MAX,
                full.accounted_peak_bytes - 1,
                2 << 30,
            )
            .unwrap();
        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let nonexpert =
            bounded.accounted_component_sum_bytes - bounded.expert_stream_payload_at_ceiling_bytes;
        let (selected, plan) = report
            .plan_explicit_ordinary_capacity(131072, u64::MAX, nonexpert + 178 * per_slot, 2 << 30)
            .unwrap();
        let budget =
            ExpertCapacityControlBudget::from_admission(&report, &selected, &plan).unwrap();
        assert_eq!(budget.initial_capacity(), 178);
        assert_eq!(
            budget.initial_serving_peak_bytes,
            selected.serving_phase_peak_bytes - (511 - 178) * per_slot
        );
        assert_eq!(budget.replacement_address_table_overlap_bytes, 3_532_032);
        assert_eq!(
            budget.preparation_budget_bytes(64, 128).unwrap(),
            114 * per_slot
        );
        assert!(budget.preparation_budget_bytes(64, 178).is_err());

        let mut corrupt = selected;
        corrupt.serving_phase_peak_bytes += 1;
        assert!(ExpertCapacityControlBudget::from_admission(&report, &corrupt, &plan).is_err());
        let mut corrupt_plan = plan;
        corrupt_plan.remaining_budget_bytes += 1;
        assert!(
            ExpertCapacityControlBudget::from_admission(&report, &selected, &corrupt_plan).is_err()
        );
    }

    #[test]
    fn recovery_budget_validates_actual_serving_obligations_and_package_credits() {
        let report = ordinary_report();
        let (_, full) = report
            .plan_explicit_ordinary_capacity(131072, u64::MAX, u64::MAX, 2 << 30)
            .unwrap();
        let (bounded, _) = report
            .plan_explicit_ordinary_capacity(
                131072,
                u64::MAX,
                full.accounted_peak_bytes - 1,
                2 << 30,
            )
            .unwrap();
        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let nonexpert =
            bounded.accounted_component_sum_bytes - bounded.expert_stream_payload_at_ceiling_bytes;
        let (selected, plan) = report
            .plan_explicit_ordinary_capacity(131072, u64::MAX, nonexpert + 178 * per_slot, 2 << 30)
            .unwrap();
        let budget =
            ExpertCapacityControlBudget::from_admission(&report, &selected, &plan).unwrap();
        assert_eq!(budget.process_overhead_allowance_bytes, 2 << 30);
        budget.validate_serving(&report, 131072, 512, 0, 0).unwrap();
        budget.validate_serving(&report, 1024, 512, 0, 0).unwrap();
        // Bounded storage allocates the admitted 511-slot table extent even
        // when the initial weight pool is smaller.
        assert!(
            budget
                .validate_serving(&report, 131072, 512, 1 << 20, 0)
                .is_err()
        );
        for (context, rows, retained, disk) in [
            (262144, 512, 0, 0),
            (131072, 2048, 0, 0),
            (131072, 512, 1 << 30, 0),
            (131072, 512, 0, 1 << 30),
            (131072, 512, u64::MAX, 0),
        ] {
            assert!(
                budget
                    .validate_serving(&report, context, rows, retained, disk)
                    .is_err()
            );
        }
        let mut wrong_credit = ordinary_report();
        wrong_credit.packed_expert_bytes_per_uniform_slot += 1;
        assert!(
            budget
                .validate_serving(&wrong_credit, 131072, 512, 0, 0)
                .is_err()
        );
        let mut wrong_metadata = ordinary_report();
        let _ = wrong_metadata.expert_layers.pop();
        assert!(
            budget
                .validate_serving(&wrong_metadata, 131072, 512, 0, 0)
                .is_err()
        );

        for omit_core in [true, false] {
            let mut incomplete = selected;
            if omit_core {
                incomplete.uncounted_core_reserve_bytes = 0;
            } else {
                incomplete.vec_spare_growth_overlap_reserve_bytes = 0;
            }
            incomplete.refresh_phase_peaks().unwrap();
            let accounted =
                incomplete.accounted_component_sum_bytes - (511 - plan.slots as u64) * per_slot;
            let incomplete_plan = CapacityPlan {
                accounted_peak_bytes: accounted,
                remaining_budget_bytes: plan.effective_budget_bytes - accounted,
                ..plan
            };
            let incomplete_budget =
                ExpertCapacityControlBudget::from_admission(&report, &incomplete, &incomplete_plan)
                    .unwrap();
            assert!(
                incomplete_budget
                    .validate_serving(&report, 131072, 512, 0, 0)
                    .is_err()
            );
        }
    }

    #[test]
    fn startup_dominant_admission_uses_the_complete_load_peak() {
        let mut report = ordinary_report();
        report.direct.largest_direct_payload_bytes = 2_000_000_000;
        let input = ordinary_input(1024, 512, 511);
        let base = report.estimate_ordinary_path(input).unwrap();
        assert_eq!(base.limiting_phase, OrdinaryMemoryPhase::Startup);
        assert_eq!(
            base.accounted_component_sum_bytes,
            base.startup_phase_peak_bytes
        );
        assert!(base.startup_phase_peak_bytes > base.serving_phase_peak_bytes);

        // A cache allowance belongs to serving even when a loading transient
        // determines admission. A sufficiently large cache changes that phase.
        let cached = report
            .estimate_ordinary_path(OrdinaryPathInputs {
                prefix_retained_budget_bytes: 300_000_000,
                snapshot_capture_restore_overlap_reserve_bytes: Some(200_000_000),
                ..input
            })
            .unwrap();
        assert_eq!(
            cached.startup_phase_peak_bytes,
            base.startup_phase_peak_bytes
        );
        assert_eq!(
            cached.engine_construction_phase_peak_bytes,
            base.engine_construction_phase_peak_bytes
        );
        assert_eq!(
            cached.serving_phase_peak_bytes,
            base.serving_phase_peak_bytes + 500_000_000
        );
        assert_eq!(
            cached.accounted_component_sum_bytes,
            base.accounted_component_sum_bytes
        );
        let large_cache = report
            .estimate_ordinary_path(OrdinaryPathInputs {
                prefix_retained_budget_bytes: 2_000_000_000,
                ..input
            })
            .unwrap();
        assert_eq!(large_cache.limiting_phase, OrdinaryMemoryPhase::Serving);
        assert_eq!(
            large_cache.startup_phase_peak_bytes,
            base.startup_phase_peak_bytes
        );
        assert_eq!(
            large_cache.serving_phase_peak_bytes,
            base.serving_phase_peak_bytes + 2_000_000_000
        );

        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let nonexpert = base.startup_phase_peak_bytes - base.expert_stream_payload_at_ceiling_bytes;
        let budget = |slots| nonexpert + slots * per_slot;
        for slots in [10, 12, 178, 511] {
            let exact =
                plan_from_ordinary_estimate(&base, per_slot, 511, u64::MAX, budget(slots)).unwrap();
            assert_eq!(exact.slots, slots as usize);
            assert_eq!(exact.accounted_peak_bytes, budget(slots));
            assert_eq!(exact.remaining_budget_bytes, 0);
            if slots > 10 {
                let below =
                    plan_from_ordinary_estimate(&base, per_slot, 511, u64::MAX, budget(slots) - 1)
                        .unwrap();
                assert_eq!(below.slots, slots as usize - 1);
            }
        }
        assert!(
            plan_from_ordinary_estimate(&base, per_slot, 511, u64::MAX, budget(10) - 1).is_err()
        );
        let host_limited =
            plan_from_ordinary_estimate(&base, per_slot, 511, budget(12), budget(178)).unwrap();
        assert_eq!(host_limited.slots, 12);
    }

    #[test]
    fn complete_phase_peaks_do_not_exceed_the_legacy_component_sum() {
        let report = ordinary_report();
        for (context, ceiling) in [(1, 10), (8448, 178), (131072, 511), (262144, 512)] {
            let estimate = report
                .estimate_ordinary_path(OrdinaryPathInputs {
                    prefix_retained_budget_bytes: 1 << 30,
                    snapshot_capture_restore_overlap_reserve_bytes: Some(512 << 20),
                    ..ordinary_input(context, context.min(512), ceiling)
                })
                .unwrap();
            let legacy = legacy_component_sum(estimate);
            for (_, bytes) in estimate.phase_component_sums().unwrap() {
                assert!(bytes <= legacy);
            }
            assert!(estimate.accounted_component_sum_bytes < legacy);
        }
    }

    #[test]
    fn phase_admission_recovers_slots_without_changing_context_or_cache_reserves() {
        let mut report = ordinary_report();
        report.direct.largest_direct_payload_bytes = report.direct.embedding_cpu_bytes;
        let (estimate, _) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                u64::MAX,
                u64::MAX,
                2 << 30,
                1 << 30,
                512 << 20,
            )
            .unwrap();
        let bounded = report
            .estimate_ordinary_path(OrdinaryPathInputs {
                context_limit: 131072,
                max_prefill_rows: 512,
                candidate_slot_ceiling: 511,
                uncounted_core_reserve_bytes: ORDINARY_CORE_BUFFER_BYTES,
                vec_spare_growth_overlap_reserve_bytes: Some(
                    report
                        .ordinary_vec_overlap_at_context(131072, false)
                        .unwrap(),
                ),
                prefix_retained_budget_bytes: 1 << 30,
                snapshot_capture_restore_overlap_reserve_bytes: Some(512 << 20),
                driver_allocator_allowance_bytes: 2 << 30,
            })
            .unwrap();
        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let legacy_nonexpert =
            legacy_component_sum(bounded) - bounded.expert_stream_payload_at_ceiling_bytes;
        let legacy_93_slot_budget = legacy_nonexpert + 93 * per_slot;
        let (selected, plan) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                legacy_93_slot_budget,
                legacy_93_slot_budget,
                2 << 30,
                1 << 30,
                512 << 20,
            )
            .unwrap();
        assert_eq!(plan.slots, 101);
        assert_eq!(selected, bounded);
        assert_eq!(selected.limiting_phase, OrdinaryMemoryPhase::Serving);
        assert!(plan.accounted_peak_bytes <= legacy_93_slot_budget);
        assert_eq!(
            selected.growing_session_logical_bytes,
            estimate.growing_session_logical_bytes
        );
        assert_eq!(
            selected.fixed_session_logical_bytes,
            estimate.fixed_session_logical_bytes
        );
        assert_eq!(selected.prefix_retained_budget_bytes, 1 << 30);
        assert_eq!(
            selected.snapshot_capture_restore_overlap_reserve_bytes,
            512 << 20
        );
        assert_eq!(selected.driver_allocator_allowance_bytes, 2 << 30);
    }

    #[test]
    fn optional_wide_estimation_failure_retains_the_validated_base_plan() {
        let report = ordinary_report();
        let (base, plan) = report
            .plan_ordinary_capacity_with_cache_budgets(
                32768,
                32 << 30,
                32 << 30,
                2 << 30,
                1 << 30,
                512 << 20,
            )
            .unwrap();
        let result = finish_optional_prefill_upgrade(
            base,
            plan,
            512,
            Err(anyhow::anyhow!("injected optional estimator failure")),
            2048,
            report.packed_expert_bytes_per_uniform_slot,
        );
        assert_eq!(result, (base, plan, 512));
    }

    #[test]
    fn prefill_upgrade_shares_bounded_budget_and_preserves_full_residency() {
        let report = ordinary_report();
        for context in [32768, 131072] {
            let (base, plan) = report
                .plan_ordinary_capacity_with_cache_budgets(
                    context,
                    32 << 30,
                    32 << 30,
                    2 << 30,
                    1 << 30,
                    512 << 20,
                )
                .unwrap();
            assert!(plan.slots < 512);
            let (chosen, chosen_plan, rows) = report
                .upgrade_prefill_capacity(context, base, plan)
                .unwrap();
            assert_eq!(rows, 2048);
            assert!((64..plan.slots).contains(&chosen_plan.slots));
            assert_eq!(
                chosen_plan.effective_budget_bytes,
                plan.effective_budget_bytes
            );
            assert!(chosen_plan.accounted_peak_bytes <= plan.effective_budget_bytes);
            assert_eq!(
                chosen.prefix_retained_budget_bytes,
                base.prefix_retained_budget_bytes
            );
            assert_eq!(
                chosen.snapshot_capture_restore_overlap_reserve_bytes,
                base.snapshot_capture_restore_overlap_reserve_bytes
            );
            assert_eq!(
                chosen.driver_allocator_allowance_bytes,
                base.driver_allocator_allowance_bytes
            );
        }
        let (base, plan) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                120 << 30,
                120 << 30,
                2 << 30,
                1 << 30,
                512 << 20,
            )
            .unwrap();
        assert_eq!(plan.slots, 512);
        let (chosen, chosen_plan, rows) =
            report.upgrade_prefill_capacity(131072, base, plan).unwrap();
        assert_eq!(rows, 2048);
        assert_eq!(chosen_plan.slots, plan.slots);
        assert_eq!(
            chosen.prefix_retained_budget_bytes,
            base.prefix_retained_budget_bytes
        );
        assert_eq!(
            chosen.snapshot_capture_restore_overlap_reserve_bytes,
            base.snapshot_capture_restore_overlap_reserve_bytes
        );
        assert!(chosen_plan.accounted_peak_bytes > plan.accounted_peak_bytes);
        assert!(chosen_plan.accounted_peak_bytes <= chosen_plan.effective_budget_bytes);
    }

    #[test]
    fn prefill_upgrade_keeps_narrow_workspace_when_wide_would_lose_floor_or_full_residency() {
        let report = ordinary_report();
        let (full, full_plan) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                120 << 30,
                120 << 30,
                2 << 30,
                1 << 30,
                512 << 20,
            )
            .unwrap();
        let exact_full_budget = full_plan.accounted_peak_bytes;
        let (base, plan) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                exact_full_budget,
                exact_full_budget,
                2 << 30,
                1 << 30,
                512 << 20,
            )
            .unwrap();
        assert_eq!(plan.slots, 512);
        assert_eq!(
            report.upgrade_prefill_capacity(131072, base, plan).unwrap(),
            (base, plan, 512)
        );
        assert_eq!(
            full.prefill_workspace_requested_bytes,
            base.prefill_workspace_requested_bytes
        );

        let (bounded, _) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                32 << 30,
                32 << 30,
                2 << 30,
                1 << 30,
                512 << 20,
            )
            .unwrap();
        let floor_budget = bounded.accounted_component_sum_bytes
            - bounded.expert_stream_payload_at_ceiling_bytes
            + 64 * report.packed_expert_bytes_per_uniform_slot;
        let (base, plan) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                floor_budget,
                floor_budget,
                2 << 30,
                1 << 30,
                512 << 20,
            )
            .unwrap();
        assert_eq!(plan.slots, 64);
        assert_eq!(
            report.upgrade_prefill_capacity(131072, base, plan).unwrap(),
            (base, plan, 512)
        );
    }

    #[test]
    fn explicit_prefill_upgrade_keeps_automatic_width_and_all_reserves() {
        let report = ordinary_report();
        let (base, plan) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                32 << 30,
                32 << 30,
                2 << 30,
                1 << 30,
                512 << 20,
            )
            .unwrap();
        let (automatic, automatic_plan, automatic_rows) =
            report.upgrade_prefill_capacity(131072, base, plan).unwrap();
        assert_eq!(automatic_rows, 2048);
        assert_eq!(
            report
                .upgrade_prefill_capacity_to(131072, base, plan, 2048)
                .unwrap(),
            (automatic, automatic_plan, automatic_rows)
        );
        let (explicit, explicit_plan, rows) = report
            .upgrade_prefill_capacity_to(131072, base, plan, 4096)
            .unwrap();
        assert_eq!(rows, 4096);
        assert!((64..automatic_plan.slots).contains(&explicit_plan.slots));
        assert_eq!(
            explicit_plan.effective_budget_bytes,
            plan.effective_budget_bytes
        );
        assert!(explicit_plan.accounted_peak_bytes <= plan.effective_budget_bytes);
        assert_eq!(
            explicit.uncounted_core_reserve_bytes,
            base.uncounted_core_reserve_bytes
        );
        assert_eq!(
            explicit.vec_spare_growth_overlap_reserve_bytes,
            base.vec_spare_growth_overlap_reserve_bytes
        );
        assert_eq!(
            explicit.prefix_retained_budget_bytes,
            base.prefix_retained_budget_bytes
        );
        assert_eq!(
            explicit.snapshot_capture_restore_overlap_reserve_bytes,
            base.snapshot_capture_restore_overlap_reserve_bytes
        );
        assert_eq!(
            explicit.driver_allocator_allowance_bytes,
            base.driver_allocator_allowance_bytes
        );
        assert_eq!(
            report
                .upgrade_prefill_capacity_to(131072, base, plan, 512)
                .unwrap(),
            (base, plan, 512)
        );
        assert!(
            report
                .upgrade_prefill_capacity_to(131072, base, plan, 513)
                .is_err()
        );
        assert!(
            report
                .upgrade_prefill_capacity_to(131072, base, plan, 4097)
                .is_err()
        );
        assert!(
            report
                .upgrade_prefill_capacity_to(131072, automatic, automatic_plan, 4096)
                .is_err()
        );
    }

    #[test]
    fn explicit_4096_refusal_retains_2048_and_full_residency_or_floor() {
        let report = ordinary_report();
        let context = 131072;
        let plan_at = |budget| {
            report
                .plan_ordinary_capacity_with_cache_budgets(
                    context,
                    budget,
                    budget,
                    2 << 30,
                    1 << 30,
                    512 << 20,
                )
                .unwrap()
        };
        let (base, plan) = plan_at(120 << 30);
        let explicit_full = report
            .upgrade_prefill_capacity_to(context, base, plan, 4096)
            .unwrap();
        assert_eq!(explicit_full.1.slots, 512);
        assert_eq!(explicit_full.2, 4096);
        assert!(explicit_full.1.accounted_peak_bytes <= plan.effective_budget_bytes);
        let (wide, _, rows) = report
            .upgrade_prefill_capacity(context, base, plan)
            .unwrap();
        assert_eq!(rows, 2048);
        let (base, plan) = plan_at(wide.accounted_component_sum_bytes);
        assert_eq!(plan.slots, 512);
        let full = report
            .upgrade_prefill_capacity(context, base, plan)
            .unwrap();
        assert_eq!(full.1.slots, 512);
        assert_eq!(full.2, 2048);
        assert_eq!(
            report
                .upgrade_prefill_capacity_to(context, base, plan, 4096)
                .unwrap(),
            full
        );

        let (base, plan) = plan_at(32 << 30);
        let (wide, _, _) = report
            .upgrade_prefill_capacity(context, base, plan)
            .unwrap();
        let floor_budget = wide.accounted_component_sum_bytes
            - wide.expert_stream_payload_at_ceiling_bytes
            + 64 * report.packed_expert_bytes_per_uniform_slot;
        let (base, plan) = plan_at(floor_budget);
        let floor = report
            .upgrade_prefill_capacity(context, base, plan)
            .unwrap();
        assert_eq!(floor.1.slots, 64);
        assert_eq!(floor.2, 2048);
        assert_eq!(
            report
                .upgrade_prefill_capacity_to(context, base, plan, 4096)
                .unwrap(),
            floor
        );
        assert_eq!(
            finish_optional_prefill_upgrade(
                floor.0,
                floor.1,
                floor.2,
                Err(anyhow::anyhow!("injected optional 4096 estimator failure")),
                4096,
                report.packed_expert_bytes_per_uniform_slot,
            ),
            floor
        );
    }

    #[test]
    fn persistent_prefill_workspace_is_charged_for_construction_and_serving() {
        let report = ordinary_report();
        let decode = report
            .estimate_ordinary_path(ordinary_input(1024, 1, 12))
            .unwrap();
        let prefill = report
            .estimate_ordinary_path(ordinary_input(1024, 512, 12))
            .unwrap();
        let requested = native_prefill_workspace_bytes(1024).unwrap() as u64;
        assert!(requested > 500_000_000);
        assert_eq!(decode.prefill_workspace_requested_bytes, requested);
        assert_eq!(prefill.prefill_workspace_requested_bytes, requested);
        for estimate in [decode, prefill] {
            let common = estimate.direct_wire_payload_bytes
                + estimate.expert_stream_payload_at_ceiling_bytes
                + estimate.expert_storage_metadata_reserve_bytes
                + estimate.uncounted_core_reserve_bytes
                + estimate.driver_allocator_allowance_bytes;
            assert_eq!(
                estimate.startup_phase_peak_bytes,
                common
                    + estimate
                        .startup_workspace_across_candidates
                        .combined_component_bytes
            );
            assert_eq!(
                estimate.engine_construction_phase_peak_bytes,
                common + requested + estimate.expert_reader_workspace_across_candidates_bytes
            );
            assert_eq!(
                estimate.serving_phase_peak_bytes,
                common
                    + estimate.fixed_session_logical_bytes
                    + estimate.growing_session_logical_bytes
                    + requested
                    + estimate
                        .runtime_workspace_across_candidates
                        .combined_component_bytes
                    + estimate.vec_spare_growth_overlap_reserve_bytes
                    + estimate.prefix_retained_budget_bytes
                    + estimate.snapshot_capture_restore_overlap_reserve_bytes
            );
            assert_eq!(
                estimate.accounted_component_sum_bytes,
                estimate
                    .startup_phase_peak_bytes
                    .max(estimate.engine_construction_phase_peak_bytes)
                    .max(estimate.serving_phase_peak_bytes)
            );
            assert_eq!(estimate.limiting_phase, OrdinaryMemoryPhase::Serving);
            assert!(
                estimate.serving_phase_peak_bytes > estimate.engine_construction_phase_peak_bytes
            );
        }
        assert_eq!(
            decode.startup_phase_peak_bytes,
            prefill.startup_phase_peak_bytes
        );
        assert_eq!(
            decode.engine_construction_phase_peak_bytes,
            prefill.engine_construction_phase_peak_bytes
        );
        assert!(prefill.serving_phase_peak_bytes > decode.serving_phase_peak_bytes);
        assert!(native_prefill_workspace_bytes(0).is_err());
        assert!(native_prefill_workspace_bytes(262_145).is_err());
    }

    #[test]
    fn ordinary_runtime_workspace_counts_queued_arguments_without_old_dequant_buffers() {
        let report = ordinary_report();
        for capacity in [12, 178, 512] {
            let estimate = report
                .estimate_ordinary_path(ordinary_input(1024, 512, capacity))
                .unwrap();
            assert!(
                estimate.runtime_metal_requested_upper_bytes
                    >= estimate
                        .runtime_workspace_across_candidates
                        .metal_requested_bytes
            );
            assert_eq!(
                estimate.expert_stream_payload_at_ceiling_bytes,
                report.packed_expert_bytes_at_capacity(capacity).unwrap()
            );
            assert!(estimate.runtime_workspace_at_ceiling.metal_requested_bytes < 4_000_000);
        }
    }

    #[test]
    fn reconstructed_qsa_arguments_have_an_explicit_queued_reserve() {
        for (rows, increment) in [(512usize, 1296u64), (2048, 4752)] {
            let prefill = runtime_workspaces(rows, 512, 0)
                .unwrap()
                .into_iter()
                .find(|stage| stage.stage == OrdinaryWorkspaceStage::PrefillQueuedGraph)
                .unwrap();
            let tiles = rows.div_ceil(64) as u64;
            let previous_qsa = 12 * (44 * rows as u64 + 52 + 36 * tiles);
            let unchanged_mlp = 48 * (2048 + 16);
            assert_eq!(
                prefill.metal_requested_bytes,
                previous_qsa + unchanged_mlp + increment
            );
        }
    }

    #[test]
    fn ordinary_workspace_bounds_bounded_groups_without_a_511_dequant_peak() {
        let report = ordinary_report();
        let bounded = report
            .estimate_ordinary_path(ordinary_input(1024, 512, 511))
            .unwrap();
        let full = report
            .estimate_ordinary_path(ordinary_input(1024, 512, 512))
            .unwrap();
        assert_eq!(
            bounded.runtime_workspace_at_ceiling.stage,
            OrdinaryWorkspaceStage::PrefillQueuedGraph
        );
        assert!(
            full.runtime_workspace_at_ceiling.combined_component_bytes
                < bounded
                    .runtime_workspace_at_ceiling
                    .combined_component_bytes
        );
        assert!(full.runtime_workspace_across_candidates.capacity_slots < 512);
        assert_eq!(
            full.startup_workspace_across_candidates.stage,
            OrdinaryWorkspaceStage::StartupDirect
        );
        assert!(bounded.runtime_workspace_at_ceiling.metal_requested_bytes < 4_000_000);
    }

    #[test]
    fn ordinary_startup_pool_and_prefix_overlap_remain_separate_components() {
        let mut report = ordinary_report();
        report.direct.largest_direct_payload_bytes = 1;
        let input = ordinary_input(1024, 512, 512);
        let base = report.estimate_ordinary_path(input).unwrap();
        assert_eq!(
            base.startup_workspace_across_candidates.stage,
            OrdinaryWorkspaceStage::StartupExpertPool
        );
        assert_eq!(
            base.startup_workspace_across_candidates.host_logical_bytes,
            (511 + 1) * 600_000
                + (crate::package::PREFILL_READ_ROWS as u64 + 1) * 1_556_480
                + crate::package::PREFILL_READ_WORKERS as u64
                    * crate::package::PREFILL_READ_STACK_BYTES as u64
                + crate::package::PREFILL_READ_CONTROL_BYTES as u64
        );
        assert_eq!(base.startup_workspace_across_candidates.capacity_slots, 511);
        let with_cache = report
            .estimate_ordinary_path(OrdinaryPathInputs {
                prefix_retained_budget_bytes: 700_000_000,
                snapshot_capture_restore_overlap_reserve_bytes: Some(400_000_000),
                ..input
            })
            .unwrap();
        assert_eq!(
            with_cache.accounted_component_sum_bytes - base.accounted_component_sum_bytes,
            1_100_000_000
        );
        assert_eq!(
            with_cache.runtime_workspace_across_candidates,
            base.runtime_workspace_across_candidates
        );
        assert_eq!(
            with_cache.startup_phase_peak_bytes,
            base.startup_phase_peak_bytes
        );
        assert_eq!(
            with_cache.engine_construction_phase_peak_bytes,
            base.engine_construction_phase_peak_bytes
        );
        assert_eq!(
            with_cache.serving_phase_peak_bytes,
            base.serving_phase_peak_bytes + 1_100_000_000
        );
    }

    #[test]
    fn ordinary_context_boundaries_and_startup_staging_are_labelled() {
        let report = ordinary_report();
        let at = |context| {
            report
                .estimate_ordinary_path(ordinary_input(context, 2, 12))
                .unwrap()
        };
        let sink = at(128);
        let before = at(8447);
        let sealed = at(8448);
        let max = at(262_144);
        assert_eq!(
            sink.growing_session_logical_bytes,
            growing_session_payload(128, 262_144, 12)
                .unwrap()
                .total_logical_bytes
        );
        assert_eq!(
            sealed.growing_session_logical_bytes - before.growing_session_logical_bytes,
            12 * (TILE_RECORD_BYTES as u64 - 4 * INDEX_WIDTH as u64)
        );
        assert!(max.growing_session_logical_bytes > sealed.growing_session_logical_bytes);
        assert_eq!(
            max.startup_workspace_across_candidates.stage,
            OrdinaryWorkspaceStage::StartupDirect
        );
        assert_eq!(
            max.startup_workspace_across_candidates.host_logical_bytes,
            520_000_000
                + (crate::package::PREFILL_READ_ROWS as u64 + 1) * 1_556_480
                + crate::package::PREFILL_READ_WORKERS as u64
                    * crate::package::PREFILL_READ_STACK_BYTES as u64
                + crate::package::PREFILL_READ_CONTROL_BYTES as u64
        );
        assert!(max.minimum_snapshot_logical_bytes > max.fixed_session_logical_bytes);
    }

    #[test]
    fn ordinary_estimator_requires_explicit_reserves_and_checked_geometry() {
        let report = ordinary_report();
        let input = ordinary_input(1024, 512, 12);
        assert!(
            report
                .estimate_ordinary_path(OrdinaryPathInputs {
                    vec_spare_growth_overlap_reserve_bytes: None,
                    ..input
                })
                .is_err()
        );
        assert!(
            report
                .estimate_ordinary_path(OrdinaryPathInputs {
                    snapshot_capture_restore_overlap_reserve_bytes: None,
                    ..input
                })
                .is_err()
        );
        for invalid in [
            OrdinaryPathInputs {
                context_limit: 0,
                ..input
            },
            OrdinaryPathInputs {
                context_limit: 262_145,
                ..input
            },
            OrdinaryPathInputs {
                max_prefill_rows: 0,
                ..input
            },
            OrdinaryPathInputs {
                max_prefill_rows: 513,
                ..input
            },
            OrdinaryPathInputs {
                candidate_slot_ceiling: 9,
                ..input
            },
            OrdinaryPathInputs {
                candidate_slot_ceiling: 513,
                ..input
            },
            OrdinaryPathInputs {
                driver_allocator_allowance_bytes: u64::MAX,
                ..input
            },
        ] {
            assert!(report.estimate_ordinary_path(invalid).is_err());
        }
        let one = report
            .estimate_ordinary_path(ordinary_input(1, 1, 10))
            .unwrap();
        assert_eq!(
            one.runtime_workspace_at_ceiling.stage,
            OrdinaryWorkspaceStage::DecodeQueuedGraph
        );
        let mut bad_header = ordinary_report();
        bad_header.direct.largest_direct_payload_bytes = u64::MAX;
        assert!(bad_header.estimate_ordinary_path(input).is_err());
    }

    #[test]
    fn explicit_ordinary_reserve_tracks_sink_seal_and_context_ceiling() {
        let report = ordinary_report();
        let minimum = 12 * QSA_DEVICE_INDEX_FIXED_CAPACITY_BYTES;
        for context in [1, 2, 3] {
            assert_eq!(
                report
                    .ordinary_vec_overlap_at_context(context, false)
                    .unwrap(),
                minimum
            );
            assert_eq!(
                report
                    .ordinary_vec_overlap_at_context(context, true)
                    .unwrap(),
                minimum
            );
        }
        for context in [4, 8_447, 8_448, 131_072, 131_075, 262_144] {
            let growing = report.growing_session_payload_at_frontier(context).unwrap();
            let per_layer_backing = growing.kvarn_record_tiles_per_qsa_layer as u64
                * TILE_RECORD_BYTES as u64
                + growing.qsa_index_complete_groups_per_layer as u64 * INDEX_WIDTH as u64 * 2;
            let bounded = report
                .ordinary_vec_overlap_at_context(context, false)
                .unwrap();
            let full = report
                .ordinary_vec_overlap_at_context(context, true)
                .unwrap();
            assert_eq!(bounded, minimum + per_layer_backing);
            assert_eq!(full, minimum + 12 * per_layer_backing);
            assert_eq!(full - bounded, 11 * per_layer_backing);
        }
        assert_eq!(
            report
                .ordinary_vec_overlap_at_context(131_072, false)
                .unwrap(),
            75_672_064
        );
        assert_eq!(
            report
                .ordinary_vec_overlap_at_context(131_072, true)
                .unwrap(),
            907_895_808
        );
        assert!(
            report
                .ordinary_vec_overlap_at_context(262_145, false)
                .is_err()
        );
        assert!(
            report
                .ordinary_vec_overlap_at_context(262_145, true)
                .is_err()
        );
    }

    #[test]
    fn explicit_ordinary_plan_separates_full_and_bounded_exact_fits() {
        let report = ordinary_report();
        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let (full_estimate, full) = report
            .plan_explicit_ordinary_capacity(131_072, u64::MAX, u64::MAX, 2 << 30)
            .unwrap();
        assert_eq!(full.slots, 512);
        let full_budget = full.accounted_peak_bytes;
        assert_eq!(full_budget, full_estimate.accounted_component_sum_bytes);
        let (exact, full_plan) = report
            .plan_explicit_ordinary_capacity(131_072, u64::MAX, full_budget, 2 << 30)
            .unwrap();
        assert_eq!(full_plan.slots, 512);
        assert_eq!(exact.vec_spare_growth_overlap_reserve_bytes, 907_895_808);
        let (bounded, bounded_plan) = report
            .plan_explicit_ordinary_capacity(131_072, u64::MAX, full_budget - 1, 2 << 30)
            .unwrap();
        assert_eq!(bounded_plan.slots, 511);
        assert_eq!(bounded.vec_spare_growth_overlap_reserve_bytes, 75_672_064);
        assert_eq!(
            bounded.expert_stream_payload_at_ceiling_bytes,
            511 * per_slot
        );
        assert_eq!(bounded.runtime_workspace_at_ceiling.capacity_slots, 511);
        assert_eq!(
            bounded_plan.accounted_peak_bytes,
            bounded.accounted_component_sum_bytes
        );
        assert_eq!(
            bounded.growing_session_logical_bytes,
            full_estimate.growing_session_logical_bytes
        );
        assert_eq!(
            bounded.fixed_session_logical_bytes,
            full_estimate.fixed_session_logical_bytes
        );
        assert_eq!(
            bounded.prefill_workspace_requested_bytes,
            full_estimate.prefill_workspace_requested_bytes
        );

        let bounded_budget = bounded.accounted_component_sum_bytes;
        let (_, exact_511) = report
            .plan_explicit_ordinary_capacity(131_072, u64::MAX, bounded_budget, 2 << 30)
            .unwrap();
        assert_eq!(exact_511.slots, 511);
        assert_eq!(exact_511.accounted_peak_bytes, bounded_budget);
        let (_, below_511) = report
            .plan_explicit_ordinary_capacity(131_072, u64::MAX, bounded_budget - 1, 2 << 30)
            .unwrap();
        assert_eq!(below_511.slots, 510);

        let bounded_nonexpert =
            bounded.accounted_component_sum_bytes - bounded.expert_stream_payload_at_ceiling_bytes;
        let minimum_budget = bounded_nonexpert + 10 * per_slot;
        let (_, minimum) = report
            .plan_explicit_ordinary_capacity(131_072, u64::MAX, minimum_budget, 2 << 30)
            .unwrap();
        assert_eq!(minimum.slots, 10);
        assert_eq!(minimum.accounted_peak_bytes, minimum_budget);
        assert!(
            report
                .plan_explicit_ordinary_capacity(131_072, u64::MAX, minimum_budget - 1, 2 << 30)
                .is_err()
        );
    }

    #[test]
    fn explicit_ordinary_plan_is_monotone_and_rejects_too_small_budget() {
        let report = ordinary_report();
        let allowance = 2 * (1u64 << 30);
        let (estimate, full) = report
            .plan_explicit_ordinary_capacity(1024, u64::MAX, u64::MAX, allowance)
            .unwrap();
        assert_eq!(full.slots, 512);
        assert_eq!(
            full.accounted_peak_bytes,
            estimate.accounted_component_sum_bytes
        );
        assert_eq!(
            estimate.uncounted_core_reserve_bytes,
            ORDINARY_CORE_BUFFER_BYTES
        );
        assert_eq!(estimate.driver_allocator_allowance_bytes, allowance);
        assert_eq!(estimate.prefix_retained_budget_bytes, 0);
        assert_eq!(estimate.snapshot_capture_restore_overlap_reserve_bytes, 0);
        assert_eq!(
            estimate.runtime_workspace_across_candidates.capacity_slots,
            10
        );

        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let base = estimate.accounted_component_sum_bytes
            - estimate.expert_stream_payload_at_ceiling_bytes;
        let budget = |slots| base + slots * per_slot;
        for slots in [10, 12, 178, 511, 512] {
            let plan =
                plan_from_ordinary_estimate(&estimate, per_slot, 512, u64::MAX, budget(slots))
                    .unwrap();
            assert_eq!(plan.slots, slots as usize);
            assert_eq!(plan.accounted_peak_bytes, budget(slots));
        }
        assert!(
            plan_from_ordinary_estimate(&estimate, per_slot, 512, u64::MAX, budget(10) - 1)
                .is_err()
        );
        let host_limited =
            plan_from_ordinary_estimate(&estimate, per_slot, 512, budget(12), budget(512)).unwrap();
        assert_eq!(host_limited.slots, 12);
        assert!(
            report
                .plan_explicit_ordinary_capacity(0, u64::MAX, u64::MAX, allowance)
                .is_err()
        );
        assert!(
            report
                .plan_explicit_ordinary_capacity(1024, u64::MAX, 0, allowance)
                .is_err()
        );
        let mut unsupported = ordinary_report();
        unsupported.qsa_layers = 11;
        assert!(
            unsupported
                .plan_explicit_ordinary_capacity(1024, u64::MAX, u64::MAX, allowance)
                .is_err()
        );
        assert!(
            report
                .plan_explicit_ordinary_capacity(1024, u64::MAX, u64::MAX, u64::MAX)
                .is_err()
        );
        let mut inconsistent = estimate;
        inconsistent.expert_stream_payload_at_ceiling_bytes =
            inconsistent.accounted_component_sum_bytes + 1;
        assert!(
            plan_from_ordinary_estimate(&inconsistent, per_slot, 512, u64::MAX, u64::MAX).is_err()
        );
        inconsistent = estimate;
        inconsistent.growing_session_logical_bytes = u64::MAX;
        assert!(
            plan_from_ordinary_estimate(&inconsistent, per_slot, 512, u64::MAX, u64::MAX).is_err()
        );
        inconsistent = estimate;
        inconsistent.expert_reader_workspace_across_candidates_bytes = u64::MAX;
        assert!(
            plan_from_ordinary_estimate(&inconsistent, per_slot, 512, u64::MAX, u64::MAX).is_err()
        );
        inconsistent = estimate;
        inconsistent.expert_storage_metadata_reserve_bytes = u64::MAX;
        assert!(
            plan_from_ordinary_estimate(&inconsistent, per_slot, 512, u64::MAX, u64::MAX).is_err()
        );
        inconsistent = estimate;
        inconsistent.startup_phase_peak_bytes += 1;
        assert!(
            plan_from_ordinary_estimate(&inconsistent, per_slot, 512, u64::MAX, u64::MAX).is_err()
        );
        inconsistent = estimate;
        inconsistent.limiting_phase = OrdinaryMemoryPhase::Startup;
        assert!(
            plan_from_ordinary_estimate(&inconsistent, per_slot, 512, u64::MAX, u64::MAX).is_err()
        );
    }

    #[test]
    fn explicit_memory_only_prefix_budget_is_charged_once() {
        let report = ordinary_report();
        let allowance = 2 * (1u64 << 30);
        let (base, _) = report
            .plan_explicit_ordinary_capacity(1, u64::MAX, u64::MAX, allowance)
            .unwrap();
        let per_slot = report.packed_expert_bytes_per_uniform_slot;
        let retained = per_slot;
        let (cached, _) = report
            .plan_explicit_ordinary_capacity_with_prefix_budget(
                1,
                u64::MAX,
                u64::MAX,
                allowance,
                retained,
            )
            .unwrap();
        assert_eq!(cached.prefix_retained_budget_bytes, retained);
        assert_eq!(cached.snapshot_capture_restore_overlap_reserve_bytes, 0);
        assert_eq!(
            cached.accounted_component_sum_bytes,
            base.accounted_component_sum_bytes + retained
        );
        assert_eq!(
            cached.startup_phase_peak_bytes,
            base.startup_phase_peak_bytes
        );
        assert_eq!(
            cached.engine_construction_phase_peak_bytes,
            base.engine_construction_phase_peak_bytes
        );
        assert_eq!(
            cached.serving_phase_peak_bytes,
            base.serving_phase_peak_bytes + retained
        );
        let nonexpert = base.accounted_component_sum_bytes
            - base.expert_stream_payload_at_ceiling_bytes
            + report.expert_storage_metadata_bytes(511).unwrap();
        let budget = nonexpert + 12 * per_slot;
        let (_, without_cache) = report
            .plan_explicit_ordinary_capacity(1, u64::MAX, budget, allowance)
            .unwrap();
        let (_, with_cache) = report
            .plan_explicit_ordinary_capacity_with_prefix_budget(
                1,
                u64::MAX,
                budget,
                allowance,
                retained,
            )
            .unwrap();
        assert_eq!(without_cache.slots, 12);
        assert_eq!(with_cache.slots, 11);
        assert!(
            report
                .plan_explicit_ordinary_capacity_with_prefix_budget(
                    1,
                    u64::MAX,
                    budget,
                    allowance,
                    3 * per_slot,
                )
                .is_err()
        );
    }

    #[test]
    fn disk_snapshot_overlap_is_separate_from_retained_memory_and_charged_once() {
        let report = ordinary_report();
        let retained = 1u64 << 30;
        let disk = 512u64 << 20;
        let (memory, _) = report
            .plan_explicit_ordinary_capacity_with_prefix_budget(
                131072,
                u64::MAX,
                u64::MAX,
                2 << 30,
                retained,
            )
            .unwrap();
        let (both, _) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                u64::MAX,
                u64::MAX,
                2 << 30,
                retained,
                disk,
            )
            .unwrap();
        assert_eq!(both.prefix_retained_budget_bytes, retained);
        assert_eq!(both.snapshot_capture_restore_overlap_reserve_bytes, disk);
        assert_eq!(
            both.accounted_component_sum_bytes,
            memory.accounted_component_sum_bytes + disk
        );
        assert_eq!(
            both.startup_phase_peak_bytes,
            memory.startup_phase_peak_bytes
        );
        assert_eq!(
            both.engine_construction_phase_peak_bytes,
            memory.engine_construction_phase_peak_bytes
        );
        assert_eq!(
            both.serving_phase_peak_bytes,
            memory.serving_phase_peak_bytes + disk
        );
        assert_eq!(
            both.fixed_session_logical_bytes,
            memory.fixed_session_logical_bytes
        );
        assert_eq!(
            both.growing_session_logical_bytes,
            memory.growing_session_logical_bytes
        );
        let (_, fits) = report
            .plan_ordinary_capacity_with_cache_budgets(
                131072,
                u64::MAX,
                both.accounted_component_sum_bytes,
                2 << 30,
                retained,
                disk,
            )
            .unwrap();
        assert_eq!(fits.slots, 512);
        assert!(
            report
                .plan_ordinary_capacity_with_cache_budgets(
                    131072,
                    u64::MAX,
                    both.accounted_component_sum_bytes,
                    2 << 30,
                    retained,
                    u64::MAX,
                )
                .is_err()
        );
    }
}

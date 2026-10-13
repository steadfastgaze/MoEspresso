//! Deep, same-engine copies of committed Qwen session state.
//!
//! Expert pools remain Engine-owned. Restoring a session leaves their current
//! residency and counters intact, including when decode uses Cache-Prior.

use std::cell::OnceCell;
use std::mem::size_of;
use std::rc::Rc;

use anyhow::{Context, Result, ensure};
use half::bf16;
use objc2::rc::autoreleasepool;

use super::{Engine, LayerState, ProducerSpan, QsaState, Session, bytes};
use crate::decoder::GdnLayerState;
use crate::kvarn::DeviceKvarnState;
use crate::model::{MixerKind, QwenConfig};
use crate::ple::{DevicePleState, PleState};
use crate::qsa_index::DeviceQsaIndexState;

#[cfg(target_os = "macos")]
mod disk;
#[cfg(target_os = "macos")]
pub use disk::{
    DiskCacheMode, DiskCheckpointSlot, DiskCheckpointStore, DiskLimits, IO_BUFFER_BYTES,
    StoreLimits,
};

// Identifies the persisted state representation and supported numerical producer.
const PRODUCER_VERSION: u32 = 4;
const VOCAB: usize = 248_320;

struct SavedGdn {
    frontier: u64,
    conv_bf16: Vec<u8>,
    recurrent_f32: Vec<u8>,
}

struct SavedKvarn {
    frontier: usize,
    records: Vec<u8>,
    sink_keys: Vec<u8>,
    sink_values: Vec<u8>,
    tail_keys: Vec<u8>,
    tail_values: Vec<u8>,
}

struct SavedQsa {
    index_frontier: u64,
    compressed_index_bf16: Vec<u16>,
    pending_index_bf16: Vec<u16>,
    kv: SavedKvarn,
}

enum SavedLayer {
    Gdn(SavedGdn),
    Qsa(SavedQsa),
}

/// A host-owned copy at one completed token frontier. Fields are private so
/// callers cannot substitute token provenance or mix layer frontiers.
/// Restoration preserves the original cache producer even after capacity changes.
pub struct SessionSnapshot {
    owner: Rc<()>,
    cache_origin: Rc<()>,
    producer_version: u32,
    prefill_capacity: usize,
    max_context: usize,
    eos_token_id: u32,
    frontier: usize,
    consumed_tokens: Vec<u32>,
    last_logits_bf16: Vec<u16>,
    producer_spans: Vec<ProducerSpan>,
    ple: PleState,
    layers: Vec<SavedLayer>,
}

fn add(total: u64, bytes: u64) -> Result<u64> {
    total
        .checked_add(bytes)
        .context("snapshot byte charge overflow")
}

fn vector_charge<T>(values: &Vec<T>) -> Result<u64> {
    u64::try_from(values.capacity())?
        .checked_mul(u64::try_from(size_of::<T>())?)
        .context("snapshot vector capacity overflow")
}

fn length_charge<T>(length: usize) -> Result<u64> {
    u64::try_from(length)?
        .checked_mul(u64::try_from(size_of::<T>())?)
        .context("snapshot payload length overflow")
}

struct SnapshotBudget {
    limit: u64,
    charged: u64,
}

impl SnapshotBudget {
    fn new(limit: u64) -> Result<Option<Self>> {
        let charged = u64::try_from(size_of::<SessionSnapshot>())?;
        Ok((charged <= limit).then_some(Self { limit, charged }))
    }

    fn can_fit(&self, bytes: u64) -> Result<bool> {
        Ok(add(self.charged, bytes)? <= self.limit)
    }

    fn charge(&mut self, bytes: u64) -> Result<bool> {
        let next = add(self.charged, bytes)?;
        if next > self.limit {
            return Ok(false);
        }
        self.charged = next;
        Ok(true)
    }

    fn copy<T: Clone>(&mut self, source: &[T]) -> Result<Option<Vec<T>>> {
        if !self.can_fit(length_charge::<T>(source.len())?)? {
            return Ok(None);
        }
        let copy = source.to_vec();
        Ok(self.charge(vector_charge(&copy)?)?.then_some(copy))
    }

    fn read_buffer(
        &mut self,
        source: &crate::metal::Buffer,
        len: usize,
    ) -> Result<Option<Vec<u8>>> {
        if !self.can_fit(length_charge::<u8>(len)?)? {
            return Ok(None);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .context("snapshot buffer readback reservation failed")?;
        if !self.charge(vector_charge(&bytes)?)? {
            return Ok(None);
        }
        bytes.resize(len, 0);
        source.read_bytes(0, &mut bytes)?;
        Ok(Some(bytes))
    }
}

impl SessionSnapshot {
    pub fn frontier(&self) -> usize {
        self.frontier
    }

    pub fn consumed_tokens(&self) -> &[u32] {
        &self.consumed_tokens
    }

    pub fn last_logits_bf16(&self) -> &[u16] {
        &self.last_logits_bf16
    }

    /// Host allocation capacities retained by this snapshot, including Vec
    /// headers. Metal uploads, allocator rounding and Rc allocation overhead
    /// are not included; admission must also budget a live restore candidate.
    pub fn retained_capacity_bytes(&self) -> Result<u64> {
        let mut total = u64::try_from(size_of::<Self>())?;
        total = add(total, vector_charge(&self.consumed_tokens)?)?;
        total = add(total, vector_charge(&self.last_logits_bf16)?)?;
        total = add(total, vector_charge(&self.producer_spans)?)?;
        total = add(total, vector_charge(&self.ple.conv)?)?;
        total = add(total, vector_charge(&self.layers)?)?;
        for layer in &self.layers {
            match layer {
                SavedLayer::Gdn(state) => {
                    total = add(total, vector_charge(&state.conv_bf16)?)?;
                    total = add(total, vector_charge(&state.recurrent_f32)?)?;
                }
                SavedLayer::Qsa(state) => {
                    total = add(total, vector_charge(&state.compressed_index_bf16)?)?;
                    total = add(total, vector_charge(&state.pending_index_bf16)?)?;
                    for values in [
                        &state.kv.records,
                        &state.kv.sink_keys,
                        &state.kv.sink_values,
                        &state.kv.tail_keys,
                        &state.kv.tail_values,
                    ] {
                        total = add(total, vector_charge(values)?)?;
                    }
                }
            }
        }
        Ok(total)
    }

    fn validate_metadata(
        &self,
        owner: &Rc<()>,
        config: &QwenConfig,
        max_context: usize,
        prefill_capacity: usize,
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, owner),
            "snapshot belongs to another engine"
        );
        ensure!(
            self.producer_version == PRODUCER_VERSION
                && self.max_context == max_context
                && self.eos_token_id == config.eos_token_id,
            "snapshot producer or engine geometry differs"
        );
        // These are scratch capacities at capture and restore, not state
        // geometry. Committed producer spans preserve historical row widths.
        crate::prefill_capacity::validate_capacity(self.prefill_capacity)?;
        crate::prefill_capacity::validate_capacity(prefill_capacity)?;
        ensure!(
            self.frontier > 0 && self.frontier <= max_context,
            "snapshot frontier is empty or exceeds context limit"
        );
        ensure!(
            self.consumed_tokens.len() == self.frontier
                && self
                    .consumed_tokens
                    .iter()
                    .all(|&token| token < VOCAB as u32),
            "snapshot consumed tokens differ from frontier or vocabulary"
        );
        ensure!(
            self.last_logits_bf16.len() == VOCAB
                && self
                    .last_logits_bf16
                    .iter()
                    .all(|&value| bf16::from_bits(value).is_finite()),
            "snapshot last-row logits have invalid shape or values"
        );
        let produced = self.producer_spans.iter().try_fold(0usize, |total, span| {
            ensure!(span.width() > 0, "empty snapshot producer span");
            total
                .checked_add(span.width())
                .context("snapshot producer span overflow")
        })?;
        ensure!(
            produced == self.frontier,
            "snapshot producer spans differ from frontier"
        );
        self.ple
            .validate_committed(&self.consumed_tokens, self.eos_token_id)?;
        ensure!(
            self.layers.len() == config.layers.len(),
            "snapshot layer count differs from engine"
        );
        for (expected, saved) in config.layers.iter().zip(&self.layers) {
            match (expected, saved) {
                (MixerKind::Gdn, SavedLayer::Gdn(state)) => ensure!(
                    state.frontier == self.frontier as u64,
                    "snapshot GDN frontier differs"
                ),
                (MixerKind::Qsa, SavedLayer::Qsa(state)) => ensure!(
                    state.index_frontier == self.frontier as u64
                        && state.kv.frontier == self.frontier,
                    "snapshot QSA frontiers differ"
                ),
                _ => anyhow::bail!("snapshot layer type differs from engine"),
            }
        }
        Ok(())
    }
}

impl Engine {
    /// Copy a nonempty committed session without capturing expert-pool state.
    pub fn snapshot(&self, session: &Session) -> Result<SessionSnapshot> {
        autoreleasepool(|_| {
            self.snapshot_inner(session, u64::MAX)?
                .context("snapshot exceeds retained byte address space")
        })
    }

    pub(super) fn validate_snapshot_source(&self, session: &Session) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.identity, &session.owner),
            "session belongs to another engine"
        );
        session.validate_committed()?;
        ensure!(session.frontier > 0, "cannot snapshot an empty session");
        ensure!(
            session.frontier <= self.max_context
                && session.layers.len() == self.config.layers.len(),
            "session geometry differs from engine"
        );
        Ok(())
    }

    /// Minimum payload charge without copying any state. Vec allocator spare
    /// capacity may make the final retained charge larger.
    pub(super) fn snapshot_minimum_retained_bytes(&self, session: &Session) -> Result<u64> {
        self.validate_snapshot_source(session)?;
        let mut total = u64::try_from(size_of::<SessionSnapshot>())?;
        total = add(total, length_charge::<u32>(session.consumed_tokens.len())?)?;
        total = add(total, length_charge::<u16>(VOCAB)?)?;
        total = add(
            total,
            length_charge::<ProducerSpan>(session.producer_spans.len())?,
        )?;
        total = add(total, u64::try_from(session.ple.conv_byte_len())?)?;
        total = add(total, length_charge::<SavedLayer>(session.layers.len())?)?;
        for (expected, layer) in self.config.layers.iter().zip(&session.layers) {
            match (expected, layer) {
                (MixerKind::Gdn, LayerState::Gdn(state)) => {
                    let (conv, recurrent) = state.recurrent()?.buffers()?;
                    total = add(total, u64::try_from(conv.byte_len())?)?;
                    total = add(total, u64::try_from(recurrent.byte_len())?)?;
                }
                (MixerKind::Qsa, LayerState::Qsa(state)) => {
                    let (compressed, pending) = state.index.logical_parts_byte_lengths()?;
                    total = add(total, u64::try_from(compressed)?)?;
                    total = add(total, u64::try_from(pending)?)?;
                    for (_, len) in state.kv.buffers()? {
                        total = add(total, u64::try_from(len)?)?;
                    }
                }
                _ => anyhow::bail!("session layer type differs from engine"),
            }
        }
        Ok(total)
    }

    /// Decline a snapshot before or during deep copying when its retained
    /// capacity would exceed `limit`. A single allocator-rounded component
    /// may briefly exceed the remaining allowance before it is discarded.
    pub(super) fn snapshot_with_budget(
        &self,
        session: &Session,
        limit: u64,
    ) -> Result<Option<SessionSnapshot>> {
        autoreleasepool(|_| self.snapshot_inner(session, limit))
    }

    fn snapshot_inner(&self, session: &Session, limit: u64) -> Result<Option<SessionSnapshot>> {
        if self.snapshot_minimum_retained_bytes(session)? > limit {
            return Ok(None);
        }
        let Some(mut budget) = SnapshotBudget::new(limit)? else {
            return Ok(None);
        };
        let Some(consumed_tokens) = budget.copy(&session.consumed_tokens)? else {
            return Ok(None);
        };
        let Some(last_logits_bf16) = budget.copy(session.last_logits_bf16()?)? else {
            return Ok(None);
        };
        let Some(producer_spans) = budget.copy(&session.producer_spans)? else {
            return Ok(None);
        };
        let ple = session.ple.to_host()?;
        if !budget.charge(vector_charge(&ple.conv)?)? {
            return Ok(None);
        }
        if !budget.can_fit(length_charge::<SavedLayer>(session.layers.len())?)? {
            return Ok(None);
        }
        let mut layers = Vec::with_capacity(session.layers.len());
        if !budget.charge(vector_charge(&layers)?)? {
            return Ok(None);
        }
        for (expected, layer) in self.config.layers.iter().zip(&session.layers) {
            layers.push(match (expected, layer) {
                (MixerKind::Gdn, LayerState::Gdn(state)) => {
                    let (conv, recurrent) = state.recurrent()?.buffers()?;
                    let minimum = add(
                        u64::try_from(conv.byte_len())?,
                        u64::try_from(recurrent.byte_len())?,
                    )?;
                    if !budget.can_fit(minimum)? {
                        return Ok(None);
                    }
                    let (conv_bf16, recurrent_f32) = state.snapshot()?;
                    if !budget.charge(vector_charge(&conv_bf16)?)?
                        || !budget.charge(vector_charge(&recurrent_f32)?)?
                    {
                        return Ok(None);
                    }
                    SavedLayer::Gdn(SavedGdn {
                        frontier: state.frontier()?,
                        conv_bf16,
                        recurrent_f32,
                    })
                }
                (MixerKind::Qsa, LayerState::Qsa(state)) => {
                    let (compressed_bytes, pending_bytes) =
                        state.index.logical_parts_byte_lengths()?;
                    if !budget.can_fit(add(
                        u64::try_from(compressed_bytes)?,
                        u64::try_from(pending_bytes)?,
                    )?)? {
                        return Ok(None);
                    }
                    let (index_frontier, compressed_index_bf16, pending_index_bf16) =
                        state.index.export_host()?.into_parts();
                    if !budget.charge(vector_charge(&compressed_index_bf16)?)?
                        || !budget.charge(vector_charge(&pending_index_bf16)?)?
                    {
                        return Ok(None);
                    }
                    let [
                        (records, records_len),
                        (sink_keys, sink_len),
                        (sink_values, _),
                        (tail_keys, tail_len),
                        (tail_values, _),
                    ] = state.kv.buffers()?;
                    let Some(records) = budget.read_buffer(records, records_len)? else {
                        return Ok(None);
                    };
                    let Some(sink_keys) = budget.read_buffer(sink_keys, sink_len)? else {
                        return Ok(None);
                    };
                    let Some(sink_values) = budget.read_buffer(sink_values, sink_len)? else {
                        return Ok(None);
                    };
                    let Some(tail_keys) = budget.read_buffer(tail_keys, tail_len)? else {
                        return Ok(None);
                    };
                    let Some(tail_values) = budget.read_buffer(tail_values, tail_len)? else {
                        return Ok(None);
                    };
                    SavedLayer::Qsa(SavedQsa {
                        index_frontier,
                        compressed_index_bf16,
                        pending_index_bf16,
                        kv: SavedKvarn {
                            frontier: state.kv.frontier()?,
                            records,
                            sink_keys,
                            sink_values,
                            tail_keys,
                            tail_values,
                        },
                    })
                }
                _ => anyhow::bail!("session layer type differs from engine"),
            });
        }
        let snapshot = SessionSnapshot {
            owner: self.cache_identity.clone(),
            cache_origin: session.cache_origin.clone(),
            producer_version: PRODUCER_VERSION,
            prefill_capacity: self.prefill_chunk_size(),
            max_context: self.max_context,
            eos_token_id: self.config.eos_token_id,
            frontier: session.frontier,
            consumed_tokens,
            last_logits_bf16,
            producer_spans,
            ple,
            layers,
        };
        snapshot.validate_metadata(
            &self.cache_identity,
            &self.config,
            self.max_context,
            self.prefill_chunk_size(),
        )?;
        ensure!(
            snapshot.retained_capacity_bytes()? == budget.charged,
            "snapshot retained charge differs from admission accounting"
        );
        Ok(Some(snapshot))
    }

    /// Restore an independently mutable candidate on this same Engine. The
    /// Engine's current expert pools are neither inspected nor rewound.
    pub fn restore(&self, snapshot: &SessionSnapshot) -> Result<Session> {
        autoreleasepool(|_| self.restore_inner(snapshot))
    }

    fn restore_inner(&self, snapshot: &SessionSnapshot) -> Result<Session> {
        snapshot.validate_metadata(
            &self.cache_identity,
            &self.config,
            self.max_context,
            self.prefill_chunk_size(),
        )?;
        let mut layers = Vec::with_capacity(snapshot.layers.len());
        for saved in &snapshot.layers {
            layers.push(match saved {
                SavedLayer::Gdn(state) => LayerState::Gdn(GdnLayerState::from_bytes(
                    &self.metal,
                    &state.conv_bf16,
                    &state.recurrent_f32,
                    state.frontier,
                )?),
                SavedLayer::Qsa(state) => LayerState::Qsa(QsaState {
                    index: DeviceQsaIndexState::from_parts(
                        &self.metal,
                        state.index_frontier,
                        &state.compressed_index_bf16,
                        &state.pending_index_bf16,
                        self.max_context,
                    )?,
                    kv: DeviceKvarnState::restore(
                        &self.metal,
                        self.max_context,
                        state.kv.frontier,
                        &state.kv.records,
                        &state.kv.sink_keys,
                        &state.kv.sink_values,
                        &state.kv.tail_keys,
                        &state.kv.tail_values,
                    )?,
                }),
            });
        }
        let logits_device = self.metal.shared_buffer(VOCAB * 2)?;
        logits_device.write_bytes(0, &bytes(&snapshot.last_logits_bf16))?;
        let last_logits_bf16 = OnceCell::new();
        let _ = last_logits_bf16.set(snapshot.last_logits_bf16.clone());
        Ok(Session {
            owner: self.identity.clone(),
            cache_origin: snapshot.cache_origin.clone(),
            frontier: snapshot.frontier,
            valid: true,
            layers,
            ple: DevicePleState::from_host(&self.metal, &snapshot.ple)?,
            eos_token_id: snapshot.eos_token_id,
            consumed_tokens: snapshot.consumed_tokens.clone(),
            logits_device,
            last_logits_bf16,
            producer_spans: snapshot.producer_spans.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::RoutingMode;

    fn fixture() -> (Rc<()>, QwenConfig, SessionSnapshot) {
        let owner = Rc::new(());
        let config = QwenConfig {
            layers: Vec::new(),
            ple_layer_index: 1,
            eps: 1e-6,
            eos_token_id: 248_044,
            max_positions: 16,
        };
        let ple = PleState {
            offset: 2,
            context: [7, 8],
            ..PleState::default()
        };
        let snapshot = SessionSnapshot {
            owner: owner.clone(),
            cache_origin: owner.clone(),
            producer_version: PRODUCER_VERSION,
            prefill_capacity: 512,
            max_context: 16,
            eos_token_id: config.eos_token_id,
            frontier: 2,
            consumed_tokens: vec![7, 8],
            last_logits_bf16: vec![0; VOCAB],
            producer_spans: vec![ProducerSpan::Steps {
                mode: RoutingMode::Unbiased,
                count: 2,
            }],
            ple,
            layers: Vec::new(),
        };
        (owner, config, snapshot)
    }

    #[test]
    fn metadata_allows_supported_workspace_changes_but_rejects_invalid_widths() {
        let (owner, config, mut snapshot) = fixture();
        for captured in [512, 2048, 4096] {
            snapshot.prefill_capacity = captured;
            for current in [512, 2048, 4096] {
                snapshot
                    .validate_metadata(&owner, &config, 16, current)
                    .unwrap();
            }
        }
        for invalid in [0, 513, 4097] {
            assert!(
                snapshot
                    .validate_metadata(&owner, &config, 16, invalid)
                    .is_err()
            );
            snapshot.prefill_capacity = invalid;
            assert!(
                snapshot
                    .validate_metadata(&owner, &config, 16, 512)
                    .is_err()
            );
            snapshot.prefill_capacity = 512;
        }
    }

    #[test]
    fn metadata_rejects_another_engine_identity() {
        let (owner, config, snapshot) = fixture();
        snapshot
            .validate_metadata(&owner, &config, 16, 512)
            .unwrap();
        let changed_identity = Rc::new(());
        assert!(
            snapshot
                .validate_metadata(&changed_identity, &config, 16, 512)
                .is_err()
        );
    }

    #[test]
    fn metadata_allows_continuation_copy_without_relabeling_cache_origin() {
        let (origin, config, mut snapshot) = fixture();
        let current_owner = Rc::new(());
        snapshot.owner = current_owner.clone();
        snapshot
            .validate_metadata(&current_owner, &config, 16, 512)
            .unwrap();
        assert!(Rc::ptr_eq(&snapshot.cache_origin, &origin));
        assert!(!Rc::ptr_eq(&snapshot.cache_origin, &current_owner));
    }

    #[test]
    fn metadata_rejects_other_numerical_producers() {
        let (owner, config, mut snapshot) = fixture();
        snapshot
            .validate_metadata(&owner, &config, 16, 512)
            .unwrap();
        for producer in [0, 1, PRODUCER_VERSION - 1, PRODUCER_VERSION + 1] {
            snapshot.producer_version = producer;
            assert!(
                snapshot
                    .validate_metadata(&owner, &config, 16, 512)
                    .is_err()
            );
        }
    }

    #[test]
    fn metadata_rejects_wrong_owner_frontier_and_provenance() {
        let (owner, config, mut snapshot) = fixture();
        snapshot
            .validate_metadata(&owner, &config, 16, 512)
            .unwrap();
        assert!(snapshot.retained_capacity_bytes().unwrap() >= (VOCAB * 2) as u64);
        assert!(
            snapshot
                .validate_metadata(&Rc::new(()), &config, 16, 512)
                .is_err()
        );
        assert!(snapshot.validate_metadata(&owner, &config, 1, 512).is_err());
        snapshot.frontier = 0;
        assert!(
            snapshot
                .validate_metadata(&owner, &config, 16, 512)
                .is_err()
        );
        snapshot.frontier = 2;
        snapshot.consumed_tokens.pop();
        assert!(
            snapshot
                .validate_metadata(&owner, &config, 16, 512)
                .is_err()
        );
        snapshot.consumed_tokens.push(8);
        snapshot.producer_spans[0] = ProducerSpan::Steps {
            mode: RoutingMode::Unbiased,
            count: 1,
        };
        assert!(
            snapshot
                .validate_metadata(&owner, &config, 16, 512)
                .is_err()
        );
    }

    #[test]
    fn metadata_rejects_nonfinite_logits_bad_ple_and_layer_order() {
        let (owner, mut config, mut snapshot) = fixture();
        snapshot.last_logits_bf16[0] = 0x7fc0;
        assert!(
            snapshot
                .validate_metadata(&owner, &config, 16, 512)
                .is_err()
        );
        snapshot.last_logits_bf16[0] = 0;
        snapshot.ple.context[1] = 9;
        assert!(
            snapshot
                .validate_metadata(&owner, &config, 16, 512)
                .is_err()
        );
        snapshot.ple.context[1] = 8;
        config.layers.push(MixerKind::Gdn);
        assert!(
            snapshot
                .validate_metadata(&owner, &config, 16, 512)
                .is_err()
        );
        snapshot.layers.push(SavedLayer::Qsa(SavedQsa {
            index_frontier: 2,
            compressed_index_bf16: Vec::new(),
            pending_index_bf16: Vec::new(),
            kv: SavedKvarn {
                frontier: 2,
                records: Vec::new(),
                sink_keys: Vec::new(),
                sink_values: Vec::new(),
                tail_keys: Vec::new(),
                tail_values: Vec::new(),
            },
        }));
        assert!(
            snapshot
                .validate_metadata(&owner, &config, 16, 512)
                .is_err()
        );
    }

    #[test]
    fn committed_tokens_and_compact_producer_spans_change_only_on_success() {
        let (owner, config, _) = fixture();
        let metal = crate::metal::Metal::new().unwrap();
        let mut session = Session {
            cache_origin: owner.clone(),
            owner,
            frontier: 0,
            valid: false,
            layers: Vec::new(),
            ple: DevicePleState::zero(&metal).unwrap(),
            eos_token_id: config.eos_token_id,
            consumed_tokens: Vec::new(),
            logits_device: metal.shared_buffer(VOCAB * 2).unwrap(),
            last_logits_bf16: OnceCell::new(),
            producer_spans: Vec::new(),
        };
        let logits = vec![0; VOCAB];
        for (token, mode, next) in [
            (7, RoutingMode::Unbiased, 1),
            (8, RoutingMode::Unbiased, 2),
            (9, RoutingMode::CachePrior, 3),
        ] {
            session.valid = false;
            session
                .commit_output(
                    &[token],
                    &logits,
                    ProducerSpan::Steps { mode, count: 1 },
                    next,
                )
                .unwrap();
        }
        session.valid = false;
        session
            .commit_output(&[10, 11], &logits, ProducerSpan::Prefill { width: 2 }, 5)
            .unwrap();
        assert_eq!(session.consumed_tokens, [7, 8, 9, 10, 11]);
        assert_eq!(
            session.producer_spans,
            [
                ProducerSpan::Steps {
                    mode: RoutingMode::Unbiased,
                    count: 2
                },
                ProducerSpan::Steps {
                    mode: RoutingMode::CachePrior,
                    count: 1
                },
                ProducerSpan::Prefill { width: 2 }
            ]
        );
        session.valid = false;
        assert!(
            session
                .commit_output(
                    &[12],
                    &[0],
                    ProducerSpan::Steps {
                        mode: RoutingMode::Unbiased,
                        count: 1,
                    },
                    6,
                )
                .is_err()
        );
        assert_eq!(session.consumed_tokens, [7, 8, 9, 10, 11]);
        assert_eq!(session.frontier, 5);
        assert!(!session.valid);
    }

    #[test]
    fn session_validation_rejects_unpublished_and_mixed_metadata() {
        let (owner, config, snapshot) = fixture();
        let metal = crate::metal::Metal::new().unwrap();
        let logits_device = metal.shared_buffer(VOCAB * 2).unwrap();
        logits_device
            .write_bytes(0, &bytes(&snapshot.last_logits_bf16))
            .unwrap();
        let last_logits_bf16 = OnceCell::new();
        let _ = last_logits_bf16.set(snapshot.last_logits_bf16.clone());
        let mut session = Session {
            cache_origin: snapshot.cache_origin.clone(),
            owner,
            frontier: snapshot.frontier,
            valid: true,
            layers: Vec::new(),
            ple: DevicePleState::from_host(&metal, &snapshot.ple).unwrap(),
            eos_token_id: config.eos_token_id,
            consumed_tokens: snapshot.consumed_tokens.clone(),
            logits_device,
            last_logits_bf16,
            producer_spans: snapshot.producer_spans.clone(),
        };
        session.validate_committed().unwrap();
        session.valid = false;
        assert!(session.validate_committed().is_err());
        session.valid = true;
        session.consumed_tokens.pop();
        assert!(session.validate_committed().is_err());
    }

    #[test]
    fn bounded_copy_rejects_insufficient_space_before_large_clone() {
        let header = u64::try_from(size_of::<SessionSnapshot>()).unwrap();
        assert!(SnapshotBudget::new(header - 1).unwrap().is_none());
        let mut budget = SnapshotBudget::new(header + 4).unwrap().unwrap();
        assert_eq!(
            budget.copy(&[1u8, 2, 3, 4]).unwrap(),
            Some(vec![1, 2, 3, 4])
        );
        assert_eq!(budget.charged, header + 4);
        assert_eq!(budget.copy(&[5u8]).unwrap(), None);
        assert_eq!(budget.charged, header + 4);
    }
}

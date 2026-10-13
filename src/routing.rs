//! Full-512, top-ten Qwen routing with optional cache residency preference.
//!
//! Slot maps are immutable selection hints. A value below 512 means that the
//! projection was resident in the published snapshot; 512 means absent. The
//! expert loader still resolves and ensures every selected projection.

use anyhow::{Context, Result, ensure};
use half::bf16;

pub const EXPERTS: usize = 512;
pub const TOP_K: usize = 10;
pub const MISSING_SLOT: u32 = 512;
pub const METAL_SOURCE: &str = include_str!("../kernels/metal/routing.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutingMode {
    /// Decode with the factor-two, top-two-protected cache prior.
    CachePrior,
    /// Original routing for prefill and exact-path requests, including one row.
    Unbiased,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutingGeometry {
    rows: usize,
}

impl RoutingGeometry {
    pub fn new(rows: usize) -> Result<Self> {
        ensure!(rows > 0, "routing requires at least one row");
        ensure!(
            rows <= u32::MAX as usize / 128,
            "routing rows exceed the Metal grid limit"
        );
        rows.checked_mul(EXPERTS * 2)
            .context("routing logits size overflow")?;
        rows.checked_mul(TOP_K * 4)
            .context("routing output size overflow")?;
        Ok(Self { rows })
    }

    pub fn from_logits(logits_bf16: &[u16]) -> Result<Self> {
        ensure!(
            !logits_bf16.is_empty() && logits_bf16.len().is_multiple_of(EXPERTS),
            "routing logits must contain complete BF16 rows of 512 experts"
        );
        Self::new(logits_bf16.len() / EXPERTS)
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn logits_bytes(self) -> usize {
        self.rows * EXPERTS * 2
    }

    pub fn ids_bytes(self) -> usize {
        self.rows * TOP_K * 4
    }

    pub fn scores_bytes(self) -> usize {
        self.rows * TOP_K * 2
    }

    pub fn changed_bytes(self) -> usize {
        self.rows * 4
    }

    pub fn grid_threads(self) -> usize {
        self.rows * 128
    }
}

#[derive(Debug)]
pub struct RoutingOutput {
    pub ids: Vec<u32>,
    /// Normalized original FP32 probabilities rounded once to BF16 bits.
    pub scores_bf16: Vec<u16>,
    /// Count of selected IDs absent from the original top ten, per row.
    pub changed: Vec<u32>,
}

pub fn validate_slot_maps(maps: [&[u32]; 3]) -> Result<()> {
    for (projection, map) in ["gate", "up", "down"].into_iter().zip(maps) {
        ensure!(
            map.len() == EXPERTS,
            "{projection} slot map must have 512 entries"
        );
        ensure!(
            map.iter().all(|&slot| slot <= MISSING_SLOT),
            "{projection} slot map contains a slot beyond the missing sentinel"
        );
    }
    Ok(())
}

/// CPU-visible BF16 bit words are validated before upload. Device producers
/// must pass the same validation before their routing results are consumed.
pub fn validate_logits(logits_bf16: &[u16]) -> Result<RoutingGeometry> {
    let geometry = RoutingGeometry::from_logits(logits_bf16)?;
    ensure!(
        logits_bf16
            .iter()
            .all(|&word| bf16::from_bits(word).is_finite()),
        "routing requires finite BF16 logits"
    );
    Ok(geometry)
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::{
        EXPERTS, METAL_SOURCE, RoutingGeometry, RoutingMode, RoutingOutput, TOP_K, validate_logits,
        validate_slot_maps,
    };
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use anyhow::{Context, Result, ensure};

    /// Buffers in the order consumed by the Metal router. All buffers and the
    /// slot-map snapshot must remain unchanged until the batch completes.
    pub struct RoutingBuffers<'a> {
        pub logits_bf16: &'a Buffer,
        pub gate_slots: &'a Buffer,
        pub up_slots: &'a Buffer,
        pub down_slots: &'a Buffer,
        pub ids: &'a Buffer,
        pub scores_bf16: &'a Buffer,
        pub changed: &'a Buffer,
    }

    /// A reusable pipeline with fixed factor-two and top-two protection.
    pub struct CachePriorRouter {
        pipeline: Pipeline,
        factor: Buffer,
        protected: Buffer,
        prior_mode: Buffer,
        unbiased_mode: Buffer,
    }

    fn uploaded(metal: &Metal, bytes: &[u8]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(bytes.len())?;
        buffer.write_bytes(0, bytes)?;
        Ok(buffer)
    }

    fn decode_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    }

    fn decode_u16(bytes: &[u8]) -> Vec<u16> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| u16::from_le_bytes(*word))
            .collect()
    }

    impl CachePriorRouter {
        pub fn new(metal: &Metal) -> Result<Self> {
            Self::with_parameters(metal, 2.0, 2)
        }

        fn with_parameters(metal: &Metal, factor: f32, protected: u32) -> Result<Self> {
            ensure!(
                factor.is_finite() && (1.0..=8.0).contains(&factor),
                "cache factor must be finite and between one and eight"
            );
            ensure!(protected <= 3, "protected routes must be at most three");
            // Preserve MLX's FP32 reduction and division boundaries before
            // rounding contribution weights to BF16. The shader explicitly
            // requests fast::exp where the reference softmax does so.
            let pipeline = metal.compile_msl_strict(METAL_SOURCE, "cache_prior_route_512")?;
            ensure!(
                pipeline.max_threads_per_threadgroup() >= 128,
                "cache-prior routing requires a 128-thread Metal group"
            );
            Ok(Self {
                pipeline,
                factor: uploaded(metal, &factor.to_le_bytes())?,
                protected: uploaded(metal, &protected.to_le_bytes())?,
                prior_mode: uploaded(metal, &0u32.to_le_bytes())?,
                unbiased_mode: uploaded(metal, &1u32.to_le_bytes())?,
            })
        }

        /// Encode independent rows against one immutable three-map snapshot.
        /// `Unbiased` is the required prefill mode even for a single-token chunk.
        ///
        /// # Safety
        ///
        /// Input logits must be initialized BF16 words. Arbitrary BF16 bits
        /// are address-safe: NaN-aware selection and ID tie-breaking keep
        /// chosen IDs within 0..512 and output ranks within 0..10. Callers must
        /// validate completed logits with `validate_logits` before consuming
        /// route outputs or publishing expert demand. Each map must contain
        /// 512 valid u32 slot hints, where 512 denotes absence. Output buffers
        /// must not alias any input or another output. Bound data must remain
        /// unchanged until the batch completes.
        pub unsafe fn encode(
            &self,
            batch: &mut Batch,
            buffers: RoutingBuffers<'_>,
            geometry: RoutingGeometry,
            mode: RoutingMode,
        ) -> Result<()> {
            for (label, actual, needed) in [
                (
                    "logits",
                    buffers.logits_bf16.byte_len(),
                    geometry.logits_bytes(),
                ),
                ("gate slots", buffers.gate_slots.byte_len(), EXPERTS * 4),
                ("up slots", buffers.up_slots.byte_len(), EXPERTS * 4),
                ("down slots", buffers.down_slots.byte_len(), EXPERTS * 4),
                ("output ids", buffers.ids.byte_len(), geometry.ids_bytes()),
                (
                    "output scores",
                    buffers.scores_bf16.byte_len(),
                    geometry.scores_bytes(),
                ),
                (
                    "changed",
                    buffers.changed.byte_len(),
                    geometry.changed_bytes(),
                ),
            ] {
                ensure!(
                    actual >= needed,
                    "{label} buffer needs {needed} bytes; has {actual}"
                );
            }
            let mode_buffer = match mode {
                RoutingMode::CachePrior => &self.prior_mode,
                RoutingMode::Unbiased => &self.unbiased_mode,
            };
            let bindings = [
                buffers.logits_bf16,
                buffers.gate_slots,
                buffers.up_slots,
                buffers.down_slots,
                &self.factor,
                &self.protected,
                mode_buffer,
                buffers.ids,
                buffers.scores_bf16,
                buffers.changed,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: The validated 128-thread geometry covers one 512-value
            // row per group. Every binding has enough bytes for all groups,
            // and the three slot maps each contain 512 u32 entries.
            unsafe { batch.dispatch_1d(&self.pipeline, &bindings, geometry.grid_threads(), 128) }
        }

        /// Convenience path for CPU-visible inputs. `encode` avoids these
        /// uploads when logits and snapshot maps already live on the device.
        pub fn route(
            &self,
            metal: &Metal,
            logits_bf16: &[u16],
            slot_maps: [&[u32]; 3],
            mode: RoutingMode,
        ) -> Result<RoutingOutput> {
            let geometry = validate_logits(logits_bf16)?;
            validate_slot_maps(slot_maps)?;
            let logits = uploaded(
                metal,
                &logits_bf16
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let maps = slot_maps
                .map(|map| {
                    uploaded(
                        metal,
                        &map.iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let ids = metal.shared_buffer(geometry.ids_bytes())?;
            let scores = metal.shared_buffer(geometry.scores_bytes())?;
            let changed = metal.shared_buffer(geometry.changed_bytes())?;
            let mut batch = metal.batch()?;
            // SAFETY: CPU inputs were validated; these uploads and outputs
            // have distinct allocations held until this batch completes.
            unsafe {
                self.encode(
                    &mut batch,
                    RoutingBuffers {
                        logits_bf16: &logits,
                        gate_slots: &maps[0],
                        up_slots: &maps[1],
                        down_slots: &maps[2],
                        ids: &ids,
                        scores_bf16: &scores,
                        changed: &changed,
                    },
                    geometry,
                    mode,
                )?;
            }
            batch
                .submit()
                .wait()
                .context("cache-prior routing command")?;
            Self::read_output(geometry, &ids, &scores, &changed)
        }

        /// Read only after successful producer completion and finite-logit
        /// validation. Buffer reservations reject unfinished command readers.
        pub(crate) fn read_output(
            geometry: RoutingGeometry,
            ids: &Buffer,
            scores: &Buffer,
            changed: &Buffer,
        ) -> Result<RoutingOutput> {
            let mut ids_bytes = vec![0; geometry.ids_bytes()];
            let mut scores_bytes = vec![0; geometry.scores_bytes()];
            let mut changed_bytes = vec![0; geometry.changed_bytes()];
            ids.read_bytes(0, &mut ids_bytes)?;
            scores.read_bytes(0, &mut scores_bytes)?;
            changed.read_bytes(0, &mut changed_bytes)?;
            let result = RoutingOutput {
                ids: decode_u32(&ids_bytes),
                scores_bf16: decode_u16(&scores_bytes),
                changed: decode_u32(&changed_bytes),
            };
            debug_assert_eq!(result.ids.len(), geometry.rows() * TOP_K);
            Ok(result)
        }
    }

    #[cfg(test)]
    pub(super) fn test_router(metal: &Metal, factor: f32) -> Result<CachePriorRouter> {
        CachePriorRouter::with_parameters(metal, factor, 2)
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{CachePriorRouter, RoutingBuffers};

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_oracle(
        row: &[u16],
        maps: [&[u32]; 3],
        factor: f32,
        mode: RoutingMode,
    ) -> ([u32; 10], u32) {
        // The CPU reference uses a full-row FP32 softmax, two lexicographic
        // orders and membership checks across three projection maps,
        // independently of the GPU's groupwise top-k search.
        let x: Vec<f32> = row
            .iter()
            .map(|&bits| bf16::from_bits(bits).to_f32())
            .collect();
        let maximum = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f32> = x
            .iter()
            .map(|&value| {
                let e = (value - maximum).exp();
                if e < f32::MIN_POSITIVE { 0.0 } else { e }
            })
            .collect();
        let sum: f32 = exponentials.iter().sum();
        let probabilities: Vec<f32> = exponentials
            .iter()
            .map(|&value| {
                let p = value / sum;
                if p < f32::MIN_POSITIVE { 0.0 } else { p }
            })
            .collect();
        let mut all: Vec<usize> = (0..EXPERTS).collect();
        // Larger IDs survive an equal-probability cutoff.
        all.sort_unstable_by(|&a, &b| {
            probabilities[b]
                .total_cmp(&probabilities[a])
                .then_with(|| b.cmp(&a))
        });
        let original_set = all[..TOP_K].to_vec();
        let mut original = original_set.clone();
        original.sort_unstable_by(|&a, &b| {
            probabilities[b]
                .total_cmp(&probabilities[a])
                .then_with(|| a.cmp(&b))
        });
        let mut selected = original.clone();
        if mode == RoutingMode::CachePrior
            && factor > 1.0
            && original[2..]
                .iter()
                .any(|&id| maps.iter().any(|map| map[id] == MISSING_SLOT))
        {
            let mut biased: Vec<usize> = (0..EXPERTS).collect();
            let protected = &original[..2];
            biased.sort_unstable_by(|&a, &b| {
                let key = |id: usize| -> f32 {
                    if protected.contains(&id) {
                        f32::INFINITY
                    } else if maps.iter().all(|map| map[id] < MISSING_SLOT) {
                        probabilities[id] * factor
                    } else {
                        probabilities[id]
                    }
                };
                key(b).total_cmp(&key(a)).then_with(|| b.cmp(&a))
            });
            selected = biased[..TOP_K].to_vec();
            selected.sort_unstable_by(|&a, &b| {
                probabilities[b]
                    .total_cmp(&probabilities[a])
                    .then_with(|| a.cmp(&b))
            });
        }
        let changed = selected
            .iter()
            .filter(|id| !original_set.contains(id))
            .count() as u32;
        (
            selected
                .into_iter()
                .map(|id| id as u32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            changed,
        )
    }

    fn missing_maps() -> [Vec<u32>; 3] {
        std::array::from_fn(|_| vec![MISSING_SLOT; EXPERTS])
    }

    #[test]
    fn geometry_and_snapshot_validation() {
        assert!(RoutingGeometry::new(0).is_err());
        assert!(RoutingGeometry::from_logits(&[0; 511]).is_err());
        assert_eq!(RoutingGeometry::new(3).unwrap().scores_bytes(), 60);
        assert!(validate_logits(&vec![0x7f80; EXPERTS]).is_err());
        let maps = missing_maps();
        assert!(validate_slot_maps([&maps[0], &maps[1], &maps[2]]).is_ok());
        let mut invalid = maps;
        invalid[1][17] = 513;
        assert!(validate_slot_maps([&invalid[0], &invalid[1], &invalid[2]]).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn arbitrary_bf16_encoding_has_bounded_unique_ids_and_preserves_guards() -> Result<()> {
        use crate::metal::Metal;
        let metal = Metal::new()?;
        let router = CachePriorRouter::new(&metal)?;
        let upload = |bytes: &[u8]| -> Result<_> {
            let b = metal.shared_buffer(bytes.len())?;
            b.write_bytes(0, bytes)?;
            Ok(b)
        };
        let mut rows = vec![
            vec![0x7fc0; EXPERTS],
            vec![0x7f81; EXPERTS],
            vec![0xffc1; EXPERTS],
            vec![0x7f80; EXPERTS],
            vec![0xff80; EXPERTS],
        ];
        rows.push(
            (0..EXPERTS)
                .map(|i| [0x7fc0, 0x7f81, 0xffc1, 0x7f80, 0xff80, 0x3f80, 0, 0x8000][i % 8])
                .collect(),
        );
        rows.push(
            (0..EXPERTS)
                .map(|i| if i % 2 == 0 { 0 } else { 0x8000 })
                .collect(),
        );
        rows.push(
            (0..EXPERTS)
                .map(|i| bf16::from_f32((i % 13) as f32 / 8.).to_bits())
                .collect(),
        );
        let logits = upload(
            &rows
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let geometry = RoutingGeometry::new(rows.len())?;
        for resident in [false, true] {
            let map: Vec<u32> = (0..EXPERTS)
                .map(|id| {
                    if resident && id % 3 == 0 {
                        (id % 12) as u32
                    } else {
                        MISSING_SLOT
                    }
                })
                .collect();
            let map_buffer = upload(&map.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
            for mode in [RoutingMode::Unbiased, RoutingMode::CachePrior] {
                let sizes = [
                    geometry.ids_bytes(),
                    geometry.scores_bytes(),
                    geometry.changed_bytes(),
                ];
                let outputs = sizes.map(|n| upload(&vec![0x5a; n + 32]).unwrap());
                let mut batch = metal.batch()?;
                // SAFETY: Initialized arbitrary BF16 inputs are address-safe;
                // maps are valid and outputs have disjoint trailing guards.
                unsafe {
                    router.encode(
                        &mut batch,
                        RoutingBuffers {
                            logits_bf16: &logits,
                            gate_slots: &map_buffer,
                            up_slots: &map_buffer,
                            down_slots: &map_buffer,
                            ids: &outputs[0],
                            scores_bf16: &outputs[1],
                            changed: &outputs[2],
                        },
                        geometry,
                        mode,
                    )?;
                }
                assert!(
                    CachePriorRouter::read_output(geometry, &outputs[0], &outputs[1], &outputs[2])
                        .is_err()
                );
                let flight = batch.submit();
                assert!(
                    CachePriorRouter::read_output(geometry, &outputs[0], &outputs[1], &outputs[2])
                        .is_err()
                );
                flight.wait()?;
                let mut readback = Vec::new();
                for (buffer, size) in outputs.iter().zip(sizes) {
                    let mut bytes = vec![0; size + 32];
                    buffer.read_bytes(0, &mut bytes)?;
                    assert_eq!(&bytes[size..], &[0x5a; 32]);
                    bytes.truncate(size);
                    readback.push(bytes);
                }
                let ids: Vec<u32> = readback[0]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_le_bytes(*v))
                    .collect();
                for (index, row) in rows.iter().enumerate() {
                    let selected = &ids[index * TOP_K..][..TOP_K];
                    assert!(selected.iter().all(|&id| id < EXPERTS as u32));
                    assert_eq!(
                        selected
                            .iter()
                            .collect::<std::collections::BTreeSet<_>>()
                            .len(),
                        TOP_K
                    );
                    if index < 6 {
                        assert!(validate_logits(row).is_err());
                        assert!(router.route(&metal, row, [&map, &map, &map], mode).is_err());
                    } else {
                        let reference = router.route(&metal, row, [&map, &map, &map], mode)?;
                        assert_eq!(selected, reference.ids);
                        let scores: Vec<u16> = readback[1][index * TOP_K * 2..][..TOP_K * 2]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|v| u16::from_le_bytes(*v))
                            .collect();
                        assert_eq!(scores, reference.scores_bf16);
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn normalization_preserves_bf16_midpoint_side() -> anyhow::Result<()> {
        let mut seed = 0xf408_8878u32;
        let logits: Vec<_> = (0..EXPERTS)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = ((seed >> 8) as i32 - 8_388_608) as f32 / 2_097_152.0;
                bf16::from_f32(value).to_bits()
            })
            .collect();
        // This synthetic row crosses a BF16 midpoint under relaxed FP32
        // normalization. Expected bits follow the reference's FP32 operators.
        let ids = [429, 413, 315, 468, 473, 130, 138, 214, 253, 36];
        let scores = [
            0x3dda, 0x3dd3, 0x3dd0, 0x3dd0, 0x3dd0, 0x3dcd, 0x3dc7, 0x3dc7, 0x3dc7, 0x3dc1,
        ];
        let metal = crate::metal::Metal::new()?;
        let router = CachePriorRouter::new(&metal)?;
        let maps = missing_maps();
        for mode in [RoutingMode::Unbiased, RoutingMode::CachePrior] {
            let actual = router.route(&metal, &logits, [&maps[0], &maps[1], &maps[2]], mode)?;
            assert_eq!(actual.ids, ids);
            assert_eq!(actual.scores_bf16, scores);
            assert_eq!(actual.changed, [0]);
        }
        Ok(())
    }

    #[test]
    #[allow(
        clippy::needless_range_loop,
        reason = "One expert ID indexes three independent maps"
    )]
    fn independent_oracle_has_both_tie_orders_and_membership_intersection() {
        let row = [0u16; EXPERTS];
        let mut maps = missing_maps();
        for id in 0..21 {
            maps[0][id] = (20 - id) as u32;
            maps[1][id] = id as u32;
            maps[2][id] = ((id + 7) % 21) as u32;
        }
        maps[2][20] = MISSING_SLOT;
        let refs = [&maps[0][..], &maps[1][..], &maps[2][..]];
        let (ids, changed) = cpu_oracle(&row, refs, 2.0, RoutingMode::CachePrior);
        assert_eq!(ids, [12, 13, 14, 15, 16, 17, 18, 19, 502, 503]);
        assert_eq!(changed, 8);
        let (plain, zero) = cpu_oracle(&row, refs, 2.0, RoutingMode::Unbiased);
        assert_eq!(plain, [502, 503, 504, 505, 506, 507, 508, 509, 510, 511]);
        assert_eq!(zero, 0);
    }

    #[cfg(target_os = "macos")]
    fn check_case(
        router: &CachePriorRouter,
        metal: &crate::metal::Metal,
        logits: &[u16],
        maps: [&[u32]; 3],
        mode: RoutingMode,
        factor: f32,
    ) {
        let result = router.route(metal, logits, maps, mode).unwrap();
        assert_eq!(result.ids.len(), logits.len() / EXPERTS * TOP_K);
        for (row_index, row) in logits.as_chunks::<EXPERTS>().0.iter().enumerate() {
            let (expected, changed) = cpu_oracle(row, maps, factor, mode);
            let actual = &result.ids[row_index * TOP_K..][..TOP_K];
            assert_eq!(actual, expected, "route mismatch at row {row_index}");
            assert_eq!(
                result.changed[row_index], changed,
                "change count at row {row_index}"
            );
            assert_eq!(
                actual
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                TOP_K
            );
            let selected: Vec<f64> = actual
                .iter()
                .map(|&id| bf16::from_bits(row[id as usize]).to_f32() as f64)
                .collect();
            let max = selected.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exponentials: Vec<f64> = selected.iter().map(|&x| (x - max).exp()).collect();
            let total: f64 = exponentials.iter().sum();
            for (column, &word) in result.scores_bf16[row_index * TOP_K..][..TOP_K]
                .iter()
                .enumerate()
            {
                let exact = exponentials[column] / total;
                let actual = bf16::from_bits(word).to_f32() as f64;
                let below = bf16::from_bits(word.saturating_sub(1)).to_f32() as f64;
                let above = bf16::from_bits(word + 1).to_f32() as f64;
                let quantum = (actual - below).max(above - actual);
                assert!(
                    (actual - exact).abs() / quantum < 0.501,
                    "weight row {row_index} column {column}: BF16 word {word:#06x}, exact {exact}, error {} ulp",
                    (actual - exact).abs() / quantum
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[allow(
        clippy::needless_range_loop,
        reason = "One expert ID indexes three independent maps"
    )]
    fn metal_matches_independent_selector_on_ties_scales_and_stale_hints() {
        use crate::metal::Metal;

        let metal = Metal::new().unwrap();
        let router = CachePriorRouter::new(&metal).unwrap();
        let mut state = 0x91e10da5u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut rows = Vec::new();
        for scale in [0.01f32, 0.3, 1.0, 3.0, 10.0, 30.0] {
            for _ in 0..8 {
                for _ in 0..EXPERTS {
                    let signed = (next() % 20001) as i32 - 10000;
                    rows.push(bf16::from_f32(signed as f32 / 10000.0 * scale).to_bits());
                }
            }
        }
        for _ in 0..8 {
            for _ in 0..EXPERTS {
                rows.push(bf16::from_f32(((next() % 17) as i32 - 8) as f32 / 4.0).to_bits());
            }
        }
        rows.extend([0u16; EXPERTS]);
        rows.extend([bf16::from_f32(1000.0).to_bits(); EXPERTS]);
        let mut spike = [bf16::from_f32(-150.0).to_bits(); EXPERTS];
        spike[17] = bf16::from_f32(1.0).to_bits();
        rows.extend(spike);
        let mut maps = missing_maps();
        for id in 0..223 {
            maps[0][id] = (222 - id) as u32;
            maps[1][id] = ((id * 37) % 223) as u32;
            if id % 3 != 0 {
                maps[2][id] = ((id * 41) % 223) as u32;
            }
        }
        let refs = [&maps[0][..], &maps[1][..], &maps[2][..]];
        check_case(&router, &metal, &rows, refs, RoutingMode::CachePrior, 2.0);
        check_case(&router, &metal, &rows, refs, RoutingMode::Unbiased, 2.0);
        let full: [Vec<u32>; 3] = std::array::from_fn(|j| {
            (0..EXPERTS)
                .map(|id| ((id * (j * 2 + 1) + j) % EXPERTS) as u32)
                .collect()
        });
        check_case(
            &router,
            &metal,
            &rows,
            [&full[0], &full[1], &full[2]],
            RoutingMode::CachePrior,
            2.0,
        );
        check_case(
            &router,
            &metal,
            &rows[..EXPERTS],
            [&maps[0], &maps[1], &maps[2]],
            RoutingMode::Unbiased,
            2.0,
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn factor_one_is_neutral_and_smallest_f32_increase_is_effective() {
        use crate::metal::Metal;

        let metal = Metal::new().unwrap();
        let row = [0u16; EXPERTS];
        let mut maps = missing_maps();
        for map in &mut maps {
            map[0] = 0;
        }
        let refs = [&maps[0][..], &maps[1][..], &maps[2][..]];
        let neutral = super::gpu::test_router(&metal, 1.0).unwrap();
        check_case(&neutral, &metal, &row, refs, RoutingMode::CachePrior, 1.0);
        let minimal = f32::from_bits(1.0f32.to_bits() + 1);
        let biased = super::gpu::test_router(&metal, minimal).unwrap();
        check_case(
            &biased,
            &metal,
            &row,
            refs,
            RoutingMode::CachePrior,
            minimal,
        );
        let changed = biased
            .route(&metal, &row, refs, RoutingMode::CachePrior)
            .unwrap();
        assert_eq!(changed.changed, [1]);
        assert!(changed.ids.contains(&0));
    }
}

use super::*;

#[test]
fn padded_gdn_admission_keeps_full_shapes_and_bounds_useful_rows() {
    for physical in [64, 80, 2048, 4096, 8192] {
        assert_eq!(
            ProjectionRows::gdn(physical, physical).unwrap(),
            ProjectionRows::full(physical).unwrap()
        );
    }
    for useful in [4096, 4097, 6144, 6928, 8191, 8192] {
        let rows = ProjectionRows::gdn(8192, useful).unwrap();
        assert_eq!(rows.physical, 8192);
        assert_eq!(rows.useful, useful);
        assert_eq!(
            rows.bytes(VALUE).unwrap(),
            (8192 * VALUE * 2, useful * VALUE * 2)
        );
    }
    for useful in [0, 1, 378, 4095, 8193, usize::MAX] {
        assert!(ProjectionRows::gdn(8192, useful).is_err());
    }
    for (physical, useful) in [
        (4096, 4095),
        (4096, 2048),
        (8193, 8193),
        (usize::MAX, usize::MAX),
    ] {
        assert!(ProjectionRows::gdn(physical, useful).is_err());
    }
    assert!(
        ProjectionRows::full(8192)
            .unwrap()
            .bytes(usize::MAX)
            .is_err()
    );
    assert!(ProjectionRows::full(8192).unwrap().bytes(0).is_err());
}

#[test]
fn padded_qsa_rows_require_supported_physical_shape() {
    assert_eq!(
        ProjectionRows::qsa(4096, 4096).unwrap(),
        ProjectionRows::gdn(4096, 4096).unwrap()
    );
    for useful in [4096, 4097, 4104, 4105, 6144, 6928, 8191, 8192] {
        assert_eq!(
            ProjectionRows::qsa(8192, useful).unwrap(),
            ProjectionRows::gdn(8192, useful).unwrap()
        );
    }
    for (physical, useful) in [
        (2048, 2048),
        (4096, 4095),
        (4096, 4097),
        (4096, 2048),
        (8192, 4095),
        (8192, 8193),
        (usize::MAX, usize::MAX),
    ] {
        assert!(ProjectionRows::qsa(physical, useful).is_err());
    }
}

#[test]
fn padded_phase_carries_layer_and_extent_without_stale_output() {
    let a = StagedProjection {
        kind: ProjectionKind::Gdn,
        layer: 0,
        rows: ProjectionRows::gdn(8192, 4097).unwrap(),
    };
    let b = StagedProjection {
        kind: ProjectionKind::Gdn,
        layer: 0,
        rows: ProjectionRows::gdn(8192, 6144).unwrap(),
    };
    let c = StagedProjection {
        kind: ProjectionKind::Gdn,
        layer: 1,
        rows: a.rows,
    };
    assert_ne!(Phase::Input(a), Phase::Input(b));
    assert_ne!(Phase::Input(a), Phase::Input(c));
    for phase in [Phase::Idle, Phase::Input(a), Phase::Output(a)] {
        assert!(phase.packed(ProjectionKind::Gdn, 0).is_err());
    }
    assert_eq!(Phase::Packing(a).packed(ProjectionKind::Gdn, 0).unwrap(), a);
    assert!(Phase::Packing(a).packed(ProjectionKind::Gdn, 1).is_err());
    assert_eq!(Phase::Input(a).input(ProjectionKind::Gdn, 0).unwrap(), a);
    assert!(Phase::Input(a).input(ProjectionKind::Gdn, 1).is_err());
    assert!(Phase::Packing(a).input(ProjectionKind::Gdn, 0).is_err());
    assert_eq!(Phase::Output(a).output(ProjectionKind::Gdn, 0).unwrap(), a);
    assert!(Phase::Input(a).output(ProjectionKind::Gdn, 0).is_err());
    assert!(Phase::Idle.output(ProjectionKind::Gdn, 0).is_err());
}

#[test]
fn padded_output_validation_checks_every_padding_word_and_length() {
    let rows = ProjectionRows::gdn(8192, 4097).unwrap();
    let mut values = vec![0_u16; rows.physical * 3];
    for channel in 0..3 {
        values[channel * rows.physical..channel * rows.physical + rows.useful].fill(0x3c00);
        values[(channel + 1) * rows.physical - 1] = 0x8000;
    }
    validate_padding(&values, rows, 3).unwrap();
    assert!(validate_padding(&values[..values.len() - 1], rows, 3).is_err());
    for offset in [rows.useful, rows.physical + rows.useful, values.len() - 1] {
        let saved = values[offset];
        for bits in [1, 0x3c00, 0x7c00, 0x7e00] {
            values[offset] = bits;
            assert!(validate_padding(&values, rows, 3).is_err());
        }
        values[offset] = saved;
    }
    let full = ProjectionRows::full(8192).unwrap();
    validate_padding(&vec![0x3c00; full.physical * 3], full, 3).unwrap();
}

// Simulate the 256-thread, 32x33 transpose, independently checking
// global source/destination formulas and one write per active cell.
#[test]
fn padded_unpack_tails_use_physical_source_and_useful_destination_strides() {
    for (physical, useful, channels) in [
        (8192, 4096, 65),
        (8192, 4097, 65),
        (8192, 6144, 65),
        (8192, 8191, 65),
        (8192, 8192, 65),
        (257, 255, 257),
    ] {
        let mut source = vec![0x7e00_u16; physical * channels];
        for channel in 0..channels {
            for row in 0..useful {
                source[channel * physical + row] = ((channel * 137 + row * 17) % 0x7bff) as u16;
            }
        }
        let mut result = vec![0x7e00; useful * channels + 13];
        let mut writes = vec![0_u8; useful * channels];
        for row_base in (0..useful).step_by(32) {
            for channel_base in (0..channels).step_by(32) {
                let mut tile = [0_u16; 32 * 33];
                for tid in 0..256 {
                    let (x, y) = (tid % 32, tid / 32);
                    for offset in (0..32).step_by(8) {
                        let (row, channel) = (row_base + x, channel_base + y + offset);
                        tile[(y + offset) * 33 + x] = if row < useful && channel < channels {
                            source[channel * physical + row]
                        } else {
                            0
                        };
                    }
                }
                for tid in 0..256 {
                    let (x, y) = (tid % 32, tid / 32);
                    for offset in (0..32).step_by(8) {
                        let (row, channel) = (row_base + y + offset, channel_base + x);
                        if row < useful && channel < channels {
                            let at = row * channels + channel;
                            result[at] = tile[x * 33 + y + offset];
                            writes[at] += 1;
                        }
                    }
                }
            }
        }
        assert!(writes.iter().all(|&n| n == 1));
        assert!(result[useful * channels..].iter().all(|&v| v == 0x7e00));
        for row in 0..useful {
            for channel in 0..channels {
                assert_eq!(
                    result[row * channels + channel],
                    source[channel * physical + row]
                );
            }
        }
    }
}

#[test]
fn padded_pack_zeroes_tail_without_reading_beyond_useful_input() {
    for useful in [4096, 4097, 6144, 6928, 8191, 8192] {
        let physical = 8192;
        let channels = 33;
        let outputs = 65;
        let input: Vec<_> = (0..useful * channels).map(|n| (n % 31337) as u16).collect();
        let mut result = vec![0xffff_u16; channels * (physical + outputs)];
        for row_base in (0..physical).step_by(32) {
            for channel_base in (0..channels).step_by(32) {
                let mut tile = [0_u16; 32 * 33];
                for tid in 0..256 {
                    let (x, y) = (tid % 32, tid / 32);
                    for offset in (0..32).step_by(8) {
                        let (row, channel) = (row_base + y + offset, channel_base + x);
                        if row < physical && channel < channels && row < useful {
                            tile[(y + offset) * 33 + x] = input[row * channels + channel];
                        }
                    }
                }
                for tid in 0..256 {
                    let (x, y) = (tid % 32, tid / 32);
                    for offset in (0..32).step_by(8) {
                        let (row, channel) = (row_base + x, channel_base + y + offset);
                        if row < physical && channel < channels {
                            result[channel * (physical + outputs) + row] =
                                tile[x * 33 + y + offset];
                        }
                    }
                }
            }
        }
        for channel in 0..channels {
            for row in 0..physical {
                let expected = if row < useful {
                    input[row * channels + channel]
                } else {
                    0
                };
                assert_eq!(result[channel * (physical + outputs) + row], expected);
            }
            assert!(
                result[channel * (physical + outputs) + physical
                    ..(channel + 1) * (physical + outputs)]
                    .iter()
                    .all(|&v| v == 0xffff)
            );
        }
    }
}

#[test]
fn padded_producer_scope_cannot_restore_prior_full_only_state() {
    let scope = producer_identity();
    assert!(scope.contains("reuse-padded"));
    assert!(scope.ends_with("v4"));
    assert_ne!(
        producer_identity(),
        "ane-gdn-z-qsa-gate-q6k-bf16-fp16-gpu-q6-dynamic-late-fp32-v1"
    );
    assert_ne!(
        producer_identity(),
        "ane-gdn-z-qsa-gate-q6k-bf16-fp16-gpu-q6-dynamic-padded-gdn-late-fp32-v2"
    );
    for prior in [
        "ane-gdn-z-qsa-gate-q6k-bf16-fp16-gpu-q6-dynamic-padded-gdn-qsa-late-fp32-v3",
        "ane-gdn-z-qsa-gate-shared-front-q6k-bf16-fp16-reuse-padded-v3",
        "ane-gdn-z-q6k-bf16-fp16-gpu-q6-dynamic-padded-gdn-v2",
        "gpu-gdn-z-qsa-gate-shared-front-q6-reuse-padded-gdn-qsa-boundary-v4",
    ] {
        assert_ne!(producer_identity(), prior);
    }
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE, private ANE support and exclusive GPU ownership"]
fn actual_4097_rows_use_96_typed_combined_evaluations_and_keep_committed_frontiers() -> Result<()> {
    use super::super::{Engine, LayerStateView, Session};
    use crate::routing::RoutingMode;
    use half::bf16;

    fn committed(session: &Session, tokens: &[u32]) -> Result<()> {
        assert_eq!(session.frontier()?, tokens.len());
        assert_eq!(session.consumed_tokens()?, tokens);
        assert_eq!(session.ple_state()?.offset, tokens.len() as u64);
        for layer in &session.layers {
            assert_eq!(layer.frontier()?, tokens.len());
        }
        Ok(())
    }
    fn finite(logits: &[u16]) {
        assert_eq!(logits.len(), 248_320);
        assert!(logits.iter().all(|&word| bf16::from_bits(word).is_finite()));
    }

    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    let mut engine = Engine::load_with_prefill_capacity(&package, 512, 8192, 8192)?;
    engine.enable_ane(&package)?;
    let expected: Vec<_> = (0..48).collect();
    assert_eq!(expected.len(), 48);
    let mut session = engine.start_session()?;
    let mut tokens: Vec<u32> = (0..4097).map(|i| 61 + i % 32).collect();
    let output = engine.prefill_chunk(&tokens, &mut session)?;
    finite(&output.logits_bf16);
    assert_eq!(output.frontier, 4097);
    committed(&session, &tokens)?;
    assert_eq!(
        engine.prefill.test_ane_state(),
        Some((expected.as_slice(), false))
    );
    let expected_trace: Vec<_> = (0..48)
        .flat_map(|index| {
            let mut entries = Vec::new();
            if index % 4 != 3 {
                entries.push((ProjectionKind::Gdn, index));
            } else {
                entries.push((ProjectionKind::QsaGate, index));
            }
            entries.push((ProjectionKind::SharedFront, index));
            entries
        })
        .collect();
    assert_eq!(engine.prefill.test_projection_trace(), expected_trace);
    assert_eq!(expected_trace.len(), 96);
    let (mut gdn, mut qsa) = (0, 0);
    session.visit_state(|_, state| {
        match state {
            LayerStateView::Gdn { .. } => gdn += 1,
            LayerStateView::Qsa { index, kv } => {
                assert_eq!(index.frontier(), 4097);
                assert_eq!(kv.frontier(), 4097);
                qsa += 1;
            }
        }
        Ok(())
    })?;
    assert_eq!((gdn, qsa), (36, 12));

    let tail: Vec<u32> = (0..17).map(|i| 93 + i % 16).collect();
    let output = engine.prefill_chunk(&tail, &mut session)?;
    finite(&output.logits_bf16);
    tokens.extend_from_slice(&tail);
    committed(&session, &tokens)?;
    let output = engine.step(7, RoutingMode::Unbiased, &mut session)?;
    finite(&output.logits_bf16);
    tokens.push(7);
    committed(&session, &tokens)?;
    assert_eq!(
        engine.prefill.test_ane_state(),
        Some((expected.as_slice(), false))
    );
    println!(
        "padded_qsa_native_trace={{\"actual_rows\":4097,\"evaluated_layers\":{expected:?},\"qsa_evaluations\":12,\"online_tail_rows\":1,\"final_frontier\":{}}}",
        session.frontier()?
    );
    assert_eq!(engine.prefill.test_projection_trace(), expected_trace);
    println!(
        "shared_front_native_trace={{\"actual_rows\":4097,\"typed_evaluations\":96,\"gdn_evaluations\":36,\"shared_evaluations\":48,\"qsa_evaluations\":12}}"
    );
    drop(session);
    engine.close_ane()
}

use super::*;

#[test]
fn qsa_phase_keeps_actual_odd_rows_in_the_typed_projection() {
    let stage = StagedProjection {
        kind: ProjectionKind::QsaGate,
        layer: 3,
        rows: ProjectionRows::qsa(8192, 4097).unwrap(),
    };
    for completed in [
        Phase::Packing(stage)
            .packed(ProjectionKind::QsaGate, 3)
            .unwrap(),
        Phase::Input(stage)
            .input(ProjectionKind::QsaGate, 3)
            .unwrap(),
        Phase::Output(stage)
            .output(ProjectionKind::QsaGate, 3)
            .unwrap(),
    ] {
        assert_eq!(completed.rows.physical, 8192);
        assert_eq!(completed.rows.useful, 4097);
        assert_eq!(completed.rows.bytes(VALUE).unwrap().1, 4097 * VALUE * 2);
    }
    assert!(
        Phase::Output(stage)
            .output(ProjectionKind::SharedFront, 3)
            .is_err()
    );
    assert!(
        Phase::Output(stage)
            .output(ProjectionKind::QsaGate, 7)
            .is_err()
    );
}

#[test]
fn projection_kind_is_part_of_every_phase_identity() {
    for kind in [
        ProjectionKind::Gdn,
        ProjectionKind::QsaGate,
        ProjectionKind::SharedFront,
    ] {
        let stage = StagedProjection {
            kind,
            layer: 3,
            rows: ProjectionRows::gdn(MAX_PHYSICAL_ROWS, 4096).unwrap(),
        };
        assert_eq!(Phase::Packing(stage).packed(kind, 3).unwrap(), stage);
        assert_eq!(Phase::Input(stage).input(kind, 3).unwrap(), stage);
        assert_eq!(Phase::Output(stage).output(kind, 3).unwrap(), stage);
        for other in [
            ProjectionKind::Gdn,
            ProjectionKind::QsaGate,
            ProjectionKind::SharedFront,
        ] {
            if kind != other {
                assert!(Phase::Packing(stage).packed(other, 3).is_err());
                assert!(Phase::Input(stage).input(other, 3).is_err());
                assert!(Phase::Output(stage).output(other, 3).is_err());
            }
        }
        assert!(Phase::Output(stage).output(kind, 4).is_err());
    }
}

#[test]
fn two_weight_wires_and_padding_cover_each_combined_cell_once() {
    for (physical, useful) in [
        (4096, 4096),
        (MAX_PHYSICAL_ROWS, 4096),
        (MAX_PHYSICAL_ROWS, 4097),
        (MAX_PHYSICAL_ROWS, 6928),
        (MAX_PHYSICAL_ROWS, 8192),
    ] {
        let rows = ProjectionRows::gdn(physical, useful).unwrap();
        assert_eq!(rows.bytes(SHARED).unwrap().1, useful * SHARED * 2);
        for column in 0..VALUE {
            let source = if column < SHARED {
                Some((0, column))
            } else if column < SHARED * 2 {
                Some((1, column - SHARED))
            } else {
                None
            };
            match source {
                Some((bank, row)) => {
                    assert!(bank < 2 && row < SHARED);
                    assert_eq!(bank * SHARED + row, column);
                }
                None => assert!(column >= 1280),
            }
        }
        assert_eq!(physical * SHARED * 2, physical * 1280);
        assert!(producer_identity().contains("shared-front"));
        assert_ne!(
            producer_identity(),
            "ane-gdn-z-qsa-gate-q6k-bf16-fp16-gpu-q6-dynamic-padded-gdn-late-fp32-v2"
        );
    }
}

#[test]
fn split_unpack_tiled_schedule_retains_physical_stride_and_odd_rows() {
    for (physical, useful, width) in [
        (257, 255, 33),
        (4096, 4096, 33),
        (8192, 4097, 33),
        (8192, 6928, 65),
    ] {
        let channels = width * 2;
        let source: Vec<u16> = (0..physical * channels)
            .map(|n| (n * 137 + 17) as u16)
            .collect();
        let mut outputs = [
            vec![0x7e00; useful * width + 13],
            vec![0x7e00; useful * width + 13],
        ];
        let mut writes = [vec![0u8; useful * width], vec![0u8; useful * width]];
        for row_base in (0..useful).step_by(32) {
            for channel_base in (0..channels).step_by(32) {
                let mut tile = [0u16; 32 * 33];
                for tid in 0..256 {
                    let (x, y) = (tid % 32, tid / 32);
                    for offset in (0..32).step_by(8) {
                        let (r, c) = (row_base + x, channel_base + y + offset);
                        tile[(y + offset) * 33 + x] = if r < useful && c < channels {
                            source[c * physical + r]
                        } else {
                            0
                        };
                    }
                }
                for tid in 0..256 {
                    let (x, y) = (tid % 32, tid / 32);
                    for offset in (0..32).step_by(8) {
                        let (r, c) = (row_base + y + offset, channel_base + x);
                        if r < useful && c < channels {
                            let (bank, at) = (c / width, r * width + c % width);
                            outputs[bank][at] = tile[x * 33 + y + offset];
                            writes[bank][at] += 1;
                        }
                    }
                }
            }
        }
        for bank in 0..2 {
            assert!(writes[bank].iter().all(|&v| v == 1));
            assert!(outputs[bank][useful * width..].iter().all(|&v| v == 0x7e00));
            for row in 0..useful {
                for column in 0..width {
                    assert_eq!(
                        outputs[bank][row * width + column],
                        source[(bank * width + column) * physical + row]
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires exclusive Metal ownership; no ANE or package required"]
fn two_separate_q6_weight_bindings_pack_and_direct_split_are_exact_at_odd_rows() -> Result<()> {
    shared_layout_case(8192, 4097)?;
    shared_layout_case(8192, 8192)
}

#[test]
#[ignore = "requires exclusive Metal ownership; no ANE or package required"]
fn two_separate_q6_weight_bindings_pack_and_direct_split_are_exact_at_4096_rows() -> Result<()> {
    shared_layout_case(4096, 4096)
}

fn shared_layout_case(physical: usize, useful: usize) -> Result<()> {
    use half::bf16;
    let metal = Metal::new()?;
    let layout = combined_layout(physical)?;
    let status = metal.shared_buffer(4)?;
    status.write_bytes(0, &0u32.to_le_bytes())?;
    let mut owner = AnePrefill {
        rows: physical,
        program: None,
        packed_input: metal.shared_buffer(layout.input_elements * 2)?,
        channel_output: metal.shared_buffer(physical * VALUE * 2)?,
        status,
        pack: metal.compile_msl_strict(PACK_SOURCE, "q6_combined_pack")?,
        qsa_pack: metal.compile_msl_strict(PACK_SOURCE, "q6_qsa_gate_pack")?,
        shared_layout: (
            metal.compile_msl_strict(PACK_SOURCE, "q6_shared_front_pack")?,
            metal.compile_msl_strict(PADDED_UNPACK_SOURCE, "tail_unpack_shared")?,
        ),
        unpack: metal.compile_msl_strict(PADDED_UNPACK_SOURCE, "tail_unpack_bf16")?,
        host_input: vec![0; layout.input_elements],
        host_output: vec![0; physical * VALUE],
        phase: Phase::Idle,
        poisoned: false,
        evaluated_layers: Vec::new(),
        evaluated_projections: Vec::new(),
    };
    let input: Vec<_> = (0..useful * HIDDEN)
        .map(|n| bf16::from_f32(((n * 17 % 127) as f32 - 63.0) / 64.0).to_bits())
        .collect();
    let hidden = metal.shared_buffer(input.len() * 2)?;
    hidden.write_bytes(0, word_bytes(&input))?;
    let mut weights = Vec::new();
    for multiplier in [1.0, 2.0] {
        let mut bytes = vec![0u8; HIDDEN / 256 * SHARED * 210];
        for block in bytes.as_chunks_mut::<210>().0 {
            block[192..208].fill(1);
            block[208..210].copy_from_slice(&f16::from_f32(multiplier).to_bits().to_le_bytes());
        }
        let buffer = metal.shared_buffer(bytes.len() + 32)?;
        buffer.write_bytes(0, &[0xa5; 32])?;
        buffer.write_bytes(32, &bytes)?;
        weights.push(buffer);
    }
    let mut stream = metal.stream()?;
    let marker = owner.encode_shared_input(
        3,
        useful,
        &mut stream,
        &hidden,
        [
            Binding {
                buffer: &weights[0],
                offset: 32,
            },
            Binding {
                buffer: &weights[1],
                offset: 32,
            },
        ],
    )?;
    owner.read_packed_input(&mut stream, marker)?;
    for channel in 0..HIDDEN {
        for row in 0..physical + VALUE {
            let expected = if row < useful {
                f16::from_f32(bf16::from_bits(input[row * HIDDEN + channel]).to_f32()).to_bits()
            } else if row < physical {
                0
            } else if row < physical + SHARED {
                f16::from_f32(-32.0).to_bits()
            } else if row < physical + SHARED * 2 {
                f16::from_f32(-64.0).to_bits()
            } else {
                0
            };
            assert_eq!(
                owner.host_input[channel * (physical + VALUE) + row],
                expected
            );
        }
    }
    let mut raw = vec![0x7e00; physical * VALUE];
    for channel in 0..SHARED * 2 {
        for row in 0..useful {
            let finite = (channel * 137 + row * 17) % 0xf800;
            raw[channel * physical + row] = if finite < 0x7c00 {
                finite as u16
            } else {
                (finite + 0x400) as u16
            };
        }
    }
    owner.channel_output.write_bytes(0, word_bytes(&raw))?;
    let bytes = useful * SHARED * 2;
    let gate = metal.shared_buffer(bytes + 32)?;
    let up = metal.shared_buffer(bytes + 32)?;
    for buffer in [&gate, &up] {
        buffer.write_bytes(0, &vec![0xa5; bytes + 32])?;
    }
    owner.phase = Phase::Output(StagedProjection {
        kind: ProjectionKind::SharedFront,
        layer: 3,
        rows: ProjectionRows::gdn(physical, useful)?,
    });
    owner.encode_gate(
        ProjectionKind::SharedFront,
        3,
        stream.batch()?,
        &[&gate, &up],
    )?;
    stream.wait()?;
    owner.check_status_after_completion()?;
    for (bank, buffer) in [&gate, &up].into_iter().enumerate() {
        let mut actual = vec![0u16; useful * SHARED + 16];
        buffer.read_bytes(0, word_bytes_mut(&mut actual))?;
        assert!(actual[useful * SHARED..].iter().all(|&v| v == 0xa5a5));
        for row in 0..useful {
            for channel in 0..SHARED {
                assert_eq!(
                    actual[row * SHARED + channel],
                    bf16::from_f32(
                        f16::from_bits(raw[(bank * SHARED + channel) * physical + row]).to_f32()
                    )
                    .to_bits()
                );
            }
        }
    }
    owner.close()
}

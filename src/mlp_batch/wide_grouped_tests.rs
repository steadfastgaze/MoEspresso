fn grouped_input(rows: usize) -> Vec<u8> {
    let mut random = 7u32;
    (0..rows * HIDDEN)
        .flat_map(|_| {
            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bf16::from_f32(((random >> 24) as f32 - 128.0) / 1024.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

fn check_grouped_guards(scratch: &MlpBatchScratch, rows: usize) -> Result<()> {
    for (buffer, width, pairs, bf16_wire) in [
        (&scratch.packed_activation, SHARED, rows * ROUTES, false),
        (&scratch.packed_down, HIDDEN, rows * ROUTES, false),
        (&scratch.base.output, HIDDEN, rows, true),
    ] {
        let bytes = read(buffer)?;
        ensure!(
            bytes[pairs * width * 2..].iter().all(|&byte| byte == 0x5a),
            "grouped producer changed the inactive output guard"
        );
        ensure!(
            bytes[..pairs * width * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .all(|word| {
                    let bits = u16::from_le_bytes(*word);
                    if bf16_wire {
                        bf16::from_bits(bits).is_finite()
                    } else {
                        half::f16::from_bits(bits).is_finite()
                    }
                }),
            "grouped producer emitted a nonfinite intermediate"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn wider_grouped_packed_preserves_completed_real_weight_mlp() -> Result<()> {
    use crate::package::ExpertRowWorkspace;

    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    let metal = Metal::new()?;
    let status = DeviceStatus::new(&metal)?;
    for capacity in [EXPERTS, 93] {
        for layer in [0, 2] {
            let source = package.expert_layer(layer)?;
            for rows in [17, 65] {
                let mut reference_pool = BoundedExperts::with_workspace_contiguous(
                    &metal,
                    source.clone(),
                    capacity,
                    ExpertRowWorkspace::with_prefill_reader(source.row_bytes)?,
                )?;
                let mut candidate_pool = BoundedExperts::with_workspace(
                    &metal,
                    source.clone(),
                    capacity,
                    ExpertRowWorkspace::with_prefill_reader(source.row_bytes)?,
                )?;
                if capacity == EXPERTS {
                    let sources = (0..EXPERTS as u32).collect::<Vec<_>>();
                    reference_pool.seed_residents(&sources)?;
                    candidate_pool.seed_residents(&sources)?;
                }
                let initial_pool = reference_pool.snapshot();
                let bytes = grouped_input(rows);
                let mlp = MlpBatch::new(&metal, &package, layer)?;
                // Below 4096 pairs the contiguous diagnostics use IQ_K GEMV
                // or decoded range GEMM, independently of grouped packed reads.
                let expected = mlp.run_stages(&bytes, rows, &mut reference_pool)?;
                if capacity == EXPERTS {
                    ensure!(
                        reference_pool.snapshot() == initial_pool
                            && expected.before == initial_pool
                            && expected.after == initial_pool,
                        "full-resident diagnostics changed pool bookkeeping"
                    );
                    let counts = &expected.output.counts;
                    ensure!(
                        counts.full_resident
                            && counts.rows == rows
                            && !counts.sorted
                            && counts.chunks == 0
                            && counts.unique_sources == 0
                            && counts.hits == 0
                            && counts.misses == 0
                            && counts.evictions == 0
                            && expected.groups.is_empty(),
                        "full-resident diagnostics performed host demand accounting"
                    );
                }
                let input = upload(&metal, &bytes)?;
                let scratch = MlpBatchScratch::new(&metal, rows + 1)?;
                let dense = DenseBatchScratch::new(&metal, mlp.dense_scratch_bytes(rows + 1)?)?;
                for buffer in [
                    &scratch.packed_activation,
                    &scratch.packed_down,
                    &scratch.base.output,
                ] {
                    buffer.write_bytes(0, &vec![0x5a; buffer.byte_len()])?;
                }
                status.reset()?;
                let mut stream = metal.stream()?;
                let output = mlp.run_device(
                    &mut stream,
                    &input,
                    rows,
                    &scratch,
                    &dense,
                    &mut candidate_pool,
                    &status,
                )?;
                stream.wait()?;
                status.check()?;
                check_grouped_guards(&scratch, rows)?;
                ensure!(
                    read(&scratch.base.ids)?[..rows * ROUTES * 4]
                        == expected.stages["selected_source_ids_u32"],
                    "grouped reconstruction changed unbiased routes"
                );
                ensure!(
                    reference_pool.snapshot() == candidate_pool.snapshot(),
                    "grouped reconstruction changed residency policy"
                );
                if capacity == EXPERTS {
                    ensure!(
                        candidate_pool.snapshot() == initial_pool,
                        "full-resident grouped execution changed pool bookkeeping"
                    );
                }
                let actual = read(&output)?;
                let expected = &expected.stages["final_bf16"];
                for (expected, actual) in expected
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(actual[..rows * HIDDEN * 2].as_chunks::<2>().0)
                {
                    let expected = bf16::from_bits(u16::from_le_bytes(*expected)).to_f32();
                    let actual = bf16::from_bits(u16::from_le_bytes(*actual)).to_f32();
                    ensure!(expected.is_finite() && actual.is_finite());
                    ensure!(
                        (actual - expected).abs() <= 0.0001 + expected.abs() / 128.0,
                        "grouped producer exceeds one BF16 relative step: {actual} versus {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}

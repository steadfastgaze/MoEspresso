use super::*;

const CANARY: u32 = 0xa5a5_a5a5;
const GUARD_WORDS: usize = 16;

fn expected(ids: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut order = Vec::new();
    let mut schedule = Vec::new();
    for source in 0..EXPERTS as u32 {
        let start = order.len();
        order.extend(
            ids.iter()
                .enumerate()
                .filter_map(|(pair, &id)| (id == source).then_some(pair as u32)),
        );
        for first in (start..order.len()).step_by(TILE_PAIRS) {
            schedule.extend([
                source,
                first as u32,
                (order.len() - first).min(TILE_PAIRS) as u32,
            ]);
        }
    }
    (order, schedule)
}

// Model block counts and stable local ranks separately from the source scan.
fn blocked(ids: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let geometry = Geometry::new(ids.len()).unwrap();
    let mut histogram = vec![0; geometry.blocks * EXPERTS];
    for (pair, &id) in ids.iter().enumerate() {
        if (id as usize) < EXPERTS {
            histogram[pair / BLOCK_PAIRS * EXPERTS + id as usize] += 1;
        }
    }
    let mut starts = vec![0; histogram.len()];
    let mut schedule = Vec::new();
    let mut global = 0;
    for source in 0..EXPERTS {
        let first = global;
        for block in 0..geometry.blocks {
            let cell = block * EXPERTS + source;
            starts[cell] = global;
            global += histogram[cell];
        }
        for at in (first..global).step_by(TILE_PAIRS) {
            schedule.extend([
                source as u32,
                at as u32,
                (global - at).min(TILE_PAIRS) as u32,
            ]);
        }
    }
    let mut order = vec![CANARY; global];
    for (block, ids) in ids.chunks(BLOCK_PAIRS).enumerate() {
        for (lane, &id) in ids.iter().enumerate() {
            if (id as usize) < EXPERTS {
                let rank = ids[..lane].iter().filter(|&&prior| prior == id).count();
                let index = starts[block * EXPERTS + id as usize] + rank;
                assert_eq!(order[index], CANARY, "each original pair has one writer");
                order[index] = (block * BLOCK_PAIRS + lane) as u32;
            }
        }
    }
    (order, schedule)
}

fn routes(rows: usize, pattern: usize) -> Vec<u32> {
    (0..rows * ROUTES)
        .map(|pair| {
            let row = pair / ROUTES;
            let route = pair % ROUTES;
            match pattern {
                0 => ((row * 13 + route * 17) % EXPERTS) as u32,
                1 => ((row + route) % ROUTES) as u32,
                _ => {
                    if route < 8 {
                        route as u32
                    } else {
                        (8 + (row * 131 + route) % (EXPERTS - 8)) as u32
                    }
                }
            }
        })
        .collect()
}

#[test]
fn full_schedule_geometry_fits_dead_activation_without_extra_allocation() {
    for rows in [2, 17, 65, 512, 2048, 4096, 6928, 8192] {
        let pairs = rows * ROUTES;
        let geometry = Geometry::new(pairs).unwrap();
        assert_eq!(geometry.block_bytes, geometry.blocks * EXPERTS * 4);
        assert_eq!(geometry.metadata_bytes, geometry.block_bytes * 2);
        assert!(geometry.metadata_bytes <= pairs * SHARED * 2);
        assert_eq!(geometry.max_tiles, EXPERTS + pairs.div_ceil(32));
    }
    assert_eq!(Geometry::new(20).unwrap().metadata_bytes, 4096);
    assert_eq!(Geometry::new(69280).unwrap().metadata_bytes, 1_110_016);
    assert_eq!(Geometry::new(81920).unwrap().metadata_bytes, 1_310_720);
    assert_eq!(Geometry::new(81920).unwrap().max_tiles, 3072);
    assert!(Geometry::new(0).is_err());
    assert!(Geometry::new(u32::MAX as usize).is_err());
    assert!(Geometry::new(usize::MAX).is_err());
}

#[test]
fn blocked_grouping_matches_stable_source_scan_and_original_pairs() {
    for rows in [2, 17, 65, 512, 2048, 4096, 6928, 8192] {
        for pattern in 0..3 {
            let ids = routes(rows, pattern);
            let reference = expected(&ids);
            let actual = blocked(&ids);
            assert_eq!(actual, reference);
            let mut sorted = actual.0.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..ids.len() as u32).collect::<Vec<_>>());
            for pair in actual.0.windows(2) {
                assert!(ids[pair[0] as usize] <= ids[pair[1] as usize]);
                if ids[pair[0] as usize] == ids[pair[1] as usize] {
                    assert!(pair[0] < pair[1]);
                }
            }
        }
    }
    let mut invalid = routes(65, 0);
    invalid[0] = 512;
    invalid[256] = u32::MAX;
    assert_eq!(blocked(&invalid), expected(&invalid));
}

fn upload_words(metal: &Metal, values: &[u32]) -> Result<Buffer> {
    upload(
        metal,
        &values
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn read_words(buffer: &Buffer) -> Result<Vec<u32>> {
    Ok(read(buffer)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

fn reset(buffer: &Buffer) -> Result<()> {
    buffer.write_bytes(
        0,
        &vec![CANARY; buffer.byte_len() / 4]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
}

#[test]
fn device_full_schedule_matches_original_and_guards_across_active_extents() -> Result<()> {
    let metal = Metal::new()?;
    let old = metal.compile_msl_strict(
        include_str!("../../../../kernels/metal/mlp_full_resident.metal"),
        "mlp_full_schedule",
    )?;
    let new = FullSchedule::new(&metal)?;
    let capacity = 8192 * ROUTES;
    let geometry = Geometry::new(capacity)?;
    let ids = metal.shared_buffer((capacity + GUARD_WORDS) * 4)?;
    let orders = [
        metal.shared_buffer((capacity + GUARD_WORDS) * 4)?,
        metal.shared_buffer((capacity + GUARD_WORDS) * 4)?,
    ];
    let schedules = [
        metal.shared_buffer((geometry.max_tiles * 3 + GUARD_WORDS) * 4)?,
        metal.shared_buffer((geometry.max_tiles * 3 + GUARD_WORDS) * 4)?,
    ];
    let totals = [
        metal.shared_buffer((1 + GUARD_WORDS) * 4)?,
        metal.shared_buffer((1 + GUARD_WORDS) * 4)?,
    ];
    let activation = metal.shared_buffer(geometry.metadata_bytes + GUARD_WORDS * 4)?;
    // Reuse the wide allocation for smaller tails and ordinary widths.
    for rows in [8192, 6928, 2, 17, 65, 512, 2048, 4096] {
        for pattern in 0..4 {
            let mut selected = routes(rows, pattern % 3);
            if pattern == 3 {
                selected[0] = 512;
                if selected.len() > 256 {
                    selected[256] = u32::MAX;
                }
            }
            let pairs = selected.len();
            let reference = expected(&selected);
            reset(&ids)?;
            ids.write_bytes(
                0,
                &selected
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            reset(&activation)?;
            for buffer in orders.iter().chain(&schedules).chain(&totals) {
                reset(buffer)?;
            }
            let dims = upload_words(&metal, &[pairs as u32, 2, 2, 0])?;
            let mut batch = metal.batch()?;
            // SAFETY: Fresh fixed-capacity buffers hold the active route count.
            // Invalid IDs are ignored by both schedulers without indexed access.
            unsafe {
                batch.dispatch_1d(
                    &old,
                    &[&ids, &dims, &orders[0], &schedules[0], &totals[0]]
                        .map(|buffer| Binding { buffer, offset: 0 }),
                    EXPERTS,
                    EXPERTS,
                )?;
            }
            new.encode(
                &mut batch,
                ScheduleBuffers {
                    ids: &ids,
                    order: &orders[1],
                    schedule: &schedules[1],
                    total: &totals[1],
                    activation: &activation,
                },
                pairs,
            )?;
            batch.submit().wait()?;
            for index in 0..2 {
                let total = read_words(&totals[index])?;
                assert_eq!(total[0] as usize, reference.1.len() / 3);
                assert!(total[1..].iter().all(|&word| word == CANARY));
                let order = read_words(&orders[index])?;
                assert_eq!(order[..reference.0.len()], reference.0);
                assert!(
                    order[reference.0.len()..]
                        .iter()
                        .all(|&word| word == CANARY)
                );
                let schedule = read_words(&schedules[index])?;
                assert_eq!(schedule[..reference.1.len()], reference.1);
                assert!(
                    schedule[reference.1.len()..]
                        .iter()
                        .all(|&word| word == CANARY)
                );
            }
            let metadata = read_words(&activation)?;
            let active_metadata = Geometry::new(pairs)?.metadata_bytes / 4;
            assert!(
                metadata[active_metadata..]
                    .iter()
                    .all(|&word| word == CANARY)
            );
            let ids_after = read_words(&ids)?;
            assert_eq!(ids_after[..pairs], selected);
            assert!(ids_after[pairs..].iter().all(|&word| word == CANARY));
        }
    }
    let too_small = metal.shared_buffer(geometry.metadata_bytes - 4)?;
    assert!(
        new.encode(
            &mut metal.batch()?,
            ScheduleBuffers {
                ids: &ids,
                order: &orders[1],
                schedule: &schedules[1],
                total: &totals[1],
                activation: &too_small,
            },
            capacity
        )
        .is_err()
    );
    Ok(())
}

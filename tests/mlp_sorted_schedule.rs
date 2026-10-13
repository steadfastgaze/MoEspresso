#![cfg(target_os = "macos")]

use moespresso::metal::{Binding, Buffer, Metal};

fn upload_words(metal: &Metal, words: &[u32]) -> Buffer {
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let buffer = metal.shared_buffer(bytes.len()).unwrap();
    buffer.write_bytes(0, &bytes).unwrap();
    buffer
}

fn read_words(buffer: &Buffer) -> Vec<u32> {
    let mut bytes = vec![0; buffer.byte_len()];
    buffer.read_bytes(0, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

fn expected_schedule(ids: &[u32]) -> Vec<u32> {
    let mut records = Vec::new();
    let mut start = 0;
    for source in 0..512u32 {
        let count = ids.iter().filter(|&&id| id == source).count();
        for tile in 0..count.div_ceil(32) {
            records.extend_from_slice(&[
                source,
                (start + tile * 32) as u32,
                (count - tile * 32).min(32) as u32,
            ]);
        }
        start += count;
    }
    assert_eq!(start, ids.len());
    records
}

#[test]
fn sorted_scheduler_records_match_linear_and_cpu() {
    let metal = Metal::new().unwrap();
    let source = include_str!("../kernels/metal/mlp_full_resident.metal");
    let linear = metal
        .compile_msl_strict(source, "mlp_full_schedule")
        .unwrap();
    let sorted = metal
        .compile_msl_strict(source, "mlp_sorted_schedule")
        .unwrap();
    let cases = [
        vec![0; 1],
        vec![511; 33],
        vec![7; 65],
        (0..512).map(|index| (index / 7) as u32).collect(),
        (0..2048).map(|index| (index * 512 / 2048) as u32).collect(),
        (0..5120).map(|index| (index * 512 / 5120) as u32).collect(),
    ];
    for ids in cases {
        let ids_buffer = upload_words(&metal, &ids);
        let dims = upload_words(&metal, &[ids.len() as u32]);
        let expected = expected_schedule(&ids);
        let mut observed = Vec::new();
        for pipeline in [&linear, &sorted] {
            let order = metal.shared_buffer(ids.len() * 4).unwrap();
            let schedule = metal.shared_buffer((512 + ids.len()) * 3 * 4).unwrap();
            let total = upload_words(&metal, &[u32::MAX]);
            let mut batch = metal.batch().unwrap();
            // SAFETY: IDs are sorted and in range, and the order and schedule
            // buffers cover every possible pair and tile record.
            unsafe {
                batch
                    .dispatch_1d(
                        pipeline,
                        &[&ids_buffer, &dims, &order, &schedule, &total]
                            .map(|buffer| Binding { buffer, offset: 0 }),
                        512,
                        512,
                    )
                    .unwrap();
            }
            batch.submit().wait().unwrap();
            let tiles = read_words(&total)[0] as usize;
            assert_eq!(tiles, expected.len() / 3);
            let order_words = read_words(&order);
            assert_eq!(order_words, (0..ids.len() as u32).collect::<Vec<_>>());
            let records = read_words(&schedule)[..tiles * 3].to_vec();
            assert_eq!(records, expected);
            observed.push((order_words, records));
        }
        assert_eq!(observed[0], observed[1]);
    }
}

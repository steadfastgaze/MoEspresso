#![cfg(target_os = "macos")]

use half::bf16;
use moespresso::metal::{Binding, Buffer, Metal};

fn upload(metal: &Metal, bytes: &[u8]) -> Buffer {
    let value = metal.shared_buffer(bytes.len()).unwrap();
    value.write_bytes(0, bytes).unwrap();
    value
}

fn read(buffer: &Buffer) -> Vec<u8> {
    let mut bytes = vec![0; buffer.byte_len()];
    buffer.read_bytes(0, &mut bytes).unwrap();
    bytes
}

#[test]
fn resident_route_preparation_preserves_source_order_and_rejects_bad_selectors() {
    let metal = Metal::new().unwrap();
    let pipeline = metal
        .compile_msl_strict(
            include_str!("../kernels/metal/expert_block.metal"),
            "prepare_resident_routes",
        )
        .unwrap();
    let base = [511u32, 4, 90, 0, 10, 340, 17, 88, 31, 2];
    for case in 0..5 {
        let mut ids = base;
        let mut scores: Vec<u16> = (0..10)
            .map(|i| bf16::from_f32((i + 1) as f32 / 64.0).to_bits())
            .collect();
        let changed = u32::from(case == 4);
        match case {
            1 => ids[5] = 512,
            2 => ids[5] = ids[4],
            3 => scores[5] = 0x7fc0,
            _ => {}
        }
        let id_buffer = upload(
            &metal,
            &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let score_buffer = upload(
            &metal,
            &scores
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let changed = upload(&metal, &changed.to_le_bytes());
        let slots = metal.shared_buffer(40).unwrap();
        let sorted_scores = metal.shared_buffer(20).unwrap();
        let order = metal.shared_buffer(40).unwrap();
        let status = upload(&metal, &0u32.to_le_bytes());
        let bindings = [
            &id_buffer,
            &score_buffer,
            &changed,
            &slots,
            &sorted_scores,
            &order,
            &status,
        ]
        .map(|buffer| Binding { buffer, offset: 0 });
        let mut batch = metal.batch().unwrap();
        // SAFETY: Fixed ten-route buffers and one validation/sort thread.
        unsafe {
            batch.dispatch_1d(&pipeline, &bindings, 1, 1).unwrap();
        }
        assert!(slots.read_bytes(0, &mut []).is_err());
        batch.submit().wait().unwrap();
        let flag = u32::from_le_bytes(read(&status).try_into().unwrap());
        let mut expected: Vec<u32> = (0..10).collect();
        expected.sort_by_key(|&route| ids[route as usize]);
        let actual_order: Vec<u32> = read(&order)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| u32::from_le_bytes(bytes))
            .collect();
        assert_eq!(actual_order, expected);
        if case == 0 {
            assert_eq!(flag, 0);
            assert_eq!(
                read(&slots),
                expected
                    .iter()
                    .flat_map(|&route| ids[route as usize].to_le_bytes())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                read(&sorted_scores),
                expected
                    .iter()
                    .flat_map(|&route| scores[route as usize].to_le_bytes())
                    .collect::<Vec<_>>()
            );
        } else {
            assert_ne!(flag, 0);
            assert_eq!(read(&slots), vec![0; 40]);
            assert_eq!(read(&sorted_scores), vec![0; 20]);
        }
    }
}

#![cfg(target_os = "macos")]

use half::bf16;
use moespresso::metal::{Binding, Metal};

#[test]
fn three_queries_match_singletons_with_disjoint_scratch_and_output_guards() {
    const S: usize = 2051;
    const Q: usize = 3;
    const VECTOR: usize = 24 * 256 * 2;
    const KV: usize = 2 * S * 256 * 2;
    let metal = Metal::new().unwrap();
    let source = include_str!("../kernels/metal/qsa_attention.metal");
    let pipelines = [
        "qsa_selected_logits",
        "qsa_selected_softmax",
        "qsa_selected_value_gate",
    ]
    .map(|name| metal.compile_msl_strict(source, name).unwrap());
    let upload = |bytes: &[u8]| {
        let buffer = metal.shared_buffer(bytes.len()).unwrap();
        buffer.write_bytes(0, bytes).unwrap();
        buffer
    };
    let data = |words: usize, salt: usize| {
        (0..words)
            .flat_map(|i| {
                bf16::from_f32(((i * 37 + salt) % 257) as f32 / 256. - 0.5)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect::<Vec<_>>()
    };
    let query = upload(&data(Q * VECTOR / 2, 3));
    let keys = upload(&data(Q * KV / 2, 71));
    let values = upload(&data(Q * KV / 2, 113));
    let gate = upload(&data(Q * VECTOR / 2, 191));
    let valid = upload(
        &(0..Q)
            .flat_map(|q| (0..S).map(move |i| u8::from(i < [1, 257, S][q])))
            .collect::<Vec<_>>(),
    );
    let sizes = [
        24 * S * 2,
        24 * S * 2,
        24 * S * 4,
        24 * S * 2,
        VECTOR,
        VECTOR,
        VECTOR,
    ];
    let mut results = Vec::new();
    for tiled in [false, true] {
        let output = sizes.map(|size| upload(&vec![0x5a; (Q + 1) * size]));
        let mut batch = metal.batch().unwrap();
        for q in 0..if tiled { 1 } else { Q } {
            let count = if tiled { Q } else { 1 };
            let shape = upload(
                &[S as u32, count as u32]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let out = q + 1;
            // SAFETY: Each dispatch reads complete finite query-major rows;
            // output row zero is a guard and all written rows are disjoint.
            unsafe {
                let bindings = [
                    (&query, q * VECTOR),
                    (&keys, q * KV),
                    (&valid, q * S),
                    (&output[0], out * sizes[0]),
                    (&output[1], out * sizes[1]),
                    (&output[2], out * sizes[2]),
                    (&shape, 0),
                ]
                .map(|(buffer, offset)| Binding { buffer, offset });
                batch
                    .dispatch_1d(&pipelines[0], &bindings, count * 24 * S, 128)
                    .unwrap();
                let bindings = [
                    (&output[2], out * sizes[2]),
                    (&output[3], out * sizes[3]),
                    (&shape, 0),
                ]
                .map(|(buffer, offset)| Binding { buffer, offset });
                batch
                    .dispatch_1d(&pipelines[1], &bindings, count * 24 * 544, 544)
                    .unwrap();
                let bindings = [
                    (&output[3], out * sizes[3]),
                    (&values, q * KV),
                    (&gate, q * VECTOR),
                    (&output[4], out * sizes[4]),
                    (&output[5], out * sizes[5]),
                    (&output[6], out * sizes[6]),
                    (&shape, 0),
                ]
                .map(|(buffer, offset)| Binding { buffer, offset });
                batch
                    .dispatch_1d(&pipelines[2], &bindings, count * 24 * 256, 128)
                    .unwrap();
            }
        }
        batch.submit().wait().unwrap();
        let stages = output
            .iter()
            .zip(sizes)
            .map(|(buffer, size)| {
                let mut bytes = vec![0; buffer.byte_len()];
                buffer.read_bytes(0, &mut bytes).unwrap();
                assert!(bytes[..size].iter().all(|&v| v == 0x5a));
                bytes
            })
            .collect::<Vec<_>>();
        results.push(stages);
    }
    for (stage, (single, tiled)) in results[0].iter().zip(&results[1]).enumerate() {
        assert_eq!(single, tiled, "query batching changed stage {stage}");
    }
}

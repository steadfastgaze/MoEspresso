#![cfg(target_os = "macos")]

use half::f16;
use moespresso::metal::{Binding, Metal};

fn replace_once(source: &str, before: &str, after: &str) -> String {
    assert_eq!(source.matches(before).count(), 1);
    source.replacen(before, after, 1)
}

fn finite_words(seed: &mut u32, length: usize, family: usize) -> Vec<u8> {
    let edges = [
        0x0000u16, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff,
        0xfbff, 0x3800, 0xb800, 0x3555, 0xb555,
    ];
    (0..length)
        .flat_map(|index| {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let bits = match family {
                0 => f16::from_f32(((*seed >> 8) as i32 - 8388608) as f32 / 1048576.).to_bits(),
                1 => edges[(*seed as usize >> 16) % edges.len()],
                2 => {
                    let exponent = ((*seed >> 24) % 31) as i32 - 15;
                    let sign = if index % 2 == 0 { 1. } else { -1. };
                    f16::from_f32(sign * (1. + (*seed & 1023) as f32 / 1024.) * 2f32.powi(exponent))
                        .to_bits()
                }
                _ => ((*seed as u16) & 0x8000) | (((*seed >> 16) as u16) & 0x03ff),
            };
            assert!(f16::from_bits(bits).is_finite());
            bits.to_le_bytes()
        })
        .collect()
}

#[test]
fn half_operands_preserve_float_accumulators_across_ranges_and_tails() {
    compare_sorted_accumulators(false);
}

#[test]
fn vector_loads_preserve_float_accumulators_across_source_tiles_and_guards() {
    compare_sorted_accumulators(true);
}

fn compare_sorted_accumulators(compare_loaders: bool) {
    let body = include_str!("../kernels/metal/mlp_batch.metal");
    // Observe unrounded accumulators so FP16 stores cannot hide a difference.
    let body = replace_once(
        body,
        "device half* output [[buffer(4)]], uint group",
        "device float* output [[buffer(4)]], uint group",
    );
    let body = replace_once(
        &body,
        "= half(accum[r][c].thread_elements()[e]);",
        "= accum[r][c].thread_elements()[e];",
    );
    let reference = if compare_loaders {
        replace_once(
            &body,
            r#"        if (a_valid)
            *((threadgroup MlpReadA*)(a_shared + ar * 24 + ak)) =
                *((device const MlpReadA*)a_src);
        else
            for (uint j = 0; j < 4; ++j) a_shared[ar * 24 + ak + j] = half(0);
        *((threadgroup MlpReadB*)(b_shared + tid * 24)) =
            *((device const MlpReadB*)b_src);
        a_src += 16;
        b_src += 16;"#,
            r#"        for (uint p = tid; p < 16 * 16; p += 64) {
            uint r = p / 16, k = p % 16;
            a_shared[r * 24 + k] = rb + r < end ? input[(rb + r) * K + kb + k] : half(0);
        }
        for (uint p = tid; p < 64 * 16; p += 64) {
            uint c = p / 16, k = p % 16;
            b_shared[c * 24 + k] = weights[(source * range + cb + c) * K + kb + k];
        }"#,
        )
    } else {
        replace_once(
            &body,
            "simdgroup_half8x8 a[2], b[4];",
            "simdgroup_float8x8 a[2], b[4];",
        )
    };
    let source = |body: &str| {
        format!(
            "{}\n{}\n{body}",
            include_str!("../kernels/metal/mlp.metal"),
            include_str!("../kernels/metal/expert_block.metal"),
        )
    };
    let metal = Metal::new().unwrap();
    let reference = metal
        .compile_msl(&source(&reference), "mlp_batch_sorted_mm")
        .unwrap();
    let candidate = metal
        .compile_msl(&source(&body), "mlp_batch_sorted_mm")
        .unwrap();
    let upload = |bytes: &[u8]| {
        let buffer = metal.shared_buffer(bytes.len()).unwrap();
        buffer.write_bytes(0, bytes).unwrap();
        buffer
    };
    for k in [768usize, 2560] {
        for range in [64usize, 256] {
            for tail in [1usize, 7, 15] {
                let first = 3usize;
                let second = first + 16 + 2;
                let rows = second + tail + 2;
                let stride = range + 128;
                let tiles = upload(
                    &[
                        first as u32,
                        (first + 16) as u32,
                        2,
                        second as u32,
                        (second + tail) as u32,
                        0,
                    ]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
                );
                let dims = upload(
                    &[k as u32, stride as u32, range as u32, 64]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                for family in 0..4 {
                    let mut seed = 0x312718a1;
                    let input = upload(&finite_words(&mut seed, rows * k, family));
                    let weights = upload(&finite_words(&mut seed, 3 * range * k, family));
                    let mut outputs = Vec::new();
                    for pipeline in [&reference, &candidate] {
                        let sentinel = 0x7fc01234u32.to_le_bytes();
                        let output = upload(&sentinel.repeat(rows * stride));
                        let mut batch = metal.batch().unwrap();
                        // SAFETY: Two source tiles contain 16 and tail rows;
                        // weights, dimensions and output cover every range index.
                        unsafe {
                            batch
                                .dispatch_1d(
                                    pipeline,
                                    &[&input, &weights, &tiles, &dims, &output]
                                        .map(|buffer| Binding { buffer, offset: 0 }),
                                    2 * (range / 64) * 64,
                                    64,
                                )
                                .unwrap();
                        }
                        batch.submit().wait().unwrap();
                        let mut bytes = vec![0; output.byte_len()];
                        output.read_bytes(0, &mut bytes).unwrap();
                        for (index, row) in bytes.chunks_exact(stride * 4).enumerate() {
                            if (first..first + 16).contains(&index)
                                || (second..second + tail).contains(&index)
                            {
                                assert_eq!(&row[..64 * 4], sentinel.repeat(64));
                                assert_eq!(&row[(64 + range) * 4..], sentinel.repeat(64));
                            } else {
                                assert_eq!(row, sentinel.repeat(stride));
                            }
                        }
                        outputs.push(bytes);
                    }
                    assert_eq!(
                        outputs[0], outputs[1],
                        "FP32 accumulator mismatch: K{k}, range{range}, tail{tail}, family{family}"
                    );
                }
            }
        }
    }
}

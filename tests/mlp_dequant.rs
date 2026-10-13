#![cfg(target_os = "macos")]

use half::f16;
use moespresso::iqk::Codec;
use moespresso::metal::{Binding, Metal};

fn random_bytes(seed: &mut u32, count: usize) -> Vec<u8> {
    (0..count)
        .map(|_| {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*seed >> 24) as u8
        })
        .collect()
}

#[test]
fn grouped_dequant_matches_scalar_reconstruction_ranges_slots_and_guards() {
    let metal = Metal::new().unwrap();
    let source = format!(
        "{}\n{}\n{}",
        include_str!("../kernels/metal/mlp.metal"),
        include_str!("../kernels/metal/expert_block.metal"),
        include_str!("../kernels/metal/mlp_batch.metal"),
    );
    let kernel = metal.compile_msl(&source, "mlp_batch_dequant").unwrap();
    let upload = |bytes: &[u8]| {
        let buffer = metal.shared_buffer(bytes.len()).unwrap();
        buffer.write_bytes(0, bytes).unwrap();
        buffer
    };
    for codec in [Codec::Iq2K, Codec::Iq3K] {
        for (k, n, range) in [(2560usize, 640usize, 64usize), (768, 2560, 256)] {
            let count = 4 * n;
            let mut seed = 0x1624_7821;
            let qs = random_bytes(&mut seed, count * k / 4);
            let qh = random_bytes(&mut seed, count * k / 32 * 4);
            let scl = random_bytes(&mut seed, count * k / 32);
            let sch = random_bytes(&mut seed, count * k / 128);
            let sex = random_bytes(&mut seed, count * k / 128);
            let edges = [
                0x0000u16, 0x8000, 0x0001, 0x03ff, 0x0400, 0x1401, 0x2801, 0x3000, 0x3555, 0x3800,
                0x3c00, 0xbc00,
            ];
            let dv: Vec<u16> = (0..count * k / 256)
                .map(|i| edges[i % edges.len()])
                .collect();
            let table = codec.value_table_fp16();
            let table_words: Vec<_> = table
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| f16::from_bits(u16::from_le_bytes(*v)).to_f32())
                .collect();
            let qs_buffer = upload(&qs);
            let scl_buffer = upload(&scl);
            // IQ2 has no high-bit or sign streams; match pool aliases.
            let qh_buffer = if codec == Codec::Iq3K {
                upload(&qh)
            } else {
                qs_buffer.clone()
            };
            let sch_buffer = if codec == Codec::Iq3K {
                upload(&sch)
            } else {
                scl_buffer.clone()
            };
            let streams = [
                qs_buffer,
                qh_buffer,
                scl_buffer,
                sch_buffer,
                upload(&sex),
                upload(&dv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
                upload(&table),
            ];
            // Nonidentity active order excludes physical slot one. Down weights
            // include nonzero synthetic values in stored columns 640..768.
            let slots = [3u32, 0, 2];
            let active = upload(
                &slots
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            for base in [0, n - range] {
                let dims = upload(
                    &[
                        k as u32,
                        n as u32,
                        base as u32,
                        range as u32,
                        if codec == Codec::Iq3K { 3 } else { 2 },
                    ]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
                );
                let bytes = slots.len() * range * k * 2;
                let output = upload(&vec![0x5a; bytes + 64]);
                let threads = slots.len() * range * (k / 32);
                assert!(threads.is_multiple_of(256));
                let mut bindings: Vec<_> = streams
                    .iter()
                    .map(|buffer| Binding { buffer, offset: 0 })
                    .collect();
                bindings.extend([
                    Binding {
                        buffer: &active,
                        offset: 0,
                    },
                    Binding {
                        buffer: &dims,
                        offset: 0,
                    },
                    Binding {
                        buffer: &output,
                        offset: 32,
                    },
                ]);
                let mut batch = metal.batch().unwrap();
                // SAFETY: Complete 256-thread groups initialize the shared table;
                // each thread writes 32 distinct words within the guarded range.
                unsafe {
                    batch.dispatch_1d(&kernel, &bindings, threads, 256).unwrap();
                }
                batch.submit().wait().unwrap();
                let mut actual = vec![0u8; bytes + 64];
                output.read_bytes(0, &mut actual).unwrap();
                assert_eq!(&actual[..32], &[0x5a; 32]);
                assert_eq!(&actual[32 + bytes..], &[0x5a; 32]);
                for (source, &slot) in slots.iter().enumerate() {
                    for row in 0..range {
                        let rid = slot as usize * n + base + row;
                        for col in 0..k {
                            let group = col / 32;
                            let sub = col % 32 / 16;
                            let low = (qs[rid * k / 4 + col / 4] >> (2 * (col % 4))) & 3;
                            let scale_byte = scl[rid * k / 32 + group];
                            let nibble = (scale_byte >> (4 * sub)) & 15;
                            let extra =
                                (sex[rid * k / 128 + group / 4] >> ((2 * group + sub) % 8)) & 1;
                            let (scale, code) = if codec == Codec::Iq3K {
                                let at = (rid * k / 32 + group) * 4;
                                let high = (u32::from_le_bytes(qh[at..at + 4].try_into().unwrap())
                                    >> (col % 32))
                                    & 1;
                                let sign =
                                    (sch[rid * k / 128 + group / 4] >> ((2 * group + sub) % 8)) & 1;
                                (
                                    (2 * i32::from(nibble) + 1) * if sign == 0 { 1 } else { -1 },
                                    low as usize + high as usize * 4 + extra as usize * 8,
                                )
                            } else {
                                (i32::from(nibble) - 8, low as usize + extra as usize * 4)
                            };
                            let d = f16::from_bits(dv[rid * k / 256 + group / 8]).to_f32()
                                * scale as f32;
                            let expected = f16::from_f32(d * table_words[code]).to_bits();
                            let at = 32 + ((source * range + row) * k + col) * 2;
                            let got = u16::from_le_bytes(actual[at..at + 2].try_into().unwrap());
                            assert_eq!(
                                got, expected,
                                "{codec:?} K{k} base{base} slot{slot} row{row} col{col}"
                            );
                        }
                    }
                }
            }
        }
    }
}

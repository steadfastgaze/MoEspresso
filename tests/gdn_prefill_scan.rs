#![cfg(target_os = "macos")]

use half::bf16;
use moespresso::metal::{Binding, Buffer, Metal};

const HEADS: usize = 48;
const WIDTH: usize = 128;
const KEYS: usize = 16 * WIDTH;
const VALUES: usize = HEADS * WIDTH;

fn upload(metal: &Metal, bytes: &[u8]) -> Buffer {
    let buffer = metal.shared_buffer(bytes.len()).unwrap();
    buffer.write_bytes(0, bytes).unwrap();
    buffer
}

fn read(buffer: &Buffer) -> Vec<u8> {
    let mut bytes = vec![0; buffer.byte_len()];
    buffer.read_bytes(0, &mut bytes).unwrap();
    bytes
}

fn bf16_data(count: usize, seed: &mut u32) -> Vec<bf16> {
    (0..count)
        .map(|_| {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bf16::from_f32(((*seed >> 24) as f32 - 128.0) / 4096.0)
        })
        .collect()
}

#[test]
fn four_row_scan_matches_fp64_recurrence_with_in_place_state() {
    let metal = Metal::new().unwrap();
    let source = format!(
        "{}\n{}",
        include_str!("../kernels/metal/gdn.metal"),
        include_str!("../kernels/metal/gdn_batch.metal")
    );
    let pipeline = metal
        .compile_msl_strict(&source, "gdn_batch_recurrence_r4")
        .unwrap();
    for rows in [9usize, 17] {
        let mut seed = 37;
        let query = bf16_data(rows * KEYS, &mut seed);
        let key = bf16_data(rows * KEYS, &mut seed);
        let value = bf16_data(rows * VALUES, &mut seed);
        let beta: Vec<_> = (0..rows * HEADS)
            .map(|i| bf16::from_f32(0.125 + (i % 7) as f32 / 32.0))
            .collect();
        let decay: Vec<f32> = (0..rows * HEADS)
            .map(|i| 0.875 + (i % 5) as f32 / 64.0)
            .collect();
        let initial: Vec<f32> = bf16_data(HEADS * WIDTH * WIDTH, &mut seed)
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let mut expected_state: Vec<f64> = initial.iter().map(|&v| f64::from(v)).collect();
        let mut expected_output = vec![0f64; rows * VALUES];
        for token in 0..rows {
            for head in 0..HEADS {
                // Three consecutive value heads share each query/key head.
                let key_start = token * KEYS + (head / 3) * WIDTH;
                let gain = f64::from(decay[token * HEADS + head]);
                let beta = f64::from(beta[token * HEADS + head].to_f32());
                for row in 0..WIDTH {
                    let state_start = (head * WIDTH + row) * WIDTH;
                    let state = &mut expected_state[state_start..state_start + WIDTH];
                    let mut prediction = 0.0;
                    for (column, state) in state.iter_mut().enumerate() {
                        *state *= gain;
                        prediction += *state * f64::from(key[key_start + column].to_f32());
                    }
                    let output_index = token * VALUES + head * WIDTH + row;
                    let correction = (f64::from(value[output_index].to_f32()) - prediction) * beta;
                    let mut output = 0.0;
                    for (column, state) in state.iter_mut().enumerate() {
                        *state += correction * f64::from(key[key_start + column].to_f32());
                        output += *state * f64::from(query[key_start + column].to_f32());
                    }
                    expected_output[output_index] = output;
                }
            }
        }
        let upload_bf16 = |values: &[bf16]| {
            upload(
                &metal,
                &values
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        let q = upload_bf16(&query);
        let k = upload_bf16(&key);
        let v = upload_bf16(&value);
        let b = upload_bf16(&beta);
        let g = upload(
            &metal,
            &decay
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let state = upload(
            &metal,
            &initial
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let output = metal.shared_buffer(rows * VALUES * 2).unwrap();
        let bindings = [&q, &k, &v, &g, &b, &state, &state, &output]
            .map(|buffer| Binding { buffer, offset: 0 });
        let mut batch = metal.batch().unwrap();
        // SAFETY: Each SIMD group owns four disjoint state rows; input and
        // output state alias deliberately, without cross-group dependencies.
        unsafe {
            batch
                .dispatch_1d_with_inline(
                    &pipeline,
                    &bindings,
                    &[&(rows as u32).to_le_bytes()],
                    HEADS * 32 * 32,
                    32,
                )
                .unwrap();
        }
        batch.submit().wait().unwrap();

        // Input magnitudes are <=1/32 and decays <1. This conservative FP32
        // allowance scales with recurrence length; output also permits one
        // BF16 ULP. Neither requires the reference's reduction tree.
        let fp32_allowance = 32.0 * f64::from(f32::EPSILON) * (rows + 1) as f64 / 32.0;
        let mut max_state_error = 0f64;
        for (&bytes, expected) in read(&state).as_chunks::<4>().0.iter().zip(expected_state) {
            let actual = f64::from(f32::from_le_bytes(bytes));
            let error = (actual - expected).abs();
            assert!(
                actual.is_finite() && error <= fp32_allowance,
                "state error {error}"
            );
            max_state_error = max_state_error.max(error);
        }
        let mut max_output_error = 0f64;
        for (&bytes, expected) in read(&output).as_chunks::<2>().0.iter().zip(expected_output) {
            let actual = bf16::from_bits(u16::from_le_bytes(bytes)).to_f32();
            let rounded = bf16::from_f32(expected.abs() as f32);
            let adjacent = bf16::from_bits(rounded.to_bits() + 1).to_f32();
            let ulp = f64::from(adjacent - rounded.to_f32());
            let error = (f64::from(actual) - expected).abs();
            assert!(
                actual.is_finite() && error <= ulp + fp32_allowance,
                "output error {error}"
            );
            max_output_error = max_output_error.max(error);
        }
        eprintln!(
            "GDN R4 rows={rows} state_max={max_state_error:.9} output_max={max_output_error:.9}"
        );
    }
}

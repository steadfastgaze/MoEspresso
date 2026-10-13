#![cfg(target_os = "macos")]

use half::bf16;
use moespresso::hc::{HcBlock, HcStages};
use moespresso::kquant::{DenseMatvec, OutputPrecision};
use moespresso::metal::{Binding, Buffer, Metal};
use moespresso::package::{DirectWeight, WeightFormat};

// The unfused activation boundaries provide an independent staging reference.
const SPLIT_SOURCE: &str = r#"
kernel void reference_front(
    device const ushort* down [[buffer(0)]], device const ushort* inj [[buffer(1)]],
    device ushort* lowrank [[buffer(2)]], device ushort* injection [[buffer(3)]],
    constant int* has_injection [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    ushort scaled = hc_bf16(hc_float(down[id]) * 0.25f);
    lowrank[id] = hc_bf16(hc_float(scaled) * hc_float(hc_sigmoid(scaled)));
    if (*has_injection != 0 && id < 4) {
        ushort scaled_inj = hc_bf16(hc_float(inj[id]) * 0.25f);
        injection[id] = hc_bf16(2.0f * hc_float(hc_sigmoid(scaled_inj)));
    }
}
kernel void reference_epilogue(
    device const ushort* up [[buffer(0)]], device const ushort* normalized [[buffer(1)]],
    device ushort* mixed [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    ushort total = 0;
    for (uint stream = 0; stream < 4; ++stream) {
        uint index = stream * 2560 + id;
        ushort product = hc_bf16(hc_float(hc_sigmoid(up[index])) * hc_float(normalized[index]));
        total = hc_bf16(hc_float(total) + hc_float(product));
    }
    mixed[id] = hc_bf16(hc_float(total) * 0.25f);
}
"#;

fn next(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

fn packed(seed: &mut u32, format: WeightFormat, n: usize, k: usize) -> DirectWeight {
    let (block, scale, blocks) = match format {
        WeightFormat::Q6K => (210, 208, n * k / 256),
        WeightFormat::Q8_0 => (34, 0, n * k / 32),
        _ => unreachable!(),
    };
    let mut bytes: Vec<u8> = (0..blocks * block)
        .map(|_| (next(seed) >> 24) as u8)
        .collect();
    let scales = [0u16, 0x8000, 0x0001, 0x0400, 0x1000, 0x1401, 0x9801, 0x2000];
    for (index, bytes) in bytes.chunks_exact_mut(block).enumerate() {
        bytes[scale..scale + 2].copy_from_slice(&scales[index % scales.len()].to_le_bytes());
    }
    DirectWeight {
        role: "synthetic HC projection".into(),
        format,
        shape: vec![n, k],
        bytes,
    }
}

fn activation(seed: &mut u32, words: usize) -> Vec<u8> {
    (0..words)
        .flat_map(|index| {
            let word = match index % 37 {
                0 => 0x0000,
                1 => 0x8000,
                2 => 0x0001,
                3 => 0x8001,
                _ => bf16::from_f32(((next(seed) >> 16) as f32 - 32768.0) / 65536.0).to_bits(),
            };
            word.to_le_bytes()
        })
        .collect()
}

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

struct SplitCell {
    down: DenseMatvec,
    up: DenseMatvec,
    injection: DenseMatvec,
}

impl SplitCell {
    fn run(
        &self,
        metal: &Metal,
        norm_weight: &[u8],
        residual: &[u8],
        pending: Option<(&[u8], &[u8])>,
        has_injection: bool,
    ) -> HcStages {
        let source = format!(
            "{}\n{SPLIT_SOURCE}",
            include_str!("../kernels/metal/hc.metal")
        );
        let norm = metal.compile_msl_strict(&source, "hc_norm").unwrap();
        let front = metal
            .compile_msl_strict(&source, "reference_front")
            .unwrap();
        let epilogue = metal
            .compile_msl_strict(&source, "reference_epilogue")
            .unwrap();
        let residual = upload(metal, residual);
        let norm_weight = upload(metal, norm_weight);
        let pending_output = upload(metal, pending.map_or(&[0; 5120], |p| p.0));
        let pending_injection = upload(metal, pending.map_or(&[0; 8], |p| p.1));
        let has_pending = upload(metal, &i32::from(pending.is_some()).to_le_bytes());
        let has_injection_buffer = upload(metal, &i32::from(has_injection).to_le_bytes());
        let eps = upload(metal, &1e-6f32.to_le_bytes());
        let normalized = metal.shared_buffer(20480).unwrap();
        let updated = metal.shared_buffer(20480).unwrap();
        let projected_down = metal.shared_buffer(640).unwrap();
        let projected_injection = upload(metal, &[0; 8]);
        let lowrank = metal.shared_buffer(640).unwrap();
        let injection = upload(metal, &[0; 8]);
        let projected_up = metal.shared_buffer(20480).unwrap();
        let mixed = metal.shared_buffer(5120).unwrap();
        let mut batch = metal.batch().unwrap();
        // SAFETY: Fixed HC geometry, separate output buffers and retained inputs.
        unsafe {
            batch
                .dispatch_1d(
                    &norm,
                    &[
                        &residual,
                        &norm_weight,
                        &pending_output,
                        &pending_injection,
                        &normalized,
                        &updated,
                        &eps,
                        &has_pending,
                    ]
                    .map(|buffer| Binding { buffer, offset: 0 }),
                    4 * 640,
                    640,
                )
                .unwrap();
            self.down
                .encode(
                    &mut batch,
                    &normalized,
                    &projected_down,
                    OutputPrecision::Bf16,
                )
                .unwrap();
            if has_injection {
                self.injection
                    .encode(
                        &mut batch,
                        &normalized,
                        &projected_injection,
                        OutputPrecision::Bf16,
                    )
                    .unwrap();
            }
            batch
                .dispatch_1d(
                    &front,
                    &[
                        &projected_down,
                        &projected_injection,
                        &lowrank,
                        &injection,
                        &has_injection_buffer,
                    ]
                    .map(|buffer| Binding { buffer, offset: 0 }),
                    320,
                    64,
                )
                .unwrap();
            self.up
                .encode(&mut batch, &lowrank, &projected_up, OutputPrecision::Bf16)
                .unwrap();
            batch
                .dispatch_1d(
                    &epilogue,
                    &[&projected_up, &normalized, &mixed]
                        .map(|buffer| Binding { buffer, offset: 0 }),
                    2560,
                    256,
                )
                .unwrap();
        }
        batch.submit().wait().unwrap();
        HcStages {
            updated_bf16: read(&updated),
            normalized_bf16: read(&normalized),
            projected_down_bf16: read(&projected_down),
            projected_injection_bf16: has_injection.then(|| read(&projected_injection)),
            lowrank_bf16: read(&lowrank),
            injection_bf16: has_injection.then(|| read(&injection)),
            projected_up_bf16: read(&projected_up),
            mixed_bf16: read(&mixed),
        }
    }
}

fn compare(actual: &HcStages, expected: &HcStages) {
    for (name, actual, expected) in [
        ("updated", &actual.updated_bf16, &expected.updated_bf16),
        (
            "normalized",
            &actual.normalized_bf16,
            &expected.normalized_bf16,
        ),
        (
            "projected down",
            &actual.projected_down_bf16,
            &expected.projected_down_bf16,
        ),
        ("lowrank", &actual.lowrank_bf16, &expected.lowrank_bf16),
        (
            "projected up",
            &actual.projected_up_bf16,
            &expected.projected_up_bf16,
        ),
        ("mixed", &actual.mixed_bf16, &expected.mixed_bf16),
    ] {
        assert_eq!(actual, expected, "{name}");
    }
    assert_eq!(
        actual.projected_injection_bf16,
        expected.projected_injection_bf16
    );
    assert_eq!(actual.injection_bf16, expected.injection_bf16);
}

#[test]
fn three_dispatch_hc_matches_split_stages_and_retains_scratch() {
    let metal = Metal::new().unwrap();
    let mut seed = 0x182b_173a;
    let down = packed(&mut seed, WeightFormat::Q6K, 320, 10240);
    let up = packed(&mut seed, WeightFormat::Q8_0, 10240, 320);
    let injection = packed(&mut seed, WeightFormat::Q6K, 4, 10240);
    let norm = activation(&mut seed, 10240);
    let residual = activation(&mut seed, 10240);
    let pending_output = activation(&mut seed, 2560);
    let pending_injection: Vec<u8> = [0.5f32, -0.25, 1.0, 0.125]
        .into_iter()
        .flat_map(|value| bf16::from_f32(value).to_bits().to_le_bytes())
        .collect();
    let split = SplitCell {
        down: DenseMatvec::new(&metal, &down).unwrap(),
        up: DenseMatvec::new(&metal, &up).unwrap(),
        injection: DenseMatvec::new(&metal, &injection).unwrap(),
    };
    for has_injection in [false, true] {
        let mut block = HcBlock::new(
            &metal,
            &norm,
            &down,
            &up,
            has_injection.then_some(&injection),
            1e-6,
        )
        .unwrap();
        for has_pending in [false, true] {
            let pending =
                has_pending.then_some((pending_output.as_slice(), pending_injection.as_slice()));
            let expected = split.run(&metal, &norm, &residual, pending, has_injection);
            let flight = block.submit(&residual, pending).unwrap();
            assert!(block.submit(&residual, pending).is_err());
            compare(&flight.wait_stages().unwrap(), &expected);
            let flight = block.submit(&residual, pending).unwrap();
            drop(flight);
            let actual = block.run(&residual, pending).unwrap();
            assert_eq!(actual.mixed_bf16, expected.mixed_bf16);
            assert_eq!(actual.updated_bf16, expected.updated_bf16);
            assert_eq!(actual.injection_bf16, expected.injection_bf16);
        }
    }
}

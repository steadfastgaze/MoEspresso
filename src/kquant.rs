//! Dense decode matvec over declared BF16, Q6_K and Q8_0 matrix wires.
//!
//! Q6_K and Q8_0 use the GGUF blocks kept by the package. One-token decode
//! accepts BF16 activations and returns BF16 unless FP32 is requested explicitly.

use crate::package::{DirectWeight, WeightFormat};
use anyhow::{Context, Result, ensure};
use half::bf16;

pub const METAL_SOURCE: &str = include_str!("../kernels/metal/kquant.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputPrecision {
    Bf16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseGeometry {
    format: WeightFormat,
    output: usize,
    input: usize,
    matrix_bytes: usize,
}

impl DenseGeometry {
    pub fn from_weight(weight: &DirectWeight) -> Result<Self> {
        ensure!(
            weight.shape.len() == 2,
            "dense matvec requires a two-dimensional matrix"
        );
        let output = weight.shape[0];
        let input = weight.shape[1];
        ensure!(
            output > 0 && input > 0,
            "dense matrix dimensions must be positive"
        );
        ensure!(
            output <= u32::MAX as usize && input <= u32::MAX as usize,
            "dense matrix dimensions exceed Metal uint range"
        );
        let row_bytes = match weight.format {
            WeightFormat::Bf16 => input.checked_mul(2),
            WeightFormat::Q6K if input.is_multiple_of(256) => {
                input.checked_div(256).and_then(|v| v.checked_mul(210))
            }
            WeightFormat::Q8_0 if input.is_multiple_of(32) => {
                input.checked_div(32).and_then(|v| v.checked_mul(34))
            }
            _ => None,
        }
        .context("dense width is invalid for its format")?;
        let matrix_bytes = output
            .checked_mul(row_bytes)
            .context("dense matrix size overflow")?;
        ensure!(
            matrix_bytes == weight.bytes.len(),
            "dense matrix has {} bytes; expected {matrix_bytes}",
            weight.bytes.len()
        );
        ensure!(
            matrix_bytes <= u32::MAX as usize,
            "dense matrix exceeds Metal address range"
        );
        Ok(Self {
            format: weight.format,
            output,
            input,
            matrix_bytes,
        })
    }

    pub fn input_bytes(self) -> usize {
        self.input * 2
    }

    pub fn format(self) -> WeightFormat {
        self.format
    }
    pub fn input(self) -> usize {
        self.input
    }
    pub fn output(self) -> usize {
        self.output
    }
    pub fn matrix_bytes(self) -> usize {
        self.matrix_bytes
    }

    pub fn output_bytes(self, precision: OutputPrecision) -> Result<usize> {
        self.output
            .checked_mul(match precision {
                OutputPrecision::Bf16 => 2,
                OutputPrecision::F32 => 4,
            })
            .context("dense output size overflow")
    }

    pub fn grid_threads(self) -> Result<usize> {
        self.output
            .div_ceil(8)
            .checked_mul(64)
            .context("dense grid size overflow")
    }
}

pub fn validate_activation(input: &[u16], geometry: DenseGeometry) -> Result<()> {
    ensure!(
        input.len() == geometry.input,
        "dense activation width differs from matrix input width"
    );
    ensure!(
        input.iter().all(|&v| bf16::from_bits(v).is_finite()),
        "dense activation requires finite BF16 values"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::{DenseGeometry, METAL_SOURCE, OutputPrecision, validate_activation};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use crate::package::{DirectWeight, WeightFormat};
    use anyhow::{Context, Result, ensure};

    /// Immutable validated matrix allocation shared by numerical operators.
    /// Storage stays private so callers cannot rewrite a published weight.
    #[derive(Clone)]
    pub(crate) struct DeviceMatrix {
        geometry: DenseGeometry,
        metal: Metal,
        storage: Buffer,
    }
    impl DeviceMatrix {
        pub(crate) fn geometry(&self) -> DenseGeometry {
            self.geometry
        }
        pub(crate) fn metal(&self) -> &Metal {
            &self.metal
        }
        pub(crate) fn upload(metal: &Metal, direct: &DirectWeight) -> Result<Self> {
            let geometry = DenseGeometry::from_weight(direct)?;
            let storage = metal.shared_buffer(direct.bytes.len())?;
            storage.write_bytes(0, &direct.bytes)?;
            Ok(Self {
                geometry,
                metal: metal.clone(),
                storage,
            })
        }
        pub(crate) fn binding(&self) -> Binding<'_> {
            Binding {
                buffer: &self.storage,
                offset: 0,
            }
        }
        pub(crate) fn check(&self, format: WeightFormat, shape: [usize; 2]) -> Result<()> {
            ensure!(
                self.geometry.format() == format
                    && self.geometry.output() == shape[0]
                    && self.geometry.input() == shape[1],
                "shared matrix role geometry differs"
            );
            Ok(())
        }
    }

    /// Reusable direct matrix and both output-precision pipelines.
    pub struct DenseMatvec {
        geometry: DenseGeometry,
        weight: DeviceMatrix,
        dims: Buffer,
        bf16: Pipeline,
        f32: Pipeline,
    }

    impl DenseMatvec {
        pub fn new(metal: &Metal, direct: &DirectWeight) -> Result<Self> {
            let weight = DeviceMatrix::upload(metal, direct)?;
            let geometry = weight.geometry;
            let (bf16_name, f32_name) = match geometry.format {
                WeightFormat::Bf16 => ("dense_raw_bf16_bf16", "dense_raw_bf16_f32"),
                WeightFormat::Q6K => ("dense_q6_k_bf16", "dense_q6_k_f32"),
                WeightFormat::Q8_0 => ("dense_q8_0_bf16", "dense_q8_0_f32"),
            };
            let bf16 = metal.compile_msl_strict(METAL_SOURCE, bf16_name)?;
            let f32 = metal.compile_msl_strict(METAL_SOURCE, f32_name)?;
            ensure!(
                bf16.max_threads_per_threadgroup() >= 64 && f32.max_threads_per_threadgroup() >= 64,
                "dense matvec requires a 64-thread Metal group"
            );
            let dims = metal.shared_buffer(8)?;
            dims.write_bytes(
                0,
                &[
                    (geometry.input as u32).to_le_bytes(),
                    (geometry.output as u32).to_le_bytes(),
                ]
                .concat(),
            )?;
            Ok(Self {
                geometry,
                weight,
                dims,
                bf16,
                f32,
            })
        }

        pub fn geometry(&self) -> DenseGeometry {
            self.geometry
        }

        pub(crate) fn shared_matrix(&self) -> DeviceMatrix {
            self.weight.clone()
        }

        /// Encode one BF16 activation vector against the stored matrix.
        ///
        /// # Safety
        ///
        /// `input` must contain one BF16 vector of declared width. Arbitrary
        /// BF16 bits are memory-safe: kernel addresses depend only on integer
        /// geometry, not activation values. Callers must validate required
        /// finite-value semantics before accepting the output. `output` must
        /// not alias `input` or the stored matrix, and inputs must remain stable
        /// until the batch completes. Public host execution validates first.
        pub unsafe fn encode(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            output: &Buffer,
            precision: OutputPrecision,
        ) -> Result<()> {
            ensure!(
                input.byte_len() >= self.geometry.input_bytes(),
                "dense input buffer is too small"
            );
            ensure!(
                output.byte_len() >= self.geometry.output_bytes(precision)?,
                "dense output buffer is too small"
            );
            let pipeline = match precision {
                OutputPrecision::Bf16 => &self.bf16,
                OutputPrecision::F32 => &self.f32,
            };
            let bindings = [
                self.weight.binding(),
                Binding {
                    buffer: input,
                    offset: 0,
                },
                Binding {
                    buffer: output,
                    offset: 0,
                },
                Binding {
                    buffer: &self.dims,
                    offset: 0,
                },
            ];
            // SAFETY: The caller supplies disjoint, typed buffers. Validated
            // dimensions cover every matrix row and one complete input vector.
            unsafe { batch.dispatch_1d(pipeline, &bindings, self.geometry.grid_threads()?, 64) }
        }

        fn run_bytes(
            &self,
            metal: &Metal,
            input: &[u16],
            precision: OutputPrecision,
        ) -> Result<Vec<u8>> {
            validate_activation(input, self.geometry)?;
            let x = metal.shared_buffer(self.geometry.input_bytes())?;
            x.write_bytes(
                0,
                &input
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let y = metal.shared_buffer(self.geometry.output_bytes(precision)?)?;
            let mut batch = metal.batch()?;
            // SAFETY: The validated input and fresh output are disjoint and
            // retained by the batch through completion.
            unsafe {
                self.encode(&mut batch, &x, &y, precision)?;
            }
            batch.submit().wait().context("dense matvec command")?;
            let mut bytes = vec![0; self.geometry.output_bytes(precision)?];
            y.read_bytes(0, &mut bytes)?;
            Ok(bytes)
        }

        pub fn run_bf16(&self, metal: &Metal, input: &[u16]) -> Result<Vec<u16>> {
            let bytes = self.run_bytes(metal, input, OutputPrecision::Bf16)?;
            Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|w| u16::from_le_bytes(*w))
                .collect())
        }

        /// Explicit FP32 output for consumers that need the wider boundary.
        pub fn run_f32(&self, metal: &Metal, input: &[u16]) -> Result<Vec<f32>> {
            let bytes = self.run_bytes(metal, input, OutputPrecision::F32)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect())
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::DenseMatvec;
#[cfg(target_os = "macos")]
pub(crate) use gpu::DeviceMatrix;

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn synthetic(format: WeightFormat, n: usize, k: usize) -> DirectWeight {
        let mut bytes = Vec::new();
        for row in 0..n {
            match format {
                WeightFormat::Bf16 => {
                    for col in 0..k {
                        let value = ((row * 17 + col * 31) % 101) as f32 / 512.0 - 0.1;
                        bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
                    }
                }
                WeightFormat::Q6K => {
                    for block in 0..k / 256 {
                        let mut wire = [0u8; 210];
                        for (i, byte) in wire[..192].iter_mut().enumerate() {
                            *byte = ((row * 13 + block * 29 + i * 17) % 256) as u8;
                        }
                        for (i, byte) in wire[192..208].iter_mut().enumerate() {
                            *byte = (((row + block + i) % 9) as i8 - 4) as u8;
                        }
                        wire[208..210]
                            .copy_from_slice(&f16::from_f32(0.00390625).to_bits().to_le_bytes());
                        bytes.extend_from_slice(&wire);
                    }
                }
                WeightFormat::Q8_0 => {
                    for block in 0..k / 32 {
                        bytes.extend_from_slice(&f16::from_f32(0.03125).to_bits().to_le_bytes());
                        for i in 0..32 {
                            bytes.push((((row * 11 + block * 13 + i * 7) % 65) as i8 - 32) as u8);
                        }
                    }
                }
            }
        }
        DirectWeight {
            role: "test".into(),
            format,
            shape: vec![n, k],
            bytes,
        }
    }

    fn decoded(weight: &DirectWeight, row: usize, col: usize) -> f64 {
        let k = weight.shape[1];
        match weight.format {
            WeightFormat::Bf16 => {
                let offset = (row * k + col) * 2;
                bf16::from_bits(u16::from_le_bytes(
                    weight.bytes[offset..offset + 2].try_into().unwrap(),
                ))
                .to_f32() as f64
            }
            WeightFormat::Q8_0 => {
                let offset = row * (k / 32 * 34) + col / 32 * 34;
                let scale = f16::from_bits(u16::from_le_bytes(
                    weight.bytes[offset..offset + 2].try_into().unwrap(),
                ))
                .to_f32();
                scale as f64 * (weight.bytes[offset + 2 + col % 32] as i8) as f64
            }
            WeightFormat::Q6K => {
                let offset = row * (k / 256 * 210) + col / 256 * 210;
                let within = col % 256;
                let half = within / 128;
                let quadrant = within % 128 / 32;
                let lane = within % 32;
                let low_at = offset + half * 64 + (quadrant % 2) * 32 + lane;
                let high_at = offset + 128 + half * 32 + lane;
                let low = if quadrant < 2 {
                    weight.bytes[low_at] & 15
                } else {
                    weight.bytes[low_at] >> 4
                };
                let high = (weight.bytes[high_at] >> (quadrant * 2)) & 3;
                let q = (low | (high << 4)) as i32 - 32;
                let scale_at = offset + 192 + half * 8 + (lane / 16) + 2 * quadrant;
                let scale = weight.bytes[scale_at] as i8;
                let d = f16::from_bits(u16::from_le_bytes(
                    weight.bytes[offset + 208..offset + 210].try_into().unwrap(),
                ))
                .to_f32();
                d as f64 * scale as f64 * q as f64
            }
        }
    }

    #[test]
    fn matrix_geometry_rejects_partial_wires() {
        for format in [WeightFormat::Bf16, WeightFormat::Q6K, WeightFormat::Q8_0] {
            let k = match format {
                WeightFormat::Q6K => 256,
                _ => 32,
            };
            let good = synthetic(format, 4, k);
            assert!(DenseGeometry::from_weight(&good).is_ok());
            let mut short = good;
            short.bytes.pop();
            assert!(DenseGeometry::from_weight(&short).is_err());
        }
        assert!(DenseGeometry::from_weight(&synthetic(WeightFormat::Q8_0, 4, 33)).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn encoded_nonfinite_activations_preserve_output_bounds_and_host_rejection() {
        use crate::metal::Metal;
        let metal = Metal::new().unwrap();
        for format in [WeightFormat::Q6K, WeightFormat::Q8_0, WeightFormat::Bf16] {
            let matrix = synthetic(format, 4, 256);
            let kernel = DenseMatvec::new(&metal, &matrix).unwrap();
            for value in [0x7fc0u16, 0x7f80, 0xff80] {
                let words = vec![value; 256];
                assert!(kernel.run_bf16(&metal, &words).is_err());
                let input = metal.shared_buffer(512).unwrap();
                input
                    .write_bytes(0, &value.to_le_bytes().repeat(256))
                    .unwrap();
                let output = metal.shared_buffer(8 + 16).unwrap();
                output.write_bytes(0, &[0x5a; 24]).unwrap();
                let mut batch = metal.batch().unwrap();
                // SAFETY: BF16 input is initialized and buffers are disjoint.
                // Nonfinite values do not participate in kernel address calculations.
                unsafe {
                    kernel
                        .encode(&mut batch, &input, &output, OutputPrecision::Bf16)
                        .unwrap();
                }
                batch.submit().wait().unwrap();
                let mut bytes = [0; 24];
                output.read_bytes(0, &mut bytes).unwrap();
                assert_eq!(&bytes[8..], &[0x5a; 16]);
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matvec_matches_independent_cpu_block_decode() {
        use crate::metal::Metal;
        let metal = Metal::new().unwrap();
        for (format, n, k) in [
            (WeightFormat::Q6K, 4, 2560),
            (WeightFormat::Q6K, 48, 2560),
            (WeightFormat::Q8_0, 16, 320),
            (WeightFormat::Q8_0, 16, 640),
            (WeightFormat::Bf16, 512, 2560),
        ] {
            let matrix = synthetic(format, n, k);
            let kernel = DenseMatvec::new(&metal, &matrix).unwrap();
            let input: Vec<u16> = (0..k)
                .map(|i| bf16::from_f32(((i * 37 % 127) as f32 - 63.0) / 128.0).to_bits())
                .collect();
            let gpu = kernel.run_f32(&metal, &input).unwrap();
            let rounded = kernel.run_bf16(&metal, &input).unwrap();
            assert_eq!(gpu.len(), n);
            for row in 0..n {
                let mut expected = 0.0f64;
                let mut absolute = 0.0f64;
                for (col, &word) in input.iter().enumerate() {
                    let term = decoded(&matrix, row, col) * bf16::from_bits(word).to_f32() as f64;
                    expected += term;
                    absolute += term.abs();
                }
                // Each QMV lane accumulates at most K/32 products and the
                // final SIMD tree adds five stages. This bound covers their
                // FP32 roundoff relative to the absolute dot magnitude.
                let bound = 5e-6 * absolute + 1e-6;
                assert!(
                    (gpu[row] as f64 - expected).abs() <= bound,
                    "{format:?} row {row}: GPU {}, CPU {expected}, bound {bound}",
                    gpu[row]
                );
                assert_eq!(
                    rounded[row],
                    bf16::from_f32(gpu[row]).to_bits(),
                    "{format:?} BF16 output rounding at row {row}"
                );
            }
        }
    }
}

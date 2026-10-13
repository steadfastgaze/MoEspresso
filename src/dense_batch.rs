//! Bounded row-major BF16 projections over the declared direct matrix wires.
//!
//! Quantized matrices remain packed and are decoded into BF16 tiles. Batch
//! projection uses a separate numerical path from single-token `DenseMatvec`.
//! Dispatch preserves the SIMD Max-class reference contract, including its
//! small-row crossovers and split-K fold, without selecting NAX variants.

use crate::kquant::DenseGeometry;
use crate::package::WeightFormat;
use anyhow::{Context, Result, ensure};

pub const METAL_SOURCE: &str = include_str!("../kernels/metal/dense_batch.metal");
pub const MIN_ROWS: usize = 1;
pub const MAX_ROWS: usize = crate::prefill_capacity::MAX_ROWS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseBatchMethod {
    Matvec,
    MultiVector,
    Matmul,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseBatchPlan {
    rows: usize,
    input: usize,
    output: usize,
    partitions: usize,
    partition_width: usize,
    method: DenseBatchMethod,
}

impl DenseBatchPlan {
    pub fn new(geometry: DenseGeometry, rows: usize) -> Result<Self> {
        ensure!(
            (MIN_ROWS..=MAX_ROWS).contains(&rows),
            "dense batch supports 1 through 4096 rows"
        );
        let input = geometry.input();
        let output = geometry.output();
        ensure!(
            geometry.format() != WeightFormat::Bf16 || output > 1,
            "raw BF16 scalar-output GEMV is not supported by the batch kernel"
        );
        ensure!(
            input.is_multiple_of(32),
            "dense batch input width must be a multiple of 32"
        );
        let elements = rows
            .checked_mul(output)
            .context("dense batch output overflow")?;
        ensure!(
            elements <= u32::MAX as usize,
            "dense batch output exceeds Metal uint indexing"
        );
        rows.checked_mul(input)
            .and_then(|v| v.checked_mul(2))
            .context("dense batch input overflow")?;
        // Preserve the reference SIMD Max-class dispatch contract, including
        // the non-NAX small-matrix QMV crossover. Environment-based alternate
        // implementations are not selected by this native numerical path.
        let vector_limit = if input <= 2048 && output <= 2048 {
            18
        } else if input <= 4096 && output <= 4096 {
            12
        } else {
            10
        };
        let method = if rows == 1 {
            DenseBatchMethod::Matvec
        } else if geometry.format() == WeightFormat::Bf16 {
            DenseBatchMethod::Matmul
        } else if (3..=12).contains(&rows) {
            DenseBatchMethod::MultiVector
        } else if rows < vector_limit {
            DenseBatchMethod::Matvec
        } else {
            DenseBatchMethod::Matmul
        };
        let mut partitions = 1;
        // The reference SIMD matmul uses split-K for small output grids. Its
        // partition count is independent of the matrix-fragment tile size.
        if geometry.format() == WeightFormat::Bf16
            && method == DenseBatchMethod::Matmul
            && rows
                .div_ceil(16)
                .checked_mul(output.div_ceil(16))
                .context("dense batch tile overflow")?
                <= 2048
            && input / 16 >= 8
            && input >= rows.max(output)
        {
            let tiles = rows
                .div_ceil(32)
                .checked_mul(output.div_ceil(32))
                .context("dense batch tile overflow")?;
            let ratio = (input / 16) / tiles;
            partitions = ratio
                .checked_next_power_of_two()
                .context("split-K partition overflow")?
                .clamp(2, 32);
        }
        let partition_width = (input / 16 / partitions) * 16;
        ensure!(
            partition_width > 0,
            "dense batch split-K partition is empty"
        );
        elements
            .checked_mul(partitions)
            .and_then(|v| v.checked_mul(4))
            .context("dense batch scratch overflow")?;
        Ok(Self {
            rows,
            input,
            output,
            partitions,
            partition_width,
            method,
        })
    }

    pub fn input_bytes(self) -> usize {
        self.rows * self.input * 2
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn partitions(self) -> usize {
        self.partitions
    }
    pub fn method(self) -> DenseBatchMethod {
        self.method
    }
    pub fn output_bytes(self) -> usize {
        self.rows * self.output * 2
    }
    pub fn scratch_bytes(self) -> usize {
        if self.partitions == 1 {
            4
        } else {
            self.rows * self.output * self.partitions * 4
        }
    }
    fn grid_threads(self) -> Result<usize> {
        if self.method != DenseBatchMethod::Matmul {
            return self
                .output
                .div_ceil(8)
                .checked_mul(64)
                .and_then(|v| {
                    v.checked_mul(if self.method == DenseBatchMethod::Matvec {
                        self.rows
                    } else {
                        1
                    })
                })
                .context("dense batch vector dispatch overflow");
        }
        self.rows
            .div_ceil(64)
            .checked_mul(self.output.div_ceil(64))
            .and_then(|v| v.checked_mul(self.partitions))
            .and_then(|v| v.checked_mul(128))
            .context("dense batch dispatch overflow")
    }
    fn dimensions(self) -> Result<Vec<u8>> {
        [
            self.rows,
            self.output,
            self.input,
            self.partitions,
            self.partition_width,
        ]
        .into_iter()
        .map(|v| Ok(u32::try_from(v)?.to_le_bytes()))
        .collect::<Result<Vec<_>>>()
        .map(|v| v.concat())
    }
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::{DenseBatchMethod, DenseBatchPlan, DenseGeometry, METAL_SOURCE};
    use crate::kquant::{DenseMatvec, DeviceMatrix};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use crate::package::{DirectWeight, WeightFormat};
    use anyhow::{Context, Result, ensure};
    use half::bf16;

    /// One packed direct matrix and its reusable batched projection pipelines.
    pub struct DenseBatch {
        metal: Metal,
        geometry: DenseGeometry,
        weight: DeviceMatrix,
        project: Pipeline,
        matvec: Pipeline,
        multi_vector: Option<Pipeline>,
        sum: Pipeline,
    }

    /// Partition storage shared by sequential projections on one stream.
    pub struct DenseBatchScratch {
        partials: Buffer,
    }

    impl DenseBatchScratch {
        pub fn new(metal: &Metal, bytes: usize) -> Result<Self> {
            Ok(Self {
                partials: metal.shared_buffer(bytes.max(4))?,
            })
        }

        pub fn byte_len(&self) -> usize {
            self.partials.byte_len()
        }
    }

    impl DenseBatch {
        pub fn new(metal: &Metal, weight: &DirectWeight) -> Result<Self> {
            Self::from_matrix(DeviceMatrix::upload(metal, weight)?)
        }

        /// Reuse the exact immutable packed allocation owned by a decode operator.
        pub fn from_matvec(source: &DenseMatvec) -> Result<Self> {
            Self::from_matrix(source.shared_matrix())
        }

        pub(crate) fn from_matrix(weight: DeviceMatrix) -> Result<Self> {
            let metal = weight.metal();
            let geometry = weight.geometry();
            let function = match geometry.format() {
                WeightFormat::Bf16 => "dense_batch_bf16",
                WeightFormat::Q6K => "dense_batch_q6_k",
                WeightFormat::Q8_0 => "dense_batch_q8_0",
            };
            let source = format!("{}\n{}", crate::kquant::METAL_SOURCE, METAL_SOURCE);
            let project = metal.compile_msl_strict(&source, function)?;
            let sum = metal.compile_msl_strict(&source, "dense_batch_split_sum")?;
            let (matvec_name, multi_name) = match geometry.format() {
                WeightFormat::Bf16 => ("dense_batch_qmv_bf16", None),
                WeightFormat::Q6K => ("dense_batch_qmv_q6_k", Some("dense_batch_ext_q6_k")),
                WeightFormat::Q8_0 => ("dense_batch_qmv_q8_0", Some("dense_batch_ext_q8_0")),
            };
            let matvec = metal.compile_msl_strict(&source, matvec_name)?;
            let multi_vector = multi_name
                .map(|name| metal.compile_msl_strict(&source, name))
                .transpose()?;
            ensure!(
                project.max_threads_per_threadgroup() >= 128,
                "dense batch requires 128-thread groups"
            );
            ensure!(
                sum.max_threads_per_threadgroup() >= 128,
                "dense batch sum requires 128-thread groups"
            );
            ensure!(
                matvec.max_threads_per_threadgroup() >= 64
                    && multi_vector
                        .as_ref()
                        .is_none_or(|p| p.max_threads_per_threadgroup() >= 64),
                "dense batch vector kernels require 64-thread groups"
            );
            Ok(Self {
                metal: metal.clone(),
                geometry,
                weight,
                project,
                matvec,
                multi_vector,
                sum,
            })
        }

        pub fn geometry(&self) -> DenseGeometry {
            self.geometry
        }

        #[cfg(test)]
        unsafe fn test_encode_quantized(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            output: &Buffer,
            rows: usize,
            method: DenseBatchMethod,
            scratch: &DenseBatchScratch,
        ) -> Result<()> {
            ensure!(
                self.geometry.format() != WeightFormat::Bf16,
                "comparison requires a quantized matrix"
            );
            ensure!(
                matches!(method, DenseBatchMethod::Matvec | DenseBatchMethod::Matmul),
                "comparison supports only matvec and matmul"
            );
            let mut plan = DenseBatchPlan::new(self.geometry, rows)?;
            ensure!(plan.partitions == 1, "quantized comparison must be unsplit");
            plan.method = method;
            // SAFETY: Only the existing unsplit quantized dispatch changes.
            unsafe { self.encode_plan(batch, input, output, plan, scratch) }
        }

        pub fn scratch_bytes(&self, capacity: usize) -> Result<usize> {
            ensure!(
                (1..=super::MAX_ROWS).contains(&capacity),
                "dense scratch capacity is invalid"
            );
            (1..=capacity).try_fold(4, |bytes, rows| {
                Ok(bytes.max(DenseBatchPlan::new(self.geometry, rows)?.scratch_bytes()))
            })
        }

        /// Encode a contiguous BF16 row matrix without submitting or waiting.
        ///
        /// # Safety
        ///
        /// Input must contain finite BF16 values with the declared row/column
        /// layout. Output must not alias input or the matrix. All prior writes
        /// must precede this batch, and consumers must wait or follow it in the
        /// same ordered batch. The batch retains every bound allocation.
        pub unsafe fn encode_rows(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            output: &Buffer,
            rows: usize,
        ) -> Result<()> {
            let scratch = DenseBatchScratch::new(
                &self.metal,
                DenseBatchPlan::new(self.geometry, rows)?.scratch_bytes(),
            )?;
            // SAFETY: The caller's input contract is unchanged; private scratch
            // is disjoint and retained by the command.
            unsafe { self.encode_rows_into(batch, input, output, rows, &scratch) }
        }

        /// Encode using caller-owned partition storage. Its preceding readers
        /// must occur earlier on the same ordered stream.
        ///
        /// # Safety
        /// The row geometry and aliasing requirements of `encode_rows` apply.
        pub unsafe fn encode_rows_into(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            output: &Buffer,
            rows: usize,
            scratch: &DenseBatchScratch,
        ) -> Result<()> {
            let plan = DenseBatchPlan::new(self.geometry, rows)?;
            // SAFETY: The selected plan preserves the caller's row and aliasing contract.
            unsafe { self.encode_plan(batch, input, output, plan, scratch) }
        }

        unsafe fn encode_plan(
            &self,
            batch: &mut Batch,
            input: &Buffer,
            output: &Buffer,
            plan: DenseBatchPlan,
            scratch: &DenseBatchScratch,
        ) -> Result<()> {
            ensure!(
                input.byte_len() >= plan.input_bytes(),
                "dense batch input buffer is too small"
            );
            ensure!(
                output.byte_len() >= plan.output_bytes(),
                "dense batch output buffer is too small"
            );
            let dimensions = plan.dimensions()?;
            ensure!(
                scratch.byte_len() >= plan.scratch_bytes(),
                "dense partition workspace is too small"
            );
            let scratch = &scratch.partials;
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
                    buffer: scratch,
                    offset: 0,
                },
            ];
            let (pipeline, threads) = match plan.method {
                DenseBatchMethod::Matmul => (&self.project, 128),
                DenseBatchMethod::Matvec => (&self.matvec, 64),
                DenseBatchMethod::MultiVector => (
                    self.multi_vector
                        .as_ref()
                        .context("missing multi-vector kernel")?,
                    64,
                ),
            };
            // SAFETY: Validated extents cover full tiles including guarded edge
            // stores. Fresh scratch/dimensions are retained by the batch.
            unsafe {
                batch.dispatch_1d_with_inline(
                    pipeline,
                    &bindings,
                    &[&dimensions],
                    plan.grid_threads()?,
                    threads,
                )?;
            }
            if plan.partitions > 1 {
                let bindings = [scratch, output].map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: The producer fills every partial before the following
                // ordered encoder sums it. Output elements are independent.
                unsafe {
                    batch.dispatch_1d_with_inline(
                        &self.sum,
                        &bindings,
                        &[&dimensions],
                        plan.rows * plan.output,
                        128,
                    )?;
                }
            }
            Ok(())
        }

        pub fn run_bf16(&self, input: &[u16], rows: usize) -> Result<Vec<u16>> {
            let plan = DenseBatchPlan::new(self.geometry, rows)?;
            ensure!(
                input.len() == plan.input_bytes() / 2,
                "dense batch activation geometry mismatch"
            );
            ensure!(
                input.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "dense batch activations must be finite"
            );
            let x = self.metal.shared_buffer(plan.input_bytes())?;
            x.write_bytes(
                0,
                &input
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let y = self.metal.shared_buffer(plan.output_bytes())?;
            let mut batch = self.metal.batch()?;
            // SAFETY: The validated input and new output have disjoint storage.
            unsafe {
                self.encode_rows(&mut batch, &x, &y, rows)?;
            }
            batch.submit().wait().context("dense batch command")?;
            let mut result = vec![0; plan.output_bytes()];
            y.read_bytes(0, &mut result)?;
            Ok(result
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes(*v))
                .collect())
        }
    }

    #[cfg(test)]
    mod projection_tests;

    #[cfg(test)]
    mod sharing_tests {
        use super::*;

        #[test]
        fn partition_workspace_is_reused_across_ordered_submissions() -> Result<()> {
            let metal = Metal::new()?;
            let make = |scale: f32| {
                DenseBatch::new(
                    &metal,
                    &DirectWeight {
                        role: "synthetic split projection".into(),
                        format: WeightFormat::Bf16,
                        shape: vec![64, 256],
                        bytes: (0..64 * 256)
                            .flat_map(|_| bf16::from_f32(scale).to_bits().to_le_bytes())
                            .collect(),
                    },
                )
            };
            let first = make(1.0 / 256.0)?;
            let second = make(1.0 / 128.0)?;
            let rows = 16;
            assert!(DenseBatchPlan::new(first.geometry, rows)?.partitions() > 1);
            let scratch = DenseBatchScratch::new(&metal, first.scratch_bytes(512)?)?;
            let input = metal.shared_buffer(rows * 256 * 2)?;
            input.write_bytes(
                0,
                &(0..rows * 256)
                    .flat_map(|_| 0x3f80u16.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let a = metal.shared_buffer(rows * 64 * 2)?;
            let b = metal.shared_buffer(rows * 64 * 2)?;
            let mut stream = metal.stream()?;
            unsafe {
                first.encode_rows_into(stream.batch()?, &input, &a, rows, &scratch)?;
            }
            stream.flush()?;
            unsafe {
                second.encode_rows_into(stream.batch()?, &input, &b, rows, &scratch)?;
            }
            assert!(scratch.partials.read_bytes(0, &mut []).is_err());
            stream.wait()?;
            for (buffer, expected) in [(&a, 0x3f80u16), (&b, 0x4000u16)] {
                let mut bytes = vec![0; rows * 64 * 2];
                buffer.read_bytes(0, &mut bytes)?;
                assert!(
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|v| u16::from_le_bytes(*v) == expected)
                );
            }
            Ok(())
        }

        #[test]
        fn shared_matrix_is_identical_and_outlives_decode_owner() -> Result<()> {
            let metal = Metal::new()?;
            let mut bytes = vec![0; 32 * 32 * 2];
            for diagonal in 0..32 {
                bytes[(diagonal * 32 + diagonal) * 2..(diagonal * 32 + diagonal) * 2 + 2]
                    .copy_from_slice(&bf16::from_f32(1.0).to_bits().to_le_bytes());
            }
            let direct = DirectWeight {
                role: "synthetic identity".into(),
                format: WeightFormat::Bf16,
                shape: vec![32, 32],
                bytes,
            };
            let decode = DenseMatvec::new(&metal, &direct)?;
            let batch = DenseBatch::from_matvec(&decode)?;
            assert!(
                batch
                    .weight
                    .binding()
                    .buffer
                    .shares_allocation(decode.shared_matrix().binding().buffer)
            );
            drop(decode);
            drop(direct);
            for rows in [2, 64, 512, 1024, 2048] {
                let input: Vec<_> = (0..rows * 32)
                    .map(|i| bf16::from_f32((i % 17) as f32 - 8.0).to_bits())
                    .collect();
                assert_eq!(batch.run_bf16(&input, rows)?, input);
            }
            let input = metal.shared_buffer(2 * 32 * 2)?;
            input.write_bytes(0, &[0; 2 * 32 * 2])?;
            let output = metal.shared_buffer(2 * 32 * 2)?;
            let mut command = metal.batch()?;
            // SAFETY: Complete zero rows and a distinct output; the command
            // must retain the shared matrix even after its last operator drops.
            unsafe {
                batch.encode_rows(&mut command, &input, &output, 2)?;
            }
            drop(batch);
            command.submit().wait()?;
            let mut result = vec![1; 2 * 32 * 2];
            output.read_bytes(0, &mut result)?;
            assert_eq!(result, vec![0; 2 * 32 * 2]);
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{DenseBatch, DenseBatchScratch};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::DirectWeight;

    fn geometry(format: WeightFormat, n: usize, k: usize) -> DenseGeometry {
        let row = match format {
            WeightFormat::Bf16 => k * 2,
            WeightFormat::Q6K => k / 256 * 210,
            WeightFormat::Q8_0 => k / 32 * 34,
        };
        DenseGeometry::from_weight(&DirectWeight {
            role: "synthetic".into(),
            format,
            shape: vec![n, k],
            bytes: vec![0; n * row],
        })
        .unwrap()
    }

    #[test]
    fn router_split_k_matches_reference_partition_rule() {
        let g = geometry(WeightFormat::Bf16, 512, 2560);
        let first = DenseBatchPlan::new(g, 64).unwrap();
        assert_eq!((first.partitions, first.partition_width), (8, 320));
        let wide = DenseBatchPlan::new(g, 512).unwrap();
        assert_eq!((wide.partitions, wide.partition_width), (2, 1280));
        for capacity in [2048, 4096] {
            let wide = DenseBatchPlan::new(g, capacity).unwrap();
            assert_eq!((wide.partitions, wide.partition_width), (1, 2560));
            let (peak_rows, maximum) = (1..=capacity)
                .map(|rows| (rows, DenseBatchPlan::new(g, rows).unwrap().scratch_bytes()))
                .max_by_key(|&(_, bytes)| bytes)
                .unwrap();
            assert_eq!((peak_rows, maximum), (1024, 4_194_304));
        }
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE; checks headers and CPU plans without Metal"]
    fn package_prefill_dense_scratch_inventory_is_accounted() -> Result<()> {
        let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let mut geometries = Vec::new();
        for header in package.direct_headers()? {
            // Embedding and the one-row terminal head do not use prefill GEMM.
            // Narrow convolution parameters are not dense projection matrices.
            if matches!(header.module_path.as_str(), "embedding" | "lm_head")
                || header.shape.len() != 2
                || !header.shape[1].is_multiple_of(32)
            {
                continue;
            }
            if geometries.iter().any(|g: &DenseGeometry| {
                g.format() == header.format
                    && g.output() == header.shape[0]
                    && g.input() == header.shape[1]
            }) {
                continue;
            }
            geometries.push(DenseGeometry::from_weight(&DirectWeight {
                role: header.module_path,
                format: header.format,
                shape: header.shape,
                bytes: vec![0; header.bytes],
            })?);
        }
        ensure!(
            geometries.iter().any(|g| {
                g.format() == WeightFormat::Bf16 && g.output() == 512 && g.input() == 2560
            }),
            "package dense inventory has no model router"
        );
        for (capacity, accounted) in [(512, 2_097_152), (2048, 4_194_304), (4096, 4_194_304)] {
            let maximum = geometries.iter().try_fold(4, |maximum, &g| {
                (1..=capacity).try_fold(maximum, |maximum, rows| {
                    Ok::<_, anyhow::Error>(
                        maximum.max(DenseBatchPlan::new(g, rows)?.scratch_bytes()),
                    )
                })
            })?;
            assert_eq!(maximum, accounted);
            println!(
                "prefill_dense_inventory capacity={capacity} geometries={} scratch_bytes={maximum}",
                geometries.len()
            );
        }
        Ok(())
    }

    #[test]
    fn quantized_gemm_is_unsplit_and_rows_are_bounded() {
        for format in [WeightFormat::Q6K, WeightFormat::Q8_0] {
            let g = geometry(format, 4, 256);
            for rows in [64, 65, 512, 1024, 2048, 2049, 4096] {
                let p = DenseBatchPlan::new(g, rows).unwrap();
                assert_eq!(p.partitions, 1);
                assert_eq!(p.scratch_bytes(), 4);
                assert_eq!(p.output_bytes(), rows * 4 * 2);
            }
            for rows in [0, 4097, usize::MAX] {
                assert!(DenseBatchPlan::new(g, rows).is_err());
            }
        }
    }

    #[test]
    fn tail_dispatch_preserves_projection_shape_crossovers() {
        let narrow = geometry(WeightFormat::Q8_0, 4, 320);
        for rows in [1, 2, 13, 16] {
            assert_eq!(
                DenseBatchPlan::new(narrow, rows).unwrap().method(),
                DenseBatchMethod::Matvec
            );
        }
        for rows in [3, 4, 8, 12] {
            assert_eq!(
                DenseBatchPlan::new(narrow, rows).unwrap().method(),
                DenseBatchMethod::MultiVector
            );
        }
        assert_eq!(
            DenseBatchPlan::new(narrow, 18).unwrap().method(),
            DenseBatchMethod::Matmul
        );
        let wide = geometry(WeightFormat::Q6K, 64, 4352);
        assert_eq!(
            DenseBatchPlan::new(wide, 13).unwrap().method(),
            DenseBatchMethod::Matmul
        );
    }
}

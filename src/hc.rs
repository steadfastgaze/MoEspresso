//! One-row Qwen gated-residual read and folded write at the native BF16 boundary.

#[cfg(target_os = "macos")]
mod gpu {
    use anyhow::{Context, Result, ensure};
    use half::bf16;

    use crate::kquant::DenseMatvec;
    use crate::metal::{Batch, Binding, Buffer, InFlight, Metal, Pipeline};
    use crate::package::{DirectWeight, WeightFormat};

    const SOURCE: &str = include_str!("../kernels/metal/hc.metal");
    const BRANCHES: usize = 4;
    const WIDTH: usize = 2560;
    const EXPANDED: usize = BRANCHES * WIDTH;
    const LOWRANK: usize = 320;

    fn checked_bf16(bytes: &[u8], words: usize, name: &str) -> Result<()> {
        ensure!(
            bytes.len() == words * 2,
            "{name} BF16 byte count differs from HC geometry"
        );
        ensure!(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .all(|word| bf16::from_bits(u16::from_le_bytes(*word)).is_finite()),
            "{name} contains nonfinite BF16 values"
        );
        Ok(())
    }

    fn weight_shape(
        weight: &DirectWeight,
        format: WeightFormat,
        shape: [usize; 2],
        name: &str,
    ) -> Result<()> {
        ensure!(
            weight.format == format && weight.shape == shape,
            "{name} has the wrong packed format or matrix shape"
        );
        Ok(())
    }

    struct Scratch {
        residual: Buffer,
        pending_output: Buffer,
        pending_injection: Buffer,
        normalized: Buffer,
        updated: Buffer,
        projected_down: Buffer,
        projected_injection: Buffer,
        lowrank: Buffer,
        injection: Buffer,
        projected_up: Buffer,
        mixed: Buffer,
    }

    pub struct HcStages {
        pub updated_bf16: Vec<u8>,
        pub normalized_bf16: Vec<u8>,
        pub projected_down_bf16: Vec<u8>,
        pub projected_injection_bf16: Option<Vec<u8>>,
        pub lowrank_bf16: Vec<u8>,
        pub injection_bf16: Option<Vec<u8>>,
        pub projected_up_bf16: Vec<u8>,
        pub mixed_bf16: Vec<u8>,
    }

    pub struct HcOutput {
        pub mixed_bf16: Vec<u8>,
        pub updated_bf16: Vec<u8>,
        pub injection_bf16: Option<Vec<u8>>,
    }

    pub(crate) struct HcDeviceOutput {
        pub mixed: Buffer,
        pub updated: Buffer,
        pub injection: Option<Buffer>,
    }

    pub struct PendingHc {
        flight: InFlight,
        updated: Buffer,
        normalized: Buffer,
        projected_down: Buffer,
        projected_injection: Buffer,
        lowrank: Buffer,
        injection: Buffer,
        projected_up: Buffer,
        mixed: Buffer,
        has_injection: bool,
    }

    impl PendingHc {
        fn read(buffer: &Buffer) -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.byte_len()];
            buffer.read_bytes(0, &mut bytes)?;
            Ok(bytes)
        }

        pub fn wait(self) -> Result<HcOutput> {
            self.flight.wait()?;
            Ok(HcOutput {
                mixed_bf16: Self::read(&self.mixed)?,
                updated_bf16: Self::read(&self.updated)?,
                injection_bf16: self
                    .has_injection
                    .then(|| Self::read(&self.injection))
                    .transpose()?,
            })
        }

        pub fn wait_stages(self) -> Result<HcStages> {
            self.flight.wait()?;
            Ok(HcStages {
                updated_bf16: Self::read(&self.updated)?,
                normalized_bf16: Self::read(&self.normalized)?,
                projected_down_bf16: Self::read(&self.projected_down)?,
                projected_injection_bf16: self
                    .has_injection
                    .then(|| Self::read(&self.projected_injection))
                    .transpose()?,
                lowrank_bf16: Self::read(&self.lowrank)?,
                injection_bf16: self
                    .has_injection
                    .then(|| Self::read(&self.injection))
                    .transpose()?,
                projected_up_bf16: Self::read(&self.projected_up)?,
                mixed_bf16: Self::read(&self.mixed)?,
            })
        }
    }

    pub struct HcBlock {
        metal: Metal,
        norm: Pipeline,
        front: Pipeline,
        epilogue: Pipeline,
        down: DenseMatvec,
        up: DenseMatvec,
        injection: Option<DenseMatvec>,
        norm_weight: Buffer,
        eps: Buffer,
        has_injection: Buffer,
        scratch: Scratch,
    }

    pub(crate) struct HcSharedWeights {
        pub down: crate::kquant::DeviceMatrix,
        pub up: crate::kquant::DeviceMatrix,
        pub injection: Option<crate::kquant::DeviceMatrix>,
        pub norm: Buffer,
        pub epsilon: Buffer,
    }

    impl HcBlock {
        pub(crate) fn shared_weights(&self) -> HcSharedWeights {
            HcSharedWeights {
                down: self.down.shared_matrix(),
                up: self.up.shared_matrix(),
                injection: self.injection.as_ref().map(DenseMatvec::shared_matrix),
                norm: self.norm_weight.clone(),
                epsilon: self.eps.clone(),
            }
        }
        pub fn new(
            metal: &Metal,
            norm_weight_bf16: &[u8],
            down_weight: &DirectWeight,
            up_weight: &DirectWeight,
            injection_weight: Option<&DirectWeight>,
            eps: f32,
        ) -> Result<Self> {
            checked_bf16(norm_weight_bf16, EXPANDED, "HC norm weight")?;
            ensure!(
                eps.is_finite() && eps > 0.0,
                "HC epsilon must be finite and positive"
            );
            weight_shape(
                down_weight,
                WeightFormat::Q6K,
                [LOWRANK, EXPANDED],
                "HC lowrank down",
            )?;
            weight_shape(
                up_weight,
                WeightFormat::Q8_0,
                [EXPANDED, LOWRANK],
                "HC lowrank up",
            )?;
            if let Some(weight) = injection_weight {
                weight_shape(
                    weight,
                    WeightFormat::Q6K,
                    [BRANCHES, EXPANDED],
                    "HC injection",
                )?;
            }
            let norm = metal.compile_msl_strict(SOURCE, "hc_norm")?;
            let front = metal.compile_msl_strict(SOURCE, "hc_front")?;
            let epilogue = metal.compile_msl_strict(SOURCE, "hc_epilogue")?;
            ensure!(
                norm.max_threads_per_threadgroup() >= 640
                    && front.max_threads_per_threadgroup() >= 64
                    && epilogue.max_threads_per_threadgroup() >= 64,
                "Metal cannot dispatch released HC threadgroup geometry"
            );
            let down = DenseMatvec::new(metal, down_weight)?;
            let up = DenseMatvec::new(metal, up_weight)?;
            let injection = injection_weight
                .map(|w| DenseMatvec::new(metal, w))
                .transpose()?;
            let norm_weight = metal.shared_buffer(EXPANDED * 2)?;
            norm_weight.write_bytes(0, norm_weight_bf16)?;
            let eps_buffer = metal.shared_buffer(4)?;
            eps_buffer.write_bytes(0, &eps.to_le_bytes())?;
            let has_injection = metal.shared_buffer(4)?;
            has_injection.write_bytes(0, &(injection.is_some() as i32).to_le_bytes())?;
            let scratch = Scratch {
                residual: metal.shared_buffer(EXPANDED * 2)?,
                pending_output: metal.shared_buffer(WIDTH * 2)?,
                pending_injection: metal.shared_buffer(BRANCHES * 2)?,
                normalized: metal.shared_buffer(EXPANDED * 2)?,
                updated: metal.shared_buffer(EXPANDED * 2)?,
                projected_down: metal.shared_buffer(LOWRANK * 2)?,
                projected_injection: metal.shared_buffer(BRANCHES * 2)?,
                lowrank: metal.shared_buffer(LOWRANK * 2)?,
                injection: metal.shared_buffer(BRANCHES * 2)?,
                projected_up: metal.shared_buffer(EXPANDED * 2)?,
                mixed: metal.shared_buffer(WIDTH * 2)?,
            };
            Ok(Self {
                metal: metal.clone(),
                norm,
                front,
                epilogue,
                down,
                up,
                injection,
                norm_weight,
                eps: eps_buffer,
                has_injection,
                scratch,
            })
        }

        /// Read a single BF16 HC row, optionally folding a pending block write.
        pub fn submit(
            &mut self,
            residual_bf16: &[u8],
            pending: Option<(&[u8], &[u8])>,
        ) -> Result<PendingHc> {
            let mut batch = self.metal.batch()?;
            self.encode_host(&mut batch, residual_bf16, pending)?;
            Ok(PendingHc {
                flight: batch.submit(),
                updated: self.scratch.updated.clone(),
                normalized: self.scratch.normalized.clone(),
                projected_down: self.scratch.projected_down.clone(),
                projected_injection: self.scratch.projected_injection.clone(),
                lowrank: self.scratch.lowrank.clone(),
                injection: self.scratch.injection.clone(),
                projected_up: self.scratch.projected_up.clone(),
                mixed: self.scratch.mixed.clone(),
                has_injection: self.injection.is_some(),
            })
        }

        pub(crate) fn encode_host(
            &self,
            batch: &mut Batch,
            residual_bf16: &[u8],
            pending: Option<(&[u8], &[u8])>,
        ) -> Result<HcDeviceOutput> {
            checked_bf16(residual_bf16, EXPANDED, "HC residual")?;
            if let Some((output, injection)) = pending {
                checked_bf16(output, WIDTH, "HC pending output")?;
                checked_bf16(injection, BRANCHES, "HC pending injection")?;
                self.scratch.pending_output.write_bytes(0, output)?;
                self.scratch.pending_injection.write_bytes(0, injection)?;
            }
            self.scratch.residual.write_bytes(0, residual_bf16)?;
            // SAFETY: The uploaded host rows have the fixed HC geometry and
            // finite BF16 elements checked above.
            unsafe {
                self.encode_device(
                    batch,
                    &self.scratch.residual,
                    pending.map(|_| {
                        (
                            &self.scratch.pending_output,
                            &self.scratch.pending_injection,
                        )
                    }),
                )
            }
        }

        /// Encode after an ordered device producer initializes the inputs.
        /// The caller validates finite-value semantics before publication.
        pub(crate) unsafe fn encode_device(
            &self,
            batch: &mut Batch,
            residual: &Buffer,
            pending: Option<(&Buffer, &Buffer)>,
        ) -> Result<HcDeviceOutput> {
            ensure!(
                residual.byte_len() == EXPANDED * 2,
                "HC residual buffer width differs"
            );
            if let Some((output, injection)) = pending {
                ensure!(
                    output.byte_len() == WIDTH * 2,
                    "HC pending output buffer width differs"
                );
                ensure!(
                    injection.byte_len() == BRANCHES * 2,
                    "HC pending injection buffer width differs"
                );
            }
            // SAFETY: Exact single-row extents were checked above.
            unsafe { self.encode_device_row(batch, residual, pending, 0) }
        }

        /// Encode one row from a contiguous residual matrix, including the
        /// matching pending output and injection when present.
        pub(crate) unsafe fn encode_device_row(
            &self,
            batch: &mut Batch,
            residual: &Buffer,
            pending: Option<(&Buffer, &Buffer)>,
            row: usize,
        ) -> Result<HcDeviceOutput> {
            let end = row.checked_add(1).context("HC row offset overflow")?;
            let required =
                |width: usize| end.checked_mul(width * 2).context("HC row extent overflow");
            ensure!(
                residual.byte_len() >= required(EXPANDED)?,
                "HC residual row exceeds buffer"
            );
            if let Some((output, injection)) = pending {
                ensure!(
                    output.byte_len() >= required(WIDTH)?
                        && injection.byte_len() >= required(BRANCHES)?,
                    "HC pending row exceeds buffer"
                );
            }
            let has_pending = (pending.is_some() as i32).to_le_bytes();
            let (pending_output, pending_injection) = pending.unwrap_or((
                &self.scratch.pending_output,
                &self.scratch.pending_injection,
            ));
            let down = self.down.shared_matrix();
            let up = self.up.shared_matrix();
            let injection = self.injection.as_ref().map(DenseMatvec::shared_matrix);
            // SAFETY: Fixed buffer lengths match the 4x2560 HC kernel.
            // Addressing does not depend on values. Host rows are validated
            // by encode_host; an earlier device producer is checked after
            // completion and before its output is consumed on the CPU.
            // Batch retains buffers and encodes the dependencies in order.
            unsafe {
                batch.dispatch_1d_with_inline(
                    &self.norm,
                    &[
                        Binding {
                            buffer: residual,
                            offset: row * EXPANDED * 2,
                        },
                        Binding {
                            buffer: &self.norm_weight,
                            offset: 0,
                        },
                        Binding {
                            buffer: pending_output,
                            offset: if pending.is_some() {
                                row * WIDTH * 2
                            } else {
                                0
                            },
                        },
                        Binding {
                            buffer: pending_injection,
                            offset: if pending.is_some() {
                                row * BRANCHES * 2
                            } else {
                                0
                            },
                        },
                        Binding {
                            buffer: &self.scratch.normalized,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.updated,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.eps,
                            offset: 0,
                        },
                    ],
                    &[&has_pending],
                    BRANCHES * 640,
                    640,
                )?;
                batch.dispatch_1d(
                    &self.front,
                    &[
                        Binding {
                            buffer: &self.scratch.normalized,
                            offset: 0,
                        },
                        down.binding(),
                        injection.as_ref().unwrap_or(&down).binding(),
                        Binding {
                            buffer: &self.scratch.lowrank,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.injection,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.has_injection,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.projected_down,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.projected_injection,
                            offset: 0,
                        },
                    ],
                    if injection.is_some() {
                        162 * 64
                    } else {
                        160 * 64
                    },
                    64,
                )?;
                batch.dispatch_1d(
                    &self.epilogue,
                    &[
                        Binding {
                            buffer: &self.scratch.lowrank,
                            offset: 0,
                        },
                        up.binding(),
                        Binding {
                            buffer: &self.scratch.normalized,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.mixed,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.projected_up,
                            offset: 0,
                        },
                    ],
                    (WIDTH / 2) * 64,
                    64,
                )?;
            }
            Ok(HcDeviceOutput {
                mixed: self.scratch.mixed.clone(),
                updated: self.scratch.updated.clone(),
                injection: self
                    .injection
                    .as_ref()
                    .map(|_| self.scratch.injection.clone()),
            })
        }

        pub fn run(
            &mut self,
            residual_bf16: &[u8],
            pending: Option<(&[u8], &[u8])>,
        ) -> Result<HcOutput> {
            self.submit(residual_bf16, pending)?
                .wait()
                .context("HC read command")
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{HcBlock, HcOutput, HcStages, PendingHc};

#[cfg(all(test, target_os = "macos"))]
mod row_tests {
    use super::HcBlock;
    use crate::metal::{Buffer, Metal};
    use crate::package::{DirectWeight, WeightFormat};
    use anyhow::Result;

    #[test]
    fn final_row_bindings_include_pending_operands_without_copy() -> Result<()> {
        let metal = Metal::new()?;
        let down = DirectWeight {
            role: "synthetic".into(),
            format: WeightFormat::Q6K,
            shape: vec![320, 10240],
            bytes: vec![0; 320 * (10240 / 256) * 210],
        };
        let up = DirectWeight {
            role: "synthetic".into(),
            format: WeightFormat::Q8_0,
            shape: vec![10240, 320],
            bytes: vec![0; 10240 * (320 / 32) * 34],
        };
        let norm: Vec<_> = (0..10240).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
        let hc = HcBlock::new(&metal, &norm, &down, &up, None, 1e-6)?;
        let input = |width: usize, seed: usize| -> Result<Buffer> {
            let bytes: Vec<_> = (0..width * 3)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 7 + seed) % 97) as f32 / 32.0 - 1.0)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect();
            let buffer = metal.shared_buffer(bytes.len())?;
            buffer.write_bytes(0, &bytes)?;
            Ok(buffer)
        };
        let residual = input(10240, 3)?;
        let output = input(2560, 11)?;
        let injection = input(4, 21)?;
        let mut stream = metal.stream()?;
        let actual = unsafe {
            hc.encode_device_row(stream.batch()?, &residual, Some((&output, &injection)), 2)?
        };
        assert!(actual.mixed.read_bytes(0, &mut []).is_err());
        stream.wait()?;
        let mut actual_bytes = vec![0; 2560 * 2];
        actual.mixed.read_bytes(0, &mut actual_bytes)?;
        let row = |buffer: &Buffer, width: usize| -> Result<Buffer> {
            let mut bytes = vec![0; width * 2];
            buffer.read_bytes(width * 2 * 2, &mut bytes)?;
            let result = metal.shared_buffer(bytes.len())?;
            result.write_bytes(0, &bytes)?;
            Ok(result)
        };
        let r = row(&residual, 10240)?;
        let o = row(&output, 2560)?;
        let i = row(&injection, 4)?;
        let expected = unsafe { hc.encode_device(stream.batch()?, &r, Some((&o, &i)))? };
        stream.wait()?;
        let mut expected_bytes = vec![0; 2560 * 2];
        expected.mixed.read_bytes(0, &mut expected_bytes)?;
        assert_eq!(actual_bytes, expected_bytes);
        assert!(
            unsafe { hc.encode_device_row(stream.batch()?, &residual, Some((&o, &i)), 2) }.is_err()
        );
        assert!(
            unsafe { hc.encode_device_row(stream.batch()?, &residual, None, usize::MAX) }.is_err()
        );
        Ok(())
    }
}

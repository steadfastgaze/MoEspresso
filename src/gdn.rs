//! One-token Qwen gated-delta cell with caller-owned recurrent state.

#[cfg(target_os = "macos")]
mod gpu {
    use anyhow::{Context, Result, ensure};
    use half::bf16;

    use crate::kquant::{DenseMatvec, OutputPrecision};
    use crate::metal::{Batch, Binding, Buffer, InFlight, Metal, Pipeline};
    use crate::package::{DirectWeight, WeightFormat};

    const SOURCE: &str = include_str!("../kernels/metal/gdn.metal");
    const HIDDEN: usize = 2560;
    const CONV: usize = 10240;
    const KEY: usize = 2048;
    const VALUE: usize = 6144;
    const HEADS: usize = 48;
    const HEAD_WIDTH: usize = 128;
    const RECURRENT_BYTES: usize = HEADS * HEAD_WIDTH * HEAD_WIDTH * 4;
    const CONV_BYTES: usize = 3 * CONV * 2;

    fn raw(weight: &DirectWeight, shape: &[usize], name: &str) -> Result<()> {
        ensure!(
            weight.format == WeightFormat::Bf16 && weight.shape == shape,
            "{name} must have the released BF16 shape"
        );
        finite_bf16(&weight.bytes, shape.iter().product(), name)
    }

    fn q6(weight: &DirectWeight, shape: [usize; 2], name: &str) -> Result<()> {
        ensure!(
            weight.format == WeightFormat::Q6K && weight.shape == shape,
            "{name} must have the released Q6_K shape"
        );
        Ok(())
    }

    fn finite_bf16(bytes: &[u8], words: usize, name: &str) -> Result<()> {
        ensure!(
            bytes.len() == words * 2,
            "{name} BF16 byte count differs from geometry"
        );
        ensure!(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .all(|w| bf16::from_bits(u16::from_le_bytes(*w)).is_finite()),
            "{name} contains nonfinite BF16 values"
        );
        Ok(())
    }

    fn uploaded(metal: &Metal, bytes: &[u8]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(bytes.len())?;
        buffer.write_bytes(0, bytes)?;
        Ok(buffer)
    }

    pub struct GdnWeights<'a> {
        pub qkv: &'a DirectWeight,
        pub gate: &'a DirectWeight,
        pub beta: &'a DirectWeight,
        pub decay: &'a DirectWeight,
        pub output: &'a DirectWeight,
        pub conv: &'a DirectWeight,
        pub a_log: &'a DirectWeight,
        pub dt_bias: &'a DirectWeight,
        pub norm: &'a DirectWeight,
        pub eps: f32,
    }

    /// Persistent state at one completed physical token frontier.
    pub struct GdnState {
        conv: Buffer,
        recurrent: Buffer,
        frontier: u64,
        valid: bool,
    }

    impl GdnState {
        pub fn zero(metal: &Metal) -> Result<Self> {
            Self::from_bytes(metal, &vec![0; CONV_BYTES], &vec![0; RECURRENT_BYTES], 0)
        }

        pub fn from_bytes(
            metal: &Metal,
            conv_bf16: &[u8],
            recurrent_f32: &[u8],
            frontier: u64,
        ) -> Result<Self> {
            Self::validate_parts(conv_bf16, recurrent_f32)?;
            Ok(Self {
                conv: uploaded(metal, conv_bf16)?,
                recurrent: uploaded(metal, recurrent_f32)?,
                frontier,
                valid: true,
            })
        }

        /// Validate borrowed state payloads without copying or uploading them.
        /// The caller separately checks the frontier against its session.
        pub(crate) fn validate_parts(conv_bf16: &[u8], recurrent_f32: &[u8]) -> Result<()> {
            finite_bf16(conv_bf16, 3 * CONV, "GDN convolution state")?;
            ensure!(
                recurrent_f32.len() == RECURRENT_BYTES,
                "GDN recurrent state byte count differs from geometry"
            );
            ensure!(
                recurrent_f32
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|w| f32::from_le_bytes(*w).is_finite()),
                "GDN recurrent state contains nonfinite FP32 values"
            );
            Ok(())
        }

        pub fn frontier(&self) -> Result<u64> {
            ensure!(self.valid, "GDN state has an unpublished or failed update");
            Ok(self.frontier)
        }

        pub(crate) fn poison(&mut self) {
            self.valid = false;
        }

        /// Publish only after the caller has waited and checked downstream
        /// rows. Submitted in-place state writes cannot be rolled back.
        pub(crate) fn publish_after_wait(&mut self) -> Result<u64> {
            self.publish_rows_after_wait(1)
        }

        pub(crate) fn publish_rows_after_wait(&mut self, rows: usize) -> Result<u64> {
            ensure!(!self.valid && rows > 0, "GDN update is not pending");
            self.conv.read_bytes(0, &mut [])?;
            self.recurrent.read_bytes(0, &mut [])?;
            let frontier = self
                .frontier
                .checked_add(u64::try_from(rows)?)
                .context("GDN frontier overflow")?;
            self.frontier = frontier;
            self.valid = true;
            Ok(frontier)
        }

        /// Exchange a separately encoded convolution history. The recurrent
        /// rows were updated in place by their exclusive device owners.
        pub(crate) fn stage_chunk_history(&mut self, next_conv: &mut Buffer) -> Result<()> {
            ensure!(
                self.valid && next_conv.byte_len() == CONV_BYTES,
                "GDN chunk history is invalid"
            );
            std::mem::swap(&mut self.conv, next_conv);
            self.valid = false;
            Ok(())
        }

        pub(crate) fn buffers(&self) -> Result<(&Buffer, &Buffer)> {
            self.frontier()?;
            Ok((&self.conv, &self.recurrent))
        }

        /// Build a candidate from completed producers on one Metal device.
        /// The crate caller must have successfully waited for the batch that
        /// filled both buffers. Encoding that batch checks their device identity.
        pub(crate) fn from_buffers(conv: Buffer, recurrent: Buffer, frontier: u64) -> Result<Self> {
            ensure!(
                conv.byte_len() == CONV_BYTES && recurrent.byte_len() == RECURRENT_BYTES,
                "GDN candidate state buffer geometry differs"
            );
            conv.read_bytes(0, &mut [])?;
            recurrent.read_bytes(0, &mut [])?;
            Ok(Self {
                conv,
                recurrent,
                frontier,
                valid: true,
            })
        }

        pub fn snapshot(&self) -> Result<(Vec<u8>, Vec<u8>)> {
            ensure!(self.valid, "GDN state has an unpublished or failed update");
            let mut conv = vec![0; CONV_BYTES];
            let mut recurrent = vec![0; RECURRENT_BYTES];
            self.conv.read_bytes(0, &mut conv)?;
            self.recurrent.read_bytes(0, &mut recurrent)?;
            Ok((conv, recurrent))
        }
    }

    struct Scratch {
        hidden: Buffer,
        qkv: Buffer,
        gate: Buffer,
        beta_logits: Buffer,
        decay_logits: Buffer,
        query: Buffer,
        key: Buffer,
        value: Buffer,
        beta: Buffer,
        decay: Buffer,
        recurrence: Buffer,
        norm_gate: Buffer,
        output: Buffer,
    }

    pub struct GdnStages {
        pub qkv_bf16: Vec<u8>,
        pub gate_bf16: Vec<u8>,
        pub beta_logits_bf16: Vec<u8>,
        pub decay_logits_bf16: Vec<u8>,
        pub query_bf16: Vec<u8>,
        pub key_bf16: Vec<u8>,
        pub value_bf16: Vec<u8>,
        pub beta_bf16: Vec<u8>,
        pub decay_f32: Vec<u8>,
        pub recurrence_bf16: Vec<u8>,
        pub norm_gate_bf16: Vec<u8>,
        pub output_bf16: Vec<u8>,
        pub conv_state_bf16: Vec<u8>,
        pub recurrent_state_f32: Vec<u8>,
        pub frontier: u64,
    }

    pub struct GdnOutput {
        pub output_bf16: Vec<u8>,
        pub frontier: u64,
    }

    pub(crate) struct GdnDeviceOutput {
        pub output: Buffer,
    }

    pub struct PendingGdn<'a> {
        flight: Option<InFlight>,
        state: &'a mut GdnState,
        scratch: &'a Scratch,
        settled: bool,
    }

    impl PendingGdn<'_> {
        fn complete(&mut self) -> Result<()> {
            let flight = self
                .flight
                .take()
                .context("GDN submission already completed")?;
            if let Err(err) = flight.wait() {
                self.state.valid = false;
                return Err(err).context("GDN command failed");
            }
            Ok(())
        }

        fn read(buffer: &Buffer) -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.byte_len()];
            buffer.read_bytes(0, &mut bytes)?;
            Ok(bytes)
        }

        fn publish(&mut self) -> Result<u64> {
            let frontier = self
                .state
                .frontier
                .checked_add(1)
                .context("GDN frontier overflow")?;
            self.state.frontier = frontier;
            self.state.valid = true;
            self.settled = true;
            Ok(frontier)
        }

        pub fn wait(mut self) -> Result<GdnOutput> {
            self.complete()?;
            let output_bf16 = Self::read(&self.scratch.output)?;
            Ok(GdnOutput {
                output_bf16,
                frontier: self.publish()?,
            })
        }

        pub fn wait_stages(mut self) -> Result<GdnStages> {
            self.complete()?;
            let qkv_bf16 = Self::read(&self.scratch.qkv)?;
            let gate_bf16 = Self::read(&self.scratch.gate)?;
            let beta_logits_bf16 = Self::read(&self.scratch.beta_logits)?;
            let decay_logits_bf16 = Self::read(&self.scratch.decay_logits)?;
            let query_bf16 = Self::read(&self.scratch.query)?;
            let key_bf16 = Self::read(&self.scratch.key)?;
            let value_bf16 = Self::read(&self.scratch.value)?;
            let beta_bf16 = Self::read(&self.scratch.beta)?;
            let decay_f32 = Self::read(&self.scratch.decay)?;
            let recurrence_bf16 = Self::read(&self.scratch.recurrence)?;
            let norm_gate_bf16 = Self::read(&self.scratch.norm_gate)?;
            let output_bf16 = Self::read(&self.scratch.output)?;
            let mut conv_state_bf16 = vec![0; CONV_BYTES];
            let mut recurrent_state_f32 = vec![0; RECURRENT_BYTES];
            self.state.conv.read_bytes(0, &mut conv_state_bf16)?;
            self.state
                .recurrent
                .read_bytes(0, &mut recurrent_state_f32)?;
            Ok(GdnStages {
                qkv_bf16,
                gate_bf16,
                beta_logits_bf16,
                decay_logits_bf16,
                query_bf16,
                key_bf16,
                value_bf16,
                beta_bf16,
                decay_f32,
                recurrence_bf16,
                norm_gate_bf16,
                output_bf16,
                conv_state_bf16,
                recurrent_state_f32,
                frontier: self.publish()?,
            })
        }
    }

    impl Drop for PendingGdn<'_> {
        fn drop(&mut self) {
            if !self.settled {
                self.state.valid = false;
            }
        }
    }

    pub struct GdnCell {
        metal: Metal,
        qkv: DenseMatvec,
        gate: DenseMatvec,
        beta: DenseMatvec,
        decay: DenseMatvec,
        output: DenseMatvec,
        prepare: Pipeline,
        recurrence: Pipeline,
        norm_gate: Pipeline,
        conv_weight: Buffer,
        a_log: Buffer,
        dt_bias: Buffer,
        norm_weight: Buffer,
        eps: Buffer,
        scratch: Scratch,
        stacked_front: Option<input_front::StackedFront>,
    }

    pub(crate) struct GdnSharedWeights {
        pub matrices: [crate::kquant::DeviceMatrix; 5],
        pub conv: Buffer,
        pub a_log: Buffer,
        pub dt_bias: Buffer,
        pub norm: Buffer,
        pub epsilon: Buffer,
    }

    impl GdnCell {
        pub(crate) fn shared_weights(&self) -> GdnSharedWeights {
            GdnSharedWeights {
                matrices: [&self.qkv, &self.gate, &self.beta, &self.decay, &self.output]
                    .map(DenseMatvec::shared_matrix),
                conv: self.conv_weight.clone(),
                a_log: self.a_log.clone(),
                dt_bias: self.dt_bias.clone(),
                norm: self.norm_weight.clone(),
                epsilon: self.eps.clone(),
            }
        }
        pub fn new(metal: &Metal, weights: GdnWeights<'_>) -> Result<Self> {
            q6(weights.qkv, [CONV, HIDDEN], "GDN qkv")?;
            q6(weights.gate, [VALUE, HIDDEN], "GDN gate")?;
            q6(weights.beta, [HEADS, HIDDEN], "GDN beta")?;
            q6(weights.decay, [HEADS, HIDDEN], "GDN decay")?;
            q6(weights.output, [HIDDEN, VALUE], "GDN output")?;
            ensure!(
                weights.conv.shape == [CONV, 1, 4] || weights.conv.shape == [CONV, 4, 1],
                "GDN convolution weight must have four taps per channel"
            );
            ensure!(
                weights.conv.format == WeightFormat::Bf16
                    && weights.conv.bytes.len() == CONV * 4 * 2,
                "GDN convolution weight has incompatible BF16 storage"
            );
            finite_bf16(&weights.conv.bytes, CONV * 4, "GDN convolution weight")?;
            raw(weights.a_log, &[HEADS], "GDN A_log")?;
            raw(weights.dt_bias, &[HEADS], "GDN dt_bias")?;
            raw(weights.norm, &[HEAD_WIDTH], "GDN norm weight")?;
            ensure!(
                weights.eps.is_finite() && weights.eps > 0.0,
                "GDN norm epsilon must be finite and positive"
            );
            let prepare = metal.compile_msl_strict(SOURCE, "gdn_prepare")?;
            let recurrence = metal.compile_msl_strict(SOURCE, "gdn_recurrence")?;
            let norm_gate = metal.compile_msl_strict(SOURCE, "gdn_norm_gate")?;
            ensure!(
                prepare.max_threads_per_threadgroup() >= 32
                    && recurrence.max_threads_per_threadgroup() >= 128
                    && norm_gate.max_threads_per_threadgroup() >= 32,
                "Metal cannot dispatch released GDN threadgroup geometry"
            );
            let scratch = Scratch {
                hidden: metal.shared_buffer(HIDDEN * 2)?,
                qkv: metal.shared_buffer(CONV * 2)?,
                gate: metal.shared_buffer(VALUE * 2)?,
                beta_logits: metal.shared_buffer(HEADS * 2)?,
                decay_logits: metal.shared_buffer(HEADS * 2)?,
                query: metal.shared_buffer(KEY * 2)?,
                key: metal.shared_buffer(KEY * 2)?,
                value: metal.shared_buffer(VALUE * 2)?,
                beta: metal.shared_buffer(HEADS * 2)?,
                decay: metal.shared_buffer(HEADS * 4)?,
                recurrence: metal.shared_buffer(VALUE * 2)?,
                norm_gate: metal.shared_buffer(VALUE * 2)?,
                output: metal.shared_buffer(HIDDEN * 2)?,
            };
            let qkv = DenseMatvec::new(metal, weights.qkv)?;
            let gate = DenseMatvec::new(metal, weights.gate)?;
            let beta = DenseMatvec::new(metal, weights.beta)?;
            let decay = DenseMatvec::new(metal, weights.decay)?;
            let stacked_front = if input_front::preferred_for_device(&metal.device_name()) {
                Some(input_front::StackedFront::new(
                    metal,
                    [&qkv, &gate, &beta, &decay],
                )?)
            } else {
                None
            };
            Ok(Self {
                metal: metal.clone(),
                qkv,
                gate,
                beta,
                decay,
                output: DenseMatvec::new(metal, weights.output)?,
                prepare,
                recurrence,
                norm_gate,
                conv_weight: uploaded(metal, &weights.conv.bytes)?,
                a_log: uploaded(metal, &weights.a_log.bytes)?,
                dt_bias: uploaded(metal, &weights.dt_bias.bytes)?,
                norm_weight: uploaded(metal, &weights.norm.bytes)?,
                eps: uploaded(metal, &weights.eps.to_le_bytes())?,
                scratch,
                stacked_front,
            })
        }

        /// Encode one token. State remains unpublished until PendingGdn waits.
        pub fn submit<'a>(
            &'a mut self,
            hidden_bf16: &[u8],
            state: &'a mut GdnState,
        ) -> Result<PendingGdn<'a>> {
            finite_bf16(hidden_bf16, HIDDEN, "GDN hidden")?;
            self.scratch.hidden.write_bytes(0, hidden_bf16)?;
            let mut batch = self.metal.batch()?;
            // SAFETY: The uploaded hidden row is finite BF16, and the state
            // belongs to this device at a published frontier.
            unsafe { self.encode_device(&mut batch, &self.scratch.hidden, state)? };
            // A forgotten pending handle releases its Rust borrow without
            // waiting. Invalidate before submission so the prior frontier
            // cannot be observed while GPU writes are still in flight.
            state.poison();
            Ok(PendingGdn {
                flight: Some(batch.submit()),
                state,
                scratch: &self.scratch,
                settled: false,
            })
        }

        // The caller supplies a producer-owned BF16 row. Validated cell
        // matrices and scratch outputs remain disjoint through completion.
        #[inline]
        unsafe fn encode_input_projections(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
        ) -> Result<()> {
            if let Some(front) = &self.stacked_front {
                front.encode(
                    batch,
                    hidden,
                    [
                        &self.scratch.qkv,
                        &self.scratch.gate,
                        &self.scratch.beta_logits,
                        &self.scratch.decay_logits,
                    ],
                )?;
                return Ok(());
            }
            // SAFETY: The cell validated all four matrix shapes, and its
            // scratch owns distinct complete BF16 output allocations.
            unsafe {
                self.qkv
                    .encode(batch, hidden, &self.scratch.qkv, OutputPrecision::Bf16)?;
                self.gate
                    .encode(batch, hidden, &self.scratch.gate, OutputPrecision::Bf16)?;
                self.beta.encode(
                    batch,
                    hidden,
                    &self.scratch.beta_logits,
                    OutputPrecision::Bf16,
                )?;
                self.decay.encode(
                    batch,
                    hidden,
                    &self.scratch.decay_logits,
                    OutputPrecision::Bf16,
                )?;
            }
            Ok(())
        }

        /// Encode against a producer-owned hidden row in the same batch.
        /// The caller waits and validates outputs before publishing state.
        pub(crate) unsafe fn encode_device(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            state: &GdnState,
        ) -> Result<GdnDeviceOutput> {
            ensure!(
                hidden.byte_len() == HIDDEN * 2,
                "GDN hidden buffer width differs"
            );
            ensure!(state.valid, "GDN state has an unpublished or failed update");
            ensure!(state.frontier < u64::MAX, "GDN frontier overflow");
            let mut empty = [];
            state.conv.read_bytes(0, &mut empty)?;
            state.recurrent.read_bytes(0, &mut empty)?;
            // SAFETY: Every packed projection and raw parameter has validated
            // geometry. Prepare owns one convolution channel per thread, and
            // recurrence owns one FP32 state row per SIMD group; in-place state
            // reads finish in that owner before its writes. The batch retains
            // all resources until completion and encodes dependencies in order.
            unsafe {
                self.encode_input_projections(batch, hidden)?;
                batch.dispatch_1d(
                    &self.prepare,
                    &[
                        Binding {
                            buffer: &self.scratch.qkv,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.beta_logits,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.decay_logits,
                            offset: 0,
                        },
                        Binding {
                            buffer: &state.conv,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.conv_weight,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.a_log,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.dt_bias,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.query,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.key,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.value,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.beta,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.decay,
                            offset: 0,
                        },
                    ],
                    80 * 32,
                    32,
                )?;
                batch.dispatch_1d(
                    &self.recurrence,
                    &[
                        Binding {
                            buffer: &self.scratch.query,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.key,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.value,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.decay,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.beta,
                            offset: 0,
                        },
                        Binding {
                            buffer: &state.recurrent,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.recurrence,
                            offset: 0,
                        },
                    ],
                    HEADS * HEAD_WIDTH * 32,
                    128,
                )?;
                batch.dispatch_1d(
                    &self.norm_gate,
                    &[
                        Binding {
                            buffer: &self.scratch.recurrence,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.gate,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.norm_weight,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.norm_gate,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.eps,
                            offset: 0,
                        },
                    ],
                    HEADS * 32,
                    32,
                )?;
                self.output.encode(
                    batch,
                    &self.scratch.norm_gate,
                    &self.scratch.output,
                    OutputPrecision::Bf16,
                )?;
            }
            Ok(GdnDeviceOutput {
                output: self.scratch.output.clone(),
            })
        }
    }

    mod input_front;

    #[cfg(test)]
    mod projection_tests;

    #[cfg(test)]
    mod cell_tests;

    #[cfg(test)]
    mod state_tests {
        use super::*;

        #[test]
        fn borrowed_parts_validate_gdn_geometry_and_finiteness() {
            let mut conv = vec![0u8; CONV_BYTES];
            let mut recurrent = vec![0u8; RECURRENT_BYTES];
            GdnState::validate_parts(&conv, &recurrent).unwrap();
            assert!(GdnState::validate_parts(&conv[..CONV_BYTES - 2], &recurrent).is_err());
            assert!(GdnState::validate_parts(&conv, &recurrent[..RECURRENT_BYTES - 4]).is_err());
            conv[..2].copy_from_slice(&bf16::INFINITY.to_bits().to_le_bytes());
            assert!(GdnState::validate_parts(&conv, &recurrent).is_err());
            conv[..2].copy_from_slice(&bf16::MAX.to_bits().to_le_bytes());
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                recurrent[RECURRENT_BYTES - 4..].copy_from_slice(&value.to_le_bytes());
                assert!(GdnState::validate_parts(&conv, &recurrent).is_err());
            }
            recurrent[RECURRENT_BYTES - 4..].copy_from_slice(&f32::MAX.to_le_bytes());
            GdnState::validate_parts(&conv, &recurrent).unwrap();
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{GdnCell, GdnOutput, GdnStages, GdnState, GdnWeights, PendingGdn};

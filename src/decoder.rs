//! Qwen one-token decoder composition with explicit recurrent frontiers.
//!
//! PLE is applied before this block. An MLP write remains deferred so the next
//! block can fold it into its residual read. A failed block invalidates its
//! state rather than exposing a partly advanced token as resumable.

#[cfg(target_os = "macos")]
mod gpu {
    use anyhow::{Context, Result, ensure};
    use half::bf16;

    use crate::device_status::DeviceStatus;
    use crate::gdn::{GdnCell, GdnState, GdnWeights};
    use crate::hc::{HcBlock, HcOutput};
    use crate::kquant::DenseMatvec;
    use crate::metal::{Batch, Buffer, Metal, Stream};
    use crate::mlp::Qwen4Mlp;
    use crate::package::{Package, WeightFormat};
    use crate::routing::RoutingMode;

    fn validate_completed_bf16(bytes: &[u8], words: usize, name: &str) -> Result<()> {
        ensure!(bytes.len() == words * 2, "{name} buffer width differs");
        ensure!(
            bytes.as_chunks::<2>().0.iter().fold(true, |valid, word| {
                valid & bf16::from_bits(u16::from_le_bytes(*word)).is_finite()
            }),
            "{name} contains nonfinite BF16 values"
        );
        Ok(())
    }

    fn completed_bf16(buffer: &Buffer, words: usize, name: &str) -> Result<Vec<u8>> {
        ensure!(
            buffer.byte_len() == words * 2,
            "{name} buffer width differs"
        );
        let mut bytes = vec![0; words * 2];
        buffer.read_bytes(0, &mut bytes)?;
        validate_completed_bf16(&bytes, words, name)?;
        Ok(bytes)
    }

    pub struct GdnLayerState {
        recurrent: GdnState,
        valid: bool,
    }

    impl GdnLayerState {
        pub(crate) fn from_buffers(conv: Buffer, recurrent: Buffer, frontier: u64) -> Result<Self> {
            Ok(Self {
                recurrent: GdnState::from_buffers(conv, recurrent, frontier)?,
                valid: true,
            })
        }

        pub fn zero(metal: &Metal) -> Result<Self> {
            Ok(Self {
                recurrent: GdnState::zero(metal)?,
                valid: true,
            })
        }

        pub(crate) fn from_bytes(
            metal: &Metal,
            conv_bf16: &[u8],
            recurrent_f32: &[u8],
            frontier: u64,
        ) -> Result<Self> {
            Ok(Self {
                recurrent: GdnState::from_bytes(metal, conv_bf16, recurrent_f32, frontier)?,
                valid: true,
            })
        }

        pub fn frontier(&self) -> Result<u64> {
            ensure!(self.valid, "decoder state has an incomplete token");
            self.recurrent.frontier()
        }

        pub fn snapshot(&self) -> Result<(Vec<u8>, Vec<u8>)> {
            self.frontier()?;
            self.recurrent.snapshot()
        }

        pub(crate) fn recurrent(&self) -> Result<&GdnState> {
            self.frontier()?;
            Ok(&self.recurrent)
        }

        pub(crate) fn begin_prefill(&mut self) -> Result<&mut GdnState> {
            self.frontier()?;
            self.valid = false;
            Ok(&mut self.recurrent)
        }

        pub(crate) fn publish_prefill_after_wait(&mut self, rows: usize) -> Result<u64> {
            ensure!(!self.valid, "decoder prefill update was not pending");
            let next = self.recurrent.publish_rows_after_wait(rows)?;
            self.valid = true;
            Ok(next)
        }

        pub(crate) fn publish_device_after_wait(&mut self) -> Result<u64> {
            ensure!(!self.valid, "decoder device update was not pending");
            let next = self.recurrent.publish_after_wait()?;
            self.valid = true;
            Ok(next)
        }
    }

    pub struct GdnLayerOutput {
        pub residual_bf16: Vec<u8>,
        pub pending_output_bf16: Vec<u8>,
        pub pending_injection_bf16: Vec<u8>,
        pub changed_routes: u32,
        pub frontier: u64,
    }

    pub struct GdnLayerStages {
        pub attention: HcOutput,
        pub mixer_bf16: Vec<u8>,
        pub mlp_read: HcOutput,
        pub output: GdnLayerOutput,
    }

    pub(crate) struct GdnDeviceLayerOutput {
        pub residual: Buffer,
        pub pending_output: Buffer,
        pub pending_injection: Buffer,
        pub changed_routes: u32,
    }

    pub struct GdnDecoderBlock {
        metal: Metal,
        #[cfg(test)]
        prerouter_submissions: usize,
        attention: HcBlock,
        mixer: GdnCell,
        mlp_read: HcBlock,
        mlp: Qwen4Mlp,
    }

    /// Final residual read followed by the package's packed vocabulary head.
    pub struct FinalProjection {
        metal: Metal,
        residual: HcBlock,
        head: DenseMatvec,
    }

    pub struct FinalOutput {
        pub hidden_bf16: Vec<u8>,
        pub logits_bf16: Vec<u16>,
    }

    impl FinalProjection {
        pub fn new(metal: &Metal, package: &Package, eps: f32) -> Result<Self> {
            let weight = |name: &str| package.direct_weight(&format!("final_residual.{name}"));
            let norm = weight("hc_norm")?;
            ensure!(
                norm.format == WeightFormat::Bf16 && norm.shape == [10240],
                "final residual norm has an incompatible format or shape"
            );
            let head = package.direct_weight("lm_head")?;
            ensure!(
                head.format == WeightFormat::Q6K && head.shape == [248320, 2560],
                "vocabulary head has an incompatible packed format or shape"
            );
            Ok(Self {
                metal: metal.clone(),
                residual: HcBlock::new(
                    metal,
                    &norm.bytes,
                    &weight("input_mix_weight_down")?,
                    &weight("input_mix_weight_up")?,
                    None,
                    eps,
                )?,
                head: DenseMatvec::new(metal, &head)?,
            })
        }

        pub(crate) fn encode_device(
            &self,
            batch: &mut Batch,
            residual: &Buffer,
            pending: Option<(&Buffer, &Buffer)>,
            status: &DeviceStatus,
            logits: &Buffer,
        ) -> Result<Buffer> {
            ensure!(
                logits.byte_len() == 248320 * 2,
                "final logits extent differs"
            );
            // SAFETY: Input geometry is checked by HC and all dependencies are
            // encoded in this stream before their consumers.
            let hidden = unsafe { self.residual.encode_device(batch, residual, pending)? };
            status.encode_bf16(batch, &hidden.mixed)?;
            unsafe {
                self.head.encode(
                    batch,
                    &hidden.mixed,
                    logits,
                    crate::kquant::OutputPrecision::Bf16,
                )?;
            }
            status.encode_bf16(batch, logits)?;
            Ok(logits.clone())
        }

        pub(crate) fn encode_last_row(
            &self,
            batch: &mut Batch,
            residual: &Buffer,
            pending: Option<(&Buffer, &Buffer)>,
            rows: usize,
            status: &DeviceStatus,
            logits: &Buffer,
        ) -> Result<Buffer> {
            ensure!(
                logits.byte_len() == 248320 * 2,
                "final logits extent differs"
            );
            let row = rows
                .checked_sub(1)
                .context("final projection has no rows")?;
            // SAFETY: HC validates every row binding extent. Earlier producers
            // and this final consumer share the same ordered stream.
            let hidden = unsafe {
                self.residual
                    .encode_device_row(batch, residual, pending, row)?
            };
            status.encode_bf16(batch, &hidden.mixed)?;
            unsafe {
                self.head.encode(
                    batch,
                    &hidden.mixed,
                    logits,
                    crate::kquant::OutputPrecision::Bf16,
                )?;
            }
            status.encode_bf16(batch, logits)?;
            Ok(logits.clone())
        }

        pub fn run(
            &mut self,
            residual: &[u8],
            pending: Option<(&[u8], &[u8])>,
        ) -> Result<FinalOutput> {
            let hidden_bf16 = self.residual.run(residual, pending)?.mixed_bf16;
            let words: Vec<u16> = hidden_bf16
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes(*v))
                .collect();
            let logits_bf16 = self.head.run_bf16(&self.metal, &words)?;
            Ok(FinalOutput {
                hidden_bf16,
                logits_bf16,
            })
        }
    }

    pub(crate) fn residual_reader(
        metal: &Metal,
        package: &Package,
        path: &str,
        eps: f32,
    ) -> Result<HcBlock> {
        let weight = |name: &str| package.direct_weight(&format!("{path}.{name}"));
        let norm = weight("hc_norm")?;
        ensure!(
            norm.format == WeightFormat::Bf16 && norm.shape == [10240],
            "decoder residual norm must be BF16 with four 2560-wide branches"
        );
        HcBlock::new(
            metal,
            &norm.bytes,
            &weight("input_mix_weight_down")?,
            &weight("input_mix_weight_up")?,
            Some(&weight("block_inject_weight")?),
            eps,
        )
    }

    impl GdnDecoderBlock {
        pub(crate) fn shared_parts(&self) -> (&HcBlock, &GdnCell, &HcBlock, &Qwen4Mlp) {
            (&self.attention, &self.mixer, &self.mlp_read, &self.mlp)
        }
        pub fn new(
            metal: &Metal,
            package: &Package,
            layer: usize,
            capacity: usize,
            eps: f32,
        ) -> Result<Self> {
            let expert_layer = package.expert_layer(layer)?;
            let workspace = crate::package::ExpertRowWorkspace::new(expert_layer.row_bytes)?;
            Self::with_mlp(metal, package, layer, eps, || {
                Qwen4Mlp::with_expert_workspace_mode(
                    metal,
                    package,
                    layer,
                    capacity,
                    expert_layer,
                    workspace,
                    false,
                )
            })
        }

        pub(crate) fn with_expert_workspace(
            metal: &Metal,
            package: &Package,
            layer: usize,
            capacity: usize,
            eps: f32,
            expert_layer: crate::package::ExpertLayer,
            workspace: crate::package::ExpertRowWorkspace,
        ) -> Result<Self> {
            Self::with_mlp(metal, package, layer, eps, || {
                Qwen4Mlp::with_expert_workspace(
                    metal,
                    package,
                    layer,
                    capacity,
                    expert_layer,
                    workspace,
                )
            })
        }

        fn with_mlp(
            metal: &Metal,
            package: &Package,
            layer: usize,
            eps: f32,
            mlp: impl FnOnce() -> Result<Qwen4Mlp>,
        ) -> Result<Self> {
            let prefix = format!("layers.{layer}");
            let weight =
                |name: &str| package.direct_weight(&format!("{prefix}.mixer.module.{name}"));
            let qkv = weight("in_proj_qkv")?;
            let gate = weight("in_proj_z")?;
            let beta = weight("in_proj_b")?;
            let decay = weight("in_proj_a")?;
            let output = weight("out_proj")?;
            let conv = weight("conv1d")?;
            let a_log = weight("A_log")?;
            let dt_bias = weight("dt_bias")?;
            let norm = weight("norm")?;
            Ok(Self {
                metal: metal.clone(),
                #[cfg(test)]
                prerouter_submissions: 0,
                attention: residual_reader(
                    metal,
                    package,
                    &format!("{prefix}.attention_residual"),
                    eps,
                )?,
                mixer: GdnCell::new(
                    metal,
                    GdnWeights {
                        qkv: &qkv,
                        gate: &gate,
                        beta: &beta,
                        decay: &decay,
                        output: &output,
                        conv: &conv,
                        a_log: &a_log,
                        dt_bias: &dt_bias,
                        norm: &norm,
                        eps,
                    },
                )?,
                mlp_read: residual_reader(metal, package, &format!("{prefix}.mlp_residual"), eps)?,
                mlp: mlp()?,
            })
        }

        pub fn seed_residents(&mut self, sources: &[u32]) -> Result<()> {
            self.mlp.seed_residents(sources)
        }

        pub fn seed_hotlist(
            &mut self,
            counts: &std::collections::BTreeMap<u32, u64>,
        ) -> Result<()> {
            self.mlp.seed_hotlist(counts)
        }

        pub fn residency_map(&self) -> Result<[u32; 512]> {
            self.mlp.residency_map()
        }

        pub(crate) fn experts_mut(&mut self) -> &mut crate::experts::BoundedExperts {
            self.mlp.experts_mut()
        }

        pub(crate) fn mlp_mut(&mut self) -> &mut Qwen4Mlp {
            &mut self.mlp
        }

        pub(crate) fn run_device(
            &mut self,
            stream: &mut Stream,
            residual: &Buffer,
            pending: Option<(&Buffer, &Buffer)>,
            mode: RoutingMode,
            state: &mut GdnLayerState,
            status: &DeviceStatus,
        ) -> Result<GdnDeviceLayerOutput> {
            self.run_device_schedule(stream, residual, pending, mode, state, status, false)
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) fn run_device_schedule(
            &mut self,
            stream: &mut Stream,
            residual: &Buffer,
            pending: Option<(&Buffer, &Buffer)>,
            mode: RoutingMode,
            state: &mut GdnLayerState,
            status: &DeviceStatus,
            queued: bool,
        ) -> Result<GdnDeviceLayerOutput> {
            state
                .frontier()?
                .checked_add(1)
                .context("decoder frontier overflow")?;
            let batch = stream.batch()?;
            // SAFETY: HC validates extents; each device consumer follows its
            // producer in the same serial command stream.
            let attention = unsafe { self.attention.encode_device(batch, residual, pending)? };
            let mixer = unsafe {
                self.mixer
                    .encode_device(batch, &attention.mixed, &state.recurrent)?
            };
            state.valid = false;
            state.recurrent.poison();
            let mlp_read = unsafe {
                self.mlp_read.encode_device(
                    batch,
                    &attention.updated,
                    Some((
                        &mixer.output,
                        attention
                            .injection
                            .as_ref()
                            .context("attention injection missing")?,
                    )),
                )?
            };
            status.encode_bf16(batch, &mlp_read.updated)?;
            let output = if queued {
                self.mlp
                    .enqueue_device(stream, &mlp_read.mixed, mode, status)?
            } else {
                self.mlp.run_device(stream, &mlp_read.mixed, mode, status)?
            };
            Ok(GdnDeviceLayerOutput {
                residual: mlp_read.updated,
                pending_output: output.output,
                pending_injection: mlp_read.injection.context("MLP injection missing")?,
                changed_routes: output.changed_routes,
            })
        }

        pub fn run(
            &mut self,
            residual: &[u8],
            pending: Option<(&[u8], &[u8])>,
            mode: RoutingMode,
            state: &mut GdnLayerState,
        ) -> Result<GdnLayerOutput> {
            self.run_prerouter(residual, pending, mode, state)
        }

        pub fn run_stages(
            &mut self,
            residual: &[u8],
            pending: Option<(&[u8], &[u8])>,
            mode: RoutingMode,
            state: &mut GdnLayerState,
        ) -> Result<GdnLayerStages> {
            self.execute(residual, pending, mode, state)
        }

        fn execute(
            &mut self,
            residual: &[u8],
            pending: Option<(&[u8], &[u8])>,
            mode: RoutingMode,
            state: &mut GdnLayerState,
        ) -> Result<GdnLayerStages> {
            let next = state
                .frontier()?
                .checked_add(1)
                .context("decoder frontier overflow")?;
            // Earlier stages may modify GPU state before a later stage fails.
            // Only a complete layer result re-enables access to the frontier.
            state.valid = false;
            let attention = self.attention.run(residual, pending)?;
            let attention_injection = attention
                .injection_bf16
                .as_deref()
                .context("attention residual has no injection weights")?;
            let mixer = self
                .mixer
                .submit(&attention.mixed_bf16, &mut state.recurrent)?
                .wait()?;
            ensure!(
                mixer.frontier == next,
                "mixer frontier differs from decoder token"
            );
            let mlp_read = self.mlp_read.run(
                &attention.updated_bf16,
                Some((&mixer.output_bf16, attention_injection)),
            )?;
            let hidden: Vec<u16> = mlp_read
                .mixed_bf16
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes(*v))
                .collect();
            let mlp = self.mlp.run(&hidden, mode)?;
            let injection = mlp_read
                .injection_bf16
                .as_ref()
                .context("MLP residual has no injection weights")?;
            let output = GdnLayerOutput {
                residual_bf16: mlp_read.updated_bf16.clone(),
                pending_output_bf16: mlp
                    .final_bf16
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                pending_injection_bf16: injection.clone(),
                changed_routes: mlp.changed_routes,
                frontier: next,
            };
            state.valid = true;
            Ok(GdnLayerStages {
                attention,
                mixer_bf16: mixer.output_bf16,
                mlp_read,
                output,
            })
        }

        /// Standalone one-token path. The engine uses `run_device`; diagnostics
        /// retain this completed-row interface and `execute` for comparisons.
        fn run_prerouter(
            &mut self,
            residual: &[u8],
            pending: Option<(&[u8], &[u8])>,
            mode: RoutingMode,
            state: &mut GdnLayerState,
        ) -> Result<GdnLayerOutput> {
            let next = state
                .frontier()?
                .checked_add(1)
                .context("decoder frontier overflow")?;
            let residency_before = self.mlp.residency_map()?;
            state.valid = false;
            let mut batch = self.metal.batch()?;
            let attention = self.attention.encode_host(&mut batch, residual, pending)?;
            let attention_injection = attention
                .injection
                .as_ref()
                .context("attention residual has no injection weights")?;
            // SAFETY: Each producer fills a fixed-width row earlier in this
            // serial batch. Integer geometry bounds all accesses; finiteness
            // is checked after completion before routing or pool demand.
            let mixer = unsafe {
                self.mixer
                    .encode_device(&mut batch, &attention.mixed, &state.recurrent)?
            };
            let mlp_read = unsafe {
                self.mlp_read.encode_device(
                    &mut batch,
                    &attention.updated,
                    Some((&mixer.output, attention_injection)),
                )?
            };
            let router = self.mlp.encode_router_device(
                &mut batch,
                &mlp_read.mixed,
                &residency_before,
                mode,
            )?;
            // GDN writes state in place. Any submitted failure leaves both
            // the recurrent and layer frontiers inaccessible.
            state.recurrent.poison();
            #[cfg(test)]
            {
                self.prerouter_submissions += 1;
            }
            batch.submit().wait().context("GDN pre-router command")?;
            completed_bf16(&attention.mixed, 2560, "attention mixed")?;
            completed_bf16(&attention.updated, 10240, "attention updated")?;
            completed_bf16(attention_injection, 4, "attention injection")?;
            completed_bf16(&mixer.output, 2560, "GDN output")?;
            let hidden = completed_bf16(&mlp_read.mixed, 2560, "MLP mixed")?;
            let (logits, route) = router.read()?;
            let hidden_words: Vec<u16> = hidden
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| u16::from_le_bytes(*word))
                .collect();
            let (mlp, _) =
                self.mlp
                    .execute_selected(&hidden_words, residency_before, logits, route, false)?;
            let residual_bf16 = completed_bf16(&mlp_read.updated, 10240, "MLP updated")?;
            let pending_injection_bf16 = completed_bf16(
                mlp_read
                    .injection
                    .as_ref()
                    .context("MLP residual has no injection weights")?,
                4,
                "MLP injection",
            )?;
            ensure!(
                state.recurrent.publish_after_wait()? == next,
                "mixer frontier differs from decoder token"
            );
            state.valid = true;
            Ok(GdnLayerOutput {
                residual_bf16,
                pending_output_bf16: mlp
                    .final_bf16
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect(),
                pending_injection_bf16,
                changed_routes: mlp.changed_routes,
                frontier: next,
            })
        }
    }

    #[cfg(test)]
    mod prerouter_tests {
        use super::*;

        #[test]
        fn completed_rows_reject_nonfinite_at_every_boundary() {
            for words in [4, 2560, 10240] {
                let mut row = vec![0u8; words * 2];
                validate_completed_bf16(&row, words, "test").unwrap();
                for index in [0, words / 2, words - 1] {
                    for invalid in [bf16::NAN, bf16::INFINITY, bf16::NEG_INFINITY] {
                        row[2 * index..2 * index + 2]
                            .copy_from_slice(&invalid.to_bits().to_le_bytes());
                        assert!(validate_completed_bf16(&row, words, "test").is_err());
                        row[2 * index..2 * index + 2].fill(0);
                    }
                }
                assert!(validate_completed_bf16(&row[..row.len() - 2], words, "test").is_err());
            }
        }

        #[test]
        #[ignore = "requires an external real package and a GPU acceptance window"]
        fn submitted_nonfinite_chain_poisoned_before_expert_demand() -> Result<()> {
            let path = std::env::var("GDN_PREROUTER_TEST_PACKAGE")
                .context("GDN_PREROUTER_TEST_PACKAGE is required")?;
            let metal = Metal::new()?;
            let package = Package::open(path)?;
            let mut block = GdnDecoderBlock::new(&metal, &package, 0, 12, 1e-6)?;
            let mut state = GdnLayerState::zero(&metal)?;
            let before = block.experts_mut().snapshot();
            let residual = vec![0u8; 10240 * 2];
            let pending_output = bf16::MAX.to_bits().to_le_bytes().repeat(2560);
            let pending_injection = bf16::MAX.to_bits().to_le_bytes().repeat(4);
            let error = block
                .run(
                    &residual,
                    Some((&pending_output, &pending_injection)),
                    RoutingMode::Unbiased,
                    &mut state,
                )
                .err()
                .context("submitted overflow unexpectedly committed")?;
            ensure!(
                error.to_string().contains("nonfinite BF16"),
                "submitted chain failed for a reason other than the postwait finite guard: {error}"
            );
            ensure!(
                state.frontier().is_err()
                    && state.snapshot().is_err()
                    && state.recurrent.frontier().is_err(),
                "failed submitted chain exposed recurrent state"
            );
            ensure!(
                block.experts_mut().snapshot() == before,
                "failed submitted chain changed expert demand or pool clocks"
            );
            ensure!(
                block.prerouter_submissions == 1,
                "joined pre-router encoded more than one submission"
            );
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) use gpu::residual_reader;

#[cfg(target_os = "macos")]
pub use gpu::{
    FinalOutput, FinalProjection, GdnDecoderBlock, GdnLayerOutput, GdnLayerStages, GdnLayerState,
};

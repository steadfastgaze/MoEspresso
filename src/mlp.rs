//! One-token Qwen MLP with cache-aware routing and a bounded expert pool.
//!
//! Selection reads a published residency snapshot before the selected rows
//! are ensured. The shared branch preserves BF16 projection, activation,
//! scalar gate, product, and final-add boundaries.

#[cfg(target_os = "macos")]
mod gpu {
    use anyhow::{Context, Result, ensure};
    use half::bf16;

    use crate::device_status::DeviceStatus;
    use crate::experts::BoundedExperts;
    use crate::kquant::{DenseMatvec, DeviceMatrix, OutputPrecision};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline, Stream};
    use crate::package::{Package, WeightFormat};
    use crate::routing::{
        CachePriorRouter, RoutingBuffers, RoutingGeometry, RoutingMode, RoutingOutput,
        validate_logits, validate_slot_maps,
    };

    const HIDDEN: usize = 2560;
    const ROUTES: usize = 10;
    const EXPERTS: usize = 512;
    const SHARED: usize = 640;
    const SOURCE: &str = include_str!("../kernels/metal/mlp.metal");

    fn shared_front_pipeline(metal: &Metal) -> Result<Pipeline> {
        let source = format!(
            "{}\n{}",
            crate::kquant::METAL_SOURCE,
            include_str!("../kernels/metal/mlp_shared_front.metal")
        );
        let pipeline = metal.compile_msl_strict(&source, "qwen4_shared_front_q6_k")?;
        ensure!(
            pipeline.max_threads_per_threadgroup() >= 64,
            "shared Q6_K front requires a 64-thread Metal group"
        );
        Ok(pipeline)
    }

    fn shared_front_matrices(projections: [&DenseMatvec; 3]) -> Result<[DeviceMatrix; 3]> {
        let matrices = projections.map(DenseMatvec::shared_matrix);
        for (matrix, rows) in matrices.iter().zip([SHARED, SHARED, 1]) {
            matrix.check(WeightFormat::Q6K, [rows, HIDDEN])?;
        }
        Ok(matrices)
    }

    // The projections have Q6_K shapes [640,2560], [640,2560], [1,2560].
    // Hidden holds one BF16 row; outputs hold 640, 640 and one BF16 word.
    // The caller excludes aliases and retains stable inputs until completion.
    unsafe fn encode_stacked_shared_front(
        batch: &mut Batch,
        pipeline: &Pipeline,
        matrices: &[DeviceMatrix; 3],
        hidden: &Buffer,
        outputs: [&Buffer; 3],
    ) -> Result<()> {
        let bindings = [
            matrices[0].binding(),
            matrices[1].binding(),
            matrices[2].binding(),
            Binding {
                buffer: hidden,
                offset: 0,
            },
            Binding {
                buffer: outputs[0],
                offset: 0,
            },
            Binding {
                buffer: outputs[1],
                offset: 0,
            },
            Binding {
                buffer: outputs[2],
                offset: 0,
            },
        ];
        // SAFETY: The fixed grid visits each declared output row once. Batch
        // retains every direct binding, including the shared matrix allocations.
        unsafe { batch.dispatch_1d(pipeline, &bindings, 161 * 64, 64) }
    }

    fn buffer(metal: &Metal, words: usize) -> Result<Buffer> {
        metal.shared_buffer(words * 2)
    }

    fn words_to_bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_words(buffer: &Buffer, words: usize) -> Result<Vec<u16>> {
        let mut bytes = vec![0; words * 2];
        buffer.read_bytes(0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| u16::from_le_bytes(*word))
            .collect())
    }

    fn checked_matrix(
        metal: &Metal,
        package: &Package,
        path: String,
        format: WeightFormat,
        shape: [usize; 2],
    ) -> Result<DenseMatvec> {
        let direct = package.direct_weight(&path)?;
        ensure!(
            direct.format == format && direct.shape == shape,
            "{path} differs from released Qwen MLP format or shape"
        );
        DenseMatvec::new(metal, &direct)
    }

    struct RouterScratch {
        logits: Buffer,
        map: Buffer,
        ids: Buffer,
        scores: Buffer,
        changed: Buffer,
    }

    impl RouterScratch {
        fn new(metal: &Metal) -> Result<Self> {
            let geometry = RoutingGeometry::new(1)?;
            Ok(Self {
                logits: metal.shared_buffer(geometry.logits_bytes())?,
                map: metal.shared_buffer(EXPERTS * 4)?,
                ids: metal.shared_buffer(geometry.ids_bytes())?,
                scores: metal.shared_buffer(geometry.scores_bytes())?,
                changed: metal.shared_buffer(geometry.changed_bytes())?,
            })
        }
    }

    pub(crate) struct EncodedRouter {
        logits: Buffer,
        ids: Buffer,
        scores: Buffer,
        changed: Buffer,
        geometry: RoutingGeometry,
    }

    impl EncodedRouter {
        /// Read only compact selector records after completed finite-logit
        /// validation by the request's device status.
        pub(crate) fn read_compact(self) -> Result<RoutingOutput> {
            CachePriorRouter::read_output(self.geometry, &self.ids, &self.scores, &self.changed)
        }

        /// Read only after successful completion of the producing batch.
        /// Validate the original projection before consuming selector output.
        pub(crate) fn read(self) -> Result<(Vec<u16>, RoutingOutput)> {
            let logits = read_words(&self.logits, EXPERTS)?;
            validate_logits(&logits)?;
            let route = CachePriorRouter::read_output(
                self.geometry,
                &self.ids,
                &self.scores,
                &self.changed,
            )?;
            Ok((logits, route))
        }
    }

    // This phase has no expert-pool access. Invalid projected logits cannot
    // return routing outputs to the caller's demand-publication phase.
    fn encode_router(
        projection: &DenseMatvec,
        selector: &CachePriorRouter,
        batch: &mut Batch,
        hidden: &Buffer,
        residency: &[u32; EXPERTS],
        mode: RoutingMode,
        scratch: &RouterScratch,
    ) -> Result<EncodedRouter> {
        ensure!(
            hidden.byte_len() == HIDDEN * 2,
            "MLP router hidden buffer width differs"
        );
        validate_slot_maps([residency, residency, residency])?;
        let geometry = RoutingGeometry::new(1)?;
        let logits = scratch.logits.clone();
        let map = &scratch.map;
        map.write_bytes(
            0,
            &residency
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let ids = scratch.ids.clone();
        let scores = scratch.scores.clone();
        let changed = scratch.changed.clone();
        // SAFETY: The caller supplied initialized BF16 hidden storage. Both
        // kernels use integer geometry for addressing and have disjoint outputs.
        // Projection may overflow. Bounded demand waits for validation;
        // resident execution guards IDs and scores before weight addressing.
        unsafe {
            projection.encode(batch, hidden, &logits, OutputPrecision::Bf16)?;
            selector.encode(
                batch,
                RoutingBuffers {
                    logits_bf16: &logits,
                    gate_slots: map,
                    up_slots: map,
                    down_slots: map,
                    ids: &ids,
                    scores_bf16: &scores,
                    changed: &changed,
                },
                geometry,
                mode,
            )?;
        }
        Ok(EncodedRouter {
            logits,
            ids,
            scores,
            changed,
            geometry,
        })
    }

    fn project_and_route(
        metal: &Metal,
        projection: &DenseMatvec,
        selector: &CachePriorRouter,
        hidden: &Buffer,
        residency: &[u32; EXPERTS],
        mode: RoutingMode,
    ) -> Result<(Vec<u16>, RoutingOutput)> {
        let mut batch = metal.batch()?;
        let scratch = RouterScratch::new(metal)?;
        let encoded = encode_router(
            projection, selector, &mut batch, hidden, residency, mode, &scratch,
        )?;
        batch
            .submit()
            .wait()
            .context("MLP projection and routing command")?;
        encoded.read()
    }

    struct SharedScratch {
        hidden: Buffer,
        gate: Buffer,
        up: Buffer,
        activation: Buffer,
        down: Buffer,
        scalar_logit: Buffer,
        shared: Buffer,
        final_output: Buffer,
    }

    /// Minimal per-token result. Full intermediate readback is in `run_stages`.
    pub struct MlpOutput {
        pub final_bf16: Vec<u16>,
        pub changed_routes: u32,
    }

    pub struct DeviceMlpOutput {
        pub output: Buffer,
        pub changed_routes: u32,
    }

    /// Diagnostic outputs for exact comparison with a reference execution.
    pub struct MlpStages {
        pub output: MlpOutput,
        pub residency_before: [u32; EXPERTS],
        pub router_logits_bf16: Vec<u16>,
        pub selected_ids: [u32; ROUTES],
        pub selected_scores_bf16: [u16; ROUTES],
        pub routed_bf16: Vec<u16>,
        pub shared_gate_proj_bf16: Vec<u16>,
        pub shared_up_proj_bf16: Vec<u16>,
        pub shared_activation_bf16: Vec<u16>,
        pub shared_down_bf16: Vec<u16>,
        pub shared_gate_logit_bf16: u16,
        pub shared_bf16: Vec<u16>,
    }

    pub struct Qwen4Mlp {
        metal: Metal,
        router_projection: DenseMatvec,
        router_scratch: RouterScratch,
        selector: CachePriorRouter,
        experts: BoundedExperts,
        shared_gate_proj: DenseMatvec,
        shared_up_proj: DenseMatvec,
        shared_down_proj: DenseMatvec,
        shared_gate: DenseMatvec,
        shared_front: Pipeline,
        shared_front_matrices: [DeviceMatrix; 3],
        activate: Pipeline,
        finish: Pipeline,
        scratch: SharedScratch,
        handoff: Option<crate::decode_handoff::DecodeHandoff>,
    }

    impl Qwen4Mlp {
        pub(crate) fn shared_matrices(&self) -> [crate::kquant::DeviceMatrix; 5] {
            [
                &self.router_projection,
                &self.shared_gate_proj,
                &self.shared_up_proj,
                &self.shared_down_proj,
                &self.shared_gate,
            ]
            .map(DenseMatvec::shared_matrix)
        }
        pub fn new(
            metal: &Metal,
            package: &Package,
            layer: usize,
            capacity: usize,
        ) -> Result<Self> {
            let expert_layer = package.expert_layer(layer)?;
            let workspace = crate::package::ExpertRowWorkspace::new(expert_layer.row_bytes)?;
            Self::with_expert_workspace_mode(
                metal,
                package,
                layer,
                capacity,
                expert_layer,
                workspace,
                false,
            )
        }

        pub(crate) fn source_layer(&self) -> crate::package::ExpertLayer {
            self.experts.source_layer()
        }

        pub(crate) fn with_expert_workspace(
            metal: &Metal,
            package: &Package,
            layer: usize,
            capacity: usize,
            expert_layer: crate::package::ExpertLayer,
            workspace: crate::package::ExpertRowWorkspace,
        ) -> Result<Self> {
            Self::with_expert_workspace_mode(
                metal,
                package,
                layer,
                capacity,
                expert_layer,
                workspace,
                true,
            )
        }

        pub(crate) fn with_expert_workspace_mode(
            metal: &Metal,
            package: &Package,
            layer: usize,
            capacity: usize,
            expert_layer: crate::package::ExpertLayer,
            workspace: crate::package::ExpertRowWorkspace,
            engine_storage: bool,
        ) -> Result<Self> {
            let prefix = format!("layers.{layer}.mlp");
            let router_projection = checked_matrix(
                metal,
                package,
                format!("{prefix}.gate"),
                WeightFormat::Bf16,
                [EXPERTS, HIDDEN],
            )?;
            let shared_gate_proj = checked_matrix(
                metal,
                package,
                format!("{prefix}.shared_expert.gate_proj"),
                WeightFormat::Q6K,
                [SHARED, HIDDEN],
            )?;
            let shared_up_proj = checked_matrix(
                metal,
                package,
                format!("{prefix}.shared_expert.up_proj"),
                WeightFormat::Q6K,
                [SHARED, HIDDEN],
            )?;
            let shared_down_proj = checked_matrix(
                metal,
                package,
                format!("{prefix}.shared_expert.down_proj"),
                WeightFormat::Q8_0,
                [HIDDEN, SHARED],
            )?;
            let shared_gate = checked_matrix(
                metal,
                package,
                format!("{prefix}.shared_expert_gate"),
                WeightFormat::Q6K,
                [1, HIDDEN],
            )?;
            let shared_front_matrices =
                shared_front_matrices([&shared_gate_proj, &shared_up_proj, &shared_gate])?;
            ensure!(
                expert_layer.num_experts == EXPERTS,
                "Qwen MLP requires 512 source experts"
            );
            let experts = if engine_storage {
                BoundedExperts::with_workspace(metal, expert_layer, capacity, workspace)?
            } else {
                BoundedExperts::with_workspace_contiguous(metal, expert_layer, capacity, workspace)?
            };
            let selector = CachePriorRouter::new(metal)?;
            let activate = metal.compile_msl(SOURCE, "qwen4_shared_activate")?;
            let finish = metal.compile_msl(SOURCE, "qwen4_shared_finish")?;
            ensure!(
                activate.max_threads_per_threadgroup() >= 256
                    && finish.max_threads_per_threadgroup() >= 256,
                "shared MLP requires a 256-thread Metal group"
            );
            let scratch = SharedScratch {
                hidden: buffer(metal, HIDDEN)?,
                gate: buffer(metal, SHARED)?,
                up: buffer(metal, SHARED)?,
                activation: buffer(metal, SHARED)?,
                down: buffer(metal, HIDDEN)?,
                scalar_logit: buffer(metal, 1)?,
                shared: buffer(metal, HIDDEN)?,
                final_output: buffer(metal, HIDDEN)?,
            };
            Ok(Self {
                metal: metal.clone(),
                router_projection,
                router_scratch: RouterScratch::new(metal)?,
                selector,
                experts,
                shared_gate_proj,
                shared_up_proj,
                shared_down_proj,
                shared_gate,
                shared_front: shared_front_pipeline(metal)?,
                shared_front_matrices,
                activate,
                finish,
                scratch,
                handoff: if capacity < 512 {
                    Some(crate::decode_handoff::DecodeHandoff::new(metal)?)
                } else {
                    None
                },
            })
        }

        pub fn residency_map(&self) -> Result<[u32; EXPERTS]> {
            self.experts.residency_map()
        }

        pub(crate) fn check_expert_quiescent(&self) -> Result<()> {
            if let Some(handoff) = &self.handoff {
                handoff.check_quiescent()?;
            }
            Ok(())
        }

        /// Enqueue a bounded decode MLP with owner-serviced expert publication.
        /// The caller services every enqueued layer before completing the stream.
        pub(crate) fn enqueue_device(
            &mut self,
            stream: &mut Stream,
            hidden: &Buffer,
            mode: RoutingMode,
            status: &DeviceStatus,
        ) -> Result<DeviceMlpOutput> {
            let residency = self.experts.residency_map()?;
            self.handoff
                .as_mut()
                .context("event decode requires bounded experts")?
                .begin(stream)?;
            #[cfg(test)]
            self.metal.test_handoff_encoding_checkpoint()?;
            let encoded = self.encode_router_device(stream.batch()?, hidden, &residency, mode)?;
            status.encode_bf16(stream.batch()?, hidden)?;
            status.encode_bf16(stream.batch()?, &encoded.logits)?;
            let handoff = self.handoff.as_ref().expect("checked bounded handoff");
            handoff.encode_export(
                stream.batch()?,
                &encoded.ids,
                &encoded.scores,
                &encoded.changed,
                status.flag_buffer(),
            )?;
            self.encode_shared_projections(stream.batch()?, hidden)?;
            // Stream takes release ownership before encoding the wait. Every
            // error and panic path releases the gate before its blocking drain.
            stream.wait_for_publication(&handoff.pending()?.publication)?;
            let routed = self
                .experts
                .encode_gated(stream.batch()?, hidden, handoff.flag())?;
            self.encode_shared_finish(stream.batch()?, &routed)?;
            status.encode_bf16(stream.batch()?, &self.scratch.final_output)?;
            stream.flush()?;
            Ok(DeviceMlpOutput {
                output: self.scratch.final_output.clone(),
                changed_routes: 0,
            })
        }

        pub(crate) fn service_demand(&mut self) -> Result<u32> {
            let handoff = self.handoff.as_mut().context("no bounded decode handoff")?;
            let routes = handoff.read()?;
            // SAFETY: The single inference owner started this token quiescent.
            // This physical layer is enqueued exactly once; every pool and
            // selector reader waits on the same unopened publication point.
            unsafe {
                self.experts.publish_gated(
                    &routes.ids,
                    &routes.scores,
                    &handoff.pending()?.publication,
                )?;
            }
            let changed = routes.changed;
            handoff.publish()?;
            Ok(changed)
        }

        /// Keep identity-resident routing on device; resolve bounded expert
        /// demand at the compact route seam before encoding the MLP tail.
        pub(crate) fn run_device(
            &mut self,
            stream: &mut Stream,
            hidden: &Buffer,
            mode: RoutingMode,
            status: &DeviceStatus,
        ) -> Result<DeviceMlpOutput> {
            let residency = self.experts.residency_map()?;
            let encoded = self.encode_router_device(stream.batch()?, hidden, &residency, mode)?;
            status.encode_bf16(stream.batch()?, hidden)?;
            status.encode_bf16(stream.batch()?, &encoded.logits)?;
            if self.experts.capacity() == EXPERTS {
                let routed = self.experts.encode_full_resident_device(
                    stream.batch()?,
                    hidden,
                    &encoded.ids,
                    &encoded.scores,
                    &encoded.changed,
                    status.flag_buffer(),
                )?;
                self.encode_shared_device(stream.batch()?, hidden, &routed)?;
                status.encode_bf16(stream.batch()?, &self.scratch.final_output)?;
                return Ok(DeviceMlpOutput {
                    output: self.scratch.final_output.clone(),
                    changed_routes: 0,
                });
            }
            // Shared projections read only the hidden row and direct weights.
            // They neither access expert pools nor bind the request status.
            // Keep them queued while the host retires the router and loads rows.
            let router_done = stream.flush_marked()?;
            self.encode_shared_projections(stream.batch()?, hidden)?;
            stream.flush()?;
            stream
                .wait_through(router_done)
                .context("MLP route demand boundary")?;
            status.check()?;
            let route = encoded.read_compact()?;
            let selected_ids: [u32; ROUTES] = route
                .ids
                .try_into()
                .map_err(|_| anyhow::anyhow!("router did not return ten IDs"))?;
            let selected_scores: [u16; ROUTES] = route
                .scores_bf16
                .try_into()
                .map_err(|_| anyhow::anyhow!("router did not return ten BF16 scores"))?;
            let changed = *route
                .changed
                .first()
                .context("router omitted change count")?;
            ensure!(
                changed <= ROUTES as u32,
                "router change count exceeds top-k"
            );
            let routed = self.experts.encode_routed_device(
                stream.batch()?,
                hidden,
                &selected_ids,
                &selected_scores,
            )?;
            self.encode_shared_finish(stream.batch()?, &routed)?;
            status.encode_bf16(stream.batch()?, &self.scratch.final_output)?;
            Ok(DeviceMlpOutput {
                output: self.scratch.final_output.clone(),
                changed_routes: changed,
            })
        }

        fn encode_shared_device(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            routed: &Buffer,
        ) -> Result<()> {
            self.encode_shared_projections(batch, hidden)?;
            self.encode_shared_finish(batch, routed)
        }

        fn encode_shared_projections(&self, batch: &mut Batch, hidden: &Buffer) -> Result<()> {
            ensure!(
                hidden.byte_len() == HIDDEN * 2,
                "shared MLP hidden width differs"
            );
            // SAFETY: Every projection reads one complete BF16 device row.
            // The ordered batch retains scratch and input through completion.
            unsafe {
                encode_stacked_shared_front(
                    batch,
                    &self.shared_front,
                    &self.shared_front_matrices,
                    hidden,
                    [
                        &self.scratch.gate,
                        &self.scratch.up,
                        &self.scratch.scalar_logit,
                    ],
                )?;
                batch.dispatch_1d(
                    &self.activate,
                    &[
                        Binding {
                            buffer: &self.scratch.gate,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.up,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.activation,
                            offset: 0,
                        },
                    ],
                    SHARED,
                    256,
                )?;
                self.shared_down_proj.encode(
                    batch,
                    &self.scratch.activation,
                    &self.scratch.down,
                    OutputPrecision::Bf16,
                )?;
            }
            Ok(())
        }

        fn encode_shared_finish(&self, batch: &mut Batch, routed: &Buffer) -> Result<()> {
            // SAFETY: The same ordered stream contains every shared and routed
            // producer. No host read is permitted before final completion.
            unsafe {
                batch.dispatch_1d(
                    &self.finish,
                    &[
                        Binding {
                            buffer: routed,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.down,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.scalar_logit,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.shared,
                            offset: 0,
                        },
                        Binding {
                            buffer: &self.scratch.final_output,
                            offset: 0,
                        },
                    ],
                    HIDDEN,
                    256,
                )?;
            }
            Ok(())
        }

        /// The caller supplies an initialized producer row from this batch.
        /// No expert demand occurs until the returned projection is validated.
        pub(crate) fn encode_router_device(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            residency: &[u32; EXPERTS],
            mode: RoutingMode,
        ) -> Result<EncodedRouter> {
            encode_router(
                &self.router_projection,
                &self.selector,
                batch,
                hidden,
                residency,
                mode,
                &self.router_scratch,
            )
        }

        pub(crate) fn experts_mut(&mut self) -> &mut BoundedExperts {
            &mut self.experts
        }

        /// Preload a declared resident set before a request. Selection still
        /// snapshots the published map at the start of each `run` call.
        pub fn seed_residents(&mut self, sources: &[u32]) -> Result<()> {
            self.experts.seed_residents(sources)
        }

        pub fn seed_hotlist(
            &mut self,
            counts: &std::collections::BTreeMap<u32, u64>,
        ) -> Result<()> {
            self.experts.seed_hotlist(counts)
        }

        pub fn run(&mut self, hidden_bf16: &[u16], mode: RoutingMode) -> Result<MlpOutput> {
            Ok(self.execute(hidden_bf16, mode, false)?.0)
        }

        pub fn run_stages(&mut self, hidden_bf16: &[u16], mode: RoutingMode) -> Result<MlpStages> {
            self.execute(hidden_bf16, mode, true)?
                .1
                .context("MLP stage capture was not requested")
        }

        fn execute(
            &mut self,
            hidden_bf16: &[u16],
            mode: RoutingMode,
            capture: bool,
        ) -> Result<(MlpOutput, Option<MlpStages>)> {
            ensure!(
                hidden_bf16.len() == HIDDEN,
                "Qwen MLP needs one BF16 hidden row"
            );
            ensure!(
                hidden_bf16.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "Qwen MLP hidden row must be finite"
            );
            self.scratch
                .hidden
                .write_bytes(0, &words_to_bytes(hidden_bf16))?;
            // Selection uses the published map before any selected demand.
            let residency_before = self.experts.residency_map()?;
            let (logits, route) = project_and_route(
                &self.metal,
                &self.router_projection,
                &self.selector,
                &self.scratch.hidden,
                &residency_before,
                mode,
            )?;
            self.execute_selected(hidden_bf16, residency_before, logits, route, capture)
        }

        /// Continue after a completed, finite router projection and checked
        /// selector readout, but before publishing expert demand.
        pub(crate) fn execute_selected(
            &mut self,
            hidden_bf16: &[u16],
            residency_before: [u32; EXPERTS],
            logits: Vec<u16>,
            route: RoutingOutput,
            capture: bool,
        ) -> Result<(MlpOutput, Option<MlpStages>)> {
            ensure!(
                hidden_bf16.len() == HIDDEN,
                "Qwen MLP needs one BF16 hidden row"
            );
            ensure!(
                hidden_bf16.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "Qwen MLP hidden row must be finite"
            );
            validate_logits(&logits)?;
            self.scratch
                .hidden
                .write_bytes(0, &words_to_bytes(hidden_bf16))?;
            let selected_ids: [u32; ROUTES] = route
                .ids
                .try_into()
                .map_err(|_| anyhow::anyhow!("router did not return ten IDs"))?;
            let selected_scores_bf16: [u16; ROUTES] = route
                .scores_bf16
                .try_into()
                .map_err(|_| anyhow::anyhow!("router did not return ten BF16 scores"))?;
            let changed = *route
                .changed
                .first()
                .context("router omitted change count")?;
            // The serving pipeline records each selected demand once.
            // Encode routed and shared work after one demand publication.
            let mut batch: Batch = self.metal.batch()?;
            let routed = self.experts.encode_routed(
                &mut batch,
                &words_to_bytes(hidden_bf16),
                &selected_ids,
                &words_to_bytes(&selected_scores_bf16),
            )?;
            self.encode_shared_device(&mut batch, &self.scratch.hidden, &routed)?;
            batch.submit().wait().context("shared MLP command")?;
            let final_bf16 = read_words(&self.scratch.final_output, HIDDEN)?;
            let output = MlpOutput {
                final_bf16,
                changed_routes: changed,
            };
            if !capture {
                return Ok((output, None));
            }
            let routed_bf16 = read_words(&routed, HIDDEN)?;
            let shared_gate_logit_bf16 = read_words(&self.scratch.scalar_logit, 1)?[0];
            let stages = MlpStages {
                output: MlpOutput {
                    final_bf16: output.final_bf16.clone(),
                    changed_routes: changed,
                },
                residency_before,
                router_logits_bf16: logits,
                selected_ids,
                selected_scores_bf16,
                routed_bf16,
                shared_gate_proj_bf16: read_words(&self.scratch.gate, SHARED)?,
                shared_up_proj_bf16: read_words(&self.scratch.up, SHARED)?,
                shared_activation_bf16: read_words(&self.scratch.activation, SHARED)?,
                shared_down_bf16: read_words(&self.scratch.down, HIDDEN)?,
                shared_gate_logit_bf16,
                shared_bf16: read_words(&self.scratch.shared, HIDDEN)?,
            };
            Ok((output, Some(stages)))
        }
    }

    #[cfg(test)]
    mod routing_tests {
        use super::*;
        use crate::package::DirectWeight;
        use serde_json::json;
        use std::io::Write;

        struct SyntheticPackage(std::path::PathBuf);
        impl Drop for SyntheticPackage {
            fn drop(&mut self) {
                for name in ["package_manifest.json", "experts.safetensors"] {
                    let _ = std::fs::remove_file(self.0.join(name));
                }
                let _ = std::fs::remove_dir(&self.0);
            }
        }

        // Only synthetic metadata and a sparse zero-filled expert payload. This
        // exercises real pool clocks/publication without an external package.
        fn synthetic_experts(metal: &Metal) -> Result<(BoundedExperts, SyntheticPackage)> {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let root = std::env::temp_dir()
                .join(format!("moespresso-router-{}-{nonce}", std::process::id()));
            std::fs::create_dir(&root)?;
            let owner = SyntheticPackage(root);
            let mut projections = serde_json::Map::new();
            let mut tensors = Vec::new();
            let mut offset = 0;
            for (name, out, logical, stored) in [
                ("gate", 640, 2560, 2560),
                ("up", 640, 2560, 2560),
                ("down", 2560, 640, 768),
            ] {
                let row = stored / 256 * 76;
                projections.insert(format!("{name}_proj"), json!({
                    "codec":"iqk", "iqk_codec":"iq2_k", "layout":"iqk_relayout",
                    "bits":2,"ggml_type":137,"bytes_per_block":76,"weights_per_block":256,
                    "row_meta_bytes":0,"bytes_per_row":row,"in_features":stored,
                    "logical_in_features":logical,"stored_in_features":stored,"zero_padding":stored-logical,
                    "blocks":{"offset":offset,"nbytes":out*row,"shape":[out,row],"dtype":"U8"}
                }));
                tensors.push(
                    json!({"kind":"expert","layer_index":0,"projection":name,"format":"iqk",
                        "key_prefix":"test.experts","shard":"experts.safetensors",
                        "format_params":{"iqk_codec":"iq2_k","layout":"iqk_relayout",
                            "logical_shape":[out,logical],"stored_shape":[out,stored],"bits":2,
                            "ggml_type":137,"bytes_per_block":76,"weights_per_block":256,
                            "row_meta_bytes":0,"zero_padding":stored-logical}
                    }),
                );
                offset += out * row;
            }
            let bundles = json!({"version":1,"layers":{"0":{"num_experts":EXPERTS,"row_bytes":offset,"projections":projections}}});
            let payload = EXPERTS * offset;
            let header = serde_json::to_vec(&json!({
                "__metadata__":{"expert_bundles":bundles.to_string()},
                "test.experts.tq_bundle":{"dtype":"U8","shape":[EXPERTS,offset],"data_offsets":[0,payload]}
            }))?;
            let bytes = 8 + header.len() + payload;
            let mut file = std::fs::File::create_new(owner.0.join("experts.safetensors"))?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            file.set_len(bytes as u64)?;
            let manifest = json!({"schema_version":{"major":1},"artifact_kind":"package_manifest",
                "artifact_id":"synthetic-router-pool","package_format":"mjtq","package_format_version":1,
                "required_features":[],"files":[{"path":"experts.safetensors","size_bytes":bytes}],
                "tensors":tensors,"architecture":{},"tokenizer":{}});
            std::fs::write(
                owner.0.join("package_manifest.json"),
                serde_json::to_vec(&manifest)?,
            )?;
            let package = Package::open(&owner.0)?;
            Ok((
                BoundedExperts::new(metal, package.expert_layer(0)?, 10)?,
                owner,
            ))
        }

        fn dense(
            metal: &Metal,
            format: WeightFormat,
            out: usize,
            input: usize,
            value: u16,
        ) -> Result<DenseMatvec> {
            let bytes = match format {
                WeightFormat::Bf16 => value.to_le_bytes().repeat(out * input),
                WeightFormat::Q6K => vec![0; out * (input / 256) * 210],
                WeightFormat::Q8_0 => vec![0; out * (input / 32) * 34],
            };
            DenseMatvec::new(
                metal,
                &DirectWeight {
                    role: "synthetic".into(),
                    format,
                    shape: vec![out, input],
                    bytes,
                },
            )
        }

        #[test]
        fn projected_nonfinite_logits_reject_before_any_pool_mutation() -> Result<()> {
            let metal = Metal::new()?;
            let (mut experts, _files) = synthetic_experts(&metal)?;
            experts.seed_residents(&[7])?;
            let before = experts.snapshot();
            assert_eq!(before.clock, 1);
            let shared_gate_proj = dense(&metal, WeightFormat::Q6K, SHARED, HIDDEN, 0)?;
            let shared_up_proj = dense(&metal, WeightFormat::Q6K, SHARED, HIDDEN, 0)?;
            let shared_gate = dense(&metal, WeightFormat::Q6K, 1, HIDDEN, 0)?;
            let shared_front_matrices =
                shared_front_matrices([&shared_gate_proj, &shared_up_proj, &shared_gate])?;
            let mut mlp = Qwen4Mlp {
                handoff: Some(crate::decode_handoff::DecodeHandoff::new(&metal)?),
                metal: metal.clone(),
                router_scratch: RouterScratch::new(&metal)?,
                router_projection: dense(
                    &metal,
                    WeightFormat::Bf16,
                    EXPERTS,
                    HIDDEN,
                    bf16::from_f32(2.).to_bits(),
                )?,
                selector: CachePriorRouter::new(&metal)?,
                experts,
                shared_gate_proj,
                shared_up_proj,
                shared_down_proj: dense(&metal, WeightFormat::Q8_0, HIDDEN, SHARED, 0)?,
                shared_gate,
                shared_front: shared_front_pipeline(&metal)?,
                shared_front_matrices,
                activate: metal.compile_msl(SOURCE, "qwen4_shared_activate")?,
                finish: metal.compile_msl(SOURCE, "qwen4_shared_finish")?,
                scratch: SharedScratch {
                    hidden: buffer(&metal, HIDDEN)?,
                    gate: buffer(&metal, SHARED)?,
                    up: buffer(&metal, SHARED)?,
                    activation: buffer(&metal, SHARED)?,
                    down: buffer(&metal, HIDDEN)?,
                    scalar_logit: buffer(&metal, 1)?,
                    shared: buffer(&metal, HIDDEN)?,
                    final_output: buffer(&metal, HIDDEN)?,
                },
            };
            let hidden = vec![0x7f7fu16; HIDDEN];
            assert!(hidden.iter().all(|&v| bf16::from_bits(v).is_finite()));
            for mode in [RoutingMode::Unbiased, RoutingMode::CachePrior] {
                let error = mlp
                    .run(&hidden, mode)
                    .err()
                    .context("overflow was accepted")?;
                assert_eq!(error.to_string(), "routing requires finite BF16 logits");
                assert_eq!(mlp.experts.snapshot(), before);
                let device_hidden = buffer(&metal, HIDDEN)?;
                device_hidden.write_bytes(0, &words_to_bytes(&hidden))?;
                let mut batch = metal.batch()?;
                let encoded = mlp.encode_router_device(
                    &mut batch,
                    &device_hidden,
                    &mlp.experts.residency_map()?,
                    mode,
                )?;
                batch.submit().wait()?;
                let error = encoded
                    .read()
                    .err()
                    .context("merged router overflow was accepted")?;
                assert_eq!(error.to_string(), "routing requires finite BF16 logits");
                assert_eq!(mlp.experts.snapshot(), before);
                let error = mlp
                    .run(&vec![0x7fc0; HIDDEN], mode)
                    .err()
                    .context("NaN hidden was accepted")?;
                assert_eq!(error.to_string(), "Qwen MLP hidden row must be finite");
                assert_eq!(mlp.experts.snapshot(), before);
                let status = DeviceStatus::new(&metal)?;
                let mut stream = metal.stream()?;
                let error = mlp
                    .run_device(&mut stream, &device_hidden, mode, &status)
                    .err()
                    .context("production router overflow was accepted")?;
                assert!(
                    format!("{error:#}").contains("device execution produced nonfinite values")
                );
                assert_eq!(mlp.experts.snapshot(), before);
                // The shared submission remains owned even if hardware has
                // finished. Status can be inspected independently of it.
                assert!(status.check().is_err());
                assert!(mlp.scratch.gate.read_bytes(0, &mut [0; 2]).is_err());
                assert!(
                    device_hidden
                        .write_bytes(0, &words_to_bytes(&hidden))
                        .is_err()
                );
                drop(stream);
                mlp.scratch.gate.read_bytes(0, &mut [0; 2])?;
                device_hidden.write_bytes(0, &words_to_bytes(&hidden))?;
                status.reset()?;
                assert_eq!(mlp.experts.snapshot(), before);
            }
            let package = Package::open(&_files.0)?;
            let batch = crate::mlp_batch::MlpBatch::from_decode(&metal, &package, 0, &mlp)?;
            let two_rows = words_to_bytes(&hidden.repeat(2));
            let error = batch
                .run_stages(&two_rows, 2, &mut mlp.experts)
                .err()
                .context("prefill overflow was accepted")?;
            assert_eq!(error.to_string(), "routing requires finite BF16 logits");
            assert_eq!(mlp.experts.snapshot(), before);
            let input = metal.shared_buffer(two_rows.len())?;
            input.write_bytes(0, &two_rows)?;
            // SAFETY: The synthetic hidden values are finite and the producer
            // upload completed; overflow occurs only in the router projection.
            let error = unsafe { batch.run_buffer(&input, 2, &mut mlp.experts) }
                .err()
                .context("noncapture prefill overflow was accepted")?;
            assert_eq!(error.to_string(), "routing requires finite BF16 logits");
            assert_eq!(mlp.experts.snapshot(), before);
            let device_hidden = buffer(&metal, HIDDEN)?;
            device_hidden.write_bytes(0, &vec![0; HIDDEN * 2])?;
            let status = DeviceStatus::new(&metal)?;
            let mut stream = metal.stream()?;
            let fault = metal.test_fail_completed_command_after(0)?;
            let error = mlp
                .run_device(
                    &mut stream,
                    &device_hidden,
                    RoutingMode::CachePrior,
                    &status,
                )
                .err()
                .context("injected router completion error was accepted")?;
            assert!(format!("{error:#}").contains("injected completed Metal command error"));
            assert!(fault.fired());
            assert_eq!(mlp.experts.snapshot(), before);
            assert!(mlp.scratch.gate.read_bytes(0, &mut [0; 2]).is_err());
            stream.abort_and_wait()?;
            drop(fault);
            mlp.scratch.gate.read_bytes(0, &mut [0; 2])?;
            device_hidden.write_bytes(0, &vec![0; HIDDEN * 2])?;
            status.reset()?;
            let output = mlp.run_device(
                &mut stream,
                &device_hidden,
                RoutingMode::CachePrior,
                &status,
            )?;
            stream.wait()?;
            status.check()?;
            assert!(
                read_words(&output.output, HIDDEN)?
                    .iter()
                    .all(|&word| word == 0)
            );
            assert_eq!(mlp.experts.snapshot().clock, before.clock + ROUTES as u64);
            Ok(())
        }

        #[test]
        fn shared_projection_selector_matches_separate_commands() -> Result<()> {
            let metal = Metal::new()?;
            let mut matrix = vec![0u8; EXPERTS * HIDDEN * 2];
            for row in 0..EXPERTS {
                let offset = (row * HIDDEN + row * 5 % HIDDEN) * 2;
                matrix[offset..offset + 2]
                    .copy_from_slice(&bf16::from_f32(1.).to_bits().to_le_bytes());
            }
            let projection = DenseMatvec::new(
                &metal,
                &DirectWeight {
                    role: "synthetic".into(),
                    format: WeightFormat::Bf16,
                    shape: vec![EXPERTS, HIDDEN],
                    bytes: matrix,
                },
            )?;
            let selector = CachePriorRouter::new(&metal)?;
            let hidden: Vec<_> = (0..HIDDEN)
                .map(|i| bf16::from_f32(((i * 37) % 29) as f32 / 16. - 1.).to_bits())
                .collect();
            let input = buffer(&metal, HIDDEN)?;
            input.write_bytes(0, &words_to_bytes(&hidden))?;
            let map: [u32; EXPERTS] = std::array::from_fn(|i| {
                if i % 3 == 0 {
                    (i % 12) as u32
                } else {
                    EXPERTS as u32
                }
            });
            let expected_logits = projection.run_bf16(&metal, &hidden)?;
            for mode in [RoutingMode::Unbiased, RoutingMode::CachePrior] {
                let expected =
                    selector.route(&metal, &expected_logits, [&map, &map, &map], mode)?;
                let (logits, route) =
                    project_and_route(&metal, &projection, &selector, &input, &map, mode)?;
                assert_eq!(logits, expected_logits);
                assert_eq!(route.ids, expected.ids);
                assert_eq!(route.scores_bf16, expected.scores_bf16);
                assert_eq!(route.changed, expected.changed);
            }
            Ok(())
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::package::DirectWeight;
        use half::f16;

        fn constant_q6(rows: usize, direction: i8) -> DirectWeight {
            let mut bytes = Vec::with_capacity(rows * HIDDEN / 256 * 210);
            for row in 0..rows {
                let scale = direction
                    * if row + 1 == rows {
                        17
                    } else {
                        (row % 11 + 1) as i8
                    };
                let mut block = [0u8; 210];
                block[..128].fill(0x11);
                block[128..192].fill(0xaa);
                block[192..208].fill(scale as u8);
                block[208..210]
                    .copy_from_slice(&f16::from_f32(1.0 / 256.0).to_bits().to_le_bytes());
                for _ in 0..HIDDEN / 256 {
                    bytes.extend_from_slice(&block);
                }
            }
            DirectWeight {
                role: "synthetic shared front".into(),
                format: WeightFormat::Q6K,
                shape: vec![rows, HIDDEN],
                bytes,
            }
        }

        #[test]
        fn shared_projection_front_preserves_nonzero_boundaries_and_output_guards() -> Result<()> {
            let metal = Metal::new()?;
            let pipeline = shared_front_pipeline(&metal)?;
            let matrices = [
                DenseMatvec::new(&metal, &constant_q6(SHARED, 1))?,
                DenseMatvec::new(&metal, &constant_q6(SHARED, -1))?,
                DenseMatvec::new(&metal, &constant_q6(1, 1))?,
            ];
            let front_matrices = shared_front_matrices([&matrices[0], &matrices[1], &matrices[2]])?;
            let words: Vec<_> = (0..HIDDEN)
                .map(|i| bf16::from_f32((i % 7 + 1) as f32 / 16.0).to_bits())
                .collect();
            let hidden = buffer(&metal, HIDDEN)?;
            hidden.write_bytes(0, &words_to_bytes(&words))?;
            let widths = [SHARED, SHARED, 1];
            let guarded = |width| -> Result<Buffer> {
                let output = metal.shared_buffer(width * 2 + 16)?;
                output.write_bytes(0, &vec![0x5a; width * 2 + 16])?;
                Ok(output)
            };
            let stacked = [guarded(SHARED)?, guarded(SHARED)?, guarded(1)?];
            let separate = [guarded(SHARED)?, guarded(SHARED)?, guarded(1)?];
            let mut batch = metal.batch()?;
            // SAFETY: Declared Q6 matrices and initialized BF16 buffers have
            // distinct allocations. Each output has its full extent and a guard.
            unsafe {
                encode_stacked_shared_front(
                    &mut batch,
                    &pipeline,
                    &front_matrices,
                    &hidden,
                    [&stacked[0], &stacked[1], &stacked[2]],
                )?;
                for (matrix, output) in matrices.iter().zip(&separate) {
                    matrix.encode(&mut batch, &hidden, output, OutputPrecision::Bf16)?;
                }
            }
            batch.submit().wait()?;
            let input_sum: f64 = words
                .iter()
                .map(|&word| f64::from(bf16::from_bits(word).to_f32()))
                .sum();
            for (projection, width) in widths.into_iter().enumerate() {
                let actual = read_words(&stacked[projection], width)?;
                let reference = read_words(&separate[projection], width)?;
                let direction = if projection == 1 { -1.0 } else { 1.0 };
                for (row, (&actual, &reference)) in actual.iter().zip(&reference).enumerate() {
                    // The packed nibbles decode to q=1. Signed row scales and
                    // the FP16 block scale give this independent dot product.
                    let scale = if row + 1 == width { 17 } else { row % 11 + 1 };
                    let expected =
                        bf16::from_f32((input_sum * direction * scale as f64 / 256.0) as f32)
                            .to_f32();
                    assert_eq!(bf16::from_bits(actual).to_f32(), expected);
                    assert_eq!(bf16::from_bits(reference).to_f32(), expected);
                }
                assert_ne!(bf16::from_bits(actual[width - 1]).to_f32(), 0.0);
                for output in [&stacked[projection], &separate[projection]] {
                    let mut guard = [0u8; 16];
                    output.read_bytes(width * 2, &mut guard)?;
                    assert_eq!(guard, [0x5a; 16]);
                }
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{MlpOutput, MlpStages, Qwen4Mlp};

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use crate::metal::{Binding, Metal};
    use half::bf16;

    #[test]
    #[ignore = "requires MLP_SIGMOID_BF16_ORACLE and exclusive Metal ownership"]
    fn stable_sigmoid_matches_external_bf16_table() {
        let path = std::env::var("MLP_SIGMOID_BF16_ORACLE")
            .expect("MLP_SIGMOID_BF16_ORACLE is required for this oracle check");
        let reference = std::fs::read(path).unwrap();
        assert_eq!(reference.len(), 65536 * 2);
        let metal = Metal::new().unwrap();
        let pipeline = metal
            .compile_msl(
                include_str!("../kernels/metal/mlp.metal"),
                "qwen4_sigmoid_bf16_words",
            )
            .unwrap();
        let input = metal.shared_buffer(65536 * 2).unwrap();
        let output = metal.shared_buffer(65536 * 2).unwrap();
        let words: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        input.write_bytes(0, &words).unwrap();
        let mut batch = metal.batch().unwrap();
        // SAFETY: Both buffers hold all 65,536 BF16 words, and one thread
        // addresses exactly one input and output word.
        unsafe {
            batch
                .dispatch_1d(
                    &pipeline,
                    &[
                        Binding {
                            buffer: &input,
                            offset: 0,
                        },
                        Binding {
                            buffer: &output,
                            offset: 0,
                        },
                    ],
                    65536,
                    256,
                )
                .unwrap();
        }
        batch.submit().wait().unwrap();
        let mut actual = vec![0u8; reference.len()];
        output.read_bytes(0, &mut actual).unwrap();
        let differences: Vec<_> = actual
            .as_chunks::<2>()
            .0
            .iter()
            .zip(reference.as_chunks::<2>().0)
            .enumerate()
            .filter(|(bits, (got, want))| bf16::from_bits(*bits as u16).is_finite() && got != want)
            .take(12)
            .collect();
        assert!(
            differences.is_empty(),
            "BF16 sigmoid differs at {differences:?}"
        );
    }
}

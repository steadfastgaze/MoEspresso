//! One-token Qwen sparse-attention projections through partial RoPE.
//!
//! This seam produces the index, query, key and value rows consumed by the
//! attention path. It does not own KV state, token selection or attention.

use anyhow::{Result, ensure};
use half::bf16;

pub const HIDDEN: usize = 2560;
pub const INDEX_HEADS: usize = 4;
pub const QUERY_HEADS: usize = 24;
pub const KV_HEADS: usize = 2;
pub const INDEX_HEAD_WIDTH: usize = 128;
pub const HEAD_WIDTH: usize = 256;
pub const ROTARY_WIDTH: usize = 64;
pub const ROPE_BASE: f32 = 10_000_000.0;

/// The shared one-token factors are BF16 after FP32 frequency and trig math.
#[derive(Clone, Debug)]
pub struct QsaRopeFactors {
    pub cosine_bf16: [u16; ROTARY_WIDTH],
    pub sine_bf16: [u16; ROTARY_WIDTH],
}

impl QsaRopeFactors {
    pub fn from_words(
        cosine_bf16: [u16; ROTARY_WIDTH],
        sine_bf16: [u16; ROTARY_WIDTH],
    ) -> Result<Self> {
        ensure!(
            cosine_bf16
                .iter()
                .chain(sine_bf16.iter())
                .all(|&v| bf16::from_bits(v).is_finite()),
            "QSA RoPE factors must be finite BF16 values"
        );
        Ok(Self {
            cosine_bf16,
            sine_bf16,
        })
    }
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use anyhow::{Context, Result, ensure};

    use crate::kquant::{DenseMatvec, OutputPrecision};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use crate::package::{Package, WeightFormat};

    const MSL: &str = include_str!("../kernels/metal/qsa_projection.metal");
    const EPSILON: f32 = 1e-6;

    /// Reusable FP32 Metal factor generation shared by projection and index compression.
    pub struct QsaRope {
        metal: Metal,
        pipeline: Pipeline,
        positions: Buffer,
        cosine: Buffer,
        sine: Buffer,
    }

    impl QsaRope {
        pub fn new(metal: &Metal) -> Result<Self> {
            let pipeline = metal.compile_msl_strict(MSL, "qsa_rope_factors")?;
            ensure!(
                pipeline.max_threads_per_threadgroup() >= 32,
                "QSA RoPE requires 32 Metal threads"
            );
            Ok(Self {
                metal: metal.clone(),
                pipeline,
                positions: metal.shared_buffer(3 * 8)?,
                cosine: metal.shared_buffer(ROTARY_WIDTH * 2)?,
                sine: metal.shared_buffer(ROTARY_WIDTH * 2)?,
            })
        }

        /// Axes are time, height and width, interleaved over 11, 11 and 10 frequencies.
        fn encode(&mut self, batch: &mut Batch, positions: [i64; 3]) -> Result<()> {
            self.positions.write_bytes(
                0,
                &positions
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let bindings = [&self.positions, &self.cosine, &self.sine]
                .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: 32 threads read one of three i64 positions and each writes
            // two disjoint BF16 lanes in each 64-element output buffer.
            unsafe {
                batch.dispatch_1d(&self.pipeline, &bindings, 32, 32)?;
            }
            Ok(())
        }

        fn completed(&self) -> Result<QsaRopeFactors> {
            QsaRopeFactors::from_words(
                read_words(&self.cosine, ROTARY_WIDTH)?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("cosine width changed"))?,
                read_words(&self.sine, ROTARY_WIDTH)?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("sine width changed"))?,
            )
        }

        pub fn at(&mut self, positions: [i64; 3]) -> Result<QsaRopeFactors> {
            let mut batch = self.metal.batch()?;
            self.encode(&mut batch, positions)?;
            batch
                .submit()
                .wait()
                .context("QSA rotary factors command")?;
            self.completed()
        }
    }

    fn uploaded(metal: &Metal, bytes: &[u8]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(bytes.len())?;
        buffer.write_bytes(0, bytes)?;
        Ok(buffer)
    }

    fn words_bytes(words: &[u16]) -> Vec<u8> {
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

    fn projection(
        metal: &Metal,
        package: &Package,
        path: String,
        output: usize,
    ) -> Result<DenseMatvec> {
        let weight = package.direct_weight(&path)?;
        ensure!(
            weight.format == WeightFormat::Q6K && weight.shape == [output, HIDDEN],
            "{path} differs from the released Q6_K QSA projection"
        );
        DenseMatvec::new(metal, &weight)
    }

    fn norm(metal: &Metal, package: &Package, path: String, width: usize) -> Result<Buffer> {
        let weight = package.direct_weight(&path)?;
        ensure!(
            weight.format == WeightFormat::Bf16 && weight.shape == [width],
            "{path} differs from the released QSA norm"
        );
        ensure!(
            weight
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .all(|word| { bf16::from_bits(u16::from_le_bytes(*word)).is_finite() }),
            "{path} contains nonfinite norm weights"
        );
        uploaded(metal, &weight.bytes)
    }

    struct Scratch {
        hidden: Buffer,
        index_projection: Buffer,
        query_projection: Buffer,
        key_projection: Buffer,
        value_projection: Buffer,
        cosine: Buffer,
        sine: Buffer,
        index_queries: Buffer,
        raw_index_key: Buffer,
        queries: Buffer,
        query_gate: Buffer,
        keys: Buffer,
    }

    pub struct QsaProjectionOutput {
        pub index_queries_bf16: Vec<u16>,
        pub raw_index_key_bf16: Vec<u16>,
        pub queries_bf16: Vec<u16>,
        pub query_gate_bf16: Vec<u16>,
        pub keys_bf16: Vec<u16>,
        pub values_bf16: Vec<u16>,
    }

    impl QsaProjectionOutput {
        #[cfg(test)]
        pub(super) fn checked_finite(&self) -> Result<()> {
            ensure!(
                [
                    &self.index_queries_bf16,
                    &self.raw_index_key_bf16,
                    &self.queries_bf16,
                    &self.query_gate_bf16,
                    &self.keys_bf16,
                    &self.values_bf16,
                ]
                .into_iter()
                .flatten()
                .all(|&word| bf16::from_bits(word).is_finite()),
                "QSA projection produced nonfinite BF16"
            );
            Ok(())
        }
    }

    pub struct QsaProjectionStages {
        pub output: QsaProjectionOutput,
        pub factors: QsaRopeFactors,
        pub index_projection_bf16: Vec<u16>,
        pub query_projection_bf16: Vec<u16>,
        pub key_projection_bf16: Vec<u16>,
        pub value_projection_bf16: Vec<u16>,
    }

    pub struct QsaProjector {
        metal: Metal,
        rope: QsaRope,
        index: DenseMatvec,
        query: DenseMatvec,
        key: DenseMatvec,
        value: DenseMatvec,
        index_norm: Buffer,
        query_norm: Buffer,
        key_norm: Buffer,
        epsilon: Buffer,
        pipeline: Pipeline,
        scratch: Scratch,
    }

    pub(crate) struct QsaSharedWeights {
        pub matrices: [crate::kquant::DeviceMatrix; 4],
        pub index_norm: Buffer,
        pub query_norm: Buffer,
        pub key_norm: Buffer,
        pub epsilon: Buffer,
    }

    impl QsaProjector {
        pub(crate) fn shared_weights(&self) -> QsaSharedWeights {
            QsaSharedWeights {
                matrices: [&self.index, &self.query, &self.key, &self.value]
                    .map(DenseMatvec::shared_matrix),
                index_norm: self.index_norm.clone(),
                query_norm: self.query_norm.clone(),
                key_norm: self.key_norm.clone(),
                epsilon: self.epsilon.clone(),
            }
        }
        pub fn new(metal: &Metal, package: &Package, layer: usize) -> Result<Self> {
            ensure!(
                layer < 48 && layer % 4 == 3,
                "released QSA uses layers 3, 7, ..., 47"
            );
            let prefix = format!("layers.{layer}.mixer.module");
            let index = projection(
                metal,
                package,
                format!("{prefix}.indexer.index_qk_proj"),
                640,
            )?;
            let query = projection(metal, package, format!("{prefix}.q_proj"), 12288)?;
            let key = projection(metal, package, format!("{prefix}.k_proj"), 512)?;
            let value = projection(metal, package, format!("{prefix}.v_proj"), 512)?;
            let index_norm = norm(
                metal,
                package,
                format!("{prefix}.indexer.q_layernorm"),
                INDEX_HEAD_WIDTH,
            )?;
            let query_norm = norm(metal, package, format!("{prefix}.q_norm"), HEAD_WIDTH)?;
            let key_norm = norm(metal, package, format!("{prefix}.k_norm"), HEAD_WIDTH)?;
            let epsilon = uploaded(metal, &EPSILON.to_le_bytes())?;
            let pipeline = metal.compile_msl_strict(MSL, "qsa_project_rope_one")?;
            ensure!(
                pipeline.max_threads_per_threadgroup() >= 32,
                "QSA projection requires one 32-thread SIMD group"
            );
            let scratch = Scratch {
                hidden: metal.shared_buffer(HIDDEN * 2)?,
                index_projection: metal.shared_buffer(640 * 2)?,
                query_projection: metal.shared_buffer(12288 * 2)?,
                key_projection: metal.shared_buffer(512 * 2)?,
                value_projection: metal.shared_buffer(512 * 2)?,
                cosine: metal.shared_buffer(ROTARY_WIDTH * 2)?,
                sine: metal.shared_buffer(ROTARY_WIDTH * 2)?,
                index_queries: metal.shared_buffer(INDEX_HEADS * INDEX_HEAD_WIDTH * 2)?,
                raw_index_key: metal.shared_buffer(INDEX_HEAD_WIDTH * 2)?,
                queries: metal.shared_buffer(QUERY_HEADS * HEAD_WIDTH * 2)?,
                query_gate: metal.shared_buffer(QUERY_HEADS * HEAD_WIDTH * 2)?,
                keys: metal.shared_buffer(KV_HEADS * HEAD_WIDTH * 2)?,
            };
            Ok(Self {
                metal: metal.clone(),
                rope: QsaRope::new(metal)?,
                index,
                query,
                key,
                value,
                index_norm,
                query_norm,
                key_norm,
                epsilon,
                pipeline,
                scratch,
            })
        }

        pub fn project_at(
            &mut self,
            hidden_bf16: &[u16],
            positions: [i64; 3],
        ) -> Result<QsaProjectionOutput> {
            let factors = self.rope.at(positions)?;
            self.project(hidden_bf16, &factors)
        }

        pub fn project(
            &mut self,
            hidden_bf16: &[u16],
            factors: &QsaRopeFactors,
        ) -> Result<QsaProjectionOutput> {
            Ok(self.execute(hidden_bf16, factors, false)?.0)
        }

        pub fn project_stages(
            &mut self,
            hidden_bf16: &[u16],
            factors: &QsaRopeFactors,
        ) -> Result<QsaProjectionStages> {
            self.execute(hidden_bf16, factors, true)?
                .1
                .context("QSA stages were not requested")
        }

        /// Encode after a preceding producer in this serial batch initializes
        /// a fixed-width hidden row. The caller checks the completed row and
        /// the projection outputs before advancing the index state. Geometry,
        /// not BF16 values, determines every address in these kernels.
        pub(crate) fn encode_device(
            &mut self,
            batch: &mut Batch,
            hidden: &Buffer,
            positions: [i64; 3],
        ) -> Result<()> {
            ensure!(
                hidden.byte_len() == HIDDEN * 2,
                "QSA hidden buffer width differs"
            );
            self.rope.encode(batch, positions)?;
            self.encode_projection(batch, hidden, &self.rope.cosine, &self.rope.sine)
        }

        /// Read only after the producing batch completed successfully.
        /// Dropping an unsubmitted batch does not initialize these outputs.
        #[cfg(test)]
        pub(crate) fn read_device_output(&self) -> Result<QsaProjectionOutput> {
            self.rope.completed()?;
            let output = self.read_output()?;
            output.checked_finite()?;
            Ok(output)
        }

        /// Raw index keys and normalized index queries for the device selector.
        pub(crate) fn index_buffers(&self) -> (&Buffer, &Buffer) {
            (&self.scratch.raw_index_key, &self.scratch.index_queries)
        }

        pub(crate) fn attention_buffers(&self) -> (&Buffer, &Buffer, &Buffer, &Buffer) {
            (
                &self.scratch.queries,
                &self.scratch.query_gate,
                &self.scratch.keys,
                &self.scratch.value_projection,
            )
        }

        pub(crate) fn encode_validation(
            &self,
            batch: &mut Batch,
            status: &crate::device_status::DeviceStatus,
        ) -> Result<()> {
            for buffer in [
                &self.scratch.queries,
                &self.scratch.query_gate,
                &self.scratch.keys,
                &self.scratch.value_projection,
            ] {
                status.encode_bf16(batch, buffer)?;
            }
            Ok(())
        }

        fn encode_projection(
            &self,
            batch: &mut Batch,
            hidden: &Buffer,
            cosine: &Buffer,
            sine: &Buffer,
        ) -> Result<()> {
            // SAFETY: The four BF16 projections have fixed disjoint outputs.
            // The final kernel has 30 groups of 32 threads and reads only
            // those completed projections, norms and fixed-width factors.
            unsafe {
                for (projection, output) in [
                    (&self.index, &self.scratch.index_projection),
                    (&self.query, &self.scratch.query_projection),
                    (&self.key, &self.scratch.key_projection),
                    (&self.value, &self.scratch.value_projection),
                ] {
                    projection.encode(batch, hidden, output, OutputPrecision::Bf16)?;
                }
                let bindings = [
                    &self.scratch.index_projection,
                    &self.scratch.query_projection,
                    &self.scratch.key_projection,
                    &self.index_norm,
                    &self.query_norm,
                    &self.key_norm,
                    cosine,
                    sine,
                    &self.scratch.index_queries,
                    &self.scratch.raw_index_key,
                    &self.scratch.queries,
                    &self.scratch.query_gate,
                    &self.scratch.keys,
                    &self.epsilon,
                ]
                .map(|buffer| Binding { buffer, offset: 0 });
                batch.dispatch_1d(&self.pipeline, &bindings, 30 * 32, 32)?;
            }
            Ok(())
        }

        fn read_output(&self) -> Result<QsaProjectionOutput> {
            let output = QsaProjectionOutput {
                index_queries_bf16: read_words(
                    &self.scratch.index_queries,
                    INDEX_HEADS * INDEX_HEAD_WIDTH,
                )?,
                raw_index_key_bf16: read_words(&self.scratch.raw_index_key, INDEX_HEAD_WIDTH)?,
                queries_bf16: read_words(&self.scratch.queries, QUERY_HEADS * HEAD_WIDTH)?,
                query_gate_bf16: read_words(&self.scratch.query_gate, QUERY_HEADS * HEAD_WIDTH)?,
                keys_bf16: read_words(&self.scratch.keys, KV_HEADS * HEAD_WIDTH)?,
                values_bf16: read_words(&self.scratch.value_projection, KV_HEADS * HEAD_WIDTH)?,
            };
            Ok(output)
        }

        fn execute(
            &mut self,
            hidden_bf16: &[u16],
            factors: &QsaRopeFactors,
            capture: bool,
        ) -> Result<(QsaProjectionOutput, Option<QsaProjectionStages>)> {
            ensure!(
                hidden_bf16.len() == HIDDEN,
                "QSA requires one BF16 hidden row"
            );
            ensure!(
                hidden_bf16.iter().all(|&v| bf16::from_bits(v).is_finite()),
                "QSA hidden row must be finite"
            );
            QsaRopeFactors::from_words(factors.cosine_bf16, factors.sine_bf16)?;
            self.scratch
                .hidden
                .write_bytes(0, &words_bytes(hidden_bf16))?;
            self.scratch
                .cosine
                .write_bytes(0, &words_bytes(&factors.cosine_bf16))?;
            self.scratch
                .sine
                .write_bytes(0, &words_bytes(&factors.sine_bf16))?;
            let mut batch: Batch = self.metal.batch()?;
            self.encode_projection(
                &mut batch,
                &self.scratch.hidden,
                &self.scratch.cosine,
                &self.scratch.sine,
            )?;
            batch.submit().wait().context("QSA projection command")?;
            let output = self.read_output()?;
            if !capture {
                return Ok((output, None));
            }
            let stages = QsaProjectionStages {
                output: QsaProjectionOutput {
                    index_queries_bf16: output.index_queries_bf16.clone(),
                    raw_index_key_bf16: output.raw_index_key_bf16.clone(),
                    queries_bf16: output.queries_bf16.clone(),
                    query_gate_bf16: output.query_gate_bf16.clone(),
                    keys_bf16: output.keys_bf16.clone(),
                    values_bf16: output.values_bf16.clone(),
                },
                factors: factors.clone(),
                index_projection_bf16: read_words(&self.scratch.index_projection, 640)?,
                query_projection_bf16: read_words(&self.scratch.query_projection, 12288)?,
                key_projection_bf16: read_words(&self.scratch.key_projection, 512)?,
                value_projection_bf16: read_words(&self.scratch.value_projection, 512)?,
            };
            Ok((output, Some(stages)))
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{QsaProjectionOutput, QsaProjectionStages, QsaProjector, QsaRope};

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::{metal::Metal, package::Package};
    #[cfg(target_os = "macos")]
    use anyhow::Context;

    #[test]
    #[cfg(target_os = "macos")]
    fn mrope_sections_interleave_three_position_axes() {
        let mut rope = QsaRope::new(&crate::metal::Metal::new().unwrap()).unwrap();
        let factors = rope.at([0, 0, 0]).unwrap();
        assert!(
            factors
                .cosine_bf16
                .iter()
                .all(|&v| v == bf16::from_f32(1.0).to_bits())
        );
        assert!(factors.sine_bf16.iter().all(|&v| v == 0));
        let mixed = rope.at([0, 1, 2]).unwrap();
        assert_eq!(mixed.cosine_bf16[0], bf16::from_f32(1.0).to_bits());
        assert_ne!(mixed.cosine_bf16[1], bf16::from_f32(1.0).to_bits());
        assert_eq!(mixed.cosine_bf16[..32], mixed.cosine_bf16[32..]);
        assert_eq!(mixed.sine_bf16[..32], mixed.sine_bf16[32..]);
    }

    #[test]
    fn factor_words_reject_nonfinite_values() {
        let mut cosine = [bf16::from_f32(1.0).to_bits(); ROTARY_WIDTH];
        let sine = [0u16; ROTARY_WIDTH];
        assert!(QsaRopeFactors::from_words(cosine, sine).is_ok());
        cosine[47] = 0x7fc0;
        assert!(QsaRopeFactors::from_words(cosine, sine).is_err());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn completed_device_outputs_reject_nonfinite_fields() {
        fn set_field(output: &mut QsaProjectionOutput, index: usize, word: u16) {
            let field = match index {
                0 => &mut output.index_queries_bf16,
                1 => &mut output.raw_index_key_bf16,
                2 => &mut output.queries_bf16,
                3 => &mut output.query_gate_bf16,
                4 => &mut output.keys_bf16,
                _ => &mut output.values_bf16,
            };
            field[0] = word;
        }
        let mut output = QsaProjectionOutput {
            index_queries_bf16: vec![0],
            raw_index_key_bf16: vec![0],
            queries_bf16: vec![0],
            query_gate_bf16: vec![0],
            keys_bf16: vec![0],
            values_bf16: vec![0],
        };
        output.checked_finite().unwrap();
        for index in 0..6 {
            set_field(&mut output, index, 0x7fc0);
            assert!(output.checked_finite().is_err());
            set_field(&mut output, index, 0);
        }
    }

    #[test]
    #[ignore = "requires QSA_PROJECTION_FIXTURE_MANIFEST and its package"]
    #[cfg(target_os = "macos")]
    fn device_encoded_projection_matches_standalone_reference() -> Result<()> {
        let manifest_path = std::env::var("QSA_PROJECTION_FIXTURE_MANIFEST")?;
        let manifest_path = std::path::Path::new(&manifest_path);
        let root = manifest_path.parent().context("fixture has no parent")?;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest_path)?)?;
        ensure!(
            manifest["schema"] == "qwen4-qsa-project-rope-fixture-v1",
            "unexpected QSA projection fixture"
        );
        let package = Package::open(manifest["package_path"].as_str().context("package path")?)?;
        let layer = manifest["layer"].as_u64().context("layer")? as usize;
        let metal = Metal::new()?;
        let mut projector = QsaProjector::new(&metal, &package, layer)?;
        for (name, case) in manifest["cases"].as_object().context("cases")? {
            let relative = case["files"]["hidden"]["path"]
                .as_str()
                .context("hidden path")?;
            let hidden_bytes = std::fs::read(root.join(relative))?;
            let hidden: Vec<u16> = hidden_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| u16::from_le_bytes(*word))
                .collect();
            ensure!(hidden.len() == HIDDEN, "{name} hidden width differs");
            let positions: [i64; 3] = case["position_ids"]
                .as_array()
                .context("positions")?
                .iter()
                .map(|v| v.as_i64().context("position"))
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| anyhow::anyhow!("position width"))?;
            let standalone = projector.project_at(&hidden, positions)?;
            let device_hidden = metal.shared_buffer(HIDDEN * 2)?;
            device_hidden.write_bytes(0, &hidden_bytes)?;
            let mut batch = metal.batch()?;
            projector.encode_device(&mut batch, &device_hidden, positions)?;
            batch.submit().wait()?;
            let joined = projector.read_device_output()?;
            for (label, actual, expected) in [
                (
                    "index_queries",
                    joined.index_queries_bf16,
                    standalone.index_queries_bf16,
                ),
                (
                    "raw_index_key",
                    joined.raw_index_key_bf16,
                    standalone.raw_index_key_bf16,
                ),
                ("queries", joined.queries_bf16, standalone.queries_bf16),
                (
                    "query_gate",
                    joined.query_gate_bf16,
                    standalone.query_gate_bf16,
                ),
                ("keys", joined.keys_bf16, standalone.keys_bf16),
                ("values", joined.values_bf16, standalone.values_bf16),
            ] {
                ensure!(actual == expected, "{name}/{label} device path differs");
            }
        }
        Ok(())
    }
}

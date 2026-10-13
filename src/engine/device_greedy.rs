//! Greedy BF16 log-probability sampling on the engine's ordered Metal stream.

use anyhow::{Context, Result, ensure};
use half::bf16;

use crate::device_status::DeviceStatus;
use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
use crate::sampling::SamplingOptions;

const GROUP: usize = 256;
const SOURCE: &str = include_str!("../../kernels/metal/device_greedy.metal");
const LSE_SOURCE: &str = include_str!("../../kernels/metal/sampling_lse.metal");

pub(super) struct DeviceGreedy {
    vocabulary: usize,
    presence: Pipeline,
    lse_block: Pipeline,
    lse_looped: Pipeline,
    blocks: Pipeline,
    finish: Pipeline,
    adjusted: Buffer,
    seen: Buffer,
    seen_host: Vec<u8>,
    count: Buffer,
    logsumexp: Buffer,
    partials: Buffer,
    selected: Buffer,
}

impl DeviceGreedy {
    pub(super) fn new(metal: &Metal, vocabulary: usize) -> Result<Self> {
        ensure!(
            (1..=i32::MAX as usize).contains(&vocabulary),
            "greedy vocabulary is invalid"
        );
        let bytes = vocabulary
            .checked_mul(2)
            .context("greedy logit byte count overflow")?;
        let blocks = vocabulary.div_ceil(GROUP);
        let seen_bytes = vocabulary.div_ceil(32) * 4;
        let presence = metal.compile_msl_strict(SOURCE, "greedy_presence_bf16")?;
        let lse_block = metal.compile_msl_strict(LSE_SOURCE, "sampling_bf16_logsumexp_block")?;
        let lse_looped = metal.compile_msl_strict(LSE_SOURCE, "sampling_bf16_logsumexp_looped")?;
        let blocks_pipeline = metal.compile_msl_strict(SOURCE, "greedy_logprob_blocks")?;
        let finish = metal.compile_msl_strict(SOURCE, "greedy_logprob_finish")?;
        ensure!(
            presence.max_threads_per_threadgroup() >= GROUP
                && blocks_pipeline.max_threads_per_threadgroup() >= GROUP
                && finish.max_threads_per_threadgroup() >= GROUP
                && lse_block.max_threads_per_threadgroup() >= 32
                && lse_looped.max_threads_per_threadgroup() >= 32,
            "greedy sampling threadgroup is unsupported"
        );
        let count = metal.shared_buffer(4)?;
        count.write_bytes(0, &u32::try_from(vocabulary)?.to_le_bytes())?;
        Ok(Self {
            vocabulary,
            presence,
            lse_block,
            lse_looped,
            blocks: blocks_pipeline,
            finish,
            adjusted: metal.shared_buffer(bytes)?,
            seen: metal.shared_buffer(seen_bytes)?,
            seen_host: vec![0; seen_bytes],
            count,
            logsumexp: metal.shared_buffer(2)?,
            partials: metal.shared_buffer(blocks * 8)?,
            selected: metal.shared_buffer(4)?,
        })
    }

    pub(super) fn requested_bytes(&self) -> usize {
        self.adjusted.byte_len()
            + self.seen.byte_len()
            + self.count.byte_len()
            + self.logsumexp.byte_len()
            + self.partials.byte_len()
            + self.selected.byte_len()
    }

    pub(super) fn host_bytes(&self) -> usize {
        self.seen_host.len()
    }

    /// The caller supplies validated sampling options and a finite committed
    /// or ordered final-head row. It waits and checks status before read_id.
    pub(super) fn encode(
        &mut self,
        batch: &mut Batch,
        logits: &Buffer,
        history: &[u32],
        options: SamplingOptions,
        status: &DeviceStatus,
    ) -> Result<()> {
        ensure!(
            options.temperature == 0.0,
            "device greedy requires zero temperature"
        );
        ensure!(
            logits.byte_len() == self.vocabulary * 2,
            "greedy logit extent differs"
        );
        let input = if options.presence_penalty != 0.0 {
            self.seen_host.fill(0);
            let start = if options.presence_context_size == 0 {
                0
            } else {
                history.len().saturating_sub(options.presence_context_size)
            };
            for &token in &history[start..] {
                let id = usize::try_from(token)?;
                ensure!(
                    id < self.vocabulary,
                    "presence history token exceeds vocabulary"
                );
                self.seen_host[id / 32 * 4 + (id % 32) / 8] |= 1u8 << (id % 8);
            }
            self.seen.write_bytes(0, &self.seen_host)?;
            let penalty = bf16::from_f32(options.presence_penalty as f32).to_f32();
            ensure!(penalty.is_finite(), "presence penalty is not finite BF16");
            let penalty_bytes = penalty.to_le_bytes();
            let count_bytes = u32::try_from(self.vocabulary)?.to_le_bytes();
            let bindings =
                [logits, &self.seen, &self.adjusted].map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: The bitset covers every vocabulary ID and each thread
            // writes one BF16 output after the head's ordered producer.
            unsafe {
                batch.dispatch_1d_with_inline(
                    &self.presence,
                    &bindings,
                    &[&penalty_bytes, &count_bytes],
                    self.vocabulary,
                    GROUP,
                )?;
            }
            status.encode_bf16(batch, &self.adjusted)?;
            &self.adjusted
        } else {
            logits
        };
        let (lse, threads) = if self.vocabulary <= 4096 {
            (&self.lse_block, self.vocabulary.div_ceil(128) * 32)
        } else {
            (
                &self.lse_looped,
                self.lse_looped.max_threads_per_threadgroup(),
            )
        };
        ensure!(
            threads > 0
                && threads.is_multiple_of(32)
                && threads <= lse.max_threads_per_threadgroup(),
            "greedy logsumexp threadgroup differs"
        );
        // SAFETY: Input and output extents match the existing BF16 LSE kernel.
        unsafe {
            batch.dispatch_1d(
                lse,
                &[
                    Binding {
                        buffer: input,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.logsumexp,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.count,
                        offset: 0,
                    },
                ],
                threads,
                threads,
            )?;
        }
        status.encode_bf16(batch, &self.logsumexp)?;
        let blocks = self.vocabulary.div_ceil(GROUP);
        let count_bytes = u32::try_from(self.vocabulary)?.to_le_bytes();
        // SAFETY: The first grid covers vocabulary words and writes one
        // candidate per 256-word block; the second reduces those candidates.
        unsafe {
            batch.dispatch_1d_with_inline(
                &self.blocks,
                &[
                    Binding {
                        buffer: input,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.logsumexp,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.partials,
                        offset: 0,
                    },
                ],
                &[&count_bytes],
                blocks * GROUP,
                GROUP,
            )?;
            let blocks_bytes = u32::try_from(blocks)?.to_le_bytes();
            batch.dispatch_1d_with_inline(
                &self.finish,
                &[
                    Binding {
                        buffer: &self.partials,
                        offset: 0,
                    },
                    Binding {
                        buffer: &self.selected,
                        offset: 0,
                    },
                ],
                &[&blocks_bytes],
                GROUP,
                GROUP,
            )?;
        }
        Ok(())
    }

    pub(super) fn read_id(&self) -> Result<u32> {
        let mut bytes = [0; 4];
        self.selected.read_bytes(0, &mut bytes)?;
        let id = u32::from_le_bytes(bytes);
        ensure!(
            usize::try_from(id)? < self.vocabulary,
            "greedy selected ID is invalid"
        );
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::Stream;
    use crate::sampling::{MetalSamplingPreparation, Sampler};

    #[test]
    fn rounded_logprob_ties_and_presence_match_reference_sampler() -> Result<()> {
        fn row(len: usize, peaks: &[(usize, f32)]) -> Vec<u16> {
            let mut values = vec![0.0f32; len];
            for &(id, value) in peaks {
                values[id] = value;
            }
            values
                .into_iter()
                .map(|v| bf16::from_f32(v).to_bits())
                .collect()
        }
        let metal = Metal::new()?;
        let status = DeviceStatus::new(&metal)?;
        let cases = [
            (row(32, &[(1, 0.001)]), vec![], 0.0, 20, 0),
            (row(257, &[(256, 2.0)]), vec![], 0.0, 20, 256),
            (
                row(248320, &[(256, 2.0), (248319, 2.0)]),
                vec![],
                0.0,
                20,
                256,
            ),
            (
                row(4105, &[(31, 1.0), (32, 1.0), (4096, 1.0)]),
                vec![31, 31, 32],
                1.0,
                0,
                4096,
            ),
            (
                row(4105, &[(31, 1.0), (32, 1.0), (4096, 1.0)]),
                vec![31, 31, 32],
                1.0,
                1,
                31,
            ),
        ];
        for (row, history, penalty, context, expected) in cases {
            let mut greedy = DeviceGreedy::new(&metal, row.len())?;
            let mut preparation = MetalSamplingPreparation::new(&metal, row.len())?;
            let input = metal.shared_buffer(row.len() * 2)?;
            let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
            input.write_bytes(0, &bytes)?;
            let options = SamplingOptions {
                temperature: 0.0,
                presence_penalty: penalty,
                presence_context_size: context,
                ..SamplingOptions::default()
            };
            let mut sampler = Sampler::new(options, 0)?;
            let reference = sampler.sample_id(&mut preparation, &row, &history)?;
            assert_eq!(reference, expected);
            status.reset()?;
            let mut stream: Stream = metal.stream()?;
            greedy.encode(stream.batch()?, &input, &history, options, &status)?;
            stream.wait()?;
            status.check()?;
            assert_eq!(greedy.read_id()?, reference);
        }
        Ok(())
    }
}

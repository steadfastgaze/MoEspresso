//! Device-only stable grouping for the full-resident packed expert kernels.

use super::*;

const BLOCK_PAIRS: usize = 256;
const TILE_PAIRS: usize = 32;
const SOURCE: &str = include_str!("../../../kernels/metal/mlp_full_schedule.metal");

pub(super) struct FullSchedule {
    histogram: Pipeline,
    offsets: Pipeline,
    scatter: Pipeline,
}

pub(super) struct ScheduleBuffers<'a> {
    pub ids: &'a Buffer,
    pub order: &'a Buffer,
    pub schedule: &'a Buffer,
    pub total: &'a Buffer,
    /// Activation has no live value until the following gate/up producer.
    pub activation: &'a Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry {
    blocks: usize,
    block_bytes: usize,
    metadata_bytes: usize,
    max_tiles: usize,
}

impl Geometry {
    fn new(pairs: usize) -> Result<Self> {
        ensure!(
            pairs > 0 && pairs <= (u32::MAX - 255) as usize,
            "full schedule pair count is invalid"
        );
        let blocks = pairs.div_ceil(BLOCK_PAIRS);
        let block_words = blocks
            .checked_mul(EXPERTS)
            .context("full schedule metadata size overflow")?;
        ensure!(
            block_words <= u32::MAX as usize,
            "full schedule metadata exceeds Metal indices"
        );
        let block_bytes = block_words
            .checked_mul(4)
            .context("full schedule metadata size overflow")?;
        let metadata_bytes = block_bytes
            .checked_mul(2)
            .context("full schedule metadata size overflow")?;
        Ok(Self {
            blocks,
            block_bytes,
            metadata_bytes,
            max_tiles: EXPERTS + pairs.div_ceil(TILE_PAIRS),
        })
    }
}

impl FullSchedule {
    pub(super) fn new(metal: &Metal) -> Result<Self> {
        let compile = |name, threads| -> Result<Pipeline> {
            let pipeline = metal.compile_msl_strict(SOURCE, name)?;
            ensure!(
                pipeline.max_threads_per_threadgroup() >= threads,
                "full schedule threadgroup unsupported"
            );
            Ok(pipeline)
        };
        Ok(Self {
            histogram: compile("mlp_full_histogram", BLOCK_PAIRS)?,
            offsets: compile("mlp_full_offsets", EXPERTS)?,
            scatter: compile("mlp_full_stable_order", BLOCK_PAIRS)?,
        })
    }

    pub(super) fn encode(
        &self,
        batch: &mut Batch,
        buffers: ScheduleBuffers<'_>,
        pairs: usize,
    ) -> Result<()> {
        let geometry = Geometry::new(pairs)?;
        ensure!(
            buffers.ids.byte_len() >= pairs * 4
                && buffers.order.byte_len() >= pairs * 4
                && buffers.schedule.byte_len() >= geometry.max_tiles * 12
                && buffers.total.byte_len() >= 4
                && buffers.activation.byte_len() >= geometry.metadata_bytes,
            "full schedule buffer extent differs"
        );
        let binding = |buffer| Binding { buffer, offset: 0 };
        let count = (pairs as u32).to_le_bytes();
        let block_starts = Binding {
            buffer: buffers.activation,
            offset: geometry.block_bytes,
        };
        // SAFETY: Every active histogram cell and block start is initialized.
        // The two metadata ranges are disjoint and fit the dead activation
        // allocation. Commands are ordered before gate/up overwrites that
        // storage. Order writes are stable original-pair indices, and all
        // source-indexed accesses guard against invalid route IDs.
        unsafe {
            batch.dispatch_1d_with_inline(
                &self.histogram,
                &[binding(buffers.ids), binding(buffers.activation)],
                &[&count],
                geometry.blocks * BLOCK_PAIRS,
                BLOCK_PAIRS,
            )?;
            batch.dispatch_1d_with_inline(
                &self.offsets,
                &[
                    binding(buffers.activation),
                    block_starts,
                    binding(buffers.schedule),
                    binding(buffers.total),
                ],
                &[&count],
                EXPERTS,
                EXPERTS,
            )?;
            batch.dispatch_1d_with_inline(
                &self.scatter,
                &[binding(buffers.ids), block_starts, binding(buffers.order)],
                &[&count],
                geometry.blocks * BLOCK_PAIRS,
                BLOCK_PAIRS,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

//! Shared QSA query preparation and FP32 scratch for late gate publication.

use super::ane_prefill::MAX_PHYSICAL_ROWS;
use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
use anyhow::{Result, ensure};

pub(super) struct QsaGate {
    rows: usize,
    pub attended: Buffer,
    pub query: Pipeline,
    pub prepare: Pipeline,
    pub deferred: Pipeline,
    late: Pipeline,
}

impl QsaGate {
    pub fn new(metal: &Metal, rows: usize) -> Result<Self> {
        let elements = gate_elements(rows)?;
        let dense = format!(
            "{}\n{}",
            crate::kquant::METAL_SOURCE,
            crate::dense_batch::METAL_SOURCE
        );
        let owner = Self {
            rows,
            attended: metal.shared_buffer(elements * 4)?,
            query: metal.compile_msl_strict(&dense, "qsa_query_half")?,
            prepare: metal.compile_msl_strict(
                include_str!("../../kernels/metal/qsa_projection.metal"),
                "qsa_project_rope_compact_rows",
            )?,
            deferred: metal.compile_msl_strict(
                include_str!("../../kernels/metal/qsa_attention.metal"),
                "qsa_prefill_indexed_deferred",
            )?,
            late: metal.compile_msl_strict(
                include_str!("../../kernels/metal/qsa_attention.metal"),
                "qsa_late_gate",
            )?,
        };
        ensure!(
            [&owner.query, &owner.deferred, &owner.late]
                .into_iter()
                .all(|p| p.max_threads_per_threadgroup() >= 128)
                && owner.prepare.max_threads_per_threadgroup() >= 32,
            "QSA gate threadgroup unsupported"
        );
        eprintln!(
            "ANE QSA gate scratch: rows={rows} attended_fp32_bytes={}",
            owner.attended.byte_len()
        );
        Ok(owner)
    }

    pub fn eligible(&self, rows: usize, ane_rows: usize) -> Result<bool> {
        ensure!(
            ane_rows == self.rows,
            "combined gate controller geometries differ"
        );
        eligible(self.rows, rows)
    }

    /// Encode after ordered gate unpacking and all FP32 attention writers.
    pub fn encode_late(
        &self,
        batch: &mut Batch,
        gate: &Buffer,
        gated: &Buffer,
        status: &Buffer,
        useful_rows: usize,
    ) -> Result<()> {
        ensure!(
            eligible(self.rows, useful_rows)?,
            "QSA late gate useful rows differ"
        );
        let elements = late_rows(useful_rows) * 6144;
        ensure!(
            gate.byte_len() >= useful_rows * 6144 * 2
                && gated.byte_len() >= useful_rows * 6144 * 2
                && status.byte_len() >= 4,
            "QSA late gate extent differs"
        );
        // SAFETY: The inference stream retains disjoint scratch and output;
        // deferred attention and gate unpack precede this single BF16 boundary.
        unsafe {
            batch.dispatch_1d_with_inline(
                &self.late,
                &[&self.attended, gate, gated, status].map(|buffer| Binding { buffer, offset: 0 }),
                &[&(elements as u32).to_le_bytes()],
                elements,
                128,
            )
        }
    }
}

fn gate_elements(rows: usize) -> Result<usize> {
    ensure!(
        matches!(rows, 4096 | MAX_PHYSICAL_ROWS),
        "QSA gate capacity differs"
    );
    Ok(rows * 6144)
}

fn eligible(ane_rows: usize, rows: usize) -> Result<bool> {
    ensure!(
        matches!(ane_rows, 4096 | MAX_PHYSICAL_ROWS),
        "combined gate controller geometries differ"
    );
    Ok(rows == ane_rows || (ane_rows == MAX_PHYSICAL_ROWS && (4096..=ane_rows).contains(&rows)))
}

// The final one-through-eight rows retain the existing online consumer after
// gate publication. Only reconstructed tiles write the FP32 attended scratch.
fn late_rows(rows: usize) -> usize {
    let tail = rows % 64;
    rows - if tail <= 8 { tail } else { 0 }
}

#[cfg(test)]
mod tests;

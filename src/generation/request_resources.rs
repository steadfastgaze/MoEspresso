//! Request-scoped scratch ownership within complete serving commitments.

use super::RequestPhase;
use crate::adaptive_memory::{Boundary, Controller};
use crate::engine::Engine;
use crate::memory::{ExpertCapacityControlBudget, PackageFootprint};
use anyhow::{Result, ensure};

const ORDINARY_ROWS: usize = 2048;
const WIDE_ROWS: usize = 4096;

pub(crate) struct PhasedPrefill {
    ordinary: ExpertCapacityControlBudget,
    wide: ExpertCapacityControlBudget,
    context: usize,
    active: bool,
}

impl PhasedPrefill {
    pub(crate) fn new(
        budget: ExpertCapacityControlBudget,
        footprint: &PackageFootprint,
        context: usize,
        retained_bytes: u64,
        disk_io_bytes: u64,
    ) -> Result<Self> {
        budget.validate_serving(
            footprint,
            context,
            ORDINARY_ROWS,
            retained_bytes,
            disk_io_bytes,
        )?;
        Ok(Self {
            ordinary: budget,
            wide: budget.with_prefill_width(footprint, context, ORDINARY_ROWS, WIDE_ROWS)?,
            context,
            active: false,
        })
    }

    #[cfg(test)]
    pub(super) fn wide_slot_ceiling(&self) -> usize {
        self.wide.maximum_capacity()
    }

    #[cfg(test)]
    pub(super) fn is_active(&self) -> bool {
        self.active
    }

    pub(crate) fn transition(
        &mut self,
        engine: &mut Engine,
        phase: RequestPhase,
        controller: &mut Controller,
    ) -> Result<()> {
        match phase {
            RequestPhase::Prepare | RequestPhase::Decode | RequestPhase::Finished => {
                self.finish(engine, controller)?;
            }
            RequestPhase::Prefill { new_tokens } if new_tokens > ORDINARY_ROWS => {
                ensure!(
                    !self.active
                        && engine.prefill_chunk_size() == ORDINARY_ROWS
                        && engine.prefill_workspace_ready(),
                    "previous workspace phase was not finalized"
                );
                let previous = engine.expert_capacity();
                let target = previous.min(self.wide.maximum_capacity());
                if !controller.try_begin_workspace_phase(engine, target)? {
                    return Ok(());
                }
                self.active = true;
                let entered: Result<()> = (|| {
                    if target < previous {
                        let release = target / 16 * 16;
                        engine.shrink_expert_capacity(release)?;
                        controller.sync_planned_workspace_capacity(engine)?;
                        if release < target {
                            engine.recover_expert_capacity(
                                target,
                                self.wide.preparation_budget_bytes(release, target)?,
                            )?;
                            controller.sync_planned_workspace_capacity(engine)?;
                        }
                    }
                    let bytes = self.wide.workspace_budget_after_release(
                        self.context,
                        WIDE_ROWS,
                        engine.expert_capacity(),
                    )?;
                    engine.replace_prefill_without_overlap(WIDE_ROWS, bytes)?;
                    Ok(())
                })();
                if let Err(error) = entered {
                    eprintln!(
                        "prefill phase: wider scratch unavailable; retaining ordinary execution: {error:#}"
                    );
                    let synchronized = controller.sync_planned_workspace_capacity(engine);
                    let cleanup = self.finish(engine, controller);
                    return match (synchronized, cleanup) {
                        (Ok(()), Ok(())) => Ok(()),
                        (Err(sync), Ok(())) => Err(error.context(format!("phase synchronization also failed: {sync:#}"))),
                        (Ok(()), Err(cleanup)) => Err(error.context(format!("ordinary scratch repair also failed: {cleanup:#}"))),
                        (Err(sync), Err(cleanup)) => Err(error.context(format!("phase synchronization and scratch repair also failed: {sync:#}; {cleanup:#}"))),
                    };
                }
            }
            RequestPhase::Prefill { .. } | RequestPhase::CompletedOperation => {}
        }
        Ok(())
    }

    fn finish(&mut self, engine: &mut Engine, controller: &mut Controller) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let current = engine.expert_capacity();
        let bytes =
            self.ordinary
                .workspace_budget_after_release(self.context, ORDINARY_ROWS, current)?;
        engine.replace_prefill_without_overlap(ORDINARY_ROWS, bytes)?;
        // Scratch replacement can outlast the current sample. Observe pressure
        // while allocation is still frozen before returning ordinary experts.
        controller.check_boundary(engine, Boundary::CompletedOperation)?;
        controller.finish_workspace_phase(engine)?;
        self.active = false;
        Ok(())
    }
}

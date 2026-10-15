//! Owner-polled expert residency under Darwin memory pressure.
//!
//! The pressure sysctl is an optional kernel observation. Missing observations
//! prohibit recovery. The compression threshold is conservative admission
//! policy, not a calibrated estimate of available memory.

use crate::engine::Engine;
use crate::memory::ExpertCapacityControlBudget;
use anyhow::{Context, Result, bail, ensure};
use std::mem::{offset_of, size_of, zeroed};
use std::ptr;
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const SAMPLE_GAP_LIMIT: Duration = Duration::from_secs(5);
const RELEASE_INTERVAL: Duration = Duration::from_secs(5);
// Provisional settling time between capacity changes, not a memory estimate.
const RECOVERY_COOLDOWN: Duration = Duration::from_secs(5);
const ACTIVE_VALIDATION: Duration = Duration::from_secs(15);
const COMPRESSION_VETO_BYTES_PER_SECOND: f64 = (16 * (1 << 20)) as f64;
const MIN_CAPACITY: usize = 64;
const GROWTH_STEP_SLOTS: usize = 16;

/// Owner boundaries distinguish idle observations from completed serving work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Boundary {
    Observe,
    RequestStart,
    CompletedOperation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pressure {
    Normal,
    Warning,
    Critical,
}

impl Pressure {
    fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            1 => Some(Self::Normal),
            2 => Some(Self::Warning),
            4 => Some(Self::Critical),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct VmCounters {
    compressions: u64,
    decompressions: u64,
    swapins: u64,
    swapouts: u64,
    page_bytes: u64,
}

impl VmCounters {
    fn quiet_since(self, before: Self, elapsed: Duration) -> Option<bool> {
        if elapsed.is_zero() || self.page_bytes == 0 || self.page_bytes != before.page_bytes {
            return None;
        }
        let rate = |after: u64, before: u64| {
            Some(
                after.checked_sub(before)?.checked_mul(self.page_bytes)? as f64
                    / elapsed.as_secs_f64(),
            )
        };
        let compression = rate(self.compressions, before.compressions)?;
        // A reset or overflow in any counter invalidates the whole sample.
        rate(self.decompressions, before.decompressions)?;
        rate(self.swapins, before.swapins)?;
        let swap_out = rate(self.swapouts, before.swapouts)?;
        Some(swap_out == 0.0 && compression < COMPRESSION_VETO_BYTES_PER_SECOND)
    }

    fn churning_since(self, before: Self, elapsed: Duration) -> Option<bool> {
        if self.quiet_since(before, elapsed)? {
            return Some(false);
        }
        let decompression = self
            .decompressions
            .checked_sub(before.decompressions)?
            .checked_mul(self.page_bytes)? as f64
            / elapsed.as_secs_f64();
        Some(self.swapouts > before.swapouts || decompression >= COMPRESSION_VETO_BYTES_PER_SECOND)
    }
}

#[derive(Clone, Copy, Debug)]
struct Observation {
    pressure: Option<Pressure>,
    counters: Option<VmCounters>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Unknown,
    NormalBusy,
    NormalQuiet,
    Warning,
    Critical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    Hold,
    Release(usize),
    Rollback(usize),
    ChurnRelease(usize),
    Recover(usize),
    Stop,
}

enum GrowthEligibility {
    Hold,
    Restore,
    Trial,
}

#[derive(Clone, Copy)]
struct WorkspacePhase {
    ordinary_capacity: usize,
    pressure_release_epoch: u64,
}

struct Policy {
    preparation_healthy: Duration,
    active_churn: Duration,
    last_churn_sample_active: bool,
    validated_in_request: Option<usize>,
    initial_capacity: usize,
    maximum_capacity: usize,
    capacity: usize,
    earned_target: Option<usize>,
    trial_base: Option<usize>,
    trial_capacity: Option<usize>,
    last_sample: Option<Duration>,
    previous_counters: Option<VmCounters>,
    quiet_since: Option<Duration>,
    request_started: Option<Duration>,
    last_sample_active: bool,
    last_sample_trial_active: bool,
    trial_adverse_active: Duration,
    request_clean_active: Duration,
    completed_clean_active: Duration,
    last_release: Option<Duration>,
    last_capacity_change: Option<Duration>,
    last_increase: Option<Duration>,
    retry_delay: Duration,
    retry_not_before: Option<Duration>,
    critical_at_floor: bool,
    stop_latched: bool,
    state: Option<State>,
    workspace_phase: Option<WorkspacePhase>,
}

impl Policy {
    fn new(initial_capacity: usize, maximum_capacity: usize) -> Result<Self> {
        ensure!(
            (MIN_CAPACITY..512).contains(&initial_capacity)
                && (initial_capacity..512).contains(&maximum_capacity),
            "adaptive memory requires a bounded expert ceiling of at least 64 slots"
        );
        Ok(Self {
            preparation_healthy: Duration::ZERO,
            active_churn: Duration::ZERO,
            last_churn_sample_active: false,
            validated_in_request: None,
            initial_capacity,
            maximum_capacity,
            capacity: initial_capacity,
            earned_target: None,
            trial_base: None,
            trial_capacity: None,
            last_sample: None,
            previous_counters: None,
            quiet_since: None,
            request_started: None,
            last_sample_active: false,
            last_sample_trial_active: false,
            trial_adverse_active: Duration::ZERO,
            request_clean_active: Duration::ZERO,
            completed_clean_active: Duration::ZERO,
            last_release: None,
            last_capacity_change: None,
            last_increase: None,
            retry_delay: RECOVERY_COOLDOWN,
            retry_not_before: None,
            critical_at_floor: false,
            stop_latched: false,
            state: None,
            workspace_phase: None,
        })
    }

    fn validation_extent_matches(&self, capacity: usize) -> bool {
        self.workspace_phase
            .is_none_or(|phase| phase.ordinary_capacity == capacity)
    }

    fn try_begin_workspace_phase(
        &mut self,
        capacity: usize,
        target_capacity: usize,
        pressure_release_epoch: u64,
    ) -> Result<bool> {
        ensure!(
            self.workspace_phase.is_none(),
            "workspace phase is already active"
        );
        ensure!(
            (MIN_CAPACITY..=self.maximum_capacity).contains(&capacity)
                && (MIN_CAPACITY..=capacity).contains(&target_capacity),
            "workspace phase capacity is outside its ordinary operating extent"
        );
        if self.stop_latched || matches!(self.state, Some(State::Warning | State::Critical)) {
            return Ok(false);
        }
        if self.capacity != capacity {
            self.clear_capacity_validation();
            self.capacity = capacity;
        }
        self.workspace_phase = Some(WorkspacePhase {
            ordinary_capacity: capacity,
            pressure_release_epoch,
        });
        Ok(true)
    }

    fn sync_planned_workspace_capacity(&mut self, now: Duration, capacity: usize) -> Result<()> {
        let phase = self
            .workspace_phase
            .context("workspace phase is not active")?;
        ensure!(
            (MIN_CAPACITY..=phase.ordinary_capacity).contains(&capacity),
            "planned workspace change differs from the active expert extent"
        );
        if self.capacity != capacity {
            if self.trial_capacity.is_some() {
                self.trial_base = self.trial_base.map(|base| base.min(capacity));
                self.trial_capacity = Some(capacity);
                self.clear_workspace_validation();
            }
            self.capacity = capacity;
            self.quiet_since = None;
            self.last_capacity_change = Some(now);
            self.last_sample_active = false;
            self.last_sample_trial_active = false;
        }
        Ok(())
    }

    fn workspace_restoration_allowed(&self, now: Duration, pressure_release_epoch: u64) -> bool {
        self.workspace_phase.is_some_and(|phase| {
            self.capacity < phase.ordinary_capacity
                && phase.pressure_release_epoch == pressure_release_epoch
                && self.trial_capacity.is_none()
                && !self.stop_latched
                && self.state == Some(State::NormalQuiet)
                && self.last_sample.is_some_and(|last| {
                    now.checked_sub(last)
                        .is_some_and(|dt| dt <= SAMPLE_INTERVAL)
                })
                && self.quiet_since.is_some_and(|since| {
                    now.checked_sub(since)
                        .is_some_and(|dt| dt >= RECOVERY_COOLDOWN)
                })
                && self.last_capacity_change.is_none_or(|changed| {
                    now.checked_sub(changed)
                        .is_some_and(|dt| dt >= RECOVERY_COOLDOWN)
                })
                && self.retry_not_before.is_none_or(|until| now >= until)
        })
    }

    fn finish_workspace_phase(&mut self) {
        if let Some(phase) = self.workspace_phase.take()
            && self.capacity != phase.ordinary_capacity
        {
            self.clear_workspace_validation();
        }
    }

    fn clear_workspace_validation(&mut self) {
        let adverse = self.trial_adverse_active;
        self.clear_capacity_validation();
        if self.trial_capacity.is_some() {
            self.trial_adverse_active = adverse;
        }
    }

    fn clear_capacity_validation(&mut self) {
        self.preparation_healthy = Duration::ZERO;
        self.active_churn = Duration::ZERO;
        self.last_churn_sample_active = false;
        self.validated_in_request = None;
        self.request_clean_active = Duration::ZERO;
        self.completed_clean_active = Duration::ZERO;
        self.last_sample_active = false;
        self.last_sample_trial_active = false;
        self.trial_adverse_active = Duration::ZERO;
    }

    fn begin_request(&mut self, now: Duration) {
        self.last_churn_sample_active = false;
        self.validated_in_request = None;
        self.request_started = Some(now);
        self.request_clean_active = Duration::ZERO;
        self.last_sample_active = false;
        self.last_sample_trial_active = false;
    }

    fn end_request(&mut self, now: Duration, completed: bool) {
        let validated = self.validated_in_request.take();
        self.last_churn_sample_active = false;
        let was_active = self.request_started.take().is_some();
        self.last_sample_active = false;
        self.last_sample_trial_active = false;
        let fresh = self.last_sample.is_some_and(|last| {
            now.checked_sub(last)
                .is_some_and(|dt| dt <= SAMPLE_GAP_LIMIT)
        });
        if was_active
            && completed
            && fresh
            && self.validation_extent_matches(self.capacity)
            && matches!(self.state, Some(State::NormalQuiet | State::NormalBusy))
        {
            if let Some(target) = validated.filter(|target| *target < self.capacity) {
                self.earned_target = Some(self.earned_target.unwrap_or(0).max(target));
                if self.trial_capacity == Some(self.capacity) {
                    self.trial_base = Some(self.trial_base.unwrap_or(target).max(target));
                }
            }
            self.completed_clean_active = self
                .completed_clean_active
                .saturating_add(self.request_clean_active)
                .min(ACTIVE_VALIDATION);
            if self.completed_clean_active >= ACTIVE_VALIDATION {
                let accepted_trial = self.trial_capacity == Some(self.capacity);
                self.earned_target = Some(self.earned_target.unwrap_or(0).max(self.capacity));
                if accepted_trial {
                    self.trial_base = None;
                    self.trial_capacity = None;
                    self.trial_adverse_active = Duration::ZERO;
                }
                if accepted_trial {
                    self.retry_delay = RECOVERY_COOLDOWN;
                    self.retry_not_before = None;
                }
            }
        }
        // An unsuccessful request contributes no credit. Prior completed work
        // and the earned target remain valid.
        self.request_clean_active = Duration::ZERO;
    }

    fn sample_due(&self, now: Duration) -> bool {
        self.last_sample.is_none_or(|last| {
            now.checked_sub(last)
                .is_some_and(|dt| dt >= SAMPLE_INTERVAL)
        })
    }

    fn is_restoring(&self, capacity: usize) -> bool {
        self.earned_target.is_some_and(|target| capacity < target)
    }

    fn target(&self, capacity: usize) -> usize {
        if self.is_restoring(capacity) {
            self.earned_target.unwrap()
        } else {
            self.maximum_capacity
        }
    }

    fn next_capacity(&self, capacity: usize) -> usize {
        let target = self
            .target(capacity)
            .min(self.maximum_capacity)
            .max(capacity);
        // An earned tail can span two physical ranges within one bounded
        // increase. Preparing them together avoids another settling interval.
        if self.is_restoring(capacity) && target - capacity <= GROWTH_STEP_SLOTS {
            return target;
        }
        ((capacity / GROWTH_STEP_SLOTS + 1) * GROWTH_STEP_SLOTS)
            .min(target)
            .max(capacity)
    }

    fn mode(&self, capacity: usize) -> &'static str {
        if self.is_restoring(capacity) {
            "restore"
        } else if capacity < self.initial_capacity {
            "startup-validation"
        } else {
            "explore"
        }
    }

    fn settled(&self, now: Duration) -> bool {
        let settle = RECOVERY_COOLDOWN;
        self.quiet_since
            .is_some_and(|since| now.checked_sub(since).is_some_and(|dt| dt >= settle))
            && self
                .last_capacity_change
                .is_none_or(|changed| now.checked_sub(changed).is_some_and(|dt| dt >= settle))
            && self.retry_not_before.is_none_or(|until| now >= until)
    }

    fn growth_eligibility(&self, now: Duration, capacity: usize) -> GrowthEligibility {
        if self.workspace_phase.is_some()
            || capacity >= self.maximum_capacity
            || self.stop_latched
            || self.state != Some(State::NormalQuiet)
            || !self.settled(now)
        {
            GrowthEligibility::Hold
        } else if self.is_restoring(capacity) {
            GrowthEligibility::Restore
        } else if self.trial_capacity.is_some()
            && (if self.request_started.is_some() {
                self.completed_clean_active + self.request_clean_active < ACTIVE_VALIDATION
            } else {
                !self.completed_clean_active.is_zero()
            })
        {
            // Preserve partial serving credit until it can qualify this extent.
            GrowthEligibility::Hold
        } else if self.trial_capacity.is_some() && self.preparation_healthy < ACTIVE_VALIDATION {
            GrowthEligibility::Hold
        } else {
            GrowthEligibility::Trial
        }
    }

    fn recovery_action(&self, now: Duration, capacity: usize) -> Action {
        match self.growth_eligibility(now, capacity) {
            GrowthEligibility::Restore | GrowthEligibility::Trial => {
                Action::Recover(self.next_capacity(capacity))
            }
            GrowthEligibility::Hold => Action::Hold,
        }
    }

    fn cached_action(&self, now: Duration, capacity: usize) -> Action {
        let fresh = self.last_sample.is_some_and(|last| {
            now.checked_sub(last)
                .is_some_and(|dt| dt <= SAMPLE_INTERVAL)
        });
        if !fresh {
            return Action::Hold;
        }
        if self.stop_latched && capacity == MIN_CAPACITY {
            return Action::Stop;
        }
        self.recovery_action(now, capacity)
    }

    fn pressure_interrupted_work(&mut self, now: Duration) {
        self.interrupted_work(now, false);
    }

    fn interrupted_work(&mut self, now: Duration, always_backoff: bool) {
        let pending_trial = self.trial_capacity.take().is_some();
        self.trial_base = None;
        let recent_increase = self.last_increase.take().is_some_and(|increased| {
            now.checked_sub(increased)
                .is_some_and(|dt| dt <= ACTIVE_VALIDATION)
        });
        if pending_trial || recent_increase || always_backoff {
            self.retry_delay = (self.retry_delay * 2).min(Duration::from_secs(120));
            self.retry_not_before = Some(now.saturating_add(self.retry_delay));
        }
        self.clear_capacity_validation();
    }

    fn observe(
        &mut self,
        now: Duration,
        capacity: usize,
        boundary: Boundary,
        observation: Observation,
    ) -> Option<Action> {
        if !self.sample_due(now) {
            return None;
        }
        self.capacity = capacity;
        let elapsed = self.last_sample.and_then(|last| now.checked_sub(last));
        let gap = elapsed.is_none_or(|dt| dt > SAMPLE_GAP_LIMIT);
        let active_churn = self.observe_churn(boundary, observation, elapsed);
        let quiet = observation
            .counters
            .zip(self.previous_counters)
            .zip(elapsed)
            .and_then(|((after, before), dt)| after.quiet_since(before, dt));
        let state = match observation.pressure {
            None => State::Unknown,
            Some(Pressure::Warning) => State::Warning,
            Some(Pressure::Critical) => State::Critical,
            Some(Pressure::Normal) if gap => State::Unknown,
            Some(Pressure::Normal) => match quiet {
                None => State::Unknown,
                Some(false) => State::NormalBusy,
                Some(true) => State::NormalQuiet,
            },
        };
        let request_active =
            self.request_started.is_some() && self.validation_extent_matches(capacity);
        let trial_active = self.workspace_phase.is_none()
            || (self.request_started.is_some() && boundary == Boundary::CompletedOperation);
        // Adverse intervals belong only to the unaccepted trial. Request failure
        // discards healthy credit but cannot erase observed activity at that extent.
        if self.trial_capacity == Some(capacity)
            && state == State::NormalBusy
            && !gap
            && trial_active
            && self.last_sample_trial_active
            && self.trial_interval_started()
        {
            self.trial_adverse_active = self
                .trial_adverse_active
                .saturating_add(elapsed.unwrap())
                .min(ACTIVE_VALIDATION);
        }
        self.last_sample_trial_active = trial_active;
        if state != State::NormalQuiet {
            self.quiet_since = None;
        } else {
            self.quiet_since.get_or_insert(now);
            if request_active
                && !gap
                && self.state == Some(State::NormalQuiet)
                && self.last_sample_active
                && self
                    .request_started
                    .is_some_and(|started| self.last_sample.is_some_and(|last| started <= last))
            {
                self.request_clean_active = self
                    .request_clean_active
                    .saturating_add(elapsed.unwrap())
                    .min(ACTIVE_VALIDATION);
            }
        }
        self.last_sample_active = state == State::NormalQuiet && request_active;
        self.observe_preparation_span(state, elapsed, gap);
        self.last_sample = Some(now);
        self.previous_counters = observation.counters;
        if matches!(state, State::Warning | State::Critical)
            && !matches!(self.state, Some(State::Warning | State::Critical))
        {
            self.pressure_interrupted_work(now);
        }
        if state != State::Critical {
            self.critical_at_floor = false;
            self.stop_latched = false;
        }
        self.state = Some(state);
        let trial_rollback = self
            .trial_base
            .filter(|base| self.trial_capacity == Some(capacity) && *base < capacity);
        let action = match state {
            State::Critical if capacity > MIN_CAPACITY => {
                self.critical_at_floor = true;
                Action::Release(MIN_CAPACITY)
            }
            State::Critical => {
                let previously_critical = self.critical_at_floor;
                self.critical_at_floor = true;
                if previously_critical {
                    self.stop_latched = true;
                    Action::Stop
                } else {
                    Action::Hold
                }
            }
            State::Warning if capacity > MIN_CAPACITY => {
                let release_due = self.last_release.is_none_or(|last| {
                    now.checked_sub(last)
                        .is_some_and(|dt| dt >= RELEASE_INTERVAL)
                });
                if release_due {
                    Action::Release(
                        ((capacity - 1) / GROWTH_STEP_SLOTS * GROWTH_STEP_SLOTS).max(MIN_CAPACITY),
                    )
                } else {
                    Action::Hold
                }
            }
            State::NormalQuiet => self.recovery_action(now, capacity),
            State::NormalBusy
                if self.trial_adverse_active >= ACTIVE_VALIDATION && trial_rollback.is_some() =>
            {
                Action::Rollback(
                    (trial_rollback.unwrap() / GROWTH_STEP_SLOTS * GROWTH_STEP_SLOTS)
                        .max(MIN_CAPACITY),
                )
            }
            _ if active_churn && capacity > MIN_CAPACITY => Action::ChurnRelease(
                self.trial_base
                    .filter(|base| *base < capacity)
                    .map_or(
                        (capacity - 1) / GROWTH_STEP_SLOTS * GROWTH_STEP_SLOTS,
                        |base| base / GROWTH_STEP_SLOTS * GROWTH_STEP_SLOTS,
                    )
                    .max(MIN_CAPACITY),
            ),
            _ => Action::Hold,
        };
        Some(action)
    }

    fn capacity_changed(&mut self, now: Duration, old_capacity: usize, new_capacity: usize) {
        let validated = self.validation_before_growth(new_capacity);
        let pending_adversity = (new_capacity > old_capacity
            && self.trial_capacity == Some(old_capacity))
        .then_some(self.trial_adverse_active);
        self.capacity = new_capacity;
        self.quiet_since = None;
        self.clear_capacity_validation();
        self.validated_in_request = validated;
        if let Some(adverse) = pending_adversity {
            self.trial_adverse_active = adverse;
        }
        self.last_capacity_change = Some(now);
        if new_capacity > old_capacity {
            self.last_increase = Some(now);
            if !self
                .earned_target
                .is_some_and(|target| new_capacity <= target)
            {
                let base = self.trial_base.unwrap_or(old_capacity);
                self.trial_base = Some(base);
                self.trial_capacity = Some(new_capacity);
            }
        } else if new_capacity < old_capacity {
            self.last_release = Some(now);
            self.trial_base = None;
            self.trial_capacity = None;
        }
    }

    fn trial_interval_started(&self) -> bool {
        if self.workspace_phase.is_none() {
            return true;
        }
        self.request_started
            .is_some_and(|started| self.last_sample.is_some_and(|last| started <= last))
    }

    fn recovery_failed(&mut self, now: Duration) {
        self.quiet_since = None;
        self.retry_delay = (self.retry_delay * 2).min(Duration::from_secs(120));
        self.retry_not_before = Some(now.saturating_add(self.retry_delay));
    }

    fn observe_churn(
        &mut self,
        boundary: Boundary,
        observation: Observation,
        elapsed: Option<Duration>,
    ) -> bool {
        let active = boundary == Boundary::CompletedOperation && self.request_started.is_some();
        let interval_active = active
            && self.last_churn_sample_active
            && self
                .request_started
                .is_some_and(|started| self.last_sample.is_some_and(|last| last >= started));
        let churn = observation
            .counters
            .zip(self.previous_counters)
            .zip(elapsed)
            .and_then(|((after, before), dt)| after.churning_since(before, dt));
        if interval_active && observation.pressure == Some(Pressure::Normal) {
            // Long completed-operation intervals may themselves be stalls.
            // Valid rate observations contribute at most one sample-gap limit;
            // they never earn healthy active-work validation.
            let credit = elapsed.unwrap_or_default().min(SAMPLE_GAP_LIMIT);
            self.active_churn = match churn {
                Some(true) => self
                    .active_churn
                    .saturating_add(credit)
                    .min(ACTIVE_VALIDATION),
                Some(false) => self.active_churn.saturating_sub(credit),
                None => Duration::ZERO,
            };
        } else if !active || observation.pressure != Some(Pressure::Normal) || churn.is_none() {
            self.active_churn = Duration::ZERO;
        }
        self.last_churn_sample_active = active;
        self.active_churn >= ACTIVE_VALIDATION
    }

    fn validation_before_growth(&self, new_capacity: usize) -> Option<usize> {
        if new_capacity <= self.capacity {
            return None;
        }
        let qualified = (self.request_started.is_some()
            && self.validation_extent_matches(self.capacity)
            && self.completed_clean_active + self.request_clean_active >= ACTIVE_VALIDATION)
            .then_some(self.capacity);
        self.validated_in_request.max(qualified)
    }

    fn observe_preparation_span(&mut self, state: State, elapsed: Option<Duration>, gap: bool) {
        if self.workspace_phase.is_some()
            || state != State::NormalQuiet
            || self.state != Some(State::NormalQuiet)
            || gap
            || !self.last_sample.is_some_and(|last| {
                self.last_capacity_change
                    .is_none_or(|changed| last >= changed)
            })
        {
            return;
        }
        self.preparation_healthy = self
            .preparation_healthy
            .saturating_add(elapsed.unwrap())
            .min(ACTIVE_VALIDATION);
        // Quiet storage permits another bounded preparation, not active-work
        // acceptance. The entire unvalidated extent retains its rollback base.
    }
}

/// Automatic capacity changes serialized with the inference owner.
pub struct Controller {
    budget: ExpertCapacityControlBudget,
    policy: Policy,
    started: Instant,
    #[cfg(test)]
    fixed_observation: Option<(Duration, Observation)>,
    pressure_release_epoch: u64,
}

impl Controller {
    /// Freeze adaptive allocation during a caller-budgeted workspace phase.
    /// Provisional storage can yield space without losing its rollback base.
    pub(crate) fn try_begin_workspace_phase(
        &mut self,
        engine: &Engine,
        target_capacity: usize,
    ) -> Result<bool> {
        self.policy.try_begin_workspace_phase(
            engine.expert_capacity(),
            target_capacity,
            self.pressure_release_epoch,
        )
    }

    /// Record a completed planned resize without opening a trial or counting
    /// the transaction as pressure. Validation remains bound to the original
    /// extent until ordinary workspace restoration completes.
    pub(crate) fn sync_planned_workspace_capacity(&mut self, engine: &Engine) -> Result<()> {
        self.policy
            .sync_planned_workspace_capacity(self.now(), engine.expert_capacity())
    }

    /// Finish after the caller has rebuilt retired ordinary scratch. Pressure
    /// or failed allocation can retain a smaller valid expert extent. Missing
    /// scratch leaves the phase active so allocation stays frozen until repair.
    pub(crate) fn finish_workspace_phase(&mut self, engine: &mut Engine) -> Result<bool> {
        let Some(phase) = self.policy.workspace_phase else {
            return Ok(false);
        };
        ensure!(
            engine.prefill_workspace_ready(),
            "ordinary prefill workspace is unavailable"
        );
        self.sync_planned_workspace_capacity(engine)?;
        let now = self.now();
        let mut changed = false;
        if self
            .policy
            .workspace_restoration_allowed(now, self.pressure_release_epoch)
        {
            let capacity = engine.expert_capacity();
            let restored = self
                .budget
                .preparation_budget_bytes(capacity, phase.ordinary_capacity)
                .and_then(|bytes| engine.recover_expert_capacity(phase.ordinary_capacity, bytes));
            match restored {
                Ok(report) => {
                    self.policy
                        .sync_planned_workspace_capacity(now, report.new_capacity)?;
                    changed = report.old_capacity != report.new_capacity;
                }
                Err(error) => {
                    self.policy.recovery_failed(now);
                    eprintln!("prefill phase: expert restoration deferred: {error:#}");
                }
            }
        }
        self.policy.finish_workspace_phase();
        Ok(changed)
    }

    pub fn new(budget: ExpertCapacityControlBudget) -> Result<Self> {
        ensure!(
            (MIN_CAPACITY..512).contains(&budget.initial_capacity()),
            "adaptive memory requires an initially bounded expert pool of at least 64 slots"
        );
        let policy = Policy::new(budget.initial_capacity(), budget.maximum_capacity())?;
        eprintln!(
            "adaptive memory: enabled=true initial_capacity={} maximum_capacity={} floor={} settle_seconds={} active_validation_seconds={}",
            budget.initial_capacity(),
            budget.maximum_capacity(),
            MIN_CAPACITY,
            RECOVERY_COOLDOWN.as_secs(),
            ACTIVE_VALIDATION.as_secs(),
        );
        Ok(Self {
            policy,
            budget,
            started: Instant::now(),
            #[cfg(test)]
            fixed_observation: None,
            pressure_release_epoch: 0,
        })
    }

    /// Start active-serving validation. Only sample intervals with both ends
    /// inside this request contribute to the validation duration.
    pub fn begin_request(&mut self) {
        self.policy.begin_request(self.now());
    }

    /// Accept validation after completed inference. Unsuccessful requests
    /// contribute no credit; prior completed work remains valid.
    pub fn end_request(&mut self, completed: bool) {
        self.policy.end_request(self.now(), completed);
    }

    #[cfg(test)]
    pub(crate) fn test_capacity_validation(&self) -> (Option<usize>, bool) {
        (
            self.policy.earned_target,
            self.policy.trial_capacity.is_some(),
        )
    }

    #[cfg(test)]
    pub(crate) fn test_pending_trial(&self) -> Option<(usize, usize)> {
        self.policy.trial_base.zip(self.policy.trial_capacity)
    }

    fn now(&self) -> Duration {
        #[cfg(test)]
        if let Some((now, _)) = self.fixed_observation {
            return now;
        }
        self.started.elapsed()
    }

    /// Replace only the host clock and observations, leaving request admission
    /// and engine capacity transactions on their production path.
    #[cfg(test)]
    pub(crate) fn fix_observation(&mut self, now: Duration, pressure_level: i32) {
        self.fixed_observation = Some((
            now,
            Observation {
                pressure: Pressure::from_raw(pressure_level),
                counters: Some(VmCounters {
                    compressions: 0,
                    decompressions: 0,
                    swapins: 0,
                    swapouts: 0,
                    page_bytes: 16_384,
                }),
            },
        ));
    }

    #[cfg(test)]
    pub(crate) fn begin_request_at(&mut self, now: Duration) {
        self.policy.begin_request(now);
    }

    #[cfg(test)]
    pub(crate) fn end_request_at(&mut self, now: Duration, completed: bool) {
        self.policy.end_request(now, completed);
    }

    /// Poll at a serialized owner boundary. Growth remains inside the admitted
    /// operating envelope; persistent critical pressure at the floor refuses work.
    pub fn check_boundary(&mut self, engine: &mut Engine, boundary: Boundary) -> Result<bool> {
        #[cfg(test)]
        if let Some((now, observation)) = self.fixed_observation {
            return self.apply_observation(engine, boundary, now, observation);
        }
        let now = self.started.elapsed();
        let observation = if self.policy.sample_due(now) {
            observe()
        } else {
            Observation {
                pressure: None,
                counters: None,
            }
        };
        self.apply_observation(engine, boundary, now, observation)
    }

    #[cfg(test)]
    pub(crate) fn check_with_observation(
        &mut self,
        engine: &mut Engine,
        starting_request: bool,
        now: Duration,
        pressure_level: Option<i32>,
        counters: Option<(u64, u64)>,
    ) -> Result<bool> {
        self.apply_observation(
            engine,
            if starting_request {
                Boundary::RequestStart
            } else if self.policy.request_started.is_some() {
                Boundary::CompletedOperation
            } else {
                Boundary::Observe
            },
            now,
            Observation {
                pressure: pressure_level.and_then(Pressure::from_raw),
                counters: counters.map(|(compressions, swapouts)| VmCounters {
                    compressions,
                    decompressions: 0,
                    swapins: 0,
                    swapouts,
                    page_bytes: 16_384,
                }),
            },
        )
    }

    fn apply_observation(
        &mut self,
        engine: &mut Engine,
        boundary: Boundary,
        now: Duration,
        observation: Observation,
    ) -> Result<bool> {
        let capacity = engine.expert_capacity();
        ensure!(
            (MIN_CAPACITY..=self.policy.maximum_capacity).contains(&capacity),
            "adaptive memory observed an expert capacity outside its operating extent"
        );
        let action = if let Some(action) = self.policy.observe(now, capacity, boundary, observation)
        {
            action
        } else {
            self.policy.cached_action(now, capacity)
        };
        match action {
            Action::Hold => Ok(false),
            Action::Stop => {
                bail!("adaptive memory: critical pressure persists at the 64-slot expert floor")
            }
            Action::Release(target) | Action::Rollback(target) | Action::ChurnRelease(target) => {
                self.release_capacity(engine, target, action, now)
            }
            Action::Recover(target) => {
                let mode = self.policy.mode(capacity);
                let preparation_started = Instant::now();
                let recovered = self
                    .budget
                    .preparation_budget_bytes(capacity, target)
                    .and_then(|bytes| engine.recover_expert_capacity(target, bytes));
                match recovered {
                    Ok(report) => {
                        self.policy
                            .capacity_changed(now, report.old_capacity, report.new_capacity);
                        eprintln!(
                            "adaptive memory: grew mode={} capacity={} -> {} requested_bytes={} preparation_bytes={} elapsed_ms={:.3}",
                            mode,
                            report.old_capacity,
                            report.new_capacity,
                            report.added_requested_bytes,
                            report.preparation_requested_bytes,
                            preparation_started.elapsed().as_secs_f64() * 1000.0,
                        );
                        Ok(true)
                    }
                    Err(error) => {
                        self.policy.recovery_failed(now);
                        eprintln!("adaptive memory: growth to {target} slots deferred: {error:#}");
                        Ok(false)
                    }
                }
            }
        }
    }

    fn release_capacity(
        &mut self,
        engine: &mut Engine,
        target: usize,
        action: Action,
        now: Duration,
    ) -> Result<bool> {
        let report = engine.shrink_expert_capacity(target)?;
        self.pressure_release_epoch += 1;
        let reason = match action {
            Action::Rollback(_) => {
                self.policy.pressure_interrupted_work(now);
                "trial-active-vm-churn"
            }
            Action::ChurnRelease(_) => {
                self.policy.interrupted_work(now, true);
                "sustained-active-vm-churn"
            }
            _ => "system-pressure",
        };
        self.policy
            .capacity_changed(now, report.old_capacity, report.new_capacity);
        eprintln!(
            "adaptive memory: released capacity={} -> {} requested_bytes={} reason={reason}",
            report.old_capacity, report.new_capacity, report.released_requested_bytes
        );
        Ok(true)
    }
}

fn pressure() -> Option<Pressure> {
    raw_pressure().and_then(Pressure::from_raw)
}

fn raw_pressure() -> Option<i32> {
    let mut raw = 0i32;
    let mut length = size_of::<i32>();
    // SAFETY: The NUL-terminated name and scalar output are valid for this
    // read-only sysctl call; an absent or differently sized value is unknown.
    let status = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&raw mut raw).cast(),
            &mut length,
            ptr::null_mut(),
            0,
        )
    };
    (status == 0 && length == size_of::<i32>()).then_some(raw)
}

unsafe extern "C" {
    fn mach_port_deallocate(
        task: libc::mach_port_t,
        name: libc::mach_port_t,
    ) -> libc::kern_return_t;
}

#[allow(deprecated)]
fn vm_counters() -> Option<VmCounters> {
    // SAFETY: HOST_VM_INFO64 fills this integer-only Darwin ABI struct. Check
    // its returned extent before accessing the compression and swap counters.
    let mut stats: libc::vm_statistics64_data_t = unsafe { zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let required = (offset_of!(libc::vm_statistics64_data_t, swapouts) + size_of::<u64>())
        .div_ceil(size_of::<libc::integer_t>()) as u32;
    // SAFETY: mach_host_self obtains a send right released after the query.
    let host = unsafe { libc::mach_host_self() };
    if host == 0 {
        return None;
    }
    // SAFETY: The struct and output count match the requested Mach flavor.
    let status = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&raw mut stats).cast(),
            &raw mut count,
        )
    };
    // SAFETY: Release only the send right obtained above.
    unsafe {
        mach_port_deallocate(libc::mach_task_self(), host);
    }
    if status != libc::KERN_SUCCESS || count < required {
        return None;
    }
    // SAFETY: sysconf only reads the page-size setting.
    let page_bytes = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    (page_bytes > 0).then_some(VmCounters {
        compressions: stats.compressions,
        decompressions: stats.decompressions,
        swapins: stats.swapins,
        swapouts: stats.swapouts,
        page_bytes,
    })
}

fn observe() -> Observation {
    Observation {
        pressure: pressure(),
        counters: vm_counters(),
    }
}

#[cfg(test)]
mod recovery_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_phase_ceiling_clamps_restoration_without_losing_earned_target() {
        let mut policy = Policy::new(202, 202).unwrap();
        policy.earned_target = Some(202);
        policy.maximum_capacity = 189;
        assert_eq!(policy.next_capacity(176), 189);
        assert_eq!(policy.earned_target, Some(202));
        assert_eq!(policy.retry_not_before, None);
        policy.maximum_capacity = 202;
        assert_eq!(policy.next_capacity(189), 202);
    }

    #[test]
    fn only_the_final_bounded_earned_tail_crosses_an_alignment_boundary() {
        for (capacity, earned, expected) in [
            (187, 201, 201),
            (188, 204, 204),
            (187, 204, 192),
            (176, 201, 192),
            (192, 201, 201),
        ] {
            let mut policy = Policy::new(capacity, 204).unwrap();
            policy.earned_target = Some(earned);
            assert_eq!(policy.next_capacity(capacity), expected);
            assert!(expected - capacity <= GROWTH_STEP_SLOTS);
        }
        let mut policy = Policy::new(187, 201).unwrap();
        complete_work(&mut policy, 0, 16, 187, 0);
        assert_eq!(policy.earned_target, Some(187));
        assert_eq!(
            policy.cached_action(Duration::from_secs(16), 187),
            Action::Recover(192)
        );
        policy.capacity_changed(Duration::from_secs(16), 187, 192);
        assert_eq!(policy.trial_capacity, Some(192));
        assert_eq!(policy.earned_target, Some(187));
    }

    #[test]
    fn earned_tail_keeps_fresh_observation_and_backoff_gates() {
        let ready = || {
            let mut policy = Policy::new(201, 201).unwrap();
            complete_work(&mut policy, 0, 16, 201, 0);
            assert!(policy.try_begin_workspace_phase(201, 187, 0).unwrap());
            policy
                .sync_planned_workspace_capacity(Duration::from_secs(17), 187)
                .unwrap();
            policy.finish_workspace_phase();
            quiet_for(&mut policy, 18, 23, 187);
            policy
        };
        let now = Duration::from_secs(23);
        let mut policy = ready();
        assert_eq!(policy.cached_action(now, 187), Action::Recover(201));
        assert_eq!(
            policy.cached_action(now + Duration::from_secs(2), 187),
            Action::Hold
        );
        policy.recovery_failed(now);
        assert_eq!(policy.cached_action(now, 187), Action::Hold);
        let mut policy = ready();
        assert!(policy.try_begin_workspace_phase(187, 187, 0).unwrap());
        assert_eq!(policy.cached_action(now, 187), Action::Hold);
        for (observation, expected) in [
            (normal(1024, 0), Action::Hold),
            (normal(0, 1), Action::Hold),
            (
                Observation {
                    pressure: None,
                    counters: None,
                },
                Action::Hold,
            ),
            (pressure_only(Pressure::Normal), Action::Hold),
            (pressure_only(Pressure::Warning), Action::Release(176)),
            (pressure_only(Pressure::Critical), Action::Release(64)),
        ] {
            let mut policy = ready();
            assert_eq!(
                step(
                    &mut policy,
                    24,
                    187,
                    Boundary::CompletedOperation,
                    observation
                ),
                expected
            );
            assert_eq!(policy.earned_target, Some(201));
        }
    }

    fn normal(compressions: u64, swapouts: u64) -> Observation {
        Observation {
            pressure: Some(Pressure::Normal),
            counters: Some(VmCounters {
                compressions,
                decompressions: 0,
                swapins: 0,
                swapouts,
                page_bytes: 16_384,
            }),
        }
    }

    fn pressure_only(pressure: Pressure) -> Observation {
        Observation {
            pressure: Some(pressure),
            counters: None,
        }
    }

    fn step(
        policy: &mut Policy,
        seconds: u64,
        capacity: usize,
        boundary: Boundary,
        observation: Observation,
    ) -> Action {
        policy
            .observe(
                Duration::from_secs(seconds),
                capacity,
                boundary,
                observation,
            )
            .unwrap()
    }

    fn quiet_for(policy: &mut Policy, first: u64, last: u64, capacity: usize) {
        for seconds in first..=last {
            assert!(matches!(
                step(policy, seconds, capacity, Boundary::Observe, normal(0, 0)),
                Action::Hold | Action::Recover(_)
            ));
        }
    }

    fn complete_work(
        policy: &mut Policy,
        first: u64,
        last: u64,
        capacity: usize,
        compressions: u64,
    ) {
        policy.begin_request(Duration::from_secs(first));
        for seconds in first..=last {
            // Isolate validation at this extent without applying growth offers.
            assert!(matches!(
                step(
                    policy,
                    seconds,
                    capacity,
                    Boundary::Observe,
                    normal(compressions, 0)
                ),
                Action::Hold | Action::Recover(_)
            ));
        }
        policy.end_request(Duration::from_secs(last), true);
    }

    #[test]
    fn fitting_workspace_trial_retains_credit_and_can_be_accepted() {
        let mut policy = open_trial(96, 112);
        policy.begin_request(Duration::from_secs(18));
        for second in 18..=26 {
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    112,
                    Boundary::CompletedOperation,
                    normal(0, 0)
                ),
                Action::Hold
            );
        }
        let credit = policy.request_clean_active;
        assert!(policy.try_begin_workspace_phase(112, 112, 0).unwrap());
        policy
            .sync_planned_workspace_capacity(Duration::from_secs(26), 112)
            .unwrap();
        assert_eq!(policy.request_clean_active, credit);
        assert_eq!(policy.trial_capacity, Some(112));
        for second in 27..=34 {
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    112,
                    Boundary::CompletedOperation,
                    normal(0, 0)
                ),
                Action::Hold
            );
        }
        policy.finish_workspace_phase();
        policy.end_request(Duration::from_secs(34), true);
        assert_eq!(policy.earned_target, Some(112));
        assert_eq!(policy.trial_capacity, None);
        assert_eq!(policy.initial_capacity, 96);
        assert_eq!(policy.maximum_capacity, 204);
    }

    #[test]
    fn workspace_phase_freezes_fresh_and_cached_growth() {
        let mut policy = Policy::new(96, 128).unwrap();
        complete_work(&mut policy, 0, 16, 96, 0);
        assert!(policy.try_begin_workspace_phase(96, 96, 0).unwrap());
        assert_eq!(
            step(
                &mut policy,
                17,
                96,
                Boundary::CompletedOperation,
                normal(0, 0)
            ),
            Action::Hold
        );
        assert_eq!(
            policy.cached_action(Duration::from_millis(17_500), 96),
            Action::Hold
        );
        assert_eq!(policy.trial_capacity, None);
        assert_eq!(policy.completed_clean_active, ACTIVE_VALIDATION);
        policy.finish_workspace_phase();
        assert_eq!(
            policy.cached_action(Duration::from_millis(17_500), 96),
            Action::Recover(112)
        );
    }

    #[test]
    fn smaller_workspace_extent_cannot_accept_retained_ordinary_credit() {
        let mut policy = Policy::new(96, 128).unwrap();
        complete_work(&mut policy, 0, 15, 96, 0);
        assert_eq!(policy.completed_clean_active, Duration::from_secs(14));
        policy.begin_request(Duration::from_secs(16));
        for second in 16..=17 {
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    96,
                    Boundary::CompletedOperation,
                    normal(0, 0)
                ),
                Action::Recover(112)
            );
        }
        assert_eq!(policy.request_clean_active, Duration::from_secs(1));
        assert!(policy.try_begin_workspace_phase(96, 80, 0).unwrap());
        policy
            .sync_planned_workspace_capacity(Duration::from_secs(18), 80)
            .unwrap();
        for second in 18..=34 {
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    80,
                    Boundary::CompletedOperation,
                    normal(0, 0)
                ),
                Action::Hold
            );
        }
        assert_eq!(policy.request_clean_active, Duration::from_secs(1));
        policy.end_request(Duration::from_secs(34), true);
        assert_eq!(policy.earned_target, None);
        assert_eq!(policy.completed_clean_active, Duration::from_secs(14));
        policy.finish_workspace_phase();
        assert_eq!(policy.completed_clean_active, Duration::ZERO);
        assert_eq!(policy.request_clean_active, Duration::ZERO);
        assert_eq!(policy.last_release, None);
        assert_eq!(policy.last_increase, None);
        assert_eq!(policy.trial_capacity, None);
    }

    #[test]
    fn returning_to_ordinary_extent_preserves_its_prior_credit() {
        let mut policy = Policy::new(96, 128).unwrap();
        complete_work(&mut policy, 0, 15, 96, 0);
        assert!(policy.try_begin_workspace_phase(96, 80, 0).unwrap());
        policy
            .sync_planned_workspace_capacity(Duration::from_secs(16), 80)
            .unwrap();
        policy.begin_request(Duration::from_secs(16));
        for second in 16..=32 {
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    80,
                    Boundary::CompletedOperation,
                    normal(0, 0)
                ),
                Action::Hold
            );
        }
        assert_eq!(policy.request_clean_active, Duration::ZERO);
        assert!(policy.workspace_restoration_allowed(Duration::from_secs(32), 0));
        assert!(!policy.workspace_restoration_allowed(Duration::from_secs(32), 1));
        policy
            .sync_planned_workspace_capacity(Duration::from_secs(32), 96)
            .unwrap();
        policy.finish_workspace_phase();
        assert_eq!(policy.completed_clean_active, Duration::from_secs(14));
        assert_eq!(policy.trial_capacity, None);
        for second in 33..=35 {
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    96,
                    Boundary::CompletedOperation,
                    normal(0, 0)
                ),
                Action::Hold
            );
        }
        policy.end_request(Duration::from_secs(35), true);
        assert_eq!(policy.earned_target, Some(96));
    }

    #[test]
    fn warning_at_workspace_floor_vetoes_restoration_without_a_release() {
        let mut policy = Policy::new(96, 128).unwrap();
        complete_work(&mut policy, 0, 16, 96, 0);
        assert!(policy.try_begin_workspace_phase(96, 64, 0).unwrap());
        policy
            .sync_planned_workspace_capacity(Duration::from_secs(17), 64)
            .unwrap();
        assert_eq!(
            step(
                &mut policy,
                18,
                64,
                Boundary::CompletedOperation,
                pressure_only(Pressure::Warning)
            ),
            Action::Hold
        );
        assert!(!policy.workspace_restoration_allowed(Duration::from_secs(18), 0));
        assert_eq!(policy.earned_target, Some(96));
        assert_eq!(policy.last_release, None);
        assert_eq!(
            step(
                &mut policy,
                19,
                64,
                Boundary::CompletedOperation,
                pressure_only(Pressure::Critical)
            ),
            Action::Hold
        );
        assert_eq!(
            step(
                &mut policy,
                20,
                64,
                Boundary::CompletedOperation,
                pressure_only(Pressure::Critical)
            ),
            Action::Stop
        );
    }

    #[test]
    fn fitting_workspace_trial_still_rolls_back_on_active_churn() {
        let mut policy = open_trial(96, 112);
        assert!(policy.try_begin_workspace_phase(112, 112, 0).unwrap());
        policy.begin_request(Duration::from_secs(18));
        for second in 18..=32 {
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    112,
                    Boundary::CompletedOperation,
                    normal((second - 17) * 2048, 0)
                ),
                Action::Hold
            );
        }
        assert_eq!(
            step(
                &mut policy,
                33,
                112,
                Boundary::CompletedOperation,
                normal(16 * 2048, 0)
            ),
            Action::Rollback(96)
        );
        assert!(policy.workspace_phase.is_some());
        assert_eq!(policy.earned_target, Some(96));
    }

    #[test]
    fn extents_and_darwin_levels_fail_closed() {
        for raw in [-1, 0, 3, 5, 7, 8] {
            assert_eq!(Pressure::from_raw(raw), None);
        }
        assert_eq!(Pressure::from_raw(1), Some(Pressure::Normal));
        assert_eq!(Pressure::from_raw(2), Some(Pressure::Warning));
        assert_eq!(Pressure::from_raw(4), Some(Pressure::Critical));
        for initial in [0, 10, 63, 512, 513] {
            assert!(Policy::new(initial, 204).is_err());
        }
        assert!(Policy::new(147, 146).is_err());
        assert!(Policy::new(147, 512).is_err());
    }

    #[test]
    fn initially_at_floor_never_releases_below_it_and_refuses_persistent_critical() {
        let mut policy = Policy::new(64, 64).unwrap();
        assert_eq!(
            step(
                &mut policy,
                0,
                64,
                Boundary::Observe,
                pressure_only(Pressure::Warning)
            ),
            Action::Hold
        );
        assert_eq!(
            step(
                &mut policy,
                1,
                64,
                Boundary::Observe,
                pressure_only(Pressure::Critical)
            ),
            Action::Hold
        );
        assert_eq!(
            step(
                &mut policy,
                2,
                64,
                Boundary::Observe,
                pressure_only(Pressure::Critical)
            ),
            Action::Stop
        );
        assert_eq!(
            policy.cached_action(Duration::from_millis(2500), 64),
            Action::Stop
        );
        assert!(Policy::new(64, 511).is_ok());
    }

    #[test]
    fn initially_at_floor_can_grow_after_completed_active_validation() {
        let mut policy = Policy::new(64, 80).unwrap();
        complete_work(&mut policy, 0, 16, 64, 0);
        assert_eq!(policy.earned_target, Some(64));
        assert_eq!(
            policy.cached_action(Duration::from_secs(17), 64),
            Action::Recover(80)
        );
        let mut fixed_floor = Policy::new(64, 64).unwrap();
        complete_work(&mut fixed_floor, 0, 16, 64, 0);
        assert_eq!(
            fixed_floor.cached_action(Duration::from_secs(17), 64),
            Action::Hold
        );
    }

    #[test]
    fn compression_pauses_allocation_without_erasing_completed_work() {
        let mut policy = Policy::new(147, 204).unwrap();
        complete_work(&mut policy, 0, 9, 147, 0);
        assert_eq!(policy.completed_clean_active, Duration::from_secs(8));
        step(&mut policy, 10, 147, Boundary::Observe, normal(1024, 0));
        assert_eq!(policy.state, Some(State::NormalBusy));
        assert_eq!(policy.completed_clean_active, Duration::from_secs(8));
        assert_eq!(policy.quiet_since, None);
        policy.begin_request(Duration::from_secs(11));
        for seconds in 11..=18 {
            step(
                &mut policy,
                seconds,
                147,
                Boundary::Observe,
                normal(1024, 0),
            );
        }
        policy.end_request(Duration::from_secs(18), true);
        assert_eq!(policy.earned_target, Some(147));
        assert_eq!(
            policy.cached_action(Duration::from_secs(18), 147),
            Action::Recover(160)
        );
    }

    #[test]
    fn cancellation_discards_only_current_request_credit() {
        let mut policy = Policy::new(147, 204).unwrap();
        complete_work(&mut policy, 0, 9, 147, 0);
        policy.begin_request(Duration::from_secs(10));
        quiet_for(&mut policy, 10, 20, 147);
        policy.end_request(Duration::from_secs(20), false);
        assert_eq!(policy.completed_clean_active, Duration::from_secs(8));
        assert_eq!(policy.request_clean_active, Duration::ZERO);
        assert_eq!(policy.earned_target, None);
        complete_work(&mut policy, 21, 28, 147, 0);
        assert_eq!(policy.earned_target, Some(147));
        policy.begin_request(Duration::from_secs(29));
        step(&mut policy, 29, 147, Boundary::Observe, normal(0, 0));
        policy.end_request(Duration::from_secs(29), false);
        assert_eq!(policy.earned_target, Some(147));
        assert_eq!(policy.completed_clean_active, ACTIVE_VALIDATION);
    }

    #[test]
    fn pressure_preserves_earned_target_and_permits_settled_recovery() {
        let mut policy = Policy::new(147, 204).unwrap();
        complete_work(&mut policy, 0, 16, 147, 0);
        assert_eq!(
            step(
                &mut policy,
                17,
                147,
                Boundary::Observe,
                pressure_only(Pressure::Warning)
            ),
            Action::Release(144)
        );
        policy.capacity_changed(Duration::from_secs(17), 147, 144);
        assert_eq!(policy.earned_target, Some(147));
        policy.begin_request(Duration::from_secs(18));
        quiet_for(&mut policy, 18, 24, 144);
        assert_eq!(
            policy.cached_action(Duration::from_secs(24), 144),
            Action::Recover(147)
        );
        assert_eq!(policy.completed_clean_active, Duration::ZERO);
        policy.capacity_changed(Duration::from_secs(24), 144, 147);
        quiet_for(&mut policy, 25, 30, 147);
        assert_eq!(
            policy.cached_action(Duration::from_secs(30), 147),
            Action::Recover(160)
        );
        assert_eq!(policy.earned_target, Some(147));
    }

    #[test]
    fn unknown_gaps_and_counter_resets_break_recent_eligibility_not_history() {
        let mut policy = Policy::new(147, 204).unwrap();
        complete_work(&mut policy, 0, 16, 147, 0);
        step(
            &mut policy,
            17,
            147,
            Boundary::Observe,
            Observation {
                pressure: None,
                counters: None,
            },
        );
        assert_eq!(policy.earned_target, Some(147));
        assert_eq!(policy.completed_clean_active, ACTIVE_VALIDATION);
        assert_eq!(policy.quiet_since, None);
        assert_eq!(
            policy.cached_action(Duration::from_secs(17), 147),
            Action::Hold
        );
        quiet_for(&mut policy, 18, 24, 147);
        assert_eq!(
            policy.cached_action(Duration::from_secs(24), 147),
            Action::Recover(160)
        );
        assert_eq!(
            step(&mut policy, 40, 147, Boundary::RequestStart, normal(0, 0)),
            Action::Hold
        );
        assert_eq!(policy.state, Some(State::Unknown));
        assert_eq!(policy.completed_clean_active, ACTIVE_VALIDATION);
        step(&mut policy, 41, 147, Boundary::Observe, normal(10, 0));
        step(&mut policy, 42, 147, Boundary::Observe, normal(0, 0));
        assert_eq!(policy.state, Some(State::Unknown));
        assert_eq!(policy.earned_target, Some(147));
    }

    #[test]
    fn interrupted_trial_is_rejected_without_demoting_the_earned_base() {
        let mut policy = Policy::new(147, 204).unwrap();
        complete_work(&mut policy, 0, 16, 147, 0);
        policy.capacity_changed(Duration::from_secs(16), 147, 160);
        assert_eq!(
            step(
                &mut policy,
                17,
                160,
                Boundary::Observe,
                pressure_only(Pressure::Warning)
            ),
            Action::Release(144)
        );
        assert_eq!(policy.earned_target, Some(147));
        assert_eq!(policy.trial_capacity, None);
        assert_eq!(policy.retry_delay, Duration::from_secs(10));
        assert_eq!(policy.retry_not_before, Some(Duration::from_secs(27)));
        policy.capacity_changed(Duration::from_secs(17), 160, 144);
        quiet_for(&mut policy, 18, 26, 144);
        assert_eq!(
            policy.cached_action(Duration::from_secs(26), 144),
            Action::Hold
        );
        step(&mut policy, 27, 144, Boundary::Observe, normal(0, 0));
        assert_eq!(
            policy.cached_action(Duration::from_secs(27), 144),
            Action::Recover(147)
        );
        policy.capacity_changed(Duration::from_secs(27), 144, 147);
        complete_work(&mut policy, 28, 44, 147, 0);
        assert_eq!(policy.retry_delay, Duration::from_secs(10));
        assert_eq!(policy.retry_not_before, Some(Duration::from_secs(27)));
    }

    #[test]
    fn repeated_actual_failed_increases_back_off_with_a_bounded_delay() {
        let mut policy = Policy::new(147, 204).unwrap();
        let mut now = 0;
        for expected in [10, 20, 40, 80, 120, 120] {
            policy.state = Some(State::NormalQuiet);
            policy.capacity_changed(Duration::from_secs(now), 147, 160);
            policy.pressure_interrupted_work(Duration::from_secs(now + 1));
            assert_eq!(policy.retry_delay, Duration::from_secs(expected));
            let deadline = policy.retry_not_before;
            policy.pressure_interrupted_work(Duration::from_secs(now + 2));
            assert_eq!(policy.retry_not_before, deadline);
            now += expected + 2;
        }
    }

    #[test]
    fn warning_release_is_small_and_critical_floor_refusal_remains_immediate() {
        let mut policy = Policy::new(160, 204).unwrap();
        let warning = pressure_only(Pressure::Warning);
        assert_eq!(
            step(&mut policy, 0, 160, Boundary::Observe, warning),
            Action::Release(144)
        );
        policy.capacity_changed(Duration::ZERO, 160, 144);
        for second in 1..5 {
            assert_eq!(
                step(&mut policy, second, 144, Boundary::Observe, warning),
                Action::Hold
            );
        }
        assert_eq!(
            step(&mut policy, 5, 144, Boundary::Observe, warning),
            Action::Release(128)
        );
        let critical = pressure_only(Pressure::Critical);
        assert_eq!(
            step(&mut policy, 6, 144, Boundary::Observe, critical),
            Action::Release(64)
        );
        policy.capacity_changed(Duration::from_secs(6), 144, 64);
        assert_eq!(
            step(&mut policy, 7, 64, Boundary::Observe, critical),
            Action::Stop
        );
        assert_eq!(
            policy.cached_action(Duration::from_millis(7500), 64),
            Action::Stop
        );
        assert_eq!(
            policy.cached_action(Duration::from_secs(9), 64),
            Action::Hold
        );
    }

    #[test]
    fn failed_allocation_retains_work_and_cannot_retry_from_cached_eligibility() {
        let mut policy = Policy::new(147, 204).unwrap();
        complete_work(&mut policy, 0, 16, 147, 0);
        policy.recovery_failed(Duration::from_secs(16));
        assert_eq!(policy.earned_target, Some(147));
        assert_eq!(policy.completed_clean_active, ACTIVE_VALIDATION);
        assert_eq!(
            policy.cached_action(Duration::from_millis(16500), 147),
            Action::Hold
        );
        quiet_for(&mut policy, 17, 26, 147);
        assert_eq!(
            policy.cached_action(Duration::from_secs(26), 147),
            Action::Recover(160)
        );
    }

    fn open_trial(base: usize, candidate: usize) -> Policy {
        let mut policy = Policy::new(base, 204).unwrap();
        complete_work(&mut policy, 0, 16, base, 0);
        policy.capacity_changed(Duration::from_secs(17), base, candidate);
        policy
    }

    #[test]
    fn brief_trial_activity_can_still_earn_the_candidate() {
        let mut policy = open_trial(145, 160);
        policy.begin_request(Duration::from_secs(18));
        step(
            &mut policy,
            18,
            160,
            Boundary::CompletedOperation,
            normal(0, 0),
        );
        assert_eq!(
            step(
                &mut policy,
                19,
                160,
                Boundary::CompletedOperation,
                normal(1024, 0)
            ),
            Action::Hold
        );
        assert_eq!(policy.trial_adverse_active, Duration::from_secs(1));
        for second in 20..=36 {
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    160,
                    Boundary::CompletedOperation,
                    normal(1024, 0)
                ),
                if second >= 35 {
                    Action::Recover(176)
                } else {
                    Action::Hold
                }
            );
        }
        policy.end_request(Duration::from_secs(36), true);
        assert_eq!(policy.earned_target, Some(160));
        assert_eq!(policy.trial_capacity, None);
        assert_eq!(policy.trial_adverse_active, Duration::ZERO);
    }

    #[test]
    fn adverse_trial_work_accumulates_across_cancelled_and_completed_requests() {
        let mut policy = open_trial(145, 160);
        let mut compressions = 0;
        policy.begin_request(Duration::from_secs(18));
        for second in 18..=26 {
            compressions += 1024;
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    160,
                    Boundary::CompletedOperation,
                    normal(compressions, 0)
                ),
                Action::Hold
            );
        }
        policy.end_request(Duration::from_secs(26), false);
        assert_eq!(policy.trial_adverse_active, Duration::from_secs(8));
        policy.begin_request(Duration::from_secs(30));
        for second in 30..=37 {
            compressions += 1024;
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    160,
                    Boundary::CompletedOperation,
                    normal(compressions, 0)
                ),
                if second == 37 {
                    Action::Rollback(144)
                } else {
                    Action::Hold
                }
            );
        }
        assert_eq!(policy.trial_adverse_active, ACTIVE_VALIDATION);
        assert_eq!(policy.trial_capacity, Some(160));
        assert_eq!(policy.earned_target, Some(145));
        // An action recommendation leaves the pending trial intact until release succeeds.
        assert_eq!(policy.retry_not_before, None);
        assert_eq!(
            policy.cached_action(Duration::from_millis(37_500), 160),
            Action::Hold
        );
        assert!(
            policy
                .observe(
                    Duration::from_millis(37_500),
                    160,
                    Boundary::CompletedOperation,
                    normal(compressions, 0)
                )
                .is_none()
        );
        assert_eq!(policy.trial_adverse_active, ACTIVE_VALIDATION);
    }

    #[test]
    fn idle_trial_adversity_counts_but_gaps_and_invalid_counters_do_not() {
        let mut policy = open_trial(145, 160);
        policy.begin_request(Duration::from_secs(18));
        step(
            &mut policy,
            18,
            160,
            Boundary::CompletedOperation,
            normal(0, 0),
        );
        step(
            &mut policy,
            19,
            160,
            Boundary::CompletedOperation,
            normal(1024, 0),
        );
        policy.end_request(Duration::from_secs(19), true);
        step(&mut policy, 20, 160, Boundary::Observe, normal(2048, 0));
        step(&mut policy, 21, 160, Boundary::Observe, normal(3072, 0));
        policy.begin_request(Duration::from_secs(22));
        step(
            &mut policy,
            22,
            160,
            Boundary::CompletedOperation,
            normal(4096, 0),
        );
        assert_eq!(policy.trial_adverse_active, Duration::from_secs(2));
        step(
            &mut policy,
            30,
            160,
            Boundary::CompletedOperation,
            normal(20_000, 0),
        );
        assert_eq!(policy.state, Some(State::Unknown));
        step(
            &mut policy,
            31,
            160,
            Boundary::CompletedOperation,
            normal(0, 0),
        );
        assert_eq!(policy.state, Some(State::Unknown));
        step(
            &mut policy,
            32,
            160,
            Boundary::CompletedOperation,
            Observation {
                pressure: Some(Pressure::Normal),
                counters: None,
            },
        );
        step(
            &mut policy,
            33,
            160,
            Boundary::CompletedOperation,
            normal(1024, 0),
        );
        assert_eq!(policy.trial_adverse_active, Duration::from_secs(2));
        step(
            &mut policy,
            34,
            160,
            Boundary::CompletedOperation,
            normal(2048, 0),
        );
        assert_eq!(policy.trial_adverse_active, Duration::from_secs(3));
    }

    #[test]
    fn accepted_capacity_is_not_released_for_one_way_compression() {
        let mut policy = Policy::new(145, 204).unwrap();
        complete_work(&mut policy, 0, 16, 145, 0);
        policy.begin_request(Duration::from_secs(17));
        for second in 17..=60 {
            assert_eq!(
                step(
                    &mut policy,
                    second,
                    145,
                    Boundary::CompletedOperation,
                    normal(second * 1024, 0)
                ),
                Action::Hold
            );
        }
        assert_eq!(policy.earned_target, Some(145));
        assert_eq!(policy.trial_adverse_active, Duration::ZERO);
    }

    #[test]
    fn aligned_trial_rollback_preserves_non_aligned_earned_target_and_delays_retrial() {
        let mut policy = open_trial(145, 160);
        policy.trial_adverse_active = ACTIVE_VALIDATION;
        assert_eq!(
            step(
                &mut policy,
                18,
                160,
                Boundary::CompletedOperation,
                normal(2048, 0)
            ),
            Action::Rollback(144)
        );
        policy.pressure_interrupted_work(Duration::from_secs(18));
        policy.capacity_changed(Duration::from_secs(18), 160, 144);
        assert_eq!(policy.earned_target, Some(145));
        assert_eq!(policy.trial_capacity, None);
        assert_eq!(policy.trial_adverse_active, Duration::ZERO);
        assert_eq!(policy.retry_not_before, Some(Duration::from_secs(28)));
        quiet_for(&mut policy, 19, 27, 144);
        assert_eq!(
            policy.cached_action(Duration::from_secs(27), 144),
            Action::Hold
        );
        step(&mut policy, 28, 144, Boundary::Observe, normal(0, 0));
        assert_eq!(
            policy.cached_action(Duration::from_secs(28), 144),
            Action::Recover(145)
        );
        policy.capacity_changed(Duration::from_secs(28), 144, 145);
        quiet_for(&mut policy, 29, 34, 145);
        assert_eq!(
            policy.cached_action(Duration::from_secs(34), 145),
            Action::Recover(160)
        );
        complete_work(&mut policy, 35, 51, 145, 0);
        assert_eq!(
            policy.cached_action(Duration::from_secs(51), 145),
            Action::Recover(160)
        );
        assert_eq!(policy.retry_delay, Duration::from_secs(10));
    }

    #[test]
    fn critical_pressure_precedes_pending_trial_rollback() {
        let mut policy = open_trial(145, 160);
        policy.trial_adverse_active = ACTIVE_VALIDATION;
        assert_eq!(
            step(
                &mut policy,
                18,
                160,
                Boundary::CompletedOperation,
                pressure_only(Pressure::Critical)
            ),
            Action::Release(64)
        );
        assert_eq!(policy.earned_target, Some(145));
        assert_eq!(policy.trial_capacity, None);
        assert_eq!(policy.trial_adverse_active, Duration::ZERO);
    }

    #[test]
    fn any_swapout_and_threshold_compression_veto_recent_eligibility() {
        let before = normal(0, 0).counters.unwrap();
        assert_eq!(
            normal(1023, 0)
                .counters
                .unwrap()
                .quiet_since(before, SAMPLE_INTERVAL),
            Some(true)
        );
        assert_eq!(
            normal(1024, 0)
                .counters
                .unwrap()
                .quiet_since(before, SAMPLE_INTERVAL),
            Some(false)
        );
        assert_eq!(
            normal(0, 1)
                .counters
                .unwrap()
                .quiet_since(before, SAMPLE_INTERVAL),
            Some(false)
        );
        let mut invalid = before;
        invalid.page_bytes = 0;
        assert_eq!(invalid.quiet_since(before, SAMPLE_INTERVAL), None);
        invalid = before;
        invalid.compressions = u64::MAX;
        assert_eq!(invalid.quiet_since(before, SAMPLE_INTERVAL), None);
    }
}

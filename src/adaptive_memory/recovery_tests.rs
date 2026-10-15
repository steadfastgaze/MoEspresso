use super::*;

fn policy(initial: usize, maximum: usize) -> Policy {
    Policy::new(initial, maximum).unwrap()
}

fn idle_prepared(initial: usize) -> Policy {
    let mut policy = policy(initial, 201);
    for second in 0..=180 {
        tick(&mut policy, second, observation(Pressure::Normal, 0, 0));
    }
    assert_eq!(policy.capacity, 201);
    assert_eq!(policy.trial_base, Some(initial));
    policy
}

fn request_tick(policy: &mut Policy, second: u64) -> Action {
    let now = Duration::from_secs(second);
    let action = policy
        .observe(
            now,
            policy.capacity,
            Boundary::CompletedOperation,
            observation(Pressure::Normal, 0, 0),
        )
        .unwrap();
    if let Action::Recover(next) = action {
        policy.capacity_changed(now, policy.capacity, next);
    }
    action
}

#[test]
fn active_preparation_retains_qualified_extents_until_successful_completion() {
    let mut policy = policy(88, 201);
    policy.begin_request(Duration::ZERO);
    for second in 0..=40 {
        request_tick(&mut policy, second);
    }
    assert_eq!(policy.capacity, 128);
    assert_eq!(policy.trial_base, Some(88));
    assert_eq!(policy.earned_target, None);
    policy.end_request(Duration::from_secs(40), true);
    assert_eq!(policy.earned_target, Some(112));
    assert_eq!(policy.trial_base, Some(112));
    assert_eq!(policy.trial_capacity, Some(128));
    policy.begin_request(Duration::from_secs(41));
    assert_eq!(request_tick(&mut policy, 41), Action::Hold);
}

#[test]
fn active_preparation_preserves_validation_across_short_requests() {
    let mut policy = policy(88, 201);
    policy.begin_request(Duration::ZERO);
    for second in 0..=16 {
        request_tick(&mut policy, second);
    }
    policy.end_request(Duration::from_secs(16), true);
    assert_eq!(policy.capacity, 96);
    assert!(policy.completed_clean_active > Duration::ZERO);
    assert_eq!(policy.earned_target, None);
    let completed_credit = policy.completed_clean_active;
    for second in 17..=30 {
        assert_eq!(
            tick(&mut policy, second, observation(Pressure::Normal, 0, 0)),
            Action::Hold
        );
    }
    assert_eq!(policy.completed_clean_active, completed_credit);
    policy.begin_request(Duration::from_secs(31));
    for second in 31..=44 {
        request_tick(&mut policy, second);
    }
    assert_eq!(policy.capacity, 112);
    policy.end_request(Duration::from_secs(44), true);
    assert_eq!(policy.earned_target, Some(96));
    assert_eq!(policy.trial_base, Some(96));
    assert_eq!(policy.trial_capacity, Some(112));
}

#[test]
fn failed_request_cannot_accept_prepared_storage() {
    let mut policy = policy(88, 201);
    policy.begin_request(Duration::ZERO);
    for second in 0..=40 {
        request_tick(&mut policy, second);
    }
    policy.end_request(Duration::from_secs(40), false);
    assert_eq!(policy.capacity, 128);
    assert_eq!(policy.earned_target, None);
    assert_eq!(policy.completed_clean_active, Duration::ZERO);
    assert_eq!(policy.trial_base, Some(88));
}

#[test]
fn pressure_discards_in_request_qualification_before_completion() {
    let mut policy = policy(88, 201);
    policy.begin_request(Duration::ZERO);
    for second in 0..=40 {
        request_tick(&mut policy, second);
    }
    assert_eq!(policy.validated_in_request, Some(112));
    assert_eq!(
        tick(&mut policy, 41, observation(Pressure::Warning, 0, 0)),
        Action::Release(112)
    );
    for second in 42..=44 {
        request_tick(&mut policy, second);
    }
    policy.end_request(Duration::from_secs(44), true);
    assert_eq!(policy.earned_target, None);
}

fn accepted_policy() -> Policy {
    let mut policy = policy(201, 201);
    policy.earned_target = Some(201);
    policy
}

fn active_tick(policy: &mut Policy, second: u64, counters: u64) -> Action {
    let mut sample = observation(Pressure::Normal, counters, 0);
    sample.counters.as_mut().unwrap().decompressions = counters;
    policy
        .observe(
            Duration::from_secs(second),
            policy.capacity,
            Boundary::CompletedOperation,
            sample,
        )
        .unwrap()
}

#[test]
fn sustained_one_way_compression_blocks_growth_without_evicting_accepted_storage() {
    let mut policy = policy(192, 201);
    policy.earned_target = Some(192);
    policy.begin_request(Duration::ZERO);
    for second in 0..=40 {
        assert_eq!(
            policy.observe(
                Duration::from_secs(second),
                192,
                Boundary::CompletedOperation,
                observation(Pressure::Normal, second * 180_000, 0)
            ),
            Some(Action::Hold)
        );
    }
    assert_eq!(policy.state, Some(State::NormalBusy));
    assert_eq!(policy.capacity, 192);
    assert_eq!(policy.earned_target, Some(192));
    assert_eq!(policy.request_clean_active, Duration::ZERO);
}

#[test]
fn active_swapouts_release_accepted_storage_without_requiring_compression() {
    let mut policy = accepted_policy();
    policy.begin_request(Duration::ZERO);
    for second in 0..15 {
        assert_eq!(
            policy.observe(
                Duration::from_secs(second),
                201,
                Boundary::CompletedOperation,
                observation(Pressure::Normal, 0, second)
            ),
            Some(Action::Hold)
        );
    }
    assert_eq!(
        policy.observe(
            Duration::from_secs(15),
            201,
            Boundary::CompletedOperation,
            observation(Pressure::Normal, 0, 15)
        ),
        Some(Action::ChurnRelease(192))
    );
}

#[test]
fn sustained_active_churn_releases_even_previously_accepted_capacity() {
    let mut policy = accepted_policy();
    policy.begin_request(Duration::ZERO);
    for second in 0..15 {
        assert_eq!(
            active_tick(&mut policy, second, second * 180_000),
            Action::Hold
        );
    }
    assert_eq!(
        active_tick(&mut policy, 15, 15 * 180_000),
        Action::ChurnRelease(192)
    );
    assert_eq!(policy.earned_target, Some(201));
    assert_eq!(policy.maximum_capacity, 201);
}

#[test]
fn stalled_owner_intervals_cannot_hide_sustained_active_churn() {
    let mut policy = accepted_policy();
    policy.begin_request(Duration::ZERO);
    assert_eq!(active_tick(&mut policy, 0, 0), Action::Hold);
    assert_eq!(active_tick(&mut policy, 8, 8 * 180_000), Action::Hold);
    assert_eq!(active_tick(&mut policy, 16, 16 * 180_000), Action::Hold);
    assert_eq!(
        active_tick(&mut policy, 24, 24 * 180_000),
        Action::ChurnRelease(192)
    );
    assert_eq!(policy.completed_clean_active, Duration::ZERO);
    assert_eq!(policy.request_clean_active, Duration::ZERO);
}

#[test]
fn brief_churn_and_idle_compression_do_not_evict_accepted_capacity() {
    let mut policy = accepted_policy();
    for second in 0..=40 {
        assert_eq!(
            tick(
                &mut policy,
                second,
                observation(Pressure::Normal, second * 180_000, 0)
            ),
            Action::Hold
        );
    }
    policy.begin_request(Duration::from_secs(41));
    for second in 41..=46 {
        assert_eq!(
            active_tick(&mut policy, second, second * 180_000),
            Action::Hold
        );
    }
    for second in 47..=70 {
        assert_eq!(active_tick(&mut policy, second, 46 * 180_000), Action::Hold);
    }
    assert_eq!(policy.capacity, 201);
}

#[test]
fn churn_release_backs_off_then_restores_without_lowering_the_ceiling() {
    let mut policy = accepted_policy();
    policy.begin_request(Duration::ZERO);
    for second in 0..15 {
        active_tick(&mut policy, second, second * 180_000);
    }
    assert_eq!(
        active_tick(&mut policy, 15, 15 * 180_000),
        Action::ChurnRelease(192)
    );
    policy.interrupted_work(Duration::from_secs(15), true);
    policy.capacity_changed(Duration::from_secs(15), 201, 192);
    assert_eq!(policy.retry_not_before, Some(Duration::from_secs(25)));
    policy.end_request(Duration::from_secs(15), false);
    for second in 16..25 {
        assert_eq!(
            tick(
                &mut policy,
                second,
                observation(Pressure::Normal, 15 * 180_000, 0)
            ),
            Action::Hold
        );
    }
    assert_eq!(
        tick(
            &mut policy,
            25,
            observation(Pressure::Normal, 15 * 180_000, 0)
        ),
        Action::Recover(201)
    );
    assert_eq!(policy.earned_target, Some(201));
    assert_eq!(policy.maximum_capacity, 201);
    policy.interrupted_work(Duration::from_secs(26), true);
    assert_eq!(policy.retry_not_before, Some(Duration::from_secs(46)));
}

#[test]
fn missing_reset_and_inter_request_samples_do_not_invent_churn() {
    for reset in [false, true] {
        let mut policy = accepted_policy();
        policy.begin_request(Duration::ZERO);
        for second in 0..14 {
            active_tick(&mut policy, second, second * 180_000);
        }
        let next = if reset {
            observation(Pressure::Normal, 0, 0)
        } else {
            Observation {
                pressure: None,
                counters: None,
            }
        };
        assert_eq!(
            policy.observe(
                Duration::from_secs(14),
                201,
                Boundary::CompletedOperation,
                next
            ),
            Some(Action::Hold)
        );
        assert_eq!(policy.active_churn, Duration::ZERO);
    }
    let mut policy = accepted_policy();
    policy.begin_request(Duration::ZERO);
    active_tick(&mut policy, 0, 0);
    policy.end_request(Duration::ZERO, true);
    policy.begin_request(Duration::from_secs(100));
    assert_eq!(active_tick(&mut policy, 100, 100 * 180_000), Action::Hold);
    assert_eq!(policy.active_churn, Duration::ZERO);
}

#[test]
fn churn_does_not_release_below_floor_and_critical_still_stops() {
    let mut policy = policy(64, 201);
    policy.earned_target = Some(201);
    policy.begin_request(Duration::ZERO);
    for second in 0..=40 {
        assert_eq!(
            active_tick(&mut policy, second, second * 180_000),
            Action::Hold
        );
    }
    assert_eq!(
        tick(
            &mut policy,
            41,
            observation(Pressure::Critical, 40 * 180_000, 0)
        ),
        Action::Hold
    );
    assert_eq!(
        tick(
            &mut policy,
            42,
            observation(Pressure::Critical, 40 * 180_000, 0)
        ),
        Action::Stop
    );
}

#[test]
fn provisional_extent_yields_to_wide_prefill_without_earning_or_restoring_its_tail() {
    let mut policy = idle_prepared(88);
    policy.trial_adverse_active = Duration::from_secs(9);
    policy.begin_request(Duration::from_secs(181));
    assert!(policy.try_begin_workspace_phase(201, 187, 0).unwrap());
    // The engine releases aligned ranges before preparing the fitting tail.
    for capacity in [176, 187] {
        policy
            .sync_planned_workspace_capacity(Duration::from_secs(181), capacity)
            .unwrap();
        assert_eq!(policy.trial_base, Some(88));
        assert_eq!(policy.trial_capacity, Some(capacity));
        assert_eq!(policy.trial_adverse_active, Duration::from_secs(9));
    }
    for second in 182..=210 {
        assert_eq!(
            tick(&mut policy, second, observation(Pressure::Normal, 0, 0)),
            Action::Hold
        );
    }
    assert!(!policy.workspace_restoration_allowed(Duration::from_secs(210), 0));
    policy.finish_workspace_phase();
    policy.end_request(Duration::from_secs(210), true);
    assert_eq!(policy.earned_target, None);
    assert_eq!(policy.trial_capacity, Some(187));
    assert_eq!(policy.trial_adverse_active, Duration::from_secs(9));
    assert_eq!(policy.completed_clean_active, Duration::ZERO);
    assert_eq!(policy.maximum_capacity, 201);
}

#[test]
fn planned_release_below_trial_base_never_rolls_back_upward() {
    let mut policy = idle_prepared(192);
    assert!(policy.try_begin_workspace_phase(201, 187, 0).unwrap());
    for capacity in [176, 187] {
        policy
            .sync_planned_workspace_capacity(Duration::from_secs(181), capacity)
            .unwrap();
    }
    assert_eq!(policy.trial_base, Some(176));
    assert_eq!(policy.trial_capacity, Some(187));
    policy.finish_workspace_phase();
    for second in 182..=200 {
        tick(
            &mut policy,
            second,
            observation(Pressure::Normal, 0, second),
        );
    }
    assert_eq!(policy.capacity, 176);
    assert_eq!(policy.earned_target, None);
}

#[test]
fn a_discarded_trial_tail_cannot_mask_churn_with_a_zero_byte_rollback() {
    let mut policy = idle_prepared(192);
    assert!(policy.try_begin_workspace_phase(201, 176, 0).unwrap());
    policy
        .sync_planned_workspace_capacity(Duration::from_secs(181), 176)
        .unwrap();
    policy.finish_workspace_phase();
    policy.begin_request(Duration::from_secs(182));
    for second in 182..197 {
        assert_eq!(
            active_tick(&mut policy, second, second * 180_000),
            Action::Hold
        );
    }
    assert_eq!(
        active_tick(&mut policy, 197, 197 * 180_000),
        Action::ChurnRelease(160)
    );
}

#[test]
fn reduced_provisional_extent_still_rolls_back_during_wide_work() {
    let mut policy = idle_prepared(88);
    policy.trial_adverse_active = Duration::from_secs(9);
    policy.begin_request(Duration::from_secs(181));
    assert!(policy.try_begin_workspace_phase(201, 187, 0).unwrap());
    policy
        .sync_planned_workspace_capacity(Duration::from_secs(181), 187)
        .unwrap();
    let mut rolled_back = false;
    for second in 182..=190 {
        let action = policy
            .observe(
                Duration::from_secs(second),
                policy.capacity,
                Boundary::CompletedOperation,
                observation(Pressure::Normal, 0, second),
            )
            .unwrap();
        if let Action::Rollback(next) = action {
            policy.pressure_interrupted_work(Duration::from_secs(second));
            policy.capacity_changed(Duration::from_secs(second), policy.capacity, next);
            rolled_back = true;
        }
    }
    assert!(rolled_back);
    assert_eq!(policy.capacity, 80);
    assert!(!policy.workspace_restoration_allowed(Duration::from_secs(190), 1));
    policy.finish_workspace_phase();
    assert_eq!(policy.capacity, 80);
    assert_eq!(policy.trial_capacity, None);
}

fn observation(pressure: Pressure, compressions: u64, swapouts: u64) -> Observation {
    Observation {
        pressure: Some(pressure),
        counters: Some(VmCounters {
            compressions,
            decompressions: 0,
            swapins: 0,
            swapouts,
            page_bytes: 16_384,
        }),
    }
}

fn tick(policy: &mut Policy, second: u64, observation: Observation) -> Action {
    let now = Duration::from_secs(second);
    let capacity = policy.capacity;
    let action = policy
        .observe(now, capacity, Boundary::Observe, observation)
        .unwrap();
    match action {
        Action::Recover(next) | Action::Release(next) | Action::Rollback(next) => {
            policy.capacity_changed(now, capacity, next);
        }
        _ => {}
    }
    action
}

#[test]
fn healthy_startup_reaches_operating_extent_without_requests() {
    for initial in [84, 87, 88, 107] {
        let mut policy = policy(initial, 201);
        for second in 0..=180 {
            tick(&mut policy, second, observation(Pressure::Normal, 0, 0));
        }
        assert_eq!(policy.capacity, 201, "startup={initial}");
        assert!(policy.request_started.is_none());
        assert_eq!(policy.completed_clean_active, Duration::ZERO);
        assert_eq!(policy.earned_target, None);
        assert_eq!(policy.trial_base, Some(initial));
        assert_eq!(policy.trial_capacity, Some(201));
    }
}

#[test]
fn startup_pressure_does_not_require_a_sacrificial_request_to_recover() {
    let mut policy = policy(107, 201);
    for second in 0..=10 {
        tick(&mut policy, second, observation(Pressure::Warning, 0, 0));
    }
    assert_eq!(policy.capacity, 64);
    for second in 11..=220 {
        tick(&mut policy, second, observation(Pressure::Normal, 0, 0));
    }
    assert_eq!(policy.capacity, 201);
    assert!(policy.request_started.is_none());
}

#[test]
fn idle_preparation_keeps_the_whole_unvalidated_extent_reversible() {
    let mut policy = policy(88, 201);
    for second in 0..=180 {
        tick(&mut policy, second, observation(Pressure::Normal, 0, 0));
    }
    assert_eq!(policy.capacity, 201);
    for second in 181..=200 {
        tick(
            &mut policy,
            second,
            observation(Pressure::Normal, 0, second - 180),
        );
    }
    assert_eq!(policy.capacity, 80);
    assert_eq!(policy.earned_target, None);
}

#[test]
fn only_completed_inference_accepts_idle_prepared_storage() {
    for completed in [false, true] {
        let mut policy = policy(88, 201);
        for second in 0..=180 {
            tick(&mut policy, second, observation(Pressure::Normal, 0, 0));
        }
        policy.begin_request(Duration::from_secs(181));
        for second in 181..=200 {
            tick(&mut policy, second, observation(Pressure::Normal, 0, 0));
        }
        policy.end_request(Duration::from_secs(200), completed);
        assert_eq!(policy.earned_target, completed.then_some(201));
        assert_eq!(policy.trial_base, (!completed).then_some(88));
    }
}

#[test]
fn another_preparation_cannot_erase_adversity_in_the_unvalidated_extent() {
    let mut policy = policy(88, 201);
    for second in 0..=6 {
        tick(&mut policy, second, observation(Pressure::Normal, 0, 0));
    }
    for second in 7..=16 {
        tick(
            &mut policy,
            second,
            observation(Pressure::Normal, 0, second - 6),
        );
    }
    let adverse = policy.trial_adverse_active;
    assert_eq!(adverse, Duration::from_secs(9));
    for second in 17..=34 {
        tick(&mut policy, second, observation(Pressure::Normal, 0, 10));
    }
    assert_eq!(policy.capacity, 112);
    assert_eq!(policy.trial_base, Some(88));
    assert_eq!(policy.trial_adverse_active, adverse);
}

#[test]
fn sustained_swapouts_stop_idle_preparation_and_roll_back() {
    let mut policy = policy(88, 201);
    for second in 0..=6 {
        tick(&mut policy, second, observation(Pressure::Normal, 0, 0));
    }
    assert_eq!(policy.capacity, 96);
    let mut rolled_back = false;
    for second in 7..=30 {
        let action = tick(
            &mut policy,
            second,
            observation(Pressure::Normal, 0, second),
        );
        rolled_back |= matches!(action, Action::Rollback(_));
        assert!(!matches!(action, Action::Recover(_)));
    }
    assert!(rolled_back);
    assert_eq!(policy.capacity, 80);
}

#[test]
fn missing_observations_and_failed_allocation_do_not_become_idle_permission() {
    let mut policy = policy(88, 201);
    for second in 0..=100 {
        assert_eq!(
            tick(
                &mut policy,
                second,
                Observation {
                    pressure: None,
                    counters: None
                }
            ),
            Action::Hold
        );
    }
    for second in 101..=105 {
        tick(&mut policy, second, observation(Pressure::Normal, 0, 0));
    }
    assert_eq!(policy.capacity, 88);
    policy.recovery_failed(Duration::from_secs(105));
    policy.recovery_failed(Duration::from_secs(105));
    policy.recovery_failed(Duration::from_secs(105));
    for second in 106..145 {
        assert_eq!(
            tick(&mut policy, second, observation(Pressure::Normal, 0, 0)),
            Action::Hold
        );
    }
    assert_eq!(
        policy.cached_action(Duration::from_millis(144_500), 88),
        Action::Hold
    );
    assert!(matches!(
        tick(&mut policy, 145, observation(Pressure::Normal, 0, 0)),
        Action::Recover(96)
    ));
}

#[test]
fn request_workspace_freezes_observed_growth_and_critical_pressure_still_stops() {
    let mut policy = policy(88, 201);
    assert!(policy.try_begin_workspace_phase(88, 88, 0).unwrap());
    for second in 0..=100 {
        assert_eq!(
            tick(&mut policy, second, observation(Pressure::Normal, 0, 0)),
            Action::Hold
        );
    }
    assert_eq!(policy.earned_target, None);
    assert_eq!(
        tick(&mut policy, 101, observation(Pressure::Critical, 0, 0)),
        Action::Release(64)
    );
    assert_eq!(
        tick(&mut policy, 102, observation(Pressure::Critical, 0, 0)),
        Action::Stop
    );
    assert_eq!(
        policy.cached_action(Duration::from_millis(102_100), 64),
        Action::Stop
    );
}

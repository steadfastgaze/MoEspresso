//! Request-owned prefill resource and committed-cache regressions.

use super::*;
use crate::adaptive_memory::Controller;
use crate::memory::{CapacityPlan, ExpertCapacityControlBudget, PackageFootprint};
use crate::package::Package;
use crate::tokenizer::ChatOptions;
use serde_json::json;
use std::time::Duration;

const CONTEXT: usize = 131072;
const CACHE_BYTES: u64 = 1 << 30;
const NARROW: usize = 2048;
const WIDE: usize = 4096;

use super::request_resources::PhasedPrefill;

fn narrow_budget(package: &Package, slots: usize) -> Result<ExpertCapacityControlBudget> {
    narrow_budget_with_disk(package, slots, 0)
}

fn narrow_budget_with_disk(
    package: &Package,
    slots: usize,
    disk_io: u64,
) -> Result<ExpertCapacityControlBudget> {
    let footprint = PackageFootprint::from_package(package)?;
    let mut estimate =
        footprint.estimate_fixed_capacity_prefill(CONTEXT, slots, NARROW, 2 << 30)?;
    estimate.prefix_retained_budget_bytes = CACHE_BYTES;
    // Rebuild complete phase sums through the ordinary estimator.
    let estimate = footprint.estimate_ordinary_path_with_prefill_capacity(
        crate::memory::OrdinaryPathInputs {
            context_limit: CONTEXT,
            max_prefill_rows: NARROW,
            candidate_slot_ceiling: slots,
            uncounted_core_reserve_bytes: estimate.uncounted_core_reserve_bytes,
            vec_spare_growth_overlap_reserve_bytes: Some(
                estimate.vec_spare_growth_overlap_reserve_bytes,
            ),
            prefix_retained_budget_bytes: CACHE_BYTES,
            snapshot_capture_restore_overlap_reserve_bytes: Some(disk_io),
            driver_allocator_allowance_bytes: 2 << 30,
        },
        NARROW,
    )?;
    let effective =
        estimate.accounted_component_sum_bytes + footprint.expert_storage_metadata_bytes(slots)?;
    let plan = CapacityPlan {
        slots,
        effective_budget_bytes: effective,
        accounted_peak_bytes: estimate.accounted_component_sum_bytes,
        remaining_budget_bytes: effective - estimate.accounted_component_sum_bytes,
    };
    ExpertCapacityControlBudget::from_admission(&footprint, &estimate, &plan)
}

fn options(cap: usize) -> GenerationOptions {
    GenerationOptions {
        max_tokens: cap,
        max_context: CONTEXT,
        decode_mode: RoutingMode::CachePrior,
        sampling: SamplingOptions {
            temperature: 0.0,
            ..SamplingOptions::default()
        },
        seed: 0,
    }
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn provisional_growth_yields_workspace_and_preserves_generated_cache() -> Result<()> {
    use crate::adaptive_memory::Boundary;
    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    let footprint = PackageFootprint::from_package(&package)?;
    let tokenizer = TextTokenizer::load(&package)?;
    let budget =
        narrow_budget(&package, 128)?.with_prefill_width(&footprint, CONTEXT, NARROW, NARROW)?;
    let mut engine = Engine::load_with_prefill_capacity(&package, 128, CONTEXT, NARROW)?;
    let mut cache = PromptPrefixCache::new(&engine, CACHE_BYTES);
    let chat = ChatOptions {
        thinking: false,
        ..ChatOptions::default()
    };
    let mut turns = vec![json!({"role":"user","content":"Explain bridge inspection briefly."})];
    let prompt = tokenizer.render(&turns, None, &chat)?.tokens;
    let mut reply = String::new();
    generate_with_cache(
        &mut engine,
        &tokenizer,
        &prompt,
        options(8),
        GenerationCaches {
            memory: Some(&mut cache),
            disk: None,
        },
        |text| {
            reply.push_str(text);
            Ok(())
        },
        || false,
    )?;
    turns.push(json!({"role":"assistant","content":reply}));
    turns.push(json!({"role":"user","content":"Why?"}));
    let followup = tokenizer.render(&turns, None, &chat)?.tokens;
    let retained = cache
        .best_match(&engine, &followup)
        .unwrap()
        .matched_tokens();
    ensure!(retained > prompt.len());

    let mut controller = Controller::new(budget)?;
    controller.fix_observation(Duration::ZERO, 4);
    controller.check_boundary(&mut engine, Boundary::Observe)?;
    ensure!(engine.expert_capacity() == 64);
    for second in 1..=100 {
        controller.fix_observation(Duration::from_secs(second), 1);
        controller.check_boundary(&mut engine, Boundary::Observe)?;
    }
    ensure!(engine.expert_capacity() == 128);
    ensure!(controller.test_pending_trial() == Some((64, 128)));
    ensure!(controller.test_capacity_validation() == (None, true));
    let mut phase = PhasedPrefill::new(budget, &footprint, CONTEXT, CACHE_BYTES, 0)?;
    controller.begin_request();
    phase.transition(
        &mut engine,
        RequestPhase::Prefill {
            new_tokens: WIDE + 1,
        },
        &mut controller,
    )?;
    let wide_slots = engine.expert_capacity();
    ensure!(
        engine.prefill_chunk_size() == WIDE
            && wide_slots == phase.wide_slot_ceiling()
            && wide_slots < 128
    );
    ensure!(controller.test_pending_trial() == Some((64, wide_slots)));
    for second in 101..=120 {
        controller.fix_observation(Duration::from_secs(second), 1);
        controller.check_boundary(&mut engine, Boundary::CompletedOperation)?;
    }
    {
        let fault = engine.test_metal().test_fail_allocation_after(0)?;
        ensure!(
            phase
                .transition(&mut engine, RequestPhase::Decode, &mut controller)
                .is_err()
        );
        ensure!(fault.fired() && phase.is_active() && !engine.prefill_workspace_ready());
    }
    controller.fix_observation(Duration::from_secs(121), 1);
    phase.transition(&mut engine, RequestPhase::Prepare, &mut controller)?;
    controller.end_request(true);
    ensure!(!phase.is_active() && engine.prefill_chunk_size() == NARROW);
    ensure!(
        engine.expert_capacity() == wide_slots,
        "repair restored unaccepted slots"
    );
    ensure!(controller.test_capacity_validation() == (None, true));
    ensure!(
        cache
            .best_match(&engine, &followup)
            .unwrap()
            .matched_tokens()
            == retained
    );

    let mut second = 122;
    for (cap, accepted, pending) in [(20, wide_slots, true), (32, 128, false)] {
        controller.fix_observation(Duration::from_secs(second), 1);
        controller.begin_request();
        let resumed = generate_with_cache_owned(
            &mut engine,
            &tokenizer,
            &followup,
            options(cap),
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |_| Ok(()),
            || false,
            |engine, event| {
                second += 1;
                controller.fix_observation(Duration::from_secs(second), 1);
                phase.transition(engine, event, &mut controller)?;
                controller.check_boundary(engine, Boundary::CompletedOperation)?;
                Ok(false)
            },
        )?;
        controller.end_request(true);
        if pending {
            ensure!(resumed.prompt_cache.cached_tokens == retained);
        }
        ensure!(resumed.prompt_cache.cached_tokens >= retained && resumed.prefill_tokens <= 20);
        ensure!(engine.expert_capacity() == 128);
        ensure!(controller.test_capacity_validation() == (Some(accepted), pending));
    }

    controller.begin_request_at(Duration::from_secs(second + 1));
    for offset in 1..=17 {
        controller.check_with_observation(
            &mut engine,
            false,
            Duration::from_secs(second + offset),
            Some(1),
            Some((0, offset)),
        )?;
    }
    ensure!(
        engine.expert_capacity() == 112,
        "accepted extent ignored sustained active swap-outs"
    );
    controller.end_request_at(Duration::from_secs(second + 17), false);
    ensure!(controller.test_capacity_validation() == (Some(128), false));
    let cached = cache
        .best_match(&engine, &followup)
        .unwrap()
        .matched_tokens();
    let resumed = generate_with_cache(
        &mut engine,
        &tokenizer,
        &followup,
        options(2),
        GenerationCaches {
            memory: Some(&mut cache),
            disk: None,
        },
        |_| Ok(()),
        || false,
    )?;
    ensure!(resumed.prompt_cache.cached_tokens == cached && resumed.generated_tokens == 2);
    Ok(())
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn planned_phase_return_preserves_generated_cache_and_repairs_missing_scratch() -> Result<()> {
    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    let footprint = PackageFootprint::from_package(&package)?;
    let tokenizer = TextTokenizer::load(&package)?;
    let budget =
        narrow_budget(&package, 128)?.with_prefill_width(&footprint, CONTEXT, NARROW, NARROW)?;
    let mut engine = Engine::load_with_prefill_capacity(&package, 128, CONTEXT, NARROW)?;
    let mut cache = PromptPrefixCache::new(&engine, CACHE_BYTES);
    let chat = ChatOptions {
        thinking: false,
        ..ChatOptions::default()
    };
    let mut turns = vec![json!({"role":"user","content":"Explain bridge inspection briefly."})];
    let prompt = tokenizer.render(&turns, None, &chat)?.tokens;
    let mut reply = String::new();
    generate_with_cache(
        &mut engine,
        &tokenizer,
        &prompt,
        options(8),
        GenerationCaches {
            memory: Some(&mut cache),
            disk: None,
        },
        |text| {
            reply.push_str(text);
            Ok(())
        },
        || false,
    )?;
    turns.push(json!({"role":"assistant","content":reply}));
    turns.push(json!({"role":"user","content":"Why?"}));
    let followup = tokenizer.render(&turns, None, &chat)?.tokens;
    let retained = cache
        .best_match(&engine, &followup)
        .unwrap()
        .matched_tokens();
    ensure!(retained > prompt.len());
    let mut controller = Controller::new(budget)?;
    controller.fix_observation(Duration::ZERO, 1);
    controller.begin_request();
    controller.check_boundary(
        &mut engine,
        crate::adaptive_memory::Boundary::CompletedOperation,
    )?;
    let mut phase = PhasedPrefill::new(budget, &footprint, CONTEXT, CACHE_BYTES, 0)?;
    phase.transition(
        &mut engine,
        RequestPhase::Prefill {
            new_tokens: WIDE + 1,
        },
        &mut controller,
    )?;
    let wide_slots = engine.expert_capacity();
    ensure!(wide_slots == phase.wide_slot_ceiling() && wide_slots < 128);
    ensure!(
        cache
            .best_match(&engine, &followup)
            .unwrap()
            .matched_tokens()
            == retained
    );
    for second in 1..=20 {
        controller.fix_observation(Duration::from_secs(second), 1);
        ensure!(!controller.check_boundary(
            &mut engine,
            crate::adaptive_memory::Boundary::CompletedOperation
        )?);
        ensure!(
            engine.expert_capacity() == wide_slots,
            "adaptive allocation escaped the wide workspace budget"
        );
    }
    {
        let fault = engine.test_metal().test_fail_allocation_after(0)?;
        ensure!(
            phase
                .transition(&mut engine, RequestPhase::Decode, &mut controller)
                .is_err()
        );
        ensure!(fault.fired() && phase.is_active() && !engine.prefill_workspace_ready());
    }
    let refused = generate_with_cache(
        &mut engine,
        &tokenizer,
        &prompt,
        options(1),
        GenerationCaches {
            memory: Some(&mut cache),
            disk: None,
        },
        |_| Ok(()),
        || false,
    );
    ensure!(
        refused
            .unwrap_err()
            .to_string()
            .contains("workspace needs repair")
    );
    ensure!(
        cache
            .best_match(&engine, &followup)
            .unwrap()
            .matched_tokens()
            == retained
    );
    // Retired-scratch repair can finish just beyond the last sample's interval.
    // Expert return must obtain a current observation rather than discard it.
    controller.fix_observation(Duration::from_millis(21_001), 1);
    phase.transition(&mut engine, RequestPhase::Prepare, &mut controller)?;
    ensure!(
        !phase.is_active()
            && engine.prefill_workspace_ready()
            && engine.prefill_chunk_size() == NARROW
    );
    ensure!(
        engine.expert_capacity() == 128,
        "ordinary experts were not restored after scratch repair: {} slots",
        engine.expert_capacity()
    );
    ensure!(
        controller.test_capacity_validation() == (None, false),
        "temporary smaller work validated the restored ordinary extent"
    );
    let resumed = generate_with_cache_owned(
        &mut engine,
        &tokenizer,
        &followup,
        options(8),
        GenerationCaches {
            memory: Some(&mut cache),
            disk: None,
        },
        |_| Ok(()),
        || false,
        |engine, event| {
            phase.transition(engine, event, &mut controller)?;
            Ok(false)
        },
    )?;
    ensure!(resumed.prompt_cache.cached_tokens == retained && resumed.prefill_tokens <= 20);
    let retained = cache
        .best_match(&engine, &followup)
        .unwrap()
        .matched_tokens();
    let long = tokenizer.render(&[json!({"role":"user","content":"The river crosses the plain. The bridge joins the banks. The town maintains the bridge. ".repeat(230)})], None, &chat)?.tokens;
    let mut cleanup_fault = None;
    let failed = generate_with_cache_owned(
        &mut engine,
        &tokenizer,
        &long,
        options(8),
        GenerationCaches {
            memory: Some(&mut cache),
            disk: None,
        },
        |_| Ok(()),
        || false,
        |engine, event| {
            if matches!(event, RequestPhase::CompletedOperation)
                && engine.test_prefill_workspace_in_progress()
            {
                anyhow::bail!("injected phase owner-boundary failure");
            }
            if matches!(event, RequestPhase::Finished) {
                cleanup_fault = Some(engine.test_metal().test_fail_allocation_after(0)?);
            }
            phase.transition(engine, event, &mut controller)?;
            Ok(false)
        },
    );
    let error = format!("{:#}", failed.unwrap_err());
    ensure!(
        error.contains("injected phase owner-boundary failure")
            && error.contains("injected Metal allocation rejection"),
        "cleanup replaced the primary request failure: {error}"
    );
    ensure!(
        cleanup_fault
            .as_ref()
            .is_some_and(crate::metal::TestFaultGuard::fired)
    );
    drop(cleanup_fault);
    ensure!(phase.is_active() && !engine.prefill_workspace_ready());
    ensure!(
        cache
            .best_match(&engine, &followup)
            .unwrap()
            .matched_tokens()
            == retained
    );
    phase.transition(&mut engine, RequestPhase::Prepare, &mut controller)?;
    ensure!(
        !phase.is_active()
            && engine.prefill_workspace_ready()
            && engine.prefill_chunk_size() == NARROW
    );
    // Start the independent pressure-return case at the ordinary admitted extent.
    engine.recover_expert_capacity(
        128,
        budget.preparation_budget_bytes(engine.expert_capacity(), 128)?,
    )?;
    ensure!(engine.expert_capacity() == 128);
    let mut controller = Controller::new(budget)?;
    controller.fix_observation(Duration::ZERO, 1);
    controller.begin_request();
    controller.check_boundary(
        &mut engine,
        crate::adaptive_memory::Boundary::CompletedOperation,
    )?;
    phase.transition(
        &mut engine,
        RequestPhase::Prefill {
            new_tokens: WIDE + 1,
        },
        &mut controller,
    )?;
    let wide_slots = engine.expert_capacity();
    for second in 1..=20 {
        controller.fix_observation(Duration::from_secs(second), 1);
        controller.check_boundary(
            &mut engine,
            crate::adaptive_memory::Boundary::CompletedOperation,
        )?;
    }
    // Pressure arriving during scratch repair takes precedence over return.
    controller.fix_observation(Duration::from_millis(21_001), 2);
    phase.transition(&mut engine, RequestPhase::Decode, &mut controller)?;
    ensure!(
        !phase.is_active()
            && engine.prefill_workspace_ready()
            && engine.prefill_chunk_size() == NARROW
            && engine.expert_capacity() < wide_slots
    );
    ensure!(
        cache
            .best_match(&engine, &followup)
            .unwrap()
            .matched_tokens()
            == retained
    );
    Ok(())
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn interrupted_wide_phase_keeps_live_disk_anchor_for_narrow_restart() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    struct TestRoot(std::path::PathBuf);
    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let root_path = std::env::temp_dir().join(format!(
        "moespresso-phase-disk-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&root_path)?;
    let root = TestRoot(root_path);
    std::fs::set_permissions(&root.0, std::fs::Permissions::from_mode(0o700))?;
    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    let tokenizer = TextTokenizer::load(&package)?;
    let footprint = PackageFootprint::from_package(&package)?;
    let io_bytes = 64 << 20;
    let budget = narrow_budget_with_disk(&package, 128, io_bytes)?
        .with_prefill_width(&footprint, CONTEXT, NARROW, NARROW)?;
    let limits = crate::engine::StoreLimits {
        mode: crate::engine::DiskCacheMode::Prefix,
        max_entries: 4,
        max_payload_bytes: 3 << 30,
        checkpoint: crate::engine::DiskLimits {
            max_file_bytes: 2 << 30,
            max_io_buffer_bytes: io_bytes,
        },
    };
    let chat = ChatOptions {
        thinking: false,
        ..ChatOptions::default()
    };
    let prompt = tokenizer.render(&[json!({"role":"user","content":"The river crosses the plain. The bridge joins the banks. The town maintains the bridge. ".repeat(230)})], None, &chat)?.tokens;
    ensure!(prompt.len() > WIDE);
    {
        let mut engine = Engine::load_with_prefill_capacity(&package, 128, CONTEXT, NARROW)?;
        let mut disk = GenerationDiskCache::new(
            DiskCheckpointStore::open(&root.0, &engine, &tokenizer, limits)?,
            1024,
            16384,
        )?;
        let mut controller = Controller::new(budget)?;
        let mut phase = PhasedPrefill::new(budget, &footprint, CONTEXT, CACHE_BYTES, io_bytes)?;
        let cancelled = std::cell::Cell::new(false);
        let mut committed_chunks = 0;
        let stats = generate_with_cache_owned(
            &mut engine,
            &tokenizer,
            &prompt,
            options(1),
            GenerationCaches {
                memory: None,
                disk: Some(&mut disk),
            },
            |_| Ok(()),
            || cancelled.get(),
            |engine, event| {
                if matches!(event, RequestPhase::CompletedOperation) {
                    if engine.test_prefill_workspace_in_progress() {
                        if committed_chunks > 0 {
                            cancelled.set(true);
                        }
                    } else {
                        committed_chunks += 1;
                    }
                }
                phase.transition(engine, event, &mut controller)?;
                Ok(false)
            },
        )?;
        ensure!(stats.finish == FinishReason::Cancelled && stats.prompt_consumed == WIDE);
        ensure!(
            stats.prompt_cache.disk_checkpoints_written == 1
                && stats.prompt_cache.last_disk_checkpoint == WIDE,
            "interrupted wider prefill lost its committed live-frontier disk anchor"
        );
        ensure!(engine.prefill_chunk_size() == NARROW && !phase.is_active());
    }
    let mut engine = Engine::load_with_prefill_capacity(&package, 96, CONTEXT, NARROW)?;
    let mut disk = GenerationDiskCache::new(
        DiskCheckpointStore::open(&root.0, &engine, &tokenizer, limits)?,
        1024,
        16384,
    )?;
    let resumed = generate_with_cache(
        &mut engine,
        &tokenizer,
        &prompt,
        options(1),
        GenerationCaches {
            memory: None,
            disk: Some(&mut disk),
        },
        |_| Ok(()),
        || false,
    )?;
    ensure!(
        resumed.prompt_cache.event == "disk_hit"
            && resumed.prompt_cache.cached_tokens == WIDE
            && resumed.prefill_tokens == prompt.len() - WIDE
            && resumed.generated_tokens == 1
    );
    Ok(())
}

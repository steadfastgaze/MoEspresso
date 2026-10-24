use super::{Engine, Layer};
use anyhow::{Context, Result, ensure};
use objc2::rc::autoreleasepool;

/// Completed release across every routed layer. Requested bytes count released
/// backing allocations, not an immediate change in process physical footprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertCapacityRelease {
    pub old_capacity: usize,
    pub new_capacity: usize,
    pub released_requested_bytes: u64,
    pub evicted_sources: usize,
}

/// Bounded capacity growth without changing routed projection arithmetic.
/// Preparation includes appended suffix slabs and a replacement address table
/// alongside the old table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertCapacityRecovery {
    pub old_capacity: usize,
    pub new_capacity: usize,
    pub added_requested_bytes: u64,
    pub preparation_requested_bytes: u64,
}

impl Engine {
    /// Grow bounded expert storage while preserving retained expert bytes.
    ///
    /// The caller supplies additional requested-buffer headroom after reserving
    /// active state, host memory, caches and other promised allocations. This
    /// is not an available-memory estimate or a physical-footprint ceiling.
    /// The complete preparation peak is checked before allocating any chunk.
    /// Existing chunks retain their allocations. New suffix slabs end at the
    /// next 16-slot boundary or the target. Growth refuses a history needing
    /// more than 64 physical segments before allocating storage. The target
    /// remains below 512. New slots start empty and fill on demand.
    ///
    /// Like release, recovery requires completed GPU and I/O users and can
    /// change residency-biased routing. Committed conversation state is independent
    /// of expert storage and remains reusable. Preparation failure drops new
    /// storage without changing any layer.
    pub fn recover_expert_capacity(
        &mut self,
        target_capacity: usize,
        preparation_budget_bytes: u64,
    ) -> Result<ExpertCapacityRecovery> {
        autoreleasepool(|_| {
            self.recover_expert_capacity_inner(target_capacity, preparation_budget_bytes)
        })
    }

    fn recover_expert_capacity_inner(
        &mut self,
        target_capacity: usize,
        preparation_budget_bytes: u64,
    ) -> Result<ExpertCapacityRecovery> {
        let mut report = ExpertCapacityRecovery {
            old_capacity: self.expert_capacity,
            new_capacity: target_capacity,
            added_requested_bytes: 0,
            preparation_requested_bytes: 0,
        };
        if target_capacity == self.expert_capacity {
            return Ok(report);
        }
        ensure!(
            !self.ane_residency_failed && self.ane_residency.is_none(),
            "fixed ANE residency forbids expert capacity changes"
        );
        ensure!(
            target_capacity > self.expert_capacity,
            "expert recovery requires a larger capacity"
        );
        self._expert_row_workspace.check_idle()?;
        for layer in &mut self.layers {
            let mlp = match layer {
                Layer::Gdn(block) => block.mlp_mut(),
                Layer::Qsa(block) => &mut block.mlp,
            };
            mlp.check_expert_quiescent()?;
            let pool = mlp.experts_mut();
            ensure!(
                pool.capacity() == self.expert_capacity,
                "expert layers differ from engine capacity"
            );
            let quote = pool.quote_growth_suffix(target_capacity)?;
            report.added_requested_bytes = report
                .added_requested_bytes
                .checked_add(quote.added_requested_bytes)
                .context("expert recovery byte count overflow")?;
            report.preparation_requested_bytes = report
                .preparation_requested_bytes
                .checked_add(quote.preparation_requested_bytes)
                .context("expert recovery preparation byte count overflow")?;
        }
        ensure!(
            report.preparation_requested_bytes <= preparation_budget_bytes,
            "expert recovery needs {} additional requested bytes, budget is {}",
            report.preparation_requested_bytes,
            preparation_budget_bytes
        );
        let mut plans: Vec<crate::experts::PreparedExpertSuffixGrowth<'_>> = Vec::new();
        plans.try_reserve_exact(self.layers.len())?;
        let mut prepared_bytes = 0u64;
        let mut added_bytes = 0u64;
        for layer in &mut self.layers {
            let pool = match layer {
                Layer::Gdn(block) => block.experts_mut(),
                Layer::Qsa(block) => block.mlp.experts_mut(),
            };
            let plan = pool.prepare_growth_suffix(target_capacity)?;
            let prepared = plan.report();
            prepared_bytes = prepared_bytes
                .checked_add(prepared.preparation_requested_bytes)
                .context("expert prepared byte count overflow")?;
            added_bytes = added_bytes
                .checked_add(prepared.added_requested_bytes)
                .context("expert prepared weight count overflow")?;
            plans.push(plan);
        }
        ensure!(
            prepared_bytes == report.preparation_requested_bytes
                && added_bytes == report.added_requested_bytes,
            "expert recovery preparation differs from quote"
        );
        for plan in plans {
            plan.commit();
        }
        self.expert_capacity = target_capacity;
        Ok(report)
    }

    /// Release complete suffix chunks after a completed engine operation.
    ///
    /// The target must retain a multiple of 16 slots and at least 64 slots.
    /// No expert weights are copied or replaced. Sessions, snapshots and cache
    /// handles remain usable. Cache-Prior may select different experts after
    /// the resident set changes.
    ///
    /// Every layer is checked before any storage is released. Active or
    /// unsubmitted readers cause an error without changing any layer. Engine
    /// prefill and decode calls normally retire their work before returning.
    pub fn shrink_expert_capacity(
        &mut self,
        target_capacity: usize,
    ) -> Result<ExpertCapacityRelease> {
        let mut report = ExpertCapacityRelease {
            old_capacity: self.expert_capacity,
            new_capacity: target_capacity,
            released_requested_bytes: 0,
            evicted_sources: 0,
        };
        if target_capacity == self.expert_capacity {
            return Ok(report);
        }
        ensure!(
            !self.ane_residency_failed && self.ane_residency.is_none(),
            "fixed ANE residency forbids expert capacity changes"
        );
        ensure!(
            target_capacity >= 64
                && target_capacity.is_multiple_of(16)
                && target_capacity < self.expert_capacity,
            "expert release requires a smaller complete chunk prefix"
        );
        self._expert_row_workspace.check_idle()?;
        for layer in &mut self.layers {
            let mlp = match layer {
                Layer::Gdn(block) => block.mlp_mut(),
                Layer::Qsa(block) => &mut block.mlp,
            };
            mlp.check_expert_quiescent()?;
            ensure!(
                mlp.experts_mut().capacity() == self.expert_capacity,
                "expert layers differ from engine capacity"
            );
        }

        // Prepare every fallible check and small host allocation before changing
        // any pool. The plans exclusively borrow their layers until commit.
        let mut plans: Vec<crate::experts::PreparedExpertSuffixRelease<'_>> = Vec::new();
        plans.try_reserve_exact(self.layers.len())?;
        for layer in &mut self.layers {
            let pool = match layer {
                Layer::Gdn(block) => block.experts_mut(),
                Layer::Qsa(block) => block.mlp.experts_mut(),
            };
            let plan = pool.prepare_release_suffix(target_capacity)?;
            let change = plan.report();
            report.released_requested_bytes = report
                .released_requested_bytes
                .checked_add(change.released_requested_bytes)
                .context("expert release byte count overflow")?;
            report.evicted_sources = report
                .evicted_sources
                .checked_add(change.evicted_sources)
                .context("expert release source count overflow")?;
            plans.push(plan);
        }
        for plan in plans {
            plan.commit();
        }
        self.expert_capacity = target_capacity;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{
        DiskCheckpointSlot, DiskCheckpointStore, DiskLimits, PromptPrefixCache, StoreLimits,
    };
    use crate::memory::PackageFootprint;
    use crate::package::Package;
    use crate::routing::RoutingMode;
    use crate::tokenizer::TextTokenizer;
    use std::collections::BTreeMap;
    use std::ffi::OsString;
    use std::fs;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    use std::path::{Path, PathBuf};
    use std::rc::Rc;

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new() -> Result<Self> {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "moespresso-capacity-release-{}-{nonce}",
                std::process::id()
            ));
            fs::DirBuilder::new().mode(0o700).create(&path)?;
            Ok(Self(path))
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn directory_state(path: &Path) -> Result<BTreeMap<OsString, (u64, u64, i64, i64)>> {
        fs::read_dir(path)?
            .map(|entry| {
                let entry = entry?;
                let meta = entry.metadata()?;
                Ok((
                    entry.file_name(),
                    (meta.ino(), meta.len(), meta.mtime(), meta.mtime_nsec()),
                ))
            })
            .collect()
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn ane_startup_rejects_attached_cache_and_previous_session() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let mut engine = Engine::load(&package, 64, 8192)?;
        let cache = PromptPrefixCache::new(&engine, 0);
        assert!(
            engine
                .enable_ane(&package)
                .unwrap_err()
                .to_string()
                .contains("cache attachment")
        );
        assert!(engine.prefer_ane(&package).is_err());
        assert!(!engine.ane_active());
        drop(cache);
        drop(engine.start_session()?);
        assert!(
            engine
                .enable_ane(&package)
                .unwrap_err()
                .to_string()
                .contains("previous sessions")
        );
        assert!(engine.prefer_ane(&package).is_err());
        assert!(!engine.ane_active());
        engine.close_ane()
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE, ANE support and exclusive GPU ownership"]
    fn bounded_ane_storage_resizes_across_workspace_phases_and_restores_state() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let mut engine = Engine::load_with_prefill_capacity(&package, 128, 8192, 4096)?;
        engine.enable_ane(&package)?;
        assert!(engine.ane_active());
        assert!(engine.ane_residency.is_none());
        let scope = engine
            .ane_prefill_cache_scope()
            .expect("an active bounded ANE engine must have a producer scope");
        assert!(
            scope.ends_with(";rows=4096"),
            "the bounded 4096-row workspace must retain a 4096-row ANE producer: {scope}"
        );
        let cache_identity = engine.cache_identity.clone();
        let mut session = engine.start_session()?;
        let tokens: Vec<_> = (0..4096).map(|index| 7 + index % 32).collect();
        engine.prefill_chunk(&tokens, &mut session)?;
        let snapshot = engine.snapshot(&session)?;
        let before = engine.memory_stats()?.live_requested_buffer_bytes;
        let released = engine.shrink_expert_capacity(96)?;
        assert!(released.released_requested_bytes > 0);
        assert_eq!(
            before - engine.memory_stats()?.live_requested_buffer_bytes,
            released.released_requested_bytes
        );
        assert_eq!(
            engine.ane_prefill_cache_scope().as_deref(),
            Some(scope.as_str())
        );
        let grown = engine.recover_expert_capacity(144, u64::MAX)?;
        assert!(grown.added_requested_bytes > released.released_requested_bytes);
        assert!(engine.ane_active());
        assert!(engine.ane_residency.is_none());
        assert!(Rc::ptr_eq(&cache_identity, &engine.cache_identity));
        assert_eq!(
            engine.ane_prefill_cache_scope().as_deref(),
            Some(scope.as_str())
        );
        engine.replace_prefill_without_overlap(2048, u64::MAX)?;
        assert_eq!(engine.prefill_chunk_size(), 2048);
        assert!(engine.ane_active());
        assert_eq!(
            engine.ane_prefill_cache_scope().as_deref(),
            Some(scope.as_str())
        );
        let mut restored = engine.restore(&snapshot)?;
        assert_eq!(restored.frontier()?, tokens.len());
        assert_eq!(restored.consumed_tokens()?, tokens);
        let ordinary = engine.step(7, RoutingMode::Unbiased, &mut session)?;
        let restored_output = engine.step(7, RoutingMode::Unbiased, &mut restored)?;
        assert_eq!(ordinary.logits_bf16, restored_output.logits_bf16);
        assert_eq!(ordinary.frontier, restored_output.frontier);

        // A completed GPU error also poisons an active bounded ANE engine,
        // although it has no persistent expert-residency set to release.
        let fault = engine.metal.test_fail_completed_command_after(0)?;
        assert!(engine.prefill_chunk(&[7, 8, 9], &mut session).is_err());
        assert!(fault.fired());
        drop(fault);
        assert!(!engine.ane_active());
        assert!(engine.ane_residency_failed);
        assert!(!engine.prefill.workspace_in_progress());
        engine._expert_row_workspace.check_idle()?;
        engine.graph.status.reset()?;
        assert!(engine.start_session().is_err());
        assert!(engine.shrink_expert_capacity(128).is_err());
        assert!(engine.recover_expert_capacity(160, u64::MAX).is_err());
        let metal = engine.metal.clone();
        drop((session, restored, snapshot));
        engine.close_ane()?;
        assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn whole_engine_resizing_preserves_sessions_and_cache_handles() -> Result<()> {
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let per_slot =
            PackageFootprint::from_package(&package)?.packed_expert_bytes_per_uniform_slot;
        let mut engine = Engine::load(&package, 178, 2048)?;
        let mut session = engine.start_session()?;
        let tokens: Vec<_> = (0..512).map(|index| 7 + index % 32).collect();
        let prefill = engine.prefill_chunk(&tokens, &mut session)?;
        let old_snapshot = engine.snapshot(&session)?;
        let disk_root = TestRoot::new()?;
        let tokenizer = TextTokenizer::load(&package)?;
        let disk_limits = DiskLimits {
            max_file_bytes: 1024 * 1024 * 1024,
            max_io_buffer_bytes: 1024 * 1024 * 1024,
        };
        let store_limits = StoreLimits {
            mode: crate::engine::DiskCacheMode::Full,
            max_entries: 2,
            max_payload_bytes: 1024 * 1024 * 1024,
            checkpoint: disk_limits,
        };
        let slot_path = disk_root.0.join("slot");
        let store_path = disk_root.0.join("store");
        fs::DirBuilder::new().mode(0o700).create(&slot_path)?;
        fs::DirBuilder::new().mode(0o700).create(&store_path)?;
        let mut old_slot = DiskCheckpointSlot::open(&slot_path, &engine, &tokenizer, disk_limits)?;
        let mut old_store =
            DiskCheckpointStore::open(&store_path, &engine, &tokenizer, store_limits)?;
        assert!(old_slot.save(&engine, &session)?);
        assert!(old_store.save(&engine, &session)?);
        let slot_files = directory_state(&slot_path)?;
        let store_files = directory_state(&store_path)?;
        let index_bytes = fs::read(store_path.join("index.json"))?;
        let mut old_cache = PromptPrefixCache::new(&engine, 1024 * 1024 * 1024);
        assert!(old_cache.offer_prompt_checkpoint(&engine, &session, &tokens)?);
        assert!(old_cache.best_match(&engine, &tokens).is_some());
        let identity = engine.cache_identity.clone();
        let before = engine.memory_stats()?;

        let residents = engine.resident_experts()?;
        let late_buffer = {
            let pool = match engine.layers.last_mut().context("missing final layer")? {
                Layer::Gdn(block) => block.experts_mut(),
                Layer::Qsa(block) => block.mlp.experts_mut(),
            };
            pool.chunk_projection(10, crate::package::Projection::Down)?
                .0
                .stream_bindings()[0]
                .buffer
                .clone()
        };
        let mut late_reader = engine.metal.batch()?;
        late_reader.use_indirect_reads(&[&late_buffer])?;
        assert!(engine.shrink_expert_capacity(160).is_err());
        assert_eq!(engine.expert_capacity(), 178);
        assert_eq!(engine.resident_experts()?, residents);
        assert!(Rc::ptr_eq(&identity, &engine.cache_identity));
        let submitted = late_reader.submit();
        assert!(engine.shrink_expert_capacity(160).is_err());
        assert_eq!(engine.resident_experts()?, residents);
        assert!(Rc::ptr_eq(&identity, &engine.cache_identity));
        submitted.wait()?;
        drop(late_buffer);

        for invalid in [0, 32, 48, 63, 127, 179, 512] {
            assert!(engine.shrink_expert_capacity(invalid).is_err());
            assert_eq!(engine.expert_capacity(), 178);
            assert!(Rc::ptr_eq(&identity, &engine.cache_identity));
            assert_eq!(
                engine.memory_stats()?.live_requested_buffer_bytes,
                before.live_requested_buffer_bytes
            );
        }
        let unchanged = engine.shrink_expert_capacity(178)?;
        assert_eq!(unchanged.released_requested_bytes, 0);
        assert!(Rc::ptr_eq(&identity, &engine.cache_identity));
        assert_eq!(directory_state(&slot_path)?, slot_files);
        assert_eq!(directory_state(&store_path)?, store_files);
        assert_eq!(fs::read(store_path.join("index.json"))?, index_bytes);

        let release = engine.shrink_expert_capacity(160)?;
        assert_eq!(release.old_capacity, 178);
        assert_eq!(release.new_capacity, 160);
        assert_eq!(release.released_requested_bytes, per_slot * 18);
        assert_eq!(engine.expert_capacity(), 160);
        let after = engine.memory_stats()?;
        assert_eq!(
            before.live_requested_buffer_bytes - after.live_requested_buffer_bytes,
            release.released_requested_bytes
        );
        assert_eq!(
            before.lifetime_peak_requested_buffer_bytes,
            after.lifetime_peak_requested_buffer_bytes
        );
        assert_eq!(session.frontier()?, 512);
        assert_eq!(session.last_logits_bf16()?, prefill.logits_bf16);
        assert!(Rc::ptr_eq(&identity, &engine.cache_identity));
        let restored = engine.restore(&old_snapshot)?;
        assert_eq!(restored.frontier()?, session.frontier()?);
        assert_eq!(restored.last_logits_bf16()?, session.last_logits_bf16()?);
        drop(restored);
        assert!(old_cache.best_match(&engine, &tokens).is_some());
        assert!(!old_cache.offer_prompt_checkpoint(&engine, &session, &tokens)?);
        let mut new_cache = PromptPrefixCache::new(&engine, 1024 * 1024 * 1024);
        assert!(new_cache.offer_prompt_checkpoint(&engine, &session, &tokens)?);
        assert!(old_slot.load_match(&engine, &tokens)?.is_some());
        assert!(old_slot.save(&engine, &session)?);
        assert!(old_store.load_match(&engine, &tokens)?.is_some());
        assert!(!old_store.save(&engine, &session)?);
        drop(old_slot);
        drop(old_store);
        let mut fresh_slot =
            DiskCheckpointSlot::open(&slot_path, &engine, &tokenizer, disk_limits)?;
        let mut fresh_store =
            DiskCheckpointStore::open(&store_path, &engine, &tokenizer, store_limits)?;
        assert!(fresh_slot.load_match(&engine, &tokens)?.is_some());
        assert!(fresh_store.load_match(&engine, &tokens)?.is_some());
        let mut fresh_session = engine.start_session()?;
        engine.prefill_chunk(&tokens, &mut fresh_session)?;
        assert!(!new_cache.offer_prompt_checkpoint(&engine, &fresh_session, &tokens)?);
        assert!(new_cache.best_match(&engine, &tokens).is_some());
        assert!(fresh_slot.save(&engine, &fresh_session)?);
        assert!(!fresh_store.save(&engine, &fresh_session)?);
        assert!(fresh_slot.load_match(&engine, &tokens)?.is_some());
        assert!(fresh_store.load_match(&engine, &tokens)?.is_some());
        drop(fresh_session);
        drop(fresh_slot);
        drop(fresh_store);
        drop(old_cache);
        drop(old_snapshot);

        // Compare resumed unbiased arithmetic against a separately loaded fixed
        // capacity engine, not against a second execution of the resize path.
        let mut reference = Engine::load(&package, 160, 2048)?;
        let mut control = reference.start_session()?;
        let expected_prefill = reference.prefill_chunk(&tokens, &mut control)?;
        assert_eq!(prefill.logits_bf16, expected_prefill.logits_bf16);
        for token in [17, 29] {
            let actual = engine.step(token, RoutingMode::Unbiased, &mut session)?;
            let expected = reference.step(token, RoutingMode::Unbiased, &mut control)?;
            assert_eq!(actual.frontier, expected.frontier);
            assert_eq!(actual.logits_bf16, expected.logits_bf16);
        }
        let second = engine.shrink_expert_capacity(64)?;
        assert_eq!(second.released_requested_bytes, per_slot * 96);
        assert_eq!(engine.expert_capacity(), 64);
        assert!(engine.shrink_expert_capacity(32).is_err());
        let suffix = [31, 32, 33, 34];
        let actual = engine.prefill_chunk(&suffix, &mut session)?;
        let expected = reference.prefill_chunk(&suffix, &mut control)?;
        assert_eq!(actual.logits_bf16, expected.logits_bf16);
        let actual = engine.step(35, RoutingMode::Unbiased, &mut session)?;
        let expected = reference.step(35, RoutingMode::Unbiased, &mut control)?;
        assert_eq!(actual.logits_bf16, expected.logits_bf16);

        let fresh = engine.snapshot(&session)?;
        let mut restored = engine.restore(&fresh)?;
        assert!(Rc::ptr_eq(&restored.cache_origin, &engine.cache_identity));
        let actual = engine.step(36, RoutingMode::Unbiased, &mut restored)?;
        let expected = reference.step(36, RoutingMode::Unbiased, &mut control)?;
        assert_eq!(actual.logits_bf16, expected.logits_bf16);

        // A budget refusal or a late allocation failure must leave every layer,
        // mapping and cache namespace intact, including above-startup growth.
        for invalid in [0, 32, 63, 512, 513] {
            assert!(engine.recover_expert_capacity(invalid, u64::MAX).is_err());
            assert_eq!(engine.expert_capacity(), 64);
        }
        let stable_identity = engine.cache_identity.clone();
        assert_eq!(
            engine.recover_expert_capacity(64, 0)?.added_requested_bytes,
            0
        );
        assert!(Rc::ptr_eq(&stable_identity, &engine.cache_identity));
        let residents = engine.resident_experts()?;
        let before_recovery = engine.memory_stats()?.live_requested_buffer_bytes;
        let metadata_overlap = 511 * 18 * 8 * engine.layers.len() as u64;
        let preparation = per_slot * 64 + metadata_overlap;
        assert!(
            engine
                .recover_expert_capacity(128, preparation - 1)
                .is_err()
        );
        assert_eq!(
            engine.memory_stats()?.live_requested_buffer_bytes,
            before_recovery
        );
        for allocation in [0, 2 * engine.layers.len() - 1] {
            let fault = engine.metal.test_fail_allocation_after(allocation)?;
            assert!(engine.recover_expert_capacity(128, preparation).is_err());
            drop(fault);
            assert_eq!(engine.expert_capacity(), 64);
            assert_eq!(engine.resident_experts()?, residents);
            assert!(Rc::ptr_eq(&stable_identity, &engine.cache_identity));
            assert_eq!(
                engine.memory_stats()?.live_requested_buffer_bytes,
                before_recovery
            );
            assert!(engine.restore(&fresh).is_ok());
        }
        for (target, token) in [(128, 37), (178, 38), (192, 39), (223, 40)] {
            let old_capacity = engine.expert_capacity();
            let old_snapshot = engine.snapshot(&restored)?;
            let before = engine.memory_stats()?;
            let additional = per_slot * (target - old_capacity) as u64;
            let preparation = additional + metadata_overlap;
            if old_capacity % 16 != 0 {
                let old_identity = engine.cache_identity.clone();
                let old_residents = engine.resident_experts()?;
                // Fail the final layer after earlier suffix slabs were prepared.
                // None of their candidate address tables may become visible.
                let allocations_per_layer = target.div_ceil(16) - old_capacity / 16 + 1;
                let fault = engine
                    .metal
                    .test_fail_allocation_after(allocations_per_layer * engine.layers.len() - 1)?;
                assert!(engine.recover_expert_capacity(target, preparation).is_err());
                drop(fault);
                assert_eq!(engine.expert_capacity(), old_capacity);
                assert_eq!(engine.resident_experts()?, old_residents);
                assert!(Rc::ptr_eq(&old_identity, &engine.cache_identity));
                assert_eq!(
                    engine.memory_stats()?.live_requested_buffer_bytes,
                    before.live_requested_buffer_bytes
                );
            }
            let recovery = engine.recover_expert_capacity(target, preparation)?;
            assert_eq!(recovery.added_requested_bytes, additional);
            assert_eq!(recovery.preparation_requested_bytes, preparation);
            assert_eq!(engine.expert_capacity(), target);
            assert_eq!(
                engine.memory_stats()?.live_requested_buffer_bytes,
                before.live_requested_buffer_bytes + additional
            );
            assert!(engine.restore(&old_snapshot).is_ok());
            assert!(new_cache.best_match(&engine, &tokens).is_some());
            assert!(Rc::ptr_eq(&restored.cache_origin, &engine.cache_identity));
            let actual = engine.step(token, RoutingMode::Unbiased, &mut restored)?;
            let expected = reference.step(token, RoutingMode::Unbiased, &mut control)?;
            assert_eq!(actual.logits_bf16, expected.logits_bf16);
        }
        let suffix = [41, 42, 43, 44];
        let actual = engine.prefill_chunk(&suffix, &mut restored)?;
        let expected = reference.prefill_chunk(&suffix, &mut control)?;
        assert_eq!(actual.logits_bf16, expected.logits_bf16);
        Ok(())
    }
}

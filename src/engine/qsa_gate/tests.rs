use super::super::{Engine, LayerStateView, PromptPrefixCache, Session};
use crate::package::Package;
use crate::routing::RoutingMode;
use anyhow::Result;
use sha2::{Digest, Sha256};

#[test]
fn full_width_extents_and_dispatches_cover_only_initialized_rows() {
    let rows = super::MAX_PHYSICAL_ROWS;
    let elements = super::gate_elements(rows).unwrap();
    assert_eq!(elements * 4, 192 * 1024 * 1024);
    assert_eq!(
        elements as u64 * 4,
        crate::memory::ane_prefill_memory(rows)
            .unwrap()
            .qsa_attended_bytes
    );
    assert_eq!(rows.div_ceil(64) * 64, rows);
    assert!(elements * 4 < u32::MAX as usize);
    assert_eq!((rows / 64) * 96 * 128, rows * 192);
    for rows in [0, 64, 2048, 4095, 4097, 8193, usize::MAX] {
        assert!(super::gate_elements(rows).is_err());
    }
}

#[test]
fn bounded_width_extents_cover_exactly_4096_rows() {
    let rows = 4096usize;
    let elements = super::gate_elements(rows).unwrap();
    assert_eq!(elements * 4, 96 * 1024 * 1024);
    assert_eq!(
        elements as u64 * 4,
        crate::memory::ane_prefill_memory(rows)
            .unwrap()
            .qsa_attended_bytes
    );
    assert_eq!(rows.div_ceil(64) * 64, rows);
    assert!(elements * 4 < u32::MAX as usize);
    assert_eq!((rows / 64) * 96 * 128, rows * 192);
}

#[test]
fn query_and_gate_rows_partition_original_layout() {
    let mut all = Vec::new();
    for gate in [false, true] {
        for row in 0..6144 {
            let mapped = (row / 256) * 512 + row % 256 + usize::from(gate) * 256;
            assert_eq!(mapped / 512, row / 256);
            assert_eq!(mapped % 512, row % 256 + usize::from(gate) * 256);
            all.push(mapped);
        }
    }
    all.sort_unstable();
    assert_eq!(all, (0..12288).collect::<Vec<_>>());
}

#[test]
fn fp32_deferred_gate_preserves_single_rounding_boundary() {
    let mut differs = false;
    for i in 1..1000 {
        let attended = 1.0 + i as f32 / 10000.;
        let logistic = 1. / (1. + (-1.0f32).exp());
        let direct = half::bf16::from_f32(attended * logistic);
        let staged = f32::from_bits(attended.to_bits());
        assert_eq!(direct, half::bf16::from_f32(staged * logistic));
        differs |=
            direct != half::bf16::from_f32(half::bf16::from_f32(attended).to_f32() * logistic);
    }
    assert!(
        differs,
        "premature BF16 attention rounding must be distinguishable"
    );
}
#[test]
fn fixed_controller_rows_survive_workspace_replacement() {
    for workspace in [512, 2048, 4096, 8192] {
        assert_eq!(
            super::eligible(super::MAX_PHYSICAL_ROWS, workspace).unwrap(),
            workspace >= 4096
        );
        assert_eq!(super::eligible(4096, workspace).unwrap(), workspace == 4096);
    }
    assert!(super::eligible(2048, 4096).is_err());
}

#[test]
fn bounded_qsa_admission_requires_exact_physical_rows() {
    assert!(super::eligible(4096, 4096).unwrap());
    for rows in [0, 1, 2048, 4095, 4097, 6144, 8192, usize::MAX] {
        assert!(!super::eligible(4096, rows).unwrap());
    }
    assert_eq!(super::late_rows(4096), 4096);
}

#[test]
fn padded_qsa_admission_and_late_rows_preserve_the_online_tail() {
    for rows in [4096, 4097, 4104, 4105, 6144, 6928, 8185, 8191, 8192] {
        assert!(super::eligible(8192, rows).unwrap());
        let reconstructed = super::late_rows(rows);
        let online = rows - reconstructed;
        assert!(reconstructed <= rows && reconstructed >= 4096);
        assert!(online <= 8);
        if online != 0 {
            assert_eq!(reconstructed % 64, 0);
            assert_eq!(online, rows % 64);
        } else {
            assert!(rows % 64 == 0 || rows % 64 > 8);
        }
        assert_eq!(reconstructed * 6144 + online * 6144, rows * 6144);
    }
    for rows in [0, 1, 378, 4095, 8193, usize::MAX] {
        assert!(!super::eligible(8192, rows).unwrap());
    }
    assert!(super::eligible(0, 0).is_err());
    assert!(!super::eligible(4096, 6144).unwrap());
}

fn state_digest(session: &Session) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    let ple = session.ple_state()?;
    hash.update(ple.offset.to_le_bytes());
    for token in ple.context {
        hash.update(token.to_le_bytes());
    }
    for word in ple.conv {
        hash.update(word.to_le_bytes());
    }
    session.visit_state(|layer, state| {
        hash.update((layer as u64).to_le_bytes());
        match state {
            LayerStateView::Gdn {
                conv_bf16,
                recurrent_f32,
            } => {
                for bytes in [conv_bf16, recurrent_f32] {
                    hash.update((bytes.len() as u64).to_le_bytes());
                    hash.update(bytes);
                }
            }
            LayerStateView::Qsa { index, kv } => {
                hash.update(index.frontier().to_le_bytes());
                for words in [index.compressed_bf16(), index.pending_bf16()] {
                    hash.update((words.len() as u64).to_le_bytes());
                    for word in words {
                        hash.update(word.to_le_bytes());
                    }
                }
                hash.update((kv.frontier() as u64).to_le_bytes());
                hash.update((kv.context_limit() as u64).to_le_bytes());
                for bytes in [
                    kv.records(),
                    kv.sink_keys(),
                    kv.sink_values(),
                    kv.tail_keys(),
                    kv.tail_values(),
                ] {
                    hash.update((bytes.len() as u64).to_le_bytes());
                    hash.update(bytes);
                }
            }
        }
        Ok(())
    })?;
    Ok(hash.finalize().into())
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE, ANE support and exclusive GPU ownership"]
fn smaller_chunks_keep_exact_gpu_logits_and_committed_state() -> Result<()> {
    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    let prompt: Vec<u32> = (0..4095).map(|i| 61 + i % 32).collect();
    let tail: Vec<u32> = (0..17).map(|i| 93 + i % 16).collect();
    let mut reference = Vec::new();
    for ane in [false, true] {
        let mut engine = Engine::load_with_prefill_capacity(&package, 512, 8192, 8192)?;
        if ane {
            engine.enable_ane(&package)?;
        }
        let mut session = engine.start_session()?;
        let mut actual = Vec::new();
        for tokens in [&prompt[..], &tail[..]] {
            let output = engine.prefill_chunk(tokens, &mut session)?;
            actual.push((output.logits_bf16, state_digest(&session)?));
        }
        let output = engine.step(7, RoutingMode::Unbiased, &mut session)?;
        actual.push((output.logits_bf16, state_digest(&session)?));
        assert_eq!(session.frontier()?, 4113);
        if ane {
            assert_eq!(actual, reference);
            assert_eq!(engine.prefill.test_ane_state(), Some((&[][..], false)));
        } else {
            reference = actual;
        }
        drop(session);
        engine.close_ane()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE, ANE support and exclusive GPU ownership"]
fn replacement_retains_fixed_ane_owner_and_prior_prompt_cache() -> Result<()> {
    retained_owner_cache_case(8192)
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE, ANE support and exclusive GPU ownership"]
fn replacement_retains_4096_ane_owner_and_prior_prompt_cache() -> Result<()> {
    retained_owner_cache_case(2048)
}

fn retained_owner_cache_case(initial_workspace: usize) -> Result<()> {
    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    let physical = crate::prefill_capacity::ane_physical_rows(initial_workspace)?;
    let mut engine = Engine::load_with_prefill_capacity(&package, 512, 10496, initial_workspace)?;
    engine.enable_ane(&package)?;
    let scope = engine.prefill.ane_cache_scope();
    let scope_suffix = format!(";rows={physical}");
    assert!(scope.as_deref().unwrap().ends_with(scope_suffix.as_str()));
    engine.replace_prefill_without_overlap(2048, u64::MAX)?;
    assert_eq!(engine.prefill.ane_cache_scope(), scope);
    let prior: Vec<u32> = (7..39).collect();
    let suffix: Vec<u32> = (39..56).collect();
    let mut cache = PromptPrefixCache::new(&engine, 1 << 30);
    let mut session = engine.start_session()?;
    engine.prefill_chunk(&prior, &mut session)?;
    assert!(cache.offer_prompt_checkpoint(&engine, &session, &prior)?);
    let reference_tail = engine.prefill_chunk(&suffix, &mut session)?.logits_bf16;
    let reference_step = engine
        .step(7, RoutingMode::Unbiased, &mut session)?
        .logits_bf16;
    drop(session);
    let retained = cache.retained_bytes()?;
    engine.replace_prefill_without_overlap(4096, u64::MAX)?;
    assert!(engine.ane_active());
    assert_eq!(engine.prefill.ane_cache_scope(), scope);
    engine.replace_prefill_without_overlap(8192, u64::MAX)?;
    assert_eq!(engine.prefill.ane_cache_scope(), scope);
    let useful: usize = if physical == 4096 { 4096 } else { 4097 };
    let prompt: Vec<u32> = (0..useful).map(|i| 61 + (i % 32) as u32).collect();
    let mut wide = engine.start_session()?;
    let output = engine.prefill_chunk(&prompt, &mut wide)?;
    assert_eq!(output.frontier, useful);
    assert_eq!(wide.frontier()?, useful);
    assert_eq!(engine.prefill.test_projection_trace().len(), 96);
    drop(wide);
    assert_eq!(cache.retained_bytes()?, retained);
    let hit = cache
        .best_match(&engine, &prior)
        .expect("prior prefix retained");
    let mut restored = engine.restore(hit.snapshot())?;
    assert_eq!(restored.consumed_tokens()?, prior);
    assert_eq!(
        engine.prefill_chunk(&suffix, &mut restored)?.logits_bf16,
        reference_tail
    );
    assert_eq!(
        engine
            .step(7, RoutingMode::Unbiased, &mut restored)?
            .logits_bf16,
        reference_step
    );
    drop(restored);
    engine.replace_prefill_without_overlap(2048, u64::MAX)?;
    assert!(!super::eligible(physical, engine.prefill_chunk_size())?);
    assert!(engine.ane_active());
    assert_eq!(engine.prefill.ane_cache_scope(), scope);
    drop(cache);
    engine.close_ane()
}

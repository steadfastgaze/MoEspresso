.DEFAULT_GOAL := help
.PHONY: help build check lint dead-code test test-model test-adaptive test-conversation test-oracles test-quality test-quality-long quality-record quality-record-long require-quality roadtest test-opencode dist-check require-metal
.NOTPARALLEL:

help:
	@printf '%s\n' \
	  'make build          Build the release executable with locked dependencies' \
	  'make check          Run formatting, strict Clippy and regular tests' \
	  'make lint           Check formatting and Clippy on every Cargo target' \
	  'make dead-code      Check compiler warnings on the library and executable' \
	  'make test           Run the regular suite, including synthetic Metal tests' \
	  'make test-model     Run selected real-package storage and recovery checks' \
	  'make test-adaptive  Check capacity changes, logits and cache ownership' \
	  'make test-conversation Check conversation reuse through resizing and interleaving' \
	  'make test-oracles   Compare sigmoid tables with supplied external oracles' \
	  'make test-quality   Compare short whole-model logits and check answers/tools locally' \
	  'make test-quality-long Compare optional 10K-token attention/cache boundary logits' \
	  'make quality-record Record a short baseline from an explicitly selected good build' \
	  'make quality-record-long Record the optional long-context baseline' \
	  'make roadtest       Exercise served memory cache, disk restore and disconnect recovery' \
	  'make test-opencode  Exercise OpenCode read, edit and read-back through the server' \
	  'make dist-check     Audit a source archive and rebuild all targets independently' \
	  'See docs/testing.md for prerequisites and GPU ownership.'

build:
	cargo build --locked --release

check: lint test

lint:
	cargo fmt --all -- --check
	cargo clippy --locked --all-targets -- -D warnings

dead-code:
	RUSTFLAGS="-D warnings" cargo check --locked --lib --bin moespresso

require-metal:
	@test "$$(uname -s)" = Darwin || { printf '%s\n' 'This gate requires a supported macOS Metal host.' >&2; exit 1; }

test: require-metal
	cargo test --locked --all-targets -- --test-threads=1

test-model: require-metal
	@test -n "$$MOESPRESSO_TEST_PACKAGE" || { printf '%s\n' 'MOESPRESSO_TEST_PACKAGE is required; reserve exclusive GPU ownership.' >&2; exit 1; }
	sh scripts/test-ignored.sh lib package::reader_tests::package_iq3_and_iq2_rows_match_allocating_reader
	sh scripts/test-ignored.sh lib experts::gpu::overlap_device_tests::prefetch_eof_preserves_publication_prefix_and_retries
	sh scripts/test-ignored.sh lib experts::gpu::overlap_device_tests::decode_prefetch_failure_preserves_sorted_publication_and_retry
	sh scripts/test-ignored.sh lib experts::gpu::overlap_device_tests::narrow_prefetch_failure_preserves_sorted_publication_and_retry
	sh scripts/test-ignored.sh lib experts::gpu::overlap_device_tests::retained_read_failure_and_workspace_conflict_preserve_pool
	sh scripts/test-ignored.sh lib experts::gpu::overlap_device_tests::package_overlap_preserves_policy_and_scope_cleanup
	sh scripts/test-ignored.sh lib engine::failure_recovery_tests::bounded_handoff_encoding_failure_allows_new_session_on_same_engine
	sh scripts/test-ignored.sh lib-release engine::failure_recovery_tests::windowed_handoff_failure_after_service_allows_same_engine_retry
	sh scripts/test-ignored.sh lib-release engine::failure_recovery_tests::bounded_backing_growth_uses_serial_decode
	sh scripts/test-ignored.sh lib engine::failure_recovery_tests::injected_metal_errors_allow_next_request_on_same_engine
	sh scripts/test-ignored.sh lib-release engine::failure_recovery_tests::explicit_4096_command_failure_preserves_prior_conversation
	sh scripts/test-ignored.sh lib engine::snapshot::disk::tests::wide_prefill_memory_disk_roundtrip_and_cross_width_reuse
	sh scripts/test-ignored.sh lib-release engine::snapshot::disk::tests::explicit_4096_prefill_memory_disk_roundtrip_and_cross_width_reuse

test-adaptive: require-metal
	@test -n "$$MOESPRESSO_TEST_PACKAGE" || { printf '%s\n' 'MOESPRESSO_TEST_PACKAGE is required; reserve exclusive GPU ownership.' >&2; exit 1; }
	sh scripts/test-ignored.sh lib engine::expert_capacity::tests::whole_engine_resizing_preserves_sessions_and_cache_handles
	sh scripts/test-ignored.sh lib generation::tests::controlled_growth_exceeds_startup_and_preserves_prefill_and_decode_logits
	sh scripts/test-ignored.sh lib generation::tests::controlled_shrink_preserves_snapshots_and_unbiased_generation
	sh scripts/test-ignored.sh lib generation::tests::controlled_boundary_failure_releases_session_before_next_generation
	sh scripts/test-ignored.sh lib generation::tests::complete_cache_hit_checks_boundary_before_single_token_publication

test-conversation: require-metal
	@test -n "$$MOESPRESSO_TEST_PACKAGE" || { printf '%s\n' 'MOESPRESSO_TEST_PACKAGE is required; reserve exclusive GPU ownership.' >&2; exit 1; }
	sh scripts/test-ignored.sh lib-release generation::tests::default_gdn_inputs_preserve_resized_generated_conversation
	sh scripts/test-ignored.sh acceptance default_phased_prefill_preserves_interleaved_generated_conversation
	sh scripts/test-ignored.sh lib-release server::conversation_tests::conversation_state_survives_resizing_and_interleaving
	sh scripts/test-ignored.sh lib-release server::conversation_tests::short_generated_conversation_survives_restart_at_another_capacity
	sh scripts/test-ignored.sh lib-release server::conversation_tests::disk_retention_keeps_interleaved_conversations_while_one_chain_advances
	sh scripts/test-ignored.sh lib-release server::conversation_tests::followup_uses_generated_disk_state_when_ram_can_retain_only_the_prompt
	sh scripts/test-ignored.sh lib-release server::conversation_tests::prefix_disk_cache_keeps_shared_anchors_without_turn_writes
	sh scripts/test-ignored.sh lib-release generation::tests::wide_prefill_cancellation_preserves_prior_conversation
	sh scripts/test-ignored.sh lib-release generation::tests::wide_prefill_pressure_release_completes_and_reuses_conversation
	sh scripts/test-ignored.sh lib-release mlp_batch::gpu::device_mapping_tests::wider_grouped_packed_preserves_completed_real_weight_mlp
	sh scripts/test-ignored.sh lib-release generation::phase_tests::planned_phase_return_preserves_generated_cache_and_repairs_missing_scratch
	sh scripts/test-ignored.sh lib-release generation::phase_tests::interrupted_wide_phase_keeps_live_disk_anchor_for_narrow_restart

test-oracles: require-metal
	@test -n "$$MLP_SIGMOID_BF16_ORACLE" || { printf '%s\n' 'MLP_SIGMOID_BF16_ORACLE is required.' >&2; exit 1; }
	@test -n "$$IQK_SIGMOID_ORACLE" || { printf '%s\n' 'IQK_SIGMOID_ORACLE is required.' >&2; exit 1; }
	sh scripts/test-ignored.sh lib mlp::tests::stable_sigmoid_matches_external_bf16_table
	sh scripts/test-ignored.sh lib experts::tests::sigmoid_matches_external_fp16_oracle

require-quality: require-metal
	@test -n "$$MOESPRESSO_TEST_PACKAGE" || { printf '%s\n' 'MOESPRESSO_TEST_PACKAGE is required; reserve exclusive GPU ownership.' >&2; exit 1; }
	@test -n "$$MOESPRESSO_QUALITY_BASELINE" || { printf '%s\n' 'MOESPRESSO_QUALITY_BASELINE is required; keep generated baselines outside the source tree.' >&2; exit 1; }

test-quality: require-quality
	MOESPRESSO_QUALITY_RECORD=0 sh scripts/test-ignored.sh lib-release-live server::quality::short

test-quality-long: require-quality
	MOESPRESSO_QUALITY_RECORD=0 sh scripts/test-ignored.sh lib-release-live server::quality::long

quality-record: require-quality
	MOESPRESSO_QUALITY_RECORD=1 sh scripts/test-ignored.sh lib-release-live server::quality::short

quality-record-long: require-quality
	MOESPRESSO_QUALITY_RECORD=1 sh scripts/test-ignored.sh lib-release-live server::quality::long

roadtest: require-metal
	@test -n "$$MOESPRESSO_TEST_PACKAGE" || { printf '%s\n' 'MOESPRESSO_TEST_PACKAGE is required; reserve exclusive GPU ownership.' >&2; exit 1; }
	sh scripts/test-ignored.sh acceptance cache_roadtest

test-opencode: require-metal
	@test -n "$$MOESPRESSO_TEST_PACKAGE" || { printf '%s\n' 'MOESPRESSO_TEST_PACKAGE is required; reserve exclusive GPU ownership.' >&2; exit 1; }
	sh scripts/test-ignored.sh acceptance opencode_read_edit_readback

dist-check:
	sh scripts/dist-check.sh

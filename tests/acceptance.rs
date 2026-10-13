mod support;

#[test]
#[ignore = "requires Apple Silicon Metal, MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn default_phased_prefill_preserves_interleaved_generated_conversation() -> anyhow::Result<()> {
    support::phase::run()
}

#[test]
#[ignore = "requires Apple Silicon Metal and MOESPRESSO_TEST_PACKAGE"]
fn cache_roadtest() -> anyhow::Result<()> {
    support::cache::run()
}

#[test]
#[ignore = "requires Apple Silicon Metal, MOESPRESSO_TEST_PACKAGE and OpenCode"]
fn opencode_read_edit_readback() -> anyhow::Result<()> {
    support::opencode::run()
}

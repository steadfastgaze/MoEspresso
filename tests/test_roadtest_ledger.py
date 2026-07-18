"""Road-test ledger: counter bookkeeping against the engine's accounting.

These tests replay usage blocks shaped exactly like the serve layer's
responses and check that the ledger predicts the same events, cached-token
values, and frontier writes the engine produces: suffix-only prompt_tokens,
in-memory hits at the previous full-plus-completion length, frontier writes
strictly inside the prefilled span, permanently unwritten frontiers in
generated spans, and disk restores at the longest written frontier after a
restart.
"""

from __future__ import annotations

from moespresso.agentlib.roadtest.ledger import (
    HealthExpectations,
    SessionLedger,
    is_list_prefix,
)


def usage(*, cached, suffix, completion, event, disk_written=0):
    block = {
        "prompt_tokens": suffix,
        "completion_tokens": completion,
        "total_tokens": suffix + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
        "prompt_cache": {"event": event, "entries": 1, "bytes": 10},
    }
    if disk_written:
        block["prompt_cache"]["disk_checkpoints_written"] = disk_written
    return block


def test_first_request_is_a_miss_with_frontier_writes():
    ledger = SessionLedger("a", stride=256)
    check = ledger.observe("t1", usage(
        cached=0, suffix=600, completion=40, event="miss", disk_written=2))
    assert check.findings == ()
    assert check.new_frontiers == (256, 512)
    assert ledger.written_frontiers == {256, 512}


def test_second_request_hits_at_full_plus_completion():
    ledger = SessionLedger("a", stride=256)
    ledger.observe("t1", usage(
        cached=0, suffix=600, completion=40, event="miss", disk_written=2))
    # Stored key is 640 tokens; the next request extends it by 300.
    check = ledger.observe("t2", usage(
        cached=640, suffix=300, completion=20, event="hit", disk_written=1))
    assert check.findings == ()
    # 768 falls inside (640, 940); 640's own decode span left no new frontier.
    assert check.new_frontiers == (768,)


def test_frontier_at_exact_prompt_length_is_not_written():
    ledger = SessionLedger("a", stride=256)
    check = ledger.observe("t1", usage(
        cached=0, suffix=512, completion=10, event="miss", disk_written=1))
    # The last prompt token goes through the decode step, so only 256 lands.
    assert check.findings == ()
    assert check.new_frontiers == (256,)


def test_decode_span_frontiers_stay_unwritten():
    ledger = SessionLedger("a", stride=256)
    ledger.observe("t1", usage(
        cached=0, suffix=200, completion=100, event="miss"))
    # 256 fell inside the generated span (200..300): never written. The next
    # request restores the memory hit at 300 and writes only above it.
    check = ledger.observe("t2", usage(
        cached=300, suffix=300, completion=10, event="hit", disk_written=1))
    assert check.findings == ()
    assert check.new_frontiers == (512,)
    assert 256 not in ledger.written_frontiers


def test_restart_expects_disk_hit_at_longest_written_frontier():
    ledger = SessionLedger("a", stride=256)
    ledger.observe("t1", usage(
        cached=0, suffix=600, completion=40, event="miss", disk_written=2))
    ledger.note_restart()
    assert ledger.expected_event_and_cached() == ("disk_hit", 512)
    check = ledger.observe("t2", usage(
        cached=512, suffix=200, completion=10, event="disk_hit"))
    assert check.findings == ()


def test_restart_restore_backfills_gap_frontiers():
    ledger = SessionLedger("a", stride=256)
    # Turn 1 prefills 200 and generates 100: frontier 256 is a gap.
    ledger.observe("t1", usage(cached=0, suffix=200, completion=100, event="miss"))
    # Turn 2 prefills to 700: writes 512 (and not 256, below the 300 hit).
    ledger.observe("t2", usage(
        cached=300, suffix=400, completion=50, event="hit", disk_written=1))
    ledger.note_restart()
    # Restore lands at 512; re-prefilling (512, 900) writes 768 only; 256
    # stays below the restored prefix forever.
    check = ledger.observe("t3", usage(
        cached=512, suffix=388, completion=10, event="disk_hit", disk_written=1))
    assert check.findings == ()
    assert check.new_frontiers == (768,)
    assert 256 not in ledger.written_frontiers


def test_wrong_event_and_wrong_cached_are_findings():
    ledger = SessionLedger("a", stride=256)
    ledger.observe("t1", usage(
        cached=0, suffix=600, completion=40, event="miss", disk_written=2))
    check = ledger.observe("t2", usage(
        cached=600, suffix=340, completion=10, event="miss", disk_written=1))
    codes = {finding.code for finding in check.findings}
    assert "cache.event" in codes
    assert "cache.cached_tokens" in codes


def test_missing_checkpoint_write_is_a_finding():
    ledger = SessionLedger("a", stride=256)
    check = ledger.observe("t1", usage(
        cached=0, suffix=600, completion=40, event="miss", disk_written=0))
    assert [finding.code for finding in check.findings] == [
        "disk.checkpoints_written"]


def test_shrinking_prompt_is_a_finding():
    ledger = SessionLedger("a", stride=256)
    ledger.observe("t1", usage(
        cached=0, suffix=600, completion=40, event="miss", disk_written=2))
    check = ledger.observe("t2", usage(
        cached=640, suffix=300, completion=20, event="hit", disk_written=1))
    assert check.findings == ()
    shrunk = ledger.observe("t3", usage(
        cached=960, suffix=10, completion=5, event="hit"))
    codes = {finding.code for finding in shrunk.findings}
    assert "cache.prompt_shrank" not in codes  # 970 > 940 grows
    truly_shrunk = ledger.observe("t4", usage(
        cached=200, suffix=10, completion=5, event="hit"))
    codes = {finding.code for finding in truly_shrunk.findings}
    assert "cache.prompt_shrank" in codes
    assert "cache.monotonicity" in codes


def test_interleaved_sessions_do_not_share_state():
    a = SessionLedger("a", stride=256)
    b = SessionLedger("b", stride=256)
    a.observe("a1", usage(cached=0, suffix=600, completion=40, event="miss",
                          disk_written=2))
    b_check = b.observe("b1", usage(cached=0, suffix=300, completion=10,
                                    event="miss", disk_written=1))
    assert b_check.findings == ()
    a_check = a.observe("a2", usage(cached=640, suffix=300, completion=20,
                                    event="hit", disk_written=1))
    assert a_check.findings == ()
    assert a.written_frontiers == {256, 512, 768}
    assert b.written_frontiers == {256}


def test_health_expectations_track_segment_counters():
    health = HealthExpectations(stride=256, budget_bytes=1000)
    ledger = SessionLedger("a", stride=256)
    check = ledger.observe("t1", usage(
        cached=0, suffix=600, completion=40, event="miss", disk_written=2))
    health.on_request(check)
    payload = {"prompt_cache": {"disk": {
        "enabled": True, "stride": 256, "restores": 0, "writes": 2,
        "evictions": 0, "quarantines": 0, "entries": 2,
        "payload_bytes": 123, "budget_bytes": 1000,
    }}}
    assert health.verify("t1", payload, expected_entries=2) == []

    health.note_restart()
    ledger.note_restart()
    restored = ledger.observe("t2", usage(
        cached=512, suffix=200, completion=10, event="disk_hit"))
    health.on_request(restored)
    payload = {"prompt_cache": {"disk": {
        "enabled": True, "stride": 256, "restores": 1, "writes": 0,
        "evictions": 0, "quarantines": 0, "entries": 2,
        "payload_bytes": 123, "budget_bytes": 1000,
    }}}
    assert health.verify("t2", payload, expected_entries=2) == []


def test_health_flags_eviction_quarantine_and_entry_drift():
    health = HealthExpectations(stride=256, budget_bytes=None)
    payload = {"prompt_cache": {"disk": {
        "enabled": True, "stride": 256, "restores": 0, "writes": 0,
        "evictions": 1, "quarantines": 2, "entries": 5,
        "payload_bytes": 0, "budget_bytes": None,
    }}}
    codes = {f.code for f in health.verify("x", payload, expected_entries=3)}
    assert codes == {"health.evictions", "health.quarantines", "health.entries"}


def test_health_flags_disabled_disk():
    health = HealthExpectations(stride=256)
    payload = {"prompt_cache": {"disk": {"enabled": False}}}
    codes = {f.code for f in health.verify("x", payload, expected_entries=0)}
    assert codes == {"health.disk_disabled"}


def test_is_list_prefix():
    assert is_list_prefix([], [1])
    assert is_list_prefix([1, 2], [1, 2])
    assert is_list_prefix([1, 2], [1, 2, 3])
    assert not is_list_prefix([1, 3], [1, 2, 3])
    assert not is_list_prefix([1, 2, 3], [1, 2])

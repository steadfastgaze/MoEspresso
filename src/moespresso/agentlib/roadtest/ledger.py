"""Counter bookkeeping and per-request assertions for the road-test.

The ledger mirrors the engine's cache accounting from the client side, using
only the per-response ``usage`` block, and checks every response against the
value the accounting predicts. The facts it encodes:

- ``usage.prompt_tokens`` counts the prefilled suffix; the full rendered
  prompt length is ``cached_tokens + prompt_tokens``.
- After a served turn, the in-memory prefix store holds the full token
  sequence including the generated tokens, so the next request of an
  append-only session must report a ``hit`` with ``cached_tokens`` equal to
  the previous request's ``cached + prompt + completion``.
- Frontier checkpoints land on stride multiples strictly inside the prefilled
  region: a frontier ``f`` is written when ``cached < f < full`` and no entry
  for the same prefix exists. A frontier equal to the full prompt length is
  proposed after the first decode step has already advanced the cache, so the
  offset gate refuses it; frontiers inside a generated span are below the
  next request's restored prefix and are never written. Both stay permanently
  unwritten holes and the ledger models them as such.
- After a server restart the memory store is empty, so the first request of a
  session must restore from disk (``disk_hit``) with ``cached_tokens`` equal
  to the longest frontier written for that session's prefix chain.

A failed check is returned as a ``Finding``, never raised: the road-test
records defects with full context and keeps driving the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Finding:
    """One assertion failure, with enough context to reproduce it."""

    code: str
    where: str
    message: str


@dataclass(frozen=True)
class RequestCheck:
    """The outcome of checking one response against the ledger's prediction."""

    where: str
    event: str
    cached_tokens: int
    prompt_tokens: int
    completion_tokens: int
    full_tokens: int
    new_frontiers: tuple[int, ...]
    findings: tuple[Finding, ...]


def is_list_prefix(earlier: list, later: list) -> bool:
    """True when ``earlier`` is an exact leading sublist of ``later``."""
    if len(earlier) > len(later):
        return False
    return later[: len(earlier)] == earlier


class SessionLedger:
    """Predicts and checks cache evidence for one append-only session.

    ``disk_enabled=False`` models a server without the disk store: no
    frontier writes are expected and no restore expectations form.
    """

    def __init__(self, name: str, *, stride: int, disk_enabled: bool = True):
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.name = name
        self.stride = int(stride)
        self.disk_enabled = bool(disk_enabled)
        self.written_frontiers: set[int] = set()
        self.requests = 0
        # Expected in-memory prefix on the current server process: the token
        # length of the entry inserted by this session's previous request.
        # None means no entry exists in memory (session start or restart).
        self._memory_key_len: int | None = None
        self._last_cached: int | None = None
        self._last_full: int = 0

    @property
    def frontier_count(self) -> int:
        return len(self.written_frontiers)

    @property
    def max_written_frontier(self) -> int | None:
        return max(self.written_frontiers) if self.written_frontiers else None

    @property
    def last_full_tokens(self) -> int:
        return self._last_full

    def note_restart(self) -> None:
        """Forget in-memory expectations; the disk bookkeeping survives."""
        self._memory_key_len = None
        self._last_cached = None

    def expected_event_and_cached(self) -> tuple[str, int]:
        """The event and cached_tokens the next request must report."""
        if self._memory_key_len is not None:
            return "hit", self._memory_key_len
        if self.written_frontiers:
            return "disk_hit", max(self.written_frontiers)
        return "miss", 0

    def expected_new_frontiers(self, cached: int, full: int) -> tuple[int, ...]:
        """Frontiers this request's prefill writes, given its observed span."""
        if not self.disk_enabled:
            return ()
        first = (cached // self.stride + 1) * self.stride
        out = []
        frontier = first
        while frontier < full:
            if frontier not in self.written_frontiers:
                out.append(frontier)
            frontier += self.stride
        return tuple(out)

    def observe(self, where: str, usage: dict) -> RequestCheck:
        """Check one response's usage block and absorb it into the ledger."""
        findings: list[Finding] = []
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        prompt_cache = usage.get("prompt_cache") or {}
        event = prompt_cache.get("event")
        disk_written = int(prompt_cache.get("disk_checkpoints_written", 0))

        if cached is None or event is None:
            findings.append(Finding(
                "usage.missing_cache_evidence", where,
                f"response usage lacks cache evidence: {usage!r}",
            ))
            cached = 0
            event = "missing"
        cached = int(cached)
        full = cached + prompt_tokens

        expected_event, expected_cached = self.expected_event_and_cached()
        if event != expected_event:
            findings.append(Finding(
                "cache.event", where,
                f"session {self.name}: expected event {expected_event!r} "
                f"(memory_key_len={self._memory_key_len}, "
                f"max_written_frontier={self.max_written_frontier}), "
                f"observed {event!r} with cached_tokens={cached}",
            ))
        if cached != expected_cached:
            findings.append(Finding(
                "cache.cached_tokens", where,
                f"session {self.name}: expected cached_tokens={expected_cached} "
                f"for event {expected_event!r}, observed {cached} "
                f"(prompt_tokens={prompt_tokens})",
            ))
        if self._last_cached is not None and cached <= self._last_cached:
            findings.append(Finding(
                "cache.monotonicity", where,
                f"session {self.name}: cached_tokens {cached} did not grow past "
                f"the previous request's {self._last_cached} on the same server "
                f"process",
            ))
        if full < self._last_full:
            findings.append(Finding(
                "cache.prompt_shrank", where,
                f"session {self.name}: full prompt length {full} is shorter than "
                f"the previous request's {self._last_full}; the session is no "
                f"longer append-only",
            ))

        new_frontiers = self.expected_new_frontiers(cached, full)
        if disk_written != len(new_frontiers):
            findings.append(Finding(
                "disk.checkpoints_written", where,
                f"session {self.name}: expected {len(new_frontiers)} frontier "
                f"write(s) {list(new_frontiers)} in span ({cached}, {full}), "
                f"response reported disk_checkpoints_written={disk_written}",
            ))

        self.written_frontiers.update(new_frontiers)
        self._memory_key_len = full + completion_tokens
        self._last_cached = cached
        self._last_full = full
        self.requests += 1

        return RequestCheck(
            where=where,
            event=event,
            cached_tokens=cached,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            full_tokens=full,
            new_frontiers=new_frontiers,
            findings=tuple(findings),
        )


@dataclass
class HealthExpectations:
    """Cross-checks ``/health`` disk counters against per-response evidence.

    The disk store's counters reset when a server process starts, while the
    index entries persist on disk, so restore and write expectations are per
    server segment and the entry count is checked against the total frontier
    bookkeeping across all sessions. Baseline offsets support attaching to an
    already-running server whose counters are not zero: expectations then
    ride on top of the attach-time snapshot.
    """

    stride: int
    budget_bytes: int | None = None
    disk_enabled: bool = True
    segment_restores: int = 0
    segment_writes: int = 0
    baseline_restores: int = 0
    baseline_writes: int = 0
    baseline_entries: int = 0
    payload_bytes_high_water: int = field(default=0)

    def attach_baseline(self, health: dict) -> None:
        """Adopt an already-running server's disk counters as the baseline."""
        disk = (health.get("prompt_cache") or {}).get("disk") or {}
        self.disk_enabled = disk.get("enabled") is True
        if not self.disk_enabled:
            return
        self.stride = int(disk.get("stride") or self.stride)
        self.budget_bytes = disk.get("budget_bytes")
        self.baseline_restores = int(disk.get("restores") or 0)
        self.baseline_writes = int(disk.get("writes") or 0)
        self.baseline_entries = int(disk.get("entries") or 0)

    def note_restart(self) -> None:
        self.segment_restores = 0
        self.segment_writes = 0
        self.baseline_restores = 0
        self.baseline_writes = 0

    def on_request(self, check: RequestCheck) -> None:
        if check.event == "disk_hit":
            self.segment_restores += 1
        self.segment_writes += len(check.new_frontiers)

    def verify(self, where: str, health: dict, *, expected_entries: int) -> list[Finding]:
        findings: list[Finding] = []
        disk = (health.get("prompt_cache") or {}).get("disk") or {}
        if not self.disk_enabled:
            if disk.get("enabled") is True:
                findings.append(Finding(
                    "health.disk_unexpectedly_enabled", where,
                    "health reports the disk store on for a run that "
                    "expected it off",
                ))
            return findings
        if disk.get("enabled") is not True:
            findings.append(Finding(
                "health.disk_disabled", where,
                f"health reports the disk store off: {disk!r}",
            ))
            return findings
        checks = {
            "stride": (disk.get("stride"), self.stride),
            "restores": (disk.get("restores"),
                         self.baseline_restores + self.segment_restores),
            "writes": (disk.get("writes"),
                       self.baseline_writes + self.segment_writes),
            "evictions": (disk.get("evictions"), 0),
            "quarantines": (disk.get("quarantines"), 0),
            "entries": (disk.get("entries"),
                        self.baseline_entries + expected_entries),
        }
        if self.budget_bytes is not None:
            checks["budget_bytes"] = (disk.get("budget_bytes"), self.budget_bytes)
        for key, (observed, expected) in checks.items():
            if observed != expected:
                findings.append(Finding(
                    f"health.{key}", where,
                    f"health disk counter {key}: expected {expected}, "
                    f"observed {observed}",
                ))
        payload_bytes = int(disk.get("payload_bytes") or 0)
        if payload_bytes < self.payload_bytes_high_water:
            findings.append(Finding(
                "health.payload_bytes_shrank", where,
                f"health disk payload_bytes {payload_bytes} dropped below the "
                f"run high-water {self.payload_bytes_high_water} with no "
                f"eviction reported",
            ))
        self.payload_bytes_high_water = max(self.payload_bytes_high_water, payload_bytes)
        return findings

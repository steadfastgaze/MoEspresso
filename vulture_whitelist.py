"""Vulture whitelist: names that look unused but are load-bearing.

`make lint` runs vulture at confidence 100 over `src` and `tests` with this
file appended, so a name listed here is treated as used. Every entry needs a
comment stating the mechanism that consumes it (a protocol contract, an
external caller, a monkeypatch seam). Remove the entry when the consumer
goes away; an entry without a living consumer is dead code with extra steps.
"""

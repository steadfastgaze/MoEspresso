# Adaptive bounded expert memory

Serving uses automatic memory admission and adapts expert capacity when the
initial pool contains 64 through 511 slots. A fully resident 512-slot pool and
automatically admitted pools of 10 through 63 slots remain static.
`--no-adaptive-memory` disables capacity changes for a static comparison or
power-sensitive run. Generate is a one-shot path and does not adapt capacity.

`--max-memory-gb GIB` caps the automatic accounted component budget. It keeps
the same host and Metal safety reserves as uncapped automatic admission, and it
bounds both the admitted startup plan and the adaptive operating ceiling. It is
not a process RSS limit or a reservation from the operating system. The
advanced `--expert-capacity` diagnostic path bypasses automatic admission,
disables serving caches, and does not adapt.

## Policy

The inference owner samples Darwin pressure and VM counters at most once per
second, at completed inference boundaries and while idle. No background worker
changes engine allocations.

- Warning pressure releases the next smaller 16-slot-aligned prefix, at most
  once every five seconds, retaining at least 64 slots.
- Critical pressure releases directly to 64 slots. If the next sampled
  observation remains critical at that floor, generation stops or a new
  request is rejected with HTTP 503. Already selected but unreported work is
  not published after a generation callback fails.
- Startup capacity remains provisional until successful serving supplies
  15 seconds of healthy sampled active intervals. Successful requests can
  accumulate this credit. The highest capacity accepted through this work
  becomes the earned target.
- Restoration advances toward the retained earned target, including while idle.
  It requires fresh healthy observations, five seconds of recent quiet and
  five seconds since the last capacity change. Each increase is at most
  16 slots. A final gap within that allowance returns directly to the earned
  target, bounded by the operating ceiling. Larger gaps advance to the next
  16-slot boundary. Physical suffix
  ranges are prepared together before publication.
- Growth beyond the earned target starts after five-second quiet and
  capacity-settling intervals. Each preparation adds at most 16 slots. An
  unaccepted extent can grow again after 15 seconds of healthy observations.
  During inference it must also accumulate 15 seconds of healthy active work;
  partial credit from completed requests pauses further preparation until that
  extent qualifies. Idle preparation does not populate experts or earn an
  accepted target. The entire unvalidated extent keeps its rollback base.
  Growth stops at the fixed operating ceiling.
- An unaccepted extent has a 15-second adverse-observation budget. Swap-out
  activity or compression above the threshold accumulates during serving and
  idle observations. Exhausting the budget rolls back to the previous aligned
  base. Further preparation does not erase earlier adverse intervals.
- Accepted storage can also be released after sustained active VM churn:
  swap-outs, or compression and decompression both at 16 MiB/s or more. Busy
  intervals accumulate toward 15 seconds; quiet intervals subtract time.
  One-way compression alone pauses growth without evicting accepted storage.
  Idle observations do not charge this active-churn budget.
- A pressure episode preserves the earned target and rejects an unaccepted
  trial. Pressure interrupting a pending trial or recent increase delays the
  next attempt by 10 seconds; repeated failed attempts increase the delay
  through 20, 40 and 80 seconds to a maximum of 120 seconds. Sustained
  active-churn releases also delay recovery. Accepting a candidate resets it.
- Swap-out activity or compression at 16 MiB/s or more pauses allocation and
  active credit for the affected interval. Missing observations, counter reset
  or a sampling gap over five seconds breaks recent eligibility. Earlier
  healthy credit and the earned target survive these observations.
- Active time counts sampled intervals between completed operations within a
  request. It includes intervening output and checkpoint work. Idle time, first
  active samples and gaps between requests contribute no credit. Cancellation
  or failure discards that request's credit while preserving earlier
  successful work. Growth can retain a smaller extent's completed validation
  within the request, but accepts it only when the request succeeds. The new
  extent starts its own validation. Pressure or a planned reduction discards
  incompatible in-request credit.

Missing counters and resets add neither healthy nor adverse time. Long sample
gaps cannot earn validation or charge provisional adversity. Valid active-churn
rates across a long operation contribute at most five seconds, so a slow
operation cannot hide sustained churn. Quiet recovery can resume without a new
request after pressure subsides and the backoff expires.

## Operating budget and capacity

Startup admission uses the smallest of installed RAM minus 5 GiB, observed
free plus inactive memory minus 3 GiB, Metal's recommended working set minus
1 GiB, and a positive reported GPU wired limit minus 1 GiB when available.
An explicit `--max-memory-gb` can lower that budget. The component planner
includes a 2 GiB process allowance and selects the largest complete estimate
across loading, engine construction and serving. Insufficient or unavailable
required observations fail before weight loading. These reserves do not
guarantee safety against later external allocations or memory pressure.

ANE admission reserves one physical program selected from the initial GPU
workspace. An 8192-row workspace selects an 8192-row program; smaller workspaces
select a 4096-row program. Their retained payloads are 690 MiB plus four bytes
and 390 MiB plus four bytes respectively, covering host staging, Metal staging,
IOSurfaces and QSA attended scratch. The selected reservation remains charged
during expert resizing and workspace changes. Native program and compiler
overhead use the separate 2 GiB process allowance; the estimate is not an RSS
guarantee.

Fully resident ANE admission first tries the complete 8192-row plan, then a
4096-row plan if the wider program and workspace do not fit. The wider plan
must retain all 512 experts. Bounded adaptive serving uses the 4096-row program
with 2048-row ordinary scratch
and admits a 4096-row request phase only when its complete plan can retain at
least 64 expert slots. Fixed-capacity execution keeps its requested experts and
requires a fitting 4096-row workspace. If ANE cannot be admitted, startup warns
and selects GPU execution, whose ordinary preference is 2048 rows with a
512-row minimum. Full expert residency takes priority over a wider workspace.
Startup reports both capacities. Cache checkpoints include numerical producer
identity and remain reusable across workspace widths within that producer.

Starting capacity uses automatic admission, including its live-memory
observation. The adaptive operating ceiling is a separate fixed budget. It
retains the physical RAM, Metal working-set and optional wired-limit ceilings
and their reserves, but does not freeze the initial free-plus-inactive
observation into the operating ceiling. An explicit `--max-memory-gb` also
bounds that ceiling. These are application policies, not OS memory reservations
or guarantees of available physical RAM.

Growth reserves the admitted maximum context, workspaces, retained prompt
cache, snapshot overlap and process allowance. It checks the complete
preparation peak, including the replacement address table alongside its
previous owner. Adaptive growth never enters the separate 512-slot fully
resident path. Healthy observations permit attempts within this budget.

Each bounded pool retains an address table sized for 511 slots. Expert weight
storage is allocated only for current capacity, in physical ranges ending at or
before the next 16-slot boundary. Growth appends missing slots and retains
existing allocations, including a partial final range. For example, capacity
145 ends with ranges 128..144 and 144..145. Growth to 160 appends 145..160
without copying or evicting the retained tail. Packed prefill uses these
physical ranges; decode uses the per-slot address table.

All layers prepare successfully before any layer publishes new storage.
Preparation failure releases only candidates. New slots are empty until demand
loads experts. Each layer permits at most 64 physical ranges, with descriptor
capacity reserved before serving. Repeated smaller requests are refused before
allocation if they would exceed this bound.

## Request workspace phases

Adaptive serving admitted with 2048 rows and at least 4096 context tokens uses
request workspace phases by default. The phase owner quotes a complete
4096-row plan from the serving context, cache allowances and package footprint.
It widens after prefix
restoration only when more than 2048 new tokens remain, then returns ordinary
scratch before the first token is delivered.

If the wider plan cannot retain the adaptive expert floor within the complete
operating budget, startup keeps ordinary execution. Allocation failure also
falls back after ordinary scratch is repaired. Static serving, full residency,
narrower admitted workspaces and smaller contexts do not select this policy.

The controller's ordinary budget, startup capacity and operating ceiling do
not change. While wider scratch is present, adaptive allocation is frozen;
pressure release, trial rollback and critical-floor stopping continue. A
pending trial can yield expert storage to wider scratch while retaining its
adverse-observation budget and a rollback base no higher than the reduced
extent. Its discarded tail is not automatically restored. Work at a temporarily
smaller extent cannot validate restored slots. Previously earned capacity and
fitting-trial credit survive an unchanged extent.

Planned resizing does not count as pressure. Returning experts requires fresh
quiet observations and ordinary settling, and never reinstates a
pressure-released or rejected trial increment. Failed expert restoration can
retain a smaller valid pool. Failed scratch restoration prevents further
generation until the request owner repairs it. Cleanup failures are attached
to the original request error rather than replacing it.

## Cache behavior

Conversation state is independent of physical expert storage. Growth, pressure
release and trial rollback preserve committed sessions and cache entries.
Warm requests can grow within the operating budget, and cache writes continue
after resizing at completed operations. Idle growth prepares empty storage;
expert rows are loaded only when inference needs them.
A capacity change can alter later residency-biased expert selection and the
resulting answer; that does not make the conversation's committed state unusable.

Cache-Prior keeps factor 2 and two protected routes by default on the bounded
path. `--cache-routing off` selects unbiased decode routing for quality checks
or comparisons. Prefill uses original routing in both modes.

## Logs

Normal logs retain request token counts and timings, cache lookup events,
checkpoint progress, capacity changes and failures. Startup reports the
initial capacity, operating ceiling and settling and validation intervals.

## Limits

Response latency includes the current GPU or I/O operation and any blocking
output callback. Polling can miss short-lived pressure events. The controller
cannot guarantee survival against fast external allocations, GPU execution
failure or OS termination, and it does not reduce the maximum-context contract
or change quantized arithmetic.

Unit tests inject pressure levels, counter deltas and monotonic times. Real
package tests exercise the controller with synthetic observations at completed
generation boundaries. Those tests do not create system-wide pressure. Normal
host serving checks establish integration; desktop coexistence and useful
recovery require constrained-host evidence. Successful serving at an earned
capacity does not establish suitability under every workload.

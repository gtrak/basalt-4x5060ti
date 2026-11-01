# TASK: park on EVICTION, and QUEUE rather than decline

Two changes, both in the same seam, behind one flag. Read the existing code first.

## Why

Parking was retired from the default because it cost 35% of decode (recorded in
`basalt-ctl` campaign entry 16):

    4 live + park, budget 24 -> SSD 20.6 GiB, decode 55.7
    2 live + park, budget 26 -> SSD 10.3 GiB, decode 77.5
    2 live, no park, budget 33 -> SSD  3.3 GiB, decode 85.3

But that measurement was of **park-on-every-slot-handoff**. The write is what hurt: a
full session image (~3.09 GiB int8, **6.0 GiB fp16** at 262144) charged against the
expert copy on EVERY conversation switch, which forced the SSD spill.

The owner's decision: **park only when a sequence is EVICTED**, and **queue** a request
that cannot be admitted rather than declining it.

Under fp16 the write is 6.0 GiB/session, so the distinction is worth twice what it was.

## Where the code is

- `park_current` - `src/program/generate.cpp:5601`, the lambda that saves the session image.
- Its ONLY call site - `generate.cpp:8871`:
  `if ((!from_live || incoming || slot_source >= 0) && !park_current(...))`
  i.e. it fires on admission/handoff, not on eviction.
- The decline path just above it - `generate.cpp:8843-8869`: on `kv_short` it logs
  `"prompt of %lld tokens refused"`, sets `admit_slot = -1`, and `continue`s. That
  `continue` returns to `for (;;)` at `generate.cpp:8230`, which calls `next_line(line)` -
  the request is DROPPED and the client must retry.
- `ConversationCache::make_room` / `evictions()` - `conversation_cache.hpp:250,317`.

## Change 1: park only on eviction

Today a conversation that stays resident is written out on every switch. It should keep
living in VRAM and be written **only when `make_room` actually evicts it**.

Gate before building: **instrument the write volume per conversation switch and per
eviction.** The number that decides this is bytes written per request, before and after.

Measured context for the gate (so you are not guessing):
- **prefix reuse works**: a live agentic session reused **145,855 of 147,135** tokens. So
  conversations DO stay hot, which is what makes "never rewrite a resident conversation"
  worth anything.
- a session image at 262144 fp16 is **6.0 GiB** (3.09 GiB int8). Writing that on every
  handoff is the cost this change removes.
- the current serving config is `--parallel 1 --kv fp16`, so there is ONE slot and a
  handoff happens on every new request that is not the live conversation.

If today's writes are already small, report that and stop rather than build.

## Change 2: queue instead of decline

When no slot can be admitted, the request must WAIT for one rather than being dropped:

- On the `kv_short` decline path, enqueue the pending request (its slot id, tokens,
  position, max_new) instead of `continue`-ing straight back to `next_line`.
- When a slot frees (a request completes, a conversation is evicted), admit the queued
  request before blocking on the next line.
- The queue must be **bounded** and must preserve FIFO. A queue that can grow without
  limit turns a memory shortage into an OOM, which is worse than a decline.
- A queued request must still be able to be CANCELLED (the BSTOP path exists already).
- Log one line per queue event: queued (position, why), admitted (how long it waited).

**Do not make the queue unbounded, and do not let a queued request hold a slot's lock.**

## Requirements

1. Both changes behind one flag, DEFAULT OFF, so they can be A/B'd: `--serve-park-on-evict`
   (or a clearer name - your call).
2. With the flag off, behavior must be bit-identical. State how you verified that.
3. A test for the queue: enqueue/admit/FIFO/bounded/cancel. This is pure host logic and
   needs no GPU - it should be testable without one.
4. Log lines through `basalt/log.hpp` with real levels, terse facts (see AGENTS.md).

## Rules (non-negotiable)

- Work only in `/home/gary/opt/basalt-pp4` (branch `pp4`).
- **Do NOT start, stop, restart or signal any `basalt` or `serve.server` process, and do
  not bind port 8000.** The serving instance is other agents' inference endpoint. Report
  the boot command and the supervisor runs it.
- **Do not run GPU tests.** Build, and run `ctest -LE gpu -E avx512`; report real counts
  and confirm `BUILD_TESTING` is ON.
- Commit small, `area: what it does`, no amend, no force-push. Do not push anywhere else.
- Platform code stays behind the seams in `src/platform/`.

## Measure, in this order

1. **Write volume** per conversation switch, today vs the flag - this decides whether
   change 1 is worth anything.
2. Queue behaviour under a slot shortage: does a queued request eventually get served,
   and how long does it wait?

Report both, with the boot commands. Then stop; the supervisor measures.

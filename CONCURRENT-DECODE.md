# Concurrent batch decode: where the 22% goes, and what to do

## The measurement

Two concurrent greedy requests against the shipped config:

| | solo | 2 concurrent |
|---|---|---|
| per-request | ~110 t/s | 34-57 t/s |
| **aggregate** | **~110 t/s** | **89-94 t/s** (-15 to -22%) |
| window | 28.46 ms | 27.24 - 35.22 ms |
| tokens/window | 2.63 | 1.54 - 2.23 |
| **acceptance** | 73-76% | **75% (40 of 53)** - unchanged |
| CPU experts/layer-window | 0.96 | 1.00 - 2.62 |
| expert hit rate | 97.7% | 93.1 - 96.7% |

## Three things it is NOT

1. **Not wall time.** The window barely grows (28.46 -> 27.24 ms in the best sample).
   Concurrency is nearly free in time.
2. **Not acceptance.** 75% concurrent against 73-76% solo. The drafter is not confused
   by a second conversation.
3. **Not cache miss cost, primarily.** Hit rate dips 97.7 -> 93.1% but the CPU expert
   term only reaches 2.62 ms of a 35 ms window (7%).

## What it IS

**Tokens per window collapse relative to the rows offered.** The clearest window: a
concurrent window with `T = 5.00` offered rows produced `tok = 1.00` - four rows offered,
one kept. Solo the same config does `T = 3.38 -> 2.63`.

So the batched window is paying for rows that do not pay back. With two slots the
window has to serve two chains, and the rows are allocated across them; a slot whose
chain is short still costs its rows in the verified window.

## Candidate fixes, in order of expected value

**C1. Per-slot window depth from that slot's own acceptance (cheapest, most direct).**
The window length T is chosen once for the whole batch (`draft_keep` over one chain),
but each slot has its own acceptance history (`sl.acc`, already tracked at
generate.cpp:7869). Give each slot its own T from its own `acc`, so a slot with a
healthy chain gets its depth and a slot with a short chain does not spend rows.
Gate: log T per slot vs tokens kept per slot under 2 concurrent requests; if the
short-chain slot is consistently over-allocated, this is the fix.

**C2. Rank the expert cache for the aggregate traffic, not one slot's.**
The cache never evicts (`expert_cache.hpp:126-131`: "it never evicts, because an
eviction policy has to be chosen from measurements") and is filled from a profile
recorded on ONE request stream. Two conversations route to overlapping-but-different
experts (86.9-96.7% hit says heavy overlap, not disjoint). Sum the profile across slots
so the resident set is ranked for the mix. Modest gain - the miss cost is only 7% of the
window - but it compounds with C1.

**C3. Do not batch the slots' drafts into one window when their chains disagree.**
If slot A would offer 4 rows and slot B only 1, running one window of 5 rows is worse
than two windows of 4 and 1 in sequence when the chain is not the binding cost. This is
a scheduling policy, measurable with C1's instrumentation before any build.

**Explicitly deprioritized** - `MR = kVerifyMaxT = 8` is not the constraint: with 2
slots `nfull = min(2, 8-2) = 2`, so both slots do receive full drafts. Checked in the
source, not assumed.

## The gate for all three

Instrument T and tokens-kept **per slot** (not per window) under 2 concurrent requests.
That single line decides which of C1/C3 is real, and it costs one boot.

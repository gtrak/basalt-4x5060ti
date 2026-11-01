# Concurrent batch decode: measurements, and two instrument traps that stalled the analysis

## What is solid

Two concurrent greedy requests against the shipped config (measured 3 times):

| | solo | 2 concurrent |
|---|---|---|
| per request | ~110 t/s | **32-60 t/s** |
| **aggregate** | **~110 t/s** | **89-97 t/s** |
| acceptance | 73-76% | **75% (40 of 53)** |
| expert hit rate | 97.7% | 93.1-96.7% |

`--exp slot_hist=1` (added, batch path, per slot-segment):

    offered 2.69 rows, kept 2.22, wasted 0.47 (worst 5)

**The slots are NOT over-allocated against each other** - each keeps 83% of what it
offers, and 0.47 wasted rows per segment is small. That refutes the leading hypothesis
(per-slot window depth from per-slot acceptance) before any build.

## The unresolved discrepancy - resolve this before building anything

Two instruments disagree by a factor of 2 on the same concurrent run:

- `slot_hist`: 2.22 tokens kept **per slot-segment**, x2 segments = **4.44 per window**
- `decode timing`: **2.23 tokens/window**

If the timing line counts all slots (as `produced_n` / `dec_windows` suggests) they
should agree. They do not.

Where the confusion comes from, verified in source:
- `produced_n` is declared at 10144 and used through 10970 in the **solo** request's
  accounting;
- the batch path tracks progress per slot as **`sl.produced`** (line 7872);
- there is only ONE `decode timing` emit (line 10774), in the batch path.

So a single "tokens/window" field is being fed by an accumulator whose meaning differs
between the two paths. **Do not quote concurrent tokens/window from that line until this
is fixed** - and do not size a fix from it.

## What the numbers bound anyway

Solo emits 2.63 tokens/window at 28.46 ms = 92 t/s. Concurrent aggregate is 89-97 t/s.
**Batching two requests does not increase aggregate throughput at all** - it is a wash
that splits one stream into two slower ones.

That is consistent with everything else measured tonight: decode is bound by arithmetic
intensity at small M, and batching two requests at M~2.7 each does not raise M for the
GEMM (the rows are still split across two independent expert sets). Batching helps only
when it grows the batch DIMENSION of the same expert operations, and two unrelated
conversations route to mostly-different experts.

## Candidate fixes, priced by that

**C1. Cache ranked for the aggregate mix** (not one slot's profile). Hit rate falls
97.7 -> ~95% under concurrency, and the CPU-expert term is only ~7% of the window, so
the ceiling here is small - a few percent at most.

**C2. Do not batch mismatched chains** - scheduling, not a speedup of the batch itself.

**C3. The one with real headroom is still the batch dimension:** raise tokens/window for
a SINGLE stream, which every lever tested tonight failed to do (draft depth, draft
floor, tree draft, two windows in flight). The 0.9%-of-compute-peak figure says the
headroom is real but it is not reachable from the scheduler.

## Instruments that survive

- `--exp slot_hist=1` - per-slot rows offered vs kept (default off)
- and the discrepancy above is a real defect in `decode timing` worth fixing first,
  because every window number quoted tonight came from it.

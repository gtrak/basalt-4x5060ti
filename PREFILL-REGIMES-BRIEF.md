# TASK: multiple prefill regimes in one read

## The request

Large uncached prompts should use the big chunk. Small uncached portions should use a
small chunk. Today one ceiling governs the whole read.

## What exists (verified in source - read it yourself before changing anything)

The engine already recomputes the chunk per request, and already has a TWO-tier notion:

- `serve_loans::request_chunk(tokens, max_chunk)` - `src/program/serve/loans.cpp:31`.
  Rounds the request's uncached token count up to the 256 grid and takes `min(max_chunk, that)`.
  So the chunk already shrinks to fit a SHORT request.
- `prompt_one_card_chunks(tokens, cap, head_ok, head_chunk)` - `src/program/multi_gpu_plan.cpp:321`.
  Returns `{want, head}`: a HEAD chunk (`kHeadChunk = 16384`) for the first read when the
  prompt is at least 2x the head chunk, and a `want` for the rest.
- `prompt_chunk_list(tokens, chunk, head)` - `src/program/multi_gpu_plan.cpp:338`.
  Already builds a LIST of chunks: the head, then uniform `chunk`s. It is already used at
  `generate.cpp:4746`, `:9338`, `:9495`.
- The per-request resolve is `generate.cpp:11043-11062` (the `--serve` request path), where
  `o.prefill_chunk` is finally set.

So the machinery for a LIST of chunk sizes exists and is used; what does NOT exist is a
policy that picks the list from the SHAPE of the uncached work.

## What to build

A regime policy: given the request's uncached token count (and whether a prompt-cache
prefix was reused, so only the tail is uncached), choose a SEQUENCE of chunk sizes rather
than one ceiling.

Concretely, the shape to implement:

- **large uncached prefix** -> the big chunk (the current `auto` ceiling, capped by what
  the expert caches can lend)
- **small uncached remainder** (a cache-hit request that appends a little, or the tail of
  a long read) -> a small chunk, so the fixed per-chunk cost is not paid on a handful of
  tokens

Expose it behind a flag, DEFAULT OFF, so it can be A/B'd:
`--prefill-regimes LO,HI` (or two thresholds - your call on the cleanest spelling), where
the request picks the small chunk below some uncached size and the big one above it.

Requirements:

1. The chunk list must still respect what the expert caches can LEND (the lending walk in
   `generate.cpp:4380-4390` and `loans.cpp`). A small chunk should be trivially satisfiable;
   a big chunk must still fail over to smaller ones exactly as today.
2. Nothing may change when the flag is off - bit-identical behavior.
3. Log one line per request saying which regimes were used and why (uncached tokens,
   chosen chunk list). That line is the measurement.

## Measure before and after, on THESE prompts

Prefill is what this optimizes. The measured baselines on this box for fp16 at
`--parallel 1`, chunk 12288, cut 8,20,34:

| prompt | prefill |
|---|---|
| 38,229 fresh | 2,435 t/s |
| 122,308 fresh | 3,656 t/s |
| 122k prefill, int8 for reference | 4,972 t/s |

Also relevant: short-request TTFT. A 92-token request currently reports `pp=92/87c` -
that is a prompt-cache hit and it is fast. The interesting case is a request with a long
cached prefix and a **small uncached tail**: measure that one specifically, since it is
the case the regime policy is really for.

## Rules (non-negotiable)

- Work only in `/home/gary/opt/basalt-pp4` (branch `pp4`).
- **Do not start, stop, restart or signal any `basalt` or `serve.server` process, and do
  not bind port 8000.** The serving instance belongs to the supervisor and is other
  agents' inference endpoint. If you need a GPU measurement, report the exact boot
  command and the supervisor will run it and hand you the numbers.
- **Do not run GPU tests.** Build, and run `ctest -LE gpu -E avx512` only; report real
  test counts and confirm `BUILD_TESTING` is ON (a suite configured off passes vacuously).
- Commit small, `area: what it does`, no amend, no force-push.
- Every new arithmetic path needs a test. A flag that changes numerics needs a quality note.

## Report

What the policy decides for each of: 122k fresh, 38k fresh, long-cached-prefix + small
tail. The exact A/B boot commands. Test counts. Then stop - the supervisor measures.

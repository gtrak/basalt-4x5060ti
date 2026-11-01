# --prefill-regimes: measured

`--prefill-regimes LO,HI` (worker commit `ec391e6e` and the four around it, default OFF).
`LO` = the small chunk, `HI` = the big chunk and the boundary. A read of `U` uncached
tokens: `U >= HI` -> big regime (unchanged from today), `U < HI` -> small regime, chunks of
at most `min(LO, cap)`, never a head chunk.

The chunk sizes are CEILINGS, not fixed sizes - `prompt_one_card_chunks` divides the read
across the fewest chunks that fit and rounds to the 256 grid. So `LO=8192` on an 11.6k tail
gave `[5888, 5695]`, and on a 653-token tail gave `[653]` (one chunk).

## The A/B (fp16, --parallel 1, cut 8,20,34, 400 W)

Same workload both arms: a ~110k-prompt warm read to fill the cache, then three requests
reusing a 98,304-token prefix with an 11,588-token uncached tail.

| arm | regime | chunks | TTFT | prefill |
|---|---|---|---|---|
| **ON** `8192,12288` | **small** | `[5888, 5695]` | **8.9 - 9.0 s** | **1,295 - 1,296 t/s** |
| OFF | big (today) | `[12032, ...]` | **9.7 - 9.8 s** | 1,189 t/s |

**+9% prefill rate, -0.8 s TTFT on the tail.** Three samples each, no overlap between the
arms, so it is outside this box's noise.

## Where the policy correctly does nothing

A 122k or a 24k uncached read is above `HI`, so both arms take the big regime and the
plans are byte-identical (`[12288 x9, 11716]` and `[12032, 11962]` respectively, 13.2 s
both ways). That is the intended behavior: the feature only speaks for small reads.

## Real measured tails the policy handled

From a live agentic session (`prompt regimes:` lines in the engine log):

    658 uncached (137223 reused of 137881), small regime, chunks of 768  [653]
    588 uncached (144907 reused of 145495), small regime, chunks of 768  [583]
   1280 uncached (145855 reused of 147135), small regime, chunks of 1280 [1275]

Also resolves an earlier confusion of mine: **prefix reuse works** (145,855 of 147,135
tokens reused). My earlier "0 cached" tail test failed because the two payloads differed
at the START, not because the cache was broken.

## Known, from the worker, not yet fixed

- the `prompt regimes:` line is ALSO emitted by `split_price`'s pricing probe, which logs a
  hypothetical small read (4,090 tokens) as if it were a request decision. Real decisions
  carry the reused count; the probe's shows 0 reused on a big read.
- with `--short-read 512` a tail of <= 512 tokens is read through the decode windows and
  logs no regime line.
- a small-regime chunk < 4096 makes `split_price` decline, so a small tail reads on CUDA0
  alone. Same as any small part today, but it is part of what `LO` changes.
- the quality gate for the ON arm is OPEN (chunking moves K/V rounding, so greedy near-ties
  can differ). `docs/precision/features.md` says so.

## Verdict

Shipped as a flag, DEFAULT OFF, worth **~9% on the cached-tail prefill** that agentic
sessions actually produce. Not worth turning on until the quality gate is run, since
chunking changes the reduction boundaries.

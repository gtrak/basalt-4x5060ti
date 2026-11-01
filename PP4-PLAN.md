# pp4: a clean branch for the pipeline-parallel engine

Branched from `fork/main` (`306f81e7`, == production main) at the owner's direction:
start a clean PP4 branch, cherry-pick only what the tp2pp2 campaign proved useful,
then optimize.

## Why PP4 and not tp2pp2

Both measured on the same binary family, same prompt, same box:

| | PP4 | tp2pp2 |
|---|---|---|
| prefill, 122k fresh | **4,232 t/s** | 2,780 t/s |
| prefill, 62k fresh | 3,471-3,545 t/s | - |
| decode | **87-95 t/s** | ~90 t/s |
| decode window | **26.8-28.5 ms** | 30.1 ms |
| GPU-reach wait | **5.5-5.9 ms** | 10.6 ms |
| per-layer host | 1.2-1.6 ms | 2.8 ms |
| **expert PCIe** | **0.0%** | 55.8% routed |
| expert cache hit | 96-97% | 90% |

PP4 wins both axes. Its decode has no peer handshake and no expert wire at all: the
four cards hold the experts resident (48.9 GiB) and each stage is one card.

## What came over from tp2pp2, and what did not

Cherry-picked (topology-agnostic):

- `3944bc4a` verify: the decode window's phases, op by op, in the prefill marker mould
- `7c542dd9` verify: the decode phase markers read device stamps, not graph-captured events
- `6ce8b733` generate: the decode profile gains the gap (what the wall clock sees between windows)
- `038a137b` generate: the per-window token lines behind `--exp batch_lines`
- `5cf9facd` spec: tail re-seeding for the suffix drafter
- `635a2c1d` program: the drafter reseed behind `--exp draft_reseed`
- `f7c67efb` mtp: the next window's store swap beside the run, behind `--exp mtp_preswap`
- `30a88702` tests: the pool's expert row must not depend on the worker count

Deliberately NOT taken: the peer-pair machinery (partial-merge combine, the staging
ring, the RAM band, the band pin, the share clamp, the attention slice, the
shared-pool cadence probe). All of it exists to move work between the two cards of a
stage, which PP4 does not have. The per-op prefill timing (`9c1b4006`) was left out
too: it inserts its markers inside `PeerPrefill`'s code, which this branch lacks.

## A retraction carried over, so it is not relearned

TP2PP2-NOTES (39) claimed the decode window's 10.59 ms "GPU-reach wait" was PCIe
store-visibility latency. It is not - `ms_wait` is "how long the GPU took to reach
layer k", i.e. the GPU computing, seen from the host end. Three fixes were tried
against it (prefetch every predicted expert, spin in place, stage interleave) and all
were null, which is what the correct reading predicts.

## The decode model that survives

The window is serial by data dependency: GPU attention+router -> rings the host ->
host computes the layer's CPU experts -> publishes -> GPU runs the experts and the
next layer's attention. The handshake is not removable with a flag.

Where the headroom actually is:

1. **tokens per window** - the window cost is nearly flat in T, so raising dl from
   ~2.6 toward 5 is close to a doubling. Acceptance sets it.
2. **two windows in flight** - priced at ~1.8x on tp2pp2 and structurally impossible
   THERE (its stages are dependent within a window). PP4's four stages are also
   dependent within a window, so this needs two different WINDOWS, which is what
   `pipe_want`'s `batch <= 0` excludes.

## Measured on this branch (live)

- decode 92.7 t/s at acc 74%, window 26.83 ms, pcie 0.0%, hit 96.6%
- prefill 3,471 t/s at 62k fresh
- no expert PCIe at all

## Per-card utilization under decode (PP4, tp2pp2 binary)

card0 35.4% / 47.8 W, card1 41.9% / 40.3 W, card2 53.3% / 45.2 W, card3 72.2% / 58.0 W.

Rising monotonically: each stage waits on its successor, and stage 3 carries the
output head on top of its own 12 layers. This is the shape to attack.

## The split cut: what moves decode (measured)

The first lever, and the biggest one found so far. The cut points were reachable
from nothing - the search always cut evenly - so the last stage carried its layers
plus the output head while every earlier stage waited on it.

**Decode t/s by cut (same prompt, same binary; two runs each):**

| cut | stages | t/s | per-card util / watt |
|---|---|---|---|
| 12,24,36 (even) | 12/12/12/12 | 92.5 | 35/42/53/72% - 48/40/45/58 W |
| 8,20,36 | 8/12/16/12 | 97.1 / 100.5 | - |
| 8,20,40 | 8/12/20/8 | 77.1 / 80.5 | c2 73%/56 W saturated |
| **8,22,34** | **8/14/12/14** | **111.7 / 106.1, then 105.6 / 105.2** | 22/39/57/87% - 50/51/54/71 W |
| 8,20,32 | 8/12/12/16 | 109.2 / 104.8 | c3 88%/73 W |
| 6,20,34 | 6/14/14/14 | 102.1 / 96.8 | - |
| 8,24,34 | 8/16/10/14 | 99.3 / 93.6 | - |
| 8,18,34 | 8/10/16/14 | 100.6 / 96.4 | - |

**8,22,34 is the shipped cut: ~+15% decode over the even cut.**

### Why it works: expert cache coverage, not layer count

The driver is not how many layers a card runs but how many experts it can hold.
A stage's cache is what is left of its VRAM after its dense weights, so MORE
layers means FEWER experts resident, and a missed expert streams from RAM.

| cut | card | layers | cache slots / needed | coverage |
|---|---|---|---|---|
| 8,20,36 | CUDA2 | 16 | 3485 / 8192 | **43%** |
| 8,20,40 | CUDA2 | 20 | 2998 / 10240 | **29%** |
| 8,22,34 | CUDA1 | 14 | 3896 / 7168 | 54% |
| 8,22,34 | CUDA2 | 12 | 3974 / 6144 | **65%** |
| 8,22,34 | CUDA3 | 14 | 3319 / 7168 | 46% |

CUDA2 at 43% coverage measured 97-101 t/s; at 29% it collapsed to 77-80; at 65%
it reached 106-112. **Coverage is the lever, and it is bought by taking layers
AWAY from a card, not giving it more.**

### Wattage as the compute proxy

An idle stage waits on its successor and draws ~42 W; a working one 52-73 W. At
the shipped cut the four cards draw 50/51/54/71 W - no card pinned at idle and
none saturated. The bad cuts show the signature plainly: 8,20,40 put c2 at
73%/56 W (saturated) while c0 sat at 17%/42 W (waiting).

### The instrument that does not reach this path

`--exp decode_profile=1` emits its per-stage line only in the single-slot solo
window (`if (stages.size() > 0)` inside the `S == 1` branch). A `--parallel 2`
boot never takes that path, so on PP4 the per-stage phases are not available from
it; `basalt decode timing` (the aggregate) and the per-card wattage are.

## Prefill: the chunk size is already right

The split recipe pins `--prefill 12288` from an older sweep. Re-measured on this
branch at the shipped cut (8,22,34):

| chunk | 62,186-token fresh | 122,288-token fresh |
|---|---|---|
| **12,288** | **3,224 t/s** | **4,232 t/s** |
| 16,384 | 2,512 t/s | 3,892 t/s |

**12,288 wins on both prompt sizes**, so the recipe was already at the optimum and
chunk size is not a lever here. (A 16k boot's second 122k request was a prompt-cache
hit, not a prefill - read `c` in `pp=NNN/NNNc` before quoting a rate.)

## State of the branch

- decode **102.7 t/s** at a live request (acc 74%), window ~26-28 ms
- prefill **3,224 t/s** at 62k, **4,232 t/s** at 122k, chunk 12,288
- expert PCIe **0.0%**, cache hit 96-97%
- shipped cut `8,22,34` (8/14/12/14) via `--layer-split`, forced with
  `--layer-split-forced` (`BASALT_FORCE_SPLIT` no longer exists)

Both axes are now ahead of tp2pp2 (2,780 prefill / ~90 decode) and of PP4's own
starting point tonight (92.5 decode).

## Flag sweep: three nulls, one crash, and the noise bound that makes them null

Each run: full 1,800-token decode window twice, then a 62k prefill. Same boot, same cut.

| run | decode t/s | prefill t/s | total W |
|---|---|---|---|
| base96 (`--adapt-swaps 96`) | 105.5 / 105.2 | 3,232 | 54.3 |
| `--exp mtp_preswap=1` | 105.5 / 105.2 | 3,225 | 52.6 |
| `--exp draft_reseed=1` | 101.5 / 108.9 | 3,225 | 53.0 |
| **base96 again** | **111.7 / 106.1** | 3,243 | 53.7 |
| `--exp host_ev=1` | **engine exits 1** | - | - |

**A baseline compared against itself spans 105.2-111.7 t/s, about 3%.** Every flag's
apparent effect is smaller than that, so all three are null - and `draft_reseed`'s
101.5/108.9 straddles both ends of the baseline's own range, which is what noise
looks like when you sample it twice.

`--exp host_ev=1` is not null but BROKEN: it enables a decode event buffer
(`decode_events().enable(4000)`, verify.cpp:1837) and the engine exits 1 with
`copy_i32x3_from_mapped` failing during a long prompt read. A pre-existing defect in
a diagnostic flag; not taken.

**Method lesson, which cost time tonight:** a single 2-run sample cannot separate a
3% flag effect from 3% machine noise. Any future flag A/B on this box needs either
more runs or a paired design; a lone "105.5 vs 105.2" is not evidence.

## Where the remaining headroom is

Priced, not yet taken:

1. **Expert cache coverage is the decode lever** (measured: 29% coverage -> 77 t/s,
   65% -> 106-112). The exhaustive four-stage search now optimizes for it at boot.
2. **`--peer-reserve-mib` / resident budget** could widen the caches directly: the
   stages drew 9-10.5 GiB of cache each with 37 GiB of RAM budget, and the box has
   room to raise it. This trades host RAM for coverage and is the cheapest untested
   knob.
3. **Two windows in flight** (~1.8x ceiling on tp2pp2, needs `batch <= 0`).

## The auto search's own best is ~9% WORSE than the hand cut (measured, and unexplained by its model)

The exhaustive four-stage search now runs (15,180 cuts through `predict()`), and it
selects K=12,26,37 - the same cut the greedy heuristic chose, so the model's optimum
was already being found. Measured, that cut is worse than a hand cut:

| cut | stages | decode t/s | window |
|---|---|---|---|
| auto (search's best) | 12/14/11/11 | **93.3 - 97.0** | 31.16 ms |
| hand | 8/14/12/14 | **105.2 - 111.7** | ~26-28 ms |

So `predict()` does not capture what decides this. The search believes its cut holds
99.2% of the routed mass; the achieved caches are:

| cut | stage caches held/needed | coverage |
|---|---|---|
| auto 12,26,37 | 3896/7168, 4008/5632, 3762/5632 | 54%, 71%, 67% |
| hand 8,22,34 | 3896/7168, 3974/6144, 3319/7168 | 54%, 65%, 46% |

**The auto cut has BETTER coverage at every stage and still measures worse.** That
kills the simple "coverage is the lever" reading this page carried earlier: coverage
is necessary but not sufficient, and the hand cut wins with strictly less of it.

What the two cuts differ in, and what the model does not price:
- **CUDA0 (12 layers auto vs 8 hand) holds NO expert cache at all** - it carries the
  main session and KV - so its layers are pure serial latency on the head of the
  chain, and `predict()` charges it the same `layer_ms` as a cache-carrying stage.
- Wattage puts the wall at the LAST stage in both cuts (c3 71%/54 W against c0-c2 at
  37-52%), so the term that binds is the tail of the chain, not the mean.

**Status: the hand cut 8,22,34 is shipped; the search's model is known-incomplete.**
Fixing `predict()` needs a per-stage cost that reflects serial position and the
CUDA0-no-cache asymmetry. Not attempted yet - it needs its own measurement to
validate, and the wrong fix would silently pick worse cuts than the heuristic it
replaced.

## THE PER-LAYER INSTRUMENTATION, AND WHAT IT ACTUALLY SAYS

`--profile` (a real flag, not an `--exp`) prints the decode window's per-layer GPU
phases: 31 named terms per layer, split GDN vs QSA, and it printed ONLY stage 1 until
this branch - under a layer split that is a quarter of the window. Now one line per
stage, labelled with its card (`generate.cpp`).

### The objective is the SUM, not the spread

The four stages are **sequential**: stage 2 cannot start layer 8's work until stage 1
hands over. So the window is the SUM of the stage times, and any placement rule that
minimizes IMBALANCE (max/min) is optimizing the wrong thing. Measured, same prompt:

| cut | c0 ms | c1 ms | c2 ms | c3 ms | **SUM** | t/s |
|---|---|---|---|---|---|---|
| **8,22,34** | 3.89 | 6.19 | 5.31 | 7.90 | **23.29** | **102-105** |
| 12,25,38 | 6.51 | 5.95 | 5.99 | 6.24 | 24.69 | 87-89 |
| 4,20,34 | 2.17 | 7.77 | 6.56 | 8.25 | 24.75 | 90-101 |

The "balanced" cut (12,25,38) has the flattest spread of the three and is the
WORST of them. The winner minimizes the sum.

### Why CUDA0 is special and both models miss it

CUDA0 carries the main session and KV and **holds no expert cache at all** - every
expert its layers route is fetched. Its ms/layer is 0.486 at 8 layers and 0.54 at 12,
so it degrades faster than the middle stages (0.442). Taking layers OFF c0 is not free
either: at 4 layers the work moves to c1 (6.19 -> 7.77) and the sum grows. 8 is the
measured optimum, and it was found by sweeping, not by a model.

### The per-layer cost, at the shipped cut

`VRAM hits` - the expert blob read inside a layer - is the dominant stamped term in
every stage: 1.24 ms for a GDN layer against 0.41 for the shared expert. `waitA`,
`waitB`, `waitCPU`, `PCIe grp` are all **0.00**: there is no inter-card stall to win
at decode, and the 55.8%-PCIe figure from tp2pp2 has no counterpart here.

### Both automatic models are wrong, for the same reason

`predict()` (the boot search, exhaustive now) minimizes a predicted WINDOW that is a
sum of per-stage `layer_ms` - but its `layer_ms` is uniform across stages, so it
cannot see CUDA0's no-cache penalty or the tail's head cost. The measured-balance
model fails identically: it minimizes spread where the objective is the sum.
**Neither is a safe replacement for the sweep until the per-stage cost is measured
per stage, not assumed uniform.**

## Can CUDA0 have an expert cache? Yes - it always had one, and it is not the lever

The question is worth the answer because the code reads as if it did not: the staging
loop is `for (auto& stp : stages)` and `stages` holds only the LATER stages, so CUDA0
never appears in those log lines. But the main card's cache is opened separately and
does exist:

    basalt generate: expert cache 2813 slots, 7.24 GiB of VRAM; policy is ...

**Measured: enlarging it does not help.** `--vram-reserve-mib` (default 700) is the
knob. At 320 MiB every cache grew ~20% (CUDA0 2813 -> 3447 slots, 7.24 -> 8.88 GiB;
total VRAM 37.3 -> 38.8 GiB, still 0.0 GiB on SSD), and decode was unchanged:

| reserve | decode t/s (all samples) | mean |
|---|---|---|
| 700 MiB (default) | 108.5, 110.2, 105.5, 105.2, 111.7, 106.1, 102.7, 103.9 | **106.7** |
| 320 MiB | 97.4, 112.4, 103.4, 107.2, 106.2, 109.9, 106.3 | **106.1** |

The hit rate was already 97-98%, so there was no miss traffic to remove - the caches
were sufficient, and a bigger one only serves more hits from VRAM (which is why the
`VRAM hits` term went UP, 1.24 -> 1.61 ms: it counts hits served, not the cost of a
miss).

### The one thing this DID expose, and why it is not patched

The boot search prices every stage at `layer_ms = 0.33 * (84.0*2.617)/(SMs*GHz)`, and
the four cards report the same SMs and clock within 3%, so all four stages are charged
the same per-layer cost (logged: 0.76/0.75/0.77/0.77). The measurement says
0.486/0.442/0.442/0.564. Trying to correct it with per-stage factors made the search
pick 12/30/47, which **broke the zero-spill contract** (8.1 GiB on SSD against the
shipped cut's 0.0) - a worse outcome than the model it replaced. Reverted.

So the honest position stands: the cut is a swept constant (8,22,34), the search's
model is known-wrong, and a correct replacement needs the per-stage cost *measured at
boot* rather than assumed - not a hand-tuned multiplier.

## The expert GEMM is the biggest term: what bounds it

`VRAM hits` - 37% of the stamped GPU time, the largest single term - is stamp 19->20,
which brackets `grouped(...)`: the NVFP4 grouped expert GEMM. It is not a VRAM read
despite the name.

**An earlier draft of this page claimed it runs at 2,181 GB/s effective and is
therefore cache-resident. That arithmetic was wrong**: `VRAM hits` counts ENTRIES
(token x expert pairs), not blobs. A window's 48.3 entries are served by ~16.8 unique
experts (48.3 / 2.87 tokens), so the bytes touched are 44 MiB per layer-window, not
128 MiB.

Corrected: a 14-layer stage reads ~623 MiB of expert weights in ~8.8 ms of GEMM, which
is **71 GB/s, or 16% of the 448 GB/s HBM peak**. That is low enough that bandwidth is
not the binding constraint - but the margin is 6x, not 20x, and part of that traffic is
HBM (16.8 blobs = 44 MiB against a 32 MiB L2).

So the surviving statement is weaker and still useful: **the expert GEMM is not
bandwidth-starved**, and at 40-61 W of 180 W it is not power-bound either. What remains
plausible is arithmetic intensity at M ~= 3 tokens, which the tokens-per-window data
below supports directly.

### The one lever the data supports: more tokens per window

taken from the engine's own decode-timing line over 60 window-samples:

| tokens/window | ms/window | ms/token | t/s |
|---|---|---|---|
| 2.19 | 23.47 | 10.72 | 93 |
| 2.21 | 22.78 | 10.31 | 97 |
| 2.66 | 25.00 | 9.40 | 106 |
| 2.73 | 25.14 | 9.21 | 109 |
| **2.87** | 25.85 | **9.01** | **111** |

More tokens per window costs sub-linearly: 31% more tokens bought 19% better
ms/token. That is the M=3 GEMM amortizing, and it is the only measured lever left.
Acceptance sets it (67-76% per draft), so the work is on the drafter, not the kernels.

### What was tried and is now closed

- card-0 expert cache: exists already; enlarging it is neutral (106.1 vs 106.7 t/s)
- interleaving the stages' host serve (`--exp nstage_pipe`): 45.3% mean card util
  against the serial 47.0% - no overlap, because the stage graphs are already
  launched up front and the dependency is in the data, not in the host loop
- the split cut: swept constant 8,22,34, worth +15% over the even cut and no further

## P2P PEERING: 7.14 GB/s, and the fetch tier is dead (an earlier claim of 194 was my bug)

Two measurements were wrong before this one; recording both, because the second was
published before it was caught.

**Wrong #1:** a 32 MiB buffer for a bandwidth test. The card's L2 is 32 MiB, so the copy
was cache-resident and reported 194 GB/s.

**Wrong #2:** the retry used 1 GiB and still reported 192 GB/s, with the payload verified
at head and tail. The bug was the allocation: `cudaSetDevice(a)` was current when BOTH
buffers were allocated, so `db` lived on device A too and the "peer copy" was a
same-device copy - local HBM, not PCIe.

**Correct**, buffers on their own devices (`cudaPointerGetAttributes` confirms device 0
and device 1), 256 MiB:

| path | streaming | 2.64 MiB (one expert blob) |
|---|---|---|
| P2P D2D (A->B) | **7.14 GB/s** | **387 us** |
| host bounce (D2H+H2D) | 3.59 GB/s | 781 us |
| H2D only | 7.22 GB/s | 390 us |

**P2P is 7.14 GB/s - the PCIe 4.0 x4 rate, the same as H2D.** Its only advantage is
skipping the second hop (2x the bounce's latency). There is no bandwidth headroom.

### The budget, which settles it

The expert GEMM is 0.44 ms per layer. One fetched expert costs **0.387 ms over P2P =
88% of the whole layer budget, for 1 of 10 experts**. And streaming a layer's 10 experts
(26 MB) at 7.14 GB/s is 3.7 ms, 8x the budget; 1.24 GB per token at a 9 ms token time
needs 138 GB/s, which the link does not have by 20x.

**A fetch-based expert tier cannot work for decode on this fabric.** The latency point
is the right one: only a RESIDENT expert has zero fetch latency, which is exactly why
more stages win - they buy cache slots (3974 vs 1358), not parallelism.


## KIMI REVIEW: what it corrected, verified independently

A reviewer pass over the branch's conclusions. Two of its findings are confirmed
wrong in my text; two of its findings are wrong themselves and I checked.

### Confirmed wrong: the utilization reading (and the causality)

nvidia-smi counts an ACTIVE KERNEL, and each stage's graph is launched at window
start with a device spin kernel (`wait_flag_ge`) waiting for its predecessor's
hand-off flag. So a card that is merely WAITING reads as busy, and the busy
fraction grows with launch order. The model predicts cumulative-spin fractions of
15 / 38 / 59 / 89% for 8/14/12/14; measured is 19 / 38 / 53 / 79%. Near-perfect.

**The real mean duty cycle is 22%** (23.29 ms of stage work / 4 cards / 26.3 ms),
not the 47% this page quoted. And "each stage waits on its successor" was backwards:
a stage waits on its PREDECESSOR. The monotone rise is an artifact of launch order,
not a "shape to attack", and there is even less real parallelism than believed.

The wattage readings are the honest signal, and they agree: c0 50 W rising to c3 71 W
tracks real work density.

### Confirmed wrong: the boot search is a coverage maximizer, not a window optimizer

`layer_ms[i]` comes from SMs x GHz and the four cards are identical, so
sum_i (le-lb)*layer_ms[i] is cut-invariant (~36.5 ms for every cut) and the ONLY
term that varies in `predict()` is `miss_ms * (1 - held_mass)`. The "exhaustive
15,180-cut search" therefore maximizes coverage and nothing else - it cannot
represent serial position, the CUDA0 session asymmetry, or the head at the tail,
because those terms do not exist in its objective. That is the mechanical
explanation of "the auto cut has better coverage and still loses", which this page
had recorded only as an anomaly.

### Wrong in the review: the stamp terms do NOT overlap

The review flagged a factor-2 gap ("terms sum ~1.5 ms against a 0.442 ms wall") and
suspected double counting. Parsed properly from one boot, the GDN terms sum to 19.04
ms and the QSA terms to 4.77 ms, totalling **23.81 ms against the instrument's own
23.81** - exact. The terms are disjoint, `VRAM hits` is a real share, and the 0.442
figure was a per-layer average of a different stage's window. No re-derivation
needed, and the grouped-GEMM work is not blocked on this.

### Wrong in the review: the prefill "regression"

PP4-STATE compared 3,545 t/s (an early tp2pp2-branch boot) against 3,224 (this
branch), which is not a regression but two different binaries and prompts. The
like-for-like 122k number is 4,232 t/s on both. Prompt-cache contamination (`c` in
`pp=NNN/NNNc`) is the trap; the 62k figures need one cold-cache rerun before anyone
quotes them.

### Corrected arithmetic

"31% more tokens bought 19% better efficiency" - the efficiency gain is
**1 - 9.01/10.72 = 16%**. And the window is NOT flat in T: it rises ~10% for +31%
tokens, so the lever is **~2.5% window cost per extra token**, i.e. each +0.5
tok/win of acceptance buys ~5%.

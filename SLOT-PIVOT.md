# The pivot: drop to 1 slot, give the VRAM to experts, put more KV in DRAM

Geometry from `include/basalt/core/layout.hpp`: 48 layers, every 4th is full attention
(12 QSA), GDN on the other 36. QSA KV is `n_head_kv 2 x head_dim 256 x (K+V)` =
**1024 B/token/QSA-layer** at int8.

## The VRAM that actually comes back: 3.00 GiB, not 6

One 262144-token slot's KV is **3.00 GiB** (1024 B x 12 QSA layers x 262144). Two slots
are 6.00 GiB, so dropping to one frees **3.00 GiB total**, spread by which cards own
QSA layers under the 8/14/12/14 cut:

| card | layers | QSA | freed | experts gained |
|---|---|---|---|---|
| c0 | 0-7 | 2 | 0.50 GiB | +194 |
| c1 | 8-21 | 3 | 0.75 GiB | +291 |
| c2 | 22-33 | 3 | 0.75 GiB | +291 |
| c3 | 34-47 | 4 | 1.00 GiB | +388 |
| **total** | | 12 | **3.00 GiB** | **+1,164 blobs** |

1,164 of 24,576 profiled pairs = **+4.7% cache capacity**. Solo hit rate is already
**97.7%**, the miss path is ~7% of the window, so removing 4.7% of an already-tiny miss
set is worth **well under 1% of decode**. And the engine already reports
**file (SSD) 0.0 GiB** - nothing is spilling, so the freed VRAM has no deficit to fill.

**Conclusion A: giving the second slot's VRAM to experts is worth <1% and costs the
ability to serve two requests at once. It is not the lever.**

## KV in DRAM is catastrophically worse, and the arithmetic is simple

Full attention at context N reads N positions of K and V per QSA layer per token. On
four PCIe 4.0 x4 links measured at **7.14 GB/s** (P2P, verified with buffers on separate
devices), against VRAM at 448 GB/s:

| ctx | per QSA layer | 1 layer in DRAM | 4 layers in DRAM | 4 layers in VRAM |
|---|---|---|---|---|
| 8,192 | 8 MiB | 1.12 ms | 4.48 ms | 0.07 ms |
| 32,768 | 32 MiB | 4.48 ms | 17.9 ms | 0.29 ms |
| 131,072 | 128 MiB | 17.9 ms | 71.7 ms | 1.14 ms |
| **262,144** | **256 MiB** | **35.9 ms** | **143 ms** | **2.29 ms** |

Decode's whole budget is **9.0 ms/token for all 48 layers**. One QSA layer's KV in DRAM
at full context costs 35.9 ms - **4x the entire token budget, for one layer**.

**Conclusion B: KV must stay in VRAM at long context. Putting more of it in DRAM is not
a trade, it is a 60x slowdown.**

The pivot only makes sense if contexts are SHORT. At 8,192 tokens, 4 DRAM-resident QSA
layers cost 4.5 ms/token - still half the current budget, so still bad. Below ~2k it
becomes affordable and pointless.

## What it does to each topology

### PP4 (4 stages, 1 card each) - the one you asked about
- frees 3.00 GiB across 4 cards, +4.7% cache -> **<1% decode gain**, measured-adjacent
- loses slot 2 entirely
- **verdict: negative.** The 0.9%-of-compute-peak problem is not a memory problem.

### tp2pp2 (2 stages x 2 cards, the previous branch)
- Same 3.00 GiB, but the pair already **splits experts across two cards** and ships 55.8%
  of routed experts over PCIe. Its cache is 13.98 GiB and holds 5,429 of 24,576 pairs
  (22%) - so +1,164 blobs there is **+4.7pp of coverage on a much weaker base**,
  proportionally a bigger lever than on PP4.
- But tp2pp2's decode is ~90 t/s against PP4's ~110, and its prefill is 2,780 against
  4,232. A 4.7pp coverage gain does not close either gap.
- **verdict: negative, but less negative than on PP4.**

### TP=4 (shelved)
- At TP=4 there is **one KV per card either way** (each card holds the full KV for its
  head shard, not a replica), so dropping a slot frees the same 3.00 GiB total.
- TP=4 was shelved on the **max-of-4 straggler** (1 ms/card -> 6.3x inflation) and a
  prefill hidden fraction of 42.7% against a 56% ceiling. **Neither is a memory
  constraint, so freeing 3 GiB changes nothing about the verdict.**
- **verdict: unchanged - still shelved.**

## The one thing the pivot DOES clarify

The 2 slots are not costing us VRAM we need. The engine has **0.0 GiB on SSD** and a
97.7% hit rate with two slots resident. The second slot is close to free, and dropping it
buys under 1%. **Concurrency is not the constraint, and neither is memory.**

What WOULD make the pivot right is a different premise: if the box needed the VRAM -
i.e. if experts were spilling - then freeing 3 GiB would matter. It is not spilling.

## Recommendation

Do not take the pivot as stated. If the goal is that a single stream goes faster, the
ledger from tonight says every scheduler and memory lever is exhausted and the bound is
arithmetic intensity at M~=3.

The pivot IS worth it under one different reading: **drop slot 2 and spend the freed
3 GiB on a longer single context** - i.e. keep 1 slot but raise `--max-context` beyond
262144 on the same KV budget. That is a product capability gain rather than a speed one,
and it is the only direction where the freed memory is not worth less than 1%.

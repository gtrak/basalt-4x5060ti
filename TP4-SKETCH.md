# TP=4 Paper Sketch — the morning's decision package (kimi #8, the corrected ledger)

Written the night of the owner redirect ("if tp4 becomes a clear win, take it, too";
"you are empowered to make the decision"). All numbers kimi #8-checked against
layout.hpp (n_embd=2560, n_expert=512, n_ff=640, K=10), RUN 9/16/21/24, and the
owner's topology correction. **The prefill case was REPRICED — my first arithmetic
under-priced the combine wire ~4x.** The verdict stands as: *arithmetically plausible
only under partial-merge BF16 combine + hidden fraction ≥ ~70% + the PPL/KLD gate.*

## 1. The corrected wire ledger (kimi #8 Q1)

The routed output crosses at **n_embd = 2560 wide**, BF16 = **5,120 B per ROUTING**
(640 is the expert FFN hidden width — nothing 640-wide crosses a card boundary; the
down weight 2560×640 lives with the expert). Volume is per **routing**, not per token:
T×K×(1−1/4) = 8,192×10×0.75 = **61,440 routings/layer/chunk** cross at TP=4.

| form | per layer-chunk | total (~48 layers × 15 chunks) | vs the d2d pool |
|---|---|---|---|
| row-shipping (what the tp2 pair runs today) | ~350 MB | ~250 GB (~63 GB/card) | **exceeds the entire 7.8–8.6 GiB/s pool at ZERO exposure — dead by capacity** |
| attention reduce (4-way ring, all 48 layers — no stages at TP=4) | 84 MB-class bursts, QSA positions | ≥ 45 GB FP32 round | on top of the combine |
| **partial-merge combine** (each card locally sums its held experts' weighted outputs; pair-set sums T×2560 partials BF16) | ~90–130 MB | ~65–95 GB ≈ 2.7–3.9 GB/s aggregate | fits, ~2× pool headroom before jitter — **straggler-coupled (max-of-4) and contending with the tier's h2d face at the root complex** |

**Combine-form note with teeth**: partial-merge is also 5× cheaper than row-shipping
AT TP=2 (42 vs 210 MB/layer-chunk) — "why is the current pair row-shipping?" is an
open question the sketch must answer; if the answer is combine correctness, it applies
at TP=4 too and the gate below is not optional.

## 2. The gates TP=4's prefill must pass (in order)

1. **The numerics gate (non-negotiable)**: partial-merge changes the MoE sum's
   reduction order → PPL/KLD through the decode path + a `docs/precision/` page.
   Budget as a FULL gate, not a footnote. (The BF16-wire gate machinery from the
   slice work carries.)
2. **The hidden fraction ≥ ~70% at 4-deep** under the CORRECTED ledger, measured by
   the rendezvous bench (two-channel cadence: the continuous partial-merge background
   + the bursty QSA reduces; {row-ship, partial-merge} × {2,4}-deep × {clean, jittered};
   SM-push for AR / copy-engine for the tier stand-in, never mixed).
3. **The straggler coupling**: max-of-4 — one card's tier jitter reaches all four.
   The bench's jitter propagation arm is the evidence slot.

The projection with all gates passed: ~24.5 s compute (the split's 32.6 s × ~0.75
stage-idle recovery) + the hidden combine ≈ **≥ the split's 3,750 t/s** — the "4,300"
figure only ships if every gate above has passed. The owner's sglang prior
(TP=4 > PP=4 on 27B dense post-PIX) is **weak evidence here**: two-axis regime
mismatch (that model was bandwidth-bound on per-link-independent links; ours is
compute-bound on a shared pool — 4-card all-to-all measured 0.64 GiB/s/link, one
full-duplex pair already saturates the pool).

## 3. Decode at TP=4 — the collective-free ruling EXTENDS (a fortiori)

Kimi #5's ruling ("decode stays collective-free PERMANENTLY on this fabric") was
fabric-scoped and **extends, strengthened**: more partners, same pool, tighter protocol.
But it does not block TP=4 decode, because the natural design is not collectives:

**Generalize the zero-collective peer-window/ring design from 2 to 3 peers.**
`dec_moe.hpp:24 kDmMaxX = 2` is the literal seam — rings, doorbells, and done-words
per peer already exist as an array; the decode graph, combine, and plan structure are
peer-count-parameterized. TP=4 decode = the M1 pattern ×3 + a residency re-carve.
The 19.4 ms question becomes "does 3-peer row-fetch hold the window" — priced by the
bench's decode-cadence mode (96 rounds/window under a tier-fill background, SM-push
vs copy-engine, at 4 cards) BEFORE any build. RUN 24's solo pair: 1.9 ms/pair with
zero margin — the 3-peer straggler term is what decides.

## 4. The residency/tier re-walk (the slice machinery carries)

Sharded experts + tier over 4 cards: the banked slice machinery carries (shard
packing with native-only pack handling, span lifecycle proofs, fits_within tier
reserve, the pair protocol generalized). The tier model re-walks with:
- the d2d pool 4-way shared (0.64 GiB/s/link all-to-all — the combine + the streaming
  contend); the h2d face has headroom (24–30 GB/s capability vs 18.4 observed) —
  4-way tier sharing is fine THERE; the pool is the cap.
- the share-v2 placement model re-parameterized for 4 (the row-rate falsifier's
  rates re-measured at 4-way contention — the lead/peer asymmetry likely compresses
  as every card's SMs are equally busy).

## 5. The decision rule (mine, the morning)

Build TP=4 iff ALL of: (a) the bench's partial-merge hidden fraction ≥ ~70% at
4-deep with jitter; (b) the decode-cadence mode shows the 3-peer window ≤ ~19.4 ms
budget (or names the gap); (c) the PPL/KLD gate passes on the partial-merge sum;
(d) the projected prefill ≥ the split's 3,750 with the corrected ledger. Any gate
failed measurably → TP=4 stays shelved with this page as its record. No single
coefficient decides; the ledger, the gate, and the cadence do.

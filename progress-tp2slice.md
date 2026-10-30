# RUN 25 — the prompt-attention slice: byte ledger and filed predictions (BEFORE any measurement)

## Supervisor rulings on record (this run)

- **Ruling B: (b) now, (a) as the funded follow-up.** For this milestone the partner gains its KV
  span and the LEAD's pools stay whole; the resulting expert-tier shrink is sized at **1.2 GiB
  (~7.6%)** under the shipped `--kv int8`, not 1.61 GiB. Decode stays untouched — that constraint
  outranks the tier cost, because decode is the campaign's working win and option (a) would touch
  session carving, the prompt cache, KV streaming and decode's pools.
  Option (a) — narrowing the lead's pools to its shard, freeing ~1.5 GiB to restore the partner's
  tier — is **filed as the slice-success contingency**, not rejected and not scope creep: it is
  funded only if the slice clears its ship bar, and dies with the slice if it does not.
- **Erratum to the delegating brief: the 0.6 GB / 3-QSA-layer figure was wrong; the 6-layer
  1.5 GiB fp16 / 1.2 GiB int8 figure above supersedes it.**
- **Ship bar calibration (kimi #5 wording):** the bar is `>= ~20 ms/layer OR a replicated positive
  at matched power`. A measured pipelined net of +15-18 ms that replicates across rule-of-3 boots
  with decode floors intact is a **ship, not a failure**. The +10 ms abort gate is unchanged.

## Status of this milestone

**Landed this step** — the ruled refusals and the gate, default-off and byte-identical:

- `--exp tp_prompt_attn` now exists and **refuses at init** at its ruled seam, `Prefill::set_peer`
  (`src/prefill/prefill.cpp`), naming the four pieces that are missing. The flag is the prompt
  path's only: it cannot arm the decode head split, and nothing in `Verifier` reads it. Off (the
  default) the path is byte-identical — the flag is read once, on the peer arm's setup, and no
  other code observes it.
- `src/core/verify.cpp` `qhs_peer_item()`'s comment now states what it always was: the **decode**
  window's peer item (RUN 17's spec), still unbuilt, `--exp decode_headsplit=1` still refusing.
  The comment names the failure arming it alone would cause (`dec_moe_mix_combine` waiting a
  done-word nothing raises; RUN 24 review #5 verdict 2, decode collective-free) and says the
  prompt gate is a different seam, so the two are never confused again — which is what the brief's
  item 3 conflated.

**Seam correction, recorded for the next reader:** task item 3 pointed at `verify.cpp:598
qhs_peer_item()` for the prompt gate. That predicate is decode's and stays false. The prompt gate
lives entirely in the prefill path, and its refusal is in `Prefill::set_peer`.

Gates at this step: build green under `-j4`; `ctest -L cpu` **67/68** — only the pre-existing
`expert_multi_test` (this CPU lacks AVX512-VNNI/VBMI); `tools/check_platform_boundary.sh` ok;
`tools/check_stderr_logging.sh` ok. `:8000` never contacted, no engine started, no GPU run while
the server holds the cards. GPU numbers below stay **TODO-for-the-window**.

**Still not built** — task items 1, 2 and commit 2:

- No peer-side QSA arm beside the lead's (`src/prefill/prefill.cpp:2844-2939` is the model), no
  `--exp tp_prompt_attn` behaviour behind the refusal, no AR call site.
- No change to the peer's expert tier carve to fund the span (ruling B) — that lands with the arm,
  through `Prefill::peer_bytes` so `pair_prompt_bytes` / `stage_room` leave the room free.
- The software-pipelined variant (commit 2, `=2`) is untouched.

## 0. Rulings taken mid-run (the supervisor's, applied to this file)

The peer keeps its own KV span, so it can be stale: it only ever sees the chunks it is handed.
Every cell the LEAD appends while the peer is out of the loop — decode's appends (decode stays
collective-free, RUN 24 verdict 2, so nothing tells the peer), cached-prefix reuse
(`session_copy_state`, generate.cpp:9634 and :9770, device-to-device on the lead),
`prompt_session_clear` (:9623), park/resume, `Prefill::set_session` — leaves the peer's span and
its indexer pooled table describing cells it never wrote. The selection covers `[0, p0)`, so a
stale span is not a crash: it is plausible tokens from garbage, the RUN 17 class this branch
refuses everywhere else. Ruled:

- **P1 approved.** Per stage, track `span_valid` = the cell count the peer's pools and indexer
  pooled table provably hold. At a prompt run's start, P2P-copy the peer's KV-head slice and its
  indexer pooled rows for `[span_valid, p0)` from the lead's pools, on the peer's own copy stream;
  the peer then appends `[p0, p0+T)` itself. The figures publish on the boot and chunk lines — the
  copy must be measurable, not assumed.
- **P2 approved.** Invalidate `span_valid` to the true value at every lead-side mutation the peer
  cannot see: both `session_copy_state` sites, `prompt_session_clear`, `Prefill::set_session`,
  park/resume, and each decode window's end. Only has to be right under the gate; off stays
  byte-identical.
- **P3, with the pricing correction applied below** in §2b.
- **The FP4 planes are NOT an attention input.** The open assumption in the first version of this
  file is closed: q/k/v must derive from the same source the lead's attention consumed, within the
  kernel's 2e-3 relative tolerance. The planes carry ~1e-2-class expert-quantization error — two
  orders outside — so q-from-planes fails parity the same class as a stale span. The planes stay
  the expert leg's input only (RUN 19 unchanged). See §2a for the mix that survives this.

## 0a. Landed earlier in the milestone

Commit `b1f9850d`, `tests: the head-split parity gains the prompt attention kernel`:

- Task item **4** (parity): case 6 in `tests/kernels/tp2_head_split_parity.cpp` runs 8 queries
  through `qsa_prompt_attn_batch` whole (24 heads / 2 KV) and once per card (12 / 1) over each
  card's KV-head slice, sharing one selection, step and permuted page table. Tolerance stated in
  the source: relative 2e-3 of the row's largest magnitude, which is the prompt kernel's own bound
  (`qsa_prompt_attn.hpp` says it is FP32-level, not bitwise). A bitwise run prints `bitwise` and
  its worst deviation, so drift toward the bound stays visible.
  GPU-labelled (`basalt_gpu_test`, skip 77 without a card), so the cpu count is unchanged at 68.
  **Not executed** — the live server holds the cards. It compiles clean under `-j4`; the two
  `-Wpointer-arith` warnings at lines 430-431 are pre-existing, on untouched lines.
- Task item **5** (predictions): this file, filed before any measurement.

Gates: `tools/check_platform_boundary.sh` ok, `tools/check_stderr_logging.sh` ok,
`ctest -L cpu` **67/68** — only the pre-existing `expert_multi_test` (this CPU lacks
AVX512-VNNI/VBMI). Build used `-j4` per the shared-CPU constraint. `:8000` was never contacted,
no engine started, no GPU test run while the server holds the cards.

**Not built** — the pipelined variant and the executor behind the gate remain open:

- No peer-side prompt-attention executor exists; `--exp tp_prompt_attn` refuses at
  `Prefill::set_peer`, and `qhs_peer_item()` (`src/core/verify.cpp`) still returns false as
  decode's unbuilt window item.
- The software-pipelined variant (`=2`) is untouched.

## 2a. The mix, re-filed under the parity invariant

The ruled invariant decides the mix: **the peer's attention must consume the same q/k/v sources as
the lead's, within the kernel's 2e-3 relative tolerance.** Applying it to the two candidate shapes:

- **q must come from the FP16 rows the lead's own `attn_q` consumed** — `m.mixed_h`
  (`prefill.cpp:2849`, the `native_proj` input), not BF16 and not the FP4 planes. BF16 has an 8-bit
  mantissa (~2e-3 relative), which sits exactly *on* the bound; the lead's own FP16 image has 11
  bits (~5e-4) and, shipped as the same bits the lead read, makes the peer's q *its own half of the
  lead's q bitwise* — not a tolerance argument. Same 41.9 MB a layer, strictly better parity.
- **The K/V-shipping substitution is declined, on arithmetic.** Shipping the lead's `m.Kc`/`m.Vc`
  slice for the peer's KV head (8.4 MB) and deleting the peer's `attn_k`/`attn_v` residency looks
  cheaper, but only if nothing else is on the wire. q parity already forces the 41.9 MB of mixed
  rows, and those rows are sufficient for the peer to project its own K/V. So the substitution
  would *add* 8.4 MB a layer and keep the weights it promised to delete. The peer keeps
  `attn_q`/`attn_k`/`attn_v` + the q/k norms; no `attn_*` weight ships per layer.

The mix that survives, per QSA layer, chunk 8192 (T), `n_embd` 2560:

    inbound   41.9 MB  the lead's FP16 mixed rows, m.mixed_h (T x 2560 x 2 B)  -> peer's q, k, v
              21.0 MB  the indexer inputs: q_idx 16.8 + idx_raw 4.2            -> peer re-runs the indexer
    outbound  41.9 MB  the o_proj partial, BF16
    total    104.8 MB a QSA layer

**The filed 104.8 MB stands** — the substitution the ruling offered cannot pay once q parity pins
the mixed rows, and the honest number is the one already filed. 104.8 MB at the staggered
3.96 GiB/s face is **26.5 ms a layer**, unchanged from §2, so §3's predictions are unaffected.
The q-source is stated explicitly because everything downstream grades against it: **q, k and v are
the peer's own projections of the lead's FP16 mixed rows**; the lead ships its indexer projections
rather than the peer recomputing those, so the selection stays the lead's.

## 2b. The span catch-up, priced per TURN (P3, with the correction applied)

The catch-up is **once per prompt run**, covering `[span_valid, p0)`, not per layer and not per
chunk. It moves the peer's KV slice plus the indexer pooled rows it must also hold:

    KV       6 layers x cells x 256 dim x 2 (K,V) x 1 B (int8)  = 134.2 MB x 6 =  805 MB at 262144 cells
    indexer  6 layers x (cells/idx_block + 2) x 128 dim x 4 B   =  33.6 MB x 6 =  201 MB
                                                            total 1.006 GiB
    at the staggered 3.96 GiB/s face                            ~254 ms ONCE per turn
    at the both-pairs 2.00 GiB/s face                           ~503 ms once per turn

**A correction to the ruled figure, in the same spirit as §1a's:** the ruled estimate was ~0.6 GiB
/~150 ms. Mine is ~1.0 GiB / ~250 ms because it bills K *and* V for all 6 layers (6 x 134 MB) and
adds the indexer's pooled table (201 MB), which the peer must hold too in order to re-run the
indexer. The arithmetic is above so it can be checked rather than trusted; if the ruling meant
K-only or a per-stage 3 layers, the figure drops to the ruled one. Flagged, not silently adopted.

Scaled by staleness, which is the point of tracking `span_valid`:

- a **fresh** prompt (`a0 = 0`, no decode since the last run) copies **nothing** — the peer's own
  per-chunk writes built the span;
- a **cached-prefix** turn reusing 110k cells copies ~338 MB KV + ~84 MB indexer = **422 MB**,
  ~107 ms at the staggered face;
- the worst case is a 262k-cell session whose whole span went stale across a long decode: the
  1.006 GiB / ~254 ms above.

Amortized over a multi-second turn this does not eat the +18/+15 margin by construction, which is
the ruling's point and I accept it. What it does put on the table is **turn-start exposure**, and
that is the number the window must read rather than assume:

- The copy runs on the peer's own copy stream, so it *can* overlap the lead's layers up to the
  stage's first QSA layer of the first chunk. Exposed term:
  `max(0, copy_wall - lead_compute_until_that_layer)`.
- The peer's attention at that first QSA layer cannot start before its span lands.
- So the honest question is whether it hides under the prompt read and the first chunks or
  serializes into ttft. At 110k cached cells (~107 ms) against one 8192-token chunk's lead compute
  before QSA layer 3 (~4 dense layers), it does **not** hide: expect ~50-100 ms added to ttft on a
  cached-prefix turn, ~250 ms on the worst-case stale session. On a fresh prompt it is free.
  **TODO-for-the-window:** the chunk-line field must print `span_valid`, cells copied, and the
  copy's wall time, so this is read and not argued.

If that measured term eats the margin, the P3 fallback verdict stands as filed: with a
peer-resident span the slice cannot pay on this fabric, and peer-reads-the-lead's-pool-over-P2P is
not the escape — it is ~50 GB a chunk (cap 2048 x 256 dim x 2 B x 8192 queries) and dead.

## 1. Corrections to the ledger I was handed

Two figures in the opening brief were wrong, and one of them changes what has to be funded.

**1a. The KV span is 6 QSA layers a stage, not 3.** `qsa_interval` is 4 over 48 layers
(`include/basalt/core/layout.hpp:30,57`), so a stage holds layers 0-23 = QSA layers
3,7,11,15,19,23 = **6** full-attention layers a stage, not 3. My first estimate used 3 and
understated the peer's KV residency by 2x.

Per stage-card, the KV one card of the pair owns for its KV group, at 262144 cells:

    6 layers x 262144 cells x 1 KV head x 256 dim x 2 (K,V) x 2 B (fp16) = 1.61 GiB

`--kv int8` (the shipped boot) prices K at ~1 B + per-64 scales instead of 2 B, which lands the
same span near **1.2 GiB**. Either way it is roughly **1.5 GiB a stage-card, not 0.6** — about
10% of the 15.8 GiB half-depth expert carve, not the ~4% decision B was taken on.

**1b. Whether that is NEW memory depends on a question the brief did not answer.** A true head
split means the lead keeps only KV head 0 and the partner only head 1. If the lead's pools are
narrowed to its own shard too, the pair's total KV is **unchanged** — the lead frees ~1.5 GiB and
the partner allocates ~1.5 GiB, and decision B's tier shrink is a redistribution, not a cost. If
the lead's pools stay whole while the partner gains a span, it is **+1.5 GiB of genuinely new
resident bytes** and the shrink is real. Narrowing the lead's pools touches session carving, the
prompt cache and KV streaming, which is a larger change than this milestone.

**Open, needs a decision before the executor is built.** The choice changes both the funding
premise of decision B and whether the slice is VRAM-neutral.

## 2. The inbound mix, pinned to the cheapest honest option

Chunk 8192, 12 of 24 query heads a card, `n_embd` 2560, `head_dim` 256, `idx_top_k` 2048 so
`cap = 2048 + idx_block - 1 = 2051`. Bytes per QSA layer, one direction.

| flow | FP32 | BF16 / narrowed |
|---|---|---|
| the peer's q (12 heads x 256) | 100.7 MB | 50.3 MB |
| the mixed rows, to compute q/k/v on the peer | 83.9 MB | 41.9 MB |
| the selection ids (T x cap x 4) | 67.2 MB | — |
| the indexer inputs (q_idx 16.8 + idx_raw 4.2) | 21.0 MB | 21.0 MB |
| the o_proj partial, peer -> lead | 83.9 MB | 41.9 MB |

**Selected mix** (supervisor's direction: peer recomputes q/k/v locally, re-runs the indexer,
BF16 partial out):

    inbound   41.9 (BF16 mixed) + 21.0 (indexer inputs) = 62.9 MB
    outbound  41.9 (BF16 o_proj partial)
    total     104.8 MB a QSA layer

= **26.5 ms a layer at the measured 3.96 GiB/s staggered face** (`TP2PP2-NOTES.md` RUN 24), which
is inside the ~105 MB / ~27 ms target, and inside the ~92.6 ms MoE cover commit 2 hides it under.

Two caveats carried forward, not smoothed over:
- 3.96 GiB/s is the **staggered** figure. Both AR pairs pushing at once measures 2.00 GiB/s, which
  doubles every number here and makes the serialized variant negative before it starts.
- RUN 24's own caveat stands: only the **minimum** is a fabric floor; medians moved 30x under load.

## 3. Predictions, filed before measuring

Graded against the task's dual rule on the same boot, vs EP+PP without the slice.

**Commit 1 — serialized reduce-to-lead (`--exp tp_prompt_attn=1`).** The 26.5 ms of wire sits on
the critical path beside the 27.2 ms of attention compute it removes.

    predicted net, stage A: -2 ms/layer-pair   (90% CI roughly [-12, +6])
    predicted net, stage B: -4 ms/layer-pair

**Prediction: commit 1 lands below the +10 ms kill gate.** It is being built for correctness and
parity, exactly as directed, not as a candidate to ship.

**Commit 2 — pipelined (`--exp tp_prompt_attn=2`).** The inbound legs move under layer l's
~92.6 ms MoE compute. They fit with room (26.5 ms of wire under 92.6 ms of cover), so the
transfer should largely disappear and the 27.2 ms of saved lead attention should mostly land.

    predicted net, stage A: +18 ms/layer-pair   (90% CI roughly [+6, +26])
    predicted net, stage B: +15 ms/layer-pair

**Prediction: commit 2 clears +10 ms but does not reach the +20 ms ship bar** — the residual is
the peer's own attention kernel, which the ledger does not remove from the pair, only relocate,
plus the reduce-to-lead the pipeline cannot hide at the last layer.

If commit 2 measures under +10 ms/layer-pair, **STOP and re-plan** per the ratified kill criterion.
I will report the number as measured; it will not be tuned to pass.

## 4. What is NOT built yet

- The peer's QSA arm, the partner's KV span allocation, `span_valid` and its catch-up copy, and the
  o_proj reduce-to-lead (task items 1 and 2). `--exp tp_prompt_attn` therefore **refuses at init**
  at `Prefill::set_peer`, naming those four, rather than running the lead's heads alone.
- `qhs_peer_item()` (`src/core/verify.cpp`) still returns false. That is decode's window item and
  is *supposed* to stay false for this milestone; `--exp decode_headsplit=1` still refuses.
- No parity extension to `tp2_head_split_parity` beyond case 6, no AR call site in
  `src/prefill/prefill.cpp`.
- Closed, no longer open: the FP4-planes-as-attention-input question. **No** — see §0 and §2a. The
  mix is the lead's FP16 mixed rows for q/k/v and the lead's indexer projections, priced at
  104.8 MB a QSA layer.

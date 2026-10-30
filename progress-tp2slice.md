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

## 2c. Item (1) landed: the span is SIZED, and the sizer corrects my hand estimate

`Prefill::PeerQsaSpan` (`include/basalt/prefill/prefill.hpp`, implemented beside `Prefill::peer_bytes`) prices
what the partner's card must hold resident for its half of the attention. One `Alloc` count walk produces both
the reserve and, later, the carve, so the tier cannot be sized for a different amount than `set_peer`
allocates. `peer_bytes` takes an optional `const PeerQsaSpan*`; the call sites are unwired until the arm
exists, because reserving tier for an arm that refuses would tax the partner for nothing.

Pinned by `tests/cuda/peer_qsa_span_test.cpp` (CPU-labelled, no device touched), which prints what it measures:

    wire   104.9 MB a QSA layer           (the filed 104.8, confirmed -- and it does NOT shrink with the split)
    span   1.69 GiB fp16 / 0.96 GiB int8  6 QSA layers of KV + indexer pooled, at 262144 cells
    total  1.58 GiB int8                  span + wire + the partner's per-chunk scratch

Three things this settles, each of which was an estimate before:

- **My 1.006 GiB was high; 0.96 GiB is the sizer's number.** I priced int8 K at "~1 B + per-64 scales" by hand
  and the layout is slightly tighter. **0.96 GiB int8 supersedes 1.006 everywhere in this file**, including in
  §2b's catch-up arithmetic -- the supervisor's adoption of my figure stands as to method (per turn, indexed
  by `span_valid`, KV *and* V, indexer table included) but the byte count is now the sizer's, not mine. The
  span catch-up moves the same 0.96 GiB at full staleness, so §2b's ~254 ms is a hair pessimistic; it stays
  as the filed number until the window measures the copy.
- **The tier cost of the slice is 1.58 GiB, not 1.2 GiB.** Ruling B was taken on 1.2 GiB (~7.6% of the
  15.8 GiB half-depth carve). The honest reserve is span + wire + scratch = **1.58 GiB, ~10% of the carve**,
  because the partner also needs the wire's staging and its own shard-width attention scratch (0.55 GiB at
  an 8192-token chunk, which the split halves -- a 16-head partner prices 0.63 GiB). Restated here rather
  than discovered at a boot.
- **Units, because the ledger mixed them:** the wire figures are 10^6 bytes, the tier figures 2^30. The test
  asserts both explicitly so a future comparison cannot silently mix them; the filed 104.8 MB wire and the
  ~1 GiB span were always consistent.

Also checked, because the share flags make it live: an uneven cut (`qsa_lead_heads` at the fired falsifier's
2.61 / 0.98 ratio puts 16 heads on the partner) is still priced, its scratch grows with its heads, and a
cut straddling a KV group is *priced* but refused for the kernels by `qsa_shard_whole_groups` -- sizing is not
where a bad ratio gets silently dropped.

**Not yet done for item (1):** the call sites (`peer_prompt_bytes` / `pair_prompt_bytes`, generate.cpp:3104 and
:3111) do not pass a span yet, so `stage_room` leaves no room. Wiring them belongs with the arm that fills it.

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

## Item (2) so far: the arm's container, allocated by the sizing walk itself

`PeerPrefill::Qsa` now exists: the partner's per-QSA-layer `QsaState` array, two wire sets, its shard-width
scratch, its resident projection pointers, its **own** `Gemm` (ruled: `Gemm::native` dequantizes on the device
the product runs on, so the lead's scratch and cuBLAS workspace cannot cross the link), and its own stream with
an `ev_in`/`ev_out` pair — the event-not-sync shape the ruling asked for, matching RUN 19's early transport.

**The anti-drift property, which is the point of this unit:** the buffers are laid out by ONE function,
`peer_qsa_layout`, and both callers drive it with a different `take` — `make_peer` with `cudaMalloc`, and
`Prefill::PeerQsaSpan::bytes()` with a counting no-op. The tier's reserve and the allocation are therefore the
same code, not two ledgers kept in step by hand. The states are carved with `qsa_prompt_state_carve` over an
allocation of `qsa_prompt_state_bytes`, so the pools cannot come out a different width than the tier was shrunk
by. `bytes()` no longer computes wire/scratch/weights/gemm itself; it reads them out of that walk.

Still missing, and the gate still refuses at `Prefill::set_peer` with the message updated to name exactly these
(the old message claimed the span and Gemm were absent; they are not):

- the per-layer QSA compute on the partner (item 2 proper, mirroring `prefill.cpp:2844-2953`),
- `span_valid` + the per-turn catch-up + its invalidation sites (item 3),
- the `o_proj` reduce-to-lead via `qsa_pair_prepare` + `Tp2Pair` SM push (item 4),
- the tier's call sites (`generate.cpp:3104/3111`) still pass no span, so `stage_room` leaves no room yet.

## Where the arithmetic lives

The byte ledger, the funding figure, the span catch-up pricing and the filed predictions are in
**`progress-tp2slice-ledger.md`** (§1 corrections, §2 the mix, §2b the catch-up, §2c/§2d the sizing rulings, §2f
the funding bug, §2i the o_proj funding, §3 the predictions and their re-basing). The kernel side of the work —
every hardcoded head count found, fixed and parity-tested — is in **`progress-tp2slice-kernels.md`**. Split to
keep each file under the project's 300-line cap.
The headline figures, restated so this file is usable on its own:

    wire traffic   104.9 MB a QSA layer        both modes; what the +10/+20 gates grade
    tier reserve   1.89 GiB int8               span 0.96 + wire 0.20 + scratch 0.64 + weights 0.05 + gemm 0.09
    span copy      <= 0.96 GiB per TURN        zero on a fresh prompt; ~254 ms worst case
    predictions    serialized -2/-4 ms         pipelined +18/+15 ms as an UPPER BOUND (no-extra-eviction)

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

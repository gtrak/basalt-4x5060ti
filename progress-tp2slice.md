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
- **Decode-window end does NOT invalidate the span — ruled, with the invariant it rests on.** Decode appends
  monotonically past everything a prompt attended over, so `[0, span_valid)` provably still holds the lead's bits
  and the catch-up imports exactly the appended cells; invalidating at every window's end would re-import the
  whole context on nearly every server prompt to defend a claim that was never broken. **The proof's assumption
  is that cells below `span_valid` are never rewritten or dropped.** Any feature that rewrites, shifts or evicts
  below the span must call `invalidate_peer_qsa_span` at the site that does it — named candidates: **KV
  compaction**, **context shift** (prefix removal / trimming / re-anchoring), and **sliding-window or
  attention-sink eviction**. Recorded as a documented invariant, not a tolerated gap.

- **Tier-reserve sizing when the split is not yet decided: option (c) — landed.** `peer_prompt_bytes` /
  `pair_prompt_bytes` price the partner's span before `set_peer` chooses the head split, so the reserve is struck
  against an unknown. Ruling: reserve the **symmetric 50/50** figure and have `set_peer` **refuse, naming both
  numbers**, when the chosen split needs more — never OOM, never silently under-reserve. The graded measurement
  pins the ratio with `qsa_ratio_stage_a|b` at 50/50, which the reserve covers by construction. Widest-case
  reserving is the fallback, decided **from measurement** only. Full wording and the implementation's figures:
  ledger §2j.
- **Ship bar calibration (kimi #5 wording):** the bar is `>= ~20 ms/layer OR a replicated positive
  at matched power`. A measured pipelined net of +15-18 ms that replicates across rule-of-3 boots
  with decode floors intact is a **ship, not a failure**. The +10 ms abort gate is unchanged.

## Status of this milestone

**The gate is open at 1.** `--exp tp_prompt_attn=1` arms the prompt-attention slice at `Prefill::set_peer`; 0
(the default) leaves the path byte-identical — the flag is read once, on the peer arm's setup, and no other code
observes it. It is the prompt path's only flag: `qhs_peer_item()` in `src/core/verify.cpp` still returns false
(decode's window item, RUN 17's spec, still unbuilt; `--exp decode_headsplit=1` still refuses), so the two seams
cannot be confused — the seam the RUN 24 brief pointed at for the prompt gate was decode's.

Gates, current: build green under `-j4`; `ctest -L cpu` **68/69** with the count verified at 69 — only the
pre-existing `expert_multi_test` (this CPU lacks AVX512-VNNI/VBMI); boundary and stderr-logging checks ok.
`:8000` never contacted, no engine started, decode paths untouched. GPU numbers below stay
**TODO-for-the-window**.

**Built along the way** — detail in `progress-tp2slice-kernels.md`: the arm's container and its sizing walk, the
resident projections and the partner's own `Gemm`, the shard-width kernels (`split_q_span`, `gate_attn_span`) and
the shard-aware cache writer (`kv_append_nkv`), the packed `attn_output` column funding and the pack's GGUF types,
and `peer_qsa_layer()` — the partner's QSA block, through the gate.

**Landed since**: `span_valid` with its catch-up, invalidation sites and the `kv_span_import` gather (case 11);
the **(c) tier reserve** (pinned symmetric cut, `set_peer` refuses a wider split by naming both figures); the
**all-reduce timeout fix** (`b34904ba`); and **pair bring-up** — `peer_qsa_pair_up()` opens one `Tp2Pair` across
the stage's two cards sized for a PROMPT partial (`chunk x n_embd`, not the decode shard's `1 x n_embd`) and packs
`attn_output` columns on BOTH cards, refusing if either card's set is missing on any QSA layer of the stage.

**LANDED (`b4cc1f70`, `e786401c`): the serialized slice runs.** (a) `peer_qsa_publish()` ships the lead's
`mixed_h` / `q_idx` / `idx_raw` — and the chunk's step counts once — into the wire set on the LEAD's compute
stream and records `qs.ev_in`; (b) the lead drives `peer_qsa_layer()` per QSA layer of the stage, at the stage's
ordinal, wire set alternating, with the partner's card current for its products (cuBLAS wants its own handle's
card, which nothing had needed to discover while the arm had no caller); (d) `peer_qsa_finish()` multiplies each
card's gated rows by its OWN packed columns and issues one `tp2_all_reduce()` per rank, on two streams. The gate
lifts for `--exp tp_prompt_attn=1` and refuses any other value rather than running it as 1. A chunk wider than
the arm's buffers is not sliced at all, decided once per layer so no layer can half-slice.
Funding: `Prefill::lead_qsa_bytes()` prices what the slice takes on the LEAD's card (its side of the pair's
staging, its packed columns), `generate.cpp` holds it back from each stage lead's tier, and `set_peer` refuses a
split whose lead side exceeds the pinned symmetric cut by naming both figures. `fits_within` still guards the
partner's side.

**Two events, because the collective has no back-pressure.** `Tp2Pair` counts its collectives, it does not credit
the staging back, so a rank a layer ahead can overwrite the peer's staging — or republish a wire set — while the
peer is still reading it. The lead's stream now waits on `ev_out` (the partner's layer is done) before the next
publish, and the partner's stream waits on `ev_red` (the lead's add is done) before its next push; every wait is
on work enqueued in an earlier layer, so the chain cannot cycle. That is what *serialized* means here, and why
only 1 is accepted: the pipelined gate needs a free-ness protocol the reduce currently supplies by sitting in the
lead's stream.

**Three findings the window must read rather than inherit:**

- **The reduce leg crosses as FP32, not the ledger's BF16.** `tp2_all_reduce()` pushes `rows x stride` FLOATS, so
  the return leg is **83.9 MB a QSA layer each way** at an 8192-token chunk, not the 41.9 MB §2a files; nothing
  writes `qs.part_out`. The pair's real traffic is ~190 MB a QSA layer, and the gates must be graded against that
  — or a BF16 push lands first (a new kernel: parity test plus a microbench line, per AGENTS.md).
- **The lead's attention is NOT halved by this leg.** Ruling B keeps the lead's pools whole, and the prompt-
  attention kernel derives a pool row from the head count it is handed (`(page * n_head_kv + kvh) * page_size`),
  so a 12-head lead run would read the wrong KV rows. The lead keeps every head's attention and gives up half an
  o_proj while the partner's attention is redundant — which is what the filed -2/-4 prediction was for, and the
  reason option (a) is the slice-success contingency rather than a free extra.
- **`span_valid` now advances with the prompt** (to `p0 + T`, after a fully sliced chunk's reduce was proved
  whole), and `set_peer_qsa_lead` finally has a call site, so a second prompt on a session that decoded imports
  its gap instead of refusing. Neither has been executed.

**The boot lines that say the slice is on** (`--exp tp_prompt_attn=1`; both must print, or the boot is not it):

    grep "peer prompt-attention pair CUDA"     # pair open, staging floats, columns on both cards, N QSA layers
    grep "peer prompt-attention arm, lead card" # the lead's reserve, the split, and the chunk ceiling

**Why the reduce was never allowed ahead of (a)+(b)**: `tp2_add_wait` spins, gives up, and returns leaving the
buffer a HALF sum while the collective still reported success. With no publisher and no driver, a lead-side reduce
would add the lead's heads alone and call it the model's output — RUN 17's class, caught at the wrapper
(`b34904ba`) but only late, which is why the pieces that make the peer's push exist had to land with it.

## 0. Rulings taken mid-run (the supervisor's, applied to this file)
**Branch doctrine (supervisor):** *cases unrun are cases that have not passed.* A GPU-labelled test that
has never executed is not evidence. Adopted after the first execution of `tp2_head_split_parity`
invalidated a kernel the head-split audit had explicitly cleared — one run, one real bug.

The peer keeps its own KV span, so it can be stale: it only ever sees the chunks it is handed.
Every cell the LEAD appends while the peer is out of the loop — decode's appends (decode stays
collective-free, RUN 24 verdict 2, so nothing tells the peer), cached-prefix reuse
(`session_copy_state`, generate.cpp:9634 and :9770, device-to-device on the lead),
`prompt_session_clear` (:9623), park/resume, `Prefill::set_session` — leaves the peer's span and
its indexer pooled table describing cells it never wrote. The selection covers `[0, p0)`, so a
stale span is not a crash: it is plausible tokens from garbage, the RUN 17 class this branch
refuses everywhere else. Ruled:

- **P1 approved.** Per stage, track `span_valid` = the cell count the peer's pools and indexer pooled table
  provably hold. At a prompt run's start, P2P-copy the peer's KV-head slice and its indexer pooled rows for
  `[span_valid, p0)` from the lead's pools, on the peer's own copy stream; the peer then appends `[p0, p0+T)`
  itself. The figures publish on the boot and chunk lines — the copy must be measurable, not assumed.
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

## 2a. The mix, re-filed under the parity invariant

The ruled invariant decides the mix: **the peer's attention must consume the same q/k/v sources as
the lead's, within the kernel's 2e-3 relative tolerance.** Applying it to the two candidate shapes:

- **q must come from the FP16 rows the lead's own `attn_q` consumed** — `m.mixed_h`
  (`prefill.cpp:2849`, the `native_proj` input), not BF16 and not the FP4 planes. BF16 has an 8-bit
  mantissa (~2e-3 relative), which sits exactly *on* the bound; the lead's own FP16 image has 11
  bits (~5e-4) and, shipped as the same bits the lead read, makes the peer's q *its own half of the
  lead's q bitwise*. Same 41.9 MB a layer, strictly better parity.
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
allocates. `peer_bytes` takes an optional `const PeerQsaSpan*` and the call sites now pass one
(`peer_prompt_bytes` / `pair_prompt_bytes`, generate.cpp), so the partner's tier leaves the room; the lead card's
side of the same arithmetic is `Prefill::lead_qsa_bytes`, held back from each stage lead's tier beside the windows.

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

`PeerPrefill::Qsa`: the partner's per-QSA-layer `QsaState` array, two wire sets, its shard-width scratch, its
resident projection pointers, its **own** `Gemm` (ruled: `Gemm::native` dequantizes on the device the product runs
on, so the lead's scratch and cuBLAS workspace cannot cross the link), and its own stream with an `ev_in`/
`ev_out` pair — the event-not-sync shape the ruling asked for, matching RUN 19's early transport.

**The anti-drift property, which is the point of this unit:** the buffers are laid out by ONE function,
`peer_qsa_layout`, and both callers drive it with a different `take` — `make_peer` with `cudaMalloc`, and
`Prefill::PeerQsaSpan::bytes()` with a counting no-op. The tier's reserve and the allocation are therefore the
same code, not two ledgers kept in step by hand, and the states are carved with `qsa_prompt_state_carve` over an
allocation of `qsa_prompt_state_bytes`, so the pools cannot come out a different width than the tier was shrunk by.

**The arm today:** `peer_qsa_layer()` has its caller (the lead, per QSA layer) and waits on the publish before it
reads the wire; its o_proj product lives in `peer_qsa_finish()`, beside the pair's collective, since both need the
packed columns. The head-split audit is closed: `split_q`, `gate_attn`, `kv_append` fixed; the selection kernels
cleared. Still unexecuted: the pair opening across two physical GPUs, the publish landing, the reduce summing and
the chunk-line verdicts — **TODO-for-the-window**, refused rather than degraded.

## Building this tree after a reboot

A fresh `cmake` configure does **not** reproduce the working build: `BUILD_TESTING` defaults off, so
`ctest -L cpu` finds **zero** tests and reports no failures, and `nvcc` is not on a non-login shell's
`PATH`. Both fail silently. Configure with:

    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
          -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DBUILD_TESTING=ON
    cmake --build build -j4
    ctest -L cpu --test-dir build -N       # MUST report 69 before a green suite means anything
    ctest -L cpu --test-dir build          # 68/69; expert_multi_test is the known failure

A suite of zero passes exactly as well as a suite of sixty-nine.

## Where the arithmetic lives

The byte ledger, funding, catch-up pricing and filed predictions are in **`progress-tp2slice-ledger.md`**; the
kernel side — every hardcoded head count found, fixed and parity-tested — in **`progress-tp2slice-kernels.md`**.
Headline figures, restated so this file stands alone:

    wire traffic   104.9 MB in a QSA layer    what the +10/+20 gates grade; the RETURN leg is FP32 today
                                              (83.9 MB, not the filed 41.9 BF16) -- see the findings above
    tier reserve   1.89 GiB int8               span 0.96 + wire 0.20 + scratch 0.64 + weights 0.05 + gemm 0.09
    span copy      <= 0.96 GiB per TURN        zero on a fresh prompt; ~254 ms worst case
    predictions    serialized -2/-4 ms         pipelined +18/+15 ms as an UPPER BOUND (no-extra-eviction)

## 4. What stays closed

`--exp tp_prompt_attn` accepts **1 (serialized) only**; any other value is refused at `Prefill::set_peer` rather
than run as 1, because the operator would read the resulting number as the pipelined one. The pipelined `=2` is
the next commit and needs the free-ness protocol the reduce currently supplies by being in the lead's stream.
`qhs_peer_item()` (`src/core/verify.cpp`) still returns false: that is decode's window item, supposed to stay
false for this milestone, and `--exp decode_headsplit=1` still refuses. Decode's paths — `qhs_peer_item`, the
peer window, `dm_peer_kernel` — are untouched by this slice. Every timing figure in these files is
**TODO-for-the-window**; none of the slice has been executed on a card.

# RUN 25 — the prompt-attention slice: byte ledger and filed predictions (BEFORE any measurement)

## Supervisor rulings on record (this run)

- **Ruling B: (b) now, (a) as the funded follow-up.** For this milestone the partner gains its KV span and the
  LEAD's pools stay whole. Decode stays untouched — that constraint outranks the tier cost, because decode is the
  campaign's working win and option (a) would touch session carving, the prompt cache, KV streaming and decode's
  pools. Option (a) — narrowing the lead's pools to its shard, freeing ~1.5 GiB to restore the partner's tier — is
  **filed as the slice-success contingency**, funded only if the slice clears its ship bar. (The 1.2 GiB / 7.6%
  shrink this ruling was struck against, and the 0.6 GB / 3-layer figure in the delegating brief before it, are
  both superseded by the sizer: see the tier table in ledger §2e.)
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

**The gate is open at 1 and 2** — since `2ea9a1ff`, not before. `--exp tp_prompt_attn=1` arms the serialized slice,
`=2` the pipelined one, both at `Prefill::set_peer`; 0 (the default) leaves the path byte-identical — the flag is
read once, on the peer arm's setup, and no other code observes it. Anything else is refused with both gates named.
It is the prompt path's only flag: decode's seam (`qhs_peer_item()`, §4) stays shut.

**The `=2` window measured nothing, because `=2` never ran** (`e2a7f0c7`, `2ea9a1ff`). The boot died into the
`set_peer` refusal — the gate still accepted only 1, though `00d2732a`'s message said otherwise and this file
repeated it — and the caller treated a refusal as benign: warn, `err.clear()`, serve. So the boot kept the
partner's ~2 GiB span/wire/scratch reserve (taken at TIER-SIZING time, from the same flag) and lost the peer share
`set_peer` never installed. What that reads as, all four markers: **1,454 t/s** (vs 1,833 at =1, 2,004 baseline)
with ttft 236 s; chunk lines **`no peer -1/0`** and **`no peer -1/2`** with lead-experts 125–162 ms and peer-wait
0.0; decode cached turn **28.5 t/s** (vs 61.6 at =1, 77–82 gate-off). **None of this is the pipeline's number** —
the =2 predictions below are still unmeasured, and the pair line's missing `pipelined` marker was the thread that
pulled it loose. A refusal now fails the boot (`peer_share_refusal_is_fatal()`); a crippled pair cannot be served.

Gates, current: build green under `-j4`; `ctest -L cpu` **68/69** with the count verified at 69 — only the
pre-existing `expert_multi_test` (no AVX512-VNNI/VBMI here); boundary and stderr checks ok. `:8000` never
contacted; `=2` numbers stay **TODO-for-the-window**.

**Three gated boots refused, each one step further than the last** — pack arithmetic (no canonical bytes; the pair
packing the whole model rather than the stage; ledger §2k), then the event contexts: boot 3 brought the pair UP on
both stages and died at the FIRST publish of a 122k prompt on `cudaEventRecord(qs.ev_in, cs)`, because `ev_in`/
`ev_red` lived on the PARTNER's context while the LEAD's stream records them. Each event now lives where it is
recorded and both refusals print the CUDA error and the bytes queued (`22cc023e`). The passing 8-token probe proves
less than it looks like: nothing logs that a chunk was sliced.

**The serialized slice (`b4cc1f70`, `e786401c`)**: `peer_qsa_publish()` ships the lead's `mixed_h` / `q_idx` /
`idx_raw` — and the chunk's step counts once — into the wire set on the LEAD's compute stream and records
`ev_in[wire]`; the lead drives `peer_qsa_layer()` per QSA layer at the stage's ordinal, wire set alternating, the
partner's card current for its products; `peer_qsa_finish()` multiplies each card's gated rows by its OWN packed
columns and issues one `tp2_all_reduce()` per rank. A chunk wider than the arm's buffers is not sliced at all,
decided once per layer so no layer can half-slice.
Funding: `Prefill::lead_qsa_bytes()` prices what the slice takes on the LEAD's card (its side of the pair's
staging, its packed columns), `generate.cpp` holds it back from each stage lead's tier, and `set_peer` refuses a
split whose lead side exceeds the pinned symmetric cut by naming both figures. `fits_within` still guards the
partner's side.

**Two events, because the collective has no back-pressure.** `Tp2Pair` counts its collectives, it does not credit
the staging back, so a rank a layer ahead can overwrite the peer's staging — or republish a wire set — while the
peer is still reading it. The lead's stream now waits on `ev_out` (the partner's layer is done) before the next
publish, and the partner's stream waits on `ev_red` (the lead's add is done) before its next push; every wait is
on work enqueued in an earlier layer, so the chain cannot cycle. That is what *serialized* means here.

**COMMIT 2 (the pipelined variant) is landed and unmeasured.** Three moves, one commit each:

- `00d2732a` — **`=2` exists.** The serialized mode's end-of-layer wait was the only thing making the two wire sets
  and the single staging safe, so the mode now supplies what that wait was supplying: `ev_in`/`ev_out`/`ev_red` are
  arrays indexed like the wire (`qsa_ord % kWireSets`) — one object cannot describe two layers in flight — and
  `part_f32` goes to two sets, funded by the same walk that prices the tier (`staging_sets`, read from the flag
  `set_peer` refuses on, so the reserve and the arm cannot disagree). The lead waits before a publish on the
  partner's `ev_out` for the layer that last READ that set (two back), and the partner's product for l+1 waits on
  the lead's add of l-2. **No wait after the arm**: the cards compute a layer's halves at once and meet at the
  reduce, the one rendezvous the mode keeps. The bring-up line says `pipelined, 2 o_proj staging sets`.
- `ed7d4d33` — **the lead's attention is halved.** `qsa_prompt_attn_batch` takes a KV-group window (`kvh0`,
  `groups`): whole groups only, pools and pitch still the model's, so the group index moves the KV rows and the
  heads that read them together. Ruling B stands — the lead's pools were never narrowed, and option (a) stayed the
  funded contingency it was. The parity case pins a group's rows bitwise against the whole run's AND every other
  head's row still holding its sentinel; the microbench line reads **2.00x** for one of two groups.
- `79fe41e3` — **a refusal fails the request, not the engine.** The arm's guards set
  `Prefill::prompt_attention_refused()`; the request loop prints `ERR` and stays up (the RAM guard's protocol),
  while a poisoned context (`cuda_context_poisoned()`: illegal address/instruction, launch failure, assert …) still
  leaves by `_Exit` whatever refused.

**Findings the window must read rather than inherit:**

- **The reduce leg crosses as FP32, not the ledger's BF16.** `tp2_all_reduce()` pushes `rows x stride` FLOATS, so
  the return leg is **83.9 MB a QSA layer each way** at an 8192-token chunk, not the 41.9 MB §2a files; nothing
  writes `qs.part_out`. The pair's real traffic is ~190 MB a QSA layer, and the gates are graded against that — or
  a BF16 push lands first (a new kernel: parity plus a microbench line, per AGENTS.md). Both priced in the ledger;
  BF16 is authorized only if the FP32 wire does not hide.
- **`span_valid` advances with the prompt** (to `p0 + T`, after a fully sliced chunk's reduce was proved whole), and
  `set_peer_qsa_lead` has a call site, so a second prompt on a session that decoded imports its gap.

**What `=2` is predicted to do, per QSA layer — STILL UNMEASURED: the one =2 boot died in the gate above, so no
number below has been read off a card** (baseline: run 25's `=1` decomposition — peer-wait 67.7-80.3 ms, lead attn
+13-15 ms over the ~26 ms it should cost, 6 QSA layers a stage, two stages):

- **peer-wait → toward the wire floor.** The serialized mode paid the partner's whole layer on the lead's stream;
  =2 waits only on a set two layers back, so the two halves of a layer's attention overlap. Filed as ~47 ms/layer
  of FP32 wire unhidden (3.96 GiB/s staggered face), ~24 with a BF16 return. If the term does NOT collapse, the
  suspect is the publish still sitting on `m.cs` — moving it to its own copy stream is the next lever, not built.
- **lead attn → ~13-15 ms (halved).** The window runs one of two KV groups; the parity case's own microbench says
  2.00x. The lead's `gate_attn` halves with it; its `attn_q` projection and `split_q`/`rms`/`rope` still cover all
  24 heads — the next thing worth halving, and not built.
- **decode stays the tier's problem**: the flag costs the partner ~2.09 GiB (=1's 2.01 plus the second staging) and
  the lead 116 MiB, and run 25 measured -20% decode for that residency. A prompt-path flag; not for serving.
- **net**: the +10 abort / +20 ship bar grades the per-layer pair against EP+PP-without-the-slice on the same boot;
  the filed `+18/+15 ms` a layer-pair stands as the **no-extra-eviction upper bound**, its instrument (the two row
  rates and `blob_mb_ms`, read before and after the slice on one boot) and the STOP rule are ledger §3.

**The boot lines that say the slice is on** (both modes print these; the first must say `pipelined` for `=2` — if
it says `serialized` the flag never reached `set_peer`, and the boot is not the mode you asked for):

    grep "peer prompt-attention pair CUDA"     # pair open, staging floats, columns, heads, MODE and staging sets
    grep "peer prompt-attention arm, lead card" # the lead's reserve, the split, and the chunk ceiling
    grep "peer prompt-attention arm:"           # the shard projections copied resident on the partner card
    grep "peer chunk at token" | head -1        # must read `mmq/p2p peer D/P`, NEVER `no peer -1/0`

The last grep is the one the =2 window needed: `no peer -1/0` means `PeerPrefill` never went live, so every
partner-held expert row fell to the lead's link and no per-layer term on that line means anything.

What boot 3 read off, the first measurement of any of it: staging **20 971 520 floats = 8192 x 2560**, so the
wire's funded width and the publisher's `rows_max` are one number as its check assumes; **79 MiB** of shard
projections per partner (the ledger's 80.0 MiB truncated); **116 MiB** off each stage lead's tier. The partner's
card delta is not that — the gate-off boot also differed in its RAM split — so ~2.01 GiB stays arithmetic.
## 0. Rulings taken mid-run (the supervisor's, applied to this file)
**Branch doctrine (supervisor):** *cases unrun are cases that have not passed.* A GPU-labelled test that
has never executed is not evidence. Adopted after the first execution of `tp2_head_split_parity`
invalidated a kernel the head-split audit had explicitly cleared — one run, one real bug.

The peer keeps its own KV span, so it can be stale: it only ever sees the chunks it is handed. Every cell the LEAD
appends while the peer is out of the loop — decode's appends (decode stays collective-free, RUN 24 verdict 2, so
nothing tells the peer), cached-prefix reuse (`session_copy_state`), `prompt_session_clear`, park/resume,
`Prefill::set_session` — leaves the peer's span and its indexer pooled table describing cells it never wrote. The
selection covers `[0, p0)`, so a stale span is not a crash: it is plausible tokens from garbage, the RUN 17 class
this branch refuses everywhere else. Ruled:

- **P1 approved.** Per stage, track `span_valid` = the cell count the peer's pools and indexer pooled table
  provably hold. At a prompt run's start, P2P-copy the peer's KV-head slice and its indexer pooled rows for
  `[span_valid, p0)` from the lead's pools, on the peer's own copy stream; the peer then appends `[p0, p0+T)`
  itself. The figures publish on the boot and chunk lines — the copy must be measurable, not assumed.
- **P2 approved.** Invalidate `span_valid` to the true value at every lead-side mutation the peer
  cannot see: both `session_copy_state` sites, `prompt_session_clear`, `Prefill::set_session`,
  park/resume, and each decode window's end. Only has to be right under the gate; off stays
  byte-identical.
- **P3, with the pricing correction applied below** in §2b.
- **The FP4 planes are NOT an attention input.** q/k/v must derive from the same source the lead's attention
  consumed, within the kernel's 2e-3 relative tolerance; the planes carry ~1e-2-class expert-quantization error,
  two orders outside, so q-from-planes fails parity in the same class as a stale span. They stay the expert leg's
  input only (RUN 19 unchanged). See §2a for the mix that survives this.

## 2a. The mix, re-filed under the parity invariant

The ruled invariant decides the mix: **the peer's attention must consume the same q/k/v sources as the lead's,
within the kernel's 2e-3 relative tolerance.** Applying it to the two candidate shapes:

- **q must come from the FP16 rows the lead's own `attn_q` consumed** — `m.mixed_h` (the `native_proj` input), not
  BF16 and not the FP4 planes. BF16 has an 8-bit mantissa (~2e-3 relative), exactly *on* the bound; the lead's own
  FP16 image has 11 bits (~5e-4) and, shipped as the same bits the lead read, makes the peer's q *its own half of
  the lead's q bitwise*. Same 41.9 MB a layer, strictly better parity.
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

## 2c. The span is SIZED, by the walk that allocates it

`Prefill::PeerQsaSpan` (`include/basalt/prefill/prefill.hpp`, implemented beside `Prefill::peer_bytes`) prices
what the partner's card must hold resident for its half of the attention. One count walk produces both the reserve
and, later, the carve, so the tier cannot be sized for a different amount than `set_peer` allocates; `peer_bytes`
takes the span and the call sites pass one (`peer_prompt_bytes` / `pair_prompt_bytes`, generate.cpp), and the lead
card's side of the same arithmetic is `Prefill::lead_qsa_bytes`.

Pinned by `tests/cuda/peer_qsa_span_test.cpp` (CPU-labelled, no device touched), which prints what it measures and
asserts the units explicitly — wire figures are 10^6 bytes, tier figures 2^30, so a future comparison cannot mix
them. **The current figures are the ledger's §2k** (span 0.96 + wire 0.20 + scratch 0.64 + weights 0.08 + columns
0.04 + gemm 0.09 = **~2.01 GiB int8** at `--kv int8`, plus 83.9 MiB for the pipelined second staging set); the 1.2 /
1.58 GiB this section first filed are superseded there, and their lesson stands: the reserve is span **+ wire + the
partner's own scratch**, never the span alone. And because the share flags make an uneven cut live: it is priced,
its scratch grows with its heads, and a cut straddling a KV group is *priced* but refused for the kernels by
`qsa_shard_whole_groups` — sizing is not where a bad ratio gets silently dropped.

## 2b. The span catch-up, priced per TURN (P3, with the correction applied)

The catch-up is **once per prompt run**, covering `[span_valid, p0)` — not per layer, not per chunk. It moves the
peer's KV slice plus the indexer pooled rows it must also hold:

    KV       6 layers x cells x 256 dim x 2 (K,V) x 1 B (int8)  = 134.2 MB x 6 =  805 MB at 262144 cells
    indexer  6 layers x (cells/idx_block + 2) x 128 dim x 4 B   =  33.6 MB x 6 =  201 MB
                                                            total 1.006 GiB
    at the staggered 3.96 GiB/s face                            ~254 ms ONCE per turn
    at the both-pairs 2.00 GiB/s face                           ~503 ms once per turn

**A correction to the ruled figure, in the same spirit as §1a's:** the ruled estimate was ~0.6 GiB / ~150 ms; this
bills K *and* V for all 6 layers plus the indexer's pooled table, which the peer needs to re-run the indexer. The
sizer's 0.96 GiB (§2c) supersedes the 1.006 above, so ~254 ms is a hair pessimistic; it stays as the filed number
until the window measures the copy.

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

If that measured term eats the margin, the P3 fallback verdict stands as filed: with a peer-resident span the
slice cannot pay on this fabric, and peer-reads-the-lead's-pool-over-P2P is not the escape — ~50 GB a chunk
(cap 2048 x 256 dim x 2 B x 8192 queries) and dead.

## The arm's container, allocated by the sizing walk itself

`PeerPrefill::Qsa`: the partner's per-QSA-layer `QsaState` array, two wire sets, its shard-width scratch, its
resident projection pointers, its **own** `Gemm` (ruled: `Gemm::native` dequantizes on the device the product runs
on, so the lead's scratch and cuBLAS workspace cannot cross the link), its stream, and its per-set events.

**The anti-drift property, which is the point of this unit:** the buffers are laid out by ONE function,
`peer_qsa_layout`, and both callers drive it with a different `take` — `make_peer` with `cudaMalloc`, and
`Prefill::PeerQsaSpan::bytes()` with a counting no-op. The tier's reserve and the allocation are therefore the
same code, not two ledgers kept in step by hand — which is also why the pipelined second staging set is pinned by
allocation COUNT in the sizing test, not only in bytes.

## Building this tree after a reboot

See **`progress-tp2slice-build.md`** — the configure line this tree needs (`BUILD_TESTING=ON` or `ctest -L cpu`
finds zero tests and "passes"), and the rule that the count is read before the pass rate means anything.

## Where the arithmetic lives

The byte ledger and filed predictions are in **`progress-tp2slice-ledger.md`** (the span catch-up's per-turn
pricing stays in §2b here); the
kernel and arm pieces — container and sizing walk, resident projections (`peer_qsa_resident_up`), the partner's own
`Gemm`, the shard-width kernels and cache writer, the packed `attn_output` column funding, `peer_qsa_layer()`, and
every hardcoded head count found, fixed and parity-tested — are in **`progress-tp2slice-kernels.md`**. The
headline figures, restated so this file stands alone:

    wire traffic   104.9 MB in a QSA layer    what the +10/+20 gates grade; the RETURN leg is FP32 today (83.9 MB)
    tier reserve   2.01 GiB int8 (=1)          span 0.96 + wire 0.20 + scratch 0.64 + weights 0.08 + cols 0.04 + gemm 0.09
    =2 adds        +83.9 MiB partner           the second `o_proj` staging set (`staging_sets`), priced by the same walk
    span copy      <= 0.96 GiB per TURN        zero on a fresh prompt; ~254 ms worst case
    predictions    =1 measured -8.5%            =2 filed: peer-wait -> ~47 ms wire floor, lead attn ~13-15, +BF16 ~24

## 4. What stays closed

`--exp tp_prompt_attn` accepts **1 (serialized) and 2 (pipelined)**; anything else is refused at
`Prefill::set_peer` rather than run as one of them, because the operator would read the resulting number as the
mode they did not ask for. `qhs_peer_item()` (`src/core/verify.cpp`) still returns false: that is decode's window
item, supposed to stay false for this milestone, and `--exp decode_headsplit=1` still refuses. Decode's paths —
`qhs_peer_item`, the peer window, `dm_peer_kernel` — are untouched by this slice and by commit 2's kernel window
(the prompt-attention kernel has no decode caller).

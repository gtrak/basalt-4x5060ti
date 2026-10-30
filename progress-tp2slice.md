# RUN 25 — the prompt-attention slice: byte ledger and filed predictions (BEFORE any measurement)

## Supervisor rulings on record (this run)

See **`progress-tp2slice-rulings.md`** - Ruling B (the lead's pools stay whole, option (a) the funded contingency), the
decode-window-end ruling and the invariant it rests on, the tier-reserve option (c), and the ship-bar wording.


## Status of this milestone

**The gate is open at 1 and 2** — since `2ea9a1ff`, not before. `--exp tp_prompt_attn=1` arms the serialized slice,
`=2` the pipelined one, both at `Prefill::set_peer`; 0 (the default) leaves the path byte-identical — the flag is
read once, on the peer arm's setup, and no other code observes it. Anything else is refused with both gates named.
It is the prompt path's only flag: decode's seam (`qhs_peer_item()`, §4) stays shut.

**The first `=2` window measured nothing, because `=2` never ran** (`2ea9a1ff`, `e2a7f0c7`): the gate still accepted
only 1 while the commit message claimed 1 and 2, and the caller treated the refusal as benign (`warn; err.clear()`).
The boot therefore kept the partner's ~2 GiB reserve and lost the peer share — **1,454 t/s** with ttft 236 s, chunk
lines **`no peer -1/0`**/**`no peer -1/2`**, lead-experts 125–162 ms, decode cached **28.5 t/s**. A refusal now fails
the boot (`peer_share_refusal_is_fatal()`); a crippled pair cannot be served, and the missing `pipelined` marker is
what pulled the thread.

Gates, current: build green under `-j4`; `ctest -L cpu` **68/69** with the count verified at 69 — only the
pre-existing `expert_multi_test` (no AVX512-VNNI/VBMI here); boundary and stderr checks ok. `:8000` never
contacted; `=2` numbers stay **TODO-for-the-window**.

**`=2` booted and the model went degenerate; the cause was the pair's collective, not the attention (`e0d94223`).**
Three boots, one cached prompt: gate-off normal, `=1` normal, `=2` replied `!` 256 times at acc=100%, and the
concurrent turn read acc=97% against **hit=44.3%** where 76-94% is normal — a distribution self-consistent and
wrong, i.e. a shifted sum rather than garbage. `Tp2Pair` had **one landing region per card** and its flag protocol
credits nothing back, so a peer's next push into a region is safe only while some other wait keeps a rank a whole
collective behind its peer — the wait `=2` removes. Layer k's o_proj came out as
`lead_partial(k) + partner_partial(k+1)` on every QSA layer of the chunk. The claim here that `ev_out`/`ev_red`
closed the staging race was **wrong**: they guard the arm's `part_f32` sets, not the pair's landing buffer. Fixed by
giving the pair `sets` regions (2 under `=2`, 1 elsewhere, decode untouched) — the pair's own flag order gaps two
uses of a region by `sets` collectives, so one per in-flight collective is exactly enough. The brief's first two
suspects are ruled out by the differential: the head→column mapping and the halved attention are shared with the
normal `=1` boot, and `peer_qsa_finish` refuses columns that do not tile the axis. **CPU-proven**:
`peer_qsa_span_test` replays the protocol (lead publish/push/flag/add, partner push/flag/add, the `ev_in` and `=1`
`ev_out` waits) against a scheduler that starves one rank while the peer runs everything it can — one region
ungated **corrupts at depth 2**, two regions are clean to 12.

**Three early gated boots refused, each one step further than the last** — pack arithmetic (no canonical bytes; the
pair packing the whole model rather than the stage; ledger §2k), then the event contexts: `ev_in`/`ev_red` lived on
the PARTNER's context while the LEAD's stream records them. Each event now lives where it is recorded and both
refusals print the CUDA error and the bytes queued (`22cc023e`). The passing 8-token probe proves less than it looks
like: nothing logs that a chunk was sliced.

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
  heads that read them together. Ruling B stands — option (a) stayed the funded contingency it was. The parity case
  pins a group's rows bitwise against the whole run's AND every other head's row keeping its sentinel; the
  microbench reads **2.00x** for one of two groups.
- `79fe41e3` — **a refusal fails the request, not the engine.** The arm's guards set
  `Prefill::prompt_attention_refused()`; the request loop prints `ERR` and stays up, while a poisoned context
  (`cuda_context_poisoned()`: illegal address/instruction, launch failure, assert …) still leaves by `_Exit`.

**Findings the window must read rather than inherit:**

- **The reduce leg crosses as FP32, not the ledger's BF16.** `tp2_all_reduce()` pushes `rows x stride` FLOATS, so
  the return leg is **83.9 MB a QSA layer each way** at an 8192-token chunk, not the 41.9 MB §2a files; nothing
  writes `qs.part_out`. The pair's real traffic is ~190 MB a QSA layer, and the gates are graded against that — or
  a BF16 push lands first (a new kernel: parity plus a microbench line, per AGENTS.md). Both priced in the ledger;
  BF16 is authorized only if the FP32 wire does not hide.
- **`span_valid` advances with the prompt** (to `p0 + T`, once a sliced chunk's reduce was proved whole), and
  `set_peer_qsa_lead` has a call site, so a second prompt on a decoded session imports its gap.

**What `=2` is predicted to do, per QSA layer — STILL UNMEASURED: the `=2` boots so far died in the gate or in the
collective race above** (run 25's `=1` decomposition: peer-wait 67.7-80.3 ms, lead attn +13-15 ms over the ~26 ms
it should cost, 6 QSA layers a stage, two stages):

- **peer-wait → toward the wire floor.** The serialized mode paid the partner's whole layer on the lead's stream;
  =2 waits only on a set two layers back, so the two halves of a layer's attention overlap. Filed as ~47 ms/layer
  of FP32 wire unhidden (3.96 GiB/s staggered face), ~24 with a BF16 return. If the term does NOT collapse, the
  suspect is the publish still sitting on `m.cs` — moving it to its own copy stream is the next lever, not built.
- **lead attn → ~13-15 ms (halved).** The window runs one of two KV groups; the parity case's own microbench says
  2.00x. The lead's `gate_attn` halves with it; its `attn_q` projection and `split_q`/`rms`/`rope` still cover all
  24 heads — the next thing worth halving, and not built.
- **decode stays the tier's problem**: the flag costs the partner ~2.09 GiB (=1's 2.01 plus the second staging) and
  the lead 116 MiB, and run 25 measured -20% decode for that residency. A prompt-path flag; not for serving.
- **the two things `=2` summed wrong, and what they read as.** First the pair's collective landing region: one per
  card, and the collective credits nothing back, so the partner's next push landed in the region the lead's add of
  the previous layer was still reading (`e0d94223`). Then the **shipped step counts**: `qs.steps` was ONE buffer for
  the whole prompt, published on a chunk's first QSA layer. The pipelined gate lets the lead begin the next chunk
  while this card still attends over the last, so the partner selected its blocks - and attended - with the NEXT
  chunk's positions: a wrong top-k, and a plausible but shifted token stream (`!` to the length cap at `acc=100%`,
  a fresh concurrent turn at `hit=44.3%` against the usual 76-94). `=1` could not hit it, which is exactly why `=1`
  read as correct: its end-of-layer wait keeps the lead a layer behind the partner's finish. The step counts are now
  a set of the wire - published with every layer's other inbound bytes, freed by the same `ev_out`, collapsed to set
  0 under the serialized gate the way `part_f32` is - and `replay_steps()` falsifies the old shape: one set corrupts
  at 2 and at 6 layers a chunk, sets of the wire do not, to six chunks deep, and serialized is clean either way.
- **net**: the +10 abort / +20 ship bar grades the per-layer pair against EP+PP-without-the-slice on the same boot;
  the filed `+18/+15 ms` a layer-pair stands as the **no-extra-eviction upper bound**, its instrument (the two row
  rates and `blob_mb_ms`, read before and after the slice on one boot) and the STOP rule are ledger §3.

**The boot lines that say the slice is on** (both modes print these; the first must say `pipelined` for `=2` — if
it says `serialized` the flag never reached `set_peer`, and the boot is not the mode you asked for):

    grep "peer prompt-attention pair CUDA"     # ... MODE and BOTH staging figures: `pipelined, 2 o_proj staging
                                               #   sets, 2 collective staging sets` for =2, `1`/`1` for =1
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

- **P1 approved, landed.** Per stage track `span_valid` = the cell count the peer's pools and indexer pooled table
  provably hold; at a prompt run's start P2P-copy the peer's KV-head slice and indexer rows for `[span_valid, p0)`
  from the lead's pools on the peer's copy stream, then let the peer append `[p0, p0+T)` itself. The figures publish
  on the boot and chunk lines — the copy must be measurable, not assumed.
- **P2 approved, landed.** Invalidate `span_valid` to the true value at every lead-side mutation the peer cannot
  see (`invalidate_peer_qsa_span`'s call sites). Right under the gate is enough; off stays byte-identical.
- **P3**, priced per turn in §2b.
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
0.04 + gemm 0.09 = **~2.01 GiB int8** at `--kv int8`); under `=2` add 83.9 MiB for the second `o_proj` staging set,
80.0 MiB for the pair's second collective landing region and 128 KiB for the second shipped step-count set, both
cards, all read from one figure
(`peer_qsa_sets()`). The 1.2 / 1.58 GiB this section first filed are superseded there, and their lesson stands: the
reserve is span **+ wire + the partner's own scratch**, never the span alone. And because the share flags make an
uneven cut live: it is priced, its scratch grows with its heads, and a cut straddling a KV group is *priced* but
refused for the kernels by `qsa_shard_whole_groups` — sizing is not where a bad ratio gets silently dropped.

## 2b. The span catch-up, priced per TURN (P3, with the correction applied)

The catch-up is **once per prompt run**, covering `[span_valid, p0)` — not per layer, not per chunk. It moves the
peer's KV slice plus the indexer pooled rows it must also hold:

    KV       6 layers x cells x 256 dim x 2 (K,V) x 1 B (int8)  = 134.2 MB x 6 =  805 MB at 262144 cells
    indexer  6 layers x (cells/idx_block + 2) x 128 dim x 4 B   =  33.6 MB x 6 =  201 MB
                                                            total 1.006 GiB
    at the staggered 3.96 GiB/s face                            ~254 ms ONCE per turn

**A correction to the ruled figure, in the same spirit as §1a's:** the ruled estimate was ~0.6 GiB / ~150 ms; this
bills K *and* V for all 6 layers plus the indexer's pooled table, which the peer needs to re-run the indexer. The
sizer's 0.96 GiB (§2c) supersedes the 1.006 above, so ~254 ms is a hair pessimistic.

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

**The anti-drift property:** the buffers are laid out by ONE function, `peer_qsa_layout`, driven by `make_peer` with
`cudaMalloc` and by `PeerQsaSpan::bytes()` with a counting no-op — the tier's reserve and the allocation are the
same code, not two ledgers kept in step by hand, which is also why the pipelined staging sets are pinned by
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

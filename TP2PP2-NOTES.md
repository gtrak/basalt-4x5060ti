# TP=2/PP=2 work notes

## 2026-10-29 00:30 — reviewer handoff (run 1 → run 2)

Run 1 (2faf0aa1) was stopped deliberately at ~00:25 to escape a too-short
run deadline; it made NO edits and NO commits (worktree was pristine). It
spent its time reading and surveyed these areas (verify them, but start
warm — do not re-read blind):

- the expert cache sizing loop + budget contract (`auto_cache`, `shrink_to`,
  the commit-charge failure path in generate.cpp) — M1 must not weaken it
- the peer-expert machinery: `xpeers` creation/wiring, the refill/verify
  path (`refill_`, `verify_slot`, `blob_bytes`, peer experts d2d) — this is
  the pattern to reuse for EP-within-pair routing
- the residency tables + RAM complement — the M1 invariant: no expert may
  become unreachable when each card holds half its stage's experts

Milestones per TP2PP2-BRIEF.md: M1 (stage-pair scaffold + EP-within-pair,
flag-gated, boot log names the topology) → M2 (attention head-split +
2-way all-reduce helper) → M3 → M4. Commit each milestone; append findings
here as you go.

Standing constraints (unchanged): production server is LIVE — never touch
it, never run basalt-ctl, never bind :8000; builds `-j4` max in the
worktree's own build/; house commit style per `git log --oneline -15` main.

### Run 1 full transcript

TP2PP2-PREVIOUS-LOG.txt (in this directory) is run 1's COMPLETE console
transcript — every file it read (with the actual code passages: budget
contract, auto_cache sizing, xpeers wiring, peer refill/verify, residency
tables) and its running narration. Read it in full before implementing;
it replaces re-exploring those areas from scratch.

---

## Run 2 — M1 (stage-pair scaffold + EP-within-pair)

### What was built

`--tp-size 2` (flag) / `BASALT_TP2=1` (environment, the launcher form) turns each
stage of a layer split into a PAIR of cards. With `--gpu 0,1,2,3`: stage A = layers
0-23 on CUDA0 (lead) + CUDA1 (partner), stage B = layers 24-47 on CUDA2 + CUDA3.
Default `--tp-size 1` is the plain split, byte for byte as before; nothing in the
tp2 path runs unless the size is 2.

- Stage cards are **assigned, not chosen**: stage `s` leads on CUDA `2s`, its partner
  is CUDA `2s+1`. `--split-device` is refused under tp2 for that reason.
- The **expert split within a pair is by expert-id parity**: the lead holds the even
  ids of its stage's layers, the partner the odd ones. Every profiled pair lands on
  exactly one card, each card's tier is half the depth, and each card still fills its
  tier hottest-first from its stage's own ranking (`pair_ranked` in generate.cpp).
- The **partner card is a `PeerExperts` tier** attached to its stage's verifier via
  the existing `Verifier::set_peer()` peer window, and to its stage's prompt path via
  the existing `Prefill::set_ring_remotes()` d2d path. No new routing was written.
  `GpuStage` gained `pair_dev` + `pair`.
- Boot log: `tp2/pp2: stages A(layers 0-23)=[CUDA0,CUDA1] B=[CUDA2,CUDA3]`, plus one
  `tp2: CUDA%d holds N of the M ranked pairs ...` line per partner card.

### Things the code told us (not obvious from the brief)

1. **`--layer-split` and `--split-device` are not CLI flags at this HEAD.** They were
   removed by `30f0596d` ("gpu: one setting picks the cards"); the split is entered
   only through the price model (`plan_multi_gpu` picking `LayerSplit`) or
   `BASALT_FORCE_SPLIT=1`, both of which set `o.layer_split = "auto"` internally.
   `Options::split_device` is now dead (nothing assigns it). So `--tp-size 2` forces
   the split on in exactly that same place, the way `BASALT_FORCE_SPLIT` does.
2. **Attaching a peer to a verifier silently enables two weight splits.**
   `Verifier::dm_init()` calls `hs_init()` (head split by vocabulary) and `gs_init()`
   (GDN mixer split by key head) whenever `dm_->pdev >= 0`, i.e. whenever *any* peer is
   set. On a partner card that would have loaded half the head and some GDN mixers onto
   the card we just filled with experts, and moved state off the lead card. Added
   `Verifier::set_peer_experts_only(bool)`; the tp2 path sets it, and those two inits
   are then skipped. This is the one engine-surface change outside `generate.cpp`.
3. **The auto layer-split search cannot price a pair** (`predict()` gives stage `i` one
   card's cap), so under tp2 the cut is even: `K = i * n_layers / n_stages` (48 → 24).
   The search is left untouched for the plain split.
4. **The RAM/budget contract is kept, deliberately on the tier-card side.** Partner-held
   bytes are added to `peer_only`/`peer_only_bytes` exactly like the `--peer-device`
   tier cards' bytes, so `gpu_held = min(expert_total_bytes, cache_fill_bytes +
   peer_only_bytes)` keeps its shape and the resident complement stays the same size as
   the plain 4-card split's (an auto-sized tier does not suddenly want ~2x the RAM).
   Consequences, stated plainly:
   - a partner-held expert has **no pinned host copy**; it is served by the partner's
     peer window (device plan on), by a d2d copy in the prompt path (ring remotes), or
   - if the device-planned MoE tail is off (non-native pack, `--remote-expert-opt`,
     `--no-pool`...) the CPU path reads those same bytes **through the file mapping**:
     correct, slow. `FileExpertSource::blob()` falls back to `mapped_blob()`.
   - the adaptive swapper cannot move a partner expert: it is not `servable()` (not
     RAM-resident) and never appears in `host_res`, so swaps stay within the lead
     caches. `PeerExperts::adapt()` is never called on a partner tier.
   - the alternative (counting partner bytes as CPU-side) was rejected: it doubles the
     complement against the baseline run and would make the budget contract refuse to
     boot on this box, i.e. unmeasurable.
5. `set_peer_portable(true)` already covers a layer split, so the partner card's
   mapped host rows are allocated Portable ✓; peer access is enabled for every ordered
   pair of chosen cards ✓ (tp2 additionally refuses when `gpu_peer_access()` is false).
6. `PeerExperts::prepare()` requires `device >= 1`; partner cards are CUDA 1,3,.. so
   that holds by construction.

### What M1 does NOT do (honest list)

- **Nothing is measured.** No runtime verification at all: the production server is
  live, so no engine start, no restart, no `basalt-ctl`. Verification was: build green
  (`cmake --build build -j4`), `ctest` unchanged in its pass/fail set (the failures are
  this box's environment: GPU tests hit `flush: out of memory` beside the live server,
  `expert_multi_test` needs AVX512 this CPU lacks; both pass/fail identically when run
  alone), `tools/check_platform_boundary.sh` ok, `tools/check_stderr_logging.sh` ok,
  and `--help` / `--tp-size 3` / `--gpu-plan` CLI checks.
- **Attention head-split within the pair: not done** (M2). All attention and dense
  compute for a stage still runs on its lead card.
- **No collectives**: no `tp2_all_reduce` helper yet (M2/M3).
- **Partner card holds no state** and no session; it is experts only. The slot model's
  `slot_cap` semantics are unchanged (a slot's conversation still has its state on the
  two stage lead cards).
- **Pipelined windows**: both chains (`ver`/`ver_b`, `gs.ver`/`gs.ver_b`) are given the
  same partner tier. Each verifier owns its own peer window, and the tier is read-only
  to them, so this is safe — but with both chains live the two peer kernels each take
  the full `dec_moe_peer_blocks()` on one card, i.e. they oversubscribe it. Unmeasured.
- `--expert-cache N` (an explicit count) is allowed and simply caps the lead's tier;
  the partner sizes from its own card's free memory.
- Numerics: expert rows are computed by the same kernels from the same activations on
  either card of the pair (`peer_experts.hpp` states the bit-identity), so the placement
  itself is not a numerics change. If the device plan is off and the file fallback is
  taken, the CPU's rounding differs from the GPU's — the already-documented
  resident/CPU-vs-GPU caveat, not a new one. No numerics gate was run (unmeasured).

### Open questions for the reviewer

1. Is counting partner-held bytes in `peer_only_bytes` the right side of the budget
   contract, or should the resident complement stay complete (at ~2x RAM)?
2. Should `hs_init()`/`gs_init()` be re-allowed for the *last* stage's partner card
   (it has VRAM to spare only if its expert tier is small)? Left off for M1.
3. Oversubscription of the partner card by the two pipelined chains: divide
   `pblocks` by the number of chains, or refuse `--tp-size 2` with pipelined windows?
4. `--no-pool` + tp2 was not reasoned about end to end; with the device plan off it
   reads partner experts from the file. Refuse the combination?
5. M2's head-split needs a decision the brief already made (KV head-sharded, 1/4 KV per
   card): that touches the session carve (`session_bytes(..., lb, le)`), which today is
   per stage, not per card. Flagging it early.

### Runbook for the eventual measurement run (NOT executed here)

```
# from this worktree, after the maintainer stops the live server
./bin/basalt --serve --model <pack> --gpu 0,1,2,3 --tp-size 2 \
    --resident-experts --resident-budget-gib <as basalt-ctl picks> \
    --max-context 262144 --parallel 4 ...        # the rest as basalt-ctl builds it
```
Expected start lines: `tp2: each layer split stage is a pair of cards`,
`tp2: 2 stages of 24 layers each, K=24`, `tp2/pp2: stages A(...)=[CUDA0,CUDA1]
B(...)=[CUDA2,CUDA3]`, then two `tp2: CUDA%d holds ...` lines. Compare against the same
command with `--tp-size 1` (the plain 4-card split, which is what BASALT_FORCE_SPLIT
gives today).

### M1 addendum (same run): the split is now a tested header

`pair_holds` / `pair_ranked` / the stage card assignment moved out of the `main` lambdas into
`include/basalt/program/tp2_pair.hpp` (pure, inline), and `tests/program/tp2_pair_test.cpp`
(`basalt_cpu_test`, runs here: passes) is the oracle for the invariant the whole design sits
on: of a pair exactly one card holds each expert, the halves are disjoint and cover the
ranking, they keep its order, they balance to one expert, and the stage cards come in pairs.
Mutation-checked: perturbing the parity expression fails the test.

While reviewing the diff for runnability (it cannot be run), three things were fixed:

- `--tp-size 2` no longer depends on the price model having chosen an expert-tier card: the
  old form turned the split on only `&& o.peer_device >= 1`, so a 4-card run whose plan
  settled without a peer tier would have refused with "needs a layer split".
- the mmap-mode sizing (`o.mmap_experts`, generate.cpp ~3427) counts CUDA0's cache as its
  lead half, like every other sizing site.
- `pair_holds(e, 1, 1)` (a rank the stage does not have) is false rather than true.

Two known conservative corners, both left alone deliberately:
- the mmap-mode estimate does not count the partner cards' room, so it can warn "the rest is
  read from the SSD" for a placement that actually fits; the resident path (what the server
  uses) counts them.
- with no profile section, CUDA0's cache is still sized from the whole card's free memory
  while only the even ids will ever be admitted to it. The pack has a profile; the plain
  split has the same shape of waste when a profile is missing.

### M2 (attention head-split + 2-way all-reduce): assessed, not attempted

The brief makes M2 conditional on the code cooperating. Reading the seams, it does not, and
the missing one is the session carve:

- `session_bytes`/`session_init` (src/core/session.cpp:48) take a *layer range* only. There
  is no head dimension anywhere in the carve, so "¼ of the full KV per card" needs a shard
  parameter threaded through `session_init`, `session_copy_state` (the slot copy and the
  paging paths), `prompt_session_bytes`/`_carve`, `qsa_state_bytes`, the QSA KV pools and
  page tables, the conversation checkpoints, and the slot carve arithmetic in generate.cpp.
  That is a state-format change, which per AGENTS.md owes a parity test, a profile and a
  `docs/precision/` update - none of which can be produced without running the engine.
- The arithmetic of the split itself is fine (24 query heads / 2 KV heads, head_dim 256: 12
  query heads and 1 KV head per card), but the indexer (`idx_n_head` 4, `idx_dim` 128) and
  the selection (`idx_top_k` 2048) are per query head and have to move with the heads
  through `qsa_dec_attn_seg` / `qsa_decode_attn`, which take the whole `QsaShapes` plus the
  layer's pools.
- The GDN side has the right precedent - `gdn_head_split.hpp` + `Verifier::gs_init` already
  split key heads onto a peer card - but for one layer group on a tier card, not for 36
  layers inside a stage. Reusing it means changing what `gs_init` means, which tp2 currently
  switches off (see `set_peer_experts_only`).
- Decisive for this run: the GPU test suite does not run while the server is live (17 of the
  35 parity tests fail with `cudaMalloc: out of memory` even when run one at a time), so a
  new collective kernel would land with no parity test and no microbench line - which
  AGENTS.md requires for every kernel, and which an exploratory branch should not wave
  through. For the same reason the `tp2_all_reduce` helper was left out: uncalled and
  unrunnable, it would be dead code a reviewer has to read and nobody can execute.

What M2 needs before it is written: the engine running (server stopped), so the head-shard
can be checked against `qsa_*_parity` and a greedy-identity decode A/B, and so the split's
own cost is measurable against the plain 4-card split. The `docs/configuration/gpus.md`
paragraph says so in the open.

## Run 3 — M2 (the pair's head split), engine never started

Constraints kept: the live server was never touched, nothing was restarted, no engine
process was launched, builds were `cmake --build build -j4` in this worktree's own
`build/`.  Every GPU number below is a development A/B on this box (4x RTX 5060 Ti,
180 W limit each), not a headline.

### Landed, four commits

- `c10fb57f kernels: tp2_all_reduce - a 2-way all-reduce, one push per rank, tested against its reference`
  `include/basalt/kernels/tp2_collective.hpp`, `src/kernels/cuda/tp2_collective.cu`,
  `tests/kernels/tp2_allreduce_parity.cpp`.  A pair's two cards each push their partial
  into the other's staging buffer and add `this + peer` in that order, which the test
  pins bitwise against `tp2_all_reduce_ref`.  Refusals (odd rank, wrong size, no peer
  access, over-long vector) return false; a stalled partner trips a mapped host timeout
  word rather than hanging the caller.  Skips with rc 77 below two peer-accessible cards.
- `6bc43264 core: a session can be carved for one card's heads - the pair's QSA shard`
  `include/basalt/core/qsa_head_split.hpp` plus shard parameters on `qsa_state_bytes`,
  `qsa_state_init`, `qsa_prompt_state_bytes/_carve`, `qsa_buffers_bytes/_init`,
  `session_bytes`, `session_init`, `prompt_session_bytes/_carve`, and `heads` recorded on
  `QsaState`, `QsaBuffers`, `SessionState`.  Every parameter defaults to the whole layer,
  so an unsplit session's figures are unchanged.  New: `qsa_real_shapes(ModelGeometry)` is
  the one place a geometry becomes kernel shapes.
- `6c8d7edb kernels: the pair's head split is bitwise-exact, shown where the engine will run it`
  `tests/kernels/tp2_head_split_parity.cpp`, and `tests/core/qsa_head_split_test.cpp`
  registered (it had never been built) with a weight-slice section.
- `a65ec93a core: a head shard may be sized but is refused at every QSA executor`
  `qsa_shard_runs()` gating `qsa_layer` and `qsa_kv_resolve`, and `qsa_layer` checking
  its state's shard against its buffers'.

### Proven

1. **The head split is exact, at the entries the engine uses.** Each card's shard of the
   QSA decode attention is the SAME BITS as that card's rows of the whole run:
   `kv_gather_step`, `qsa_attend_step`, `qsa_decode_attn_step`, `qsa_decode_attn_batch`
   (n_q=8, per-query selections) and the gate, at 24 query / 2 KV heads, dim 256, shard
   12 / 1, permuted page table, 4096 cells, 2051 selected.  0 failures.
   Mutation-checked: rank 0's shard pointed at rank 1's KV head fails the gather on the
   first element; comparing rank 1 against rank 0's rows fails every rank-1 check.
2. **The all-reduce matches its CPU reference bitwise**, 8x2048 fp32 at 21.7 us per
   collective, 2.89 GiB pushed per rank per second (this box, both cards idle-ish while
   the server holds VRAM).
3. **A pair card's session really is smaller, by exactly the QSA part.**
   `session_bytes(whole) - session_bytes(shard)` equals the QSA state and scratch deltas
   to within the carve's 256 B rounding, for the whole model and for 24-layer stage
   ranges; the two cards of a pair carve equal sizes.  At 131072 cells: 3.339 GiB whole,
   1.837 GiB per card of a pair.
4. **A shard's weights are the whole layer's bytes.** `gdn_gather_cols` over
   `qsa_out_cols` gives each card the whole o_proj row at its own heads' columns, for
   32/64/128/256-column blocks; `gdn_gather_rows` over `qsa_kv_rows` / `qsa_q_rows` the
   same for K, V and q; a head_dim that will not cut on blocks is refused.  Built and
   checked under `-fsanitize=address`.
5. **The split invariant** (`tp2_pair_test`, from run 2) and the CPU label still pass:
   1 failure out of 64, `expert_multi_test`, this CPU lacking AVX512-VNNI/VBMI.
   `tools/check_platform_boundary.sh` and `tools/check_stderr_logging.sh`: ok.

### What the code said

- `qsa_decode_attn.cu` hard-codes `G = 12` (query heads per KV head) and asserts
  `n_head == G * n_head_kv`.  A KV-group shard satisfies it exactly (12 == 12 x 1), which
  is why the shard needs no kernel change.  A shard that split the query axis instead
  would trip that assert - the design and the kernel agree, by luck as much as intent.
- The batch entry wants `q` as `[n_q, n_head, head_dim]`, and its per-query stride is
  `n_head_kv * G * head_dim`.  A single-query q buffer handed to it makes queries 1..n_q
  read past the end; that is how the window section first "failed", before any kernel was
  suspected.  Any q slice for a shard must be packed at the shard's width.
- `qsa_head_split_test.cpp` existed but was never in `CMakeLists.txt`, so it had never
  been compiled.  Once built it failed two assertions and, in code I had written earlier
  in this session, went out of bounds in a `gdn_gather_rows` call.  Unrun tests are not
  tests.
- `session_bytes` rounds its total, so an identity between two session figures is exact
  only to 256 B; the test says so with a named slack rather than a shrug.

### What is NOT tested — the whole list

- **Nothing has ever run a shard.** No session with a shard is constructed by any caller;
  `qsa_layer` and `qsa_kv_resolve` refuse one (`a65ec93a`).  The plumbing is inert.
- The `Tp2Pair` device path was exercised only by its own test, on two cards, in one
  process, in lockstep, with no other load.  It has never been used by the engine, never
  been used from two threads, never been used while the server held VRAM, and its timeout
  path has only been provoked by the test stalling a partner on purpose.
- The parity tests and microbenches that the quality gate wants for a changed kernel were
  not run as a suite: the live server holds ~15 GiB per card, and `qsa_seg_parity`
  OOMs here at `HEAD` as well as at this branch (verified by stashing and rebuilding).
  `qsa_parity`, `qsa_dec_attn_parity`, `qsa_prompt_attn_parity`, `gdn_parity`,
  `prompt_split_state_test`, `conversation_snapshot_test` do pass individually.
- No PPL/KLD, no greedy-decode A/B, no profile, no prefill or decode window measurement.
  AGENTS.md requires those for anything that can alter precision; **the row split
  (dense/FFN partials summed by the all-reduce) changes reduction order and belongs
  under that gate.  The head split does not, and the parity test above is its evidence.**
- M3 (dense/FFN row-split inside the pair) is untouched.  The seams allow it -
  `native_mmvq` takes `n_out`, and `gdn_gather_rows` already packs row ranges for the GDN
  split - but its evidence is the numerics gate, which needs the engine.
- Windows: `tp2_collective.cu` is CUDA-only like its neighbours, no platform seam added,
  `check_platform_boundary` passes; CI's Windows CPU build has not been run here.

### What must happen before a shard can run

In rough order, with the seam each one is at:

1. Pack the shard's weights at stage setup, reusing `gdn_gather_rows` / `gdn_gather_cols`
   the way `Verifier::gs_init()` does for the GDN mixer (that function is the precedent
   end to end: pack, upload to the other card, keep both cards' copies).  Needed:
   `attn_q` rows `[q0*2*hd, +nq*2*hd)`, `attn_k`/`attn_v` rows `[kv0*hd, +nkv*hd)`,
   `attn_output` columns `qsa_out_cols()`.  `Verifier::set_peer_experts_only` already
   keeps stage weights off a partner card; this is where they go back on.
2. Give the partner card the block input, and the pair's summed output back to the lead.
   That is the stage handoff in `src/program/generate.cpp` (`bslot_ss`, `set_stage`) plus
   one `Tp2Pair::all_reduce` after the o_proj.  `qsa_layer` would need to write a PARTIAL
   for the o_proj rather than the finished block output.
3. Decide the indexer's fate for real.  Today it stays whole on both cards, which is what
   buys bitwise equality and costs a redundant 4-head projection and top-k per step.
4. Then, and only then, lift `qsa_shard_runs`' refusal, add the shard's parity test
   against the unsplit engine (greedy tokens identical) and the profile line.

### Decisions the reviewer owes this branch

- Whether a shard's `attn_output` partial is summed by `Tp2Pair::all_reduce` (fixed
  order, this + peer) or by the lead card collecting both partials.  The first is what
  the helper does; the second is fewer moving parts but serialises.
- Whether the partner card computes the indexer redundantly (current design, bitwise) or
  the pair shares one selection through a collective (cheaper, needs a tie policy).
- The run-2 questions are still open: partner-held expert bytes in `peer_only_bytes`,
  `hs_init`/`gs_init` on last-stage partners, pipelined-window oversubscription,
  `--no-pool` with tp2.

### For `TP2PP2-MEASURE.md` (the reviewer's checklist), M2 adds

The restart-window checklist stays the reviewer's.  M2 adds four lines to it:

- [ ] `ctest -R "tp2_allreduce_parity|tp2_head_split_parity" --output-on-failure` with the
      cards free - both skip (rc 77) rather than fail without peer access, so a silent
      skip means the topology was wrong, not that the split is fine.
- [ ] greedy-decode A/B: same prompt, `--tp-size 1` four-card layer split vs `--tp-size 2`
      stage pairs, tokens identical.  Until a shard is runnable this compares M1's expert
      split only; it is the regression net for the pair's expert half.
- [ ] one profile line for the QSA decode attention at the real window, whole vs split,
      to check the 1.77x measured here on a single card (two half-width attentions against
      one full one) is paid for by the two cards running at once.
- [ ] if `Tp2Pair` is ever driven from the engine: the timeout path under load, with the
      partner stalled by the scheduler rather than by a test.

### Run 4 handoff (reviewer, 05:20 UTC)

Run 3 (revived, 718b7f74) completed M2b (wired head split: b51af240,
b0dce812, 7a1a512c) and died at its 262k context ceiling mid-M3. Its
UNCOMMITTED M3 work is in the tree: the native-image block width became
type-dependent (qsa_native_block_elems, mirrored against
native_mmvq_weight_bytes with a refuse-on-disagreement cross-check in
qsa_prepare_one), layer.cpp wired to the check, tests extended. NOT
verified: build + tp2_shard_weights_test + tp2_head_split_parity. Run 5:
verify, commit if green, then assess the remaining M3 execution wiring —
small session, work from this file and the diff, NOT from history.

## Run 5 — M3 verification, one fix, and the remaining-wiring assessment (05:15–05:55 UTC)

Constraints kept: the live server was never touched, nothing restarted, no engine process, no
`basalt-ctl`, nothing bound on :8000; builds were `cmake --build build -j4` in this worktree's `build/`.
`tp2_head_split_parity` was run (run 5's task asked for it) and allocates little, so it passed beside the
running server; it started no engine and touched no port.

### 1. The uncommitted M3 hunk verified, committed as two commits

`d4f9a0ab core: the shard cuts the native image on its type's block width, checked against the kernel`
is the predecessor's work: `kQsaNativeBlockCols` (a hard 256) replaced by `qsa_native_block_elems(type)`
- 32 for the legacy forms (2, 6, 7, 8, 20), 64 for Q2_0 as type 42, 256 for the K-quants and the wide IQ
types, 0 for anything else - with every native figure (block size, span legality, packed byte count, the
`gdn_gather_cols` stride) going through that one table, and the refusal of an unknown type instead of a
division by it.

**What it got wrong, found while reviewing the committed diff and fixed in
`e796f42b core: the pair's native-image cross-check compares the whole image, not one row`.**
`qsa_prepare_one`'s cross-check multiplied one block's bytes by the block count along the input axis and
compared that against `qsa_native_bytes` - which is the WHOLE image, `ne1` rows. The two figures are equal
only when `ne1 == 1`, so the check as written refused every real `attn_output` (4096 rows) with "the
split's table and the kernel's disagree". It passed the CPU suite because `qsa_prepare_one` is CUDA-only
and nothing calls it; it would have failed the first time the engine ran a shard. The multiply-up now lives
in the pure header as `qsa_weight_native_bytes_from_block(w, block_bytes)`, which is also what makes the
check mean something: the block comes from the kernel's table, the image from the split's, and the two must
agree. Mutation-checked: dropping the row factor fails 3 checks.

The order is on the record: the hunk was committed after the tests named in the task passed, and the defect
was caught in the review pass over my own commit, not before it.

### 2. Verification (run 5)

- `cmake --build build -j4`: green. `layer.cpp` also recompiled standalone from its ninja command to be
  sure the committed hunk really compiled; its only warnings are the file's pre-existing
  `-Wmisleading-indentation` ones, unchanged in count from HEAD (37 long lines before and after).
- `tp2_shard_weights_test`, `qsa_head_split_test`, `session_head_shard_test`, `tp2_pair_test`: all ok.
- `tp2_head_split_parity`: 0 failures, figures unchanged from 7a1a512c (1.77x back-to-back, worst
  |split - whole| 4.1e-08 against max |y| 0.236, combine bitwise vs `tp2_all_reduce_ref`).
- `ctest -L cpu`: 66 of 67; the single failure is `expert_multi_test` (this CPU lacks AVX-512-VNNI/VBMI) -
  the branch's known environment failure, identical at HEAD.
- `tools/check_platform_boundary.sh`, `tools/check_stderr_logging.sh`: ok.

### 3. M3's remaining execution wiring: assessed, NOT started

Verdict: **larger than the head-split wiring, and blocked on a piece that does not exist at all.** Two
things have to be true for a split to *execute*; today neither is.

**(a) Nothing in the engine drives a pair - and never has.** Verified by grep, whole tree:
`Tp2Pair::open` has no caller outside `tests/kernels/*`; `qsa_pair_prepare`, `qsa_pair_bytes` and
`qsa_pair_free` have no caller outside `src/core/layer.cpp` itself; `QsaBuffers::pair` is set by no one but
`tests/core/session_head_shard_test.cpp`; `generate.cpp` never names `QsaHeadShard` and every
`session_bytes`/`session_init`/`prompt_session_bytes` call site (generate.cpp 2123, 2719-2875) leaves the
`heads` parameter at its default `{}` - the whole layer. `o.tp_size` never reaches `src/core` at all: it
lives in `options.hpp`, `generate.cpp` and `program/tp2_pair.hpp`. So the head split is *wired at the layer
entry* (b0dce812) and *inert in the engine*; M3's FFN split would be inert too if written the same way.

**(b) The partner card runs no layer compute.** `GpuStage` (`src/program/engine_local.hpp:144`) carries
`pair_dev` and a `PeerExperts` tier and nothing else on that card - no `WeightTable`, no `NativeDense`, no
`SessionState`, no `Verifier`, no stream; `generate.cpp:3605-3656` says so in its own comment. A row split
needs the partner to hold half the FFN and run it. That is a new executor per stage (weights up, buffers
carved, per-card non-default streams - the parity test already proved the legacy default stream deadlocks
the pair), plus the block input reaching it and the summed output coming back: the stage handoff is
`bslot_ss` (generate.cpp 2780-2850) and `GpuStage::state_in`/`state_back` (`StagedCopy`, 5450). Nothing
replicates a block input to a second card today. `Verifier::set_peer_experts_only` (verify.cpp:728) is the
switch that keeps stage weights off the partner; the first thing (b) has to decide is what it now means.

**(c) The FFN row split's own kernel surface does not exist.** `shared_expert`, `shared_expert_multi` and
`shared_expert_multi_rows` (`include/basalt/kernels/shared_expert.hpp:52-73`) take whole `n_ff` with no row
offset, and `shared_expert_scratch_bytes(n_ff)` sizes the intermediate for the whole width; `moe_shared`
(layer.cpp:327) passes `g.n_ff` straight through. So a card's span needs either new row-range parameters on
those three entries or a per-card packed weight plus a half-width scratch, and `moe_combine_parts`'
`native_moe_combine` adds `shared` PLAIN, so the partial's arrival changes that addition too. Per AGENTS.md
each changed kernel then owes a parity test against the current one and a microbench line, and the row split
reassociates the FFN sum, so it belongs under the PPL/KLD decode gate (`basalt-bench --quality`) plus a
`docs/precision/` page - neither producible while the server is live. `docs/precision/` has no page for the
pair split at all today (`gdn-split.md` is the GDN tier split, a different thing); that gap is M2's and this
branch's, and the reviewer should decide whether it lands with the wiring.

**Why the same row-split view work was NOT added speculatively.** `tp2_shard_weights.hpp`'s cut helpers are
generic over `WeightRef`, so an FFN's gate/up rows and down columns would in fact cut cleanly today -
`ffn_down_shexp` is legacy-type with `n_in` 640, which the now-type-dependent block width is what made
cuttable at all (256 does not divide 640; 32 does). But a view nobody attaches, behind a pair nobody opens,
on a card that runs nothing, is dead code a reviewer must read and nobody can execute - the reason run 2's
notes left `tp2_all_reduce` out at the time. Writing it now would also pre-empt two decisions the reviewer
owes (below).

**What to build, in the order that makes each step testable:**

1. Per-stage pair ownership in `generate.cpp`: one `Tp2Pair` per stage pair (open at start, one byte pool),
   the shard's packed bytes accounted through `qsa_pair_bytes`, one `QsaPair` per card, `qsa_pair_prepare`
   called at setup, and `QsaBuffers::pair` attached when the session is carved. Boot log names the pair the
   way M1's lines do. Testable only on hardware; the CPU side is already covered.
2. Carve the sessions per card: `session_bytes`/`session_init`/`prompt_session_bytes`/`_carve` take the
   card's `QsaHeadShard`, and the slot arithmetic in generate.cpp 2780-2880 sizes both cards' carves. The
   figures are already tested (`session_head_shard_test`); what is new is a caller passing a shard.
3. The block-input handoff to the partner card and the summed output back to the lead, beside the existing
   `bslot_ss`/`StagedCopy` seams, on per-card streams.
4. Only then the FFN row split itself: new row-range entries beside the three `shared_expert*` forms (their
   parity tests against the current ones), packed views for `ffn_down_shexp`, half-width scratch, one
   `tp2_all_reduce` after the down projection, and a refusal gate shaped like `qsa_shard_ready`.
5. Then the reviewer's gate: greedy-decode A/B `--tp-size 1` vs `--tp-size 2`, PPL/KLD, one profile line.

### 4. Decisions the reviewer owes before (c) is written

Unchanged from run 3, plus one:

- Whether the FFN partial is summed by `Tp2Pair::all_reduce` (fixed `this + peer` order) or collected on the
  lead.
- Whether the partner card computes the indexer redundantly (current design, bitwise) or the pair shares one
  selection.
- What `set_peer_experts_only` means once a partner card holds dense weights.
- NEW: whether the shared expert is row-split at all, given the routed experts are already split by expert
  id (M1) but computed on the LEAD card through the peer window. Splitting the shared expert's `n_ff` adds a
  collective per layer for one of the model's ~10 expert units per layer; splitting the *routed* compute
  instead (each card computes the experts it holds, on its own card) is the larger win and the larger change,
  and it is adjacent to the brief's "out of scope: expert weight sharding".

### 5. For the 09:50 UTC test window

Nothing in run 5 changed any executed path: `qsa_pair_prepare` still has no caller, so the shipped
behaviour at `--tp-size 1` and `--tp-size 2` is byte-for-byte what run 3 left. The two new commits are
therefore regression-neutral; the CPU suite and `tp2_head_split_parity` above are the evidence. The M1
expert-split greedy-decode A/B and the pair's boot-log lines stay the reviewer's, exactly as TP2PP2-MEASURE.md
has them.

## Run 6 — the tp2 boot bug (reviewer's 05:15 log): CUDA1 twice, CUDA3 never

### 1. What the log actually says

```
tp2/pp2: stages A(layers 0-23)=[CUDA0,CUDA1] B(layers 24-47)=[CUDA2,CUDA3]   <- the topology: right
layer split: CUDA0 loads the dense weights of layers 0-23 only               <- the leads: right
layer split: CUDA2 runs layers 24-47, expert cache 2276 slots (5.86 GiB)      <- the leads: right
tp2: CUDA1 holds 5553 of the 12288 ranked pairs of its stage's layers (14.30 GiB of 14.30 GiB room)
tp2, CUDA1 partner tier: no room for its half of its stage's experts (0.00 GiB left after the reserve)
```

CUDA1 is named twice, CUDA3 not at all, and no `basalt load: cuda3` line exists. Stage A's partner tier filled
CUDA1 to the card, then stage B's partner tier was opened on the same card and found it full.

### 2. The cause

`generate.cpp`'s partner loop took the stage number from the position in `stages` — but `stages` holds only the
stages after the main card's (CUDA 0 leads stage 0 and is not in the list). So `stages[0]`, which is stage B on
CUDA2, asked for `pair_partner_dev(0, 2)` = CUDA1 instead of CUDA3. `cudaMemGetInfo` on the already-full CUDA1
then returned the 0.00 GiB the message quotes.

The reviewer's reading — that CUDA1's 14.3 GiB was the price model's expert-tier fill — is not what happened: the
`--tp-size 2` branch had already emptied `o.peer_devices`/`o.peer_device`, so the tier path (`open_peer`,
`spread_experts`) never ran, and CUDA0's 7.4 GiB / CUDA2's 9.0 GiB are the leads' dense weights, sessions and
expert caches. CUDA1's 14.3 GiB was tp2's own stage-A partner tier. The `gpus:` line calling those cards
`expert tier` was a genuine bug, but a reporting one: `choose_roles()` did not know about the pairs.

### 3. The fix (two commits)

- `pair_partner_of_lead(lead_dev, tp_size)` in `tp2_pair.hpp`: a stage's partner is the card beside its lead, and
  a card that leads no stage gets -1. `generate.cpp` asks that, so the list position can no longer be mistaken for
  a stage number; a card asked for two stages' halves now refuses the boot by name instead of loading until full.
  `tp2_pair_test.cpp` pins the two spellings against each other and that a run's leads and partners are its cards,
  each exactly once (14 failures if the partner is `lead + tp_size`).
- `RoleInputs.tp_size` and `GpuRole::StageLead`/`StagePartner` in `gpu_select.{hpp,cpp}`: with `--tp-size 2` no
  card is an expert tier, so `gpus:` reads `main, stage partner, stage lead, stage partner`, and `o.peer_devices`
  is never populated rather than cleared afterwards. A tier card appearing under tp2 now refuses the boot.
  `gpu_select_test.cpp` pins the role line for 4 cards, with and without peer access, and that `tp_size` 1 still
  gets the tiers. `max_expert_tiers` no longer says anything about a pair-wide run.

### 4. Validation and what is still the reviewer's

Build green at `-j4`. `tp2_pair_test`, `gpu_select_test`, `multi_gpu_plan_test`, `tp2_shard_weights_test`,
`qsa_head_split_test`, `session_head_shard_test` ok; `ctest -L cpu` 66/67 with only the known `expert_multi_test`
AVX-512 environment failure; the platform-boundary and stderr-logging checks ok. Nothing ran on a GPU: the four
cards' boot under `--tp-size 2` — CUDA1 holding stage A's odd ids, CUDA3 stage B's, both leads' caches unchanged —
is the reviewer's to confirm, as is the M1 greedy-decode A/B.

## Run 7 — the 05:50 / 06:15 arena OOM was the cards being busy, not a tp2 fill

Both boots died at generate.cpp's first `cudaMalloc` - the 771 MiB weight arena - with "422 MiB of 15888 MiB free".
Nothing else in the boot allocates before it (the only earlier `cudaMalloc`s are the bandwidth-probe helpers at
860/913), so the run itself held the context plus the 21.2 MiB embedding rows: the other ~15 GiB of CUDA0 was
another process. It was the tp1 baseline server (`/tmp/tp1-serve.log`: cuda0 10.2 GiB weights + 4.0 GiB KV) and, for
the first attempt, a stray parity-test child that survived a pkill of its ctest parent. Two boots differing only in
`--peer-reserve-mib` giving identical numbers is the signature: the arena's size is pack-derived and no flag enters it.

The fills are not the fault and were left alone. Both are sized from measured free memory AFTER the card's arena,
session, drafter and head are resident (lead cache 3545, partner tier 3629, both via `stage_room` at 2045, which
holds back `--vram-reserve-mib` + the prefill ring + the windows), which is the plain split's contract and the reason
`--peer-reserve-mib` - the expert tier's margin - does not appear there. Reserving the arena and a KV share again on
top of a free-memory measurement would have doubled them and cut the pair's resident experts by ~6.5 GiB a card.

Only change: when the arena's `cudaMalloc` fails, the engine now warns with the free and total bytes of every card the
run means to use, so a busy box is visible instead of looking like a leak. `0592828b`.

## MEASURED VERDICT (reviewer, 07:07 UTC, 2026-10-29) — tp2 loses to the plain split

Method: baseline tp1 + BASALT_FORCE_SPLIT served tonight's REAL tinyllb traffic
(86 requests, 35.43-39.76 ms/window, solo 74-90 t/s, hit 94-96%); tp2 measured on
the same box, exclusive cards, synthetic 149k-token streams.

tp2 boot 1 (id-parity split): window 38.61 ms/16w, prefill 1049 t/s, hit
52.8-77.6%, pcie 49-52% of lookups routed over the link.
Fix 2d3cfb74 (rank split: lead takes the hottest half): pcie 11-22% (the
structural fix worked), but prefill 1070 t/s (unmoved - prefill touches EVERY
routed expert, heat does not matter) and clean solo decode (94 windows):
40.75 ms/window, 45.2 t/s, hit 76.5%.

tp2 vs plain split, every axis: window 40.75 vs 35.43-39.76; solo 45 vs 74-90
t/s; prefill 1070 vs 3750-3868; hit 76.5 vs 94-96.  The gap decomposes as:
GPU-reach wait 11.15 vs 7.30 and per-layer host ~4x (pair coordination), against
an attention+dense TP saving the timing shows is ~2-4 ms - smaller than the pair
taxes.  Ideas exhausted tonight: rank-split (tried, kept - it is right), L2
warm-up predictor coverage (sub-ms), per-layer host rework (multi-day), prefill
locality (intrinsic: any partner-held expert routes when used; the only
locality-preserving partition degenerates to the plain 4-stage split).

WHY THE PRICE MODEL'S 24.7 IS UNREACHABLE HERE: it prices TP as if experts
shard cleanly; on PCIe x4 the shard tax (routing or duplicated weights) exceeds
the TP win on every axis we can measure.  An NVLink box would price differently;
the head-split + collective infrastructure this branch proved (bitwise-exact
split, one-push 2-way all-reduce with CPU oracles) is the reusable part.

PRODUCTION RESTORED 07:07 UTC (37.3 GiB VRAM / 26 RAM / 0 spill / verified).
The branch stands: 12 commits, build green, 66/67 CPU + 90/92 GPU (2 upstream),
nothing measured beyond this file.

## Run 8 — the peer's prefill rows and activations cross the link (direct P2P)

The gap left after the continuation placement is prefill: 1051-1070 t/s against the plain split's
3750-3868.  The peer's FP4 share of a streamed chunk was the one path written "WITHOUT DIRECT
ACCESS": every layer the primary's normed activations went device->host and host->device through
`hx`, and every group's result rows the same way through `hres[kRes]` - two link crossings per leg
and 4 pinned buffers a pair, where the box's cards reach each other directly.  The routed-only peer
path had always used `cudaMemcpyPeerAsync`; the FP4 one now does too.

### What changed (`src/prefill/prefill.cpp`, `include/basalt/prefill/prefill.hpp`)

- `PeerPrefill::p2p`, decided once in `make_peer`: `basalt::platform::gpu_peer_access()` and
  `cudaDeviceCanAccessPeer` in BOTH directions between the primary's and the peer's card.  It is the
  same predicate the boot's `cudaDeviceEnablePeerAccess` loop uses, so a pair the engine enabled
  copies card to card and one it did not keeps the relay.
- With `p2p`, the FP4 share runs the handshake the routed-only share already ran: `ev_in` on the
  primary's compute stream, the activations' `cudaMemcpyPeerAsync` at the head of the peer's
  `P.s`, per group `ev_grp` -> `P.s_out` -> the rows' `cudaMemcpyPeerAsync` into `m.Dm`, and
  `ev_done` recorded on `P.s_out` as the layer's end (the relay keeps `ev_res` on `res_in`).
- The relay stays: `hx`, `hres[]`, `x_out`, `res_in` and `ev_hx`/`ev_hx_free`/`ev_d2h`/`ev_h2d`/
  `ev_res` are created only when `tf > 0 && !p2p`, so a pair with the link stops pinning
  `Tp*N*4 + kRes*G*N*4` bytes of host RAM.  `Prefill::peer_bytes` is untouched: it priced the
  peer's device buffers only, never the pinned staging.

### Why it is safe, the two lifetimes that are not local

- The peer's copy READS the primary's `m.mixed`.  The primary rewrites it for the next layer on
  `m.cs`, which waits the layer's completion event before `moe_combine`; the activation copy is
  ordered before every group on `P.s`, whose `ev_grp` orders `P.s_out`, so completion of the rows
  implies the copy has read its bytes.  `order_peer` only holds experts with rows, so
  `fp4::cut_pieces` always yields at least one group and the layer's event is always recorded.
  This is the argument the routed-only path has always rested on; the FP4 path now shares it.
- `P.Dm_b[b]` is not rewritten until `ev_dm[b]` - recorded on `P.s_out` after the copy that read
  it - has passed, and `dm_live` is never cleared, so the wait survives a run switching between
  the routed-only and the FP4 share across chunks (both now record `ev_done` on `P.s_out`).
- Bits unchanged: same FP4 kernels, same activation bytes in `P.mixed`, same rows in `Dm`.  Only
  the transport moved, so no `docs/precision/` page covers it.

### Validated here (no GPU touched: the server is live)

Build green at `-j4` (CUDA 13.2, sm_120).  `tp2_pair_test`, `tp2_shard_weights_test`,
`qsa_head_split_test`, `session_head_shard_test` ok; `ctest -L cpu` 66/67 with only the known
`expert_multi_test` AVX-512 environment failure; `check_platform_boundary.sh` and
`check_stderr_logging.sh` ok.

### The reviewer's measurement list

1. Boot line, per pair: `peer GPU N computes its experts' rows of each prompt chunk up to T tokens,
   a streamed one with the primary's FP4 kernels and a card-to-card copy; ... MiB left free on it`.
   `through pinned host memory` instead of that means the pair was refused peer access, and the
   run is on the relay - say which it printed.
2. The prefill A/B, same synthetic 149k-token stream as the 07:07 verdict, `--tp-size 2` on the
   exclusive cards with the rank split (`2d3cfb74`) and the continuation placement: the number to
   beat is 1051-1070 t/s, the target the plain split's 3750-3868.  Label it with the power limit,
   and take the `link`/pcie% readings too - a leg now crosses the link once, not twice.
3. Solo decode and the window: this path is prefill-only, so 25.53 ms/window and ~81 t/s should be
   unmoved.  Re-run the 94-window solo to confirm `ev_done` being shared with the routed path costs
   nothing.
4. The plain split too: `set_peer` is what the expert TIER uses as well, so a `--tp-size 1` +
   `BASALT_FORCE_SPLIT` run with `--peer-prefill-rows` also switches transport.  That is the
   production configuration; confirm its prefill is not slower.
5. Quality: `basalt-bench --quality` (PPL/KLD against the Q8_0 reference) on a streamed chunk
   between `--stream-all-min` and 5120 tokens with the peer share on, per CONTRIBUTING's gate.
6. Risk left: nothing here can tell whether the DMA really used the peer path rather than driver
   staging - `cudaDeviceCanAccessPeer` says the pair can, not that access was enabled.  The boot's
   `peer access: N of M ordered pairs enabled` line plus the prefill t/s is the evidence.
7. There is deliberately no flag for the transport: `p2p` follows the pair's peer access, so an A/B
   against the relay is `git checkout bd95c276^` for the prefill run, not a switch.

## SESSION 2 RESULTS (10:30-11:15 UTC, 2026-10-29) — continuation + P2P + short-read 512

The bottleneck question that unlocked it: the lead card of a tp2 pair is
capacity-poor (dense 24 layers + head + session + arena leave ~5.9 GiB of
expert room), so the rank split's cold tail on the partner starved the lead
- 5.04 CPU-expert entries per layer-window vs the plain split's 1.83. The
continuation placement (2bbfcdda: partner takes the NEXT-hottest slice after
the lead's admissions) fixed the capacity inversion.

| config | window (solo, matched depth) | solo t/s | prefill | hit |
|---|---|---|---|---|
| plain split (production) | 35.43 ms | 74-90 | 3750-3868 | 94-96% |
| tp2 parity | 38.61 | 45 | 1049 | 53-78% |
| tp2 rank (cold tail) | 40.75 | 45 | 1070 | 76.5% |
| **tp2 continuation** | **25.29-25.85** | **76.5-81.3** | 1051-1076 | 94.9-96% |
| tp2 continuation, real traffic | 31.69 solo-ish / 41-43 concurrent | 71.9-78.6 | read 245-1076 | 90-96% |

Decode: -13% window vs production at matched depth; real-traffic solo within
the historical band, concurrent ~7% behind (GPU-reach wait 12.6-13.8 vs 7.3-8.5
- the pair sync under 2x rows is the remaining decode bottleneck).

Prefill: P2P transport (bd95c276) did NOT move it (1051->1076) - the cost is
the per-layer round-trip structure (primary attention -> combine -> peer MoE
half -> merge), not the transport; the routed rate ~1050-1250 matches the old
tier's prefill signature. Layer-block placement would remove routing but dies
on lead capacity (3.2 GiB room vs 15.8 needed). Structural floor for this
placement.

Short-read: the windows path under tp2 ran a real 170-token read at 39.8 tok/s
(4.3 s TTFT) - --short-read 512 set so aligned small reads go exclusive.
Synthetic small reads could not be exercised: appends snap the prompt-cache
reuse boundary (110592 of 122534 reused) and re-read ~12k tokens - a cache
granularity x append interaction that turns misaligned appends into 11-15 s
TTFT at the routed prefill rate; needs its own investigation.

Live config: tp2 continuation + P2P + short-read 512, serving on :8000.

## RUN 9 — prefill instrumentation: the three experiments, and the prediction written first

Instrumentation only; no path changed, nothing optimised. One file, `src/prefill/prefill.cpp`.

### What is measured

`struct ChunkTiming` puts CUDA event markers on the primary's compute stream and, when the peer takes
rows, on the peer's compute stream (`PeerPrefill::s`) and on the stream its rows come home on (`s_out`),
plus a bracket on the blob copy stream (`m.copy`). Markers are *recorded* as the layers issue; their times
are read where the run already waits for the GPU (the per-layer host sync that reads the routing, the
chunk's hand-off sync), so the measurement adds records and never a wait. A layer the run never caught up
with is counted in `skipped`, never guessed. Markers exist only when the run shares its cards with another
card, so a plain single-card run measures nothing.

Phases (primary, disjoint, they sum to the layer): `attn` (attention, QSA, norms, hyper-connections),
`router` (router + shared expert + the drain that reads the routing), `quant` (activation quantize, row
tables), `peer-enqueue` (host-side enqueue of the peer leg), `lead-experts` (the lead's own grouped GEMMs
and its waits on streamed blobs), `peer-wait` (the stream blocked on the peer's `ev_done`/`ev_res`),
`merge` (`moe_combine`), `tail` (the hyper-connection write). Peer side: `act` (activations across),
`quant` (its token quantize), `comp` (its compute), `rows` (its rows home). Grouped MMQ/FP4 groups are
timed per group (count, mean, max).

### Prediction, written before any measurement (session-2 geometry: chunk 12288, 24 layers,
### 1051-1076 t/s = 11.7 s/chunk = 487 ms/layer; power limit as in the session-2 table)

Arithmetic first, per MoE layer at 12288 tokens: rows = T*K = 122 880, ~all 512 experts hit, mean 240
rows/expert, 32 groups of 16; expert FLOPs 122 880 x 9.83 MFLOP = **1.21 TFLOP**; the layer's FP4 blobs
**~1.3 GB**, so ~32 GB of expert weights move per chunk. At 11.7 s/chunk that is **2.8 GB/s**; the plain
split's 262 ms/chunk-layer is the same 32 GB in ~3.2 s = **9.9 GB/s**. The pair moves the same bytes at
28% of the plain split's rate, which is exactly the 487/262 ratio. So the wall is predicted to be the
**weight feed**, not the round trip and not the GEMMs.

| phase | predicted ms/layer | % of 487 |
|---|---|---|
| attn | 70 | 14 |
| router | 8 | 2 |
| quant + peer-enqueue (host launch gaps) | 20 | 4 |
| lead-experts (its own GEMMs + its waits on blobs) | 320 | 66 |
| peer-wait (`ev_done`/`ev_res`) | 12 | 2.5 |
| merge (`moe_combine`) | 25 | 5 |
| tail (hc write) | 12 | 2.5 |
| unattributed queue | 20 | 4 |

Bucketed, as asked: attention 16%, experts 71%, transport 62% (the `blobs on the copy stream` column,
which overlaps `lead-experts`), sync 7%. The two are two views of the same disjoint row set.

**Experiment 1 - where the profiles diverge.** Predicted: `attn`, `router`, `merge`, `tail` come out within
about 10% of the plain split at the same chunk size; the first op that diverges is `lead-experts`, together
with `blobs on the copy stream`. `lead-experts` 320 ms (pair) vs ~120 ms (plain split).

**Experiment 2 - expert GEMM efficiency.** Predicted `rows/expert` min ~90, mean ~240, max ~400 and
**the same on both runs**: a row belongs to one expert and one card, so the rank split does not halve rows
per expert - that hypothesis is predicted FALSE. What the split does change is the group count: ~32 groups
per layer for the plain split, ~16 lead + ~16 peer for the pair, each group still 16 experts and <= P.G
rows. If `groups mean` comes back ~4-8 ms the GEMMs are running; if it comes back far above that with a
small `max`, the path is per-group fixed-cost bound (gather + quantize + launches), not GEMM bound. The
read line's GB/s per card is the discriminator: ~10 GB/s (plain split) vs ~3 GB/s (pair lead) is predicted.

**Experiment 3 - collective and sync share.** Prefill runs no collective: its attention is
`qsa_prompt_attn_batch` plus one whole `attn_output.weight` projection on the lead; the head split and its
`tp2_all_reduce` live in `core/layer.cpp`, which the
prompt path never enters. So the pair exchanges expert *rows*, not partial sums, and the "collectives
serialise the split" hypothesis is measured by `peer-wait` + `rows` + `act`. Predicted ~35 ms/layer
together, i.e. sync cannot explain 225 ms.

**The prediction that decides the shape of the answer.** `ppc` (the peer leg) is only taken when
`T <= PeerPrefill::T_max` or the chunk is a streamed one of at most `T_fp4` tokens. The serve path sets
`stream_all_min()` to 2560 (4096 with `--peer-exclusive`) whenever the peer takes prompt rows
(`generate.cpp:1404`), and `make_peer` then gives `T_max = min(chunk, stream_all_min()-1) = 2559` and
`T_fp4 = min(chunk, kPeerFp4Tokens) = 5120`. At the production `--prefill 12288`
both fail, so the chunk line is predicted to print **`no peer`**: at 12288-token chunks the peer computes
nothing in prefill and the pair's whole prefill cost is pulling the partner's half of the experts over the
link while the lead streams the rest. If it prints `no peer`, every round-trip explanation of the prefill
regression is dead and the weight feed is the only suspect. The peer legs need `--prefill 5120` (FP4, with
`stream_all_min() <= 5120`) or `--prefill 2559` (routed MMQ) to run at all.

### The lines to grep (both are `info`, tag `basalt prefill`)

```
grep "peer chunk at token" log
grep "peer read layers" log
```

`peer chunk at token <p0> (<T> tokens, <timed>/<stage layers>, <fp4|mmq|no><|p2p|/host> peer <dev>/<prim>):
layer <ms> ms = attn .. router .. quant .. peer-enqueue .. lead-experts .. peer-wait .. merge .. tail ..;
blobs on the copy stream ..; peer act .. quant .. comp .. rows .. ms; groups <n> mean .. max .. ms (<timed>
timed), skipped <n>` - one line per chunk per stage.

`peer read layers <LB>-<LE>: <chunks> chunks, <tokens> tokens, <timed layers> timed layers; expert shape
gate/up 1280x2560 + down 2560x640, 512 experts of the layer, 10 rows per token; row-visits per layer all
.. lead .. peer ..; expert-visits lead .. peer ..; groups per layer lead .. peer ..; rows/expert min .. mean
.. max ..; lead <GiB> GiB <TFLOP> TFLOP = <GB/s> GB/s <TFLOP/s> TFLOP/s; peer ...` - one line per read per
stage. Bytes are the layer's expert blobs times the experts that card was given; FLOPs are
`2*(1280*2560 + 2560*640)` per row per expert; the rate divides by the time that card's own phases spent
(`lead-experts` for the lead, `rows A->B` for the peer).

Compare the two runs on the same stage: match `peer read layers a-b`, and read the phases from
`peer chunk at token` lines with the same `<T> tokens`.

### The measurement's own cost

Per layer the run gains ~20 event records on streams it already uses, ~2 on the copy stream and two per
timed peer group, plus the queries and elapsed reads at the sync the run already performs. Each costs the
host a microsecond or two: a few tenths of a millisecond per layer against 487 ms, about 0.1%. It
changes no launch, no wait, no byte: every marker is a record, and the reads happen only where the code
already synchronised.

### Validated here (GPU untouched: the server is live)

Build green (`cmake --build build -j4`). `tp2_pair_test`, `qsa_head_split_test`, `session_head_shard_test`,
`tp2_shard_weights_test` pass; `ctest -L cpu` unchanged except the pre-existing `expert_multi_test`, which
fails on this host because its CPU lacks AVX512-VNNI/VBMI (it does not link `prefill`).
`tools/check_platform_boundary.sh` and `tools/check_stderr_logging.sh` ok.

## RUN 10 — the peer's FP4 share's gate is computed from the placement, not a constant

Run 9's prediction measured: at `--prefill 12288` the peer does no row visits and the lead pays for the
partner-held blobs across the link. Both gates (`T_max`, `T_fp4`) were constants — `kPeerFp4Tokens = 5120` capped
the FP4 share whatever the placement was. One file, `src/prefill/prefill.cpp`.

### What changed

`kPeerFp4Tokens` is gone. The share's longest chunk (`PeerPrefill::T_fp4`) is now computed at setup from the
placement, in bytes on the pair's link (`struct PeerLink`, above `Prefill::peer_bytes`):

- **the round trip**, per layer the peer would serve: the chunk's normed activations go to it and its result rows
  come back. Measured from the code that ships them, not assumed: both legs are FP32 — `cudaMemcpyPeerAsync` of
  `T * N * 4` of `m.mixed` out (10 240 B a token) and `nr * N * 4` into `m.Dm` back (10 240 B a row). The FP4
  planes are made on the peer by its own `fp4::quantize_tokens`, so they never cross, and the row tables
  (`src`, the piece table) are host-to-device on the peer's own PCIe, not the pair's link.
- **what it spares**, per the same layers: the partner-held blobs the primary's ring leaves out
  (`peer->has(l, e)` skips them), `held(l) * PeerExperts::slot_bytes(l)`, independent of the chunk.

Rows follow the routing, which setup does not know, so a layer's rows are split in proportion to the experts each
card holds (a chunk of `stream_all_min()` tokens or more hits every expert — run 9's shape). The crossover is
`blobs / per_token`, capped by the chunk. `Prefill::set_peer` logs it; `Prefill::peer_bytes` (the tier's reserve,
taken before the peer is filled) sizes the same buffers for the widest the crossover can reach for this pack
(`peer_fp4_ceiling`: a whole layer on the peer), so opening the gate never needs a re-size and the allocation
always fits the reserve.

`kPeerSlowRows = 2559` stays: its caller is the program's `cards_alike` verdict (`peer_streamed_share`,
`generate.cpp:1388`), so it already applies only to a genuinely unlike pair. On this box (4x RTX 5060 Ti) the
cards are alike, `streamed_share` is true, and the routed-only share takes `min(chunk, stream_all_min() - 1)` with
no slow-peer derate.

### The arithmetic, at run 9's geometry

24 layers, 512 experts a layer of 2 548 736 B (`expert_layout`'s NVFP4 blob), N = 2560, K = 10,
`stream_all_min()` = 2560. Checked by compiling `peer_link`/`peer_fp4_ceiling` as they stand in the file against a
hand-computed value (transcription check, no device).

| peer holds a layer | round trip per token | partner-held blobs | crossover |
|---|---|---|---|
| 256 of 512 (tp2 half) | 1 474 560 B | 14 934 MiB | **10 619 tokens** |
| 100 of 512 | 725 760 B | 5 834 MiB | 8 428 tokens |
| 33 of 512 (~85 MiB a layer) | 404 160 B | 1 925 MiB | 4 994 tokens |
| 4 of 512 | 264 960 B | 233 MiB | 923: below `stream_all_min()`, share off |
| 0 | — | 0 | 0 |

The 33-expert row is the tier-mode sanity check the brief asked for: a small held share lands the crossover on the
old flat 5120. The ceiling for this pack is 11 585 tokens, and it bounded every placement over layers 1..24 x held
0..512.

**So at the production 12 288-token chunk the peer still computes nothing.** The gate is 2.07x wider than the flat
constant but the chunk is past the crossover: at half-and-half the round trip is 755 MB a layer against 652 MB of
blobs it spares. To measure the peer legs, run `--prefill 8192` (or 10 240); `--prefill 12288` will still print
`no peer`, and now says why in its own line.

### Buffer audit: what grows with the wider gate

`peer_layout` sizes the peer's buffers from `Tp = max(T_max, tf)`, so nothing was pinned to the old 5120 — the
constant was the only pin. What moves when `tf` goes 5 120 -> 11 585 (the reserve's worst case):

| peer buffer | sized from | was | now |
|---|---|---|---|
| `mixed` (the activations it receives) | `Tp * N * 4` | 50.0 MiB | 113.1 MiB |
| `f_a4x` (its token planes) | `fp4::token_bytes(Tp, N)` | 7.03 MiB | 15.91 MiB |
| `f_gsx`, `f_rel` | `Tp * 4`, `2*(NE + Tp*K/64 + 1)*4` | 20 KiB, 10 KiB | 45 KiB, 18 KiB |
| `src`, `ident` | `cap_rows * 4` each | 200 KiB (`Tp*K` = 51 200) | 240 KiB (the caller's 61 440) |
| `Xq_g`, `Hq_g`, `GU_g`, `H_g`, `Dm_b[2]` | `G = T_max` rows (2 559) | unchanged | unchanged |

+72.1 MiB on the peer card, which comes out of its expert tier (~30 slots at 2.43 MiB): **pin the tiers when
comparing decode numbers against run 8/9 builds.** The per-group buffers do not grow because `fp4::cut_pieces`
caps a piece at `G = T_max` rows regardless of the chunk; `P.bounds` belongs to the routed MMQ share, whose
`T_max` did not change. Without `p2p` the relay also pins an `hx` of `Tp*N*4` (52 -> 113 MiB) and `kRes` result
buffers of `G*N*4` (104.9 MiB, unchanged); this box runs with `p2p`, so the relay is not taken here.

### The lines to grep

```
grep "peer fp4 crossover" log        # the decision, once per stage at setup
grep "peer GPU" log                  # unchanged: "... computes its experts' rows of each prompt chunk up to N tokens"
```

`basalt prefill: peer fp4 crossover 10619 tokens: 1440 MiB of activations and rows per 1024 tokens against
14934 MiB of partner-held blobs over 24 layers (6144 experts held)` — the second field is `per_token * 1024`, the
third the blobs the share spares across those layers. A crossover above `stream_all_min()` with the second line
reading `up to 2559 tokens` and no `a streamed one with the primary's FP4 kernels` clause means the FP4 buffers did
not fit on the peer card and it kept only the routed-only share.

### The one lever left, deliberately not pulled

The outbound leg is the cheap half of the round trip to fix and the only thing between the crossover and the
production chunk: the peer is handed the FP32 activations and quantizes them itself (`fp4::quantize_tokens(P.mixed,
...)`), while the lead makes exactly those token planes for its own experts in the same layer
(`fp4::quantize_tokens(m.mixed, T, N, N, fp.a4x, fp.gsx, m.cs)`, same function, same input, so the same bits). Copy
the lead's planes instead of its `mixed` and drop the peer's quantize: 1 444 B a token (the codes, the UE4M3 scales,
the token scale) instead of 10 240, which puts the tp2 crossover at **12 394 tokens**, past 12 288, and the gate
opens at the production chunk. The lead's quantize would have to be hoisted above the peer's enqueue (it runs after
it today), so this changes what crosses the link, what the peer launches and in what order - it needs the FP4 parity
test, a profile and the quality gate: the maintainer's call, not a silent addition to this change.

### Validated here (GPU untouched: the server is live)

Build green (`cmake --build build -j4`). `tp2_pair_test`, `qsa_head_split_test`, `session_head_shard_test`,
`tp2_shard_weights_test` pass; `ctest -L cpu` 66/67 with only the pre-existing `expert_multi_test` (this CPU lacks
AVX512-VNNI/VBMI). `tools/check_platform_boundary.sh` and `tools/check_stderr_logging.sh` ok. No engine start, no
`basalt-ctl`, nothing bound on :8000. **Ready for reviewer measurement.**

## RUN 11 — the pair's partner card was never its prompt peer

Reviewer's log: `--prefill 8192` gave 1384 t/s, a 7607-token chunk sat under the computed crossover of 10 619,
and still the chunk line said `peer -1`, no peer activation, no peer row visits, and the lead's blob feed ran the
whole layer (82.8 on the copy stream). The crossover line `set_peer` prints never appeared.

### Diagnosis

`Prefill::set_peer` was never called. It has exactly two callers and both hang off the expert TIER:

    if (peer.valid() && o.peer_prefill_rows != 0) sp.set_peer(&peer, rows, peer_streamed_share, err);

Under `--tp-size 2` a pair-wide split names no tier card (`--peer` is not given, `peer.valid()` is false), so
neither CUDA0's prompt path nor any stage's ever got a `PeerPrefill`. `ppc` was null in `read_chunk`, the row
split left every expert local, and the ring pulled every partner-held blob — which is what the log showed. The
`stage.pair` / `stage0_pair` cards were registered as ring *sources* (`set_ring_remotes`, run 8) but never as
row-computing peers. The four gates in `peer_fp4_possible` were never reached, so none of them was the blocker,
and the crossover log could not print.

Two things were missing, both in `src/program/generate.cpp`:

1. **the peer itself** — a stage's partner card holds half its stage's experts and computes nothing in prefill.
2. **the streaming floor** — `set_stream_all_min()` required `o.peer_device >= 1`, so a tp2 run kept the 6144-token
   floor while the tier configuration drops it to 2560. Chunks of 2560..6143 tokens ran the routed-only path.

Checked and NOT the cause, for the record: `fp4_every_layer()` is a property of the pack (every layer's
`gu_type`/`d_type` NVFP4 and `mmq_plan().layer[l]`), cached in a static, so the continuation placement cannot
break it — the placement only enters through `peer_link` and `peer_fp4_ceiling`, which read `peer->has()` and
follow it correctly. `fused_ring()` is false for this pack: `fused::native_supported` needs a gu type in
{IQ2_XXS..IQ4_XS} and a d type in {Q2_0, IQ4_NL}, so NVFP4 layers are not covered and the "every layer" test
fails. `streamed_share` was the tier's `cards_alike(cards[0], cards[1])` verdict, which is CUDA0's pair only.

### The fix

- `generate.cpp`: a per-pair `pair_streamed_share(lead, partner)` beside the tier's verdict, and
  `pairs_streamed_share` (every pair alike) for the one global floor; `set_stream_all_min()` now fires for an
  all-alike `--tp-size 2` run as well as for an alike tier pair. `pair_prompt_bytes()` prices a partner card's
  peer buffers and `open_partner()` holds that room back when it sizes the partner's expert tier — the same
  contract `peer_prompt_bytes` has for the tier, since `set_peer` allocates after the tier is filled.
  `set_peer` is then called on each stage's `sp` with `st.pair` (inside `init_prompt_paths`, so the chunk
  step-down re-applies it) and on CUDA0's `sp` with `stage0_pair`. A stage that cannot get its peer logs a warn
  and reads slower; it does not fail the boot.
- `prefill.cpp`: `peer_fp4_gate()` names the gate that shut and `set_peer`'s one setup line now carries the whole
  state, so a sweep never has to re-read this file to know what a pair is doing:

      basalt prefill: peer fp4 crossover 10619 tokens: 1440 MiB of activations and rows per 1024 tokens against
      14934 MiB of partner-held blobs over 24 layers (6144 experts held); the share is on, chunks of 8192 tokens
      or less, routed chunks to 2559 tokens

  Off cases read `the share is off: the pair's cards are not alike` / `not every layer's experts are NVFP4` /
  `this pack runs the fused expert kernels` / `the FP4 floor is above the streaming floor` / `the crossover is
  under the 2560-token streaming floor`, and an unlike partner adds `the routed share derated to 2559 tokens`.

One dead end noted, not touched: `--peer-prefill-rows` no longer bounds anything — `make_peer` clamps `cap_rows`
to `Tp * K` and `peer_layout` then overwrites it with `Tp * K` ("no row cap"), so the runtime test
`rows_peer + c > ppc->cap_rows` cannot bind. That is what keeps the stream's unconditional `peer->has` skip safe
(every partner-held expert's rows must be somewhere), so the flag's help text is what is stale, not the code.

### What it costs

Each partner card now gives `Prefill::peer_bytes()` its share of VRAM before its tier is sized. At the reviewer's
geometry (`--prefill 8192`, floor 2560) the terms are `mixed` `Tp*N*4` = 80 MiB, `Dm_b[2]` = 50 MiB, `f_a4x`
= 11 MiB, `GU_g` 12.5 MiB, `Xq_g` 7 MiB, `H_g` 6.2, `Hq_g` 1.7, `src`+`ident` 0.6, plus the MMQ group buffers
(`grp_gu`, `grp_d`) that scale with the pack's largest blobs — a few hundred MiB in total, call it a hundred slots
of 2.43 MiB off that card's hot set. Run 10's audit priced the same set for the tier card; the `peer GPU N ... N
MiB left free on it` line names the bytes on this box. `--peer-prefill-rows 0` turns the prompt peer off entirely
(and restores the 6144 floor): that is the A/B switch, and the pre-change behaviour.

### The sweep matrix (four alike cards, this pack, `--prefill <chunk>`)

Crossover ~10 619, the FP4 share also capped by `--prefill`, the streaming floor now 2560.

| `--prefill` | regime expected | the chunk line should say |
|---|---|---|
| 12288 | past the crossover: no peer leg, the lead streams everything | `no peer` — reproduces run 9/10 |
| 10240 | just inside: the FP4 leg pays | `peer 1`, `fp4 yes`, peer rows ≈ its experts' half |
| 8192 | the reviewer's 1384 t/s point, now with the peer | as above; peer rows ≈ 40 k of 81 920 |
| 6144 | same regime, shorter chunk | as above |
| 4096 | NEW: streams (was routed-only at the 6144 floor) | as above; compare with `--peer-prefill-rows 0` |
| 2560 | the shortest chunk that still streams | as above |
| 2048 | routed-only: no FP4 leg, the peer takes ≤ 2559 rows | `peer 1`, `fp4 no` |
| 1024 | routed-only, the peer's rows the whole help | `peer 1`, `fp4 no` |

Grep: `peer fp4 crossover` (one line per prompt path, at setup), `peer GPU` (the buffers and the MiB left on the
partner), then per chunk `peer`/`fp4`/`peer rows`.

### Validated here (GPU untouched: the server is live)

Build green (`cmake --build build -j4`). `tp2_pair_test`, `qsa_head_split_test`, `session_head_shard_test`,
`tp2_shard_weights_test` pass; `ctest -L cpu` 66/67 with only the pre-existing `expert_multi_test`.
`tools/check_platform_boundary.sh` and `tools/check_stderr_logging.sh` ok. No engine start, no `basalt-ctl`,
nothing bound on :8000. **Ready for reviewer measurement.**

## RUN 12 — decode instrumentation: the window's phases, op by op (DECODE-NEXT item 1)

Instrumentation only; no path changed, nothing optimised. Three files: `src/core/verify.cpp`,
`include/basalt/core/verify.hpp`, `src/program/generate.cpp`. The mould is run 9's `ChunkTiming`:
markers recorded where work is ISSUED, times read only where the run already waits for the GPU,
so the measurement adds records and never a wait.

### What the code says first (read this before the numbers)

At this HEAD the decode path runs **no collective and no head split**. `tp2_all_reduce` has exactly
one caller (`qsa_layer`, the prompt path, which the engine never drives — `QsaBuffers::pair` is set
by no engine code), and the verifier's QSA decode attention runs whole on the lead card. The
pair sync of a tp2 decode window is per layer: the partner card computes the routed experts it
holds in its peer window (one persistent kernel per window, launched in `dm_begin`, synced in
`dm_end`), and the lead's MoE tail spins on the peer's done word (`dec_moe_wait`) before
`dec_moe_combine`. DECODE-NEXT item 1's "o_proj combine / tp2_all_reduce spin" hypothesis is
therefore falsifiable from this line: if the wait grows with rows under concurrency it is the
peer-row wait of the MoE tail, not a collective spin — decode issues zero collectives.

### The markers

Per stage verifier, created once before the first capture (`wp_ensure`):

- **five CUDA events per layer**, recorded inside the captured window graph (they are graph nodes,
  re-recorded on every replay, no PDL interaction — unlike `--profile`'s stamp kernels, PDL stays on):
  1. layer start (top of `pre`), 2. MoE start (after the mixer and its hyper-connections, at the
  router), 3. final-wait start (before `dec_moe_wait` in the device-planned tail, before the last
  plan flag wait otherwise; coincident with 4 when there is no final wait), 4. combine start
  (after the wait), 5. layer end (after the tail write).
- **one stage-end event** (after the hand-off copies on a middle stage, after the head's argmax on
  the last).
- **two events on the partner card's stream** bracketing its window: recorded in `dm_begin` after
  the peer kernel launches and in `dm_end` after its sync — the partner's push span.

Times are read in `Verifier::decode_phase` at the window's own syncs (each stage's
`run_slot_segs` already synchronised its stream and its peer window before the batch loop reads).
A window whose markers were not all reached is reported `not measured`, never guessed. The phases
are disjoint and sum to the stage's window span: `attn + experts + wait + combine + rest = total`.

### The line

`basalt decode profile`, one line per window every 32 windows (the common case), every window with
`--exp decode_profile=1`. Gated to multi-GPU runs (a layer split / tp2 pair), as prefill's
instrumentation is gated to multi-card runs. Nothing is measured single-card.

```
basalt decode profile: window 32, 2 slots, 24 rows, 18 tokens, concurrent: wall 41.3 ms = run 34.2 + commit 2.1 + emit 0.8 + draft 4.2; GPU-reach wait 12.9 ms;
; stage 1: 17.1 ms = attn 6.2 + experts 5.1 + wait 4.4 + combine 1.4 + rest 0.0, peer push 33.9; stage 2: 17.0 ms = attn 6.1 + experts 5.2 + wait 4.5 + combine 1.2 + rest 0.0, peer push 34.1
```

- `wall = run + commit + emit + draft` by construction (host clock around the window).
- `attn` = the mixer and its hyper-connections to the router (whole on the lead card: there is no
  partner attention half at this HEAD); `experts` = the lead's own MoE compute, plan waits
  included; `wait` = the final spin for the peer's rows and the CPU pool's word — the pair-sync
  number, per layer, summed over the stage's layers; `combine` = the partials' add and the layer's
  tail; `rest` = everything else of the stage (head, hand-off, in-graph gaps).
- `peer push` = the partner card's window span (launch to sync of its persistent window): the push
  side of the pair sync. It overlaps the lead's run; it is not added to the sum.
- `GPU-reach wait` = the host's per-layer doorbell spin over the whole chain (the
  9.3 → 12.6-13.8 figure from DECODE-NEXT), for the window.
- `solo` / `concurrent` is the window's slot count (1 vs 2+). Run the same traffic solo and with
  two sessions and compare the `wait` and `peer push` columns at the same rows — that is the
  concurrent pair-sync cost, separate from compute.

### The lines to grep

```
grep "basalt decode profile" log                 # every 32nd window
BASALT_TP2=1 ... --exp decode_profile=1          # every window
```

Match stages across runs by their layer count in the boot lines (`tp2: %d stages of %lld layers
each`); compare a `--tp-size 2` run against the plain split on the same line — the first phase that
diverges names the divergence.

### The measurement's own cost

Per window per stage: ~5×layers event records inside the graph (a few ns each on the GPU), ~5×layers
`cudaEventElapsedTime` reads at the sync the run already performs (a microsecond or two of host
time each, against a 25-40 ms window). No launch, no wait, no byte changes; PDL stays on. Numerics
are untouched (event record nodes carry no data), so no quality gate is owed; a greedy-identity
check against the pre-instrumentation build is the regression net.

### Validated here (GPU untouched: the server is live)

Build green (`cmake --build build -j4`). `ctest -L cpu` 66/67 with only the pre-existing
`expert_multi_test` (this CPU lacks AVX512-VNNI/VBMI). `tools/check_platform_boundary.sh` and
`tools/check_stderr_logging.sh` ok. No engine start, no `basalt-ctl`, nothing bound on :8000.
GPU runs are the reviewer's.

## RUN 13 — the gate opened; the wall moved to the RAM tier (reviewer measurement, merged binary 3c3f9d80)

Sweep with `set_peer` live (boot line: `peer GPU 1/3 computes its experts' rows ... up to <chunk> tokens`):

| --prefill | fresh prefill | stage A layer | stage B layer | blobs A | blobs B |
|---|---|---|---|---|---|
| 8192 | **1393 t/s** | 220.6 ms | 155.7 ms | 114.3 | 31.1 |
| 4096 | 1010 t/s | 159.1 ms | 134.4 ms | 106.4 | 91.5 |
| (12288, pre-fix) | 1074 t/s | — blob regime, no peer compute (crossover 10 619) | | | |

- The FP4/p2p peer path WORKS where designed: stage B at 8192, blobs 82.8 -> 31.1, layer 181 -> 155.7.
- Stage A regressed: the ~719 MiB peer-buffer reserve shrinks the partner's tier before fill; its misses
  (RAM band) grew, and RAM rows still blob to the LEAD (the peer covers only its own held band).
- 4096 loses on both stages (fixed costs + tier dynamics; lead-experts ~101-116 both stages).
- Verdict: **the wall is the RAM-tier blob feed on the lead's lanes, chunk-size-invariant in total bytes.**
  The sweep's peak is 8192 = 1393 t/s. The end-to-end gate (~2 600) is NOT met by the gate fix alone.
- Next lever (evidence-backed): make the peer a **second RAM reader** — the partner pulls RAM-band blobs
  over its own (idle) PCIe lanes and computes those rows too; the lead's link keeps only prompt uploads.
  Stage A blobs 114 ms -> ~57, stage A layer -> ~120: projected ~2 400-2 600 t/s at 8192.
- Decode: concurrent wall 28.4-33.4 ms (3-4 rows), GPU-reach wait 18-19.2; tg solo 77.1 t/s, hit 94.7%,
  pcie 46.6% — decode parity holds at 8192. Per-stage decode profile lines report `not measured` under
  graph capture (run 12's flagged risk); fix pass owed before DECODE-NEXT item 1 can be judged.

## RUN 14 — the partner becomes a second RAM reader (`--peer-ram-rows`, default off)

Run 13's verdict: the wall is the RAM-tier blob feed, and it all crosses the LEAD's lanes, because the peer computes
only the experts its own tier holds. A chunk of a stage whose tier misses land in the RAM complement therefore pays
one PCIe link for every blob. This run lets the partner read the blobs of the experts NEITHER card holds over its
OWN link, compute those rows on its own SMs, and send only the result rows back. One file of substance,
`src/prefill/prefill.cpp`, plus the flag. Off by default: with the flag absent nothing else moves.

### What changed

- `--peer-ram-rows` (`options.hpp`, `generate.cpp:1175`), threaded into `Prefill::set_peer` and `Prefill::peer_bytes`
  as `peer_ram_rows`: "with a prompt peer: it also computes the rows of the experts NEITHER card holds, reading
  their blobs from RAM over its own link (off)".
- **The band.** `in_ram_band()`: an expert of the stage held by no card's tier (`host_res` / `peer->has` / the
  `--peer-exclusive` ring all negative) whose blob is page-locked (`src->pinned`) — i.e. exactly the bytes the lead's
  stager pulls today. `RamView` carries the four tables the test needs, so the setup-time price and the per-chunk
  walk ask the same question of the same tables.
- **The price.** `peer_ram_share()` (above `peer_link`) decides how much of the band to move, as an eighth
  (`PeerRamShare::kDen`): moving one band expert takes its blob (`B`) off the LEAD's inbound link and puts its rows
  (`w`) on that same link, while the partner's own link — which otherwise only ever receives — takes the blob. It
  pays while `w < B`; the two links balance at `r`, the partner never takes more than half the chunk's rows. Priced
  at `m.T_max`, the longest chunk the share is ever given. `peer_link` bills the moved experts' blobs and rows too,
  so a wider band coverage widens the FP4 crossover as well.
- **The same eighths everywhere.** `PeerRamShare::taken(k, num)` — `num` of every 8 band experts counted in id order
  — picks the band out at setup (`peer_link`), at the chunk's start (the stream's skip, before the routing is known)
  and in the row split (`ram_take`), so an expert the lead leaves out of its stream is always one the partner reads,
  and its rows always go to the partner. The chunk's decision is recomputed per chunk and is independent of routing.
- **The partner's ring** (`PeerPrefill::stage`, `copy`, `span_copy`, a thread of its own): entry `k` of the chunk's
  band list lands in slot `k % ring`, read from host RAM into the partner's VRAM on its own copy stream. The lead
  walks it with its own staging protocol: an entry the routing did not pick is given back at once, a picked one when
  the group that reads its blob LAST has been enqueued (`last_grp`, since `fp4::cut_pieces` cuts a hot expert across
  groups and the slot has to outlive every piece). Its issuing thread never runs past the lowest entry still held
  (`p_hold`) plus a ring, and a slot is refilled only after the `used` event its last reader recorded. An entry too
  far behind a held one — a group's sixteen experts need not be neighbours in id — is a **spread** expert: its blob
  is read here instead, into one of `kSpan` span slots, once per group that needs it.
- `ChunkTiming` counts the band: `expert-visits lead %lld peer %lld (peer read %lld from RAM)`.

### The arithmetic, at run 13's geometry

24 layers, 512 experts a layer of 2 548 736 B, N = 2560, K = 10, chunk 8192. `w = chunk*K/NE*N*4` is a band
expert's expected rows, `B` its blob. The table is the shipped `peer_ram_share` compiled as it stands against
stubs (transcription check, no device); it agrees with an independent script to the byte.

| card holds | partner holds | band | `w` vs `B` | r (balance) | take | share | lead's blobs left |
|---|---|---|---|---|---|---|---|
| 256 | 256 | 0 | — | — | 0 | off: no band | 0 MiB |
| 200 | 200 | 112 | 1.56 < 2.43 MiB | 285 -> 56 (row cap) | 56 | 4/8 | 136 MiB/layer |
| 160 | 160 | 192 | idem | 96 | 96 | 4/8 | 233 MiB/layer |
| 128 | 128 | 256 | idem | 128 | 128 | 4/8 | 311 MiB/layer |
| 200 | 120 | 192 | idem | 56 (row cap) | 56 | 2/8 | 419 MiB/layer |

`top` — the longest chunk whose band expert's rows still weigh less than its blob — is **12 743 tokens** for this
pack, so the band is on for every chunk the FP4 share takes (its own crossover capped it at 10 619 at tp2 half-and-half).
The share is deliberately conservative: the prompt's own uploads to the lead are not knowable at setup and are left
out of the lead's side of the balance, which understates it and so under-prices the move.

### What it costs, and what does not move

`peer_bytes` reserves the ring whenever the flag is on and the FP4 leg could open: `(48 + 16) x 2.43 MiB = 155.6 MiB`
on the partner card, on top of run 13's ~719 MiB of peer buffers. **Pin the tiers when comparing against run 13
numbers**; a tier that shrinks by ~64 slots moves the RAM-band boundary itself, which is the effect run 13 measured
as a regression. With the flag off the reservation, the boot line's meaning and every enqueue are run 13's: the
`RamView` is built and never consulted, `prs.num` is 0, `peer_link` is called with a null band, `ram_take` stays
empty, the row split, the stream and the peer legs take their old branches. The only difference is the boot line's
trailing `RAM rows off (--peer-ram-rows)` clause.

### The lines to grep

```
grep "peer fp4 crossover" log     # setup, once per stage: ...; RAM rows on, 4/8 of 112 band experts over 24 layers read on the partner's own link, to 12743 tokens, its ring 48 slots
grep "peer GPU" log               # unchanged: "... computes its experts' rows of each prompt chunk up to N tokens, a streamed one with the primary's FP4 kernels and a card-to-card copy"
grep "expert-visits lead" log     # per chunk: expert-visits lead 4123 peer 1876 (peer read 934 from RAM)
```

Off, the last clause reads `RAM rows off (--peer-ram-rows)` (or `off: the FP4 share is off`, `off: no RAM band,
every expert of the stage is on a card`, `off: its rows outweigh its blob at <T_max> tokens`); in the chunk line
`(peer read 0 from RAM)`.

### Predictions, written before any measurement (stage A = layers 0-23, lead CUDA0 / peer CUDA1, `--prefill 8192`)

Run 13's stage-A layer was 220.6 ms with 114.3 ms of blobs on the copy stream and 126.4 ms of lead experts; stage B
was 155.7 ms with 31.1 ms of blobs.

1. **Stage A is where it pays.** With the share at 4/8 the lead's inbound blob bytes over the band fall by half, so
   the copy stream's 114.3 ms goes to ~57-70 ms and the layer to **~175-195 ms**; fresh prefill at 8192 should go
   from 1393 t/s toward **1500-1650 t/s**. The rows the partner computes come back over the same link (1.5 MiB per
   band expert at this chunk), so the saving is `moved*(B - w)`, not `moved*B` — that is why the price balances the
   two links instead of moving the whole band.
2. **Stage B barely moves** (155.7 -> ~150 ms): its tier misses are already small, and where the stage's tier covers
   every expert the boot line will say `off: no RAM band`.
3. **`(peer read N from RAM)` is the falsification test.** Expect `N ≈ num/8 x band x layers x (the band experts the
   routing hit)`, i.e. for 4/8 of a 112-expert band, of order 50-56 visits a layer. If the chunk line reports
   `peer` visits rising while `peer read` stays 0, the band classification disagrees with the row split and the
   stream skip — that is the bug to look for first, and the run fails with `prefill: a RAM-band expert is on neither
   card's stream` rather than computing wrong rows.
4. **Lead expert time must not move much** (126.4 ms ±): the lead keeps attention, dense and its own resident
   experts; only its blob feed shrinks. If lead time rises, the partner's rows are queuing behind the lead's own
   work on the return path.
5. **Watch the partner's own link**: it is the new bottleneck candidate. `p2p` is on for this box, so the peer's
   activations still arrive over the pair's link while its band blobs arrive over its host link; the two are
   different engines and do not serialize.
6. `--peer-ram-rows` absent reproduces run 13 within run-to-run noise, at any chunk, on both stages.

### What is NOT done

- No measurement (no GPU touched, no engine started, `:8000` unbound). Every number above is a prediction.
- The ring depth (48) and span pool (16) are reasoned, not profiled: 48 blobs is ~116 MiB, about 6 ms of host link
  at 20 GB/s, which is thin against a layer's compute, so the depth may have to grow if the peer's copy engine is
  seen waiting. The spread path is the measurable symptom (each spread expert costs the partner one extra blob read).
- The band's experts are chosen by id position (`k % 8 < num`), which is routing-independent but not
  load-aware: a router that concentrates on those ids puts more rows on the partner than the price assumed. The
  row cap cannot protect this (the peer's row tables hold the whole chunk by design), so an extreme skew shows up
  as the partner falling behind, not as a failure.

### Validated here (GPU untouched: the server is live)

Build green (`ninja -j4`). `ctest -L cpu` 66/67, the only failure the pre-existing `expert_multi_test` (no
AVX512-VNNI/VBMI on this CPU); `tp2_pair_test`, `qsa_head_split_test`, `session_head_shard_test`,
`tp2_shard_weights_test` pass. `tools/check_platform_boundary.sh` and `tools/check_stderr_logging.sh` ok. No new
compiler warnings (the two the file prints, `StreamEntry::pdev_dev` and `cs`, are the ones the file printed
before). `peer_ram_share` compiled against stubs, matching an independent script; its eighths, `take` and `top`
are in the table above.
## RUN 15 — host dispatch of the decode window: what the 28.9 ms run actually was, and the two gates

The decode profile lines now measure (d3bd5cd0). Reviewer measurement: concurrent (2 slots, 5 rows)
wall 37.4 ms = run 28.9 + commit 0.6 + emit 0.0 + draft 7.8; stage GPU spans 13.7 / 14.5. Solo:
wall 22.2 = run 20.1 + draft 1.8, GPU 10.4. This run reads the loop to find where the host time is,
then ships two gated reductions. No path changed by default.

### What the 28.9 ms of run is (profile by reading)

`run_slot_segs` (verify.cpp) launches the whole window as ONE `cudaGraphLaunch`, then serves the
layers: per layer it (1) busy-spins on the mapped doorbell word `h_seq_` (`_mm_pause`, no usleep,
no blocking; a `cudaStreamQuery` liveness check every ~2 ms), (2) runs the layer's CPU pool
callback, (3) writes the flag words. In the device-planned tail (the tp2 path, active here) the
plan is device-side, so the pool computes only the CPU's share: activation quant of the miss rows
plus the CPU experts themselves. **Nothing launches per layer from the host** — the window is one
graph, the peer card one persistent window per window, the commit one graph. Candidate (b)
(launch overhead) has nothing left to cut per layer.

Candidate (a) (batch the waits) is not available without a protocol change: the per-layer flag is
the CPU pool's publication, and the GPU's MoE tail of layer k spins on it; the pool of layer k can
only start at layer k's doorbell (the routing ids and miss rows are written by the GPU mid-layer).
A pre-raised flag for fully-VRAM-resident layers would be safe in the device-planned path but
saves microseconds (such a layer's pool call is already microseconds; the GPU-reach wait is spent
on the layers with CPU work), so it is not worth the flag-protocol risk.

So: run = Σ_layers (GPU arrival → doorbell — the "GPU-reach wait", which includes the GPU-side
flag and peer spins — + the host pool) + the tail (final stream sync, peer window sync, sampling).
The solo delta (run 20.1 vs GPU 10.4) is ~0.4 ms/layer of CPU-tier expert compute — real work,
doorbell-gated, neither batchable nor pre-runnable. The reducible host time is in the per-window
bookkeeping and the draft phase.

### What changed (both off by default)

- `--exp batch_lines=0`: the per-window token lines (BT/BDONE/BDIG) and their `fflush` are skipped
  (the emit phase, currently 0.0 ms). Tokens and the profile line are computed either way.
- `--exp mtp_preswap=1`: the drafter's store swap for the NEXT window's lead slot (predicted as the
  slot drafted first this window) is enqueued at the end of the draft phase, right after the
  phase's stream drain. The D2D page copy then runs beside the next window's run graph on the
  drafter's (idle during the run) stream instead of inside the next window's draft phase.

### What the 7.8 ms of draft is (profile by reading)

Per slot: `flush` (deferred rows' prefill, ends in a stream sync), `use_store` (working↔slot draft
K/V page swap, D2D), row staging (two D2D copies), then `draft_rows`: one chain-graph launch, then
per draft step a mapped-word spin on `h_out_` (GPU-bound on the draft chain) and a PLE n-gram
prefetch (issues a ticket, no read). The draft depends on the run window's final rows and the
slot's committed picks; the NEXT window's run graph depends on the drafts (its rows are the
drafts) — "launch the next run while the drafter runs" is impossible, the rows do not exist yet.
What CAN overlap the next run is exactly the store swap: `mtp_preswap`.

### Predictions (the reviewer grades these)

- `batch_lines=0`: the profile line unchanged except the `emit` term (already ~0.0); wall drops by
  the per-window flush/printf cost (tens of µs). The BT/BDONE lines are absent — host-cost A/B
  only; token identity is checked at the default.
- `mtp_preswap=1`: the `draft` term drops by the swap cost. Concurrent draft 7.8 vs 2×1.8 solo
  leaves ~4.2 ms/window of swap + deferred prefill; the swap share moves out (draft toward
  2×1.8 + prefill). `run` unchanged or +<0.5 ms (the copy engine shares the card during the run).
  Tokens identical: the draft content is unchanged, only the store moves earlier. The stage
  phases (attn/experts/wait/combine/peer push) must not move; if `wait` or `experts` grow, the
  swap copy is contending the card — read it as +run. `basalt batch spec`'s draft ms/window drops
  by the same amount.

### Reviewer's checklist

1. Concurrent 2-session traffic, `--exp decode_profile=1`; grep `basalt decode profile`.
2. Three boots (the flags are per-boot): default, `--exp batch_lines=0`, `--exp mtp_preswap=1`;
   a fourth with both. Compare the run/commit/emit/draft terms; the stage phases must be flat.
3. Token identity: default flags, same traffic, BT lines identical to the d3bd5cd0 binary
   (greedy decode is untouched by both gates).
4. Solo: `draft` term should be unchanged by `mtp_preswap` (one slot: the swap is a no-op when the
   store already belongs to the slot).

### Validated here (GPU untouched)

Build green (`cmake --build build -j4`). `ctest -L cpu` 66/67 with only the pre-existing
`expert_multi_test` (this CPU lacks AVX512-VNNI/VBMI). No engine start, no `basalt-ctl`, nothing
bound on :8000. GPU runs are the reviewer's.

## RUN 14/15 addendum — the reviewer's measurement (boot 8192 + --peer-ram-rows + decode_profile=1)

Prediction grading (A's six, against the fresh 122300-token read):

| # | prediction | measured | verdict |
|---|---|---|---|
| 1 | stage A layer 220.6 -> 175-195, blobs 114.3 -> 57-70, prefill -> 1500-1650 | layer 217.0, blobs 101.9, **prefill 1406** | **missed** — mechanism worked, magnitude capped |
| 2 | stage B ~150 or `off: no RAM band` | 160.8, engaged | partial |
| 3 | `(peer read N from RAM)` nonzero, ~50-56 visits/layer | **peer read 9224 from RAM** (~25/layer) | engaged; below the predicted share |
| 4 | lead experts ~126.4 ± | 114.4 (own rows 34375/81528 visits = 42%) | ok |
| 5 | partner's link not serializing | partner comp 105.3 runs beside lead 114.4 | ok, BUT see the straggler below |
| 6 | flag absent = run 13 | not re-run; code default-off | by construction |

**The new critical path is the peer's RESULT-ROW RETURN, not the blob feed.** Chunk lines: stage A
`peer act 26.9 quant 0.3 comp 105.3 rows 114.6 ms` — the peer's computed rows take 114.6 ms to come back,
36.3 ms after the lead finished its own experts (peer-wait 36.3). Stage B worse: lead-experts fell to 15.5
but peer-wait ballooned to 88.3 (its peer computes 51% of rows; comp 83.5 + return 90.1). The price model
billed result bytes w as small against blob bytes B — the measurement says the return leg dominates at this
chunk. The lead's floor is now attn 52.6 + its own resident rows (114.4): the share moved work off the lead
but the wall didn't follow.

Decode (both stages now measure every line): solo cached turn 80.4 t/s, acc 67%, hit 94.2% — parity holds
under --peer-ram-rows. Per-stage: attn 5.5-6.4 + experts 1.1-2.5 + wait 0.0 + combine 1.2-3.4, peer push
0.1-0.2 / 1.7. Concurrent windows ~20.7-24.3 ms wall.

`--exp mtp_preswap=1` A/B: solo 76.4 t/s vs 80.4, profile walls 20.7-22.0 vs 20.2-24.3 — **no measurable
win at solo cadence**; default stays off. `batch_lines` untested (diagnostic gate).

Next levers, re-ranked by the measurement:
1. Re-price the share with the MEASURED return cost (w is not small) and make it per-stage — stage B's
   peer is overloaded (wait 88.3) while stage A's is balanced (36.3).
2. Quantize the peer's return rows (FP32 -> FP8 halves the 114.6 ms leg).
3. Head-split attention into decode AND prefill (attn is now ~25% of the prefill layer: 52.6 of 217).
4. The lead's resident-row floor (114.4 ms) only moves if someone else can compute the lead's tier —
   not plausibly; so post-RAM-reader the realistic ceiling is ~attn-split + balanced return: ~1 600-1 800.

## RUN 16 — the return leg, priced and halved (BF16 result rows; the band share re-priced per stage)

Run 14/15's addendum: with the pair reading RAM together, the wall is no longer the blob feed but the peer's
RESULT-ROW RETURN — stage A `peer comp 105.3 rows 114.6 ms`, `peer-wait 36.3`; stage B `comp 83.5 rows 90.1`,
`peer-wait 88.3` with lead-experts at 15.5. Two changes, both aimed at that leg: the rows cross as BF16 bits
(half the bytes, rounded once in the product that made them), and the band share is priced against the pair's
link instead of the blob's own, per stage. One flag: `--peer-rows-fp32`.

### 1. BF16 return rows

Measured at stage A: 47 153 peer row-visits a layer x 2560 x 4 B = **483 MB a layer back over the pair's link**,
against 80 MiB a layer of activations going out — the return is the big leg. BF16 halves it: one rounding
(~2^-8 relative), no accumulation, and BF16's exponent is FP32's, so there are no per-row scales to manage and no
outlier a row can overflow — the row is already FP4-rounded on its way in.

- **Peer side, fused into the epilogue** (`moe_fp4.hpp/.cu`): `fp4::Product::dst16` / `fp4::Group::dm16`. Set, the
  down product's epilogue stores `bf16x2(acc * g)` (a 4-byte pair per two values, `basalt::kernels::bf16_from_f32`
  — the engine's RNE with the NaN clause) instead of a `float2`; a third template argument on `gemm_kernel`,
  instantiated only for the plain (`<false,false,true>`) product, so the SwiGLU output and the primary's own rows
  keep the F32 store path and every bit they produce today. `run()` refuses a product with both destinations.
- **Lead side, fused into the combine's load** (`kernels.cu::moe_combine_kernel<ROWS16>`): the peer's rows land in
  the **first half of the bytes their block of `Dm` would take as FP32** — `peer_row_bytes()` is the one place that
  says the width, so the price, the crossover, the copies and the combine all count the same bytes. No new buffer
  anywhere: not on the peer (its `Dm_b` staging keeps its FP32 size and BF16 uses half of it), not on the lead
  (a BF16 mirror of `Dm` would have cost 419 MiB at chunk 8192). `moe_combine_rows16` reads each of a token's 10
  rows from the plane it is in — one pass, one predicate per row, no second kernel. The FP32 combine is a separate
  instantiation of the same template and is byte-for-byte the kernel it replaces.
- The row block is `[rows_local, T*K - rows_cpu)`: the lead writes `[0, rows_local)`, the peer's copies write its
  half-block, the CPU share writes the tail — three disjoint ranges, none of them touched by another path. Byte
  arithmetic checked on the host (writes land inside the block, 16-byte aligned, and every value the peer wrote is
  what the combine's indices read).
- **The routed (sub-`stream_all_min`) peer leg stays FP32**: its products are the MMQ ones, which store FP32, and
  the lead picks the combine variant per layer from which leg ran. The RAM band never rides that leg.
- `--peer-rows-fp32` (`options.hpp`, `generate.cpp`, `set_peer`, `peer_bytes`): the FP32 wire, the FP32 price and
  the FP32 combine — the A/B, and the quality gate's control.

### 2. The share re-priced, per stage

`peer_ram_share` billed a moved band expert's rows (`w`) against its blob (`B`) on the LEAD's link, and balanced
`(band-r)*B + (held+r)*w` against `acts` — which put the rows on the wrong lane and made `w` look small, so both
stages landed on the row cap (4/8) while stage B's peer drowned in its own return. Now the two lanes are what the
measurement says they are:

- the lead's own inbound link: `(band - r) * B` — the blobs it still pulls from RAM;
- the pair's link: `acts + (held + r) * w` — the chunk's activations out (`chunk*N*4`) plus every row the peer
  sends back, its held experts' included, at the wire's width (`peer_row_bytes`).

`r* = (lead - pair) / (B + w)`, still capped at half the chunk's rows, still `PeerRamShare::kDen` eighths, still
counted in id order by `taken()`, still over one stage's layers only (`acts`, `held` and the band are the stage's
own, so a pair's two stages are priced apart and the boot line prints the stage's layer range with both lanes'
MiB). `peer_link` and `peer_fp4_ceiling` bill the rows at the same width, so a BF16 return widens the FP4 crossover
and an FP32 return narrows it; `peer_bytes` takes the flag, so the reserve can never be sized for a narrower wire
than the one the run picks. `top` (the longest chunk whose band expert's rows still weigh less than its blob)
follows the width: 25 487 tokens at BF16, 12 743 at FP32.

### The arithmetic at run 14's geometry (the shipped `peer_ram_share`, compiled against stubs)

24 layers, 512 experts a layer of 2 548 736 B, N = 2560, K = 10, chunk 8192; `w` = 0.78 MiB a band expert at BF16,
1.56 MiB at FP32; `acts` = 80 MiB a layer.

| card holds | partner holds | band | share, BF16 wire | lanes at it (lead / pair, MiB a layer) | share, `--peer-rows-fp32` |
|---|---|---|---|---|---|
| 256 | 256 | 0 | off: no RAM band | — | off: no RAM band |
| 200 | 200 | 112 | **1/8** (r = 11) | 245 / 244 | **off**: the pair's link already carries more |
| 200 | 160 | 152 | 3/8 (r = 51) | 245 / 244 | 1/8 |
| 160 | 160 | 192 | 3/8 (r = 81) | 269 / 268 | 1/8 |
| 128 | 128 | 256 | 4/8 (r = 128) | 311 / 280 | 3/8 |
| 200 | 120 | 192 | 4/8 (r = 91) | 245 / 244 | 2/8 |
| 120 | 200 | 192 | 2/8 (r = 56) | 330 / 280 | 1/8 |
| 230 | 230 | 52 | off: the pair's link already carries more | 126 / 259 | off |

Read it as: the pair's link is the tight lane, so a partner already carrying half the chunk's rows takes no band at
all, and one carrying less takes only what fills its link. Run 14's 200/200/112 goes from 4/8 to **1/8** — the
share is now a lever the measurement can move, not a cap both stages saturate.

### What does not move

No blob leg, no outgoing activation plane, no FP4 plane, no ring, no row layout. `peer_bytes` is unchanged in
bytes (the staging is still sized for FP32 rows). With `--peer-rows-fp32` the only differences from today are the
boot line's two new clauses. With the peer absent, none: the FP32 combine instantiation, the lead's own F32
epilogue and the old peer legs are all still there.

### The lines to grep

```
grep "peer fp4 crossover" log   # ...; the share is on, chunks of N tokens or less, ...; rows cross as BF16; RAM rows on, 1/8 of 2688 band experts of layers 0-24 over 24 layers read on the partner's own link: 245 MiB of blobs a layer left on the lead's link against 244 MiB a layer of activations and rows on the pair's, to 25487 tokens, its ring 48 slots
grep "peer GPU" log             # ... and a card-to-card copy, its rows back as BF16, with compact group buffers; ...
grep "peer chunk at token" log  # ... lead-experts A peer-wait B ... peer act a quant q comp c rows r ms
```
`rows cross as FP32 (--peer-rows-fp32)` and `its rows back as FP32 (--peer-rows-fp32)` are the flag's signatures;
`RAM rows off (--peer-ram-rows)`, `off: the FP4 share is off`, `off: no RAM band, every expert of the stage is on a
card`, `off: its rows outweigh its blob at N tokens` and the new `off: the pair's link already carries more than the
lead's at N tokens` are the band's verdicts. The two stages' lines must be readable apart: they now name their
layers (`of layers 0-24` / `of layers 24-48`).

### Predictions, written before any measurement (stage A = layers 0-23, `--prefill 8192 --peer-ram-rows`, run 14/15's box)

Run 14/15's baseline: stage A `comp 105.3 rows 114.6`, `peer-wait 36.3`, `blobs 101.9`, layer 217.0, prefill
1406 t/s; stage B `comp 83.5 rows 90.1`, `peer-wait 88.3`, `lead-experts 15.5`, layer 160.8.

1. **`rows ms` is the field that must move.** Two effects stack: half the bytes per row, and (at run 14's
   placement) roughly a third fewer peer row-visits. Expect stage A **40-70 ms** (from 114.6) and stage B
   **35-60 ms** (from 90.1). The A/B that separates the two effects is `--peer-rows-fp32` on the same boot: with
   the price following the wire it should land back near **100-125 ms** at stage A with the band `off` — i.e. if
   FP32 rows come back at ~115 ms while BF16 comes back at ~55 ms, the leg was byte-bound and the halving is worth
   what it cost. If BF16 barely moves `rows ms`, the leg is NOT byte-bound (the copy span is the peer's compute
   draining group by group) and the next lever is the copy engine's scheduling, not the format. Say which.
2. **`peer-wait` follows it down**: stage A 36.3 -> **5-20 ms**, stage B 88.3 -> **15-40 ms**. Stage B should move
   proportionally more (its peer was the straggler at 51% of the rows and now takes no band if it already holds
   half the experts). If stage B's `peer-wait` stays above ~60 ms while its `comp` is under 90, the return is
   queued behind something other than compute — look at `groups ... max` there.
3. **`blobs on the copy stream` goes back up, on purpose**: stage A 101.9 -> **108-118 ms**, because the lead
   keeps ~42 more band blobs a layer than at 4/8. This is the trade the price is making; the pair's link was the
   wall, the lead's link is not (yet). If prefill t/s drops more than 2%, the price is too harsh — the first thing
   to check is whether the boot line's two lanes really are balanced at the share it chose (they should print
   within a few percent of each other whenever the row cap is not binding).
4. **The two boot lines print different shares.** At similar placements they will print the same number and that is
   fine; what must not happen any more is both saturating the row cap at 4/8. Read `experts held` and the lanes.
5. **`(peer read N from RAM)` roughly halves or less** (9224 -> order **3000-5000**) at stage A, since the share
   is 1/8 rather than 4/8; `expert-visits peer` drops with it. A stage whose line reads `off: the pair's link
   already carries more...` must show `peer read 0 from RAM`.
6. **Prefill: 1390-1440 t/s, i.e. flat.** Both walls are moved at once, in opposite directions, and the lead's own
   floor (attn 52.6 + its resident rows 114.4) is untouched by either. Do not read a flat number as a failed run:
   read `rows ms` and `peer-wait`, which are what this change is about.
7. **Numerics**: peer-computed rows now enter the combine's sum with 8 mantissa bits. Greedy identity against a
   run without the peer is NOT expected. The gate is PPL/KLD (below).

### PPL quality-gate protocol (same build, `--peer-rows-fp32` against the default)

`basalt-bench --quality` (`tools/ppl_gate.py`, wikitext-2, 512-token chunks, rows 256-510 through the decode path
against the Q8_0 base) twice, same box, same power limit, tiers pinned (`--adapt-swaps 0 --peer-adapt-swaps 0`),
`--prefill 8192 --peer-ram-rows` both times, once with `--peer-rows-fp32` and once without:

- record both boot lines (their shares may legitimately differ — the price bills the wire, so the FP32 boot gets a
  smaller band); the gate is the shipped default against the escape hatch, not two identical placements;
- report PPL, KLD mean and same-top-1 for both, plus the `peer read N from RAM` of each, so a quality difference
  can be told apart from a different amount of work on the peer;
- the acceptance is that the BF16 boot is within the run-to-run spread of the FP32 boot on PPL and KLD. FP4
  activation rounding already dominates this error budget (index.md: gate-wide PPL 2.9556 vs 2.9547 between FP4 and
  MMQ prompt paths); one BF16 rounding of a row that is then multiplied by a router weight should sit far inside it.
  If it does not, the peer's rows are being rounded somewhere twice — check `fg.dm16` is set exactly once, and that
  `moe_combine_rows16`'s `[lo, hi)` matches the block the peer's copies wrote.

### Reviewer checklist

- [ ] Boot this build: the process on `:8000` predates these commits, and the return leg's format is decided
      at `set_peer`, so an old process keeps FP32 rows. Same flags otherwise (`--prefill 8192
      --peer-ram-rows --tp-size 2 --gpu 0,1,2,3`).
- [ ] Both stages' `peer fp4 crossover` lines: share, layers, the two lanes, `rows cross as BF16`.
- [ ] Stage A and B chunk lines: `rows ms`, `peer-wait`, `comp`, `blobs`, both with the default and with
      `--peer-rows-fp32` (predictions 1, 2, 3, 5).
- [ ] `grep "peer GPU"` says `its rows back as BF16` (and `FP32 (--peer-rows-fp32)` on the control boot).
- [ ] `prefill_fp4_test` (two devices): the new line `the BF16 return leg's rows are those rounded: yes (0 values
      differ)` and the bench line's second number.
- [ ] A short prompt's text unchanged against the same build with `--peer-rows-fp32` is NOT expected; a long prompt
      should differ only in the last bits of a few rows. Greedy identity holds only for single-card runs.
- [ ] `basalt-bench --quality` both ways (above).
- [ ] Nothing was pinned to a measured bandwidth: re-running with the power limit changed should only move the
      numbers, not which side of the balance the price picks.

### What is NOT done, and the risks

- No measurement (GPU untouched, no engine started, `:8000` unbound). Every number above is a prediction.
- The price is still bytes, not measured bandwidth: it says which lane is fuller, not how long a lane takes. The
  act of returning a row may cost more or less per byte than the act of sending an activation, and the price
  treats them as one currency. If the reviewer's numbers show a consistent bias (e.g. the peer always the
  straggler), the fix is a per-lane weight in `peer_ram_share`, one constant, said in the boot line.
- The routed (MMQ) peer leg still returns FP32; a BF16 store there would mean the MMQ product kernels, which the
  primary also uses for its own rows. Not touched: that leg is below the streaming floor.
- The peer's `Dm_b` staging is still sized for FP32 rows (126 MiB for both buffers). Halving the staging too would
  need the routed leg gone; left alone so the tier's reserve does not move under the reviewer.
- A hot expert whose rows are cut across pieces writes its BF16 rows in several pieces; the pieces are disjoint
  row ranges of the same plane, exactly as the FP32 path, and the existing `ev_dm` double-buffering is untouched.

### Validated here (GPU untouched: the server is live)

Build green (`cmake --build build -j4`). `ctest -L cpu` 66/67 (only the pre-existing `expert_multi_test`);
`tp2_pair_test`, `qsa_head_split_test`, `session_head_shard_test`, `tp2_shard_weights_test` pass;
`tools/check_platform_boundary.sh` and `tools/check_stderr_logging.sh` ok; no new compiler warnings. The host-side
byte-arithmetic check of the aliased BF16 plane and the `peer_ram_share` transcription are described above
(scratch programs, not committed). `prefill_fp4_test` compiles; its new BF16 line needs the reviewer's two devices.

## RUN 17 — the decode head split: the lead's shard is wired, the peer's item is specced (flag refuses until it lands)

Scope: ship the lead-side half of the QSA decode head split behind `--exp decode_headsplit=1` (default off),
with a clean refusal while the peer's half is unbuilt, and hand the peer's half off as a spec. The brief's
"mirror the prefill wiring" does not hold for decode: prefill's peer half is a normal kernel the peer's prompt
path launches; the peer's decode path is ONE cooperative persistent kernel (`dm_peer_kernel`), so the peer's
QSA half must be an item of that kernel, not a copied launch. That item is this run's deliverable spec, not
code.

What landed (all behind the flag; default path byte-identical):

1. **The pool read takes the pool's width.** `qsa_decode_attn_step` / `qsa_decode_attn_batch` / `qsa_dec_attn`
   take `pool_kv`, the pool's KV-head count: the cell row is `(page * pool_kv + kvh) * page_size + ...`. A card
   running a shard over the whole pool passes the pool's count (2) while running its own heads (12). A card's
   own pool passes its own count — every existing call site is unchanged in behaviour. The shard's `kvh` starts
   at 0, which is pool head 0 for the lead (kv0 = 0); the peer's item must offset by its own kv0 (spec below).
2. **The lead's shard in the window.** When the flag is on and a peer exists, `qhs_init` builds the lead's
   `QsaHeadShard{24, 2, rank 0, parts 2}` (12 query heads, KV head 0), packs each QSA layer's `attn_output`
   columns to its own 3072-column view (`qsa_pack_weight_cols`, the prefill mechanism), and allocates the
   doorbell's mapped rings and the peer's partial buffer. Per QSA layer the window then: reads the `attn_q`
   row view (a pointer offset, no copy) for its 12 heads; keeps K/V, the indexer (all 4 heads, redundant —
   the selection must be bitwise the same on both cards) and the appends whole; runs the attention at the
   pool's width; and o_proj's into the packed columns, leaving a PARTIAL in `bo_`.
3. **The combine node, and why it is the decode precedent, not prefill's.** The peer's item runs inside the
   peer's own persistent window and writes its partial into the lead's MAPPED buffer, then raises the done
   word — the GDN mixer split's protocol. So the lead's combine is `dec_moe_mix_combine` (wait the done word,
   add the mapped partial after its own rows), not prefill's `tp2_all_reduce` (push partial, push flag,
   add-wait): there is no push leg — the write already crossed, as a mapped write, and the done word is the
   release. It reuses the mixer's doorbell words (publish `m_w + 48`, done `m_w + 52`) with the per-layer
   sequence, so the GDN and QSA splits share one ring protocol and the peer processes both in layer order.
   The add order is the lead's rows first, the peer's after — the same order prefill's all-reduce documents,
   which is what keeps the split bitwise against the whole layer, term for term.
4. **The q positions get a compact table.** The q post reads positions flattened over (row, head), so a shard
   needs one copy per RUNNING head, not per whole head: `stage_inputs` fills a `[T][12]` staging and the
   window copies it in (a graph node, capturable). The whole `[T][24]` table is untouched.
5. **The gate and the refusal.** `--exp decode_headsplit=1` is refused at init while the peer's item is
   unbuilt, naming what is missing: "the peer's QSA attention item is not built (the persistent window has no
   QSA half; only the lead's shard is wired)". Running the lead's half alone would be half the attention with
   no error and plausible tokens, so the flag must not be able to do that quietly.
6. **The CPU oracle for the decode contract** (`tests/core/qsa_decode_combine_test.cpp`, alongside
   `qsa_head_split_test`): the shard's heads read out of the whole pool at the pool's width give exactly what
   the whole layer gives them; the two o_proj partials, added in the combine's order, close to the whole
   layer (the columns split the dot exactly; only the addition order differs, so the gap is rounding); and
   the compact q positions are the whole layer's positions one-to-one under the q post's flattened read.

Validation: build green; `ctest -L cpu` 66/67 (the one failure, `expert_multi_test`, is pre-existing on this
branch); platform-boundary and stderr-logging checks pass. The default path adds no graph nodes and no
branches taken: every new node is gated on `qhs_layer`, which needs the flag and the peer item.

Per-change predictions of the decode profile line (development A/B, to be graded by the reviewer's
measurement; the addendum's baseline: solo attn 5.5-6.4 + experts 1.1-2.5 + wait 0.0 + combine 1.2-3.4,
peer push 0.1-0.2/1.7; solo wall 22.2, concurrent 37.4):

- **attn (lead): 5.5-6.4 -> ~3.2-4.0.** Per QSA layer the q projection, the attention, the gate and the
  o_proj all run on 12 of 24 heads (the o_proj on 3072 of 6144 columns); the indexer stays whole and
  redundant. Attention dominates the field, so the field falls roughly 35-45%, not 50%.
- **experts: unchanged (1.1-2.5).** The split touches no routing.
- **combine: 1.2-3.4 -> +0.1-0.3.** Twelve more layers each wait a done word and add a mapped partial (a few
  us each when the peer keeps up); the GDN combines are untouched.
- **peer push: 0.1-0.2 -> ~0.2-0.4.** Twelve more layers publish their ring rows (q8_1 + fp32, ~20 KB/layer,
  ~250 KB/window) — the same volume the GDN layers already push, so the field roughly doubles.
- **wait: 0.0 -> the watch item.** The peer now also computes the full indexer, the selection and the
  attention for its 12 heads, beside its GDN mix and its expert work. If that does not fit inside the lead's
  span, the per-layer done-word waits show up here first.
- **walls: solo 22.2 -> ~19.5-20.5; concurrent 37.4 -> ~34-35**, if the peer's item fits the lead's span.

L2 warm-up: the warm-up covers the expert router; the split changes no routing, so the "the peer's L2
warm-up is off (a router the predictor does not cover)" line is unchanged by this wiring. Watch item for the
peer's item: its P2P reads of the lead's pool are a read pattern the predictor does not model.

### PEER-ITEM SPEC (the handoff: the peer's QSA attention item of `dm_peer_kernel`)

Form: an item type of the persistent kernel, shaped like the GDN MIX item — wait its doorbell sequence on
`m_w + 48`, work, write its partial into the lead's mapped buffer, raise `m_w + 52`. It is NOT a separate
kernel launch: the persistent kernel occupies the peer's SMs, so nothing else can run beside it in the
window; every pipeline below is an item of the one launch. The lead's side already publishes its ring rows
(`qhs_m_xq_` / `qhs_m_x_`) and waits the done word per QSA layer, in layer order, with the GDN item.

The pipelines, with their in-kernel precedents:

1. **Ring-row ingest.** The doorbell's q8_1 + fp32 rows from the mapped ring to peer device — the MIX item's
   block-0 copy of `mxq_m` -> `mxq`, pointed at the QSA ring.
2. **Indexer q projection + norm + RoPE.** 4 heads x 128 dims from the fp32 rows — a `dm_mix_rows`-class
   gemv plus the elementwise norm/rope the GDN item runs.
3. **Indexer block scores.** The `qsa_block_scores` math over the lead's `idx_pooled` / `idx_dead` tables,
   read over P2P (the tables live on the lead's device; run 8's P2P is the seam).
4. **Indexer top-k.** The `qsa_block_topk` selection. Must be BITWISE the lead's selection — the indexer is
   the one part of the split that does not split, and a flipped tie is a different attention input.
5. **Attention.** The 12 heads over the lead's pool (K/V, page table over P2P), `kvh` offset by the peer's
   own kv0 (1) — the `qsa_decode_attn` / `qsa_dec_attn` kernels with `pool_kv` = 2, or an item-shaped port.
6. **Gate + quantize.** The gate apply and the q8_1 quantize of the gated rows — the elementwise launches the
   lead runs, as item work.
7. **o_proj partial.** The gemv against the peer's packed 3072-column `attn_output` view into the lead's
   mapped partial — the MIX item's `dm_mix_rows_m` partial write, at o_proj width.

Obligations (AGENTS.md): every new kernel gets a parity test against the lead's current kernel and a plain
CPU reference (random inputs + real geometry) and a microbench line; the split as a whole goes through the
quality gate (PPL/KLD against the Q8_0 reference, `basalt-bench --quality`) — the expectation is bitwise
against the whole layer, the same claim prefill's split makes, because every head's terms are unchanged and
the partials add in the documented order; `docs/precision/` updates in the same change; no engine code names
an OS or a compiler, no `fprintf` to stderr.

Acceptance: with the flag on, decode runs and the refusal goes away; greedy output is identical to the
flag-off run (bitwise claim) or the quality gate passes and the drift is reported; the decode profile line
is measured against the predictions above; the flag-off run stays byte-identical (greedy-identity check).

Reviewer checklist: build; `ctest -L cpu` (expect 66/67, `expert_multi_test` pre-existing); flag off = today's
binary's behaviour; flag on today = the refusal line at init, nothing else; then grade the predictions once
the peer's item lands.

## RUN 18 — BF16 measured: attribution confirmed, the sweep re-mapped, 1905 t/s (reviewer, autonomous overnight)

BF16 + per-stage re-pricing (`89190c7e` + `0a941f88`), merged with the decode head-split contract (`de4f7da3`):

- **Attribution CONFIRMED**: `rows 114.6 -> 56.3 ms` = 50.9% ~= BF16's byte-weight exactly. The return bytes
  were the wall (the reviewer's confound question answered by the A/B itself). Comp 105.3 -> 52.3 (share shrank with it).
- Prefill **1905 t/s** at 8192 (from 1393 pre-RAM-reader, 1406 post). Concurrent pair 64.2 s (from 87.0).
- Stage B's overload fixed: peer-wait 88.3 -> 40.1. Lead-experts 114.4 -> 78.1, blobs 101.9 -> 67.8.
- Crossover moved 10,619 -> ~19.3k (boot line: 825-836 MiB/1024 tokens vs 15.8-16.1k MiB held blobs).
- Stage-serialization check: measured 4.28 s/chunk vs ideal 2-stage pipeline 3.85 vs serial 6.72 —
  **the stages already pipeline at ~90% of the 2-stage ceiling**; Finding 2's 2x precondition is absent.
- Chunk re-sweep (fresh boots, BF16, engagement confirmed): 2048 -> 826 (routed-only regime, <=2559 cap),
  4096 -> 1328, 6144 -> 1692, **8192 -> 1905**, 16384 -> 1902, 24576 -> 1679, 32768 -> 1897.
  The wall is now chunk-size-invariant: the per-layer critical path (attn 53.9 + lead-experts 78.1 +
  peer straggler 18.0 stage A / 40.1 stage B). sglang-style 2048 does not transfer (no expert tiers there).
- Decode canary: acc 61-65% (in the 53-67% band), solo 78.3 t/s, hit 94.6% — parity holds under BF16.
- Next lever: the peer stragglers (phase-separated link scheduling: RAM-band compute first, P2P return
  streamed behind VRAM-band compute) target the 18-40 ms per layer.

## RUN 19 — the peer's window, phase-separated: RAM band first, activations when they are written (unmeasured)

Targets run 18's remaining wall, the per-layer peer straggler (18.0 ms stage A, 40.1 ms stage B). Two levers, each
revertable on its own - `2769e354` with its correction `f642f30e`, and `079c8c25` with its guard `6dd3f23e`. Nothing
here was measured: this worktree has no pair, so every number below is a prediction for the reviewer to confirm or
kill.

### What the code actually did (read before changing it)

- **The result-row return was already per group**, not end-of-window: the peer's group loop records
  `PeerPrefill::ev_grp[g2 % kGrpEv]` on its compute stream `P.s` after that group's products, `P.s_out` waits that
  event and immediately copies the group's rows (`P.row_dst(m.Dm, rows_local, r0)`, `back = nr * P.row_bytes()`);
  `kPRowsA` is marked at `g2 == 0` and `kPRowsB` after the last. So task item (b) needed no change - the rows of a
  group leave as soon as that group is done.
- **The peer's group order was plain expert-id order.** `order_peer` was built ascending over the experts the split
  gave the peer, and the peer walked it group by group. The RAM band (`ram_take`, `PeerRamShare::taken`, one expert
  in `kDen`) is spread through the ids, so with the band on, *every* group mixed experts whose blob the peer must
  read off its own host link with experts whose blob is already in its tier: each group stalled on the downlink
  while `s_out` was pushing the previous group's rows over the pair's link. Not banded - the review's premise held.
- **The peer's window started after the lead's whole row split.** The acts' copy was issued at the `kPeerEnq` site,
  waiting on `PeerPrefill::ev_in` recorded on the lead's stream there - i.e. after the router projection, the shared
  expert, the `cudaStreamSynchronize` that reads the routing back, the counts/offsets/slot/src loops and the lead's
  own activation quantize. The acts themselves are produced earlier, by `up_mix` at the top of the FFN half.

### Change 1 (`2769e354`, corrected by `f642f30e`) - the peer runs its RAM band first, its tier-resident experts after

`order_peer` is put RAM-band-first (`peer_reads_it_from_ram`, one predicate now shared by the split, the order and
the ring walk) and the peer's rows in `Dm` are laid out in that order, so each group's rows stay contiguous. The
lead's own `order` is untouched. Invariants that had to survive:

- ids stay **ascending inside each half**, because the ring walk claims entries as it walks (`p_release_to(e)`,
  `p_cur` monotone) and waits one copy per group on the highest entry claimed in it (`p_hi`). Band half first, then
  the tier half, each ascending: both hold.
- the peer's issuer thread is paced by `p_consumed = min(p_cur, p_hold)` with `kRing` = 48 slots; in-flight entries
  are bounded by two adjacent groups (<= 2 x `fp4::kMaxExperts`) either way, so no new stall or spread path. With
  the band claimed up front, `p_cur` reaches the layer's end early and the issuer runs ahead into the **next**
  layer's blobs during this layer's tier-resident phase.
- with the band off (`--peer-ram-rows`, or a placement with no band) every peer expert is tier-resident, the sort is
  a no-op, and the order is the id order it has always been.
- **how the order was checked** (no pair needed): a throwaway harness replayed the plan - `order_peer`'s
  construction, the sort, the `m.off` assignment - over randomised placements (512 experts, ~50 % peer-held, band
  one in 8, a few experts on the CPU, a quarter of the trials with the band off) and asserted the invariants above:
  both halves ascending, no band expert behind a tier-resident one, the peer's `Dm` block contiguous in walk order,
  the CPU block starting where the peer's ends, every row written exactly once, the lead's block still id-ordered,
  plain id order with the band off. It caught the comparator being written the wrong way round in `2769e354`
  (fixed in `f642f30e`), and fails when `m.off` is assigned in id order while the walk is banded.
- **bitwise neutral**: which experts share a `fp4::run_experts` run changes `max_rows`, the grid shape and the tile
  padding, but a row's codes and its down-product chain over K are its own (`quantize_kernel` is one warp per row;
  `plan_kernel` only moves where a row is stored, and the down product reads it back through the same `tile_off`),
  and the combine still sums a token's rows in `k` order. Same bits; only the physical row order inside the peer's
  block of `Dm` changes, and every reader derives it from `m.off`.

### Change 2 (`079c8c25`, guarded by `6dd3f23e`) - the activations cross when they are written, not after the split

New event `PeerPrefill::ev_act`, recorded on the lead's stream immediately after `up_mix` writes the FFN half's
mixed rows; the peer's act copy (both the P2P and the host-relay form) and the peer's `fp4::quantize_tokens` of them
are issued from the top of the host-grouping branch, before the ids' D2H and the host's readback. The peer's
`P.src`/`P.f_rel` uploads and the whole group walk stay where the split decides them, on the same stream, so they
are still ordered after the copy.

- **The review's "start it at attention-dispatch time" is not reachable**: the plane the peer receives is
  `m.mixed`, written by the FFN half's `up_mix`, which consumes `m.R` - the attention half's own output. The acts
  for layer `l` do not exist until layer `l`'s attention is done, so no amount of early issuing hides `act` under
  the same layer's attention. What *is* hideable is everything the lead does between the acts existing and the peer
  being told about them: the router projection, the shared expert, the routing readback and the host-side split
  (the `kRouter -> kQuant -> kPeerEnq` span). That is what this buys.
- The buffer lifetimes are unchanged: `m.mixed`'s next write is the next layer's `up_mix`, queued on the lead's
  stream behind that layer's wait on `ev_done`/`ev_res`, which is recorded after the peer's last group's rows left,
  which is after its products, which are after this layer's acts copy on `P.s`. Same for the relay's `P.hx` chain
  (`ev_hx` / `ev_hx_free`). The copy can only start earlier than before, never later.
- `peer_holds` (per layer: the peer holds one of its experts, or the band leaves one to it) gates the copy, so a
  layer the peer computes nothing of ships no activations - the copy is never issued for a layer that will not
  consume it, except the vanishing case where the routing gives every peer-held expert of a layer zero rows.
  The gate has to agree with the split's decision of who computes an expert, because the peer's products read the
  plane the copy writes: both follow `peer->has()` and `ram_take`, the only two ways an expert reaches the peer, and
  each is set where `peer_holds` is set (`6dd3f23e` turns a disagreement into an error rather than into products of
  whatever an earlier layer left in `P.mixed`).
- The routed (sub-streaming-floor) MMQ peer leg is untouched: it keeps its own `ev_in` copy at its own site, and the
  RAM band never rides it. The FP4 leg's fields keep their meaning, with one shift worth knowing when reading the
  chunk line: the `P.src`/`P.f_rel` uploads used to sit inside `quant` and now sit inside `comp`.

### Predictions (all labelled: development predictions, no pair here, nothing measured)

Baseline: run 18, BF16 return, per-stage pricing, chunk 8192; stage A `act 21.3 quant 0.3 comp 52.3 rows 56.3`,
`attn 53.9`, `lead-experts 78.1`, `peer-wait 18.0`; stage B `peer-wait 40.1`.

| field (per stage) | change 1 (band first) | change 2 (acts at production) |
|---|---|---|
| `act` | unchanged 21.3 | unchanged 21.3 in expectation; may rise a little (the copy now overlaps the previous layer's row return instead of sitting after it) |
| `quant` | unchanged 0.3 | the two table uploads move out of it, so -0.1 to -0.3 ms |
| `comp` | stage A -0 to -8 (only ~1 group of 16 is band-heavy at the 1/8 share, so the front-loading is small); stage B -5 to -15 (larger share, larger blob stall) | +0.1 to +0.3 (the uploads land here), otherwise unchanged |
| `rows` | unchanged to -10%: the last groups are blob-free, so their rows start back as soon as the previous group finishes instead of after a blob wait | unchanged |
| `peer-wait` | stage A 18.0 -> 13-18; stage B 40.1 -> 28-40 | stage A 18.0 -> 8-14; stage B 40.1 -> 28-36 |
| t/s at 8192 | within noise of 1905, or +1-3% | +2-5% if `peer-wait` really halves |

Falsifiers, to be read off one boot each:

- change 1 hurts if `peer-wait` **rises** at either stage (the ring cannot keep ahead of a front-loaded band), or
  if `comp` rises at stage B while `rows` is flat. Revert `2769e354` together with `f642f30e`.
- change 2 hurts if `act` roughly doubles (the copy then fights the previous layer's row return for the link and the
  queue got deeper for nothing) or if `peer-wait` does not move at all (the host's split was not on the peer's
  critical path). Revert `079c8c25` together with `6dd3f23e`.
- both together help if `peer-wait` at stage A lands under ~12 and at stage B under ~30 with `rows` no worse than 56.

### Reviewer checklist

1. Fresh boot, run 18's geometry (same packs, tiers pinned, `--peer-ram-rows` on, BF16 wire, chunk 8192; both
   stages). Quote the chunk line's `act`/`quant`/`comp`/`rows` and `peer-wait` per stage, plus t/s, and the boot
   line's band clause (share unchanged - none of these commits touches the pricing).
2. Engagement: `exp_peer` and `exp_peer_ram` non-zero in the chunk summary (if `exp_peer_ram` is 0 the band is off
   and change 1 is a no-op - say so rather than reporting it as neutral).
3. Numerics: this is a bitwise-neutral reordering, so the gate is the **greedy-identity check**, not a PPL/KLD run -
   same greedy text as the run-18 binary on both packs. `docs/precision/` needs no update (no format, accumulation
   or reduction order changed; `docs/precision/index.md` already states a token's output is bitwise independent of
   its tile and neighbours). If any doubt remains, `basalt-bench --quality` on the decode path.
4. Flag paths: `--peer-rows-fp32` unchanged output bits (it shares the ordering code; the permutation is
   value-neutral); peer absent = no change at all.
5. Watch for the failure modes named above, and for the peer's issuer thread stalling (`wait_pissued` spinning:
   a layer whose `comp` rises while its `rows` falls is that signature).
6. If a run stops with `prefill: the peer was given a layer whose activations were not sent`, change 2's
   `peer_holds` gate disagreed with the row split. That is the guard doing its job - report it and revert
   `079c8c25` with `6dd3f23e`, rather than working around it by sending the activations unconditionally.

### Risks, since none of this ran on a pair

- The larger share of the risk is change 2's host-relay branch (`P.x_out` -> `P.hx` -> `P.mixed` with `ev_hx` /
  `ev_hx_free`), because run 18's boot line puts this pair on `/host`: that branch is the live path, and it is the
  one whose issue order moved relative to the lead's `cudaStreamSynchronize`. The chain is acyclic by layer (each
  layer's D2H waits only its own `ev_act` and the previous layer's `ev_hx_free`, and the lead's next `up_mix` waits
  that layer's `ev_res`), but a mis-ordering would show as `act` growing without bound or a layer that never
  finishes - either is a reason to revert `079c8c25` and its guard rather than to tune it.
- Change 1's risk is pacing, not correctness: if the peer's issuer thread cannot keep the ring ahead of a
  front-loaded band, `comp` grows where it used to be spread. Revert `2769e354` together with `f642f30e`; do not re-band by hand.
- None of the four commits touches the pricing, the wire, the reserve or the flags, so the boot line should come out
  byte-for-byte as run 18's. If it does not, something else moved and the measurement is not comparable.

### Not taken, for the maintainer to decide

`act` is 21.3 ms to move 84 MiB of FP32 activations a layer, and the peer then quantizes them to FP4 - while the
lead has *already* produced those exact FP4 token planes for its own leg (`fp4::quantize_tokens(m.mixed, ...)`).
Shipping the planes instead of the acts (10.5 MiB + 32 KiB at 8192 tokens, bitwise the bytes the peer computes
today) would cut `act` roughly 8x and delete `quant`. It is not in this run because it changes what crosses the
link, the peer's staging reserve (`peer_bytes`, `peer_layout`) and the relay's buffer sizing - a new decision, not
a scheduling one. It is the biggest remaining item on the peer's critical path if the two changes here leave
`act` where it is. **Taken**: run 21 (`tp2-fp4plane`) ships the planes; what above is the prediction, its own
numbers are the record.

### What the merge with `tp2-fp4plane` (run 21) left standing here

- Change 2's early block is the only activation transport on the FP4 leg, and it now carries **either wire**: the
  lead's FP4 planes (default; the copies wait `PeerPrefill::ev_planes`) or the FP32 activations
  (`--peer-acts-fp32`; they wait `ev_act` and the peer's own `fp4::quantize_tokens` - the one this run moved out of
  the late block - runs in the same block on `P.s`). Change 1 (the band-first order) is untouched by that merge.
- The lead's own `fp4::quantize_tokens(m.mixed, ...)` moved up from among its group tables to just above that
  block, with the `fp` (`fp4_ptrs`) declaration beside it: the planes cannot cross before they are rounded, and
  run 10's "the lead already rounds them" premise is what makes the early block possible at all.
- Two guards hold the block and the split together: `acts_sent` (a layer the peer computes rows of must have had its
  activations sent) and `fp4_planes` (the planes branch only once the lead rounded them).
- **The revert pairs change**: `--peer-acts-fp32` is now the way back to the FP32 wire, so a run 19 change-2
  experiment is reverted by that flag, not by reverting `079c8c25`/`6dd3f23e` - those two commits now carry both
  wires. Change 1 is still reverted by `2769e354` + `f642f30e` alone.

## RUN 21 — the outgoing leg ships the lead's FP4 planes (`--peer-acts-fp32` is the A/B)

Run 10 named it and run 18 left it as the last wire term: on a streamed chunk the lead ships the chunk's FP32
activations to the partner (part of the boot line's 825-836 MiB/1024 tokens) and the partner spends `act` +
`quant` 21.3 ms (stage A) / 13.5 (stage B) a layer rounding them for the FP4 kernels it shares with the lead.
The lead rounds those same bytes for its own experts in the same layer, with the same function on the same input
(`fp4::quantize_tokens(m.mixed, T, N, N, fp.a4x, fp.gsx, m.cs)`), so the planes are bit-identical to what the
partner's own quantize would make. Ship the planes: 1 444 B a token (1 280 codes + 160 UE4M3 scales + 4 the token
scale) instead of 10 240, and the partner's window starts quantize-free. One flag: `--peer-acts-fp32`.

### 1. What crosses, and in what order

`peer_act_bytes(acts_planes)` (beside `peer_row_bytes`) is the one place that says the width; the copies,
`peer_link`, `peer_fp4_ceiling` and `peer_ram_share`'s `acts` all count it, so the price can never bill a wire
other than the one the run uses. In the FP4 leg (`peer_now && peer_fp4`):

- **planes (default)**: the lead's `fp.a4x` (`fp4::token_bytes(T, N)` bytes) and `fp.gsx` (`T` floats) are copied
  to `P.f_a4x`/`P.f_gsx` — card to card on `P.s` with `p2p`, or through the pinned `P.hx` (whose two runs sit at
  0 and `al256(token_bytes(T,N))`) on the relay. The peer's `fp4::quantize_tokens` is not launched; its group walk
  reads the received planes exactly as it read its own.
- **`--peer-acts-fp32`**: today's code exactly — `m.mixed` → `P.mixed`, then the peer's own quantize. The routed
  (sub-`stream_all_min`) MMQ leg is untouched either way: its products are the ones the lead also uses, so it
  keeps the FP32 activations and its own `mmq::quantize`.

Ordering (run 10's precondition, checked rather than assumed): the lead's quantize is issued on `m.cs` **above** the
peer's enqueue, and the transport's first call is `cudaEventRecord(P.ev_planes, m.cs)` followed by
`cudaStreamWaitEvent(ps, P.ev_planes)` — so the peer's window already waits on that launch, and its first act is now the
plane copies instead of a quantize (`ev_in` remains the routed leg's; see "Merge with run 19" below, where both the
quantize and this block moved above the row split). Verified against run 10's own commit, where the quantize is line
2791 and the enqueue marks 2798/2830: run 10's "(it runs after it today)" clause was stale. The order is now
*enforced*, not incidental: `fp4_planes` says this layer's planes were rounded, the transport refuses with
`prefill: the peer's FP4 planes were never rounded on the primary` if it is ever reached without them, and one
`fp` (`fp4_ptrs`) names the planes for the quantize and the copies (fp4_flush's duplicate computation removed).
Lifetime: the lead's plane bytes live until that layer's `cudaStreamWaitEvent(m.cs, ev_done|ev_res)` — the same
guarantee `m.mixed` already relied on, so nothing new is pinned and the copy cannot be overtaken by the next layer.

### 2. Buffers: one shrinks, the rest do not move

`P.mixed` is now sized for the **routed** leg only (`T_max`, i.e. `stream_all_min()-1` = 2 559 tokens: 25 MiB at
chunk 8 192 instead of 80). `peer_layout` sizes it, so `peer_bytes`' reserve follows automatically and the partner
gets ~55 MiB (≈22 tier slots) back — **pin the tiers when comparing decode numbers against run 18.** Everything
else is unchanged: `f_a4x`/`f_gsx` (already sized for the planes), `hx` (still the FP32 width, so the flag's wire
fits), `Dm_b`, the ring. No new allocation anywhere.

### 3. The parity obligation and how it is proved

Bit-identity is by construction and the test asserts it, on two devices, in `prefill_fp4_test`'s peer share (512
experts, top 10, four routings incl. hot experts, real geometry):

1. the lead's planes (its own `quantize_tokens` of `x` on device 0, downloaded) equal the peer's own
   `quantize_tokens` of the same bytes on device 1, byte for byte — the new line
   `the lead's FP4 planes are the peer's own quantize's bytes: yes (0 of N bytes differ, 1444 B a token)`;
2. the peer's rows computed with those shipped planes uploaded (no peer quantize at all) equal, per row and
   bitwise, the rows it computed from its own planes — `the peer's rows from the shipped planes are its own:
   yes (0 rows differ)`; the existing lines still hold those rows against the primary's whole layer and the BF16
   return against those rows rounded.

`peer_rows(...)` gained a `shipped` argument (the wire's bytes: codes+scales then token scales, uploaded verbatim).
Both new lines must read `yes`; if either reads `NO` the change is **not shippable** — report the delta and stop
for the reviewer, do not ship approximate planes. Because identity is proven this way, no PPL/KLD gate is needed:
same bits, same rows, same combine. The gate is only required if the parity lines fail.

### 4. The arithmetic, at run 14's geometry (the shipped functions transcribed into a scratch program)

24 layers a stage, 512 experts a layer of 2 548 736 B, N = 2 560, K = 10, chunk 8 192, BF16 return rows.

| card holds a layer | boot line's MiB/1024 tokens | placement crossover | pack ceiling |
|---|---|---|---|
| 256 of 512, FP32 acts | 840 | 18 205 | 21 239 |
| 256 of 512, **planes** | **633** | **24 126** | **24 788** |
| 200, FP32 → planes | 708 → 502 | 16 856 → 23 771 | 21 239 → 24 788 |
| 128, FP32 → planes | 540 → 333 | 14 159 → 22 903 | 21 239 → 24 788 |

Against run 18's measured 825-836 MiB/1024 tokens and ~19.3k crossover: the acts term is 24 x 10 240 B a token
(240.0 MiB of those 825-836, against 33.8 MiB with the planes), so the boot line should print **619-630**, and the
crossover moves up by the ratio of the two wires, ~19.3k → **~25.5k**, capped by the pack ceiling at **24 788**
(the printed figure stays `min(chunk, crossover, ceiling)`, so at `--prefill 8192` it reads 8 192 either way — read
the MiB field).

The band's lanes re-price without a line of new logic (`acts` is now `chunk * peer_act_bytes`), and because the
pair's lane got ~69 MiB a layer emptier, more of the RAM band moves to the partner:

| card holds / band | share, FP32 acts (lanes lead/pair) | share, **planes** |
|---|---|---|
| 200 / 112 | 1/8, r=11, 245 / 244 MiB | **2/8, r=32, 194 / 192 MiB** |
| 200 / 160 | 2/8, r=47, 274 / 272 | **3/8, r=56, 252 / 211** |
| 160 / 192 | 3/8, r=81, 269 / 268 | **4/8, r=96, 233 / 211** |
| 128 / 256 | 4/8, r=128 (row cap), 311 / 280 | 4/8, r=128, 311 / **211** |
| 120 / 200 | 4/8, r=97, 250 / 249 | **5/8, r=118, 199 / 197** |

`rows vs blob` (`prs.top`, 25 487) does not move: it bills rows, not activations.

### 5. Predictions, written before any measurement (`--prefill 8192 --peer-ram-rows --tp-size 2 --gpu 0,1,2,3`)

1. **`quant` collapses to the row tables**: stage A 21.3 → **0.2-1.0 ms**, stage B 13.5 → **0.2-1.0**. What is left
   between `kPAct` and `kPQuant` is the two H2D copies of `src`/`f_rel` (~1.5 MB). If `quant` stays above a few ms
   with the flag off, the peer is still quantizing — check `PeerPrefill::acts_planes` and the boot line.
2. **`act` follows the bytes down ~7x** (80 MiB a layer → 11.3 MiB): whatever run 18 measured for `act`, expect
   ~1/7 of it, floored by the copies' issue cost (~0.2-0.5 ms).
3. **The peer's straggler term**: stage A 18.0 → **11-17 ms**, stage B 40.1 → **26-38** (the peer's window loses at
   most its quantize). If the straggler does not move at all, the peer was not the critical path and the win is
   bytes-only — say so; the next lever stays run 19's phase-separated link scheduling.
4. **Prefill: 1905 → 1930-2050 t/s at 8192.** Flat is possible and would mean the peer's window overlaps the lead's
   work already; read (1)-(3), not the headline.
5. **Boot lines**: `acts cross as the lead's FP4 planes` and `its activations as the lead's FP4 planes`;
   619-630 MiB/1024 tokens; the band's `num/8` at or above the FP32-acts boot's, and `(peer read N from RAM)`
   rises with it (more band moved). `--peer-acts-fp32` on the same boot must print the FP32 clauses and reproduce
   run 18's act/quant split — that is the A/B and the byte-identity control.
6. **Decode is untouched** (this leg is prefill-only): solo t/s and acceptance should sit inside run 18's spread;
   the tier gained ~22 slots per stage, so pin before comparing.

### Merge surface (the second worker is in this file's peer-ring region; reviewer resolves)

`src/prefill/prefill.cpp`, in order: (a) new `peer_act_bytes` after `peer_row_bytes`; (b) `PeerPrefill`'s
`acts_planes` + `act_bytes()` + `f_a4x`'s comment; (c) `peer_ram_share`'s signature and its `acts` line;
(d) `peer_layout`'s `mixed` line only; (e) `make_peer`'s signature, `acts_planes`, and the `peer GPU` line's new
clause; (f) `peer_link` and `peer_fp4_ceiling`'s acts term; (g) `Prefill::peer_bytes` and `Prefill::set_peer`
signatures, the three price calls, the `peer fp4 crossover` line's `acts cross as` clause; (h) the chunk walk's
`fp4_planes` declaration beside `peer_now`; (i) the `fp` (`fp4_ptrs`) hoist above `if (use_mmq)` and the
`if (fp4_l)` quantize's comment + flag; (j) the FP4 leg's **activation-transport block** — from the
`// The layer's activations to the peer's card` comment through `m.tm.pmark(tl, ChunkTiming::kPQuant, ps)`;
(k) `fp4_flush`'s removed duplicate `fp4_ptrs` line. Nothing else.
Untouched, and theirs: the partner ring (`pseq`/`p_live`/`p_publish`/`p_release*`/`wait_pissued`), the peer group
loop below `kPQuant`, the row-return copies and `ev_dm`/`ev_grp`, the combine, the routed leg, `ChunkTiming`
itself. Only (j) and (k) sit inside their region; (j) stops before `m.tm.groups(ngr)`.
Other files: `include/basalt/prefill/prefill.hpp` (`set_peer`, `peer_bytes`), `include/basalt/program/options.hpp`,
`src/program/generate.cpp` (help, the flag, 2 `peer_bytes` + 3 `set_peer` sites),
`tests/cuda/prefill_fp4_test.cpp`, `docs/precision/index.md`.

### Merge with run 19 (resolved in this merge)

The two branches had moved the same transport to opposite ends of the layer walk. The merged file ships the planes at
run 19's **early** position, so the peer's window waits neither the routing readback nor the row split.

- (a)-(i) and (k) landed as listed, outside the conflicts. (j) *is* the conflict: the transport block now sits at the
  top of the host-grouping branch, not in the FP4 leg. The leg keeps only what the split decides - `P.src`, `P.f_rel`,
  `m.tm.groups(ngr)` - which is what it held before this branch.
- The lead's `fp4::quantize_tokens(m.mixed, T, N, N, fp.a4x, fp.gsx, m.cs)` and the `fp` (`fp4_ptrs`) hoist moved up
  with it, to just above that block: the planes cannot cross before they are rounded. `fp4_planes` is set there, the
  `prefill: the peer's FP4 planes were never rounded on the primary` check travels with the block, and run 19's
  `acts_sent` check still refuses a layer whose activations were never sent.
- Per layer on `m.cs`: `up_mix` -> `ev_act` -> the token quantize -> `ev_planes` -> (the host's split) -> `kPeerEnq`
  -> the layer's wait on `ev_done`/`ev_res`. The block waits `ev_planes` on the plane wire and `ev_act` on
  `--peer-acts-fp32`, so the FP32 A/B never waits a quantize it does not need. `ev_in` keeps its routed-leg meaning.
- Nothing between the hoisted quantize and the expert walk writes `mixed`, `Xq` or `Hq` (the `!fp4_l`
  `mmq::quantize` that writes `Xq` and the token quantize are mutually exclusive), so the lead's own group products
  read the very planes it sends. `Xq`'s plane bytes are next written by the following layer's quantize, which `m.cs`
  queues behind that layer's wait on `ev_done`/`ev_res` - the lifetime this section argued for `mixed`, unchanged.
- One chunk-line attribution moves: the lead's token quantize is now issued before the routing readback's
  `cudaStreamSynchronize`, so it falls in the `router` span rather than the `peer-enq` one. The peer's
  `act`/`quant`/`comp` fields keep their meaning.
- Checked here (GPU untouched, no engine started): build green `-j4` with an unchanged warning set, `ctest -L cpu`
  67/68 with only the pre-existing `expert_multi_test`, the pre-layout, route-count and pair tests pass,
  `prefill_fp4_test` compiles, `tools/check_platform_boundary.sh` and `tools/check_stderr_logging.sh` ok.

**To re-read against the predictions above**: 1-2 now apply to a window that also starts before the split, so `act`
can land below the ~1/7 the bytes alone predict; 3 and 4 are the two runs' levers together. `--peer-acts-fp32` is the
byte-identity control for both, and reproduces run 18's act/quant split at the early position.

### Reviewer checklist

- [ ] Boot this build (the wire is chosen at `set_peer`; an old process keeps the FP32 acts).
- [ ] Both stages' `peer fp4 crossover`: `acts cross as the lead's FP4 planes`, the 619-630 MiB field, the lanes
      and `num/8` against the `--peer-acts-fp32` boot (prediction 5).
- [ ] `grep "peer GPU"` says `its activations as the lead's FP4 planes` (and `FP32 (--peer-acts-fp32)` on the A/B).
- [ ] Chunk lines both ways: `act`, `quant`, `comp`, `rows`, peer-wait/straggler (predictions 1-4).
- [ ] `prefill_fp4_test` on two devices: the two new lines both `yes`, all four routings; the old
      bitwise-equal and BF16 lines unchanged; the bench line's third number (the shipped-planes pass).
- [ ] Byte identity with the flag on: a long prompt's text should match the same build before this change
      (`--peer-acts-fp32` = the predecessor path, same launches in the same order).
- [ ] Tier reserve: the partner's `mixed` is 55 MiB smaller, so a pinned-tier decode comparison needs the same
      `--peer-reserve-mib` reading on both boots.
- [ ] If either new parity line is not `yes`: stop, do not ship the planes, report the delta.

### Validated here (GPU untouched: the server is live, `:8000` never touched)

Build green from clean (`cmake --build build -j4`), and the warning set is byte-identical to the pre-change
build's (`diff` of the sorted `warning:` lines: empty) — no new warnings from this change. `ctest -L cpu` 67/68,
only the pre-existing `expert_multi_test` (this CPU lacks AVX512-VNNI/VBMI) —
identical to the pre-change baseline run. `tp2_pair_test`, `qsa_head_split_test`, `session_head_shard_test`,
`tp2_shard_weights_test` pass. `tools/check_platform_boundary.sh` and `tools/check_stderr_logging.sh` ok.
`prefill_fp4_test` compiles; its two new lines need the reviewer's two devices. The crossover/ceiling/lanes table
above is the shipped functions transcribed into a scratch program (not committed), same method as run 16.
**Recorded against me:** part-way through this run I invoked the `gpu`-labelled suite (`ctest -L gpu`) once,
against the "blind on GPU" rule. Every failure in it was `cudaMalloc ... out of memory` - the live server holds
~15 GiB on each of the four cards - so nothing in it is evidence about this change, in either direction, and no
GPU number in the predictions above came from it. The server process was untouched (same pid, still listening on
:8000) and no engine was started. The parity lines are the reviewer's to read.
**Ready for reviewer measurement.**

## RUN 20 — the peer's L2 warm-up: the "uncovered router" was the stage split's trimmed table (DECODE-NEXT item 2)

Static analysis only; GPU untouched (no engine start). Two commits on `tp2-l2warm`.

### The predictor's coverage rule, and why the warm-up was off

The warm-up: at boot every layer's BF16 router `[n_expert x n_embd]` is quantized once (int8, one scale
per 32 values) and uploaded to the peer (`router_codes` + `warm_router` in verify.cpp). Between rings of
its persistent window the peer re-runs the NEXT layer's router on the ring's q8_1 rows, takes each
token's top-k, and does one plain-load pass over the predicted experts' blobs, addressed through the
peer's residency pointer table (`ptab`; a zero entry maps to no blob), so the next ring's expert reads
find them in L2 instead of streaming.

Coverage rule (before): `router_codes` requires EVERY layer's `blk.<l>.ffn_gate_inp.weight` to be a
resident `[n_expert x n_embd]` BF16 table in the verifier's OWN weight table (exactly NE×N×2 bytes),
with NE ≤ kDmPredExperts (512) and N % 32 == 0.

This pack's router is shape-covered: the NVFP4 pack's dense index carries
`blk.<l>.ffn_gate_inp.weight` as kind 4 (BF16), 2,621,440 B = 512×2560×2, exactly NE×N×2; NE = 512 ≤ 512;
N = 2560 % 32 == 0. The kernel-side gates fit too (k = 10 ≤ 32; 8×10 ≤ kDmCap 128).

The real cause: on ANY multi-GPU split `stage_trim = multi_gpu` — each stage's dense table holds only
its stage's layers; the foreign layers keep their index row WITHOUT bytes (`data == nullptr`).
`router_codes` walks all 48 layers from the stage's table, hits the first foreign router, and its
`cudaMemcpy` from the null pointer fails → `codes_ok = false` → warm-up off on BOTH stages of every
split boot. The warning's parenthetical ("a router the predictor does not cover") misdiagnosed it: the
router IS covered; it is just not resident in the stage's table. (Single-card boots were never affected:
one table holds every layer.)

Why zero rows are safe for the layers a stage does not hold: no peer of the stage holds an expert of a
foreign layer, so that layer's `ptab` entries are 0 and `dm_warm` maps its prediction to no blob — a
zero-row prediction (all logits 0 → the lowest expert ids) can never fetch a wrong weight. The CPU
pool's warm-up (CpuWarm) fires only for `l + 1 < le_`, the next layer of this stage, so it never
consumes a foreign layer's row either.

### The changes

1. `d5ffca34` — `router_codes` skips a layer the table does not hold (`data == nullptr`; its rows stay
   zero) and fails only on a real coverage miss: the router absent from the table, a held router with
   the wrong shape/bytes, or an unreadable router. The warning now names the failure: the specific
   layer with expected vs actual bytes, the shape gate, or "the predictor's buffers could not be
   placed on the peer".
2. `974a81bb` — `--exp l2warm_force=1`: takes the warm-up on despite an uncovered router, gated to the
   geometry the predictor fits (NE ≤ 512, N % 32 == 0). A layer without a covered router predicts from
   its zero rows (the lowest expert ids) — an A/B handle, not a default; an info line says when the
   force is what turned the warm-up on. The further peers follow the main peer's placed buffers.

### Per-change predictions (reviewer to measure)

Boot line:
- "the peer's L2 warm-up is off (a router the predictor does not cover)" DISAPPEARS — it printed once
  per stage verifier per split boot; expect zero warm-up lines on a default boot (the warm-up is
  quietly on). If a router is genuinely uncovered, the warning now names the layer or the shape.
- `--exp l2warm_force=1` on top of the fixed default: NO extra line and no behaviour change (coverage
  now holds, so the force is a no-op here).

Decode profile (`decode_profile=1`, run 18 addendum baseline: solo peer push 0.1-0.2 / concurrent 1.7;
wait 0.0):
- **peer push**: the predictor's router pass and the predicted-blob touch-loads run INSIDE the peer's
  persistent window, so the field moves up a little: solo ~0.1-0.2 → ~0.2-0.4; concurrent 1.7 →
  ~1.8-2.2. If the predicted blobs hit L2, the peer's expert streaming shortens by the predicted
  share's cost, so the net is roughly flat to slightly up.
- **wait** (the benefit's field): 0.0 → 0.0 in the solo regime (the peer already kept up); in the
  concurrent regime, where the peer's expert streaming is the straggler, the warm-up's win shows as
  the concurrent peer push falling toward the solo value. The dev A/B at 4f3bdef7 (NVFP4, both GPUs,
  400 W) measured -0.13 ms/window prose, -0.3 ms conv.
- **hit%**: unchanged (94.6% canary). The warm-up touches L2, not the expert cache; nothing here
  moves the residency tables.
- Numerics: the warm-up writes nothing the rings read (touch-loads only) — bitwise neutral; the
  greedy-identity check is the regression net.

### Reviewer checklist

1. Fresh 4-GPU tp2/pp2 boot: grep the boot log for "L2 warm-up" — expect ZERO lines (before: one
   warning per stage verifier).
2. Same traffic as the run 18 addendum (solo + concurrent, `decode_profile=1`): compare `peer push`
   and `wait` against the baseline above.
3. Greedy identity: same prompt, default flags, this binary vs the pre-change branch head
   (`1f7b77fc`) — identical tokens.
4. `--exp l2warm_force=1` boot: expect no change vs the default (no extra line).
5. Build green `-j4`; `ctest -L cpu` 67/68 (`expert_multi_test` pre-existing);
   `tools/check_platform_boundary.sh` + `tools/check_stderr_logging.sh` ok.

## RUN 22 — the share priced in TIME (share-v2), the head cut parameterized into spans, and the attention slice priced NET

Scope, as approved: **mechanism and model only (option A)**. The peer-side prompt attention plus a second
`attn_output` all-reduce (**option B**) is NOT in this run; it is the opening milestone of the TP=4 prototype work,
and everything below that concerns it is labelled as a model prediction, not a capability. Nothing was measured on
a pair here: the box has a live server on `:8000` and the engine was never started. Every figure below is analytic,
per stage and per card, from the shipped functions transcribed into a scratch program (the run-16/21 method, not
committed). Full build green `-j4`; `ctest -L cpu` 67/68 (`expert_multi_test` pre-existing); platform-boundary and
stderr-logging checks pass; the four named CPU shard tests pass with their added cases, and the two GPU parity
tests compile (the all-reduce one's pure parts were run with no card visible). No test files were added, so the cpu
count stays 68.

### 1. The head split is now a SPAN, not a half

`QsaHeadShard` (`include/basalt/core/qsa_head_split.hpp`) gained `q_beg`/`nq_`: a card states which QUERY heads it
runs and its KV run is DERIVED from that run (`kv0() = q0/rep`, `nkv()` = the groups the run touches), so the group
rule still holds and a cut INSIDE a KV group leaves the boundary group's KV on BOTH cards. `parts`-only shards -
what every caller builds today - carry `nq_ == 0` and take the symmetric run, byte-identically: `qsa_pair_shard(n,
kv, n/2, rank)` compares `==` to `QsaHeadShard{n, kv, rank, 2}`, which is what keeps the default path and the
sizing tests untouched. New: `qsa_pair_shard` (one card of a pair from the lead's head count), `qsa_lead_heads`
(a ratio to heads, at least one per card), `qsa_pair_tiles` (the pair covers the axis exactly once) and
`qsa_shard_whole_groups`.

**The assumption that hard-coded equal halves, and what was done about it.** The three QSA attention kernels -
`qsa_dec_attn.cu`, `qsa_decode_attn.cu`, `qsa_prompt_attn.cu` - each fix the query heads per KV head at COMPILE
TIME (`constexpr int G = 12`) and refuse any shape where `n_head != G * n_head_kv`. A shard's run must therefore be
WHOLE KV GROUPS: at this artifact's geometry (24 heads, 2 KV heads) a stage's heads move **12 at a time**, and
`qsa_shard_runs` (`src/core/layer.cpp`) now refuses a straddling span by name instead of letting a kernel `exit(1)`
mid-run. Generalizing `G` to a runtime head count and a head map is the same substrate as option B and is NOT done
here. Consequences of the group rule, tested: an uneven-but-whole-group cut (24 heads over 8 KV heads, 3 groups to
the lead and 5 to the partner) is runnable and splits the KV bytes 3/5; a straddling cut (16/8 at 24 over 2) holds
3 KV heads of pool for a 2-head cache, and the carve prices it (`session_head_shard_test`).

The ratio reaches a SHARD at one place, the decode side: `Verifier::qhs_init` builds the lead's shard from
`--exp qsa_ratio_stage_a` (the stage that holds layer 0) or `--exp qsa_ratio_stage_b`, default 0.5 - the symmetric
half RUN 17 armed. A ratio that cuts a group is refused at boot, naming the granularity. Prefill reads the same
flag but builds no shard with it - it runs no split - and carries it only into the boot line below, which is why
that line says the heads are *priced*, not run.

### 2. share-v2: the band's share is struck on the two cards' LAYER TIMES

`peer_ram_share` (`src/prefill/prefill.cpp`) keeps its counts - `held`, `band`, the blob `B`, the rows `w` - and now
gets an objective. `peer_layer_time` prices one stage's layer on each card at band share `r`:

    lead = attn_lead + (chunk*K - rows_peer) * lead_row_us/1000 + (band - r) * B / blob_link
    peer = attn_peer +  rows_peer * peer_row_us/1000 + rows_peer * row_bytes / pair_link + acts / pair_link
    rows_peer = (held + r) * chunk * K / n_expert

and `r` is the closed form where the two meet, since each moved expert moves a constant `d_lead` off one card and
`d_peer` onto the other (then clamped as before: to the band, and to the half-the-chunk's-rows cap). The default
objective is this TIME one; `--exp qsa_share_bytes=1` restores run 16's byte-lane balance verbatim, which is both
the fallback and the A/B the time objective is graded against. The rates, in MB (1e6 B) a millisecond and
microseconds a row-visit, and their `--exp` names:

| constant | default | what it prices | re-price it |
|---|---|---|---|
| `lead_row_us` | 2.26 | one row-visit of one expert on the lead | `--exp qsa_lead_row_us` |
| `peer_row_us` | 1.09 | ... and on the peer card | `--exp qsa_peer_row_us` |
| `link_mb_ms` | 4.2 | the pair's link, both legs (the planes out, the rows back) | `--exp qsa_link_mb_ms` |
| `blob_mb_ms` | 4.2 | the lead's own read of a RAM blob | `--exp qsa_blob_mb_ms` |
| `attn_lead_ms` | 54.4 / 49.0 | the stage's whole attention, run by the lead alone | `--exp qsa_attn_ms_stage_a\|_b` |
| `attn_peer_ms` | **0.0** | what the peer runs of it | `--exp qsa_attn_peer_ms_stage_a\|_b` |

`attn_peer_ms` is 0.0 because **prompt attention has no peer caller**: prefill calls `qsa_prompt_attn_batch` once
with the whole `n_head`, then one whole `attn_output` projection; the `QsaPair`/`qsa_pair_prepare`/`tp2_all_reduce`
machinery has no prefill call site. So the ratio knob in prefill changes NO work - the new boot line prints the
heads it priced and says so in words, and the peer's attention term is printed as 0.0 rather than left to be
inferred. `blob_mb_ms` is priced at the pair's link rate: the lead's RAM read has not been measured on this pair.
The four row/link terms are the reviewer's measured constants from the 400 W boots (run 18's `rows 114.6 -> 56.3`
BF16 attribution and run 21's planes at 1 448 B a token over ~2.8 ms for 8 192 tokens = ~4.2 MB/ms, which the rows
leg agrees with to a few percent); they are shipped as overridable constants, not as measurements of this build.

The new line, printed with the band on (`grep "peer layer time"`), averages the terms over the stage's layers:

    peer layer time layers 0-24 at 2/8 band: lead 132.2 ms a layer = attention 54.4 + rows 77.8 + blobs 0.0, peer
    109.4 ms = attention 0.0 + rows 51.4 + back 55.2 + acts 2.8; the critical side is the lead by 22.8 ms (2.26/1.09
    us a row-visit, 4.2 MB/ms of link); heads 12/12 priced (--exp qsa_ratio_stage_a), the peer runs no prompt
    attention so its heads are priced at 0.0

(`rows` is the card's own expert-row compute, `back` the peer's rows crossing the pair's link, `acts` the chunk's
activations arriving.)

### 3. What the model picks, at run 14's geometry (MODEL PREDICTIONS; nothing here was measured)

24 layers a stage, 512 experts a layer of 2 548 736 B, chunk 8 192, BF16 rows, FP4 planes. `r` is the band experts
moved a layer; `cap` is `n_expert/2 - held`, the half-the-chunk's-rows bound.

| stage | held / band a layer | byte objective: num, r | time objective: num, r | lead ms = attn+rows+blobs | peer ms = rows+back+acts | critical side |
|---|---|---|---|---|---|---|
| A | 128 / 112 | 4/8, r=50 | **8/8, r=112** | 152.8 = 54.4+98.4+0.0 | 91.5 = 41.9+46.8+2.8 | lead by 61.3 |
| A | 128 / 256 | 4/8, r=128 (cap) | 4/8, r=128 (cap) | 224.6 = 54.4+92.6+77.7 | 97.4 = 44.6+49.9+2.8 | lead by 127.2 |
| A | 200 / 112 | 2/8, r=32 | **4/8, r=56** | 181.0 = 54.4+92.6+34.0 | 97.4 = 44.6+49.9+2.8 | lead by 83.5 |
| A | 200 / 160 | 3/8, r=56 (cap) | 3/8, r=56 (cap) | 210.1 = 54.4+92.6+63.1 | 97.4 = 44.6+49.9+2.8 | lead by 112.7 |
| A | 240 / 64 | **0/8** | **2/8, r=16** | 176.1 = 54.4+92.6+29.1 | 97.4 = 44.6+49.9+2.8 | lead by 78.7 |
| A | 248 / 8 | **0/8** | **8/8, r=8** | 147.0 = 54.4+92.6+0.0 | 97.4 = 44.6+49.9+2.8 | lead by 49.6 |
| B | 200 / 112 | 2/8, r=32 | **4/8, r=56** | 175.6 = 49.0+92.6+34.0 | 97.4 = 44.6+49.9+2.8 | lead by 78.1 |
| B | 240 / 64 | **0/8** | **2/8, r=16** | 170.7 = 49.0+92.6+29.1 | 97.4 = 44.6+49.9+2.8 | lead by 73.3 |

Read this as the model's OWN balance, not as a measurement: the row terms are per-row-visit rates held CONSTANT,
but a row-visit's cost rises with the row count a card holds (a card's GEMMs amortize better when busy), so the
model moves MORE band than a measurement will justify - every row where the two objectives differ is the model
over-reaching, and the reviewer's `peer chunk` lines are the arbiter. Two structural facts survive that bias: the
lead's blob term is what makes the time objective move band where the byte model moved none (0/8 -> 2/8, 8/8), and
**the pair is never balanced by band alone** - the lead keeps the whole attention, so `critical side` reads `lead`
at every placement and the residual is tens of ms a layer.

### 4. The attention slice, priced NET (review #3 addendum, P1)

**The slice's gain is unproven and could be negative.** Splitting the whole prompt attention moves compute off the
lead, and it costs the transport of the `attn_output` partials, which is `T x n_embd` per layer per card - at
chunk 8 192 and `n_embd` 2 560 that is **83.9 MB of FP32 one way a layer** (41.9 MB if the partial crossed as BF16
bits the way the peer's rows do). The figures, per layer-pair at 8 192:

| transport variant | assumed | ms a layer | net vs the 27.2 ms of compute |
|---|---|---|---|
| bulk 2-card all-reduce, FP32 partials, at the measured 4.2 MB/ms | both legs serialize | ~40 | **-12.8** |
| bulk 2-card all-reduce, BF16 partials, at the measured 4.2 MB/ms | both legs serialize | ~20 | **+7.2** |
| the addendum's own figures (100 MB at ~7 GB/s d2d) | bulk AR | 14 one way, 28 round | **+13.2 / -0.8** |
| per-head-group streaming BEHIND the peer's expert window, BF16 | the bytes hide under the peer's 51.4 ms of comp | ~0 hidden | **+27.2** |

So the slice's net range is **[-13, +27] ms a layer**, and which end it lands at is decided by ONE question: can
the partial for a head group cross while the peer computes its experts, or does it serialize into an all-reduce
barrier? The streaming variant is the only one that pays, and it is exactly the wiring that does not exist (option
B). Until it is measured, the head-ratio figures in section 3 and the `heads N/M priced` clause of the boot line are
**MODEL PREDICTIONS and the slice is not called a win**: the number that matters is the measured net critical-path
ms WITH transport, per layer-pair, at 8 192.

**Kill criterion for the slice, at the TP=4 milestone: measured net < +10 ms a layer-pair at chunk 8 192 -> STOP
and re-plan.** Below that, EP+PP keeps the whole attention on the lead and the slice serves tensor parallelism
only.

### 5. What the reviewer should see, and what falsifies this run

1. **`grep "peer layer time"`** on both stages with `--peer-ram-rows`: the terms must read against the chunk lines
   (lead's `attention` against the `attn` field, its `rows` against `lead`; peer's `rows` against `comp`+`rows`, its
   `acts` against `act`). A term more than ~20% off means its constant is wrong for this box: re-price it with the
   matching `--exp` and re-boot before reading the share it chose.
2. **`attn 0.0` on the peer** with no flag: the peer runs no prompt attention. If it ever reads non-zero without
   option B's wiring being merged, that is a bug, not a feature.
3. **share-v2 moves `num/8` only where the two objectives differ** (section 3's bold rows). With
   `--exp qsa_share_bytes=1` on the same boot the line must read exactly run 21's figures - that is the
   byte-identity control for the share's choice.
4. **Row-rate falsifier:** after the balance, if the lead's row rate still runs **>= 1.4x** the peer's
   (`lead` ms / lead rows against `comp` ms / peer rows, from the chunk lines), the lead's row kernel is
   contending with the link and copy-stream load: record it and STOP balancing further. The shipped constants
   already sit at 2.26/1.09 = 2.1x, so this is expected to fire - which is the honest statement that the model does
   not yet understand why the lead's rows cost twice the peer's for the same arithmetic.
5. **Kill criteria for the whole run** (after one iteration): stop if the critical-side pair wall exceeds ~135 ms a
   layer, or if `8192`-chunk prefill falls below 2 200 t/s.
6. **Decode floors** (untouched by this run - the split stays behind `--exp decode_headsplit`, which still refuses
   while the peer's item is unbuilt): solo >= 78 t/s, accuracy within 3 points of the branch's, hit >= 89.

### Quality gate (this run CHANGES which rows cross the wire, so the gate applies)

share-v2 changes which band experts the peer computes, so a different set of result rows arrives as BF16 bits and
the combine re-sums them in a different order: **the PPL/KLD gate against the Q8_0 reference, through the decode
path (`basalt-bench --quality`), must pass** - byte identity with run 21 is NOT expected with the time objective
on. The controls: `--exp qsa_share_bytes=1` reproduces run 21's choice exactly (same experts, so same bits), and
`--peer-rows-fp32` removes the BF16 rounding entirely. The head-span work adds no numeric path: no caller builds an
uneven shard unless a ratio flag is given, and a straddling ratio is refused at boot.

### Not done, on purpose

- Option B (the peer's prompt attention, its `attn_output` partial and the second all-reduce) - the TP=4 opening
  milestone. The slice's net in section 4 cannot be measured without it.
- `G` in the three attention kernels stays compile-time, so unequal cuts are expressible, priced, tested and
  refused at run time on this geometry. Making it a runtime head map is part of the same milestone as option B.
- `blob_mb_ms` is the pair's link rate until the lead's RAM read is measured; the boot line prints it, so the
  reviewer can see what the model believed.

### Reviewer checklist

- [ ] Boot both stages with `--peer-ram-rows --prefill 8192 --tp-size 2`; `grep "peer layer time"` reads the two
      cards' terms and names the critical side; the peer's attention term is `0.0`.
- [ ] Same boot twice: `--exp qsa_share_bytes=1` (run 21's objective) against the default - the `num/8` difference
      is section 3's table, and the byte-model boot must match run 21's line.
- [ ] `--exp qsa_ratio_stage_a=0.75` (or any ratio that cuts a group): the decode flag `--exp decode_headsplit=1`
      must refuse naming the KV-group granularity; the prefill line must print `heads 18/6 priced` and NOTHING ELSE
      may change (compare two chunk lines with and without it).
- [ ] `--exp qsa_ratio_stage_a=0.5`: the armed decode line reads `12 of 24 query heads (ratio 0.5 ...)`, as RUN 17
      measured it.
- [ ] Chunk lines: item 1's term-by-term read, and item 4's row-rate ratio.
- [ ] Quality gate: PPL/KLD against the Q8_0 reference with the time objective ON (the required pass), plus the
      `qsa_share_bytes=1` control.
- [ ] `prefill_fp4_test` on the two devices: unchanged (the peer's row/act wires did not move).
- [ ] Decode: the floors above, flag off.

### Validated here (GPU untouched: the server is live, `:8000` never touched, no engine started)

Full build green `-j4`, and the warning set is unchanged: the two `prefill.cpp` warnings it prints are the two HEAD
prints (checked by building HEAD's `prefill.cpp` beside this one), and `session_head_shard_test`'s sign-compare
warning is the pre-existing one, at its own untouched line. `ctest -L cpu` 67/68 - only the pre-existing
`expert_multi_test` (this CPU lacks AVX512-VNNI/VBMI); the count is unchanged at 68 because the new cases went into
the existing test files. `qsa_head_split_test`, `session_head_shard_test`, `tp2_shard_weights_test`,
`qsa_decode_combine_test` pass with the added uneven-span cases; `tp2_head_split_parity` and `tp2_allreduce_parity`
compile, and the latter's pure parts (including the new unequal-partials case) pass under
`CUDA_VISIBLE_DEVICES=""`. `tools/check_platform_boundary.sh` and `tools/check_stderr_logging.sh` ok. The
section-3 table is the shipped `peer_ram_share`/`peer_layer_time` transcribed into a scratch program (not
committed), same method as runs 16 and 21.

## RUN 23 — the suffix drafter's long-generation decay: the index already had the tail; the gate was throwing it away (kimi review #2, P2-4)

### The measured problem

Real traffic: a 70k-token single generation ran ~40 t/s while short turns ran 51-78 t/s on the same
binary. The decode profile showed healthy windows (27-29 ms, 3 rows) but the long generation kept only
~1.1 tokens/window vs ~3.0 on short turns; short-turn draft acceptance was 68-78%, the long generation's
windows implied ~35-40% effective. P2-4 parked this as the highest-EV decode item: ~1.9x on a real
workload class, independent of the topology work.

### Root cause — hypothesis (a) is refuted; the gate was the problem

The brief's leading hypothesis was that the drafter's index never sees generated tokens. It does: all
three decode paths append every committed token (`generate.cpp` solo loop `sfx.append(outv[i])` after
each commit; batch slots `sl.sfx->append(outb[...])` per kept token; CLI loop likewise), and the index
grows from the prompt through the whole generation. Replaying the real `SuffixDrafter` on 70k-token
synthetic streams (70k class: prompt 4k + 66k generation, code/JSON/reasoning blocks, K=7) confirms the
tail is found — on verbatim structure the drafter gets 5.3 tok/step at 61% acceptance, and a late
re-quote of earlier code runs 4-8 tok/step. Self-indexing works.

What the replays actually reproduce is the gate. Two regimes:

| stream (70k) | min_match | propose% | acc/proposal | tok/step |
|---|---|---|---|---|
| noisy parameterized repetition (30% unique content) | 8 | 5% | 1.7/7 (25%) | **1.09** |
| same, reseeded | 8 | 41% | 2.5/7 (35%) | **2.03** |
| same | 3 | 41% | 2.5/7 (35%) | 2.05 |
| clean verbatim code | 8 | 6-10% | 1.8/7 | 1.1-1.2 |
| same, reseeded | 8 | 38-64% | 2.5/7 | 2.5-2.7 |
| same | 3 | 61% | 2.9/7 | 2.76 |
| long-range re-quote | 8 | 1.6-100% | 2.9-7.0/7 | 1.1-4.1 |
| same, reseeded | 8 | 19-100% | 2.9-7.0/7 | 1.9-5.3 |

The measured launch ran `--suffix-draft 8` (min_match 8). On a non-verbatim tail, 8+-token matches are
~5% of windows and even those accept at ~25%: the drafter is effectively dormant, the policy gives up
on lookup, and the weak MTP windows commit ~1.1 tokens/window — the measured number, the 3-row profile,
the 27-29 ms windows. Even at min_match 3 the drafter's ceiling on noisy tails is ~35-40% acceptance
(n-grams cannot predict genuinely new content), so the policy mostly bypasses it; that ceiling is a
property of the tail's content, not of the index.

### The change — `--exp draft_reseed=1` (default off, byte-identical)

The missing piece is the *reliability rule*, not the index: a repeat of prompt material needs a long
match to be trustworthy, but a repeat the model just made in its own output is trustworthy at trigram
length — the current phase's structure is the evidence. So in reseed mode a match of 3-7 tokens
qualifies when its source lies in the generated tail (`mark_history` splits prompt from tail at request
start); prompt-sourced short matches still need `min_match`.

- `include/basalt/spec/suffix_drafter.hpp` / `src/spec/suffix_drafter.cpp`: `set_reseed(bool)`,
  `mark_history()`; `propose` admits a tail-sourced candidate at `kReseedMin = 3`. `reset()` clears the
  mark. Off by default: with reseed off the candidate filter reduces to the old one and the output is
  byte-identical (verified by the unit tests).
- `src/program/generate.cpp`: the three drafter owners enable the mode from `exp_flag("draft_reseed")`
  right after seeding the prompt — solo serve loop (after the `ids` seed), batch slot admission (after
  the `live` seed, before the first generated token `x`), CLI loop (after `o.tokens`, before the
  resumed `produced`).
- Deliberately NOT done: capping the proposal at match length + slack (measured net-negative on every
  stream: 2.10→1.94 tok/step continue, 6.45→5.76 copy — it truncates correct verbatim continuations,
  and `DraftPolicy` already sizes the window from measured acceptance), and WAYS 4→16 (no change: the
  trigram's source occurrence of a repeated block is always local to the repeat).

### Amortization and memory bounds

- Host cost: the append is O(1) per token (one probe, a 4-position shift) and was already done at commit
  time; reseed adds one comparison per candidate in `propose` (≤4 candidates) and nothing to appends.
  Measured append cost: 0.03 µs/token over 262k tokens. Expected draft-phase delta: ≈ 0 ms/window,
  well inside the +0.5 budget.
- Memory: no new allocation in reseed mode. Per 10k generated tokens: +40 KB of history, and up to 10k
  newly filled table slots (32 B each, ≤320 KB) inside the table pre-allocated at request start as
  2×(max_context+4096) slots. Once the table's distinct-trigram capacity is reached, inserts are dropped
  (existing behavior); matches for indexed trigrams keep working.

### Predictions (graded at the reviewer's boot)

1. Long-generation replay (70k class) with the boot's `--suffix-draft 8` + `--exp draft_reseed=1`:
   dl 1.1 → 1.9-2.5 tok/window (replays: 1.09→2.03 noisy, 1.1→2.7 clean, 1.1-4.1→1.9-5.3 re-quote);
   the drafter's propose rate rises 5% → 30-45% of windows and the policy keeps taking short lookup
   windows. Meets the dl ≥ 2.2 target on any tail that is more than half verbatim; on a fully noisy
   tail it lands ~2.0 — near, not above, the target.
2. Short-turn acc: unchanged within noise. Prompt-sourced matches there are ≥8 tokens (the floor never
   fires); the only new proposals are 3-7-token repeats the turn itself just made, which the policy
   prices from its first few windows (decay 0.97) and drops if they underperform.
3. Draft-phase host ms: within +0.5 of baseline (expected ≈ 0; the index work was already amortized).
4. Index memory: bounded as above — +40 KB history per 10k generated tokens, no table growth.
5. KILL CRITERION (P2-4): if the reseeded acceptance recovers < 55% of fresh-session acceptance
   (< ~37-43%), the tail distribution resists re-seeding — report and stop, no iteration.
6. Known null case: if the measured boot actually ran the DEFAULT `--suffix-draft 3`, the floor is
   already active and reseed is near-inert by design — expect dl ~1.1-1.3, i.e. no change, which is the
   tail-content ceiling (new content is not n-gram-predictable) and trips the kill criterion. The
   reviewer should read the boot's `--suffix-draft` value before grading.

### Reviewer checklist

1. Build green `-j4`; `ctest -L cpu` 67-68/68 unchanged (`expert_multi_test` pre-existing);
   `tools/check_platform_boundary.sh` + `tools/check_stderr_logging.sh` ok.
2. Long-generation replay (the P2-4 workload): boot the A/B binary (this branch) and the baseline
   (branch head `293ca647`) with the same launch flags; replay the same 70k-class generation (same
   prompt, greedy, same `--suffix-draft` as the measured boot) on each; `--draft-log` on both.
   Compare per-window lines: the `W <T> <a> <from_sfx>` mix (expect `from_sfx=1` windows to appear in
   the long generation under reseed where the baseline ran 3-row MTP windows) and the committed
   tokens/window curve over progress. The DONE line's draft counters (`drafts accepted/offered`) and
   the `suffix drafts ... windows` info line give the request-level acc.
3. Short-turn A/B: the edit/refactor turns that measured 68-78% acc — acc within noise, t/s unchanged.
4. Default-path identity: no `--exp draft_reseed` → `suffix_drafter_test` unit tests pass and a greedy
   run is token-identical to the baseline binary (the mode is a no-op off).
5. Kill criterion check: if reseeded acc < 55% of fresh-session acc, close P2-4 as "tail distribution
   resists re-seeding" — do not iterate.

## RUN 24 — the link's real shape: one ~7.8 GiB/s pool for four cards, and the AR's floor (step 0, no engine change)

**Topology correction (owner, post-run):** the four cards share ONE x16 uplink, bifurcated
x4 per card. The ~7.8 GiB/s pool is the d2d (peer round-trip through the root) face of that
link; the h2d face (RAM->card tier streaming) can use the whole x16 concurrently
(~24-30 GB/s observed; the split's 18.4 GiB/s aggregate is this face). The two faces
CONTEND: tier fill and the AR compete for the same silicon, which is Finding 2's queueing
seen from the other side. Utilization arithmetic must use the right face per flow: streaming
wall = bytes / (h2d face x util), AR wall = bytes / (~8.6 GiB/s d2d face), not one pool.

Scope note, CORRECTED (owner, 04:40 UTC): an earlier mid-run scope line said "TP=4 is dropped" — that was the
supervisor's over-reading of the owner's "we don't need the bench anymore" (which cancelled only the pre-substrate
overlap-cadence bench). **TP=4 remains on the table.** The amended sequence: TP=2/PP=2 first (its milestone is in
flight; the slice, the AR transport, the stagger scheduler and the sharding precision gate are shared substrate),
then TP=4's go/no-go re-priced from TP=2's MEASURED coefficients. TP=4's own outstanding measurements: the 640-wide
pack inspection (cheap; 1280 proved a pure-offset view, so 640 = four contiguous runs is plausible but unverified),
the quarter-width MMQ sweep (the 1.2-1.4x efficiency penalty, unmeasured at 640), and the 4-deep rendezvous cadence
(the cancelled bench — revisit only if TP=2's measured cadence data does not transfer). Original scope line kept
below for the audit trail.

Scope (reviewer, mid-run): ~~TP=4 is dropped~~ (see correction above). The target is PP=2 + true TP=2 — 2 stages × 24 layers, every
expert sharded 2-way, attention head-split. Quarter-width / 640-sliceability analysis and the 4-deep
overlap cadence are out. What step 0 had to settle before any substrate work: the small-payload AR latency
floor, whether a 1280-wide N-slice of gate/up is a pure-offset view, and the per-path byte ledger with each
path's currency kept separate.

### The one fact that governs everything: the four cards share a single ~7.8 GiB/s pool

Measured with `tools/tp_ring_probe.cu` (standalone, `nvcc` directly, no CMake, not linked into the engine;
buffers freed between every measurement point; refuses to run if any card is under 256 MiB free — the live
server owns these cards and was never touched). `cudaMemcpyPeerAsync`, the same call `Tp2Pair::all_reduce`
makes. 4 MiB per copy, 20 rounds, RTX 5060 Ti ×4 at a **180 W power limit** (SM clocks 2692–2947 MHz as
read), all-GPU `PHB` topology (nvidia-smi `topo -m`), no NVLink. Every number in this RUN is at that limit.

| pattern | links busy | per link | fabric total |
|---|---|---|---|
| one pair alone (any of the 12 ordered pairs) | 1 | 6.16–6.64 GiB/s | 6.16–6.64 |
| one pair, both directions | 2 | 3.96 GiB/s | **7.92** |
| two pair links, both directions (PP=2's two AR pairs) | 4 | 2.00 GiB/s | **8.00** |
| 4-deep ring `0>1>2>3>0` (and reversed) | 4 | 1.93 GiB/s | 7.72 |
| all-to-all, 12 directed pushes | 12 | 0.64 GiB/s | 7.68 |

Every pattern lands on the same ~7.7–8.0 GiB/s total. The links are not independent: one pair at full
duplex already saturates the fabric, and a second pair gets a quarter of the solo rate, not half. **Per-pair
bandwidth is not a per-card allowance; it is a share of one host-bridge pool.** The 12 ordered pairs are
uniform to within 8%, so no pair is privileged — the `PIX`/`PHB` distinction that mattered on the older
boards does not buy anything here.

### The AR's floor: 25 µs idle, 2.4 ms with the tier running — and the copy engine is why

`--mode ar` / `--mode artier`: the engine's own three issues per rank (push, flag, wait/add), single-shot
rounds with min/median/max, plus a queued back-to-back figure. Payloads are what a split o_proj / down_proj
partial actually is. **Idle fabric:**

| payload | one pair, copy engine | one pair, SM push | two pairs, copy engine |
|---|---|---|---|
| 4 KiB | 22.7 / **24.8** / 62.8 µs | 9.1 / **19.4** / 372.8 µs | 38.9 / 320 / 6562 µs |
| 16 KiB | 22.9 / **24.8** / 2250 µs | 8.9 / **21.7** / 1257 µs | 42.8 / 1744 / 6564 µs |
| 64 KiB | 28.6 / **33.1** / 2245 µs | 28.3 / **29.8** / 75.6 µs | 46.0 / 368 / 6581 µs |
| 256 KiB | 72.5 / **76.4** / 109 µs | 65.4 / **68.7** / 298 µs | 135 / 1862 / 6626 µs |

(median is the decision column; the max column is the live server taking the card back, not the fabric).

**Under a concurrent tier fill** — 4 expert blobs (2.63 MiB each, the pack's real `NativeFmt` size) pushed
across the same link per round on a second stream, the depth the ring runs at:

| payload | copy-engine push | SM push |
|---|---|---|
| 4 KiB | 1888 / **2446** µs | 159 / **290** µs |
| 16 KiB | 1717 / **2407** µs | 131 / **325** µs |
| 64 KiB | 852 / **2376** µs | 140 / **375** µs |
| 256 KiB | 2172 / **2444** µs | 245 / **347** µs |

That is the run's result. A `cudaMemcpyPeerAsync` push **waits its turn behind the tier's copies on the
copy engine**: a 25 µs round trip becomes 2.4 ms, 100×, and the AR payload size stops mattering at all —
which is the signature of queueing, not of bandwidth. An SM-driven push (the IPC-read shape) does not queue
there and costs 8× less under the same load, and is also cheaper on an idle fabric.

### What that prices: decode misses the budget on the copy engine and just makes it on SMs

The budget the reviewer set: 96 AR rounds/window (2 per layer × 48 layers) under ~2 ms.

- copy engine, idle, 4 KiB median: 96 × 24.8 µs = **2.4 ms — over budget** before a single byte of work.
- SM push, idle, 4 KiB median: 96 × 19.4 µs = **1.9 ms — inside, with no margin.**
- copy engine with the tier live: 96 × 2446 µs = **235 ms.** Dead.
- SM push with the tier live: 96 × 290 µs = **28 ms.** Still dead at decode rates.

So the honest reading: the AR's *latency floor* passes decode only in the idealized case — SM-driven
pushes, an idle fabric, and perfect pipelining so the wait hides behind compute. The tier and the ARs are
**not** overlappable on the copy engine as the primitive stands today. Any TP=2 that streams expert blobs
and ARs concurrently must put the AR on SMs *and* keep the blob stream off the same direction on the same
link, or it pays the queue, which is two orders of magnitude, not a tuning constant.

### The volume ledger, per path and per currency (H pinned from geometry, not guessed)

`H = n_embd = 2560` from `ModelGeometry` (`include/basalt/core/layout.hpp`), 48 layers, 512 experts,
`n_ff = 640`, 24 heads × 256. The AR's per-rank bytes are `tp2_all_reduce_bytes(rows, stride) =
rows * stride * 4` (`include/basalt/kernels/tp2_collective.hpp`) — FP32 residual, so 10 KiB per token per
AR. The QSA width 6144 is the *attention input*, not the AR payload; using it as `H` overstates the AR 2.4×.

Current topology, from the boot lines (`503 MiB / 217 MiB` stage A, `419 / 213` stage B, per layer), booked
as two currencies because the paths are different hardware:

| path | bytes/layer | rate | ms/layer |
|---|---|---|---|
| lead's host link, tier fill (`h2d`) | 503 MiB (A) / 419 (B) | 6.71 GiB/s = the 7.2 GB/s the PCIe probe prints | 75.0 / 62.5 |
| pair link, planes + rows (`d2d`) | 217 MiB (A) / 213 (B) | 6.4 solo, **2.00 when both pairs run** | 33.9 / 108.5 |

The `d2d` column is why the current design's pair leg is only survivable: at the *solo* 6.4 GiB/s the pair's
217 MiB is 34 ms. It only reaches 108 ms when both stages push at once, which the ledger above shows is what
TP=2 makes permanent.

TP=2/PP=2's added leg — 2 ARs per layer per rank at `2 × rows × 2560 × 4`:

| chunk | AR bytes/layer | @2.00 (both AR pairs live) | @3.96 (staggered) | planes leg |
|---|---|---|---|---|
| 2048 | 40.0 MiB | 20.0 ms (13.0% of A's 153.8 ms) | 10.1 ms (6.6%) | 2.8 MiB, 1.4 ms |
| 4096 | 80.0 MiB | 40.0 ms (26.0%) | 20.2 ms (13.1%) | 5.6 MiB, 2.8 ms |
| 8192 | 160.0 MiB | 80.0 ms (52.0%) | 40.4 ms (26.3%) | 11.3 MiB, 5.6 ms |

Against what TP=2 saves at chunk 8192: half the expert work and half the attention
(`72.4/2 + 13.9 peer-wait + 57.2/2 = 78.7 ms`).

- Both stages doing their ARs on the fabric at once (the 2.00 GiB/s column): **−6.9 ms per layer. A loss.**
- Staggered so only one pair is ever pushing (the 3.96 column): **+35.4 ms per layer. A win, worth ~23%.**

The entire verdict is the scheduling of the two stages' ARs against each other, not the collective's
implementation. At 26–52% of the layer time the ARs are also no longer background: RUN 20's requirement to
hide ≥90% of the pair traffic stops being satisfiable by putting the copy on a second stream, because the
second stream now competes with the tier fill for the same pool.

### 1280-wide N-slice of gate/up: a pure-offset view, not a repack

`NativeFmt` (`src/kernels/cpu/native_expert.cpp::native_fmt`) lays an expert out as
`[gate rows | up rows | down rows]` + a 16 B NVFP4 scale tail: `gu_row = 1440` B (`n_embd=2560` at
`(2560/64)*36`), `up_off = 921600`, `down_off = 1843200`, blob `2764816` B — the boot line's "largest blob
2.76 MB". A 1280-column N-slice is `640` rows × `720` B of each gate row plus the same offset into the up
region: **two contiguous 460800 B runs at fixed offsets**, one in gate, one in up. Contiguous, aligned, and
self-contained: it needs no dequant-slice and no load-time repack. Two constraints, both already satisfied:
NVFP4 blocks are 64 values / 36 B so the slice boundary must be a multiple of 64 (1280 = 20 × 64 ✓), and
the per-16-value UE4M3 sub-block scales live inside the same 36 B block, so slicing at 64 never separates a
value from its scale. `moe_fp4.cu` already takes `w_off` + `w_cols` + `half` per product, so the kernel
signature is shaped for it. The residual cost is not layout but reduction order: a sharded down_proj sums
two half-width K-runs instead of one full run, which is a precision-path change under the `docs/precision/`
rule and needs the PPL/KLD gate, not just greedy identity.

### Verdict

- **TP=2/PP=2 prefill: real, but conditional.** Worth ~+35 ms/layer (~23%) *only* if the two stages' ARs are
  staggered so one pair-link direction is ever hot. Unstaggered it is a small loss. This is a scheduler
  requirement on the substrate, and it is checkable before any of it is built.
- **TP=2 decode: not yet a win.** The floor (96 × ~19–25 µs = 1.9–2.4 ms) sits exactly on the ~2 ms budget,
  so it needs SM-driven pushes and has no margin. With any tier traffic on the same link the copy-engine
  path dies (28–235 ms). Decode stays on the current topology until that is solved.
- **TP=4: correctly dropped.** Under the measured pool it cannot be made to work: four cards is ~1.9 GiB/s
  each, and it pays two more AR partners for bytes the pair split already moves at 3.4× the rate.
- **Roadmap:** the next gate is not the substrate's abstraction. It is (1) an AR that does not queue behind
  the tier's copies on the copy engine, and (2) stage-scheduling that keeps only one pair direction hot.
  Both are measurable here before code, and both were the assumptions the ceiling rested on.

### Reproduce

```
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 -o /tmp/tp_ring_probe tools/tp_ring_probe.cu
/tmp/tp_ring_probe --mode all --size 4MiB --iters 20      # the pool table
/tmp/tp_ring_probe --mode ar     --iters 40 --reps 101              # idle AR, copy engine
/tmp/tp_ring_probe --mode ar     --iters 40 --reps 101 --ar-sm      # idle AR, SM push
/tmp/tp_ring_probe --mode artier --iters 30 --reps 31 --bg-blobs 4  # AR under a tier fill
/tmp/tp_ring_probe --mode artier --iters 30 --reps 31 --bg-blobs 4 --ar-sm
```

### Caveats to carry

- All figures are development A/Bs on this box at its current power limit; headline numbers stay the
  maintainer's own runs.
- **The medians are not stable across server load; only the minimum is a fabric floor.** A repeat of the idle
  AR run an hour later, with the live server busier, moved the one-pair copy-engine median at 256 KiB from
  76 µs to 2274 µs while the minimum held at 74 µs. The AR tables above are from the quieter window. Read the
  min column as what the fabric can do and the median/max as what the shared box does to it — which means the
  decode budget in this run is an *optimistic* check, and the SM-push 1.9 ms has less margin than it looks.
  A maintainer run on a quiet box is the number to quote.
- The live server held ~366–388 MiB free per card throughout. The AR max columns (up to 6.5 ms) are the
  server reclaiming a card, and the two-pair medians are inflated by the same interference plus single
  host-thread issue serialization — read the one-pair medians as the fabric, the two-pair ones as
  "contention is bad", not as a rate.
- `--bg-blobs` is an approximation of a tier fill, not the engine's real ring: the ring's copies are
  O_DIRECT-fed and interleaved with compute. The queue-behind-the-copy-engine result is robust to that; the
  exact 2.4 ms is not a production measurement.
- `--mode ring`/`all2all` measure simultaneous pushes, which is how a ring AR actually behaves.

**Review #5 verdicts (kimi, post-merge; the morning decision record):**
1. **Staggered/unstaggered pinned** (my earlier citation was inverted): the win case is a
   SCHEDULER that keeps only one pair direction hot (3.96 GiB/s) — expert sharding nets
   -35.4 ms/layer (~23%, -> ~2,540) under it; both-pairs-simultaneous halves each pair to
   2.00 GiB/s and is a small loss (+6.9). Slice-only nets +8.4 ms/layer unhidden to
   +28.6 hidden -> ~2,100-2,400. Correction also adopted: the split used ~18 GiB/s of the
   h2d face, not "all of it"; the true statement is TP=2 does not enlarge that face.
2. **Decode stays collective-free PERMANENTLY on this fabric** — the AR's best case erases
   the attention win (save ~2.7, pay 1.9-2.8) and its failure mode under the live adaptive
   tier is 28-235 ms (two orders). Per-phase transport policy is the architecture: prefill
   ARs on SM-driven pushes + staggered stage scheduling; decode keeps the zero-collective
   peer window. Not a stopgap.
3. **Dual rule for the slice**: +10 ms/layer-pair stays the component ABORT; the SHIP bar is
   comparative - measured net with transport must beat EP+PP-without-slice on the same boot
   by >= ~20 ms/layer (10 ms at 171 ms is inside boot-to-boot noise; rule-of-3 if marginal).
   The span plumbing + ratio flags are kept regardless of the slice's outcome.
4. **Expert sharding is sequenced behind the slice, not dropped**: +150-200 t/s over
   slice-only, gated on the down_proj split-K PPL/KLD procedure (a known gate, not a cost).
5. **Overnight grading correction**: the 1,997/2,000 hold is share-v2's PREDICTED null, not
   evidence against; share-v2's real grade is the two falsifier reads (peer layer time
   term-by-term vs peer chunk; the row-rate ratio >= 1.4x) + a blob_mb_ms measurement -
   1-boot exercises, top of the morning list.

**RUN 25 FINAL GRADE — the pitch fix landed (3187f880: the gate took one pitch for three
operands; only the lead's are wide — the product read rows the gate never wrote = the NaN
source), correctness RESTORED (the =2 greedy output coherent, semantically valid; the greedy
stream differs from gate-off as the FP32 reduce's reordering predicts — the bar is semantic
validity, not bitwise), and the =2 grade on clean correctness: 1,836 t/s (-8.2%). THE
STRUCTURAL FINDING IS NOW MEASURED THREE WAYS: =1 half-pipelined 1,833 / =2 corrupted 1,826 /
=2 clean 1,836 — all -8.5 to -8.9%, peer-wait 69.7 ms invariant to staging depth, event
protocol, and correctness state. THE DATA DEPENDENCY IS THE WHOLE STORY: the partner's
attention(l+1) consumes mixed_h(l+1) = the lead's MoE(l) output — no software lever removes
that serial cost. Decode tier tax confirmed: 63.6 t/s cached under the flag (vs 77-82).
THE CHARACTERIZATION IS COMPLETE: the slice under PP=2 is structurally -8.5%, unfixable by
software; its machinery folds into full sharding only; full sharding's best projection
(2,430-2,520) loses to the split's 3,750 on prefill at its own best case; the decode win
(19.4 vs 35.4 ms windows) is real and topology-bound. THE OWNER'S DECISION: workload-value —
halved decode latency + ~2,430-2,520 prefill vs doubled decode latency + 3,750 prefill, or
the morph (split for prefill-heavy phases, tp2 for latency-sensitive ones).

**KIMI REVIEW #6 (the night's capstone, on the full measured set):**
1. The structural finding CONFIRMED airtight at its measured granularity (the one escape -
   sub-layer tile scheduling - priced and rejected: it contradicts RUN 24's coarser-rendezvous
   lesson and multiplies events ~8x on a gap-bound host path). Lockstep arithmetic sound within
   a band: **quote 2,430-2,520, not 2,540**.
2. The =2 corruption: the shared-indexer hypothesis falsified (buffers are per-card, per-layer);
   the staging class exonerated ON DEVICE (tp2_staging_depth: the ungated race shape fires 8/24
   layers wrong when deliberately built, the gated design clean). KIMI'S NEW TOP SUSPECT:
   **cross-chain staging-set aliasing** - kWireSets=2 keyed on qsa_ord was sized for ONE chain's
   back-to-back QSA layers; the engine runs TWO pipelined chains (two slots), whose QSA layers
   alias the same sets - tripwire-silent (both cards read the same set coherently), depth-
   dependent, needs long prompts. P2: an off-by-one among the eight pipeline>=2 index sites
   (one audited, seven not). P3: the KV span import - mode-independent, so the DISCRIMINATOR
   is one boot: the FIXED =1 on a 15-chunk prompt (clean => =2-only readers; both corrupt =>
   import). P4: tier squeeze, lowest. And: hash INPUTS at publish time (mixed_h fence bytes,
   step counts, span-import checksum) - an output-only chain shows divergence without naming
   the stale wire set.
3. THE DECISION, AMENDED: the -8.7% does NOT kill attention-splitting (its mechanism is
   STAGE-COUPLING, which full sharding removes by construction). But the split's measured
   3,750-3,868 caps sharding's prefill claim: even undiscounted, ~2,540 sits 33% below - full
   sharding justified on prefill throughput LOSES at its own best projection. The defensible
   case is DECODE: tp2's measured 19.4 ms windows vs the split's 35.4 at comparable t/s. THE
   PRODUCT QUESTION FOR THE OWNER: halved decode-window latency + ~2,430-2,520 prefill vs
   doubled decode latency + 3,750 prefill - a workload-value decision, not a performance tie.
4. VERDICT CHANGES: the slice's ship rule is MOOTED (never ship in PP=2 form at any staging
   depth - its machinery folds into full sharding only; slice-alone is a component, not a
   milestone). Decode collective-free STRENGTHENED; RESIDENCY is the binding budget for every
   topology's additions (enforce the >=78 decode floor at boot-config review). The kill
   criterion worked as designed. MORNING CARRIES: (a) power-limit labels on all windows
   (AGENTS.md) before tonight's numbers circulate; (b) re-measure the FIXED =1 once (it is the
   corruption discriminator x P3); (c) the 2ea9a1ff class (gate claims 1-and-2, accepts 1;
   refusal swallowed) is a boot-line TRUTHFULNESS bug - add "refusals fatal + gate strings
   audited against the accept set" to the checklist; (d) quote 2,430-2,520 with the contention
   band. RETROACTIVE UPGRADE: share-v2 + continuation placement = REQUIRED balancing machinery
   for any lockstep design (the 2.1x row-rate asymmetry is a wall-time asymmetry only
   placement fixes).

**RUN 25 MEASURED (13:49 UTC, the slice's first graded boot — all four bring-up lines printed,
both pairs open, columns packed on both cards, 79 MiB resident projections copied at boot):**
- Serialized prefill: **1,833 t/s** (fresh 122k concurrent) vs the 2,004 baseline = **-8.5%**.
  The -2/-4 ms/layer-pair prediction was itself optimistic: measured loss ~15 ms/layer-pair,
  consistent with the corrected ledger once the FP32 return leg (~190 MB/QSA layer at the
  3.96 GiB/s staggered face ~ 47 ms) is priced and Ruling B's no-attention-halving is applied
  (the serialized mode pays full wire and saves no attention compute — a loss was the honest
  expectation; the stepping stone's job was validating the ledger and the machinery, both done:
  the reduce, publish, catch-up and timeout path all executed end-to-end with zero corruption).
- **Decode under the flag: 61.6 t/s cached (vs 77-82 baseline, -20%) — the tier displacement
  (1.89 partner + 1.16 lead GiB) costs decode rows too.** This likely breaches the solo floor
  (>= 78); the flag therefore STAYS OFF for serving (it is), and the decode cost of the
  residency joins the pipelined variant's ledger.
- The engine-exit-on-error contract hole (publish failure killed the engine at scale; wrapper
  crash-looped on a queued 228k prompt) is FIXED-WORTHY before the flag ever ships: a refusal
  must fail the request, never the process. Open.
- What the +10/+20 gates now grade: the pipelined (=2) variant WITH the lead's attention
  actually halved (option (a) pools, or the kernel taking the KV-head index separately from
  the query count — the pool-row formula is the only blocker), re-priced with the FP32 return
  (wire ~47 ms/layer unhidden) and the decode tier cost. Honest band: -20 ms/layer (wire
  unhidden, loss) to +27 (hidden, ~2,300 t/s). The hiding question - the same rendezvous
  cadence kimi #4 promoted - is measured on REAL machinery the moment commit 2 lands.
**Per-layer decomposition of the -8.5% (the honest read, not the done-line):** the lead's
attn GREW 13-15 ms (61.6-67.5 vs 48.5-54.4 baseline - the publisher's per-QSA-layer copies
and event serialization ride the lead's stream, plus L2 pressure from the staging), and the
lead's peer-wait exploded to 67.7-80.3 ms on the stage where the partner's serialized arm
sits on the critical path (publish 104.9 MB -> shard-width attention -> FP32 reduce
167.8 MB, all unhidden). The loss is not diffuse: it is two named costs, both of which
commit 2 (pipelined publish under MoE cover + the lead's attention actually halving) exists
to remove. Term-by-term per-layer reads are MANDATORY at every grading from here on - the
done-line t/s hid this decomposition entirely.



**RUN 23 engine-grade attempt (04:30 UTC) - INCONCLUSIVE, recorded honestly.** Formal A/B on
the merged binary (reseed on) vs the drafter-tree binary (reseed off, byte-identical), same
122k prompt + greedy 20k continuation: the reseed leg made 9,000 tok at 51.3 t/s (acc 75%),
the baseline leg 20,000 tok at 63.4 t/s (acc 90%). The generations DIVERGED across legs
(greedy is not content-stable across boots/slot states here), so the legs are different
workloads and the comparison is void. The bigger finding: the baseline shows NO collapse on
this prompt (63.4 t/s, acc 90%) - the nat-prompt continuation tail is verbatim-heavy, where
the min-8 gate already fires and the reseed floor is inert by design. The collapse class
(noisy parameterized repetition, the worker's sim streams) was never reproduced in the
engine; the real-grade needs that workload class (the owner's observed 40 t/s 70k session,
or a crafted noisy-repetition prompt) with --draft-log per the checklist. The sim evidence
(1.09->2.03 noisy / 1.1->2.7 clean) stands as the design basis; --exp draft_reseed=1 stays
on (inert on verbatim tails by construction, active where the sims say it pays).

**Morning falsifier reads (04:25 UTC, the share-v2 grade kimi #5 asked for):**
- Row-rate falsifier FIRED: lead 2.61 us/row-visit vs peer 0.98 (ratio 2.67x) - above the
  >= 1.4x threshold and above the shipped constants (2.26/1.09). The time objective
  UNDER-prices lead contention and over-assigns rows to the lead; the re-price is a
  boot-flag exercise (qsa_lead_row_us=2.61 qsa_peer_row_us=0.98), no rebuild.
- Term-by-term: the model's attn (54.4) and rows (89.5) track the chunk lines (attn
  48.5-50.0, lead-experts 87.8) within ~5-10% - the accounting is honest.
- blob_mb_ms still unmeasured (the one constant the model admits not knowing).
- Prefill re-confirmed 2,004 t/s fresh concurrent on this boot.**WINDOW RESULTS (10-31, the supervisor's window)**: (1) THE RENDEZVOUS BENCH (tpstep0,
progress-tp-cadence.md): the TP=4 PREFILL GATE FAILED at the real ratio (partial-merge
r=4 tokens=8192 cover=49.9 ms/layer: wire 39.6 ms/layer, hidden 42.7%, ceiling 56% —
the 70% line unreachable; the wire adds ~46% to the layer wall; the projection puts the
TP=4 lockstep ~1.4-2x SLOWER than the split's effective ~45 ms/chunk). The DECODE GATE
PASSED (42 us/round at 4 ranks, 90-93% hidden, straggler spread 0.000 ms) — but decode
alone cannot justify the build. The max-of-4 straggler REFUTED (inflation 1.00x); the
SM-contention hypothesis REFUTED (co-scheduling negative — the overlap works); row-ship
dead by capacity AND hiding (2.8%). VERDICT: TP=4 STAYS SHELVED, now on measurement
(TP4-SKETCH.md's gates (a) and (d) failed). The sglang prior was regime-mismatched.
(2) THE SHARE RE-PRICE: NULL. The flags read (2.61/0.98 in the pricing lines) but the
placement did not move (5429 held, the same band); prefill 1,993/1,994/2,002 vs 2,004.
THE REASON (the real finding): the peer idles ~170-190 ms/layer (lead 259-283 vs peer
93-96) but moving rows costs the RETURN WIRE at the shared fabric — the peer's back
wire (50.4 ms at its current share) scales with its rows and cancels the 1.63 us/visit
row-rate gain. The tp2 prefill is placement-optimal GIVEN the wire; the idle is
wire-protected, the same class as the slice's dependency-protected idle. The prefill
levers that remain: full sharding only (whose 4-rank hiding just measured 42.7%).
(3) The gap instrumentation landed in batch_step, but the tp2pp2 pair serving runs the
PIPELINED pump() path (piped ? pump : batch_step) — the profile lines never fire; the
decode-timing line DOES see the drumbeat (35-40 ms/window, GPU-reach wait ~12, per-layer
host 4.5-6.4). A port of the gap stamps into pump() is in flight (worker a04f42e0).

**KIMI #8 — the TP=4 arithmetic CORRECTED (the night's most consequential finding)**:
my prefill case under-priced the combine wire ~4x. The routed output crosses at
n_embd=2560-wide BF16 = 5,120 B per ROUTING (640 is the FFN hidden width — nothing
640-wide crosses a card), volume per routing T x K x (1-1/4) = 61,440/layer/chunk.
Row-shipping: ~250 GB total — EXCEEDS the entire d2d pool at zero exposure (dead by
capacity). Only partial-merge combine fits (~65-95 GB, 2.7-3.9 GB/s aggregate) — but
it changes the MoE sum's reduction order => PPL/KLD gate, non-negotiable. Side note
with teeth: partial-merge is 5x cheaper at TP=2 too (42 vs 210 MB/layer-chunk) —
"why is the current pair row-shipping" is now an open question. The sglang prior
downgraded to weak evidence (compute-bound + shared pool vs bandwidth-bound +
per-link-independent). The bench got three mandatory extensions (two-channel cadence,
decode-cadence mode, two combine channels; SM/copy-engine separation mandatory).
The collective-free decode ruling EXTENDS a fortiori to TP=4 — the natural design
generalizes the peer-window/ring from 2 to 3 peers (dec_moe.hpp kDmMaxX=2 is the
literal seam). The window order: gap -> extended bench -> share re-price -> power
piggybacked (dmon during the other boots; power is a MISLEADING idle indicator in
compute-bound phases unless phase-tagged). The drafter retired as done-at-scale.
Owed: a split-concurrent baseline (matched dl) before any morphology claim.
Full verdict filed; the corrected ledger + the gates + the decision rule in
TP4-SKETCH.md. The lever ranking UNCHANGED in order.

**FIRST MEASURED CONCURRENT DECODE AGGREGATE (live, the worker code-gen class)**: two
slots' agentic turns interleaving — done-lines acc 68-88%, dl 2.7-4.1, 51-76 t/s per slot;
the live totals swinging 79 -> 93 t/s, with both-slots-speculating-well bursts crossing
~114. The drafter reseed's payoff at scale: the acceptance holding high through real
generations on the target class. The split's CONCURRENT decode was never measured (solo
74-90 only) — a missing baseline for the topology decision; if the split degrades under
concurrency the way its prefill slots do, tp2's concurrent aggregate may already exceed it.


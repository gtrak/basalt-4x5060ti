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

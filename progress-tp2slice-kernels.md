# RUN 25 — the head split in the kernels: what was hardcoded, and what proved it

The companion to `progress-tp2slice-ledger.md` (bytes, funding, predictions). The rule this file exists for:
**a split is only real where the writer and the reader agree**, so every kernel the partner's arm passes through
is audited on both sides, not just the ones that read a cache.

## The two kernels the arm needs, found before writing the compute

`split_q_kernel` and `gate_attn_kernel` (`src/prefill/kernels.cu:556-568`) **hardcode 24 query heads**:
`if (i >= T * 24 * 256) return;` and `qf[t * 24 * 512 + h * 512 + d]`. Everything else the arm calls takes its
counts as arguments — `rms_rows(x, w, rows, cols, ld, ...)`, `rope(x, T, heads, dim, ld, pos0, ...)`, and
`qsa_prompt_attn_batch(..., s, ...)` which runs at shard shapes already (proven by case 6 of
`tests/kernels/tp2_head_split_parity.cpp`, which calls it with `s0`/`s1`). So the head split reached the
attention kernel and not the two elementwise ones around it.

Two ways to give the partner a shard-width arm:

- **(a) full-width scratch on the partner.** Allocate `Qf`/`q`/`attn`/`attn_h` for 24 heads and reuse the two
  kernels unmodified. No new kernels, but the partner pays VRAM for heads it does not compute — `Qf` alone
  doubles — taking scratch from 0.64 to roughly 1.0 GiB and the tier figure from 1.89 to ~2.3 GiB.
- **(b) shard-width variants of the two kernels.** Two mechanical mirrors of the existing ones, taking the head
  count and the `Qf` row stride as arguments. Keeps scratch at 0.64 GiB. AGENTS.md is explicit about what this
  owes: every new kernel gets a parity test (random inputs + real geometry) and a microbench line.

**Taking (b)**, because the head split's premise is that a card does not pay for heads it does not run, and
`qsa_shard_shapes` exists to narrow exactly this; making the partner reserve 24 heads to save writing twelve lines
of kernel would contradict the design to save effort. The parity test extends case 6's harness rather than
inventing a new oracle, and the microbench line goes in the same run's table. If this is judged the wrong trade,
(a) is the fallback and the tier figure above is the number that changes.

Also settled while reading, and it changes what crosses the wire: **the lead ships its own `steps`** rather than
the partner recomputing them. They are host-computed per token (`qsa_step_fill` at prefill.cpp:2703) and uploaded,
they are a property of the chunk's positions rather than of a layer (so one set, not one per wire set), and
shipping the lead's bits means `width` and `active` cannot disagree between the two cards' selections. Same
reasoning took the indexer **q** norm off the partner's list: the lead ships `q_idx` already normalized and
rotated — the exact bits it fed its own block scores — so the selection is the lead's by construction rather than
by two cards repeating a computation and hoping. The indexer **k** norm stays resident, because the partner's
append normalizes each cell's raw key by it into a pooled table of its own.

## 2g. The third hardcoded-head site: the cache writer

Found while transcribing the lead's QSA block into the partner's arm, and the most dangerous of the three:
`kv_append_kernel` took its KV-head count, its `K`/`V` row stride **and both pool row formulas** from a literal 2
(`dim3(T, 2, 8)`, `K[t * 512 + ...]`, `(page * 2 + kvh) * page_size`, `((pos / page_size) * 2 + kvh) * page_size`).
A partner holding **one** KV head would have read one head past its own `K`/`V` and written each cell at a
two-heads-per-page stride into a pool carved one head per page — a corrupted KV cache, which poisons every layer
after the one that corrupted it and still produces plausible tokens.

It was invisible because the *reader* was already shard-aware: `qsa_prompt_attn_batch` takes `QsaShapes`, which is
why case 6 passed. A split is only real where both the writer and the reader agree on it, so a head-split audit
has to cover writers too.

Now `kv_append_nkv(..., nkv)` with `kv_append` = that at the model's `n_head_kv`, so lead and card share one body
and `prefill.cpp` is untouched. Case 8 compares each card's pools against its KV heads' rows of the whole run's
pools, in **both** the FP16 and the shipped INT8 layouts — the two take different branches of the kernel and
share only the row formulas, so a formula error visible in one format but not the other is exactly what a
single-format test would miss. Bitwise, unaligned `pos0` so the tail block has to land, and the card's `K`/`V`
built as its own contiguous `[T, nkv, head_dim]` rather than a strided view, because that is what the arm hands it.

Three sites now audited: `split_q`, `gate_attn`, `kv_append`. **Not yet audited**, and the arm will pass through
each: `native_qsa_indexer_append_batch`, `qsa_block_scores_tc`, `qsa_block_topk`, `rope`, `rms_rows`, `fwht256`.
The last two take their counts as arguments, so the arm must pass **its own head count and its own `Qf` stride**
rather than the lead's `24`/`6144` — the same class, in the caller this time.

## 2h. The audit's other three: the selection kernels need no parameterization

The rule says audits cover writers, so the three the arm still passes through were read line by line rather than
assumed safe. **Result: none of them hardcodes anything on the split axis, and none needs a head-count argument.**

| kernel | what it reads of `QsaShapes` | on a geometry mismatch |
|---|---|---|
| `native_qsa_indexer_append_batch` | `idx_dim`, `idx_block`, `n_rot` | **throws** `std::invalid_argument` |
| `qsa_block_scores_tc` | `idx_dim`, `idx_n_head`, `idx_block` | **returns false**; caller keeps the warp kernel |
| `qsa_block_topk` | `idx_block`, and `cap` vs `qsa_selection_width` | **exits** with the geometry named |

None names `n_head`, `n_head_kv` or `head_dim` — not in the wrapper and not in any kernel it launches
(`block_scores_tc`, `block_scores_tail`, `block_topk_reg`, `block_topk_wide`, `block_topk_cluster`, the ref and seg
variants). The reason is structural, not lucky: **the indexer does not split**, because two of four indexer heads
would pick a different top-k, so both cards compute the same selection redundantly and every one of these shapes
stays whole on the partner. `qsa_shard_shapes` narrows `n_head`/`n_head_kv` and nothing else.

So the arm passes `core::qsa_shard_shapes(g, heads)` to all three unchanged, and the failure mode if that claim
were ever broken is loud in each case — a throw, a refusal that falls back, or an exit naming the geometry. That
is the property the other three kernels lacked: `kv_append` and friends were silently wrong where these are loud.

The claim these three's safety rests on — the shard shapes keep the indexer whole — is pinned by
`the_indexer_stays_whole()` in `tests/core/qsa_head_split_test.cpp`, which now also pins `head_dim` and `n_rot`
staying whole, because the arm hands both straight to `rope` and `rms_rows` where a narrowed value is a wrong
rotation rather than a crash.

**Audit closed.** Hardcoded-head sites found and fixed: `split_q_kernel`, `gate_attn_kernel`, `kv_append_kernel`.
Found and cleared as not-applicable: the three above. The remaining exposure is not hidden constants but the
caller's own arguments: `rope` and `rms_rows` take head count and strides as parameters, so the arm must pass its
**own** `nq`/`nkv` and its own `Qf` stride (`nq * 2 * head_dim`), never the lead's `24` and `6144`.

## The arm, and the selection claim it is built to satisfy

`peer_qsa_layer()` in `src/prefill/prefill.cpp` is the mirror of the lead's QSA block for one card: the lead's
shipped rows in, this card's own `Gemm` over its boot-copied weights, its heads split and rotated at **its** head
count and **its** `Qf` stride, its cells into **its** pools via `kv_append_nkv`, the indexer appended into its own
pooled table, the selection, its heads' attention, the gate. It runs on `qs.s` and refuses rather than approximates
wherever it could not match the lead bit-for-bit: a KV format it does not implement (q4, rotated, hybrid,
streamed), a pack type it was never given, weights it was never handed. Every one of those refusals is the
silent-corruption class this slice keeps hitting, arriving as an error instead of as plausible tokens.

Two deliberate asymmetries with the lead, both load-bearing:

- **the selection is the lead's by construction.** The partner receives the lead's `steps` and its already-normalized,
  already-rotated `q_idx`, and skips the q norm and the q rope the lead applies - applying them again would rotate
  bits twice. Case 9 is the check that this works: it runs the selection at the whole layer's shapes and at each
  card's shard shapes and asserts the ids are identical, slot for slot. This is a *different* claim from case 6's
  (which covers outputs), and a wrong answer there means the card attended to different cells than the lead - no
  bad number anywhere, no way for anything downstream to notice. It also pins the audit result: if
  `qsa_shard_shapes` ever narrowed an indexer field, or one of the selection kernels started following `n_head`,
  the ids part here.
- **it ends at `attn_h`, not at the partial.** The o_proj product needs this card's packed `attn_output` columns,
  which `qsa_pair_prepare` allocates and which arrive with the pair's collective. The product, the BF16 conversion
  and the reduce to the lead are one unit; splitting them would leave a half-finished reduce on the branch. The
  arm therefore has **no caller yet** - it is wired by the `span_valid` and reduce-to-lead steps, not before.

The attention has no decode fallback, unlike the lead's: that fallback exists for shapes this arm already refused
above, and it would need scratch the walk does not fund. A refusal there is an error naming itself, not a slow path.

## The suite ran, and found one the audit missed

The parity cases above were written in earlier windows and **had never executed** — GPU-labelled, cards busy every
time. With the cards idle they ran, and case 8 failed immediately: `gate_attn_kernel` took a head offset and applied
it to `qf` but not to `attn`, so a wide buffer plus a head span multiplied head `h0+h`'s activations by head `h`'s
gate. Plausible numbers, no crash, in the kernel whose output feeds o_proj.

It is not the same bug as the three hardcoded counts, and the audit did not catch it because the audit asked *does
this kernel assume a head count* — and this one took its count as an argument, correctly. The question it should
also have asked is **does it apply every one of its index arguments to every input consistently**. A one-sided
offset is invisible to a head-count audit.

The offset is gone rather than fixed, because both of the kernel's inputs are the card's own rows and no caller ever
wanted it. `split_q_span` keeps its offset — it genuinely reads wide and writes narrow — and the header now states
the two contracts separately instead of one comment covering both.

The GPU run also found two ways the *tests* could lie, both fixed: the cache-writer case compared the INT8 **scale**
plane at the data stride (one amax per 64 dims, not 256), so it was comparing untouched memory and reporting it as
a mismatch — and the V scales were allocated and never compared at all. A test that fails on a stride bug looks
exactly like a test that fails on a kernel bug; the only way to tell them apart is to read the indexing.

**Status change worth noting: the GPU-labelled parity cases are no longer unrun.** All of cases 1–10 report bitwise,
including both KV formats. Which means the remaining unverified surface is the run path, not the kernels.

## Step 4: the span's proof, and why decode does not need one

`span_valid` is the number of cells `[0, span_valid)` that the partner's pools and indexer planes provably hold,
for every QSA layer of its stage. One number for the stage, not one a layer: the layers advance with the sequence
and are invalidated together, and a per-layer value would only add a way for one layer to be quietly left behind.

It is a **proof, not an estimate**. The partner writes its own cells during a prompt and knows them first-hand;
what it cannot know is what the lead appended while it was out of the loop. Anything that can move the lead's
state without the partner appending it drops the number to 0 and the next prompt re-imports — re-importing is
correct and merely slow, while believing a stale number is the silent-corruption class this branch refuses
everywhere. `invalidate_peer_qsa_span(why)` names the site on the log line, so a boot that keeps invalidating
shows up as cost rather than as a mystery.

**Sites that invalidate**: `Prefill::set_session`; `prompt_session_clear`; the state copied *in* from the main
session; the state copied *back*. **The parked-conversation restore** is the one that is easy to miss —
`conversation_snapshot_restore` writes K/V directly into a session and goes through neither copy, so the copy
sites above cannot see it.

**Site that deliberately does not: the end of a decode window.** Ruled, not tolerated. Decode only ever *appends*
cells past everything a prompt attended over, so `[0, span_valid)` still holds the lead's bits afterwards and the
proof survives; the catch-up then imports exactly what decode added. Dropping the span at every window's end would
be *safe*, and would re-import the whole context on nearly every prompt a server reads, to defend a claim that was
never broken.

**The invariant this rests on: cells below `span_valid` are never rewritten or dropped.** Decode append satisfies it
because decode only ever writes at positions past everything a prompt attended over. It is the guard, not a
footnote — any feature that rewrites, shifts or evicts below `span_valid` must invalidate there, and the obvious
candidates are named so nobody has to rediscover the list:

- **KV compaction** — repacking or moving cells to reclaim pages rewrites rows by definition;
- **context shift** — prefix removal, prompt trimming or a position re-anchor relabels which cell holds which token,
  so the partner's rows still contain *something* and it is the wrong something;
- **sliding-window / attention-sink eviction** — dropping old cells leaves the partner holding cells the lead no
  longer has, which is the mirror image of the stale-span case this whole protocol exists to prevent.

Each of those would need its own `invalidate_peer_qsa_span("<feature>")` at the site that does the rewriting, not
a new rule here: the protocol is already "anything that can change the lead's cells without the partner appending
them drops the proof". What would change is that such a feature *can* now occur, where today only state arriving
from elsewhere can.

**The catch-up runs once a turn**, before the first chunk, not once a layer: the gap is a property of the
sequence, and pricing it per layer would charge for the same bytes six times.

Two things the import refuses rather than approximating: a KV format it cannot represent (q4, rotated, hybrid,
streamed) and **any disagreement between the two cards' KV format** — the bytes copied are raw, so INT8 codes
written into an FP16 pool would be a plausible cache of nonsense. It also refuses outright when the lead's states
or their card were never handed over: importing nothing while reporting the span caught up is the exact failure
shape this slice keeps hitting, so the run fails instead of degrading. `lead_dev` exists because a peer copy is
stated as `(dst, dstDev, src, srcDev)` and `QsaState` carries no device of its own — guessing it would make a copy
read this card's own memory and call it done.

## Two questions per kernel, not one

The audit rule this branch now applies to every shard-facing kernel, from the supervisor after the
`gate_attn` find: for each kernel a card calls, ask **both** —

1. does it *assume* a head count (a literal 24, a 2-heads-per-page row formula, a 6144 stride)?
2. does it apply every index argument it takes to every input **consistently**?

`gate_attn_span` passed question 1 cleanly — it took its count as an argument — and failed question 2:
the offset reached `qf` but not `attn`. A head-count audit alone cleared it. Question 2 is what catches a
kernel that is told about the split by half its arguments.

Applied so far: `split_q_kernel` (1: yes, fixed), `gate_attn_kernel` (2: yes, fixed by deleting the
argument), `kv_append_kernel` (1: yes, fixed), the three selection kernels (neither, cleared — they read
no head field and refuse a bad `idx_*`), and `kv_span_import_kernel`, which is written against both: it
takes each side's `nkv` and each side's page table explicitly and resolves the row twice, once per card,
rather than sharing either.

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

**Landed** — commit `b1f9850d`, `tests: the head-split parity gains the prompt attention kernel`:

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

**Not built** — task items 1, 2, 3 and commit 2 remain open:

- `qhs_peer_item()` still returns false (`src/core/verify.cpp:598`); no peer-side prompt-attention
  executor exists, and the init refusal still stands.
- No `--exp tp_prompt_attn` flag, no AR call site in `src/prefill/prefill.cpp`.
- No change to the peer's expert tier carve to fund the 1.2 GiB span (ruling B).
- The software-pipelined variant (commit 2, `=2`) is untouched.

Unverified assumption that must be settled first, because the whole inbound mix depends on it:
whether the peer can consume the **FP4 act planes** it already receives (RUN 19) as the attention
input, and what plane consumption does to attention parity. If the tolerance above cannot cover
it, the table in section 2 (BF16 mixed rows) is the fallback and is what is priced.

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

- `qhs_peer_item()` still returns false (`src/core/verify.cpp:598`); no peer-side prompt-attention
  executor exists, and the init refusal still stands.
- No `--exp tp_prompt_attn` flag, no parity extension to `tp2_head_split_parity`, no AR call site
  in `src/prefill/prefill.cpp`.
- Unverified assumption that must be tested first, because the whole inbound mix depends on it:
  whether the peer can consume the **FP4 act planes** it already receives (RUN 19) as the
  attention input, and what plane consumption does to attention parity. If the tolerance cannot
  cover it, the table above (BF16 mixed rows) is the fallback and is what is priced.

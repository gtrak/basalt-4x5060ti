# RUN 25 ledger — the prompt-attention slice: bytes, funding, predictions

The status, rulings and open items live in `progress-tp2slice.md`; this file is the arithmetic. Every figure here
is either derived from a sizer a test pins (`tests/cuda/peer_qsa_span_test.cpp`) or labelled as an estimate
awaiting the GPU window. No number here was measured on hardware.

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

## 2d. Two rulings applied to the sizing: wire doubled, predictions re-based

**(b) ruled: the wire staging is resident in TWO sets; the attention scratch stays in one.** The pipelined gate's
whole purpose is layer l+1's inbound on the link under layer l's compute, and a single staging set cannot do
that — the lead would wait for the partner to finish reading layer l's rows before writing layer l+1's. The
returning partial is doubled for the same reason. The scratch stays single: every buffer in it is written and
read inside the partner's own compute for one layer, on its own stream, and dies when that layer's partial is
pushed — the overlap is on the link, not on the partner's compute. If the arm's structure ever does hold one
across the boundary it gets its own second set and the comment names which; the block is not doubled on spec.

The sizer test separates the two currencies and prints the delta, so neither can be mistaken for the other:

    wire: traffic  104.9 MB a QSA layer   <- what the ledger prices and the +10/+20 gates grade
            resident 209.7 MB (2 sets)    <- what the partner's tier pays
    second wire set costs +105 MB of tier   (fp16 2.30 -> 2.40 GiB; int8 1.58 -> 1.67 GiB)

**The published tier figure is 1.67 GiB int8** (span 0.96 + wire 0.21 + scratch 0.55), superseding §2c's 1.58
and my hand-estimated 1.006/1.2. Traffic is unchanged at 104.8 MB — the doubling buys overlap, not bandwidth.

**(a) ruled: 1.67 GiB (~10.6% of the 15.8 GiB half-depth carve) clears Ruling B's intent**, because that ruling
was about the *source* (the peer's expert tier — not `--peer-reserve-mib`, not the lead) and the *discipline*
(the shrink priced in share-v2's model rather than hidden in a reserve). Neither moved when the figure did.
What moved is the arithmetic the projections carry: see §3a.

## 2e. The resident projections and the partner's own GEMM context (RUN 25 rulings)

**(2) ruled: RESIDENT COPIES, P2P weight reads rejected.** The partner projects its own q/k/v, so its shard's
rows of `attn_q`, its KV head's rows of `attn_k`/`attn_v`, and the q/k RMS norms live resident on the partner's
card, copied at boot: a boot-time cost and **zero per-layer link traffic**. P2P reads of the lead's weights were
rejected because they would put an unpriced weight-sized transfer on every chunk, invalidating the 104.9 MB the
gates grade against, and re-introduce the move-data-inefficiently pattern this campaign exists to delete.

The sizer sees geometry only and block geometry belongs to the pack, so `q_row_bytes`/`kv_row_bytes` come from
the caller's `WeightRef`s; **with them unset the sizer prices no weights at all rather than inventing a
quantization table.** The test pins the row runs as literals — 6144 rows of `attn_q` (12 heads x 2 x 256) and
256 rows each of `attn_k`/`attn_v` at the symmetric cut, x 6 layers, plus the two norms — so a wrong row run
shows up rather than quietly under-funding the tier.

**(1) settled from the API: the partner runs its own `Gemm`.** `Gemm::native` dequantizes each weight's GGUF
blocks into a scratch **on the device the product runs on**, and the partner's products run on the partner, so
it cannot borrow the lead's scratch or cuBLAS workspace across the link. The span therefore carries a second
card's worth of both: `GEMM_SCRATCH * 2 + GEMM_WS` = **0.093 GiB**. This is why the partner's arm will hold its
own `Gemm` rather than share `Impl::gemm`.

The tier figure, one honest number, int8 shipped boot (`--kv int8`, 262144 cells, 8192-token chunk):

    span     0.96 GiB   KV span + indexer pooled, 6 QSA layers at the shard
    wire     0.20 GiB   2 sets (traffic stays 104.9 MB a layer)
    scratch  0.52 GiB   shard-width q/k/v/attn + selection, one set
    weights  0.05 GiB   resident projections at a Q2_0-class row (pack-dependent: 6 x 6144 x q_row_bytes + ...)
    gemm     0.09 GiB   the partner's own dequant scratch + cuBLAS workspace
    TOTAL    1.82 GiB   ~11.5% of the 15.8 GiB half-depth carve

1.77 GiB is the figure with the projections unpriced (the test's default, pack not consulted); 1.82 GiB is with
them priced at a Q2_0-class 1280 B row. **The boot reads the real row bytes from the pack, so the boot line must
print this total** — the projection term is the only one that depends on the quantization type, and a heavier
type moves the tier figure without moving anything else.

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

### 3a. The re-based predictions: what the extra tier shrink does to them

The `+18/+15` above was taken while the slice's tier cost was believed to be **1.2 GiB (7.6% of the 15.8 GiB
half-depth carve)**. It is **1.67 GiB (~10.6%)**: span 0.96 + wire 0.21 + scratch 0.55. The traffic the
predictions are built on is unchanged (104.8 MB a layer, both modes), so the re-basing touches only the
expert-residency term, and that term is the one the window measures rather than the one this file models:

- The shrink is ~0.47 GiB deeper into the partner's expert tier. Experts evicted from that tier stop being
  tier hits and become streamed rows, which raises the **partner's per-row time** — the quantity the share-v2
  model prices with `qsa_peer_row_us`, and the quantity the fired falsifier read at **0.98 us/visit** against
  the lead's **2.61**. The slice's gain is proportional to that asymmetry, so a shrink that pushes the partner's
  row rate up moves the net down directly, not secondarily.
- **No fabricated number for it.** The cost depends on how many of the partner's tier experts the eviction
  displaces at this boot's placement, which is a per-boot fact this file cannot price. So the prediction is
  re-filed as a **bounded statement, not a point**: the `+18/+15` stands as the **no-extra-eviction** case,
  and it is an **upper bound**. The window's falsifier re-read — `qsa_lead_row_us` / `qsa_peer_row_us` and
  `blob_mb_ms` read **after** the slice is resident, against the same boot's pre-slice values — is the
  instrument that prices the difference.
- **What to read at the window, per boot:** the two row rates and `blob_mb_ms` before and after the slice, plus
  the chunk line's span-copy fields (`span_valid`, cells copied, copy wall). If the partner's row rate rises and
  the net lands under +10 ms, **the honest verdict is STOP with the eviction cost named as the cause**, per the
  ratified kill criterion — not a re-tune of `qsa_ratio_stage_*` to recover the margin, and not a re-basing of
  the bar. Per the ruling: if the measured net misses the ship bar because of the extra shrink, that is the
  number.
- The **serialized (`=1`) prediction of `-2/-4` ms is unaffected**: it is dominated by the 26.5 ms of wire on
  the critical path, which the tier change does not move.
- The **turn-start span copy is a separate term** from both (§2b), and it is unaffected by the doubling: it is
  a per-turn move of at most the 0.96 GiB span, not a per-layer one.

If commit 2 measures under +10 ms/layer-pair, **STOP and re-plan** per the ratified kill criterion.
I will report the number as measured; it will not be tuned to pass.


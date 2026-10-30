# Supervisor rulings on record (RUN 25 onward)

Split out of `progress-tp2slice.md` to keep that handoff inside the 300-line limit. These are the supervisor's rulings, verbatim; the file that acts on them is `progress-tp2slice.md`, and the arithmetic is in `progress-tp2slice-ledger.md`.


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


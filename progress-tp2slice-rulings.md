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
- **The whole-width-partner-pools ruling is RETRACTED by the supervisor; the worker's refusal of it is ratified.**
  The ruling was that `peer_qsa_layer()` should run `qsa_prompt_attn_batch` at whole geometry (`n_head = 24,
  n_head_kv = 2, kvh0 = 1, groups = 1`) because the partner was "reading the lead's KV rows". Measured against the
  code the premise is false: the partner's pools are carved at its shard (`qsa_prompt_layout` sizes them
  `pages * s.n_head_kv * s.page_size` from the shard shapes; 0.281 GiB a QSA layer against 0.531 GiB whole), its
  writer agrees (`kv_append_nkv`), and running it whole would index `(page * 2 + 1) * page_size` past its own
  allocation - an out-of-bounds read, not a correction. The real `=2` defect was elsewhere (the single shipped
  step-count set). What stands from that ruling is the **enforced head-width invariant**: the arm refuses a layer
  whose `QsaState` width disagrees with the `QsaShapes` handed to the kernel. Standing rule that came out of it: if
  measurement contradicts a ruling's premise, surface the measurement and hold the change rather than implementing
  the ruling.
- **Ruling (A) - `=1` must be what it claims.** The `ev_out` index asymmetry (record at `ev_out[0]`, wait on
  `ev_out[wire]`) meant half the serialized mode's layers ran pipelined; fix it, re-label the filed `=1 -8.5%` as a
  half-pipelined number, and re-measure both modes. Accepted knowing it makes `=1` slower than its own baseline.
- **Ruling (B) - the pair audits itself.** Approved as specified: 8 floats a layer a card into a ring, compared at
  the chunk boundary beside the `timed_out()` read, failing the run named. Its purpose is to turn the `=2`
  corruption hunt from N boots of guessing into one boot that says "in the pair" or "outside the pair".
- **The structural finding, and what it does to the ship bar.** With `=1` and `=2` measuring the same and the
  peer-wait unmoved by staging depth, the supervisor's ruling is that **-8.7% under PP=2 is the cost of the
  rendezvous, not an unhidden corner** - the partner's attention of l+1 consumes what the lead's MoE of l produces,
  so no buffering removes the dependency. The +10/+20 bar is therefore not reachable by this slice under this stage
  structure; it is reachable only under full TP=2 sharding (no stages, both cards in lockstep, the MoE halved too).
  **Whether sharding's complexity buys enough is the owner's decision**, with `progress-tp2slice-verdict.md` as the
  evidence for it.
- **Device-test yield doctrine.** A GPU test run while the serving engine is up is not automatically invalid (the
  race is device-side; host load does not serialize cross-card streams), but it must be **re-run in a quiet window
  for the record**, and the same courtesy the engine is owed applies to any test that occupies cards.

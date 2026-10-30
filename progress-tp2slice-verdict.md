# RUN 25 — the verdict the measurements reached: splitting attention under PP=2 costs what it costs

Figures in this file come from the maintainer's own windows (owner-run boots, engine at the pinned 50/50 head
split, `--exp tp_prompt_attn` per mode). **The window logs do not record the GPU power limit, so these numbers carry
no limit label** — a gap worth closing before any of them is quoted outside the branch. Where a figure is the
branch's own arithmetic rather than a measurement it says `derived`.

## 1. What was measured

Both modes, same cached `nat` prompt, same boot configuration, machine otherwise quiet:

    =1  (serialized, and see §3: it was half-pipelined)   ~1,833 t/s
    =2  (pipelined, two staging sets everywhere)           ~1,826 t/s
    =2  per-QSA-layer peer-wait term                      52-66 ms
    =2  a small chunk (854 tokens)                          52.2 ms a layer

The two modes measure the same, to within the run-to-run spread, and the pipelined mode's per-layer peer-wait did
not move from the serialized mode's. **Buffering the wire bought nothing.** That single observation is the finding:
it is not a tuning gap, it is what the structure requires.

## 2. Why the wire cannot hide

The pipelined mode removes the lead's end-of-layer wait, so the lead's stream queues ahead. What it cannot remove is
what the partner's work *consumes*:

    partner's attention(l+1)  ←  mixed_h(l+1)  ←  the LEAD's MoE of layer l

`mixed_h` is the lead's mixed-precision activation rows for layer l+1, and the lead produces them out of its own
MoE of layer l. So the publish of layer l+1 cannot be issued before layer l's MoE has completed on the lead's
stream, and the partner's attention and reduce for l+1 therefore sit **on the lead's critical path at any number of
staging sets**. The second set (and the second collective region, and the second step-count set) overlap the
*transfer*; the compute dependency is immovable. No depth, event, or copy stream changes that, because the bytes the
partner needs do not exist yet.

The arithmetic closes on it — `derived`, from run 25's own per-layer decomposition:

    partner attention serialized after the lead's MoE   ~27 ms
    the pair's reduce                                    ~40 ms
    less the lead's own halved attention                 -23 ms
    over the 6 QSA layers of a 24-layer stage, against a ~48 ms/layer baseline
    ⇒                                                        ~ -8.7%

which is the -8.5/-8.7% the two modes measured. The slice's cost is the price of the *rendezvous*, paid once per
QSA layer, and the rendezvous is where the two cards' halves of one layer's output have to meet.

## 3. What this retrospectively disproves about the `=1` baseline

`--exp tp_prompt_attn=1` recorded its end-of-layer proof at `ev_out[0]` every layer but waited on `ev_out[wire]`,
which alternates — so **every odd QSA layer waited on an event nothing had recorded**, which CUDA answers
immediately. Half the layers of "serialized" ran pipelined. The -8.5% figure is therefore a half-pipelined number
(consistent with §1: it landed on top of =2's). Fixed in this change: both sides now use the staging set's index,
and `=1` is truly serialized. Both modes need re-measuring with the fix and the tripwire live; expect `=1` to move
*down* toward the serialized cost its definition implies, and the gap between the modes to stay ~zero for the reason
in §2.

## 4. What the slice would have to become to pay

The cost above comes from an asymmetry: one card leads, the other waits on the lead's activations. So the slice
pays only when the asymmetry is gone — **full TP=2 sharding, without the stage structure**:

- both cards compute the SAME layers in lockstep, so neither waits on the other's half of a *previous* layer;
- the attention halves for both, and the MoE halves too — halving only one card's work leaves the other card's
  unhalved work on the critical path, which is what §2 measured;
- the reduce's ~40 ms a QSA layer is then paid by both cards equally, and it is the only added synchronization;
- the projection residency, the packed `attn_output` columns, the shard kernels, the KV/indexer state and its span
  proof, and this slice's parity tests all carry over unchanged: they are sharding machinery, not stage machinery.

TP=4's compute side measured clean (no width penalty), so the sharding itself is not the risk; the risk is the
orchestration the stage structure currently buys (EP+PP placement, the expert tier, and the peer share that reuses
this slice's wire protocol).

## 5. The decision this hands the owner

Under PP=2 the attention slice is **structurally -8.7%**: not an unhidden corner, not a depth problem, not the
wire's format. Under full TP=2 sharding it projects positive, on the arithmetic in §4 and on TP=4's clean compute
side. The open question is whether full sharding's complexity — losing the stage structure's expert placement and
its peer share — buys enough to be worth building. This file is the evidence for that choice, not an argument for
it.

## 6. What is NOT concluded

- **Nothing here indicts the wire's format.** A BF16 return leg is still priced (halves the FP32 return, ~47 →
  ~24 ms of wire) and still irrelevant to §2: bytes that are ready early were never the constraint.
- **Nothing here says the arm's protocol was right the first time.** Three real `=2`-only defects were found and
  fixed in this run: the single collective landing region (`e0d94223`), the single shipped step-count set
  (`0db8cc63`), and `=1`'s half-pipelined wait (this change). §2 says fixing them would not have made the mode
  faster; it does not say they were optional — each one was a wrong answer reaching the model.
- **The `=2` token corruption is not yet closed.** The pair now carries an agreement check (each card samples the
  sum it ended with; the rings are compared at the chunk boundary and a disagreement fails the run named). The next
  window either confirms the structural number with a clean run or says "in the pair", which is where the remaining
  hunt lives.

## 7. What `=2` actually changes relative to `=1` — read from the code, not guessed

The depth decision has exactly one source (`peer_qsa_sets()`, `prefill.cpp:3117`, reading `--exp tp_prompt_attn`),
one stored figure (`PeerQsaSpan::staging_sets`), and one derived mode flag (`qsa.pipeline`, `:2905`). Every reader of
them, in full:

- **the arm's set indexing** — `part_f32` and `steps` allocation (`:2146`, `:2156`), the pair's `open(…, sets)`
  (`:2279`), and the eight `pipeline >= 2 ? wire : 0` index sites across publish, arm, finish and the driver's wait
  (`:2509`, `:2534`, `:2539`, `:2588`, `:2630`, `:2660`, `:2733`, `:4392`); and
- **the lead's end-of-layer wait**, present at `=1` and absent at `=2` (`:4391`);
- **two residency figures**, both priced from the same number: the partner's tier carries a second `o_proj` staging
  set and a second step-count set (~84 MB), and the lead's reserve carries a second collective landing region
  (~80 MB, `:3304`).

Nothing outside the prompt-attention arm reads the flag, so a `=2`-only corruption lives in the arm.

**The paragraph that used to end this section was wrong, and the wrongness is worth keeping where it can be read.**
It said the agreement check's silence would leave only the residency delta, because the wait and the set indexing
were "already accounted for". They are accounted for as *asymmetric* failures only. The check compares what the two
cards ended up holding, and each card ends up holding `own_partial + peer_partial` — so if the partner's attention
ran over the wrong KV, or with another chunk's step counts, or on a wire set the lead overwrote early, both cards
hold the SAME wrong sum and agree element for element. **The check is blind by construction to a wrong input, which
is the exact signature the window returned: silent rings, `!` to the cap.** A pair that agrees on a wrong value
agrees perfectly, and "the pair is sound" was never what the check could prove.

## 8. The chain: what it can say, and the measurement that limited it

> **READ THIS FIRST — the cross-boot use of this instrument is RETRACTED.** It was built to diff gate-off against
> `=1` against `=2`, and the three boots were captured. They cannot be compared: the engine's prefill kernels
> accumulate in a schedule-dependent order, so the two passes of a SINGLE boot disagree with each other — measured
> on the captures, 16 of 180 layer-lines of `mix` differ between pass 0 and pass 1 of one `=1` boot, and `sel`,
> `kpool`, `vpool` and `bo` diverge on all 180. A cross-boot diff of that is noise against noise. What survives is
> WITHIN-boot structure (did a mode record the components at all, does a peer's copy of an input match the lead's
> within one pass, which layers were touched) and the ratios. The correctness question moved to §8a.
>
> Two structural facts from the capture, both worth keeping: each boot contains **two** passes of the prompt and
> **two** `qsa chain: on` headers (one per `Prefill` that armed the dump), so a split by header count halves the
> data; and every `(card, chunk, layer)` key appears exactly twice, which is the two passes, not a duplicated
> record.

`--exp qsa_dump_chain=1` fingerprints every component of a sliced QSA layer as the layer completes — on the stream
that produced it, so it records what the layer used rather than what the buffer holds by the chunk's end — and prints
one line per card per layer. Indexed by MODEL layer, so a line names a layer the boot names elsewhere.

**The read, within one boot** — the lines are grep-able by construction:

    grep 'qsa chain:' boot.log | sed 's/^.*qsa chain: //' > chain.<mode>.txt

The comparisons that survive the retraction are the ones whose two sides come from the SAME pass of the SAME boot,
because kernel nondeterminism cannot confound them — both buffers were written by the same run:

| within one pass, this pair differing means | |
|---|---|
| lead `mix` ≠ peer `pub` | the link or the wire set: the partner attended on rows the lead did not publish |
| lead `steps` ≠ peer `stepsr` | the class that made `=2` select with another chunk's positions, still alive |
| lead `kpool`/`vpool` vs peer `kspan`/`vspan` | the import's shape: the two cards hold different KV for the cells they both claim |
| `suml` ≠ `sump` | the collective itself — and the agreement rings should have failed the run already |
| a component reading `--------` | that mode never recorded it: an answer, not a gap |

The comparisons that do NOT survive are any across boots or across passes — `bo` against gate-off's `bo`, a layer's
hash against the same layer in another boot — which is what the table here used to invite. The determinism question
those were trying to answer is answered by §8a instead.

One caution the capture itself taught: `kpool`/`vpool` and the FP16 components hash the raw bits, but their `sum` and
`maxabs` fields reinterpret half-precision as float and so read as `nan` or as meaningless magnitudes — only the
hash means anything for those. The FP32 components (`attn`, `bo`, `partl`, `partp`, `suml`, `sump`) carry all three
figures honestly.

Its price, measured rather than assumed: ~48 us per component at the widest shape without the sketch
(`qsa_chain_parity`'s line), which is launch cost; ~138 us per component with the sketch (§8b) on. A sliced layer
pays a few tenths of a millisecond per card either way, so a boot with the flag on is a slightly slower boot.
Nothing is allocated against the tier; the arrays are the pair's own, and a boot with the flag off pays neither the
records nor the read-back.

## 8a. The CPU differential: the instrument that answers the question the boots could not

`tests/cuda/qsa_layer_diff_test.cpp` (`qsa_layer_diff_test`, a `cpu` test, 1.2 s) recomputes one sliced QSA layer's
whole data path on the CPU and runs it three ways — **R** the whole attention and the whole `attn_output` multiply
(what gate-off computes), **S** the pair at `=1`, **P** the pair at `=2` — with integer arithmetic and a replay of
the protocol's own rules: wire sets, staging sets, collective landing regions, and the `ev_in` / `ev_out` / `ev_red`
waits as coded. Integers are the point: with no rounding there is no associativity to disagree about, so a
difference between runs can only mean **a different set of terms was added**, which is the question.

The graded signal is the **per-layer, per-component cosine** against the reference, floor `kCosineOk = 0.9999`. The
floor is justified rather than chosen: re-adding the same terms in a different order costs a float a few ulps, which
lands 1e-7 to 1e-5 off unity at these widths, while a wrong KV row, a stale partial or a swapped head block turns a
tensor and lands far below — so the metric tolerates exactly the noise the GPU chain could not distinguish and still
sees the shifted-distribution class. On the CPU replay everything is deterministic, so **P is also required to equal
S bitwise**, which is the stronger assertion for the purely-ordering question.

**The result, both geometries (the model's real 24×256/2560 and a scaled one walked twelve layers deep): every cell
is exactly 1.00000000 and P equals S element for element.** The slice's math reproduces the whole layer and the
pipelined ordering changes nothing about what is summed — for the protocol as the code writes it.

The suite is not allowed to report that on its own say-so; five checks make it mean something:

- **Three injected math faults** — the partner mapping its heads to the wrong KV head, the two column blocks
  overlapping by a head, the partner selecting with the chunk's first layer's step counts — must each move the
  table. All three are caught, so a unity table is a finding and not blindness.
- **The staging-depth fault** (pipelining on one set), the bug this branch already paid a window for, is caught.
- **Each coded ordering, dropped.** Four of five — the wire-set gate, the `ev_red` gate before a push, the lead's
  push behind its own earlier add, and the `=1` end-of-layer wait — change nothing: the pair's safety rests on a
  transitive chain (the lead's stream order → the collective's flag handshake → the partner's stream order), and
  those four are belt-and-braces on top of it. That is a finding about the protocol, and it is said as "redundant"
  rather than scored as a pass.
- **The control on that claim**: dropping the **flag handshake** itself — letting an add read whatever its landing
  region holds — must and does corrupt the sums. Without this control, "redundant" would be indistinguishable from a
  scheduler that simply waits its way to correctness. It is graded on the **cosine**, not on a buffer comparison:
  the run reports its worst cosine (`0.5585` against the clean `P`) and the suite fails unless that figure falls
  below the floor.

**Two defects the control's own construction exposed, both now fixed.** Neither was visible while the suite was
read as green, and both were found by mutating the instrument and asking whether it noticed.

- **The cosine was decoration.** The header called the cosine the graded signal, but every fault and the control
  were gated on a *different, stronger* metric — exact integer inequality between the run's buffers (`differs()`).
  So `cosine()` could be replaced by a function returning `1.0` unconditionally and the suite still reported
  `0 failures`: the three per-layer tables printed unity because they were told to, and the faults were still
  "caught" by the comparison. Now every fault and the control must drive a cosine **below** `kCosineOk`, which is
  what makes the floor falsifiable. Verified by mutation: with the stubbed `cosine()` the suite now reports
  **5 failures** (was 0) — the instrument sees, or the run says so.
- **The staging-depth fault was caught by the bookkeeping and not by the data.** `run_pair` replays two schedules
  (each rank in turn made the victim) into **shared** output buffers. Schedule 0 completed cleanly and filled
  them; schedule 1 detected the corrupt read and returned early, so the buffers graded against the reference still
  held schedule 0's correct figures — the replay *knew* the pair had overrun a landing region while the tensors it
  reported said nothing had. Each schedule now owns its artifacts, and the case reports a worst cosine of **0.0**
  rather than the 1.0 it used to print behind a "caught" label. The general lesson is the same one as the
  half-pipelined `=1` mode in §3: a verdict computed from one pass and a detection recorded in another is a
  verdict about neither.

**What this settles, and what it does not.** It settles that the split's arithmetic and the `=2` ordering are not
the corruption, on the model of the protocol this file writes down — which is why the agreement rings were silent:
both cards really do sum the same two partials. It cannot convict the engine's kernels, because the arithmetic here
is this file's reading of them (they have their own parity tests), and it cannot convict the span import's bytes,
the peer expert share or the tier squeeze, none of which exist in the replay. Those are what is left, and §7's
component list is the map for them.

## 8b. The GPU cosine differential

Moved to **`progress-tp2slice-cosine.md`** to keep this file inside the 300-line rule. It is the design of the
cross-boot instrument the retraction in §8 makes necessary: a projection sketch on every chain component, so a
per-layer per-component cosine is read off two boots' log lines without a buffer crossing to the host; the
window that runs it, the three comparisons and what each convicts, and the one log line whose SILENCE settles
the cross-chain-aliasing question (kimi review #6 P1).

## 9. Two audits that came out of placing the records

Both are the same class as the `=1` wait defect — one CUDA object carrying two meanings, nothing complaining — and
both were found by asking "what does this event/flag actually mean?" at the moment of writing a record next to it.

- **The span import was borrowing the arm's end-of-layer proof.** `peer_qsa_span_catch_up` recorded `ev_out[0]` on
  the arm's stream and host-synced it to wait on its own DMAs; `ev_out[0]` is the proof the driver's serialized wait
  and the next publish's set-freeing consume. Harmless today (every wait follows its own record), and "today" is
  precisely when a borrowed event costs nothing — the day an import's record and a layer's record exchange places is
  the day a gate silently stops gating. The host now asks the stream it queued on (`prefill.cpp`, the import's
  barrier). `ev_in`, `ev_red`, the driver's wait and the remaining `pipeline >= 2 ? wire : 0` index sites were
  each checked against the rule the `=1` fix established — record side and wait side name the same set — and are
  consistent; this was the one that was not.
- **The index-site count this file filed was wrong, and the recount found nothing wrong.** It said "six remaining"
  sites (kimi's review #6 P2 said eight, one audited, seven not). They are **ten** — `prefill.cpp` 2586, 2611,
  2616, 2665, 2672, 2719, 2762, 2806, 2839 and 4611 — each a `qs.pipeline >= 2 ? wire : 0` over `steps`, `ev_in`,
  `part_f32` or `ev_out`. All ten were read against the rule the `=1` fix established and each event's record and
  wait name the same index: `ev_in` is recorded at 2616 and waited at 2665 under the identical expression, and
  `ev_red`/`ev_out` are recorded at 2905-2906 from the same `w` that 2840 waits on. No off-by-one among them. The
  count being wrong matters less than that it had not been counted: "audited, N sites" is only checkable if N is.
- **`ev_out` has no consumer in the pipelined mode, and that is the design rather than an oversight.** It is
  recorded every layer at 2905; the only `cudaStreamWaitEvent` on it is 4611, inside the `pipeline < 2` branch. So
  under `=2` nothing waits on the partner's end-of-layer proof — which is the mode's whole point, the lead
  charging ahead — and what keeps the pair honest is the `ev_red`/`ev_in` per-set waits taken two layers back,
  audited above. Worth writing down because the CPU replay found four of five coded orderings redundant, and an
  event nobody waits on is exactly the shape that reading could mistake for a missing guard. It is not missing; it
  is the guard the pipelined mode deliberately does not take.
- **Cross-chain aliasing was audited, not assumed.** The two wire sets, the two collective sets and the chain's
  arrays are indexed by `qsa_ord`, the QSA ordinal of ONE chunk of THIS `Prefill`; nothing in that index names a
  chain or a slot, so two chains driving one arm would share the sets — and the agreement check would be silent, by
  the argument above. What the code says today: one `Prefill` per slot (`engine_local.hpp`, the single `sp` beside
  `ver` and `ver_b`, which are decode verifiers, not prefills), and the only thread that enters a run is waited for
  before its caller returns (`generate.cpp`, the oversubscription thread's `mail.fin`). So the class cannot explain
  a corruption that appears on a single cached prompt with no second slot prefilling — and it is now observable
  rather than argued: `run_impl` notices a second entry into the same arm and says so on one log line. A detectable
  class is one a boot can rule out; a boot that died on its own invariant would have taught the hunt nothing.

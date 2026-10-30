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

## 8. The chain: three boots that name the component, for the price of one hypothesis each

`--exp qsa_dump_chain=1` fingerprints every component of a sliced QSA layer as the layer completes — on the stream
that produced it, so it records what the layer used rather than what the buffer holds by the chunk's end — and prints
one line per card per layer. Three boots of one prompt, three chains, one diff. The instrument is the point: it turns
the hunt from one hypothesis per boot (five boots so far, one class eliminated) into one boot per class boundary,
which composes with rule 12 because the budget is the same three boots the differential already costs.

**The boots** — the same server, the same cached `nat` prompt, `max_tokens=64`, greedy, as §1's differential, with
`--exp qsa_dump_chain=1` added to each and the chunk-line figures (`qsa_lead_row_us`, `qsa_peer_row_us`,
`blob_mb_ms`) captured as before:

1. gate off — the reference; its chain exists because the lead-side components are recorded in every mode, and
   `m.bo` is the same buffer sliced or whole, which is what makes the three comparable;
2. `--exp tp_prompt_attn=1 --exp qsa_dump_chain=1`;
3. `--exp tp_prompt_attn=2 --exp qsa_dump_chain=1`.

**The read** — the lines are grep-able by construction:

    grep 'qsa chain:' boot.log | sed 's/^.*qsa chain: //' > chain.<mode>.txt
    diff chain.off.txt chain.2.txt

The first line that differs names the chunk and the model layer; within it, the first field that differs names the
component. `--------` means that mode never recorded it, which is an answer rather than a gap.

**The shape is the diagnosis**, which is why the fields carry a sum and a max-abs beside the hash — a hash alone
says "moved", the pair says by how much:

| what the diff shows | what it means |
|---|---|
| `bo`/`suml` differ from the FIRST sliced QSA layer of the first chunk, and stay wrong | a wrong input reached the layer: read down the component list to the first one that also differs |
| `bo` matches for the first layers and the difference grows with the layer index | an accumulation — something is being added, or lost, per layer |
| `bo` matches for early chunks and diverges mid-context | a race, or state that only diverges once the context is wide enough |
| lead `mix` ≠ peer `pub` at the same layer | the link or the wire set: the partner attended on rows the lead did not publish |
| lead `steps` ≠ peer `stepsr` | the class that made `=2` select with another chunk's positions, still alive |
| `kspan`/`vspan` move between boots while `pub` did not | the IMPORT — the path §1's case 11 never validated in context: cells landing where the attention does not look |
| `partl`/`partp` match the gate-off path but `suml`/`sump` do not | the collective itself |
| everything matches through `suml`/`sump` and only `bo` differs | outside the pair after all — the residency delta is then the only suspect left standing |

Its price, measured rather than assumed: ~48 us per component at the widest shape (`qsa_chain_parity`'s line), which
is launch cost; a sliced layer pays a few tenths of a millisecond per card, so a boot with the flag on is a slightly
slower boot. Nothing is allocated against the tier; the arrays are the pair's own, and a boot with the flag off pays
neither the records nor the read-back.

## 9. Two audits that came out of placing the records

Both are the same class as the `=1` wait defect — one CUDA object carrying two meanings, nothing complaining — and
both were found by asking "what does this event/flag actually mean?" at the moment of writing a record next to it.

- **The span import was borrowing the arm's end-of-layer proof.** `peer_qsa_span_catch_up` recorded `ev_out[0]` on
  the arm's stream and host-synced it to wait on its own DMAs; `ev_out[0]` is the proof the driver's serialized wait
  and the next publish's set-freeing consume. Harmless today (every wait follows its own record), and "today" is
  precisely when a borrowed event costs nothing — the day an import's record and a layer's record exchange places is
  the day a gate silently stops gating. The host now asks the stream it queued on (`prefill.cpp`, the import's
  barrier). `ev_in`, `ev_red`, the driver's wait and the six remaining `pipeline >= 2 ? wire : 0` index sites were
  each checked against the rule the `=1` fix established — record side and wait side name the same set — and are
  consistent; this was the one that was not.
- **Cross-chain aliasing was audited, not assumed.** The two wire sets, the two collective sets and the chain's
  arrays are indexed by `qsa_ord`, the QSA ordinal of ONE chunk of THIS `Prefill`; nothing in that index names a
  chain or a slot, so two chains driving one arm would share the sets — and the agreement check would be silent, by
  the argument above. What the code says today: one `Prefill` per slot (`engine_local.hpp`, the single `sp` beside
  `ver` and `ver_b`, which are decode verifiers, not prefills), and the only thread that enters a run is waited for
  before its caller returns (`generate.cpp`, the oversubscription thread's `mail.fin`). So the class cannot explain
  a corruption that appears on a single cached prompt with no second slot prefilling — and it is now observable
  rather than argued: `run_impl` notices a second entry into the same arm and says so on one log line. A detectable
  class is one a boot can rule out; a boot that died on its own invariant would have taught the hunt nothing.

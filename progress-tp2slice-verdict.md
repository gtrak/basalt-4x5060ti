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

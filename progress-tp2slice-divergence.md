# TP=2 prompt-attention slice: where the `=2` chain actually diverges

Read of the three captured chain boots (`/tmp/chain.off.txt`, `/tmp/chain.1.txt`, `/tmp/chain.2.txt`,
taken 18:54-19:02 on the day the instrument landed) against the engine log
(`~/.cache/basalt/5059bef732e5da6c/engine.log`) and the code. It moves the hunt off `=2` and off the event
index, onto the lead's half-width `o_proj` product.

## 0. The boots are complete, and they are the boots we think they are

The "`=2` truncated" reading was an artefact of counting, not a lost boot:

- each boot prints its header twice, once per pipeline stage (`qsa chain: on, 22 components per layer over
  48 layers` at log lines 410227/410235, 411884/411898, 413847/413861), and
- each boot runs the prompt twice, so one boot is two passes of 15 chunks x 12 QSA layers.

Per boot: 360 lead records, 360 peer records when the slice is armed. The files are cumulative greps, which
is why `chain.2.txt` holds six passes: off, off, =1, =1, =2, =2. The mode of each boot is confirmed by its
own boot line in the engine log, not inferred: `serialized, 1 o_proj staging set, 1 collective staging set`
(411858) for the `=1` boot, `pipelined, 2 o_proj staging sets, 2 collective staging sets` (413829) for the
`=2` boot. The `=0` boot's mode line is absent because the arm was never opened. No `qsa chain: layer ...`
launch failure was logged in any boot, so no slot is missing for a silent reason.

The boots predate `783acd17`, so they carry hash/sum/maxabs only - no projection sketch, no cosine. Every
figure below is a hash, a sum or a max-abs.

## 1. The instrument's own anchor was reading out of bounds - fixed

`qsa_chain_put(..., kChainBo, m.bo, T, ZV, m.cs)` walked `m.bo`'s rows at the ATTENTION's width while
`o_proj` writes it `n_embd` wide (`native_proj` sizes it by `wo->ne1`, the carve by `T * N`): the sampled
rows ran ~2.4x past the end of the buffer, so `bo` - the component the header calls "the figure all three
modes must agree on" - was reporting whatever the arena held beyond `bo`, on every boot. The gate-off boots
showed a finite `bo` (`33c56c8e/-51630.293/9.1217852` at chunk 0 layer 3) precisely because out-of-bounds
arena bytes are usually finite floats.

Fixed in `0efe4931` (`ZV` -> `g.n_embd`). The other slots are right: `partl`/`suml` use `N`, `attn` uses
`ZV` and `gated` `ZV/2`, and those two buffers really are the attention's width. So from this commit on,
`bo` is a usable figure; in the three captured boots it is not, and nothing below rests on it.

## 2. The first divergence: the lead's `o_proj` partial, at the first sliced layer of every chunk

Per sliced pass (180 lead records = 15 chunks x 12 layers; identical figures for `=1` and `=2`):

| slot | records with a nan | finite where |
| --- | --- | --- |
| lead `mix`, `attn`, `gated` | 165 / 180 | layer 3 of every chunk |
| lead `sel` | 11 / 180 | - |
| lead `kpool`, `vpool` | 30 / 180 | (gate off: 180 / 180 - the pool slots sample `cap` rows, most of which no cell reaches) |
| lead `partl`, `suml` | 180 / 180 | nowhere |
| peer `pub`, `q`, `attnp`, `gatedp`, `partp` | 165 / 180 | layer 3 of every chunk |
| peer `sump` | 180 / 180 | nowhere |

Chunk 0, layer 3 - the first sliced layer of the first chunk - side by side:

```
                      mix                        attn                gated              partl              suml
gate off   c57f6a7f/-59.27/26.46      c7f7efa5/3480/3.74     79ca2b85/-0.0046/0.0061      -                -
=1         61f2767a/-57.77/26.33      e329c7ff/2021/2.66     f7c3a466/-4113/5.80    26e6e010/nan/nan  0fbf8769/nan/nan
=2         af36ffdc/-58.69/26.33      9bac4a3b/2021/2.66     39fb3147/-4111/5.92    34187c09/nan/nan  522ea724/nan/nan

                      pub                        attnp              gatedp             partp              sump
=1         61f2767a/-57.77/26.33      4088b66f/321/2.36      f7a63e50/-0.016/0.0035  37a00094/92.8/0.81  0fbf8769/nan/nan
=2         af36ffdc/-58.69/26.33      17520159/1585/1.88     c62ab52f/-0.013/0.015   90a4b6ac/-39.4/0.58 522ea724/nan/nan
```

and within each boot, over all 180 records of each pass: `lead.mix == peer.pub` 180/180,
`lead.steps == peer.stepsr` 180/180, `lead.suml == peer.sump` 180/180.

What that says, in order:

- **The pair's inputs and its sums agree.** The published rows arrive byte-identical, the step counts
  arrive byte-identical, and both cards finish the collective with the same buffer. A stale wire set, a
  half reduce and a neighbouring layer's partial are all excluded in both gates - not because the design
  rules them out but because the fingerprints that would show them do not.
- **Everything upstream of the lead's `o_proj` is finite at that layer**: `mix`, `attn` and `gated` are
  finite, and `gated` samples the whole row, so the columns the product multiplies are finite. The
  partner's `partp` - same call, same shape, other card - is finite.
- **The first component that is not is the lead's own `o_proj` partial.** `partl` is taken on the lead's
  stream, with the correct stride, between the two product launches and the two adds, so it is the lead's
  product's output and nothing else. It is nan at that layer in both gates, in both passes, in every chunk.
- **The shape is a wrong input, not an accumulation and not a race**: constant from the first sliced layer
  of each chunk (165 = 180 - 15: exactly one finite layer per chunk, layer 3), never appearing mid-context,
  never growing with the layer index. The residual inside a chunk then goes nan from layer 7 on, and the
  next chunk starts from a fresh residual - the chunks share the recurrent and KV state, which is the only
  reading of a finite `mix` at every chunk's layer 3 - so every chunk repeats the same first-poisoned-layer
  pattern. A logit vector out of a nan'd tail is argmax-stable, which is the shape of 256 identical tokens at
  self-consistency 100%.

The chain's own header states the blind spot this walks into: the pair-agreement check "is silent when both
cards hold the SAME wrong value". `suml == sump` 180/180 is exactly that - both cards adding the lead's nan
and agreeing on it.

## 3. `=1` is corrupted exactly like `=2`

The `=1` boot of this instrument run is not a healthy baseline: same slots, same layer, same chunk, same
shape. So the gate is not the discriminator, and the question "why does `=2` corrupt" is the wrong question
- the shared sliced path corrupts, and `=0` is the only mode that does not run it.

## 4. The `9512e0d2` audit: the fix is right, and it is not the mechanism

Asked: did the fix change the RECORD index at `=1`, and did the WAIT follow it?

- **The record side never changed.** `peer_qsa_finish` records the arm's end-of-layer proof at
  `ev_out[w]`, `w = pipeline >= 2 ? wire : 0` - index 0 at `=1`, always. Before and after the commit.
- **Only the wait moved**: `ev_out[wire]` -> `ev_out[pipeline >= 2 ? wire : 0]` (`9512e0d2`, nine lines,
  one hunk). At `=1` that is `ev_out[wire]` -> `ev_out[0]`, i.e. the wait moved onto the event the record
  side actually writes.
- **The correct `=1` semantics are record[0] / wait[0] with the end-of-layer wait**, which is what the code
  does now. One staging set means one proof event per role. The wire sets stay two and keep alternating
  (`kWireSets`, always funded): with the end-of-layer wait the partner is never more than one layer behind,
  so a set is never overwritten while it is being read. There is no wire alternation in the `=1` proof
  events, and no case for it: a wire-alternating proof with one staging set would make the mode's own
  definition ("the lead waits for the partner's whole layer") depend on the layer's parity.
- All six sites of the three event pairs now spell the index the same way: `ev_in` record 2616 / wait 2665,
  `ev_red` wait 2840 / record 2906, `ev_out` record 2905 / wait 4611 - each as
  `qs.pipeline >= 2 ? wire : 0` or the local `w` derived from it. The host barrier for the span import is
  `cudaStreamSynchronize(qs.s)` (3007 records why it stopped borrowing `ev_out[0]`), so no event carries two
  meanings.
- **Therefore the fix did not tangle the `=1` protocol; it enforced it.** And it cannot be the mechanism for
  this corruption: at chunk 0, layer 3 of the first pass, nothing is in flight to alias with - every wire
  set, every staging set and every event is being used for the first time, and the poisoned value is already
  there, identically in both gates. An event-index bug cannot produce a nan in a product that has no
  predecessor.

What the commit does change is the *meaning of every `=1` measurement*: earlier `=1` boots were half
pipelined, so the correlation "`!` first appears at the fixed `=1` boot" is the first boot where `=1` really
serialized, not the first boot where a wait was misindexed. It is also the first `=1` boot that ran after
`ed7d4d33` halved the lead's attention, so what changed for `=1` is both things at once.

## 5. Mechanism, narrowed

The lead's product is

```cpp
lead_gemm.native(lead_gated /* m.attn_h */, sp.o_type, lc.native, lead_partial /* m.bo */,
                 T, N, lc.ne0, /*ldy*/ 0, /*beta*/ 0.0f, /*ldx*/ lead_ld /* ZV */);
```

and its inputs are all checked against the geometry before it is reached (`lc.ne0 + pc.ne0 == n_head * D`,
`lead_ld >= lc.ne0`, `sp.heads.q0() * D == lc.ne0` - which forces the partner to own the HIGH heads and the
lead columns `[0, lc.ne0)`, exactly the columns `gate_attn_span` writes). The chain proves the X side finite
at the poisoned layer, and the pitch is `m.attn_h`'s own carve width (`T * ZV` halfs). That leaves the weight
byte stream - `qs.lead_cols.o_cols[layer].native`, dequantized with `sp.o_type` - as the only input not
otherwise accounted for, and it is the only one of the three that differs between the two cards.

It is not random corruption of one buffer: the lead's packed `attn_output` columns are twelve buffers (six
per stage, 36 MiB each stage, on both lead cards: "CUDA0/CUDA2 holds rank 0's attn_output columns, 36 MiB
over layers [...], 6 buffers"), and every one of them decodes to nan at every sliced layer, while the
partner's twelve are all finite. And the packer cannot leave holes:
`qsa_prepare_one` fills a zero-initialized host vector, checks the plane sizes add to what was allocated,
and copies it up whole under a checked `cudaMemcpy`, symmetrically for both cards from one weight table. So
the difference is not in the cut's bookkeeping.

The cheapest thing that separates "bad bytes under `lc.native`" from "something in the lead's `Gemm`": run
one `=1` boot with the lead's product pointed at the PARTNER's columns -
`qs.peer_cols.o_cols[layer].native` instead of `lc.native` - and nothing else changed. `partl` finite says
the lead's packed buffer is the bad stream; `partl` still nan says the byte stream is innocent and the
suspect is the lead's `Gemm` (its dequant scratch is the model's shared one) or the pitch, and the next cut
is to fingerprint `lc.native` itself.

## 6. Notes for whoever runs it

- `gated`'s sum is not comparable between `=0` and the sliced modes: in the sliced modes the lead writes
  only its own `lc.ne0` columns of `m.attn_h` and the slot samples the whole row, so the partner's
  never-written half shows up in the figure (`-0.0046/0.0061` gate off against `-4111/5.92` sliced). It
  still proves what it needs to here: no nan anywhere in the sampled row, so the columns the product reads
  are finite. Narrowing the slot to `lc.ne0/2` words would make it comparable and stop it comparing
  uninitialized bytes; left alone because the chain's rule is one shape across boots.
- Both prompt passes of a boot give identical figures, and gate-off passes differ from each other in every
  hash (reduction order is not deterministic across launches), so cross-boot comparisons are evidence only
  where a value goes from finite to nan - which is how this section reads them.
- CPU gates at `0efe4931`: 69/70 with the count at 70 (`expert_multi_test` pre-existing),
  `peer_qsa_span_test` ok, `tp2_*` parity ok, platform boundary and stderr logging ok. No numerics change:
  the fixed stride is a fingerprint argument under `--exp qsa_dump_chain`.

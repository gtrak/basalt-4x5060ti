# The RAM band's share, and `--exp peer_band_pin`

A prefill pair splits a stage's experts three ways: the partner card's tier holds some, the lead card's tier holds
some, and the rest sit page-locked in host RAM with no place on either card - the **RAM band**. Either card can read
a band expert's blob over its own link, so the pair may decide that the PARTNER reads some of them and computes
their rows. `--peer-ram-rows` funds the partner's ring of blob slots; how many of the band it is given is the
**share**.

The share is in eighths. `num` of every `kPeerBandDen` (= 8) of a layer's band members go to the partner, counted by
rank in id order: `peer_band_taken(k, num)` is `k % 8 < num`. Its closed form, `peer_band_take(band, num)`, is the
count the price charges for it. Both live in `include/basalt/prefill/peer_band.hpp`, which the engine calls and
`tests/prefill/peer_band_parity.cpp` (`ctest -L cpu`) grades - the price (`peer_ram_share`), the link bill
(`peer_link`), the lead's streamed walk and the row split each count the same band with a counter of its own, and
those counters must land on the same experts.

## What the share is struck on, and why two boots differ

By default the share is a decision, not a setting. `peer_ram_share` (`src/prefill/prefill.cpp`) solves for the share
where the lead's predicted layer time comes down to the partner's (`peer_layer_time`; `--exp qsa_share_bytes=1`
strikes the two links' bytes instead), rounds it to eighths, and vetoes a layer whose rows outweigh its blob and a
chunk longer than `PeerRamShare::top`. Two boots of one build, one command line and one pack can then hand over
different numbers, because the band's **membership** is a placement question: `in_ram_band` asks whether the LEAD's
tier holds the expert and whether its host copy is page-locked, and both follow how full the lead's tier came out.
The fill is set by the memory that is free, not by a budget, so a shift of tens of MiB moves a few dozen experts
across the band's border.

That shift is not local, because the choice is by RANK. Take one member out of the band and every rank above it
drops by one, so `k % 8 < num` selects a different set from that expert's id upward. One run's reference arm came
home disagreeing with itself at 99.13% top-1 with band counts of 591 against 703, which is that shift and nothing
else.

## The pin

`--exp peer_band_pin=N` (default 0, off) takes the share out of the model's hands:

| | unpinned | pinned to `N` |
|---|---|---|
| membership (`in_ram_band`) | bytes in no card's tier, and page-locked in host RAM | every expert the PARTNER's tier does not hold |
| the share | solved from the two cards' predicted times, rounded to eighths | `N`, whatever the model says |
| the count (`PeerRamShare::take`) | the sum of the solved per-layer `r` | `peer_band_take(band, N)`, exactly what the rank walk picks |
| the vetoes | rows outweighing the blob; a chunk longer than `top` | none |

Membership then depends on the partner's placement and the expert's id alone. The partner's tier is already held
equal across the arms by `--exp peer_charge_partial=1`
([moe-partial-merge.md](moe-partial-merge.md), the room the tier fills into), so two pinned boots hand over the same
experts and their row sums agree.

`N` is a count of eighths: `0` to `PeerRamShare::kDen` (8). `--exp peer_band_pin=0` is the band off - with no share
there is nothing to pin, so there is no separate `peer_band_off`; any other value is refused at boot rather than
rounded, so a sweep that misspoke itself cannot be read as a neighbouring share. The boot's `RAM rows` fact ends by
naming the pin, and asks for it by name when the band could not open at all.

## What a pinned boot is not

The canonical membership is **not** the placement this pair would have chosen. An expert the LEAD's tier holds can
be selected by the pin, and the walk hands it to the partner anyway: its blob goes over from host memory, its rows
come back, and the copy that already sat in the lead's VRAM idles for that chunk. The pin also steps over the two
vetoes that keep the pair's cards balanced. The wall a pinned boot prints is therefore not a throughput figure, and
the split its `peer layer time` line prints describes a placement the engine would never choose on its own.

So the pin grades the fold's **arithmetic** under a canonical placement, and nothing else. A `PASS` from a pinned
quality gate is evidence that the tested flag does not move the head's distribution; it is no evidence about
production's placement, its wall, or its link. **It is a gate instrument, and it never belongs in the serving
recipe.** Unset - the default, and every shipping boot - adds no counter, allocates nothing differently, prints no
line, and the walk and the row split run the code they ran before the flag existed.

It is also a numerics change, which is the whole reason the gate needs it: a row the partner computes crosses the
link as BF16 bits and enters the combine with 8 mantissa bits, while the same row computed on the lead stays FP32
([moe-partial-merge.md](moe-partial-merge.md)). The pin moves rows from the second category into the first, so the
arms differ by the fold's reduction order and by nothing else - and the share it pins should be the one an unpinned
boot struck for itself, read off that boot's `RAM rows` fact, or the gate is grading an arithmetic no production
boot runs.

Three sites carry it, all of them reading the one `in_ram_band`: the streamed walk
(`src/prefill/prefill.cpp`) tests band membership AHEAD of the residency skip, so a resident member goes to the
partner rather than being claimed by the lead; the row split routes a band expert to the partner whether or not the
lead's tier holds it, so its rows follow its blob instead of being computed twice; and a layer whose experts a
pinned band has taken runs the MMQ path, which is the one that honours the row split - the fused path walks every
expert of the layer and would compute the partner's rows here as well.

## Telling two boots apart: `--exp chunk_dump=1`

A pinned band makes the band's MEMBERSHIP canonical; it does not make the rest of what a chunk decides visible.
`--exp chunk_dump=1` prints one line per layer of a prompt chunk, naming the expert kernel the layer ran on
(`fp16`, `mmq`, `fp4`, `fused`, `fused-nat`), the gates that picked it (`ring`, `stream_all`, `peer_fp4`, the band's
`ram` count and `canon`, `T_fp4`/`T_max`, `bf16`, `merge`, `p2p`, `native`, `fused`), who served the layer's rows
(`lead`/`peer`/`band`/`cpu` expert and row counts, `stream`, and `holds` - what each card's tier held that layer),
and two fingerprints: `ids` over the routing as the host read it, `split` over which card each expert's rows went
to. Boot the same command twice, diff the lines, and the first layer whose line differs names the decision that
parted the runs - `ids` differing means a router's input had already parted upstream of that layer, `split`
differing while `ids` agrees means the row division moved, which is a numerics change because a partner's rows
cross the link as BF16 bits and this card's stay FP32. A layer on the fused path prints no routing fingerprint (the
path keeps the routing on the card, which is its point) and no division, because it makes none.

Nothing in the dump reads the device: a fingerprint that needed one would add a sync to the window whose scheduling
is being examined, and hide what it was built to find. The classification and the fingerprints are
`include/basalt/prefill/chunk_dump.hpp`, which the engine calls and `tests/prefill/chunk_dump_parity.cpp`
(`ctest -L cpu`) grades. Unset - the default - allocates nothing, prints nothing, and the layer loop runs the code
it ran before the flag existed.

## When the router is the parting: `--exp router_dump=N`

`chunk_dump` says WHICH LAYER parted. When that layer is the router's own decision, it cannot say on WHAT: the
`ids` word is a fingerprint of the routing, so three different facts all print as one changed word - the
projection's logits differed, the logits matched while the selection did not, or the token's own top-K boundary was
a tie that the tie-break alone decided. Those bytes never leave the card, so answering needs a probe that does the
one thing the dump above refuses to do: read the device.

`--exp router_dump=N` copies four blocks of one model layer of one prompt chunk to the host - the BF16 activations
the layer's router projection read (`in`), the FP32 logits it produced (`lg`), the expert ids it chose (`ids`) and
their weights (`w`) - and prints one summary line of whole-block fingerprints plus one line per token. The copies
are queued on the compute stream behind the router and read back at the sync the streamed walk already makes for
the routing, so the probe adds no rendezvous of its own; they are issued ahead of the routing guess, which
overwrites the same logits buffer with the NEXT layer's router, so what comes back is what this layer's `route`
read and wrote. The probe then replays the top-K over the read-back logits with the kernel's own rule (descending
logit, lower id on a tie) and reports the K/K+1 boundary:

| field | what it settles |
|---|---|
| `lg=` differs between boots | the logits differed, so the projection or its input moved - check `in=` next |
| `lg=` same, `in=` differs | the router's INPUT had already parted upstream of the projection |
| `lg=` and `in=` both same, `ids=` differs | one set of numbers gave two routings: the selection, not the arithmetic |
| `ties>0`, or `gap=` at its tightest | a token whose K-th and (K+1)-th largest logits are the SAME float: which expert holds the seat is decided by the tie-break and nothing else |
| `disagree=` | the first token whose ids differ from this record's own replay of the selection - a disagreement **in one boot**, with no second boot to compare against |

`disagree` is a lead and not a verdict: the kernel ranks the softmax probabilities and the replay ranks the logits,
and `expf` is monotone without being strictly monotone in FP32, so two logits a few bits apart can give one
probability and a genuine tie the replay cannot see. That is the boundary shape the field exists to notice.

Three keys, because the probe is confined deliberately: `router_dump=N` names the MODEL layer (`-1`/absent is off,
which is what lets layer 0 be named - the first layer is the control on every claim the probe makes),
`router_dump_p0=V` names the chunk, and `router_dump_all=1` probes every layer of the chunk, which is the sweep
that asks where it FIRST parts and pays a sync per layer. A layer that ran the fused path brings its own silence
instead of nothing: the path keeps the routing on the card, so there is no readback sync to ride, and a dump that
printed nothing there would be read as "the routers agreed".

The arithmetic is `include/basalt/prefill/router_dump.hpp`, which the engine calls and
`tests/prefill/router_dump_parity.cpp` (`ctest -L cpu`) grades against a sort, over random rows with deliberate
ties and non-finite entries. Unset - the default - allocates nothing, copies nothing, prints nothing, and the layer
loop runs the code it ran before the flag existed.


# The partial merge (`--exp peer_partial_merge`)

A two-card prefill pair splits the routed experts: the lead keeps some, the partner card computes the rows of the
rest and sends them back, and the lead's combine weights and sums every row. What crosses the pair's link is rows —
`K` rows of `n_embd` per token the partner holds rows for. The **partial merge** moves the sum to the card that
holds the rows: the partner folds its weighted rows into one `T x n_embd` vector a layer and sends that, and the
lead adds the one vector into its combine.

Engine-wide this is the same sum with a different reduction order, so it belongs here. It is a prefill-only mode
(`src/prefill/`), on the FP4 expert path with a partner card, and it is off unless
`--exp peer_partial_merge=1` is given. With the flag off no buffer is allocated, no weight is read back, and the
combine is the same code the flag-on boot takes when a chunk has no peer rows; `tests/kernels/moe_part_parity.cpp`
asserts the two are bitwise equal.

## The two chains

One card, all K routings (the reference every other page here describes). Per `(token, d)`, `slot[t*K + k]` the row
its k-th routing produced, `w` the router weight, then the shared expert's term last:

```
s = 0
for k in 0..K-1:  s = fmaf(w[t*K + k], rows[slot[t*K + k] * n_embd + d], s)   // fmaf, one rounding per row
out = s + shared[t * n_embd + d] * sigmoid(gate[t])
```

Partial merge, the same `(token, d)`. The lead's combine skips the partner's block `[rows_lo, rows_hi)` and adds the
partner's partial in its place; the partner's fold adds only its own rows:

```
lead:  s = 0
       for k: if slot[t*K+k] not in [rows_lo, rows_hi): s = fmaf(w, rows[slot[t*K+k] * n_embd + d], s)
       out = (s + part[t * n_embd + d]) + shared[t * n_embd + d] * sigmoid(gate[t])

peer:  part[t * n_embd + d] = 0                       // zeroed once a layer
       for each expert group, in launch order:         // groups are the FP4 pieces, cut to the staging
           for each of its rows: part[tok[r] * n_embd + d] =
               fmaf(w_row[r], rows_of_that_card[r * n_embd + d], part[tok[r] * n_embd + d])
```

Three properties make this a reduction-order change and nothing else:

- **The lead's order is unchanged.** Its own rows still go in k order with one `fmaf` each, and the partial joins
  before the shared expert's term so nothing else can reassociate between them (`moe_combine_out`).
- **The partner's order is row-major, not k-order.** Within a token's rows it sorts by k, so a token's run is added
  in the same order the single card would use; but a token whose rows fall in several expert groups is chained
  group by group, in the order the groups were launched. `tests/kernels/moe_part_parity.cpp` (check 4) shows this
  costs nothing by itself: grouping the same rows and adding them group by group is bitwise one chain over those
  rows. What it cannot be is the *single* chain, because the join of the two cards is one add.
- **`w_row` comes from the lead's router.** The partner has no router; the lead copies its `m.w` to the host with
  the row tables and the fold weights travel with the group's other tables. A token's rows therefore carry exactly
  the weights the single-card combine would have used.

All of it lives in `include/basalt/prefill/moe_part.hpp` (`moe_term`, `moe_routed_sum`, `moe_fold_rows`,
`moe_part_pack`, `moe_combine_out`), which the device kernels and the CPU test both call: no path writes the chain
again by hand.

## The wire's one extra rounding

Rows already cross as BF16 bits by default (`docs/precision/index.md`, the FP4 prefill row): the down product's
epilogue rounds them once, and the combine reads each from the BF16 plane. The partial merge keeps that format and
adds **one rounding of the finished partial** (`moe_part_pack`, its own kernel after the group loop so the fold's
intermediate cells stay fp32). `--peer-rows-fp32` keeps the rows FP32 and sends the partial FP32, which removes both
the row rounding and the partial rounding — the A/B, and the mode the cost model prices the wire at when asked.

So the whole numerical cost is: reassociation of the K-term chain across two cards, plus one BF16 rounding of the
summed run. bf16 keeps eight significant bits, so that rounding is at worst half a `2^-7` ulp of the partial.

## What the CPU test measures

`tests/kernels/moe_part_parity.cpp` (CPU-only, `ctest -L cpu`) builds synthetic layers at the model's `K = 10` with
a shuffled routing and a two-thirds partner share, and compares the split combine against the flag-off combine over
every `(token, d)`. The first six checks cut the peer's rows into fixed groups of 7; the last hands the same rows to
the engine's own cutter:

| Check | Measured |
|---|---|
| no partner rows: split vs single-card | 0 of 3,880 outputs differ (bitwise) |
| FP32 wire (reassociation alone) | max abs difference 1.2e-7, i.e. one fp32 ulp of the terms, 5% of the bound |
| BF16 wire (plus the partial's rounding) | max abs difference 1.9e-3, 0.93-0.98 of its bound, 3.6-3.8e-3 of the partial |
| fold grouped per expert group vs one chain | 0 of 3,880 partial cells differ |
| fold's per-token list vs the routing table | max abs difference 0 (same chain, same order) |
| a packed partial unpacked again | 0 of 65,282 bf16 values changed |
| term set under the engine's own group cut, 45 geometries | 0 mismatched cases, 276 groups |

The last row is the one that grades the fold rather than a transcription of it. The peer's groups come from
`fp4::cut_pieces` (`include/basalt/prefill/moe_fp4.hpp`, host code kept in the header for exactly this reason), over
expert runs that mix cold single-row experts, long hot runs the staging cap splits at whole tiles, and experts
holding no rows at all; the row space is `lead | peer | CPU tier`, so the combine reads an FP32 tail above the
peer's block that the fold never sees. Per geometry it asserts the fold's multiset of (token, k) equals the lead's
skipped multiset, that every folded row carries its own router weight, and that the pieces tile the block in order -
a row read at another row's weight passes a row count and is still a wrong sum. Five mutations of the fold's lists
each turn it FAILED.

The bound the BF16 row is graded against is half a bf16 ulp **of the partial** (`2^-8` relative) plus 32 fp32 ulps of
the terms' magnitudes: the partial's rounding is relative to the partial, not to the token's whole routed sum, which
is why moving the sum to the partner card does not put a second BF16 rounding under the whole MoE output.

## The share model

`peer_layer_time` (`src/prefill/prefill.cpp`) is what decides how many experts cross, and under the flag it stops
charging per row on the return leg. Per layer, with `c` the chunk's tokens, `rows` the partner's rows, `N` = 2,560:

| Term | Rows returned | Partial returned |
|---|---|---|
| out: token planes | `c * 1444` B (FP4) | unchanged |
| back | `rows * N * 2` B (BF16 rows) | `c * N * 2` B, plus `rows * (20 + 4)` B of fold tables |
| peer compute | `rows * qsa_peer_row_us` | unchanged |
| peer fold | — | `rows * qsa_peer_fold_us` (default `N * 10 / 400000` us, 10 FMAs a row) |

The back leg becomes a function of the chunk, not of the share, so the share can move further before the partner
becomes the critical side. `--exp qsa_peer_fold_us=0` removes the fold term (the A/B for whether the fold is what
limits the share); `qsa_lead_row_us`, `qsa_peer_row_us` and `qsa_link_mb_ms` re-price the rest as before. The start
line prints `partial T x 2560 (...)` instead of `rows ...` in its `fp4 peer share` entry, and the peer's rows in the
`peer layer time` line's parens now name the fold.

## How the gate measures it

The gate is `tools/quality_gate.py` over the committed scored set `tests/data/quality/partial_merge.jsonl` (built and
validated by `tools/quality_fixture.py`). Everything the two arms score is committed as token ids, so a tokenizer
change cannot move the scored set: each chunk is exactly `n_ctx` tokens, its first `n_ctx/2 + 1` are the prompt and
the rest are teacher-forced through verify windows of `--t` tokens. Nothing is generated, so the arms cannot diverge
into two different sequences - which is what makes them comparable at all.

One base file, three runs against it:

| arm | what the engine does | what it is for |
|---|---|---|
| `emit` | writes its own head log-probs into the base's logit rows (`--ppl-emit`) | the reference: the flag-off arm as its own base |
| `check` | reads that base back (`--ppl-base`), flag still off | proves the instrument: the emitter and the reader must agree |
| `gate` | reads the same base, with `--exp peer_partial_merge=1` | the verdict |

The base is a llama.cpp `_logits_` file laid out by the tool (header, every chunk's tokens, the logit rows left as a
hole) and filled by the engine, so the reference is read back through the untouched KL-divergence reader that
[quality.md](quality.md) already quotes - one measurement surface, not two. A chunk the engine never scored reads
back with every log-prob at 0.0, which makes KLD enormous rather than plausible: a missing row fails loudly. A row's
quantization is `PplGate::quantize_row`, shared by the writer and the reader and pinned by
`tests/program/ppl_emit_parity_test.cpp`.

### The three runs

`$PACK` is `models/Qwen3.8-FN-FP4Local.basalt`, `$OUT` a scratch directory. The engine flags are the production
server's, so the gate measures the shipping configuration; the two arms differ by exactly one word **once they hold
the same experts** - which is what the next section makes them do.

```sh
COMMON="--model $PACK --prefill 8192 --resident-budget-gib 37 --spec 8 --ple-io direct --kv int8
        --max-context 262144 --resident-experts --peer-reserve-mib 600 --short-read 512
        --tp-size 2 --peer-ram-rows --gpu 0,1,2,3 --batch 2"

python3 tools/quality_gate.py --pack $PACK --bucket 512 --out $OUT --arm emit  $COMMON \
    --base-args '--exp peer_charge_partial=1'
python3 tools/quality_gate.py --pack $PACK --bucket 512 --out $OUT --arm check $COMMON \
    --base-args '--exp peer_charge_partial=1'
python3 tools/quality_gate.py --pack $PACK --bucket 512 --out $OUT --arm gate  $COMMON \
    --flag peer_partial_merge=1
python3 tools/quality_gate.py --verdict $OUT
```

Flags the tool does not know go to the engine as they are, the way `tools/ppl_gate.py` passes them; an engine flag
given here replaces the tool's default for that flag. The tool boots its own engine and drives it over its stdin
(`ppl=K ppl_t=T` is an engine protocol key, not part of the server's HTTP API); it never connects to a running
server. `--adapt-swaps 0 --peer-adapt-swaps 0` are forced on both arms so the adaptive tier cannot differ between
them, and `--max-context` must clear the chunk plus its window. Run both length buckets (512 and 1024 tokens, 12
and 6 chunks; the emitted base is 1.42 GiB each).

### The arms must hold the same experts, and the verdict checks that they do

Which card computes a row decides whether it crosses as BF16 bits or stays FP32, so an arm that places the experts
differently is not measuring the flag. This flag does move placement, and the place it moves it is the room the
partner card fills its expert tier into. `open_partner` (`src/program/generate.cpp`) sizes that room as

```
room = free(partner) - ( vram_reserve_mib + split_pf_mib + kWindowMib + (pipe ? kPairMibLater : 0)
                         + ceil(Prefill::peer_bytes(chunk, ...) / 1 MiB) )
```

and `peer_bytes` is one counting walk of `peer_layout`, the walk that allocates the peer's prompt buffers. So the
room moves exactly as far as the buffer set that walk prices - and the partial merge's price is one entry in it:

| the flag's own buffers | bytes | at this chunk |
|---|---|---|
| `part`, the FP32 partial | `T_p * n_embd * 4` | 80 MiB (`T_p` = 8192) |
| `part16`, packed for the BF16 wire | `T_p * n_embd * 2` | 40 MiB |
| `fold_tok`, `fold_off`, `fold_rel`, `fold_w` | `4 * G * 4`, `G = T_max` | 0.1 MiB (`G` = 6143) |
| **the tier gives up** | | **121 MiB** |

`T_p = max(T_max, t_fp4)` is the same in both arms here: `peer_fp4_ceiling` is 26,891 tokens with the row wire and
213,072 with the fold, both far above the 8192-token chunk that caps it. 121 MiB is what one run measured as 13.98
GiB against 13.86, and at 2.77 MiB a ranked pair it is the 5429 against 5384 that run reported. Nothing is
REMOVED by the flag: the peer still computes its rows into the double-buffered `Dm_b` staging the fold reads, so
the row wire's buffers are charged in both arms and the charge is legitimate, not stale. The charge pins the arms
equal only while the two `T_p`s come out equal, which is what the ceiling being far above the chunk buys; a pack
whose ceiling binds below the chunk would price the two arms at different `T_p`s, and `--verdict` prints that room
gap rather than passing it.

Two knobs do not move a tp2 partner's tier, and one re-gate was spent on the wrong one. `--peer-reserve-mib`
prices an EXPERT TIER (`peer_expert_room`, `open_peer`, both gated on `o.peer_device >= 1`) and a `--tp-size 2`
boot has no such card, so raising it 600 to 723 filled the tier identically. `--vram-reserve-mib` is in the
formula but shared: raising it shrinks the LEAD cards' tiers beside the partner's and moves the placement again in
the other direction. The lever is the buffer set itself: **`--exp peer_charge_partial=1`** makes `peer_bytes` price
the partial's buffers on an arm that does not fold, so the reference arms reach the gate arm's room through the
same arithmetic the gate arm pays - no MiB figure to re-derive when the chunk changes. The boot prints what the
charge holds back beside the whole figure it priced for the partner's prompt buffers, and `make_peer` still follows
`peer_partial_merge` alone, so nothing folds and no weight is read back.

`--verdict` therefore reads each boot's own log - the `experts live in VRAM ...` line, the two
`(N experts held, M band)` pairings, the `tp2: CUDA%d holds ... ranked pairs ... room` lines, the RAM-band split -
and prints `VOID` when the `gate` arm or either reference arm disagrees with `check`, naming each disagreement fact.
It also voids the run when an arm left no `engine.log` or a log with none of those lines in it: an arm whose tier
cannot be read has not been shown to hold the same experts as any other. It measures the room gap and prints which
arm to charge, by how much, and the room to reach. A flag that costs the partner card bytes is paid for on the
reference arms with `--base-args`, which is withheld from the gate arm; `--gate-args` is its mirror for a flag that
frees room. `--seed` does not pin placement across different reserves.

## The bounds

Not measured yet - **DERIVED-PRIOR**, to be calibrated by the first real run. The reasoning, and why each number is
where it is:

| bound | value | why |
|---|---|---|
| self-check KLD mean | `<= 1e-3` | the emitter's own floor is 7.3e-05 (see below); an order of magnitude above it means the writer and the reader have drifted, and then nothing the `gate` arm says means anything. **Violating this voids the run**, it is not a FAIL |
| self-check same-top-1 | `== 100%` | the same argument: a row whose top-1 disagrees with itself is a row the instrument cannot see |
| KLD mean | `<= 2e-3` | the change is a K-term reassociation plus one BF16 rounding of the partial - at worst half a `2^-7` ulp of the partial, not of the MoE output. ~27x the emitter's floor, and ~10% of the 0.0233 that separates two independent implementations of this model, so a genuine error of this class cannot hide under it while a real bug still trips it |
| KLD p99 | `<= 2e-2` | the tail is where a reassociation error shows first: a token whose rows land badly across two cards moves its whole row, not the mean of the set |
| same-top-1 | `>= 99%` | argmax agreement; near-ties may legitimately flip, and the CPU test's own worst BF16 case sits at 0.93-0.98 of its bound, so this is not free |
| dlogp max | `<= 0.05` nats | the base's own top-1 token must not lose that much mass under the tested arm: a single row can be badly wrong and still leave the means untouched |

One more condition, outside the numbers: **any row whose NLL moves by more than 1 nat is a FAIL regardless of the
means** - a distribution that has moved that far is not the same distribution, however few such rows there are.

A `PASS` licenses shipping the flag; a `FAIL` on `KLD mean` while `same-top-1` stays at 100% is the signature of
reassociation noise above the bound rather than a wrong answer, and the first thing to try is `--peer-rows-fp32`,
which removes the partial's rounding and should collapse the difference.

### The emitter's floor, measured

`tests/program/ppl_emit_parity_test.cpp` (`ctest -L cpu -R ppl_emit`) feeds synthetic head rows through
`quantize_row` and decodes them with the reader's own arithmetic, at the model's vocabulary of 248,320:

| measured | value |
|---|---|
| worst \|decoded − true\| log-prob | 5.5e-04, within half the row's uint16 step plus float slack |
| worst KL(base ‖ test) against its own logits | **7.33e-05** |
| emitted top-1 vs the logits' top-1 | agrees on every row |

That 7.33e-05 is the resolution of the instrument, not a property of the model: it is what the `check` arm will
report if nothing is wrong, and every bound above is stated relative to it.

## Status of the gate

Built, unit-tested, **and once run with a confound**: one set of three arms was taken and returned `same top-1
94.41%` against the 99% bound, but the arms turned out not to have held the same experts (the paragraph above), so
that verdict is VOID and says nothing about the fold. The re-gate that should have fixed it - `--base-args
'--peer-reserve-mib 723'` on the reference arms - filled the tier to 13.98 GiB again, because that reserve prices
an expert tier and a tp2 boot has none: the room the tier fills into is priced by `peer_bytes`, and the paragraph
above now carries that algebra and the lever that reaches it. Treat everything in this page as the arithmetic's
bounds measured on synthetic rows, and the `+34-35%` prefill figure as a development A/B. **The next run must pass
`--base-args '--exp peer_charge_partial=1'` to the reference arms and print no placement `VOID` before its numbers
mean anything.** The flag stays dark (`--exp peer_partial_merge=1`, off by default) until `--verdict` prints PASS
on both length buckets and the bounds above are replaced by the run's own.


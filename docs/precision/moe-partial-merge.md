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
a shuffled routing, a two-thirds partner share and expert groups of 7 rows, and compares the split combine against
the flag-off combine over every `(token, d)`:

| Check | Measured |
|---|---|
| no partner rows: split vs single-card | 0 of 3,880 outputs differ (bitwise) |
| FP32 wire (reassociation alone) | max abs difference 1.2e-7, i.e. one fp32 ulp of the terms, 5% of the bound |
| BF16 wire (plus the partial's rounding) | max abs difference 1.9e-3, 0.93-0.98 of its bound, 3.6-3.8e-3 of the partial |
| fold grouped per expert group vs one chain | 0 of 3,880 partial cells differ |
| fold's per-token list vs the routing table | max abs difference 0 (same chain, same order) |
| a packed partial unpacked again | 0 of 65,282 bf16 values changed |

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

## Status of the gate

The mode is **not yet graded**. It needs the project's quality gate on the real pack — PPL and KLD against the
Q8_0 reference through the decode path (`basalt-bench --quality`) — and per-layer timing from a boot with the flag
on against one with it off. Until those exist, treat the numbers above as the arithmetic's bounds, measured on
synthetic rows, and nothing as a quality or performance claim.

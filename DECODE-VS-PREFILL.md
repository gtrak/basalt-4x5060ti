# Why decode is 21x more expensive per token than prefill

Measured on branch `pp4`, shipped config, same boot family.

| | value |
|---|---|
| prefill (fresh 38k prompt) | **2,593 t/s** = 0.386 ms/token |
| decode | ~111 t/s = **9.01 ms/token** |
| gap | **23x per token** |

## The window decomposed

    window 25.87 ms for 2.87 tokens
      stage GPU work   23.29 ms  (90%)   [c0 3.89, c1 6.19, c2 5.31, c3 7.90]
      drafter           2.29 ms  (9%)
      other/host        0.29 ms  (1%)

Per token per layer (48 layers):

| | us/token/layer | TFLOP/s | % of the ~44 TFLOP/s of 4 cards |
|---|---|---|---|
| **decode** | **169** | 0.39 | **0.9%** |
| **prefill** | **8.0** | 8.19 | **18.6%** |

The arithmetic per token is identical in both paths (65.5 MFLOP for top-10 on this
geometry). So this is not "decode does more work" - **decode spends 21x longer doing
the same work, at 1% of compute peak.**

## What it is NOT

Two plausible explanations were tested and both are wrong:

- **Not weight bandwidth.** A decode window touches 2,129 MiB of expert weights over 48
  layers, which across 4 cards is ~21 GB/s each = **5% of the 448 GB/s HBM peak**. And
  the earlier corrected figure put the expert GEMM at 71 GB/s = 16% of peak. Neither is
  saturated.
- **Not power.** The cards draw 40-61 W of 180 W during decode.
- **Not the drafter.** It is 9% of the window.
- **Not the hand-off wire.** `waitA`, `waitB`, `waitCPU`, `PCIe grp` are all 0.00, and
  expert PCIe is 0.0% (experts are resident).

It IS true that decode moves ~140x more weight bytes PER TOKEN than prefill (15.8 MB vs
113 KB) - prefill amortizes the same weights over 12288 tokens. But since those bytes
are not near peak bandwidth, that reframing does not by itself explain the time.

## What it is

**Arithmetic intensity.** The expert GEMM runs at M ~= 3 (a window's tokens) where
prefill runs at M = 12288. At M=3 a grouped GEMM cannot fill tensor cores: the fixed
cost per launch group dominates the useful math. Both paths pay ~800 launch groups per
window (16 groups x 48 layers); prefill spreads that over 12288 tokens and decode over
2.87.

That is why the only lever the data ever supported was **tokens per window**, and why
every attempt to raise it failed on measurement:

- deeper draft chains: the chain halves per position and stops at 1.92 offered
- a higher/lower draft confidence floor: measured both ways, t/s unchanged
- a depth-2 tree draft: +0.029 tokens/window = +0.8%, inside the noise band, and it
  needs same-slot branching the verify window cannot express
- two windows in flight: 86 t/s aggregate against 110 solo - the slots contend

**Decode at ~111 t/s is a local optimum for a 4-deep pipeline at batch 1 on this box.**
The 0.9%-of-peak figure says the headroom is real, but it is in the BATCH DIMENSION,
and the batch dimension is bounded by acceptance, which is bounded by the target model's
own predictability - not by anything in this engine's decode path.

# The GPU cosine differential: what the boots can answer now

Split out of `progress-tp2slice-verdict.md` §8b (which stays the account of what the chain is and why its use
across boots was retracted). Read §8 first: this file is the instrument that replaces the retracted comparison,

## 8b. The GPU cosine differential: what the boots can answer now

The retraction in §8 killed the hash as a cross-boot instrument, and the CPU replay of §8a cannot convict the
kernels. What is left is a **direction**: reduction-order noise between two boots of one prompt rotates a
component's sampled window by ulps and leaves its direction alone, while the classes this hunt is about — a stale
wire set, an imported span cell landing where the attention does not look, a column block that fails to tile —
substitute a tensor and destroy it. A cosine separates those two without needing the bits to reproduce, which is
exactly what the hash could not do.

**The sampler extension that makes it readable off a log line.** `QsaChainFp` gained a projection sketch:
`kChainProj = 32` fixed Rademacher projections of the same sampled window the hash covers, taken against
`qsa_chain_sign(p, row, col)` — a Johnson–Lindenstrauss sketch, spelled once in `qsa_chain.hpp` so the kernel that
fills it and the host that grades it cannot drift, and mirrored chunk-for-chunk in `qsa_chain_ref`. It prints as
32 fixed-width hex words appended to the existing field, so `qsa_chain_cosine(a, b)` rebuilds a per-component
cosine from two log lines and **no buffer ever crosses to the host**. `qsa_chain_parity` grades the sketch bitwise
between kernel and reference, and its CPU contract — which runs with no card — asserts the anchor (a window against
itself is 1.0), and falsifies it: a structureless window against the ramp must read **below 0.5**, a 0.02 %
perturbation must stay **above 0.999**, an unrecorded or all-zero side must read `kChainCosineNa` rather than a
misreadable 0 or 1, and a filled field must carry all 32 words. Its price: ~48 µs per component bare, ~138 µs
with the sketch (`qsa_chain_parity`'s line, 8192 × 1280) — read on a card the serving engine was using, so an
upper bound under contention until re-measured with the server stopped. Read a cosine as shape, never as a
measurement —
the estimator's own error is ~`1/sqrt(kChainProj)` ≈ 18 %, which is enough to tell "same distribution" from
"unrelated tensor" and nothing finer.

**The window** — the same server, the same cached prompt, `max_tokens=64`, greedy, `--exp qsa_dump_chain=1` on
each, in the order that keeps each comparison inside one boot where that matters:

1. gate off (the reference), 2. `--exp tp_prompt_attn=1`, 3. `--exp tp_prompt_attn=2`.

```
# one file per boot; the line is already grepped by construction
grep 'qsa chain:' boot.off.log | sed 's/^.*qsa chain: //' > chain.off.txt
grep 'qsa chain:' boot.1.log   | sed 's/^.*qsa chain: //' > chain.1.txt
grep 'qsa chain:' boot.2.log   | sed 's/^.*qsa chain: //' > chain.2.txt

# the three reads, in this order:
grep 'qsa pair: a second chunk entered the same prefill' boot.*.log   # P1: see below. SILENT is the expected answer.
grep 'peer chunk at token' | head -1                                  # must NOT read `no peer -1/0`; a boot whose
                                                                       #   pair never went live grades nothing
grep 'qsa chain: on' | head -2                                        # both cards armed, or the chain is half there
```

Then the cosines, per (card, chunk, layer, component), three ways — and **each comparison names whether it is
within-boot or across-boot**, because that is the difference between a reading and a confounded one:

| comparison | what a low cosine convicts | why it survives the retraction |
|---|---|---|
| **within one pass**: lead `mix` vs peer `pub` | the link or the wire set — the partner attended on rows the lead did not publish | both buffers written by the same run |
| **within one pass**: lead `kpool`/`vpool` vs peer `kspan`/`vspan` | the span import (§7's P3) | same |
| **across boots**: `bo`, `suml`, `sump` vs gate-off's | the collective, or outside the pair | a direction, not bits: order noise leaves it, a substituted tensor does not |
| **across boots**: `partl`/`partp` hold while `bo` moves | outside the pair after all — the residency delta is the suspect left standing | same |

The discriminator kimi's review #6 P3 asked for is read off the same three files without any new boot: on the
**fixed `=1`** at ~15 chunks, if the within-pass `kspan`/`vspan` cosines are clean then the corruption is in
`=2`-only readers, and if they are not it is the import — mode-independent, so one boot decides it.

**P1, settled rather than argued.** The sets are keyed on `qsa_ord % kWireSets` (`prefill.cpp:4483`), where
`qsa_ord = qsa_index - qsa_first` is the QSA ordinal of **one chunk of one `Prefill`** — nothing in the index names
a chain or a slot, so two chains driving one arm would indeed share the sets, coherently, and the agreement rings
would stay silent. The audit and what it rests on are §9's; the boot's job is only to confirm the guard never
spoke. **The line above must print nothing.** If it prints, cross-chain aliasing is confirmed as the mechanism and
the fix is per-chain set keying (a slot offset folded into the set index), not another measurement.

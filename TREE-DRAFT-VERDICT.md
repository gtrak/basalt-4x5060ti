# VERDICT: the tree draft is NOT built - it is worth ~+1%, inside the noise band

The gate PASSED (P(runner-up accepted | primary missed) = 38% at pos2, 45% at pos3,
pre-registered threshold 25%). The gate was answered correctly. **The projection built
on it was wrong**, and that is what kills the project.

## The arithmetic that decides it

The gate measures a CONDITIONAL rate. What pays is the unconditional contribution:

| quantity | value |
|---|---|
| windows that even reach position 2 | **22%** (325 of 1450) |
| rescue rate there | 38% of the 34% misses |
| extra tokens/window | +0.029 = **+1.0%** |
| one extra verified row, in those 22% of windows | +7.6% of rows = ~+0.2% window time |
| decode t/s | 110.9 -> **111.8 (+0.8%)** |

**+0.8% is far inside the +/-3% run-to-run band this box measures** (baseline against
itself spans 105.2-111.7 t/s). It cannot be validated here without heroic sampling, and
it does not justify the build.

The error was mine: "pos2 acceptance 66% -> 79%, so +5%" applies the conditional rate to
the whole population, but only 22% of windows have a position 2 to rescue. The chain
stops at 1.92 offered drafts, so most windows never reach the branch.

## And it needs architecture the engine does not have

A second independent reason, found while the build was being scoped:

- the verify window maps **row t to position pos0 + t**, a strict linear chain
  (`verify.cpp:3479` - `qsa_step_fill(..., pos0 + t, s)`);
- `stage_batch` requires every segment to own a DISTINCT slot - "a batch segment's slot
  is out of range or twice" (`verify.cpp:4786`).

A tree needs two rows at the SAME position on the SAME slot. The window cannot express
that, so this is a branching-verify feature, not a flag - a real subsystem for a +0.8%
return.

## Process note, so it does not repeat

The gate should have been `P(reach pos2) x P(runner-up rescue)`, not the conditional
alone. And the window-capability check (3 minutes of reading) should have preceded
dispatching a 30-minute worker, not followed it.

## What survives

The instruments, both committed and default-off:
- `--exp draft_hist=1` - the chain-survival histogram (solo path)
- `--exp draft_hist2=1` - the runner-up scoring, and the only route to the drafter's
  rank-2 token (`diag_top`, previously gated behind `spec_log`)

Both are cheap and answer "is acceptance worth attacking" in one boot.

# GATE RESULT: PASSED - build the depth-2 tree draft

Measured on branch `pp4` (commits `08071679`, `7ba7048f`), binary `/tmp/basalt_hist2`,
serving config as shipped plus `--exp draft_hist2=1`, one 1800-token greedy decode:

```
draft hist2: pos2 reached 325, primary missed 78, runner-up hit 30 (38%);
             pos3 reached 157, primary missed 22, runner-up hit 10 (45%); stale 0
```

**P(runner-up accepted | primary missed) = 38% at position 2, 45% at position 3.**

The pre-registered threshold was 25%. **PASS.**

`stale 0` confirms the step-to-position mapping is exact: the rank-1 pick the diag
buffer reports always equaled the draft actually offered, so the comparison is
measuring what it claims to.

Projected: position-2 acceptance 66% -> 66% + 34% x 38% = **79%**.

Caveat to check in the build: this is a first-order projection. Rescuing position 2
also lets the chain survive further, so the effect on tokens/window should be MEASURED,
not extrapolated - and the extra verified row per window costs ~2.5% of window time.

## What to build

A 2-wide tree at position 2 only (not deeper). Build it behind `--exp tree_draft=1`,
DEFAULT OFF. The gate for the build itself: tokens/window must rise enough to beat the
extra row's cost, i.e. decode t/s must improve beyond the +/-3% run-to-run noise band
(it is 105.2-111.7 t/s on this box). Use >= 5 paired samples, not 2.

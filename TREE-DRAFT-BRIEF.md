# TASK: depth-2 tree draft (measure conditional acceptance at position 2)

## The measured problem

Decode runs ~110 t/s at 2.9 tokens/window. An acceptance histogram (new
`--exp draft_hist=1`) shows:

    offered 1.92 drafts/window, accepted 1.35 (70%)
    chain reached:  +1 100%  +2 66%  +3 35%  +4 17%  +5 9%
    CONDITIONAL:     1->2 66%  2->3 53%  3->4 49%  4->5 53%

So position 2 succeeds only 66% of the time when the chain got there. Every token we
fail to accept is a row the verify window computed and threw away.

## The idea

The drafter emits ONE candidate per position. At position 2, offer TWO candidates
(a 2-wide tree: `[a, b1]` and `[a, b2]`). If either matches, the row is kept. Cost: one
extra verified row per window; benefit: position-2 acceptance should rise toward
66% + (1-66%)*P(b2 matches | b1 missed).

## GATE FIRST - do not build the engine change until this number exists

Add `--exp draft_hist2=1` that reports, for every window where the chain reached
position 2: of the windows where the primary candidate MISSED, what share WOULD have
been accepted by the runner-up candidate the drafter ranked second?

You need the drafter's top-2 at position 2. Check whether `sl.dprob` / the MTP head's
top-k already exposes it (`mtp.diag_top(j, k, ids, probs)` exists - see
`include/basalt/core/mtp.hpp`). If the runner-up is not available, EXTEND the draft
path to record it (ids+probs for rank 2 at each position), gated behind the flag, and
measure. Do NOT change the verify path for the gate.

Report the number: `P(runner-up accepted | primary missed)` at position 2. Also
position 3 if cheap.

Decision rule, stated before measuring: **build only if that number is >= 25%.** At
25%, expected position-2 acceptance goes 66% -> ~74%, which at ~2.5% window cost per
extra row is roughly break-even; below that it loses.

## If the gate passes

Implement the 2-wide tree at position 2 only (not deeper): the window offers
`[tok0, a, b1]` and `[tok0, a, b2]` as two rows, keeps whichever hits. The verify path
already accepts a run of matching rows, so this is mostly a draft-side change plus one
extra row in the window. Keep it behind `--exp tree_draft=1`, DEFAULT OFF.

## Rules (non-negotiable)

- Work only in /home/gary/opt/basalt-pp4 (branch `pp4`). Do NOT touch
  /home/gary/opt/basalt (production).
- **Do not start, stop, restart or signal any basalt or serve.server process.** The
  supervisor owns the serving instance on :8000; it is other agents' inference
  endpoint. You build and run CPU tests only. Ask for a measurement by reporting back
  the binary path and the exact boot command you want run.
- Build: `ninja -C /home/gary/opt/basalt-pp4/build basalt` and the relevant test
  target. Verify BUILD_TESTING is ON (`grep BUILD_TESTING build/CMakeCache.txt`).
  Confirm the test COUNT is non-zero - a suite configured with testing off passes
  vacuously.
- Every new kernel or arithmetic path needs a parity test against a CPU reference. A
  flag that changes numerics needs a quality note.
- Commit small, `area: what it does`, no amend, no force-push.
- Report: the gate number, what you built, the exact A/B boot commands, and the test
  results with real counts. If the gate fails, STOP and report the number - do not
  build.

## Context you need

- Drafting: `src/program/generate.cpp` around line 10188 (`S_mtp` window length,
  `draft_keep`, `cum_adapt.th()`), and the MTP chain in `src/core/mtp.cpp`.
  `mtp.draft_seq` / `mtp.draft_more` produce drafts; `sl.dr[]`, `sl.dprob[]` hold them.
- The acceptance decision is at generate.cpp:~10340 (`a` = last accepted index).
- `--exp draft_hist=1` is already implemented in the SOLO path (line ~10355). NOTE:
  the batch path (`batch_step`, ~7598) is NOT entered for a single request -
  `batch_on()` needs an active batch slot. Instrument the SOLO path.
- Per-layer/per-stage decode profile: `--profile` (a real flag, not `--exp`).

# RUN 26 — the gap term: what the decode profile now sees, and how to read it

EXPERIMENT ZERO of the decode campaign. The profile line (`basalt decode profile`, printed from the batch window in
`src/program/generate.cpp`) reported a window as `wall = run + commit + emit + draft`. The window's own stamps start at
its graph launch and stop at its last host action, so the time *between* two windows was invisible: the measured
drumbeat (~36.5 ms) minus the profiled window (~19.4 ms) is ~17 ms the line could not account for. The gap term closes
that accounting; the host-slimming lever (target solo 105-120 t/s) is aimed by it, not guessed at.

## 1. The term

    ; gap 17.2 ms = between 11.4 + prep 5.8;

    gap     = this window's launch stamp (w0) - the previous window's end stamp
    between = the previous window's end  -> this window's entry into the window function
    prep    = this window's entry        -> its launch (w0)

Read it on the window whose line it appears in: `wall + gap` is the wall-clock period between two launches, i.e. the
drumbeat the t/s number is the reciprocal of. `between + prep = gap` by construction. The line's four window terms
(`run + commit + emit + draft`) and the gap term are disjoint: the window's end stamp is taken after the last host
action of the window (the drafts and the tail commit) and before the line is written, so nothing is counted twice
except the line's own stderr write, which lands in the *following* window's `between`.

Two host-side `Clock::now()` calls per window (the entry and the end); the launch stamp already existed. No new
synchronisation, no new allocation, no GPU-side work. The term is printed only where the profile line is: the pair
topology (a layer split), every 32 windows, or every window with `--exp decode_profile=1`.

The first window after the slots go idle carries no gap term (there is no previous window to measure from), and a
measured gap is never an idle wait: the stamps reset when every slot ends.

## 2. What the decomposition will name

`prep` is the window's own pre-launch host work, in this order in the code:

| candidate | where it lives | shows up in |
|---|---|---|
| the KV back-ahead loop (each slot's next cells backed ahead of its windows) | window prologue | `prep` |
| the slot scan (active slots, row/pos/lens setup) | window prologue | `prep` |
| the draft-length rule and the draft-row ingest (the policy's row budget, `sl.dr`/`sl.sbuf` into `tok[]`) | window prologue | `prep` |
| `apply_pending` (the tier's pending changes applied before the run) | window prologue | `prep` |
| the adaptive refill round (every `--adapt-every` windows, with its `dm_presync`) | window prologue | `prep` |
| the run-graph launch itself | inside `run_slot_segs` | **not** the gap: it is in `run` |

`between` is everything the caller does outside the window function between two windows:

| candidate | where it lives | shows up in |
|---|---|---|
| the overlapped prompt read's chunk jobs (a read running beside decode pays windows in between) | the read thread's mail loop | `between` |
| the input-line handling (`BSTOP`/`BYIELD` scanning), admissions, slot lending/refill | the serving loop | `between` |
| the per-window log writes: this profile line and the batch summary line | after the window's end stamp | `between` |
| the BT/BDONE token lines | the emit phase, inside the window | the `emit` term, not the gap |
| the doorbell arm and the peer-window handshake | inside the launch path | in `run`, not the gap — see §3 |
| the sampling of the window's rows (`sample_rows`) | inside `run_slot_segs` | in `run`, not the gap |
| the drafter's row ingest (`mtp.draft_seq`, the deferred rows) | the window's draft phase | the `draft` term, not the gap |

## 3. What the gap does *not* cover, and where it went instead

Anything reached from `run_slot_segs` is inside `run`, not the gap: the graph launch, the doorbell arm, the peer
launch and the per-layer waits. Those already have their own readings — `GPU-reach wait`, the per-stage
`wait`/`combine`/`peer push` terms, and `--exp host_ev=1`'s `DecodeEvents` host stations. If `gap` comes back small
(~1-3 ms) and the drumbeat still exceeds `wall + gap`, the missing time is inside `run` and the question is the launch
path, not the host loop. If `gap` comes back at ~17 ms with `prep` dominant, the target is the window prologue (the
table above); if `between` dominates, it is the caller's loop.

## 4. Reading table for the next boot

Boot with `--exp decode_profile=1` (every window) so the distribution, not one sample, is visible. Take the mean and
the spread of `gap`, `between` and `prep` over at least a few hundred windows; a single line is not a measurement.

    grep 'basalt decode profile' engine.log | grep -o 'gap [0-9.]* ms = between [0-9.]* + prep [0-9.]*'

Prints nothing if the term is absent (an old binary, or no layer split in the boot).

| what the boot reads | it means | the next lever |
|---|---|---|
| `gap ≈ 0`, `wall ≈ drumbeat` | there was never a 17 ms host gap; the drumbeat was the window | the profiled-window number is the whole story; slim the window |
| `gap ≈ 17`, `prep` dominant | the window spends its pre-launch host time before the graph runs | the prologue table in §2, term by term |
| `gap ≈ 17`, `between` dominant | the caller's loop (admissions, the read thread's jobs, the log writes) holds the windows | host slimming in the serving loop |
| `gap` large with an overlapped read active | the read's chunk jobs are being paid in the gaps | expected, not a defect; compare against a quiet-decode boot |
| `gap` small, `wall + gap < drumbeat` | time unaccounted inside `run` (launch, doorbell, peer window) | `--exp host_ev=1`, the `DecodeEvents` stations |

Label every number quoted from this line with the GPU power limit it was measured under (AGENTS.md).

## 5. Where it is in the code

`src/program/generate.cpp`: the window's entry stamp (`w_in`) and end stamp (`wp_end`) plus `wp_gap`/`wp_prep`, next to
the existing `wp_*` window-profile state; the line is built in the `basalt decode profile` block. The gap is read at
the launch stamp because the window stamps its own end after the drafts, so the value printed on a line belongs to
that line's window.

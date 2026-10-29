# TP2PP2-MEASURE — measurement checklist for the restart window

The M1 branch (stage-pair scaffold + EP-within-pair, `--tp-size 2`) was built and
reviewed but **nothing was measured**: the production server was live, so no engine
was started, no restart, no `basalt-ctl`. This file is the precise, ordered checklist
to run **once the maintainer opens a restart window** (stops the live server):
stop → build → GPU parity tests → tp2 boot → measurement requests → the four numbers
→ baselines → pass/fail → rollback. The QAD pack for this run is
`models/Qwen3.8-FN-FP4Local.basalt`.

Work from this worktree. Do **not** run any of this while the production server is
live: the GPU parity suite does not run beside it (the parity tests hit
`cudaMalloc: out of memory` when the live engine holds the cards), and a second
engine will not boot against occupied cards.

---

## 0. Preconditions

- [ ] Maintainer has stopped the live production server (step 1 below).
- [ ] All four cards free: `nvidia-smi` shows ~0 MiB used on CUDA 0-3. A killed engine
  holds its cards' VRAM 10-40 s past exit; wait for it.
- [ ] Free RAM ≥ 14 GB and SwapFree ≥ 4 GB (the pinned expert complement hard-fails
  when short). `free -g`. In the worktree root: `cd /home/gary/opt/basalt-tp2pp2`.

---

## 1. Stop the server

The production server is the Python front (`python -m serve.server`) wrapping the
engine, bound to `:8000`. Stop whichever is running:

```bash
# if it was started under basalt-ctl:
./basalt-ctl stop

# otherwise, find and stop the front + engine (they are a parent/child pair):
pgrep -af 'serve.server|bin/basalt'
kill <serve.server pid>          # the front; the engine child exits with it
```

Confirm the cards actually freed (a lingering child holds VRAM):

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # wait until every row is near 0
```

---

## 2. Build

Build in the worktree's own `build/`, `-j4` max:

```bash
cmake --build build -j4
```

- [ ] Build is green (no errors). Stop here and report if it is not.

---

## 3. GPU parity tests (server must be stopped)

Run the full GPU-labeled parity suite (the engine regression gate) plus the tp2 tests:

```bash
ctest --test-dir build -L gpu --output-on-failure

# the tp2-specific parity tests, individually, if you want them isolated:
ctest --test-dir build -R 'tp2_head_split_parity|tp2_allreduce_parity' --output-on-failure

# the CPU-side invariant test (no GPU needed; the oracle for the pair split):
ctest --test-dir build -R 'tp2_pair_test' --output-on-failure
```

Known **environment-only** failures (not regressions — they fail identically on the
main build):

- `expert_multi_test` needs AVX-512, which this CPU lacks.
- Any parity test that fails with `cudaMalloc: out of memory` means a card is not
  actually free — go back to step 1, do not read it as a regression.

- [ ] Record the pass/fail set. It must match the main build's set (no *new* failures).

---

## 4. Boot the engine

The engine itself only takes requests on stdin; the OpenAI-compatible HTTP API the
measurement requests hit is the Python front. Launch the front wrapping the engine,
with the QAD pack and `--tp-size 2`. This mirrors the production flag set (the
measured values from `basalt-ctl`), with three deliberate changes: the model is the
QAD pack, `--tp-size 2` is added, and `--expert-profile-save` is dropped (it writes a
file). Host is `127.0.0.1` (no need to expose it for a local measurement).

```bash
cd /home/gary/opt/basalt-tp2pp2
python -m serve.server \
  --engine-bin bin/basalt \
  --model models/Qwen3.8-FN-FP4Local.basalt \
  --alias local --host 127.0.0.1 --port 8000 \
  --gpu 0,1,2,3 --parallel 2 --slot-full-context \
  --prefill 12288 --resident-budget-gib 37 --prompt-cache 8 --spec 8 \
  --ple-io direct --kv int8 --max-context 262144 --adapt-swaps 96 \
  --resident-experts --peer-reserve-mib 600 --decode-timing --suffix-draft 8 \
  --batch 2 --grammar on \
  --tp-size 2 \
  2>&1 | tee /tmp/tp2-serve.log
```

Wait for readiness (the front prints `Ready!` and `/v1/models` lists the model):

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

### 4a. Baseline boot (plain 4-card split, `--tp-size 1`)

To compare, boot the *same* command with the split forced but no tp2. The production
split is the plain 4-card layer split, forced by `BASALT_FORCE_SPLIT=1`; under
`--tp-size 1` that is the exact baseline this work is measured against. So: stop the
tp2 boot (Ctrl-C), wait for the cards to free, then re-run the step-4 command with
`BASALT_FORCE_SPLIT=1` prefixed and `--tp-size 2` removed, teeing to `/tmp/tp1-serve.log`.

Run the measurement requests (step 6) against **both** boots and record both sets of
numbers. The tp2 run is the subject; the tp1 run is the comparison.

---

## 5. Confirm the boot log lines (tp2 run)

Grep `/tmp/tp2-serve.log`. All of these must be present before measuring:

```bash
grep 'tp2: each layer split stage is a pair of cards' /tmp/tp2-serve.log
grep 'tp2: .*stages of .* layers each, K=' /tmp/tp2-serve.log        # expect: 2 stages of 24 layers each, K=24
grep 'tp2/pp2: stages' /tmp/tp2-serve.log                            # expect: A(layers 0-23)=[CUDA0,CUDA1] B(layers 24-47)=[CUDA2,CUDA3]
grep 'tp2: CUDA' /tmp/tp2-serve.log                                  # expect: two "CUDA%d holds N of the M ranked pairs" lines (CUDA1 and CUDA3)
```

- [ ] All four present, and the topology reads A=[CUDA0,CUDA1] B=[CUDA2,CUDA3].
- [ ] No `tp2, CUDA%d ... no room` / `refuse` / error line (a partner tier that cannot
  size its half refuses to boot). In the tp1 log, confirm the *absence* of the `tp2/` lines.

---

## 6. Measurement requests

Two measurements: **decode** (window ms, tok/s per stream, hit rate) on a cached
~150k-token conversation, and **prefill** (t/s) on a fresh prompt. Send them to the
OpenAI-compatible endpoint on `:8000`.

### 6a. Decode — two concurrent ~150k-context streams

The baselines are at ~150k context. Build a ~150k-token conversation and continue it
so the context is cached and decode dominates (the production box has served a
242,707-token request end to end). Send **two concurrent** requests (matching
`--parallel 2`) to read per-stream tok/s — put a `&`/`wait` around the two curls:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[ ...~150k-token history... ,
       {"role":"user","content":"Continue."}],"max_tokens":512,"temperature":0}' &
# ... a second identical curl for the other stream, then: wait
```

- [ ] Both finish with `end=stop` (not `cancelled`, not an error).
- [ ] Output is coherent (no garbage — a first-order correctness check).

### 6b. Prefill — one fresh prompt

A fresh (uncached) ~12288-token prompt (matching the measured chunk) so the number is
prefill t/s, not a cache hit: `messages` holds only the fresh prompt, `max_tokens:16`.

Repeat step 6 on the tp1 boot for its comparison numbers.

---

## 7. The four numbers — exact log lines to grep

Collect these from the tp2 run's log (`/tmp/tp2-serve.log`) and record them per boot.

### (1) Decode window ms

The `--decode-timing` line, once per decode request. The window is the `%.2f ms/window`
field (the total per decode window, after `tokens/window`):

```bash
grep 'basalt decode timing' /tmp/tp2-serve.log
#   ... %lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = verify ...
```

Read the `ms/window` value. (The `[server] ... done ... tg=N X t/s` line is the
wall-clock per-stream rate; the engine's `ms/window` is the decode-window figure the
baselines quote.)

### (2) tok/s per stream

The per-request server done line (one per stream):

```bash
grep '\[server\].*done' /tmp/tp2-serve.log
#   [server] slot.. done  pp=.../...c ...  tg=512 84.3t/s acc=..% dl=..  ttft=..s end=stop hit=..% pcie=..%
```

The `tg=... <rate>t/s` field is that stream's decode tok/s. Record each stream's rate
(and note the lower of the two under concurrency).

### (3) Expert cache hit rate

The per-request engine line (VRAM share of routed experts):

```bash
grep 'decode expert cache hit rate' /tmp/tp2-serve.log
#   decode expert cache hit rate: 93.5% (545 hits / 583 lookups); ... more read ... over PCIe ...
```

The `done` line's `hit=..%` field is the same value, server-side.

### (4) Prefill t/s

The fresh-prefill engine line (rate over the read, uncached):

```bash
grep 'prefill .* tokens in .* chunks' /tmp/tp2-serve.log
#   prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts streamed ...
```

The `tok/s` field is the prefill rate (the `done` line's `pp=... <rate>t/s` agrees for
the fresh-prefill request). Record all four for the tp2 boot and all four for tp1.

---

## 8. Baselines to compare against

Measured on 4× RTX 5060 Ti (sm_120), NVFP4 pack, ~150k context — the plain 4-card
split (`--tp-size 1`, `BASALT_FORCE_SPLIT=1`):

| number | baseline |
|---|---|
| decode window | **34.5–38.7 ms** at ~150k |
| prefill | **3750–3868 t/s** (fresh, exclusive) |
| decode | **72–88 tok/s per stream** |

The tp2 napkin target (window ~20–24 ms) is a **later-milestone** goal (it needs M2's
head-split + the all-reduce, which M1 does not have). It is **not** the M1 pass bar —
M1 moves expert placement within a pair, it does not halve per-layer compute.

---

## 9. Pass / fail criteria

**PASS** (M1 is a good scaffold; safe to proceed to M2) when all hold:

- [ ] Build green; GPU parity set matches the main build (no new failures).
- [ ] tp2 boot names the topology (`tp2/pp2: stages A=[CUDA0,CUDA1] B=[CUDA2,CUDA3]`)
      and both partner tiers size their half (no `no room` / refuse).
- [ ] Both decode streams finish cleanly with coherent output.
- [ ] decode window ≤ ~45 ms (no large regression vs the 34.5–38.7 ms baseline; a
      modest rise from the partner d2d + pipelined-chain oversubscription is expected).
- [ ] tok/s per stream ≥ ~60 (within ~20% of the 72–88 baseline floor).
- [ ] prefill ≥ ~3400 t/s (within ~10% of the 3750–3868 baseline floor).
- [ ] hit rate ≥ ~90% (partner-held experts served from VRAM, not spilling to file).

**FAIL** (stop, record the finding in `TP2PP2-NOTES.md`, roll back) if any:

- build fails, or a *new* GPU parity test fails (vs the main build).
- tp2 boot refuses, crashes, or an exception/OOM appears in the log.
- output is incoherent (a correctness break, not just a speed change).
- decode window > ~48 ms, or any stream < ~40 tok/s, or prefill < ~3000 t/s — i.e. a
  large regression, not a modest one.
- hit rate drops well below ~90% (experts being read from the file/SSD instead of the
  partner card).

A FAIL is a documented dead end, not a hacked-through clean build: append the finding
(dated) to `TP2PP2-NOTES.md`, roll back, and report the state of the evidence.

---

## 10. Rollback

The tp2 path is flag-gated and default-off; turning it off is the rollback — nothing
else changes. Stop the tp2 boot and re-run the step-4 command **without** `--tp-size 2`
and with `BASALT_FORCE_SPLIT=1` prefixed (the step-4a baseline command): `--tp-size 1`
(the default) with the split forced is byte-for-byte today's plain 4-card split, which
is what production serves.

---

## What is NOT measured here (carry over from M1)

- Attention head-split within the pair (M2) — all attention/dense still runs on the
  stage lead card, so the window is not yet halved.
- The 2-way all-reduce / collectives (M2/M3) — not wired in; the pair's cost is the
  partner's d2d read, not a collective.
- Pipelined-window oversubscription of the partner card: both chains (`ver`/`ver_b`)
  drive the same partner tier, so the two peer kernels each take the full
  `dec_moe_peer_blocks()` on one card. `--parallel 2` without pipelined windows does
  not hit this; flag it if enabled.
- The `--no-pool` + tp2 corner (partner experts read from the file) is not exercised;
  the resident path is what is measured.
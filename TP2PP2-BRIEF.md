# TP=2 / PP=2 exploratory branch — design brief

## Mission

Add an exploratory tensor-parallel-2 × pipeline-parallel-2 execution mode to the
basalt engine, behind a flag, sufficient to MEASURE the decode window and prefill
against the current layer-split baseline. This is an exploration branch: honest
plumbing over polish, every assumption written down, no claim without a number or
an explicit "unmeasured."

## Hard constraints

1. **The production server is LIVE and must not be touched.** Never run
   `basalt-ctl`, never signal any `basalt`/python process, never bind port 8000.
   There is no restart in this milestone. Verification = the build compiles, the
   unit tests that exist pass, and a written runbook for the eventual restart.
2. Builds: `cmake --build build -j4` (NOT -j8: the live server's host threads
   share this CPU; a full-tilt build visibly stalls it). Use the worktree's own
   `build/` dir.
3. Commit in logical units as you go (the branch is the deliverable); the commit
   style of `git log --oneline -15` on main is the house style: lowercase area
   prefix, the why in the body with numbers.

## Topology (the design)

- Cards 0,1 = **stage A** (layers 0–23), cards 2,3 = **stage B** (24–47). The
  existing stage machinery (`stages`, the per-stage carve, `bslot_ss`,
  `set_stage`, the handoff buffers) is the scaffold — TP=2 means each STAGE
  becomes a *pair* of cards computing cooperatively instead of one card
  sequentially owning its layer range.
- **Attention**: head-split within the pair. Each card computes half the heads
  (KV also head-sharded — ¼ of the full KV per card). Head-split attention is
  bitwise-exact: no numerics gate for it.
- **Dense/FFN (non-MoE) linears**: row-split within the pair, partial sums
  reduced by a 2-way all-reduce. NOTE: changes reduction order → the numerics
  gate applies (document; do not implement the gate).
- **MoE experts**: EP-within-pair — each card holds HALF the experts for its
  stage's layers (≈15.8 GiB/card, same VRAM as today). Tokens routed to the
  peer's experts go by the EXISTING d2d machinery (`set_ring_remotes` /
  the tier's copy path). No expert weight sharding in this branch (the
  small-GEMM collapse + numerics gate — rejected, see WORKLOG rationale).
- **Collectives**: 2-way all-reduce = each card copies its full buffer to its
  peer over `cudaMemcpyPeerAsync` (peer access is already enabled at start) and
  adds locally — ONE exchange per collective on 2 ranks. Implement a small
  helper (`tp2_all_reduce(buf, rows, stride, stream, peer_dev)`) in the kernels
  layer; do NOT pull in NCCL for the exploration (the sglang worklog lesson:
  on this PCIe/PHB box an IPC-buffer exchange beats the NCCL ring; NCCL is a
  later production decision).
- **KV/session**: each card in the pair holds its head-shard's KV. The slot
  model's `slot_cap` semantics must hold per pair (a slot's conversation spans
  the pair's two shards).

## Codebase map (what matters, from a measured campaign on this exact box)

- `src/program/generate.cpp` (~11k lines): the serve loop, `stages` + the carve
  (per-stage layer ranges and sessions), `read_part` (prompt reads — on this
  branch leave the split's exclusive-read policy alone), the batch windows
  (`ver.run_slot_segs`), `set_ring_remotes` call sites, the decode-timing log
  line (~10600+), `ov_active`.
- `src/prefill/prefill.cpp` + `include/basalt/prefill/prefill.hpp`:
  `set_ring_remotes` — the d2d expert copy path the pair's EP reuses.
- `src/core/expert_source.cpp` + `include/basalt/core/expert_source.hpp`:
  the expert cache/slots/blob layout (`expert_layout()`, `blob_bytes`), the
  complement/pin logic, the budget contract (do not weaken it).
- The price model (`plan_multi_gpu` / `auto_plan.cpp`): tier 27.0, split 31.2,
  tp 24.7 (not built). Leave the model alone; the new mechanism is flag-gated
  (`BASALT_TP2=1` env or `--tp-size 2`), default off, and the model's TP line
  stays aspirational until measured.
- Baselines to beat/compare (measured, 4× 5060 Ti gen4 x4, NVFP4 pack, 150k ctx):
  decode window 34.7–38.7 ms (72–88 tok/s per conversation), fresh prefill
  3750–3898 tok/s exclusive, TTFT on cached 150k turns 0.2–2.5 s.
- TP=2/PP=2 napkin target: window ~20–24 ms (2 stage handoffs not 4, halved
  per-layer compute, 48 × ~30 µs pair-collectives ≈ 1.5–4 ms comm).

## Scope for THIS branch (milestones, in order, each committed)

1. **M1 — scaffold**: the flag; stage-pair construction (2 stages × 2 cards);
   expert EP-within-pair reusing the existing routing; expert VRAM carve at half
   depth per card; build green; the boot log names the topology
   ("tp2/pp2: stages A(layers 0-23)=[CUDA0,CUDA1] B=[CUDA2,CUDA3]").
2. **M2 — attention head-split + the 2-way all-reduce helper.** Unit-testable
   without GPUs where possible (a CPU reference path or a skip-if-no-4-GPUs
   guard in the test).
3. **M3 — dense/FFN row-split** + wire the collectives into the verify path.
4. **M4 — measurement runbook** (TP2PP2-RUNBOOK.md): the exact commands, the
   expected log lines, the four numbers to read (window ms, tok/s per stream,
   hit rate, prefill t/s), and the rollback (the flag default-off = today's
   split). No claims without numbers; list what is unmeasured.

Out of scope: numerics gate, NCCL, expert weight sharding, --tp-size 4,
production polish, the price model wiring.

## Reviewer contract

The reviewer reads every commit; keep commits self-contained and honest. If a
milestone turns out wrong (e.g., the pair EP fights the slot model), STOP,
write the finding in TP2PP2-NOTES.md (append, dated), commit, and end the run
with the state of the evidence. A well-documented dead end is a deliverable;
a hacked-through clean build is not.

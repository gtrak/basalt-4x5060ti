# Continuation — the tp2 campaign on 4x RTX 5060 Ti

## 0. PROCESS RULES (owner-mandated 2026-10-30; these supersede anything below and must survive compaction)

1. **SLOTS**: 1 desktop worker (a different box, fully isolated) + 2 local workers. The LOCAL
   box hosts the RUNNING basalt instance whose code the workers are changing.
2. **THE RUNNING BASALT INSTANCE IS THE LOCAL WORKERS' INFERENCE ENDPOINT.** Their LLM
   serving IS :8000 — bouncing basalt bounces their minds mid-generation; benching while they
   generate competes with them for the GPUs and contaminates both directions. The local
   workers' "Retry the previous request" log spam is inference retrying against a bounced
   endpoint; the power-outage connection error was this. (It also means worker code-generation
   traffic IS the drafter's target workload class — valid for its validation, but never call
   it uncontaminated while benches run.)
3. **SERVER-DOWN PROTOCOL**: merge everything pending on the feature branch FIRST, then take
   the server down, run the full battery (build, CPU, GPU, boot, bench), then BRING THE SERVER
   BACK UP before any next dispatch. Batch tests; no piecemeal boots per merge.
4. **ALL WORKERS DONE BEFORE THE SERVER GOES DOWN** — because down means their inference is
   gone. No window while any worker is in flight, regardless of what has merged. Benches
   while workers run are also contaminated both ways: grade from quiet-machine runs, or
   compare only like-for-like (both legs with comparable worker load).
5. **LONG WORKER TIMEOUTS**: pass explicit timeoutMs (3h) on dispatches and resumes. The 30-min
   default fragmented a whole night into ~12 resume cycles.
6. **DELEGATION LOSES LONG BRIEFS**: dispatches routinely arrive with an empty Task field.
   Expect the empty-task supervisor request; answer it AND steer the full brief (both channels).
   Never let a worker pick its own task from the production tree's docs.
7. **PER-LAYER METRICS ARE MANDATORY AT EVERY GRADING**: never grade from done-line t/s alone —
   read the per-layer/per-stage breakdowns (peer chunk, decode profile, layer time) term by
   term. The done-line hid RUN 25's entire decomposition (attn +13-15, peer-wait +50-60).
8. **OVERNIGHT AUTHORITY**: when the owner grants an overnight window ("do what you can until
   X"), do not stop prematurely; keep working, keep workers fed, keep the server serving.
9. **WORKER HYGIENE (enforced)**: commit coherent units before any wall clock; progress files
   are the handoff (context death at 262k is routine); predictions filed before measuring;
   honest numbers, never tuned to pass; refusals name their cause; CPU gates with the TEST
   COUNT verified (BUILD_TESTING-off makes ctest pass vacuously).
10. **FORK PUSHES AS CHECKPOINTS** after every accepted unit; the branch of record is tp2pp2.


State as of `293ca647` (2026-10-30 ~01:30 UTC). Written as compaction insurance: a future
session should be able to resume from this file plus TP2PP2-NOTES.md without archaeology.
Facts, measured numbers, decisions, and the gated roadmap. Interpretations are labeled.

## 1. Mission and constraints

- Engine: basalt (fork `gtrak/basalt-4x5060ti`, upstream jesdga95/basalt). Model:
  Qwen3.8-Flash-Next NVFP4 MoE — 48 layers, 512 experts/layer, top-8 (~10 row-visits/
  token/layer), **6B active params** (owner-stated; makes prefill weight-streaming-bound
  and decode latency-bound). Expert shape gate/up 1280x2560 + down 2560x640.
- Hardware: 4x RTX 5060 Ti 16 GB, PCIe GEN4 x4 each, NO NVLink; d2d ~6.6 GiB/s;
  aggregate one-way ~26.4 GiB/s; NCCL 4-way 360 µs@1MB with `P2P_LEVEL=PIX`
  (SYS was 658 µs — the sglang WORKLOG's key fix); FlashInfer PCIe-IPC AR +17.6%
  decode conc 6. 60 GB RAM, tiers: VRAM experts + RAM page-locked + SSD.
- Mandate: decode-first (agentic traffic: long cached contexts, 200-3k generations,
  periodic 122k-token compactions). Owner decision (2026-10-30): **the tp2 topology is
  KEPT**; the goal is now "prefill close to the split's 3,750-3,868 t/s AND faster decode,"
  optimized by cost/benefit, with TP=4 as the primary candidate and TP=2/PP=2 the fallback.
- Hard constraints: never modify swap; KV fixed at 2 slots x 262144; minimal fork diff;
  flags over engine code; the production split tree `/home/gary/opt/basalt` untouched.

## 2. Measured state (this box, tonight)

| config | prefill (122,300 tok fresh) | decode |
|---|---|---|
| split production (PP=4, no AR, 4-way tiers) | **3,750-3,868 t/s** | 35.4-39.8 ms/windows, 74-90 t/s |
| sglang TP=4 + PIX + PCIe-IPC (27B DENSE, owner's box) | ~2,000 sustained | allreduce-bound at conc |
| **tp2 EP+PP current (`293ca647`)** | **1,993-1,998 t/s @ 8192** | **solo 78-80 t/s, window 19.4 ms; concurrent parity-to-win** |
| tp2 at start of tonight | 1,393 t/s | — |

Night's climb: 1,393 -> 1,406 (RAM-reader) -> 1,905 (BF16 rows + re-priced share)
-> 1,993 (FP4-plane wire + early transport + phase separation) -> 1,998 (L2 warm-up).
Every step attribution-confirmed by its own A/B.

Layer anatomy (stage A, 153.2 ms): **whole** attention 54.4 on the lead (NOT split —
see §4 correction) + lead-experts 77.8 + peer-wait 11.0. Stage B inverted (lead 77.1,
peer 92.2). Leads' PCIe RX ~7.5 GB/s during prefill (tier fill + uploads + returns).

## 3. Architecture: what "tp2" actually is

TP=2/PP=2 in NAME only. Reality: **expert-parallel + pipeline-parallel** — stage A =
layers 0-23 on pair [CUDA0 lead, CUDA1 partner], stage B = 24-47 on [CUDA2, CUDA3].
Whole-expert continuation placement: lead holds the hottest ~2,900 (layer,expert) pairs,
partner the continuation band, RAM tier the cold tail. Attention runs WHOLE on the lead
in BOTH prefill and decode (the head-split machinery exists but the prompt path never
calls it — run 9; the only wired piece is RUN 17's decode lead-side, flag-gated, refusing
pending the peer item). The entire wire program (runs 8-21: crossover pricing, plane
shipping, BF16 row returns, share models, phase scheduling) is the tax on whole-expert
placement. TRUE tensor parallel (shard every expert 4- or 2-way, all-reduce partials)
would DELETE that program and balance by construction — that is the TP=4/TP=2 roadmap.

## 4. What we tried — the complete lever ledger

Landed and measured (tonight, in order):
1. Engagement fix (run 11): `Prefill::set_peer` was never called under tp2 (guarded on
   expert-tier validity) — the partner was a ring source, never a row-computing peer.
   Boot line now names crossover + share + wires. 1,384 -> 1,393.
2. RAM-reader `--peer-ram-rows` (run 14): partner fetches RAM-band blobs over its own
   lanes, computes those rows. Engaged (9,224 experts from RAM) but only +1%: the wall
   moved to the return leg. Mechanism kept.
3. BF16 return rows (run 16): `rows 114.6 -> 56.3 ms` = 50.9% ~= exactly BF16's
   byte-weight — **the return-bytes diagnosis confirmed by its own A/B**. Share
   re-priced per stage (4/8 -> 1/8). 1,406 -> 1,905.
4. FP4-plane outgoing leg (run 21): the lead ships planes it already rounds for its own
   experts (1,444 vs 10,240 B/token) — bit-identical by construction (parity test proves
   shipped planes == the peer's own). Peer act 21.3 -> 2.8 ms. The "hoist" premise was
   stale (quantize already preceded the enqueue) — enforced with a named guard instead.
5. Phase separation (run 19): RAM-band rows first, activations cross when written
   (ev_act), per-group row streaming (already existed). Peer-wait 36.3 -> 11.0 (A),
   88.3 -> 27.4 (B). Worker self-caught an inverted sort comparator via a 4,000-placement
   harness with negative controls.
6. L2 warm-up coverage (run 20): the stage-split's trimmed routers were the uncovered
   case; fixed. Warning gone; solo window 19.4 ms (best recorded).
7. Chunk sweep re-mapped (run 18): 8192-16384 flat ~1,905-1,998 (peak); 2048 -> 826
   (routed-only regime <= 2,559); 24576 -> 1,679; 32768 -> 1,897. Wall is
   chunk-size-invariant — the streaming dividend hides behind copy streams.
8. Decode instrumentation (run 12, fixed by reviewer): CUDA events inside captured graphs
   are BROKEN (invalid argument) — replaced with device-side %globaltimer stamp kernels
   (the prof_ mould); also fixed an OOB mask index (stage-end mask read at the event
   index L*5 instead of the layer index L). Both stages now measure every window.

Dead ends, corrections, and retracted claims (do not re-litigate):
- The original TP=2-first ordering (kimi #2) — withdrawn in #3 when the "o_proj AR
  already runs at 2-way scale" pillar proved false (no TP attention substrate exists at
  any degree; test-only).
- "AR hides under compute" (my inference, kimi #3 Finding 2): wrong in a streaming-bound
  regime — AR bytes and tier bytes compete for the same links. The surviving mechanism is
  4-way link-load splitting (aggregate throughput), not hiding.
- "Quarter-expert coverage saves streaming bytes 4x" (my inference, kimi #4 killed):
  coverage fraction is byte-limited; a quarter-expert miss makes all 4 cards fetch their
  quarters — total fetched = the whole blob. TP=4 ADDS bytes (AR 144-360 GB/prompt,
  H-dependent) and DELETES the wire program (~26 GiB).
- "Prefill attention is already head-split" (my claim to the owner) — WRONG: whole
  attention on the lead; the head-split is decode-side/test-only for the prompt path.
- `mtp_preswap` (run 15): no measurable win; default off. Drafter cannot overlap the
  next run (GPU-chain-bound). The combine-spin hypothesis (DECODE-NEXT item 1) KILLED:
  decode runs zero collectives, per-stage wait 0.0.
- The byte-inequality as a TP=4 kill gate (kimi #3) — DEMOTED by kimi #4: savings ~= 0
  against AR bytes means it fails on volume trivially; the utilization reframe is the
  correct wall model. Trace-replay demoted to parameter estimation.

## 5. The kimi review series (synthetic/hf:moonshotai/Kimi-K3) — evolution

- #2: bounded push + kill-switch; findings: chunk-passes lever, stage-serialization
  check (ran: stages already pipeline ~90% of the 2-stage ceiling), comp-vs-return
  confound (resolved: return-bytes), FP8-AR rejected, morph amortization weak.
- #3: ratified gated TP=4-first; P1s: attention-slice net unproven ([-5,+25] ms/layer —
  must ship with net-ms transport measured), ledger currency discipline (h2d vs d2d
  paths), decode floors per step, MMQ bench needs 32- AND 64-row points.
- #4 (final): corner arithmetic corrected — honest TP=4 band **2,000-3,140 t/s
  (39-61 s)**, pessimistic corner = the sglang dense anchor exactly; the fourth limit is
  **rendezvous-cadence utilization** (structurally harder than the split's trivially
  parallel 4-deep overlap; must be measured on synthetic buffers first); FP4 pack
  sliceability at 640-wide is a load-time question (three outcomes priced, runtime
  dequant-slice = kill); byte-inequality demoted, overlap-cadence bench promoted.

## 6. What's next — the gated sequence (kimi #4's go/no-go package, owner-approved shape)

Step 0 (in flight or next, cheapest-decisive first):
1. Volume ledger close-out: pin H from geometry; fetched volume 744-960 GiB;
   trace-replay demoted to miss-rate-vs-coverage refinement (NOT a kill gate).
2. Ring matrix (worker `tp2-tpstep0` in flight): AR latency floor KiB->64 MB, per-link
   BW under concurrent tier fill; pins H; answers d2d-vs-bridged per pair.
3. **Overlap cadence bench (NEW, promoted ahead of MMQ)**: synthetic 4-card x {2,4}-deep
   interleave, measure AR-hidden fraction + max-of-4 cadence inflation under injected
   tier-fill jitter. **KILL TP=4 if hidden fraction < 70% at 4-deep** (wall flips
   wire-exposed -> <= ~2,200, below the EP+PP+slice floor).
4. Pack inspection: is a 640-wide N-slice self-contained in the on-disk FP4 pack (column
   interleave may scatter)? Pure-offset view (free) / load-time repack (acceptable) /
   runtime dequant-slice (KILL TP=4; evaluate TP=2 at 1280).
5. MMQ sweep {2560 ref / 1280 / 640}: efficiency coefficient + parity (32-row AND 64-row
   points — 64 prices the TP=2 fallback free).
6. Wall-model recompute with measured coefficients. **GO criterion: projected TP=4 must
   clear 2,600 t/s at the PESSIMISTIC corner** (beat EP+PP+slice end-state ~2,400-2,600
   at worst case). Corner <= 2,600 -> TP=2/PP=2 (repriced free from items 3-5).
7. If GO, milestone order: attention slice (peer-side prompt attention + o_proj AR —
   ships only with net critical-path ms including transport; head ratios stay labeled
   model predictions) -> MoE TP substrate on the parity harness (pinned EP+PP = oracle)
   -> prefill wall vs the item-6 floor -> decode measured vs floors (solo >= 78 t/s,
   acc within -3 pts, hit >= 89; the 8% sync estimate gates nothing until measured).

Parked (formal): topology morph (split layout for compactions, tp2 for decode) —
kimi #2 P2-5: net 3-13 s/compaction best case; trigger: post-TP=4 prefill < 2,600 AND
switch <= 10 s with acc/hit unchanged. FP8-AR: rejected unless a ledger shows
AR > 15% of layer time. Speculative-prefill: instrumentation spike only.

In-flight at this writing (all blind-on-GPU, predictions filed):
- `tp2pp2-balance` (run 753f6bbe/801595cb): share-v2 time-objective + parameterized
  QsaHeadShard spans + unequal-span parity + decode contract rebase. Capped at one
  iteration; kill: pair wall > 135 ms, prefill < 2,200, or the row-rate gap
  (lead 2.26 vs peer 1.09 µs/row-visit — UNEXPLAINED) persists > 1.4x after balance
  (that proves lead-side contention and deflates every TP estimate).
- `tp2-tpstep0` (run 277918db): ledger + ring matrix + pack inspection (per §6.1-4).
- `tp2-draftseed` (run 58be06ee): suffix-drafter re-seeding from the generated tail —
  the 70k-generation class ran 40 t/s vs 78 (acceptance collapse, dl ~1.1 vs ~3.0).
  kimi #2 P2-4: highest-EV parked item (~1.9x decode on a real workload class).
  Kill: < 55% acceptance recovery after one session.

## 7. Operational trivia (saves an hour)

- Boot (tp2, from /home/gary/opt/basalt-tp2pp2):
  `/home/gary/opt/basalt/.venv/bin/python -m serve.server --engine-bin bin/basalt
  --model /home/gary/opt/basalt/models/Qwen3.8-FN-FP4Local.basalt --alias local
  --host 127.0.0.1 --port 8000 --gpu 0,1,2,3 --parallel 2 --slot-full-context
  --prefill 8192 --resident-budget-gib 37 --prompt-cache 8 --spec 8 --ple-io direct
  --kv int8 --max-context 262144 --adapt-swaps 96 --resident-experts
  --peer-reserve-mib 600 --decode-timing --suffix-draft 8 --grammar --short-read 512
  --tp-size 2 --peer-ram-rows --exp decode_profile=1`
  (--peer-rows-fp32 / --peer-acts-fp32 / --exp mtp_preswap=1 / --exp batch_lines=0 /
  --exp decode_headsplit=1 / --exp l2warm_force=1 are the A/B gates.)
- Logs: wrapper `/tmp/tp2-serve.log` (per-boot); engine
  `~/.cache/basalt/5059bef732e5da6c/engine.log` (append-across-boots — find the current
  boot at the last `peer GPU 1 computes` line). Grep: "peer chunk" (per-layer op
  breakdown), "peer fp4 crossover" (boot state), "basalt decode profile" (per-window
  stage phases), "s1 done" (request rates; NOTE: done-line t/s are wall-contaminated
  under concurrency — read the live line's `total`).
- Rules that bit us: NEVER run the GPU-labelled ctest suite while the server holds the
  cards (OOM'd once; server survived; run GPU tests in the boot window after kill);
  engine shutdown takes ~20 s (a fast re-boot races the port); engine.log is append-
  across-boots; done-line ttft includes the other slot's prefill.
- Worktrees: tp2pp2 (branch of record, balance worker in flight), tp2-fp4plane (merged),
  tp2-l2warm (merged), tp2-headsplit (merged), tp2-hostloop (merged, mtp_preswap),
  tp2-tpstep0 (ledger/ring, in flight), tp2-draftseed (drafter, in flight),
  tp2-decode (decode instrumentation, merged). Production tree /home/gary/opt/basalt
  untouched at 306f81e7 lineage; basalt-ctl exists but the tp2 server runs manually.
- Subagent lessons: workers die at 262k context (work economically: notes + diffs, no
  transcript re-reads); direct async children cannot resume after stop but COMPLETED
  runs revive via `subagent({action:"resume", id})` (used 3x successfully); steering a
  completed run's id revives it; the supervisor reply AND steer both (the reply alone
  did not always route); "desktop" is a MODEL (worker agent + model desktop), kimi-k3
  is `synthetic/hf:moonshotai/Kimi-K3` (bare `kimi-k3` resolves to the quota-dead
  opencode-go route).
- Bench kit: `/tmp/tp2_bench.sh PORT LABEL` + payloads /tmp/tp2_r1..r7.json
  (r1 = 122k fresh + 300 gen; the nat prompt /tmp/tp2_prompt_nat.txt ~122k tokens;
  random-word prompts starve the drafter — a confound). Kill pattern:
  `pgrep -f "^/home/gary/opt/basalt/.venv/bin/python -m serve.server"` (plain pkill
  hits unrelated processes).

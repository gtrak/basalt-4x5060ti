# Optimization plan — Basalt, Qwen3.8-FN-NVFP4, 4x RTX 5060 Ti

Status: **plan only, nothing implemented.** Every claim below is either read out of the source in this
checkout or was measured earlier on this box and is recorded in [WORKLOG](WORKLOG.md). No profiling was
done while writing this (the inference server was in use).

Numbering is `P0`…`P6`, ordered by (expected gain) / (risk × effort). Each item states the evidence, the
change, the expected effect, the cost, and the gate that decides whether it shipped.

---

## The measured starting point

| thing | value | source |
|---|---|---|
| decode, `--spec 8`, cold bench (conv/struct/prose) | 77 / 115 / 82 t/s | WORKLOG |
| decode, `--spec 0` | 52 / 54 / 60 t/s | WORKLOG |
| decode under live agent load, ~115k context | 60–106 t/s | `/metrics` |
| prefill | 501 t/s @6.7k, 832 @26.5k, 1263 @242k | server log |
| prefill chunk actually chosen | **8192** (`auto`) | engine log |
| KV @256k, int8 | 0.7/4.0 GiB elastic | boot log |
| expert slots (main card) | 3352–3483 | boot log |
| experts: VRAM / peer / RAM / **SSD** | 9.0 / 43.9 / **0.0** / **10.4 GiB** | boot log |
| pinned RAM arena | 42.0 GiB | `/metrics` |
| **cumulative SSD read by the live server** | **1.55 TB** | `/metrics` engine log |
| PCIe per card | 4.0 **x4** (device max Gen5, host max Gen4), 7.2 GB/s h2d. **P2P read AND write OK on all 12 pairs** (`nvidia-smi topo -p2p`) | `basalt-bench --hardware`, `nvidia-smi topo -p2p` |
| GPU-to-GPU, PIX vs SYS | 4.37 GB/s vs 2.39 GB/s allreduce bus BW (1 MB, 4 GPUs) — 83% | `../sglang/WORKLOG.md` 2026-09-09 |
| IOMMU | `amd_iommu=on iommu=pt` (passthrough, so DMA is not remapped) | `/proc/cmdline` |
| CPU | 5900X 12C/24T, **no AVX-512** | engine warning |

Two facts dominate everything else:

1. **10.4 GiB of experts (16%) live only on the SSD**, because the pinned RAM copy is planned without
   knowing about the peer cards. That is the single largest avoidable cost on the box.
2. **The prompt split is already priced on this hardware and switched off.** The engine prints, every boot:
   `58897 ms on CUDA0, 32266 ms with layers 9-20 on CUDA3, 21-33 on CUDA2, 34-47 on CUDA1
   (4 stages 32266 ms); --exp prompt_split=4 reads across them`. That is a **1.82×** prefill improvement
   the engine has already computed and declined only because 3+ cards are "not measured".
3. **GPU-to-GPU access exists on this box and the engine acts on it, but the prompt split stages through
   host anyway** — see P0b, which corrects an earlier wrong assumption in this plan (`topo -m` showing
   PHB does not mean no P2P; the sglang worklog already established that here, at 4.37 vs 2.39 GB/s).

---

## P0 — Free the RAM copy from the peer cards (fixes SSD spill)

**Evidence.** `pin_cache_complement()` (`src/core/expert_source.cpp:777`) pins "the experts the GPU cache
does not hold". Its skip list is `gpu_only`, assembled at `generate.cpp:4663` from `peer_only` +
`stage_held`, and `peer_only` is populated **only** when `--peer-exclusive` is set (`generate.cpp:4603`).
`--peer-exclusive` is itself rejected with more than one peer card (`generate.cpp:1701`). So with three
peer tiers the RAM copy is planned blind to all of them.

Confirmed by sweep: `--resident-budget-gib 12` and `20` both still report `file (SSD) 9.5 / 9.6 GiB`,
because the budget only re-sizes the redundant copy. `top_up()` (`generate.cpp:6660`) *is* peer-aware —
it skips `host_res` and `peer.has(l, e)` — so the two paths already disagree.

**Change.** Build `gpu_only` from the peer tiers' actual ownership regardless of `--peer-exclusive`, the way
`--peer-exclusive` already does for one tier (`generate.cpp:4603` pre-opens the peer to get its ownership
before pinning). The boot order is the reason the skip list is empty today: the complement is pinned
*before* the peer tiers are filled (the log prints `resident RAM mode: 45.17 GiB` before `peer GPU 1:
5708 experts`), so at pin time no peer ownership exists yet. The fix pre-computes the tiers' placement
(the fill is deterministic: profile-ranked, least-loaded tier) and skips those pairs. Well-scoped, not
quite one line — the ownership must exist before the pin, as the `--peer-exclusive` path already shows.

**Expected — larger than first estimated here.** The complement the copy should hold is the experts no
GPU holds: 63.3 − 9.4 (main cache) − 44.1 (peer tiers) ≈ **9.8 GiB**, so the pinned copy drops from
42 GiB to **~10 GiB, freeing ~32 GiB of RAM** (first draft of this plan said ~30 GiB total, which
under-counted). Two second-order effects: `file (SSD)` 10.4 GiB → **0** (removing SSD reads from the
miss path), and the RAM floor drops from 9.2 GiB to just the 3 GiB margin (`running.md`: "When every
expert has a place, only the margin is kept"), freeing ~6 GiB more. This is what makes P5 and P7
affordable — they are the point of the fix, not side effects.

**Cost/risk.** Low. Placement changes, so output token-for-token identity against today is not expected
(docs already say adaptation makes placement-dependent output); quality is unaffected.

**Gate.** Boot log shows `file (SSD) 0.0 GiB`; bench decode within noise or better; `basalt-bench
--quality` PPL/KLD against Q8_0 unchanged (needs the KLD base, see "Missing tooling").

---

## P0b — The prompt split's staging is host-staged even though P2P is available

**Correction to an earlier draft of this plan.** I recorded "no P2P" from `nvidia-smi topo -m` showing
`PHB` for every pair. That was wrong, and `../sglang/WORKLOG.md` had already proven it wrong on this box:
PHB means "same PCIe host bridge", the cards still talk **PIX** (single bridge), and choosing `SYS` over
`PIX` cost 83% of allreduce bandwidth (2.39 → 4.37 GB/s). `nvidia-smi topo -p2p r` and `-p2p w` both
report **OK on all 12 ordered pairs**, and the driver here is modded for this. IOMMU is `iommu=pt`
(passthrough), so DMA is not remapped.

**Evidence.** The engine *does* model peer access, as a coarse per-platform boolean:
`basalt::platform::gpu_peer_access()` returns `true` on Linux (`system.cpp:139`) and `false` on the
Windows body (`system.cpp:87`). It is consumed only by `choose_roles` (`gpu_select.cpp:225`), where
`false` makes every card after the main **idle**. So the engine never calls `cudaDeviceEnablePeerAccess`
and never asks `cudaCanAccessPeer` — it assumes "Linux has it, Windows does not". Nothing in `src/`
or `include/` enables peer access explicitly.

Meanwhile the hot cross-card paths already use `cudaMemcpyPeerAsync`:
`peer_experts.cpp:415` (`copy_to`), `prefill.cpp:1784` (staging into tier slots), and
`prefill.cpp:2472/2540/2597` (the cross-card prompt split). And `prefill.cpp:1392` states the intent
plainly: the split's activations and rows *"cross through **host staging** (`cudaMemcpyPeerAsync`)"*.
Without peer access enabled that peer copy is resolved host→device internally — DMA in and out — so it
does **not** take the PIX path.

**Why this matters beyond the split.** The multi-GPU price model is fitted to that transport:
`kExchangeUs = 5.4` is commented "one **mailbox** exchange of 10 KiB between two cards"
(`multi_gpu_plan.cpp:24`). Tensor parallel is priced at 24.7 ms/window against the expert tier's 27.0
and marked `runs = false` on that basis. TP here is therefore dismissed by a cost model that assumes the
slowest possible transport, not by the hardware.

**Change.**
1. Add a real check: `cudaDeviceCanAccessPeer` per ordered pair, then `cudaDeviceEnablePeerAccess` for the
   tiers, in the platform seam next to `gpu_peer_access()` so Windows keeps its own body.
2. Have the prompt-split staging (`prefill.cpp:1784`, `:2597`) use the peer copy directly when access is
   enabled, keeping the host-staged path as the fallback.
3. Re-derive `kExchangeUs` on this box instead of inheriting the mailbox figure, so TP is priced honestly.

**Expected.** P1's ceiling rises above the priced 1.82× (the split's 32.3 s includes host staging of
activations and result rows). The peer expert path and `copy_to` improve as a side effect. TP becomes
re-priced rather than "not built" — a research outcome, not a commitment.

**Cost/risk.** Medium. It changes transport under paths that currently work, and the price constants were
fitted against host staging, so every measurement taken before this lands is on the old transport. Note
`peer_exclusive` swaps route via `p2p`/`xchg` branches (`generate.cpp:6140-6229`) that must keep working
on both transports. Windows must not regress: its body returns `false` and its cards stay idle.

**Gate.** Same-boot A/B of the split with and without peer access; greedy-identity check (docs rule: a
layer gives the same bits on any stage of the same card model); `conversation_*` and `prefill_*` parity
tests green; a boot on Windows or with access disabled still starting with the idle warning.

**Sequencing.** P0b and P1 target the same path. Running P0b first means P1's A/B measures the transport
it will actually ship on; running P1 first gives a cheaper read on whether the split pays at all. Either
order is defensible — recommend P1's A/B first (pure flag, no risk) then P0b, so the split's value is
proven before transport is reworked.

---

## P1 — Turn the prompt split on, then measure it properly

**Evidence.** `generate.cpp:1406-1409`: `split_cap = peer_devices.size() <= 1 ? 2 : exp_flag("prompt_split", 1.0)`,
so on 4 cards the cap is 1 stage and the price prints "off on 4 cards until measured there". The printed
price already assumes the 4-stage split runs and is 1.82× faster than CUDA0 alone.

**Change.** Two steps, in this order:
1. `--exp prompt_split=4` for a bounded A/B on this box. This is an existing experiment knob, no code edit.
2. If it holds, change the gate at `generate.cpp:1409` so 3+ cards are allowed when the priced saving
   clears the existing 5% rule (the same rule 2 cards already use), and drop the "until measured" text.

**Expected.** Prefill ~1263 → ~2200–2400 t/s at long context on this hardware class (the fork measured
2,420 t/s at 220k with its own N-way prompt path; the engine's own price here says 58.9 s → 32.3 s).
Agent turns at 115k context currently spend 2–80 s in prefill, so this is the largest single win for
interactive use.

**Cost/risk.** Medium. `prompt_split` is an `--exp` flag precisely because it is unmeasured on 3+ cards;
it stages weights from a pinned RAM copy into tier slots, so it interacts with P0 (P0 must land first or
the RAM copy is short). Correctness risk is in which card reads which layers, not in numerics — docs say
a layer gives the same bits on any stage of the same card model.

**Gate.** Same-boot A/B prefill at 16k and 128k; greedy output identical to the single-card read
(the docs' own identity check); no regression in decode while a prompt reads.

---

## P2 — Raise the prompt chunk above 8192

**Evidence.** `auto_ceiling` starts at `max(8192, min(prefill_auto_max, max_context))`
(`generate.cpp:4365`) and is then clamped by `biggest_chunk()` to what the expert cache can lend. Chosen
chunk on this box: **8192**. `kMaxAutoChunk` is 32768 for packs whose blobs exceed 1.5 MiB, and this pack
qualifies (largest blob 2.76 MB). The fork measured the same effect from the other direction: clamping
24576 → 4096 cost 2159 → 1778 t/s at 16k on a 256k boot, "the four-sub-chunk loop overhead, not residency,
is the 18%".

**Change.** Sequence matters. Try `--prefill auto:16384` and `auto:32768` first (no code). If lend is the
binding constraint, raise `--vram-reserve-mib` slightly or free main-card bytes (P0 helps). Only if the
ceiling itself is wrong on 4 cards does `generate.cpp:4365` change.

**Expected.** Prefill +10–20% at 16–32k prompts, if lend allows. Must be measured against P1, since a
bigger chunk needs more staging and the two compete.

**Cost/risk.** Low mechanics, but it trades VRAM with the expert cache — a starved main cache is exactly
what bit the fork at 524K.

**Gate.** Prefill A/B at 16k/32k; expert slots must not drop below ~3200; no prefill allocation failures.

---

## P3 — Let the VRAM tier actually learn (it currently swaps zero)

**Evidence.** Every run on this box logs `adaptive tier 0 experts swapped into the VRAM tier`, and
`resident RAM: 42.10 GiB of experts in RAM, 0 exchanged with the VRAM tier`. Yet the online server has
read **1.55 TB** from the SSD. `--adapt-every` defaults to 1 only on a single GPU with `--serve`
(`generate.cpp:1502`: `!multi_gpu && !remote_expert_opt`); on 4 cards it defaults to **4**. The
`--expert-profile-save` flag exists and is never set here, so a cold expert stays cold forever.

**Change.** Two parts: (a) run with `--adapt-every 1` and `--expert-profile-save` so learned routing
persists across boots; (b) audit why swaps are 0 on this configuration — the profile pre-fills all 3483
slots, so the round may believe every slot is already optimal. That audit is the real work; the flag is
the cheap probe.

**Expected.** Fewer SSD blob reads, so better tail latency and a lower decode floor. Hard to predict
without the audit; the 1.55 TB counter says the headroom is large. This is the item most likely to change
the *shape* of latency rather than the mean.

**Cost/risk.** Low-medium. Adaptation moves experts between CPU and GPU, so output is not bit-identical
run to run — already true today. `expert-profile-save` writes a file (placement only).

**Gate.** SSD blob reads per 100k generated tokens falls; decode mean within noise, p10 improves.

---

## P4 — Consider `--relaxed-drafting` for sampled traffic

**Evidence.** Docs (`precision/features.md:257`) measure it on this model: P=0.5 gives **+5% at temp 0.6,
+6–10% at temp 1.0**, at a total-variation cost of 0.03–0.04 per token, and it *improves* measured task
scores (gsm8k 0.947→0.973, MMLU-Pro 0.682→0.714). Greedy requests stay exact.

**Change.** `"relaxed-drafting": 0.5` in the engine config, or per-request `relaxed_drafting` for
non-greedy clients only. Single-stream only (`--parallel` slots stay exact).

**Expected.** +5–10% on sampled decode. Note the acceptance is **request**-side and the standalone decode is
only part of a turn, so end-to-end gain is smaller for long prompts.

**Cost/risk.** Low to enable; it is a real, documented departure from the model's sampling distribution.
It is listed here so the decision is recorded, **not** as a default recommendation — that is the
maintainer's call, and it needs P6's quality gate to land first.

---

## P5 — Parked conversations in RAM (needs P0)

**Evidence.** `--conversation-cache-mib` defaults to **0 (off)**; `--prompt-cache 6` is on but its
checkpoints carry **no KV** (`ConversationCheckpoint` holds GDN state + PLE history + indexer tails,
~113 MiB, context-independent). Parking a whole conversation costs KV: int8 = 12,800 B/token over 12 QSA
layers, so 0.39 / 0.78 / 1.56 / 3.12 GiB at 32k / 64k / 128k / 256k, plus ~113 MiB.

Today there is no room: `MemTotal 60.7 GiB − 42.0 arena − 2.5 floor ≈ 16 GiB`, and most of that is load
and serving slack. **P0 frees ~12 GiB, which is what makes this affordable at all.**

**Change.** After P0, set `--conversation-cache-mib` to ~4096 with `--conversation-cache-slots 2–4`
(sized by observed session lengths: 2 slots at 256k, or 4–6 at 64k). Constraints verified in source:
parking is refused with a GPU layer split (`generate.cpp:1457`), and any park oversize is skipped with a
log line, never fatal.

**Expected.** Warm replay of a long conversation without re-reading it. The fork reports a 312k prompt
replayed in 12 ms from its checkpoint cache; even a portion of that is a large win for agent workloads
that re-send a long prefix. Today a cache miss at 115k costs 78 s (measured: r_0037).

**Cost/risk.** Low code risk, RAM cost only. Interacts with `--parallel` (slot KV also comes from the
expert cache).

**Gate.** Warm replay of a ≥100k conversation reuses ≥90% of the prefix; MemAvailable never dips under
the floor; no `skip parking` lines in steady state.

---

## P6 — Quality and regression harness (must precede any default change)

**Evidence.** None of P0/P3/P4 is quality-neutral by construction, and the repo's own rule is a PPL/KLD
gate against the Q8_0 reference through the decode path (`AGENTS.md`). That harness needs a llama.cpp
`.kld` base file which is **not on this machine**, so no quality number has been produced here at all.

**Change.** Produce the base once (`llama-perplexity --kl-divergence-base` on the Q8_0 pack), then keep
`basalt-bench --quality` as the gate for P0/P3/P4 and for any KV format change. Also record the KV
format decision matrix measured in WORKLOG (fp16 7.1 / int8 4.0 / k8v4 3.3 / q4_0 2.4 GiB at 256k) with
real quality figures, since `k8v4` and `q4_0` currently have **no** numbers anywhere.

**Expected.** No direct speed gain; it is what allows the other items to be adopted safely and what makes
`k8v4` usable to buy back ~0.7 GiB of KV ceiling.

**Cost/risk.** Time, and ~180 GiB of disk for the Q8_0 pack (117 GiB free — needs a cleanup first).

**Gate.** PPL/KLD within the repo's own noise band for every accepted change.

---

## P7 — The concurrency pool: 2–3 full-256k streams + up to 4 smaller, int8 KV

**Requirement** (operator): 2–3 concurrent full-context 256k streams in one pool, plus up to 4 smaller
streams, at int8 KV quality or better. All four GPU lanes are CPU-direct (no chipset hop), which helps
both PIX peer traffic and streamed-attention DMA.

**Why pure-VRAM KV cannot do it.** Main-card budget for cache+KV ≈ 10.1 GiB (16.3 total − ~4.8 fixed
weights − ~1.4 reserve/draft-head/later). One 256k int8 session costs ~3.2 GiB K/V + ~0.4 pooled indexer
keys + ~0.1 GDN ≈ 3.5–4.0 GiB (measured ceiling 4.0). With `--parallel N --slot-full-context`, the
session count is N slots **plus one solo session** (docs' kv line: `solo=262144 slots=2x262144`), so
3 full streams ≈ 12–16 GiB of ceiling against 10.1 GiB of budget. Three simultaneous full-256k streams
in VRAM do not fit with any expert cache left. Two do — barely (see Regime C).

**Per-session cost detail** (verified in `layer.cpp:425-475`): the indexer's pooled block keys are
`max_cells/idx_block + 2` rows × 512 B × 12 layers ≈ **0.4 GiB per 256k session, always in VRAM**, in
every mode; RoPE tables ~64 MiB (one owner, shared); GDN state 0.11 GiB; the K/V pools are the only part
a smaller format or streaming can shrink.

### Regime C — elastic KV, no streaming (keeps P1)

```
--parallel 7 --slot-full-context --max-context 262144 --kv int8   (+ P0)
```

Elastic (`running.md`): each slot backs 4096 cells at boot and grows by 4096 as its request needs, taking
cache slots and giving them back. Verified in code: `kKvElasticFloorBatch = 4096` (`generate.cpp:2681`);
elastic is on with peer tiers (a layer split turns it off, we run none).

- Boot cost: 8 sessions × 4096 cells ≈ 0.4 GiB → cache stays ~9.6 GiB (~3600 slots) while streams are
  small. The 4 smaller streams are nearly free and fully VRAM-resident.
- 2 streams at full 256k: 2 × 3.8 ≈ 7.6 GiB taken → cache ~2.5 GiB (~950 slots). Decode survives on the
  peer tiers (44.1 GiB) + pinned RAM — the fork ran 80–89 t/s with a *smaller* main cache than this.
- 3rd deep stream: only ~1.8 GiB of headroom left → it grows to roughly 120–150k tokens, then the cache is
  drained to the reserve. So Regime C delivers **2 full + 4 small + 1 partial**, not 3 full.
- **The prompt split (P1) works in this regime** — prefill speed is kept.

### Regime B — KV streaming (`--kv-resident`), all 7 slots deep-capable

```
--parallel 7 --slot-full-context --max-context 262144 --kv int8 --kv-resident 32768   (+ P0)
```

`--kv-resident N` keeps N cells of each layer in a VRAM ring and the **whole K/V in pinned RAM**; elastic
is off (its pools are a fixed set of pages, `generate.cpp:2679`). Per session: ring 32768 × 12.75 KiB ≈
0.4 GiB + pooled keys 0.4 (fixed) + GDN 0.1 ≈ **0.9 GiB at boot, whether the stream is deep or not** —
8 sessions ≈ 6.5–7.3 GiB, cache ~3 GiB (~1100 slots) permanently.

- All 7 slots + solo can run 256k conversations simultaneously: pinned RAM holds the full K/V,
  3 × 3.2 + 4 × ~0.4 ≈ **11 GiB** — fits the ~40 GiB freed by P0 with room for P5 parking.
- int8 quality is kept (streaming moves cells, it does not reformat them). k8v4 is refused with
  `--kv-resident`; int8 is not.
- Streams at or under 32768 cells are **not streamed** (`layer.cpp`: a context that fits N cells is not
  streamed) — the 4 smaller streams keep full VRAM speed.
- **The prompt split is OFF in this regime** — `generate.cpp:4732` names `KV streaming` as a reason the
  split stays off. Deep-stream decode pays PCIe on attention gathers past the resident window; the
  measured 4.4 t/s at 242k (VRAM) will drop further. Unmeasured on this box.

### The choice

| | Regime C (elastic) | Regime B (`--kv-resident 32768`) |
|---|---|---|
| simultaneous full-256k streams | 2 (+1 partial ~150k) | all 7 |
| VRAM cost at boot | ~0.4 GiB | ~7 GiB fixed |
| expert cache with 2 deep streams | ~2.5 GiB | ~3 GiB always |
| prompt split (P1) | **works** | **forbidden** |
| deep-stream decode at 240k+ | VRAM gather (4.4 t/s measured) | streamed over PCIe — measure |
| small streams (≤32k) | full speed | full speed (unstreamed) |

Both regimes need P0 first: as slots drain the main cache, the evicted experts need RAM homes, and the
copy's complement must grow to cover them (Regime C with 2 deep: ~17 GiB pinned; Regime B: ~16 GiB) —
impossible inside today's mis-sized 42 GiB copy.

**Change.** Config only, after P0: `basalt-ctl` gains a `BASALT_POOL` switch selecting the regime
(C default, B opt-in), `--parallel 7 --slot-full-context` shared by both. No engine code.

**Gates.** A concurrent-load bench the current `basalt-bench` cannot express (it is single-stream):
3 × 240k-token prompts + 4 × 16k, all at once, then per-slot decode t/s, TTFT, no request-stall warnings,
MemAvailable never under the floor. Regime C is default unless 3-deep concurrency measurably matters more
than prefill speed. A future item (research, not planned): lifting the prompt-split/KV-streaming
incompatibility upstream.

---

## P8 — Kernel-level work: measured as not-the-bottleneck, one candidate deferred

**The engine is already fused where fusion pays.** Verified in source: a persistent GDN mid-chain
kernel fusing qkv+z, the recurrence and the out projection into one launch, bitwise identical
(`mid_chain.cu`, `docs/unified-kernel.md`); a commit finisher whose in-window commits read the next
layer's projection weights into L2 while a combine waits for the CPU tier (`verify_kernels.cu`); the
device-planned MoE tail; a 3-stage TMA bulk pipeline for the VRAM expert share (`dec_moe.cuh:1836`,
`cp.async.bulk` + mbarrier, stages issued ahead); **thread-block clusters** in the GDN recurrence
(`dec_gdn_rec.cu`, 4-block cluster per value head, `map_shared_rank` exchange — running on our sm_120);
cp.async staged hyper-connection reads; programmatic dependent launches (201–298 per captured window);
the one-launch fused sampler; the two-stage Q6_K verify head; fused hc up+mix in prefill.

**The maximal fusion was tried and retired with data** (`docs/unified-kernel.md`): a whole-layer kernel
(qkv/z/recurrence/out + experts half + QSA attention half) was bitwise identical to the separate
kernels and **lost 12% tokens/s** (324.6 → 288.2, same session, 400 W): the experts half is bound by
cross-CTA dependency latency (58 us at T=3 against a 30 us byte floor), the QSA half serialises
projections the separate kernels overlap, 1.5 MB of SASS stalled 20% of PC samples on instruction
fetch, and two CTAs per SM do not fit (99 KB arena). Only the GDN middle and the finisher survived.
Anyone re-attempting layer fusion must start from that doc's lessons list — it is the checklist.

**On this box, kernel work is the wrong lever today.** Single-stream decode is CPU-pool-bound: measured
pool 58–64 ms/round against the GPU window price of ~27 ms. The GPU floor hides under the pool, so a
faster kernel changes nothing until P0/P3 move expert work onto the GPUs (worth up to ~2× on round
time, before any kernel is touched) or P7 batch windows amortise the dense path across slots.

**The one candidate worth keeping: FP4 tensor cores in decode.** Prefill already runs the NVFP4 experts
as W4A4 `mxf4nvf4` mma (`moe_fp4.cu:51`); decode's grouped/decode-once kernels compute the same E2M1
codes with dp4a (`vec_dot_nvfp4_q8_1`, `iq_kernels.cu:495`). At T ≤ 8 dp4a is bandwidth-bound, so mma
buys little; in P7 batch windows (7 slots × 8 tokens ≈ 56 rows) the window becomes compute-heavy and
the mma path could pay the way it does in prefill. It changes numerics (E2M1 activation rounding
against q8_1), so it carries the full repo gate: kernel parity test + microbench line, PPL/KLD through
the decode path, and a `docs/precision/` update — the prefill FP4 numbers (rel rms 9.4e-2 against
MMQ's 1.6e-2, on the unrounded reference) give the quality-cost shape.

**Deferred micro-items** (recorded so they are not re-derived): L2 access-policy windows pinning hot
expert blobs (L2 is tens of MB — a handful of 2.76 MB blobs, marginal); TMA in the depth-attention
gather; further read-ahead (already reads next-layer projections into L2). **Not worth chasing**:
full-layer fusion (measured loss), warp specialization on this silicon, fused GEMM+allreduce (TP is
not built and priced out).

**Gate for any kernel project.** The engine ships its own floor measurements: `--gpu-only-full` (the
true per-token GPU floor — the flag's own help says "quote this one"), `--profile`, `--decode-timing`,
`--stage-timing`. Run the floor under the P7 batch load first; chase kernels only if the floor-to-actual
gap is on the GPU side. Sequencing: strictly after P0/P3/P7.

---

## Deliberately rejected

- **`--mmap-experts`** (drop the pinned copy, let the page cache hold the tail). Measured: cold decode
  35/118/52 t/s against 77/115/82. Thrash on the sampled paths. Rejected.
- **Migrating to the strata-4gpu fork.** It is 1 main + 3 *tiers* with no tensor parallelism, i.e. the
  same architecture Basalt already has, on a Q4_K_XL pack. Its value here is ideas and pitfalls, not code.
  It also has no `cudaDeviceEnablePeerAccess` call anywhere, so it shares P0b's gap.
- **Tensor parallel across the cards *as currently priced*.** The engine prices it at 24.7 ms/window and
  does not build it, but that price assumes host-staged exchanges. P0b is what makes this a real
  question; until it lands, TP stays rejected on a transport assumption rather than on hardware.
- **`--kv q4_0` for the 256k budget.** Only 1.6 GiB better than int8 at 256k, and unquantified quality.
  P6 decides.

## What the fork taught us (transferred as pitfalls, not code)

1. **`--vram-reserve-mib` too low kills the boot** when context is large — prefill cuBLAS side-stream
   handles fail to allocate and the engine dies *before* READY with a line that looks like progress.
   We run 700; the fork needed 1500 at 256k. Worth adding to the memory-floor docs and the launcher.
2. **A killed loaded engine holds its pinned arena 10–40 s.** Relaunching immediately fails on
   `cudaMalloc`. `basalt-ctl restart` currently sleeps 2 s — that is too short and should wait for VRAM
   to read idle.
3. **The bound draft vocabulary file must be present** or head-split silently no-ops for ~10% decode.
   This pack ships `mtp/draft_vocab`, and our logs show the draft head over 106,299 tokens, so we are
   fine — but it belongs in a startup check.
4. **Consumer Blackwell has no thread-block clusters**, so cluster-based kernels silently take a slower
   fallback. Relevant to any future long-context work; nothing to configure.
5. **N-way prompt offload is worth +15% prefill** on their box — independent confirmation that P1/P2 are
   the right places to look.
6. **Same-boot A/B is the only real signal**; between-boot wobble is ±3%, and generated-token t/s moves
   with accept rate. Every gate above is written as a same-boot A/B for that reason.

## Suggested order

```
P0  (arena / SSD spill)      — foundation; frees ~32 GiB of RAM, unblocks P5 and P7
P6a (KLD base produced)      — so P1/P3 can be judged, not just timed
P1  (prompt split A/B → on)  — pure flag first, proves the split's value
P0b (real peer access)       — then raise P1's ceiling, re-price TP
P2  (chunk 8192 → 16/32k)    — retune after P1 moves the prefill paths
P7  (concurrency pool)        — config-only after P0; regime C default, B opt-in
P3  (adaptation audit)       — tail latency, needs its own investigation
P5  (parking, needs P0)      — RAM budget after P0 is known
P4  (relaxed drafting)       — policy call, last
```

P1's flag-only A/B comes before P0b deliberately: it costs nothing and tells us whether the split pays
before we rework transport under it. P0, P1 and P2 are one coherent prefill/placement push and should be
measured as a set; P3 is genuinely open-ended and should not block the others. Nothing here is
implemented yet.

---

# Appendix — research gathered

Everything below was collected while writing this plan, so it does not have to be re-derived. Sources
are marked `[src]` (read in this checkout), `[box]` (measured on this machine, recorded in WORKLOG or
the sglang worklog), `[fork]` (strata-4gpu), `[docs]` (this repo's docs).

## A. Code facts that decide the plan

| Fact | Where | Why it matters |
|---|---|---|
| `peer_only` is built only under `--peer-exclusive`, which is refused with >1 peer | `generate.cpp:4603`, `:1701` | P0: RAM copy is peer-blind on 4 cards |
| `top_up()` skips `host_res` and `peer.has()`; the initial pin does not | `generate.cpp:6660` | P0: the two paths already disagree |
| `split_cap` = 1 stage unless `--exp prompt_split=N` | `generate.cpp:1409` | P1: the split is gated off on 3+ cards |
| `gpu_peer_access()` is a per-platform boolean (Linux true, Windows false) | `platform/system.cpp:139`, `:87` | P0b: never asks the driver |
| No `cudaDeviceEnablePeerAccess` / `cudaCanAccessPeer` anywhere | `[src]` grep | P0b: peer access is assumed, not enabled |
| `kExchangeUs = 5.4` = "one mailbox exchange of 10 KiB" | `multi_gpu_plan.cpp:24` | P0b: TP priced on host-staged transport |
| `choose_roles` idles cards after main when `peer_access` false | `gpu_select.cpp:225` | P0b: the only consumer of the boolean |
| `auto_ceiling` starts at `max(8192, min(prefill_auto_max, max_context))` | `generate.cpp:4365` | P2: chunk chosen was 8192 |
| `kMaxAutoChunk` = 32768 for blobs > 1.5 MiB (this pack: 2.76 MB) | `generate.cpp:4336` | P2: 32768 is legal for this pack |
| `--adapt-every` defaults to 1 only if `!multi_gpu && !remote_expert_opt` | `generate.cpp:1502` | P3: defaults to 4 here |
| `ConversationCheckpoint` holds GDN/PLE/tails, **no KV** | `conversation_cache.hpp:85-101` | P5: `--prompt-cache` is not a KV cache |
| Parked KV lives in `SavedConversation.kv` | `conversation_cache.hpp:184-203` | P5: parking is the KV-in-RAM mechanism |
| Parking refused with a GPU layer split | `generate.cpp:1457` | P5: constraint |
| int8 KV = 1,056 B/cell/layer; q4_0 = 576; k8v4 = 816; fp16 = 2,048 | `kv_q8.hpp:24`, `kv_q4.hpp:35`, `layer.cpp:451` | KV budget table |
| PLE default is `PleIo::Direct` (unbuffered SSD, never in RAM) | `generate.cpp:2413` | PLE already correct |
| TMA 3-stage bulk pipeline exists for the VRAM expert share | `dec_moe.cuh:1836` (`cp.async.bulk` + mbarrier) | P8: TMA already used |
| Thread-block clusters run on sm_120 (GDN recurrence, 4-block/value head) | `dec_gdn_rec.cu:167-323` | P8: clusters available here |
| Full-layer kernel fusion tried and retired: −12% t/s, bitwise identical | `docs/unified-kernel.md` | P8: the ceiling of fusion |
| Decode NVFP4 = dp4a; prefill = `mxf4nvf4` W4A4 mma | `iq_kernels.cu:495`, `moe_fp4.cu:51` | P8: the one kernel candidate |

## B. Measured on this box

Decode (`basalt-bench -t conv,struct,prose`, cold, 180 W) `[box]`:

| config | conv | struct | prose |
|---|---|---|---|
| `--spec 0` | 52 | 54 | 60 |
| `--spec 8` | 77 | 115 | 82 |
| `--spec 8 --mmap-experts` | 35 | 118 | 52 |

Prefill `[box]`: 501 t/s @6.7k, 832 @26.5k, **1263 @242.7k**; chunk chosen 8192.
Decode at depth: **4.4 t/s at 242k**.
Live agent load, ~115k ctx: 60–106 t/s decode, 27–107 t/s observed range.

KV ceiling at ctx=262144 `[box]`:

| `--kv` | KiB/token | ceiling | backed at boot |
|---|---|---|---|
| fp16 | 24.38 | 7.1 GiB | 1.0 |
| int8 | 12.75 | 4.0 GiB | 0.7 |
| k8v4 | 9.94 | 3.3 GiB | 0.6 |
| q4_0 | 7.12 | 2.4 GiB | 0.5 |

128k → 256k at int8 costs ~3 expert slots (3483 → 3354) and 1.6 GiB of ceiling.

Placement `[box]`: VRAM 9.0 / peer 43.9 / RAM **0.0** / SSD **10.4 GiB**; pinned arena 42.0 GiB;
`adaptive tier 0 experts swapped` on every run; **1.55 TB cumulative SSD reads** on the live server.

Links `[box]`: PCIe 4.0 x4 (device Gen5, host Gen4), 7.2 GB/s h2d per card, P2P r+w OK on all 12 pairs,
`iommu=pt`. PIX 4.37 GB/s vs SYS 2.39 GB/s allreduce (83%), from `../sglang/WORKLOG.md`.

## C. What the strata-4gpu fork established `[fork]`

Repo: `johnmosesventura06/strata-4gpu`, branch `custom-4gpu`, base `custom@3a19944` (Strata, Basalt's
upstream). 1 main + 3 tiers, no tensor parallel — same architecture as Basalt.

Its speed ladder on 4x 5060 Ti but **PCIe gen5 x8/x4/x8/x4 and a 9950X**, `UD-Q4_K_XL`, MTP+lookup:

| step | decode |
|---|---|
| main + 1 tier | 49–53 t/s |
| main + 2 tiers | 72–78 |
| main + 3 tiers | 80–89 |
| + bound draft head | 91–94 |
| + head-split 0.40–0.75 | 94–99 |
| + N-way prompt offload | 92–94 sustained (99–110 short) |

Prefill 1878 → 2159 t/s at 16k with N-way offload; 220k prompt at 2420 t/s (was 2129–2175).
YaRN: 524K context boots and serves; decode 84–111 t/s at depth 225–312k.

**Pitfalls worth importing** (also in the plan body):

1. `--vram-reserve-mib` too low with big context → prefill cuBLAS side-stream alloc fails and the engine
   dies *before* READY, printing something that looks like a progress line.
2. A killed loaded engine holds its ~71 GiB pinned arena 10–40 s; immediate relaunch fails on
   `cudaMalloc`. Our `basalt-ctl restart` sleeps 2 s — too short.
3. Missing `data/draft_vocab.bin` in the MTP runtime silently no-ops `--head-split`, ~10% decode lost
   with one buried log line. (Our pack ships `mtp/draft_vocab`; draft head logs 106,299 tokens.)
4. `--head-split 1.0` is silently refused (guard wants `< 1.0`).
5. Consumer Blackwell (sm_120) has **no thread-block clusters**; cluster kernels silently take a slower
   fallback. Relevant to long-context selection work.
6. Between-boot wobble ±3%; generated t/s moves with accept rate. Only same-boot A/B is signal.
7. At 524K the real cost is the prompt chunk: staging scales with `--prefill`, and a starved main cache
   cannot allocate 24,576. Clamped to 4,096: 16k prefill 2159 → 1778 t/s *with the cache untouched* —
   sub-chunk loop overhead, not residency.

It has **no `cudaDeviceEnablePeerAccess`**, so it shares P0b's gap. `DESIGN-4GPU.md` documents the tier
protocol (kind table, per-tier flag2 lines, disjoint profile windows, round-robin prefetch) — useful
reference for how the tier fan-in is structured upstream.

## D. Options considered and their status

| option | status | reason |
|---|---|---|
| `--mmap-experts` | rejected `[box]` | 35/118/52 vs 77/115/82 — page-cache thrash |
| migrate to strata-4gpu | rejected | same architecture, Q4_K_XL, no magic |
| tensor parallel | re-open after P0b | priced 24.7 ms on host-staged exchanges |
| `--kv q4_0` for 256k | deferred to P6 | only 1.6 GiB better than int8, no quality data |
| `--relaxed-drafting 0.5` | P4, policy call | documented +5–10%, changes sampling distribution |
| `--parallel N` | untested | every number here is single stream |
| PLE in RAM | rejected | `direct` is the default and correct |

## E. Missing tooling / open questions

- **No KLD base on the machine**, so zero quality numbers exist for any config. Needs
  `llama-perplexity --kl-divergence-base` on the Q8_0 pack (~180 GiB; 117 GiB free). Blocks P6, and with
  it the safe adoption of P4 and any KV format change.
- **Why do adaptation rounds swap 0 experts here** while the SSD serves 1.55 TB? Profile pre-fills all
  3483 slots, so the round may believe placement is already optimal. Unresolved; P3.
- **Does PIX survive the 4-way all-to-all** rather than only pairwise? The sglang number is an allreduce,
  which is all-to-all, so presumably yes — but the modded driver makes this worth re-measuring.
- **Do `PeerExperts` and the prompt split share one device context** such that peer access applies to
  both? P0b needs this answered before the change is scoped.
- **`--parallel` concurrency** is now scoped by P7 but entirely unmeasured; `basalt-bench` is
  single-stream, so the gate needs a custom concurrent-load bench.
- **`basalt-probe` has never been run** (it needs the server stopped, and no maintenance window has
  been taken). It measures `--pool-workers` and `--prefill-cpu-share` suggestions plus the machine's
  rates in one report — the cheapest missing measurement on this list.

## F. Smaller levers not yet A/B'd (quick wins, low risk)

Each is a flag change or a one-line measurement; none justify a plan section but all belong on the
checklist once the server can be bounced:

| lever | default | why it might matter here |
|---|---|---|
| `--suffix-draft N` | 3 | prompt-lookup drafting: agent traffic repeats context heavily (code, logs); the fork ran lookup 16 and its bench accepted 6.9–7.5 of 8 drafts. Our bench showed 0 suffix windows because the fixtures do not repeat. |
| `--short-read N` | 64 | turns shorter than this skip the batched prompt path entirely; short agent turns get faster TTFT. |
| `--prompt-cache N` | 6 | checkpoints are ~113 MiB each and hold no KV; more resume points are nearly free after P0. |
| `--peer-reserve-mib` | 600 (min 128) | each ~470 MiB freed per tier ≈ ~150 more experts per card (~450 total across tiers). |
| `--vram-reserve-mib` | 700 (min 256) | ~400 MiB more main-card cache (~150 slots); the fork needed 1500 at 256k for prefill side-streams, so raise, not lower, if prefill allocs fail. |
| `--ple-row-cache N` | 1 Mi rows (~90 MiB) | long prompts touch more rows; RAM is plentiful after P0. |
| power limits | 180 W (max 220/198) | a maintainer decision, not an agent one: raising to max is a measurable decode gain on 180 W-capped cards; every number must then carry the new limit label. |
| `--no-grammar` | grammar on | grammar masking is enforced inside the engine per window (`generate.cpp:7583` resample under mask); tool-heavy agent traffic pays it. A/B `--no-grammar` on a tools benchmark before deciding. |
| SMT pool workers | 11 (physical cores − host) | the CPU pool is the decode bottleneck (58–64 ms/round); `basalt-probe` measures per-worker-count rates, so whether SMT helps is one probe run. |
| locked clocks | unset | sustained MoE decode on 180 W cards downclocks; `nvidia-smi -lgc` pins it (again: label every number). |

---

# Campaign log

Running record of the optimization campaign. Newest entries at the bottom.

## Entry 1 — P0 landed (with two detours that produced their own findings)

**Shipped change set** (working tree, uncommitted):

1. `generate.cpp` — tiers open **before** the RAM complement is pinned; the skip list covers every tier's
   ownership (was: only `--peer-exclusive`'s single tier). The complement now holds only the experts no
   GPU holds plus the prompt path's lend slots.
2. `generate.cpp` — `rank_swaps` filters adaptation candidates to **servable** ones (a RAM copy exists, or
   a p2p copy out of a tier can fetch them). A non-servable candidate used to read the mapped file on the
   adapt stream (2.76 MB random SSD read per swap) and cost its victim its RAM place too.
3. `generate.cpp` — p2p swap-in is legal by **ownership** (any tier), not by the `--peer-exclusive` flag;
   copies and slot reassignments dispatch to the owning tier. Currently inert in production because (2)
   excludes tier-held candidates anyway; it exists so adaptation can be re-enabled safely once (5) lands.
4. `generate.cpp` — the post-load RAM top-up skips **every** tier's experts (was: tier 0 only). Found
   because the first production boot ballooned the complement to 48.7 GiB and drove the box into swap.
5. `basalt-ctl` — `--adapt-swaps 0` default for multi-tier (see economics below).

**Production placement now** (ctx 256k, 4 tiers): VRAM 8.6 + tiers 44.2 + RAM 10.5 + **SSD 0.0** GiB;
complement pinned **19.09 GiB** (was 42–48); `RAM 33.90 GiB free` after load (was 5.16); 20,513 of
24,576 experts on the GPUs; 33 GiB available to the OS.

**Decode, interleaved same-session A/B, adaptation OFF on both sides** (cold, 180 W):

| fixture | stock | campaign build | delta |
|---|---|---|---|
| conv | 80.0 / 78.4 | **124.7 / 117.7** | **+50%** |
| struct | 83.3 / 86.3 | **118.9 / 117.6** | **+39%** |
| prose | 87.0 / 72.2 | **95.8 / 91.9** | **+16%** |

File reads during the bench: **0 blobs** (stock: ~100 GB — the 10.4 GiB SSD-resident set plus adaptation
churn, silently taxing every window).

**Gates passed.** Greedy identity A/A on the build: IDENTICAL. Single-card pinned old-vs-new (all
changes dormant): IDENTICAL. Multi-card old-vs-new differs — expected: placement moved, and the docs
specify identity only within one configuration. Server verified end to end at 256k through tinyllb.

**Findings made on the way (each is its own plan item):**

- **The redundant 42 GiB copy was load-bearing.** Three code paths consumed it implicitly: adaptation
  swap-ins (file reads without it), victim eviction (no RAM place without it), and the prompt path's
  expert sourcing. Removing it surfaced all three; two are fixed, one is not (next bullet).
- **The prompt path is tier-blind (open, sizeable).** `FileExpertSource::blob` backtraces during a
  98-token prompt showed `launch_mul_mat_q` (the MMQ prompt kernels) reading **tier-held** experts from
  the file: the prompt path checks the main cache and the complement but not the tier cards. Affects
  every prompt. The fork's answer was a "held-anywhere skip table" + ownership split (its commit
  4878ee5) — the same shape needed here, plus d2d copies for the rows. This is the next code item; it
  should also re-enable adaptation profitably (with prompt reads fixed and p2p live, swap churn loses
  its file-read component).
- **Adaptation economics on the new placement.** With every expert placed, adaptation's swaps now read
  tier-held candidates from files (prompt-path + swap-in), which costs more than the promotion gains:
  adapt ON gave conv 68–74 / struct 127–145 / prose 68–72 against OFF's 111–122 / 119 / 95–96. OFF is
  the default until the prompt path is tier-aware. Note the stock build's adaptation was a large
  **struct** win (78.7 → 138 with it on) — the follow-up should aim to keep that.
- **The stock build's "0 experts swapped" lines were misleading.** Its adaptation moved tens of
  thousands of blobs; the counters that read zero measure a different thing (src-side exchanges).
- **Boot-to-boot tier sizing wobbles ±33 experts/card**, and tier rooms depend on whether the link
  probes ran before the rooms were measured (context creation ~90 MiB). Placement is deterministic per
  binary; identity gates must pin or match placements.
- **Two engine logs to read together**: `~/.cache/basalt/<model-hash>/engine.log` accumulates across
  boots — grep by tail, not by pattern.

**Not yet done from the plan:** P1 (prompt split A/B — now unblocked, and should be re-priced after the
prompt-path fix), P2 (chunk), P6a (KLD base — still blocks quality judgement), P5 (parking — the RAM
is now there: ~33 GiB free), P7 (concurrency pool regimes).

## Entry 2 — P1 and P0b measured and rejected; P0 verified prefill-neutral

**P0 is prefill-neutral.** Committed build vs the commit before it, one boot each, 64k-token prompt:

| build | 64k prefill |
|---|---|
| c750 (pre-P0) | 1372.7 t/s |
| P0 (committed) | **1368.7 t/s** |

So P0's 30 GiB of freed RAM and 10.4 GiB of removed SSD spill cost nothing at depth. Warm replays
(`7 fresh` of 66,688) run at ~0.02 s in both, as they should.

**P1 (prompt split) measured: rejected.** `--exp prompt_split=4` engages exactly as the boot price
promises — CUDA1 holds layers 19-47, CUDA2 11-36, CUDA3 6-23, "up to 4 stages over CUDA0 and 3 tier
cards" — and measures **1060 t/s against 1374 baseline (−23%)**. The boot-time estimate (58,896 ms on
CUDA0 against 32,297 ms across stages, 1.82x) inverts on the real workload. That is the third
independent sign this box's cross-card staging prices optimistically: host-staged activations and rows
cost more than the price model's `kExchangeUs` mailbox figure allows for. P1 stays off; the flag remains
the A/B.

**P0b (peer access) measured: rejected, and it explained a 3x prefill collapse.** Enabling
`cudaDeviceEnablePeerAccess` for all 12 ordered pairs succeeded, and a standalone CUDA test showed the
driver's peer copy is genuinely fast (d2d 6.6 GiB/s against h2d 6.7 GiB/s, both at 2 MiB and 256 MiB).
But in the engine it collapsed 64k prefill to 470 t/s (from 1374), reproducibly. Tier-aware prompt
sourcing over d2d was built on top and reverted; the peer-access block itself was then reverted, and the
collapse went with it — confirming the cause was peer access, not the sourcing code. Note the confound
that made this expensive to isolate: the collapse appeared at 64k but looked like a win at 16k
(851 vs 708 t/s), and the auto-chunk differed between the two boots (29184 against 16640), so several
A/Bs compared different workloads before the forced-chunk controls (`--prefill 16640`: 311 t/s against
983) showed the effect was real and chunk-independent.

**Methodology corrections, so the next session does not repeat them:**

- `basalt-bench` re-booted the engine per measurement *and* its auto-chunk follows boot-time lend
  headroom, so its numbers moved for reasons unrelated to the change. The fast harness
  (`/tmp/pp.py`: one boot, N prompts over pipes, reads the engine's own per-request rate lines) is
  ~90 s per build instead of ~15 min and holds the chunk fixed. Use it first, then confirm with the bench.
- Peer-access effects must be measured at depth, not only at 16k: the sign flipped between depths.
- The engine's per-request lines go to **stderr**; the harness must read there.
- A 276 KB single-line prompt is fine on the engine's stdin reader (unbounded), but the writer must be
  connected by pipe at exec time — a pre-written file is never read.

**Still open, in priority order:** the depth-prefill mystery (see next entry), P2 (chunk), P6a (KLD
base), P5 (parking — 33 GiB free now), P7 (concurrency pool), P3 (adaptation economics).

## Entry 3 — depth-first on peer access: **Entry 2's peer-access verdict was WRONG**

Reopened at the operator's instruction, and it paid off. The chain of evidence:

1. Raw d2d is genuinely fast on this modded driver: **6.64 GiB/s** at 256 MiB and 6.51 GiB/s at 2.76 MB
   blob size, against h2d 6.73 GiB/s — so peer copies are not slow.
2. Enable **ordering** is not a factor: enabling before or after the allocations gives identical rates.
3. Peer access does **not** disturb h2d (6.73 either way) or the CPU pool's pinned reads (30.5 vs 29.7
   GiB/s) — the two things a naive "peer access slows prefill" story would blame.
4. A one-variable build (committed + peer access only) measures **1366 t/s at 64k against the baseline's
   1370** — neutral, and decode-neutral to slightly positive (conv 115.7/127.0 against 120.6/113.0).

So the earlier 3x collapse was **not** peer access. Two causes in my own methodology, both now fixed:

- the "P0b" 470 t/s readings were **the d2d tier-sourcing build**, and my attempts to re-test it were
  defeated by a *stale binary* left from a failed compile (`my earlier "peer access only" rerun` printed
  the same 470 three times, which should have been the tell);
- the auto-chunk differs between boots (29184 against 16640), so several A/Bs compared different
  workloads before the forced-chunk controls.

**Peer access is kept** (commit `9c82c20f`) — it is free, it is what the split's and tier-copy prices
assume, and it is what makes d2d real. Windows still returns `false` from `gpu_peer_access()`.

**d2d tier sourcing stays off, for the right reason this time.** Not because of a collapse: because
prefill at depth reads the *same* experts every chunk, so the file path is served from page cache at RAM
speed (48 GB/s) while PCIe is 7.2 GB/s. The tier-aware sourcing is correct at 16k (cold cache) and
correct for *rare* copies; it is wrong for a repeated deep-prefill walk.

**Adaptation economics overturned (the real prize).** With candidates restricted to the RAM complement,
adaptation is now a clear win on every fixture, interleaved, cold, 180 W:

| | stock engine | this build, adapt OFF | this build, adapt ON |
|---|---|---|---|
| conv | 77–81 | 114–116 | **117–120** |
| struct | 128–155 | 117 | **196** |
| prose | 55–85 | 91–94 | **98–101** |

Promoting *tier-held* experts is what cost performance (conv 119.7 → 97.1, struct 196.4 → 163.7, prose
98.1 → 87.0): the tier computes its own experts beside the main card, so promoting one concentrates the
work. That is now a comment in the code and a commit. The launcher's `--adapt-swaps 0` default is
reverted to 96.

## Entry 4 — P7 concurrency: **PASSED at the operator's target**

`basalt-ctl` now runs `--batch 3 --max-context 262144`: the startup line reads
`par=3 ctx/slot=262144, others queue` and `kv 2.1/12.4GiB cuda0 elastic slots=3` — an on-device pool of
**786,432 tokens**, three concurrent streams of the full 262,144 each, which is the stated requirement
(2+ streams with a >500k pool; elastic, so idle slots back only 4096 cells each).

**Three concurrent ~99k-token requests: all completed, 774 / 773.5 / 773.5 t/s, wall 128.1 s** — the same
wall time a single stream would take, i.e. ~3x aggregate throughput. This is the strongest result of the
campaign and it inverts the picture from P1, where the *split* (sharing one prompt across cards)
measured −23%: independent streams on independent cards scale, a serialized split does not.

Pool headroom by KV format at `--batch 3 --max-context 786432`: int8 leaves 1695 expert slots, k8v4 1722,
q4_0 2433. Every format fits three streams; q4_0 is the headroom option once P6 measures its quality.

## Entry 5 — sweeps, limits, and what is blocked

**Speculation window: 8 confirmed optimal** (draft window capped at 8; sweep on `struct`, 400 tokens,
2 reps, one boot each):

| `--spec` | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| tok/s | 70.6 | 119.4 | 153.4 | 187.8 | **195.6** | — | — |

10 and 12 land at 105 (capped, wasted window): the shipped default stands. `--spec 1` at 70.6 shows
speculation is worth 2.8x on the JSON fixture.

**P7 concurrency is the headline result** (Entry 4): three concurrent 99k-token streams at 774 t/s each,
wall time equal to a single stream — ~3x aggregate. The contrast with P1 is the campaign's clearest
lesson: **independent streams on independent cards scale; a serialized prompt split does not** (P1 was
−23% while doing the same total work across the same four cards).

**Physical limits — what could not be measured, and why (recorded so nobody retries the same wall):**

- `--gpu-only-full`, the engine's own "true per-token GPU floor" flag, is **refused with multi-GPU**
  (`generate.cpp:1736`), so the one measurement that would isolate the GPU ceiling from the CPU pool is
  unavailable on a 4-card box.
- `--no-pool` (the GPU-only floor) cannot run on this pack: `--spec needs the device residency table`
  and `verify: needs the profile-filled VRAM expert tier` — the NVFP4 pack's graphed paths require the
  pool's residency table, so there is no pool-free configuration to compare against.
- **The quality gate (P6) is blocked outright**: it needs llama.cpp's Q8_0 reference GGUF (178 GiB in 6
  shards; 116 GiB free) plus `wiki.test.raw` (absent). No GGUF exists anywhere on this machine. The API
  exposes no `logprobs`, so the "same top-1" substitute is not available either. **Consequence: no
  quality number exists for any configuration tested in this campaign**, including the KV formats and
  `--relaxed-drafting`. Every performance number here is a speed number only, and the operator should
  treat KV-format and drafting choices as unvalidated on quality until that base is produced.
- Boot hygiene, confirmed twice: a running server starves the bench (`hc8: staging the
  hyper-connection matrices failed`). `basalt-ctl bench` now stops the server first.

**Net effect of the campaign** (cold, 180 W, interleaved A/B against the stock engine):

| fixture | stock | campaign build |
|---|---|---|
| conv | 77–81 | **117–120** |
| struct | 128–155 | **193–196** |
| prose | 55–85 | **98–101** |

plus 30 GiB of RAM freed, SSD spill eliminated (10.4 GiB → 0), and a 786,432-token on-device KV pool
serving three concurrent full-context streams.

## Entry 6 — KV offload to host RAM, and why prefill is stuck near 1500

**`--kv-resident` is Basalt's HiCache: the whole K/V in pinned host RAM, a VRAM ring in front.** Per
262144-cell session the host cost is **int8 3.09 GiB / fp16 6.00 GiB** (measured), and the ring occupies
VRAM up front because streaming replaces elastic KV. Measured at `--parallel 6`: 384 expert slots left;
at 4 slots, 1429.

**It is mutually exclusive with `--parallel`.** The engine refuses with
`--batch, stage 1: verify: batch windows need resident INT8 K/V without rotation (--kv int8)`. So the two
ways to serve many long streams are a genuine either/or:

| | A: live slots, KV resident in VRAM | B: KV streamed to host RAM |
|---|---|---|
| flag | `--parallel N --slot-full-context` | `--parallel 1 --kv-resident N` |
| live full-context streams | 4 measured (6 fails: draft logits + RAM) | 1 at a time |
| long contexts in all | bounded by VRAM (~6 slots × 3.7 GiB) | unbounded by VRAM, priced in host RAM |
| expert cache at 4 slots | 1429 slots | n/a |

**Two limits found at 6 live slots** (both real, both recorded): `mtp: the draft logits do not fit`, and
RAM exhaustion (6 × 3.09 GiB host KV plus the 19 GiB expert complement against 32.7 GiB free). 4 is the
measured working maximum at 262144/slot; the launcher default is now 4 with A, and `BASALT_RESIDENT` +
`BASALT_PARALLEL=1` selects B.

**bf16 KV is viable** under B: 6.00 GiB of host RAM per 262144 session, leaving 1085 expert slots at 3
slots. Under A it is not (VRAM-resident fp16 would be ~6.1 GiB per session on top of the expert cache).

**Why prefill is stuck ~1500 t/s while a 2-card 60 GB machine reports 2370** (community table, IQ3_S,
2× 5060 Ti). A card-count sweep at 16k, same build, same box:

| cards | 16k prefill | decode (struct) | peer | pinned RAM |
|---|---|---|---|---|
| 1 | 812 | 29.0 | — | 40.9 GiB |
| **2** | **1669** | 99.0 | 14.6 | 40.0 |
| 3 | 1662 | 78.4 | 29.4 | 25.2 |
| 4 | 1508 | **151.6** | 44.2 | 10.5 |

**More cards make prefill slower and decode faster.** Prefill streams experts every chunk and its 60% CPU
share reads them from pinned RAM at 48 GB/s; each card that takes experts off that fast local tier moves
them behind 7.2 GB/s PCIe. P0 made this visible by deleting the duplicate pinned copy: the 2-card config,
which keeps 40 GiB pinned locally, is the fastest prefill shape on this box, while 4 cards is 5.2x the
1-card decode. The fix is the fork's **N-way prompt offload** — ownership-then-spread expert split with a
held-anywhere skip table, so each tier computes the prompt rows of the experts it already holds instead of
streaming them — worth +15% prefill there and the reason it reaches 2159 t/s at 16k. That is the single
largest remaining prefill item and it is a feature, not a flag.

**Consequence for the launcher:** 4 cards stays the default because decode is the interactive axis
(151.6 against 99.0), and prefill on this hardware is settled around 1400-1670 t/s until N-way prompt
offload lands.

## Entry 7 — the "2500 t/s" comparison: same hardware, different model and host

Corrected twice, and the final answer is that the comparison is not like-for-like.

**It is the same 4x 5060 Ti** (my first claim that the fork ran different cards was wrong). But:

| | strata-4gpu fork | this box |
|---|---|---|
| cards | 4x RTX 5060 Ti | 4x RTX 5060 Ti |
| **pack** | **UD-Q4_K_XL** | **NVFP4 (122 GiB)** |
| CPU | 9950X, **AVX-512** | 5900X, **AVX2 only** |
| RAM | **128 GB DDR5-5000** | 60 GB DDR4, 48 GB/s |
| PCIe | **gen5** x8/x4/x8/x4 | **gen4 x4** all four |

Its 16k prefill (1878 cold, 2159 with N-way prompt offload) is on a pack **~25 GiB smaller and with a
faster CPU, more RAM and a wider link**. Those three host differences land exactly on prefill, which is
CPU-pool- and RAM-bound: 60% of streamed experts are computed on the CPU from pinned RAM.

**exllamav3 on this same box is the more useful comparison, and it is not like-for-like either.** Its
`WORKLOG.md`: *"Prefill scales 649 t/s (2k) -> 1524 (8k) -> 2655 (16k solo)"*. But it runs a
**3.05-bit exl3 quant, 85 GB**, against our **NVFP4 pack at 122 GiB** — a 37 GiB difference, which is
about the size of the expert set that has to stream past the host on every prompt chunk. It also
tolerates `chunk_size 512` where our small-chunk behaviour collapses (measured below).

**Measured on our box, to close the question:**

| lever | result |
|---|---|
| `--prefill` chunk sweep at 16k | 2048: 358 | 4096: 483 | 5120: 571 | 16384: 740 | **32768: 1502** |
| `--prefill-cpu-share` at 16k | 0.0: 1318 | 0.3: 1518 | 0.6: 1513 | 1.0: 1507 |
| experts moved peer -> RAM (`--peer-reserve-mib 3000`) | 1505 -> **1559** (+3.6%), peer 44.2 -> 37.2 GiB |

So: small chunks are *worse* here (the opposite of exllamav3's regime, because Basalt streams every
expert for a chunk at or above 6,144 tokens and stages routed-only below it, and the loop overhead
dominates), the CPU share is already at its optimum at the default 0.6, and the peer/RAM split barely
matters. **~1500 t/s at 16k is the practical prefill ceiling for a 122 GiB pack on this host**, and the
remaining lever is the fork's N-way prompt offload (a feature: each tier computes the prompt rows of the
experts it already holds instead of streaming them), not a setting.

**A pack-level lever worth naming:** the three smaller packs on Hugging Face (GSQ-RCO IQ3_S 81 GiB,
IQ3_XXS 74 GiB, Q2_0 65 GiB) would stream 40-57 GiB less per chunk, which is exactly the quantity prefill
is bound by. That is the single highest-leverage prefill change available on this box, at a quality cost
that **cannot be measured here** (see Entry 5: no KLD base, no logprobs).

## Entry 8 — pipeline parallel is implemented and loses to the expert tier here

The operator's question: exllamav3 ran pipeline parallel and did not stream experts to one card, so why
does Basalt? **Basalt has it** — `--layer-split` (auto-selected by the price model, no user flag; a
`BASALT_FORCE_SPLIT=1` opt-in was added to measure it). Each stage holds its own layers and its own expert
cache, so nothing needs to stream. Measured, same build, 4 cards, 16k prompt:

| | expert tier (default) | layer split |
|---|---|---|
| prefill 16k | **1611 t/s** | 699 t/s (2.3x slower) |
| decode (struct) | 167.0 t/s | **167.4 t/s** |
| **experts in VRAM** | **8.6 GiB (one card)** | **48.9 GiB (all four)** |
| SSD reads per prompt | 0 | 0 |

**The split places experts 5.7x better and still loses on prefill**, because its stages hand work off
sequentially while the expert tier has all four cards compute every layer's experts at once (each taking
the rows of the experts it holds). So the architecture that keeps experts where they live costs 2.3x on
prefill here, and the price model's 31.2 ms against 27.0 ms for the tier is measuring the right thing.
That is where the fork's N-way prompt offload differs from both: one runner per tier, ownership-then-spread
expert split, per-runner loans — parallel, not serialized. It is the shape that would beat both, and it is
a feature.

**Also found and fixed (the real prefill defect):** at the chunks that bench best (`>= stream_all_min`,
6144) the stream-all walk read **every** non-resident expert through the host — the peer's row-share is
capped at `stream_all_min() - 1` and only tier 0 was ever wired to the ring. A 16k prompt therefore read
**17161 blobs / 47.5 GiB from the SSD** for experts sitting in tier VRAM. Wiring every tier's cache into
the prompt path (`set_ring_remotes` replacing `set_ring_remote`, d2d from the owning card) gives
**1502 -> 1621 t/s and 17161 -> 0 file reads**. Note this is the same d2d path I rejected in Entry 2:
that rejection was made against a **stale binary**, and re-testing it with peer access on shows it wins.

**Pack-level finding, correcting Entry 7's arithmetic.** `per_layer_token_embd.weight` — the PLE table —
is **50.66 GiB, Q8_0, one tensor**, which is 41% of the 122 GiB file. Our actual model weights are
~69 GiB against exllamav3's 79.3 GiB (3.05-bit exl3): we run a *smaller* model and a *larger file*. The
table is streamed per row from the SSD (`--ple-io direct`) at 3-7 ms per prompt and never enters RAM, so
it is not a prefill lever — but it means the three smaller packs (IQ3_S 81 GiB, IQ3_XXS 74, Q2_0 65) only
shrink the expert set, not the table: 63.28 GiB of experts -> ~28 GiB on IQ3_S is the real prize there.

## Entry 9 — prefill is PCIe-bound; the parallel architecture is the only route left

**The N-way row-compute hypothesis, tested and rejected.** The fork's prompt offload has tiers *compute*
rows for experts they hold. Basalt caps that (`kPeerFp4Tokens = 5120`), so at the chunks that bench best
(>= 16384) tiers compute nothing. Raising the cap to 32768 and measuring: **1106 t/s against 1611** — the
source's own reasoning holds (`it grows with the chunk while the blobs it saves do not, so a longer chunk
streams the blobs instead`): the per-layer activation round-trip scales with T while the saved bytes do
not. Reverted. So "tiers compute rows" is not the lever at these chunk sizes.

**Card count re-measured with the d2d fix — the earlier conclusion was an artifact.** Entry 7's "more
cards make prefill slower" was entirely the 47.5 GiB of SSD reads; with tier-held experts copied d2d:

| cards | prefill 16k | decode (struct) | file reads |
|---|---|---|---|
| 2 | **1651** | 81.8 | 14 |
| 3 | 1634 | — | 0 |
| 4 | 1613 | **148.2** | **0** |

Prefill is now **flat across card counts** (1613-1651, i.e. within noise of each other) while decode needs
all four cards (148.2 against 81.8 — the expert tier's four cards computing one layer's experts in
parallel). Four cards therefore remains right: decode gains 81%, prefill loses 2%.

**Why prefill is flat, and what it is actually bound by — measured arithmetic.** A 16k prompt streams
24,576 - 3,353 = **21,223 experts = 54.6 GiB** into the main card. At one PCIe link's measured 6.6 GiB/s
that is **8.3 s against a 10.3 s measured prefill**: roughly **80% of prefill time is the transfer of
expert bytes over PCIe**. Everything else measured this campaign (chunk size, CPU share, placement,
adaptation) moves the remaining 20%. This is the wall, and it is why the fork needed gen5 links and why
their N-way prompt offload helps (it computes rows where the experts already are, cutting the bytes moved,
rather than making the transfer faster).

**The lever that follows from it: fewer expert bytes per chunk.** 54.6 GiB is 86% of the 63.3 GiB expert
set, so prefill moves nearly the whole expert set for every prompt of 16k or more. Cutting the expert set
is the only lever of the right size:

| pack | experts | bytes/prompt | est. prefill |
|---|---|---|---|
| NVFP4 (current) | 63.3 GiB | 54.6 GiB | 1613 (measured) |
| IQ3_XXS | ~40 GiB | ~33 GiB | ~ 2,300 |
| IQ3_S | ~28 GiB | ~23 GiB | ~ 2,900 |
| Q2_0 | ~20 GiB | ~16 GiB | ~ 3,400 |

(Estimates scale the 80% PCIe share only; the other 20% and the quality cost are unknown here — no KLD
base, see Entry 5.) **A smaller pack is the largest single prefill win available on this box**, and it is
a download plus a config change, not code.

**The parallel architecture, stated properly.** The fork's N-way design is *not* "tiers compute rows"
(measured above as a loss) — it is one runner per tier with an ownership split and per-runner loans, so
each tier's prompt work stays on its own card and only activations cross. That cuts the 54.6 GiB of
*weight* bytes to a fraction of activation bytes, which is the size of the win. Implementing it in Basalt
means a `TierSet` in the prompt path, replacing the single `PeerPrefill` monolith (40 members, one device)
with per-tier runner instances, an ownership-then-spread expert split, and a held-anywhere skip table for
source selection. The fork's own commit for this is ~350 lines across 5 files against a simpler codebase;
Basalt's `PeerPrefill` is larger and its MMQ/FP4 paths are already dual, so this is a multi-session
feature, not a config change.

## Entry 10 — pipeline parallel wins; it was configured wrong, not slow

Entry 8 concluded the layer split "loses 2.3x on prefill". **That was a chunk-size artifact of my own
test setup**, and the operator's instinct was right: the split is the better architecture here.

| | expert tier | **layer split (PP)** |
|---|---|---|
| prefill 16k, chunk 12288 | 1611 t/s | **2354 t/s (+46%)** |
| decode (struct) | **181.8 t/s** | 167.8 t/s (−8%) |
| **experts in VRAM** | 8.6 GiB (one card) | **48.9 GiB (all four)** |
| SSD reads | 0 | 0 |
| 3 concurrent 99k streams | 774 t/s each | **815 t/s each** |

Chunk sweep (split): 4096: 2249 | **8192: 2338** | **12288: 2354** | 16384: 2343 | 32768: 655.
The earlier 699 t/s was measured at a 32768-token chunk, where the per-stage handoff dominates. At the
chunk the split actually wants (8192-16384) it is **+46% on prefill** and sets the campaign's best prefill
number, while also placing 5.7x more experts in VRAM — the thing the operator asked for.

**Why it works, in one line:** with a layer split the prompt path never has to move expert bytes. Each
stage computes its own layers with the experts already in its own cache; only activations cross the link.
The expert tier instead streams 54.6 GiB of experts into card 0 per 16k prompt (Entry 9), which at PCIe
gen4 x4 is 8.3 s of a 10.3 s prefill. **Pipeline parallel is not faster because of the links — it is
faster because it does not use them for weights.**

**Enabling it.** The price model always picked the expert tier (31.2 ms against 27.0 ms, and it is now
clear that price is fitted to the wrong regime), and the split had no user flag. `BASALT_FORCE_SPLIT=1`
now selects it (commit `7c0aa1a8`), and `basalt-ctl` defaults to it with `--prefill 12288` and
`--resident-budget-gib 38`.

**Bonus: the split fixes the KV spread.** `[server] kv` on the split boot reads
`4.8GiB cuda1 solo=262144 slots=4x262144`, `cuda2`, `cuda3`, `cuda0 ... draft=1.6GiB` — the sessions are
distributed across all four cards instead of piling 16.3 GiB onto card 0. Verified live: four concurrent
full-262144 slots (a 1,048,576-token device pool) and three concurrent 99k-token streams at 815 t/s each.

**Still pairs with `--parallel`** (unlike `--kv-resident`), so the live-stream requirement is met on the
better architecture.

**What remains.** Decode is 8% behind the tier (167.8 against 181.8) — the split's window hands off between
stages. The levers there are the overlapped-windows path the docs describe for two alike cards (not
exercised on four) and tuning `split_at`. Prefill's remaining headroom is bounded by the gen4 x4 links and
the 63 GiB expert set; a smaller pack is still the largest single lever (Entry 9).

## Entry 11 — parallel decode: both architectures scale poorly, the split better

Parallel decode measured with 4 concurrent prose requests (300 tokens each), `--parallel 4`:

| streams | split: aggregate | per-stream | tier: aggregate | per-stream |
|---|---|---|---|---|
| 1 | 49.0 | 49 | **59.0** | 59 |
| 2 | 68.0 | 38/34 | **89.1** | 47/45 |
| 3 | 72.8 | 27/27/24 | **84.9** | 30/29/28 |
| 4 | **87.1-95.3** | 24/23/23/22 | 81.4 | 22/21/21/20 |
| scaling at 4 | **~1.9x** | | 1.38x | |

**Neither scales well; the split scales better under load (1.9x against 1.38x) while the tier is better
single-stream (59 against 49).** So the architectures cross over at 2 streams, and the split's advantage
grows with concurrency — the opposite of its single-stream position, and further support for it as the
default for an agent workload.

**Why the tier degrades: it re-streams 75% of its experts every window.** Its own log, per request:
`decode expert cache hit rate: 93.5% (545 hits / 583 lookups); 1817 more read by the GPU over PCIe or from
another GPU (75.7% of all 2400 routed)`. Batching multiplies the rows per window (4 streams x 8 tokens =
32 rows against 8 solo) while the PCIe link stays at 7.1 GB/s, so the transfer, not the GPU, sets the
window: ~6 s of expert bytes per window at 4 streams if the routed sets were disjoint. The split moves no
weights at all, which is why it holds up.

**Drafting is worth keeping under concurrency.** `--spec 4` at 4 streams measured **71.8 against 95.3
aggregate** — dropping the window to cut rows-per-window loses, because acceptance stays high (93.5%
expert hit, ~75% drafts accepted). Reverted to `--spec 8`.

**Practical reading for the operator's workload:** 4 full-context streams at ~24 t/s each (87-95
aggregate) is where this box lands; 2 streams at ~38 each (68-89 aggregate) is the sweet spot. The
limiter is that decode is memory-bandwidth-bound on 4 cards sharing a 122 GiB model whose active expert
set exceeds any single card's cache.

## Entry 12 — decode: the expert tier wins, and the reason is the handoff

Matched comparison (ctx 32768, single stream, 4 reps each, `--parallel 1`), which is what Entry 11's
concurrency numbers could not show:

| workload | expert tier | layer split | acceptance (tier / split) |
|---|---|---|---|
| prose | **109.9** | 96.1 | 70.1% / 66.9% |
| json | **130.3** | 125.1 | 82.6% / 82.3% |

**The tier is 14% faster on prose and 4% on json**, with acceptance nearly identical — so this is not the
drafting; it is the split's per-window stage handoff. That is the expected shape: the tier has all four
cards compute one layer's experts at once, while the split runs each window through the cards in sequence.

**Note the earlier API numbers in Entry 11 were the wrong baseline for this question.** They were measured
at ctx 262144 with the full 4-slot configuration, where prose lands at 81 rather than 110; and reasoning
tokens *are* counted (`completion_tokens` matches the engine's `tg=` exactly), so the low figures were
context and workload, not accounting. Reality by workload and context:

| | tier | split |
|---|---|---|
| decode prose, 32k | 109.9 | 96.1 |
| decode json, 32k | 130.3 | 125.1 |
| decode struct (bench fixture), tier | 193-196 | — |
| decode prose, one stream at ctx 262144 | ~93.9 | ~81.1 |
| decode, 4 concurrent prose streams | 81-96 aggregate | 87-95 aggregate |

**The architecture choice is therefore a clean trade, not a win:**

| | prefill 16k | decode (prose @32k) | experts in VRAM |
|---|---|---|---|
| expert tier | 1611 t/s | **109.9** | 8.6 GiB (1 card) |
| layer split | **2354 t/s** | 96.1 | **48.9 GiB (4 cards)** |

For an agent workload that is prompt-heavy (long context re-read, short answers) the split's +46% prefill
dominates; for generation-heavy chat the tier's +14% decode does. The split also scales better under
concurrency (1.9x against 1.38x at 4 streams, Entry 11) and spreads KV across the cards. `basalt-ctl`
defaults to the split; `BASALT_SPLIT=0` selects the tier.

## Entry 13 — dormant sessions: parking works, the RAM is the constraint

**Parking plus 4 live slots boots and works on the split.** Startup:
`par=4 ctx/slot=262144 (slot-full-context), others queue`. (The "parking does not support a layer split"
error is only for the degenerate `--split-device 0` case, `generate.cpp:1482`.)

**Resume verified end to end.** Six distinct conversations were run and parked, then all six revisited:

| | prompt | reused | ttft |
|---|---|---|---|
| first turns | 2346 | 0 (session 0) / then shared prefix | 1.7-2.0 s |
| **all six RESUME** | 2349 | **2203** | **1.70-2.02 s** |

Every resume reused 2203 of 2349 tokens from the parked checkpoint and paid a steady ~101-118 tok/s
prefill for the rest, against 329 tok/s on a cold first touch. So dormant sessions come back without
re-reading their bodies.

**The constraint is RAM, and it is tight.** Parking's reservation is charged against the expert copy:

| budget | expert copy pinned | SSD spill |
|---|---|---|
| 33 (no parking) | 33 GiB | 1.3 GiB |
| 26 (engine's max with parking) | 26 GiB | 8.3 GiB |
| 24 | 24 GiB | 10.3 GiB |

The engine refuses above 26.0 GiB with parking on (`52.5 GiB free, 9.1 GiB floor + 17.3 GiB for serving`),
and `--prompt-cache-every 4096` raises the serving reservation to 18.2 GiB, dropping the ceiling to 25.2.
A parked 262144-token session costs 3.09 GiB of host RAM (measured, Entry 6), so **eight parked
full-context sessions would need 24.7 GiB that does not exist** alongside a 26 GiB expert copy, a 9.1 GiB
floor and 17.3 GiB of serving reservations.

**Practical configuration for "more dormant than running":**

- `--parallel 4` (4 live), `--conversation-cache-mib 8192`, `--conversation-cache-slots 8`,
  `--resident-budget-gib 24`. That accepts ~10 GiB of SSD spill (the experts read from the SSD cost
  decode latency) in exchange for dormancy.
- Parked sessions are cheap when short and expensive at full context: 8 parked 32k sessions fit in 1 GiB,
  8 parked 262k do not fit at all. **Parking pays when the conversations are shorter than the context
  cap**, which is the normal agent pattern (long context available, most sessions well under it).
- The alternative reading: the split already answers "more dormant than running" for *idle* sessions
  without parking at all — a slot that is not running costs only its 4.8 GiB of committed KV, and the
  prompt cache (6 checkpoints by default, ~113 MiB each, no KV) already resumes turns within a session.
  Parking is what extends that to *many* conversations.

## Entry 14 — the default is now 2 live / 4 dormant

Operator's requirement: more sessions dormant than running. Set and verified:

```
--parallel 2 --slot-full-context --max-context 262144
--conversation-cache-mib 8192 --conversation-cache-slots 4 --prompt-cache 8
--resident-budget-gib 26 --prefill 12288
```

`[server] kv 2.9GiB cuda1/2/3 solo=262144 slots=2x262144` (cuda0 4.0GiB + draft), `expert_slots=14490`,
`conv_cache=8192MiB/4 slots`. Verified: five conversations parked and all five resumed with **2203 of
2349 tokens reused** at ~110-171 t/s.

**Why 2 live beats 4 live when parking supplies the dormancy** (measured, split, 262144 ctx):

| shape | SSD spill | decode |
|---|---|---|
| 4 live + park (budget 24) | 20.6 GiB | 55.7 |
| **2 live + park (budget 26)** | 10.3 GiB | **77.5** |
| 2 live, no park (budget 33) | 3.3 GiB | 85.3 |

Fewer live slots leave their committed KV VRAM (`4.9 GiB` per card at 2 slots against 4.8 at 4 slots
across 12 layers each) to the expert cache, and parking recovers the concurrency as dormant storage
instead. The 2-live shape gives up 9% decode against no parking, in exchange for four resumable
conversations.

**Levers, and what each costs** (`BASALT_*` in `basalt-ctl`): `BASALT_PARALLEL` (live slots),
`BASALT_PARK_MIB`/`BASALT_PARK_SLOTS` (dormant; 0 disables, which restores the 33 GiB budget),
`BASALT_RESIDENT_GIB` (expert copy), `BASALT_CTX`, `BASALT_KV`, `BASALT_SPLIT`.

**Also fixed:** `basalt-ctl` failed at startup with `BASALT_PARK_MIB: unbound variable` under `set -u`
(the guard used `:-` but the assignment did not). Audited every `BASALT_*` use for the same shape.

## Entry 15 — prefill: the governor, and the chunk-size trap

**The prefill ramp was two separate things, and only the first is a tuning win.**

**1. CPU governor (+35%).** Bench to the side; the governor change is a system setting: `scaling_governor`
was `powersave` with `balance_performance` EPP under `amd-pstate` active mode, so clocks ramped 2.9 → 4.37
GHz *during* the read. Same 48k prompt:

| | rate |
|---|---|
| `powersave` (cold, ramping) | 2103 t/s |
| `performance`, cold | **2839 t/s** |
| `performance`, warm clocks | 3228 t/s |

`cpupower frequency-set -g performance` plus EPP `performance` on every cpu (the unit in the 09:xx note
makes it persistent — `/sys` resets on boot). This is the one lever that applies to *every* workload and
costs nothing.

**2. The cold expert cache, which does not tune away.** The first ~30k tokens run 2-3x slower than the
plateau in *every* configuration tested (1206-1314 t/s against a 4767 t/s steady state). It is not clocks
(measured flat-high throughout) and not chunk size (identical across 4096/8192/12288/16384). It is the
expert cache warming: with `--resident-budget-gib 26` and parking on, 10.3 GiB of experts live on the SSD
and the first pass routes through them.

**3. The chunk-size trap — my microbench was wrong and the operator caught it.** A 36k-token synthetic
prompt favoured `--prefill 4096` (total 3139, first-chunk 2600 against 12288's 3017/1628), and I shipped
that default. On a real 201k-token read it was **strictly worse** (647 → 2197 t/s, ~1800 average). Full
comparison, all under the performance governor, real ~200k reads:

| `--prefill` | 200k in | overall | first 30k | steady 100-200k |
|---|---|---|---|---|
| 4096 | 83.2 s | 2404 | 1642 | 2731 (plateaus) |
| 8192 | 59.8 s | 3344 | 1734 | 3935 |
| **12288** | **50.8 s** | **3934** | **1970** | **4767** |
| 16384 | 55.2 s | 3625 | 1796 | 4195 |

**12288 is the default**, best on every axis. The lesson worth keeping: **per-chunk overhead scales with
prompt length, the cold-cache ramp does not**, so a short-prompt microbench will always favour a small
chunk. Chunk tuning must be measured on a prompt of the length the workload actually uses.

**Net for a single long read** (cold, 2 live slots, parking, performance governor):

| | before this session | now |
|---|---|---|
| 200k-token prefill | ~3696 t/s (powersave, 12288) | **~3934 t/s** |
| 48k-token prefill, cold | 2103 t/s | **2839 t/s** |
| first 30k of any long read | ~1600 t/s | ~1970 t/s |

## Entry 16 — decode-first campaign (plan only; parking retired)

Written post-merge (`8885ff1d`); nothing below is implemented yet. Decode is the priority, prefill
secondary. **Parking is off this campaign**: if prefill gets fast enough, a dormant conversation can
resume by re-reading its body, so the RAM the park pool reserved goes back to the expert copy.

### The measured starting point (live server, merged binary, 2 live slots)

Per-window lines from `engine.log` during decode, agent traffic at 55-62k ctx, spec 8, acceptance
65-76%, `~45 ms/window`:

| where the window went | measured | source |
|---|---|---|
| CUDA0 GPU-reach wait | ~4.5 ms | `basalt batch: ... run (CUDA0 GPU-reach wait ... + CPU experts ...) + commit + emit` |
| CPU-pool expert rows | 5-11 ms; ~5.3 of ~67 routed rows per layer-window (92% VRAM hit) | same line |
| draft | ~6.5 ms/window (3.4 per slot-window) | `basalt batch spec: ... draft %.2f ms/window` |
| commit + emit | ~0.9 ms | same |
| stage compute + gather | the remaining ~25 ms | |
| expert streaming over PCIe | 0.00 | per-layer-window counter |

- **The spill taxes decode continuously**, not only the prefill ramp: `file_experts_bytes` 10.34 GiB,
  **3.07 TB read since boot**, ~8 MB/s of disk read *while generating*; misses split RAM 1.06M / file
  **1.11M** blobs — the file path served more blobs than the pinned copy. Hit rate 88-93% per request.
- **Stage imbalance**: decode GPU utils 37 / 45 / 62 / **90%** — CUDA3 (layers 36-47) gates every window
  while CUDA0 idles 63% despite carrying the draft head. The auto balancer (predicted window time,
  `generate.cpp:2229`) landed on equal 12/12/12/12.
- Swap 14 GiB used, MemAvailable 18 GiB — the boot-time safe cap on the RAM copy is suppressed
  (`budget = min(budget, safe)`, the safe cap derived from `MemAvailable`). Governor `performance`,
  persisted. Best prefill this boot: 231897 tokens at 3881.5 t/s.

Two beliefs from earlier entries this corrects: the CPU pool is **not** idle on the split (7-11 ms of
every window measured), and the merged binary is not the binary entries 10-15 were tuned on.

### Config baseline

`BASALT_PARK_MIB=0` (the `conversation_cache_mib` term drops out of `serve_growth_bytes`, serving
reserve ~10.2 → ~1.8 GiB), `BASALT_PARALLEL=2`, `BASALT_PREFILL=12288`, `BASALT_CTX=262144`,
`BASALT_KV=int8`, spec 8. `--prompt-cache` checkpoints (~113 MiB, no KV) still resume turns within a
*running* conversation. The accepted trade, stated: a parked 60k resume used to cost ~2 s; re-reading
the body costs ~13 s at the steady 4767 t/s (more until D1 fixes the first-30k ramp). With parking gone,
the live-slot count (`--parallel 2`) is a free sweep later — it trades committed KV against cache
(entry 14 measured the spill side of that trade).

### The items

**D1 — zero spill (parking retired; the budget is a contract).** Landed: parking off, and
`--resident-budget-gib` refuses a start it cannot serve in full instead of silently shrinking
(commits `2eb33ea4`, `adedaba8`). The pre-merge "zero spill at 29-37" table was misread: the pin's
plan is not the 26.0 GiB unique complement but complement **plus the stage cards' lent slots**
(1342 slots/card at prefill 12288, 3.46 GiB x 3) — measured plan **36.34 GiB**; budget 30 left its
coldest 6.3 GiB on file. Launcher default is 37: the whole plan, SSD 0.0. Swap is left alone
(operator's rule); the contract instead demands ~53 GiB available at the pin (37 + ~16 headroom)
and names the largest safe value when short. Gate at the next boots: `file_experts_bytes` → 0, the
boot guard warning gone, request hit rates and the `CPU experts ms/window` line down, and the
first 30k of a real 200k read re-measured — also the answer to open question 5 (the ramp is the
spill or the cold cache).

**D2 — rebalance the split (`--layer-split a,b,c`; flag only; a concurrency lever, not a latency one).**
Decode GPU busy means at zero spill measured 52 / 47 / 59 / **81%** (CUDA3, layers 36-47). But a single
window is a serial chain through the stages: for one active slot the window time is the *sum* of the
stages plus overheads, and moving a layer between stages leaves that sum unchanged - boundaries buy
nothing single-stream. They pay only under sustained concurrency, where the busiest card is the
ceiling: equalizing 81% to ~60% bounds the aggregate gain at ~+35%. Decide from the `--decode-timing`
lines under concurrent load (stage compute dominating the attribution) before spending a restart on
boundaries. Bitwise-neutral by the docs' own rule (`docs/precision/index.md`: "a layer gives the same
bits on any stage of the same card model") → greedy identity gate only.

**D3 — shrink the pool's row share.** `--vram-reserve-mib` / `--peer-reserve-mib` trims (~470 MiB ≈
150 slots per card, never measured on the split), and drop `--slot-full-context`: 2 slots commit
full-262144 KV (~5 GiB across the cards) that 60k sessions never touch, while elastic slots start at
4096 cells and grow — and upstream `b6125f60` now protects the prompt chunk's loan, removing the old
starve risk. Gate: hit rate up, pool ms down, a deep conversation mid-session must not drain the cache.

**D4 — `basalt-probe`, pool workers, SMT.** The pool is a decode item on the split after all (7-11 ms
of 45). 11 workers on 12C/24T AVX2; the probe measures per-worker rates and suggests
`--pool-workers` / `--prefill-cpu-share`. One maintenance window, server stopped.

**D5 — `--suffix-draft` sweep (flag only).** Acceptance 65-76% at lookup depth 3; agent traffic
re-sends growing context. A/B 8/16 on the real stream (fork evidence: 6.9-7.5 of 8 drafts accepted).

**D6 — the handoff wait (code; last).** The ~4.5 ms GPU-reach wait is the split's per-window boundary
cost — the measured 8-14% decode gap to the tier. Re-attribute after D1-D3 change the window mix;
double-buffering the boundary is the candidate (PDL already runs 201-298 launches per captured window).
Bitwise-neutral gate. Only if the wait survives D2.

**Prefill (secondary).** The ramp answer rides D1. Chunk re-sweep only if D2 moves boundaries
materially. Verify the merged overlapped prompt-read path engages with 2 live slots (the
`overlapped prompt read` log line, `generate.cpp:9808`) — free TTFT under load if it does.

### Diff against upstream (the engine diff is 5 hunks, ~200 lines; keep it minimal)

- **Peer-access enable (~20 lines)**: candidate to delete. Upstream never calls
  `cudaDeviceEnablePeerAccess` anywhere; `cudaMemcpyPeerAsync` is driver-managed either way, and the
  one-variable A/B already measured the enable neutral (1366 vs 1370 t/s at 64k). One d2d-rate +
  decode A/B after removal; if neutral, drop it.
- **`BASALT_FORCE_SPLIT` (~15 lines)**: load-bearing today — the merged price model still prints
  "expert tier 27.0, layer split 31.2; picked expert tier". May become droppable (master experiment).
- **Tier-aware RAM pin, adaptation servable filter, `set_ring_remotes` (~180 lines)**: upstream still
  lacks all three (`peer_only` is still built only under `--peer-exclusive` in `origin/main`). They are
  upstream-shaped already; the endgame is sending them upstream (maintainer approval first).
- Everything above is flags or existing instrumentation except D6 — this campaign adds no engine diff.

**Master experiment (after D1): split vs expert tier at zero spill.** The tier's decode win (109.9 vs
96.1) was measured with 3.3 GiB of spill, and the 4-stage prompt split's −23% prefill verdict predates
`caf9450a`'s host-bandwidth pricing and the deep-prompt correctness fixes — while the 4-stage split is
upstream's default on exactly four alike cards, needing no fork code. Zero spill is easier on the tier
(its complement is ~11 GiB). If tier + zero spill matches split + zero spill on both axes, drop
`BASALT_FORCE_SPLIT` and the engine diff shrinks to the placement fixes upstream should take anyway.

### Order

```
drain swap → D1 budget sweep (zero spill; ramp answer) → D2 boundaries (--decode-timing first)
D4 probe (one window) → D3 elastic slots / reserves → D5 suffix-draft (same boots where possible)
master experiment (tier vs split, both at zero spill) → D6 handoff (only if the wait survives D2)
```

Same-boot A/B only; real-length prompts for anything touching the chunk; every number labeled 180 W;
bench stops the server first (boots starve benches).

# Continuation — 4x RTX 5060 Ti Basalt campaign

State as of merge commit `8885ff1d`. Facts and explicit doubts; no interpretation dressed as fact.

## 1. Where the work stands

| | |
|---|---|
| Branch | `main`, local only |
| HEAD | `8885ff1d` (merge of upstream `55548875`) |
| Our commits | 21 (`b2fe96bf..cc809179`) + 1 merge |
| Base | `b2fe96bf` (upstream) |
| Remotes | `origin` = jesdga95/basalt (upstream), `fork` = gtrak/basalt-4x5060ti (pushed, force-pushed) |
| Rescue refs | tag `pre-upstream-merge` and branch `fork-main-rescue` on the fork = upstream's 19 commits as they were before the force-push |
| Uncommitted | none. `bin/basalt` is built post-merge and a server is running |

## 2. Hardware and model (measured)

- 4x RTX 5060 Ti, sm_120, 16311 MiB each, 180 W. **PCIe 4.0 x4 on all four** (GPU0 is x16-capable, negotiates x4).
- h2d 7.2 GB/s, d2d 6.64 GiB/s, no NVLink. Peer access 12 of 12 ordered pairs enabled.
- Ryzen 9 5900X, no AVX-512. 60.7 GiB RAM, 16 GiB swap. CUDA 13.2.
- Model `models/Qwen3.8-FN-NVFP4.basalt`, 122 GiB on disk = experts 63.28 + gguf 55.60 + dense 1.43 + vision 0.86 + mtp 0.77. Actual weights ~69 GiB.
- PLE `per_layer_token_embd.weight` is 50.66 GiB Q8_0, 320M rows, streamed from SSD, 90 B/row.

## 3. Campaign outcomes

### Accepted (in the shipped config)

| Change | Measured effect |
|---|---|
| RAM complement skips every tier's experts | pinned copy 42 → 19 GiB; ~30 GiB RAM freed |
| Top-up skip list; servable swap filter (2 duplication bugs) | removes duplicate resident copies |
| Peer access on (12/12 pairs) | d2d 6.64 GiB/s; required for the split |
| Adaptation candidates kept in the RAM complement | struct 117 → 196 t/s |
| Tier-held experts copied d2d from owning card | prefill 1502 → 1621 t/s; 17161 → 0 file reads |
| Layer split (`BASALT_FORCE_SPLIT=1`) as default | +46% prefill vs expert tier (2354 vs 1611) |
| 2 live / 4 dormant default with parking | 5 conversations resumed, 2203/2349 tokens reused |
| CPU governor → performance + EPP performance | cold 48k prefill 2103 → 2839 t/s (+35%) |
| `--prefill 12288` | 200k read 50.8 s / 3934 t/s, best of 4096/8192/12288/16384 |

The split is **not** an engine default — on this box the engine's own price model still picks the expert
tier. `basalt-ctl` sets `BASALT_FORCE_SPLIT=1` (read at `generate.cpp:1466`) and clears it when
`BASALT_SPLIT=0`. The engine-side `split_cap` controls the stage count *within* a split (see §4).

### Expert placement, read from the boot log

The split and the expert tier have **different VRAM shares, so the same `--resident-budget-gib` gives
different spill on each.** Split: VRAM 37.3 GiB, complement (experts not in VRAM) 26.0 GiB. Peer tier:
main VRAM 8.6 + peer cards 43.9, so the RAM copy is sized differently.

Split boots (VRAM 37.3), from `engine.log`:

| `--resident-budget-gib` | serving reserve | RAM copy | SSD spill | RAM free |
|---|---|---|---|---|
| 29, 30, 32, 37 | 1.8 GiB (parking off) | 26.0 | **0.0** | 10.4 – 21.9 |
| 34 | 1.8 | 22.6 | 3.3 | 13.8 |
| 34 | 1.8 | 25.2 | 0.8 | 15.0 |
| 27 | 10.2 (parking on) | 15.6 | 10.3 | 21.0 |

- **Zero spill on the split is reachable** — RAM copy 26.0 of the 26.0 GiB complement. Seen at budgets 29,
  30, 32 and 37, all with the serving reserve at 1.79 GiB.
- RAM copy = `min(budget, safe)` where `safe` is derived from `MemAvailable` at boot
  (`generate.cpp:4714-4732`, `basalt::platform::max_safe_copy`). Identical budgets therefore land
  differently across boots: budget 34 appears above as 22.6, 25.2 **and** 19.1 GiB of copy. Do not read a
  budget as a guaranteed copy size.
- **The live server currently has 10.3 GiB of SSD spill** (`resident-budget=26GiB`, log line: `RAM 15.6 GiB
  (25%), file (SSD) 10.3 GiB (16%)`). This is the config in §6 and it does **not** meet the stated goal of
  no SSD spill.

### Why reducing park *slots* would not empty the spill (code fact, not yet tested)

`serve_growth_bytes` (`generate.cpp:857-871`) adds the park pool to the serving reserve as
**`(uint64_t) o.conversation_cache_mib << 20`** (`generate.cpp:865`) — the *pool size*, not slots times a
per-slot size. Slots
divide that pool among conversations; they do not change the RAM the engine reserves for it. The
measured result is consistent: park 8192/4sl, 4096/2sl and 2048/1sl at budget 33 all gave
RAM 22.6 / SSD 3.3 GiB — the slot count moved nothing. **The lever for freeing that reserve is
`BASALT_PARK_MIB` (the pool), not `BASALT_PARK_SLOTS`.** Not tested; see §7.

### Decode, current config (split, adaptation on)

conv ~117-120, struct ~193-196, prose ~98-101 t/s.

### Prefill, current config

200k read ~3934 t/s; 231897-token read post-merge 59.7 s / 3882 t/s; first 30k ~1970 t/s.

### Chunk sweep, real ~200k reads, performance governor

| `--prefill` | 200k in | overall | first 30k | steady 100-200k |
|---|---|---|---|---|
| 4096 | 83.2 s | 2404 | 1642 | 2731 (plateaus) |
| 8192 | 59.8 s | 3344 | 1734 | 3935 |
| **12288** | **50.8 s** | **3934** | **1970** | **4767** |
| 16384 | 55.2 s | 3625 | 1796 | 4195 |

### Rejected, with the number that rejected it

| Option | Result |
|---|---|
| `--mmap-experts` | slower |
| d2d tier sourcing for repeated deep prefill | slower |
| chunk 32768 | 655 t/s |
| `--spec 10` / `--spec 12` | worse than `--spec 8` |
| `--spec 4` under concurrency | worse |
| peek row cap `kPeerFp4Tokens` 5120 → 32768 | 1106 vs 1611 t/s |
| 4-stage prompt split (P1) | 1060 vs 1374 t/s (−23%) |
| 4 live + park | 55.7 t/s at 20.6 GiB spill vs 2 live + park 77.5 t/s at 10.3 GiB |
| KV streaming `--kv-resident` | mutually exclusive with `--parallel`/`--batch` |

### Measurement errors found in this campaign

1. **A 36k microbench favoured `--prefill 4096`** and it was shipped as default. On a real 201k read 4096 was strictly worse (647 → 2197 t/s). Per-chunk cost scales with prompt length; the cold-cache ramp does not.
2. A `--prefill 8192` result was misread as "flat t/s" from the wrong log column.
3. **The `basalt-ctl` comment block above `--resident-budget-gib` quotes spill figures taken from
   *peer-tier* sessions** (VRAM 29.0), where budget 33 → RAM 21.6 / SSD 12.6. Those numbers are correct for
   that architecture and wrong for the split the launcher actually runs. Left in place as-is; do not use
   them to size the split.

### Unresolved observations

- **First ~30k tokens of every long read run 2-3x slower** (1206-1314 t/s vs 4767 steady) in *every* configuration. Not clocks (measured flat-high). Not chunk size (identical across all four). Attributed to cold expert cache + SSD spill. The spill explanation is untested against a zero-spill boot.
- **`Mlocked: 0.0 GiB`** in `/proc/PID/smaps_rollup` while the engine log reports the RAM copy as "page-locked". Unconfirmed whether it is actually locked; if it is not, the copy is reclaimable.
- Swap 8 GiB used and not drained — leftover from earlier 38 GiB-budget runs.
- No quality number exists for any config (see §5).

## 4. Upstream work that changes the calculus

Upstream `main` = `55548875`, 19 commits past our base `b2fe96bf`. These were the fork's contents before the
force-push and are preserved on `fork-main-rescue` and tag `pre-upstream-merge`.

### Directly contradicts a local conclusion

**`a4f4d06f` — the 4-stage prompt split is now default on alike cards.**

```c
const int split_cap = o.peer_devices.size() <= 1 ? 2 : split_alike ? (int) o.peer_devices.size() + 1 : 1;
```

- Was: `o.peer_devices.size() <= 1 ? 2 : (int) basalt::core::exp_flag("prompt_split", 1.0)`, i.e. 2 stages unless `--exp prompt_split` overrode it.
- Now: 4 stages by default when the cards match in bandwidth and VRAM ("alike" — our four identical cards qualify).
- Our P1 measured the 4-stage split at 1060 vs 1374 t/s (−23%) and rejected it. Commit `caf9450a` ("the prompt split price counts the host bandwidth every card streams over at once") changed the pricing that measurement was made against.

### Adjacent to local work, not yet assessed against it

- `b6125f60` (264 lines) — "elastic KV never takes the expert cache below a prompt chunk's loan; a request that cannot fit is refused, not fatal". Touches the resident-budget / spill path this doc's §3 describes.
- `6e615677` — "a tier card slower than the main one takes no share of the hottest experts" (placement, which we rewrote for the RAM complement).
- `db5f0741` — "a request's last round lands after its reply is out".
- `4353ef35`, `7d8d0508` — deep-prompt split correctness: a split read plans the chunks the prompt path reads; the draft layer's rows of earlier chunks wait in hand-off buffers.
- `88126657` — a long prompt part not read across the cards logs why, with its numbers.

### Tooling

- `4d4ff9e3`, `1cbf2212` — `/metrics` per-card and per-slot history, queue, best requests.
- `4fca1326`, `851e6d59`, `e51f561e` — stats page reworked.
- `4e4139dc` — bench uses the engine speculation default.
- `81a4af0c`, `4a591d96` — vision cache borrowing; refused image logged.
- `5a23e713` — a tier card without room declines the split at start.

### Merge state

One conflict, in the prompt-split stage setup: upstream dropped a redundant `size_t fb` redeclaration in the
same block where we changed `set_ring_remote` → `set_ring_remotes`. Resolved by keeping our API call and
upstream's removal (`fb` is declared above and read in the log line below). Merged tree builds clean.

**Landing review at merge `8885ff1d`**: the merged binary booted with the live config and read a
231897-token prompt in 59.7 s / 3882 t/s with prose decode at 81.4 t/s, consistent with the pre-merge
figures above. **This is not an A/B** — the two builds were not run back to back on the same prompt
contents, and no pre-merge reference run was taken at that prompt length.

## 5. Blocked

- **No quality number exists for any configuration.** PPL/KLD needs a llama.cpp Q8_0 KLD base (178 GiB Q8_0 GGUF; ~117 GiB free) and `wiki.test.raw`; the API exposes no `logprobs`. So KV formats, relaxed drafting and pack swaps are all unvalidated. `AGENTS.md` requires PPL/KLD through the decode path for any change that can alter numerics.
- `--gpu-only-full` refused with multi-GPU (`generate.cpp:1796` after the merge; was 1736).
- `--no-pool` cannot run (graphed paths need the residency table).

## 6. Live configuration

`basalt-ctl` env defaults: `BASALT_SPLIT=1`, `BASALT_PARALLEL=2`, `BASALT_PREFILL=12288`,
`BASALT_RESIDENT_GIB=26`, `BASALT_CTX=262144`, `BASALT_KV=int8`, `BASALT_PARK_MIB=8192`,
`BASALT_PARK_SLOTS=4`, `BASALT_PROMPT_CACHE=8`, port 8000, alias `local`. Also spec 8, ple-io direct,
adapt-swaps 96, park 8192/4sl.

- Engine-side, from the boot log: `prefill=12288 resident-budget=26GiB conversation-cache=8192MiB conversation-cache-slots=4 prompt-cache=8 spec=8`, `par=2 ctx/slot=262144 (slot-full-context)`, integer KV, 10.3 GiB SSD spill.
- `--parallel` is a **server** flag; the engine flag is `--batch`. Passing `--parallel` to the engine binary errors.
- `--kv-resident` is mutually exclusive with `--parallel`/`--batch`.
- Boots take ~2 min. A running server starves bench boots (`hc8: staging the hyper-connection matrices failed`); `basalt-ctl bench` stops the server first.
- `guard_host_memory` refuses to start below `MemAvailable` 14 GB or `SwapFree` 4 GB.
- Greedy identity: A/A identical; single-card pinned old-vs-new identical; multi-card differs (placement-dependent, expected).

## 7. Open questions

1. **Zero spill with parking still on.** Unreached so far. Two untested candidates: lower `BASALT_RESIDENT_GIB`-independent `BASALT_PARK_MIB` (frees the serving reserve, per §3), and a higher budget on a boot where `MemAvailable` is high. The stated goal — no SSD spill, parking on, fewer slots — has no confirmed configuration.
2. Does the 4-stage split perform differently now that pricing counts host bandwidth (`caf9450a`)? Decides whether P1's −23% still holds and whether split-vs-tier is still the right pick.
3. Does `b6125f60` change the resident-budget / spill behaviour the local config depends on?
4. Does `6e615677` conflict with our RAM-complement placement rewrite?
5. Can the first-30k ramp be hidden (pre-warm request)? Would also test whether the ramp is really the spill.
6. Is the RAM expert copy actually page-locked?
7. Quality harness: none exists.

## 8. Planned, not started

**N-way prompt offload** — one runner per tier, ownership-then-spread split, held-anywhere skip table;
~350 lines. Upstream's default 4-stage split may already cover this ground, so re-evaluate against it
before building.

## 9. References

- `OPTIMIZATION-PLAN.md` — entries 1-15, per-entry detail and trade tables.
- `WORKLOG.md` — campaign log.
- Test harness: `/tmp/pp.py`, `/tmp/pp2.py`, `/tmp/fc*.py`, `/tmp/firstchunk.py`, `/tmp/paradecode.py` (scratch; `/tmp` is not durable).
- Engine log `~/.cache/basalt/e51e4b1d90f2a2b3/engine.log` accumulates; use `tail`, not `grep`. The `resident RAM mode:` line and the `experts live in VRAM ... RAM ... file (SSD) ...` line are the pair to read for placement.
- Local tooling is `basalt-ctl` and this doc; there is no `scripts/` directory.

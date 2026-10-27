# WORKLOG — Basalt on 4x RTX 5060 Ti

Chronological record of getting https://github.com/jesdga95/basalt running on this box,
and of what the machine actually does with it. Newest sections at the bottom.

## Machine

- 4x RTX 5060 Ti, sm_120, 16,311 MiB each, 180 W enforced (220/198/180/180 max).
- PCIe 4.0 **x4** per card, no P2P, no NVLink (`topo -m`: all PHB). Host-to-device
  7.2 GB/s measured per card.
- Ryzen 9 5900X (12C/24T, **no AVX-512**), 60 GB RAM, 15 GB swap, 240 GB free disk.
- CUDA 13.2 (`/usr/local/cuda`), GCC 15.2, CMake 4.2.3.

## Build

Native, no container (per operator). `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`
with `CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`. Clean; the only warnings are
`%llu`/`uint64_t` format mismatches in `src/core/weights.cpp:195`. Python env is
`uv venv --python 3.12 .venv` + `requirements.txt` (the system python3.14 has no
`ensurepip`, so `python3 -m venv` fails).

## Model

`Qwen3.8-FN-NVFP4.basalt` from `jesdga/Qwen3.8-Flash-Next-Basalt`, 122 GiB.
NVFP4 experts (NVIDIA's codes bit for bit, 4.5 bits/weight), Q6_K dense.

| section | size |
|---|---|
| experts | 64,800 MiB (63.3 GiB) |
| gguf (PLE table 320,001,536 rows, embedding, head) | 56,931 MiB |
| dense | 1,467 MiB |
| mtp (drafter) | 786 MiB |
| vision | 876 MiB |

The download runs at 32 MiB/s single stream, which is this line's rate; ~1h.

## PLE is already on the SSD

The 320M-row n-gram table is **not** supposed to be in RAM. `--ple-io direct` is the
default: unbuffered `O_DIRECT` reads per row, never entering RAM or the file cache
(`generate.cpp:2413`, `pio.mode = PleIo::Direct`). That is why the pack is 122 GiB and
why RAM is free for expert copies. `--ple-io ram` is the slower alternative. No action
needed; recorded so it does not get "fixed" later.

## Placement: the pinned RAM copy duplicates the peer cards

Startup on 4 cards, all defaults:

```
experts live in VRAM 9.4 GiB (15%), peer GPU 44.1 GiB (70%), RAM 1.1 GiB (2%),
              file (SSD) 9.7 GiB (15%); 63.3 GiB in all
```

but separately:

```
resident RAM mode: 45.17 GiB of experts in RAM (page-locked)
```

62 GiB of experts total, against ~9.4 GiB VRAM + 44.1 GiB peer + 45.2 GiB RAM = 98.7 GiB
of capacity. Yet 9.7 GiB is still read from the SSD.

**Cause.** `pin_cache_complement()` (`src/core/expert_source.cpp:777`) pins the experts
"the GPU cache does not hold". Its skip list is `gpu_only`, built at
`generate.cpp:4663` as `peer_only` + `stage_held`. `peer_only` is filled only when
`--peer-exclusive` is set (`generate.cpp:4603`), and `--peer-exclusive` is **refused
with more than one peer card** (`generate.cpp:1701`). So with three peer tiers the RAM
copy is planned without knowing about any of them, and pins ~45 GiB of experts the peer
cards already hold.

Perfectly half the pinned copy is waste, and it costs the 9.7 GiB that spills to SSD.
Confirmed by sweeping `--resident-budget-gib`: 12 GiB and 20 GiB both still report
`file (SSD) 9.5 / 9.6 GiB`, because the budget only re-sizes the redundant copy.

**Also:** `top_up()` (`generate.cpp:6660`, the post-load top-up when a budget is not
given) *does* skip both `host_res` and `peer.has(...)`. So the two paths disagree: the
top-up is peer-aware, the initial pin is not.

Not patched: this is upstream's placement logic, the fix is a one-line change to the
`gpu_only` construction, and it wants measuring before it is sent anywhere. Worth
reporting upstream.

`--mmap-experts` (drop the pinned copy, let the OS page cache hold the tail) is not a
workaround: cold decode drops to 35/118/52 tok/s against 77/115/82, because the sampled
paths thrash the page cache.

## Performance

Decode t/s, `basalt-bench -t conv,struct,prose`, cold, 180 W:

| config | conv | struct | prose | notes |
|---|---|---|---|---|
| `--spec 0` (bench default) | 52 | 54 | 60 | no drafting |
| `--spec 8` | 77 | 115 | 82 | acc 68/92/64 %, dl 2.4/6.8/2.1 |
| `--spec 8 --mmap-experts` | 35 | 118 | 52 | page-cache thrash on sampled paths |

Speculation is the single biggest lever and the pack ships a drafter: never run `--spec 0`.

Server, single stream, `--spec 8`, 256k ctx: 27.3 / 27.8 tok/s (short/code), 10.9 tok/s
on a reasoning prompt. Lower than bench because these are sampled with short replies and
the first request pays the elastic-KV and checkpoint costs.

Prefill, measured through the server: 501 t/s at 6.7k, 832 t/s at 26.5k, **1,263 t/s at
242.7k**. Expert cache hit 97% at 26k, PCIe share 65-67%.

At depth, decode falls to **4.4 tok/s** at 242k tokens. That is 12 QSA layers of
attention over the full cache on a 180 W card at PCIe 4.0 x4, and it is expected rather
than a misconfiguration — but it sets the long-context expectation: fine for
summarise-a-repo, painful for chat.

The CPU tier is the other limit: 58-64 ms of a decode window is the pool
(`pool multi gate/up 46 ms`, `dispatch run 58 ms`), on an AVX2-only 5900X.

## KV

Per-token cost over the 12 QSA layers (from `kv_q8.hpp` / `kv_q4.hpp`, plus the indexer's
pooled keys and the rope tables), and the measured ceiling at ctx=262144:

| `--kv` | B/cell/layer | KiB/token | 256k KV | measured max | backed at boot |
|---|---|---|---|---|---|
| fp16 | 2048 | 24.38 | 6.09 GiB | 7.1 GiB | 1.0 GiB |
| int8 | 1056 | 12.75 | 3.19 GiB | 4.0 GiB | 0.7 GiB |
| k8v4 | 816 | 9.94 | 2.48 GiB | 3.3 GiB | 0.6 GiB |
| q4_0 | 576 | 7.12 | 1.78 GiB | 2.4 GiB | 0.5 GiB |

(Measured max is above the arithmetic because the draft layer's KV, the indexer state and
per-session padding ride along.) KV is **elastic** above 32k: 0.7 GiB is backed at boot
and cells are taken from the expert cache only as a request needs them.

256k is affordable: going 128k -> 256k at int8 costs ~3 expert slots (3483 -> 3354 of
3,669) and 1.6 GiB of the KV ceiling.

**A 242,707-token request completes end to end** on int8 / 256k (200 s, then 32 tokens).

## Current configuration

`basalt-ctl` (no systemd, per operator): `--gpu 0,1,2,3 --spec 8 --ple-io direct
--kv int8 --max-context 262144 --resident-experts`, port 8000 (8080 is filebrowser,
8000/1234 are the sglang stack). Verified: `ready ... ctx=262144`, and a chat completion
returns `"Four"` with `reasoning_content` populated.

## Open items

1. Upstream's `gpu_only` gap above — ~45 GiB of pinned RAM is redundant, and 9.7 GiB
   still hits the SSD. Report it; a one-line fix.
2. `--parallel` is untested; every measurement here is single stream. The slots' sessions
   come out of the expert cache, so this trades directly against KV.
3. No quality gate run (`--quality` needs a llama.cpp KLD base, not present).
4. `k8v4` / `q4_0` KV: no quality numbers exist in the repo or here.

## 2026-10-26 (evening): optimization campaign, entry 1 — P0 landed

Implemented, measured and logged in OPTIMIZATION-PLAN.md (Campaign log, entry 1). Headlines:

- Complement 42 → 19.09 GiB pinned at 256k; SSD spill 10.4 → 0.0 GiB; RAM after load 5.16 → 33.90 GiB.
- Decode, interleaved A/B, adapt OFF both sides: conv +50%, struct +39%, prose +16%; 0 file reads
  (stock: ~100 GB per bench).
- Second duplication bug found and fixed on the way: the post-load top-up skipped only tier 0 — it
  pinned 29.6 GiB of tier-held experts and drove the box into swap on the first production boot.
- Open: the prompt path (MMQ) reads tier-held experts from the file — tier-blind; the fork's
  "held-anywhere skip table" is the proven shape. Adaptation stays OFF until that lands.
- Greedy identity: A/A identical; single-card pinned old-vs-new identical; multi-card differs
  (placement moved — expected per docs).

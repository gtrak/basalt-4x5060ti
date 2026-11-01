# fp16 KV: the shipped configuration, and the sweep that chose it

## The coupling nobody had written down

**fp16 KV is incompatible with batch windows.** The engine refuses at boot:

    verify: batch windows need resident INT8 K/V without rotation (--kv int8)

(`verify.cpp:4551` - the batch verify path checks `st.kv_mode == 0 && st.kv_int8 && !st.kv_rot
&& !st.kv_hybrid && !st.kv_q4` for every QSA layer on every slot.)

So fp16 KV and `--parallel 2` cannot coexist. fp16 **requires** `--parallel 1`. That is
consistent with the owner's decision (single sequence anyway), but it means the fp16
pivot is not a free swap - it is a different serving shape.

## The configuration

```
--parallel 1 --kv fp16 --prefill 12288 --adapt-swaps 96 --max-context 262144
--layer-split 8,22,34 --slot-full-context --resident-experts ...
```

## The sweep (5 decode samples each, this box's band is ~3%)

| config | decode mean | min | max | prefill |
|---|---|---|---|---|
| **base: chunk 12288, sw 96** | **108.1** | 105.0 | 111.5 | **2,414** |
| chunk 16384 | 107.9 | 103.9 | 111.7 | 2,363 |
| chunk 8192 | 106.9 | 101.5 | 115.3 | 2,411 |
| chunk 24576 | 107.0 | 100.7 | 113.1 | **1,741** |
| adapt-swaps 48 | 106.5 | 104.3 | 108.8 | 2,404 |
| adapt-swaps 192 | 106.3 | 100.7 | 109.1 | 2,417 |

**Nothing beats the defaults.** Every decode difference is inside the noise band; prefill
clearly rejects 24576 (1,741 t/s, -28%). The existing chunk 12288 and `--adapt-swaps 96`
are already optimal under fp16 as well.

## Against the int8 two-slot configuration it replaces

| | int8, 2 slots | **fp16, 1 slot** |
|---|---|---|
| decode (mean of 5) | 104.2 | **108.1** |
| prefill (38k fresh) | **2,590** | 2,414 |
| KV precision | int8 quantized | **fp16** |

Decode is slightly FASTER and quality is better; prefill is 7% slower. That is a good
trade for the owner's stated priority.

## The cost, understood but not yet measured at long context

The fp16 penalty falls on the attention read, and only at long context: at 262144 the
KV bytes double, so the head stage's read goes 2.40 -> 4.79 ms per token
(PROJECTED from bytes and the 448 GB/s VRAM rate, not measured). At 32768 it is 0.60 ms
(2% of the window) and at 8192, 0.15 ms (0.5%). A long-context decode measurement is
still owed.

## Harness note that cost a run

`--max-context` defaults to **4096**. A sweep script that omits it silently boots a
4096-token model, and the first long prompt fails with "prompt exceeds the context
window" - which reads like an fp16 limitation but is not.

## The layer rebalance for fp16: 8,20,34

fp16 changes what the cut optimizes, because KV is allocated per card and only for that
card's QSA layers - measured **1.7 GiB per card at 8/22/34, 2.2 on c3**, against
**0.250 GiB per QSA layer at 262144**. So a cut's KV cost follows its QSA COUNT, not its
layer count, and the layers are 3,7,11,...,47.

Swept on fp16 (5 decode samples each, 122k fresh prompt for prefill):

| cut | layers | QSA | decode | 122k prefill |
|---|---|---|---|---|
| 8,22,34 (previous) | 8/14/12/14 | 2/3/3/4 | 107.7 | 3,656 |
| **8,20,34 (shipped)** | **8/12/14/14** | **2/3/3/4** | **107.6 - 111.6** | **3,654 - 3,656** |
| 10,22,34 | 10/12/12/14 | 2/3/3/4 | 107.4 | 3,669 |
| 8,20,32 | 8/12/12/16 | 2/3/3/4 | 108.7 | 3,269 |
| 12,24,36 | 12/12/12/12 | 3/3/3/3 | 96.8 | **3,955** |
| 12,26,38 | 12/14/12/10 | 3/3/3/3 | 97.3 | 3,952 |

**QSA-balanced cuts (3/3/3/3) buy +8% prefill for -10% decode.** 8,20,34 is the best
decode-side cut and holds the same prefill as 8/22/34, so it ships.

The rebalance is smaller than int8's was because fp16 doubles the KV the balance has to
pay for: the even cut equalizes KV (0.75 GiB each) but gives every card the same LAYER
count, which is what decode's expert cache wants to be uneven.

## Prefill is now 3,65x, not 4,97x - and that is fp16's price

Measured on the same 122,308-token fresh prompt:

| | 122k prefill |
|---|---|
| int8, --parallel 1 | **4,972** |
| int8, --parallel 2 | 4,141 |
| **fp16, --parallel 1 (shipped)** | **3,656** |

**fp16 costs 26% of long-prompt prefill.** Decode pays nothing measurable, because a
2.76-token window touches almost no KV; prefill writes and reads the whole thing, and at
122k that is 1.5 GiB int8 against 3.0 GiB fp16 of KV traffic on the critical path.

Also note: **--parallel 1 is 20% FASTER than --parallel 2 on prefill** (4,972 vs 4,141).
The "4,232 peak" this branch was quoting was the two-slot number; one slot beats it.

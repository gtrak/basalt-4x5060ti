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

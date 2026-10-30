# MMQ width sweep - the quarter-shard ceiling, priced

Task tree `/home/gary/opt/basalt-tpstep0`, branch `tp2-tpstep0`. The question this answers is TP=4's ceiling: the
filed assumption (the kimi band) says a narrowed NVFP4 gate/up product costs 1.2-1.4x more per row at a quarter
width and ~1.05-1.15x at half width, unmeasured, and the whole quarter-shard plan turns on whether it holds -
`>1.5x` at the narrow end kills TP=4's projected ceiling against tp2pp2+sharding's ~2,540, `<1.25x` strengthens it.

## MMQ sweep, this worktree

### The harness

`tools/mmq_width_probe.cu` - a standalone `nvcc` binary, not a CMake target and not linked into the engine, on the
pattern of `tools/tp_ring_probe.cu`. It measures the engine's own path rather than a copy of it: it calls
`basalt::prefill::mmq::Context::run(Product)` (src/prefill/moe_mmq.cu), the same call src/prefill/prefill.cpp makes
for a group's gate/up, with the engine's `mmq::quantize` for the q8_1 activations, `mmq::iota` for the row map and
`mmq::Context` for the stream-k scratch pool. `nm` confirms the engine's kernel is what is in the binary: 112
`launch_mul_mat_q<(ggml_type)40, ...>` instantiations (NVFP4), 6090 `mul_mat_q` symbols overall.

Reproduce the build (the `mmq` target's own flags; the ggml format table is C so the host C compiler builds it):

    cd /home/gary/opt/basalt-tpstep0
    gcc -O2 -c src/kernels/cpu/ggml/ggml_quants.c -Iinclude -Isrc/kernels/cpu/ggml -o /tmp/ggml_quants.o
    /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 --extended-lambda -use_fast_math \
      -Iinclude -Isrc/prefill -o /tmp/mmq_width_probe tools/mmq_width_probe.cu \
      src/prefill/moe_mmq.cu src/prefill/mmq/mmq_host.cu src/prefill/mmq/quantize.cu \
      src/prefill/mmq/instances/mmq_q2_0.cu src/prefill/mmq/instances/mmq_q8_0.cu \
      src/prefill/mmq/instances/mmq_q5_1.cu src/prefill/mmq/instances/mmq_q4_k.cu \
      src/prefill/mmq/instances/mmq_q5_k.cu src/prefill/mmq/instances/mmq_q6_k.cu \
      src/prefill/mmq/instances/mmq_iq2_xxs.cu src/prefill/mmq/instances/mmq_iq2_xs.cu \
      src/prefill/mmq/instances/mmq_iq2_s.cu src/prefill/mmq/instances/mmq_iq3_xxs.cu \
      src/prefill/mmq/instances/mmq_iq3_s.cu src/prefill/mmq/instances/mmq_iq4_nl.cu \
      src/prefill/mmq/instances/mmq_iq4_xs.cu src/prefill/mmq/instances/mmq_nvfp4.cu \
      /tmp/ggml_quants.o -lcudart

Verified: that command links clean (4m01s; `moe_mmq.cu`'s switch names every instantiated format, so all fourteen
instances are required). `./tools/check_platform_boundary.sh` and `./tools/check_stderr_logging.sh` both pass with
the file present.

### CPU-side findings (measured without a GPU)

The tile config is a host-side table, so `--layout` answers the block-layout and byte questions with no CUDA call
at all. NVFP4 on this path always lands on `I=128, nthreads=256, sram_layout=2 (Q3_K), K_vram=256, stream_k=1`;
the only thing that varies across the grid is `nty = ceil(N/128)` and `ntx = ceil(T/J)`:

| N | K | T | weights B/expert | q8 B | dst B | one-card footprint | J | ntx x nty |
|---|---|---|---|---|---|---|---|---|
| 2560 | 2560 | 2048 | 3 686 400 | 5 916 672 | 20 971 520 | 49 MiB | 128 | 16 x 20 |
| 1280 | 2560 | 2048 | 1 843 200 | 5 916 672 | 10 485 760 | 37 MiB | 128 | 16 x 10 |
| 640 | 2560 | 2048 | 921 600 | 5 916 672 | 5 242 880 | 31 MiB | 128 | 16 x 5 |
| 2560 | 2560 | 8192 | 3 686 400 | 23 611 392 | 83 886 080 | 186 MiB | 128 | 64 x 20 |
| 1280 | 2560 | 8192 | 1 843 200 | 23 611 392 | 41 943 040 | 144 MiB | 128 | 64 x 10 |
| 640 | 2560 | 8192 | 921 600 | 23 611 392 | 20 971 520 | 123 MiB | 128 | 64 x 5 |

Three things follow, before any timing:

1. **The bytes are the pack's own.** `matrix_bytes(NVFP4, 1280, 2560) = 1 843 200` is exactly `gu_row x n_ff`
   (`ggml_row_size(NVFP4, 2560) = 1440`, `1440 x 1280`), which is the pack's gate/up image. The sweep is narrowing
   a real product, not a synthetic shape.
2. **`T=8192` breaks the alongside-a-live-server budget.** 123-186 MiB per card against the 64 MiB cap, so those
   points refuse unless `--allow-big` is passed. The server-down window is exactly when they can run; the
   `T=2048` points fit the budget without the flag.
3. **The tile count is the mechanism the sweep is testing.** At fixed `T`, halving `N` halves `nty`, so a 640-wide
   point runs 64 x 5 = 320 tiles where a 2560-wide point runs 64 x 20 = 1280. Whether the SMs idle or the
   fixed per-launch cost dominates is what the multipliers below will say.

### The GPU points - commands for the supervised window

Cards free and the server on `:8000` down, from `/home/gary/opt/basalt-tpstep0`:

    # the decision run: the whole {2560,1280,640} x {2048,8192} grid, the slice check, and the multiplier table
    /tmp/mmq_width_probe --sweep --parity --allow-big

    # the same grid on the second card, for the paired reading
    /tmp/mmq_width_probe --sweep --parity --allow-big --device 1

    # the way prefill actually launches it: sixteen experts' weights in one product
    /tmp/mmq_width_probe --sweep --parity --allow-big --experts 16

    # single points, if a point has to be re-taken on its own
    /tmp/mmq_width_probe --n 2560 --rows 2048 --reps 25
    /tmp/mmq_width_probe --n 1280 --rows 2048 --reps 25
    /tmp/mmq_width_probe --n 640  --rows 2048 --reps 25
    /tmp/mmq_width_probe --n 2560 --rows 8192 --reps 25 --allow-big
    /tmp/mmq_width_probe --n 1280 --rows 8192 --reps 25 --allow-big
    /tmp/mmq_width_probe --n 640  --rows 8192 --reps 25 --allow-big

`--reps` is validated to be odd and at least 20 (default 25) so the reported median is one of the samples, not an
interpolated mean; three untimed warm-up launches precede them. Each point prints `median / min / max` ns per row,
weight GiB/s (the K-loop bytes: `matrix_bytes x experts`, read once per launch), q8 GiB/s and TFLOP/s.

Parity is `--parity`: it runs `N=2560` then `N=640`, and `N=1280` then `N=640`, at each `T`, on the same bytes -
weights are drawn row-major from a fixed seed (`20261030`, so a narrower point's rows are byte-identical to the
wider buffer's leading rows) and activations from a seed of their own (`0x5bd1e995`, so they do not move with `N`).
The narrow destination must equal the wide run's leading columns; MMQ's stream-k split depends on the tile count,
so equality is fp32-reassociation tight rather than bitwise, against a 1e-3 absolute and relative bound. A
mis-sliced stride misses by O(1), so the check cannot pass for the wrong reason.

The probe reads `cudaMemGetInfo` immediately before each point and refuses it unless the card has 256 MiB free,
frees every allocation between points, and creates its own stream. It never enters the engine's allocator or the
peer path, and touches no file.

### The results

**Not yet taken.** The GPU points are the parent's to run in the server-down window (see the commands above); this
section is filled from their output. Placeholders, so the shape of the answer is unambiguous:

    T=2048: N=2560 ____ ns/row | N=1280 ____ (____x) | N=640 ____ (____x)
    T=8192: N=2560 ____ ns/row | N=1280 ____ (____x) | N=640 ____ (____x)
    PARITY: ____

### What the 640 multiplier does to TP=4's ceiling

The projection to beat is tp2pp2+sharding at ~2,540 tok/s. TP=4's claim on it is that quartering the expert
matrices quarters the per-card K-loop bytes at the decode step, and the cost side of that claim is entirely this
one number: the per-row cost of the 640-wide product against the 2560-wide one. At the filed `1.2-1.4x` the
quartering still nets out and the ceiling stays above ~2,540; at `<1.25x` the shard path is clearly the better of
the two and TP=4 is worth building. At `>1.5x` the narrowing eats the byte saving - a quarter of the work at two
thirds the efficiency is no win over a half at near-full efficiency - and TP=4's projected ceiling falls under
tp2pp2+sharding, which is the same wall RUN 24 hit from the pool side. The `T=8192` column is the prefill-relevant
one and the `T=2048` column the chunk-relevant one; if they disagree, the prefill column is the one that moves the
ceiling, because that is where the expert products are paid for. Read the number against the tile counts in the
table above: a 320-tile launch on this card's SM count is where a flat cost stops being plausible.

# Drafted addendum for `TP2PP2-NOTES.md` RUN 25 - the MMQ width sweep

Drafted in `/home/gary/opt/basalt-tpstep0` (branch `tp2-tpstep0`), **not** applied to the tp2pp2 tree: it belongs
there only once the GPU points exist. Append it under the RUN 25 block, after the `**RUN 25 MEASURED**` paragraph
and its per-layer decomposition, where the (c)-tier and sharding sequencing is discussed. Numbers are blanks until
the sweep runs - do not land it with the blanks filled from an unsupervised run.

---

**MMQ WIDTH SWEEP (the sharding-sequencing input, item 4 above - harness landed, measurement pending the window).**
Item 4 sequences expert sharding behind the slice on a PPL/KLD gate; it never priced the other thing that can kill
the quarter-shard plan, the per-row cost of a narrowed NVFP4 product. That cost is an assumption on record and
nothing more - the kimi band of 1.2-1.4x at a quarter width, ~1.05-1.15x at half, never measured on this card. At
`>1.5x` the quarter's byte saving is eaten by its own inefficiency and TP=4's projected ceiling drops under this
slice's ~2,540; at `<1.25x` the shard path is clearly the better of the two. `tools/mmq_width_probe.cu`
(tpstep0 worktree; standalone `nvcc` binary on the `tools/tp_ring_probe.cu` pattern, not a CMake target, not linked
into the engine) measures it through the engine's own path - `mmq::Context::run(Product)`, the call
`src/prefill/prefill.cpp` makes for a group's gate/up, with the engine's `mmq::quantize`, `mmq::iota` and the
stream-k scratch pool, so the number is the path that would ship, not a paraphrase of it.

The block layout is settled CPU-side, with no CUDA call (`--layout`): NVFP4 takes `I=128, nthreads=256,
sram_layout=Q3_K, K_vram=256, stream_k=1, 61 952 B` of shared memory at every point of the grid, and `matrix_bytes`
at the real shape is `1 843 200 B = gu_row(1440) x n_ff(1280)`, the pack's own gate/up image - so the sweep
narrows a real product, and the only thing that moves with the width is the tile count (`64 x 20` at N=2560 down to
`64 x 5` at N=640). Two consequences known before timing: the `T=8192` points need 123-186 MiB on one card, so they
are outside the 64 MiB alongside-a-live-server budget and belong in a server-down window; and a 320-tile launch is
where a flat per-row cost stops being plausible, which is the hypothesis the sweep tests. Parity is built in - the
640-wide destination must equal the 2560-wide run's leading columns over byte-identical weights and activations
(fixed seeds `20261030` / `0x5bd1e995`, printed per line), against a 1e-3 bound: tight enough to be the point of the
check, loose enough for stream-k's reassociation, and a mis-sliced stride misses by O(1).

PENDING: the `{2560,1280,640} x {2048,8192}` multiplier table, the parity verdict, and the paragraph that says
whether TP=4's ceiling survives. Commands and the placeholder table live in this worktree's
`progress-mmq-sweep.md`; the run needs the cards free, so it is sequenced with the `=2` pipelined grading rather
than run beside the live server.

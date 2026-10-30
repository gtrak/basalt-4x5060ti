# Building this tree after a reboot

Split out of `progress-tp2slice.md` to keep that handoff inside the 300-line limit. Nothing here is slice-specific;
it is how this tree must be configured for its gates to mean anything.

A fresh `cmake` configure does **not** reproduce the working build: `BUILD_TESTING` defaults off, so
`ctest -L cpu` finds **zero** tests and reports no failures, and `nvcc` is not on a non-login shell's
`PATH`. Both fail silently. Configure with:

    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
          -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DBUILD_TESTING=ON
    cmake --build build -j4
    ctest -L cpu --test-dir build -N       # MUST report 69 before a green suite means anything
    ctest -L cpu --test-dir build          # 68/69; expert_multi_test is the known failure

A suite of zero passes exactly as well as a suite of sixty-nine. Build with `-j4`, not more: this worktree builds
on the host that runs the serving engine.

The slice's own CPU tests are `peer_qsa_span_test` (the partner's span: wire bytes, the staging sets, the resident
projections, and two protocol replays of the pair - `replay_collective()` for the collective's landing regions and
`replay_steps()` for the shipped step counts, each falsified by running the depth it should fail at) and
`tp2_shard_weights_test` (a weight the pack serves only as a native image can still be
cut); the attention window's parity case is `qsa_prompt_attn_parity` and needs a card.

The pair's own device tests, both needing two cards and skipping (77) with fewer: `tp2_allreduce_parity` (the
collective bitwise against `tp2_all_reduce_ref`, the refusals, the microbench line, and now the pair-agreement
sampler against `tp2_sample_ref` - the engine compares two cards' sampled sums element for element at its chunk
boundary, so a sampler reading an index the host does not use would make a broken pair look sound) and
`tp2_staging_depth` (the arm's whole per-layer pair protocol - the publish's rows, the partner's read of them, both
partials, both reduces - at each gate's depth, the lead running ahead the way the pipelined gate lets it, plus the
tripwire's falsifier: one layer summed on one card only must be seen or the check is decoration). Its second-set
case matters because the set-1 region of a `sets` 2 pair had never been touched by a kernel before it: the parity
test reduces at set 0 only.

The chain's own test needs one card, not two, and skips (77) with none: `qsa_chain_parity` grades the fingerprint
kernel against `qsa_chain_ref` - the host rule in the same header the kernel includes - at the widths the arm really
presents (mixed rows at 2560 columns, 24-head and 12-head attention, a chunk's step counts), bitwise on hash, sum
and max-abs, and then the things a fingerprint must be sensitive to that a test can only ask for: one bit flipped, two
rows swapped, one row's columns rotated, a NaN in the window, a row the sampled window does not select (which must be
invisible - the case asks the rule which rows it selects rather than assuming, and caught its author guessing wrong).
Its cost line is the instrument's price in a boot: ~48 us per component at the widest shape, launch cost, so a
sliced layer pays a few tenths of a millisecond per card and a chunk pays what a debug flag should but a default must
not. It caught its own harness before landing: the first draft copied its windows on the legacy stream and recorded
on a non-blocking one, so the kernel read the previous shape's bytes and correctly disagreed with a reference
computed from bytes still in flight. That is the instrument's contract in miniature - it fingerprints what a STREAM
has written - and the engine meets it because every component it records is produced on the stream it records onto.

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

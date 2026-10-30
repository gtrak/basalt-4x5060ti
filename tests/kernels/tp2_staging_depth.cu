// tests/kernels/tp2_staging_depth.cu - the prompt-attention arm's per-layer pair protocol, replayed on two cards at
// the depth each gate runs it: the lead's publish of a layer's rows, the partner's read of them, both cards'
// `o_proj` partials and their reduce, with the lead running ahead the way the pipelined gate lets it.
//
// WHY A DEVICE TEST FOR A PROTOCOL `peer_qsa_span_test` already replays on the CPU: the CPU replay models the
// protocol (who waits on what, in what order) and it is what found the single-region and single-step-set bugs.
// It cannot model what this file checks - that a peer copy really lands before a peer's kernel reads it, that a
// cross-context `ev_in` wait really orders two cards, and that a SECOND landing region is really a second set of
// bytes at the right offset.  The set-1 region of a `sets` 2 pair had never been touched by a kernel before this
// test (`tp2_allreduce_parity` runs every collective at set 0), and an off-by-`max_elems` there is a plausible
// token, not a fault.
//
// THREE CASES, the shapes the engine's two gates run, and the falsifier of the check the pair now carries:
//   sets 1, lead gated    - the serialized gate: one region per card, and the lead's end-of-layer wait.
//   sets 1, lead ungated  - what the serialized mode looks like when the wait is gone, at one region.
//   sets 2, lead ungated  - the pipelined gate as `peer_qsa_finish` runs it, second landing region and all.
//   sets 2, one card short - one layer's sum taken on one card only: the pair-agreement rule must fail it, or the
//                            engine's chunk-boundary verdict is decoration.
// A case that corrupts is reported per layer, with the sum it got and the one it should have got: a sum off by one
// layer's figure is the peer's NEXT partial, which is the failure this protocol exists to prevent.
//
// Needs two cards.  With fewer it exits 77, which ctest reads as a skip (label `gpu`).
#include "basalt/kernels/tp2_collective.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace basalt::kernels;

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

// The arm's figures, at the scale a test can hold: the ledger chunk's row count, `n_embd` columns for the partial,
// and a narrow wire row for what the publish ships (the mixed rows the partner's projections read).
constexpr int64_t kRows = 2048;
constexpr int64_t kCols = 2560;          // n_embd: the partial's width, which no head cut changes
constexpr int64_t kWire = 512;           // one QSA layer's shipped row, narrow on purpose
constexpr int kLayers = 24;              // twice the 12 QSA layers, so each set wraps many times
constexpr long long kArmCycles = 120000;  // the partner's per-layer compute, so the lead really does run ahead

// The fills are integers in float: exact, so a sum has one value and a wrong layer's partial is a visible offset
// rather than a tolerance.  Rank 1's figure sits a million above rank 0's, so a sum missing its peer is obvious,
// and every layer is a thousand apart, so a sum taken from the peer's NEXT layer is too.
__global__ void fill_rows(float* buf, int64_t n, int rank, int layer, int stride) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (float) (rank * 1000000 + layer * 1000 + (int) ((i / stride) % 8) * 4 + (int) (i % stride % 97));
}

/// The partner's per-layer compute: attention and its neighbours, modelled as the time they take, so the lead's
/// queue really is ahead of this card's when the next layer's push is queued.
__global__ void arm_hold(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}

float* alloc(int dev, int64_t floats) {
    if (cudaSetDevice(dev) != cudaSuccess) return nullptr;
    float* p = nullptr;
    return cudaMalloc((void**) &p, (size_t) floats * sizeof(float)) == cudaSuccess ? p : nullptr;
}

int64_t partial_elems() { return kRows * kCols; }
int64_t wire_elems() { return kRows * kWire; }

// The figures `fill_rows` writes, read the same way here: the test compares what the pair summed against these.
float expected_own(int rank, int layer, int64_t i, int64_t stride) {
    return (float) (rank * 1000000 + layer * 1000 + (int) ((i / stride) % 8) * 4 + (int) (i % stride % 97));
}

/// One case: `sets` landing regions per card, and whether the lead waits for the partner's layer the way the
/// serialized gate does.  `sabotage` takes ONE layer's partner sample before its collective instead of after it,
/// so that card's ring holds its own partial where its peer's holds the sum - the tripwire's falsifier: a check
/// that cannot see a one-sided reduce is not a check.  Returns the number of layers whose sampled figures are not
/// the sum they should be (a one-sided layer counts as wrong, which is what makes the falsifier case fail).
int replay(int dev_a, int dev_b, int sets, bool lead_gated, const char* what, int sabotage = -1) {
    std::string err;
    Tp2Pair pair;
    if (!pair.open(dev_a, dev_b, partial_elems(), err, sets)) {
        std::printf("SKIP staging depth: %s\n", err.c_str());
        return -1;
    }
    cudaSetDevice(dev_a);
    cudaStream_t s_a = nullptr;
    if (cudaStreamCreateWithFlags(&s_a, cudaStreamNonBlocking) != cudaSuccess) return -1;
    cudaSetDevice(dev_b);
    cudaStream_t s_b = nullptr;
    if (cudaStreamCreateWithFlags(&s_b, cudaStreamNonBlocking) != cudaSuccess) return -1;

    // The arm's events, on the contexts that record them: `ev_in`/`ev_red` behind work on the LEAD's stream,
    // `ev_out` behind the partner's, exactly as `make_peer` and `peer_qsa_finish` do it.
    cudaSetDevice(dev_a);
    std::vector<cudaEvent_t> ev_in(sets, nullptr), ev_red(sets, nullptr);
    for (int b = 0; b < sets; ++b)
        if (cudaEventCreateWithFlags(&ev_in[b], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_red[b], cudaEventDisableTiming) != cudaSuccess) return -1;
    cudaSetDevice(dev_b);
    std::vector<cudaEvent_t> ev_out(sets, nullptr);
    for (int b = 0; b < sets; ++b)
        if (cudaEventCreateWithFlags(&ev_out[b], cudaEventDisableTiming) != cudaSuccess) return -1;

    // The lead's own sources (rewritten every layer, single-set on this card, as `mixed_h` is) and its partial per
    // set; the partner's wire sets (where the publish lands) and its partial per set.
    cudaSetDevice(dev_a);
    float* src_a = alloc(dev_a, wire_elems());
    float* part_a[2] = {alloc(dev_a, partial_elems()), alloc(dev_a, partial_elems())};
    cudaSetDevice(dev_b);
    float* wire_b[2] = {alloc(dev_b, wire_elems()), alloc(dev_b, wire_elems())};
    float* part_b[2] = {alloc(dev_b, partial_elems()), alloc(dev_b, partial_elems())};
    if (!src_a || !part_a[0] || !part_a[1] || !wire_b[0] || !wire_b[1] || !part_b[0] || !part_b[1]) {
        std::printf("SKIP staging depth: out of memory\n");
        return -1;
    }
    cudaSetDevice(dev_a);
    float* chk_a = nullptr;   // one sampled row per layer, per card
    cudaMalloc((void**) &chk_a, (size_t) kLayers * basalt::kernels::kTp2Samples * sizeof(float));
    cudaSetDevice(dev_b);
    float* chk_b = nullptr;
    cudaMalloc((void**) &chk_b, (size_t) kLayers * basalt::kernels::kTp2Samples * sizeof(float));
    float* chk_wire = nullptr;
    cudaMalloc((void**) &chk_wire, (size_t) kLayers * basalt::kernels::kTp2Samples * sizeof(float));
    cudaMemset(chk_a, 0, (size_t) kLayers * basalt::kernels::kTp2Samples * sizeof(float));
    cudaMemset(chk_b, 0, (size_t) kLayers * basalt::kernels::kTp2Samples * sizeof(float));
    cudaMemset(chk_wire, 0, (size_t) kLayers * basalt::kernels::kTp2Samples * sizeof(float));
    const int kIdx = basalt::kernels::kTp2Samples;
    std::string serr;   // the production sampler's, so this test grades the kernel the engine actually carries

    const int threads = 256;
    const int blocks = (int) ((partial_elems() + threads - 1) / threads);
    const int wblocks = (int) ((wire_elems() + threads - 1) / threads);

    // ---- the whole chunk enqueued, host ahead of both cards, as the driver does: nothing here syncs until the
    // layers are all queued, so whatever order the two cards settle into is the protocol's, not the test's.
    bool issued = true;
    for (int l = 0; l < kLayers && issued; ++l) {
        const int set = l % sets;

        // THE PUBLISH: the lead writes the rows it will ship, then copies them into the partner's wire set on its
        // own stream, then records `ev_in` - so the partner's wait is on the copies, not on the lead's queue.
        cudaSetDevice(dev_a);
        fill_rows<<<wblocks, threads, 0, s_a>>>(src_a, wire_elems(), 0, l, kWire);
        if (cudaMemcpyPeerAsync(wire_b[set], dev_b, src_a, dev_a, (size_t) wire_elems() * 4, s_a) != cudaSuccess)
            issued = false;
        if (issued && cudaEventRecord(ev_in[set], s_a) != cudaSuccess) issued = false;

        // THE ARM: the partner waits on the publish, does its layer's work, reads the rows it was sent, then makes
        // and pushes its partial.  All on one stream, in this order, which is what `peer_qsa_layer` guarantees.
        cudaSetDevice(dev_b);
        if (cudaStreamWaitEvent(s_b, ev_in[set], 0) != cudaSuccess) issued = false;
        if (issued) arm_hold<<<1, 32, 0, s_b>>>(kArmCycles);
        if (issued) tp2_sample_partial(chk_wire + l * kIdx, wire_b[set], kRows, kWire, s_b, serr);
        if (issued) fill_rows<<<blocks, threads, 0, s_b>>>(part_b[set], partial_elems(), 1, l, kCols);
        // The falsifier's one deviation: sample this layer on this card BEFORE the add, so the two rings describe
        // different sums.  Nothing else about the run changes, so only the pair-agreement rule can catch it.
        if (issued && l == sabotage)
            tp2_sample_partial(chk_b + l * kIdx, part_b[set], kRows, kCols, s_b, serr);
        if (issued && !pair.all_reduce(part_b[set], kRows, kCols, s_b, dev_a, 1, err, set)) {
            std::printf("FAIL staging depth: partner layer %d: %s\n", l, err.c_str());
            ++g_fail;
            issued = false;
        }

        // THE LEAD's own product and its side of the reduce, in the same order `peer_qsa_finish` issues them.
        cudaSetDevice(dev_a);
        if (issued) fill_rows<<<blocks, threads, 0, s_a>>>(part_a[set], partial_elems(), 0, l, kCols);
        if (issued && !pair.all_reduce(part_a[set], kRows, kCols, s_a, dev_b, 0, err, set)) {
            std::printf("FAIL staging depth: lead layer %d: %s\n", l, err.c_str());
            ++g_fail;
            issued = false;
        }

        // The two proofs the next layer's publish and push wait on, recorded after both adds were enqueued.
        cudaSetDevice(dev_b);
        if (issued && cudaEventRecord(ev_out[set], s_b) != cudaSuccess) issued = false;
        cudaSetDevice(dev_a);
        if (issued && cudaEventRecord(ev_red[set], s_a) != cudaSuccess) issued = false;

        // Each card's consumer of its own summed partial, on the stream that produced it.  The sabotaged layer
        // keeps the pre-add sample taken above, which is the whole point of that layer.
        if (issued) tp2_sample_partial(chk_a + l * kIdx, part_a[set], kRows, kCols, s_a, serr);
        cudaSetDevice(dev_b);
        if (issued && l != sabotage) tp2_sample_partial(chk_b + l * kIdx, part_b[set], kRows, kCols, s_b, serr);

        // THE WAIT, which is the whole difference between the engine's two gates.
        cudaSetDevice(dev_a);
        if (issued && lead_gated && cudaStreamWaitEvent(s_a, ev_out[set], 0) != cudaSuccess) issued = false;
    }

    cudaSetDevice(dev_a);
    cudaStreamSynchronize(s_a);
    cudaSetDevice(dev_b);
    cudaStreamSynchronize(s_b);

    // ---- grade every layer.  A sampled element must be the lead's figure plus the partner's for THAT layer.
    std::vector<float> ha((size_t) kLayers * kIdx), hb((size_t) kLayers * kIdx), hw((size_t) kLayers * kIdx);    cudaSetDevice(dev_a);
    cudaMemcpy(ha.data(), chk_a, ha.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaSetDevice(dev_b);
    cudaMemcpy(hb.data(), chk_b, hb.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hw.data(), chk_wire, hw.size() * sizeof(float), cudaMemcpyDeviceToHost);

    int bad = 0;
    int disagreed = 0;
    for (int l = 0; l < kLayers; ++l) {
        for (int k = 0; k < kIdx; ++k) {
            const int64_t i = tp2_sample_index(k, kRows, kCols);
            const float want = expected_own(0, l, i, kCols) + expected_own(1, l, i, kCols);
            // THE PAIR-AGREEMENT RULE, the same one the engine applies at its chunk boundary: two cards that summed
            // the same two partials hold bit-identical elements (IEEE addition commutes), and a pair that does not
            // is a run that must fail rather than a token.  NaN agrees with NaN, so an overflowed partial reports
            // as the sum it is instead of as a disagreement.
            const float x = ha[(size_t) l * kIdx + k], y = hb[(size_t) l * kIdx + k];
            if (!(x == y) && !(std::isnan(x) && std::isnan(y))) ++disagreed;
            if (x != want || y != want) {
                if (bad < 4)
                    std::printf("  layer %2d set %d col %4lld: got lead %.0f partner %.0f, want %.0f (%+.0f)\n", l,
                                l % sets, (long long) (i % kCols), (double) ha[(size_t) l * kIdx + k],
                                (double) hb[(size_t) l * kIdx + k], (double) want,
                                (double) (ha[(size_t) l * kIdx + k] - want));
                ++bad;
            }
            if (disagreed && bad <= 4 && k == 0)
                std::printf("  layer %2d: the two cards do not hold the same sum\n", l);
            // The rows the partner was sent must be the lead's for THIS layer: a stale wire set is the other half
            // of the same class of bug.
            const float want_wire = expected_own(0, l, tp2_sample_index(k, kRows, kWire), kWire);
            if (hw[(size_t) l * kIdx + k] != want_wire) {
                if (bad < 4)
                    std::printf("  layer %2d set %d: the partner read a published row of %.0f, want %.0f\n", l,
                                l % sets, (double) hw[(size_t) l * kIdx + k], (double) want_wire);
                ++bad;
            }
        }
    }
    if (pair.timed_out()) { std::printf("  %s: a rank gave up waiting for its peer\n", what); ++bad; }
    CHECK(pair.issued(0) == pair.issued(1));   // lockstep, or the test issued the pair wrong

    std::printf("%-28s sets %d, lead %s: %d of %d layers wrong, %d of %d samples the pair disagreed on, "
                "%d collectives per card\n",
                what, sets, lead_gated ? "gated" : "ungated", bad, kLayers, disagreed, kLayers * kIdx,
                (int) pair.issued(0));
    // A healthy pair never disagrees about its sums; a one-sided layer must always be seen.
    if (sabotage < 0) CHECK(disagreed == 0);
    else CHECK(disagreed > 0);

    // ---- tear down
    cudaSetDevice(dev_a);
    for (int b = 0; b < sets; ++b) { cudaEventDestroy(ev_in[b]); cudaEventDestroy(ev_red[b]); }
    cudaFree(src_a); cudaFree(part_a[0]); cudaFree(part_a[1]); cudaFree(chk_a);
    cudaStreamDestroy(s_a);
    cudaSetDevice(dev_b);
    for (int b = 0; b < sets; ++b) cudaEventDestroy(ev_out[b]);
    cudaFree(wire_b[0]); cudaFree(wire_b[1]); cudaFree(part_b[0]); cudaFree(part_b[1]);
    cudaFree(chk_b); cudaFree(chk_wire);
    cudaStreamDestroy(s_b);
    pair.close();
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: tp2_staging_depth [--selftest]\n"); return 2; }
    }

    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs < 2) {
        std::printf("tp2_staging_depth: %d card%s, the pair needs two\n", devs, devs == 1 ? "" : "s");
        cudaGetLastError();
        return selftest ? 0 : 77;
    }

    // The serialized gate: one region, and the lead waits for the partner's layer.  Safe by construction.
    CHECK(replay(0, 1, 1, true, "serialized shape") == 0);
    // The pipelined shape WITHOUT the second region: clean, at this scale.  The peer's next push is queued behind
    // its own product and its own add, so at these sizes it cannot overtake this card's add - it stayed clean with
    // the modelled arm at 0 cycles and at 120k.  That is why the serialized mode never looked broken on one region
    // even though the protocol gives it no proof; the proof, rather than the timing, is what `sets` 2 buys, and
    // `peer_qsa_span_test`'s `replay_collective()` is what pins the depth argument with the timings stripped away.
    CHECK(replay(0, 1, 1, false, "pipelined shape, one region") == 0);
    CHECK(replay(0, 1, 1, false, "pipelined shape, one region") == 0);
    // The pipelined gate as the arm runs it: two regions, the lead ungated, and nothing sums the wrong layer.
    // This is the second landing region's first run under a kernel: `tp2_allreduce_parity` reduces at set 0 only.
    CHECK(replay(0, 1, 2, false, "pipelined, two regions") == 0);
    // The tripwire's falsifier: one layer summed on one card only.  If the pair-agreement rule cannot see THAT,
    // the check the engine now carries is decoration.
    CHECK(replay(0, 1, 2, false, "one card short by design", 5) > 0);

    std::printf("\ntp2 staging depth: %d failures\n", g_fail);
    if (g_fail) return 1;
    if (selftest) std::printf("tp2_staging_depth OK\n");
    return 0;
}

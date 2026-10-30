// tests/kernels/tp2_allreduce_parity.cpp - the 2-way all-reduce against its CPU reference: random values, real
// geometry, a repeated sequence, the refusals, and a microbench line.  Bitwise, not a tolerance: an add of two
// partials has one order and the kernel must have that one.
//
// Needs two cards.  With fewer it exits 77, which ctest reads as a skip (the label is `gpu`; the CPU parts of the
// contract - the byte figure, the geometry check, the reference's order - run here regardless).
#include "basalt/kernels/tp2_collective.hpp"
#include "basalt/core/qsa_head_split.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

using namespace basalt::kernels;

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

/// The widths a split linear actually produces: n_embd 2048 columns for a decode row, a verify window of 8
/// queries, and a prompt chunk's worth of rows through the widest projection.  `head_dim` 256 and 512 columns are
/// the other figures the pair sees.
struct Shape {
    int64_t rows, stride;
    const char* what;
};
const Shape kShapes[] = {
    {1, 2048, "decode row, o_proj 2048"},
    {8, 2048, "verify window of 8"},
    {8, 256, "verify window, one head group"},
    {64, 4096, "prompt chunk, wide projection"},
    {3, 2049, "an odd width"},
    {1, 1, "the smallest"},
};

float* dev_alloc(int dev, size_t floats) {
    if (cudaSetDevice(dev) != cudaSuccess) return nullptr;
    float* p = nullptr;
    if (cudaMalloc((void**) &p, floats * sizeof(float)) != cudaSuccess) return nullptr;
    return p;
}

// ---------------------------------- the pure parts, on any machine ----------------------------------

void pure_contract() {
    CHECK(tp2_all_reduce_bytes(8, 2048) == 8 * 2048 * 4ull);
    CHECK(tp2_all_reduce_bytes(0, 2048) == 0);
    CHECK(tp2_all_reduce_bytes(8, 0) == 0);

    CHECK(tp2_all_reduce_ok(8, 2048, 8 * 2048));
    CHECK(!tp2_all_reduce_ok(9, 2048, 8 * 2048));   // one row past the staging
    CHECK(!tp2_all_reduce_ok(0, 2048, 8 * 2048));
    CHECK(!tp2_all_reduce_ok(8, 0, 8 * 2048));
    CHECK(tp2_all_reduce_ok(1, 1, 1));

    CHECK(tp2_peer(0) == 1 && tp2_peer(1) == 0);

    // lockstep: both ranks count from 1, so the push a rank waits for carries its own number
    CHECK(tp2_push_arrived(1, 1) && tp2_push_arrived(7, 7));
    CHECK(!tp2_push_arrived(6, 7));       // the peer has not issued this one yet
    CHECK(tp2_push_arrived(8, 7));        // a peer that ran ahead still satisfies the wait

    // the reference adds THIS rank's partial first: a+b, not b+a, and a float add has an order
    const int64_t rows = 4, stride = 8;
    std::vector<float> a(rows * stride), b(rows * stride), got(rows * stride), want(rows * stride);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0e3f, 1.0e3f);
    std::uniform_real_distribution<float> tiny(-1.0e-30f, 1.0e-30f);
    for (int64_t i = 0; i < rows * stride; ++i) {
        // a mixture that makes the order observable: a big number plus a tiny one loses the tiny one in one
        // direction and keeps it in the other unless both operands round the same way
        a[i] = u(rng) * (i % 3 == 0 ? 1.0e22f : 1.0f);
        b[i] = i % 5 == 0 ? tiny(rng) : u(rng);
        want[i] = a[i] + b[i];
    }
    tp2_all_reduce_ref(got.data(), a.data(), b.data(), rows, stride);
    for (int64_t i = 0; i < rows * stride; ++i) CHECK(std::memcmp(&got[i], &want[i], sizeof(float)) == 0);
    // a row is `stride` floats and rows are `stride` apart: what the reference touches is rows*stride, no more
    std::vector<float> pad((size_t) rows * stride + 4, 1234.5f);
    for (int64_t i = 0; i < rows * stride; ++i) pad[i] = 1.0f;
    tp2_all_reduce_ref(pad.data(), pad.data(), pad.data(), rows, stride);
    for (int64_t i = rows * stride; i < rows * stride + 4; ++i) CHECK(pad[i] == 1234.5f);
    // in place (out aliasing a) is the same answer
    std::vector<float> c2 = a;
    tp2_all_reduce_ref(c2.data(), c2.data(), b.data(), rows, stride);
    for (int64_t i = 0; i < rows * stride; ++i) CHECK(c2[i] == want[i]);

    // UNEQUAL halves do not change what the collective moves.  A rank's o_proj partial is a FULL-WIDTH row: the
    // projection sums its OWN columns into all of n_embd, so the head cut decides which terms a rank adds, never
    // how many bytes cross or in what order the two ranks' rows are summed.  The runs below are an uneven cut.
    {
        using basalt::core::qsa_lead_heads;
        using basalt::core::qsa_out_cols;
        using basalt::core::qsa_pair_shard;
        using basalt::core::QsaHeadRun;
        const int64_t lead_heads = qsa_lead_heads(24, 2.0 / 3.0);
        const auto lead = qsa_pair_shard(24, 2, lead_heads, 0);
        const auto peer = qsa_pair_shard(24, 2, lead_heads, 1);
        CHECK(lead.ok() && peer.ok() && lead.nq() == 16 && peer.nq() == 8);
        const QsaHeadRun c0 = qsa_out_cols(lead, 256), c1 = qsa_out_cols(peer, 256);
        CHECK(c0.start == 0 && c0.len == 16 * 256 && c1.start == 16 * 256 && c1.len == 8 * 256);
        CHECK(c0.len + c1.len == 24 * 256);   // the two partials' columns make the whole row
        const int64_t wide = 2048;            // n_embd: the partial's width, which no head cut changes
        CHECK(tp2_all_reduce_bytes(rows, wide) == rows * wide * 4ull);
        CHECK(tp2_all_reduce_ok(rows, wide, (uint64_t) rows * wide * 4));
        std::vector<float> pa((size_t) rows * wide, 0.25f), pb((size_t) rows * wide, -0.5f);
        std::vector<float> po((size_t) rows * wide), pr((size_t) rows * wide);
        for (int64_t i = 0; i < rows * wide; ++i) pr[i] = pa[i] + pb[i];
        tp2_all_reduce_ref(po.data(), pa.data(), pb.data(), rows, wide);
        for (int64_t i = 0; i < rows * wide; ++i) CHECK(std::memcmp(&po[i], &pr[i], sizeof(float)) == 0);
    }
}

// ---------------------------------- the device path ----------------------------------

bool bit_eq(const float* x, const float* y, size_t n) { return std::memcmp(x, y, n * sizeof(float)) == 0; }

/// The pair-agreement sampler: the indices are one rule shared with the kernel, so both halves are checked - that
/// the rule stays inside the partial and spreads over both axes, and that the kernel reads exactly it.  The
/// engine's chunk-boundary verdict compares the two cards' rings element for element, so a sampler that read an
/// index the host did not use would make a broken pair look sound.
void sampler_contract() {
    const int64_t rows = 64, stride = 2048;
    int64_t row_of[kTp2Samples], col_of[kTp2Samples];
    for (int k = 0; k < kTp2Samples; ++k) {
        const int64_t i = tp2_sample_index(k, rows, stride);
        CHECK(i >= 0 && i < rows * stride);
        row_of[k] = i / stride;
        col_of[k] = i % stride;
    }
    CHECK(row_of[0] == 0 && row_of[kTp2Samples - 1] == rows - 1);   // spread over the rows, not clustered
    for (int k = 1; k < kTp2Samples; ++k) CHECK(row_of[k] >= row_of[k - 1]);
    {
        int distinct = 0;
        for (int k = 0; k < kTp2Samples; ++k)
            if (std::find(col_of, col_of + kTp2Samples, col_of[k]) == col_of + k) ++distinct;
        CHECK(distinct > 1);   // ...and over the columns: a wrong region offset shifts elements along a row
    }
    CHECK(tp2_sample_index(-1, rows, stride) == -1);
    CHECK(tp2_sample_index(kTp2Samples, rows, stride) == -1);
    CHECK(tp2_sample_index(0, 0, stride) == -1);
    {
        std::vector<float> a((size_t) rows * stride), b((size_t) rows * stride);
        for (size_t i = 0; i < a.size(); ++i) { a[i] = (float) (i % 97) * 0.5f; b[i] = (float) (i % 13) * 2.0f; }
        for (int k = 0; k < kTp2Samples; ++k) {
            const int64_t i = tp2_sample_index(k, rows, stride);
            CHECK(tp2_sample_ref(a.data(), b.data(), k, rows, stride) == a[i] + b[i]);
        }
    }
}

int device_path(int dev_a, int dev_b) {
    const int64_t max_elems = 64 * 4096;
    Tp2Pair pair;
    std::string err;
    if (!pair.open(dev_a, dev_b, max_elems, err)) {
        std::printf("SKIP tp2 pair: %s\n", err.c_str());
        return 77;
    }
    CHECK(pair.valid());
    CHECK(pair.max_elems() == max_elems);

    float* a_host = nullptr;
    if (cudaSetDevice(dev_a) != cudaSuccess ||
        cudaHostAlloc((void**) &a_host, (size_t) max_elems * 4, cudaHostAllocDefault) != cudaSuccess) {
        std::printf("SKIP tp2 pair: no pinned host buffer\n");
        return 77;
    }
    cudaStream_t s_a = nullptr, s_b = nullptr;
    cudaSetDevice(dev_a);
    if (cudaStreamCreate(&s_a) != cudaSuccess) return 77;
    cudaSetDevice(dev_b);
    if (cudaStreamCreate(&s_b) != cudaSuccess) return 77;

    std::mt19937 rng(19);
    std::uniform_real_distribution<float> u(-4.0f, 4.0f);

    for (const Shape& sh : kShapes) {
        const int64_t n = sh.rows * sh.stride;
        for (int64_t i = 0; i < n; ++i) a_host[i] = u(rng);
        std::vector<float> b_host((size_t) n);
        for (int64_t i = 0; i < n; ++i) b_host[i] = u(rng);

        cudaSetDevice(dev_a);
        float* da = dev_alloc(dev_a, sh.rows * sh.stride);
        cudaSetDevice(dev_b);
        float* db = dev_alloc(dev_b, sh.rows * sh.stride);
        if (da == nullptr || db == nullptr) { std::printf("SKIP tp2 pair: out of memory\n"); return 77; }
        cudaSetDevice(dev_a);
        cudaMemcpyAsync(da, a_host, n * 4, cudaMemcpyHostToDevice, s_a);
        cudaSetDevice(dev_b);
        cudaMemcpyAsync(db, b_host.data(), n * 4, cudaMemcpyHostToDevice, s_b);
        cudaSetDevice(dev_a);
        cudaStreamSynchronize(s_a);
        cudaSetDevice(dev_b);
        cudaStreamSynchronize(s_b);

        // ---- the refusals, on a live pair
        {
            std::string e;
            const int64_t issued0 = pair.issued(0), issued1 = pair.issued(1);
            // too wide for the staging: a push this size would run past the peer's buffer
            CHECK(!pair.all_reduce(da, pair.max_elems() + 1, 1, s_a, dev_b, 0, e));
            CHECK(!pair.all_reduce(da, 1, pair.max_elems() + 1, s_a, dev_b, 0, e));
            CHECK(!pair.all_reduce(da, pair.max_elems(), 2, s_a, dev_b, 0, e));
            CHECK(!e.empty());
            CHECK(!pair.all_reduce(da, sh.rows, sh.stride, s_a, dev_a, 0, e));          // its own peer is itself
            CHECK(!pair.all_reduce(da, sh.rows, sh.stride, s_a, dev_b, 2, e));          // no rank 2
            CHECK(!pair.all_reduce(nullptr, sh.rows, sh.stride, s_a, dev_b, 0, e));
            CHECK(!pair.all_reduce(da, 0, sh.stride, s_a, dev_b, 0, e));
            CHECK(pair.issued(0) == issued0 && pair.issued(1) == issued1);   // a refusal issues nothing
        }

        // ---- three rounds: the sequence counter has to carry the pair past its first collective
        for (int round = 0; round < 3; ++round) {
            if (round > 0) {   // new values, so a stale staging buffer cannot pass by luck
                for (int64_t i = 0; i < n; ++i) a_host[i] = u(rng);
                for (int64_t i = 0; i < n; ++i) b_host[i] = u(rng);
                cudaSetDevice(dev_a);
                cudaMemcpyAsync(da, a_host, n * 4, cudaMemcpyHostToDevice, s_a);
                cudaSetDevice(dev_b);
                cudaMemcpyAsync(db, b_host.data(), n * 4, cudaMemcpyHostToDevice, s_b);
                cudaSetDevice(dev_a);
                cudaStreamSynchronize(s_a);
                cudaSetDevice(dev_b);
                cudaStreamSynchronize(s_b);
            }
            std::vector<float> want_a(n), want_b(n);
            tp2_all_reduce_ref(want_a.data(), a_host, b_host.data(), sh.rows, sh.stride);
            tp2_all_reduce_ref(want_b.data(), b_host.data(), a_host, sh.rows, sh.stride);

            const int64_t before_a = pair.issued(0), before_b = pair.issued(1);
            std::string e;
            // each rank on its own stream: the waits must not serialize behind each other's pushes
            if (!pair.all_reduce(da, sh.rows, sh.stride, s_a, dev_b, 0, e) ||
                !pair.all_reduce(db, sh.rows, sh.stride, s_b, dev_a, 1, e)) {
                std::printf("FAIL %s:%d: collective failed: %s\n", __FILE__, __LINE__, e.c_str());
                ++g_fail;
                break;
            }
            CHECK(pair.issued(0) == before_a + 1);
            CHECK(pair.issued(1) == before_b + 1);
            cudaSetDevice(dev_a);
            CHECK(cudaStreamSynchronize(s_a) == cudaSuccess);
            cudaSetDevice(dev_b);
            CHECK(cudaStreamSynchronize(s_b) == cudaSuccess);
            CHECK(!pair.timed_out());

            std::vector<float> got_a((size_t) n), got_b((size_t) n);
            cudaSetDevice(dev_a);
            cudaMemcpy(got_a.data(), da, n * 4, cudaMemcpyDeviceToHost);
            cudaSetDevice(dev_b);
            cudaMemcpy(got_b.data(), db, n * 4, cudaMemcpyDeviceToHost);
            CHECK(bit_eq(got_a.data(), want_a.data(), n));
            CHECK(bit_eq(got_b.data(), want_b.data(), n));
            if (g_fail) {
                for (int64_t i = 0; i < n; ++i)
                    if (got_a[i] != want_a[i]) {
                        std::printf("  %s round %d element %lld: got %.9g want %.9g\n", sh.what, round,
                                    (long long) i, (double) got_a[i], (double) want_a[i]);
                        break;
                    }
                break;
            }
        }
        cudaSetDevice(dev_a);
        cudaFree(da);
        cudaSetDevice(dev_b);
        cudaFree(db);
    }

    // ---- the sampler, on the card: what the kernel wrote must be what the host's rule says, or the pair's
    // agreement check is grading a different sample than it thinks it is.
    {
        const int64_t rows = 64, stride = 2048, n = rows * stride;
        std::vector<float> a((size_t) n), b((size_t) n);
        for (size_t i = 0; i < a.size(); ++i) { a[i] = (float) (i % 97) * 0.5f; b[i] = (float) (i % 13) * 2.0f; }
        cudaSetDevice(dev_a);
        float* da = dev_alloc(dev_a, n);
        float* dchk = dev_alloc(dev_a, kTp2Samples);
        if (da == nullptr || dchk == nullptr) { std::printf("SKIP tp2 pair: out of memory\n"); return 77; }
        cudaMemcpyAsync(da, a.data(), n * 4, cudaMemcpyHostToDevice, s_a);
        cudaStreamSynchronize(s_a);
        std::string se;
        CHECK(tp2_sample_partial(dchk, da, rows, stride, s_a, se));
        float got[kTp2Samples];
        cudaMemcpy(got, dchk, sizeof(got), cudaMemcpyDeviceToHost);
        for (int k = 0; k < kTp2Samples; ++k)
            CHECK(got[k] == a[(size_t) tp2_sample_index(k, rows, stride)]);   // the buffer holds this card's own
        CHECK(tp2_sample_ref(a.data(), b.data(), 0, rows, stride) ==
              a[(size_t) tp2_sample_index(0, rows, stride)] + b[(size_t) tp2_sample_index(0, rows, stride)]);
        CHECK(!tp2_sample_partial(nullptr, da, rows, stride, s_a, se));   // nowhere to write
        CHECK(!tp2_sample_partial(dchk, nullptr, rows, stride, s_a, se)); // nothing to read
        CHECK(!tp2_sample_partial(dchk, da, 0, stride, s_a, se));         // a partial with no element in it
        CHECK(!se.empty());                                               // and it says so
        cudaFree(da); cudaFree(dchk);
    }

    // ---- the microbench line: one shape, many collectives, both ranks in lockstep
    {
        const Shape& sh = kShapes[1];
        const int64_t n = sh.rows * sh.stride, reps = 2000;
        cudaSetDevice(dev_a);
        float* da = dev_alloc(dev_a, n);
        cudaSetDevice(dev_b);
        float* db = dev_alloc(dev_b, n);
        cudaMemset(da, 0, n * 4);
        cudaMemset(db, 0, n * 4);
        cudaEvent_t t0, t1;
        cudaSetDevice(dev_a);
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        cudaSetDevice(dev_a);
        cudaEventRecord(t0, s_a);
        std::string e;
        for (int64_t i = 0; i < reps; ++i) {
            // the ranks stay in lockstep: rank 1's push is queued before either stream is waited on
            if (!pair.all_reduce(da, sh.rows, sh.stride, s_a, dev_b, 0, e) ||
                !pair.all_reduce(db, sh.rows, sh.stride, s_b, dev_a, 1, e)) {
                std::printf("FAIL microbench: %s\n", e.c_str());
                ++g_fail;
                break;
            }
        }
        cudaEventRecord(t1, s_a);
        cudaSetDevice(dev_a);
        cudaStreamSynchronize(s_a);
        cudaSetDevice(dev_b);
        cudaStreamSynchronize(s_b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        const double gib = 1073741824.0;
        std::printf("tp2_all_reduce %lldx%lld fp32: %.3f us per collective, %.2f GiB pushed per rank per second\n",
                (long long) sh.rows, (long long) sh.stride, reps > 0 ? ms * 1000.0 / reps : 0.0,
                ms > 0 ? (double) pair.pushed_bytes(0) / gib / (ms / 1000.0) : 0.0);
        std::printf("tp2_all_reduce counters: rank 0 issued %lld (%.2f MiB), rank 1 issued %lld (%.2f MiB)\n",
                (long long) pair.issued(0), (double) pair.pushed_bytes(0) / 1048576.0,
                (long long) pair.issued(1), (double) pair.pushed_bytes(1) / 1048576.0);
        CHECK(pair.issued(0) == pair.issued(1));   // lockstep, or the test issued them wrong
        cudaSetDevice(dev_a);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(da);
        cudaSetDevice(dev_b);
        cudaFree(db);
    }

    cudaSetDevice(dev_a);
    cudaStreamDestroy(s_a);
    cudaFreeHost(a_host);
    cudaSetDevice(dev_b);
    cudaStreamDestroy(s_b);
    pair.close();
    CHECK(!pair.valid());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: tp2_allreduce_parity [--selftest]\n"); return 2; }
    }

    pure_contract();
    sampler_contract();

    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs < 2) {
        std::printf("tp2_allreduce_parity: %d card%s, the collective needs two\n", devs, devs == 1 ? "" : "s");
        cudaGetLastError();
        if (g_fail) { std::printf("tp2 all-reduce parity: %d failures\n", g_fail); return 1; }
        return selftest ? 0 : 77;
    }

    const int rc = device_path(0, 1);
    std::printf("\ntp2 all-reduce parity: %d failures\n", g_fail);
    if (g_fail) return 1;
    if (rc == 77) return 77;
    if (selftest) std::printf("tp2_allreduce_parity OK\n");
    return 0;
}

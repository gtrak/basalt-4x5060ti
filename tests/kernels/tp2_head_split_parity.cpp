// tests/kernels/tp2_head_split_parity.cpp - the pair's head split, at the level the engine will run it: run the QSA
// decode attention with the whole geometry and with each card's shard, and require the shard's output to be the
// SAME BITS as that card's rows of the whole run.  Not a tolerance - the branch's claim is that splitting heads
// moves sums, never forms one differently, and only a bitwise compare says that.
//
// Four entry points, because the engine uses all four depending on the arm:
//   kv_append_step        the new token's K and V into the paged cache
//   kv_gather_step        the selection through the page table into a contiguous scratch
//   qsa_attend_step       softmax GQA over that scratch
//   qsa_decode_attn_step  the fused pools path with split-K scratch (what a captured layer runs)
//   qsa_decode_attn_batch the same for a verify window of n_q queries
// plus the gate, which is per head and so splits with them.
//
// The shard's pools are built on the host as the KV-head slice of the full pools, which is exactly what a pair's
// card has: the same layout, its own KV heads only, and NOTHING else different - the page table, the selection and
// the step are shared, because the indexer does not split (core/qsa_head_split.hpp).
//   tp2_head_split_parity [max_cells]
#include "basalt/core/qsa_head_split.hpp"
#include "basalt/prefill/kernels.hpp"
#include "basalt/core/tp2_shard_weights.hpp"
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/qsa_prompt_attn.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/kernels/s_gemv.hpp"
#include "basalt/kernels/tp2_collective.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;
namespace pr = basalt::prefill;
using basalt::core::QsaHeadShard;

namespace {
int g_fail = 0;
#define CHECK(...)                                                                                          \
    do {                                                                                                    \
        if (!(__VA_ARGS__)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__); ++g_fail; } \
    } while (0)
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}
uint16_t f16(float x) { __half h = __float2half(x); uint16_t b; std::memcpy(&b, &h, 2); return b; }

/// The pools a rank reads: its own KV heads, the shared page table.
k::QsaAttnPools pools_of(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* table) {
    k::QsaAttnPools p;
    p.k_pool = k_pool;
    p.v_pool = v_pool;
    p.page_table = table;
    return p;
}

void bits(const float* a, const float* b, size_t n, const char* what) {
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            std::printf("FAIL %s: element %zu differs, whole %.9g shard %.9g\n", what, i, (double) a[i],
                    (double) b[i]);
            ++g_fail;
            return;
        }
    }
}
void bits16(const uint16_t* a, const uint16_t* b, size_t n, const char* what) {
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) {
            std::printf("FAIL %s: element %zu differs (%04x vs %04x)\n", what, i, a[i], b[i]);
            ++g_fail;
            return;
        }
}
}  // namespace

int main(int argc, char** argv) {
    const int devs = [] { int n = 0; cudaGetDeviceCount(&n); return n; }();
    if (devs < 1) { std::printf("tp2_head_split_parity: no card\n"); return 77; }
    const int max_cells = argc > 1 ? std::atoi(argv[1]) : 4096;
    const k::QsaShapes full = k::qsa_real_shapes();
    const int64_t D = full.head_dim, P = full.page_size, NH = full.n_head, NKV = full.n_head_kv;
    if (NKV % 2 != 0 || NH % NKV != 0) { std::printf("this geometry does not split two ways\n"); return 77; }
    // The shards are the ones the RATIO API produces, so the kernels are compared through the same plumbing a
    // stage's --exp qsa_ratio_stage_* hands it.  At this geometry the only runnable cut is the symmetric one (the
    // attention kernels move a whole KV group at a time), so an uneven ratio is refused at setup rather than run
    // here; a geometry whose group is narrower than a card's run cuts unevenly and lands in this parity too.
    const QsaHeadShard r0 = basalt::core::qsa_pair_shard(NH, NKV, basalt::core::qsa_lead_heads(NH, 0.5), 0);
    const QsaHeadShard r1 = basalt::core::qsa_pair_shard(NH, NKV, basalt::core::qsa_lead_heads(NH, 0.5), 1);
    if (!(r0.ok() && r1.ok()) || !basalt::core::qsa_shard_whole_groups(r0) ||
        !basalt::core::qsa_shard_whole_groups(r1) || !basalt::core::qsa_pair_tiles(r0, r1)) {
        std::printf("this geometry does not split two ways for the attention kernels\n");
        return 77;
    }
    const k::QsaShapes s0 = basalt::core::qsa_shard_shapes(full, r0), s1 = basalt::core::qsa_shard_shapes(full, r1);
    const int64_t pages = max_cells / P;
    const int64_t n_kv = pages * P - 1;             // one short of full, so a mask bug has somewhere to hide
    const int64_t cap = k::qsa_selection_width(n_kv, full);
    std::printf("heads %lld q / %lld kv, dim %lld, page %lld; %d cells, %lld selected; shard %lld q / %lld kv\n",
            (long long) NH, (long long) NKV, (long long) D, (long long) P, max_cells, (long long) cap,
            (long long) s0.n_head, (long long) s0.n_head_kv);

    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);

    // ---- the pools: the whole cache, and each card's KV-head slice of it
    const size_t rows_full = (size_t) pages * NKV * P, rows_shard = (size_t) pages * r0.nkv() * P;
    std::vector<uint16_t> kf(rows_full * D), vf(rows_full * D);
    for (auto& x : kf) x = f16(nd(rng) * 0.6f);
    for (auto& x : vf) x = f16(nd(rng) * 0.6f);
    auto slice_pool = [&](const std::vector<uint16_t>& full_pool, int64_t kv0) {
        std::vector<uint16_t> out(rows_shard * D);
        for (int64_t pg = 0; pg < pages; ++pg)
            for (int64_t h = 0; h < r0.nkv(); ++h)
                for (int64_t c = 0; c < P; ++c)
                    for (int64_t d = 0; d < D; ++d) {
                        const size_t src =
                            ((size_t) (pg * NKV + kv0 + h) * P + c) * D + d;
                        const size_t dst = ((size_t) (pg * r0.nkv() + h) * P + c) * D + d;
                        out[dst] = full_pool[src];
                    }
        return out;
    };
    std::vector<uint16_t> k0 = slice_pool(kf, r0.kv0()), k1 = slice_pool(kf, r1.kv0());
    std::vector<uint16_t> v0 = slice_pool(vf, r0.kv0()), v1 = slice_pool(vf, r1.kv0());

    // ---- the shared per-token state: one selection, one step, one page table.  Shared is the point.
    std::vector<int32_t> step(k::kStepCount, 0);
    k::qsa_step_fill(step.data(), n_kv - 1, full);
    std::vector<int32_t> ids((size_t) cap);
    for (int64_t i = 0; i < cap; ++i) ids[i] = (int32_t) i;   // ascending, ties by index, what the reference returns
    std::vector<int32_t> table(pages);
    for (int64_t p = 0; p < pages; ++p) table[p] = (int32_t) ((p * 7 + 3) % pages);   // permuted: no identity shortcuts

    // ---- q: one row per query head, and each card's rows of it
    std::vector<float> qf((size_t) NH * D);
    for (auto& x : qf) x = nd(rng);
    auto slice_q = [&](int64_t q0) {
        std::vector<float> out((size_t) s0.n_head * D);
        for (size_t i = 0; i < out.size(); ++i) out[i] = qf[(size_t) q0 * D + i];
        return out;
    };

    // ---- device
    uint16_t *kf_d = dalloc<uint16_t>(rows_full * D), *vf_d = dalloc<uint16_t>(rows_full * D);
    uint16_t *k0_d = dalloc<uint16_t>(rows_shard * D), *v0_d = dalloc<uint16_t>(rows_shard * D);
    uint16_t *k1_d = dalloc<uint16_t>(rows_shard * D), *v1_d = dalloc<uint16_t>(rows_shard * D);
    int32_t *ids_d = dalloc<int32_t>(cap), *table_d = dalloc<int32_t>(pages), *step_d = dalloc<int32_t>(k::kStepCount);
    float *qf_d = dalloc<float>((size_t) NH * D), *q0_d = dalloc<float>((size_t) s0.n_head * D),
          *q1_d = dalloc<float>((size_t) s0.n_head * D);
    uint16_t *ksf_d = dalloc<uint16_t>((size_t) cap * NKV * D), *vsf_d = dalloc<uint16_t>((size_t) cap * NKV * D);
    uint16_t *ks0_d = dalloc<uint16_t>((size_t) cap * r0.nkv() * D), *vs0_d = dalloc<uint16_t>((size_t) cap * r0.nkv() * D);
    uint16_t *ks1_d = dalloc<uint16_t>((size_t) cap * r1.nkv() * D), *vs1_d = dalloc<uint16_t>((size_t) cap * r1.nkv() * D);
    float *attn_f = dalloc<float>((size_t) NH * D), *attn0 = dalloc<float>((size_t) s0.n_head * D),
          *attn1 = dalloc<float>((size_t) s0.n_head * D);
    float *scr_f = dalloc<float>((size_t) k::qsa_decode_attn_scratch_floats(cap, full) + 16);
    float *scr0 = dalloc<float>((size_t) k::qsa_decode_attn_scratch_floats(cap, s0) + 16);
    float *scr1 = dalloc<float>((size_t) k::qsa_decode_attn_scratch_floats(cap, s1) + 16);
    up(kf_d, kf); up(vf_d, vf); up(k0_d, k0); up(v0_d, v0); up(k1_d, k1); up(v1_d, v1);
    up(ids_d, ids); up(table_d, table); up(step_d, step);
    up(qf_d, qf); up(q0_d, slice_q(r0.q0())); up(q1_d, slice_q(r1.q0()));

    // ---- 1. the gather: each card's scratch must be its KV heads' columns of the whole scratch
    k::kv_gather_step(kf_d, vf_d, table_d, ids_d, step_d, cap, full, ksf_d, vsf_d, nullptr);
    k::kv_gather_step(k0_d, v0_d, table_d, ids_d, step_d, cap, s0, ks0_d, vs0_d, nullptr);
    k::kv_gather_step(k1_d, v1_d, table_d, ids_d, step_d, cap, s1, ks1_d, vs1_d, nullptr);
    {
        const std::vector<uint16_t> kf_s = down(ksf_d, (size_t) cap * NKV * D);
        for (const QsaHeadShard* r : {&r0, &r1}) {
            const std::vector<uint16_t> got = down(r == &r0 ? ks0_d : ks1_d, (size_t) cap * r->nkv() * D);
            for (int64_t c = 0; c < cap; ++c)
                for (int64_t h = 0; h < r->nkv(); ++h) {
                    const uint16_t* want = kf_s.data() + ((size_t) c * NKV + r->kv0() + h) * D;
                    const uint16_t* have = got.data() + ((size_t) c * r->nkv() + h) * D;
                    char what[96];
                    std::snprintf(what, sizeof what, "kv_gather K rank %d cell %lld", r->rank, (long long) c);
                    if (c < 4 || c == cap - 1) bits16(have, want, D, what);   // every cell costs a print, sample them
                }
            if (g_fail) break;
        }
        const std::vector<uint16_t> vf_s = down(vsf_d, (size_t) cap * NKV * D);
        for (const QsaHeadShard* r : {&r0, &r1}) {
            const std::vector<uint16_t> got = down(r == &r0 ? vs0_d : vs1_d, (size_t) cap * r->nkv() * D);
            for (int64_t c = 0; c < cap; ++c)
                for (int64_t h = 0; h < r->nkv(); ++h) {
                    const uint16_t* want = vf_s.data() + ((size_t) c * NKV + r->kv0() + h) * D;
                    const uint16_t* have = got.data() + ((size_t) c * r->nkv() + h) * D;
                    if (c < 4 || c == cap - 1) {
                        char what[96];
                        std::snprintf(what, sizeof what, "kv_gather V rank %d cell %lld", r->rank, (long long) c);
                        bits16(have, want, D, what);
                    }
                }
            if (g_fail) break;
        }
    }

    // ---- 2. the split attention over the gathered scratch
    k::qsa_attend_step(qf_d, ksf_d, vsf_d, step_d, cap, full, attn_f, nullptr, nullptr);
    k::qsa_attend_step(q0_d, ks0_d, vs0_d, step_d, cap, s0, attn0, nullptr, nullptr);
    k::qsa_attend_step(q1_d, ks1_d, vs1_d, step_d, cap, s1, attn1, nullptr, nullptr);
    {
        const std::vector<float> w = down(attn_f, (size_t) NH * D);
        const std::vector<float> a0 = down(attn0, (size_t) s0.n_head * D);
        const std::vector<float> a1 = down(attn1, (size_t) s1.n_head * D);
        bits(w.data() + (size_t) r0.q0() * D, a0.data(), (size_t) s0.n_head * D, "qsa_attend_step rank 0");
        bits(w.data() + (size_t) r1.q0() * D, a1.data(), (size_t) s1.n_head * D, "qsa_attend_step rank 1");
    }

    // ---- 3. the fused pools path (what a captured layer runs)
    k::qsa_decode_attn_step(qf_d, pools_of(kf_d, vf_d, table_d), ids_d, step_d, cap, full, full.n_head_kv, scr_f, attn_f, nullptr);
    k::qsa_decode_attn_step(q0_d, pools_of(k0_d, v0_d, table_d), ids_d, step_d, cap, s0, s0.n_head_kv, scr0, attn0, nullptr);
    k::qsa_decode_attn_step(q1_d, pools_of(k1_d, v1_d, table_d), ids_d, step_d, cap, s1, s1.n_head_kv, scr1, attn1, nullptr);
    {
        const std::vector<float> w = down(attn_f, (size_t) NH * D);
        const std::vector<float> a0 = down(attn0, (size_t) s0.n_head * D);
        const std::vector<float> a1 = down(attn1, (size_t) s1.n_head * D);
        bits(w.data() + (size_t) r0.q0() * D, a0.data(), (size_t) s0.n_head * D, "qsa_decode_attn_step rank 0");
        bits(w.data() + (size_t) r1.q0() * D, a1.data(), (size_t) s1.n_head * D, "qsa_decode_attn_step rank 1");
    }

    // ---- 4. a verify window: n_q queries, each with its own selection
    {
        const int64_t n_q = 8;
        std::vector<int32_t> steps((size_t) n_q * k::kStepCount);
        std::vector<int32_t> qids((size_t) n_q * cap);
        for (int64_t q = 0; q < n_q; ++q) {
            k::qsa_step_fill(steps.data() + (size_t) q * k::kStepCount, n_kv - 1 - q, full);
            for (int64_t i = 0; i < cap; ++i) qids[(size_t) q * cap + i] = (int32_t) ((i * 3 + q) % n_kv);
        }
        int32_t* steps_d = dalloc<int32_t>(steps.size());
        int32_t* qids_d = dalloc<int32_t>(qids.size());
        up(steps_d, steps);
        up(qids_d, qids);
        // the batch entry takes [n_q, n_head, head_dim]: each query needs its own q rows, and each card its slice
        std::vector<float> qw((size_t) n_q * NH * D);
        for (auto& x : qw) x = nd(rng);
        std::vector<float> wh0((size_t) n_q * s0.n_head * D), wh1((size_t) n_q * s1.n_head * D);
        for (int64_t q = 0; q < n_q; ++q)
            for (int64_t i = 0; i < s0.n_head * D; ++i) {
                wh0[(size_t) q * s0.n_head * D + i] = qw[(size_t) q * NH * D + (size_t) r0.q0() * D + i];
                wh1[(size_t) q * s1.n_head * D + i] = qw[(size_t) q * NH * D + (size_t) r1.q0() * D + i];
            }
        float* qw_d = dalloc<float>(qw.size());
        float* wh0_d = dalloc<float>(wh0.size());
        float* wh1_d = dalloc<float>(wh1.size());
        up(qw_d, qw);
        up(wh0_d, wh0);
        up(wh1_d, wh1);
        const size_t sq = (size_t) k::qsa_decode_attn_scratch_floats(cap, full) * n_q + 16;
        float* sq_f = dalloc<float>(sq);
        float* sq_0 = dalloc<float>(sq);
        float* sq_1 = dalloc<float>(sq);
        float* af = dalloc<float>((size_t) n_q * NH * D);
        float* a0 = dalloc<float>((size_t) n_q * s0.n_head * D);
        float* a1 = dalloc<float>((size_t) n_q * s1.n_head * D);
        k::qsa_decode_attn_batch(qw_d, pools_of(kf_d, vf_d, table_d), qids_d, steps_d, cap, full, full.n_head_kv, sq_f, af, n_q, nullptr);
        k::qsa_decode_attn_batch(wh0_d, pools_of(k0_d, v0_d, table_d), qids_d, steps_d, cap, s0, s0.n_head_kv, sq_0, a0, n_q, nullptr);
        k::qsa_decode_attn_batch(wh1_d, pools_of(k1_d, v1_d, table_d), qids_d, steps_d, cap, s1, s1.n_head_kv, sq_1, a1, n_q, nullptr);
        const std::vector<float> w = down(af, (size_t) n_q * NH * D);
        const std::vector<float> h0 = down(a0, (size_t) n_q * s0.n_head * D);
        const std::vector<float> h1 = down(a1, (size_t) n_q * s1.n_head * D);
        for (int64_t q = 0; q < n_q; ++q) {
            char what[80];
            std::snprintf(what, sizeof what, "window query %lld rank 0", (long long) q);
            bits(w.data() + (size_t) q * NH * D + (size_t) r0.q0() * D, h0.data() + (size_t) q * s0.n_head * D,
                 (size_t) s0.n_head * D, what);
            std::snprintf(what, sizeof what, "window query %lld rank 1", (long long) q);
            bits(w.data() + (size_t) q * NH * D + (size_t) r1.q0() * D, h1.data() + (size_t) q * s1.n_head * D,
                 (size_t) s1.n_head * D, what);
        }
        cudaFree(steps_d); cudaFree(qids_d); cudaFree(sq_f); cudaFree(sq_0); cudaFree(sq_1);
        cudaFree(qw_d); cudaFree(wh0_d); cudaFree(wh1_d);
        cudaFree(af); cudaFree(a0); cudaFree(a1);
    }

    // ---- 5. the gate, which is per head.  It insists on a real stream, unlike the attention entries.
    {
        cudaStream_t st = nullptr;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        std::vector<float> qgate((size_t) NH * 2 * D);   // q | gate per head, as attn_q leaves it
        for (auto& x : qgate) x = nd(rng);
        float* qg_d = dalloc<float>(qgate.size());
        up(qg_d, qgate);
        float* g_f = dalloc<float>((size_t) NH * D);
        float* g0 = dalloc<float>((size_t) s0.n_head * D);
        float* g1 = dalloc<float>((size_t) s1.n_head * D);
        k::native_qsa_gate_apply(attn_f, qg_d, g_f, (int) NH, (int) D, st);
        // the shard's gate rows are its query heads' halves of the same buffer
        std::vector<float> qg0((size_t) s0.n_head * 2 * D), qg1((size_t) s1.n_head * 2 * D);
        for (size_t i = 0; i < qg0.size(); ++i) qg0[i] = qgate[(size_t) r0.q0() * 2 * D + i];
        for (size_t i = 0; i < qg1.size(); ++i) qg1[i] = qgate[(size_t) r1.q0() * 2 * D + i];
        float* qg0_d = dalloc<float>(qg0.size());
        float* qg1_d = dalloc<float>(qg1.size());
        up(qg0_d, qg0);
        up(qg1_d, qg1);
        k::native_qsa_gate_apply(attn0, qg0_d, g0, (int) s0.n_head, (int) D, st);
        k::native_qsa_gate_apply(attn1, qg1_d, g1, (int) s1.n_head, (int) D, st);
        const std::vector<float> w = down(g_f, (size_t) NH * D);
        const std::vector<float> h0 = down(g0, (size_t) s0.n_head * D);
        const std::vector<float> h1 = down(g1, (size_t) s1.n_head * D);
        bits(w.data() + (size_t) r0.q0() * D, h0.data(), (size_t) s0.n_head * D, "gate rank 0");
        bits(w.data() + (size_t) r1.q0() * D, h1.data(), (size_t) s1.n_head * D, "gate rank 1");
        ck(cudaStreamSynchronize(st), "gate wait");
        cudaStreamDestroy(st);
        cudaFree(qg_d); cudaFree(qg0_d); cudaFree(qg1_d); cudaFree(g_f); cudaFree(g0); cudaFree(g1);
    }

    // ---- 6. the microbench line: the attention is what a split window is supposed to halve
    {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        const int reps = 200;
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i)
            k::qsa_decode_attn_step(qf_d, pools_of(kf_d, vf_d, table_d), ids_d, step_d, cap, full, full.n_head_kv, scr_f, attn_f, nullptr);
        cudaEventRecord(b);
        cudaStreamSynchronize(nullptr);
        float whole_ms = 0.f;
        cudaEventElapsedTime(&whole_ms, a, b);
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) {
            k::qsa_decode_attn_step(q0_d, pools_of(k0_d, v0_d, table_d), ids_d, step_d, cap, s0, s0.n_head_kv, scr0, attn0, nullptr);
            k::qsa_decode_attn_step(q1_d, pools_of(k1_d, v1_d, table_d), ids_d, step_d, cap, s1, s1.n_head_kv, scr1, attn1, nullptr);
        }
        cudaEventRecord(b);
        cudaStreamSynchronize(nullptr);
        float pair_ms = 0.f;
        cudaEventElapsedTime(&pair_ms, a, b);
        std::printf("qsa decode attention %lld selected: whole %.2f us, the pair's two cards back to back on one "
                    "card %.2f us (%.2fx)\n",
                (long long) cap, whole_ms * 1000.0 / reps, pair_ms * 1000.0 / reps,
                whole_ms > 0 ? pair_ms / whole_ms : 0.0);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }

    // ---- 7. the WIRED combine.  Everything above is what a card computes for its own heads; this is what the
    // engine does with it: each card's gated attention goes through ITS columns of `attn_output` (the packed
    // view of `tp2_shard_weights.hpp`, driven here through the same GEMV entry the layer calls) and the pair's
    // all-reduce sums the two partials.  Step and batch forms, because the layer uses both.
    //
    // The comparison is a tolerance, not bitwise, and that is the branch's honest position: the column split
    // reassociates the o_proj sum (`tp2_all_reduce_ref` fixes the order, the whole-layer GEMV does not have it).
    // The COMBINE itself is checked bitwise against that reference, which is the part that must not drift.
    {
        const int64_t n_out = 64;                    // n_embd stand-in: this section is about the split, not the size
        const int64_t n_in = NH * D;                 // the o_proj's input axis, the head axis
        const int64_t group = 32, bits = 4;
        basalt::kernels::SForm form;
        form.code_bits = (int) bits;
        form.code_bias = -8;
        form.group_elems = (int) group;
        form.has_offset = true;
        form.codebook = basalt::kernels::Codebook::Affine;
        form.act_kind = 1;   // Q8_K, as every K-quant attn_output is

        const uint64_t codes_bytes = (uint64_t) n_out * (n_in * bits / 8);
        const uint64_t scales_bytes = (uint64_t) n_out * (n_in / group) * 4;
        std::vector<uint8_t> codes((size_t) codes_bytes), pc0, pc1;
        std::vector<float> scales((size_t) n_out * (n_in / group)), offsets((size_t) n_out * (n_in / group));
        std::mt19937 rw(99);
        std::uniform_int_distribution<int> cd(0, 255);
        for (auto& c : codes) c = (uint8_t) cd(rw);
        std::uniform_real_distribution<float> sd(0.004f, 0.05f);
        for (auto& x : scales) x = sd(rw);
        for (auto& x : offsets) x = (float) (int) (rw() % 16) * 0.001f;

        basalt::core::WeightRef w;
        w.ne0 = n_in;
        w.ne1 = n_out;
        w.code_bits = (int) bits;
        w.code_bias = -8;
        w.group_elems = (int) group;
        w.has_offset = true;
        w.codes_bytes = codes_bytes;
        w.scales_bytes = scales_bytes;
        w.offset_bytes = scales_bytes;
        w.bytes = codes_bytes + 2 * scales_bytes;
        // the packer reads bytes, so it gets a host tensor: the planes are laid out as the loader lays them
        std::vector<uint8_t> host_plane((size_t) w.bytes);
        memcpy(host_plane.data(), codes.data(), (size_t) codes_bytes);
        memcpy(host_plane.data() + codes_bytes, scales.data(), (size_t) scales_bytes);
        memcpy(host_plane.data() + codes_bytes + scales_bytes, offsets.data(), (size_t) scales_bytes);
        w.data = host_plane.data();

        const basalt::core::QsaHeadRun c0 = basalt::core::qsa_out_cols(r0, D),
            c1 = basalt::core::qsa_out_cols(r1, D);
        const uint64_t want0 = basalt::core::qsa_weight_cols_bytes(w, c0.start, c0.len, 0);
        CHECK(want0 > 0 && want0 == basalt::core::qsa_weight_cols_bytes(w, c1.start, c1.len, 0));
        pc0.resize((size_t) want0);
        pc1.resize((size_t) want0);
        const basalt::core::QsaShardWeight v0 =
            basalt::core::qsa_pack_weight_cols(w, c0.start, c0.len, 0, pc0.data());
        const basalt::core::QsaShardWeight v1 =
            basalt::core::qsa_pack_weight_cols(w, c1.start, c1.len, 0, pc1.data());
        CHECK(v0.ok && v1.ok && v0.ne0 == c0.len && v1.ne0 == c1.len && v0.ne1 == n_out);

        // device copies: the whole planes for the reference GEMV, the packed ones for each card
        auto upload = [](const void* p, size_t n) {
            void* d = nullptr;
            ck(cudaMalloc(&d, n + 256), "gemv malloc");
            ck(cudaMemset(d, 0, n + 256), "gemv memset");
            ck(cudaMemcpy(d, p, n, cudaMemcpyHostToDevice), "gemv up");
            return d;
        };
        void* cw = upload(host_plane.data(), (size_t) w.bytes);
        const uint8_t* cc = (const uint8_t*) cw;
        const float* cs = (const float*) (cw + (size_t) codes_bytes);
        const float* co = (const float*) (cw + (size_t) (codes_bytes + scales_bytes));
        const basalt::core::QsaShardWeight vw0 = [&] {
            void* d = upload(pc0.data(), (size_t) want0);
            basalt::core::QsaShardWeight v = v0;
            v.codes = (const uint8_t*) d;
            v.scales = (const float*) ((uint8_t*) d + v.codes_bytes);
            v.offset = (const float*) ((uint8_t*) d + v.codes_bytes + v.scales_bytes);
            return v;
        }();
        const basalt::core::QsaShardWeight vw1 = [&] {
            void* d = upload(pc1.data(), (size_t) want0);
            basalt::core::QsaShardWeight v = v1;
            v.codes = (const uint8_t*) d;
            v.scales = (const float*) ((uint8_t*) d + v.codes_bytes);
            v.offset = (const float*) ((uint8_t*) d + v.codes_bytes + v.scales_bytes);
            return v;
        }();

        // the gated attention outputs, which is what the layer feeds the o_proj
        float *g_full_d = dalloc<float>((size_t) NH * D), *g_a = dalloc<float>((size_t) s0.n_head * D),
              *g_b = dalloc<float>((size_t) s1.n_head * D);
        {
            cudaStream_t st = nullptr;
            ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "gate stream");
            std::vector<float> qgate((size_t) NH * 2 * D);
            for (auto& x : qgate) x = nd(rng);
            float* qg_d = dalloc<float>(qgate.size());
            up(qg_d, qgate);
            std::vector<float> qg0((size_t) s0.n_head * 2 * D), qg1((size_t) s1.n_head * 2 * D);
            for (size_t i = 0; i < qg0.size(); ++i) qg0[i] = qgate[(size_t) r0.q0() * 2 * D + i];
            for (size_t i = 0; i < qg1.size(); ++i) qg1[i] = qgate[(size_t) r1.q0() * 2 * D + i];
            float* qg0_d = dalloc<float>(qg0.size());
            float* qg1_d = dalloc<float>(qg1.size());
            up(qg0_d, qg0);
            up(qg1_d, qg1);
            // the attention outputs of sections 2-4 are still on the device; gate them exactly as the layer does
            k::native_qsa_gate_apply(attn_f, qg_d, g_full_d, (int) NH, (int) D, st);
            k::native_qsa_gate_apply(attn0, qg0_d, g_a, (int) s0.n_head, (int) D, st);
            k::native_qsa_gate_apply(attn1, qg1_d, g_b, (int) s1.n_head, (int) D, st);
            ck(cudaStreamSynchronize(st), "gate wait");
            cudaStreamDestroy(st);
            cudaFree(qg_d); cudaFree(qg0_d); cudaFree(qg1_d);
        }
        uint8_t *xk_f = dalloc<uint8_t>((size_t) n_in * 34 + 256), *xk_a = dalloc<uint8_t>((size_t) n_in * 34 + 256),
                *xk_b = dalloc<uint8_t>((size_t) n_in * 34 + 256);
        k::quantize_q8_K(g_full_d, xk_f, n_in, nullptr);
        // each card quantizes only its own heads' columns: the same bytes of the same vector, half as long
        k::quantize_q8_K(g_a, xk_a, c0.len, nullptr);
        k::quantize_q8_K(g_b, xk_b, c1.len, nullptr);
        float *y_f = dalloc<float>(n_out), *y_a = dalloc<float>(n_out), *y_b = dalloc<float>(n_out);
        k::s_gemv_q8k_split(xk_f, cc, cs, co, y_f, n_in, n_out, form, nullptr);
        k::s_gemv_q8k_split(xk_a, vw0.codes, vw0.scales, vw0.offset, y_a, c0.len, n_out, form, nullptr);
        k::s_gemv_q8k_split(xk_b, vw1.codes, vw1.scales, vw1.offset, y_b, c1.len, n_out, form, nullptr);
        std::vector<float> wy = down(y_f, n_out), pa = down(y_a, n_out), pb = down(y_b, n_out);
        double maxabs = 0.0;
        for (float v : wy) maxabs = std::max(maxabs, (double) std::fabs(v));
        // a GEMV that produced nothing would make every comparison below vacuously true
        CHECK(maxabs > 1e-3);
        double worst = 0.0;
        for (int64_t i = 0; i < n_out; ++i)
            worst = std::max(worst, std::fabs((double) pa[i] + (double) pb[i] - (double) wy[i]));
        if (!(worst <= 1e-4 * std::max(1.0, maxabs))) {
            std::printf("FAIL the packed column views do not reproduce the whole o_proj: worst %.3g, max |y| %.3g\n",
                    worst, maxabs);
            ++g_fail;
        }
        std::printf("  o_proj through the packed column views: worst |split - whole| %.3g, max |y| %.3g\n", worst,
                maxabs);

        // ---- the combine itself, on TWO cards, against `tp2_all_reduce_ref`: bitwise, because the order is the
        // contract.  A wrong offset or a wrong rank here shows up as a differing bit, not a tolerance.
        basalt::kernels::Tp2Pair pair;
        std::string perr;
        const int dev0 = 0, dev1 = 1;
        bool can = false;
        {
            int peer01 = 0;
            cudaDeviceCanAccessPeer(&peer01, dev0, dev1);
            can = peer01 == 1;
        }
        if (!can || !pair.open(dev0, dev1, 4096, perr)) {
            std::printf("  the combine needs two peer-accessible cards; skipped (%s)\n", perr.c_str());
        } else {
            // One stream PER CARD, which is the pair's stated condition: on the legacy default stream rank 0's
            // waiting kernel would block rank 1's push, and the collective would only end at the timeout.
            float* buf[2] = {nullptr, nullptr};
            cudaStream_t str[2] = {nullptr, nullptr};
            for (int r = 0; r < 2; ++r) {
                ck(cudaSetDevice(r), "set dev");
                ck(cudaMalloc((void**) &buf[r], (size_t) n_out * 4 + 256), "pair buf");
                ck(cudaMemset(buf[r], 0, (size_t) n_out * 4 + 256), "pair zero");
                ck(cudaMemcpy(buf[r], (r == 0 ? pa : pb).data(), (size_t) n_out * 4, cudaMemcpyHostToDevice), "pair up");
                ck(cudaStreamCreateWithFlags(&str[r], cudaStreamNonBlocking), "pair stream");
            }
            std::string e;
            CHECK(pair.all_reduce(buf[0], 1, n_out, str[0], dev1, 0, e));
            CHECK(pair.all_reduce(buf[1], 1, n_out, str[1], dev0, 1, e));
            for (int r = 0; r < 2; ++r) {
                ck(cudaSetDevice(r), "set dev for the wait");
                ck(cudaStreamSynchronize(str[r]), "pair wait");
            }
            std::vector<float> got0, got1;
            ck(cudaSetDevice(dev0), "down 0");
            got0 = down(buf[0], n_out);
            ck(cudaSetDevice(dev1), "down 1");
            got1 = down(buf[1], n_out);
            ck(cudaSetDevice(dev0), "back");
            cudaStreamDestroy(str[0]);
            cudaStreamDestroy(str[1]);
            std::vector<float> ref((size_t) n_out);
            basalt::kernels::tp2_all_reduce_ref(ref.data(), pa.data(), pb.data(), 1, n_out);
            for (int i = 0; i < n_out; ++i) {
                if (memcmp(&got0[i], &ref[i], 4) || memcmp(&got1[i], &ref[i], 4)) {
                    std::printf("FAIL the pair's sum is not tp2_all_reduce_ref at %d\n", i);
                    ++g_fail;
                    break;
                }
            }
            // both cards end with the SAME sum: that is what the residual add downstream depends on
            CHECK(got0 == got1);
            CHECK(pair.issued(0) == 1 && pair.issued(1) == 1);
            CHECK(pair.pushed_bytes(0) == basalt::kernels::tp2_all_reduce_bytes(1, n_out));
            CHECK(!pair.timed_out());

            // ---- the batch (verify-window) form at the SAME combine point: `rows` partials per card, one
            // collective, each partial built from this card's columns of the weight.  This is the shape a verify
            // window hands the layer, and the one a pitch or stride bug would survive unnoticed.
            const int64_t rows = 8;
            const int64_t per[2] = {c0.len, c1.len};
            std::vector<std::vector<float>> xrow((size_t) rows, std::vector<float>((size_t) n_in));
            std::mt19937 rb(4242);
            std::normal_distribution<float> nx(0.f, 1.f);
            for (int64_t q = 0; q < rows; ++q)
                for (int64_t i = 0; i < n_in; ++i) xrow[q][(size_t) i] = nx(rb);
            float* bb[2] = {nullptr, nullptr};
            float* xin[2] = {nullptr, nullptr};
            uint8_t* xk[2] = {nullptr, nullptr};
            cudaStream_t bstr[2] = {nullptr, nullptr};
            for (int r = 0; r < 2; ++r) {
                ck(cudaSetDevice(r), "batch set dev");
                ck(cudaMalloc((void**) &bb[r], (size_t) rows * n_out * 4 + 256), "batch buf");
                ck(cudaMemset(bb[r], 0, (size_t) rows * n_out * 4 + 256), "batch zero");
                ck(cudaMalloc((void**) &xin[r], (size_t) rows * per[r] * 4 + 256), "batch in");
                ck(cudaMalloc((void**) &xk[r], (size_t) rows * per[r] * 34 + 256), "batch q8k");
                ck(cudaStreamCreateWithFlags(&bstr[r], cudaStreamNonBlocking), "batch stream");
                // card r holds the SAME window's columns for its heads: card 0 the first `per`, card 1 the last
                for (int64_t q = 0; q < rows; ++q)
                    ck(cudaMemcpy(xin[r] + (size_t) q * per[r], xrow[q].data() + (size_t) r * per[r],
                                  (size_t) per[r] * 4, cudaMemcpyHostToDevice), "batch in up");
            }
            for (int r = 0; r < 2; ++r) {
                ck(cudaSetDevice(r), "batch gemv dev");
                const basalt::core::QsaShardWeight& v = r == 0 ? vw0 : vw1;
                for (int64_t q = 0; q < rows; ++q) {
                    // one GEMV per row: the entry is a mat-vec, so a window is `rows` of them, each with its own
                    // quantized activation image - the same thing the layer does per token
                    k::quantize_q8_K(xin[r] + (size_t) q * per[r], xk[r] + (size_t) q * per[r] * 34, per[r], bstr[r]);
                    k::s_gemv_q8k_split(xk[r] + (size_t) q * per[r] * 34, v.codes, v.scales, v.offset,
                                        bb[r] + (size_t) q * n_out, per[r], n_out, form, bstr[r]);
                }
            }
            std::string be;
            CHECK(pair.all_reduce(bb[0], rows, n_out, bstr[0], dev1, 0, be));
            CHECK(pair.all_reduce(bb[1], rows, n_out, bstr[1], dev0, 1, be));
            for (int r = 0; r < 2; ++r) {
                ck(cudaSetDevice(r), "batch wait dev");
                ck(cudaStreamSynchronize(bstr[r]), "batch wait");
            }
            std::vector<float> b0, b1;
            ck(cudaSetDevice(dev0), "batch down 0");
            b0 = down(bb[0], rows * n_out);
            ck(cudaSetDevice(dev1), "batch down 1");
            b1 = down(bb[1], rows * n_out);
            ck(cudaSetDevice(dev0), "back from batch");
            CHECK(b0 == b1);                                  // both cards, every row of the window
            CHECK(pair.issued(0) == 2 && pair.issued(1) == 2);  // one issue per collective, whatever the shape
            CHECK(pair.pushed_bytes(0) == basalt::kernels::tp2_all_reduce_bytes(1, n_out) +
                                                basalt::kernels::tp2_all_reduce_bytes(rows, n_out));
            CHECK(!pair.timed_out());
            std::printf("  the pair's combine, step and %lld-row window: bitwise vs tp2_all_reduce_ref, both cards"
                        " equal, %llu B pushed per rank\n",
                    (long long) rows, (unsigned long long) pair.pushed_bytes(0));
            for (int64_t q = 0; q < rows; ++q) {
                // the whole layer's GEMV for that window row, on one card, one activation of the full width
                float* xs = nullptr;
                uint8_t* xsq = nullptr;
                float* yq = nullptr;
                ck(cudaMalloc((void**) &xs, (size_t) n_in * 4 + 256), "row x");
                ck(cudaMalloc((void**) &xsq, (size_t) n_in * 34 + 256), "row q8k");
                ck(cudaMalloc((void**) &yq, (size_t) n_out * 4 + 256), "row y");
                ck(cudaMemcpy(xs, xrow[q].data(), (size_t) n_in * 4, cudaMemcpyHostToDevice), "row x up");
                k::quantize_q8_K(xs, xsq, n_in, nullptr);
                k::s_gemv_q8k_split(xsq, cc, cs, co, yq, n_in, n_out, form, nullptr);
                const std::vector<float> whole_row = down(yq, n_out);
                double m = 0.0;
                for (float v : whole_row) m = std::max(m, (double) std::fabs(v));
                CHECK(m > 1e-3);   // a row of nothing would make the comparison below vacuous
                for (int64_t i = 0; i < n_out; ++i) {
                    const double d = std::fabs((double) b0[(size_t) q * n_out + i] - (double) whole_row[i]);
                    if (!(d <= 1e-4 * std::max(1.0, m))) {
                        std::printf("FAIL batch combine row %lld col %lld: %.9g vs %.9g\n", (long long) q,
                                (long long) i, b0[(size_t) q * n_out + i], whole_row[i]);
                        ++g_fail;
                        break;
                    }
                }
                cudaFree(xs); cudaFree(xsq); cudaFree(yq);
            }
            for (int r = 0; r < 2; ++r) {
                ck(cudaSetDevice(r), "batch free dev");
                cudaFree(bb[r]); cudaFree(xin[r]); cudaFree(xk[r]);
                cudaStreamDestroy(bstr[r]);
            }
            ck(cudaSetDevice(dev0), "back after batch free");
            cudaFree(buf[0]);
            cudaFree(buf[1]);
            pair.close();
        }
        cudaFree(g_full_d); cudaFree(g_a); cudaFree(g_b);
        cudaFree(xk_f); cudaFree(xk_a); cudaFree(xk_b);
        cudaFree(y_f); cudaFree(y_a); cudaFree(y_b);
        cudaFree(cw);
    }

    // ---- 6. the PROMPT attention, split vs whole: the case the prompt-attention slice runs
    //
    // `qsa_prompt_attn_batch` is the tensor-core prompt kernel (qsa_prompt_attn.hpp), the one prefill calls once a
    // QSA layer with the whole n_head.  The slice's claim is that a card running its own query-head run over its
    // own KV-head slice reproduces THAT card's rows of the whole run, so the o_proj column split sums two partials
    // that are already correct.  Several queries, because the kernel is a batch one block per (query, KV head) and
    // a single query would leave its per-query indexing untested.
    //
    // TOLERANCE, and why it is not the bitwise compare the decode cases above use: the prompt kernel is FP32-level
    // but NOT bitwise equal to the decode kernel it replaces (qsa_prompt_attn.hpp says so, and
    // `qsa_prompt_attn_parity` bounds the difference).  Both sides of THIS compare run the same kernel, and a
    // head's sum touches only its own KV head and its own head dimension, so the sums are expected to be identical
    // and are reported as such; the accepted bound is the prompt kernel's own: relative 2e-3 of the row's largest
    // magnitude, the same shape as the combine check below.  A bitwise run prints `bitwise`; a run inside the
    // bound prints its worst deviation, so a regression towards the bound is visible and not silently allowed.
    {
        const int64_t n_q = 8;
        std::vector<int32_t> p_ids((size_t) n_q * cap), p_steps((size_t) n_q * k::kStepCount);
        for (int64_t i = 0; i < n_q; ++i) {
            const int64_t nkv_i = std::max<int64_t>(1, n_kv - 1 - 16 * i);   // a different selection per query
            k::qsa_step_fill(p_steps.data() + (size_t) i * k::kStepCount, nkv_i, full);
            for (int64_t j = 0; j < cap; ++j) p_ids[(size_t) i * cap + j] = (int32_t) j;
        }
        std::vector<float> pqf((size_t) n_q * NH * D);
        for (auto& x : pqf) x = nd(rng);

        int32_t *pids_d = dalloc<int32_t>((size_t) n_q * cap), *pstep_d = dalloc<int32_t>((size_t) n_q * k::kStepCount);
        float *pqf_d = dalloc<float>((size_t) n_q * NH * D);
        float *pq0_d = dalloc<float>((size_t) n_q * s0.n_head * D), *pq1_d = dalloc<float>((size_t) n_q * s0.n_head * D);
        float *pattn_f = dalloc<float>((size_t) n_q * NH * D);
        float *pattn0 = dalloc<float>((size_t) n_q * s0.n_head * D), *pattn1 = dalloc<float>((size_t) n_q * s0.n_head * D);
        up(pids_d, p_ids); up(pstep_d, p_steps); up(pqf_d, pqf);
        {
            std::vector<float> qa((size_t) n_q * s0.n_head * D), qb((size_t) n_q * s0.n_head * D);
            for (int64_t i = 0; i < n_q; ++i)
                for (int64_t h = 0; h < s0.n_head; ++h) {
                    for (int64_t d = 0; d < D; ++d)   // rank 0 runs heads [0, nq), rank 1 the rest
                        qa[(size_t) i * s0.n_head * D + h * D + d] = pqf[(size_t) i * NH * D + r0.q0() * D + h * D + d];
                    for (int64_t d = 0; d < D; ++d)
                        qb[(size_t) i * s0.n_head * D + h * D + d] = pqf[(size_t) i * NH * D + r1.q0() * D + h * D + d];
                }
            up(pq0_d, qa); up(pq1_d, qb);
        }
        const k::QsaAttnPools pools_full{kf_d, vf_d, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, table_d};
        const k::QsaAttnPools pools0{k0_d, v0_d, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, table_d};
        const k::QsaAttnPools pools1{k1_d, v1_d, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, table_d};
        const bool took_f = k::qsa_prompt_attn_batch(pqf_d, pools_full, pids_d, pstep_d, cap, full, pattn_f, n_q, nullptr);
        const bool took_0 = k::qsa_prompt_attn_batch(pq0_d, pools0, pids_d, pstep_d, cap, s0, pattn0, n_q, nullptr);
        const bool took_1 = k::qsa_prompt_attn_batch(pq1_d, pools1, pids_d, pstep_d, cap, s1, pattn1, n_q, nullptr);
        if (!took_f || !took_0 || !took_1) {
            std::printf("[prompt attn] skipped: the tensor-core prompt kernel refused the geometry "
                        "(whole %d, rank0 %d, rank1 %d)\n", (int) took_f, (int) took_0, (int) took_1);
        } else {
            const std::vector<float> wf = down(pattn_f, (size_t) n_q * NH * D);
            double worst = 0.0;   bool bitwise = true;
            for (const QsaHeadShard* r : {&r0, &r1}) {
                const std::vector<float> got = down(r == &r0 ? pattn0 : pattn1, (size_t) n_q * s0.n_head * D);
                for (int64_t i = 0; i < n_q; ++i) {
                    double m = 0.0;
                    for (int64_t h = 0; h < NH; ++h)
                        for (int64_t d = 0; d < D; ++d)
                            m = std::max(m, (double) std::fabs(wf[(size_t) i * NH * D + (r->q0() + h) * D + d]));
                    CHECK(m > 1e-3);   // a row of nothing would make the comparison vacuous
                    for (int64_t h = 0; h < r->nq(); ++h)
                        for (int64_t d = 0; d < D; ++d) {
                            const double a = got[(size_t) i * r->nq() * D + h * D + d];
                            const double b = wf[(size_t) i * NH * D + (r->q0() + h) * D + d];
                            const double dd = std::fabs(a - b);
                            if (dd != 0.0) bitwise = false;
                            if (dd > worst) worst = dd;
                            if (dd <= 2e-3 * std::max(1.0, m)) continue;
                            std::printf("FAIL prompt attn rank %d query %lld head %lld dim %lld: %.9g vs %.9g\n",
                                    r->rank, (long long) i, (long long) (r->q0() + h), (long long) d, a, b);
                            ++g_fail;
                            h = r->nq();   // one report per query is enough to localise it
                            break;
                        }
                }
            }
            std::printf("[prompt attn] %lld queries x %lld heads: %s, worst deviation %.3g\n", (long long) n_q,
                    (long long) NH, bitwise ? "bitwise vs the whole run" : "inside the prompt kernel's 2e-3 relative bound",
                    worst);
        }
        cudaFree(pids_d); cudaFree(pstep_d); cudaFree(pqf_d); cudaFree(pq0_d); cudaFree(pq1_d);
        cudaFree(pattn_f); cudaFree(pattn0); cudaFree(pattn1);
    }

    // ---- 7. split_q and gate_attn, split vs whole: the two kernels the prompt-attention arm needed widened
    //
    // These two were the head split's blind spot: every other entry point here takes its counts as arguments,
    // but these hardcoded 24 query heads and a 24-head q_full stride, so a card running 12 heads would have
    // read heads 12..23 of the lead's rows and produced plausible tokens silently.  They are now one span
    // kernel with the whole layer as its h0=0 case, so the pair's shard and the lead's layer share a body.
    //
    // BITWISE, not a tolerance: both sides run the same body over the same inputs, and neither forms a sum -
    // split_q is a copy and gate_attn is one elementwise product per output.  Anything but identical bits means
    // the span's indexing disagrees with the whole layer's, which is not a precision question but a wrong read.
    // Random inputs, real geometry, and the shards the engine actually runs.
    {
        const int64_t T7 = 37;   // not a multiple of anything: a stride error shows up at the row edges
        const int64_t nq7 = 2 * 256;   // q_full's per-head float stride (query | gate)
        std::vector<float> qf((size_t) T7 * NH * nq7);
        for (auto& x : qf) x = nd(rng);
        std::vector<float> attn((size_t) T7 * NH * D);
        for (auto& x : attn) x = nd(rng);
        float* qf_d = dalloc<float>((size_t) T7 * NH * nq7);
        float* at_d = dalloc<float>((size_t) T7 * NH * D);
        up(qf_d, qf); up(at_d, attn);

        float* qf_whole = dalloc<float>((size_t) T7 * NH * D);
        uint16_t* g_whole = dalloc<uint16_t>((size_t) T7 * NH * D);
        pr::split_q(qf_d, qf_whole, T7, nullptr);
        pr::gate_attn(at_d, qf_d, g_whole, T7, nullptr);
        const std::vector<float> qw = down(qf_whole, (size_t) T7 * NH * D);
        const std::vector<uint16_t> gw = down(g_whole, (size_t) T7 * NH * D);

        bool bitwise = true;
        for (const QsaHeadShard* r : {&r0, &r1}) {
            const int64_t nh = r->nq();
            float* qs_d = dalloc<float>((size_t) T7 * nh * D);
            uint16_t* gs_d = dalloc<uint16_t>((size_t) T7 * nh * D);
            pr::split_q_span(qf_d, qs_d, T7, r->q0(), nh, NH * nq7, nullptr);
            pr::gate_attn_span(at_d, qf_d, gs_d, T7, r->q0(), nh, NH * nq7, nullptr);
            const std::vector<float> gq = down(qs_d, (size_t) T7 * nh * D);
            const std::vector<uint16_t> gg = down(gs_d, (size_t) T7 * nh * D);
            int reported = 0;
            for (int64_t t = 0; t < T7; ++t)
                for (int64_t h = 0; h < nh && reported < 3; ++h)
                    for (int64_t d = 0; d < D && reported < 3; ++d) {
                        const size_t w = (size_t) t * NH * D + (r->q0() + h) * D + d;
                        const size_t g = (size_t) t * nh * D + h * D + d;
                        if (gq[g] == qw[w] && gg[g] == gw[w]) continue;
                        bitwise = false;
                        std::printf("FAIL split span rank %d t %lld head %lld dim %lld: %.9g/%u vs %.9g/%u\n",
                                r->rank, (long long) t, (long long) (r->q0() + h), (long long) d,
                                (double) gq[g], (unsigned) gg[g], (double) qw[w], (unsigned) gw[w]);
                        ++g_fail;
                        ++reported;
                    }
            cudaFree(qs_d); cudaFree(gs_d);
        }
        std::printf("[split/gate span] T %lld x %lld heads over shards %lld/%lld: %s\n", (long long) T7,
                (long long) NH, (long long) r0.nq(), (long long) r1.nq(),
                bitwise ? "bitwise vs the whole run" : "NOT bitwise");

        // microbench line (AGENTS.md: every new kernel gets one).  The span is the same work per element, so
        // this exists to catch a shard costing more than the whole layer it replaces, not to claim a win.
        {
            const int reps = 50;
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            float* qs_d = dalloc<float>((size_t) T7 * r0.nq() * D);
            uint16_t* gs_d = dalloc<uint16_t>((size_t) T7 * r0.nq() * D);
            for (int i = 0; i < 5; ++i) {   // warm up: the first launch pays module load and clock ramp
                pr::split_q(qf_d, qf_whole, T7, nullptr);
                pr::gate_attn(at_d, qf_d, g_whole, T7, nullptr);
                pr::split_q_span(qf_d, qs_d, T7, r0.q0(), r0.nq(), NH * nq7, nullptr);
                pr::gate_attn_span(at_d, qf_d, gs_d, T7, r0.q0(), r0.nq(), NH * nq7, nullptr);
            }
            cudaEventRecord(e0);
            for (int i = 0; i < reps; ++i) { pr::split_q(qf_d, qf_whole, T7, nullptr);
                                             pr::gate_attn(at_d, qf_d, g_whole, T7, nullptr); }
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float whole_ms = 0; cudaEventElapsedTime(&whole_ms, e0, e1);
            cudaEventRecord(e0);
            for (int i = 0; i < reps; ++i) { pr::split_q_span(qf_d, qs_d, T7, r0.q0(), r0.nq(), NH * nq7, nullptr);
                                             pr::gate_attn_span(at_d, qf_d, gs_d, T7, r0.q0(), r0.nq(), NH * nq7, nullptr); }
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float span_ms = 0; cudaEventElapsedTime(&span_ms, e0, e1);
            std::printf("[split/gate span] microbench over %d reps: whole %.3f us a launch-pair, shard %.3f us "
                        "(%.2fx, %lld of %lld heads)\n", reps, whole_ms * 1000 / reps, span_ms * 1000 / reps,
                    whole_ms > 0 ? span_ms / whole_ms : 0.0, (long long) r0.nq(), (long long) NH);
            CHECK(span_ms <= whole_ms * 1.25f);   // half the elements must not cost more than the whole
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            cudaFree(qs_d); cudaFree(gs_d);
        }
        cudaFree(qf_d); cudaFree(at_d); cudaFree(qf_whole); cudaFree(g_whole);
    }

    // ---- 8. kv_append over one card's KV run: the cache writer, both formats
    //
    // The third hardcoded-head site found while writing the partner's arm, and the worst of the three:
    // kv_append_kernel took its KV-head count, its K/V row stride and BOTH pool row formulas from a literal 2.
    // A partner holding one KV head would have read one head past its own K/V and written each cell at a
    // two-heads-per-page stride into a pool carved one head per page - a corrupted cache that poisons every
    // later layer while still producing plausible tokens.  The reader (qsa_prompt_attn_batch) was already
    // shard-aware, which is why case 6 passed and this was invisible.
    //
    // The check is the whole run against each card's own pools, in BOTH the FP16 and the shipped INT8 layout,
    // because the two take different branches of the kernel and share only the row formulas - a formula error
    // shows in one and not the other is exactly what a single-format test would miss.  Bitwise: both sides
    // quantize the same input value the same way.
    {
        const int64_t PS = 4;                       // the artifact's page size, in cells
        const int64_t T8 = 12, pos8 = 8;            // not page-aligned: the tail block must land too
        const int64_t cells = pos8 + T8;
        const int64_t pages = (cells + PS - 1) / PS;
        std::vector<int32_t> table((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;   // identity, mode 0
        std::vector<float> K8((size_t) T8 * 2 * D), V8((size_t) T8 * 2 * D);
        for (auto& x : K8) x = nd(rng) * 4.0f;
        for (auto& x : V8) x = nd(rng) * 4.0f;
        float* K_d = dalloc<float>((size_t) T8 * 2 * D);
        float* V_d = dalloc<float>((size_t) T8 * 2 * D);
        int32_t* tab_d = dalloc<int32_t>((size_t) pages);
        up(K_d, K8); up(V_d, V8); up(tab_d, table);

        for (int fmt = 0; fmt < 2; ++fmt) {
            const bool int8_mode = fmt == 1;
            // the whole layer's pools: n_head_kv heads per page
            uint16_t* kp_f = dalloc<uint16_t>((size_t) pages * 2 * PS * D);
            uint16_t* vp_f = dalloc<uint16_t>((size_t) pages * 2 * PS * D);
            int8_t* kq_f = dalloc<int8_t>((size_t) pages * 2 * PS * D);
            int8_t* vq_f = dalloc<int8_t>((size_t) pages * 2 * PS * D);
            uint16_t* ks_f = dalloc<uint16_t>((size_t) pages * 2 * PS * (D / 64));
            uint16_t* vs_f = dalloc<uint16_t>((size_t) pages * 2 * PS * (D / 64));
            pr::kv_append(K_d, V_d, T8, pos8, tab_d, PS, int8_mode ? nullptr : kp_f,
                          int8_mode ? nullptr : vp_f, int8_mode ? kq_f : nullptr, int8_mode ? vq_f : nullptr,
                          int8_mode ? ks_f : nullptr, int8_mode ? vs_f : nullptr, nullptr, nullptr, nullptr);

            bool bitwise = true;
            for (const QsaHeadShard* r : {&r0, &r1}) {
                const int nkv = (int) r->nkv();
                // the card's OWN K and V: [T, nkv, head_dim] contiguous, its KV head's rows of the lead's
                std::vector<float> Ks((size_t) T8 * nkv * D), Vs((size_t) T8 * nkv * D);
                for (int64_t t = 0; t < T8; ++t)
                    for (int64_t h = 0; h < nkv; ++h)
                        for (int64_t d = 0; d < D; ++d) {
                            Ks[(size_t) t * nkv * D + h * D + d] = K8[(size_t) t * 2 * D + (r->kv0() + h) * D + d];
                            Vs[(size_t) t * nkv * D + h * D + d] = V8[(size_t) t * 2 * D + (r->kv0() + h) * D + d];
                        }
                float* Ks_d = dalloc<float>((size_t) T8 * nkv * D);
                float* Vs_d = dalloc<float>((size_t) T8 * nkv * D);
                up(Ks_d, Ks); up(Vs_d, Vs);
                // the card's pools, carved nkv heads per page - what qsa_prompt_state_carve gives it
                uint16_t* kp = dalloc<uint16_t>((size_t) pages * nkv * PS * D);
                uint16_t* vp = dalloc<uint16_t>((size_t) pages * nkv * PS * D);
                int8_t* kq = dalloc<int8_t>((size_t) pages * nkv * PS * D);
                int8_t* vq = dalloc<int8_t>((size_t) pages * nkv * PS * D);
                uint16_t* ks = dalloc<uint16_t>((size_t) pages * nkv * PS * (D / 64));
                uint16_t* vs = dalloc<uint16_t>((size_t) pages * nkv * PS * (D / 64));
                pr::kv_append_nkv(Ks_d, Vs_d, T8, pos8, tab_d, PS, int8_mode ? nullptr : kp,
                                  int8_mode ? nullptr : vp, int8_mode ? kq : nullptr, int8_mode ? vq : nullptr,
                                  int8_mode ? ks : nullptr, int8_mode ? vs : nullptr, nullptr, nullptr, nullptr,
                                  nkv);
                // compare the card's pool rows against its KV heads' rows of the whole run's pools
                const size_t nrow_f = (size_t) pages * 2 * PS, nrow_c = (size_t) pages * nkv * PS;
                auto cmp = [&](const int8_t* a, const int8_t* b, const char* what) {
                    int rep = 0;
                    for (int64_t pos = pos8; pos < cells && rep < 3; ++pos)
                        for (int64_t h = 0; h < nkv && rep < 3; ++h)
                            for (int64_t d = 0; d < D && rep < 3; ++d) {
                                const int64_t page = pos / PS;
                                const size_t cf = ((size_t) page * 2 + (r->kv0() + h)) * PS + pos % PS;
                                const size_t cc = ((size_t) page * nkv + h) * PS + pos % PS;
                                if (a[cc * D + d] == b[cf * D + d]) continue;
                                bitwise = false;
                                std::printf("FAIL kv_append %s rank %d pos %lld kv %lld dim %lld: %d vs %d\n", what,
                                        r->rank, (long long) pos, (long long) (r->kv0() + h), (long long) d,
                                        (int) a[cc * D + d], (int) b[cf * D + d]);
                                ++g_fail; ++rep;
                            }
                };
                auto cmp16 = [&](const uint16_t* a, const uint16_t* b, const char* what) {
                    int rep = 0;
                    for (int64_t pos = pos8; pos < cells && rep < 3; ++pos)
                        for (int64_t h = 0; h < nkv && rep < 3; ++h)
                            for (int64_t d = 0; d < D && rep < 3; ++d) {
                                const int64_t page = pos / PS;
                                const size_t cf = ((size_t) page * 2 + (r->kv0() + h)) * PS + pos % PS;
                                const size_t cc = ((size_t) page * nkv + h) * PS + pos % PS;
                                if (a[cc * D + d] == b[cf * D + d]) continue;
                                bitwise = false;
                                std::printf("FAIL kv_append %s rank %d pos %lld kv %lld dim %lld: %u vs %u\n", what,
                                        r->rank, (long long) pos, (long long) (r->kv0() + h), (long long) d,
                                        (unsigned) a[cc * D + d], (unsigned) b[cf * D + d]);
                                ++g_fail; ++rep;
                            }
                };
                if (int8_mode) {
                    std::vector<int8_t> a_k(nrow_c * D), b_k(nrow_f * D), a_v(nrow_c * D), b_v(nrow_f * D);
                    std::vector<uint16_t> a_ks(nrow_c * D / 64), b_ks(nrow_f * D / 64);
                    a_k = down(kq, a_k.size()); b_k = down(kq_f, b_k.size());
                    a_v = down(vq, a_v.size()); b_v = down(vq_f, b_v.size());
                    a_ks = down(ks, a_ks.size()); b_ks = down(ks_f, b_ks.size());
                    cmp(a_k.data(), b_k.data(), "int8 K");
                    cmp(a_v.data(), b_v.data(), "int8 V");
                    cmp16(a_ks.data(), b_ks.data(), "int8 K scale");
                } else {
                    std::vector<uint16_t> a_k(nrow_c * D), b_k(nrow_f * D), a_v(nrow_c * D), b_v(nrow_f * D);
                    a_k = down(kp, a_k.size()); b_k = down(kp_f, b_k.size());
                    a_v = down(vp, a_v.size()); b_v = down(vp_f, b_v.size());
                    cmp16(a_k.data(), b_k.data(), "fp16 K");
                    cmp16(a_v.data(), b_v.data(), "fp16 V");
                }
                cudaFree(Ks_d); cudaFree(Vs_d); cudaFree(kp); cudaFree(vp);
                cudaFree(kq); cudaFree(vq); cudaFree(ks); cudaFree(vs);
            }
            std::printf("[kv_append %s] T %lld from pos %lld, %d cells, pages %lld: %s\n",
                    int8_mode ? "int8" : "fp16", (long long) T8, (long long) pos8, (int) cells, (long long) pages,
                    bitwise ? "bitwise vs the whole run's pools" : "NOT bitwise");
            cudaFree(kp_f); cudaFree(vp_f); cudaFree(kq_f); cudaFree(vq_f); cudaFree(ks_f); cudaFree(vs_f);
        }
        cudaFree(K_d); cudaFree(V_d); cudaFree(tab_d);
    }

    std::printf("\ntp2 head split parity: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}

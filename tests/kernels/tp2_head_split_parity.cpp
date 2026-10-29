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
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;
using basalt::core::QsaHeadShard;

namespace {
int g_fail = 0;
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
    const QsaHeadShard r0{NH, NKV, 0, 2}, r1{NH, NKV, 1, 2};
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
    k::qsa_decode_attn_step(qf_d, pools_of(kf_d, vf_d, table_d), ids_d, step_d, cap, full, scr_f, attn_f, nullptr);
    k::qsa_decode_attn_step(q0_d, pools_of(k0_d, v0_d, table_d), ids_d, step_d, cap, s0, scr0, attn0, nullptr);
    k::qsa_decode_attn_step(q1_d, pools_of(k1_d, v1_d, table_d), ids_d, step_d, cap, s1, scr1, attn1, nullptr);
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
        k::qsa_decode_attn_batch(qw_d, pools_of(kf_d, vf_d, table_d), qids_d, steps_d, cap, full, sq_f, af, n_q, nullptr);
        k::qsa_decode_attn_batch(wh0_d, pools_of(k0_d, v0_d, table_d), qids_d, steps_d, cap, s0, sq_0, a0, n_q, nullptr);
        k::qsa_decode_attn_batch(wh1_d, pools_of(k1_d, v1_d, table_d), qids_d, steps_d, cap, s1, sq_1, a1, n_q, nullptr);
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
            k::qsa_decode_attn_step(qf_d, pools_of(kf_d, vf_d, table_d), ids_d, step_d, cap, full, scr_f, attn_f, nullptr);
        cudaEventRecord(b);
        cudaStreamSynchronize(nullptr);
        float whole_ms = 0.f;
        cudaEventElapsedTime(&whole_ms, a, b);
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) {
            k::qsa_decode_attn_step(q0_d, pools_of(k0_d, v0_d, table_d), ids_d, step_d, cap, s0, scr0, attn0, nullptr);
            k::qsa_decode_attn_step(q1_d, pools_of(k1_d, v1_d, table_d), ids_d, step_d, cap, s1, scr1, attn1, nullptr);
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

    std::printf("\ntp2 head split parity: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}

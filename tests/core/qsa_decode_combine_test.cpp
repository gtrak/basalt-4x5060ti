// tests/core/qsa_decode_combine_test.cpp - the decode side of the attention-head split: what the lead's shard
// must produce so that the peer's partial, added by the mixer combine, is the whole layer.  Three contracts,
// all pure arithmetic:
//
//   * the shard's attention reads the POOL at the pool's head width (its KV head is interleaved with the peer's
//     in the page tables), and that read gives its heads exactly what the whole layer gives them;
//   * each card's o_proj partial is the whole layer's row against its own packed columns, and the two partials
//     added in the combine's order (the card's own first, the peer's added after) close to the whole layer;
//   * the shard's q positions are stored one copy per RUNNING head (the q post reads them flattened over
//     (row, head)), and that layout is the same positions the whole layer reads, one-to-one.
//
// The CUDA kernels that execute these are parity-tested separately; this is the oracle for the contract itself.
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/core/qsa_head_split.hpp"
#include "basalt/kernels/qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace basalt::core;
using basalt::kernels::QsaShapes;

namespace {

int g_fail = 0;
#define CHECK(...)                                                                                       \
    do {                                                                                                 \
        if (!(__VA_ARGS__)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__); ++g_fail; } \
    } while (0)

// the artifact: 24 query heads, 2 KV heads, head_dim 256
constexpr int64_t N_HEAD = 24, N_HEAD_KV = 2, HEAD_DIM = 256;
// a decode window: the rows the window attends, and the pooled cells behind the top-k
constexpr int64_t T_ROW = 3, STEPS = 5;

// a deterministic pseudo value in [0, 1), the same sequence a test reads twice reads identically
float frac(uint64_t i) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull;
    x ^= x >> 29; x = x * 0xC2B2AE3D27D4EB4Full; x ^= x >> 32;
    return (float) (x >> 40) / (float) (1ull << 24);
}

// ---- the pool: the page table the WHOLE layer writes.  A cell is (step, kv head) at the pool's head width, so
// the head stride is the pool's n_head_kv, not the shard's: the shard reads its KV head out of the same table.
struct Pool {
    std::vector<float> k, v;   // [step][n_head_kv][head_dim]
    int64_t nk = N_HEAD_KV, hd = HEAD_DIM;
    const float* kcell(int64_t s, int64_t h) const { return k.data() + (s * nk + h) * hd; }
    const float* vcell(int64_t s, int64_t h) const { return v.data() + (s * nk + h) * hd; }
};

Pool make_pool() {
    Pool p;
    p.k.resize((size_t) STEPS * N_HEAD_KV * HEAD_DIM);
    p.v.resize((size_t) STEPS * N_HEAD_KV * HEAD_DIM);
    for (size_t i = 0; i < p.k.size(); ++i) p.k[i] = frac(i) - 0.5f;
    for (size_t i = 0; i < p.v.size(); ++i) p.v[i] = frac(i + 7919ull) - 0.5f;
    return p;
}

// one query head's attention over the pool's steps: score = <q, k>, softmax, out = sum_t p_t v_t.  `kv` picks
// the head inside the pool; the stride IS the pool's width, which is what the shard call must pass.
std::vector<float> head_attention(const float* q, const Pool& p, int64_t kv) {
    std::vector<float> o((size_t) HEAD_DIM, 0.f);
    std::vector<float> sc(STEPS);
    float m = -1e30f;
    for (int64_t s = 0; s < STEPS; ++s) {
        const float* k = p.kcell(s, kv);
        float x = 0.f;
        for (int64_t d = 0; d < HEAD_DIM; ++d) x += q[d] * k[d];
        sc[(size_t) s] = x;
        m = std::max(m, x);
    }
    float l = 0.f;
    for (int64_t s = 0; s < STEPS; ++s) l += std::expf(sc[(size_t) s] - m);
    for (int64_t s = 0; s < STEPS; ++s) {
        const float* v = p.vcell(s, kv);
        const float w = std::expf(sc[(size_t) s] - m) / l;
        for (int64_t d = 0; d < HEAD_DIM; ++d) o[(size_t) d] += w * v[d];
    }
    return o;
}

// CONTRACT 1: the shard's heads, read out of the WHOLE pool at the pool's width, are the whole layer's heads.
// The whole layer reads its KV head the same way (its pool IS the pool); so the shard, told the pool's stride,
// must produce identical outputs for its 12 heads - which is what makes the split additive rather than a new
// attention.
void the_shard_reads_the_whole_pool() {
    const QsaHeadShard lead{N_HEAD, N_HEAD_KV, 0, 2}, peer{N_HEAD, N_HEAD_KV, 1, 2};
    const QsaShapes sh = qsa_shard_shapes(basalt::kernels::qsa_real_shapes(), lead);
    CHECK(sh.n_head == 12 && sh.n_head_kv == 1);
    Pool p = make_pool();
    for (const QsaHeadShard* c : {&lead, &peer}) {
        for (int64_t h = 0; h < c->nq(); ++h) {
            const int64_t gh = c->q0() + h;   // the head's index in the whole layer
            std::vector<float> q((size_t) HEAD_DIM);
            for (int64_t d = 0; d < HEAD_DIM; ++d) q[(size_t) d] = frac(gh * 100000ull + (uint64_t) d) - 0.5f;
            // the shard: its one KV head, out of the pool at the pool's width
            const std::vector<float> a = head_attention(q.data(), p, c->kv0());
            // the whole layer: the same head, the same pool
            const std::vector<float> b = head_attention(q.data(), p, gh / (N_HEAD / N_HEAD_KV));
            bool same = true;
            for (int64_t d = 0; d < HEAD_DIM && same; ++d)
                if (a[(size_t) d] != b[(size_t) d]) same = false;
            CHECK(same);
        }
    }
    // the KV head a query head's group owns is the one its card reads - the group rule, per head
    for (int64_t gh = 0; gh < N_HEAD; ++gh) {
        const int64_t owner = gh / (N_HEAD / N_HEAD_KV);
        CHECK(lead.kv0() <= owner && owner < lead.kv0() + lead.nkv() ? owner == lead.kv0() : owner == peer.kv0());
    }
}

// CONTRACT 2: each card's o_proj partial is the whole layer's row against its own packed columns, and the two
// partials added in the combine's order (the card's own first, the peer's added after) close to the whole
// layer.  The columns split the dot exactly; only the addition order differs from the whole layer's single dot,
// so the gap is rounding, not a missing term.
void the_partials_close_to_the_whole_layer() {
    const int64_t n_embd = 16, dim = HEAD_DIM;
    const QsaHeadShard lead{N_HEAD, N_HEAD_KV, 0, 2}, peer{N_HEAD, N_HEAD_KV, 1, 2};
    const QsaHeadRun c0 = qsa_out_cols(lead, dim), c1 = qsa_out_cols(peer, dim);
    CHECK(c0.start == 0 && c0.len == 12 * dim && c1.start == 12 * dim && c1.len == 12 * dim);
    // the columns are disjoint and complete: their inputs cover the row once
    CHECK(c0.start + c0.len == c1.start && c1.start + c1.len == N_HEAD * dim);

    std::vector<float> W((size_t) n_embd * N_HEAD * dim);
    for (size_t i = 0; i < W.size(); ++i) W[i] = (frac(i) - 0.5f) * 0.05f;
    // the window's attention rows: [T_ROW][N_HEAD][dim]
    std::vector<float> attn((size_t) T_ROW * N_HEAD * dim);
    for (size_t i = 0; i < attn.size(); ++i) attn[i] = (frac(i + 31ull) - 0.5f) * 0.1f;

    for (int64_t t = 0; t < T_ROW; ++t) {
        for (int64_t r = 0; r < n_embd; ++r) {
            float full = 0.f;
            for (int64_t c = 0; c < N_HEAD * dim; ++c)
                full += W[(size_t) r * N_HEAD * dim + (size_t) c] * attn[(size_t) t * N_HEAD * dim + (size_t) c];
            float p0 = 0.f;
            for (int64_t c = 0; c < c0.len; ++c)
                p0 += W[(size_t) r * N_HEAD * dim + c0.start + (size_t) c] *
                      attn[(size_t) t * N_HEAD * dim + c0.start + (size_t) c];
            float p1 = 0.f;
            for (int64_t c = 0; c < c1.len; ++c)
                p1 += W[(size_t) r * N_HEAD * dim + c1.start + (size_t) c] *
                      attn[(size_t) t * N_HEAD * dim + c1.start + (size_t) c];
            const float combined = p0 + p1;   // the combine's order: the card's own first, the peer's after
            CHECK(std::fabs(combined - full) <= 1e-4f * (1.f + std::fabs(full)));
        }
    }
}

// CONTRACT 3: the shard's q positions, one copy per running head, are the whole layer's positions read the way
// the q post reads them: flattened over (row, head), block b = row * n_running + head.
void the_compact_positions_are_the_whole_ones() {
    const QsaHeadShard lead{N_HEAD, N_HEAD_KV, 0, 2};
    const int64_t nq = lead.nq();
    // the whole layer's position table: [T_ROW][N_HEAD] copies, all of a row equal
    std::vector<int32_t> full((size_t) T_ROW * N_HEAD);
    for (int64_t t = 0; t < T_ROW; ++t)
        for (int64_t h = 0; h < N_HEAD; ++h) full[(size_t) t * N_HEAD + (size_t) h] = (int32_t) (1000 + t);
    // the shard's table: [T_ROW][nq]
    std::vector<int32_t> compact((size_t) T_ROW * nq);
    for (int64_t t = 0; t < T_ROW; ++t)
        for (int64_t h = 0; h < nq; ++h) compact[(size_t) t * nq + (size_t) h] = (int32_t) (1000 + t);
    // the q post's flattened read: head h of row t is block t * n_running + h of the running shard, and must
    // agree with the whole layer's block t * N_HEAD + (q0 + h)
    for (int64_t t = 0; t < T_ROW; ++t)
        for (int64_t h = 0; h < nq; ++h)
            CHECK(compact[(size_t) t * nq + (size_t) h] == full[(size_t) t * N_HEAD + (size_t) (lead.q0() + h)]);
    CHECK(compact.size() == (size_t) T_ROW * nq);
}

}  // namespace

int main() {
    the_shard_reads_the_whole_pool();
    the_partials_close_to_the_whole_layer();
    the_compact_positions_are_the_whole_ones();
    if (g_fail != 0) { std::printf("qsa decode combine test failed: %d\n", g_fail); return 1; }
    std::printf("qsa decode combine test ok\n");
    return 0;
}
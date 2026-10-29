// tests/core/session_head_shard_test.cpp - the session carve with a head shard: a pair's card carves the same
// session with a narrower QSA part and nothing else changes.  Sizing only - no device, no allocation - so it runs
// anywhere; the carve itself (pointers over real memory, copies between sessions) is prompt_split_state_test, and
// it gains a shard case there.
//
// The identity this test is about:
//
//     session_bytes(whole) - session_bytes(shard)
//         == [qsa_state_bytes(whole) - qsa_state_bytes(shard)] * states + [qsa_buffers_bytes(whole) - qsa_buffers_bytes(shard)]
//
// which says the shard reached the QSA state and scratch and NOWHERE ELSE: the GDN state, the residual stack, the
// expert handoff and the PLE history are whole on both cards of the pair, because both run every GDN layer.
#include "basalt/core/session.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using namespace basalt::core;
using basalt::kernels::QsaShapes;

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

const ModelGeometry g;   // the artifact's geometry: 48 layers, 12 QSA, 24 heads, 2 KV heads, 256 dims
constexpr int64_t K = 10;
constexpr int64_t CELLS = 8192;

QsaHeadShard shard(int rank, int parts) { return {g.n_head, g.n_head_kv, rank, parts}; }

/// The states a range carves: `session_bytes`/`session_init` keep ONE primary QSA state even when the range has
/// no QSA layer, so the count here has to match theirs or the identity below is comparing different things.
/// The session rounds its total to 256 B and adds a page, so an identity between two session figures is exact
/// only up to that rounding.  Anything bigger than this is a real difference, not a pad.
constexpr int64_t ALIGN_SLACK = 256;

int64_t qsa_states(int64_t lo, int64_t hi) {
    const int64_t I = g.qsa_interval;
    const int64_t n = hi / I - lo / I;
    return n > 0 ? n : (g.n_qsa_layers() > 0 ? 1 : 0);
}

void the_default_is_the_whole_layer() {
    const QsaHeadShard none;   // what every session that is not part of a pair passes
    CHECK(none.parts == 1);
    CHECK(session_bytes(g, CELLS, K) == session_bytes(g, CELLS, K, 0, -1, none));
    CHECK(session_bytes(g, CELLS, K, 0, -1) == session_bytes(g, CELLS, K, 0, -1, shard(0, 1)));
    CHECK(qsa_state_bytes(g, CELLS, true) == qsa_state_bytes(g, CELLS, true, 0, shard(0, 1)));
    CHECK(qsa_buffers_bytes(g, CELLS) == qsa_buffers_bytes(g, CELLS, shard(0, 1)));
    CHECK(prompt_session_bytes(g, CELLS, 0, g.n_layers) ==
          prompt_session_bytes(g, CELLS, 0, g.n_layers, shard(0, 1)));
}

void a_pair_card_carves_less_and_the_rest_is_untouched() {
    const QsaHeadShard a = shard(0, 2), b = shard(1, 2);
    CHECK(a.ok() && b.ok());
    // symmetric: the two cards of a pair are the same size, or one of them is the bottleneck
    CHECK(session_bytes(g, CELLS, K, 0, -1, a) == session_bytes(g, CELLS, K, 0, -1, b));
    CHECK(qsa_state_bytes(g, CELLS, true, 0, a) == qsa_state_bytes(g, CELLS, true, 0, b));
    CHECK(qsa_buffers_bytes(g, CELLS, a) == qsa_buffers_bytes(g, CELLS, b));

    const int64_t whole = session_bytes(g, CELLS, K, 0, -1), half = session_bytes(g, CELLS, K, 0, -1, a);
    CHECK(half < whole);

    // the identity: what the shard saved is exactly the QSA state and scratch it saved
    const int64_t dstate = (qsa_state_bytes(g, CELLS, true, 0, shard(0, 1)) - qsa_state_bytes(g, CELLS, true, 0, a)) +
                           (qsa_states(0, g.n_layers) - 1) *
                               (qsa_state_bytes(g, CELLS, false, 0, shard(0, 1)) - qsa_state_bytes(g, CELLS, false, 0, a));
    const int64_t dbuf = qsa_buffers_bytes(g, CELLS, shard(0, 1)) - qsa_buffers_bytes(g, CELLS, a);
    CHECK(dstate > 0 && dbuf > 0);
    CHECK(std::abs((whole - half) - (dstate + dbuf)) <= ALIGN_SLACK);
    std::printf("  131072 cells: session %.3f GiB whole, %.3f GiB per card of a pair (QSA state %.3f -> %.3f, "
                "scratch %.1f -> %.1f MiB)\n",
            (double) session_bytes(g, 131072, K) / 1073741824.0,
            (double) session_bytes(g, 131072, K, 0, -1, a) / 1073741824.0,
            (double) qsa_state_bytes(g, 131072, true) / 1073741824.0,
            (double) qsa_state_bytes(g, 131072, true, 0, a) / 1073741824.0,
            (double) qsa_buffers_bytes(g, 131072) / 1048576.0,
            (double) qsa_buffers_bytes(g, 131072, a) / 1048576.0);

    // the same on a stage's layer range (a split stage carves [lo,hi) and the pair splits that too)
    for (int64_t lo = 0; lo < g.n_layers; lo += 24) {
        const int64_t hi = lo + 24;
        const int64_t w = session_bytes(g, CELLS, K, lo, hi), h = session_bytes(g, CELLS, K, lo, hi, a);
        CHECK(h < w);
        const int64_t ds =
            (qsa_state_bytes(g, CELLS, true, 0, shard(0, 1)) - qsa_state_bytes(g, CELLS, true, 0, a)) +
            (qsa_states(lo, hi) - 1) * (qsa_state_bytes(g, CELLS, false, 0, shard(0, 1)) -
                                         qsa_state_bytes(g, CELLS, false, 0, a));
        CHECK(std::abs((w - h) - (ds + dbuf)) <= ALIGN_SLACK);
        CHECK(session_bytes(g, CELLS, K, lo, hi, a) == session_bytes(g, CELLS, K, lo, hi, b));
    }
}

void the_kv_pool_is_half_the_cache() {
    // fp16 pools (the process default): cells x kv heads x head_dim x 2 B x (K,V), plus the shard's position rows
    const QsaHeadShard a = shard(0, 2);
    const int64_t pages = (CELLS + 511) / 512;   // qsa_real_shapes()'s page granule
    const int64_t kv_full = pages * 512 * g.n_head_kv * g.head_dim * 2 * 2;
    const int64_t kv_half = pages * 512 * a.nkv() * g.head_dim * 2 * 2;
    const int64_t pos_full = g.n_head * 4, pos_half = a.nq() * 4;
    const int64_t d = qsa_state_bytes(g, CELLS, true) - qsa_state_bytes(g, CELLS, true, 0, a);
    CHECK(d == (kv_full - kv_half) + (pos_full - pos_half));
    CHECK(kv_half * 2 == kv_full);   // the two cards together are the cache the whole layer has
}

void the_scratch_keeps_the_indexer_and_the_selection() {
    // what must NOT shrink with the shard: the indexer's own buffers and the selection's, because both cards
    // compute the same top-k (qsa_head_split.hpp's reason the split is bitwise-exact)
    const QsaHeadShard a = shard(0, 2);
    const QsaShapes w = qsa_real_shapes(g), h = qsa_shard_shapes(g, a);
    CHECK(h.idx_n_head == w.idx_n_head && h.idx_dim == w.idx_dim && h.idx_top_k == w.idx_top_k);
    CHECK(h.n_head == g.n_head / 2 && h.n_head_kv == g.n_head_kv / 2);
    // q_idx (idx_q_heads x idx_key_dim), idx_raw, cell_scores and ids are geometry-wide, so the shard saves
    // something but never all of the scratch
    const int64_t d = qsa_buffers_bytes(g, CELLS, shard(0, 1)) - qsa_buffers_bytes(g, CELLS, a);
    CHECK(d > 0);
    CHECK(d < qsa_buffers_bytes(g, CELLS, shard(0, 1)));   // never all of it: something stayed whole
}

void a_copy_between_shards_is_refused() {
    QsaState from, to;
    from.heads = shard(0, 2);
    to.heads = shard(1, 2);
    from.max_cells = to.max_cells = CELLS;
    std::string err;
    int copied = 0;
    const bool ok = qsa_state_copy_cells(to, from, g, 0, 16,
                                         [&](void*, const void*, uint64_t) { ++copied; return true; }, err);
    CHECK(!ok);
    CHECK(copied == 0);   // refused before queueing anything
    CHECK(err.find("shard") != std::string::npos);
    // same shard: the copy goes ahead (nothing is allocated, so it queues zero-byte copies of null pointers, which
    // the fixture's own DeviceCopy accepts - what is being checked is that it is not refused)
    to.heads = from.heads;
    std::string err2;
    CHECK(qsa_state_copy_cells(to, from, g, 0, 0, [&](void*, const void*, uint64_t) { return true; }, err2));
}

void the_prompt_session_shards_too() {
    const QsaHeadShard a = shard(0, 2);
    const int64_t w = prompt_session_bytes(g, CELLS, 0, g.n_layers);
    const int64_t h = prompt_session_bytes(g, CELLS, 0, g.n_layers, a);
    CHECK(h < w);
    CHECK(prompt_session_bytes(g, CELLS, 0, g.n_layers, shard(0, 2)) ==
          prompt_session_bytes(g, CELLS, 0, g.n_layers, shard(1, 2)));
    // a stage's range, the way a split stage's prompt path carves it
    CHECK(prompt_session_bytes(g, CELLS, 24, 48, a) < prompt_session_bytes(g, CELLS, 24, 48));
}

void the_shapes_come_from_one_place() {
    const QsaShapes w = qsa_real_shapes(g);
    CHECK(w.n_head == g.n_head && w.n_head_kv == g.n_head_kv && w.head_dim == g.head_dim);
    CHECK(w.idx_n_head == g.idx_q_heads && w.idx_dim == g.idx_key_dim);
    CHECK(w.n_head % w.n_head_kv == 0);
    const QsaShapes h = qsa_shard_shapes(g, shard(1, 2));
    CHECK(h.head_dim == w.head_dim && h.page_size == w.page_size && h.n_rot == w.n_rot);
    CHECK(h.idx_block == w.idx_block);
}

}  // namespace

int main() {
    the_default_is_the_whole_layer();
    a_pair_card_carves_less_and_the_rest_is_untouched();
    the_kv_pool_is_half_the_cache();
    the_scratch_keeps_the_indexer_and_the_selection();
    a_copy_between_shards_is_refused();
    the_prompt_session_shards_too();
    the_shapes_come_from_one_place();
    if (g_fail != 0) { std::printf("session head shard test failed: %d\n", g_fail); return 1; }
    std::printf("session head shard test ok\n");
    return 0;
}

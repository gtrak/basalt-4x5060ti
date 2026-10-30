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

// An UNEVEN cut on the artifact's own geometry can only be made by STRADDLING a KV group (12 heads is the unit the
// kernels move), and a straddling pair holds the boundary group twice: 16 heads on the lead, 8 on the partner, and
// 3 KV heads of pool between them for a 2-head cache.  The run gate refuses such a pair (`who_may_run_a_shard`),
// so this checks the CARVE, which is what a pair that generalizes the kernels would be sized for.
void an_uneven_cut_carves_by_its_own_head_count() {
    const QsaHeadShard a = qsa_pair_shard(g.n_head, g.n_head_kv, 16, 0);
    const QsaHeadShard b = qsa_pair_shard(g.n_head, g.n_head_kv, 16, 1);
    CHECK(a.ok() && b.ok());
    CHECK(a.nq() == 16 && a.nkv() == 2 && b.nq() == 8 && b.nkv() == 1);
    CHECK(qsa_pair_tiles(a, b));
    CHECK(!qsa_shard_whole_groups(a) && !qsa_shard_whole_groups(b));
    // the boundary KV head is held twice over
    CHECK(qsa_shard_kv_cell_bytes(a, g.head_dim) + qsa_shard_kv_cell_bytes(b, g.head_dim) ==
          3ull * g.head_dim * 2 * 2);
    // the two cards are no longer the same size, and each follows its own head count
    const int64_t whole = session_bytes(g, CELLS, K, 0, -1);
    const int64_t sa = session_bytes(g, CELLS, K, 0, -1, a), sb = session_bytes(g, CELLS, K, 0, -1, b);
    CHECK(sb < sa && sa < whole);
    // the identity, per card: what the shard saved is exactly the QSA state and scratch it saved
    for (const QsaHeadShard* c : {&a, &b}) {
        const int64_t s = c == &a ? sa : sb;
        const int64_t dstate =
            (qsa_state_bytes(g, CELLS, true, 0, shard(0, 1)) - qsa_state_bytes(g, CELLS, true, 0, *c)) +
            (qsa_states(0, g.n_layers) - 1) *
                (qsa_state_bytes(g, CELLS, false, 0, shard(0, 1)) - qsa_state_bytes(g, CELLS, false, 0, *c));
        const int64_t dbuf = qsa_buffers_bytes(g, CELLS, shard(0, 1)) - qsa_buffers_bytes(g, CELLS, *c);
        CHECK(std::abs((whole - s) - (dstate + dbuf)) <= ALIGN_SLACK);
    }
    // the pools follow the KV run, the position rows the query run
    const int64_t cells = ((CELLS + 511) / 512) * 512;   // qsa_real_shapes()'s page granule
    const int64_t pos_a = a.nq() * 4, pos_b = b.nq() * 4;
    CHECK(pos_a == 2 * pos_b);
    CHECK((int64_t) (qsa_state_bytes(g, CELLS, true, 0, a) - qsa_state_bytes(g, CELLS, true, 0, b)) ==
          cells * (int64_t) (qsa_shard_kv_cell_bytes(a, g.head_dim) - qsa_shard_kv_cell_bytes(b, g.head_dim)) +
              (pos_a - pos_b));
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

// Who is allowed to RUN a shard.  Two questions, and the split between them is the safety property:
// `qsa_shard_runs` asks whether the shard is a well-formed half of a pair (geometry only), and
// `qsa_shard_ready` asks whether THIS session may run it - which needs the stage pair attached to the
// buffers, the same rank on both sides, an open Tp2Pair and a packed attn_output for the layer.  A session
// that was never carved for the pair under tp2 has no pair pointer, so it cannot run a shard by accident.
void who_may_run_a_shard() {
    std::string err;
    // ---- the geometry half
    CHECK(qsa_shard_runs(QsaHeadShard{}, err));           // the unsplit layer, which is what everyone asks for
    CHECK(err.empty());
    CHECK(qsa_shard_runs(QsaHeadShard{g.n_head, g.n_head_kv, 0, 1}, err));
    CHECK(err.empty());
    CHECK(qsa_shard_runs(shard(0, 2), err));              // a well-formed half of a pair is runnable IN PRINCIPLE
    CHECK(err.empty());
    CHECK(qsa_shard_runs(shard(1, 2), err));
    err.clear();
    CHECK(!qsa_shard_runs(QsaHeadShard{g.n_head, g.n_head_kv, 2, 4}, err));   // rank outside the pair
    CHECK(!err.empty());
    err.clear();
    CHECK(!qsa_shard_runs(QsaHeadShard{g.n_head, 3, 0, 3}, err));  // three well-formed pieces: the collective is a pair's
    CHECK(err.find("two") != std::string::npos);
    err.clear();
    CHECK(!qsa_shard_runs(QsaHeadShard{g.n_head, 3, 0, 2}, err));             // 3 KV heads do not cut in two
    CHECK(err.find("not a well-formed split") != std::string::npos);

    // ---- an UNEVEN cut: runnable when each card's run is whole KV groups, refused when it cuts one
    err.clear();
    CHECK(qsa_shard_runs(qsa_pair_shard(24, 8, 9, 0), err));   // 9 heads = 3 groups, the lead's end of the axis
    CHECK(err.empty());
    CHECK(qsa_shard_runs(qsa_pair_shard(24, 8, 9, 1), err));   // 15 heads = 5 groups
    err.clear();
    CHECK(!qsa_shard_runs(qsa_pair_shard(g.n_head, g.n_head_kv, 16, 0), err));   // 16 of 24 cuts a group of 12
    CHECK(err.find("KV group") != std::string::npos);
    err.clear();
    CHECK(!qsa_shard_runs(qsa_pair_shard(24, 8, 10, 1), err));   // 14 heads is not whole groups either
    CHECK(err.find("KV group") != std::string::npos);
    err.clear();
    CHECK(!qsa_shard_runs(QsaHeadShard{24, 8, 0, 2, 6, 6}, err));   // the lead's run starts at head 0
    CHECK(err.find("query axis") != std::string::npos);
    err.clear();
    CHECK(!qsa_shard_runs(qsa_pair_shard(24, 2, 0, 0), err));       // a card with no heads is not a pair
    CHECK(!err.empty());

    // ---- the "may it run HERE" half: no pair, no shard
    QsaState st{};
    QsaBuffers bufs{};
    st.heads = shard(0, 2);
    bufs.heads = shard(0, 2);
    err.clear();
    CHECK(!qsa_shard_ready(st, bufs, 0, err));            // sized but no pair: the tp2 carve is what's missing
    CHECK(err.find("no stage pair") != std::string::npos);
    CHECK(qsa_shard_ready(QsaState{}, QsaBuffers{}, 0, err));   // a whole layer needs nothing

    // a pair whose collective was never opened
    st.heads = bufs.heads = shard(1, 2);
    QsaPair link{};
    link.rank = 1;
    link.peer_dev = 2;
    link.device = 3;
    QsaShardWeight view{};
    view.ne0 = 12 * 256;
    view.ne1 = 4096;
    view.ok = true;
    link.o_cols.assign((size_t) g.n_layers, QsaShardWeight{});
    link.o_cols[7] = view;
    bufs.pair = &link;
    err.clear();
    CHECK(!qsa_shard_ready(st, bufs, 7, err));            // valid() is false on an unopened pair
    CHECK(err.find("not open") != std::string::npos);

    // mismatched shards between the state and its buffers
    QsaState other{};
    other.heads = shard(0, 2);
    err.clear();
    CHECK(!qsa_shard_ready(other, bufs, 7, err));
    CHECK(err.find("its buffers for rank") != std::string::npos);

    // a well-formed shard, a matching rank, but the layer was never prepared: only reachable once the pair is
    // open, so it is checked through the geometry-only half here; `qsa_shard_ready`'s own packed-view branch is
    // exercised on a real pair by tp2_head_split_parity.
    CHECK(!link.o_cols[0].ok);
}

}  // namespace

int main() {
    the_default_is_the_whole_layer();
    a_pair_card_carves_less_and_the_rest_is_untouched();
    the_kv_pool_is_half_the_cache();
    the_scratch_keeps_the_indexer_and_the_selection();
    a_copy_between_shards_is_refused();
    the_prompt_session_shards_too();
    an_uneven_cut_carves_by_its_own_head_count();
    the_shapes_come_from_one_place();
    who_may_run_a_shard();
    if (g_fail != 0) { std::printf("session head shard test failed: %d\n", g_fail); return 1; }
    std::printf("session head shard test ok\n");
    return 0;
}

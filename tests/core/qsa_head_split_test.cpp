// tests/core/qsa_head_split_test.cpp - the pair's attention-head split: every head on exactly one card, the KV
// groups kept whole, the projection runs on their block boundaries, and the indexer left whole.  Pure arithmetic,
// so this is the whole oracle for the shard (the kernels that consume it are parity-tested separately).
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/core/qsa_head_split.hpp"

#include <algorithm>
#include <cstdio>
#include <set>
#include <vector>

using namespace basalt::core;
using basalt::kernels::QsaShapes;

namespace {

int g_fail = 0;
// variadic so a check may contain a braced initializer, whose commas the preprocessor would otherwise split
#define CHECK(...)                                                                                       \
    do {                                                                                                 \
        if (!(__VA_ARGS__)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__); ++g_fail; } \
    } while (0)

// the artifact: 24 query heads, 2 KV heads, head_dim 256, the indexer's own 4 heads of 128
constexpr int64_t N_HEAD = 24, N_HEAD_KV = 2, HEAD_DIM = 256;

QsaShapes real_shapes() {
    QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = N_HEAD;
    s.n_head_kv = N_HEAD_KV;
    s.head_dim = HEAD_DIM;
    return s;
}

void the_real_pair() {
    const QsaHeadShard a{N_HEAD, N_HEAD_KV, 0, 2}, b{N_HEAD, N_HEAD_KV, 1, 2};
    CHECK(a.ok() && b.ok());
    CHECK(a.rep() == 12);
    CHECK(a.kv0() == 0 && a.nkv() == 1 && a.q0() == 0 && a.nq() == 12);
    CHECK(b.kv0() == 1 && b.nkv() == 1 && b.q0() == 12 && b.nq() == 12);
    // a whole KV group, so neither card reads the other's KV
    CHECK(a.q0() / a.rep() == a.kv0());
    CHECK(b.q0() / b.rep() == b.kv0());
    // half the KV bytes per card, and the two cards together are the whole cache
    CHECK(qsa_shard_kv_cell_bytes(a, HEAD_DIM) * 2 == (uint64_t) N_HEAD_KV * HEAD_DIM * 2 * 2);
    CHECK(qsa_shard_kv_cell_bytes(a, HEAD_DIM) == (uint64_t) 1 * HEAD_DIM * 2 * 2);

    // every head of the model is on exactly one card of the pair
    std::set<int64_t> q, kv;
    for (int64_t h = a.q0(); h < a.q0() + a.nq(); ++h) q.insert(h);
    for (int64_t h = b.q0(); h < b.q0() + b.nq(); ++h) CHECK(q.insert(h).second);
    CHECK(q.size() == (size_t) N_HEAD);
    for (int64_t h = a.kv0(); h < a.kv0() + a.nkv(); ++h) kv.insert(h);
    for (int64_t h = b.kv0(); h < b.kv0() + b.nkv(); ++h) CHECK(kv.insert(h).second);
    CHECK(kv.size() == (size_t) N_HEAD_KV);

    // the group rule holds for EVERY query head of the shard, not just the first
    for (const QsaHeadShard* c : {&a, &b})
        for (int64_t h = c->q0(); h < c->q0() + c->nq(); ++h)
            CHECK(h / c->rep() >= c->kv0() && h / c->rep() < c->kv0() + c->nkv());
}

void unsplit_is_everything() {
    const QsaHeadShard one{N_HEAD, N_HEAD_KV, 0, 1};
    CHECK(one.whole() && one.ok());
    CHECK(one.q0() == 0 && one.nq() == N_HEAD && one.kv0() == 0 && one.nkv() == N_HEAD_KV);
    // the shapes the engine runs today are what a whole shard produces: no behaviour change for a single card
    const QsaShapes s = real_shapes();
    const QsaShapes w = qsa_shard_shapes(s, one);
    CHECK(w.n_head == s.n_head && w.n_head_kv == s.n_head_kv && w.idx_n_head == s.idx_n_head &&
          w.idx_dim == s.idx_dim && w.idx_top_k == s.idx_top_k && w.page_size == s.page_size &&
          w.head_dim == s.head_dim && w.n_rot == s.n_rot && w.idx_block == s.idx_block);
}

void the_indexer_stays_whole() {
    // the reason the split is bitwise-exact: two of four indexer heads would pick a different top-k
    const QsaShapes s = real_shapes();
    for (int r = 0; r < 2; ++r) {
        const QsaHeadShard h{N_HEAD, N_HEAD_KV, r, 2};
        const QsaShapes x = qsa_shard_shapes(s, h);
        CHECK(x.idx_n_head == s.idx_n_head);
        CHECK(x.idx_dim == s.idx_dim);
        CHECK(x.idx_block == s.idx_block);
        CHECK(x.idx_top_k == s.idx_top_k);
        // head_dim and n_rot are per-head, not per-card, so nothing should narrow them - but the partner's arm
        // hands both straight to rope and rms_rows, where a narrowed value is a wrong rotation, not a crash
        CHECK(x.head_dim == s.head_dim && x.n_rot == s.n_rot);
        CHECK(x.n_head == 12 && x.n_head_kv == 1);
    }
    // an unusable shard leaves the shapes alone rather than narrowing them
    const QsaHeadShard bad{N_HEAD, N_HEAD_KV, 5, 4};
    const QsaShapes kept = qsa_shard_shapes(s, bad);
    CHECK(kept.n_head == s.n_head && kept.n_head_kv == s.n_head_kv);
}

void what_is_refused() {
    CHECK(!QsaHeadShard{N_HEAD, N_HEAD_KV, 0, 3}.ok());     // 2 KV heads do not cut three ways
    CHECK(!QsaHeadShard{N_HEAD, N_HEAD_KV, 2, 2}.ok());     // no third card in a pair
    CHECK(!QsaHeadShard{N_HEAD, N_HEAD_KV, -1, 2}.ok());
    CHECK(!QsaHeadShard{N_HEAD, N_HEAD_KV, 0, 0}.ok());
    CHECK(!QsaHeadShard{0, N_HEAD_KV, 0, 1}.ok());
    CHECK(!QsaHeadShard{N_HEAD, 0, 0, 1}.ok());
    CHECK(!QsaHeadShard{25, N_HEAD_KV, 0, 1}.ok());         // query heads not a multiple of the KV heads
    CHECK(!qsa_shard_fits(N_HEAD, N_HEAD_KV, 3));
    CHECK(qsa_shard_fits(N_HEAD, N_HEAD_KV, 2));
    CHECK(qsa_shard_fits(N_HEAD, N_HEAD_KV, 1));
    // a geometry that does cut: 8 KV heads over four cards
    CHECK(qsa_shard_fits(24, 8, 4));
    const QsaHeadShard f{24, 8, 2, 4};
    CHECK(f.nkv() == 2 && f.kv0() == 4 && f.rep() == 3 && f.q0() == 12 && f.nq() == 6);
    // an odd indexer head count is never a reason to refuse: the indexer does not split
    CHECK(qsa_shard_fits(N_HEAD, N_HEAD_KV, 2));
}

void the_projection_runs() {
    const QsaHeadShard a{N_HEAD, N_HEAD_KV, 0, 2}, b{N_HEAD, N_HEAD_KV, 1, 2};
    // attn_q: two head_dim rows per query head, query then gate
    CHECK(qsa_q_rows(a, HEAD_DIM).start == 0 && qsa_q_rows(a, HEAD_DIM).len == 12 * 2 * HEAD_DIM);
    CHECK(qsa_q_rows(b, HEAD_DIM).start == 12 * 2 * HEAD_DIM && qsa_q_rows(b, HEAD_DIM).len == 12 * 2 * HEAD_DIM);
    CHECK(qsa_q_rows(a, HEAD_DIM).start + qsa_q_rows(a, HEAD_DIM).len == qsa_q_rows(b, HEAD_DIM).start);
    // attn_k / attn_v: one head_dim row per KV head
    CHECK(qsa_kv_rows(a, HEAD_DIM).start == 0 && qsa_kv_rows(a, HEAD_DIM).len == HEAD_DIM);
    CHECK(qsa_kv_rows(b, HEAD_DIM).start == HEAD_DIM && qsa_kv_rows(b, HEAD_DIM).len == HEAD_DIM);
    // attn_output columns: the card's query heads' columns, which is the partial the all-reduce sums
    CHECK(qsa_out_cols(a, HEAD_DIM).start == 0 && qsa_out_cols(a, HEAD_DIM).len == 12 * HEAD_DIM);
    CHECK(qsa_out_cols(b, HEAD_DIM).start == 12 * HEAD_DIM && qsa_out_cols(b, HEAD_DIM).len == 12 * HEAD_DIM);
    CHECK(qsa_out_cols(a, HEAD_DIM).len + qsa_out_cols(b, HEAD_DIM).len == N_HEAD * HEAD_DIM);

    // blocked weights slice columns on whole blocks only
    for (int64_t blk : {32ll, 64ll, 128ll}) {
        CHECK(qsa_shard_cols_aligned(a, HEAD_DIM, blk));
        CHECK(qsa_shard_cols_aligned(b, HEAD_DIM, blk));
    }
    CHECK(!qsa_shard_cols_aligned(a, HEAD_DIM, 0));
    CHECK(!qsa_shard_cols_aligned(a, 100, 128));  // 12 heads of 100 columns is not a whole number of blocks
    CHECK(qsa_shard_rows_aligned(a, HEAD_DIM, 1));
    CHECK(!qsa_shard_rows_aligned(a, HEAD_DIM, 0));

    // the runs of every card tile the query axis exactly once
    CHECK(qsa_head_runs(QsaHeadShard{N_HEAD, N_HEAD_KV, 0, 4}).empty());   // 2 KV heads do not cut four ways
    const std::vector<QsaHeadRun> two = qsa_head_runs(QsaHeadShard{N_HEAD, N_HEAD_KV, 0, 2});
    CHECK(two.size() == 2);
    int64_t covered = 0;
    for (const QsaHeadRun& r : two) covered += r.len;
    CHECK(covered == N_HEAD);
    CHECK(qsa_head_runs(QsaHeadShard{N_HEAD, 8, 0, 4}).size() == 4);
}

// The packer the engine will use for a shard's weights: gdn_head_split.hpp's row and column gathers, which are
// generic over a row-major matrix.  A card's weight must be the SAME BYTES the whole layer would have read for
// that head range - otherwise the shard is a different model, not the same model's half.
// An UNEVEN cut: the two cards run different numbers of query heads.  The same two things must hold - the runs
// tile the query axis exactly once, and every running head's KV head is on its own card - and the second one is
// what makes a cut INSIDE a KV group cost both cards that group's KV.  24 heads over 8 KV heads is the geometry
// that cuts unevenly into whole groups (3 groups to the lead, 5 to the partner), which is also what the attention
// kernels can run; 24 over 2 only moves 12 heads at a time, so its uneven cuts straddle.
void an_uneven_cut_is_still_one_model() {
    const QsaHeadShard a = qsa_pair_shard(24, 8, 9, 0), b = qsa_pair_shard(24, 8, 9, 1);
    CHECK(a.ok() && b.ok());
    CHECK(a.rep() == 3);
    CHECK(a.q0() == 0 && a.nq() == 9 && a.kv0() == 0 && a.nkv() == 3);
    CHECK(b.q0() == 9 && b.nq() == 15 && b.kv0() == 3 && b.nkv() == 5);
    CHECK(qsa_shard_whole_groups(a) && qsa_shard_whole_groups(b));   // whole groups: the kernels run this pair
    CHECK(qsa_pair_tiles(a, b));
    // the runs tile the axis once, from either shard's point of view
    const std::vector<QsaHeadRun> runs = qsa_head_runs(a);
    CHECK(runs.size() == 2 && runs[0].start == 0 && runs[0].len == 9 && runs[1].start == 9 && runs[1].len == 15);
    CHECK(qsa_head_runs(b).size() == 2 && qsa_head_runs(b)[0].start == runs[0].start &&
          qsa_head_runs(b)[0].len == runs[0].len && qsa_head_runs(b)[1].start == runs[1].start &&
          qsa_head_runs(b)[1].len == runs[1].len);
    // the group rule, head by head, with no KV head held twice: whole groups do not overlap
    std::set<int64_t> q, kv;
    for (const QsaHeadShard* c : {&a, &b}) {
        CHECK(c->q0() % c->rep() == 0 && c->nq() % c->rep() == 0);
        for (int64_t h = c->q0(); h < c->q0() + c->nq(); ++h) {
            CHECK(q.insert(h).second);                                  // one card runs each query head
            CHECK(h / c->rep() >= c->kv0() && h / c->rep() < c->kv0() + c->nkv());
        }
        for (int64_t h = c->kv0(); h < c->kv0() + c->nkv(); ++h) CHECK(kv.insert(h).second);
    }
    CHECK(q.size() == 24 && kv.size() == 8);
    // the KV bytes split with the heads, and the two cards together are exactly the whole cache
    CHECK(qsa_shard_kv_cell_bytes(a, HEAD_DIM) == (uint64_t) 3 * HEAD_DIM * 2 * 2);
    CHECK(qsa_shard_kv_cell_bytes(a, HEAD_DIM) + qsa_shard_kv_cell_bytes(b, HEAD_DIM) ==
          (uint64_t) 8 * HEAD_DIM * 2 * 2);
    // and the projection runs follow the uneven head counts
    CHECK(qsa_q_rows(a, HEAD_DIM).len == 9 * 2 * HEAD_DIM && qsa_q_rows(b, HEAD_DIM).len == 15 * 2 * HEAD_DIM);
    CHECK(qsa_out_cols(a, HEAD_DIM).start == 0 && qsa_out_cols(a, HEAD_DIM).len == 9 * HEAD_DIM);
    CHECK(qsa_out_cols(b, HEAD_DIM).start == 9 * HEAD_DIM && qsa_out_cols(b, HEAD_DIM).len == 15 * HEAD_DIM);
    CHECK(qsa_out_cols(a, HEAD_DIM).len + qsa_out_cols(b, HEAD_DIM).len == 24 * HEAD_DIM);
    // an uneven cut is still on the block boundaries a quantized o_proj needs
    for (int64_t blk : {32ll, 64ll, 128ll}) {
        CHECK(qsa_shard_cols_aligned(a, HEAD_DIM, blk));
        CHECK(qsa_shard_cols_aligned(b, HEAD_DIM, blk));
    }
    // the narrow geometry: half the heads is still the pair everyone builds
    CHECK(qsa_pair_shard(N_HEAD, N_HEAD_KV, 12, 0) == QsaHeadShard{N_HEAD, N_HEAD_KV, 0, 2});
    CHECK(qsa_pair_shard(N_HEAD, N_HEAD_KV, 12, 1) == QsaHeadShard{N_HEAD, N_HEAD_KV, 1, 2});
}

// A cut inside a KV group: the arithmetic is still self-consistent - the runs tile the axis, every running head
// still finds its KV head on its own card - but the boundary group is now on BOTH cards, and the attention
// kernels, which hold one group's heads together, cannot run such a shard.
void a_cut_inside_a_group_costs_both_cards_that_group() {
    const QsaHeadShard a = qsa_pair_shard(24, 2, 16, 0), b = qsa_pair_shard(24, 2, 16, 1);
    CHECK(a.ok() && b.ok());
    CHECK(a.q0() == 0 && a.nq() == 16 && a.kv0() == 0 && a.nkv() == 2);   // heads 12-15 read KV head 1
    CHECK(b.q0() == 16 && b.nq() == 8 && b.kv0() == 1 && b.nkv() == 1);   // ... and so do heads 16-23
    CHECK(qsa_pair_tiles(a, b));
    CHECK(!qsa_shard_whole_groups(a) && !qsa_shard_whole_groups(b));      // the kernels refuse this pair
    // the boundary group is held twice: the two cards' KV bytes are MORE than the whole cache, by one head
    CHECK(qsa_shard_kv_cell_bytes(a, HEAD_DIM) + qsa_shard_kv_cell_bytes(b, HEAD_DIM) ==
          (uint64_t) 3 * HEAD_DIM * 2 * 2);
    // every query head still reads a KV head its own card holds
    for (const QsaHeadShard* c : {&a, &b})
        for (int64_t h = c->q0(); h < c->q0() + c->nq(); ++h)
            CHECK(h / c->rep() >= c->kv0() && h / c->rep() < c->kv0() + c->nkv());

    // the ratio form, and what a ratio outside the axis means
    CHECK(qsa_lead_heads(24, 0.5) == 12);
    CHECK(qsa_lead_heads(24, 0.6) == 14);
    CHECK(qsa_lead_heads(24, 0.02) == 1);
    CHECK(qsa_lead_heads(24, 0.99) == 23);
    CHECK(qsa_lead_heads(24, 0.0) == 1);
    CHECK(qsa_lead_heads(24, 1.5) == 23);
    CHECK(qsa_lead_heads(0, 0.5) == 0);
    CHECK(!qsa_pair_shard(24, 2, 0, 0).ok());     // a card with no heads is not a pair
    CHECK(!qsa_pair_shard(24, 2, 24, 1).ok());

    // what the tiling check refuses: a gap, an overlap, two leads, a mismatched geometry
    const QsaHeadShard lead = qsa_pair_shard(24, 8, 9, 0);
    CHECK(!qsa_pair_tiles(lead, qsa_pair_shard(24, 8, 9, 0)));   // the same card twice
    CHECK(!qsa_pair_tiles(lead, QsaHeadShard{24, 8, 1, 2}));     // the partner's symmetric run is heads 12-23
    CHECK(!qsa_pair_tiles(lead, qsa_pair_shard(24, 6, 9, 1)));   // different geometries
    CHECK(qsa_pair_tiles(lead, qsa_pair_shard(24, 8, 9, 1)));
}

void the_weight_slices_are_the_whole_layers_bytes() {
    const int64_t dim = HEAD_DIM, n_embd = 8;   // 8 rows of output; columns are the head axis
    const QsaHeadShard r0{N_HEAD, N_HEAD_KV, 0, 2}, r1{N_HEAD, N_HEAD_KV, 1, 2};
    // ---- o_proj: [n_embd rows][n_head * head_dim columns], blocked along columns
    for (const int64_t block : {32LL, 64LL, 128LL, 256LL}) {
        if (!qsa_shard_cols_aligned(r0, dim, block)) { std::printf("skip block %lld\n", (long long) block); continue; }
        std::vector<float> full((size_t) n_embd * N_HEAD * dim);
        for (size_t i = 0; i < full.size(); ++i) full[i] = (float) i * 0.25f;
        const size_t block_bytes = (size_t) block * sizeof(float);
        std::vector<float> part0((size_t) n_embd * r0.nq() * dim), part1((size_t) n_embd * r1.nq() * dim);
        const QsaHeadRun c0 = qsa_out_cols(r0, dim), c1 = qsa_out_cols(r1, dim);
        gdn_gather_cols(full.data(), n_embd, N_HEAD * dim, block, block_bytes, {{c0.start, c0.len}}, part0.data());
        gdn_gather_cols(full.data(), n_embd, N_HEAD * dim, block, block_bytes, {{c1.start, c1.len}}, part1.data());
        // each card's packed weight is the whole layer's row, its own columns, in order - so the two cards
        // side by side are the whole weight, and a card's matvec reads the same numbers the whole layer would
        const int64_t half = r0.nq() * dim;
        CHECK(2 * half == N_HEAD * dim);
        CHECK(part0.size() == (size_t) n_embd * half && part1.size() == (size_t) n_embd * half);
        bool same = true;
        for (int64_t row = 0; row < n_embd && same; ++row)
            for (int64_t c = 0; c < half; ++c) {
                const size_t w = (size_t) row * N_HEAD * dim;
                if (part0[(size_t) row * half + c] != full[w + (size_t) r0.q0() * dim + c] ||
                    part1[(size_t) row * half + c] != full[w + (size_t) r1.q0() * dim + c]) {
                    same = false;
                    break;
                }
            }
        CHECK(same);
    }
    // an odd head_dim cannot be cut on blocks: the packer must be refused, not silently mis-read
    CHECK(!qsa_shard_cols_aligned(QsaHeadShard{N_HEAD, N_HEAD_KV, 0, 2}, 100, 128));
    CHECK(qsa_shard_cols_aligned(QsaHeadShard{N_HEAD, N_HEAD_KV, 0, 2}, 100, 100));

    // the same, cut UNEVENLY (24 heads over 8 KV heads, 3 groups to the lead and 5 to the partner): the runs
    // still tile the column axis and the gathered bytes are still the whole weight's, in order
    {
        const QsaHeadShard u0 = qsa_pair_shard(24, 8, 9, 0), u1 = qsa_pair_shard(24, 8, 9, 1);
        const int64_t block = 64;
        std::vector<float> full((size_t) n_embd * 24 * dim);
        for (size_t i = 0; i < full.size(); ++i) full[i] = (float) i * 0.5f;
        const QsaHeadRun d0 = qsa_out_cols(u0, dim), d1 = qsa_out_cols(u1, dim);
        CHECK(d0.start == 0 && d0.len == 9 * dim && d1.start == 9 * dim && d1.len == 15 * dim);
        std::vector<float> q0((size_t) n_embd * d0.len), q1((size_t) n_embd * d1.len);
        gdn_gather_cols(full.data(), n_embd, 24 * dim, block, (size_t) block * sizeof(float), {{d0.start, d0.len}},
                q0.data());
        gdn_gather_cols(full.data(), n_embd, 24 * dim, block, (size_t) block * sizeof(float), {{d1.start, d1.len}},
                q1.data());
        bool same = true;
        for (int64_t row = 0; row < n_embd && same; ++row)
            for (int64_t c = 0; c < d0.len; ++c)
                if (q0[(size_t) row * d0.len + c] != full[(size_t) row * 24 * dim + c]) same = false;
        for (int64_t row = 0; row < n_embd && same; ++row)
            for (int64_t c = 0; c < d1.len; ++c)
                if (q1[(size_t) row * d1.len + c] != full[(size_t) row * 24 * dim + d1.start + c]) same = false;
        CHECK(same);
        // and the uneven q rows: the lead's 9 heads' query+gate rows, then the partner's 15
        const QsaHeadRun ur0 = qsa_q_rows(u0, dim), ur1 = qsa_q_rows(u1, dim);
        CHECK(ur0.start == 0 && ur0.len == 9 * 2 * dim && ur1.start == 9 * 2 * dim && ur1.len == 15 * 2 * dim);
    }

    // ---- attn_k / attn_v / attn_q: whole rows per head, so gdn_gather_rows on the head axis
    // attn_k is [n_head_kv * head_dim rows][n_embd columns]; the card takes whole rows, so this is a row gather
    constexpr int64_t k_cols = 4;   // stands in for the model's 4096 input dim, which the gather cannot see
    const QsaHeadRun kr = qsa_kv_rows(r0, dim);
    std::vector<float> kfull((size_t) N_HEAD_KV * dim * k_cols), kpart((size_t) r0.nkv() * dim * k_cols);
    for (size_t i = 0; i < kfull.size(); ++i) kfull[i] = (float) (i % 977);
    gdn_gather_rows(kfull.data(), sizeof(float) * (size_t) k_cols, std::vector<GdnRun>{{kr.start, kr.len}},
            kpart.data());
    // the card's K rows are the whole matrix's rows [kv0 * head_dim, ...) verbatim
    CHECK(std::equal(kpart.begin(), kpart.end(), kfull.begin() + (ptrdiff_t) kr.start * k_cols));
    const QsaHeadRun qr = qsa_q_rows(r0, dim);
    CHECK(qr.start == r0.q0() * 2 * dim && qr.len == r0.nq() * 2 * dim);
    std::vector<float> qfull((size_t) N_HEAD * 2 * dim), qpart((size_t) r0.nq() * 2 * dim);
    for (size_t i = 0; i < qfull.size(); ++i) qfull[i] = (float) (i % 311) - 7.f;
    gdn_gather_rows(qfull.data(), sizeof(float), std::vector<GdnRun>{{qr.start, qr.len}}, qpart.data());
    CHECK(std::equal(qpart.begin(), qpart.end(), qfull.begin() + (ptrdiff_t) qr.start));
    // the pair's q rows cover every query head exactly once
    const QsaHeadRun qr1 = qsa_q_rows(r1, dim);
    CHECK(qr.start + qr.len == qr1.start && qr1.start + qr1.len == N_HEAD * 2 * dim);
}

void shards_are_comparable() {
    const QsaHeadShard a{N_HEAD, N_HEAD_KV, 0, 2}, a2{N_HEAD, N_HEAD_KV, 0, 2}, b{N_HEAD, N_HEAD_KV, 1, 2};
    CHECK(a == a2);
    CHECK(a != b);
    // a session's state and its buffers must be carved for the same shard; this is what the engine compares
    CHECK(QsaHeadShard(N_HEAD, N_HEAD_KV, 0, 1) != a);
}

}  // namespace

int main() {
    the_real_pair();
    unsplit_is_everything();
    the_indexer_stays_whole();
    what_is_refused();
    the_projection_runs();
    an_uneven_cut_is_still_one_model();
    a_cut_inside_a_group_costs_both_cards_that_group();
    the_weight_slices_are_the_whole_layers_bytes();
    shards_are_comparable();
    if (g_fail != 0) { std::printf("qsa head split test failed: %d\n", g_fail); return 1; }
    std::printf("qsa head split test ok\n");
    return 0;
}

// include/basalt/core/qsa_head_split.hpp - the QSA (full-attention) mixer split across a pair of cards: which query
// heads and which KV heads one card of the pair owns, and what that means for the state, the pools and the
// projections.  The GDN mixer's own split is gdn_head_split.hpp; this is its sibling for the 12 attention layers.
//
// THE SPLIT IS BY KV GROUP, NOT BY QUERY HEAD.  `QsaShapes` states the map: query head q reads KV head
// `q / (n_head / n_head_kv)`.  So a card takes whole KV heads and the query heads that read them, which keeps
// every card's work self-contained: `qsa_attend` for a query head touches exactly one KV head, and that KV head
// is on this card.  Cutting the query axis instead would put one query head's KV on the other card and turn
// every attention into a remote read.
//
// THE INDEXER DOES NOT SPLIT.  Its `idx_n_head` heads are not the attention's query heads - they are a separate
// 4-head projection whose scores are RELU'd per head and SUMMED over heads to pick the cells every query head
// then attends to.  Give each card two of the four and the two cards pick different top-k, which is a different
// attention input and a different token.  So the shard shapes keep the indexer whole and each card computes the
// same selection redundantly; that redundancy is what buys the bitwise claim.  Splitting it is possible (one
// collective to share the selection, plus a decision about whether a flipped tie is acceptable) and is not this
// branch's business.
//
// WHY IT IS BITWISE-EXACT AND THE ROW SPLIT IS NOT.  A head's attention is a sum over ITS OWN selected cells and
// its own head dimension; taking half the heads changes which sums happen where, never the terms of any sum.
// The projections contribute rows (and columns) of the same weight in the same order, and the o_proj column
// split's partials are summed by tp2_all_reduce in the documented order.  No tolerance and no reassociation.
//
// PURE ARITHMETIC: nothing here touches CUDA, so tests/core/qsa_head_split_test.cpp is the whole oracle.
#pragma once

#include "basalt/kernels/qsa.hpp"

#include <cstdint>
#include <vector>

namespace basalt::core {

/// One card's share of one stage's attention heads.  `rank` names the card within the pair (`tp2_pair.hpp`'s
/// lead is 0, its partner 1); `parts` is the pair's width (the engine's `--tp-size`).  `parts` 1 is every head on
/// one card, which is how a session without a pair describes itself.
struct QsaHeadShard {
    int64_t n_head = 0;     ///< the model's query heads (24)
    int64_t n_head_kv = 0;  ///< the model's KV heads (2)
    int rank = 0;           ///< which card of the pair this is
    int parts = 1;          ///< how many cards the stage's heads are cut between

    /// Query heads per KV head: the `q / rep` map of `QsaShapes`.
    int64_t rep() const { return n_head_kv > 0 ? n_head / n_head_kv : 0; }
    int64_t nkv() const { return parts > 0 ? n_head_kv / parts : 0; }
    int64_t kv0() const { return rank * nkv(); }
    int64_t nq() const { return nkv() * rep(); }
    int64_t q0() const { return kv0() * rep(); }
    bool whole() const { return parts <= 1; }

    /// Does this geometry cut at all?  The KV heads must divide, and the query axis must follow them exactly
    /// (`n_head == n_head_kv * rep()`), which is what makes the group rule hold.  The indexer is absent from
    /// this test on purpose: it never splits, so an odd `idx_n_head` is not a reason to refuse.
    bool ok() const {
        return n_head > 0 && n_head_kv > 0 && parts > 0 && rank >= 0 && rank < parts && n_head_kv % parts == 0 &&
               n_head_kv > 0 && n_head % n_head_kv == 0 && nkv() > 0;
    }
    bool operator==(const QsaHeadShard& o) const {
        return n_head == o.n_head && n_head_kv == o.n_head_kv && rank == o.rank && parts == o.parts;
    }
    bool operator!=(const QsaHeadShard& o) const { return !(*this == o); }
};

inline bool qsa_shard_fits(int64_t n_head, int64_t n_head_kv, int parts) {
    return QsaHeadShard{n_head, n_head_kv, 0, parts}.ok();
}

/// A row/column run of one projection axis, in the same shape gdn_head_split.hpp uses.
struct QsaHeadRun {
    int64_t start, len;
};

/// `attn_q.weight`'s rows for this card: each query head owns `2 * head_dim` rows, the query and its gate, so
/// the run is contiguous.  Same for the gate half - they are one block per head by construction.
inline QsaHeadRun qsa_q_rows(const QsaHeadShard& h, int64_t head_dim) {
    return {h.q0() * 2 * head_dim, h.nq() * 2 * head_dim};
}
/// `attn_k.weight`'s and `attn_v.weight`'s rows for this card.
inline QsaHeadRun qsa_kv_rows(const QsaHeadShard& h, int64_t head_dim) {
    return {h.kv0() * head_dim, h.nkv() * head_dim};
}
/// `attn_output.weight`'s COLUMNS for this card: the o_proj is summed over `n_head * head_dim` inputs, and a card
/// supplies its query heads' columns - which is why its output is a PARTIAL and needs the all-reduce.
inline QsaHeadRun qsa_out_cols(const QsaHeadShard& h, int64_t head_dim) {
    return {h.q0() * head_dim, h.nq() * head_dim};
}
/// True when the card's o_proj columns start and end on a `block_cols` boundary, which is what a blocked
/// (quantized) weight needs to slice columns at all.  head_dim 256 with 32/64/128-column blocks always passes;
/// the check exists so an odd geometry is refused rather than mis-read.
inline bool qsa_shard_cols_aligned(const QsaHeadShard& h, int64_t head_dim, int64_t block_cols) {
    if (block_cols <= 0) return false;
    const QsaHeadRun r = qsa_out_cols(h, head_dim);
    return r.start % block_cols == 0 && r.len % block_cols == 0;
}
/// The same for the K/V rows of a quantized `attn_k`/`attn_v` (their block axis is `n_embd`, so the rows are
/// whole rows; what has to align is the head count against whatever the caller packs per row).
inline bool qsa_shard_rows_aligned(const QsaHeadShard& h, int64_t head_dim, int64_t block_rows) {
    if (block_rows <= 0) return false;
    return (qsa_q_rows(h, head_dim).len % block_rows == 0) && (qsa_kv_rows(h, head_dim).len % block_rows == 0);
}

/// The bytes one cell of KV costs THIS card, for the fp16 layout: the pool is `[page][kv_head][page_size][head_dim]`
/// and the card holds `nkv()` of the KV heads, for K and V.  The other formats are `QsaState`'s business; this is
/// the figure the sizing sanity check reads.
inline uint64_t qsa_shard_kv_cell_bytes(const QsaHeadShard& h, int64_t head_dim) {
    return (uint64_t) h.nkv() * (uint64_t) head_dim * 2u * 2u;   // heads x dims x fp16 x (K,V)
}

/// The shapes the QSA kernels run with on this card.  Everything the attention follows is narrowed to the card's
/// heads; everything the indexer follows stays whole, because the selection has to be the same on both cards
/// (see the header).  An invalid shard returns the shapes unchanged, so a caller that forgot to check `ok()` gets
/// the unsplit layer rather than a half-width one.
inline basalt::kernels::QsaShapes qsa_shard_shapes(const basalt::kernels::QsaShapes& s, const QsaHeadShard& h) {
    basalt::kernels::QsaShapes out = s;
    if (!h.ok() || h.whole()) return out;
    out.n_head = h.nq();
    out.n_head_kv = h.nkv();
    return out;
}

/// The runs of the query axis, one per card: what the engine checks at start to be sure the pair covers every
/// head exactly once.  `parts` runs, each `nq()` long, starting at each rank's `q0()`.
inline std::vector<QsaHeadRun> qsa_head_runs(const QsaHeadShard& h) {
    std::vector<QsaHeadRun> r;
    if (!h.ok()) return r;
    for (int p = 0; p < h.parts; ++p) {
        const QsaHeadShard one{h.n_head, h.n_head_kv, p, h.parts};
        r.push_back({one.q0(), one.nq()});
    }
    return r;
}

}  // namespace basalt::core

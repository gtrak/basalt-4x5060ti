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
// THE CUT MAY BE UNEVEN, THE GROUP RULE STILL RULES.  A card states which QUERY heads it runs (`q_beg`/`nq_`);
// its KV run is derived from that run, so it holds every KV head its query heads read and a cut inside a KV group
// leaves the boundary group on both cards.  `parts` alone - the symmetric cut - is the special case where the
// query run is exactly whole groups, which is what the attention kernels run: they hold one group's heads
// together, so `qsa_shard_whole_groups` is what `qsa_shard_runs` refuses a straddling span on.
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
    /// An UNEVEN cut: this card's query-head run, [q_beg, q_beg+nq_).  `nq_` 0 - what a `parts`-only shard
    /// carries - is the symmetric run `rank`/`parts` implies.  The KV run FOLLOWS the query run, so a cut inside
    /// a KV group leaves that group's KV head on both cards; `qsa_shard_whole_groups` says which spans the
    /// attention kernels can run at all.
    int64_t q_beg = 0, nq_ = 0;

    /// Query heads per KV head: the `q / rep` map of `QsaShapes`.
    int64_t rep() const { return n_head_kv > 0 ? n_head / n_head_kv : 0; }
    bool span() const { return nq_ > 0; }
    /// The symmetric run `rank`/`parts` implies, used when no span was given.
    int64_t sym_nkv() const { return parts > 0 ? n_head_kv / parts : 0; }
    int64_t sym_kv0() const { return rank * sym_nkv(); }
    int64_t nq() const { return span() ? nq_ : sym_nkv() * rep(); }
    int64_t q0() const { return span() ? q_beg : sym_kv0() * rep(); }
    /// The first KV head this card holds: the group its first query head reads.
    int64_t kv0() const { return nq() > 0 && rep() > 0 ? q0() / rep() : 0; }
    /// The KV heads this card holds: every group its query heads read, the boundary group included.
    int64_t nkv() const {
        if (nq() <= 0 || rep() <= 0) return 0;
        return (q0() + nq() - 1) / rep() - kv0() + 1;
    }
    bool whole() const { return !span() && parts <= 1; }

    /// Does this geometry cut at all?  The query axis must follow the KV groups exactly
    /// (`n_head == n_head_kv * rep()`), which is what makes the group rule hold, and the card's run - the
    /// symmetric one or the span it was given - must sit inside it.  The indexer is absent from this test on
    /// purpose: it never splits, so an odd `idx_n_head` is not a reason to refuse.
    bool ok() const {
        if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0) return false;
        if (parts <= 0 || rank < 0 || rank >= parts) return false;
        if (span()) return q_beg >= 0 && q_beg + nq_ <= n_head;
        return n_head_kv % parts == 0 && nkv() > 0;
    }
    bool operator==(const QsaHeadShard& o) const {
        return n_head == o.n_head && n_head_kv == o.n_head_kv && rank == o.rank && parts == o.parts &&
               q_beg == o.q_beg && nq_ == o.nq_;
    }
    bool operator!=(const QsaHeadShard& o) const { return !(*this == o); }
};

/// One card of a pair whose stage's heads are cut unevenly: the lead runs query heads [0, lead_heads), the
/// partner runs the rest, so the two runs tile the query axis exactly once.  `lead_heads == n_head/2` gives the
/// symmetric cut in the form every pair has always used (no span), so a shard built this way with half the heads
/// is `==` to one built from `parts`.  An out-of-range head count gives a shard `ok()` refuses, which is how a
/// caller reports a ratio it was handed but cannot use.  The pair is two cards: the collective is a pair's.
inline QsaHeadShard qsa_pair_shard(int64_t n_head, int64_t n_head_kv, int64_t lead_heads, int rank) {
    QsaHeadShard h{n_head, n_head_kv, rank, 2};
    if (lead_heads <= 0 || lead_heads >= n_head) { h.nq_ = n_head + 1; return h; }   // off the axis: ok() refuses it
    if (lead_heads * 2 == n_head) return h;   // the symmetric cut, in its own form
    if (rank == 0) { h.q_beg = 0; h.nq_ = lead_heads; }
    else { h.q_beg = lead_heads; h.nq_ = n_head - lead_heads; }
    return h;
}

/// The lead's query heads for a ratio of the stage's heads - `--exp qsa_ratio_stage_a` and its sibling - kept
/// inside the axis with at least one head for each card.  Rounding to a KV group is NOT done here: the ratio is
/// the caller's model of the cut, and `qsa_shard_whole_groups` is what says whether it is runnable.
inline int64_t qsa_lead_heads(int64_t n_head, double lead_ratio) {
    if (n_head <= 0) return 0;
    if (!(lead_ratio > 0.0)) return 1;
    if (!(lead_ratio < 1.0)) return n_head - 1;
    const double h = lead_ratio * (double) n_head;
    int64_t r = (int64_t) (h + 0.5);
    if (r < 1) r = 1;
    if (r > n_head - 1) r = n_head - 1;
    return r;
}

/// Can the QSA attention kernels run this shard?  They keep one KV head's query heads together, sized by a
/// compile-time heads-per-KV-head (the `G` of qsa_dec_attn.cu, qsa_decode_attn.cu and qsa_prompt_attn.cu), so a
/// card runs WHOLE KV groups: its query run starts and ends on a group boundary.  The symmetric cut always does;
/// an uneven one only at the granularity `rep()` allows - 12 heads at a time for 24 heads over 2 KV heads.
inline bool qsa_shard_whole_groups(const QsaHeadShard& h) {
    if (!h.ok() || h.whole()) return h.ok();
    const int64_t r = h.rep();
    return r > 0 && h.q0() % r == 0 && h.nq() % r == 0;
}

/// Do these two shards cover every query head exactly once?  What a pair checks at start; the two KV runs may
/// overlap on a boundary group (`qsa_shard_whole_groups` refuses such a pair for the attention kernels, but the
/// pools are sized for it, so the check is about the query axis alone).
inline bool qsa_pair_tiles(const QsaHeadShard& a, const QsaHeadShard& b) {
    if (!a.ok() || !b.ok() || a.n_head != b.n_head || a.n_head_kv != b.n_head_kv) return false;
    if (a.parts != b.parts || a.rank == b.rank) return false;
    const int64_t a0 = a.q0(), an = a.nq(), b0 = b.q0(), bn = b.nq();
    const int64_t lo = a0 <= b0 ? a0 : b0, lo_n = a0 <= b0 ? an : bn;
    const int64_t hi = a0 <= b0 ? b0 : a0, hi_n = a0 <= b0 ? bn : an;
    return lo == 0 && lo + lo_n == hi && hi + hi_n == a.n_head;
}

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
/// head exactly once.  A shard that carries a span names its pair's lead's run, so the complement is derived;
/// a symmetric shard gives `parts` runs, each `nq()` long, starting at each rank's `q0()`.
inline std::vector<QsaHeadRun> qsa_head_runs(const QsaHeadShard& h) {
    std::vector<QsaHeadRun> r;
    if (!h.ok()) return r;
    if (h.span()) {   // the pair the span belongs to: the lead's run, then the rest of the axis
        const int64_t lead = h.rank == 0 ? h.nq_ : h.q_beg;
        r.push_back({0, lead});
        r.push_back({lead, h.n_head - lead});
        return r;
    }
    for (int p = 0; p < h.parts; ++p) {
        const QsaHeadShard one{h.n_head, h.n_head_kv, p, h.parts};
        r.push_back({one.q0(), one.nq()});
    }
    return r;
}

}  // namespace basalt::core

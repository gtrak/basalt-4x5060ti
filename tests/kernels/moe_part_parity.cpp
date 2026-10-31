// tests/prefill/moe_part_parity.cpp - the PARTIAL MERGE's reduction order, on the CPU (no device, no CUDA).
//
// `--exp peer_partial_merge` lets a partner card sum the rows of the experts it holds into one `T x n_embd`
// partial a layer and send THAT across, instead of the rows.  The MoE sum of a token is then two fma chains
// joined by one add, where the single-card combine runs one chain over all K routings - and on the BF16 wire the
// partner's whole run is rounded once more before the join.  This test pins what that costs:
//
//   1. no peer rows at all: the split combine is BITWISE the single-card combine (a boot with the flag off is not
//      a new arithmetic, which is the identity the mode is graded against);
//   2. the FP32 wire (`--peer-rows-fp32`): the difference against the flag-off combine is reassociation alone, a
//      handful of fp32 ulps of the terms' magnitudes;
//   3. the BF16 wire (the default): against the flag-off combine, which ALREADY reads the partner's rows as BF16
//      bits, the difference is the partial's one extra rounding - half an ulp of the PARTIAL rather than of the
//      total, which is the numerical argument of docs/precision/moe-partial-merge.md;
//   4. the fold's order is reproducible: a token's rows chained group by group give bit for bit the same partial
//      as one chain over the same rows in the same order (group order, then k order within a group);
//   5. the fold's compact per-token list and the routing table agree: summing a token's held rows from either
//      comes out bitwise equal, which is what `peer_fold_tables` promises the fold kernel it feeds.
//
// The arithmetic under all of this is `basalt/prefill/moe_part.hpp`, which the device kernels call, and the fold's
// per-token list is built by `basalt/prefill/moe_fold.hpp`, which the prefill thread calls - so this measures the
// engine's chain and the engine's own table builder, not transcriptions of them.
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/prefill/moe_fold.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/prefill/moe_part.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <vector>

namespace {

using basalt::kernels::bf16_from_f32;
using basalt::kernels::f32_from_bf16;
using basalt::kernels::bf16_from_f32;
using basalt::kernels::f32_from_bf16;
using basalt::prefill::kMoeK;
using basalt::prefill::moe_fold_tables;
using basalt::prefill::MoeFoldTables;
using basalt::prefill::PeerRows;
namespace fp4 = basalt::prefill::fp4;   // only for the group cutter the prefill thread itself calls

constexpr int64_t T = 97;             // a chunk, deliberately not a round number
constexpr int64_t ND = 40;            // the test's n_embd: the functions price it as a parameter
constexpr int64_t ROWS = T * kMoeK;
constexpr int32_t GROUP_ROWS = 7;     // a fold group's rows, small so several groups cover the peer's block
constexpr double kEps32 = 1.1920929e-7;      // fp32's ulp at 1
// bf16 keeps 7 mantissa bits explicitly and one implicitly: eight significant bits, an ulp of 2^-7 relative, and
// round-to-nearest-even lands within half of that (see basalt/kernels/bf16_bits.hpp for the rounding itself)
constexpr double kHalfUlpBf16 = 1.0 / 256.0;

/// One synthetic MoE layer.  `slot[t * K + k]` is the absolute row a routing produced, `inv[r]` the routing that
/// produced a row, and `rows[r * ND + d]` the down products' FP32 output.
struct Layer {
    std::vector<int32_t> slot, inv;
    std::vector<float> rows, w, shared, sg;
};

/// A routing that looks like one: rows scattered over the row space, and one token's K weights a softmax of sorts.
Layer make_layer(uint32_t seed) {
    Layer L;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    L.slot.resize((size_t) ROWS);
    L.inv.resize((size_t) ROWS);
    for (int64_t i = 0; i < ROWS; ++i) L.slot[(size_t) i] = (int32_t) i;
    for (int64_t i = ROWS - 1; i > 0; --i) {   // a Fisher-Yates shuffle: expert order and k order are unrelated
        std::uniform_int_distribution<int64_t> d(0, i);
        std::swap(L.slot[(size_t) i], L.slot[(size_t) d(rng)]);
    }
    for (int64_t i = 0; i < ROWS; ++i) L.inv[(size_t) L.slot[(size_t) i]] = (int32_t) i;
    L.rows.resize((size_t) ROWS * ND);
    for (float& x : L.rows) x = u(rng);
    L.w.assign((size_t) ROWS, 0.0f);
    for (int64_t t = 0; t < T; ++t) {
        float s = 0;
        for (int k = 0; k < kMoeK; ++k) {
            const float v = std::abs(u(rng)) + 0.01f;
            L.w[(size_t) (t * kMoeK + k)] = v;
            s += v;
        }
        for (int k = 0; k < kMoeK; ++k) L.w[(size_t) (t * kMoeK + k)] /= s;
    }
    L.shared.resize((size_t) T * ND);
    for (float& x : L.shared) x = u(rng);
    L.sg.resize((size_t) T);
    for (float& x : L.sg) x = u(rng);
    return L;
}

/// A routing with heavy hitters: the last few tokens own a contiguous run of `K` rows each at the top of the row
/// space, so all of a token's routings fall inside the peer's block and straddle fold groups - a run of K split
/// across groups is the case that a per-group count can get wrong, and it is what a hot expert produces in a real
/// chunk. Every other routing is placed as in `make_layer`.
Layer make_layer_skewed(uint32_t seed, int heavy) {
    Layer L;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    L.slot.resize((size_t) ROWS);
    L.inv.resize((size_t) ROWS);
    std::vector<char> used((size_t) ROWS, 0);
    for (int32_t h = 0; h < heavy; ++h) {   // heavy token h owns rows [ROWS-(h+1)K, ROWS-hK)
        const int32_t t = (int32_t) T - 1 - h;
        for (int k = 0; k < kMoeK; ++k) {
            const int32_t r = (int32_t) ROWS - (h + 1) * kMoeK - (kMoeK - 1 - k);
            L.slot[(size_t) (t * kMoeK + k)] = r;
            used[(size_t) r] = 1;
        }
    }
    std::vector<int32_t> free_rows;
    for (int32_t r = 0; r < (int32_t) ROWS; ++r) if (!used[(size_t) r]) free_rows.push_back(r);
    for (size_t i = free_rows.size(); i-- > 1;) {   // a Fisher-Yates over what is left
        std::uniform_int_distribution<size_t> d(0, i);
        std::swap(free_rows[i], free_rows[d(rng)]);
    }
    size_t next = 0;
    for (int64_t i = 0; i < ROWS; ++i) {
        const int32_t t = (int32_t) (i / kMoeK);
        if (t >= (int32_t) T - heavy) continue;   // already placed
        L.slot[(size_t) i] = free_rows[next++];
    }
    for (int64_t i = 0; i < ROWS; ++i) L.inv[(size_t) L.slot[(size_t) i]] = (int32_t) i;
    L.rows.resize((size_t) ROWS * ND);
    for (float& x : L.rows) x = u(rng);
    L.w.assign((size_t) ROWS, 0.0f);
    for (int64_t t = 0; t < T; ++t) {
        float s = 0;
        for (int k = 0; k < kMoeK; ++k) {
            const float v = std::abs(u(rng)) + 0.01f;
            L.w[(size_t) (t * kMoeK + k)] = v;
            s += v;
        }
        for (int k = 0; k < kMoeK; ++k) L.w[(size_t) (t * kMoeK + k)] /= s;
    }
    L.shared.resize((size_t) T * ND);
    for (float& x : L.shared) x = u(rng);
    L.sg.resize((size_t) T);
    for (float& x : L.sg) x = u(rng);
    return L;
}

float shared_term(const Layer& L, int64_t t, int64_t d) {
    const float sig = 1.0f / (1.0f + std::exp(-L.sg[(size_t) t]));   // the combine's `sigm`, same shape
    return L.shared[(size_t) (t * ND + d)] * sig;
}

/// The single-card combine: one chain over the token's K routings, then the shared expert's term.  This is
/// `moe_combine`'s call.  With `rows16` the rows in `[lo, hi)` - the partner's block - are read from a BF16 plane
/// instead of the FP32 row space, which is today's wire (`--peer-rows-fp32` off): the rows already cross as
/// BF16 bits, so a boot with the flag OFF is the baseline the candidate is graded against, rounding and all.
float combine_one_card(const Layer& L, int64_t t, int64_t d, int32_t lo, int64_t hi, const uint16_t* rows16) {
    // no BF16 plane means no peer block either: with the FP32 row wire the partner's rows sit in the FP32 row
    // space like the lead's own, which is `--peer-rows-fp32`'s (and the flag-off, routed-chunk) case
    const int32_t blo = rows16 ? lo : 0, bhi = rows16 ? hi : 0;
    const float s = basalt::prefill::moe_routed_sum<PeerRows::kPlane>(L.slot.data(), L.w.data(), t * kMoeK,
                                                                      L.rows.data(), rows16, blo, bhi, ND, d);
    return s + shared_term(L, t, d);
}

/// The lead's half of a partial merge: its own chain with the partner's block skipped, the partner's partial
/// joined to it, and only then the shared expert's term (`moe_combine_partial`'s last line).
float combine_split(const Layer& L, int64_t t, int64_t d, int32_t lo, int32_t hi, float part) {
    const float s = basalt::prefill::moe_routed_sum<PeerRows::kSkip>(L.slot.data(), L.w.data(), t * kMoeK,
                                                                     L.rows.data(), nullptr, lo, hi, ND, d);
    return basalt::prefill::moe_combine_out(s, part, shared_term(L, t, d));
}

/// One fold group in the shape the ENGINE builds for the fold kernel: `tok[e]` a token the group holds rows of,
/// `off[e..e+1]` its run in `rel` and `wt`, the run in k order. The list is built by `moe_fold_tables` - the very
/// function `peer_fold_tables` calls - from the group's rows described the same way the prefill thread describes
/// them, so a bug in the builder fails here instead of only on a card.
struct Group {
    int32_t lo, hi;      // the group's rows within the peer's block
    int32_t base;        // the block's first row: the BF16 plane is numbered from it, the staging from 0
    std::vector<int32_t> tok, off, rel;   // `rel`: the row's index WITHIN the group, how the group's staging reads it
    std::vector<float> wt;
};

/// The same, with the group's row count chosen by `cut(first, remaining)` - which is how the engine's own group cut
/// (`fp4::cut_pieces`) works: contiguous runs of the peer's block, each bounded by the staging's cap, an expert's
/// run split at whole tiles, and a group of at most `kMaxExperts` pieces. A fixed 7 can miss what a real cut does,
/// so the tests hand it the shapes the FP4 leg actually produces.
std::vector<Group> make_groups_cut(const Layer& L, int32_t lo, int32_t hi,
                                   std::function<int32_t(int32_t, int32_t)> cut) {
    std::vector<Group> gs;
    MoeFoldTables scratch;
    scratch.cnt.assign((size_t) T + 1, 0);
    scratch.cur.assign((size_t) T + 1, 0);
    for (int32_t g0 = lo; g0 < hi;) {
        Group g;
        g.lo = g0;
        g.hi = std::min(hi, g0 + std::max<int32_t>(1, cut(g0, hi - g0)));
        g.base = lo;
        g0 = g.hi;
        const int32_t nr = g.hi - g.lo;
        std::vector<int32_t> tok_of_row((size_t) nr), k_of_row((size_t) nr);
        std::vector<float> w_of_row((size_t) nr);
        for (int32_t q = 0; q < nr; ++q) {
            const int32_t i = L.inv[(size_t) (g.lo + q)];   // the routing that produced this row
            tok_of_row[(size_t) q] = i / kMoeK;
            k_of_row[(size_t) q] = i % kMoeK;
            w_of_row[(size_t) q] = L.w[(size_t) i];
        }
        moe_fold_tables(scratch, tok_of_row.data(), k_of_row.data(), w_of_row.data(), nr);
        g.tok = scratch.tok;
        g.off = scratch.off;
        g.rel = scratch.rel;
        g.wt = scratch.wt;
        gs.push_back(std::move(g));
    }
    return gs;
}

/// Describe the group's rows as the prefill thread does - group-relative - and let the shared builder sort them.
/// `k_of_row`/`w_of_row` stand in for `PeerPrefill::fold_k_h`/`w_row_h`, which the lead fills from its router.
std::vector<Group> make_groups(const Layer& L, int32_t lo, int32_t hi) {
    return make_groups_cut(L, lo, hi, [](int32_t, int32_t) { return GROUP_ROWS; });
}

/// The group's rows as the block's BF16 staging numbers them. The peer folds out of its OWN per-group staging,
/// which starts at 0, but the returned rows land in the lead's block at `lo + q`, which is how the flag-off
/// combine reads them back - so the plane this test builds is numbered from the block, and a folded row has to
/// be named in the same numbering to be comparable.
std::vector<int32_t> block_ids(const Group& g) {
    std::vector<int32_t> ids(g.rel.size());
    for (size_t i = 0; i < g.rel.size(); ++i) ids[i] = g.lo + g.rel[i] - g.base;
    return ids;
}

/// The group's rows as the FP32 row space numbers them: absolute ids, which is how `rows` is indexed.
std::vector<int32_t> absolute_ids(const Group& g) {
    std::vector<int32_t> abs(g.rel.size());
    for (size_t i = 0; i < g.rel.size(); ++i) abs[i] = g.lo + g.rel[i];
    return abs;
}

/// The partner's fold, exactly as `peer_part_fold_kernel` runs it: one thread per (token, dimension) of a group,
/// the group's rows chained onto whatever the cell already holds, the groups in launch order.  `plane` is the
/// staging's format - FP32 rows, or the BF16 bits the down product's epilogue wrote (`--peer-rows-fp32` off).  The
/// partial is rounded for the BF16 wire ONCE, after every group has added to it, exactly where the pack kernel
/// runs - after the group loop - so the fold's intermediate cells stay fp32.
std::vector<float> fold_partial(const Layer& L, const std::vector<Group>& gs, const uint16_t* plane, bool pack) {
    std::vector<float> part((size_t) T * ND, 0.0f);
    for (const Group& g : gs)
        for (size_t e = 0; e < g.tok.size(); ++e)
            for (int64_t d = 0; d < ND; ++d) {
                float* cell = &part[(size_t) g.tok[e] * ND + d];
                *cell = basalt::prefill::moe_fold_rows(*cell,
                                                       plane ? block_ids(g).data() + g.off[e]
                                                             : absolute_ids(g).data() + g.off[e],
                                                       g.wt.data() + g.off[e], (int32_t) (g.off[e + 1] - g.off[e]),
                                                       plane, plane ? nullptr : L.rows.data(), ND, d);
            }
    if (pack)
        for (float& x : part) x = basalt::prefill::moe_part_unpack(basalt::prefill::moe_part_pack(x));
    return part;
}

/// The peer block's rows as BF16 bits: the staging the down products' epilogue filled.  Numbered from the block's
/// first row, which is how the combine reads a returned row and how the fold reads a staged one.
std::vector<uint16_t> bf16_plane(const Layer& L, int32_t lo, int32_t hi) {
    std::vector<uint16_t> plane((size_t) (hi - lo) * ND);
    for (int32_t r = lo; r < hi; ++r)
        for (int64_t d = 0; d < ND; ++d)
            plane[(size_t) (r - lo) * ND + d] = bf16_from_f32(L.rows[(size_t) r * ND + d]);
    return plane;
}

struct Bound {
    double max_abs = 0;        // the worst |split - single| over the chunk
    double worst_ratio = 0;    // ... as a fraction of the bound it should obey
    double worst_vs_partial = 0;   // ... and as a fraction of the partial it rounds
};

void check_groups(const Layer& L, int32_t lo, int32_t hi, bool bf16_wire, const std::vector<Group>& gs, Bound& b) {
    const std::vector<uint16_t> plane = bf16_plane(L, lo, hi);
    const uint16_t* rows16 = bf16_wire ? plane.data() : nullptr;
    const std::vector<float> part = fold_partial(L, gs, rows16, bf16_wire);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t d = 0; d < ND; ++d) {
            double terms = 0;   // every term of the token's routed sum, both cards'
            for (int k = 0; k < kMoeK; ++k) {
                const int64_t i = t * kMoeK + k;
                const int32_t r = L.slot[(size_t) i];
                const double v = rows16 ? f32_from_bf16(rows16[(size_t) (r - lo) * ND + d])
                                        : (double) L.rows[(size_t) r * ND + d];
                terms += std::abs((double) L.w[(size_t) i] * v);
            }
            const double p = std::abs((double) part[(size_t) t * ND + d]);
            const double err = std::abs((double) combine_one_card(L, t, d, lo, hi, rows16) -
                                        (double) combine_split(L, t, d, lo, hi, part[(size_t) t * ND + d]));
            const double lim = (bf16_wire ? kHalfUlpBf16 * p : 0.0) + 32.0 * kEps32 * (terms + p);
            b.max_abs = std::max(b.max_abs, err);
            if (lim > 0) b.worst_ratio = std::max(b.worst_ratio, err / lim);
            // the partial's own ulp only bounds the error when the partial is a real part of the sum: a partial
            // that cancelled to nothing has an ulp of its own to no purpose, and the fp32 chain's ulps dominate
            if (p > 0.1 * terms) b.worst_vs_partial = std::max(b.worst_vs_partial, err / p);
        }
}

void check(const Layer& L, int32_t lo, int32_t hi, bool bf16_wire, Bound& b) {
    check_groups(L, lo, hi, bf16_wire, make_groups(L, lo, hi), b);
}

/// Two floats the same bits, or both zero: `-0.0 + 0.0` is `+0.0`, and a sum that happens to land on zero must
/// not make the identity check fail over its sign.
bool same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0 || (a == 0.0f && b == 0.0f);
}

}  // namespace

int main() {
    int fails = 0;

    // 1. no peer rows at all: the split path is the single-card path, bit for bit
    {
        const Layer L = make_layer(20241031);
        const std::vector<Group> none;
        const std::vector<float> part = fold_partial(L, none, nullptr, false);
        int64_t differs = 0;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < ND; ++d)
                if (!same_bits(combine_one_card(L, t, d, 0, 0, nullptr),
                              combine_split(L, t, d, 0, 0, part[(size_t) t * ND + d])))
                    ++differs;
        std::printf("1. no peer rows: %lld of %lld outputs differ from the single-card combine\n",
                    (long long) differs, (long long) (T * ND));
        if (differs != 0) ++fails;
    }

    // 2/3. a real split, four routings: the FP32 wire is reassociation alone; the BF16 wire adds one rounding of
    // the partial, which must dominate and must stay inside half an ulp of the partial itself
    const int32_t lo = (int32_t) (ROWS / 3);   // the partner holds two thirds of the row space
    for (uint32_t seed = 0; seed < 4; ++seed) {
        const Layer L = make_layer(1000 + seed);
        Bound f32, bf16;
        check(L, lo, (int32_t) ROWS, false, f32);
        check(L, lo, (int32_t) ROWS, true, bf16);
        std::printf("2. seed %u FP32 wire: max |diff| %.3e, %.3f of its bound\n", seed, f32.max_abs, f32.worst_ratio);
        std::printf("3. seed %u BF16 wire: max |diff| %.3e, %.3f of its bound, %.3e of the partial\n",
                    seed, bf16.max_abs, bf16.worst_ratio, bf16.worst_vs_partial);
        if (!(f32.worst_ratio <= 1.0)) { std::printf("   FAIL: the FP32 wire exceeded its bound\n"); ++fails; }
        if (!(bf16.worst_ratio <= 1.0)) { std::printf("   FAIL: the BF16 wire exceeded its bound\n"); ++fails; }
        if (!(bf16.worst_vs_partial <= kHalfUlpBf16 + 32.0 * kEps32)) {
            std::printf("   FAIL: the BF16 wire moved the output by more than half an ulp of the partial\n");
            ++fails;
        }
        if (bf16.max_abs < f32.max_abs) {   // the extra rounding can only add to the difference
            std::printf("   FAIL: the BF16 wire differed less than the FP32 one\n");
            ++fails;
        }
    }

    const Layer L = make_layer(20241031);
    const std::vector<Group> gs = make_groups(L, lo, (int32_t) ROWS);

    // 4. the fold's order is reproducible: the same rows, chained group by group or in one chain over the same
    // sequence, are bitwise the same partial - so the fold may be cut into groups by the placement and re-cut
    {
        std::vector<float> flat((size_t) T * ND, 0.0f);
        for (const Group& g : gs)
            for (size_t e = 0; e < g.tok.size(); ++e) {
                std::vector<int32_t> rel;
                std::vector<float> wt;
                for (const Group& g2 : gs)      // the token's rows in group order...
                    for (size_t e2 = 0; e2 < g2.tok.size(); ++e2)
                        if (g2.tok[e2] == g.tok[e])
                            for (int32_t q = g2.off[e2]; q < g2.off[e2 + 1]; ++q) {
                                rel.push_back(g2.rel[(size_t) q] + g2.lo);   // ... and k order within each
                                wt.push_back(g2.wt[(size_t) q]);
                            }
                for (int64_t d = 0; d < ND; ++d)
                    flat[(size_t) g.tok[e] * ND + d] = basalt::prefill::moe_fold_rows(
                        0.0f, rel.data(), wt.data(), (int32_t) rel.size(), nullptr, L.rows.data(), ND, d);            }
        const std::vector<float> grouped = fold_partial(L, gs, nullptr, false);
        int64_t differs = 0;
        for (size_t i = 0; i < flat.size(); ++i)
            if (!same_bits(flat[i], grouped[i])) ++differs;
        std::printf("4. group-by-group vs one chain: %lld of %lld partial cells differ\n", (long long) differs,
                    (long long) flat.size());
        if (differs != 0) ++fails;
    }

    // 5. the fold's compact list and the routing table say the same thing, bitwise: `peer_fold_tables` owes the
    // fold kernel a list of exactly the token's rows in the block, in k order, with their weights
    {
        double worst = 0;
        for (const Group& g : gs)
            for (size_t e = 0; e < g.tok.size(); ++e)
                for (int64_t d = 0; d < ND; ++d) {
                    const int64_t t = g.tok[e];
                    float via_list = 0.0f, via_table = 0.0f;
                    for (int32_t q = g.off[e]; q < g.off[e + 1]; ++q)
                        via_list = basalt::prefill::moe_term(via_list, g.wt[(size_t) q],
                                                             L.rows[(size_t) (g.lo + g.rel[(size_t) q]) * ND + d]);
                    for (int k = 0; k < kMoeK; ++k) {
                        const int64_t i = t * kMoeK + k;
                        const int32_t r = L.slot[(size_t) i];
                        if (r < g.lo || r >= g.hi) continue;
                        via_table = basalt::prefill::moe_term(via_table, L.w[(size_t) i],
                                                              L.rows[(size_t) r * ND + d]);
                    }
                    worst = std::max(worst, (double) std::abs(via_list - via_table));
                }
        std::printf("5. fold list vs routing table: max |diff| %.3e (the same chain, so 0)\n", worst);
        if (worst != 0.0) ++fails;
    }

    // and the wire's format: a partial the peer packed for the BF16 leg survives the lead's unpack unchanged, so
    // the lead adds the bits it was sent and not a value re-rounded on the way.  NaNs are excluded: the engine's
    // BF16 rounding quiets a NaN's payload, as ggml's does, which is not a change this mode makes.
    {
        int64_t differs = 0, checked = 0;
        for (int32_t h = 0; h < 65536; ++h) {
            if ((h & 0x7F80) == 0x7F80 && (h & 0x7F) != 0) continue;   // a NaN: quieting loses the payload
            const float v = f32_from_bf16((uint16_t) h);
            ++checked;
            if (basalt::prefill::moe_part_unpack(basalt::prefill::moe_part_pack(v)) != v) ++differs;
        }
        std::printf("   a packed partial round-trips: %lld of %d bf16 values changed\n", (long long) differs,
                    (int) checked);
        if (differs != 0) ++fails;
    }

    // 6. the fold list covers the block EXACTLY once, under the routings that stress it: a heavy hitter every
    //    token routes to, one token whose whole K sits in one group, groups the routing leaves empty, and the
    //    chunk's boundary tokens. A dropped row leaves the MoE sum silently short and a duplicated one silently
    //    long, and neither shows as a bad shape - so this counts rows rather than comparing chains.
    {
        int rows_missed = 0, rows_doubled = 0, list_broken = 0;
        // four placements of the peer's block: a third in, at the clamp, at the very end, and one row wide
        const int32_t bases[] = {(int32_t) (ROWS / 3), (int32_t) (ROWS / 2), (int32_t) (ROWS - ROWS / 8),
                                 (int32_t) (ROWS - 1)};
        for (const int32_t b : bases) {
            for (uint32_t seed = 0; seed < 3; ++seed) {
                const Layer L = (seed & 1) ? make_layer_skewed(7000 + seed, 4) : make_layer(7000 + seed);
                const std::vector<Group> g2 = make_groups(L, b, (int32_t) ROWS);
                std::vector<int> seen((size_t) ROWS, 0);
                for (const Group& g : g2) {
                    if (g.off.empty() || g.off.back() != g.hi - g.lo) ++list_broken;   // the runs must tile the group
                    for (size_t e = 0; e < g.tok.size(); ++e) {
                        for (int32_t q = g.off[e]; q < g.off[e + 1]; ++q) {
                            const int32_t r = g.lo + g.rel[(size_t) q];
                            if (r < g.lo || r >= g.hi || L.inv[(size_t) r] / kMoeK != g.tok[e]) ++list_broken;
                            ++seen[(size_t) r];
                        }
                    }
                }
                for (int32_t r = b; r < (int32_t) ROWS; ++r) {
                    if (seen[(size_t) r] == 0) ++rows_missed;
                    else if (seen[(size_t) r] > 1) ++rows_doubled;
                }
            }
        }
        std::printf("6. fold coverage over 4 block placements x 3 seeds (half skewed): missed=%d doubled=%d malformed=%d\n",
                    rows_missed, rows_doubled, list_broken);
        if (rows_missed || rows_doubled || list_broken) ++fails;
    }

    // 7. the TERM SET, not just the row count. The partner's fold must read exactly the rows the lead skips, each
    //    at its own router weight: a row read at another row's weight passes a coverage count and is still a wrong
    //    sum, which is the failure the gate saw and this test could not. Three things real routing does and the
    //    checks above hold fixed:
    //      - the peer's block is NOT the top of the row space. The row space is lead | peer | CPU tier, so the
    //        combine reads an FP32 tail above the peer's block that the fold never sees;
    //      - the groups come from the engine's own cutter (`fp4::cut_pieces`) over real expert runs - hot experts
    //        with long runs cut at whole tiles, cold ones with one piece, groups the cap and the kMaxExperts limit
    //        split - rather than fixed sevens;
    //      - a token's K routings straddle several of those groups, so its chain is joined group by group.
    {
        int ledger_bad = 0, empty_groups = 0, split_experts = 0, groups_seen = 0;
        // (peer share, CPU tier) as row counts, and the staging cap the group is cut to
        const std::pair<int32_t, int32_t> splits[] = {
            {(int32_t) (ROWS / 2), (int32_t) (ROWS / 4)},   // lead | peer | cpu, the running case
            {(int32_t) (ROWS / 3), (int32_t) (ROWS / 3)},   // the 50% clamp's neighbourhood, a fat CPU tier
            {(int32_t) (ROWS / 8), (int32_t) (ROWS / 2)},   // a thin peer share under a fat CPU tier
            {(int32_t) (ROWS - ROWS / 16), 0},              // no CPU tier: the peer flush with the top
            {0, (int32_t) (ROWS / 3)},                       // an empty peer block: nothing to fold, nothing to skip
        };
        const int64_t caps[] = {64, 256, (int64_t) ROWS};   // a tile, a small staging, the whole block
        for (const auto& sp : splits)
            for (uint32_t seed = 0; seed < 3; ++seed)
                for (const int64_t cap : caps) {
                    const Layer L = (seed & 1) ? make_layer_skewed(31000 + seed, 5) : make_layer(31000 + seed);
                    const int32_t lo = (int32_t) (ROWS - sp.first - sp.second), hi = lo + sp.first;

                    // The peer's experts in the order the lead's split puts them, each with its own run: the runs
                    // are contiguous over the block exactly as `m.off` lays them out, and the run lengths mix short
                    // cold experts with a few long hot ones, which is what gives the cutter something to cut.
                    // Every 5th is cold enough to hold no rows at all: a run of nothing, the geometry
                    // `moe_fold_tables` guards against and `cut_pieces` must skip without shifting a later piece.
                    int32_t n_expert = std::min<int32_t>(512, std::max<int32_t>(1, sp.first / 4));
                    if (sp.first == 0) n_expert = 0;   // no expert holds a row of an empty block
                    std::vector<int32_t> cnt((size_t) n_expert, 1);
                    for (int32_t left = sp.first - n_expert, e = 0; left > 0; ++e) {
                        const int32_t take = std::min(left, (e % 7 == 6) ? 24 : 3);   // every 7th expert runs long
                        cnt[(size_t) (e % n_expert)] += take;
                        left -= take;
                    }
                    for (int32_t e = 4; e < n_expert; e += 5) {   // the cold experts, their rows to a hot one
                        cnt[(size_t) 0] += cnt[(size_t) e];
                        cnt[(size_t) e] = 0;
                    }
                    std::vector<int32_t> bounds((size_t) n_expert + 1);
                    bounds[(size_t) 0] = 0;   // the cutter sees the peer's block-relative rows, as `m.off` gives them
                    for (int32_t e = 0; e < n_expert; ++e)
                        bounds[(size_t) e + 1] = bounds[(size_t) e] + cnt[(size_t) e];
                    if (bounds.back() != sp.first) {   // the rows above must conserve exactly, or the case is moot
                        ++ledger_bad;
                        std::printf("   FAIL: peer %d cpu %d seed %u cap %lld: the expert runs hold %d rows, not %d\n",
                                    sp.first, sp.second, seed, (long long) cap, bounds.back(), sp.first);
                        continue;
                    }
                    fp4::Pieces pc;
                    fp4::cut_pieces(bounds.data(), (int) n_expert, cap, pc);
                    const int ngr = (int) pc.group.size() - 1;
                    // the pieces must tile the peer's block exactly, in order, with no piece for a cold expert
                    if (!pc.expert.empty() && (pc.bounds.front() != 0 || pc.bounds[pc.expert.size()] != sp.first))
                        ++ledger_bad;
                    for (size_t j = 0; j + 1 < pc.bounds.size(); ++j)
                        if (pc.bounds[j] > pc.bounds[j + 1]) ++ledger_bad;   // a cutter that shuffles the runs

                    std::vector<Group> gs;
                    for (int g2 = 0; g2 < ngr; ++g2) {
                        const int i0 = pc.group[(size_t) g2], ns = pc.group[(size_t) g2 + 1] - i0;
                        const int32_t r0 = lo + pc.bounds[(size_t) i0];   // the group's first row, absolute
                        const int32_t nr = pc.bounds[(size_t) i0 + ns] - pc.bounds[(size_t) i0];
                        if (ns > 1) ++split_experts;   // a group whose pieces came from more than one expert
                        if (nr == 0) ++empty_groups;
                        ++groups_seen;
                        Group g;
                        g.lo = r0; g.hi = r0 + nr; g.base = lo;
                        std::vector<int32_t> tok_of_row((size_t) nr), k_of_row((size_t) nr);
                        std::vector<float> w_of_row((size_t) nr);
                        for (int32_t q = 0; q < nr; ++q) {
                            const int32_t i = L.inv[(size_t) (r0 + q)];
                            tok_of_row[(size_t) q] = i / kMoeK;
                            k_of_row[(size_t) q] = i % kMoeK;
                            w_of_row[(size_t) q] = L.w[(size_t) i];
                        }
                        MoeFoldTables scratch;
                        scratch.cnt.assign((size_t) T + 1, 0);
                        scratch.cur.assign((size_t) T + 1, 0);
                        moe_fold_tables(scratch, tok_of_row.data(), k_of_row.data(), w_of_row.data(), nr);
                        g.tok = scratch.tok; g.off = scratch.off; g.rel = scratch.rel; g.wt = scratch.wt;
                        gs.push_back(std::move(g));
                    }

                    // What the lead skips, per token: exactly its routings whose rows sit in the peer's block.
                    std::map<int64_t, std::map<int64_t, int>> skipped, folded;
                    for (int64_t t = 0; t < T; ++t)
                        for (int k = 0; k < kMoeK; ++k) {
                            const int64_t i = t * kMoeK + k;
                            const int32_t r = L.slot[(size_t) i];
                            if (r >= lo && r < hi) ++skipped[t][i % kMoeK];
                        }
                    for (const Group& g : gs)
                        for (size_t e = 0; e < g.tok.size(); ++e)
                            for (int32_t q = g.off[e]; q < g.off[e + 1]; ++q) {
                                const int32_t r = g.lo + g.rel[(size_t) q];
                                const int64_t i = L.inv[(size_t) r];
                                folded[(int64_t) g.tok[e]][i % kMoeK] += 1;
                                if (g.wt[(size_t) q] != L.w[(size_t) i]) ++ledger_bad;   // the wrong row's weight
                                if (r < lo || r >= hi) ++ledger_bad;                     // a row outside the block
                            }
                    if (skipped != folded) {
                        ++ledger_bad;
                        std::printf("   FAIL: peer %d cpu %d seed %u cap %lld: the fold read a different term set "
                                    "than the lead skipped\n", sp.first, sp.second, seed, (long long) cap);
                    }
                    // and the sum on the FP32 wire: a term out of place is more than reassociation can absorb
                    Bound b;
                    check_groups(L, lo, hi, false, gs, b);
                    if (!(b.worst_ratio <= 1.0)) {
                        ++ledger_bad;
                        std::printf("   FAIL: peer %d cpu %d seed %u cap %lld exceeded the reassociation bound "
                                    "(%.3f of it, max |diff| %.3e)\n",
                                    sp.first, sp.second, seed, (long long) cap, b.worst_ratio, b.max_abs);
                    }
                }
        std::printf("7. term-set ledger over 5 lead/peer/cpu splits x 3 seeds x 3 staging caps, cut by "
                    "fp4::cut_pieces (%d groups, %d multi-piece, %d empty): mismatches=%d\n",
                    groups_seen, split_experts, empty_groups, ledger_bad);
        if (ledger_bad) ++fails;
    }

    std::printf("%s\n", fails == 0 ? "moe_part_parity: OK" : "moe_part_parity: FAILED");
    return fails == 0 ? 0 : 1;
}

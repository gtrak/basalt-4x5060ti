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
// The arithmetic under all five is `basalt/prefill/moe_part.hpp`, which the device kernels call, so this measures
// the engine's chain rather than a transcription of it.
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/prefill/moe_part.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using basalt::kernels::bf16_from_f32;
using basalt::kernels::f32_from_bf16;
using basalt::kernels::bf16_from_f32;
using basalt::kernels::f32_from_bf16;
using basalt::prefill::kMoeK;
using basalt::prefill::PeerRows;

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

/// One fold group in the shape `peer_fold_tables` builds for the fold kernel: `tok[e]` a token the group holds
/// rows of, `off[e..e+1]` its run in `rel` (the row's ABSOLUTE id, where the kernel indexes its group's staging)
/// and `wt` (its router weight), the run in k order.
struct Group {
    int32_t lo, hi;
    std::vector<int32_t> tok, off, rel;   // `rel`: the absolute row id, which is how the FP32 row space is read
    std::vector<float> wt;
    std::vector<int32_t> blk;             // ... and the same rows numbered from the peer block's first row, which
                                          // is how a BF16 plane indexes them: row `lo + i` sits at `i * n_embd`
};

size_t index_of(const std::vector<int32_t>& v, int32_t x) {
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] == x) return i;
    return v.size();   // unreachable: a group only lists tokens it has a row of
}

std::vector<Group> make_groups(const Layer& L, int32_t lo, int32_t hi) {
    std::vector<Group> gs;
    for (int32_t g0 = lo; g0 < hi; g0 += GROUP_ROWS) {
        Group g;
        g.lo = g0;
        g.hi = std::min(hi, g0 + GROUP_ROWS);
        const int32_t nr = g.hi - g.lo;
        for (int32_t q = 0; q < nr; ++q) {
            const int32_t t = L.inv[(size_t) (g.lo + q)] / kMoeK;
            if (std::find(g.tok.begin(), g.tok.end(), t) == g.tok.end()) g.tok.push_back(t);
        }
        g.off.assign(g.tok.size() + 1, 0);
        for (int32_t q = 0; q < nr; ++q) g.off[index_of(g.tok, L.inv[(size_t) (g.lo + q)] / kMoeK) + 1]++;
        for (size_t e = 0; e < g.tok.size(); ++e) g.off[e + 1] += g.off[e];
        g.rel.assign((size_t) g.off.back(), 0);
        g.wt.assign((size_t) g.off.back(), 0.0f);
        std::vector<int32_t> cur(g.tok.size(), 0);
        for (int32_t q = 0; q < nr; ++q) {   // the row order first, then a stable insertion sort into k order
            const int32_t r = g.lo + q;
            const size_t e = index_of(g.tok, L.inv[(size_t) r] / kMoeK);
            const int32_t p = (int32_t) g.off[e] + cur[e]++;
            g.rel[(size_t) p] = r;
            g.wt[(size_t) p] = L.w[(size_t) L.inv[(size_t) r]];
        }
        for (size_t e = 0; e < g.tok.size(); ++e) {
            int32_t* rel = g.rel.data() + g.off[e];
            float* wt = g.wt.data() + g.off[e];
            const int32_t n = g.off[e + 1] - g.off[e];
            for (int32_t i = 1; i < n; ++i) {
                const int32_t r = rel[i];
                const float v = wt[i];
                const int32_t key = L.inv[(size_t) r] % kMoeK;
                int32_t j = i - 1;
                while (j >= 0 && L.inv[(size_t) rel[j]] % kMoeK > key) {
                    rel[j + 1] = rel[j];
                    wt[j + 1] = wt[j];
                    --j;
                }
                rel[j + 1] = r;
                wt[j + 1] = v;
            }
        }
        g.blk.resize(g.rel.size());
        for (size_t i = 0; i < g.rel.size(); ++i) g.blk[i] = g.rel[i] - lo;
        gs.push_back(std::move(g));
    }
    return gs;
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
                                                       plane ? g.blk.data() + g.off[e] : g.rel.data() + g.off[e],
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

void check(const Layer& L, int32_t lo, int32_t hi, bool bf16_wire, Bound& b) {
    const std::vector<Group> gs = make_groups(L, lo, hi);
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
                                rel.push_back(g2.rel[(size_t) q]);   // ... and k order within each
                                wt.push_back(g2.wt[(size_t) q]);
                            }
                for (int64_t d = 0; d < ND; ++d)
                    flat[(size_t) g.tok[e] * ND + d] = basalt::prefill::moe_fold_rows(
                        0.0f, rel.data(), wt.data(), (int32_t) rel.size(), nullptr, L.rows.data(), ND, d);
            }
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
                                                             L.rows[(size_t) g.rel[(size_t) q] * ND + d]);
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

    std::printf("%s\n", fails == 0 ? "moe_part_parity: OK" : "moe_part_parity: FAILED");
    return fails == 0 ? 0 : 1;
}

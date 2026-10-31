// include/basalt/prefill/moe_part.hpp - the MoE combine's reduction order, in one place.
//
// The combine sums a token's K routed rows, each weighted by its router weight.  One card doing all K adds them
// in k order.  A PARTIAL MERGE (`--exp peer_partial_merge`) splits that sum across the pair: the peer folds the
// rows of the experts it holds into a `T x n_embd` partial of its own and sends that one vector a layer, and the
// lead adds it to the sum of the rows it kept.  Two chains instead of one, joined once - which is the whole
// numerical cost of the mode, so the two cards' kernels and the CPU parity test all go through the functions here
// and no code path writes the fma chain again by hand (docs/precision/moe-partial-merge.md).
//
// The order each card keeps is its own: the lead adds the rows it kept in k order, exactly as the single-card
// combine does; the peer adds the rows of one group in ROW order (the order its expert runs lay them out) and
// groups in launch order.  Both are fixed by the routing and the placement, so a run is reproducible; neither is
// the old single chain, and the difference is the ~1 bf16 ulp the gate measures.
//
// Header-only and `__host__ __device__`, like `bf16_bits.hpp`: the kernels call these and so does the test, which
// is what makes the test a check of the engine rather than of a transcription of it.
#pragma once

#include "basalt/kernels/bf16_bits.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace basalt::prefill {

/// The prompt path's routings per token.
constexpr int kMoeK = 10;

/// One weighted row added to the running sum: `sum + w * v`, one rounding.  Every MoE sum in the engine is built
/// from this, on either card.
BF16_HD inline float moe_term(float sum, float w, float v) {
    return fmaf(w, v, sum);
}

/// What a combine does with a row of the block [lo, hi) - the rows of the experts a PARTNER card holds: read them
/// from the BF16 plane it returned (`kPlane`), or ignore them, because the partner folded them into the layer's
/// partial and the caller adds that (`kSkip`).
enum class PeerRows { kPlane, kSkip };

/// One (token, dimension) of the routed sum over the token's K routings, in k order.
///
/// A row outside [lo, hi) is this card's own: FP32 at `rows[r * n_embd + d]`, or nothing when `rows` is null (a
/// card that holds no rows of the layer at all).  A row inside it is the partner's, read per `PEER` from the plane
/// at `(r - lo) * n_embd + d`.  A card with no partner passes `lo` = `hi` = 0, which leaves the plain k-order chain.
template <PeerRows PEER>
BF16_HD inline float moe_routed_sum(const int32_t* slot, const float* w, int64_t k0, const float* rows,
                                    const uint16_t* plane, int32_t lo, int32_t hi, int64_t n_embd, int64_t d) {
    float s = 0.0f;
#if defined(__CUDACC__)
#pragma unroll
#endif
    for (int k = 0; k < kMoeK; ++k) {
        const int32_t r = slot[k0 + k];
        if (r >= lo && r < hi) {
            if (PEER == PeerRows::kSkip || plane == nullptr) continue;
            s = moe_term(s, w[k0 + k], basalt::kernels::f32_from_bf16(plane[(int64_t) (r - lo) * n_embd + d]));
        } else if (rows != nullptr) {
            s = moe_term(s, w[k0 + k], rows[(int64_t) r * n_embd + d]);
        }
    }
    return s;
}

/// The peer's fold of one group's rows into the layer's partial: `run` rows, in row order, `rel[i]` the row's
/// index in the group's staging and `wt[i]` its router weight, chained onto whatever the cell already holds.
/// FP32 rows (`--peer-rows-fp32`) or the BF16 bits the down product's epilogue wrote, exactly the format the rows
/// would have crossed in.
BF16_HD inline float moe_fold_rows(float sum, const int32_t* rel, const float* wt, int32_t run,
                                   const uint16_t* rows16, const float* rows, int64_t n_embd, int64_t d) {
    for (int32_t i = 0; i < run; ++i) {
        const int64_t r = (int64_t) rel[i] * n_embd + d;
        sum = moe_term(sum, wt[i], rows16 != nullptr ? basalt::kernels::f32_from_bf16(rows16[r]) : rows[r]);
    }
    return sum;
}

/// The summed partial on the wire: ONE rounding of the whole run, where the row-wise wire rounded every row.
/// Null `part16` on the combine side means the pair was given the FP32 wire (`--peer-rows-fp32`).
BF16_HD inline uint16_t moe_part_pack(float part) { return basalt::kernels::bf16_from_f32(part); }
BF16_HD inline float moe_part_unpack(uint16_t packed) { return basalt::kernels::f32_from_bf16(packed); }

/// The combine's last step: the routed sum, the partner's partial (0 with none), then the shared expert's term -
/// in that order, so the pair's two chains join before anything else joins them.
BF16_HD inline float moe_combine_out(float routed, float part, float shared_term) {
    return (routed + part) + shared_term;
}

}  // namespace basalt::prefill

// include/basalt/program/tp2_pair.hpp - how a layer-split stage shares its experts between the two cards of its pair
// (`--tp-size 2`).  Pure arithmetic: generate.cpp places what these name, tests/program/tp2_pair_test.cpp is the oracle.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace basalt::program {

/// The stage's lead card: the card a stage's dense weights, session and state live on.  Stage `s` of a tp_size-wide
/// split leads on CUDA `s * tp_size`.  The cards are assigned in pairs, which is why `--split-device` is refused.
inline int pair_lead_dev(int stage, int tp_size) { return stage * tp_size; }
/// The stage's partner card: the card that holds the half of the stage's experts its lead card does not, and nothing
/// else.  CUDA `s * tp_size + 1`.
inline int pair_partner_dev(int stage, int tp_size) { return stage * tp_size + 1; }

/// The partner of the stage that leads on `lead_dev`, from the card itself rather than a stage number: a stage's
/// partner is the card beside its lead.  Asking for a partner with a stage number read off the engine's stage list
/// (which holds only the stages after the main card's) puts two stages' expert halves on one card and leaves the
/// other partner card empty, so a place that knows only the card asks here.  -1 when `lead_dev` leads no stage.
inline int pair_partner_of_lead(int lead_dev, int tp_size) {
    if (tp_size <= 1 || lead_dev < 0 || lead_dev % tp_size != 0) return -1;
    return lead_dev + 1;
}

/// True when the card of `rank` of a `tp_size`-wide pair holds `expert`.  Only the single-card stage (tp_size 1: the
/// lead holds every expert, there is no partner rank) has an id-only answer; under a rank-split pair the decision is
/// the expert's position in the stage's ranked list, which `pair_ranked` owns.
inline bool pair_holds(int32_t expert, int tp_size, int rank) {
    (void) expert;
    return tp_size == 1 && rank == 0;
}

/// An expert cache's ranked (layer, expert) pairs, hottest first, restricted to one card of the pair - in the same
/// order, so a card fills its own half of its stage's ranking hottest first.  The two halves are disjoint and cover
/// `ranked`, which is the invariant the resident complement and the RAM copy are built on.
inline std::vector<std::pair<int32_t, int32_t>> pair_ranked(
    const std::vector<std::pair<int32_t, int32_t>>& ranked, int tp_size, int rank) {
    if (tp_size <= 1) return ranked;
    // The split is by RANK POSITION, not expert id: the lead takes the hottest half of its stage's ranking, the
    // partner the cold half.  Measured (first tp2 boot, 06:51): an id-parity split routes ~50% of expert lookups
    // over PCIe (hit 52.8-77.6%, pcie 49-52%) because half of the HOT experts sit on the partner and route on every
    // token; the profile is the router's bias, so the hot half held locally takes most lookups and the routed tail
    // is the cold one.  Same tier depth (half the list each), same VRAM, disjoint and covering as before.
    const size_t mid = (ranked.size() + 1) / 2;   // the lead takes the extra hot pair on an odd count
    return rank == 0 ? std::vector<std::pair<int32_t, int32_t>>(ranked.begin(), ranked.begin() + (ptrdiff_t) mid)
                     : std::vector<std::pair<int32_t, int32_t>>(ranked.begin() + (ptrdiff_t) mid, ranked.end());
}

}  // namespace basalt::program

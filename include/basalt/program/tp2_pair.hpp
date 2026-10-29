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

/// True when the card of `rank` (0 the lead, 1 its partner) of a `tp_size`-wide pair holds `expert`.  The split is by
/// expert id, so of a pair of cards exactly one holds any expert and the two tiers come out the same depth.  With
/// `tp_size` 1 the stage's single card holds every expert - the plain layer split - and there is no partner rank.
/// Expert ids are non-negative (a source will not load one otherwise), `rank` names a card of the pair.
inline bool pair_holds(int32_t expert, int tp_size, int rank) {
    return tp_size > 0 && rank >= 0 && rank < tp_size && (int32_t) (expert % tp_size) == rank;
}

/// An expert cache's ranked (layer, expert) pairs, hottest first, restricted to one card of the pair - in the same
/// order, so a card fills its own half of its stage's ranking hottest first.  The two halves are disjoint and cover
/// `ranked`, which is the invariant the resident complement and the RAM copy are built on.
inline std::vector<std::pair<int32_t, int32_t>> pair_ranked(
    const std::vector<std::pair<int32_t, int32_t>>& ranked, int tp_size, int rank) {
    std::vector<std::pair<int32_t, int32_t>> out;
    for (const auto& pr : ranked)
        if (pair_holds(pr.second, tp_size, rank)) out.push_back(pr);
    return out;
}

}  // namespace basalt::program

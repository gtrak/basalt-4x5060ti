// include/basalt/prefill/peer_band.hpp - the RAM band's share, in one place.
//
// The band is the experts of a stage whose blobs the PARTNER card reads for itself, so whose rows it computes.
// Which of a layer's band members those are is decided by rank alone: `peer_band_taken` hands the partner `num` of
// every `kPeerBandDen` of them, counted in id order, so the price, the lead's stream and the row split all pick
// out the same experts (docs/precision/moe-partial-merge.md).
//
// Because the choice is by RANK and not by id, it moves when the band's MEMBERSHIP moves: one member added at a
// low id shifts every rank above it.  Membership is therefore the second half of what a run must hold still, and
// the reason `--exp peer_band_pin` fixes it to a canonical test rather than leaving it to the placement.
//
// Header-only so the CPU parity test grades the functions the engine itself calls, the way `moe_part.hpp` does.
#pragma once

#include <cstdint>

namespace basalt::prefill {

/// The share's denominator, fixed so every card counts the band alike.
constexpr int kPeerBandDen = 8;

/// The `k`-th band member of a layer, counted in id order: the partner reads it at the share `num` when `num` of
/// every `kPeerBandDen` come up.  `num` outside [0, kPeerBandDen] selects none or all, which is what a share of
/// 0 or kPeerBandDen means.
inline bool peer_band_taken(int64_t k, int num) {
    return num > 0 && k >= 0 && k % kPeerBandDen < num;
}

/// How many of a layer's `band` members `peer_band_taken` picks at the share `num`, without walking them: whole
/// turns of the denominator, plus what the last incomplete turn has room for.  The price counts with this and the
/// stream counts with `peer_band_taken`; an agreement between the two is what the test below grades.
inline int64_t peer_band_take(int64_t band, int num) {
    if (band <= 0 || num <= 0) return 0;
    const int64_t whole = (band / kPeerBandDen) * (int64_t) (num < kPeerBandDen ? num : kPeerBandDen);
    const int64_t rest = band % kPeerBandDen;
    return whole + (rest < (int64_t) num ? rest : (int64_t) (num < kPeerBandDen ? num : kPeerBandDen));
}

/// The band's share: the eighths `--exp peer_band_pin` ordered (`pin`, 0 when unset, in which case the share the
/// price model struck stands).  An empty band has no share whatever the pin says, so a stage every expert of
/// which sits on a card still reads as a band that is off.
inline int peer_band_num(int64_t band, int pin, int priced) {
    if (band <= 0) return 0;
    return pin > 0 ? (pin < kPeerBandDen ? pin : kPeerBandDen) : priced;
}

}  // namespace basalt::prefill

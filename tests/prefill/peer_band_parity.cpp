// tests/prefill/peer_band_parity.cpp - the RAM band's share, on the CPU (no device, no CUDA).
//
// The band is the experts of a stage whose blobs the PARTNER card reads for itself, so whose rows it computes.
// Two functions decide which ones they are, and the engine calls both: `peer_band_taken` walks the band by rank
// (the price, the lead's stream and the row split each keep a counter of it), and `peer_band_take` says how many
// that walk selects without walking it.  They are two ways of counting one thing, so an agreement between them
// IS the invariant that keeps a blob the lead left out of its stream from being an expert nobody computes.
//
// `--exp peer_band_pin=N` fixes the share to N of every `kPeerBandDen` instead of leaving it to the price model,
// which follows the lead card's resident fill and so differs between two boots of one build
// (docs/precision/moe-partial-merge.md).  This grades what the pin does to the share:
//
//   1. the closed form and the rank walk agree, for every share and every band size up to a layer's experts;
//   2. pinned, the share IS the pin: `peer_band_num` returns N, and the count is the walk's own at that share;
//   3. unset (`pin` 0), nothing moves: the share is the figure the caller handed in, so a boot without the flag
//      counts exactly what it counted before the flag existed;
//   4. the share is monotone in N and ends where it must: none at 0, the whole band at `kPeerBandDen`;
//   5. a pinned decision is reproducible: the same band and the same N pick the same experts twice over;
//   6. over a whole stage, the count the boot prices its share with is the count the rank walk hands over, and the
//      share it prints is the pin.
//
// The arithmetic under all of this is `basalt/prefill/peer_band.hpp`, which `src/prefill/prefill.cpp` calls, so
// the test measures the engine's counters rather than transcriptions of them.
#include "basalt/prefill/peer_band.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using basalt::prefill::kPeerBandDen;
using basalt::prefill::peer_band_num;
using basalt::prefill::peer_band_take;
using basalt::prefill::peer_band_taken;

/// What the lead's stream does: walk the band in id order and count the members the share hands to the partner.
/// The engine keeps this counter in three places (`peer_ram_share`, `peer_link`, the streamed walk); all three
/// must land on the figure `peer_band_take` gives closed-form, which is check 1.
int64_t walk_take(int64_t band, int num) {
    int64_t n = 0;
    for (int64_t k = 0; k < band; ++k)
        if (peer_band_taken(k, num)) ++n;
    return n;
}

/// The experts a layer hands over, as a set of ranks: what the price, the stream and the row split must agree on.
std::vector<int32_t> selected(int64_t band, int num) {
    std::vector<int32_t> out;
    for (int64_t k = 0; k < band; ++k)
        if (peer_band_taken(k, num)) out.push_back((int32_t) k);
    return out;
}

}  // namespace

int main() {
    int fails = 0;

    // 1. the closed form against the walk, every share over every band size a layer's worth of experts can hold
    {
        int bad = 0;
        for (int num = -1; num <= kPeerBandDen + 1; ++num) {
            for (int64_t band = 0; band <= 512; ++band) {
                const int64_t closed = peer_band_take(band, num), walked = walk_take(band, num);
                if (closed != walked) {
                    ++bad;
                    if (bad < 8)
                        std::printf("   FAIL: band %lld at share %d/%d: closed %lld, walked %lld\n",
                                    (long long) band, num, (int) kPeerBandDen, (long long) closed, (long long) walked);
                }
            }
        }
        std::printf("1. take(band, num) == the rank walk over %d shares x 513 band sizes: failures=%d\n",
                    kPeerBandDen + 3, bad);
        if (bad) ++fails;
    }

    // 2. pinned: the share IS the pin, whatever the model would have struck, and the count is the walk's own
    {
        int bad = 0;
        std::mt19937 rng(20260214u);
        for (int trial = 0; trial < 4000; ++trial) {
            const int64_t band = (int64_t) (rng() % 600u) + 1;   // a layer's band, never empty here
            const int pin = (int) (rng() % (unsigned) kPeerBandDen) + 1;
            const int priced = (int) (rng() % (unsigned) (kPeerBandDen + 1));   // what the model struck, unrelated
            const int num = peer_band_num(band, pin, priced);
            if (num != pin) {
                ++bad;
                if (bad < 8)
                    std::printf("   FAIL: pinned to %d of %d over a band of %lld, the share is %d (priced was %d)\n",
                                pin, (int) kPeerBandDen, (long long) band, num, priced);
            }
            const int64_t take = peer_band_take(band, num);
            if (take != walk_take(band, pin) || take != (int64_t) selected(band, pin).size()) {
                ++bad;
                if (bad < 8)
                    std::printf("   FAIL: band %lld pinned to %d: take %lld, walked %lld, listed %zu\n",
                                (long long) band, pin, (long long) take, (long long) walk_take(band, pin),
                                selected(band, pin).size());
            }
        }
        std::printf("2. pinned: num == N and take == the walk's count, over 4000 random (band, N, priced): failures=%d\n",
                    bad);
        if (bad) ++fails;
    }

    // 3. unset: the share is the figure the caller handed in, so a boot without the flag is the boot it always was
    {
        int bad = 0;
        std::mt19937 rng(31415926u);
        for (int trial = 0; trial < 4000; ++trial) {
            const int64_t band = (int64_t) (rng() % 700u);   // including the empty band
            const int priced = (int) (rng() % (unsigned) (kPeerBandDen + 1));
            const int num = peer_band_num(band, 0, priced);
            const int want = band > 0 ? priced : 0;   // an empty band has no share to name
            if (num != want) {
                ++bad;
                if (bad < 8)
                    std::printf("   FAIL: unpinned over a band of %lld: share %d, expected %d\n",
                                (long long) band, num, want);
            }
        }
        // and the arithmetic that named the share before the pin existed still names it: eighths of the ratio the
        // counts strike, clamped to [1, kPeerBandDen] the moment there is a band and a take to divide
        for (int64_t take = 1; take <= 4096; ++take) {
            for (int64_t band = 1; band <= 512; ++band) {
                const int priced = (int) std::min<int64_t>(
                    kPeerBandDen, std::max<int64_t>(1, (take * kPeerBandDen + band / 2) / band));
                if (peer_band_num(band, 0, priced) != priced) {
                    ++bad;
                    break;
                }
            }
        }
        std::printf("3. unpinned: the share is the priced figure, 4000 random (band, priced) plus every (take, band) "
                    "under 4096 x 512: failures=%d\n",
                    bad);
        if (bad) ++fails;
    }

    // 4. the share's ends and its order: none at 0, the whole band at kPeerBandDen, never a step backwards
    {
        int bad = 0;
        if (peer_band_take(37, 0) != 0 || peer_band_take(0, kPeerBandDen) != 0) {
            ++bad;
            std::printf("   FAIL: an empty share or an empty band is not empty\n");
        }
        for (int64_t band = 0; band <= 512; ++band) {
            if (peer_band_take(band, kPeerBandDen) != band) {
                ++bad;
                if (bad < 8)
                    std::printf("   FAIL: band %lld at %d/%d is %lld, not whole\n", (long long) band,
                                (int) kPeerBandDen, (int) kPeerBandDen,
                                (long long) peer_band_take(band, kPeerBandDen));
            }
            int64_t prev = 0;
            for (int num = 0; num <= kPeerBandDen; ++num) {
                const int64_t t = peer_band_take(band, num);
                if (t < prev || t > band) {
                    ++bad;
                    if (bad < 8)
                        std::printf("   FAIL: band %lld at %d/%d is %lld, outside [0, band] or below the share "
                                    "before it\n",
                                    (long long) band, num, (int) kPeerBandDen, (long long) t);
                }
                prev = t;
            }
        }
        std::printf("4. the share's ends and monotonicity over 513 band sizes: failures=%d\n", bad);
        if (bad) ++fails;
    }

    // 5. reproducibility, which is the reason the pin exists: two "boots" that count the same canonical band at
    // the same N must pick the same experts, so the gate's arms differ by nothing the counters can move
    {
        int bad = 0;
        std::mt19937 rng(161803398u);
        for (int trial = 0; trial < 2000; ++trial) {
            const int64_t band = (int64_t) (rng() % 600u) + 1;
            const int pin = (int) (rng() % (unsigned) (kPeerBandDen + 1));
            if (selected(band, pin) != selected(band, pin)) {
                ++bad;
                if (bad < 8) std::printf("   FAIL: band %lld at %d/%d picked different experts twice\n",
                                         (long long) band, pin, (int) kPeerBandDen);
            }
        }
        // and the ranks a share picks are spread over the band, not stacked at one end: a pinned gate arm and a
        // pinned reference arm see the same spread whatever the band's size
        for (int num = 1; num < kPeerBandDen; ++num) {
            const std::vector<int32_t> s = selected(1000, num);
            if ((int64_t) s.size() != peer_band_take(1000, num)) { ++bad; continue; }
            if (s.empty() || s.front() != 0 || s.back() < 1000 - kPeerBandDen) {
                ++bad;
                std::printf("   FAIL: share %d/%d of a band of 1000 is not spread over it (first %d, last %d)\n",
                            num, (int) kPeerBandDen, s.front(), s.back());
            }
        }
        std::printf("5. a pinned selection is reproducible and spread over the band: failures=%d\n", bad);
        if (bad) ++fails;
    }

    // 6. a stage: the count the boot prices its share with is the count the three rank walkers hand over. The
    // per-layer counters in `peer_ram_share`, `peer_link` and the streamed walk each keep their own `k`, so the
    // figure the boot line prints (`take of band experts ... over N layers`) has to come out of the same relation
    {
        int bad = 0;
        std::mt19937 rng(271828182u);
        for (int trial = 0; trial < 500; ++trial) {
            const int pin = (int) (rng() % (unsigned) (kPeerBandDen + 1));
            int64_t band = 0, take = 0;
            for (int layer = 0; layer < 24; ++layer) {
                const int64_t b = (int64_t) (rng() % 400u);   // a layer's band, possibly empty
                band += b;
                take += peer_band_take(b, pin);
                if (peer_band_take(b, pin) != (int64_t) selected(b, pin).size()) {
                    ++bad;
                    if (bad < 8)
                        std::printf("   FAIL: layer %d of a band of %lld at %d/%d: %lld priced, %zu handed over\n",
                                    layer, (long long) b, pin, (int) kPeerBandDen,
                                    (long long) peer_band_take(b, pin), selected(b, pin).size());
                }
            }
            const int num = peer_band_num(band, pin, 7);   // 7: a share the model might have struck
            const int want = pin > 0 ? std::min(pin, kPeerBandDen) : 7;
            if (num != want) {
                ++bad;
                if (bad < 8)
                    std::printf("   FAIL: a stage of %lld band experts at pin %d gives share %d, expected %d\n",
                                (long long) band, pin, num, want);
            }
            if (pin > 0 && band > 0 && take > band) { ++bad; }   // never more than the band
        }
        std::printf("6. a 24-layer stage: the share is the pin and the count is what the rank walk hands over, "
                    "500 random stages: failures=%d\n",
                    bad);
        if (bad) ++fails;
    }

    std::printf("%s\n", fails == 0 ? "peer_band_parity: OK" : "peer_band_parity: FAILED");
    return fails == 0 ? 0 : 1;
}

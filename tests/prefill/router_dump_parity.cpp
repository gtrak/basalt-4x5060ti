// tests/prefill/router_dump_parity.cpp - the router probe's fingerprints and its selection replay, on the CPU.
//
// `--exp router_dump=N` copies a layer's router products off the card - the activations the projection read, its
// logits, the routing and its weights - so that two boots of one build can be told apart by the BYTES the router
// saw rather than by the ids it emitted (docs/precision/peer-band-pin.md covers the dump this one completes).
// The probe is only worth a boot pair if four things hold, and this grades all four against the functions
// `src/prefill/prefill.cpp` itself calls:
//
//   1. a word says what it claims: one logit's bits moved and that token's word moved, one token moved and the
//      block's word moved - a probe blind to a change is worse than no probe, because it is believed;
//   2. a word is stable: the same bytes twice over give the same words, so a pair of boots that agree in fact
//      print a pair that looks like it;
//   3. the replayed top-K is the selection `route_kernel` makes - descending logit, the LOWER id winning a tie -
//      graded against a sort, over random rows with deliberate ties and non-finite entries;
//   4. the boundary's gap is the gap: `gap == 0` exactly when the K-th and (K+1)-th largest logits are the same
//      float, which is the one case where the tie-break alone decides an expert seat.
//
// Fingerprints are FNV-1a, so a collision is possible in principle and is not what this grades: sensitivity and
// stability are what the diff between two boots relies on. Random rows, the way every kernel gets graded.
#include "basalt/prefill/router_dump.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using basalt::prefill::kFnvOffset;
using basalt::prefill::kRouterMaxK;
using basalt::prefill::RouterDump;
using basalt::prefill::RouterProbe;
using basalt::prefill::router_dump_collect;
using basalt::prefill::router_select;

int fails = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++fails;
    if (fails <= 12) std::printf("   FAIL: %s\n", what.c_str());
}

/// A row of logits at the model's width. Deliberately coarse: whole levels, a floor of exact ties, and - rarely -
/// a non-finite entry, because a probe graded only on smooth noise never meets the case it exists for.
std::vector<float> row(int64_t n_expert, uint32_t seed, bool wild) {
    std::mt19937 rng(seed);
    std::vector<float> r((size_t) n_expert);
    for (int64_t x = 0; x < n_expert; ++x) {
        const uint32_t u = rng();
        if (u % 13u == 0u) r[(size_t) x] = 0.0f;                              // a floor of exact ties
        else if (u % 17u == 0u) r[(size_t) x] = 2.0f;                          // ties at a level
        else if (u % 29u == 0u) r[(size_t) x] = -1.5f;                         // ties below zero
        else if (u % 7u == 0u) r[(size_t) x] = 1e4f;                           // a plateau ABOVE the distinct
        // levels, so the K/K+1 boundary lands INSIDE a tie group - which is where the lower-id rule is the only
        // thing deciding the seat, and where a replay that got it wrong would otherwise never be caught
        else r[(size_t) x] = (float) ((int32_t) (u % 4096u) - 2048) / 16.0f;   // a few hundred distinct levels
    }
    if (wild && n_expert > 3) {
        r[(size_t) (rng() % (uint32_t) n_expert)] = std::numeric_limits<float>::quiet_NaN();
        r[(size_t) (rng() % (uint32_t) n_expert)] = std::numeric_limits<float>::infinity();
        r[(size_t) (rng() % (uint32_t) n_expert)] = -std::numeric_limits<float>::infinity();
    }
    return r;
}

/// The top-K a reader would write with no cleverness at all: sort the ids by (value descending, id ascending),
/// skipping the non-finite. This is the oracle the replay is graded against, spelled differently on purpose.
void reference(const std::vector<float>& r, int64_t k, std::vector<int32_t>& pick, double& gap) {
    std::vector<int32_t> order;
    for (size_t i = 0; i < r.size(); ++i)
        if (r[i] == r[i]) order.push_back((int32_t) i);   // a nan never seats, as in the kernel
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return r[a] != r[b] ? r[a] > r[b] : a < b;
    });
    pick.assign((size_t) k, -1);
    for (int64_t i = 0; i < k && i < (int64_t) order.size(); ++i) pick[(size_t) i] = order[(size_t) i];
    gap = (int64_t) order.size() > k ? (double) r[(size_t) order[(size_t) k - 1]] - (double) r[(size_t) order[(size_t) k]]
                                     : std::nan("");
}

/// A chunk's routing at the model's geometry: K of `n_expert` per token, as the card wrote them.
std::vector<int32_t> routing(int64_t T, int64_t K, int32_t n_expert, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> ids((size_t) T * K);
    for (int64_t i = 0; i < T * K; ++i) ids[(size_t) i] = (int32_t) (rng() % (uint32_t) n_expert);
    return ids;
}

RouterDump make(int64_t T, int64_t K, int64_t n_expert, int64_t n_embd) {
    RouterDump d;
    d.layer = 20;
    d.p0 = 0;
    d.T = T;
    d.K = K;
    d.n_expert = n_expert;
    d.n_embd = n_embd;
    return d;
}

/// A whole block at the model's geometry: logits whose rows are real rows, activations, ids and weights.
struct Block {
    std::vector<float> lg, w;
    std::vector<uint16_t> in;
    std::vector<int32_t> ids;
};

Block block(int64_t T, int64_t K, int64_t n_expert, int64_t n_embd, uint32_t seed) {
    Block b;
    b.lg.reserve((size_t) T * n_expert);
    for (int64_t t = 0; t < T; ++t) {
        const std::vector<float> r = row(n_expert, seed + (uint32_t) t * 7919u, t % 11 == 0);
        b.lg.insert(b.lg.end(), r.begin(), r.end());
    }
    b.in.resize((size_t) T * n_embd);
    b.ids = routing(T, K, (int32_t) n_expert, seed);
    b.w.resize((size_t) T * K);
    std::mt19937 rng(seed ^ 0x5u);
    for (auto& v : b.in) v = (uint16_t) rng();
    for (auto& v : b.w) v = (float) ((int32_t) (rng() % 1000u)) / 1000.0f;
    return b;
}

}  // namespace

int main() {
    // 1. the selection replay, against a sort: rows with ties, non-finite entries, and widths from one expert up
    {
        std::mt19937 rng(20251101u);
        int mismatches = 0;
        for (int trial = 0; trial < 3000; ++trial) {
            const int64_t ne = 1 + (int64_t) (rng() % 700u);
            const int64_t k = 1 + (int64_t) (rng() % 12u);
            const std::vector<float> r = row(ne, rng(), trial % 5 == 0);
            std::vector<int32_t> got((size_t) k, -1);
            double gap = 0.0;
            router_select(r.data(), ne, k, got.data(), &gap);
            std::vector<int32_t> want;
            double wg = 0.0;
            reference(r, k, want, wg);
            if (got != want) {
                ++mismatches;
                if (mismatches <= 3)
                    std::printf("   picks differ: ne %lld k %lld trial %d\n", (long long) ne, (long long) k, trial);
            }
            const bool both_nan = std::isnan(gap) && std::isnan(wg);
            if (!both_nan && gap != wg) {
                ++mismatches;
                if (mismatches <= 3)
                    std::printf("   gaps differ: ne %lld k %lld %.9g vs %.9g\n", (long long) ne, (long long) k, gap, wg);
            }
        }
        check(mismatches == 0, "the replayed top-K or its boundary gap left the sort");
        // the tie-break, spelled out: two equal logits at the K/K+1 boundary must seat the LOWER id
        {
            std::vector<float> r(512, -100.0f);
            r[200] = 1.0f;
            r[201] = 1.0f;
            r[100] = 0.5f;
            r[101] = 0.5f;
            std::vector<int32_t> pick(10, -1);
            double gap = 0.0;
            router_select(r.data(), 512, 10, pick.data(), &gap);
            check(pick[0] == 200 && pick[1] == 201, "equal top logits did not seat in id order");
            check(gap == 0.0, "a tie at the K/K+1 boundary did not read as a tie");
            // and the count: 512 - 4 distinct values above the filler means the boundary sits inside the filler,
            // where the tie-break decides every seat - the shape the near-tie hunt is about
            int in_pick = 0;
            for (int32_t e : pick)
                if (e >= 0) ++in_pick;
            check(in_pick == 10, "a row of 512 logits left a seat unfilled");
        }
        // a row that cannot fill K seats says so with -1 ids and a NaN gap, rather than inventing experts.  Only
        // the NON-FINITE entries are unseatable: an -inf logit is a rankable value (the kernel's softmax puts it
        // last, not nowhere), so a row of them still fills every seat.
        {
            std::vector<float> r(512, std::numeric_limits<float>::quiet_NaN());
            r[3] = 1.0f;
            std::vector<int32_t> pick(10, -1);
            double gap = 1.0;
            router_select(r.data(), 512, 10, pick.data(), &gap);
            check(pick[0] == 3, "the only finite logit did not take the first seat");
            for (int64_t i = 1; i < 10; ++i) check(pick[i] == -1, "a seat was invented for a row of nan");
            check(std::isnan(gap), "a row with no (K+1)-th logit reported a gap anyway");
            // and the rankable side of the same line: -inf fills the row out, lowest ids first among equals
            std::vector<float> n512(512, -INFINITY);
            n512[3] = 1.0f;
            router_select(n512.data(), 512, 10, pick.data(), &gap);
            check(pick[0] == 3, "a lone finite logit lost to a row of -inf");
            for (int64_t i = 1; i < 10; ++i) check(pick[i] >= 0, "an -inf logit refused its seat");
        }
        // a K past the cap is refused as an absence, not answered
        {
            const std::vector<float> r = row(512, 5u, false);
            std::vector<int32_t> pick((size_t) kRouterMaxK + 1, -1);
            double gap = 1.0;
            router_select(r.data(), 512, kRouterMaxK + 1, pick.data(), &gap);
            check(std::isnan(gap), "a K past the cap reported a gap");
            bool all_absent = true;
            for (int32_t e : pick)
                if (e != -1) all_absent = false;
            check(all_absent, "a K past the cap invented seats");
        }
        std::printf("1. the selection replay: the sort's picks and gap, over rows with ties and non-finite entries\n");
    }

    // 1b. a row the probe never got is still asked for by name: no seats filled, no boundary claimed, and no read
    //     of nowhere - the fused layer's absence reaches the replay as a null pointer
    {
        std::vector<int32_t> pick(10, 7);
        double gap = 1.0;
        router_select(nullptr, 512, 10, pick.data(), &gap);
        bool all_none = true;
        for (int32_t p : pick) all_none = all_none && p == -1;
        check(all_none, "a row that is not there filled seats");
        check(std::isnan(gap), "a row that is not there claimed a boundary");
        std::fill(pick.begin(), pick.end(), 7);
        gap = 1.0;
        router_select(nullptr, 0, 0, pick.data(), &gap);
        check(pick[0] == 7 && std::isnan(gap), "an empty geometry changed what it was handed");
        // a K the replay cannot hold says so by claiming nothing at all - every seat empty, no boundary - rather
        // than picking over part of the row and printing a partial answer as an answer
        std::vector<int32_t> wide((size_t) (kRouterMaxK + 8), 5);
        gap = 1.0;
        router_select(row(512, 3u, false).data(), 512, kRouterMaxK + 8, wide.data(), &gap);
        bool none_claimed = true;
        for (int32_t p : wide) none_claimed = none_claimed && p == -1;
        check(none_claimed && std::isnan(gap), "a K past the replay's reach made a claim anyway");
        std::printf("1b. the replay on a row that is not there, and on a K it cannot hold\n");
    }

    // 2. the fingerprints are sensitive: one logit's bits move and the token's word moves; one token moves and the
    //    block's word moves - both directions of the claim the two boots are diffed on
    {
        const int64_t T = 37, K = 10, ne = 512, n = 2560;
        Block b = block(T, K, ne, n, 11u);
        RouterDump a = make(T, K, ne, n);
        router_dump_collect(a, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        RouterDump same = make(T, K, ne, n);
        router_dump_collect(same, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        check(a.lg == same.lg && a.in == same.in && a.ids == same.ids && a.w == same.w,
              "one block fingerprinted twice gave two words");
        check(a.lg_t == same.lg_t && a.gap_t == same.gap_t, "a token's word was not stable");

        // one bit of one logit of one token: that token's word, and the block's, must both move
        int blind = 0;
        for (int trial = 0; trial < 40; ++trial) {
            Block c = b;
            const int64_t t = (int64_t) ((trial * 31 + 7) % T);
            const int64_t x = (int64_t) ((trial * 131 + 3) % ne);
            float* p = &c.lg[(size_t) t * ne + x];
            const float was = *p;
            *p = was == 0.0f ? 1.0f : was * (1.0f + 1.19e-7f);   // one fp32 ulp, the smallest claim possible
            if (*p == was) *p = was + 1.0f;                        // a zero times anything is still zero
            RouterDump d = make(T, K, ne, n);
            router_dump_collect(d, c.lg.data(), c.in.data(), c.ids.data(), c.w.data());
            if (d.lg_t[(size_t) t] == a.lg_t[(size_t) t]) ++blind;
            if (d.lg == a.lg) ++blind;
            for (int64_t o = 0; o < T; ++o)
                if (o != t && d.lg_t[(size_t) o] != a.lg_t[(size_t) o]) ++blind;   // its neighbours untouched
        }
        check(blind == 0, "a logit's last bit moved and a fingerprint did not notice");

        // the routing and the activations likewise, and only for the token touched
        {
            Block c = b;
            c.ids[(size_t) 9 * K + 4] = (int32_t) ((c.ids[(size_t) 9 * K + 4] + 1) % ne);
            RouterDump d = make(T, K, ne, n);
            router_dump_collect(d, c.lg.data(), c.in.data(), c.ids.data(), c.w.data());
            check(d.ids_t[9] != a.ids_t[9], "a routed id moved and its word did not");
            check(d.ids != a.ids, "a routed id moved and the block's word did not");
            check(d.ids_t[8] == a.ids_t[8] && d.ids_t[10] == a.ids_t[10], "another token's routing word moved");
        }
        {
            Block c = b;
            c.in[(size_t) 3 * n + 1234] = (uint16_t) (c.in[(size_t) 3 * n + 1234] ^ 0x1u);
            RouterDump d = make(T, K, ne, n);
            router_dump_collect(d, c.lg.data(), c.in.data(), c.ids.data(), c.w.data());
            check(d.in_t[3] != a.in_t[3], "an activation bit moved and the input's word did not");
            check(d.in != a.in, "an activation bit moved and the block's input word did not");
            check(d.lg_t[3] == a.lg_t[3], "the logits' word moved with the input's");
        }
        // the order the tokens are folded in is part of the claim, not an accident of the walk
        {
            Block c = b;
            for (int64_t x = 0; x < ne; ++x)
                std::swap(c.lg[(size_t) 4 * ne + x], c.lg[(size_t) 5 * ne + x]);
            RouterDump d = make(T, K, ne, n);
            router_dump_collect(d, c.lg.data(), c.in.data(), c.ids.data(), c.w.data());
            check(d.lg != a.lg, "two tokens traded rows and the block's word stood");
        }
        // and every token feeds every block word: drop nothing by position, in either direction.  A fold that
        // skipped a token by its index would still answer correctly for the tokens a test happens to perturb,
        // which is exactly the kind of probe that passes and lies.
        int deaf = 0;
        for (int64_t t = 0; t < T; ++t) {
            Block c = b;
            c.lg[(size_t) t * ne + ((t * 17 + 5) % ne)] += 1.0f;
            c.in[(size_t) t * n + ((t * 7 + 11) % n)] = (uint16_t) (c.in[(size_t) t * n + ((t * 7 + 11) % n)] + 3u);
            c.ids[(size_t) t * K + (t % K)] = (int32_t) ((c.ids[(size_t) t * K + (t % K)] + 7) % ne);
            c.w[(size_t) t * K + ((t + 3) % K)] += 0.25f;
            RouterDump d = make(T, K, ne, n);
            router_dump_collect(d, c.lg.data(), c.in.data(), c.ids.data(), c.w.data());
            if (d.lg == a.lg) ++deaf;
            if (d.in == a.in) ++deaf;
            if (d.ids == a.ids) ++deaf;
            if (d.w == a.w) ++deaf;
            if (d.lg_t[(size_t) t] == a.lg_t[(size_t) t]) ++deaf;
            if (d.in_t[(size_t) t] == a.in_t[(size_t) t]) ++deaf;
            if (d.ids_t[(size_t) t] == a.ids_t[(size_t) t]) ++deaf;
            if (d.w_t[(size_t) t] == a.w_t[(size_t) t]) ++deaf;
            for (int64_t o = 0; o < T; ++o) {
                if (o == t) continue;
                if (d.lg_t[(size_t) o] != a.lg_t[(size_t) o]) ++deaf;
                if (d.in_t[(size_t) o] != a.in_t[(size_t) o]) ++deaf;
                if (d.ids_t[(size_t) o] != a.ids_t[(size_t) o]) ++deaf;
                if (d.w_t[(size_t) o] != a.w_t[(size_t) o]) ++deaf;
            }
        }
        check(deaf == 0, "a token's bytes did not reach its block's word, or reached another token's");
        std::printf("2. the fingerprints: one bit moves its token and its block, and nothing else\n");
    }

    // 3. the boundary's facts: `ties` counts the rows whose K/K+1 logits are one float, `gap_min` the tightest
    //    row, and a non-finite entry is said rather than silently unseated
    {
        const int64_t ne = 512, K = 10, n = 64;
        Block b = block(8, K, ne, n, 3u);
        // force exactly one tied boundary and one clean row: a rising ramp has distinct logits, and its K/K+1
        // boundary sits between the 10th and 11th largest, which is where a tie has to be put to be one
        for (int64_t t = 0; t < 8; ++t)
            for (int64_t x = 0; x < ne; ++x) b.lg[(size_t) t * ne + x] = -100.0f + (float) x;
        RouterDump clean = make(8, K, ne, n);
        router_dump_collect(clean, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        check(clean.ties == 0, "a row of distinct logits reported a boundary tie");
        check(clean.gap_tok == 0 && clean.gap_min == 1.0, "the clean block had no boundary to report");
        check(clean.bad == 0, "a block of finite logits reported a non-finite one");

        b.lg[(size_t) 5 * ne + 501] = b.lg[(size_t) 5 * ne + 502];   // an exact tie right AT the K/K+1 boundary
        RouterDump tied = make(8, K, ne, n);
        router_dump_collect(tied, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        check(tied.ties == 1, "an exact boundary tie went uncounted");
        check(tied.gap_tok == 5 && tied.gap_min == 0.0, "the tightest boundary was not the tied row");
        check(tied.gap_t[5] == 0.0 && tied.gap_t[4] > 0.0, "the per-token gap disagreed with the block's tally");

        b.lg[(size_t) 2 * ne + 7] = std::numeric_limits<float>::quiet_NaN();
        RouterDump nanned = make(8, K, ne, n);
        router_dump_collect(nanned, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        check(nanned.bad == 1, "a nan logit went unreported");
        check(nanned.bad_t[2] == 1 && nanned.bad_t[1] == 0, "the nan was blamed on the wrong token");

        // a whole row of nan: no seats, so no gap, and the token is named as bad rather than innocent
        Block z = block(1, K, ne, n, 9u);
        for (auto& f : z.lg) f = std::numeric_limits<float>::quiet_NaN();
        RouterDump all = make(1, K, ne, n);
        router_dump_collect(all, z.lg.data(), z.in.data(), z.ids.data(), z.w.data());
        check(all.bad == 1 && all.bad_t[0] == 1, "a row of nan read as a good row");
        check(std::isnan(all.gap_t[0]), "a row of nan reported a boundary gap");
        std::printf("3. the boundary: ties counted, the tightest row named, a non-finite logit said\n");
    }

    // 4. the selection replay against the card's own ids: agreement is silence, one flipped id opens at its token
    {
        const int64_t T = 24, K = 10, ne = 512, n = 128;
        // build ids that ARE the selection over the logits, so agreement is the expected answer
        Block b = block(T, K, ne, n, 21u);
        for (int64_t t = 0; t < T; ++t) {
            std::vector<int32_t> pick((size_t) K, -1);
            double gap = 0.0;
            router_select(&b.lg[(size_t) t * ne], ne, K, pick.data(), &gap);
            for (int64_t i = 0; i < K; ++i) b.ids[(size_t) t * K + i] = pick[i];
        }
        RouterDump d = make(T, K, ne, n);
        router_dump_collect(d, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        check(d.disagree == -1, "the card's own routing did not match the replay of it");
        check(d.sel_cmp == T, "the replay was not made for every token");
        b.ids[(size_t) 11 * K + 9] = (int32_t) ((b.ids[(size_t) 11 * K + 9] + 3) % ne);
        RouterDump e = make(T, K, ne, n);
        router_dump_collect(e, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        check(e.disagree == 11, "one flipped id did not open the replay at its token");
        // a second token, further along: the FIRST disagreement is the one worth walking to, so a record that kept
        // the last would send the hunt to the wrong token while still printing a number that looks like a finding
        b.ids[(size_t) 19 * K] = (int32_t) ((b.ids[(size_t) 19 * K] + 7) % ne);
        RouterDump e2 = make(T, K, ne, n);
        router_dump_collect(e2, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        check(e2.disagree == 11, "with two tokens out, the replay named the later one instead of the first");
        check(e2.sel_cmp == T, "the replay stopped short of the chunk");
        std::printf("4. the replay against the card's ids: agreement is silence, the first bad token is the one named\n");
    }

    // 5. a path that holds a block on the card (the fused path's routing) fingerprints as absent, not as zeros,
    //    and an empty chunk reads as nothing rather than as agreement
    {
        const int64_t T = 6, K = 10, ne = 512, n = 64;
        Block b = block(T, K, ne, n, 33u);
        RouterDump d = make(T, K, ne, n);
        router_dump_collect(d, b.lg.data(), b.in.data(), nullptr, nullptr);
        check(d.ids == kFnvOffset && d.w == kFnvOffset, "a routing the path never held read as a real one");
        for (int64_t t = 0; t < T; ++t)
            check(d.ids_t[(size_t) t] == kFnvOffset && d.w_t[(size_t) t] == kFnvOffset,
                  "a token's absent routing read as a real one");
        check(d.disagree == -1 && d.sel_cmp == 0, "the replay claimed a comparison it could not make");
        check(d.lg != kFnvOffset, "the block that WAS given read as absent");

        // the other way round: a layer whose LOGITS never came back is the case that makes the whole probe read
        // right, because it is what a fused layer looks like. Absent must be the offset basis, not a fingerprint of
        // the offset basis - a second boot would otherwise match a word that stands for "nothing was here"
        RouterDump nol = make(T, K, ne, n);
        router_dump_collect(nol, nullptr, nullptr, b.ids.data(), b.w.data());
        check(nol.lg == kFnvOffset && nol.in == kFnvOffset,
              "a routing read back without its logits or input reported a fingerprint");
        for (int64_t t = 0; t < T; ++t)
            check(nol.lg_t[(size_t) t] == kFnvOffset && nol.in_t[(size_t) t] == kFnvOffset,
                  "a token without logits or input reported a word");
        check(nol.ties == 0 && nol.bad == 0 && nol.gap_tok == -1,
              "a layer with no logits still reported a boundary");
        check(nol.disagree == -1 && nol.sel_cmp == 0, "the replay ran without logits to replay");
        check(nol.ids != kFnvOffset && nol.w != kFnvOffset,
              "the routing that WAS given read as absent beside a missing logits block");

        RouterDump empty_ = make(0, K, ne, n);
        router_dump_collect(empty_, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        check(empty_.lg == kFnvOffset && empty_.in == kFnvOffset, "an empty chunk reported a fingerprint");
        check(empty_.ties == 0 && empty_.bad == 0 && empty_.gap_tok == -1, "an empty chunk reported boundary facts");
        check(empty_.lg_t.empty() && empty_.gap_t.empty(), "an empty chunk carried token words");
        std::printf("5. a block the path never held reads absent; an empty chunk reads nothing\n");
    }

    // 6. the keys: which (layer, chunk) gets probed. Layer 0 must be reachable, which is why off is negative
    {
        RouterProbe off;
        check(!off.armed() && !off.takes(0, 0), "an unarmed probe took a layer");
        RouterProbe l0;
        l0.layer = 0;
        check(l0.armed() && l0.takes(0, 7), "layer 0 could not be named");
        check(!l0.takes(1, 7), "a probe named for layer 0 took another layer");
        RouterProbe one;
        one.layer = 20;
        one.p0 = 8192;
        check(one.takes(20, 8192), "the named layer of the named chunk was refused");
        check(!one.takes(20, 0), "a chunk beside the named one was probed");
        check(!one.takes(19, 8192), "a layer beside the named one was probed");
        RouterProbe allp;
        allp.all = true;
        allp.p0 = 4096;
        check(allp.armed() && allp.takes(0, 4096) && allp.takes(47, 4096), "the sweep did not take every layer");
        check(!allp.takes(3, 0), "the sweep left its named chunk");
        RouterProbe stray;
        stray.layer = -3;
        check(!stray.armed() && !stray.takes(0, 0), "a negative layer armed the probe");

        // the KEYS themselves, through `router_probe()`: a probe that silently never arms, or arms on every layer
        // of a hot boot because a default read the wrong way, is the failure this hunt has already paid for twice
        auto& exp = basalt::core::diagnostics().exp;
        exp.clear();
        RouterProbe none = basalt::prefill::router_probe();
        check(!none.armed() && !none.takes(0, 0) && !none.takes(20, 0), "no key at all still probed a layer");
        exp["router_dump"] = 0.0;   // layer 0 is reachable: the off value is negative, not zero
        RouterProbe k0 = basalt::prefill::router_probe();
        check(k0.armed() && k0.layer == 0 && k0.takes(0, 0) && !k0.takes(1, 0), "layer 0 could not be named by key");
        exp.clear();
        exp["router_dump"] = -1.0;   // the documented off value, and any other negative, means off
        exp["router_dump_p0"] = 4096.0;
        RouterProbe neg = basalt::prefill::router_probe();
        check(!neg.armed() && !neg.takes(0, 4096), "a negative layer armed the probe");
        exp.clear();
        exp["router_dump"] = 20.0;
        exp["router_dump_p0"] = 8192.0;
        RouterProbe k20 = basalt::prefill::router_probe();
        check(k20.armed() && k20.takes(20, 8192) && !k20.takes(20, 0) && !k20.takes(19, 8192),
              "the named layer of the named chunk was not the only one probed by key");
        exp.clear();
        exp["router_dump_all"] = 1.0;
        RouterProbe sweep = basalt::prefill::router_probe();
        check(sweep.armed() && sweep.takes(0, 0) && sweep.takes(47, 0), "the sweep key did not take every layer");
        exp.clear();
        std::printf("6. the keys: layer 0 reachable, the chunk honoured, a sweep every layer, off negative\n");
    }

    // 7. the printed lines: every field the hunt reads is IN the text, the hex is fixed width so two logs diff,
    //    and a word in the line is the word in the record - a field that quietly stopped being printed fails here
    //    rather than after a boot pair has been spent on it
    {
        const int64_t T = 5, K = 10, ne = 512, n = 64;
        Block b = block(T, K, ne, n, 77u);
        RouterDump d = make(T, K, ne, n);
        router_dump_collect(d, b.lg.data(), b.in.data(), b.ids.data(), b.w.data());
        const std::string s = basalt::prefill::router_dump_summary(d);
        const std::string tok = basalt::prefill::router_dump_token(d, 2);
        for (const char* field : {"router dump", "p0=", "T=", "layer=", "K=", "n_expert=", "n_embd=",
                                  "lg=", "in=", "ids=", "w=", "ties=", "bad=", "gap=", "at", "sel", "differ"})
            check(s.find(field) != std::string::npos, std::string("the summary line lost its ") + field);
        for (const char* field : {"router tok", "p0=", "layer=", "t=", "lg=", "in=", "ids=", "w=", "gap=", "bad="})
            check(tok.find(field) != std::string::npos, std::string("the token line lost its ") + field);
        // the words in the text are the words in the record, at 16 hex digits, so a diff sees a move when one moved
        auto hex = [](uint64_t v) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long) v);
            return std::string(buf);
        };
        check(s.find("lg=" + hex(d.lg)) != std::string::npos, "the summary's lg word is not the record's");
        check(s.find("in=" + hex(d.in)) != std::string::npos, "the summary's in word is not the record's");
        check(s.find("ids=" + hex(d.ids)) != std::string::npos, "the summary's ids word is not the record's");
        check(s.find("w=" + hex(d.w)) != std::string::npos, "the summary's w word is not the record's");
        check(tok.find("lg=" + hex(d.lg_t[2])) != std::string::npos, "the token line's lg is not the record's");
        check(tok.find("ids=" + hex(d.ids_t[2])) != std::string::npos, "the token line's ids is not the record's");
        check(tok.find("t=2 ") != std::string::npos, "the token line does not name its token");
        check(tok.find("p0=0 ") != std::string::npos, "the token line does not name its chunk");
        check(tok.find(hex(d.lg_t[2])) != tok.find(hex(d.in_t[2])) || d.lg_t[2] == d.in_t[2],
              "two different words printed in the same place");
        // a token outside the record reads absent rather than reading the previous token's bytes
        const std::string past = basalt::prefill::router_dump_token(d, T + 3);
        check(past.find("t=" + std::to_string(T + 3) + " ") != std::string::npos, "an out-of-range token line lost its index");
        check(past.find("lg=0000000000000000") != std::string::npos, "an out-of-range token line printed stale words");
        std::printf("7. the printed lines: every field present, the text is the record, an out-of-range row reads absent\n");
    }

    std::printf("%s\n", fails == 0 ? "router_dump_parity: OK" : "router_dump_parity: FAILED");
    return fails == 0 ? 0 : 1;
}

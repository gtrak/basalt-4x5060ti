// include/basalt/prefill/router_dump.hpp - the device-side router probe `--exp router_dump` prints.
//
// `chunk_dump` names the layer where two boots of one build part company (its `ids` word), but it carries only what
// the host already held. When the first parting IS the router's decision, that says the two routers disagreed and
// not on what. Three readings remain - the projection's LOGITS differ (so its input or its product moved), the
// logits MATCH while the IDS differ (so one set of numbers gave two routings), or a token's own K/K+1 boundary is
// an exact tie, where the tie-break alone decides the seat - and all three are questions about bytes that live on
// the card. This is the record that reads them back.
//
// It is therefore the one dump that DOES touch the device, and that is both its cost and its limit: the copies have
// to retire before their bytes mean anything, so the probe rides a sync the layer already pays and is confined to
// ONE layer of ONE chunk. A probe that synced every layer would move the schedule it was built to examine, and a
// probe that printed every token of every layer would be read as nothing.
//
// Header-only and free of CUDA, so `tests/prefill/router_dump_parity.cpp` (`ctest -L cpu`) grades the arithmetic
// the engine prints - the same reason `chunk_dump.hpp` and `peer_band.hpp` exist.
#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "basalt/core/diag.hpp"            // the `--exp` keys the probe is read from
#include "basalt/prefill/chunk_dump.hpp"   // the one FNV-1a, so a word here means the same as a word there

namespace basalt::prefill {

/// The widest top-K this record replays its own selection for. `route_kernel` caps K at 64; past that the replay
/// fields read as absent (`disagree` stays -1, `sel_cmp` stays 0) rather than as agreement.
constexpr int64_t kRouterMaxK = 64;

/// The three keys `--exp router_dump` is read from, in the one place the engine reads them.
///
///   * `router_dump=N` names the MODEL layer to probe. Unset or negative is off, which is what lets layer 0 be
///     named - the first layer is a control on every claim the probe makes, so a scheme that read 0 as "off"
///     would put the one layer the hunt needs most out of reach.
///   * `router_dump_all=1` probes every layer of the chunk instead. Every layer pays a sync, so it is the sweep
///     that asks "where does it FIRST part", not the boot that settles one question.
///   * `router_dump_p0=V` confines the probe to the chunk that starts at token V. Chunks of one prompt are read
///     from the same tokens, so two boots name the same chunk; without it a long prompt probes one layer once per
///     chunk, which multiplies the lines and the syncs by nothing but the prompt's length.
struct RouterProbe {
    int64_t layer = -1;    // the model layer named; -1: off
    bool all = false;      // every layer of the chunk instead
    int64_t p0 = -1;       // the chunk named; -1: every chunk

    /// Does this (layer, chunk) get probed?
    bool takes(int64_t l, int64_t chunk_p0) const {
        if (!armed()) return false;
        if (p0 >= 0 && chunk_p0 != p0) return false;
        return all || l == layer;
    }
    /// Was a layer asked for at all?  A named layer outside the stage is worth a log line, so "off" and "armed but
    /// unreachable" have to be distinguishable.  Stated as the two ways the probe can be armed, and only those:
    /// a key that read as a layer number when nobody meant it would put a readback into a hot layer of every boot.
    bool armed() const { return all || layer >= 0; }
};

inline RouterProbe router_probe() {
    RouterProbe r;
    r.layer = (int64_t) core::exp_flag("router_dump", -1.0);
    r.all = core::exp_flag("router_dump_all", 0.0) != 0.0;
    r.p0 = (int64_t) core::exp_flag("router_dump_p0", -1.0);
    if (r.layer < 0) r.layer = -1;   // a negative value and no key at all read the same: off, whatever `all` says
    return r;
}

/// What one probed (chunk, layer) held on the card, as the host read it back. Every word is a fingerprint of
/// bytes, never a quantity: a word that moved says something moved, a word that did not says nothing did. The
/// whole-block words are folds of the per-token ones IN TOKEN ORDER, so the summary line cannot disagree with the
/// token lines about what the block held.
struct RouterDump {
    // the geometry the probe read, so a line names the rows its words cover
    int64_t layer = -1, p0 = -1, T = 0, K = 0, n_expert = 0, n_embd = 0;
    // the block, as the card wrote it: the activations the projection read, its logits, the routing, its weights
    uint64_t lg = kFnvOffset, in = kFnvOffset, ids = kFnvOffset, w = kFnvOffset;
    // the boundary's facts, the near-tie question `chunk_dump` could only infer
    int64_t ties = 0;        // tokens whose K-th and (K+1)-th largest logits are the SAME float
    int64_t bad = 0;         // tokens whose row holds a logit that is not finite
    int64_t gap_tok = -1;    // the token with the tightest K/K+1 boundary of the block
    double gap_min = 0.0;    // ... and how tight it was
    // this record's own top-K over the read-back logits, against the ids the card chose: a disagreement says one
    // set of numbers gave two routings, in ONE boot, with no second boot to compare with. A lead, not a verdict -
    // see `router_select`'s caveat about the softmax.
    int64_t disagree = -1;   // the first such token; -1 when none, or when K is past `kRouterMaxK`
    int64_t sel_cmp = 0;     // tokens the replay could be made for at all
    // one word per token, so a diff between two boots opens at the token that parted, not at the layer
    std::vector<uint64_t> lg_t, in_t, ids_t, w_t;
    std::vector<double> gap_t;    // the K/K+1 margin; NaN when the row has no (K+1)-th finite logit
    std::vector<uint8_t> bad_t;   // the row holds a logit that is not finite
};

/// The K seats as `route_kernel` takes them - descending logit, the LOWER expert id winning an exact tie - read off
/// the row as the card wrote it. A non-finite logit never seats (the kernel turns a nan into -FLT_MAX, so it is no
/// candidate for a real seat), which is why a row can come back with fewer than K seats and why `bad` exists
/// beside this. `gap` is the K-th largest minus the (K+1)-th largest, so `gap == 0` is an exact tie AT THE
/// BOUNDARY: which of the two experts holds the seat is then decided by the tie-break and nothing else. `gap` is
/// NaN when the row holds no (K+1)-th finite logit.
///
/// CAVEAT, and the reason a disagreement is a lead rather than a verdict: the kernel ranks the SOFTMAX
/// probabilities, not the logits. `expf` is monotone but not strictly monotone in FP32, so two logits a few bits
/// apart can give one probability, the kernel then ties, and its lower-id rule picks the expert this logit-space
/// walk would not. That is the shape of a boundary case, which is what the field is for.
inline void router_select(const float* row, int64_t n_expert, int64_t k, int32_t* pick, double* gap) {
    *gap = 0.0 / 0.0;
    for (int64_t i = 0; i < k; ++i) pick[i] = -1;
    if (row == nullptr || n_expert <= 0 || k <= 0 || k > kRouterMaxK) return;
    float v[kRouterMaxK + 1];
    int32_t e[kRouterMaxK + 1];
    for (int64_t i = 0; i <= k; ++i) { v[i] = -FLT_MAX; e[i] = -1; }
    int64_t filled = 0;                       // seats the walk has actually filled, at most k + 1
    for (int64_t x = 0; x < n_expert; ++x) {
        const float f = row[x];
        if (!(f == f)) continue;              // a nan never seats; `bad` says the row held one
        // past a full table, a value earns a seat only by beating the weakest one, or by tying it with a lower id
        if (filled > k && (f < v[k] || (f == v[k] && x >= e[k]))) continue;
        int64_t at = filled < k + 1 ? filled : k;
        while (at > 0 && (f > v[at - 1] || (f == v[at - 1] && x < e[at - 1]))) --at;
        for (int64_t i = (filled < k + 1 ? filled : k); i > at; --i) { v[i] = v[i - 1]; e[i] = e[i - 1]; }
        v[at] = f;
        e[at] = (int32_t) x;
        if (filled < k + 1) ++filled;
    }
    for (int64_t i = 0; i < k; ++i) pick[i] = e[i];
    if (filled > k) *gap = (double) v[k - 1] - (double) v[k];   // NaN when there is no (K+1)-th seat
}

/// Fill `d` from the four blocks the probe read back, over the geometry already set in it. Pure host arithmetic
/// over copied bytes - no CUDA, no printing - so the parity test calls exactly what the engine calls. A block a
/// path never had is passed null and fingerprints as the offset basis, the same word `chunk_dump` uses for a
/// routing it never held.
inline void router_dump_collect(RouterDump& d, const float* logits, const uint16_t* in, const int32_t* ids,
                                const float* w) {
    const int64_t T = d.T > 0 ? d.T : 0, ne = d.n_expert, k = d.K, n = d.n_embd;
    d.lg_t.assign((size_t) T, kFnvOffset);
    d.in_t.assign((size_t) T, kFnvOffset);
    d.ids_t.assign((size_t) T, kFnvOffset);
    d.w_t.assign((size_t) T, kFnvOffset);
    d.gap_t.assign((size_t) T, 0.0 / 0.0);
    d.bad_t.assign((size_t) T, 0);
    d.lg = d.in = d.ids = d.w = kFnvOffset;
    d.ties = d.bad = d.sel_cmp = 0;
    d.gap_tok = -1;
    d.disagree = -1;
    d.gap_min = 0.0;
    if (T <= 0 || ne <= 0 || k <= 0) return;
    std::vector<int32_t> pick((size_t) k, -1);
    for (int64_t t = 0; t < T; ++t) {
        const float* lrow = logits != nullptr ? logits + t * ne : nullptr;
        const uint64_t hl = lrow != nullptr ? chunk_hash(lrow, (size_t) ne * sizeof(float)) : kFnvOffset;
        const uint64_t hi = in != nullptr ? chunk_hash(in + t * n, (size_t) n * sizeof(uint16_t)) : kFnvOffset;
        const uint64_t hd = ids != nullptr ? chunk_hash_ids(ids + t * k, k) : kFnvOffset;
        const uint64_t hw = w != nullptr ? chunk_hash(w + t * k, (size_t) k * sizeof(float)) : kFnvOffset;
        d.lg_t[(size_t) t] = hl;
        d.in_t[(size_t) t] = hi;
        d.ids_t[(size_t) t] = hd;
        d.w_t[(size_t) t] = hw;
        // the block folds its tokens in order, so one summary word is a claim about every token word under it -
        // and a block the path never held folds not at all, so its word stays the offset basis that says "absent"
        // rather than becoming a fingerprint of nothing that a second boot could match
        if (lrow != nullptr) d.lg = chunk_hash(d.lg, &hl, sizeof hl);
        if (in != nullptr) d.in = chunk_hash(d.in, &hi, sizeof hi);
        if (ids != nullptr) d.ids = chunk_hash(d.ids, &hd, sizeof hd);
        if (w != nullptr) d.w = chunk_hash(d.w, &hw, sizeof hw);
        if (lrow == nullptr) continue;
        bool bad_row = false;
        for (int64_t x = 0; x < ne && !bad_row; ++x)
            if (!std::isfinite(lrow[x])) bad_row = true;
        if (bad_row) { d.bad_t[(size_t) t] = 1; ++d.bad; }
        double gap = 0.0 / 0.0;
        router_select(lrow, ne, k, pick.data(), &gap);
        d.gap_t[(size_t) t] = gap;
        if (gap == 0.0) ++d.ties;
        if (gap == gap && (d.gap_tok < 0 || gap < d.gap_min)) { d.gap_min = gap; d.gap_tok = t; }
        if (ids != nullptr && k <= kRouterMaxK) {
            ++d.sel_cmp;
            bool same = true;
            for (int64_t i = 0; i < k && same; ++i) same = ids[t * k + i] == pick[i];
            if (!same && d.disagree < 0) d.disagree = t;
        }
    }
}

/// The summary line's text, minus its tag: built HERE rather than at the call site so the parity test grades the
/// exact words two boots are diffed on, and a field that silently stopped being printed is a failing test rather
/// than a boot wasted. Fixed-width hex, so a pair of logs diffs with `diff` and no second tool has to be trusted.
inline std::string router_dump_summary(const RouterDump& d) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "router dump p0=%lld T=%lld layer=%lld K=%lld n_expert=%lld n_embd=%lld | lg=%016llx in=%016llx "
                  "ids=%016llx w=%016llx | ties=%lld bad=%lld gap=%.9g at %lld | sel %lld differ %lld",
                  (long long) d.p0, (long long) d.T, (long long) d.layer, (long long) d.K, (long long) d.n_expert,
                  (long long) d.n_embd, (unsigned long long) d.lg, (unsigned long long) d.in,
                  (unsigned long long) d.ids, (unsigned long long) d.w, (long long) d.ties, (long long) d.bad,
                  d.gap_min, (long long) d.gap_tok, (long long) d.sel_cmp, (long long) d.disagree);
    return buf;
}

/// One token's line. Every line names its own (chunk, layer, token), so a log other threads interleaved still reads.
inline std::string router_dump_token(const RouterDump& d, int64_t t) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "router tok p0=%lld layer=%lld t=%lld lg=%016llx in=%016llx ids=%016llx w=%016llx gap=%.9g bad=%d",
                  (long long) d.p0, (long long) d.layer, (long long) t,
                  t < (int64_t) d.lg_t.size() ? (unsigned long long) d.lg_t[(size_t) t] : 0ull,
                  t < (int64_t) d.in_t.size() ? (unsigned long long) d.in_t[(size_t) t] : 0ull,
                  t < (int64_t) d.ids_t.size() ? (unsigned long long) d.ids_t[(size_t) t] : 0ull,
                  t < (int64_t) d.w_t.size() ? (unsigned long long) d.w_t[(size_t) t] : 0ull,
                  t < (int64_t) d.gap_t.size() ? d.gap_t[(size_t) t] : 0.0 / 0.0,
                  t < (int64_t) d.bad_t.size() ? (int) d.bad_t[(size_t) t] : 0);
    return buf;
}

}  // namespace basalt::prefill

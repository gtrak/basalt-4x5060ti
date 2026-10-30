// tests/cuda/peer_qsa_span_test.cpp - the partner's prompt-attention span (--exp tp_prompt_attn), sized: the
// numbers RUN 25 published, checked against the sizers rather than restated.  Nothing here touches a device -
// `qsa_prompt_state_bytes` and the `Alloc` count walk are arithmetic - so the figures the partner's expert tier
// is shrunk by are pinned by a test, not by a comment.
//
// The identities:
//   1. the gate closed prices nothing (the tier is byte-identical to before the slice existed);
//   2. the span part is exactly `n_qsa_layers` x the session's own per-layer prompt-state figure, so a KV format
//      change moves it with the session instead of drifting from it;
//   3. the shard narrows the KV and NOT the indexer (qsa_head_split.hpp: the indexer never splits), which is the
//      one place a mistake would silently over- or under-fund the partner;
//   4. the wire is the mix RUN 25 settled on, and its total is the 104.8 MB a layer the ledger quotes.
#include "basalt/core/layer.hpp"
#include "basalt/prefill/prefill.hpp"

#include <cstdio>
#include <cstdint>

using namespace basalt::core;
using basalt::prefill::Prefill;

namespace {

int g_fail = 0;
#define CHECK(c)                                                                    \
    do {                                                                            \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

const ModelGeometry g;   // the artifact's geometry: 48 layers, 12 QSA, 24 heads, 2 KV heads, 256 dims
constexpr int64_t CELLS = 262144;   // the shipped context, which is what the tier is sized for
constexpr int64_t T = 8192;         // the chunk the ledger is priced at

Prefill::PeerQsaSpan span_for(int64_t lb, int64_t le, int64_t lead_heads, int rank, bool on = true) {
    Prefill::PeerQsaSpan sp;
    sp.g = &g;
    sp.max_cells = CELLS;
    sp.lb = lb;
    sp.le = le;
    sp.heads = qsa_pair_shard(g.n_head, g.n_head_kv, lead_heads, rank);
    sp.on = on;
    return sp;
}

uint64_t part(const Prefill::PeerQsaSpan& s, int which) {
    uint64_t a = 0, b = 0, c = 0, t = 0;
    s.bytes(T, which == 0 ? &a : nullptr, which == 1 ? &b : nullptr, which == 2 ? &c : nullptr,
            which == 3 ? &t : nullptr);
    return which == 0 ? a : which == 1 ? b : which == 2 ? c : t;
}

// The ledger's "MB" are 10^6 bytes (how the wire figures are quoted) and its "GiB" are 2^30 (how the tier
// figures are quoted).  Both are asserted here so the two never get compared with one another's units.
constexpr double MB = 1000.0 * 1000.0;
constexpr double MiB = 1024.0 * 1024.0;
constexpr double GiB = 1024.0 * MiB;

void test_gate_closed() {
    const Prefill::PeerQsaSpan off = span_for(0, 24, 12, 1, /*on=*/false);
    CHECK(part(off, 3) == 0);
    CHECK(part(off, 0) == 0);
    // a span the caller never turned on must not shrink a tier
    const Prefill::PeerQsaSpan bad = span_for(0, 24, 12, 1);
    Prefill::PeerQsaSpan no_geom = bad;
    no_geom.g = nullptr;
    CHECK(part(no_geom, 3) == 0);
    Prefill::PeerQsaSpan no_shard = bad;
    no_shard.heads = QsaHeadShard{};
    CHECK(part(no_shard, 3) == 0);
}

void test_layer_count() {
    // qsa_interval is 4 over 48 layers, so each half holds 6 QSA layers (RUN 25 correction 1a)
    CHECK(span_for(0, 24, 12, 1).n_qsa_layers() == 6);
    CHECK(span_for(24, 48, 12, 1).n_qsa_layers() == 6);
    CHECK(span_for(0, 48, 12, 1).n_qsa_layers() == 12);
    CHECK(span_for(0, 4, 12, 1).n_qsa_layers() == 1);    // layer 3
    CHECK(span_for(1, 4, 12, 1).n_qsa_layers() == 1);    // layer 3 is inside [1, 4)
    CHECK(span_for(1, 3, 12, 1).n_qsa_layers() == 0);
    CHECK(span_for(5, 5, 12, 1).n_qsa_layers() == 0);
}

void test_span_tracks_the_session_sizer() {
    const Prefill::PeerQsaSpan s = span_for(0, 24, 12, 1);
    const QsaHeadShard whole{g.n_head, g.n_head_kv, 0, 1};
    uint64_t want = 0;
    for (int64_t i = 0; i < 6; ++i) want += qsa_prompt_state_bytes(g, CELLS, whole);
    // the whole-layer shard prices the same as the pair's rank 0 at the symmetric cut for the INDEXER part, but
    // not for the KV: compare against the partner's own shard, which is what funds
    CHECK(part(s, 0) == 6 * qsa_prompt_state_bytes(g, CELLS, s.heads));
    CHECK(want > part(s, 0));   // the partner holds half the KV, so strictly less than a whole layer
}

void test_shard_narrows_kv_only() {
    // rank 0 and rank 1 of the symmetric cut hold the SAME total KV, and each holds one KV head: the difference
    // between a whole state and a shard's is exactly the KV it gave up (heads x dims x (K,V) x bytes), and the
    // indexer's pooled table is identical in both - it never splits (qsa_head_split.hpp).
    const int64_t hd = 256;
    const uint64_t whole = qsa_prompt_state_bytes(g, CELLS, QsaHeadShard{g.n_head, g.n_head_kv, 0, 1});
    const Prefill::PeerQsaSpan r1 = span_for(0, 24, 12, 1);
    const Prefill::PeerQsaSpan r0 = span_for(0, 24, 12, 0);
    CHECK(r1.heads.nkv() == 1 && r0.heads.nkv() == 1);
    const uint64_t kv_one_head = (uint64_t) CELLS * hd * 2u * (qsa_kv_int8() ? 1u : 2u);
    CHECK(whole - qsa_prompt_state_bytes(g, CELLS, r0.heads) == kv_one_head);
    CHECK(whole - qsa_prompt_state_bytes(g, CELLS, r1.heads) == kv_one_head);
    CHECK(part(r0, 0) == part(r1, 0));   // the two cards of a symmetric pair are funded the same
}

void test_wire_is_the_filed_mix() {
    const Prefill::PeerQsaSpan s = span_for(0, 24, 12, 1);
    const uint64_t one = (uint64_t) T * g.n_embd * 2          // the lead's FP16 mixed rows (the q source)
                       + (uint64_t) T * 4 * 128 * 4           // q_idx
                       + (uint64_t) T * 128 * 4               // idx_raw
                       + (uint64_t) T * g.n_embd * 2;         // the o_proj partial, BF16
    // The TRAFFIC figure is what the ledger prices and the +10/+20 gates grade against: RUN 25 filed 104.8 MB
    // a QSA layer.  Checked to 1%, which catches a dropped term (a 41.9 MB class error).
    CHECK(s.wire_one_set(T) == one);
    const double wire_mb = (double) one / MB;
    CHECK(wire_mb > 104.0 && wire_mb < 105.6);
    // The RESIDENT wire is `kWireSets` sets: the pipelined gate overlaps layer l+1's inbound with layer l's
    // compute, which one set cannot do.  Pinned here so the tier figure and the traffic figure can never be
    // mistaken for one another - the mistake would either hide 105 MB from the tier or double the ledger.
    CHECK(Prefill::PeerQsaSpan::kWireSets == 2);
    CHECK(part(s, 1) == one * Prefill::PeerQsaSpan::kWireSets);
    std::printf("wire: traffic %6.1f MB a QSA layer (filed 104.8)   resident %6.1f MB (%d sets)\n", wire_mb,
                (double) part(s, 1) / MB, Prefill::PeerQsaSpan::kWireSets);
    // the wire does not shrink with the head split: the split buys compute, not bytes.  A narrower partner
    // pays the same for the mixed rows it needs to project its own q/k/v.
    const Prefill::PeerQsaSpan narrow = span_for(0, 24, 20, 1);   // the partner takes only 4 heads
    CHECK(narrow.wire_one_set(T) == one);
    CHECK(part(narrow, 2) < part(s, 2));                          // ...and its scratch does shrink
}

void test_total_is_the_sum_of_parts() {
    const Prefill::PeerQsaSpan s = span_for(0, 24, 12, 1);
    const uint64_t a = part(s, 0), b = part(s, 1), c = part(s, 2), t = part(s, 3);
    CHECK(t == a + b + c);
    CHECK(a > 0 && b > 0 && c > 0);
    // the span dominates, and it is the term the partner's tier pays for the slice: 6 QSA layers of KV + indexer
    CHECK(a > b && a > c);
    std::printf("fp16: span %.2f GiB  wire %.1f MB  scratch %.1f MB  total %.2f GiB\n", (double) a / GiB,
                (double) b / MB, (double) c / MB, (double) t / GiB);
    // the reservation delta the second wire set buys: honest about what the overlap costs the partner's tier
    const uint64_t single = a + s.wire_one_set(T) + c;
    std::printf("        second wire set costs %+.0f MB of tier (%.2f -> %.2f GiB total)\n",
                (double) (t - single) / MB, (double) single / GiB, (double) t / GiB);
    CHECK(t > single);
    CHECK(t - single == s.wire_one_set(T));   // exactly one extra set, no more
    // the FP16 span, from geometry: 6 x (262144 cells x 256 dim x 2 x 2 B) of KV, plus the pooled table
    CHECK((double) a / GiB > 1.6 && (double) a / GiB < 1.85);
    // the shipped boot: --kv int8, which is what RUN 25's adopted ~1.0 GiB figure prices
    qsa_set_kv_int8(true);
    const uint64_t i_a = part(s, 0), i_t = part(s, 3);
    qsa_set_kv_int8(false);
    CHECK(i_a < a);   // int8 really did narrow it; a format the sizer ignored would fail here
    CHECK(i_t == i_a + b + c);
    std::printf("int8: span %.2f GiB  total %.2f GiB (RUN 25 adopted 1.006)\n", (double) i_a / GiB,
                (double) i_t / GiB);
    CHECK((double) i_a / GiB > 0.90 && (double) i_a / GiB < 1.12);
}

void test_uneven_ratio_stays_funded() {
    // the share flags price the cut at qsa_lead_row_us 2.61 / qsa_peer_row_us 0.98, i.e. the partner takes MORE
    // than half; the span has to follow the ratio the boot was given, not assume 50/50
    const Prefill::PeerQsaSpan even = span_for(0, 24, 12, 1);
    const Prefill::PeerQsaSpan heavy = span_for(0, 24, 8, 1);   // the lead 8 heads, the partner 16
    CHECK(heavy.heads.nq() == 16 && even.heads.nq() == 12);
    CHECK(part(heavy, 2) > part(even, 2));   // more scratch for more heads
    CHECK(qsa_pair_tiles(qsa_pair_shard(g.n_head, g.n_head_kv, 8, 0), heavy.heads));
    // a ratio that is not a whole KV group is still priced (the attention kernels refuse it elsewhere, at
    // qsa_shard_whole_groups, and the sizing must not be the place that silently drops it)
    const Prefill::PeerQsaSpan straddle = span_for(0, 24, 10, 1);
    CHECK(part(straddle, 3) > 0);
    CHECK(!qsa_shard_whole_groups(straddle.heads));
}

}  // namespace

int main() {
    test_gate_closed();
    test_layer_count();
    test_span_tracks_the_session_sizer();
    test_shard_narrows_kv_only();
    test_wire_is_the_filed_mix();
    test_total_is_the_sum_of_parts();
    test_uneven_ratio_stays_funded();
    if (g_fail == 0) std::printf("peer_qsa_span_test: ok\n");
    else std::printf("peer_qsa_span_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

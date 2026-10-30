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
#include "basalt/core/diag.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/kernels/tp2_collective.hpp"
#include "basalt/prefill/prefill.hpp"

#include <cstdio>
#include <cstdint>
#include <vector>

using namespace basalt::core;
using basalt::kernels::tp2_staging_elems;
using basalt::kernels::tp2_staging_set_ok;
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
constexpr int64_t max_cells_for_test = CELLS;

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
    uint64_t a = 0, b = 0, c = 0, w = 0, g = 0, o = 0, t = 0;
    s.bytes(T, which == 0 ? &a : nullptr, which == 1 ? &b : nullptr, which == 2 ? &c : nullptr,
            which == 3 ? &w : nullptr, which == 4 ? &g : nullptr, which == 5 ? &o : nullptr,
            which == 6 ? &t : nullptr);
    return which == 0 ? a : which == 1 ? b : which == 2 ? c : which == 3 ? w : which == 4 ? g : which == 5 ? o : t;
}

// The ledger's "MB" are 10^6 bytes (how the wire figures are quoted) and its "GiB" are 2^30 (how the tier
// figures are quoted).  Both are asserted here so the two never get compared with one another's units.
constexpr double MB = 1000.0 * 1000.0;
constexpr double MiB = 1024.0 * 1024.0;
constexpr double GiB = 1024.0 * MiB;

void test_gate_closed() {
    const Prefill::PeerQsaSpan off = span_for(0, 24, 12, 1, /*on=*/false);
    CHECK(part(off, 6) == 0);
    CHECK(part(off, 0) == 0);
    // a span the caller never turned on must not shrink a tier
    const Prefill::PeerQsaSpan bad = span_for(0, 24, 12, 1);
    Prefill::PeerQsaSpan no_geom = bad;
    no_geom.g = nullptr;
    CHECK(part(no_geom, 6) == 0);
    Prefill::PeerQsaSpan no_shard = bad;
    no_shard.heads = QsaHeadShard{};
    CHECK(part(no_shard, 6) == 0);
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
    // the walk reports what is actually HELD, so each layer's allocation carries the 256 B round-up `take`
    // applies; the raw sizer figure is the floor, and the gap must be rounding only, never a missing term
    uint64_t raw = 0, rounded = 0;
    for (int64_t i = 0; i < 6; ++i) {
        const uint64_t b = qsa_prompt_state_bytes(g, CELLS, s.heads);
        raw += b;
        rounded += (b + 255) & ~uint64_t(255);
    }
    CHECK(part(s, 0) == rounded);
    CHECK(part(s, 0) >= raw);
    CHECK(part(s, 0) - raw < 6u * 256u);
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
    const uint64_t a = part(s, 0), b = part(s, 1), c = part(s, 2), t = part(s, 6);
    CHECK(t == a + b + c + part(s, 3) + part(s, 4));   // every term, so none can go unpriced
    CHECK(a > 0 && b > 0 && c > 0);
    // the span dominates, and it is the term the partner's tier pays for the slice: 6 QSA layers of KV + indexer
    CHECK(a > b && a > c);
    std::printf("fp16: span %.2f GiB  wire %.1f MB  scratch %.1f MB  total %.2f GiB\n", (double) a / GiB,
                (double) b / MB, (double) c / MB, (double) t / GiB);
    // the reservation delta the second wire set buys: honest about what the overlap costs the partner's tier
    const uint64_t single = a + s.wire_one_set(T) + c + part(s, 3) + part(s, 4);
    std::printf("        second wire set costs %+.0f MB of tier (%.2f -> %.2f GiB total)\n",
                (double) (t - single) / MB, (double) single / GiB, (double) t / GiB);
    CHECK(t > single);
    CHECK(t - single == s.wire_one_set(T));   // exactly one extra set, no more
    // the FP16 span, from geometry: 6 x (262144 cells x 256 dim x 2 x 2 B) of KV, plus the pooled table
    CHECK((double) a / GiB > 1.6 && (double) a / GiB < 1.85);
    // the shipped boot: --kv int8, which is what RUN 25's adopted ~1.0 GiB figure prices
    qsa_set_kv_int8(true);
    const uint64_t i_a = part(s, 0), i_t = part(s, 6);
    qsa_set_kv_int8(false);
    CHECK(i_a < a);   // int8 really did narrow it; a format the sizer ignored would fail here
    CHECK(i_t == i_a + b + c + part(s, 3) + part(s, 4));
    std::printf("int8: span %.2f GiB  wire %.2f GiB  scratch %.2f GiB  weights %.2f GiB  gemm %.2f GiB  "
                "TOTAL %.2f GiB\n",
                (double) i_a / GiB, (double) b / GiB, (double) c / GiB, (double) part(s, 3) / GiB,
                (double) part(s, 4) / GiB, (double) i_t / GiB);
    CHECK((double) i_a / GiB > 0.90 && (double) i_a / GiB < 1.12);
    // the partner's own GEMM context: it cannot borrow the lead's dequant scratch or cuBLAS workspace across
    // the link, so the pair's second card pays for its own of both
    CHECK(part(s, 4) == (uint64_t) (32ll << 20) * 2 + (32u << 20));
    CHECK(part(s, 4) > GiB * 0.09 && part(s, 4) < GiB * 0.10);
}

void test_resident_projections_are_priced() {
    // The RUN 25 ruling: the partner's projections are resident on the partner's card, copied at boot, so a QSA
    // layer moves no weight bytes.  With the pack never consulted the sizer prices NOTHING rather than invent a
    // block layout - a guessed quantization table would put a wrong number on the tier figure.
    Prefill::PeerQsaSpan s = span_for(0, 24, 12, 1);
    CHECK(part(s, 3) == 0);
    const int64_t qrow = 1280, kvrow = 1280;   // a Q2_0-class row of n_embd 2560
    s.q_row_bytes = qrow;
    s.kv_row_bytes = kvrow;
    // 6 layers x (12 heads x 2 x 256 rows of attn_q  +  1 KV head x 256 rows each of attn_k and attn_v)
    // the row runs are asserted as literals, not via qsa_q_rows, so a wrong run would show here
    // attn_q rows, attn_k + attn_v rows, the two attention norms, and the two indexer norms (only k is read,
    // but q is allocated so the figure does not move if the shipped-q choice is revisited)
    const uint64_t per_layer = (6144ull * qrow) + 2ull * (256ull * kvrow)
                             + 2ull * 256ull * 4ull + 2ull * 128ull * 4ull;
    CHECK(part(s, 3) == 6ull * per_layer);
    // the allocation COUNT, not just the bytes: span 6 + wire 4x2 + scratch 8 + weights 7x6 + gemm 2.  A tensor
    // taken once instead of once a layer keeps the byte figure correct and this number wrong.
    s.bytes(T, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    // one QSA state a layer, four wire buffers a set, ten scratch buffers, seven tensors a layer, two for the
    // partner's own Gemm
    const int expect_takes = 6 /*states*/ + 4 * Prefill::PeerQsaSpan::kWireSets + 10 /*scratch*/ + 7 * 6 + 2;
    std::printf("the walk made %d allocations, as counted\n", s.last_takes());
    CHECK(s.last_takes() == expect_takes);
    // The PIPELINED gate (`--exp tp_prompt_attn=2`) holds a second `o_proj` staging set, so the walk takes ONE
    // more buffer and the tier must have left its bytes free: an arm that allocates what the reserve did not
    // price is an allocation failure at the first gated prompt, not a slower boot.
    {
        Prefill::PeerQsaSpan pip = span_for(0, 24, 12, 1);
        pip.staging_sets = 1;
        pip.bytes(T, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        const int serial_takes = pip.last_takes();
        pip.staging_sets = Prefill::PeerQsaSpan::kStagingSets;
        const uint64_t pipelined_scratch = part(pip, 2);
        const uint64_t row = ((uint64_t) T * 2560ull * 4ull + 255) & ~uint64_t(255);   // rows x n_embd x FP32
        pip.bytes(T, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        const int pipelined_takes = pip.last_takes();
        pip.staging_sets = 1;
        const uint64_t serial_scratch = part(pip, 2);
        CHECK(pipelined_scratch == serial_scratch + row);
        CHECK(pipelined_takes == serial_takes + 1);
        std::printf("pipelined staging: %d allocations against %d, +%.2f GiB for the second o_proj set\n",
                    pipelined_takes, serial_takes, (double) (pipelined_scratch - serial_scratch) / GiB);
    }
    std::printf("resident projections: %.3f GiB over 6 QSA layers (attn_q rows 6144, attn_k/v rows 256)\n",
                (double) part(s, 3) / GiB);
    // half the heads is half the attn_q residency, but the norms are not halved away
    Prefill::PeerQsaSpan narrow = span_for(0, 24, 20, 1);   // the partner takes 4 heads
    narrow.q_row_bytes = qrow;
    narrow.kv_row_bytes = kvrow;
    CHECK(part(narrow, 3) < part(s, 3));
    // an uneven ratio gives the partner MORE heads and therefore more resident weight
    Prefill::PeerQsaSpan heavy = span_for(0, 24, 8, 1);     // the partner takes 16 heads
    heavy.q_row_bytes = qrow;
    heavy.kv_row_bytes = kvrow;
    CHECK(part(heavy, 3) > part(s, 3));
    CHECK(part(heavy, 6) > part(s, 6));   // and the tier figure carries it, one honest number
    // residency, not traffic: the projections do not grow with the chunk length or the context, only the span
    // does -- which is what makes them a boot-time cost rather than a per-layer one
    Prefill::PeerQsaSpan twice = s;
    twice.max_cells = max_cells_for_test * 2;   // a longer context grows the SPAN and the block scores...
    CHECK(part(twice, 3) == part(s, 3));        // ...but not the resident projections
    CHECK(part(twice, 0) > part(s, 0));
    // the selection-width buffers scale with the CHUNK, not the context: a shorter chunk is less scratch
    Prefill::PeerQsaSpan short_chunk = s;
    const uint64_t long_scratch = part(short_chunk, 2);
    uint64_t a2 = 0, b2 = 0, c2 = 0, w2 = 0, g2 = 0, o2 = 0, t2 = 0;
    short_chunk.bytes(T / 2, &a2, &b2, &c2, &w2, &g2, &o2, &t2);
    CHECK(c2 < long_scratch);
    CHECK(a2 == part(short_chunk, 0));   // the span is cells, not tokens: unchanged
}

void test_the_reserve_refuses_a_wider_split_by_name() {
    // The ruling is that the tier is reserved at the pinned symmetric cut and a WIDER partner is refused, with
    // both splits named so the operator knows what to pin.  Tested here, on the CPU, because a rule that can
    // only be reached from a boot that has already committed to the split is a rule nobody can check: the
    // refusal path is the one that has to be right, and it is the one least likely to be exercised.
    const int64_t qrow = 1280, kvrow = 1280, ocol = 3200;
    auto funded = [&](Prefill::PeerQsaSpan sp) {
        sp.q_row_bytes = qrow; sp.kv_row_bytes = kvrow; sp.o_col_bytes = ocol;
        sp.q_type = sp.kv_type = sp.o_type = 0;   // set, so nothing is refused for want of a format
        return sp;
    };
    const Prefill::PeerQsaSpan reserve = funded(span_for(0, 24, 12, 1));   // the symmetric cut: lead 12, partner 12

    // the symmetric split fits its own reserve, exactly
    std::string why;
    CHECK(funded(span_for(0, 24, 12, 1)).fits_within(reserve, T, why));
    CHECK(why.empty());
    // a partner with FEWER heads than the reserve priced also fits: less to hold is never a problem
    CHECK(funded(span_for(0, 24, 16, 1)).fits_within(reserve, T, why));
    // a partner with MORE does not, and the message has to name both splits and both figures - an operator
    // reading it must know what to pin without reading the source
    const Prefill::PeerQsaSpan wide = funded(span_for(0, 24, 8, 1));       // partner takes 16 of 24
    why.clear();
    CHECK(!wide.fits_within(reserve, T, why));
    CHECK(why.find("16/24") != std::string::npos);
    CHECK(why.find("priced 12") != std::string::npos);
    CHECK(why.find("qsa_ratio_stage") != std::string::npos);   // says what to pin, not just that it failed
    std::printf("the refusal reads: %s\n", why.c_str());
    // a declared span with no reserve behind it is a configuration error, not a zero-need case
    const Prefill::PeerQsaSpan none = funded(span_for(0, 24, 12, 1));
    why.clear();
    CHECK(!wide.fits_within(span_for(0, 24, 12, 1, /*on=*/false), T, why));
    CHECK(!why.empty());
    CHECK(none.on);
    // nothing declared is always covered, whatever the reserve says
    why.clear();
    CHECK(span_for(0, 24, 12, 1, /*on=*/false).fits_within(reserve, T, why));
    // the check is on bytes, not only heads: the same head count over a longer context needs more and must refuse
    Prefill::PeerQsaSpan longer = funded(span_for(0, 24, 12, 1));
    longer.max_cells = CELLS * 2;
    why.clear();
    CHECK(!longer.fits_within(reserve, T, why));   // the span grows with cells; the reserve did not
}

void test_o_cols_and_types_are_funded() {
    // The o_proj half of the partner's arm: its columns of attn_output, PACKED, because a column slice of a
    // blocked weight is not a pointer.  qsa_pair_prepare allocates them on the partner's card, so nothing in the
    // walk takes a pointer for them - which is exactly why they need their own priced part.  An unfunded copy
    // is not a slow boot, it is an allocation failure at the first gated prompt, or an o_proj reading a buffer
    // that was never reserved.
    const int64_t qrow = 1280, kvrow = 1280, ocol = 3200;   // a Q2_0-class row, and one layer's packed columns
    Prefill::PeerQsaSpan s = span_for(0, 24, 12, 1);
    s.q_row_bytes = qrow; s.kv_row_bytes = kvrow;
    // unset means the pack was never consulted: priced as nothing, not as a guessed block layout
    CHECK(s.o_col_bytes == 0);
    CHECK(part(s, 5) == 0);
    CHECK(s.q_type == -1 && s.kv_type == -1 && s.o_type == -1);   // the arm reads these; unset must not look set

    s.o_col_bytes = ocol;
    const uint64_t per_layer = (ocol + 255) & ~uint64_t(255);      // rounded the way every other part is
    CHECK(part(s, 5) == 6ull * per_layer);
    // it is a real part of the tier figure, not a footnote: the total moved by exactly the columns
    const uint64_t without = [&] { Prefill::PeerQsaSpan x = s; x.o_col_bytes = 0; return part(x, 6); }();
    CHECK(part(s, 6) == without + 6ull * per_layer);
    CHECK(part(s, 0) + part(s, 1) + part(s, 2) + part(s, 3) + part(s, 4) + part(s, 5) == part(s, 6));

    // packed columns are per QSA layer, and narrower for a card that runs fewer heads - the byte figure has to
    // follow the shard the way every other part does
    Prefill::PeerQsaSpan narrow = span_for(0, 24, 20, 1);   // the partner takes 4 heads, not 12
    narrow.o_col_bytes = ocol / 3;
    CHECK(part(narrow, 5) < part(s, 5));
    // and it is residency, not traffic: a longer context or a shorter chunk does not move it
    Prefill::PeerQsaSpan twice = s; twice.max_cells = max_cells_for_test * 2;
    CHECK(part(twice, 5) == part(s, 5));
    Prefill::PeerQsaSpan shorty = s;
    uint64_t o_short = 0, t_short = 0;
    shorty.bytes(T / 2, nullptr, nullptr, nullptr, nullptr, nullptr, &o_short, &t_short);
    CHECK(o_short == part(s, 5));
    // MiB, not GiB: this prices a SYNTHETIC column size to test the arithmetic, and the real figure comes from
    // the pack at boot (`qsa_weight_cols_bytes`), where the boot line reports it in the tier's own units
    std::printf("packed attn_output columns: %.3f MiB over 6 QSA layers at %lld B a layer\n",
                (double) part(s, 5) / MiB, (long long) ocol);
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
    CHECK(part(straddle, 6) > 0);
    CHECK(!qsa_shard_whole_groups(straddle.heads));
}

}  // namespace

namespace basalt::core {
/// Builds a `WeightTable` row by row (the friend the shape-check test uses, defined again in this TU) so the
/// pair's column sizer can be exercised on a table shaped like a STAGE's: the attention projections the shipped
/// artifact serves from its GGUF section exist as a native image only for the layers that stage was given
/// (`NativeDense::set_layer_range(st.lo, st.hi)`), and carry neither form anywhere else.
struct WeightTableFixture {
    static void put(WeightTable& t, const std::string& name, const WeightRef& r) { t.table_[name] = r; }
};
}  // namespace basalt::core

namespace {

/// The pair's packed `attn_output` columns, priced over a LAYER RANGE, on a table that only carries the stage's
/// own layers.  The first gated boot died here: `qsa_pair_prepare` walked the whole model and refused on the
/// other stage's first QSA layer, whose `attn_output` has no bytes at all on this card - it read as a pack that
/// could not be cut (`native 0 B`) and was really a stage-scoped table asked for a whole-model range.
void test_the_pair_prices_its_stages_layers() {
    const int64_t Q6K_ROW = 210 * 24;   // one row of the artifact's Q6_K attn_output
    const uint64_t image = (uint64_t) Q6K_ROW * 2560;
    std::vector<uint8_t> dummy(1);      // never read: the sizer prices, it does not dereference
    auto native_only = [&](WeightRef& r) {
        r.ne0 = 24 * 256;
        r.ne1 = 2560;
        r.kind = WeightKind::Verbatim;
        r.code_bits = 8;   // quantized metadata, but no canonical bytes: what the index records for this pack
        r.group_elems = 32;
        r.resident = false;
        r.data = nullptr;
        r.bytes = r.codes_bytes = r.scales_bytes = r.offset_bytes = 0;
        r.has_offset = false;
        r.native_data = dummy.data();
        r.native_type = 14;   // Q6_K
    };
    WeightTable stage_a;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!is_qsa_layer(g, l)) continue;
        WeightRef r;
        if (l < 24) native_only(r);   // stage A's table: its own layers have images, the rest have nothing
        else { r.ne0 = 24 * 256; r.ne1 = 2560; r.resident = false; }   // no planes, no native image
        WeightTableFixture::put(stage_a, "blk." + std::to_string(l) + ".attn_output.weight", r);
    }
    const QsaHeadShard rank1 = qsa_pair_shard(g.n_head, g.n_head_kv, g.n_head / 2, /*rank*/ 1);
    const uint64_t one_layer = (uint64_t) (6451200ull + 15) & ~uint64_t(15);
    CHECK(qsa_weight_cols_bytes(*LayerView(stage_a, 3).get("attn_output.weight"), 12 * 256, 12 * 256, image) ==
          6451200ull);
    // the stage's range prices six layers; the other stage's prices nothing, and the whole model is the same
    // figure rather than double it - the property `lead_qsa_bytes` relies on when it reserves over [0, n_layers)
    CHECK(qsa_pair_bytes(stage_a, g, rank1, 0, 24) == 6 * one_layer);
    CHECK(qsa_pair_bytes(stage_a, g, rank1, 24, 48) == 0);
    CHECK(qsa_pair_bytes(stage_a, g, rank1, 0, g.n_layers) == qsa_pair_bytes(stage_a, g, rank1, 0, 24));
    // a range with no QSA layer in it prices nothing rather than under-flowing, and one that leaves the model is
    // clamped rather than reading rows that do not exist
    CHECK(qsa_pair_bytes(stage_a, g, rank1, 4, 6) == 0);
    CHECK(qsa_pair_bytes(stage_a, g, rank1, -8, 60) == qsa_pair_bytes(stage_a, g, rank1, 0, 24));
    std::printf("pair columns: %.1f MiB a QSA layer, %.1f MiB over a stage of 6\n", one_layer / MiB,
                (6.0 * one_layer) / MiB);
}

}  // namespace

// ===================== the pair's collective: the staging-set protocol, replayed =====================

/// Adversarial replay of `Tp2Pair`'s flag protocol (src/kernels/cuda/tp2_collective.cu) for `sets` landing regions.
///
/// The lead's queue per collective k mirrors the driver: publish the wire, push the o_proj partial, flag, add. The
/// partner's queue per collective k is push, flag, add - its push waits on the lead's PUBLISH of k (the arm's
/// `ev_in`, which sits ahead of the lead's own collective), and its add waits on the lead's flag of k. Nothing
/// credits a region back after an add read it, so the question is only whether a peer's next push into a region
/// can execute before this rank's add of the earlier collective in that region.
///
/// The scheduler starves one rank while the other runs every op it can, once with each rank as the victim: on the
/// machine the two streams make independent progress, so a rank that CAN run ahead will, and a protocol is safe
/// only if no such schedule corrupts a sum. `serialized` adds the `=1` gate's wait (the lead queues nothing more
/// until its peer's add of the previous collective). Returns the first collective whose add read a region holding
/// another collective's partial, 0 when every add read its own, -1 if the schedule deadlocks.
int replay_collective(int n, int sets, bool serialized) {
    if (sets < 1) sets = 1;
    if (n < 1 || n > 16) return -2;
    int pc[2] = {0, 0};
    int pushed[2] = {0, 0};                              // pushed_[r]: the k the peer last flagged to card r
    int adds_done[2] = {0, 0};
    bool published[17] = {false};                        // the lead published collective k's wire
    std::vector<std::vector<int>> last(2, std::vector<int>(sets, 0));   // last write into card r's region s

    // One op of rank r, if its head is runnable. 0 = ran, 1 = blocked, 2 = corrupt read, 3 = queue done.
    auto step = [&](int r) -> int {
        const int p = 1 - r, qn = (r == 0 ? 4 : 3) * n;
        if (pc[r] >= qn) return 3;
        const int idx = pc[r], kind = idx % (r == 0 ? 4 : 3), k = idx / (r == 0 ? 4 : 3) + 1;
        if (r == 0 && kind == 0) { published[k] = true; pc[r] = idx + 1; return 0; }        // wire publish
        const bool lead_push = (r == 0 && kind == 1), part_push = (r == 1 && kind == 0);
        if (part_push && !published[k]) return 1;                                          // the arm's ev_in
        if (lead_push && serialized && k > 1 && adds_done[p] < k - 1) return 1;             // the =1 ev_out wait
        const bool is_push = lead_push || part_push;
        const bool is_flag = (r == 0 && kind == 2) || (r == 1 && kind == 1);
        const bool is_add = (r == 0 && kind == 3) || (r == 1 && kind == 2);
        if (is_push) last[p][(k - 1) % sets] = k;
        if (is_flag) pushed[p] = k;
        if (is_add) {
            if (pushed[r] < k) return 1;                                                   // peer has not flagged it
            if (last[r][(k - 1) % sets] != k) return 2;                                    // another collective's
            adds_done[r] = k;
        }
        pc[r] = idx + 1;
        return 0;
    };

    for (int victim = 0; victim < 2; ++victim) {
        pc[0] = pc[1] = 0; pushed[0] = pushed[1] = 0; adds_done[0] = adds_done[1] = 0;
        for (int i = 0; i <= n; ++i) published[i] = false;
        for (int s = 0; s < sets; ++s) last[0][s] = last[1][s] = 0;
        const int other = 1 - victim;
        while (pc[0] < 4 * n || pc[1] < 3 * n) {
            bool ran = false;
            while (step(other) == 0) ran = true;          // the peer charges as far as it can
            const int v = step(victim);
            if (v == 2) return 1;                          // an add read another collective's partial
            if (v != 0 && !ran) return -1;                 // neither rank can move: the protocol deadlocks
        }
    }
    return 0;
}

void test_the_pair_funds_the_depth_it_runs() {
    // The pair's own pure rules: sets are counted from one, a region is addressable only below the count, and the
    // allocation is sets x the widest partial - which is what the tier's reserve has to book.
    CHECK(tp2_staging_elems(0, 4) == 0);
    CHECK(tp2_staging_elems(100, 1) == 100);
    CHECK(tp2_staging_elems(100, 2) == 200);
    CHECK(tp2_staging_elems(100, 0) == 100);        // a caller that asks for no sets still gets the one it needs
    CHECK(tp2_staging_set_ok(0, 1) && tp2_staging_set_ok(1, 2));
    CHECK(!tp2_staging_set_ok(1, 1));               // the pipelined depth, refused on a serialized pair
    CHECK(!tp2_staging_set_ok(-1, 2));
    CHECK(tp2_staging_set_ok(0, 0));                 // no sets asked for: the one region every pair must have is there

    // SERIALIZED (the =1 gate): one region, and the waits the mode carries are enough at any depth.
    CHECK(replay_collective(1, 1, true) == 0);
    CHECK(replay_collective(6, 1, true) == 0);
    CHECK(replay_collective(12, 1, true) == 0);

    // PIPELINED (the =2 gate) on ONE region: the peer's next push lands in the region this rank's add is still
    // reading. If this ever stops reproducing, the model has drifted from the kernel - or the protocol changed.
    CHECK(replay_collective(2, 1, false) > 0);
    CHECK(replay_collective(12, 1, false) > 0);

    // PIPELINED on TWO regions: the flag chain alone orders the two collectives that share a region, at any depth
    // the slice reaches (6 QSA layers a stage, and the wire alternates, so 12 is deeper than one chunk runs).
    CHECK(replay_collective(2, 2, false) == 0);
    CHECK(replay_collective(6, 2, false) == 0);
    CHECK(replay_collective(12, 2, false) == 0);
    std::printf("collective staging: one region corrupts at depth 2 with the lead ungated; two regions do not, to 12\n");

    // And the reserves must book what the pair allocates, or the second region is an allocation failure at the
    // first gated prompt.  `lead_qsa_bytes` reads the depth from the same gate `set_peer` refuses on, so the gate
    // is set here and the figure is required to grow by exactly one landing region per card.
    const uint64_t row = (((uint64_t) T * 2560ull * 4ull) + 255) & ~uint64_t(255);
    std::vector<uint8_t> dummy(1);      // never read: the sizer prices, it does not dereference
    WeightTable stage_a;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!is_qsa_layer(g, l)) continue;
        WeightRef r;
        r.ne0 = 24 * 256; r.ne1 = 2560;
        r.kind = WeightKind::Verbatim; r.code_bits = 8; r.group_elems = 32; r.resident = false;
        r.data = nullptr; r.bytes = r.codes_bytes = r.scales_bytes = r.offset_bytes = 0; r.has_offset = false;
        r.native_data = dummy.data(); r.native_type = 14;   // Q6_K, as the shipped pack serves it
        WeightTableFixture::put(stage_a, "blk." + std::to_string(l) + ".attn_output.weight", r);
    }
    auto& diag = basalt::core::diagnostics();
    diag.exp["tp_prompt_attn"] = 1;
    const uint64_t lead_one = Prefill::lead_qsa_bytes(g, stage_a, T, 12);
    diag.exp["tp_prompt_attn"] = 2;
    const uint64_t lead_two = Prefill::lead_qsa_bytes(g, stage_a, T, 12);
    diag.exp.erase("tp_prompt_attn");
    CHECK(lead_two == lead_one + row);      // the lead's landing region for the pair's second set
    std::printf("collective staging reserve: the lead's figure grows %.1f MiB for the pair's second set\n",
                (double) (lead_two - lead_one) / (1024.0 * 1024.0));
}

int main() {
    test_the_reserve_refuses_a_wider_split_by_name();
    test_o_cols_and_types_are_funded();
    test_gate_closed();
    test_layer_count();
    test_span_tracks_the_session_sizer();
    test_shard_narrows_kv_only();
    test_wire_is_the_filed_mix();
    test_total_is_the_sum_of_parts();
    test_resident_projections_are_priced();
    test_uneven_ratio_stays_funded();
    test_the_pair_prices_its_stages_layers();
    test_the_pair_funds_the_depth_it_runs();
    if (g_fail == 0) std::printf("peer_qsa_span_test: ok\n");
    else std::printf("peer_qsa_span_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}

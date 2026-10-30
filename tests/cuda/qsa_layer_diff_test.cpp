// tests/cuda/qsa_layer_diff_test.cpp - one sliced QSA layer's data path on the CPU, run three ways, compared per
// layer and per component.
//
// WHY THIS EXISTS INSTEAD OF A BOOT. The GPU chain cannot answer "is =2 wrong": the engine's prefill kernels
// accumulate in a schedule-dependent order, so two passes of ONE boot already disagree - measured on the captured
// chains, 16 of 180 layer-lines of `mix` differ between pass 0 and pass 1 of the same =1 boot, and `sel`, `kpool`,
// `vpool` and `bo` diverge on all 180. A cross-boot comparison of that instrument compares noise against noise.
// Here the same layer is recomputed where nothing is schedule-dependent, so a divergence means the TERMS differed
// rather than the order they were added in.
//
// THREE RUNS OF ONE PROMPT (layers x chunks x tokens):
//   R - the whole layer on one card: every head's attention, the whole `attn_output` multiply. The math gate-off
//       computes, and the definition of what the slice must reproduce;
//   S - the pair at `tp_prompt_attn=1`: one wire set, one staging set, one collective landing region, and the
//       lead's end-of-layer wait (`prefill.cpp`, the `sap->qsa.pipeline < 2` wait);
//   P - the pair at `tp_prompt_attn=2`: two sets of each, no end-of-layer wait, the lead charging ahead as far as
//       the coded orderings allow.
//
// THE GRADED SIGNAL. Per layer-instance and per component, the cosine against the reference: the slice's math is
// right iff every cell holds >= kCosineOk. The threshold is not arbitrary - re-adding the same terms in a different
// order costs a float a handful of ulps at most, which lands 1e-7 to 1e-5 off unity at these widths, while a wrong
// KV row, a stale partial or a swapped head block moves a whole tensor's direction and lands far below. So the
// graded signal tolerates reduction-order noise and cannot miss the shifted-distribution class this hunt is about.
// P is then compared to S BITWISE as well: on this replay both are deterministic, so bitwise is available and is
// the stronger assertion for the purely-ordering question of whether pipelining changed what was summed.
//
// WHY THE FAULTS ARE GRADED ON THAT SAME COSINE, NOT ON A VECTOR DIFFERENCE. An earlier draft asked only whether an
// injected fault made the run's buffers DIFFER, which is a second, stronger metric - exact integer equality - and
// leaves the cosine gating nothing but the happy path. A `cosine()` that reported 1.0 for every input then kept the
// suite green: the faults were still "caught" by the difference check, so the graded signal had become decoration.
// Every fault below must therefore drop a cosine BELOW kCosineOk, which is what makes the floor falsifiable.
//
// THE TABLE IS THE ANSWER, not just the pass/fail: the first sub-threshold cell names layer and component, and the
// SHAPE of the gradient across layers says which class it is - constant from layer 0 is a wrong input, growing with
// the layer index is an accumulation, appearing mid-context is a race.
//
// THREE MATH PATHS, NOT ONE. The engine reaches the figure by two code paths today - the lead runs
// `qsa_prompt_attn_batch(..., lead_groups)` over its whole pools and multiplies its columns, the partner runs the
// arm's own projections, selection and attention in `peer_qsa_layer` and multiplies the rest - so R and the two
// halves are written as SEPARATE functions. A shared helper would make S == R true by construction and the table
// worthless; separate paths let a mismatched GQA group, a gate applied to the wrong head range, or column blocks
// that fail to tile the input axis show up as a dropped cosine.
//
// WHAT THIS CAN AND CANNOT GRADE. It grades the PROTOCOL (set indexing, the per-set waits, the landing regions),
// the TILING, and the two paths against each other. It does not grade the engine's kernels - those have their own
// parity tests - because the arithmetic here is this file's reading of them. `Drop` runs each coded ordering away:
// a constraint whose removal changes nothing is decoration, and one the model needs but the code lacks is the bug.
#include "basalt/prefill/prefill.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::prefill;

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

/// Below this, a component is not the same tensor any more. See the header for why unity minus a few ulps is the
/// expected neighbourhood when only the order of accumulation differs.
constexpr double kCosineOk = 0.9999;

/// A small deterministic value from the indices that name it. Nothing is generated once and stored: a buffer that
/// exists can be read by the wrong layer without anyone noticing, while a function of position computes whatever
/// the read's position says it should - which is the behaviour a protocol bug has to be visible in.
int64_t bit_val(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return (int64_t)((x ^ (x >> 31)) % 9ull) - 4;   // [-4,4]
}

struct Cfg {
    int n_head, n_head_kv, head_dim, n_embd;
    int lead_heads, partner_heads;      // the shard: lead heads [0, lead_heads), partner [lead_heads, n_head)
    int layers, chunks, T, cap;         // QSA layers per chunk, chunks, tokens per chunk, selectable cells
};

/// The model's own figures at two sizes. `cfg_real` is the width the engine runs at - the case that would catch a
/// stride or a head mapping wrong where it matters, so it runs despite being the slow one. `cfg_small` trades width
/// for layers and chunks, which is where the protocol gets walked deeper and the falsifiers live.
Cfg cfg_real() { return Cfg{24, 2, 256, 2560, 12, 12, 1, 1, 4, 64}; }
Cfg cfg_small() { return Cfg{8, 2, 32, 128, 5, 3, 6, 2, 48, 32}; }

/// Which coded ordering to pretend does not exist. Each mirrors something the engine has; if removing it changes
/// nothing, that ordering is covered by the others - which is a finding about the protocol, not a failure of the
/// instrument, and is said as such.
enum Drop { kDropNone = 0, kDropWireGate, kDropEvRed, kDropStreamOrder, kDropSerialized, kDropFlag };

/// A fault injected into ONE math path, to prove the cosine table can see the class it is graded on. Without these
/// the table could be reporting unity because it cannot see anything else.
enum Fault {
    kFaultNone = 0,
    kFaultGqa,          ///< the partner maps a query head to the wrong KV head (`h % nkv` where the code says `h / rep`)
    kFaultTile,         ///< the two column blocks overlap by a head instead of tiling the input axis
    kFaultStepsChunk,   ///< the partner selects with the chunk's first layer's step counts: the bug `replay_steps` pins
};

int rep(const Cfg& c) { return c.n_head / c.n_head_kv; }              // query heads per KV group
int k_axis(const Cfg& c) { return c.n_head * c.head_dim; }            // attn_output's input axis

// ---- the layer's inputs, each a pure function of the position that reads it ---------------------------------

int64_t q_row(const Cfg&, int l, int t, int k) {                       // the arm's q projection, one element
    return 2 + bit_val((uint64_t)(l * 131 + 1) * 1000003ull + (uint64_t)t * 977 + k);
}
int64_t q_gate(const Cfg&, int l, int h, int d) {                       // the q-norm gate on the attention output
    return bit_val((uint64_t)(l * 577 + 7) * 4093ull + (uint64_t)h * 313 + d);
}
int64_t o_w(const Cfg&, int l, int j, int k) {                          // attn_output, one element
    return bit_val((uint64_t)(l * 9091 + 3) * 8191ull + (uint64_t)j * 7717 + k);
}
int64_t v_row(const Cfg&, int l, int kvh, int64_t cell, int d) {        // a V pool row this card holds
    return bit_val(((uint64_t)(l * 101 + 11)) * 65537ull + (uint64_t)kvh * 4099 + (uint64_t)cell * 263 + d);
}
int64_t pub_mixed(const Cfg&, int l, int t, int i) {                    // the mixed_h row the lead publishes
    return bit_val(((uint64_t)l * 7919 + 23) * 4231ull + (uint64_t)t * 1321 + i);
}
int32_t step_width(const Cfg& c, int l, int t) {                        // how many cells this token attends over
    const int64_t v = bit_val(((uint64_t)l * 613 + 5) * 3001ull + (uint64_t)t * 17);
    return (int32_t)(1 + (v < 0 ? -v : v) % c.cap);
}
/// The cell selection picked for slot s, a function of the layer whose published indexer rows it came from: the
/// arm selects from `q_idx`/`idx_raw` off the wire, so the tag of that set is what decides which cells this card
/// attends over (`qsa_prompt_attn_parity` case 9 is the engine-side proof the two cards' ids agree).
int64_t sel_cell(const Cfg&, int l, int t, int s) {
    const int64_t v = bit_val(((uint64_t)l * 89 + 3) * 6151ull + (uint64_t)t * 409 + s * 23);
    return v < 0 ? -v : v;
}

/// One query element, as BOTH paths compute it: the q projection applied to the mixed row this layer was given.
/// Shared on purpose - the engine's arm multiplies the same `attn_q` rows the lead does, so given the same published
/// bytes the two must agree, and the only thing that can make them disagree is those bytes differing. `src_pub` is
/// the layer whose published rows are being read: the lead reads its own, the partner reads what the wire set holds.
int64_t query_at(const Cfg& c, int l, int t, int k, int src_pub) {
    return q_row(c, l, t, k) + pub_mixed(c, src_pub, t, k % c.n_embd);
}

// ---- the three math paths, deliberately written separately --------------------------------------------------

/// The gated rows for heads [h0, h1) on this card's pools. `src_*` name the layers whose published bytes the query,
/// the widths and the selected cells came from: the lead's are its own layer's, and a partner whose set went stale
/// gets another layer's - which is how a protocol bug becomes an arithmetic one.
void gated_rows(const Cfg& c, int l, int h0, int h1, int src_pub, int src_step, int src_sel, int64_t* g,
                bool gqa_ok = true) {
    for (int t = 0; t < c.T; ++t) {
        const int32_t wid = step_width(c, src_step, t);
        for (int h = h0; h < h1; ++h) {
            const int kvh = gqa_ok ? h / rep(c) : h % c.n_head_kv;   // the GQA mapping the two paths must agree on
            for (int d = 0; d < c.head_dim; ++d) {
                const int k = h * c.head_dim + d;
                int64_t a = 0;
                for (int s = 0; s < wid; ++s)
                    a += query_at(c, l, t, k, src_pub) * v_row(c, l, kvh, sel_cell(c, src_sel, t, s) % c.cap, d);
                g[(size_t)t * k_axis(c) + k] = a * q_gate(c, l, h, d);
            }
        }
    }
}

/// The `attn_output` multiply over columns [k0, k0+nk) of the input axis: what one card's product computes.
void o_project(const Cfg& c, int l, const int64_t* g, int k0, int nk, int64_t* part) {
    for (int t = 0; t < c.T; ++t)
        for (int j = 0; j < c.n_embd; ++j) {
            int64_t acc = 0;
            const int kend = k0 + nk;
            for (int k = k0; k < kend; ++k) acc += g[(size_t)t * k_axis(c) + k] * o_w(c, l, j, k);
            part[(size_t)t * c.n_embd + j] = acc;
        }
}

/// What one run produced, per layer-instance: the two partials as they left their products and the summed o_proj
/// the model goes on to read. Cosine is taken component by component, so a half that drifted while the sum held -
/// or the other way round - is visible instead of cancelling.
struct Run {
    std::vector<int64_t> lead_part, peer_part, summed;
    std::string corrupt;
    bool deadlock = false;
    void size_to(const Cfg& c, int n) {
        const size_t sz = (size_t)n * c.T * c.n_embd;
        lead_part.assign(sz, 0); peer_part.assign(sz, 0); summed.assign(sz, 0);
    }
};

/// R - the whole layer as one card computes it, and the same path's two column blocks so each half has something
/// to be compared against. The two blocks sum to `summed` exactly, which is the tiling property in another form.
Run run_reference(const Cfg& c) {
    Run r;
    const int n = c.layers * c.chunks;
    r.size_to(c, n);
    std::vector<int64_t> g((size_t)c.T * k_axis(c), 0);
    std::vector<int64_t> lp((size_t)c.T * c.n_embd, 0), pp((size_t)c.T * c.n_embd, 0);
    for (int i = 0; i < n; ++i) {
        std::fill(g.begin(), g.end(), 0);
        gated_rows(c, i, 0, c.n_head, i, i, i, g.data());                       // every head, the whole path
        o_project(c, i, g.data(), 0, c.lead_heads * c.head_dim, lp.data());  // its own columns
        o_project(c, i, g.data(), c.lead_heads * c.head_dim, c.partner_heads * c.head_dim, pp.data());
        std::copy(lp.begin(), lp.end(), r.lead_part.begin() + (size_t)i * c.T * c.n_embd);
        std::copy(pp.begin(), pp.end(), r.peer_part.begin() + (size_t)i * c.T * c.n_embd);
        for (size_t e = 0; e < lp.size(); ++e)
            r.summed[(size_t)i * c.T * c.n_embd + e] = lp[e] + pp[e];
    }
    return r;
}

/// The lead's half: every head's attention (Ruling B left its pools whole; halving it is the funded follow-up, not
/// a silent extra), multiplied by ITS columns of attn_output.
void run_lead_half(const Cfg& c, int l, std::vector<int64_t>& part) {
    std::vector<int64_t> g((size_t)c.T * k_axis(c), 0);
    part.assign((size_t)c.T * c.n_embd, 0);
    gated_rows(c, l, 0, c.n_head, l, l, l, g.data());
    o_project(c, l, g.data(), 0, c.lead_heads * c.head_dim, part.data());
}

/// The partner's half: its shard's heads only, on the bytes the sets currently hold, multiplied by the remaining
/// columns. Its query, its attention width and its selected cells come from the published rows, so a stale set -
/// another layer's or another chunk's - moves this card's tensor exactly as it would on the card.
void run_peer_half(const Cfg& c, int l, int src_pub, int src_step, int src_sel, std::vector<int64_t>& part,
                   Fault fault = kFaultNone) {
    std::vector<int64_t> g((size_t)c.T * k_axis(c), 0);
    part.assign((size_t)c.T * c.n_embd, 0);
    gated_rows(c, l, c.lead_heads, c.n_head, src_pub, src_step, src_sel, g.data(), fault != kFaultGqa);
    const int k0 = (fault == kFaultTile ? c.lead_heads - 1 : c.lead_heads) * c.head_dim;
    o_project(c, l, g.data(), k0, c.partner_heads * c.head_dim, part.data());
}

/// One landing region or staging set: whose bytes it holds, and them.
struct Slot {
    int tag = -1;
    std::vector<int64_t> v;
};

/// Replay one prompt's sliced layers. `mode` 1 = serialized, 2 = pipelined; `sets` funds wire, staging and landing
/// depth together (1 at =1, 2 at =2, as `PeerQsaSpan::staging_sets` decides it).
Run run_pair(const Cfg& c, int mode, int sets, Drop drop, Fault fault = kFaultNone) {
    Run r;
    const int n = c.layers * c.chunks;
    r.size_to(c, n);

    const int wi = mode >= 2 ? (int)Prefill::PeerQsaSpan::kWireSets : 1;   // the code's own wire depth
    const int st = mode >= 2 ? sets : 1;
    const int ln = mode >= 2 ? sets : 1;
    auto wset = [&](int i) { return mode >= 2 ? i % wi : 0; };
    auto sset = [&](int i) { return mode >= 2 ? i % st : 0; };
    auto lset = [&](int i) { return mode >= 2 ? i % ln : 0; };

    std::vector<Slot> wire_pub(wi), wire_step(wi), wire_idx(wi);   // the partner's copies of what the lead ships
    std::vector<Slot> own(2 * st), land(2 * ln);                   // each card's staging; the peer's push on this card
    for (auto& s : own) s.v.assign((size_t)c.T * c.n_embd, 0);
    for (auto& s : land) s.v.assign((size_t)c.T * c.n_embd, 0);
    std::vector<int> ev_in(wi, -1), ev_out(wi, -1), ev_red(ln, -1);

    const int LEAD_OPS = 4, PEER_OPS = 3;   // lead: publish, product, push, add. peer: product, push, add.
    std::vector<int> pc(2, 0);
    std::vector<char> lead_add(n, 0), peer_fin(n, 0);
    auto mark = [&](const std::string& what) { if (r.corrupt.empty()) r.corrupt = what; };

    for (int victim = 0; victim < 2 && r.corrupt.empty() && !r.deadlock; ++victim) {
        // Each schedule owns the reported artifacts. Without this the first schedule fills the buffers, the second
        // detects a corrupt read and returns early, and the cosine is graded against the CLEAN pass's data - so a
        // protocol fault that the tags see but the tensors do not reads as unity, which is exactly the blind spot
        // the graded signal exists to close.
        std::fill(r.lead_part.begin(), r.lead_part.end(), 0);
        std::fill(r.peer_part.begin(), r.peer_part.end(), 0);
        std::fill(r.summed.begin(), r.summed.end(), 0);
        std::fill(pc.begin(), pc.end(), 0);
        std::fill(ev_in.begin(), ev_in.end(), -1);
        std::fill(ev_out.begin(), ev_out.end(), -1);
        std::fill(ev_red.begin(), ev_red.end(), -1);
        std::fill(lead_add.begin(), lead_add.end(), (char)0);
        std::fill(peer_fin.begin(), peer_fin.end(), (char)0);
        for (auto& s : wire_pub) s.tag = -1;
        for (auto& s : wire_step) s.tag = -1;
        for (auto& s : wire_idx) s.tag = -1;
        for (auto& s : land) s.tag = -1;
        for (auto& s : own) s.tag = -1;   // a tag left from the other schedule would let a read pass on stale bytes

        auto step = [&](int card) -> int {   // 0 ran, 1 blocked, 2 corrupt read, 3 finished
            const int ops = card == 0 ? LEAD_OPS : PEER_OPS;
            if (pc[card] >= n * ops) return 3;
            const int idx = pc[card], kind = idx % ops, i = idx / ops;
            const int w = wset(i), s = sset(i), lg = lset(i);

            if (card == 0) {
                if (kind == 0) {                                          // the publish
                    if (i > 0 && !lead_add[i - 1]) return 1;              // its own stream: after its own add
                    if (mode == 1 && drop != kDropSerialized && i > 0 && !peer_fin[i - 1]) return 1;
                    if (drop != kDropWireGate) {                          // the set's last user must have finished
                        for (int j = i - 1; j >= 0; --j)
                            if (wset(j) == w) { if (!peer_fin[j]) return 1; break; }
                    }
                    wire_pub[w].tag = wire_step[w].tag = wire_idx[w].tag = i;
                    ev_in[w] = i;
                } else if (kind == 1) {                                   // its product
                    if (ev_in[w] < i) return 1;
                    std::vector<int64_t> p;
                    run_lead_half(c, i, p);
                    own[0 * st + s].v = p;
                    own[0 * st + s].tag = i;
                    std::copy(p.begin(), p.end(), r.lead_part.begin() + (size_t)i * c.T * c.n_embd);
                } else if (kind == 2) {                                   // its push into the peer's landing region
                    if (own[0 * st + s].tag != i) return 1;
                    if (drop != kDropStreamOrder && i >= ln && !lead_add[i - ln]) return 1;
                    land[1 * ln + lg].v = own[0 * st + s].v;
                    land[1 * ln + lg].tag = i;
                } else {                                                  // its add
                    // The peer's push of THIS layer is what the collective's flag waits on. A region still holding
                    // an EARLIER layer's bytes means that push has not been queued: a wait, not a fault (and the
                    // real spin budget turns a permanent one into `timed_out()`, which this scheduler reports as a
                    // deadlock). A region holding a LATER layer's is the corruption - the next collective in this
                    // one overwrote it before this add read it, which is what one landing region cannot survive.
                    if (land[0 * ln + lg].tag > i && drop != kDropFlag) {
                        mark("layer-instance " + std::to_string(i) + ": the lead's add read landing region " +
                             std::to_string(lg) + " holding layer " + std::to_string(land[0 * ln + lg].tag) +
                             "'s partial");
                        return 2;
                    }
                    if (land[0 * ln + lg].tag != i && drop != kDropFlag) return 1;
                    for (size_t e = 0; e < own[0 * st + s].v.size(); ++e)
                        own[0 * st + s].v[e] += land[0 * ln + lg].v[e];
                    std::copy(own[0 * st + s].v.begin(), own[0 * st + s].v.end(),
                              r.summed.begin() + (size_t)i * c.T * c.n_embd);
                    lead_add[i] = 1;
                    ev_red[lg] = i;
                }
            } else {
                if (kind == 0) {                                          // the arm: the publish's proof, then attend
                    if (ev_in[w] < i) return 1;
                    const int sp = wire_pub[w].tag, ss = wire_step[w].tag, si = wire_idx[w].tag;
                    if (sp > i || ss > i || si > i) {                     // overwritten before the arm read it
                        mark("layer-instance " + std::to_string(i) + " read a wire set holding layer " +
                             std::to_string(std::max({sp, ss, si})) + "'s bytes");
                        return 2;
                    }
                    if (sp != i || ss != i || si != i) return 1;          // the publish has not landed: the ev_in wait
                    std::vector<int64_t> p;
                    const int ss_step = fault == kFaultStepsChunk ? (i / c.layers) : ss;
                    run_peer_half(c, i, sp, ss_step, si, p, fault);
                    own[1 * st + s].v = p;
                    own[1 * st + s].tag = i;
                    std::copy(p.begin(), p.end(), r.peer_part.begin() + (size_t)i * c.T * c.n_embd);
                } else if (kind == 1) {                                   // its push, gated by the lead's earlier add
                    if (own[1 * st + s].tag != i) return 1;
                    if (drop != kDropEvRed) {
                        for (int j = i - 1; j >= 0; --j)
                            if (lset(j) == lg) { if (!lead_add[j]) return 1; break; }
                    }
                    land[0 * ln + lg].v = own[1 * st + s].v;
                    land[0 * ln + lg].tag = i;
                } else {                                                  // its add, and the layer's proof
                    if (land[1 * ln + lg].tag > i && drop != kDropFlag) {
                        mark("layer-instance " + std::to_string(i) + ": the peer's add read landing region " +
                             std::to_string(lg) + " holding layer " + std::to_string(land[1 * ln + lg].tag) +
                             "'s partial");
                        return 2;
                    }
                    if (land[1 * ln + lg].tag != i && drop != kDropFlag) return 1;
                    for (size_t e = 0; e < own[1 * st + s].v.size(); ++e)
                        own[1 * st + s].v[e] += land[1 * ln + lg].v[e];
                    peer_fin[i] = 1;
                    ev_out[w] = i;
                }
            }
            ++pc[card];
            return 0;
        };

        while (pc[0] < n * LEAD_OPS || pc[1] < n * PEER_OPS) {
            bool ran = false;
            while (step(1 - victim) == 0) ran = true;        // the other rank charges as far as it can
            const int v = step(victim);
            if (v == 2) return r;
            if (v != 0 && !ran) { r.deadlock = true; return r; }
        }
    }
    return r;
}

// ---- the graded comparison ---------------------------------------------------------------------------------

double cosine(const int64_t* a, const int64_t* b, size_t n) {
    long double dot = 0.0L, na = 0.0L, nb = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const long double x = (long double)a[i], y = (long double)b[i];
        dot += x * y; na += x * x; nb += y * y;
    }
    if (na == 0.0L && nb == 0.0L) return 1.0;   // both empty: nothing diverged
    if (na == 0.0L || nb == 0.0L) return 0.0;   // one side computed nothing: that is the worst divergence
    return (double)(dot / std::sqrt(na * nb));
}

const char* kComponent[] = {"lead partial", "peer partial", "summed o_proj"};

/// The worst cosine a run shows against a reference, over all three components and every layer-instance. Kept
/// separate from the printout so a fault can be graded on the SAME metric the pass/fail rests on: an injected
/// fault that leaves every cosine at unity was not caught by the graded signal, whatever the buffers say.
double worst_cosine(const Cfg& c, const Run& got, const Run& ref) {
    const int n = c.layers * c.chunks;
    const size_t per = (size_t)c.T * c.n_embd;
    double worst = 1.0;
    const std::vector<int64_t>* comp[3] = {&got.lead_part, &got.peer_part, &got.summed};
    const std::vector<int64_t>* base[3] = {&ref.lead_part, &ref.peer_part, &ref.summed};
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < n; ++i)
            worst = std::min(worst, cosine(comp[k]->data() + (size_t)i * per, base[k]->data() + (size_t)i * per, per));
    return worst;
}

/// One run's three components against a reference's, per layer-instance: the table. Returns the first cell that
/// fell below the threshold, or -1.
int table(const Cfg& c, const Run& got, const Run& ref, const char* what) {
    const int n = c.layers * c.chunks;
    const size_t per = (size_t)c.T * c.n_embd;
    std::printf("  %s: per-layer cosine against the reference\n", what);
    std::printf("    %-14s", "component");
    for (int i = 0; i < n; ++i) std::printf("%12s", ("layer " + std::to_string(i)).c_str());
    std::printf("\n");
    int first_bad = -1;
    const std::vector<int64_t>* comp[3] = {&got.lead_part, &got.peer_part, &got.summed};
    const std::vector<int64_t>* base[3] = {&ref.lead_part, &ref.peer_part, &ref.summed};
    for (int k = 0; k < 3; ++k) {
        std::printf("    %-14s", kComponent[k]);
        for (int i = 0; i < n; ++i) {
            const double cs = cosine(comp[k]->data() + (size_t)i * per, base[k]->data() + (size_t)i * per, per);
            const bool ok = cs >= kCosineOk;
            std::printf("%12.8f%s", cs, ok ? "" : "*");
            if (!ok && first_bad < 0) {
                first_bad = i;
                std::printf("\n      FIRST BELOW THRESHOLD: %s at layer-instance %d, cosine %.10f (floor %.4f)\n",
                            kComponent[k], i, cs, kCosineOk);
                std::printf("    %-14s", "");
                for (int j = i + 1; j < n; ++j) std::printf("%12s", "");
            }
        }
        std::printf("\n");
    }
    return first_bad;
}

bool bitwise(const std::vector<int64_t>& a, const std::vector<int64_t>& b, size_t per, int n, int& layer,
             size_t& elem) {
    for (int i = 0; i < n; ++i)
        for (size_t e = 0; e < per; ++e)
            if (a[(size_t)i * per + e] != b[(size_t)i * per + e]) { layer = i; elem = e; return false; }
    return true;
}

}  // namespace

int main() {
    for (int which = 0; which < 2; ++which) {
        const Cfg c = which ? cfg_real() : cfg_small();
        const int n = c.layers * c.chunks;
        std::printf("%s geometry: %d heads x %d dim, %d cols, %d layer-instance(s), %d tokens, shard %d/%d\n",
                    which ? "REAL" : "scaled", c.n_head, c.head_dim, c.n_embd, n, c.T, c.lead_heads,
                    c.partner_heads);

        const Run R = run_reference(c);
        size_t nz = 0;
        for (int64_t v : R.summed) if (v != 0) ++nz;
        CHECK(nz * 2 > R.summed.size());   // a reference that computed nothing would agree with anything

        const Run S = run_pair(c, 1, 1, kDropNone);
        const Run P = run_pair(c, 2, 2, kDropNone);
        CHECK(!S.deadlock);
        CHECK(!P.deadlock);
        if (S.deadlock || P.deadlock) { std::printf("  DEADLOCK in the protocol replay\n"); ++g_fail; }
        if (!S.corrupt.empty()) { std::printf("  S: %s\n", S.corrupt.c_str()); ++g_fail; }
        if (!P.corrupt.empty()) { std::printf("  P: %s\n", P.corrupt.c_str()); ++g_fail; }

        // (1) the slice's math against the whole layer, and (2) the pipelined ordering against the serialized one.
        if (table(c, S, R, "S vs R") >= 0) ++g_fail;
        if (table(c, P, R, "P vs R") >= 0) ++g_fail;
        if (table(c, P, S, "P vs S") >= 0) ++g_fail;

        // (3) the stronger assertion the CPU replay affords: on this replay nothing is nondeterministic, so P must
        // equal S to the element, not merely directionally.
        const size_t per = (size_t)c.T * c.n_embd;
        const std::vector<int64_t>* gp[3] = {&P.lead_part, &P.peer_part, &P.summed};
        const std::vector<int64_t>* gs[3] = {&S.lead_part, &S.peer_part, &S.summed};
        for (int k = 0; k < 3; ++k) {
            int layer = -1; size_t elem = 0;
            if (!bitwise(*gp[k], *gs[k], per, n, layer, elem)) {
                std::printf("  P != S BITWISE in %s at layer-instance %d element %zu: %lld vs %lld\n",
                            kComponent[k], layer, elem, (long long)(*gp[k])[(size_t)layer * per + elem],
                            (long long)(*gs[k])[(size_t)layer * per + elem]);
                ++g_fail;
            }
        }

        if (which == 0) {
            auto differs = [&](const Run& x, const Run& y) {
                return !x.corrupt.empty() || x.deadlock || x.lead_part != y.lead_part ||
                       x.peer_part != y.peer_part || x.summed != y.summed;
            };
            // (a) the coded orderings, each run away. An ordering nothing else covers must be load-bearing: if the
            // table is unchanged the ordering is redundant, which is reported as what it is rather than scored as
            // a pass or a failure - the protocol having four guards where two suffice is a fact, not a bug.
            struct F { Drop d; const char* why; };
            const F fs[] = {
                {kDropWireGate, "no gate on the wire set's previous user"},
                {kDropEvRed, "no ev_red before a push into a landing region"},
                {kDropStreamOrder, "the lead's push not behind its own earlier add"},
                {kDropSerialized, "the =1 end-of-layer wait removed"},
                {kDropFlag, "the collective's flag handshake: an add reads whatever its region holds"},
            };
            int load_bearing = 0;
            for (const F& f : fs) {
                const Run x = run_pair(c, 2, 2, f.d);
                const bool caught = differs(x, P);
                std::printf("  ordering, %s: %s\n", f.why,
                            caught ? "load-bearing (removing it moves the table)" : "redundant (the others cover it)");
                if (caught) ++load_bearing;
            }
            std::printf("  (%d of %d orderings the model relies on are load-bearing)\n", load_bearing, (int)(sizeof(fs) / sizeof(fs[0])));
            // The control on the scheduler itself: with the flag handshake dropped, an add reads the region as it
            // finds it and the overrun must become visible. If THAT one is also clean, the model is not exercising
            // the race at all and every "redundant" above would be an artifact of a scheduler that waits its way
            // to correctness - so the run is only trusted when this one is caught. It is graded on the COSINE,
            // not on the buffers, so a `cosine()` blind to a shifted sum cannot keep this green.
            {
                const Run x = run_pair(c, 2, 2, kDropFlag);
                const double wc = worst_cosine(c, x, P);
                std::printf("  control, the flag handshake dropped: worst cosine %.8f (floor %.4f)\n", wc, kCosineOk);
                CHECK(wc < kCosineOk);
            }

            // (b) the faults the table must NEVER let through: each breaks one path in a way a real bug takes, and
            // each must land a cosine BELOW the floor. An instrument that reports unity through these is decoration
            // - and this is the check that makes the graded signal itself falsifiable rather than a printout.
            struct G { Fault f; const char* why; };
            const G gs[] = {
                {kFaultGqa, "partner maps its heads to the wrong KV head"},
                {kFaultTile, "the two column blocks overlap by one head"},
                {kFaultStepsChunk, "the partner selects with the chunk's first layer's step counts"},
            };
            for (const G& g : gs) {
                const Run x = run_pair(c, 2, 2, kDropNone, g.f);
                const double wc = worst_cosine(c, x, R);
                const bool caught = wc < kCosineOk;
                std::printf("  fault, %s: %s (worst cosine %.8f)\n", g.why, caught ? "caught" : "NOT CAUGHT", wc);
                if (!caught) ++g_fail;
            }

            // (c) the two protocol shapes this branch has already paid for once, likewise graded on the cosine.
            const Run one = run_pair(c, 2, 1, kDropNone);
            const double wc_one = worst_cosine(c, one, R);
            std::printf("  fault, pipelined with one staging set: %s (worst cosine %.8f)\n",
                        wc_one < kCosineOk ? "caught" : "NOT CAUGHT", wc_one);
            if (!(wc_one < kCosineOk)) ++g_fail;
        }
    }
    std::printf("qsa layer diff: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}

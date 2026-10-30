#pragma once
// The per-layer chain: a fingerprint of every component of one sliced QSA layer's data path, taken as the layer
// completes, so three boots of one prompt - gate off, `tp_prompt_attn=1`, `tp_prompt_attn=2` - each leave behind a
// chain that can be diffed on the CPU.  The pair-agreement check answers one question (did the two cards sum the
// same two partials); it is silent when both cards hold the SAME wrong value, which is the shape a wrong input
// makes.  A chain is not silent: the first layer whose fingerprint moves names the layer, and the first component
// whose fingerprint moves names the buffer.  The shape of the divergence is the diagnosis - constant from the
// first QSA layer is a wrong input, growing with the layer index is an accumulation, appearing mid-context is a
// race - so the three boots answer what a hypothesis-at-a-time hunt cannot.
//
// The rules below are read by the kernel that takes the fingerprint and by the host that grades it, so they are
// spelled once and compiled both ways (`tp2_collective.hpp` does the same for its sample rule): two spellings of
// the same rule is one way for a chain to match for the wrong reason.

#include <cstdint>
#include <string>

#if defined(__CUDACC__)
#define QSA_CHAIN_HD __host__ __device__
#else
#define QSA_CHAIN_HD
#endif

namespace basalt::kernels {

/// Rows fingerprinted out of any buffer, spread over its rows.  A window this small keeps the instrument to
/// kilobytes per layer, which is what lets it stay in a boot's log rather than needing somewhere to live.
constexpr int kChainRows = 32;
/// The block the reduction runs in.  Both the kernel and the host reference reduce in this order, which is why
/// the sum is reproducible: an FP add in a fixed tree is deterministic, an atomic one is not.
constexpr int kChainBlock = 256;
/// What a slot holds when nobody recorded it.  A component a mode never reached must read as absent in the diff,
/// never as whatever bytes the buffer happened to keep from the previous chunk.
constexpr uint32_t kChainUnrecorded = 0xFFFFFFFFu;

/// The components, in the order the data path walks them.  The first eight exist in EVERY mode - the lead's own
/// path is the reference the other two chains are diffed against - and they are what makes a gate-off boot worth
/// taking.  The rest belong to the pair and only appear when a layer is sliced.
enum QsaChainSlot : int {
    // ---- the lead's card, recorded whether or not the slice is on
    kChainMix = 0,     // the FP16 mixed rows the layer's projections read: what the arm is given to work with
    kChainSteps,       // the chunk's shipped step counts, as the lead holds them
    kChainSel,         // the lead's selection ids
    kChainKPool,       // the lead's K pool rows the attention reads
    kChainVPool,       // the lead's V pool rows
    kChainAttn,        // the lead's attention output, before the gate
    kChainGated,       // the gated rows the lead's `o_proj` multiplies
    kChainPartL,       // the lead's `o_proj` partial, before the collective (sliced layers only)
    kChainSumL,        // the lead's buffer after the collective (sliced layers only)
    kChainBo,          // the summed `o_proj` as the model reads it: the figure all three modes must agree on
    // ---- the partner's card, only when the layer went through the arm
    kChainPub,         // the mixed rows as the PARTNER holds them: the lead's `kChainMix` must equal this
    kChainStepsR,      // the step counts as the partner holds them
    kChainQIdxR,       // the indexer q rows the partner was given
    kChainIdxRaw,      // the indexer k rows the partner was given
    kChainKSpan,       // the partner's own K pool rows - the imported span it attends over
    kChainVSpan,       // the partner's own V pool rows
    kChainQ,           // the Q rows the partner's attention consumed
    kChainSelR,        // the partner's selection ids
    kChainAttnP,       // the partner's attention output, before the gate
    kChainGatedP,      // the gated rows the partner's `o_proj` multiplies
    kChainPartP,       // the partner's `o_proj` partial, before the collective
    kChainSumP,        // the partner's buffer after the collective: must equal `kChainSumL`
    kChainCount
};

/// One component's fingerprint.  Three figures, because they answer different questions: `hash` says whether the
/// bits moved at all, `sum` says by how much when they did (which is what separates a rounding drift from a wrong
/// buffer), and `maxabs` catches the scale errors and the overflow a cancelling sum would hide.
struct QsaChainFp {
    uint32_t hash = kChainUnrecorded;
    float sum = 0.0f;
    float maxabs = 0.0f;
};

/// The slot's name as it appears in the log, and the word the CPU diff greps for.  One table, so a line and a
/// grep can drift apart only by editing this file.
const char* qsa_chain_slot_name(int slot);

/// How many of a buffer's rows the fingerprint covers.
QSA_CHAIN_HD inline int64_t qsa_chain_sampled(int64_t rows) {
    return rows <= 0 ? 0 : (rows < kChainRows ? rows : kChainRows);
}

/// Row `k` of the sampled window out of `rows`, spread so the first and last row are both in it.  Negative when
/// `k` is past the window - the caller's loop bound is `qsa_chain_sampled(rows)`.
QSA_CHAIN_HD inline int64_t qsa_chain_row(int k, int64_t rows) {
    if (k < 0 || rows <= 0 || k >= qsa_chain_sampled(rows)) return -1;
    return rows <= kChainRows ? k : (rows - 1) * k / (kChainRows - 1);
}

/// One element's contribution: its bits mixed with its position, so a buffer whose values were permuted between
/// boots fingerprints differently rather than agreeing by coincidence.
QSA_CHAIN_HD inline uint32_t qsa_chain_mix(uint32_t bits, uint32_t row, uint32_t col) {
    uint32_t x = bits ^ ((row * 0x9E3779B9u) + col * 0x85EBCA6Bu);
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 13;
    return x;
}

/// The largest of two absolute values.  Written so a NaN wins from either side and stays winning: an overflowed
/// component must read as broken in the chain, not as the largest finite thing sitting beside it.
QSA_CHAIN_HD inline float qsa_chain_max(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    return (b > a) ? b : a;
}

/// The reference the device kernel is graded against: the fingerprint of a host-side window of `rows` rows of
/// `stride_words` words.  Same sampling, same block width, same reduction order as the kernel, spelled here so
/// `qsa_chain_parity` can hold both and compare.
QsaChainFp qsa_chain_ref(const uint32_t* words, int64_t rows, int64_t stride_words);

/// Fingerprint `rows` rows of `stride_words` 4-byte words at `base` into `dst`, on `stream`.  `base` must be
/// 4-byte aligned (everything this engine fingerprints comes off `cudaMalloc`, which over-aligns); anything else
/// is refused rather than read unaligned.  The caller's device must be current, as everywhere in the arm.
bool qsa_chain_record(QsaChainFp* dst, const void* base, int64_t rows, int64_t stride_words, void* stream,
                      std::string& err);

/// One slot as it appears in a chain line: `mix=9f3a12c4/0.12345679/1.5`, or `mix=--------` when the slot was
/// never recorded.  Fixed width hex so two lines diff by eye.
std::string qsa_chain_field(int slot, const QsaChainFp& fp);

}  // namespace basalt::kernels

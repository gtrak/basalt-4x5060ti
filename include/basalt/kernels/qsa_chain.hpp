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

#include <cmath>
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
/// Projections per component in the direction sketch below.  A fixed-count sketch trades estimator error for log
/// size: the cosine it supports carries a relative error of about `1/sqrt(kChainProj)`, which at 32 is ~18%.  That
/// is enough to separate the two cases the hunt asks about - a same-distribution pair sits within a few per cent of
/// 1, an unrelated tensor sits near 0 and many sigma away - and not enough to read a cosine as a measurement.
constexpr int kChainProj = 32;
/// Projections reduced per pass of the tree.  Bounds the kernel's shared-memory footprint; must divide
/// `kChainProj`, which `qsa_chain_parity` asserts.
constexpr int kChainProjChunk = 8;
/// What `qsa_chain_cosine` returns for a pair it cannot grade - an unrecorded side, or one whose sketch is empty.
/// Deliberately outside `[-1, 1]` so a caller cannot read it as a poor score rather than as no answer.
constexpr float kChainCosineNa = 2.0f;

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
    kChainColsL,       // the lead's PACKED `attn_output` columns, whole: the W its product dequantizes (sliced only)
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
    kChainColsP,       // the partner's PACKED `attn_output` columns, whole: the W ITS product dequantizes
    kChainPartP,       // the partner's `o_proj` partial, before the collective
    kChainSumP,        // the partner's buffer after the collective: must equal `kChainSumL`
    kChainCount
};

// THE TWO COLUMN SLOTS are the only weights in the chain - everything else is an activation - and they are
// fingerprinted WHOLE (`rows` = 1, stride the whole image) and printed as bits only. One bad block poisons every
// element of the partial a product feeds (its sum runs over the whole input axis), so a row sample could report
// the image innocent beside a partial of nan; and their float figures are meaningless, which
// `qsa_chain_slot_is_bytes` explains where it is enforced.

/// One component's fingerprint.  Three figures, because they answer different questions: `hash` says whether the
/// bits moved at all, `sum` says by how much when they did (which is what separates a rounding drift from a wrong
/// buffer), and `maxabs` catches the scale errors and the overflow a cancelling sum would hide.
///
/// THE PROJECTION SKETCH, and why the three figures above are not enough for the cross-boot question. A hash
/// answers a yes/no question about exact bits, so across boots it is useless here: the engine's prefill kernels
/// accumulate in a schedule-dependent order, two passes of ONE boot already disagree, and a cross-boot hash diff
/// compares noise against noise (the retraction in `progress-tp2slice-verdict.md` §8). What survives that is a
/// DIRECTION, and a direction is exactly what the corruptions this hunt is about change - a stale wire set, an
/// imported span cell landing where the attention does not look, a column block that fails to tile: the tensor keeps
/// its scale but points elsewhere. So each component also carries `kChainProj` fixed projections of its sampled
/// window against a fixed set of Rademacher signs (`qsa_chain_sign`): a Johnson-Lindenstrauss sketch, spelled once
/// in this header so the kernel that fills it and the host that reads it cannot disagree. A cosine between two
/// components then comes from the two log lines alone - no buffer crosses to the host, which is what keeps this a
/// log line rather than a dump. Read it as shape, never as bits: `hash` remains the only claim about exactness, and
/// it stays good only WITHIN one boot and one pass.
struct QsaChainFp {
    uint32_t hash = kChainUnrecorded;
    float sum = 0.0f;
    float maxabs = 0.0f;
    /// The sketch: `proj[p] = sum over the sampled window of value * qsa_chain_sign(p, row, col)`. Zero when
    /// unrecorded, which is why `hash` (not these) is the unrecorded test.
    float proj[kChainProj] = {};
};

/// The Rademacher sign of projection `p` on element (`row`, `col`). Any cheap, well-mixed function of the three
/// indices would do; this one is the value-hash above WITHOUT the value's bits, so a projection cannot accidentally
/// weight itself. Independent across `p`, which is what the error bound assumes.
QSA_CHAIN_HD inline float qsa_chain_sign(int p, uint32_t row, uint32_t col) {
    uint32_t x = (row * 0x27D4EB2Du) ^ (col * 0x165667B1u) ^ (uint32_t)(p * 0x9E3779B9u);
    x ^= x >> 15; x *= 0x2545F491u;
    x ^= x >> 13; x *= 0x9E3779B1u;
    x ^= x >> 16;
    return (x & 1u) ? 1.0f : -1.0f;
}

/// The cosine of two components' sampled windows, read off their sketches. Returns `kChainCosineNa` when either side
/// was never recorded or one of them has no direction, so an absent component is never mistaken for an orthogonal
/// one - the failure mode that would make a mode which skipped a layer read as the corruption.
QSA_CHAIN_HD inline float qsa_chain_cosine(const QsaChainFp& a, const QsaChainFp& b) {
    if (a.hash == kChainUnrecorded || b.hash == kChainUnrecorded) return kChainCosineNa;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int p = 0; p < kChainProj; ++p) {
        dot += (double)a.proj[p] * (double)b.proj[p];
        na += (double)a.proj[p] * (double)a.proj[p];
        nb += (double)b.proj[p] * (double)b.proj[p];
    }
    if (na <= 0.0 || nb <= 0.0) return kChainCosineNa;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

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

// THE TWO COLUMN SLOTS FINGERPRINT OPAQUE BYTES, not an array of floats, so only their `hash` is a claim about
// anything.  A packed Q6_K image is codes and quantized scales: read as fp32 words it contains NaN bit patterns
// in every image that ever was packed correctly, so its `sum`, its `maxabs` and its sketch are garbage BY
// CONSTRUCTION - a finite `partp` beside a nan `colsp` is exactly what a correct pair prints, and reading that
// nan as "unwritten memory" sends the hunt to the wrong card.  `qsa_chain_field` prints those slots hash-only,
// and `qsa_chain_record` is asked not to compute the sketch for them: the walk that produces the hash is the
// only one that has to happen.  A slot that cannot say "the bytes are wrong" AND "the bytes are right" is worse
// than no slot, because the first reading of the second answer is the one that wastes the boot.
inline bool qsa_chain_slot_is_bytes(int slot) {
    return slot == kChainColsL || slot == kChainColsP;
}

/// What the whole-buffer byte slots hash, computed where the bytes were made.  Exactly the window the kernel
/// accumulates for `rows` = 1 - same mix, same positions, and integer addition is associative so the block split
/// cannot move it - which makes a packer's own figure and the device-side slot comparable across a boot.  Equal
/// figures mean the card holds the bytes that were packed; unequal ones mean something between them is lying.
inline uint32_t qsa_chain_bytes_hash(const uint32_t* words, int64_t words_count) {
    uint32_t h = 0;
    for (int64_t c = 0; c < words_count; ++c) h += qsa_chain_mix(words[c], 0u, (uint32_t) c);
    return h;
}

/// Fingerprint `rows` rows of `stride_words` 4-byte words at `base` into `dst`, on `stream`.  `base` must be
/// 4-byte aligned (everything this engine fingerprints comes off `cudaMalloc`, which over-aligns); anything else
/// is refused rather than read unaligned.  `sketch` is the projection pass, which only a component made of floats
/// has a use for; with it off `proj` is left zero and the window is walked once.  The caller's device must be
/// current, as everywhere in the arm.
bool qsa_chain_record(QsaChainFp* dst, const void* base, int64_t rows, int64_t stride_words, void* stream,
                      std::string& err, bool sketch = true);

/// One slot as it appears in a chain line: `mix=9f3a12c4/0.12345679/1.5`, or `mix=--------` when the slot was
/// never recorded.  Fixed width hex so two lines diff by eye.
std::string qsa_chain_field(int slot, const QsaChainFp& fp);

}  // namespace basalt::kernels

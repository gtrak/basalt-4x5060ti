// tests/kernels/qsa_chain_parity.cpp - the per-layer chain's fingerprint against its CPU reference: the sampling
// rule, what the fingerprint is sensitive to, the refusals, and the device kernel against the host rule at the
// widths the arm really uses.  Bitwise, not a tolerance: the chain's whole worth is that two boots of one prompt
// print the same figures unless the bytes moved, so a fingerprint that drifted by one ulp between the kernel and
// the reference would be an instrument that reports a corruption that is not there.
//
// The rules are in the header precisely so this file can hold both spellings and compare them; a kernel-only test
// would leave the host reference ungraded.  The device cases need one card; with none they exit 77, which ctest
// reads as a skip (the label is `gpu`; the contract above runs regardless).
#include "basalt/kernels/qsa_chain.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

using namespace basalt::kernels;

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

/// Bit pattern of a float, so the cases below can build values by hand without aliasing.
uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

/// The widths the chain is asked for in a real boot: the FP16 mixed rows as words (n_embd 2560 / 2), the lead's
/// whole attention (24 heads x 256), the partner's shard (12 x 256), a chunk's step counts, and a decode row.
struct Shape {
    int64_t rows, stride;
    const char* what;
};
const Shape kShapes[] = {
    {8192, 1280, "the mixed rows of a ledger chunk"},
    {8192, 6144, "the lead's attention, 24 heads"},
    {8192, 3072, "the partner's shard, 12 heads"},
    {8192, 8, "a chunk's step counts"},
    {128, 1280, "a short chunk"},
    {1, 1280, "one row"},
    {7, 3, "a shape no width produces, to catch a boundary"},
};

std::vector<float> make_window(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-4.0f, 4.0f);
    std::vector<float> v((size_t) n);
    for (float& x : v) x = d(rng);
    return v;
}

// ---- the sampling rule ------------------------------------------------------------------------------

void sampling_contract() {
    for (const int64_t rows : {1LL, 2LL, 3LL, 31LL, 32LL, 33LL, 100LL, 4096LL, 8192LL}) {
        const int64_t sampled = qsa_chain_sampled(rows);
        CHECK(sampled == std::min(rows, (int64_t) kChainRows));
        int64_t prev = -1;
        for (int64_t k = 0; k < sampled; ++k) {
            const int64_t r = qsa_chain_row((int) k, rows);
            CHECK(r >= 0 && r < rows);          // in bounds
            CHECK(r > prev); prev = r;          // strictly increasing, so no row is taken twice
        }
        CHECK(qsa_chain_row(0, rows) == 0);                       // the first row is always in the window
        CHECK(qsa_chain_row((int) (sampled - 1), rows) == rows - 1);   // and the last one, so a torn tail shows
    }
    CHECK(qsa_chain_sampled(0) == 0);
    CHECK(qsa_chain_sampled(-5) == 0);
    CHECK(qsa_chain_row(-1, 10) < 0);
    CHECK(qsa_chain_row(10, 10) < 0);      // past the window, not wrapped into it
    CHECK(qsa_chain_row(0, 0) < 0);
}

// ---- the reference's own behaviour -------------------------------------------------------------------

void reference_contract() {
    // A window whose sum and maximum are known exactly: small integers, so the FP add has nothing to round.
    std::vector<uint32_t> w(2 * 4);
    const float vals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < 8; ++i) w[i] = bits(vals[i]);
    const QsaChainFp fp = qsa_chain_ref(w.data(), 2, 4);
    CHECK(fp.hash != kChainUnrecorded);
    CHECK(fp.sum == 36.0f);
    CHECK(fp.maxabs == 8.0f);

    // Deterministic: the same bytes fingerprint the same way, twice over.
    CHECK(qsa_chain_ref(w.data(), 2, 4).hash == fp.hash);

    // One bit of one element moves it.  A fingerprint that survived this would be a checksum of nothing.
    auto changed = w; changed[5] ^= 0x1u;
    CHECK(qsa_chain_ref(changed.data(), 2, 4).hash != fp.hash);

    // The same values in a different order move it too - the two rows swapped, and the columns of one row rotated.
    // A layer whose rows were written in the wrong order is exactly what a wrong row mapping looks like, and a
    // position-blind hash would call it identical.
    std::vector<uint32_t> sw = w;
    for (int c = 0; c < 4; ++c) { sw[(size_t) c] = w[4 + c]; sw[4 + (size_t) c] = w[c]; }
    CHECK(qsa_chain_ref(sw.data(), 2, 4).hash != fp.hash);
    std::vector<uint32_t> rot = w;
    for (int c = 0; c < 4; ++c) rot[(size_t) c] = w[(c + 1) % 4];
    CHECK(qsa_chain_ref(rot.data(), 2, 4).hash != fp.hash);

    // A NaN is reported as broken rather than skipped: `maxabs` carries it, since a cancelling sum can hide one.
    std::vector<uint32_t> nanw(8, 0);
    nanw[3] = bits(std::nanf(""));
    const QsaChainFp nf = qsa_chain_ref(nanw.data(), 2, 4);
    CHECK(std::isnan(nf.maxabs));
    // A negative value contributes its magnitude, and zeros fingerprint as zeros rather than as absent.
    std::vector<uint32_t> zw(8, 0);
    const QsaChainFp zf = qsa_chain_ref(zw.data(), 2, 4);
    CHECK(zf.sum == 0.0f && zf.maxabs == 0.0f);
    CHECK(zf.hash != kChainUnrecorded);
    std::vector<uint32_t> negw(8, bits(-2.5f));
    CHECK(qsa_chain_ref(negw.data(), 2, 4).maxabs == 2.5f);

    // Nothing to fingerprint reads as absent, not as a fingerprint of nothing.
    CHECK(qsa_chain_ref(nullptr, 2, 4).hash == kChainUnrecorded);
    CHECK(qsa_chain_ref(w.data(), 0, 4).hash == kChainUnrecorded);
    CHECK(qsa_chain_ref(w.data(), 2, 0).hash == kChainUnrecorded);

    // Rows outside the sampled window are not read at all, so changing one cannot move the figure.  Which rows are
    // in the window is the rule's business, so the case asks the rule rather than assuming: one it provably does
    // not select must be invisible, one it does must not be.
    std::vector<uint32_t> big(200 * 2, bits(0.5f));
    const QsaChainFp before = qsa_chain_ref(big.data(), 200, 2);
    int64_t unselected = -1;
    for (int64_t r = 1; r < 200; ++r) {
        bool in = false;
        for (int64_t k = 0; k < qsa_chain_sampled(200); ++k)
            if (qsa_chain_row((int) k, 200) == r) in = true;
        if (!in) { unselected = r; break; }
    }
    CHECK(unselected > 0);
    big[(size_t) unselected * 2 + 1] = bits(97.25f);
    CHECK(qsa_chain_ref(big.data(), 200, 2).hash == before.hash);
    big[(size_t) qsa_chain_row(5, 200) * 2 + 1] = bits(13.5f);
    CHECK(qsa_chain_ref(big.data(), 200, 2).hash != before.hash);
}

// ---- the sketch: the figures a cross-boot cosine is read from ---------------------------------------------
//
// The hash above cannot answer a cross-boot question (the engine's own kernels are not reproducible between runs,
// so a cross-boot hash diff is noise against noise - the retraction in progress-tp2slice-verdict.md §8). The
// sketch is the instrument that can: a cosine taken between two boots' log lines, with no buffer crossing to the
// host. An ungraded sketch would be the worst kind of instrument, because a cosine of 1.0 read off a broken sketch
// would be believed exactly as much as a correct one, so every property below is asserted AND falsified.

/// A window of `rows` x `stride` words whose values are a smooth ramp plus a deterministic wobble: a cosine
/// against a perturbed copy of it is a meaningful figure rather than a coincidence of zeros.
std::vector<uint32_t> ramp_window(int64_t rows, int64_t stride, uint32_t seed) {
    std::vector<uint32_t> v((size_t) rows * stride);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < stride; ++c) {
            const float x = (float)(r * 0.125 + c * 0.03125) + (float)((r * 7 + c * 13 + seed) % 11) * 0.05f;
            v[(size_t) r * stride + c] = bits(x);
        }
    return v;
}

/// A zero-mean, structureless window: no shared trend with `ramp_window`, so the two are close to orthogonal in
/// the direction the sketch measures. Two smooth ramps of different seeds would NOT do - they share their
/// dominant low-frequency trend, and a cosine near 1 between them is the sketch working, not failing.
std::vector<uint32_t> noise_window(int64_t rows, int64_t stride, uint32_t seed) {
    std::vector<uint32_t> v((size_t) rows * stride);
    uint32_t x = seed * 2654435761u + 1u;
    for (auto& e : v) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        e = bits(((x >> 8) % 2001u < 1000u) ? -1.0f : 1.0f);
    }
    return v;
}

void sketch_contract() {
    // The chunking is an implementation detail the kernel and the reference must agree on; if it stopped dividing
    // the sketch, projections would be written out of bounds and the cosine would silently read zeros.
    CHECK(kChainProj > 1);
    CHECK(kChainProj % kChainProjChunk == 0);

    // A component nobody recorded has no direction to have: `NA` out of range, never a 0 that reads as orthogonal
    // and never a 1 that reads as agreement.
    QsaChainFp blank, one;
    one.hash = 0x12345678u;
    for (int p = 0; p < kChainProj; ++p) one.proj[p] = (float)(p % 7) - 3.0f;
    CHECK(qsa_chain_cosine(blank, blank) == kChainCosineNa);
    CHECK(qsa_chain_cosine(one, blank) == kChainCosineNa);
    CHECK(qsa_chain_cosine(blank, one) == kChainCosineNa);
    // A recorded component whose sketch is empty (all projections zero) is also not a direction.
    QsaChainFp empty_sketch;
    empty_sketch.hash = 0xabcdef01u;
    CHECK(qsa_chain_cosine(empty_sketch, empty_sketch) == kChainCosineNa);

    const int64_t rows = 64, stride = 40;   // 64 rows so the sampling actually thins, as it does at real widths
    const std::vector<uint32_t> base = ramp_window(rows, stride, 0);
    const QsaChainFp a = qsa_chain_ref(base.data(), rows, stride);
    CHECK(a.hash != kChainUnrecorded);

    // (1) the same window against itself is exactly 1. Both sides are the same computation on the same bytes, so
    // this is not a tolerance - it is the anchor the other two cases are read against.
    CHECK(std::fabs(qsa_chain_cosine(a, qsa_chain_ref(base.data(), rows, stride)) - 1.0f) < 1e-5f);

    // (2) the sketch is not constant: a window with no shared structure must NOT read as the same direction.
    // Without this, a sketch left at zero by a broken fill would score 1.0 on everything and the whole instrument
    // would be decoration - this is the falsifier that makes (1) mean something. `noise_window` is zero-mean and
    // structureless, so against the ramp it is near orthogonal; the tolerance is loose because the estimate
    // carries the `1/sqrt(kChainProj)` the header quotes, and this case asks only for "clearly not 1".
    const std::vector<uint32_t> other = noise_window(rows, stride, 5);
    const QsaChainFp b = qsa_chain_ref(other.data(), rows, stride);
    CHECK(a.hash != b.hash);
    const float c_far = qsa_chain_cosine(a, b);
    CHECK(c_far < 0.5f);
    // And the sketch is not all zeros on a real window, which is what would make every cosine read NA or 1.
    bool any_nonzero = false;
    for (int p = 0; p < kChainProj; ++p) any_nonzero |= a.proj[p] != 0.0f;
    CHECK(any_nonzero);

    // (3) a small perturbation keeps the direction. This is the property the hunt actually relies on: reduction-
    // order noise between boots must NOT move the cosine, while a substituted tensor must.
    std::vector<uint32_t> nudged = base;
    for (size_t i = 0; i < nudged.size(); i += 3)
        nudged[i] = bits(std::bit_cast<float>(nudged[i]) * 1.0002f + 1e-6f);
    const float c_near = qsa_chain_cosine(a, qsa_chain_ref(nudged.data(), rows, stride));
    CHECK(c_near > 0.999f);

    // (4) half the window zeroed - a mild version of the class the =2 corruption is (a stale wire set, another
    // layer's rows) - must be visible.
    std::vector<uint32_t> half = base;
    for (size_t i = 0; i < half.size(); i += 2) half[i] = bits(0.0f);
    const float c_half = qsa_chain_cosine(a, qsa_chain_ref(half.data(), rows, stride));
    CHECK(c_half < 0.99f);

    // (5) the sign rule must depend on all three of its arguments, or the projections are not independent and the
    // quoted error is a fiction.
    CHECK(qsa_chain_sign(0, 3, 4) != qsa_chain_sign(1, 3, 4) ||
          qsa_chain_sign(0, 3, 4) != qsa_chain_sign(0, 4, 4) ||
          qsa_chain_sign(0, 3, 4) != qsa_chain_sign(0, 3, 5));
    int plus = 0;
    for (int p = 0; p < kChainProj; ++p) plus += qsa_chain_sign(p, 7, 9) > 0.0f ? 1 : 0;
    CHECK(plus * 3 > kChainProj && plus * 3 < kChainProj * 2);   // roughly balanced, not a constant

    // (6) the sketch survives the line: a field's sketch round-trips, so the boot's log is enough to rebuild a
    // cosine and no buffer has to be shipped off the card.
    const std::string line = qsa_chain_field(kChainMix, a);
    size_t commas = 0;
    for (char ch : line) if (ch == ',') ++commas;
    CHECK(commas == (size_t)kChainProj - 1);
}

// ---- the names and the line ---------------------------------------------------------------------------

void names_and_fields() {
    std::vector<std::string> seen;
    for (int s = 0; s < kChainCount; ++s) {
        const std::string n = qsa_chain_slot_name(s);
        CHECK(!n.empty() && n != "?");
        CHECK(std::find(seen.begin(), seen.end(), n) == seen.end());   // one name per component
        seen.push_back(n);
    }
    CHECK(std::string(qsa_chain_slot_name(-1)) == "?");
    CHECK(std::string(qsa_chain_slot_name(kChainCount)) == "?");

    QsaChainFp fp;
    CHECK(qsa_chain_field(kChainMix, fp) == "mix=--------");   // absent reads as absent, in the line too
    fp.hash = 0x0000abcd; fp.sum = 1.5f; fp.maxabs = 3.25f;
    const std::string line = qsa_chain_field(kChainMix, fp);
    CHECK(line.rfind("mix=", 0) == 0);
    CHECK(line.find("/1.5/3.25/") != std::string::npos);   // hash / sum / maxabs, then the sketch
    size_t slashes = 0;
    for (char c : line) if (c == '/') ++slashes;
    CHECK(slashes == 3);   // hash / sum / maxabs / the sketch
}

// ---- the byte slots: the figure the pack prints and the figure the card prints must be one number --------------

// A packed weight image is opaque code bytes, so the chain compares its bits and nothing else.  The pack computes
// a hash on the host from the bytes it wrote and logs it; the device slot hashes the same window on the card; the
// two are comparable only because both are the kernel's own rule.  If they drift - a change to the sampler, to the
// `rows` = 1 window, to the sketch switch - the boot's two lines stop answering each other, and the hunt is back
// to reading a nan as a verdict.  The width is one card's real `attn_output` columns: half of a 6144-column Q6_K
// image over 2560 rows, which is 6 451 200 B - the buffer the arm ships and the product reads.
void byte_slots_carry_the_packs_own_number() {
    const int64_t words = 6451200 / 4;
    std::vector<uint32_t> img((size_t) words);
    std::mt19937 rng(20261030u);
    for (size_t i = 0; i < img.size(); ++i) img[i] = rng();   // arbitrary bits: codes and scales, not floats

    // (a) no block hides: one word changed anywhere in 6.45 MB moves the figure.  This is the whole reason the
    // slot walks the image instead of sampling rows of it, so it is graded, not asserted.
    std::vector<uint32_t> poisoned = img;
    poisoned[words - 7] = 0x7fc00000u;   // a NaN word: the shape a poisoned weight block arrives as
    CHECK(qsa_chain_bytes_hash(poisoned.data(), words) != qsa_chain_bytes_hash(img.data(), words));
    std::vector<uint32_t> swapped = img;
    std::swap(swapped[17], swapped[19]);   // same multiset, wrong place
    CHECK(qsa_chain_bytes_hash(swapped.data(), words) != qsa_chain_bytes_hash(img.data(), words));

    // (b) the host figure IS the kernel's window: `rows` = 1 of `stride` words, per the slot's put.
    CHECK(qsa_chain_bytes_hash(img.data(), words) == qsa_chain_ref(img.data(), 1, words).hash);
    CHECK(qsa_chain_bytes_hash(poisoned.data(), words) == qsa_chain_ref(poisoned.data(), 1, words).hash);

    // (c) a byte slot prints its bits and nothing else.  A correctly packed Q6_K image is FULL of NaN bit
    // patterns, so a `nan` in its line is what correct looks like - and a reader who took it for unwritten memory
    // would send the hunt to the wrong card, which is the mistake this line shape makes impossible.
    const QsaChainFp fp = qsa_chain_ref(img.data(), 1, words);
    char hb[16];
    std::snprintf(hb, sizeof hb, "%08x", fp.hash);
    CHECK(qsa_chain_field(kChainColsL, fp) == std::string("colsl=") + hb);
    CHECK(qsa_chain_field(kChainColsL, fp).find('/') == std::string::npos);
    CHECK(qsa_chain_field(kChainColsP, fp).rfind("colsp=", 0) == 0);
    QsaChainFp absent;
    CHECK(qsa_chain_field(kChainColsL, absent) == "colsl=--------");   // an absence still reads as an absence

    // (d) only the two column slots are bytes: an activation keeps its sum, its maxabs and its sketch.
    CHECK(qsa_chain_slot_is_bytes(kChainColsL) && qsa_chain_slot_is_bytes(kChainColsP));
    CHECK(!qsa_chain_slot_is_bytes(kChainGated) && !qsa_chain_slot_is_bytes(kChainPartL) &&
          !qsa_chain_slot_is_bytes(kChainBo) && !qsa_chain_slot_is_bytes(kChainSumL));
    CHECK(!qsa_chain_slot_is_bytes(-1) && !qsa_chain_slot_is_bytes(kChainCount));
}

// ---- the device kernel against the host rule ----------------------------------------------------------

int device_parity() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 77;
    if (cudaSetDevice(0) != cudaSuccess) return 77;

    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    uint32_t* dev = nullptr;
    size_t cap_words = 0;
    for (const Shape& sh : kShapes) cap_words = std::max(cap_words, (size_t) (sh.rows * sh.stride));
    CHECK(cudaMalloc((void**) &dev, cap_words * sizeof(uint32_t)) == cudaSuccess);
    QsaChainFp* slot = nullptr;
    CHECK(cudaMalloc((void**) &slot, sizeof(QsaChainFp)) == cudaSuccess);

    QsaChainFp host;
    for (const Shape& sh : kShapes) {
        const int64_t n = sh.rows * sh.stride;
        const std::vector<float> f = make_window(n, (uint32_t) (sh.rows * 31 + sh.stride));
        // The window goes onto the card on the SAME stream the fingerprint runs on.  A plain `cudaMemcpy` belongs
        // to the legacy stream and has no ordering with a non-blocking one, and the first draft of this test did
        // exactly that: the kernel read the previous shape's bytes and disagreed with the reference while the
        // reference was right.  That is the instrument's contract in miniature - it fingerprints what a STREAM has
        // written - and the engine meets it because every component it reads is produced on the stream it records
        // onto.  A test that broke the contract would grade the kernel for the test's own race.
        CHECK(cudaMemcpyAsync(dev, f.data(), (size_t) n * 4, cudaMemcpyHostToDevice, stream) == cudaSuccess);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        const QsaChainFp want = qsa_chain_ref(reinterpret_cast<const uint32_t*>(f.data()), sh.rows, sh.stride);
        std::string err;
        CHECK(qsa_chain_record(slot, dev, sh.rows, sh.stride, stream, err));
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        CHECK(cudaMemcpy(&host, slot, sizeof host, cudaMemcpyDeviceToHost) == cudaSuccess);
        // Bitwise on all three figures: same bytes, same rows, same tree, so there is no tolerance to grant.
        CHECK(host.hash == want.hash);
        CHECK(host.sum == want.sum);
        CHECK(host.maxabs == want.maxabs);
        // The sketch is graded bitwise too, and it is the figure a cross-boot cosine is read from. A kernel and a
        // reference that disagreed here would print cosines that meant nothing, which is the failure the whole
        // instrument exists to avoid - so an ulp is as much a failure as a hash mismatch.
        for (int p = 0; p < kChainProj; ++p) CHECK(host.proj[p] == want.proj[p]);
        if (host.hash != want.hash)
            std::printf("  %s: kernel %08x vs host %08x\n", sh.what, host.hash, want.hash);

        // Recording is idempotent: a component read twice is the same component.
        CHECK(qsa_chain_record(slot, dev, sh.rows, sh.stride, stream, err));
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        QsaChainFp again;
        CHECK(cudaMemcpy(&again, slot, sizeof again, cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(again.hash == want.hash && again.sum == want.sum);
    }

    // The byte slot's device side, at the pack's real size: the kernel's hash must be the figure the pack printed
    // from the same bytes, and the sketch must be left zero rather than computed from code bits (a byte slot's
    // projections are not a quantity, and a caller must not pay a second walk of 6.45 MB for nothing).
    {
        const int64_t words = 6451200 / 4;
        std::vector<uint32_t> img((size_t) words);
        std::mt19937 rng(4242u);
        for (size_t i = 0; i < img.size(); ++i) img[i] = rng();
        CHECK(words <= (int64_t) cap_words);
        CHECK(cudaMemcpyAsync(dev, img.data(), (size_t) words * 4, cudaMemcpyHostToDevice, stream) == cudaSuccess);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        std::string berr;
        CHECK(qsa_chain_record(slot, dev, 1, words, stream, berr, /*sketch*/ false));
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        QsaChainFp bytes;
        CHECK(cudaMemcpy(&bytes, slot, sizeof bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(bytes.hash == qsa_chain_bytes_hash(img.data(), words));
        CHECK(bytes.hash == qsa_chain_ref(img.data(), 1, words).hash);
        for (int p = 0; p < kChainProj; ++p) CHECK(bytes.proj[p] == 0.0f);
        // And the same window with the sketch on still agrees on the hash: the switch costs the projections, never
        // the figure the two sides of a boot are read from.
        CHECK(qsa_chain_record(slot, dev, 1, words, stream, berr, /*sketch*/ true));
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        QsaChainFp with_sketch;
        CHECK(cudaMemcpy(&with_sketch, slot, sizeof with_sketch, cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(with_sketch.hash == bytes.hash);
    }

    // A slot cleared to the sentinel reads as absent until a record lands in it - the state a chunk starts in.
    CHECK(cudaMemset(slot, 0xFF, sizeof(QsaChainFp)) == cudaSuccess);
    CHECK(cudaMemcpy(&host, slot, sizeof host, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(host.hash == kChainUnrecorded);
    CHECK(qsa_chain_field(kChainBo, host) == "bo=--------");

    // The refusals.  Each must say so rather than launch a kernel over someone else's memory.
    const std::vector<float> f = make_window(64, 7);
    CHECK(cudaMemcpyAsync(dev, f.data(), 64 * 4, cudaMemcpyHostToDevice, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    std::string err;
    CHECK(!qsa_chain_record(nullptr, dev, 4, 16, stream, err));
    CHECK(!err.empty());
    CHECK(!qsa_chain_record(slot, nullptr, 4, 16, stream, err));
    CHECK(!qsa_chain_record(slot, dev, 0, 16, stream, err));
    CHECK(!qsa_chain_record(slot, dev, 4, 0, stream, err));
    CHECK(!qsa_chain_record(slot, dev, -1, 16, stream, err));
    CHECK(!qsa_chain_record(slot, dev, 4, 16, nullptr, err));
    const char* unaligned = reinterpret_cast<const char*>(dev) + 1;
    CHECK(!qsa_chain_record(slot, unaligned, 4, 16, stream, err));
    CHECK(cudaGetLastError() == cudaSuccess);   // a refusal must leave no error behind for the next launch

    // One kernel per component, so the instrument's own cost is a line in a profile rather than a surprise.  The
    // figure below is what a QSA layer pays when the flag is on; multiply by the components and the layers.
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const Shape& big = kShapes[0];
    cudaEventRecord(a, stream);
    for (int i = 0; i < 100; ++i) qsa_chain_record(slot, dev, big.rows, big.stride, stream, err);
    cudaEventRecord(b, stream);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    std::printf("chain fingerprint: %.2f us per component at %lld rows x %lld words\n", ms * 1000.0f / 100.0f,
                (long long) big.rows, (long long) big.stride);

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(slot); cudaFree(dev); cudaStreamDestroy(stream);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    sampling_contract();
    reference_contract();
    sketch_contract();
    names_and_fields();
    byte_slots_carry_the_packs_own_number();
    if (g_fail) {
        std::printf("qsa chain parity: %d failures\n", g_fail);
        return 1;
    }
    const int rc = device_parity();
    std::printf("qsa chain parity: %d failures\n", g_fail);
    if (g_fail) return 1;
    if (rc == 77) return 77;
    if (selftest) std::printf("qsa_chain_parity OK\n");
    return 0;
}

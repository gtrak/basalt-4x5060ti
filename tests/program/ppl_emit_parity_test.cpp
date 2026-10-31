// tests/program/ppl_emit_parity_test.cpp - the EMITTED base's layout and quantization, on the CPU (no device, no CUDA).
//
// The differential quality gate (docs/precision/moe-partial-merge.md) has one engine arm write its own log-probs
// into a base file, and the OTHER arm read them back through the untouched llama.cpp KL-divergence reader. That is
// only an instrument if the writer and the reader agree exactly, so this pins what the gate's conclusion rests on:
//
//   1. the emitted row decodes back, through the reader's own arithmetic, to the log-probs of the logits it was
//      given, within half the row's uint16 step: the row is a quantization of the truth, not another distribution;
//   2. the row's (scale, min log-prob) header spans the row, so nothing is clipped at either end - a clipped tail
//      would read back as a floor and quietly bias every KLD, in both the self-check and the gate;
//   3. KL(base ‖ test) is zero when the test's logits are the logits the base was emitted from, which is the
//      self-check arm of the gate; if the emitter drifts from the reader, this is where it shows;
//   4. the padded tail of the row is zeroed, so a reader that walks the padding sees a log-prob below the -16
//      skip threshold rather than whatever the file held before.
//
// The vocabulary size is odd on purpose: the layout pads each row to an even uint16 count, and an even size would
// leave that padding untested.
#include "basalt/program/ppl_gate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

// the reader's decode of one emitted row: the same lines PplGate::row runs against a llama.cpp base
std::vector<float> decode_row(const std::vector<uint16_t>& q, int64_t n_vocab) {
    float scale, min_lp;
    std::memcpy(&scale, q.data(), 4);
    std::memcpy(&min_lp, q.data() + 2, 4);
    std::vector<float> lp((size_t) n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) lp[(size_t) i] = scale * q[(size_t) (4 + i)] + min_lp;
    return lp;
}

// the reference log-probs of a logits row, in double so the bound the test quotes means something
std::vector<double> log_softmax(const std::vector<float>& lg) {
    const double mx = *std::max_element(lg.begin(), lg.end());
    double sum = 0.0;
    for (const float v : lg) sum += std::exp((double) v - mx);
    const double lse = mx + std::log(sum);
    std::vector<double> lp(lg.size());
    for (size_t i = 0; i < lg.size(); ++i) lp[i] = (double) lg[i] - lse;
    return lp;
}

// logits like a real head, in two shapes: a broad one and one with a single logit far above the rest, so neither
// a flat row nor a spiked row is the only case measured
std::vector<float> synthetic_logits(int64_t n_vocab, uint64_t seed, int shape) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> n(0.0f, shape == 0 ? 3.0f : 8.0f);
    std::vector<float> lg((size_t) n_vocab);
    for (auto& v : lg) v = n(rng);
    if (shape == 1) {
        size_t top = 0;
        for (size_t i = 1; i < lg.size(); ++i)
            if (lg[i] < lg[top]) top = i;
        lg[top] += 60.0f;
    }
    return lg;
}

}  // namespace

int main() {
    int fails = 0;
    // 4099 is odd, so the even-count padding is exercised; 248320 is the model's vocabulary, and the row's
    // quantization step - and with it the floor the gate's self-check arm is graded against - is a function of it,
    // so the floor is measured here rather than scaled up from the small case
    const int64_t vocab_sizes[] = {4099, 248320};
    const int rows_per_shape[] = {8, 2};

    for (int vs = 0; vs < 2; ++vs) {
        const int64_t n_vocab = vocab_sizes[vs];
        double worst_err = 0.0, worst_kld = 0.0;
        int64_t out_of_bound = 0, bad_pad = 0, rows = 0;

        for (int shape = 0; shape < 2; ++shape) {
            for (uint64_t seed = 1; seed <= (uint64_t) rows_per_shape[vs]; ++seed) {
            const std::vector<float> lg = synthetic_logits(n_vocab, seed * 31 + shape, shape);
            std::vector<uint16_t> q;
            float scale = 0.0f, min_lp = 0.0f;
            basalt::program::PplGate::quantize_row(lg.data(), n_vocab, q, scale, min_lp);
            ++rows;

            const int64_t want_u16 = 2 * ((n_vocab + 1) / 2) + 4;
            if ((int64_t) q.size() != want_u16) {
                std::printf("  *** row %lld: %zu u16, expected %lld\n", (long long) rows, q.size(),
                            (long long) want_u16);
                ++fails;
            }
            for (int64_t i = n_vocab + 4; i < (int64_t) q.size(); ++i)
                if (q[(size_t) i] != 0) ++bad_pad;

            const std::vector<float> lp = decode_row(q, n_vocab);
            const std::vector<double> want = log_softmax(lg);

            // the row's own span, from the header the reader trusts: everything the decode costs is a multiple of
            // this step, plus the float slack of forming the row's log-sum-exp in float rather than in double
            const float lo = *std::min_element(lg.begin(), lg.end());
            const float hi = *std::max_element(lg.begin(), lg.end());
            const double span = (double) hi - (double) lo;
            const double bound = 0.5 * (double) scale + 4.0 * span * std::numeric_limits<float>::epsilon();

            float hi_u16 = 0.0f;
            for (int64_t i = 0; i < n_vocab; ++i) hi_u16 = std::max(hi_u16, (float) q[(size_t) (4 + i)]);

            for (int64_t i = 0; i < n_vocab; ++i) {
                const double err = std::fabs((double) lp[(size_t) i] - want[(size_t) i]);
                worst_err = std::max(worst_err, err);
                if (err > bound) ++out_of_bound;
            }
            // the header has to span the row: its largest log-prob sits at (or one step below) 65535
            if (hi_u16 < 65534.0f) {
                std::printf("  *** row %lld (shape %d, seed %llu): the row tops out at %g of 65535, a tail is clipped\n",
                            (long long) rows, shape, (unsigned long long) seed, (double) hi_u16);
                ++fails;
            }

            // KL(base ‖ test) where both sides come from the same logits, summed the reader's way. Not zero: the
            // base side is the row's uint16 quantization of the same distribution, so the residue is of the order
            // of one quantization step of the row. This is the number the gate's self-check arm is graded against
            double kld = 0.0;
            for (int64_t i = 0; i < n_vocab; ++i) {
                const double lb = (double) lp[(size_t) i];
                if (lb > -16.0) kld += std::exp(lb) * (lb - want[(size_t) i]);
            }
            worst_kld = std::max(worst_kld, kld);

            // the emitted row's own top-1 is the true top-1: it is the entry that quantized to 65535
            int64_t top = 0, top_q = 0;
            for (int64_t i = 1; i < n_vocab; ++i) {
                if (lg[i] > lg[top]) top = i;
                if (q[(size_t) (4 + i)] > q[(size_t) (4 + top_q)]) top_q = i;
            }
            if (top != top_q) {
                std::printf("  *** row %lld: the emitted row's top-1 is %lld, the logits' is %lld\n", (long long) rows,
                            (long long) top_q, (long long) top);
                ++fails;
                }
            }
        }

        std::printf("ppl_emit_parity: %lld rows of %lld vocab\n", (long long) rows, (long long) n_vocab);
        std::printf("   worst |decoded - true| log-prob: %.3g (half a uint16 step, plus float slack, is the bound)\n",
                    worst_err);
        std::printf("   entries outside that bound: %lld\n", (long long) out_of_bound);
        std::printf("   worst KL(base || test) against its own logits: %.3g (one quantization step is the scale)\n",
                    worst_kld);
        std::printf("   non-zero padding words: %lld\n", (long long) bad_pad);
        if (out_of_bound != 0) ++fails;
        if (bad_pad != 0) ++fails;
        // the self-check arm of the gate is only meaningful if a row read back against its own emission is
        // indistinguishable from itself at the resolution the gate grades at
        if (worst_kld > 1e-3) {
            std::printf("   *** the self-check arm would not read back as its own base\n");
            ++fails;
        }
    }

    std::printf("%s\n", fails == 0 ? "ppl_emit_parity: OK" : "ppl_emit_parity: FAILED");
    return fails == 0 ? 0 : 1;
}

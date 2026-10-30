// tools/mmq_width_probe.cu - the MMQ width sweep: NVFP4 gate/up efficiency as a function of the output width N,
// measured, for the TP=4 ceiling decision.
//
// WHAT THIS IS. A standalone CUDA program, not wired into the engine and not built by CMake: it exists to answer
// one question - when a gate/up product is narrowed, how much MORE does a row cost? - before the quarter-shard
// plan is priced against the filed assumption. The assumption on record (the kimi band) is 1.2-1.4x at a quarter
// width and ~1.05-1.15x at half width; it has never been measured, and TP=4's projected ceiling against
// tp2pp2's ~2,540 turns on it: >1.5x at the narrow end kills the ceiling, <1.25x strengthens it.
//
// WHAT IT MEASURES. The engine's own MMQ path, not a copy of it: `basalt::prefill::mmq::Context::run(Product)`
// from src/prefill/moe_mmq.cu, the same call src/prefill/prefill.cpp makes for a group's gate/up
// (`gu.w_rows = 1280; gu.w_cols = N`), with the engine's own activation quantizer (`mmq::quantize`), identity row
// map (`mmq::iota`) and expert bounds. One launch per timed rep; the per-expert weight bytes, the q8_1 activation
// bytes and the tile config MMQ chose are printed beside the median so the number can be audited afterwards.
//
// THE GEOMETRY. The pack's experts are NVFP4 (NativeFmt, src/kernels/cpu/native_expert.cpp:52-64: gu_row =
// ggml_row_size(NVFP4, n_embd) = (2560/64)*36 = 1440 B; the 16 B scale tail is applied in SwiGLU, not read by the
// product). The gate/up product of one expert is [w_rows = 2*n_ff = 1280, w_cols = n_embd = 2560]; T is the
// activation rows of the launch. The sweep narrows w_rows (MMQ's nrows_x, the direction a tensor-parallel expert
// shard cuts) at fixed w_cols (the K loop), over {2560, 1280, 640} x {2048, 8192}.
//
// WHICH WIDTH IS "WHOLE". The real gate/up product is 1280 columns wide, so this table's 1280 point is the
// engine's actual shape, 640 is what a 2-way shard of it is, and 320 would be a 4-way shard of the real thing;
// 2560 is the doubled width the ceiling was quoted against. Widths are therefore reported by width, never by
// shard count, so the ceiling arithmetic can be re-based on either reading without measuring again.
//
// PARITY, because a wrong slice is worse than a slow one. --parity runs the widest point and keeps its
// destination; the narrower points then run over the same bytes - weights are generated row-major with a fixed
// seed, so the leading N rows of a wider buffer are byte-identical to a narrower point's own generation, and the
// activations come from a seed of their own so they are identical across points - and the narrow destination must
// equal the wide run's leading columns. MMQ splits the K loop across blocks (stream-k) and the split depends on
// the tile count, so equality is fp32-reassociation tight rather than bitwise; the reported max absolute and
// relative error are what a mis-sliced stride would blow up by orders of magnitude, which is the point of it.
//
// BUILD (deliberately not a CMake target: this is a probe, not a shipped artifact). The ggml format table is C,
// so it is compiled by the host C compiler and linked in:
//   cd /home/gary/opt/basalt-tpstep0
//   gcc -O2 -c src/kernels/cpu/ggml/ggml_quants.c -Iinclude -Isrc/kernels/cpu/ggml -o /tmp/ggml_quants.o
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 --extended-lambda -use_fast_math \
//     -Iinclude -Isrc/prefill -o /tmp/mmq_width_probe tools/mmq_width_probe.cu \
//     src/prefill/moe_mmq.cu src/prefill/mmq/mmq_host.cu src/prefill/mmq/quantize.cu \
//     src/prefill/mmq/instances/mmq_q2_0.cu src/prefill/mmq/instances/mmq_q8_0.cu \
//     src/prefill/mmq/instances/mmq_q5_1.cu src/prefill/mmq/instances/mmq_q4_k.cu \
//     src/prefill/mmq/instances/mmq_q5_k.cu src/prefill/mmq/instances/mmq_q6_k.cu \
//     src/prefill/mmq/instances/mmq_iq2_xxs.cu src/prefill/mmq/instances/mmq_iq2_xs.cu \
//     src/prefill/mmq/instances/mmq_iq2_s.cu src/prefill/mmq/instances/mmq_iq3_xxs.cu \
//     src/prefill/mmq/instances/mmq_iq3_s.cu src/prefill/mmq/instances/mmq_iq4_nl.cu \
//     src/prefill/mmq/instances/mmq_iq4_xs.cu src/prefill/mmq/instances/mmq_nvfp4.cu \
//     /tmp/ggml_quants.o -lcudart
// (moe_mmq.cu's switch names every instantiated format, so every instance links; the minutes are MMQ's templates,
// not this file. The flags are the ones CMake gives the `mmq` target: -O3, --extended-lambda, -use_fast_math.)
//
// RUN
//   /tmp/mmq_width_probe --sweep --parity                                   # the decision run
//   /tmp/mmq_width_probe --sweep --parity --device 1                        # the same on another card
//   /tmp/mmq_width_probe --sweep --experts 16 --parity                      # a whole group, as prefill launches it
//   /tmp/mmq_width_probe --n 640 --rows 8192 --reps 25                      # one point
//   /tmp/mmq_width_probe --layout                                           # the byte and tile table, no CUDA call
//   /tmp/mmq_width_probe --layout --smpbo 101376 --nsm 45                   # ... with the card's real limits
//
// BUFFER DISCIPLINE. The probe refuses a point unless the card has at least 256 MiB free (cudaMemGetInfo, read
// immediately before the point, the discipline tools/tp_ring_probe.cu used), and refuses any point whose own
// footprint exceeds --budget-mib (default 64) unless --allow-big is given: the T=8192, N=2560 points want
// ~115 MiB on one card, which is fine on free cards and not fine beside a live server. It frees everything
// between points and touches no stream but its own.
#include "basalt/prefill/moe_mmq.hpp"

#include "mmq/mmq.cuh"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace mmq = basalt::prefill::mmq;

constexpr int kType = GGML_TYPE_NVFP4;           // the pack's expert format (NativeFmt's gu_type/d_type)
constexpr int64_t kK = 2560;                     // n_embd: the gate/up product's w_cols, the K loop
constexpr int64_t kGuWhole = 1280;               // 2*n_ff: the real gate/up product's output width
constexpr size_t kMinFreeBytes = 256ull << 20;   // refuse a point on a card with less than this free
constexpr uint32_t kSeedW = 20261030u;           // the weight stream's fixed seed
constexpr uint32_t kSeedX = 0x5bd1e995u;         // the activation stream's fixed seed, so T-only points share x

bool fail(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "mmq probe: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];   // --reps is odd, so the median is one of the samples
}

struct Point {
    int64_t n = 0;       // the product's output width (MMQ nrows_x, Product::w_rows)
    int64_t k = kK;      // the K loop (MMQ ncols_x, Product::w_cols)
    int64_t rows = 0;    // activation rows in the launch (Product::total_rows)
    int experts = 1;     // experts per launch (Product::n); the rows split evenly across them
    int reps = 25;
};

/// The tile config MMQ will pick for this launch, for the record: the same loop mul_mat_q_switch_J runs. `smpbo`
/// and `nsm` come from the device at run time; in --layout mode they may be 0, which prints every candidate
/// instead of the chosen one so the layout is readable without booting CUDA.
void print_layout(int64_t w_rows, int64_t opt_rows, size_t smpbo, int nsm) {
    const bool fallback = w_rows % 128 != 0;
    if (smpbo == 0 || nsm == 0) {
        std::printf("  tile candidates (smpbo/nsm unknown; the chosen one is the widest J that fits):\n");
        for (int J = 8; J <= 128; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) kType, J, fallback);
            if (c.type == GGML_TYPE_COUNT) continue;
            std::printf("    J=%d I=%d nthreads=%d sram_layout=%d K_vram=%d stream_k=%d shared=%zu B ntx=%d nty=%d\n",
                        c.J, c.I, c.nthreads, (int) c.sram_layout, c.K_vram, (int) c.stream_k,
                        mmq_get_nbytes_shared(c), (int) ((opt_rows + c.J - 1) / c.J),
                        (int) ((w_rows + c.I - 1) / c.I));
        }
        return;
    }
    int j_best = 0, ntiles_best = 1 << 30;
    size_t shared = 0;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) kType, J, fallback);
        if (c.type == GGML_TYPE_COUNT) continue;
        const size_t nb = mmq_get_nbytes_shared(c);
        if (nb > smpbo) continue;
        const int nt = (int) ((opt_rows + c.J - 1) / c.J);
        if (nt < ntiles_best) { j_best = J; ntiles_best = nt; shared = nb; }
    }
    const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) kType, j_best, fallback);
    std::printf("  tile: %s J=%d I=%d nthreads=%d occupancy=%d sram_layout=%d K_vram=%d stream_k=%d fallback=%d "
                "shared=%zu B (smpbo %zu B), nsm=%d, nty=%d ntx=%d\n",
                ggml_type_name((ggml_type) kType), c.J, c.I, c.nthreads, c.occupancy, (int) c.sram_layout, c.K_vram,
                (int) c.stream_k, (int) c.fallback, shared, smpbo, nsm,
                (int) ((w_rows + c.I - 1) / c.I), ntiles_best);
}

/// Refuses a point the discipline does not allow; prints the reason.
bool card_has_room(int dev, size_t need, size_t budget, bool allow_big) {
    if (!fail(cudaSetDevice(dev), "set device")) return false;
    size_t free_b = 0, total = 0;
    if (!fail(cudaMemGetInfo(&free_b, &total), "mem get info")) return false;
    if (free_b < kMinFreeBytes) {
        std::printf("  SKIPPED: card %d has %llu MiB free, the floor is %llu MiB\n", dev,
                    (unsigned long long) (free_b >> 20), (unsigned long long) (kMinFreeBytes >> 20));
        return false;
    }
    if (need > budget && !allow_big) {
        std::printf("  SKIPPED: the point needs %llu MiB on one card, the budget is %llu MiB (pass --allow-big)\n",
                    (unsigned long long) (need >> 20), (unsigned long long) (budget >> 20));
        return false;
    }
    std::printf("  card %d: %llu MiB free of %llu MiB, this point's footprint %llu MiB\n", dev,
                (unsigned long long) (free_b >> 20), (unsigned long long) (total >> 20),
                (unsigned long long) (need >> 20));
    return true;
}

struct Buffers {
    void* w = nullptr;     // experts' weights, stride matrix_bytes: a narrower N reads each expert's leading rows
    void* x = nullptr;     // the float activations
    void* xq = nullptr;    // the q8_1 rows the product reads
    void* dst = nullptr;   // the product's output
    int32_t* bounds = nullptr;
    int32_t* ids = nullptr;
    cudaStream_t s = nullptr;
    size_t bytes = 0;
};

size_t footprint(const Point& p) {
    return mmq::matrix_bytes(kType, p.n, p.k) * (size_t) p.experts + 4096 + (size_t) p.rows * p.k * 4 +
           mmq::q8_bytes(p.rows, p.k) + (size_t) p.rows * p.n * 4 + (size_t) p.rows * 4 +
           (size_t) (p.experts + 1) * 4;
}

bool alloc(Buffers& b, const Point& p) {
    const size_t w_b = mmq::matrix_bytes(kType, p.n, p.k) * (size_t) p.experts + 4096;
    bool ok = fail(cudaMalloc(&b.w, w_b), "malloc w") && fail(cudaMalloc(&b.x, (size_t) p.rows * p.k * 4), "malloc x") &&
              fail(cudaMalloc(&b.xq, mmq::q8_bytes(p.rows, p.k)), "malloc xq") &&
              fail(cudaMalloc(&b.dst, (size_t) p.rows * p.n * 4), "malloc dst") &&
              fail(cudaMalloc((void**) &b.ids, (size_t) p.rows * 4), "malloc ids") &&
              fail(cudaMalloc((void**) &b.bounds, (size_t) (p.experts + 1) * 4), "malloc bounds") &&
              fail(cudaStreamCreate(&b.s), "stream");
    b.bytes = footprint(p);
    return ok;
}

void free_all(Buffers& b) {
    cudaFree(b.w); cudaFree(b.x); cudaFree(b.xq); cudaFree(b.dst);
    cudaFree((void*) b.ids); cudaFree((void*) b.bounds);
    if (b.s) cudaStreamDestroy(b.s);
    b = Buffers{};
}

/// Seeded NVFP4 weights and float activations. The weight stream is per (expert, row), so a narrower point's
/// rows are byte-identical to the wider point's leading rows; the activation stream is per (row, column).
void fill(std::mt19937& rw, std::mt19937& rx, const Buffers& b, const Point& p) {
    const auto* tr = ggml_get_type_traits((ggml_type) kType);
    const size_t row_bytes = ggml_row_size((ggml_type) kType, p.k);
    std::vector<uint8_t> host((size_t) p.n * row_bytes);
    std::vector<float> row((size_t) p.k);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ex(-10.0f, 3.0f);   // sub-block scales over 13 binades, as the parity test
    std::vector<uint8_t> one(host.size());
    for (int e = 0; e < p.experts; ++e) {
        std::mt19937 re(rw());   // this expert's own stream, so expert 0 is the same bytes whatever `experts` is
        for (int64_t r = 0; r < p.n; ++r) {
            for (int64_t c = 0; c < p.k; c += 16) {
                const float m = std::exp2(ex(re));
                for (int j = 0; j < 16; ++j) row[(size_t) (c + j)] = nd(re) * m;
            }
            tr->from_float_ref(row.data(), one.data() + (size_t) r * row_bytes, p.k);
        }
        std::memcpy(host.data() + (size_t) e * host.size(), one.data(), host.size());
    }
    if (!fail(cudaMemcpy(b.w, host.data(), host.size(), cudaMemcpyHostToDevice), "weight upload")) std::exit(1);

    std::vector<float> xhost((size_t) p.rows * p.k);
    std::mt19937 rxs(rx);   // restarted from the fixed activation seed: (row, column) always draws the same value
    std::normal_distribution<float> xd(0.0f, 1.0f);
    for (auto& v : xhost) v = xd(rxs);
    if (!fail(cudaMemcpy(b.x, xhost.data(), xhost.size() * 4, cudaMemcpyHostToDevice), "activation upload")) std::exit(1);
}

/// One timed point. Appends the per-row median to `out` when asked and reads the destination back into `ref`.
bool run_point(mmq::Context& ctx, const Point& p, int dev, size_t budget, bool allow_big, bool keep_dst,
               std::vector<float>* ref, std::vector<double>* out) {
    std::printf("point N=%lld K=%lld T=%lld experts=%d reps=%d", (long long) p.n, (long long) p.k,
                (long long) p.rows, p.experts, p.reps);
    std::printf("%s\n", p.n == kGuWhole ? "  <- the real gate/up width" : "");
    if (!mmq::fits(kType, p.n)) {
        std::printf("  SKIPPED: no MMQ tile for %lld weight rows on this card\n", (long long) p.n);
        return false;
    }
    if (!card_has_room(dev, footprint(p), budget, allow_big)) return false;

    Buffers b;
    if (!alloc(b, p)) { free_all(b); return false; }
    std::mt19937 rw(kSeedW), rx(kSeedX);
    fill(rw, rx, b, p);

    // the engine's own quantizer, identity row map and per-expert bounds
    mmq::quantize((const float*) b.x, nullptr, b.xq, kType, p.k, p.k, p.rows, b.s);
    mmq::iota(b.ids, p.rows, b.s);
    std::vector<int32_t> hb((size_t) p.experts + 1);
    for (int e = 0; e <= p.experts; ++e) hb[(size_t) e] = (int32_t) ((int64_t) e * p.rows / p.experts);
    if (!fail(cudaMemcpy(b.bounds, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice), "bounds")) {
        free_all(b); return false;
    }

    mmq::Product pr;
    pr.w = b.w; pr.type = kType; pr.w_rows = p.n; pr.w_cols = p.k;
    pr.expert_bytes = mmq::matrix_bytes(kType, p.n, p.k);
    pr.n = p.experts; pr.xq = b.xq; pr.bounds = b.bounds; pr.ids = b.ids;
    pr.total_rows = p.rows; pr.max_rows = p.rows / p.experts; pr.dst = (float*) b.dst; pr.ld_dst = p.n;
    pr.opt_rows = 0;   // as the engine leaves it: the tile width is sized against max_rows

    print_layout(p.n, pr.max_rows, ggml_cuda_info().devices[dev].smpbo, ggml_cuda_info().devices[dev].nsm);

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    fail(cudaEventCreate(&e0), "event"); fail(cudaEventCreate(&e1), "event");
    for (int r = 0; r < 3; ++r) ctx.run(pr, b.s);   // warm-up, not timed
    bool ok = fail(cudaStreamSynchronize(b.s), "warmup sync");
    std::vector<double> per_row;
    for (int r = 0; ok && r < p.reps; ++r) {
        ok = ok && fail(cudaEventRecord(e0, b.s), "rec 0");
        ctx.run(pr, b.s);
        ok = ok && fail(cudaEventRecord(e1, b.s), "rec 1") && fail(cudaEventSynchronize(e1), "sync");
        if (!ok) break;
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        per_row.push_back((double) ms * 1e6 / (double) p.rows);
    }
    if (ok) {
        const double ns = median(per_row);
        const std::vector<double> sorted = per_row;
        std::vector<double> srt = sorted;
        std::sort(srt.begin(), srt.end());
        const double wt = (double) pr.expert_bytes * p.experts;
        const double act = (double) mmq::q8_bytes(p.rows, p.k);
        const double flops = 2.0 * (double) p.rows * (double) p.n * (double) p.k * p.experts;
        std::printf("  RESULT N=%lld T=%lld experts=%d: median %.1f ns/row (min %.1f, max %.1f), weights %.1f MiB "
                    "-> %.2f GiB/s, q8 %.1f MiB -> %.2f GiB/s, %.2f TFLOP/s\n",
                    (long long) p.n, (long long) p.rows, p.experts, ns, srt.front(), srt.back(),
                    wt / (1024.0 * 1024.0), wt / (ns * 1e-9) / (1024.0 * 1024.0),
                    act / (1024.0 * 1024.0), act / (ns * 1e-9) / (1024.0 * 1024.0),
                    flops / (ns * 1e-9) / 1e12);
        if (out) out->push_back(ns);
        if (keep_dst && ref) {
            ref->resize((size_t) p.rows * p.n);
            ok = fail(cudaMemcpy(ref->data(), b.dst, ref->size() * 4, cudaMemcpyDeviceToHost), "dst readback");
        }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    free_all(b);   // between points, so the next point reads a clean card
    return ok;
}

/// The slice check: the narrow run's whole destination must equal the wide run's leading columns.
void parity(mmq::Context& ctx, const Point& wide, const Point& narrow, int dev, size_t budget, bool allow_big) {
    std::vector<float> ref, got;
    if (!run_point(ctx, wide, dev, budget, allow_big, true, &ref, nullptr)) return;
    if (!run_point(ctx, narrow, dev, budget, allow_big, true, &got, nullptr)) return;
    double max_abs = 0, max_rel = 0, rms = 0;
    int64_t wr = -1, wc = -1;
    for (int64_t r = 0; r < narrow.rows; ++r) {
        for (int64_t c = 0; c < narrow.n; ++c) {
            const double a = ref[(size_t) r * wide.n + c];
            const double bv = got[(size_t) r * narrow.n + c];
            const double d = std::fabs(a - bv);
            rms += d * d;
            if (d > max_abs) { max_abs = d; wr = r; wc = c; }
            const double rel = d / std::max(std::fabs(a), 1.0);
            if (rel > max_rel) max_rel = rel;
        }
    }
    const double n = (double) narrow.rows * (double) narrow.n;
    std::printf("  PARITY N=%lld vs N=%lld (T=%lld, experts=%d): max|d|=%.3e at row %lld col %lld, max rel=%.3e, "
                "RMS=%.3e -> %s (1e-3 bound)\n",
                (long long) wide.n, (long long) narrow.n, (long long) narrow.rows, narrow.experts, max_abs,
                (long long) wr, (long long) wc, max_rel, std::sqrt(rms / n),
                (max_abs <= 1e-3 && max_rel <= 1e-3) ? "PASS" : "FAIL");
}

}  // namespace

int main(int argc, char** argv) {
    bool sweep = false, do_parity = false, allow_big = false, layout_only = false;
    int64_t n = 0, rows = 0;
    int experts = 1, reps = 25, dev = 0;
    long long smpbo_flag = -1, nsm_flag = -1;
    size_t budget = 64ull << 20;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "mmq probe: %s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--sweep") sweep = true;
        else if (a == "--parity") do_parity = true;
        else if (a == "--allow-big") allow_big = true;
        else if (a == "--layout") layout_only = true;
        else if (a == "--smpbo") smpbo_flag = std::atoll(next("--smpbo"));
        else if (a == "--nsm") nsm_flag = std::atoll(next("--nsm"));
        else if (a == "--n") n = std::atoll(next("--n"));
        else if (a == "--rows") rows = std::atoll(next("--rows"));
        else if (a == "--experts") experts = std::atoi(next("--experts"));
        else if (a == "--reps") reps = std::atoi(next("--reps"));
        else if (a == "--device") dev = std::atoi(next("--device"));
        else if (a == "--budget-mib") budget = (size_t) std::atoll(next("--budget-mib")) << 20;
        else { std::fprintf(stderr, "mmq probe: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (reps < 20 || reps % 2 == 0) {
        std::fprintf(stderr, "mmq probe: --reps is an odd number of at least 20 (so the median is a sample)\n");
        return 2;
    }
    if (experts < 1) { std::fprintf(stderr, "mmq probe: --experts is >= 1\n"); return 2; }

    // --layout answers the block-layout and byte questions with no CUDA call at all: the tile config is a host
    // constexpr table and the byte counts are arithmetic, so the card's smpbo/nsm are taken from flags instead.
    if (layout_only) {
        std::printf("mmq layout (no CUDA call): type %s, K=%lld, experts=%d, smpbo=%lld B, nsm=%lld\n",
                    ggml_type_name((ggml_type) kType), (long long) kK, experts, smpbo_flag, nsm_flag);
        for (int64_t r : {2048LL, 8192LL}) {
            for (int64_t w : {2560LL, 1280LL, 640LL}) {
                const Point p{w, kK, r, experts, reps};
                std::printf("N=%lld T=%lld experts=%d: weights %zu B/expert (%zu B total), q8 %zu B, dst %zu B, "
                            "one-card footprint %zu MiB\n", (long long) w, (long long) r, experts,
                            mmq::matrix_bytes(kType, w, kK), mmq::matrix_bytes(kType, w, kK) * (size_t) experts,
                            mmq::q8_bytes(r, kK), (size_t) r * w * 4, footprint(p) >> 20);
                print_layout(w, r / experts, smpbo_flag < 0 ? 0 : (size_t) smpbo_flag,
                             nsm_flag < 0 ? 0 : (int) nsm_flag);
            }
        }
        return 0;
    }

    int count = 0;
    if (!fail(cudaGetDeviceCount(&count), "device count")) return 1;
    if (dev < 0 || dev >= count) { std::fprintf(stderr, "mmq probe: no card %d (%d present)\n", dev, count); return 1; }
    if (!fail(cudaSetDevice(dev), "set device")) return 1;
    if (!mmq::built()) { std::fprintf(stderr, "mmq probe: this build has no MMQ path\n"); return 1; }
    if (!mmq::supported(kType)) { std::fprintf(stderr, "mmq probe: NVFP4 is not covered\n"); return 1; }

    std::printf("mmq width probe: card %d, type %s, K=%lld, experts=%d, reps=%d, seed w=%u x=%u, budget=%llu MiB\n",
                dev, ggml_type_name((ggml_type) kType), (long long) kK, experts, reps, kSeedW, kSeedX,
                (unsigned long long) (budget >> 20));

    mmq::Context ctx;

    if (!sweep && n > 0 && rows > 0) {
        const Point p{n, kK, rows, experts, reps};
        return run_point(ctx, p, dev, budget, allow_big, false, nullptr, nullptr) ? 0 : 1;
    }

    // the decision grid: three widths x two chunk depths, then the slice check at both depths
    static const int64_t kWidths[] = {2560, 1280, 640};
    static const int64_t kRows[] = {2048, 8192};
    std::map<std::pair<int64_t, int64_t>, double> ns_per_row;   // {(width, rows): median}
    for (int64_t r : kRows) {
        for (int64_t w : kWidths) {
            const Point p{w, kK, r, experts, reps};
            std::vector<double> v;
            if (run_point(ctx, p, dev, budget, allow_big, false, nullptr, &v) && !v.empty())
                ns_per_row[{w, r}] = v[0];
        }
    }
    if (do_parity) {
        for (int64_t r : kRows) {
            parity(ctx, Point{2560, kK, r, experts, reps}, Point{640, kK, r, experts, reps}, dev, budget, allow_big);
            parity(ctx, Point{1280, kK, r, experts, reps}, Point{640, kK, r, experts, reps}, dev, budget, allow_big);
        }
    }

    // the multiplier table: per-row cost against the widest point at the same depth
    std::printf("\nMMQ width multipliers (per-row cost against N=2560 at the same T; 1.000 is perfectly flat):\n");
    std::printf("  T          N=2560 ns/row   N=1280 mult   N=640 mult\n");
    for (int64_t r : kRows) {
        const auto base = ns_per_row.find({2560, r});
        if (base == ns_per_row.end()) { std::printf("  %-10lld (the N=2560 point did not run)\n", (long long) r); continue; }
        auto cell = [&](int64_t w) -> std::string {
            const auto it = ns_per_row.find({w, r});
            if (it == ns_per_row.end()) return "     n/a   ";
            char buf[32];
            std::snprintf(buf, sizeof buf, "   %7.3fx", it->second / base->second);
            return buf;
        };
        std::printf("  %-10lld   %9.1f %s %s\n", (long long) r, base->second, cell(1280).c_str(), cell(640).c_str());
    }
    return 0;
}

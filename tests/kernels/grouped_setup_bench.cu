// tests/kernels/cpu/../../kernels/grouped_setup_bench.cu
// Is the grouped expert GEMM's cost per-GROUP setup or per-entry math?
//
// The decode window runs it at M ~= 3 tokens with ~16 groups a layer, and `VRAM hits` is
// 37% of a GDN layer's time - the largest single term. A persistent/fused kernel would
// attack the per-group setup, so the size of that prize depends on which one binds.
//
// The test holds TOTAL ENTRIES CONSTANT and varies how they are split:
//   A x G entries, as G groups of A each.
//   * time flat in G  -> the cost is per-entry math; a persistent kernel buys nothing.
//   * time rising with G -> the cost is per-group setup; the rise IS the prize.
// M (tokens per group) is varied the same way at constant entries.
//
//     build/grouped_setup_bench [usable_entries] [reps]
//
// Real decode geometry: n_embd 2560, n_ff 640, NVFP4 blob.
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/iq_kernels.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int64_t kNEmbd = 2560, kNFF = 640;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T>
T* dev_alloc(size_t n) {
    T* p = nullptr;
    check(cudaMalloc((void**) &p, n * sizeof(T)), "malloc");
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    const int64_t usable = argc > 1 ? std::atoll(argv[1]) : 48;   // total entries held constant
    const int reps = argc > 2 ? std::atoi(argv[2]) : 200;

    basalt::kernels::cpu::NativeFmt f;
    std::string err;
    if (!basalt::kernels::cpu::native_fmt(40, 40, kNEmbd, kNFF, f, err)) {
        std::printf("native_fmt: %s\n", err.c_str());
        return 1;
    }
    const auto L = basalt::kernels::native_expert_layout(40, 40, kNEmbd, kNFF);
    std::printf("blob %.2f MiB, n_embd %lld, n_ff %lld, %lld total entries\n", (double) L.bytes / 1048576.0,
                (long long) kNEmbd, (long long) kNFF, (long long) usable);

    // one blob, shared by every group (the math is identical; the point is timing, not data)
    const int64_t max_groups = 64;
    std::vector<uint8_t> host_blob(L.bytes, 0x11);
    uint8_t* blob = dev_alloc<uint8_t>(L.bytes);
    check(cudaMemcpy(blob, host_blob.data(), L.bytes, cudaMemcpyHostToDevice), "memcpy blob");
    std::vector<unsigned long long> ptr(max_groups);
    for (int64_t g = 0; g < max_groups; ++g) ptr[(size_t) g] = (unsigned long long) blob;
    auto* d_ptr = dev_alloc<unsigned long long>(max_groups);

    // activations: one row of q8_1 per entry (the real kernel takes one per entry)
    const size_t act_row = basalt::kernels::cpu::kNativeActBytes;
    std::vector<uint8_t> host_act(act_row * (size_t) usable, 0x01);
    uint8_t* d_act = dev_alloc<uint8_t>(act_row * (size_t) usable);
    check(cudaMemcpy(d_act, host_act.data(), act_row * (size_t) usable, cudaMemcpyHostToDevice), "memcpy act");

    auto* d_start = dev_alloc<int32_t>(max_groups + 1);
    auto* d_ng = dev_alloc<int32_t>(1);
    auto* d_tok = dev_alloc<int32_t>(usable);
    auto* d_dst = dev_alloc<int32_t>(usable);
    std::vector<int32_t> tok((size_t) usable), dst((size_t) usable);
    for (int64_t e = 0; e < usable; ++e) {
        tok[(size_t) e] = (int32_t) (e % 8);
        dst[(size_t) e] = (int32_t) e;
    }
    check(cudaMemcpy(d_tok, tok.data(), usable * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy tok");
    check(cudaMemcpy(d_dst, dst.data(), usable * sizeof(int32_t), cudaMemcpyDeviceToDevice), "memcpy dst");

    const size_t scratch_bytes = basalt::kernels::native_expert_scratch_bytes(usable, kNFF);
    void* scratch = nullptr;
    check(cudaMalloc(&scratch, scratch_bytes), "malloc scratch");
    float* out = dev_alloc<float>((size_t) usable * (size_t) kNEmbd);

    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream create");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // run(G, A): G groups of A entries each, G*A == usable
    auto run = [&](int64_t G, int64_t A) {
        std::vector<int32_t> st((size_t) G + 1);
        for (int64_t g = 0; g <= G; ++g) st[(size_t) g] = (int32_t) (g * A);
        check(cudaMemcpyAsync(d_ptr, ptr.data(), (size_t) G * sizeof(unsigned long long), cudaMemcpyHostToDevice, s),
              "ptr");
        check(cudaMemcpyAsync(d_start, st.data(), (size_t) (G + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, s),
              "start");
        const int32_t ng = (int32_t) G;
        check(cudaMemcpyAsync(d_ng, &ng, sizeof ng, cudaMemcpyHostToDevice, s), "ng");
        check(cudaStreamSynchronize(s), "sync setup");
        for (int i = 0; i < 5; ++i)   // warm
            basalt::kernels::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, max_groups, usable, d_act,
                                                   scratch, out, s);
        check(cudaStreamSynchronize(s), "warm sync");
        cudaEventRecord(e0, s);
        for (int i = 0; i < reps; ++i)
            basalt::kernels::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, max_groups, usable, d_act,
                                                   scratch, out, s);
        cudaEventRecord(e1, s);
        check(cudaStreamSynchronize(s), "timed sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return (double) ms / (double) reps;
    };

    std::printf("\n=== A: total entries constant (%lld), vary the NUMBER OF GROUPS ===\n", (long long) usable);
    std::printf("%6s %8s %10s %12s %12s\n", "groups", "per grp", "ms/call", "us/group", "ns/entry");
    double base = 0;
    for (int64_t G : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48}) {
        if (G > usable) continue;
        const int64_t A = usable / G;
        if (A < 1) continue;
        const double ms = run(G, A);
        if (G == 1) base = ms;
        std::printf("%6lld %8lld %10.4f %12.2f %12.1f\n", (long long) G, (long long) A, ms, ms * 1000.0 / (double) G,
                    ms * 1e6 / (double) (G * A));
    }

    std::printf("\n=== B: TOTAL ENTRIES constant, vary M (tokens per group), groups = entries/M ===\n");
    std::printf("%6s %8s %10s %12s\n", "M", "groups", "ms/call", "vs M=1");
    const int64_t tot_b = usable;
    double b1 = 0;
    for (int64_t M : {1, 2, 3, 4, 6, 8, 12, 16}) {
        const int64_t G = tot_b / M;
        if (G < 1) continue;
        const int64_t A = M;
        const double ms = run(G, A);
        if (M == 1) b1 = ms;
        std::printf("%6lld %8lld %10.4f %11.2fx\n", (long long) M, (long long) G, ms, b1 > 0 ? ms / b1 : 0.0);
    }

    std::printf(
        "\nHow to read it:\n"
        "  A: if ms/call is FLAT as groups rise at constant entries, cost is per-entry math ->\n"
        "     a persistent/fused kernel buys nothing. If it RISES, the rise is per-group setup\n"
        "     and that rise is the prize.\n"
        "  B: if time FALLS as M rises at constant entries, the same entries cost less when\n"
        "     batched wider - the fixed cost is being amortized.\n");

    cudaFree(blob); cudaFree(d_ptr); cudaFree(d_act); cudaFree(d_start); cudaFree(d_ng);
    cudaFree(d_tok); cudaFree(d_dst);
    cudaFree(scratch); cudaFree(out);
    return 0;
}

// src/kernels/cuda/qsa_chain.cu - the fingerprint kernel of the per-layer chain.  See the header for what the
// chain is for; everything that decides an ANSWER (which rows, how they combine) lives in the header so the host
// that grades the chain and the kernel that takes it cannot drift.
#include "basalt/kernels/qsa_chain.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace basalt::kernels {
namespace {

__global__ void qsa_chain_kernel(QsaChainFp* __restrict__ dst, const uint32_t* __restrict__ w, int64_t rows,
                                 int64_t stride_words) {
    __shared__ uint32_t sh[kChainBlock];
    __shared__ float sf[kChainBlock];
    __shared__ float sm[kChainBlock];
    const int tid = threadIdx.x;
    const int64_t sampled = qsa_chain_sampled(rows);
    const int64_t total = sampled * stride_words;

    uint32_t h = 0;
    float s = 0.0f, mx = 0.0f;
    for (int64_t e = tid; e < total; e += kChainBlock) {
        const int64_t r = e / stride_words, c = e % stride_words;
        const int64_t row = qsa_chain_row((int) r, rows);
        const uint32_t bits = w[(size_t) row * (size_t) stride_words + c];
        h += qsa_chain_mix(bits, (uint32_t) r, (uint32_t) c);
        const float v = __uint_as_float(bits);
        s += v;
        mx = qsa_chain_max(mx, fabsf(v));
    }

    // The reduction is a fixed tree, not an atomic: the chain's whole value is that two boots of one prompt
    // print the same figures unless something really moved, and a float sum accumulated in a schedule-dependent
    // order would print different figures every boot.
    sh[tid] = h; sf[tid] = s; sm[tid] = mx;
    __syncthreads();
    for (int off = kChainBlock / 2; off > 0; off >>= 1) {
        if (tid < off) {
            sh[tid] += sh[tid + off];
            sf[tid] += sf[tid + off];
            sm[tid] = qsa_chain_max(sm[tid], sm[tid + off]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst->hash = sh[0];
        dst->sum = sf[0];
        dst->maxabs = sm[0];
    }
}

}  // namespace

const char* qsa_chain_slot_name(int slot) {
    static const char* const names[kChainCount] = {
        "mix", "steps", "sel", "kpool", "vpool", "attn", "gated", "partl", "suml", "bo",
        "pub", "stepsr", "qidxr", "idxraw", "kspan", "vspan", "q", "selr", "attnp", "gatedp", "partp", "sump",
    };
    if (slot < 0 || slot >= kChainCount) return "?";
    return names[slot];
}

QsaChainFp qsa_chain_ref(const uint32_t* words, int64_t rows, int64_t stride_words) {
    QsaChainFp fp;   // unrecorded, until a window exists to fingerprint
    const int64_t sampled = qsa_chain_sampled(rows);
    if (words == nullptr || sampled <= 0 || stride_words <= 0) return fp;

    uint32_t sh[kChainBlock];
    float sf[kChainBlock], sm[kChainBlock];
    const int64_t total = sampled * stride_words;
    for (int t = 0; t < kChainBlock; ++t) {
        uint32_t h = 0;
        float s = 0.0f, mx = 0.0f;
        for (int64_t e = t; e < total; e += kChainBlock) {
            const int64_t r = e / stride_words, c = e % stride_words;
            const int64_t row = qsa_chain_row((int) r, rows);
            const uint32_t bits = words[(size_t) row * (size_t) stride_words + c];
            h += qsa_chain_mix(bits, (uint32_t) r, (uint32_t) c);
            float v;
            std::memcpy(&v, &bits, sizeof v);
            s += v;
            mx = qsa_chain_max(mx, v < 0.0f ? -v : v);
        }
        sh[t] = h; sf[t] = s; sm[t] = mx;
    }
    for (int off = kChainBlock / 2; off > 0; off >>= 1)
        for (int t = 0; t < off; ++t) {
            sh[t] += sh[t + off];
            sf[t] += sf[t + off];
            sm[t] = qsa_chain_max(sm[t], sm[t + off]);
        }
    fp.hash = sh[0]; fp.sum = sf[0]; fp.maxabs = sm[0];
    return fp;
}

bool qsa_chain_record(QsaChainFp* dst, const void* base, int64_t rows, int64_t stride_words, void* stream,
                      std::string& err) {
    if (dst == nullptr || base == nullptr || stream == nullptr) {
        err = "qsa chain: no slot, no component or no stream";
        return false;
    }
    if (reinterpret_cast<uintptr_t>(base) % 4 != 0) {
        err = "qsa chain: a component was handed to the fingerprint unaligned";
        return false;
    }
    if (qsa_chain_sampled(rows) <= 0 || stride_words <= 0) {
        err = "qsa chain: a " + std::to_string(rows) + " x " + std::to_string(stride_words) +
              " component has nothing to fingerprint";
        return false;
    }
    qsa_chain_kernel<<<1, kChainBlock, 0, (cudaStream_t) stream>>>(
        dst, reinterpret_cast<const uint32_t*>(base), rows, stride_words);
    if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
        err = std::string("qsa chain: the fingerprint kernel did not launch: ") + cudaGetErrorString(le);
        return false;
    }
    return true;
}

std::string qsa_chain_field(int slot, const QsaChainFp& fp) {
    std::string out = qsa_chain_slot_name(slot);
    out += '=';
    if (fp.hash == kChainUnrecorded) {
        out += "--------";   // nobody recorded this component: a mode that never reached it, not a buffer of zeros
        return out;
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%08x/%.8g/%.8g", fp.hash, fp.sum, fp.maxabs);
    out += buf;
    return out;
}

}  // namespace basalt::kernels

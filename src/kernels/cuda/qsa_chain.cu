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
    // One chunk of the sketch at a time: `kChainProj` rows of `kChainBlock` floats would blow the static shared
    // budget, and the instrument may not pay for dynamic shared allocation to be allowed to exist.
    __shared__ float sp[kChainProjChunk][kChainBlock];
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
    // order would print different figures every boot. The sketch reduces in the same order, chunk by chunk.
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

    // THE SKETCH, in one walk of the window. `kChainProj` register accumulators per thread keep the window read
    // once however large the sketch is (re-walking it per chunk of projections cost 5x the base fingerprint, which
    // a debug flag may not); the tree then reduces it `kChainProjChunk` rows at a time to bound shared memory.
    float pr[kChainProj];
    for (int p = 0; p < kChainProj; ++p) pr[p] = 0.0f;
    for (int64_t e = tid; e < total; e += kChainBlock) {
        const int64_t r = e / stride_words, c = e % stride_words;
        const int64_t row = qsa_chain_row((int) r, rows);
        const float v = __uint_as_float(w[(size_t) row * (size_t) stride_words + c]);
        for (int p = 0; p < kChainProj; ++p)
            pr[p] += v * qsa_chain_sign(p, (uint32_t) r, (uint32_t) c);
    }
    for (int base = 0; base < kChainProj; base += kChainProjChunk) {
        for (int q = 0; q < kChainProjChunk; ++q) sp[q][tid] = pr[base + q];
        __syncthreads();
        for (int off = kChainBlock / 2; off > 0; off >>= 1) {
            if (tid < off)
                for (int q = 0; q < kChainProjChunk; ++q) sp[q][tid] += sp[q][tid + off];
            __syncthreads();
        }
        if (tid == 0)
            for (int q = 0; q < kChainProjChunk && base + q < kChainProj; ++q) dst->proj[base + q] = sp[q][0];
    }
}

}  // namespace

const char* qsa_chain_slot_name(int slot) {
    static const char* const names[kChainCount] = {
        "mix", "steps", "sel", "kpool", "vpool", "attn", "gated", "colsl", "partl", "suml", "bo",
        "pub", "stepsr", "qidxr", "idxraw", "kspan", "vspan", "q", "selr", "attnp", "gatedp", "colsp", "partp",
        "sump",
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

    // The sketch, mirroring the kernel chunk for chunk: same window, same per-thread strided walk, same
    // increasing-`e` accumulation, same tree. `qsa_chain_parity` holds both sides and fails on any drift, which is
    // the only reason a cosine read off two boots' log lines means anything.
    for (int base = 0; base < kChainProj; base += kChainProjChunk) {
        float sp[kChainProjChunk][kChainBlock];
        for (int t = 0; t < kChainBlock; ++t) {
            float pr[kChainProjChunk];
            for (int q = 0; q < kChainProjChunk; ++q) pr[q] = 0.0f;
            for (int64_t e = t; e < total; e += kChainBlock) {
                const int64_t r = e / stride_words, c = e % stride_words;
                const int64_t row = qsa_chain_row((int) r, rows);
                const uint32_t bits = words[(size_t) row * (size_t) stride_words + c];
                float v;
                std::memcpy(&v, &bits, sizeof v);
                for (int q = 0; q < kChainProjChunk; ++q)
                    pr[q] += v * qsa_chain_sign(base + q, (uint32_t) r, (uint32_t) c);
            }
            for (int q = 0; q < kChainProjChunk; ++q) sp[q][t] = pr[q];
        }
        for (int off = kChainBlock / 2; off > 0; off >>= 1)
            for (int t = 0; t < off; ++t)
                for (int q = 0; q < kChainProjChunk; ++q) sp[q][t] += sp[q][t + off];
        for (int q = 0; q < kChainProjChunk && base + q < kChainProj; ++q) fp.proj[base + q] = sp[q][0];
    }
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
    // THE SKETCH, printed as fixed-width hex so two boots' lines can be compared by eye and parsed by a script
    // without a float parser. The chain line is a debug instrument behind `--exp qsa_dump_chain=1`; it is not a
    // line the server parses. It costs ~1.9 KiB of log per component (`kChainProj` × 8 hex digits + separators),
    // so a full sliced boot prints a few MB - the price of the only cross-boot comparison this instrument can make
    // that kernel nondeterminism cannot confound.
    out += '/';
    for (int p = 0; p < kChainProj; ++p) {
        uint32_t bits;
        std::memcpy(&bits, &fp.proj[p], sizeof bits);
        char pb[16];
        std::snprintf(pb, sizeof pb, "%08x", bits);
        if (p) out += ',';
        out += pb;
    }
    return out;
}

}  // namespace basalt::kernels

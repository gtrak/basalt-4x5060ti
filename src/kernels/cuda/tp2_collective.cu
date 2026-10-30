// src/kernels/cuda/tp2_collective.cu - the two kernels of the 2-way all-reduce and the pair's device state.  See the
// header for the protocol and for why this is not NCCL.
#include "basalt/kernels/tp2_collective.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>

namespace basalt::kernels {
namespace {

constexpr int kTp2Threads = 256;
constexpr int kTp2Blocks = 128;         // enough blocks to saturate a PCIe/PHB copy at the sizes a partial has
// how long one block waits for its peer's push before the pair is declared dead: 2 s of SM clock at 1 GHz is a
// conservative figure, and it is a lifetime, not a poll interval - a healthy wait is a few microseconds.
constexpr long long kTp2SpinBudget = 2000000000ll;

__device__ __forceinline__ int32_t tp2_ld_flag(const int32_t* p) {
    int32_t v;
    asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void tp2_st_flag(int32_t* p, int32_t v) {
    asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

/// After my push has landed (stream order guarantees it), publish the sequence number on the peer and advance mine.
/// One thread: the whole word is the message.
__global__ void tp2_push_flag(int32_t* my_seq, int32_t* peer_pushed) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int32_t v = my_seq[0] + 1;
    my_seq[0] = v;                 // my wait kernel below reads this
    __threadfence_system();        // the pushed bytes are visible before the flag is
    tp2_st_flag(peer_pushed, v);
}

/// `buf[i] += peer_partial[i]` once the peer's push of THIS collective has arrived; `buf` is this rank's own partial
/// and `peer_partial` is the peer's, sitting in this card's staging buffer.
///
/// Every block waits on the same flag, which is why this cannot deadlock with the peer: the peer's `push_flag` runs
/// on the PEER's SMs, so a block here holding this card's SMs never blocks the thing it waits for.
__global__ void tp2_add_wait(float* __restrict__ buf, const float* __restrict__ peer_partial, const int32_t* pushed,
                             const int32_t* my_seq, volatile int32_t* timeout, int64_t n) {
    __shared__ int32_t ready;
    if (threadIdx.x == 0) {
        const int32_t want = my_seq[0];
        const long long t0 = clock64();
        int32_t got = 0;
        while (true) {
            got = tp2_ld_flag(pushed);
            if (got >= want) break;
            if (clock64() - t0 > kTp2SpinBudget) {
                *timeout = 1;   // mapped host memory: the host sees it without a copy
                break;
            }
        }
        ready = got >= want;
    }
    __syncthreads();
    if (!ready) return;   // timed out: `buf` keeps this rank's partial, and the caller's error path runs
    const int64_t stride_grid = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride_grid)
        buf[i] = buf[i] + peer_partial[i];   // mine + the peer's: the order tp2_all_reduce_ref states
}

/// The pair-agreement sample: `kTp2Samples` elements of a summed partial, at the indices the host uses too.  One
/// warp, indices from the header's one rule, so a check and the thing it checks cannot drift.
__global__ void tp2_sample_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t rows,
                                  int64_t stride) {
    const int k = threadIdx.x;
    if (k >= kTp2Samples) return;
    const int64_t i = tp2_sample_index(k, rows, stride);
    dst[k] = i >= 0 ? src[i] : 0.0f;
}

}  // namespace

bool tp2_sample_partial(float* dst, const float* src, int64_t rows, int64_t stride, void* stream, std::string& err) {
    if (dst == nullptr || src == nullptr || stream == nullptr) {
        err = "tp2 sample: no buffer or no stream";
        return false;
    }
    if (tp2_sample_index(0, rows, stride) < 0 || tp2_sample_index(kTp2Samples - 1, rows, stride) < 0) {
        err = "tp2 sample: a " + std::to_string(rows) + " x " + std::to_string(stride) +
              " partial has no sample to take";
        return false;
    }
    tp2_sample_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(dst, src, rows, stride);
    if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
        err = std::string("tp2 sample: the sampling kernel did not launch: ") + cudaGetErrorString(le);
        return false;
    }
    return true;
}

Tp2Pair::~Tp2Pair() { close(); }

void Tp2Pair::close() {
    for (int r = 0; r < 2; ++r) {
        if (staging_[r]) cudaFree(staging_[r]);
        if (pushed_[r]) cudaFree(pushed_[r]);
        if (seq_[r]) cudaFree(seq_[r]);
        staging_[r] = nullptr;
        pushed_[r] = nullptr;
        seq_[r] = nullptr;
    }
    if (timeout_[0]) cudaFreeHost(timeout_[0]);
    if (timeout_[1]) cudaFreeHost(timeout_[1]);
    timeout_[0] = timeout_[1] = nullptr;
    max_elems_ = 0;
    issued_a_ = issued_b_ = 0;
    bytes_a_ = bytes_b_ = 0;
    dev_a_ = dev_b_ = -1;
    cudaGetLastError();   // a free on a device that went away is not this pair's problem
}

bool Tp2Pair::open(int dev_a, int dev_b, int64_t max_elems, std::string& err, int sets) {
    close();
    if (dev_a == dev_b || dev_a < 0 || dev_b < 0) {
        err = "tp2 all-reduce: a pair is two different cards";
        return false;
    }
    if (max_elems <= 0) {
        err = "tp2 all-reduce: the widest partial must be at least one float";
        return false;
    }
    if (sets < 1) {
        err = "tp2 all-reduce: a pair needs at least one staging set, not " + std::to_string(sets);
        return false;
    }
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, dev_a, dev_b) != cudaSuccess || !can ||
        cudaDeviceCanAccessPeer(&can, dev_b, dev_a) != cudaSuccess || !can) {
        err = "tp2 all-reduce: CUDA" + std::to_string(dev_a) + " and CUDA" + std::to_string(dev_b) +
              " have no GPU-to-GPU access";
        return false;
    }
    // the engine enables peer access for every pair it selects; asking again is free and makes the helper usable
    // on its own (which is what the parity test is)
    for (int r = 0; r < 2; ++r) {
        const int me = r == 0 ? dev_a : dev_b, other = r == 0 ? dev_b : dev_a;
        if (cudaSetDevice(me) != cudaSuccess) {
            err = "tp2 all-reduce: no CUDA" + std::to_string(me);
            return false;
        }
        if (cudaDeviceEnablePeerAccess(other, 0) != cudaSuccess) cudaGetLastError();   // already on: fine
        // One allocation, `sets` regions of `max_elems` floats each: collective n and collective n+sets are the
        // only pair that can want the same region, and the flag protocol orders them (tp2_collective.hpp).
        const uint64_t region = tp2_staging_elems(max_elems, sets);
        if (cudaMalloc((void**) &staging_[r], (size_t) region * 4) != cudaSuccess ||
            cudaMalloc((void**) &pushed_[r], sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &seq_[r], sizeof(int32_t)) != cudaSuccess) {
            err = "tp2 all-reduce: CUDA" + std::to_string(me) + " out of memory for the staging buffer";
            close();
            return false;
        }
        if (cudaMemset(staging_[r], 0, (size_t) region * 4) != cudaSuccess ||
            cudaMemset(pushed_[r], 0, sizeof(int32_t)) != cudaSuccess ||
            cudaMemset(seq_[r], 0, sizeof(int32_t)) != cudaSuccess) {
            err = "tp2 all-reduce: CUDA" + std::to_string(me) + " could not clear its staging buffer";
            close();
            return false;
        }
        // the timeout word is mapped host memory on THIS card, so the host reads it without a copy and a kernel
        // here writes it without crossing the link
        if (cudaHostAlloc((void**) &timeout_[r], sizeof(int32_t),
                          cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaMemset(timeout_[r], 0, sizeof(int32_t)) != cudaSuccess) {
            err = "tp2 all-reduce: no host word for the timeout on CUDA" + std::to_string(me);
            close();
            return false;
        }
    }
    if (cudaSetDevice(dev_a) != cudaSuccess) {
        err = "tp2 all-reduce: no CUDA" + std::to_string(dev_a);
        close();
        return false;
    }
    dev_a_ = dev_a;
    dev_b_ = dev_b;
    max_elems_ = max_elems;
    sets_ = sets > 1 ? sets : 1;
    return true;
}

bool Tp2Pair::timed_out(int rank) const {
    return rank >= 0 && rank < 2 && timeout_[rank] != nullptr && *(volatile int32_t*) timeout_[rank] != 0;
}

bool Tp2Pair::all_reduce(float* buf, int64_t rows, int64_t stride, void* stream, int peer_dev, int rank,
                         std::string& err, int set) {
    if (!valid()) {
        err = "tp2 all-reduce: the pair is not open";
        return false;
    }
    if (rank != 0 && rank != 1) {
        err = "tp2 all-reduce: a rank is 0 or 1, not " + std::to_string(rank);
        return false;
    }
    const int other = tp2_peer(rank);
    const int dev_me = rank == 0 ? dev_a_ : dev_b_, dev_other = other == 0 ? dev_a_ : dev_b_;
    if (peer_dev != dev_other) {
        err = "tp2 all-reduce: rank " + std::to_string(rank) + "'s peer is CUDA" + std::to_string(dev_other) +
              ", not CUDA" + std::to_string(peer_dev);
        return false;
    }
    if (buf == nullptr || stream == nullptr) {
        err = "tp2 all-reduce: no buffer or no stream";
        return false;
    }
    if (!tp2_all_reduce_ok(rows, stride, max_elems_)) {
        err = "tp2 all-reduce: a " + std::to_string(rows) + " x " + std::to_string(stride) +
              " partial does not fit the pair's staging (" + std::to_string(max_elems_) + " floats)";
        return false;
    }
    // Both ranks must name the same region for the same collective; a caller that runs two layers at once and
    // passes one set for both gets a sum built from the peer's NEXT layer's partial, so this is refused, not warned.
    if (!tp2_staging_set_ok(set, sets_)) {
        err = "tp2 all-reduce: staging set " + std::to_string(set) + " is not one of this pair's " +
              std::to_string(sets_) + "; a pair running this deep needs open(..., sets) raised to match";
        return false;
    }
    const cudaStream_t s = (cudaStream_t) stream;
    if (cudaSetDevice(dev_me) != cudaSuccess) {
        err = "tp2 all-reduce: cannot run on CUDA" + std::to_string(dev_me);
        return false;
    }
    const uint64_t bytes = tp2_all_reduce_bytes(rows, stride);
    const int64_t off = (int64_t) set * max_elems_;
    // 1. push my partial into the peer's staging (one peer copy; the link the pair already has)
    if (cudaMemcpyPeerAsync(staging_[other] + off, dev_other, buf, dev_me, bytes, s) != cudaSuccess) {
        err = "tp2 all-reduce: the push to CUDA" + std::to_string(dev_other) + " failed";
        return false;
    }
    // 2. raise the peer's flag once it has landed, and 3. add the peer's push into my buffer once mine has
    tp2_push_flag<<<1, 32, 0, s>>>(seq_[rank], pushed_[other]);
    const int blocks = (int) std::min<int64_t>(kTp2Blocks, (rows * stride + kTp2Threads - 1) / kTp2Threads);
    tp2_add_wait<<<blocks, kTp2Threads, 0, s>>>(buf, staging_[rank] + off, pushed_[rank], seq_[rank], timeout_[rank],
                                                rows * stride);
    if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
        err = std::string("tp2 all-reduce: the collective on CUDA") + std::to_string(dev_me) + " did not launch: " +
              cudaGetErrorString(le);
        return false;
    }
    if (rank == 0) { ++issued_a_; bytes_a_ += bytes; } else { ++issued_b_; bytes_b_ += bytes; }
    return true;
}

}  // namespace basalt::kernels

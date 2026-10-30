// tools/tp_ring_probe.cu - the pair's and the ring's effective link bytes, and the collective's round-trip
// latency, measured, for the TP decision.
//
// WHAT THIS IS. A standalone CUDA program, not wired into the engine and not built by CMake: it exists to answer
// two questions - how many bytes a second per link do these four cards actually move when several links are busy
// at once, and how long one collective's round trip takes at the payload sizes a split linear actually has -
// before any re-shard work is priced against them. The patterns are the ones the topologies really make: the 12
// ordered pairs alone, one pair both directions, both pair links at once (PP=2's two AR pairs), the 4-deep ring,
// and all-to-all. The engine's own collective (`Tp2Pair::all_reduce`, src/kernels/cuda/tp2_collective.cu) pushes
// with `cudaMemcpyPeerAsync`, so that is the primitive the AR mode copies - push, flag, wait/add, the same three
// issues per rank; an SM-driven push is offered beside it because a pull-style collective would read peer memory
// instead of writing it, and because the copy engine is a queue the tier fill also uses.
//
// WHY IT IS NOT THE ENGINE'S TEST. `tp2_allreduce_parity_test` proves the collective is CORRECT; it says nothing
// about how much of the fabric one link gets when its neighbours are hot, or what an AR costs while blobs are
// streaming. The roadmap turns on those two numbers, so they are measured here, once, and written into the notes
// - not re-derived per change.
//
// BUILD (deliberately not a CMake target: this is a probe, not a shipped artifact)
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 -o /tmp/tp_ring_probe tools/tp_ring_probe.cu
// RUN
//   /tmp/tp_ring_probe                                  # every bandwidth pattern, the default size sweep
//   /tmp/tp_ring_probe --mode ring                      # one pattern (pairs pairbidir threelink pair2 ring all2all)
//   /tmp/tp_ring_probe --mode ar        --reps 101      # the collective's round trip, idle, copy engine
//   /tmp/tp_ring_probe --mode ar        --reps 101 --ar-sm          # ... pushed by SMs
//   /tmp/tp_ring_probe --mode artier    --reps 31 --bg-blobs 4      # ... with expert blobs in flight behind it
//   /tmp/tp_ring_probe --size 4MiB --iters 20           # one size, one repeat count
//
// BUFFER DISCIPLINE (the production server is LIVE on these four cards). The probe checks `cudaMemGetInfo` on
// every card it is about to touch and REFUSES to run if any has less than 256 MiB free. Its heaviest point holds
// at most 60 MiB on a card - two bandwidth buffers of at most 31 MiB, or the AR's two small buffers beside the
// tier load's two blob buffers - it frees everything between measurement points, and touches no stream but its
// own. If the server's allocations have grown, the probe exits with a non-zero status and says which card is too
// full; that is the correct outcome, not a failure to fix.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kMaxDevices = 4;
constexpr size_t kMinFreeBytes = 256ull << 20;      // refuse to run a card with less than this free
constexpr size_t kMaxBufferBytes = 32ull << 20;     // one buffer; two per card = the 64 MiB the discipline allows
/// One expert blob of this pack: gate/up 2*640 rows x 1440 B + down 2560 rows x 360 B + the 16 B scale tail
/// (`NativeFmt`), the unit a tier fill actually moves. The boot line's "largest blob 2.76 MB" is this figure.
constexpr size_t kTierBlob = 2u * 640u * 1440u + 2560u * 360u + 16u;

struct Card {
    int dev = -1;
    void* src = nullptr;   // what this card pushes
    void* dst = nullptr;   // what lands on this card
    int32_t* flag = nullptr;    // the peer's "my push landed" word, on THIS card
    int32_t* seq = nullptr;     // this card's own issue counter
    int32_t* timeout = nullptr; // mapped host memory: this card's own giving-up word
    void* bg_src = nullptr;     // the tier-load stream's own source, sized for the blob-like copy
    void* bg_dst = nullptr;     // ... and its own destination, so it never shares the AR's buffers
    cudaStream_t s = nullptr;
    cudaEvent_t t0 = nullptr, t1 = nullptr;
};

struct Link {
    int from = 0, to = 0;   // card indices, not device ids
    double gib_s = 0;       // the measured rate of this link during the pattern
};

bool fail(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "probe: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

/// The copy the engine's collective makes: the copy engine writes into the peer's buffer.
bool copy_engine(Card& from, Card& to, size_t bytes, cudaStream_t s) {
    return fail(cudaSetDevice(from.dev), "set source device") &&
           fail(cudaMemcpyPeerAsync(to.dst, to.dev, from.src, from.dev, bytes, s), "peer copy");
}

__global__ void sm_copy(float4* __restrict__ dst, const float4* __restrict__ src, size_t n4) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) dst[i] = src[i];
}

// ================================ the collective, as the engine runs it ================================
//
// A copy of `Tp2Pair`'s three-step exchange (src/kernels/cuda/tp2_collective.cu): push my partial into the peer's
// staging, raise the peer's flag once it landed, add the peer's push into my buffer once mine arrived. Copying the
// protocol rather than linking the helper is deliberate: the probe must measure the SAME three issues the engine
// issues (one peer copy and two kernels per rank per AR), but it must not need the engine, its build or a pair
// handed to it by generate.cpp. If that file's protocol changes, this copy has to follow it - the numbers below are
// only evidence while the two agree, which is why the kernel names and the spin budget are the engine's.
constexpr int kTp2Threads = 256;
constexpr int kTp2Blocks = 128;
constexpr long long kTp2SpinBudget = 2000000000ll;

__device__ __forceinline__ int32_t ld_flag(const int32_t* p) {
    int32_t v;
    asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_flag(int32_t* p, int32_t v) {
    asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__global__ void push_flag(int32_t* my_seq, int32_t* peer_pushed) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int32_t v = my_seq[0] + 1;
    my_seq[0] = v;
    __threadfence_system();
    st_flag(peer_pushed, v);
}

__global__ void add_wait(float* __restrict__ buf, const float* __restrict__ peer_partial, const int32_t* pushed,
                         const int32_t* my_seq, volatile int32_t* timeout, int64_t n) {
    __shared__ int32_t ready;
    if (threadIdx.x == 0) {
        const int32_t want = my_seq[0];
        const long long t0 = clock64();
        int32_t got = 0;
        while (true) {
            got = ld_flag(pushed);
            if (got >= want) break;
            if (clock64() - t0 > kTp2SpinBudget) {
                *timeout = 1;
                break;
            }
        }
        ready = got >= want;
    }
    __syncthreads();
    if (!ready) return;
    const int64_t stride_grid = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride_grid)
        buf[i] = buf[i] + peer_partial[i];
}

/// One rank's issue of the collective, the engine's three issues in the engine's order. `sm_push` picks the
/// push's engine: the copy engine, as the helper uses it, or SMs writing across the link. The second matters
/// because the measurements below show a `cudaMemcpyPeerAsync` push waits for the copy engine to come free, so a
/// tier fill in flight turns a 25 us round trip into a millisecond-scale one; an SM push does not queue there.
bool ar_issue(Card& me, Card& peer, size_t bytes, cudaStream_t s, bool sm_push) {
    if (!fail(cudaSetDevice(me.dev), "set device for the AR")) return false;
    if (sm_push) {
        const size_t n4 = bytes / 16;
        const int blocks = (int) std::min<size_t>(512, (n4 + 255) / 256);
        sm_copy<<<blocks, 256, 0, s>>>((float4*) peer.dst, (const float4*) me.src, n4);
        if (!fail(cudaGetLastError(), "the AR's SM push")) return false;
    } else if (!fail(cudaMemcpyPeerAsync(peer.dst, peer.dev, me.src, me.dev, bytes, s), "the AR's push")) {
        return false;
    }
    push_flag<<<1, 32, 0, s>>>(me.seq, peer.flag);
    const int blocks = (int) std::min<int64_t>(kTp2Blocks, (int64_t)(bytes / 4 + kTp2Threads - 1) / kTp2Threads);
    add_wait<<<blocks, kTp2Threads, 0, s>>>((float*) me.src, (const float*) me.dst, me.flag, me.seq, me.timeout,
                                            (int64_t)(bytes / 4));
    return fail(cudaGetLastError(), "the AR's kernels");
}

/// A tier-fill copy engine load running behind the ARs: the shape of traffic a prefill chunk puts on the same link
/// (a blob-sized peer copy, back to back). Without it the latency figures below would be an idle-fabric number.
/// It runs on the card's own `bg_src`/`bg_dst` pair, never the AR's, so the two never alias.
bool bg_push(Card& me, Card& peer, size_t bytes, cudaStream_t s) {
    if (!fail(cudaSetDevice(me.dev), "set device for the background copy")) return false;
    return fail(cudaMemcpyPeerAsync(peer.bg_dst, peer.dev, me.bg_src, me.dev, bytes, s), "background peer copy");
}

/// The pull an IPC-style collective makes: SMs on the DESTINATION read across the link.
bool copy_sm(Card& from, Card& to, size_t bytes, cudaStream_t s) {
    if (!fail(cudaSetDevice(to.dev), "set destination device")) return false;
    const size_t n4 = bytes / 16;
    const int blocks = (int) std::min<size_t>(512, (n4 + 255) / 256);
    sm_copy<<<blocks, 256, 0, s>>>((float4*) to.dst, (const float4*) from.src, n4);
    return fail(cudaGetLastError(), "sm copy");
}

bool ref_fill(Card& c, size_t bytes) {
    if (!fail(cudaSetDevice(c.dev), "set device for the source fill")) return false;
    return fail(cudaMemset(c.src, 0x3f, bytes), "fill the source");
}

/// What the probe can hold on one card at its heaviest point: the bandwidth sweep's two buffers at the largest
/// sweep size, or the AR's two small buffers plus the tier load's two blob buffers. The modes never coexist, so
/// the larger of the two is the honest figure to check the floor against.
constexpr size_t kMaxPerCard = (31457280ull << 1) > (262144ull << 1) + (kTierBlob << 1)
                                   ? (31457280ull << 1)
                                   : (262144ull << 1) + (kTierBlob << 1);

/// Every card's free memory, before anything is allocated. The server's own bytes are what this is checking
/// against; the probe is a guest on these cards.
bool check_free(const std::vector<Card>& cards) {
    for (const Card& c : cards) {
        if (cudaSetDevice(c.dev) != cudaSuccess) {
            std::fprintf(stderr, "probe: cannot select CUDA%d\n", c.dev);
            return false;
        }
        size_t free_b = 0, total_b = 0;
        if (!fail(cudaMemGetInfo(&free_b, &total_b), "memory information")) return false;
        const double mib = 1024.0 * 1024.0;
        std::printf("  CUDA%d: %.0f MiB free of %.0f MiB, probe holds at most %.0f MiB\n", c.dev, free_b / mib,
                    total_b / mib, kMaxPerCard / mib);
        if (free_b < kMinFreeBytes) {
            std::fprintf(stderr,
                         "probe: CUDA%d has %.0f MiB free, below the %llu MiB floor; not running (the live server "
                         "owns these cards)\n",
                         c.dev, free_b / (1024.0 * 1024.0), (unsigned long long)(kMinFreeBytes >> 20));
            return false;
        }
    }
    return true;
}

bool allocate(std::vector<Card>& cards, size_t bytes, bool flags, size_t bg_bytes = 0) {
    for (Card& c : cards) {
        if (!fail(cudaSetDevice(c.dev), "set device for allocation")) return false;
        if (!fail(cudaMalloc(&c.src, bytes), "allocate the source")) return false;
        if (!fail(cudaMalloc(&c.dst, bytes), "allocate the destination")) return false;
        if (bg_bytes) {   // the tier load's own buffers: a blob-sized copy beside the ARs
            if (!fail(cudaMalloc(&c.bg_src, bg_bytes), "allocate the tier source")) return false;
            if (!fail(cudaMalloc(&c.bg_dst, bg_bytes), "allocate the tier destination")) return false;
            if (!fail(cudaMemset(c.bg_src, 0x3c, bg_bytes), "fill the tier source")) return false;
        }
        if (!fail(cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking), "create the stream")) return false;
        if (!fail(cudaEventCreate(&c.t0), "create an event")) return false;
        if (!fail(cudaEventCreate(&c.t1), "create an event")) return false;
        if (flags) {   // the collective's own words, same shapes as `Tp2Pair::open`
            if (!fail(cudaMalloc((void**) &c.flag, 4), "allocate the flag")) return false;
            if (!fail(cudaMalloc((void**) &c.seq, 4), "allocate the sequence")) return false;
            if (!fail(cudaMemset(c.flag, 0, 4), "clear the flag")) return false;
            if (!fail(cudaMemset(c.seq, 0, 4), "clear the sequence")) return false;
            if (!fail(cudaHostAlloc((void**) &c.timeout, 4, cudaHostAllocMapped), "allocate the timeout word"))
                return false;
            *c.timeout = 0;
        }
        if (!ref_fill(c, bytes)) return false;
    }
    return true;
}

void release(std::vector<Card>& cards) {
    for (Card& c : cards) {
        if (c.s) cudaStreamDestroy(c.s);
        if (c.t0) cudaEventDestroy(c.t0);
        if (c.t1) cudaEventDestroy(c.t1);
        if (c.src) cudaFree(c.src);
        if (c.dst) cudaFree(c.dst);
        if (c.flag) cudaFree(c.flag);
        if (c.seq) cudaFree(c.seq);
        if (c.timeout) cudaFreeHost(c.timeout);
        if (c.bg_src) cudaFree(c.bg_src);
        if (c.bg_dst) cudaFree(c.bg_dst);
        c.s = nullptr;
        c.t0 = nullptr;
        c.t1 = nullptr;
        c.src = c.dst = nullptr;
        c.flag = nullptr;
        c.seq = nullptr;
        c.timeout = nullptr;
        c.bg_src = nullptr;
        c.bg_dst = nullptr;
    }
}

/// Peer access both ways for every pair of cards the pattern uses, which is what the engine does for the cards it
/// selects. A pair that cannot reach across is reported, not worked around: that is the answer the roadmap needs.
bool open_peers(const std::vector<Card>& cards, std::vector<std::string>& no_access) {
    for (const Card& a : cards)
        for (const Card& b : cards) {
            if (a.dev == b.dev) continue;
            int can = 0;
            if (cudaSetDevice(a.dev) != cudaSuccess) continue;
            if (cudaDeviceCanAccessPeer(&can, a.dev, b.dev) != cudaSuccess || !can) {
                char buf[64];
                std::snprintf(buf, sizeof buf, "CUDA%d->CUDA%d", a.dev, b.dev);
                no_access.push_back(buf);
                continue;
            }
            cudaError_t e = cudaDeviceEnablePeerAccess(b.dev, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();   // clear it: the probe keeps going and reports the pair as unusable
                char buf[64];
                std::snprintf(buf, sizeof buf, "CUDA%d->CUDA%d", a.dev, b.dev);
                no_access.push_back(buf);
            }
        }
    return no_access.empty();
}

using CopyFn = bool (*)(Card&, Card&, size_t, cudaStream_t);

/// The AR round trip at one payload size, over the given pairs, with an optional background tier load on the same
/// links. Returns the per-round microseconds (the figure the decode budget is spent in: 96 AR rounds a window on a
/// 48-layer TP=2 stage pair, so the round's latency times 96 is what the collective costs a window).
///
/// Each round is one collective per rank on every pair, issued on separate non-blocking streams so the two ranks'
/// waits make independent progress (the helper's own precondition). The span is the slowest participating card's,
/// events on that card's own stream - the same rule as `run_pattern`.
struct ArResult {
    double single_min = -1, single_med = -1, single_max = -1;   // us, one round issued alone
    double queued = -1;                                        // us per round, back to back
    double gib_s = 0;                                          // the queued span's rate, for scale
};

/// Issue `rounds` AR rounds on every pair and return the slowest participating card's span, in ms.
double ar_span(std::vector<Card>& cards, const std::vector<std::pair<int, int>>& pairs, size_t bytes, int rounds,
             bool sm_push) {
    for (Card& c : cards)
        if (cudaEventRecord(c.t0, c.s) != cudaSuccess) return -1;
    for (int it = 0; it < rounds; ++it)
        for (const auto& pr : pairs) {
            if (!ar_issue(cards[pr.first], cards[pr.second], bytes, cards[pr.first].s, sm_push)) return -1;
            if (!ar_issue(cards[pr.second], cards[pr.first], bytes, cards[pr.second].s, sm_push)) return -1;
        }
    for (Card& c : cards)
        if (cudaEventRecord(c.t1, c.s) != cudaSuccess) return -1;
    double worst = 0;
    for (const auto& pr : pairs)
        for (int rnk = 0; rnk < 2; ++rnk) {
            Card& c = cards[rnk == 0 ? pr.first : pr.second];
            if (cudaSetDevice(c.dev) != cudaSuccess) return -1;
            if (cudaEventSynchronize(c.t1) != cudaSuccess) return -1;
            float ms = 0;
            if (cudaEventElapsedTime(&ms, c.t0, c.t1) != cudaSuccess) return -1;
            worst = std::max(worst, (double) ms);
        }
    return worst;
}

ArResult run_ar(std::vector<Card>& cards, const std::vector<std::pair<int, int>>& pairs, size_t bytes, int iters,
                int reps, bool tier_load, int bg_blobs, bool sm_push) {
    ArResult r;
    if (pairs.empty()) return r;
    // warm the path once (first peer copy of a buffer pays the mapping)
    for (const auto& pr : pairs) {
        if (!ar_issue(cards[pr.first], cards[pr.second], bytes, cards[pr.first].s, sm_push)) return r;
        if (!ar_issue(cards[pr.second], cards[pr.first], bytes, cards[pr.second].s, sm_push)) return r;
    }
    for (Card& c : cards)
        if (cudaSetDevice(c.dev) != cudaSuccess || cudaStreamSynchronize(c.s) != cudaSuccess) return r;

    // the background load: a second stream per pushing card, kept at a few blobs in flight - the depth the
    // engine's own ring runs at - so the AR meets a BUSY link, not a 700 MiB queue it must wait out. Pre-issuing
    // the whole background here would measure the drain, not the contention.
    std::vector<cudaStream_t> bg;
    if (tier_load) {
        for (const auto& pr : pairs)
            for (int d = 0; d < 2; ++d) {
                Card& c = cards[d == 0 ? pr.first : pr.second];
                cudaStream_t st = nullptr;
                if (!fail(cudaSetDevice(c.dev), "set device for the tier stream")) return r;
                if (!fail(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "create the tier stream")) return r;
                bg.push_back(st);
            }
    }

    const int bg_per_round = tier_load ? std::max(1, bg_blobs) : 0;   // blobs in flight per round beside the AR

    // one round at a time, `reps` of them: the fabric's floor, and the live server's interference, separately visible
    std::vector<double> one;
    one.reserve((size_t) std::max(1, reps));
    for (int i = 0; i < std::max(1, reps); ++i) {
        // keep the tier streams fed just ahead of this round, so the round runs beside live tier traffic
        for (int q = 0; q < bg_per_round; ++q)
            for (size_t k = 0; k < pairs.size(); ++k) {
                const auto& pr = pairs[k];
                if (!bg_push(cards[pr.first], cards[pr.second], kTierBlob, bg[2 * k])) return r;
                if (!bg_push(cards[pr.second], cards[pr.first], kTierBlob, bg[2 * k + 1])) return r;
            }
        const double s = ar_span(cards, pairs, bytes, 1, sm_push);
        if (s < 0) return r;
        one.push_back(s * 1000.0);
        for (cudaStream_t st : bg)   // drain, so the next round starts with the same few blobs in flight
            if (cudaStreamSynchronize(st) != cudaSuccess) return r;
    }
    std::sort(one.begin(), one.end());
    r.single_min = one.front();
    r.single_med = one[one.size() / 2];
    r.single_max = one.back();

    // and back to back, the way a chunk issues them: one collective per split linear, per layer, into a busy stream
    for (int q = 0; q < bg_per_round * iters; ++q)
        for (size_t k = 0; k < pairs.size(); ++k) {
            const auto& pr = pairs[k];
            if (!bg_push(cards[pr.first], cards[pr.second], kTierBlob, bg[2 * k])) return r;
            if (!bg_push(cards[pr.second], cards[pr.first], kTierBlob, bg[2 * k + 1])) return r;
        }
    const double worst = ar_span(cards, pairs, bytes, iters, sm_push);
    for (cudaStream_t st : bg) {
        cudaStreamSynchronize(st);
        cudaStreamDestroy(st);
    }
    if (worst <= 0) return r;
    r.queued = worst * 1000.0 / iters;
    r.gib_s = (double) bytes * 2 * pairs.size() * iters / (worst / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    return r;
}

/// Issue one pattern's copies and time them. The pattern is a set of links that run at once, one stream per card;
/// the reported figure is each link's own bytes over the span every card's events agree on (the slowest stream's,
/// which is the honest one when links contend - the contention IS the measurement).
double run_pattern(std::vector<Card>& cards, const std::vector<Link>& links, size_t bytes, int iters, CopyFn copy) {
    for (const Link& l : links)
        if (!copy(cards[l.from], cards[l.to], bytes, cards[l.from].s)) return -1.0;   // one untimed round
    for (Card& c : cards) {
        if (cudaSetDevice(c.dev) != cudaSuccess) return -1.0;
        if (cudaStreamSynchronize(c.s) != cudaSuccess) return -1.0;
    }
    for (const Link& l : links)
        if (!fail(cudaSetDevice(cards[l.from].dev), "set device")) return -1.0;
    for (Card& c : cards)
        if (cudaEventRecord(c.t0, c.s) != cudaSuccess) return -1.0;
    for (int it = 0; it < iters; ++it)
        for (const Link& l : links)
            if (!copy(cards[l.from], cards[l.to], bytes, cards[l.from].s)) return -1.0;
    for (Card& c : cards)
        if (cudaEventRecord(c.t1, c.s) != cudaSuccess) return -1.0;
    double worst = 0;
    for (size_t ci = 0; ci < cards.size(); ++ci) {
        Card& c = cards[ci];
        bool pushes = false;
        for (const Link& l : links) pushes |= (l.from == (int) ci);
        if (!pushes) continue;   // only a card that actually pushed can set the span
        if (cudaSetDevice(c.dev) != cudaSuccess) return -1.0;
        if (cudaEventSynchronize(c.t1) != cudaSuccess) return -1.0;
        float ms = 0;
        if (cudaEventElapsedTime(&ms, c.t0, c.t1) != cudaSuccess) return -1.0;
        worst = std::max(worst, (double) ms);
    }
    if (worst <= 0) return -1.0;
    return (double) bytes * iters / (worst / 1000.0) / (1024.0 * 1024.0 * 1024.0);
}

void print_links(const char* what, const std::vector<Link>& links) {
    std::printf("  %-46s", what);
    for (const Link& l : links) std::printf(" CUDA%d>CUDA%d %6.2f GiB/s", l.from, l.to, l.gib_s);
    std::printf("\n");
}

struct Pattern {
    const char* name;
    std::vector<Link> links;
};

/// Pattern names live for the whole run; the arena is reserved up front so no `c_str()` ever moves.
const char* pattern_name(std::vector<std::string>& arena, const std::string& s) {
    arena.push_back(s);
    return arena.back().c_str();
}

std::vector<Pattern> patterns_for(int n, const char* mode, std::vector<std::string>& arena) {
    std::vector<Pattern> out;
    const bool all = !mode || !*mode || std::strcmp(mode, "all") == 0;
    if (all || !std::strcmp(mode, "pairs")) {
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                if (a != b) {
                    char buf[64];
                    std::snprintf(buf, sizeof buf, "pair %d>%d alone", a, b);
                    out.push_back({pattern_name(arena, buf), {{a, b}}});
                }
    }
    if ((all || !std::strcmp(mode, "ring")) && n >= 4)
        out.push_back({pattern_name(arena, "ring 0>1>2>3>0, four links at once"), {{0, 1}, {1, 2}, {2, 3}, {3, 0}}});
    if ((all || !std::strcmp(mode, "ring")) && n >= 4)
        out.push_back({pattern_name(arena, "ring 0>3>2>1>0, the other way"), {{0, 3}, {3, 2}, {2, 1}, {1, 0}}});
    if ((all || !std::strcmp(mode, "pair2")) && n >= 4)
        out.push_back({pattern_name(arena, "two pair links, both directions (TP=2/PP=2's two ARs)"),
                       {{0, 1}, {1, 0}, {2, 3}, {3, 2}}});
    if ((all || !std::strcmp(mode, "pairbidir")) && n >= 2)
        out.push_back({pattern_name(arena, "one pair both directions (staggered stages, one AR pair)"),
                       {{0, 1}, {1, 0}}});
    if ((all || !std::strcmp(mode, "threelink")) && n >= 4)
        out.push_back({pattern_name(arena, "three links at once (0>1, 1>0, 2>3)"), {{0, 1}, {1, 0}, {2, 3}}});
    if ((all || !std::strcmp(mode, "all2all")) && n >= 4) {
        std::vector<Link> l;
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                if (a != b) l.push_back({a, b});
        out.push_back({pattern_name(arena, "all-to-all, every card pushes to every other"), l});
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const char* mode = nullptr;
    const char* copy_kind = "engine";
    size_t bytes = 0;   // 0: the default sweep
    int iters = 20;
    int reps = 51;   // single-shot AR rounds: an odd count so the median is one of them
    int bg_blobs = 4;  // --bg-blobs: expert blobs in flight per round in the `artier` mode
    bool ar_sm = false;  // --ar-sm: push the partial with SMs instead of the copy engine
    int ndev = kMaxDevices;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "probe: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--mode") mode = next("--mode");
        else if (a == "--copy") copy_kind = next("--copy");
        else if (a == "--iters") iters = std::atoi(next("--iters"));
        else if (a == "--reps") reps = std::atoi(next("--reps"));
        else if (a == "--bg-blobs") bg_blobs = std::atoi(next("--bg-blobs"));
        else if (a == "--ar-sm") ar_sm = true;
        else if (a == "--devices") ndev = std::atoi(next("--devices"));
        else if (a == "--size") {
            const char* v = next("--size");
            const double n = std::atof(v);
            bytes = (size_t)(n * (strstr(v, "MiB") ? (1ull << 20) : 1));
        } else {
            std::fprintf(stderr, "probe: unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (ndev < 2 || ndev > kMaxDevices) {
        std::fprintf(stderr, "probe: --devices is 2..%d\n", kMaxDevices);
        return 2;
    }
    CopyFn copy = copy_engine;
    if (!std::strcmp(copy_kind, "sm")) copy = copy_sm;
    else if (std::strcmp(copy_kind, "engine")) {
        std::fprintf(stderr, "probe: --copy is engine or sm\n");
        return 2;
    }

    int count = 0;
    if (!fail(cudaGetDeviceCount(&count), "device count")) return 1;
    if (count < ndev) {
        std::fprintf(stderr, "probe: %d cards asked for, %d present\n", ndev, count);
        return 1;
    }
    std::vector<Card> cards((size_t) ndev);
    for (int i = 0; i < ndev; ++i) cards[(size_t) i].dev = i;

    std::printf("link probe: %d cards, copy=%s, mode=%s\n", ndev, copy_kind, mode ? mode : "all");
    for (int i = 0; i < ndev; ++i) {
        cudaDeviceProp p;
        if (!fail(cudaGetDeviceProperties(&p, i), "device properties")) return 1;
        std::printf("  CUDA%d %s, %d SMs, %d-bit bus\n", i, p.name, p.multiProcessorCount, p.memoryBusWidth);
    }

    std::printf("free-memory floor (%llu MiB):\n", (unsigned long long)(kMinFreeBytes >> 20));
    if (!check_free(cards)) return 1;

    std::vector<std::string> no_access;
    open_peers(cards, no_access);
    if (!no_access.empty()) {
        std::printf("pairs with NO GPU-to-GPU access:");
        for (const auto& s : no_access) std::printf(" %s", s.c_str());
        std::printf("\n");
    } else {
        std::printf("every ordered pair has GPU-to-GPU access\n");
    }

    static const size_t kSweep[] = {4096, 65536, 1u << 20, 4u << 20, 16u << 20, 31457280u};
    std::vector<size_t> sizes;
    if (bytes) sizes.push_back(std::min(bytes, kMaxBufferBytes));
    else for (size_t b : kSweep) sizes.push_back(b);

    std::vector<std::string> arena;
    arena.reserve(64);   // the 12 ordered pairs plus the collective patterns, none of them ever reallocated
    const bool ar_mode = mode && (!std::strcmp(mode, "ar") || !std::strcmp(mode, "ar2") ||
                                  !std::strcmp(mode, "ar4") || !std::strcmp(mode, "artier"));
    if (ar_mode) {
        // The number the TP=2 decode budget is spent in: one collective's round trip, at the sizes a split
        // o_proj / down_proj partial actually has at decode ({4,16,64,256} KiB), for one pair (TP=2's two
        // stages' ARs, measured alone) and for both pairs at once (the concurrent case), with and without a
        // tier-fill load running behind it on the same links.
        std::vector<std::pair<int, int>> one = {{0, 1}}, both = {{0, 1}, {2, 3}};
        if (!std::strcmp(mode, "ar2")) one = both = {{0, 1}};
        if (!std::strcmp(mode, "ar4")) one = both = {{0, 1}, {2, 3}};
        const bool tier = !std::strcmp(mode, "artier");
        static const size_t kAr[] = {4096, 16384, 65536, 262144};
        std::printf("\nAR round trip: the engine's three issues per rank, per round (%d single shots, %d queued, push=%s%s)\n",
                    reps, iters, ar_sm ? "SM" : "copy engine", tier ? ", tier load in flight" : "");
        std::printf("  %-9s | %-26s | %-26s | %s\n", "payload", "one pair, single (min/med/max us)",
                    "two pairs, single (min/med/max us)", "queued us/round: 1 pair, 2 pairs (GiB/s)");
        for (size_t b : kAr) {
            if (!allocate(cards, b, true, tier ? kTierBlob : 0)) {
                std::fprintf(stderr, "probe: allocation of %zu B per buffer failed; stopping\n", b);
                release(cards);
                return 1;
            }
            ArResult a1 = run_ar(cards, one, b, iters, reps, tier, bg_blobs, ar_sm);
            ArResult a2 = run_ar(cards, both, b, iters, reps, tier, bg_blobs, ar_sm);
            char s1[64], s2[64];
            std::snprintf(s1, sizeof s1, "%.1f / %.1f / %.1f", a1.single_min, a1.single_med, a1.single_max);
            std::snprintf(s2, sizeof s2, "%.1f / %.1f / %.1f", a2.single_min, a2.single_med, a2.single_max);
            std::printf("  %-9zu | %-26s | %-26s | %.1f, %.1f (%.2f, %.2f GiB/s)%s\n", b, s1, s2, a1.queued,
                        a2.queued, a1.gib_s, a2.gib_s, tier ? "   [tier load behind]" : "");
            for (const Card& c : cards)
                if (c.timeout && *c.timeout) std::printf("    CUDA%d timed out waiting for its peer's push\n", c.dev);
            release(cards);
        }
        std::printf("\ndone; every buffer freed\n");
        return 0;
    }

    auto pats = patterns_for(ndev, mode, arena);
    if (pats.empty()) {
        std::fprintf(stderr, "probe: no pattern for mode '%s'\n", mode ? mode : "");
        return 2;
    }
    for (size_t b : sizes) {
        std::printf("\nsize %zu B (%.3f MiB), %d rounds\n", b, (double) b / (1 << 20), iters);
        if (!allocate(cards, b, false)) {
            std::fprintf(stderr, "probe: allocation of %zu B per buffer failed; stopping\n", b);
            release(cards);
            return 1;
        }
        for (auto& pat : pats) {
            double rate = run_pattern(cards, pat.links, b, iters, copy);
            for (auto& l : pat.links) l.gib_s = rate;
            print_links(pat.name, pat.links);
        }
        release(cards);   // back between zero and the server's bytes before the next point
    }
    std::printf("\ndone; every buffer freed\n");
    return 0;
}

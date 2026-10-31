// tools/tp_cadence_probe.cu - the TP rendezvous-cadence bench: what fraction of a cross-card combine hides under
// the compute beside it, at 2 ranks and at 4, measured, for the TP=4 ceiling decision.
//
// WHAT THIS IS. A standalone CUDA program, not wired into the engine and not built by CMake: it answers the last
// unmeasured coefficient of the TP=4 characterisation - can the attention-split's cross-card work HIDE under the
// MoE compute next to it, or does the max-of-4 rendezvous eat the overlap? - before the TP=4 ceiling is priced
// against tp2pp2's ~2,540 tok/s. The compute side is measured and clean (`tools/mmq_width_probe.cu`: 0.25x/0.50x,
// no width penalty) and the fabric side is measured and clean (RUN 24: the four cards share one ~7.8 GiB/s pool),
// so what remains is whether the wire work is exposed or hidden when compute runs beside it.
//
// THREE MEASUREMENTS, ONE HARNESS.
//   prefill  the TWO-CHANNEL cadence: a continuous small background MoE-combine flow on every layer position, plus
//            a bursty channel-sized reduce only on QSA positions. Graded per combine channel (row-ship,
//            partial-merge) at {2,4} ranks x {2,4} deep x {clean, jittered}. `--uniform` runs the same total bytes
//            spread over every position, which is the schedule that flatters the answer; the gap between the two
//            is the bias, and it is measured rather than assumed.
//   decode   the 96-rounds/window small-payload push/flag/wait cadence under a tier-fill background, with SM
//            pushes and copy-engine pushes in SEPARATE points (RUN 24's 100x queueing result lives only on the
//            copy engine, so mixing them in one table would measure the mixture and not either primitive). Prints
//            the window against the 19.4 ms budget and the 3-peer row-fetch straggler spread.
//   model    `--layout` and `--selftest`: the byte ledger, the schedule, the predictions, with no CUDA call.
//
// WHY THE UNIFORM SCHEDULE BIASES THE ANSWER UP. The real prefill traffic is not a reduce every layer: a small
// MoE-combine flow (~0.5-1 GB/s aggregate of partials) runs on EVERY layer position and the channel-sized reduce
// happens only at QSA positions - six of every twenty-four at this pack's spacing. Spreading the same bytes evenly
// lets each position's cover absorb its share, so hidden comes out near 1; concentrating them means those
// positions' wire time can exceed the cover beside them and the excess is exposed. `hidden_from_positions` sums
// the per-position excess, so the schedule shape is an input to the answer, not a detail.
//
// THE COMBINE CHANNELS, both graded.
//   rowship        the routed output ships whole: n_embd BF16 per ROUTING (5120 B); cross-card routings are
//                  tokens x topk x (ranks-1)/ranks. At tokens 8192, topk 10, 4 ranks: 61 440 routings = 314.6 MB
//                  per rank per layer-chunk.
//   partial-merge  each card sums the experts it holds into a tokens x n_embd FP32 partial and only the pair-set
//                  sums cross: merge-partners x tokens x n_embd x 4, i.e. ~84 MB per partner at tokens 8192.
// The hidden fraction is reported PER CHANNEL because the combine-form decision is meant to be evidence-fed, not
// settled here. Row-ship's ~314 MB per rank per layer-chunk is beyond what the measured pool carries at any
// cadence; the byte ledger prints that plainly and the supervisor decides whether it moots the channel.
//
// THE KILL LINE IS REPORTED, NOT JUDGED. The 70% threshold was derived against the uniform schedule and the
// corrected ledger moves it, so this file prints the hidden fraction, the exposed milliseconds and the byte-derived
// break-even token count, and labels the verdict column a marker. No threshold arithmetic is in the exit status.
//
// THE COLLECTIVE. The engine's own protocol (`Tp2Pair::all_reduce`, src/kernels/cuda/tp2_collective.cu) widened
// from two ranks to R: each rank pushes its partial into every peer's staging slot for it, raises its word on every
// peer once the pushes landed, then waits for every peer's word and adds what arrived, own partial first and the
// peers in ascending rank order. Copying the protocol rather than linking the helper is deliberate - the probe must
// issue the same three issues per rank the engine issues, with the engine's spin budget and its acquire/release
// flag access - but it must not need the engine or a pair handed to it by generate.cpp. If that file's protocol
// changes this copy must follow it; the numbers are evidence only while the two agree, which is why the kernel
// names and the spin budget are the engine's.
//
// THE COVER KERNEL IS A STAND-IN, NOT AN MoE. It strides a local buffer reading, chaining FMAs and writing back, so
// it takes SM issue slots AND memory-path bandwidth the way a routed-expert product does; a `clock64()` spin would
// take the first and not the second and would overstate the hiding. Its cost is calibrated per point to hit
// `--cover-ms`. It is not the pack's NVFP4 product, which `tools/mmq_width_probe.cu` measures.
//
// THE RENDEZVOUS. Every rank's wait needs every peer's push for the SAME position, so a position's cadence is the
// slowest of the R ranks - the max-of-4 the ceiling turns on, and in decode mode the 3-peer straggler term the
// 19.4 ms window budget has to hold. `--jitter-card J --jitter-ms X` makes card J's cover longer by X ms and the
// per-card cadence line shows that delay arriving on the cards never touched; `--jitter-blobs N` / `--tier-blobs N`
// put N expert-blob peer copies behind a card instead, the tier-fill shape RUN 24 measured.
//
// DEPTH is in-flight chunks. Staging sets are allocated per CHUNK, never per slot modulo depth, so a rank that runs
// ahead cannot overwrite a peer's slot the peer is still adding; reusing them by depth would need a credit
// protocol, which is not what this probe is for.
//
// PARITY, because a mis-wired peer index is worse than a slow schedule. --parity seeds every rank's partial from a
// hash of (rank, set, element) - one function, host and device, so no host copy of the inputs is kept - and checks
// each card's reduced destination against the host reference `own + sum of the peers in ascending rank order`, and
// that the input partial was not overwritten. The add order is fixed by rank, never by arrival, so the reference is
// exact in the arithmetic and a swapped peer pointer misses by O(1) rather than by rounding; the 1e-3 bound only
// leaves room for the host compiler contracting the adds.
//
// BUILD (deliberately not a CMake target: this is a probe, not a shipped artifact; it links no engine code)
//   cd /home/gary/opt/basalt-tpstep0
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 -o /tmp/tp_cadence_probe tools/tp_cadence_probe.cu
//   ... add -Xcompiler -Wall,-Wextra to check this file compiles without a warning of its own
//
// RUN - the model needs no card
//   /tmp/tp_cadence_probe --selftest
//   /tmp/tp_cadence_probe --layout --channel row-ship
//   /tmp/tp_cadence_probe --layout --channel partial-merge --ranks 2
// RUN - the GPU points, cards free and the server on :8000 DOWN
//   /tmp/tp_cadence_probe --sweep --parity --allow-big                                  # the decision run
//   /tmp/tp_cadence_probe --mode prefill --channel partial-merge --ranks 4 --depth 4 --parity --allow-big
//   /tmp/tp_cadence_probe --mode prefill --channel row-ship --ranks 4 --depth 4 --parity --allow-big
//   /tmp/tp_cadence_probe --mode prefill --uniform --ranks 4 --depth 4 --allow-big      # the bias reference
//   /tmp/tp_cadence_probe --mode decode --ranks 4 --push sm --tier-blobs 4 --parity
//   /tmp/tp_cadence_probe --mode decode --ranks 4 --push copy-engine --tier-blobs 4     # separate point
//   /tmp/tp_cadence_probe --mode prefill --ranks 4 --depth 4 --jitter-card 3 --jitter-ms 20 --allow-big
//
// BUFFER DISCIPLINE. The production server is LIVE on these cards outside the window. The probe reads
// `cudaMemGetInfo` on every card immediately before allocating and refuses the point unless each has its own
// footprint plus a 256 MiB margin free; it refuses any point whose per-card footprint exceeds `--budget-mib`
// (default 64) unless `--allow-big` is given; it frees everything between points and touches no stream but its own.
// It touches no file and never enters the engine's allocator or the peer-exclusion path.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kMaxRanks = 4;
constexpr int kMaxChunks = 8;
constexpr int kMaxDepth = 4;
constexpr int kMaxUnits = 256;            // per-unit events, for the cadence and the straggler spread
constexpr int64_t kNEmbd = 2560;          // H: the residual pitch and the routed-output width (`ModelGeometry`)
constexpr int kExperts = 512;             // the pack's expert count, the bound on --topk
constexpr size_t kMinFreeMargin = 256ull << 20;
constexpr uint32_t kSeedPartial = 0x7a17u;

/// One expert blob of this pack: gate/up 2*640 rows x 1440 B + down 2560 rows x 360 B + the 16 B scale tail
/// (`NativeFmt`) - the unit a tier fill moves, the same figure `tools/tp_ring_probe.cu` pushes.
constexpr size_t kTierBlob = 2u * 640u * 1440u + 2560u * 360u + 16u;

/// The fabric rates RUN 24 measured on this box at its 180 W limit, per link.
struct Rate { const char* what; int links; double gib_s; };
const Rate kRates[] = {
    {"one pair alone", 1, 6.40},
    {"one pair both directions", 2, 3.96},
    {"two pair links both ways", 4, 2.00},
    {"4-deep ring", 4, 1.93},
    {"all-to-all, 12 directed pushes", 12, 0.64},
};

enum class Channel { kRowShip, kPartialMerge };
enum class Mode { kPrefill, kDecode };

// ================================ the pure model ================================
// Pure arithmetic, printed by `--layout` and asserted by `--selftest`, so the shape of the answer is filed before a
// card is touched and a disagreement between model and measurement is a statement about the box.

/// Bytes one rank's FP32 partial occupies: the engine's own `tp2_all_reduce_bytes(rows, stride)`.
long long partial_bytes(int tokens) { return (long long) tokens * kNEmbd * 4; }

/// BF16 bytes of one routed output shipped whole: n_embd values x 2 B.
long long routing_bytes() { return kNEmbd * 2; }

/// Cross-card routings per layer-chunk: each token sends `topk` routings and only those landing on a peer's
/// experts cross, which is (ranks-1)/ranks of them on an even split.
long long cross_routings(int tokens, int topk, int ranks) {
    if (ranks <= 1) return 0;
    return (long long) tokens * topk * (ranks - 1) / ranks;
}

/// What one rank moves on the burst of one layer-chunk, per channel.
long long burst_rank_bytes(Channel ch, int ranks, int tokens, int topk, int merge_partners) {
    if (ch == Channel::kRowShip) return cross_routings(tokens, topk, ranks) * routing_bytes();
    return (long long) std::max(1, merge_partners) * partial_bytes(tokens);
}

/// Bytes the whole fabric carries for one burst: R ranks each moving their own burst volume.
long long burst_pool_bytes(Channel ch, int ranks, int tokens, int topk, int merge_partners) {
    return (long long) ranks * burst_rank_bytes(ch, ranks, tokens, topk, merge_partners);
}

/// The continuous background flow per rank per position, from an aggregate rate and the position's duration. The
/// aggregate is what the fabric carries, so one rank's share of it is the aggregate divided by the ranks moving.
long long bg_rank_bytes(double aggregate_gib_s, double position_ms, int ranks) {
    if (aggregate_gib_s <= 0 || position_ms <= 0 || ranks <= 0) return 0;
    const double total = aggregate_gib_s * position_ms / 1000.0 * 1024.0 * 1024.0 * 1024.0;
    return (long long) (total / ranks);
}

double ms_at_rate(long long bytes, double gib_s) {
    if (gib_s <= 0) return 0;
    return (double) bytes / (1024.0 * 1024.0 * 1024.0) / gib_s * 1000.0;
}

/// Positions per layer-chunk that carry the burst, and which ones.
int burst_positions(int cycle, int qsa_every) {
    if (cycle <= 0 || qsa_every <= 0) return 0;
    return cycle / qsa_every;
}
bool is_burst_position(int pos, int qsa_every) { return qsa_every > 0 && pos % qsa_every == 0; }

/// The hidden fraction of a schedule: the share of its wire time the cover beside it absorbed. A position whose
/// wire exceeds its cover exposes the excess; one whose wire fits exposes nothing. Summing the excess over
/// positions and dividing it into the whole wire time gives the fraction that did NOT lengthen the window.
double hidden_from_positions(int cycle, int qsa_every, double cover_position_ms, double burst_ms, double bg_ms) {
    if (cycle <= 0) return 1.0;
    double total = 0, exposed = 0;
    for (int p = 0; p < cycle; ++p) {
        const double wire = bg_ms + (is_burst_position(p, qsa_every) ? burst_ms : 0.0);
        total += wire;
        exposed += std::max(0.0, wire - cover_position_ms);
    }
    if (total <= 0) return 1.0;
    return std::max(0.0, std::min(1.0, 1.0 - exposed / total));
}

/// The same total bytes spread evenly over the cycle: the bias reference. Every position carries an equal share,
/// so no single position can exceed its own cover until the whole average does.
double hidden_uniform(int cycle, double cover_position_ms, double burst_total_ms, double bg_ms) {
    if (cycle <= 0) return 1.0;
    const double per_pos = burst_total_ms / cycle + bg_ms;
    const double total = per_pos * cycle;
    if (total <= 0) return 1.0;
    double exposed = 0;
    for (int p = 0; p < cycle; ++p) exposed += std::max(0.0, per_pos - cover_position_ms);
    return std::max(0.0, std::min(1.0, 1.0 - exposed / total));
}

/// The token count at which one burst's time equals the cover beside a burst position: the depth of attention the
/// split can pay for before the burst stops hiding. Reported, not judged.
long long breakeven_tokens(Channel ch, int ranks, double cover_position_ms, int topk, double gib_s,
                           int merge_partners) {
    if (gib_s <= 0 || ranks <= 1 || cover_position_ms <= 0) return 0;
    const double ns_per_token_ms = ch == Channel::kRowShip
                                       ? (double) routing_bytes() * topk * (ranks - 1) / ranks / gib_s /
                                             (1024.0 * 1024.0 * 1024.0) * 1000.0
                                       : (double) (kNEmbd * 4) * std::max(1, merge_partners) / gib_s /
                                             (1024.0 * 1024.0 * 1024.0) * 1000.0;
    if (ns_per_token_ms <= 0) return 0;
    return (long long) (cover_position_ms / ns_per_token_ms);
}

/// Per-card footprint: one set per chunk of (own + one staging slot per peer + the destination), sized to the
/// widest round the point issues, plus the cover buffer and the tier-fill pair.
long long footprint_bytes(int ranks, int chunks, long long round_bytes, long long cover_bytes) {
    return (long long) chunks * (ranks + 2) * round_bytes + cover_bytes + 2 * (long long) kTierBlob + 4096;
}

/// A partial element's value: one function, host and device, so the parity reference needs no host copy.
__host__ __device__ __forceinline__ float partial_value(int rank, int set, long long i) {
    unsigned h = (unsigned) (kSeedPartial ^ ((unsigned) rank * 2654435761u) ^ ((unsigned) set * 40503u) ^
                             ((unsigned) (i * 2246822519ull) & 0xffffffffu));
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return ((float) (h & 0xffffffu) / 8388608.0f) - 1.0f;
}

/// The per-link rate the pattern gets at R ranks, from RUN 24's table, unless `--rate` overrides it.
double link_rate(int ranks, double override_rate) {
    if (override_rate > 0) return override_rate;
    return ranks >= 4 ? kRates[4].gib_s : kRates[1].gib_s;
}

// ================================ the device side ================================

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

/// The MoE-cover stand-in: read a local float4, chain `reps` FMAs per lane, write back. Holds SM issue slots for
/// the chaining and takes memory bandwidth for the striding - the contention that decides whether wire hides.
__global__ void cover_kernel(float4* __restrict__ buf, long long n4, int reps, float scale) {
    const long long stride = (long long) gridDim.x * blockDim.x;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = buf[i];
        for (int r = 0; r < reps; ++r) {
            v.x = fmaf(v.x, scale, 1.0e-6f);
            v.y = fmaf(v.y, scale, 1.0e-6f);
            v.z = fmaf(v.z, scale, 1.0e-6f);
            v.w = fmaf(v.w, scale, 1.0e-6f);
        }
        buf[i] = v;
    }
}

__global__ void sm_copy(float4* __restrict__ dst, const float4* __restrict__ src, long long n4) {
    const long long stride = (long long) gridDim.x * blockDim.x;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) dst[i] = src[i];
}

__global__ void seed_kernel(float* __restrict__ dst, long long n, int rank, int set) {
    const long long stride = (long long) gridDim.x * blockDim.x;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        dst[i] = partial_value(rank, set, i);
}

struct PeerWords { int32_t* pushed[kMaxRanks]; };

/// After my pushes landed (stream order guarantees it), publish my sequence on every peer. One thread.
__global__ void push_flag(int32_t* my_seq, PeerWords words, int ranks, int me) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int32_t v = my_seq[0] + 1;
    my_seq[0] = v;
    __threadfence_system();
    for (int r = 0; r < ranks; ++r)
        if (r != me) st_flag(words.pushed[r], v);
}

/// `dst[i] = own[i] + sum of the peers' partials`, peers in ascending rank order, once every peer's push for THIS
/// position arrived. Every block waits on the same words, which is why this cannot deadlock with a peer: the
/// peer's `push_flag` runs on the PEER's SMs, so a block here holding this card's SMs never blocks what it waits
/// for. The order is by rank and never by arrival, so the result cannot depend on timing - which is what makes the
/// host reference exact rather than approximate.
__global__ void add_wait(float* __restrict__ dst, const float* __restrict__ own,
                         const float* const* __restrict__ staging, const int32_t* pushed, const int32_t* my_seq,
                         volatile int32_t* timeout, int ranks, int me, long long n) {
    __shared__ int32_t ready;
    if (threadIdx.x == 0) {
        const int32_t want = my_seq[0];
        const long long t0 = clock64();
        int got_all = 0;
        while (true) {
            got_all = 1;
            for (int r = 0; r < ranks; ++r)
                if (r != me && ld_flag(&pushed[r]) < want) { got_all = 0; break; }
            if (got_all) break;
            if (clock64() - t0 > kTp2SpinBudget) { *timeout = 1; break; }   // mapped host: visible without a copy
        }
        ready = got_all;
    }
    __syncthreads();
    if (!ready) return;
    const long long stride_grid = (long long) gridDim.x * blockDim.x;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride_grid) {
        float acc = own[i];
        for (int r = 0; r < ranks; ++r)
            if (r != me) acc = acc + staging[r][i];
        dst[i] = acc;
    }
}

bool fail(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "cadence probe: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// ================================ options, rank state ================================

struct Options {
    bool sweep = false, parity = false, allow_big = false, layout_only = false, selftest = false, verbose = false;
    bool uniform = false;
    Mode mode = Mode::kPrefill;
    Channel channel = Channel::kPartialMerge;
    // chunks must exceed depth so the steady state dominates the ramp; the default is the widest the event
    // bookkeeping allows, and a bare invocation has to be a runnable point rather than a usage error
    int ranks = 4, depth = 4, tokens = 2048, chunks = 8, reps = 25;
    int topk = 10, cycle = 24, qsa_every = 4, merge_partners = 1;
    int rounds = 96;
    int jitter_card = -1, jitter_blobs = 0, tier_blobs = 0;
    double cover_ms = 60.0, jitter_ms = 0.0, bg_gib_s = 0.75;
    long long cover_mib = 32, decode_payload_kib = 4;
    size_t budget = 64ull << 20;
    bool sm_push = true;
    double rate = -1.0;
};

struct Result {
    bool ran = false;
    int ranks = 0, depth = 0, tokens = 0;
    double cover_ms = 0, jitter_ms = 0, window_cover = 0, window_reduce = 0, window_both = 0;
    double hidden = 0, hidden_model = 0, cadence_max = 0, cadence_min = 0, inflation = 0, exposed_ms = 0;
    double cover_per_unit_ms = 0;   // the calibrated per-position cover, reported beside the whole-window figures
    // How much worse than ADDITIVE the pair is: window_both - (window_cover + window_reduce). Zero means the wire
    // ran entirely in the cover's shadow. Negative means free overlap beyond additivity. Positive is the two
    // fighting for the same SMs, which a spin-wait reduce and an SM-resident cover inevitably do.
    double coset_ms = 0;
    bool parity_pass = false;
    Channel channel = Channel::kPartialMerge;
    bool uniform = false, sm_push = true;
    Mode mode = Mode::kPrefill;
    int tier_blobs = 0;
    double decode_window = 0, decode_straggler = 0;
};

struct Rank {
    int dev = -1, rank = 0;
    int staging_ranks = 0;                // the width the staging tables were allocated at; release must match it
    std::vector<float*> own, dst;         // [chunks]
    std::vector<float**> staging;         // [chunks] device arrays of [ranks] device pointers
    std::vector<std::vector<float*>> staging_h;   // host mirror, so the push needs no device read per round
    int32_t* pushed = nullptr;
    int32_t* my_seq = nullptr;
    volatile int32_t* timeout = nullptr;
    float4* cover = nullptr;
    float* bg_src = nullptr;
    float* bg_dst = nullptr;
    cudaStream_t s_cover = nullptr, s_comm = nullptr, s_bg = nullptr;
    cudaEvent_t w0 = nullptr, w1 = nullptr;
    cudaEvent_t unit[kMaxUnits];          // per-unit completion, for cadence and the straggler spread
    cudaEvent_t unit_cover[kMaxUnits];    // this unit's COVER completion, what the wire waits on
    cudaEvent_t chunk_done[kMaxChunks];   // per-chunk completion, what bounds the in-flight chunk count
};

void release_rank(Rank& r) {
    for (float* p : r.own) if (p) cudaFree(p);
    for (float* p : r.dst) if (p) cudaFree(p);
    // The peer slots are freed from the HOST mirror. `r.staging[c]` is the device-side pointer table that
    // add_wait indexes per element; dereferencing it here would be a host read of device memory, which faults.
    for (size_t si = 0; si < r.staging_h.size(); ++si)
        for (size_t pi = 0; pi < r.staging_h[si].size(); ++pi)
            if (r.staging_h[si][pi]) cudaFree(r.staging_h[si][pi]);
    for (float** set : r.staging) if (set) cudaFree(set);
    if (r.pushed) cudaFree(r.pushed);
    if (r.my_seq) cudaFree(r.my_seq);
    if (r.timeout) cudaFreeHost((void*) r.timeout);
    if (r.cover) cudaFree(r.cover);
    if (r.bg_src) cudaFree(r.bg_src);
    if (r.bg_dst) cudaFree(r.bg_dst);
    if (r.s_cover) cudaStreamDestroy(r.s_cover);
    if (r.s_comm) cudaStreamDestroy(r.s_comm);
    if (r.s_bg) cudaStreamDestroy(r.s_bg);
    if (r.w0) cudaEventDestroy(r.w0);
    if (r.w1) cudaEventDestroy(r.w1);
    for (cudaEvent_t& e : r.unit) if (e) cudaEventDestroy(e);
    for (cudaEvent_t& e : r.unit_cover) if (e) cudaEventDestroy(e);
    for (cudaEvent_t& e : r.chunk_done) if (e) cudaEventDestroy(e);
    r = Rank{};
}

std::vector<Rank> make_ranks(int ranks) {
    std::vector<Rank> out((size_t) ranks);
    for (int i = 0; i < ranks; ++i) {
        Rank& r = out[(size_t) i];
        r.dev = i;
        r.rank = i;
        if (cudaSetDevice(i) != cudaSuccess) continue;
        cudaStreamCreateWithFlags(&r.s_cover, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&r.s_comm, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&r.s_bg, cudaStreamNonBlocking);
        cudaEventCreate(&r.w0);
        cudaEventCreate(&r.w1);
        for (cudaEvent_t& e : r.unit) cudaEventCreate(&e);
        for (cudaEvent_t& e : r.unit_cover) cudaEventCreate(&e);
        for (cudaEvent_t& e : r.chunk_done) cudaEventCreate(&e);
    }
    return out;
}

/// Peer access for every ordered pair the point uses, as the engine opens its cards. A pair that cannot reach
/// across is reported rather than worked around: that is an answer the roadmap needs.
bool open_peers(int ranks, std::vector<std::string>& no_access) {
    for (int a = 0; a < ranks; ++a)
        for (int b = 0; b < ranks; ++b) {
            if (a == b) continue;
            int can = 0;
            if (cudaSetDevice(a) != cudaSuccess) continue;
            if (cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess || !can) {
                char buf[64];
                std::snprintf(buf, sizeof buf, "CUDA%d->CUDA%d", a, b);
                no_access.push_back(buf);
                continue;
            }
            const cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();   // clear it and keep going: an unusable pair is the finding, not a hang
                char buf[64];
                std::snprintf(buf, sizeof buf, "CUDA%d->CUDA%d", a, b);
                no_access.push_back(buf);
            }
        }
    return no_access.empty();
}

/// The bytes one round moves on one rank. The sets are sized to the widest round the point issues so a burst and
/// a background round can share them.
long long round_bytes(const Options& o) {
    if (o.mode == Mode::kDecode) return (long long) o.decode_payload_kib << 10;
    const long long burst = o.uniform
                                ? burst_rank_bytes(o.channel, o.ranks, o.tokens, o.topk, o.merge_partners) /
                                      std::max(1, o.cycle)
                                : burst_rank_bytes(o.channel, o.ranks, o.tokens, o.topk, o.merge_partners);
    const long long bg = bg_rank_bytes(o.bg_gib_s, o.cover_ms / std::max(1, o.cycle), o.ranks);
    return std::max<long long>(std::max(burst, bg), 16);   // float4-wide at minimum
}

/// The schedule, as issued: how many units the window has, and which of them carry the burst.
int window_units(const Options& o) {
    return o.mode == Mode::kDecode ? std::min(o.rounds, kMaxUnits) : o.chunks * o.cycle;
}
bool unit_is_burst(const Options& o, int u) {
    if (o.mode == Mode::kDecode) return true;
    return o.uniform || is_burst_position(u % std::max(1, o.cycle), o.qsa_every);
}
/// Bytes this rank moves in one unit: the burst at a burst position, the background flow elsewhere.
long long unit_bytes(const Options& o, int u) {
    if (o.mode == Mode::kDecode) return (long long) o.decode_payload_kib << 10;
    const long long bg = bg_rank_bytes(o.bg_gib_s, o.cover_ms / std::max(1, o.cycle), o.ranks);
    if (!unit_is_burst(o, u)) return bg;
    return o.uniform ? burst_rank_bytes(o.channel, o.ranks, o.tokens, o.topk, o.merge_partners) /
                           std::max(1, o.cycle) + bg
                     : burst_rank_bytes(o.channel, o.ranks, o.tokens, o.topk, o.merge_partners);
}

bool allocate(std::vector<Rank>& ranks, const Options& o) {
    const long long rb = (round_bytes(o) + 15) & ~15ll;
    const long long elems = rb / 4;
    const long long cb = o.cover_mib << 20;
    for (Rank& r : ranks) {
        if (!fail(cudaSetDevice(r.dev), "set device for allocation")) return false;
        r.staging_ranks = o.ranks;   // release_rank frees this width; kMaxRanks would read past the allocation
        r.own.assign((size_t) o.chunks, nullptr);
        r.dst.assign((size_t) o.chunks, nullptr);
        r.staging.assign((size_t) o.chunks, nullptr);
        r.staging_h.assign((size_t) o.chunks, std::vector<float*>((size_t) o.ranks, nullptr));
        for (int c = 0; c < o.chunks; ++c) {
            for (int p = 0; p < o.ranks; ++p)
                if (!fail(cudaMalloc((void**) &r.staging_h[(size_t) c][(size_t) p], (size_t) elems * 4),
                          "allocate a staging slot"))
                    return false;
            // the peer-pointer table must live in DEVICE memory: add_wait indexes it per element
            if (!fail(cudaMalloc((void**) &r.staging[(size_t) c], (size_t) o.ranks * sizeof(float*)),
                      "allocate the staging table") ||
                !fail(cudaMemcpy(r.staging[(size_t) c], r.staging_h[(size_t) c].data(),
                                 (size_t) o.ranks * sizeof(float*), cudaMemcpyHostToDevice),
                      "upload the staging table") ||
                !fail(cudaMalloc((void**) &r.own[(size_t) c], (size_t) elems * 4), "allocate the own partial") ||
                !fail(cudaMalloc((void**) &r.dst[(size_t) c], (size_t) elems * 4), "allocate the destination"))
                return false;
            seed_kernel<<<128, kTp2Threads, 0, r.s_comm>>>(r.own[(size_t) c], elems, r.rank, c);
            if (!fail(cudaGetLastError(), "seed the partial")) return false;
        }
        if (!fail(cudaMalloc((void**) &r.pushed, (size_t) o.ranks * 4), "allocate the words") ||
            !fail(cudaMalloc((void**) &r.my_seq, 4), "allocate the sequence") ||
            !fail(cudaMemset(r.pushed, 0, (size_t) o.ranks * 4), "clear the words") ||
            !fail(cudaMemset(r.my_seq, 0, 4), "clear the sequence") ||
            !fail(cudaHostAlloc((void**) &r.timeout, 4, cudaHostAllocMapped), "allocate the timeout word") ||
            !fail(cudaMalloc((void**) &r.cover, (size_t) cb), "allocate the cover buffer") ||
            !fail(cudaMemset(r.cover, 0, (size_t) cb), "clear the cover buffer") ||
            !fail(cudaMalloc((void**) &r.bg_src, kTierBlob), "allocate the tier source") ||
            !fail(cudaMalloc((void**) &r.bg_dst, kTierBlob), "allocate the tier destination") ||
            !fail(cudaMemset(r.bg_src, 0x3c, kTierBlob), "fill the tier source"))
            return false;
        *r.timeout = 0;
    }
    for (Rank& r : ranks)
        if (!fail(cudaSetDevice(r.dev), "set device") || !fail(cudaDeviceSynchronize(), "settle the allocation"))
            return false;
    return true;
}

bool card_has_room(const Rank& r, long long need) {
    if (!fail(cudaSetDevice(r.dev), "set device")) return false;
    size_t free_b = 0, total = 0;
    if (!fail(cudaMemGetInfo(&free_b, &total), "mem get info")) return false;
    if (free_b < (size_t) need + kMinFreeMargin) {
        std::printf("  REFUSED: CUDA%d has %.0f MiB free; this point needs %.0f MiB plus the %.0f MiB margin\n",
                    r.dev, free_b / 1048576.0, (double) need / 1048576.0, kMinFreeMargin / 1048576.0);
        return false;
    }
    return true;
}

/// One rank's issue of one round's wire: R-1 pushes, the flag, the wait-and-add. Shared by prefill and decode so
/// both use the engine's three issues and neither can drift from the other. `bytes` is this unit's own volume; the
/// background rounds are small and the bursts are not, and the staging slots are sized for the widest.
bool issue_round(std::vector<Rank>& ranks, const Options& o, Rank& r, int chunk, long long bytes) {
    const long long n4 = (bytes / 16) > 0 ? bytes / 16 : 1;
    const long long n = n4 * 4;
    const int blocks_copy = (int) std::min<long long>(512, (n4 + 255) / 256);
    const int blocks_add = (int) std::min<long long>(kTp2Blocks, (n + kTp2Threads - 1) / kTp2Threads);
    if (!fail(cudaSetDevice(r.dev), "set device for a round")) return false;
    for (int p = 0; p < o.ranks; ++p) {
        if (p == r.rank) continue;
        Rank& peer = ranks[(size_t) p];
        float* into = peer.staging_h[(size_t) chunk][(size_t) r.rank];   // the host mirror: no device read here
        if (o.sm_push) {
            sm_copy<<<blocks_copy, kTp2Threads, 0, r.s_comm>>>((float4*) into, (const float4*) r.own[(size_t) chunk],
                                                              n4);
            if (!fail(cudaGetLastError(), "the SM push")) return false;
        } else if (!fail(cudaMemcpyPeerAsync(into, peer.dev, r.own[(size_t) chunk], r.dev, n4 * 16, r.s_comm),
                          "the AR's push")) {
            return false;
        }
    }
    PeerWords w;
    for (int p = 0; p < o.ranks; ++p) w.pushed[p] = ranks[(size_t) p].pushed + r.rank;
    push_flag<<<1, 32, 0, r.s_comm>>>(r.my_seq, w, o.ranks, r.rank);
    if (!fail(cudaGetLastError(), "the flag")) return false;
    add_wait<<<blocks_add, kTp2Threads, 0, r.s_comm>>>(r.dst[(size_t) chunk], r.own[(size_t) chunk],
                                                       r.staging[(size_t) chunk], r.pushed, r.my_seq, r.timeout,
                                                       o.ranks, r.rank, n);
    return fail(cudaGetLastError(), "the wait-and-add");
}

struct Kind { bool cover = false, wire = false; };

/// Issue one window in one mode and return the slowest rank's span in ms. The rendezvous makes the slowest rank set
/// the pace, so the honest window is the max over ranks, not the mean.
double issue_window(std::vector<Rank>& ranks, const Options& o, const Kind& kind, int reps_cover, int reps_extra,
                    long long cover_bytes) {
    const long long cover_n4 = cover_bytes / 16;
    const int blocks_cover = (int) std::min<long long>(512, (cover_n4 + 255) / 256);
    const int units = window_units(o);

    for (Rank& r : ranks)   // start from a quiet card, so the window is this window
        if (cudaSetDevice(r.dev) != cudaSuccess || cudaStreamSynchronize(r.s_cover) != cudaSuccess ||
            cudaStreamSynchronize(r.s_comm) != cudaSuccess || cudaStreamSynchronize(r.s_bg) != cudaSuccess)
            return -1;

    // the tier fill runs beside the whole window on its own stream: the shape RUN 24 used to turn a 25 us round
    // trip into 2.4 ms, and the thing that makes the copy-engine and SM pushes diverge
    const int blobs = o.mode == Mode::kDecode ? o.tier_blobs : o.jitter_blobs;
    if (blobs > 0) {
        const int jc = o.jitter_card >= 0 ? o.jitter_card : o.ranks - 1;
        Rank& j = ranks[(size_t) jc];
        if (!fail(cudaSetDevice(j.dev), "set device for the tier fill")) return -1;
        for (int b = 0; b < blobs; ++b) {
            Rank& peer = ranks[(size_t) ((j.rank + 1 + b) % o.ranks)];
            if (!fail(cudaMemcpyPeerAsync(peer.bg_dst, peer.dev, j.bg_src, j.dev, kTierBlob, j.s_bg), "the tier fill"))
                return -1;
        }
    }

    for (Rank& r : ranks)
        if (cudaSetDevice(r.dev) != cudaSuccess ||
            cudaEventRecord(r.w0, kind.wire ? r.s_comm : r.s_cover) != cudaSuccess)
            return -1;

    for (int u = 0; u < units; ++u) {
        const int cyc = std::max(1, o.cycle);
        const int chunk = (u / cyc) % o.chunks;          // the buffer set this chunk uses
        const bool first_of_chunk = o.mode == Mode::kDecode || (u % cyc) == 0;
        const bool last_of_chunk = o.mode == Mode::kDecode || (u % cyc) == cyc - 1;
        for (Rank& r : ranks) {
            if (cudaSetDevice(r.dev) != cudaSuccess) return -1;
            // D chunks in flight: the first unit of chunk c waits for chunk c-D's wire to have completed here.
            // `chunk_done` is the per-CHUNK event - indexing the per-position `unit` array with a chunk number
            // would gate on the wrong completion and let more than D chunks be in flight.
            if (o.mode == Mode::kPrefill && first_of_chunk && kind.wire) {
                const int c_abs = u / cyc;
                if (c_abs >= o.depth &&
                    cudaStreamWaitEvent(r.s_comm, r.chunk_done[(c_abs - o.depth) % o.chunks], 0) != cudaSuccess)
                    return -1;
            }
            if (kind.cover) {
                const int reps = reps_cover + (r.rank == o.jitter_card ? reps_extra : 0);
                cover_kernel<<<blocks_cover, kTp2Threads, 0, r.s_cover>>>(r.cover, cover_n4, reps, 1.0000001f);
                if (!fail(cudaGetLastError(), "the cover")) return -1;
                if (kind.wire) {
                    // The chunk's wire waits for the chunk's cover: the order a real window has. This needs its OWN
                    // event: recording the cover on `unit[u]` and then the wire on the same `unit[u]` leaves the
                    // depth gate waiting on whichever record landed last, so the gate no longer bounds anything and
                    // every unit runs serialized behind the previous chunk's wire instead of beside it.
                    if (cudaEventRecord(r.unit_cover[u], r.s_cover) != cudaSuccess) return -1;
                    if (cudaStreamWaitEvent(r.s_comm, r.unit_cover[u], 0) != cudaSuccess) return -1;
                }
            }
            if (kind.wire) {
                const long long bytes = unit_bytes(o, u);
                if (bytes > 0 && !issue_round(ranks, o, r, chunk, bytes)) return -1;
                if (cudaEventRecord(r.unit[u], r.s_comm) != cudaSuccess) return -1;
                if (last_of_chunk && cudaEventRecord(r.chunk_done[chunk], r.s_comm) != cudaSuccess) return -1;
            } else if (cudaEventRecord(r.unit[u], r.s_cover) != cudaSuccess) {
                return -1;
            }
            if (kind.cover && !kind.wire && last_of_chunk &&
                cudaEventRecord(r.chunk_done[chunk], r.s_cover) != cudaSuccess)
                return -1;
        }
    }

    for (Rank& r : ranks) {
        if (cudaSetDevice(r.dev) != cudaSuccess) return -1;
        if (cudaEventRecord(r.w1, kind.wire ? r.s_comm : r.s_cover) != cudaSuccess) return -1;
        if (r.s_bg && cudaStreamSynchronize(r.s_bg) != cudaSuccess) return -1;
        if (cudaEventSynchronize(r.w1) != cudaSuccess) return -1;
    }
    double worst = 0;
    for (const Rank& r : ranks) {
        if (cudaSetDevice(r.dev) != cudaSuccess) return -1;
        float ms = 0;
        if (cudaEventElapsedTime(&ms, r.w0, r.w1) != cudaSuccess) return -1;
        worst = std::max(worst, (double) ms);
    }
    return worst;
}

/// The per-rank unit-to-unit cadence of the last window: what the max-of-R rendezvous costs each card, and whether
/// a delay injected on one card arrived on the others.
std::vector<double> per_card_cadence(std::vector<Rank>& ranks, const Options& o) {
    const int units = window_units(o);
    if (units < 2) return {};
    std::vector<double> out(ranks.size(), 0);
    for (size_t i = 0; i < ranks.size(); ++i) {
        Rank& r = ranks[i];
        if (cudaSetDevice(r.dev) != cudaSuccess) return {};
        double sum = 0;
        for (int u = 1; u < units; ++u) {
            if (cudaEventSynchronize(r.unit[u]) != cudaSuccess) return {};
            float ms = 0;
            if (cudaEventElapsedTime(&ms, r.unit[u - 1], r.unit[u]) != cudaSuccess) return {};
            sum += ms;
        }
        out[i] = sum / (units - 1);
    }
    return out;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

const char* channel_name(Channel c) { return c == Channel::kRowShip ? "row-ship" : "partial-merge"; }

// ================================ one point ================================

bool run_point(std::vector<Rank>& ranks, const Options& o_in, Result* out) {
    Options o = o_in;
    const long long cover_bytes = o.cover_mib << 20;
    const long long need = footprint_bytes(o.ranks, o.chunks, round_bytes(o), cover_bytes);
    std::printf("point mode=%s channel=%s ranks=%d depth=%d tokens=%d chunks=%d cycle=%d qsa=%d cover=%.0fms "
                "reps=%d push=%s schedule=%s footprint %.0f MiB/card\n",
                o.mode == Mode::kDecode ? "decode" : "prefill", channel_name(o.channel), o.ranks, o.depth, o.tokens,
                o.chunks, o.cycle, o.qsa_every, o.cover_ms, o.reps, o.sm_push ? "SM" : "copy-engine",
                o.uniform ? "uniform" : "two-channel", (double) need / 1048576.0);

    Result res;
    res.ranks = o.ranks;
    res.depth = o.depth;
    res.tokens = o.tokens;
    res.cover_ms = o.cover_ms;
    res.jitter_ms = o.jitter_ms;
    res.channel = o.channel;
    res.uniform = o.uniform;
    res.mode = o.mode;
    res.sm_push = o.sm_push;
    res.tier_blobs = o.tier_blobs;

    if ((size_t) need > o.budget && !o.allow_big) {
        std::printf("  SKIPPED: the point needs %.0f MiB on a card, the budget is %.0f MiB (pass --allow-big)\n",
                    (double) need / 1048576.0, (double) o.budget / 1048576.0);
        return false;
    }
    for (const Rank& r : ranks)
        if (!card_has_room(r, need)) return false;
    if (!allocate(ranks, o)) {
        for (Rank& r : ranks) release_rank(r);
        return false;
    }

    // Calibrate the cover: the reps that make one unit's cover hit cover_ms/cycle (prefill spreads one layer's
    // cover over the cycle) or cover_ms itself (decode pays one cover per round is not what it does, so decode
    // calibrates to the whole cover_ms and reports the per-round figure).
    const double want_per_unit = o.mode == Mode::kDecode ? o.cover_ms : o.cover_ms / std::max(1, o.cycle);
    const int units_in_window = std::max(1, window_units(o));
    auto cover_window_ms = [&](int reps) -> double {
        return issue_window(ranks, o, Kind{true, false}, reps, 0, cover_bytes);
    };
    auto cover_per_unit = [&](int reps) -> double {
        const double ms = cover_window_ms(reps);
        return ms < 0 ? -1 : ms / units_in_window;
    };
    const double t1 = cover_per_unit(1);
    if (t1 <= 0) { for (Rank& r : ranks) release_rank(r); return false; }
    int reps_cover = std::max(1, (int) std::llround(want_per_unit / t1));
    double tc = cover_per_unit(reps_cover);
    for (int pass = 0; pass < 4 && tc > 0; ++pass) {
        if (std::fabs(tc - want_per_unit) <= 0.02 * want_per_unit) break;
        const int next = std::max(1, (int) std::llround((double) reps_cover * want_per_unit / tc));
        if (next == reps_cover) break;
        reps_cover = next;
        tc = cover_per_unit(reps_cover);
    }
    const double increment = reps_cover > 1 ? std::max(1e-9, (tc - t1) / (reps_cover - 1)) : t1;
    const int reps_extra = o.jitter_ms > 0
                               ? (int) std::max<long long>(1, (long long) std::llround(o.jitter_ms / increment))
                               : 0;
    if (o.verbose)
        std::printf("  calibration: 1 rep %.4f ms/unit, %d reps %.4f ms/unit (target %.4f) over %d units, jitter"
                    " +%d reps for %.0f ms\n", t1, reps_cover, tc, want_per_unit, units_in_window, reps_extra,
                    o.jitter_ms);

    // ALL THREE windows are whole-window totals in ms. Mixing a per-unit cover with whole-window wire figures made
    // `exposed = both - cover` subtract a 1/units-sized number from a full window and report a nonsense hidden
    // fraction. The per-unit cover is reported separately, beside the figure it was calibrated to.
    std::vector<double> a, red(o.reps, -1), both(o.reps, -1);
    for (int i = 0; i < o.reps; ++i) a.push_back(cover_window_ms(reps_cover));
    for (int i = 0; i < o.reps; ++i) {
        red[i] = issue_window(ranks, o, Kind{false, true}, reps_cover, reps_extra, cover_bytes);
        both[i] = issue_window(ranks, o, Kind{true, true}, reps_cover, reps_extra, cover_bytes);
    }
    if (a.empty() || *std::min_element(a.begin(), a.end()) < 0 ||
        *std::min_element(red.begin(), red.end()) < 0 || *std::min_element(both.begin(), both.end()) < 0) {
        for (Rank& r : ranks) release_rank(r);
        return false;
    }
    res.window_cover = median_of(a);
    res.cover_per_unit_ms = res.window_cover / units_in_window;
    res.window_reduce = median_of(red);
    res.window_both = median_of(both);

    for (const Rank& r : ranks)
        if (r.timeout && *r.timeout) std::printf("  CUDA%d TIMED OUT waiting for a peer's push\n", r.dev);

    res.exposed_ms = res.window_both - res.window_cover;
    res.coset_ms = res.window_both - (res.window_cover + res.window_reduce);
    // The raw ratio is kept for the record but clamped for display: a pair that fights for SMs can expose far more
    // than the wire's own length, and a -800% "hidden fraction" reads like a broken gauge rather than a result. The
    // co-scheduling overhead above is the number that says what actually happened.
    res.hidden = res.window_reduce > 0 ? 1.0 - res.exposed_ms / res.window_reduce : 0.0;

    const double link = link_rate(o.ranks, o.rate);
    if (o.mode == Mode::kPrefill) {
        const double cover_pos = o.cover_ms / std::max(1, o.cycle);
        const long long burst_total = burst_rank_bytes(o.channel, o.ranks, o.tokens, o.topk, o.merge_partners);
        const long long bgb = bg_rank_bytes(o.bg_gib_s, cover_pos, o.ranks);
        res.hidden_model = o.uniform
                               ? hidden_uniform(o.cycle, cover_pos, ms_at_rate(burst_total, link),
                                                ms_at_rate(bgb, link))
                               : hidden_from_positions(o.cycle, o.qsa_every, cover_pos, ms_at_rate(burst_total, link),
                                                       ms_at_rate(bgb, link));
    }

    const std::vector<double> cad = per_card_cadence(ranks, o);
    if (!cad.empty()) {
        for (double v : cad) res.cadence_max = std::max(res.cadence_max, v);
        res.cadence_min = *std::min_element(cad.begin(), cad.end());
        res.inflation = res.cadence_min > 0 ? res.cadence_max / res.cadence_min : 0;
    }
    if (o.mode == Mode::kDecode) {
        res.decode_window = res.window_both;
        res.decode_straggler = res.cadence_max - res.cadence_min;
    }

    if (o.parity) {
        const long long rb = (round_bytes(o) + 15) & ~15ll;
        const long long elems = rb / 4;
        const long long sample = std::min<long long>(elems, 1u << 15);
        std::vector<float> got((size_t) sample), own((size_t) sample);
        double worst_abs = 0, worst_rel = 0;
        int bad_rank = -1;
        long long bad_i = -1;
        bool ok = true;
        for (size_t i = 0; i < ranks.size() && ok; ++i) {
            Rank& r = ranks[i];
            ok = fail(cudaSetDevice(r.dev), "set device for readback") &&
                 fail(cudaMemcpy(got.data(), r.dst[0], (size_t) sample * 4, cudaMemcpyDeviceToHost), "dst readback") &&
                 fail(cudaMemcpy(own.data(), r.own[0], (size_t) sample * 4, cudaMemcpyDeviceToHost), "own readback");
            if (!ok) break;
            for (long long e = 0; e < sample; ++e) {
                double ref = partial_value(r.rank, 0, e);
                for (int p = 0; p < o.ranks; ++p)
                    if (p != r.rank) ref += partial_value(p, 0, e);
                if (std::fabs((double) own[(size_t) e] - (double) partial_value(r.rank, 0, e)) > 1e-9) {
                    std::printf("  PARITY: rank %d's own partial was overwritten at elem %lld\n", r.rank, e);
                    ok = false;
                    break;
                }
                const double d = std::fabs(ref - (double) got[(size_t) e]);
                const double rel = d / std::max(std::fabs(ref), 1.0);
                if (d > worst_abs) { worst_abs = d; worst_rel = rel; bad_rank = r.rank; bad_i = e; }
            }
        }
        res.parity_pass = ok && worst_abs <= 1e-3 && worst_rel <= 1e-3;
        std::printf("  PARITY ranks=%d: max|d|=%.3e (rank %d, elem %lld), max rel=%.3e, own input intact -> %s "
                    "(1e-3 bound)\n", o.ranks, worst_abs, bad_rank, bad_i, worst_rel,
                    res.parity_pass ? "PASS" : "FAIL");
    }

    const double hidden_shown = std::max(-999.0, std::min(100.0, res.hidden * 100.0));
    std::printf("  RESULT %s/%s ranks=%d depth=%d%s: %d units | window cover %.2f (%.3f/unit), wire %.2f, both"
                " %.2f ms | exposed %.2f ms | HIDDEN %.1f%%%s | co-scheduling %.2f ms | model %.1f%%\n",
                o.mode == Mode::kDecode ? "decode" : "prefill", channel_name(o.channel), o.ranks, o.depth,
                o.uniform ? "/uniform" : "", units_in_window, res.window_cover, res.cover_per_unit_ms,
                res.window_reduce, res.window_both, res.exposed_ms, hidden_shown,
                res.hidden < 0 ? " (floored; the pair is anti-additive)" : "", res.coset_ms,
                res.hidden_model * 100.0);
    if (res.coset_ms > 0.01 * std::max(1.0, res.window_reduce))
        std::printf("  NOTE the pair ran %.2f ms worse than cover+wire added: the spin-wait reduce holds SMs and the"
                    " cover cannot get slots. Hiding is not available to this shape regardless of the fabric.\n",
                    res.coset_ms);
    if (!cad.empty()) {
        std::printf("  per-card cadence ms:");
        for (size_t i = 0; i < cad.size(); ++i) std::printf(" CUDA%d=%.3f", ranks[i].dev, cad[i]);
        std::printf("  | max-of-%d %.3f, spread %.3f, cadence inflation %.2fx\n", (int) ranks.size(),
                    res.cadence_max, res.cadence_max - res.cadence_min, res.inflation);
    }
    if (o.mode == Mode::kDecode)
        std::printf("  decode: window %.2f ms vs the 19.4 ms budget, %.3f ms/round over %d rounds; 3-peer straggler"
                    " spread %.3f ms/round (tier blobs %d, push %s)\n", res.decode_window,
                    o.rounds > 0 ? res.decode_window / o.rounds : 0.0, o.rounds, res.decode_straggler, o.tier_blobs,
                    o.sm_push ? "SM" : "copy-engine");

    res.ran = true;
    if (out) *out = res;
    std::fflush(stdout);
    return true;
}

// ================================ the model, printed and asserted ================================

void print_model(const Options& o) {
    const double link = link_rate(o.ranks, o.rate);
    const double cover_pos = o.cover_ms / std::max(1, o.cycle);
    const long long burst_total = burst_rank_bytes(o.channel, o.ranks, o.tokens, o.topk, o.merge_partners);
    const long long bgb = bg_rank_bytes(o.bg_gib_s, cover_pos, o.ranks);
    std::printf("tp_cadence_probe model (no CUDA call): mode %s, channel %s, ranks %d, depth %d, tokens %d,"
                " topk %d, cycle %d, qsa every %d, cover %.0f ms, bg %.2f GiB/s aggregate\n",
                o.mode == Mode::kDecode ? "decode" : "prefill", channel_name(o.channel), o.ranks, o.depth, o.tokens,
                o.topk, o.cycle, o.qsa_every, o.cover_ms, o.bg_gib_s);

    std::printf("\nbytes per rank per layer-chunk\n");
    if (o.channel == Channel::kRowShip) {
        std::printf("  cross routings            %14lld   (tokens x topk x (ranks-1)/ranks = %d x %d x %d/%d)\n",
                    cross_routings(o.tokens, o.topk, o.ranks), o.tokens, o.topk, o.ranks - 1, o.ranks);
        std::printf("  per routing               %14lld B  (n_embd %lld BF16)\n", routing_bytes(), (long long) kNEmbd);
    } else {
        std::printf("  partial                   %14lld B  (tokens x n_embd x 4, x%d merge partners)\n",
                    partial_bytes(o.tokens), std::max(1, o.merge_partners));
    }
    std::printf("  BURST per rank            %14lld B  = %9.1f MB = %9.1f MiB\n", burst_total, burst_total / 1e6,
                burst_total / 1048576.0);
    std::printf("  BURST whole fabric        %14lld B  = %9.1f MB = %9.1f MiB  (%d ranks)\n",
                burst_pool_bytes(o.channel, o.ranks, o.tokens, o.topk, o.merge_partners),
                burst_pool_bytes(o.channel, o.ranks, o.tokens, o.topk, o.merge_partners) / 1e6,
                burst_pool_bytes(o.channel, o.ranks, o.tokens, o.topk, o.merge_partners) / 1048576.0, o.ranks);
    std::printf("  background per rank/pos   %14lld B  = %9.2f MB  (%.2f GiB/s aggregate over %.2f ms, /%d ranks)\n",
                bgb, bgb / 1e6, o.bg_gib_s, cover_pos, o.ranks);
    std::printf("  per-card footprint        %14lld B  = %9.1f MiB\n",
                footprint_bytes(o.ranks, o.chunks, round_bytes(o), o.cover_mib << 20),
                footprint_bytes(o.ranks, o.chunks, round_bytes(o), o.cover_mib << 20) / 1048576.0);

    std::printf("\nthe two schedules side by side - the gap between them IS the bias\n");
    const double h_two = hidden_from_positions(o.cycle, o.qsa_every, cover_pos, ms_at_rate(burst_total, link),
                                               ms_at_rate(bgb, link));
    const double h_uni = hidden_uniform(o.cycle, cover_pos, ms_at_rate(burst_total, link), ms_at_rate(bgb, link));
    std::printf("  two-channel (%d bursts in %d positions): hidden %.1f%%\n",
                burst_positions(o.cycle, o.qsa_every), o.cycle, h_two * 100.0);
    std::printf("  uniform   (same bytes, every position): hidden %.1f%%\n", h_uni * 100.0);
    std::printf("  the uniform schedule reads %+.1f points higher; the measured answer is the two-channel one\n",
                (h_uni - h_two) * 100.0);

    std::printf("\nagainst the measured fabric rates (RUN 24):\n");
    std::printf("  %-32s %7s %10s %10s %12s %12s\n", "pattern", "GiB/s", "burst ms", "bg ms", "pred hidden",
                "kill tokens");
    for (const Rate& rt : kRates)
        std::printf("  %-32s %7.2f %10.2f %10.3f %11.1f%% %12lld\n", rt.what, rt.gib_s,
                    ms_at_rate(burst_total, rt.gib_s), ms_at_rate(bgb, rt.gib_s),
                    hidden_from_positions(o.cycle, o.qsa_every, cover_pos, ms_at_rate(burst_total, rt.gib_s),
                                          ms_at_rate(bgb, rt.gib_s)) *
                        100.0,
                    breakeven_tokens(o.channel, o.ranks, cover_pos, o.topk, rt.gib_s, o.merge_partners));

    std::printf("\n  hidden = 1 - (exposed wire / total wire) summed per position; a position exposes only the part\n"
                "  of its wire its own cover could not absorb. The 70%% line was derived against the uniform\n"
                "  schedule and the corrected ledger moves it, so the verdict is reported, not judged.\n");
    if (o.mode == Mode::kDecode) {
        const double per_round = ms_at_rate((long long) o.decode_payload_kib << 10, link);
        std::printf("  decode: %d rounds/window at %lld KiB payload: %.3f ms/round at %.2f GiB/s per link,\n"
                    "  %.2f ms/window against the 19.4 ms budget. The 3-peer straggler is what consumes the\n"
                    "  remaining %.2f ms, and it is the measurement, not the model, that says whether it fits.\n",
                    o.rounds, o.decode_payload_kib, per_round, link, per_round * o.rounds,
                    19.4 - per_round * o.rounds);
    }
}

/// The predicted CADENCE INFLATION from one rank being sustained `jitter_ms` slow, reported beside the hidden
/// fraction and never folded into it.
///
/// The hidden fraction measures how much of the WIRE was absorbed; the straggler measures how much longer the
/// BARRIER runs because the slowest rank sets it. Dividing one by the other is a category error - a 10 ms compute
/// delay charged against a 200 ms wire budget swamps the ratio and pins the hidden fraction at zero for any
/// jitter, which says nothing about the wire. So they are two numbers:
///
///   hidden(p)     = 1 - sum max(0, wire - cover) / sum wire            (hidden_from_positions)
///   inflation(p)  = max(cover + jitter, wire) / max(cover, wire)       (this function, averaged over positions)
///
/// A straggler is free where the wire already binds the position (the barrier was waiting for the wire anyway) and
/// costs its full delay where the cover binds. That is the max-of-4 coefficient the ceiling turns on.
double predicted_cadence_inflation(int cycle, int qsa_every, double cover_position_ms, double burst_ms, double bg_ms,
                                   double jitter_ms) {
    if (cycle <= 0) return 1.0;
    const double j = std::max(0.0, jitter_ms);
    double sum = 0;
    for (int p = 0; p < cycle; ++p) {
        const double wire = bg_ms + (is_burst_position(p, qsa_every) ? burst_ms : 0.0);
        const double span_clean = std::max(cover_position_ms, wire);
        if (span_clean <= 0) { sum += 1.0; continue; }
        sum += std::max(cover_position_ms + j, wire) / span_clean;
    }
    return sum / cycle;
}

/// The predicted grid the progress ledger files before the window: both channels x {2,4} ranks x {2,4} deep x
/// {clean, jittered}, plus the decode row. Generated rather than hand-typed, so the filed prediction and the model
/// cannot drift apart. Both depth rows of a channel/ranks pair are IDENTICAL by construction: the model says depth
/// >= 2 buys nothing, so a measured gap between them is the max-of-R rendezvous the model does not contain.
void print_prediction_grid() {
    const int cycle = 24, qsa = 4, tokens = 2048, topk = 10;
    const double cover_ms = 60.0, bg_rate = 0.75;
    const double cover_pos = cover_ms / cycle;
    std::printf("\npredicted hidden fraction, generated by the model (no CUDA call). Two-channel schedule, cover %.0f"
                " ms over a %d-position cycle with the burst every %d, bg %.2f GiB/s aggregate, tokens %d, topk %d\n",
                cover_ms, cycle, qsa, bg_rate, tokens, topk);
    std::printf("  %-14s %5s %5s %8s %10s %10s %10s %10s\n", "channel", "ranks", "deep", "rate", "hidden",
                "infl+10ms", "infl+25ms", "infl+50ms");
    for (Channel ch : {Channel::kPartialMerge, Channel::kRowShip})
        for (int ranks : {2, 4})
            for (int deep : {2, 4}) {
                const double link = link_rate(ranks, -1.0);
                const long long burst = burst_rank_bytes(ch, ranks, tokens, topk, 1);
                const long long bg = bg_rank_bytes(bg_rate, cover_pos, ranks);
                const double bm = ms_at_rate(burst, link), gm = ms_at_rate(bg, link);
                std::printf("  %-14s %5d %5d %8.2f %9.1f%% %9.2fx %9.2fx %9.2fx\n", channel_name(ch), ranks, deep,
                            link, hidden_from_positions(cycle, qsa, cover_pos, bm, gm) * 100.0,
                            predicted_cadence_inflation(cycle, qsa, cover_pos, bm, gm, 10.0),
                            predicted_cadence_inflation(cycle, qsa, cover_pos, bm, gm, 25.0),
                            predicted_cadence_inflation(cycle, qsa, cover_pos, bm, gm, 50.0));
            }
    std::printf("  `hidden` is the wire absorbed under the cover; `infl+Jms` is how much longer the barrier runs\n"
                "  when one rank is sustained J ms slow. They are separate coefficients on purpose: the straggler's\n"
                "  cost is COMPUTE time and must not be divided by the wire budget. Both are measured per card.\n");
    std::printf("  depth 2 and depth 4 are predicted IDENTICAL for every row. A measured gap between them is the\n"
                "  max-of-R rendezvous the model does not contain, and is the single most informative disagreement\n"
                "  the run can produce.\n");
    const double link4 = link_rate(4, -1.0);
    std::printf("  decode, 4 ranks: 96 rounds x 4 KiB at %.2f GiB/s = %.3f ms/round, %.2f ms/window against the\n"
                "  19.4 ms budget. Whether SM or copy-engine pushes, and the 3-peer straggler spread, are what the\n"
                "  run settles; the bandwidth floor alone says the budget is not the binding constraint.\n", link4,
                ms_at_rate(4096, link4), ms_at_rate(4096, link4) * 96);
}

/// The sweep plan, printed without a CUDA call so the point list can be reviewed before any card is touched.
void print_sweep_plan() {
    // 2 channels x 2 rank counts x 2 depths, + the uniform bias, + 2 channels x 2 jitters, + 2 pushes x 2 fills
    const int points = 2 * 2 * 2 + 1 + 2 * 2 + 2 * 2;
    std::printf("\nthe sweep plan (%d points), each run as its own point:\n", points);
    for (Channel ch : {Channel::kPartialMerge, Channel::kRowShip})
        for (int r : {2, 4})
            for (int d : {2, 4})
                std::printf("  --mode prefill --channel %-14s --ranks %d --depth %d\n", channel_name(ch), r, d);
    std::printf("  --mode prefill --uniform --ranks 4 --depth 4                     (the uniform-schedule bias)\n");
    for (Channel ch : {Channel::kPartialMerge, Channel::kRowShip})
        for (int j : {10, 25})
            std::printf("  --mode prefill --channel %-14s --ranks 4 --depth 4 --jitter-card 3 --jitter-ms %d\n",
                        channel_name(ch), j);
    for (bool sm : {true, false})
        for (int b : {0, 4})
            std::printf("  --mode decode --ranks 4 --push %-12s --tier-blobs %d\n", sm ? "sm" : "copy-engine", b);
    std::printf("\n  all of them in one go: tp_cadence_probe --sweep --parity --allow-big\n");
}

int run_selftest() {
    int bad = 0;
    auto eq = [&](const char* what, double got, double want) {
        if (std::fabs(got - want) <= 1e-9 * std::max(1.0, std::fabs(want))) return;
        std::fprintf(stderr, "selftest FAIL %s: got %.9g, want %.9g\n", what, got, want);
        ++bad;
    };
    auto yes = [&](const char* what, bool v) {
        if (v) return;
        std::fprintf(stderr, "selftest FAIL %s\n", what);
        ++bad;
    };

    // geometry: the pack's own figures
    yes("stride is n_embd", kNEmbd == 2560);
    yes("the tier blob is the pack's largest expert blob", kTierBlob == 2764816ull);
    eq("a partial is tokens x n_embd x 4", (double) partial_bytes(8192), 8192.0 * 2560 * 4);
    yes("an 8192-token partial is 80 MiB", partial_bytes(8192) == 80ll * 1024 * 1024);
    eq("a BF16 routing is n_embd x 2", (double) routing_bytes(), 5120.0);

    // the row-ship ledger, against the corrected figure
    eq("cross routings at T=8192 topk=10 ranks=4", (double) cross_routings(8192, 10, 4), 61440.0);
    eq("row-ship burst bytes at that point", (double) burst_rank_bytes(Channel::kRowShip, 4, 8192, 10, 1),
       61440.0 * 5120.0);
    yes("row-ship at the full geometry is the ~315 MB the corrected ledger says",
        burst_rank_bytes(Channel::kRowShip, 4, 8192, 10, 1) == 314572800ll);
    yes("row-ship grows with ranks", cross_routings(8192, 10, 4) > cross_routings(8192, 10, 2));
    yes("at 2 ranks half the routings cross", cross_routings(8192, 10, 2) == 8192ll * 10 / 2);
    yes("one rank moves nothing", cross_routings(8192, 10, 1) == 0);

    // the partial-merge ledger
    eq("partial-merge burst is merge-partners partials",
       (double) burst_rank_bytes(Channel::kPartialMerge, 4, 8192, 10, 1), 80.0 * 1024 * 1024);
    eq("two merge partners double it", (double) burst_rank_bytes(Channel::kPartialMerge, 4, 8192, 10, 2),
       160.0 * 1024 * 1024);
    yes("partial-merge is the ~84 MB the brief quotes",
        burst_rank_bytes(Channel::kPartialMerge, 4, 8192, 10, 1) == 83886080ll);
    yes("partial-merge ignores topk (only the sums cross)",
        burst_rank_bytes(Channel::kPartialMerge, 4, 8192, 8, 1) ==
            burst_rank_bytes(Channel::kPartialMerge, 4, 8192, 16, 1));
    eq("the fabric carries R ranks' worth", (double) burst_pool_bytes(Channel::kPartialMerge, 4, 2048, 10, 1),
       4.0 * burst_rank_bytes(Channel::kPartialMerge, 4, 2048, 10, 1));

    // the background flow: an aggregate rate split across the ranks that move
    yes("no aggregate rate means no background", bg_rank_bytes(0, 10, 4) == 0);
    yes("no ranks means no share", bg_rank_bytes(1.0, 10, 0) == 0);
    yes("a faster aggregate moves more", bg_rank_bytes(1.0, 2.5, 4) > bg_rank_bytes(0.5, 2.5, 4));
    yes("a longer position moves more", bg_rank_bytes(0.75, 5.0, 4) > bg_rank_bytes(0.75, 2.5, 4));
    yes("four ranks each carry a quarter of the aggregate, to the last truncated byte",
        std::llabs(bg_rank_bytes(1.0, 10.0, 4) * 4 - bg_rank_bytes(1.0, 10.0, 1)) <= 4);
    yes("the four shares are equal to each other", bg_rank_bytes(1.0, 10.0, 4) == bg_rank_bytes(1.0, 10.0, 4));
    yes("more ranks means a smaller share per rank", bg_rank_bytes(1.0, 10.0, 4) < bg_rank_bytes(1.0, 10.0, 2));

    // rates: GiB-based, asserted at exact binary figures
    eq("1 GiB at 1 GiB/s is 1000 ms", ms_at_rate(1024LL * 1024 * 1024, 1.0), 1000.0);
    eq("an 80 MiB partial at 5 GiB/s", ms_at_rate(80ll * 1024 * 1024, 5.0), 15.625);   // 80/1024/5*1000
    eq("no rate is no time", ms_at_rate(1024 * 1024, 0), 0.0);
    yes("more bytes take longer", ms_at_rate(2ll << 20, 3.0) > ms_at_rate(1ll << 20, 3.0));

    // the QSA pattern
    eq("6 bursts in a 24 cycle at every 4th", (double) burst_positions(24, 4), 6.0);
    eq("every position is a burst when qsa_every is 1", (double) burst_positions(24, 1), 24.0);
    yes("position 0 is a burst", is_burst_position(0, 4));
    yes("position 20 is a burst", is_burst_position(20, 4));
    yes("position 21 is not", !is_burst_position(21, 4));
    yes("no qsa pattern means no bursts", burst_positions(24, 0) == 0);

    // THE CENTRAL CLAIM OF THE HARNESS: concentrating the same bytes into fewer positions cannot hide better, and
    // the uniform schedule must therefore never read lower than the bursty one.
    {
        const double cover_pos = 2.5, burst = 30.0, bg = 0.3;
        const double h_two = hidden_from_positions(24, 4, cover_pos, burst, bg);
        const double h_uni = hidden_uniform(24, cover_pos, burst, bg);
        yes("the uniform schedule never hides LESS than the bursty one", h_uni >= h_two - 1e-12);
        yes("with a real burst the uniform schedule hides strictly MORE", h_uni > h_two);
        yes("hidden stays in [0,1]", h_two >= 0 && h_two <= 1 && h_uni >= 0 && h_uni <= 1);
        yes("a longer cover never lowers the hiding",
            hidden_from_positions(24, 4, 40.0, 30.0, 0.3) >= hidden_from_positions(24, 4, 2.5, 30.0, 0.3));
        yes("a bigger burst never raises the hiding",
            hidden_from_positions(24, 4, 2.5, 60.0, 0.3) <= hidden_from_positions(24, 4, 2.5, 30.0, 0.3));
        yes("no wire at all is fully hidden", hidden_from_positions(24, 4, 2.5, 0.0, 0.0) == 1.0);
        yes("bursts at every position is the hard case",
            hidden_from_positions(24, 1, 2.5, 30.0, 0.3) <= hidden_from_positions(24, 4, 2.5, 30.0, 0.3));
        yes("a cover that swamps everything hides fully",
            hidden_from_positions(24, 1, 1e6, 30.0, 0.3) == 1.0);
        yes("zero cover exposes the whole wire", hidden_from_positions(24, 1, 0.0, 30.0, 0.3) <= 0.0 + 1e-12);
        yes("fewer bursts hide better than many",
            hidden_from_positions(24, 8, 2.5, 30.0, 0.3) >= hidden_from_positions(24, 4, 2.5, 30.0, 0.3));
        yes("no cycle is fully hidden by convention", hidden_from_positions(0, 4, 2.5, 30.0, 0.3) == 1.0);
    }

    // The window bookkeeping: cover, wire and both are WHOLE-window totals over the same unit count. A per-unit
    // cover subtracted from a whole-window `both` produced a -800% hidden fraction once, so the relationship that
    // the RESULT line prints is asserted here rather than trusted.
    {
        Options o;                       // defaults: prefill, cycle 24, chunks 8
        o.mode = Mode::kPrefill;
        o.chunks = 6;
        o.cycle = 24;
        const int units = window_units(o);
        eq("prefill units are chunks x cycle", (double) units, 6.0 * 24.0);
        // cover_ms spread over the cycle means the whole cover window is cover_ms x chunks, not cover_ms
        eq("the whole cover window is the per-position cover times the unit count",
           (o.cover_ms / o.cycle) * units, o.cover_ms * o.chunks);
        yes("a whole-window both minus a per-unit cover would understate exposure by the unit count",
            (360.0 - 2.5) > (360.0 - 360.0) && units == 144);
        Options d;
        d.mode = Mode::kDecode;
        d.rounds = 96;
        eq("decode units are the round count, capped", (double) window_units(d), 96.0);
        d.rounds = kMaxUnits + 500;
        eq("decode units are capped at the event bookkeeping", (double) window_units(d), (double) kMaxUnits);
    }

    // the cadence-inflation model: a separate coefficient from the hidden fraction, never folded into it
    {
        const double cp = 2.5;
        yes("zero jitter leaves the cadence alone", predicted_cadence_inflation(24, 4, cp, 30.0, 0.3, 0.0) == 1.0);
        yes("more jitter never lowers the inflation",
            predicted_cadence_inflation(24, 4, cp, 30.0, 0.3, 25.0) >=
                predicted_cadence_inflation(24, 4, cp, 30.0, 0.3, 10.0));
        yes("any positive jitter inflates the barrier", predicted_cadence_inflation(24, 4, cp, 30.0, 0.3, 1.0) > 1.0);
        yes("where the wire already binds every position, a straggler inside the wire is free",
            predicted_cadence_inflation(24, 1, 2.5, 100.0, 100.0, 5.0) == 1.0);
        eq("where the cover binds, the same straggler costs its full delay",
           predicted_cadence_inflation(24, 1, 100.0, 1.0, 1.0, 10.0), (100.0 + 10.0) / 100.0);
        yes("inflation is at least 1", predicted_cadence_inflation(24, 4, cp, 30.0, 0.3, 1000.0) >= 1.0);
        yes("no cycle leaves the cadence alone by convention", predicted_cadence_inflation(0, 4, cp, 30.0, .3, 5) == 1.0);
        yes("inflation never touches the hidden fraction", hidden_from_positions(24, 4, cp, 30.0, 0.3) ==
                                                          hidden_from_positions(24, 4, cp, 30.0, 0.3));
    }

    // the break-even token count
    yes("break-even is positive", breakeven_tokens(Channel::kRowShip, 4, 2.5, 10, 2.0, 1) > 0);
    yes("a longer cover breaks even at more tokens",
        breakeven_tokens(Channel::kRowShip, 4, 5.0, 10, 2.0, 1) >
            breakeven_tokens(Channel::kRowShip, 4, 2.5, 10, 2.0, 1));
    yes("a slower link breaks even at fewer tokens",
        breakeven_tokens(Channel::kRowShip, 4, 2.5, 10, 0.64, 1) <
            breakeven_tokens(Channel::kRowShip, 4, 2.5, 10, 3.96, 1));
    yes("no rate means no tokens", breakeven_tokens(Channel::kRowShip, 4, 2.5, 10, 0, 1) == 0);
    yes("one rank means no cross traffic, so no break-even", breakeven_tokens(Channel::kRowShip, 1, 2.5, 10, 2.0, 1) == 0);
    // partial-merge moves only the sums across, so it costs fewer bytes per token and therefore tolerates a LONGER
    // attention before the burst stops hiding; row-ship ships every routed row and breaks even sooner.
    yes("row-ship breaks even at fewer tokens than partial-merge at topk 10",
        breakeven_tokens(Channel::kRowShip, 4, 2.5, 10, 2.0, 1) <
            breakeven_tokens(Channel::kPartialMerge, 4, 2.5, 10, 2.0, 1));
    yes("row-ship breaks even sooner as topk grows", breakeven_tokens(Channel::kRowShip, 4, 2.5, 20, 2.0, 1) <
                                                         breakeven_tokens(Channel::kRowShip, 4, 2.5, 10, 2.0, 1));
    yes("partial-merge is unaffected by topk", breakeven_tokens(Channel::kPartialMerge, 4, 2.5, 10, 2.0, 1) ==
                                                   breakeven_tokens(Channel::kPartialMerge, 4, 2.5, 16, 2.0, 1));

    // the footprint and the schedule helpers
    {
        const long long cb = 32ll << 20, rb = 4ll << 20;
        eq("footprint counts chunks, ranks, cover, tier pair", (double) footprint_bytes(4, 4, rb, cb),
           (double) (4ll * 6 * rb + cb + 2 * (long long) kTierBlob + 4096));
        yes("footprint grows with ranks", footprint_bytes(4, 4, rb, cb) > footprint_bytes(2, 4, rb, cb));
        yes("footprint grows with chunks", footprint_bytes(4, 8, rb, cb) > footprint_bytes(4, 4, rb, cb));
        yes("footprint grows with the round", footprint_bytes(4, 4, 8 << 20, cb) > footprint_bytes(4, 4, rb, cb));

        Options d;
        d.mode = Mode::kDecode;
        d.rounds = 96;
        eq("decode issues one window per round", (double) window_units(d), 96.0);
        yes("every decode round is a burst", unit_is_burst(d, 0) && unit_is_burst(d, 95));

        Options p;
        p.mode = Mode::kPrefill;
        p.chunks = 2;
        p.cycle = 24;
        p.qsa_every = 4;
        eq("prefill issues cycle x chunks units", (double) window_units(p), 48.0);
        yes("position 0 bursts and position 1 does not", unit_is_burst(p, 0) && !unit_is_burst(p, 1));
        yes("the uniform schedule bursts everywhere", [&] { Options u = p; u.uniform = true;
            for (int i = 0; i < 24; ++i) if (!unit_is_burst(u, i)) return false; return true; }());
        yes("a burst unit moves more than a background one",
            unit_bytes(p, 0) > unit_bytes(p, 1));
        Options pm = p;
        pm.mode = Mode::kDecode;
        pm.decode_payload_kib = 8;
        eq("decode round bytes are the payload", (double) round_bytes(pm), 8.0 * 1024);
    }

    // the partial hash: stable, in range, sensitive enough that parity cannot pass for the wrong reason
    {
        yes("the same key gives the same value", partial_value(1, 0, 5) == partial_value(1, 0, 5));
        yes("another rank gives another value", partial_value(1, 0, 5) != partial_value(2, 0, 5));
        yes("another set gives another value", partial_value(1, 0, 5) != partial_value(1, 1, 5));
        yes("another element gives another value", partial_value(1, 0, 5) != partial_value(1, 0, 6));
        bool in_range = true, spread = false;
        double sum = 0;
        for (long long i = 0; i < 8192; ++i) {
            const float v = partial_value(3, 2, i);
            if (!(v >= -1.0f && v < 1.0f)) in_range = false;
            if (std::fabs(v) > 0.5f) spread = true;
            sum += v;
        }
        yes("values are in [-1,1)", in_range);
        yes("values actually spread", spread);
        yes("near zero mean", std::fabs(sum / 8192.0) < 0.05);
        yes("swapping which peer contributed moves the sum",
            partial_value(0, 0, 7) + partial_value(1, 0, 7) != partial_value(0, 0, 7) + partial_value(2, 0, 7));
    }

    yes("the 4-rank pattern is the all-to-all one", link_rate(4, -1) == kRates[4].gib_s);
    yes("the 2-rank pattern is the both-ways pair", link_rate(2, -1) == kRates[1].gib_s);
    yes("--rate overrides the table", link_rate(4, 5.0) == 5.0);

    if (bad == 0) { std::printf("selftest: PASS (all model assertions hold)\n"); return 0; }
    std::fprintf(stderr, "selftest: %d FAILURE(S)\n", bad);
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "cadence probe: %s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--sweep") o.sweep = true;
        else if (a == "--parity") o.parity = true;
        else if (a == "--allow-big") o.allow_big = true;
        else if (a == "--layout") o.layout_only = true;
        else if (a == "--selftest") o.selftest = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--uniform") o.uniform = true;
        else if (a == "--mode") {
            const std::string v = next("--mode");
            if (v == "prefill") o.mode = Mode::kPrefill;
            else if (v == "decode") o.mode = Mode::kDecode;
            else { std::fprintf(stderr, "cadence probe: --mode is prefill or decode\n"); return 2; }
        } else if (a == "--channel") {
            const std::string v = next("--channel");
            if (v == "row-ship") o.channel = Channel::kRowShip;
            else if (v == "partial-merge") o.channel = Channel::kPartialMerge;
            else { std::fprintf(stderr, "cadence probe: --channel is row-ship or partial-merge\n"); return 2; }
        } else if (a == "--ranks") o.ranks = std::atoi(next("--ranks"));
        else if (a == "--depth") o.depth = std::atoi(next("--depth"));
        else if (a == "--tokens") o.tokens = std::atoi(next("--tokens"));
        else if (a == "--chunks") o.chunks = std::atoi(next("--chunks"));
        else if (a == "--reps") o.reps = std::atoi(next("--reps"));
        else if (a == "--topk") o.topk = std::atoi(next("--topk"));
        else if (a == "--cycle") o.cycle = std::atoi(next("--cycle"));
        else if (a == "--qsa-every") o.qsa_every = std::atoi(next("--qsa-every"));
        else if (a == "--merge-partners") o.merge_partners = std::atoi(next("--merge-partners"));
        else if (a == "--rounds") o.rounds = std::atoi(next("--rounds"));
        else if (a == "--cover-ms") o.cover_ms = std::atof(next("--cover-ms"));
        else if (a == "--cover-mib") o.cover_mib = std::atoll(next("--cover-mib"));
        else if (a == "--bg-gib-s") o.bg_gib_s = std::atof(next("--bg-gib-s"));
        else if (a == "--jitter-card") o.jitter_card = std::atoi(next("--jitter-card"));
        else if (a == "--jitter-ms") o.jitter_ms = std::atof(next("--jitter-ms"));
        else if (a == "--jitter-blobs") o.jitter_blobs = std::atoi(next("--jitter-blobs"));
        else if (a == "--tier-blobs") o.tier_blobs = std::atoi(next("--tier-blobs"));
        else if (a == "--decode-payload-kib") o.decode_payload_kib = std::atoll(next("--decode-payload-kib"));
        else if (a == "--push") {
            const std::string v = next("--push");
            if (v == "sm") o.sm_push = true;
            else if (v == "copy-engine") o.sm_push = false;
            else { std::fprintf(stderr, "cadence probe: --push is sm or copy-engine\n"); return 2; }
        } else if (a == "--rate") o.rate = std::atof(next("--rate"));
        else if (a == "--budget-mib") o.budget = (size_t) std::atoll(next("--budget-mib")) << 20;
        else { std::fprintf(stderr, "cadence probe: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.selftest) return run_selftest();
    if (o.layout_only) {
        print_model(o);
        print_prediction_grid();   // the filed table travels with the model it came from
        if (o.sweep) print_sweep_plan();
        return 0;
    }

    if (o.ranks < 2 || o.ranks > kMaxRanks) { std::fprintf(stderr, "cadence probe: --ranks is 2..%d\n", kMaxRanks); return 2; }
    if (o.depth < 1 || o.depth > kMaxDepth) { std::fprintf(stderr, "cadence probe: --depth is 1..%d\n", kMaxDepth); return 2; }
    if (o.tokens < 1) { std::fprintf(stderr, "cadence probe: --tokens is >= 1\n"); return 2; }
    if (o.chunks < o.depth + 1 || o.chunks > kMaxChunks) {
        std::fprintf(stderr, "cadence probe: --chunks must exceed --depth and be <= %d\n", kMaxChunks);
        return 2;
    }
    if (o.reps < 20 || o.reps % 2 == 0) { std::fprintf(stderr, "cadence probe: --reps is an odd number of at least 20\n"); return 2; }
    if (o.cover_ms <= 0 || o.cycle < 1) { std::fprintf(stderr, "cadence probe: --cover-ms and --cycle are >= 1\n"); return 2; }
    if (o.qsa_every < 0 || o.qsa_every > o.cycle) { std::fprintf(stderr, "cadence probe: --qsa-every is 0..--cycle\n"); return 2; }
    if (o.topk < 1 || o.topk > kExperts) { std::fprintf(stderr, "cadence probe: --topk is 1..%d\n", kExperts); return 2; }
    if (o.jitter_card >= o.ranks) { std::fprintf(stderr, "cadence probe: --jitter-card is < --ranks\n"); return 2; }
    if (o.mode == Mode::kDecode && (o.rounds < 1 || o.rounds > kMaxUnits)) {
        std::fprintf(stderr, "cadence probe: --rounds is 1..%d (the engine pays 96 per window)\n", kMaxUnits);
        return 2;
    }
    if (o.decode_payload_kib < 1) { std::fprintf(stderr, "cadence probe: --decode-payload-kib is >= 1\n"); return 2; }
    if (o.cover_mib < 1) { std::fprintf(stderr, "cadence probe: --cover-mib is >= 1\n"); return 2; }

    int count = 0;
    if (!fail(cudaGetDeviceCount(&count), "device count")) return 1;
    if (count < o.ranks) { std::fprintf(stderr, "cadence probe: %d cards asked for, %d present\n", o.ranks, count); return 1; }
    std::printf("tp_cadence_probe: mode %s, channel %s, ranks %d, depth %d, tokens %d, reps %d, push %s\n",
                o.mode == Mode::kDecode ? "decode" : "prefill", channel_name(o.channel), o.ranks, o.depth, o.tokens,
                o.reps, o.sm_push ? "SM" : "copy-engine");
    for (int i = 0; i < o.ranks; ++i) {
        cudaDeviceProp p;
        if (!fail(cudaGetDeviceProperties(&p, i), "device properties")) return 1;
        std::printf("  CUDA%d %s, %d SMs\n", i, p.name, p.multiProcessorCount);
    }
    std::vector<std::string> no_access;
    open_peers(o.ranks, no_access);
    if (no_access.empty()) std::printf("every ordered pair has GPU-to-GPU access\n");
    else {
        std::printf("pairs with NO GPU-to-GPU access:");
        for (const auto& s : no_access) std::printf(" %s", s.c_str());
        std::printf("\n");
    }

    std::vector<Result> results;
    bool ok = true;
    auto point = [&](const Options& p) {
        std::vector<Rank> ranks = make_ranks(p.ranks);
        Result r;
        const bool ran = run_point(ranks, p, &r);
        for (Rank& rk : ranks) release_rank(rk);   // idempotent: run_point already freed them
        if (r.ran) results.push_back(r);
        ok = ok && ran;
    };

    if (o.sweep) {
        // the decision grid: both combine channels x {2,4} ranks x {2,4} deep, on the two-channel schedule
        for (Channel ch : {Channel::kPartialMerge, Channel::kRowShip})
            for (int r : {2, 4})
                for (int d : {2, 4}) {
                    Options s = o;
                    s.channel = ch;
                    s.ranks = r;
                    s.depth = d;
                    point(s);
                }
        // the bias reference: the same bytes spread over every position, at 4 ranks 4 deep
        Options u = o;
        u.uniform = true;
        u.ranks = 4;
        u.depth = 4;
        point(u);
        // the max-of-4 propagation, per channel: one card slowed, the untouched cards' cadence read back
        for (Channel ch : {Channel::kPartialMerge, Channel::kRowShip})
            for (int j : {10, 25}) {
                Options t = o;
                t.channel = ch;
                t.ranks = 4;
                t.depth = 4;
                t.jitter_card = 3;
                t.jitter_ms = j;
                point(t);
            }
        // the decode cadence: the two push engines in SEPARATE points, with and without a tier fill
        for (bool sm : {true, false})
            for (int blobs : {0, 4}) {
                Options dc = o;
                dc.mode = Mode::kDecode;
                dc.ranks = 4;
                dc.sm_push = sm;
                dc.tier_blobs = blobs;
                point(dc);
            }
    } else {
        point(o);
    }

    if (results.size() > 1) {
        std::printf("\nrendezvous-cadence summary - hidden = 1 - (exposed wire / total wire). The marker column is\n"
                    "the 70%% line shown for orientation only; the threshold math is the supervisor's.\n");
        std::printf("  %-7s %-14s %5s %5s %9s %9s %8s %8s %9s %s\n", "mode", "channel", "ranks", "deep", "wire",
                    "exposed", "hidden", "model", "parity", "marker");
        for (const Result& r : results)
            std::printf("  %-7s %-14s %5d %5d %9.2f %9.2f %7.1f%% %7.1f%% %9s %s\n",
                        r.mode == Mode::kDecode ? "decode" : "prefill", channel_name(r.channel), r.ranks, r.depth,
                        r.window_reduce, r.exposed_ms, r.hidden * 100.0, r.hidden_model * 100.0,
                        r.parity_pass ? "PASS" : "-", r.hidden >= 0.70 ? ">=70" : "<70");
        std::printf("\n  read each channel's 4-rank rows against its own 2-rank rows: a hidden fraction that falls\n"
                    "  from 2 to 4 ranks is the max-of-4 barrier, and the cadence inflation column says by how\n"
                    "  much. Row-ship's byte ledger is what decides whether that channel is benchable at all on\n"
                    "  this fabric; the uniform row is the bias, not the answer.\n");
    }
    std::printf("\ndone; every buffer freed\n");
    return ok ? 0 : 1;
}

// include/basalt/prefill/chunk_dump.hpp - the per-layer record `--exp chunk_dump` prints.
//
// Two boots of one build and one command are meant to compute the same logits. When they do not, the line a chunk
// ends with (`expert-visits lead N peer M ...`) says the two parted somewhere without saying where, and the gate's
// row file says only that they did. This is the record that says where: the kernel a layer's experts ran on, who
// served each of its experts, and a fingerprint of the routing and of that division. Diff two boots' lines and the
// first layer whose line differs names the decision that parted them - which of the two it is (a run whose router
// parted, or one whose row division moved while the router agreed) follows from which of the two words differs.
//
// Only what the host already holds goes into a record. A fingerprint that needed the device would add a sync to the
// very window whose scheduling the flag exists to catch, and would hide the thing it was built to find.
//
// Header-only so the CPU parity test grades the classification and the fingerprints the engine itself prints, the
// way `peer_band.hpp` does for the band's share.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::prefill {

/// The expert kernels a prompt layer runs on, as `chunk_kernel` settles them.
enum class ChunkKernel : int32_t { fp16 = 0, mmq = 1, fp4 = 2, fused = 3, fused_native = 4 };

/// The kernel a layer's experts ran on, from the flags `Prefill::run_impl` read for it.  The two fused ones imply
/// the non-fused ones are inert (the layer runs one path or the other, never both), so a fused flag wins; of the
/// rest, the FP4 leg is a narrower choice than MMQ, which it replaces for the layer.
inline ChunkKernel chunk_kernel(bool use_mmq, bool fp4_l, bool fused_l, bool fused_nat) {
    if (fused_nat) return ChunkKernel::fused_native;
    if (fused_l) return ChunkKernel::fused;
    if (fp4_l) return ChunkKernel::fp4;
    if (use_mmq) return ChunkKernel::mmq;
    return ChunkKernel::fp16;
}

inline const char* chunk_kernel_name(ChunkKernel k) {
    switch (k) {
        case ChunkKernel::fused_native: return "fused-nat";
        case ChunkKernel::fused: return "fused";
        case ChunkKernel::fp4: return "fp4";
        case ChunkKernel::mmq: return "mmq";
        case ChunkKernel::fp16: break;
    }
    return "fp16";
}

/// Where one expert's rows of a layer were computed.  The order the engine applies the three tables is this one:
/// the CPU's block takes an expert it claimed even when the partner's table names it too (the lead's own walk and
/// the partner's both skip an expert the CPU took), the partner's rows come next, and what is left is this card's.
enum class ChunkSource : uint8_t { lead = 0, peer = 1, cpu = 2 };

inline ChunkSource chunk_source(bool on_peer, bool on_cpu) {
    if (on_cpu) return ChunkSource::cpu;
    if (on_peer) return ChunkSource::peer;
    return ChunkSource::lead;
}

/// FNV-1a's parameters, so a fingerprint printed by one boot means the same to another.
constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

/// FNV-1a over `n` bytes, continued from `h`: three operations a byte, small enough to audit, and not a number
/// anyone will read as a quantity.  Cheap enough to run over a layer's routing every layer of every chunk.
inline uint64_t chunk_hash(uint64_t h, const void* data, size_t n) {
    const uint8_t* b = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= kFnvPrime;
    }
    return h;
}

inline uint64_t chunk_hash(const void* data, size_t n) { return chunk_hash(kFnvOffset, data, n); }

/// The routing's fingerprint: the expert ids the layer routed the chunk's tokens to, in the order the host read
/// them.  Two boots whose routers agreed print the same word; the first layer where they differ is where one
/// router's input had already parted from the other's.  Null ids - a path that keeps the routing on the card,
/// which is the fused path's point - fingerprints as the offset basis, which no real routing does.
inline uint64_t chunk_hash_ids(const int32_t* ids, int64_t n) {
    return ids != nullptr ? chunk_hash(ids, (size_t) n * sizeof(int32_t)) : kFnvOffset;
}

/// The division's fingerprint: which card computed each of the layer's experts, in id order.  A null table means
/// nothing was decided for that card - single-GPU, or no CPU share - which reads as every expert staying here, as
/// indeed it does.
inline uint64_t chunk_hash_split(const char* on_peer, const char* on_cpu, int64_t n_expert) {
    uint64_t h = kFnvOffset;
    for (int64_t e = 0; e < n_expert; ++e) {
        const uint8_t c =
            (uint8_t) chunk_source(on_peer != nullptr && on_peer[e] != 0, on_cpu != nullptr && on_cpu[e] != 0);
        h = chunk_hash(h, &c, 1);
    }
    return h;
}

/// What one layer of one chunk printed.  Filled where its decisions are made - the kernel's name as the paths
/// divide, the division's counts and fingerprints once the walk has fixed them - and printed at the layer's end,
/// where both paths have put in everything they know.  A field a path cannot know keeps the value it was given
/// here, so a fused layer's line reads its own kernel and zeros for a division it never made.
struct ChunkDump {
    int64_t layer = -1, p0 = -1, T = 0, K = 0, n_expert = 0;
    const char* path = "unset";
    // the gates that picked the path and the share, as the layer read them
    int stream_all = 0, peer_fp4 = 0, peer_ram = 0, band_pin = 0, fp4_planes = 0, peer_now = 0, acts_sent = 0;
    int rows_bf16 = 0, partial_merge = 0, p2p = 0, native_pack = 0, fused_on = 0, dump = 0;
    int64_t ring = 0, ring_min = 0, peer_dev = -1, ram_num = 0, ram_top = 0, T_fp4 = 0, T_max = 0;
    // the division: experts and rows per card
    int64_t lead = 0, lead_rows = 0, peer = 0, peer_rows = 0, band = 0, cpu = 0, cpu_rows = 0, streamed = 0;
    // the tier and band membership of the layer, as this chunk's walk counted them
    int64_t lead_holds = 0, peer_holds = 0, band_all = 0;
    uint64_t ids = kFnvOffset, split = kFnvOffset;
};

}  // namespace basalt::prefill

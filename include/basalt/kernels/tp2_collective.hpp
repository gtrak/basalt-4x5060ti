// include/basalt/kernels/tp2_collective.hpp - the one collective a tensor-parallel pair needs: a 2-way all-reduce of
// a layer's partial sums.
//
// Two ranks, one exchange each: a rank PUSHES its partial into the peer's staging buffer with `cudaMemcpyPeerAsync`
// (peer access is on for every pair of cards the engine selects), raises a flag on the peer, then waits for its own
// flag and adds what arrived.  Two ranks x one push is the minimum traffic a sum of two partials can have - a
// pull model needs the same bytes and a ring needs two hops - and there is no barrier: the flag IS the ordering, so
// a rank that gets there first waits inside its own kernel and holds no link open.
//
// WHY NOT NCCL.  An NCCL ring on this box's PCIe/PHB links costs more than the GEMM it saves (the sglang worklog's
// lesson, recorded in the TP2 notes), and it wants its own communicator, its own buffers and its own graph-capture
// rules.  NCCL stays a later production decision; this file is the exploration's helper.
//
// WHY THE PURE PARTS ARE HERE AND INLINE.  `tp2_all_reduce_ref`, the byte figure and the geometry check are the
// contract the parity test reads and the engine's sizing calls; they must be usable without a CUDA link, the way
// gdn_head_split.hpp is.  The kernels are in src/kernels/cuda/tp2_collective.cu.
//
// THE NUMBERS: `rows` x `stride` floats is 4 B per element, so a 2048-column partial of 8 tokens is 64 KiB per
// rank per collective, one per split linear per layer.  A layer with a split o_proj and a split down_proj at 48
// layers over a 24-layer stage is 24 collectives per window.
#pragma once

#include <cstdint>
#include <string>

namespace basalt::kernels {

// ================================ the pure parts (no CUDA, no device, testable as they stand) ================================

/// Bytes one rank moves in one collective: its own partial, pushed to the peer.  `stride` is the row pitch in
/// floats, which is what is copied - a padded pitch keeps the copy one operation and the add one index rule.
inline uint64_t tp2_all_reduce_bytes(int64_t rows, int64_t stride) {
    return rows > 0 && stride > 0 ? (uint64_t) rows * (uint64_t) stride * 4u : 0u;
}

/// True when a collective of this shape is well-formed: non-empty, inside the pair's staging capacity, and with a
/// pitch no narrower than the rows it holds.  The engine sizes its staging once from the widest partial it will
/// reduce; every later call has to fit that, or the push would run past the peer's buffer.
inline bool tp2_all_reduce_ok(int64_t rows, int64_t stride, int64_t max_elems) {
    return rows > 0 && stride > 0 && rows <= max_elems / stride;
}

/// `out[i] = a[i] + b[i]` over `rows` x `stride`, the row pitch honoured on all three buffers.  This is the
/// reference `tp2_all_reduce` is parity-tested against, and what the CPU-side row-split A/B compares to.
///
/// The ORDER of the add is the contract: `a` is this rank's partial and `b` the peer's, and a float add is not
/// associative.  A split linear whose partials are summed here is therefore NOT bitwise equal to the same linear
/// computed whole on one card - that is the numerics gate the brief attaches to the row split, not to the head
/// split.  `in place` is legal: `out` may alias `a`.
inline void tp2_all_reduce_ref(float* out, const float* a, const float* b, int64_t rows, int64_t stride) {
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t o = r * stride;
        for (int64_t c = 0; c < stride; ++c) out[o + c] = a[o + c] + b[o + c];
    }
}

/// The two cards of a stage pair, in the order the engine opens them: rank 0 is the stage's lead card, rank 1 its
/// partner.  Pure, so the sequence bookkeeping below can be reasoned about (and tested) without a device.
inline int tp2_peer(int rank) { return rank ^ 1; }

/// Does one rank's issue of collective number `seq` (1-based, as the device counters count them) wait for a push
/// whose flag carries this value?  Both ranks count their own issues from 1, so in lockstep the two figures are
/// equal; a rank that has issued more than its peer is the one that waits.  Kept as a function because the
/// parity test asserts the lockstep property rather than assuming it.
inline bool tp2_push_arrived(int64_t pushed_flag, int64_t seq) { return pushed_flag >= seq; }

// ================================ the pair (device state; needs a CUDA link) ================================

/// One tensor-parallel pair's collective state: a staging buffer, a "pushed by my peer" flag and a sequence counter
/// on EACH card, plus a host-visible timeout word.  Opened once at start with the widest partial the engine will
/// reduce; every collective of that shape or smaller reuses it.
///
/// THREADS: the pair is driven by whichever thread runs the stage's window, and both ranks' work is issued from it
/// (the stage's two cards are one stage).  The sequence counters live on the devices, which is what lets a captured
/// graph replay a collective: nothing per-call is baked into the graph.
class Tp2Pair {
public:
    Tp2Pair() = default;
    Tp2Pair(const Tp2Pair&) = delete;
    Tp2Pair& operator=(const Tp2Pair&) = delete;
    ~Tp2Pair();

    /// Allocates the staging and the words on both cards.  `dev_r` are the pair's CUDA devices, `max_elems` the
    /// largest partial (in floats) any collective will carry.  Refuses when peer access between the two is not
    /// available: a pair that cannot reach across has no collective to run.
    bool open(int dev_a, int dev_b, int64_t max_elems, std::string& err);
    void close();
    bool valid() const { return staging_[0] != nullptr && staging_[1] != nullptr; }
    int64_t max_elems() const { return max_elems_; }
    int device(int rank) const { return rank == 0 ? dev_a_ : dev_b_; }

    /// The collective, with the brief's parameter names: `buf` (`rows` x `stride` floats, on `dev`) gets its
    /// peer's partial added to it, IN PLACE, so both cards end with the same sum.  Issued on `stream`; every later
    /// reader of `buf` on that stream sees the sum, and nothing on the peer's stream may read its own `buf` before
    /// its own call here.
    ///
    /// `rank` says which side of the pair `dev` is (the caller knows: the lead card is rank 0).  A call the peer
    /// never issues spins out at the timeout and returns false - the pair is then dead and must be closed.
    ///
    /// **THE TWO RANKS' CALLS MUST BE ON STREAMS THAT MAKE INDEPENDENT PROGRESS.**  Both cards have their own
    /// stream in the engine, which is what lets rank 0's waiting kernel run while rank 1's push is still queued.
    /// Two collectives on ONE stream deadlock until the timeout: the first rank's add would wait for a push the
    /// same stream has not issued yet.
    bool all_reduce(float* buf, int64_t rows, int64_t stride, void* stream, int peer_dev, int rank, std::string& err);

    /// Collectives issued per rank, and the bytes pushed per rank (the microbench reports both).
    int64_t issued(int rank) const { return rank == 0 ? issued_a_ : issued_b_; }
    uint64_t pushed_bytes(int rank) const { return rank == 0 ? bytes_a_ : bytes_b_; }
    /// A rank that waited too long for its peer's push set its word; the engine logs it and stops using the pair.
    bool timed_out() const { return timed_out(0) || timed_out(1); }
    bool timed_out(int rank) const;

private:
    int dev_a_ = -1, dev_b_ = -1;
    float* staging_[2] = {nullptr, nullptr};
    // one device word per rank: what the peer's last push carried, and how many this rank has issued
    int32_t* pushed_[2] = {nullptr, nullptr};
    int32_t* seq_[2] = {nullptr, nullptr};
    int32_t* timeout_[2] = {nullptr, nullptr};   // mapped host memory, one per card: that rank sets its own
    int64_t max_elems_ = 0;
    int64_t issued_a_ = 0, issued_b_ = 0;
    uint64_t bytes_a_ = 0, bytes_b_ = 0;
};

/// The same collective as a free function, for a caller that already holds the pair.
///
/// THE TIMEOUT IS PART OF THE RESULT.  `tp2_add_wait` spins to its budget, sets the rank's mapped flag and
/// RETURNS WITHOUT ADDING, leaving `buf` holding this rank's partial alone - and `all_reduce` still returns true
/// with an empty `err`.  A caller that checks only the return value therefore cannot tell a completed reduce
/// from a half one, and a half reduce is not an error the tokens recover from: it is attention summed over one
/// card's heads, presented as the model's own output.  Failing loudly here is the only place that catches it,
/// because no engine caller checks `timed_out()` itself.
///
/// The check is at ENTRY, not after the launch: the spin budget is around a second of device time, so a flag
/// read immediately after `all_reduce` returns describes a kernel that has not run yet and would be a check that
/// always passes.  At entry, a spin that gave up has long since been written.  What entry cannot see is the
/// collective it is itself starting - so the property this buys is "a half reduce fails the run no later than
/// the next collective of the pair", which for a per-layer reduce means the very next layer.  A caller that must
/// know within one layer reads `timed_out(rank)` after its own stream sync.
inline bool tp2_all_reduce(Tp2Pair& pair, float* buf, int64_t rows, int64_t stride, void* stream, int peer_dev,
                           int rank, std::string& err) {
    // Both ranks are checked, not just this one: the stage drives both cards from one thread, so a peer whose
    // last collective gave up has already left THIS one's sum short, whatever this rank's own flag says.
    if (const int r = pair.timed_out(0) ? 0 : (pair.timed_out(1) ? 1 : -1); r >= 0) {
        err = "tp2 all-reduce: CUDA" + std::to_string(pair.device(r)) + " gave up waiting for its peer's o_proj "
              "partial; the last reduce on this pair left that card holding only its own sum, so the pair is not "
              "in a state to reduce again";
        return false;
    }
    return pair.all_reduce(buf, rows, stride, stream, peer_dev, rank, err);
}

}  // namespace basalt::kernels

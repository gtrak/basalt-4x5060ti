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

// The sample rule below is read by the host that grades a pair and by the kernel that takes the sample, so it is
// spelled once and compiled both ways (`f16_bits.hpp` does the same for the bit helpers).
#if defined(__CUDACC__)
#define TP2_HD __host__ __device__
#else
#define TP2_HD
#endif

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

/// THE STAGING SETS, AND WHY ONE IS NOT ALWAYS ENOUGH.  Each card has one landing buffer per SET: a rank pushes
/// its partial into the peer's region `set`, raises the peer's flag, then adds the peer's region `set` into its own
/// buffer.  Nothing credits a region back after the add read it, so the ONLY thing protecting a region is the gap
/// between the two collectives that use it.  With `sets` regions that gap is `sets` collectives, and the pair's own
/// flag protocol puts a provable order inside it: my add of collective n is followed, on my stream, by my push of
/// n+1 and its flag; the peer's add of n+1 cannot complete before that flag; and the peer's push of n+`sets` - the
/// next one to touch region n%sets - sits behind that add on ITS stream.  So my add of n has finished before the
/// peer may write that region again, and one region per set is exactly enough.
///
/// With `sets` 1 the gap is one collective, and the chain above needs the peer's add of n+1 to be gated on my flag
/// of n+1 - which is only there if some other wait keeps a rank a whole collective behind its peer.  A caller that
/// removes such a wait (running two layers in flight, say) must raise `sets` to match, or a peer's next push lands
/// in a region this rank's add is still reading: the sum silently takes the NEXT layer's partial.
/// `tp2_all_reduce_ref` cannot see that, and neither can a parity test of one collective.
inline uint64_t tp2_staging_elems(int64_t max_elems, int sets) {
    return (uint64_t) (sets > 1 ? sets : 1) * (uint64_t) (max_elems > 0 ? max_elems : 0);
}

/// Is `set` one of the regions a pair opened with `sets` has?  Pure, so the refusal's rule reads without a device.
inline bool tp2_staging_set_ok(int set, int sets) { return set >= 0 && set < (sets > 0 ? sets : 1); }

/// THE PAIR-AGREEMENT SAMPLE.  After a collective both cards hold the SAME sum: each added its own partial and its
/// peer's, and IEEE addition commutes, so element for element the two buffers must be bit-identical.  Nothing else
/// in the engine can see a pair that disagreed - a sum built from the peer's NEXT layer's partial, or one the spin
/// budget abandoned, is a plausible token, not an error - so the engine samples a few elements of each card's
/// summed partial and compares them where it can still fail the run.
///
/// EIGHT elements, spread over both axes: the rows the sum is taken over, and the columns, which is what makes a
/// wrong landing-region offset visible rather than merely possible.  One rule, here, read by the kernel and by the
/// host that grades it: two spellings of a sample index is one way for the check to pass for the wrong reason.
constexpr int kTp2Samples = 8;
TP2_HD inline int64_t tp2_sample_index(int k, int64_t rows, int64_t stride) {
    if (k < 0 || k >= kTp2Samples || rows <= 0 || stride <= 0) return -1;
    const int64_t r = kTp2Samples > 1 ? (rows - 1) * k / (kTp2Samples - 1) : 0;
    const int64_t c = (k * 37) % stride;   // coprime-ish step, so the columns are not all one residue class
    return r * stride + c;
}

/// What the host expects to find at sample `k`, given both cards' partials: the same element of each, added in the
/// order the kernel adds them.  Reference-shaped, so the parity test can grade the kernel without a pair.
inline float tp2_sample_ref(const float* a, const float* b, int k, int64_t rows, int64_t stride) {
    const int64_t i = tp2_sample_index(k, rows, stride);
    return i < 0 ? 0.0f : a[i] + b[i];
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
    /// largest partial (in floats) any collective will carry, and `sets` how many such regions each card gets
    /// (see THE STAGING SETS above; 1 is the historical single buffer).  Refuses when peer access between the two
    /// is not available: a pair that cannot reach across has no collective to run.
    bool open(int dev_a, int dev_b, int64_t max_elems, std::string& err, int sets = 1);
    void close();
    bool valid() const { return staging_[0] != nullptr && staging_[1] != nullptr; }
    int64_t max_elems() const { return max_elems_; }
    int sets() const { return sets_; }
    int device(int rank) const { return rank == 0 ? dev_a_ : dev_b_; }

    /// The collective, with the brief's parameter names: `buf` (`rows` x `stride` floats, on `dev`) gets its
    /// peer's partial added to it, IN PLACE, so both cards end with the same sum.  Issued on `stream`; every later
    /// reader of `buf` on that stream sees the sum, and nothing on the peer's stream may read its own `buf` before
    /// its own call here.
    ///
    /// `rank` says which side of the pair `dev` is (the caller knows: the lead card is rank 0).  A call the peer
    /// never issues spins out at the timeout and returns false - the pair is then dead and must be closed.
    ///
    /// `set` chooses the landing region, and BOTH ranks must pass the SAME one for the same collective - they do
    /// when both derive it from the same layer ordinal, which is what the prompt-attention arm does with its wire
    /// ordinal.  A caller that keeps more than one collective in flight must pass a different set per in-flight
    /// collective, or open the pair with enough sets for the depth it runs (see THE STAGING SETS above).
    ///
    /// **THE TWO RANKS' CALLS MUST BE ON STREAMS THAT MAKE INDEPENDENT PROGRESS.**  Both cards have their own
    /// stream in the engine, which is what lets rank 0's waiting kernel run while rank 1's push is still queued.
    /// Two collectives on ONE stream deadlock until the timeout: the first rank's add would wait for a push the
    /// same stream has not issued yet.
    bool all_reduce(float* buf, int64_t rows, int64_t stride, void* stream, int peer_dev, int rank, std::string& err,
                    int set = 0);

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
    int sets_ = 1;                        // landing regions per card; see THE STAGING SETS above
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
                           int rank, std::string& err, int set = 0) {
    // Both ranks are checked, not just this one: the stage drives both cards from one thread, so a peer whose
    // last collective gave up has already left THIS one's sum short, whatever this rank's own flag says.
    if (const int r = pair.timed_out(0) ? 0 : (pair.timed_out(1) ? 1 : -1); r >= 0) {
        err = "tp2 all-reduce: CUDA" + std::to_string(pair.device(r)) + " gave up waiting for its peer's o_proj "
              "partial; the last reduce on this pair left that card holding only its own sum, so the pair is not "
              "in a state to reduce again";
        return false;
    }
    return pair.all_reduce(buf, rows, stride, stream, peer_dev, rank, err, set);
}

/// Copy the pair-agreement sample of a summed partial into `dst` (`kTp2Samples` floats), on `stream`, so the caller
/// can compare what its card summed against what its peer's card summed.  Issued on the collective's stream, which
/// is the point: the sample sees exactly that layer's sum, and nothing a later layer writes into the buffer.
/// `dst` is device memory on the card `src` is on; the engine zeroes it at the chunk's start, so a layer that
/// sampled on one card and not the other is a mismatch rather than a coincidence of stale bytes.
///
/// The kernel is in src/kernels/cuda/tp2_collective.cu; the indices it reads are `tp2_sample_index` above, which is
/// what lets the host grade it (and what `tp2_allreduce_parity` compares it against).
bool tp2_sample_partial(float* dst, const float* src, int64_t rows, int64_t stride, void* stream, std::string& err);

}  // namespace basalt::kernels

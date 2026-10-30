// include/basalt/prefill/prefill.hpp - batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights: every quantized projection must carry its GGUF blocks.
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/session.hpp"
#include "basalt/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core { class PeerExperts; }
namespace basalt::kernels::cpu { class ExpertPool; }

namespace basalt::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    int64_t experts_cpu = 0;        ///< experts computed on the CPU pool instead of streamed
    double cpu_share = 0;           ///< the share of the streamed experts the CPU took last
    double ms_ple = 0;
};

}  // namespace basalt::prefill
namespace basalt::core { class MtpDrafter; }
namespace basalt::prefill {

class Prefill {
public:
    /// The draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual rows `R_rows`
    /// (device) and `next_tokens` (host: the token at cell+1), in batches through this path's GEMMs and its idle
    /// scratch - call it from on_chunk.  false with `err` empty: not applicable here (a hybrid K/V, another
    /// device, too little scratch), the caller runs the drafter's own pass.  Not bit-identical to that pass: the
    /// drafts may differ, never the target's tokens' logits.
    bool draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                  std::string& err);
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// Frees every buffer, stream and event `init` made (as the destructor does) and starts over empty, so `init` can
    /// run again - with a smaller chunk when the first one did not fit.  The stage range (`set_stage`) and the
    /// callbacks stay.  The device `init` ran on must be current.
    void reset();

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);
    int64_t chunk() const;
    /// The next runs read their first chunk with at most `tokens` tokens (0: every chunk is `chunk()` long), so the
    /// rows it waits for before layer 1 are few; the chunks after it are `chunk()` long.
    void set_head_chunk(int64_t tokens) { head_chunk_ = tokens; }
    /// Points the prompt path at another session's state (a view of `ss` with a batch slot's state fields) for the
    /// next runs; the buffers and the layer range stay.
    void set_session(core::SessionState& ss);

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all); sizes the prompt path's
    /// lend cap.
    static void set_pinned_share(double share);

    /// The chunk size from which every expert streams (below it only the routed experts are staged and the peer card
    /// computes the rows of its own); before init, which sizes the buffers by it.  Default 1024.
    static void set_stream_all_min(int64_t tokens);

    /// The streamed ring's depth for the layouts made from now on (`relayout`, `bytes_needed`, `ring_slots_for`); at
    /// most the one `init` laid out.  A shallower ring takes fewer of the lent cache slots.
    static void set_ring_depth(int slots);
    /// The ring depth of the current layout.
    int ring_depth() const;
    static double pinned_share();
    /// The pack's prompt experts run on the FP4 tensor cores (NVFP4 gate/up and down in every layer).
    static bool fp4_experts();

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// Lays every buffer of a chunk of `chunk` tokens out over a region of exactly `bytes_needed` bytes without
    /// touching device memory (the pointers are never dereferenced) and reports the bytes the layout took; false when
    /// a buffer, or one of the sets that share the scratch region, does not fit.  `with_source`: the layout streams
    /// experts.  For tests of the sizing.
    static bool layout_fits(const core::ModelGeometry& g, core::SessionState& ss, int64_t chunk, bool with_source,
                            uint64_t* used, uint64_t* capacity);

    /// The streamed ring's slot count for a chunk of `chunk` tokens.
    static int64_t ring_slots_for(int64_t chunk);

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    /// For the next `run` (which must start at `run_pos0`): once its first chunk's n-gram rows have landed, the rows of
    /// `tokens[from, to)` (the prompt's later parts) are read into the row cache on a thread, so the runs that read
    /// them find the rows in memory.  Needs `from >= 2`; `to` is cut to what the cache holds.  Cleared by that run.
    void set_ple_warm(const int64_t* tokens, int64_t run_pos0, int64_t from, int64_t to);

    const PrefillStats& stats() const { return stats_; }

    /// Starts counting, per (layer, expert), the routed entries of the prompt rows this stage and the stages after it
    /// read, from zero.  Call with the stream idle (between requests).
    void route_hist_reset();
    /// Adds the counts since `route_hist_reset` (n_layers x n_expert, each stage fills its own layers) into `out`
    /// and the prompt tokens they cover into `tokens`, and stops counting.  False when counting was not started.
    bool route_hist_take(std::vector<uint32_t>& out, int64_t& tokens);

    /// multi-GPU: the experts the peer GPU holds are computed THERE for every prompt chunk (up to `cap_rows` routed
    /// rows per layer).  Allocates the peer's buffers for chunks of up to init's `chunk` tokens: the routed-only
    /// chunks, and the streamed ones up to a fixed size on the FP4 experts (the same bits as this GPU's), with the
    /// activations and the rows crossing card to card when the pair has peer access and through host memory when it
    /// does not.  `peer_ram_rows` (--peer-ram-rows) goes further: for the streamed chunks the partner also reads the
    /// blobs of the experts NEITHER card holds - the RAM tier's - over its own PCIe link and computes their rows, so
    /// this card's link carries only the rows back.  Off leaves them here, as before.  The rows of a streamed chunk
    /// come back as BF16 bits, half the bytes of the FP32 ones (one rounding in transit, and BF16's exponent is
    /// FP32's); `peer_rows_fp32` (--peer-rows-fp32) keeps the FP32 wire, for the A/B and the quality gate.  A streamed
    /// chunk's activations go out as the FP4 planes this card rounded for its own experts - the same
    /// `fp4::quantize_tokens` on the same activations, so the partner's rows are the bits it would have made and it
    /// quantizes nothing, 1 444 B a token instead of 10 240; `peer_acts_fp32` (--peer-acts-fp32) keeps the FP32 wire
    /// and the partner's own quantize, for the same A/B.
    bool set_peer(core::PeerExperts* peer, int64_t cap_rows, bool streamed_share, bool peer_ram_rows,
                  bool peer_rows_fp32, bool peer_acts_fp32, std::string& err);
    /// The partner card's prompt-attention span (`--exp tp_prompt_attn`, TP2PP2-NOTES RUN 25): what the partner holds
    /// resident on its own card to run its half of the prompt QSA attention - its KV span and indexer pooled table for
    /// every QSA layer of the stage, the wire's staging, and its own attention scratch.  `peer_bytes` adds it, so the
    /// partner's expert tier is sized to leave the room free (`pair_prompt_bytes` / `stage_room`).  Default off: with
    /// the gate closed the partner reserves nothing and its tier is byte-identical to before the slice existed.
    struct PeerQsaSpan {
        const core::ModelGeometry* g = nullptr;
        int64_t max_cells = 0;    ///< the sequence capacity the span must cover (the session's, not a chunk's)
        int64_t lb = 0, le = 0;   ///< the stage's layer range, [lb, le)
        core::QsaHeadShard heads; ///< the PARTNER's shard: which query heads and KV heads it runs
        bool on = false;
        /// Bytes ONE row of `attn_q`, and of `attn_k`/`attn_v`, occupies in the pack's native GGUF blocks, as the
        /// caller read them from the `WeightRef`s.  The RUN 25 ruling put the partner's projections RESIDENT on the
        /// partner's card (boot-time cost, zero per-layer link traffic; P2P weight reads are rejected because they
        /// would add unpriced per-chunk bytes and invalidate the ledger the gates grade against).  This sizer sees
        /// geometry only, and the block geometry is the pack's, so the figures come from the caller: zero means the
        /// pack was never consulted, and `bytes` prices NO weights rather than inventing a quantization table.
        int64_t q_row_bytes = 0, kv_row_bytes = 0;
        /// The packed bytes ONE QSA layer's `attn_output` columns occupy for this shard, as the caller read them
        /// (`qsa_weight_cols_bytes` over `qsa_out_cols`, the same figure `qsa_pair_bytes` sums per layer).  The
        /// columns are PACKED rather than a pointer because a column slice of a blocked weight is not a slice:
        /// `tp2_shard_weights.hpp`.  `qsa_pair_prepare` ALLOCATES these on the card that runs the shard, so they
        /// are priced here but allocated there - and must still be funded, or the tier leaves no room for a copy
        /// the partner's o_proj cannot run without.  Zero: the pack was never consulted, and nothing is priced.
        int64_t o_col_bytes = 0;
        /// The pack's GGUF types, as the caller read them from the `WeightRef`s (`attn_q`, `attn_k`/`attn_v`,
        /// `attn_output`).  The walk's row bytes say HOW BIG a row is; the product needs to know how to READ it -
        /// `Gemm::native` takes the type beside the pointer, and `verify.cpp`'s decode shard carries
        /// `wo->native_type` for exactly this reason.  Negative means unset, and the arm refuses to project
        /// rather than assume a quantization the pack never stated.
        int q_type = -1, kv_type = -1, o_type = -1;
        /// Why `make_peer_qsa_span` refused, when it did: which tensor of which layer, and what the pack did not
        /// say about it.  A refusal that only says "no figure" costs a boot to diagnose; this one does not.
        std::string why;
        /// How many allocations the last `bytes()` call walked.  Diagnostics for the sizing test, which pins it:
        /// the byte figures alone do not catch a tensor taken once where it should be taken once a QSA layer.
        mutable int m_takes_ = 0;
        int last_takes() const { return m_takes_; }
        /// QSA layers of this stage: `qsa_interval` over [lb, le).
        int64_t n_qsa_layers() const;
        /// How many resident sets of the wire's staging.  Two: the pipelined gate puts layer l+1's inbound on the
        /// link under layer l's compute, which one set cannot do (the lead would wait on the partner's read of
        /// the previous layer).  The per-layer traffic the ledger prices is ONE set; this is resident bytes.
        static constexpr int kWireSets = 2;
        /// The bytes this spans, split so the boot line can name each part.  Zero with `on` false or a geometry it
        /// cannot price.  `T`: the widest chunk the partner can be given, which sizes the wire and the scratch.
        /// `wire` reports the RESIDENT bytes (`kWireSets` sets); one set is the per-layer traffic figure.
        /// `weights` is the resident shard copies of the projections and the q/k norms, `gemm` the partner's own
        /// dequant scratch and cuBLAS workspace.  Any of them may be null.
        /// `o_cols` is the packed `attn_output` columns, priced here and allocated by `qsa_pair_prepare`.
        void bytes(int64_t T, uint64_t* span, uint64_t* wire, uint64_t* scratch, uint64_t* weights,
                   uint64_t* gemm, uint64_t* o_cols, uint64_t* total) const;
        /// One set of the wire: the bytes a QSA layer actually moves across the pair's link, which is what the
        /// ledger prices and what the +10/+20 gates are graded against.
        uint64_t wire_one_set(int64_t T) const;

        /// Is this span covered by what the peer tier was sized to leave free (`reserved`, the pinned symmetric
        /// cut)?  False fills `why` with BOTH the split the caller declared and the split the reserve priced, so
        /// a refusal tells the operator what to pin rather than only that something did not fit.  Pure: the
        /// figures are the same arithmetic `bytes` reports, so the rule is testable without a device.
        bool fits_within(const PeerQsaSpan& reserved, int64_t T, std::string& why) const;
    };

    /// The span the partner must be given, with the pack-derived figures (row bytes, GGUF types, the packed
    /// `attn_output` columns) read from `wt` rather than assumed - so the tier's reserve and the arm's allocation
    /// come from one place.  `lead_heads` is what the LEAD keeps; the partner takes the rest, and the ruled
    /// reserve prices the pinned symmetric cut.  `on` comes back false when the pack cannot answer, which reserves
    /// nothing rather than pricing a span half of whose bytes were guessed.
    static PeerQsaSpan make_peer_qsa_span(const core::ModelGeometry& g, const core::WeightTable& wt,
                                          int64_t max_cells, int64_t lb, int64_t le, int64_t lead_heads, bool on);

    /// The bytes `set_peer` takes on the peer card for chunks of up to `chunk` tokens (0: no peer share); the peer tier
    /// leaves them free when it sizes itself.  `qsa` adds the prompt-attention span above.
    static uint64_t peer_bytes(int64_t chunk, bool streamed_share, bool peer_ram_rows, bool peer_rows_fp32,
                               bool peer_acts_fp32, const PeerQsaSpan* qsa = nullptr);
    /// What the prompt-attention slice takes on the LEAD's own card, which `peer_bytes` does not cover: the pair's
    /// staging on this side (`Tp2Pair::open` allocates a `chunk x n_embd` FP32 buffer on EACH card, and the peer
    /// tier only books the partner's) and the lead's packed `attn_output` columns, which `qsa_pair_prepare`
    /// allocates there because the lead is the card that runs that shard.  Priced at the pinned symmetric cut, as
    /// the partner's span is, and `set_peer` refuses a split that needs more on EITHER card by name: the lead's
    /// columns grow as the partner's shrink, so the partner fitting does not prove the lead fits.  Zero with the
    /// gate closed, which is what leaves a boot that never asked for the slice untouched.
    static uint64_t lead_qsa_bytes(const core::ModelGeometry& g, const core::WeightTable& wt, int64_t chunk,
                                   int64_t lead_heads);
    /// --peer-exclusive: experts only `peer` holds have no host copy; the prompt path copies them from its memory
    /// through host staging into the ring.  Independent of set_peer.
    void set_ring_peer(core::PeerExperts* peer);
    /// A stage on another card than the main one: an expert with no pinned host copy that the cache on `device` holds
    /// (`res`: n_layers x n_expert slot or -1, read at every chunk) is copied from that card through host staging into
    /// the ring.  Null `res`: none.
    void set_ring_remotes(std::vector<std::tuple<const int32_t*, const core::ExpertCache*, int>> remotes);
    /// Off: the peer card's share (set_peer) is skipped, every expert of a chunk stays on this card - while the peer
    /// card runs a stage of the same prompt.
    void set_peer_share(bool on);
    /// The longest routed-only chunk whose rows of its experts the peer computes (0: no peer share).
    int64_t peer_routed_rows_max() const;

    /// Hand the partner the LEAD's QSA states, one per QSA layer of this stage, and the card they live on, so a
    /// prompt can catch its span up from them.  Borrowed pointers, as `set_ring_remotes` borrows the experts
    /// another card's cache holds: this Prefill neither owns them nor frees them, and the lead's session must
    /// outlive every prompt that imports from it.  Drops the span's proof: nothing has been copied by handing
    /// them over.
    void set_peer_qsa_lead(std::vector<const core::QsaState*> lead, int lead_dev);
    /// Forget the proof that the partner's pools hold the lead's cells, so the next prompt re-imports the whole
    /// span rather than attending over a context that has moved underneath it.  Call it wherever the lead's KV
    /// or indexer state can change without the partner having appended it itself: a session swap, a prefix
    /// copied in or back, a cleared prompt session, a park or resume, the end of a decode window.  Cheap and
    /// idempotent - a no-op with no peer arm or an already-empty span - so the sites err toward calling it.
    void invalidate_peer_qsa_span(const char* why);

    /// The CPU expert pool (idle while a prompt is read). A chunk below stream_all_min() tokens hands it the
    /// non-resident experts with a host copy that few of its tokens route to, instead of streaming them; their rows
    /// go to the tail of Dm unweighted, like the peer's, and the combine weights them. The share is set by
    /// set_cpu_share(). Only for a pool no other thread runs meanwhile. Null: every expert on the GPU.
    void set_cpu_pool(kernels::cpu::ExpertPool* pool);
    /// The share, 0 to 1 (default 0.6), of the streamed experts the CPU computes; 0 leaves them all to the GPU. Fixed
    /// for the run, so the output does not depend on timing.
    void set_cpu_share(double share);

    /// Called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;

    /// Called on the host before each layer of a chunk with the layers done so far, the layer count, the chunk's
    /// length and first position.  No device work is added; the MoE layers' routing readback keeps the host in step
    /// with the GPU.  Not called under a layer split.
    std::function<void(int64_t layers_done, int64_t n_layers, int64_t T, int64_t pos0)> on_layer;

    /// Layer split: called by every stage when it has read a chunk, with the position reached, while its own state
    /// is still at that chunk's end (its stream synchronized; the last stage calls it just before `on_chunk`).  An
    /// earlier stage is a chunk or more ahead of the last one by the time `on_chunk` runs, so this is where a
    /// mid-prompt checkpoint takes each stage's part.  Runs on that stage's thread, with its device current.
    std::function<bool(int64_t done, std::string& err)> on_stage_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;
    /// Layer split, a first stage with `set_keep_last`: checked at each chunk's end (the position reached) before the
    /// chunk is handed on; true keeps that chunk as the last one and ends the run there.
    std::function<bool(int64_t done)> keep_here;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// LAYER SPLIT (multi-GPU): this prompt path runs layers [layer_begin, layer_end) (-1: to the last) on the
    /// device `init` runs on.  A stage that does not start at layer 0 reads each chunk's residual rows from the
    /// previous stage instead of embedding the tokens; a stage that does not end at the last layer copies its rows
    /// to pinned host buffers (two, allocated by `init`) and runs `next` on them - on a thread, so the next stage
    /// reads chunk c while this one reads chunk c + 1.  `on_chunk` belongs on the last stage.  Set before `init`, or
    /// between runs (a successor set after `init` needs `reserve_hand_off` first).
    void set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next);
    /// The two pinned hand-off buffers for chunks of up to `rows` tokens, for a successor set after `init`.
    bool reserve_hand_off(int64_t rows, std::string& err);
    /// Layer split: the first stage keeps a run's last chunk - it reads that chunk's layers and neither hands it on nor
    /// reports it - for its card to read the later layers too (`continue_in_place`) once the later stages are done.
    void set_keep_last(bool on) { keep_last_ = on; }
    /// The next run reads the rows this stage's own buffers hold (the chunk it kept), not tokens or a hand-off.
    void continue_in_place() { rows_in_place_ = true; hand_in_ = nullptr; }
    /// A later stage: the hand-off buffer it read its last chunk from (pinned host, the stage before's), or null.
    const float* hand_in() const { return hand_in_; }
    /// A stage that hands chunks on: its hand-off buffer `b` (0 or 1, pinned host, null before any), and the rows of
    /// the full residual width each holds.  A run writes a chunk's rows at the start of one of them.
    float* hand_buffer(int b) const;
    int64_t hand_rows() const;


private:
    // Stage-1 pipeline: intermediate stages return after handing their chunk to
    // the direct successor. The public run() drains the chain once at prompt end.
    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    bool drain_pipeline(std::string& err);

    kernels::cpu::ExpertPool* cpu_pool_ = nullptr;
    int64_t stage_lb_ = 0, stage_le_ = -1;
    int64_t head_chunk_ = 0;
    Prefill* next_ = nullptr;
    bool single_chunk_ = false;         ///< a later stage: the prompt is one chunk (set by the stage before)
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)
    bool rows_in_place_ = false;        ///< the next run's rows are in this stage's own buffers (continue_in_place)
    bool keep_last_ = false;            ///< set_keep_last

    std::string next_err_;
    std::future<bool> next_run_;
    int hand_buf_ = 0;

    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace basalt::prefill

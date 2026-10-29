// include/basalt/core/tp2_shard_weights.hpp - the weight VIEWS one card of a pair reads when it runs half a
// layer's heads, and the bytes the packed ones cost.
//
// A head shard touches three tensors, in two different directions:
//
//   * `attn_q` / `attn_k` / `attn_v` are cut on their OUTPUT axis (the head axis is `ne1`).  A card's rows are
//     CONTIGUOUS in every plane, so its view is three pointer offsets and no copy at all.
//   * `attn_output` is cut on its INPUT axis (`ne0` = n_head * head_dim).  Here a row of the weight is the full
//     6144-column width and a card owns a span of it, so the view has to be PACKED: each plane's span copied out
//     of each row, laid end to end, so the row pitch the GEMV derives from `ne0` matches what it reads.  A
//     pointer offset into the original would make row i land at row i of a narrower matrix - a plausible-looking
//     product from the wrong weights, which is the failure this file exists to make impossible.
//
// THE PLANES ARE NOT RE-DERIVED, ever. `WeightRef` records `codes_bytes`, `scales_bytes` and `offset_bytes` from
// the index (weights.hpp: the one time a caller recomputed them it assumed 4 B scales and was wrong for every
// fp16-scale tensor).  Everything here divides those recorded figures by `ne1` / by the block count, so the
// arithmetic follows what was loaded.
//
// THE CUT IS ON WHOLE QUANTIZATION BLOCKS. A block is `group_elems` values of the input axis (and 256 values in
// the native GGUF image - gdn_head_split.hpp's `kBlockCols`, the same families), with its scales beside it.  A
// span that starts or ends mid-block has no bytes that could represent it, so the views refuse rather than read
// a neighbouring block's scales.  head_dim 256 makes every head-aligned span legal; the check is what makes that
// a fact rather than an assumption.
//
// PURE: `native_row_bytes` comes in as a number, so nothing here needs a CUDA link and
// tests/core/tp2_shard_weights_test.cpp is the whole oracle for the byte layout.
#pragma once

#include "basalt/core/gdn_head_split.hpp"
#include "basalt/core/weights.hpp"

#include <cstdint>
#include <cstring>
#include <string>

namespace basalt::core {

/// The native GGUF image's block width in VALUES, for a given GGML type.  The native image is packed in blocks
/// along the input axis - which is what makes a column cut possible at all - and the width is a property of the
/// TYPE, not a constant: 32 values for the legacy forms (Q4_0, Q5_0, Q8_0, IQ4_NL, and Q2_0 as type 20), 64 for
/// Q2_0 as type 42, 256 for the K-quants and the IQ wide types.  This table MIRRORS the switch in
/// `native_mmvq_weight_bytes` (src/kernels/cuda/native_mmvq.cu); the two are checked against each other by
/// `qsa_pair_prepare` before anything is packed, so a type added on one side and not the other refuses rather
/// than cutting a weight at the wrong stride.  0 for a type the image cannot be cut with.
inline int64_t qsa_native_block_elems(int native_type) {
    switch (native_type) {
        case 2: case 6: case 7: case 8: case 20: return 32;
        case 42: return 64;
        case 11: case 12: case 13: case 14: case 23:
        case 16: case 17: case 18: case 21: case 22: case 29: return 256;
        default: return 0;
    }
}

/// Where one card's weight bytes are.  `codes`/`scales`/`offset` are the three canonical planes in the order the
/// loader lays them out, `native` the native image when the tensor has one; `ne0`/`ne1` are the VIEW's shape, which
/// is what the projection call must be told, and the plane byte counts match them.
struct QsaShardWeight {
    const uint8_t* codes = nullptr;
    const float* scales = nullptr;
    const float* offset = nullptr;   ///< null when the form has no offset plane
    const void* native = nullptr;    ///< null when the tensor has no native projection
    int64_t ne0 = 0, ne1 = 0;
    uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0, native_bytes = 0;
    bool ok = false;
};

/// One block's bytes in one plane, for one row: the recorded plane size divided by the rows and the blocks it
/// covers.  `plane_bytes` is the WHOLE plane as the loader sized it, never a row.
inline uint64_t qsa_plane_block_bytes(uint64_t plane_bytes, int64_t rows, int64_t blocks) {
    if (plane_bytes == 0 || rows <= 0 || blocks <= 0) return 0;
    const uint64_t per = (uint64_t) rows * (uint64_t) blocks;
    if (plane_bytes % per != 0) return 0;
    return plane_bytes / per;
}

/// One block of the native image, for one row: the native blocks are 256 values wide (not `group_elems`), which
/// is why the image needs its own figure.  0 when the tensor has no native image or the size does not divide.
inline uint64_t qsa_weight_native_block_bytes(const WeightRef& w, uint64_t native_row_bytes) {
    if (!w.native_data || native_row_bytes == 0 || w.ne1 <= 0) return 0;
    const int64_t be = qsa_native_block_elems(w.native_type);
    if (be <= 0 || w.ne0 % be != 0) return 0;   // a type whose image cannot be cut: refuse, never divide by it
    return qsa_plane_block_bytes(native_row_bytes, w.ne1, w.ne0 / be);
}

/// True when the whole card's span of the input axis is legal to cut: on this form's blocks, for both the
/// canonical planes and the native image.  `head_dim` times a whole number of heads is what the shard's
/// `qsa_out_cols()` produces, so for the artifact this is a tautology - and a tautology checked is cheaper than
/// a wrong product.
inline bool qsa_weight_cols_fits(const WeightRef& w, int64_t col0, int64_t cols, uint64_t native_row_bytes) {
    if (!w.quantized() || cols <= 0 || col0 < 0 || col0 + cols > w.ne0 || w.ne1 <= 0 || w.group_elems <= 0)
        return false;
    if (col0 % w.group_elems != 0 || cols % w.group_elems != 0) return false;
    const int64_t blocks = w.ne0 / w.group_elems;
    if (blocks <= 0) return false;
    if (qsa_plane_block_bytes(w.codes_bytes, w.ne1, blocks) == 0) return false;
    if (qsa_plane_block_bytes(w.scales_bytes, w.ne1, blocks) == 0) return false;
    if (w.offset_bytes && qsa_plane_block_bytes(w.offset_bytes, w.ne1, blocks) == 0) return false;
    if (w.has_offset != (w.offset_bytes != 0)) return false;
    if (w.native_data && qsa_weight_native_block_bytes(w, native_row_bytes) == 0) return false;
    if (w.native_data) {
        const int64_t be = qsa_native_block_elems(w.native_type);
        if (be <= 0 || col0 % be != 0 || cols % be != 0) return false;
    }
    return true;
}

/// The native image's bytes for a card's span, over `rows` rows: `native_row_bytes` is the WHOLE image.
inline uint64_t qsa_weight_native_cols_bytes(const WeightRef& w, int64_t cols, uint64_t native_row_bytes,
                                             int64_t rows) {
    const uint64_t block = qsa_weight_native_block_bytes(w, native_row_bytes);
    const int64_t be = qsa_native_block_elems(w.native_type);
    if (block == 0 || be <= 0 || rows <= 0 || cols <= 0 || cols % be != 0) return 0;
    return block * (uint64_t) (cols / be) * (uint64_t) rows;
}

/// What the split's block width says the WHOLE native image weighs, given one block's bytes: `block_bytes` per
/// block, `ne0 / block_elems` blocks per row, `ne1` rows.  This is the cross-check's side of the ledger - the
/// caller gets `block_bytes` from the kernel (`native_mmvq_weight_bytes` at one block and one row) and compares
/// this against the image the loader actually sized, so the two tables must agree.  0 when the type's width
/// cannot cut this tensor.
inline uint64_t qsa_weight_native_bytes_from_block(const WeightRef& w, uint64_t block_bytes) {
    const int64_t be = qsa_native_block_elems(w.native_type);
    if (be <= 0 || block_bytes == 0 || w.ne0 <= 0 || w.ne1 <= 0 || w.ne0 % be != 0) return 0;
    return block_bytes * (uint64_t) (w.ne0 / be) * (uint64_t) w.ne1;
}

/// The bytes one packed column view costs: the three canonical planes with their spans copied out of every row,
/// and the native image's span behind them.  The caller allocates one buffer of this size per view and passes it
/// to `qsa_pack_weight_cols`.
inline uint64_t qsa_weight_cols_bytes(const WeightRef& w, int64_t col0, int64_t cols, uint64_t native_row_bytes) {
    if (!qsa_weight_cols_fits(w, col0, cols, native_row_bytes)) return 0;
    const int64_t blocks = w.ne0 / w.group_elems;
    const int64_t take = cols / w.group_elems;
    uint64_t n = (qsa_plane_block_bytes(w.codes_bytes, w.ne1, blocks) +
                  qsa_plane_block_bytes(w.scales_bytes, w.ne1, blocks)) *
                 (uint64_t) take * (uint64_t) w.ne1;
    if (w.offset_bytes)
        n += qsa_plane_block_bytes(w.offset_bytes, w.ne1, blocks) * (uint64_t) take * (uint64_t) w.ne1;
    n += qsa_weight_native_cols_bytes(w, cols, native_row_bytes, w.ne1);
    return n;
}

/// The card's columns of a weight whose INPUT axis is the head axis, packed into `dst` (which must be
/// `qsa_weight_cols_bytes` long) in the loader's plane order, and described by the returned view.  `dst` may be
/// null, which only describes the view (a caller sizing its arena does that first).  Refuses - returns !ok - when
/// the span is not on blocks or a recorded plane does not divide into rows and blocks.
inline QsaShardWeight qsa_pack_weight_cols(const WeightRef& w, int64_t col0, int64_t cols, uint64_t native_row_bytes,
                                           void* dst) {
    QsaShardWeight v;
    if (!qsa_weight_cols_fits(w, col0, cols, native_row_bytes)) return v;
    const int64_t blocks = w.ne0 / w.group_elems;
    const int64_t take = cols / w.group_elems;
    const size_t codes_block = (size_t) qsa_plane_block_bytes(w.codes_bytes, w.ne1, blocks);
    const size_t scales_block = (size_t) qsa_plane_block_bytes(w.scales_bytes, w.ne1, blocks);
    const size_t offset_block = w.offset_bytes ? (size_t) qsa_plane_block_bytes(w.offset_bytes, w.ne1, blocks) : 0;
    const uint8_t* base = (const uint8_t*) w.data;
    auto* out = (uint8_t*) dst;
    const int64_t rows = w.ne1;
    v.codes_bytes = (uint64_t) codes_block * (uint64_t) take * (uint64_t) rows;
    v.scales_bytes = (uint64_t) scales_block * (uint64_t) take * (uint64_t) rows;
    v.offset_bytes = (uint64_t) offset_block * (uint64_t) take * (uint64_t) rows;
    v.native_bytes = qsa_weight_native_cols_bytes(w, cols, native_row_bytes, rows);
    // one run per plane, in VALUES: `gdn_gather_cols` takes a run in the axis's elements and divides by the
    // block width itself (the way the GDN split calls it), which is the one place a doubled division hides
    const std::vector<GdnRun> run{{col0, cols}};
    if (out) {
        gdn_gather_cols(base, rows, w.ne0, w.group_elems, codes_block, run, out);
        gdn_gather_cols(base + w.codes_bytes, rows, w.ne0, w.group_elems, scales_block, run, out + v.codes_bytes);
        if (offset_block)
            gdn_gather_cols(base + w.codes_bytes + w.scales_bytes, rows, w.ne0, w.group_elems, offset_block, run,
                            out + v.codes_bytes + v.scales_bytes);
        if (v.native_bytes)
            gdn_gather_cols((const uint8_t*) w.native_data, rows, w.ne0,
                            qsa_native_block_elems(w.native_type),
                            (size_t) qsa_weight_native_block_bytes(w, native_row_bytes),
                            {{col0, cols}},
                            out + v.codes_bytes + v.scales_bytes + v.offset_bytes);
    }
    v.codes = (const uint8_t*) dst;
    v.scales = (const float*) ((uint8_t*) dst + v.codes_bytes);
    v.offset = offset_block ? (const float*) ((uint8_t*) dst + v.codes_bytes + v.scales_bytes) : nullptr;
    if (w.native_data) {
        // the native image trails the canonical planes in the same buffer
        if (v.native_bytes == 0) return QsaShardWeight{};
        v.native = (const uint8_t*) dst + v.codes_bytes + v.scales_bytes + v.offset_bytes;
    }
    v.ne0 = cols;
    v.ne1 = rows;
    v.ok = true;
    return v;
}

/// The card's rows of a weight whose OUTPUT axis is the head axis (`attn_q`, `attn_k`, `attn_v`): three pointer
/// offsets into the loaded tensor, no copy, because rows are contiguous in every plane.  `rows` is
/// `qsa_q_rows()`/`qsa_kv_rows()`' `len` and `row0` its `start`, in ELEMENTS of this axis.
inline QsaShardWeight qsa_weight_rows(const WeightRef& w, int64_t row0, int64_t rows, uint64_t native_row_bytes) {
    QsaShardWeight v;
    if (row0 < 0 || rows <= 0 || row0 + rows > w.ne1 || w.ne1 <= 0) return v;
    if (!w.quantized() && !w.native_data) return v;
    if (w.quantized() && (w.codes_bytes == 0 || w.codes_bytes % (uint64_t) w.ne1 != 0 ||
                          w.scales_bytes % (uint64_t) w.ne1 != 0 ||
                          (w.offset_bytes && w.offset_bytes % (uint64_t) w.ne1 != 0)))
        return v;
    if (w.has_offset != (w.offset_bytes != 0)) return v;
    const uint8_t* base = (const uint8_t*) w.data;
    v.codes = base + (uint64_t) row0 * (w.codes_bytes / (uint64_t) w.ne1);
    v.scales = (const float*) (base + w.codes_bytes + (uint64_t) row0 * (w.scales_bytes / (uint64_t) w.ne1));
    v.offset = w.offset_bytes
                   ? (const float*) (base + w.codes_bytes + w.scales_bytes +
                                      (uint64_t) row0 * (w.offset_bytes / (uint64_t) w.ne1))
                   : nullptr;
    v.codes_bytes = (uint64_t) rows * (w.codes_bytes / (uint64_t) w.ne1);
    v.scales_bytes = (uint64_t) rows * (w.scales_bytes / (uint64_t) w.ne1);
    v.offset_bytes = w.offset_bytes ? (uint64_t) rows * (w.offset_bytes / (uint64_t) w.ne1) : 0;
    if (w.native_data) {
        // A tensor that HAS a native image must keep using it: dropping to the canonical planes for the shard
        // would run a different decoder than the whole layer ran, and the two are not bitwise equal.
        if (native_row_bytes == 0 || native_row_bytes % (uint64_t) w.ne1 != 0) return QsaShardWeight{};
        v.native = (const uint8_t*) w.native_data + (uint64_t) row0 * (native_row_bytes / (uint64_t) w.ne1);
        v.native_bytes = (uint64_t) rows * (native_row_bytes / (uint64_t) w.ne1);
    }
    v.ne0 = w.ne0;
    v.ne1 = rows;
    v.ok = true;
    return v;
}

/// The packed bytes a stage's QSA layers need for their `attn_output` column views, over `n_qsa_layers` of the
/// same form: what the caller allocates once and hands to `qsa_pack_weight_cols` per layer.
inline uint64_t qsa_weight_cols_arena_bytes(const WeightRef& w, int64_t col0, int64_t cols,
                                            uint64_t native_row_bytes, int64_t n_layers) {
    return qsa_weight_cols_bytes(w, col0, cols, native_row_bytes) * (uint64_t) (n_layers > 0 ? n_layers : 0);
}

}  // namespace basalt::core

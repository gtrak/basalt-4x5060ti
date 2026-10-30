// tests/core/tp2_shard_weights_test.cpp - the bytes a pair's card reads when it runs half a layer's heads.
//
// The claim under test is narrow and absolute: a shard's weight view reads EXACTLY the bytes the whole layer
// would have read for that head range - same codes, same scales, same offsets, same native image - and refuses
// when the span cannot be expressed on the form's quantization blocks.  A view that is one block off is not a
// fault; it is a plausible product from the wrong weights, which is why this is a byte test and not a size test.
//
// Synthetic tensors throughout: the layout is what matters, and a WeightRef carries the plane sizes the loader
// recorded, so the test writes those and the header must follow them (weights.hpp's own rule).
#include "basalt/core/qsa_head_split.hpp"
#include "basalt/core/tp2_shard_weights.hpp"
#include "basalt/artifact/dequant.hpp"   // the scalar Q6_K reference the cut is graded against

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace basalt::core;

namespace {
int g_fail = 0;
#define CHECK(...)                                                                                         \
    do {                                                                                                   \
        if (!(__VA_ARGS__)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__); ++g_fail; } \
    } while (0)

/// A quantized tensor laid out the way the loader lays one out: codes | scales | offsets, each row `ne0` values
/// wide in `blocks` blocks of `group` values.
struct FakeWeight {
    int64_t ne0 = 1024, ne1 = 8, group = 128;
    int bits = 4;
    bool has_offset = true;
    bool with_native = true;
    int native_type = 12;   ///< a 256-value-block K-quant by default; 8 is a 32-value legacy form
    static constexpr int64_t kQsaFakeBlockFallback = 256;   ///< only for the fake's byte-fill with a bad type
    uint64_t native_row = 512;   ///< ONE row of the native image; the API takes the whole image's bytes
    uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
    std::vector<uint8_t> canon, native;
    WeightRef w;

    /// The native image's bytes for the WHOLE tensor, which is the figure `tp2_shard_weights.hpp` takes.
    uint64_t native_all() const { return with_native ? (uint64_t) native.size() : 0; }

    void build() {
        const int64_t blocks = ne0 / group;
        w.ne0 = ne0;
        w.ne1 = ne1;
        w.elements = ne0 * ne1;
        w.kind = WeightKind::Verbatim;
        w.code_bits = bits;
        w.code_bias = -8;
        w.group_elems = (int) group;
        w.has_offset = has_offset;
        w.codes_bytes = (uint64_t) ne1 * (ne0 * bits / 8);
        w.scales_bytes = (uint64_t) ne1 * blocks * 4;
        w.offset_bytes = has_offset ? (uint64_t) ne1 * blocks * 4 : 0;
        w.bytes = w.codes_bytes + w.scales_bytes + w.offset_bytes;
        codes_bytes = w.codes_bytes;
        scales_bytes = w.scales_bytes;
        offset_bytes = w.offset_bytes;
        canon.resize((size_t) w.bytes);
        // every byte identifiable by (plane, row, block, offset in block), so a mis-gathered byte is visible
        size_t i = 0;
        for (int64_t row = 0; row < ne1; ++row)
            for (int64_t blk = 0; blk < blocks; ++blk)
                for (int64_t k = 0; k < ne0 * bits / 8 / blocks; ++k)
                    canon[i++] = (uint8_t) (0x10 * 0 + row * 17 + blk * 5 + k);
        for (int plane = 1; plane <= (has_offset ? 2 : 1); ++plane) {
            const uint64_t off = plane == 1 ? w.codes_bytes : w.codes_bytes + w.scales_bytes;
            const uint64_t sz = plane == 1 ? w.scales_bytes : w.offset_bytes;
            uint8_t* p = canon.data() + off;
            size_t q = 0;
            for (int64_t row = 0; row < ne1; ++row)
                for (int64_t blk = 0; blk < blocks; ++blk)
                    for (int64_t k = 0; k < 4; ++k) p[q++] = (uint8_t) (0x80 | plane << 4 | (row * 9 + blk * 3 + k));
            (void) sz;
        }
        w.data = canon.data();
        if (with_native) {
            native.resize((size_t) native_row * ne1);
            size_t n = 0;
            const int64_t nblocks = ne0 / (qsa_native_block_elems(w.native_type) > 0
                                                       ? qsa_native_block_elems(w.native_type)
                                                       : kQsaFakeBlockFallback);
            for (int64_t row = 0; row < ne1; ++row)
                for (int64_t blk = 0; blk < nblocks; ++blk)
                    for (int64_t k = 0; k < (int64_t) (native_row / nblocks); ++k)
                        native[n++] = (uint8_t) (0x40 ^ (row * 13 + blk * 7 + k));
            w.native_data = native.data();
            w.native_type = native_type;
        }
    }
    int64_t blocks() const { return ne0 / group; }
};

/// What a correct gather of blocks [b0, b0+nb) of one plane looks like, computed here without the header.
std::vector<uint8_t> expect_rows(const std::vector<uint8_t>& plane_bytes, int64_t rows, int64_t blocks,
                                 int64_t b0, int64_t nb) {
    const size_t block_bytes = plane_bytes.size() / (size_t)(rows * blocks);
    std::vector<uint8_t> out;
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t b = b0; b < b0 + nb; ++b) {
            const size_t from = (size_t)(row * blocks + b) * block_bytes;
            out.insert(out.end(), plane_bytes.begin() + from, plane_bytes.begin() + from + block_bytes);
        }
    return out;
}

std::vector<uint8_t> slice(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    return std::vector<uint8_t>(b, b + n);
}

void the_column_view_is_the_wholes_bytes() {
    FakeWeight f;
    f.build();
    const int64_t half = f.ne0 / 2;
    std::vector<uint8_t> dst(qsa_weight_cols_bytes(f.w, 0, half, f.native_all()));
    CHECK(!dst.empty());
    const QsaShardWeight v = qsa_pack_weight_cols(f.w, 0, half, f.native_all(), dst.data());
    CHECK(v.ok);
    CHECK(v.ne0 == half && v.ne1 == f.ne1);

    const int64_t blocks = f.blocks(), nb = (half / f.group);
    CHECK(slice(v.codes, v.codes_bytes) ==
          expect_rows(std::vector<uint8_t>(f.canon.begin(), f.canon.begin() + (ptrdiff_t) f.codes_bytes), f.ne1,
                      blocks, 0, nb));
    CHECK(slice(v.scales, v.scales_bytes) ==
          expect_rows(std::vector<uint8_t>(f.canon.begin() + (ptrdiff_t) f.codes_bytes,
                                           f.canon.begin() + (ptrdiff_t) (f.codes_bytes + f.scales_bytes)),
                      f.ne1, blocks, 0, nb));
    CHECK(slice(v.offset, v.offset_bytes) ==
          expect_rows(std::vector<uint8_t>(f.canon.begin() + (ptrdiff_t) (f.codes_bytes + f.scales_bytes),
                                           f.canon.end()),
                      f.ne1, blocks, 0, nb));
    CHECK(v.native != nullptr);
    CHECK(slice(v.native, v.native_bytes) ==
          expect_rows(f.native, f.ne1, f.ne0 / qsa_native_block_elems(f.w.native_type), 0,
              (half / qsa_native_block_elems(f.w.native_type))));
    // the sizes the caller allocated for are exactly the sizes the view describes
    CHECK(v.codes_bytes + v.scales_bytes + v.offset_bytes + v.native_bytes == dst.size());
}

void the_two_halves_are_the_whole_row() {
    FakeWeight f;
    f.build();
    const int64_t half = f.ne0 / 2;
    std::vector<uint8_t> d0(qsa_weight_cols_bytes(f.w, 0, half, f.native_all())),
        d1(qsa_weight_cols_bytes(f.w, half, half, f.native_all()));
    const QsaShardWeight a = qsa_pack_weight_cols(f.w, 0, half, f.native_all(), d0.data());
    const QsaShardWeight b = qsa_pack_weight_cols(f.w, half, half, f.native_all(), d1.data());
    CHECK(a.ok && b.ok);
    // per row, card 0's columns then card 1's are the whole row, for every plane: the pair is not missing or
    // duplicating a block, which is the o_proj's correctness condition
    struct P {
        const void* x;
        const void* y;
        uint64_t n;
    };
    const uint8_t* whole_codes = (const uint8_t*) f.w.data;
    const uint8_t* whole_scales = whole_codes + f.codes_bytes;
    const uint8_t* whole_offset = whole_scales + f.scales_bytes;
    const P planes[4] = {{a.codes, b.codes, a.codes_bytes},
                         {a.scales, b.scales, a.scales_bytes},
                         {a.offset, b.offset, a.offset_bytes},
                         {a.native, b.native, a.native_bytes}};
    const uint8_t* wholes[4] = {whole_codes, whole_scales, whole_offset, (const uint8_t*) f.native.data()};
    const uint64_t totals[4] = {f.codes_bytes, f.scales_bytes, f.offset_bytes, f.native_all()};
    for (int p = 0; p < 4; ++p) {
        if (!planes[p].x) continue;
        CHECK(2 * planes[p].n == totals[p]);
        const uint8_t* x = (const uint8_t*) planes[p].x;
        const uint8_t* y = (const uint8_t*) planes[p].y;
        const uint8_t* w = wholes[p];
        const size_t row = (size_t) planes[p].n / (size_t) f.ne1;
        bool same = true;
        for (int64_t r = 0; r < f.ne1 && same; ++r) {
            if (std::memcmp(x + (size_t) r * row, w + (size_t) r * 2 * row, row) != 0) same = false;
            if (std::memcmp(y + (size_t) r * row, w + (size_t) r * 2 * row + row, row) != 0) same = false;
        }
        CHECK(same);
    }
}

void a_whole_view_is_the_tensor_itself() {
    FakeWeight f;
    f.build();
    std::vector<uint8_t> dst(qsa_weight_cols_bytes(f.w, 0, f.ne0, f.native_all()));
    const QsaShardWeight v = qsa_pack_weight_cols(f.w, 0, f.ne0, f.native_all(), dst.data());
    CHECK(v.ok);
    CHECK(v.codes_bytes == f.codes_bytes && v.scales_bytes == f.scales_bytes && v.offset_bytes == f.offset_bytes);
    CHECK(std::memcmp(v.codes, f.canon.data(), (size_t) f.codes_bytes) == 0);          // codes are first, in place
    CHECK(std::memcmp((const uint8_t*) v.scales, f.canon.data() + f.codes_bytes, (size_t) f.scales_bytes) == 0);
    CHECK(std::memcmp((const uint8_t*) v.offset, f.canon.data() + f.codes_bytes + f.scales_bytes,
                      (size_t) f.offset_bytes) == 0);
    CHECK(v.native_bytes == f.native_all());
    CHECK(std::memcmp(v.native, f.native.data(), (size_t) v.native_bytes) == 0);
}

void the_row_view_is_a_pointer() {
    FakeWeight f;
    f.build();
    const int64_t row0 = 2, rows = 3;
    const QsaShardWeight v = qsa_weight_rows(f.w, row0, rows, f.native_all());
    CHECK(v.ok);
    CHECK(v.ne0 == f.ne0 && v.ne1 == rows);
    const uint8_t* base = f.canon.data();
    CHECK(v.codes == base + (uint64_t) row0 * (f.codes_bytes / (uint64_t) f.ne1));
    CHECK(v.scales == (const float*) (base + f.codes_bytes + row0 * (f.scales_bytes / (uint64_t) f.ne1)));
    CHECK(v.offset == (const float*) (base + f.codes_bytes + f.scales_bytes +
                                      row0 * (f.offset_bytes / (uint64_t) f.ne1)));
    CHECK(v.codes_bytes == (uint64_t) rows * (f.codes_bytes / (uint64_t) f.ne1));
    CHECK(v.native == (const uint8_t*) f.native.data() + row0 * (f.native_all() / (uint64_t) f.ne1));
    // and it reads the right values: the whole tensor's rows [row0, row0+rows) of the codes plane
    CHECK(std::memcmp(v.codes, base + (size_t) row0 * (f.codes_bytes / (size_t) f.ne1),
                      (size_t) v.codes_bytes) == 0);
    // no copy happened: the view points inside the loaded tensor
    CHECK(v.codes > base && v.codes < base + f.canon.size());
}

/// The native image is cut on ITS block width, which is a property of the GGML type: 32 values for the legacy
/// forms, 64 for one Q2_0, 256 for the K-quants and the wide IQ types.  A 256-wide assumption would let a
/// legacy-type span through that the kernel cannot decode from a partial block.
void the_native_block_width_follows_the_type() {
    CHECK(qsa_native_block_elems(8) == 32);    // Q8_0
    CHECK(qsa_native_block_elems(2) == 32);    // Q4_0
    CHECK(qsa_native_block_elems(42) == 64);   // Q2_0, the 64-wide one
    CHECK(qsa_native_block_elems(12) == 256);  // Q4_K
    CHECK(qsa_native_block_elems(23) == 256);  // IQ4_XS
    CHECK(qsa_native_block_elems(99) == 0);    // not a native type at all
    CHECK(qsa_native_block_elems(-1) == 0);

    FakeWeight leg;
    leg.ne0 = 1024;
    leg.group = 32;        // so a 96-column span is legal on BOTH axes
    leg.native_type = 8;   // 32-value native blocks
    leg.native_row = 1024 / 32 * 34;
    leg.build();
    CHECK(qsa_weight_cols_fits(leg.w, 0, leg.ne0 / 2, leg.native_all()));
    const uint64_t want = qsa_weight_cols_bytes(leg.w, 96, 96, leg.native_all());
    CHECK(want > 0);
    std::vector<uint8_t> d((size_t) want);
    const QsaShardWeight v = qsa_pack_weight_cols(leg.w, 96, 96, leg.native_all(), d.data());
    CHECK(v.ok);   // 96 is on the 32-value native blocks and on this tensor's canonical group
    CHECK(v.native_bytes == (uint64_t) leg.ne1 * (96 / 32) * 34);
    FakeWeight bad;
    bad.build();
    bad.w.native_type = 99;   // set after the build: the fake's own layout follows the type
    CHECK(bad.w.native_data != nullptr);
    CHECK(!qsa_weight_cols_fits(bad.w, 0, bad.ne0 / 2, bad.native_all()));
    // a ROW view is unaffected: cutting along the output axis never splits a block, whatever its width
    CHECK(qsa_weight_rows(bad.w, 0, 2, bad.native_all()).ok);

    // a 256-wide image cannot be cut at 32, even where the canonical group allows it
    FakeWeight kq;
    kq.group = 32;
    kq.native_type = 12;
    kq.build();
    CHECK(qsa_weight_cols_fits(kq.w, 0, 32, kq.native_all()) == false);
    CHECK(qsa_weight_cols_fits(kq.w, 0, 256, kq.native_all()));

    // The cross-check's own figure: one block's bytes, times the blocks in a row, times the ROWS.  A figure that
    // leaves `ne1` out accepts a one-row image for an eight-row tensor - which is the failure this comparison
    // exists to catch, so it is pinned from both sides.
    CHECK(qsa_weight_native_bytes_from_block(leg.w, 34) == leg.native_all());   // Q8_0: 34 B per 32 values
    CHECK(qsa_weight_native_bytes_from_block(leg.w, 34) == 34ull * (leg.ne0 / 32) * leg.ne1);
    CHECK(qsa_weight_native_bytes_from_block(leg.w, 32) != leg.native_all());   // a block figure 2 B off is caught
    CHECK(qsa_weight_native_bytes_from_block(kq.w, 128) == kq.native_all());    // Q4_K as the fake lays it out
    CHECK(128ull * (kq.ne0 / 256) != kq.native_all());                          // the one-row figure is NOT the image
    CHECK(qsa_weight_native_bytes_from_block(kq.w, 144) != kq.native_all());    // Q4_K's real figure, wrong here
    CHECK(qsa_weight_native_bytes_from_block(kq.w, 0) == 0);
    CHECK(qsa_weight_native_bytes_from_block(bad.w, 144) == 0);                 // a type with no width has no figure
}

void what_is_refused() {
    FakeWeight f;
    f.build();
    const int64_t half = f.ne0 / 2;
    // mid-block cuts, on the canonical form and on the native image
    CHECK(!qsa_weight_cols_fits(f.w, f.group / 2, half, f.native_all()));
    CHECK(!qsa_weight_cols_fits(f.w, 0, half + f.group / 2, f.native_all()));
    CHECK(!qsa_weight_cols_fits(f.w, half, half + 1, f.native_all()));
    // a group narrower than the native block: canonical spans are legal where the native image cannot be cut
    FakeWeight narrow;
    narrow.group = 64;
    narrow.build();
    CHECK(narrow.ne0 % 64 == 0);
    CHECK(qsa_weight_cols_fits(narrow.w, 0, narrow.ne0 / 2, narrow.native_all()));          // 256-aligned, fine
    CHECK(!qsa_weight_cols_fits(narrow.w, 64, 64, narrow.native_all()));                    // 64 is not a native block
    // out of range, and the empty cases
    CHECK(!qsa_weight_cols_fits(f.w, 0, 0, f.native_all()));
    CHECK(!qsa_weight_cols_fits(f.w, f.ne0 - f.group, f.group * 2, f.native_all()));
    CHECK(!qsa_weight_cols_fits(f.w, -f.group, f.group, f.native_all()));
    // a tensor whose recorded planes do not divide into blocks is refused, not guessed at
    FakeWeight odd;
    odd.build();
    odd.w.scales_bytes += 4;
    CHECK(!qsa_weight_cols_fits(odd.w, 0, half, odd.native_all()));
    // has_offset must agree with the offset plane
    FakeWeight noff;
    noff.has_offset = false;
    noff.build();
    CHECK(noff.offset_bytes == 0 && noff.w.offset_bytes == 0);
    CHECK(qsa_weight_cols_fits(noff.w, 0, half, noff.native_all()));
    FakeWeight lying;
    lying.build();
    lying.w.has_offset = false;
    CHECK(!qsa_weight_cols_fits(lying.w, 0, half, lying.native_all()));
    // a native tensor sized without its native row figure
    CHECK(!qsa_weight_cols_fits(f.w, 0, half, 0));
    FakeWeight nonative;
    nonative.with_native = false;
    nonative.build();
    CHECK(nonative.w.native_data == nullptr);
    CHECK(qsa_weight_cols_fits(nonative.w, 0, half, 0));
    const QsaShardWeight vn = qsa_pack_weight_cols(nonative.w, 0, half, 0, nullptr);
    CHECK(vn.ok && vn.native == nullptr && vn.native_bytes == 0);
    // row views outside the tensor
    CHECK(!qsa_weight_rows(f.w, 0, 0, f.native_all()).ok);
    CHECK(!qsa_weight_rows(f.w, -1, 2, f.native_all()).ok);
    CHECK(!qsa_weight_rows(f.w, f.ne1 - 1, 2, f.native_all()).ok);
    CHECK(qsa_weight_rows(f.w, f.ne1 - 1, 1, f.native_all()).ok);
}

void the_sizes_agree_with_the_packer() {
    // qsa_weight_cols_bytes is what the caller allocates; a figure smaller than what the packer writes is a
    // device-side overflow, so the two must agree for every shape and form the engine can meet.
    for (const int bits : {2, 4, 8}) {
        for (const int64_t group : {32LL, 64LL, 128LL, 256LL}) {
            for (const bool off : {true, false}) {
                for (const bool nat : {true, false}) {
                    FakeWeight f;
                    f.bits = bits;
                    f.group = group;
                    f.has_offset = off;
                    f.with_native = nat;
                    if (bits == 2 && !off) continue;   // an S2 form has no offset plane, but this is not a form check
                    f.build();
                    const int64_t cols = f.ne0 / 2, col0 = f.ne0 / 2;
                    if (!qsa_weight_cols_fits(f.w, col0, cols, f.native_all())) { std::printf("skip %d/%lld\n", bits, (long long) group); continue; }
                    std::vector<uint8_t> dst(qsa_weight_cols_bytes(f.w, col0, cols, f.native_all()));
                    const QsaShardWeight v = qsa_pack_weight_cols(f.w, col0, cols, f.native_all(), dst.data());
                    CHECK(v.ok);
                    CHECK(v.codes_bytes + v.scales_bytes + v.offset_bytes + v.native_bytes == dst.size());
                    if (f.with_native) {
                        // every half is exactly half the whole, which is the VRAM claim the split rests on
                        const uint64_t w = qsa_weight_cols_bytes(f.w, 0, f.ne0, f.native_all());
                        CHECK(w == 2 * dst.size());
                    }
                }
            }
        }
    }
}

void the_artifacts_geometry_cuts_cleanly() {
    // the o_proj: [n_embd rows][n_head * head_dim columns] = [4096][6144], the forms this pack uses for it
    FakeWeight o;
    o.ne0 = 24 * 256;
    o.ne1 = 4096;
    o.group = 32;
    o.bits = 4;
    o.native_row = 24 * 256 / 2;   // 4 bits per value in the native image
    o.build();
    const QsaHeadShard r0{24, 2, 0, 2}, r1{24, 2, 1, 2};
    const QsaHeadRun c0 = qsa_out_cols(r0, 256), c1 = qsa_out_cols(r1, 256);
    CHECK(c0.len == 12 * 256 && c1.start == 12 * 256);
    CHECK(qsa_weight_cols_fits(o.w, c0.start, c0.len, o.native_all()));
    CHECK(qsa_weight_cols_fits(o.w, c1.start, c1.len, o.native_all()));
    const uint64_t whole = qsa_weight_cols_bytes(o.w, 0, o.ne0, o.native_all());
    const uint64_t each = qsa_weight_cols_bytes(o.w, c0.start, c0.len, o.native_all());
    CHECK(whole > 0 && each * 2 == whole);
    std::printf("  o_proj 4096x6144 group 32: whole %llu B, one card of the pair %llu B\n",
            (unsigned long long) whole, (unsigned long long) each);
    // the K/V and q tensors are cut on rows, so they cost nothing to view
    FakeWeight k;
    k.ne0 = 4096;
    k.ne1 = 2 * 256;   // n_head_kv * head_dim rows
    k.build();
    CHECK(qsa_weight_rows(k.w, qsa_kv_rows(r1, 256).start, qsa_kv_rows(r1, 256).len, k.native_all()).ok);
    FakeWeight q;
    q.ne0 = 4096;
    q.ne1 = 24 * 2 * 256;
    q.build();
    const QsaHeadRun qr = qsa_q_rows(r1, 256);
    const QsaShardWeight vq = qsa_weight_rows(q.w, qr.start, qr.len, q.native_all());
    CHECK(vq.ok && vq.ne1 == qr.len);
    // a tensor WITH a native image keeps using it: a row view that drops to the canonical planes would run a
    // different decoder than the whole layer ran, and those two are not bitwise equal
    CHECK(vq.native != nullptr && vq.native_bytes > 0);
    CHECK(!qsa_weight_rows(q.w, qr.start, qr.len, 0).ok);
    CHECK(vq.codes == (const uint8_t*) q.canon.data() + (uint64_t) qr.start * (q.codes_bytes / (uint64_t) q.ne1));
}

/// The same claim with the pair cut UNEVENLY: the lead runs 16 of the 24 query heads and the partner 8.  The two
/// packed views are different sizes now, so what has to hold is that they concatenate to the whole row of every
/// plane, in order, and that a cut still lands on the blocks the form quantizes - a head is `head_dim` columns,
/// and head_dim is a multiple of every block width the pack uses, so any head-aligned cut is legal.
void an_uneven_cut_of_the_artifacts_o_proj() {
    FakeWeight o;
    o.ne0 = 24 * 256;
    o.ne1 = 4096;
    o.group = 32;
    o.bits = 4;
    o.native_row = 24 * 256 / 2;
    o.build();
    const QsaHeadShard lead = qsa_pair_shard(24, 2, 16, 0), peer = qsa_pair_shard(24, 2, 16, 1);
    CHECK(lead.ok() && peer.ok());
    const QsaHeadRun c0 = qsa_out_cols(lead, 256), c1 = qsa_out_cols(peer, 256);
    CHECK(c0.start == 0 && c0.len == 16 * 256 && c1.start == 16 * 256 && c1.len == 8 * 256);
    CHECK(qsa_weight_cols_fits(o.w, c0.start, c0.len, o.native_all()));
    CHECK(qsa_weight_cols_fits(o.w, c1.start, c1.len, o.native_all()));
    const uint64_t n0 = qsa_weight_cols_bytes(o.w, c0.start, c0.len, o.native_all());
    const uint64_t n1 = qsa_weight_cols_bytes(o.w, c1.start, c1.len, o.native_all());
    const uint64_t whole = qsa_weight_cols_bytes(o.w, 0, o.ne0, o.native_all());
    CHECK(n0 + n1 == whole);                       // unequal halves, one whole weight between them
    CHECK(n0 == 2 * n1);                           // 16 heads against 8
    std::vector<uint8_t> d0((size_t) n0), d1((size_t) n1);
    const QsaShardWeight a = qsa_pack_weight_cols(o.w, c0.start, c0.len, o.native_all(), d0.data());
    const QsaShardWeight b = qsa_pack_weight_cols(o.w, c1.start, c1.len, o.native_all(), d1.data());
    CHECK(a.ok && b.ok && a.ne0 == c0.len && b.ne0 == c1.len);
    struct P {
        const void* x;
        uint64_t nx;      // the lead's bytes of this plane (16 heads)
        const void* y;
        uint64_t ny;      // the partner's (8 heads)
        const void* w;
        uint64_t total;   // the whole weight's (24 heads)
    };
    const uint8_t* codes = (const uint8_t*) o.w.data;
    const P planes[4] = {{a.codes, a.codes_bytes, b.codes, b.codes_bytes, codes, o.codes_bytes},
                         {a.scales, a.scales_bytes, b.scales, b.scales_bytes, codes + o.codes_bytes, o.scales_bytes},
                         {a.offset, a.offset_bytes, b.offset, b.offset_bytes,
                          codes + o.codes_bytes + o.scales_bytes, o.offset_bytes},
                         {a.native, a.native_bytes, b.native, b.native_bytes, o.native.data(), o.native_all()}};
    for (const P& p : planes) {
        if (!p.x) continue;
        const uint8_t* x = (const uint8_t*) p.x;
        const uint8_t* y = (const uint8_t*) p.y;
        const uint8_t* w = (const uint8_t*) p.w;
        const size_t rowx = p.nx / (size_t) o.ne1, rowy = p.ny / (size_t) o.ne1;
        const size_t wrow = p.total / (size_t) o.ne1;   // the whole weight's row of this plane
        CHECK(rowx + rowy == wrow);                      // 16 heads then 8 heads make the row
        CHECK(rowx == 2 * rowy);                         // and the cut is uneven by exactly that
        bool same = true;
        for (int64_t r = 0; r < o.ne1 && same; ++r) {
            if (std::memcmp(x + (size_t) r * rowx, w + (size_t) r * wrow, rowx) != 0) same = false;
            if (std::memcmp(y + (size_t) r * rowy, w + (size_t) r * wrow + rowx, rowy) != 0) same = false;
        }
        CHECK(same);
    }
    // a cut between two heads is on every block width the pack uses; a cut inside a head is not
    CHECK(qsa_weight_cols_fits(o.w, 15 * 256, 9 * 256, o.native_all()));
    CHECK(!qsa_weight_cols_fits(o.w, 15 * 256 + 16, 9 * 256, o.native_all()));
    // and the row views follow the uneven head counts too: q rows 2 head_dim per head, K/V rows 1 per KV head
    const QsaHeadRun qr = qsa_q_rows(lead, 256);
    CHECK(qr.start == 0 && qr.len == 16 * 2 * 256);
    FakeWeight q;
    q.ne0 = 4096;
    q.ne1 = 24 * 2 * 256;
    q.build();
    CHECK(qsa_weight_rows(q.w, qr.start, qr.len, q.native_all()).ok);
    const QsaHeadRun kr = qsa_kv_rows(peer, 256);   // heads 16-23 read KV head 1 only
    CHECK(kr.start == 256 && kr.len == 256);
}

}  // namespace

/// The shipped artifact's attention projections as the pack actually keeps them: served from its GGUF section,
/// so the loader records the shape and NO canonical bytes (`resident` false, `data` null, `bytes` 0, every plane
/// 0) and the tensor exists only as its native image.  The pair's packed `attn_output` columns are then the
/// native span alone - and a sizer that priced the arena figures would price them at nothing, which is the
/// refusal a boot hit.  Pinned here because nothing but a pack that ships this way catches it.
void a_native_only_pack_cuts_too() {
    FakeWeight o;   // the artifact's attn_output: input axis the heads, native Q6_K (a 256-value block form)
    o.ne0 = 24 * 256;
    o.ne1 = 2560;
    o.group = 32;
    o.native_type = 14;      // Q6_K
    o.native_row = 24 * 192; // one row of that form: 24 blocks of 192 B
    o.build();
    // strip the canonical form, exactly as the loader does for a tensor served from the GGUF
    o.w.resident = false;
    o.w.data = nullptr;
    o.w.bytes = 0;
    o.w.codes_bytes = 0;
    o.w.scales_bytes = 0;
    o.w.offset_bytes = 0;
    o.w.has_offset = false;
    const int64_t half = 12 * 256;
    CHECK(!qsa_weight_has_planes(o.w));
    CHECK(qsa_weight_cols_fits(o.w, half, half, o.native_all()));
    const uint64_t each = qsa_weight_cols_bytes(o.w, half, half, o.native_all());
    CHECK(each > 0);
    CHECK(each == qsa_weight_native_cols_bytes(o.w, half, o.native_all(), o.ne1));   // native span, nothing else
    std::vector<uint8_t> dst(each);
    const QsaShardWeight v = qsa_pack_weight_cols(o.w, half, half, o.native_all(), dst.data());
    CHECK(v.ok);
    CHECK(v.codes == nullptr && v.scales == nullptr && v.offset == nullptr);   // no canonical plane to point at
    CHECK(v.native != nullptr);
    CHECK(v.codes_bytes + v.scales_bytes + v.offset_bytes + v.native_bytes == each);
    CHECK(v.ne0 == half && v.ne1 == o.ne1);
    // and the packed span is the bytes the whole native image holds for those columns, byte for byte
    const size_t row = (size_t) o.native_row;
    const size_t take = (size_t) (half / 256) * (row / (size_t) (o.ne0 / 256));
    bool same = true;
    for (int64_t r = 0; r < o.ne1 && same; ++r)
        if (std::memcmp((const uint8_t*) v.native + (size_t) r * take,
                        o.native.data() + (size_t) r * row + take, take) != 0)
            same = false;
    CHECK(same);
    // the two halves still tile the input axis, and a native type whose width cannot cut it still refuses
    CHECK(qsa_weight_cols_bytes(o.w, 0, half, o.native_all()) == each);
    FakeWeight bad;   // a fresh one: `FakeWeight`'s WeightRef points into its own vectors
    bad.ne0 = o.ne0;
    bad.ne1 = o.ne1;
    bad.group = o.group;
    bad.native_type = 14;
    bad.native_row = o.native_row;
    bad.build();
    bad.w.resident = false;
    bad.w.data = nullptr;
    bad.w.bytes = bad.w.codes_bytes = bad.w.scales_bytes = bad.w.offset_bytes = 0;
    bad.w.has_offset = false;
    bad.w.native_type = 99;   // a type the mirror table does not know: its columns cannot be cut at all
    CHECK(bad.w.native_data != nullptr);
    CHECK(!qsa_weight_cols_fits(bad.w, half, half, bad.native_all()));
    CHECK(qsa_weight_cols_bytes(bad.w, half, half, bad.native_all()) == 0);
    // and a tensor with NEITHER form is nothing to cut
    FakeWeight none;
    none.ne0 = o.ne0;
    none.ne1 = o.ne1;
    none.group = o.group;
    none.build();
    none.w.native_data = nullptr;
    none.w.native_type = -1;
    none.w.resident = false;
    none.w.data = nullptr;
    none.w.bytes = none.w.codes_bytes = none.w.scales_bytes = none.w.offset_bytes = 0;
    CHECK(!qsa_weight_cols_fits(none.w, half, half, 0));
}

/// The artifact's OWN `attn_output` as its pack keeps it (Q6_K, the input axis the 24 heads, the output axis
/// n_embd), so the figure the partner's tier is shrunk by is pinned by its real value and not by a round number.
/// Q6_K is 210 B per 256 values (6.5625 bits a value), which is what makes the whole image 12 902 400 B - the
/// byte count the pack's own metadata carries, so this test and the pack agree by construction.
void the_artifacts_native_o_proj_prices_what_the_tier_holds() {
    constexpr int64_t NE0 = 24 * 256, NE1 = 2560, ROW = 210 * 24;
    FakeWeight o;
    o.ne0 = NE0;
    o.ne1 = NE1;
    o.group = 32;
    o.native_type = 14;   // Q6_K
    o.native_row = ROW;
    o.build();
    o.w.resident = false;
    o.w.data = nullptr;
    o.w.bytes = o.w.codes_bytes = o.w.scales_bytes = o.w.offset_bytes = 0;
    o.w.has_offset = false;
    CHECK((uint64_t) ROW * NE1 == 12902400ull);   // the pack's own `attn_output` byte count
    const QsaHeadShard partner = qsa_pair_shard(24, 2, 12, 1);
    const QsaHeadRun cols = qsa_out_cols(partner, 256);
    CHECK(cols.start == 12 * 256 && cols.len == 12 * 256);
    const uint64_t one_layer = qsa_weight_cols_bytes(o.w, cols.start, cols.len, (uint64_t) ROW * NE1);
    CHECK(one_layer == 6451200ull);   // half the native image, and nothing else
    std::printf("the pack's attn_output: %llu B whole, one card's columns %llu B a QSA layer\n",
                (unsigned long long) (ROW * NE1), (unsigned long long) one_layer);
}

/// Q6_K's block: 256 values in 210 bytes (`ql[128] qh[64] scales[16]` then the fp16 `d` LAST).  The two places
/// that must agree on these are the CUT (`qsa_weight_native_block_bytes`, derived from the loader's image size)
/// and the DEQUANTIZER (`dequant_bf16.cu`'s `geometry()`, from its own table); `qsa_prepare_one` cross-checks the
/// two against the image's total, and this pins what that check cannot: that a buffer cut at one pitch still
/// DECODES as the columns it claims, when the product tells the kernel the view's `ne0` and the kernel rebuilds
/// the pitch from it.
constexpr int64_t kQ6kElems = 256, kQ6kBytes = 210;

/// One row of a blocked image, decoded the way `dequant_f16` will decode it: `pitch_blocks` blocks per row, so a
/// caller can ask what the same bytes would mean at a pitch the packer did not use.
void decode_q6k_row(const uint8_t* blocks, int64_t pitch_blocks, int64_t row, int64_t cols,
                    std::vector<float>& out) {
    out.resize((size_t) cols);
    for (int64_t b = 0; b < cols / kQ6kElems; ++b)
        basalt::dequantize_q6_K(blocks + ((size_t) row * (size_t) pitch_blocks + (size_t) b) * (size_t) kQ6kBytes,
                                out.data() + (size_t) b * (size_t) kQ6kElems);
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

/// A card's packed `attn_output` columns must decode, element for element, to the columns of the whole image they
/// claim to be - at the artifact's own form and geometry, with the packer's pitch and the dequantizer's own.
///
/// This is the claim the pair's `o_proj` rests on and the one no byte test makes: `qsa_pack_weight_cols` can copy
/// exactly the right bytes and the product still reads nonsense, because the bytes mean nothing without the pitch
/// the decoder assumes, and the two are computed in two files.  A mismatch here is the nan this hunt is chasing:
/// a block read across a boundary decodes its scale bits as a code and its code bits as an fp16 scale, which is
/// how a weight that packed cleanly arrives as a partial of nan.
void the_cut_decodes_as_the_wholes_columns() {
    constexpr int64_t NE0 = 24 * 256, NE1 = 2560, HALF = 12 * 256;
    FakeWeight f;
    f.ne0 = NE0;
    f.ne1 = NE1;
    f.group = 32;
    f.native_type = 14;                       // Q6_K, as the pack keeps `attn_output`
    f.native_row = (NE0 / kQ6kElems) * kQ6kBytes;
    f.build();
    f.w.resident = false;                     // served from the GGUF section: one form, the native image
    f.w.data = nullptr;
    f.w.bytes = f.w.codes_bytes = f.w.scales_bytes = f.w.offset_bytes = 0;
    f.w.has_offset = false;
    CHECK((uint64_t) f.native.size() == 12902400ull);   // the pack's own attn_output byte count
    // Q6_K's `d` is the LAST two bytes of the block, and the fake's pattern is free to set NaN bits there: a NaN
    // would compare unequal to itself and read as a broken cut, so give every block a scale that decodes.
    for (int64_t r = 0; r < NE1; ++r)
        for (int64_t b = 0; b < NE0 / kQ6kElems; ++b) {
            uint8_t* blk = f.native.data() + (size_t) r * (size_t) f.native_row + (size_t) b * (size_t) kQ6kBytes;
            blk[208] = 0x00;
            blk[209] = (uint8_t) (0x3C + (b % 4));   // d in [1.0, 2.0], finite by construction
        }

    const QsaHeadShard lead = qsa_pair_shard(24, 2, 12, 0), partner = qsa_pair_shard(24, 2, 12, 1);
    const QsaHeadRun lc = qsa_out_cols(lead, 256), pc = qsa_out_cols(partner, 256);
    CHECK(lc.start == 0 && lc.len == HALF);
    CHECK(pc.start == HALF && pc.len == HALF);

    std::vector<uint8_t> lb(qsa_weight_cols_bytes(f.w, lc.start, lc.len, f.native_all())),
        pb(qsa_weight_cols_bytes(f.w, pc.start, pc.len, f.native_all()));
    CHECK((uint64_t) lb.size() == 6451200ull && (uint64_t) pb.size() == 6451200ull);
    const QsaShardWeight lv = qsa_pack_weight_cols(f.w, lc.start, lc.len, f.native_all(), lb.data());
    const QsaShardWeight pv = qsa_pack_weight_cols(f.w, pc.start, pc.len, f.native_all(), pb.data());
    CHECK(lv.ok && pv.ok);
    // the pitch `dequant_f16` will compute from the VIEW's ne0, and the pitch the packer actually wrote
    const int64_t pitch_blocks = lv.ne0 / kQ6kElems;
    CHECK(pitch_blocks == HALF / kQ6kElems);
    CHECK((uint64_t) pitch_blocks * (uint64_t) kQ6kBytes * (uint64_t) lv.ne1 == lv.native_bytes);

    for (const int64_t row : {int64_t{0}, int64_t{1}, NE1 / 2, NE1 - 1}) {
        std::vector<float> whole, cut;
        decode_q6k_row((const uint8_t*) f.native.data(), NE0 / kQ6kElems, row, NE0, whole);
        CHECK(all_finite(whole));
        decode_q6k_row((const uint8_t*) lv.native, pitch_blocks, row, lv.ne0, cut);
        CHECK(all_finite(cut));
        CHECK(cut == std::vector<float>(whole.begin(), whole.begin() + HALF));   // the lead's columns
        decode_q6k_row((const uint8_t*) pv.native, pitch_blocks, row, pv.ne0, cut);
        CHECK(all_finite(cut));
        CHECK(cut == std::vector<float>(whole.begin() + HALF, whole.end()));     // the partner's
    }

    // And the test can fail: the same bytes read at the WHOLE image's pitch - what happens when one side computes
    // the stride the other side did not - must not reproduce those columns, or the comparison above is vacuous.
    {
        std::vector<float> whole, wrong;
        decode_q6k_row((const uint8_t*) f.native.data(), NE0 / kQ6kElems, 1, NE0, whole);
        decode_q6k_row((const uint8_t*) lv.native, NE0 / kQ6kElems, 1, HALF, wrong);
        CHECK(wrong != std::vector<float>(whole.begin(), whole.begin() + HALF));
    }
}

int main() {
    the_column_view_is_the_wholes_bytes();
    the_two_halves_are_the_whole_row();
    a_whole_view_is_the_tensor_itself();
    the_row_view_is_a_pointer();
    the_native_block_width_follows_the_type();
    what_is_refused();
    the_sizes_agree_with_the_packer();
    the_artifacts_geometry_cuts_cleanly();
    an_uneven_cut_of_the_artifacts_o_proj();
    a_native_only_pack_cuts_too();
    the_artifacts_native_o_proj_prices_what_the_tier_holds();
    the_cut_decodes_as_the_wholes_columns();
    if (g_fail != 0) { std::printf("tp2 shard weights test failed: %d\n", g_fail); return 1; }
    std::printf("tp2 shard weights test ok\n");
    return 0;
}

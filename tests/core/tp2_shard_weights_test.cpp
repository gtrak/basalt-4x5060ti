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
    if (g_fail != 0) { std::printf("tp2 shard weights test failed: %d\n", g_fail); return 1; }
    std::printf("tp2 shard weights test ok\n");
    return 0;
}

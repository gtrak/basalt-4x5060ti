// include/basalt/prefill/moe_fold.hpp - the partial merge's per-token fold list, built in one place.
//
// `--exp peer_partial_merge` makes the partner fold the rows of the experts it holds into the layer's `T x n_embd`
// partial. The fold kernel is handed a compact list - which tokens this group touches, each token's run of rows,
// and each row's weight - and it trusts that list completely: it adds exactly what the list says, so a row the list
// drops is a row that silently leaves the MoE sum, and a row carrying another row's weight is silently mis-weighted.
// Neither shows up as a crash or a bad shape, only as a changed answer, so the list is not something the prefill
// thread should build by hand and a test should not rebuild from scratch.
//
// This is the builder the prefill thread runs, extracted so the CPU parity test calls the same code rather than a
// second opinion of it (`tests/kernels/moe_part_parity.cpp`). The arithmetic it feeds lives beside it in
// `moe_part.hpp`; the order of the terms is what `docs/precision/moe-partial-merge.md` covers.
//
// Host-only, unlike `moe_part.hpp`: the device reads the finished list, it never builds one.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::prefill {

/// The scratch and the finished list for ONE expert group, kept between groups so no group re-zeroes the whole
/// per-token count. `tok`/`off` cover the group's touched tokens, `rel`/`wt` its rows.
struct MoeFoldTables {
    std::vector<int32_t> tok;   // a token this group holds rows of, each listed once
    std::vector<int32_t> off;   // nt + 1: token e's run is [off[e], off[e + 1]) in rel/wt
    std::vector<int32_t> rel;   // the row's index within the group, i.e. its row id minus the block's first row
    std::vector<float> wt;      // the row's router weight, as the lead's router weighed it
    std::vector<int32_t> cnt;   // scratch, per chunk token: the rows of it this group holds
    std::vector<int32_t> cur;   // scratch, per chunk token: where the next row of it lands
};

/// Count one group's rows into a per-token list, in k order.
///
/// Row `q` of the group belongs to token `tok_of_row[q]`, was the `k_of_row[q]`-th routing of that token, and is
/// worth `w_of_row[q]`; all three are indexed group-relative, so the caller offsets them to the group's first row.
/// The count is a counting sort over tokens, so the cost is the group's rows and a token the group never touched
/// keeps its zero for the next group. A token's run comes out sorted by `k` - the order the single-card combine
/// adds its rows in, which keeps the two chains as close as a split one can.
///
/// Returns the number of tokens the group touches. A group with no rows touches none.
inline int32_t moe_fold_tables(MoeFoldTables& F, const int32_t* tok_of_row, const int32_t* k_of_row,
                               const float* w_of_row, int32_t nr) {
    F.tok.clear();
    if (nr <= 0) return 0;

    // One entry per token the group touches, and a count of its rows.
    for (int32_t q = 0; q < nr; ++q)
        if (F.cnt[(size_t) tok_of_row[q]]++ == 0) F.tok.push_back(tok_of_row[q]);
    const int32_t nt = (int32_t) F.tok.size();

    // Prefix-sum the counts into the runs, then scatter the rows into them, then sort each run into k order. The
    // count doubles as the "have I seen this token" flag, so it is cleared here, per touched token, for the next
    // group; the cursor needs no clearing because its token's run start is written before the token is used.
    F.off.resize((size_t) nt + 1);
    F.rel.resize((size_t) nr);
    F.wt.resize((size_t) nr);
    int32_t run = 0;
    for (int32_t e = 0; e < nt; ++e) {
        const int32_t t = F.tok[(size_t) e];
        F.off[(size_t) e] = run;
        F.cur[(size_t) t] = run;
        run += F.cnt[(size_t) t];
    }
    F.off[(size_t) nt] = run;

    for (int32_t q = 0; q < nr; ++q) {
        const int32_t p = F.cur[(size_t) tok_of_row[q]]++;
        F.rel[(size_t) p] = q;
        F.wt[(size_t) p] = w_of_row[q];
    }

    for (int32_t e = 0; e < nt; ++e) {   // a run is one token's rows: at most K of them, so sort by insertion
        int32_t* rel = F.rel.data() + F.off[(size_t) e];
        float* wt = F.wt.data() + F.off[(size_t) e];
        const int32_t n = F.off[(size_t) e + 1] - F.off[(size_t) e];
        for (int32_t i = 1; i < n; ++i) {
            const int32_t r = rel[i];
            const float v = wt[i];
            const int32_t key = k_of_row[r];
            int32_t j = i - 1;
            while (j >= 0 && k_of_row[rel[j]] > key) {
                rel[j + 1] = rel[j];
                wt[j + 1] = wt[j];
                --j;
            }
            rel[j + 1] = r;
            wt[j + 1] = v;
        }
        F.cnt[(size_t) F.tok[(size_t) e]] = 0;   // clean for the next group
    }
    return nt;
}

}  // namespace basalt::prefill

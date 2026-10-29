// tests/program/tp2_pair_test.cpp - the invariant `--tp-size 2` places on: of a stage's pair of cards exactly one
// holds each expert, each card's half keeps the ranking's order, and the stage cards fall out in pairs.
#include "basalt/program/tp2_pair.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace basalt::program;

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

using Ranked = std::vector<std::pair<int32_t, int32_t>>;

// the model's geometry: 48 layers, 512 experts, every expert hot somewhere
Ranked real_geometry() {
    Ranked r;
    for (int32_t e = 0; e < 512; ++e)
        for (int32_t l = 0; l < 48; ++l) r.push_back({l, e});
    return r;
}

void cards() {
    CHECK(pair_lead_dev(0, 2) == 0);
    CHECK(pair_partner_dev(0, 2) == 1);
    CHECK(pair_lead_dev(1, 2) == 2);
    CHECK(pair_partner_dev(1, 2) == 3);
    CHECK(pair_lead_dev(2, 2) == 4);
    CHECK(pair_partner_dev(3, 2) == 7);
    // the cards of a stage are distinct, and no card is the partner of two stages
    for (int s = 0; s < 4; ++s) {
        CHECK(pair_lead_dev(s, 2) != pair_partner_dev(s, 2));
        CHECK(pair_lead_dev(s + 1, 2) == pair_partner_dev(s, 2) + 1);
    }
    // tp_size 1 is the plain split: one card per stage, and it holds every expert
    CHECK(pair_lead_dev(3, 1) == 3);
    CHECK(pair_holds(411, 1, 0));
    CHECK(!pair_holds(411, 1, 1));   // a stage alone has no partner to hold anything
}

void exactly_one_card_holds_each_expert() {
    for (int32_t e = 0; e < 512; ++e) {
        CHECK((int) pair_holds(e, 2, 0) + (int) pair_holds(e, 2, 1) == 1);
        CHECK(pair_holds(e, 2, (int) (e % 2)));
        CHECK(!pair_holds(e, 2, (int) ((e + 1) % 2)));
    }
}

void halves_cover_and_disjoint() {
    const Ranked r = real_geometry();
    const Ranked lead = pair_ranked(r, 2, 0), partner = pair_ranked(r, 2, 1);
    CHECK(lead.size() + partner.size() == r.size());
    CHECK(lead.size() == 512 * 48 / 2);
    CHECK(partner.size() == lead.size());
    // disjoint: no (layer, expert) in both halves
    size_t both = 0;
    for (const auto& a : lead)
        for (const auto& b : partner)
            if (a == b) ++both;
    CHECK(both == 0);
    // and the order survives: each half is the input filtered, hottest first
    size_t i = 0;
    for (const auto& pr : r) {
        if (i < lead.size() && lead[i] == pr) ++i;
    }
    CHECK(i == lead.size());
    size_t j = 0;
    for (const auto& pr : r) {
        if (j < partner.size() && partner[j] == pr) ++j;
    }
    CHECK(j == partner.size());
}

void halves_balance() {
    // an odd number of experts leaves one card with a single expert more, never a whole group
    for (int32_t n = 1; n < 64; ++n) {
        Ranked r;
        for (int32_t e = 0; e < n; ++e) r.push_back({0, e});
        const size_t a = pair_ranked(r, 2, 0).size(), b = pair_ranked(r, 2, 1).size();
        CHECK(a + b == (size_t) n);
        CHECK(a >= b ? a - b <= 1 : b - a <= 1);
    }
}

void an_empty_half_is_empty() {
    const Ranked none;
    CHECK(pair_ranked(none, 2, 0).empty());
    CHECK(pair_ranked(none, 2, 1).empty());
    // a stage's layers hold ids of one parity only: the other card's half is empty, which is what the start says so
    Ranked evens;
    for (int32_t e = 0; e < 8; e += 2) evens.push_back({3, e});
    CHECK(pair_ranked(evens, 2, 0).size() == 4);
    CHECK(pair_ranked(evens, 2, 1).empty());
}

}  // namespace

int main() {
    cards();
    exactly_one_card_holds_each_expert();
    halves_cover_and_disjoint();
    halves_balance();
    an_empty_half_is_empty();
    if (g_fail != 0) { std::printf("tp2 pair split test failed: %d\n", g_fail); return 1; }
    std::printf("tp2 pair split test ok\n");
    return 0;
}

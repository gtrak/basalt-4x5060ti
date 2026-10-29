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

void partner_follows_the_lead() {
    // The engine asks a stage's partner by the card the stage leads on, because its stage list holds only the stages
    // after the main card's: asked by a position in that list, a stage gets the card beside the PREVIOUS stage.
    CHECK(pair_partner_of_lead(0, 2) == 1);
    CHECK(pair_partner_of_lead(2, 2) == 3);
    CHECK(pair_partner_of_lead(6, 2) == 7);
    CHECK(pair_partner_of_lead(3, 2) == -1);   // a partner card leads no stage, so it has no partner of its own
    CHECK(pair_partner_of_lead(5, 2) == -1);
    CHECK(pair_partner_of_lead(-2, 2) == -1);
    CHECK(pair_partner_of_lead(0, 1) == -1);   // the plain split's stage has no partner
    for (int s = 0; s < 8; ++s)
        CHECK(pair_partner_of_lead(pair_lead_dev(s, 2), 2) == pair_partner_dev(s, 2));
    // a run's stages and partners are its cards, each holding one stage's half: no card twice over, none empty
    const int stages[] = {0, 1};
    for (int c = 0; c < 4; ++c) {
        int uses = 0;
        for (int s : stages) {
            if (pair_lead_dev(s, 2) == c) ++uses;
            if (pair_partner_of_lead(pair_lead_dev(s, 2), 2) == c) ++uses;
        }
        CHECK(uses == 1);
    }
}

void exactly_one_card_holds_each_expert() {
    // Under a rank split the id alone cannot say which card holds an expert - the ranked list decides - so the
    // partition property is asserted on `pair_ranked` (halves_cover_and_disjoint below).  `pair_holds` answers only
    // the single-card stage (tp_size 1: the lead holds everything, there is no partner rank).
    CHECK(pair_holds(411, 1, 0));
    CHECK(!pair_holds(411, 1, 1));
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
    // a rank split keeps the ranking's order inside each half: the lead's half is the hottest prefix, in order
    Ranked four;
    for (int32_t e = 0; e < 8; e += 2) four.push_back({3, e});
    const Ranked lead = pair_ranked(four, 2, 0), partner = pair_ranked(four, 2, 1);
    CHECK(lead.size() == 2 && partner.size() == 2);
    CHECK(lead[0] == four[0] && lead[1] == four[1]);
    CHECK(partner[0] == four[2] && partner[1] == four[3]);
}

}  // namespace

int main() {
    cards();
    partner_follows_the_lead();
    exactly_one_card_holds_each_expert();
    halves_cover_and_disjoint();
    halves_balance();
    an_empty_half_is_empty();
    if (g_fail != 0) { std::printf("tp2 pair split test failed: %d\n", g_fail); return 1; }
    std::printf("tp2 pair split test ok\n");
    return 0;
}

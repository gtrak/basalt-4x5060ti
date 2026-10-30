// tests/spec/suffix_drafter_test.cpp - suffix-lookup drafter tests and an offline simulation.
//
//   suffix_drafter_test                      unit tests
//   suffix_drafter_test --simulate [--k K] FILE.ids ...
//
// The simulation replays text as if a model had produced it: each step drafts up to K tokens, accepts the
// prefix that matches the true next tokens, and commits that prefix plus one model token (the verify pass's
// bonus). tokens/step is the lookup drafter's ceiling speedup on that text when a K+1-token verify costs the same
// as a 1-token step. Two regimes per file:
//   continue  history = first half of the prompt; generate its second half (natural continuation)
//   copy      history = the whole prompt; generate its middle third again (output quoting input: edits)
// No model runs; this measures the drafter on text, not on the model's own outputs.
#include "basalt/spec/suffix_drafter.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using basalt::spec::SuffixDrafter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

std::vector<int32_t> read_ids(const char* path) {
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream in(text);
    std::vector<int32_t> ids;
    for (int32_t v; in >> v;) ids.push_back(v);
    return ids;
}

struct Sim { long steps = 0, tokens = 0, proposals = 0, accepted = 0; };

Sim simulate(const std::vector<int32_t>& history, const std::vector<int32_t>& target, int K, bool reseed) {
    SuffixDrafter d(3, 32, history.size() + target.size() + 16);
    d.append(history.data(), history.size());
    if (reseed) { d.set_reseed(true); d.mark_history(); }
    Sim s;
    std::vector<int32_t> draft(K);
    size_t i = 0;
    while (i < target.size()) {
        const int n = d.propose(K, draft.data());
        int a = 0;
        while (a < n && i + a < target.size() && draft[a] == target[i + a]) ++a;
        if (n > 0) { ++s.proposals; s.accepted += a; }
        const size_t commit = std::min(target.size() - i, (size_t) a + 1);
        d.append(&target[i], commit);
        i += commit;
        s.tokens += (long) commit;
        ++s.steps;
    }
    return s;
}

void unit_tests() {
    {   // a repeated passage is proposed verbatim
        SuffixDrafter d;
        const std::vector<int32_t> doc = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
        d.append(doc.data(), doc.size());
        const int32_t again[] = {10, 11, 12};
        d.append(again, 3);
        int32_t out[8];
        const int n = d.propose(5, out);
        check(n == 5 && out[0] == 13 && out[4] == 17, "repeat proposes the continuation");
        check(d.last_match() == 3, "match length is the shared suffix");
    }
    {   // no earlier occurrence: nothing proposed
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 4, 5, 6, 7};
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) == 0, "no match proposes nothing");
    }
    {   // two occurrences of the trigram: the longer match wins even though it is older
        SuffixDrafter d;
        const std::vector<int32_t> doc = {7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        const int n = d.propose(2, out);
        check(n == 2 && out[0] == 100 && out[1] == 101, "longest match preferred over most recent");
        check(d.last_match() == 5, "longest match length 5");
    }
    {   // a match shorter than min_match is ignored (only a bigram in common)
        SuffixDrafter d(4);
        const std::vector<int32_t> doc = {1, 2, 3, 9, 5, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        check(d.propose(2, out) == 0, "min_match respected");
    }
    {   // periodic text: the continuation may run into the current suffix
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 1, 2, 3, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[6];
        const int n = d.propose(6, out);
        check(n >= 3 && out[0] == 1 && out[1] == 2 && out[2] == 3, "periodic continuation");
    }
    {   // bounded memory: appending beyond the nominal capacity does not crash and still works
        SuffixDrafter d(3, 32, 1024);
        std::vector<int32_t> doc;
        for (int i = 0; i < 5000; ++i) doc.push_back(i % 997);
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) > 0, "propose after overflow of nominal capacity");
    }
    {   // reseed: a short match sourced from the generated tail qualifies below min_match
        SuffixDrafter d(8, 32, 1024);
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        d.append(prompt.data(), prompt.size());
        d.mark_history();
        d.set_reseed(true);
        // the tail repeats [100..107] partly: the current suffix's shared stretch with the earlier occurrence is
        // 5 tokens - below min_match, but it is the tail's own repeat
        const std::vector<int32_t> block = {100, 101, 102, 103, 104, 105, 106, 107};
        d.append(block.data(), block.size());
        const std::vector<int32_t> again = {100, 101, 102, 103, 104};
        d.append(again.data(), again.size());
        int32_t out[8];
        const int n = d.propose(8, out);
        check(n == 8 && out[0] == 105 && out[7] == 104, "tail-sourced match below min_match proposes the full window");
        check(d.last_match() == 5, "short tail match length reported");
    }
    {   // reseed: a short match sourced from the prompt does not qualify
        SuffixDrafter d(8, 32, 1024);
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        d.append(prompt.data(), prompt.size());
        d.mark_history();
        d.set_reseed(true);
        const std::vector<int32_t> tail = {200, 201, 1, 2, 3};   // the trigram comes from the prompt
        d.append(tail.data(), tail.size());
        int32_t out[4];
        check(d.propose(4, out) == 0, "prompt-sourced short match stays below min_match");
    }
    {   // reseed: without mark_history the tail floor does not apply
        SuffixDrafter d(8, 32, 1024);
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        d.append(prompt.data(), prompt.size());
        d.set_reseed(true);
        const std::vector<int32_t> tail = {30, 31, 32, 40, 41, 42, 43, 32, 40, 41, 42};   // a 3-token tail repeat
        d.append(tail.data(), tail.size());
        int32_t out[4];
        check(d.propose(4, out) == 0, "no mark: no tail floor");
    }
    {   // reseed: a long match is proposed in full (the cap only bites below it)
        SuffixDrafter d(8, 32, 1024);
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        d.append(prompt.data(), prompt.size());
        d.mark_history();
        d.set_reseed(true);
        const std::vector<int32_t> again = {1, 2, 3, 4, 5, 6, 7, 8};
        d.append(again.data(), again.size());
        int32_t out[8];
        const int n = d.propose(8, out);
        check(n == 8 && out[0] == 9 && out[7] == 16, "long match proposes the full window");
    }
    {   // reseed off: the same tail fixture proposes nothing (byte-identical default)
        SuffixDrafter d(8, 32, 1024);
        const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        d.append(prompt.data(), prompt.size());
        const std::vector<int32_t> block = {100, 101, 102, 103, 104, 105, 106, 107};
        d.append(block.data(), block.size());
        const std::vector<int32_t> again = {100, 101, 102, 103, 104};
        d.append(again.data(), again.size());
        int32_t out[8];
        check(d.propose(8, out) == 0, "default mode: no short-match proposals");
    }
    {   // index growth: a long mixed history stays queryable, and appends stay O(1) amortized
        SuffixDrafter d(3, 64, 1 << 17);
        std::vector<int32_t> doc;
        doc.reserve(1 << 18);
        for (int i = 0; i < (1 << 18); ++i) doc.push_back(i % 997 == 500 ? 0 : i % 4099);
        const auto t0 = std::chrono::steady_clock::now();
        d.append(doc.data(), doc.size());
        const double us_per_tok = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / doc.size();
        check(d.size() == doc.size(), "history size tracks appends");
        int32_t out[4];
        check(d.propose(4, out) > 0, "propose after a 256k-token history");
        check(us_per_tok < 10.0, "amortized append under 10 us/token (measured ~0.1)");
        std::printf("append cost: %.2f us/token over %zu tokens\n", us_per_tok, doc.size());
    }
    std::printf("suffix_drafter unit tests: %s\n", g_fail ? "FAILED" : "OK");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--simulate") == 0) {
        int K = 8;
        bool reseed = false;
        int first = 2;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--k") == 0) K = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--reseed") == 0) reseed = true;
            else { first = i; break; }
        }
        std::printf("%-28s %8s | %-44s | %-44s\n", "prompt", "tokens", "continue: tok/step  proposals  acc/proposal",
                    "copy: tok/step  proposals  acc/proposal");
        for (int f = first; f < argc; ++f) {
            const std::vector<int32_t> ids = read_ids(argv[f]);
            if (ids.size() < 64) continue;
            const size_t half = ids.size() / 2;
            const Sim c = simulate({ids.begin(), ids.begin() + half}, {ids.begin() + half, ids.end()}, K, reseed);
            const size_t a = ids.size() / 3, b = 2 * ids.size() / 3;
            const Sim p = simulate(ids, {ids.begin() + a, ids.begin() + b}, K, reseed);
            const char* name = std::strrchr(argv[f], '/') ? std::strrchr(argv[f], '/') + 1 : argv[f];
            std::printf("%-28s %8zu | %8.2f %10.1f%% %10.2f              | %8.2f %10.1f%% %10.2f\n", name, ids.size(),
                        (double) c.tokens / c.steps, 100.0 * c.proposals / c.steps,
                        c.proposals ? (double) c.accepted / c.proposals : 0.0, (double) p.tokens / p.steps,
                        100.0 * p.proposals / p.steps, p.proposals ? (double) p.accepted / p.proposals : 0.0);
        }
        return 0;
    }
    unit_tests();
    return g_fail ? 1 : 0;
}

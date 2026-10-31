// tests/prefill/chunk_dump_parity.cpp - the per-layer dump's classification and fingerprints, on the CPU (no device).
//
// `--exp chunk_dump=1` prints one line per layer of a prompt chunk so that two boots of one build and one command
// can be told apart by the first line they differ on (docs/precision/moe-partial-merge.md covers why a run's row
// division is a numerics decision, not a scheduling one). The line is only worth diffing if three things hold, and
// this grades all three, on the functions `src/prefill/prefill.cpp` itself calls:
//
//   1. the kernel name is the path the layer ran on, for every combination of the four flags the engine reads -
//      a line that names the wrong kernel is worse than no line, because it is believed;
//   2. a fingerprint says what it claims: the routing's changes when a routed id changes or the order of them
//      does, and the division's changes when one expert moves between cards and not when an unrelated one does;
//   3. a fingerprint is stable: the same tables twice over give the same words, so a pair of boots that agree in
//      fact print the same line rather than a pair that merely looks like one.
//
// Fingerprints are FNV-1a, so a collision here is possible in principle and is not what the test is for; what it
// grades is sensitivity (a change shows) and stability (no change does not), which is what the diff between two
// boots relies on. Random geometries, the way every kernel gets graded.
#include "basalt/prefill/chunk_dump.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using basalt::prefill::chunk_hash;
using basalt::prefill::chunk_hash_ids;
using basalt::prefill::chunk_hash_split;
using basalt::prefill::chunk_kernel;
using basalt::prefill::chunk_kernel_name;
using basalt::prefill::chunk_source;
using basalt::prefill::ChunkKernel;
using basalt::prefill::ChunkSource;
using basalt::prefill::kFnvOffset;

int fails = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++fails;
    if (fails <= 12) std::printf("   FAIL: %s\n", what.c_str());
}

/// The four flags `Prefill::run_impl` reads for a layer, as the strings the line prints.
std::string path_of(bool use_mmq, bool fp4_l, bool fused_l, bool fused_nat) {
    return std::string(chunk_kernel_name(chunk_kernel(use_mmq, fp4_l, fused_l, fused_nat)));
}

/// A chunk's routing, as the host read it: `n` expert ids in token order.
std::vector<int32_t> routing(int64_t n, int32_t n_expert, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> ids((size_t) n);
    for (int64_t i = 0; i < n; ++i) ids[(size_t) i] = (int32_t) (rng() % (uint32_t) n_expert);
    return ids;
}

}  // namespace

int main() {
    // 1. the kernel name, over every combination of the flags - 16 of them, each with the name the line prints
    {
        struct Row { bool mmq, fp4, fused, nat; const char* want; };
        const Row rows[] = {
            {false, false, false, false, "fp16"},
            {true, false, false, false, "mmq"},
            {true, true, false, false, "fp4"},
            {false, false, true, false, "fused"},
            {false, false, false, true, "fused-nat"},
            {true, false, true, false, "fused"},      // a streamed chunk of a covered pack: MMQ's flags stand,
            {true, true, true, false, "fused"},       //   the layer still runs the fused kernels and nothing else
            {true, true, true, true, "fused-nat"},
            {false, true, false, false, "fp4"},       // an impossible pairing still names the narrower leg
            {false, false, true, true, "fused-nat"},
            {true, false, false, true, "fused-nat"},
            {false, true, true, false, "fused"},
            {false, true, false, true, "fused-nat"},
            {false, false, false, false, "fp16"},
            {true, true, false, true, "fused-nat"},
            {true, false, true, true, "fused-nat"},
            {false, true, true, true, "fused-nat"},
        };
        for (const Row& r : rows)
            check(path_of(r.mmq, r.fp4, r.fused, r.nat) == r.want,
                  "path(" + std::to_string(r.mmq) + "," + std::to_string(r.fp4) + "," + std::to_string(r.fused) +
                      "," + std::to_string(r.nat) + ") = " + path_of(r.mmq, r.fp4, r.fused, r.nat) +
                      ", expected " + r.want);
        // the enum's own order, since the line prints a name and a diff reads the name
        check((int) ChunkKernel::fp16 == 0 && (int) ChunkKernel::mmq == 1 && (int) ChunkKernel::fp4 == 2 &&
                  (int) ChunkKernel::fused == 3 && (int) ChunkKernel::fused_native == 4,
              "the kernel enum moved, and a stored value would mean something else");
        std::printf("1. the kernel a layer ran on, over the flags the engine reads\n");
    }

    // 2. the source's precedence: an expert the CPU took goes to the CPU even when the partner's table names it,
    //    because the lead's walk and the partner's both skip an expert the CPU claimed
    {
        check(chunk_source(false, false) == ChunkSource::lead, "an expert nobody else took stays here");
        check(chunk_source(true, false) == ChunkSource::peer, "the partner's table sends its rows there");
        check(chunk_source(false, true) == ChunkSource::cpu, "the CPU's table takes the expert");
        check(chunk_source(true, true) == ChunkSource::cpu, "the CPU's table wins over the partner's");
        std::printf("2. which card computed an expert, out of the three tables\n");
    }

    // 3. the routing's fingerprint: stable, and sensitive to a value and to an order
    {
        const int32_t ne = 512;
        std::mt19937 rng(20250614u);
        int unstable = 0, blind = 0;
        for (int trial = 0; trial < 300; ++trial) {
            const int64_t n = 1 + (int64_t) (rng() % 40000u);
            std::vector<int32_t> ids = routing(n, ne, rng());
            const uint64_t h = chunk_hash_ids(ids.data(), n);
            if (h != chunk_hash_ids(ids.data(), n)) ++unstable;
            if (h == kFnvOffset) ++blind;   // a real routing of one id at least does not read as "no routing"
            // one token routed elsewhere: a different layer, a different word
            const int64_t at = (int64_t) (rng() % (uint64_t) n);
            const int32_t was = ids[(size_t) at];
            ids[(size_t) at] = (int32_t) ((was + 1 + rng() % (uint32_t) (ne - 1)) % (uint32_t) ne);
            if (chunk_hash_ids(ids.data(), n) == h) ++blind;
            ids[(size_t) at] = was;
            // the same ids with two of them swapped: the host's order is the walk's, so a permutation must show -
            // unless the two ids were equal, which leaves the table, and the word, unchanged
            if (n > 3 && ids[(size_t) at] != ids[(size_t) ((at + 1) % n)]) {
                std::swap(ids[(size_t) at], ids[(size_t) ((at + 1) % n)]);
                if (chunk_hash_ids(ids.data(), n) == h) ++blind;
            }
        }
        check(unstable == 0, "a routing fingerprinted twice gave two words");
        check(blind == 0, "a routing fingerprint failed to notice a change");
        check(chunk_hash_ids(nullptr, 100) == kFnvOffset, "a path that keeps its routing on the card says so");
        check(chunk_hash_ids(routing(1, ne, 7).data(), 0) == kFnvOffset, "zero routed ids read as none");
        std::printf("3. the routing's fingerprint: stable, and it notices a moved id and a swapped pair\n");
    }

    // 4. the division's fingerprint: an expert moving between cards changes it, and the single-card case - both
    //    tables absent - is its own word, not one any division with an expert elsewhere can give
    {
        const int64_t ne = 512;
        std::mt19937 rng(99991u);
        int unstable = 0, blind = 0;
        for (int trial = 0; trial < 300; ++trial) {
            std::vector<char> peer((size_t) ne, 0), cpu((size_t) ne, 0);
            bool any_elsewhere = false;
            for (int64_t e = 0; e < ne; ++e) {
                const uint32_t r = rng() % 10u;
                peer[(size_t) e] = (char) (r < 2u ? 1 : 0);
                cpu[(size_t) e] = (char) (r >= 9u ? 1 : 0);
                any_elsewhere = any_elsewhere || peer[(size_t) e] || cpu[(size_t) e];
            }
            const uint64_t h = chunk_hash_split(peer.data(), cpu.data(), ne);
            if (h != chunk_hash_split(peer.data(), cpu.data(), ne)) ++unstable;
            // the single-card case: nothing handed anywhere, which the null tables and all-zero tables must agree on
            std::vector<char> none((size_t) ne, 0);
            if (chunk_hash_split(none.data(), none.data(), ne) != chunk_hash_split(nullptr, nullptr, ne)) ++blind;
            if (any_elsewhere && h == chunk_hash_split(nullptr, nullptr, ne)) ++blind;
            // one expert moves from wherever it was to this card: the division the word describes has changed,
            // unless it was already this card's and the CPU's at once (which `chunk_source` settles on the CPU)
            const int64_t e = (int64_t) (rng() % (uint64_t) ne);
            const char wp = peer[(size_t) e], wc = cpu[(size_t) e];
            peer[(size_t) e] = 0;
            cpu[(size_t) e] = 0;
            const bool same = chunk_source(wp != 0, wc != 0) == ChunkSource::lead;
            if ((chunk_hash_split(peer.data(), cpu.data(), ne) == h) != same) ++blind;
        }
        check(unstable == 0, "a division fingerprinted twice gave two words");
        check(blind == 0, "a division fingerprint failed to notice an expert moving");
        check(chunk_hash_split(nullptr, nullptr, 0) == kFnvOffset, "no experts, nothing handed over");
        std::printf("4. the division's fingerprint: stable, and it notices an expert moving cards\n");
    }

    // 5. the fold composes: continuing a hash over two halves is the hash of the whole, so a record that hashed its
    //    tables in pieces still means the same as one that hashed them at once
    {
        std::mt19937 rng(4242u);
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<uint8_t> b((size_t) (rng() % 900u));
            for (uint8_t& x : b) x = (uint8_t) rng();
            const size_t cut = b.size() / 2;
            const uint64_t whole = chunk_hash(b.data(), b.size());
            const uint64_t halves = chunk_hash(chunk_hash(kFnvOffset, b.data(), cut), b.data() + cut, b.size() - cut);
            check(whole == halves, "a fingerprint folded in pieces differs from the whole");
        }
        std::printf("5. the fingerprint folds: pieces and whole agree\n");
    }

    // 6. two boots that decided alike print alike, and two that did not differ at exactly the layer they parted:
    //    a stage's worth of layers, each with its own routing and division, one boot moving one expert at one layer
    {
        const int64_t layers = 24, ne = 512, tk = 40960;
        std::mt19937 rng(31337u);
        std::vector<std::vector<int32_t>> ids((size_t) layers);
        std::vector<std::vector<char>> peer((size_t) layers), cpu((size_t) layers);
        for (int64_t l = 0; l < layers; ++l) {
            ids[(size_t) l] = routing(tk, ne, rng());
            peer[(size_t) l].assign((size_t) ne, 0);
            cpu[(size_t) l].assign((size_t) ne, 0);
            for (int64_t e = 0; e < ne; ++e) {
                peer[(size_t) l][(size_t) e] = (char) (rng() % 3u == 0u ? 1 : 0);
                cpu[(size_t) l][(size_t) e] = (char) (rng() % 50u == 0u ? 1 : 0);
            }
        }
        auto word = [&](int64_t l) {
            return chunk_hash_ids(ids[(size_t) l].data(), tk) ^ chunk_hash_split(peer[(size_t) l].data(),
                                                                                 cpu[(size_t) l].data(), ne);
        };
        std::vector<uint64_t> boot_a((size_t) layers), boot_b((size_t) layers);
        for (int64_t l = 0; l < layers; ++l) {
            boot_a[(size_t) l] = word(l);
            boot_b[(size_t) l] = word(l);
        }
        check(boot_a == boot_b, "the same decisions twice over gave two boots");
        const int64_t moved = 9;
        peer[(size_t) moved][(size_t) 17] = (char) !peer[(size_t) moved][(size_t) 17];
        cpu[(size_t) moved][(size_t) 17] = 0;   // certain to be a different source than before
        int first = -1;
        for (int64_t l = 0; l < layers; ++l)
            if (word(l) != boot_a[(size_t) l]) { first = (int) l; break; }
        check(first == moved, "one expert moved at layer " + std::to_string(moved) + ", the diff opened at " +
                                  std::to_string(first));
        std::printf("6. a stage of records: alike twice over, and one moved expert opens the diff at its layer\n");
    }

    std::printf("%s\n", fails == 0 ? "chunk_dump_parity: OK" : "chunk_dump_parity: FAILED");
    return fails == 0 ? 0 : 1;
}

#include "tokenizer/bpe/BpeTrainer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace domlm::tokenizer {

namespace {

// One distinct chunk-id sequence shared by `count` corpus occurrences.
// Deduplicating identical sequences preserves exact BPE semantics (pair
// frequencies are weighted sums) while collapsing repetitive corpora.
struct Word {
    std::vector<int> ids;
    std::size_t count{0};
};

constexpr std::uint64_t EncodePair(int a, int b) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(b));
}

}  // namespace

std::size_t BpeTrainer::VecHash::operator()(const std::vector<int>& v) const noexcept {
    // FNV-1a over the raw ints; cheap and well-distributed here.
    std::size_t h = 1469598103934665603U;
    for (int x : v) {
        h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(x));
        h *= 1099511628211U;
    }
    return h;
}

BpeTrainer::BpeTrainer(std::size_t target_vocab_size,
                       domlm::tokenizer::PreTokenizerPipeline pipeline)
    : target_(target_vocab_size), pipeline_(std::move(pipeline)) {
    if (target_ < Vocabulary::kBaseSize) {
        throw std::invalid_argument("BpeTrainer: target vocab size must be >= 256");
    }
}

void BpeTrainer::train(const std::vector<std::string>& lines) {
    FreqMap freq;
    for (const auto& line : lines) {
        ingestLine(line, freq);
    }
    runMerges(freq);
}

void BpeTrainer::train(CorpusReader& reader) {
    FreqMap freq;
    std::string line;
    while (reader.nextLine(line)) {
        ingestLine(line, freq);
    }
    runMerges(freq);
}

std::map<BpeTrainer::Pair, std::size_t> BpeTrainer::countPairs(
    const std::vector<std::vector<int>>& sequences) {
    std::map<Pair, std::size_t> freq;
    for (const auto& seq : sequences) {
        for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
            ++freq[{seq[i], seq[i + 1]}];
        }
    }
    return freq;
}

void BpeTrainer::ingestLine(const std::string& line, FreqMap& freq) {
    // One SplitResult per line; each chunk becomes an independent sequence.
    // Chunks are sanitized valid UTF-8, but BPE operates on raw bytes, so a
    // multi-byte code point contributes one id per byte (0..255 identity).
    const domlm::tokenizer::SplitResult result = pipeline_.split(line);
    for (const std::string_view chunk : result.chunks) {
        if (chunk.empty()) {
            continue;
        }
        std::vector<int> seq;
        seq.reserve(chunk.size());
        for (unsigned char b : chunk) {
            seq.push_back(static_cast<int>(b));
        }
        ++freq[std::move(seq)];
    }
}

void BpeTrainer::runMerges(FreqMap& freq) {
    // Flatten to a deterministically ordered word list. Order is irrelevant
    // to results (frequencies are multiset sums) but keeps behavior stable.
    std::vector<Word> words;
    words.reserve(freq.size());
    for (auto& [seq, count] : freq) {
        words.push_back(Word{std::move(seq), count});
    }
    freq.clear();
    freq.rehash(0);
    std::sort(words.begin(), words.end(),
              [](const Word& a, const Word& b) { return a.ids < b.ids; });

    std::unordered_map<std::uint64_t, std::size_t> weighted;
    std::vector<std::uint64_t> keys;
    while (vocab_.size() < target_) {
        bool anyPair = false;
        for (const auto& w : words) {
            if (w.ids.size() >= 2) {
                anyPair = true;
                break;
            }
        }
        if (!anyPair) {
            break;  // every sequence has length < 2: nothing left to merge
        }
        weighted.clear();
        for (const auto& w : words) {
            for (std::size_t i = 0; i + 1 < w.ids.size(); ++i) {
                weighted[EncodePair(w.ids[i], w.ids[i + 1])] += w.count;
            }
        }
        // Ascending key order == ascending (id_a, id_b): a strictly-greater
        // comparison below keeps the smallest pair on frequency ties.
        keys.clear();
        keys.reserve(weighted.size());
        for (const auto& [key, count] : weighted) {
            (void)count;
            keys.push_back(key);
        }
        std::sort(keys.begin(), keys.end());
        std::uint64_t bestKey = 0;
        std::size_t bestCount = 0;
        bool haveBest = false;
        for (const std::uint64_t key : keys) {
            const std::size_t c = weighted[key];
            if (!haveBest || c > bestCount) {
                bestKey = key;
                bestCount = c;
                haveBest = true;
            }
        }
        if (!haveBest) {
            break;
        }
        const int bestA = static_cast<int>(bestKey >> 32);
        const int bestB = static_cast<int>(bestKey & 0xFFFFFFFFU);
        const int newId = vocab_.addMerge(bestA, bestB);
        for (auto& w : words) {
            if (w.ids.size() < 2) {
                continue;
            }
            std::vector<int> merged;
            merged.reserve(w.ids.size());
            for (std::size_t i = 0; i < w.ids.size();) {
                if (i + 1 < w.ids.size() && w.ids[i] == bestA && w.ids[i + 1] == bestB) {
                    merged.push_back(newId);
                    i += 2;  // non-overlapping, left to right
                } else {
                    merged.push_back(w.ids[i]);
                    ++i;
                }
            }
            w.ids = std::move(merged);
        }
    }
}

}  // namespace domlm::tokenizer

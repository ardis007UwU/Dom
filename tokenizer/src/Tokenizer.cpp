#include "tokenizer/bpe/Tokenizer.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/bpe/Serialization.hpp"

namespace domlm::tokenizer {

Tokenizer::Tokenizer(Vocabulary vocab, domlm::tokenizer::PreTokenizerPipeline pipeline)
    : vocab_(std::move(vocab)), pipeline_(std::move(pipeline)) {}

Tokenizer Tokenizer::load(const std::string& vocab_path, const std::string& merges_path,
                          const std::string& ruleset) {
    Vocabulary vocab = LoadVocabulary(vocab_path, merges_path);
    domlm::tokenizer::PreTokenizerPipeline pipeline;
    if (ruleset != "cl100k_base" && ruleset != "o200k_base") {
        throw std::invalid_argument("Tokenizer: unknown ruleset '" + ruleset + "'");
    }
    pipeline.loadPreTokenizerRuleset(ruleset);
    return Tokenizer(std::move(vocab), std::move(pipeline));
}

std::vector<int> Tokenizer::encode(std::string_view text) const {
    std::vector<int> out;
    encodeInto(text, out);
    return out;
}

void Tokenizer::encodeInto(std::string_view text, std::vector<int>& out) const {
    const domlm::tokenizer::SplitResult result = pipeline_.split(text);
    for (const std::string_view chunk : result.chunks) {
        if (!chunk.empty()) {
            encodeChunk(chunk, out);
        }
    }
}

std::string Tokenizer::decode(std::span<const int> ids) const {
    std::size_t total = 0;
    for (int id : ids) {
        total += vocab_.bytesOf(id).size();  // range-checked
    }
    std::string out;
    out.reserve(total);
    for (int id : ids) {
        out += vocab_.bytesOf(id);
    }
    return out;
}

void Tokenizer::encodeChunk(std::string_view chunk, std::vector<int>& out) const {
    // Seed: one base id per raw byte (base vocab is the byte identity).
    std::vector<int> work;
    work.reserve(chunk.size());
    for (unsigned char b : chunk) {
        work.push_back(static_cast<int>(b));
    }
    // Second buffer sized once, reused across rounds via swap.
    std::vector<int> next;
    next.reserve(chunk.size());

    for (;;) {
        // Single scan: lowest-rank (smallest new_id) pair present wins.
        int bestA = -1;
        int bestB = -1;
        int bestRank = std::numeric_limits<int>::max();
        for (std::size_t i = 0; i + 1 < work.size(); ++i) {
            const int rank = vocab_.idOfPair(work[i], work[i + 1]);
            if (rank >= 0 && rank < bestRank) {
                bestRank = rank;
                bestA = work[i];
                bestB = work[i + 1];
            }
        }
        if (bestA < 0) {
            break;  // no mergeable pair left in this chunk
        }
        next.clear();
        for (std::size_t i = 0; i < work.size();) {
            if (i + 1 < work.size() && work[i] == bestA && work[i + 1] == bestB) {
                next.push_back(bestRank);
                i += 2;  // non-overlapping, left to right (matches training)
            } else {
                next.push_back(work[i]);
                ++i;
            }
        }
        work.swap(next);
    }
    out.insert(out.end(), work.begin(), work.end());
}

}  // namespace domlm::tokenizer

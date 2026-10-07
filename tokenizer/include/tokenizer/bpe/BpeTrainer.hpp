#pragma once

// Phase 2: BPE training engine.
//
// Pipeline per input line:
//   1. line -> domlm::tokenizer::PreTokenizerPipeline::split(line)
//   2. each chunk -> byte-id sequence (one id per raw byte, base vocab)
//   3. count adjacent pairs STRICTLY WITHIN each chunk sequence
//      (never across chunk boundaries)
//   4. merge the most frequent pair; ties broken by lexicographically
//      smallest (id_a, id_b); new id = 256, 257, ... until target size V
//      or no pairs remain
//   5. rewrite every sequence left-to-right (non-overlapping) and repeat.

#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizer/bpe/CorpusReader.hpp"
#include "tokenizer/bpe/Vocabulary.hpp"
#include "tokenizer/pre/PreTokenizerPipeline.hpp"

namespace domlm::tokenizer {

class BpeTrainer {
public:
    static constexpr std::size_t kDefaultVocabSize = 32000;

    explicit BpeTrainer(std::size_t target_vocab_size = kDefaultVocabSize,
                        domlm::tokenizer::PreTokenizerPipeline pipeline = {});

    BpeTrainer(const BpeTrainer&) = default;
    BpeTrainer& operator=(const BpeTrainer&) = default;
    BpeTrainer(BpeTrainer&&) noexcept = default;
    BpeTrainer& operator=(BpeTrainer&&) noexcept = default;
    ~BpeTrainer() = default;

    [[nodiscard]] std::size_t targetVocabSize() const noexcept { return target_; }
    [[nodiscard]] const Vocabulary& vocabulary() const noexcept { return vocab_; }
    [[nodiscard]] domlm::tokenizer::PreTokenizerPipeline& pipeline() noexcept { return pipeline_; }

    /// Train on in-memory lines (one corpus line per element).
    void train(const std::vector<std::string>& lines);
    /// Train by streaming lines from an open reader (one getline at a time).
    void train(CorpusReader& reader);

    using Pair = std::pair<int, int>;

    /// Frequency of every adjacent id pair over chunk sequences.
    /// Exposed for testing the within-chunk invariant.
    [[nodiscard]] static std::map<Pair, std::size_t> countPairs(
        const std::vector<std::vector<int>>& sequences);

private:
    struct VecHash {
        std::size_t operator()(const std::vector<int>& v) const noexcept;
    };
    using FreqMap = std::unordered_map<std::vector<int>, std::size_t, VecHash>;

    std::size_t target_;
    Vocabulary vocab_;
    domlm::tokenizer::PreTokenizerPipeline pipeline_;

    void runMerges(FreqMap& freq);
    void ingestLine(const std::string& line, FreqMap& freq);
};

}  // namespace domlm::tokenizer

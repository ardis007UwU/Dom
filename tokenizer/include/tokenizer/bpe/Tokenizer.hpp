#pragma once

// Phase 3: inference-time BPE tokenizer (encode/decode).
//
// Encoding mirrors training exactly:
//   1. text -> PreTokenizerPipeline::split(text); each chunk is independent
//      (merges never cross chunk boundaries).
//   2. chunk bytes -> base ids (one id per raw byte).
//   3. repeatedly merge the adjacent pair with the lowest merge rank
//      (smallest new_id in the learned rule list = earliest learned),
//      rewriting all non-overlapping occurrences left-to-right, until no
//      adjacent pair carries a rank.
//
// Throughput layout:
//   - chunk text is consumed via std::string_view (zero copy; SplitResult
//     owns the bytes).
//   - token ids are flat, contiguous int buffers (SIMD/GPU-warp friendly).
//   - per chunk at most two scratch buffers, each sized once to the chunk
//     length and reused across merge rounds (swap, no reallocation).
//   - encodeInto() appends into a caller-owned buffer so batch loops run
//     with zero steady-state allocations after warmup.
//
// Decoding concatenates token byte payloads; the result is a raw byte
// stream and is NOT guaranteed to be valid UTF-8 for arbitrary id input.

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/bpe/Vocabulary.hpp"
#include "tokenizer/pre/PreTokenizerPipeline.hpp"

namespace domlm::tokenizer {

class Tokenizer {
public:
    explicit Tokenizer(Vocabulary vocab,
                       domlm::tokenizer::PreTokenizerPipeline pipeline = {});

    Tokenizer(const Tokenizer&) = default;
    Tokenizer& operator=(const Tokenizer&) = default;
    Tokenizer(Tokenizer&&) noexcept = default;
    Tokenizer& operator=(Tokenizer&&) noexcept = default;
    ~Tokenizer() = default;

    /// Builds a Tokenizer from train_bpe artifacts. Throws
    /// std::runtime_error on IO/parse errors, std::invalid_argument on a
    /// bad ruleset name.
    [[nodiscard]] static Tokenizer load(const std::string& vocab_path,
                                        const std::string& merges_path,
                                        const std::string& ruleset = "cl100k_base");

    [[nodiscard]] std::size_t vocabSize() const noexcept { return vocab_.size(); }
    [[nodiscard]] const Vocabulary& vocabulary() const noexcept { return vocab_; }
    [[nodiscard]] domlm::tokenizer::PreTokenizerPipeline& pipeline() noexcept { return pipeline_; }

    /// Encodes text to token ids (fresh output buffer).
    [[nodiscard]] std::vector<int> encode(std::string_view text) const;
    /// Appends token ids to `out` (reuse `out` across calls for zero
    /// steady-state allocation).
    void encodeInto(std::string_view text, std::vector<int>& out) const;
    /// Decodes ids back to the raw byte stream. Throws std::out_of_range
    /// on unknown ids.
    [[nodiscard]] std::string decode(std::span<const int> ids) const;

private:
    Vocabulary vocab_;
    domlm::tokenizer::PreTokenizerPipeline pipeline_;

    void encodeChunk(std::string_view chunk, std::vector<int>& out) const;
};

}  // namespace domlm::tokenizer

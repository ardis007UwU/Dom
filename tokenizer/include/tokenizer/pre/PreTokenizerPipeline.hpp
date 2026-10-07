#pragma once

#include "tokenizer/pre/UnicodeNormalizer.hpp"
#include "tokenizer/pre/PreTokenizer.hpp"

#include <string_view>
#include <string>

namespace tokenizer::pre {

/**
 * Thin pipeline composing Unicode normalization and pre-tokenization.
 * Normalization is opt-in (default Identity). Always copies to storage.
 */
class PreTokenizerPipeline {
public:
    PreTokenizerPipeline();
    ~PreTokenizerPipeline() = default;

    PreTokenizerPipeline(const PreTokenizerPipeline&) = default;
    PreTokenizerPipeline& operator=(const PreTokenizerPipeline&) = default;
    PreTokenizerPipeline(PreTokenizerPipeline&&) noexcept = default;
    PreTokenizerPipeline& operator=(PreTokenizerPipeline&&) noexcept = default;

    /**
     * Set normalization mode.
     */
    void setNormalizationMode(NormalizationMode mode);

    /**
     * Get normalization mode.
     */
    [[nodiscard]] NormalizationMode getNormalizationMode() const noexcept;

    /**
     * Load pre-tokenizer rule set.
     */
    void loadPreTokenizerRuleset(const std::string& name);

    /**
     * Set pre-tokenizer pattern directly.
     */
    void setPreTokenizerPattern(std::string_view pattern);

    /**
     * Process input: normalize (if not Identity) then split.
     * Returns SplitResult with storage containing the processed text
     * and chunks pointing into storage.
     */
    [[nodiscard]] SplitResult process(std::string_view input) const;

    /**
     * Alias of process() kept for call sites written against the
     * `PreTokenizerPipeline::split()` spelling.
     */
    [[nodiscard]] SplitResult split(std::string_view input) const;

private:
    UnicodeNormalizer normalizer_;
    PreTokenizer preTokenizer_;
};

} // namespace tokenizer::pre

#include "tokenizer/pre/PreTokenizerPipeline.hpp"

namespace tokenizer::pre {

PreTokenizerPipeline::PreTokenizerPipeline() = default;

void PreTokenizerPipeline::setNormalizationMode(NormalizationMode mode) {
    normalizer_.setMode(mode);
}

NormalizationMode PreTokenizerPipeline::getNormalizationMode() const noexcept {
    return normalizer_.getMode();
}

void PreTokenizerPipeline::loadPreTokenizerRuleset(const std::string& name) {
    preTokenizer_.loadRuleset(name);
}

void PreTokenizerPipeline::setPreTokenizerPattern(std::string_view pattern) {
    preTokenizer_.setPattern(pattern);
}

SplitResult PreTokenizerPipeline::process(std::string_view input) const {
    // Normalize first (opt-in; Identity returns copy)
    std::string normalized = normalizer_.normalize(input);
    // Split on normalized text
    return preTokenizer_.split(std::string_view(normalized));
}

SplitResult PreTokenizerPipeline::split(std::string_view input) const {
    return process(input);
}

} // namespace tokenizer::pre

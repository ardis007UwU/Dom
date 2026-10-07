#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <cstddef>

struct pcre2_code;

namespace tokenizer::pre {

/**
 * Result of pre-tokenization containing storage and chunk views.
 * INVARIANT: chunks always point into storage; storage is owned by this struct.
 */
struct SplitResult {
    std::string storage;
    std::vector<std::string_view> chunks;
};

/**
 * PCRE2-based pre-tokenizer with named rule sets matching Tiktoken byte-for-byte.
 * Compiled with PCRE2_UTF | PCRE2_UCP. No PCRE2_NO_UTF_CHECK.
 */
class PreTokenizer {
public:
    // Pre-defined rule sets (byte-for-byte per tiktoken openai_public.py).
    static constexpr const char* CL100K_BASE =
        R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s)";
    static constexpr const char* O200K_BASE =
        R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/*]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

    PreTokenizer();
    ~PreTokenizer();

    PreTokenizer(const PreTokenizer&) = delete;
    PreTokenizer& operator=(const PreTokenizer&) = delete;
    PreTokenizer(PreTokenizer&&) noexcept;
    PreTokenizer& operator=(PreTokenizer&&) noexcept;

    /**
     * Set the regex pattern. Compiles with PCRE2_UTF | PCRE2_UCP
     * (never PCRE2_NO_UTF_CHECK). Throws std::runtime_error on failure.
     */
    void setPattern(std::string_view pattern);

    /**
     * Split input text using current pattern.
     * Defined invalid-UTF-8 behavior (unconditional): input is first
     * sanitized (invalid -> single U+FFFD each, see UnicodeNormalizer docs);
     * SplitResult::storage holds the sanitized copy and chunks alias it.
     * Concatenated chunks equal storage exactly. Never aliases caller memory.
     */
    [[nodiscard]] SplitResult split(std::string_view input) const;

    /**
     * Load a named rule set: "cl100k_base" or "o200k_base".
     */
    void loadRuleset(const std::string& name);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tokenizer::pre

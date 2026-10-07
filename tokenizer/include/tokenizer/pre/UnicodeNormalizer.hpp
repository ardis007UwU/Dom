#pragma once

#include <string>
#include <string_view>
#include <cstddef>
#include <memory>

namespace tokenizer::pre {

/**
 * Unicode normalization modes supported by ICU.
 * Default mode is Identity (no normalization).
 */
enum class NormalizationMode {
    Identity,  // No normalization (pass-through)
    NFC,       // Canonical composition
    NFD,       // Canonical decomposition
    NFKC,      // Compatibility decomposition followed by canonical composition
    NFKD       // Compatibility decomposition
};

/**
 * ICU-backed Unicode normalizer.
 * Normalization is strictly opt-in - default is Identity.
 */
class UnicodeNormalizer {
public:
    UnicodeNormalizer();
    ~UnicodeNormalizer();
    
    UnicodeNormalizer(const UnicodeNormalizer&) = delete;
    UnicodeNormalizer& operator=(const UnicodeNormalizer&) = delete;
    UnicodeNormalizer(UnicodeNormalizer&&) noexcept;
    UnicodeNormalizer& operator=(UnicodeNormalizer&&) noexcept;

    /**
     * Set the normalization mode. Defaults to Identity.
     */
    void setMode(NormalizationMode mode);

    /**
     * Get the current normalization mode.
     */
    [[nodiscard]] NormalizationMode getMode() const noexcept;

    /**
     * Normalize input UTF-8 string.
     * Defined invalid-UTF-8 behavior (unconditional): the input byte stream
     * is first sanitized per RFC 3629; every invalid byte / truncated /
     * overlong / surrogate / out-of-range sequence is replaced by exactly
     * one U+FFFD ("\xEF\xBF\xBD"), scanning resumes at the next byte, and
     * valid sequences pass through unchanged. Output is always valid UTF-8.
     * If mode is Identity, returns the sanitized copy (still always copies,
     * never aliases caller memory). No UB, no crashes.
     */
    [[nodiscard]] std::string normalize(std::string_view input) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tokenizer::pre

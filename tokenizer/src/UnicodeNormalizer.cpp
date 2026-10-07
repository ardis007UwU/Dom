#include "tokenizer/pre/UnicodeNormalizer.hpp"
#include "Utf8Sanitize.hpp"

#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/utypes.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tokenizer::pre {

class UnicodeNormalizer::Impl {
public:
    Impl() {
        UErrorCode status = U_ZERO_ERROR;
        nfc = unorm2_getNFCInstance(&status);
        if (U_FAILURE(status) || nfc == nullptr) {
            throw std::runtime_error("UnicodeNormalizer: failed to get NFC instance");
        }
        nfd = unorm2_getNFDInstance(&status);
        if (U_FAILURE(status) || nfd == nullptr) {
            throw std::runtime_error("UnicodeNormalizer: failed to get NFD instance");
        }
        nfkc = unorm2_getNFKCInstance(&status);
        if (U_FAILURE(status) || nfkc == nullptr) {
            throw std::runtime_error("UnicodeNormalizer: failed to get NFKC instance");
        }
        nfkd = unorm2_getNFKDInstance(&status);
        if (U_FAILURE(status) || nfkd == nullptr) {
            throw std::runtime_error("UnicodeNormalizer: failed to get NFKD instance");
        }
        mode = NormalizationMode::Identity;
    }

    ~Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    NormalizationMode mode{NormalizationMode::Identity};
    // Non-owning pointers to ICU's cached singleton instances. Must not be freed.
    const UNormalizer2* nfc{nullptr};
    const UNormalizer2* nfd{nullptr};
    const UNormalizer2* nfkc{nullptr};
    const UNormalizer2* nfkd{nullptr};
};

UnicodeNormalizer::UnicodeNormalizer() : impl_(std::make_unique<Impl>()) {}

UnicodeNormalizer::~UnicodeNormalizer() = default;

UnicodeNormalizer::UnicodeNormalizer(UnicodeNormalizer&&) noexcept = default;
UnicodeNormalizer& UnicodeNormalizer::operator=(UnicodeNormalizer&&) noexcept = default;

void UnicodeNormalizer::setMode(NormalizationMode mode) {
    impl_->mode = mode;
}

NormalizationMode UnicodeNormalizer::getMode() const noexcept {
    return impl_->mode;
}

std::string UnicodeNormalizer::normalize(std::string_view input) const {
    // Unconditional sanitization: output is always valid UTF-8, invalid
    // sequences replaced by U+FFFD per detail::SanitizeUtf8. Identity mode
    // still copies through the sanitizer (never aliases caller memory).
    std::string clean = detail::SanitizeUtf8(input);
    if (impl_->mode == NormalizationMode::Identity) {
        return clean;
    }
    if (clean.empty()) {
        return clean;
    }
    if (clean.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("UnicodeNormalizer: input too large for ICU (int32 limit)");
    }

    const UNormalizer2* norm = nullptr;
    switch (impl_->mode) {
        case NormalizationMode::NFC:
            norm = impl_->nfc;
            break;
        case NormalizationMode::NFD:
            norm = impl_->nfd;
            break;
        case NormalizationMode::NFKC:
            norm = impl_->nfkc;
            break;
        case NormalizationMode::NFKD:
            norm = impl_->nfkd;
            break;
        case NormalizationMode::Identity:
            return clean;
    }

    const int32_t utf8Len = static_cast<int32_t>(clean.size());

    // UTF-8 -> UTF-16. clean is valid UTF-8, so strict conversion must succeed.
    UErrorCode status = U_ZERO_ERROR;
    int32_t utf16Cap = 0;
    u_strFromUTF8(nullptr, 0, &utf16Cap, clean.data(), utf8Len, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) {
        // Defensive: sanitized input must preflight cleanly.
        return clean;
    }
    status = U_ZERO_ERROR;
    std::vector<UChar> utf16(static_cast<std::size_t>(utf16Cap) + 1U);
    int32_t utf16Len = 0;
    u_strFromUTF8(utf16.data(), utf16Cap + 1, &utf16Len, clean.data(), utf8Len, &status);
    if (U_FAILURE(status)) {
        return clean;
    }

    // Normalize UTF-16 -> UTF-16.
    status = U_ZERO_ERROR;
    int32_t normCap = unorm2_normalize(norm, utf16.data(), utf16Len, nullptr, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        return clean;
    }
    status = U_ZERO_ERROR;
    std::vector<UChar> norm16(static_cast<std::size_t>(normCap) + 1U);
    const int32_t normLen =
        unorm2_normalize(norm, utf16.data(), utf16Len, norm16.data(), normCap + 1, &status);
    if (U_FAILURE(status)) {
        return clean;
    }

    // UTF-16 -> UTF-8.
    status = U_ZERO_ERROR;
    int32_t outCap = 0;
    u_strToUTF8(nullptr, 0, &outCap, norm16.data(), normLen, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) {
        return clean;
    }
    status = U_ZERO_ERROR;
    std::string out(static_cast<std::size_t>(outCap), '\0');
    int32_t outLen = 0;
    u_strToUTF8(out.data(), outCap, &outLen, norm16.data(), normLen, &status);
    if (U_FAILURE(status)) {
        return clean;
    }
    out.resize(static_cast<std::size_t>(outLen));
    return out;
}

}  // namespace tokenizer::pre

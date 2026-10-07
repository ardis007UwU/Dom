#include "tokenizer/pre/PreTokenizer.hpp"
#include "Utf8Sanitize.hpp"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tokenizer::pre {

namespace {

// RAII wrapper for pcre2_match_data. No raw new/delete.
struct MatchDataDeleter {
    void operator()(pcre2_match_data* p) const noexcept {
        if (p != nullptr) {
            pcre2_match_data_free(p);
        }
    }
};
using MatchDataPtr = std::unique_ptr<pcre2_match_data, MatchDataDeleter>;

}  // namespace

class PreTokenizer::Impl {
public:
    Impl() : code(nullptr, pcre2_code_free) {}

    ~Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)> code;
    std::string pattern;
};

PreTokenizer::PreTokenizer() : impl_(std::make_unique<Impl>()) {
    loadRuleset("cl100k_base");
}

PreTokenizer::~PreTokenizer() = default;

PreTokenizer::PreTokenizer(PreTokenizer&&) noexcept = default;
PreTokenizer& PreTokenizer::operator=(PreTokenizer&&) noexcept = default;

void PreTokenizer::setPattern(std::string_view pattern) {
    impl_->pattern.assign(pattern.data(), pattern.size());

    PCRE2_SPTR patternPtr = reinterpret_cast<PCRE2_SPTR>(impl_->pattern.c_str());
    PCRE2_SIZE errorOffset = 0;
    int errorNumber = 0;
    // Required: PCRE2_UTF | PCRE2_UCP. Never PCRE2_NO_UTF_CHECK.
    pcre2_code* re = pcre2_compile(patternPtr, PCRE2_ZERO_TERMINATED,
                                   PCRE2_UTF | PCRE2_UCP, &errorNumber, &errorOffset, nullptr);
    if (re == nullptr) {
        std::array<char, 256> msg{};
        pcre2_get_error_message(errorNumber, reinterpret_cast<PCRE2_UCHAR*>(msg.data()),
                                msg.size());
        throw std::runtime_error(std::string("PreTokenizer: PCRE2 compile failed at offset ") +
                                 std::to_string(errorOffset) + ": " + msg.data());
    }
    impl_->code.reset(re);
}

void PreTokenizer::loadRuleset(const std::string& name) {
    if (name == "cl100k_base") {
        setPattern(CL100K_BASE);
    } else if (name == "o200k_base") {
        setPattern(O200K_BASE);
    } else {
        throw std::invalid_argument("PreTokenizer: unknown ruleset '" + name +
                                    "' (expected 'cl100k_base' or 'o200k_base')");
    }
}

SplitResult PreTokenizer::split(std::string_view input) const {
    SplitResult result;
    // INVARIANT: always copy (sanitized) text into storage, even for Identity
    // / empty / no-match cases, so chunks can safely alias storage.
    // Sanitization is unconditional: invalid UTF-8 -> U+FFFD, output valid.
    result.storage = detail::SanitizeUtf8(input);

    if (impl_->code == nullptr) {
        throw std::logic_error("PreTokenizer: no pattern compiled");
    }
    if (result.storage.empty()) {
        return result;
    }

    MatchDataPtr matchData(pcre2_match_data_create_from_pattern(impl_->code.get(), nullptr));
    if (matchData == nullptr) {
        throw std::runtime_error("PreTokenizer: failed to create PCRE2 match data");
    }

    // Freeze the buffer: no further mutation of storage after this point,
    // so string_views into it remain valid.
    const char* const data = result.storage.data();
    const PCRE2_SIZE subjectLength = static_cast<PCRE2_SIZE>(result.storage.size());
    PCRE2_SPTR subject = reinterpret_cast<PCRE2_SPTR>(data);
    PCRE2_SIZE currentOffset = 0;

    while (currentOffset < subjectLength) {
        const int rc = pcre2_match(impl_->code.get(), subject, subjectLength, currentOffset, 0,
                                   matchData.get(), nullptr);
        if (rc == PCRE2_ERROR_NOMATCH) {
            // No match at/after currentOffset: emit the remainder to preserve
            // the concat(chunks) == storage invariant, then stop.
            result.chunks.emplace_back(data + currentOffset, subjectLength - currentOffset);
            break;
        }
        if (rc < 0) {
            // Sanitized input must not trigger UTF errors; any other PCRE2
            // error is handled defensively by consuming one code point so we
            // never loop forever and never drop bytes (invariant preserved).
            const auto len =
                detail::Utf8CharLen(static_cast<unsigned char>(data[currentOffset]));
            result.chunks.emplace_back(data + currentOffset, len);
            currentOffset += static_cast<PCRE2_SIZE>(len);
            continue;
        }
        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(matchData.get());
        const PCRE2_SIZE matchStart = ovector[0];
        const PCRE2_SIZE matchEnd = ovector[1];

        if (matchEnd <= matchStart) {
            // Empty match: consume one code point to make progress.
            const auto len =
                detail::Utf8CharLen(static_cast<unsigned char>(data[currentOffset]));
            result.chunks.emplace_back(data + currentOffset, len);
            currentOffset += static_cast<PCRE2_SIZE>(len);
            continue;
        }
        if (matchStart > currentOffset) {
            // Gap between previous end and this match start (defensive; the
            // tiktoken patterns partition fully, but gaps must not be dropped).
            result.chunks.emplace_back(data + currentOffset, matchStart - currentOffset);
        }
        result.chunks.emplace_back(data + matchStart, matchEnd - matchStart);
        currentOffset = matchEnd;
    }

    return result;
}

}  // namespace tokenizer::pre

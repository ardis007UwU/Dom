#pragma once

// Internal UTF-8 sanitizer shared by UnicodeNormalizer and PreTokenizer.
//
// Defined behavior (documented):
//  - Input is a raw byte stream viewed as std::string_view (no NUL dependence).
//  - Output is always valid UTF-8 (RFC 3629).
//  - Every valid sequence passes through unchanged.
//  - Every byte position that cannot begin a valid sequence (stray
//    continuation 0x80..0xBF, overlong leads 0xC0..0xC1, out-of-range
//    leads 0xF5..0xFF), and every lead byte of a truncated, non-continuation,
//    overlong, surrogate (U+D800..U+DFFF), or out-of-range (> U+10FFFF)
//    sequence is replaced by exactly one U+FFFD ("\xEF\xBF\xBD").
//  - Scanning resumes at the next byte after the offending lead byte.
//    Example: "hello\xFFworld" -> "hello\xEF\xBF\xBDworld".
//  - No UB, no crashes, no exceptions, no raw new/delete (RAII only:
//    std::string storage owned by caller).
//
// This runs unconditionally, including Identity normalization mode and
// pre-tokenizer splitting, so PCRE2 (compiled WITHOUT PCRE2_NO_UTF_CHECK)
// never observes invalid UTF-8.

#include <cstddef>
#include <string>
#include <string_view>

namespace tokenizer::pre::detail {

inline std::string SanitizeUtf8(std::string_view in) {
    static constexpr char kReplacement[] = "\xEF\xBF\xBD";  // U+FFFD
    std::string out;
    out.reserve(in.size());
    const std::size_t n = in.size();
    std::size_t i = 0;
    auto byteAt = [&](std::size_t k) -> unsigned char {
        return static_cast<unsigned char>(in[k]);
    };
    auto isCont = [&](unsigned char b) -> bool {
        return b >= 0x80 && b <= 0xBF;
    };

    while (i < n) {
        const unsigned char c = byteAt(i);
        if (c <= 0x7F) {
            out.push_back(static_cast<char>(c));
            ++i;
        } else if (c >= 0xC2 && c <= 0xDF) {
            // 2-byte sequence: C2..DF + 80..BF
            if (i + 1 < n && isCont(byteAt(i + 1))) {
                out.append(in.data() + i, 2);
                i += 2;
            } else {
                out.append(kReplacement, 3);
                ++i;
            }
        } else if (c >= 0xE0 && c <= 0xEF) {
            // 3-byte sequence with overlong/surrogate guards.
            const bool have = (i + 2 < n);
            bool ok = false;
            if (have) {
                const unsigned char c1 = byteAt(i + 1);
                const unsigned char c2 = byteAt(i + 2);
                if (isCont(c1) && isCont(c2)) {
                    if (c == 0xE0) {
                        ok = (c1 >= 0xA0 && c1 <= 0xBF);  // no overlong
                    } else if (c == 0xED) {
                        ok = (c1 >= 0x80 && c1 <= 0x9F);  // no surrogates
                    } else {
                        ok = true;
                    }
                }
            }
            if (ok) {
                out.append(in.data() + i, 3);
                i += 3;
            } else {
                out.append(kReplacement, 3);
                ++i;
            }
        } else if (c >= 0xF0 && c <= 0xF4) {
            // 4-byte sequence with overlong/range guards (<= U+10FFFF).
            const bool have = (i + 3 < n);
            bool ok = false;
            if (have) {
                const unsigned char c1 = byteAt(i + 1);
                const unsigned char c2 = byteAt(i + 2);
                const unsigned char c3 = byteAt(i + 3);
                if (isCont(c1) && isCont(c2) && isCont(c3)) {
                    if (c == 0xF0) {
                        ok = (c1 >= 0x90 && c1 <= 0xBF);
                    } else if (c == 0xF4) {
                        ok = (c1 >= 0x80 && c1 <= 0x8F);
                    } else {
                        ok = true;
                    }
                }
            }
            if (ok) {
                out.append(in.data() + i, 4);
                i += 4;
            } else {
                out.append(kReplacement, 3);
                ++i;
            }
        } else {
            // Stray continuation (80..BF), overlong C0/C1, F5..FF.
            out.append(kReplacement, 3);
            ++i;
        }
    }
    return out;
}

// Length in bytes of the UTF-8 code point starting at lead byte `c`.
// Caller must guarantee `c` begins a valid sequence (true for sanitized text).
inline std::size_t Utf8CharLen(unsigned char c) noexcept {
    if (c <= 0x7F) {
        return 1;
    }
    if (c >= 0xC2 && c <= 0xDF) {
        return 2;
    }
    if (c >= 0xE0 && c <= 0xEF) {
        return 3;
    }
    return 4;  // F0..F4 for sanitized text
}

}  // namespace tokenizer::pre::detail

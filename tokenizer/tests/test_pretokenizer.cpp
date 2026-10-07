#include <gtest/gtest.h>

#include "tokenizer/pre/PreTokenizer.hpp"
#include "tokenizer/pre/PreTokenizerPipeline.hpp"
#include "tokenizer/pre/UnicodeNormalizer.hpp"

#include <string>
#include <string_view>
#include <vector>

using namespace domlm::tokenizer;

namespace {

// Concatenated chunks must equal SplitResult::storage exactly.
std::string Concat(const SplitResult& r) {
    std::string out;
    for (const auto& c : r.chunks) {
        out.append(c.data(), c.size());
    }
    return out;
}

// Every chunk must alias storage memory (no dangling references).
void ExpectAliasesStorage(const SplitResult& r) {
    const char* begin = r.storage.data();
    const char* end = begin + r.storage.size();
    for (const auto& c : r.chunks) {
        if (c.empty()) {
            continue;
        }
        EXPECT_GE(c.data(), begin);
        EXPECT_LE(c.data() + c.size(), end);
    }
}

void ExpectChunks(const SplitResult& r, const std::vector<std::string>& expected) {
    ASSERT_EQ(r.chunks.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(std::string(r.chunks[i]), expected[i]) << "mismatch at index " << i;
    }
    EXPECT_EQ(Concat(r), r.storage);
    ExpectAliasesStorage(r);
}

}  // namespace

TEST(PreTokenizerTest, Newlines) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    ExpectChunks(tok.split("a\n\nb"), {"a", "\n\n", "b"});
    ExpectChunks(tok.split("a\nb"), {"a", "\n", "b"});
    ExpectChunks(tok.split("\n"), {"\n"});
    ExpectChunks(tok.split("\n\n"), {"\n\n"});
}

TEST(PreTokenizerTest, UppercaseContractions) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    // cl100k splits the ASCII apostrophe off: IT | 'S
    ExpectChunks(tok.split("IT'S"), {"IT", "'S"});
}

TEST(PreTokenizerTest, DatePaths) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    // \p{N}{1,3}+ is greedy possessive: "2024" -> "202" + "4".
    ExpectChunks(tok.split("2024/01/15"), {"202", "4", "/", "01", "/", "15"});
}

TEST(PreTokenizerTest, NonBreakingSpace) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    // U+00A0 encoded as C2 A0. Requires PCRE2_UTF (one code point, never
    // split into stray bytes) and PCRE2_UCP (\s / \p{L} Unicode-aware).
    // cl100k alt-2 consumes the NBSP as the optional leading symbol of the
    // following letter run, hence 2 chunks (byte-exact under UCP).
    std::string input = std::string("hello") + "\xC2\xA0" + "world";
    auto r = tok.split(input);
    ExpectChunks(r, {std::string("hello"), std::string("\xC2\xA0") + "world"});
    // The NBSP bytes must survive intact (C2 A0) inside the second chunk.
    ASSERT_EQ(r.chunks.size(), 2U);
    EXPECT_EQ(static_cast<unsigned char>(r.chunks[1][0]), 0xC2);
    EXPECT_EQ(static_cast<unsigned char>(r.chunks[1][1]), 0xA0);
}

TEST(PreTokenizerTest, LeadingAndConsecutiveSpaces) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    ExpectChunks(tok.split("   hello  world"), {"  ", " hello", " ", " world"});
    ExpectChunks(tok.split("hello world"), {"hello", " world"});
    ExpectChunks(tok.split("  a"), {" ", " a"});
    ExpectChunks(tok.split("a  "), {"a", "  "});
    ExpectChunks(tok.split("   "), {"   "});
    ExpectChunks(tok.split(" a"), {" a"});
    ExpectChunks(tok.split("a "), {"a", " "});
}

TEST(PreTokenizerTest, CJKCharacters) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    // CJK Unified Ideographs are \p{L}: contiguous letters stay together.
    ExpectChunks(tok.split("hello\xE4\xB8\x96\xE7\x95\x8C"), {"hello\xE4\xB8\x96\xE7\x95\x8C"});
    ExpectChunks(tok.split("\xE4\xB8\x96\xE7\x95\x8C"), {"\xE4\xB8\x96\xE7\x95\x8C"});
}

TEST(PreTokenizerTest, EmojiHandling) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    // U+1F600 is not \p{L}/\p{N}/\s: alt-2 attaches it to the letter run.
    ExpectChunks(tok.split("hi\xF0\x9F\x98\x80there"), {"hi", "\xF0\x9F\x98\x80there"});
}

TEST(PreTokenizerTest, WhitespaceOnly) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    // cl100k trailing \s++$ folds the whole whitespace-only buffer into one.
    ExpectChunks(tok.split("   \n  "), {"   \n  "});
    ExpectChunks(tok.split("   "), {"   "});
}

TEST(PreTokenizerTest, EmptyString) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    auto r = tok.split("");
    EXPECT_TRUE(r.chunks.empty());
    EXPECT_EQ(r.storage, "");
}

TEST(PreTokenizerTest, InvalidUtf8Bytes) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    // 0xFF can never start a valid sequence -> single U+FFFD (EF BF BD).
    std::string bad = "hello";
    bad.push_back(static_cast<char>(0xFF));
    bad += "world";
    auto r = tok.split(bad);
    const std::string fffd = "\xEF\xBF\xBD";
    ExpectChunks(r, {"hello", fffd + "world"});
    EXPECT_EQ(r.storage, std::string("hello") + fffd + "world");

    // Lone continuation byte.
    std::string lone(1, static_cast<char>(0x80));
    auto r2 = tok.split(lone);
    EXPECT_EQ(r2.storage, fffd);
    EXPECT_EQ(Concat(r2), r2.storage);
    ExpectAliasesStorage(r2);

    // Truncated 3-byte sequence (E2 82 missing final byte): each offending
    // lead byte yields one U+FFFD; scanning resumes at the next byte.
    std::string trunc = std::string("a") + static_cast<char>(0xE2) + static_cast<char>(0x82);
    auto r3 = tok.split(trunc);
    EXPECT_EQ(r3.storage, std::string("a") + fffd + fffd);
    EXPECT_EQ(Concat(r3), r3.storage);
    ExpectAliasesStorage(r3);

    // Overlong "/" (C0 AF) and surrogate U+D800 (ED A0 80): rejected.
    std::string over = std::string("a") + static_cast<char>(0xC0) + static_cast<char>(0xAF);
    auto r4 = tok.split(over);
    EXPECT_EQ(Concat(r4), r4.storage);
    EXPECT_NE(r4.storage.find(fffd), std::string::npos);
}

TEST(PreTokenizerTest, NormalizerDefaultsToIdentity) {
    UnicodeNormalizer n;
    EXPECT_EQ(n.getMode(), NormalizationMode::Identity);
    // Identity still sanitizes but otherwise passes through.
    EXPECT_EQ(n.normalize("hello"), "hello");
    const std::string lig = "\xEF\xAC\x81";  // U+FB01
    EXPECT_EQ(n.normalize(lig), lig);
    std::string bad = "a";
    bad.push_back(static_cast<char>(0xFF));
    EXPECT_EQ(n.normalize(bad), std::string("a") + "\xEF\xBF\xBD");
}

TEST(PreTokenizerTest, NFKCNormalization) {
    UnicodeNormalizer n;
    n.setMode(NormalizationMode::NFKC);
    EXPECT_EQ(n.normalize("\xEF\xAC\x81"), "fi");  // U+FB01 -> fi
}

TEST(PreTokenizerTest, AllNormalizationModes) {
    UnicodeNormalizer n;
    // e + combining acute (U+0065 U+0301)
    const std::string decomposed = std::string("e") + "\xCC\x81";
    const std::string composed = "\xC3\xA9";  // U+00E9
    n.setMode(NormalizationMode::NFC);
    EXPECT_EQ(n.normalize(decomposed), composed);
    n.setMode(NormalizationMode::NFD);
    EXPECT_EQ(n.normalize(composed), decomposed);
    n.setMode(NormalizationMode::NFKC);
    EXPECT_EQ(n.normalize("\xEF\xAC\x81"), "fi");
    n.setMode(NormalizationMode::NFKD);
    EXPECT_EQ(n.normalize("\xEF\xAC\x81"), "fi");
}

TEST(PreTokenizerTest, PipelineComposition) {
    PreTokenizerPipeline pipe;
    pipe.loadPreTokenizerRuleset("cl100k_base");
    EXPECT_EQ(pipe.getNormalizationMode(), NormalizationMode::Identity);
    ExpectChunks(pipe.process("hello world"), {"hello", " world"});

    // Opt-in NFKC flows through the pipeline into storage.
    pipe.setNormalizationMode(NormalizationMode::NFKC);
    auto r = pipe.process("\xEF\xAC\x81");  // U+FB01
    EXPECT_EQ(r.storage, "fi");
    EXPECT_EQ(Concat(r), r.storage);
    ExpectAliasesStorage(r);
}

TEST(PreTokenizerTest, StorageCopyInvariant) {
    PreTokenizer tok;
    tok.loadRuleset("cl100k_base");
    // Even with no transformation, storage is an owned copy and chunks alias
    // storage (never caller memory).
    const std::string input = "hello";
    auto r = tok.split(std::string_view(input));
    EXPECT_EQ(r.storage, input);
    EXPECT_NE(r.storage.data(), input.data());
    ExpectAliasesStorage(r);

    PreTokenizerPipeline pipe;
    auto r2 = pipe.process("hello");
    EXPECT_NE(r2.storage.data(), input.data());
    ExpectAliasesStorage(r2);
    EXPECT_EQ(Concat(r2), r2.storage);
}

TEST(PreTokenizerTest, UnknownRulesetThrows) {
    PreTokenizer tok;
    EXPECT_THROW(tok.loadRuleset("not_a_ruleset"), std::invalid_argument);
}

TEST(PreTokenizerTest, O200kRuleset) {
    PreTokenizer tok;
    tok.loadRuleset("o200k_base");
    // o200k keeps English contractions attached via ('s|'t|...)? groups.
    ExpectChunks(tok.split("IT'S"), {"IT'S"});
    ExpectChunks(tok.split("hello world"), {"hello", " world"});
    ExpectChunks(tok.split("a\n\nb"), {"a", "\n\n", "b"});
    ExpectChunks(tok.split("2024/01/15"), {"202", "4", "/", "01", "/", "15"});
}

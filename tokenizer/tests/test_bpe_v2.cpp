// v2 engine tests: matrix ranks, DAT specials, byte-exactness vs v1,
// thread/file-path determinism, vocab.bin round-trip.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "BpeEngineV2.hpp"
#include "tokenizer/bpe/BpeTrainer.hpp"
#include "tokenizer/bpe/Serialization.hpp"
#include "tokenizer/bpe/Tokenizer.hpp"
#include "tokenizer/bpe/Vocabulary.hpp"

using namespace domlm;
using namespace domlm::engine;

namespace {

// Mini vocab with fully known merges:
//   train({"aaa"}) -> (97,97)->256 ("aa"), (256,97)->257 ("aaa").
domlm::tokenizer::Vocabulary MiniVocab() {
    domlm::tokenizer::BpeTrainer trainer(300);
    trainer.train(std::vector<std::string>{"aaa"});
    return trainer.vocabulary();
}

std::vector<int> ToInts(const std::vector<uint16_t>& ids) {
    return std::vector<int>(ids.begin(), ids.end());
}

const std::vector<std::string>& TrickyCorpus() {
    static const std::vector<std::string> k = {
        "hello world",
        "they're 2024",
        "IT'S",
        "a\n\nb",
        "   hello  world",
        "2024/01/15",
        std::string("hello\xC2\xA0world"),
        "hello\xE4\xB8\x96\xE7\x95\x8C",
        "hi\xF0\x9F\x98\x80there",
        "   \n  ",
        "",
        "aaa",
        "aaaa",
        "abc",
        "a!b",
        std::string("x\xFFy"),  // invalid UTF-8: both engines sanitize
    };
    return k;
}

}  // namespace

TEST(BpeV2Test, MatrixRanks) {
    BpeEngineV2 e = BpeEngineV2::FromVocabulary(MiniVocab(), "cl100k_base");
    EXPECT_EQ(e.matrix().rank(97, 97), 256);
    EXPECT_EQ(e.matrix().rank(256, 97), 257);
    EXPECT_EQ(e.matrix().rank(97, 98), -1);
    EXPECT_EQ(e.matrix().rank(0, 1), -1);
    EXPECT_EQ(e.matrix().rank(-1, 97), -1);
    EXPECT_EQ(e.matrix().rank(9999, 0), -1);
    EXPECT_EQ(e.matrix().rank(0, 9999), -1);
    EXPECT_EQ(e.vocabSize(), 258);
    EXPECT_THROW((void)BpeEngineV2::FromVocabulary(MiniVocab(), "nope"),
                 std::invalid_argument);
}

TEST(BpeV2Test, SpecialTrie) {
    BpeEngineV2 e = BpeEngineV2::FromVocabulary(MiniVocab(), "cl100k_base");
    const SpecialTrie& t = e.specials();
    EXPECT_EQ(t.matchAt("<|im_start|>hello", 0), 12);
    EXPECT_EQ(t.idAt("<|im_start|>hello", 0), 0);
    EXPECT_EQ(t.matchAt("x<|endoftext|>", 1), 13);
    EXPECT_EQ(t.idAt("x<|endoftext|>", 1), 2);
    EXPECT_EQ(t.matchAt("<|pad|>", 0), 7);
    EXPECT_EQ(t.idAt("<|pad|>", 0), 3);
    // Partial prefix is not a hit; scanning never backtracks the cursor.
    EXPECT_EQ(t.matchAt("<|im_ x", 0), -1);
    EXPECT_EQ(t.matchAt("plain text", 0), -1);
    EXPECT_EQ(t.findFirst("ab<|im_end|>cd", 0), 2U);
    EXPECT_EQ(t.findFirst("nothing here", 0), std::string_view::npos);
    EXPECT_EQ(t.findFirst("<|pad|><|pad|>", 1), 7U);
}

TEST(BpeV2Test, ExactVsV1MiniVocabs) {
    const std::vector<std::vector<std::string>> corpora = {
        {"aaa"}, {"ab", "cd"}, {"hello world", "abc"}};
    for (const auto& corpus : corpora) {
        domlm::tokenizer::BpeTrainer trainer(300);
        trainer.train(corpus);
        const domlm::tokenizer::Tokenizer v1(trainer.vocabulary());
        BpeEngineV2 v2 = BpeEngineV2::FromVocabulary(trainer.vocabulary(), "cl100k_base");
        for (const auto& s : TrickyCorpus()) {
            EXPECT_EQ(ToInts(v2.encode(s)), v1.encode(s)) << "input: " << s;
            EXPECT_EQ(v2.decode(v2.encode(s)), v1.decode(v1.encode(s)));
        }
    }
}

TEST(BpeV2Test, ThreadsAndFilePathDeterminism) {
    BpeEngineV2 e = BpeEngineV2::FromVocabulary(MiniVocab(), "cl100k_base");
    const std::vector<std::string> lines(TrickyCorpus());
    const auto st = e.encodeLines(lines, 1);
    for (std::size_t t : {std::size_t{2}, std::size_t{6}}) {
        const auto mt = e.encodeLines(lines, t);
        ASSERT_EQ(mt.size(), st.size());
        for (std::size_t i = 0; i < st.size(); ++i) {
            EXPECT_EQ(mt[i], st[i]) << "threads=" << t << " line=" << i;
        }
    }
    // File path (mmap + newline-aligned blocks) must agree line for line.
    const auto tmp = std::filesystem::temp_directory_path() / "bpe_v2_lines.txt";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.is_open());
        for (const auto& l : lines) {
            out.write(l.data(), static_cast<std::streamsize>(l.size()));
            out.put('\n');
        }
    }
    // Note: file lines carry their '\n', but encodeFileLines strips a single
    // terminator (getline parity). Corpus entries with embedded newlines
    // ("a\n\nb", "   \n  ") fan out to multiple file lines, so compare the
    // file path against the encoding of the file's ACTUAL lines.
    const auto fileOut = e.encodeFileLines(tmp.string(), 4);
    std::vector<std::string> fileLines;
    {
        std::ifstream in(tmp, std::ios::binary);
        std::string ln;
        while (std::getline(in, ln)) {
            if (!ln.empty() && ln.back() == '\r') {
                ln.pop_back();
            }
            fileLines.push_back(ln);
        }
    }
    const auto memOut = e.encodeLines(fileLines, 1);
    ASSERT_EQ(fileOut.size(), memOut.size());
    ASSERT_EQ(fileOut.size(), fileLines.size());
    for (std::size_t i = 0; i < memOut.size(); ++i) {
        EXPECT_EQ(fileOut[i], memOut[i]) << "line=" << i;
    }
    std::filesystem::remove(tmp);
}

TEST(BpeV2Test, SpecialTokenHandling) {
    BpeEngineV2 e = BpeEngineV2::FromVocabulary(MiniVocab(), "cl100k_base");
    EXPECT_THROW((void)e.encode("<|im_start|>hello"), std::runtime_error);
    // Mini vocab has no word merges: "hello" stays 5 byte ids behind the
    // special id.
    const auto ids = e.encode("<|im_start|>hello", true);
    ASSERT_EQ(ids.size(), 6U);
    EXPECT_EQ(ids[0], 258);  // vocabSize() + special index 0
    EXPECT_EQ(e.decode(ids), "<|im_start|>hello");
    const auto pieces = e.splitSpecials("a<|pad|>b");
    ASSERT_EQ(pieces.size(), 3U);
    EXPECT_FALSE(pieces[0].isSpecial);
    EXPECT_TRUE(pieces[1].isSpecial);
    EXPECT_EQ(pieces[1].specialIdx, 3);
    EXPECT_EQ(e.decode(std::vector<uint16_t>{260}), "<|endoftext|>");  // 258 + 2
    EXPECT_THROW((void)e.decode(std::vector<uint16_t>{9999}), std::out_of_range);
}

TEST(BpeV2Test, VocabBinRoundTrip) {
    const auto tmp = std::filesystem::temp_directory_path() / "bpe_v2_test.bin";
    domlm::tokenizer::SaveVocabBin(MiniVocab(), tmp.string());
    BpeEngineV2 e = BpeEngineV2::FromBinFile(tmp.string(), "cl100k_base");
    EXPECT_EQ(e.vocabSize(), 258);
    EXPECT_TRUE(e.matrix().isMapped());  // zero-copy adopt path, not rebuild
    domlm::tokenizer::Tokenizer v1(MiniVocab());
    for (const auto& s : TrickyCorpus()) {
        EXPECT_EQ(ToInts(e.encode(s)), v1.encode(s)) << "input: " << s;
    }
    // Corrupt magic must fail loudly, never silently.
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write("NOTABIN!", 8);
    }
    EXPECT_THROW((void)BpeEngineV2::FromBinFile(tmp.string(), "cl100k_base"),
                 std::runtime_error);
    EXPECT_THROW((void)BpeEngineV2::FromBinFile("/nonexistent/x.bin", "cl100k_base"),
                 std::runtime_error);
    std::filesystem::remove(tmp);
}

TEST(BpeV2Test, LegacyV1BinStillLoads) {
    // Hand-written version-1 layout (no matrix blob): the reader must take
    // the rebuild path (!isMapped) and still agree with v1 exactly.
    const domlm::tokenizer::Vocabulary vocab = MiniVocab();
    const auto tmp = std::filesystem::temp_directory_path() / "bpe_v2_legacy.bin";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.is_open());
        auto putU32 = [&](uint32_t v) {
            const char b[4] = {static_cast<char>(v & 0xFF),
                               static_cast<char>((v >> 8) & 0xFF),
                               static_cast<char>((v >> 16) & 0xFF),
                               static_cast<char>((v >> 24) & 0xFF)};
            out.write(b, 4);
        };
        auto putU16 = [&](uint32_t v) {
            const char b[2] = {static_cast<char>(v & 0xFF),
                               static_cast<char>((v >> 8) & 0xFF)};
            out.write(b, 2);
        };
        out.write("DOMBPE02", 8);
        putU32(1);
        putU32(static_cast<uint32_t>(vocab.size()));
        putU32(static_cast<uint32_t>(vocab.merges().size()));
        uint32_t off = 0;
        for (std::size_t i = 0; i < vocab.size(); ++i) {
            putU32(off);
            off += static_cast<uint32_t>(vocab.bytesOf(static_cast<int>(i)).size());
        }
        putU32(off);
        for (std::size_t i = 0; i < vocab.size(); ++i) {
            const std::string& b = vocab.bytesOf(static_cast<int>(i));
            out.write(b.data(), static_cast<std::streamsize>(b.size()));
        }
        for (const auto& m : vocab.merges()) {
            putU16(static_cast<uint32_t>(m.id_a));
            putU16(static_cast<uint32_t>(m.id_b));
            putU16(static_cast<uint32_t>(m.new_id));
            putU16(0);
        }
        out.flush();
    }
    BpeEngineV2 e = BpeEngineV2::FromBinFile(tmp.string(), "cl100k_base");
    EXPECT_FALSE(e.matrix().isMapped());
    domlm::tokenizer::Tokenizer v1(MiniVocab());
    for (const auto& s : TrickyCorpus()) {
        EXPECT_EQ(ToInts(e.encode(s)), v1.encode(s)) << "input: " << s;
    }
    std::filesystem::remove(tmp);
}

TEST(BpeV2Test, RealVocabAgreement) {
    // Gated on repo artifacts; skipped (not failed) when absent so the
    // hermetic unit tests above stay dependency-free.
    // Search same-directory pairs so vocab.json and merges.txt always come
    // from one training run (root and data/ artifacts differ).
    const std::vector<std::pair<std::string, std::string>> pairs = {
        {"data/vocab.json", "data/merges.txt"},
        {"../../data/vocab.json", "../../data/merges.txt"},
        {"/home/ardis/Desktop/Dom/tokenizer/data/vocab.json",
         "/home/ardis/Desktop/Dom/tokenizer/data/merges.txt"},
        {"/home/ardis/Desktop/Dom/tokenizer/vocab.json",
         "/home/ardis/Desktop/Dom/tokenizer/merges.txt"},
    };
    std::string vpath;
    std::string mpath;
    for (const auto& [v, m] : pairs) {
        if (std::filesystem::exists(v) && std::filesystem::exists(m)) {
            vpath = v;
            mpath = m;
            break;
        }
    }
    if (vpath.empty()) {
        GTEST_SKIP() << "repo vocab artifacts absent";
    }
    const domlm::tokenizer::Vocabulary vocab = domlm::tokenizer::LoadVocabulary(vpath, mpath);
    const domlm::tokenizer::Tokenizer v1(vocab);
    BpeEngineV2 v2 = BpeEngineV2::FromVocabulary(vocab, "cl100k_base");
    for (const auto& s : TrickyCorpus()) {
        EXPECT_EQ(ToInts(v2.encode(s)), v1.encode(s)) << "input: " << s;
    }
    // First 300 valid.txt lines when available: ST vs MT(6) vs mmap path.
    const std::vector<std::string> lineCandidates = {
        "../../PreTrain/valid.txt",
        "/home/ardis/Desktop/Dom/PreTrain/valid.txt",
    };
    std::string lpath;
    for (const auto& c : lineCandidates) {
        if (std::filesystem::exists(c)) {
            lpath = c;
            break;
        }
    }
    if (lpath.empty()) {
        GTEST_SKIP() << "repo corpus absent";
    }
    std::vector<std::string> lines;
    {
        std::ifstream in(lpath, std::ios::binary);
        std::string ln;
        while (lines.size() < 300 && std::getline(in, ln)) {
            if (!ln.empty() && ln.back() == '\r') {
                ln.pop_back();
            }
            lines.push_back(ln);
        }
    }
    const auto st = v2.encodeLines(lines, 1);
    const auto mt = v2.encodeLines(lines, 6);
    ASSERT_EQ(mt.size(), st.size());
    std::vector<std::vector<int>> ref;
    for (const auto& s : lines) {
        ref.push_back(v1.encode(s));
    }
    for (std::size_t i = 0; i < st.size(); ++i) {
        EXPECT_EQ(ToInts(st[i]), ref[i]) << "line=" << i;
        EXPECT_EQ(mt[i], st[i]) << "line=" << i;
    }
}

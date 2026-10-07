#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "tokenizer/bpe/BpeTrainer.hpp"
#include "tokenizer/bpe/Serialization.hpp"
#include "tokenizer/bpe/Tokenizer.hpp"
#include "tokenizer/bpe/Vocabulary.hpp"

using namespace domlm::tokenizer;

namespace {

// Mini vocab whose merges are fully known:
//   train({"aaa"}) -> (97,97)->256 ("aa"), (256,97)->257 ("aaa").
Tokenizer MiniTokenizer() {
    BpeTrainer trainer(300);
    trainer.train(std::vector<std::string>{"aaa"});
    return Tokenizer(trainer.vocabulary());
}

}  // namespace

TEST(BpeTokenizerTest, KnownEncodingsMatchTraining) {
    Tokenizer tok = MiniTokenizer();
    EXPECT_EQ(tok.vocabSize(), 258U);
    EXPECT_EQ(tok.encode("aaa"), std::vector<int>({257}));
    EXPECT_EQ(tok.encode("aa"), std::vector<int>({256}));
    EXPECT_EQ(tok.encode("a"), std::vector<int>({97}));
    // [97,97,97,97] -> merge all non-overlapping (97,97) -> [256,256];
    // (256,256) was never learned, so encoding stops there.
    EXPECT_EQ(tok.encode("aaaa"), std::vector<int>({256, 256}));
    EXPECT_EQ(tok.decode(std::vector<int>({257})), "aaa");
    EXPECT_EQ(tok.decode(std::vector<int>({256, 256})), "aaaa");
}

TEST(BpeTokenizerTest, RankOrderNotGreedySubstring) {
    // train({"abc"}): tie (97,98)|(98,99) -> (97,98)->256, then (256,99)->257.
    BpeTrainer trainer(300);
    trainer.train(std::vector<std::string>{"abc"});
    Tokenizer tok(trainer.vocabulary());
    EXPECT_EQ(tok.encode("abc"), std::vector<int>({257}));
    EXPECT_EQ(tok.encode("ab"), std::vector<int>({256}));
    // (98,99) was never a learned rule: no merge happens here.
    EXPECT_EQ(tok.encode("bc"), std::vector<int>({98, 99}));
}

TEST(BpeTokenizerTest, MergesNeverCrossChunkBoundaries) {
    // train({"ab"}) learns (97,98)->256, but "a\nb" pre-splits into three
    // single-char chunks, so the encoder must emit bytes, never the merge.
    BpeTrainer trainer(300);
    trainer.train(std::vector<std::string>{"ab"});
    Tokenizer tok(trainer.vocabulary());
    EXPECT_EQ(tok.encode("ab"), std::vector<int>({256}));
    EXPECT_EQ(tok.encode("a\nb"), std::vector<int>({97, 10, 98}));
}

TEST(BpeTokenizerTest, RoundTripValidText) {
    BpeTrainer trainer(300);
    trainer.train(std::vector<std::string>{"hello world", "aaa", "abc"});
    Tokenizer tok(trainer.vocabulary());
    const std::vector<std::string> cases = {
        "hello world",
        "aaa",
        " a\n\nb IT'S 2024/01/15",
        "hello\xE4\xB8\x96\xE7\x95\x8C",
        "hi\xF0\x9F\x98\x80there",
        "",
    };
    for (const auto& s : cases) {
        EXPECT_EQ(tok.decode(tok.encode(s)), s) << "input: " << s;
    }
    EXPECT_TRUE(tok.encode("").empty());
    EXPECT_EQ(tok.decode(std::vector<int>{}), "");
}

TEST(BpeTokenizerTest, EncodeIntoReusesCallerBuffer) {
    Tokenizer tok = MiniTokenizer();
    std::vector<int> buf;
    tok.encodeInto("aaa", buf);
    EXPECT_EQ(buf, std::vector<int>({257}));
    buf.clear();
    tok.encodeInto("aa", buf);
    EXPECT_EQ(buf, std::vector<int>({256}));
}

TEST(BpeTokenizerTest, DecodeRejectsUnknownIds) {
    Tokenizer tok = MiniTokenizer();
    EXPECT_THROW((void)tok.decode(std::vector<int>({99999})), std::out_of_range);
}

TEST(BpeTokenizerTest, FileRoundTripPreservesVocabulary) {
    BpeTrainer trainer(300);
    trainer.train(std::vector<std::string>{"hello world", "aaa"});
    const Vocabulary& src = trainer.vocabulary();

    const auto dir = std::filesystem::temp_directory_path();
    const auto vpath = (dir / "bpe_tok_vocab.json").string();
    const auto mpath = (dir / "bpe_tok_merges.txt").string();
    SaveVocabJson(src, vpath);
    SaveMerges(src, mpath);

    const Vocabulary loaded = LoadVocabulary(vpath, mpath);
    EXPECT_EQ(loaded.size(), src.size());
    EXPECT_EQ(loaded.merges().size(), src.merges().size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        EXPECT_EQ(loaded.bytesOf(static_cast<int>(i)), src.bytesOf(i));
    }

    const Tokenizer fromFiles = Tokenizer::load(vpath, mpath);
    const Tokenizer fromMemory(src);
    for (const std::string s : {"hello world", "aaa", "a\nb"}) {
        EXPECT_EQ(fromFiles.encode(s), fromMemory.encode(s));
    }

    EXPECT_THROW((void)LoadVocabulary("/nonexistent/v.json", mpath), std::runtime_error);
    EXPECT_THROW((void)Tokenizer::load(vpath, mpath, "nope"), std::invalid_argument);

    std::filesystem::remove(vpath);
    std::filesystem::remove(mpath);
}

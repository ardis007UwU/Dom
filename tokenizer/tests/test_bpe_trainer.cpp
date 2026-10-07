#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "tokenizer/bpe/BpeTrainer.hpp"
#include "tokenizer/bpe/CorpusReader.hpp"
#include "tokenizer/bpe/Vocabulary.hpp"
#include "tokenizer/pre/PreTokenizerPipeline.hpp"

using namespace domlm::tokenizer;
using domlm::tokenizer::PreTokenizerPipeline;

// Test 1: base 256 byte vocabulary setup.
TEST(BpeTrainerTest, BaseByteVocabulary) {
    Vocabulary vocab;
    EXPECT_EQ(vocab.size(), 256U);
    EXPECT_TRUE(vocab.merges().empty());
    for (int b = 0; b < 256; ++b) {
        const std::string bytes(1, static_cast<char>(static_cast<unsigned char>(b)));
        EXPECT_TRUE(vocab.contains(bytes));
        EXPECT_EQ(vocab.idOf(bytes), b);
        EXPECT_EQ(vocab.bytesOf(b), bytes);
    }
    // Display names live apart from byte payloads.
    EXPECT_EQ(vocab.displayOf(0), "<0x00>");
    EXPECT_EQ(vocab.displayOf(0x41), "<0x41>");
    EXPECT_EQ(vocab.displayOf(0xFF), "<0xFF>");
    EXPECT_EQ(vocab.idOfPair(97, 98), -1);
    EXPECT_THROW((void)vocab.bytesOf(256), std::out_of_range);
    EXPECT_THROW((void)vocab.idOf("ab"), std::out_of_range);
}

// Test 2: pairs are never counted across pre-token chunk boundaries.
TEST(BpeTrainerTest, PreTokenBoundaryRestriction) {
    // Self-validating premise: cl100k splits "a\nb" into three single-char
    // chunks, so no within-chunk pair exists at all.
    PreTokenizerPipeline pipe;
    const auto split = pipe.split("a\nb");
    ASSERT_EQ(split.chunks.size(), 3U);
    for (const auto& c : split.chunks) {
        EXPECT_EQ(c.size(), 1U);
    }

    // Raw-adjacent pairs (a,\n) and (\n,b) must never be counted or merged.
    BpeTrainer trainer(300);
    trainer.train(std::vector<std::string>{"a\nb"});
    EXPECT_EQ(trainer.vocabulary().size(), 256U);
    EXPECT_TRUE(trainer.vocabulary().merges().empty());
}

// Test 3: deterministic tie-breaking on equal-frequency pairs.
TEST(BpeTrainerTest, DeterministicTieBreaking) {
    // "ab" -> {(97,98):1}, "cd" -> {(99,100):1}: tie at 1, smallest tuple
    // (97,98) must win.
    BpeTrainer first(257);
    first.train(std::vector<std::string>{"ab", "cd"});
    const auto& merges = first.vocabulary().merges();
    ASSERT_EQ(merges.size(), 1U);
    EXPECT_EQ(merges[0].id_a, 97);
    EXPECT_EQ(merges[0].id_b, 98);
    EXPECT_EQ(merges[0].new_id, 256);

    // Re-run with room for two merges: the runner-up follows.
    BpeTrainer second(258);
    second.train(std::vector<std::string>{"ab", "cd"});
    const auto& merges2 = second.vocabulary().merges();
    ASSERT_EQ(merges2.size(), 2U);
    EXPECT_EQ(merges2[1].id_a, 99);
    EXPECT_EQ(merges2[1].id_b, 100);
    EXPECT_EQ(merges2[1].new_id, 257);
}

// Test 4: full mini-corpus run with exact vocabulary and merge order.
TEST(BpeTrainerTest, MiniCorpusExactMerges) {
    // "aaa": pairs {(97,97):2} -> merge (97,97)->256 ("aa");
    // re-encoded [256,97]: pairs {(256,97):1} -> merge (256,97)->257 ("aaa").
    BpeTrainer trainer(300);
    trainer.train(std::vector<std::string>{"aaa"});
    const Vocabulary& vocab = trainer.vocabulary();
    EXPECT_EQ(vocab.size(), 258U);
    const auto& merges = vocab.merges();
    ASSERT_EQ(merges.size(), 2U);
    EXPECT_EQ(merges[0].id_a, 97);
    EXPECT_EQ(merges[0].id_b, 97);
    EXPECT_EQ(merges[0].new_id, 256);
    EXPECT_EQ(merges[1].id_a, 256);
    EXPECT_EQ(merges[1].id_b, 97);
    EXPECT_EQ(merges[1].new_id, 257);
    EXPECT_EQ(vocab.bytesOf(256), "aa");
    EXPECT_EQ(vocab.bytesOf(257), "aaa");
    EXPECT_EQ(vocab.idOf("aa"), 256);
    EXPECT_EQ(vocab.idOf("aaa"), 257);
}

// Test 5: streaming reader (train/valid/test separation) + equivalence.
TEST(BpeTrainerTest, CorpusReaderStreaming) {
    const auto dir = std::filesystem::temp_directory_path();
    const auto trainPath = (dir / "bpe_test_train.txt").string();
    const auto validPath = (dir / "bpe_test_valid.txt").string();
    {
        std::ofstream out(trainPath, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.is_open());
        out << "aaa\nb\n";
    }
    {
        std::ofstream out(validPath, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.is_open());
        out << "held-out\n";
    }

    CorpusReader trainReader(trainPath, CorpusSplit::Train);
    EXPECT_EQ(trainReader.split(), CorpusSplit::Train);
    std::vector<std::string> lines;
    std::string line;
    while (trainReader.nextLine(line)) {
        lines.push_back(line);
    }
    ASSERT_EQ(lines.size(), 2U);
    EXPECT_EQ(lines[0], "aaa");
    EXPECT_EQ(lines[1], "b");
    EXPECT_EQ(trainReader.linesRead(), 2U);

    // Evaluation corpora use separate reader instances over separate paths.
    CorpusReader validReader(validPath, CorpusSplit::Valid);
    EXPECT_EQ(validReader.split(), CorpusSplit::Valid);
    ASSERT_TRUE(validReader.nextLine(line));
    EXPECT_EQ(line, "held-out");

    EXPECT_THROW(CorpusReader("/nonexistent/path/train.txt"), std::runtime_error);

    // Reader-fed training matches in-memory training exactly.
    CorpusReader rerun(trainPath, CorpusSplit::Train);
    BpeTrainer viaReader(300);
    viaReader.train(rerun);
    BpeTrainer viaMemory(300);
    viaMemory.train(std::vector<std::string>{"aaa", "b"});
    EXPECT_EQ(viaReader.vocabulary().size(), viaMemory.vocabulary().size());
    EXPECT_EQ(viaReader.vocabulary().merges().size(), viaMemory.vocabulary().merges().size());

    std::filesystem::remove(trainPath);
    std::filesystem::remove(validPath);
}

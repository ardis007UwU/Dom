// Phase 2 CLI: train a byte-level BPE vocabulary from a text corpus.
//
//   ./train_bpe --input Train.txt --vocab-size 32000 --output-vocab vocab.json
//       --output-merges merges.txt
//
// vocab.json layout: {"version":1,"vocab_size":N,
//   "tokens":[{"id":0,"text":"<0x00>"}, ...]} where "text" is JSON-escaped:
//   printable ASCII (0x20..0x7E except '"' and '\\') verbatim, '"'/\\ and
//   C0 controls via short escapes or \u00XX, all other bytes (0x7F, >=0x80)
//   as \u00XX per byte, keeping the file pure ASCII and safely escaping
//   non-printable sequences.
// merges.txt layout: one "<id_a> <id_b> -> <new_id>" per line, in training
//   order (line i <=> new id 256+i).

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "tokenizer/bpe/BpeTrainer.hpp"
#include "tokenizer/bpe/CorpusReader.hpp"
#include "tokenizer/bpe/Vocabulary.hpp"

namespace {

using tokenizer::bpe::BpeTrainer;
using tokenizer::bpe::CorpusReader;
using tokenizer::bpe::CorpusSplit;
using tokenizer::bpe::Vocabulary;

void PrintUsage(std::string_view prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " --input Train.txt --vocab-size 32000 "
              << "--output-vocab vocab.json --output-merges merges.txt\n"
              << "  " << prog << " --input train.txt [--vocab-size N] "
              << "[--output-vocab vocab.json] [--output-merges merges.txt] "
              << "[--ruleset cl100k_base|o200k_base]\n";
}

// JSON-escape one token's raw bytes to pure-ASCII text.
std::string EscapeJsonBytes(std::string_view bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() + 2);
    for (unsigned char c : bytes) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c >= 0x20 && c <= 0x7E) {
                    out.push_back(static_cast<char>(c));
                } else {
                    out += "\\u00";
                    out.push_back(kHex[(c >> 4) & 0xF]);
                    out.push_back(kHex[c & 0xF]);
                }
                break;
        }
    }
    return out;
}

bool WriteVocabJson(const std::string& path, const Vocabulary& vocab) {
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out << "{\"version\":1,\"vocab_size\":" << vocab.size() << ",\"tokens\":[";
    for (std::size_t id = 0; id < vocab.size(); ++id) {
        if (id != 0) {
            out << ',';
        }
        out << "{\"id\":" << id << ",\"text\":\""
            << EscapeJsonBytes(vocab.bytesOf(static_cast<int>(id))) << "\"}";
    }
    out << "]}";
    out.flush();
    return static_cast<bool>(out);
}

bool WriteMerges(const std::string& path, const Vocabulary& vocab) {
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    for (const auto& m : vocab.merges()) {
        out << m.id_a << ' ' << m.id_b << " -> " << m.new_id << '\n';
    }
    out.flush();
    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string input;
    std::size_t vocabSize = BpeTrainer::kDefaultVocabSize;
    std::string outputVocab = "vocab.json";
    std::string outputMerges = "merges.txt";
    std::string ruleset = "cl100k_base";

    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        auto needValue = [&](std::string_view flag) -> std::string_view {
            if (i + 1 >= argc) {
                std::cerr << "train_bpe: flag " << flag << " needs a value\n";
                PrintUsage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            PrintUsage(argv[0]);
            return 0;
        }
        if (a == "--input") {
            input.assign(needValue(a));
        } else if (a == "--vocab-size") {
            const std::string v(needValue(a));
            try {
                vocabSize = static_cast<std::size_t>(std::stoul(v));
            } catch (const std::exception&) {
                std::cerr << "train_bpe: invalid --vocab-size '" << v << "'\n";
                return 2;
            }
        } else if (a == "--output-vocab") {
            outputVocab.assign(needValue(a));
        } else if (a == "--output-merges") {
            outputMerges.assign(needValue(a));
        } else if (a == "--ruleset") {
            ruleset.assign(needValue(a));
        } else {
            std::cerr << "train_bpe: unknown flag '" << a << "'\n";
            PrintUsage(argv[0]);
            return 2;
        }
    }

    if (input.empty()) {
        std::cerr << "train_bpe: --input is required\n";
        PrintUsage(argv[0]);
        return 2;
    }
    if (vocabSize < Vocabulary::kBaseSize) {
        std::cerr << "train_bpe: --vocab-size must be >= 256\n";
        return 2;
    }
    if (ruleset != "cl100k_base" && ruleset != "o200k_base") {
        std::cerr << "train_bpe: --ruleset must be cl100k_base or o200k_base\n";
        return 2;
    }

    try {
        // Training consumes the training corpus only; validation / test
        // corpora stay in separate reader instances for evaluation.
        CorpusReader reader(input, CorpusSplit::Train);
        BpeTrainer trainer(vocabSize);
        trainer.pipeline().loadPreTokenizerRuleset(ruleset);
        trainer.train(reader);

        const Vocabulary& vocab = trainer.vocabulary();
        if (!WriteVocabJson(outputVocab, vocab)) {
            std::cerr << "train_bpe: cannot write '" << outputVocab << "'\n";
            return 1;
        }
        if (!WriteMerges(outputMerges, vocab)) {
            std::cerr << "train_bpe: cannot write '" << outputMerges << "'\n";
            return 1;
        }
        std::cout << "trained vocab_size=" << vocab.size() << " merges=" << vocab.merges().size()
                  << " ruleset=" << ruleset << "\n";
    } catch (const std::exception& e) {
        std::cerr << "train_bpe: error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

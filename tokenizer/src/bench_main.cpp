// Throughput benchmark for the Phase 3 BPE inference engine.
//
//   ./bench_tokenize --vocab vocab.json --merges merges.txt --input valid.txt
//   ./bench_tokenize --vocab vocab.json --merges merges.txt --input valid.txt --repeat 5 --decode
//
// Method: files/tokenizer load once (reported separately), one warmup pass,
// then N timed encode passes over all lines with a caller-reused id buffer
// (zero steady-state allocation). Reports tok/s, MiB/s and us/line.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "domlm/Version.hpp"
#include "tokenizer/bpe/Tokenizer.hpp"

namespace {

using domlm::tokenizer::Tokenizer;

void PrintUsage(std::string_view prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " --vocab vocab.json --merges merges.txt --input valid.txt\n"
              << "  " << prog << " --vocab vocab.json --merges merges.txt --input valid.txt "
              << "[--repeat N] [--decode] [--ruleset cl100k_base|o200k_base]\n"
              << "  " << prog << " --version\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string vocabPath;
    std::string mergesPath;
    std::string inputPath;
    long repeat = 3;
    bool doDecode = false;
    std::string ruleset = "cl100k_base";

    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        auto needValue = [&](std::string_view flag) -> std::string_view {
            if (i + 1 >= argc) {
                std::cerr << "bench_tokenize: flag " << flag << " needs a value\n";
                PrintUsage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            PrintUsage(argv[0]);
            return 0;
        }
        if (a == "-v" || a == "--version") {
            std::cout << "DomLM v" << DOMLM_VERSION_STRING << "\n";
            return 0;
        }
        if (a == "--vocab") {
            vocabPath.assign(needValue(a));
        } else if (a == "--merges") {
            mergesPath.assign(needValue(a));
        } else if (a == "--input") {
            inputPath.assign(needValue(a));
        } else if (a == "--repeat") {
            try {
                repeat = std::stol(std::string(needValue(a)));
            } catch (const std::exception&) {
                std::cerr << "bench_tokenize: invalid --repeat\n";
                return 2;
            }
            if (repeat < 1) {
                std::cerr << "bench_tokenize: --repeat must be >= 1\n";
                return 2;
            }
        } else if (a == "--decode") {
            doDecode = true;
        } else if (a == "--ruleset") {
            ruleset.assign(needValue(a));
        } else {
            std::cerr << "bench_tokenize: unknown flag '" << a << "'\n";
            PrintUsage(argv[0]);
            return 2;
        }
    }
    if (vocabPath.empty() || mergesPath.empty() || inputPath.empty()) {
        std::cerr << "bench_tokenize: --vocab, --merges and --input are required\n";
        PrintUsage(argv[0]);
        return 2;
    }

    try {
        const auto t0 = std::chrono::steady_clock::now();
        Tokenizer tok = Tokenizer::load(vocabPath, mergesPath, ruleset);
        std::vector<std::string> lines;
        {
            std::ifstream in(inputPath, std::ios::in | std::ios::binary);
            if (!in.is_open()) {
                std::cerr << "bench_tokenize: cannot open '" << inputPath << "'\n";
                return 1;
            }
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                lines.push_back(line);
            }
        }
        const auto t1 = std::chrono::steady_clock::now();

        std::vector<int> ids;
        ids.reserve(4096);
        std::uint64_t totalTokens = 0;
        std::uint64_t totalBytes = 0;
        for (const auto& l : lines) {
            totalBytes += l.size();
        }
        // Warmup (caches hot paths, grows the reusable buffer once).
        for (const auto& l : lines) {
            ids.clear();
            tok.encodeInto(l, ids);
        }
        totalTokens = 0;

        const auto t2 = std::chrono::steady_clock::now();
        for (long r = 0; r < repeat; ++r) {
            for (const auto& l : lines) {
                ids.clear();
                tok.encodeInto(l, ids);
                totalTokens += ids.size();
            }
        }
        const auto t3 = std::chrono::steady_clock::now();

        const double loadMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double encMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
        const double mib = static_cast<double>(totalBytes) * repeat / (1024.0 * 1024.0);
        const double linesDone = static_cast<double>(lines.size()) * repeat;
        std::cout << "vocab=" << tok.vocabSize() << " lines=" << lines.size()
                  << " repeat=" << repeat << " load_ms=" << loadMs << "\n";
        std::cout << "encode: tokens=" << totalTokens << " ms=" << encMs
                  << " tok_per_s=" << (totalTokens * 1000.0 / encMs)
                  << " mib_per_s=" << (mib * 1000.0 / encMs)
                  << " us_per_line=" << (encMs * 1000.0 / linesDone) << "\n";

        if (doDecode) {
            // Decode pass over one repetition's worth of ids.
            std::vector<std::vector<int>> snapshot;
            snapshot.reserve(lines.size());
            for (const auto& l : lines) {
                ids.clear();
                tok.encodeInto(l, ids);
                snapshot.push_back(ids);
            }
            std::string text;
            const auto t4 = std::chrono::steady_clock::now();
            std::uint64_t decBytes = 0;
            for (long r = 0; r < repeat; ++r) {
                for (const auto& s : snapshot) {
                    text = tok.decode(s);
                    decBytes += text.size();
                }
            }
            const auto t5 = std::chrono::steady_clock::now();
            const double decMs = std::chrono::duration<double, std::milli>(t5 - t4).count();
            std::cout << "decode: bytes=" << decBytes << " ms=" << decMs
                      << " mib_per_s=" << (decBytes / (1024.0 * 1024.0) * 1000.0 / decMs)
                      << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "bench_tokenize: error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

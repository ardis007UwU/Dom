// Lightweight manual smoke-test CLI for the pre-tokenizer pipeline.
//
// Usage:
//   ./tokenize "hello world"
//   ./tokenize --o200k "hello world"
//   ./tokenize --cl100k "hello world"
//   ./tokenize --o200k --nfkc "text"
//   echo "hello" | ./tokenize        (reads stdin when no text arg given)
//
// Passes the string through PreTokenizerPipeline and prints each split chunk
// with index, character length (Unicode code points), byte length, escaped
// text, and byte hex. No raw new/delete (RAII only).

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <string_view>

#include "domlm/Version.hpp"
#include "tokenizer/pre/PreTokenizerPipeline.hpp"

namespace {

using domlm::tokenizer::NormalizationMode;
using domlm::tokenizer::PreTokenizerPipeline;

// Count Unicode code points in valid UTF-8 (storage is always sanitized).
std::size_t CountCodePoints(std::string_view s) {
    std::size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) {
            ++n;
        }
    }
    return n;
}

std::string Escape(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            default:
                if (c >= 0x20 && c < 0x7F) {
                    out.push_back(static_cast<char>(c));
                } else {
                    char buf[5];
                    std::snprintf(buf, sizeof(buf), "\\x%02X", c);
                    out += buf;
                }
                break;
        }
    }
    return out;
}

std::string HexBytes(std::string_view s) {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        char buf[4];
        std::snprintf(buf, sizeof(buf), "%02X", static_cast<unsigned char>(s[i]));
        out += buf;
    }
    return out.empty() ? "(empty)" : out;
}

void PrintUsage(std::string_view prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " \"hello world\"\n"
              << "  " << prog << " --o200k \"hello world\"\n"
              << "  " << prog << " --cl100k \"hello world\"\n"
              << "  " << prog << " [--o200k|--cl100k] [--nfc|--nfd|--nfkc|--nfkd] \"text\"\n"
              << "  echo \"hello\" | " << prog << "\n"
              << "  " << prog << " --version\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string ruleset = "cl100k_base";
    NormalizationMode mode = NormalizationMode::Identity;
    std::string text;
    bool haveText = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        if (a == "-h" || a == "--help") {
            PrintUsage(argv[0]);
            return 0;
        }
        if (a == "-v" || a == "--version") {
            std::cout << "DomLM v" << DOMLM_VERSION_STRING << "\n";
            return 0;
        }
        if (a == "--o200k") {
            ruleset = "o200k_base";
        } else if (a == "--cl100k") {
            ruleset = "cl100k_base";
        } else if (a == "--nfc") {
            mode = NormalizationMode::NFC;
        } else if (a == "--nfd") {
            mode = NormalizationMode::NFD;
        } else if (a == "--nfkc") {
            mode = NormalizationMode::NFKC;
        } else if (a == "--nfkd") {
            mode = NormalizationMode::NFKD;
        } else if (!haveText) {
            text.assign(a.data(), a.size());
            haveText = true;
        } else {
            std::cerr << "Unexpected extra argument: " << a << "\n";
            PrintUsage(argv[0]);
            return 2;
        }
    }

    if (!haveText) {
        // No text arg: read all of stdin (allows piping).
        std::string piped;
        char buf[4096];
        std::size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0) {
            piped.append(buf, n);
        }
        if (piped.empty()) {
            PrintUsage(argv[0]);
            return 2;
        }
        text = std::move(piped);
    }

    try {
        PreTokenizerPipeline pipe;
        pipe.loadPreTokenizerRuleset(ruleset);
        pipe.setNormalizationMode(mode);
        auto result = pipe.process(std::string_view(text));

        std::cout << "ruleset=" << ruleset << " storage_bytes=" << result.storage.size()
                  << " chunks=" << result.chunks.size() << "\n";
        for (std::size_t i = 0; i < result.chunks.size(); ++i) {
            const std::string_view c = result.chunks[i];
            std::cout << "[" << i << "] chars=" << CountCodePoints(c) << " bytes=" << c.size()
                      << " text=\"" << Escape(c) << "\" hex=[" << HexBytes(c) << "]\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "tokenize: error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

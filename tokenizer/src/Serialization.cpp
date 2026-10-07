#include "tokenizer/bpe/Serialization.hpp"

#include <bit>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace domlm::tokenizer {

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

void SaveVocabJson(const Vocabulary& vocab, const std::string& path) {
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("SaveVocabJson: cannot open '" + path + "'");
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
    if (!out) {
        throw std::runtime_error("SaveVocabJson: write failed for '" + path + "'");
    }
}

void SaveMerges(const Vocabulary& vocab, const std::string& path) {
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("SaveMerges: cannot open '" + path + "'");
    }
    for (const auto& m : vocab.merges()) {
        out << m.id_a << ' ' << m.id_b << " -> " << m.new_id << '\n';
    }
    out.flush();
    if (!out) {
        throw std::runtime_error("SaveMerges: write failed for '" + path + "'");
    }
}

namespace {

// Minimal JSON string unescaper for our own vocab.json "text" fields.
// Handles \" \\ \b \f \n \r \t and \uXXXX (only \u00XX values occur;
// values > 0xFF are rejected). Returns raw bytes.
std::string UnescapeJsonString(std::string_view esc, const std::string& ctx) {
    auto hexVal = [&](char h) -> int {
        if (h >= '0' && h <= '9') {
            return h - '0';
        }
        if (h >= 'a' && h <= 'f') {
            return h - 'a' + 10;
        }
        if (h >= 'A' && h <= 'F') {
            return h - 'A' + 10;
        }
        return -1;
    };
    std::string out;
    out.reserve(esc.size());
    for (std::size_t i = 0; i < esc.size(); ++i) {
        const char c = esc[i];
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (++i >= esc.size()) {
            throw std::runtime_error("LoadVocabulary: dangling escape (" + ctx + ")");
        }
        switch (esc[i]) {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                if (i + 4 >= esc.size()) {
                    throw std::runtime_error("LoadVocabulary: truncated \\u (" + ctx + ")");
                }
                int v = 0;
                for (int k = 1; k <= 4; ++k) {
                    const int d = hexVal(esc[i + k]);
                    if (d < 0) {
                        throw std::runtime_error("LoadVocabulary: bad \\u digits (" + ctx + ")");
                    }
                    v = v * 16 + d;
                }
                if (v > 0xFF) {
                    throw std::runtime_error("LoadVocabulary: \\u out of byte range (" + ctx + ")");
                }
                out.push_back(static_cast<char>(static_cast<unsigned char>(v)));
                i += 4;
                break;
            }
            default:
                throw std::runtime_error("LoadVocabulary: bad escape (" + ctx + ")");
        }
    }
    return out;
}

// Extracts (id, raw-bytes) entries from our vocab.json layout by scanning
// for {"id":<int>,"text":"<esc>"} records. Order-independent.
std::vector<std::pair<int, std::string>> ParseVocabJson(const std::string& doc) {
    std::vector<std::pair<int, std::string>> entries;
    std::size_t pos = 0;
    while (true) {
        pos = doc.find("\"id\"", pos);
        if (pos == std::string::npos) {
            break;
        }
        pos += 4;
        while (pos < doc.size() &&
               (doc[pos] == ' ' || doc[pos] == '\t' || doc[pos] == ':')) {
            ++pos;
        }
        if (pos >= doc.size() || !std::isdigit(static_cast<unsigned char>(doc[pos]))) {
            throw std::runtime_error("LoadVocabulary: malformed \"id\" value");
        }
        int id = 0;
        while (pos < doc.size() && std::isdigit(static_cast<unsigned char>(doc[pos]))) {
            id = id * 10 + (doc[pos] - '0');
            ++pos;
        }
        pos = doc.find("\"text\"", pos);
        if (pos == std::string::npos) {
            throw std::runtime_error("LoadVocabulary: missing \"text\" field");
        }
        pos = doc.find('"', pos + 6);
        if (pos == std::string::npos) {
            throw std::runtime_error("LoadVocabulary: unterminated \"text\" field");
        }
        ++pos;  // first content char
        std::string esc;
        bool closed = false;
        while (pos < doc.size()) {
            const char c = doc[pos];
            if (c == '\\') {
                if (pos + 1 >= doc.size()) {
                    break;
                }
                esc.push_back(c);
                esc.push_back(doc[pos + 1]);
                pos += 2;
            } else if (c == '"') {
                closed = true;
                ++pos;
                break;
            } else {
                esc.push_back(c);
                ++pos;
            }
        }
        if (!closed) {
            throw std::runtime_error("LoadVocabulary: unterminated \"text\" string");
        }
        entries.emplace_back(id, UnescapeJsonString(esc, "id " + std::to_string(id)));
    }
    if (entries.empty()) {
        throw std::runtime_error("LoadVocabulary: no token entries found");
    }
    return entries;
}

}  // namespace

Vocabulary LoadVocabulary(const std::string& vocab_path, const std::string& merges_path) {
    std::ifstream vin(vocab_path, std::ios::in | std::ios::binary);
    if (!vin.is_open()) {
        throw std::runtime_error("LoadVocabulary: cannot open '" + vocab_path + "'");
    }
    std::ostringstream vdoc;
    vdoc << vin.rdbuf();
    const auto entries = ParseVocabJson(vdoc.str());

    // Base bytes must be exactly 0..255 single bytes.
    std::vector<std::string> baseBytes(256);
    std::size_t baseCount = 0;
    for (const auto& [id, bytes] : entries) {
        if (id >= 0 && id < 256) {
            if (bytes.size() != 1) {
                throw std::runtime_error("LoadVocabulary: base id with != 1 byte");
            }
            baseBytes[static_cast<std::size_t>(id)] = bytes;
            ++baseCount;
        }
    }
    if (baseCount != 256) {
        throw std::runtime_error("LoadVocabulary: expected 256 base entries");
    }
    for (int b = 0; b < 256; ++b) {
        const std::string want(1, static_cast<char>(static_cast<unsigned char>(b)));
        if (baseBytes[static_cast<std::size_t>(b)] != want) {
            throw std::runtime_error("LoadVocabulary: base byte mismatch");
        }
    }

    std::ifstream min(merges_path, std::ios::in | std::ios::binary);
    if (!min.is_open()) {
        throw std::runtime_error("LoadVocabulary: cannot open '" + merges_path + "'");
    }
    Vocabulary vocab;  // fresh base 0..255
    std::string line;
    int expectId = 256;
    while (std::getline(min, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        // "<a> <b> -> <c>"
        std::istringstream ls(line);
        int a = -1;
        int b = -1;
        int c = -1;
        std::string arrow;
        if (!(ls >> a >> b >> arrow >> c) || arrow != "->" || a < 0 || b < 0 || c != expectId) {
            throw std::runtime_error("LoadVocabulary: malformed merges line: '" + line + "'");
        }
        const int got = vocab.addMerge(a, b);
        if (got != expectId) {
            throw std::runtime_error("LoadVocabulary: merge id mismatch");
        }
        ++expectId;
    }

    // Cross-check merged entries when the vocab file carries them.
    for (const auto& [id, bytes] : entries) {
        if (id < 256) {
            continue;
        }
        if (static_cast<std::size_t>(id) >= vocab.size() || vocab.bytesOf(id) != bytes) {
            throw std::runtime_error("LoadVocabulary: vocab/merges mismatch at id " +
                                     std::to_string(id));
        }
    }
    if (static_cast<int>(vocab.size()) != expectId) {
        throw std::runtime_error("LoadVocabulary: vocab/merges size mismatch");
    }
    return vocab;
}

void SaveVocabBin(const Vocabulary& vocab, const std::string& path) {
    const std::size_t n = vocab.size();
    if (n == 0 || n > 65536) {
        throw std::invalid_argument("SaveVocabBin: vocab size out of range");
    }
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("SaveVocabBin: cannot open '" + path + "'");
    }
    auto putU32 = [&](uint32_t v) {
        // Explicit little-endian bytes (portable across host endianness).
        const char b[4] = {static_cast<char>(v & 0xFF),
                           static_cast<char>((v >> 8) & 0xFF),
                           static_cast<char>((v >> 16) & 0xFF),
                           static_cast<char>((v >> 24) & 0xFF)};
        out.write(b, 4);
    };
    auto putU16 = [&](uint32_t v) {
        if (v > 0xFFFFU) {
            throw std::invalid_argument("SaveVocabBin: id outside uint16 range");
        }
        const char b[2] = {static_cast<char>(v & 0xFF),
                           static_cast<char>((v >> 8) & 0xFF)};
        out.write(b, 2);
    };
    out.write("DOMBPE02", 8);
    putU32(2);  // version 2: padded u16 merge triples + mmapped matrix blob
    putU32(static_cast<uint32_t>(n));
    putU32(static_cast<uint32_t>(vocab.merges().size()));
    uint32_t off = 0;
    for (std::size_t i = 0; i < n; ++i) {
        putU32(off);
        off += static_cast<uint32_t>(vocab.bytesOf(static_cast<int>(i)).size());
    }
    putU32(off);
    for (std::size_t i = 0; i < n; ++i) {
        const std::string& b = vocab.bytesOf(static_cast<int>(i));
        out.write(b.data(), static_cast<std::streamsize>(b.size()));
    }
    for (const auto& m : vocab.merges()) {
        putU16(static_cast<uint32_t>(m.id_a));
        putU16(static_cast<uint32_t>(m.id_b));
        putU16(static_cast<uint32_t>(m.new_id));
        putU16(0);
    }
    // Pad so the matrix that follows starts 64-byte aligned, then append
    // the flat rank table (id+1 per cell, 0 = none) for zero-copy mmap.
    // Layout must mirror BpeEngineV2::FromBinFile.
    {
        const std::streampos pos = out.tellp();
        if (pos == std::streampos(-1)) {
            throw std::runtime_error("SaveVocabBin: tell failed for '" + path + "'");
        }
        const auto used = static_cast<std::size_t>(pos);
        const std::size_t pad = (64 - (used % 64)) % 64;
        static constexpr char kZero[64] = {};
        out.write(kZero, static_cast<std::streamsize>(pad));
        const std::size_t nv = n > 2048 ? 2048 : n;
        std::vector<uint16_t> table(nv * 2048, 0);
        for (const auto& m : vocab.merges()) {
            if (m.id_a < 0 || m.id_b < 0 || m.new_id < 0 || m.new_id > 0xFFFE) {
                throw std::invalid_argument("SaveVocabBin: merge id out of range");
            }
            table[static_cast<std::size_t>(m.id_a) * 2048 +
                  static_cast<std::size_t>(m.id_b)] = static_cast<uint16_t>(m.new_id + 1);
        }
        if constexpr (std::endian::native == std::endian::little) {
            out.write(reinterpret_cast<const char*>(table.data()),
                      static_cast<std::streamsize>(table.size() * sizeof(uint16_t)));
        } else {
            for (uint16_t cell : table) {
                putU16(cell);
            }
        }
    }
    out.flush();
    if (!out) {
        throw std::runtime_error("SaveVocabBin: write failed for '" + path + "'");
    }
}

}  // namespace domlm::tokenizer

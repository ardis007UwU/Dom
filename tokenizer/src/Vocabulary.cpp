#include "tokenizer/bpe/Vocabulary.hpp"

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace domlm::tokenizer {

Vocabulary::Vocabulary() {
    id_to_token_.reserve(kBaseSize);
    token_to_id_.reserve(kBaseSize * 2);
    for (int b = 0; b < 256; ++b) {
        std::string bytes(1, static_cast<char>(static_cast<unsigned char>(b)));
        token_to_id_.emplace(bytes, b);
        id_to_token_.push_back(bytes);
    }
}

bool Vocabulary::contains(std::string_view bytes) const {
    // Heterogeneous lookup would need transparent hashing; copy is tiny here
    // (tokens are short) and keeps the interface simple.
    return token_to_id_.find(std::string(bytes)) != token_to_id_.end();
}

int Vocabulary::idOf(std::string_view bytes) const {
    auto it = token_to_id_.find(std::string(bytes));
    if (it == token_to_id_.end()) {
        throw std::out_of_range("Vocabulary: unknown token bytes");
    }
    return it->second;
}

const std::string& Vocabulary::bytesOf(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
        throw std::out_of_range("Vocabulary: token id out of range");
    }
    return id_to_token_[static_cast<std::size_t>(id)];
}

std::string Vocabulary::displayOf(int id) const {
    const std::string& bytes = bytesOf(id);  // range-checked
    if (static_cast<std::size_t>(id) < kBaseSize) {
        return BaseDisplay(static_cast<unsigned char>(bytes[0]));
    }
    // Merged token: printable ASCII runs verbatim, all other bytes escaped
    // as <0xXX> so display names stay single-line ASCII. Display-only.
    std::string out;
    out.reserve(bytes.size() + 4);
    for (unsigned char c : bytes) {
        if (c >= 0x20 && c <= 0x7E && c != '"' && c != '\\') {
            out.push_back(static_cast<char>(c));
        } else {
            out += BaseDisplay(c);
        }
    }
    return out;
}

int Vocabulary::addMerge(int id_a, int id_b) {
    const auto asSize = static_cast<int>(id_to_token_.size());
    if (id_a < 0 || id_a >= asSize || id_b < 0 || id_b >= asSize) {
        throw std::invalid_argument("Vocabulary: merge component id out of range");
    }
    if (const int existing = idOfPair(id_a, id_b); existing >= 0) {
        return existing;
    }
    const int newId = static_cast<int>(id_to_token_.size());
    std::string bytes;
    bytes.reserve(id_to_token_[static_cast<std::size_t>(id_a)].size() +
                  id_to_token_[static_cast<std::size_t>(id_b)].size());
    bytes += id_to_token_[static_cast<std::size_t>(id_a)];
    bytes += id_to_token_[static_cast<std::size_t>(id_b)];
    token_to_id_.emplace(bytes, newId);
    id_to_token_.push_back(bytes);
    merges_.push_back(MergeRule{id_a, id_b, newId});
    pair_to_id_.emplace(std::pair<int, int>{id_a, id_b}, newId);
    return newId;
}

int Vocabulary::idOfPair(int id_a, int id_b) const noexcept {
    auto it = pair_to_id_.find(std::pair<int, int>{id_a, id_b});
    return it == pair_to_id_.end() ? -1 : it->second;
}

std::string Vocabulary::BaseDisplay(unsigned char byte) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string out("<0x__>");
    out[3] = kHex[(byte >> 4) & 0xF];
    out[4] = kHex[byte & 0xF];
    return out;
}

}  // namespace domlm::tokenizer

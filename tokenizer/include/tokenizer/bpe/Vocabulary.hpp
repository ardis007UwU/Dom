#pragma once

// Phase 2: base byte vocabulary & token representation.
//
// - Internal token representation is std::string byte streams (arbitrary
//   bytes; NOT assumed to be valid UTF-8).
// - IDs 0..255 are pre-initialized to raw single bytes \x00..\xFF.
// - Human-readable display names (<0x00>..<0xFF> for base bytes, escaped
//   ASCII runs for merges) are stored separately from byte payloads and are
//   display-only: never used as lookup keys.
// - Bidirectional lookups: token_to_id (bytes -> id) and id_to_token
//   (id -> bytes, index-addressed).
// - Ordered merge-rule list (id_a, id_b) -> new_id in training order.

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace domlm::tokenizer {

/// One learned merge rule, in training order. `id` is the new token id
/// assigned to bytes(id_a) + bytes(id_b).
struct MergeRule {
    int id_a{-1};
    int id_b{-1};
    int new_id{-1};
};

/// Byte-level vocabulary with insertion-ordered merge rules.
class Vocabulary {
public:
    struct PairHash {
        std::size_t operator()(const std::pair<int, int>& p) const noexcept {
            return static_cast<std::size_t>(p.first) * 1315423911U +
                   static_cast<std::size_t>(p.second);
        }
    };

    /// Builds the base 256-entry byte vocabulary (no merges).
    Vocabulary();

    Vocabulary(const Vocabulary&) = default;
    Vocabulary& operator=(const Vocabulary&) = default;
    Vocabulary(Vocabulary&&) noexcept = default;
    Vocabulary& operator=(Vocabulary&&) noexcept = default;
    ~Vocabulary() = default;

    /// Total entries including base bytes and learned merges.
    [[nodiscard]] std::size_t size() const noexcept { return id_to_token_.size(); }

    /// Ordered merge rules (insertion order == new_id - 256 order).
    [[nodiscard]] const std::vector<MergeRule>& merges() const noexcept { return merges_; }

    [[nodiscard]] bool contains(std::string_view bytes) const;
    /// Throws std::out_of_range when absent.
    [[nodiscard]] int idOf(std::string_view bytes) const;
    /// Throws std::out_of_range when id >= size().
    [[nodiscard]] const std::string& bytesOf(int id) const;
    /// Display-only name; never a lookup key. Throws on bad id.
    [[nodiscard]] std::string displayOf(int id) const;

    /// Returns existing id when (a, b) was merged before; otherwise appends
    /// bytes(a)+bytes(b) as the next id and records the rule.
    /// Throws std::invalid_argument on out-of-range a/b.
    int addMerge(int id_a, int id_b);

    /// Id already assigned to pair (a, b), or -1 when unmerged.
    [[nodiscard]] int idOfPair(int id_a, int id_b) const noexcept;

    /// Base byte count (always 256).
    static constexpr std::size_t kBaseSize = 256;

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<MergeRule> merges_;
    std::unordered_map<std::pair<int, int>, int, PairHash> pair_to_id_;

    static std::string BaseDisplay(unsigned char byte);
};

}  // namespace domlm::tokenizer

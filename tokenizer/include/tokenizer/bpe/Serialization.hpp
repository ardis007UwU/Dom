#pragma once

// Vocabulary serialization: vocab.json + merges.txt round-trip.
//
// vocab.json layout: {"version":1,"vocab_size":N,
//   "tokens":[{"id":0,"text":"..."}, ...]} where "text" is JSON-escaped:
//   printable ASCII (0x20..0x7E except '"' and '\\') verbatim, '"'/\\ and
//   C0 controls via short escapes or \u00XX, all other bytes (0x7F, >=0x80)
//   as \u00XX per byte, keeping the file pure ASCII and safely escaping
//   non-printable sequences.
// merges.txt layout: one "<id_a> <id_b> -> <new_id>" per line, in training
//   order (line i <=> new id 256+i). Blank lines and '#' comments skipped.

#include <string>
#include <string_view>

#include "tokenizer/bpe/Vocabulary.hpp"

namespace domlm::tokenizer {

// JSON-escape one token's raw bytes to pure-ASCII text.
[[nodiscard]] std::string EscapeJsonBytes(std::string_view bytes);

/// Throws std::runtime_error on IO failure.
void SaveVocabJson(const Vocabulary& vocab, const std::string& path);
/// Throws std::runtime_error on IO failure.
void SaveMerges(const Vocabulary& vocab, const std::string& path);

/// Rebuilds the exact vocabulary: base bytes from vocab.json token entries,
/// merge order/rules from merges.txt. Throws std::runtime_error on malformed
/// input or mismatch between the two files.
[[nodiscard]] Vocabulary LoadVocabulary(const std::string& vocab_path,
                                        const std::string& merges_path);

// Compact binary vocabulary for the v2 engine (see BpeEngineV2.hpp for the
// layout: magic + header + offsets + token blob + u16 merge triples).
// Throws std::runtime_error on IO failure or ids outside uint16 range.
void SaveVocabBin(const Vocabulary& vocab, const std::string& path);

}  // namespace domlm::tokenizer

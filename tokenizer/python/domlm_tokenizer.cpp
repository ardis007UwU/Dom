// DomLM v3.0 Python C-extension: `import domlm_tokenizer`.
//
// In-process pybind11 wrapper over the C++ BpeEngineV2 multi-threaded
// encoding engine. No subprocesses, no file-I/O round trips per call: text
// crosses the boundary as UTF-8 bytes, ids come back as Python ints, and the
// GIL is released around every C++ encode/decode call so Python threads stay
// unblocked.
//
// API:
//   tok = domlm_tokenizer.Tokenizer("data/vocab.bin")          # mmap, zero-copy load
//   tok = domlm_tokenizer.Tokenizer.from_json("vocab.json", "merges.txt")
//   tok.encode("hello world")              -> [261, ...]
//   tok.encode_batch(["a", "b"], threads=0) -> [[...], [...]]  (0 = auto)
//   tok.decode([261, ...])                 -> b"hello world" (raw bytes)
//   tok.vocab_size / tok.ruleset / domlm_tokenizer.__version__

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BpeEngineV2.hpp"
#include "domlm/Version.hpp"
#include "tokenizer/bpe/Serialization.hpp"

namespace py = pybind11;
using domlm::engine::BpeEngineV2;

namespace {

// pybind11 has no uint16_t caster issues (casts to int), but converting
// vector<uint16_t> -> list[int] element-wise in C++ avoids surprises and
// keeps the hot loop in native code.
std::vector<int> ToIntList(const std::vector<uint16_t>& ids) {
    std::vector<int> out;
    out.reserve(ids.size());
    for (uint16_t id : ids) {
        out.push_back(static_cast<int>(id));
    }
    return out;
}

std::vector<uint16_t> ToU16List(const std::vector<int>& ids) {
    std::vector<uint16_t> out;
    out.reserve(ids.size());
    for (int id : ids) {
        if (id < 0 || id > 0xFFFF) {
            throw std::out_of_range("token id out of uint16 range");
        }
        out.push_back(static_cast<uint16_t>(id));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(domlm_tokenizer, m) {
    m.attr("__version__") = DOMLM_VERSION_STRING;
    m.doc() = "DomLM v3.0 frozen BPE tokenizer (C++ BpeEngineV2, in-process)";

    py::class_<BpeEngineV2>(m, "Tokenizer")
        .def(py::init([](const std::string& vocab_bin, const std::string& ruleset) {
                 return BpeEngineV2::FromBinFile(vocab_bin, ruleset);
             }),
             py::arg("vocab_bin"), py::arg("ruleset") = "cl100k_base",
             "Load from a vocab.bin mmap artifact (zero-copy, no JSON parsing).")
        .def_static(
            "from_json",
            [](const std::string& vocab_json, const std::string& merges_txt,
               const std::string& ruleset) {
                auto vocab = domlm::tokenizer::LoadVocabulary(vocab_json, merges_txt);
                return BpeEngineV2::FromVocabulary(std::move(vocab), ruleset);
            },
            py::arg("vocab_json"), py::arg("merges_txt"),
            py::arg("ruleset") = "cl100k_base",
            "Load from vocab.json + merges.txt (rebuilds the rank table).")
        .def_property_readonly("vocab_size", &BpeEngineV2::vocabSize)
        .def_property_readonly("ruleset", &BpeEngineV2::ruleset)
        .def(
            "encode",
            [](const BpeEngineV2& self, const std::string& text) {
                py::gil_scoped_release release;
                return ToIntList(self.encode(text));
            },
            py::arg("text"), "Encode one string -> list[int] token ids.")
        .def(
            "encode_batch",
            [](BpeEngineV2& self, const std::vector<std::string>& lines,
               std::size_t threads) {
                py::gil_scoped_release release;
                std::vector<std::vector<int>> out;
                out.reserve(lines.size());
                for (const auto& ids : self.encodeLines(lines, threads)) {
                    out.push_back(ToIntList(ids));
                }
                return out;
            },
            py::arg("lines"), py::arg("threads") = 0,
            "Encode many strings (jthread pool, order-preserving). threads=0 "
            "selects automatically; threads=1 is strict single-thread.")
        .def(
            "decode",
            [](const BpeEngineV2& self, const std::vector<int>& ids) {
                std::string raw;
                {
                    py::gil_scoped_release release;
                    raw = self.decode(ToU16List(ids));
                }
                return py::bytes(raw);  // GIL held again: Python API is safe
            },
            py::arg("ids"),
            "Decode ids -> raw bytes (returned as bytes: ids may encode "
            "arbitrary byte streams, not just valid UTF-8).");
}

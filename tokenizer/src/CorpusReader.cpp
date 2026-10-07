#include "tokenizer/bpe/CorpusReader.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace domlm::tokenizer {

std::string_view DefaultFileName(CorpusSplit split) noexcept {
    switch (split) {
        case CorpusSplit::Train:
            return "train.txt";
        case CorpusSplit::Valid:
            return "valid.txt";
        case CorpusSplit::Test:
            return "test.txt";
    }
    return "train.txt";
}

CorpusReader::CorpusReader(const std::string& path, CorpusSplit split)
    : in_(path, std::ios::in | std::ios::binary), path_(path), split_(split) {
    if (!in_.is_open()) {
        throw std::runtime_error("CorpusReader: cannot open '" + path + "'");
    }
}

bool CorpusReader::nextLine(std::string& line) {
    if (!std::getline(in_, line)) {
        return false;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    ++lines_read_;
    return true;
}

bool CorpusReader::isOpen() const {
    return in_.is_open();
}

}  // namespace domlm::tokenizer

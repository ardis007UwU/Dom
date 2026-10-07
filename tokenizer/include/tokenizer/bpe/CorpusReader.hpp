#pragma once

// Phase 2: streaming corpus ingestion.
//
// - Streams text files line-by-line (std::getline) so peak memory is one
//   line, never the whole file.
// - Training vs evaluation separation is by path: construct one reader over
//   the training corpus (train.txt) and separate reader instances over
//   validation / test corpora (valid.txt / test.txt). The CorpusSplit label
//   records each reader's role; default file-name hints match the repo's
//   PreTrain/*.txt layout.

#include <cstddef>
#include <fstream>
#include <string>
#include <string_view>

namespace domlm::tokenizer {

/// Role of a corpus file. Training must only consume Train readers;
///
/// evaluation only Valid/Test readers.
enum class CorpusSplit { Train, Valid, Test };

[[nodiscard]] std::string_view DefaultFileName(CorpusSplit split) noexcept;

/// Line-by-line file reader (RAII: ifstream owned, no raw new/delete).
class CorpusReader {
public:
    explicit CorpusReader(const std::string& path,
                          CorpusSplit split = CorpusSplit::Train);

    CorpusReader(const CorpusReader&) = delete;
    CorpusReader& operator=(const CorpusReader&) = delete;
    CorpusReader(CorpusReader&&) noexcept = default;
    CorpusReader& operator=(CorpusReader&&) noexcept = default;
    ~CorpusReader() = default;

    /// Next line without its terminator (a trailing '\r' is stripped for
    /// CRLF files). Returns false at EOF. Empty lines yield `line` empty
    /// with a true return.
    bool nextLine(std::string& line);

    [[nodiscard]] bool isOpen() const;
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] CorpusSplit split() const noexcept { return split_; }
    [[nodiscard]] std::size_t linesRead() const noexcept { return lines_read_; }

private:
    std::ifstream in_;
    std::string path_;
    CorpusSplit split_{CorpusSplit::Train};
    std::size_t lines_read_{0};
};

}  // namespace domlm::tokenizer

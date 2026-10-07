#pragma once

// Dom SLM v2.0 hyper-speed BPE inference engine.
//
// Zero-overhead execution paths:
//  1. Flat 2D merge matrix: rows of uint16_t[2048], each row 64-byte
//     aligned (alignas(64)); O(1) lookup rank = matrix[a][b] (0 = none,
//     else merged_id + 1, so id 0 stays representable).
//  2. std::jthread pool with lock-free atomic dispatch (fetch_add over
//     block indices; mutexes only sleep/wake, never in the hot path).
//  3. vocab.bin zero-copy load via mmap (Linux) / MapViewOfFile (Windows):
//     header + offsets + token blob + merge table map straight into memory,
//     no JSON/text parsing on the cold path.
//  4. AVX2 256-bit newline scan (_mm256_cmpeq_epi8 + movemask) to snap
//     parallel block boundaries to line starts; scalar fallback otherwise.
//  5. In-place doubly-linked-list arena (preallocated id/next/prev arrays):
//     merges splice pointers, no vector shifts.
//  6. Branchless best-rank selection (integer predicates -> cmov).
//  7. _mm_prefetch of upcoming matrix rows 4 iterations ahead of scans.
//  8. Double-array trie (base/check) single-pass special-token parser
//     (<|im_start|>, <|im_end|>, <|endoftext|>, <|pad|>).
//
// Correctness contract: byte-identical output to domlm::tokenizer::Tokenizer
// (v1) for all inputs without special tokens. Chunking reuses the exact
// Phase-1 PCRE2 pre-tokenizer; only the merge-application path is new.
//
// Pure C++20 STL + standard OS APIs (+ x86 intrinsics, guarded).

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "tokenizer/bpe/Vocabulary.hpp"
#include "tokenizer/pre/PreTokenizerPipeline.hpp"

namespace domlm::engine {

// Fixed Dom vocabulary geometry: 256 base bytes + 1792 merges.
inline constexpr int kMaxVocab = 2048;
inline constexpr int kBaseVocab = 256;
// Stored matrix cell: 0 = no merge, otherwise merged_id + 1.
inline constexpr uint16_t kNoMerge = 0;

// Special tokens resolved outside the 2048 base vocabulary; when allowed
// they encode as vocabSize()+index (2048..2051) and decode back by name.
inline constexpr const char* kSpecialTokens[] = {
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|pad|>",
};
inline constexpr int kSpecialCount = 4;

// vocab.bin layout (all integers little-endian):
//   char   magic[8]      == {'D','O','M','B','P','E','0','2'}
//   uint32 version       == 2 (1 = legacy, no matrix blob)
//   uint32 vocab_size           (2048)
//   uint32 merge_count          (1792)
//   uint32 offsets[vocab_size+1] (byte offsets into blob)
//   uint8  blob                 (concatenated token bytes)
//   uint8  pad to 64 B          (matrix alignment; version 2 only)
//   struct { uint16 a, b, id; uint16 pad; } merges[merge_count]
//   MergeRow matrix[vocab_size]  (version 2 only; 8 MiB, demand-paged)
// Version-1 files are still accepted (matrix rebuilt from merges).
struct VocabBinHeader {
    char magic[8];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t merge_count;
};
inline constexpr uint32_t kVocabBinVersion = 2;

// One matrix row: 2048*2 B = 64 cache lines, 64-byte aligned start.
struct alignas(64) MergeRow {
    uint16_t m[kMaxVocab];
};

// O(1) rank table. Lookup returns merged id, or -1 when unmerged.
class MergeMatrix {
public:
    MergeMatrix();
    void build(const domlm::tokenizer::Vocabulary& vocab);
    void build(const std::vector<std::tuple<int, int, int>>& merges, int vocabSize);
    // Adopt externally-owned (e.g. mmap-backed) storage. Rows must be
    // 64-byte aligned; the caller keeps the mapping alive (engine lifetime
    // for FromBinFile). Replaces any owned storage.
    void adopt(const MergeRow* rows, int vocabSize);
    // Returns merged token id, or -1. Out-of-range ids return -1.
    [[nodiscard]] int rank(int idA, int idB) const noexcept;
    [[nodiscard]] int vocabSize() const noexcept { return vocabSize_; }
    [[nodiscard]] std::size_t memBytes() const noexcept {
        return rows_.size() * sizeof(MergeRow);
    }
    [[nodiscard]] bool isMapped() const noexcept { return mapped_; }
    // Direct row access for the prefetching hot loop.
    [[nodiscard]] const MergeRow* memRows() const noexcept { return rowsPtr_; }

private:
    std::vector<MergeRow> rows_;  // owned storage (build path)
    const MergeRow* rowsPtr_ = nullptr;
    bool mapped_ = false;
    int vocabSize_ = 0;
};

// Double-array trie (base/check) over raw bytes for special tokens.
class SpecialTrie {
public:
    SpecialTrie();
    void build(const char* const* patterns, int count);
    // Length of the pattern starting exactly at pos, or -1.
    [[nodiscard]] int matchAt(std::string_view text, std::size_t pos) const noexcept;
    // Byte index of the pattern id matched, or -1.
    [[nodiscard]] int idAt(std::string_view text, std::size_t pos) const noexcept;
    // First position >= from where any pattern starts, or npos.
    [[nodiscard]] std::size_t findFirst(std::string_view text,
                                        std::size_t from) const noexcept;

private:
    std::vector<int> base_;
    std::vector<int> check_;
    std::vector<int> out_;  // node -> pattern id, -1 when none
};

// RAII read-only memory mapping (mmap / MapViewOfFile).
class MmapFile {
public:
    MmapFile();
    ~MmapFile();
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;
    void open(const std::string& path);  // throws on failure
    void close() noexcept;
    [[nodiscard]] bool isOpen() const noexcept { return data_ != nullptr; }
    [[nodiscard]] const uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    const uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
#if defined(_WIN32)
    void* fileHandle_ = nullptr;
    void* mapHandle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// Lock-free-dispatch jthread pool: the hot path is a single atomic
// fetch_add per block. Mutexes/condition variables only sleep and wake
// workers between parallel_for calls, never during dispatch.
class JThreadPool {
public:
    explicit JThreadPool(std::size_t numThreads) : shared_(std::make_shared<Shared>()) {
        if (numThreads == 0) {
            throw std::invalid_argument("JThreadPool: need >= 1 thread");
        }
        workers_.reserve(numThreads);
        shared_->workers = numThreads;  // visible before any worker runs
        for (std::size_t i = 0; i < numThreads; ++i) {
            workers_.emplace_back([s = shared_](std::stop_token stop) {
                std::size_t myGen = 0;
                for (;;) {
                    // Wait for a new generation (or stop).
                    {
                        std::unique_lock<std::mutex> lk(s->mutex);
                        s->wake.wait(lk, [&] {
                            return stop.stop_requested() || s->generation != myGen;
                        });
                        if (stop.stop_requested()) {
                            return;
                        }
                        myGen = s->generation;
                    }
                    // Lock-free work stealing across the block range.
                    for (;;) {
                        const std::size_t idx =
                            s->cursor.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= s->total) {
                            break;
                        }
                        s->task(idx);
                    }
                    if (s->done.fetch_add(1, std::memory_order_acq_rel) + 1 ==
                        s->workers) {
                        std::lock_guard<std::mutex> lk(s->mutex);
                        s->finished = true;
                        s->doneCv.notify_one();
                    }
                }
            });
        }
    }
    ~JThreadPool() {
        // condition_variable::wait does not observe jthread stop requests,
        // so wake sleepers explicitly before joining.
        for (auto& w : workers_) {
            w.request_stop();
        }
        shared_->wake.notify_all();
    }
    JThreadPool(const JThreadPool&) = delete;
    JThreadPool& operator=(const JThreadPool&) = delete;

    // Runs fn(i) for i in [0, count) across workers; returns when done.
    // count==0 returns immediately. Called from one thread at a time.
    template <typename Fn>
    void parallelFor(std::size_t count, Fn&& fn) {
        if (count == 0) {
            return;
        }
        auto& s = *shared_;
        {
            std::lock_guard<std::mutex> lk(s.mutex);
            s.task = std::function<void(std::size_t)>(std::forward<Fn>(fn));
            s.cursor.store(0, std::memory_order_relaxed);
            s.total = count;
            s.done.store(0, std::memory_order_relaxed);
            s.finished = false;
            ++s.generation;
        }
        s.wake.notify_all();
        std::unique_lock<std::mutex> lk(s.mutex);
        s.doneCv.wait(lk, [&] { return s.finished; });
        s.task = nullptr;
    }

    [[nodiscard]] std::size_t numThreads() const noexcept { return workers_.size(); }

private:
    struct Shared {
        std::mutex mutex;
        std::condition_variable wake;
        std::condition_variable doneCv;
        std::function<void(std::size_t)> task;
        std::atomic<std::size_t> cursor{0};
        std::size_t total = 0;
        std::atomic<std::size_t> done{0};
        std::size_t workers = 0;
        std::size_t generation = 0;
        bool finished = false;
    };
    std::shared_ptr<Shared> shared_;
    std::vector<std::jthread> workers_;
};

// Piece of input split around special-token spans.
struct Piece {
    std::string_view text;
    bool isSpecial = false;
    int specialIdx = -1;
};

class BpeEngineV2 {
public:
    // Build from an in-memory vocabulary (matrix filled directly).
    static BpeEngineV2 FromVocabulary(domlm::tokenizer::Vocabulary vocab, const std::string& ruleset);
    // Zero-copy load from vocab.bin (mmap; no JSON parsing).
    static BpeEngineV2 FromBinFile(const std::string& path, const std::string& ruleset);

    BpeEngineV2(const BpeEngineV2&) = delete;
    BpeEngineV2& operator=(const BpeEngineV2&) = delete;
    BpeEngineV2(BpeEngineV2&&) noexcept = default;
    BpeEngineV2& operator=(BpeEngineV2&&) noexcept = default;
    ~BpeEngineV2() = default;

    [[nodiscard]] int vocabSize() const noexcept { return vocabSize_; }
    [[nodiscard]] const std::string& ruleset() const noexcept { return ruleset_; }
    [[nodiscard]] const MergeMatrix& matrix() const noexcept { return matrix_; }
    [[nodiscard]] const SpecialTrie& specials() const noexcept { return trie_; }

    // Split out special-token spans (single DAT pass, no backtracking).
    [[nodiscard]] std::vector<Piece> splitSpecials(std::string_view text) const;

    // Encode one text. Special spans throw unless allowSpecials (then they
    // encode as vocabSize()+idx). Ids fit uint16_t (Dom uint16 alignment).
    [[nodiscard]] std::vector<uint16_t> encode(std::string_view text,
                                               bool allowSpecials = false) const;
    void encodeInto(std::string_view text, std::vector<uint16_t>& out,
                    bool allowSpecials = false) const;
    [[nodiscard]] std::string decode(std::span<const uint16_t> ids) const;

    // Batch encode; threads==1 is strict single-thread. Output order matches
    // input order; results are identical for any thread count.
    [[nodiscard]] std::vector<std::vector<uint16_t>> encodeLines(
        const std::vector<std::string>& lines, std::size_t threads = 1,
        bool allowSpecials = false);
    // Same, additionally recording per-line encode latency (microseconds)
    // into lineUs (cleared first; one entry per line, disjoint worker writes).
    // Enables honest P50/P90/P99 under parallel execution.
    void encodeLines(const std::vector<std::string>& lines,
                     std::vector<std::vector<uint16_t>>& out,
                     std::vector<double>& lineUs, std::size_t threads = 1,
                     bool allowSpecials = false);
    // mmap file -> AVX2 newline-aligned blocks -> parallel line encode.
    [[nodiscard]] std::vector<std::vector<uint16_t>> encodeFileLines(
        const std::string& path, std::size_t threads, bool allowSpecials = false);

private:
    BpeEngineV2() = default;

    struct Scratch {
        std::vector<int> ids;
        std::vector<int> nxt;
        std::vector<int> prv;
    };
    void encodeChunk(std::string_view chunk, std::vector<uint16_t>& out,
                     Scratch& scratch) const;
    void encodeNormal(std::string_view text, std::vector<uint16_t>& out, Scratch& scratch,
                      const domlm::tokenizer::PreTokenizerPipeline& pipeline) const;
    // One line (specials split + normal chunks); buf is cleared first.
    // Shared by the single- and multi-threaded batch paths.
    void encodeOneLine(std::string_view line, std::vector<uint16_t>& buf, Scratch& scratch,
                       const domlm::tokenizer::PreTokenizerPipeline& pipeline,
                       bool allowSpecials) const;

    MergeMatrix matrix_;
    SpecialTrie trie_;
    std::vector<std::string> ownedBytes_;     // decode source (non-mmap path)
    std::vector<std::string_view> bytes_;    // views into ownedBytes_ or mapping_
    MmapFile mapping_;                       // alive iff loaded from vocab.bin
    domlm::tokenizer::PreTokenizerPipeline pipeline_;  // compiled once per engine
    int vocabSize_ = 0;
    std::string ruleset_ = "cl100k_base";
};

}  // namespace domlm::engine

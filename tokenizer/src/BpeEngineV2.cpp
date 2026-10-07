#include "BpeEngineV2.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace domlm::engine {

namespace {

// ---- tiny arch helpers ----------------------------------------------------

inline bool HasAvx2() noexcept {
#if defined(__x86_64__)
    static const bool v = __builtin_cpu_supports("avx2");
    return v;
#else
    return false;
#endif
}

inline void PrefetchRow(const MergeRow* rows, int id) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_prefetch(reinterpret_cast<const char*>(&rows[id].m[0]), _MM_HINT_T0);
#else
    (void)rows;
    (void)id;
#endif
}

#if defined(__x86_64__)
// AVX2 newline scan: returns pointer to first '\n' in [p, end), or the first
// unaligned tail position when none found in full 32B lanes (scalar finishes).
__attribute__((target("avx2"))) const uint8_t* FindNlAvx2(const uint8_t* p,
                                                               const uint8_t* end) {
    const __m256i nl = _mm256_set1_epi8('\n');
    for (; p + 32 <= end; p += 32) {
        const __m256i v =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const auto mask =
            static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, nl)));
        if (mask != 0) {
            return p + __builtin_ctz(mask);
        }
    }
    return p;
}
#endif

const uint8_t* FindNextNewline(const uint8_t* p, const uint8_t* end) noexcept {
#if defined(__x86_64__)
    if (HasAvx2()) {
        p = FindNlAvx2(p, end);
    }
#endif
    while (p < end && *p != '\n') {
        ++p;
    }
    return p;
}

}  // namespace

// ---- MergeMatrix ----------------------------------------------------------

MergeMatrix::MergeMatrix() = default;

void MergeMatrix::adopt(const MergeRow* rows, int vocabSize) {
    if (rows == nullptr || vocabSize < 0 || vocabSize > kMaxVocab) {
        throw std::invalid_argument("MergeMatrix: bad adopted storage");
    }
    rows_.clear();
    rows_.shrink_to_fit();
    rowsPtr_ = rows;
    mapped_ = true;
    vocabSize_ = vocabSize;
}

void MergeMatrix::build(const domlm::tokenizer::Vocabulary& vocab) {
    std::vector<std::tuple<int, int, int>> merges;
    merges.reserve(vocab.merges().size());
    for (const auto& m : vocab.merges()) {
        merges.emplace_back(m.id_a, m.id_b, m.new_id);
    }
    build(merges, static_cast<int>(vocab.size()));
}

void MergeMatrix::build(const std::vector<std::tuple<int, int, int>>& merges,
                        int vocabSize) {
    if (vocabSize < 0 || vocabSize > kMaxVocab) {
        throw std::invalid_argument("MergeMatrix: vocab size out of range");
    }
    static_assert(sizeof(MergeRow) == 4096, "row must be 64 cache lines");
    static_assert(alignof(MergeRow) >= 64, "rows must be 64-byte aligned");
    // Single zeroed allocation (8 MiB for the full Dom geometry).
    rows_.assign(kMaxVocab, MergeRow{});
    rowsPtr_ = rows_.data();
    mapped_ = false;
    for (const auto& [a, b, id] : merges) {
        if (a < 0 || a >= vocabSize || b < 0 || b >= vocabSize || id < kBaseVocab ||
            id >= vocabSize) {
            throw std::invalid_argument("MergeMatrix: merge id out of range");
        }
        rows_[static_cast<std::size_t>(a)].m[static_cast<std::size_t>(b)] =
            static_cast<uint16_t>(id + 1);
    }
    vocabSize_ = vocabSize;
}

int MergeMatrix::rank(int idA, int idB) const noexcept {
    if (rowsPtr_ == nullptr || idA < 0 || idA >= kMaxVocab || idB < 0 ||
        idB >= kMaxVocab) {
        return -1;
    }
    const uint16_t s =
        rowsPtr_[static_cast<std::size_t>(idA)].m[static_cast<std::size_t>(idB)];
    return s == kNoMerge ? -1 : static_cast<int>(s) - 1;
}

// ---- SpecialTrie (double-array) -------------------------------------------

SpecialTrie::SpecialTrie() : base_{0}, check_{-1}, out_{-1} {}

void SpecialTrie::build(const char* const* patterns, int count) {
    if (count <= 0) {
        throw std::invalid_argument("SpecialTrie: need >= 1 pattern");
    }
    // Transient raw trie.
    struct RawNode {
        int child[256];
        int out;
        RawNode() : out(-1) { std::fill(child, child + 256, -1); }
    };
    std::vector<RawNode> raw(1);
    for (int p = 0; p < count; ++p) {
        const std::string_view pat(patterns[p]);
        if (pat.empty()) {
            throw std::invalid_argument("SpecialTrie: empty pattern");
        }
        int node = 0;
        for (unsigned char c : pat) {
            if (raw[static_cast<std::size_t>(node)].child[c] < 0) {
                raw[static_cast<std::size_t>(node)].child[c] =
                    static_cast<int>(raw.size());
                raw.emplace_back();
            }
            node = raw[static_cast<std::size_t>(node)].child[c];
        }
        raw[static_cast<std::size_t>(node)].out = p;
    }
    // First-fit double-array placement (grow on demand; no relocation needed
    // for correctness at this pattern scale).
    base_.assign(1, 0);
    check_.assign(1, -1);
    out_.assign(1, -1);
    auto ensure = [&](std::size_t n) {
        if (base_.size() < n) {
            base_.resize(n, 0);
            check_.resize(n, -1);
            out_.resize(n, -1);
        }
    };
    std::function<void(int, int)> place = [&](int state, int rawNode) {
        int b = 1;
        for (;; ++b) {
            bool ok = true;
            for (int c = 0; c < 256; ++c) {
                if (raw[static_cast<std::size_t>(rawNode)].child[c] >= 0) {
                    ensure(static_cast<std::size_t>(b + c + 1));
                    if (check_[static_cast<std::size_t>(b + c)] != -1) {
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) {
                break;
            }
        }
        base_[static_cast<std::size_t>(state)] = b;
        // Reserve ALL outgoing edges before recursing: a child's subtree
        // placement must observe its siblings' slots as taken, otherwise a
        // first sibling's subtree can steal a later sibling's edge slot
        // (whose check_ entry is still -1) and corrupt both edges.
        for (int c = 0; c < 256; ++c) {
            const int child = raw[static_cast<std::size_t>(rawNode)].child[c];
            if (child >= 0) {
                const int t = b + c;
                check_[static_cast<std::size_t>(t)] = state;
                out_[static_cast<std::size_t>(t)] =
                    raw[static_cast<std::size_t>(child)].out;
            }
        }
        for (int c = 0; c < 256; ++c) {
            const int child = raw[static_cast<std::size_t>(rawNode)].child[c];
            if (child >= 0) {
                place(b + c, child);
            }
        }
    };
    place(0, 0);
}

int SpecialTrie::matchAt(std::string_view text, std::size_t pos) const noexcept {
    int state = 0;
    int hitLen = -1;
    for (std::size_t k = pos; k < text.size(); ++k) {
        const auto t = static_cast<std::size_t>(
            base_[static_cast<std::size_t>(state)] +
            static_cast<unsigned char>(text[k]));
        if (t >= check_.size() || check_[t] != state) {
            break;
        }
        state = static_cast<int>(t);
        if (out_[static_cast<std::size_t>(state)] >= 0) {
            hitLen = static_cast<int>(k - pos + 1);  // prefer longest
        }
    }
    return hitLen;
}

int SpecialTrie::idAt(std::string_view text, std::size_t pos) const noexcept {
    int state = 0;
    int hitId = -1;
    for (std::size_t k = pos; k < text.size(); ++k) {
        const auto t = static_cast<std::size_t>(
            base_[static_cast<std::size_t>(state)] +
            static_cast<unsigned char>(text[k]));
        if (t >= check_.size() || check_[t] != state) {
            break;
        }
        state = static_cast<int>(t);
        if (out_[static_cast<std::size_t>(state)] >= 0) {
            hitId = out_[static_cast<std::size_t>(state)];
        }
    }
    return hitId;
}

std::size_t SpecialTrie::findFirst(std::string_view text, std::size_t from) const noexcept {
    const int rootBase = base_.empty() ? 0 : base_[0];
    for (std::size_t p = from; p < text.size(); ++p) {
        // O(1) root transition rejects ~all positions with one array access.
        const auto t = static_cast<std::size_t>(
            rootBase + static_cast<unsigned char>(text[p]));
        if (t < check_.size() && check_[t] == 0 && matchAt(text, p) >= 0) {
            return p;
        }
    }
    return std::string_view::npos;
}

// ---- MmapFile ---------------------------------------------------------------

MmapFile::MmapFile() = default;

MmapFile::~MmapFile() {
    close();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : data_(other.data_),
      size_(other.size_)
#if defined(_WIN32)
      ,
      fileHandle_(other.fileHandle_),
      mapHandle_(other.mapHandle_)
#else
      ,
      fd_(other.fd_)
#endif
{
    other.data_ = nullptr;
    other.size_ = 0;
#if defined(_WIN32)
    other.fileHandle_ = nullptr;
    other.mapHandle_ = nullptr;
#else
    other.fd_ = -1;
#endif
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        close();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
#if defined(_WIN32)
        fileHandle_ = other.fileHandle_;
        mapHandle_ = other.mapHandle_;
        other.fileHandle_ = nullptr;
        other.mapHandle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
    }
    return *this;
}

void MmapFile::open(const std::string& path) {
#if defined(_WIN32)
    void* fh =
        CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("MmapFile: cannot open '" + path + "'");
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(fh, &sz) || sz.QuadPart <= 0) {
        CloseHandle(fh);
        throw std::runtime_error("MmapFile: bad size '" + path + "'");
    }
    void* mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mh == nullptr) {
        CloseHandle(fh);
        throw std::runtime_error("MmapFile: mapping failed '" + path + "'");
    }
    const uint8_t* p =
        static_cast<const uint8_t*>(MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0));
    if (p == nullptr) {
        CloseHandle(mh);
        CloseHandle(fh);
        throw std::runtime_error("MmapFile: view failed '" + path + "'");
    }
    data_ = p;
    size_ = static_cast<std::size_t>(sz.QuadPart);
    fileHandle_ = fh;
    mapHandle_ = mh;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("MmapFile: cannot open '" + path + "'");
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("MmapFile: bad size '" + path + "'");
    }
    void* p = ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ,
                     MAP_PRIVATE, fd, 0);
    ::close(fd);  // mapping persists without the fd
    if (p == MAP_FAILED) {
        throw std::runtime_error("MmapFile: mmap failed '" + path + "'");
    }
    data_ = static_cast<const uint8_t*>(p);
    size_ = static_cast<std::size_t>(st.st_size);
    fd_ = -1;
#endif
}

void MmapFile::close() noexcept {
    if (data_ == nullptr) {
        return;
    }
#if defined(_WIN32)
    UnmapViewOfFile(data_);
    if (mapHandle_ != nullptr) {
        CloseHandle(mapHandle_);
    }
    if (fileHandle_ != nullptr) {
        CloseHandle(fileHandle_);
    }
    fileHandle_ = nullptr;
    mapHandle_ = nullptr;
#else
    ::munmap(const_cast<uint8_t*>(data_), size_);
#endif
    data_ = nullptr;
    size_ = 0;
}

// ---- BpeEngineV2 ------------------------------------------------------------

namespace {

uint32_t ReadU32LE(const uint8_t* p) noexcept {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t ReadU16LE(const uint8_t* p) noexcept {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

}  // namespace

BpeEngineV2 BpeEngineV2::FromVocabulary(domlm::tokenizer::Vocabulary vocab, const std::string& ruleset) {
    if (ruleset != "cl100k_base" && ruleset != "o200k_base") {
        throw std::invalid_argument("BpeEngineV2: unknown ruleset '" + ruleset + "'");
    }
    if (static_cast<int>(vocab.size()) > kMaxVocab) {
        throw std::invalid_argument("BpeEngineV2: vocab exceeds 2048");
    }
    BpeEngineV2 e;
    e.matrix_.build(vocab);
    e.trie_.build(kSpecialTokens, kSpecialCount);
    e.pipeline_.loadPreTokenizerRuleset(ruleset);  // compiled once per engine
    e.ownedBytes_.reserve(vocab.size());
    for (std::size_t i = 0; i < vocab.size(); ++i) {
        e.ownedBytes_.push_back(vocab.bytesOf(static_cast<int>(i)));
    }
    e.bytes_.reserve(e.ownedBytes_.size());
    for (const auto& b : e.ownedBytes_) {
        e.bytes_.emplace_back(b.data(), b.size());
    }
    e.vocabSize_ = static_cast<int>(vocab.size());
    e.ruleset_ = ruleset;
    return e;
}

BpeEngineV2 BpeEngineV2::FromBinFile(const std::string& path, const std::string& ruleset) {
    if (ruleset != "cl100k_base" && ruleset != "o200k_base") {
        throw std::invalid_argument("BpeEngineV2: unknown ruleset '" + ruleset + "'");
    }
    BpeEngineV2 e;
    e.mapping_.open(path);  // zero-copy: views below alias this mapping
    const uint8_t* p = e.mapping_.data();
    const std::size_t n = e.mapping_.size();
    if (n < sizeof(VocabBinHeader)) {
        throw std::runtime_error("BpeEngineV2: truncated vocab.bin '" + path + "'");
    }
    static constexpr char kMagic[8] = {'D', 'O', 'M', 'B', 'P', 'E', '0', '2'};
    if (std::memcmp(p, kMagic, 8) != 0) {
        throw std::runtime_error("BpeEngineV2: bad magic '" + path + "'");
    }
    const uint32_t version = ReadU32LE(p + 8);
    const uint32_t vocabSize = ReadU32LE(p + 12);
    const uint32_t mergeCount = ReadU32LE(p + 16);
    if ((version != 1 && version != kVocabBinVersion) || vocabSize == 0 ||
        vocabSize > kMaxVocab) {
        throw std::runtime_error("BpeEngineV2: bad header '" + path + "'");
    }
    const std::size_t offBase = 20;
    const std::size_t blobBase = offBase + (static_cast<std::size_t>(vocabSize) + 1) * 4;
    if (n < blobBase) {
        throw std::runtime_error("BpeEngineV2: truncated offsets '" + path + "'");
    }
    // Bounds-check the offset table before trusting it.
    uint32_t prev = 0;
    for (uint32_t i = 0; i <= vocabSize; ++i) {
        const uint32_t o = ReadU32LE(p + offBase + i * 4);
        if (o < prev) {
            throw std::runtime_error("BpeEngineV2: corrupt offsets '" + path + "'");
        }
        prev = o;
    }
    const std::size_t blobEnd = blobBase + prev;
    const std::size_t mergesEnd = blobEnd + static_cast<std::size_t>(mergeCount) * 8;
    if (n < mergesEnd) {
        throw std::runtime_error("BpeEngineV2: truncated body '" + path + "'");
    }
    e.bytes_.reserve(vocabSize);
    for (uint32_t i = 0; i < vocabSize; ++i) {
        const uint32_t a = ReadU32LE(p + offBase + i * 4);
        const uint32_t b = ReadU32LE(p + offBase + (i + 1) * 4);
        const char* s = reinterpret_cast<const char*>(p + blobBase + a);
        e.bytes_.emplace_back(s, b - a);
    }
    std::vector<std::tuple<int, int, int>> merges;
    merges.reserve(mergeCount);
    for (uint32_t i = 0; i < mergeCount; ++i) {
        const uint8_t* m = p + blobEnd + i * 8;
        merges.emplace_back(ReadU16LE(m), ReadU16LE(m + 2), ReadU16LE(m + 4));
    }
    if (version == kVocabBinVersion) {
        // Zero-copy rank table: demand-paged straight from the mapping.
        // blobEnd + merges may be unaligned, hence the writer's 64 B pad.
        std::size_t matBase = blobEnd + static_cast<std::size_t>(mergeCount) * 8;
        matBase = (matBase + 63) & ~static_cast<std::size_t>(63);
        if (n < matBase + static_cast<std::size_t>(vocabSize) * sizeof(MergeRow)) {
            throw std::runtime_error("BpeEngineV2: truncated matrix '" + path + "'");
        }
        const auto* rows = reinterpret_cast<const MergeRow*>(p + matBase);
        if (reinterpret_cast<uintptr_t>(rows) % alignof(MergeRow) != 0) {
            throw std::runtime_error("BpeEngineV2: misaligned matrix '" + path + "'");
        }
        e.matrix_.adopt(rows, static_cast<int>(vocabSize));
    } else {
        e.matrix_.build(merges, static_cast<int>(vocabSize));
    }
    e.trie_.build(kSpecialTokens, kSpecialCount);
    e.pipeline_.loadPreTokenizerRuleset(ruleset);  // compiled once per engine
    e.vocabSize_ = static_cast<int>(vocabSize);
    e.ruleset_ = ruleset;
    return e;
}

std::vector<Piece> BpeEngineV2::splitSpecials(std::string_view text) const {
    std::vector<Piece> pieces;
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t hit = trie_.findFirst(text, pos);
        if (hit == std::string_view::npos) {
            pieces.push_back(Piece{text.substr(pos), false, -1});
            break;
        }
        if (hit > pos) {
            pieces.push_back(Piece{text.substr(pos, hit - pos), false, -1});
        }
        const int len = trie_.matchAt(text, hit);
        const int id = trie_.idAt(text, hit);
        pieces.push_back(Piece{text.substr(hit, static_cast<std::size_t>(len)), true, id});
        pos = hit + static_cast<std::size_t>(len);
    }
    return pieces;
}

void BpeEngineV2::encodeChunk(std::string_view chunk, std::vector<uint16_t>& out,
                              Scratch& scratch) const {
    const std::size_t n = chunk.size();
    if (n == 0) {
        return;
    }
    if (n > static_cast<std::size_t>(0x7fffffff)) {
        throw std::length_error("BpeEngineV2: chunk too large for arena");
    }
    // Arena setup: id/next/prev, capacity reused across chunks.
    auto& ids = scratch.ids;
    auto& nxt = scratch.nxt;
    auto& prv = scratch.prv;
    ids.resize(n);
    nxt.resize(n);
    prv.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        ids[i] = static_cast<unsigned char>(chunk[i]);
        nxt[i] = (i + 1 < n) ? static_cast<int>(i + 1) : -1;
        prv[i] = (i > 0) ? static_cast<int>(i - 1) : -1;
    }
    int head = 0;
    const MergeRow* rows = matrix_.memRows();

    for (;;) {
        // Branchless best-rank scan over live adjacent pairs.
        uint32_t bestS = 0xFFFFFFFFU;
        int bestPos = -1;
        int i = head;
        uint32_t step = 0;
        while (i >= 0) {
            const int j = nxt[static_cast<std::size_t>(i)];
            if (j < 0) {
                break;
            }
            // Prefetch the matrix row needed ~4 iterations ahead.
            if ((step & 3U) == 0) {
                int t = i;
                for (int k = 0; k < 4 && t >= 0; ++k) {
                    t = nxt[static_cast<std::size_t>(t)];
                }
                if (t >= 0) {
                    PrefetchRow(rows, ids[static_cast<std::size_t>(t)]);
                }
            }
            ++step;
            const uint16_t s =
                rows[static_cast<std::size_t>(ids[static_cast<std::size_t>(i)])]
                    .m[static_cast<std::size_t>(ids[static_cast<std::size_t>(j)])];
            const uint32_t isRanked = (s != kNoMerge);  // setcc, no mispredict
            const uint32_t better = isRanked & (s < bestS);
            bestS = better ? s : bestS;      // cmov
            bestPos = better ? i : bestPos;  // cmov
            i = j;
        }
        if (bestPos < 0) {
            break;  // no mergeable pair left in this chunk
        }
        // Sweep: splice ALL non-overlapping occurrences left to right,
        // exactly matching v1's rewrite pass.
        const int newId = static_cast<int>(bestS) - 1;
        const int wantA = ids[static_cast<std::size_t>(bestPos)];
        const int wantB = ids[static_cast<std::size_t>(nxt[static_cast<std::size_t>(bestPos)])];
        i = head;
        while (i >= 0) {
            const int j = nxt[static_cast<std::size_t>(i)];
            if (j >= 0 && ids[static_cast<std::size_t>(i)] == wantA &&
                ids[static_cast<std::size_t>(j)] == wantB) {
                ids[static_cast<std::size_t>(i)] = newId;
                const int k = nxt[static_cast<std::size_t>(j)];
                nxt[static_cast<std::size_t>(i)] = k;
                if (k >= 0) {
                    prv[static_cast<std::size_t>(k)] = i;
                }
                i = k;  // skip the consumed node (non-overlapping)
            } else {
                i = j;
            }
        }
    }
    for (int k = head; k >= 0; k = nxt[static_cast<std::size_t>(k)]) {
        out.push_back(static_cast<uint16_t>(ids[static_cast<std::size_t>(k)]));
    }
}

void BpeEngineV2::encodeNormal(std::string_view text, std::vector<uint16_t>& out,
                               Scratch& scratch,
                               const domlm::tokenizer::PreTokenizerPipeline& pipeline) const {
    const domlm::tokenizer::SplitResult result = pipeline.split(text);
    for (const std::string_view chunk : result.chunks) {
        if (!chunk.empty()) {
            encodeChunk(chunk, out, scratch);
        }
    }
}

void BpeEngineV2::encodeInto(std::string_view text, std::vector<uint16_t>& out,
                             bool allowSpecials) const {
    Scratch scratch;
    for (const Piece& pc : splitSpecials(text)) {
        if (pc.isSpecial) {
            if (!allowSpecials) {
                throw std::runtime_error(std::string("BpeEngineV2: special token '") +
                                         std::string(pc.text) +
                                         "' not allowed (pass allowSpecials=true)");
            }
            out.push_back(static_cast<uint16_t>(vocabSize_ + pc.specialIdx));
        } else {
            // pipeline_ is compiled once per engine and read-only during
            // matching, so shared const use across threads is safe.
            encodeNormal(pc.text, out, scratch, pipeline_);
        }
    }
}

std::vector<uint16_t> BpeEngineV2::encode(std::string_view text, bool allowSpecials) const {
    std::vector<uint16_t> out;
    encodeInto(text, out, allowSpecials);
    return out;
}

std::string BpeEngineV2::decode(std::span<const uint16_t> ids) const {
    std::size_t total = 0;
    for (uint16_t id : ids) {
        if (static_cast<int>(id) >= vocabSize_ + kSpecialCount) {
            throw std::out_of_range("BpeEngineV2: token id out of range");
        }
        total += (id < vocabSize_) ? bytes_[id].size()
                                   : std::char_traits<char>::length(
                                         kSpecialTokens[id - vocabSize_]);
    }
    std::string out;
    out.reserve(total);
    for (uint16_t id : ids) {
        if (id < vocabSize_) {
            const std::string_view b = bytes_[id];
            out.append(b.data(), b.size());
        } else {
            out += kSpecialTokens[id - vocabSize_];
        }
    }
    return out;
}

void BpeEngineV2::encodeOneLine(std::string_view line, std::vector<uint16_t>& buf,
                               Scratch& scratch,
                               const domlm::tokenizer::PreTokenizerPipeline& pipeline,
                               bool allowSpecials) const {
    buf.clear();
    for (const Piece& pc : splitSpecials(line)) {
        if (pc.isSpecial) {
            if (!allowSpecials) {
                throw std::runtime_error("BpeEngineV2: special token not allowed");
            }
            buf.push_back(static_cast<uint16_t>(vocabSize_ + pc.specialIdx));
        } else {
            encodeNormal(pc.text, buf, scratch, pipeline);
        }
    }
}

std::vector<std::vector<uint16_t>> BpeEngineV2::encodeLines(
    const std::vector<std::string>& lines, std::size_t threads, bool allowSpecials) {
    std::vector<std::vector<uint16_t>> slots(lines.size());
    std::vector<double> ignored;
    if (threads <= 1) {
        domlm::tokenizer::PreTokenizerPipeline pipeline;
        pipeline.loadPreTokenizerRuleset(ruleset_);
        Scratch scratch;
        std::vector<uint16_t> buf;
        for (std::size_t i = 0; i < lines.size(); ++i) {
            encodeOneLine(lines[i], buf, scratch, pipeline, allowSpecials);
            slots[i] = buf;
        }
        return slots;
    }
    encodeLines(lines, slots, ignored, threads, allowSpecials);
    return slots;
}

void BpeEngineV2::encodeLines(const std::vector<std::string>& lines,
                              std::vector<std::vector<uint16_t>>& out,
                              std::vector<double>& lineUs, std::size_t threads,
                              bool allowSpecials) {
    out.assign(lines.size(), {});
    lineUs.assign(lines.size(), 0.0);
    if (lines.empty()) {
        return;
    }
    if (threads <= 1) {
        domlm::tokenizer::PreTokenizerPipeline pipeline;
        pipeline.loadPreTokenizerRuleset(ruleset_);
        Scratch scratch;
        std::vector<uint16_t> buf;
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const auto s = std::chrono::high_resolution_clock::now();
            encodeOneLine(lines[i], buf, scratch, pipeline, allowSpecials);
            out[i] = buf;
            lineUs[i] =
                std::chrono::duration<double, std::micro>(
                    std::chrono::high_resolution_clock::now() - s)
                    .count();
        }
        return;
    }
    // Multi-threaded: one pipeline + scratch per worker, lock-free dispatch.
    // Worker exceptions are captured and rethrown by the caller (an
    // exception escaping a pool worker would terminate the process).
    const std::size_t nThreads = threads == 0 ? 1 : threads;
    JThreadPool pool(nThreads);
    std::exception_ptr failure;
    std::mutex failureMutex;
    pool.parallelFor(lines.size(), [&](std::size_t idx) {
        try {
            thread_local domlm::tokenizer::PreTokenizerPipeline pipeline;
            thread_local bool pipeInit = false;
            thread_local Scratch scratch;
            thread_local std::string ruleCache;
            thread_local std::vector<uint16_t> buf;
            if (!pipeInit || ruleCache != ruleset_) {
                pipeline.loadPreTokenizerRuleset(ruleset_);
                ruleCache = ruleset_;
                pipeInit = true;
            }
            const auto s = std::chrono::high_resolution_clock::now();
            encodeOneLine(lines[idx], buf, scratch, pipeline, allowSpecials);
            out[idx] = buf;
            lineUs[idx] =
                std::chrono::duration<double, std::micro>(
                    std::chrono::high_resolution_clock::now() - s)
                    .count();
        } catch (...) {
            std::lock_guard<std::mutex> lk(failureMutex);
            if (!failure) {
                failure = std::current_exception();
            }
        }
    });
    if (failure) {
        std::rethrow_exception(failure);
    }
}

std::vector<std::vector<uint16_t>> BpeEngineV2::encodeFileLines(const std::string& path,
                                                               std::size_t threads,
                                                               bool allowSpecials) {
    MmapFile file;
    file.open(path);
    const uint8_t* base = file.data();
    const std::size_t size = file.size();
    const uint8_t* end = base + size;

    // Split into newline-aligned blocks (AVX2 scan); each block starts at a
    // line start so lines (hence pre-token chunks) never straddle workers.
    const std::size_t nThreads = threads == 0 ? 1 : threads;
    std::vector<std::pair<const uint8_t*, const uint8_t*>> blocks;
    if (nThreads <= 1 || size < 65536) {
        blocks.emplace_back(base, end);
    } else {
        const uint8_t* cur = base;
        for (std::size_t b = 1; b < nThreads && cur < end; ++b) {
            const uint8_t* probe = base + (size * b) / nThreads;
            if (probe <= cur) {
                probe = cur + 1;
            }
            const uint8_t* nl = FindNextNewline(std::min(probe, end), end);
            const uint8_t* bound = (nl < end) ? nl + 1 : end;
            if (bound <= cur) {
                break;  // remaining tail too small; fold into last block
            }
            blocks.emplace_back(cur, bound);
            cur = bound;
        }
        if (cur < end) {
            blocks.emplace_back(cur, end);
        }
    }

    // Line views straight into the mapping (zero-copy file read; the
    // mapping outlives this call). A single trailing \n / \r\n terminator
    // is stripped to mirror line-based callers (getline semantics).
    using LineView = std::string_view;
    std::vector<std::vector<LineView>> blockLines(blocks.size());
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        const uint8_t* p = blocks[b].first;
        const uint8_t* e = blocks[b].second;
        while (p < e) {
            const uint8_t* nl = FindNextNewline(p, e);
            const uint8_t* lineEnd = nl;
            if (lineEnd > p && *(lineEnd - 1) == '\r') {
                --lineEnd;  // strip \r of a \r\n terminator (getline parity)
            }
            blockLines[b].emplace_back(reinterpret_cast<const char*>(p),
                                       static_cast<std::size_t>(lineEnd - p));
            p = (nl < e) ? nl + 1 : e;
        }
    }
    std::vector<std::vector<std::vector<uint16_t>>> blockOut(blocks.size());
    if (blocks.size() <= 1) {
        // Single block: reuse encodeLines directly on materialized lines.
        std::vector<std::string> only;
        only.reserve(blockLines[0].size());
        for (const auto& v : blockLines[0]) {
            only.emplace_back(v.data(), v.size());
        }
        blockOut[0] = encodeLines(only, 1, allowSpecials);
    } else {
        JThreadPool pool(blocks.size());
        std::exception_ptr failure;
        std::mutex failureMutex;
        pool.parallelFor(blocks.size(), [&](std::size_t b) {
            try {
                // One pipeline + scratch per block (nesting the pool here
                // would oversubscribe; blocks are independent).
                BpeEngineV2* self = this;
                domlm::tokenizer::PreTokenizerPipeline pipeline;
                pipeline.loadPreTokenizerRuleset(ruleset_);
                Scratch scratch;
                std::vector<std::vector<uint16_t>> local;
                local.reserve(blockLines[b].size());
                for (const LineView& ln : blockLines[b]) {
                    std::vector<uint16_t> buf;
                    self->encodeOneLine(ln, buf, scratch, pipeline, allowSpecials);
                    local.push_back(std::move(buf));
                }
                blockOut[b] = std::move(local);
            } catch (...) {
                std::lock_guard<std::mutex> lk(failureMutex);
                if (!failure) {
                    failure = std::current_exception();
                }
            }
        });
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
    std::vector<std::vector<uint16_t>> all;
    for (auto& bo : blockOut) {
        for (auto& v : bo) {
            all.push_back(std::move(v));
        }
    }
    return all;
}

}  // namespace domlm::engine

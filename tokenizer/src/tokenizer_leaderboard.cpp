// Dom v2.0 gate: tokenizer leaderboard / benchmarking harness.
//
// Profiles registered tokenizer engine variants against tiered corpora
// (small ~1KB, medium ~1MB, large ~10MB+) and prints an ASCII leaderboard
// plus a structured `leaderboard_results.json` for historical tracking.
//
// Variants shipped:
//   v1.0-stl-baseline  cl100k_base + encodeInto() into a reused id buffer
//   v1.0-o200k         o200k_base ruleset + encodeInto() (ruleset-cost axis)
//   v1.0-fresh-alloc   cl100k_base + encode() fresh vector per line
//                      (ablation: quantifies the buffer-reuse design win)
//
// Method per variant x tier: cold-start load timed once; round-trip
// correctness gate on the small tier; W warmup passes (L1/L2 priming);
// R timed repeats; outlier rejection drops the fastest/slowest repeat
// (R>=3) and averages the rest; per-line latencies pooled from kept
// repeats for P50/P90/P99/mean. Peak RSS sampled via getrusage().
//
// Pure C++20 STL + standard OS APIs only (getrusage, chrono).

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "tokenizer/bpe/Tokenizer.hpp"
#include "BpeEngineV2.hpp"
#include "domlm/Version.hpp"

namespace {

using domlm::tokenizer::Tokenizer;
using domlm::engine::BpeEngineV2;
using Clock = std::chrono::high_resolution_clock;

enum class EngineKind { V1Reuse, V1Fresh, V2Single, V2Multi };

struct VariantSpec {
    std::string name;
    std::string ruleset;
    bool reuseBuffer;  // v1 only: encodeInto() reuse vs encode() fresh per line
    EngineKind kind = EngineKind::V1Reuse;
};

struct TierSpec {
    std::string name;
    std::size_t targetBytes;
};

struct TierResult {
    std::string tier;
    std::size_t lines = 0;
    std::size_t inputBytes = 0;
    std::size_t totalTokens = 0;
    double encTokPerSec = 0.0;
    double encMbPerSec = 0.0;
    double decMbPerSec = 0.0;
    double latP50 = 0.0;
    double latP90 = 0.0;
    double latP99 = 0.0;
    double latMean = 0.0;
    double bytesPerToken = 0.0;
    double peakRssMb = 0.0;
};

struct VariantResult {
    std::string name;
    std::string ruleset;
    double loadMs = 0.0;
    bool roundtripOk = false;
    std::vector<TierResult> tiers;
};

struct Config {
    std::string vocabPath = "data/vocab.json";
    std::string mergesPath = "data/merges.txt";
    std::string vocabBinPath = "data/vocab.bin";
    std::string inputPath = "../PreTrain/valid.txt";
    std::string largeInputPath = "../PreTrain/train.txt";
    std::string outputPath = "leaderboard_results.json";
    int warmupPasses = 2;
    int repeats = 5;
    std::size_t threads = 0;  // 0 = hardware_concurrency for v2-mt
};

void PrintUsage(std::string_view prog) {
    std::cout << "Usage:\n  " << prog
              << " [--vocab data/vocab.json] [--merges data/merges.txt]\n"
              << "      [--vocab-bin data/vocab.bin] [--threads N (0=auto)]\n"
              << "      [--input ../PreTrain/valid.txt] [--large-input ../PreTrain/train.txt]\n"
              << "      [--output leaderboard_results.json] [--warmup 2] [--repeat 5]\n"
              << "  " << prog << " --version\n"
              << "(run from the tokenizer/ directory so relative defaults resolve)\n";
}

std::vector<std::string> ReadLines(const std::string& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open '" + path + "'");
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    if (lines.empty()) {
        throw std::runtime_error("no lines in '" + path + "'");
    }
    return lines;
}

std::size_t ByteSum(const std::vector<std::string>& lines) {
    std::size_t n = 0;
    for (const auto& l : lines) {
        n += l.size();
    }
    return n;
}

// First lines totalling >= targetBytes; if the corpus is smaller, cycle it
// (deterministic 10MB+ scaling for the large tier).
std::vector<std::string> MakeTier(const std::vector<std::string>& lines,
                                   std::size_t targetBytes) {
    std::vector<std::string> tier;
    std::size_t bytes = 0;
    while (bytes < targetBytes) {
        for (const auto& l : lines) {
            tier.push_back(l);
            bytes += l.size();
            if (bytes >= targetBytes) {
                break;
            }
        }
    }
    return tier;
}

double PeakRssMb() {
#if defined(__linux__) || defined(__APPLE__)
    struct rusage ru {};
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return 0.0;
    }
#if defined(__linux__)
    return static_cast<double>(ru.ru_maxrss) / 1024.0;  // KB -> MB
#else
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);  // B -> MB
#endif
#else
    return 0.0;
#endif
}

double Percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double pos = q / 100.0 * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = (lo + 1 < sorted.size()) ? lo + 1 : lo;
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

struct RepeatStat {
    double tokPerSec = 0.0;
    double seconds = 0.0;
    std::size_t tokens = 0;
    std::vector<double> lineUs;  // per-line encode latency, microseconds
};

RepeatStat RunEncodeRepeat(const Tokenizer& tok, const std::vector<std::string>& lines,
                           const VariantSpec& spec) {
    RepeatStat st;
    st.lineUs.reserve(lines.size());
    std::vector<int> reused;
    reused.reserve(4096);
    const auto t0 = Clock::now();
    for (const auto& l : lines) {
        const auto s = Clock::now();
        std::size_t n = 0;
        if (spec.reuseBuffer) {
            reused.clear();
            tok.encodeInto(l, reused);
            n = reused.size();
        } else {
            std::vector<int> fresh = tok.encode(l);
            n = fresh.size();
        }
        const auto e = Clock::now();
        st.lineUs.push_back(
            std::chrono::duration<double, std::micro>(e - s).count());
        st.tokens += n;
    }
    const auto t1 = Clock::now();
    st.seconds = std::chrono::duration<double>(t1 - t0).count();
    st.tokPerSec = static_cast<double>(st.tokens) / (st.seconds > 0 ? st.seconds : 1e-9);
    return st;
}

TierResult BenchmarkTier(const Tokenizer& tok, const std::vector<std::string>& lines,
                         const std::string& tierName, const VariantSpec& spec,
                         const Config& cfg) {
    TierResult r;
    r.tier = tierName;
    r.lines = lines.size();
    r.inputBytes = ByteSum(lines);

    for (int w = 0; w < cfg.warmupPasses; ++w) {
        (void)RunEncodeRepeat(tok, lines, spec);  // L1/L2 priming only
    }
    std::vector<RepeatStat> reps;
    reps.reserve(static_cast<std::size_t>(cfg.repeats));
    for (int i = 0; i < cfg.repeats; ++i) {
        reps.push_back(RunEncodeRepeat(tok, lines, spec));
    }
    // Outlier rejection: drop fastest/slowest repeat (R>=3), mean the rest.
    std::vector<std::size_t> order(reps.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return reps[a].tokPerSec < reps[b].tokPerSec; });
    std::size_t lo = 0;
    std::size_t hi = order.size();
    if (order.size() >= 3) {
        lo = 1;
        hi = order.size() - 1;
    }
    double tokSum = 0.0;
    double secSum = 0.0;
    std::vector<double> lat;
    for (std::size_t k = lo; k < hi; ++k) {
        const auto& s = reps[order[k]];
        tokSum += static_cast<double>(s.tokens);
        secSum += s.seconds;
        lat.insert(lat.end(), s.lineUs.begin(), s.lineUs.end());
    }
    std::sort(lat.begin(), lat.end());
    r.totalTokens = static_cast<std::size_t>(tokSum / static_cast<double>(hi - lo));
    r.encTokPerSec = tokSum / (secSum > 0 ? secSum : 1e-9);
    r.encMbPerSec =
        static_cast<double>(r.inputBytes) * static_cast<double>(hi - lo) / (secSum > 0 ? secSum : 1e-9) /
        (1024.0 * 1024.0);
    r.latP50 = Percentile(lat, 50.0);
    r.latP90 = Percentile(lat, 90.0);
    r.latP99 = Percentile(lat, 99.0);
    r.latMean = lat.empty()
                    ? 0.0
                    : std::accumulate(lat.begin(), lat.end(), 0.0) / static_cast<double>(lat.size());
    r.bytesPerToken = static_cast<double>(r.inputBytes) /
                      (r.totalTokens > 0 ? static_cast<double>(r.totalTokens) : 1.0);

    // Decode: re-encode once, then time raw byte generation.
    std::vector<std::vector<int>> idLines;
    idLines.reserve(lines.size());
    {
        std::vector<int> tmp;
        tmp.reserve(4096);
        for (const auto& l : lines) {
            tmp.clear();
            tok.encodeInto(l, tmp);
            idLines.push_back(tmp);
        }
    }
    std::size_t outBytes = 0;
    const auto d0 = Clock::now();
    for (const auto& ids : idLines) {
        outBytes += tok.decode(ids).size();
    }
    const auto d1 = Clock::now();
    const double dsec = std::chrono::duration<double>(d1 - d0).count();
    r.decMbPerSec = static_cast<double>(outBytes) / (dsec > 0 ? dsec : 1e-9) / (1024.0 * 1024.0);
    r.peakRssMb = PeakRssMb();
    return r;
}

RepeatStat RunEncodeRepeatV2(BpeEngineV2& eng, const std::vector<std::string>& lines,
                             std::size_t threads) {
    RepeatStat st;
    st.lineUs.reserve(lines.size());
    std::vector<std::vector<uint16_t>> slots;
    std::vector<double> lat;
    const auto t0 = Clock::now();
    if (threads <= 1) {
        // Timed per line like the v1 path (apples-to-apples latencies).
        std::vector<uint16_t> reused;
        reused.reserve(4096);
        for (const auto& l : lines) {
            const auto s = Clock::now();
            reused.clear();
            eng.encodeInto(l, reused);
            const auto e = Clock::now();
            st.lineUs.push_back(
                std::chrono::duration<double, std::micro>(e - s).count());
            st.tokens += reused.size();
        }
    } else {
        // Parallel batch; per-line latencies measured inside the workers.
        eng.encodeLines(lines, slots, lat, threads);
        st.lineUs = std::move(lat);
        for (const auto& v : slots) {
            st.tokens += v.size();
        }
    }
    const auto t1 = Clock::now();
    st.seconds = std::chrono::duration<double>(t1 - t0).count();
    st.tokPerSec = static_cast<double>(st.tokens) / (st.seconds > 0 ? st.seconds : 1e-9);
    return st;
}

TierResult BenchmarkTierV2(BpeEngineV2& eng, const std::vector<std::string>& lines,
                           const std::string& tierName, std::size_t threads,
                           const Config& cfg) {
    TierResult r;
    r.tier = tierName;
    r.lines = lines.size();
    r.inputBytes = ByteSum(lines);

    // Warmup also primes the engine's thread-local pipelines/scratch.
    std::vector<std::vector<uint16_t>> slots;
    std::vector<double> lat;
    for (int w = 0; w < cfg.warmupPasses; ++w) {
        if (threads <= 1) {
            (void)RunEncodeRepeatV2(eng, lines, 1);
        } else {
            eng.encodeLines(lines, slots, lat, threads);
        }
    }
    std::vector<RepeatStat> reps;
    reps.reserve(static_cast<std::size_t>(cfg.repeats));
    for (int i = 0; i < cfg.repeats; ++i) {
        reps.push_back(RunEncodeRepeatV2(eng, lines, threads));
    }
    // Same outlier rejection as the v1 path.
    std::vector<std::size_t> order(reps.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return reps[a].tokPerSec < reps[b].tokPerSec; });
    std::size_t lo = 0;
    std::size_t hi = order.size();
    if (order.size() >= 3) {
        lo = 1;
        hi = order.size() - 1;
    }
    double tokSum = 0.0;
    double secSum = 0.0;
    std::vector<double> latAll;
    for (std::size_t k = lo; k < hi; ++k) {
        const auto& s = reps[order[k]];
        tokSum += static_cast<double>(s.tokens);
        secSum += s.seconds;
        latAll.insert(latAll.end(), s.lineUs.begin(), s.lineUs.end());
    }
    std::sort(latAll.begin(), latAll.end());
    r.totalTokens = static_cast<std::size_t>(tokSum / static_cast<double>(hi - lo));
    r.encTokPerSec = tokSum / (secSum > 0 ? secSum : 1e-9);
    r.encMbPerSec =
        static_cast<double>(r.inputBytes) * static_cast<double>(hi - lo) / (secSum > 0 ? secSum : 1e-9) /
        (1024.0 * 1024.0);
    r.latP50 = Percentile(latAll, 50.0);
    r.latP90 = Percentile(latAll, 90.0);
    r.latP99 = Percentile(latAll, 99.0);
    r.latMean = latAll.empty() ? 0.0
                               : std::accumulate(latAll.begin(), latAll.end(), 0.0) /
                                     static_cast<double>(latAll.size());
    r.bytesPerToken = static_cast<double>(r.inputBytes) /
                      (r.totalTokens > 0 ? static_cast<double>(r.totalTokens) : 1.0);

    eng.encodeLines(lines, slots, lat, threads <= 1 ? 1 : threads);
    std::size_t outBytes = 0;
    const auto d0 = Clock::now();
    for (const auto& ids : slots) {
        outBytes += eng.decode(ids).size();
    }
    const auto d1 = Clock::now();
    const double dsec = std::chrono::duration<double>(d1 - d0).count();
    r.decMbPerSec = static_cast<double>(outBytes) / (dsec > 0 ? dsec : 1e-9) / (1024.0 * 1024.0);
    r.peakRssMb = PeakRssMb();
    return r;
}

// v2 engine construction: zero-copy vocab.bin mmap when present, otherwise
// build the same matrix from the already-loaded v1 vocabulary.
BpeEngineV2 MakeV2Engine(const Config& cfg, const std::string& ruleset) {
    std::ifstream probe(cfg.vocabBinPath, std::ios::in | std::ios::binary);
    if (probe.is_open()) {
        probe.close();
        return BpeEngineV2::FromBinFile(cfg.vocabBinPath, ruleset);
    }
    Tokenizer tok = Tokenizer::load(cfg.vocabPath, cfg.mergesPath, ruleset);
    return BpeEngineV2::FromVocabulary(tok.vocabulary(), ruleset);
}

std::size_t ResolveThreads(const Config& cfg) {
    if (cfg.threads != 0) {
        return cfg.threads;
    }
    const unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : n;
}

VariantResult BenchmarkVariant(const VariantSpec& spec, const Config& cfg,
                               const std::vector<std::string>& baseLines,
                               const std::vector<std::string>& largeLines) {
    VariantResult vr;
    vr.name = spec.name;
    vr.ruleset = spec.ruleset;
    const std::vector<std::string> small = MakeTier(baseLines, 1024);
    const std::vector<std::pair<std::string, std::vector<std::string>>> tiers = {
        {"small-1KB", small},
        {"medium-1MB", MakeTier(baseLines, 1024 * 1024)},
        {"large-10MB", MakeTier(largeLines, 10 * 1024 * 1024)},
    };

    const bool isV2 = spec.kind == EngineKind::V2Single || spec.kind == EngineKind::V2Multi;
    const std::size_t threads =
        spec.kind == EngineKind::V2Multi ? ResolveThreads(cfg) : 1;
    const auto l0 = Clock::now();
    if (!isV2) {
        Tokenizer tok = Tokenizer::load(cfg.vocabPath, cfg.mergesPath, spec.ruleset);
        vr.loadMs =
            std::chrono::duration<double, std::milli>(Clock::now() - l0).count();
        // Correctness gate: exact round-trip on the small tier proves the
        // timed numbers below are apples-to-apples across variants.
        vr.roundtripOk = true;
        for (std::size_t i = 0; i < small.size() && i < 50; ++i) {
            if (tok.decode(tok.encode(small[i])) != small[i]) {
                vr.roundtripOk = false;
                break;
            }
        }
        for (const auto& [name, lines] : tiers) {
            vr.tiers.push_back(BenchmarkTier(tok, lines, name, spec, cfg));
        }
        return vr;
    }

    BpeEngineV2 eng = MakeV2Engine(cfg, spec.ruleset);
    vr.loadMs =
        std::chrono::duration<double, std::milli>(Clock::now() - l0).count();
    vr.roundtripOk = true;
    for (std::size_t i = 0; i < small.size() && i < 50; ++i) {
        if (eng.decode(eng.encode(small[i])) != small[i]) {
            vr.roundtripOk = false;
            break;
        }
    }
    for (const auto& [name, lines] : tiers) {
        vr.tiers.push_back(BenchmarkTierV2(eng, lines, name, threads, cfg));
    }
    return vr;
}

std::string JsonEscape(std::string_view s) {
    std::string o;
    for (char c : s) {
        if (c == '"') {
            o += "\\\"";
        } else if (c == '\\') {
            o += "\\\\";
        } else {
            o += c;
        }
    }
    return o;
}

void WriteJson(const std::string& path, const std::vector<VariantResult>& results,
               const Config& cfg) {
    std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open '" + path + "'");
    }
    const std::time_t now = std::time(nullptr);
    char stamp[32] = {};
    (void)std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    out << "{\"generated_utc\":\"" << stamp << "\",\"input\":\"" << JsonEscape(cfg.inputPath)
        << "\",\"variants\":[";
    for (std::size_t v = 0; v < results.size(); ++v) {
        const auto& vr = results[v];
        if (v != 0) {
            out << ',';
        }
        out << "{\"name\":\"" << JsonEscape(vr.name) << "\",\"ruleset\":\""
            << JsonEscape(vr.ruleset) << "\",\"load_ms\":" << std::fixed << std::setprecision(3)
            << vr.loadMs << ",\"roundtrip_ok\":" << (vr.roundtripOk ? "true" : "false")
            << ",\"tiers\":[";
        for (std::size_t t = 0; t < vr.tiers.size(); ++t) {
            const auto& tr = vr.tiers[t];
            if (t != 0) {
                out << ',';
            }
            out << "{\"tier\":\"" << JsonEscape(tr.tier) << "\",\"lines\":" << tr.lines
                << ",\"input_bytes\":" << tr.inputBytes << ",\"tokens\":" << tr.totalTokens
                << ",\"enc_tok_per_s\":" << std::setprecision(1) << tr.encTokPerSec
                << ",\"enc_mb_per_s\":" << std::setprecision(3) << tr.encMbPerSec
                << ",\"dec_mb_per_s\":" << tr.decMbPerSec << ",\"lat_p50_us\":"
                << tr.latP50 << ",\"lat_p90_us\":" << tr.latP90 << ",\"lat_p99_us\":" << tr.latP99
                << ",\"lat_mean_us\":" << tr.latMean << ",\"bytes_per_token\":"
                << tr.bytesPerToken << ",\"peak_rss_mb\":" << tr.peakRssMb << "}";
        }
        out << "]}";
    }
    out << "]}";
    out.flush();
    if (!out) {
        throw std::runtime_error("write failed for '" + path + "'");
    }
}

void PrintTable(const std::vector<VariantResult>& results) {
    std::cout << "\n==================== Dom Tokenizer Leaderboard ====================\n";
    std::cout << std::left << std::setw(20) << "variant" << std::setw(11) << "tier"
              << std::right << std::setw(11) << "tok/s" << std::setw(9) << "MB/s"
              << std::setw(11) << "decMB/s" << std::setw(9) << "p50us" << std::setw(9) << "p90us"
              << std::setw(9) << "p99us" << std::setw(9) << "meanUs" << std::setw(9) << "loadMs"
              << std::setw(9) << "rssMB" << std::setw(8) << "B/tok" << "\n";
    std::cout << std::string(128, '-') << "\n";
    std::cout << std::fixed;
    for (const auto& vr : results) {
        for (const auto& tr : vr.tiers) {
            std::cout << std::left << std::setw(20) << vr.name << std::setw(11) << tr.tier
                      << std::right << std::setprecision(0) << std::setw(11) << tr.encTokPerSec
                      << std::setprecision(2) << std::setw(9) << tr.encMbPerSec << std::setw(11)
                      << tr.decMbPerSec << std::setprecision(1) << std::setw(9) << tr.latP50
                      << std::setw(9) << tr.latP90 << std::setw(9) << tr.latP99 << std::setw(9)
                      << tr.latMean << std::setprecision(2) << std::setw(9) << vr.loadMs
                      << std::setw(9) << tr.peakRssMb << std::setprecision(2) << std::setw(8)
                      << tr.bytesPerToken << (vr.roundtripOk ? "" : "  ROUNDTRIP-FAIL") << "\n";
        }
    }
    std::cout << std::string(128, '-') << "\n";
    for (const char* tier : {"small-1KB", "medium-1MB", "large-10MB"}) {
        const VariantResult* best = nullptr;
        double bestTps = 0.0;
        for (const auto& vr : results) {
            for (const auto& tr : vr.tiers) {
                if (tr.tier == tier && tr.encTokPerSec > bestTps) {
                    bestTps = tr.encTokPerSec;
                    best = &vr;
                }
            }
        }
        if (best != nullptr) {
            std::cout << "fastest @" << tier << ": " << best->name << "\n";
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        auto needValue = [&](std::string_view flag) -> std::string_view {
            if (i + 1 >= argc) {
                std::cerr << "tokenizer_leaderboard: flag " << flag << " needs a value\n";
                PrintUsage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            PrintUsage(argv[0]);
            return 0;
        }
        if (a == "-v" || a == "--version") {
            std::cout << "DomLM v" << DOMLM_VERSION_STRING << "\n";
            return 0;
        }
        if (a == "--vocab") {
            cfg.vocabPath.assign(needValue(a));
        } else if (a == "--merges") {
            cfg.mergesPath.assign(needValue(a));
        } else if (a == "--vocab-bin") {
            cfg.vocabBinPath.assign(needValue(a));
        } else if (a == "--threads") {
            cfg.threads = static_cast<std::size_t>(
                std::stoul(std::string(needValue(a))));
        } else if (a == "--input") {
            cfg.inputPath.assign(needValue(a));
        } else if (a == "--large-input") {
            cfg.largeInputPath.assign(needValue(a));
        } else if (a == "--output") {
            cfg.outputPath.assign(needValue(a));
        } else if (a == "--warmup") {
            cfg.warmupPasses = std::stoi(std::string(needValue(a)));
        } else if (a == "--repeat") {
            cfg.repeats = std::stoi(std::string(needValue(a)));
        } else {
            std::cerr << "tokenizer_leaderboard: unknown flag '" << a << "'\n";
            PrintUsage(argv[0]);
            return 2;
        }
    }
    if (cfg.warmupPasses < 0 || cfg.repeats < 1) {
        std::cerr << "tokenizer_leaderboard: need --warmup >= 0 and --repeat >= 1\n";
        return 2;
    }

    try {
        const std::vector<std::string> baseLines = ReadLines(cfg.inputPath);
        std::vector<std::string> largeLines;
        try {
            largeLines = ReadLines(cfg.largeInputPath);
        } catch (const std::exception&) {
            largeLines = baseLines;  // MakeTier() cycles it to 10MB+
        }
        const std::vector<VariantSpec> variants = {
            {"v1.0-stl-baseline", "cl100k_base", true, EngineKind::V1Reuse},
            {"v1.0-o200k", "o200k_base", true, EngineKind::V1Reuse},
            {"v1.0-fresh-alloc", "cl100k_base", false, EngineKind::V1Fresh},
            {"v2.0-flatmat", "cl100k_base", true, EngineKind::V2Single},
            {"v2.0-flatmat-mt", "cl100k_base", true, EngineKind::V2Multi},
        };
        std::vector<VariantResult> results;
        for (const auto& v : variants) {
            std::cout << "profiling " << v.name << " (" << v.ruleset << ")..."
                      << std::flush;
            results.push_back(BenchmarkVariant(v, cfg, baseLines, largeLines));
            std::cout << " done\n";
        }
        PrintTable(results);
        WriteJson(cfg.outputPath, results, cfg);
        std::cout << "wrote " << cfg.outputPath << "\n";
    } catch (const std::exception& e) {
        std::cerr << "tokenizer_leaderboard: error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

cmake_minimum_required(VERSION 3.20)

# DomLM single-source-of-truth version: <repo-root>/VERSION (one line,
# e.g. "2.0.0"). Bump that file, then re-run configuration.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" DOMLM_VERSION_STRING)
string(STRIP "${DOMLM_VERSION_STRING}" DOMLM_VERSION_STRING)
if(NOT DOMLM_VERSION_STRING MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "DomLM: malformed VERSION file contents: '${DOMLM_VERSION_STRING}'")
endif()
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" _ "${DOMLM_VERSION_STRING}")
set(DOMLM_VERSION_MAJOR ${CMAKE_MATCH_1})
set(DOMLM_VERSION_MINOR ${CMAKE_MATCH_2})
set(DOMLM_VERSION_PATCH ${CMAKE_MATCH_3})

project(DomLM VERSION ${DOMLM_VERSION_STRING} LANGUAGES CXX)
message(STATUS "DomLM v${DOMLM_VERSION_STRING}")

# Generate include/domlm/Version.hpp from the template above.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/domlm/Version.hpp.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/domlm/Version.hpp"
  @ONLY
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# Find dependencies
find_package(ICU REQUIRED COMPONENTS uc)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)

# Include directories
include_directories(include)

# Library
add_library(tokenizer_pre
    src/UnicodeNormalizer.cpp
    src/PreTokenizer.cpp
    src/PreTokenizerPipeline.cpp
)

target_include_directories(tokenizer_pre
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(tokenizer_pre
    PUBLIC
        ICU::uc
        ${PCRE2_LIBRARIES}
)

target_include_directories(tokenizer_pre PRIVATE ${PCRE2_INCLUDE_DIRS})
target_compile_options(tokenizer_pre PRIVATE ${PCRE2_CFLAGS_OTHER})

# Smoke-test CLI
add_executable(domlm_tokenize src/main.cpp)
target_link_libraries(domlm_tokenize PRIVATE tokenizer_pre)

# Phase 2: BPE training & vocabulary engine
# Phase 3: inference-time tokenizer (encode/decode) + serialization
add_library(tokenizer_bpe
    src/Vocabulary.cpp
    src/CorpusReader.cpp
    src/BpeTrainer.cpp
    src/Serialization.cpp
    src/Tokenizer.cpp
    src/BpeEngineV2.cpp
)

target_include_directories(tokenizer_bpe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(tokenizer_bpe
    PUBLIC
        tokenizer_pre
)

# BPE training CLI
add_executable(domlm_train_bpe src/train_main.cpp)
target_link_libraries(domlm_train_bpe PRIVATE tokenizer_bpe)

# BPE throughput benchmark
add_executable(domlm_bench src/bench_main.cpp)
target_link_libraries(domlm_bench PRIVATE tokenizer_bpe)

# v2.0 gate: leaderboard / benchmarking harness (Ninja + Release)
add_executable(domlm_leaderboard src/tokenizer_leaderboard.cpp)
target_link_libraries(domlm_leaderboard PRIVATE tokenizer_bpe)

# Tests
enable_testing()
add_subdirectory(tests)

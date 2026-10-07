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
set(TOKENIZER_PRE_SOURCES
    src/UnicodeNormalizer.cpp
    src/PreTokenizer.cpp
    src/PreTokenizerPipeline.cpp
)
add_library(tokenizer_pre ${TOKENIZER_PRE_SOURCES})

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
# PIC: the static libs also link into the domlm_tokenizer shared module
# (whose thread_local TLS model requires position-independent objects).
set_target_properties(tokenizer_pre PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Smoke-test CLI
add_executable(domlm_tokenize src/main.cpp)
target_link_libraries(domlm_tokenize PRIVATE tokenizer_pre)

# Phase 2: BPE training & vocabulary engine
# Phase 3: inference-time tokenizer (encode/decode) + serialization
set(TOKENIZER_BPE_SOURCES
    src/Vocabulary.cpp
    src/CorpusReader.cpp
    src/BpeTrainer.cpp
    src/Serialization.cpp
    src/Tokenizer.cpp
    src/BpeEngineV2.cpp
)
add_library(tokenizer_bpe ${TOKENIZER_BPE_SOURCES})

target_include_directories(tokenizer_bpe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(tokenizer_bpe
    PUBLIC
        tokenizer_pre
)
set_target_properties(tokenizer_bpe PROPERTIES POSITION_INDEPENDENT_CODE ON)

# BPE training CLI
add_executable(domlm_train_bpe src/train_main.cpp)
target_link_libraries(domlm_train_bpe PRIVATE tokenizer_bpe)

# BPE throughput benchmark
add_executable(domlm_bench src/bench_main.cpp)
target_link_libraries(domlm_bench PRIVATE tokenizer_bpe)

# v2.0 gate: leaderboard / benchmarking harness (Ninja + Release)
add_executable(domlm_leaderboard src/tokenizer_leaderboard.cpp)
target_link_libraries(domlm_leaderboard PRIVATE tokenizer_bpe)

# v3.0: Python C-extension (pybind11, in-process BpeEngineV2). Guarded so a
# missing interpreter/pybind11 only skips this target, never the core build.
option(DOMLM_BUILD_PYTHON "Build domlm_tokenizer Python C-extension" ON)
if(DOMLM_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module QUIET)
  if(Python_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
  find_package(pybind11 CONFIG QUIET)
  if(pybind11_FOUND AND Python_FOUND)
    # Sanitizer-free OBJECT copies of the core sources: the extension .so
    # loads into an uninstrumented CPython process, so none of its code may
    # reference sanitizer runtimes (plain `import` must work, zero
    # LD_PRELOAD). The sanitized static libs above remain what ctest covers.
    add_library(tokenizer_pre_nosan OBJECT ${TOKENIZER_PRE_SOURCES})
    target_include_directories(tokenizer_pre_nosan
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_include_directories(tokenizer_pre_nosan PRIVATE ${PCRE2_INCLUDE_DIRS})
    target_compile_options(tokenizer_pre_nosan PRIVATE ${PCRE2_CFLAGS_OTHER} -fno-sanitize=all)
    target_link_libraries(tokenizer_pre_nosan PUBLIC ICU::uc)
    set_target_properties(tokenizer_pre_nosan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(tokenizer_bpe_nosan OBJECT ${TOKENIZER_BPE_SOURCES})
    target_include_directories(tokenizer_bpe_nosan
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    target_include_directories(tokenizer_bpe_nosan PRIVATE ${PCRE2_INCLUDE_DIRS})
    target_compile_options(tokenizer_bpe_nosan PRIVATE ${PCRE2_CFLAGS_OTHER} -fno-sanitize=all)
    target_link_libraries(tokenizer_bpe_nosan PUBLIC ICU::uc tokenizer_pre_nosan)
    set_target_properties(tokenizer_bpe_nosan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    pybind11_add_module(domlm_tokenizer
        python/domlm_tokenizer.cpp
        $<TARGET_OBJECTS:tokenizer_pre_nosan>
        $<TARGET_OBJECTS:tokenizer_bpe_nosan>)
    target_link_libraries(domlm_tokenizer PRIVATE ICU::uc ${PCRE2_LIBRARIES})
    target_compile_options(domlm_tokenizer PRIVATE -fno-sanitize=all)
    target_link_options(domlm_tokenizer PRIVATE -fno-sanitize=all)
    message(STATUS "DomLM: building Python extension domlm_tokenizer")
  else()
    message(STATUS "DomLM: skipping Python extension (need CPython dev + pybind11)")
  endif()
endif()

# Tests
enable_testing()
add_subdirectory(tests)

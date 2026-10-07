cmake_minimum_required(VERSION 3.20)
project(tokenizer_pretokenizer LANGUAGES CXX)

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
add_executable(tokenize src/main.cpp)
target_link_libraries(tokenize PRIVATE tokenizer_pre)

# Phase 2: BPE training & vocabulary engine
add_library(tokenizer_bpe
    src/Vocabulary.cpp
    src/CorpusReader.cpp
    src/BpeTrainer.cpp
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
add_executable(train_bpe src/train_main.cpp)
target_link_libraries(train_bpe PRIVATE tokenizer_bpe)

# Tests
enable_testing()
add_subdirectory(tests)

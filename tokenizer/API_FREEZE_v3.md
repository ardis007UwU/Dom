# DomLM Tokenizer — API FREEZE Declaration (v3.0)

Status: **FINAL / PRODUCTION-READY** as of DomLM v3.0.0.

The following C++ tokenizer APIs are frozen. No breaking changes
(signatures, namespaces, file formats, CLI flags) without a major version
bump and a new freeze review:

- Namespaces: `domlm::tokenizer` (pre-tokenizer, BPE vocab/trainer,
  inference `Tokenizer`, serialization), `domlm::engine` (`BpeEngineV2`,
  `MergeMatrix`, `SpecialTrie`, `MmapFile`, `JThreadPool`).
- Artifacts: `vocab.json` (v1 JSON) + `merges.txt` (`a b -> c` lines) +
  `vocab.bin` v2 layout (`DOMBPE02`, offsets + token blob + mmapped rank
  matrix). v1 `vocab.bin` files remain loadable (rebuild path).
- Binaries: `domlm_tokenize`, `domlm_train_bpe`, `domlm_bench`,
  `domlm_leaderboard` — all answer `--version` / `-v` with `DomLM vX.Y.Z`.
- Python: `import domlm_tokenizer` (`Tokenizer`, `encode`, `encode_batch`,
  `decode`, `from_json`, `__version__`).

Compatibility contract (enforced by `ctest`):
`test_pretokenizer`, `test_bpe_trainer`, `test_bpe_tokenizer`, `test_bpe_v2`
(byte-exact v2-vs-v1 agreement incl. multi-threaded and mmap file paths).

All tokenizer R&D effort now pivots to GPU model pre-training.

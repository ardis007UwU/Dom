#!/usr/bin/env bash
# DomLM v3.0 release verification — single command:
#   bash scripts/v3_verify.sh
# Runs from anywhere. Stages:
#   1. CMake configure + full build (asan-ubsan preset, incl. domlm_tokenizer)
#   2. ctest (all 4 C++ suites, ASan+UBSan+leaks)
#   3. Binary --version pins vs root VERSION
#   4. Python extension: import, roundtrip, batch/single + thread determinism,
#      vocab.bin vs from_json parity (zero-copy binary compatibility)
#   5. PyTorch smoke: tiny CPU/GPU train_v2.py run (SKIP if torch missing)
# Exit status: 0 only if every non-skipped stage passes.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/tokenizer/build/asan-ubsan"
EXPECT_VERSION="$(tr -d ' \t\r\n' < "$ROOT/VERSION")"
FAIL=0
SKIPPED=""

stage() { echo "=== [$1] $2 ==="; }
pass()  { echo "PASS: $1"; }
fail()  { echo "FAIL: $1"; FAIL=1; }
skip()  { echo "SKIP: $1"; SKIPPED="$SKIPPED $1"; }

stage 1 "configure + build"
if cmake --preset asan-ubsan -S "$ROOT/tokenizer" -B "$BUILD" >/tmp/v3_cmake.log 2>&1 \
   && cmake --build "$BUILD" >/tmp/v3_build.log 2>&1; then
  pass "configure + build"
else
  fail "configure/build (see /tmp/v3_cmake.log /tmp/v3_build.log)"; tail -5 /tmp/v3_build.log
fi

stage 2 "ctest (ASan+UBSan)"
if [ "$FAIL" -eq 0 ] && ctest --test-dir "$BUILD" --output-on-failure 2>&1 | tee /tmp/v3_ctest.log | grep -q "100% tests passed"; then
  pass "ctest 4/4 suites"
else
  fail "ctest"; grep -E "Failed|Timeout" /tmp/v3_ctest.log | head -5
fi

stage 3 "binary --version pins"
if [ "$FAIL" -eq 0 ]; then
  ok=1
  for bin in domlm_tokenize domlm_train_bpe domlm_bench domlm_leaderboard; do
    got=$("$BUILD/$bin" --version 2>&1)
    [ "$got" = "DomLM v$EXPECT_VERSION" ] || { echo "  $bin -> '$got'"; ok=0; }
  done
  [ "$ok" -eq 1 ] && pass "all CLIs report DomLM v$EXPECT_VERSION" || fail "version pin mismatch"
fi

stage 4 "Python C-extension (in-process, no subprocess/file IO per call)"
if [ "$FAIL" -eq 0 ]; then
  cd "$ROOT/tokenizer" || exit 1
  if ASAN_OPTIONS=detect_leaks=0 PYTHONPATH="$BUILD" python3 - <<'EOF' 2>&1 | tee /tmp/v3_ext.log | grep -q "EXT_ALL_OK"; then
import domlm_tokenizer as dt
assert dt.__version__ == open("../VERSION").read().strip(), dt.__version__
tok = dt.Tokenizer("data/vocab.bin")
assert tok.vocab_size == 2048
lines = [l.rstrip("\n") for l in open("../PreTrain/valid.txt", encoding="utf-8")][:200]
single = [tok.encode(s) for s in lines]
assert all(tok.decode(ids) == s.encode("utf-8") for s, ids in zip(lines, single))
batch = tok.encode_batch(lines, threads=0)
assert batch == single, "batch/single mismatch"
assert tok.encode_batch(lines, threads=1) == single, "thread determinism"
if any(len(v) == 0 for v in single):
    raise SystemExit("empty encoding produced")
t2 = dt.Tokenizer.from_json("data/vocab.json", "data/merges.txt")
assert [t2.encode(s) for s in lines] == single, "vocab.bin vs from_json mismatch"
print("EXT_ALL_OK")
EOF
    pass "extension roundtrip + determinism + bin/json parity (200 lines)"
  else
    fail "extension checks"; tail -5 /tmp/v3_ext.log
  fi
  cd "$ROOT" || exit 1
fi

stage 5 "PyTorch smoke (train_v2.py, tiny run)"
if python3 -c "import torch" 2>/dev/null; then
  if [ "$FAIL" -eq 0 ] && python3 train_v2.py --data-dir data --out-dir /tmp/v3smoke \
      --batch-size 2 --seq-len 64 --max-steps 3 --grad-accum 1 \
      --eval-every 3 --eval-batches 1 --ckpt-every 3 --log-every 1 \
      >/tmp/v3_torch.log 2>&1 \
      && [ -f /tmp/v3smoke/dom_v3_checkpoint.pt ]; then
    pass "torch smoke (3 steps + dom_v3_checkpoint.pt)"
  else
    fail "torch smoke"; tail -8 /tmp/v3_torch.log
  fi
else
  skip "torch not installed (no GPU env here)"
fi

echo "=== v3_verify: $([ "$FAIL" -eq 0 ] && echo ALL_GREEN || echo FAILURES) | skipped:$SKIPPED ==="
exit "$FAIL"

#!/usr/bin/env python3
"""DomLM dataset packer: high-throughput binary dataset generator.

Reads raw text files line by line, tokenizes each line against
``tokenizer/vocab.json`` + ``tokenizer/merges.txt`` with a byte-exact port
of the C++ pre-tokenizer (cl100k_base / o200k_base, PCRE2_UTF|PCRE2_UCP
semantics) and rank-ordered BPE merge application, then packs token ids
into a contiguous little-endian uint16 stream (``train.bin`` / ``valid.bin``).

Stdlib only (no third-party imports): ``\\p{L}``/``\\p{N}`` classes are
evaluated with :mod:`unicodedata`, possessive quantifiers need no
backtracking here (each alternative is tried in order with maximal munch,
which is observationally identical for these patterns).

Byte-exactness notes vs the C++ engine (``domlm::tokenizer``):
  * Input files must be valid UTF-8 (decoded ``strict``; anything else fails
    loudly instead of silently diverging from ICU sanitization).
  * Newlines are preserved: the file is streamed in binary and each raw line
    (terminator included) is tokenized, so ``\\n`` ids appear in the stream.
  * BPE operates on raw UTF-8 bytes; merge choice is lowest-rank-first with
    left-to-right non-overlapping rewrites, never across chunk boundaries.

Usage:
  python3 scripts/prepare_dataset.py --input PreTrain/train.txt \\
      --vocab tokenizer/vocab.json --merges tokenizer/merges.txt \\
      --output data/train.bin --jobs 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import struct
import sys
import time
import unicodedata


def domlm_version() -> str:
    """Read the single-source-of-truth <repo-root>/VERSION (locked in sync
    with the C++ core's domlm/Version.hpp)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "VERSION"), "r", encoding="ascii") as f:
        ver = f.read().strip()
    if ver.count(".") != 2 or not all(p.isdigit() for p in ver.split(".")):
        raise ValueError(f"malformed VERSION file: {ver!r}")
    return ver

# ---------------------------------------------------------------------------
# Unicode property helpers (PCRE2_UTF | PCRE2_UCP semantics)
# ---------------------------------------------------------------------------


def _cat(ch: str) -> str:
    return unicodedata.category(ch)


def is_letter(ch: str) -> bool:
    """\\p{L}: Lu Ll Lt Lm Lo."""
    return _cat(ch).startswith("L")


def is_number(ch: str) -> bool:
    """\\p{N}: Nd Nl No."""
    return _cat(ch).startswith("N")


def is_space(ch: str) -> bool:
    """\\s under UCP: ASCII whitespace + NEL + Unicode spaces (Z*)."""
    if ch in " \t\n\r\x0b\x0c":
        return True
    if ch == "\x85":
        return True
    return _cat(ch).startswith("Z")


def is_nonspace(ch: str) -> bool:
    """\\S under UCP."""
    return not is_space(ch)


def is_upper_word(ch: str) -> bool:
    """o200k leading class: Lu Lt Lm Lo + marks."""
    c = _cat(ch)
    return c in ("Lu", "Lt", "Lm", "Lo") or c.startswith("M")


def is_lower_word(ch: str) -> bool:
    """o200k trailing class: Ll Lm Lo + marks."""
    c = _cat(ch)
    return c in ("Ll", "Lm", "Lo") or c.startswith("M")


# ---------------------------------------------------------------------------
# cl100k_base splitter
# Pattern (byte-exact copy of PreTokenizer::CL100K_BASE):
#   '(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+
#   | ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*+|\\s++$|\\s*[\\r\\n]|\\s+(?!\\S)|\\s
# ---------------------------------------------------------------------------

_CL100K_SINGLE = frozenset("sdmtSDMT")
_CL100K_DOUBLE = frozenset(("ll", "LL", "lL", "Ll", "ve", "VE", "vE", "Ve",
                            "re", "RE", "rE", "Re"))


def split_cl100k(text: str) -> list[str]:
    """Leftmost-first-match scan with the cl100k alternatives in order."""
    chunks: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        # Alt 1: '(?i:[sdmt]|ll|ve|re) -- single-char branch first (ordered).
        if c == "'":
            if i + 1 < n and text[i + 1] in _CL100K_SINGLE:
                chunks.append(text[i:i + 2])
                i += 2
                continue
            if i + 2 < n and text[i + 1:i + 3] in _CL100K_DOUBLE:
                chunks.append(text[i:i + 3])
                i += 3
                continue
        # Alt 2: [^\r\n\p{L}\p{N}]?+ \p{L}++ (maximal letter run).
        j = i
        if c not in "\r\n" and not is_letter(c) and not is_number(c):
            j = i + 1
        k = j
        while k < n and is_letter(text[k]):
            k += 1
        if k > j:
            chunks.append(text[i:k])
            i = k
            continue
        # Alt 3: \p{N}{1,3}+ (maximal, capped at 3).
        if is_number(c):
            k = i
            while k < n and k - i < 3 and is_number(text[k]):
                k += 1
            chunks.append(text[i:k])
            i = k
            continue
        # Alt 4: ' '? symbol++ [\r\n]* (symbols = not space/letter/number).
        k = i
        if text[k] == " ":
            k += 1
        m = k
        while m < n and is_nonspace(text[m]) and not is_letter(text[m]) \
                and not is_number(text[m]):
            m += 1
        if m > k:
            while m < n and text[m] in "\r\n":
                m += 1
            chunks.append(text[i:m])
            i = m
            continue
        # Alts 5-8: whitespace handling (c must be a space here).
        if is_space(c):
            r = i
            while r < n and is_space(text[r]):
                r += 1
            # Alt 5: \s++$ (whole tail when it reaches end of input).
            if r == n:
                chunks.append(text[i:r])
                i = r
                continue
            # Alt 6: \s*[\r\n] with backtracking = up to and including the
            # last \r|\n inside the whitespace run.
            p = r - 1
            while p >= i and text[p] not in "\r\n":
                p -= 1
            if p >= i:
                chunks.append(text[i:p + 1])
                i = p + 1
                continue
            # Alt 7: \s+(?!\S): longest run prefix followed by space or EOS.
            # Run ends mid-input before a non-space, so drop one char;
            # length-1 runs fall through to alt 8.
            if r - 1 > i:
                chunks.append(text[i:r - 1])
                i = r - 1
                continue
            # Alt 8: \s (single char).
            chunks.append(c)
            i += 1
            continue
        # Unreachable for well-formed alternatives, but never drop bytes:
        # (defensive; keeps concat(chunks) == text invariant by construction).
        raise AssertionError(f"cl100k splitter stalled at {i!r} in {text!r}")
    return chunks


# ---------------------------------------------------------------------------
# o200k_base splitter
# Pattern (byte-exact copy of PreTokenizer::O200K_BASE):
#   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
#     (?i:'s|'t|'re|'ve|'m|'ll|'d)?
#   |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
#     (?i:'s|'t|'re|'ve|'m|'ll|'d)?
#   |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/*]*|\s*[\r\n]+|\s+(?!\S)|\s+
# ---------------------------------------------------------------------------

_O200K_CONTRACTIONS = ("'s", "'S", "'t", "'T", "'re", "'RE", "'rE", "'Re",
                       "'ve", "'VE", "'vE", "'Ve", "'m", "'M", "'ll", "'lL",
                       "'Ll", "'LL", "'d", "'D")


def _match_o200k_contraction(text: str, pos: int) -> int:
    """Length of the optional trailing contraction at pos (0 if none)."""
    for form in _O200K_CONTRACTIONS:
        if text.startswith(form, pos):
            return len(form)
    return 0


def split_o200k(text: str) -> list[str]:
    """Leftmost-first-match scan with the o200k alternatives in order."""
    chunks: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        # Alts 1-2: letter words with optional prefix + contraction suffix.
        j = i
        if c not in "\r\n" and not is_letter(c) and not is_number(c):
            j = i + 1
        # Alt 1: upper* lower+ contraction?
        k = j
        while k < n and is_upper_word(text[k]):
            k += 1
        m = k
        while m < n and is_lower_word(text[m]):
            m += 1
        if m > k:
            m += _match_o200k_contraction(text, m)
            chunks.append(text[i:m])
            i = m
            continue
        # Alt 2: upper+ lower* contraction?
        k = j
        while k < n and is_upper_word(text[k]):
            k += 1
        if k > j:
            m = k
            while m < n and is_lower_word(text[m]):
                m += 1
            m += _match_o200k_contraction(text, m)
            chunks.append(text[i:m])
            i = m
            continue
        # Alt 3: \p{N}{1,3} (maximal, capped at 3).
        if is_number(c):
            k = i
            while k < n and k - i < 3 and is_number(text[k]):
                k += 1
            chunks.append(text[i:k])
            i = k
            continue
        # Alt 4: ' '? symbol+ [\r\n/*]*.
        k = i
        if text[k] == " ":
            k += 1
        m = k
        while m < n and is_nonspace(text[m]) and not is_letter(text[m]) \
                and not is_number(text[m]):
            m += 1
        if m > k:
            while m < n and text[m] in "\r\n/*":
                m += 1
            chunks.append(text[i:m])
            i = m
            continue
        # Alts 5-7: whitespace handling.
        if is_space(c):
            r = i
            while r < n and is_space(text[r]):
                r += 1
            # Alt 5: \s*[\r\n]+ = run truncated after its last \r|\n.
            p = r - 1
            while p >= i and text[p] not in "\r\n":
                p -= 1
            if p >= i:
                q = p + 1
                while q < r and text[q] in "\r\n":
                    q += 1
                # q stops at first non-newline after the cluster; but the
                # longest valid match extends through trailing newlines only
                # up to the run region already covered -- clamp to run end
                # handling: trailing newlines run to q, spaces after break it.
                # Recompute: match ends at end of the \r\n cluster containing p.
                e = p + 1
                while e < n and text[e] in "\r\n":
                    e += 1
                chunks.append(text[i:e])
                i = e
                continue
            # Alt 6: \s+(?!\S).
            if r == n or not is_nonspace(text[r]):
                chunks.append(text[i:r])
                i = r
                continue
            if r - 1 > i:
                chunks.append(text[i:r - 1])
                i = r - 1
                continue
            # Alt 7: \s+ (maximal run).
            chunks.append(text[i:r])
            i = r
            continue
        raise AssertionError(f"o200k splitter stalled at {i!r} in {text!r}")
    return chunks


SPLITTERS = {"cl100k_base": split_cl100k, "o200k_base": split_o200k}

# ---------------------------------------------------------------------------
# Vocabulary + rank-ordered BPE application (mirrors domlm::tokenizer)
# ---------------------------------------------------------------------------


def load_vocab(vocab_path: str) -> list[bytes]:
    """vocab.json -> index-addressed raw byte tokens (latin-1 restores bytes).

    Our writer emits printable ASCII verbatim and every other byte as
    \\u00XX (plus short escapes for C0 controls); every escape value fits
    in one byte, so latin-1 maps each decoded char back to its source byte.
    """
    with open(vocab_path, "r", encoding="ascii") as f:
        doc = json.load(f)
    tokens = doc["tokens"]
    if doc.get("vocab_size", len(tokens)) != len(tokens):
        raise ValueError(f"{vocab_path}: vocab_size disagrees with token count")
    by_id: list[bytes | None] = [None] * len(tokens)
    for entry in tokens:
        i = entry["id"]
        if not isinstance(i, int) or i < 0 or i >= len(tokens):
            raise ValueError(f"{vocab_path}: token id out of range: {i!r}")
        if by_id[i] is not None:
            raise ValueError(f"{vocab_path}: duplicate token id {i}")
        by_id[i] = entry["text"].encode("latin-1")
    if any(t is None for t in by_id):
        raise ValueError(f"{vocab_path}: non-sequential token ids")
    if len(by_id[255]) != 1 or by_id[0] != b"\x00":
        raise ValueError(f"{vocab_path}: base 0..255 byte table corrupt")
    return [t for t in by_id if t is not None]


def load_merges(merges_path: str) -> dict[tuple[int, int], int]:
    """merges.txt ('a b -> c' per line, '#' comments skipped) -> rank map."""
    rank: dict[tuple[int, int], int] = {}
    with open(merges_path, "r", encoding="ascii") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4 or parts[2] != "->":
                raise ValueError(f"{merges_path}:{lineno}: malformed line {raw!r}")
            a, b, c = int(parts[0]), int(parts[1]), int(parts[3])
            if (a, b) in rank:
                raise ValueError(f"{merges_path}:{lineno}: duplicate pair {(a, b)}")
            rank[(a, b)] = c
    return rank


class BpeEncoder:
    """Stateless-ish encoder: pre-split, then lowest-rank-first merges."""

    def __init__(self, vocab: list[bytes], rank: dict[tuple[int, int], int],
                 ruleset: str = "cl100k_base") -> None:
        if ruleset not in SPLITTERS:
            raise ValueError(f"unknown ruleset {ruleset!r}")
        self._vocab = vocab
        self._rank = rank
        self._split = SPLITTERS[ruleset]
        self.ruleset = ruleset

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode_chunk(self, chunk: str, out: list[int]) -> None:
        """Appends chunk token ids (chunk = pre-tokenizer output unit)."""
        raw = chunk.encode("utf-8")  # chunks are sanitized valid UTF-8
        work = list(raw)  # one base id per byte (byte identity)
        while True:
            best_pair: tuple[int, int] | None = None
            best_rank: int | None = None
            for x, y in zip(work, work[1:]):
                r = self._rank.get((x, y))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = (x, y)
            if best_pair is None or best_rank is None:
                break
            merged: list[int] = []
            k = 0
            while k < len(work):
                if k + 1 < len(work) and work[k] == best_pair[0] \
                        and work[k + 1] == best_pair[1]:
                    merged.append(best_rank)
                    k += 2
                else:
                    merged.append(work[k])
                    k += 1
            work = merged
        out.extend(work)

    def encode_text(self, text: str) -> list[int]:
        """Full pipeline for one line (terminator included by caller)."""
        ids: list[int] = []
        for chunk in self._split(text):
            if chunk:
                self.encode_chunk(chunk, ids)
        return ids


# ---------------------------------------------------------------------------
# Multiprocessing driver (order-preserving) + LE uint16 packing
# ---------------------------------------------------------------------------

_GLOBAL_ENCODER: BpeEncoder | None = None


def _worker_init(vocab: list[bytes], rank: dict[tuple[int, int], int],
                 ruleset: str) -> None:
    global _GLOBAL_ENCODER
    _GLOBAL_ENCODER = BpeEncoder(vocab, rank, ruleset)


def _worker_batch(lines: list[str]) -> list[int]:
    assert _GLOBAL_ENCODER is not None
    ids: list[int] = []
    for text in lines:
        for chunk in _GLOBAL_ENCODER._split(text):
            if chunk:
                _GLOBAL_ENCODER.encode_chunk(chunk, ids)
    return ids


def pack_ids_le(ids: list[int], handle) -> None:
    """Appends ids as little-endian uint16 in 64k blocks (no numpy)."""
    for off in range(0, len(ids), 65536):
        block = ids[off:off + 65536]
        handle.write(struct.pack("<%dH" % len(block), *block))


def prepare(input_path: str, vocab_path: str, merges_path: str,
            output_path: str, ruleset: str, jobs: int) -> dict:
    t0 = time.perf_counter()
    vocab = load_vocab(vocab_path)
    rank = load_merges(merges_path)
    if len(vocab) > 65536:
        raise ValueError("vocab too large for uint16 packing")
    if rank and max(rank.values()) >= len(vocab):
        raise ValueError("merges reference ids outside the vocabulary")

    with open(input_path, "rb") as f:
        raw_lines = f.read().splitlines(keepends=True)
    # splitlines drops a phantom: b"a\n" -> [b"a\n"]; exact file bytes kept.
    texts = [b.decode("utf-8") for b in raw_lines]  # strict: fail loudly

    total_ids = 0
    with open(output_path, "wb") as out:
        if jobs < 2:
            enc = BpeEncoder(vocab, rank, ruleset)
            pending: list[int] = []
            for text in texts:
                pending.extend(enc.encode_text(text))
                if len(pending) >= 1 << 20:
                    pack_ids_le(pending, out)
                    total_ids += len(pending)
                    pending = []
            if pending:
                pack_ids_le(pending, out)
                total_ids += len(pending)
        else:
            batch = max(1, len(texts) // (jobs * 8))
            batches = [texts[k:k + batch] for k in range(0, len(texts), batch)]
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=jobs,
                    initializer=_worker_init,
                    initargs=(vocab, rank, ruleset)) as pool:
                for ids in pool.map(_worker_batch, batches, chunksize=1):
                    pack_ids_le(ids, out)
                    total_ids += len(ids)

    dt = time.perf_counter() - t0
    return {"lines": len(texts), "tokens": total_ids,
            "seconds": dt, "tok_per_s": total_ids / max(dt, 1e-9)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"DomLM v{domlm_version()} dataset packer")
    ap.add_argument("--version", action="version",
                    version=f"DomLM {domlm_version()}")
    ap.add_argument("--input", required=True, help="raw text corpus")
    ap.add_argument("--vocab", required=True, help="tokenizer/vocab.json")
    ap.add_argument("--merges", required=True, help="tokenizer/merges.txt")
    ap.add_argument("--output", required=True, help="packed train.bin")
    ap.add_argument("--ruleset", default="cl100k_base",
                    choices=["cl100k_base", "o200k_base"])
    ap.add_argument("--jobs", type=int, default=4,
                    help="worker processes (0/1 = serial)")
    args = ap.parse_args(argv)

    stats = prepare(args.input, args.vocab, args.merges, args.output,
                    args.ruleset, max(0, args.jobs))
    size = os.path.getsize(args.output)
    print(f"DomLM v{domlm_version()}: packed {stats['lines']} lines -> {stats['tokens']} tokens "
          f"({size} bytes) in {stats['seconds']:.1f}s "
          f"[{stats['tok_per_s']:.0f} tok/s] -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

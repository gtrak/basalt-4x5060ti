#!/usr/bin/env python3
"""Tests for tools/quality_fixture.py: the chunk builder and the fixture validator.

The pack's tokenizer is a 105 GiB file no test should open, so the builder is driven with a stand-in tokenizer that
splits on whitespace and is therefore exactly invertible - which is what makes the assertions about chunk length,
coverage and duplication observable rather than approximate.

    python3 tools/test_quality_fixture.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quality_fixture as QF                     # noqa: E402


class WordTokenizer:
    """A tokenizer with the shape the builder uses: encode splits words into ids, decode is its exact inverse."""

    def __init__(self):
        self.vocab = {}
        self.tokens = []

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.tokens)
            self.tokens.append(w)
        return self.vocab[w]

    def encode(self, text):
        return [self._id(w) for w in text.split() if w]

    def decode(self, ids):
        return " ".join(self.tokens[i] for i in ids)


def repo_with(files: dict) -> pathlib.Path:
    root = pathlib.Path(tempfile.mkdtemp())
    for name, body in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def repo_from_spans() -> pathlib.Path:
    """Every source the builder names, present and long enough that each named offset actually lands inside it - a
    slice past EOF would come back empty and make two chunks identical for a reason that has nothing to do with
    the builder."""
    need = max(off + ln for _, off, ln, _ in QF.SPANS) + 1000
    files = {}
    for src, _, _, _ in QF.SPANS:
        if src in files:
            continue
        words, n = [], 0
        while sum(len(w) + 1 for w in words) < need:
            words.append(f"{src.replace('/', '_')}_{n}")
            n += 1
        files[src] = " ".join(words)
    return repo_with(files)


class Builder(unittest.TestCase):
    def test_every_chunk_is_exactly_its_bucket(self):
        tk = WordTokenizer()
        chunks = QF.build(tk, repo_from_spans())
        self.assertEqual(len(chunks), sum(w for _, w in QF.BUCKETS))
        for e in chunks:
            self.assertEqual(len(e["ids"]), e["bucket"], f"{e['id']}: ids must fill the bucket exactly")
            self.assertEqual(e["text"], tk.decode(e["ids"]), f"{e['id']}: text must decode from the ids")
            self.assertTrue(e["source"], f"{e['id']} names no source")
            self.assertTrue(e["domain"], f"{e['id']} names no domain")

    def test_no_two_chunks_are_the_same_tokens(self):
        chunks = QF.build(WordTokenizer(), repo_from_spans())
        seen = {tuple(e["ids"]) for e in chunks}
        self.assertEqual(len(seen), len(chunks), "two chunks carry the same scored tokens")

    def test_the_buckets_are_the_ones_the_gate_prices(self):
        chunks = QF.build(WordTokenizer(), repo_from_spans())
        got = {}
        for e in chunks:
            got[e["bucket"]] = got.get(e["bucket"], 0) + 1
        self.assertEqual(got, dict(QF.BUCKETS))

    def test_a_missing_source_is_refused_not_skipped(self):
        root = repo_from_spans()
        (root / QF.SPANS[0][0]).unlink()
        with self.assertRaises(SystemExit):
            QF.build(WordTokenizer(), root)

    def test_spans_too_short_to_fill_a_bucket_are_refused(self):
        with self.assertRaises(SystemExit):
            QF.build(WordTokenizer(), repo_with({s[0]: "far too short" for s in QF.SPANS}))


class Validator(unittest.TestCase):
    def check(self, entries, tk):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            path = pathlib.Path(f.name)
        code = QF.check(path, tk)          # the same tokenizer that built the ids: a fresh one knows no words
        path.unlink()
        return code

    def entry(self, tk, n, bucket=8, i=0):
        ids = tk.encode(" ".join(f"w{i}_{j}" for j in range(n)))
        return {"id": f"c{i:03d}", "domain": "prose", "bucket": bucket,
                "source": "x", "through": "x", "text": tk.decode(ids), "ids": ids}

    def test_a_good_fixture_validates(self):
        tk = WordTokenizer()
        self.assertEqual(self.check([self.entry(tk, 8, i=0), self.entry(tk, 8, i=1)], tk), 0)

    def test_a_short_chunk_is_caught(self):
        tk = WordTokenizer()
        e = self.entry(tk, 8)
        e["ids"] = e["ids"][:5]
        self.assertEqual(self.check([e], tk), 1)

    def test_text_that_does_not_decode_from_the_ids_is_caught(self):
        tk = WordTokenizer()
        e = self.entry(tk, 8)
        e["text"] = "something else entirely here now"
        self.assertEqual(self.check([e], tk), 1)

    def test_a_duplicate_chunk_is_caught(self):
        tk = WordTokenizer()
        e = self.entry(tk, 8)
        dup = dict(e)
        dup["id"] = "c999"
        self.assertEqual(self.check([e, dup], tk), 1)

    def test_an_id_outside_the_vocabulary_is_caught(self):
        tk = WordTokenizer()
        e = self.entry(tk, 8)
        e["ids"][0] = 10 ** 6
        self.assertEqual(self.check([e], tk), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

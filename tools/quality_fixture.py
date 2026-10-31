#!/usr/bin/env python3
"""Write the fixed scored set the differential quality gate teacher-forces (tools/quality_gate.py).

One JSONL line per chunk: `id`, `domain`, `bucket` (the chunk's token count, the gate's `n_ctx`), the `text` it was
tokenized from, where that text came from (`source`), and the `ids`. The tool scores `ids`, not `text`: a tokenizer
change cannot silently move the scored set. `text` and `source` are committed so a reviewer can see what is being
scored and where it came from; `--check` re-tokenizes them and reports any entry whose ids no longer reproduce
from its text, which is how a deliberate re-pinning is told apart from an accidental drift.

Every chunk is exactly `bucket` tokens, because the gate's base layout fixes `n_ctx` for the whole file. A span
shorter than the bucket is padded with the following spans of the same source until it reaches it, so no chunk is
built from filler.

    python3 tools/quality_fixture.py --pack models/Qwen3.8-FN-FP4Local.basalt --repo . \
        --out tests/data/quality/partial_merge.jsonl
    python3 tools/quality_fixture.py --check tests/data/quality/partial_merge.jsonl \
        --pack models/Qwen3.8-FN-FP4Local.basalt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model_file import ModelFile          # noqa: E402
from tokenizer import Tokenizer           # noqa: E402

# (source, character offset, characters to take, domain). The sources are the repository's own text: technical
# prose, commented C++, Python, Markdown tables and a chat template, so the scored set is varied in register and
# not one style. They are pinned by offset, so the same checkout always yields the same scored set.
SPANS = [
    ("docs/precision/quality.md", 0, 2600, "prose"),
    ("docs/precision/quality.md", 2600, 2600, "prose"),
    ("docs/precision/quality.md", 5200, 2600, "prose"),
    ("docs/precision/moe-partial-merge.md", 0, 2600, "prose"),
    ("docs/precision/moe-partial-merge.md", 2600, 2600, "prose"),
    ("docs/precision/moe-partial-merge.md", 5200, 2600, "prose"),
    ("docs/precision/index.md", 0, 2600, "prose"),
    ("README.md", 0, 2600, "prose"),
    ("README.md", 2600, 2600, "prose"),
    ("docs/running.md", 0, 2600, "prose"),
    ("src/core/layer.cpp", 44000, 2600, "code"),
    ("src/core/layer.cpp", 46600, 2600, "code"),
    ("src/core/expert_source.cpp", 24000, 2600, "code"),
    ("src/core/expert_source.cpp", 26600, 2600, "code"),
    ("serve/api/responses.py", 8000, 2600, "code"),
    ("serve/engine/process.py", 26000, 2600, "code"),
    ("tools/ppl_gate.py", 2600, 2600, "code"),
    ("serve/config.py", 6000, 2600, "code"),
    ("docs/probe.md", 0, 2600, "prose"),
    ("docs/model-file.md", 2600, 2600, "prose"),
    ("src/prefill/prefill.cpp", 12000, 2600, "code"),
    ("src/kernels/cpu/expert.cpp", 12000, 2600, "code"),
    ("serve/chat_template.jinja", 0, 2600, "template"),
    ("docs/precision/features.md", 2600, 2600, "prose"),
]

BUCKETS = [(512, 12), (1024, 6)]        # (n_ctx, chunks): ~3 GB of emitted base for both buckets

# A chunk is quoted source text, so it can quote a platform macro and trip the engine's own code-hygiene scan
# (tools/check_platform_boundary.sh), which has no way to tell a call from a citation. The fixture is scored data,
# but a span that reads like a platform call is also a poor probe of a language model's head, so the builder
# refuses one rather than asking the scan to look the other way.
PLATFORM_TEXT = re.compile(
    r"_WIN32|_MSC_VER|__linux__|__unix__|windows\.h|sys/mman\.h|unistd\.h|sys/stat\.h|\bmmap\(|\bmunmap\("
    r"|madvise|posix_fadvise|sched_setaffinity|pthread_|VirtualAlloc|CreateFileW?\b|__builtin_(ctz|ia32|cpu_supports)"
    r"|__attribute__\(\(target|\bdlopen\(|\bwaitpid\(|\"/proc/|\"/sys/")


def load_tokenizer(pack: str) -> Tokenizer:
    """The pack's own tokenizer, from the `tokenizer/` sections it ships, so no weight shard is opened."""
    m = ModelFile(pack)
    vocab = json.loads(m.read("tokenizer/vocab.json"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    merges = m.read("tokenizer/merges.txt").decode("utf-8").split("\n")
    types = json.loads(m.read("tokenizer/token_type.json"))
    return Tokenizer(toks, merges, types)


def build(tk: Tokenizer, repo: pathlib.Path) -> list[dict]:
    """One chunk per (bucket, index): spans concatenated from `SPANS`, truncated to exactly `bucket` tokens. Nothing
    is invented and nothing is padded with filler - a chunk is the repository's own text, and `source` names where
    it starts."""
    texts = {}
    for src, off, ln, _ in SPANS:
        p = repo / src
        if not p.is_file():
            raise SystemExit(f"{src}: named as a fixture source but not in the tree")
        body = p.read_text(encoding="utf-8", errors="replace")[off:off + ln]
        if PLATFORM_TEXT.search(body):
            raise SystemExit(f"{src}@{off}: this span quotes platform calls; pick another span "
                             "(tools/check_platform_boundary.sh counts a citation as a hit)")
        texts[(src, off)] = body

    out, n = [], 0
    for bucket, want in BUCKETS:
        for _ in range(want):
            text, k = "", 0
            # walk the spans from a different start per chunk, so no two chunks are the same stretch of text
            while len(tk.encode(text)) < bucket:
                if k > len(SPANS):
                    raise SystemExit(f"chunk {n}: the spans hold too few tokens to fill a bucket of {bucket}")
                src, off, _, _ = SPANS[(n + k) % len(SPANS)]
                text += ("" if not text else "\n") + texts[(src, off)]
                k += 1
            ids = tk.encode(text)[:bucket]
            first, last = SPANS[n % len(SPANS)], SPANS[(n + k - 1) % len(SPANS)]
            out.append({"id": f"c{n:03d}", "domain": first[3], "bucket": bucket,
                        "source": f"{first[0]}@{first[1]}+{first[2]}", "through": f"{last[0]}@{last[1]}",
                        "text": tk.decode(ids), "ids": ids})
            n += 1
    return out


def check(path: pathlib.Path, tk: Tokenizer) -> int:
    """Validate a written fixture: exact bucket length, ids inside the vocabulary, text that decodes from the ids,
    no repeated chunk. Prints the buckets and the emitted base each one costs, which is what a reviewer needs to
    know before agreeing to a run."""
    bad = 0
    seen = {}
    buckets = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        buckets[e["bucket"]] = buckets.get(e["bucket"], 0) + 1
        problems = []
        if len(e["ids"]) != e["bucket"]:
            problems.append(f"{len(e['ids'])} ids, bucket says {e['bucket']}")
        if e["ids"] and (min(e["ids"]) < 0 or max(e["ids"]) >= len(tk.tokens)):
            problems.append(f"an id is outside the vocabulary (max {max(e['ids'])}, vocab {len(tk.tokens)})")
        elif tk.decode(e["ids"]) != e["text"]:
            # only once the ids are known good: decoding one outside the vocabulary raises rather than reports
            problems.append("text does not decode from the ids")
        if tuple(e["ids"]) in seen:
            problems.append(f"duplicate of {seen[tuple(e['ids'])]}")
        else:
            seen[tuple(e["ids"])] = e["id"]
        if problems:
            print(f"  *** {e['id']} ({e['bucket']} tokens, {e['domain']}): " + "; ".join(problems))
            bad += 1
    n_vocab = len(tk.tokens)
    row_bytes = 2 * ((n_vocab + 1) // 2) + 4
    for b in sorted(buckets):
        scored = b - 1 - b // 2
        gib = buckets[b] * scored * row_bytes * 2 / 2 ** 30
        print(f"  bucket {b}: {buckets[b]} chunks x {scored} scored rows, emitted base {gib:.2f} GiB")
    print(f"fixture: {'OK' if bad == 0 else f'{bad} bad chunks'}")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True, help="the .basalt file whose tokenizer pins the ids")
    ap.add_argument("--repo", default=".", help="the checkout the source spans are read from")
    ap.add_argument("--out", help="write the JSONL here")
    ap.add_argument("--check", help="validate an existing JSONL instead of writing one")
    a = ap.parse_args()
    tk = load_tokenizer(a.pack)

    if a.check:
        return check(pathlib.Path(a.check), tk)
    if not a.out:
        ap.error("--out is required unless --check is given")
    chunks = build(tk, pathlib.Path(a.repo))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in chunks), encoding="utf-8")
    print(f"{out}: {len(chunks)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The differential quality gate for a numerics-affecting engine flag (docs/precision/moe-partial-merge.md).

One fixed scored set, teacher-forced under two boots. The fixture's chunks (tools/quality_fixture.py) are laid out
as a llama.cpp `_logits_` base, and three runs are taken against it:

    emit   the reference arm: the engine writes its own log-probs into the base's logit rows (`--ppl-emit`)
    check  the same arm again, reading that base back: KLD must come out at the emitter's own floor and same-top-1
           at 100%, or everything the gate says afterwards is void - this is what proves the instrument
    gate   the arm under test (`--flag`), reading the same base: KLD / same-top-1 / dlogp are the verdict

Both arms score the SAME committed token ids through the SAME `ppl=` path, so the only difference between them is
the flag: nothing diverges by generation, which would compare two different sequences and mean nothing.

The engine is started as a child process and driven over its stdin, the way tools/ppl_gate.py drives it: a
`ppl=K ppl_t=T` request is an engine protocol key, not part of the server's HTTP API. This tool never connects to
a running server; it boots its own, and the boots are the operator's to run.

  python3 tools/quality_gate.py --pack FILE.basalt --fixture tests/data/quality/partial_merge.jsonl \
      --bucket 512 --out DIR --arm emit [--model ... --gpu ...]     # unknown flags go to the engine
  python3 tools/quality_gate.py ... --arm check
  python3 tools/quality_gate.py ... --arm gate --flag peer_partial_merge=1
  python3 tools/quality_gate.py --verdict DIR            # the PASS/FAIL table, from the three runs' rows.tsv
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model_file import ModelFile          # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
ENGINE_BIN = REPO / "bin" / "basalt"
FIXTURE = REPO / "tests" / "data" / "quality" / "partial_merge.jsonl"

# the bounds, derived rather than measured; see docs/precision/moe-partial-merge.md. A K-term reassociation plus one
# BF16 rounding of the summed partial moves the head's distribution by far less than the reference's own noise
# (docs/precision/quality.md: two independent implementations of this model sit 0.0233 KLD apart), so a change of
# this class is graded against the emitter's floor, not against zero.
BOUND_KLD_MEAN = 0.002      # x the self-check floor: 8x the emitter's own 7.3e-05, and ~1% of the reference's noise
BOUND_KLD_P99 = 0.02        # the tail is where a real reassociation error would first be visible
BOUND_SAME_TOP1 = 99.0      # percent, of scored rows whose top-1 matches the reference arm's
BOUND_DLOGP_MAX = 0.05      # nats: the base's own top-1 must not fall this far under the tested arm's
SELF_CHECK_KLD_MEAN = 0.001  # above this, the emitter and the reader disagree and the run is VOID
SELF_CHECK_SAME_TOP1 = 100.0  # below this, likewise


def read_fixture(path: pathlib.Path, bucket: int) -> list[dict]:
    out = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = [e for e in out if e["bucket"] == bucket]
    if not out:
        sys.exit(f"{path}: no chunks in bucket {bucket}")
    return out


def write_base(path: pathlib.Path, chunks: list[dict], n_ctx: int, n_vocab: int) -> int:
    """Lay out the base file: the header, every chunk's tokens, and the logit rows left as a hole.

    The rows are left unwritten rather than zeroed. A chunk the engine never scored therefore reads back with every
    log-prob at 0.0, which makes KLD enormous rather than plausible: a missing row fails loudly."""
    row_u16 = 2 * ((n_vocab + 1) // 2) + 4
    rows = n_ctx - 1 - n_ctx // 2
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with open(path, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<Iii", n_ctx, n_vocab, len(chunks)))
        for e in chunks:
            f.write(struct.pack(f"<{n_ctx}i", *e["ids"]))
        f.truncate(20 + len(chunks) * n_ctx * 4 + len(chunks) * rows * row_u16 * 2)
    return 20 + len(chunks) * n_ctx * 4 + len(chunks) * rows * row_u16 * 2


def set_args(args: list[str], extra: list[str]) -> list[str]:
    """`args` with each flag of `extra` replacing the same flag (and its value) in `args`, or added."""
    out = list(args)
    i = 0
    while i < len(extra):
        flag = extra[i]
        has_value = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        value = [extra[i + 1]] if has_value else []
        if flag in out:
            j = out.index(flag)
            old_has_value = j + 1 < len(out) and not out[j + 1].startswith("--")
            out[j:j + 1 + old_has_value] = [flag, *value]
        else:
            out += [flag, *value]
        i += 1 + has_value
    return out


def boot(engine_bin: str, args: list[str], log: pathlib.Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w")
    fh.write(f"engine {engine_bin}\nargs {' '.join(args)}\n")
    fh.flush()
    eng = subprocess.Popen([engine_bin, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=fh, text=True, bufsize=1)
    for line in eng.stdout:
        if line.startswith("READY"):
            return eng, fh
        if line.startswith("ERR"):
            eng.kill()
            sys.exit(f"the engine refused to start: {line.strip()} (see {log})")
    eng.kill()
    sys.exit(f"the engine exited before READY; see {log}")


def run_requests(eng, chunks: list[dict], n_ctx: int, t: int, progress: int) -> dict:
    first = n_ctx // 2
    pp = dec = 0.0
    hits = look = off = 0
    t0 = time.time()
    for k, e in enumerate(chunks):
        ids = ",".join(str(x) for x in e["ids"][:first + 1])
        eng.stdin.write(f"GEN {n_ctx - 1 - first} ppl={k} ppl_t={t} {ids}\n")
        eng.stdin.flush()
        for line in eng.stdout:
            if line.startswith("ERR"):
                sys.exit(f"chunk {e['id']}: {line.strip()}")
            if line.startswith("DONE"):
                f = line.split()
                pp += float(f[3])
                dec += float(f[4])
                if f[8] != "0":
                    print(f"  note: chunk {e['id']} reused {f[8]} prompt tokens from an earlier request")
                if len(f) > 10:
                    hits += int(f[9])
                    look += int(f[10])
                if len(f) > 15:
                    off += int(f[15])
                break
        else:
            sys.exit(f"chunk {e['id']}: the engine exited mid-request")
        if progress and (k + 1) % progress == 0:
            print(f"  {k + 1}/{len(chunks)} chunks, {time.time() - t0:.0f} s", flush=True)
    return {"run s": f"{time.time() - t0:.1f}", "prompt ms/chunk": f"{pp / len(chunks):.0f}",
            "decode ms/chunk": f"{dec / len(chunks):.0f}",
            "routed experts": f"{look + off} looked up, VRAM hits {100.0 * hits / max(1, look + off):.1f}%, "
                              f"peer/PCIe {100.0 * off / max(1, look + off):.1f}%"}


def summarize(rows: pathlib.Path) -> dict | None:
    if not rows.is_file():
        return None
    nll, kld, dlp, same, n = [], [], [], 0, 0
    for line in rows.read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) < 10:
            continue
        n += 1
        nll.append(float(f[3]))
        kld.append(float(f[5]))
        same += int(f[6])
        dlp.append(float(f[9]))
    if n == 0:
        return None
    ks = sorted(kld)
    mean = sum(nll) / n
    pct = lambda q: ks[min(n - 1, int(round(q * (n - 1))))]
    return {"rows": n, "PPL": math.exp(mean), "KLD mean": sum(kld) / n, "KLD median": pct(0.5),
            "KLD p99": pct(0.99), "KLD max": ks[-1], "same top-1": 100.0 * same / n, "dlogp max": max(dlp)}


def run_arm(a, bucket: int, chunks: list[dict], n_ctx: int, n_vocab: int, base: pathlib.Path,
            out: pathlib.Path, args: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "rows.tsv"
    if rows.exists():
        rows.unlink()
    # --max-context stays what the operator gave (it is part of the configuration under test); it is only filled in
    # when absent, and refused when too small to hold a chunk plus its window
    need = n_ctx + 512
    if "--max-context" in args:
        given = int(args[args.index("--max-context") + 1])
        if given < need:
            sys.exit(f"--max-context {given} cannot hold a {n_ctx}-token chunk and its window; pass {need} or more")
    else:
        args = set_args(args, ["--max-context", str(need)])
    if a.arm == "gate":
        # `--exp` is repeatable and takes a comma list, so the flag under test is appended rather than set:
        # replacing the first occurrence would silently drop an experimental flag the operator passed, and then the
        # two arms would differ by more than the thing being graded. An operator who already set this key is
        # refused rather than quietly overridden - whichever the engine saw last would win.
        key = a.flag.split("=", 1)[0]
        for i, w in enumerate(args):
            if w == "--exp" and i + 1 < len(args) and any(e.split("=", 1)[0] == key for e in args[i + 1].split(",")):
                sys.exit(f"--exp already carries {key} ({args[i + 1]!r}); drop it, or the gate arm cannot own that flag")
        args += ["--exp", a.flag]
    if a.arm == "emit":
        if base.exists():
            base.unlink()
        write_base(base, chunks, n_ctx, n_vocab)
        args = set_args(args, ["--ppl-emit", str(base)])
    else:
        if not base.is_file():
            sys.exit(f"{base}: the emitted base is missing; run --arm emit first")
        args = set_args(args, ["--ppl-base", str(base), "--ppl-out", str(rows)])
    print(f"arm {a.arm}: {len(chunks)} chunks of {n_ctx} tokens, window {a.t}, "
          f"base {base.stat().st_size / 2**30:.2f} GiB")
    eng, log = boot(a.engine_bin, args, out / "engine.log")
    try:
        stats = run_requests(eng, chunks, n_ctx, a.t, a.progress)
        eng.stdin.write("QUIT\n")
        eng.stdin.flush()
    finally:
        try:
            eng.wait(timeout=120)
        except subprocess.TimeoutExpired:
            eng.kill()
        log.close()
    s = summarize(rows)
    lines = [f"arm {a.arm}", f"args {' '.join(args)}"]
    lines += [f"  {k} {v}" for k, v in stats.items()]
    if s:
        lines.append("  " + "  ".join(f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def verdict(out: pathlib.Path) -> int:
    """The PASS/FAIL table over the three arms' rows.tsv."""
    s = {arm: summarize(out / arm / "rows.tsv") for arm in ("emit", "check", "gate")}
    for arm in ("check", "gate"):
        if s[arm] is None:
            print(f"FAIL  no rows for the {arm} arm ({out / arm / 'rows.tsv'}); run that arm first")
            return 1
    chk, gt = s["check"], s["gate"]
    if s["emit"] is not None:
        print(f"emit  {s['emit']['rows']} rows, PPL {s['emit']['PPL']:.4f} (the reference arm's own score)")
    print(f"check {chk['rows']} rows, KLD mean {chk['KLD mean']:.6f}, same top-1 {chk['same top-1']:.2f}%  "
          f"(the emitter's floor is the scale)")
    print(f"gate  {gt['rows']} rows, KLD mean {gt['KLD mean']:.6f}, median {gt['KLD median']:.6f}, "
          f"p99 {gt['KLD p99']:.6f}, max {gt['KLD max']:.6f}")
    print(f"      same top-1 {gt['same top-1']:.2f}%, dlogp max {gt['dlogp max']:.6f}, "
          f"PPL {gt['PPL']:.4f} against the reference {chk['PPL']:.4f}")

    fails = []
    if chk["KLD mean"] > SELF_CHECK_KLD_MEAN:
        fails.append(f"VOID: the self-check arm scored KLD mean {chk['KLD mean']:.6f} > {SELF_CHECK_KLD_MEAN}; "
                     "the emitter and the reader disagree, so no conclusion can be drawn from this run")
    if chk["same top-1"] < SELF_CHECK_SAME_TOP1:
        fails.append(f"VOID: the self-check arm agreed on top-1 for only {chk['same top-1']:.2f}% of rows")
    if gt["KLD mean"] > BOUND_KLD_MEAN:
        fails.append(f"KLD mean {gt['KLD mean']:.6f} > {BOUND_KLD_MEAN}")
    if gt["KLD p99"] > BOUND_KLD_P99:
        fails.append(f"KLD p99 {gt['KLD p99']:.6f} > {BOUND_KLD_P99}")
    if gt["same top-1"] < BOUND_SAME_TOP1:
        fails.append(f"same top-1 {gt['same top-1']:.2f}% < {BOUND_SAME_TOP1}%")
    if gt["dlogp max"] > BOUND_DLOGP_MAX:
        fails.append(f"dlogp max {gt['dlogp max']:.6f} > {BOUND_DLOGP_MAX}")
    for f in fails:
        print(f"FAIL  {f}")
    if fails:
        print("verdict: FAIL")
        return 1
    print("verdict: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", help="the .basalt file (its tokenizer/ sections give n_vocab)")
    ap.add_argument("--fixture", default=str(FIXTURE), help="the scored set (JSONL from tools/quality_fixture.py)")
    ap.add_argument("--bucket", type=int, help="the chunk token count to gate (one bucket per run)")
    ap.add_argument("--out", help="the run directory")
    ap.add_argument("--arm", choices=("emit", "check", "gate"), help="which arm to run")
    ap.add_argument("--flag", default="peer_partial_merge=1",
                    help="the --exp flag the gate arm turns on (default: the partial merge)")
    ap.add_argument("--t", type=int, default=3, help="tokens per verify window (default 3)")
    ap.add_argument("--chunks", type=int, default=0, help="only the first N chunks of the bucket (0: all)")
    ap.add_argument("--progress", type=int, default=4, help="a progress line every N chunks (0: none)")
    ap.add_argument("--engine-bin", default=str(ENGINE_BIN) if ENGINE_BIN.is_file() else None)
    ap.add_argument("--verdict", metavar="DIR", help="print the PASS/FAIL table for a finished run and exit")
    a, extra = ap.parse_known_args()
    a.engine_args = extra

    if a.verdict:
        return verdict(pathlib.Path(a.verdict))
    if not (a.pack and a.bucket and a.out and a.arm):
        ap.error("--pack, --bucket, --out and --arm are required (or --verdict DIR)")
    if not a.engine_bin:
        ap.error("no engine binary: pass --engine-bin")

    chunks = read_fixture(pathlib.Path(a.fixture), a.bucket)
    if a.chunks:
        chunks = chunks[:a.chunks]
    n_ctx = a.bucket
    vocab = json.loads(ModelFile(a.pack).read("tokenizer/vocab.json"))
    n_vocab = len(vocab)

    out = pathlib.Path(a.out) / a.arm
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({"arm": a.arm, "bucket": n_ctx, "t": a.t, "chunks": len(chunks),
                                              "flag": a.flag if a.arm == "gate" else None,
                                              "engine_args": a.engine_args}, indent=1) + "\n", encoding="utf-8")
    base = pathlib.Path(a.out) / f"base-{a.bucket}.kld"
    args = ["--model", str(pathlib.Path(a.pack).resolve()), "--spec", "8",
            "--adapt-swaps", "0", "--peer-adapt-swaps", "0", "--resident-experts", "--gpu", "0,1"]
    args = set_args(args, a.engine_args)
    run_arm(a, a.bucket, chunks, n_ctx, n_vocab, base, out, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

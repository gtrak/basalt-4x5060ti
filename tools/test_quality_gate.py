#!/usr/bin/env python3
"""Tests for tools/quality_gate.py: the base layout the engine reads, the rows.tsv parsing, and the verdict.

No engine, no GPU, no server: the layout is checked against the byte offsets `PplGate` reads at, the parsing against
synthetic rows, and the verdict against synthetic runs whose numbers were chosen to sit on each side of every bound.

    python3 tools/test_quality_gate.py
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quality_gate as QG                      # noqa: E402

N_CTX = 64
N_VOCAB = 17                  # odd, so the even-count row padding is exercised
ROW_U16 = 2 * ((N_VOCAB + 1) // 2) + 4


def chunks(n=3, n_ctx=N_CTX):
    return [{"id": f"c{i:03d}", "domain": "prose", "bucket": n_ctx, "source": "x", "through": "x",
             "text": "t", "ids": [(i * 7 + j) % N_VOCAB for j in range(n_ctx)]} for i in range(n)]


def write_rows(path: pathlib.Path, n_rows: int, kld: float, same_top1: float, nll: float = 1.0,
               dlogp: float = 0.001) -> None:
    """rows.tsv as PplGate writes it: chunk pos target nll nll_base kld same_top top top_base dlogp."""
    lines = []
    n_same = int(round(same_top1 * n_rows / 100.0))
    for i in range(n_rows):
        lines.append("\t".join([str(i // 8), str(32 + (i % 8)), str(100 + i), f"{nll:.6f}", f"{nll:.6f}",
                                f"{kld:.7g}", "1" if i < n_same else "0", "42", "42", f"{dlogp:.6f}"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class BaseLayout(unittest.TestCase):
    def test_header_and_tokens_where_the_engine_reads_them(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d) / "b.kld"
            size = QG.write_base(base, chunks(), N_CTX, N_VOCAB)
            raw = base.read_bytes()
            self.assertEqual(raw[:8], b"_logits_", "the magic the engine checks first")
            n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", raw[8:20])
            self.assertEqual((n_ctx, n_vocab, n_chunk), (N_CTX, N_VOCAB, 3))
            self.assertEqual(size, 20 + 3 * N_CTX * 4 + 3 * (N_CTX - 1 - N_CTX // 2) * ROW_U16 * 2)
            self.assertEqual(len(raw), size, "the file is sized to the rows as well as the tokens")
            # chunk 2's tokens live at 20 + 2 * n_ctx * 4, which is where begin() reads them
            want = chunks()[2]["ids"]
            got = list(struct.unpack(f"<{N_CTX}i", raw[20 + 2 * N_CTX * 4: 20 + 3 * N_CTX * 4]))
            self.assertEqual(got, want)
            # row 0 of chunk 1 sits past every chunk's tokens: a written row cannot land on another chunk's tokens
            off = 20 + 3 * N_CTX * 4 + 1 * (N_CTX - 1 - N_CTX // 2) * ROW_U16 * 2
            self.assertEqual(raw[off:off + 8], b"\x00" * 8, "rows start as a hole, not as data")

    def test_a_scored_row_never_overruns_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d) / "b.kld"
            size = QG.write_base(base, chunks(), N_CTX, N_VOCAB)
            rows = N_CTX - 1 - N_CTX // 2
            last = 20 + 3 * N_CTX * 4 + (2 * rows + rows - 1) * ROW_U16 * 2
            self.assertEqual(last + ROW_U16 * 2, size, "the last row of the last chunk must end exactly at EOF")


class Parsing(unittest.TestCase):
    def test_summary_over_synthetic_rows(self):
        with tempfile.TemporaryDirectory() as d:
            rows = pathlib.Path(d) / "rows.tsv"
            write_rows(rows, 40, kld=0.01, same_top1=97.5, nll=1.2, dlogp=0.02)
            s = QG.summarize(rows)
            self.assertEqual(s["rows"], 40)
            self.assertAlmostEqual(s["PPL"], math.exp(1.2), places=6)
            self.assertAlmostEqual(s["KLD mean"], 0.01, places=6)
            self.assertAlmostEqual(s["same top-1"], 97.5, places=6)
            self.assertAlmostEqual(s["dlogp max"], 0.02, places=6)

    def test_summary_of_nothing_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(QG.summarize(pathlib.Path(d) / "absent.tsv"))
            empty = pathlib.Path(d) / "rows.tsv"
            empty.write_text("not a row\n\n", encoding="utf-8")
            self.assertIsNone(QG.summarize(empty))

    def test_short_lines_are_skipped_not_crashed_on(self):
        with tempfile.TemporaryDirectory() as d:
            rows = pathlib.Path(d) / "rows.tsv"
            rows.write_text("1\t2\t3\n" + "\t".join(["0"] * 10) + "\n", encoding="utf-8")
            self.assertEqual(QG.summarize(rows)["rows"], 1)


class Verdict(unittest.TestCase):
    def arm(self, root: pathlib.Path, name: str, **kw):
        write_rows(root / name / "rows.tsv", 40, **kw)

    def good(self, root):  # the numbers a passing flag-on run would show
        self.arm(root, "check", kld=0.00007, same_top1=100.0)
        self.arm(root, "gate", kld=0.0002, same_top1=99.5)

    def test_a_run_inside_every_bound_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.good(root)
            self.assertEqual(QG.verdict(root), 0)

    def test_self_check_drift_voids_the_run(self):
        # 97.5% of 40 rows is 39 rows: below the self-check's 100% without rounding ambiguity
        for kw in ({"kld": 0.5, "same_top1": 100.0}, {"kld": 0.00007, "same_top1": 97.5}):
            with self.subTest(**kw), tempfile.TemporaryDirectory() as d:
                root = pathlib.Path(d)
                self.good(root)
                self.arm(root, "check", **kw)
                self.assertEqual(QG.verdict(root), 1)

    def test_each_gate_bound_fails_on_its_own(self):
        cases = [{"kld": QG.BOUND_KLD_MEAN * 3, "same_top1": 99.5},        # KLD mean
                 {"kld": 0.0002, "same_top1": QG.BOUND_SAME_TOP1 - 1.0},   # same top-1
                 {"kld": 0.0002, "same_top1": 99.5, "dlogp": 9.0}]         # dlogp max
        for kw in cases:
            with self.subTest(**kw), tempfile.TemporaryDirectory() as d:
                root = pathlib.Path(d)
                self.good(root)
                self.arm(root, "gate", **kw)
                self.assertEqual(QG.verdict(root), 1)

    def test_kld_p99_fails_when_only_the_tail_moves(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.good(root)
            rows = root / "gate" / "rows.tsv"
            lines = rows.read_text(encoding="utf-8").rstrip("\n").split("\n")
            lines[-1] = "\t".join(["5.0" if i in (5, 9) else v for i, v in enumerate(lines[-1].split("\t"))])
            rows.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertGreater(QG.summarize(rows)["KLD p99"], QG.BOUND_KLD_P99)
            self.assertEqual(QG.verdict(root), 1)

    def test_a_missing_arm_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.arm(root, "check", kld=0.00007, same_top1=100.0)
            self.assertEqual(QG.verdict(root), 1)          # no gate arm yet
            self.assertEqual(QG.verdict(root / "nowhere"), 1)


class Fixture(unittest.TestCase):
    def test_the_committed_fixture_is_internally_consistent(self):
        """The fixture is the scored set, so its ids are the contract: exact bucket length, in-vocabulary, no
        duplicates. Text round-trip needs the pack, so that half is the fixture tool's job, not this one's."""
        path = QG.FIXTURE
        if not path.is_file():
            self.skipTest(f"{path} is not in the tree")
        seen, buckets = set(), {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            self.assertEqual(len(e["ids"]), e["bucket"], f"{e['id']}: ids must fill the bucket exactly")
            self.assertGreaterEqual(min(e["ids"]), 0, e["id"])
            self.assertNotIn(tuple(e["ids"]), seen, f"{e['id']} repeats another chunk")
            seen.add(tuple(e["ids"]))
            buckets[e["bucket"]] = buckets.get(e["bucket"], 0) + 1
        self.assertGreaterEqual(len(buckets), 2, "the fixture should cover more than one length bucket")
        self.assertTrue(all(c >= 4 for c in buckets.values()), "a bucket with a couple of chunks grades nothing")


class ArgHandling(unittest.TestCase):
    def test_set_args_replaces_and_appends(self):
        base = ["--model", "m", "--spec", "8", "--gpu", "0,1"]
        self.assertEqual(QG.set_args(base, ["--spec", "0"]), ["--model", "m", "--spec", "0", "--gpu", "0,1"])
        self.assertEqual(QG.set_args(base, ["--kv", "int8"]), base + ["--kv", "int8"])
        self.assertEqual(QG.set_args(base, ["--gpu"]), ["--model", "m", "--spec", "8", "--gpu"])

    def _arm(self, args, bucket=512, arm="check"):
        """Drive run_arm as far as the point where it would start the engine. Returns (args_it_settled_on,
        exit_message): the second is None when it got that far, or the SystemExit text when it refused first."""
        ns = argparse.Namespace(arm=arm, t=3, progress=0, engine_bin="/nonexistent/basalt", flag="peer_partial_merge=1")
        seen = {}
        real_boot = QG.boot

        def fake_boot(bin_, got, log):
            seen["args"] = got
            raise SystemExit(0)          # stop before anything would be executed

        QG.boot = fake_boot
        msg = None
        try:
            with tempfile.TemporaryDirectory() as d:
                out = pathlib.Path(d) / "check"
                base = pathlib.Path(d) / "base.kld"
                QG.write_base(base, chunks(1, bucket), bucket, N_VOCAB)
                try:
                    QG.run_arm(ns, bucket, chunks(1, bucket), bucket, N_VOCAB, base, out, list(args))
                except SystemExit as e:
                    if e.code != 0:
                        msg = str(e.code)
        finally:
            QG.boot = real_boot
        return seen.get("args"), msg

    def test_the_operators_max_context_is_kept(self):
        """The configuration under test is the operator's; a chunk-sized default must not quietly replace it."""
        got, msg = self._arm(["--model", "m", "--max-context", "262144"])
        self.assertIsNone(msg)
        self.assertEqual(got[got.index("--max-context") + 1], "262144")

    def test_max_context_is_filled_in_only_when_absent(self):
        got, msg = self._arm(["--model", "m"])
        self.assertIsNone(msg)
        self.assertEqual(got[got.index("--max-context") + 1], "1024")      # 512 + 512

    def test_a_max_context_that_cannot_hold_the_chunk_is_refused(self):
        got, msg = self._arm(["--model", "m", "--max-context", "1024"], bucket=4096)
        self.assertIsNone(got, "it must refuse before trying to start the engine")
        self.assertIn("--max-context 1024", msg)

    def test_the_gate_arm_keeps_the_operators_exp_flags(self):
        """`--exp` is repeatable, so the flag under test is appended: an operator's experimental flags must survive
        into both arms, or the arms differ by more than the thing being graded."""
        got, msg = self._arm(["--model", "m", "--exp", "decode_profile=1", "--exp", "draft_reseed=1"], arm="gate")
        self.assertIsNone(msg)
        exps = [got[i + 1] for i, w in enumerate(got) if w == "--exp"]
        self.assertEqual(exps, ["decode_profile=1", "draft_reseed=1", "peer_partial_merge=1"])

    def test_the_gate_arm_refuses_a_flag_the_operator_already_set(self):
        got, msg = self._arm(["--model", "m", "--exp", "peer_partial_merge=0"], arm="gate")
        self.assertIsNone(got, "it must refuse before trying to start the engine")
        self.assertIn("peer_partial_merge", msg)

    def test_the_gate_arm_spots_a_key_inside_a_comma_list(self):
        got, msg = self._arm(["--model", "m", "--exp", "decode_profile=1,peer_partial_merge=1"], arm="gate")
        self.assertIsNone(got)
        self.assertIn("peer_partial_merge", msg)

    def test_the_untouched_arms_never_set_the_flag(self):
        for arm in ("check",):
            got, msg = self._arm(["--model", "m"], arm=arm)
            self.assertIsNone(msg)
            self.assertNotIn("peer_partial_merge", " ".join(got))

    def test_unknown_flags_reach_the_engine_and_known_ones_are_replaced(self):
        """The docs promise that flags the tool does not know go to the engine, and an engine flag given here
        replaces the tool's default for it - that is how both arms are run on the production configuration."""
        captured = {}
        real_boot, real_mf = QG.boot, QG.ModelFile

        class StubModelFile:                      # the unit test must not open the 105 GiB pack
            def __init__(self, path):
                pass

            def read(self, name):
                return json.dumps({f"t{i}": i for i in range(64)})

        def fake_boot(bin_, got, log):
            captured["args"] = got
            raise SystemExit(0)

        QG.boot, QG.ModelFile = fake_boot, StubModelFile
        saved = sys.argv
        with tempfile.TemporaryDirectory() as d:
            QG.write_base(pathlib.Path(d) / "base-512.kld", chunks(1, 512), 512, 64)
            try:
                sys.argv = ["quality_gate.py", "--pack", "p", "--bucket", "512", "--out", d, "--arm", "check",
                            "--model", "/m/pack.basalt", "--gpu", "2,3", "--tp-size", "2", "--spec", "4"]
                with self.assertRaises(SystemExit) as cm:
                    QG.main()
            finally:
                sys.argv, QG.boot, QG.ModelFile = saved, real_boot, real_mf
        self.assertEqual(cm.exception.code, 0, "the stubbed boot exits 0; anything else is a real refusal")
        a = captured["args"]
        self.assertEqual(a[a.index("--model") + 1], "/m/pack.basalt", "the tool's default model must not win")
        self.assertEqual(a[a.index("--gpu") + 1], "2,3", "the operator's cards must not be overridden")
        self.assertEqual(a[a.index("--spec") + 1], "4", "an engine flag given here replaces the tool's default")
        self.assertEqual(a[a.index("--tp-size") + 1], "2", "an unknown flag must still reach the engine")
        self.assertEqual(a[a.index("--adapt-swaps") + 1], "0", "the determinism flags stay forced on")
        self.assertNotIn("peer_partial_merge", " ".join(a), "the check arm must not carry the flag")

    def test_the_gate_arm_is_the_only_one_that_adds_the_flag(self):
        got, msg = self._arm(["--model", "m"], arm="gate")
        self.assertIsNone(msg)
        self.assertIn("peer_partial_merge=1", got)

    def test_help_and_the_required_args_are_enforced(self):
        for argv, code in ((["--help"], 0), ([], 2), (["--pack", "x"], 2)):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as cm:
                sys.argv = ["quality_gate.py", *argv]
                try:
                    QG.main()
                except SystemExit as e:
                    raise e
                self.fail("main() accepted an incomplete command line")
            self.assertEqual(cm.exception.code, code)


if __name__ == "__main__":
    unittest.main(verbosity=2)

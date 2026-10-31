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


# What a boot says about where its experts landed, in the engine's own words. `same` and `moved` are two boots whose
# partner card filled a different number of ranked pairs into a different room - the confound the gate must refuse.
BOOT_LINES = {"same": [
    "basalt generate: tp2: CUDA1 holds 5429 of the 12288 ranked pairs of its stage's layers "
    "(13.98 GiB of 13.98 GiB room); its lead card holds 1358\n",
    "basalt prefill: peer fp4 crossover 8192 tokens: 621 MiB (5429 experts held, 584 band); "
    "RAM rows on, 1/8 of 4583 band experts of layers 24-48\n",
    "basalt generate: experts live in VRAM 37.3 GiB (59%), RAM 26.0 GiB (41%)\n",
], "moved": [
    "basalt generate: tp2: CUDA1 holds 5384 of the 12288 ranked pairs of its stage's layers "
    "(13.86 GiB of 13.86 GiB room); its lead card holds 1358\n",
    "basalt prefill: peer fp4 crossover 8192 tokens: 154 MiB (5384 experts held, 591 band); "
    "RAM rows on, 1/8 of 4628 band experts of layers 24-48\n",
    "basalt generate: experts live in VRAM 37.1 GiB (59%), RAM 26.2 GiB (41%)\n",
]}


def write_log(path: pathlib.Path, lines: list[str] | None = None) -> None:
    """engine.log as `boot()` leaves it, holding the lines the verdict reads the placement from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(BOOT_LINES["same"] if lines is None else lines), encoding="utf-8")


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
        # a real run always leaves a log; without one the verdict must refuse the run, not pass it
        for arm in ("emit", "check", "gate"):
            write_log(root / arm / "engine.log")

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


class Placement(unittest.TestCase):
    """The arms must run on one placement. `--exp peer_partial_merge=1` is read inside `Prefill::peer_bytes`, which
    sizes the reserve the partner's expert tier is charged, so a flag-on boot fills the partner's tier into a
    smaller room and holds fewer ranked pairs - and which card computes a row decides whether it crosses as BF16
    bits or stays FP32. The gate must refuse that comparison rather than grade it."""

    BOOT = BOOT_LINES

    def arms(self, root: pathlib.Path, gate_lines):
        self.good(root)
        for arm, lines in (("emit", self.BOOT["same"]), ("check", self.BOOT["same"]), ("gate", gate_lines)):
            write_log(root / arm / "engine.log", lines)

    def good(self, root):
        write_rows(root / "check" / "rows.tsv", 40, kld=0.00007, same_top1=100.0)
        write_rows(root / "gate" / "rows.tsv", 40, kld=0.0002, same_top1=99.5)

    def test_facts_are_read_from_the_boot_its_own_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = pathlib.Path(d) / "engine.log"
            write_log(log)
            f = QG.placement_facts(log)
            self.assertEqual(f["partner held"], "(5429 experts held, 584 band)")
            self.assertIn("CUDA1 holds 5429", f["partner tier"])
            self.assertIsNone(QG.placement_facts(pathlib.Path(d) / "absent.log"),
                              "no log is not the same fact as a log that says nothing about placement")

    def test_identical_placements_do_not_void_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.arms(root, self.BOOT["same"])
            self.assertEqual(QG.placement_mismatch(root), [])
            self.assertEqual(QG.verdict(root), 0)

    def test_a_moved_tier_voids_the_run_however_good_the_numbers_are(self):
        """The numbers the gate arm printed were measured on a different tier, so a PASS would be an artefact."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.arms(root, self.BOOT["moved"])
            bad = QG.placement_mismatch(root)
            self.assertTrue(bad, "5429 held against 5384 must be caught")
            self.assertIn("5384 experts held", "\n".join(bad))
            self.assertEqual(QG.verdict(root), 1)

    def test_the_refusal_names_the_lever_that_pins_the_placement(self):
        """13.98 GiB against 13.86 GiB is ~123 MiB of partner VRAM: the verdict must hand the operator the figure
        and the arm to charge it on, or the re-gate is a guess. It must name a lever that reaches a tp2 PARTNER
        card's tier - `--peer-reserve-mib` prices an expert tier, which a tp2 boot has none of, so pointing at it
        is what made one re-gate run change nothing."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.arms(root, self.BOOT["moved"])
            bad = "\n".join(QG.placement_mismatch(root))
            self.assertIn("--base-args", bad)
            self.assertIn("--exp peer_charge_partial=1", bad)
            self.assertIn("123 MiB", bad)
            self.assertIn("--peer-reserve-mib moves no tp2 partner's tier", bad)

    def test_the_refusal_attributes_each_room_to_the_right_arm(self):
        """The tighter card belongs to whichever arm runs the flag that costs it bytes. Naming the two rooms
        against the wrong arms would send the operator to raise the reserve on the wrong side."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.arms(root, self.BOOT["moved"])          # the gate arm holds the tighter tier
            bad = "\n".join(QG.placement_mismatch(root))
            self.assertIn("13.86 GiB on the gate arm", bad)
            self.assertIn("13.98 GiB on the check arm", bad)
            self.assertIn("reference (emit, check)", bad)
            # the arm being charged must be shrunk INTO the tighter room, not left in its own wider one
            self.assertIn("fill them into 13.86 GiB", bad)

    def test_a_flag_that_frees_room_points_at_the_gate_arm(self):
        """The mirror image: when the tested flag leaves the GATE arm the wider card, the reference arms are
        already right and it is the gate arm that must be charged."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.arms(root, self.BOOT["same"])           # gate into 13.98 ...
            write_log(root / "check" / "engine.log", self.BOOT["moved"])   # ... check into 13.86
            bad = "\n".join(QG.placement_mismatch(root))
            self.assertIn("--gate-args", bad)
            self.assertNotIn("--base-args", bad)
            self.assertIn("123 MiB", bad)
            self.assertIn("fill them into 13.86 GiB", bad)

    def test_the_reference_arm_moving_also_voids(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.arms(root, self.BOOT["same"])
            write_log(root / "emit" / "engine.log", self.BOOT["moved"])
            self.assertTrue(any("emit arm" in m for m in QG.placement_mismatch(root)))

    def test_an_arm_that_left_no_log_voids_the_run(self):
        """An untraceable boot has not been shown to hold the same experts as any other. Reading its absence as
        agreement is how a run whose arms held 5429 and 5384 ranked pairs was graded at all."""
        for missing in ("emit", "check", "gate"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as d:
                root = pathlib.Path(d)
                self.arms(root, self.BOOT["same"])
                (root / missing / "engine.log").unlink()
                bad = "\n".join(QG.placement_mismatch(root))
                self.assertIn("no engine.log", bad)
                self.assertIn(f"the {missing} arm", bad)
                self.assertEqual(QG.verdict(root), 1)

    def test_a_log_that_says_nothing_about_placement_voids_the_run(self):
        """A log that exists but holds none of the placement lines is a run whose tier was never read - not a run
        whose arms agreed, and not a reason to print nothing."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.arms(root, self.BOOT["same"])
            (root / "gate" / "engine.log").write_text("engine /usr/local/bin/basalt\nargs --model m\n",
                                                      encoding="utf-8")
            bad = "\n".join(QG.placement_mismatch(root))
            self.assertIn("holds none of the lines", bad)
            self.assertEqual(QG.verdict(root), 1)

    def test_a_run_with_no_logs_at_all_is_refused_not_quietly_passed(self):
        """An old run directory, or one whose boots died before the tier was filled, must not reach the bounds."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.good(root)          # rows.tsv for the two scored arms, and no engine.log anywhere
            bad = QG.placement_mismatch(root)
            self.assertEqual(len(bad), 3, "one line naming each arm whose log is missing")
            self.assertTrue(all("no engine.log" in m for m in bad))
            self.assertEqual(QG.verdict(root), 1)


class ArgHandling(unittest.TestCase):
    def test_set_args_replaces_and_appends(self):
        base = ["--model", "m", "--spec", "8", "--gpu", "0,1"]
        self.assertEqual(QG.set_args(base, ["--spec", "0"]), ["--model", "m", "--spec", "0", "--gpu", "0,1"])
        self.assertEqual(QG.set_args(base, ["--kv", "int8"]), base + ["--kv", "int8"])
        self.assertEqual(QG.set_args(base, ["--gpu"]), ["--model", "m", "--spec", "8", "--gpu"])

    def _arm(self, args, bucket=512, arm="check"):
        """Drive run_arm as far as the point where it would start the engine. Returns (args_it_settled_on,
        exit_message): the second is None when it got that far, or the SystemExit text when it refused first."""
        ns = argparse.Namespace(arm=arm, t=3, progress=0, engine_bin="/nonexistent/basalt",
                                flag="peer_partial_merge=1", gate_args="", base_args="")
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

    def test_the_reference_arms_are_charged_the_bytes_the_flag_costs(self):
        """The partial's buffers live on the partner card, so the gate arm fills its expert tier into a narrower
        room and holds fewer experts - which changes which rows cross as BF16. `--base-args '--exp
        peer_charge_partial=1'` charges the reference arms those bytes through the engine's own pricing walk; the
        key must reach the engine on emit and check and not on the gate arm."""
        for arm, wants in (("check", True), ("emit", True), ("gate", False)):
            with self.subTest(arm=arm):
                ns = argparse.Namespace(arm=arm, t=3, progress=0, engine_bin="/nonexistent/basalt",
                                        flag="peer_partial_merge=1", gate_args="",
                                        base_args="--exp peer_charge_partial=1")
                got = {}
                real_boot = QG.boot
                QG.boot = lambda b, a_, log: (got.setdefault("args", a_), (_ for _ in ()).throw(SystemExit(0)))[0]
                try:
                    with tempfile.TemporaryDirectory() as d:
                        base = pathlib.Path(d) / f"base-{512}.kld"
                        QG.write_base(base, chunks(1, 512), 512, N_VOCAB)
                        try:
                            QG.run_arm(ns, 512, chunks(1, 512), 512, N_VOCAB, base,
                                       pathlib.Path(d) / arm, ["--model", "m"])
                        except SystemExit as e:
                            self.assertEqual(e.code, 0)
                finally:
                    QG.boot = real_boot
                exps = [v for i, w in enumerate(got["args"]) if w == "--exp" for v in got["args"][i + 1].split(",")]
                if wants:
                    self.assertIn("peer_charge_partial=1", exps, f"{arm} must carry the pinned charge")
                else:
                    self.assertNotIn("peer_charge_partial=1", exps, f"{arm} must not carry it")

    def test_an_arm_charge_does_not_evict_the_operators_exp_flags(self):
        """`--exp` is repeatable and takes a comma list, so an arm's own key must be ADDED beside the operator's
        rather than replace the first occurrence: an arm that quietly lost `decode_profile` is a second difference
        between the arms, and the verdict would grade it as the flag under test."""
        for arm, extra in (("check", "--exp peer_charge_partial=1"), ("gate", "--exp qsa_peer_fold_us=0")):
            with self.subTest(arm=arm):
                ns = argparse.Namespace(arm=arm, t=3, progress=0, engine_bin="/nonexistent/basalt",
                                        flag="peer_partial_merge=1", gate_args="", base_args="")
                setattr(ns, "base_args" if arm != "gate" else "gate_args", extra)
                seen = {}
                real_boot = QG.boot
                QG.boot = lambda b, a_, log: (seen.setdefault("args", a_), (_ for _ in ()).throw(SystemExit(0)))[0]
                try:
                    with tempfile.TemporaryDirectory() as d:
                        base = pathlib.Path(d) / f"base-{512}.kld"
                        QG.write_base(base, chunks(1, 512), 512, N_VOCAB)
                        try:
                            QG.run_arm(ns, 512, chunks(1, 512), 512, N_VOCAB, base, pathlib.Path(d) / arm,
                                       ["--model", "m", "--exp", "decode_profile=1,draft_reseed=1"])
                        except SystemExit as e:
                            self.assertEqual(e.code, 0)
                finally:
                    QG.boot = real_boot
                exps = [v for i, w in enumerate(seen["args"]) if w == "--exp" for v in seen["args"][i + 1].split(",")]
                self.assertIn("decode_profile=1", exps)
                self.assertIn("draft_reseed=1", exps)
                self.assertIn(extra.split()[1], exps)

    def test_an_arm_charge_refuses_a_key_the_operator_already_set(self):
        """Two arms, one key, two values: the engine would keep whichever `--exp` it parsed last, so the tool
        refuses instead of letting argument order pick which arm ran which."""
        ns = argparse.Namespace(arm="check", t=3, progress=0, engine_bin="/nonexistent/basalt",
                                flag="peer_partial_merge=1", gate_args="", base_args="--exp peer_charge_partial=1")
        real_boot = QG.boot
        QG.boot = lambda b, a_, log: (_ for _ in ()).throw(AssertionError("the engine must not be reached"))
        try:
            with tempfile.TemporaryDirectory() as d:
                base = pathlib.Path(d) / f"base-{512}.kld"
                QG.write_base(base, chunks(1, 512), 512, N_VOCAB)
                with self.assertRaises(SystemExit) as cm:
                    QG.run_arm(ns, 512, chunks(1, 512), 512, N_VOCAB, base, pathlib.Path(d) / "check",
                               ["--model", "m", "--exp", "peer_charge_partial=0"])
        finally:
            QG.boot = real_boot
        self.assertIn("peer_charge_partial", str(cm.exception.code))

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

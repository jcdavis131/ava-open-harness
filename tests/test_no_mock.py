# Solo personal project, no connection to employer, built with public/free-tier only
"""Anti-mock guard (HARNESS_SPEC 'Anti-mock Guard').

Enforces the invariant this repo exists for: no fabricated numbers presented as
measurements. Three checks, matching the spec:

1. Dynamic — jspace tests run with seeds 1 and 2 (mock, ckpt none) produce DIFFERENT
   measured dicts. Static fabricated constants can't vary by seed, so this catches them
   without a brittle source grep of legitimate seed-noise base values.
2. Report grep — a full mock run's report JSON does not contain any forbidden literal as
   an exact serialized value (mock noise guarantees non-exactness; a static value would
   round-trip verbatim).
3. Real-mode honesty — every eval whose real path is unwired returns measured=None,
   pass=False, and an error (never an invented number); and run_harness(mode='real')
   with no real model produces a structured honest-failure report, not fabricated passes.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.common import MockModel, MockTokenizer  # noqa: E402
from harness.evals import jspace_tests as J  # noqa: E402
from harness.runner import run_harness  # noqa: E402

FORBIDDEN = [
    "0.82",
    "0.22",
    "0.064",
    "0.88",
    "0.75",
    "0.91",
    "0.94",
    "0.92",
    "5.2",
    "4.5",
    "0.983",
    "0.967",
]
JSPACE = [
    "spider_ant",
    "france_china",
    "soccer_rugby",
    "spanish_french",
    "safety_blackmail",
]


def _run_eval(name, seed):
    fn = getattr(J, name)
    return fn(MockModel(seed=seed), MockTokenizer(), "cpu")


class TestDynamicVariation:
    @pytest.mark.parametrize("name", JSPACE)
    def test_measured_differs_across_seeds(self, name):
        m1 = _run_eval(name, 1).get("measured")
        m2 = _run_eval(name, 2).get("measured")
        # A static fabricated measured dict would be identical across seeds.
        assert m1 != m2, (
            f"{name} measured did not vary with seed → looks static/fabricated"
        )


class TestReportGrep:
    def test_mock_report_has_no_exact_forbidden_literals(self, tmp_path):
        res = run_harness(eval_names=JSPACE, mode="mock")
        blob = json.dumps(res)
        # Exact-token check: a fabricated static value round-trips verbatim; seed-noise
        # values serialize with long float tails and won't match these short literals.
        for lit in FORBIDDEN:
            assert f": {lit}," not in blob and f": {lit}}}" not in blob, (
                f"forbidden literal {lit} appears verbatim in mock report"
            )


class TestStableSeedReproducibility:
    """A "seeded" mock draw must be identical across separate interpreter
    processes given the same --seed. Regression for a real bug: builtin
    hash() is salted per-process by PYTHONHASHSEED (default since Python 3.3),
    so any mock path that seeded random.seed(hash(name) + ...) silently
    produced a DIFFERENT "reproducible" measurement every run/process — the
    same class of fabricated-looking non-determinism this file's anti-mock
    guard exists to catch, just triggered by process restart instead of code."""

    def _openwiki_recall_mass(self, hashseed: str, tmp_path: Path) -> float:
        wiki = tmp_path / "openwiki"
        wiki.mkdir(exist_ok=True)
        (wiki / "a.md").write_text("alpha page content", encoding="utf-8")
        (wiki / "b.md").write_text("beta page content", encoding="utf-8")
        code = (
            "from harness.common import MockModel, MockTokenizer\n"
            "from harness.evals.openwiki_knowledge import openwiki_knowledge\n"
            f"r = openwiki_knowledge(MockModel(seed=5), MockTokenizer(), 'cpu', wiki_path={str(wiki)!r})\n"
            "print(r['measured']['recall_mass'])\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            env={
                "PYTHONHASHSEED": hashseed,
                "PATH": __import__("os").environ.get("PATH", ""),
            },
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip())

    def test_openwiki_mock_recall_mass_stable_across_hashseeds(self, tmp_path):
        m1 = self._openwiki_recall_mass("1", tmp_path)
        m2 = self._openwiki_recall_mass("2", tmp_path)
        assert m1 == m2, (
            "openwiki_knowledge mock recall_mass changed across PYTHONHASHSEED "
            f"values ({m1} vs {m2}) — a 'seeded' mock draw must be "
            "process-stable, not just seed-stable"
        )


class TestWikiScanOrderStability:
    """Regression: scan_wiki() must select files by sorted path, not by
    filesystem-enumeration order.

    pathlib.rglob() is built on os.scandir(), which yields directory entries
    in arbitrary, filesystem-dependent order (verified: NOT lexicographic).
    scan_wiki() truncates to the first 50 matches per candidate dir, and
    openwiki_knowledge() further truncates to the first 20/5. If that
    truncation happens on an unsorted list, the very files backing a
    "reproducible" recall_mass depend on directory-enumeration order — the
    same wiki corpus can score differently across machines/filesystems or
    across a fresh checkout, even with the seed fully pinned. Sorting first
    makes the selected subset a pure function of the corpus content."""

    def test_scan_wiki_returns_sorted_paths(self, tmp_path):
        from harness.evals.openwiki_knowledge import scan_wiki

        wiki = tmp_path / "openwiki"
        wiki.mkdir()
        # Create filenames in a shuffled (non-lexicographic) order so any
        # accidental reliance on creation/insertion order would be caught.
        names = [f"page_{i:03d}.md" for i in range(60)]
        for name in sorted(names, key=lambda n: (hash(n) * 2654435761) & 0xFFFF):
            (wiki / name).write_text("content", encoding="utf-8")

        found = scan_wiki(str(wiki))
        found_names = [p.name for p in found]

        assert found_names == sorted(found_names), (
            "scan_wiki() must return paths in sorted order so truncation to "
            "the first-N cap is deterministic and content-derived, not "
            "filesystem-enumeration-derived"
        )
        # The 50-cap must keep the lexicographically first 50 names, not an
        # arbitrary filesystem-order-dependent subset.
        assert found_names == sorted(names)[:50]


class TestRealModeHonesty:
    @pytest.mark.parametrize("name", JSPACE)
    def test_unwired_real_paths_fail_honestly(self, name, monkeypatch):
        # Real paths now delegate to the factory repo when importable; simulate
        # a machine WITHOUT the factory — the real path must fail honestly with
        # a structured record, never simulate a measurement.
        monkeypatch.setenv("AVA_FACTORY_ROOT", "/nonexistent-factory-root")
        res = getattr(J, name)(
            object(), MockTokenizer(), "cpu"
        )  # non-MockModel → real path
        assert res["pass"] is False
        assert res.get("measured") is None
        assert res.get("error")

    def test_run_harness_real_without_model_is_structured_failure(self):
        res = run_harness(eval_names=JSPACE, mode="real")
        assert res["meta"].get("real_load_failed") is True
        assert res["meta"]["passed"] == 0
        assert all(
            e["pass"] is False and e.get("measured") is None
            for e in res["evals"].values()
        )

    def test_run_harness_real_does_not_raise(self):
        # Regression: real-mode-with-mock must be a report, not an exception a caller
        # could swallow and then fabricate around.
        res = run_harness(eval_names=["spider_ant"], mode="real")
        assert isinstance(res, dict) and "evals" in res

"""Phase 14: the quickstart must work from a clean clone, and must stay exact.

A quickstart that fails is worse than none, because it signals the rest is unreliable too.
These tests run the same path a stranger runs, against the same committed fixture and the
same committed golden file.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "corpus" / "bronze_quickstart.jsonl.gz"
GOLDEN = REPO_ROOT / "tests" / "golden" / "quickstart_evaluation.json"
SCRIPT = REPO_ROOT / "scripts" / "14_quickstart.py"

#: Small enough that git and the 512 KB pre-commit hook both stay happy. The uncompressed
#: corpus is 232 MB, which is why the fixture is compressed rather than sliced thinner.
MAX_FIXTURE_BYTES = 512_000


def test_fixture_and_golden_are_committed() -> None:
    """Both must exist, or the quickstart cannot run on a clean clone."""
    assert FIXTURE.is_file(), "the quickstart fixture is not committed"
    assert GOLDEN.is_file(), "the quickstart golden file is not committed"


def test_fixture_stays_under_the_large_file_threshold() -> None:
    """The pre-commit hook rejects anything larger, and it should."""
    assert FIXTURE.stat().st_size < MAX_FIXTURE_BYTES


def test_fixture_covers_every_narrative_family() -> None:
    """A fixture missing a family would leave a template family unexercised.

    The full corpus has eleven families and the stratified draw takes twenty from each,
    so a shortfall means the draw silently changed.
    """
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == 220
    families = {r["generator"]["family"] for r in records}
    assert len(families) == 11, f"expected 11 families, got {sorted(families)}"
    assert {r["tier"] for r in records} == {"bronze"}


def test_fixture_carries_no_real_identifiers() -> None:
    """Invariant 8: AMLworld is synthetic, and nothing else may enter a fixture."""
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        blob = handle.read()
    assert "@" not in blob.replace("@package", ""), "an email address reached the fixture"
    for token in ("BEGIN RSA", "BEGIN PRIVATE", "sk-ant-", "hf_"):
        assert token not in blob


@pytest.mark.slow
def test_quickstart_runs_and_matches_the_golden_file_exactly() -> None:
    """The documented quickstart, run as documented.

    Bronze renders deterministically from the fact record, so this is an exact assertion
    rather than a tolerance. A difference here is a bug in the fact layer or the evaluation
    harness -- the two things invariant 1 exists to protect.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"quickstart failed:\n{proc.stdout}\n{proc.stderr}"
    assert "QUICKSTART OK" in proc.stdout


def test_golden_file_records_the_load_bearing_assertions() -> None:
    """A golden file that omits these would pass while the harness was broken."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    faith = golden["faithfulness"]

    # Bronze is faithful by construction; anything else means the renderer or the checker
    # has drifted.
    assert faith["zero_hallucination_rate"] == 1.0
    assert faith["hallucination_rate"] == 0.0
    assert faith["critical_error_rate"] == 0.0

    # The one that makes the rest mean anything: a perfect score over an empty claim set
    # is what a broken extractor produces.
    assert faith["n_narratives_with_no_claims"] == 0
    assert faith["n_claims"] > 0

    # H9 is a real measured defect and must not silently become zero.
    assert golden["taxonomy_rate_by_class"]["H9"] > 0

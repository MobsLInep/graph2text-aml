#!/usr/bin/env python
"""The quickstart: reproduce one published result from a clean clone, in under a minute.

Scores the committed 220-record Bronze fixture with the **real** Phase 10 evaluation
harness -- the same `scripts/10_evaluate.py` the CI gate runs, not a reimplementation --
and asserts the result against a committed golden file.

Why a fixture rather than the corpus: `bronze.jsonl` is 232 MB and is gitignored, and the
raw AMLworld release is a 20 GB manual Kaggle download. A quickstart that begins with
either is not a quickstart. The fixture is a stratified slice -- 20 records from each of
the eleven narrative families -- so it exercises every template family and every fact
family the full corpus does.

**Why the assertion is exact:** Bronze is rendered deterministically from the fact record,
so the fixture's scores are reproducible to the bit on any machine. A tolerance here would
hide the class of bug this check exists to catch. The stochastic stages, and their real
tolerances, are in `docs/REPRODUCTION.md` s6.

The numbers differ from the full-corpus numbers in `RESULTS.md` and that is expected: this
is a stratified 220-record slice, so the families are evenly weighted rather than weighted
as they occur. Fact Coverage reads 0.8359 here against 0.8595 over all 15,707 records.

Usage:
    uv run python scripts/14_quickstart.py
    uv run python scripts/14_quickstart.py --keep      # leave the staged tree in place
    uv run python scripts/14_quickstart.py --regenerate-golden
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "corpus" / "bronze_quickstart.jsonl.gz"
GOLDEN = REPO_ROOT / "tests" / "golden" / "quickstart_evaluation.json"
EVALUATE = REPO_ROOT / "scripts" / "10_evaluate.py"

#: The fields asserted. Deliberately not the whole report: `run_id`, timings and the
#: absolute paths in `metadata` differ every run and asserting them would make the check
#: fail for reasons that are not about correctness.
ASSERTED = (
    "n_cases",
    "zero_hallucination_rate",
    "fact_precision",
    "hallucination_rate",
    "unverifiable_rate",
    "fact_coverage",
    "fact_f1",
    "numeric_accuracy",
    "typology_accuracy",
    "ordering_accuracy",
    "critical_error_rate",
    "n_claims",
    "n_narratives_with_no_claims",
)


def stage_fixture(dest: Path) -> Path:
    """Decompress the fixture into a corpus tree the evaluator can read.

    Args:
        dest: Directory to build the tree under.

    Returns:
        The path of the staged ``bronze.jsonl``.

    Raises:
        FileNotFoundError: If the committed fixture is missing.
    """
    if not FIXTURE.is_file():
        raise FileNotFoundError(f"quickstart fixture missing: {FIXTURE}")
    corpus = dest / "processed" / "amlworld_hi_small" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    target = corpus / "bronze.jsonl"
    with gzip.open(FIXTURE, "rb") as src, target.open("wb") as out:
        shutil.copyfileobj(src, out)
    return target


def run_evaluation(dest: Path) -> dict[str, Any]:
    """Run the real evaluation harness over the staged fixture.

    Only two path roots are overridden, so the vocabulary, the schemas and the frozen
    policy all come from the repository exactly as they would in a full run.

    Args:
        dest: The directory the fixture was staged under.

    Returns:
        The parsed ``evaluation.json``.

    Raises:
        RuntimeError: If the evaluator exits non-zero or writes no report.
    """
    metrics = dest / "metrics"
    proc = subprocess.run(
        [
            sys.executable,
            str(EVALUATE),
            f"paths.processed_dir={dest / 'processed'}",
            f"paths.metrics_dir={metrics}",
            f"paths.runs_dir={dest / 'runs'}",
            "eval.surface.bertscore=false",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"scripts/10_evaluate.py exited {proc.returncode}")

    reports = sorted(metrics.glob("eval/*/evaluation.json"))
    if not reports:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"no evaluation.json written under {metrics}")
    parsed = json.loads(reports[-1].read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("evaluation.json is not an object")
    return parsed


def extract(report: dict[str, Any]) -> dict[str, Any]:
    """Pull the asserted fields out of a full evaluation report.

    Args:
        report: The parsed ``evaluation.json``.

    Returns:
        A flat dict of the faithfulness fields plus the per-class taxonomy rates.

    Raises:
        KeyError: If the Bronze system is absent from the report.
    """
    entry = report["systems"]["bronze/balanced"]
    faith = entry["faithfulness"]
    return {
        "faithfulness": {k: faith[k] for k in ASSERTED},
        "taxonomy_rate_by_class": entry["taxonomy"]["rate_by_class"],
        "taxonomy_critical_error_rate": entry["taxonomy"]["critical_error_rate"],
    }


def compare(observed: dict[str, Any], golden: dict[str, Any]) -> list[str]:
    """Diff observed against golden, exactly.

    Args:
        observed: This run's extracted fields.
        golden: The committed expected fields.

    Returns:
        One human-readable line per mismatch; empty when they agree.
    """
    problems: list[str] = []
    for section, expected in golden.items():
        actual = observed.get(section)
        if not isinstance(expected, dict):
            if actual != expected:
                problems.append(f"{section}: expected {expected!r}, got {actual!r}")
            continue
        if not isinstance(actual, dict):
            problems.append(f"{section}: missing from this run")
            continue
        for key, want in expected.items():
            got = actual.get(key)
            if got != want:
                problems.append(f"{section}.{key}: expected {want!r}, got {got!r}")
    return problems


def main() -> int:
    """Stage the fixture, score it, and assert the result.

    Returns:
        0 when the observed report matches the golden file exactly, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce one published result from a clean clone, in under a minute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Runs the real Phase 10 harness over the committed 220-record Bronze fixture "
            "and asserts the result exactly. See docs/REPRODUCTION.md section 3."
        ),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the staged tree and the written report in place for inspection",
    )
    parser.add_argument(
        "--regenerate-golden",
        action="store_true",
        help="overwrite the committed golden file. Read the diff before committing it.",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="g2t-quickstart-"))
    try:
        staged = stage_fixture(workdir)
        n_records = sum(1 for _ in staged.open(encoding="utf-8"))
        print(f"staged {n_records} Bronze records from {FIXTURE.name}")

        observed = extract(run_evaluation(workdir))

        if args.regenerate_golden:
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n", "utf-8")
            print(f"wrote {GOLDEN.relative_to(REPO_ROOT)} — read the diff before committing")
            return 0

        if not GOLDEN.is_file():
            print(f"no golden file at {GOLDEN}; run with --regenerate-golden", file=sys.stderr)
            return 1
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

        faith = observed["faithfulness"]
        print("")
        print(f"  Zero-Hallucination Rate        {faith['zero_hallucination_rate']:.4f}")
        print(f"  Fact Precision                 {faith['fact_precision']:.4f}")
        print(f"  Hallucination Rate             {faith['hallucination_rate']:.4f}")
        print(f"  Fact Coverage                  {faith['fact_coverage']:.4f}")
        print(f"  Critical Error Rate            {faith['critical_error_rate']:.4f}")
        print(
            f"  H9 omission of exculpatory fact" f"  {observed['taxonomy_rate_by_class']['H9']:.4f}"
        )
        print(f"  claims scored                  {faith['n_claims']}")
        print(f"  narratives with no claims      {faith['n_narratives_with_no_claims']}")
        print("")

        problems = compare(observed, golden)
        if problems:
            print("QUICKSTART FAILED — observed does not match the golden file:", file=sys.stderr)
            for line in problems:
                print(f"  {line}", file=sys.stderr)
            print("", file=sys.stderr)
            print(
                "Bronze is deterministic, so this is a real difference and not "
                "variance. See docs/REPRODUCTION.md section 9.",
                file=sys.stderr,
            )
            return 1

        print("QUICKSTART OK — matches tests/golden/quickstart_evaluation.json exactly")
        return 0
    finally:
        if args.keep:
            print(f"staged tree kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

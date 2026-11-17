#!/usr/bin/env python
"""Phase 11: score every completed matrix run with the Phase 10 evaluator.

The matrix runner writes a run's ``generations.jsonl`` and its completion marker; the
aggregator reads a run's ``metrics.json`` and ``per_case.jsonl``. **Nothing wrote the
second pair.** The aggregation tests wrote them by hand, so the gap was invisible until
the first real B1 and B2 runs completed and ``make aggregate`` reported that no system had
produced metrics (D-112). This script is the missing step, and it does nothing the Phase 10
evaluator does not already do: the same ``evaluate`` call, the same Bronze slot index as
Method A's alignment reference, the same fact records, one system per call.

``metrics.json`` is :meth:`EvaluationReport.to_dict` verbatim, which is the shape
``experiments.aggregate`` was written against. ``per_case.jsonl`` is one row per narrative
with the per-case value of every main-table metric, which is what the aggregator's paired
Wilcoxon tests pair on.

A run is scored once. A run that already has ``metrics.json`` is skipped unless
``--force``; the old file is then replaced, because it is derived from ``generations.jsonl``
in the same directory and re-deriving it with the same code is not a new result.

Usage:
    uv run python scripts/11c_score_matrix.py
    uv run python scripts/11c_score_matrix.py --systems S1,A1 --force
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from g2t_aml.eval.layer2_faithfulness import CaseFaithfulness
from g2t_aml.eval.report import evaluate
from g2t_aml.eval.types import load_system_outputs
from g2t_aml.experiments.registry import all_systems, get_system
from g2t_aml.experiments.runner import COMPLETION_MARKER
from g2t_aml.facts.schema import CASE_FACTS_SCHEMA_VERSION
from g2t_aml.facts.vocab import load_vocabulary
from g2t_aml.utils.io import write_json, write_jsonl
from g2t_aml.utils.logging import configure_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATIONS_FILE = "generations.jsonl"
METRICS_FILE = "metrics.json"
PER_CASE_FILE = "per_case.jsonl"


def _load_corpus_reader() -> Any:
    """Import the Phase 10 entrypoint's corpus reader rather than writing a second one.

    Returns:
        ``scripts/10_evaluate.py``'s ``_load_corpus``. A second reader would be a second
        place for the fact record a narrative is scored against to drift from the one
        ``make eval`` uses.
    """
    spec = importlib.util.spec_from_file_location(
        "phase10_evaluate", REPO_ROOT / "scripts" / "10_evaluate.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load scripts/10_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_corpus


def per_case_row(case: CaseFaithfulness) -> dict[str, Any]:
    """Return one narrative's values under the aggregator's metric names.

    Args:
        case: The per-case faithfulness.

    Returns:
        The case's own record plus every main-table metric as a float. The binary
        headline is stored as ``zero_hallucination_rate`` (0 or 1) so that its mean over a
        set of cases is the set's Zero-Hallucination Rate.
    """
    row = case.to_dict()
    row.pop("results", None)
    row.update(
        {
            "zero_hallucination_rate": float(case.zero_hallucination),
            "fact_precision": float(case.fact_precision),
            "hallucination_rate": float(case.hallucination_rate),
            "unverifiable_rate": float(case.unverifiable_rate),
            "fact_coverage": float(case.fact_coverage),
            "fact_f1": float(case.fact_f1),
            "critical_error_rate": float(case.n_critical > 0),
        }
    )
    return row


def _completed_runs(root: Path, system_ids: list[str]) -> list[tuple[str, int, Path]]:
    """List the newest completed run directory of every (system, seed).

    Args:
        root: The matrix root.
        system_ids: Systems to look at.

    Returns:
        ``(system, seed, directory)`` for each one that has completed.
    """
    found = []
    for system_id in system_ids:
        spec = get_system(system_id)
        for seed in sorted(spec.seeds):
            seed_dir = root / system_id / f"seed{seed}"
            if not seed_dir.is_dir():
                continue
            done = [d for d in seed_dir.iterdir() if (d / COMPLETION_MARKER).is_file()]
            if done:
                newest = max(done, key=lambda d: (d / COMPLETION_MARKER).stat().st_mtime)
                found.append((system_id, seed, newest))
    return found


def main() -> int:
    """Score every completed run that has generations and no metrics.

    Returns:
        0 when every eligible run was scored, 1 when any failed. A run with no
        generations file (an API arm that was blocked, say) is reported and skipped.
    """
    parser = argparse.ArgumentParser(description="Score completed matrix runs.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts" / "matrix")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "amlworld_hi_small" / "corpus" / "bronze.jsonl",
        help="the corpus whose fact records and Bronze slots the narratives are scored against",
    )
    parser.add_argument("--systems", default="", help="comma-separated ids; all when omitted")
    parser.add_argument("--force", action="store_true", help="re-score runs already scored")
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()

    configure_logging()
    log = get_logger("score_matrix")

    system_ids = (
        [s.strip() for s in args.systems.split(",") if s.strip()]
        if args.systems
        else [spec.system_id for spec in all_systems()]
    )
    runs = _completed_runs(args.root, system_ids)
    todo = [
        (system, seed, directory)
        for system, seed, directory in runs
        if (directory / GENERATIONS_FILE).is_file()
        and (args.force or not (directory / METRICS_FILE).is_file())
    ]
    for system, seed, directory in runs:
        if not (directory / GENERATIONS_FILE).is_file():
            log.warning(
                "%s seed %d completed with no %s; nothing to score", system, seed, GENERATIONS_FILE
            )
    log.info("%d completed runs, %d to score", len(runs), len(todo))
    if not todo:
        return 0

    _texts, bronze_index, facts = _load_corpus_reader()(args.corpus)
    vocabulary = load_vocabulary()
    log.info("loaded %d fact records from %s", len(facts), args.corpus)

    failed = 0
    for system, seed, directory in todo:
        try:
            outputs = load_system_outputs(directory / GENERATIONS_FILE, system=system, seed=seed)
            report = evaluate(
                outputs,
                {},
                facts,
                bronze=bronze_index,
                run_id=f"{system}_seed{seed}",
                vocabulary=vocabulary,
                bertscore_model=None,
                metadata={
                    "dataset": "amlworld_hi_small",
                    "schema_version": CASE_FACTS_SCHEMA_VERSION,
                    "system": system,
                    "seed": seed,
                    "generations": str(directory / GENERATIONS_FILE),
                    "n_narratives": len(outputs),
                },
                n_resamples=args.resamples,
                seed=seed,
            )
            write_json(directory / METRICS_FILE, report.to_dict())
            rows = [
                per_case_row(case)
                for block in report.systems.values()
                for case in block.faithfulness.per_case
            ]
            write_jsonl(directory / PER_CASE_FILE, rows)
            headline = next(iter(report.systems.values())).faithfulness
            log.info(
                "%s seed %d: %d narratives, Zero-Hallucination %.4f, coverage %.4f -> %s",
                system,
                seed,
                headline.n_cases,
                headline.zero_hallucination_rate,
                headline.fact_coverage,
                directory,
            )
        except Exception:
            failed += 1
            log.exception("%s seed %d could not be scored", system, seed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

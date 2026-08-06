#!/usr/bin/env python
"""Phase 11: aggregate every run's metrics, run the significance battery, emit the paper.

Collects the matrix into one tidy long-format table (system x seed x metric x substrate x
test set), computes across-seed means and bootstrap CIs, runs Wilcoxon signed-rank with
Holm--Bonferroni across each metric's comparison family, and writes the main, ablation and
taxonomy tables as LaTeX plus every figure.

**Runs against an incomplete matrix on purpose.** A system that did not run is reported by
name with its reason, in the tables' captions and in ``missing_runs.md``; it is never a
zero and never silently dropped. That is invariant 7, and it is also what lets this script
be exercised -- and RESULTS.md be written -- before any arm has trained.

Usage:
    uv run python scripts/11a_aggregate.py
    uv run python scripts/11a_aggregate.py --no-figures
    uv run python scripts/11a_aggregate.py --root artifacts/matrix --out artifacts/metrics/phase11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from g2t_aml.experiments.aggregate import (
    aggregate_matrix,
    missing_report,
    write_outputs,
)
from g2t_aml.experiments.registry import matrix_summary
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger
from g2t_aml.utils.run_context import RunContext

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Aggregate the matrix and write every artifact.

    Returns:
        0 always, including on an empty matrix. **Aggregation failing because nothing has
        run yet would be the wrong behaviour**: the absences are the deliverable at that
        point, and a non-zero exit here would stop the reporting path from being tested
        until the day the last GPU job finishes.
    """
    parser = argparse.ArgumentParser(description="Aggregate the Phase 11 matrix.")
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT / "artifacts" / "matrix", help="matrix root"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "metrics" / "phase11",
        help="where the tables and the tidy table are written",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "figures" / "phase11",
        help="where the figures are written",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip rendering figures")
    parser.add_argument(
        "--stream", default="balanced", help="which stream to tabulate; never pooled"
    )
    parser.add_argument(
        "--resamples", type=int, default=10000, help="bootstrap resamples per interval"
    )
    args = parser.parse_args()

    configure_logging()
    log = get_logger("aggregate")

    summary = matrix_summary()
    log.info("matrix declares %d systems / %d runs", summary["n_systems"], summary["n_runs"])

    result = aggregate_matrix(args.root, n_resamples=args.resamples)
    log.info(
        "collected %d rows from %d/%d systems; %d declared runs produced no metrics",
        len(result.rows),
        len(result.systems_present),
        summary["n_systems"],
        len(result.missing),
    )

    written = write_outputs(result, args.out, streams=(args.stream,))
    for name, path in sorted(written.items()):
        log.info("%s -> %s", name, path)

    missing_path = Path(args.out) / "missing_runs.md"
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.write_text(
        "# Declared runs with no metrics\n\n"
        "Invariant 7: absences are a deliverable, not a diagnostic. Every row below is a\n"
        "run the registry declares and the matrix root does not contain.\n\n"
        + missing_report(result)
        + "\n",
        encoding="utf-8",
    )
    log.info("missing_runs -> %s", missing_path)

    context = RunContext.capture(
        experiment_name="phase11_aggregate",
        cfg=result.metadata,
        seeds={"bootstrap": args.resamples},
        repo_root=REPO_ROOT,
        matrix_root=str(args.root),
        n_systems_declared=summary["n_systems"],
        n_systems_present=len(result.systems_present),
        n_missing_runs=len(result.missing),
    )
    write_json(Path(args.out) / "run_context.json", context.to_dict())

    if not args.no_figures:
        try:
            from g2t_aml.experiments.figures import render_all
        except ImportError:
            log.warning("matplotlib is not installed; skipping figures")
        else:
            figures = render_all(result, args.figures_dir, stream=args.stream)
            for name, path in sorted(figures.items()):
                log.info("figure %s -> %s", name, path)

    if not result.systems_present:
        log.warning(
            "NO SYSTEM produced metrics. Every artifact above was written and every "
            "number in it is an absence. See missing_runs.md for why each run is absent."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
r"""Gate 8: compare S1 against A1 and state the answer plainly.

This script exists so that the project's decision point is a command with an output rather
than a judgement someone forms while reading two plots. It reads two runs' diagnostic
histories, computes the difference in faithfulness, and prints the verdict — including the
verdict nobody wants, which is that the two arms are indistinguishable and the paper's
contribution is the dataset and the evaluation framework rather than the architecture.

Usage:
    uv run python scripts/09b_compare_arms.py \\
        --treatment artifacts/runs/.../history_S1.jsonl \\
        --control    artifacts/runs/.../history_A1.jsonl \\
        --out        artifacts/metrics/generator/gate8.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from g2t_aml.models.generator.callbacks import compare_arms
from g2t_aml.utils.io import write_json
from g2t_aml.utils.logging import configure_logging, get_logger


def main() -> int:
    """Run the comparison.

    Returns:
        0 when the treatment beats the control, 2 when they are indistinguishable or the
        control wins. The non-zero code is deliberate: a CI run of the Phase 9 gate should
        not pass when the central claim does not hold.
    """
    parser = argparse.ArgumentParser(description="Compare a treatment arm against its control.")
    parser.add_argument("--treatment", required=True, type=Path, help="treatment history JSONL")
    parser.add_argument("--control", required=True, type=Path, help="control history JSONL")
    parser.add_argument("--out", type=Path, default=None, help="where to write the comparison")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="difference below which the arms count as indistinguishable",
    )
    args = parser.parse_args()

    configure_logging()
    log = get_logger("gate8")

    comparison = compare_arms(args.treatment, args.control, tolerance=args.tolerance)
    verdict = comparison.verdict(tolerance=args.tolerance)

    log.info("%s supported rate: %.4f", comparison.treatment_arm, comparison.treatment_supported)
    log.info("%s supported rate: %.4f", comparison.control_arm, comparison.control_supported)
    log.info(
        "difference:        %+.4f over %d aligned steps",
        comparison.difference,
        comparison.n_steps_compared,
    )
    if comparison.tracked_throughout:
        log.warning("the two arms tracked each other at EVERY compared step")
    log.info("VERDICT: %s", verdict)

    if args.out:
        write_json(
            args.out,
            {
                "treatment_arm": comparison.treatment_arm,
                "control_arm": comparison.control_arm,
                "treatment_supported": comparison.treatment_supported,
                "control_supported": comparison.control_supported,
                "difference": comparison.difference,
                "n_steps_compared": comparison.n_steps_compared,
                "tracked_throughout": comparison.tracked_throughout,
                "tolerance": args.tolerance,
                "verdict": verdict,
            },
        )
        log.info("written to %s", args.out)

    return 0 if comparison.difference > args.tolerance else 2


if __name__ == "__main__":
    sys.exit(main())

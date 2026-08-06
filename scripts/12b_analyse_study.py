#!/usr/bin/env python
"""Unblind the Phase 12 responses and produce the analysis.

Reads the blind key. This and ``12c_release_study.py`` are the only two entrypoints that
do, which is what makes "could the interface have seen the system?" answerable by looking
at which files each process opens.

Writes ``study_analysis.json`` plus a critical-difference diagram per metric.

**Every number is reported with what qualifies it.** The analysis carries a warnings list —
responses timed by the fallback clock, dimensions with no doubly-rated cell, tests that
could not be run — and this script prints it last and in full rather than burying it.

Usage:
    uv run python scripts/12b_analyse_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from g2t_aml.human.study_analysis import (
    analyse_study,
    critical_difference_diagram,
    load_blind_key,
    nemenyi_posthoc,
)
from g2t_aml.human.study_design import load_design
from g2t_aml.human.study_ui import ResponseStore
from g2t_aml.utils.io import atomic_path


def _load_automatic_scores(path: Path) -> dict[str, float]:
    """Read the automatic Layer-2 factual score per item, if it has been computed.

    Args:
        path: JSON mapping of item id to score.

    Returns:
        Item id to score. Empty when the file is absent.
    """
    if not path.is_file():
        return {}
    return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


#: Exit code captured out of the Hydra-decorated entrypoint. `@hydra.main` discards the
#: wrapped function's return value; see D-051.
_EXIT_CODE: list[int] = []


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def _run(cfg: DictConfig) -> None:
    """Run the analysis.

    Args:
        cfg: The composed Hydra config.

    Returns:
        Nothing. The exit code is recorded in :data:`_EXIT_CODE`: 2 when there are no
        responses to analyse, 0 otherwise.
    """
    study_dir = Path(cfg.paths.artifacts_dir) / "human_study"
    store = ResponseStore(root=study_dir / "responses")
    responses = store.read_all()
    if not responses:
        print(
            f"No responses under {study_dir / 'responses'}.\n"
            "The study has not been run. See docs/human_study/README.md for the blockers.",
            file=sys.stderr,
        )
        _EXIT_CODE.append(2)
        return

    design = load_design(study_dir / "design.json")
    key = load_blind_key(study_dir / "key.json")
    automatic = _load_automatic_scores(study_dir / "automatic_scores.json")

    analysis = analyse_study(
        responses,
        key,
        systems=design.systems,
        baseline=str(getattr(cfg.get("study", {}), "baseline", "Bronze")),
        automatic_scores=automatic or None,
    )

    out = study_dir / "study_analysis.json"
    with atomic_path(out) as tmp:
        tmp.write_text(
            json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    figures = Path(cfg.paths.figures_dir) / "human_study"
    for metric, entry in analysis.posthoc.items():
        for test_name, payload in entry.items():
            result = nemenyi_posthoc(payload["mean_ranks"], payload["n_blocks"])
            critical_difference_diagram(
                result,
                figures / f"cd_{metric}_{test_name}.png",
                title=f"{metric.replace('_', ' ')} ({test_name})",
            )

    print(f"{analysis.n_responses} ratings from {analysis.n_raters} raters")
    print(f"\nTime-to-usable-draft vs {analysis.baseline}:")
    for system, e in analysis.timing["paired_vs_baseline"].items():
        print(f"  {system:8s} {e['mean_difference']:+9.1f} s   p={e['p_value']}")
    print(f"\nEdit distance vs {analysis.baseline}:")
    for system, e in analysis.edit_distance["paired_vs_baseline"].items():
        print(f"  {system:8s} {e['mean_difference']:+9.4f}     p={e['p_value']}")
    print("\nWould file:")
    for system, e in analysis.would_file.items():
        print(f"  {system:8s} {e['rate']:.3f}  (n={e['n']})")
    print("\nInter-rater agreement (ordinal alpha):")
    for dimension, e in analysis.agreement.items():
        print(f"  {dimension:22s} {e.get('alpha_ordinal')}  (n_units={e['n_units']})")
    print(f"\nAutomatic vs human factual correctness: {analysis.metric_validation}")

    if analysis.warnings:
        print(f"\n{'=' * 70}\nWARNINGS — these qualify the numbers above\n{'=' * 70}")
        for warning in analysis.warnings:
            print(f"  - {warning}")

    print(f"\nWrote {out}")
    print(f"Wrote critical-difference diagrams under {figures}")


def main() -> int:
    """Run the Hydra entrypoint and return its exit code.

    ``@hydra.main`` discards its wrapped function's return value, so a code produced
    inside ``_run`` has to travel out through :data:`_EXIT_CODE` (D-051).

    Returns:
        The code ``_run`` produced, or 0 when it produced none.
    """
    _run()
    return _EXIT_CODE[-1] if _EXIT_CODE else 0


if __name__ == "__main__":
    sys.exit(main())

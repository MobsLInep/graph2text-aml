#!/usr/bin/env python
"""Render `encoder_report.json` as the markdown tables PHASE_LOG.md carries.

The write-up quotes this script's output rather than numbers retyped by hand. Retyping is
where a results table stops matching the artifact it claims to summarise, and invariant 7
is only worth anything if the reported numbers are the produced ones.

Usage:
    uv run python scripts/07c_report_tables.py
    uv run python scripts/07c_report_tables.py artifacts/metrics/encoder/encoder_report.json
"""

from __future__ import annotations

import sys
from math import isnan
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "metrics" / "encoder" / "encoder_report.json"


def _pm(summary: dict[str, Any], places: int = 4) -> str:
    """Format a seed aggregate as ``mean ± std``."""
    mean, std = summary.get("mean"), summary.get("std")
    if mean is None or isnan(mean):
        return "—"
    return f"{mean:.{places}f} ± {std:.{places}f}"


def _interval(entry: dict[str, Any] | None) -> str:
    """Format a bootstrap interval as ``point [lo, hi]``."""
    if not entry:
        return "—"
    return f"{entry['point']:.4f} [{entry['lo']:.4f}, {entry['hi']:.4f}]"


def arms_table(report: dict[str, Any]) -> str:
    """Render the per-arm results table.

    Args:
        report: The loaded ``encoder_report.json``.

    Returns:
        A markdown table, arms ordered by mean test AUC-PR descending.
    """
    arms = report["arms"]
    bootstrap = report.get("comparison", {}).get("bootstrap", {})
    order = sorted(arms, key=lambda a: -(arms[a]["test"]["mean"] or 0))

    lines = [
        "| Arm | Params | test AUC-PR | test AUC-ROC | realistic AUC-PR | "
        "typology macro-F1 (structural) | chance | test AUC-PR CI (seed 42) |",
        "|---|---:|---|---|---|---|---:|---|",
    ]
    for name in order:
        entry = arms[name]
        first_seed = bootstrap.get(name, {}).get("per_seed", {})
        ci = next(iter(first_seed.values()), {}).get("test") if first_seed else None
        lines.append(
            f"| `{name}` | {entry['n_parameters']:,} | {_pm(entry['test'])} | "
            f"{_pm(entry['test_auc_roc'])} | {_pm(entry['realistic'])} | "
            f"{_pm(entry['typology_macro_f1_structural'], 3)} | "
            f"{entry['typology_chance']['mean']:.3f} | {_interval(ci)} |"
        )
    return "\n".join(lines)


def gate_table(report: dict[str, Any]) -> str:
    """Render the GATv2-versus-MLP gate, per seed.

    Args:
        report: The loaded report.

    Returns:
        A markdown block stating the gate outcome and its per-seed intervals.
    """
    gate = report.get("comparison", {}).get("gate", {})
    if "paired_difference" not in gate:
        return "_The gate did not run: the primary arm or the MLP control is absent._"

    difference = gate["paired_difference"]
    lines = [
        f"**Gate: `{gate['primary']}` vs the `{gate['control']}` control on test AUC-PR "
        f"— {'PASSED' if gate['passed'] else 'FAILED'}.**",
        "",
        "| Seed | AUC-PR difference | 95% CI (paired bootstrap) | excludes zero |",
        "|---|---:|---|---|",
    ]
    for seed, value in difference["per_seed"].items():
        lines.append(
            f"| {seed} | {value['difference']:+.4f} | "
            f"[{value['lo']:+.4f}, {value['hi']:+.4f}] | "
            f"{'yes' if value['excludes_zero'] else 'no'} |"
        )
    lines += [
        f"| **mean** | **{difference['mean_difference']:+.4f}** | — | "
        f"{'all seeds' if difference['excludes_zero_at_every_seed'] else 'not at every seed'} |",
        "",
        f"Marginal intervals at the first seed: "
        f"`{gate['primary']}` {_interval(gate['primary_ci'])}, "
        f"`{gate['control']}` {_interval(gate['control_ci'])} — "
        f"{'non-overlapping' if gate['non_overlapping_marginal_cis'] else 'overlapping'}.",
    ]
    realistic = gate.get("paired_difference_realistic")
    if realistic and realistic["per_seed"]:
        lines.append(
            f"On the realistic-imbalance stream the mean difference is "
            f"{realistic['mean_difference']:+.4f}, excluding zero at "
            f"{'every seed' if realistic['excludes_zero_at_every_seed'] else 'not every seed'}."
        )
    return "\n".join(lines)


def arm_comparison_table(report: dict[str, Any]) -> str:
    """Render the primary arm against every other arm, paired and per seed.

    Args:
        report: The loaded report.

    Returns:
        A markdown table, or a note when the comparison did not run.
    """
    comparison = report.get("comparison", {}).get("arms", {})
    if not comparison:
        return "_No arm-versus-arm comparison in this report._"
    primary = report.get("comparison", {}).get("gate", {}).get("primary", "primary")
    lines = [
        f"| `{primary}` minus | mean AUC-PR difference | per-seed | "
        "excludes zero at every seed |",
        "|---|---:|---|---|",
    ]
    for other, value in sorted(comparison.items(), key=lambda x: x[1]["mean_difference"]):
        per_seed = ", ".join(f"{v['difference']:+.4f}" for v in value["per_seed"].values())
        lines.append(
            f"| `{other}` | {value['mean_difference']:+.4f} | {per_seed} | "
            f"{'yes' if value['excludes_zero_at_every_seed'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "A negative difference means that arm beat the primary. Three seeds' means and "
        "standard deviations overlap long before a paired difference does, which is why "
        "the primary-arm decision is made on this table and not on the results table."
    )
    return "\n".join(lines)


def ablation_table(report: dict[str, Any]) -> str:
    """Render the ablation comparison against the primary arm.

    Args:
        report: The loaded report.

    Returns:
        A markdown table, or a note when no ablation ran.
    """
    ablations = report.get("comparison", {}).get("ablations", {})
    if not ablations:
        return "_No ablation ran in this sweep._"
    lines = [
        "| Ablation | mean AUC-PR cost | per-seed differences | excludes zero at every seed |",
        "|---|---:|---|---|",
    ]
    for tag, value in sorted(ablations.items()):
        per_seed = ", ".join(f"{v['difference']:+.4f}" for v in value["per_seed"].values())
        lines.append(
            f"| `{tag}` | {value['mean_difference']:+.4f} | {per_seed} | "
            f"{'yes' if value['excludes_zero_at_every_seed'] else 'no'} |"
        )
    return "\n".join(lines)


def embedding_table(report: dict[str, Any]) -> str:
    """Render the embedding-quality battery.

    Args:
        report: The loaded report.

    Returns:
        A markdown table of kNN purity, silhouette and linear-probe numbers per arm.
    """
    analyses = report.get("embedding_analysis", {})
    if not analyses:
        return "_No embedding analysis in this report._"

    lines = [
        "| Arm | kNN purity k=5 (null) | z | k=10 | k=20 | silhouette (null) | "
        "probe acc. (shuffled) | probe structural macro-F1 | converged |",
        "|---|---|---:|---|---|---|---|---:|---|",
    ]
    for name, analysis in sorted(analyses.items()):
        purity = {p["k"]: p for p in analysis["purity"]}
        probes = {p["representation"]: p for p in analysis["probes"]}
        probe = probes.get("pooled_tokens") or probes.get("graph_embedding") or {}
        p5 = purity.get(5, {})
        lines.append(
            f"| `{name}` | {p5.get('purity', float('nan')):.4f} "
            f"({p5.get('null_mean', float('nan')):.4f}) | {p5.get('z_score', float('nan')):.1f} | "
            f"{purity.get(10, {}).get('purity', float('nan')):.4f} | "
            f"{purity.get(20, {}).get('purity', float('nan')):.4f} | "
            f"{analysis['silhouette']:.4f} ({analysis['silhouette_null_mean']:.4f}) | "
            f"{probe.get('accuracy', float('nan')):.4f} "
            f"({probe.get('shuffled_accuracy', float('nan')):.4f}) | "
            f"{probe.get('structural_macro_f1', float('nan')):.4f} | "
            f"{'yes' if probe.get('converged') else 'NO — lower bound'} |"
        )
    lines.append("")
    lines.append(
        "The probe is fitted on train embeddings and scored on test. `pooled_tokens` is "
        "the representation Phase 8 consumes, so its structural macro-F1 is the column "
        "that forecasts whether fusion can recover a typology."
    )
    return "\n".join(lines)


def alignment_block(report: dict[str, Any]) -> str:
    """Render the attention-alignment finding.

    Args:
        report: The loaded report.

    Returns:
        A markdown paragraph, or a note when the measurement did not run.
    """
    alignment = report.get("attention_alignment") or {}
    if not alignment or not alignment.get("n_cases"):
        return "_Attention alignment was not measured in this run._"
    lines = [
        f"Over {alignment['n_cases']} suspicious test cases, "
        f"**{alignment['mean_path_attention']:.3f}** of the pooling attention mass falls on "
        f"accounts touching a flagged transaction, against **{alignment['mean_path_share']:.3f}** "
        f"for a uniformly-attending model — a lift of **{alignment['lift']:.2f}**. "
        f"The single highest-attention account is on the laundering path in "
        f"**{alignment['top1_hit_rate']:.1%}** of cases.",
    ]
    per_typology = alignment.get("per_typology_lift") or {}
    if per_typology:
        lines += [
            "",
            "| Typology | attention lift |",
            "|---|---:|",
            *(f"| {k} | {v:.2f} |" for k, v in sorted(per_typology.items(), key=lambda x: -x[1])),
        ]
    return "\n".join(lines)


def main() -> int:
    """Print every table for the report named on the command line.

    Returns:
        0 on success, 1 when the report is absent.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The report is produced by `make train-encoder`. This script only renders it; "
            "it never recomputes a number."
        ),
    )
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"path to encoder_report.json (default: {DEFAULT_REPORT.relative_to(REPO_ROOT)})",
    )
    path = parser.parse_args().report
    if not path.is_file():
        print(f"no encoder report at {path}; run `make train-encoder` first", file=sys.stderr)
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))

    print("### Results across arms\n")
    print(arms_table(report))
    print("\n### The gate\n")
    print(gate_table(report))
    print("\n### The primary arm against the others\n")
    print(arm_comparison_table(report))
    print("\n### Ablations\n")
    print(ablation_table(report))
    print("\n### Embedding quality\n")
    print(embedding_table(report))
    print("\n### Attention alignment\n")
    print(alignment_block(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

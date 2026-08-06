"""Phase 11: collect every run's metrics into one table and run the significance battery.

The unit of this module is the **tidy long-format row**: one system, one seed, one metric,
one substrate, one stream, one value. Everything the paper reports is a projection of that
table, which is why it is written to disk as its own artifact -- a reviewer asking "where
does the number in row 3 of Table 2 come from" is answered by a filter, not by a rerun.

Four things here are decisions, not mechanics:

**Missing runs are reported, never imputed.** A system that did not run is a named absence
carrying its reason (:class:`MissingRun`), and it appears in the results file and in the
LaTeX table caption. It is never a zero, never an interpolation, and never silently
dropped -- a table with fifteen rows where the matrix declares sixteen is a table that lies
by omission.

**Single-seed rows print an em dash for their standard deviation**, because
:class:`~g2t_aml.eval.statistics.SeedSummary` sets ``std=None`` at one seed. Printing 0.0
would report a single-seed baseline as having no variance, which is the specific
misreading the seed asymmetry has to survive.

**The correction family is one metric on one slice**, which is what one call to
:func:`~g2t_aml.eval.statistics.compare_systems` produces (D-079). With sixteen systems the
difference between correcting over 120 comparisons and over 15 is the difference between a
finding and a coincidence, so the family is never assembled by a caller across calls.

**Seed-level values are averaged for the seed summary and pooled for the paired test.** The
across-seed mean answers "how does this system score"; the per-case paired test answers
"does this system beat that one", and it runs on the cases the two systems share. Mixing
the two -- a paired test over three seed means -- would throw away 3,192 cases of paired
information to gain nothing.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from g2t_aml.eval.statistics import (
    BOOTSTRAP_RESAMPLES,
    Interval,
    PairedComparison,
    SeedSummary,
    bootstrap_ci,
    compare_systems,
    seed_summary,
)
from g2t_aml.experiments.registry import (
    SystemSpec,
    all_systems,
    central_claim_family,
    comparison_family,
    get_system,
)
from g2t_aml.experiments.runner import COMPLETION_MARKER, RunStatus
from g2t_aml.utils.io import read_json, read_jsonl, write_json
from g2t_aml.utils.logging import get_logger

__all__ = [
    "HEADLINE_METRIC",
    "MAIN_TABLE_METRICS",
    "AggregateResult",
    "MetricRow",
    "MissingRun",
    "PerCaseValues",
    "ablation_table_latex",
    "aggregate_matrix",
    "collect_rows",
    "long_table",
    "main_table_latex",
    "taxonomy_table_latex",
]

log = get_logger(__name__)

#: The headline, per D-077: the per-narrative binary, not averaged precision, because one
#: fabricated fact makes a SAR unfileable regardless of the rest.
HEADLINE_METRIC = "zero_hallucination_rate"

#: The main results table's columns, in reporting order. Layer 2 faithfulness leads
#: everywhere and Layer 1 overlap follows under its own heading (D-080).
MAIN_TABLE_METRICS: tuple[str, ...] = (
    "zero_hallucination_rate",
    "fact_precision",
    "hallucination_rate",
    "unverifiable_rate",
    "fact_coverage",
    "fact_f1",
    "critical_error_rate",
)

#: The ablation table reports the headline plus what each ablation is meant to move.
ABLATION_TABLE_METRICS: tuple[str, ...] = (
    "zero_hallucination_rate",
    "fact_coverage",
    "fact_f1",
    "critical_error_rate",
)

#: The nine hallucination classes, in taxonomy order.
TAXONOMY_CLASSES: tuple[str, ...] = ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9")

_METRICS_FILE = "metrics.json"
_PER_CASE_FILE = "per_case.jsonl"


@dataclass(frozen=True)
class MetricRow:
    """One cell of the tidy table.

    Attributes:
        system: The system id.
        seed: The seed this value came from.
        metric: The metric name.
        substrate: The data substrate, e.g. ``amlworld_hi_small``.
        stream: ``balanced`` or ``realistic``. Kept apart and never pooled: the two
            answer different questions and averaging them answers neither.
        test_set: Which held-out set, e.g. ``test`` or ``gold``.
        typology: The laundering typology, or ``all`` for the pooled value.
        value: The number.
        n_cases: How many narratives it aggregates. Carried on every row so a metric over
            a subset is never mistaken for one over the whole test set.
    """

    system: str
    seed: int
    metric: str
    substrate: str
    stream: str
    test_set: str
    typology: str
    value: float
    n_cases: int

    def to_dict(self) -> dict[str, Any]:
        """Return the row as a JSON-serialisable mapping.

        Returns:
            Every field.
        """
        return {
            "system": self.system,
            "seed": self.seed,
            "metric": self.metric,
            "substrate": self.substrate,
            "stream": self.stream,
            "test_set": self.test_set,
            "typology": self.typology,
            "value": self.value,
            "n_cases": self.n_cases,
        }


@dataclass(frozen=True)
class MissingRun:
    """A run the matrix declares and the results directory does not contain.

    Attributes:
        system: The system id.
        seed: The seed.
        reason: Why it is absent, in a form a reader of RESULTS.md can act on.
        status: The runner's status, when a run record exists for it.
    """

    system: str
    seed: int
    reason: str
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-serialisable mapping.

        Returns:
            Every field.
        """
        return {
            "system": self.system,
            "seed": self.seed,
            "reason": self.reason,
            "status": self.status,
        }


@dataclass(frozen=True)
class PerCaseValues:
    """One system's per-case values on one metric, for the paired tests.

    Attributes:
        system: The system id.
        metric: The metric.
        stream: The stream these came from.
        values: Case id to value, averaged across seeds where a system has several. The
            average is over seeds within a case, so the pairing across systems survives.
    """

    system: str
    metric: str
    stream: str
    values: Mapping[str, float]


@dataclass(frozen=True)
class AggregateResult:
    """Everything the aggregation produced.

    Attributes:
        rows: The tidy long-format table.
        summaries: ``(stream, metric, system)`` to the across-seed summary.
        intervals: ``(stream, metric, system)`` to the bootstrap CI.
        comparisons: ``"<stream>/<metric>"`` to that family's corrected comparisons.
        missing: Every declared run that produced no metrics.
        taxonomy: ``(stream, system)`` to class-name-to-rate.
        systems_present: The systems that contributed at least one number.
        metadata: Provenance -- what was aggregated, from where, under what policy.
    """

    rows: tuple[MetricRow, ...]
    summaries: Mapping[tuple[str, str, str], SeedSummary] = field(default_factory=dict)
    intervals: Mapping[tuple[str, str, str], Interval] = field(default_factory=dict)
    comparisons: Mapping[str, tuple[PairedComparison, ...]] = field(default_factory=dict)
    missing: tuple[MissingRun, ...] = ()
    taxonomy: Mapping[tuple[str, str], Mapping[str, float]] = field(default_factory=dict)
    systems_present: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        """Return the tidy table as a DataFrame.

        Returns:
            One row per :class:`MetricRow`, columns in the dataclass's field order. Empty
            with the right columns when nothing was collected, so a downstream ``groupby``
            does not raise on an empty matrix.
        """
        return long_table(self.rows)

    def to_dict(self) -> dict[str, Any]:
        """Return the whole aggregation as a JSON-serialisable mapping.

        Returns:
            The summaries, intervals, comparisons, taxonomy and missing runs. The tidy
            table itself is written separately as parquet and CSV -- it is the largest
            artifact and the one most likely to be loaded by something that is not this
            module.
        """
        return {
            "headline_metric": HEADLINE_METRIC,
            "systems_present": list(self.systems_present),
            "n_rows": len(self.rows),
            "summaries": {
                f"{stream}/{metric}/{system}": summary.to_dict()
                for (stream, metric, system), summary in sorted(self.summaries.items())
            },
            "intervals": {
                f"{stream}/{metric}/{system}": interval.to_dict()
                for (stream, metric, system), interval in sorted(self.intervals.items())
            },
            "comparisons": {
                key: [c.to_dict() for c in group] for key, group in sorted(self.comparisons.items())
            },
            "taxonomy": {
                f"{system}/{stream}": dict(sorted(rates.items()))
                for (stream, system), rates in sorted(self.taxonomy.items())
            },
            "missing_runs": [m.to_dict() for m in self.missing],
            "metadata": dict(self.metadata),
        }


def long_table(rows: Sequence[MetricRow]) -> pd.DataFrame:
    """Build the tidy long-format DataFrame.

    Args:
        rows: The collected rows.

    Returns:
        The table, sorted by system, stream, metric, typology and seed so two
        aggregations of the same runs produce byte-identical CSVs.
    """
    columns = [
        "system",
        "seed",
        "metric",
        "substrate",
        "stream",
        "test_set",
        "typology",
        "value",
        "n_cases",
    ]
    if not rows:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
    frame = pd.DataFrame([row.to_dict() for row in rows], columns=columns)
    return frame.sort_values(
        ["system", "stream", "metric", "typology", "seed"], kind="stable"
    ).reset_index(drop=True)


# ------------------------------------------------------------------- collection ---


def _metric_rows_from_report(  # noqa: PLR0912 -- one pass over the report's four
    # nested blocks; a helper per block would hide that they share a row shape.
    report: Mapping[str, Any],
    *,
    system: str,
    seed: int,
    substrate_default: str,
    test_set: str,
) -> list[MetricRow]:
    """Extract every metric row from one run's evaluation report.

    Reads the JSON shape :meth:`g2t_aml.eval.report.EvaluationReport.to_dict` writes, so
    the aggregator consumes exactly what Phase 10 emits rather than a format invented here.

    Args:
        report: The parsed ``metrics.json``.
        system: The system id to attribute rows to.
        seed: The seed.
        substrate_default: Substrate name when the report's slice does not name one.
        test_set: Which held-out set this run scored.

    Returns:
        The rows, including the per-typology and per-substrate breakdowns.
    """
    rows: list[MetricRow] = []
    systems = report.get("systems", {})
    if not isinstance(systems, dict):
        return rows

    for key, block in systems.items():
        if not isinstance(block, dict):
            continue
        stream = str(block.get("stream") or (key.split("/")[-1] if "/" in key else "balanced"))
        faithfulness = block.get("faithfulness")
        if isinstance(faithfulness, dict):
            n_cases = int(faithfulness.get("n_cases") or 0)
            for metric, value in faithfulness.items():
                if metric in ("system", "n_cases") or not isinstance(value, int | float):
                    continue
                rows.append(
                    MetricRow(
                        system=system,
                        seed=seed,
                        metric=metric,
                        substrate=substrate_default,
                        stream=stream,
                        test_set=test_set,
                        typology="all",
                        value=float(value),
                        n_cases=n_cases,
                    )
                )

        for typology, block_metrics in (block.get("by_typology") or {}).items():
            if not isinstance(block_metrics, dict):
                continue
            n_cases = int(block_metrics.get("n_cases") or 0)
            for metric, value in block_metrics.items():
                if metric in ("system", "n_cases") or not isinstance(value, int | float):
                    continue
                rows.append(
                    MetricRow(
                        system=system,
                        seed=seed,
                        metric=metric,
                        substrate=substrate_default,
                        stream=stream,
                        test_set=test_set,
                        typology=str(typology),
                        value=float(value),
                        n_cases=n_cases,
                    )
                )

        for dataset, block_metrics in (block.get("by_dataset") or {}).items():
            if not isinstance(block_metrics, dict):
                continue
            n_cases = int(block_metrics.get("n_cases") or 0)
            for metric, value in block_metrics.items():
                if metric in ("system", "n_cases") or not isinstance(value, int | float):
                    continue
                rows.append(
                    MetricRow(
                        system=system,
                        seed=seed,
                        metric=metric,
                        substrate=str(dataset),
                        stream=stream,
                        test_set=test_set,
                        typology="all",
                        value=float(value),
                        n_cases=n_cases,
                    )
                )

        layer1 = block.get("layer1")
        if isinstance(layer1, dict):
            for metric, value in layer1.items():
                if not isinstance(value, int | float):
                    continue
                rows.append(
                    MetricRow(
                        system=system,
                        seed=seed,
                        metric=f"layer1.{metric}",
                        substrate=substrate_default,
                        stream=stream,
                        test_set=test_set,
                        typology="all",
                        value=float(value),
                        n_cases=int((faithfulness or {}).get("n_cases") or 0)
                        if isinstance(faithfulness, dict)
                        else 0,
                    )
                )
    return rows


def _taxonomy_from_report(report: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, float]]:
    """Extract the H1-H9 per-narrative rates from a report.

    Args:
        report: The parsed ``metrics.json``.

    Returns:
        ``(stream, system)`` to class-to-rate, covering only classes the report carries.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, block in (report.get("systems") or {}).items():
        if not isinstance(block, dict):
            continue
        taxonomy = block.get("taxonomy")
        if not isinstance(taxonomy, dict):
            continue
        stream = str(block.get("stream") or (key.split("/")[-1] if "/" in key else "balanced"))
        system = str(block.get("system") or key.split("/")[0])
        # `rate_by_class` is TaxonomyReport's own key: the fraction of NARRATIVES carrying
        # at least one finding of that class, not the per-claim share.
        rates = taxonomy.get("rate_by_class") or {}
        if isinstance(rates, dict):
            out[(stream, system)] = {
                str(k): float(v) for k, v in rates.items() if isinstance(v, int | float)
            }
    return out


def _per_case_from_run(path: Path, *, metric: str, stream: str) -> dict[str, float]:
    """Read one run's per-case values for one metric.

    Args:
        path: The ``per_case.jsonl`` file.
        metric: Which metric to pull.
        stream: Which stream's cases to keep.

    Returns:
        Case id to value. Empty when the file is absent, which degrades the paired tests
        to "not computed" rather than to a wrong number.
    """
    if not path.is_file():
        return {}
    values: dict[str, float] = {}
    for record in read_jsonl(path):
        if not isinstance(record, dict):
            continue
        if str(record.get("stream", "balanced")) != stream:
            continue
        case_id = record.get("case_id")
        value = record.get(metric)
        if isinstance(case_id, str) and isinstance(value, int | float):
            values[case_id] = float(value)
    return values


def collect_rows(
    root: Path | str,
    *,
    specs: Sequence[SystemSpec] | None = None,
    substrate: str = "amlworld_hi_small",
    test_set: str = "test",
) -> tuple[list[MetricRow], list[MissingRun], dict[tuple[str, str], dict[str, float]]]:
    """Walk the matrix root and collect every completed run's metrics.

    A run contributes only when its directory holds both a completion marker and a metrics
    file. A directory with a marker and no metrics is reported as missing with that as its
    reason -- it means the run finished and the evaluation did not, which is a different
    problem from a run that never started and should not look the same in the results file.

    Args:
        root: The matrix root.
        specs: Systems to look for; the whole registry when omitted.
        substrate: Substrate name attributed to rows whose report does not name one.
        test_set: Which held-out set these runs scored.

    Returns:
        ``(rows, missing, taxonomy)``.
    """
    root = Path(root)
    chosen = list(specs if specs is not None else all_systems())
    rows: list[MetricRow] = []
    missing: list[MissingRun] = []
    taxonomy: dict[tuple[str, str], dict[str, float]] = {}

    for spec in chosen:
        for seed in sorted(spec.seeds):
            seed_dir = root / spec.system_id / f"seed{seed}"
            candidates = (
                sorted(d for d in seed_dir.iterdir() if d.is_dir()) if seed_dir.is_dir() else []
            )
            completed = [d for d in candidates if (d / COMPLETION_MARKER).is_file()]
            if not completed:
                missing.append(
                    MissingRun(
                        system=spec.system_id,
                        seed=seed,
                        reason=(
                            "no completed run directory"
                            if candidates
                            else "run never started (no directory)"
                        ),
                        status=str(RunStatus.PENDING) if not candidates else str(RunStatus.FAILED),
                    )
                )
                continue
            # Several config hashes can coexist under one seed (invariant 6 keeps the old
            # ones). The most recently completed is the current one; the others stay on
            # disk and stay readable.
            chosen_dir = max(completed, key=lambda d: (d / COMPLETION_MARKER).stat().st_mtime)
            metrics_path = chosen_dir / _METRICS_FILE
            if not metrics_path.is_file():
                missing.append(
                    MissingRun(
                        system=spec.system_id,
                        seed=seed,
                        reason=f"run completed but wrote no {_METRICS_FILE}",
                        status=str(RunStatus.COMPLETED),
                    )
                )
                continue
            report = read_json(metrics_path)
            if not isinstance(report, dict):
                missing.append(
                    MissingRun(
                        system=spec.system_id,
                        seed=seed,
                        reason=f"{_METRICS_FILE} is not a JSON object",
                        status=str(RunStatus.COMPLETED),
                    )
                )
                continue
            rows.extend(
                _metric_rows_from_report(
                    report,
                    system=spec.system_id,
                    seed=seed,
                    substrate_default=substrate,
                    test_set=test_set,
                )
            )
            for key, rates in _taxonomy_from_report(report).items():
                taxonomy[(key[0], spec.system_id)] = rates

    return rows, missing, taxonomy


def _per_case_values(
    root: Path | str,
    *,
    specs: Sequence[SystemSpec],
    metric: str,
    stream: str,
) -> dict[str, dict[str, float]]:
    """Collect per-case values for every system, averaged across seeds within a case.

    Averaging within a case rather than across the whole test set is what keeps the pairing
    intact: two systems compared on case ``c`` are compared on the same case, whichever
    seeds each of them ran.

    Args:
        root: The matrix root.
        specs: The systems.
        metric: The metric.
        stream: The stream.

    Returns:
        System id to case-id-to-value, omitting systems with no per-case file.
    """
    root = Path(root)
    out: dict[str, dict[str, float]] = {}
    for spec in specs:
        accumulated: dict[str, list[float]] = {}
        for seed in sorted(spec.seeds):
            seed_dir = root / spec.system_id / f"seed{seed}"
            if not seed_dir.is_dir():
                continue
            completed = [d for d in sorted(seed_dir.iterdir()) if (d / COMPLETION_MARKER).is_file()]
            if not completed:
                continue
            chosen_dir = max(completed, key=lambda d: (d / COMPLETION_MARKER).stat().st_mtime)
            for case_id, value in _per_case_from_run(
                chosen_dir / _PER_CASE_FILE, metric=metric, stream=stream
            ).items():
                accumulated.setdefault(case_id, []).append(value)
        if accumulated:
            out[spec.system_id] = {
                case_id: sum(values) / len(values) for case_id, values in accumulated.items()
            }
    return out


# ------------------------------------------------------------------ aggregation ---


def aggregate_matrix(
    root: Path | str,
    *,
    specs: Sequence[SystemSpec] | None = None,
    metrics: Sequence[str] = MAIN_TABLE_METRICS,
    substrate: str = "amlworld_hi_small",
    test_set: str = "test",
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> AggregateResult:
    """Collect, summarise and compare every run in the matrix.

    Args:
        root: The matrix root.
        specs: Systems to aggregate; the whole registry when omitted.
        metrics: Metrics to summarise and compare.
        substrate: Substrate name for rows whose report does not name one.
        test_set: Which held-out set was scored.
        n_resamples: Bootstrap resamples per interval.
        seed: Seeds the bootstrap.

    Returns:
        The aggregation. **A matrix with zero completed runs returns a valid, empty
        result** carrying every declared run as missing, which is what lets the reporting
        path be exercised -- and RESULTS.md be written -- before any arm has trained.
    """
    chosen = list(specs if specs is not None else all_systems())
    rows, missing, taxonomy = collect_rows(
        root, specs=chosen, substrate=substrate, test_set=test_set
    )
    present = sorted({row.system for row in rows})

    summaries: dict[tuple[str, str, str], SeedSummary] = {}
    intervals: dict[tuple[str, str, str], Interval] = {}
    streams = sorted({row.stream for row in rows}) or ["balanced"]

    for stream in streams:
        for metric in metrics:
            for system in present:
                per_seed = {
                    row.seed: row.value
                    for row in rows
                    if row.system == system
                    and row.stream == stream
                    and row.metric == metric
                    and row.typology == "all"
                    and row.substrate == substrate
                }
                if not per_seed:
                    continue
                summaries[(stream, metric, system)] = seed_summary(system, metric, per_seed)

    comparisons: dict[str, tuple[PairedComparison, ...]] = {}
    for stream in streams:
        for metric in metrics:
            values_by_system = _per_case_values(root, specs=chosen, metric=metric, stream=stream)
            for system, values in values_by_system.items():
                if values:
                    intervals[(stream, metric, system)] = bootstrap_ci(
                        list(values.values()), n_resamples=n_resamples, seed=seed
                    )
            if len(values_by_system) >= 2:  # noqa: PLR2004 -- a pair is two
                family = compare_systems(
                    metric,
                    values_by_system,
                    n_resamples=n_resamples,
                    seed=seed,
                    family=comparison_family(metric),
                )
                if family:
                    comparisons[f"{stream}/{metric}"] = tuple(family)

    for missing_run in missing:
        log.warning(
            "missing: %s seed %d -- %s", missing_run.system, missing_run.seed, missing_run.reason
        )

    return AggregateResult(
        rows=tuple(rows),
        summaries=summaries,
        intervals=intervals,
        comparisons=comparisons,
        missing=tuple(missing),
        taxonomy=taxonomy,
        systems_present=tuple(present),
        metadata={
            "root": str(root),
            "substrate": substrate,
            "test_set": test_set,
            "metrics": list(metrics),
            "n_resamples": n_resamples,
            "bootstrap_seed": seed,
            "n_systems_declared": len(chosen),
            "n_systems_present": len(present),
            "n_missing_runs": len(missing),
            "central_claim_comparisons": [list(p) for p in central_claim_family()],
        },
    )


# ----------------------------------------------------------------------- tables ---


def _fmt(value: float | None, places: int = 4) -> str:
    """Format a number for a table, or an em dash when there is none.

    Args:
        value: The number, or None.
        places: Decimal places.

    Returns:
        The formatted cell. **None prints an em dash, never a zero** -- a missing standard
        deviation and a zero standard deviation are different claims.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{places}f}"


def _mean_std_cell(summary: SeedSummary | None, places: int = 4) -> str:
    r"""Render a mean-and-standard-deviation cell.

    Args:
        summary: The across-seed summary, or None.
        places: Decimal places.

    Returns:
        ``mean $\\pm$ std`` at more than one seed, the bare mean with a dagger at one seed.
        The dagger is what carries the seed asymmetry into the table itself rather than
        leaving it in a caption a reader may skip.
    """
    if summary is None:
        return "--"
    if summary.std is None:
        return f"{summary.mean:.{places}f}$^{{\\dagger}}$"
    return f"{summary.mean:.{places}f} $\\pm$ {summary.std:.{places}f}"


def _latex_escape(text: str) -> str:
    """Escape the LaTeX specials that appear in system ids and roles.

    Args:
        text: Raw text.

    Returns:
        The escaped text.
    """
    for char, replacement in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")):
        text = text.replace(char, replacement)
    return text


def _significance_cell(comparisons: Sequence[PairedComparison], system: str, reference: str) -> str:
    """Render the significance cell for one system against the reference arm.

    Args:
        comparisons: The corrected family.
        system: The row's system.
        reference: The arm every row is tested against.

    Returns:
        The marker and Holm-adjusted p, or an em dash where the comparison is the
        reference against itself or was not computed.
    """
    if system == reference:
        return "--"
    for comparison in comparisons:
        if {comparison.system_a, comparison.system_b} == {system, reference}:
            if comparison.p_adjusted is None:
                return "--"
            # `marker` returns a unicode em dash for the uncorrected case, which is
            # already excluded above; the replace guards a future third caller rather
            # than letting a stray U+2014 reach pdflatex.
            return f"{comparison.marker.replace('—', '--')} ({comparison.p_adjusted:.4f})"
    return "--"


def main_table_latex(
    result: AggregateResult,
    *,
    stream: str = "balanced",
    metrics: Sequence[str] = MAIN_TABLE_METRICS,
    reference: str = "S1",
    label: str = "tab:main",
) -> str:
    """Emit the main results table.

    Args:
        result: The aggregation.
        stream: Which stream to tabulate. Streams are never pooled.
        metrics: Columns, in order.
        reference: The arm the significance column tests against.
        label: LaTeX label.

    Returns:
        A complete ``table`` environment. Every declared system appears: one that produced
        no numbers gets a row of em dashes and its absence is named in the caption, because
        a results table that quietly contains fewer rows than the matrix declares is the
        one way a null result disappears.
    """
    headline = metrics[0] if metrics else HEADLINE_METRIC
    family = result.comparisons.get(f"{stream}/{headline}", ())
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{ll" + "r" * len(metrics) + "l}",
        r"\toprule",
        "System & Role & "
        + " & ".join(_latex_escape(m) for m in metrics)
        + r" & vs "
        + _latex_escape(reference)
        + r" \\",
        r"\midrule",
    ]

    absent: list[str] = []
    for spec in all_systems():
        cells: list[str] = []
        any_value = False
        for metric in metrics:
            summary = result.summaries.get((stream, metric, spec.system_id))
            cells.append(_mean_std_cell(summary))
            any_value = any_value or summary is not None
        if not any_value:
            absent.append(spec.system_id)
        lines.append(
            f"{_latex_escape(spec.system_id)} & {_latex_escape(spec.role)} & "
            + " & ".join(cells)
            + " & "
            + _significance_cell(family, spec.system_id, reference)
            + r" \\"
        )

    n_missing = len(result.missing)
    caption = (
        f"Main results on the {stream} stream. "
        r"$^{\dagger}$ marks a single-seed row: the seed policy is three seeds on "
        r"S1, S2, A1 and B7 -- the systems carrying the central claim -- and one seed "
        r"elsewhere, stated rather than hidden. A dash is a number that does not exist, "
        r"never a zero. Significance is Wilcoxon signed-rank on paired per-case values, "
        r"Holm--Bonferroni corrected across every pairwise comparison of this metric on "
        r"this stream."
    )
    if absent:
        caption += (
            " Systems with no numbers in this table: "
            + ", ".join(_latex_escape(s) for s in absent)
            + f" ({n_missing} declared runs produced no metrics; see RESULTS.md for each "
            "one's reason)."
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def ablation_table_latex(
    result: AggregateResult,
    *,
    stream: str = "balanced",
    metrics: Sequence[str] = ABLATION_TABLE_METRICS,
    reference: str = "S1",
    label: str = "tab:ablation",
) -> str:
    """Emit the ablation table: every A-row against S1, with what each one varies.

    Args:
        result: The aggregation.
        stream: Which stream.
        metrics: Columns.
        reference: The arm ablations are read against.
        label: LaTeX label.

    Returns:
        A complete ``table`` environment, S1 first and then each ablation with the axis it
        moves named in its own column, so a reader never has to consult the registry to
        learn what ``A4`` changed.
    """
    varies = {
        "A1": "graph tokens deranged (control)",
        "A2": "MLP encoder, no message passing",
        "A3_F3": "linear projector",
        "A3_F4": "perceiver projector",
        "A4": "encoder unfrozen",
        "A5": "inference guard off",
        "A6": "second base model (Qwen3-8B)",
        "B8": "gate off (F1)",
    }
    headline = metrics[0] if metrics else HEADLINE_METRIC
    family = result.comparisons.get(f"{stream}/{headline}", ())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lll" + "r" * len(metrics) + "l}",
        r"\toprule",
        "System & Varies & "
        + " & ".join(_latex_escape(m) for m in metrics)
        + r" & vs "
        + _latex_escape(reference)
        + r" \\",
        r"\midrule",
    ]
    order = [reference, *sorted(varies)]
    for system_id in order:
        if system_id == reference:
            label_text = r"\emph{(reference)}"
        else:
            label_text = _latex_escape(varies.get(system_id, ""))
        cells = [
            _mean_std_cell(result.summaries.get((stream, metric, system_id))) for metric in metrics
        ]
        lines.append(
            f"{_latex_escape(system_id)} & {label_text} & "
            + " & ".join(cells)
            + " & "
            + _significance_cell(family, system_id, reference)
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Ablations, each read against "
        + _latex_escape(reference)
        + r". Every row differs from the reference on exactly the named axis. "
        r"A1 is the sanity control and is the comparison Gate 8 rests on; A5 requires no "
        r"training run, being S1's checkpoint decoded with the guard disabled, so the "
        r"guarded and unguarded results are two rows and neither is the headline alone.}",
        r"\label{" + label + "}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def taxonomy_table_latex(
    result: AggregateResult,
    *,
    stream: str = "balanced",
    classes: Sequence[str] = TAXONOMY_CLASSES,
    label: str = "tab:taxonomy",
) -> str:
    """Emit the hallucination taxonomy table: per-narrative rate by class and system.

    Args:
        result: The aggregation.
        stream: Which stream.
        classes: The classes, in taxonomy order.
        label: LaTeX label.

    Returns:
        A complete ``table`` environment. The Critical classes are marked, because H4, H6
        and H7 are what the Critical Error Rate is built from and a reader scanning the
        table should see which columns carry that weight.
    """
    critical = {"H4", "H6", "H7"}
    header = " & ".join((rf"\textbf{{{c}}}" if c in critical else c) for c in classes)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(classes) + "}",
        r"\toprule",
        f"System & {header}" + r" \\",
        r"\midrule",
    ]
    for spec in all_systems():
        rates = result.taxonomy.get((stream, spec.system_id))
        cells = [_fmt(rates.get(c) if rates else None, places=4) for c in classes]
        lines.append(_latex_escape(spec.system_id) + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Hallucination taxonomy: per-narrative rate of each class. Bold columns "
        r"(H4, H6, H7) are the Critical classes -- a fabricated entity, an invented "
        r"regulatory citation, a fabricated causal claim -- and are what the Critical "
        r"Error Rate counts. H9, omission of an exculpatory fact, is the only class "
        r"detected by absence rather than assertion, and is the one the Bronze template "
        r"triggers (0.9179), which is why the template floor is not uniformly 1.0.}",
        r"\label{" + label + "}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def write_outputs(
    result: AggregateResult,
    out_dir: Path | str,
    *,
    streams: Iterable[str] = ("balanced",),
) -> dict[str, Path]:
    """Write the tidy table, the JSON aggregation and every LaTeX table.

    Args:
        result: The aggregation.
        out_dir: Destination directory.
        streams: Streams to emit tables for. Each gets its own file; they are never
            pooled into one table.

    Returns:
        Artifact name to the path written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    frame = result.frame()
    csv_path = out / "metrics_long.csv"
    frame.to_csv(csv_path, index=False)
    written["metrics_long_csv"] = csv_path

    written["aggregate_json"] = write_json(out / "aggregate.json", result.to_dict())

    for stream in streams:
        for name, builder in (
            ("main", main_table_latex),
            ("ablation", ablation_table_latex),
            ("taxonomy", taxonomy_table_latex),
        ):
            path = out / f"table_{name}_{stream}.tex"
            path.write_text(builder(result, stream=stream), encoding="utf-8")
            written[f"{name}_{stream}_tex"] = path
    return written


def missing_report(result: AggregateResult) -> str:
    """Render the missing-run report that goes into RESULTS.md.

    Args:
        result: The aggregation.

    Returns:
        A markdown table, one row per declared run with no metrics, or a line saying the
        matrix is complete. This is invariant 7 in a function: the absences are a
        deliverable, not a diagnostic.
    """
    if not result.missing:
        return "Every declared run produced metrics."
    lines = [
        "| System | Seed | Status | Reason |",
        "|---|---:|---|---|",
    ]
    for run in result.missing:
        spec_note = ""
        try:
            spec_note = get_system(run.system).role
        except KeyError:
            spec_note = "unregistered"
        lines.append(
            f"| `{run.system}` ({spec_note}) | {run.seed} | {run.status or '—'} | {run.reason} |"
        )
    return "\n".join(lines)

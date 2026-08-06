"""Do the automatic faithfulness metrics agree with human judgement? Built now, run later.

**This correlation is the validation of the automatic metric.** Layer 2 measures whether
a narrative's claims hold against a fact record; whether that is the same thing as an
investigator judging the narrative factually correct is an empirical question, and until
it is answered the metric is a proposal. Phase 12 collects the human ratings; this module
is what turns them into the figure and the number the paper leans on.

Written in Phase 10, before any human rating exists, deliberately. An analysis specified
after seeing the ratings is an analysis chosen to fit them — which metric to correlate,
whether to pool raters or average them, whether to drop the cases raters disagreed on.
Those choices are made here, in the open, against no data.

**Both coefficients, always.** Spearman is primary — the human scale is ordinal, five
points, and its intervals are not equal — and Pearson is reported beside it because a
large gap between the two is itself a finding about the metric's shape. Neither is
reported without its confidence interval: a correlation of 0.61 on forty cases and one on
four hundred are different claims, and the point estimate alone does not distinguish them.

**Per-metric, not just for the headline.** A figure showing Zero-Hallucination Rate
correlating with human factual-correctness is the one the paper wants; a table showing
that BLEU does not is the one that justifies the whole three-layer design, and it costs
nothing to compute both.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CorrelationResult",
    "MetricValidationReport",
    "correlate",
    "fisher_interval",
    "validate_metrics",
]

#: Below this many paired observations a correlation is reported but flagged. Twenty is
#: not a threshold for significance, it is the point below which a Fisher interval spans
#: most of the plausible range and the point estimate should not be read on its own.
_MIN_RELIABLE_N = 20

#: Fisher's transform needs four observations; below that the interval is degenerate.
_MIN_FOR_FISHER = 4

#: A correlation is undefined below three pairs. The honest output there is no number.
_MIN_FOR_CORRELATION = 3


def fisher_interval(r: float, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Return a confidence interval for a correlation via Fisher's z transform.

    Args:
        r: The correlation coefficient.
        n: Paired observations.
        confidence: Nominal coverage.

    Returns:
        ``(lo, hi)``, clipped to [-1, 1]. Returns ``(r, r)`` for fewer than four
        observations or a perfect correlation, where the transform is undefined —
        degenerate rather than raising, because a metric with three human ratings should
        appear in the table flagged, not crash the report.

    Raises:
        ValueError: If ``confidence`` is not in (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n < _MIN_FOR_FISHER or abs(r) >= 1.0:
        return (r, r)

    from scipy.stats import norm

    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    critical = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    lo, hi = math.tanh(z - critical * se), math.tanh(z + critical * se)
    return (max(-1.0, lo), min(1.0, hi))


@dataclass(frozen=True)
class CorrelationResult:
    """One automatic metric against one human rating dimension.

    Attributes:
        metric: The automatic metric.
        human_dimension: The rating dimension, e.g. ``"factual_correctness"``.
        n: Paired observations.
        spearman: Spearman's rho. The primary coefficient: the human scale is ordinal.
        spearman_p: Its p-value.
        spearman_ci: 95% interval on rho, via Fisher's z.
        pearson: Pearson's r.
        pearson_p: Its p-value.
        pearson_ci: 95% interval on r.
        reliable: False when ``n`` is below :data:`_MIN_RELIABLE_N`, in which case the
            point estimate should not be read without the interval.
    """

    metric: str
    human_dimension: str
    n: int
    spearman: float
    spearman_p: float
    spearman_ci: tuple[float, float]
    pearson: float
    pearson_p: float
    pearson_ci: tuple[float, float]
    reliable: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-serialisable mapping.

        Returns:
            Every field, intervals as two-element lists.
        """
        return {
            "metric": self.metric,
            "human_dimension": self.human_dimension,
            "n": self.n,
            "spearman": self.spearman,
            "spearman_p": self.spearman_p,
            "spearman_ci": list(self.spearman_ci),
            "pearson": self.pearson,
            "pearson_p": self.pearson_p,
            "pearson_ci": list(self.pearson_ci),
            "reliable": self.reliable,
        }


def correlate(
    metric: str,
    human_dimension: str,
    automatic: Sequence[float],
    human: Sequence[float],
    *,
    confidence: float = 0.95,
) -> CorrelationResult:
    """Correlate one automatic metric against one human dimension.

    Args:
        metric: The automatic metric's name.
        human_dimension: The rating dimension's name.
        automatic: The metric's per-case values.
        human: The human ratings, index-aligned.
        confidence: Nominal coverage for both intervals.

    Returns:
        Both coefficients with their p-values and intervals.

    Raises:
        ValueError: If the sequences are different lengths, or hold fewer than three
            pairs. Three is the floor at which a correlation is defined at all; below it
            the honest output is no number rather than a number nobody should read.
        ImportError: If scipy is not installed.
    """
    if len(automatic) != len(human):
        raise ValueError(f"correlation needs paired values; got {len(automatic)} and {len(human)}")
    if len(automatic) < _MIN_FOR_CORRELATION:
        raise ValueError(
            f"{metric} against {human_dimension} has {len(automatic)} pairs; a "
            "correlation needs at least three"
        )

    from scipy.stats import pearsonr, spearmanr

    rho_result = spearmanr(list(automatic), list(human))
    rho, rho_p = float(rho_result.statistic), float(rho_result.pvalue)
    r_result = pearsonr(list(automatic), list(human))
    r, r_p = float(r_result.statistic), float(r_result.pvalue)

    # A constant metric column — every case scoring 1.0 — makes both coefficients NaN.
    # That is a real and likely outcome here (Bronze is 100% supported by construction),
    # and it is reported as zero correlation with a degenerate interval rather than as a
    # NaN that would propagate into the figure.
    rho = 0.0 if math.isnan(rho) else rho
    r = 0.0 if math.isnan(r) else r
    rho_p = 1.0 if math.isnan(rho_p) else rho_p
    r_p = 1.0 if math.isnan(r_p) else r_p

    n = len(automatic)
    return CorrelationResult(
        metric=metric,
        human_dimension=human_dimension,
        n=n,
        spearman=rho,
        spearman_p=rho_p,
        spearman_ci=fisher_interval(rho, n, confidence=confidence),
        pearson=r,
        pearson_p=r_p,
        pearson_ci=fisher_interval(r, n, confidence=confidence),
        reliable=n >= _MIN_RELIABLE_N,
    )


@dataclass(frozen=True)
class MetricValidationReport:
    """Every automatic metric against every human dimension.

    Attributes:
        results: The correlations.
        n_cases: Cases with both an automatic score and a human rating.
        headline: The correlation the paper leads with — Zero-Hallucination Rate against
            factual correctness — or None when it could not be computed.
        missing: Metric/dimension pairs that could not be correlated, with the reason.
    """

    results: tuple[CorrelationResult, ...]
    n_cases: int
    headline: CorrelationResult | None
    missing: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping.

        Returns:
            Every correlation, the headline, and what was skipped.
        """
        return {
            "n_cases": self.n_cases,
            "headline": self.headline.to_dict() if self.headline is not None else None,
            "results": [result.to_dict() for result in self.results],
            "missing": dict(sorted(self.missing.items())),
        }

    def markdown(self) -> str:
        """Render the correlations as a markdown table.

        Returns:
            One row per metric/dimension pair, ordered by |rho| descending, with the
            unreliable rows flagged.
        """
        if not self.results:
            return "_No human ratings are available yet; the correlation analysis has not run._"
        lines = [
            "| Metric | Human dimension | n | Spearman rho [95% CI] | p | Pearson r [95% CI] |",
            "|---|---|---:|---|---:|---|",
        ]
        for result in sorted(self.results, key=lambda x: -abs(x.spearman)):
            flag = "" if result.reliable else " ⚠"
            lines.append(
                f"| `{result.metric}` | {result.human_dimension} | {result.n}{flag} | "
                f"{result.spearman:+.3f} [{result.spearman_ci[0]:+.3f}, "
                f"{result.spearman_ci[1]:+.3f}] | {result.spearman_p:.4f} | "
                f"{result.pearson:+.3f} [{result.pearson_ci[0]:+.3f}, "
                f"{result.pearson_ci[1]:+.3f}] |"
            )
        lines.append("")
        lines.append(
            "⚠ marks a correlation on fewer than 20 pairs; read the interval, not the point."
        )
        return "\n".join(lines)


def validate_metrics(
    automatic: Mapping[str, Mapping[str, float]],
    human: Mapping[str, Mapping[str, float]],
    *,
    headline_metric: str = "zero_hallucination",
    headline_dimension: str = "factual_correctness",
    confidence: float = 0.95,
) -> MetricValidationReport:
    """Correlate every automatic metric against every human dimension.

    Runs the moment Phase 12's ratings land, over whatever they contain — the dimensions
    are read from the data rather than hard-coded, so a protocol that adds a rating scale
    does not need this module changed.

    Args:
        automatic: Metric name to ``{case_id: value}``.
        human: Rating dimension to ``{case_id: rating}``. Ratings averaged across
            annotators by the caller, because how to combine annotators is a protocol
            decision that belongs with the protocol.
        headline_metric: The metric the paper's figure uses.
        headline_dimension: The rating dimension it is plotted against.
        confidence: Nominal coverage for the intervals.

    Returns:
        The report. A pair that cannot be correlated — too few shared cases — appears in
        ``missing`` with the reason rather than being dropped silently.
    """
    results: list[CorrelationResult] = []
    missing: dict[str, str] = {}
    all_cases: set[str] = set()

    for metric_name, metric_values in sorted(automatic.items()):
        for dimension, ratings in sorted(human.items()):
            shared = sorted(set(metric_values) & set(ratings))
            all_cases.update(shared)
            key = f"{metric_name}/{dimension}"
            if len(shared) < _MIN_FOR_CORRELATION:
                missing[key] = f"only {len(shared)} cases have both a score and a rating"
                continue
            results.append(
                correlate(
                    metric_name,
                    dimension,
                    [metric_values[case] for case in shared],
                    [ratings[case] for case in shared],
                    confidence=confidence,
                )
            )

    headline = next(
        (
            result
            for result in results
            if result.metric == headline_metric and result.human_dimension == headline_dimension
        ),
        None,
    )
    return MetricValidationReport(
        results=tuple(results),
        n_cases=len(all_cases),
        headline=headline,
        missing=missing,
    )

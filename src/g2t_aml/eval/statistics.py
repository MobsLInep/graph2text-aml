"""Confidence intervals, paired tests, multiplicity correction and effect sizes.

Three rules are enforced by the shapes here rather than by remembering them.

**Never a single-seed number.** :func:`seed_summary` takes a mapping keyed by seed and
refuses a mapping of one, so a table cell that quotes a mean has more than one run behind
it. A system trained once is reported as a system trained once, with its variance marked
unknown.

**Never a p-value without an effect size.** :class:`PairedComparison` carries both, has no
constructor path that produces one without the other, and :func:`compare_systems` computes
them together. A significant difference with a negligible Cliff's δ is a fact about the
sample size, and reporting the p alone invites the reader to conclude the opposite.

**Never an uncorrected family.** With sixteen systems there are 120 pairwise comparisons
per metric; at alpha = 0.05 six of them are expected to be "significant" under a complete
null. :func:`holm_bonferroni` corrects across the family, and
:func:`compare_systems` defines the family explicitly — every pairwise comparison of one
metric on one test stream — so what was corrected over is written down rather than
inferred. See D-079.

**Cliff's δ is the primary effect size.** The per-case metrics here are bounded rates,
several are near their ceiling, and Zero-Hallucination is a per-narrative binary; none of
those is remotely normal, and Cohen's d on a bounded skewed variable overstates. Cohen's d
is computed and reported anyway, because reviewers ask for it, and the two are shown
together so a disagreement between them is visible.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "CLIFF_BANDS",
    "Interval",
    "PairedComparison",
    "SeedSummary",
    "bootstrap_ci",
    "cliffs_delta",
    "cliffs_delta_band",
    "cohens_d",
    "compare_systems",
    "holm_bonferroni",
    "paired_bootstrap_ci",
    "publication_table",
    "seed_summary",
    "wilcoxon_signed_rank",
]

#: Resamples for every bootstrap interval. Ten thousand: enough that the interval's
#: endpoints are stable to the third decimal, which is the precision the tables quote.
BOOTSTRAP_RESAMPLES = 10_000

#: Below this many observations a spread is undefined and the statistic that needs it
#: returns a degenerate value rather than raising.
_MIN_FOR_SPREAD = 2

#: The conventional significance level. Named so the marker thresholds and the default
#: familywise rate cannot drift apart.
_ALPHA = 0.05

#: Romano, Kromrey and Coraggio's thresholds for |δ|. Quoted rather than invented so the
#: bands in the paper are the ones a reader can look up.
CLIFF_BANDS: tuple[tuple[float, str], ...] = (
    (0.147, "negligible"),
    (0.33, "small"),
    (0.474, "medium"),
)


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval.

    Attributes:
        point: The statistic on the observed sample.
        lo: Lower bound.
        hi: Upper bound.
        confidence: The nominal coverage, e.g. 0.95.
        n: Sample size the interval was computed from.
        n_resamples: Bootstrap resamples used.
    """

    point: float
    lo: float
    hi: float
    confidence: float = 0.95
    n: int = 0
    n_resamples: int = BOOTSTRAP_RESAMPLES

    @property
    def excludes_zero(self) -> bool:
        """Report whether the interval lies entirely on one side of zero.

        Returns:
            True when both bounds share a sign, which for a difference interval is the
            statement the gate tables make.
        """
        return (self.lo > 0 and self.hi > 0) or (self.lo < 0 and self.hi < 0)

    def format(self, places: int = 4) -> str:
        """Render as ``point [lo, hi]``.

        Args:
            places: Decimal places.

        Returns:
            The formatted interval.
        """
        return f"{self.point:.{places}f} [{self.lo:.{places}f}, {self.hi:.{places}f}]"

    def to_dict(self) -> dict[str, Any]:
        """Return the interval as a JSON-serialisable mapping.

        Returns:
            Every field.
        """
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "confidence": self.confidence,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "excludes_zero": self.excludes_zero,
        }


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[npt.NDArray[np.float64]], float] | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 42,
) -> Interval:
    """Compute a percentile bootstrap confidence interval.

    Percentile rather than BCa: the per-case metrics here are bounded in [0, 1] and
    several sit against a boundary, where BCa's acceleration term is estimated from a
    jackknife that is itself degenerate. The percentile interval is the conservative
    choice and is what the resampling count is set high enough to make stable.

    Args:
        values: The per-case observations.
        statistic: What to compute on each resample. The mean when omitted.
        n_resamples: How many resamples.
        confidence: Nominal coverage.
        seed: Seeds the resampling, so an interval is reproducible.

    Returns:
        The interval. Over a single observation the interval is that observation twice —
        degenerate, correctly so, and ``n`` says why.

    Raises:
        ValueError: If ``values`` is empty, or ``confidence`` is not in (0, 1).
    """
    if not values:
        raise ValueError("a bootstrap interval needs at least one observation")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    fn = statistic if statistic is not None else (lambda a: float(np.mean(a)))
    sample = np.asarray(values, dtype=float)
    point = fn(sample)
    if sample.size == 1:
        return Interval(point=point, lo=point, hi=point, confidence=confidence, n=1, n_resamples=0)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(n_resamples, sample.size))
    resampled = np.array([fn(sample[row]) for row in indices], dtype=float)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(resampled, [alpha, 1.0 - alpha])
    return Interval(
        point=point,
        lo=float(lo),
        hi=float(hi),
        confidence=confidence,
        n=int(sample.size),
        n_resamples=n_resamples,
    )


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 42,
) -> Interval:
    """Bootstrap the mean paired difference ``a - b``.

    Paired, resampling *cases* rather than the two arms independently. The arms are
    evaluated on the same cases, so case difficulty is a shared nuisance term; resampling
    independently discards that pairing and inflates the interval by exactly the variance
    the design was set up to remove.

    Args:
        a: One system's per-case values.
        b: The other's, index-aligned.
        n_resamples: How many resamples.
        confidence: Nominal coverage.
        seed: Seeds the resampling.

    Returns:
        The interval on the mean difference.

    Raises:
        ValueError: If the sequences are different lengths or empty.
    """
    if len(a) != len(b):
        raise ValueError(f"a paired interval needs equal lengths; got {len(a)} and {len(b)}")
    differences = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(differences, n_resamples=n_resamples, confidence=confidence, seed=seed)


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Run a two-sided Wilcoxon signed-rank test on paired observations.

    Args:
        a: One system's per-case values.
        b: The other's, index-aligned.

    Returns:
        ``(statistic, p_value)``. When every pair is tied — two systems that scored
        identically on every case — the test is undefined and this returns
        ``(0.0, 1.0)``: no evidence of a difference, which is the correct reading, rather
        than the NaN scipy raises.

    Raises:
        ValueError: If the sequences are different lengths or empty.
        ImportError: If scipy is not installed.
    """
    if len(a) != len(b):
        raise ValueError(f"Wilcoxon needs paired observations; got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("Wilcoxon is undefined over an empty sample")
    if all(float(x) == float(y) for x, y in zip(a, b, strict=True)):
        return 0.0, 1.0

    from scipy.stats import wilcoxon

    result = wilcoxon(list(a), list(b), zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Cliff's δ, the non-parametric effect size.

    δ is ``P(a > b) - P(a < b)`` over all pairs. Unlike Cohen's d it assumes nothing
    about the distributions, which is what makes it right for bounded rates piled against
    a ceiling.

    Args:
        a: One system's values. Need not be paired with ``b`` — δ is defined over the two
            samples, not over their pairing.
        b: The other's.

    Returns:
        δ in [-1, 1], and 0.0 when either sample is empty.
    """
    if not a or not b:
        return 0.0
    left = np.asarray(a, dtype=float)[:, None]
    right = np.asarray(b, dtype=float)[None, :]
    greater = int(np.sum(left > right))
    less = int(np.sum(left < right))
    return (greater - less) / (left.size * right.size)


def cliffs_delta_band(delta: float) -> str:
    """Return the conventional magnitude band for a δ.

    Args:
        delta: The effect size.

    Returns:
        ``"negligible"``, ``"small"``, ``"medium"`` or ``"large"``.
    """
    magnitude = abs(delta)
    for threshold, name in CLIFF_BANDS:
        if magnitude < threshold:
            return name
    return "large"


def cohens_d(a: Sequence[float], b: Sequence[float], *, paired: bool = True) -> float:
    """Compute Cohen's d.

    Args:
        a: One system's values.
        b: The other's, index-aligned when ``paired``.
        paired: Whether to compute the paired d (mean difference over the standard
            deviation of the differences) or the independent-samples d over the pooled
            standard deviation. Paired by default, because every comparison in this
            harness is on the same cases.

    Returns:
        d, and 0.0 when the relevant standard deviation is zero — two systems with
        identical per-case values have no effect between them, and returning an infinity
        would put one in a results table.

    Raises:
        ValueError: If ``paired`` and the sequences are different lengths, or either is
            empty.
    """
    if not a or not b:
        raise ValueError("Cohen's d is undefined over an empty sample")
    if paired:
        if len(a) != len(b):
            raise ValueError(f"paired d needs equal lengths; got {len(a)} and {len(b)}")
        differences = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
        if len(differences) < _MIN_FOR_SPREAD:
            return 0.0
        spread = statistics.stdev(differences)
        return statistics.fmean(differences) / spread if spread > 0 else 0.0

    if len(a) < _MIN_FOR_SPREAD or len(b) < _MIN_FOR_SPREAD:
        return 0.0
    var_a, var_b = (
        statistics.variance(list(map(float, a))),
        statistics.variance(list(map(float, b))),
    )
    n_a, n_b = len(a), len(b)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled <= 0:
        return 0.0
    return (statistics.fmean(map(float, a)) - statistics.fmean(map(float, b))) / pooled


def holm_bonferroni(
    p_values: Sequence[float], *, alpha: float = _ALPHA
) -> tuple[list[float], list[bool]]:
    """Apply the Holm-Bonferroni step-down correction to a family of p-values.

    Holm rather than Bonferroni because it is uniformly more powerful at the same
    familywise error rate, and rather than Benjamini—Hochberg because the claims here are
    individual — "S1 beats B7 on Zero-Hallucination" is asserted on its own, not as one
    of a set whose false-discovery *proportion* is controlled.

    Args:
        p_values: The family's p-values, in any order.
        alpha: Familywise error rate.

    Returns:
        ``(adjusted, reject)``, both in the input order. Adjusted values are the
        step-down monotone-enforced values, capped at 1.0, so they can be compared
        against ``alpha`` directly and match ``statsmodels.stats.multitest.multipletests``
        under ``method="holm"``.

    Raises:
        ValueError: If any p-value is outside [0, 1].
    """
    for p in p_values:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value {p} is outside [0, 1]")
    n = len(p_values)
    if n == 0:
        return [], []

    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted_sorted: list[float] = []
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * p_values[index]
        running = max(running, candidate)
        adjusted_sorted.append(min(running, 1.0))

    adjusted = [0.0] * n
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted, [p <= alpha for p in adjusted]


@dataclass(frozen=True)
class SeedSummary:
    """A metric across seeds for one system.

    Attributes:
        system: The arm.
        metric: The metric name.
        per_seed: Seed to that seed's value.
        mean: Mean across seeds.
        std: Sample standard deviation across seeds. None for a single seed, which is
            what makes "variance unknown" visible in a table instead of printing 0.0.
        n_seeds: How many seeds.
    """

    system: str
    metric: str
    per_seed: Mapping[int, float]
    mean: float
    std: float | None
    n_seeds: int

    @property
    def single_seed(self) -> bool:
        """Report whether this summary rests on one run.

        Returns:
            True when only one seed contributed. A True here is a caveat the report
            prints, not a condition it hides.
        """
        return self.n_seeds < _MIN_FOR_SPREAD

    def format(self, places: int = 4) -> str:
        """Render as ``mean ± std``, or as a flagged single-seed value.

        Args:
            places: Decimal places.

        Returns:
            The formatted summary.
        """
        if self.std is None:
            return f"{self.mean:.{places}f} (1 seed)"
        return f"{self.mean:.{places}f} ± {self.std:.{places}f}"

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a JSON-serialisable mapping.

        Returns:
            Every field.
        """
        return {
            "system": self.system,
            "metric": self.metric,
            "per_seed": {str(k): v for k, v in sorted(self.per_seed.items())},
            "mean": self.mean,
            "std": self.std,
            "n_seeds": self.n_seeds,
            "single_seed": self.single_seed,
        }


def seed_summary(system: str, metric: str, per_seed: Mapping[int, float]) -> SeedSummary:
    """Summarise one metric across seeds.

    Args:
        system: The arm.
        metric: The metric name.
        per_seed: Seed to value.

    Returns:
        The summary, with ``std=None`` for a single seed.

    Raises:
        ValueError: If ``per_seed`` is empty.
    """
    if not per_seed:
        raise ValueError(f"no seeds supplied for {system}/{metric}")
    values = [float(v) for v in per_seed.values()]
    return SeedSummary(
        system=system,
        metric=metric,
        per_seed=dict(per_seed),
        mean=statistics.fmean(values),
        std=statistics.stdev(values) if len(values) > 1 else None,
        n_seeds=len(values),
    )


@dataclass(frozen=True)
class PairedComparison:
    """One system against another on one metric, with everything needed to read it.

    There is no way to construct this with a p-value and no effect size, which is the
    point: the two are computed together in :func:`compare_systems` and reported together
    everywhere.

    Attributes:
        metric: The metric compared.
        system_a: The first arm.
        system_b: The second.
        n_cases: Paired cases.
        mean_a: ``system_a``'s mean.
        mean_b: ``system_b``'s mean.
        difference_ci: Bootstrap interval on the mean paired difference ``a - b``.
        statistic: The Wilcoxon statistic.
        p_value: Its two-sided p-value, uncorrected.
        p_adjusted: The Holm-corrected p-value. None until
            :func:`compare_systems` has defined the family.
        cliffs_delta: The non-parametric effect size.
        cliffs_band: Its magnitude band.
        cohens_d: The paired standardised mean difference.
        family: What the correction was applied over, recorded so a reader knows.
    """

    metric: str
    system_a: str
    system_b: str
    n_cases: int
    mean_a: float
    mean_b: float
    difference_ci: Interval
    statistic: float
    p_value: float
    p_adjusted: float | None
    cliffs_delta: float
    cliffs_band: str
    cohens_d: float
    family: str = ""

    @property
    def significant(self) -> bool:
        """Report whether the corrected p-value clears alpha = 0.05.

        Returns:
            True when the Holm-adjusted p is at or below 0.05. False when no correction
            has been applied — an uncorrected comparison is never reported as
            significant, which is what stops a single call site from quietly bypassing
            the family.
        """
        return self.p_adjusted is not None and self.p_adjusted <= _ALPHA

    @property
    def marker(self) -> str:
        """Return the significance marker for a table cell.

        Returns:
            ``"***"``, ``"**"``, ``"*"`` on the corrected p, ``"ns"`` otherwise, and
            ``"—"`` when uncorrected.
        """
        if self.p_adjusted is None:
            return "—"
        for threshold, mark in ((0.001, "***"), (0.01, "**"), (_ALPHA, "*")):
            if self.p_adjusted <= threshold:
                return mark
        return "ns"

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as a JSON-serialisable mapping.

        Returns:
            Every field, with the interval expanded.
        """
        return {
            "metric": self.metric,
            "system_a": self.system_a,
            "system_b": self.system_b,
            "n_cases": self.n_cases,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "difference_ci": self.difference_ci.to_dict(),
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "cliffs_delta": self.cliffs_delta,
            "cliffs_band": self.cliffs_band,
            "cohens_d": self.cohens_d,
            "significant": self.significant,
            "marker": self.marker,
            "family": self.family,
        }


def compare_systems(
    metric: str,
    values_by_system: Mapping[str, Mapping[str, float]],
    *,
    alpha: float = _ALPHA,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    family: str | None = None,
) -> list[PairedComparison]:
    """Compare every pair of systems on one metric, corrected across the family.

    **The family is every pairwise comparison of this metric on this slice** — which is
    what one call to this function produces. Defining it as the function's own output is
    deliberate: a family assembled by a caller across several calls is a family nobody
    can reconstruct from the results file, and with sixteen systems the difference
    between correcting over 120 comparisons and over 15 is the difference between a
    finding and a coincidence.

    Args:
        metric: The metric name, for the records.
        values_by_system: System to ``{case_id: value}``. Comparisons run on the cases
            two systems have in common, so a system evaluated on a subset is compared
            fairly rather than dropped.
        alpha: Familywise error rate.
        n_resamples: Bootstrap resamples per comparison.
        seed: Seeds the bootstrap.
        family: Label recorded on every comparison. Defaults to naming the metric and
            the systems.

    Returns:
        One comparison per unordered pair, sorted by system name, each carrying its
        Holm-adjusted p-value. Empty when fewer than two systems were supplied.
    """
    systems = sorted(values_by_system)
    if len(systems) < _MIN_FOR_SPREAD:
        return []

    label = family or f"{metric} over {len(systems)} systems"
    comparisons: list[PairedComparison] = []
    for i, name_a in enumerate(systems):
        for name_b in systems[i + 1 :]:
            shared = sorted(set(values_by_system[name_a]) & set(values_by_system[name_b]))
            if not shared:
                continue
            a = [float(values_by_system[name_a][case]) for case in shared]
            b = [float(values_by_system[name_b][case]) for case in shared]
            statistic, p_value = wilcoxon_signed_rank(a, b)
            delta = cliffs_delta(a, b)
            comparisons.append(
                PairedComparison(
                    metric=metric,
                    system_a=name_a,
                    system_b=name_b,
                    n_cases=len(shared),
                    mean_a=statistics.fmean(a),
                    mean_b=statistics.fmean(b),
                    difference_ci=paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed),
                    statistic=statistic,
                    p_value=p_value,
                    p_adjusted=None,
                    cliffs_delta=delta,
                    cliffs_band=cliffs_delta_band(delta),
                    cohens_d=cohens_d(a, b, paired=True),
                    family=f"{label} ({len(systems) * (len(systems) - 1) // 2} comparisons)",
                )
            )

    adjusted, _ = holm_bonferroni([c.p_value for c in comparisons], alpha=alpha)
    return [
        PairedComparison(
            metric=c.metric,
            system_a=c.system_a,
            system_b=c.system_b,
            n_cases=c.n_cases,
            mean_a=c.mean_a,
            mean_b=c.mean_b,
            difference_ci=c.difference_ci,
            statistic=c.statistic,
            p_value=c.p_value,
            p_adjusted=p,
            cliffs_delta=c.cliffs_delta,
            cliffs_band=c.cliffs_band,
            cohens_d=c.cohens_d,
            family=c.family,
        )
        for c, p in zip(comparisons, adjusted, strict=True)
    ]


def publication_table(
    metric: str,
    summaries: Sequence[SeedSummary],
    intervals: Mapping[str, Interval],
    comparisons: Sequence[PairedComparison] = (),
    *,
    reference: str | None = None,
    places: int = 4,
) -> str:
    """Emit a markdown table of mean ± std, CI and significance against a reference arm.

    Args:
        metric: The metric being tabulated.
        summaries: One per system, carrying the across-seed mean and standard deviation.
        intervals: System to its bootstrap interval on the metric.
        comparisons: The corrected pairwise comparisons, for the significance column.
        reference: The arm every other row is tested against. The first row's system when
            omitted.
        places: Decimal places.

    Returns:
        A markdown table, systems ordered by mean descending. Every cell that has no
        number prints an em dash rather than a zero.
    """
    if not summaries:
        return f"_No systems were scored on {metric}._"

    ordered = sorted(summaries, key=lambda s: -s.mean)
    baseline = reference or ordered[0].system
    lookup: dict[frozenset[str], PairedComparison] = {
        frozenset((c.system_a, c.system_b)): c for c in comparisons
    }

    lines = [
        f"| System | {metric} (mean ± std) | 95% CI | vs `{baseline}` | Cliff's δ | p (Holm) |",
        "|---|---|---|---|---|---|",
    ]
    for summary in ordered:
        interval = intervals.get(summary.system)
        comparison = lookup.get(frozenset((summary.system, baseline)))
        if summary.system == baseline or comparison is None:
            marker, delta, p_text = "—", "—", "—"
        else:
            marker = comparison.marker
            delta = f"{comparison.cliffs_delta:+.3f} ({comparison.cliffs_band})"
            p_text = "—" if comparison.p_adjusted is None else f"{comparison.p_adjusted:.4f}"
        lines.append(
            f"| `{summary.system}` | {summary.format(places)} | "
            f"{interval.format(places) if interval else '—'} | {marker} | {delta} | {p_text} |"
        )
    lines.append("")
    lines.append(
        f"Significance is Holm-Bonferroni-corrected across the family of "
        f"{len(comparisons)} pairwise comparisons of {metric} over {len(summaries)} "
        "systems; markers are `***` p≤0.001, `**` p≤0.01, `*` p≤0.05, `ns` otherwise. "
        "Effect sizes accompany every p-value by construction."
    )
    return "\n".join(lines)

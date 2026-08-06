"""What the study found: agreement, ranks, times, edits, and whether the metric predicts them.

This is where the study is unblinded, and it is the only module that reads a
:class:`~g2t_aml.human.study_design.BlindKey`. Everything upstream — the design, the
interface, the response store — is written so that the join from item id to system happens
here, once, in code that a reviewer can read in one sitting.

**The reporting order is a decision.** A Likert table is what every paper submits and it is
the weakest evidence in this file. The order is:

1. **Time-to-usable-draft**, against the Bronze template baseline. A measured reduction in
   investigator drafting time is a deployment claim.
2. **Edit distance** from the presented draft to the rater's filable version, same
   baseline. It says how much of the draft survived contact with an expert.
3. **Would you file this after review** — a decision rather than an opinion.
4. The five ordinal scales, and never without an agreement statistic beside them.
5. **The correlation between the automatic Layer-2 metric and human factual correctness**,
   which is what licenses every automatic number in the rest of the paper.

**Friedman's test needs complete blocks, and this design has none.** A rater never sees the
same case twice (:mod:`g2t_aml.human.study_design` explains why), so no case carries one
observation per system and the obvious blocking variable is unavailable. Two tests are
therefore reported rather than one, and they are not alternatives:

- :func:`friedman_test` over **rater-blocked means** — each rater contributes one mean per
  system, giving a complete blocks-by-treatments matrix. This is the brief's test and the
  primary. Its weakness is that it has as many blocks as raters, so with a panel of eight
  it is underpowered and will fail to reject far more readily than it will mislead.
- :func:`durbin_test` over the **case-blocked incomplete design**, which is Friedman's
  generalisation to exactly this situation and uses every item-level observation rather
  than a per-rater average. It reduces algebraically to Friedman when the design is
  complete, which is how its implementation is tested.

Reporting only the first would throw away most of the data; reporting only the second would
answer a question the brief did not ask. Where they disagree, that disagreement is the
finding and is reported as one.

**Nothing here invents a tolerance.** The ordinal Krippendorff difference function, the
Friedman tie correction and the Nemenyi critical values are the published ones, and each is
tested against a published worked example rather than against a second implementation of
itself — the same rule that governs :mod:`g2t_aml.facts` under invariant 1, for the same
reason: two implementations by one author agree on their shared misreading.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from g2t_aml.human.study_design import BlindKey, DesignError
from g2t_aml.human.study_ui import LIKERT_DIMENSIONS, RatingResponse

__all__ = [
    "NEMENYI_Q",
    "CorrelationResult",
    "DurbinResult",
    "FriedmanResult",
    "NemenyiResult",
    "StudyAnalysis",
    "analyse_study",
    "critical_difference_diagram",
    "durbin_test",
    "friedman_test",
    "intra_rater_reliability",
    "krippendorff_alpha_ordinal",
    "load_blind_key",
    "nemenyi_posthoc",
    "normalised_levenshtein",
    "pearson_with_ci",
    "rater_effects",
    "spearman_with_ci",
]

#: Studentised range statistic divided by sqrt(2), indexed by number of systems, for the
#: Nemenyi post-hoc test. Demsar (2006), Table 5. The division by sqrt(2) is already applied
#: -- these are the ``q_alpha`` that go straight into the critical-difference formula, which
#: is the form the source table publishes and the form every implementation disagrees about.
NEMENYI_Q: dict[str, dict[int, float]] = {
    "0.05": {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    },
    "0.10": {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
    },
}

#: Smallest number of coders on a unit for it to contribute to an agreement coefficient.
#: A unit rated once carries no information about agreement and is excluded rather than
#: counted as perfect.
_MIN_PAIRABLE = 2

#: Smallest n for a correlation, for a Fisher-z interval, and for a signed-rank test to be
#: able to reach significance at all. The last is the one that matters in practice: at five
#: pairs the smallest attainable two-sided p is 0.0625, so a p-value from it can never
#: support a claim and printing one invites it to be read as though it could.
_MIN_CORRELATION_N = 3
_MIN_FISHER_N = 4
_MIN_WILCOXON_PAIRS = 6

#: Smallest panel for a random-effect variance component to be worth estimating.
_MIN_RATERS_FOR_MIXED = 5

#: Guards for the continued-fraction evaluations: the floor that keeps a denominator away
#: from zero, and the convergence tolerances.
_TINY = 1e-30
_CF_TOL = 1e-14
_BETA_TOL = 1e-12

#: Bootstrap resamples for every interval in this module. 10,000 rather than 1,000 because
#: an agreement coefficient's sampling distribution is skewed at small panel sizes and the
#: interval's tails are what a reviewer reads.
_N_BOOTSTRAP = 10_000


def normalised_levenshtein(a: str, b: str) -> float:
    """Return character-level Levenshtein distance normalised to ``[0, 1]``.

    The study's second behavioural measure. Normalised by the length of the longer string,
    so 0 means the rater changed nothing and 1 means nothing of the draft survived, and a
    long narrative edited lightly does not outscore a short one rewritten wholesale.

    Character-level rather than token-level deliberately. A rater fixing ``9,435`` to
    ``9,434.82`` has made a small, real correction that token-level distance would score as
    a whole-token substitution — the same weight as replacing a sentence. On this corpus
    the corrections that matter most are numeric (D-054), so the finer granularity is the
    one that measures the thing.

    Computed with the two-row dynamic program: narratives run to a few thousand characters
    and the full matrix is unnecessary.

    Args:
        a: The presented narrative.
        b: The rater's corrected version.

    Returns:
        Distance in ``[0, 1]``. Zero when both are empty.
    """
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1] / max(len(a), len(b))


def _ordinal_deltas(
    values: Sequence[int], marginals: Mapping[int, float]
) -> dict[tuple[int, int], float]:
    """Return the squared ordinal difference for every pair of observed values.

    Krippendorff's ordinal metric. Unlike the interval metric it depends on the *marginal
    frequencies* of the observed values, not just their numeric distance: the gap between
    two adjacent categories is wide when few observations fall between them and narrow when
    many do. This is why an ordinal alpha cannot be computed from the interval one by
    rescaling, and why a Likert scale analysed as interval data reports a different number.

    Args:
        values: The distinct observed values, ascending.
        marginals: Value to its marginal frequency in the coincidence matrix.

    Returns:
        ``(c, k)`` to squared difference, for every ordered pair.
    """
    deltas: dict[tuple[int, int], float] = {}
    for c in values:
        for k in values:
            lo, hi = (c, k) if c <= k else (k, c)
            between = sum(marginals.get(g, 0.0) for g in values if lo <= g <= hi)
            deltas[c, k] = (between - (marginals.get(c, 0.0) + marginals.get(k, 0.0)) / 2.0) ** 2
    return deltas


def _alpha_ordinal(units: Sequence[Sequence[int | None]]) -> float:
    """Return Krippendorff's ordinal alpha for a units-by-coders matrix.

    Args:
        units: One row per unit, one column per coder, None where a coder did not rate it.

    Returns:
        Alpha. 1.0 when observed disagreement is zero, which includes the degenerate case
        of every value being identical.

    Raises:
        ValueError: If no unit carries at least two values, so nothing is pairable.
    """
    pairable = [[v for v in row if v is not None] for row in units]
    pairable = [row for row in pairable if len(row) >= _MIN_PAIRABLE]
    if not pairable:
        raise ValueError(
            "no unit was rated by two or more raters, so there is no agreement to measure"
        )

    coincidences: dict[tuple[int, int], float] = defaultdict(float)
    for row in pairable:
        weight = 1.0 / (len(row) - 1)
        for i, c in enumerate(row):
            for j, k in enumerate(row):
                if i != j:
                    coincidences[c, k] += weight

    marginals: dict[int, float] = defaultdict(float)
    for (c, _), n in coincidences.items():
        marginals[c] += n
    total = sum(marginals.values())
    if total <= 1:
        return 1.0

    values = sorted(marginals)
    deltas = _ordinal_deltas(values, marginals)

    observed = sum(n * deltas[c, k] for (c, k), n in coincidences.items())
    expected = sum(
        marginals[c] * marginals[k] * deltas[c, k] for c in values for k in values if c != k
    )
    if expected == 0:
        return 1.0
    do = observed / total
    de = expected / (total * (total - 1))
    return 1.0 - do / de


@dataclass(frozen=True)
class CorrelationResult:
    """A correlation with its interval and the n it was computed on.

    Attributes:
        statistic: Spearman's rho or Pearson's r.
        p_value: Two-sided p, from the t approximation.
        ci_low: Lower bound of the 95% Fisher-z interval.
        ci_high: Upper bound.
        n: Pairs.
        method: ``"spearman"`` or ``"pearson"``.
    """

    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    n: int
    method: str

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised result.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "method": self.method,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "n": self.n,
        }


def _rank(values: Sequence[float]) -> list[float]:
    """Return midranks, averaging ties.

    Args:
        values: The values to rank.

    Returns:
        Ranks, 1-based, ties averaged.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    return ranks


def _fisher_ci(r: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a Fisher z-transformed confidence interval for a correlation.

    Args:
        r: The correlation.
        n: Pairs.
        confidence: Coverage.

    Returns:
        ``(low, high)``, clamped to ``[-1, 1]``. ``(nan, nan)`` when n < 4, where the
        transform has no standard error.
    """
    if n < _MIN_FISHER_N or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    # 1.959964 is the two-sided normal quantile at 95%; recomputed rather than hardcoded
    # for other coverages via the inverse error function.
    crit = math.sqrt(2) * _erfinv(confidence)
    lo, hi = z - crit * se, z + crit * se
    return (math.tanh(lo), math.tanh(hi))


def _erfinv(x: float) -> float:
    """Return the inverse error function by Newton refinement of a rational start.

    Used only for normal quantiles in confidence intervals. Implemented here rather than
    taken from scipy because this module runs in the base environment, where scipy is not
    installed -- ``scipy`` is in the ``eval`` extra.

    Args:
        x: Argument in ``(-1, 1)``.

    Returns:
        ``erfinv(x)``.
    """
    a = 0.147
    ln = math.log(1 - x * x)
    t1 = 2 / (math.pi * a) + ln / 2
    y = math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), x)
    for _ in range(3):
        err = math.erf(y) - x
        y -= err / (2 / math.sqrt(math.pi) * math.exp(-y * y))
    return y


def _t_sf(t: float, df: int) -> float:
    """Return the two-sided tail probability of Student's t.

    Args:
        t: The statistic.
        df: Degrees of freedom.

    Returns:
        Two-sided p in ``[0, 1]``.
    """
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Return the regularised incomplete beta function ``I_x(a, b)``.

    Continued-fraction evaluation (Lentz). Present for the same reason as :func:`_erfinv`:
    the p-values in this module must be computable without the ``eval`` extra.

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        x: Upper limit in ``[0, 1]``.

    Returns:
        The regularised incomplete beta.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)

    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        d = _TINY if abs(d) < _TINY else d
        d = 1.0 / d
        c = 1.0 + numerator / c
        c = _TINY if abs(c) < _TINY else c
        f *= c * d
        if abs(1.0 - c * d) < _BETA_TOL:
            break
    return front * (f - 1.0)


def _chi2_sf(x: float, df: int) -> float:
    """Return the upper tail of the chi-square distribution.

    Args:
        x: The statistic.
        df: Degrees of freedom.

    Returns:
        ``P(X > x)``.
    """
    if x <= 0:
        return 1.0
    return _gammaincc(df / 2.0, x / 2.0)


def _gammaincc(a: float, x: float) -> float:
    """Return the regularised upper incomplete gamma ``Q(a, x)``.

    Args:
        a: Shape.
        x: Lower limit.

    Returns:
        ``Q(a, x)``.
    """
    if x < a + 1:
        # Series for the lower tail, then complement.
        term = 1.0 / a
        total = term
        n = a
        for _ in range(1000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * _CF_TOL:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for the upper tail.
    b = x + 1 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = _TINY if abs(d) < _TINY else d
        c = b + an / c
        c = _TINY if abs(c) < _TINY else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_TOL:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def krippendorff_alpha_ordinal(
    units: Sequence[Sequence[int | None]],
    *,
    n_bootstrap: int = _N_BOOTSTRAP,
    seed: int = 20260805,
) -> tuple[float, float, float]:
    """Return ordinal Krippendorff's alpha with a bootstrap confidence interval.

    The agreement statistic the brief requires beside every Likert mean, and ordinal rather
    than nominal because the scales are ordered: a 6-against-7 disagreement is not the same
    event as a 1-against-7, and nominal alpha scores them identically.

    The interval is a **unit bootstrap** — units are resampled with replacement and alpha
    recomputed — rather than a resample of individual ratings. Resampling ratings would
    break the unit structure that alpha is defined over and produce an interval for a
    quantity that is not alpha.

    Args:
        units: One row per unit, one column per coder, None where a coder did not rate it.
        n_bootstrap: Resamples.
        seed: Bootstrap seed, recorded in the report.

    Returns:
        ``(alpha, ci_low, ci_high)`` at 95%. The bounds are ``nan`` when fewer than two
        units are pairable, where a bootstrap distribution is meaningless.

    Raises:
        ValueError: If no unit carries at least two values.
    """
    alpha = _alpha_ordinal(units)
    usable = [row for row in units if sum(1 for v in row if v is not None) >= _MIN_PAIRABLE]
    if len(usable) < _MIN_PAIRABLE or n_bootstrap <= 0:
        return (alpha, float("nan"), float("nan"))

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        sample = [usable[rng.randrange(len(usable))] for _ in range(len(usable))]
        try:
            draws.append(_alpha_ordinal(sample))
        except (ValueError, ZeroDivisionError):
            continue
    if len(draws) < _MIN_PAIRABLE:
        return (alpha, float("nan"), float("nan"))
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(math.ceil(0.975 * (len(draws) - 1)))]
    return (alpha, lo, hi)


def intra_rater_reliability(responses: Sequence[RatingResponse]) -> dict[str, Any]:
    """Return per-rater reproducibility, measured on the planted repeat items.

    The check that reports on the raters rather than on the systems. A panel whose members
    cannot reproduce their own judgement two hours later has no business having its
    between-system differences believed, and this is the only measurement in the study that
    can detect that.

    Pairs each repeat with its original by ``(rater_id, case_id)`` — the design guarantees a
    repeat shares both with exactly one earlier item, and validates that guarantee.

    Args:
        responses: Every response, repeats included.

    Returns:
        A mapping with ``n_pairs``, ``per_dimension`` (dimension to ordinal alpha over the
        first-versus-second matrix), ``mean_absolute_difference`` per dimension, and
        ``would_file_agreement``. Empty counts when the study carried no repeats.
    """
    originals: dict[tuple[str, str], RatingResponse] = {}
    repeats: list[RatingResponse] = []
    for response in sorted(responses, key=lambda r: r.position):
        key = (response.rater_id, response.case_id)
        if response.is_repeat:
            repeats.append(response)
        elif key not in originals:
            originals[key] = response

    pairs = [
        (originals[r.rater_id, r.case_id], r)
        for r in repeats
        if (r.rater_id, r.case_id) in originals
    ]
    if not pairs:
        return {"n_pairs": 0, "per_dimension": {}, "mean_absolute_difference": {}}

    per_dimension: dict[str, float] = {}
    mean_abs: dict[str, float] = {}
    for dimension in LIKERT_DIMENSIONS:
        units = [[int(getattr(a, dimension.key)), int(getattr(b, dimension.key))] for a, b in pairs]
        try:
            per_dimension[dimension.key] = round(_alpha_ordinal(units), 4)
        except (ValueError, ZeroDivisionError):
            per_dimension[dimension.key] = float("nan")
        mean_abs[dimension.key] = round(sum(abs(u[0] - u[1]) for u in units) / len(units), 4)
    agree = sum(1 for a, b in pairs if a.would_file == b.would_file) / len(pairs)
    return {
        "n_pairs": len(pairs),
        "per_dimension": per_dimension,
        "mean_absolute_difference": mean_abs,
        "would_file_agreement": round(agree, 4),
    }


@dataclass(frozen=True)
class FriedmanResult:
    """Friedman's test over a complete blocks-by-treatments matrix.

    Attributes:
        statistic: The tie-corrected chi-square statistic.
        p_value: Upper-tail p with ``k - 1`` degrees of freedom.
        n_blocks: Blocks. Raters, in this study.
        n_treatments: Systems.
        mean_ranks: System to its mean rank. Lower is better when the metric is one where
            lower is better, and the caller is responsible for having oriented it — see
            :func:`analyse_study`, which ranks time and edit distance ascending and the
            Likert dimensions descending, so a low mean rank always means "better".
    """

    statistic: float
    p_value: float
    n_blocks: int
    n_treatments: int
    mean_ranks: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised result.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "test": "friedman",
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "n_blocks": self.n_blocks,
            "n_treatments": self.n_treatments,
            "mean_ranks": {k: round(v, 4) for k, v in self.mean_ranks.items()},
        }


def friedman_test(
    blocks: Mapping[str, Mapping[str, float]], systems: Sequence[str]
) -> FriedmanResult:
    """Run Friedman's test over complete blocks.

    Args:
        blocks: Block id to a mapping of system to that block's value. Every block must
            carry every system.
        systems: The systems, fixing the column order.

    Returns:
        The result.

    Raises:
        ValueError: If a block is incomplete, or fewer than two blocks or systems are
            present. An incomplete block is not dropped silently: Friedman on a subset of
            blocks is a different test on a different population, and the caller has to
            decide that rather than discover it in a footnote.
    """
    if len(systems) < _MIN_PAIRABLE:
        raise ValueError("Friedman needs at least two systems")
    rows: list[list[float]] = []
    for block_id, values in sorted(blocks.items()):
        missing = set(systems) - set(values)
        if missing:
            raise ValueError(
                f"block {block_id!r} is missing {sorted(missing)}. Friedman requires "
                "complete blocks; use durbin_test for an incomplete design."
            )
        rows.append([float(values[s]) for s in systems])
    n, k = len(rows), len(systems)
    if n < _MIN_PAIRABLE:
        raise ValueError("Friedman needs at least two blocks")

    ranked = [_rank(row) for row in rows]
    rank_sums = [sum(r[j] for r in ranked) for j in range(k)]

    # Tie correction: without it a matrix with ties reports an inflated statistic, and
    # rater-blocked means tie readily once two systems perform alike.
    ties = 0.0
    for row in ranked:
        counts: dict[float, int] = defaultdict(int)
        for value in row:
            counts[value] += 1
        ties += sum(c**3 - c for c in counts.values())

    base = 12.0 / (n * k * (k + 1)) * sum(s * s for s in rank_sums) - 3.0 * n * (k + 1)
    correction = 1.0 - ties / (n * k * (k * k - 1))
    statistic = base / correction if correction > 0 else float("nan")
    return FriedmanResult(
        statistic=statistic,
        p_value=_chi2_sf(statistic, k - 1),
        n_blocks=n,
        n_treatments=k,
        mean_ranks={s: rank_sums[j] / n for j, s in enumerate(systems)},
    )


@dataclass(frozen=True)
class DurbinResult:
    """Durbin's test: Friedman's generalisation to a balanced incomplete block design.

    Attributes:
        statistic: The Durbin statistic.
        p_value: Upper-tail chi-square p with ``t - 1`` degrees of freedom.
        n_blocks: Blocks. Cases, in this study.
        n_treatments: Systems.
        k_per_block: Systems appearing in each block.
        r_per_treatment: Blocks each system appears in.
        mean_ranks: System to its mean within-block rank.
    """

    statistic: float
    p_value: float
    n_blocks: int
    n_treatments: int
    k_per_block: int
    r_per_treatment: int
    mean_ranks: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised result.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "test": "durbin",
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "n_blocks": self.n_blocks,
            "n_treatments": self.n_treatments,
            "k_per_block": self.k_per_block,
            "r_per_treatment": self.r_per_treatment,
            "mean_ranks": {k: round(v, 4) for k, v in self.mean_ranks.items()},
        }


def durbin_test(blocks: Mapping[str, Mapping[str, float]], systems: Sequence[str]) -> DurbinResult:
    """Run Durbin's test over a balanced incomplete block design.

    The test this study's design actually calls for. Cases are the blocks, and a case
    carries only the systems some rater happened to see it under, so blocks are incomplete
    by construction. Durbin ranks within each block over the systems present, which is what
    makes an incomplete block usable at all.

    Reduces to Friedman when every block carries every system, and
    ``tests/unit/test_study_analysis.py`` asserts that equivalence on a complete matrix
    rather than trusting the algebra.

    Args:
        blocks: Block id to a mapping of the systems present in it and their values.
        systems: Every system in the design.

    Returns:
        The result.

    Raises:
        ValueError: If the design is not balanced — blocks of differing size, or systems
            appearing in differing numbers of blocks. Durbin's null distribution assumes
            both, and applying it to an unbalanced design produces a p-value for a test
            that was not run.
    """
    usable = {b: dict(v) for b, v in blocks.items() if len(v) >= _MIN_PAIRABLE}
    if len(usable) < _MIN_PAIRABLE:
        raise ValueError("Durbin needs at least two blocks carrying two or more systems")

    sizes = {len(v) for v in usable.values()}
    if len(sizes) != 1:
        raise ValueError(
            f"blocks have differing sizes {sorted(sizes)}; Durbin's null distribution "
            "assumes a balanced design. Restrict to the balanced subset explicitly."
        )
    k = sizes.pop()

    appearances: dict[str, int] = defaultdict(int)
    for values in usable.values():
        for system in values:
            appearances[system] += 1
    present = [s for s in systems if appearances[s] > 0]
    counts = set(appearances.values())
    if len(counts) != 1:
        raise ValueError(
            f"systems appear in differing numbers of blocks {sorted(counts)}; Durbin's "
            "null distribution assumes a balanced design"
        )
    r = counts.pop()
    t = len(present)
    b = len(usable)
    if t < _MIN_PAIRABLE:
        raise ValueError("Durbin needs at least two systems")

    rank_sums: dict[str, float] = defaultdict(float)
    for values in usable.values():
        ordered = sorted(values)
        ranks = _rank([values[s] for s in ordered])
        for system, rank in zip(ordered, ranks, strict=True):
            rank_sums[system] += rank

    correction = r * (k + 1) / 2.0
    numerator = 12.0 * (t - 1) * sum((rank_sums[s] - correction) ** 2 for s in present)
    denominator = r * t * (k - 1) * (k + 1)
    statistic = numerator / denominator if denominator else float("nan")
    return DurbinResult(
        statistic=statistic,
        p_value=_chi2_sf(statistic, t - 1),
        n_blocks=b,
        n_treatments=t,
        k_per_block=k,
        r_per_treatment=r,
        mean_ranks={s: rank_sums[s] / r for s in present},
    )


@dataclass(frozen=True)
class NemenyiResult:
    """The Nemenyi post-hoc comparison and its critical difference.

    Attributes:
        critical_difference: Two systems differ significantly when their mean ranks differ
            by at least this much.
        alpha: The level the critical value was taken at.
        mean_ranks: System to mean rank.
        significant_pairs: Pairs whose rank difference exceeds the critical difference.
        n_blocks: Blocks the ranks were computed over, which sets the test's power.
    """

    critical_difference: float
    alpha: str
    mean_ranks: dict[str, float]
    significant_pairs: tuple[tuple[str, str], ...]
    n_blocks: int

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised result.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "test": "nemenyi",
            "alpha": self.alpha,
            "critical_difference": round(self.critical_difference, 4),
            "n_blocks": self.n_blocks,
            "mean_ranks": {k: round(v, 4) for k, v in self.mean_ranks.items()},
            "significant_pairs": [list(p) for p in self.significant_pairs],
        }


def nemenyi_posthoc(
    mean_ranks: Mapping[str, float], n_blocks: int, *, alpha: str = "0.05"
) -> NemenyiResult:
    """Return the Nemenyi critical difference and the pairs that clear it.

    Args:
        mean_ranks: System to mean rank, from :func:`friedman_test` or :func:`durbin_test`.
        n_blocks: How many blocks the ranks came from.
        alpha: ``"0.05"`` or ``"0.10"``.

    Returns:
        The result.

    Raises:
        ValueError: If alpha is unknown or the number of systems is outside the published
            critical-value table. Extrapolating the studentised range beyond the table is
            how implementations quietly disagree, so this refuses instead.
    """
    if alpha not in NEMENYI_Q:
        raise ValueError(f"no critical values tabulated for alpha={alpha!r}")
    k = len(mean_ranks)
    if k not in NEMENYI_Q[alpha]:
        raise ValueError(
            f"no tabulated Nemenyi critical value for {k} systems; the published table "
            f"covers {min(NEMENYI_Q[alpha])} to {max(NEMENYI_Q[alpha])}"
        )
    if n_blocks < 1:
        raise ValueError("Nemenyi needs at least one block")

    cd = NEMENYI_Q[alpha][k] * math.sqrt(k * (k + 1) / (6.0 * n_blocks))
    names = sorted(mean_ranks)
    pairs = tuple(
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if abs(mean_ranks[a] - mean_ranks[b]) >= cd
    )
    return NemenyiResult(
        critical_difference=cd,
        alpha=alpha,
        mean_ranks=dict(mean_ranks),
        significant_pairs=pairs,
        n_blocks=n_blocks,
    )


def critical_difference_diagram(result: NemenyiResult, path: Path, *, title: str = "") -> Path:
    """Draw a Demsar critical-difference diagram.

    Systems on a rank axis, best on the left, with a bar joining any group whose mean ranks
    are all within the critical difference of each other — the standard way to show "these
    are not distinguishable at this sample size" without a wall of pairwise p-values.

    Args:
        result: The post-hoc result.
        path: Where to write the figure. Parent directories are created.
        title: Optional figure title.

    Returns:
        The path written.

    Raises:
        RuntimeError: If matplotlib is not installed, naming the extra.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the critical-difference diagram needs matplotlib: uv sync --extra graph"
        ) from exc

    ordered = sorted(result.mean_ranks.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ordered]
    ranks = [r for _, r in ordered]
    lo, hi = math.floor(min(ranks)), math.ceil(max(ranks))

    # Cliques: maximal runs of adjacent systems spanned by the critical difference. Computed
    # before the layout because the number of them sets how much room is needed below the
    # axis -- with the bars drawn at a fixed offset they collide with the lowest system's
    # label, which is what the first version of this figure did.
    cliques: list[tuple[int, int]] = []
    for i in range(len(ranks)):
        j = i
        while j + 1 < len(ranks) and ranks[j + 1] - ranks[i] < result.critical_difference:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))

    n = len(names)
    top = n + 1.0
    clique_gap = 0.34
    floor = 0.55 - clique_gap * max(0, len(cliques) - 1) - 0.3

    fig, ax = plt.subplots(figsize=(7.5, 1.9 + 0.42 * n + clique_gap * len(cliques)))
    ax.set_xlim(lo - 0.25, hi + 0.2)
    ax.set_ylim(floor, top + 1.5)
    ax.set_yticks([])
    for side in ("left", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("mean rank (lower is better)")
    ax.xaxis.set_label_position("top")

    for i, (name, rank) in enumerate(ordered):
        y = n - i
        ax.plot([rank, rank], [y, top], color="0.4", linewidth=0.8)
        ax.plot([rank, lo - 0.2], [y, y], color="0.4", linewidth=0.8)
        ax.text(lo - 0.23, y, f"{name} ({rank:.2f})", ha="right", va="center", fontsize=9)

    for depth, (i, j) in enumerate(cliques):
        y = 0.55 - clique_gap * depth
        ax.plot([ranks[i] - 0.04, ranks[j] + 0.04], [y, y], color="black", linewidth=3.5)

    # The CD ruler sits above the axis, clear of the tick labels and of the title.
    ruler = top + 1.05
    ax.plot([lo, lo + result.critical_difference], [ruler, ruler], color="black", linewidth=1.5)
    for end in (lo, lo + result.critical_difference):
        ax.plot([end, end], [ruler - 0.09, ruler + 0.09], color="black", linewidth=1.5)
    ax.text(
        lo + result.critical_difference / 2,
        ruler + 0.16,
        f"CD = {result.critical_difference:.2f} (alpha={result.alpha}, N={result.n_blocks})",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    if title:
        ax.set_title(title, fontsize=11, pad=46)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def pearson_with_ci(x: Sequence[float], y: Sequence[float]) -> CorrelationResult:
    """Return Pearson's r with a Fisher-z confidence interval.

    Args:
        x: First variable.
        y: Second variable, same length.

    Returns:
        The result.

    Raises:
        ValueError: If the sequences differ in length or carry fewer than three pairs.
    """
    return _correlate(x, y, "pearson")


def spearman_with_ci(x: Sequence[float], y: Sequence[float]) -> CorrelationResult:
    """Return Spearman's rho with a Fisher-z confidence interval.

    The primary of the two for validating the automatic metric. Spearman rather than
    Pearson leads because a Likert scale is ordinal and the automatic score is a bounded
    rate: the relationship between them has no reason to be linear, and rho asks the
    question that matters — does the metric order narratives the way an expert does.

    Args:
        x: First variable.
        y: Second variable, same length.

    Returns:
        The result.

    Raises:
        ValueError: If the sequences differ in length or carry fewer than three pairs.
    """
    return _correlate(_rank(x), _rank(y), "spearman")


def _correlate(x: Sequence[float], y: Sequence[float], method: str) -> CorrelationResult:
    """Return a product-moment correlation with its interval and p-value.

    Args:
        x: First variable, already ranked when the method is Spearman.
        y: Second variable.
        method: Recorded on the result.

    Returns:
        The result.

    Raises:
        ValueError: If the sequences differ in length or carry fewer than three pairs.
    """
    if len(x) != len(y):
        raise ValueError(f"correlation needs equal-length inputs, got {len(x)} and {len(y)}")
    n = len(x)
    if n < _MIN_CORRELATION_N:
        raise ValueError(f"correlation needs at least three pairs, got {n}")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return CorrelationResult(float("nan"), float("nan"), float("nan"), float("nan"), n, method)
    r = sxy / math.sqrt(sxx * syy)
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        p = _t_sf(t, n - 2)
    lo, hi = _fisher_ci(r, n)
    return CorrelationResult(r, p, lo, hi, n, method)


def rater_effects(
    responses: Sequence[RatingResponse],
    systems_by_item: Mapping[str, str],
    dimension: str,
) -> dict[str, Any]:
    """Fit a random-intercept model with rater as the random effect.

    Answers "is the between-system difference still there once each rater's own severity is
    accounted for?". Raters differ in how they use a scale — one person's 5 is another's 7 —
    and with an incomplete design that severity is not automatically balanced across
    systems.

    Requires ``statsmodels`` (the ``eval`` extra) and a panel large enough to estimate a
    variance component. Both are checked, and when either fails the function returns a
    result saying so rather than raising: a missing mixed-effects model is a caveat on the
    study, not a reason for the analysis to stop.

    Args:
        responses: Every non-repeat response.
        systems_by_item: Item id to system, from the blind key.
        dimension: Which rating to model.

    Returns:
        A mapping with ``fitted`` (bool), and when fitted the fixed effects, their standard
        errors and p-values, plus the rater variance component. When not fitted, ``reason``
        says why.
    """
    rows = [
        (r.rater_id, systems_by_item[r.item_id], float(getattr(r, dimension)))
        for r in responses
        if not r.is_repeat and r.item_id in systems_by_item
    ]
    n_raters = len({r for r, _, _ in rows})
    if n_raters < _MIN_RATERS_FOR_MIXED:
        return {
            "fitted": False,
            "reason": (
                f"{n_raters} raters is too few to estimate a variance component; a random "
                "effect fitted on this many groups reports a number whose standard error "
                "is not interpretable"
            ),
        }
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return {
            "fitted": False,
            "reason": "statsmodels is not installed; it is in the `eval` extra",
        }

    frame = pd.DataFrame(rows, columns=["rater", "system", "value"])
    try:
        model = smf.mixedlm("value ~ C(system)", frame, groups=frame["rater"]).fit()
    except Exception as exc:  # convergence failures are a caveat, not a crash
        return {"fitted": False, "reason": f"mixed model did not converge: {exc}"}

    return {
        "fitted": True,
        "dimension": dimension,
        "n_observations": len(rows),
        "n_raters": n_raters,
        "fixed_effects": {k: round(float(v), 4) for k, v in model.params.items()},
        "standard_errors": {k: round(float(v), 4) for k, v in model.bse.items()},
        "p_values": {k: round(float(v), 6) for k, v in model.pvalues.items()},
        "rater_variance": round(float(model.cov_re.iloc[0, 0]), 6),
    }


def load_blind_key(path: Path) -> BlindKey:
    """Read the blind key.

    Deliberately lives here and not in :mod:`g2t_aml.human.study_design`: the design module
    is imported by the rating interface, and a key loader in it would put unblinding one
    attribute access away from a rater's session.

    Args:
        path: The key file.

    Returns:
        The key.

    Raises:
        DesignError: If the file is missing or malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise DesignError(f"no blind key at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DesignError(f"{path}: malformed JSON: {exc}") from exc
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise DesignError(f"{path}: no assignments block; this is not a blind key")
    return BlindKey(
        assignments={str(k): str(v) for k, v in assignments.items()},
        salt=str(payload.get("salt", "")),
    )


@dataclass(frozen=True)
class StudyAnalysis:
    """Everything the study found, in reporting order.

    Attributes:
        n_responses: Ratings analysed, repeats excluded.
        n_raters: Raters contributing.
        systems: The arms, in registry order.
        timing: Time-to-usable-draft per system, and paired tests against the baseline.
        edit_distance: Normalised edit distance per system, same shape.
        would_file: Filing rate per system.
        likert: Per-dimension per-system means, each with its agreement statistic.
        agreement: Inter-rater ordinal alpha per dimension, with bootstrap intervals.
        intra_rater: Reproducibility from the repeat items.
        omnibus: Friedman and Durbin results per metric.
        posthoc: Nemenyi results per metric.
        metric_validation: Automatic-versus-human correlations.
        rater_model: The mixed-effects result, or why it was not fitted.
        baseline: The system every paired test is against.
        warnings: Anything that qualifies a number above. Never empty in a real study.
    """

    n_responses: int
    n_raters: int
    systems: tuple[str, ...]
    timing: dict[str, Any]
    edit_distance: dict[str, Any]
    would_file: dict[str, Any]
    likert: dict[str, Any]
    agreement: dict[str, Any]
    intra_rater: dict[str, Any]
    omnibus: dict[str, Any]
    posthoc: dict[str, Any]
    metric_validation: dict[str, Any]
    rater_model: dict[str, Any]
    baseline: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised analysis.

        Returns:
            A JSON-serialisable mapping.
        """
        return {
            "n_responses": self.n_responses,
            "n_raters": self.n_raters,
            "systems": list(self.systems),
            "baseline": self.baseline,
            "timing": self.timing,
            "edit_distance": self.edit_distance,
            "would_file": self.would_file,
            "likert": self.likert,
            "agreement": self.agreement,
            "intra_rater": self.intra_rater,
            "omnibus": self.omnibus,
            "posthoc": self.posthoc,
            "metric_validation": self.metric_validation,
            "rater_model": self.rater_model,
            "warnings": list(self.warnings),
        }


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or nan for an empty sequence.

    Args:
        values: The values.

    Returns:
        The mean.
    """
    return sum(values) / len(values) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    """Return the median, or nan for an empty sequence.

    Args:
        values: The values.

    Returns:
        The median.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _per_rater_means(
    responses: Sequence[RatingResponse],
    systems_by_item: Mapping[str, str],
    value: Any,
) -> dict[str, dict[str, float]]:
    """Return each rater's mean of some quantity under each system.

    The rater-blocked matrix Friedman is run over, and the unit the paired baseline tests
    use. Pairing by rater rather than by case is forced by the design: no case is rated
    under two systems by the same person, so a case-level pairing does not exist.

    Args:
        responses: Non-repeat responses.
        systems_by_item: Item id to system.
        value: Callable taking a response and returning the quantity.

    Returns:
        Rater id to system to mean.
    """
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for response in responses:
        system = systems_by_item.get(response.item_id)
        if system is not None:
            buckets[response.rater_id][system].append(float(value(response)))
    return {r: {s: _mean(v) for s, v in by_system.items()} for r, by_system in buckets.items()}


def _paired_against_baseline(
    per_rater: Mapping[str, Mapping[str, float]], systems: Sequence[str], baseline: str
) -> dict[str, Any]:
    """Run a paired Wilcoxon of each system against the baseline, over raters.

    Args:
        per_rater: Rater to system to that rater's mean.
        systems: The systems.
        baseline: The system to compare against.

    Returns:
        System to a mapping with ``n_pairs``, ``mean_difference``, ``median_difference``,
        ``statistic`` and ``p_value``. Systems with fewer than six pairs report their
        differences and a null p-value: Wilcoxon at n < 6 cannot reach p < 0.05 whatever
        the data, and printing a p-value from it invites it to be read as evidence.
    """
    from g2t_aml.eval.statistics import wilcoxon_signed_rank

    out: dict[str, Any] = {}
    for system in systems:
        if system == baseline:
            continue
        pairs = [
            (values[system], values[baseline])
            for values in per_rater.values()
            if system in values and baseline in values
        ]
        differences = [a - b for a, b in pairs]
        entry: dict[str, Any] = {
            "n_pairs": len(pairs),
            "mean_difference": round(_mean(differences), 4) if differences else float("nan"),
            "median_difference": round(_median(differences), 4) if differences else float("nan"),
        }
        if len(pairs) >= _MIN_WILCOXON_PAIRS:
            statistic, p = wilcoxon_signed_rank([a for a, _ in pairs], [b for _, b in pairs])
            entry["statistic"] = round(float(statistic), 4)
            entry["p_value"] = round(float(p), 6)
        else:
            entry["p_value"] = None
            entry["note"] = (
                f"{len(pairs)} pairs: a signed-rank test cannot reach significance below "
                "six pairs regardless of the data, so no p-value is reported"
            )
        out[system] = entry
    return out


def _omnibus_for(
    per_rater: Mapping[str, Mapping[str, float]],
    per_case: Mapping[str, Mapping[str, float]],
    systems: Sequence[str],
    warnings: list[str],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Friedman and Durbin for one metric, and Nemenyi on whichever succeeded.

    Args:
        per_rater: Rater-blocked matrix.
        per_case: Case-blocked matrix.
        systems: The systems.
        warnings: Appended to when a test could not be run.
        label: Metric name, for the warning text.

    Returns:
        ``(omnibus, posthoc)`` mappings.
    """
    omnibus: dict[str, Any] = {}
    posthoc: dict[str, Any] = {}
    try:
        friedman = friedman_test(per_rater, systems)
        omnibus["friedman"] = friedman.to_dict()
        posthoc["friedman"] = nemenyi_posthoc(friedman.mean_ranks, friedman.n_blocks).to_dict()
    except ValueError as exc:
        warnings.append(f"{label}: Friedman not run ({exc})")
    balanced, note = _balanced_subset(per_case)
    try:
        durbin = durbin_test(balanced, systems)
        entry = durbin.to_dict()
        if note:
            entry["subset"] = note
        omnibus["durbin"] = entry
        posthoc["durbin"] = nemenyi_posthoc(durbin.mean_ranks, durbin.n_blocks).to_dict()
        if note:
            warnings.append(f"{label}: Durbin ran on a subset -- {note}")
    except ValueError as exc:
        warnings.append(f"{label}: Durbin not run ({exc})")
    return omnibus, posthoc


def _balanced_subset(
    per_case: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, dict[str, float]], str]:
    """Return the largest balanced sub-design Durbin's test can legally be applied to.

    A greedy allocator spreads cases thinly, so cases end up carrying differing numbers of
    systems and the raw case-blocked matrix is not a balanced incomplete block design.
    Durbin's null distribution assumes balance in both directions, so rather than refuse the
    test outright this takes the modal block size, keeps only the cases of that size, and
    then drops the least-represented systems until every remaining system appears in an
    equal number of blocks.

    Dropping is done in the open: the returned note names how many blocks and which systems
    survived, and the caller records it on the result and in the study's warnings. A subset
    analysis reported as though it were the whole is the failure mode here, not the subset
    itself.

    Args:
        per_case: Case id to the systems it carries and their values.

    Returns:
        ``(subset, note)``. The note is empty when the input was already balanced.
    """
    if not per_case:
        return ({}, "")
    sizes = Counter(len(v) for v in per_case.values())
    modal = max(sizes, key=lambda s: (sizes[s] * s, s))
    subset = {c: dict(v) for c, v in per_case.items() if len(v) == modal}

    while subset:
        appearances = Counter(s for v in subset.values() for s in v)
        if len(set(appearances.values())) <= 1:
            break
        fewest = min(appearances, key=lambda s: (appearances[s], s))
        subset = {
            c: {s: x for s, x in v.items() if s != fewest}
            for c, v in subset.items()
            if fewest not in v or len(v) > 1
        }
        subset = {c: v for c, v in subset.items() if len(v) == modal}

    dropped = len(per_case) - len(subset)
    if not dropped:
        return (subset, "")
    kept = sorted({s for v in subset.values() for s in v})
    return (
        subset,
        f"{len(subset)} of {len(per_case)} case-blocks carried exactly {modal} systems "
        f"and an equal number of blocks each; systems compared: {kept}",
    )


def analyse_study(  # noqa: PLR0912, PLR0915 -- a report assembler: one branch and one
    # block per reported section, and splitting it would scatter the reporting order that
    # the module docstring argues for across five functions.
    responses: Sequence[RatingResponse],
    key: BlindKey,
    *,
    systems: Sequence[str],
    baseline: str = "Bronze",
    automatic_scores: Mapping[str, float] | None = None,
) -> StudyAnalysis:
    """Unblind the responses and produce the whole analysis.

    Args:
        responses: Every response, repeats included. Repeats are used for intra-rater
            reliability and excluded from everything else.
        key: The blind key.
        systems: The arms, in registry order.
        baseline: The system every paired test compares against. The Bronze template, whose
            drafting time is the number a deployment claim is made against.
        automatic_scores: Optional item id to that narrative's automatic Layer-2 factual
            score, for the metric-validation correlation. When absent, that section reports
            that it was not computed rather than being omitted.

    Returns:
        The analysis.

    Raises:
        DesignError: If a response names an item the key does not cover.
    """
    unknown = {r.item_id for r in responses} - set(key.assignments)
    if unknown:
        raise DesignError(
            f"{len(unknown)} responses name items absent from the blind key, e.g. "
            f"{sorted(unknown)[:3]}. The key and the responses are from different builds."
        )
    systems_by_item = key.assignments
    main = [r for r in responses if not r.is_repeat]
    warnings: list[str] = []

    server_timed = sum(1 for r in main if r.timing_source == "server")
    if server_timed:
        warnings.append(
            f"{server_timed} of {len(main)} responses were timed by the server clock, which "
            "cannot subtract time spent with the tab hidden. Their times are upper bounds."
        )

    def edit(r: RatingResponse) -> float:
        return normalised_levenshtein(r.presented_narrative, r.corrected_narrative)

    per_rater_time = _per_rater_means(main, systems_by_item, lambda r: r.seconds_to_usable_draft)
    per_rater_edit = _per_rater_means(main, systems_by_item, edit)

    by_case_system: dict[str, dict[str, list[RatingResponse]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for response in main:
        system = systems_by_item.get(response.item_id)
        if system is not None:
            by_case_system[response.case_id][system].append(response)

    def case_matrix(value: Any) -> dict[str, dict[str, float]]:
        return {
            case_id: {s: _mean([float(value(r)) for r in rs]) for s, rs in by_system.items()}
            for case_id, by_system in by_case_system.items()
        }

    per_system: dict[str, list[RatingResponse]] = defaultdict(list)
    for response in main:
        system = systems_by_item.get(response.item_id)
        if system is not None:
            per_system[system].append(response)

    timing = {
        "per_system": {
            s: {
                "n": len(rs),
                "mean_seconds": round(_mean([r.seconds_to_usable_draft for r in rs]), 2),
                "median_seconds": round(_median([r.seconds_to_usable_draft for r in rs]), 2),
            }
            for s, rs in sorted(per_system.items())
        },
        "paired_vs_baseline": _paired_against_baseline(per_rater_time, systems, baseline),
    }
    edit_distance = {
        "per_system": {
            s: {
                "n": len(rs),
                "mean": round(_mean([edit(r) for r in rs]), 4),
                "median": round(_median([edit(r) for r in rs]), 4),
            }
            for s, rs in sorted(per_system.items())
        },
        "paired_vs_baseline": _paired_against_baseline(per_rater_edit, systems, baseline),
    }
    would_file = {
        s: {
            "n": len(rs),
            "rate": round(sum(1 for r in rs if r.would_file) / len(rs), 4) if rs else float("nan"),
        }
        for s, rs in sorted(per_system.items())
    }

    # Inter-rater agreement is computed over (case, system) cells rated by two or more
    # raters. In an incomplete design most cells are rated once, so this is deliberately a
    # measurement on the doubly-rated subset and the n is reported beside every alpha.
    agreement: dict[str, Any] = {}
    likert: dict[str, Any] = {}
    for dimension in LIKERT_DIMENSIONS:
        units: list[list[int | None]] = []
        for by_system in by_case_system.values():
            for rs in by_system.values():
                if len(rs) >= _MIN_PAIRABLE:
                    units.append([int(getattr(r, dimension.key)) for r in rs])
        if units:
            alpha, lo, hi = krippendorff_alpha_ordinal(units)
            agreement[dimension.key] = {
                "alpha_ordinal": round(alpha, 4),
                "ci_low": None if math.isnan(lo) else round(lo, 4),
                "ci_high": None if math.isnan(hi) else round(hi, 4),
                "n_units": len(units),
            }
        else:
            agreement[dimension.key] = {
                "alpha_ordinal": None,
                "n_units": 0,
                "note": (
                    "no (case, system) cell was rated by two raters, so inter-rater "
                    "agreement is not estimable from this design as run"
                ),
            }
            warnings.append(
                f"{dimension.key}: no doubly-rated cell, so no inter-rater agreement. A "
                "Likert mean without an agreement statistic must not be reported."
            )
        likert[dimension.key] = {
            "per_system": {
                s: {
                    "n": len(rs),
                    "mean": round(_mean([float(getattr(r, dimension.key)) for r in rs]), 3),
                    "median": _median([float(getattr(r, dimension.key)) for r in rs]),
                }
                for s, rs in sorted(per_system.items())
            },
            "agreement": agreement[dimension.key],
        }

    # The aggregate: each response's mean across the five dimensions, then averaged per
    # system. Reported because the brief asks for it, and reported with the caveat it
    # needs -- the five scales measure different things and a narrative can be factually
    # perfect and unfilable (calibration item 2 in the training pack is exactly that), so
    # the mean of them is a summary rather than a quantity anything is true of. The
    # per-dimension table above is what a conclusion should rest on.
    likert["aggregate"] = {
        "per_system": {
            s: {
                "n": len(rs),
                "mean": round(
                    _mean(
                        [_mean([float(getattr(r, d.key)) for d in LIKERT_DIMENSIONS]) for r in rs]
                    ),
                    3,
                ),
            }
            for s, rs in sorted(per_system.items())
        },
        "note": (
            "unweighted mean of the five ordinal dimensions per response, then averaged "
            "per system. A summary, not a measurement: the dimensions are not "
            "commensurable and a draft can score 7 on factual correctness and 1 on "
            "regulatory tone. Read the per-dimension table."
        ),
    }

    omnibus: dict[str, Any] = {}
    posthoc: dict[str, Any] = {}
    for label, per_rater_matrix, value in (
        ("time_to_usable_draft", per_rater_time, lambda r: r.seconds_to_usable_draft),
        ("edit_distance", per_rater_edit, edit),
    ):
        o, p = _omnibus_for(per_rater_matrix, case_matrix(value), systems, warnings, label)
        omnibus[label], posthoc[label] = o, p
    for dimension in LIKERT_DIMENSIONS:
        # Negated so that a low mean rank always means "better", matching time and edit
        # distance, where low is better. A diagram mixing the two orientations is read
        # backwards by half its readers.
        def negated(r: RatingResponse, _k: str = dimension.key) -> float:
            return -float(getattr(r, _k))

        o, p = _omnibus_for(
            _per_rater_means(main, systems_by_item, negated),
            case_matrix(negated),
            systems,
            warnings,
            dimension.key,
        )
        omnibus[dimension.key], posthoc[dimension.key] = o, p

    metric_validation: dict[str, Any] = {}
    if automatic_scores:
        paired = [
            (automatic_scores[r.item_id], float(r.factual_correctness))
            for r in main
            if r.item_id in automatic_scores
        ]
        if len(paired) >= _MIN_CORRELATION_N:
            xs = [a for a, _ in paired]
            ys = [b for _, b in paired]
            metric_validation = {
                "n": len(paired),
                "spearman": spearman_with_ci(xs, ys).to_dict(),
                "pearson": pearson_with_ci(xs, ys).to_dict(),
                "note": (
                    "automatic Layer-2 factual score against human factual correctness, at "
                    "the item level. This is what licenses the automatic metric to stand in "
                    "for a human judgement anywhere else in the paper."
                ),
            }
        else:
            metric_validation = {"n": len(paired), "note": "too few paired items to correlate"}
            warnings.append(
                "automatic-versus-human correlation not computed: fewer than three items "
                "had both an automatic score and a human rating"
            )
    else:
        metric_validation = {"n": 0, "note": "no automatic scores supplied"}
        warnings.append(
            "automatic-versus-human correlation not computed: no automatic Layer-2 scores "
            "were supplied. The Phase 12 gate requires this number."
        )

    return StudyAnalysis(
        n_responses=len(main),
        n_raters=len({r.rater_id for r in main}),
        systems=tuple(systems),
        timing=timing,
        edit_distance=edit_distance,
        would_file=would_file,
        likert=likert,
        agreement=agreement,
        intra_rater=intra_rater_reliability(responses),
        omnibus=omnibus,
        posthoc=posthoc,
        metric_validation=metric_validation,
        rater_model=rater_effects(main, systems_by_item, "factual_correctness"),
        baseline=baseline,
        warnings=tuple(warnings),
    )

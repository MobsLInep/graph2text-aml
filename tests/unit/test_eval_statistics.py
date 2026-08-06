"""Statistics, against hand-computed values and an independent reference implementation.

Every number in this file was worked out by hand or comes from `statsmodels`/`scipy`,
never from this module's own output. A statistics module tested against itself proves the
code is deterministic and nothing else.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from g2t_aml.eval.statistics import (
    Interval,
    bootstrap_ci,
    cliffs_delta,
    cliffs_delta_band,
    cohens_d,
    compare_systems,
    holm_bonferroni,
    paired_bootstrap_ci,
    publication_table,
    seed_summary,
    wilcoxon_signed_rank,
)

# ------------------------------------------------------------- bootstrap ---


def test_bootstrap_recovers_a_known_mean_and_brackets_it():
    # A standard normal, n=500. The mean is ~0 and the 95% interval on the mean is
    # ~[-0.09, +0.09] (1.96 / sqrt(500) = 0.0877). The bootstrap must land near that.
    rng = np.random.default_rng(7)
    values = rng.normal(loc=0.0, scale=1.0, size=500).tolist()

    interval = bootstrap_ci(values, n_resamples=2000, seed=1)

    assert interval.lo < interval.point < interval.hi
    analytic = 1.96 / math.sqrt(500)
    assert interval.hi - interval.lo == pytest.approx(2 * analytic, rel=0.25)
    assert interval.n == 500


def test_bootstrap_coverage_on_a_known_distribution():
    # The property that makes a CI a CI: over repeated samples from a known
    # distribution, ~95% of intervals contain the true mean. Twenty replications is a
    # coarse check -- it catches an interval that is systematically far too narrow,
    # which is the failure mode that would silently make every comparison significant.
    rng = np.random.default_rng(11)
    true_mean = 0.4
    covered = 0
    for i in range(20):
        sample = rng.binomial(1, true_mean, size=200).astype(float).tolist()
        interval = bootstrap_ci(sample, n_resamples=800, seed=i)
        covered += interval.lo <= true_mean <= interval.hi
    assert covered >= 17


def test_bootstrap_is_reproducible_under_a_seed():
    values = [0.1, 0.9, 0.4, 0.7, 0.2, 0.55]
    assert bootstrap_ci(values, n_resamples=500, seed=3) == bootstrap_ci(
        values, n_resamples=500, seed=3
    )


def test_bootstrap_on_one_observation_is_degenerate_and_says_so():
    interval = bootstrap_ci([0.42])
    assert (interval.point, interval.lo, interval.hi) == (0.42, 0.42, 0.42)
    assert interval.n == 1
    assert interval.n_resamples == 0


def test_bootstrap_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="at least one observation"):
        bootstrap_ci([])


def test_paired_bootstrap_is_on_the_difference():
    # b is a exactly 0.1 lower everywhere, so every resample gives the same difference
    # and the interval collapses onto 0.1. This is the pairing doing its job: an
    # unpaired interval over these two samples would be wide.
    a = [0.5, 0.7, 0.9, 0.3]
    b = [0.4, 0.6, 0.8, 0.2]
    interval = paired_bootstrap_ci(a, b, n_resamples=500, seed=2)
    assert interval.point == pytest.approx(0.1)
    assert interval.lo == pytest.approx(0.1)
    assert interval.hi == pytest.approx(0.1)
    assert interval.excludes_zero


def test_interval_excludes_zero_only_when_both_bounds_share_a_sign():
    assert Interval(0.1, 0.05, 0.2).excludes_zero
    assert Interval(-0.1, -0.2, -0.05).excludes_zero
    assert not Interval(0.1, -0.05, 0.2).excludes_zero


# ----------------------------------------------------- Holm-Bonferroni ---


def test_holm_matches_the_statsmodels_reference():
    statsmodels = pytest.importorskip("statsmodels.stats.multitest")
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.6, 0.9, 0.0001]

    adjusted, reject = holm_bonferroni(p_values, alpha=0.05)
    expected_reject, expected_adjusted, _, _ = statsmodels.multipletests(
        p_values, alpha=0.05, method="holm"
    )

    assert adjusted == pytest.approx(list(expected_adjusted))
    assert reject == list(expected_reject)


def test_holm_matches_the_reference_on_random_families():
    statsmodels = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(19)
    for size in (2, 5, 17, 120):
        p_values = rng.uniform(0, 1, size=size).tolist()
        adjusted, reject = holm_bonferroni(p_values)
        expected_reject, expected_adjusted, _, _ = statsmodels.multipletests(
            p_values, alpha=0.05, method="holm"
        )
        assert adjusted == pytest.approx(list(expected_adjusted))
        assert reject == list(expected_reject)


def test_holm_is_monotone_and_capped():
    adjusted, _ = holm_bonferroni([0.02, 0.03, 0.04, 0.9])
    # Step-down enforces monotonicity: a later (larger) raw p can never adjust below an
    # earlier one. Without that enforcement 0.03 * 3 = 0.09 would sit below 0.02 * 4.
    assert adjusted == sorted(adjusted)
    assert all(p <= 1.0 for p in adjusted)


def test_holm_on_an_empty_family_returns_empty():
    assert holm_bonferroni([]) == ([], [])


def test_holm_rejects_a_p_value_outside_the_unit_interval():
    with pytest.raises(ValueError, match="outside"):
        holm_bonferroni([0.5, 1.5])


# --------------------------------------------------------- effect sizes ---


def test_cliffs_delta_is_one_when_every_pair_is_greater():
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0


def test_cliffs_delta_hand_computed():
    # a = [1, 3], b = [2, 4]. Pairs: 1<2, 1<4, 3>2, 3<4 -> one greater, three less.
    # delta = (1 - 3) / 4 = -0.5.
    assert cliffs_delta([1, 3], [2, 4]) == pytest.approx(-0.5)


def test_cliffs_delta_is_zero_for_identical_samples():
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_cliffs_bands_follow_the_published_thresholds():
    assert cliffs_delta_band(0.10) == "negligible"
    assert cliffs_delta_band(0.20) == "small"
    assert cliffs_delta_band(0.40) == "medium"
    assert cliffs_delta_band(0.60) == "large"
    assert cliffs_delta_band(-0.60) == "large"


def test_cohens_d_paired_hand_computed():
    # Differences are all +2, so the standard deviation of the differences is 0 and d is
    # defined as 0 rather than infinite -- an infinity would land in a results table.
    assert cohens_d([3, 4, 5], [1, 2, 3], paired=True) == 0.0

    # Differences [1, 2, 3]: mean 2, sample sd 1 -> d = 2.
    assert cohens_d([2, 4, 6], [1, 2, 3], paired=True) == pytest.approx(2.0)


def test_cohens_d_independent_hand_computed():
    # a mean 5 sd 1, b mean 2 sd 1 -> pooled sd 1, d = 3.
    assert cohens_d([4, 5, 6], [1, 2, 3], paired=False) == pytest.approx(3.0)


def test_cohens_d_rejects_empty_and_mismatched():
    with pytest.raises(ValueError, match="empty"):
        cohens_d([], [1.0])
    with pytest.raises(ValueError, match="equal lengths"):
        cohens_d([1.0, 2.0], [1.0], paired=True)


# ------------------------------------------------------------ Wilcoxon ---


def test_wilcoxon_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    a = [0.9, 0.8, 0.7, 0.95, 0.85, 0.6, 0.75]
    b = [0.5, 0.6, 0.4, 0.55, 0.45, 0.35, 0.5]
    statistic, p = wilcoxon_signed_rank(a, b)
    reference = scipy_stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    assert statistic == pytest.approx(float(reference.statistic))
    assert p == pytest.approx(float(reference.pvalue))


def test_wilcoxon_on_all_ties_is_no_evidence_rather_than_a_crash():
    # scipy raises on an all-zero difference vector. Two systems that scored identically
    # on every case are two systems with no measured difference, and that has to be
    # expressible: it is the expected outcome of comparing an arm against itself.
    assert wilcoxon_signed_rank([0.5, 0.5], [0.5, 0.5]) == (0.0, 1.0)


def test_wilcoxon_rejects_unpaired_input():
    with pytest.raises(ValueError, match="paired"):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])


# ---------------------------------------------- seeds and the comparison ---


def test_seed_summary_reports_variance_only_when_there_is_more_than_one_seed():
    single = seed_summary("s1", "fact_f1", {42: 0.8})
    assert single.std is None
    assert single.single_seed
    assert "1 seed" in single.format()

    several = seed_summary("s1", "fact_f1", {42: 0.8, 7: 0.6, 13: 0.7})
    assert several.mean == pytest.approx(0.7)
    assert several.std == pytest.approx(0.1)
    assert not several.single_seed
    assert several.format(2) == "0.70 ± 0.10"


def test_seed_summary_rejects_no_seeds():
    with pytest.raises(ValueError, match="no seeds"):
        seed_summary("s1", "fact_f1", {})


def test_compare_systems_corrects_across_its_own_family():
    # Three systems -> three pairwise comparisons, so Holm's largest multiplier is 3.
    values = {
        "a": {f"c{i}": 0.9 for i in range(30)},
        "b": {f"c{i}": 0.5 for i in range(30)},
        "c": {f"c{i}": 0.1 for i in range(30)},
    }
    comparisons = compare_systems("fact_f1", values, n_resamples=200)

    assert len(comparisons) == 3
    assert all(c.p_adjusted is not None for c in comparisons)
    assert all("3 comparisons" in c.family for c in comparisons)
    # Every pair is separated at every case, so delta is +-1 and the difference interval
    # excludes zero.
    for comparison in comparisons:
        assert abs(comparison.cliffs_delta) == 1.0
        assert comparison.difference_ci.excludes_zero


def test_compare_systems_uses_only_the_cases_two_systems_share():
    values = {
        "a": {"c1": 1.0, "c2": 1.0, "c3": 1.0},
        "b": {"c2": 0.0, "c3": 0.0},
    }
    (comparison,) = compare_systems("fact_f1", values, n_resamples=100)
    assert comparison.n_cases == 2


def test_compare_systems_needs_two_systems():
    assert compare_systems("fact_f1", {"a": {"c1": 1.0}}) == []


def test_an_uncorrected_comparison_is_never_reported_as_significant():
    # The guard against a call site that builds a PairedComparison by hand and skips the
    # family: with no adjusted p there is no significance and no marker.
    (comparison,) = compare_systems(
        "fact_f1",
        {
            "a": {f"c{i}": float(i) for i in range(20)},
            "b": {f"c{i}": float(i) + 1 for i in range(20)},
        },
        n_resamples=100,
    )
    stripped = type(comparison)(
        **{**comparison.__dict__, "p_adjusted": None}  # type: ignore[arg-type]
    )
    assert not stripped.significant
    assert stripped.marker == "—"


def test_publication_table_renders_markers_and_names_the_family():
    values = {
        "a": {f"c{i}": 1.0 for i in range(40)},
        "b": {f"c{i}": 0.0 for i in range(40)},
    }
    comparisons = compare_systems("fact_f1", values, n_resamples=200)
    summaries = [
        seed_summary("a", "fact_f1", {1: 1.0, 2: 1.0}),
        seed_summary("b", "fact_f1", {1: 0.0, 2: 0.0}),
    ]
    intervals = {
        "a": bootstrap_ci([1.0] * 40, n_resamples=100),
        "b": bootstrap_ci([0.0] * 40, n_resamples=100),
    }

    table = publication_table("fact_f1", summaries, intervals, comparisons)

    assert "| `a` |" in table and "| `b` |" in table
    assert "Holm-Bonferroni-corrected" in table
    assert "1 pairwise comparisons" in table


def test_publication_table_over_no_systems_says_so():
    assert "No systems" in publication_table("fact_f1", [], {})

"""Metric and analysis tests.

Mostly these check that the numbers behave the way the claims about them require: that
AUC-PR moves where ROC-AUC does not, that the bootstrap is paired when it says it is, and
that the shuffled-label nulls actually sit where a null should.
"""

from __future__ import annotations

import numpy as np
import pytest

# scikit-learn lives in the optional `graph` extra alongside torch (D-004), so a CPU-only
# checkout skips this module rather than failing collection.
pytest.importorskip("sklearn")

from g2t_aml.models.encoder.analysis import (  # noqa: E402
    knn_purity,
    linear_probe,
    silhouette_by_typology,
)
from g2t_aml.models.encoder.metrics import (  # noqa: E402
    aggregate_over_seeds,
    binary_metrics,
    bootstrap_auc_pr,
    paired_difference,
    stratified_bootstrap_indices,
    typology_metrics,
)

CLASSES = ("fan_out", "fan_in", "cycle", "unclassified")


def _imbalanced(n=2000, prevalence=0.05, separation=1.0, seed=0):
    rng = np.random.default_rng(seed)
    targets = (rng.random(n) < prevalence).astype(int)
    scores = rng.normal(0, 1, n) + separation * targets
    return scores, targets


# ------------------------------------------------------------------ binary ---


def test_auc_pr_baseline_is_prevalence_not_a_half():
    """The reason a bare AUC-PR is uninterpretable and prevalence is always reported."""
    rng = np.random.default_rng(0)
    targets = (rng.random(5000) < 0.05).astype(int)
    metrics = binary_metrics(rng.random(5000), targets)
    assert metrics.auc_pr == pytest.approx(metrics.prevalence, abs=0.02)
    assert metrics.auc_roc == pytest.approx(0.5, abs=0.03)
    assert metrics.lift == pytest.approx(1.0, abs=0.4)


def test_auc_roc_flatters_where_auc_pr_does_not():
    """The concrete reason selection goes through AUC-PR (see metrics.py's docstring)."""
    scores, targets = _imbalanced(prevalence=0.02, separation=1.6)
    metrics = binary_metrics(scores, targets)
    assert metrics.auc_roc > 0.85
    assert metrics.auc_pr < 0.35
    assert metrics.auc_roc - metrics.auc_pr > 0.4


def test_degenerate_population_returns_nan_rather_than_raising():
    """A sweep must not die because one split happened to be single-class."""
    metrics = binary_metrics(np.random.default_rng(0).random(50), np.zeros(50, dtype=int))
    assert np.isnan(metrics.auc_pr)
    assert np.isnan(metrics.auc_roc)
    assert metrics.n == 50 and metrics.n_positive == 0


def test_metrics_report_the_population_they_were_computed_on():
    scores, targets = _imbalanced(n=1000, prevalence=0.1)
    metrics = binary_metrics(scores, targets)
    assert metrics.n == 1000
    assert metrics.n_positive == int(targets.sum())
    assert 0.0 <= metrics.recall_at_best_f1 <= 1.0


# --------------------------------------------------------------- bootstrap ---


def test_stratified_resamples_preserve_the_positive_count():
    """Plain resampling occasionally lands with too few positives to score."""
    _, targets = _imbalanced(n=1000, prevalence=0.03)
    indices = stratified_bootstrap_indices(targets, 50, np.random.default_rng(0))
    counts = targets[indices].sum(axis=1)
    assert (counts == targets.sum()).all()


def test_bootstrap_interval_brackets_the_point_estimate():
    scores, targets = _imbalanced(separation=1.5)
    interval = bootstrap_auc_pr(scores, targets, n_resamples=200, seed=1)
    assert interval["lo"] < interval["point"] < interval["hi"]
    assert interval["n_resamples"] == 200


def test_shared_indices_make_two_arms_comparable():
    scores_a, targets = _imbalanced(separation=1.5, seed=2)
    scores_b = scores_a + np.random.default_rng(3).normal(0, 0.01, scores_a.size)
    indices = stratified_bootstrap_indices(targets, 100, np.random.default_rng(4))
    a = bootstrap_auc_pr(scores_a, targets, indices=indices)
    b = bootstrap_auc_pr(scores_b, targets, indices=indices)
    assert abs(a["point"] - b["point"]) < 0.05


def test_paired_difference_detects_a_real_gap():
    _, targets = _imbalanced(n=3000, prevalence=0.08, seed=5)
    rng = np.random.default_rng(5)
    strong = rng.normal(0, 1, targets.size) + 2.0 * targets
    weak = rng.normal(0, 1, targets.size) + 0.3 * targets
    result = paired_difference(strong, weak, targets, n_resamples=300, seed=6)
    assert result["difference"] > 0
    assert result["excludes_zero"]
    assert result["p_gt_zero"] > 0.95


def test_paired_difference_does_not_invent_a_gap():
    """The gate depends on this interval not excluding zero when the arms are equal."""
    _, targets = _imbalanced(n=3000, prevalence=0.08, seed=7)
    rng = np.random.default_rng(7)
    scores = rng.normal(0, 1, targets.size) + 1.0 * targets
    result = paired_difference(scores, scores.copy(), targets, n_resamples=300, seed=8)
    assert result["difference"] == pytest.approx(0.0, abs=1e-9)
    assert not result["excludes_zero"]


def test_aggregate_over_seeds_uses_the_sample_standard_deviation():
    summary = aggregate_over_seeds([0.70, 0.74, 0.78])
    assert summary["mean"] == pytest.approx(0.74)
    assert summary["std"] == pytest.approx(np.std([0.70, 0.74, 0.78], ddof=1))
    assert summary["n_seeds"] == 3


# --------------------------------------------------------------- typology ---


def test_structural_macro_f1_excludes_the_catch_all_class():
    """`unclassified` is 93.8% of the corpus and would dominate an all-class macro-F1."""
    targets = np.asarray([0, 0, 1, 1, 2, 2] + [3] * 60)
    perfect_structural = np.asarray([0, 0, 1, 1, 2, 2] + [0] * 60)
    metrics = typology_metrics(perfect_structural, targets, CLASSES)
    assert metrics.macro_f1_structural == pytest.approx(1.0)
    assert metrics.macro_f1_all < 1.0
    assert metrics.n_structural == 6


def test_unlabelled_cases_are_dropped_not_scored_as_wrong():
    targets = np.asarray([-1, -1, 0, 1])
    predictions = np.asarray([2, 2, 0, 1])
    assert typology_metrics(predictions, targets, CLASSES).accuracy == pytest.approx(1.0)


def test_chance_baseline_is_reported_and_is_low():
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 3, 300)
    metrics = typology_metrics(rng.integers(0, 3, 300), targets, CLASSES)
    assert metrics.chance_macro_f1_structural < 0.5
    assert not np.isnan(metrics.chance_macro_f1_structural)


# -------------------------------------------------------------- embeddings ---


def _clustered(n_per=60, dim=8, spread=0.25, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 6, (3, dim))
    embeddings = np.concatenate([c + rng.normal(0, spread, (n_per, dim)) for c in centres])
    labels = np.repeat([0, 1, 2], n_per)
    return embeddings, labels


def test_knn_purity_beats_its_null_on_clustered_embeddings():
    embeddings, labels = _clustered()
    results = knn_purity(embeddings, labels, k_values=(5, 10), n_shuffles=10, seed=0)
    assert len(results) == 2
    for result in results:
        assert result.purity > 0.95
        assert result.null_mean == pytest.approx(1 / 3, abs=0.1)
        assert result.z_score > 5


def test_knn_purity_matches_its_null_on_structureless_embeddings():
    """The null is what makes the number mean something; assert it is not free to beat."""
    rng = np.random.default_rng(1)
    results = knn_purity(
        rng.normal(0, 1, (300, 8)), rng.integers(0, 3, 300), k_values=(10,), n_shuffles=20, seed=1
    )
    assert abs(results[0].z_score) < 4


def test_silhouette_beats_its_null_on_clustered_embeddings():
    embeddings, labels = _clustered()
    observed, null = silhouette_by_typology(embeddings, labels, n_shuffles=5, seed=0)
    assert observed > 0.6
    assert null < 0.1


def test_linear_probe_recovers_linearly_separable_structure():
    embeddings, labels = _clustered(n_per=200, seed=2)
    split = len(labels) // 2
    order = np.random.default_rng(2).permutation(len(labels))
    train, test = order[:split], order[split:]
    result = linear_probe(
        embeddings[train], labels[train], embeddings[test], labels[test], CLASSES, seed=2
    )
    assert result.accuracy > 0.95
    assert result.shuffled_accuracy < 0.6
    assert result.n_train == split


def test_linear_probe_shuffled_null_collapses_to_chance():
    """A probe that scores well on shuffled labels is measuring its own capacity."""
    rng = np.random.default_rng(3)
    embeddings = rng.normal(0, 1, (400, 8))
    labels = rng.integers(0, 3, 400)
    result = linear_probe(
        embeddings[:200], labels[:200], embeddings[200:], labels[200:], CLASSES, seed=3
    )
    assert abs(result.accuracy - result.shuffled_accuracy) < 0.2


def test_linear_probe_keeps_every_structural_case_when_capping():
    """The cap subsamples the `unclassified` bulk only; rare classes are all retained."""
    rng = np.random.default_rng(4)
    labels = np.concatenate([rng.integers(0, 3, 40), np.full(500, 3)])
    embeddings = rng.normal(0, 1, (labels.size, 6)) + labels[:, None]
    result = linear_probe(embeddings, labels, embeddings, labels, CLASSES, seed=4, max_train=100)
    assert result.n_train == 100
    assert result.structural_macro_f1 > 0.5

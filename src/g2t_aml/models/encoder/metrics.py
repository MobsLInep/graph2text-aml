"""Evaluation metrics, with AUC-PR as the primary and bootstrap intervals on everything.

**AUC-PR is the selection metric, not AUC-ROC.** Under heavy imbalance ROC-AUC is
flattering to the point of being misleading: it is computed against the false-positive
*rate*, and with 9,270 negatives against 730 positives on the realistic stream, a
thousand false positives moves the FPR by 0.11 while moving precision from 0.42 to 0.13.
A model can look excellent on ROC while being useless to an investigator who has to work
the alerts. Both are reported — ROC-AUC is what the rest of the AML literature quotes, so
omitting it would make this work incomparable — but selection, early stopping and every
claim of superiority go through AUC-PR.

**Every headline number carries a bootstrap interval.** The Phase 7 gate asks whether
GATv2 beats the MLP control with non-overlapping intervals, and an interval computed the
wrong way would make that gate decorative. Resampling is over *cases*, stratified by
label so a resample cannot end up with no positives, and the same resample indices are
used for every arm so the intervals are paired rather than independent — which is the
comparison actually being made.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

#: Bootstrap resamples for every reported interval.
DEFAULT_N_BOOTSTRAP = 2000

#: Two-sided interval coverage.
DEFAULT_CI = 0.95


@dataclass
class BinaryMetrics:
    """Binary classification metrics for one evaluation population.

    Attributes:
        auc_pr: Average precision. **The primary metric.**
        auc_roc: ROC-AUC, reported for comparability with the AML literature.
        best_f1: The best F1 over all thresholds.
        best_f1_threshold: The threshold achieving it.
        precision_at_best_f1: Precision there.
        recall_at_best_f1: Recall there.
        prevalence: Positive rate in the population. AUC-PR's baseline is prevalence,
            not 0.5, so a number is uninterpretable without it.
        n: Population size.
        n_positive: Positive count.
        lift: ``auc_pr / prevalence`` — how much better than a random ranker.
    """

    auc_pr: float
    auc_roc: float
    best_f1: float
    best_f1_threshold: float
    precision_at_best_f1: float
    recall_at_best_f1: float
    prevalence: float
    n: int
    n_positive: int
    lift: float

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a plain dict.

        Returns:
            Every field.
        """
        return asdict(self)


def binary_metrics(scores: np.ndarray, targets: np.ndarray) -> BinaryMetrics:
    """Compute the binary metric set for one population.

    Args:
        scores: ``[n]`` predicted probabilities or any monotone score.
        targets: ``[n]`` binary ground truth.

    Returns:
        The metric set. A population with only one class returns NaN for the two AUCs
        rather than raising, since a degenerate split is a fact about the data and
        should surface as NaN in a table rather than crash a sweep.
    """
    targets = targets.astype(int)
    n, n_positive = int(targets.size), int(targets.sum())
    prevalence = n_positive / n if n else float("nan")

    if n_positive in (0, n):
        return BinaryMetrics(
            auc_pr=float("nan"),
            auc_roc=float("nan"),
            best_f1=float("nan"),
            best_f1_threshold=float("nan"),
            precision_at_best_f1=float("nan"),
            recall_at_best_f1=float("nan"),
            prevalence=prevalence,
            n=n,
            n_positive=n_positive,
            lift=float("nan"),
        )

    auc_pr = float(average_precision_score(targets, scores))
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    # precision_recall_curve returns one more point than thresholds; the final point is
    # the degenerate (recall 0, precision 1) corner and has no threshold behind it.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.nanargmax(f1[:-1])) if thresholds.size else 0
    return BinaryMetrics(
        auc_pr=auc_pr,
        auc_roc=float(roc_auc_score(targets, scores)),
        best_f1=float(f1[best]),
        best_f1_threshold=float(thresholds[best]) if thresholds.size else float("nan"),
        precision_at_best_f1=float(precision[best]),
        recall_at_best_f1=float(recall[best]),
        prevalence=prevalence,
        n=n,
        n_positive=n_positive,
        lift=auc_pr / prevalence if prevalence else float("nan"),
    )


@dataclass
class TypologyMetrics:
    """Auxiliary typology-head metrics.

    Two macro-F1 numbers, deliberately.

    ``macro_f1_all`` covers all nine classes including ``unclassified``, which is 93.8%
    of the corpus — so it is dominated by a class that is trivially easy and is the
    number that flatters. ``macro_f1_structural`` restricts to the eight real typologies
    *and to the cases that have one*, which is the number that says whether the encoder
    can tell a fan-out from a cycle. The second is the one the narrative generator
    depends on.

    Attributes:
        accuracy: Overall accuracy across all labelled cases.
        macro_f1_all: Macro-F1 over all nine classes.
        macro_f1_structural: Macro-F1 over the eight structural typologies, computed on
            the subpopulation that carries one.
        chance_macro_f1_structural: Macro-F1 of a stratified random guesser on that same
            subpopulation, which is what "well above chance" is measured against.
        per_class_f1: F1 per class name.
        support: Case count per class name.
        n_structural: Size of the structural subpopulation.
    """

    accuracy: float
    macro_f1_all: float
    macro_f1_structural: float
    chance_macro_f1_structural: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)
    n_structural: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a plain dict.

        Returns:
            Every field.
        """
        return asdict(self)


def typology_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: tuple[str, ...],
    *,
    unclassified: str = "unclassified",
    rng: np.random.Generator | None = None,
) -> TypologyMetrics:
    """Compute the typology-head metric set.

    Args:
        predictions: ``[n]`` predicted class indices.
        targets: ``[n]`` true class indices; negative entries are unlabelled and are
            dropped before anything is computed.
        class_names: Class index to name.
        unclassified: The name of the catch-all class, excluded from the structural
            macro-F1.
        rng: Generator for the chance baseline. Defaults to a fixed seed so the baseline
            is reproducible across runs.

    Returns:
        The metric set, all-NaN when no case in the population carries a typology.
    """
    labelled = targets >= 0
    predictions, targets = predictions[labelled], targets[labelled]
    if targets.size == 0:
        return TypologyMetrics(
            accuracy=float("nan"),
            macro_f1_all=float("nan"),
            macro_f1_structural=float("nan"),
            chance_macro_f1_structural=float("nan"),
        )

    labels = list(range(len(class_names)))
    per_class = f1_score(targets, predictions, labels=labels, average=None, zero_division=0)
    support = {name: int((targets == i).sum()) for i, name in enumerate(class_names)}

    unclassified_index = class_names.index(unclassified) if unclassified in class_names else -1
    structural = targets != unclassified_index
    if structural.sum():
        structural_labels = [i for i in labels if i != unclassified_index]
        macro_structural = float(
            f1_score(
                targets[structural],
                predictions[structural],
                labels=structural_labels,
                average="macro",
                zero_division=0,
            )
        )
        generator = rng or np.random.default_rng(0)
        # Stratified guesser: draws from the observed structural class distribution,
        # which is a stronger and fairer null than uniform guessing.
        guesses = generator.choice(targets[structural], size=int(structural.sum()), replace=True)
        chance = float(
            f1_score(
                targets[structural],
                guesses,
                labels=structural_labels,
                average="macro",
                zero_division=0,
            )
        )
    else:
        macro_structural, chance = float("nan"), float("nan")

    return TypologyMetrics(
        accuracy=float((predictions == targets).mean()),
        macro_f1_all=float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        macro_f1_structural=macro_structural,
        chance_macro_f1_structural=chance,
        per_class_f1={name: float(per_class[i]) for i, name in enumerate(class_names)},
        support=support,
        n_structural=int(structural.sum()),
    )


def stratified_bootstrap_indices(
    targets: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw label-stratified bootstrap resample indices.

    Stratified rather than plain: on the realistic stream a plain resample of 10,000
    cases at 7.3% prevalence occasionally lands with far fewer positives than the
    original, and average precision on such a resample measures the resample rather than
    the model. Stratifying holds the positive count fixed, which is the standard
    correction and is what makes the intervals comparable across populations of different
    prevalence.

    Args:
        targets: ``[n]`` binary ground truth.
        n_resamples: Number of resamples.
        rng: Random generator.

    Returns:
        An ``[n_resamples, n]`` index array.
    """
    positive = np.flatnonzero(targets == 1)
    negative = np.flatnonzero(targets == 0)
    draws = np.empty((n_resamples, targets.size), dtype=np.int64)
    for i in range(n_resamples):
        draws[i] = np.concatenate(
            [
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            ]
        )
    return draws


def bootstrap_auc_pr(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    seed: int = 0,
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Bootstrap a confidence interval for AUC-PR.

    Args:
        scores: ``[n]`` predicted scores.
        targets: ``[n]`` binary ground truth.
        n_resamples: Number of resamples.
        ci: Two-sided coverage.
        seed: Seed for the resample draw, ignored when ``indices`` is given.
        indices: Pre-drawn resample indices, so two arms can be compared on exactly the
            same resamples. This is what makes the gate's "non-overlapping intervals"
            comparison a paired one.

    Returns:
        ``{"point", "lo", "hi", "std", "n_resamples"}``.
    """
    rng = np.random.default_rng(seed)
    if indices is None:
        indices = stratified_bootstrap_indices(targets, n_resamples, rng)

    values = np.empty(indices.shape[0])
    for i, draw in enumerate(indices):
        resampled = targets[draw]
        values[i] = (
            average_precision_score(resampled, scores[draw])
            if 0 < resampled.sum() < resampled.size
            else np.nan
        )
    alpha = (1.0 - ci) / 2.0
    return {
        "point": float(average_precision_score(targets, scores)),
        "lo": float(np.nanquantile(values, alpha)),
        "hi": float(np.nanquantile(values, 1 - alpha)),
        "std": float(np.nanstd(values)),
        "n_resamples": int(indices.shape[0]),
    }


def paired_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    targets: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the AUC-PR difference between two arms on the same population.

    The Phase 7 gate is a claim about a difference, and the interval on a difference is
    not recoverable from two separate intervals — two overlapping marginal intervals are
    entirely compatible with a difference that excludes zero, because the two arms'
    errors are correlated across cases. This resamples once and evaluates both arms on
    each resample.

    Args:
        scores_a: ``[n]`` scores from the first arm.
        scores_b: ``[n]`` scores from the second arm.
        targets: ``[n]`` binary ground truth, shared.
        n_resamples: Number of resamples.
        ci: Two-sided coverage.
        seed: Seed for the resample draw.

    Returns:
        ``{"difference", "lo", "hi", "p_gt_zero", "excludes_zero"}`` where ``difference``
        is ``a - b`` on the full population and ``p_gt_zero`` is the fraction of
        resamples on which ``a`` beat ``b``.
    """
    rng = np.random.default_rng(seed)
    indices = stratified_bootstrap_indices(targets, n_resamples, rng)

    deltas = np.empty(n_resamples)
    for i, draw in enumerate(indices):
        resampled = targets[draw]
        if not 0 < resampled.sum() < resampled.size:
            deltas[i] = np.nan
            continue
        deltas[i] = average_precision_score(resampled, scores_a[draw]) - average_precision_score(
            resampled, scores_b[draw]
        )
    alpha = (1.0 - ci) / 2.0
    lo = float(np.nanquantile(deltas, alpha))
    hi = float(np.nanquantile(deltas, 1 - alpha))
    return {
        "difference": float(
            average_precision_score(targets, scores_a) - average_precision_score(targets, scores_b)
        ),
        "lo": lo,
        "hi": hi,
        "p_gt_zero": float(np.nanmean(deltas > 0)),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def aggregate_over_seeds(values: list[float]) -> dict[str, float]:
    """Summarise a metric across seeds.

    Args:
        values: One value per seed.

    Returns:
        ``{"mean", "std", "min", "max", "n_seeds"}``. The standard deviation is the
        sample one (``ddof=1``), because three seeds are a sample from the seed
        distribution and not the population.
    """
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "n_seeds": int(array.size),
    }

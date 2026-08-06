"""Quantifying embedding quality. A picture is not a result.

The original Phase 7 gate was "embeddings show typology structure under UMAP". A UMAP
plot is a two-dimensional non-linear projection with tunable neighbourhood and minimum
distance, and it can be made to show clusters that a linear classifier cannot find and to
hide clusters that one can. It goes in the paper as an illustration, with its numbers in
the caption, and it settles nothing on its own.

Three numbers settle it instead:

- **kNN purity** — the fraction of a point's ``k`` nearest neighbours sharing its
  typology, against a shuffled-label null computed on the *same* embeddings. The null
  matters: purity has a floor set by class prevalence, and 93.8% of cases are
  ``unclassified``, so a purity of 0.9 could mean everything or nothing.
- **Silhouette score** by typology — whether the classes occupy separated regions rather
  than merely having locally-pure neighbourhoods.
- **Linear probe accuracy** — a logistic regression on frozen embeddings predicting
  typology. **This is the number that predicts whether Phase 8 will work.** The fusion
  layer is a linear projection into the language model's embedding space, so if typology
  is not linearly decodable from the pooled tokens, the projection cannot make it
  decodable and the language model is unlikely to recover it. A high kNN purity with a
  low probe accuracy would mean the structure is there but not in a form the fusion layer
  can use, which is a specific and actionable finding.

The probe is trained on the training split's embeddings and scored on test, never
cross-validated within test — a probe fitted and scored on the same population measures
memorisation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from g2t_aml.utils.io import write_json

#: Neighbourhood sizes the purity is reported at, per the Phase 7 brief.
DEFAULT_K_VALUES: tuple[int, ...] = (5, 10, 20)

#: Shuffles used to build the purity and silhouette nulls.
DEFAULT_N_SHUFFLES = 20

#: Fewest labelled points a neighbourhood, a silhouette or a probe fit can be computed on.
_MIN_POINTS = 2

#: Fewest labelled points worth drawing a UMAP of.
_MIN_UMAP_POINTS = 10


@dataclass
class PurityResult:
    """kNN purity at one neighbourhood size, against its shuffled-label null.

    Attributes:
        k: Neighbourhood size.
        purity: Observed mean fraction of neighbours sharing the point's label.
        null_mean: Mean purity under label shuffling.
        null_std: Its standard deviation.
        z_score: ``(purity - null_mean) / null_std``. The interpretable number: purity
            alone has a prevalence floor and is not comparable across populations.
    """

    k: int
    purity: float
    null_mean: float
    null_std: float
    z_score: float


@dataclass
class ProbeResult:
    """Linear probe performance on frozen embeddings.

    Attributes:
        representation: Which embedding was probed — ``graph_embedding`` or
            ``pooled_tokens``.
        accuracy: Test accuracy over labelled cases.
        macro_f1: Macro-F1 over all classes present.
        structural_macro_f1: Macro-F1 restricted to the eight real typologies, on the
            subpopulation that carries one. The number Phase 8 depends on.
        majority_baseline: Accuracy of always predicting the most frequent class.
        shuffled_accuracy: Accuracy of the same probe trained on shuffled labels — the
            null that says how much of the number is the probe's capacity rather than
            the embedding's structure.
        n_train: Probe training set size.
        n_test: Probe test set size.
        per_class_f1: F1 per class name.
        converged: Whether the solver reached its tolerance rather than its iteration
            cap. **Reported because an unconverged probe is a lower bound**, not a
            measurement: it under-states what is linearly decodable, and quoting it as
            "typology is only X% decodable" would blame the embedding for the solver.
        n_iterations: Iterations the solver actually used.
    """

    representation: str
    accuracy: float
    macro_f1: float
    structural_macro_f1: float
    majority_baseline: float
    shuffled_accuracy: float
    n_train: int
    n_test: int
    per_class_f1: dict[str, float] = field(default_factory=dict)
    converged: bool = True
    n_iterations: int = 0


@dataclass
class EmbeddingAnalysis:
    """The complete embedding-quality report for one arm at one seed.

    Attributes:
        arm: Architecture name.
        seed: The seed.
        purity: kNN purity at each ``k``.
        silhouette: Observed silhouette over the structural typologies.
        silhouette_null_mean: Its shuffled-label null.
        probes: One probe result per probed representation.
        n_structural: Size of the structural-typology subpopulation.
    """

    arm: str
    seed: int
    purity: list[PurityResult] = field(default_factory=list)
    silhouette: float = float("nan")
    silhouette_null_mean: float = float("nan")
    probes: list[ProbeResult] = field(default_factory=list)
    n_structural: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form.

        Returns:
            Every field, with the nested dataclasses expanded.
        """
        return asdict(self)


def knn_purity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = 0,
) -> list[PurityResult]:
    """Measure kNN label purity against a shuffled-label null.

    The null is computed by shuffling the labels while holding the embeddings fixed, so
    it preserves both the class prevalence and the geometry — the only thing removed is
    the correspondence between them. A null that resampled points instead would also
    remove the density structure and would be too easy to beat.

    Args:
        embeddings: ``[n, d]`` embeddings.
        labels: ``[n]`` class labels. Negative entries are dropped as unlabelled.
        k_values: Neighbourhood sizes.
        n_shuffles: Shuffles per ``k``.
        seed: Seed for the shuffles.

    Returns:
        One result per ``k``. Empty when fewer than two labelled points exist.
    """
    mask = labels >= 0
    embeddings, labels = embeddings[mask], labels[mask]
    n = embeddings.shape[0]
    if n < _MIN_POINTS:
        return []

    rng = np.random.default_rng(seed)
    results: list[PurityResult] = []
    for k in k_values:
        effective = min(k, n - 1)
        if effective < 1:
            continue
        # +1 because the first neighbour of a point is always itself.
        finder = NearestNeighbors(n_neighbors=effective + 1).fit(embeddings)
        neighbours = finder.kneighbors(embeddings, return_distance=False)[:, 1:]

        observed = float((labels[neighbours] == labels[:, None]).mean())
        nulls = np.empty(n_shuffles)
        for i in range(n_shuffles):
            shuffled = rng.permutation(labels)
            nulls[i] = (shuffled[neighbours] == shuffled[:, None]).mean()
        null_mean, null_std = float(nulls.mean()), float(nulls.std(ddof=1))
        results.append(
            PurityResult(
                k=k,
                purity=observed,
                null_mean=null_mean,
                null_std=null_std,
                z_score=(observed - null_mean) / null_std if null_std > 0 else float("nan"),
            )
        )
    return results


def silhouette_by_typology(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = 0,
    max_points: int = 5000,
) -> tuple[float, float]:
    """Compute the silhouette score over typologies, and its shuffled-label null.

    Args:
        embeddings: ``[n, d]`` embeddings.
        labels: ``[n]`` class labels. Negative entries are dropped.
        n_shuffles: Shuffles for the null.
        seed: Seed for subsampling and shuffling.
        max_points: Subsample above this size. The silhouette is O(n^2) in distance
            computations and the test split is 3,196 cases, so this rarely fires; it
            exists so the function stays usable on the 10,000-case realistic stream.

    Returns:
        ``(observed, null_mean)``, both NaN when fewer than two classes are present.
    """
    mask = labels >= 0
    embeddings, labels = embeddings[mask], labels[mask]
    if embeddings.shape[0] < _MIN_POINTS or len(np.unique(labels)) < _MIN_POINTS:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    if embeddings.shape[0] > max_points:
        pick = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        embeddings, labels = embeddings[pick], labels[pick]

    observed = float(silhouette_score(embeddings, labels))
    nulls = [
        float(silhouette_score(embeddings, rng.permutation(labels))) for _ in range(n_shuffles)
    ]
    return observed, float(np.mean(nulls))


def linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    class_names: tuple[str, ...],
    *,
    representation: str = "graph_embedding",
    unclassified: str = "unclassified",
    seed: int = 0,
    max_iter: int = 4000,
    max_train: int = 6000,
    tol: float = 1e-3,
) -> ProbeResult:
    """Fit a logistic regression on frozen embeddings and score it on a held-out split.

    Embeddings are standardised with statistics fitted on the probe's training split
    only. The probe is multinomial with balanced class weights, because without them it
    predicts ``unclassified`` for everything and reports an accuracy of 0.94 that says
    nothing about the eight classes anyone cares about.

    Args:
        train_embeddings: ``[n_train, d]``.
        train_labels: ``[n_train]`` class indices; negative entries are dropped.
        test_embeddings: ``[n_test, d]``.
        test_labels: ``[n_test]`` class indices; negative entries are dropped.
        class_names: Class index to name.
        representation: Which embedding is being probed, recorded in the result.
        unclassified: Name of the catch-all class, excluded from the structural F1.
        seed: Seed for the solver and the shuffled-label null.
        max_iter: Solver iteration cap. Generous on purpose: an unconverged probe
            under-reports what is linearly decodable, and this number is the Phase 8
            forecast. ``converged`` on the result says whether the cap was reached.
        max_train: Cap on the probe's training set. Above it, **every case carrying a
            structural typology is kept** and only the ``unclassified`` bulk is
            subsampled — the rare classes are the entire point of the measurement and
            throwing any of them away to save solver time would be measuring the
            subsample. On the pooled tokens the design matrix is 4,096-dimensional and
            an uncapped fit costs minutes per arm per seed, which is what this exists to
            avoid.
        tol: Solver tolerance. Looser than scikit-learn's 1e-4 default: on the
            4,096-dimensional pooled-token matrix the default does not converge inside
            any tolerable iteration budget, and a converged fit at 1e-3 is a better
            measurement than an unconverged one at 1e-4.

    Returns:
        The probe result. All-NaN when either split has fewer than two classes.
    """
    train_mask, test_mask = train_labels >= 0, test_labels >= 0
    x_train, y_train = train_embeddings[train_mask], train_labels[train_mask]
    x_test, y_test = test_embeddings[test_mask], test_labels[test_mask]

    if y_train.size > max_train:
        bulk_index = class_names.index(unclassified) if unclassified in class_names else -1
        structural = np.flatnonzero(y_train != bulk_index)
        bulk = np.flatnonzero(y_train == bulk_index)
        budget = max(max_train - structural.size, 0)
        keep = np.concatenate(
            [
                structural,
                np.random.default_rng(seed).choice(
                    bulk, size=min(budget, bulk.size), replace=False
                ),
            ]
        )
        keep.sort()
        x_train, y_train = x_train[keep], y_train[keep]

    empty = ProbeResult(
        representation=representation,
        accuracy=float("nan"),
        macro_f1=float("nan"),
        structural_macro_f1=float("nan"),
        majority_baseline=float("nan"),
        shuffled_accuracy=float("nan"),
        n_train=int(y_train.size),
        n_test=int(y_test.size),
    )
    if y_train.size < _MIN_POINTS or y_test.size < 1 or len(np.unique(y_train)) < _MIN_POINTS:
        return empty

    scaler = StandardScaler().fit(x_train)
    x_train_s, x_test_s = scaler.transform(x_train), scaler.transform(x_test)

    def _fit(labels: np.ndarray) -> LogisticRegression:
        # `multi_class` is deliberately left at its default: lbfgs is multinomial for
        # more than two classes, and passing the argument explicitly is deprecated in
        # scikit-learn 1.5 and removed in 1.7.
        return LogisticRegression(
            max_iter=max_iter,
            tol=tol,
            class_weight="balanced",
            random_state=seed,
        ).fit(x_train_s, labels)

    probe = _fit(y_train)
    predicted = probe.predict(x_test_s)

    rng = np.random.default_rng(seed)
    shuffled_probe = _fit(rng.permutation(y_train))
    shuffled_accuracy = float((shuffled_probe.predict(x_test_s) == y_test).mean())

    present = sorted(set(np.unique(y_train)) | set(np.unique(y_test)))
    unclassified_index = class_names.index(unclassified) if unclassified in class_names else -1
    structural_mask = y_test != unclassified_index
    structural_labels = [i for i in present if i != unclassified_index]
    structural = (
        float(
            f1_score(
                y_test[structural_mask],
                predicted[structural_mask],
                labels=structural_labels,
                average="macro",
                zero_division=0,
            )
        )
        if structural_mask.sum() and structural_labels
        else float("nan")
    )

    per_class = f1_score(y_test, predicted, labels=present, average=None, zero_division=0)
    counts = np.bincount(y_test, minlength=len(class_names))
    return ProbeResult(
        representation=representation,
        accuracy=float((predicted == y_test).mean()),
        macro_f1=float(
            f1_score(y_test, predicted, labels=present, average="macro", zero_division=0)
        ),
        structural_macro_f1=structural,
        majority_baseline=float(counts.max() / counts.sum()) if counts.sum() else float("nan"),
        shuffled_accuracy=shuffled_accuracy,
        n_train=int(y_train.size),
        n_test=int(y_test.size),
        per_class_f1={class_names[i]: float(f) for i, f in zip(present, per_class, strict=True)},
        converged=bool(np.max(probe.n_iter_) < max_iter),
        n_iterations=int(np.max(probe.n_iter_)),
    )


def analyse_embeddings(
    # of the six arguments; bundling them would not reduce what the caller must supply.
    *,
    arm: str,
    seed: int,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_tokens: np.ndarray | None,
    class_names: tuple[str, ...],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    train_tokens: np.ndarray | None = None,
) -> EmbeddingAnalysis:
    """Run the complete embedding-quality battery for one arm at one seed.

    Args:
        arm: Architecture name.
        seed: The seed.
        train_embeddings: ``[n, d]`` graph embeddings for the probe's training split.
        train_labels: ``[n]`` typology indices for it.
        test_embeddings: ``[m, d]`` graph embeddings for the evaluation split.
        test_labels: ``[m]`` typology indices for it.
        test_tokens: ``[m, k, d]`` pooled tokens for the evaluation split, or None. When
            present they are flattened and probed too, because they and not the graph
            embedding are what Phase 8 actually consumes.
        class_names: Class index to name.
        k_values: Neighbourhood sizes for the purity.
        train_tokens: ``[n, k, d]`` pooled tokens for the probe's training split.

    Returns:
        The complete analysis.
    """
    purity = knn_purity(test_embeddings, test_labels, k_values=k_values, seed=seed)
    silhouette, null = silhouette_by_typology(test_embeddings, test_labels, seed=seed)

    probes = [
        linear_probe(
            train_embeddings,
            train_labels,
            test_embeddings,
            test_labels,
            class_names,
            representation="graph_embedding",
            seed=seed,
        )
    ]
    if test_tokens is not None and train_tokens is not None:
        probes.append(
            linear_probe(
                train_tokens.reshape(train_tokens.shape[0], -1),
                train_labels,
                test_tokens.reshape(test_tokens.shape[0], -1),
                test_labels,
                class_names,
                representation="pooled_tokens",
                seed=seed,
            )
        )

    unclassified_index = class_names.index("unclassified") if "unclassified" in class_names else -1
    return EmbeddingAnalysis(
        arm=arm,
        seed=seed,
        purity=purity,
        silhouette=silhouette,
        silhouette_null_mean=null,
        probes=probes,
        n_structural=int(((test_labels >= 0) & (test_labels != unclassified_index)).sum()),
    )


def umap_figure(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...],
    path: str | Path,
    *,
    caption_metrics: dict[str, float] | None = None,
    seed: int = 0,
) -> Path | None:
    """Render the UMAP illustration, with its quantitative caption baked in.

    The caption is not decoration. A UMAP plot without the purity and probe numbers
    beside it invites the reader to judge cluster separation by eye, which is exactly the
    inference the projection is not entitled to support.

    Args:
        embeddings: ``[n, d]`` embeddings.
        labels: ``[n]`` typology indices; unlabelled and ``unclassified`` points are
            drawn in grey underneath so the structural classes are legible.
        class_names: Class index to name.
        path: Destination image path.
        caption_metrics: Numbers to print under the figure, e.g. kNN purity and probe
            accuracy.
        seed: UMAP random state.

    Returns:
        The written path, or None when UMAP or matplotlib is unavailable — the figure is
        an illustration and its absence must never fail a training run.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
    except ImportError:  # pragma: no cover - optional visualisation dependency
        return None

    mask = labels >= 0
    if mask.sum() < _MIN_UMAP_POINTS:
        return None

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed)
    projected = reducer.fit_transform(embeddings[mask])
    shown = labels[mask]

    unclassified_index = class_names.index("unclassified") if "unclassified" in class_names else -1
    figure, axis = plt.subplots(figsize=(8, 7))
    background = shown == unclassified_index
    axis.scatter(
        projected[background, 0],
        projected[background, 1],
        s=3,
        c="0.85",
        label="unclassified",
        rasterized=True,
    )
    colours = plt.get_cmap("tab10")
    for i, name in enumerate(class_names):
        if i == unclassified_index:
            continue
        points = shown == i
        if not points.any():
            continue
        axis.scatter(
            projected[points, 0],
            projected[points, 1],
            s=10,
            color=colours(i % 10),
            label=name,
            alpha=0.85,
        )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="upper right", fontsize=7, markerscale=2, framealpha=0.9)
    if caption_metrics:
        axis.set_xlabel(
            "  ".join(f"{k} = {v:.3f}" for k, v in caption_metrics.items()),
            fontsize=8,
        )
    figure.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def save_analysis(analysis: EmbeddingAnalysis, path: str | Path) -> Path:
    """Write an analysis report atomically.

    Args:
        analysis: The report.
        path: Destination file.

    Returns:
        The path written.
    """
    return write_json(path, analysis.to_dict())

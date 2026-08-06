"""Building, caching and loading the encoder's tensor view of the case corpus.

Two rules run through this module.

**Splits are read from the frozen manifest by id and never recomputed** (invariant 2,
D-006). :func:`load_case_ids` is the only door to a split in the whole encoder stack, it
goes through ``splits.load_split_manifest`` — which verifies each id list against its
committed sha256 — and it refuses to fall back to anything if the manifest is absent.
There is no code path here that could produce a split from a seed.

**The feature space is fitted on the training split alone.** Currency vocabularies,
payment-format vocabularies and per-currency amount statistics all come from training
edges. Fitting them on the corpus would let the test window's amount distribution
influence how a training case is encoded, which is a mild leak but a real one, and
invisible in every aggregate metric.

The cache is a per-split ``.pt`` of PyG ``Data`` objects plus the serialised feature
space. Building it costs a few minutes over 26,000 cases; every training run after that
reads it in seconds, which is what makes a six-arm three-seed sweep tractable on one GPU.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from g2t_aml.data.canonical import TYPOLOGY_VOCABULARY, CanonicalGraph
from g2t_aml.data.case_extraction import GraphIndex
from g2t_aml.data.case_sampling import CaseCollection
from g2t_aml.data.splits import load_split_manifest
from g2t_aml.models.encoder.features import (
    FEATURE_SPEC_VERSION,
    FeatureError,
    FeatureSpace,
    build_case_data,
    fit_feature_space,
)
from g2t_aml.utils.hashing import hash_id_list
from g2t_aml.utils.io import read_json, write_json

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import Data

#: Typology class order for the auxiliary head. Taken from the canonical vocabulary so
#: the head's class indices and the fact record's labels cannot drift apart.
TYPOLOGY_CLASSES: tuple[str, ...] = TYPOLOGY_VOCABULARY

#: The splits held in the frozen manifest.
MANIFEST_SPLITS: tuple[str, ...] = ("train", "val", "test")

#: The realistic-imbalance evaluation stream (D-023). Not in the manifest: it is a
#: separately built population over the test window, used for evaluation only and never
#: for training or model selection.
REALISTIC_SPLIT = "realistic_test"

#: Every split the cache holds.
ALL_SPLITS: tuple[str, ...] = (*MANIFEST_SPLITS, REALISTIC_SPLIT)


class DatasetError(RuntimeError):
    """Raised when the case corpus, the manifest and the cache do not agree."""


def typology_index(label: str | None) -> int:
    """Map a typology label to its class index.

    Args:
        label: A member of :data:`TYPOLOGY_CLASSES`, or None where no typology ground
            truth exists for the case.

    Returns:
        The class index, or -1 for None. -1 is the ``ignore_index`` the auxiliary loss
        uses, so a case without typology ground truth contributes to the binary head and
        not to the typology head, rather than being dropped from the batch entirely.
    """
    if label is None:
        return -1
    try:
        return TYPOLOGY_CLASSES.index(label)
    except ValueError:
        return -1


def load_case_ids(splits_dir: str | Path) -> dict[str, list[str]]:
    """Read the frozen split manifest and return its id lists.

    The only way a split is obtained anywhere in the encoder stack. The manifest's
    per-split content hashes are verified by ``load_split_manifest``, so a hand-edited
    manifest raises rather than quietly training on a different population.

    Args:
        splits_dir: Directory holding ``splits.json``, normally
            ``schemas/splits/<substrate>``.

    Returns:
        Split name to ordered case-id list, for the three manifest splits.

    Raises:
        DatasetError: If the manifest is absent. There is deliberately no fallback:
            invariant 2 says splits are never regenerated at runtime, so the correct
            response to a missing manifest is to stop.
        SplitError: If a split's id list does not match its recorded hash.
    """
    path = Path(splits_dir)
    if not (path / "splits.json").is_file():
        raise DatasetError(
            f"no frozen split manifest at {path}; run `make cases` to build one. "
            "Splits are never regenerated at runtime (invariant 2)."
        )
    manifest = load_split_manifest(path)
    return {name: list(manifest["splits"][name]["case_ids"]) for name in MANIFEST_SPLITS}


def load_typologies(facts_parquet: str | Path) -> dict[str, str]:
    """Read the per-case typology from the fact aggregate.

    Read from the **fact record**, not from ``CaseRecord.typology``. D-036 draws the
    distinction and PHASE_LOG quantifies it: the Phase 2 column names the typology of the
    *seeding stream*, while the fact record names what the case's own transactions
    actually show, and 1,447 cases carry a stream typology whose evidence fell outside the
    48-hour window. Training the auxiliary head on the seeding column would ask it to name
    a pattern that is not in its input.

    Args:
        facts_parquet: Path to ``facts.parquet``.

    Returns:
        Case id to typology label. Empty when the aggregate is absent, which leaves the
        auxiliary head unsupervised rather than failing the run.
    """
    path = Path(facts_parquet)
    if not path.is_file():
        return {}
    frame = pl.read_parquet(path, columns=["case_id", "typology"])
    return dict(zip(frame["case_id"].to_list(), frame["typology"].to_list(), strict=True))


@dataclass(frozen=True)
class CacheManifest:
    """What a built feature cache was built from.

    Attributes:
        feature_spec_version: :data:`FEATURE_SPEC_VERSION` at build time.
        dataset: Substrate key.
        source_manifest_hash: The interim graph the case positions index into.
        split_id_hashes: Per-split sha256 of the id list actually encoded, so a cache
            built against a different manifest is detected rather than reused.
        counts: Cases per split.
        positives: Suspicious cases per split.
        typology_counts: Typology label counts across all cached splits.
        lap_pe_dim: Laplacian components per node.
        rw_pe_dim: Random-walk steps per node.
    """

    feature_spec_version: str
    dataset: str
    source_manifest_hash: str
    split_id_hashes: dict[str, str]
    counts: dict[str, int]
    positives: dict[str, int]
    typology_counts: dict[str, int]
    lap_pe_dim: int
    rw_pe_dim: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form.

        Returns:
            Every field as a plain dict.
        """
        return dataclasses.asdict(self)


def _edges_of(collection: CaseCollection, index: GraphIndex, case_ids: list[str]) -> pl.DataFrame:
    """Return the interim edge rows belonging to the given cases, de-duplicated."""
    wanted = collection.edge_membership.filter(pl.col("case_id").is_in(case_ids))
    positions = wanted["edge_index"].unique().to_list()
    return index.edges[positions]


def build_feature_cache(
    # artifact; bundling them into a config object would only move the list.
    *,
    cases_dir: str | Path,
    realistic_dir: str | Path | None,
    interim_dir: str | Path,
    splits_dir: str | Path,
    facts_parquet: str | Path,
    out_dir: str | Path,
    lap_pe_dim: int = 8,
    rw_pe_dim: int = 16,
    limit: int | None = None,
    log: Any = None,
) -> CacheManifest:
    """Encode every case in the frozen splits and write the cache.

    Args:
        cases_dir: The Phase 2 case store.
        realistic_dir: The realistic-imbalance stream's store, or None to skip it.
        interim_dir: The ingested canonical graph the case positions index into.
        splits_dir: Directory holding the frozen ``splits.json``.
        facts_parquet: The fact aggregate, read for typology targets.
        out_dir: Destination for the cache.
        lap_pe_dim: Laplacian eigenvector components per node.
        rw_pe_dim: Random-walk steps per node.
        limit: Encode at most this many cases per split. For smoke runs only; a cache
            built with a limit records a different id hash and so cannot be mistaken for
            a full one.
        log: Optional logger.

    Returns:
        The cache manifest, also written to ``out_dir/cache_manifest.json``.

    Raises:
        DatasetError: If the case store and the interim graph disagree about which graph
            the case positions belong to.
        FeatureError: If a case cannot be encoded.
    """
    import torch

    collection = CaseCollection.load(cases_dir)
    index = GraphIndex(CanonicalGraph.load(interim_dir))
    if collection.dataset != index.graph.dataset:
        raise DatasetError(
            f"case store was cut from {collection.dataset!r} but the interim graph is "
            f"{index.graph.dataset!r}; positions are not portable between graphs"
        )

    ids = load_case_ids(splits_dir)
    known = set(collection.by_id())
    for name in MANIFEST_SPLITS:
        if missing := [c for c in ids[name] if c not in known]:
            raise DatasetError(
                f"{len(missing)} {name} case ids from the frozen manifest are absent "
                f"from the case store at {cases_dir}; first few: {missing[:3]}. "
                "Rebuild the cases, do not re-split."
            )
        if limit:
            ids[name] = ids[name][:limit]

    space = fit_feature_space(
        _edges_of(collection, index, ids["train"]),
        dataset=collection.dataset,
        availability=index.graph.availability.to_dict(),
        n_train_cases=len(ids["train"]),
        lap_pe_dim=lap_pe_dim,
        rw_pe_dim=rw_pe_dim,
    )

    typologies = load_typologies(facts_parquet)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "feature_space.json", space.to_dict(), canonical=True)

    counts: dict[str, int] = {}
    positives: dict[str, int] = {}
    typology_counts: dict[str, int] = {}
    id_hashes: dict[str, str] = {}

    def _encode(name: str, source: CaseCollection, case_ids: list[str]) -> None:
        by_id = source.by_id()
        graphs: list[Data] = []
        n_positive = 0
        for i, case_id in enumerate(case_ids, start=1):
            record = by_id[case_id]
            graph = source.materialise(case_id, index)
            label = 1 if record.label == "suspicious" else 0
            n_positive += label
            typology = typologies.get(case_id)
            typology_counts[typology or "no_fact_record"] = (
                typology_counts.get(typology or "no_fact_record", 0) + 1
            )
            graphs.append(
                build_case_data(
                    graph,
                    space,
                    seed_node=record.seed_node,
                    label=label,
                    typology_index=typology_index(typology),
                )
            )
            if log is not None and i % 2000 == 0:
                log.info("  %s: %d / %d", name, i, len(case_ids))
        torch.save(graphs, out / f"{name}.pt")
        counts[name] = len(graphs)
        positives[name] = n_positive
        id_hashes[name] = hash_id_list(case_ids)

    for name in MANIFEST_SPLITS:
        if log is not None:
            log.info("encoding %s (%d cases)", name, len(ids[name]))
        _encode(name, collection, ids[name])

    if realistic_dir is not None and (Path(realistic_dir) / "cases.jsonl").is_file():
        stream = CaseCollection.load(realistic_dir)
        stream_ids = stream.case_ids[:limit] if limit else stream.case_ids
        if log is not None:
            log.info("encoding %s (%d cases)", REALISTIC_SPLIT, len(stream_ids))
        _encode(REALISTIC_SPLIT, stream, stream_ids)

    manifest = CacheManifest(
        feature_spec_version=FEATURE_SPEC_VERSION,
        dataset=collection.dataset,
        source_manifest_hash=collection.source_manifest_hash,
        split_id_hashes=id_hashes,
        counts=counts,
        positives=positives,
        typology_counts=dict(sorted(typology_counts.items())),
        lap_pe_dim=lap_pe_dim,
        rw_pe_dim=rw_pe_dim,
    )
    write_json(out / "cache_manifest.json", manifest.to_dict(), canonical=True)
    return manifest


def load_feature_space(cache_dir: str | Path) -> FeatureSpace:
    """Read the fitted feature space that a cache was built with.

    Args:
        cache_dir: The cache directory.

    Returns:
        The feature space.

    Raises:
        DatasetError: If the cache has not been built.
        FeatureError: If the space was written by a different feature-spec version.
    """
    path = Path(cache_dir) / "feature_space.json"
    if not path.is_file():
        raise DatasetError(
            f"no feature space at {path}; run `make encoder-features` to build the cache"
        )
    return FeatureSpace.from_dict(read_json(path))


def load_split(cache_dir: str | Path, split: str) -> list[Data]:
    """Read one cached split.

    Args:
        cache_dir: The cache directory.
        split: A member of :data:`ALL_SPLITS`.

    Returns:
        The encoded cases, in manifest order.

    Raises:
        DatasetError: If the split has not been cached.
    """
    import torch

    path = Path(cache_dir) / f"{split}.pt"
    if not path.is_file():
        raise DatasetError(f"split {split!r} is not in the feature cache at {cache_dir}")
    # weights_only=False: these are PyG Data objects carrying python lists of node ids,
    # written by this repository's own build step, not third-party checkpoints.
    return torch.load(path, weights_only=False)


def verify_cache_against_manifest(cache_dir: str | Path, splits_dir: str | Path) -> None:
    """Check that a cache was built from the current frozen manifest.

    A cache is only as trustworthy as the split it encodes. Comparing id-list hashes
    catches the case where the manifest was rebuilt and the cache was not, which would
    otherwise train on one population and report metrics against another.

    Args:
        cache_dir: The cache directory.
        splits_dir: Directory holding ``splits.json``.

    Raises:
        DatasetError: If the cache manifest is absent, or a split's id hash differs from
            the frozen manifest's.
    """
    path = Path(cache_dir) / "cache_manifest.json"
    if not path.is_file():
        raise DatasetError(f"no cache manifest at {path}; rebuild the feature cache")
    cached = read_json(path)
    if cached.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise FeatureError(
            f"feature cache at {cache_dir} was built by feature spec "
            f"{cached.get('feature_spec_version')!r}, code is at {FEATURE_SPEC_VERSION!r}"
        )
    frozen = load_split_manifest(splits_dir)
    for name in MANIFEST_SPLITS:
        expected = frozen["splits"][name]["id_list_sha256"]
        actual = cached["split_id_hashes"].get(name)
        if actual != expected:
            raise DatasetError(
                f"feature cache split {name!r} was built from a different id list than "
                f"the frozen manifest ({actual} != {expected}); rebuild the cache"
            )

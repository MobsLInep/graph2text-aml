"""The canonical graph representation both substrates map into.

Everything downstream of Phase 1 — fact extraction, corpus generation, the GAT encoder —
consumes :class:`CanonicalGraph` and never a substrate-specific frame. That is what keeps
those modules substrate-agnostic.

The load-bearing piece is :class:`AvailabilityMask`. AMLworld and Elliptic2 do not support
the same classes of assertion: Elliptic2 has no amounts, no currencies, no wall-clock
timestamps and no entity types, because its features are anonymised. Invariant 4 says
nothing may assert a fact that does not exist for its substrate, and the mask is how that
invariant is carried from the loader all the way to the generator. Downstream code MUST
consult the mask rather than assuming a column is meaningful because it is present.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from g2t_aml.utils.io import atomic_path, read_json, write_json

# Bumping this invalidates every interim artifact written under the old layout, in the
# same way CASE_FACTS_SCHEMA_VERSION invalidates a generated corpus (invariant 3).
CANONICAL_SCHEMA_VERSION = "0.1.0"

#: Columns every canonical node table carries, in order. Substrate-specific columns are
#: appended after these and named in ``node_feature_names``.
NODE_REQUIRED_COLUMNS: tuple[str, ...] = ("node_id", "node_type")

#: Columns every canonical edge table carries, in order.
EDGE_REQUIRED_COLUMNS: tuple[str, ...] = ("src", "dst")

#: The controlled typology vocabulary. ``unclassified`` is a first-class member: AMLworld
#: labels 1,968 HI-Small transactions as laundering while matching none of the eight
#: structural patterns, and a narrative must be able to say so rather than guess.
TYPOLOGY_VOCABULARY: tuple[str, ...] = (
    "fan_out",
    "fan_in",
    "gather_scatter",
    "scatter_gather",
    "cycle",
    "random",
    "bipartite",
    "stack",
    "unclassified",
)


@dataclass(frozen=True)
class AvailabilityMask:
    """Which classes of fact are derivable for this substrate.

    Downstream code MUST consult this before asserting anything. A ``False`` flag means
    the underlying quantity does not exist in the source data at all — not that it is
    missing for a particular row. Asserting a masked-out fact is a hallucination by
    construction, and the corpus verifier rejects it.
    """

    absolute_timestamps: bool
    fine_temporal_resolution: bool
    """Hour-level or better. Ordering may still be available when this is False."""
    monetary_amounts: bool
    currencies: bool
    institution_identity: bool
    entity_types: bool
    """Mixer / exchange / merchant business-type labels."""
    node_labels: bool
    typology_ground_truth: bool
    semantic_node_features: bool
    """Whether the substrate *supplies* node features whose columns have published
    meanings. False for Elliptic2 because its features are anonymised, and False for
    AMLworld because it ships no node feature table at all — its node attributes are
    aggregates we derive from the transactions. This flag does not govern those derived
    aggregates: a statement about an account's degree or total sent is licensed by
    ``monetary_amounts`` and the edge data it was computed from, not by this flag. What it
    forbids is attaching a named meaning to a supplied feature column."""

    def to_dict(self) -> dict[str, bool]:
        """Return the mask as a plain dict.

        Returns:
            Field name to flag, in declaration order.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvailabilityMask:
        """Rebuild a mask from a dict.

        Args:
            data: Mapping with exactly the mask's field names.

        Returns:
            The reconstructed mask.

        Raises:
            ValueError: If a field is missing or unknown, so a stale serialised mask
                fails loudly rather than defaulting a flag to False.
        """
        expected = {f.name for f in dataclasses.fields(cls)}
        got = set(data)
        if missing := expected - got:
            raise ValueError(f"availability mask is missing fields: {sorted(missing)}")
        if unknown := got - expected:
            raise ValueError(f"availability mask has unknown fields: {sorted(unknown)}")
        return cls(**{k: bool(data[k]) for k in expected})

    def to_config_mask(self) -> dict[str, bool]:
        """Project onto the eight-key vocabulary used by ``configs/data/*.yaml``.

        The Hydra configs predate this dataclass and use a coarser vocabulary, asserted by
        ``REQUIRED_AVAILABILITY_KEYS`` in the composition tests. Two vocabularies that
        drift apart would be worse than one imperfect vocabulary, so the projection is
        defined here and a test asserts the two agree for both substrates.

        ``account_ids`` has no counterpart: both substrates carry node identifiers, so it
        is always True. Everything else maps one-to-one.

        Returns:
            The config-shaped mask.
        """
        return {
            "amounts": self.monetary_amounts,
            "currencies": self.currencies,
            "real_timestamps": self.absolute_timestamps,
            "bank_ids": self.institution_identity,
            "entity_types": self.entity_types,
            "account_ids": True,
            "typology_labels": self.typology_ground_truth,
            "node_features": self.semantic_node_features,
        }

    def assert_available(self, *fields: str) -> None:
        """Raise unless every named fact class is available.

        Args:
            *fields: Mask field names required by the caller.

        Raises:
            ValueError: If a name is not a mask field.
            PermissionError: If any named field is False. This is deliberately not a
                ValueError: it marks an invariant-4 violation, so it is easy to grep for
                and impossible to catch by accident alongside ordinary data errors.
        """
        known = {f.name for f in dataclasses.fields(self)}
        if unknown := set(fields) - known:
            raise ValueError(f"not availability mask fields: {sorted(unknown)}")
        if denied := [f for f in fields if not getattr(self, f)]:
            raise PermissionError(
                f"substrate does not support these fact classes: {sorted(denied)} "
                "(invariant 4: nothing may assert a fact that does not exist)"
            )


#: AMLworld: synthetic but complete. Bank identity is present, so institution_identity is
#: True. There are no business-type labels on accounts — the schema carries a bank code
#: and nothing that says "exchange" or "mixer" — so entity_types is False. Account
#: identifiers are opaque hex strings with no semantics, and the node table's features are
#: derived by us from transaction aggregates rather than supplied, so
#: semantic_node_features is False as well.
AMLWORLD_AVAILABILITY = AvailabilityMask(
    absolute_timestamps=True,
    fine_temporal_resolution=True,
    monetary_amounts=True,
    currencies=True,
    institution_identity=True,
    entity_types=False,
    node_labels=True,
    typology_ground_truth=True,
    semantic_node_features=False,
)

#: Elliptic2: real Bitcoin, and almost entirely masked. The unit is a *cluster* of
#: addresses, features are anonymised, and the timestamps are coarse step indices rather
#: than wall-clock times. Only the subgraph-level licit/suspicious label survives.
ELLIPTIC2_AVAILABILITY = AvailabilityMask(
    absolute_timestamps=False,
    fine_temporal_resolution=False,
    monetary_amounts=False,
    currencies=False,
    institution_identity=False,
    entity_types=False,
    node_labels=True,
    typology_ground_truth=False,
    semantic_node_features=False,
)


@dataclass
class CanonicalGraph:
    """A substrate-agnostic graph: node table, edge table, and what may be said about it.

    Attributes:
        graph_id: Unique within ``dataset``.
        dataset: Substrate key, e.g. ``"amlworld_hi_small"`` or ``"elliptic2"``.
        nodes: One row per node. Carries at least ``node_id`` and ``node_type``.
        edges: One row per edge. Carries at least ``src`` and ``dst``, both referencing
            ``nodes.node_id``.
        node_feature_names: Node columns that are model features, in a fixed order.
        edge_feature_names: Edge columns that are model features, in a fixed order.
        availability: What may be asserted about this graph. See invariant 4.
        label: Case-level label when one exists, else None.
        typology: A member of :data:`TYPOLOGY_VOCABULARY`, or None when unknown. Note the
            difference from ``"unclassified"``: None means no typology ground truth exists
            for this substrate at all; ``"unclassified"`` means it exists and says the
            activity matches no structural pattern.
        provenance: Free-form record of where this came from — source files, checksums,
            loader version. Written into the manifest.
    """

    graph_id: str
    dataset: str
    nodes: pl.DataFrame
    edges: pl.DataFrame
    node_feature_names: list[str]
    edge_feature_names: list[str]
    availability: AvailabilityMask
    label: str | None = None
    typology: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate structural invariants at construction time.

        Raises:
            ValueError: If a required column is absent, a declared feature name is not a
                column, or ``typology`` is outside the controlled vocabulary.
        """
        for column in NODE_REQUIRED_COLUMNS:
            if column not in self.nodes.columns:
                raise ValueError(f"node table is missing required column {column!r}")
        for column in EDGE_REQUIRED_COLUMNS:
            if column not in self.edges.columns:
                raise ValueError(f"edge table is missing required column {column!r}")
        if missing := [c for c in self.node_feature_names if c not in self.nodes.columns]:
            raise ValueError(f"declared node features absent from node table: {missing}")
        if missing := [c for c in self.edge_feature_names if c not in self.edges.columns]:
            raise ValueError(f"declared edge features absent from edge table: {missing}")
        if self.typology is not None and self.typology not in TYPOLOGY_VOCABULARY:
            raise ValueError(
                f"typology {self.typology!r} is outside the controlled vocabulary "
                f"{TYPOLOGY_VOCABULARY}"
            )

    @property
    def num_nodes(self) -> int:
        """Return the node count.

        Returns:
            Number of rows in the node table.
        """
        return self.nodes.height

    @property
    def num_edges(self) -> int:
        """Return the edge count.

        Returns:
            Number of rows in the edge table.
        """
        return self.edges.height

    def validate_referential_integrity(self) -> None:
        """Check that every edge endpoint exists in the node table.

        This is O(E) and is not run automatically in ``__post_init__``, because the
        full HI-Small graph has five million edges and the check is only worth paying
        for at ingest time and in tests.

        Raises:
            ValueError: If any ``src`` or ``dst`` is absent from ``nodes.node_id``.
        """
        known = set(self.nodes["node_id"].to_list())
        for side in ("src", "dst"):
            unknown = set(self.edges[side].to_list()) - known
            if unknown:
                sample = sorted(str(u) for u in unknown)[:5]
                raise ValueError(
                    f"{len(unknown)} edge {side} endpoints are not in the node table; "
                    f"first few: {sample}"
                )

    # ------------------------------------------------------------------ io ---

    def save(self, directory: str | Path) -> Path:
        """Write the graph to a directory as Parquet plus a JSON sidecar.

        Node and edge tables go to ``nodes.parquet`` / ``edges.parquet``; everything else
        goes to ``canonical.json``. Both writes are atomic (utils/io discipline), so a
        killed ingest never leaves a half-written table a later phase would treat as
        valid.

        Args:
            directory: Destination directory, created if absent.

        Returns:
            The directory written to.

        Raises:
            OSError: If a write or rename fails.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        with atomic_path(out / "nodes.parquet", suffix=".parquet.tmp") as tmp:
            self.nodes.write_parquet(tmp, compression="zstd")
        with atomic_path(out / "edges.parquet", suffix=".parquet.tmp") as tmp:
            self.edges.write_parquet(tmp, compression="zstd")
        write_json(
            out / "canonical.json",
            {
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "graph_id": self.graph_id,
                "dataset": self.dataset,
                "node_feature_names": self.node_feature_names,
                "edge_feature_names": self.edge_feature_names,
                "availability": self.availability.to_dict(),
                "label": self.label,
                "typology": self.typology,
                "provenance": self.provenance,
            },
        )
        return out

    @classmethod
    def load(cls, directory: str | Path) -> CanonicalGraph:
        """Read a graph written by :meth:`save`.

        Args:
            directory: Directory containing ``nodes.parquet``, ``edges.parquet`` and
                ``canonical.json``.

        Returns:
            The reconstructed graph.

        Raises:
            FileNotFoundError: If any of the three files is absent.
            ValueError: If the sidecar was written by an incompatible schema version, or
                the reconstructed graph fails validation.
        """
        src = Path(directory)
        meta = read_json(src / "canonical.json")
        if meta.get("schema_version") != CANONICAL_SCHEMA_VERSION:
            raise ValueError(
                f"canonical schema version mismatch: file has "
                f"{meta.get('schema_version')!r}, code expects {CANONICAL_SCHEMA_VERSION!r}"
            )
        return cls(
            graph_id=meta["graph_id"],
            dataset=meta["dataset"],
            nodes=pl.read_parquet(src / "nodes.parquet"),
            edges=pl.read_parquet(src / "edges.parquet"),
            node_feature_names=list(meta["node_feature_names"]),
            edge_feature_names=list(meta["edge_feature_names"]),
            availability=AvailabilityMask.from_dict(meta["availability"]),
            label=meta["label"],
            typology=meta["typology"],
            provenance=dict(meta["provenance"]),
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact description for logs and manifests.

        Returns:
            Identity, counts, label/typology and the availability mask.
        """
        return {
            "graph_id": self.graph_id,
            "dataset": self.dataset,
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "label": self.label,
            "typology": self.typology,
            "node_feature_names": list(self.node_feature_names),
            "edge_feature_names": list(self.edge_feature_names),
            "availability": self.availability.to_dict(),
        }

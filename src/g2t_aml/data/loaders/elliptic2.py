"""Elliptic2 loader: labelled Bitcoin cluster subgraphs over a large background graph.

Elliptic2 is access-gated and not redistributable, so this loader is written against the
documented file schema and its tests skip when the files are absent. It is deliberately
conservative about what it claims.

**The unit is a cluster, not an address and not a transaction.** A node is a set of
Bitcoin addresses believed to share ownership. Language like "this account" or "this
wallet" is wrong for this substrate.

**Node features are anonymised.** The feature columns arrive as ``feat_0 ... feat_n`` with
no published semantics. This module never maps a column index onto a named financial
quantity, and :data:`ELLIPTIC2_AVAILABILITY` sets ``semantic_node_features=False`` so
downstream code cannot either. If a future release documents the columns, that is a
decision entry and a mask change, not a quiet rename here.

**The background graph does not fit in memory.** 49M nodes and 196M edges will not sit
comfortably in 32 GB alongside anything else, so background access is lazy by default:
:func:`load_background_graph` returns Polars ``LazyFrame`` scans, and
:func:`build_subgraph` pushes its filter down into the scan rather than materialising.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from g2t_aml.data.canonical import ELLIPTIC2_AVAILABILITY, CanonicalGraph

#: Files the official distribution unzips to, under ``data/raw/elliptic2/``.
NODES_FILE = "nodes.csv"
EDGES_FILE = "edges.csv"
COMPONENTS_FILE = "connected_components.csv"
BACKGROUND_NODES_FILE = "background_nodes.csv"
BACKGROUND_EDGES_FILE = "background_edges.csv"

REQUIRED_FILES: tuple[str, ...] = (
    NODES_FILE,
    EDGES_FILE,
    COMPONENTS_FILE,
    BACKGROUND_NODES_FILE,
    BACKGROUND_EDGES_FILE,
)

#: Subgraph-level labels. Per the construction procedure: a subgraph is licit when its
#: senders and receivers are all licit; suspicious when its receivers are licit but its
#: senders are illicit. Note what that does *not* say — "suspicious" is a property of the
#: money's provenance, not a proven offence by any cluster in the subgraph.
LABEL_LICIT = "licit"
LABEL_SUSPICIOUS = "suspicious"
LABELS: tuple[str, ...] = (LABEL_LICIT, LABEL_SUSPICIOUS)

#: Published headline figures, for the data card's observed-vs-published table.
PUBLISHED_STATISTICS: dict[str, int] = {
    "num_labelled_subgraphs": 122_000,
    "num_background_nodes": 49_000_000,
    "num_background_edges": 196_000_000,
}

#: Candidate column names for the cluster identifier, in preference order. The official
#: tooling has used more than one spelling across releases, so the loader probes rather
#: than hardcoding — and fails loudly if none is present.
_ID_CANDIDATES: tuple[str, ...] = ("clusterId", "cluster_id", "nodeId", "node_id", "id")
_COMPONENT_CANDIDATES: tuple[str, ...] = ("ccId", "cc_id", "componentId", "component_id")
_LABEL_CANDIDATES: tuple[str, ...] = ("ccLabel", "cc_label", "label", "class")
_SRC_CANDIDATES: tuple[str, ...] = ("clusterId1", "src", "source", "srcId", "txId1")
_DST_CANDIDATES: tuple[str, ...] = ("clusterId2", "dst", "target", "dstId", "txId2")


class Elliptic2UnavailableError(FileNotFoundError):
    """Raised when Elliptic2 files are absent. Carries the access instructions."""


class Elliptic2SchemaError(ValueError):
    """Raised when a file is present but lacks an expected column."""


@dataclass(frozen=True)
class BackgroundGraph:
    """Lazy handles on the background node and edge tables.

    Attributes:
        nodes: Scan over ``background_nodes.csv``. Not materialised.
        edges: Scan over ``background_edges.csv``. Not materialised.
        node_id_column: Resolved cluster-identifier column in ``nodes``.
        src_column: Resolved source column in ``edges``.
        dst_column: Resolved destination column in ``edges``.
        feature_columns: Anonymised feature columns, in file order. Their meaning is
            unknown and must stay unknown; see the module docstring.
    """

    nodes: pl.LazyFrame
    edges: pl.LazyFrame
    node_id_column: str
    src_column: str
    dst_column: str
    feature_columns: tuple[str, ...]

    def collect_counts(self) -> dict[str, int]:
        """Count background nodes and edges.

        This is a full streaming pass over ~245M rows and takes minutes. It is called at
        ingest time to fill the data card, never on a hot path.

        Returns:
            ``{"num_background_nodes": int, "num_background_edges": int}``.
        """
        nodes = self.nodes.select(pl.len()).collect(streaming=True).item()
        edges = self.edges.select(pl.len()).collect(streaming=True).item()
        return {"num_background_nodes": int(nodes), "num_background_edges": int(edges)}


def dataset_dir(raw_dir: str | Path) -> Path:
    """Return the directory holding Elliptic2's files.

    Args:
        raw_dir: The ``paths.raw_dir`` root.

    Returns:
        ``raw_dir/elliptic2``.
    """
    return Path(raw_dir) / "elliptic2"


def is_available(raw_dir: str | Path) -> bool:
    """Report whether every required Elliptic2 file is present.

    Cheap and non-raising, so it is safe inside a ``pytest.mark.skipif`` condition.

    Args:
        raw_dir: The ``paths.raw_dir`` root.

    Returns:
        True if all five files exist.
    """
    root = dataset_dir(raw_dir)
    return all((root / name).exists() for name in REQUIRED_FILES)


def _require(raw_dir: str | Path, name: str) -> Path:
    """Return a required file's path, or explain how to obtain it.

    Args:
        raw_dir: The ``paths.raw_dir`` root.
        name: Filename.

    Returns:
        The existing path.

    Raises:
        Elliptic2UnavailableError: If the file is absent.
    """
    path = dataset_dir(raw_dir) / name
    if not path.exists():
        from g2t_aml.data.download import REGISTRY

        raise Elliptic2UnavailableError(
            f"Elliptic2 file {name} not found at {path}\n\n{REGISTRY['elliptic2'].acquisition}"
        )
    return path


def _resolve(columns: list[str], candidates: tuple[str, ...], *, role: str, path: Path) -> str:
    """Pick the first candidate column present, case-insensitively.

    Args:
        columns: Columns actually in the file.
        candidates: Accepted names, in preference order.
        role: What the column is for, used in the error message.
        path: File being read, used in the error message.

    Returns:
        The matching column name as it appears in the file.

    Raises:
        Elliptic2SchemaError: If no candidate is present. The loader will not guess by
            position — a wrong guess here silently mislabels the whole graph.
    """
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise Elliptic2SchemaError(
        f"{path}: no {role} column found. Looked for {list(candidates)}; "
        f"file has {columns}. Refusing to guess by position."
    )


def _feature_columns(columns: list[str], exclude: set[str]) -> tuple[str, ...]:
    """Return the anonymised feature columns.

    Args:
        columns: All columns in the node file.
        exclude: Identifier and label columns to leave out.

    Returns:
        Remaining columns in file order. Deliberately returned as opaque names; no
        semantic interpretation is attached anywhere in this module.
    """
    return tuple(c for c in columns if c not in exclude)


def load_background_graph(raw_dir: str | Path) -> BackgroundGraph:
    """Open lazy scans over the background node and edge tables.

    Nothing is read into memory: the returned frames are Polars scans, so a caller may
    filter and only ever materialise the rows it needs. Materialising either table whole
    is not supported and should not be attempted — 49M nodes and 196M edges will not fit
    alongside anything else in 32 GB.

    Args:
        raw_dir: The ``paths.raw_dir`` root.

    Returns:
        A :class:`BackgroundGraph` with resolved column names.

    Raises:
        Elliptic2UnavailableError: If either background file is absent.
        Elliptic2SchemaError: If an identifier column cannot be resolved.
    """
    nodes_path = _require(raw_dir, BACKGROUND_NODES_FILE)
    edges_path = _require(raw_dir, BACKGROUND_EDGES_FILE)

    nodes = pl.scan_csv(nodes_path)
    edges = pl.scan_csv(edges_path)
    node_columns = nodes.collect_schema().names()
    edge_columns = edges.collect_schema().names()

    node_id = _resolve(node_columns, _ID_CANDIDATES, role="cluster id", path=nodes_path)
    src = _resolve(edge_columns, _SRC_CANDIDATES, role="edge source", path=edges_path)
    dst = _resolve(edge_columns, _DST_CANDIDATES, role="edge destination", path=edges_path)

    return BackgroundGraph(
        nodes=nodes,
        edges=edges,
        node_id_column=node_id,
        src_column=src,
        dst_column=dst,
        feature_columns=_feature_columns(node_columns, {node_id}),
    )


def load_labelled_subgraphs(raw_dir: str | Path) -> pl.DataFrame:
    """Load the labelled subgraphs and their cluster memberships.

    Args:
        raw_dir: The ``paths.raw_dir`` root.

    Returns:
        One row per (subgraph, cluster) membership, with columns ``subgraph_id`` (Utf8),
        ``node_id`` (Utf8) and ``label`` (Utf8, a member of :data:`LABELS`). The frame is
        materialised: 122K subgraphs of memberships is small, unlike the background graph.

    Raises:
        Elliptic2UnavailableError: If ``connected_components.csv`` is absent.
        Elliptic2SchemaError: If the component, cluster or label column cannot be
            resolved, or a label value is outside :data:`LABELS`.
    """
    path = _require(raw_dir, COMPONENTS_FILE)
    frame = pl.read_csv(path, infer_schema_length=0)
    columns = frame.columns

    component = _resolve(columns, _COMPONENT_CANDIDATES, role="component id", path=path)
    node_id = _resolve(columns, _ID_CANDIDATES, role="cluster id", path=path)
    label = _resolve(columns, _LABEL_CANDIDATES, role="label", path=path)

    out = frame.select(
        pl.col(component).cast(pl.Utf8).alias("subgraph_id"),
        pl.col(node_id).cast(pl.Utf8).alias("node_id"),
        _normalise_label(pl.col(label)).alias("label"),
    )
    if unknown := set(out["label"].unique().to_list()) - set(LABELS):
        raise Elliptic2SchemaError(
            f"{path}: unrecognised subgraph labels {sorted(unknown)}; expected {list(LABELS)}"
        )
    return out


def _normalise_label(column: pl.Expr) -> pl.Expr:
    """Map the source label encoding onto :data:`LABELS`.

    The distribution has used both string labels and a 1/2 integer encoding across
    releases. Anything unrecognised is passed through unchanged so that
    :func:`load_labelled_subgraphs` can reject it by name rather than silently defaulting
    it to ``licit`` — defaulting a label is how a suspicious subgraph becomes invisible.

    Args:
        column: Expression yielding the raw label.

    Returns:
        An expression yielding a normalised label.
    """
    lowered = column.cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    return (
        pl.when(lowered.is_in(["1", "suspicious", "illicit"]))
        .then(pl.lit(LABEL_SUSPICIOUS))
        .when(lowered.is_in(["2", "licit"]))
        .then(pl.lit(LABEL_LICIT))
        .otherwise(lowered)
    )


def subgraph_labels(memberships: pl.DataFrame) -> pl.DataFrame:
    """Reduce memberships to one row per subgraph.

    Args:
        memberships: Frame from :func:`load_labelled_subgraphs`.

    Returns:
        Columns ``subgraph_id``, ``label`` and ``num_nodes``, sorted by ``subgraph_id``.

    Raises:
        Elliptic2SchemaError: If a subgraph carries more than one distinct label, which
            would mean the membership file is inconsistent.
    """
    grouped = memberships.group_by("subgraph_id").agg(
        pl.col("label").unique().alias("labels"),
        pl.len().alias("num_nodes"),
    )
    conflicting = grouped.filter(pl.col("labels").list.len() > 1)
    if not conflicting.is_empty():
        ids = conflicting["subgraph_id"].to_list()[:5]
        raise Elliptic2SchemaError(
            f"{conflicting.height} subgraphs carry conflicting labels; first few: {ids}"
        )
    return grouped.select(
        "subgraph_id",
        pl.col("labels").list.first().alias("label"),
        "num_nodes",
    ).sort("subgraph_id")


def build_subgraph(
    subgraph_id: str,
    *,
    raw_dir: str | Path,
    memberships: pl.DataFrame | None = None,
    background: BackgroundGraph | None = None,
) -> CanonicalGraph:
    """Build the canonical graph for one labelled subgraph.

    The member cluster ids are resolved first, then the background scans are filtered down
    to those ids. The filter is pushed into the scan, so the background tables are streamed
    rather than materialised.

    Args:
        subgraph_id: Connected-component identifier.
        raw_dir: The ``paths.raw_dir`` root.
        memberships: Pre-loaded frame from :func:`load_labelled_subgraphs`. Pass it when
            building many subgraphs, to avoid re-reading the file each time.
        background: Pre-opened background scans, likewise.

    Returns:
        A :class:`CanonicalGraph` whose nodes are clusters carrying anonymised features,
        whose edges are background edges internal to the subgraph, and whose availability
        mask is :data:`ELLIPTIC2_AVAILABILITY` — everything semantic masked off except the
        subgraph label. ``typology`` is None, not ``"unclassified"``: Elliptic2 has no
        typology ground truth at all, which is a different statement from "flagged but
        matching no pattern".

    Raises:
        Elliptic2UnavailableError: If a required file is absent.
        KeyError: If ``subgraph_id`` is not in the membership table.
        Elliptic2SchemaError: If a column cannot be resolved.
    """
    memberships = load_labelled_subgraphs(raw_dir) if memberships is None else memberships
    background = load_background_graph(raw_dir) if background is None else background

    members = memberships.filter(pl.col("subgraph_id") == subgraph_id)
    if members.is_empty():
        raise KeyError(f"subgraph {subgraph_id!r} is not in the membership table")
    member_ids = members["node_id"].unique().to_list()
    label = members["label"][0]

    nodes = (
        background.nodes.filter(pl.col(background.node_id_column).cast(pl.Utf8).is_in(member_ids))
        .collect(streaming=True)
        .rename({background.node_id_column: "node_id"})
        .with_columns(
            pl.col("node_id").cast(pl.Utf8),
            # Every node is a cluster of addresses. Not an account, not a wallet.
            pl.lit("cluster").alias("node_type"),
        )
    )
    edges = (
        background.edges.filter(
            pl.col(background.src_column).cast(pl.Utf8).is_in(member_ids)
            & pl.col(background.dst_column).cast(pl.Utf8).is_in(member_ids)
        )
        .collect(streaming=True)
        .rename({background.src_column: "src", background.dst_column: "dst"})
        .with_columns(pl.col("src").cast(pl.Utf8), pl.col("dst").cast(pl.Utf8))
    )

    ordered = ["node_id", "node_type", *background.feature_columns]
    nodes = nodes.select([c for c in ordered if c in nodes.columns])

    return CanonicalGraph(
        graph_id=f"elliptic2_{subgraph_id}",
        dataset="elliptic2",
        nodes=nodes,
        edges=edges,
        # Feature names are carried verbatim and never interpreted.
        node_feature_names=[c for c in background.feature_columns if c in nodes.columns],
        edge_feature_names=[],
        availability=ELLIPTIC2_AVAILABILITY,
        label=label,
        typology=None,
        provenance={
            "subgraph_id": subgraph_id,
            "source": "elliptic2",
            "unit": "bitcoin address cluster",
            "features": "anonymised; column semantics are not published",
        },
    )

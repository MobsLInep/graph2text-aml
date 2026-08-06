"""Conversion from :class:`CanonicalGraph` to PyTorch Geometric.

This lives in its own module so that importing ``g2t_aml.data`` never imports torch.
Phases 1-6 and 10 are CPU-only and must stay installable without CUDA (D-004); the graph
extra is optional, and everything except this file works without it. Importing this module
in an environment without the ``graph`` extra raises at import time with a clear message
rather than failing obscurely later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from g2t_aml.data.canonical import CanonicalGraph

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import HeteroData

_IMPORT_HINT = (
    "torch and torch-geometric are required for g2t_aml.data.pyg_adapter. They live in "
    "the optional `graph` extra, which the CPU-only default install omits by design "
    "(DECISIONS.md D-004). Install with: make install-gpu"
)


def _require_torch() -> tuple[Any, Any]:
    """Import torch and PyG, or explain how to install them.

    Returns:
        The ``torch`` module and the ``HeteroData`` class.

    Raises:
        ImportError: If either package is unavailable.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_IMPORT_HINT) from exc
    return torch, HeteroData


def to_pyg(
    graph: CanonicalGraph,
    *,
    node_type: str | None = None,
    edge_type: str = "transacts",
) -> HeteroData:
    """Convert a canonical graph to a PyG ``HeteroData``.

    ``HeteroData`` is used even though both substrates are currently single-node-type:
    AMLworld already distinguishes accounts from banks in its raw schema, and a later
    phase adding a bank node type should not have to change every consumer. Node
    identifiers are mapped to contiguous integer indices; the original ids are kept on the
    store as ``node_id`` so a prediction can be traced back to an account or cluster.

    The availability mask is carried across as ``graph.availability``. Invariant 4 does not
    stop applying because the data is now a tensor.

    Args:
        graph: The graph to convert.
        node_type: Node-store name. Defaults to the graph's own ``node_type`` value when
            it is uniform, else ``"node"``.
        edge_type: Relation name for the edge store.

    Returns:
        A ``HeteroData`` with one node store and one edge store. Node features are the
        columns named in ``graph.node_feature_names``, cast to float32; likewise edges.
        A graph with no declared features gets no ``x`` tensor rather than a zero-width
        one, since PyG handles an absent ``x`` more gracefully than an empty one.

    Raises:
        ImportError: If the ``graph`` extra is not installed.
        ValueError: If an edge endpoint is not present in the node table.
    """
    torch, hetero_data = _require_torch()
    data = hetero_data()

    if node_type is None:
        types = graph.nodes["node_type"].unique().to_list() if graph.num_nodes else []
        node_type = str(types[0]) if len(types) == 1 else "node"

    node_ids = graph.nodes["node_id"].to_list()
    index = {node: i for i, node in enumerate(node_ids)}

    store = data[node_type]
    store.num_nodes = graph.num_nodes
    store.node_id = node_ids
    if graph.node_feature_names:
        store.x = torch.tensor(
            graph.nodes.select(graph.node_feature_names).fill_null(0.0).cast(pl.Float32).to_numpy(),
            dtype=torch.float32,
        )

    try:
        src = [index[s] for s in graph.edges["src"].to_list()]
        dst = [index[d] for d in graph.edges["dst"].to_list()]
    except KeyError as exc:
        raise ValueError(
            f"edge endpoint {exc.args[0]!r} is not in the node table; "
            "call graph.validate_referential_integrity() to find them all"
        ) from exc

    relation = (node_type, edge_type, node_type)
    edge_store = data[relation]
    edge_store.edge_index = torch.tensor([src, dst], dtype=torch.long)
    if graph.edge_feature_names:
        edge_store.edge_attr = torch.tensor(
            graph.edges.select(graph.edge_feature_names).fill_null(0.0).cast(pl.Float32).to_numpy(),
            dtype=torch.float32,
        )
    if "is_laundering" in graph.edges.columns:
        edge_store.y = torch.tensor(
            graph.edges["is_laundering"].fill_null(False).cast(pl.Int8).to_numpy(),
            dtype=torch.long,
        )

    # Invariant 4 travels with the tensors.
    data.graph_id = graph.graph_id
    data.dataset = graph.dataset
    data.availability = graph.availability
    data.label = graph.label
    data.typology = graph.typology
    return data

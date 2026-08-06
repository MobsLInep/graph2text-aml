"""Dataset statistics.

These numbers go into the data cards and get compared against the published figures. That
comparison is the point: a loader that silently disagrees with the paper it cites is the
classic way a data phase goes wrong, so every statistic here is computed from the data and
none is copied from a table.

Everything is computed with Polars aggregates rather than by materialising Python lists.
The HI-Small graph has five million edges, and a ``.to_list()`` in the wrong place turns a
three-second job into a memory problem.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from g2t_aml.data.canonical import CanonicalGraph

#: Quantiles reported for degree and component-size distributions.
QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999)


def _numeric_summary(series: pl.Series) -> dict[str, Any]:
    """Summarise a numeric series.

    Args:
        series: Values to describe. May be empty.

    Returns:
        Count, min, max, mean, std and the :data:`QUANTILES`. All values are plain Python
        scalars, so the result is JSON-serialisable. Empty input yields nulls rather than
        raising, because an empty subgraph is a legitimate edge case.
    """
    if series.is_empty():
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    summary: dict[str, Any] = {
        "count": int(series.len()),
        "min": _scalar(series.min()),
        "max": _scalar(series.max()),
        "mean": _scalar(series.mean()),
        "std": _scalar(series.std()),
    }
    for q in QUANTILES:
        summary[f"p{q * 100:g}"] = _scalar(series.quantile(q, interpolation="nearest"))
    return summary


def _scalar(value: Any) -> Any:
    """Convert a Polars scalar to a plain Python value.

    Args:
        value: Scalar from an aggregate, possibly None or a numpy type.

    Returns:
        An int, float, str or None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    """Count values in a column, largest first.

    Args:
        frame: Source frame.
        column: Column to count. Nulls are excluded.

    Returns:
        Value to count, ordered by descending count then by value for stability. Empty if
        the column is absent, so callers need not pre-check optional columns.
    """
    if column not in frame.columns:
        return {}
    counted = (
        frame.select(pl.col(column))
        .drop_nulls()
        .group_by(column)
        .len()
        .sort(["len", column], descending=[True, False])
    )
    return {str(row[column]): int(row["len"]) for row in counted.to_dicts()}


def degree_statistics(graph: CanonicalGraph) -> dict[str, Any]:
    """Summarise the degree distribution.

    Degrees are recomputed from the edge table rather than read from node columns, so this
    stays correct for substrates whose node tables carry no precomputed degrees.

    Args:
        graph: The graph to describe.

    Returns:
        ``in``, ``out`` and ``total`` degree summaries, plus ``num_isolated`` — nodes with
        no incident edge.
    """
    out_deg = graph.edges.group_by("src").len().rename({"src": "node_id", "len": "out_degree"})
    in_deg = graph.edges.group_by("dst").len().rename({"dst": "node_id", "len": "in_degree"})
    degrees = (
        graph.nodes.select("node_id")
        .join(out_deg, on="node_id", how="left")
        .join(in_deg, on="node_id", how="left")
        .with_columns(
            pl.col("out_degree").fill_null(0),
            pl.col("in_degree").fill_null(0),
        )
        .with_columns((pl.col("in_degree") + pl.col("out_degree")).alias("degree"))
    )
    return {
        "in": _numeric_summary(degrees["in_degree"]),
        "out": _numeric_summary(degrees["out_degree"]),
        "total": _numeric_summary(degrees["degree"]),
        "num_isolated": int(degrees.filter(pl.col("degree") == 0).height),
    }


def structural_statistics(graph: CanonicalGraph) -> dict[str, Any]:
    """Count self-loops and multi-edges.

    Both matter downstream. Self-loops in AMLworld are largely ``Reinvestment`` rows and
    are numerous — over half a million in HI-Small — so a GAT that treats them as ordinary
    neighbours will be dominated by them. Multi-edges mean the graph is a multigraph, so
    any adjacency-matrix formulation would silently collapse repeated transfers between
    the same pair of accounts into one.

    Args:
        graph: The graph to describe.

    Returns:
        ``num_self_loops``, ``num_distinct_pairs``, ``num_multi_edge_pairs`` and
        ``max_parallel_edges``.
    """
    self_loops = int(graph.edges.filter(pl.col("src") == pl.col("dst")).height)
    pairs = graph.edges.group_by(["src", "dst"]).len()
    return {
        "num_self_loops": self_loops,
        "num_distinct_pairs": int(pairs.height),
        "num_multi_edge_pairs": int(pairs.filter(pl.col("len") > 1).height),
        "max_parallel_edges": int(pairs["len"].max() or 0),
    }


def temporal_statistics(graph: CanonicalGraph) -> dict[str, Any]:
    """Describe the temporal span and its granularity.

    Args:
        graph: The graph to describe.

    Returns:
        ``available`` False with a reason when the substrate has no usable timestamps —
        Elliptic2 — otherwise first/last timestamp, span in days, and the inferred
        granularity. Granularity is *observed*, not assumed: it is derived from whether
        the second and minute components are always zero.
    """
    if not graph.availability.absolute_timestamps or "timestamp" not in graph.edges.columns:
        return {
            "available": False,
            "reason": (
                "substrate has no absolute timestamps "
                "(availability.absolute_timestamps is False)"
            ),
        }
    ts = graph.edges["timestamp"]
    if ts.is_empty():
        return {"available": True, "first": None, "last": None, "span_days": None}

    first, last = ts.min(), ts.max()
    distinct_seconds = graph.edges.select(pl.col("timestamp").dt.second().n_unique()).item()
    distinct_minutes = graph.edges.select(pl.col("timestamp").dt.minute().n_unique()).item()
    if distinct_seconds > 1:
        granularity = "second"
    elif distinct_minutes > 1:
        granularity = "minute"
    else:
        granularity = "hour or coarser"

    return {
        "available": True,
        "first": str(first),
        "last": str(last),
        "span_days": round((last - first).total_seconds() / 86_400, 4),  # type: ignore[operator]
        "granularity": granularity,
        "fine_resolution_declared": graph.availability.fine_temporal_resolution,
        "num_distinct_timestamps": int(ts.n_unique()),
    }


def component_statistics(graph: CanonicalGraph, *, max_nodes: int = 2_000_000) -> dict[str, Any]:
    """Summarise weakly-connected component sizes.

    Computed with an iterative union-find over the edge list. networkx is avoided here on
    purpose: building a five-million-edge ``MultiDiGraph`` costs several gigabytes and
    minutes, where union-find over two integer arrays is seconds.

    Args:
        graph: The graph to describe.
        max_nodes: Refuse to run above this many nodes, returning a skip record instead.
            Guards against an accidental invocation on the Elliptic2 background graph.

    Returns:
        ``num_components``, ``largest_component_size``, the fraction of nodes in the
        largest component, and a size summary. When skipped, ``{"computed": False, ...}``.
    """
    if graph.num_nodes > max_nodes:
        return {
            "computed": False,
            "reason": f"graph has {graph.num_nodes} nodes, above max_nodes={max_nodes}",
        }
    if graph.num_nodes == 0:
        return {"computed": True, "num_components": 0, "largest_component_size": 0}

    index = {node: i for i, node in enumerate(graph.nodes["node_id"].to_list())}
    parent = list(range(len(index)))

    def find(x: int) -> int:
        """Find the representative of ``x``, with full path compression.

        Args:
            x: Element index.

        Returns:
            Root index.
        """
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for src, dst in zip(graph.edges["src"].to_list(), graph.edges["dst"].to_list(), strict=True):
        a, b = find(index[src]), find(index[dst])
        if a != b:
            parent[a] = b

    sizes = pl.Series("size", [find(i) for i in range(len(parent))]).value_counts()["count"]
    return {
        "computed": True,
        "num_components": int(sizes.len()),
        "largest_component_size": int(sizes.max() or 0),
        "largest_component_fraction": round(float(sizes.max() or 0) / graph.num_nodes, 6),
        "size_summary": _numeric_summary(sizes),
    }


def compute_dataset_statistics(
    graph: CanonicalGraph,
    *,
    include_components: bool = True,
    max_component_nodes: int = 2_000_000,
) -> dict[str, Any]:
    """Compute the full statistics record for a canonical graph.

    Args:
        graph: The graph to describe.
        include_components: Whether to run the connected-component pass, the most
            expensive part.
        max_component_nodes: Passed to :func:`component_statistics`.

    Returns:
        A JSON-serialisable record covering identity, node/edge counts, degree
        distribution, class balance, temporal span and granularity, typology
        distribution, currency and payment-format distributions, component sizes, and
        self-loop/multi-edge counts. Fields that do not apply to the substrate are present
        and empty rather than absent, so the two data cards stay comparable.

    Raises:
        pl.exceptions.ColumnNotFoundError: If the graph is missing a required column.
    """
    edges = graph.edges
    record: dict[str, Any] = {
        "graph_id": graph.graph_id,
        "dataset": graph.dataset,
        "label": graph.label,
        "typology": graph.typology,
        "availability": graph.availability.to_dict(),
        "counts": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "density": (
                round(graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1)), 12)
                if graph.num_nodes > 1
                else None
            ),
        },
        "degree": degree_statistics(graph),
        "structural": structural_statistics(graph),
        "temporal": temporal_statistics(graph),
    }

    # Class balance. Present only where the substrate carries an edge-level label.
    if "is_laundering" in edges.columns:
        positives = int(edges.filter(pl.col("is_laundering")).height)
        record["class_balance"] = {
            "num_laundering_edges": positives,
            "num_clean_edges": graph.num_edges - positives,
            "laundering_rate": (round(positives / graph.num_edges, 9) if graph.num_edges else None),
            "one_in": round(graph.num_edges / positives) if positives else None,
        }
    else:
        record["class_balance"] = {}

    record["typology_distribution"] = _value_counts(edges, "typology")
    record["currency_distribution"] = {
        "receiving": _value_counts(edges, "receiving_currency"),
        "payment": _value_counts(edges, "payment_currency"),
    }
    record["payment_format_distribution"] = _value_counts(edges, "payment_format")
    record["node_type_distribution"] = _value_counts(graph.nodes, "node_type")

    if "amount_paid" in edges.columns and graph.availability.monetary_amounts:
        record["amount_summary"] = _numeric_summary(edges["amount_paid"])
    else:
        record["amount_summary"] = {}

    record["components"] = (
        component_statistics(graph, max_nodes=max_component_nodes)
        if include_components
        else {"computed": False, "reason": "disabled by caller"}
    )
    return record


def compare_to_published(
    observed: dict[str, Any], published: dict[str, int], *, path: tuple[str, ...] = ("counts",)
) -> dict[str, dict[str, Any]]:
    """Build an observed-versus-published table.

    Args:
        observed: Record from :func:`compute_dataset_statistics`.
        published: Published figures, keyed as in ``observed[path]``.
        path: Where in ``observed`` to look, as a sequence of keys.

    Returns:
        Per-key ``{"published", "observed", "matches", "delta"}``. ``observed`` is None and
        ``matches`` False when the key is absent, so a missing statistic reads as a
        failure rather than silently disappearing from the table.
    """
    node: Any = observed
    for key in path:
        node = node.get(key, {}) if isinstance(node, dict) else {}
    table: dict[str, dict[str, Any]] = {}
    for key, expected in published.items():
        actual = node.get(key) if isinstance(node, dict) else None
        table[key] = {
            "published": expected,
            "observed": actual,
            "matches": actual == expected,
            "delta": (actual - expected) if isinstance(actual, int) else None,
        }
    return table

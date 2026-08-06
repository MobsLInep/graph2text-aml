"""Dataset statistics.

Computed on small graphs whose correct answers can be worked out by hand, so a regression
in the aggregation logic shows up here rather than as a wrong number in a data card.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from g2t_aml.data.canonical import (
    AMLWORLD_AVAILABILITY,
    ELLIPTIC2_AVAILABILITY,
    CanonicalGraph,
)
from g2t_aml.data.stats import (
    compare_to_published,
    component_statistics,
    compute_dataset_statistics,
    degree_statistics,
    structural_statistics,
    temporal_statistics,
)


def graph_from(
    node_ids: list[str], edge_pairs: list[tuple[str, str]], **overrides
) -> CanonicalGraph:
    """Build a bare graph from an adjacency listing.

    Any constructor argument may be overridden by keyword, including ``nodes`` and
    ``edges``, so tests needing extra columns can supply their own frames.
    """
    kwargs = {
        "graph_id": "g",
        "dataset": "amlworld_hi_small",
        "nodes": pl.DataFrame({"node_id": node_ids, "node_type": ["account"] * len(node_ids)}),
        "edges": pl.DataFrame(
            {"src": [s for s, _ in edge_pairs], "dst": [d for _, d in edge_pairs]}
        ),
        "node_feature_names": [],
        "edge_feature_names": [],
        "availability": AMLWORLD_AVAILABILITY,
    }
    return CanonicalGraph(**(kwargs | overrides))


# ------------------------------------------------------------------- degree ---


def test_degree_counts_are_computed_from_the_edges():
    graph = graph_from(["a", "b", "c"], [("a", "b"), ("a", "c")])
    degree = degree_statistics(graph)
    assert degree["out"]["max"] == 2
    assert degree["in"]["max"] == 1
    assert degree["total"]["max"] == 2


def test_isolated_nodes_are_counted():
    graph = graph_from(["a", "b", "lonely"], [("a", "b")])
    assert degree_statistics(graph)["num_isolated"] == 1


def test_degree_summary_covers_quantiles():
    graph = graph_from(["a", "b"], [("a", "b")])
    assert "p99" in degree_statistics(graph)["total"]


# --------------------------------------------------------------- structural ---


def test_self_loops_are_counted():
    graph = graph_from(["a", "b"], [("a", "a"), ("a", "b")])
    assert structural_statistics(graph)["num_self_loops"] == 1


def test_multi_edges_are_counted():
    graph = graph_from(["a", "b"], [("a", "b"), ("a", "b"), ("a", "b")])
    stats = structural_statistics(graph)
    assert stats["num_distinct_pairs"] == 1
    assert stats["num_multi_edge_pairs"] == 1
    assert stats["max_parallel_edges"] == 3


def test_a_simple_graph_reports_no_multi_edges():
    graph = graph_from(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert structural_statistics(graph)["num_multi_edge_pairs"] == 0


# ----------------------------------------------------------------- temporal ---


def test_temporal_is_unavailable_when_the_mask_says_so():
    graph = graph_from(["a", "b"], [("a", "b")], availability=ELLIPTIC2_AVAILABILITY)
    result = temporal_statistics(graph)
    assert result["available"] is False
    assert "absolute_timestamps" in result["reason"]


def test_temporal_span_and_granularity_are_observed_not_assumed():
    edges = pl.DataFrame(
        {
            "src": ["a", "a"],
            "dst": ["b", "b"],
            "timestamp": ["2022-09-01 00:20:00", "2022-09-03 00:35:00"],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())
    graph = graph_from(["a", "b"], [("a", "b")], edges=edges)
    result = temporal_statistics(graph)
    assert result["available"] is True
    assert result["span_days"] == pytest.approx(2.0104, abs=1e-3)
    assert result["granularity"] == "minute"


# --------------------------------------------------------------- components ---


def test_components_are_found():
    graph = graph_from(["a", "b", "c", "d"], [("a", "b"), ("c", "d")])
    result = component_statistics(graph)
    assert result["num_components"] == 2
    assert result["largest_component_size"] == 2


def test_a_connected_graph_is_one_component():
    graph = graph_from(["a", "b", "c"], [("a", "b"), ("b", "c")])
    assert component_statistics(graph)["num_components"] == 1
    assert component_statistics(graph)["largest_component_fraction"] == 1.0


def test_isolated_node_is_its_own_component():
    graph = graph_from(["a", "b", "lonely"], [("a", "b")])
    assert component_statistics(graph)["num_components"] == 2


def test_components_refuse_to_run_on_an_oversized_graph():
    graph = graph_from(["a", "b"], [("a", "b")])
    result = component_statistics(graph, max_nodes=1)
    assert result["computed"] is False
    assert "max_nodes" in result["reason"]


def test_empty_graph_does_not_crash():
    graph = graph_from([], [])
    assert component_statistics(graph)["num_components"] == 0


# ------------------------------------------------------------------- record ---


def test_full_record_is_json_serialisable():
    graph = graph_from(["a", "b"], [("a", "b")])
    json.dumps(compute_dataset_statistics(graph))


def test_record_carries_the_availability_mask():
    graph = graph_from(["a", "b"], [("a", "b")], availability=ELLIPTIC2_AVAILABILITY)
    record = compute_dataset_statistics(graph)
    assert record["availability"]["monetary_amounts"] is False


def test_class_balance_is_computed_when_labels_exist():
    edges = pl.DataFrame(
        {"src": ["a"] * 4, "dst": ["b"] * 4, "is_laundering": [True, False, False, False]}
    )
    graph = graph_from(["a", "b"], [("a", "b")], edges=edges)
    balance = compute_dataset_statistics(graph)["class_balance"]
    assert balance["num_laundering_edges"] == 1
    assert balance["one_in"] == 4


def test_class_balance_is_empty_without_labels():
    graph = graph_from(["a", "b"], [("a", "b")])
    assert compute_dataset_statistics(graph)["class_balance"] == {}


def test_amounts_are_omitted_when_the_mask_forbids_them():
    """Invariant 4 reaches the statistics report too."""
    edges = pl.DataFrame({"src": ["a"], "dst": ["b"], "amount_paid": [10.0]})
    graph = graph_from(["a", "b"], [("a", "b")], edges=edges, availability=ELLIPTIC2_AVAILABILITY)
    assert compute_dataset_statistics(graph)["amount_summary"] == {}


def test_typology_distribution_is_reported():
    edges = pl.DataFrame(
        {"src": ["a", "a"], "dst": ["b", "b"], "typology": ["fan_out", "unclassified"]}
    )
    graph = graph_from(["a", "b"], [("a", "b")], edges=edges)
    distribution = compute_dataset_statistics(graph)["typology_distribution"]
    assert distribution == {"fan_out": 1, "unclassified": 1}


def test_components_can_be_disabled():
    graph = graph_from(["a", "b"], [("a", "b")])
    record = compute_dataset_statistics(graph, include_components=False)
    assert record["components"]["computed"] is False


# ----------------------------------------------------------- comparison ---


def test_comparison_flags_a_match_and_a_mismatch():
    observed = {"counts": {"num_nodes": 10, "num_edges": 99}}
    table = compare_to_published(observed, {"num_nodes": 10, "num_edges": 100})
    assert table["num_nodes"]["matches"] is True
    assert table["num_edges"]["matches"] is False
    assert table["num_edges"]["delta"] == -1


def test_a_missing_statistic_reads_as_a_failure_not_a_silence():
    table = compare_to_published({"counts": {}}, {"num_nodes": 10})
    assert table["num_nodes"]["observed"] is None
    assert table["num_nodes"]["matches"] is False

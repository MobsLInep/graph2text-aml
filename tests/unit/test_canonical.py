"""The canonical graph representation: validation and lossless round-tripping.

Every downstream phase reads this structure, so a graph that survives a save/load cycle
only approximately would corrupt everything built on it.
"""

from __future__ import annotations

import polars as pl
import pytest

from g2t_aml.data.canonical import (
    AMLWORLD_AVAILABILITY,
    CANONICAL_SCHEMA_VERSION,
    ELLIPTIC2_AVAILABILITY,
    TYPOLOGY_VOCABULARY,
    CanonicalGraph,
)


def make_graph(**overrides) -> CanonicalGraph:
    """Build a small synthetic graph. Invariant 8: synthetic identifiers only."""
    nodes = pl.DataFrame(
        {
            "node_id": ["B1|ACCT-000001", "B1|ACCT-000002", "B2|ACCT-000003"],
            "node_type": ["account"] * 3,
            "bank": ["B1", "B1", "B2"],
            "degree": [2, 2, 2],
            "total_sent": [100.0, 50.0, 0.0],
        }
    )
    edges = pl.DataFrame(
        {
            "src": ["B1|ACCT-000001", "B1|ACCT-000002", "B1|ACCT-000001"],
            "dst": ["B1|ACCT-000002", "B2|ACCT-000003", "B1|ACCT-000001"],
            "amount_paid": [100.0, 50.0, 7.5],
            "is_laundering": [True, False, False],
        }
    )
    kwargs = {
        "graph_id": "test_graph",
        "dataset": "amlworld_hi_small",
        "nodes": nodes,
        "edges": edges,
        "node_feature_names": ["degree", "total_sent"],
        "edge_feature_names": ["amount_paid"],
        "availability": AMLWORLD_AVAILABILITY,
        "typology": "fan_out",
        "provenance": {"source": "unit test"},
    }
    return CanonicalGraph(**(kwargs | overrides))


def test_counts():
    graph = make_graph()
    assert graph.num_nodes == 3
    assert graph.num_edges == 3


def test_rejects_missing_node_column():
    with pytest.raises(ValueError, match="node_id"):
        make_graph(nodes=pl.DataFrame({"node_type": ["account"]}))


def test_rejects_missing_edge_column():
    with pytest.raises(ValueError, match="'dst'"):
        make_graph(edges=pl.DataFrame({"src": ["B1|ACCT-000001"]}))


def test_rejects_declared_feature_that_is_not_a_column():
    with pytest.raises(ValueError, match="declared node features absent"):
        make_graph(node_feature_names=["degree", "vibes"])


def test_rejects_typology_outside_the_controlled_vocabulary():
    with pytest.raises(ValueError, match="outside the controlled vocabulary"):
        make_graph(typology="spiral")


@pytest.mark.parametrize("typology", TYPOLOGY_VOCABULARY)
def test_accepts_every_controlled_typology(typology):
    assert make_graph(typology=typology).typology == typology


def test_unclassified_is_a_valid_typology():
    """1,968 HI-Small laundering transactions match no structural pattern."""
    assert "unclassified" in TYPOLOGY_VOCABULARY


def test_none_typology_differs_from_unclassified():
    """None means the substrate has no typology truth; unclassified is a real verdict."""
    assert make_graph(typology=None).typology is None


def test_referential_integrity_passes_on_a_consistent_graph():
    make_graph().validate_referential_integrity()


def test_referential_integrity_catches_a_dangling_endpoint():
    edges = pl.DataFrame(
        {
            "src": ["B1|ACCT-000001"],
            "dst": ["B9|ACCT-999999"],
            "amount_paid": [1.0],
            "is_laundering": [False],
        }
    )
    with pytest.raises(ValueError, match="dst endpoints are not in the node table"):
        make_graph(edges=edges).validate_referential_integrity()


# ------------------------------------------------------------- round trip ---


def test_round_trips_losslessly(tmp_path):
    original = make_graph()
    original.save(tmp_path)
    restored = CanonicalGraph.load(tmp_path)

    assert restored.graph_id == original.graph_id
    assert restored.dataset == original.dataset
    assert restored.node_feature_names == original.node_feature_names
    assert restored.edge_feature_names == original.edge_feature_names
    assert restored.availability == original.availability
    assert restored.label == original.label
    assert restored.typology == original.typology
    assert restored.provenance == original.provenance
    assert restored.nodes.equals(original.nodes)
    assert restored.edges.equals(original.edges)


def test_round_trip_preserves_dtypes(tmp_path):
    original = make_graph()
    original.save(tmp_path)
    restored = CanonicalGraph.load(tmp_path)
    assert restored.nodes.schema == original.nodes.schema
    assert restored.edges.schema == original.edges.schema


def test_round_trip_preserves_temporal_dtype(tmp_path):
    """Datetimes are the column most likely to come back as a string."""
    edges = pl.DataFrame(
        {
            "src": ["B1|ACCT-000001"],
            "dst": ["B1|ACCT-000002"],
            "timestamp": ["2022-09-01 00:20:00"],
            "amount_paid": [1.0],
            "is_laundering": [False],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())
    graph = make_graph(edges=edges)
    graph.save(tmp_path)
    restored = CanonicalGraph.load(tmp_path)
    assert restored.edges.schema["timestamp"] == pl.Datetime("us")


def test_save_writes_the_expected_files(tmp_path):
    make_graph().save(tmp_path)
    for name in ("nodes.parquet", "edges.parquet", "canonical.json"):
        assert (tmp_path / name).exists()


def test_load_rejects_a_foreign_schema_version(tmp_path):
    import json

    make_graph().save(tmp_path)
    meta = json.loads((tmp_path / "canonical.json").read_text())
    meta["schema_version"] = "0.0.1-ancient"
    (tmp_path / "canonical.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema version mismatch"):
        CanonicalGraph.load(tmp_path)


def test_summary_is_json_serialisable_and_carries_the_mask():
    import json

    summary = make_graph(availability=ELLIPTIC2_AVAILABILITY).summary()
    json.dumps(summary)
    assert summary["availability"]["monetary_amounts"] is False
    assert summary["schema_version"] == CANONICAL_SCHEMA_VERSION

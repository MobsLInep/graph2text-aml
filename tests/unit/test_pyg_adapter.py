"""PyG conversion. Requires the optional `graph` extra, so it skips without it.

The import-boundary test that keeps the CPU phases torch-free lives in
``test_import_boundary.py``, deliberately unguarded — guarding it with importorskip would
disable it on exactly the machines it exists to protect.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="the graph extra is not installed")
pytest.importorskip("torch_geometric", reason="the graph extra is not installed")


def test_converts_a_graph_to_hetero_data():
    import polars as pl

    from g2t_aml.data.canonical import AMLWORLD_AVAILABILITY, CanonicalGraph
    from g2t_aml.data.pyg_adapter import to_pyg

    graph = CanonicalGraph(
        graph_id="g",
        dataset="amlworld_hi_small",
        nodes=pl.DataFrame(
            {
                "node_id": ["B1|A1", "B1|A2"],
                "node_type": ["account", "account"],
                "degree": [1, 1],
            }
        ),
        edges=pl.DataFrame(
            {"src": ["B1|A1"], "dst": ["B1|A2"], "amount_paid": [5.0], "is_laundering": [True]}
        ),
        node_feature_names=["degree"],
        edge_feature_names=["amount_paid"],
        availability=AMLWORLD_AVAILABILITY,
    )
    data = to_pyg(graph)
    assert data["account"].num_nodes == 2
    assert data["account", "transacts", "account"].edge_index.shape == (2, 1)
    # Invariant 4 travels with the tensors.
    assert data.availability.entity_types is False

"""Feature-layer tests: leakage, availability, determinism and the positional encodings.

The leakage test in here is the one the Phase 7 brief asks to run every time.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from g2t_aml.data.canonical import AMLWORLD_AVAILABILITY, CanonicalGraph
from g2t_aml.data.leakage_audit import LABEL_PROXY_COLUMNS
from g2t_aml.models.encoder.features import (
    EDGE_FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
    NODE_FEATURE_NAMES,
    PERMITTED_EDGE_COLUMNS,
    PERMITTED_NODE_COLUMNS,
    FeatureError,
    FeatureSpace,
    assert_no_label_columns,
    fit_feature_space,
)
from g2t_aml.models.encoder.positional import (
    laplacian_pe,
    random_walk_pe,
    undirected_adjacency,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from g2t_aml.models.encoder.features import build_case_data  # noqa: E402

DATETIME = pl.Datetime("us")


def _edges(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "src": pl.Utf8,
            "dst": pl.Utf8,
            "timestamp": DATETIME,
            "amount_paid": pl.Float64,
            "payment_currency": pl.Utf8,
            "amount_received": pl.Float64,
            "receiving_currency": pl.Utf8,
            "payment_format": pl.Utf8,
            "is_laundering": pl.Boolean,
            "typology": pl.Utf8,
            "pattern_id": pl.Utf8,
        },
    )


def _case(n_nodes: int = 4) -> CanonicalGraph:
    from datetime import datetime

    node_ids = [f"BANK|ACCT{i:03d}" for i in range(n_nodes)]
    rows = []
    for i in range(n_nodes - 1):
        rows.append(
            {
                "src": node_ids[i],
                "dst": node_ids[i + 1],
                "timestamp": datetime(2022, 9, 7, 10 + i, 0),
                "amount_paid": 1000.0 * (i + 1),
                "payment_currency": "US Dollar",
                "amount_received": 1000.0 * (i + 1),
                "receiving_currency": "US Dollar",
                "payment_format": "Wire",
                # Present in the source frame and never read: that is the point. The
                # flag lands on a *middle* edge with a mid-range amount and a typical
                # inter-transaction gap, so that no legitimate feature is accidentally
                # correlated with it -- otherwise the separation test below would fire on
                # the fixture rather than on a leak.
                "is_laundering": i == (n_nodes - 1) // 2,
                "typology": "stack" if i == (n_nodes - 1) // 2 else None,
                "pattern_id": "P1" if i == (n_nodes - 1) // 2 else None,
            }
        )
    return CanonicalGraph(
        graph_id="case-1",
        dataset="amlworld_hi_small",
        nodes=pl.DataFrame(
            {
                "node_id": node_ids,
                "node_type": ["account"] * n_nodes,
                "bank": ["BANK"] * n_nodes,
                # Global aggregates from the interim table. Also never read.
                "in_degree": list(range(n_nodes)),
                "out_degree": list(range(n_nodes)),
                "degree": list(range(n_nodes)),
                "total_received": [9e9] * n_nodes,
                "total_sent": [9e9] * n_nodes,
            }
        ),
        edges=_edges(rows),
        node_feature_names=[],
        edge_feature_names=[],
        availability=AMLWORLD_AVAILABILITY,
        label="suspicious",
        typology="stack",
    )


@pytest.fixture
def space() -> FeatureSpace:
    return fit_feature_space(
        _case(6).edges,
        dataset="amlworld_hi_small",
        availability=AMLWORLD_AVAILABILITY.to_dict(),
        n_train_cases=1,
    )


# ------------------------------------------------------------------ leakage ---


def test_permitted_columns_contain_no_label_or_proxy():
    """The brief's standing check: run it every time."""
    assert_no_label_columns(PERMITTED_EDGE_COLUMNS | PERMITTED_NODE_COLUMNS)
    assert not (PERMITTED_EDGE_COLUMNS & LABEL_PROXY_COLUMNS)
    assert not (PERMITTED_NODE_COLUMNS & LABEL_PROXY_COLUMNS)
    assert "is_laundering" not in PERMITTED_EDGE_COLUMNS
    assert "typology" not in PERMITTED_EDGE_COLUMNS
    assert "pattern_id" not in PERMITTED_EDGE_COLUMNS


def test_assert_no_label_columns_fires_on_every_proxy():
    """The guard is only credible if it is known to fail."""
    for proxy in sorted(LABEL_PROXY_COLUMNS):
        with pytest.raises(FeatureError, match=proxy):
            assert_no_label_columns(frozenset({"src", proxy}))


def test_no_feature_name_is_a_label_proxy():
    """A feature named after a label would survive the column check but not this one."""
    names = set(NODE_FEATURE_NAMES) | set(EDGE_FEATURE_NAMES)
    assert not (names & LABEL_PROXY_COLUMNS)


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"is_laundering": True}, id="all-flagged"),
        pytest.param({"is_laundering": False}, id="none-flagged"),
        pytest.param({"typology": "fan_out"}, id="typology-rewritten"),
        pytest.param({"pattern_id": "P999"}, id="pattern-rewritten"),
    ],
)
def test_label_columns_cannot_influence_any_tensor(space, mutation):
    """Rewrite the label columns in the source frame; every tensor must be unchanged.

    The decisive form of the leakage check, and the one the Phase 7 brief asks to run
    every time. A separation heuristic over a small fixture is fragile — any non-monotone
    feature will single out a middle row by chance. Proving the tensors are *invariant*
    to the label columns proves the label cannot have entered them, whatever the fixture
    happens to look like.
    """
    baseline = build_case_data(_case(6), space, seed_node="BANK|ACCT000", label=1)

    graph = _case(6)
    column, value = next(iter(mutation.items()))
    dtype = graph.edges.schema[column]
    graph.edges = graph.edges.with_columns(pl.lit(value, dtype=dtype).alias(column))
    mutated = build_case_data(graph, space, seed_node="BANK|ACCT000", label=1)

    assert torch.equal(baseline.x, mutated.x)
    assert torch.equal(baseline.edge_attr, mutated.edge_attr)
    assert torch.equal(baseline.edge_index, mutated.edge_index)
    assert torch.equal(baseline.edge_currency_paid, mutated.edge_currency_paid)
    assert torch.equal(baseline.edge_format, mutated.edge_format)


def test_targets_live_on_the_record_not_in_the_features(space):
    data = build_case_data(_case(5), space, label=1, typology_index=7)
    assert int(data.y.item()) == 1
    assert int(data.y_typ.item()) == 7
    assert data.x.shape[1] == space.node_dim
    assert data.edge_attr.shape[1] == len(EDGE_FEATURE_NAMES)


def test_global_node_aggregates_are_not_read(space):
    """The interim node table's global degree columns must not influence the tensor.

    CLAUDE.md note 8: those columns aggregate over the whole 515,088-account graph, so
    reading them would let a test-window account's encoding depend on its training-window
    activity. Changing them must change nothing.
    """
    graph = _case(4)
    baseline = build_case_data(graph, space, label=1).x.clone()

    tampered = _case(4)
    tampered.nodes = tampered.nodes.with_columns(
        pl.lit(999).alias("in_degree"),
        pl.lit(999).alias("out_degree"),
        pl.lit(999).alias("degree"),
        pl.lit(1.23e12).alias("total_received"),
        pl.lit(4.56e12).alias("total_sent"),
    )
    assert torch.equal(baseline, build_case_data(tampered, space, label=1).x)


# ------------------------------------------------------------------- shapes ---


def test_feature_widths_match_the_declared_space(space):
    data = build_case_data(_case(7), space, label=0)
    assert data.x.shape == (7, space.node_dim)
    assert data.edge_attr.shape == (6, len(EDGE_FEATURE_NAMES))
    assert data.edge_currency_paid.shape == (6,)
    assert data.edge_format.shape == (6,)
    assert space.node_dim == len(NODE_FEATURE_NAMES) + space.lap_pe_dim + space.rw_pe_dim


def test_pe_slice_locates_the_positional_block(space):
    start, stop = space.pe_slice
    assert start == len(NODE_FEATURE_NAMES)
    assert stop - start == space.lap_pe_dim + space.rw_pe_dim
    lap_start, lap_stop = space.lap_pe_slice
    assert (lap_start, lap_stop) == (start, start + space.lap_pe_dim)


def test_single_node_case_encodes_without_positional_components(space):
    """18.1% of the corpus is a two-account case; a one-node case must not crash."""
    graph = _case(2)
    graph.nodes = graph.nodes.head(1)
    graph.edges = graph.edges.head(0)
    data = build_case_data(graph, space, label=0)
    assert data.x.shape == (1, space.node_dim)
    assert torch.isfinite(data.x).all()


# ------------------------------------------------------------- determinism ---


def test_encoding_is_deterministic(space):
    a = build_case_data(_case(6), space, seed_node="BANK|ACCT000", label=1)
    b = build_case_data(_case(6), space, seed_node="BANK|ACCT000", label=1)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_attr, b.edge_attr)
    assert torch.equal(a.edge_index, b.edge_index)


def test_everything_is_finite(space):
    data = build_case_data(_case(9), space, label=1)
    assert torch.isfinite(data.x).all()
    assert torch.isfinite(data.edge_attr).all()


# ----------------------------------------------------------- availability ---


def test_unavailable_families_are_zero_with_a_mask_channel():
    """Invariant 4 in the feature space: absence is zero *plus* a flag saying so."""
    masked = dict(AMLWORLD_AVAILABILITY.to_dict())
    masked["monetary_amounts"] = False
    masked["absolute_timestamps"] = False
    space = fit_feature_space(
        _case(4).edges,
        dataset="elliptic2",
        availability=masked,
        n_train_cases=1,
    )
    data = build_case_data(_case(4), space, label=1)

    for name in ("amt_in_z_sum", "amt_out_z_max", "amt_balance_z"):
        assert float(data.x[:, NODE_FEATURE_NAMES.index(name)].abs().max()) == 0.0
    assert float(data.x[:, NODE_FEATURE_NAMES.index("has_amounts")].max()) == 0.0
    assert float(data.x[:, NODE_FEATURE_NAMES.index("has_timestamps")].max()) == 0.0
    assert float(data.edge_attr[:, EDGE_FEATURE_NAMES.index("amt_paid_z")].abs().max()) == 0.0
    assert float(data.edge_attr[:, EDGE_FEATURE_NAMES.index("has_amounts")].max()) == 0.0


def test_amounts_are_standardised_within_a_currency():
    """D-033: cross-currency sums are meaningless, so amounts are z-scored per currency."""
    frame = _edges(
        [
            {
                "src": "A",
                "dst": "B",
                "timestamp": None,
                "amount_paid": amount,
                "payment_currency": currency,
                "amount_received": amount,
                "receiving_currency": currency,
                "payment_format": "Wire",
                "is_laundering": False,
                "typology": None,
                "pattern_id": None,
            }
            # Yen amounts are ~100x dollar amounts; standardising within each makes the
            # two comparable, which is exactly what a raw sum cannot do.
            for currency, amounts in (("US Dollar", [100, 1000, 10000]), ("Yen", [1e4, 1e5, 1e6]))
            for amount in amounts
        ]
    )
    space = fit_feature_space(
        frame,
        dataset="amlworld_hi_small",
        availability=AMLWORLD_AVAILABILITY.to_dict(),
        n_train_cases=1,
    )
    # The median amount in each currency standardises to roughly zero in both.
    assert abs(space.standardise(1000.0, "US Dollar")) < 0.2
    assert abs(space.standardise(1e5, "Yen")) < 0.2
    # And the two currencies' statistics are genuinely different.
    assert space.amount_stats["Yen"][0] > space.amount_stats["US Dollar"][0] + 2


def test_unseen_currency_maps_to_the_oov_slot(space):
    assert space.currency_index("Klingon Darsek") == 0
    assert space.format_index("Carrier Pigeon") == 0
    assert space.currency_index("US Dollar") > 0


def test_feature_space_round_trips(space):
    assert FeatureSpace.from_dict(space.to_dict()).to_dict() == space.to_dict()


def test_feature_space_rejects_a_stale_version(space):
    payload = space.to_dict() | {"version": "0.0.1"}
    with pytest.raises(FeatureError, match="version mismatch"):
        FeatureSpace.from_dict(payload)
    assert space.version == FEATURE_SPEC_VERSION


# ------------------------------------------------------ positional encodings ---


def test_random_walk_pe_detects_a_triangle():
    """The encoding exists to see cycles, so assert it sees the smallest one."""
    src = np.asarray([0, 1, 2])
    dst = np.asarray([1, 2, 0])
    adjacency = undirected_adjacency(3, src, dst)
    walk = random_walk_pe(adjacency, 4)
    # Return probability is 0 at one step and 0.5 at two on a triangle.
    assert np.allclose(walk[:, 0], 0.0)
    assert np.allclose(walk[:, 1], 0.5)


def test_random_walk_pe_separates_a_triangle_from_a_path():
    triangle = undirected_adjacency(3, np.asarray([0, 1, 2]), np.asarray([1, 2, 0]))
    path = undirected_adjacency(3, np.asarray([0, 1]), np.asarray([1, 2]))
    assert not np.allclose(random_walk_pe(triangle, 4), random_walk_pe(path, 4))


def test_laplacian_pe_is_zero_padded_on_small_graphs():
    adjacency = undirected_adjacency(2, np.asarray([0]), np.asarray([1]))
    encoding = laplacian_pe(adjacency, 8)
    assert encoding.shape == (2, 8)
    # Two nodes give one non-trivial eigenvector; the remaining seven are padding.
    assert np.allclose(encoding[:, 1:], 0.0)


def test_self_loops_are_excluded_from_the_adjacency():
    """The fact layer's convention (caseview): adjacency ignores self-loops."""
    adjacency = undirected_adjacency(2, np.asarray([0, 0]), np.asarray([0, 1]))
    assert adjacency[0, 0] == 0.0
    assert adjacency[0, 1] == adjacency[1, 0] == 1.0


def test_isolated_node_has_zero_encodings():
    adjacency = undirected_adjacency(3, np.asarray([0]), np.asarray([1]))
    assert np.allclose(random_walk_pe(adjacency, 5)[2], 0.0)

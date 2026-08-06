"""Case extraction: determinism, provenance, and the pruning guarantee.

Determinism is a Phase 2 gate criterion, and pruning that severs a laundering path would
silently remove the evidence a case exists to describe. Both are tested against fixtures
constructed so the naive implementation fails.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import polars as pl
import pytest

from g2t_aml.data.canonical import AMLWORLD_AVAILABILITY, CanonicalGraph
from g2t_aml.data.case_extraction import (
    DEFAULT_MAX_NEIGHBOURS,
    EXTRACTION_PROTOCOL_VERSION,
    CaseExtractionError,
    ExtractionParams,
    GraphIndex,
    TimeWindow,
    case_id_for,
    cut_case,
    extract_case,
    passthrough_case,
)

T0 = datetime(2022, 9, 1, 0, 0)


def _parquet_bytes(frame: pl.DataFrame) -> bytes:
    """Serialise a frame the way CanonicalGraph.save does, for byte-identity checks."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer, compression="zstd")
    return buffer.getvalue()


def _graph(
    nodes: list[str], edges: list[tuple[str, str, float, bool, str | None]]
) -> CanonicalGraph:
    """Build a small synthetic graph. Invariant 8: synthetic identifiers only."""
    node_table = pl.DataFrame(
        {
            "node_id": nodes,
            "node_type": ["account"] * len(nodes),
            "degree": [1] * len(nodes),
            "first_seen": [T0] * len(nodes),
            "last_seen": [T0 + timedelta(days=30)] * len(nodes),
        }
    )
    edge_table = pl.DataFrame(
        {
            "src": [e[0] for e in edges],
            "dst": [e[1] for e in edges],
            "timestamp": [T0 + timedelta(hours=i) for i in range(len(edges))],
            "amount_paid": [e[2] for e in edges],
            "is_laundering": [e[3] for e in edges],
            "pattern_id": [e[4] for e in edges],
            "typology": ["fan_out" if e[4] else None for e in edges],
        }
    )
    return CanonicalGraph(
        graph_id="fixture",
        dataset="fixture",
        nodes=node_table,
        edges=edge_table,
        node_feature_names=["degree"],
        edge_feature_names=["amount_paid"],
        availability=AMLWORLD_AVAILABILITY,
    )


def _chain(n: int) -> CanonicalGraph:
    """A -> B -> C -> ... with ascending amounts, no laundering."""
    nodes = [f"B1|ACCT-{i:04d}" for i in range(n)]
    edges = [(nodes[i], nodes[i + 1], 100.0 * (i + 1), False, None) for i in range(n - 1)]
    return _graph(nodes, edges)


WINDOW = TimeWindow(T0 - timedelta(days=1), T0 + timedelta(days=30))


# ------------------------------------------------------------- time window ---


def test_window_rejects_inverted_interval():
    with pytest.raises(ValueError, match="ends before it starts"):
        TimeWindow(T0 + timedelta(hours=1), T0)


def test_window_bounds_are_inclusive():
    window = TimeWindow(T0, T0 + timedelta(hours=1))
    assert window.contains(T0)
    assert window.contains(T0 + timedelta(hours=1))
    assert not window.contains(T0 + timedelta(hours=1, minutes=1))


def test_window_round_trips_through_a_dict():
    window = TimeWindow(T0, T0 + timedelta(hours=5))
    assert TimeWindow.from_dict(window.to_dict()) == window


def test_window_straddles_only_strictly_interior_boundaries():
    window = TimeWindow(T0, T0 + timedelta(hours=4))
    assert window.straddles(T0 + timedelta(hours=2))
    assert not window.straddles(T0)
    assert not window.straddles(T0 + timedelta(hours=4))


def test_padding_rejects_a_negative_width():
    with pytest.raises(ValueError, match="must not be negative"):
        TimeWindow(T0, T0).padded(-1)


# ---------------------------------------------------------------- params ----


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k_hops": -1}, "k_hops"),
        ({"n_max": 0}, "n_max"),
        ({"prune_rule": "alphabetical"}, "prune_rule"),
        ({"max_neighbours_per_node": 0}, "max_neighbours"),
    ],
)
def test_extraction_params_reject_nonsense(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ExtractionParams(**kwargs)


def test_extraction_params_reject_unknown_keys_on_load():
    with pytest.raises(ValueError, match="unknown extraction parameters"):
        ExtractionParams.from_dict({"k_hops": 2, "hops": 3})


# ------------------------------------------------------------ determinism ---


def test_extraction_is_byte_identical_across_repeats():
    """The Phase 2 gate: same inputs, byte-identical serialisation."""
    graph = _chain(12)
    index = GraphIndex(graph)
    first = extract_case(graph, "B1|ACCT-0000", WINDOW, index=index)
    second = extract_case(graph, "B1|ACCT-0000", WINDOW, index=index)
    assert first.graph_id == second.graph_id
    assert first.nodes.pipe(_parquet_bytes) == second.nodes.pipe(_parquet_bytes)
    assert first.edges.pipe(_parquet_bytes) == second.edges.pipe(_parquet_bytes)


def test_extraction_is_identical_across_separately_built_indices():
    graph = _chain(12)
    first = extract_case(graph, "B1|ACCT-0000", WINDOW, index=GraphIndex(graph))
    second = extract_case(graph, "B1|ACCT-0000", WINDOW, index=GraphIndex(graph))
    assert first.nodes.pipe(_parquet_bytes) == second.nodes.pipe(_parquet_bytes)
    assert first.edges.pipe(_parquet_bytes) == second.edges.pipe(_parquet_bytes)


def test_extraction_survives_a_reordered_edge_table():
    """Ordering decisions are made on content, so input row order must not matter."""
    graph = _chain(10)
    shuffled = CanonicalGraph(
        graph_id=graph.graph_id,
        dataset=graph.dataset,
        nodes=graph.nodes.reverse(),
        edges=graph.edges.reverse(),
        node_feature_names=graph.node_feature_names,
        edge_feature_names=graph.edge_feature_names,
        availability=graph.availability,
    )
    original = extract_case(graph, "B1|ACCT-0000", WINDOW)
    reordered = extract_case(shuffled, "B1|ACCT-0000", WINDOW)
    assert original.nodes.pipe(_parquet_bytes) == reordered.nodes.pipe(_parquet_bytes)
    assert original.edges.pipe(_parquet_bytes) == reordered.edges.pipe(_parquet_bytes)


def test_case_id_depends_on_every_parameter():
    window = TimeWindow(T0, T0 + timedelta(hours=1))
    base = case_id_for("fixture", "B1|ACCT-0000", window, ExtractionParams())
    assert base == case_id_for("fixture", "B1|ACCT-0000", window, ExtractionParams())
    assert base != case_id_for("fixture", "B1|ACCT-0001", window, ExtractionParams())
    assert base != case_id_for("other", "B1|ACCT-0000", window, ExtractionParams())
    assert base != case_id_for("fixture", "B1|ACCT-0000", window.padded(1), ExtractionParams())
    assert base != case_id_for("fixture", "B1|ACCT-0000", window, ExtractionParams(k_hops=3))
    assert base != case_id_for("fixture", "B1|ACCT-0000", window, ExtractionParams(seed=7))


# ---------------------------------------------------------------- protocol ---


def test_k_hops_bounds_the_reachable_set():
    graph = _chain(20)
    index = GraphIndex(graph)
    sizes = [
        extract_case(graph, "B1|ACCT-0000", WINDOW, k_hops=k, index=index).num_nodes
        for k in (1, 2, 3)
    ]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_window_excludes_transactions_outside_it():
    graph = _chain(10)
    narrow = TimeWindow(T0, T0 + timedelta(hours=2))
    case = extract_case(graph, "B1|ACCT-0000", narrow, k_hops=3)
    assert case.edges["timestamp"].max() <= narrow.end


def test_a_seed_with_no_activity_in_the_window_is_an_error():
    graph = _chain(5)
    empty = TimeWindow(T0 + timedelta(days=10), T0 + timedelta(days=11))
    with pytest.raises(CaseExtractionError, match="no transaction"):
        extract_case(graph, "B1|ACCT-0000", empty)


def test_an_unknown_seed_is_an_error():
    with pytest.raises(CaseExtractionError, match="not in"):
        extract_case(_chain(3), "B1|ACCT-9999", WINDOW)


def test_the_neighbour_cap_bounds_a_hub_and_is_reported():
    hub = "B1|ACCT-HUB"
    leaves = [f"B1|ACCT-{i:04d}" for i in range(200)]
    graph = _graph([hub, *leaves], [(hub, leaf, 1.0, False, None) for leaf in leaves])
    case = extract_case(graph, hub, WINDOW, k_hops=1, n_max=1000, max_neighbours_per_node=10)
    assert case.provenance["neighbour_cap_triggered"] is True
    assert case.num_nodes <= 11


def test_the_default_neighbour_cap_clears_amlworlds_largest_fan():
    """AMLworld generates fans up to 16-degree; the cap must never truncate one."""
    assert DEFAULT_MAX_NEIGHBOURS > 16


# ---------------------------------------------------------------- pruning ---


def test_pruning_preserves_a_laundering_path_the_amount_rule_would_sever():
    """The fixture is built so amount-descending pruning drops the laundering path.

    Ten high-value licit transfers surround a three-hop laundering chain whose amounts are
    the smallest in the graph. A pruner that only ranks by amount spends the whole node
    budget on the licit traffic and returns a "suspicious" case with no laundering edge in
    it -- which would be a narrative asserting a scheme it cannot show.
    """
    launder = ["B1|ACCT-L0", "B1|ACCT-L1", "B1|ACCT-L2", "B1|ACCT-L3"]
    noise = [f"B1|ACCT-N{i:03d}" for i in range(30)]
    edges: list[tuple[str, str, float, bool, str | None]] = [
        (launder[i], launder[i + 1], 1.0, True, "stack_00001") for i in range(3)
    ]
    edges += [(launder[0], n, 1_000_000.0, False, None) for n in noise]
    graph = _graph([*launder, *noise], edges)

    case = extract_case(graph, launder[0], WINDOW, k_hops=3, n_max=8)
    assert case.provenance["pruning_triggered"] is True
    assert case.provenance["preserved_laundering_edges"] == 3
    kept = set(case.edges.filter(pl.col("is_laundering"))["dst"].to_list())
    assert {"B1|ACCT-L1", "B1|ACCT-L2", "B1|ACCT-L3"} <= kept


def test_disabling_preservation_does_sever_that_path():
    """The mirror image, so the previous test is known to be testing something."""
    launder = ["B1|ACCT-L0", "B1|ACCT-L1", "B1|ACCT-L2", "B1|ACCT-L3"]
    noise = [f"B1|ACCT-N{i:03d}" for i in range(30)]
    edges: list[tuple[str, str, float, bool, str | None]] = [
        (launder[i], launder[i + 1], 1.0, True, "stack_00001") for i in range(3)
    ]
    edges += [(launder[0], n, 1_000_000.0, False, None) for n in noise]
    graph = _graph([*launder, *noise], edges)

    case = extract_case(
        graph, launder[0], WINDOW, k_hops=3, n_max=8, preserve_laundering_paths=False
    )
    assert case.provenance["preserved_laundering_edges"] < 3


def test_preservation_may_overrun_the_budget_and_says_so():
    launder = [f"B1|ACCT-L{i:03d}" for i in range(40)]
    edges = [(launder[i], launder[i + 1], 1.0, True, "stack_00001") for i in range(39)]
    graph = _graph(launder, edges)
    case = extract_case(graph, launder[0], WINDOW, k_hops=40, n_max=5)
    assert case.provenance["n_max_exceeded"] is True
    assert case.num_nodes > 5


def test_pruning_is_not_triggered_below_the_budget():
    case = extract_case(_chain(6), "B1|ACCT-0000", WINDOW, n_max=150)
    assert case.provenance["pruning_triggered"] is False
    assert case.provenance["n_max_exceeded"] is False


@pytest.mark.parametrize("rule", ["amount_desc", "recency", "degree"])
def test_every_prune_rule_respects_the_budget(rule):
    hub = "B1|ACCT-HUB"
    leaves = [f"B1|ACCT-{i:04d}" for i in range(80)]
    graph = _graph(
        [hub, *leaves], [(hub, leaf, float(i), False, None) for i, leaf in enumerate(leaves)]
    )
    case = extract_case(
        graph, hub, WINDOW, k_hops=1, n_max=20, prune_rule=rule, max_neighbours_per_node=80
    )
    assert case.num_nodes <= 20


def test_an_unknown_prune_rule_is_rejected():
    with pytest.raises(ValueError, match="prune_rule"):
        extract_case(_chain(4), "B1|ACCT-0000", WINDOW, prune_rule="by_vibes")


# ------------------------------------------------------------- provenance ---


def test_provenance_records_every_parameter_and_both_sizes():
    graph = _chain(8)
    case = extract_case(graph, "B1|ACCT-0000", WINDOW, k_hops=2, n_max=99, seed=7)
    provenance = case.provenance
    assert provenance["extraction_method"] == "constructed"
    assert provenance["extraction_protocol_version"] == EXTRACTION_PROTOCOL_VERSION
    assert provenance["seed_node"] == "B1|ACCT-0000"
    assert provenance["k_hops"] == 2
    assert provenance["n_max"] == 99
    assert provenance["seed"] == 7
    assert provenance["prune_rule"] == "amount_desc"
    assert provenance["window"] == WINDOW.to_dict()
    for key in (
        "pre_prune_nodes",
        "pre_prune_edges",
        "post_prune_nodes",
        "post_prune_edges",
        "pruning_triggered",
        "n_max_exceeded",
        "neighbour_cap_triggered",
        "preserved_laundering_edges",
    ):
        assert key in provenance, key


def test_the_case_label_follows_the_edges():
    clean = extract_case(_chain(5), "B1|ACCT-0000", WINDOW)
    assert clean.label == "licit"

    dirty = _graph(
        ["B1|ACCT-A", "B1|ACCT-B"], [("B1|ACCT-A", "B1|ACCT-B", 5.0, True, "fan_out_00001")]
    )
    assert extract_case(dirty, "B1|ACCT-A", WINDOW).label == "suspicious"


def test_the_case_typology_comes_from_the_controlled_vocabulary():
    dirty = _graph(
        ["B1|ACCT-A", "B1|ACCT-B"], [("B1|ACCT-A", "B1|ACCT-B", 5.0, True, "fan_out_00001")]
    )
    assert extract_case(dirty, "B1|ACCT-A", WINDOW).typology == "fan_out"


def test_the_case_inherits_the_substrates_availability_mask():
    """Invariant 4 travels with the case, not only with the substrate."""
    case = extract_case(_chain(5), "B1|ACCT-0000", WINDOW)
    assert case.availability == AMLWORLD_AVAILABILITY


# --------------------------------------------------------------- elliptic ---


def test_a_provided_subgraph_is_recorded_as_provided_and_not_constructed():
    graph = _chain(6)
    graph.label = "suspicious"
    case = passthrough_case(graph, case_id="elliptic2-000001")
    assert case.graph_id == "elliptic2-000001"
    assert case.provenance["extraction_method"] == "provided"
    assert case.provenance["pruning_triggered"] is False
    assert case.num_nodes == graph.num_nodes
    assert case.num_edges == graph.num_edges
    assert "k_hops" not in case.provenance


def test_a_provided_subgraph_is_canonically_ordered():
    graph = _chain(6)
    once = passthrough_case(graph)
    twice = passthrough_case(passthrough_case(graph))
    assert once.nodes.pipe(_parquet_bytes) == twice.nodes.pipe(_parquet_bytes)


# ------------------------------------------------------------------- cuts ---


def test_a_cut_and_a_materialised_case_agree():
    graph = _chain(10)
    index = GraphIndex(graph)
    cut = cut_case(graph, "B1|ACCT-0000", WINDOW, ExtractionParams(), index=index)
    case = extract_case(graph, "B1|ACCT-0000", WINDOW, index=index)
    assert cut.case_id == case.graph_id
    assert cut.node_positions.size == case.num_nodes
    assert cut.edge_positions.size == case.num_edges
    assert cut.label == case.label


def test_the_index_rejects_a_duplicate_node_id():
    graph = _chain(4)
    graph.nodes = pl.concat([graph.nodes, graph.nodes.head(1)])
    with pytest.raises(CaseExtractionError, match="duplicate node_id"):
        GraphIndex(graph)

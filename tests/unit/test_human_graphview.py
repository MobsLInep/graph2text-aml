"""The graph view: encoding, the display cap, and what it refuses to imply."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.factories import (
    acct,
    as_laundering_stream,
    at,
    elliptic2_case,
    fan_out_case,
    make_case,
    view_of,
)

from g2t_aml.human.graphview import (
    build_graph_view,
    to_plotly_figure,
)


def view(case):
    return view_of(case)


@pytest.fixture
def amlworld():
    case = as_laundering_stream(fan_out_case(width=6), "fan_out")
    return build_graph_view(view(case), acct(1))


@pytest.fixture
def elliptic2():
    return build_graph_view(view(elliptic2_case()), acct(1))


def wide_case(width: int):
    """A star with `width` spokes, for the display-cap tests."""
    return make_case(
        [
            {"src": acct(1), "dst": acct(i), "timestamp": at(i * 0.1), "amount": 100.0 * i}
            for i in range(2, width + 2)
        ],
        seed_node=acct(1),
        case_id="fixture-wide",
    )


# ---------------------------------------------------------------- encoding ---


def test_the_focal_account_is_marked(amlworld):
    focal = [n for n in amlworld.nodes if n.is_focal]
    assert len(focal) == 1
    assert focal[0].node_id == acct(1)


def test_a_focal_id_outside_the_case_is_refused():
    case = fan_out_case(width=3)
    with pytest.raises(ValueError, match="is not in case"):
        build_graph_view(view(case), "999|NOTHERE")


def test_marker_size_grows_with_degree_but_sub_linearly(amlworld):
    hub = max(amlworld.nodes, key=lambda n: n.degree)
    spoke = min(amlworld.nodes, key=lambda n: n.degree)
    assert hub.marker_size > spoke.marker_size
    assert hub.marker_size < spoke.marker_size * hub.degree


def test_flagged_accounts_are_coloured_differently_from_unflagged(amlworld):
    colours = {n.label for n in amlworld.nodes}
    assert "flagged" in colours


def test_layout_is_deterministic_for_a_case():
    case = as_laundering_stream(fan_out_case(width=5), "fan_out")
    first = build_graph_view(view(case), acct(1))
    second = build_graph_view(view(case), acct(1))
    assert [(n.x, n.y) for n in first.nodes] == [(n.x, n.y) for n in second.nodes]


def test_edge_width_encodes_amount_where_amounts_exist():
    graph = build_graph_view(view(wide_case(6)), acct(1))
    widths = {e.width for e in graph.edges}
    assert len(widths) > 1


# -------------------------------------------------- invariant 4, in pixels ---


def test_elliptic2_edges_carry_no_amount_or_currency(elliptic2):
    """A masked amount rendered as a thin line reads as 'a small amount'."""
    assert not elliptic2.has_amounts
    assert all(e.amount is None and e.currency is None for e in elliptic2.edges)


def test_elliptic2_edges_are_all_one_width(elliptic2):
    assert len({e.width for e in elliptic2.edges}) == 1


def test_elliptic2_nodes_are_all_unknown_never_defaulted_to_unflagged(elliptic2):
    """Grey-as-licit would be invariant 4 violated in pixels."""
    assert not elliptic2.has_labels
    assert {n.label for n in elliptic2.nodes} == {"unknown"}


def test_elliptic2_has_no_timeline(elliptic2):
    assert elliptic2.timeline == ()


def test_amlworld_has_a_timeline(amlworld):
    assert len(amlworld.timeline) > 1


# ------------------------------------------------------------ the display cap ---


def test_a_small_case_is_not_truncated(amlworld):
    assert not amlworld.truncated
    assert amlworld.n_hidden_nodes == 0


def test_a_large_case_is_capped():
    graph = build_graph_view(view(wide_case(120)), acct(1), max_nodes=40)
    assert len(graph.nodes) == 40
    assert graph.truncated
    assert graph.n_hidden_nodes == 121 - 40


def test_the_caption_says_the_case_is_incomplete():
    """A silently truncated graph produces a confident narrative about two thirds of a case."""
    graph = build_graph_view(view(wide_case(120)), acct(1), max_nodes=40)
    assert "not displayed" in graph.caption
    assert "do not describe this case as complete" in graph.caption


def test_the_focal_account_survives_the_cap():
    graph = build_graph_view(view(wide_case(200)), acct(1), max_nodes=20)
    assert any(n.is_focal for n in graph.nodes)


def test_hidden_edges_are_counted():
    graph = build_graph_view(view(wide_case(120)), acct(1), max_nodes=40)
    assert graph.n_hidden_edges > 0


# --------------------------------------------------------------- scrubbing ---


def test_the_scrubber_filters_by_timestamp(amlworld):
    midpoint = amlworld.timeline[len(amlworld.timeline) // 2]
    assert len(amlworld.edges_before(midpoint)) < len(amlworld.edges)
    assert len(amlworld.edges_before(None)) == len(amlworld.edges)


def test_undated_edges_are_always_shown(elliptic2):
    """Hiding them would assert they happened outside the window."""
    assert len(elliptic2.edges_before(None)) == len(elliptic2.edges)


# ------------------------------------------------------------ performance ---


@pytest.mark.slow
def test_a_150_node_case_lays_out_in_acceptable_time():
    """The extraction cap is 150 nodes, so the largest cases really do hit this."""
    case = wide_case(149)
    start = time.perf_counter()
    graph = build_graph_view(view(case), acct(1), max_nodes=150)
    elapsed = time.perf_counter() - start
    assert len(graph.nodes) == 150
    assert elapsed < 5.0, f"layout took {elapsed:.2f}s"


# ------------------------------------------------------------------ figure ---


def test_the_figure_builds(amlworld):
    plotly = pytest.importorskip("plotly")
    assert plotly
    figure = to_plotly_figure(amlworld)
    assert figure.data


def test_the_figure_title_carries_the_truncation_notice():
    pytest.importorskip("plotly")
    graph = build_graph_view(view(wide_case(120)), acct(1), max_nodes=30)
    figure = to_plotly_figure(graph)
    assert "not displayed" in figure.layout.title.text


def test_view_serialises_to_json(amlworld):
    import json

    json.dumps(amlworld.to_dict())

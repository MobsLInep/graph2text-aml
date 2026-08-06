"""The eight motif detectors: a hand-constructed positive and negative for each.

Every detector gets both. A detector tested only on positives cannot be distinguished
from one that returns True unconditionally, and on this corpus that mistake would show up
as an implausibly high typology-agreement number rather than as a test failure.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from tests.factories import (
    acct,
    at,
    bipartite_case,
    chain_case,
    cycle_case,
    fan_in_case,
    fan_out_case,
    flat_case,
    gather_scatter_case,
    make_case,
    scatter_gather_case,
    stack_case,
    view_of,
)

from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.motifs import (
    detect_bipartite,
    detect_chain,
    detect_cycle,
    detect_fan_in,
    detect_fan_out,
    detect_gather_scatter,
    detect_scatter_gather,
    detect_stack,
)

CONFIG = FactConfig()


# ------------------------------------------------------------------- fans ---


def test_fan_out_positive():
    motif = detect_fan_out(view_of(fan_out_case(width=5, hours_apart=2.0)), CONFIG)
    assert motif.present is True
    assert motif.descriptors["width"] == 5
    assert motif.descriptors["hub"] == acct(0)
    # 5 transactions two hours apart span 8 hours from first to last.
    assert motif.descriptors["window_hours"] == pytest.approx(8.0)


def test_fan_out_negative_below_the_floor():
    # Two recipients is a payment, not a pattern. fan_min_width is 3.
    motif = detect_fan_out(view_of(fan_out_case(width=2)), CONFIG)
    assert motif.present is False
    assert motif.descriptors["width"] is None
    assert motif.descriptors["hub"] is None


def test_fan_out_counts_distinct_recipients_not_transactions():
    # Ten transactions to two accounts is NOT a ten-wide fan.
    hub = acct(0)
    edges = [
        {"src": hub, "dst": acct(1 + (i % 2)), "timestamp": at(i), "amount": 100.0}
        for i in range(10)
    ]
    motif = detect_fan_out(view_of(make_case(edges, seed_node=hub)), CONFIG)
    assert motif.present is False


def test_fan_in_positive():
    motif = detect_fan_in(view_of(fan_in_case(width=4)), CONFIG)
    assert motif.present is True
    assert motif.descriptors["width"] == 4
    assert motif.descriptors["hub"] == acct(0)


def test_fan_in_negative_on_a_fan_out():
    # A pure fan-out has max in-degree 1. The two detectors must not both fire.
    assert detect_fan_in(view_of(fan_out_case(width=6)), CONFIG).present is False


def test_fan_ignores_self_loops():
    hub = acct(0)
    edges = [{"src": hub, "dst": hub, "timestamp": at(i), "amount": 100.0} for i in range(8)]
    assert detect_fan_out(view_of(make_case(edges, seed_node=hub)), CONFIG).present is False


# ------------------------------------------------------------------ chain ---


def test_chain_positive_reports_edge_length():
    motif = detect_chain(view_of(chain_case(length=4)), CONFIG)
    assert motif.present is True
    assert motif.descriptors["max_length"] == 4
    assert len(motif.witness) == 5  # 4 edges => 5 accounts


def test_chain_negative_but_length_still_reported():
    # A single forward is length 1, below chain_min_length of 3. The length is reported
    # anyway, so a narrative claiming a 5-step chain is CONTRADICTED, not UNVERIFIABLE.
    motif = detect_chain(view_of(flat_case()), CONFIG)
    assert motif.present is False
    assert motif.descriptors["max_length"] == 1


def test_chain_witness_is_a_real_directed_path():
    view = view_of(chain_case(length=5))
    motif = detect_chain(view, CONFIG)
    path = motif.witness
    for a, b in pairwise(path):
        assert b in view.successors[a]


def test_chain_falls_back_to_a_greedy_bound_on_a_large_case():
    # Above path_node_budget the search must UNDER-report, never over-report: a claim
    # then fails as CONTRADICTED rather than being wrongly SUPPORTED.
    small = FactConfig(path_node_budget=3)
    case = chain_case(length=10)
    exact = detect_chain(view_of(case), FactConfig(path_node_budget=100))
    greedy = detect_chain(view_of(case), small)
    assert greedy.descriptors["max_length"] <= exact.descriptors["max_length"]


# ------------------------------------------------------------------ cycle ---


def test_cycle_positive_reports_shortest_length():
    motif = detect_cycle(view_of(cycle_case(length=4)), CONFIG)
    assert motif.present is True
    assert motif.descriptors["length"] == 4


def test_cycle_negative_on_a_chain():
    motif = detect_cycle(view_of(chain_case(length=5)), CONFIG)
    assert motif.present is False
    assert motif.descriptors["length"] is None


def test_two_node_back_and_forth_is_not_a_cycle():
    # A refund. cycle_min_length is 3, and HI-Small is full of these.
    a, b = acct(1), acct(2)
    view = view_of(
        make_case(
            [
                {"src": a, "dst": b, "timestamp": at(0)},
                {"src": b, "dst": a, "timestamp": at(1)},
            ]
        )
    )
    assert detect_cycle(view, CONFIG).present is False


def test_self_loop_is_not_a_cycle():
    a = acct(1)
    view = view_of(
        make_case(
            [
                {"src": a, "dst": a, "timestamp": at(0)},
                {"src": a, "dst": acct(2), "timestamp": at(1)},
            ]
        )
    )
    assert detect_cycle(view, CONFIG).present is False


def test_cycle_witness_really_closes():
    view = view_of(cycle_case(length=5))
    motif = detect_cycle(view, CONFIG)
    nodes = motif.witness
    assert len(nodes) == motif.descriptors["length"]
    for a, b in pairwise(nodes):
        assert b in view.successors[a]
    assert nodes[0] in view.successors[nodes[-1]]  # closes


def test_shortest_cycle_wins_when_two_exist():
    # A 3-cycle and a longer 4-cycle sharing a node; the 3 must be reported.
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": acct(2), "timestamp": at(0)},
                {"src": acct(2), "dst": acct(3), "timestamp": at(1)},
                {"src": acct(3), "dst": acct(1), "timestamp": at(2)},
                {"src": acct(1), "dst": acct(4), "timestamp": at(3)},
                {"src": acct(4), "dst": acct(5), "timestamp": at(4)},
                {"src": acct(5), "dst": acct(6), "timestamp": at(5)},
                {"src": acct(6), "dst": acct(1), "timestamp": at(6)},
            ]
        )
    )
    assert detect_cycle(view, CONFIG).descriptors["length"] == 3


# -------------------------------------------------------------- bipartite ---


def test_bipartite_positive_is_exact():
    motif = detect_bipartite(view_of(bipartite_case(left=3, right=3)), CONFIG)
    assert motif.present is True
    assert motif.descriptors["score"] == pytest.approx(1.0)
    assert {motif.descriptors["left_size"], motif.descriptors["right_size"]} == {3}


def test_bipartite_negative_when_an_odd_cycle_exists():
    # A triangle is the canonical non-bipartite graph. score is still reported.
    motif = detect_bipartite(view_of(cycle_case(length=3)), CONFIG)
    assert motif.present is False
    assert motif.descriptors["left_size"] is None
    assert motif.descriptors["score"] < 1.0


def test_bipartite_negative_when_a_side_is_too_small():
    # 1 x 3 star: two-colourable, but one side has a single node.
    motif = detect_bipartite(view_of(fan_out_case(width=3)), CONFIG)
    assert motif.present is False


# ------------------------------------------------------------------ stack ---


def test_stack_positive_reports_depth_and_widths():
    motif = detect_stack(view_of(stack_case(depth=3, layer_width=2)), CONFIG)
    assert motif.present is True
    assert motif.descriptors["depth"] == 3
    assert motif.descriptors["layer_widths"] == [2, 2, 2]


def test_stack_negative_on_a_narrow_chain():
    # Each "layer" holds one account, so it is a chain, not a stack.
    motif = detect_stack(view_of(chain_case(length=5)), CONFIG)
    assert motif.present is False
    assert motif.descriptors["depth"] is None


def test_stack_negative_when_too_shallow():
    motif = detect_stack(view_of(stack_case(depth=2, layer_width=2)), CONFIG)
    assert motif.present is False


# -------------------------------------------------- two-sided composites ---


def test_gather_scatter_positive():
    motif = detect_gather_scatter(view_of(gather_scatter_case(gather=4, scatter=3)), CONFIG)
    assert motif.present is True
    assert motif.descriptors["gather_width"] == 4
    assert motif.descriptors["scatter_width"] == 3
    assert motif.descriptors["hub"] == acct(0)


def test_gather_scatter_negative_on_a_one_sided_fan():
    # 12 in, 0 out is a fan-in. min(gather, scatter) is what refuses to call it layering.
    motif = detect_gather_scatter(view_of(fan_in_case(width=12)), CONFIG)
    assert motif.present is False
    assert motif.descriptors["gather_width"] is None


def test_gather_scatter_excludes_counterparties_on_both_sides():
    # Money returned to its sender is not a gather-scatter. All three counterparties are
    # reciprocal, so both widths are 0.
    hub = acct(0)
    edges = []
    for i in range(1, 4):
        edges.append({"src": acct(i), "dst": hub, "timestamp": at(i)})
        edges.append({"src": hub, "dst": acct(i), "timestamp": at(10 + i)})
    motif = detect_gather_scatter(view_of(make_case(edges, seed_node=hub)), CONFIG)
    assert motif.present is False


def test_scatter_gather_positive():
    motif = detect_scatter_gather(view_of(scatter_gather_case(width=4)), CONFIG)
    assert motif.present is True
    assert motif.descriptors["width"] == 4
    assert motif.descriptors["origin"] == acct(0)
    assert motif.descriptors["destination"] == acct(999)


def test_scatter_gather_negative_when_paths_do_not_recombine():
    # A two-layer tree: the branches never meet at one destination.
    origin = acct(0)
    edges = []
    for i in range(1, 4):
        edges.append({"src": origin, "dst": acct(i), "timestamp": at(i)})
        edges.append({"src": acct(i), "dst": acct(100 + i), "timestamp": at(10 + i)})
    motif = detect_scatter_gather(view_of(make_case(edges, seed_node=origin)), CONFIG)
    assert motif.present is False


def test_scatter_gather_witness_is_two_disjoint_hops():
    view = view_of(scatter_gather_case(width=3))
    motif = detect_scatter_gather(view, CONFIG)
    origin, *middles, destination = motif.witness
    for middle in middles:
        assert middle in view.successors[origin]
        assert destination in view.successors[middle]


# ----------------------------------------------------------------- global ---


def test_flat_case_fires_nothing():
    view = view_of(flat_case())
    for detector in (
        detect_fan_in,
        detect_fan_out,
        detect_chain,
        detect_cycle,
        detect_bipartite,
        detect_stack,
        detect_gather_scatter,
        detect_scatter_gather,
    ):
        assert detector(view, CONFIG).present is False, detector.__name__


def test_detectors_are_deterministic_across_repeated_runs():
    view = view_of(gather_scatter_case(gather=5, scatter=5))
    first = detect_gather_scatter(view, CONFIG)
    second = detect_gather_scatter(view, CONFIG)
    assert first.descriptors == second.descriptors
    assert first.witness == second.witness

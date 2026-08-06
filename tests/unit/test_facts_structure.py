"""Structure sub-extractor: every value worked out by hand from the fixture shape."""

from __future__ import annotations

import pytest
from tests.factories import acct, at, bipartite_case, chain_case, fan_out_case, make_case, view_of

from g2t_aml.facts.structure import (
    density,
    diameter,
    distinct_pairs,
    extract_structure,
    reciprocity,
)


def test_counts_on_a_fan_out():
    # HUB -> S1..S5: 6 accounts, 5 transactions, one component.
    view = view_of(fan_out_case(width=5))
    facts = extract_structure(view)
    assert facts.n_nodes == 6
    assert facts.n_edges == 5
    assert facts.n_components == 1
    assert facts.max_out_degree == 5
    assert facts.max_in_degree == 1
    assert facts.n_self_loops == 0


def test_density_is_over_distinct_ordered_pairs():
    # 6 nodes, 5 distinct non-loop pairs => 5 / (6*5) = 1/6.
    view = view_of(fan_out_case(width=5))
    assert density(view) == pytest.approx(5 / 30)


def test_parallel_transactions_do_not_inflate_density():
    # Two transactions between the same pair are ONE relationship. n_edges sees both.
    a, b = acct(1), acct(2)
    view = view_of(
        make_case(
            [
                {"src": a, "dst": b, "timestamp": at(0), "amount": 100.0},
                {"src": a, "dst": b, "timestamp": at(1), "amount": 200.0},
            ]
        )
    )
    assert len(distinct_pairs(view)) == 1
    assert density(view) == pytest.approx(0.5)  # 1 / (2*1)
    assert extract_structure(view).n_edges == 2


def test_self_loops_are_counted_but_excluded_from_structure():
    # D-017 keeps self-loops. They must not create a pair, a degree or a component edge.
    a, b = acct(1), acct(2)
    view = view_of(
        make_case(
            [
                {"src": a, "dst": a, "timestamp": at(0), "amount": 100.0},
                {"src": a, "dst": b, "timestamp": at(1), "amount": 200.0},
            ]
        )
    )
    facts = extract_structure(view)
    assert facts.n_self_loops == 1
    assert facts.n_edges == 2
    assert facts.max_out_degree == 1  # the loop does not add a counterparty
    assert distinct_pairs(view) == {(a, b)}


def test_reciprocity_of_a_mutual_pair_is_one():
    a, b = acct(1), acct(2)
    view = view_of(
        make_case(
            [
                {"src": a, "dst": b, "timestamp": at(0), "amount": 100.0},
                {"src": b, "dst": a, "timestamp": at(1), "amount": 100.0},
            ]
        )
    )
    assert reciprocity(view) == pytest.approx(1.0)


def test_reciprocity_of_a_one_way_fan_is_zero():
    assert reciprocity(view_of(fan_out_case(width=4))) == pytest.approx(0.0)


def test_diameter_is_measured_on_the_undirected_projection():
    # A0->A1->A2->A3 has no directed path back, but the undirected distance is 3.
    assert diameter(view_of(chain_case(length=3))) == 3


def test_diameter_of_a_star_is_two():
    # Any two spokes are two undirected hops apart, through the hub.
    assert diameter(view_of(fan_out_case(width=4))) == 2


def test_diameter_is_none_below_two_nodes():
    view = view_of(make_case([{"src": acct(1), "dst": acct(1), "timestamp": at(0)}]))
    assert view.n_nodes == 1
    assert diameter(view) is None
    assert density(view) == 0.0


def test_components_counted_across_a_disconnected_case():
    # Two disjoint pairs => two weakly-connected components.
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": acct(2), "timestamp": at(0)},
                {"src": acct(3), "dst": acct(4), "timestamp": at(1)},
            ]
        )
    )
    facts = extract_structure(view)
    assert facts.n_components == 2
    # The diameter is the widest span WITHIN a component, not a null across them.
    assert facts.diameter == 1


def test_isolated_node_is_its_own_component():
    view = view_of(
        make_case(
            [{"src": acct(1), "dst": acct(2), "timestamp": at(0)}],
            extra_nodes=[acct(9)],
        )
    )
    assert extract_structure(view).n_components == 2


def test_bipartite_case_degrees():
    # 3 left x 3 right complete: every left node sends to 3, every right receives from 3.
    facts = extract_structure(view_of(bipartite_case(left=3, right=3)))
    assert facts.n_nodes == 6
    assert facts.n_edges == 9
    assert facts.max_out_degree == 3
    assert facts.max_in_degree == 3
    assert facts.density == pytest.approx(9 / 30)

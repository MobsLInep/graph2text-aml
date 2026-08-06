"""Labels sub-extractor: the three-way counterparty split and hop distance."""

from __future__ import annotations

import pytest
from tests.factories import EUR, USD, acct, at, elliptic2_case, make_case, view_of

from g2t_aml.facts.labels import (
    classify_counterparties,
    extract_labels,
    illicit_inflow_share,
    illicit_nodes,
    min_hops_to_illicit,
)
from g2t_aml.facts.schema import Unavailable


def test_illicit_nodes_are_both_endpoints_of_a_flagged_transaction():
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": acct(2), "timestamp": at(0), "is_laundering": True},
                {"src": acct(3), "dst": acct(4), "timestamp": at(1), "is_laundering": False},
            ]
        )
    )
    assert illicit_nodes(view) == frozenset({acct(1), acct(2)})


def test_three_way_split_separates_unlabelled_from_licit():
    # The bucket that matters: an UNLABELLED counterparty is not a licit one, and a
    # narrative describing it as licit would assert what the data does not support.
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": focal, "timestamp": at(0), "is_laundering": True},
                {"src": acct(2), "dst": focal, "timestamp": at(1), "is_laundering": False},
                {"src": acct(3), "dst": focal, "timestamp": at(2), "is_laundering": None},
            ],
            seed_node=focal,
        )
    )
    assert classify_counterparties(view, focal) == (1, 1, 1)


def test_amlworld_style_complete_labels_leave_no_unknowns():
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": focal, "timestamp": at(0), "is_laundering": True},
                {"src": acct(2), "dst": focal, "timestamp": at(1), "is_laundering": False},
            ],
            seed_node=focal,
        )
    )
    illicit, licit, unknown = classify_counterparties(view, focal)
    assert (illicit, licit, unknown) == (1, 1, 0)


def test_hops_is_zero_when_the_focal_entity_is_itself_flagged():
    focal = acct(0)
    view = view_of(
        make_case(
            [{"src": focal, "dst": acct(1), "timestamp": at(0), "is_laundering": True}],
            seed_node=focal,
        )
    )
    assert min_hops_to_illicit(view, focal) == 0


def test_hops_counts_undirected_distance():
    # focal -> A -> B, with only the A->B transaction flagged. A is one hop away.
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": focal, "dst": acct(1), "timestamp": at(0), "is_laundering": False},
                {"src": acct(1), "dst": acct(2), "timestamp": at(1), "is_laundering": True},
            ],
            seed_node=focal,
        )
    )
    assert min_hops_to_illicit(view, focal) == 1


def test_hops_is_none_when_no_flagged_account_exists():
    # A measured value -- "there is none" -- not an availability sentinel.
    focal = acct(0)
    view = view_of(
        make_case(
            [{"src": focal, "dst": acct(1), "timestamp": at(0), "is_laundering": False}],
            seed_node=focal,
        )
    )
    assert min_hops_to_illicit(view, focal) is None


def test_hops_is_none_when_the_flagged_account_is_in_another_component():
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": focal, "dst": acct(1), "timestamp": at(0), "is_laundering": False},
                {"src": acct(5), "dst": acct(6), "timestamp": at(1), "is_laundering": True},
            ],
            seed_node=focal,
        )
    )
    assert min_hops_to_illicit(view, focal) is None


def test_illicit_inflow_share_is_by_value_not_by_count():
    # One flagged transaction of 900 and three licit of 100 each: 900/1200 = 0.75 by
    # value, but only 0.25 by count. Value is the number an investigator acts on.
    focal = acct(0)
    edges = [
        {"src": acct(1), "dst": focal, "amount": 900.0, "timestamp": at(0), "is_laundering": True},
    ]
    edges += [
        {
            "src": acct(i + 2),
            "dst": focal,
            "amount": 100.0,
            "timestamp": at(i + 1),
            "is_laundering": False,
        }
        for i in range(3)
    ]
    view = view_of(make_case(edges, seed_node=focal))
    assert illicit_inflow_share(view, focal) == pytest.approx(0.75)


def test_illicit_inflow_share_refuses_across_currencies():
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {
                    "src": acct(1),
                    "dst": focal,
                    "amount": 100.0,
                    "currency": USD,
                    "timestamp": at(0),
                    "is_laundering": True,
                },
                {
                    "src": acct(2),
                    "dst": focal,
                    "amount": 100.0,
                    "currency": EUR,
                    "timestamp": at(1),
                    "is_laundering": False,
                },
            ],
            seed_node=focal,
        )
    )
    assert isinstance(illicit_inflow_share(view, focal), Unavailable)


def test_illicit_inflow_share_is_a_sentinel_with_no_inbound_value():
    focal = acct(0)
    view = view_of(make_case([{"src": focal, "dst": acct(1), "timestamp": at(0)}], seed_node=focal))
    result = illicit_inflow_share(view, focal)
    assert isinstance(result, Unavailable)
    assert result.reason == "no_inbound_value_to_take_a_share_of"


def test_block_is_a_sentinel_without_per_transaction_labels():
    # Elliptic2 sets availability.node_labels=True, but that is a SUBGRAPH-level label
    # and licenses no statement about a counterparty. The gate is the transaction label.
    view = view_of(elliptic2_case())
    block = extract_labels(view, view.node_ids[0])
    assert isinstance(block, Unavailable)
    assert block.reason == "substrate_has_no_per_transaction_illicit_labels"
    assert view.availability.node_labels is True


def test_block_totals_agree_with_the_counterparty_count():
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": focal, "timestamp": at(0), "is_laundering": True},
                {"src": acct(2), "dst": focal, "timestamp": at(1), "is_laundering": False},
                {"src": focal, "dst": acct(3), "timestamp": at(2), "is_laundering": False},
            ],
            seed_node=focal,
        )
    )
    block = extract_labels(view, focal)
    assert not isinstance(block, Unavailable)
    assert (
        block.n_illicit_counterparties
        + block.n_licit_counterparties
        + block.n_unknown_counterparties
        == block.n_counterparties
    )
    assert block.n_illicit_transactions == 1
    assert block.focal_is_illicit is True

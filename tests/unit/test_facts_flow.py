"""Flow sub-extractor, and above all the multi-currency discipline."""

from __future__ import annotations

import pytest
from tests.factories import EUR, USD, acct, at, elliptic2_case, make_case, view_of

from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.flow import (
    MULTI_CURRENCY_REASON,
    NO_TRANSFERS_REASON,
    aggregate,
    extract_flow,
    maximum,
    n_near_threshold,
    retained,
)
from g2t_aml.facts.schema import Money, Unavailable

CONFIG = FactConfig()


def test_single_currency_aggregate_sums():
    assert aggregate([(100.0, USD), (250.5, USD)]) == Money(350.5, USD)


def test_multi_currency_aggregate_refuses_to_sum():
    # The whole point: 400 USD + 3 Bitcoin is 403 of nothing.
    result = aggregate([(400.0, USD), (3.0, "Bitcoin")])
    assert isinstance(result, Unavailable)
    assert result.reason == MULTI_CURRENCY_REASON


def test_empty_aggregate_is_a_sentinel_not_zero():
    result = aggregate([])
    assert isinstance(result, Unavailable)
    assert result.reason == NO_TRANSFERS_REASON


def test_maximum_across_currencies_is_also_undefined():
    # Without a rate, 3 Bitcoin and 40,000 Rupee cannot be ordered.
    assert isinstance(maximum([(3.0, "Bitcoin"), (40000.0, "Rupee")]), Unavailable)


def test_retained_is_the_difference_in_one_currency():
    assert retained(Money(1000.0, USD), Money(400.0, USD)) == Money(600.0, USD)


def test_retained_refuses_across_currencies():
    assert isinstance(retained(Money(1000.0, USD), Money(400.0, EUR)), Unavailable)


def test_retained_is_a_sentinel_when_outflow_exceeds_inflow():
    # Legitimate: a padded window can catch dispersal of funds received before it opened.
    # Reporting a negative would invite a narrative to describe money never present.
    result = retained(Money(100.0, USD), Money(400.0, USD))
    assert isinstance(result, Unavailable)
    assert result.reason == "outflow_exceeds_inflow_within_window"


# --------------------------------------------------------- near threshold ---


def test_near_threshold_band_is_inclusive_below_and_exclusive_at_the_threshold():
    # Band is [9000, 10000). 8999.99 is outside; 9000 is in; 9999.99 is in; 10000 is NOT
    # -- a transfer AT the threshold is reportable, so it is not structured around it.
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": acct(2), "amount": 8999.99, "timestamp": at(0)},
                {"src": acct(1), "dst": acct(3), "amount": 9000.00, "timestamp": at(1)},
                {"src": acct(1), "dst": acct(4), "amount": 9999.99, "timestamp": at(2)},
                {"src": acct(1), "dst": acct(5), "amount": 10000.00, "timestamp": at(3)},
            ]
        )
    )
    assert n_near_threshold(view, CONFIG) == 2


def test_near_threshold_ignores_other_currencies():
    # Counting a 9,500 Euro transfer against a USD threshold would need a rate the
    # substrate does not carry.
    view = view_of(
        make_case(
            [
                {
                    "src": acct(1),
                    "dst": acct(2),
                    "amount": 9500.0,
                    "currency": EUR,
                    "timestamp": at(0),
                },
                {
                    "src": acct(1),
                    "dst": acct(3),
                    "amount": 9500.0,
                    "currency": USD,
                    "timestamp": at(1),
                },
            ]
        )
    )
    assert n_near_threshold(view, CONFIG) == 1


# ----------------------------------------------------------------- block ---


def test_directional_totals_use_the_currency_each_side_saw():
    # A cross-currency transfer: focal pays 1000 USD, counterparty receives 900 EUR.
    # Inflow to focal is in the RECEIVING currency; outflow in the PAYMENT currency.
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {
                    "src": acct(1),
                    "dst": focal,
                    "amount_paid": 500.0,
                    "payment_currency": EUR,
                    "amount_received": 550.0,
                    "receiving_currency": USD,
                    "timestamp": at(0),
                },
                {
                    "src": focal,
                    "dst": acct(2),
                    "amount_paid": 1000.0,
                    "payment_currency": USD,
                    "amount_received": 900.0,
                    "receiving_currency": EUR,
                    "timestamp": at(1),
                },
            ],
            seed_node=focal,
        )
    )
    flow = extract_flow(view, focal, CONFIG)
    assert not isinstance(flow, Unavailable)
    assert flow.total_inflow == Money(550.0, USD)
    assert flow.total_outflow == Money(1000.0, USD)
    assert set(flow.currencies_involved) == {USD, EUR}


def test_per_currency_breakdown_is_populated_even_when_the_aggregate_is_withheld():
    # This is what makes the sentinel safe: nothing is lost but the meaningless sum.
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
                },
                {
                    "src": acct(2),
                    "dst": focal,
                    "amount": 200.0,
                    "currency": EUR,
                    "timestamp": at(1),
                },
            ],
            seed_node=focal,
        )
    )
    flow = extract_flow(view, focal, CONFIG)
    assert not isinstance(flow, Unavailable)
    assert isinstance(flow.total_inflow, Unavailable)
    assert flow.total_inflow.reason == MULTI_CURRENCY_REASON
    assert [(t.currency, t.value, t.n_transfers) for t in flow.inflow_by_currency] == [
        (EUR, 200.0, 1),
        (USD, 100.0, 1),
    ]


def test_cross_border_is_permanently_unavailable():
    # No substrate carries jurisdiction. The field exists so the claim is explicitly
    # forbidden rather than silently absent.
    focal = acct(0)
    view = view_of(make_case([{"src": focal, "dst": acct(1), "timestamp": at(0)}], seed_node=focal))
    flow = extract_flow(view, focal, CONFIG)
    assert not isinstance(flow, Unavailable)
    assert isinstance(flow.cross_border, Unavailable)
    assert flow.cross_border.reason == "no_substrate_carries_jurisdiction"


def test_cross_institution_is_derivable_and_is_not_cross_border():
    focal = acct(0, bank="001")
    view = view_of(
        make_case([{"src": focal, "dst": acct(1, bank="077"), "timestamp": at(0)}], seed_node=focal)
    )
    flow = extract_flow(view, focal, CONFIG)
    assert not isinstance(flow, Unavailable)
    assert flow.cross_institution is True
    assert flow.n_distinct_banks == 2


def test_single_bank_case_is_not_cross_institution():
    focal = acct(0, bank="001")
    view = view_of(
        make_case([{"src": focal, "dst": acct(1, bank="001"), "timestamp": at(0)}], seed_node=focal)
    )
    flow = extract_flow(view, focal, CONFIG)
    assert not isinstance(flow, Unavailable)
    assert flow.cross_institution is False
    assert flow.n_distinct_banks == 1


def test_elliptic2_flow_is_a_sentinel_despite_populated_amount_columns():
    case = elliptic2_case()
    view = view_of(case)
    flow = extract_flow(view, view.node_ids[0], CONFIG)
    assert isinstance(flow, Unavailable)
    assert flow.reason == "substrate_has_no_monetary_amounts"


def test_max_single_transfer_spans_the_whole_case_not_just_the_focal_entity():
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": focal, "dst": acct(1), "amount": 100.0, "timestamp": at(0)},
                {"src": acct(2), "dst": acct(3), "amount": 9999.0, "timestamp": at(1)},
            ],
            seed_node=focal,
        )
    )
    flow = extract_flow(view, focal, CONFIG)
    assert not isinstance(flow, Unavailable)
    assert flow.max_single_transfer == Money(9999.0, USD)


def test_threshold_parameters_are_recorded_on_the_record():
    focal = acct(0)
    view = view_of(make_case([{"src": focal, "dst": acct(1), "timestamp": at(0)}], seed_node=focal))
    flow = extract_flow(view, focal, CONFIG)
    assert not isinstance(flow, Unavailable)
    assert flow.threshold_reference == pytest.approx(10_000.0)
    assert flow.threshold_currency == USD
    assert flow.threshold_band_fraction == pytest.approx(0.10)

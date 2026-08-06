"""Temporal sub-extractor: burst detection and the phase-ordering heuristic."""

from __future__ import annotations

import pytest
from tests.factories import BASE, acct, at, elliptic2_case, make_case, view_of

from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.schema import Unavailable
from g2t_aml.facts.temporal import detect_burst, event_ordering, extract_temporal, span_hours

CONFIG = FactConfig()


def stamps(*hours: float):
    return [at(h) for h in hours]


# ------------------------------------------------------------------ burst ---


def test_no_burst_below_the_transaction_minimum():
    # 4 transactions, all within a minute, but N=5. A burst is not "a few, fast".
    result = detect_burst(stamps(0, 0.01, 0.02, 0.03), CONFIG)
    assert result.detected is False
    assert result.count is None
    assert result.span_hours is None


def test_burst_fires_at_exactly_the_minimum():
    result = detect_burst(stamps(0, 0.1, 0.2, 0.3, 0.4), CONFIG)
    assert result.detected is True
    assert result.count == 5


def test_no_burst_when_transactions_are_spread_beyond_the_window():
    # 5 transactions, 30 hours apart: many transactions, no burst. This is the case the
    # definition exists to exclude.
    result = detect_burst(stamps(0, 30, 60, 90, 120), CONFIG)
    assert result.detected is False


def test_reported_span_is_observed_not_the_configured_cap():
    # 6 transactions inside 2 hours, under a 24-hour cap. The record must say 2, not 24 --
    # otherwise the rapid_dispersal binding would be trivially satisfiable.
    result = detect_burst(stamps(0, 0.4, 0.8, 1.2, 1.6, 2.0), CONFIG)
    assert result.detected is True
    assert result.count == 6
    assert result.span_hours == pytest.approx(2.0)
    assert result.start == BASE


def test_burst_maximises_count_then_minimises_span():
    # A tight cluster of 5 and a wider cluster of 7, both inside 24h but far apart.
    # The larger cluster wins on count.
    result = detect_burst(stamps(0, 0.1, 0.2, 0.3, 0.4, 100, 101, 103, 105, 107, 109, 111), CONFIG)
    assert result.count == 7
    assert result.span_hours == pytest.approx(11.0)


def test_burst_at_exactly_the_window_boundary_is_included():
    # The window is inclusive: 5 transactions spanning exactly 24.0 hours qualifies.
    result = detect_burst(stamps(0, 6, 12, 18, 24), CONFIG)
    assert result.detected is True
    assert result.span_hours == pytest.approx(24.0)


def test_burst_just_beyond_the_boundary_is_excluded():
    result = detect_burst(stamps(0, 6, 12, 18, 24.001), CONFIG)
    assert result.detected is False


def test_burst_is_deterministic_under_input_reordering():
    ordered = detect_burst(stamps(0, 1, 2, 3, 4, 5), CONFIG)
    shuffled = detect_burst(stamps(3, 0, 5, 2, 4, 1), CONFIG)
    assert (ordered.count, ordered.span_hours, ordered.start) == (
        shuffled.count,
        shuffled.span_hours,
        shuffled.start,
    )


# --------------------------------------------------------- phase ordering ---


def test_ordering_inflow_only():
    focal = acct(0)
    view = view_of(make_case([{"src": acct(1), "dst": focal, "timestamp": at(0)}]))
    assert event_ordering(view, focal, CONFIG) == ("inflow_phase",)


def test_ordering_outflow_only():
    focal = acct(0)
    view = view_of(make_case([{"src": focal, "dst": acct(1), "timestamp": at(0)}]))
    assert event_ordering(view, focal, CONFIG) == ("outflow_phase",)


def test_ordering_with_a_holding_gap_reports_consolidation():
    # All inflow, a 5-hour quiet gap, then all outflow.
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": focal, "timestamp": at(0)},
                {"src": acct(2), "dst": focal, "timestamp": at(1)},
                {"src": focal, "dst": acct(3), "timestamp": at(6)},
            ]
        )
    )
    assert event_ordering(view, focal, CONFIG) == (
        "inflow_phase",
        "consolidation",
        "outflow_phase",
    )


def test_ordering_without_a_gap_omits_consolidation():
    # Same shape, but the outflow follows six minutes later: same-batch processing, not
    # deliberate holding.
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": focal, "timestamp": at(0)},
                {"src": focal, "dst": acct(3), "timestamp": at(0.1)},
            ]
        )
    )
    assert event_ordering(view, focal, CONFIG) == ("inflow_phase", "outflow_phase")


def test_ordering_reports_interleaved_when_the_phases_overlap():
    # An inflow AFTER an outflow means the account did not receive-then-disperse. The
    # strict rule is what stops "layering" being claimed of ordinary two-way activity.
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": acct(1), "dst": focal, "timestamp": at(0)},
                {"src": focal, "dst": acct(2), "timestamp": at(5)},
                {"src": acct(3), "dst": focal, "timestamp": at(10)},
            ]
        )
    )
    assert event_ordering(view, focal, CONFIG) == ("interleaved",)


def test_ordering_reports_outflow_before_inflow():
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": focal, "dst": acct(1), "timestamp": at(0)},
                {"src": acct(2), "dst": focal, "timestamp": at(5)},
            ]
        )
    )
    assert event_ordering(view, focal, CONFIG) == ("outflow_phase", "inflow_phase")


def test_ordering_is_empty_when_the_focal_entity_only_self_loops():
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": focal, "dst": focal, "timestamp": at(0)},
                {"src": acct(1), "dst": acct(2), "timestamp": at(1)},
            ]
        )
    )
    assert event_ordering(view, focal, CONFIG) == ()


# ---------------------------------------------------------------- block ---


def test_span_is_the_observed_extent():
    focal = acct(0)
    view = view_of(
        make_case(
            [
                {"src": focal, "dst": acct(1), "timestamp": at(0)},
                {"src": focal, "dst": acct(2), "timestamp": at(36)},
            ],
            seed_node=focal,
        )
    )
    block = extract_temporal(view, focal, CONFIG)
    assert not isinstance(block, Unavailable)
    assert block.span_hours == pytest.approx(36.0)
    assert block.n_transactions == 2


def test_elliptic2_gets_a_sentinel_not_a_zeroed_block():
    # The fixture HAS a populated timestamp column. The mask is what governs.
    case = elliptic2_case()
    view = view_of(case)
    block = extract_temporal(view, view.node_ids[0], CONFIG)
    assert isinstance(block, Unavailable)
    assert block.reason == "substrate_has_no_absolute_timestamps"
    assert block.available is False


def test_span_hours_helper():
    assert span_hours(at(0), at(2.5)) == pytest.approx(2.5)

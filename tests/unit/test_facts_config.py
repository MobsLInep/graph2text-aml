"""FactConfig and ToleranceConfig validation: every guard, exercised.

These are the paths that fire when someone misconfigures a run. A threshold that silently
accepted a nonsense value would produce a fact record whose `present` flags are
unreproducible, which is precisely what `provenance.config` exists to prevent.
"""

from __future__ import annotations

import pytest

from g2t_aml.facts.config import FactConfig, ToleranceConfig


def test_defaults_construct():
    config = FactConfig()
    assert config.burst_min_transactions == 5
    assert config.threshold_currency == "US Dollar"
    assert isinstance(config.tolerance, ToleranceConfig)


def test_threshold_floor_is_derived():
    config = FactConfig(threshold_reference=10_000.0, threshold_band_fraction=0.10)
    assert config.threshold_floor == pytest.approx(9_000.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("burst_min_transactions", 0),
        ("hub_min_degree", 0),
        ("fan_min_width", 0),
        ("two_sided_min_width", 0),
        ("chain_min_length", 0),
        ("cycle_min_length", 1),
        ("cycle_max_length", 1),
        ("cycle_edge_budget", 0),
        ("path_node_budget", 0),
        ("stack_min_depth", 0),
        ("stack_min_layer_width", 0),
        ("bipartite_min_side", 0),
        ("max_inventory_nodes", 0),
    ],
)
def test_count_thresholds_below_their_minimum_raise(field, value):
    with pytest.raises(ValueError, match=field):
        FactConfig(**{field: value})


def test_cycle_min_above_max_raises():
    # Otherwise no cycle could ever be reported, and the detector would be silently dead.
    with pytest.raises(ValueError, match="exceeds cycle_max_length"):
        FactConfig(cycle_min_length=8, cycle_max_length=4)


def test_non_positive_burst_window_raises():
    with pytest.raises(ValueError, match="burst_window_hours"):
        FactConfig(burst_window_hours=0.0)


def test_negative_consolidation_gap_raises():
    with pytest.raises(ValueError, match="consolidation_min_gap_hours"):
        FactConfig(consolidation_min_gap_hours=-1.0)


def test_non_positive_threshold_reference_raises():
    with pytest.raises(ValueError, match="threshold_reference"):
        FactConfig(threshold_reference=0.0)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_band_fraction_outside_the_unit_interval_raises(fraction):
    with pytest.raises(ValueError, match="threshold_band_fraction"):
        FactConfig(threshold_band_fraction=fraction)


def test_empty_threshold_currency_raises():
    # An amount without a unit is checkable against nothing.
    with pytest.raises(ValueError, match="threshold_currency"):
        FactConfig(threshold_currency="")


def test_counts_exact_cannot_be_disabled():
    # A published commitment, not a knob: a run that relaxed it would report a number
    # that does not mean what the paper says it means.
    with pytest.raises(ValueError, match="counts_exact cannot be disabled"):
        ToleranceConfig(counts_exact=False)


@pytest.mark.parametrize(
    "field",
    [
        "monetary_relative",
        "monetary_absolute_floor",
        "duration_granularity_units",
        "share_absolute",
        "score_absolute",
    ],
)
def test_negative_tolerances_raise(field):
    with pytest.raises(ValueError, match=field):
        ToleranceConfig(**{field: -0.5})


def test_round_trip_through_dict():
    config = FactConfig(fan_min_width=7, threshold_reference=5000.0)
    assert FactConfig.from_dict(config.to_dict()) == config


def test_from_dict_rejects_an_unknown_key():
    # A stale serialised config must fail loudly rather than defaulting a threshold that
    # would silently change a detector's verdict.
    with pytest.raises(ValueError, match="unknown fact config keys"):
        FactConfig.from_dict({"fan_min_width": 3, "not_a_field": 1})


def test_to_dict_is_json_serialisable():
    import json

    json.dumps(FactConfig().to_dict())


def test_tolerance_to_dict():
    assert ToleranceConfig().to_dict()["monetary_relative"] == pytest.approx(0.01)

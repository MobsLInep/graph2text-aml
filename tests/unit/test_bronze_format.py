"""Formatting and its inverse, and the reconciliation with the checker's tolerances.

The table in ``corpus/bronze/format.py`` is a claim about rounding error against the
published tolerance policy. It is asserted here by property test rather than by reading,
because the failure it guards against is silent: a formatter whose rounding drifted outside
tolerance would make every Bronze narrative carrying that field CONTRADICTED, and the first
sign would be a gate failure fifteen thousand records into a build.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from g2t_aml.corpus.bronze import format as fmt
from g2t_aml.facts.config import ToleranceConfig

TOLERANCE = ToleranceConfig()


class TestCounts:
    def test_round_trips_exactly(self) -> None:
        assert fmt.parse_count(fmt.format_count(1204)) == 1204

    def test_uses_thousands_separators(self) -> None:
        assert fmt.format_count(1234567) == "1,234,567"

    def test_refuses_a_non_integral_count(self) -> None:
        with pytest.raises(fmt.FormatError, match="not integral"):
            fmt.format_count(3.5)

    @given(st.integers(min_value=0, max_value=10**9))
    def test_counts_are_never_rounded(self, value: int) -> None:
        assert fmt.parse_count(fmt.format_count(value)) == value


class TestMoney:
    def test_rounds_to_four_significant_figures(self) -> None:
        assert fmt.format_money(482299.87, "US Dollar") == "482,300 US Dollar"

    def test_small_amounts_keep_the_cents(self) -> None:
        assert fmt.format_money(842.174, "Euro") == "842.17 Euro"

    def test_carries_the_currency_through_the_round_trip(self) -> None:
        value, currency = fmt.parse_money(fmt.format_money(294203.05, "Saudi Riyal"))
        assert currency == "Saudi Riyal"
        assert value == pytest.approx(294200.0)

    def test_refuses_an_amount_without_a_currency(self) -> None:
        with pytest.raises(fmt.FormatError, match="must carry its currency"):
            fmt.format_money(100.0, "  ")

    @given(
        st.floats(min_value=0.01, max_value=1e12, allow_nan=False, allow_infinity=False),
        st.sampled_from(["US Dollar", "Euro", "Bitcoin", "Saudi Riyal", "Swiss Franc"]),
    )
    @settings(max_examples=400)
    def test_rounding_stays_inside_the_checker_tolerance(self, value: float, currency: str) -> None:
        """The reconciliation the module docstring promises, asserted."""
        stated, parsed_currency = fmt.parse_money(fmt.format_money(value, currency))
        assert parsed_currency == currency
        allowed = max(abs(value) * TOLERANCE.monetary_relative, TOLERANCE.monetary_absolute_floor)
        assert abs(stated - value) <= allowed

    @given(st.floats(min_value=1000, max_value=1e12, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_large_amounts_keep_a_factor_of_ten_of_margin(self, value: float) -> None:
        """4 significant figures is 0.05%, against a 1% tolerance. The margin is the point."""
        stated, _ = fmt.parse_money(fmt.format_money(value, "US Dollar"))
        assert abs(stated - value) <= abs(value) * 0.001


class TestDurations:
    @pytest.mark.parametrize(
        ("hours", "expected"),
        [(0.3, "18 minutes"), (1.0, "1.0 hours"), (41.57, "41.6 hours"), (76.0, "3 days")],
    )
    def test_chooses_a_natural_granularity(self, hours: float, expected: str) -> None:
        assert fmt.format_duration(hours) == expected

    def test_parses_back_to_a_plural_unit(self) -> None:
        assert fmt.parse_duration("1 day") == (1.0, "days")

    @given(st.floats(min_value=0.0, max_value=10_000, allow_nan=False, allow_infinity=False))
    @settings(max_examples=400)
    def test_rounding_stays_inside_one_stated_unit(self, hours: float) -> None:
        """The checker allows one unit of whatever the narrative states (D-027)."""
        value, unit = fmt.parse_duration(fmt.format_duration(hours))
        factors = {"minutes": 1 / 60, "hours": 1.0, "days": 24.0}
        stated_hours = value * factors[unit]
        allowed = TOLERANCE.duration_granularity_units * factors[unit]
        assert abs(stated_hours - hours) <= allowed


class TestSharesAndScores:
    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_percent_rounding_stays_inside_the_share_tolerance(self, share: float) -> None:
        assert abs(fmt.parse_percent(fmt.format_percent(share)) - share) <= TOLERANCE.share_absolute

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_density_rounding_stays_inside_the_score_tolerance(self, value: float) -> None:
        assert abs(fmt.parse_density(fmt.format_density(value)) - value) <= TOLERANCE.score_absolute


class TestTimestamps:
    def test_renders_at_minute_resolution(self) -> None:
        assert fmt.format_timestamp(datetime(2022, 9, 5, 16, 7, 42)) == "2022-09-05 16:07"

    def test_round_trip_stays_inside_the_sixty_second_tolerance(self) -> None:
        moment = datetime(2022, 9, 5, 16, 7, 42)
        parsed = fmt.parse_timestamp(fmt.format_timestamp(moment))
        assert abs((parsed - moment).total_seconds()) <= 60


class TestParsersRejectNonsense:
    @pytest.mark.parametrize(
        ("parser", "text"),
        [
            (fmt.parse_count, "twelve"),
            (fmt.parse_money, "482300"),
            (fmt.parse_duration, "41.6"),
            (fmt.parse_percent, "62"),
            (fmt.parse_timestamp, "yesterday"),
        ],
    )
    def test_a_parser_raises_rather_than_guessing(self, parser: object, text: str) -> None:
        with pytest.raises(fmt.FormatError):
            parser(text)  # type: ignore[operator]

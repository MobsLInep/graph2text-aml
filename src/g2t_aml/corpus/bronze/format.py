"""Number formatting, and the parsers that read it back.

**Every function here comes in a pair**, and the pair is the point. ``format_money``
writes ``"482,300 US Dollar"``; ``parse_money`` reads it back to ``(482300.0, "US
Dollar")``. The claim the checker verifies is produced by the *parser reading the rendered
text*, never by handing it the value the formatter started from.

That indirection is what makes Bronze's 100%-faithful result mean anything. A generator
that emitted its claims straight from the record would be comparing the record with
itself: Phase 3 learned that lesson the expensive way (D-034, three injected extractor bugs
that the round trip reported as 100% SUPPORTED). Here the same trap is available and
avoided — if ``format_money`` dropped a factor of a thousand, or wrote a thousands
separator the parser reads as a decimal point, the claim parsed out of the text would
disagree with the record and the case would be CONTRADICTED rather than quietly wrong.

**Rounding is reconciled against the published tolerance policy, not chosen for looks.**
Each formatter's rounding error is bounded strictly inside the tolerance the checker will
apply to it, with margin:

=================  ==========================  =========================  ==============
Quantity           Rendering                   Worst-case error           Tolerance
=================  ==========================  =========================  ==============
Counts             Exact, ``,`` separated      0                          exact
Money >= 1000      4 significant figures       <= 0.05% relative          1% relative
Money < 1000       2 decimal places            <= 0.005 absolute          1% or 0.01
Duration >= 48h    Whole days                  <= 12 h                    24 h (1 day)
Duration 1..48h    1 decimal place, hours      <= 0.05 h                  1 h
Duration < 1h      Whole minutes               <= 0.5 min                 1 min
Shares             Whole percent               <= 0.005                   0.01
Density, scores    3 decimal places            <= 0.0005                  0.01
Timestamps         Minute resolution           <= 30 s                    60 s
=================  ==========================  =========================  ==============

Every row has at least a factor-of-two margin, and ``tests/unit/test_bronze_format.py``
asserts the whole table by property test rather than by inspection. The one row worth
staring at is money: 4 significant figures is chosen over 3 because 3 gives 0.5% error
against a 1% tolerance, which leaves no room for a later change to either side. The
narrative always hedges a rounded amount ("approximately"), because a number rendered to
4 significant figures is not being claimed exactly and should not read as though it is.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

__all__ = [
    "MONEY_RE",
    "format_count",
    "format_density",
    "format_duration",
    "format_money",
    "format_percent",
    "format_timestamp",
    "parse_count",
    "parse_density",
    "parse_duration",
    "parse_money",
    "parse_percent",
    "parse_timestamp",
]

#: A rendered amount: digits with optional thousands separators and decimals, then the
#: currency name as the substrate spells it ("US Dollar", "Saudi Riyal", "Bitcoin").
MONEY_RE = re.compile(r"^(?P<value>-?[\d,]+(?:\.\d+)?)\s+(?P<currency>[A-Za-z][A-Za-z .'-]*)$")

_DURATION_RE = re.compile(r"^(?P<value>-?[\d,]+(?:\.\d+)?)\s+(?P<unit>minutes?|hours?|days?)$")
_PERCENT_RE = re.compile(r"^(?P<value>-?\d+(?:\.\d+)?)%$")
_COUNT_RE = re.compile(r"^-?[\d,]+$")

#: Timestamp rendering. Minute resolution, matching AMLworld's own; the checker allows
#: 60 s, so a narrative cannot be required to be more precise than the substrate is.
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"

#: Significant figures a large amount is rounded to. See the module docstring.
_MONEY_SIGNIFICANT_FIGURES = 4

#: Below this, an amount is written to the cent rather than to significant figures.
_MONEY_EXACT_BELOW = 1000.0

#: At or above this many hours a duration is written in whole days.
_DAYS_ABOVE_HOURS = 48.0


class FormatError(ValueError):
    """Raised when a rendered value cannot be parsed back, which is always a bug here."""


def format_count(value: int | float) -> str:
    """Render an exact count.

    Counts get no tolerance at all, so they get no rounding either.

    Args:
        value: The count.

    Returns:
        The integer with thousands separators, e.g. ``"1,204"``.

    Raises:
        FormatError: If ``value`` is not integral. A count that is not a whole number is
            a bug upstream, and rounding it here would hide it.
    """
    if float(value) != int(value):
        raise FormatError(f"count {value!r} is not integral; counts are checked exactly")
    return f"{int(value):,}"


def parse_count(text: str) -> int:
    """Read back a rendered count.

    Args:
        text: Output of :func:`format_count`.

    Returns:
        The count.

    Raises:
        FormatError: If ``text`` is not a rendered count.
    """
    stripped = text.strip()
    if not _COUNT_RE.match(stripped):
        raise FormatError(f"{text!r} is not a rendered count")
    return int(stripped.replace(",", ""))


def _round_significant(value: float, figures: int) -> float:
    """Round to a number of significant figures.

    Args:
        value: The value.
        figures: Significant figures to keep.

    Returns:
        The rounded value. Zero rounds to zero.
    """
    if value == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    quantum = 10.0 ** (magnitude - figures + 1)
    return round(value / quantum) * quantum


def format_money(value: float, currency: str) -> str:
    """Render a monetary amount with its currency.

    Args:
        value: The amount.
        currency: The currency name as the substrate spells it.

    Returns:
        ``"482,300 US Dollar"`` for a large amount, ``"842.17 US Dollar"`` for a small
        one. Never a bare number: an amount without its unit is a claim the checker
        cannot contradict, which is the one kind of wrong sentence this project cannot
        detect.

    Raises:
        FormatError: If ``currency`` is empty, or would not survive the round trip
            through :func:`parse_money` — a currency containing a digit, say, which no
            substrate has but which would silently corrupt the parse if one ever did.
    """
    if not currency.strip():
        raise FormatError("an amount must carry its currency")
    if abs(value) < _MONEY_EXACT_BELOW:
        rendered = f"{value:,.2f}"
    else:
        rounded = _round_significant(value, _MONEY_SIGNIFICANT_FIGURES)
        rendered = f"{rounded:,.0f}" if rounded == int(rounded) else f"{rounded:,.2f}"
    text = f"{rendered} {currency}"
    if not MONEY_RE.match(text):
        raise FormatError(
            f"rendered amount {text!r} cannot be parsed back; currency names must be "
            "alphabetic so the value and the unit stay separable"
        )
    return text


def parse_money(text: str) -> tuple[float, str]:
    """Read back a rendered amount.

    Args:
        text: Output of :func:`format_money`.

    Returns:
        ``(value, currency)``.

    Raises:
        FormatError: If ``text`` is not a rendered amount.
    """
    match = MONEY_RE.match(text.strip())
    if match is None:
        raise FormatError(f"{text!r} is not a rendered monetary amount")
    return float(match.group("value").replace(",", "")), match.group("currency").strip()


def format_duration(hours: float) -> str:
    """Render a duration at the granularity a narrative would naturally choose.

    The granularity is part of the claim: the checker allows one unit of whatever the
    text states, so writing "3 days" claims less precision than "76.0 hours" and is
    judged accordingly (D-027).

    Args:
        hours: The duration in hours.

    Returns:
        ``"3 days"``, ``"36.5 hours"`` or ``"18 minutes"``.
    """
    if hours >= _DAYS_ABOVE_HOURS:
        days = round(hours / 24)
        return f"{days:,} {'day' if days == 1 else 'days'}"
    if hours >= 1.0:
        value = round(hours, 1)
        return f"{value:,.1f} hours"
    minutes = round(hours * 60)
    return f"{minutes:,} {'minute' if minutes == 1 else 'minutes'}"


def parse_duration(text: str) -> tuple[float, str]:
    """Read back a rendered duration.

    Args:
        text: Output of :func:`format_duration`.

    Returns:
        ``(value, unit)`` where unit is plural: ``"minutes"``, ``"hours"`` or ``"days"``.
        The plural form is what :class:`~g2t_aml.facts.checkers.DurationClaim` expects.

    Raises:
        FormatError: If ``text`` is not a rendered duration.
    """
    match = _DURATION_RE.match(text.strip())
    if match is None:
        raise FormatError(f"{text!r} is not a rendered duration")
    unit = match.group("unit")
    return float(match.group("value").replace(",", "")), unit if unit.endswith("s") else unit + "s"


def format_percent(share: float) -> str:
    """Render a share in [0, 1] as a whole percentage.

    Args:
        share: The share.

    Returns:
        ``"62%"``.
    """
    return f"{round(share * 100)}%"


def parse_percent(text: str) -> float:
    """Read back a rendered percentage as a share in [0, 1].

    Args:
        text: Output of :func:`format_percent`.

    Returns:
        The share.

    Raises:
        FormatError: If ``text`` is not a rendered percentage.
    """
    match = _PERCENT_RE.match(text.strip())
    if match is None:
        raise FormatError(f"{text!r} is not a rendered percentage")
    return float(match.group("value")) / 100.0


def format_density(value: float) -> str:
    """Render a density or bounded score to three decimal places.

    Args:
        value: The value, normally in [0, 1].

    Returns:
        ``"0.071"``.
    """
    return f"{value:.3f}"


def parse_density(text: str) -> float:
    """Read back a rendered density or score.

    Args:
        text: Output of :func:`format_density`.

    Returns:
        The value.

    Raises:
        FormatError: If ``text`` is not a rendered decimal.
    """
    try:
        return float(text.strip())
    except ValueError as exc:
        raise FormatError(f"{text!r} is not a rendered decimal") from exc


def format_timestamp(value: datetime) -> str:
    """Render a timestamp at minute resolution.

    Args:
        value: The moment.

    Returns:
        ``"2022-09-05 16:07"``.
    """
    return value.strftime(_TIMESTAMP_FORMAT)


def parse_timestamp(text: str) -> datetime:
    """Read back a rendered timestamp.

    Args:
        text: Output of :func:`format_timestamp`.

    Returns:
        The moment, with seconds zeroed.

    Raises:
        FormatError: If ``text`` is not a rendered timestamp.
    """
    try:
        return datetime.strptime(text.strip(), _TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise FormatError(f"{text!r} is not a rendered timestamp") from exc

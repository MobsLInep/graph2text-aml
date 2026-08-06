"""When things happened: window, span, burst detection, phase ordering.

Two algorithms in this module are decisions rather than calculations, and both are defined
here in full because a narrative's most persuasive sentences rest on them.

**Burst detection is not "many transactions".** A case with forty transactions spread
evenly over two days is not bursty; a case with six inside twenty minutes is, even though
it has fewer. The definition used here is explicit and has two parameters:

    A burst exists iff there is some time window of length at most H hours containing at
    least N transactions. The burst REPORTED is the tightest such window: maximise the
    transaction count, and among windows achieving that maximum, minimise the span.

``H`` is ``FactConfig.burst_window_hours`` (default 24, aligned with the
``rapid_dispersal`` binding in the vocabulary so the phrase and the detector cannot
disagree about what "short" means) and ``N`` is ``FactConfig.burst_min_transactions``
(default 5). The reported ``burst_window_hours`` is the **observed span of that tightest
window**, not the configured cap — which is what makes the ``rapid_dispersal`` binding
meaningful, since a narrative saying "dispersed within a short window" is claiming
something about the observed span and would be trivially true against the cap.

The implementation is a two-pointer sweep over sorted timestamps: O(n log n) for the sort
and O(n) for the sweep, with a total order over (count, -span, start) so two runs cannot
disagree about which of several equally tight windows to report.

**Phase ordering is a documented heuristic, and it is deliberately strict.** The
temptation is to declare "inflow then outflow" whenever the median inflow precedes the
median outflow, which is true of almost any account and would make the phrase meaningless.
Instead:

    - no non-loop transactions            -> []
    - inbound only                        -> [inflow_phase]
    - outbound only                       -> [outflow_phase]
    - every inbound strictly precedes every outbound:
        gap >= consolidation_min_gap_hours -> [inflow_phase, consolidation, outflow_phase]
        otherwise                          -> [inflow_phase, outflow_phase]
    - every outbound precedes every inbound -> [outflow_phase, inflow_phase]
    - otherwise                            -> [interleaved]

``interleaved`` is a first-class outcome, not a failure to decide. It states that inflows
and outflows overlap in time, which forbids any narrative claim of a receive-then-disperse
sequence — and on real data it is the common case, which is exactly why the strict rule is
the honest one.
"""

from __future__ import annotations

from datetime import datetime

from g2t_aml.facts.caseview import CaseEdge, CaseView
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.schema import TemporalFacts, Unavailable

__all__ = [
    "FIELD_PRODUCERS",
    "BurstResult",
    "detect_burst",
    "event_ordering",
    "extract_temporal",
    "span_hours",
]

FIELD_PRODUCERS: dict[str, str] = {
    "temporal.window_start": "temporal.observed_extent",
    "temporal.window_end": "temporal.observed_extent",
    "temporal.span_hours": "temporal.observed_extent",
    "temporal.burst_detected": "temporal.sliding_window_burst",
    "temporal.burst_window_hours": "temporal.sliding_window_burst",
    "temporal.burst_txn_count": "temporal.sliding_window_burst",
    "temporal.burst_start": "temporal.sliding_window_burst",
    "temporal.event_ordering": "temporal.phase_ordering_heuristic",
    "temporal.n_transactions": "temporal.transaction_count",
}

#: The closed phase vocabulary. Mirrors ``$defs.phase`` in the JSON Schema.
PHASES: tuple[str, ...] = ("inflow_phase", "consolidation", "outflow_phase", "interleaved")

_SECONDS_PER_HOUR = 3600.0


class BurstResult:
    """The tightest qualifying burst, or the absence of one.

    Attributes:
        detected: Whether a qualifying burst exists.
        count: Transactions inside the burst, or None.
        span_hours: Observed span of the burst, or None.
        start: First transaction of the burst, or None.
    """

    __slots__ = ("count", "detected", "span_hours", "start")

    def __init__(
        self,
        detected: bool,
        count: int | None = None,
        span_hours: float | None = None,
        start: datetime | None = None,
    ) -> None:
        """Create a burst result.

        Args:
            detected: Whether a qualifying burst exists.
            count: Transactions inside it.
            span_hours: Its observed span.
            start: Its first transaction.
        """
        self.detected = detected
        self.count = count
        self.span_hours = span_hours
        self.start = start


def _timestamps(edges: tuple[CaseEdge, ...]) -> list[datetime]:
    """Return the sorted, non-null timestamps of a set of transactions.

    Args:
        edges: Transactions to read.

    Returns:
        Timestamps ascending. Transactions without one are dropped, which cannot happen
        on a case that passed :attr:`~g2t_aml.facts.caseview.CaseView.has_timestamps`.
    """
    return sorted(e.timestamp for e in edges if e.timestamp is not None)


def span_hours(start: datetime, end: datetime) -> float:
    """Return the interval between two moments, in hours.

    Args:
        start: Earlier moment.
        end: Later moment.

    Returns:
        ``(end - start)`` in hours, rounded to six places so a serialised record is
        byte-stable across runs.
    """
    return round((end - start).total_seconds() / _SECONDS_PER_HOUR, 6)


def detect_burst(stamps: list[datetime], config: FactConfig) -> BurstResult:
    """Find the tightest window of at least N transactions within at most H hours.

    See the module docstring for the definition and why it is not simply "many
    transactions". The sweep is two-pointer over sorted timestamps and the tie-break is a
    total order, so the result is deterministic.

    Args:
        stamps: Transaction timestamps. Sorted internally, so the caller need not.
        config: Supplies ``burst_min_transactions`` (N) and ``burst_window_hours`` (H).

    Returns:
        The tightest qualifying burst, or ``BurstResult(detected=False)`` when no window
        of at most H hours holds N transactions.
    """
    ordered = sorted(stamps)
    if len(ordered) < config.burst_min_transactions:
        return BurstResult(detected=False)

    limit = config.burst_window_hours * _SECONDS_PER_HOUR
    best_count = 0
    best_span = 0.0
    best_start: datetime | None = None

    left = 0
    for right in range(len(ordered)):
        # Shrink from the left until the window fits inside H hours. Each index is
        # advanced at most once overall, so the whole sweep is linear.
        while (ordered[right] - ordered[left]).total_seconds() > limit:
            left += 1
        count = right - left + 1
        if count < config.burst_min_transactions:
            continue
        span = (ordered[right] - ordered[left]).total_seconds()
        # Total order: more transactions wins; then a tighter span; then an earlier
        # start. Without the third term two windows of equal count and equal span would
        # be chosen by list order, which is stable but not meaningful.
        if best_start is None or (count, -span) > (best_count, -best_span):
            best_count, best_span, best_start = count, span, ordered[left]

    if best_start is None:
        return BurstResult(detected=False)
    return BurstResult(
        detected=True,
        count=best_count,
        span_hours=round(best_span / _SECONDS_PER_HOUR, 6),
        start=best_start,
    )


def event_ordering(  # noqa: PLR0911 -- one return per documented phase outcome
    view: CaseView, focal: str, config: FactConfig
) -> tuple[str, ...]:
    """Classify the focal entity's inflow/outflow phase sequence.

    Implements the heuristic in the module docstring verbatim.

    Args:
        view: The case view.
        focal: The focal entity.
        config: Supplies ``consolidation_min_gap_hours``.

    Returns:
        A tuple of members of :data:`PHASES`. Empty when the focal entity has no non-loop
        transactions at all.
    """
    inbound = _timestamps(view.edges_into(focal))
    outbound = _timestamps(view.edges_out_of(focal))

    if not inbound and not outbound:
        return ()
    if not outbound:
        return ("inflow_phase",)
    if not inbound:
        return ("outflow_phase",)

    if max(inbound) <= min(outbound):
        gap = span_hours(max(inbound), min(outbound))
        if gap >= config.consolidation_min_gap_hours:
            return ("inflow_phase", "consolidation", "outflow_phase")
        return ("inflow_phase", "outflow_phase")
    if max(outbound) <= min(inbound):
        return ("outflow_phase", "inflow_phase")
    return ("interleaved",)


def extract_temporal(view: CaseView, focal: str, config: FactConfig) -> TemporalFacts | Unavailable:
    """Extract the whole temporal block, or the sentinel that replaces it.

    Args:
        view: The case view.
        focal: The focal entity, whose transactions drive the phase ordering.
        config: Burst and consolidation thresholds.

    Returns:
        The populated :class:`~g2t_aml.facts.schema.TemporalFacts`, or an
        :class:`~g2t_aml.facts.schema.Unavailable` when the substrate has no absolute
        timestamps or the case carries no usable ones. Never a zeroed block: "the window
        is 0 hours long" and "there is no clock" are different statements and only one of
        them would be true.
    """
    if not view.availability.absolute_timestamps:
        return Unavailable("substrate_has_no_absolute_timestamps")
    stamps = _timestamps(view.edges)
    if not stamps:
        return Unavailable("case_carries_no_transaction_timestamps")

    window_start, window_end = stamps[0], stamps[-1]
    burst = detect_burst(stamps, config)
    return TemporalFacts(
        window_start=window_start,
        window_end=window_end,
        span_hours=span_hours(window_start, window_end),
        burst_detected=burst.detected,
        burst_window_hours=burst.span_hours,
        burst_txn_count=burst.count,
        burst_start=burst.start,
        event_ordering=event_ordering(view, focal, config),
        n_transactions=len(stamps),
    )

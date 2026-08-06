"""How much moved: inflow, outflow, retention, near-threshold detection.

**The multi-currency rule is the load-bearing decision in this module.** HI-Small carries
fifteen currencies and 72,170 cross-currency transactions, and it ships no exchange rates.
Summing 400,000 US Dollars and 3,000 Bitcoin produces 403,000 of nothing. This module
therefore never sums across currencies:

    - Aggregates are reported as :class:`~g2t_aml.facts.schema.Money` when every
      contributing transfer shares one currency.
    - When they do not, the aggregate is an
      :class:`~g2t_aml.facts.schema.Unavailable` with reason
      ``multi_currency_aggregate_undefined`` — and the per-currency breakdown is
      populated regardless, so nothing is lost except the sum that had no meaning.

That is why ``inflow_by_currency`` and ``outflow_by_currency`` are always present rather
than only appearing in the multi-currency case: the sentinel is only safe because the
information it withholds is available in a form that is actually true.

**Direction is measured in the currency each side actually saw.** An inflow is credited to
the focal entity in ``receiving_currency`` at ``amount_received``; an outflow is debited in
``payment_currency`` at ``amount_paid``. On a cross-currency transfer these are different
numbers, and using either one for both directions would invent a conversion.

**Near-threshold detection is currency-restricted.** A transfer counts only when it is
denominated in ``FactConfig.threshold_currency``. Counting a 9,500 Rupee transfer against a
10,000 US Dollar threshold would require the rate the substrate does not carry. Transfers
in other currencies are not counted and not treated as evidence.
"""

from __future__ import annotations

from collections import defaultdict

from g2t_aml.facts.caseview import CaseEdge, CaseView
from g2t_aml.facts.config import FactConfig
from g2t_aml.facts.schema import FlowFacts, Money, PerCurrencyTotal, Unavailable

__all__ = [
    "FIELD_PRODUCERS",
    "MULTI_CURRENCY_REASON",
    "NO_TRANSFERS_REASON",
    "aggregate",
    "by_currency",
    "extract_flow",
    "n_near_threshold",
]

FIELD_PRODUCERS: dict[str, str] = {
    "flow.total_inflow": "flow.single_currency_aggregate",
    "flow.total_outflow": "flow.single_currency_aggregate",
    "flow.retained": "flow.inflow_minus_outflow",
    "flow.max_single_transfer": "flow.single_currency_maximum",
    "flow.inflow_by_currency": "flow.per_currency_breakdown",
    "flow.outflow_by_currency": "flow.per_currency_breakdown",
    "flow.n_transfers_near_threshold": "flow.threshold_band_count",
    "flow.threshold_reference": "flow.config_constant",
    "flow.threshold_currency": "flow.config_constant",
    "flow.threshold_band_fraction": "flow.config_constant",
    "flow.currencies_involved": "flow.currency_inventory",
    "flow.cross_border": "flow.jurisdiction_unavailable",
    "flow.cross_institution": "flow.distinct_bank_count",
    "flow.n_distinct_banks": "flow.distinct_bank_count",
    "flow.payment_formats": "flow.format_inventory",
}

#: Reason code for an aggregate that would require an exchange rate the substrate does
#: not carry.
MULTI_CURRENCY_REASON = "multi_currency_aggregate_undefined"

#: Reason code for an aggregate with nothing to aggregate.
NO_TRANSFERS_REASON = "no_transfers_in_this_direction"

#: Reason code for ``cross_border``, which no substrate can ever license. Constant rather
#: than inline so the permanence is visible at the definition site.
NO_JURISDICTION_REASON = "no_substrate_carries_jurisdiction"


def by_currency(pairs: list[tuple[float, str]]) -> tuple[PerCurrencyTotal, ...]:
    """Total a set of (amount, currency) transfers per currency.

    Args:
        pairs: Transfers as ``(amount, currency)``.

    Returns:
        One total per currency, sorted by currency name so serialisation is stable.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for amount, currency in pairs:
        totals[currency] += amount
        counts[currency] += 1
    return tuple(
        PerCurrencyTotal(currency=c, value=round(totals[c], 2), n_transfers=counts[c])
        for c in sorted(totals)
    )


def aggregate(pairs: list[tuple[float, str]]) -> Money | Unavailable:
    """Sum transfers, or refuse to when they span more than one currency.

    Args:
        pairs: Transfers as ``(amount, currency)``.

    Returns:
        A :class:`~g2t_aml.facts.schema.Money` when every transfer shares one currency,
        an :class:`~g2t_aml.facts.schema.Unavailable` otherwise — with
        :data:`NO_TRANSFERS_REASON` when there is nothing to sum, and
        :data:`MULTI_CURRENCY_REASON` when a sum would require a conversion rate.
    """
    if not pairs:
        return Unavailable(NO_TRANSFERS_REASON)
    currencies = {c for _, c in pairs}
    if len(currencies) > 1:
        return Unavailable(MULTI_CURRENCY_REASON)
    return Money(value=round(sum(a for a, _ in pairs), 2), currency=next(iter(currencies)))


def maximum(pairs: list[tuple[float, str]]) -> Money | Unavailable:
    """Return the largest single transfer, when the comparison is meaningful.

    A maximum across currencies is as undefined as a sum: without a rate, 3 Bitcoin and
    40,000 Rupee cannot be ordered.

    Args:
        pairs: Transfers as ``(amount, currency)``.

    Returns:
        The largest transfer, or a sentinel under the same rules as :func:`aggregate`.
    """
    if not pairs:
        return Unavailable(NO_TRANSFERS_REASON)
    currencies = {c for _, c in pairs}
    if len(currencies) > 1:
        return Unavailable(MULTI_CURRENCY_REASON)
    return Money(value=round(max(a for a, _ in pairs), 2), currency=next(iter(currencies)))


def retained(inflow: Money | Unavailable, outflow: Money | Unavailable) -> Money | Unavailable:
    """Compute what stayed behind, when both sides are comparable.

    Args:
        inflow: Total received by the focal entity.
        outflow: Total sent by it.

    Returns:
        ``inflow - outflow`` when both are available in the same currency. A sentinel
        when either is unavailable, when the currencies differ, or when the difference
        would be negative — which happens legitimately, because a case window is padded
        (D-019) and may catch an account dispersing funds it received before the window
        opened. Reporting a negative "retained" would invite a narrative to describe
        money that was never there, so the sentinel says so instead.
    """
    if isinstance(inflow, Unavailable) or isinstance(outflow, Unavailable):
        return Unavailable(MULTI_CURRENCY_REASON)
    if inflow.currency != outflow.currency:
        return Unavailable(MULTI_CURRENCY_REASON)
    difference = round(inflow.value - outflow.value, 2)
    if difference < 0:
        return Unavailable("outflow_exceeds_inflow_within_window")
    return Money(value=difference, currency=inflow.currency)


def n_near_threshold(view: CaseView, config: FactConfig) -> int:
    """Count transfers sitting just below the reporting threshold.

    Only transfers denominated in ``config.threshold_currency`` are counted; see the
    module docstring. The band is ``[threshold * (1 - band), threshold)`` — **strictly
    below** the threshold, because a transfer *at* the threshold is reportable and is
    therefore not evidence of structuring around it.

    Args:
        view: The case view.
        config: Supplies the threshold, its currency and the band width.

    Returns:
        The count of qualifying transfers.
    """
    floor, ceiling = config.threshold_floor, config.threshold_reference
    count = 0
    for edge in view.edges:
        if (
            edge.payment_currency == config.threshold_currency
            and edge.amount_paid is not None
            and floor <= edge.amount_paid < ceiling
        ):
            count += 1
    return count


def _inflow_pairs(view: CaseView, focal: str) -> list[tuple[float, str]]:
    """Return the focal entity's inbound transfers as (amount, currency).

    Args:
        view: The case view.
        focal: The focal entity.

    Returns:
        ``(amount_received, receiving_currency)`` per inbound transaction.
    """
    return [
        (e.amount_received, e.receiving_currency)
        for e in view.edges_into(focal)
        if e.amount_received is not None and e.receiving_currency is not None
    ]


def _outflow_pairs(view: CaseView, focal: str) -> list[tuple[float, str]]:
    """Return the focal entity's outbound transfers as (amount, currency).

    Args:
        view: The case view.
        focal: The focal entity.

    Returns:
        ``(amount_paid, payment_currency)`` per outbound transaction.
    """
    return [
        (e.amount_paid, e.payment_currency)
        for e in view.edges_out_of(focal)
        if e.amount_paid is not None and e.payment_currency is not None
    ]


def _all_transfer_pairs(edges: tuple[CaseEdge, ...]) -> list[tuple[float, str]]:
    """Return every transfer in the case as (amount_paid, payment_currency).

    Args:
        edges: The case's transactions.

    Returns:
        One pair per transaction carrying both values.
    """
    return [
        (e.amount_paid, e.payment_currency)
        for e in edges
        if e.amount_paid is not None and e.payment_currency is not None
    ]


def _bank_facts(view: CaseView) -> tuple[bool | Unavailable, int | Unavailable]:
    """Return institution facts, or sentinels when the substrate has no bank identity.

    Args:
        view: The case view.

    Returns:
        ``(cross_institution, n_distinct_banks)``.
    """
    if not view.availability.institution_identity or view.banks is None:
        sentinel = Unavailable("substrate_has_no_institution_identity")
        return sentinel, sentinel
    distinct = len({view.banks[n] for n in view.node_ids if n in view.banks})
    return distinct > 1, distinct


def extract_flow(view: CaseView, focal: str, config: FactConfig) -> FlowFacts | Unavailable:
    """Extract the whole flow block, or the sentinel that replaces it.

    Args:
        view: The case view.
        focal: The focal entity, whose perspective inflow and outflow are measured from.
        config: Supplies the near-threshold parameters.

    Returns:
        The populated :class:`~g2t_aml.facts.schema.FlowFacts`, or an
        :class:`~g2t_aml.facts.schema.Unavailable` when the substrate carries no monetary
        amounts. Never zeroed: Elliptic2's absence of amounts is not the same fact as a
        case in which nothing moved.
    """
    if not view.availability.monetary_amounts:
        return Unavailable("substrate_has_no_monetary_amounts")

    inflow_pairs = _inflow_pairs(view, focal)
    outflow_pairs = _outflow_pairs(view, focal)
    total_inflow = aggregate(inflow_pairs)
    total_outflow = aggregate(outflow_pairs)

    currencies: set[str] = set()
    formats: set[str] = set()
    for edge in view.edges:
        if edge.payment_currency is not None:
            currencies.add(edge.payment_currency)
        if edge.receiving_currency is not None:
            currencies.add(edge.receiving_currency)
        if edge.payment_format is not None:
            formats.add(edge.payment_format)

    cross_institution, n_distinct_banks = _bank_facts(view)

    return FlowFacts(
        total_inflow=total_inflow,
        total_outflow=total_outflow,
        retained=retained(total_inflow, total_outflow),
        max_single_transfer=maximum(_all_transfer_pairs(view.edges)),
        inflow_by_currency=by_currency(inflow_pairs),
        outflow_by_currency=by_currency(outflow_pairs),
        n_transfers_near_threshold=n_near_threshold(view, config),
        threshold_reference=config.threshold_reference,
        threshold_currency=config.threshold_currency,
        threshold_band_fraction=config.threshold_band_fraction,
        currencies_involved=tuple(sorted(currencies)),
        cross_border=Unavailable(NO_JURISDICTION_REASON),
        cross_institution=cross_institution,
        n_distinct_banks=n_distinct_banks,
        payment_formats=tuple(sorted(formats)),
    )

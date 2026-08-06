"""Counterparty ground truth: who is flagged, how close, and how much value came from them.

**Three-way, not two-way.** A counterparty is illicit, licit, or *unknown*, and the third
bucket is not padding. AMLworld's ground truth is complete, so ``n_unknown`` is zero there
— but a substrate with partial labels puts real counterparties in that bucket, and a
narrative that describes an unknown counterparty as licit is making a claim the data does
not support. The distinction is drawn on the label itself: ``is_laundering`` is ``None``
for unlabelled, ``False`` for labelled-licit, and the view preserves that difference
rather than filling nulls with False.

**Node labels are derived, not supplied.** Neither substrate labels accounts. AMLworld
labels transactions, so a node here counts as illicit iff it sits on at least one
laundering-flagged transaction *inside this case*. "Inside this case" matters: the same
account may be flagged elsewhere in the substrate, and a fact record describes the case it
was cut from, not the whole graph.

**Elliptic2 gets a sentinel for the whole block**, even though its availability mask sets
``node_labels=True``. That flag records a subgraph-level label, and a subgraph-level label
licenses no statement about any individual counterparty. The gate is therefore whether the
case carries per-transaction labels, not what the mask says about node labels — see
:attr:`~g2t_aml.facts.caseview.CaseView.has_labels`.
"""

from __future__ import annotations

from g2t_aml.facts.caseview import CaseView
from g2t_aml.facts.flow import MULTI_CURRENCY_REASON
from g2t_aml.facts.schema import LabelFacts, Unavailable

__all__ = [
    "FIELD_PRODUCERS",
    "classify_counterparties",
    "extract_labels",
    "illicit_nodes",
    "min_hops_to_illicit",
]

FIELD_PRODUCERS: dict[str, str] = {
    "labels.n_illicit_counterparties": "labels.counterparty_classification",
    "labels.n_licit_counterparties": "labels.counterparty_classification",
    "labels.n_unknown_counterparties": "labels.counterparty_classification",
    "labels.n_counterparties": "labels.counterparty_classification",
    "labels.min_hops_to_known_illicit": "labels.bfs_to_nearest_illicit",
    "labels.illicit_inflow_share": "labels.flagged_inflow_value_share",
    "labels.n_illicit_transactions": "labels.flagged_transaction_count",
    "labels.focal_is_illicit": "labels.node_label_derivation",
}


def illicit_nodes(view: CaseView) -> frozenset[str]:
    """Return every account sitting on a laundering-flagged transaction in this case.

    Args:
        view: The case view.

    Returns:
        Account identifiers. Both endpoints of a flagged transaction qualify, including
        the endpoints of a flagged self-loop.
    """
    flagged: set[str] = set()
    for edge in view.edges:
        if edge.is_laundering:
            flagged.add(edge.src)
            flagged.add(edge.dst)
    return frozenset(flagged)


def classify_counterparties(view: CaseView, focal: str) -> tuple[int, int, int]:
    """Split the focal entity's counterparties into illicit, licit and unknown.

    A counterparty is:

    - **illicit** if any transaction it shares with anyone in this case is flagged;
    - **unknown** if it is not illicit and at least one of its incident transactions
      carries no label at all;
    - **licit** otherwise.

    Args:
        view: The case view.
        focal: The focal entity.

    Returns:
        ``(n_illicit, n_licit, n_unknown)``.
    """
    flagged = illicit_nodes(view)
    unlabelled: set[str] = set()
    for edge in view.edges:
        if edge.is_laundering is None:
            unlabelled.add(edge.src)
            unlabelled.add(edge.dst)

    illicit = licit = unknown = 0
    for counterparty in sorted(view.neighbours(focal)):
        if counterparty in flagged:
            illicit += 1
        elif counterparty in unlabelled:
            unknown += 1
        else:
            licit += 1
    return illicit, licit, unknown


def min_hops_to_illicit(view: CaseView, focal: str) -> int | None:
    """Return the undirected hop distance to the nearest flagged account.

    Args:
        view: The case view.
        focal: The account to measure from.

    Returns:
        ``0`` when the focal entity is itself flagged, the hop count otherwise, and
        ``None`` when no flagged account is reachable inside the case. That ``None`` is a
        measured value — "there is none" — not an availability sentinel, and the checker
        treats a distance claim against it as CONTRADICTED rather than UNVERIFIABLE.
    """
    flagged = illicit_nodes(view)
    if not flagged:
        return None
    if focal in flagged:
        return 0
    distances = view.hops_from(focal)
    reachable = [d for node, d in distances.items() if node in flagged]
    return min(reachable) if reachable else None


def illicit_inflow_share(view: CaseView, focal: str) -> float | Unavailable:
    """Return the share of inbound *value* arriving on flagged transactions.

    Value rather than count, deliberately: "61% of inbound funds came from flagged
    counterparties" is a materially different — and more useful — claim than "61% of
    inbound transactions did", and value is the one an investigator acts on.

    Args:
        view: The case view.
        focal: The focal entity.

    Returns:
        A share in [0, 1], or an :class:`~g2t_aml.facts.schema.Unavailable` when amounts
        are absent, when inflows span multiple currencies (the ratio would need a
        conversion rate), or when there is no inbound value to take a share of.
    """
    if not view.availability.monetary_amounts:
        return Unavailable("substrate_has_no_monetary_amounts")

    inbound = [
        e
        for e in view.edges_into(focal)
        if e.amount_received is not None and e.receiving_currency is not None
    ]
    if not inbound:
        return Unavailable("no_inbound_value_to_take_a_share_of")
    if len({e.receiving_currency for e in inbound}) > 1:
        return Unavailable(MULTI_CURRENCY_REASON)

    total = sum(e.amount_received or 0.0 for e in inbound)
    if total <= 0:
        return Unavailable("no_inbound_value_to_take_a_share_of")
    flagged = sum(e.amount_received or 0.0 for e in inbound if e.is_laundering)
    return round(flagged / total, 6)


def extract_labels(view: CaseView, focal: str) -> LabelFacts | Unavailable:
    """Extract the whole labels block, or the sentinel that replaces it.

    Args:
        view: The case view.
        focal: The focal entity, whose counterparties are classified.

    Returns:
        The populated :class:`~g2t_aml.facts.schema.LabelFacts`, or an
        :class:`~g2t_aml.facts.schema.Unavailable` when the case carries no
        per-transaction illicit labels. See the module docstring for why the gate is the
        transaction labels rather than ``availability.node_labels``.
    """
    if not view.has_labels:
        return Unavailable("substrate_has_no_per_transaction_illicit_labels")

    illicit, licit, unknown = classify_counterparties(view, focal)
    return LabelFacts(
        n_illicit_counterparties=illicit,
        n_licit_counterparties=licit,
        n_unknown_counterparties=unknown,
        n_counterparties=len(view.neighbours(focal)),
        min_hops_to_known_illicit=min_hops_to_illicit(view, focal),
        illicit_inflow_share=illicit_inflow_share(view, focal),
        n_illicit_transactions=sum(1 for e in view.edges if e.is_laundering),
        focal_is_illicit=focal in illicit_nodes(view),
    )

"""A trivially faithful narrative probe, for the round-trip gate.

The gate is: extract facts → render a narrative that only says true things → run the
checker → assert 100% SUPPORTED and zero CONTRADICTED. Any failure is a bug in the
extractor, in the checker, or in their disagreement about what a field *means*.

**Why this lives in tests/ and not in src/.** It is not a narrative template — Phase 4
owns those — and it must never become one. It exists solely to exercise the extractor and
checker against each other.

**Why it does not merely echo the record back.** A probe that emitted each field's exact
value as a claim would test only that `x == x`, and would pass even if every duration were
computed in seconds and checked in hours. So the probe deliberately *re-expresses* each
value the way a narrative would:

- durations are stated in days when they run long, at day granularity
- amounts are rounded to a readable figure, inside the 1% tolerance
- counts are stated exactly, because the tolerance policy allows nothing else
- qualitative descriptors are emitted only when their binding actually holds
- masked fields are skipped entirely, which is what a mask-respecting generator does

That is what makes the round trip informative: it passes only if the two directions agree
on units, on rounding, on availability semantics, and on what each field path denotes.
"""

from __future__ import annotations

from dataclasses import dataclass

from g2t_aml.facts.checkers import CheckContext, Claim, ClaimType, DurationClaim
from g2t_aml.facts.schema import CaseFacts, Money, Unavailable, is_available
from g2t_aml.facts.vocab import ControlledVocabulary


@dataclass
class Probe:
    """A rendered narrative and the claims it makes."""

    text: str
    claims: list[Claim]


class _Builder:
    """Accumulates sentences and the claims they carry, keeping spans in sync."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.claims: list[Claim] = []
        self.length = 0

    def say(
        self,
        sentence: str,
        *,
        path: str | None,
        kind: ClaimType,
        value: object,
    ) -> None:
        """Append a sentence and record the claim it makes, with real character spans."""
        start = self.length
        self.parts.append(sentence)
        self.length += len(sentence) + 1
        self.claims.append(
            Claim(
                text_span=(start, start + len(sentence)),
                field_path=path,
                claim_type=kind,
                value=value,
                raw_text=sentence,
            )
        )

    def build(self) -> Probe:
        return Probe(text=" ".join(self.parts), claims=self.claims)


def _round_money(amount: Money) -> float:
    """Round an amount the way a narrative would, staying inside the 1% tolerance.

    Rounds to three significant figures for large amounts and leaves small ones alone,
    which is what "approximately USD 482,000" looks like. Deliberately NOT the exact
    value: an echo would not exercise the tolerance at all.
    """
    value = amount.value
    if value >= 1000:
        magnitude = 10 ** (len(str(int(value))) - 3)
        return float(round(value / magnitude) * magnitude)
    return value


def _duration_claim(hours: float) -> DurationClaim:
    """State a duration at the granularity a narrative would choose."""
    if hours >= 48:
        return DurationClaim(value=round(hours / 24), unit="days")
    if hours >= 1:
        return DurationClaim(value=round(hours, 1), unit="hours")
    return DurationClaim(value=round(hours * 60), unit="minutes")


def render_probe(  # noqa: PLR0912, PLR0915 -- one linear block per fact family; splitting
    # it would scatter the availability guards that make the probe mask-respecting.
    facts: CaseFacts,
    vocabulary: ControlledVocabulary,
) -> Probe:
    """Render a narrative asserting only what the record supports.

    Args:
        facts: The record to describe.
        vocabulary: The controlled vocabulary, used to emit only descriptors that hold.

    Returns:
        The narrative and its claims.
    """
    b = _Builder()
    structure = facts.structure
    focal = facts.focal_entity

    # --- structure: always available on every substrate.
    b.say(
        f"The case subgraph contains {structure.n_nodes} accounts.",
        path="structure.n_nodes",
        kind=ClaimType.NUMERIC,
        value=structure.n_nodes,
    )
    b.say(
        f"It contains {structure.n_edges} transactions.",
        path="structure.n_edges",
        kind=ClaimType.NUMERIC,
        value=structure.n_edges,
    )
    b.say(
        f"They fall into {structure.n_components} connected component(s).",
        path="structure.n_components",
        kind=ClaimType.NUMERIC,
        value=structure.n_components,
    )
    b.say(
        f"The subgraph density is {structure.density}.",
        path="structure.density",
        kind=ClaimType.NUMERIC,
        value=structure.density,
    )
    if structure.diameter is not None:
        b.say(
            f"Its diameter is {structure.diameter} hops.",
            path="structure.diameter",
            kind=ClaimType.NUMERIC,
            value=structure.diameter,
        )

    # --- focal entity.
    b.say(
        f"The account under review is {focal.id}.",
        path=None,
        kind=ClaimType.ENTITY,
        value=focal.id,
    )
    b.say(
        f"It acts as {focal.role} within the case.",
        path="focal_entity.role",
        kind=ClaimType.CATEGORICAL,
        value=focal.role,
    )
    b.say(
        f"It received from {focal.in_degree} distinct counterparties.",
        path="focal_entity.in_degree",
        kind=ClaimType.NUMERIC,
        value=focal.in_degree,
    )
    b.say(
        f"It sent to {focal.out_degree} distinct counterparties.",
        path="focal_entity.out_degree",
        kind=ClaimType.NUMERIC,
        value=focal.out_degree,
    )

    # --- temporal, only when the substrate has a clock.
    temporal = facts.temporal
    if is_available(temporal):
        duration = _duration_claim(temporal.span_hours)
        b.say(
            f"Activity spanned approximately {duration.value} {duration.unit}.",
            path="temporal.span_hours",
            kind=ClaimType.TEMPORAL,
            value=duration,
        )
        b.say(
            f"The case covers {temporal.n_transactions} timestamped transactions.",
            path="temporal.n_transactions",
            kind=ClaimType.NUMERIC,
            value=temporal.n_transactions,
        )
        b.say(
            "A burst of concentrated activity was detected."
            if temporal.burst_detected
            else "No burst of concentrated activity was detected.",
            path="temporal.burst_detected",
            kind=ClaimType.CATEGORICAL,
            value=temporal.burst_detected,
        )
        if temporal.burst_detected and temporal.burst_txn_count is not None:
            b.say(
                f"That burst held {temporal.burst_txn_count} transactions.",
                path="temporal.burst_txn_count",
                kind=ClaimType.NUMERIC,
                value=temporal.burst_txn_count,
            )
        if temporal.event_ordering:
            b.say(
                "The observed phase ordering was " + " then ".join(temporal.event_ordering) + ".",
                path="temporal.event_ordering",
                kind=ClaimType.TEMPORAL,
                value=list(temporal.event_ordering),
            )

    # --- flow, only what is denominated in a single currency.
    flow = facts.flow
    if is_available(flow):
        for path, amount in (
            ("flow.total_inflow", flow.total_inflow),
            ("flow.total_outflow", flow.total_outflow),
            ("flow.max_single_transfer", flow.max_single_transfer),
        ):
            if isinstance(amount, Money):
                stated = _round_money(amount)
                b.say(
                    f"The {path.split('.')[-1].replace('_', ' ')} was approximately "
                    f"{stated:,.2f} {amount.currency}.",
                    path=path,
                    kind=ClaimType.NUMERIC,
                    value=(stated, amount.currency),
                )
        b.say(
            f"{flow.n_transfers_near_threshold} transfers fell just below the reporting "
            "threshold.",
            path="flow.n_transfers_near_threshold",
            kind=ClaimType.NUMERIC,
            value=flow.n_transfers_near_threshold,
        )
        if not isinstance(flow.n_distinct_banks, Unavailable):
            b.say(
                f"The accounts sit across {flow.n_distinct_banks} institution(s).",
                path="flow.n_distinct_banks",
                kind=ClaimType.NUMERIC,
                value=flow.n_distinct_banks,
            )
        for currency in flow.currencies_involved:
            b.say(
                f"Transactions were denominated in {currency}.",
                path="flow.currencies_involved",
                kind=ClaimType.CATEGORICAL,
                value=currency,
            )

    # --- labels, only when per-transaction ground truth exists.
    labels = facts.labels
    if is_available(labels):
        b.say(
            f"{labels.n_illicit_counterparties} counterparties are associated with "
            "flagged transactions.",
            path="labels.n_illicit_counterparties",
            kind=ClaimType.NUMERIC,
            value=labels.n_illicit_counterparties,
        )
        b.say(
            f"The focal account has {labels.n_counterparties} counterparties in total.",
            path="labels.n_counterparties",
            kind=ClaimType.NUMERIC,
            value=labels.n_counterparties,
        )
        if labels.min_hops_to_known_illicit is not None:
            b.say(
                f"The nearest flagged account is {labels.min_hops_to_known_illicit} "
                "hop(s) away.",
                path="labels.min_hops_to_known_illicit",
                kind=ClaimType.NUMERIC,
                value=labels.min_hops_to_known_illicit,
            )

    # --- motifs, presence and every non-null descriptor.
    for name, motif in facts.motifs.as_mapping().items():
        b.say(
            f"A {name.replace('_', ' ')} structure was "
            + ("detected." if motif.present else "not detected."),
            path=f"motifs.{name}.present",
            kind=ClaimType.CATEGORICAL,
            value=motif.present,
        )
        for descriptor, value in motif.descriptors.items():
            if value is None or isinstance(value, list):
                continue
            path = f"motifs.{name}.{descriptor}"
            if descriptor in {"hub", "origin", "destination"}:
                b.say(
                    f"The {name} {descriptor} is {value}.",
                    path=None,
                    kind=ClaimType.ENTITY,
                    value=value,
                )
            elif descriptor == "window_hours":
                duration = _duration_claim(float(value))
                b.say(
                    f"The {name} formed over about {duration.value} {duration.unit}.",
                    path=path,
                    kind=ClaimType.TEMPORAL,
                    value=duration,
                )
            else:
                b.say(
                    f"Its {descriptor} is {value}.",
                    path=path,
                    kind=ClaimType.NUMERIC,
                    value=value,
                )

    # --- qualitative descriptors: emitted ONLY where the binding actually holds. This is
    # what a well-behaved generator does, and it exercises the binding table in the
    # direction that must never produce a CONTRADICTED.
    mask = facts.availability.to_dict()
    for descriptor in vocabulary.risk_descriptors.values():
        if any(not mask.get(flag, False) for flag in descriptor.requires):
            continue
        value = _resolve(facts, descriptor.binds_to)
        if value is None or isinstance(value, Unavailable | bool):
            continue
        if not isinstance(value, int | float):
            continue
        if descriptor.holds_for(value):
            phrase = descriptor.phrase_variants[0]
            b.say(
                f"The activity shows {phrase}.",
                path=None,
                kind=ClaimType.QUALITATIVE,
                value=phrase,
            )

    # --- typology: asserted only when it is ground truth. An inferred label is this
    # system's own detector talking, and asserting it would be UNVERIFIABLE by design.
    if facts.typology.source == "ground_truth":
        b.say(
            f"The case belongs to a {facts.typology.label} stream.",
            path="typology.label",
            kind=ClaimType.CATEGORICAL,
            value=facts.typology.label,
        )

    return b.build()


def _resolve(facts: CaseFacts, path: str) -> object:
    """Resolve a dotted field path, mirroring the checker's own resolution."""
    from g2t_aml.facts.salience import field_value

    return field_value(facts, path)


def run_probe(facts: CaseFacts, vocabulary: ControlledVocabulary) -> list:
    """Render the probe and check every claim it makes."""
    from g2t_aml.facts.checkers import check_claim

    probe = render_probe(facts, vocabulary)
    ctx = CheckContext(facts=facts, vocabulary=vocabulary)
    return [check_claim(c, ctx) for c in probe.claims]

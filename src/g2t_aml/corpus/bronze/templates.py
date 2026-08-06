"""Template families: the four-part SAR structure, keyed to typology, substrate-aware.

Every Bronze narrative has the same four parts, because every SAR does::

    [1] SUBJECT & SCOPE     entities, accounts, time window, total volume
    [2] ACTIVITY OBSERVED   the factual transaction sequence
    [3] PATTERN & TYPOLOGY  what structure this matches, hedged appropriately
    [4] BASIS & ACTION      why suspicious, recommended next step

**Twelve families, five structural realisations each, 1,080 distinct narratives each.**
One per AMLworld typology, plus
``unclassified_suspicious`` (flagged activity with no ground-truth typology),
``no_finding`` (licit, and it is a *statement* rather than an absence — see D-035),
``topology_only`` (Elliptic2: no amounts, no clock, no per-transaction labels) and
``minimal_activity`` (a subgraph with a single counterparty, where a pattern claim would
be vacuous).

**Why more than one realisation.** Bronze is epoch-1 fine-tuning data. A template pack with
one surface form per typology teaches the generator that a fan-out case has exactly one
correct sentence, and the model will reproduce it — the rigidity would show up in the paper
as a diversity collapse that no amount of Silver could undo.

Sections 1, 2 and 4 draw from shared pools at a family-specific offset, so two families
rarely open the same way; section 3 is written separately per family, which is what keeps
the families *distinguishable* under the inter-family n-gram overlap measured by
:mod:`g2t_aml.corpus.diversity`. **The four sections are composed independently**, so a
family offers 6 x 6 x 5 x 6 = 1,080 whole narratives rather than five. That was not the
first design, and the reason it is the design now is measured: locking the sections
together gave a corpus self-BLEU of 0.81, which is collapse. See D-042.

**Substrate awareness is structural, not advisory.** A family declares the availability
flags it needs; the renderer refuses to run it against a record whose mask lacks one, and
refuses again at each individual slot. ``topology_only`` is the only family whose pools
contain no monetary and no temporal slot at all, which is why it is the only family
Elliptic2 can render. See :mod:`g2t_aml.corpus.bronze.renderer` for the two-layer guard
and why a hard error is the correct response rather than a skipped sentence.

**Slot syntax.** A segment is a sentence carrying placeholders:

``{flow.total_outflow:money}``
    A fact slot. Emits a :class:`~g2t_aml.corpus.record.SlotAnnotation` and, downstream, a
    checkable claim. The kind selects the formatter and the tolerance rule.
``{temporal.burst_detected:bool:detected|not detected}``
    A boolean slot with its two surface forms, mapped back exactly on parse.
``{~rapid_dispersal}``
    Every controlled risk descriptor whose binding *holds* for this record, rendered as a
    list. Emits QUALITATIVE claims. A descriptor whose condition fails is never written,
    which is what keeps Bronze free of CONTRADICTED claims by construction.
``{_p:labels.n_counterparties:counterparty:counterparties}``
    Plain text, no claim: number agreement only.
``{_threshold_reference}``
    A whitelisted regulatory citation, checked against the vocabulary's whitelist.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FAMILIES",
    "FAMILY_FOR_TYPOLOGY",
    "MINIMAL_FAMILY",
    "PHASE_DISPLAY",
    "ROLE_DISPLAY",
    "SALIENCE_SENTENCES",
    "TOPOLOGY_FAMILY",
    "TYPOLOGY_DISPLAY",
    "UNCLASSIFIED_SUSPICIOUS_FAMILY",
    "Family",
    "Segment",
    "Variant",
    "family_for",
    "n_variants",
]

#: Surface forms for the phase vocabulary. Bijective: the parser maps back exactly, so
#: ``temporal.event_ordering`` stays a checkable ordered claim rather than prose.
PHASE_DISPLAY: dict[str, str] = {
    "inflow_phase": "inbound activity",
    "consolidation": "a consolidation interval",
    "outflow_phase": "outbound activity",
    "interleaved": "interleaved inbound and outbound activity",
}

#: Surface forms for the typology vocabulary. Bijective.
TYPOLOGY_DISPLAY: dict[str, str] = {
    "fan_out": "fan-out",
    "fan_in": "fan-in",
    "gather_scatter": "gather-scatter",
    "scatter_gather": "scatter-gather",
    "cycle": "cycle",
    "random": "random",
    "bipartite": "bipartite",
    "stack": "stack",
    "unclassified": "unclassified",
}

#: Surface forms for each entity role. **Every one of these is a `phrase_variants` entry
#: in schemas/vocab_v1.yaml**, and ``tests/unit/test_bronze_templates.py`` asserts it: the
#: renderer may not invent a way of naming a role that the controlled vocabulary has not
#: licensed. Several forms per role, so the same role does not read identically across the
#: corpus.
ROLE_DISPLAY: dict[str, tuple[str, ...]] = {
    "originator": ("the originating account", "the sending account"),
    "beneficiary": ("the receiving account", "the destination account"),
    "pass_through": ("a pass-through account", "a conduit account"),
    "hub": ("a hub account", "a central account"),
    "intermediary": ("an intermediary", "an intermediate account"),
    "terminal": ("an isolated account", "an account with no counterparties in scope"),
}


@dataclass(frozen=True)
class Segment:
    """One sentence of a narrative, with its placeholders.

    Attributes:
        text: The sentence, carrying the placeholders documented in the module docstring.
        required: When True, a segment whose slots cannot all be filled makes the whole
            variant inapplicable. When False — the default — the segment is simply
            dropped, which is how a record with no inbound value loses its inflow
            sentence without losing its narrative. A required segment is also immune to
            length trimming: it carries the subject, the scope or the typology, and a
            report without those is not short, it is incomplete.
        optional_detail: Marks a segment the length controller should drop *first*. Any
            non-required segment may be trimmed, but these are the ones chosen at
            authoring time as the least missed. A segment carrying a field on the case's
            salience list is never dropped whatever this says — that guard lives in the
            renderer, where the salience list is known.
    """

    text: str
    required: bool = False
    optional_detail: bool = False


@dataclass(frozen=True)
class Variant:
    """One surface realisation: four sections, each a sequence of segments.

    Attributes:
        subject: Section 1 — subject and scope.
        activity: Section 2 — the factual transaction sequence.
        pattern: Section 3 — structure and typology.
        basis: Section 4 — basis for suspicion and recommended action.
    """

    subject: tuple[Segment, ...]
    activity: tuple[Segment, ...]
    pattern: tuple[Segment, ...]
    basis: tuple[Segment, ...]


@dataclass(frozen=True)
class Family:
    """A template family: the narratives available for one typology on one substrate.

    **The four sections are chosen independently.** An early draft locked them together —
    variant *i* meant subject *i*, activity *i*, pattern *i*, basis *i* — which gave five
    realisations per family and a corpus self-BLEU of 0.81. Since the sections are
    grammatically independent, composing them independently costs nothing in authoring
    and multiplies the realisation count: six subjects x six activities x five patterns x
    six bases is 1,080 distinct narratives per family rather than five. See
    :meth:`realisation` for the encoding, and D-042 for the measured effect.

    Attributes:
        key: Family name, written into ``generator.family``.
        typologies: Typology labels this family serves.
        requires_mask: Availability flags every record must carry. Checked before a
            single slot is filled, so an amount-bearing family fails loudly on Elliptic2
            rather than quietly rendering a shorter narrative.
        pattern_sections: The family's own section-3 phrasings, one per structural
            realisation. This is the section that must differ between families.
        subject_pool: Section-1 alternatives this family draws on.
        activity_pool: Section-2 alternatives.
        basis_pool: Section-4 alternatives.
        offset: Where this family enters the shared pools.
        description: What this family is for, quoted into the phase log.
    """

    key: str
    typologies: tuple[str, ...]
    requires_mask: tuple[str, ...]
    pattern_sections: tuple[tuple[Segment, ...], ...]
    subject_pool: tuple[tuple[Segment, ...], ...]
    activity_pool: tuple[tuple[Segment, ...], ...]
    basis_pool: tuple[tuple[Segment, ...], ...]
    offset: int
    description: str

    @property
    def n_realisations(self) -> int:
        """Return how many distinct narratives this family can produce.

        Returns:
            The product of the four section counts.
        """
        return (
            len(self.pattern_sections)
            * len(self.subject_pool)
            * len(self.activity_pool)
            * len(self.basis_pool)
        )

    @property
    def n_surface_variants(self) -> int:
        """Return the family's structural realisation count, for the acceptance criteria.

        The brief asks for four to six surface realisations per family. That is the
        pattern-section count: the number of genuinely different ways this family
        describes *its own structure*. The larger :attr:`n_realisations` counts whole
        narratives and is the number that governs corpus diversity.

        Returns:
            The pattern-section count.
        """
        return len(self.pattern_sections)

    def realisation(self, index: int) -> Variant:
        """Decode a realisation index into one composed variant.

        The index is a mixed-radix encoding of the four section choices, so it both
        identifies the narrative exactly and is a single integer the training record can
        carry in ``generator.variant``.

        Args:
            index: A realisation index. Taken modulo :attr:`n_realisations`, so any
                integer is valid and the mapping stays total.

        Returns:
            The composed variant.
        """
        remaining = index % self.n_realisations
        pattern = remaining % len(self.pattern_sections)
        remaining //= len(self.pattern_sections)
        subject = remaining % len(self.subject_pool)
        remaining //= len(self.subject_pool)
        activity = remaining % len(self.activity_pool)
        remaining //= len(self.activity_pool)
        basis = remaining % len(self.basis_pool)
        return Variant(
            subject=self.subject_pool[(subject + self.offset) % len(self.subject_pool)],
            activity=self.activity_pool[(activity + self.offset) % len(self.activity_pool)],
            pattern=self.pattern_sections[pattern],
            basis=self.basis_pool[(basis + self.offset) % len(self.basis_pool)],
        )


# --------------------------------------------------------------- shared pools ---
# Sections 1, 2 and 4 are substrate-general: what changes between a fan-out case and a
# cycle case is the *pattern* section, not how the subject is introduced. Each family
# enters these pools at its own offset so the corpus does not open every narrative the
# same way.

_SUBJECT_POOL: tuple[tuple[Segment, ...], ...] = (
    (
        Segment(
            "This report concerns account {focal_entity.id:entity}, which acts as "
            "{focal_entity.role:role} within the reviewed subgraph.",
            required=True,
        ),
        Segment(
            "The subgraph comprises {structure.n_nodes:count} accounts connected by "
            "{structure.n_edges:count} transactions.",
            required=True,
        ),
        Segment(
            "Activity was observed from {temporal.window_start:timestamp} to "
            "{temporal.window_end:timestamp}, a span of {temporal.span_hours:duration}."
        ),
        Segment(
            "Inbound value to the subject account totalled approximately "
            "{flow.total_inflow:money}."
        ),
        Segment(
            "Outbound value from the subject account totalled approximately "
            "{flow.total_outflow:money}."
        ),
        Segment(
            "The accounts in scope are held at {flow.n_distinct_banks:count} distinct "
            "{_p:flow.n_distinct_banks:institution:institutions}.",
        ),
    ),
    (
        Segment(
            "The subject of this report is account {focal_entity.id:entity}.",
            required=True,
        ),
        Segment(
            "Within the reviewed subgraph it occupies the position of "
            "{focal_entity.role:role}, among {structure.n_nodes:count} accounts and "
            "{structure.n_edges:count} recorded transactions.",
            required=True,
        ),
        Segment(
            "The review window runs from {temporal.window_start:timestamp} to "
            "{temporal.window_end:timestamp}, covering {temporal.span_hours:duration} of "
            "observed activity."
        ),
        Segment("Over that window the account received approximately {flow.total_inflow:money}."),
        Segment("It sent approximately {flow.total_outflow:money} onward."),
        Segment(
            "{flow.n_distinct_banks:count} distinct "
            "{_p:flow.n_distinct_banks:institution:institutions} are represented among "
            "the accounts in scope.",
        ),
    ),
    (
        Segment(
            "Account {focal_entity.id:entity} is the subject of this referral, appearing "
            "as {focal_entity.role:role} in the activity described below.",
            required=True,
        ),
        Segment(
            "The subgraph under review holds {structure.n_nodes:count} accounts and "
            "{structure.n_edges:count} transactions between them.",
            required=True,
        ),
        Segment(
            "The observed activity extends over {temporal.span_hours:duration}, beginning "
            "{temporal.window_start:timestamp} and ending {temporal.window_end:timestamp}."
        ),
        Segment(
            "Value received by the subject account was approximately {flow.total_inflow:money}."
        ),
        Segment(
            "Value dispersed by the subject account was approximately {flow.total_outflow:money}."
        ),
        Segment(
            "The counterparty accounts are distributed across "
            "{flow.n_distinct_banks:count} {_p:flow.n_distinct_banks:institution:institutions}.",
        ),
    ),
    (
        Segment(
            "This referral describes activity centred on account "
            "{focal_entity.id:entity}, identified as {focal_entity.role:role}.",
            required=True,
        ),
        Segment(
            "In scope are {structure.n_nodes:count} accounts and {structure.n_edges:count} "
            "transactions.",
            required=True,
        ),
        Segment(
            "The transactions fall between {temporal.window_start:timestamp} and "
            "{temporal.window_end:timestamp}, a period of {temporal.span_hours:duration}."
        ),
        Segment("Aggregate inflow to the subject account was around {flow.total_inflow:money}."),
        Segment(
            "Aggregate outflow from the subject account was around {flow.total_outflow:money}."
        ),
        Segment(
            "Accounts in scope sit at {flow.n_distinct_banks:count} separate "
            "{_p:flow.n_distinct_banks:institution:institutions}.",
        ),
    ),
    (
        Segment(
            "Account {focal_entity.id:entity} is reported here as "
            "{focal_entity.role:role} of the reviewed activity.",
            required=True,
        ),
        Segment(
            "The reviewed subgraph contains {structure.n_nodes:count} accounts, linked by "
            "{structure.n_edges:count} transactions.",
            required=True,
        ),
        Segment(
            "Activity runs {temporal.span_hours:duration} in total, from "
            "{temporal.window_start:timestamp} through {temporal.window_end:timestamp}."
        ),
        Segment("The subject account took in approximately {flow.total_inflow:money}."),
        Segment("The subject account paid out approximately {flow.total_outflow:money}."),
        Segment(
            "Holdings span {flow.n_distinct_banks:count} "
            "{_p:flow.n_distinct_banks:institution:institutions}.",
        ),
    ),
    (
        Segment(
            "The account under review is {focal_entity.id:entity}, which the subgraph "
            "places as {focal_entity.role:role}.",
            required=True,
        ),
        Segment(
            "It sits among {structure.n_nodes:count} accounts joined by "
            "{structure.n_edges:count} transactions.",
            required=True,
        ),
        Segment(
            "Observed activity spans {temporal.span_hours:duration}, opening "
            "{temporal.window_start:timestamp} and closing "
            "{temporal.window_end:timestamp}."
        ),
        Segment("Inflow across the window came to roughly {flow.total_inflow:money}."),
        Segment("Outflow across the window came to roughly {flow.total_outflow:money}."),
        Segment(
            "{flow.n_distinct_banks:count} {_p:flow.n_distinct_banks:institution:institutions} "
            "hold the accounts in scope.",
        ),
    ),
)

_ACTIVITY_POOL: tuple[tuple[Segment, ...], ...] = (
    (
        Segment(
            "The subject account received from {focal_entity.in_degree:count} distinct "
            "{_p:focal_entity.in_degree:counterparty:counterparties} across "
            "{focal_entity.n_transactions_in:count} inbound "
            "{_p:focal_entity.n_transactions_in:transaction:transactions}, and sent to "
            "{focal_entity.out_degree:count} distinct "
            "{_p:focal_entity.out_degree:counterparty:counterparties} across "
            "{focal_entity.n_transactions_out:count} outbound "
            "{_p:focal_entity.n_transactions_out:transaction:transactions}.",
            required=True,
        ),
        Segment("The sequence of activity was {temporal.event_ordering:ordering}."),
        Segment("The largest single transfer in scope was {flow.max_single_transfer:money}."),
        Segment(
            "The densest cluster of activity held {temporal.burst_txn_count:count} "
            "transactions within {temporal.burst_window_hours:duration}, beginning "
            "{temporal.burst_start:timestamp}."
        ),
        Segment(
            "Of {labels.n_counterparties:count} "
            "{_p:labels.n_counterparties:counterparty:counterparties} in scope, "
            "{labels.n_illicit_counterparties:count} are associated with transactions "
            "flagged in the source data.",
        ),
        Segment(
            "{labels.n_illicit_transactions:count} transactions in the subgraph carry a "
            "flag in the source data."
        ),
        Segment(
            "{flow.n_transfers_near_threshold:count} transfers fall in the band "
            "immediately below {_threshold_reference}."
        ),
        Segment(
            "Settlement used {flow.payment_formats:set}.",
        ),
    ),
    (
        Segment(
            "Inbound activity reached the subject account from "
            "{focal_entity.in_degree:count} distinct "
            "{_p:focal_entity.in_degree:counterparty:counterparties} over "
            "{focal_entity.n_transactions_in:count} "
            "{_p:focal_entity.n_transactions_in:transaction:transactions}; outbound "
            "activity left it for {focal_entity.out_degree:count} distinct "
            "{_p:focal_entity.out_degree:counterparty:counterparties} over "
            "{focal_entity.n_transactions_out:count} "
            "{_p:focal_entity.n_transactions_out:transaction:transactions}.",
            required=True,
        ),
        Segment("Phases ran in the order {temporal.event_ordering:ordering}."),
        Segment("No single transfer exceeded {flow.max_single_transfer:money}."),
        Segment(
            "Activity concentrated into {temporal.burst_txn_count:count} transactions "
            "inside {temporal.burst_window_hours:duration} from "
            "{temporal.burst_start:timestamp}."
        ),
        Segment(
            "{labels.n_illicit_counterparties:count} of the "
            "{labels.n_counterparties:count} "
            "{_p:labels.n_counterparties:counterparty:counterparties} appear on "
            "transactions the source data flags."
        ),
        Segment("The subgraph carries {labels.n_illicit_transactions:count} flagged transactions."),
        Segment(
            "Transfers falling just under {_threshold_reference} number "
            "{flow.n_transfers_near_threshold:count}."
        ),
        Segment(
            "Payment channels observed were {flow.payment_formats:set}.",
        ),
    ),
    (
        Segment(
            "Counterparty structure around the subject account is "
            "{focal_entity.in_degree:count} inbound and "
            "{focal_entity.out_degree:count} outbound distinct "
            "{_p:focal_entity.out_degree:counterparty:counterparties}, carried by "
            "{focal_entity.n_transactions_in:count} inbound and "
            "{focal_entity.n_transactions_out:count} outbound transactions.",
            required=True,
        ),
        Segment("Observed phase sequence: {temporal.event_ordering:ordering}."),
        Segment("The single largest movement of value was {flow.max_single_transfer:money}."),
        Segment(
            "A cluster of {temporal.burst_txn_count:count} transactions occurred within "
            "{temporal.burst_window_hours:duration}, opening "
            "{temporal.burst_start:timestamp}."
        ),
        Segment(
            "Source-data flags attach to {labels.n_illicit_counterparties:count} of the "
            "{labels.n_counterparties:count} "
            "{_p:labels.n_counterparties:counterparty:counterparties}."
        ),
        Segment(
            "Flagged transactions within the subgraph total "
            "{labels.n_illicit_transactions:count}."
        ),
        Segment(
            "{flow.n_transfers_near_threshold:count} transfers sit just below "
            "{_threshold_reference}."
        ),
        Segment(
            "The transactions settled through {flow.payment_formats:set}.",
        ),
    ),
    (
        Segment(
            "The account drew on {focal_entity.in_degree:count} distinct inbound "
            "{_p:focal_entity.in_degree:counterparty:counterparties} and paid "
            "{focal_entity.out_degree:count} distinct outbound "
            "{_p:focal_entity.out_degree:counterparty:counterparties}, across "
            "{focal_entity.n_transactions_in:count} and "
            "{focal_entity.n_transactions_out:count} transactions respectively.",
            required=True,
        ),
        Segment("Activity proceeded as {temporal.event_ordering:ordering}."),
        Segment("The maximum single transfer recorded was {flow.max_single_transfer:money}."),
        Segment(
            "{temporal.burst_txn_count:count} transactions fell inside a "
            "{temporal.burst_window_hours:duration} cluster starting "
            "{temporal.burst_start:timestamp}."
        ),
        Segment(
            "Among {labels.n_counterparties:count} "
            "{_p:labels.n_counterparties:counterparty:counterparties}, "
            "{labels.n_illicit_counterparties:count} are linked to flagged transactions."
        ),
        Segment(
            "The subgraph includes {labels.n_illicit_transactions:count} transactions "
            "flagged at source."
        ),
        Segment(
            "Transfers immediately below {_threshold_reference} number "
            "{flow.n_transfers_near_threshold:count}."
        ),
        Segment(
            "Settlement channels in use were {flow.payment_formats:set}.",
        ),
    ),
    (
        Segment(
            "Distinct counterparties number {focal_entity.in_degree:count} on the inbound "
            "side and {focal_entity.out_degree:count} on the outbound side, over "
            "{focal_entity.n_transactions_in:count} inbound and "
            "{focal_entity.n_transactions_out:count} outbound transactions.",
            required=True,
        ),
        Segment("The phases observed were {temporal.event_ordering:ordering}."),
        Segment("The largest transfer observed reached {flow.max_single_transfer:money}."),
        Segment(
            "The tightest cluster of activity was {temporal.burst_txn_count:count} "
            "transactions in {temporal.burst_window_hours:duration}, from "
            "{temporal.burst_start:timestamp}."
        ),
        Segment(
            "{labels.n_illicit_counterparties:count} "
            "{_p:labels.n_illicit_counterparties:counterparty:counterparties} of the "
            "{labels.n_counterparties:count} in scope carry source-data flags."
        ),
        Segment(
            "Transactions flagged in the source data number "
            "{labels.n_illicit_transactions:count}."
        ),
        Segment(
            "A further {flow.n_transfers_near_threshold:count} transfers fall just short "
            "of {_threshold_reference}."
        ),
        Segment(
            "Observed payment formats were {flow.payment_formats:set}.",
        ),
    ),
    (
        Segment(
            "The subject account transacted with {focal_entity.in_degree:count} distinct "
            "senders and {focal_entity.out_degree:count} distinct recipients, over "
            "{focal_entity.n_transactions_in:count} inbound and "
            "{focal_entity.n_transactions_out:count} outbound transactions.",
            required=True,
        ),
        Segment("The order of phases was {temporal.event_ordering:ordering}."),
        Segment("Peak single-transfer value was {flow.max_single_transfer:money}."),
        Segment(
            "Within the window, {temporal.burst_txn_count:count} transactions fell inside "
            "{temporal.burst_window_hours:duration}, beginning "
            "{temporal.burst_start:timestamp}."
        ),
        Segment(
            "{labels.n_counterparties:count} "
            "{_p:labels.n_counterparties:counterparty:counterparties} were involved, of "
            "which {labels.n_illicit_counterparties:count} appear on flagged "
            "transactions."
        ),
        Segment(
            "Flags in the source data attach to {labels.n_illicit_transactions:count} "
            "transactions here."
        ),
        Segment(
            "{flow.n_transfers_near_threshold:count} transfers were placed just under "
            "{_threshold_reference}."
        ),
        Segment(
            "The formats used to settle were {flow.payment_formats:set}.",
        ),
    ),
)

_BASIS_POOL: tuple[tuple[Segment, ...], ...] = (
    (
        Segment("Indicators observed: {~indicators}."),
        Segment(
            "The nearest account carrying a source-data flag is "
            "{labels.min_hops_to_known_illicit:count} "
            "{_p:labels.min_hops_to_known_illicit:hop:hops} from the subject account."
        ),
        Segment(
            "{labels.illicit_inflow_share:share} of inbound value arrived from flagged "
            "counterparties.",
        ),
        Segment(
            "On the basis of the structure and timing described, the activity warrants "
            "further review. This report describes only the transactions observed within "
            "the review window and draws no conclusion about the account holder.",
            required=True,
        ),
    ),
    (
        Segment("The following indicators were noted: {~indicators}."),
        Segment(
            "A flagged account lies {labels.min_hops_to_known_illicit:count} "
            "{_p:labels.min_hops_to_known_illicit:hop:hops} away in the subgraph."
        ),
        Segment(
            "Flagged counterparties account for {labels.illicit_inflow_share:share} of "
            "inbound value.",
        ),
        Segment(
            "Taken together these observations merit further enquiry. No finding is made "
            "here beyond what the reviewed transactions show.",
            required=True,
        ),
    ),
    (
        Segment("Supporting indicators: {~indicators}."),
        Segment(
            "Distance from the subject account to the nearest flagged account is "
            "{labels.min_hops_to_known_illicit:count} "
            "{_p:labels.min_hops_to_known_illicit:hop:hops}."
        ),
        Segment(
            "Inbound value originating with flagged counterparties comes to "
            "{labels.illicit_inflow_share:share}.",
        ),
        Segment(
            "The pattern described warrants further review by an investigator with "
            "access to customer records not represented in this subgraph.",
            required=True,
        ),
    ),
    (
        Segment("Indicators supporting this referral: {~indicators}."),
        Segment(
            "The subject account sits {labels.min_hops_to_known_illicit:count} "
            "{_p:labels.min_hops_to_known_illicit:hop:hops} from the nearest flagged "
            "account."
        ),
        Segment(
            "{labels.illicit_inflow_share:share} of the value received came from flagged "
            "counterparties.",
        ),
        Segment(
            "These observations merit further enquiry. The account holder's intent is "
            "not addressed and cannot be established from transaction data alone.",
            required=True,
        ),
    ),
    (
        Segment("Indicators present in the reviewed activity: {~indicators}."),
        Segment(
            "{labels.min_hops_to_known_illicit:count} "
            "{_p:labels.min_hops_to_known_illicit:hop:hops} separate the subject account "
            "from the nearest flagged account."
        ),
        Segment(
            "The share of inbound value from flagged counterparties is "
            "{labels.illicit_inflow_share:share}.",
        ),
        Segment(
            "On this basis the activity warrants further review. The observations here "
            "are limited to the transactions inside the review window.",
            required=True,
        ),
    ),
    (
        Segment("Observed indicators: {~indicators}."),
        Segment(
            "The shortest path from the subject account to a flagged account is "
            "{labels.min_hops_to_known_illicit:count} "
            "{_p:labels.min_hops_to_known_illicit:hop:hops}."
        ),
        Segment(
            "Flagged inbound value represents {labels.illicit_inflow_share:share} of the "
            "total received.",
        ),
        Segment(
            "The activity merits further enquiry. This report records what the "
            "transaction data shows and does not characterise the parties involved.",
            required=True,
        ),
    ),
)


#: Typology statements for a case whose label names a laundering scheme. The label is a
#: checkable CATEGORICAL claim, and it is always accompanied by the scope caveat: an
#: AMLworld case is *part of* a stream of its typology and holds 65% of that stream's
#: transactions on average (D-019). A template describing the scheme as complete would be
#: H8, and the vocabulary's ``completeness`` forbidden list would catch it.
_STREAM_TYPOLOGY_PHRASINGS = (
    "The source data labels this case as part of a {typology.label:typology} stream; the "
    "subgraph reviewed here covers only the transactions inside the review window and "
    "need not contain the pattern in full.",
    "This case is recorded in the source data as belonging to a "
    "{typology.label:typology} stream. Only the portion of that stream falling inside the "
    "review window is described above.",
    "Ground-truth labelling places the case within a {typology.label:typology} stream. "
    "The transactions in scope are a subset of that stream.",
    "The case is labelled {typology.label:typology} in the source data, and the activity "
    "described is the part of that stream visible in the review window.",
    "Source labelling assigns this case to a {typology.label:typology} stream, of which "
    "the reviewed transactions form one part.",
)

#: Typology statements for a case the source data leaves unclassified. The claim is still
#: emitted — "unclassified" is a ground-truth statement about the case and is SUPPORTED,
#: not an absence to leave unsaid (D-035) — but it is phrased as a label rather than as
#: membership of a scheme that does not exist.
_LABEL_TYPOLOGY_PHRASINGS = (
    "The typology recorded for this case in the source data is " "{typology.label:typology}.",
    "Source-data typology for this case: {typology.label:typology}.",
    "The case carries the typology label {typology.label:typology}.",
    "Ground truth records the typology of this case as {typology.label:typology}.",
    "The typology assigned in the source data is {typology.label:typology}.",
)


def _pattern_variants(
    motif_sentences: tuple[str, ...], typology_style: str = "stream"
) -> tuple[tuple[Segment, ...], ...]:
    """Assemble a family's five pattern sections from its motif sentences.

    Args:
        motif_sentences: Five phrasings of the family's structural finding, each carrying
            the motif descriptors the typology's salience list requires.
        typology_style: ``"stream"`` for a family whose label names a laundering scheme,
            ``"label"`` for one the source data leaves unclassified, ``"none"`` for
            ``topology_only``, where no typology ground truth exists at all and asserting
            one would be UNVERIFIABLE by construction.

    Returns:
        Five pattern sections.

    Raises:
        ValueError: If ``typology_style`` is not one of the three.
    """
    phrasings = {
        "stream": _STREAM_TYPOLOGY_PHRASINGS,
        "label": _LABEL_TYPOLOGY_PHRASINGS,
        "none": (),
    }
    if typology_style not in phrasings:
        raise ValueError(f"unknown typology style {typology_style!r}")
    bank = phrasings[typology_style]
    sections: list[tuple[Segment, ...]] = []
    for i, sentence in enumerate(motif_sentences):
        segments = [Segment(sentence)]
        if bank:
            segments.append(Segment(bank[i % len(bank)], required=True))
        sections.append(tuple(segments))
    return tuple(sections)


_FAN_OUT_PATTERN = _pattern_variants(
    (
        "The subgraph exhibits a fan-out centred on account {motifs.fan_out.hub:entity}, "
        "which dispersed to {motifs.fan_out.width:count} distinct recipients within "
        "{motifs.fan_out.window_hours:duration}.",
        "A fan-out structure is present: account {motifs.fan_out.hub:entity} distributed "
        "to {motifs.fan_out.width:count} separate recipients over "
        "{motifs.fan_out.window_hours:duration}.",
        "Structurally the case appears consistent with dispersal from a single point: "
        "{motifs.fan_out.hub:entity} paid {motifs.fan_out.width:count} distinct "
        "recipients inside {motifs.fan_out.window_hours:duration}.",
        "The dominant structure is a fan-out of width {motifs.fan_out.width:count} from "
        "account {motifs.fan_out.hub:entity}, formed over "
        "{motifs.fan_out.window_hours:duration}.",
        "Account {motifs.fan_out.hub:entity} sits at the head of a fan-out reaching "
        "{motifs.fan_out.width:count} recipients across "
        "{motifs.fan_out.window_hours:duration}.",
    )
)

_FAN_IN_PATTERN = _pattern_variants(
    (
        "The subgraph exhibits a fan-in on account {motifs.fan_in.hub:entity}, which "
        "received from {motifs.fan_in.width:count} distinct senders within "
        "{motifs.fan_in.window_hours:duration}.",
        "A fan-in structure is present: {motifs.fan_in.width:count} distinct senders paid "
        "into account {motifs.fan_in.hub:entity} over {motifs.fan_in.window_hours:duration}.",
        "The case appears consistent with aggregation at a single point: "
        "{motifs.fan_in.hub:entity} collected from {motifs.fan_in.width:count} senders "
        "inside {motifs.fan_in.window_hours:duration}.",
        "The dominant structure is a fan-in of width {motifs.fan_in.width:count} into "
        "account {motifs.fan_in.hub:entity}, formed over "
        "{motifs.fan_in.window_hours:duration}.",
        "Account {motifs.fan_in.hub:entity} sits at the point of collection for a fan-in "
        "of {motifs.fan_in.width:count} senders across "
        "{motifs.fan_in.window_hours:duration}.",
    )
)

_GATHER_SCATTER_PATTERN = _pattern_variants(
    (
        "Account {motifs.gather_scatter.hub:entity} collected from "
        "{motifs.gather_scatter.gather_width:count} senders and then dispersed to "
        "{motifs.gather_scatter.scatter_width:count} recipients.",
        "A gather-scatter structure runs through {motifs.gather_scatter.hub:entity}: "
        "inbound from {motifs.gather_scatter.gather_width:count} distinct senders, "
        "outbound to {motifs.gather_scatter.scatter_width:count} distinct recipients.",
        "The subgraph is indicative of collection followed by dispersal at "
        "{motifs.gather_scatter.hub:entity}, with "
        "{motifs.gather_scatter.gather_width:count} sources and "
        "{motifs.gather_scatter.scatter_width:count} destinations.",
        "Value converged on {motifs.gather_scatter.hub:entity} from "
        "{motifs.gather_scatter.gather_width:count} accounts before moving on to "
        "{motifs.gather_scatter.scatter_width:count} others.",
        "The dominant structure is gather-scatter at "
        "{motifs.gather_scatter.hub:entity}: {motifs.gather_scatter.gather_width:count} "
        "in, {motifs.gather_scatter.scatter_width:count} out.",
    )
)

_SCATTER_GATHER_PATTERN = _pattern_variants(
    (
        "Value left {motifs.scatter_gather.origin:entity}, divided across "
        "{motifs.scatter_gather.width:count} intermediate accounts, and recombined at "
        "{motifs.scatter_gather.destination:entity}.",
        "A scatter-gather structure connects {motifs.scatter_gather.origin:entity} to "
        "{motifs.scatter_gather.destination:entity} through "
        "{motifs.scatter_gather.width:count} parallel intermediaries.",
        "The subgraph is indicative of division and recombination: "
        "{motifs.scatter_gather.width:count} parallel paths run from "
        "{motifs.scatter_gather.origin:entity} to "
        "{motifs.scatter_gather.destination:entity}.",
        "{motifs.scatter_gather.width:count} intermediate accounts separate "
        "{motifs.scatter_gather.origin:entity} from "
        "{motifs.scatter_gather.destination:entity}, with value passing through all of "
        "them.",
        "The dominant structure is scatter-gather of width "
        "{motifs.scatter_gather.width:count}, originating at "
        "{motifs.scatter_gather.origin:entity} and terminating at "
        "{motifs.scatter_gather.destination:entity}.",
    )
)

_CYCLE_PATTERN = _pattern_variants(
    (
        "The subgraph contains a directed cycle of length {motifs.cycle.length:count}, in "
        "which value returns to an account it previously left.",
        "A closed loop of {motifs.cycle.length:count} accounts is present, so value "
        "returns to its point of origin within the review window.",
        "The case appears consistent with circular movement: a directed cycle spanning "
        "{motifs.cycle.length:count} accounts was detected.",
        "Value traverses a cycle of {motifs.cycle.length:count} accounts and returns to "
        "the account it started from.",
        "The dominant structure is a directed cycle of {motifs.cycle.length:count} " "accounts.",
    )
)

_BIPARTITE_PATTERN = _pattern_variants(
    (
        "The accounts divide cleanly into two groups of {motifs.bipartite.left_size:count} "
        "and {motifs.bipartite.right_size:count}, with transactions running only between "
        "the groups and never inside them.",
        "A bipartite structure is present: {motifs.bipartite.left_size:count} accounts on "
        "one side, {motifs.bipartite.right_size:count} on the other, and no transaction "
        "within either side.",
        "The subgraph is two-colourable, separating {motifs.bipartite.left_size:count} "
        "accounts from {motifs.bipartite.right_size:count} with all value crossing "
        "between them.",
        "Transactions run strictly between a group of "
        "{motifs.bipartite.left_size:count} accounts and a group of "
        "{motifs.bipartite.right_size:count}.",
        "The dominant structure is bipartite, with sides of "
        "{motifs.bipartite.left_size:count} and {motifs.bipartite.right_size:count} "
        "accounts.",
    )
)

_STACK_PATTERN = _pattern_variants(
    (
        "The subgraph is layered {motifs.stack.depth:count} deep, value passing through "
        "successive tiers of accounts rather than moving directly.",
        "A stacked structure of {motifs.stack.depth:count} layers is present, each tier "
        "forwarding to the next.",
        "The case is indicative of layering: {motifs.stack.depth:count} successive "
        "account tiers separate the earliest sender from the final recipient.",
        "Value moves through {motifs.stack.depth:count} distinct layers of accounts "
        "before reaching its destination.",
        "The dominant structure is a stack of depth {motifs.stack.depth:count}.",
    )
)

_RANDOM_PATTERN = _pattern_variants(
    (
        "No single dominant motif organises the subgraph; its density is "
        "{structure.density:density} over {structure.n_nodes:count} accounts.",
        "The subgraph shows no one governing structure, with density "
        "{structure.density:density} across its {structure.n_nodes:count} accounts.",
        "Connections are distributed rather than organised around one account: density "
        "is {structure.density:density}.",
        "The structure is diffuse, at density {structure.density:density} over "
        "{structure.n_nodes:count} accounts.",
        "No dominant motif was detected; the {structure.n_nodes:count} accounts connect "
        "at density {structure.density:density}.",
    )
)

_UNCLASSIFIED_SUSPICIOUS_PATTERN = _pattern_variants(
    (
        "The source data assigns no laundering typology to this case, though "
        "{labels.n_illicit_transactions:count} of its transactions carry a flag. The "
        "subgraph connects {structure.n_nodes:count} accounts at density "
        "{structure.density:density}.",
        "No typology is recorded for this case, although flagged transactions are "
        "present. Structurally, {structure.n_nodes:count} accounts are connected at "
        "density {structure.density:density}.",
        "The case carries flagged transactions but no ground-truth typology. Its "
        "{structure.n_nodes:count} accounts connect at density "
        "{structure.density:density}.",
        "Flagged activity is present without an accompanying typology label. The "
        "subgraph's density across {structure.n_nodes:count} accounts is "
        "{structure.density:density}.",
        "This case is unlabelled as to typology despite carrying flagged transactions; "
        "{structure.n_nodes:count} accounts connect at density "
        "{structure.density:density}.",
    ),
    typology_style="label",
)

_NO_FINDING_PATTERN = _pattern_variants(
    (
        "No structural laundering pattern is indicated for this case, and the source data "
        "records no flagged transaction within it. The {structure.n_nodes:count} accounts "
        "in scope connect at density {structure.density:density}.",
        "The source data records this case as carrying no laundering typology and no "
        "flagged transaction. Its {structure.n_nodes:count} accounts connect at density "
        "{structure.density:density}.",
        "Ground truth indicates no laundering pattern here: no transaction in the "
        "subgraph is flagged. Density across {structure.n_nodes:count} accounts is "
        "{structure.density:density}.",
        "This case carries neither a laundering typology nor a flagged transaction in the "
        "source data. The subgraph holds {structure.n_nodes:count} accounts at density "
        "{structure.density:density}.",
        "No laundering typology and no flagged transaction are recorded for this case. "
        "The {structure.n_nodes:count} accounts connect at density "
        "{structure.density:density}.",
    ),
    typology_style="label",
)

_MINIMAL_PATTERN = _pattern_variants(
    (
        "The subgraph holds only {structure.n_nodes:count} accounts, which is too few for "
        "any multi-party structure to be present, so no typology claim is made from "
        "shape.",
        "With {structure.n_nodes:count} accounts in scope there is no multi-party "
        "structure to characterise, and none is asserted.",
        "Only {structure.n_nodes:count} accounts fall inside the review window, which "
        "does not support a structural finding.",
        "The case comprises {structure.n_nodes:count} accounts; no motif can be present "
        "at that size and none is claimed.",
        "At {structure.n_nodes:count} accounts the subgraph is below the size at which "
        "the structural detectors report anything.",
    ),
    typology_style="label",
)

_TOPOLOGY_PATTERN = _pattern_variants(
    (
        "The subgraph connects {structure.n_nodes:count} accounts through "
        "{structure.n_edges:count} transactions at density {structure.density:density}, "
        "with a maximum in-degree of {structure.max_in_degree:count} and a maximum "
        "out-degree of {structure.max_out_degree:count}.",
        "Topologically the case is {structure.n_nodes:count} accounts and "
        "{structure.n_edges:count} transactions, density "
        "{structure.density:density}, peak in-degree "
        "{structure.max_in_degree:count}, peak out-degree "
        "{structure.max_out_degree:count}.",
        "The structure comprises {structure.n_nodes:count} accounts and "
        "{structure.n_edges:count} transactions; density is "
        "{structure.density:density} and the widest single point of collection has "
        "{structure.max_in_degree:count} inbound counterparties.",
        "Across {structure.n_nodes:count} accounts and {structure.n_edges:count} "
        "transactions the subgraph reaches density {structure.density:density}, with "
        "{structure.max_out_degree:count} the largest number of distinct recipients from "
        "any one account.",
        "The case is described by topology alone: {structure.n_nodes:count} accounts, "
        "{structure.n_edges:count} transactions, density "
        "{structure.density:density}, reciprocity {structure.reciprocity:density}.",
    ),
    typology_style="none",
)

# --------------------------------------------------------- topology-only pools ---
# Elliptic2 carries no amounts, no clock and no per-transaction labels. These pools
# therefore contain no monetary, temporal or label slot at all — the family cannot assert
# a masked fact because it has no way to name one.

_TOPOLOGY_SUBJECT: tuple[tuple[Segment, ...], ...] = (
    (
        Segment(
            "This report concerns account {focal_entity.id:entity}, which acts as "
            "{focal_entity.role:role} within the reviewed subgraph.",
            required=True,
        ),
        Segment(
            "The subgraph comprises {structure.n_nodes:count} accounts connected by "
            "{structure.n_edges:count} transactions in "
            "{structure.n_components:count} "
            "{_p:structure.n_components:component:components}.",
            required=True,
        ),
        Segment(
            "The substrate carries no transaction amounts, no absolute timestamps and no "
            "per-transaction labelling, so this report is confined to structure.",
            required=True,
        ),
    ),
    (
        Segment(
            "The subject of this report is account {focal_entity.id:entity}, positioned "
            "as {focal_entity.role:role}.",
            required=True,
        ),
        Segment(
            "In scope are {structure.n_nodes:count} accounts, {structure.n_edges:count} "
            "transactions and {structure.n_components:count} connected "
            "{_p:structure.n_components:component:components}.",
            required=True,
        ),
        Segment(
            "Amounts, timestamps and per-transaction labels are absent from this "
            "substrate; only topology is available and only topology is described.",
            required=True,
        ),
    ),
    (
        Segment(
            "Account {focal_entity.id:entity} is the subject of this referral, appearing "
            "as {focal_entity.role:role}.",
            required=True,
        ),
        Segment(
            "The reviewed subgraph holds {structure.n_nodes:count} accounts and "
            "{structure.n_edges:count} transactions across "
            "{structure.n_components:count} "
            "{_p:structure.n_components:component:components}.",
            required=True,
        ),
        Segment(
            "This substrate supplies neither monetary amounts nor absolute timing, so no "
            "statement about value or elapsed time is made.",
            required=True,
        ),
    ),
    (
        Segment(
            "This referral describes structure around account "
            "{focal_entity.id:entity}, identified as {focal_entity.role:role}.",
            required=True,
        ),
        Segment(
            "The subgraph contains {structure.n_nodes:count} accounts and "
            "{structure.n_edges:count} transactions, forming "
            "{structure.n_components:count} "
            "{_p:structure.n_components:component:components}.",
            required=True,
        ),
        Segment(
            "No amounts, timestamps or transaction-level labels exist for this substrate, "
            "and none are asserted below.",
            required=True,
        ),
    ),
    (
        Segment(
            "The account under review is {focal_entity.id:entity}, which the subgraph "
            "places as {focal_entity.role:role}.",
            required=True,
        ),
        Segment(
            "It sits among {structure.n_nodes:count} accounts joined by "
            "{structure.n_edges:count} transactions in "
            "{structure.n_components:count} "
            "{_p:structure.n_components:component:components}.",
            required=True,
        ),
        Segment(
            "Because the substrate is anonymised, this report describes connectivity "
            "only and makes no claim about value or timing.",
            required=True,
        ),
    ),
)

_TOPOLOGY_ACTIVITY: tuple[tuple[Segment, ...], ...] = (
    (
        Segment(
            "The subject account has {focal_entity.in_degree:count} distinct inbound "
            "{_p:focal_entity.in_degree:counterparty:counterparties} over "
            "{focal_entity.n_transactions_in:count} transactions, and "
            "{focal_entity.out_degree:count} distinct outbound "
            "{_p:focal_entity.out_degree:counterparty:counterparties} over "
            "{focal_entity.n_transactions_out:count} transactions.",
            required=True,
        ),
        Segment(
            "The longest directed path in the subgraph runs "
            "{motifs.chain.max_length:count} transactions.",
        ),
        Segment(
            "The subgraph's diameter is {structure.diameter:count} "
            "{_p:structure.diameter:hop:hops} and its "
            "reciprocity {structure.reciprocity:density}.",
        ),
    ),
    (
        Segment(
            "Inbound connectivity is {focal_entity.in_degree:count} distinct "
            "{_p:focal_entity.in_degree:counterparty:counterparties} across "
            "{focal_entity.n_transactions_in:count} transactions; outbound connectivity "
            "is {focal_entity.out_degree:count} distinct "
            "{_p:focal_entity.out_degree:counterparty:counterparties} across "
            "{focal_entity.n_transactions_out:count} transactions.",
            required=True,
        ),
        Segment(
            "Chains of up to {motifs.chain.max_length:count} consecutive transactions " "occur.",
        ),
        Segment(
            "Diameter is {structure.diameter:count} "
            "{_p:structure.diameter:hop:hops}; reciprocity is "
            "{structure.reciprocity:density}.",
        ),
    ),
    (
        Segment(
            "Around the subject account there are {focal_entity.in_degree:count} distinct "
            "senders and {focal_entity.out_degree:count} distinct recipients, carried by "
            "{focal_entity.n_transactions_in:count} inbound and "
            "{focal_entity.n_transactions_out:count} outbound transactions.",
            required=True,
        ),
        Segment(
            "The longest simple directed path spans {motifs.chain.max_length:count} "
            "transactions.",
        ),
        Segment(
            "The subgraph spans {structure.diameter:count} "
            "{_p:structure.diameter:hop:hops} end to end, at "
            "reciprocity {structure.reciprocity:density}.",
        ),
    ),
    (
        Segment(
            "The account connects to {focal_entity.in_degree:count} distinct inbound and "
            "{focal_entity.out_degree:count} distinct outbound "
            "{_p:focal_entity.out_degree:counterparty:counterparties}, over "
            "{focal_entity.n_transactions_in:count} and "
            "{focal_entity.n_transactions_out:count} transactions respectively.",
            required=True,
        ),
        Segment(
            "Directed paths of {motifs.chain.max_length:count} transactions are present.",
        ),
        Segment(
            "End-to-end distance in the subgraph is {structure.diameter:count} "
            "{_p:structure.diameter:hop:hops}, with "
            "reciprocity {structure.reciprocity:density}.",
        ),
    ),
    (
        Segment(
            "Distinct counterparties number {focal_entity.in_degree:count} inbound and "
            "{focal_entity.out_degree:count} outbound, over "
            "{focal_entity.n_transactions_in:count} inbound and "
            "{focal_entity.n_transactions_out:count} outbound transactions.",
            required=True,
        ),
        Segment(
            "The deepest directed path observed is {motifs.chain.max_length:count} "
            "transactions long.",
        ),
        Segment(
            "The subgraph measures {structure.diameter:count} "
            "{_p:structure.diameter:hop:hops} across, at reciprocity "
            "{structure.reciprocity:density}.",
        ),
    ),
)

_TOPOLOGY_BASIS: tuple[tuple[Segment, ...], ...] = (
    (
        Segment("Indicators observed: {~indicators}."),
        Segment(
            "This substrate supports no statement about value, timing or counterparty "
            "labelling, so the referral rests on connectivity alone and warrants further "
            "review against records this subgraph does not contain.",
            required=True,
        ),
    ),
    (
        Segment("The following indicators were noted: {~indicators}."),
        Segment(
            "Because amounts and timings are unavailable here, the observations above are "
            "structural only and merit further enquiry alongside data this subgraph does "
            "not carry.",
            required=True,
        ),
    ),
    (
        Segment("Supporting indicators: {~indicators}."),
        Segment(
            "The referral is grounded in connectivity alone. It warrants further review "
            "by an analyst with access to the value and timing data absent from this "
            "substrate.",
            required=True,
        ),
    ),
    (
        Segment("Indicators supporting this referral: {~indicators}."),
        Segment(
            "No claim about value or elapsed time can be supported here. On structure "
            "alone the activity merits further enquiry.",
            required=True,
        ),
    ),
    (
        Segment("Indicators present in the reviewed activity: {~indicators}."),
        Segment(
            "The substrate's anonymisation limits this report to structure; on that basis "
            "the activity warrants further review.",
            required=True,
        ),
    ),
)


def _make_family(
    key: str,
    typologies: tuple[str, ...],
    requires_mask: tuple[str, ...],
    pattern: tuple[tuple[Segment, ...], ...],
    offset: int,
    description: str,
    *,
    topology: bool = False,
) -> Family:
    """Assemble a family from its pattern sections and the pools it draws on.

    Args:
        key: Family name.
        typologies: Typology labels this family serves.
        requires_mask: Availability flags every record must carry.
        pattern: The family's pattern sections, one per structural phrasing.
        offset: Where this family enters the shared pools, so two families rarely open a
            narrative with the same sentence.
        description: What the family is for.
        topology: Draw on the topology-only pools rather than the shared ones.

    Returns:
        The family.
    """
    return Family(
        key=key,
        typologies=typologies,
        requires_mask=requires_mask,
        pattern_sections=pattern,
        subject_pool=_TOPOLOGY_SUBJECT if topology else _SUBJECT_POOL,
        activity_pool=_TOPOLOGY_ACTIVITY if topology else _ACTIVITY_POOL,
        basis_pool=_TOPOLOGY_BASIS if topology else _BASIS_POOL,
        offset=offset,
        description=description,
    )


#: Availability flags every amount-and-clock-bearing family requires.
_FULL_MASK = ("monetary_amounts", "absolute_timestamps", "node_labels")

FAMILIES: dict[str, Family] = {
    "fan_out": _make_family(
        key="fan_out",
        typologies=("fan_out",),
        requires_mask=_FULL_MASK,
        pattern=_FAN_OUT_PATTERN,
        offset=0,
        description="Dispersal from one account to many.",
    ),
    "fan_in": _make_family(
        key="fan_in",
        typologies=("fan_in",),
        requires_mask=_FULL_MASK,
        pattern=_FAN_IN_PATTERN,
        offset=1,
        description="Aggregation into one account from many.",
    ),
    "gather_scatter": _make_family(
        key="gather_scatter",
        typologies=("gather_scatter",),
        requires_mask=_FULL_MASK,
        pattern=_GATHER_SCATTER_PATTERN,
        offset=2,
        description="Collection at a hub followed by dispersal from it.",
    ),
    "scatter_gather": _make_family(
        key="scatter_gather",
        typologies=("scatter_gather",),
        requires_mask=_FULL_MASK,
        pattern=_SCATTER_GATHER_PATTERN,
        offset=3,
        description="Division across parallel paths and recombination at one destination.",
    ),
    "cycle": _make_family(
        key="cycle",
        typologies=("cycle",),
        requires_mask=_FULL_MASK,
        pattern=_CYCLE_PATTERN,
        offset=4,
        description="Value returning to an account it previously left.",
    ),
    "bipartite": _make_family(
        key="bipartite",
        typologies=("bipartite",),
        requires_mask=_FULL_MASK,
        pattern=_BIPARTITE_PATTERN,
        offset=5,
        description="Two disjoint groups with transactions only between them.",
    ),
    "stack": _make_family(
        key="stack",
        typologies=("stack",),
        requires_mask=_FULL_MASK,
        pattern=_STACK_PATTERN,
        offset=0,
        description="Successive layers of accounts forwarding value onward.",
    ),
    "random": _make_family(
        key="random",
        typologies=("random",),
        requires_mask=_FULL_MASK,
        pattern=_RANDOM_PATTERN,
        offset=2,
        description="A labelled laundering stream with no dominant structural motif.",
    ),
    "unclassified_suspicious": _make_family(
        key="unclassified_suspicious",
        typologies=(),
        requires_mask=_FULL_MASK,
        pattern=_UNCLASSIFIED_SUSPICIOUS_PATTERN,
        offset=4,
        description="Flagged transactions present, but no ground-truth typology.",
    ),
    "no_finding": _make_family(
        key="no_finding",
        typologies=(),
        requires_mask=_FULL_MASK,
        pattern=_NO_FINDING_PATTERN,
        offset=1,
        description=(
            "Licit activity. The absence of a laundering pattern is stated positively "
            "rather than papered over (D-035)."
        ),
    ),
    "minimal_activity": _make_family(
        key="minimal_activity",
        typologies=(),
        requires_mask=_FULL_MASK,
        pattern=_MINIMAL_PATTERN,
        offset=3,
        description="A subgraph too small for any multi-party structure to exist.",
    ),
    "topology_only": _make_family(
        key="topology_only",
        typologies=(),
        requires_mask=(),
        pattern=_TOPOLOGY_PATTERN,
        offset=0,
        topology=True,
        description=(
            "Elliptic2. No amounts, no clock, no per-transaction labels: the only family "
            "whose pools contain no slot for any of them."
        ),
    ),
}

#: Typology label to the family that serves it.
FAMILY_FOR_TYPOLOGY: dict[str, str] = {
    typology: family.key for family in FAMILIES.values() for typology in family.typologies
}

#: The families reached by a rule rather than by a typology label.
UNCLASSIFIED_SUSPICIOUS_FAMILY = "unclassified_suspicious"
NO_FINDING_FAMILY = "no_finding"
MINIMAL_FAMILY = "minimal_activity"
TOPOLOGY_FAMILY = "topology_only"

#: Accounts below which no multi-party structure can exist, so ``minimal_activity`` is
#: used regardless of what else the record says. Two accounts is a transfer; the motif
#: detectors need three (``fan_min_width`` is 3, ``cycle_min_length`` is 3).
MINIMAL_NODE_CEILING = 2

#: One sentence per salient field, used only when a variant's own segments happen not to
#: mention a field the typology's salience list requires. Bronze therefore reaches 100%
#: salience coverage **by construction**, which is the ceiling Phase 6 and Phase 10 score
#: learned systems against. Every sentence here carries the field as a checkable slot; a
#: fallback that mentioned a field without annotating it would raise coverage while
#: lowering verifiability, which is the wrong trade in both directions.
SALIENCE_SENTENCES: dict[str, str] = {
    "focal_entity.id": "The account under review is {focal_entity.id:entity}.",
    "focal_entity.role": "Its position in the subgraph is {focal_entity.role:role}.",
    "structure.n_nodes": "The subgraph holds {structure.n_nodes:count} accounts.",
    "structure.n_edges": "It records {structure.n_edges:count} transactions.",
    "structure.density": "Its density is {structure.density:density}.",
    "temporal.span_hours": "The activity spans {temporal.span_hours:duration}.",
    "temporal.event_ordering": "The phase sequence was {temporal.event_ordering:ordering}.",
    "flow.total_inflow": "Total inbound value was approximately {flow.total_inflow:money}.",
    "flow.total_outflow": "Total outbound value was approximately {flow.total_outflow:money}.",
    "labels.n_illicit_counterparties": (
        "{labels.n_illicit_counterparties:count} "
        "{_p:labels.n_illicit_counterparties:counterparty:counterparties} are associated "
        "with flagged transactions."
    ),
    "motifs.fan_out.width": "The fan-out reaches {motifs.fan_out.width:count} recipients.",
    "motifs.fan_in.width": "The fan-in draws on {motifs.fan_in.width:count} senders.",
    "motifs.cycle.length": "The cycle spans {motifs.cycle.length:count} accounts.",
    "motifs.stack.depth": "The stack is {motifs.stack.depth:count} layers deep.",
    "motifs.bipartite.left_size": (
        "One side of the bipartition holds {motifs.bipartite.left_size:count} accounts."
    ),
    "motifs.bipartite.right_size": (
        "The other side holds {motifs.bipartite.right_size:count} accounts."
    ),
    "motifs.gather_scatter.gather_width": (
        "Collection drew on {motifs.gather_scatter.gather_width:count} accounts."
    ),
    "motifs.gather_scatter.scatter_width": (
        "Dispersal reached {motifs.gather_scatter.scatter_width:count} accounts."
    ),
    "motifs.scatter_gather.width": (
        "The scatter-gather runs {motifs.scatter_gather.width:count} accounts wide."
    ),
}


def family_for(key: str) -> Family:
    """Look up a family by name.

    Args:
        key: The family key.

    Returns:
        The family.

    Raises:
        KeyError: If no family has that key.
    """
    if key not in FAMILIES:
        raise KeyError(f"unknown template family {key!r}; known families are {sorted(FAMILIES)}")
    return FAMILIES[key]


def n_variants(key: str) -> int:
    """Return how many *structural* surface realisations a family has.

    This is the count the acceptance criteria speak of: how many genuinely different ways
    the family describes its own structure. The number of distinct whole narratives it can
    produce is :attr:`Family.n_realisations`, which is larger by the product of the three
    shared section pools.

    Args:
        key: The family key.

    Returns:
        The pattern-section count.

    Raises:
        KeyError: If no family has that key.
    """
    return family_for(key).n_surface_variants

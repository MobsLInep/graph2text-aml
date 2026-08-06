"""The fact record, rendered for a human to read while they write.

Not :mod:`g2t_aml.facts.serialiser`. That module flattens a record into text for a
*language model* — the B7 baseline — and its discipline is that every fact reaches the
output, availability sentinels rendered explicitly and all. Both of those are wrong here.
An annotator reading forty lines of ``key=value`` with a third of them saying
``unavailable`` will skim, and a skimmed fact record produces a narrative that asserts
what the annotator assumed rather than what the record says.

So this panel is the opposite discipline:

**A masked field is not shown at all.** Not as ``null``, not as ``—``, not as
"unavailable". Invariant 4 says nothing may assert a fact the substrate cannot support,
and the most reliable way to stop someone writing about an amount is for no amount to be
on their screen. On Elliptic2 the money section does not exist. What *is* shown, once, at
the top, is the list of fact families the substrate cannot support — a property of the
substrate the annotator needs to know, stated as a property of the substrate rather than
as forty individual absences.

**A measured null is shown**, and labelled as measured. "No cycle was found" is a fact
about this case; "amounts are not available on this substrate" is a fact about Elliptic2.
The two look identical in a naive renderer and mean opposite things to the checker
(D-025), so they are rendered differently here too.

**Salience is marked in place.** A field on this case's salience list carries a marker, so
"what must this narrative mention" is answered by the panel rather than by the annotator
remembering a table from the guidelines. The list comes from
:func:`~g2t_aml.facts.salience.salience_report`, already filtered for availability, so the
panel and the automated adequacy metric are reading one definition.

**Values are rendered with the Bronze formatters, and that is load-bearing.** Ingestion
aligns a Gold narrative against Bronze's slot values by *exact* string match (D-048), so
an amount displayed here as ``9,434.82 Canadian Dollar`` where Bronze renders ``9,435
Canadian Dollar`` would mean that an annotator who copied the panel correctly produced a
value that aligns to nothing: scored as a dropped fact *and* as an invented quantity, on
every monetary case in the corpus. Sharing the formatters is what makes "write down what
the panel says" the behaviour the pipeline rewards. This module imports the formatters
only — no template, no phrasing, nothing that would put generated prose on the screen.

The panel is a data structure, not a widget. :func:`build_fact_panel` returns rows and
sections; :mod:`g2t_aml.human.annotation_ui` decides how they look. That split is what
lets the Elliptic2 masking be unit-tested without starting a browser.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from g2t_aml.corpus.bronze.format import (
    format_count,
    format_density,
    format_duration,
    format_money,
    format_percent,
    format_timestamp,
)
from g2t_aml.corpus.bronze.templates import ROLE_DISPLAY
from g2t_aml.facts.salience import salience_report
from g2t_aml.facts.schema import (
    MOTIF_NAMES,
    CaseFacts,
    FlowFacts,
    LabelFacts,
    Money,
    TemporalFacts,
    Unavailable,
    is_available,
)
from g2t_aml.facts.vocab import ControlledVocabulary, load_vocabulary

__all__ = [
    "MASKED_FAMILY_LABELS",
    "FactPanel",
    "FactRow",
    "FactSection",
    "build_fact_panel",
]

#: Human-readable names for the fact families a substrate can mask out, keyed by the
#: record attribute that carries the sentinel. Shown once as a banner; never per field.
MASKED_FAMILY_LABELS: dict[str, str] = {
    "temporal": "timing — this substrate has no wall-clock timestamps",
    "flow": "value — this substrate has no monetary amounts or currencies",
    "labels": "counterparty labels — this substrate labels whole subgraphs, not accounts",
}

#: How close a whitelisted reference's threshold must be to the case's before the two are
#: taken to be the same threshold. One cent: these are round figures in a named currency,
#: so anything looser would match a different rule.
_THRESHOLD_TOLERANCE = 0.01

#: Motif key to the phrase the guidelines use. Defined here rather than taken from the
#: template pack: no motif name is a slot value, so nothing aligns against these, and the
#: guidelines and the panel should agree with each other rather than with a renderer.
_MOTIF_LABELS: dict[str, str] = {
    "fan_in": "fan-in",
    "fan_out": "fan-out",
    "chain": "chain",
    "cycle": "cycle",
    "bipartite": "bipartite",
    "stack": "stack",
    "gather_scatter": "gather-scatter",
    "scatter_gather": "scatter-gather",
}


@dataclass(frozen=True)
class FactRow:
    """One line of the panel: a fact, its value, and whether it must be mentioned.

    Attributes:
        label: What to call it on screen.
        value: The rendered value. Never the string ``"None"`` and never ``"unavailable"``
            — a row that would carry either is not built at all.
        field_path: The dotted path into the fact record, so the row can be matched
            against the salience list and quoted in a review comment.
        salient: Whether this case's salience list requires the narrative to mention it.
        measured_null: True when the record holds a measured ``None`` — the motif is
            absent, no flagged node is reachable. A fact about the case, not a mask.
        note: Optional clarification shown beside the value.
    """

    label: str
    value: str
    field_path: str
    salient: bool = False
    measured_null: bool = False
    note: str = ""


@dataclass(frozen=True)
class FactSection:
    """A group of rows under one heading.

    Attributes:
        name: The heading.
        rows: The rows, in display order.
        blurb: One line saying what an annotator should do with this section.
    """

    name: str
    rows: tuple[FactRow, ...]
    blurb: str = ""

    def __bool__(self) -> bool:
        """Report whether the section has anything to show.

        Returns:
            True when it holds at least one row. An empty section is dropped rather than
            rendered as a heading over nothing.
        """
        return bool(self.rows)


@dataclass(frozen=True)
class FactPanel:
    """Everything the annotator is shown about the case, minus the graph.

    Attributes:
        case_id: The case.
        dataset: Substrate key.
        typology: The typology label, and whether it is ground truth or inferred.
        typology_source: ``ground_truth``, ``inferred`` or ``none``.
        typology_scope: ``stream_membership`` or ``case_structure``. Load-bearing:
            stream membership means the case may not exhibit the typology in full
            (D-019), and a narrative claiming a complete scheme is H8.
        sections: The populated sections, in display order.
        masked_families: One line per fact family the substrate cannot support.
        required_fields: The salience list for this case, availability already applied.
        excused_fields: Salient fields excused because the record cannot support them.
    """

    case_id: str
    dataset: str
    typology: str
    typology_source: str
    typology_scope: str
    sections: tuple[FactSection, ...]
    masked_families: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    excused_fields: tuple[str, ...] = ()

    def section(self, name: str) -> FactSection | None:
        """Return a section by name.

        Args:
            name: The heading.

        Returns:
            The section, or None when it was dropped for holding no rows.
        """
        return next((s for s in self.sections if s.name == name), None)

    def all_rows(self) -> tuple[FactRow, ...]:
        """Return every row in the panel.

        Returns:
            The rows, section order preserved.
        """
        return tuple(row for section in self.sections for row in section.rows)

    def rendered_text(self) -> str:
        """Return the whole panel as plain text.

        Used by the calibration report and by tests that need to assert what an annotator
        could possibly have read.

        Returns:
            The panel as text, one row per line.
        """
        lines = [f"CASE {self.case_id}  [{self.dataset}]"]
        if self.masked_families:
            lines += ["", "Not available on this substrate:"]
            lines += [f"  - {m}" for m in self.masked_families]
        for section in self.sections:
            lines += ["", section.name.upper()]
            for row in section.rows:
                marker = "*" if row.salient else " "
                note = f"   ({row.note})" if row.note else ""
                lines.append(f" {marker} {row.label}: {row.value}{note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialised panel.

        Returns:
            A JSON-serialisable mapping, for the annotation record's provenance.
        """
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "typology": self.typology,
            "typology_source": self.typology_source,
            "typology_scope": self.typology_scope,
            "masked_families": list(self.masked_families),
            "required_fields": list(self.required_fields),
            "excused_fields": list(self.excused_fields),
            "sections": [
                {
                    "name": s.name,
                    "rows": [
                        {
                            "label": r.label,
                            "value": r.value,
                            "field_path": r.field_path,
                            "salient": r.salient,
                            "measured_null": r.measured_null,
                        }
                        for r in s.rows
                    ],
                }
                for s in self.sections
            ],
        }


def _money(value: Money | Unavailable) -> str | None:
    """Render a monetary field, or return None when it is masked.

    Args:
        value: A :class:`~g2t_aml.facts.schema.Money` or a sentinel.

    Returns:
        The rendered amount, or None when the field is unavailable and must not appear.
    """
    if isinstance(value, Unavailable):
        return None
    return format_money(value.value, value.currency)


def _timestamp(value: datetime | Unavailable | None) -> str | None:
    """Render a timestamp, or return None when it is masked or absent.

    Args:
        value: A datetime, a sentinel or None.

    Returns:
        ``YYYY-MM-DD HH:MM``, or None.
    """
    if value is None or isinstance(value, Unavailable):
        return None
    return format_timestamp(value)


class _Builder:
    """Accumulates rows, dropping anything the record cannot support.

    A tiny class rather than a pile of local functions so the "is this salient" lookup and
    the "drop a masked value" rule are applied at exactly one place each.
    """

    def __init__(self, required: frozenset[str]) -> None:
        """Bind the builder to this case's salience list.

        Args:
            required: Field paths the narrative must mention.
        """
        self.required = required
        self.rows: list[FactRow] = []

    def add(
        self,
        label: str,
        value: object,
        path: str,
        *,
        note: str = "",
        measured_null: bool = False,
    ) -> None:
        """Add a row, unless the value is masked or absent.

        Args:
            label: On-screen name.
            value: The rendered value. None drops the row entirely — this is the single
                point at which invariant 4 is applied to the panel.
            path: The field path.
            note: Optional clarification.
            measured_null: Whether the value is a measured null rather than a value.
        """
        if value is None or value == "":
            return
        self.rows.append(
            FactRow(
                label=label,
                value=str(value),
                field_path=path,
                salient=path in self.required,
                measured_null=measured_null,
                note=note,
            )
        )

    def take(self) -> tuple[FactRow, ...]:
        """Return the accumulated rows and reset.

        Returns:
            The rows in insertion order.
        """
        rows = tuple(self.rows)
        self.rows = []
        return rows


def _subject_rows(facts: CaseFacts, builder: _Builder) -> tuple[FactRow, ...]:
    """Build the subject-and-scope rows.

    Args:
        facts: The record.
        builder: The row builder.

    Returns:
        The rows.
    """
    focal = facts.focal_entity
    builder.add("Focal account", focal.id, "focal_entity.id")
    builder.add(
        "Role in this case",
        _role_phrase(focal.role),
        "focal_entity.role",
        note=f"vocabulary term: {focal.role}",
    )
    builder.add(
        "Counterparties sending in", format_count(focal.in_degree), "focal_entity.in_degree"
    )
    builder.add(
        "Counterparties receiving out", format_count(focal.out_degree), "focal_entity.out_degree"
    )
    builder.add(
        "Inbound transactions",
        format_count(focal.n_transactions_in),
        "focal_entity.n_transactions_in",
    )
    builder.add(
        "Outbound transactions",
        format_count(focal.n_transactions_out),
        "focal_entity.n_transactions_out",
    )
    builder.add("First activity", _timestamp(focal.first_seen), "focal_entity.first_seen")
    builder.add("Last activity", _timestamp(focal.last_seen), "focal_entity.last_seen")
    return builder.take()


def _role_phrase(role: str) -> str:
    """Return the surface form for an entity role.

    Read from ``ROLE_DISPLAY``, which is the map
    :mod:`g2t_aml.corpus.claims` **inverts** when it parses a role back out of a
    narrative — so this is the only spelling that aligns. The controlled vocabulary
    carries its own variants for the same roles and they are not the same strings
    ("conduit account" against "a conduit account"): showing an annotator the
    vocabulary's spelling would produce a correctly-written role that the parser could
    not read back, on every case whose salience list requires it.
    ``tests/unit/test_human_factpanel.py`` asserts the round trip rather than trusting
    this comment.

    Args:
        role: The role from the fact record.

    Returns:
        The phrase an annotator should write, or the raw role when it has no display
        form.
    """
    forms = ROLE_DISPLAY.get(role, ())
    return forms[0] if forms else role.replace("_", " ")


def _structure_rows(facts: CaseFacts, builder: _Builder) -> tuple[FactRow, ...]:
    """Build the topology rows.

    Args:
        facts: The record.
        builder: The row builder.

    Returns:
        The rows.
    """
    structure = facts.structure
    builder.add("Accounts in scope", format_count(structure.n_nodes), "structure.n_nodes")
    builder.add("Transactions", format_count(structure.n_edges), "structure.n_edges")
    builder.add(
        "Connected components", format_count(structure.n_components), "structure.n_components"
    )
    builder.add("Density", format_density(structure.density), "structure.density")
    builder.add("Reciprocity", format_density(structure.reciprocity), "structure.reciprocity")
    builder.add(
        "Longest shortest path",
        structure.diameter,
        "structure.diameter",
        measured_null=structure.diameter is None,
    )
    builder.add(
        "Largest in-degree", format_count(structure.max_in_degree), "structure.max_in_degree"
    )
    builder.add(
        "Largest out-degree", format_count(structure.max_out_degree), "structure.max_out_degree"
    )
    if structure.n_self_loops:
        builder.add(
            "Self-transactions",
            structure.n_self_loops,
            "structure.n_self_loops",
            note="counted in transactions, excluded from degrees and motifs",
        )
    return builder.take()


def _temporal_rows(temporal: TemporalFacts, builder: _Builder) -> tuple[FactRow, ...]:
    """Build the timing rows.

    Args:
        temporal: The available temporal block.
        builder: The row builder.

    Returns:
        The rows.
    """
    builder.add("Window start", _timestamp(temporal.window_start), "temporal.window_start")
    builder.add("Window end", _timestamp(temporal.window_end), "temporal.window_end")
    builder.add(
        "Observed span",
        format_duration(temporal.span_hours),
        "temporal.span_hours",
        note="the observed extent, not the extraction window",
    )
    builder.add(
        "Transactions timed", format_count(temporal.n_transactions), "temporal.n_transactions"
    )
    if temporal.burst_detected:
        builder.add(
            "Burst span",
            format_duration(temporal.burst_window_hours)
            if temporal.burst_window_hours is not None
            else None,
            "temporal.burst_window_hours",
        )
        builder.add(
            "Transactions in burst",
            format_count(temporal.burst_txn_count)
            if temporal.burst_txn_count is not None
            else None,
            "temporal.burst_txn_count",
        )
        builder.add("Burst start", _timestamp(temporal.burst_start), "temporal.burst_start")
    else:
        builder.add(
            "Burst",
            "none detected",
            "temporal.burst_detected",
            measured_null=True,
            note="a measured absence, not a masked field",
        )
    if temporal.event_ordering:
        builder.add(
            "Event ordering",
            " then ".join(p.replace("_", " ") for p in temporal.event_ordering),
            "temporal.event_ordering",
        )
    return builder.take()


def _threshold_phrase(flow: FlowFacts, vocabulary: ControlledVocabulary | None) -> str:
    """Return the whitelisted way to name the reporting threshold, if there is one.

    The panel must show the citation an annotator is *allowed* to write, not a
    reconstruction of it. The controlled vocabulary whitelists exact phrase variants
    ("the USD 10,000 reporting threshold"), and both the checker and the claim extractor
    match against those strings; a panel that displayed "the 10,000 US Dollar threshold"
    would teach a phrasing that resolves to no whitelisted reference, so a correct and
    permitted citation would be scored as an unbacked quantity. This was found on a real
    hand-written narrative during Phase 6.

    Args:
        flow: The flow block, for the threshold and its currency.
        vocabulary: The controlled vocabulary, or None to fall back to a plain rendering.

    Returns:
        The whitelisted phrase when one matches this case's threshold and currency,
        otherwise a plain description that makes no citation.
    """
    if vocabulary is not None:
        for reference in vocabulary.regulatory.values():
            if (
                reference.currency == flow.threshold_currency
                and abs(reference.threshold - flow.threshold_reference) < _THRESHOLD_TOLERANCE
                and reference.phrase_variants
            ):
                return reference.phrase_variants[0]
    return f"the {flow.threshold_reference:,.0f} {flow.threshold_currency} reporting threshold"


def _flow_rows(
    flow: FlowFacts, builder: _Builder, vocabulary: ControlledVocabulary | None
) -> tuple[FactRow, ...]:
    """Build the value rows.

    Individual aggregates carry their own sentinels even when the block is available: a
    multi-currency case has no defined total, and D-033 withholds the sum rather than
    inventing one. Such a row is dropped and the per-currency breakdown carries the
    information instead.

    Args:
        flow: The available flow block.
        builder: The row builder.
        vocabulary: The controlled vocabulary, for the whitelisted threshold phrase.

    Returns:
        The rows.
    """
    builder.add("Total received by focal", _money(flow.total_inflow), "flow.total_inflow")
    builder.add("Total sent by focal", _money(flow.total_outflow), "flow.total_outflow")
    builder.add("Retained", _money(flow.retained), "flow.retained")
    builder.add(
        "Largest single transfer", _money(flow.max_single_transfer), "flow.max_single_transfer"
    )
    for total in flow.inflow_by_currency:
        builder.add(
            f"Received in {total.currency}",
            f"{format_money(total.value, total.currency)} over "
            f"{format_count(total.n_transfers)} transfers",
            "flow.inflow_by_currency",
        )
    for total in flow.outflow_by_currency:
        builder.add(
            f"Sent in {total.currency}",
            f"{format_money(total.value, total.currency)} over "
            f"{format_count(total.n_transfers)} transfers",
            "flow.outflow_by_currency",
        )
    if len(flow.currencies_involved) > 1:
        builder.add(
            "Currencies involved",
            ", ".join(flow.currencies_involved),
            "flow.currencies_involved",
            note="no exchange rates exist; a cross-currency total is undefined and is " "not shown",
        )
    builder.add(
        f"Transfers near {_threshold_phrase(flow, vocabulary)}",
        format_count(flow.n_transfers_near_threshold),
        "flow.n_transfers_near_threshold",
        note="cite the threshold in exactly these words if you mention it",
    )
    if is_available(flow.n_distinct_banks):
        builder.add(
            "Institutions involved",
            format_count(flow.n_distinct_banks),
            "flow.n_distinct_banks",
        )
    if flow.payment_formats:
        builder.add("Payment formats", ", ".join(flow.payment_formats), "flow.payment_formats")
    return builder.take()


def _label_rows(labels: LabelFacts, builder: _Builder) -> tuple[FactRow, ...]:
    """Build the counterparty-label rows.

    Args:
        labels: The available labels block.
        builder: The row builder.

    Returns:
        The rows.
    """
    builder.add(
        "Counterparties on a flagged transaction",
        format_count(labels.n_illicit_counterparties),
        "labels.n_illicit_counterparties",
    )
    builder.add(
        "Counterparties with no flag",
        format_count(labels.n_licit_counterparties),
        "labels.n_licit_counterparties",
        note="a licit majority weakens the suspicion and omitting it is H9",
    )
    builder.add(
        "Counterparties unlabelled",
        format_count(labels.n_unknown_counterparties),
        "labels.n_unknown_counterparties",
    )
    builder.add("Counterparties in total", labels.n_counterparties, "labels.n_counterparties")
    builder.add(
        "Flagged transactions", labels.n_illicit_transactions, "labels.n_illicit_transactions"
    )
    builder.add(
        "Hops to the nearest flagged account",
        format_count(labels.min_hops_to_known_illicit)
        if labels.min_hops_to_known_illicit is not None
        else None,
        "labels.min_hops_to_known_illicit",
        measured_null=labels.min_hops_to_known_illicit is None,
    )
    if labels.min_hops_to_known_illicit is None:
        builder.add(
            "Nearest flagged account",
            "none reachable in this case",
            "labels.min_hops_to_known_illicit",
            measured_null=True,
        )
    if is_available(labels.illicit_inflow_share):
        builder.add(
            "Share of inbound value from flagged counterparties",
            format_percent(labels.illicit_inflow_share),
            "labels.illicit_inflow_share",
        )
    builder.add(
        "Focal account itself flagged",
        "yes" if labels.focal_is_illicit else "no",
        "labels.focal_is_illicit",
    )
    return builder.take()


def _motif_rows(facts: CaseFacts, builder: _Builder) -> tuple[FactRow, ...]:
    """Build the structural-pattern rows.

    Every detector is listed, present or not. A motif that did **not** fire is shown as a
    measured absence rather than omitted: "no cycle was found" is evidence an annotator
    needs in order not to write one, and it is the row that most often prevents an H5.

    Args:
        facts: The record.
        builder: The row builder.

    Returns:
        The rows.
    """
    for name in MOTIF_NAMES:
        motif = getattr(facts.motifs, name)
        label = _MOTIF_LABELS[name]
        if not motif.present:
            builder.add(
                label,
                "not detected",
                f"motifs.{name}.present",
                measured_null=True,
            )
            continue
        described = [
            f"{key.replace('_', ' ')} {value}"
            for key, value in sorted(motif.descriptors.items())
            if value is not None
        ]
        builder.add(
            label,
            "detected" + (f" — {', '.join(described)}" if described else ""),
            f"motifs.{name}.present",
        )
        for key, value in sorted(motif.descriptors.items()):
            path = f"motifs.{name}.{key}"
            if value is not None and path in builder.required:
                builder.add(f"{label}: {key.replace('_', ' ')}", value, path)
    return builder.take()


def build_fact_panel(facts: CaseFacts, vocabulary: ControlledVocabulary | None = None) -> FactPanel:
    """Render a fact record into the panel an annotator reads.

    Args:
        facts: The record.
        vocabulary: The controlled vocabulary, for the salience list and for the
            whitelisted spelling of the reporting threshold. **Loaded from disk when
            omitted rather than left as None**: without it the threshold row falls back to
            a phrasing that is not whitelisted, and the panel would teach an annotator a
            citation that scores as H6.

    Returns:
        The panel. Sections holding no rows are dropped, so a substrate that masks a whole
        family produces a panel with no heading for it at all.
    """
    vocabulary = vocabulary if vocabulary is not None else load_vocabulary()
    report = salience_report(facts, vocabulary)
    required = frozenset(report.required)
    builder = _Builder(required)

    masked = tuple(
        MASKED_FAMILY_LABELS[name]
        for name in ("temporal", "flow", "labels")
        if isinstance(getattr(facts, name), Unavailable)
    )

    sections: list[FactSection] = [
        FactSection(
            "Subject",
            _subject_rows(facts, builder),
            "Who the report is about, and what they did inside this window.",
        ),
        FactSection(
            "Scope",
            _structure_rows(facts, builder),
            "The size and shape of what you are looking at.",
        ),
    ]

    temporal = facts.temporal
    if is_available(temporal):
        sections.append(
            FactSection(
                "Timing",
                _temporal_rows(temporal, builder),
                "When it happened. Durations are the observed extent of the case.",
            )
        )
    flow = facts.flow
    if is_available(flow):
        sections.append(
            FactSection(
                "Value",
                _flow_rows(flow, builder, vocabulary),
                "How much moved. Never sum across currencies.",
            )
        )
    labels = facts.labels
    if is_available(labels):
        sections.append(
            FactSection(
                "Counterparty labels",
                _label_rows(labels, builder),
                "Ground truth about counterparties. The licit counts matter as much as "
                "the flagged ones.",
            )
        )
    sections.append(
        FactSection(
            "Structural patterns",
            _motif_rows(facts, builder),
            "What the detectors found, and what they did not. Presence is not proof.",
        )
    )

    return FactPanel(
        case_id=facts.case_id,
        dataset=facts.dataset,
        typology=facts.typology.label,
        typology_source=facts.typology.source,
        typology_scope=facts.typology.scope,
        sections=tuple(s for s in sections if s),
        masked_families=masked,
        required_fields=report.required,
        excused_fields=report.excused,
    )

"""Fact record to flat text: the input to the text-serialisation baseline (system B7).

**B7 is the baseline that threatens our contribution, and it is built here in good faith.**
The claim of this project is that injecting graph structure through a learned fusion layer
beats flattening the same facts into a prompt. If B7 is weak, that comparison proves
nothing — and a reviewer who reimplements B7 competently will find the gap closes, which is
worse than never having claimed it. Deliberately weakening a baseline is research
misconduct, and it is also the easiest kind to detect.

So the discipline in this module is: **every fact in the record reaches the text.** No
family is dropped, no aggregate is silently rounded away, motif descriptors are named with
their quantities, and the availability mask is rendered explicitly rather than by omission
— because a baseline told "this substrate has no amounts" is strictly better informed than
one left to infer it from silence, and the fair comparison is against the better one.

Two styles:

- ``verbose`` — natural sentences. What a language model reads best, and what B7 actually
  consumes.
- ``compact`` — pipe-delimited ``key=value``. Denser per token, which matters when the
  baseline's context budget is the binding constraint, and easier to diff in tests.

**The compact delimiter is the *spaced* pipe ``" | "``, not a bare ``"|"``.** AMLworld
account identifiers are ``"<bank>|<account>"`` (D-011), so a bare pipe appears inside
almost every value in the record and a bare-pipe delimiter would make the format
ambiguous — ``001|8000ABCD`` would split into two fields, and any consumer parsing it back
would silently recover the wrong account. Identifiers contain no whitespace, so the spaced
pipe is unambiguous. :func:`_compact` asserts no rendered value contains the delimiter,
so if a future substrate introduces one the format fails loudly rather than corrupting a
baseline's input.

The serialiser writes *facts*, never conclusions. It does not say "this is layering"; it
says the stack depth is 3. Interpretation is the generator's job, and a serialiser that
editorialised would hand B7 conclusions our own system has to earn.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from g2t_aml.facts.schema import (
    CaseFacts,
    FlowFacts,
    LabelFacts,
    MotifResult,
    TemporalFacts,
    Unavailable,
    is_available,
)

__all__ = ["COMPACT_DELIMITER", "SerialisationStyle", "serialise_facts"]

SerialisationStyle = Literal["verbose", "compact"]

#: Field separator for the compact style. The SPACED pipe, deliberately: AMLworld account
#: identifiers embed a bare ``|`` (D-011) and identifiers contain no whitespace, so this
#: is the shortest unambiguous choice. See the module docstring.
COMPACT_DELIMITER = " | "

#: Human-readable names for the motif keys, used by the verbose style.
_MOTIF_LABELS: dict[str, str] = {
    "fan_in": "fan-in",
    "fan_out": "fan-out",
    "chain": "chain",
    "cycle": "cycle",
    "bipartite": "bipartite structure",
    "stack": "stack",
    "gather_scatter": "gather-scatter",
    "scatter_gather": "scatter-gather",
}


def _money(value: object) -> str:
    """Render a monetary field, sentinel included.

    Args:
        value: A :class:`~g2t_aml.facts.schema.Money` or an
            :class:`~g2t_aml.facts.schema.Unavailable`.

    Returns:
        ``"482300.00 US Dollar"``, or ``"unavailable (reason)"``.
    """
    if isinstance(value, Unavailable):
        return f"unavailable ({value.reason})"
    return f"{value.value:,.2f} {value.currency}"  # type: ignore[attr-defined]


def _scalar(value: object) -> str:
    """Render a scalar that may be a sentinel or a measured null.

    Args:
        value: Any scalar from the record.

    Returns:
        Its string form, ``"unavailable (reason)"`` for a sentinel, or ``"none"`` for a
        measured null — the two are rendered differently on purpose, because they mean
        different things and a baseline that could not tell them apart would be at an
        artificial disadvantage.
    """
    if isinstance(value, Unavailable):
        return f"unavailable ({value.reason})"
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _timestamp(value: datetime | Unavailable) -> str:
    """Render a timestamp field, sentinel included.

    Args:
        value: A datetime or an :class:`~g2t_aml.facts.schema.Unavailable`.

    Returns:
        The ISO-8601 string, or ``"unavailable (reason)"``.
    """
    if isinstance(value, Unavailable):
        return f"unavailable ({value.reason})"
    return value.isoformat()


def _motif_phrase(name: str, motif: MotifResult) -> str:
    """Render one motif with all its descriptors.

    Args:
        name: The motif key.
        motif: The detector result.

    Returns:
        A sentence fragment naming the motif, whether it is present, and every
        descriptor it carries.
    """
    label = _MOTIF_LABELS[name]
    detail = ", ".join(f"{k}={_scalar(v)}" for k, v in motif.descriptors.items())
    state = "detected" if motif.present else "not detected"
    return f"{label} {state}" + (f" ({detail})" if detail else "")


def _temporal_verbose(temporal: TemporalFacts | Unavailable) -> list[str]:
    """Render the temporal block as sentences.

    Args:
        temporal: The block or its sentinel.

    Returns:
        One or more sentences.
    """
    if isinstance(temporal, Unavailable):
        return [f"Timing: unavailable ({temporal.reason})."]
    lines = [
        f"Activity ran from {temporal.window_start.isoformat()} to "
        f"{temporal.window_end.isoformat()}, a span of {temporal.span_hours:g} hours "
        f"across {temporal.n_transactions} transactions."
    ]
    if temporal.burst_detected:
        lines.append(
            f"The densest cluster held {temporal.burst_txn_count} transactions within "
            f"{temporal.burst_window_hours:g} hours, beginning "
            f"{temporal.burst_start.isoformat() if temporal.burst_start else 'unknown'}."
        )
    else:
        lines.append("No qualifying burst of activity was detected.")
    ordering = " then ".join(temporal.event_ordering) if temporal.event_ordering else "none"
    lines.append(f"Phase ordering around the focal account: {ordering}.")
    return lines


def _flow_verbose(flow: FlowFacts | Unavailable) -> list[str]:
    """Render the flow block as sentences.

    Args:
        flow: The block or its sentinel.

    Returns:
        One or more sentences, including the per-currency breakdown whenever an
        aggregate is withheld.
    """
    if isinstance(flow, Unavailable):
        return [f"Monetary flow: unavailable ({flow.reason})."]
    lines = [
        f"Inflow to the focal account totalled {_money(flow.total_inflow)}; outflow "
        f"totalled {_money(flow.total_outflow)}; retained {_money(flow.retained)}.",
        f"The largest single transfer was {_money(flow.max_single_transfer)}.",
    ]
    if flow.inflow_by_currency:
        parts = ", ".join(
            f"{t.value:,.2f} {t.currency} over {t.n_transfers} transfers"
            for t in flow.inflow_by_currency
        )
        lines.append(f"Inflow by currency: {parts}.")
    if flow.outflow_by_currency:
        parts = ", ".join(
            f"{t.value:,.2f} {t.currency} over {t.n_transfers} transfers"
            for t in flow.outflow_by_currency
        )
        lines.append(f"Outflow by currency: {parts}.")
    lines.append(
        f"{flow.n_transfers_near_threshold} transfers fell within "
        f"{flow.threshold_band_fraction:.0%} below the {flow.threshold_reference:,.2f} "
        f"{flow.threshold_currency} reporting threshold."
    )
    lines.append(
        f"Currencies involved: {', '.join(flow.currencies_involved) or 'none'}. "
        f"Payment formats: {', '.join(flow.payment_formats) or 'none'}. "
        f"Distinct banks: {_scalar(flow.n_distinct_banks)}. "
        f"Spans more than one institution: {_scalar(flow.cross_institution)}. "
        f"Cross-border: {_scalar(flow.cross_border)}."
    )
    return lines


def _labels_verbose(labels: LabelFacts | Unavailable) -> list[str]:
    """Render the labels block as sentences.

    Args:
        labels: The block or its sentinel.

    Returns:
        One or more sentences.
    """
    if isinstance(labels, Unavailable):
        return [f"Counterparty labels: unavailable ({labels.reason})."]
    return [
        f"Of {labels.n_counterparties} counterparties, "
        f"{labels.n_illicit_counterparties} are associated with flagged transactions, "
        f"{labels.n_licit_counterparties} are not, and "
        f"{labels.n_unknown_counterparties} are unlabelled.",
        f"The case contains {labels.n_illicit_transactions} flagged transactions. "
        f"The focal account is itself on a flagged transaction: "
        f"{_scalar(labels.focal_is_illicit)}. "
        f"Hops to the nearest flagged account: "
        f"{_scalar(labels.min_hops_to_known_illicit)}. "
        f"Share of inbound value from flagged counterparties: "
        f"{_scalar(labels.illicit_inflow_share)}.",
    ]


def _verbose(facts: CaseFacts) -> str:
    """Render the whole record as natural sentences.

    Args:
        facts: The record to render.

    Returns:
        The serialised text.
    """
    structure = facts.structure
    focal = facts.focal_entity
    first_seen = _timestamp(focal.first_seen)
    last_seen = _timestamp(focal.last_seen)
    lines: list[str] = [
        f"Case {facts.case_id} from substrate {facts.dataset}.",
        f"The subgraph holds {structure.n_nodes} accounts and {structure.n_edges} "
        f"transactions in {structure.n_components} component(s), with density "
        f"{structure.density:g}, diameter {_scalar(structure.diameter)}, reciprocity "
        f"{structure.reciprocity:g}, maximum in-degree {structure.max_in_degree}, "
        f"maximum out-degree {structure.max_out_degree}, and "
        f"{structure.n_self_loops} self-transfers.",
        f"The focal account is {focal.id} (selected by {focal.selection_rule}), acting "
        f"as {focal.role}, with {focal.in_degree} distinct senders and "
        f"{focal.out_degree} distinct recipients across "
        f"{focal.n_transactions_in} inbound and {focal.n_transactions_out} outbound "
        f"transactions. First seen {first_seen}, last seen {last_seen}.",
    ]
    lines += _temporal_verbose(facts.temporal)
    lines += _flow_verbose(facts.flow)
    lines += _labels_verbose(facts.labels)

    motif_parts = [_motif_phrase(name, m) for name, m in facts.motifs.as_mapping().items()]
    lines.append("Structural motifs: " + "; ".join(motif_parts) + ".")

    typology = facts.typology
    lines.append(
        f"Typology: {typology.label} (source {typology.source}, confidence "
        f"{typology.confidence:g}, scope {typology.scope})."
    )
    if typology.scope == "stream_membership":
        lines.append(
            "The case is part of a stream of this typology and may not contain it in " "full."
        )

    signal = facts.model_signal
    if signal.gnn_risk_score is not None:
        contributors = ", ".join(f"{n} ({a:g})" for n, a in signal.top_contributing_nodes)
        lines.append(
            f"Model risk score {signal.gnn_risk_score:g} (percentile "
            f"{_scalar(signal.score_percentile)}, model {_scalar(signal.model_version)})."
            + (f" Top contributing accounts: {contributors}." if contributors else "")
        )

    unavailable = sorted(k for k, v in facts.availability.to_dict().items() if not v)
    if unavailable:
        lines.append(
            "This substrate does not support claims about: " + ", ".join(unavailable) + "."
        )
    return "\n".join(lines)


def _compact(facts: CaseFacts) -> str:
    """Render the whole record as pipe-delimited key-value pairs.

    Args:
        facts: The record to render.

    Returns:
        The serialised text, one ``key=value`` per field, ``|``-separated.
    """
    pairs: list[tuple[str, str]] = [
        ("case_id", facts.case_id),
        ("dataset", facts.dataset),
        ("n_nodes", str(facts.structure.n_nodes)),
        ("n_edges", str(facts.structure.n_edges)),
        ("n_components", str(facts.structure.n_components)),
        ("density", f"{facts.structure.density:g}"),
        ("diameter", _scalar(facts.structure.diameter)),
        ("max_in_degree", str(facts.structure.max_in_degree)),
        ("max_out_degree", str(facts.structure.max_out_degree)),
        ("reciprocity", f"{facts.structure.reciprocity:g}"),
        ("n_self_loops", str(facts.structure.n_self_loops)),
        ("focal", facts.focal_entity.id),
        ("focal_role", facts.focal_entity.role),
        ("focal_in_degree", str(facts.focal_entity.in_degree)),
        ("focal_out_degree", str(facts.focal_entity.out_degree)),
        ("focal_txn_in", str(facts.focal_entity.n_transactions_in)),
        ("focal_txn_out", str(facts.focal_entity.n_transactions_out)),
    ]

    temporal = facts.temporal
    if is_available(temporal):
        pairs += [
            ("window_start", temporal.window_start.isoformat()),
            ("window_end", temporal.window_end.isoformat()),
            ("span_hours", f"{temporal.span_hours:g}"),
            ("n_transactions", str(temporal.n_transactions)),
            ("burst", "yes" if temporal.burst_detected else "no"),
            ("burst_window_hours", _scalar(temporal.burst_window_hours)),
            ("burst_txn_count", _scalar(temporal.burst_txn_count)),
            ("event_ordering", ">".join(temporal.event_ordering) or "none"),
        ]
    elif isinstance(temporal, Unavailable):
        pairs.append(("temporal", f"unavailable:{temporal.reason}"))

    flow = facts.flow
    if is_available(flow):
        pairs += [
            ("total_inflow", _money(flow.total_inflow)),
            ("total_outflow", _money(flow.total_outflow)),
            ("retained", _money(flow.retained)),
            ("max_single_transfer", _money(flow.max_single_transfer)),
            (
                "inflow_by_currency",
                ";".join(
                    f"{t.currency}:{t.value:.2f}x{t.n_transfers}" for t in flow.inflow_by_currency
                )
                or "none",
            ),
            (
                "outflow_by_currency",
                ";".join(
                    f"{t.currency}:{t.value:.2f}x{t.n_transfers}" for t in flow.outflow_by_currency
                )
                or "none",
            ),
            ("n_near_threshold", str(flow.n_transfers_near_threshold)),
            ("threshold", f"{flow.threshold_reference:g} {flow.threshold_currency}"),
            ("currencies", ";".join(flow.currencies_involved) or "none"),
            ("payment_formats", ";".join(flow.payment_formats) or "none"),
            ("n_distinct_banks", _scalar(flow.n_distinct_banks)),
            ("cross_institution", _scalar(flow.cross_institution)),
            ("cross_border", _scalar(flow.cross_border)),
        ]
    elif isinstance(flow, Unavailable):
        pairs.append(("flow", f"unavailable:{flow.reason}"))

    labels = facts.labels
    if is_available(labels):
        pairs += [
            ("n_illicit_counterparties", str(labels.n_illicit_counterparties)),
            ("n_licit_counterparties", str(labels.n_licit_counterparties)),
            ("n_unknown_counterparties", str(labels.n_unknown_counterparties)),
            ("n_counterparties", str(labels.n_counterparties)),
            ("min_hops_to_illicit", _scalar(labels.min_hops_to_known_illicit)),
            ("illicit_inflow_share", _scalar(labels.illicit_inflow_share)),
            ("n_illicit_transactions", str(labels.n_illicit_transactions)),
            ("focal_is_illicit", _scalar(labels.focal_is_illicit)),
        ]
    elif isinstance(labels, Unavailable):
        pairs.append(("labels", f"unavailable:{labels.reason}"))

    for name, motif in facts.motifs.as_mapping().items():
        detail = ",".join(f"{k}={_scalar(v)}" for k, v in motif.descriptors.items())
        pairs.append((name, ("yes" if motif.present else "no") + (f"[{detail}]" if detail else "")))

    pairs += [
        ("typology", facts.typology.label),
        ("typology_source", facts.typology.source),
        ("typology_confidence", f"{facts.typology.confidence:g}"),
        ("typology_scope", facts.typology.scope),
        ("gnn_risk_score", _scalar(facts.model_signal.gnn_risk_score)),
        (
            "unavailable_fact_classes",
            ";".join(sorted(k for k, v in facts.availability.to_dict().items() if not v)) or "none",
        ),
    ]
    if ambiguous := [k for k, v in pairs if COMPACT_DELIMITER in v]:
        raise ValueError(
            f"compact serialisation of case {facts.case_id!r} would be ambiguous: fields "
            f"{ambiguous} contain the delimiter {COMPACT_DELIMITER!r}. Change the "
            "delimiter rather than emitting a record a consumer would parse wrongly."
        )
    return COMPACT_DELIMITER.join(f"{k}={v}" for k, v in pairs)


def serialise_facts(facts: CaseFacts, style: SerialisationStyle = "verbose") -> str:
    """Render a fact record as flat text for the serialisation baseline.

    Args:
        facts: The record to render.
        style: ``"verbose"`` for natural sentences, ``"compact"`` for pipe-delimited
            key-value pairs.

    Returns:
        The serialised text. Every fact family in the record reaches the output in both
        styles, availability sentinels included — see the module docstring for why that
        completeness is a research-integrity requirement rather than a nicety.

    Raises:
        ValueError: If ``style`` is not one of the two.
    """
    if style == "verbose":
        return _verbose(facts)
    if style == "compact":
        return _compact(facts)
    raise ValueError(f"unknown serialisation style {style!r}; expected 'verbose' or 'compact'")

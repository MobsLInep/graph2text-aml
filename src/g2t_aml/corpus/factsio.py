"""Read a serialised ``case_facts`` record back into the typed object.

Phase 3 wrote 30,000 validated JSON records and no way to read them. Re-extracting them
to render Bronze would cost 31 minutes per build and, worse, would make the corpus depend
on the *cases* being on disk rather than on the fact records that were gated — so a later
change to case extraction could silently move the corpus out from under the gate that
passed.

**This module is the exact inverse of :func:`g2t_aml.facts.schema.facts_to_dict`, and it
proves it on every single record it loads.** :func:`load_case_facts` reconstructs the
dataclass tree, re-serialises it, and refuses to return unless the result is byte-for-byte
the mapping it was given. That check is not defensive politeness. Bronze's whole claim is
that it renders from a fact record and is then verified against *that same* record: a
deserialiser that quietly dropped a currency or coerced a measured ``None`` into a sentinel
would produce a narrative that is faithful to a record nobody ever extracted, and the
checker — reading the same corrupted object — would report 100% SUPPORTED. The identity
assertion is what stops that, and it is cheap next to rendering.

**Why this lives in ``corpus/`` and not in ``facts/``.** ``facts/`` is the measurement
instrument (invariant 1) and is frozen at schema 1.0.0. Reading is a Phase 4 need, so the
code and its risk live in Phase 4's package; ``facts/`` is imported, never touched.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from g2t_aml.data.canonical import AvailabilityMask
from g2t_aml.facts.schema import (
    CASE_FACTS_SCHEMA_VERSION,
    MOTIF_NAMES,
    CaseFacts,
    EntityInventory,
    FlowFacts,
    FocalEntity,
    LabelFacts,
    ModelSignal,
    Money,
    MotifFacts,
    MotifResult,
    PerCurrencyTotal,
    Provenance,
    StructureFacts,
    TemporalFacts,
    Typology,
    Unavailable,
    facts_to_dict,
)
from g2t_aml.utils.io import read_json

__all__ = ["FactsIOError", "load_case_facts", "load_case_facts_file", "facts_from_dict"]


class FactsIOError(ValueError):
    """Raised when a serialised record cannot be reconstructed faithfully."""


def _sentinel(payload: Any) -> Unavailable | None:
    """Return the sentinel a mapping encodes, or None when it encodes a value.

    Args:
        payload: Any JSON value from a fact record.

    Returns:
        The reconstructed :class:`~g2t_aml.facts.schema.Unavailable`, or None when
        ``payload`` is not a serialised sentinel.
    """
    if isinstance(payload, dict) and payload.get("available") is False:
        return Unavailable(str(payload["reason"]))
    return None


def _money(payload: Any) -> Money | Unavailable:
    """Rebuild a monetary field.

    Args:
        payload: ``{"value": ..., "currency": ...}`` or a serialised sentinel.

    Returns:
        The :class:`~g2t_aml.facts.schema.Money` or the sentinel.

    Raises:
        FactsIOError: If the mapping is neither.
    """
    if (sentinel := _sentinel(payload)) is not None:
        return sentinel
    if isinstance(payload, dict) and {"value", "currency"} <= set(payload):
        return Money(value=float(payload["value"]), currency=str(payload["currency"]))
    raise FactsIOError(f"expected a monetary amount or a sentinel, got {payload!r}")


def _timestamp(payload: Any) -> datetime | Unavailable:
    """Rebuild a timestamp field.

    Args:
        payload: An ISO-8601 string or a serialised sentinel.

    Returns:
        The datetime or the sentinel.

    Raises:
        FactsIOError: If the value is neither.
    """
    if (sentinel := _sentinel(payload)) is not None:
        return sentinel
    if isinstance(payload, str):
        return datetime.fromisoformat(payload)
    raise FactsIOError(f"expected an ISO-8601 timestamp or a sentinel, got {payload!r}")


def _scalar(payload: Any) -> Any:
    """Rebuild a scalar that may be a sentinel.

    Args:
        payload: A scalar, or a serialised sentinel.

    Returns:
        The scalar unchanged, or the sentinel. A bare ``None`` is returned as ``None``:
        it is a *measured* null and means something different from a sentinel (D-025).
    """
    sentinel = _sentinel(payload)
    return sentinel if sentinel is not None else payload


def _temporal(payload: dict[str, Any]) -> TemporalFacts | Unavailable:
    """Rebuild the temporal group.

    Args:
        payload: The serialised group or a serialised sentinel.

    Returns:
        The group or the sentinel.
    """
    if (sentinel := _sentinel(payload)) is not None:
        return sentinel
    burst_start = payload["burst_start"]
    return TemporalFacts(
        window_start=datetime.fromisoformat(str(payload["window_start"])),
        window_end=datetime.fromisoformat(str(payload["window_end"])),
        span_hours=float(payload["span_hours"]),
        burst_detected=bool(payload["burst_detected"]),
        burst_window_hours=payload["burst_window_hours"],
        burst_txn_count=payload["burst_txn_count"],
        burst_start=datetime.fromisoformat(str(burst_start)) if burst_start else None,
        event_ordering=tuple(str(p) for p in payload["event_ordering"]),
        n_transactions=int(payload["n_transactions"]),
    )


def _flow(payload: dict[str, Any]) -> FlowFacts | Unavailable:
    """Rebuild the flow group.

    Args:
        payload: The serialised group or a serialised sentinel.

    Returns:
        The group or the sentinel.

    Raises:
        FactsIOError: If ``cross_border`` is not a sentinel. It is permanently
            unavailable on every substrate (D-030), so a record carrying a value there is
            corrupt rather than merely unexpected.
    """
    if (sentinel := _sentinel(payload)) is not None:
        return sentinel
    cross_border = _sentinel(payload["cross_border"])
    if cross_border is None:
        raise FactsIOError(
            "flow.cross_border must be an availability sentinel on every substrate "
            f"(D-030), got {payload['cross_border']!r}"
        )
    return FlowFacts(
        total_inflow=_money(payload["total_inflow"]),
        total_outflow=_money(payload["total_outflow"]),
        retained=_money(payload["retained"]),
        max_single_transfer=_money(payload["max_single_transfer"]),
        inflow_by_currency=tuple(_per_currency(t) for t in payload["inflow_by_currency"]),
        outflow_by_currency=tuple(_per_currency(t) for t in payload["outflow_by_currency"]),
        n_transfers_near_threshold=int(payload["n_transfers_near_threshold"]),
        threshold_reference=float(payload["threshold_reference"]),
        threshold_currency=str(payload["threshold_currency"]),
        threshold_band_fraction=float(payload["threshold_band_fraction"]),
        currencies_involved=tuple(str(c) for c in payload["currencies_involved"]),
        cross_border=cross_border,
        cross_institution=_scalar(payload["cross_institution"]),
        n_distinct_banks=_scalar(payload["n_distinct_banks"]),
        payment_formats=tuple(str(f) for f in payload["payment_formats"]),
    )


def _per_currency(payload: dict[str, Any]) -> PerCurrencyTotal:
    """Rebuild one per-currency total.

    Args:
        payload: The serialised total.

    Returns:
        The reconstructed total.
    """
    return PerCurrencyTotal(
        currency=str(payload["currency"]),
        value=float(payload["value"]),
        n_transfers=int(payload["n_transfers"]),
    )


def _labels(payload: dict[str, Any]) -> LabelFacts | Unavailable:
    """Rebuild the labels group.

    Args:
        payload: The serialised group or a serialised sentinel.

    Returns:
        The group or the sentinel.
    """
    if (sentinel := _sentinel(payload)) is not None:
        return sentinel
    return LabelFacts(
        n_illicit_counterparties=int(payload["n_illicit_counterparties"]),
        n_licit_counterparties=int(payload["n_licit_counterparties"]),
        n_unknown_counterparties=int(payload["n_unknown_counterparties"]),
        n_counterparties=int(payload["n_counterparties"]),
        min_hops_to_known_illicit=payload["min_hops_to_known_illicit"],
        illicit_inflow_share=_scalar(payload["illicit_inflow_share"]),
        n_illicit_transactions=int(payload["n_illicit_transactions"]),
        focal_is_illicit=bool(payload["focal_is_illicit"]),
    )


def _motifs(payload: dict[str, Any]) -> MotifFacts:
    """Rebuild the motifs group.

    Args:
        payload: The serialised group, motif name to ``{"present": ..., **descriptors}``.

    Returns:
        The reconstructed group. The witness is not serialised, so it comes back empty —
        the one field this module cannot restore, and the only one no consumer of a
        written record can have been reading.
    """
    results: dict[str, MotifResult] = {}
    for name in MOTIF_NAMES:
        body = dict(payload[name])
        present = bool(body.pop("present"))
        results[name] = MotifResult(present=present, descriptors=body)
    return MotifFacts(**results)


def facts_from_dict(payload: dict[str, Any]) -> CaseFacts:
    """Rebuild a fact record from its serialised form, without verifying the round trip.

    Prefer :func:`load_case_facts`, which adds the identity assertion. This entrypoint
    exists for the test that *proves* the assertion is meaningful, which needs the
    unguarded reconstruction to compare against.

    Args:
        payload: The output of :func:`~g2t_aml.facts.schema.facts_to_dict`.

    Returns:
        The reconstructed record.

    Raises:
        FactsIOError: If a group cannot be reconstructed.
        KeyError: If a required key is absent.
    """
    provenance_payload = payload["provenance"]
    return CaseFacts(
        case_id=str(payload["case_id"]),
        dataset=str(payload["dataset"]),
        availability=AvailabilityMask.from_dict(payload["availability"]),
        entity_inventory=EntityInventory(
            node_ids=tuple(str(n) for n in payload["entity_inventory"]["node_ids"]),
            focal_id=str(payload["entity_inventory"]["focal_id"]),
        ),
        structure=StructureFacts(**payload["structure"]),
        focal_entity=FocalEntity(
            id=str(payload["focal_entity"]["id"]),
            selection_rule=str(payload["focal_entity"]["selection_rule"]),
            in_degree=int(payload["focal_entity"]["in_degree"]),
            out_degree=int(payload["focal_entity"]["out_degree"]),
            n_transactions_in=int(payload["focal_entity"]["n_transactions_in"]),
            n_transactions_out=int(payload["focal_entity"]["n_transactions_out"]),
            role=str(payload["focal_entity"]["role"]),
            first_seen=_timestamp(payload["focal_entity"]["first_seen"]),
            last_seen=_timestamp(payload["focal_entity"]["last_seen"]),
        ),
        temporal=_temporal(payload["temporal"]),
        flow=_flow(payload["flow"]),
        labels=_labels(payload["labels"]),
        motifs=_motifs(payload["motifs"]),
        typology=Typology(**payload["typology"]),
        model_signal=ModelSignal(
            gnn_risk_score=payload["model_signal"]["gnn_risk_score"],
            score_percentile=payload["model_signal"]["score_percentile"],
            top_contributing_nodes=tuple(
                (str(c["node_id"]), float(c["attribution"]))
                for c in payload["model_signal"]["top_contributing_nodes"]
            ),
            model_version=payload["model_signal"]["model_version"],
        ),
        provenance=Provenance(
            case_extraction=dict(provenance_payload["case_extraction"]),
            field_producers=dict(provenance_payload["field_producers"]),
            computed_at=datetime.fromisoformat(str(provenance_payload["computed_at"])),
            config=dict(provenance_payload["config"]),
        ),
        schema_version=str(payload["schema_version"]),
        extractor_version=str(payload["extractor_version"]),
    )


def load_case_facts(payload: dict[str, Any]) -> CaseFacts:
    """Rebuild a fact record and prove the reconstruction is lossless.

    Args:
        payload: The output of :func:`~g2t_aml.facts.schema.facts_to_dict`, as written by
            ``scripts/03_extract_facts.py``.

    Returns:
        The reconstructed record.

    Raises:
        FactsIOError: If the record declares a schema version other than the frozen one
            (invariant 3), or if re-serialising the reconstruction does not reproduce
            ``payload`` exactly. The second is the important one: it is what stops a
            silently lossy read from producing a corpus that verifies against a record
            that was never extracted.
    """
    declared = str(payload.get("schema_version", ""))
    if declared != CASE_FACTS_SCHEMA_VERSION:
        raise FactsIOError(
            f"record {payload.get('case_id')!r} declares case_facts schema {declared!r} "
            f"but the code is frozen at {CASE_FACTS_SCHEMA_VERSION!r} (invariant 3); "
            "regenerate the fact records rather than reading them under a different "
            "contract"
        )
    facts = facts_from_dict(payload)
    reserialised = facts_to_dict(facts)
    if reserialised != payload:
        differing = sorted(
            key
            for key in set(reserialised) | set(payload)
            if reserialised.get(key) != payload.get(key)
        )
        raise FactsIOError(
            f"deserialising case {payload.get('case_id')!r} was lossy: re-serialising "
            f"the reconstruction disagrees with the file at {differing}. A narrative "
            "rendered from a mis-read record would be verified against that same "
            "mis-read record and would pass, so this is refused rather than warned about."
        )
    return facts


def load_case_facts_file(path: str | Path) -> CaseFacts:
    """Read and reconstruct one fact record from disk.

    Args:
        path: Path to a ``facts/<case_id>.json`` file.

    Returns:
        The reconstructed record.

    Raises:
        FactsIOError: If the reconstruction is not lossless, or the schema version
            disagrees with the frozen one.
        FileNotFoundError: If the file does not exist.
    """
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise FactsIOError(f"{path} does not contain a fact record object")
    return load_case_facts(payload)
